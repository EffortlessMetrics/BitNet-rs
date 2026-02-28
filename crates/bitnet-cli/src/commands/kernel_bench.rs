//! Kernel-level micro-benchmarks for matmul, quantisation, and softmax.

use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Args;
use console::style;
use serde::Serialize;

use bitnet_common::QuantizationType;
use bitnet_kernels::{KernelManager, KernelProvider};

/// Run kernel-level micro-benchmarks.
#[derive(Args, Debug)]
pub struct KernelBenchCommand {
    /// Matrix dimension sizes to benchmark (comma-separated).
    #[arg(long, default_value = "128,256,512,1024", value_delimiter = ',')]
    pub sizes: Vec<usize>,

    /// Number of timed iterations per size.
    #[arg(long, default_value = "10")]
    pub iterations: usize,

    /// Number of warmup iterations per size.
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Output as JSON instead of table.
    #[arg(long)]
    pub json: bool,
}

/// One benchmark result row.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub operation: String,
    pub size: usize,
    pub provider: String,
    pub mean_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub throughput_gflops: Option<f64>,
}

/// Full benchmark report.
#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub provider: String,
    pub results: Vec<BenchResult>,
}

impl KernelBenchCommand {
    pub async fn execute(&self) -> Result<()> {
        let manager = KernelManager::new();
        let kernel = manager.select_best()?;
        let provider_name = kernel.name().to_string();

        let mut results = Vec::new();

        for &sz in &self.sizes {
            results.push(bench_matmul(kernel, sz, self.warmup, self.iterations));
            results.push(bench_quantize(kernel, sz, self.warmup, self.iterations));
            results.push(bench_softmax(sz, self.warmup, self.iterations));
        }

        let report = BenchReport {
            provider: provider_name,
            results,
        };

        if self.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_table(&report);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Individual benchmarks
// ---------------------------------------------------------------------------

fn bench_matmul(
    kernel: &dyn KernelProvider,
    n: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let m = n;
    let k = n;
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
    let b: Vec<u8> = (0..k * n).map(|i| (i % 4) as u8).collect();
    let mut c = vec![0.0f32; m * n];

    for _ in 0..warmup {
        let _ = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        c.fill(0.0);
        let t0 = Instant::now();
        let _ = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
        times.push(t0.elapsed());
    }

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    to_result("matmul_i2s", n, kernel.name(), &times, Some(flops))
}

fn bench_quantize(
    kernel: &dyn KernelProvider,
    n: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult {
    let len = n * n;
    let input: Vec<f32> = (0..len).map(|i| (i as f32) / (len as f32)).collect();
    let mut output = vec![0u8; len / 4];
    let num_blocks = (len + 127) / 128;
    let mut scales = vec![0.0f32; num_blocks];

    for _ in 0..warmup {
        let _ = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        output.fill(0);
        let t0 = Instant::now();
        let _ = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        times.push(t0.elapsed());
    }

    to_result("quantize_i2s", n, kernel.name(), &times, None)
}

/// Pure-Rust softmax benchmark (not kernel-provider based).
fn bench_softmax(n: usize, warmup: usize, iters: usize) -> BenchResult {
    let len = n * n;
    let mut data: Vec<f32> = (0..len).map(|i| (i as f32) / (len as f32)).collect();

    for _ in 0..warmup {
        softmax_inplace(&mut data, n);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        softmax_inplace(&mut data, n);
        times.push(t0.elapsed());
    }

    to_result("softmax", n, "rust", &times, None)
}

/// Row-wise softmax in place.
fn softmax_inplace(data: &mut [f32], row_len: usize) {
    for row in data.chunks_mut(row_len) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_result(
    op: &str,
    size: usize,
    provider: &str,
    times: &[Duration],
    flops: Option<f64>,
) -> BenchResult {
    let us: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    let min = us.iter().copied().fold(f64::INFINITY, f64::min);
    let max = us.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let throughput_gflops = flops.map(|f| f / (mean * 1e-6) / 1e9);

    BenchResult {
        operation: op.to_string(),
        size,
        provider: provider.to_string(),
        mean_us: mean,
        min_us: min,
        max_us: max,
        throughput_gflops,
    }
}

fn print_table(report: &BenchReport) {
    println!(
        "{} {}",
        style("Kernel Provider:").bold(),
        report.provider
    );
    println!();
    println!(
        "  {:<16} {:>6} {:>12} {:>12} {:>12} {:>10}",
        "Operation", "Size", "Mean (µs)", "Min (µs)", "Max (µs)", "GFLOPS"
    );
    println!("  {}", "-".repeat(72));
    for r in &report.results {
        let gf = r
            .throughput_gflops
            .map(|g| format!("{g:.2}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "  {:<16} {:>6} {:>12.1} {:>12.1} {:>12.1} {:>10}",
            r.operation, r.size, r.mean_us, r.min_us, r.max_us, gf
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut data, 4);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row must sum to 1");
    }

    #[test]
    fn test_softmax_inplace_multi_row() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        softmax_inplace(&mut data, 4);
        let s1: f32 = data[..4].iter().sum();
        let s2: f32 = data[4..].iter().sum();
        assert!((s1 - 1.0).abs() < 1e-5);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_to_result_computes_stats() {
        let times = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(150),
        ];
        let r = to_result("op", 64, "test", &times, Some(1e9));
        assert!((r.mean_us - 150.0).abs() < 1.0);
        assert!((r.min_us - 100.0).abs() < 1.0);
        assert!((r.max_us - 200.0).abs() < 1.0);
        assert!(r.throughput_gflops.is_some());
    }

    #[test]
    fn test_to_result_no_flops() {
        let times = vec![Duration::from_micros(50)];
        let r = to_result("q", 32, "test", &times, None);
        assert!(r.throughput_gflops.is_none());
    }

    #[test]
    fn test_bench_report_json_round_trip() {
        let report = BenchReport {
            provider: "fallback".to_string(),
            results: vec![BenchResult {
                operation: "matmul_i2s".to_string(),
                size: 128,
                provider: "fallback".to_string(),
                mean_us: 100.0,
                min_us: 90.0,
                max_us: 110.0,
                throughput_gflops: Some(1.5),
            }],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["provider"], "fallback");
        assert!(parsed["results"].is_array());
    }

    #[test]
    fn test_print_table_no_panic() {
        let report = BenchReport {
            provider: "test".to_string(),
            results: vec![
                BenchResult {
                    operation: "matmul_i2s".to_string(),
                    size: 64,
                    provider: "test".to_string(),
                    mean_us: 50.0,
                    min_us: 40.0,
                    max_us: 60.0,
                    throughput_gflops: Some(2.0),
                },
                BenchResult {
                    operation: "softmax".to_string(),
                    size: 64,
                    provider: "rust".to_string(),
                    mean_us: 30.0,
                    min_us: 25.0,
                    max_us: 35.0,
                    throughput_gflops: None,
                },
            ],
        };
        print_table(&report);
    }

    #[test]
    fn test_bench_softmax_runs() {
        let r = bench_softmax(64, 1, 2);
        assert_eq!(r.operation, "softmax");
        assert_eq!(r.size, 64);
        assert!(r.mean_us > 0.0);
    }

    #[test]
    fn test_bench_matmul_runs() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();
        let r = bench_matmul(kernel, 64, 1, 2);
        assert_eq!(r.operation, "matmul_i2s");
        assert!(r.mean_us > 0.0);
    }

    #[test]
    fn test_bench_quantize_runs() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();
        let r = bench_quantize(kernel, 64, 1, 2);
        assert_eq!(r.operation, "quantize_i2s");
        assert!(r.mean_us > 0.0);
    }
}
