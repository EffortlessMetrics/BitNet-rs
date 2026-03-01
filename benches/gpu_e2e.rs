//! End-to-end GPU inference benchmarks.
//!
//! Measures full pipeline: model load → tokenize → inference → detokenize
//! across OpenCL, Vulkan, and CPU baseline backends.
//!
//! Gated behind `#[cfg(feature = "bench")]` so these only compile when
//! explicitly requested.
#![cfg(feature = "bench")]

use criterion::{
    BenchmarkId, Criterion, criterion_group, criterion_main,
};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Backend discriminant
// ---------------------------------------------------------------------------

/// GPU compute backend used for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchBackend {
    /// CPU baseline (no GPU).
    Cpu,
    /// OpenCL 3.0 backend.
    OpenCl,
    /// Vulkan compute backend.
    Vulkan,
}

impl std::fmt::Display for BenchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchBackend::Cpu => write!(f, "cpu"),
            BenchBackend::OpenCl => write!(f, "opencl"),
            BenchBackend::Vulkan => write!(f, "vulkan"),
        }
    }
}

// ---------------------------------------------------------------------------
// Throughput & latency measurement helpers
// ---------------------------------------------------------------------------

/// Result of a single inference run used for metric computation.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Number of tokens generated in this run.
    pub tokens_generated: usize,
    /// Wall-clock duration of the generation phase.
    pub generation_duration: Duration,
    /// Peak memory usage in bytes observed during the run.
    pub peak_memory_bytes: u64,
}

impl InferenceMetrics {
    /// Token throughput in tokens per second.
    #[inline]
    pub fn tokens_per_sec(&self) -> f64 {
        if self.generation_duration.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.tokens_generated as f64 / self.generation_duration.as_secs_f64()
    }

    /// Latency in milliseconds per token.
    #[inline]
    pub fn ms_per_token(&self) -> f64 {
        if self.tokens_generated == 0 {
            return 0.0;
        }
        self.generation_duration.as_secs_f64() * 1000.0
            / self.tokens_generated as f64
    }
}

// ---------------------------------------------------------------------------
// Memory high-water-mark tracker
// ---------------------------------------------------------------------------

/// Thread-safe memory high-water-mark tracker.
///
/// Records the peak allocation observed across `record` calls.
#[derive(Debug)]
pub struct MemoryHighWaterMark {
    peak_bytes: AtomicUsize,
}

impl MemoryHighWaterMark {
    /// Create a new tracker with zero peak.
    pub fn new() -> Self {
        Self {
            peak_bytes: AtomicUsize::new(0),
        }
    }

    /// Record an allocation size, updating the peak if this is the largest.
    pub fn record(&self, bytes: usize) {
        self.peak_bytes.fetch_max(bytes, Ordering::Relaxed);
    }

    /// Return the peak observed allocation in bytes.
    pub fn peak(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Reset the tracker to zero.
    pub fn reset(&self) {
        self.peak_bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryHighWaterMark {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Simulated pipeline stages (stand-ins until real GPU paths are wired)
// ---------------------------------------------------------------------------

/// Simulate model loading for a given backend.
fn simulate_model_load(backend: BenchBackend) -> Duration {
    let start = Instant::now();
    // Simulate work proportional to backend initialisation cost.
    let iters = match backend {
        BenchBackend::Cpu => 100,
        BenchBackend::OpenCl => 200,
        BenchBackend::Vulkan => 250,
    };
    let mut acc: u64 = 0;
    for i in 0..iters {
        acc = acc.wrapping_add(black_box(i));
    }
    black_box(acc);
    start.elapsed()
}

/// Simulate tokenization of an input prompt. Returns token count.
fn simulate_tokenize(prompt: &str) -> usize {
    // Rough approximation: ~4 chars per token.
    let count = (prompt.len() / 4).max(1);
    black_box(count)
}

/// Simulate inference producing `max_tokens` tokens.
fn simulate_inference(
    backend: BenchBackend,
    _input_tokens: usize,
    max_tokens: usize,
    hwm: &MemoryHighWaterMark,
) -> InferenceMetrics {
    let start = Instant::now();
    let mut acc: u64 = 0;
    let per_token_work = match backend {
        BenchBackend::Cpu => 50,
        BenchBackend::OpenCl => 30,
        BenchBackend::Vulkan => 25,
    };
    for t in 0..max_tokens {
        for i in 0..per_token_work {
            acc = acc.wrapping_add(black_box(i as u64 + t as u64));
        }
        // Simulate growing KV cache memory.
        hwm.record((t + 1) * 2048);
    }
    black_box(acc);
    InferenceMetrics {
        tokens_generated: max_tokens,
        generation_duration: start.elapsed(),
        peak_memory_bytes: hwm.peak() as u64,
    }
}

/// Simulate detokenization.
fn simulate_detokenize(token_count: usize) -> String {
    let out: String = (0..token_count).map(|i| format!("tok{i} ")).collect();
    black_box(out)
}

// ---------------------------------------------------------------------------
// End-to-end pipeline
// ---------------------------------------------------------------------------

/// Run the full pipeline for a backend and return metrics.
fn run_e2e_pipeline(
    backend: BenchBackend,
    prompt: &str,
    max_tokens: usize,
) -> InferenceMetrics {
    let hwm = MemoryHighWaterMark::new();
    let _load_time = simulate_model_load(backend);
    let input_tokens = simulate_tokenize(prompt);
    let metrics =
        simulate_inference(backend, input_tokens, max_tokens, &hwm);
    let _text = simulate_detokenize(metrics.tokens_generated);
    metrics
}

// ---------------------------------------------------------------------------
// Criterion benchmarks
// ---------------------------------------------------------------------------

fn bench_e2e_per_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_e2e_inference");
    let prompt = "Explain the theory of relativity in simple terms.";
    let max_tokens = 32;

    for backend in &[
        BenchBackend::Cpu,
        BenchBackend::OpenCl,
        BenchBackend::Vulkan,
    ] {
        group.bench_with_input(
            BenchmarkId::new("pipeline", backend),
            backend,
            |b, &backend| {
                b.iter(|| {
                    black_box(run_e2e_pipeline(backend, prompt, max_tokens))
                });
            },
        );
    }
    group.finish();
}

fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_throughput_scaling");
    let prompt = "Hello world";

    for &token_count in &[8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("cpu_tokens", token_count),
            &token_count,
            |b, &tc| {
                b.iter(|| {
                    black_box(run_e2e_pipeline(
                        BenchBackend::Cpu,
                        prompt,
                        tc,
                    ))
                });
            },
        );
    }
    group.finish();
}

fn bench_memory_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_hwm");

    group.bench_function("hwm_record_overhead", |b| {
        let hwm = MemoryHighWaterMark::new();
        b.iter(|| {
            hwm.record(black_box(4096));
            black_box(hwm.peak())
        });
    });

    group.finish();
}

fn bench_tokenize_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_tokenize");
    let prompt = "Explain quantum computing to a five-year-old child.";

    group.bench_function("tokenize_detokenize", |b| {
        b.iter(|| {
            let n = simulate_tokenize(black_box(prompt));
            black_box(simulate_detokenize(n))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_e2e_per_backend,
    bench_throughput_scaling,
    bench_memory_tracking,
    bench_tokenize_roundtrip,
);
criterion_main!(benches);

// ---------------------------------------------------------------------------
// Unit tests for measurement helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_per_sec() {
        let m = InferenceMetrics {
            tokens_generated: 100,
            generation_duration: Duration::from_secs(2),
            peak_memory_bytes: 0,
        };
        let tps = m.tokens_per_sec();
        assert!((tps - 50.0).abs() < 1e-6, "expected 50 tok/s, got {tps}");
    }

    #[test]
    fn test_ms_per_token() {
        let m = InferenceMetrics {
            tokens_generated: 10,
            generation_duration: Duration::from_millis(500),
            peak_memory_bytes: 0,
        };
        let mpt = m.ms_per_token();
        assert!(
            (mpt - 50.0).abs() < 1e-6,
            "expected 50 ms/tok, got {mpt}"
        );
    }

    #[test]
    fn test_zero_tokens_no_panic() {
        let m = InferenceMetrics {
            tokens_generated: 0,
            generation_duration: Duration::from_millis(100),
            peak_memory_bytes: 0,
        };
        assert_eq!(m.ms_per_token(), 0.0);
        // tokens_per_sec when duration > 0 but tokens == 0 → 0
        assert_eq!(m.tokens_per_sec(), 0.0);
    }

    #[test]
    fn test_memory_high_water_mark() {
        let hwm = MemoryHighWaterMark::new();
        assert_eq!(hwm.peak(), 0);

        hwm.record(1024);
        assert_eq!(hwm.peak(), 1024);

        hwm.record(512); // lower → no change
        assert_eq!(hwm.peak(), 1024);

        hwm.record(4096); // higher → update
        assert_eq!(hwm.peak(), 4096);

        hwm.reset();
        assert_eq!(hwm.peak(), 0);
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(BenchBackend::Cpu.to_string(), "cpu");
        assert_eq!(BenchBackend::OpenCl.to_string(), "opencl");
        assert_eq!(BenchBackend::Vulkan.to_string(), "vulkan");
    }

    #[test]
    fn test_e2e_pipeline_produces_metrics() {
        let m = run_e2e_pipeline(BenchBackend::Cpu, "hello", 8);
        assert_eq!(m.tokens_generated, 8);
        assert!(m.generation_duration > Duration::ZERO);
        assert!(m.peak_memory_bytes > 0);
    }
}
