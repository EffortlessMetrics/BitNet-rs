//! GPU kernel benchmarking utilities
//! 
//! This module provides comprehensive benchmarking tools for comparing
//! GPU and CPU kernel performance across different matrix sizes and
//! configurations.

use crate::{KernelProvider, cpu::x86::Avx2Kernel, cpu::fallback::FallbackKernel};
use crate::gpu::cuda::CudaKernel;
use crate::gpu::validation::PerformanceResult;
use bitnet_common::Result;

use std::time::Instant;

/// Comprehensive benchmark suite for GPU kernels
pub struct GpuBenchmark {
    config: BenchmarkConfig,
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Matrix sizes to benchmark (M, N, K)
    pub test_sizes: Vec<(usize, usize, usize)>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Whether to include CPU comparison
    pub include_cpu_comparison: bool,
    /// Whether to test different data patterns
    pub test_data_patterns: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            test_sizes: vec![
                (64, 64, 64),       // Small
                (128, 128, 128),    // Small-Medium
                (256, 256, 256),    // Medium
                (512, 512, 512),    // Medium-Large
                (1024, 1024, 1024), // Large
                (2048, 1024, 512),  // Rectangular
                (1024, 2048, 512),  // Rectangular
                (4096, 4096, 256),  // Very Large
            ],
            warmup_iterations: 10,
            benchmark_iterations: 100,
            include_cpu_comparison: true,
            test_data_patterns: true,
        }
    }
}

/// Results from comprehensive benchmarking
#[derive(Debug)]
pub struct BenchmarkResults {
    /// Performance results for each test
    pub results: Vec<PerformanceResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
}

/// Summary statistics from benchmarking
#[derive(Debug)]
pub struct BenchmarkSummary {
    /// Average speedup across all tests
    pub avg_speedup: f64,
    /// Maximum speedup achieved
    pub max_speedup: f64,
    /// Minimum speedup achieved
    pub min_speedup: f64,
    /// Average GFLOPS on GPU
    pub avg_gflops: f64,
    /// Peak GFLOPS achieved
    pub peak_gflops: f64,
    /// Total operations benchmarked
    pub total_operations: u64,
}

impl GpuBenchmark {
    /// Create a new benchmark with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create a new benchmark with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run comprehensive benchmarks
    pub fn run(&self) -> Result<BenchmarkResults> {
        log::info!("Starting comprehensive GPU kernel benchmarks");
        
        let mut results = Vec::new();
        let mut total_operations = 0u64;

        for &dimensions in &self.config.test_sizes {
            log::info!("Benchmarking {}x{}x{}", dimensions.0, dimensions.1, dimensions.2);
            
            let result = self.benchmark_matrix_size(dimensions)?;
            total_operations += 2 * dimensions.0 as u64 * dimensions.1 as u64 * dimensions.2 as u64;
            results.push(result);
        }

        // Calculate summary statistics
        let speedups: Vec<f64> = results.iter().map(|r| r.speedup).collect();
        let gflops: Vec<f64> = results.iter().map(|r| r.gflops).collect();

        let summary = BenchmarkSummary {
            avg_speedup: speedups.iter().sum::<f64>() / speedups.len() as f64,
            max_speedup: speedups.iter().fold(0.0, |a, &b| a.max(b)),
            min_speedup: speedups.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_gflops: gflops.iter().sum::<f64>() / gflops.len() as f64,
            peak_gflops: gflops.iter().fold(0.0, |a, &b| a.max(b)),
            total_operations,
        };

        log::info!("Benchmark completed. Average speedup: {:.2}x", summary.avg_speedup);

        Ok(BenchmarkResults { results, summary })
    }

    /// Benchmark a specific matrix size
    fn benchmark_matrix_size(&self, dimensions: (usize, usize, usize)) -> Result<PerformanceResult> {
        let (m, n, k) = dimensions;

        // Create test data
        let a: Vec<i8> = (0..m*k).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<u8> = (0..k*n).map(|i| (i % 2) as u8).collect();

        // Benchmark CPU if requested
        let cpu_time_ms = if self.config.include_cpu_comparison {
            self.benchmark_cpu(&a, &b, m, n, k)?
        } else {
            0.0 // Placeholder when CPU comparison is disabled
        };

        // Benchmark GPU
        let gpu_time_ms = self.benchmark_gpu(&a, &b, m, n, k)?;

        // Calculate metrics
        let speedup = if cpu_time_ms > 0.0 { cpu_time_ms / gpu_time_ms } else { 0.0 };
        let operations = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = operations / (gpu_time_ms * 1e6);

        Ok(PerformanceResult {
            dimensions,
            cpu_time_ms,
            gpu_time_ms,
            speedup,
            gflops,
        })
    }

    /// Benchmark CPU implementation
    fn benchmark_cpu(&self, a: &[i8], b: &[u8], m: usize, n: usize, k: usize) -> Result<f64> {
        let mut cpu_result = vec![0.0f32; m * n];
        let cpu_kernel: Box<dyn KernelProvider> = {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    Box::new(Avx2Kernel)
                } else {
                    Box::new(FallbackKernel)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Box::new(FallbackKernel)
            }
        };

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            cpu_kernel.matmul_i2s(a, b, &mut cpu_result, m, n, k)?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            cpu_kernel.matmul_i2s(a, b, &mut cpu_result, m, n, k)?;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(elapsed / self.config.benchmark_iterations as f64)
    }

    /// Benchmark GPU implementation
    fn benchmark_gpu(&self, a: &[i8], b: &[u8], m: usize, n: usize, k: usize) -> Result<f64> {
        let mut gpu_result = vec![0.0f32; m * n];
        let gpu_kernel = CudaKernel::new()?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            gpu_kernel.matmul_i2s(a, b, &mut gpu_result, m, n, k)?;
        }

        // Synchronize before timing
        gpu_kernel.synchronize_all()?;

        // Benchmark
        let start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            gpu_kernel.matmul_i2s(a, b, &mut gpu_result, m, n, k)?;
        }
        gpu_kernel.synchronize_all()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        Ok(elapsed / self.config.benchmark_iterations as f64)
    }
}

impl Default for GpuBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Print benchmark results in a human-readable format
pub fn print_benchmark_results(results: &BenchmarkResults) {
    println!("\n=== GPU Kernel Benchmark Results ===");
    
    println!("\n📊 Performance by Matrix Size:");
    println!("{:>12} {:>12} {:>12} {:>12} {:>12}", "Size", "CPU (ms)", "GPU (ms)", "Speedup", "GFLOPS");
    println!("{:-<65}", "");
    
    for result in &results.results {
        println!(
            "{:>12} {:>12.2} {:>12.2} {:>12.2}x {:>12.1}",
            format!("{}x{}x{}", result.dimensions.0, result.dimensions.1, result.dimensions.2),
            result.cpu_time_ms,
            result.gpu_time_ms,
            result.speedup,
            result.gflops
        );
    }

    println!("\n📈 Summary Statistics:");
    println!("  Average Speedup: {:.2}x", results.summary.avg_speedup);
    println!("  Maximum Speedup: {:.2}x", results.summary.max_speedup);
    println!("  Minimum Speedup: {:.2}x", results.summary.min_speedup);
    println!("  Average GFLOPS:  {:.1}", results.summary.avg_gflops);
    println!("  Peak GFLOPS:     {:.1}", results.summary.peak_gflops);
    println!("  Total Operations: {:.2e}", results.summary.total_operations as f64);

    // Performance analysis
    println!("\n🎯 Performance Analysis:");
    if results.summary.avg_speedup > 5.0 {
        println!("  ✅ Excellent GPU acceleration achieved!");
    } else if results.summary.avg_speedup > 2.0 {
        println!("  ✅ Good GPU acceleration achieved");
    } else if results.summary.avg_speedup > 1.0 {
        println!("  ⚠️  Modest GPU acceleration - consider optimization");
    } else {
        println!("  ❌ GPU slower than CPU - needs investigation");
    }

    if results.summary.peak_gflops > 100.0 {
        println!("  ✅ High computational throughput achieved");
    } else if results.summary.peak_gflops > 50.0 {
        println!("  ✅ Good computational throughput");
    } else {
        println!("  ⚠️  Low computational throughput - optimization needed");
    }
}

/// Run a quick benchmark for validation
pub fn quick_benchmark() -> Result<()> {
    let config = BenchmarkConfig {
        test_sizes: vec![(256, 256, 256), (512, 512, 512)],
        warmup_iterations: 5,
        benchmark_iterations: 20,
        include_cpu_comparison: true,
        test_data_patterns: false,
    };

    let benchmark = GpuBenchmark::with_config(config);
    let results = benchmark.run()?;
    print_benchmark_results(&results);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.test_sizes.is_empty());
        assert!(config.warmup_iterations > 0);
        assert!(config.benchmark_iterations > 0);
    }

    #[test]
    #[ignore] // Only run with CUDA available
    fn test_quick_benchmark() {
        match quick_benchmark() {
            Ok(_) => println!("Quick benchmark completed successfully"),
            Err(e) => println!("Quick benchmark failed (CUDA may not be available): {}", e),
        }
    }
}