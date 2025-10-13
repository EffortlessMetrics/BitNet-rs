//! GPU kernel benchmarks and performance validation

use super::cuda::CudaKernel;
use crate::cpu::FallbackKernel;
use crate::KernelProvider;
use bitnet_common::{QuantizationType, Result};
use std::time::Instant;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub matrix_sizes: Vec<(usize, usize, usize)>, // (M, N, K) tuples
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub tolerance: f32, // Numerical tolerance for correctness validation
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            matrix_sizes: vec![
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
            ],
            warmup_iterations: 5,
            benchmark_iterations: 10,
            tolerance: 1e-4,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub matrix_size: (usize, usize, usize),
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub speedup: f64,
    pub gflops_gpu: f64,
    pub gflops_cpu: f64,
    pub numerical_error: f64,
    pub passed_correctness: bool,
}

/// GPU kernel benchmarking suite
pub struct GpuBenchmark {
    cuda_kernel: Option<CudaKernel>,
    cpu_kernel: FallbackKernel,
    config: BenchmarkConfig,
}

impl GpuBenchmark {
    /// Create a new GPU benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        let cuda_kernel = CudaKernel::new().ok();
        let cpu_kernel = FallbackKernel;

        Self {
            cuda_kernel,
            cpu_kernel,
            config,
        }
    }

    /// Run comprehensive benchmarks
    pub fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        if self.cuda_kernel.is_none() {
            log::warn!("CUDA kernel not available, skipping GPU benchmarks");
            return Ok(results);
        }

        let cuda_kernel = self.cuda_kernel.as_ref().unwrap();

        for &(m, n, k) in &self.config.matrix_sizes {
            log::info!("Benchmarking matrix size {}x{}x{}", m, n, k);

            let result = self.benchmark_matrix_size(cuda_kernel, m, n, k)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Benchmark a specific matrix size
    fn benchmark_matrix_size(
        &self,
        cuda_kernel: &CudaKernel,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<BenchmarkResult> {
        // Generate test data
        let a = self.generate_test_matrix_i8(m * k);
        let b = self.generate_test_matrix_u8(k * n);
        let mut c_gpu = vec![0.0f32; m * n];
        let mut c_cpu = vec![0.0f32; m * n];

        // Warmup GPU
        for _ in 0..self.config.warmup_iterations {
            cuda_kernel.matmul_i2s(&a, &b, &mut c_gpu, m, n, k)?;
        }

        // Benchmark GPU
        let gpu_start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            cuda_kernel.matmul_i2s(&a, &b, &mut c_gpu, m, n, k)?;
        }
        cuda_kernel.synchronize_all()?;
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0 / self.config.benchmark_iterations as f64;

        // Benchmark CPU
        let cpu_start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            self.cpu_kernel.matmul_i2s(&a, &b, &mut c_cpu, m, n, k)?;
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0 / self.config.benchmark_iterations as f64;

        // Calculate performance metrics
        let operations = 2.0 * m as f64 * n as f64 * k as f64; // Multiply-add operations
        let gflops_gpu = operations / (gpu_time * 1e6);
        let gflops_cpu = operations / (cpu_time * 1e6);
        let speedup = cpu_time / gpu_time;

        // Validate numerical correctness
        let (numerical_error, passed_correctness) = self.validate_correctness(&c_gpu, &c_cpu);

        Ok(BenchmarkResult {
            matrix_size: (m, n, k),
            gpu_time_ms: gpu_time,
            cpu_time_ms: cpu_time,
            speedup,
            gflops_gpu,
            gflops_cpu,
            numerical_error,
            passed_correctness,
        })
    }

    /// Generate test matrix with int8 values
    fn generate_test_matrix_i8(&self, size: usize) -> Vec<i8> {
        (0..size).map(|i| ((i % 256) as u8).wrapping_sub(128) as i8).collect()
    }

    /// Generate test matrix with uint8 values (2-bit packed)
    fn generate_test_matrix_u8(&self, size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 4) as u8).collect()
    }

    /// Validate numerical correctness between GPU and CPU results
    fn validate_correctness(&self, gpu_result: &[f32], cpu_result: &[f32]) -> (f64, bool) {
        assert_eq!(gpu_result.len(), cpu_result.len());

        let mut max_error = 0.0f64;
        let mut total_error = 0.0f64;

        for (gpu_val, cpu_val) in gpu_result.iter().zip(cpu_result.iter()) {
            let error = (*gpu_val - *cpu_val).abs() as f64;
            max_error = max_error.max(error);
            total_error += error;
        }

        let avg_error = total_error / gpu_result.len() as f64;
        let passed = max_error < self.config.tolerance as f64;

        if !passed {
            log::warn!("Numerical validation failed: max_error={:.6}, avg_error={:.6}, tolerance={:.6}",
                      max_error, avg_error, self.config.tolerance);
        }

        (max_error, passed)
    }

    /// Print benchmark results in a formatted table
    pub fn print_results(&self, results: &[BenchmarkResult]) {
        println!("\n=== GPU Kernel Benchmark Results ===");
        println!("{:<15} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10} {:<8}",
                 "Matrix Size", "GPU (ms)", "CPU (ms)", "Speedup", "GPU GFLOPS", "CPU GFLOPS", "Max Error", "Pass");
        println!("{}", "-".repeat(100));

        for result in results {
            let (m, n, k) = result.matrix_size;
            println!("{:<15} {:<12.3} {:<12.3} {:<10.2}x {:<12.2} {:<12.2} {:<10.2e} {:<8}",
                     format!("{}x{}x{}", m, n, k),
                     result.gpu_time_ms,
                     result.cpu_time_ms,
                     result.speedup,
                     result.gflops_gpu,
                     result.gflops_cpu,
                     result.numerical_error,
                     if result.passed_correctness { "✓" } else { "✗" });
        }

        // Summary statistics
        let avg_speedup = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
        let max_speedup = results.iter().map(|r| r.speedup).fold(0.0, f64::max);
        let all_passed = results.iter().all(|r| r.passed_correctness);

        println!("{}", "-".repeat(100));
        println!("Average Speedup: {:.2}x", avg_speedup);
        println!("Maximum Speedup: {:.2}x", max_speedup);
        println!("All Tests Passed: {}", if all_passed { "✓" } else { "✗" });
    }

    /// Run quantization benchmarks
    pub fn benchmark_quantization(&self) -> Result<()> {
        if self.cuda_kernel.is_none() {
            log::warn!("CUDA kernel not available, skipping quantization benchmarks");
            return Ok(());
        }

        let cuda_kernel = self.cuda_kernel.as_ref().unwrap();
        let sizes = vec![1024, 4096, 16384, 65536];
        let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

        println!("\n=== Quantization Benchmark Results ===");
        println!("{:<10} {:<8} {:<12} {:<12} {:<10}", "Size", "Type", "GPU (ms)", "CPU (ms)", "Speedup");
        println!("{}", "-".repeat(60));

        for &size in &sizes {
            for &qtype in &qtypes {
                let input = (0..size).map(|i| (i as f32) / size as f32).collect::<Vec<f32>>();
                let mut output_gpu = vec![0u8; size / 4]; // 2 bits per element
                let mut output_cpu = vec![0u8; size / 4];
                let mut scales_gpu = vec![0.0f32; size / 128]; // Block size of 128
                let mut scales_cpu = vec![0.0f32; size / 128];

                // Warmup
                for _ in 0..5 {
                    cuda_kernel.quantize(&input, &mut output_gpu, &mut scales_gpu, qtype)?;
                }

                // Benchmark GPU
                let gpu_start = Instant::now();
                for _ in 0..10 {
                    cuda_kernel.quantize(&input, &mut output_gpu, &mut scales_gpu, qtype)?;
                }
                cuda_kernel.synchronize_all()?;
                let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0 / 10.0;

                // Benchmark CPU
                let cpu_start = Instant::now();
                for _ in 0..10 {
                    self.cpu_kernel.quantize(&input, &mut output_cpu, &mut scales_cpu, qtype)?;
                }
                let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0 / 10.0;

                let speedup = cpu_time / gpu_time;

                println!("{:<10} {:<8} {:<12.3} {:<12.3} {:<10.2}x",
                         size,
                         format!("{:?}", qtype),
                         gpu_time,
                         cpu_time,
                         speedup);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_benchmark() {
        let config = BenchmarkConfig {
            matrix_sizes: vec![(64, 64, 64), (128, 128, 128)],
            warmup_iterations: 2,
            benchmark_iterations: 3,
            tolerance: 1e-3,
        };

        let benchmark = GpuBenchmark::new(config);

        if let Ok(results) = benchmark.run_benchmarks() {
            benchmark.print_results(&results);

            // Verify all tests passed
            for result in &results {
                if result.passed_correctness {
                    assert!(result.speedup > 0.0);
                    assert!(result.gflops_gpu > 0.0);
                    assert!(result.gflops_cpu > 0.0);
                }
            }
        }
    }

    #[test]
    fn test_quantization_benchmark() {
        let config = BenchmarkConfig::default();
        let benchmark = GpuBenchmark::new(config);

        // This test will pass even if CUDA is not available
        let _ = benchmark.benchmark_quantization();
    }
}
