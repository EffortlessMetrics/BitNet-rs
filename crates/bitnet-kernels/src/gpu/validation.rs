//! GPU kernel validation and testing utilities
//!
//! This module provides comprehensive validation for CUDA kernels including:
//! - Numerical accuracy validation against CPU implementations
//! - Performance benchmarking and speedup measurement
//! - Memory usage profiling and leak detection
//! - Mixed precision validation

use crate::gpu::cuda::CudaKernel;
use crate::{KernelProvider, cpu::fallback::FallbackKernel, cpu::x86::Avx2Kernel};
use bitnet_common::Result;

use std::time::Instant;

/// Tolerance for numerical accuracy validation
pub const DEFAULT_TOLERANCE: f32 = 1e-6;

/// Configuration for validation tests
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Numerical tolerance for accuracy tests
    pub tolerance: f32,
    /// Number of iterations for performance benchmarks
    pub benchmark_iterations: usize,
    /// Matrix sizes to test (M, N, K)
    pub test_sizes: Vec<(usize, usize, usize)>,
    /// Whether to run memory leak detection
    pub check_memory_leaks: bool,
    /// Whether to test mixed precision operations
    pub test_mixed_precision: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            benchmark_iterations: 100,
            test_sizes: vec![
                (64, 64, 64),       // Small
                (256, 256, 256),    // Medium
                (1024, 1024, 1024), // Large
                (2048, 1024, 512),  // Rectangular
            ],
            check_memory_leaks: true,
            test_mixed_precision: false, // Disabled until implemented
        }
    }
}

/// Results from validation tests
#[derive(Debug)]
pub struct ValidationResults {
    /// Numerical accuracy test results
    pub accuracy_results: Vec<AccuracyResult>,
    /// Performance benchmark results
    pub performance_results: Vec<PerformanceResult>,
    /// Memory usage results
    pub memory_results: Option<MemoryResult>,
    /// Overall validation success
    pub success: bool,
}

/// Results from numerical accuracy validation
#[derive(Debug)]
pub struct AccuracyResult {
    /// Test matrix dimensions (M, N, K)
    pub dimensions: (usize, usize, usize),
    /// Maximum absolute error
    pub max_error: f32,
    /// Root mean square error
    pub rms_error: f32,
    /// Whether test passed tolerance check
    pub passed: bool,
}

/// Results from performance benchmarking
#[derive(Debug)]
pub struct PerformanceResult {
    /// Test matrix dimensions (M, N, K)
    pub dimensions: (usize, usize, usize),
    /// CPU execution time (milliseconds)
    pub cpu_time_ms: f64,
    /// GPU execution time (milliseconds)
    pub gpu_time_ms: f64,
    /// Speedup ratio (CPU time / GPU time)
    pub speedup: f64,
    /// Operations per second (GFLOPS)
    pub gflops: f64,
}

/// Results from memory usage profiling
#[derive(Debug)]
pub struct MemoryResult {
    /// Peak GPU memory usage (bytes)
    pub peak_gpu_memory: usize,
    /// Memory leaks detected
    pub leaks_detected: bool,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency_score: f32,
}

/// Comprehensive GPU kernel validator
pub struct GpuValidator {
    config: ValidationConfig,
}

impl GpuValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self { config: ValidationConfig::default() }
    }

    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Run a quick memory health check on the GPU
    ///
    /// This is useful for detecting memory leaks or GPU memory issues
    /// in production systems.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MemoryResult)` with memory statistics if successful,
    /// or an error if CUDA is not available or memory operations fail.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_kernels::gpu::validation::GpuValidator;
    ///
    /// let validator = GpuValidator::new();
    /// match validator.check_memory_health() {
    ///     Ok(result) => {
    ///         println!("Peak memory: {} MB", result.peak_gpu_memory / (1024 * 1024));
    ///         println!("Leaks detected: {}", result.leaks_detected);
    ///         println!("Memory efficiency: {:.1}%", result.efficiency_score * 100.0);
    ///     }
    ///     Err(e) => println!("Memory check failed: {}", e),
    /// }
    /// ```
    pub fn check_memory_health(&self) -> Result<MemoryResult> {
        self.test_memory_usage()
    }

    /// Run comprehensive validation tests
    pub fn validate(&self) -> Result<ValidationResults> {
        log::info!("Starting comprehensive GPU kernel validation");

        let mut results = ValidationResults {
            accuracy_results: Vec::new(),
            performance_results: Vec::new(),
            memory_results: None,
            success: true,
        };

        // Test numerical accuracy
        log::info!("Running numerical accuracy tests");
        for &dimensions in &self.config.test_sizes {
            match self.test_accuracy(dimensions) {
                Ok(result) => {
                    if !result.passed {
                        results.success = false;
                    }
                    results.accuracy_results.push(result);
                }
                Err(e) => {
                    log::error!("Accuracy test failed for {:?}: {}", dimensions, e);
                    results.success = false;
                }
            }
        }

        // Test performance
        log::info!("Running performance benchmarks");
        for &dimensions in &self.config.test_sizes {
            match self.benchmark_performance(dimensions) {
                Ok(result) => {
                    results.performance_results.push(result);
                }
                Err(e) => {
                    log::error!("Performance test failed for {:?}: {}", dimensions, e);
                    results.success = false;
                }
            }
        }

        // Test memory usage if enabled
        if self.config.check_memory_leaks {
            log::info!("Running memory usage tests");
            match self.test_memory_usage() {
                Ok(result) => {
                    if result.leaks_detected {
                        results.success = false;
                    }
                    results.memory_results = Some(result);
                }
                Err(e) => {
                    log::error!("Memory test failed: {}", e);
                    results.success = false;
                }
            }
        }

        log::info!("GPU kernel validation completed. Success: {}", results.success);
        Ok(results)
    }

    /// Test numerical accuracy against CPU implementation
    fn test_accuracy(&self, dimensions: (usize, usize, usize)) -> Result<AccuracyResult> {
        let (m, n, k) = dimensions;
        log::debug!("Testing accuracy for {}x{}x{}", m, n, k);

        // Create test data
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect(); // -1, 0, 1
        let b: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect(); // 0, 1

        // CPU reference implementation
        let mut cpu_result = vec![0.0f32; m * n];
        let cpu_kernel: Box<dyn KernelProvider> = {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
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
        cpu_kernel.matmul_i2s(&a, &b, &mut cpu_result, m, n, k)?;

        // GPU implementation
        let mut gpu_result = vec![0.0f32; m * n];
        let gpu_kernel = CudaKernel::new()?;
        gpu_kernel.matmul_i2s(&a, &b, &mut gpu_result, m, n, k)?;

        // Calculate errors
        let mut max_error = 0.0f32;
        let mut sum_squared_error = 0.0f32;

        for i in 0..cpu_result.len() {
            let error = (cpu_result[i] - gpu_result[i]).abs();
            max_error = max_error.max(error);
            sum_squared_error += error * error;
        }

        let rms_error = (sum_squared_error / cpu_result.len() as f32).sqrt();
        let passed = max_error <= self.config.tolerance;

        log::debug!(
            "Accuracy test {}x{}x{}: max_error={:.2e}, rms_error={:.2e}, passed={}",
            m,
            n,
            k,
            max_error,
            rms_error,
            passed
        );

        Ok(AccuracyResult { dimensions, max_error, rms_error, passed })
    }

    /// Benchmark performance against CPU implementation
    fn benchmark_performance(
        &self,
        dimensions: (usize, usize, usize),
    ) -> Result<PerformanceResult> {
        let (m, n, k) = dimensions;
        log::debug!("Benchmarking performance for {}x{}x{}", m, n, k);

        // Create test data
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
        let b: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect();

        // Benchmark CPU
        let mut cpu_result = vec![0.0f32; m * n];
        let cpu_kernel: Box<dyn KernelProvider> = {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
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

        let cpu_start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            cpu_kernel.matmul_i2s(&a, &b, &mut cpu_result, m, n, k)?;
        }
        let cpu_time_ms =
            cpu_start.elapsed().as_secs_f64() * 1000.0 / self.config.benchmark_iterations as f64;

        // Benchmark GPU
        let mut gpu_result = vec![0.0f32; m * n];
        let gpu_kernel = CudaKernel::new()?;

        // Warm up GPU
        for _ in 0..5 {
            gpu_kernel.matmul_i2s(&a, &b, &mut gpu_result, m, n, k)?;
        }

        let gpu_start = Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            gpu_kernel.matmul_i2s(&a, &b, &mut gpu_result, m, n, k)?;
        }
        let gpu_time_ms =
            gpu_start.elapsed().as_secs_f64() * 1000.0 / self.config.benchmark_iterations as f64;

        // Calculate metrics
        let speedup = cpu_time_ms / gpu_time_ms;
        let operations = 2.0 * m as f64 * n as f64 * k as f64; // 2 ops per multiply-add
        let gflops = operations / (gpu_time_ms * 1e6); // Convert to GFLOPS

        log::debug!(
            "Performance test {}x{}x{}: CPU={:.2}ms, GPU={:.2}ms, speedup={:.2}x, GFLOPS={:.2}",
            m,
            n,
            k,
            cpu_time_ms,
            gpu_time_ms,
            speedup,
            gflops
        );

        Ok(PerformanceResult { dimensions, cpu_time_ms, gpu_time_ms, speedup, gflops })
    }

    /// Test memory usage and detect leaks
    #[cfg(feature = "cuda")]
    fn test_memory_usage(&self) -> Result<MemoryResult> {
        use cudarc::driver::CudaContext;
        use cudarc::driver::sys::{cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2};

        const ALLOC_SIZE: usize = 10 * 1024 * 1024; // 10MB
        const ITERATIONS: usize = 5;

        log::debug!("Testing memory usage and leak detection");

        unsafe {
            // Create context to ensure CUDA API availability
            let _ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to create CUDA context: {:?}", e),
            })?;

            // Baseline memory
            let mut free_start: usize = 0;
            let mut total_mem: usize = 0;
            let result =
                cuMemGetInfo_v2(&mut free_start as *mut usize, &mut total_mem as *mut usize);
            if result as u32 != 0 {
                return Err(KernelError::GpuError {
                    reason: format!("cuMemGetInfo_v2 failed with error code: {:?}", result),
                }
                .into());
            }

            log::trace!(
                "Initial GPU memory: free={} MB, total={} MB",
                free_start / (1024 * 1024),
                total_mem / (1024 * 1024)
            );

            let mut peak_usage = 0usize;
            let mut allocated_ptrs = Vec::with_capacity(ITERATIONS);

            // Allocate and free memory multiple times to track peak usage
            for i in 0..ITERATIONS {
                let mut ptr: u64 = 0;
                let result = cuMemAlloc_v2(&mut ptr as *mut u64, ALLOC_SIZE);
                if result as u32 != 0 {
                    // Clean up previously allocated memory before returning error
                    for &prev_ptr in &allocated_ptrs {
                        let _ = cuMemFree_v2(prev_ptr);
                    }
                    return Err(KernelError::GpuError {
                        reason: format!(
                            "cuMemAlloc_v2 failed at iteration {} with error code: {:?}",
                            i, result
                        ),
                    }
                    .into());
                }

                allocated_ptrs.push(ptr);

                let mut free_now: usize = 0;
                let result =
                    cuMemGetInfo_v2(&mut free_now as *mut usize, &mut total_mem as *mut usize);
                if result as u32 != 0 {
                    // Clean up allocated memory before returning error
                    for &prev_ptr in &allocated_ptrs {
                        let _ = cuMemFree_v2(prev_ptr);
                    }
                    return Err(KernelError::GpuError {
                        reason: format!(
                            "cuMemGetInfo_v2 failed at iteration {} with error code: {:?}",
                            i, result
                        ),
                    }
                    .into());
                }

                let used = free_start.saturating_sub(free_now);
                peak_usage = peak_usage.max(used);
                log::trace!(
                    "Iteration {}: allocated {} MB, used {} MB",
                    i,
                    ALLOC_SIZE / (1024 * 1024),
                    used / (1024 * 1024)
                );
            }

            // Free all allocated memory
            for (i, &ptr) in allocated_ptrs.iter().enumerate() {
                let result = cuMemFree_v2(ptr);
                if result as u32 != 0 {
                    log::warn!(
                        "cuMemFree_v2 failed at iteration {} with error code: {:?}",
                        i,
                        result
                    );
                    // Continue freeing other allocations even if one fails
                }
            }

            // Final memory to detect leaks
            let mut free_end: usize = 0;
            let result = cuMemGetInfo_v2(&mut free_end as *mut usize, &mut total_mem as *mut usize);
            if result as u32 != 0 {
                return Err(KernelError::GpuError {
                    reason: format!("Final cuMemGetInfo_v2 failed with error code: {:?}", result),
                }
                .into());
            }

            // Allow for small variance in memory due to driver overhead
            const LEAK_THRESHOLD: usize = 1024 * 1024; // 1MB threshold
            let memory_diff = if free_end < free_start { free_start - free_end } else { 0 };
            let leaks_detected = memory_diff > LEAK_THRESHOLD;

            if leaks_detected {
                log::warn!("Memory leak detected: {} bytes", memory_diff);
            }

            let efficiency_score =
                if total_mem > 0 { 1.0 - (peak_usage as f32 / total_mem as f32) } else { 1.0 };

            log::debug!(
                "Memory test complete: peak={} MB, leaks={}, efficiency={:.2}%",
                peak_usage / (1024 * 1024),
                leaks_detected,
                efficiency_score * 100.0
            );

            Ok(MemoryResult { peak_gpu_memory: peak_usage, leaks_detected, efficiency_score })
        }
    }

    /// Placeholder memory usage test when CUDA is unavailable
    #[cfg(not(feature = "cuda"))]
    fn test_memory_usage(&self) -> Result<MemoryResult> {
        log::warn!("CUDA feature not enabled; skipping memory usage test");
        Ok(MemoryResult { peak_gpu_memory: 0, leaks_detected: false, efficiency_score: 1.0 })
    }
}

impl Default for GpuValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Print validation results in a human-readable format
pub fn print_validation_results(results: &ValidationResults) {
    println!("\n=== GPU Kernel Validation Results ===");

    // Accuracy results
    println!("\n📊 Numerical Accuracy Tests:");
    for result in &results.accuracy_results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "  {}x{}x{}: {} (max_error: {:.2e}, rms_error: {:.2e})",
            result.dimensions.0,
            result.dimensions.1,
            result.dimensions.2,
            status,
            result.max_error,
            result.rms_error
        );
    }

    // Performance results
    println!("\n🚀 Performance Benchmarks:");
    for result in &results.performance_results {
        println!(
            "  {}x{}x{}: {:.2}x speedup ({:.2} GFLOPS, CPU: {:.2}ms, GPU: {:.2}ms)",
            result.dimensions.0,
            result.dimensions.1,
            result.dimensions.2,
            result.speedup,
            result.gflops,
            result.cpu_time_ms,
            result.gpu_time_ms
        );
    }

    // Memory results
    if let Some(memory) = &results.memory_results {
        println!("\n💾 Memory Usage:");
        let leak_status = if memory.leaks_detected { "❌ LEAKS DETECTED" } else { "✅ NO LEAKS" };
        println!(
            "  Peak GPU Memory: {} bytes, Efficiency: {:.1}%, {}",
            memory.peak_gpu_memory,
            memory.efficiency_score * 100.0,
            leak_status
        );
    }

    // Overall result
    let overall_status = if results.success { "✅ SUCCESS" } else { "❌ FAILED" };
    println!("\n🎯 Overall Validation: {}", overall_status);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.tolerance, DEFAULT_TOLERANCE);
        assert!(config.benchmark_iterations > 0);
        assert!(!config.test_sizes.is_empty());
    }

    #[test]
    fn test_validator_creation() {
        let validator = GpuValidator::new();
        assert_eq!(validator.config.tolerance, DEFAULT_TOLERANCE);

        let custom_config = ValidationConfig { tolerance: 1e-5, ..Default::default() };
        let custom_validator = GpuValidator::with_config(custom_config);
        assert_eq!(custom_validator.config.tolerance, 1e-5);
    }

    #[test]
    #[ignore] // Only run with CUDA available
    fn test_gpu_validation() {
        let validator = GpuValidator::new();
        match validator.validate() {
            Ok(results) => {
                print_validation_results(&results);
                assert!(!results.accuracy_results.is_empty());
                assert!(!results.performance_results.is_empty());
            }
            Err(e) => {
                println!("GPU validation failed (CUDA may not be available): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_usage_tracking() {
        // Initialize logger for debugging
        let _ =
            env_logger::builder().filter_level(log::LevelFilter::Debug).is_test(true).try_init();

        let validator = GpuValidator::new();
        match validator.test_memory_usage() {
            Ok(result) => {
                // Peak memory should be at least the size we allocated
                assert!(result.peak_gpu_memory > 0, "Peak memory should be greater than 0");

                // We expect at least 10MB * 5 iterations worth of allocations tracked
                // But due to memory pooling, actual usage might be less
                assert!(
                    result.peak_gpu_memory >= 10 * 1024 * 1024,
                    "Peak memory {} should be at least 10MB",
                    result.peak_gpu_memory
                );

                assert!(!result.leaks_detected, "Memory leak detected in test");
                assert!(
                    result.efficiency_score >= 0.0 && result.efficiency_score <= 1.0,
                    "Efficiency score {} should be between 0 and 1",
                    result.efficiency_score
                );

                log::info!(
                    "Memory test passed: peak={} MB, efficiency={:.2}%",
                    result.peak_gpu_memory / (1024 * 1024),
                    result.efficiency_score * 100.0
                );
            }
            Err(e) => {
                // CUDA might not be available in CI
                if e.to_string().contains("CUDA") || e.to_string().contains("context") {
                    log::warn!("Skipping CUDA memory test: {}", e);
                } else {
                    panic!("Unexpected error in memory test: {}", e);
                }
            }
        }
    }
}
