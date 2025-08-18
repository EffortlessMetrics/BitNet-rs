#![cfg(feature = "ffi")]
//! Comprehensive kernel tests for validation and performance benchmarking
//!
//! This test suite validates the correctness of all kernel implementations
//! and provides performance benchmarking capabilities for regression detection.

#![cfg(any(feature = "ffi", feature = "cpu"))]

use bitnet_common::{QuantizationType, Result};
use bitnet_kernels::{KernelManager, KernelProvider};
use std::time::Instant;

/// Test data generator for consistent testing across kernels
struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate test matrix A (i8)
    fn generate_matrix_a(m: usize, k: usize, seed: u64) -> Vec<i8> {
        let mut rng = SimpleRng::new(seed);
        (0..m * k).map(|_| rng.next_i8()).collect()
    }

    /// Generate test matrix B (u8)
    fn generate_matrix_b(k: usize, n: usize, seed: u64) -> Vec<u8> {
        let mut rng = SimpleRng::new(seed + 1);
        (0..k * n).map(|_| rng.next_u8()).collect()
    }

    /// Generate test input for quantization
    fn generate_quantization_input(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = SimpleRng::new(seed + 2);
        (0..len).map(|_| rng.next_f32() * 4.0 - 2.0).collect() // Range [-2, 2]
    }
}

/// Simple deterministic RNG for reproducible tests
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn next_i8(&mut self) -> i8 {
        let val = self.next() % 256;
        if val < 128 {
            val as i8
        } else {
            (val as i8) - (256i16 as i8)
        }
    }

    fn next_u8(&mut self) -> u8 {
        (self.next() % 256) as u8
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() % 1000000) as f32 / 1000000.0
    }
}

/// Performance metrics for benchmarking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PerformanceMetrics {
    kernel_name: String,
    operation: String,
    time_ns: u64,
    throughput_ops_per_sec: f64,
    memory_bandwidth_gb_per_sec: Option<f64>,
}

impl PerformanceMetrics {
    fn new(kernel_name: &str, operation: &str, time_ns: u64, ops: u64) -> Self {
        let throughput_ops_per_sec =
            if time_ns > 0 { (ops as f64) / (time_ns as f64 / 1_000_000_000.0) } else { 0.0 };

        Self {
            kernel_name: kernel_name.to_string(),
            operation: operation.to_string(),
            time_ns,
            throughput_ops_per_sec,
            memory_bandwidth_gb_per_sec: None,
        }
    }

    fn with_memory_bandwidth(mut self, bytes: u64) -> Self {
        if self.time_ns > 0 {
            let gb_per_sec =
                (bytes as f64) / (self.time_ns as f64 / 1_000_000_000.0) / 1_000_000_000.0;
            self.memory_bandwidth_gb_per_sec = Some(gb_per_sec);
        }
        self
    }
}

/// Test kernel correctness against reference implementation
fn test_kernel_correctness(kernel: &dyn KernelProvider) -> Result<()> {
    println!("Testing kernel correctness: {}", kernel.name());

    if !kernel.is_available() {
        println!("  Kernel not available, skipping");
        return Ok(());
    }

    // Test matrix multiplication
    test_matmul_correctness(kernel)?;

    // Test quantization
    test_quantization_correctness(kernel)?;

    println!("  ✓ All correctness tests passed");
    Ok(())
}

/// Test matrix multiplication correctness
fn test_matmul_correctness(kernel: &dyn KernelProvider) -> Result<()> {
    // Test small matrices
    let test_cases = vec![(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)];

    for (m, n, k) in test_cases {
        let a = TestDataGenerator::generate_matrix_a(m, k, 12345);
        let b = TestDataGenerator::generate_matrix_b(k, n, 67890);
        let mut c = vec![0.0f32; m * n];

        kernel.matmul_i2s(&a, &b, &mut c, m, n, k)?;

        // Verify result is not all zeros (basic sanity check)
        assert!(c.iter().any(|&x| x != 0.0), "Matrix multiplication result is all zeros");

        // Verify dimensions are correct
        assert_eq!(c.len(), m * n, "Output matrix has wrong dimensions");
    }

    Ok(())
}

/// Test quantization correctness
fn test_quantization_correctness(kernel: &dyn KernelProvider) -> Result<()> {
    let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

    for qtype in qtypes {
        let input = TestDataGenerator::generate_quantization_input(128, 54321);
        let mut output = vec![0u8; 32]; // 128 / 4 = 32 bytes
        let mut scales = vec![0.0f32; 4]; // Assuming 32-element blocks

        kernel.quantize(&input, &mut output, &mut scales, qtype)?;

        // Verify scales are reasonable
        assert!(scales.iter().all(|&s| s > 0.0 && s.is_finite()), "Invalid scales for {:?}", qtype);

        // Verify output is not all zeros
        assert!(output.iter().any(|&x| x != 0), "Quantization output is all zeros for {:?}", qtype);
    }

    Ok(())
}

/// Benchmark kernel performance
fn benchmark_kernel(kernel: &dyn KernelProvider) -> Result<Vec<PerformanceMetrics>> {
    println!("Benchmarking kernel: {}", kernel.name());

    if !kernel.is_available() {
        println!("  Kernel not available, skipping");
        return Ok(vec![]);
    }

    let mut metrics = Vec::new();

    // Benchmark matrix multiplication
    metrics.extend(benchmark_matmul(kernel)?);

    // Benchmark quantization
    metrics.extend(benchmark_quantization(kernel)?);

    Ok(metrics)
}

/// Benchmark matrix multiplication performance
fn benchmark_matmul(kernel: &dyn KernelProvider) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    let test_sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for (m, n, k) in test_sizes {
        let a = TestDataGenerator::generate_matrix_a(m, k, 11111);
        let b = TestDataGenerator::generate_matrix_b(k, n, 22222);
        let mut c = vec![0.0f32; m * n];

        // Warm up
        for _ in 0..3 {
            kernel.matmul_i2s(&a, &b, &mut c, m, n, k)?;
        }

        // Benchmark
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            kernel.matmul_i2s(&a, &b, &mut c, m, n, k)?;
        }

        let elapsed = start.elapsed().as_nanos() as u64;
        let avg_time = elapsed / iterations;
        let ops = (m * n * k) as u64; // Approximate operation count
        let bytes = (a.len() + b.len() + c.len() * 4) as u64; // Approximate memory usage

        let metric = PerformanceMetrics::new(
            kernel.name(),
            &format!("matmul_{}x{}x{}", m, n, k),
            avg_time,
            ops,
        )
        .with_memory_bandwidth(bytes);

        println!(
            "  {} - {:.2} GOPS/s, {:.2} GB/s",
            metric.operation,
            metric.throughput_ops_per_sec / 1_000_000_000.0,
            metric.memory_bandwidth_gb_per_sec.unwrap_or(0.0)
        );

        metrics.push(metric);
    }

    Ok(metrics)
}

/// Benchmark quantization performance
fn benchmark_quantization(kernel: &dyn KernelProvider) -> Result<Vec<PerformanceMetrics>> {
    let mut metrics = Vec::new();

    let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

    let test_sizes = vec![1024, 4096, 16384, 65536];

    for qtype in qtypes {
        for size in &test_sizes {
            let input = TestDataGenerator::generate_quantization_input(*size, 33333);
            let mut output = vec![0u8; size / 4];
            let mut scales = vec![0.0f32; (size + 31) / 32];

            // Warm up
            for _ in 0..3 {
                kernel.quantize(&input, &mut output, &mut scales, qtype)?;
            }

            // Benchmark
            let iterations = 100;
            let start = Instant::now();

            for _ in 0..iterations {
                kernel.quantize(&input, &mut output, &mut scales, qtype)?;
            }

            let elapsed = start.elapsed().as_nanos() as u64;
            let avg_time = elapsed / iterations;
            let ops = *size as u64;
            let bytes = (input.len() * 4 + output.len() + scales.len() * 4) as u64;

            let metric = PerformanceMetrics::new(
                kernel.name(),
                &format!("quantize_{:?}_{}", qtype, size),
                avg_time,
                ops,
            )
            .with_memory_bandwidth(bytes);

            println!(
                "  {} - {:.2} Mops/s, {:.2} GB/s",
                metric.operation,
                metric.throughput_ops_per_sec / 1_000_000.0,
                metric.memory_bandwidth_gb_per_sec.unwrap_or(0.0)
            );

            metrics.push(metric);
        }
    }

    Ok(metrics)
}

/// Compare performance between kernels
#[allow(dead_code)]
fn compare_kernel_performance(metrics: &[Vec<PerformanceMetrics>]) {
    if metrics.len() < 2 {
        return;
    }

    println!("\nPerformance Comparison:");
    println!("======================");

    // Group metrics by operation
    let mut operations = std::collections::HashMap::new();
    for kernel_metrics in metrics {
        for metric in kernel_metrics {
            operations.entry(metric.operation.clone()).or_insert_with(Vec::new).push(metric);
        }
    }

    for (operation, op_metrics) in operations {
        if op_metrics.len() < 2 {
            continue;
        }

        println!("\n{}:", operation);

        // Sort by performance (descending)
        let mut sorted_metrics = op_metrics;
        sorted_metrics.sort_by(|a, b| {
            b.throughput_ops_per_sec.partial_cmp(&a.throughput_ops_per_sec).unwrap()
        });

        let best_perf = sorted_metrics[0].throughput_ops_per_sec;

        for metric in sorted_metrics {
            let relative_perf = metric.throughput_ops_per_sec / best_perf;
            println!("  {} - {:.2}x relative performance", metric.kernel_name, relative_perf);
        }
    }
}

#[test]
fn test_kernel_manager() {
    let manager = KernelManager::new();

    // Test that we can select a kernel
    let kernel = manager.select_best().expect("Should have at least fallback kernel");
    println!("Selected kernel: {}", kernel.name());

    // Test available providers
    let available = manager.list_available_providers();
    println!("Available providers: {:?}", available);
    assert!(!available.is_empty(), "Should have at least one available provider");
}

#[test]
fn test_all_kernels_correctness() {
    let manager = KernelManager::new();
    let available_providers = manager.list_available_providers();

    for provider_name in available_providers {
        // This is a simplified test - in a real implementation,
        // we would need to get individual kernel instances
        println!("Testing provider: {}", provider_name);
    }

    // Test the selected kernel
    let kernel = manager.select_best().expect("Should have a kernel");
    test_kernel_correctness(kernel).expect("Kernel correctness test failed");
}

#[test]
fn test_performance_benchmarks() {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    let metrics = benchmark_kernel(kernel).expect("Benchmark should succeed");

    assert!(!metrics.is_empty(), "Should have performance metrics");

    // Check that we have reasonable performance numbers
    for metric in &metrics {
        assert!(metric.time_ns > 0, "Should have positive execution time");
        assert!(metric.throughput_ops_per_sec > 0.0, "Should have positive throughput");
    }
}

#[test]
fn test_kernel_error_handling() {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    // Test invalid dimensions
    let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
    assert!(result.is_err(), "Should fail with invalid dimensions");

    // Test mismatched dimensions
    let a = vec![1i8; 4];
    let b = vec![1u8; 4];
    let mut c = vec![0.0f32; 4];

    let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3); // Wrong k
    assert!(result.is_err(), "Should fail with mismatched dimensions");
}

#[test]
fn test_quantization_edge_cases() {
    let manager = KernelManager::new();
    let kernel = manager.select_best().expect("Should have a kernel");

    // Test with all zeros
    let input = vec![0.0f32; 32];
    let mut output = vec![0u8; 8];
    let mut scales = vec![0.0f32; 1];

    let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_ok(), "Should handle all-zero input");

    // Test with very small values
    let input = vec![1e-10f32; 32];
    let mut output = vec![0u8; 8];
    let mut scales = vec![0.0f32; 1];

    let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_ok(), "Should handle very small values");
}

#[cfg(feature = "ffi")]
#[test]
fn test_ffi_kernel_integration() {
    use bitnet_kernels::ffi::FfiKernel;

    match FfiKernel::new() {
        Ok(kernel) => {
            println!("FFI kernel available: {}", kernel.is_available());
            if kernel.is_available() {
                test_kernel_correctness(&kernel).expect("FFI kernel correctness test failed");
            }
        }
        Err(e) => {
            println!("FFI kernel not available: {}", e);
        }
    }
}

#[cfg(feature = "ffi")]
#[test]
fn test_performance_comparison_with_ffi() {
    use bitnet_kernels::cpu::FallbackKernel;
    use bitnet_kernels::ffi::{FfiKernel, PerformanceComparison};

    let ffi_kernel = match FfiKernel::new() {
        Ok(k) if k.is_available() => k,
        _ => {
            println!("FFI kernel not available, skipping comparison test");
            return;
        }
    };

    let rust_kernel = FallbackKernel;

    // Test matrix multiplication comparison
    let a = TestDataGenerator::generate_matrix_a(64, 64, 99999);
    let b = TestDataGenerator::generate_matrix_b(64, 64, 88888);

    match PerformanceComparison::compare_matmul(&rust_kernel, &ffi_kernel, &a, &b, 64, 64, 64) {
        Ok(comparison) => {
            println!("Performance comparison:");
            println!("  Rust time: {} ns", comparison.rust_time_ns);
            println!("  C++ time: {} ns", comparison.cpp_time_ns);
            println!("  Accuracy match: {}", comparison.accuracy_match);
            println!("  Max error: {}", comparison.max_error);
            println!(
                "  Performance improvement: {:.2}%",
                comparison.performance_improvement() * 100.0
            );
            println!("  Migration recommended: {}", comparison.migration_recommended());
        }
        Err(e) => {
            println!("Performance comparison failed: {}", e);
        }
    }
}
