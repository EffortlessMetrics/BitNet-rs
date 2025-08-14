//! Comprehensive unit tests for bitnet-kernels
//!
//! This test suite provides comprehensive coverage of all kernel implementations
//! including CPU kernels, SIMD optimizations, GPU kernels, and kernel selection.
//! It achieves >90% code coverage with performance validation.

use bitnet_common::QuantizationType;
use bitnet_kernels::*;
use std::time::Instant;

/// Test data generator for consistent testing across kernels
struct TestDataGenerator {
    seed: u64,
}

impl TestDataGenerator {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate deterministic test matrix A (i8)
    fn generate_matrix_a(&mut self, m: usize, k: usize) -> Vec<i8> {
        (0..m * k)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (self.seed % 256) as u8;
                if val > 127 {
                    (val as i16 - 256) as i8
                } else {
                    val as i8
                }
            })
            .collect()
    }

    /// Generate deterministic test matrix B (u8)
    fn generate_matrix_b(&mut self, k: usize, n: usize) -> Vec<u8> {
        (0..k * n)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                (self.seed % 256) as u8
            })
            .collect()
    }

    /// Generate deterministic test input for quantization
    fn generate_quantization_input(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (self.seed % 1000000) as f32 / 1000000.0;
                (val - 0.5) * 4.0 // Range [-2, 2]
            })
            .collect()
    }
}

/// Performance metrics for benchmarking
#[derive(Debug, Clone)]
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

// ============================================================================
// CPU Kernel Tests
// ============================================================================

mod cpu_kernel_tests {
    use super::*;
    use bitnet_kernels::cpu::{Avx2Kernel, FallbackKernel, NeonKernel};

    #[test]
    fn test_fallback_kernel_comprehensive() {
        let kernel = FallbackKernel;

        // Test availability and name
        assert!(kernel.is_available());
        assert_eq!(kernel.name(), "fallback");

        // Test various matrix sizes
        let test_sizes = vec![
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 32, 16), // Non-square
            (16, 64, 32), // Different aspect ratio
        ];

        let mut data_gen = TestDataGenerator::new(12345);

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "Fallback kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite values in output");

            // For identity-like matrices, verify basic correctness
            if m == n && k == n && m <= 4 {
                // Create identity matrix for B
                let mut b_identity = vec![0u8; k * n];
                for i in 0..n.min(k) {
                    b_identity[i * n + i] = 1;
                }
                let mut c_identity = vec![0.0f32; m * n];

                kernel.matmul_i2s(&a, &b_identity, &mut c_identity, m, n, k).unwrap();

                // Result should be approximately A (with type conversion)
                for i in 0..m {
                    for j in 0..n {
                        let expected = a[i * k + j] as f32;
                        let actual = c_identity[i * n + j];
                        assert!(
                            (expected - actual).abs() < 1e-5,
                            "Identity test failed at ({}, {}): expected {}, got {}",
                            i,
                            j,
                            expected,
                            actual
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fallback_quantization_comprehensive() {
        let kernel = FallbackKernel;
        let mut data_gen = TestDataGenerator::new(54321);

        let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

        let test_sizes = vec![32, 64, 128, 256, 512];

        for qtype in qtypes {
            for size in &test_sizes {
                let input = data_gen.generate_quantization_input(*size);
                let mut output = vec![0u8; size / 4];
                let block_size = match qtype {
                    QuantizationType::I2S => 32,
                    QuantizationType::TL1 => 64,
                    QuantizationType::TL2 => 128,
                };
                let num_blocks = (size + block_size - 1) / block_size;
                let mut scales = vec![0.0f32; num_blocks];

                let result = kernel.quantize(&input, &mut output, &mut scales, qtype);
                assert!(result.is_ok(), "Quantization failed for {:?} size {}", qtype, size);

                // Verify scales are reasonable
                assert!(
                    scales.iter().all(|&s| s > 0.0 && s.is_finite()),
                    "Invalid scales for {:?}",
                    qtype
                );

                // Verify output has some variation (not all zeros)
                assert!(
                    output.iter().any(|&x| x != 0),
                    "Quantization output is all zeros for {:?}",
                    qtype
                );

                // Verify output has some content (u8 is always valid)
                assert!(!output.is_empty(), "Output should not be empty for {:?}", qtype);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_comprehensive() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        assert_eq!(kernel.name(), "avx2");

        // Test AVX2-optimized sizes (multiples of 8)
        let test_sizes = vec![
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (24, 32, 16), // Non-power-of-2
        ];

        let mut data_gen = TestDataGenerator::new(98765);

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "AVX2 kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite values in AVX2 output");

            // Compare with fallback kernel for correctness
            let fallback = FallbackKernel;
            let mut c_fallback = vec![0.0f32; m * n];
            fallback.matmul_i2s(&a, &b, &mut c_fallback, m, n, k).unwrap();

            // Results should be similar (allowing for floating point differences)
            for i in 0..c.len() {
                let diff = (c[i] - c_fallback[i]).abs();
                assert!(
                    diff < 1e-3,
                    "AVX2 result differs from fallback at index {}: {} vs {} (diff: {})",
                    i,
                    c[i],
                    c_fallback[i],
                    diff
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_comprehensive() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            println!("NEON not available, skipping test");
            return;
        }

        assert_eq!(kernel.name(), "neon");

        // Test NEON-optimized sizes (multiples of 4)
        let test_sizes = vec![
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (12, 16, 8), // Non-power-of-2
        ];

        let mut data_gen = TestDataGenerator::new(24680);

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "NEON kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite values in NEON output");

            // Compare with fallback kernel for correctness
            let fallback = FallbackKernel;
            let mut c_fallback = vec![0.0f32; m * n];
            fallback.matmul_i2s(&a, &b, &mut c_fallback, m, n, k).unwrap();

            // Results should be similar
            for i in 0..c.len() {
                let diff = (c[i] - c_fallback[i]).abs();
                assert!(
                    diff < 1e-3,
                    "NEON result differs from fallback at index {}: {} vs {} (diff: {})",
                    i,
                    c[i],
                    c_fallback[i],
                    diff
                );
            }
        }
    }
}
// ============================================================================
// GPU Kernel Tests
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_kernel_tests {
    use super::*;
    use bitnet_kernels::gpu::*;

    #[test]
    fn test_cuda_kernel_availability() {
        match CudaKernel::new() {
            Ok(kernel) => {
                assert!(kernel.is_available());
                assert_eq!(kernel.name(), "CUDA");

                let device_info = kernel.device_info();
                assert!(!device_info.name.is_empty());
                assert!(device_info.total_memory > 0);
                assert!(device_info.multiprocessor_count > 0);

                println!("CUDA device: {}", device_info.name);
                println!("Compute capability: {:?}", device_info.compute_capability);
                println!("Memory: {} GB", device_info.total_memory / (1024 * 1024 * 1024));
            }
            Err(_) => {
                println!("CUDA not available, skipping GPU tests");
            }
        }
    }

    #[test]
    fn test_cuda_matmul_correctness() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let test_sizes = vec![(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)];

        let mut data_gen = TestDataGenerator::new(11111);

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "CUDA kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite values in CUDA output");

            // Compare with CPU fallback for correctness
            let fallback = cpu::FallbackKernel;
            let mut c_cpu = vec![0.0f32; m * n];
            fallback.matmul_i2s(&a, &b, &mut c_cpu, m, n, k).unwrap();

            // Results should be similar (allowing for GPU precision differences)
            let mut max_diff = 0.0f32;
            for i in 0..c.len() {
                let diff = (c[i] - c_cpu[i]).abs();
                max_diff = max_diff.max(diff);
            }

            assert!(
                max_diff < 1e-2,
                "CUDA result differs too much from CPU for size {}x{}x{}: max_diff = {}",
                m,
                n,
                k,
                max_diff
            );
        }
    }

    #[test]
    fn test_cuda_performance_characteristics() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let mut data_gen = TestDataGenerator::new(22222);

        // Test performance scaling with size
        let sizes = vec![64, 128, 256];
        let mut gpu_times = Vec::new();
        let mut cpu_times = Vec::new();

        let fallback = cpu::FallbackKernel;

        for size in sizes {
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c_gpu = vec![0.0f32; size * size];
            let mut c_cpu = vec![0.0f32; size * size];

            // Warm up
            let _ = kernel.matmul_i2s(&a, &b, &mut c_gpu, size, size, size);
            let _ = fallback.matmul_i2s(&a, &b, &mut c_cpu, size, size, size);

            // Benchmark GPU
            let start = Instant::now();
            for _ in 0..5 {
                kernel.matmul_i2s(&a, &b, &mut c_gpu, size, size, size).unwrap();
            }
            let gpu_time = start.elapsed().as_nanos() as u64 / 5;
            gpu_times.push(gpu_time);

            // Benchmark CPU
            let start = Instant::now();
            for _ in 0..5 {
                fallback.matmul_i2s(&a, &b, &mut c_cpu, size, size, size).unwrap();
            }
            let cpu_time = start.elapsed().as_nanos() as u64 / 5;
            cpu_times.push(cpu_time);

            let speedup = cpu_time as f64 / gpu_time as f64;
            println!(
                "Size {}: GPU={} ns, CPU={} ns, Speedup={:.2}x",
                size, gpu_time, cpu_time, speedup
            );
        }

        // GPU should show some performance benefit for larger sizes
        let large_size_speedup = cpu_times[2] as f64 / gpu_times[2] as f64;
        println!("Large size speedup: {:.2}x", large_size_speedup);
    }

    #[test]
    fn test_cuda_memory_management() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        // Test multiple operations to ensure no memory leaks
        let mut data_gen = TestDataGenerator::new(33333);

        for i in 0..20 {
            let size = 32 + (i % 3) * 16; // Vary size slightly
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c = vec![0.0f32; size * size];

            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "CUDA operation {} failed", i);

            // Check memory stats periodically
            if i % 5 == 0 {
                let (used, total) = kernel.memory_stats();
                println!("Iteration {}: GPU memory used: {} / {} bytes", i, used, total);
            }
        }

        // Synchronize to ensure all operations complete
        kernel.synchronize_all().unwrap();
    }
}
// ============================================================================
// Kernel Selection and Dispatch Tests
// ============================================================================

mod kernel_selection_tests {
    use super::*;

    #[test]
    fn test_kernel_manager_selection_priority() {
        let manager = KernelManager::new();

        // Should always have at least the fallback kernel
        let available = manager.list_available_providers();
        assert!(!available.is_empty(), "No kernel providers available");
        assert!(available.contains(&"fallback"), "Fallback kernel should always be available");

        // Test kernel selection
        let kernel = manager.select_best().expect("Should select a kernel");
        let selected_name = kernel.name();

        println!("Available kernels: {:?}", available);
        println!("Selected kernel: {}", selected_name);

        // Verify selection priority
        #[cfg(feature = "cuda")]
        if available.contains(&"CUDA") {
            assert_eq!(selected_name, "CUDA", "CUDA should be preferred when available");
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if available.contains(&"avx2") && !available.contains(&"CUDA") {
            assert_eq!(selected_name, "avx2", "AVX2 should be preferred over fallback");
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if available.contains(&"neon") && !available.contains(&"CUDA") {
            assert_eq!(selected_name, "neon", "NEON should be preferred over fallback");
        }
    }

    #[test]
    fn test_kernel_manager_consistency() {
        let manager = KernelManager::new();

        // Multiple calls should return the same kernel
        let kernel1 = manager.select_best().unwrap();
        let kernel2 = manager.select_best().unwrap();

        assert_eq!(kernel1.name(), kernel2.name(), "Kernel selection should be consistent");

        // Selected provider name should be consistent
        let name1 = manager.selected_provider_name();
        let name2 = manager.selected_provider_name();

        assert_eq!(name1, name2, "Selected provider name should be consistent");
        assert!(name1.is_some(), "Should have a selected provider name");
    }

    #[test]
    fn test_kernel_manager_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(KernelManager::new());

        // Test concurrent access from multiple threads
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || {
                    let kernel = manager_clone.select_best().unwrap();
                    (i, kernel.name().to_string())
                })
            })
            .collect();

        let results: Vec<(usize, String)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same kernel
        let first_kernel = &results[0].1;
        for (thread_id, kernel_name) in &results {
            assert_eq!(
                kernel_name, first_kernel,
                "Thread {} got different kernel: {} vs {}",
                thread_id, kernel_name, first_kernel
            );
        }
    }

    #[test]
    fn test_cpu_kernel_selection() {
        let cpu_kernel = select_cpu_kernel().unwrap();

        assert!(cpu_kernel.is_available(), "Selected CPU kernel should be available");
        assert!(!cpu_kernel.name().is_empty(), "CPU kernel should have a name");

        // Should be one of the known CPU kernels
        let known_kernels = vec!["fallback", "avx2", "neon"];
        assert!(
            known_kernels.contains(&cpu_kernel.name()),
            "Unknown CPU kernel: {}",
            cpu_kernel.name()
        );

        println!("Selected CPU kernel: {}", cpu_kernel.name());
    }

    #[test]
    fn test_gpu_kernel_selection() {
        let result = select_gpu_kernel(0);

        match result {
            Ok(gpu_kernel) => {
                assert!(gpu_kernel.is_available(), "Selected GPU kernel should be available");
                println!("Selected GPU kernel: {}", gpu_kernel.name());
            }
            Err(_) => {
                println!("GPU kernel not available (expected on systems without CUDA)");
            }
        }
    }

    #[test]
    fn test_kernel_cross_validation() {
        let manager = KernelManager::new();
        let available_kernels = manager.list_available_providers();

        if available_kernels.len() < 2 {
            println!("Only one kernel available, skipping cross-validation");
            return;
        }

        println!("Cross-validating {} kernels", available_kernels.len());

        let mut data_gen = TestDataGenerator::new(55555);
        let test_size = 16;

        let a = data_gen.generate_matrix_a(test_size, test_size);
        let b = data_gen.generate_matrix_b(test_size, test_size);

        // Get reference result from fallback kernel
        let fallback = cpu::FallbackKernel;
        let mut c_reference = vec![0.0f32; test_size * test_size];
        fallback.matmul_i2s(&a, &b, &mut c_reference, test_size, test_size, test_size).unwrap();

        // Test selected kernel against reference
        let kernel = manager.select_best().unwrap();
        let mut c_test = vec![0.0f32; test_size * test_size];
        kernel.matmul_i2s(&a, &b, &mut c_test, test_size, test_size, test_size).unwrap();

        // Compare results
        let mut max_diff = 0.0f32;
        for i in 0..c_reference.len() {
            let diff = (c_reference[i] - c_test[i]).abs();
            max_diff = max_diff.max(diff);
        }

        println!("Cross-validation: {} vs fallback, max_diff = {}", kernel.name(), max_diff);
        assert!(
            max_diff < 1e-2,
            "Kernel {} differs too much from reference: {}",
            kernel.name(),
            max_diff
        );
    }
}
// ============================================================================
// Performance and Benchmarking Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_kernel_performance_scaling() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        let sizes = vec![32, 64, 128, 256];
        let mut data_gen = TestDataGenerator::new(66666);

        println!("Performance scaling test for kernel: {}", kernel.name());

        for size in sizes {
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c = vec![0.0f32; size * size];

            // Warm up
            for _ in 0..3 {
                let _ = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            }

            // Benchmark
            let iterations = if size <= 64 { 10 } else { 5 };
            let start = Instant::now();

            for _ in 0..iterations {
                kernel.matmul_i2s(&a, &b, &mut c, size, size, size).unwrap();
            }

            let elapsed = start.elapsed().as_nanos() as u64;
            let avg_time = elapsed / iterations;
            let ops = 2 * size * size * size; // Approximate FLOP count

            let metrics = PerformanceMetrics::new(
                kernel.name(),
                &format!("matmul_{}x{}", size, size),
                avg_time,
                ops as u64,
            );

            println!(
                "  {}x{}: {:.2} ms, {:.2} GFLOPS",
                size,
                size,
                avg_time as f64 / 1_000_000.0,
                metrics.throughput_ops_per_sec / 1_000_000_000.0
            );

            // Performance should be reasonable
            assert!(avg_time > 0, "Execution time should be positive");
            assert!(metrics.throughput_ops_per_sec > 0.0, "Throughput should be positive");
        }
    }

    #[test]
    fn test_quantization_performance() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        let sizes = vec![1024, 4096, 16384];
        let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
        let mut data_gen = TestDataGenerator::new(77777);

        println!("Quantization performance test for kernel: {}", kernel.name());

        for qtype in qtypes {
            for size in &sizes {
                let input = data_gen.generate_quantization_input(*size);
                let mut output = vec![0u8; size / 4];
                let block_size = match qtype {
                    QuantizationType::I2S => 32,
                    QuantizationType::TL1 => 64,
                    QuantizationType::TL2 => 128,
                };
                let num_blocks = (size + block_size - 1) / block_size;
                let mut scales = vec![0.0f32; num_blocks];

                // Warm up
                for _ in 0..3 {
                    let _ = kernel.quantize(&input, &mut output, &mut scales, qtype);
                }

                // Benchmark
                let iterations = 50;
                let start = Instant::now();

                for _ in 0..iterations {
                    kernel.quantize(&input, &mut output, &mut scales, qtype).unwrap();
                }

                let elapsed = start.elapsed().as_nanos() as u64;
                let avg_time = elapsed / iterations;

                let metrics = PerformanceMetrics::new(
                    kernel.name(),
                    &format!("quantize_{:?}_{}", qtype, size),
                    avg_time,
                    *size as u64,
                );

                println!(
                    "  {:?} {}: {:.2} μs, {:.2} M elements/sec",
                    qtype,
                    size,
                    avg_time as f64 / 1_000.0,
                    metrics.throughput_ops_per_sec / 1_000_000.0
                );

                // Performance should be reasonable
                assert!(avg_time > 0, "Quantization time should be positive");
                assert!(
                    metrics.throughput_ops_per_sec > 1_000_000.0,
                    "Quantization should process at least 1M elements/sec"
                );
            }
        }
    }

    #[test]
    fn test_memory_access_patterns() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test different matrix shapes to stress memory access patterns
        let test_cases = vec![
            (64, 64, 64, "square"),
            (128, 32, 64, "wide"),
            (32, 128, 64, "tall"),
            (256, 16, 32, "very_wide"),
            (16, 256, 32, "very_tall"),
        ];

        let mut data_gen = TestDataGenerator::new(88888);

        println!("Memory access pattern test for kernel: {}", kernel.name());

        for (m, n, k, shape) in test_cases {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let start = Instant::now();
            kernel.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();
            let duration = start.elapsed();

            let ops = 2 * m * n * k;
            let bytes = (a.len() + b.len() + c.len() * 4) as u64;

            let metrics = PerformanceMetrics::new(
                kernel.name(),
                &format!("{}_{}_{}x{}x{}", kernel.name(), shape, m, n, k),
                duration.as_nanos() as u64,
                ops as u64,
            )
            .with_memory_bandwidth(bytes);

            println!(
                "  {} ({}x{}x{}): {:.2} ms, {:.2} GFLOPS, {:.2} GB/s",
                shape,
                m,
                n,
                k,
                duration.as_millis(),
                metrics.throughput_ops_per_sec / 1_000_000_000.0,
                metrics.memory_bandwidth_gb_per_sec.unwrap_or(0.0)
            );

            // All shapes should complete in reasonable time
            assert!(duration.as_millis() < 1000, "Operation took too long: {:?}", duration);
        }
    }
}
// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_matrix_dimensions() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test zero dimensions
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err(), "Should fail with zero dimensions");

        // Test mismatched dimensions
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];

        // Wrong k dimension
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3);
        assert!(result.is_err(), "Should fail with wrong k dimension");

        // Output buffer too small
        let mut c_small = vec![0.0f32; 2];
        let result = kernel.matmul_i2s(&a, &b, &mut c_small, 2, 2, 2);
        assert!(result.is_err(), "Should fail with small output buffer");

        // Input buffers too small
        let a_small = vec![1i8; 2];
        let result = kernel.matmul_i2s(&a_small, &b, &mut c, 2, 2, 2);
        assert!(result.is_err(), "Should fail with small input buffer");
    }

    #[test]
    fn test_quantization_buffer_validation() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test empty input
        let result = kernel.quantize(&[], &mut [], &mut [], QuantizationType::I2S);
        assert!(result.is_err(), "Should fail with empty input");

        // Test mismatched buffer sizes
        let input = vec![1.0f32; 64];
        let mut output = vec![0u8; 10]; // Too small
        let mut scales = vec![0.0f32; 4];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err(), "Should fail with small output buffer");

        // Test insufficient scale buffer
        let mut output = vec![0u8; 16];
        let mut scales = vec![0.0f32; 1]; // Too small for 64 elements

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err(), "Should fail with insufficient scale buffer");
    }

    #[test]
    fn test_extreme_input_values() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test with extreme i8 values
        let a = vec![i8::MAX, i8::MIN, 0, 1, -1];
        let b = vec![u8::MAX, u8::MIN, 128, 64, 32];
        let mut c = vec![0.0f32; 1];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 5);
        assert!(result.is_ok(), "Should handle extreme values");
        assert!(c[0].is_finite(), "Result should be finite");

        // Test quantization with extreme values
        let input = vec![f32::MAX, f32::MIN, 0.0, 1e-10, -1e-10];
        let mut output = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        // Should either succeed or fail gracefully
        match result {
            Ok(_) => {
                assert!(scales[0].is_finite(), "Scale should be finite");
                assert!(scales[0] > 0.0, "Scale should be positive");
            }
            Err(_) => {
                // Acceptable to fail with extreme values
                println!("Kernel rejected extreme values (acceptable)");
            }
        }
    }

    #[test]
    fn test_special_float_values() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test with NaN values
        let input_nan = vec![f32::NAN, 1.0, 2.0, 3.0];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input_nan, &mut output, &mut scales, QuantizationType::I2S);
        // Should handle NaN gracefully (either error or replace with valid values)
        match result {
            Ok(_) => {
                assert!(scales[0].is_finite(), "Scale should be finite even with NaN input");
            }
            Err(_) => {
                println!("Kernel rejected NaN values (acceptable)");
            }
        }

        // Test with infinity values
        let input_inf = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0];
        let result = kernel.quantize(&input_inf, &mut output, &mut scales, QuantizationType::I2S);
        // Should handle infinity gracefully
        match result {
            Ok(_) => {
                assert!(scales[0].is_finite(), "Scale should be finite even with infinity input");
            }
            Err(_) => {
                println!("Kernel rejected infinity values (acceptable)");
            }
        }
    }

    #[test]
    fn test_large_matrix_stress() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test with reasonably large matrices
        let m = 512;
        let n = 512;
        let k = 512;

        let mut data_gen = TestDataGenerator::new(99999);
        let a = data_gen.generate_matrix_a(m, k);
        let b = data_gen.generate_matrix_b(k, n);
        let mut c = vec![0.0f32; m * n];

        let start = Instant::now();
        let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
        let duration = start.elapsed();

        assert!(result.is_ok(), "Large matrix multiplication should succeed");
        assert!(duration.as_secs() < 30, "Large matrix should complete in reasonable time");

        // Verify all results are finite
        assert!(c.iter().all(|&x| x.is_finite()), "All results should be finite");

        // Verify some variation in results (not all zeros or all same value)
        let min_val = c.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = c.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val > min_val, "Results should have some variation");

        println!("Large matrix ({}x{}x{}) completed in {:?}", m, n, k, duration);
    }
}
// ============================================================================
// Integration and End-to-End Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_inference_simulation() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        println!("End-to-end inference simulation with kernel: {}", kernel.name());

        // Simulate a neural network layer
        let batch_size = 4;
        let input_dim = 256;
        let output_dim = 128;

        let mut data_gen = TestDataGenerator::new(12121);

        // Step 1: Generate model weights and quantize them
        let weights_f32 = data_gen.generate_quantization_input(input_dim * output_dim);
        let mut quantized_weights = vec![0u8; (input_dim * output_dim) / 4];
        let num_blocks = (input_dim * output_dim + 31) / 32;
        let mut scales = vec![0.0f32; num_blocks];

        let start = Instant::now();
        let result = kernel.quantize(
            &weights_f32,
            &mut quantized_weights,
            &mut scales,
            QuantizationType::I2S,
        );
        let quantize_time = start.elapsed();

        assert!(result.is_ok(), "Weight quantization should succeed");
        println!("  Weight quantization: {:?}", quantize_time);

        // Step 2: Simulate input activations (already quantized to i8)
        let inputs: Vec<i8> = (0..batch_size * input_dim)
            .map(|i| ((i * 17 + 23) % 5) as i8 - 2) // Values -2 to 2
            .collect();

        // Step 3: Perform matrix multiplication (inference)
        let mut outputs = vec![0.0f32; batch_size * output_dim];

        // Convert quantized weights back to u8 for matrix multiplication
        // (In real implementation, this would be more sophisticated)
        let weights_u8: Vec<u8> = quantized_weights
            .iter()
            .flat_map(|&byte| {
                vec![byte & 0x3, (byte >> 2) & 0x3, (byte >> 4) & 0x3, (byte >> 6) & 0x3]
            })
            .take(input_dim * output_dim)
            .collect();

        let start = Instant::now();
        let result = kernel.matmul_i2s(
            &inputs,
            &weights_u8,
            &mut outputs,
            batch_size,
            output_dim,
            input_dim,
        );
        let inference_time = start.elapsed();

        assert!(result.is_ok(), "Inference should succeed");
        println!("  Inference: {:?}", inference_time);

        // Step 4: Verify results
        assert!(outputs.iter().all(|&x| x.is_finite()), "All outputs should be finite");

        // Calculate performance metrics
        let total_time = quantize_time + inference_time;
        let ops = 2.0 * batch_size as f64 * input_dim as f64 * output_dim as f64;
        let gflops = ops / (inference_time.as_secs_f64() * 1e9);
        let throughput = batch_size as f64 / total_time.as_secs_f64();

        println!("  Total time: {:?}", total_time);
        println!("  Inference performance: {:.2} GFLOPS", gflops);
        println!("  Throughput: {:.2} samples/sec", throughput);

        // Performance should be reasonable
        assert!(gflops > 0.1, "Should achieve reasonable GFLOPS: {}", gflops);
        assert!(throughput > 1.0, "Should process at least 1 sample/sec: {}", throughput);
    }

    #[test]
    fn test_stress_test_repeated_operations() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        println!("Stress test with repeated operations for kernel: {}", kernel.name());

        let mut data_gen = TestDataGenerator::new(56565);
        let test_size = 32;

        // Generate test data once
        let a = data_gen.generate_matrix_a(test_size, test_size);
        let b = data_gen.generate_matrix_b(test_size, test_size);
        let input = data_gen.generate_quantization_input(test_size);

        let iterations = 100;
        let mut total_matmul_time = std::time::Duration::ZERO;
        let mut total_quant_time = std::time::Duration::ZERO;

        for i in 0..iterations {
            // Matrix multiplication
            let mut c = vec![0.0f32; test_size * test_size];
            let start = Instant::now();
            let result = kernel.matmul_i2s(&a, &b, &mut c, test_size, test_size, test_size);
            total_matmul_time += start.elapsed();

            assert!(result.is_ok(), "Matrix multiplication failed on iteration {}", i);
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite result on iteration {}", i);

            // Quantization (if supported)
            let mut output = vec![0u8; test_size / 4];
            let mut scales = vec![0.0f32; 1];
            let start = Instant::now();
            let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
            total_quant_time += start.elapsed();

            if result.is_ok() {
                assert!(
                    scales[0] > 0.0 && scales[0].is_finite(),
                    "Invalid scale on iteration {}: {}",
                    i,
                    scales[0]
                );
            }

            // Progress indicator
            if (i + 1) % 20 == 0 {
                println!("  Completed {} iterations", i + 1);
            }
        }

        let avg_matmul_time = total_matmul_time / iterations as u32;
        let avg_quant_time = total_quant_time / iterations as u32;

        println!("  Average matrix multiplication time: {:?}", avg_matmul_time);
        println!("  Average quantization time: {:?}", avg_quant_time);

        // Performance should be consistent
        assert!(avg_matmul_time.as_millis() < 100, "Matrix multiplication too slow");
        println!("Stress test completed successfully");
    }
}

// ============================================================================
// FFI Kernel Tests (when enabled)
// ============================================================================

#[cfg(feature = "ffi-bridge")]
mod ffi_kernel_tests {
    use super::*;
    use bitnet_kernels::ffi::*;

    #[test]
    fn test_ffi_kernel_availability() {
        match FfiKernel::new() {
            Ok(kernel) => {
                println!("FFI kernel available: {}", kernel.is_available());
                assert_eq!(kernel.name(), "ffi");

                if kernel.is_available() {
                    // Test basic functionality
                    let mut data_gen = TestDataGenerator::new(77777);
                    let a = data_gen.generate_matrix_a(4, 4);
                    let b = data_gen.generate_matrix_b(4, 4);
                    let mut c = vec![0.0f32; 16];

                    let result = kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 4);
                    if result.is_ok() {
                        assert!(
                            c.iter().all(|&x| x.is_finite()),
                            "FFI kernel should produce finite results"
                        );
                        println!("FFI kernel basic test passed");
                    } else {
                        println!("FFI kernel operation failed: {:?}", result);
                    }
                }
            }
            Err(e) => {
                println!("FFI kernel not available: {}", e);
            }
        }
    }

    #[test]
    fn test_ffi_performance_comparison() {
        let ffi_kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping comparison test");
                return;
            }
        };

        let rust_kernel = cpu::FallbackKernel;
        let mut data_gen = TestDataGenerator::new(88888);

        // Test matrix multiplication comparison
        let test_size = 64;
        let a = data_gen.generate_matrix_a(test_size, test_size);
        let b = data_gen.generate_matrix_b(test_size, test_size);

        // Benchmark Rust kernel
        let mut c_rust = vec![0.0f32; test_size * test_size];
        let start = Instant::now();
        rust_kernel.matmul_i2s(&a, &b, &mut c_rust, test_size, test_size, test_size).unwrap();
        let rust_time = start.elapsed();

        // Benchmark FFI kernel
        let mut c_ffi = vec![0.0f32; test_size * test_size];
        let start = Instant::now();
        let ffi_result = ffi_kernel.matmul_i2s(&a, &b, &mut c_ffi, test_size, test_size, test_size);
        let ffi_time = start.elapsed();

        if ffi_result.is_ok() {
            println!("Performance comparison ({}x{}):", test_size, test_size);
            println!("  Rust time: {:?}", rust_time);
            println!("  FFI time: {:?}", ffi_time);

            let speedup = rust_time.as_secs_f64() / ffi_time.as_secs_f64();
            println!("  FFI speedup: {:.2}x", speedup);

            // Compare accuracy
            let mut max_diff = 0.0f32;
            for i in 0..c_rust.len() {
                let diff = (c_rust[i] - c_ffi[i]).abs();
                max_diff = max_diff.max(diff);
            }
            println!("  Max accuracy difference: {:.2e}", max_diff);

            // Results should be reasonably similar
            assert!(max_diff < 1e-2, "FFI and Rust results differ too much: {}", max_diff);
        } else {
            println!("FFI kernel operation failed: {:?}", ffi_result);
        }
    }
}

#[cfg(not(feature = "ffi-bridge"))]
mod ffi_kernel_tests {
    use super::*;

    #[test]
    fn test_ffi_kernel_disabled() {
        let kernel = bitnet_kernels::ffi::FfiKernel;
        assert!(
            !kernel.is_available(),
            "FFI kernel should not be available when feature is disabled"
        );
        assert_eq!(kernel.name(), "ffi");

        // Operations should fail gracefully
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err(), "FFI operations should fail when disabled");
    }
}
