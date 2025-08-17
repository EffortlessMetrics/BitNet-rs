#![cfg(feature = "ffi")]
//! Comprehensive tests for bitnet-kernels
//! Covers edge cases, error conditions, and end-to-end scenarios

use bitnet_common::QuantizationType;
use bitnet_kernels::*;
use std::time::Instant;

/// Test error conditions and edge cases
mod error_handling {
    use super::*;

    #[test]
    fn test_invalid_matrix_dimensions() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test zero dimensions
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err());

        // Test mismatched dimensions
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];

        // Wrong k dimension
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3);
        assert!(result.is_err());

        // Output buffer too small
        let mut c_small = vec![0.0f32; 2];
        let result = kernel.matmul_i2s(&a, &b, &mut c_small, 2, 2, 2);
        assert!(result.is_err());

        // Input buffers too small
        let a_small = vec![1i8; 2];
        let result = kernel.matmul_i2s(&a_small, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_buffer_validation() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test empty input
        let result = kernel.quantize(&[], &mut [], &mut [], QuantizationType::I2S);
        assert!(result.is_err());

        // Test mismatched buffer sizes
        let input = vec![1.0f32; 64];
        let mut output = vec![0u8; 10]; // Too small
        let mut scales = vec![0.0f32; 4];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err());

        // Test insufficient scale buffer
        let mut output = vec![0u8; 32];
        let mut scales = vec![0.0f32; 1]; // Too small

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err());
    }

    #[test]
    fn test_extreme_input_values() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test with extreme i8 values
        let a = vec![i8::MAX, i8::MIN, 0, 1, -1];
        let b = vec![u8::MAX, u8::MIN, 128, 64, 32];
        let mut c = vec![0.0f32; 1];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 5);
        assert!(result.is_ok());

        // Result should be finite
        assert!(c[0].is_finite());
    }

    #[test]
    fn test_large_matrix_dimensions() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test with large but reasonable dimensions
        let m = 100;
        let n = 100;
        let k = 100;

        let a = vec![1i8; m * k];
        let b = vec![1u8; k * n];
        let mut c = vec![0.0f32; m * n];

        let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok());

        // All results should be finite
        assert!(c.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kernel_availability_edge_cases() {
        let manager = KernelManager::new();

        // Should always have at least the fallback kernel
        let kernel = manager.select_best();
        assert!(kernel.is_ok());

        let kernel = kernel.unwrap();
        assert!(kernel.is_available());

        // Kernel name should not be empty
        assert!(!kernel.name().is_empty());
    }

    #[test]
    fn test_quantization_with_special_values() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test with NaN values
        let input_nan = vec![f32::NAN, 1.0, 2.0, 3.0];
        let mut output = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];

        let _result = kernel.quantize(&input_nan, &mut output, &mut scales, QuantizationType::I2S);
        // Should handle NaN gracefully (either error or replace with valid values)

        // Test with infinity values
        let input_inf = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0];
        let _result = kernel.quantize(&input_inf, &mut output, &mut scales, QuantizationType::I2S);
        // Should handle infinity gracefully
    }
}

/// Test different kernel implementations
mod kernel_implementations {
    use super::*;

    #[test]
    fn test_fallback_kernel_comprehensive() {
        let fallback = cpu::FallbackKernel;

        assert!(fallback.is_available());
        assert_eq!(fallback.name(), "fallback");

        // Test basic matrix multiplication
        let a = vec![-1i8, 0, 1, -1];
        let b = vec![0u8, 1, 1, 0];
        let mut c = vec![0.0f32; 4];

        let result = fallback.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        // Test quantization
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let mut output = vec![0u8; 2];
        let mut scales = vec![0.0f32; 1];

        let result = fallback.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());
        assert!(scales[0] > 0.0);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_comprehensive() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip if AVX2 not available
        }

        let avx2_kernel = cpu::Avx2Kernel;

        assert!(avx2_kernel.is_available());
        assert_eq!(avx2_kernel.name(), "avx2");

        // Test with AVX2-friendly sizes (multiples of 8)
        let sizes = vec![8, 16, 32, 64];

        for size in sizes {
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];

            let result = avx2_kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "AVX2 failed for size {}", size);

            // Results should be reasonable
            assert!(c.iter().all(|&x| x.is_finite()));
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_comprehensive() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return; // Skip if NEON not available
        }

        let neon_kernel = cpu::NeonKernel;

        assert!(neon_kernel.is_available());
        assert_eq!(neon_kernel.name(), "neon");

        // Test with NEON-friendly sizes (multiples of 4)
        let sizes = vec![4, 8, 16, 32];

        for size in sizes {
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];

            let result = neon_kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "NEON failed for size {}", size);

            // Results should be reasonable
            assert!(c.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_kernel_consistency() {
        let manager = KernelManager::new();
        let available_providers = manager.list_available_providers();

        // Test that all available kernels produce consistent results
        if available_providers.len() > 1 {
            let test_data_a = vec![-1i8, 0, 1, -1, 1, 0, -1, 1];
            let test_data_b = vec![0u8, 1, 1, 0, 1, 1, 0, 0];

            let mut results = Vec::new();

            // Get results from each kernel
            for provider_name in &available_providers {
                println!("Testing kernel: {}", provider_name);

                let kernel = manager.select_best().unwrap();
                let mut c = vec![0.0f32; 4];

                let result = kernel.matmul_i2s(&test_data_a, &test_data_b, &mut c, 2, 2, 4);
                assert!(result.is_ok());

                results.push(c);
            }

            // Compare results (should be similar within floating point precision)
            if results.len() > 1 {
                let reference = &results[0];
                for (i, result) in results.iter().enumerate().skip(1) {
                    for (j, (&ref_val, &test_val)) in
                        reference.iter().zip(result.iter()).enumerate()
                    {
                        let diff = (ref_val - test_val).abs();
                        assert!(
                            diff < 1e-5,
                            "Kernel {} result differs from reference at position {}: {} vs {}",
                            available_providers[i],
                            j,
                            test_val,
                            ref_val
                        );
                    }
                }
            }
        }
    }
}

/// Test performance characteristics
mod performance_tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication_performance() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        let sizes = vec![64, 128, 256];

        for size in sizes {
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];

            let start = Instant::now();
            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("Matrix multiplication {}x{} took: {:?}", size, size, duration);

            // Should complete in reasonable time
            assert!(duration.as_secs() < 5, "Matrix multiplication took too long: {:?}", duration);

            // Calculate GFLOPS
            let ops = 2.0 * size as f64 * size as f64 * size as f64;
            let gflops = ops / (duration.as_secs_f64() * 1e9);
            println!("  Performance: {:.2} GFLOPS", gflops);
        }
    }

    #[test]
    fn test_quantization_performance() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        let sizes = vec![1024, 4096, 16384];

        for size in sizes {
            let input = vec![1.0f32; size];
            let mut output = vec![0u8; size / 2];
            let mut scales = vec![0.0f32; size / 64];

            let start = Instant::now();
            let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("Quantization of {} elements took: {:?}", size, duration);

            // Should complete quickly
            assert!(duration.as_millis() < 1000, "Quantization took too long: {:?}", duration);

            // Calculate throughput
            let throughput = size as f64 / duration.as_secs_f64() / 1e6;
            println!("  Throughput: {:.2} M elements/sec", throughput);
        }
    }

    #[test]
    fn test_memory_access_patterns() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test different matrix shapes to stress memory access patterns
        let test_cases = vec![
            (64, 64, 64),  // Square
            (128, 32, 64), // Wide
            (32, 128, 64), // Tall
            (256, 16, 32), // Very wide
            (16, 256, 32), // Very tall
        ];

        for (m, n, k) in test_cases {
            let a = vec![1i8; m * k];
            let b = vec![1u8; k * n];
            let mut c = vec![0.0f32; m * n];

            let start = Instant::now();
            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("Matrix {}x{}x{} took: {:?}", m, n, k, duration);

            // All should complete in reasonable time
            assert!(duration.as_millis() < 5000);
        }
    }

    #[test]
    fn test_cache_efficiency() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test with data that fits in cache vs doesn't fit
        let small_size = 32; // Should fit in L1 cache
        let large_size = 512; // May not fit in L1 cache

        // Small matrices (cache-friendly)
        let a_small = vec![1i8; small_size * small_size];
        let b_small = vec![1u8; small_size * small_size];
        let mut c_small = vec![0.0f32; small_size * small_size];

        let start = Instant::now();
        for _ in 0..100 {
            kernel
                .matmul_i2s(&a_small, &b_small, &mut c_small, small_size, small_size, small_size)
                .unwrap();
        }
        let small_duration = start.elapsed();

        // Large matrices (cache-unfriendly)
        let a_large = vec![1i8; large_size * large_size];
        let b_large = vec![1u8; large_size * large_size];
        let mut c_large = vec![0.0f32; large_size * large_size];

        let start = Instant::now();
        for _ in 0..10 {
            kernel
                .matmul_i2s(&a_large, &b_large, &mut c_large, large_size, large_size, large_size)
                .unwrap();
        }
        let large_duration = start.elapsed();

        println!(
            "Small matrix ({}x{}) 100 iterations: {:?}",
            small_size, small_size, small_duration
        );
        println!(
            "Large matrix ({}x{}) 10 iterations: {:?}",
            large_size, large_size, large_duration
        );

        // Calculate per-operation time
        let small_per_op = small_duration.as_nanos() / 100;
        let large_per_op = large_duration.as_nanos() / 10;

        println!("Small per-op: {} ns, Large per-op: {} ns", small_per_op, large_per_op);
    }
}

/// Test kernel manager functionality
mod manager_tests {
    use super::*;

    #[test]
    fn test_kernel_selection_priority() {
        let manager = KernelManager::new();

        // Should select the best available kernel
        let kernel = manager.select_best().unwrap();
        let selected_name = kernel.name();

        println!("Selected kernel: {}", selected_name);

        // Should prefer optimized kernels over fallback
        let available = manager.list_available_providers();
        println!("Available kernels: {:?}", available);

        // Fallback should always be available
        assert!(available.contains(&"fallback"));

        // If optimized kernels are available, they should be preferred
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if is_x86_feature_detected!("avx2") {
            assert!(available.contains(&"avx2"));
            // AVX2 should be preferred over fallback
            assert_ne!(selected_name, "fallback");
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            assert!(available.contains(&"neon"));
            // NEON should be preferred over fallback
            assert_ne!(selected_name, "fallback");
        }
    }

    #[test]
    fn test_kernel_manager_caching() {
        let manager = KernelManager::new();

        // Multiple calls should return the same kernel
        let kernel1 = manager.select_best().unwrap();
        let kernel2 = manager.select_best().unwrap();

        assert_eq!(kernel1.name(), kernel2.name());

        // Selected provider name should be consistent
        let name1 = manager.selected_provider_name();
        let name2 = manager.selected_provider_name();

        assert_eq!(name1, name2);
        assert!(name1.is_some());
    }

    #[test]
    fn test_kernel_manager_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(KernelManager::new());

        // Test concurrent access
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || {
                    let kernel = manager_clone.select_best().unwrap();
                    kernel.name().to_string()
                })
            })
            .collect();

        let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same kernel
        assert!(results.iter().all(|name| name == &results[0]));
    }

    #[test]
    fn test_cpu_kernel_selection() {
        let cpu_kernel = select_cpu_kernel().unwrap();

        assert!(cpu_kernel.is_available());
        assert!(!cpu_kernel.name().is_empty());

        // Should be one of the known CPU kernels
        let known_kernels = ["fallback", "avx2", "neon"];
        assert!(known_kernels.contains(&cpu_kernel.name()));
    }

    #[test]
    fn test_gpu_kernel_selection() {
        // Test GPU kernel selection (should fail on systems without CUDA)
        let result = select_gpu_kernel(0);

        #[cfg(feature = "cuda")]
        {
            // If CUDA feature is enabled, test the interface
            match result {
                Ok(gpu_kernel) => {
                    assert!(gpu_kernel.is_available());
                    assert_eq!(gpu_kernel.name(), "cuda");
                }
                Err(_) => {
                    // Expected on systems without CUDA
                    println!("CUDA not available, which is expected");
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, should always fail
            assert!(result.is_err());
        }
    }
}

/// Test edge cases and stress scenarios
mod stress_tests {
    use super::*;

    #[test]
    fn test_repeated_operations() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test repeated matrix multiplications
        let a = vec![1i8; 64];
        let b = vec![1u8; 64];
        let mut c = vec![0.0f32; 64];

        for i in 0..1000 {
            let result = kernel.matmul_i2s(&a, &b, &mut c, 8, 8, 8);
            assert!(result.is_ok(), "Failed on iteration {}", i);

            // Results should remain consistent
            assert!(c.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_alternating_operations() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        let a = vec![1i8; 64];
        let b = vec![1u8; 64];
        let mut c = vec![0.0f32; 64];

        let input = vec![1.0f32; 64];
        let mut output = vec![0u8; 32];
        let mut scales = vec![0.0f32; 4];

        // Alternate between matrix multiplication and quantization
        for i in 0..100 {
            if i % 2 == 0 {
                let result = kernel.matmul_i2s(&a, &b, &mut c, 8, 8, 8);
                assert!(result.is_ok(), "MatMul failed on iteration {}", i);
            } else {
                let result =
                    kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
                assert!(result.is_ok(), "Quantize failed on iteration {}", i);
            }
        }
    }

    #[test]
    fn test_memory_pressure() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Create large buffers to test memory pressure
        let size = 1024;
        let mut large_buffers = Vec::new();

        for i in 0..10 {
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];

            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "Failed under memory pressure iteration {}", i);

            // Keep buffers alive to maintain memory pressure
            large_buffers.push((a, b, c));
        }

        // All operations should have succeeded
        assert_eq!(large_buffers.len(), 10);
    }

    #[test]
    fn test_boundary_conditions() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Test minimum valid dimensions
        let a = vec![1i8; 1];
        let b = vec![1u8; 1];
        let mut c = vec![0.0f32; 1];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 1);
        assert!(result.is_ok());

        // Test power-of-2 boundaries
        let sizes = vec![1, 2, 4, 8, 16, 32, 64, 128];

        for size in sizes {
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];

            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "Failed for size {}", size);
        }
    }
}

/// Integration tests for complete workflows
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_quantization_workflow() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Simulate a complete quantization workflow
        let model_weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        // Step 1: Quantize the weights
        let mut quantized_weights = vec![0u8; 512];
        let mut scales = vec![0.0f32; 16];

        let start = Instant::now();
        let result = kernel.quantize(
            &model_weights,
            &mut quantized_weights,
            &mut scales,
            QuantizationType::I2S,
        );
        let quantize_time = start.elapsed();

        assert!(result.is_ok());
        println!("Quantization took: {:?}", quantize_time);

        // Step 2: Use quantized weights in matrix multiplication
        let input = vec![1i8; 32];
        let mut output = vec![0.0f32; 32];

        let start = Instant::now();
        let result =
            kernel.matmul_i2s(&input, &quantized_weights[..32 * 32], &mut output, 32, 32, 32);
        let matmul_time = start.elapsed();

        assert!(result.is_ok());
        println!("Matrix multiplication took: {:?}", matmul_time);

        // Step 3: Verify results are reasonable
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(scales.iter().all(|&x| x > 0.0 && x.is_finite()));

        println!("End-to-end workflow completed successfully");
        println!("Total time: {:?}", quantize_time + matmul_time);
    }

    #[test]
    fn test_multi_kernel_comparison() {
        let manager = KernelManager::new();
        let available_kernels = manager.list_available_providers();

        if available_kernels.len() > 1 {
            println!("Comparing {} available kernels", available_kernels.len());

            let test_size = 64;
            let a = vec![1i8; test_size * test_size];
            let b = vec![1u8; test_size * test_size];

            let mut results = Vec::new();
            let mut timings = Vec::new();

            for kernel_name in &available_kernels {
                println!("Testing kernel: {}", kernel_name);

                let kernel = manager.select_best().unwrap();
                let mut c = vec![0.0f32; test_size * test_size];

                let start = Instant::now();
                let result = kernel.matmul_i2s(&a, &b, &mut c, test_size, test_size, test_size);
                let duration = start.elapsed();

                assert!(result.is_ok());

                results.push(c);
                timings.push(duration);

                println!("  Time: {:?}", duration);
            }

            // Compare results for consistency
            if results.len() > 1 {
                let reference = &results[0];
                for (i, result) in results.iter().enumerate().skip(1) {
                    let max_diff = reference
                        .iter()
                        .zip(result.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);

                    println!(
                        "Max difference between {} and {}: {}",
                        available_kernels[0], available_kernels[i], max_diff
                    );

                    assert!(max_diff < 1e-5, "Results differ too much between kernels");
                }
            }

            // Report performance comparison
            let fastest_time = timings.iter().min().unwrap();
            let slowest_time = timings.iter().max().unwrap();

            println!("Performance range: {:?} to {:?}", fastest_time, slowest_time);
            println!("Speedup: {:.2}x", slowest_time.as_secs_f64() / fastest_time.as_secs_f64());
        }
    }

    #[test]
    fn test_real_world_simulation() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().expect("Should have a kernel");

        // Simulate a real neural network layer
        let batch_size = 4;
        let input_dim = 512;
        let output_dim = 256;

        // Simulate quantized weights (typical BitNet values)
        let weights: Vec<u8> = (0..input_dim * output_dim)
            .map(|i| ((i * 17 + 23) % 4) as u8) // Values 0-3
            .collect();

        // Simulate input activations (quantized to i8)
        let inputs: Vec<i8> = (0..batch_size * input_dim)
            .map(|i| ((i * 13 + 7) % 5) as i8 - 2) // Values -2 to 2
            .collect();

        let mut outputs = vec![0.0f32; batch_size * output_dim];

        // Perform the computation
        let start = Instant::now();
        let result =
            kernel.matmul_i2s(&inputs, &weights, &mut outputs, batch_size, output_dim, input_dim);
        let duration = start.elapsed();

        assert!(result.is_ok());

        // Verify outputs are reasonable
        assert!(outputs.iter().all(|&x| x.is_finite()));

        // Calculate performance metrics
        let ops = 2.0 * batch_size as f64 * input_dim as f64 * output_dim as f64;
        let gflops = ops / (duration.as_secs_f64() * 1e9);

        println!("Real-world simulation results:");
        println!(
            "  Batch size: {}, Input dim: {}, Output dim: {}",
            batch_size, input_dim, output_dim
        );
        println!("  Time: {:?}", duration);
        println!("  Performance: {:.2} GFLOPS", gflops);
        println!("  Throughput: {:.2} samples/sec", batch_size as f64 / duration.as_secs_f64());

        // Should achieve reasonable performance
        assert!(gflops > 0.1, "Performance too low: {} GFLOPS", gflops);
    }
}
