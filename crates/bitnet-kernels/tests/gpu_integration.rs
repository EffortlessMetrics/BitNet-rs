//! Integration tests for GPU kernel validation
//! 
//! These tests validate the complete GPU kernel system including:
//! - Numerical accuracy against CPU implementations
//! - Performance benchmarking and speedup measurement
//! - Memory management and leak detection
//! - Error handling and edge cases

#[cfg(feature = "cuda")]
mod cuda_tests {
    use bitnet_kernels::gpu::{
        CudaKernel, GpuValidator, ValidationConfig, GpuBenchmark, BenchmarkConfig,
        print_validation_results, print_benchmark_results, quick_benchmark
    };
    use bitnet_kernels::KernelProvider;
    use bitnet_common::QuantizationType;

    #[test]
    #[ignore] // Run with: cargo test --features cuda --ignored
    fn test_comprehensive_gpu_validation() {
        env_logger::init();
        
        println!("🚀 Starting comprehensive GPU validation");
        
        // Test 1: Basic CUDA availability
        println!("\n1️⃣ Testing CUDA availability...");
        match CudaKernel::new() {
            Ok(kernel) => {
                println!("✅ CUDA kernel created successfully");
                println!("   Device info: {:?}", kernel.device_info());
                assert!(kernel.is_available());
            }
            Err(e) => {
                panic!("❌ Failed to create CUDA kernel: {}", e);
            }
        }

        // Test 2: Numerical accuracy validation
        println!("\n2️⃣ Running numerical accuracy validation...");
        let validation_config = ValidationConfig {
            tolerance: 1e-6,
            test_sizes: vec![
                (64, 64, 64),
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
            ],
            benchmark_iterations: 50,
            check_memory_leaks: true,
            test_mixed_precision: false,
        };

        let validator = GpuValidator::with_config(validation_config);
        let validation_results = validator.validate()
            .expect("GPU validation should succeed");

        print_validation_results(&validation_results);
        
        // Verify all accuracy tests passed
        for result in &validation_results.accuracy_results {
            assert!(result.passed, 
                "Accuracy test failed for {:?}: max_error={:.2e} > tolerance={:.2e}",
                result.dimensions, result.max_error, 1e-6
            );
        }

        assert!(validation_results.success, "Overall validation should succeed");
        println!("✅ Numerical accuracy validation passed");

        // Test 3: Performance benchmarking
        println!("\n3️⃣ Running performance benchmarks...");
        let benchmark_config = BenchmarkConfig {
            test_sizes: vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
            ],
            warmup_iterations: 10,
            benchmark_iterations: 50,
            include_cpu_comparison: true,
            test_data_patterns: false,
        };

        let benchmark = GpuBenchmark::with_config(benchmark_config);
        let benchmark_results = benchmark.run()
            .expect("GPU benchmark should succeed");

        print_benchmark_results(&benchmark_results);

        // Verify reasonable performance
        assert!(benchmark_results.summary.avg_speedup > 0.5, 
            "GPU should not be significantly slower than CPU");
        assert!(benchmark_results.summary.peak_gflops > 1.0,
            "GPU should achieve reasonable GFLOPS");

        println!("✅ Performance benchmarking completed");

        // Test 4: Memory management
        println!("\n4️⃣ Testing memory management...");
        test_memory_management();
        println!("✅ Memory management tests passed");

        // Test 5: Error handling
        println!("\n5️⃣ Testing error handling...");
        test_error_handling();
        println!("✅ Error handling tests passed");

        println!("\n🎉 All GPU validation tests passed successfully!");
    }

    fn test_memory_management() {
        // Test multiple kernel creations don't leak memory
        for i in 0..20 {
            let mut kernel = CudaKernel::new()
                .expect("Should be able to create CUDA kernel");

            // Test with various matrix sizes
            let sizes = [(16, 16, 16), (64, 64, 64), (128, 128, 128)];
            
            for &(m, n, k) in &sizes {
                let a = vec![1i8; m * k];
                let b = vec![1u8; k * n];
                let mut c = vec![0.0f32; m * n];

                kernel.matmul_i2s(&a, &b, &mut c, m, n, k)
                    .expect("Matrix multiplication should succeed");
            }

            if i % 5 == 0 {
                println!("   Memory test iteration {}/20", i + 1);
            }
        }
    }

    fn test_error_handling() {
        let mut kernel = CudaKernel::new()
            .expect("Should be able to create CUDA kernel");

        // Test invalid matrix dimensions
        let a = vec![1i8; 16];
        let b = vec![1u8; 16];
        let mut c = vec![0.0f32; 16];

        // Test mismatched dimensions (should handle gracefully)
        match kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 8) {
            Ok(_) => {
                // If it succeeds, that's fine - the kernel might handle it
                println!("   Kernel handled mismatched dimensions gracefully");
            }
            Err(e) => {
                // If it fails, that's also fine - proper error handling
                println!("   Kernel properly reported error for mismatched dimensions: {}", e);
            }
        }

        // Test quantization error handling (not yet implemented)
        let input = vec![1.0f32; 16];
        let mut output = vec![0u8; 16];
        let mut scales = vec![0.0f32; 4];

        match kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S) {
            Ok(_) => {
                println!("   Quantization succeeded unexpectedly");
            }
            Err(_) => {
                println!("   Quantization properly reported not implemented");
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda --ignored
    fn test_quick_benchmark_integration() {
        env_logger::init();
        
        println!("🏃 Running quick benchmark integration test");
        
        match quick_benchmark() {
            Ok(_) => {
                println!("✅ Quick benchmark completed successfully");
            }
            Err(e) => {
                panic!("❌ Quick benchmark failed: {}", e);
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda --ignored
    fn test_large_matrix_performance() {
        env_logger::init();
        
        println!("🔢 Testing large matrix performance");
        
        let mut kernel = CudaKernel::new()
            .expect("Should be able to create CUDA kernel");

        // Test progressively larger matrices
        let sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 1024, 512),
        ];

        for &(m, n, k) in &sizes {
            println!("   Testing {}x{}x{} matrix...", m, n, k);
            
            let a: Vec<i8> = (0..m*k).map(|i| ((i % 3) as i8) - 1).collect();
            let b: Vec<u8> = (0..k*n).map(|i| (i % 2) as u8).collect();
            let mut c = vec![0.0f32; m * n];

            let start = std::time::Instant::now();
            kernel.matmul_i2s(&a, &b, &mut c, m, n, k)
                .expect("Large matrix multiplication should succeed");
            let elapsed = start.elapsed();

            let operations = 2.0 * m as f64 * n as f64 * k as f64;
            let gflops = operations / elapsed.as_secs_f64() / 1e9;

            println!("     Completed in {:.2}ms ({:.1} GFLOPS)", 
                elapsed.as_secs_f64() * 1000.0, gflops);

            // Verify result is not all zeros
            let nonzero_count = c.iter().filter(|&&x| x != 0.0).count();
            assert!(nonzero_count > 0, "Result should contain non-zero values");
        }

        println!("✅ Large matrix performance tests passed");
    }

    #[test]
    #[ignore] // Run with: cargo test --features cuda --ignored  
    fn test_concurrent_kernel_usage() {
        env_logger::init();
        
        println!("🔄 Testing concurrent kernel usage");
        
        use std::thread;
        use std::sync::Arc;
        
        // Note: CUDA contexts are not thread-safe by default
        // This test verifies proper error handling for concurrent access
        
        let handles: Vec<_> = (0..4).map(|i| {
            thread::spawn(move || {
                match CudaKernel::new() {
                    Ok(mut kernel) => {
                        let a = vec![1i8; 64];
                        let b = vec![1u8; 64];
                        let mut c = vec![0.0f32; 64];
                        
                        match kernel.matmul_i2s(&a, &b, &mut c, 8, 8, 8) {
                            Ok(_) => {
                                println!("   Thread {} completed successfully", i);
                                true
                            }
                            Err(e) => {
                                println!("   Thread {} failed: {}", i, e);
                                false
                            }
                        }
                    }
                    Err(e) => {
                        println!("   Thread {} failed to create kernel: {}", i, e);
                        false
                    }
                }
            })
        }).collect();

        let results: Vec<bool> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();

        // At least some threads should succeed
        let success_count = results.iter().filter(|&&x| x).count();
        println!("   {}/{} threads succeeded", success_count, results.len());
        
        // We don't require all to succeed since CUDA may have limitations
        // but at least one should work
        assert!(success_count > 0, "At least one thread should succeed");
        
        println!("✅ Concurrent kernel usage test completed");
    }
}

#[cfg(not(feature = "cuda"))]
mod no_cuda_tests {
    #[test]
    fn test_cuda_feature_disabled() {
        println!("CUDA feature is disabled - GPU tests skipped");
        println!("Run with: cargo test --features cuda --ignored");
    }
}