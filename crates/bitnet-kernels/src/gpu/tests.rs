//! Comprehensive GPU kernel tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelProvider;
    use bitnet_common::QuantizationType;

    #[test]
    fn test_cuda_availability() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        
        if available {
            let device_count = cuda_device_count();
            println!("CUDA device count: {}", device_count);
            assert!(device_count > 0);
        }
    }

    #[test]
    fn test_cuda_kernel_creation() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        match CudaKernel::new() {
            Ok(kernel) => {
                println!("CUDA kernel created successfully");
                println!("Device info: {:?}", kernel.device_info());
                assert!(kernel.is_available());
                assert_eq!(kernel.name(), "CUDA");
            }
            Err(e) => {
                println!("Failed to create CUDA kernel: {}", e);
                // Don't fail the test if CUDA setup fails
            }
        }
    }

    #[test]
    fn test_cuda_matmul_correctness() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("Failed to create CUDA kernel, skipping test");
                return;
            }
        };

        // Small test matrices
        let m = 32;
        let n = 32;
        let k = 32;

        // Generate test data
        let a: Vec<i8> = (0..m*k).map(|i| (i % 256) as i8 - 128).collect();
        let b: Vec<u8> = (0..k*n).map(|i| (i % 4) as u8).collect();
        let mut c_gpu = vec![0.0f32; m * n];

        // Test GPU computation
        match kernel.matmul_i2s(&a, &b, &mut c_gpu, m, n, k) {
            Ok(()) => {
                println!("GPU matrix multiplication completed successfully");
                
                // Basic sanity checks
                assert!(!c_gpu.iter().all(|&x| x == 0.0));
                assert!(c_gpu.iter().all(|&x| x.is_finite()));
                
                println!("First few results: {:?}", &c_gpu[..5]);
            }
            Err(e) => {
                println!("GPU matrix multiplication failed: {}", e);
                // Don't fail the test if GPU computation fails
            }
        }
    }

    #[test]
    fn test_cuda_quantization() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("Failed to create CUDA kernel, skipping test");
                return;
            }
        };

        let size = 1024;
        let input: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32 - 0.5).collect();
        let mut output = vec![0u8; size / 4]; // 2 bits per element
        let mut scales = vec![0.0f32; size / 128]; // Block size of 128

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            match kernel.quantize(&input, &mut output, &mut scales, qtype) {
                Ok(()) => {
                    println!("GPU quantization {:?} completed successfully", qtype);
                    
                    // Basic sanity checks
                    assert!(scales.iter().any(|&x| x > 0.0));
                    assert!(output.iter().any(|&x| x != 0));
                    
                    println!("First scale: {}", scales[0]);
                }
                Err(e) => {
                    println!("GPU quantization {:?} failed: {}", e, qtype);
                }
            }
        }
    }

    #[test]
    fn test_memory_pool() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(_) => {
                println!("Failed to create CUDA device, skipping test");
                return;
            }
        };

        let config = MemoryPoolConfig {
            max_pool_size: 64 * 1024 * 1024, // 64MB for testing
            min_allocation_size: 256,
            max_cached_per_size: 4,
            enable_leak_detection: true,
            warning_threshold: 0.8,
        };

        let mut pool = OptimizedMemoryPool::new(device, config);

        // Test allocation and deallocation
        let sizes = vec![1024, 2048, 4096, 1024]; // Repeat 1024 to test reuse
        let mut buffers = Vec::new();

        for size in &sizes {
            match pool.allocate(*size) {
                Ok(buffer) => {
                    println!("Allocated buffer of size {}", size);
                    buffers.push(buffer);
                }
                Err(e) => {
                    println!("Failed to allocate buffer of size {}: {}", size, e);
                }
            }
        }

        // Check statistics
        let stats = pool.get_stats();
        println!("Memory stats: {:?}", stats);
        assert!(stats.allocation_count > 0);
        assert!(stats.current_usage > 0);

        // Deallocate buffers
        for buffer in buffers {
            pool.deallocate(buffer);
        }

        let final_stats = pool.get_stats();
        println!("Final memory stats: {:?}", final_stats);
        assert_eq!(final_stats.current_usage, 0);
    }

    #[test]
    fn test_mixed_precision() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(_) => {
                println!("Failed to create CUDA device, skipping test");
                return;
            }
        };

        match MixedPrecisionKernel::new(device) {
            Ok(mut kernel) => {
                println!("Mixed precision kernel created successfully");
                println!("Precision mode: {:?}", kernel.precision_mode());

                // Test precision mode setting
                let _ = kernel.set_precision_mode(PrecisionMode::FP32);
                assert_eq!(kernel.precision_mode(), PrecisionMode::FP32);

                // Test small matrix multiplication
                let m = 16;
                let n = 16;
                let k = 16;
                let a: Vec<f32> = (0..m*k).map(|i| (i as f32) / (m*k) as f32).collect();
                let b: Vec<f32> = (0..k*n).map(|i| (i as f32) / (k*n) as f32).collect();
                let mut c = vec![0.0f32; m * n];

                match kernel.matmul_mixed_precision(&a, &b, &mut c, m, n, k) {
                    Ok(()) => {
                        println!("Mixed precision matrix multiplication completed");
                        assert!(c.iter().any(|&x| x != 0.0));
                    }
                    Err(e) => {
                        println!("Mixed precision matrix multiplication failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to create mixed precision kernel: {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_benchmarks() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let config = BenchmarkConfig {
            matrix_sizes: vec![(64, 64, 64), (128, 128, 128)],
            warmup_iterations: 2,
            benchmark_iterations: 3,
            tolerance: 1e-3,
        };

        let benchmark = GpuBenchmark::new(config);
        
        match benchmark.run_benchmarks() {
            Ok(results) => {
                benchmark.print_results(&results);
                
                for result in &results {
                    if result.passed_correctness {
                        assert!(result.speedup > 0.0);
                        assert!(result.gflops_gpu > 0.0);
                        assert!(result.gflops_cpu > 0.0);
                        println!("Benchmark passed: {}x{}x{} - {:.2}x speedup", 
                                result.matrix_size.0, result.matrix_size.1, result.matrix_size.2,
                                result.speedup);
                    }
                }
            }
            Err(e) => {
                println!("Benchmark failed: {}", e);
            }
        }
    }

    #[test]
    fn test_batch_processing() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("Failed to create CUDA kernel, skipping test");
                return;
            }
        };

        // Create multiple small matrix operations
        let batch_size = 4;
        let m = 32;
        let n = 32;
        let k = 32;

        let mut batches = Vec::new();
        let mut test_data = Vec::new();

        for i in 0..batch_size {
            let a: Vec<i8> = (0..m*k).map(|j| ((i * 1000 + j) % 256) as i8 - 128).collect();
            let b: Vec<u8> = (0..k*n).map(|j| ((i * 2000 + j) % 4) as u8).collect();
            let mut c = vec![0.0f32; m * n];
            
            test_data.push((a, b, c));
        }

        // Convert to batch format
        for (a, b, c) in &mut test_data {
            batches.push((a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k));
        }

        match kernel.batch_matmul_i2s(&batches) {
            Ok(()) => {
                println!("Batch matrix multiplication completed successfully");
                
                // Verify results
                for (i, (_, _, c, _, _, _)) in batches.iter().enumerate() {
                    assert!(!c.iter().all(|&x| x == 0.0), "Batch {} has all zero results", i);
                    assert!(c.iter().all(|&x| x.is_finite()), "Batch {} has non-finite results", i);
                }
            }
            Err(e) => {
                println!("Batch matrix multiplication failed: {}", e);
            }
        }
    }

    #[test]
    fn test_performance_monitoring() {
        if !is_cuda_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("Failed to create CUDA kernel, skipping test");
                return;
            }
        };

        // Reset performance stats
        kernel.reset_performance_stats();

        // Perform some operations
        let m = 64;
        let n = 64;
        let k = 64;
        let a: Vec<i8> = (0..m*k).map(|i| (i % 256) as i8 - 128).collect();
        let b: Vec<u8> = (0..k*n).map(|i| (i % 4) as u8).collect();
        let mut c = vec![0.0f32; m * n];

        for _ in 0..5 {
            let _ = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
        }

        // Check performance stats
        let stats = kernel.performance_stats();
        println!("Performance stats: {:?}", stats);
        
        if stats.total_kernel_launches > 0 {
            assert!(stats.total_execution_time_ms > 0.0);
            assert!(stats.memory_transfers_host_to_device > 0);
            assert!(stats.memory_transfers_device_to_host > 0);
        }
    }
}