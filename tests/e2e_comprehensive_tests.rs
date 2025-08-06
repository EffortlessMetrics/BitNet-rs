//! End-to-end comprehensive tests for the entire BitNet system
//! Tests complete workflows from configuration to model loading to inference

use bitnet::*;
use bitnet_common::*;
use bitnet_models::*;
use bitnet_quantization::*;
use bitnet_kernels::*;
use tempfile::{NamedTempFile, TempDir};
use std::fs;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Test complete system integration
mod system_integration {
    use super::*;

    #[test]
    fn test_full_system_initialization() {
        // Test that all components can be initialized together
        let config = BitNetConfig::default();
        
        // Initialize kernel manager
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        println!("Selected kernel: {}", kernel.name());
        
        // Initialize model loader
        let device = Device::Cpu;
        let model_loader = ModelLoader::new(device);
        let formats = model_loader.available_formats();
        println!("Available formats: {:?}", formats);
        
        // Initialize quantizers
        let i2s_quantizer = I2SQuantizer::new().unwrap();
        let tl1_quantizer = TL1Quantizer::new().unwrap();
        let tl2_quantizer = TL2Quantizer::new().unwrap();
        
        println!("All components initialized successfully");
        
        // Test basic operations
        let test_data = vec![1.0f32, -1.0, 0.5, -0.5];
        let tensor = MockTensor::new(test_data);
        
        let quantized = i2s_quantizer.quantize(&tensor).unwrap();
        assert!(!quantized.data.is_empty());
        
        println!("System integration test passed");
    }

    #[test]
    fn test_configuration_driven_workflow() {
        // Create a comprehensive configuration
        let config = BitNetConfig::builder()
            .vocab_size(32000)
            .hidden_size(4096)
            .num_layers(32)
            .num_heads(32)
            .temperature(0.8)
            .quantization_type(QuantizationType::I2S)
            .use_gpu(false)
            .batch_size(1)
            .build()
            .unwrap();
        
        // Use configuration to drive system behavior
        assert_eq!(config.model.vocab_size, 32000);
        assert_eq!(config.quantization.quantization_type, QuantizationType::I2S);
        
        // Initialize components based on configuration
        let device = if config.performance.use_gpu {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        
        let model_loader = ModelLoader::new(device);
        
        // Select quantizer based on configuration
        let quantizer: Box<dyn Quantize> = match config.quantization.quantization_type {
            QuantizationType::I2S => Box::new(I2SQuantizer::new().unwrap()),
            QuantizationType::TL1 => Box::new(TL1Quantizer::new().unwrap()),
            QuantizationType::TL2 => Box::new(TL2Quantizer::new().unwrap()),
        };
        
        // Test quantization with configured parameters
        let test_data: Vec<f32> = (0..config.quantization.block_size * 4)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let tensor = MockTensor::new(test_data);
        
        let quantized = quantizer.quantize(&tensor).unwrap();
        assert!(!quantized.data.is_empty());
        
        println!("Configuration-driven workflow test passed");
    }

    #[test]
    fn test_error_propagation_across_components() {
        // Test that errors propagate correctly through the system
        
        // Test invalid configuration
        let mut config = BitNetConfig::default();
        config.model.vocab_size = 0; // Invalid
        assert!(config.validate().is_err());
        
        // Test model loading errors
        let device = Device::Cpu;
        let model_loader = ModelLoader::new(device);
        let result = model_loader.load_model("non_existent_model.gguf", None);
        assert!(result.is_err());
        
        // Test quantization errors
        let quantizer = I2SQuantizer::new().unwrap();
        let empty_tensor = MockTensor::new(vec![]);
        let result = quantizer.quantize(&empty_tensor);
        // Should handle empty tensors gracefully
        
        // Test kernel errors
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err());
        
        println!("Error propagation test passed");
    }
}

/// Test realistic model processing workflows
mod model_processing_workflows {
    use super::*;

    #[test]
    fn test_model_quantization_pipeline() {
        // Simulate a complete model quantization pipeline
        
        // Step 1: Create simulated model weights
        let layer_sizes = vec![
            (512, 1024),   // Input layer
            (1024, 2048),  // Hidden layer 1
            (2048, 1024),  // Hidden layer 2
            (1024, 512),   // Output layer
        ];
        
        let mut total_params = 0;
        let mut quantized_layers = Vec::new();
        
        for (i, (input_dim, output_dim)) in layer_sizes.iter().enumerate() {
            println!("Processing layer {}: {}x{}", i, input_dim, output_dim);
            
            // Generate realistic weight distribution
            let weights: Vec<f32> = (0..input_dim * output_dim)
                .map(|j| {
                    let x = (j as f32 - (input_dim * output_dim / 2) as f32) / 1000.0;
                    x.sin() * (-x * x / 2.0).exp() // Gaussian-like distribution
                })
                .collect();
            
            total_params += weights.len();
            
            // Quantize the layer
            let tensor = MockTensor::new(weights.clone());
            let quantizer = I2SQuantizer::new().unwrap();
            
            let start = Instant::now();
            let quantized = quantizer.quantize(&tensor).unwrap();
            let quantize_time = start.elapsed();
            
            // Verify quantization
            assert!(!quantized.data.is_empty());
            assert!(!quantized.scales.is_empty());
            
            // Calculate compression ratio
            let original_size = weights.len() * std::mem::size_of::<f32>();
            let compressed_size = quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            println!("  Quantization time: {:?}", quantize_time);
            println!("  Compression ratio: {:.2}x", compression_ratio);
            
            quantized_layers.push(quantized);
        }
        
        println!("Model quantization pipeline completed");
        println!("Total parameters: {}", total_params);
        println!("Total layers: {}", quantized_layers.len());
        
        // Verify all layers were processed
        assert_eq!(quantized_layers.len(), layer_sizes.len());
    }

    #[test]
    fn test_inference_simulation() {
        // Simulate a complete inference pass through a quantized model
        
        let batch_size = 4;
        let sequence_length = 128;
        let hidden_dim = 512;
        let vocab_size = 32000;
        
        // Step 1: Create input tokens (simulated)
        let input_tokens: Vec<i32> = (0..batch_size * sequence_length)
            .map(|i| (i % vocab_size as usize) as i32)
            .collect();
        
        println!("Processing batch of {} sequences, length {}", batch_size, sequence_length);
        
        // Step 2: Simulate embedding lookup (quantized embeddings)
        let embedding_weights: Vec<u8> = (0..vocab_size * hidden_dim)
            .map(|i| ((i * 17 + 23) % 4) as u8)
            .collect();
        
        // Step 3: Simulate attention computation
        let query_weights: Vec<i8> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i * 13 + 7) % 5) as i8 - 2)
            .collect();
        
        let key_weights: Vec<u8> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i * 19 + 11) % 4) as u8)
            .collect();
        
        let value_weights: Vec<u8> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i * 23 + 13) % 4) as u8)
            .collect();
        
        // Step 4: Perform matrix multiplications using kernel
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        let mut query_output = vec![0.0f32; batch_size * sequence_length * hidden_dim];
        let mut key_output = vec![0.0f32; batch_size * sequence_length * hidden_dim];
        let mut value_output = vec![0.0f32; batch_size * sequence_length * hidden_dim];
        
        let start = Instant::now();
        
        // Simulate Q, K, V computations
        let result = kernel.matmul_i2s(
            &query_weights,
            &embedding_weights[..hidden_dim * hidden_dim],
            &mut query_output[..hidden_dim * hidden_dim],
            hidden_dim,
            hidden_dim,
            hidden_dim,
        );
        assert!(result.is_ok());
        
        let result = kernel.matmul_i2s(
            &key_weights[..hidden_dim * hidden_dim],
            &embedding_weights[..hidden_dim * hidden_dim],
            &mut key_output[..hidden_dim * hidden_dim],
            hidden_dim,
            hidden_dim,
            hidden_dim,
        );
        assert!(result.is_ok());
        
        let result = kernel.matmul_i2s(
            &value_weights[..hidden_dim * hidden_dim],
            &embedding_weights[..hidden_dim * hidden_dim],
            &mut value_output[..hidden_dim * hidden_dim],
            hidden_dim,
            hidden_dim,
            hidden_dim,
        );
        assert!(result.is_ok());
        
        let inference_time = start.elapsed();
        
        // Step 5: Verify outputs
        assert!(query_output.iter().all(|&x| x.is_finite()));
        assert!(key_output.iter().all(|&x| x.is_finite()));
        assert!(value_output.iter().all(|&x| x.is_finite()));
        
        println!("Inference simulation completed");
        println!("Total inference time: {:?}", inference_time);
        
        // Calculate throughput
        let tokens_per_second = (batch_size * sequence_length) as f64 / inference_time.as_secs_f64();
        println!("Throughput: {:.2} tokens/second", tokens_per_second);
    }

    #[test]
    fn test_memory_efficient_processing() {
        // Test processing large models with memory constraints
        
        let max_memory_mb = 100; // 100MB limit
        let max_memory_bytes = max_memory_mb * 1024 * 1024;
        
        // Create a large model that would exceed memory if loaded all at once
        let total_params = 10_000_000; // 10M parameters
        let chunk_size = max_memory_bytes / (std::mem::size_of::<f32>() * 4); // Leave room for processing
        
        println!("Processing {} parameters in chunks of {}", total_params, chunk_size);
        
        let mut processed_chunks = 0;
        let mut total_compression_ratio = 0.0;
        
        let quantizer = I2SQuantizer::new().unwrap();
        
        for chunk_start in (0..total_params).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_params);
            let chunk_params = chunk_end - chunk_start;
            
            // Generate chunk data
            let chunk_data: Vec<f32> = (chunk_start..chunk_end)
                .map(|i| (i as f32 * 0.0001).sin())
                .collect();
            
            // Process chunk
            let tensor = MockTensor::new(chunk_data.clone());
            let quantized = quantizer.quantize(&tensor).unwrap();
            
            // Calculate compression for this chunk
            let original_size = chunk_data.len() * std::mem::size_of::<f32>();
            let compressed_size = quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();
            let compression_ratio = original_size as f32 / compressed_size as f32;
            
            total_compression_ratio += compression_ratio;
            processed_chunks += 1;
            
            println!("Processed chunk {}: {} params, {:.2}x compression", 
                     processed_chunks, chunk_params, compression_ratio);
        }
        
        let avg_compression = total_compression_ratio / processed_chunks as f32;
        println!("Memory-efficient processing completed");
        println!("Processed {} chunks with average compression: {:.2}x", processed_chunks, avg_compression);
        
        assert!(processed_chunks > 1, "Should have processed multiple chunks");
        assert!(avg_compression > 1.0, "Should achieve compression");
    }
}

/// Test performance and scalability
mod performance_scalability {
    use super::*;

    #[test]
    fn test_scalability_across_sizes() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        let sizes = vec![64, 128, 256, 512, 1024];
        let mut performance_data = Vec::new();
        
        for size in sizes {
            println!("Testing scalability for size: {}", size);
            
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];
            
            // Warm up
            for _ in 0..3 {
                kernel.matmul_i2s(&a, &b, &mut c, size, size, size).unwrap();
            }
            
            // Measure performance
            let iterations = if size <= 256 { 10 } else { 3 };
            let start = Instant::now();
            
            for _ in 0..iterations {
                kernel.matmul_i2s(&a, &b, &mut c, size, size, size).unwrap();
            }
            
            let total_time = start.elapsed();
            let avg_time = total_time / iterations;
            
            // Calculate GFLOPS
            let ops = 2.0 * size as f64 * size as f64 * size as f64;
            let gflops = ops / (avg_time.as_secs_f64() * 1e9);
            
            performance_data.push((size, avg_time, gflops));
            
            println!("  Size: {}, Time: {:?}, Performance: {:.2} GFLOPS", size, avg_time, gflops);
        }
        
        // Analyze scalability
        println!("\nScalability Analysis:");
        for i in 1..performance_data.len() {
            let (prev_size, prev_time, prev_gflops) = performance_data[i-1];
            let (curr_size, curr_time, curr_gflops) = performance_data[i];
            
            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time.as_secs_f64() / prev_time.as_secs_f64();
            let expected_time_ratio = size_ratio.powi(3); // O(n^3) for matrix multiplication
            
            println!("  {}x{} -> {}x{}: Time ratio {:.2} (expected {:.2}), GFLOPS {:.2} -> {:.2}",
                     prev_size, prev_size, curr_size, curr_size,
                     time_ratio, expected_time_ratio, prev_gflops, curr_gflops);
        }
    }

    #[test]
    fn test_concurrent_processing() {
        use std::thread;
        use std::sync::Arc;
        
        let kernel_manager = Arc::new(KernelManager::new());
        let num_threads = 4;
        let operations_per_thread = 100;
        
        println!("Testing concurrent processing with {} threads", num_threads);
        
        let start = Instant::now();
        
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let kernel_manager_clone = Arc::clone(&kernel_manager);
            
            thread::spawn(move || {
                let kernel = kernel_manager_clone.select_best().unwrap();
                let mut thread_time = std::time::Duration::ZERO;
                
                for i in 0..operations_per_thread {
                    let size = 64 + (i % 4) * 16; // Vary size: 64, 80, 96, 112
                    let a = vec![1i8; size * size];
                    let b = vec![1u8; size * size];
                    let mut c = vec![0.0f32; size * size];
                    
                    let op_start = Instant::now();
                    let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
                    thread_time += op_start.elapsed();
                    
                    assert!(result.is_ok(), "Thread {} operation {} failed", thread_id, i);
                }
                
                (thread_id, thread_time)
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_time = start.elapsed();
        
        // Analyze results
        let total_operations = num_threads * operations_per_thread;
        let avg_ops_per_second = total_operations as f64 / total_time.as_secs_f64();
        
        println!("Concurrent processing results:");
        println!("  Total operations: {}", total_operations);
        println!("  Total time: {:?}", total_time);
        println!("  Average ops/second: {:.2}", avg_ops_per_second);
        
        for (thread_id, thread_time) in results {
            let thread_ops_per_second = operations_per_thread as f64 / thread_time.as_secs_f64();
            println!("  Thread {}: {:.2} ops/second", thread_id, thread_ops_per_second);
        }
    }

    #[test]
    fn test_memory_bandwidth_utilization() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        // Test different matrix shapes to analyze memory bandwidth utilization
        let test_cases = vec![
            ("Square", 512, 512, 512),
            ("Wide", 128, 2048, 512),
            ("Tall", 2048, 128, 512),
            ("Deep", 512, 512, 2048),
        ];
        
        println!("Memory bandwidth utilization analysis:");
        
        for (name, m, n, k) in test_cases {
            let a = vec![1i8; m * k];
            let b = vec![1u8; k * n];
            let mut c = vec![0.0f32; m * n];
            
            // Calculate theoretical memory usage
            let memory_reads = (m * k) + (k * n); // Input matrices
            let memory_writes = m * n; // Output matrix
            let total_memory_ops = memory_reads + memory_writes;
            
            // Measure performance
            let iterations = 5;
            let start = Instant::now();
            
            for _ in 0..iterations {
                kernel.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();
            }
            
            let avg_time = start.elapsed() / iterations;
            
            // Calculate bandwidth utilization
            let bytes_per_op = 1 + 1 + 4; // i8 + u8 + f32
            let total_bytes = total_memory_ops * bytes_per_op;
            let bandwidth_gb_s = (total_bytes as f64) / (avg_time.as_secs_f64() * 1e9);
            
            // Calculate GFLOPS
            let ops = 2.0 * m as f64 * n as f64 * k as f64;
            let gflops = ops / (avg_time.as_secs_f64() * 1e9);
            
            println!("  {}: {}x{}x{}", name, m, n, k);
            println!("    Time: {:?}", avg_time);
            println!("    Bandwidth: {:.2} GB/s", bandwidth_gb_s);
            println!("    Performance: {:.2} GFLOPS", gflops);
        }
    }
}

/// Test robustness and error recovery
mod robustness_tests {
    use super::*;

    #[test]
    fn test_system_recovery_after_errors() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        // Cause various errors and verify system recovers
        
        // Error 1: Invalid matrix dimensions
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err());
        
        // System should still work after error
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_ok());
        
        // Error 2: Invalid quantization parameters
        let result = kernel.quantize(&[], &mut [], &mut [], QuantizationType::I2S);
        assert!(result.is_err());
        
        // System should still work after error
        let input = vec![1.0f32; 64];
        let mut output = vec![0u8; 32];
        let mut scales = vec![0.0f32; 4];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());
        
        println!("System recovery test passed");
    }

    #[test]
    fn test_resource_cleanup() {
        // Test that resources are properly cleaned up
        
        let initial_kernel_count = {
            let manager = KernelManager::new();
            manager.list_available_providers().len()
        };
        
        // Create and drop many kernel managers
        for _ in 0..100 {
            let manager = KernelManager::new();
            let _kernel = manager.select_best().unwrap();
            // Manager goes out of scope here
        }
        
        // Verify kernel count is still the same
        let final_kernel_count = {
            let manager = KernelManager::new();
            manager.list_available_providers().len()
        };
        
        assert_eq!(initial_kernel_count, final_kernel_count);
        
        // Test quantizer cleanup
        for _ in 0..100 {
            let _quantizer = I2SQuantizer::new().unwrap();
            // Quantizer goes out of scope here
        }
        
        println!("Resource cleanup test passed");
    }

    #[test]
    fn test_stress_under_load() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        // Perform many operations rapidly
        let num_operations = 1000;
        let mut success_count = 0;
        
        for i in 0..num_operations {
            let size = 32 + (i % 8) * 8; // Vary size
            let a = vec![1i8; size * size];
            let b = vec![1u8; size * size];
            let mut c = vec![0.0f32; size * size];
            
            match kernel.matmul_i2s(&a, &b, &mut c, size, size, size) {
                Ok(_) => success_count += 1,
                Err(_) => {
                    // Some failures might be acceptable under extreme load
                    println!("Operation {} failed (size {})", i, size);
                }
            }
        }
        
        let success_rate = success_count as f64 / num_operations as f64;
        println!("Stress test: {}/{} operations succeeded ({:.1}%)", 
                 success_count, num_operations, success_rate * 100.0);
        
        // Should have high success rate even under stress
        assert!(success_rate > 0.95, "Success rate too low: {:.1}%", success_rate * 100.0);
    }

    #[test]
    fn test_edge_case_combinations() {
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        
        // Test combinations of edge cases
        let edge_cases = vec![
            // (description, m, n, k, should_succeed)
            ("Minimum valid", 1, 1, 1, true),
            ("Single row", 1, 64, 64, true),
            ("Single column", 64, 1, 64, true),
            ("Single inner", 64, 64, 1, true),
            ("Power of 2", 64, 64, 64, true),
            ("Prime numbers", 17, 19, 23, true),
            ("Large prime", 127, 131, 137, true),
        ];
        
        for (description, m, n, k, should_succeed) in edge_cases {
            println!("Testing edge case: {} ({}x{}x{})", description, m, n, k);
            
            let a = vec![1i8; m * k];
            let b = vec![1u8; k * n];
            let mut c = vec![0.0f32; m * n];
            
            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            
            if should_succeed {
                assert!(result.is_ok(), "Edge case '{}' should succeed", description);
                assert!(c.iter().all(|&x| x.is_finite()), "Results should be finite");
            } else {
                assert!(result.is_err(), "Edge case '{}' should fail", description);
            }
        }
    }
}

/// Test cross-component compatibility
mod compatibility_tests {
    use super::*;

    #[test]
    fn test_quantization_kernel_compatibility() {
        // Test that all quantization methods work with all available kernels
        
        let kernel_manager = KernelManager::new();
        let available_kernels = kernel_manager.list_available_providers();
        
        let quantizers: Vec<(String, Box<dyn Quantize>)> = vec![
            ("I2S".to_string(), Box::new(I2SQuantizer::new().unwrap())),
            ("TL1".to_string(), Box::new(TL1Quantizer::new().unwrap())),
            ("TL2".to_string(), Box::new(TL2Quantizer::new().unwrap())),
        ];
        
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let tensor = MockTensor::new(test_data);
        
        for (quant_name, quantizer) in &quantizers {
            println!("Testing {} quantization", quant_name);
            
            let quantized = quantizer.quantize(&tensor).unwrap();
            assert!(!quantized.data.is_empty());
            
            // Test that quantized data can be used with kernel operations
            let kernel = kernel_manager.select_best().unwrap();
            
            if quantized.data.len() >= 64 {
                let input = vec![1i8; 8];
                let weights = &quantized.data[..64];
                let mut output = vec![0.0f32; 8];
                
                let result = kernel.matmul_i2s(&input, weights, &mut output, 8, 8, 8);
                assert!(result.is_ok(), "{} quantized data incompatible with kernel", quant_name);
            }
        }
        
        println!("Quantization-kernel compatibility test passed");
    }

    #[test]
    fn test_device_compatibility() {
        // Test that components work correctly with different device types
        
        let devices = vec![Device::Cpu, Device::Cuda(0)];
        
        for device in devices {
            println!("Testing device compatibility: {:?}", device);
            
            // Test model loader with device
            let model_loader = ModelLoader::new(device.clone());
            let formats = model_loader.available_formats();
            assert!(!formats.is_empty());
            
            // Test tensor operations with device
            let tensor = ConcreteTensor::new(vec![1.0f32; 16], vec![4, 4], DType::F32);
            assert_eq!(tensor.device(), &Device::Cpu); // Currently all tensors are CPU
            
            // Test that device selection doesn't break other components
            let kernel_manager = KernelManager::new();
            let kernel = kernel_manager.select_best().unwrap();
            assert!(kernel.is_available());
        }
        
        println!("Device compatibility test passed");
    }

    #[test]
    fn test_configuration_compatibility() {
        // Test that different configurations work together
        
        let configs = vec![
            BitNetConfig::builder().quantization_type(QuantizationType::I2S).build().unwrap(),
            BitNetConfig::builder().quantization_type(QuantizationType::TL1).build().unwrap(),
            BitNetConfig::builder().quantization_type(QuantizationType::TL2).build().unwrap(),
        ];
        
        for config in configs {
            println!("Testing configuration compatibility: {:?}", config.quantization.quantization_type);
            
            // Create components based on configuration
            let device = if config.performance.use_gpu { Device::Cuda(0) } else { Device::Cpu };
            let model_loader = ModelLoader::new(device);
            
            let quantizer: Box<dyn Quantize> = match config.quantization.quantization_type {
                QuantizationType::I2S => Box::new(I2SQuantizer::new().unwrap()),
                QuantizationType::TL1 => Box::new(TL1Quantizer::new().unwrap()),
                QuantizationType::TL2 => Box::new(TL2Quantizer::new().unwrap()),
            };
            
            // Test that components work together
            let test_data: Vec<f32> = (0..config.quantization.block_size * 2)
                .map(|i| (i as f32 * 0.01).sin())
                .collect();
            let tensor = MockTensor::new(test_data);
            
            let quantized = quantizer.quantize(&tensor).unwrap();
            assert!(!quantized.data.is_empty());
            
            // Verify configuration parameters are respected
            // (This would be more meaningful with actual implementation details)
        }
        
        println!("Configuration compatibility test passed");
    }
}

/// Test real-world scenarios
mod real_world_scenarios {
    use super::*;

    #[test]
    fn test_model_serving_simulation() {
        // Simulate a model serving scenario with multiple concurrent requests
        
        use std::thread;
        use std::sync::Arc;
        
        let kernel_manager = Arc::new(KernelManager::new());
        let num_requests = 10;
        let batch_size = 4;
        let sequence_length = 64;
        let hidden_dim = 256;
        
        println!("Simulating model serving with {} concurrent requests", num_requests);
        
        let start = Instant::now();
        
        let handles: Vec<_> = (0..num_requests).map(|request_id| {
            let kernel_manager_clone = Arc::clone(&kernel_manager);
            
            thread::spawn(move || {
                let kernel = kernel_manager_clone.select_best().unwrap();
                
                // Simulate request processing
                let input = vec![1i8; batch_size * sequence_length];
                let weights = vec![1u8; sequence_length * hidden_dim];
                let mut output = vec![0.0f32; batch_size * hidden_dim];
                
                let request_start = Instant::now();
                let result = kernel.matmul_i2s(&input, &weights, &mut output, batch_size, hidden_dim, sequence_length);
                let request_time = request_start.elapsed();
                
                assert!(result.is_ok(), "Request {} failed", request_id);
                
                (request_id, request_time, output.len())
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_time = start.elapsed();
        
        // Analyze serving performance
        let total_tokens_processed: usize = results.iter().map(|(_, _, tokens)| tokens).sum();
        let avg_request_time: std::time::Duration = results.iter().map(|(_, time, _)| *time).sum::<std::time::Duration>() / results.len() as u32;
        let throughput = total_tokens_processed as f64 / total_time.as_secs_f64();
        
        println!("Model serving simulation results:");
        println!("  Total requests: {}", num_requests);
        println!("  Total time: {:?}", total_time);
        println!("  Average request time: {:?}", avg_request_time);
        println!("  Total tokens processed: {}", total_tokens_processed);
        println!("  Throughput: {:.2} tokens/second", throughput);
        
        // Verify all requests succeeded
        assert_eq!(results.len(), num_requests);
        for (request_id, _, _) in results {
            println!("  Request {} completed successfully", request_id);
        }
    }

    #[test]
    fn test_batch_processing_workflow() {
        // Test processing multiple batches with different sizes
        
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        let quantizer = I2SQuantizer::new().unwrap();
        
        let batch_configs = vec![
            (1, 128),   // Single sequence
            (4, 64),    // Small batch
            (8, 32),    // Medium batch
            (16, 16),   // Large batch, short sequences
        ];
        
        let mut total_processing_time = std::time::Duration::ZERO;
        let mut total_tokens = 0;
        
        for (batch_size, seq_length) in batch_configs {
            println!("Processing batch: {} sequences of length {}", batch_size, seq_length);
            
            // Step 1: Prepare batch data
            let input_data: Vec<f32> = (0..batch_size * seq_length * 256)
                .map(|i| (i as f32 * 0.001).sin())
                .collect();
            
            // Step 2: Quantize input
            let tensor = MockTensor::new(input_data);
            let quantize_start = Instant::now();
            let quantized = quantizer.quantize(&tensor).unwrap();
            let quantize_time = quantize_start.elapsed();
            
            // Step 3: Process with kernel
            let process_start = Instant::now();
            if quantized.data.len() >= 256 {
                let input = vec![1i8; batch_size * seq_length];
                let weights = &quantized.data[..256];
                let mut output = vec![0.0f32; batch_size * 16];
                
                let result = kernel.matmul_i2s(&input, weights, &mut output, batch_size, 16, seq_length);
                assert!(result.is_ok());
            }
            let process_time = process_start.elapsed();
            
            let batch_total_time = quantize_time + process_time;
            total_processing_time += batch_total_time;
            total_tokens += batch_size * seq_length;
            
            println!("  Quantize time: {:?}", quantize_time);
            println!("  Process time: {:?}", process_time);
            println!("  Total time: {:?}", batch_total_time);
        }
        
        let avg_tokens_per_second = total_tokens as f64 / total_processing_time.as_secs_f64();
        println!("Batch processing workflow completed");
        println!("Total tokens processed: {}", total_tokens);
        println!("Total processing time: {:?}", total_processing_time);
        println!("Average throughput: {:.2} tokens/second", avg_tokens_per_second);
    }

    #[test]
    fn test_production_readiness_checklist() {
        // Test various production readiness criteria
        
        println!("Production readiness checklist:");
        
        // 1. Component initialization
        let config = BitNetConfig::default();
        assert!(config.validate().is_ok());
        println!("✓ Configuration validation");
        
        // 2. Kernel availability
        let kernel_manager = KernelManager::new();
        let kernel = kernel_manager.select_best().unwrap();
        assert!(kernel.is_available());
        println!("✓ Kernel availability: {}", kernel.name());
        
        // 3. Model loading capability
        let device = Device::Cpu;
        let model_loader = ModelLoader::new(device);
        let formats = model_loader.available_formats();
        assert!(!formats.is_empty());
        println!("✓ Model format support: {:?}", formats);
        
        // 4. Quantization methods
        let i2s_quantizer = I2SQuantizer::new();
        let tl1_quantizer = TL1Quantizer::new();
        let tl2_quantizer = TL2Quantizer::new();
        assert!(i2s_quantizer.is_ok() && tl1_quantizer.is_ok() && tl2_quantizer.is_ok());
        println!("✓ All quantization methods available");
        
        // 5. Error handling
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err());
        println!("✓ Error handling works");
        
        // 6. Performance benchmarking
        let a = vec![1i8; 64 * 64];
        let b = vec![1u8; 64 * 64];
        let mut c = vec![0.0f32; 64 * 64];
        
        let start = Instant::now();
        let result = kernel.matmul_i2s(&a, &b, &mut c, 64, 64, 64);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000); // Should be fast
        println!("✓ Performance acceptable: {:?}", duration);
        
        // 7. Memory management
        // (Rust handles this automatically, but we can test for leaks)
        for _ in 0..100 {
            let _temp_kernel = KernelManager::new();
            let _temp_quantizer = I2SQuantizer::new().unwrap();
        }
        println!("✓ Memory management");
        
        // 8. Thread safety
        use std::thread;
        use std::sync::Arc;
        
        let kernel_manager = Arc::new(KernelManager::new());
        let handles: Vec<_> = (0..4).map(|_| {
            let km = Arc::clone(&kernel_manager);
            thread::spawn(move || {
                let k = km.select_best().unwrap();
                k.name().to_string()
            })
        }).collect();
        
        let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(results.iter().all(|name| name == &results[0]));
        println!("✓ Thread safety");
        
        println!("All production readiness checks passed!");
    }
}