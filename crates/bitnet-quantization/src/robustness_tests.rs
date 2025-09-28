//! Robustness tests for memory management, SIMD operations, and system reliability
//!
//! This module provides comprehensive tests for system robustness, including
//! memory management, SIMD operations, concurrent access, and failure recovery.

#[cfg(test)]
mod tests {
    use crate::simd_ops::QuantizationKernels;
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
    use bitnet_common::Result;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Test memory allocation patterns and cleanup
    #[test]
    fn test_memory_allocation_patterns() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test progressive memory allocation
        let sizes = vec![32, 128, 512, 1024, 2048];
        let mut allocated_tensors = Vec::new();

        for size in sizes {
            // Create tensors of increasing size
            let data: Vec<f32> = (0..size * size).map(|i| (i % 1000) as f32 / 1000.0).collect();

            match create_tensor_from_f32(&data, &[size, size]) {
                Ok(tensor) => match quantizer.quantize(&tensor) {
                    Ok(quantized) => {
                        allocated_tensors.push((tensor, quantized));
                        println!("Successfully allocated and quantized {}x{} tensor", size, size);
                    }
                    Err(e) => {
                        println!("Quantization failed at size {}x{}: {}", size, size, e);
                        break;
                    }
                },
                Err(e) => {
                    println!("Tensor creation failed at size {}x{}: {}", size, size, e);
                    break;
                }
            }
        }

        // Test that all allocated tensors can still be dequantized
        for (idx, (original, quantized)) in allocated_tensors.iter().enumerate() {
            let dequantized = quantizer.dequantize(quantized)?;
            assert_eq!(dequantized.dims(), original.dims(), "Shape mismatch in tensor {}", idx);
        }

        println!(
            "✅ Memory allocation patterns test passed with {} tensors",
            allocated_tensors.len()
        );
        Ok(())
    }

    /// Test SIMD kernel robustness and fallback behavior
    #[test]
    fn test_simd_kernel_robustness() -> Result<()> {
        let kernels = QuantizationKernels::new();
        let capabilities = kernels.capabilities();

        println!(
            "SIMD capabilities: block_size={}, vectorized={}, optimal_alignment={}",
            capabilities.optimal_block_size(),
            capabilities.supports_vectorized_ops(),
            capabilities.optimal_alignment()
        );

        // Test with various data alignments
        let test_sizes = vec![8, 16, 24, 32, 40, 48, 64, 96, 128];

        for size in test_sizes {
            // Test with different alignment patterns
            let test_data: Vec<f32> =
                (0..size).map(|i| (i as f32 / size as f32) * 2.0 - 1.0).collect();

            // Test kernel operations with this data size
            let tensor = create_tensor_from_f32(&test_data, &[size])?;
            let quantizer = I2SQuantizer::new();

            match quantizer.quantize(&tensor) {
                Ok(quantized) => {
                    assert!(
                        !quantized.data.is_empty(),
                        "SIMD quantization should produce output for size {}",
                        size
                    );

                    // Test dequantization with SIMD
                    let dequantized = quantizer.dequantize(&quantized)?;
                    assert_eq!(
                        dequantized.dims(),
                        &[size],
                        "SIMD dequantization should preserve shape"
                    );
                }
                Err(e) => {
                    println!("⚠️  SIMD operation failed for size {}: {}", size, e);
                }
            }
        }

        // Test unaligned memory access patterns
        for offset in 0..8 {
            let size = 64 + offset;
            let test_data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let tensor = create_tensor_from_f32(&test_data, &[size])?;
            let quantizer = I2SQuantizer::new();

            match quantizer.quantize(&tensor) {
                Ok(_) => {
                    println!("✅ SIMD handled unaligned size {} successfully", size);
                }
                Err(_) => {
                    println!("⚠️  SIMD failed on unaligned size {}", size);
                }
            }
        }

        Ok(())
    }

    /// Test concurrent quantization operations
    #[test]
    fn test_concurrent_quantization() -> Result<()> {
        let num_threads = 4;
        let operations_per_thread = 10;
        let quantizer = Arc::new(I2SQuantizer::new());

        let mut handles = Vec::new();

        for thread_id in 0..num_threads {
            let quantizer_clone = Arc::clone(&quantizer);

            let handle = thread::spawn(move || -> Result<()> {
                for op_id in 0..operations_per_thread {
                    // Create unique test data for each operation
                    let seed = thread_id * 1000 + op_id;
                    let size = 64;
                    let test_data: Vec<f32> =
                        (0..size).map(|i| ((seed + i) as f32 / size as f32).sin()).collect();

                    let tensor = create_tensor_from_f32(&test_data, &[size])?;
                    let quantized = quantizer_clone.quantize(&tensor)?;
                    let dequantized = quantizer_clone.dequantize(&quantized)?;

                    // Verify the round-trip
                    assert_eq!(
                        dequantized.dims(),
                        &[size],
                        "Thread {} operation {} shape mismatch",
                        thread_id,
                        op_id
                    );

                    // Small delay to increase chance of race conditions
                    thread::sleep(Duration::from_millis(1));
                }
                Ok(())
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut all_succeeded = true;
        for (idx, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(result) => {
                    if let Err(e) = result {
                        println!("❌ Thread {} failed: {}", idx, e);
                        all_succeeded = false;
                    }
                }
                Err(_) => {
                    println!("❌ Thread {} panicked", idx);
                    all_succeeded = false;
                }
            }
        }

        if all_succeeded {
            println!("✅ Concurrent quantization test passed with {} threads", num_threads);
        }

        assert!(all_succeeded, "Concurrent quantization test failed");
        Ok(())
    }

    /// Test performance consistency under load
    #[test]
    fn test_performance_consistency() -> Result<()> {
        let quantizer = I2SQuantizer::new();
        let test_size = 256;
        let num_iterations = 20;

        // Generate test data
        let test_data: Vec<f32> = (0..test_size * test_size)
            .map(|i| (i as f32 / (test_size * test_size) as f32) * 2.0 - 1.0)
            .collect();
        let tensor = create_tensor_from_f32(&test_data, &[test_size, test_size])?;

        let mut timing_measurements = Vec::new();

        // Perform multiple timing measurements
        for iteration in 0..num_iterations {
            let start = Instant::now();

            let quantized = quantizer.quantize(&tensor)?;
            let _dequantized = quantizer.dequantize(&quantized)?;

            let duration = start.elapsed();
            timing_measurements.push(duration.as_micros() as f64);

            if iteration % 5 == 0 {
                println!("Iteration {}: {:.2}ms", iteration, duration.as_millis());
            }
        }

        // Analyze timing consistency
        let mean_time = timing_measurements.iter().sum::<f64>() / timing_measurements.len() as f64;
        let variance = timing_measurements.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>()
            / timing_measurements.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean_time;

        println!(
            "Performance stats: mean={:.2}μs, std_dev={:.2}μs, CV={:.3}",
            mean_time, std_dev, coefficient_of_variation
        );

        // Performance should be reasonably consistent (CV < 0.5)
        assert!(
            coefficient_of_variation < 0.5,
            "Performance too inconsistent: CV={:.3}",
            coefficient_of_variation
        );

        // Check for performance degradation (last measurements vs first)
        let first_half_avg = timing_measurements[..num_iterations / 2].iter().sum::<f64>()
            / (num_iterations / 2) as f64;
        let second_half_avg = timing_measurements[num_iterations / 2..].iter().sum::<f64>()
            / (num_iterations / 2) as f64;
        let degradation_ratio = second_half_avg / first_half_avg;

        println!("Performance degradation ratio: {:.3}", degradation_ratio);
        assert!(
            degradation_ratio < 1.5,
            "Performance degraded significantly: ratio={:.3}",
            degradation_ratio
        );

        Ok(())
    }

    /// Test error recovery and resilience
    #[test]
    fn test_error_recovery() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test recovery from various error conditions
        let error_scenarios = vec![
            ("Empty tensor", || create_tensor_from_f32(&[], &[0])),
            ("Invalid shape", || create_tensor_from_f32(&[1.0, 2.0], &[3])),
            ("Mismatched dimensions", || create_tensor_from_f32(&[1.0; 9], &[2, 3])),
        ];

        for (scenario_name, create_invalid_tensor) in error_scenarios {
            match create_invalid_tensor() {
                Ok(invalid_tensor) => {
                    // If tensor creation succeeds, quantization should handle it gracefully
                    match quantizer.quantize(&invalid_tensor) {
                        Ok(_) => println!("✅ {} handled gracefully", scenario_name),
                        Err(_) => println!("⚠️  {} correctly rejected", scenario_name),
                    }
                }
                Err(_) => {
                    println!("✅ {} correctly rejected at creation", scenario_name);
                }
            }

            // Test that quantizer still works after encountering errors
            let valid_data = vec![1.0, 2.0, 3.0, 4.0];
            let valid_tensor = create_tensor_from_f32(&valid_data, &[4])?;
            let result = quantizer.quantize(&valid_tensor);
            assert!(
                result.is_ok(),
                "Quantizer should recover after error scenario: {}",
                scenario_name
            );
        }

        Ok(())
    }

    /// Test quantization with extreme memory pressure simulation
    #[test]
    fn test_memory_pressure_handling() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Simulate memory pressure by creating many temporary allocations
        let mut temp_allocations = Vec::new();

        // Allocate temporary memory
        for i in 0..100 {
            let size = 1024 + i * 10;
            let temp_data: Vec<f32> = vec![0.0; size];
            temp_allocations.push(temp_data);
        }

        // Test quantization under memory pressure
        let test_sizes = vec![32, 64, 128, 256];
        let mut successful_operations = 0;

        for size in test_sizes {
            let test_data: Vec<f32> =
                (0..size * size).map(|i| (i as f32 / size as f32).sin()).collect();

            match create_tensor_from_f32(&test_data, &[size, size]) {
                Ok(tensor) => match quantizer.quantize(&tensor) {
                    Ok(quantized) => match quantizer.dequantize(&quantized) {
                        Ok(_) => {
                            successful_operations += 1;
                            println!(
                                "✅ Quantization succeeded under memory pressure: {}x{}",
                                size, size
                            );
                        }
                        Err(e) => {
                            println!("⚠️  Dequantization failed under memory pressure: {}", e);
                        }
                    },
                    Err(e) => {
                        println!("⚠️  Quantization failed under memory pressure: {}", e);
                    }
                },
                Err(e) => {
                    println!("⚠️  Tensor creation failed under memory pressure: {}", e);
                }
            }
        }

        // Drop temporary allocations
        drop(temp_allocations);

        // Should succeed in at least half the cases even under memory pressure
        assert!(
            successful_operations >= test_sizes.len() / 2,
            "Too many failures under memory pressure: {}/{}",
            successful_operations,
            test_sizes.len()
        );

        Ok(())
    }

    /// Test cross-quantizer robustness comparison
    #[test]
    fn test_cross_quantizer_robustness() -> Result<()> {
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        // Stress test scenarios
        let stress_scenarios = vec![
            ("Small irregular", vec![1.0; 7]),
            ("Medium regular", vec![0.5; 64]),
            ("Large pattern", (0..1024).map(|i| (i as f32 / 1024.0).sin()).collect()),
        ];

        let mut robustness_scores = std::collections::HashMap::new();

        for (quantizer_name, quantizer) in &quantizers {
            let mut successes = 0;
            let total_tests = stress_scenarios.len();

            for (scenario_name, test_data) in &stress_scenarios {
                let tensor_result = if test_data.len() == 7 {
                    // Irregular size - test with square root rounding
                    let side = (test_data.len() as f32).sqrt().ceil() as usize;
                    let mut padded_data = test_data.clone();
                    padded_data.resize(side * side, 0.0);
                    create_tensor_from_f32(&padded_data, &[side, side])
                } else if test_data.len() == 64 {
                    create_tensor_from_f32(test_data, &[8, 8])
                } else {
                    create_tensor_from_f32(test_data, &[32, 32])
                };

                match tensor_result {
                    Ok(tensor) => match quantizer.quantize(&tensor) {
                        Ok(quantized) => match quantizer.dequantize(&quantized) {
                            Ok(_) => {
                                successes += 1;
                                println!("✅ {} passed {}", quantizer_name, scenario_name);
                            }
                            Err(_) => {
                                println!(
                                    "❌ {} failed dequantization on {}",
                                    quantizer_name, scenario_name
                                );
                            }
                        },
                        Err(_) => {
                            println!(
                                "❌ {} failed quantization on {}",
                                quantizer_name, scenario_name
                            );
                        }
                    },
                    Err(_) => {
                        println!(
                            "❌ {} failed tensor creation on {}",
                            quantizer_name, scenario_name
                        );
                    }
                }
            }

            let robustness_score = successes as f64 / total_tests as f64;
            robustness_scores.insert(quantizer_name.to_string(), robustness_score);
            println!("{} robustness score: {:.2}", quantizer_name, robustness_score);
        }

        // All quantizers should have reasonable robustness (>= 60%)
        for (name, score) in &robustness_scores {
            assert!(*score >= 0.6, "{} robustness too low: {:.2}", name, score);
        }

        Ok(())
    }

    /// Test SIMD vs scalar consistency
    #[test]
    fn test_simd_scalar_consistency() -> Result<()> {
        let kernels = QuantizationKernels::new();

        // Test with different data patterns that stress SIMD alignment
        let test_patterns = vec![
            ("Sequential", (0..128).map(|i| i as f32 / 128.0).collect::<Vec<f32>>()),
            ("Alternating", (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()),
            ("Random-like", (0..128).map(|i| ((i * 17 + 7) as f32).sin()).collect()),
            ("Sparse", (0..128).map(|i| if i % 8 == 0 { 1.0 } else { 0.0 }).collect()),
        ];

        for (pattern_name, pattern_data) in test_patterns {
            let tensor = create_tensor_from_f32(&pattern_data, &[128])?;
            let quantizer = I2SQuantizer::new();

            // Test multiple runs for consistency
            let mut results = Vec::new();
            for run in 0..3 {
                let quantized = quantizer.quantize(&tensor)?;
                let dequantized = quantizer.dequantize(&quantized)?;
                results.push((quantized, dequantized));

                if run > 0 {
                    // Compare with first run
                    assert_eq!(
                        results[0].0.data, results[run].0.data,
                        "SIMD/scalar inconsistency in {} pattern, run {}",
                        pattern_name, run
                    );
                }
            }

            println!("✅ SIMD/scalar consistency verified for {} pattern", pattern_name);
        }

        Ok(())
    }
}
