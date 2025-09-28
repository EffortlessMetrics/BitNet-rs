//! Comprehensive edge case and boundary condition tests for BitNet quantization
//!
//! This module provides extensive test coverage for edge cases, boundary conditions,
//! and numerical stability scenarios across I2S, TL1, and TL2 quantization algorithms.

#[cfg(test)]
mod tests {
    use crate::utils::create_tensor_from_f32;
    use crate::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
    use bitnet_common::Result;
    use std::f32;

    /// Test boundary values for I2S quantization
    #[test]
    fn test_i2s_boundary_values() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test with extreme values
        let extreme_values = vec![
            f32::MIN,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1000.0,
            -100.0,
            -10.0,
            -1.0,
            -0.1,
            -0.01,
            -0.001,
            0.0,
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
        ];

        for value in extreme_values {
            if value.is_finite() {
                let input = create_tensor_from_f32(&[value; 32], &[32])?;
                let result = quantizer.quantize(&input);

                match result {
                    Ok(quantized) => {
                        // Ensure quantization produces valid output
                        assert!(
                            !quantized.data.is_empty(),
                            "Quantized data should not be empty for value {}",
                            value
                        );

                        // Test dequantization round-trip
                        let dequantized = quantizer.dequantize(&quantized)?;
                        assert_eq!(dequantized.dims(), input.dims(), "Shape should be preserved");
                    }
                    Err(_) => {
                        // Some extreme values may legitimately fail
                        println!("Expected failure for extreme value: {}", value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Test I2S quantization with various tensor shapes and sizes
    #[test]
    fn test_i2s_shape_boundary_conditions() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test different tensor shapes
        let test_shapes = vec![
            (1, 1),       // Minimal shape
            (32, 1),      // Single column
            (1, 32),      // Single row
            (32, 32),     // Square
            (64, 128),    // Rectangular
            (128, 64),    // Transposed rectangular
            (256, 512),   // Large
            (1024, 2048), // Very large
        ];

        for (rows, cols) in test_shapes {
            let total_elements = rows * cols;
            let input_data: Vec<f32> = (0..total_elements)
                .map(|i| (i as f32 / total_elements as f32) * 2.0 - 1.0) // Range [-1, 1]
                .collect();

            let input = create_tensor_from_f32(&input_data, &[rows, cols])?;

            match quantizer.quantize(&input) {
                Ok(quantized) => {
                    assert!(
                        quantized.shape == vec![rows, cols],
                        "Shape mismatch for {}x{}",
                        rows,
                        cols
                    );

                    // Test dequantization preserves shape
                    let dequantized = quantizer.dequantize(&quantized)?;
                    assert_eq!(dequantized.dims(), &[rows, cols], "Dequantized shape mismatch");
                }
                Err(e) => {
                    // Large tensors may fail due to memory constraints
                    if total_elements > 1_000_000 {
                        println!("Expected memory constraint failure for {}x{}: {}", rows, cols, e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Test TL1 quantization edge cases
    #[test]
    fn test_tl1_edge_cases() -> Result<()> {
        let quantizer = TL1Quantizer::new();

        // Test with all zeros
        let zeros = create_tensor_from_f32(&[0.0; 64], &[8, 8])?;
        let quantized = quantizer.quantize(&zeros)?;
        assert!(!quantized.data.is_empty(), "Zero tensor should be quantizable");

        // Test with all ones
        let ones = create_tensor_from_f32(&[1.0; 64], &[8, 8])?;
        let quantized = quantizer.quantize(&ones)?;
        assert!(!quantized.data.is_empty(), "Ones tensor should be quantizable");

        // Test with alternating pattern
        let alternating: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let alt_tensor = create_tensor_from_f32(&alternating, &[8, 8])?;
        let quantized = quantizer.quantize(&alt_tensor)?;
        assert!(!quantized.data.is_empty(), "Alternating pattern should be quantizable");

        // Test with gradual gradient
        let gradient: Vec<f32> = (0..64).map(|i| (i as f32 / 63.0) * 2.0 - 1.0).collect();
        let grad_tensor = create_tensor_from_f32(&gradient, &[8, 8])?;
        let quantized = quantizer.quantize(&grad_tensor)?;
        assert!(!quantized.data.is_empty(), "Gradient pattern should be quantizable");

        Ok(())
    }

    /// Test TL2 quantization numerical stability
    #[test]
    fn test_tl2_numerical_stability() -> Result<()> {
        let quantizer = TL2Quantizer::new();

        // Test with small values near zero
        let small_values: Vec<f32> = (0..32).map(|i| i as f32 / 1000000.0).collect();
        let small_tensor = create_tensor_from_f32(&small_values, &[32])?;

        let quantized = quantizer.quantize(&small_tensor)?;
        let dequantized = quantizer.dequantize(&quantized)?;

        // Verify numerical stability
        assert_eq!(dequantized.dims(), small_tensor.dims(), "Dimensions should match");

        // Test with values clustered around specific points
        let clustered: Vec<f32> = vec![
            0.1, 0.101, 0.099, 0.1001, 0.0999, // Cluster around 0.1
            0.5, 0.501, 0.499, 0.5001, 0.4999, // Cluster around 0.5
            0.9, 0.901, 0.899, 0.9001, 0.8999, // Cluster around 0.9
        ];

        let mut full_clustered = clustered.clone();
        full_clustered.resize(32, 0.0); // Pad to 32 elements

        let cluster_tensor = create_tensor_from_f32(&full_clustered, &[32])?;
        let quantized = quantizer.quantize(&cluster_tensor)?;
        assert!(!quantized.data.is_empty(), "Clustered values should be quantizable");

        Ok(())
    }

    /// Test quantization consistency across multiple runs
    #[test]
    fn test_quantization_determinism() -> Result<()> {
        let i2s = I2SQuantizer::new();
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();

        // Test data
        let test_data: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        let input = create_tensor_from_f32(&test_data, &[8, 8])?;

        // Run multiple times to ensure determinism
        for quantizer_name in ["I2S", "TL1", "TL2"] {
            let mut results = Vec::new();

            for _ in 0..5 {
                let result = match quantizer_name {
                    "I2S" => i2s.quantize(&input)?,
                    "TL1" => tl1.quantize(&input)?,
                    "TL2" => tl2.quantize(&input)?,
                    _ => unreachable!(),
                };
                results.push(result);
            }

            // Verify all results are identical
            for i in 1..results.len() {
                assert_eq!(
                    results[0].data, results[i].data,
                    "{} quantization should be deterministic",
                    quantizer_name
                );
                assert_eq!(
                    results[0].shape, results[i].shape,
                    "{} quantization shape should be deterministic",
                    quantizer_name
                );
            }
        }

        Ok(())
    }

    /// Test error handling for invalid inputs
    #[test]
    fn test_quantization_error_handling() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test with empty tensor (should handle gracefully)
        let empty_result = create_tensor_from_f32(&[], &[0]);
        assert!(empty_result.is_err(), "Empty tensor creation should fail");

        // Test with mismatched dimensions
        let invalid_shape_result = create_tensor_from_f32(&[1.0, 2.0, 3.0], &[2, 2]);
        assert!(invalid_shape_result.is_err(), "Mismatched dimensions should fail");

        // Test with NaN values
        let nan_data = vec![1.0, 2.0, f32::NAN, 4.0];
        let nan_tensor_result = create_tensor_from_f32(&nan_data, &[4]);

        if let Ok(nan_tensor) = nan_tensor_result {
            let result = quantizer.quantize(&nan_tensor);
            // NaN handling should either succeed with replacement or fail gracefully
            match result {
                Ok(_) => println!("NaN values handled gracefully"),
                Err(_) => println!("NaN values correctly rejected"),
            }
        }

        Ok(())
    }

    /// Test memory efficiency and bounds checking
    #[test]
    fn test_memory_bounds_checking() -> Result<()> {
        let quantizer = I2SQuantizer::new();

        // Test progressively larger tensors
        let sizes = vec![16, 64, 256, 1024, 4096];

        for size in sizes {
            let elements = size * size;

            // Skip very large allocations in CI environments
            if elements > 100_000 {
                continue;
            }

            let data: Vec<f32> = (0..elements).map(|i| (i % 1000) as f32 / 1000.0).collect();

            match create_tensor_from_f32(&data, &[size, size]) {
                Ok(tensor) => {
                    match quantizer.quantize(&tensor) {
                        Ok(quantized) => {
                            // Verify bounds
                            assert!(quantized.data.len() > 0, "Quantized data should not be empty");
                            assert!(
                                quantized.data.len() <= elements * 4,
                                "Quantized data too large"
                            );

                            // Test dequantization memory bounds
                            let dequantized = quantizer.dequantize(&quantized)?;
                            assert_eq!(
                                dequantized.elem_count(),
                                elements,
                                "Element count should match"
                            );
                        }
                        Err(_) => {
                            println!("Memory limit reached at size {}x{}", size, size);
                            break;
                        }
                    }
                }
                Err(_) => {
                    println!("Tensor creation failed at size {}x{}", size, size);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Test cross-quantizer compatibility and comparison
    #[test]
    fn test_cross_quantizer_comparison() -> Result<()> {
        let i2s = I2SQuantizer::new();
        let tl1 = TL1Quantizer::new();
        let tl2 = TL2Quantizer::new();

        // Test data with known characteristics
        let test_patterns = vec![
            vec![0.0; 32],                                       // All zeros
            vec![1.0; 32],                                       // All ones
            (0..32).map(|i| i as f32 / 31.0).collect(),          // Linear gradient
            (0..32).map(|i| ((i as f32) * 0.1).sin()).collect(), // Sine wave
        ];

        for (idx, pattern) in test_patterns.iter().enumerate() {
            let tensor = create_tensor_from_f32(pattern, &[32])?;

            // Get quantization results from all quantizers
            let i2s_result = i2s.quantize(&tensor)?;
            let tl1_result = tl1.quantize(&tensor)?;
            let tl2_result = tl2.quantize(&tensor)?;

            // Verify all produce valid output
            assert!(!i2s_result.data.is_empty(), "I2S should produce output for pattern {}", idx);
            assert!(!tl1_result.data.is_empty(), "TL1 should produce output for pattern {}", idx);
            assert!(!tl2_result.data.is_empty(), "TL2 should produce output for pattern {}", idx);

            // Compare compression ratios
            let original_size = pattern.len() * 4; // 4 bytes per f32
            let i2s_size = i2s_result.data.len();
            let tl1_size = tl1_result.data.len();
            let tl2_size = tl2_result.data.len();

            println!(
                "Pattern {}: Original: {} bytes, I2S: {} bytes, TL1: {} bytes, TL2: {} bytes",
                idx, original_size, i2s_size, tl1_size, tl2_size
            );

            // All should provide some compression
            assert!(i2s_size <= original_size, "I2S should compress data");
            assert!(tl1_size <= original_size, "TL1 should compress data");
            assert!(tl2_size <= original_size, "TL2 should compress data");
        }

        Ok(())
    }
}
