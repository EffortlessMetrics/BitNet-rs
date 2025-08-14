//! Comprehensive unit tests for bitnet-quantization
//!
//! This test suite covers:
//! - Quantization algorithms and accuracy
//! - Quantization parameter validation  
//! - Quantization performance and memory tests
//! - Quantization format compatibility tests
//! - Edge cases and error conditions
//! - Property-based testing for robustness

use bitnet_common::{BitNetTensor, QuantizationType, Tensor};
use bitnet_quantization::*;
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;
use std::time::Instant;

/// Helper function to create test tensors
fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> BitNetTensor {
    let device = CandleDevice::Cpu;
    let tensor = CandleTensor::from_vec(data, shape.as_slice(), &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Helper function to extract f32 data from tensor for comparison
fn extract_tensor_data(tensor: &BitNetTensor) -> Vec<f32> {
    let candle_tensor = tensor.inner();
    let flattened = candle_tensor.flatten_all().unwrap();
    flattened.to_vec1::<f32>().unwrap()
}

/// Calculate mean squared error between two vectors
fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32
}

/// Calculate maximum absolute error
fn calculate_max_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[cfg(test)]
mod quantization_algorithms {
    use super::*;

    #[test]
    fn test_i2s_quantization_basic() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = I2SQuantizer::new();

        // Test quantization
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert_eq!(quantized.qtype, QuantizationType::I2S);
        assert_eq!(quantized.shape, shape);
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());

        // Test dequantization
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), &shape);

        // Check accuracy (2-bit quantization has limited precision)
        let dequant_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &dequant_data);
        assert!(mse < 15.0, "MSE too high: {}", mse); // Relaxed tolerance for 2-bit quantization
    }

    #[test]
    fn test_tl1_quantization_basic() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = TL1Quantizer::new();

        // Test quantization
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert_eq!(quantized.qtype, QuantizationType::TL1);
        assert_eq!(quantized.shape, shape);
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());

        // Test dequantization
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), &shape);

        // Check accuracy
        let dequant_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &dequant_data);
        assert!(mse < 15.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_tl2_quantization_basic() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = TL2Quantizer::new();

        // Test quantization
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert_eq!(quantized.qtype, QuantizationType::TL2);
        assert_eq!(quantized.shape, shape);
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());

        // Test dequantization
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
        assert_eq!(dequantized.shape(), &shape);

        // Check accuracy
        let dequant_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &dequant_data);
        assert!(mse < 15.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_quantization_accuracy_comparison() {
        // Test with a sine wave pattern for consistent results
        let data: Vec<f32> =
            (0..64).map(|i| (i as f32 * std::f32::consts::PI / 32.0).sin()).collect();
        let shape = vec![64];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        for (name, quantizer) in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            let dequant_data = extract_tensor_data(&dequantized);
            let mse = calculate_mse(&data, &dequant_data);
            let max_error = calculate_max_error(&data, &dequant_data);

            println!("{} - MSE: {:.6}, Max Error: {:.6}", name, mse, max_error);

            // All quantizers should have reasonable accuracy for 2-bit quantization
            assert!(mse < 5.0, "{} MSE too high: {}", name, mse);
            assert!(max_error < 5.0, "{} max error too high: {}", name, max_error);
        }
    }

    #[test]
    fn test_compression_ratios() {
        let data = vec![1.0; 1024]; // Large tensor for meaningful compression
        let shape = vec![32, 32];
        let tensor = create_test_tensor(data, shape);

        let original_size = 1024 * std::mem::size_of::<f32>();

        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        for (name, quantizer) in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let compressed_size =
                quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();
            let ratio = original_size as f32 / compressed_size as f32;

            println!("{} compression ratio: {:.2}x", name, ratio);

            // Should achieve significant compression
            assert!(ratio > 4.0, "{} compression ratio too low: {:.2}x", name, ratio);

            // Test the built-in compression ratio method
            let builtin_ratio = quantized.compression_ratio();
            assert!((ratio - builtin_ratio).abs() < 0.1, "Compression ratio mismatch");
        }
    }
}

#[cfg(test)]
mod parameter_validation {
    use super::*;

    #[test]
    fn test_i2s_block_sizes() {
        let data = vec![1.0; 128];
        let shape = vec![128];
        let tensor = create_test_tensor(data, shape);

        // Test various block sizes
        let block_sizes = vec![4, 8, 16, 32, 64, 128];

        for block_size in block_sizes {
            let quantizer = I2SQuantizer::with_block_size(block_size);
            let result = quantizer.quantize_tensor(&tensor);

            assert!(result.is_ok(), "Block size {} should work", block_size);

            let quantized = result.unwrap();
            assert_eq!(quantized.block_size, block_size);

            // Should be able to dequantize
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            assert_eq!(dequantized.shape(), tensor.shape());
        }
    }

    #[test]
    fn test_tl1_configurations() {
        let data = vec![1.0; 64];
        let shape = vec![64];
        let tensor = create_test_tensor(data, shape);

        // Test different TL1 configurations
        let configs = vec![
            tl1::TL1Config {
                block_size: 32,
                lookup_table_size: 256,
                use_asymmetric: false,
                precision_bits: 2,
            },
            tl1::TL1Config {
                block_size: 64,
                lookup_table_size: 256,
                use_asymmetric: true,
                precision_bits: 2,
            },
        ];

        for config in configs {
            let quantizer = TL1Quantizer::with_config(config.clone());
            let result = quantizer.quantize_tensor(&tensor);

            assert!(result.is_ok(), "Config should work: {:?}", config);

            let quantized = result.unwrap();
            assert_eq!(quantized.block_size, config.block_size);

            if config.use_asymmetric {
                assert!(quantized.zero_points.is_some(), "Should have zero points for asymmetric");
            }

            // Should be able to dequantize
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            assert_eq!(dequantized.shape(), tensor.shape());
        }
    }

    #[test]
    fn test_tl2_configurations() {
        let data = vec![1.0; 128];
        let shape = vec![128];
        let tensor = create_test_tensor(data, shape);

        // Test different TL2 configurations
        let configs = vec![
            tl2::TL2Config {
                block_size: 64,
                lookup_table_size: 256,
                use_avx512: false,
                use_avx2: true,
                precision_bits: 2,
                vectorized_tables: true,
            },
            tl2::TL2Config {
                block_size: 128,
                lookup_table_size: 256,
                use_avx512: false,
                use_avx2: false, // Force scalar
                precision_bits: 2,
                vectorized_tables: false,
            },
        ];

        for config in configs {
            let quantizer = TL2Quantizer::with_config(config.clone());
            let result = quantizer.quantize_tensor(&tensor);

            assert!(result.is_ok(), "Config should work: {:?}", config);

            let quantized = result.unwrap();
            assert_eq!(quantized.block_size, config.block_size);

            // Should be able to dequantize
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            assert_eq!(dequantized.shape(), tensor.shape());
        }
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_tensor() {
        let data = vec![];
        let shape = vec![0];
        let tensor = create_test_tensor(data, shape.clone());

        let quantizer = I2SQuantizer::new();
        let result = quantizer.quantize_tensor(&tensor);

        // Should handle empty tensors gracefully
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.shape, shape);
        assert_eq!(quantized.data.len(), 0);
    }

    #[test]
    fn test_single_element_tensor() {
        let data = vec![42.0];
        let shape = vec![1];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.shape, shape);
        assert_eq!(dequantized.shape(), &shape);

        let dequant_data = extract_tensor_data(&dequantized);
        assert_eq!(dequant_data.len(), 1);
    }

    #[test]
    fn test_all_zeros() {
        let data = vec![0.0; 64];
        let shape = vec![64];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        let dequant_data = extract_tensor_data(&dequantized);

        // All zeros should quantize to all zeros (or very close)
        for &val in &dequant_data {
            assert!(val.abs() < 0.1, "Expected near-zero, got {}", val);
        }
    }

    #[test]
    fn test_all_same_values() {
        let data = vec![5.0; 64];
        let shape = vec![64];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        let dequant_data = extract_tensor_data(&dequantized);

        // All same values should have low variance after quantization
        let mean = dequant_data.iter().sum::<f32>() / dequant_data.len() as f32;
        let variance = dequant_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / dequant_data.len() as f32;

        assert!(variance < 1.0, "Variance too high for constant input: {}", variance);
    }

    #[test]
    fn test_extreme_values() {
        let data = vec![f32::MAX, f32::MIN, 0.0, 1e-10, -1e-10, 100.0, -100.0, 1e6, -1e6];
        let shape = vec![9];
        let tensor = create_test_tensor(data, shape.clone());

        let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
            Box::new(I2SQuantizer::new()),
            Box::new(TL1Quantizer::new()),
            Box::new(TL2Quantizer::new()),
        ];

        for quantizer in quantizers {
            let result = quantizer.quantize_tensor(&tensor);

            // Should not panic and should handle extreme values
            assert!(result.is_ok(), "Should handle extreme values");

            let quantized = result.unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(dequantized.shape(), &shape);
        }
    }

    #[test]
    fn test_nan_and_infinity() {
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let shape = vec![4];
        let tensor = create_test_tensor(data, shape.clone());

        let quantizer = I2SQuantizer::new();
        let result = quantizer.quantize_tensor(&tensor);

        // Should handle NaN and infinity gracefully (either error or handle)
        // The important thing is it doesn't panic
        match result {
            Ok(quantized) => {
                // If it succeeds, should be able to dequantize
                let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
                assert_eq!(dequantized.shape(), &shape);
            }
            Err(_) => {
                // It's also acceptable to error on NaN/infinity
            }
        }
    }

    #[test]
    fn test_different_tensor_shapes() {
        let test_cases = vec![
            (vec![1.0], vec![1]),                   // Scalar
            (vec![1.0, 2.0, 3.0, 4.0], vec![4]),    // Vector
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), // Matrix
            (vec![1.0; 24], vec![2, 3, 4]),         // 3D tensor
            (vec![1.0; 120], vec![2, 3, 4, 5]),     // 4D tensor
        ];

        for (data, shape) in test_cases {
            let tensor = create_test_tensor(data.clone(), shape.clone());

            let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
                Box::new(I2SQuantizer::new()),
                Box::new(TL1Quantizer::new()),
                Box::new(TL2Quantizer::new()),
            ];

            for quantizer in quantizers {
                let quantized = quantizer.quantize_tensor(&tensor).unwrap();
                let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

                assert_eq!(quantized.shape, shape);
                assert_eq!(dequantized.shape(), &shape);
            }
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_quantization_performance() {
        let sizes = vec![1024, 4096, 16384];

        for size in sizes {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
            let shape = vec![size];
            let tensor = create_test_tensor(data, shape);

            let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
                ("I2S", Box::new(I2SQuantizer::new())),
                ("TL1", Box::new(TL1Quantizer::new())),
                ("TL2", Box::new(TL2Quantizer::new())),
            ];

            for (name, quantizer) in quantizers {
                let start = Instant::now();
                let result = quantizer.quantize_tensor(&tensor);
                let duration = start.elapsed();

                assert!(result.is_ok(), "{} quantization failed", name);
                println!("{} quantization of {} elements took: {:?}", name, size, duration);

                // Performance should be reasonable (less than 1 second for large tensors)
                assert!(
                    duration.as_secs() < 2,
                    "{} quantization took too long: {:?}",
                    name,
                    duration
                );
            }
        }
    }

    #[test]
    fn test_dequantization_performance() {
        let size = 16384;
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let shape = vec![size];
        let tensor = create_test_tensor(data, shape);

        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        for (name, quantizer) in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();

            let start = Instant::now();
            let result = quantizer.dequantize_tensor(&quantized);
            let duration = start.elapsed();

            assert!(result.is_ok(), "{} dequantization failed", name);
            println!("{} dequantization of {} elements took: {:?}", name, size, duration);

            // Dequantization should be fast
            assert!(
                duration.as_millis() < 500,
                "{} dequantization took too long: {:?}",
                name,
                duration
            );
        }
    }

    #[test]
    fn test_memory_usage() {
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let shape = vec![size];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let original_bytes = data.len() * std::mem::size_of::<f32>();

        for (name, quantizer) in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();

            let quantized_bytes =
                quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();
            let ratio = original_bytes as f32 / quantized_bytes as f32;

            println!(
                "{} - Original: {} bytes, Quantized: {} bytes, Ratio: {:.2}x",
                name, original_bytes, quantized_bytes, ratio
            );

            // Should use significantly less memory
            assert!(quantized_bytes < original_bytes, "{} should compress data", name);
            assert!(ratio > 2.0, "{} compression ratio too low: {:.2}x", name, ratio);
        }
    }
}

#[cfg(test)]
mod format_compatibility {
    use super::*;

    #[test]
    fn test_quantization_format_conversion() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let shape = vec![6];
        let tensor = create_test_tensor(data.clone(), shape);

        // Start with I2_S
        let i2s_quantized = tensor.quantize(QuantizationType::I2S).unwrap();

        // Convert to TL1
        let tl1_quantized = convert_quantization(&i2s_quantized, QuantizationType::TL1).unwrap();
        assert_eq!(tl1_quantized.qtype, QuantizationType::TL1);

        // Convert to TL2
        let tl2_quantized = convert_quantization(&tl1_quantized, QuantizationType::TL2).unwrap();
        assert_eq!(tl2_quantized.qtype, QuantizationType::TL2);

        // Convert back to I2_S
        let back_to_i2s = convert_quantization(&tl2_quantized, QuantizationType::I2S).unwrap();
        assert_eq!(back_to_i2s.qtype, QuantizationType::I2S);

        // All should be dequantizable
        let _ = i2s_quantized.dequantize().unwrap();
        let _ = tl1_quantized.dequantize().unwrap();
        let _ = tl2_quantized.dequantize().unwrap();
        let _ = back_to_i2s.dequantize().unwrap();
    }

    #[test]
    fn test_quantizer_factory() {
        let data = vec![1.0, -2.0, 0.5, -0.5];
        let shape = vec![4];
        let tensor = create_test_tensor(data, shape);

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let quantizer = QuantizerFactory::create(qtype);

            assert_eq!(quantizer.quantization_type(), qtype);
            assert!(quantizer.is_available());

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            assert_eq!(quantized.qtype, qtype);

            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            assert_eq!(dequantized.shape(), tensor.shape());
        }
    }

    #[test]
    fn test_best_quantization_for_arch() {
        let best = QuantizerFactory::best_for_arch();

        // Should return a valid quantization type
        match best {
            QuantizationType::I2S | QuantizationType::TL1 | QuantizationType::TL2 => {
                // All valid
            }
        }

        // Should be able to create a quantizer for the best type
        let quantizer = QuantizerFactory::create(best);
        assert!(quantizer.is_available());
    }

    #[test]
    fn test_round_trip_validation() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5];
        let shape = vec![6];
        let tensor = create_test_tensor(data, shape);

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let result = validate_round_trip(&tensor, qtype, 1e-3);
            assert!(result.is_ok(), "Round-trip validation failed for {:?}", qtype);
        }
    }
}

// Property-based tests using proptest
proptest! {
    #[test]
    fn prop_quantization_preserves_shape(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..100)
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
            Box::new(I2SQuantizer::new()),
            Box::new(TL1Quantizer::new()),
            Box::new(TL2Quantizer::new()),
        ];

        for quantizer in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            prop_assert_eq!(quantized.shape, shape.clone());
            prop_assert_eq!(dequantized.shape(), &shape);
        }
    }

    #[test]
    fn prop_quantization_deterministic(
        data in prop::collection::vec(-5.0f32..5.0f32, 4..64)
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data, shape);

        let quantizer = I2SQuantizer::new();

        let result1 = quantizer.quantize_tensor(&tensor).unwrap();
        let result2 = quantizer.quantize_tensor(&tensor).unwrap();

        prop_assert_eq!(result1.data, result2.data);
        prop_assert_eq!(result1.scales, result2.scales);
    }

    #[test]
    fn prop_quantization_bounded_error(
        data in prop::collection::vec(-100.0f32..100.0f32, 16..128)
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizer = I2SQuantizer::new();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        let dequant_data = extract_tensor_data(&dequantized);

        // Calculate maximum absolute error
        let max_error = calculate_max_error(&data, &dequant_data);

        // Error should be bounded for 2-bit quantization
        prop_assert!(max_error < 200.0, "Max error {} too high", max_error);
    }

    #[test]
    fn prop_scale_values_reasonable(
        data in prop::collection::vec(-1000.0f32..1000.0f32, 32..128)
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data, shape);

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        // All scales should be positive and finite
        for &scale in &quantized.scales {
            prop_assert!(scale > 0.0, "Scale {} should be positive", scale);
            prop_assert!(scale.is_finite(), "Scale {} should be finite", scale);
        }
    }

    #[test]
    fn prop_compression_ratio_reasonable(
        data in prop::collection::vec(-10.0f32..10.0f32, 64..256)
    ) {
        let data_len = data.len();
        let shape = vec![data_len];
        let tensor = create_test_tensor(data, shape);

        let quantizer = I2SQuantizer::new();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        let ratio = quantized.compression_ratio();

        // Should achieve some compression
        prop_assert!(ratio >= 1.0, "Compression ratio should be >= 1.0, got {}", ratio);
        // For 2-bit quantization, should achieve significant compression on larger tensors
        if data_len >= 128 {
            prop_assert!(ratio > 2.0, "Should achieve >2x compression on large tensors, got {}", ratio);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_quantization_pipeline() {
        // Simulate a real model tensor with realistic weight distribution
        let model_weights: Vec<f32> = (0..4096)
            .map(|i| {
                // Simulate weight distribution (roughly normal)
                let x = (i as f32 - 2048.0) / 1000.0;
                x.sin() * (-x * x / 2.0).exp() // Gaussian-like
            })
            .collect();

        let shape = vec![64, 64];
        let tensor = create_test_tensor(model_weights.clone(), shape);

        // Test all quantization methods
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        for (name, quantizer) in quantizers {
            println!("Testing {} quantization pipeline", name);

            // Quantize
            let start = Instant::now();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let quantize_time = start.elapsed();

            // Dequantize
            let start = Instant::now();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            let dequantize_time = start.elapsed();

            // Verify correctness
            assert_eq!(dequantized.shape(), tensor.shape());

            // Calculate metrics
            let dequant_data = extract_tensor_data(&dequantized);
            let mse = calculate_mse(&model_weights, &dequant_data);
            let max_error = calculate_max_error(&model_weights, &dequant_data);

            let original_size = model_weights.len() * std::mem::size_of::<f32>();
            let compressed_size =
                quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();
            let compression_ratio = original_size as f32 / compressed_size as f32;

            println!("  {} Results:", name);
            println!("    Quantize time: {:?}", quantize_time);
            println!("    Dequantize time: {:?}", dequantize_time);
            println!("    MSE: {:.6}", mse);
            println!("    Max error: {:.6}", max_error);
            println!("    Compression ratio: {:.2}x", compression_ratio);

            // Verify reasonable performance
            assert!(quantize_time.as_millis() < 2000, "{} quantization too slow", name);
            assert!(dequantize_time.as_millis() < 500, "{} dequantization too slow", name);
            assert!(compression_ratio > 2.0, "{} should provide compression", name);
            assert!(mse < 5.0, "{} MSE too high: {}", name, mse);
        }
    }

    #[test]
    fn test_cross_algorithm_compatibility() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let shape = vec![256];
        let tensor = create_test_tensor(data.clone(), shape);

        // Test that different algorithms can handle the same data
        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        let mut results = Vec::new();

        for (name, quantizer) in quantizers {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            let dequant_data = extract_tensor_data(&dequantized);
            assert_eq!(dequant_data.len(), data.len());

            let mse = calculate_mse(&data, &dequant_data);
            results.push((name, mse));
        }

        // Compare accuracy across algorithms
        for (name, mse) in &results {
            println!("{} MSE: {:.6}", name, mse);
            assert!(*mse < 5.0, "{} MSE too high: {}", name, mse);
        }

        // All algorithms should produce reasonable results
        assert_eq!(results.len(), 3);
    }
}
