//! Comprehensive tests for bitnet-quantization
//! Covers edge cases, error conditions, and end-to-end scenarios

use bitnet_common::{BitNetTensor, ConcreteTensor, MockTensor, QuantizationType};
use bitnet_quantization::*;
use candle_core::{DType, Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;

/// Helper function to create test tensors
fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> BitNetTensor {
    let device = CandleDevice::Cpu;
    let tensor = CandleTensor::from_vec(data, shape.as_slice(), &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Test error conditions and edge cases
mod error_handling {
    use super::*;

    #[test]
    fn test_invalid_block_sizes() {
        // Test zero block size
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 0,
            precision: 1e-4,
        });
        assert!(result.is_err());

        // Test non-power-of-2 block size
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 63, // Not power of 2
            precision: 1e-4,
        });
        assert!(result.is_err());

        // Test extremely large block size
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 1024 * 1024, // Very large
            precision: 1e-4,
        });
        // Should succeed but might be impractical
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_precision() {
        // Test zero precision
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 64,
            precision: 0.0,
        });
        assert!(result.is_err());

        // Test negative precision
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 64,
            precision: -1e-4,
        });
        assert!(result.is_err());

        // Test extremely small precision
        let result = I2SQuantizer::new_with_config(I2SConfig {
            block_size: 64,
            precision: 1e-20,
        });
        // Should succeed
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_tensor_quantization() {
        let quantizer = I2SQuantizer::new().unwrap();
        let empty_tensor = MockTensor::new(vec![]);

        let result = quantizer.quantize(&empty_tensor);
        // Should handle empty tensors gracefully
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert_eq!(quantized.data.len(), 0);
        assert_eq!(quantized.scales.len(), 0);
    }

    #[test]
    fn test_single_element_tensor() {
        let quantizer = I2SQuantizer::new().unwrap();
        let tensor = MockTensor::new(vec![42.0]);

        let result = quantizer.quantize(&tensor);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        assert!(!quantized.data.is_empty());
        assert!(!quantized.scales.is_empty());
    }

    #[test]
    fn test_mismatched_tensor_dimensions() {
        let quantizer = I2SQuantizer::new().unwrap();

        // Create tensor with shape that doesn't match data length
        let data = vec![1.0f32; 10];
        let tensor = ConcreteTensor::new(data, vec![3, 4], DType::F32); // 3*4=12 but data has 10 elements

        // This should be caught by the tensor implementation
        // The test verifies our quantizer handles such cases gracefully
        let result = quantizer.quantize(&tensor);
        // Depending on implementation, this might succeed or fail
        // The important thing is it doesn't panic
    }

    #[test]
    fn test_extreme_values() {
        let quantizer = I2SQuantizer::new().unwrap();

        // Test with very large values
        let large_values = vec![f32::MAX, f32::MAX / 2.0, f32::MAX / 4.0, f32::MAX / 8.0];
        let tensor = MockTensor::new(large_values);
        let result = quantizer.quantize(&tensor);
        assert!(result.is_ok());

        // Test with very small values
        let small_values = vec![
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 2.0,
            f32::MIN_POSITIVE * 4.0,
            f32::MIN_POSITIVE * 8.0,
        ];
        let tensor = MockTensor::new(small_values);
        let result = quantizer.quantize(&tensor);
        assert!(result.is_ok());

        // Test with infinity
        let inf_values = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];
        let tensor = MockTensor::new(inf_values);
        let result = quantizer.quantize(&tensor);
        // Should handle infinity gracefully (might clamp or error)
        // The important thing is it doesn't panic
    }

    #[test]
    fn test_nan_values() {
        let quantizer = I2SQuantizer::new().unwrap();

        let nan_values = vec![f32::NAN, 1.0, 2.0, 3.0];
        let tensor = MockTensor::new(nan_values);
        let result = quantizer.quantize(&tensor);

        // NaN handling should be well-defined
        // Either error or handle gracefully, but no panic
    }

    #[test]
    fn test_all_zero_tensor() {
        let quantizer = I2SQuantizer::new().unwrap();

        let zero_tensor = MockTensor::new(vec![0.0f32; 64]);
        let result = quantizer.quantize(&zero_tensor);
        assert!(result.is_ok());

        let quantized = result.unwrap();
        // All zeros should quantize to all zeros
        assert!(quantized.data.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_all_same_value_tensor() {
        let quantizer = I2SQuantizer::new().unwrap();

        let same_value_tensor = MockTensor::new(vec![64]);
        let result = quantizer.quantize(&same_value_tensor);
        assert!(result.is_ok());

        // All same values should have zero variance
        // Quantization should handle this case
    }
}

/// Test different quantization algorithms comprehensively
mod algorithm_comprehensive {
    use super::*;

    #[test]
    fn test_i2s_comprehensive() {
        let quantizer = I2SQuantizer::new().unwrap();

        // Test with different data patterns
        let patterns = vec![
            // Linear sequence
            (0..64).map(|i| i as f32).collect::<Vec<_>>(),
            // Sine wave
            (0..64).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>(),
            // Random-like pattern
            (0..64)
                .map(|i| ((i * 17 + 23) % 100) as f32 / 100.0)
                .collect::<Vec<_>>(),
            // Exponential decay
            (0..64).map(|i| (-i as f32 * 0.1).exp()).collect::<Vec<_>>(),
        ];

        for (i, pattern) in patterns.iter().enumerate() {
            let tensor = MockTensor::new(pattern.clone());
            let result = quantizer.quantize(&tensor);
            assert!(result.is_ok(), "Pattern {} failed", i);

            let quantized = result.unwrap();
            assert!(!quantized.data.is_empty());
            assert!(!quantized.scales.is_empty());

            // Test dequantization
            let dequantized = quantizer.dequantize(&quantized).unwrap();
            assert_eq!(dequantized.len(), pattern.len());

            // Check that dequantized values are reasonably close to original
            let mse: f32 = pattern
                .iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).powi(2))
                .sum::<f32>()
                / pattern.len() as f32;

            // MSE should be reasonable (this is lossy compression)
            assert!(mse < 1.0, "MSE too high for pattern {}: {}", i, mse);
        }
    }

    #[test]
    fn test_tl1_comprehensive() {
        let quantizer = TL1Quantizer::new().unwrap();

        // Test with different block sizes
        let block_sizes = vec![16, 32, 64, 128];

        for block_size in block_sizes {
            let config = TL1Config {
                block_size,
                precision: 1e-4,
                symmetric: true,
            };

            let quantizer = TL1Quantizer::new_with_config(config).unwrap();
            let data: Vec<f32> = (0..block_size * 4)
                .map(|i| (i as f32 - block_size as f32 * 2.0) / 10.0)
                .collect();
            let tensor = MockTensor::new(data.clone());

            let result = quantizer.quantize(&tensor);
            assert!(result.is_ok(), "Block size {} failed", block_size);

            let quantized = result.unwrap();
            let dequantized = quantizer.dequantize(&quantized).unwrap();

            // Verify round-trip accuracy
            let max_error = data
                .iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).abs())
                .fold(0.0f32, f32::max);

            assert!(
                max_error < 0.5,
                "Max error too high for block size {}: {}",
                block_size,
                max_error
            );
        }
    }

    #[test]
    fn test_tl2_comprehensive() {
        let quantizer = TL2Quantizer::new().unwrap();

        // Test with different precision settings
        let precisions = vec![1e-3, 1e-4, 1e-5, 1e-6];

        for precision in precisions {
            let config = TL2Config {
                block_size: 64,
                precision,
                use_vectorized: true,
            };

            let quantizer = TL2Quantizer::new_with_config(config).unwrap();
            let data: Vec<f32> = (0..256).map(|i| (i as f32).sin() * 10.0).collect();
            let tensor = MockTensor::new(data.clone());

            let result = quantizer.quantize(&tensor);
            assert!(result.is_ok(), "Precision {} failed", precision);

            let quantized = result.unwrap();
            let dequantized = quantizer.dequantize(&quantized).unwrap();

            // Higher precision should give better accuracy
            let mse: f32 = data
                .iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).powi(2))
                .sum::<f32>()
                / data.len() as f32;

            // MSE should be inversely related to precision
            let expected_mse = precision as f32 * 1000.0; // Rough heuristic
            assert!(
                mse < expected_mse,
                "MSE {} too high for precision {}",
                mse,
                precision
            );
        }
    }

    #[test]
    fn test_quantization_compression_ratios() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let tensor = MockTensor::new(data.clone());

        let original_size = data.len() * std::mem::size_of::<f32>();

        // Test I2S compression
        let i2s_quantizer = I2SQuantizer::new().unwrap();
        let i2s_result = i2s_quantizer.quantize(&tensor).unwrap();
        let i2s_size = i2s_result.data.len() + i2s_result.scales.len() * std::mem::size_of::<f32>();
        let i2s_ratio = original_size as f32 / i2s_size as f32;

        // Test TL1 compression
        let tl1_quantizer = TL1Quantizer::new().unwrap();
        let tl1_result = tl1_quantizer.quantize(&tensor).unwrap();
        let tl1_size = tl1_result.data.len() + tl1_result.scales.len() * std::mem::size_of::<f32>();
        let tl1_ratio = original_size as f32 / tl1_size as f32;

        // Test TL2 compression
        let tl2_quantizer = TL2Quantizer::new().unwrap();
        let tl2_result = tl2_quantizer.quantize(&tensor).unwrap();
        let tl2_size = tl2_result.data.len() + tl2_result.scales.len() * std::mem::size_of::<f32>();
        let tl2_ratio = original_size as f32 / tl2_size as f32;

        println!(
            "Compression ratios - I2S: {:.2}x, TL1: {:.2}x, TL2: {:.2}x",
            i2s_ratio, tl1_ratio, tl2_ratio
        );

        // All should provide some compression
        assert!(i2s_ratio > 1.0);
        assert!(tl1_ratio > 1.0);
        assert!(tl2_ratio > 1.0);
    }
}

/// Test performance characteristics
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_quantization_performance() {
        let sizes = vec![1024, 4096, 16384, 65536];

        for size in sizes {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
            let tensor = MockTensor::new(data);

            // Test I2S performance
            let quantizer = I2SQuantizer::new().unwrap();
            let start = Instant::now();
            let result = quantizer.quantize(&tensor);
            let duration = start.elapsed();

            assert!(result.is_ok());
            println!("I2S quantization of {} elements took: {:?}", size, duration);

            // Performance should be reasonable (less than 1 second for large tensors)
            assert!(
                duration.as_secs() < 1,
                "Quantization took too long: {:?}",
                duration
            );
        }
    }

    #[test]
    fn test_dequantization_performance() {
        let size = 16384;
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let tensor = MockTensor::new(data);

        let quantizer = I2SQuantizer::new().unwrap();
        let quantized = quantizer.quantize(&tensor).unwrap();

        // Test dequantization performance
        let start = Instant::now();
        let result = quantizer.dequantize(&quantized);
        let duration = start.elapsed();

        assert!(result.is_ok());
        println!(
            "I2S dequantization of {} elements took: {:?}",
            size, duration
        );

        // Dequantization should be fast
        assert!(
            duration.as_millis() < 100,
            "Dequantization took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_memory_usage() {
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = MockTensor::new(data.clone());

        let quantizer = I2SQuantizer::new().unwrap();
        let quantized = quantizer.quantize(&tensor).unwrap();

        // Check memory usage
        let original_bytes = data.len() * std::mem::size_of::<f32>();
        let quantized_bytes =
            quantized.data.len() + quantized.scales.len() * std::mem::size_of::<f32>();

        println!(
            "Original: {} bytes, Quantized: {} bytes, Ratio: {:.2}x",
            original_bytes,
            quantized_bytes,
            original_bytes as f32 / quantized_bytes as f32
        );

        // Should use less memory
        assert!(quantized_bytes < original_bytes);
    }
}

/// Property-based tests for comprehensive coverage
mod property_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_quantization_preserves_shape(
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..1000)
        ) {
            let tensor = MockTensor::new(data.clone());
            let quantizer = I2SQuantizer::new().unwrap();

            let quantized = quantizer.quantize(&tensor).unwrap();
            let dequantized = quantizer.dequantize(&quantized).unwrap();

            prop_assert_eq!(dequantized.len(), data.len());
        }

        #[test]
        fn test_quantization_deterministic(
            data in prop::collection::vec(prop::num::f32::NORMAL, 1..100)
        ) {
            let tensor = MockTensor::new(data);
            let quantizer = I2SQuantizer::new().unwrap();

            let result1 = quantizer.quantize(&tensor).unwrap();
            let result2 = quantizer.quantize(&tensor).unwrap();

            prop_assert_eq!(result1.data, result2.data);
            prop_assert_eq!(result1.scales, result2.scales);
        }

        #[test]
        fn test_quantization_bounded_error(
            data in prop::collection::vec(-100.0f32..100.0f32, 64..256)
        ) {
            let tensor = MockTensor::new(data.clone());
            let quantizer = I2SQuantizer::new().unwrap();

            let quantized = quantizer.quantize(&tensor).unwrap();
            let dequantized = quantizer.dequantize(&quantized).unwrap();

            // Calculate maximum absolute error
            let max_error = data.iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).abs())
                .fold(0.0f32, f32::max);

            // Error should be bounded (this is lossy compression)
            prop_assert!(max_error < 10.0, "Max error {} too high", max_error);
        }

        #[test]
        fn test_scale_values_reasonable(
            data in prop::collection::vec(-1000.0f32..1000.0f32, 64..256)
        ) {
            let tensor = MockTensor::new(data);
            let quantizer = I2SQuantizer::new().unwrap();

            let quantized = quantizer.quantize(&tensor).unwrap();

            // All scales should be positive and finite
            for &scale in &quantized.scales {
                prop_assert!(scale > 0.0, "Scale {} should be positive", scale);
                prop_assert!(scale.is_finite(), "Scale {} should be finite", scale);
            }
        }
    }
}

/// End-to-end integration tests
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_quantization_pipeline() {
        // Simulate a real model tensor
        let model_weights: Vec<f32> = (0..4096)
            .map(|i| {
                // Simulate weight distribution (roughly normal)
                let x = (i as f32 - 2048.0) / 1000.0;
                x.sin() * (-x * x / 2.0).exp() // Gaussian-like
            })
            .collect();

        let tensor = MockTensor::new(model_weights.clone());

        // Test all quantization methods
        let methods = vec![
            (
                "I2S",
                Box::new(I2SQuantizer::new().unwrap()) as Box<dyn Quantize>,
            ),
            (
                "TL1",
                Box::new(TL1Quantizer::new().unwrap()) as Box<dyn Quantize>,
            ),
            (
                "TL2",
                Box::new(TL2Quantizer::new().unwrap()) as Box<dyn Quantize>,
            ),
        ];

        for (name, quantizer) in methods {
            println!("Testing {} quantization pipeline", name);

            // Quantize
            let start = std::time::Instant::now();
            let quantized = quantizer.quantize(&tensor).unwrap();
            let quantize_time = start.elapsed();

            // Dequantize
            let start = std::time::Instant::now();
            let dequantized = quantizer.dequantize(&quantized).unwrap();
            let dequantize_time = start.elapsed();

            // Verify correctness
            assert_eq!(dequantized.len(), model_weights.len());

            // Calculate metrics
            let mse: f32 = model_weights
                .iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).powi(2))
                .sum::<f32>()
                / model_weights.len() as f32;

            let max_error = model_weights
                .iter()
                .zip(dequantized.iter())
                .map(|(orig, deq)| (orig - deq).abs())
                .fold(0.0f32, f32::max);

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
            assert!(
                quantize_time.as_millis() < 1000,
                "{} quantization too slow",
                name
            );
            assert!(
                dequantize_time.as_millis() < 100,
                "{} dequantization too slow",
                name
            );
            assert!(
                compression_ratio > 1.0,
                "{} should provide compression",
                name
            );
            assert!(mse < 1.0, "{} MSE too high: {}", name, mse);
        }
    }

    #[test]
    fn test_cross_algorithm_compatibility() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let tensor = MockTensor::new(data.clone());

        // Test that different algorithms can handle the same data
        let i2s_quantizer = I2SQuantizer::new().unwrap();
        let tl1_quantizer = TL1Quantizer::new().unwrap();
        let tl2_quantizer = TL2Quantizer::new().unwrap();

        let i2s_result = i2s_quantizer.quantize(&tensor).unwrap();
        let tl1_result = tl1_quantizer.quantize(&tensor).unwrap();
        let tl2_result = tl2_quantizer.quantize(&tensor).unwrap();

        let i2s_deq = i2s_quantizer.dequantize(&i2s_result).unwrap();
        let tl1_deq = tl1_quantizer.dequantize(&tl1_result).unwrap();
        let tl2_deq = tl2_quantizer.dequantize(&tl2_result).unwrap();

        // All should produce valid results
        assert_eq!(i2s_deq.len(), data.len());
        assert_eq!(tl1_deq.len(), data.len());
        assert_eq!(tl2_deq.len(), data.len());

        // Compare accuracy
        let i2s_mse: f32 = data
            .iter()
            .zip(i2s_deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        let tl1_mse: f32 = data
            .iter()
            .zip(tl1_deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        let tl2_mse: f32 = data
            .iter()
            .zip(tl2_deq.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        println!(
            "Cross-algorithm MSE comparison - I2S: {:.6}, TL1: {:.6}, TL2: {:.6}",
            i2s_mse, tl1_mse, tl2_mse
        );

        // All should have reasonable accuracy
        assert!(i2s_mse < 0.1);
        assert!(tl1_mse < 0.1);
        assert!(tl2_mse < 0.1);
    }
}
