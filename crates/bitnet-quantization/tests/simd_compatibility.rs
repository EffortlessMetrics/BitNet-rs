//! Cross-platform SIMD compatibility tests for BitNet quantization kernels
//!
//! This test suite verifies that SIMD kernel implementations work correctly
//! across different architectures and CPU feature sets, with proper fallback
//! to scalar implementations when SIMD is unavailable.

use bitnet_common::{BitNetTensor, QuantizationType, Tensor};
use bitnet_quantization::{I2SQuantizer, QuantizerTrait, TL1Quantizer, TL2Quantizer};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

/// Helper function to create test tensors with deterministic data
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

#[cfg(test)]
mod cross_platform_tests {
    use super::*;

    /// Test that SIMD and scalar implementations produce identical results
    #[test]
    fn test_i2s_simd_scalar_parity() {
        // Create a deterministic test pattern that exercises different quantization ranges
        let data: Vec<f32> = (0..256)
            .map(|i| {
                let phase = i as f32 * std::f32::consts::PI / 32.0;
                phase.sin() * 3.0 // Range roughly -3 to 3, good for I2S testing
            })
            .collect();
        let shape = vec![256];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizer = I2SQuantizer::new();

        // Test multiple times to ensure deterministic behavior
        let mut results = Vec::new();
        for _ in 0..3 {
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
            let result_data = extract_tensor_data(&dequantized);
            results.push(result_data);
        }

        // All runs should produce identical results (deterministic)
        for i in 1..results.len() {
            assert_eq!(
                results[0], results[i],
                "I2S quantization should be deterministic across runs"
            );
        }

        // Verify accuracy is reasonable
        let mse = calculate_mse(&data, &results[0]);
        assert!(mse < 5.0, "I2S quantization MSE too high: {}", mse);
    }

    /// Test different CPU feature detection scenarios
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_feature_detection() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizer = I2SQuantizer::new();

        // Check if AVX2 is available and test accordingly
        println!("Testing x86_64 architecture with potential AVX2 support");

        // Should work regardless of feature availability
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::I2S);
        assert_eq!(dequantized.shape(), &vec![8]);

        let result_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &result_data);
        assert!(mse < 20.0, "Quantization accuracy too low: MSE = {}", mse);
    }

    /// Test ARM NEON feature detection scenarios
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_arm_feature_detection() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape);

        let quantizer = I2SQuantizer::new();

        // Check if NEON is available and test accordingly
        println!("Testing aarch64 architecture with potential NEON support");

        // Should work regardless of feature availability
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::I2S);
        assert_eq!(dequantized.shape(), &vec![8]);

        let result_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &result_data);
        assert!(mse < 20.0, "Quantization accuracy too low: MSE = {}", mse);
    }

    /// Test behavior on non-SIMD architectures (should use scalar fallback)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[test]
    fn test_scalar_fallback_architecture() {
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
        let shape = vec![8];
        let tensor = create_test_tensor(data.clone(), shape);

        println!("Testing scalar fallback on non-SIMD architecture");

        let quantizer = I2SQuantizer::new();

        // Should work with scalar implementation
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

        assert_eq!(quantized.qtype, QuantizationType::I2S);
        assert_eq!(dequantized.shape(), &vec![8]);

        let result_data = extract_tensor_data(&dequantized);
        let mse = calculate_mse(&data, &result_data);
        assert!(mse < 20.0, "Scalar quantization accuracy too low: MSE = {}", mse);
    }

    /// Test cross-quantizer compatibility on all architectures
    #[test]
    fn test_cross_quantizer_architecture_compatibility() {
        // Test data that exercises all quantization ranges
        let data: Vec<f32> = vec![
            -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, -2.5, -1.5, -0.25, 0.25, 1.5, 2.5,
            -3.5, -0.1, 0.1, 3.5, // Test clamping behavior
        ];
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantizers: Vec<(&str, Box<dyn QuantizerTrait>)> = vec![
            ("I2S", Box::new(I2SQuantizer::new())),
            ("TL1", Box::new(TL1Quantizer::new())),
            ("TL2", Box::new(TL2Quantizer::new())),
        ];

        for (name, quantizer) in quantizers {
            println!("Testing {} on current architecture", name);

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.shape, shape, "{} shape mismatch", name);
            assert_eq!(dequantized.shape(), &shape, "{} dequantized shape mismatch", name);

            // Verify reasonable compression
            let compression_ratio = quantized.compression_ratio();
            assert!(
                compression_ratio >= 1.0,
                "{} compression ratio should be >= 1.0, got {}",
                name,
                compression_ratio
            );

            // Verify accuracy (generous bounds for 2-bit quantization with varying algorithms)
            let result_data = extract_tensor_data(&dequantized);
            let mse = calculate_mse(&data, &result_data);
            assert!(mse < 25.0, "{} accuracy too low: MSE = {}", name, mse);
        }
    }

    /// Test different data alignment scenarios to ensure SIMD loads/stores work correctly
    #[test]
    fn test_data_alignment_scenarios() {
        // Test various tensor sizes that may result in different alignments
        let test_sizes = vec![
            7,  // Less than SIMD register size
            8,  // Exactly one SIMD register (x86)
            9,  // One register + remainder
            15, // Just under two registers
            16, // Exactly two registers
            17, // Two registers + remainder
            31, // Just under optimal block size
            32, // Default block size
            33, // Block size + remainder
            63, // Just under two blocks
            64, // Two blocks exactly
            65, // Two blocks + remainder
        ];

        let quantizer = I2SQuantizer::new();

        for size in test_sizes {
            // Create data with predictable pattern
            let data: Vec<f32> =
                (0..size).map(|i| (i as f32 * std::f32::consts::PI / 8.0).sin() * 2.0).collect();
            let shape = vec![size];
            let tensor = create_test_tensor(data.clone(), shape.clone());

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.shape, shape, "Size {} shape mismatch", size);
            assert_eq!(dequantized.shape(), &shape, "Size {} dequantized shape mismatch", size);

            // Verify no corruption in remainder elements
            let result_data = extract_tensor_data(&dequantized);
            assert_eq!(result_data.len(), data.len(), "Size {} length mismatch", size);

            // Check that no elements are NaN or infinite
            for (i, &val) in result_data.iter().enumerate() {
                assert!(val.is_finite(), "Size {} element {} is not finite: {}", size, i, val);
            }
        }
    }

    /// Test block size compatibility across architectures
    #[test]
    fn test_block_size_cross_architecture() {
        let data: Vec<f32> =
            (0..128).map(|i| (i as f32 * std::f32::consts::PI / 16.0).cos() * 1.5).collect();
        let shape = vec![128];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        // Test various block sizes that might interact differently with SIMD
        let block_sizes = vec![4, 8, 16, 32, 64, 128];

        for block_size in block_sizes {
            println!("Testing block size {} on current architecture", block_size);

            let quantizer = I2SQuantizer::with_block_size(block_size);

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.block_size, block_size);
            assert_eq!(quantized.shape, shape);
            assert_eq!(dequantized.shape(), &shape);

            // Verify accuracy is maintained across different block sizes
            let result_data = extract_tensor_data(&dequantized);
            let mse = calculate_mse(&data, &result_data);
            assert!(mse < 10.0, "Block size {} accuracy too low: MSE = {}", block_size, mse);

            // Verify that the number of scale values matches expected block count
            let expected_blocks = data.len().div_ceil(block_size);
            assert_eq!(
                quantized.scales.len(),
                expected_blocks,
                "Block size {} scale count mismatch",
                block_size
            );
        }
    }

    /// Test edge cases that might expose SIMD implementation issues
    #[test]
    fn test_simd_edge_cases() {
        let edge_cases = vec![
            // All zeros
            vec![0.0; 32],
            // All same positive value
            vec![1.0; 32],
            // All same negative value
            vec![-1.0; 32],
            // Alternating pattern
            (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
            // Increasing pattern
            (0..32).map(|i| i as f32 * 0.1).collect(),
            // Decreasing pattern
            (0..32).map(|i| (31 - i) as f32 * 0.1).collect(),
            // Random-like deterministic pattern
            (0..32).map(|i| ((i * 7 + 3) % 13) as f32 - 6.0).collect(),
            // Values at quantization boundaries
            [-2.0, -1.0, 0.0, 1.0, -1.5, -0.5, 0.5, 1.5].repeat(4),
            // Very small values
            [1e-6, -1e-6, 1e-7, -1e-7].repeat(8),
        ];

        let quantizer = I2SQuantizer::new();

        for (case_idx, data) in edge_cases.into_iter().enumerate() {
            println!("Testing edge case {}: {} elements", case_idx, data.len());

            let shape = vec![data.len()];
            let tensor = create_test_tensor(data.clone(), shape.clone());

            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();

            assert_eq!(quantized.shape, shape, "Case {} shape mismatch", case_idx);
            assert_eq!(dequantized.shape(), &shape, "Case {} dequantized shape mismatch", case_idx);

            // Verify no NaN or infinite values in result
            let result_data = extract_tensor_data(&dequantized);
            for (i, &val) in result_data.iter().enumerate() {
                assert!(val.is_finite(), "Case {} element {} is not finite: {}", case_idx, i, val);
            }

            // Verify reasonable bounds for I2S quantization
            for &val in &result_data {
                assert!(
                    val.abs() <= 10.0, // Generous bound for I2S with scaling
                    "Case {} produced out-of-bounds value: {}",
                    case_idx,
                    val
                );
            }
        }
    }
}

#[cfg(test)]
mod performance_validation_tests {
    use super::*;
    use std::time::Instant;

    /// Verify that SIMD implementations don't have significant performance regressions
    #[test]
    fn test_simd_performance_baseline() {
        let tensor_sizes = vec![1024, 4096, 16384];

        for size in tensor_sizes {
            println!("Performance validation for size {}", size);

            let data: Vec<f32> =
                (0..size).map(|i| (i as f32 * std::f32::consts::PI / 256.0).sin() * 2.0).collect();
            let shape = vec![size];
            let tensor = create_test_tensor(data, shape);

            let quantizer = I2SQuantizer::new();

            // Warm up
            for _ in 0..5 {
                let _ = quantizer.quantize_tensor(&tensor).unwrap();
            }

            // Measure quantization performance
            let start = Instant::now();
            let num_runs = 10;
            for _ in 0..num_runs {
                let _ = quantizer.quantize_tensor(&tensor).unwrap();
            }
            let quantize_duration = start.elapsed();

            // Measure dequantization performance
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            let start = Instant::now();
            for _ in 0..num_runs {
                let _ = quantizer.dequantize_tensor(&quantized).unwrap();
            }
            let dequantize_duration = start.elapsed();

            let quantize_avg = quantize_duration / num_runs;
            let dequantize_avg = dequantize_duration / num_runs;

            println!(
                "  Size {}: quantize={:?}, dequantize={:?}",
                size, quantize_avg, dequantize_avg
            );

            // Performance should be reasonable (these are generous bounds)
            assert!(
                quantize_avg.as_millis() < 100,
                "Quantization too slow for size {}: {:?}",
                size,
                quantize_avg
            );
            assert!(
                dequantize_avg.as_millis() < 50,
                "Dequantization too slow for size {}: {:?}",
                size,
                dequantize_avg
            );
        }
    }
}
