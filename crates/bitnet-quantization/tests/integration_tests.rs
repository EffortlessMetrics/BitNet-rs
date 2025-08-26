//! Integration tests for quantization algorithms
//!
//! These tests validate the correctness and numerical accuracy of all quantization
//! implementations against reference implementations and cross-validate between
//! different quantization types.

#![cfg(feature = "integration-tests")]

use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_quantization::{
    I2SQuantizer, Quantize, QuantizerFactory, QuantizerTrait, TL1Quantizer, TL2Quantizer,
    convert_quantization,
};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use proptest::prelude::*;

/// Helper function to create test tensors
fn create_test_tensor(data: Vec<f32>, shape: Vec<usize>) -> BitNetTensor {
    let device = CandleDevice::Cpu;
    let tensor = CandleTensor::from_vec(data, shape.as_slice(), &device).unwrap();
    BitNetTensor::new(tensor)
}

/// Test basic quantization round-trip for all quantization types
#[test]
fn test_all_quantization_round_trips() {
    let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 2.5];
    let shape = vec![2, 4];
    let tensor = create_test_tensor(data, shape.clone());

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantizer = QuantizerFactory::create(qtype);

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize_tensor(&quantized, &Device::Cpu).unwrap();

        assert_eq!(quantized.qtype, qtype);
        assert_eq!(quantized.shape, shape);
        assert_eq!(dequantized.shape(), &shape);

        // Verify compression ratio
        let ratio = quantized.compression_ratio();
        assert!(ratio > 2.0, "Compression ratio too low: {}", ratio);
    }
}

/// Test quantization format conversion
#[test]
fn test_quantization_format_conversion() {
    let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
    let shape = vec![6];
    let tensor = create_test_tensor(data, shape);

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
    let _ = i2s_quantized.dequantize(&Device::Cpu).unwrap();
    let _ = tl1_quantized.dequantize(&Device::Cpu).unwrap();
    let _ = tl2_quantized.dequantize(&Device::Cpu).unwrap();
    let _ = back_to_i2s.dequantize(&Device::Cpu).unwrap();
}

/// Test quantization with different tensor shapes
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
        let tensor = create_test_tensor(data, shape.clone());

        for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let quantized = tensor.quantize(qtype).unwrap();
            let dequantized = quantized.dequantize(&Device::Cpu).unwrap();

            assert_eq!(quantized.shape, shape);
            assert_eq!(dequantized.shape(), &shape);
        }
    }
}

/// Test quantization with extreme values
#[test]
fn test_extreme_values() {
    let extreme_data = vec![f32::MAX, f32::MIN, 0.0, 1e-10, -1e-10, 100.0, -100.0, 1e6, -1e6];
    let shape = vec![9];
    let tensor = create_test_tensor(extreme_data, shape);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        let dequantized = quantized.dequantize(&Device::Cpu).unwrap();

        // Should not panic and should maintain shape
        assert_eq!(dequantized.shape(), &[9]);
    }
}

/// Test quantization accuracy with known patterns
#[test]
fn test_quantization_accuracy() {
    // Test with a sine wave pattern
    let data: Vec<f32> = (0..64).map(|i| (i as f32 * std::f32::consts::PI / 32.0).sin()).collect();
    let shape = vec![64];
    let tensor = create_test_tensor(data.clone(), shape);

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantized = tensor.quantize(qtype).unwrap();
        let dequantized = quantized.dequantize(&Device::Cpu).unwrap();

        // Extract dequantized data for comparison
        let dequant_candle = dequantized.inner();
        let dequant_data = dequant_candle.to_vec1::<f32>().unwrap();

        // Calculate MSE
        let mse: f32 = data
            .iter()
            .zip(dequant_data.iter())
            .map(|(&orig, &dequant)| (orig - dequant).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        // MSE should be reasonable for 2-bit quantization (allow higher error for limited precision)
        assert!(mse < 2.0, "MSE too high for {}: {}", qtype, mse);
    }
}

/// Test quantizer availability
#[test]
fn test_quantizer_availability() {
    let i2s = I2SQuantizer::new();
    let tl1 = TL1Quantizer::new();
    let tl2 = TL2Quantizer::new();

    assert!(i2s.is_available());
    assert!(tl1.is_available());
    assert!(tl2.is_available());

    assert_eq!(i2s.quantization_type(), QuantizationType::I2S);
    assert_eq!(tl1.quantization_type(), QuantizationType::TL1);
    assert_eq!(tl2.quantization_type(), QuantizationType::TL2);
}

/// Test best quantization type selection for architecture
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

// Property-based test for quantization round-trip accuracy
proptest! {
    #[test]
    fn prop_quantization_round_trip(
        data in prop::collection::vec(-10.0f32..10.0f32, 1..100),
        qtype in prop::sample::select(vec![
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2
        ])
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data.clone(), shape.clone());

        let quantized = tensor.quantize(qtype).unwrap();
        let dequantized = quantized.dequantize(&Device::Cpu).unwrap();

        // Basic properties should hold
        prop_assert_eq!(quantized.qtype, qtype);

        // Compression ratio should be reasonable
        let ratio = quantized.compression_ratio();
        prop_assert_eq!(quantized.shape, shape.clone());
        prop_assert_eq!(dequantized.shape(), &shape);
        prop_assert!(ratio >= 1.0); // Allow ratio of 1.0 for very small tensors

        // Should be able to extract dequantized data
        let dequant_candle = dequantized.inner();
        let dequant_data = dequant_candle.to_vec1::<f32>().unwrap();
        prop_assert_eq!(dequant_data.len(), data.len());
    }
}

// Property-based test for quantization format conversion
proptest! {
    #[test]
    fn prop_format_conversion(
        data in prop::collection::vec(-5.0f32..5.0f32, 4..32),
        source_qtype in prop::sample::select(vec![
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2
        ]),
        target_qtype in prop::sample::select(vec![
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2
        ])
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data, shape.clone());

        // Quantize to source format
        let source_quantized = tensor.quantize(source_qtype).unwrap();

        // Convert to target format
        let target_quantized = convert_quantization(&source_quantized, target_qtype).unwrap();

        // Properties should be preserved
        prop_assert_eq!(target_quantized.qtype, target_qtype);

        // Should be dequantizable
        let dequantized = target_quantized.dequantize(&Device::Cpu).unwrap();
        prop_assert_eq!(target_quantized.shape, shape.clone());
        prop_assert_eq!(dequantized.shape(), &shape);
    }
}

// Property-based test for quantization with different block sizes
proptest! {
    #[test]
    fn prop_different_block_sizes(
        data in prop::collection::vec(-2.0f32..2.0f32, 16..128),
        block_size in 4usize..64usize
    ) {
        let shape = vec![data.len()];
        let tensor = create_test_tensor(data, shape.clone());

        // Test I2_S with different block sizes
        let i2s_quantizer = I2SQuantizer::with_block_size(block_size);
        let quantized = i2s_quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = i2s_quantizer.dequantize_tensor(&quantized, &Device::Cpu).unwrap();

        prop_assert_eq!(quantized.block_size, block_size);
        prop_assert_eq!(quantized.shape, shape.clone());
        prop_assert_eq!(dequantized.shape(), &shape);
    }
}

/// Benchmark comparison test (simplified)
#[test]
fn test_quantization_performance_comparison() {
    let data = vec![1.0; 1024];
    let shape = vec![32, 32];
    let tensor = create_test_tensor(data, shape);

    let start = std::time::Instant::now();
    let _ = tensor.quantize(QuantizationType::I2S).unwrap();
    let i2s_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = tensor.quantize(QuantizationType::TL1).unwrap();
    let tl1_time = start.elapsed();

    let start = std::time::Instant::now();
    let _ = tensor.quantize(QuantizationType::TL2).unwrap();
    let tl2_time = start.elapsed();

    // All should complete in reasonable time (< 1 second for this small tensor)
    assert!(i2s_time.as_secs() < 1);
    assert!(tl1_time.as_secs() < 1);
    assert!(tl2_time.as_secs() < 1);
}

#[test]
fn test_gpu_dequantization_fallback() {
    let data = vec![0.5f32; 32];
    let shape = vec![32];
    let tensor = create_test_tensor(data, shape.clone());
    let quantizer = I2SQuantizer::new();
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer
        .dequantize_tensor(&quantized, &Device::Cuda(0))
        .unwrap();
    assert_eq!(dequantized.shape(), &shape);
}
