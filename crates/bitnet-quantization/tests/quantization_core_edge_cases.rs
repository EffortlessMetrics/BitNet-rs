//! Edge-case tests for `bitnet-quantization` core types:
//! QuantizedTensor, QuantizerFactory, qk256_tolerance_bytes, I2S/TL1/TL2 quantizers.

use bitnet_common::{BitNetTensor, QuantizationType};
use bitnet_quantization::{
    I2SQuantizer, QK256_SIZE_TOLERANCE_PERCENT, QuantizedTensor, QuantizerFactory, QuantizerTrait,
    TL1Quantizer, TL2Quantizer, qk256_tolerance_bytes,
};

// ---------------------------------------------------------------------------
// qk256_tolerance_bytes
// ---------------------------------------------------------------------------

#[test]
fn tolerance_bytes_large_tensor() {
    // 1 MB → 0.1% = 1000 bytes
    assert_eq!(qk256_tolerance_bytes(1_000_000), 1000);
}

#[test]
fn tolerance_bytes_medium_tensor() {
    // 100 KB → 100 bytes
    assert_eq!(qk256_tolerance_bytes(100_000), 100);
}

#[test]
fn tolerance_bytes_small_tensor_clamped_to_8() {
    // Very small → minimum 8 bytes
    assert_eq!(qk256_tolerance_bytes(20), 8);
    assert_eq!(qk256_tolerance_bytes(0), 8);
    assert_eq!(qk256_tolerance_bytes(1), 8);
}

#[test]
fn tolerance_bytes_boundary_at_8000() {
    // 8000 → 0.1% = 8.0, exactly at the minimum
    assert_eq!(qk256_tolerance_bytes(8_000), 8);
}

#[test]
fn tolerance_bytes_ceiling_rounding() {
    // 131072 → 0.1% = 131.072, ceiling = 132
    assert_eq!(qk256_tolerance_bytes(131_072), 132);
}

#[test]
fn qk256_tolerance_percent_is_0_001() {
    assert!((QK256_SIZE_TOLERANCE_PERCENT - 0.001).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// QuantizedTensor construction
// ---------------------------------------------------------------------------

#[test]
fn quantized_tensor_new() {
    let qt =
        QuantizedTensor::new(vec![1, 2, 3, 4], vec![0.5, 1.0], vec![2, 2], QuantizationType::I2S);
    assert_eq!(qt.data.len(), 4);
    assert_eq!(qt.scales.len(), 2);
    assert_eq!(qt.shape, vec![2, 2]);
    assert_eq!(qt.qtype, QuantizationType::I2S);
    assert_eq!(qt.block_size, 32); // default
    assert!(qt.zero_points.is_none());
}

#[test]
fn quantized_tensor_new_with_params() {
    let qt = QuantizedTensor::new_with_params(
        vec![10, 20],
        vec![1.5],
        Some(vec![0]),
        vec![4],
        QuantizationType::TL1,
        64,
    );
    assert_eq!(qt.block_size, 64);
    assert_eq!(qt.zero_points, Some(vec![0]));
    assert_eq!(qt.qtype, QuantizationType::TL1);
}

// ---------------------------------------------------------------------------
// numel
// ---------------------------------------------------------------------------

#[test]
fn numel_2d() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![3, 4], QuantizationType::I2S);
    assert_eq!(qt.numel(), 12);
}

#[test]
fn numel_1d() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![10], QuantizationType::I2S);
    assert_eq!(qt.numel(), 10);
}

#[test]
fn numel_3d() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![2, 3, 4], QuantizationType::I2S);
    assert_eq!(qt.numel(), 24);
}

#[test]
fn numel_empty_shape() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![], QuantizationType::I2S);
    assert_eq!(qt.numel(), 1); // Product of empty = 1
}

// ---------------------------------------------------------------------------
// compression_ratio
// ---------------------------------------------------------------------------

#[test]
fn compression_ratio_basic() {
    // 4 elements * 4 bytes = 16 bytes original
    // 2 bytes data + 1 scale * 4 bytes = 6 compressed
    let qt = QuantizedTensor::new(vec![0, 0], vec![1.0], vec![4], QuantizationType::I2S);
    let ratio = qt.compression_ratio();
    // 16 / 6 ≈ 2.67
    assert!(ratio > 2.0 && ratio < 3.0, "ratio = {ratio}");
}

#[test]
fn compression_ratio_empty_data_returns_1() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![4], QuantizationType::I2S);
    assert!((qt.compression_ratio() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn compression_ratio_at_least_1() {
    // Even if compressed is larger than original, ratio is clamped to ≥ 1.0
    let qt = QuantizedTensor::new(
        vec![0; 1000],   // 1000 bytes data
        vec![1.0; 1000], // 1000 scales × 4 bytes = 4000 bytes
        vec![1],         // 1 element × 4 bytes = 4 bytes original
        QuantizationType::I2S,
    );
    assert!(qt.compression_ratio() >= 1.0);
}

// ---------------------------------------------------------------------------
// QuantizerFactory
// ---------------------------------------------------------------------------

#[test]
fn factory_creates_i2s() {
    let q = QuantizerFactory::create(QuantizationType::I2S);
    assert_eq!(q.quantization_type(), QuantizationType::I2S);
    assert!(q.is_available());
}

#[test]
fn factory_creates_tl1() {
    let q = QuantizerFactory::create(QuantizationType::TL1);
    assert_eq!(q.quantization_type(), QuantizationType::TL1);
}

#[test]
fn factory_creates_tl2() {
    let q = QuantizerFactory::create(QuantizationType::TL2);
    assert_eq!(q.quantization_type(), QuantizationType::TL2);
}

#[test]
fn factory_best_for_arch_returns_valid_type() {
    let best = QuantizerFactory::best_for_arch();
    // Should be one of the three types
    assert!(matches!(best, QuantizationType::I2S | QuantizationType::TL1 | QuantizationType::TL2));
}

// ---------------------------------------------------------------------------
// Individual quantizer types
// ---------------------------------------------------------------------------

#[test]
fn i2s_quantizer_type() {
    let q = I2SQuantizer::new();
    assert_eq!(q.quantization_type(), QuantizationType::I2S);
    assert!(q.is_available());
}

#[test]
fn tl1_quantizer_type() {
    let q = TL1Quantizer::new();
    assert_eq!(q.quantization_type(), QuantizationType::TL1);
}

#[test]
fn tl2_quantizer_type() {
    let q = TL2Quantizer::new();
    assert_eq!(q.quantization_type(), QuantizationType::TL2);
}

// ---------------------------------------------------------------------------
// QuantizedTensor clone
// ---------------------------------------------------------------------------

#[test]
fn quantized_tensor_clone() {
    let qt = QuantizedTensor::new(vec![1, 2, 3], vec![0.5], vec![3], QuantizationType::I2S);
    let clone = qt.clone();
    assert_eq!(clone.data, qt.data);
    assert_eq!(clone.scales, qt.scales);
    assert_eq!(clone.shape, qt.shape);
    assert_eq!(clone.qtype, qt.qtype);
}

// ---------------------------------------------------------------------------
// I2S quantize/dequantize roundtrip on small tensor
// ---------------------------------------------------------------------------

#[test]
fn i2s_roundtrip_small_tensor() {
    // I2S encodes values as {-1, 0, +1} with block_size=32
    let data: Vec<f32> = vec![
        1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0,
        0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0,
    ];
    let device = bitnet_common::Device::Cpu;
    let tensor = BitNetTensor::from_slice(&data, &[32], &device).unwrap();
    let q = I2SQuantizer::new();
    let quantized = q.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.qtype, QuantizationType::I2S);
    assert_eq!(quantized.shape, vec![32]);

    let dequantized = q.dequantize_tensor(&quantized).unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    assert_eq!(deq_data.len(), 32);
    // Values should be close to original (exact for ternary inputs)
    for (orig, deq) in data.iter().zip(deq_data.iter()) {
        assert!((orig - deq).abs() < 0.1, "orig={orig}, deq={deq}");
    }
}

// ---------------------------------------------------------------------------
// Quantization type equality
// ---------------------------------------------------------------------------

#[test]
fn quantization_type_eq() {
    assert_eq!(QuantizationType::I2S, QuantizationType::I2S);
    assert_ne!(QuantizationType::I2S, QuantizationType::TL1);
    assert_ne!(QuantizationType::TL1, QuantizationType::TL2);
}
