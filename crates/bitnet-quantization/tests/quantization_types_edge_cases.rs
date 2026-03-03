//! Edge-case tests for bitnet-quantization: QuantizedTensor, bit-packing,
//! quantize/dequantize value functions, scale calculations, metrics,
//! SimdCapabilities, QuantizationKernels, I2SLayout, QuantizerFactory, and QK256.

use bitnet_common::QuantizationType;
use bitnet_quantization::simd_ops::{QuantizationKernels, QuantizationStrategy, SimdCapabilities};
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_scale, calculate_snr, dequantize_value,
    dequantize_value_with_offset, pack_2bit_values, pack_unsigned_2bit_values, quantize_value,
    quantize_value_with_offset, unpack_2bit_values, unpack_unsigned_2bit_values,
};
use bitnet_quantization::{
    I2SLayout, I2SQuantizer, QuantizedTensor, QuantizerFactory, TL1Quantizer, TL2Quantizer,
};

// ===========================================================================
// pack / unpack 2-bit values
// ===========================================================================

#[test]
fn pack_unpack_2bit_roundtrip() {
    let values: Vec<i8> = vec![-2, -1, 0, 1];
    let packed = pack_2bit_values(&values);
    let unpacked = unpack_2bit_values(&packed, 4);
    assert_eq!(unpacked, values);
}

#[test]
fn pack_2bit_single_byte() {
    let values: Vec<i8> = vec![0, 0, 0, 0];
    let packed = pack_2bit_values(&values);
    assert_eq!(packed.len(), 1);
}

#[test]
fn pack_2bit_partial_block() {
    let values: Vec<i8> = vec![-1, 0, 1];
    let packed = pack_2bit_values(&values);
    let unpacked = unpack_2bit_values(&packed, 3);
    assert_eq!(unpacked, values);
}

#[test]
fn pack_2bit_clamping() {
    // Values outside [-2, 1] should be clamped
    let values: Vec<i8> = vec![-3, 5, -10, 100];
    let packed = pack_2bit_values(&values);
    let unpacked = unpack_2bit_values(&packed, 4);
    // All should be within [-2, 1]
    for &v in &unpacked {
        assert!((-2..=1).contains(&v));
    }
}

#[test]
fn pack_2bit_empty() {
    let packed = pack_2bit_values(&[]);
    assert!(packed.is_empty());
    let unpacked = unpack_2bit_values(&[], 0);
    assert!(unpacked.is_empty());
}

// ===========================================================================
// unsigned 2-bit pack / unpack
// ===========================================================================

#[test]
fn pack_unpack_unsigned_2bit_roundtrip() {
    let values: Vec<i8> = vec![0, 1, 2, 3];
    let packed = pack_unsigned_2bit_values(&values);
    let unpacked = unpack_unsigned_2bit_values(&packed, 4);
    assert_eq!(unpacked, values);
}

#[test]
fn pack_unsigned_2bit_partial() {
    let values: Vec<i8> = vec![1, 2];
    let packed = pack_unsigned_2bit_values(&values);
    let unpacked = unpack_unsigned_2bit_values(&packed, 2);
    assert_eq!(unpacked, values);
}

// ===========================================================================
// quantize_value / dequantize_value
// ===========================================================================

#[test]
fn quantize_dequantize_single_value() {
    let scale = 0.5;
    let q = quantize_value(1.0, scale, 2);
    let dq = dequantize_value(q, scale);
    // 2-bit quantization has very limited range; just check the value is finite
    assert!(dq.is_finite());
}

#[test]
fn quantize_value_zero() {
    assert_eq!(quantize_value(0.0, 1.0, 2), 0);
}

#[test]
fn quantize_value_nan() {
    assert_eq!(quantize_value(f32::NAN, 1.0, 2), 0);
}

#[test]
fn quantize_value_infinity() {
    assert_eq!(quantize_value(f32::INFINITY, 1.0, 2), 0);
}

#[test]
fn quantize_value_zero_scale() {
    assert_eq!(quantize_value(5.0, 0.0, 2), 0);
}

#[test]
fn dequantize_value_inf_scale() {
    assert_eq!(dequantize_value(1, f32::INFINITY), 0.0);
}

// ===========================================================================
// quantize_value_with_offset / dequantize_value_with_offset
// ===========================================================================

#[test]
fn quantize_with_offset_basic() {
    let q = quantize_value_with_offset(2.0, 1.0, 0, 4);
    assert_eq!(q, 2);
}

#[test]
fn dequantize_with_offset_basic() {
    let val = dequantize_value_with_offset(2, 1.0, 0);
    assert!((val - 2.0).abs() < f32::EPSILON);
}

#[test]
fn offset_roundtrip() {
    let original = 3.0f32;
    let scale = 1.5;
    let offset = 1;
    let q = quantize_value_with_offset(original, scale, offset, 4);
    let dq = dequantize_value_with_offset(q, scale, offset);
    assert!((dq - original).abs() < scale);
}

// ===========================================================================
// calculate_scale
// ===========================================================================

#[test]
fn scale_normal_data() {
    let data = vec![1.0, -2.0, 0.5, -0.5];
    let scale = calculate_scale(&data, 2);
    assert!(scale > 0.0);
    assert!(scale.is_finite());
}

#[test]
fn scale_all_zeros() {
    let data = vec![0.0; 8];
    let scale = calculate_scale(&data, 2);
    assert_eq!(scale, 1.0); // fallback
}

#[test]
fn scale_with_nan() {
    let data = vec![1.0, f32::NAN, -2.0];
    let scale = calculate_scale(&data, 2);
    assert!(scale.is_finite());
    assert!(scale > 0.0);
}

#[test]
fn scale_tiny_values() {
    let data = vec![1e-35, -1e-35];
    let scale = calculate_scale(&data, 2);
    assert_eq!(scale, 1.0); // fallback for near-zero
}

#[test]
fn scale_huge_values() {
    let data = vec![1e35, -1e35];
    let scale = calculate_scale(&data, 2);
    assert!(scale.is_finite());
    assert!(scale > 0.0);
}

// ===========================================================================
// calculate_grouped_scales
// ===========================================================================

#[test]
fn grouped_scales_single_block() {
    let data = vec![1.0, -1.0, 0.5, -0.5];
    let scales = calculate_grouped_scales(&data, 4, 2);
    assert_eq!(scales.len(), 1);
}

#[test]
fn grouped_scales_multiple_blocks() {
    let data = vec![1.0; 12];
    let scales = calculate_grouped_scales(&data, 4, 2);
    assert_eq!(scales.len(), 3);
}

#[test]
fn grouped_scales_partial_last_block() {
    let data = vec![1.0; 5];
    let scales = calculate_grouped_scales(&data, 4, 2);
    assert_eq!(scales.len(), 2);
}

// ===========================================================================
// calculate_mse / calculate_snr
// ===========================================================================

#[test]
fn mse_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let mse = calculate_mse(&a, &b).unwrap();
    assert!(mse < f32::EPSILON);
}

#[test]
fn mse_different() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 3.0, 4.0];
    let mse = calculate_mse(&a, &b).unwrap();
    assert!((mse - 1.0).abs() < f32::EPSILON);
}

#[test]
fn mse_length_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    assert!(calculate_mse(&a, &b).is_err());
}

#[test]
fn snr_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let snr = calculate_snr(&a, &b).unwrap();
    assert!(snr > 100.0); // Very high SNR for identical signals
}

#[test]
fn snr_length_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    assert!(calculate_snr(&a, &b).is_err());
}

// ===========================================================================
// QuantizedTensor
// ===========================================================================

#[test]
fn quantized_tensor_new() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    assert_eq!(qt.numel(), 32);
    assert_eq!(qt.block_size, 32);
}

#[test]
fn quantized_tensor_with_params() {
    let qt = QuantizedTensor::new_with_params(
        vec![0u8; 16],
        vec![1.0, 2.0],
        Some(vec![0, 0]),
        vec![4, 16],
        QuantizationType::TL1,
        64,
    );
    assert_eq!(qt.numel(), 64);
    assert_eq!(qt.block_size, 64);
    assert!(qt.zero_points.is_some());
}

#[test]
fn quantized_tensor_compression_ratio() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    let ratio = qt.compression_ratio();
    assert!(ratio > 1.0);
}

#[test]
fn quantized_tensor_compression_ratio_empty_data() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![0], QuantizationType::I2S);
    let ratio = qt.compression_ratio();
    assert_eq!(ratio, 1.0); // Division by zero fallback
}

// ===========================================================================
// I2SLayout
// ===========================================================================

#[test]
fn i2s_layout_default_block_size() {
    let layout = I2SLayout::with_block_size(32);
    assert_eq!(layout.block_size, 32);
    assert_eq!(layout.data_bytes_per_block, 8);
    assert_eq!(layout.scale_bytes_per_block, 2);
    assert_eq!(layout.bytes_per_block, 10);
}

#[test]
fn i2s_layout_custom_block_size() {
    let layout = I2SLayout::with_block_size(64);
    assert_eq!(layout.block_size, 64);
    assert_eq!(layout.data_bytes_per_block, 16);
}

// ===========================================================================
// SimdCapabilities
// ===========================================================================

#[test]
fn simd_capabilities_detect() {
    let caps = SimdCapabilities::detect();
    let _strategy = caps.best_quantization_strategy();
    let block_size = caps.optimal_block_size();
    assert!(block_size > 0);
}

#[test]
fn simd_strategy_variants() {
    let strategies = [
        QuantizationStrategy::Scalar,
        QuantizationStrategy::SSE4_1,
        QuantizationStrategy::AVX2,
        QuantizationStrategy::AVX512,
        QuantizationStrategy::NEON,
    ];
    for s in &strategies {
        let dbg = format!("{s:?}");
        assert!(!dbg.is_empty());
    }
}

// ===========================================================================
// QuantizationKernels
// ===========================================================================

#[test]
fn quantization_kernels_new() {
    let kernels = QuantizationKernels::new();
    let caps = kernels.capabilities();
    let _strategy = caps.best_quantization_strategy();
}

#[test]
fn quantization_kernels_scalar_quantize() {
    let kernels = QuantizationKernels::new();
    let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75];
    let scales = calculate_grouped_scales(&data, 4, 2);
    let quantized = kernels.quantize_scalar(&data, &scales, 4, 2).unwrap();
    assert_eq!(quantized.len(), data.len());
}

#[test]
fn quantization_kernels_scalar_dequantize() {
    let kernels = QuantizationKernels::new();
    let quantized: Vec<i8> = vec![1, -1, 0, 1];
    let scales = vec![0.5];
    let dequantized = kernels.dequantize_scalar(&quantized, &scales, 4).unwrap();
    assert_eq!(dequantized.len(), quantized.len());
}

// ===========================================================================
// QuantizerFactory
// ===========================================================================

#[test]
fn factory_create_i2s() {
    let q = QuantizerFactory::create(QuantizationType::I2S);
    assert_eq!(q.quantization_type(), QuantizationType::I2S);
    assert!(q.is_available());
}

#[test]
fn factory_create_tl1() {
    let q = QuantizerFactory::create(QuantizationType::TL1);
    assert_eq!(q.quantization_type(), QuantizationType::TL1);
}

#[test]
fn factory_create_tl2() {
    let q = QuantizerFactory::create(QuantizationType::TL2);
    assert_eq!(q.quantization_type(), QuantizationType::TL2);
}

#[test]
fn factory_best_for_arch() {
    let qtype = QuantizerFactory::best_for_arch();
    let _ = format!("{qtype:?}");
}

// ===========================================================================
// QK256 tolerance
// ===========================================================================

#[test]
fn qk256_tolerance_small() {
    let tol = bitnet_quantization::qk256_tolerance_bytes(1000);
    assert!(tol >= 1);
}

#[test]
fn qk256_tolerance_large() {
    let tol = bitnet_quantization::qk256_tolerance_bytes(1_000_000);
    assert!(tol >= 1);
}

#[test]
fn qk256_size_tolerance_constant() {
    assert!(bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT > 0.0);
    assert!(bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT < 1.0);
}

// ===========================================================================
// I2SQuantizer / TL1Quantizer / TL2Quantizer construction
// ===========================================================================

#[test]
fn i2s_quantizer_new() {
    let _q = I2SQuantizer::new();
}

#[test]
fn tl1_quantizer_new() {
    let _q = TL1Quantizer::new();
}

#[test]
fn tl2_quantizer_new() {
    let _q = TL2Quantizer::new();
}
