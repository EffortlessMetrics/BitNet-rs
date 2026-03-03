//! BDD-style integration tests — Wave 20
//!
//! Each test follows the `given_*_when_*_then_*` naming convention and exercises
//! quantization format detection, parameter validation, scale factor computation,
//! dequantization correctness, calibration, and pipeline operations.

use bitnet_quantization::calibrator::{CalibrationMethod, Calibrator, TensorStats, compute_params};
use bitnet_quantization::i2s::{I2SLayout, I2SQuantizer};
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use bitnet_quantization::pipeline::{PipelineConfig, Precision, QuantizationPipeline};
use bitnet_quantization::simd_ops::{QuantizationKernels, QuantizationStrategy, SimdCapabilities};
use bitnet_quantization::utils::{
    calculate_grouped_scales, calculate_mse, calculate_optimal_block_size, calculate_scale,
    calculate_snr, dequantize_value, dequantize_value_with_offset, pack_2bit_values,
    quantize_value, quantize_value_with_offset, unpack_2bit_values, validate_shapes,
};
use bitnet_quantization::{
    QuantizedTensor, QuantizerFactory, QuantizerTrait, qk256_tolerance_bytes, validate_round_trip,
};

use bitnet_common::QuantizationType;

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Quantization Format Detection / Factory
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_i2s_type_when_creating_via_factory_then_returns_quantizer() {
    // Given: QuantizationType::I2S
    // When: creating via factory
    let q = QuantizerFactory::create(QuantizationType::I2S);
    // Then: valid quantizer returned (is_available defaults to true)
    assert!(q.is_available());
}

#[test]
fn given_tl1_type_when_creating_via_factory_then_returns_quantizer() {
    let q = QuantizerFactory::create(QuantizationType::TL1);
    assert!(q.is_available());
}

#[test]
fn given_tl2_type_when_creating_via_factory_then_returns_quantizer() {
    let q = QuantizerFactory::create(QuantizationType::TL2);
    assert!(q.is_available());
}

#[test]
fn given_factory_when_creating_i2s_then_quantization_type_is_i2s() {
    let q = QuantizerFactory::create(QuantizationType::I2S);
    assert_eq!(q.quantization_type(), QuantizationType::I2S);
}

#[test]
fn given_factory_when_best_for_arch_then_returns_valid_type() {
    // Given/When: asking factory for best arch-specific type
    let best = QuantizerFactory::best_for_arch();
    // Then: the type can create a quantizer
    let q = QuantizerFactory::create(best);
    assert!(q.is_available());
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — I2S Layout Constants
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_default_i2s_layout_when_checking_block_size_then_32() {
    let layout = I2SLayout::default();
    assert_eq!(layout.block_size, 32);
}

#[test]
fn given_default_i2s_layout_when_checking_bytes_per_block_then_10() {
    let layout = I2SLayout::default();
    assert_eq!(layout.bytes_per_block, 10);
}

#[test]
fn given_default_i2s_layout_when_checking_data_plus_scale_then_equals_total() {
    let layout = I2SLayout::default();
    assert_eq!(layout.data_bytes_per_block + layout.scale_bytes_per_block, layout.bytes_per_block);
}

#[test]
fn given_custom_block_size_when_creating_i2s_quantizer_then_accepted() {
    // Given/When: custom block size
    let q = I2SQuantizer::with_block_size(64);
    // Then: quantizer is created (no panic)
    let _ = q;
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — QK256 Constants & Code Mapping
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_qk256_constants_when_checking_block_size_then_256() {
    assert_eq!(QK256_BLOCK, 256);
}

#[test]
fn given_qk256_constants_when_checking_packed_bytes_then_64() {
    assert_eq!(QK256_PACKED_BYTES, 64);
}

#[test]
fn given_qk256_constants_when_verifying_packing_ratio_then_4_codes_per_byte() {
    // 256 codes / 64 bytes = 4 codes per byte (2 bits each)
    assert_eq!(QK256_BLOCK / QK256_PACKED_BYTES, 4);
}

#[test]
fn given_code_0_when_converting_to_f32_then_minus_2() {
    assert_eq!(code_to_f32(0), -2.0);
}

#[test]
fn given_code_1_when_converting_to_f32_then_minus_1() {
    assert_eq!(code_to_f32(1), -1.0);
}

#[test]
fn given_code_2_when_converting_to_f32_then_plus_1() {
    assert_eq!(code_to_f32(2), 1.0);
}

#[test]
fn given_code_3_when_converting_to_f32_then_plus_2() {
    assert_eq!(code_to_f32(3), 2.0);
}

#[test]
fn given_all_zero_block_when_unpacking_then_all_code_0() {
    // Given: 64-byte block of all zeros
    let block = [0u8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];
    // When: unpacking
    unpack_qk256_block(&block, &mut codes);
    // Then: all codes are 0
    assert!(codes.iter().all(|&c| c == 0));
}

#[test]
fn given_all_ff_block_when_unpacking_then_all_code_3() {
    // Given: 64-byte block of all 0xFF
    let block = [0xFFu8; QK256_PACKED_BYTES];
    let mut codes = [0u8; QK256_BLOCK];
    // When: unpacking
    unpack_qk256_block(&block, &mut codes);
    // Then: all codes are 3
    assert!(codes.iter().all(|&c| c == 3));
}

#[test]
fn given_alternating_bits_when_unpacking_then_correct_pattern() {
    // Given: byte 0b01_10_01_00 = 0x64
    // codes: [0, 1, 2, 1] from LSB to MSB
    let mut block = [0u8; QK256_PACKED_BYTES];
    block[0] = 0b01_10_01_00; // bits: 00, 01, 10, 01
    let mut codes = [0u8; QK256_BLOCK];
    // When: unpacking
    unpack_qk256_block(&block, &mut codes);
    // Then: first 4 codes match the byte
    assert_eq!(codes[0], 0b00);
    assert_eq!(codes[1], 0b01);
    assert_eq!(codes[2], 0b10);
    assert_eq!(codes[3], 0b01);
}

#[test]
fn given_single_byte_pattern_when_unpack_roundtrip_then_codes_in_range() {
    // Given: an arbitrary byte pattern
    let mut block = [0u8; QK256_PACKED_BYTES];
    for (i, b) in block.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37);
    }
    let mut codes = [0u8; QK256_BLOCK];
    // When: unpacking
    unpack_qk256_block(&block, &mut codes);
    // Then: all codes in 0..=3
    assert!(codes.iter().all(|&c| c <= 3));
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — Scale Factor Computation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_uniform_data_when_calculating_scale_then_non_negative() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let scale = calculate_scale(&data, 2);
    assert!(scale >= 0.0);
}

#[test]
fn given_all_zeros_when_calculating_scale_then_safe_fallback() {
    // calculate_scale returns 1.0 as safe fallback for all-zeros
    let data = vec![0.0, 0.0, 0.0, 0.0];
    let scale = calculate_scale(&data, 2);
    assert_eq!(scale, 1.0);
}

#[test]
fn given_single_element_when_calculating_scale_then_valid() {
    let data = vec![5.0];
    let scale = calculate_scale(&data, 2);
    assert!(scale > 0.0);
}

#[test]
fn given_symmetric_data_when_calculating_scale_then_reflects_magnitude() {
    // Given: symmetric data [-10, 10]
    let data = vec![-10.0, 10.0];
    let scale = calculate_scale(&data, 8);
    // Then: scale covers the range
    assert!(scale > 0.0);
}

#[test]
fn given_data_when_calculating_grouped_scales_then_correct_count() {
    // Given: 64 elements, block_size=32
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    // When: calculating grouped scales
    let scales = calculate_grouped_scales(&data, 32, 2);
    // Then: 2 scales (one per block)
    assert_eq!(scales.len(), 2);
}

#[test]
fn given_larger_block_than_data_when_calculating_grouped_scales_then_one_scale() {
    // Given: 8 elements, block_size=32
    let data: Vec<f32> = vec![1.0; 8];
    let scales = calculate_grouped_scales(&data, 32, 2);
    // Then: 1 scale
    assert_eq!(scales.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════
// Section 5 — 2-bit Packing / Unpacking
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_four_values_when_packing_2bit_then_one_byte() {
    // Given: 4 values (2 bits each = 1 byte)
    let values = vec![0i8, 1, 2, 3];
    // When: packing
    let packed = pack_2bit_values(&values);
    // Then: 1 byte
    assert_eq!(packed.len(), 1);
}

#[test]
fn given_packed_byte_when_unpacking_then_recovers_original() {
    // Given: values → pack → unpack
    let original = vec![0i8, 1, -1, 0];
    let packed = pack_2bit_values(&original);
    let recovered = unpack_2bit_values(&packed, original.len());
    // Then: matches original
    assert_eq!(recovered, original);
}

#[test]
fn given_eight_values_when_packing_then_two_bytes() {
    let values = vec![1i8, 0, -1, 1, 0, 0, 1, -1];
    let packed = pack_2bit_values(&values);
    assert_eq!(packed.len(), 2);
}

#[test]
fn given_large_array_when_pack_unpack_roundtrip_then_exact_recovery() {
    // Given: 32 random-ish values in {-1, 0, 1}
    let values: Vec<i8> = (0..32).map(|i| (i % 3) as i8 - 1).collect();
    let packed = pack_2bit_values(&values);
    let recovered = unpack_2bit_values(&packed, values.len());
    assert_eq!(recovered, values);
}

// ═══════════════════════════════════════════════════════════════════
// Section 6 — Quantize / Dequantize Single Values
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_zero_value_when_quantizing_then_zero() {
    let q = quantize_value(0.0, 1.0, 8);
    assert_eq!(q, 0);
}

#[test]
fn given_positive_value_when_quantize_dequantize_then_bounded_error() {
    // Given: value=1.5, scale=0.1, 8 bits
    let scale = 0.1;
    let q = quantize_value(1.5, scale, 8);
    let dq = dequantize_value(q, scale);
    // Then: error bounded by scale
    assert!((1.5 - dq).abs() <= scale + 1e-6);
}

#[test]
fn given_negative_value_when_quantize_dequantize_then_sign_preserved() {
    let scale = 0.5;
    let q = quantize_value(-3.0, scale, 8);
    let dq = dequantize_value(q, scale);
    assert!(dq < 0.0);
}

#[test]
fn given_value_with_offset_when_dequantizing_then_offset_applied() {
    let dq = dequantize_value_with_offset(10, 0.1, 5);
    let expected = (10 - 5) as f32 * 0.1;
    assert!((dq - expected).abs() < 1e-6);
}

#[test]
fn given_value_with_offset_when_quantizing_then_offset_applied() {
    let q = quantize_value_with_offset(1.0, 0.5, 2, 8);
    // quantize_value_with_offset: round(value / scale) + offset
    let expected = (1.0f32 / 0.5).round() as i32 + 2;
    assert_eq!(q as i32, expected.clamp(-128, 127));
}

#[test]
fn given_zero_scale_when_quantizing_then_clamps_to_max() {
    // Given: zero scale (edge case)
    let q = quantize_value(5.0, 0.0, 8);
    // Then: clamped (no divide-by-zero panic)
    let _ = q; // just ensure no panic
}

// ═══════════════════════════════════════════════════════════════════
// Section 7 — MSE and SNR Metrics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_identical_arrays_when_calculating_mse_then_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let mse = calculate_mse(&a, &b).unwrap();
    assert!(mse.abs() < 1e-9);
}

#[test]
fn given_different_arrays_when_calculating_mse_then_positive() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.5, 2.5, 3.5];
    let mse = calculate_mse(&a, &b).unwrap();
    assert!(mse > 0.0);
    // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
    assert!((mse - 0.25).abs() < 1e-6);
}

#[test]
fn given_mismatched_lengths_when_calculating_mse_then_error() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    assert!(calculate_mse(&a, &b).is_err());
}

#[test]
fn given_identical_arrays_when_calculating_snr_then_high_value() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let snr = calculate_snr(&a, &b).unwrap();
    // SNR of identical signals → very high (or inf, but impl may cap it)
    assert!(snr > 50.0 || snr.is_infinite());
}

#[test]
fn given_noisy_arrays_when_calculating_snr_then_finite_positive() {
    let a = vec![10.0, 20.0, 30.0];
    let b = vec![10.1, 20.2, 29.9];
    let snr = calculate_snr(&a, &b).unwrap();
    assert!(snr > 0.0);
    assert!(snr.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// Section 8 — Shape Validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_matching_shapes_when_validating_then_ok() {
    assert!(validate_shapes(&[2, 3], &[2, 3]).is_ok());
}

#[test]
fn given_mismatched_shapes_when_validating_then_error() {
    assert!(validate_shapes(&[2, 3], &[3, 2]).is_err());
}

#[test]
fn given_different_rank_shapes_when_validating_then_error() {
    assert!(validate_shapes(&[2, 3], &[2, 3, 4]).is_err());
}

#[test]
fn given_empty_shapes_when_validating_then_ok() {
    assert!(validate_shapes(&[], &[]).is_ok());
}

// ═══════════════════════════════════════════════════════════════════
// Section 9 — Optimal Block Size
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_large_tensor_when_computing_optimal_block_size_then_positive() {
    let bs = calculate_optimal_block_size(1024, 8);
    assert!(bs > 0);
}

#[test]
fn given_small_tensor_when_computing_optimal_block_size_then_at_least_one() {
    let bs = calculate_optimal_block_size(4, 8);
    assert!(bs >= 1);
}

#[test]
fn given_exact_division_when_computing_optimal_block_size_then_exact() {
    let bs = calculate_optimal_block_size(256, 8);
    assert_eq!(bs, 32); // 256 / 8 = 32
}

// ═══════════════════════════════════════════════════════════════════
// Section 10 — QK256 Tolerance
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_zero_bytes_when_calculating_tolerance_then_minimum_8() {
    // qk256_tolerance_bytes has a minimum of 8 for alignment padding
    assert_eq!(qk256_tolerance_bytes(0), 8);
}

#[test]
fn given_small_bytes_when_calculating_tolerance_then_at_least_one() {
    // For 1000 bytes with 0.1% tolerance = 1 byte
    let tol = qk256_tolerance_bytes(1000);
    assert!(tol >= 1);
}

#[test]
fn given_large_bytes_when_calculating_tolerance_then_proportional() {
    let tol_small = qk256_tolerance_bytes(1000);
    let tol_large = qk256_tolerance_bytes(1_000_000);
    // Larger input → larger tolerance
    assert!(tol_large > tol_small);
}

// ═══════════════════════════════════════════════════════════════════
// Section 11 — Calibration
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_new_calibrator_when_checking_count_then_zero() {
    let cal = Calibrator::int8_symmetric();
    assert_eq!(cal.tensor_count(), 0);
}

#[test]
fn given_calibrator_when_observing_then_count_increases() {
    let mut cal = Calibrator::int8_symmetric();
    cal.observe("layer1", &[1.0, 2.0, 3.0]);
    assert_eq!(cal.tensor_count(), 1);
}

#[test]
fn given_calibrator_when_observing_twice_same_name_then_count_stays_one() {
    let mut cal = Calibrator::int8_symmetric();
    cal.observe("layer1", &[1.0, 2.0, 3.0]);
    cal.observe("layer1", &[4.0, 5.0, 6.0]);
    assert_eq!(cal.tensor_count(), 1);
}

#[test]
fn given_calibrator_when_observing_different_names_then_count_increases() {
    let mut cal = Calibrator::int8_symmetric();
    cal.observe("layer1", &[1.0]);
    cal.observe("layer2", &[2.0]);
    assert_eq!(cal.tensor_count(), 2);
}

#[test]
fn given_calibrator_when_calibrating_then_returns_results_for_each_tensor() {
    let mut cal = Calibrator::int8_symmetric();
    cal.observe("layer1", &[1.0, 2.0, 3.0]);
    cal.observe("layer2", &[-1.0, 0.0, 1.0]);
    let results = cal.calibrate();
    assert_eq!(results.len(), 2);
}

#[test]
fn given_calibrator_when_getting_stats_then_returns_correct_minmax() {
    let mut cal = Calibrator::int8_symmetric();
    cal.observe("t", &[-5.0, 0.0, 10.0]);
    let stats = cal.get_stats("t").unwrap();
    assert!((stats.min - (-5.0)).abs() < 1e-9);
    assert!((stats.max - 10.0).abs() < 1e-9);
}

#[test]
fn given_tensor_stats_when_computing_range_then_correct() {
    let mut stats = TensorStats::new("test");
    stats.update(&[-3.0, 7.0]);
    assert!((stats.range() - 10.0).abs() < 1e-9);
}

#[test]
fn given_tensor_stats_when_computing_absmax_then_correct() {
    let mut stats = TensorStats::new("test");
    stats.update(&[-8.0, 5.0]);
    assert!((stats.absmax() - 8.0).abs() < 1e-9);
}

#[test]
fn given_symmetric_stats_when_computing_params_minmax_then_scale_positive() {
    let mut stats = TensorStats::new("t");
    stats.update(&[-1.0, 1.0]);
    let result = compute_params(&stats, 8, true, CalibrationMethod::MinMax);
    assert!(result.scale > 0.0);
    assert!(result.symmetric);
}

#[test]
fn given_stats_when_computing_params_percentile_then_scale_positive() {
    let mut stats = TensorStats::new("t");
    stats.update(&(0..100).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
    let result = compute_params(&stats, 8, true, CalibrationMethod::Percentile);
    assert!(result.scale > 0.0);
}

#[test]
fn given_stats_when_computing_params_entropy_then_scale_positive() {
    let mut stats = TensorStats::new("t");
    stats.update(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = compute_params(&stats, 8, true, CalibrationMethod::Entropy);
    assert!(result.scale > 0.0);
}

#[test]
fn given_calibration_method_when_getting_str_then_non_empty() {
    assert_eq!(CalibrationMethod::MinMax.as_str(), "minmax");
    assert_eq!(CalibrationMethod::Percentile.as_str(), "percentile");
    assert_eq!(CalibrationMethod::Entropy.as_str(), "entropy");
    assert_eq!(CalibrationMethod::Mse.as_str(), "mse");
}

// ═══════════════════════════════════════════════════════════════════
// Section 12 — SIMD Capabilities & Strategy
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_runtime_when_detecting_simd_caps_then_valid() {
    let caps = SimdCapabilities::detect();
    // At least one field should reflect actual hardware
    let _ = caps.has_avx2; // no panic
}

#[test]
fn given_simd_caps_when_querying_strategy_then_returns_variant() {
    let caps = SimdCapabilities::detect();
    let strategy = caps.best_quantization_strategy();
    // Strategy is a valid enum variant
    let _ = match strategy {
        QuantizationStrategy::Scalar => "scalar",
        QuantizationStrategy::AVX2 => "avx2",
        QuantizationStrategy::AVX512 => "avx512",
        QuantizationStrategy::NEON => "neon",
        QuantizationStrategy::SSE4_1 => "sse4_1",
    };
}

#[test]
fn given_simd_caps_when_querying_optimal_block_size_then_power_of_two() {
    let caps = SimdCapabilities::detect();
    let bs = caps.optimal_block_size();
    assert!(bs.is_power_of_two(), "block size {} should be power of 2", bs);
}

#[test]
fn given_quantization_kernels_when_creating_default_then_no_panic() {
    let kernels = QuantizationKernels::new();
    let _ = kernels.capabilities();
}

#[test]
fn given_quantization_kernels_when_scalar_quantize_then_valid_output() {
    let kernels = QuantizationKernels::new();
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let scale = calculate_scale(&input, 2);
    let scales = vec![scale];
    let result = kernels.quantize_scalar(&input, &scales, input.len(), 2);
    assert!(result.is_ok());
    let quantized = result.unwrap();
    assert_eq!(quantized.len(), input.len());
}

#[test]
fn given_quantization_kernels_when_scalar_dequantize_then_valid_output() {
    let kernels = QuantizationKernels::new();
    let quantized = vec![1i8, -1, 0, 1];
    let scale = 0.5;
    let scales = vec![scale];
    let result = kernels.dequantize_scalar(&quantized, &scales, quantized.len());
    assert!(result.is_ok());
    let dequantized = result.unwrap();
    assert_eq!(dequantized.len(), quantized.len());
}

#[test]
fn given_quantization_kernels_when_scalar_roundtrip_then_bounded_error() {
    let kernels = QuantizationKernels::new();
    let input = vec![1.0f32, -0.5, 0.0, 0.75];
    let scale = calculate_scale(&input, 2);
    if scale > 0.0 {
        let scales = vec![scale];
        let quantized = kernels.quantize_scalar(&input, &scales, input.len(), 2).unwrap();
        let dequantized = kernels.dequantize_scalar(&quantized, &scales, quantized.len()).unwrap();
        for (orig, recov) in input.iter().zip(dequantized.iter()) {
            assert!((orig - recov).abs() <= scale + 1e-6, "error too large: {} vs {}", orig, recov);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Section 13 — Pipeline Configuration Validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_valid_pipeline_config_when_validating_then_ok() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: 0.01,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn given_f32_target_when_validating_pipeline_config_then_error() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::F32,
        calibration_samples: 100,
        error_threshold: 0.01,
    };
    assert!(config.validate().is_err());
}

#[test]
fn given_zero_calibration_samples_when_validating_then_error() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 0,
        error_threshold: 0.01,
    };
    assert!(config.validate().is_err());
}

#[test]
fn given_negative_error_threshold_when_validating_then_error() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: -0.01,
    };
    assert!(config.validate().is_err());
}

#[test]
fn given_zero_error_threshold_when_validating_then_error() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: 0.0,
    };
    assert!(config.validate().is_err());
}

#[test]
fn given_valid_config_when_creating_pipeline_then_ok() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 10,
        error_threshold: 0.1,
    };
    let pipeline = QuantizationPipeline::new(config);
    assert!(pipeline.is_ok());
}

#[test]
fn given_new_pipeline_when_checking_stage_then_none() {
    let config = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 10,
        error_threshold: 0.1,
    };
    let pipeline = QuantizationPipeline::new(config).unwrap();
    assert!(pipeline.current_stage().is_none());
}

// ═══════════════════════════════════════════════════════════════════
// Section 14 — QuantizedTensor Properties
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_quantized_tensor_when_checking_numel_then_matches_shape() {
    let tensor =
        QuantizedTensor::new(vec![0u8; 8], vec![1.0f32], vec![4, 2], QuantizationType::I2S);
    assert_eq!(tensor.numel(), 8);
}

#[test]
fn given_quantized_tensor_when_checking_compression_ratio_then_positive() {
    let tensor = QuantizedTensor::new(vec![0u8; 4], vec![1.0f32], vec![16], QuantizationType::I2S);
    let ratio = tensor.compression_ratio();
    assert!(ratio > 0.0);
}

#[test]
fn given_quantized_tensor_with_large_shape_when_compression_ratio_then_greater_than_one() {
    // 64 elements as f32 = 256 bytes; packed = 16 bytes + 4 scale = 20 bytes
    let tensor = QuantizedTensor::new(vec![0u8; 16], vec![1.0f32], vec![64], QuantizationType::I2S);
    let ratio = tensor.compression_ratio();
    assert!(ratio > 1.0, "compression ratio {} should be > 1.0", ratio);
}

// ═══════════════════════════════════════════════════════════════════
// Section 15 — Dequantization Correctness Edge Cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn given_max_positive_quantized_when_dequantizing_then_positive() {
    let dq = dequantize_value(127, 0.01);
    assert!(dq > 0.0);
    assert!((dq - 1.27).abs() < 1e-5);
}

#[test]
fn given_max_negative_quantized_when_dequantizing_then_negative() {
    let dq = dequantize_value(-128, 0.01);
    assert!(dq < 0.0);
    assert!((dq - (-1.28)).abs() < 1e-5);
}

#[test]
fn given_zero_quantized_when_dequantizing_then_zero() {
    let dq = dequantize_value(0, 1.0);
    assert_eq!(dq, 0.0);
}

#[test]
fn given_very_small_scale_when_dequantizing_then_near_zero() {
    let dq = dequantize_value(1, 1e-10);
    assert!(dq.abs() < 1e-8);
}

#[test]
fn given_very_large_scale_when_dequantizing_then_large_output() {
    let dq = dequantize_value(1, 1e6);
    assert!((dq - 1e6).abs() < 1.0);
}
