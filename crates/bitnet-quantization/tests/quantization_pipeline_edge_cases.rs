//! Edge-case tests for quantization pipeline, factory, and tolerance APIs.

use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_quantization::pipeline::{
    PipelineConfig, Precision, QuantizationPipeline, QuantizationStage,
};
use bitnet_quantization::simd_ops::{QuantizationStrategy, SimdCapabilities};
use bitnet_quantization::{
    I2SLayout, I2SQuantizer, QK256_SIZE_TOLERANCE_PERCENT, QuantizationConfig, Quantize,
    QuantizedTensor, QuantizerFactory, TL1Quantizer, TL2Quantizer, ToleranceConfig,
    convert_quantization, qk256_tolerance_bytes, validate_round_trip,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_tensor(values: &[f32]) -> BitNetTensor {
    BitNetTensor::from_slice(values, &[values.len()], &Device::Cpu).expect("tensor creation")
}

fn make_tensor_2d(values: &[f32], rows: usize, cols: usize) -> BitNetTensor {
    BitNetTensor::from_slice(values, &[rows, cols], &Device::Cpu).expect("tensor creation")
}

// ===========================================================================
// 1. ToleranceConfig — defaults, fields, custom values
// ===========================================================================

#[test]
fn tolerance_config_default_values() {
    let cfg = ToleranceConfig::default();
    assert!((cfg.i2s_tolerance - 1e-3).abs() < 1e-12, "I2S tolerance default");
    assert!((cfg.tl_tolerance - 1e-2).abs() < 1e-12, "TL tolerance default");
    assert!((cfg.perplexity_tolerance - 0.001).abs() < 1e-12, "Perplexity tolerance default");
    assert!(cfg.strict_validation, "strict_validation should default to true");
}

#[test]
fn tolerance_config_custom_values() {
    let cfg = ToleranceConfig {
        i2s_tolerance: 0.5,
        tl_tolerance: 0.25,
        perplexity_tolerance: 0.01,
        strict_validation: false,
    };
    assert!((cfg.i2s_tolerance - 0.5).abs() < 1e-12);
    assert!((cfg.tl_tolerance - 0.25).abs() < 1e-12);
    assert!(!cfg.strict_validation);
}

#[test]
fn tolerance_config_clone() {
    let cfg = ToleranceConfig::default();
    let cloned = cfg.clone();
    assert!((cfg.i2s_tolerance - cloned.i2s_tolerance).abs() < 1e-12);
    assert!((cfg.tl_tolerance - cloned.tl_tolerance).abs() < 1e-12);
}

// ===========================================================================
// 2. QuantizationType — Display, serde roundtrip
// ===========================================================================

#[test]
fn quantization_type_display_formatting() {
    assert_eq!(QuantizationType::I2S.to_string(), "I2_S");
    assert_eq!(QuantizationType::TL1.to_string(), "TL1");
    assert_eq!(QuantizationType::TL2.to_string(), "TL2");
}

#[test]
fn quantization_type_serde_roundtrip() {
    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let json = serde_json::to_string(&qtype).unwrap();
        let restored: QuantizationType = serde_json::from_str(&json).unwrap();
        assert_eq!(qtype, restored, "serde roundtrip for {qtype}");
    }
}

#[test]
fn quantization_type_equality_and_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(QuantizationType::I2S);
    set.insert(QuantizationType::TL1);
    set.insert(QuantizationType::TL2);
    set.insert(QuantizationType::I2S); // duplicate
    assert_eq!(set.len(), 3);
}

// ===========================================================================
// 3. I2S quantization — encode/decode, boundary values
// ===========================================================================

#[test]
fn i2s_roundtrip_small_array() {
    let quantizer = I2SQuantizer::new();
    let data = vec![1.0, -1.0, 0.0, 0.5];
    let tensor = make_tensor(&data);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[4]);
}

#[test]
fn i2s_boundary_values_minus_one_zero_one() {
    let quantizer = I2SQuantizer::new();
    let data = vec![-1.0, 0.0, 1.0, -1.0];
    let tensor = make_tensor(&data);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.qtype, QuantizationType::I2S);

    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    // Each dequantized value should be in [-1, 1] range
    for &v in &deq_data {
        assert!(v >= -1.5 && v <= 1.5, "dequantized {v} out of range");
    }
}

#[test]
fn i2s_quantize_weights_api() {
    let quantizer = I2SQuantizer::new();
    let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let quantized = quantizer.quantize_weights(&weights).unwrap();
    assert_eq!(quantized.shape, vec![8]);
    assert_eq!(quantized.qtype, QuantizationType::I2S);
}

#[test]
fn i2s_with_custom_block_size() {
    let quantizer = I2SQuantizer::with_block_size(8);
    let data = vec![0.5; 16];
    let tensor = make_tensor(&data);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.block_size, 8);
}

#[test]
fn i2s_layout_defaults() {
    let layout = I2SLayout::default();
    assert_eq!(layout.block_size, 32);
    assert_eq!(layout.bytes_per_block, 10);
    assert_eq!(layout.data_bytes_per_block, 8);
    assert_eq!(layout.scale_bytes_per_block, 2);
}

#[test]
fn i2s_layout_custom_block_size() {
    let layout = I2SLayout::with_block_size(64);
    assert_eq!(layout.block_size, 64);
    // 64 * 2 bits = 128 bits = 16 bytes
    assert_eq!(layout.data_bytes_per_block, 16);
    assert_eq!(layout.bytes_per_block, 18); // 16 + 2 scale
}

// ===========================================================================
// 4. TL1/TL2 — table lookup encoding/decoding, boundary conditions
// ===========================================================================

#[test]
fn tl1_roundtrip() {
    let quantizer = TL1Quantizer::new();
    let data = vec![0.5, -0.5, 0.25, -0.25];
    let tensor = make_tensor(&data);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.qtype, QuantizationType::TL1);
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[4]);
}

#[test]
fn tl2_roundtrip() {
    let quantizer = TL2Quantizer::new();
    let data = vec![0.5, -0.5, 0.25, -0.25];
    let tensor = make_tensor(&data);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.qtype, QuantizationType::TL2);
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[4]);
}

#[test]
fn tl1_quantize_weights_api() {
    let quantizer = TL1Quantizer::new();
    let weights = vec![1.0, -1.0, 0.5, -0.5];
    let quantized = quantizer.quantize_weights(&weights).unwrap();
    assert_eq!(quantized.shape, vec![4]);
}

#[test]
fn tl2_quantize_weights_api() {
    let quantizer = TL2Quantizer::new();
    let weights = vec![1.0, -1.0, 0.5, -0.5];
    let quantized = quantizer.quantize_weights(&weights).unwrap();
    assert_eq!(quantized.shape, vec![4]);
}

// ===========================================================================
// 5. QK256 tolerance constants
// ===========================================================================

#[test]
fn qk256_tolerance_percent_constant() {
    assert!((QK256_SIZE_TOLERANCE_PERCENT - 0.001).abs() < 1e-12);
}

#[test]
fn qk256_tolerance_bytes_large_tensor() {
    assert_eq!(qk256_tolerance_bytes(1_000_000), 1000);
}

#[test]
fn qk256_tolerance_bytes_medium_tensor() {
    assert_eq!(qk256_tolerance_bytes(131_072), 132);
}

#[test]
fn qk256_tolerance_bytes_minimum_clamp() {
    // Small tensors get minimum 8 bytes tolerance
    assert_eq!(qk256_tolerance_bytes(20), 8);
    assert_eq!(qk256_tolerance_bytes(0), 8);
    assert_eq!(qk256_tolerance_bytes(1), 8);
}

#[test]
fn qk256_tolerance_bytes_boundary() {
    // 8000 bytes → 0.1% = 8.0, exactly the minimum
    assert_eq!(qk256_tolerance_bytes(8000), 8);
    // 8001 → ceil(8.001) = 9
    assert_eq!(qk256_tolerance_bytes(8001), 9);
}

// ===========================================================================
// 6. Quantization accuracy — quantize then dequantize, verify tolerance
// ===========================================================================

#[test]
fn validate_round_trip_i2s_passes_with_generous_tolerance() {
    let tensor = make_tensor(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
    let result = validate_round_trip(&tensor, QuantizationType::I2S, 10.0).unwrap();
    assert!(result, "round-trip with generous tolerance should pass");
}

#[test]
fn validate_round_trip_i2s_fails_with_zero_tolerance() {
    // Non-trivial values can't survive 2-bit quantization with zero tolerance
    let tensor = make_tensor(&[0.1, -0.2, 0.3, -0.4]);
    let result = validate_round_trip(&tensor, QuantizationType::I2S, 0.0).unwrap();
    assert!(!result, "round-trip with zero tolerance should fail for non-trivial data");
}

#[test]
fn validate_round_trip_tl1() {
    let tensor = make_tensor(&[0.1, -0.1, 0.2, -0.2]);
    let result = validate_round_trip(&tensor, QuantizationType::TL1, 10.0).unwrap();
    assert!(result);
}

#[test]
fn validate_round_trip_tl2() {
    let tensor = make_tensor(&[0.1, -0.1, 0.2, -0.2]);
    let result = validate_round_trip(&tensor, QuantizationType::TL2, 10.0).unwrap();
    assert!(result);
}

// ===========================================================================
// 7. Edge cases — all-zeros, all-same-value, 2D tensors
// ===========================================================================

#[test]
fn i2s_all_zeros() {
    let quantizer = I2SQuantizer::new();
    let tensor = make_tensor(&[0.0; 32]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    for &v in &deq_data {
        assert!(v.abs() < 1e-6, "all-zero input should dequantize to ~0, got {v}");
    }
}

#[test]
fn i2s_all_same_positive_value() {
    let quantizer = I2SQuantizer::new();
    let tensor = make_tensor(&[0.42; 16]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[16]);
}

#[test]
fn i2s_all_same_negative_value() {
    let quantizer = I2SQuantizer::new();
    let tensor = make_tensor(&[-0.7; 8]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[8]);
}

#[test]
fn tl1_all_zeros() {
    let quantizer = TL1Quantizer::new();
    let tensor = make_tensor(&[0.0; 8]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    for &v in &deq_data {
        assert!(v.abs() < 1e-6, "TL1 all-zero round-trip should be ~0, got {v}");
    }
}

#[test]
fn tl2_all_zeros() {
    let quantizer = TL2Quantizer::new();
    let tensor = make_tensor(&[0.0; 8]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    for &v in &deq_data {
        assert!(v.abs() < 1e-6, "TL2 all-zero round-trip should be ~0, got {v}");
    }
}

#[test]
fn i2s_2d_tensor() {
    let quantizer = I2SQuantizer::new();
    let data: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.1).collect();
    let tensor = make_tensor_2d(&data, 3, 4);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.shape, vec![3, 4]);
    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    assert_eq!(dequantized.shape(), &[3, 4]);
}

// ===========================================================================
// 8. QuantizedTensor — constructors, numel, compression_ratio
// ===========================================================================

#[test]
fn quantized_tensor_new() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    assert_eq!(qt.numel(), 32);
    assert_eq!(qt.block_size, 32);
    assert!(qt.zero_points.is_none());
}

#[test]
fn quantized_tensor_new_with_params() {
    let qt = QuantizedTensor::new_with_params(
        vec![0u8; 16],
        vec![1.0, 2.0],
        Some(vec![0, 1]),
        vec![64],
        QuantizationType::TL1,
        64,
    );
    assert_eq!(qt.numel(), 64);
    assert_eq!(qt.block_size, 64);
    assert!(qt.zero_points.is_some());
}

#[test]
fn quantized_tensor_compression_ratio_positive() {
    let qt = QuantizedTensor::new(vec![0u8; 32], vec![1.0; 4], vec![256], QuantizationType::I2S);
    let ratio = qt.compression_ratio();
    // 256 * 4 bytes original / (32 + 4*4) compressed
    assert!(ratio >= 1.0, "compression ratio should be >= 1.0, got {ratio}");
}

#[test]
fn quantized_tensor_compression_ratio_empty_data() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![0], QuantizationType::I2S);
    // numel() = 0, so original_bytes = 0, compressed_bytes = 0 → returns 1.0
    assert!((qt.compression_ratio() - 1.0).abs() < 1e-6);
}

// ===========================================================================
// 9. QuantizerFactory — create, best_for_arch
// ===========================================================================

#[test]
fn factory_creates_all_types() {
    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantizer = QuantizerFactory::create(qtype);
        assert_eq!(quantizer.quantization_type(), qtype);
        assert!(quantizer.is_available());
    }
}

#[test]
fn factory_best_for_arch_returns_valid_type() {
    let best = QuantizerFactory::best_for_arch();
    // Should return one of the known types
    assert!(
        best == QuantizationType::I2S
            || best == QuantizationType::TL1
            || best == QuantizationType::TL2,
        "best_for_arch returned unexpected type: {best}"
    );
}

#[test]
fn factory_created_quantizer_can_roundtrip() {
    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let quantizer = QuantizerFactory::create(qtype);
        let tensor = make_tensor(&[0.3, -0.3, 0.6, -0.6]);
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let _dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    }
}

// ===========================================================================
// 10. convert_quantization
// ===========================================================================

#[test]
fn convert_quantization_same_type_is_identity() {
    let quantizer = I2SQuantizer::new();
    let tensor = make_tensor(&[0.1, -0.2, 0.3, -0.4]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let converted = convert_quantization(&quantized, QuantizationType::I2S).unwrap();
    assert_eq!(converted.qtype, QuantizationType::I2S);
    assert_eq!(converted.shape, quantized.shape);
}

#[test]
fn convert_quantization_i2s_to_tl1() {
    let quantizer = I2SQuantizer::new();
    let tensor = make_tensor(&[0.5, -0.5, 0.25, -0.25]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let converted = convert_quantization(&quantized, QuantizationType::TL1).unwrap();
    assert_eq!(converted.qtype, QuantizationType::TL1);
}

#[test]
fn convert_quantization_tl1_to_tl2() {
    let quantizer = TL1Quantizer::new();
    let tensor = make_tensor(&[0.5, -0.5, 0.25, -0.25]);
    let quantized = quantizer.quantize_tensor(&tensor).unwrap();
    let converted = convert_quantization(&quantized, QuantizationType::TL2).unwrap();
    assert_eq!(converted.qtype, QuantizationType::TL2);
}

// ===========================================================================
// 11. Quantize trait on BitNetTensor
// ===========================================================================

#[test]
fn bitnet_tensor_quantize_trait_i2s() {
    let tensor = make_tensor(&[1.0, -1.0, 0.5, -0.5]);
    let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
    assert_eq!(quantized.qtype, QuantizationType::I2S);
}

#[test]
fn bitnet_tensor_dequantize_is_identity() {
    let tensor = make_tensor(&[1.0, -1.0, 0.5, -0.5]);
    let dequantized = tensor.dequantize().unwrap();
    let orig_data = tensor.to_vec().unwrap();
    let deq_data = dequantized.to_vec().unwrap();
    assert_eq!(orig_data, deq_data, "dequantize on raw tensor should be identity");
}

#[test]
fn quantized_tensor_requantize_same_type() {
    let tensor = make_tensor(&[0.3, -0.3, 0.6, -0.6]);
    let quantized = tensor.quantize(QuantizationType::I2S).unwrap();
    let requantized = quantized.quantize(QuantizationType::I2S).unwrap();
    // Same type → clone
    assert_eq!(requantized.qtype, QuantizationType::I2S);
    assert_eq!(requantized.data, quantized.data);
}

// ===========================================================================
// 12. Pipeline — config validation edge cases
// ===========================================================================

#[test]
fn pipeline_config_f32_target_rejected() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::F32,
        calibration_samples: 4,
        error_threshold: 1.0,
    };
    assert!(QuantizationPipeline::new(cfg).is_err());
}

#[test]
fn pipeline_config_zero_calibration_rejected() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 0,
        error_threshold: 1.0,
    };
    assert!(QuantizationPipeline::new(cfg).is_err());
}

#[test]
fn pipeline_config_negative_threshold_rejected() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 4,
        error_threshold: -1.0,
    };
    assert!(QuantizationPipeline::new(cfg).is_err());
}

#[test]
fn pipeline_config_zero_threshold_rejected() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 4,
        error_threshold: 0.0,
    };
    assert!(QuantizationPipeline::new(cfg).is_err());
}

#[test]
fn pipeline_empty_layers_rejected() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 4,
        error_threshold: 1.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    assert!(pipeline.execute(&[]).is_err());
}

#[test]
fn pipeline_single_layer() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 2,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.1, -0.2, 0.3, -0.4])];
    let result = pipeline.execute(&layers).unwrap();
    assert_eq!(result.per_layer_errors.len(), 1);
    assert_eq!(result.quantized_tensors.len(), 1);
}

#[test]
fn pipeline_stage_progression() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    assert_eq!(pipeline.current_stage(), None);

    let layers = vec![make_tensor(&[0.1, -0.1, 0.2, -0.2])];
    pipeline.execute(&layers).unwrap();
    assert_eq!(pipeline.current_stage(), Some(QuantizationStage::PackingOptimization));
}

#[test]
fn pipeline_double_execute_fails() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.1, -0.1, 0.2, -0.2])];
    pipeline.execute(&layers).unwrap();
    assert!(pipeline.execute(&layers).is_err(), "second execute should fail");
}

#[test]
fn pipeline_tight_threshold_triggers_violation() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 1e-15,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[10.0, -10.0, 5.0, -5.0])];
    let result = pipeline.execute(&layers).unwrap();
    assert!(result.threshold_violated);
}

#[test]
fn pipeline_generous_threshold_no_violation() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 1e6,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.1, -0.1, 0.2, -0.2])];
    let result = pipeline.execute(&layers).unwrap();
    assert!(!result.threshold_violated);
}

#[test]
fn pipeline_tl1_target() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL1,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.5, -0.5, 0.3, -0.3])];
    let result = pipeline.execute(&layers).unwrap();
    assert!(!result.quantized_tensors.is_empty());
}

#[test]
fn pipeline_tl2_target() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL2,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.5, -0.5, 0.3, -0.3])];
    let result = pipeline.execute(&layers).unwrap();
    assert!(!result.quantized_tensors.is_empty());
}

#[test]
fn pipeline_all_zeros_layer() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[0.0; 64])];
    let result = pipeline.execute(&layers).unwrap();
    assert!(result.per_layer_errors[0] < 1e-6, "all-zero layer should have near-zero MSE");
}

#[test]
fn pipeline_calibration_statistics() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 5,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let layers = vec![make_tensor(&[1.0, -2.0, 3.0, -4.0]), make_tensor(&[0.5; 4])];
    let result = pipeline.execute(&layers).unwrap();
    assert_eq!(result.calibration.min_values.len(), 2);
    assert_eq!(result.calibration.max_values.len(), 2);
    assert_eq!(result.calibration.num_samples, 5);
}

#[test]
fn pipeline_compression_ratio_positive() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 1,
        error_threshold: 100.0,
    };
    let mut pipeline = QuantizationPipeline::new(cfg).unwrap();
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let layers = vec![make_tensor(&data)];
    let result = pipeline.execute(&layers).unwrap();
    assert!(result.compression_ratio > 1.0, "should achieve compression");
}

// ===========================================================================
// 13. Precision enum coverage
// ===========================================================================

#[test]
fn precision_variants_exist() {
    let _f32 = Precision::F32;
    let _i2s = Precision::I2S;
    let _tl1 = Precision::TL1;
    let _tl2 = Precision::TL2;
}

#[test]
fn precision_equality() {
    assert_eq!(Precision::F32, Precision::F32);
    assert_ne!(Precision::I2S, Precision::TL1);
}

// ===========================================================================
// 14. QuantizationStage ordering
// ===========================================================================

#[test]
fn quantization_stage_ordering() {
    assert!(QuantizationStage::Calibration < QuantizationStage::Quantization);
    assert!(QuantizationStage::Quantization < QuantizationStage::Verification);
    assert!(QuantizationStage::Verification < QuantizationStage::PackingOptimization);
}

// ===========================================================================
// 15. SimdCapabilities
// ===========================================================================

#[test]
fn simd_capabilities_detect_returns_valid() {
    let caps = SimdCapabilities::detect();
    // On x86_64, at least SSE4.1 is very common; on other archs, all may be false
    let _ = caps.has_avx512;
    let _ = caps.has_avx2;
    let _ = caps.has_neon;
    let _ = caps.has_sse4_1;
}

#[test]
fn simd_capabilities_best_strategy() {
    let caps = SimdCapabilities::detect();
    let strategy = caps.best_quantization_strategy();
    // Should be one of the known strategies
    assert!(
        strategy == QuantizationStrategy::Scalar
            || strategy == QuantizationStrategy::SSE4_1
            || strategy == QuantizationStrategy::AVX2
            || strategy == QuantizationStrategy::AVX512
            || strategy == QuantizationStrategy::NEON
    );
}

#[test]
fn simd_capabilities_optimal_block_size() {
    let caps = SimdCapabilities::detect();
    let block_size = caps.optimal_block_size();
    assert!(block_size >= 32 && block_size <= 256, "block size {block_size} out of range");
}

// ===========================================================================
// 16. DeviceQuantizationType (from device_aware_quantizer re-export)
// ===========================================================================

#[test]
fn device_quantization_type_display() {
    use bitnet_quantization::DeviceQuantizationType;
    assert_eq!(DeviceQuantizationType::I2S.to_string(), "I2_S");
    assert_eq!(DeviceQuantizationType::TL1.to_string(), "TL1");
    assert_eq!(DeviceQuantizationType::TL2.to_string(), "TL2");
    assert_eq!(DeviceQuantizationType::IQ2S.to_string(), "IQ2_S");
    assert_eq!(DeviceQuantizationType::FP32.to_string(), "FP32");
}

#[test]
fn device_quantization_type_serde_roundtrip() {
    use bitnet_quantization::DeviceQuantizationType;
    for qtype in [
        DeviceQuantizationType::I2S,
        DeviceQuantizationType::TL1,
        DeviceQuantizationType::TL2,
        DeviceQuantizationType::IQ2S,
        DeviceQuantizationType::FP32,
    ] {
        let json = serde_json::to_string(&qtype).unwrap();
        let restored: DeviceQuantizationType = serde_json::from_str(&json).unwrap();
        assert_eq!(qtype, restored);
    }
}

// ===========================================================================
// 17. AccuracyValidator (from device_aware_quantizer)
// ===========================================================================

#[test]
fn accuracy_validator_creation() {
    use bitnet_quantization::AccuracyValidator;
    let _validator = AccuracyValidator::new(ToleranceConfig::default());
}

// ===========================================================================
// 18. QuantizationConfig re-export
// ===========================================================================

#[test]
fn quantization_config_reexport_accessible() {
    let _cfg = QuantizationConfig::default();
}
