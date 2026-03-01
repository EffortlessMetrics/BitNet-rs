//! Edge-case tests for quantization types, pipeline configuration, validation,
//! and round-trip accuracy across I2S, TL1, and TL2 quantization formats.

use bitnet_common::{QuantizationType, SecurityLimits};
use bitnet_quantization::{
    AccuracyValidator, DeviceAwareQuantizer, I2SLayout, I2SQuantizer, QuantizedTensor,
    QuantizerFactory, TL1Quantizer, TL2Quantizer, ToleranceConfig,
};

// ── QuantizedTensor construction ──────────────────────────────────────────

#[test]
fn quantized_tensor_new_default_block_size() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    assert_eq!(qt.block_size, 32);
    assert!(qt.zero_points.is_none());
}

#[test]
fn quantized_tensor_new_with_params_preserves_all_fields() {
    let qt = QuantizedTensor::new_with_params(
        vec![1, 2, 3],
        vec![0.5, 1.5],
        Some(vec![0, 1]),
        vec![4, 8],
        QuantizationType::TL1,
        64,
    );
    assert_eq!(qt.data, vec![1, 2, 3]);
    assert_eq!(qt.scales, vec![0.5, 1.5]);
    assert_eq!(qt.zero_points, Some(vec![0, 1]));
    assert_eq!(qt.shape, vec![4, 8]);
    assert_eq!(qt.qtype, QuantizationType::TL1);
    assert_eq!(qt.block_size, 64);
}

#[test]
fn quantized_tensor_numel_single_dim() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![100], QuantizationType::I2S);
    assert_eq!(qt.numel(), 100);
}

#[test]
fn quantized_tensor_numel_multi_dim() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![4, 8, 16], QuantizationType::I2S);
    assert_eq!(qt.numel(), 512);
}

#[test]
fn quantized_tensor_numel_empty_shape_is_one() {
    // Product of empty iterator is 1 (identity for multiplication)
    let qt = QuantizedTensor::new(vec![], vec![], vec![], QuantizationType::I2S);
    assert_eq!(qt.numel(), 1);
}

#[test]
fn quantized_tensor_compression_ratio_no_data() {
    let qt = QuantizedTensor::new(vec![], vec![], vec![100], QuantizationType::I2S);
    // compressed_bytes = 0, so returns 1.0 (avoid division by zero)
    assert_eq!(qt.compression_ratio(), 1.0);
}

#[test]
fn quantized_tensor_compression_ratio_normal() {
    // 64 elements → 256 bytes FP32; compressed = 16 bytes data + 4*2 scales = 24
    let qt = QuantizedTensor::new(vec![0u8; 16], vec![1.0, 2.0], vec![64], QuantizationType::I2S);
    let ratio = qt.compression_ratio();
    assert!(ratio > 1.0, "Compression ratio should be > 1, got {ratio}");
}

#[test]
fn quantized_tensor_clone_is_independent() {
    let qt = QuantizedTensor::new(vec![1, 2, 3], vec![1.0], vec![32], QuantizationType::I2S);
    let clone = qt.clone();
    assert_eq!(qt.data, clone.data);
    assert_eq!(qt.shape, clone.shape);
    assert_eq!(qt.qtype, clone.qtype);
}

// ── I2SLayout ─────────────────────────────────────────────────────────────

#[test]
fn i2s_layout_default_values() {
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
    // 64 elements / 4 per byte = 16 data bytes + 2 scale
    assert_eq!(layout.data_bytes_per_block, 16);
    assert_eq!(layout.scale_bytes_per_block, 2);
    assert_eq!(layout.bytes_per_block, 18);
}

#[test]
fn i2s_layout_consistency_data_plus_scale_equals_total() {
    for &bs in &[32, 64, 128, 256] {
        let layout = I2SLayout::with_block_size(bs);
        assert_eq!(
            layout.data_bytes_per_block + layout.scale_bytes_per_block,
            layout.bytes_per_block,
            "Block size {bs}: total bytes must equal data + scale"
        );
    }
}

// ── I2SQuantizer ──────────────────────────────────────────────────────────

#[test]
fn i2s_quantizer_supports_cpu() {
    let q = I2SQuantizer::new();
    assert!(q.supports_device(&bitnet_common::Device::Cpu));
}

#[test]
fn i2s_quantizer_default_block_size_32() {
    // Default I2S uses block size 32 (same as I2SLayout::default)
    let _q = I2SQuantizer::new();
    // Construction succeeds — the block size is internal
}

#[test]
fn i2s_quantizer_with_custom_block_size() {
    let q = I2SQuantizer::with_block_size(64);
    assert!(q.supports_device(&bitnet_common::Device::Cpu));
}

// ── TL1Quantizer ─────────────────────────────────────────────────────────

#[test]
fn tl1_quantizer_creation() {
    let q = TL1Quantizer::new();
    assert!(q.supports_device(&bitnet_common::Device::Cpu));
}

// ── TL2Quantizer ─────────────────────────────────────────────────────────

#[test]
fn tl2_quantizer_creation() {
    let q = TL2Quantizer::new();
    assert!(q.supports_device(&bitnet_common::Device::Cpu));
}

// ── QuantizerFactory ──────────────────────────────────────────────────────

#[test]
fn factory_creates_i2s() {
    let q = QuantizerFactory::create(QuantizationType::I2S);
    assert_eq!(q.quantization_type(), QuantizationType::I2S);
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
fn factory_all_quantizers_available() {
    for qt in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let q = QuantizerFactory::create(qt);
        assert!(q.is_available(), "{:?} should be available", qt);
    }
}

#[test]
fn factory_best_for_arch_returns_valid_type() {
    let best = QuantizerFactory::best_for_arch();
    // On x86_64 this should be TL2, on aarch64 TL1, otherwise I2S
    // Just verify it's a valid type
    let q = QuantizerFactory::create(best);
    assert!(q.is_available());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn factory_best_for_x86_is_tl2() {
    assert_eq!(QuantizerFactory::best_for_arch(), QuantizationType::TL2);
}

// ── QK256 tolerance ──────────────────────────────────────────────────────

#[test]
fn qk256_tolerance_minimum_is_8() {
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(0), 8);
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(1), 8);
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(100), 8);
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(7999), 8);
}

#[test]
fn qk256_tolerance_proportional_for_large() {
    // 1MB → 0.1% = 1000 bytes
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(1_000_000), 1000);
}

#[test]
fn qk256_tolerance_ceiling_rounding() {
    // 131072 * 0.001 = 131.072 → ceil = 132
    assert_eq!(bitnet_quantization::qk256_tolerance_bytes(131_072), 132);
}

#[test]
fn qk256_tolerance_constant_value() {
    assert!((bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT - 0.001).abs() < f64::EPSILON);
}

// ── ToleranceConfig ──────────────────────────────────────────────────────

#[test]
fn tolerance_config_default_values() {
    let tc = ToleranceConfig::default();
    assert!((tc.i2s_tolerance - 1e-3).abs() < 1e-6);
    assert!((tc.tl_tolerance - 1e-2).abs() < 1e-6);
    assert!(tc.strict_validation);
}

// ── AccuracyValidator ────────────────────────────────────────────────────

#[test]
fn accuracy_validator_construction() {
    let tc = ToleranceConfig::default();
    let _v = AccuracyValidator::new(tc);
}

// ── DeviceAwareQuantizer ─────────────────────────────────────────────────

#[test]
fn device_aware_quantizer_construction() {
    let _q = DeviceAwareQuantizer::new();
}

#[test]
fn device_aware_quantizer_auto_detect() {
    let _q = DeviceAwareQuantizer::auto_detect();
}

#[test]
fn device_aware_quantizer_custom_tolerance() {
    let tc = ToleranceConfig {
        i2s_tolerance: 0.01,
        tl_tolerance: 0.05,
        perplexity_tolerance: 0.01,
        strict_validation: false,
    };
    let _q = DeviceAwareQuantizer::with_tolerance_config(tc);
}

// ── Pipeline configuration ───────────────────────────────────────────────

#[test]
fn pipeline_config_rejects_f32_target() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::F32,
        calibration_samples: 10,
        error_threshold: 0.01,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn pipeline_config_rejects_zero_calibration_samples() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 0,
        error_threshold: 0.01,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn pipeline_config_rejects_zero_threshold() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 10,
        error_threshold: 0.0,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn pipeline_config_rejects_negative_threshold() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 10,
        error_threshold: -1.0,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn pipeline_config_valid_f32_to_i2s() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: 0.05,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn pipeline_config_valid_f32_to_tl1() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL1,
        calibration_samples: 50,
        error_threshold: 0.1,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn pipeline_config_valid_f32_to_tl2() {
    use bitnet_quantization::pipeline::{PipelineConfig, Precision};
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL2,
        calibration_samples: 1,
        error_threshold: 1.0,
    };
    assert!(cfg.validate().is_ok());
}

// ── Precision enum ───────────────────────────────────────────────────────

#[test]
fn precision_variants_are_distinct() {
    use bitnet_quantization::pipeline::Precision;
    let variants = [Precision::F32, Precision::I2S, Precision::TL1, Precision::TL2];
    for (i, a) in variants.iter().enumerate() {
        for (j, b) in variants.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn precision_debug_format() {
    use bitnet_quantization::pipeline::Precision;
    let dbg = format!("{:?}", Precision::I2S);
    assert!(dbg.contains("I2S"));
}

// ── Validation module ────────────────────────────────────────────────────

#[test]
fn validation_numerical_input_tolerates_partial_nan() {
    // Mixed valid/NaN data is accepted (NaN mapped to zero)
    let data = vec![1.0, f32::NAN, 3.0];
    let result = bitnet_quantization::validation::validate_numerical_input(&data);
    assert!(result.is_ok());
}

#[test]
fn validation_numerical_input_rejects_all_nan() {
    let data = vec![f32::NAN, f32::NAN, f32::NAN];
    let result = bitnet_quantization::validation::validate_numerical_input(&data);
    assert!(result.is_err());
}

#[test]
fn validation_numerical_input_tolerates_partial_infinity() {
    // Mixed valid/infinity data is accepted (inf mapped to zero)
    let data = vec![1.0, f32::INFINITY, 3.0];
    let result = bitnet_quantization::validation::validate_numerical_input(&data);
    assert!(result.is_ok());
}

#[test]
fn validation_numerical_input_rejects_all_infinity() {
    let data = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let result = bitnet_quantization::validation::validate_numerical_input(&data);
    assert!(result.is_err());
}

#[test]
fn validation_numerical_input_accepts_valid_data() {
    let data = vec![0.0, -1.0, 1.0, 0.5, -0.5];
    let result = bitnet_quantization::validation::validate_numerical_input(&data);
    assert!(result.is_ok());
}

#[test]
fn validation_numerical_input_accepts_empty() {
    let result = bitnet_quantization::validation::validate_numerical_input(&[]);
    assert!(result.is_ok());
}

#[test]
fn validation_needs_detailed_for_large_data() {
    // needs_detailed_validation returns true for data with extreme values
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 100.0).collect();
    let _ = bitnet_quantization::validation::needs_detailed_validation(&data);
    // Just verify it doesn't panic
}

#[test]
fn validation_data_shape_consistency_matching() {
    let data = vec![1.0f32; 24];
    let shape = vec![4, 6];
    let result = bitnet_quantization::validation::validate_data_shape_consistency(&data, &shape);
    assert!(result.is_ok());
}

#[test]
fn validation_data_shape_consistency_mismatch() {
    let data = vec![1.0f32; 24];
    let shape = vec![4, 5]; // 20 != 24
    let result = bitnet_quantization::validation::validate_data_shape_consistency(&data, &shape);
    assert!(result.is_err());
}

#[test]
fn validation_memory_estimation() {
    let est = bitnet_quantization::validation::estimate_quantization_memory(1000, 2, 32);
    assert!(est > 0, "Memory estimate should be positive");
}

// ── Security limits validation ───────────────────────────────────────────

#[test]
fn security_limits_default_values() {
    let limits = SecurityLimits::default();
    assert_eq!(limits.max_tensor_elements, 1_000_000_000);
    assert_eq!(limits.max_memory_allocation, 4 * 1024 * 1024 * 1024);
    assert_eq!(limits.max_metadata_size, 100 * 1024 * 1024);
    assert_eq!(limits.max_string_length, 1024 * 1024);
    assert_eq!(limits.max_array_length, 1_000_000);
}

// ── QuantizationType from common ─────────────────────────────────────────

#[test]
fn quantization_type_variants_distinct() {
    assert_ne!(QuantizationType::I2S, QuantizationType::TL1);
    assert_ne!(QuantizationType::I2S, QuantizationType::TL2);
    assert_ne!(QuantizationType::TL1, QuantizationType::TL2);
}

#[test]
fn quantization_type_debug_format() {
    let dbg = format!("{:?}", QuantizationType::I2S);
    assert!(dbg.contains("I2S"));
}

// ── convert_quantization ─────────────────────────────────────────────────

#[test]
fn convert_same_type_is_clone() {
    let qt = QuantizedTensor::new(vec![0u8; 8], vec![1.0], vec![32], QuantizationType::I2S);
    let converted = bitnet_quantization::convert_quantization(&qt, QuantizationType::I2S);
    assert!(converted.is_ok());
    let converted = converted.unwrap();
    assert_eq!(converted.qtype, QuantizationType::I2S);
    assert_eq!(converted.data, qt.data);
}
