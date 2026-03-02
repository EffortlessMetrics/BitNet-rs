//! Wave 25 snapshot tests for bitnet-quantization.
//!
//! Covers: Quantization format detection snapshots, block layout snapshots,
//! scale factor computation snapshots, and error message format snapshots.

use bitnet_common::QuantizationType;
use bitnet_quantization::device_aware_quantizer::ToleranceConfig;
use bitnet_quantization::i2s::I2SLayout;
use bitnet_quantization::int4_quant::{Int4QuantConfig, NibblePacked, quantize_tensor_int4};
use bitnet_quantization::int8_quant::{CalibrationMethod, Int8QuantConfig};
use bitnet_quantization::pipeline::{PipelineConfig, Precision, QuantizationStage};
use bitnet_quantization::simd_ops::{QuantizationStrategy, SimdCapabilities};
use bitnet_quantization::tl1::TL1Config;
use bitnet_quantization::tl2::TL2Config;

// =========================================================================
// Section 1 — Quantization format detection
// =========================================================================

#[test]
fn w25_quantization_type_all_variants_debug() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w25_quantization_type_display_all() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let displays: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w25_precision_all_variants_debug() {
    let variants = [Precision::F32, Precision::I2S, Precision::TL1, Precision::TL2];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w25_quantization_stage_all_debug() {
    let stages = [
        QuantizationStage::Calibration,
        QuantizationStage::Quantization,
        QuantizationStage::Verification,
        QuantizationStage::PackingOptimization,
    ];
    let labeled: Vec<String> = stages.iter().map(|s| format!("{s:?}={}", *s as u8)).collect();
    insta::assert_snapshot!(labeled.join(", "));
}

// =========================================================================
// Section 2 — Block layout snapshots
// =========================================================================

#[test]
fn w25_i2s_layout_default_snapshot() {
    let layout = I2SLayout::default();
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block
    ));
}

#[test]
fn w25_i2s_layout_field_values() {
    let layout = I2SLayout::default();
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block
    ));
}

#[test]
fn w25_tl1_config_default_debug() {
    let cfg = TL1Config::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_tl2_config_default_debug() {
    let cfg = TL2Config::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_int4_config_default_debug() {
    let cfg = Int4QuantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_int4_config_asymmetric() {
    let cfg = Int4QuantConfig { group_size: 64, symmetric: false, block_wise: true };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_int8_config_default_debug() {
    let cfg = Int8QuantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_int8_config_asymmetric_percentile() {
    let cfg = Int8QuantConfig {
        per_channel: false,
        symmetric: false,
        calibration_method: CalibrationMethod::Percentile(99.9),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_calibration_method_all_variants() {
    let methods =
        [CalibrationMethod::MinMax, CalibrationMethod::Percentile(99.0), CalibrationMethod::MSE];
    insta::assert_debug_snapshot!(methods);
}

// =========================================================================
// Section 3 — Scale factor computation snapshots
// =========================================================================

#[test]
fn w25_int4_quantize_uniform() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
    let cfg = Int4QuantConfig::default();
    let (packed, params) = quantize_tensor_int4(&data, &cfg);
    insta::assert_snapshot!(format!(
        "num_groups={} group_size={} symmetric={} scale_0={:.6} len={}",
        params.num_groups, params.group_size, params.symmetric, params.scales[0], packed.len
    ));
}

#[test]
fn w25_int4_quantize_zeros() {
    let data = vec![0.0f32; 64];
    let cfg = Int4QuantConfig::default();
    let (packed, params) = quantize_tensor_int4(&data, &cfg);
    insta::assert_snapshot!(format!(
        "num_groups={} scale_0={:.6} len={}",
        params.num_groups, params.scales[0], packed.len
    ));
}

#[test]
fn w25_int4_quantize_empty() {
    let data: Vec<f32> = vec![];
    let cfg = Int4QuantConfig::default();
    let (packed, params) = quantize_tensor_int4(&data, &cfg);
    insta::assert_snapshot!(format!("num_groups={} len={}", params.num_groups, packed.len));
}

#[test]
fn w25_nibble_packed_roundtrip() {
    let values: Vec<i8> = vec![-8, -4, -1, 0, 1, 4, 7, 3];
    let packed = NibblePacked::pack(&values);
    let unpacked = packed.unpack();
    insta::assert_debug_snapshot!(unpacked);
}

#[test]
fn w25_nibble_packed_single_value() {
    let packed = NibblePacked::pack(&[5]);
    insta::assert_snapshot!(format!(
        "len={} data_bytes={} value={}",
        packed.len,
        packed.data.len(),
        packed.get(0)
    ));
}

#[test]
fn w25_tolerance_config_debug() {
    let cfg = ToleranceConfig {
        i2s_tolerance: 0.05,
        tl_tolerance: 0.1,
        perplexity_tolerance: 0.5,
        strict_validation: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 4 — Error message format snapshots
// =========================================================================

#[test]
fn w25_pipeline_config_f32_target_error() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::F32,
        calibration_samples: 100,
        error_threshold: 0.01,
    };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_pipeline_config_zero_samples_error() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 0,
        error_threshold: 0.01,
    };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_pipeline_config_zero_threshold_error() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL1,
        calibration_samples: 50,
        error_threshold: 0.0,
    };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn w25_pipeline_config_valid_i2s() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 512,
        error_threshold: 0.05,
    };
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_pipeline_config_valid_tl2() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL2,
        calibration_samples: 256,
        error_threshold: 0.1,
    };
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — SIMD capabilities and quantization strategy
// =========================================================================

#[test]
fn w25_simd_capabilities_detect() {
    let caps = SimdCapabilities::detect();
    insta::assert_snapshot!(format!(
        "avx512={} avx2={} neon={} sse4_1={}",
        caps.has_avx512, caps.has_avx2, caps.has_neon, caps.has_sse4_1
    ));
}

#[test]
fn w25_quantization_strategy_all_variants() {
    let strategies = [
        QuantizationStrategy::Scalar,
        QuantizationStrategy::SSE4_1,
        QuantizationStrategy::AVX2,
        QuantizationStrategy::AVX512,
        QuantizationStrategy::NEON,
    ];
    insta::assert_debug_snapshot!(strategies);
}

#[test]
fn w25_simd_best_strategy_and_block_size() {
    let caps = SimdCapabilities::detect();
    let strategy = caps.best_quantization_strategy();
    let block_size = caps.optimal_block_size();
    insta::assert_snapshot!(format!("strategy={strategy:?} block_size={block_size}"));
}

// =========================================================================
// Section 6 — Int4 quantization with asymmetric mode
// =========================================================================

#[test]
fn w25_int4_quantize_asymmetric() {
    let data: Vec<f32> = (0..64).map(|i| i as f32 / 10.0).collect();
    let cfg = Int4QuantConfig { group_size: 32, symmetric: false, block_wise: true };
    let (packed, params) = quantize_tensor_int4(&data, &cfg);
    insta::assert_snapshot!(format!(
        "num_groups={} symmetric={} scales_len={} len={}",
        params.num_groups,
        params.symmetric,
        params.scales.len(),
        packed.len
    ));
}

#[test]
fn w25_int4_quantize_per_tensor() {
    let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let cfg = Int4QuantConfig { group_size: 128, symmetric: true, block_wise: false };
    let (packed, params) = quantize_tensor_int4(&data, &cfg);
    insta::assert_snapshot!(format!(
        "num_groups={} group_size={} len={}",
        params.num_groups, params.group_size, packed.len
    ));
}
