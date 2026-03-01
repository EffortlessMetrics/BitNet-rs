//! Wave 10 snapshot tests for bitnet-quantization.
//!
//! Covers: PipelineConfig validation errors, Precision variants, QuantizationStage
//! ordering, QuantizedTensor metadata, error message formats.

use bitnet_common::{QuantizationError, QuantizationType};
use bitnet_quantization::pipeline::{PipelineConfig, Precision, QuantizationStage};

// -- Precision variants Debug ------------------------------------------------

#[test]
fn precision_all_variants_debug() {
    let variants = [Precision::F32, Precision::I2S, Precision::TL1, Precision::TL2];
    let debug: Vec<String> = variants.iter().map(|p| format!("{p:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

// -- QuantizationStage variants Debug ----------------------------------------

#[test]
fn quantization_stage_all_variants_debug() {
    let stages = [
        QuantizationStage::Calibration,
        QuantizationStage::Quantization,
        QuantizationStage::Verification,
        QuantizationStage::PackingOptimization,
    ];
    let debug: Vec<String> = stages.iter().map(|s| format!("{s:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

#[test]
fn quantization_stage_ordering() {
    let ordered = [
        QuantizationStage::Calibration,
        QuantizationStage::Quantization,
        QuantizationStage::Verification,
        QuantizationStage::PackingOptimization,
    ];
    let order_vals: Vec<String> = ordered.iter().map(|s| format!("{s:?}={}", *s as u8)).collect();
    insta::assert_snapshot!(order_vals.join(", "));
}

// -- PipelineConfig validation errors ----------------------------------------

#[test]
fn pipeline_config_target_f32_error() {
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
fn pipeline_config_zero_calibration_samples_error() {
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
fn pipeline_config_zero_error_threshold_error() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: 0.0,
    };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn pipeline_config_negative_error_threshold_error() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL2,
        calibration_samples: 50,
        error_threshold: -0.5,
    };
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// -- PipelineConfig valid configuration Debug --------------------------------

#[test]
fn pipeline_config_valid_i2s_debug() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::I2S,
        calibration_samples: 100,
        error_threshold: 0.01,
    };
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pipeline_config_valid_tl1_debug() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL1,
        calibration_samples: 200,
        error_threshold: 0.005,
    };
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn pipeline_config_valid_tl2_debug() {
    let cfg = PipelineConfig {
        source_precision: Precision::F32,
        target_precision: Precision::TL2,
        calibration_samples: 50,
        error_threshold: 0.1,
    };
    assert!(cfg.validate().is_ok());
    insta::assert_debug_snapshot!(cfg);
}

// -- QuantizationError Display messages --------------------------------------

#[test]
fn quantization_error_unsupported_type_display() {
    let err = QuantizationError::UnsupportedType { qtype: "Q4_0".to_string() };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn quantization_error_failed_display() {
    let err =
        QuantizationError::QuantizationFailed { reason: "tensor dimension mismatch".to_string() };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn quantization_error_invalid_block_size_display() {
    let err = QuantizationError::InvalidBlockSize { size: 17 };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn quantization_error_resource_limit_display() {
    let err = QuantizationError::ResourceLimit { reason: "exceeded 16GB memory limit".to_string() };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn quantization_error_invalid_input_display() {
    let err = QuantizationError::InvalidInput {
        reason: "tensor must have at least 32 elements".to_string(),
    };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn quantization_error_memory_allocation_display() {
    let err = QuantizationError::MemoryAllocation {
        reason: "failed to allocate 4GB for quantized output".to_string(),
    };
    insta::assert_snapshot!(err.to_string());
}

// -- QuantizationType all variants -------------------------------------------

#[test]
fn quantization_type_all_variants_display() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let displays: Vec<String> = types.iter().map(|t| format!("{t}")).collect();
    insta::assert_snapshot!(displays.join(", "));
}

// -- QuantizerFactory --------------------------------------------------------

#[test]
fn quantizer_factory_create_i2s_type() {
    use bitnet_quantization::QuantizerFactory;
    let quantizer = QuantizerFactory::create(QuantizationType::I2S);
    insta::assert_snapshot!(
        "quantizer_factory_i2s_type",
        format!("{:?}", quantizer.quantization_type())
    );
}

#[test]
fn quantizer_factory_create_tl1_type() {
    use bitnet_quantization::QuantizerFactory;
    let quantizer = QuantizerFactory::create(QuantizationType::TL1);
    insta::assert_snapshot!(
        "quantizer_factory_tl1_type",
        format!("{:?}", quantizer.quantization_type())
    );
}

#[test]
fn quantizer_factory_create_tl2_type() {
    use bitnet_quantization::QuantizerFactory;
    let quantizer = QuantizerFactory::create(QuantizationType::TL2);
    insta::assert_snapshot!(
        "quantizer_factory_tl2_type",
        format!("{:?}", quantizer.quantization_type())
    );
}
