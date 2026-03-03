//! Snapshot wave 30 — bitnet-common
//!
//! Pins Debug/Display/JSON representations of foundational types:
//! Device enum, QuantizationType, error format strings, config defaults.

use bitnet_common::config::{
    ActivationType, BitNetConfig, ModelFormat, NormType, PerformanceConfig, QuantizationConfig,
};
use bitnet_common::error::{
    InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    ValidationErrorDetails,
};
use bitnet_common::types::{Device, QuantizationType};

// =========================================================================
// Section 1 — Device enum
// =========================================================================

#[test]
fn snapshot_wave30__device_cpu_debug() {
    insta::assert_debug_snapshot!(Device::Cpu);
}

#[test]
fn snapshot_wave30__device_all_variants_debug() {
    let devices = [
        Device::Cpu,
        Device::Cuda(0),
        Device::Cuda(1),
        Device::Hip(0),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(0),
    ];
    insta::assert_debug_snapshot!(devices);
}

#[test]
fn snapshot_wave30__device_all_variants_json() {
    let devices = [
        Device::Cpu,
        Device::Cuda(0),
        Device::Hip(0),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(0),
    ];
    insta::assert_json_snapshot!(devices);
}

#[test]
fn snapshot_wave30__device_default() {
    insta::assert_debug_snapshot!(Device::default());
}

// =========================================================================
// Section 2 — QuantizationType
// =========================================================================

#[test]
fn snapshot_wave30__quantization_type_all_display() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let output: Vec<String> = types.iter().map(|t| format!("{t}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn snapshot_wave30__quantization_type_all_json() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_json_snapshot!(types);
}

// =========================================================================
// Section 3 — Error format strings
// =========================================================================

#[test]
fn snapshot_wave30__model_error_not_found_display() {
    let err = ModelError::NotFound { path: "/tmp/missing.gguf".to_string() };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn snapshot_wave30__model_error_invalid_format_display() {
    let err = ModelError::InvalidFormat { format: "safetensors-v3".to_string() };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn snapshot_wave30__kernel_error_no_provider_display() {
    let err = KernelError::NoProvider;
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn snapshot_wave30__quantization_error_invalid_block_display() {
    let err = QuantizationError::InvalidBlockSize { size: 257 };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn snapshot_wave30__inference_error_context_exceeded_display() {
    let err = InferenceError::ContextLengthExceeded { length: 65536 };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn snapshot_wave30__security_error_resource_limit_display() {
    let err = SecurityError::ResourceLimit {
        resource: "tensor_elements".to_string(),
        value: 2_000_000_000,
        limit: 1_000_000_000,
    };
    insta::assert_snapshot!(format!("{err}"));
}

// =========================================================================
// Section 4 — Config defaults
// =========================================================================

#[test]
fn snapshot_wave30__quantization_config_default() {
    let cfg = QuantizationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__performance_config_default() {
    let cfg = PerformanceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__config_enums_defaults_debug() {
    let enums = (NormType::default(), ActivationType::default(), ModelFormat::default());
    insta::assert_debug_snapshot!(enums);
}
