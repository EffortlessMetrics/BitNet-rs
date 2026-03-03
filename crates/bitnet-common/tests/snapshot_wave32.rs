//! Snapshot wave 32 — error message formatting, device display strings,
//! tensor type descriptions, and quantization type output.

use bitnet_common::{
    BitNetError, Device, InferenceError, KernelError, ModelError, QuantizationError,
    QuantizationType, SecurityError,
};

// ── Device debug strings ────────────────────────────────────────────

#[test]
fn device_debug_cpu() {
    insta::assert_snapshot!(format!("{:?}", Device::Cpu));
}

#[test]
fn device_debug_cuda_0() {
    insta::assert_snapshot!(format!("{:?}", Device::Cuda(0)));
}

#[test]
fn device_debug_cuda_1() {
    insta::assert_snapshot!(format!("{:?}", Device::Cuda(1)));
}

#[test]
fn device_debug_hip() {
    insta::assert_snapshot!(format!("{:?}", Device::Hip(0)));
}

#[test]
fn device_debug_npu() {
    insta::assert_snapshot!(format!("{:?}", Device::Npu));
}

#[test]
fn device_debug_metal() {
    insta::assert_snapshot!(format!("{:?}", Device::Metal));
}

#[test]
fn device_debug_opencl() {
    insta::assert_snapshot!(format!("{:?}", Device::OpenCL(0)));
}

#[test]
fn device_debug_all_variants() {
    let devices = vec![
        Device::Cpu,
        Device::Cuda(0),
        Device::Hip(0),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(0),
    ];
    insta::assert_debug_snapshot!(devices);
}

// ── QuantizationType display strings ────────────────────────────────

#[test]
fn quantization_type_display_i2s() {
    insta::assert_snapshot!(QuantizationType::I2S.to_string());
}

#[test]
fn quantization_type_display_tl1() {
    insta::assert_snapshot!(QuantizationType::TL1.to_string());
}

#[test]
fn quantization_type_display_tl2() {
    insta::assert_snapshot!(QuantizationType::TL2.to_string());
}

#[test]
fn quantization_type_debug_all() {
    let types = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_debug_snapshot!(types);
}

// ── Error message formatting ────────────────────────────────────────

#[test]
fn error_model_not_found() {
    let err =
        BitNetError::Model(ModelError::NotFound { path: "/tmp/nonexistent.gguf".to_string() });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_model_invalid_format() {
    let err =
        BitNetError::Model(ModelError::InvalidFormat { format: "Unsupported GGUF v1".to_string() });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_model_loading_failed() {
    let err = BitNetError::Model(ModelError::LoadingFailed {
        reason: "Corrupted tensor data at offset 0x1000".to_string(),
    });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_quantization_unsupported() {
    let err =
        BitNetError::Quantization(QuantizationError::UnsupportedType { qtype: "Q3_K".to_string() });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_quantization_invalid_block_size() {
    let err = BitNetError::Quantization(QuantizationError::InvalidBlockSize { size: 17 });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_kernel_no_provider() {
    let err = BitNetError::Kernel(KernelError::NoProvider);
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_kernel_execution_failed() {
    let err = BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "SIMD alignment violation".to_string(),
    });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_inference_generation_failed() {
    let err = BitNetError::Inference(InferenceError::GenerationFailed {
        reason: "Token limit exceeded".to_string(),
    });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_inference_context_exceeded() {
    let err = BitNetError::Inference(InferenceError::ContextLengthExceeded { length: 4096 });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_security_resource_limit() {
    let err = BitNetError::Security(SecurityError::ResourceLimit {
        resource: "memory".to_string(),
        value: 8_589_934_592,
        limit: 4_294_967_296,
    });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn error_security_memory_bomb() {
    let err = BitNetError::Security(SecurityError::MemoryBomb {
        reason: "Tensor metadata claims 1TB allocation".to_string(),
    });
    insta::assert_snapshot!(err.to_string());
}
