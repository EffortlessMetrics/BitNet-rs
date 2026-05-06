//! Wave 28 insta snapshot tests for bitnet-common types.

use bitnet_common::{
    BitNetError, Device, InferenceError, KernelBackend, KernelCapabilities, ModelError,
    QuantizationError, QuantizationType, SimdLevel,
};

// ── Device ───────────────────────────────────────────────────────────────────

#[test]
fn snapshot_device_cpu() {
    insta::assert_debug_snapshot!(Device::Cpu);
}

#[test]
fn snapshot_device_cuda_0() {
    insta::assert_debug_snapshot!(Device::Cuda(0));
}

#[test]
fn snapshot_device_cuda_1() {
    insta::assert_debug_snapshot!(Device::Cuda(1));
}

// ── QuantizationType ─────────────────────────────────────────────────────────

#[test]
fn snapshot_quantization_type_all_variants() {
    let variants = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snapshot_quantization_type_display() {
    let displays: Vec<String> =
        vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2]
            .into_iter()
            .map(|q| q.to_string())
            .collect();
    insta::assert_debug_snapshot!(displays);
}

// ── KernelBackend / SimdLevel ────────────────────────────────────────────────

#[test]
fn snapshot_kernel_backend_variants() {
    let backends = vec![
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
        KernelBackend::CppFfi,
    ];
    insta::assert_debug_snapshot!(backends);
}

#[test]
fn snapshot_kernel_backend_display() {
    let displays: Vec<String> = vec![
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
        KernelBackend::CppFfi,
    ]
    .into_iter()
    .map(|b| b.to_string())
    .collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn snapshot_simd_level_all_variants() {
    let levels = vec![
        SimdLevel::Scalar,
        SimdLevel::Neon,
        SimdLevel::Sse42,
        SimdLevel::Avx2,
        SimdLevel::Avx512,
    ];
    insta::assert_debug_snapshot!(levels);
}

#[test]
fn snapshot_kernel_capabilities_compile_time() {
    let caps = KernelCapabilities::from_compile_time();
    insta::assert_debug_snapshot!(caps);
}

// ── Error Display ────────────────────────────────────────────────────────────

#[test]
fn snapshot_bitnet_error_config_display() {
    let err = BitNetError::Config("missing model path".to_string());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_bitnet_error_validation_display() {
    let err = BitNetError::Validation("tensor shape mismatch".to_string());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_bitnet_error_strict_mode_display() {
    let err = BitNetError::StrictMode("suspicious LayerNorm gamma".to_string());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_inference_error_display() {
    let err = BitNetError::Inference(InferenceError::GenerationFailed {
        reason: "out of memory".to_string(),
    });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_model_error_not_found_display() {
    let err = BitNetError::Model(ModelError::NotFound { path: "/tmp/missing.gguf".to_string() });
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn snapshot_quantization_error_invalid_block_display() {
    let err = BitNetError::Quantization(QuantizationError::InvalidBlockSize { size: 0 });
    insta::assert_snapshot!(err.to_string());
}
