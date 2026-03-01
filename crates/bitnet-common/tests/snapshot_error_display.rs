//! Wave 8 snapshot tests — error type Display implementations.
//!
//! Pins the user-facing error messages so that accidental changes to
//! Display formatting are caught at review time.

use bitnet_common::backend_selection::BackendRequest;
use bitnet_common::kernel_registry::KernelBackend;
use bitnet_common::tensor_validation::TensorValidationError;
use bitnet_common::{
    BackendSelectionError, InferenceError, KernelError, ModelError, QuantizationError,
    SecurityError,
};

// ---------------------------------------------------------------------------
// ModelError Display
// ---------------------------------------------------------------------------

#[test]
fn model_error_not_found_display() {
    let err = ModelError::NotFound { path: "/models/missing.gguf".into() };
    insta::assert_snapshot!("model_error_not_found", err.to_string());
}

#[test]
fn model_error_invalid_format_display() {
    let err = ModelError::InvalidFormat { format: "safetensors-v3".into() };
    insta::assert_snapshot!("model_error_invalid_format", err.to_string());
}

#[test]
fn model_error_loading_failed_display() {
    let err = ModelError::LoadingFailed { reason: "corrupted header at offset 0x42".into() };
    insta::assert_snapshot!("model_error_loading_failed", err.to_string());
}

#[test]
fn model_error_unsupported_version_display() {
    let err = ModelError::UnsupportedVersion { version: "4.0".into() };
    insta::assert_snapshot!("model_error_unsupported_version", err.to_string());
}

// ---------------------------------------------------------------------------
// QuantizationError Display
// ---------------------------------------------------------------------------

#[test]
fn quantization_error_all_variants_display() {
    let errors: Vec<QuantizationError> = vec![
        QuantizationError::UnsupportedType { qtype: "Q4_K_M".into() },
        QuantizationError::QuantizationFailed { reason: "scale underflow".into() },
        QuantizationError::InvalidBlockSize { size: 7 },
        QuantizationError::ResourceLimit { reason: "exceeded 4GB allocation".into() },
        QuantizationError::InvalidInput { reason: "shape [0, 128] has zero dim".into() },
        QuantizationError::MemoryAllocation { reason: "out of memory".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("quantization_error_all_variants", displays);
}

// ---------------------------------------------------------------------------
// KernelError Display
// ---------------------------------------------------------------------------

#[test]
fn kernel_error_all_variants_display() {
    let errors: Vec<KernelError> = vec![
        KernelError::NoProvider,
        KernelError::ExecutionFailed { reason: "CUDA out of memory".into() },
        KernelError::UnsupportedArchitecture { arch: "riscv64".into() },
        KernelError::GpuError { reason: "device lost".into() },
        KernelError::UnsupportedHardware { required: "AVX2".into(), available: "SSE4.2".into() },
        KernelError::InvalidArguments { reason: "batch size must be > 0".into() },
        KernelError::QuantizationFailed { reason: "invalid block alignment".into() },
        KernelError::MatmulFailed { reason: "dimension mismatch: 128 vs 256".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_error_all_variants", displays);
}

// ---------------------------------------------------------------------------
// InferenceError Display
// ---------------------------------------------------------------------------

#[test]
fn inference_error_all_variants_display() {
    let errors: Vec<InferenceError> = vec![
        InferenceError::GenerationFailed { reason: "KV cache overflow".into() },
        InferenceError::InvalidInput { reason: "empty prompt".into() },
        InferenceError::ContextLengthExceeded { length: 8192 },
        InferenceError::TokenizationFailed { reason: "unknown byte sequence".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("inference_error_all_variants", displays);
}

// ---------------------------------------------------------------------------
// SecurityError Display
// ---------------------------------------------------------------------------

#[test]
fn security_error_all_variants_display() {
    let errors: Vec<SecurityError> = vec![
        SecurityError::InputValidation { reason: "prompt exceeds 1MB".into() },
        SecurityError::MemoryBomb { reason: "tensor claims 999TB".into() },
        SecurityError::ResourceLimit {
            resource: "memory".into(),
            value: 5_000_000_000,
            limit: 4_000_000_000,
        },
        SecurityError::MalformedData { reason: "negative tensor dimension".into() },
        SecurityError::UnsafeOperation {
            operation: "mmap".into(),
            reason: "file from untrusted source".into(),
        },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("security_error_all_variants", displays);
}

// ---------------------------------------------------------------------------
// BackendSelectionError Display
// ---------------------------------------------------------------------------

#[test]
fn backend_selection_error_no_backend_display() {
    let err = BackendSelectionError::NoBackendAvailable;
    insta::assert_snapshot!("backend_selection_error_no_backend", err.to_string());
}

#[test]
fn backend_selection_error_unavailable_display() {
    let err = BackendSelectionError::RequestedUnavailable {
        requested: BackendRequest::Gpu,
        available: vec![KernelBackend::CpuRust],
    };
    insta::assert_snapshot!("backend_selection_error_unavailable", err.to_string());
}

// ---------------------------------------------------------------------------
// TensorValidationError Display
// ---------------------------------------------------------------------------

#[test]
fn tensor_validation_error_all_variants_display() {
    let errors: Vec<TensorValidationError> = vec![
        TensorValidationError::ZeroDimension { axis: 1, shape: vec![32, 0, 64] },
        TensorValidationError::TotalElementsExceeded { total: 2_000_000_000, max: 1_000_000_000 },
        TensorValidationError::DimensionsExceeded { ndim: 9, max: 8 },
        TensorValidationError::NanDetected { index: 42 },
        TensorValidationError::InfDetected { index: 7, value: f32::INFINITY },
        TensorValidationError::ValueOutOfRange { index: 3, value: 999.0, min: -1.0, max: 1.0 },
        TensorValidationError::StrideInconsistent { axis: 2, expected: 64, actual: 128 },
        TensorValidationError::AlignmentViolation { required: 32, actual: 4 },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!("tensor_validation_error_all_variants", displays);
}

// ---------------------------------------------------------------------------
// Error source chains (std::error::Error::source)
// ---------------------------------------------------------------------------

#[test]
fn model_error_wraps_in_bitnet_error() {
    let inner = ModelError::NotFound { path: "/missing.gguf".into() };
    let outer = bitnet_common::BitNetError::Model(inner);
    insta::assert_snapshot!("bitnet_error_model_wrapper", outer.to_string());
}

#[test]
fn quantization_error_wraps_in_bitnet_error() {
    let inner = QuantizationError::InvalidBlockSize { size: 0 };
    let outer = bitnet_common::BitNetError::Quantization(inner);
    insta::assert_snapshot!("bitnet_error_quantization_wrapper", outer.to_string());
}

#[test]
fn kernel_error_wraps_in_bitnet_error() {
    let inner = KernelError::NoProvider;
    let outer = bitnet_common::BitNetError::Kernel(inner);
    insta::assert_snapshot!("bitnet_error_kernel_wrapper", outer.to_string());
}

#[test]
fn inference_error_wraps_in_bitnet_error() {
    let inner = InferenceError::ContextLengthExceeded { length: 4096 };
    let outer = bitnet_common::BitNetError::Inference(inner);
    insta::assert_snapshot!("bitnet_error_inference_wrapper", outer.to_string());
}

#[test]
fn security_error_wraps_in_bitnet_error() {
    let inner = SecurityError::MemoryBomb { reason: "oversized allocation".into() };
    let outer = bitnet_common::BitNetError::Security(inner);
    insta::assert_snapshot!("bitnet_error_security_wrapper", outer.to_string());
}

#[test]
fn config_error_display() {
    let err = bitnet_common::BitNetError::Config("missing required field 'model_path'".into());
    insta::assert_snapshot!("bitnet_error_config", err.to_string());
}

#[test]
fn validation_error_display() {
    let err = bitnet_common::BitNetError::Validation("LayerNorm gamma RMS out of range".into());
    insta::assert_snapshot!("bitnet_error_validation", err.to_string());
}

#[test]
fn strict_mode_error_display() {
    let err = bitnet_common::BitNetError::StrictMode("mock inference rejected".into());
    insta::assert_snapshot!("bitnet_error_strict_mode", err.to_string());
}

// ---------------------------------------------------------------------------
// Error source() chain verification
// ---------------------------------------------------------------------------

#[test]
fn model_error_source_chain() {
    use std::error::Error;
    let inner = ModelError::NotFound { path: "/model.gguf".into() };
    let outer = bitnet_common::BitNetError::Model(inner);
    let source = outer.source().map(|s| s.to_string()).unwrap_or_default();
    insta::assert_snapshot!("bitnet_error_model_source", source);
}

#[test]
fn kernel_error_source_chain() {
    use std::error::Error;
    let inner = KernelError::ExecutionFailed { reason: "timeout".into() };
    let outer = bitnet_common::BitNetError::Kernel(inner);
    let source = outer.source().map(|s| s.to_string()).unwrap_or_default();
    insta::assert_snapshot!("bitnet_error_kernel_source", source);
}
