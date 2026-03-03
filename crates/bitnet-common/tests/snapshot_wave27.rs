//! Snapshot wave 27 — bitnet-common Display/Debug output stability.

use bitnet_common::backend_selection::{
    BackendRequest, BackendSelectionError, BackendSelectionResult, BackendStartupSummary,
};
use bitnet_common::error::{
    BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    ValidationErrorDetails,
};
use bitnet_common::kernel_registry::{KernelBackend, SimdLevel};
use bitnet_common::tensor_validation::ShapeError;
use bitnet_common::types::{Device, ModelMetadata, PerformanceMetrics, QuantizationType};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[test]
fn snapshot_device_debug_all_variants() {
    let devices = vec![
        Device::Cpu,
        Device::Cuda(0),
        Device::Cuda(3),
        Device::Hip(1),
        Device::Npu,
        Device::Metal,
        Device::OpenCL(0),
        Device::OpenCL(7),
    ];
    insta::assert_debug_snapshot!("device_debug_all_variants", devices);
}

#[test]
fn snapshot_device_default() {
    let dev = Device::default();
    insta::assert_debug_snapshot!("device_default", dev);
}

// ---------------------------------------------------------------------------
// QuantizationType
// ---------------------------------------------------------------------------

#[test]
fn snapshot_quantization_type_display_all() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    let displays: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    insta::assert_debug_snapshot!("quantization_type_display_all", displays);
}

#[test]
fn snapshot_quantization_type_debug_all() {
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    insta::assert_debug_snapshot!("quantization_type_debug_all", types);
}

#[test]
fn snapshot_quantization_type_json() {
    let qt = QuantizationType::I2S;
    insta::assert_json_snapshot!("quantization_type_json_i2s", qt);
}

// ---------------------------------------------------------------------------
// Error types — Display output
// ---------------------------------------------------------------------------

#[test]
fn snapshot_model_error_not_found() {
    let err = ModelError::NotFound { path: "/models/missing.gguf".into() };
    insta::assert_snapshot!("model_error_not_found", err.to_string());
}

#[test]
fn snapshot_model_error_invalid_format() {
    let err = ModelError::InvalidFormat { format: "safetensors-v3".into() };
    insta::assert_snapshot!("model_error_invalid_format", err.to_string());
}

#[test]
fn snapshot_model_error_gguf_with_details() {
    let err = ModelError::GGUFFormatError {
        message: "invalid magic number".into(),
        details: ValidationErrorDetails {
            errors: vec!["magic mismatch".into()],
            warnings: vec!["old format version".into()],
            recommendations: vec!["re-export with latest converter".into()],
        },
    };
    insta::assert_snapshot!("model_error_gguf_with_details", err.to_string());
}

#[test]
fn snapshot_quantization_error_variants() {
    let errors: Vec<String> = vec![
        QuantizationError::UnsupportedType { qtype: "Q4_K_M".into() }.to_string(),
        QuantizationError::InvalidBlockSize { size: 17 }.to_string(),
        QuantizationError::QuantizationFailed { reason: "overflow in scale".into() }.to_string(),
        QuantizationError::ResourceLimit { reason: "exceeds 4GB".into() }.to_string(),
    ];
    insta::assert_debug_snapshot!("quantization_error_variants", errors);
}

#[test]
fn snapshot_kernel_error_variants() {
    let errors: Vec<String> = vec![
        KernelError::NoProvider.to_string(),
        KernelError::ExecutionFailed { reason: "CUDA OOM".into() }.to_string(),
        KernelError::UnsupportedArchitecture { arch: "wasm32".into() }.to_string(),
        KernelError::UnsupportedHardware { required: "avx512".into(), available: "avx2".into() }
            .to_string(),
    ];
    insta::assert_debug_snapshot!("kernel_error_variants", errors);
}

#[test]
fn snapshot_inference_error_variants() {
    let errors: Vec<String> = vec![
        InferenceError::GenerationFailed { reason: "max retries exceeded".into() }.to_string(),
        InferenceError::InvalidInput { reason: "empty prompt".into() }.to_string(),
        InferenceError::ContextLengthExceeded { length: 131072 }.to_string(),
        InferenceError::TokenizationFailed { reason: "unknown BPE token".into() }.to_string(),
    ];
    insta::assert_debug_snapshot!("inference_error_variants", errors);
}

#[test]
fn snapshot_security_error_variants() {
    let errors: Vec<String> = vec![
        SecurityError::InputValidation { reason: "null byte in prompt".into() }.to_string(),
        SecurityError::MemoryBomb { reason: "tensor claims 1TB".into() }.to_string(),
        SecurityError::ResourceLimit {
            resource: "tensor_elements".into(),
            value: 2_000_000_000,
            limit: 1_000_000_000,
        }
        .to_string(),
        SecurityError::MalformedData { reason: "cyclic reference in metadata".into() }.to_string(),
        SecurityError::UnsafeOperation {
            operation: "mmap".into(),
            reason: "path traversal".into(),
        }
        .to_string(),
    ];
    insta::assert_debug_snapshot!("security_error_variants", errors);
}

#[test]
fn snapshot_bitnet_error_wrapping() {
    let inner = ModelError::NotFound { path: "test.gguf".into() };
    let outer: BitNetError = inner.into();
    insta::assert_snapshot!("bitnet_error_wrapping_model", outer.to_string());
}

// ---------------------------------------------------------------------------
// ShapeError
// ---------------------------------------------------------------------------

#[test]
fn snapshot_shape_error_matmul_mismatch() {
    let err = ShapeError::MatmulMismatch { a_inner: 768, b_inner: 512 };
    insta::assert_snapshot!("shape_error_matmul_mismatch", err.to_string());
}

#[test]
fn snapshot_shape_error_broadcast_incompatible() {
    let err = ShapeError::BroadcastIncompatible { dim: 2, a: 3, b: 5 };
    insta::assert_snapshot!("shape_error_broadcast", err.to_string());
}

#[test]
fn snapshot_shape_error_attention() {
    let err = ShapeError::AttentionShape {
        q: vec![1, 8, 64, 96],
        k: vec![1, 8, 128, 96],
        v: vec![1, 8, 128, 48],
        reason: "V head_dim != Q head_dim".into(),
    };
    insta::assert_snapshot!("shape_error_attention", err.to_string());
}

// ---------------------------------------------------------------------------
// KernelBackend / SimdLevel
// ---------------------------------------------------------------------------

#[test]
fn snapshot_kernel_backend_display_all() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::OpenCL,
        KernelBackend::CppFfi,
    ];
    let displays: Vec<String> = backends.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_backend_display_all", displays);
}

#[test]
fn snapshot_simd_level_ordering() {
    let mut levels =
        [SimdLevel::Avx512, SimdLevel::Scalar, SimdLevel::Avx2, SimdLevel::Neon, SimdLevel::Sse42];
    levels.sort();
    let sorted: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!("simd_level_sorted_order", sorted);
}

// ---------------------------------------------------------------------------
// BackendStartupSummary
// ---------------------------------------------------------------------------

#[test]
fn snapshot_backend_startup_summary_log_line() {
    let summary =
        BackendStartupSummary::new("auto", vec!["cpu-rust".into(), "cuda".into()], "cuda");
    insta::assert_snapshot!("backend_startup_log_line", summary.log_line());
}

#[test]
fn snapshot_backend_startup_summary_json() {
    let summary = BackendStartupSummary::new("gpu", vec!["cpu-rust".into()], "cpu-rust");
    insta::assert_json_snapshot!("backend_startup_summary_json", summary);
}

// ---------------------------------------------------------------------------
// BackendSelectionResult
// ---------------------------------------------------------------------------

#[test]
fn snapshot_backend_selection_result_summary() {
    let result = BackendSelectionResult {
        requested: BackendRequest::Auto,
        detected: vec![KernelBackend::CpuRust, KernelBackend::Cuda],
        selected: KernelBackend::Cuda,
        rationale: "auto-selected best available backend".into(),
    };
    insta::assert_snapshot!("backend_selection_result_summary", result.summary());
}

#[test]
fn snapshot_backend_selection_error_display() {
    let err = BackendSelectionError::RequestedUnavailable {
        requested: BackendRequest::Cuda,
        available: vec![KernelBackend::CpuRust],
    };
    insta::assert_snapshot!("backend_selection_error_unavailable", err.to_string());
}

// ---------------------------------------------------------------------------
// ModelMetadata
// ---------------------------------------------------------------------------

#[test]
fn snapshot_model_metadata_json() {
    let meta = ModelMetadata {
        name: "bitnet-b1.58-2B-4T".into(),
        version: "1.0".into(),
        architecture: "bitnet".into(),
        vocab_size: 32000,
        context_length: 4096,
        quantization: Some(QuantizationType::I2S),
        fingerprint: Some("sha256-abcdef1234567890".into()),
        corrections_applied: None,
    };
    insta::assert_json_snapshot!("model_metadata_json", meta);
}

// ---------------------------------------------------------------------------
// PerformanceMetrics
// ---------------------------------------------------------------------------

#[test]
fn snapshot_performance_metrics_json() {
    let metrics = PerformanceMetrics {
        tokens_per_second: 42.5,
        latency_ms: 23.5,
        memory_usage_mb: 1024.0,
        gpu_utilization: Some(0.85),
    };
    insta::assert_json_snapshot!("performance_metrics_json", metrics);
}

#[test]
fn snapshot_performance_metrics_default() {
    let metrics = PerformanceMetrics::default();
    insta::assert_json_snapshot!("performance_metrics_default", metrics);
}
