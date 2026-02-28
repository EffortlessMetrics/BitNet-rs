//! Snapshot tests for `bitnet-common` public API surface.
//!
//! These tests pin the Display / Debug formats of key types so that
//! unintentional changes are caught at review time.

use bitnet_common::BitNetConfig;
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

// ---------------------------------------------------------------------------
// BitNetConfig
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_default_json_snapshot() {
    let cfg = BitNetConfig::default();
    insta::assert_json_snapshot!("bitnet_config_default", cfg);
}

// ---------------------------------------------------------------------------
// SimdLevel
// ---------------------------------------------------------------------------

#[test]
fn simd_level_display_all_variants() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let displays: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!("simd_level_display_variants", displays);
}

#[test]
fn simd_level_ordering_is_ascending() {
    let ordered = {
        let mut v = [
            SimdLevel::Avx512,
            SimdLevel::Scalar,
            SimdLevel::Neon,
            SimdLevel::Avx2,
            SimdLevel::Sse42,
        ];
        v.sort();
        v.iter().map(|l| format!("{l}")).collect::<Vec<_>>()
    };
    insta::assert_debug_snapshot!("simd_level_sorted_order", ordered);
}

// ---------------------------------------------------------------------------
// KernelBackend
// ---------------------------------------------------------------------------

#[test]
fn kernel_backend_display_all_variants() {
    let backends = [KernelBackend::CpuRust, KernelBackend::Cuda, KernelBackend::CppFfi];
    let displays: Vec<String> = backends.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_backend_display_variants", displays);
}

// ---------------------------------------------------------------------------
// KernelCapabilities
// ---------------------------------------------------------------------------

#[test]
fn kernel_capabilities_cpu_only_snapshot() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    insta::assert_debug_snapshot!("kernel_capabilities_cpu_avx2", caps);
}

#[test]
fn kernel_capabilities_from_compile_time_is_deterministic() {
    let a = KernelCapabilities::from_compile_time();
    let b = KernelCapabilities::from_compile_time();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

// -- Wave 3: kernel registry types -------------------------------------------

#[test]
fn kernel_backend_all_variants_debug() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::CppFfi,
    ];
    let debug: Vec<String> = backends.iter().map(|b| format!("{b:?}")).collect();
    insta::assert_debug_snapshot!("kernel_backend_all_variants", debug);
}

#[test]
fn kernel_capabilities_full_gpu_snapshot() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx512,
    };
    insta::assert_debug_snapshot!("kernel_capabilities_full_gpu", caps);
}

#[test]
fn kernel_simd_level_all_variants_debug() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon];
    let debug: Vec<String> = levels.iter().map(|l| format!("{l:?}")).collect();
    insta::assert_debug_snapshot!("kernel_simd_level_all_variants", debug);
}

// -- Wave 3: error type Display output ---------------------------------------

use bitnet_common::error::{
    BitNetError, InferenceError, ModelError, QuantizationError, SecurityError, SecurityLimits,
    ValidationErrorDetails,
};

#[test]
fn error_model_not_found_display() {
    let err = BitNetError::Model(ModelError::NotFound { path: "/tmp/model.gguf".into() });
    insta::assert_snapshot!("error_model_not_found", format!("{err}"));
}

#[test]
fn error_model_invalid_format_display() {
    let err = BitNetError::Model(ModelError::InvalidFormat { format: "bad magic bytes".into() });
    insta::assert_snapshot!("error_model_invalid_format", format!("{err}"));
}

#[test]
fn error_inference_generation_failed_display() {
    let err =
        BitNetError::Inference(InferenceError::GenerationFailed { reason: "NaN logits".into() });
    insta::assert_snapshot!("error_inference_generation_failed", format!("{err}"));
}

#[test]
fn error_inference_context_exceeded_display() {
    let err = BitNetError::Inference(InferenceError::ContextLengthExceeded { length: 8192 });
    insta::assert_snapshot!("error_inference_context_exceeded", format!("{err}"));
}

#[test]
fn error_quantization_unsupported_display() {
    let err =
        BitNetError::Quantization(QuantizationError::UnsupportedType { qtype: "IQ3_S".into() });
    insta::assert_snapshot!("error_quantization_unsupported", format!("{err}"));
}

#[test]
fn error_quantization_invalid_block_display() {
    let err = BitNetError::Quantization(QuantizationError::InvalidBlockSize { size: 128 });
    insta::assert_snapshot!("error_quantization_invalid_block", format!("{err}"));
}

#[test]
fn error_config_display() {
    let err = BitNetError::Config("temperature must be >= 0".into());
    insta::assert_snapshot!("error_config_display", format!("{err}"));
}

#[test]
fn error_validation_display() {
    let err = BitNetError::Validation("tensor shape mismatch for output.weight".into());
    insta::assert_snapshot!("error_validation_display", format!("{err}"));
}

#[test]
fn error_strict_mode_display() {
    let err = BitNetError::StrictMode("mock inference rejected".into());
    insta::assert_snapshot!("error_strict_mode_display", format!("{err}"));
}

#[test]
fn model_error_variants_display() {
    let errors: Vec<ModelError> = vec![
        ModelError::NotFound { path: "missing.gguf".into() },
        ModelError::InvalidFormat { format: "corrupt header".into() },
        ModelError::UnsupportedVersion { version: "99".into() },
        ModelError::LoadingFailed { reason: "mmap failed".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("model_error_variants", displays);
}

#[test]
fn inference_error_variants_display() {
    let errors: Vec<InferenceError> = vec![
        InferenceError::GenerationFailed { reason: "NaN logits".into() },
        InferenceError::InvalidInput { reason: "empty prompt".into() },
        InferenceError::ContextLengthExceeded { length: 4096 },
        InferenceError::TokenizationFailed { reason: "unknown token".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("inference_error_variants", displays);
}

#[test]
fn quantization_error_variants_display() {
    let errors: Vec<QuantizationError> = vec![
        QuantizationError::UnsupportedType { qtype: "IQ4_XS".into() },
        QuantizationError::InvalidBlockSize { size: 128 },
        QuantizationError::QuantizationFailed { reason: "overflow".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("quantization_error_variants", displays);
}

#[test]
fn security_error_variants_display() {
    let errors: Vec<SecurityError> = vec![
        SecurityError::InputValidation { reason: "null bytes in input".into() },
        SecurityError::MemoryBomb { reason: "allocation > 4GB".into() },
        SecurityError::ResourceLimit {
            resource: "tensor_elements".into(),
            value: 2_000_000_000,
            limit: 1_000_000_000,
        },
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("security_error_variants", displays);
}

// -- Wave 3: security & validation types -------------------------------------

#[test]
fn security_limits_default_snapshot() {
    let limits = SecurityLimits::default();
    insta::assert_snapshot!(
        "security_limits_default",
        format!(
            "max_tensor_elements={} max_memory_allocation={} max_metadata_size={} max_string_length={} max_array_length={}",
            limits.max_tensor_elements,
            limits.max_memory_allocation,
            limits.max_metadata_size,
            limits.max_string_length,
            limits.max_array_length,
        )
    );
}

#[test]
fn validation_error_details_debug() {
    let details = ValidationErrorDetails {
        errors: vec!["missing required tensor".into()],
        warnings: vec!["unusual embedding size".into()],
        recommendations: vec!["re-export with F16 LayerNorm".into()],
    };
    insta::assert_debug_snapshot!("validation_error_details", details);
}
