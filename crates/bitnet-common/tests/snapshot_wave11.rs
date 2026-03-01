//! Wave 11 snapshot tests for `bitnet-common` — error types, strict mode
//! types, backend selection, and config enums not covered in wave 5.
//!
//! Pins Debug/Display representations so that unintentional changes are
//! caught at review time.

// =========================================================================
// Section 1 — Error type Display snapshots
// =========================================================================

use bitnet_common::{
    InferenceError, KernelError, QuantizationError, SecurityError, ValidationErrorDetails,
};

// ── QuantizationError ─────────────────────────────────────────────

#[test]
fn quantization_error_all_variants_display() {
    let errors: Vec<Box<dyn std::fmt::Display>> = vec![
        Box::new(QuantizationError::UnsupportedType { qtype: "Q4_K".into() }),
        Box::new(QuantizationError::QuantizationFailed { reason: "overflow".into() }),
        Box::new(QuantizationError::InvalidBlockSize { size: 7 }),
        Box::new(QuantizationError::ResourceLimit { reason: "OOM".into() }),
        Box::new(QuantizationError::InvalidInput { reason: "wrong shape".into() }),
        Box::new(QuantizationError::MemoryAllocation { reason: "no memory".into() }),
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ── KernelError ───────────────────────────────────────────────────

#[test]
fn kernel_error_all_variants_display() {
    let errors: Vec<KernelError> = vec![
        KernelError::NoProvider,
        KernelError::ExecutionFailed { reason: "timeout".into() },
        KernelError::UnsupportedArchitecture { arch: "mips".into() },
        KernelError::GpuError { reason: "no device".into() },
        KernelError::UnsupportedHardware { required: "avx512".into(), available: "sse4.2".into() },
        KernelError::InvalidArguments { reason: "dim mismatch".into() },
        KernelError::QuantizationFailed { reason: "bad block".into() },
        KernelError::MatmulFailed { reason: "shape error".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ── InferenceError ────────────────────────────────────────────────

#[test]
fn inference_error_all_variants_display() {
    let errors: Vec<InferenceError> = vec![
        InferenceError::GenerationFailed { reason: "diverged".into() },
        InferenceError::InvalidInput { reason: "empty prompt".into() },
        InferenceError::ContextLengthExceeded { length: 8192 },
        InferenceError::TokenizationFailed { reason: "unknown token".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ── SecurityError ─────────────────────────────────────────────────

#[test]
fn security_error_all_variants_display() {
    let errors: Vec<SecurityError> = vec![
        SecurityError::InputValidation { reason: "bad utf8".into() },
        SecurityError::MemoryBomb { reason: "2TB alloc".into() },
        SecurityError::ResourceLimit {
            resource: "tensor_elements".into(),
            value: 2_000_000_000,
            limit: 1_000_000_000,
        },
        SecurityError::MalformedData { reason: "truncated header".into() },
        SecurityError::UnsafeOperation { operation: "mmap".into(), reason: "symlink".into() },
    ];
    let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// ── ValidationErrorDetails ────────────────────────────────────────

#[test]
fn validation_error_details_debug() {
    let details = ValidationErrorDetails {
        errors: vec!["missing weight".into()],
        warnings: vec!["quantized LayerNorm".into()],
        recommendations: vec!["re-export with F16".into()],
    };
    insta::assert_debug_snapshot!(details);
}

#[test]
fn validation_error_details_empty_debug() {
    let details =
        ValidationErrorDetails { errors: vec![], warnings: vec![], recommendations: vec![] };
    insta::assert_debug_snapshot!(details);
}

// =========================================================================
// Section 2 — Backend selection types
// =========================================================================

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_common::{
    BackendRequest, BackendSelectionError, BackendSelectionResult, select_backend,
};

#[test]
fn backend_selection_result_debug() {
    let result = BackendSelectionResult {
        requested: BackendRequest::Auto,
        detected: vec![KernelBackend::CpuRust],
        selected: KernelBackend::CpuRust,
        rationale: "auto-selected best available backend".into(),
    };
    insta::assert_debug_snapshot!(result);
}

#[test]
fn backend_selection_result_summary() {
    let result = BackendSelectionResult {
        requested: BackendRequest::Gpu,
        detected: vec![KernelBackend::Cuda, KernelBackend::CpuRust],
        selected: KernelBackend::Cuda,
        rationale: "GPU explicitly requested".into(),
    };
    insta::assert_snapshot!(result.summary());
}

#[test]
fn backend_selection_error_no_backend_display() {
    let err = BackendSelectionError::NoBackendAvailable;
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn backend_selection_error_unavailable_display() {
    let err = BackendSelectionError::RequestedUnavailable {
        requested: BackendRequest::Cuda,
        available: vec![KernelBackend::CpuRust],
    };
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn select_backend_auto_cpu_only() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    let result = select_backend(BackendRequest::Auto, &caps).unwrap();
    insta::assert_snapshot!(result.summary());
}

// =========================================================================
// Section 3 — Strict mode types
// =========================================================================

use bitnet_common::types::{Device, QuantizationType};
use bitnet_common::{ComputationType, MissingKernelScenario, MockInferencePath};

#[test]
fn computation_type_all_variants_debug() {
    let types = [ComputationType::Real, ComputationType::Mock];
    let debugs: Vec<String> = types.iter().map(|t| format!("{t:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

#[test]
fn computation_type_default_is_real() {
    let ct = ComputationType::default();
    insta::assert_snapshot!(format!("{ct:?}"));
}

#[test]
fn mock_inference_path_debug() {
    let path = MockInferencePath {
        description: "synthetic forward pass".into(),
        uses_mock_computation: true,
        fallback_reason: "no CUDA runtime".into(),
    };
    insta::assert_debug_snapshot!(path);
}

#[test]
fn missing_kernel_scenario_debug() {
    let scenario = MissingKernelScenario {
        quantization_type: QuantizationType::I2S,
        device: Device::Cuda(0),
        fallback_available: true,
    };
    insta::assert_debug_snapshot!(scenario);
}

// =========================================================================
// Section 4 — StrictModeConfig
// =========================================================================

use bitnet_common::StrictModeConfig;

#[test]
fn strict_mode_config_disabled_debug() {
    let cfg = StrictModeConfig {
        enabled: false,
        fail_on_mock: false,
        require_quantization: false,
        enforce_quantized_inference: false,
        validate_performance: false,
        ci_enhanced_mode: false,
        log_all_validations: false,
        fail_fast_on_any_mock: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn strict_mode_config_enabled_debug() {
    let cfg = StrictModeConfig {
        enabled: true,
        fail_on_mock: true,
        require_quantization: true,
        enforce_quantized_inference: true,
        validate_performance: true,
        ci_enhanced_mode: true,
        log_all_validations: true,
        fail_fast_on_any_mock: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

// =========================================================================
// Section 5 — KernelCapabilities Debug and methods
// =========================================================================

#[test]
fn kernel_capabilities_full_debug() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn kernel_capabilities_best_available_cpu_only() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    };
    insta::assert_debug_snapshot!(caps.best_available());
}

#[test]
fn kernel_capabilities_best_available_with_cuda() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    insta::assert_debug_snapshot!(caps.best_available());
}

#[test]
fn simd_level_all_variants_debug() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let debugs: Vec<String> = levels.iter().map(|l| format!("{l:?}")).collect();
    insta::assert_debug_snapshot!(debugs);
}

#[test]
fn simd_level_display_all_variants() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let displays: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!(displays);
}

// =========================================================================
// Section 6 — BitNetError wrapper Display
// =========================================================================

use bitnet_common::BitNetError;

#[test]
fn bitnet_error_kernel_variant_display() {
    let err = BitNetError::Kernel(KernelError::NoProvider);
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn bitnet_error_config_variant_display() {
    let err = BitNetError::Config("missing field 'vocab_size'".into());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn bitnet_error_strict_mode_variant_display() {
    let err = BitNetError::StrictMode("suspicious LayerNorm gamma".into());
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn bitnet_error_validation_variant_display() {
    let err = BitNetError::Validation("shape mismatch in layer 3".into());
    insta::assert_snapshot!(err.to_string());
}
