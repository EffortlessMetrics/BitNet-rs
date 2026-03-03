//! Edge-case tests for bitnet-common kernel registry, architecture registry,
//! and backend selection.
//!
//! Tests cover: SimdLevel (Display, Ord), KernelBackend (Display, requires_gpu,
//! is_compiled), KernelCapabilities (from_compile_time, builders, compiled_backends,
//! best_available, summary), ArchitectureRegistry (lookup, known_architectures,
//! is_known, case-insensitivity), BackendRequest (Display), BackendStartupSummary
//! (new, log_line, serde), BackendSelectionResult (summary), select_backend (Auto,
//! Cpu, Gpu, Cuda, Hip, OneApi with various capability states).

use bitnet_common::backend_selection::{
    BackendRequest, BackendSelectionError, BackendStartupSummary, select_backend,
};
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

// ===========================================================================
// SimdLevel
// ===========================================================================

#[test]
fn simd_level_display_scalar() {
    assert_eq!(format!("{}", SimdLevel::Scalar), "scalar");
}

#[test]
fn simd_level_display_all_variants() {
    let expected = [
        (SimdLevel::Scalar, "scalar"),
        (SimdLevel::Neon, "neon"),
        (SimdLevel::Sse42, "sse4.2"),
        (SimdLevel::Avx2, "avx2"),
        (SimdLevel::Avx512, "avx512"),
    ];
    for (level, name) in &expected {
        assert_eq!(format!("{}", level), *name);
    }
}

#[test]
fn simd_level_ordering() {
    assert!(SimdLevel::Scalar < SimdLevel::Neon);
    assert!(SimdLevel::Neon < SimdLevel::Sse42);
    assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

#[test]
fn simd_level_equality() {
    assert_eq!(SimdLevel::Avx2, SimdLevel::Avx2);
    assert_ne!(SimdLevel::Avx2, SimdLevel::Avx512);
}

#[test]
fn simd_level_clone_copy() {
    let level = SimdLevel::Avx2;
    let cloned = level.clone();
    let copied = level;
    assert_eq!(cloned, copied);
}

// ===========================================================================
// KernelBackend
// ===========================================================================

#[test]
fn kernel_backend_display() {
    assert_eq!(format!("{}", KernelBackend::CpuRust), "cpu-rust");
    assert_eq!(format!("{}", KernelBackend::Cuda), "cuda");
    assert_eq!(format!("{}", KernelBackend::Hip), "hip");
    assert_eq!(format!("{}", KernelBackend::OneApi), "oneapi");
    assert_eq!(format!("{}", KernelBackend::OpenCL), "opencl");
    assert_eq!(format!("{}", KernelBackend::CppFfi), "cpp-ffi");
}

#[test]
fn kernel_backend_requires_gpu() {
    assert!(!KernelBackend::CpuRust.requires_gpu());
    assert!(KernelBackend::Cuda.requires_gpu());
    assert!(KernelBackend::Hip.requires_gpu());
    assert!(KernelBackend::OneApi.requires_gpu());
    assert!(KernelBackend::OpenCL.requires_gpu());
    assert!(!KernelBackend::CppFfi.requires_gpu());
}

#[test]
fn kernel_backend_is_compiled_cpu() {
    // Built with --features cpu, so CpuRust should be compiled
    if cfg!(feature = "cpu") {
        assert!(KernelBackend::CpuRust.is_compiled());
    }
}

#[test]
fn kernel_backend_cpp_ffi_never_compiled() {
    // CppFfi always returns false in is_compiled
    assert!(!KernelBackend::CppFfi.is_compiled());
}

// ===========================================================================
// KernelCapabilities — from_compile_time
// ===========================================================================

#[test]
fn kernel_capabilities_from_compile_time() {
    let caps = KernelCapabilities::from_compile_time();
    // Runtime fields should always be false when from_compile_time
    assert!(!caps.cuda_runtime);
    assert!(!caps.hip_runtime);
    assert!(!caps.oneapi_runtime);
    assert!(!caps.opencl_runtime);
    assert!(!caps.cpp_ffi);
}

// ===========================================================================
// KernelCapabilities — builder chain
// ===========================================================================

#[test]
fn kernel_capabilities_builder_chain() {
    let caps = KernelCapabilities::from_compile_time()
        .with_cuda_runtime(true)
        .with_hip_runtime(true)
        .with_oneapi_runtime(true)
        .with_opencl_runtime(true)
        .with_cpp_ffi(true);
    assert!(caps.cuda_runtime);
    assert!(caps.hip_runtime);
    assert!(caps.oneapi_runtime);
    assert!(caps.opencl_runtime);
    assert!(caps.cpp_ffi);
}

// ===========================================================================
// KernelCapabilities — compiled_backends
// ===========================================================================

#[test]
fn kernel_capabilities_compiled_backends_cpu_only() {
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
    let backends = caps.compiled_backends();
    assert_eq!(backends, vec![KernelBackend::CpuRust]);
}

#[test]
fn kernel_capabilities_compiled_backends_all() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: false,
        opencl_compiled: true,
        opencl_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    };
    let backends = caps.compiled_backends();
    assert_eq!(
        backends,
        vec![
            KernelBackend::Cuda,
            KernelBackend::Hip,
            KernelBackend::OneApi,
            KernelBackend::OpenCL,
            KernelBackend::CppFfi,
            KernelBackend::CpuRust,
        ]
    );
}

#[test]
fn kernel_capabilities_compiled_backends_empty() {
    let caps = KernelCapabilities {
        cpu_rust: false,
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
    assert!(caps.compiled_backends().is_empty());
}

// ===========================================================================
// KernelCapabilities — best_available
// ===========================================================================

#[test]
fn kernel_capabilities_best_available_cuda() {
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
    assert_eq!(caps.best_available(), Some(KernelBackend::Cuda));
}

#[test]
fn kernel_capabilities_best_available_cpu_fallback() {
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
    assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
}

#[test]
fn kernel_capabilities_best_available_none() {
    let caps = KernelCapabilities {
        cpu_rust: false,
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
    assert_eq!(caps.best_available(), None);
}

#[test]
fn kernel_capabilities_best_available_hip() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        opencl_compiled: false,
        opencl_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::Hip));
}

#[test]
fn kernel_capabilities_best_available_ffi_over_cpu() {
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
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CppFfi));
}

// ===========================================================================
// KernelCapabilities — summary
// ===========================================================================

#[test]
fn kernel_capabilities_summary_not_empty() {
    let caps = KernelCapabilities::from_compile_time();
    let summary = caps.summary();
    assert!(!summary.is_empty());
}

#[test]
fn kernel_capabilities_debug() {
    let caps = KernelCapabilities::from_compile_time();
    let dbg = format!("{caps:?}");
    assert!(dbg.contains("cpu_rust"));
}

// ===========================================================================
// ArchitectureRegistry
// ===========================================================================

#[test]
fn arch_registry_lookup_phi4() {
    use bitnet_common::config::{ActivationType, NormType};
    let defaults = bitnet_common::arch_registry::ArchitectureRegistry::lookup("phi-4").unwrap();
    assert_eq!(defaults.norm_type, NormType::RmsNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
    assert_eq!(defaults.default_context_length, Some(16384));
}

#[test]
fn arch_registry_lookup_case_insensitive() {
    let d1 = bitnet_common::arch_registry::ArchitectureRegistry::lookup("PHI-4");
    let d2 = bitnet_common::arch_registry::ArchitectureRegistry::lookup("phi-4");
    assert!(d1.is_some());
    assert!(d2.is_some());
}

#[test]
fn arch_registry_lookup_bitnet() {
    use bitnet_common::config::{ActivationType, NormType};
    let defaults = bitnet_common::arch_registry::ArchitectureRegistry::lookup("bitnet").unwrap();
    assert_eq!(defaults.norm_type, NormType::LayerNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
}

#[test]
fn arch_registry_lookup_unknown() {
    assert!(bitnet_common::arch_registry::ArchitectureRegistry::lookup("nonexistent").is_none());
}

#[test]
fn arch_registry_known_architectures_nonempty() {
    let archs = bitnet_common::arch_registry::ArchitectureRegistry::known_architectures();
    assert!(archs.len() > 20);
    assert!(archs.contains(&"phi-4"));
    assert!(archs.contains(&"llama"));
    assert!(archs.contains(&"bitnet"));
}

#[test]
fn arch_registry_is_known() {
    assert!(bitnet_common::arch_registry::ArchitectureRegistry::is_known("phi-4"));
    assert!(bitnet_common::arch_registry::ArchitectureRegistry::is_known("llama"));
    assert!(!bitnet_common::arch_registry::ArchitectureRegistry::is_known("bogus"));
}

// ===========================================================================
// BackendRequest
// ===========================================================================

#[test]
fn backend_request_display() {
    assert_eq!(format!("{}", BackendRequest::Auto), "auto");
    assert_eq!(format!("{}", BackendRequest::Cpu), "cpu");
    assert_eq!(format!("{}", BackendRequest::Gpu), "gpu");
    assert_eq!(format!("{}", BackendRequest::Cuda), "cuda");
    assert_eq!(format!("{}", BackendRequest::Hip), "hip");
    assert_eq!(format!("{}", BackendRequest::OneApi), "oneapi");
}

#[test]
fn backend_request_equality() {
    assert_eq!(BackendRequest::Auto, BackendRequest::Auto);
    assert_ne!(BackendRequest::Auto, BackendRequest::Cpu);
}

// ===========================================================================
// BackendStartupSummary
// ===========================================================================

#[test]
fn backend_startup_summary_new() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".into()], "cpu-rust");
    assert_eq!(summary.requested, "auto");
    assert_eq!(summary.detected, vec!["cpu-rust"]);
    assert_eq!(summary.selected, "cpu-rust");
}

#[test]
fn backend_startup_summary_log_line() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".into()], "cpu-rust");
    let line = summary.log_line();
    assert_eq!(line, "requested=auto detected=[cpu-rust] selected=cpu-rust");
}

#[test]
fn backend_startup_summary_log_line_multiple_detected() {
    let summary = BackendStartupSummary::new("gpu", vec!["cuda".into(), "cpu-rust".into()], "cuda");
    let line = summary.log_line();
    assert!(line.contains("cuda, cpu-rust"));
}

#[test]
fn backend_startup_summary_serde_roundtrip() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".into()], "cpu-rust");
    let json = serde_json::to_string(&summary).unwrap();
    let summary2: BackendStartupSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(summary, summary2);
}

// ===========================================================================
// select_backend
// ===========================================================================

fn cpu_only_caps() -> KernelCapabilities {
    KernelCapabilities {
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
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_runtime_caps() -> KernelCapabilities {
    KernelCapabilities {
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
    }
}

fn no_backend_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
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
    }
}

#[test]
fn select_backend_auto_cpu_only() {
    let result = select_backend(BackendRequest::Auto, &cpu_only_caps()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
    assert_eq!(result.requested, BackendRequest::Auto);
}

#[test]
fn select_backend_auto_with_cuda() {
    let result = select_backend(BackendRequest::Auto, &cuda_runtime_caps()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

#[test]
fn select_backend_auto_no_backends() {
    let result = select_backend(BackendRequest::Auto, &no_backend_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_cpu_explicit() {
    let result = select_backend(BackendRequest::Cpu, &cpu_only_caps()).unwrap();
    assert_eq!(result.selected, KernelBackend::CpuRust);
    assert!(result.rationale.contains("CPU"));
}

#[test]
fn select_backend_cpu_not_available() {
    let result = select_backend(BackendRequest::Cpu, &no_backend_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_gpu_with_cuda() {
    let result = select_backend(BackendRequest::Gpu, &cuda_runtime_caps()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

#[test]
fn select_backend_gpu_no_gpu() {
    let result = select_backend(BackendRequest::Gpu, &cpu_only_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_cuda_explicit() {
    let result = select_backend(BackendRequest::Cuda, &cuda_runtime_caps()).unwrap();
    assert_eq!(result.selected, KernelBackend::Cuda);
}

#[test]
fn select_backend_cuda_not_available() {
    let result = select_backend(BackendRequest::Cuda, &cpu_only_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_hip_not_available() {
    let result = select_backend(BackendRequest::Hip, &cpu_only_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_oneapi_not_available() {
    let result = select_backend(BackendRequest::OneApi, &cpu_only_caps());
    assert!(result.is_err());
}

#[test]
fn select_backend_result_summary() {
    let result = select_backend(BackendRequest::Auto, &cpu_only_caps()).unwrap();
    let summary = result.summary();
    assert!(summary.contains("requested=auto"));
    assert!(summary.contains("selected=cpu-rust"));
}

// ===========================================================================
// BackendSelectionError Display
// ===========================================================================

#[test]
fn backend_selection_error_no_backend_display() {
    let err = BackendSelectionError::NoBackendAvailable;
    let msg = format!("{err}");
    assert!(msg.contains("no kernel backend"));
}

#[test]
fn backend_selection_error_unavailable_display() {
    let err = BackendSelectionError::RequestedUnavailable {
        requested: BackendRequest::Cuda,
        available: vec![KernelBackend::CpuRust],
    };
    let msg = format!("{err}");
    assert!(msg.contains("cuda"));
    assert!(msg.contains("cpu-rust"));
}

#[test]
fn backend_selection_error_is_std_error() {
    let err = BackendSelectionError::NoBackendAvailable;
    let _: &dyn std::error::Error = &err;
}
