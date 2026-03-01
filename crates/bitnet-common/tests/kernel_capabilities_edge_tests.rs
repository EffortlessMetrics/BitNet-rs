//! Edge case tests for kernel registry capabilities and backend selection.
//!
//! Validates KernelCapabilities, KernelBackend, and SimdLevel behavior under
//! various compile-time and runtime configurations.

use bitnet_common::{KernelBackend, KernelCapabilities, SimdLevel};

// --- SimdLevel ordering and properties ---

#[test]
fn simd_level_scalar_is_lowest() {
    assert!(SimdLevel::Scalar <= SimdLevel::Sse42);
    assert!(SimdLevel::Scalar <= SimdLevel::Avx2);
    assert!(SimdLevel::Scalar <= SimdLevel::Avx512);
    assert!(SimdLevel::Scalar <= SimdLevel::Neon);
}

#[test]
fn simd_level_x86_chain() {
    assert!(SimdLevel::Scalar < SimdLevel::Sse42);
    assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

#[test]
fn simd_level_all_variants_clone() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for level in &levels {
        let cloned = *level;
        assert_eq!(*level, cloned);
    }
}

#[test]
fn simd_level_debug_non_empty() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for level in &levels {
        assert!(!format!("{level:?}").is_empty());
    }
}

#[test]
fn simd_level_partial_eq() {
    assert_eq!(SimdLevel::Avx2, SimdLevel::Avx2);
    assert_ne!(SimdLevel::Avx2, SimdLevel::Avx512);
    assert_ne!(SimdLevel::Neon, SimdLevel::Sse42);
}

// --- KernelBackend ---

#[test]
fn kernel_backend_requires_gpu() {
    assert!(KernelBackend::Cuda.requires_gpu());
    assert!(KernelBackend::Hip.requires_gpu());
    assert!(!KernelBackend::CpuRust.requires_gpu());
    assert!(!KernelBackend::CppFfi.requires_gpu());
}

#[test]
fn kernel_backend_oneapi_requires_gpu() {
    // OneApi may or may not require GPU depending on implementation
    let _ = KernelBackend::OneApi.requires_gpu();
}

#[test]
fn kernel_backend_all_variants_debug() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::CppFfi,
    ];
    for b in &backends {
        assert!(!format!("{b:?}").is_empty());
    }
}

#[test]
fn kernel_backend_clone_eq() {
    let b = KernelBackend::Cuda;
    let b2 = b.clone();
    assert_eq!(b, b2);
}

#[test]
fn kernel_backend_all_distinct() {
    let backends = [
        KernelBackend::CpuRust,
        KernelBackend::Cuda,
        KernelBackend::Hip,
        KernelBackend::OneApi,
        KernelBackend::CppFfi,
    ];
    for (i, a) in backends.iter().enumerate() {
        for (j, b) in backends.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b, "{a:?} should differ from {b:?}");
            }
        }
    }
}

// --- KernelCapabilities ---

#[test]
fn capabilities_from_compile_time_has_cpu_rust() {
    let caps = KernelCapabilities::from_compile_time();
    assert!(caps.cpu_rust, "CPU Rust should always be available");
}

#[test]
fn capabilities_from_compile_time_simd_is_valid() {
    let caps = KernelCapabilities::from_compile_time();
    // SIMD level should be one of the known variants
    let valid = matches!(
        caps.simd_level,
        SimdLevel::Scalar
            | SimdLevel::Neon
            | SimdLevel::Sse42
            | SimdLevel::Avx2
            | SimdLevel::Avx512
    );
    assert!(valid, "SIMD level should be a valid variant");
}

#[test]
fn capabilities_compiled_backends_includes_cpu() {
    let caps = KernelCapabilities::from_compile_time();
    let backends = caps.compiled_backends();
    assert!(
        backends.contains(&KernelBackend::CpuRust),
        "Compiled backends should always include CpuRust"
    );
}

#[test]
fn capabilities_best_available_is_not_none() {
    let caps = KernelCapabilities::from_compile_time();
    let best = caps.best_available();
    // Should always have at least CpuRust
    assert!(best.is_some(), "Best available should always return Some");
}

#[test]
fn capabilities_best_available_is_cpu_without_gpu() {
    let caps = KernelCapabilities::from_compile_time();
    if !caps.cuda_runtime && !caps.hip_runtime && !caps.oneapi_runtime {
        assert_eq!(
            caps.best_available(),
            Some(KernelBackend::CpuRust),
            "Without GPU, best should be CpuRust"
        );
    }
}

#[test]
fn capabilities_summary_non_empty() {
    let caps = KernelCapabilities::from_compile_time();
    let summary = caps.summary();
    assert!(!summary.is_empty(), "Summary should be non-empty");
}

#[test]
fn capabilities_summary_contains_cpu() {
    let caps = KernelCapabilities::from_compile_time();
    let summary = caps.summary();
    assert!(summary.to_lowercase().contains("cpu"), "Summary should mention CPU: '{summary}'");
}

#[test]
fn capabilities_with_cuda_runtime() {
    let caps = KernelCapabilities::from_compile_time().with_cuda_runtime(true);
    assert!(caps.cuda_runtime);

    let caps2 = caps.with_cuda_runtime(false);
    assert!(!caps2.cuda_runtime);
}

#[test]
fn capabilities_with_hip_runtime() {
    let caps = KernelCapabilities::from_compile_time().with_hip_runtime(true);
    assert!(caps.hip_runtime);
}

#[test]
fn capabilities_with_oneapi_runtime() {
    let caps = KernelCapabilities::from_compile_time().with_oneapi_runtime(true);
    assert!(caps.oneapi_runtime);
}

#[test]
fn capabilities_with_cpp_ffi() {
    let caps = KernelCapabilities::from_compile_time().with_cpp_ffi(true);
    assert!(caps.cpp_ffi);
    let backends = caps.compiled_backends();
    assert!(backends.contains(&KernelBackend::CppFfi));
}

#[test]
fn capabilities_chain_builders() {
    let caps = KernelCapabilities::from_compile_time()
        .with_cuda_runtime(true)
        .with_hip_runtime(false)
        .with_oneapi_runtime(true)
        .with_cpp_ffi(true);
    assert!(caps.cuda_runtime);
    assert!(!caps.hip_runtime);
    assert!(caps.oneapi_runtime);
    assert!(caps.cpp_ffi);
}

#[test]
fn capabilities_clone_preserves_all_fields() {
    let caps = KernelCapabilities::from_compile_time().with_cuda_runtime(true).with_cpp_ffi(true);
    let cloned = caps.clone();
    assert_eq!(caps.cpu_rust, cloned.cpu_rust);
    assert_eq!(caps.cuda_compiled, cloned.cuda_compiled);
    assert_eq!(caps.cuda_runtime, cloned.cuda_runtime);
    assert_eq!(caps.hip_compiled, cloned.hip_compiled);
    assert_eq!(caps.hip_runtime, cloned.hip_runtime);
    assert_eq!(caps.oneapi_compiled, cloned.oneapi_compiled);
    assert_eq!(caps.oneapi_runtime, cloned.oneapi_runtime);
    assert_eq!(caps.cpp_ffi, cloned.cpp_ffi);
    assert_eq!(caps.simd_level, cloned.simd_level);
}

#[test]
fn capabilities_debug_non_empty() {
    let caps = KernelCapabilities::from_compile_time();
    assert!(!format!("{caps:?}").is_empty());
}

// --- is_compiled tests ---

#[test]
fn kernel_backend_is_compiled_cpu_always_true() {
    assert!(KernelBackend::CpuRust.is_compiled());
}

#[test]
fn kernel_backend_is_compiled_reflects_feature_gates() {
    // These depend on compile-time features, so just verify they don't panic
    let _ = KernelBackend::Cuda.is_compiled();
    let _ = KernelBackend::Hip.is_compiled();
    let _ = KernelBackend::OneApi.is_compiled();
    let _ = KernelBackend::CppFfi.is_compiled();
}
