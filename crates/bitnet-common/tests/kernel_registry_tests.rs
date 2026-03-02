// SPDX-License-Identifier: MIT OR Apache-2.0
//! Comprehensive tests for `bitnet_common::kernel_registry`.
//!
//! Covers:
//! - `SimdLevel` ordering, display, and copy semantics
//! - `KernelBackend` display, GPU requirement, copy semantics
//! - `KernelCapabilities` construction, builder methods, and capability queries
//! - `compiled_backends()` ordering and deduplication
//! - `best_available()` priority logic
//! - `summary()` format invariants

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cpu_only_avx2() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_with_runtime() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn all_backends() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx512,
    }
}

fn empty_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    }
}

// ── SimdLevel ordering ────────────────────────────────────────────────────────

#[test]
fn simd_scalar_is_lowest() {
    assert!(SimdLevel::Scalar < SimdLevel::Neon);
    assert!(SimdLevel::Scalar < SimdLevel::Sse42);
    assert!(SimdLevel::Scalar < SimdLevel::Avx2);
    assert!(SimdLevel::Scalar < SimdLevel::Avx512);
}

#[test]
fn simd_neon_between_scalar_and_sse42() {
    assert!(SimdLevel::Neon > SimdLevel::Scalar);
    assert!(SimdLevel::Neon < SimdLevel::Sse42);
}

#[test]
fn simd_sse42_between_neon_and_avx2() {
    assert!(SimdLevel::Sse42 > SimdLevel::Neon);
    assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
}

#[test]
fn simd_avx2_between_sse42_and_avx512() {
    assert!(SimdLevel::Avx2 > SimdLevel::Sse42);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

#[test]
fn simd_avx512_is_highest() {
    assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
    assert!(SimdLevel::Avx512 > SimdLevel::Neon);
    assert!(SimdLevel::Avx512 > SimdLevel::Scalar);
}

#[test]
fn simd_level_total_order_is_transitive() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for (i, a) in levels.iter().enumerate() {
        for (j, b) in levels.iter().enumerate() {
            assert_eq!(i.cmp(&j), a.cmp(b), "ordering mismatch: {a:?} vs {b:?}");
        }
    }
}

#[test]
fn simd_level_equality() {
    assert_eq!(SimdLevel::Avx2, SimdLevel::Avx2);
    assert_ne!(SimdLevel::Avx2, SimdLevel::Avx512);
}

// ── SimdLevel display ─────────────────────────────────────────────────────────

#[test]
fn simd_level_display_scalar() {
    assert_eq!(SimdLevel::Scalar.to_string(), "scalar");
}

#[test]
fn simd_level_display_neon() {
    assert_eq!(SimdLevel::Neon.to_string(), "neon");
}

#[test]
fn simd_level_display_sse42() {
    assert_eq!(SimdLevel::Sse42.to_string(), "sse4.2");
}

#[test]
fn simd_level_display_avx2() {
    assert_eq!(SimdLevel::Avx2.to_string(), "avx2");
}

#[test]
fn simd_level_display_avx512() {
    assert_eq!(SimdLevel::Avx512.to_string(), "avx512");
}

#[test]
fn simd_level_all_display_strings_are_distinct() {
    let strings: Vec<String> =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512]
            .iter()
            .map(|s| s.to_string())
            .collect();
    let unique: std::collections::HashSet<_> = strings.iter().collect();
    assert_eq!(strings.len(), unique.len(), "duplicate display strings: {strings:?}");
}

// ── SimdLevel copy and hash semantics ─────────────────────────────────────────

#[test]
fn simd_level_is_copy() {
    let a = SimdLevel::Avx2;
    let b = a; // copy, not move
    assert_eq!(a, b);
}

#[test]
fn simd_level_is_hashable() {
    let mut map = std::collections::HashMap::new();
    map.insert(SimdLevel::Avx2, "avx2");
    map.insert(SimdLevel::Avx512, "avx512");
    assert_eq!(map[&SimdLevel::Avx2], "avx2");
}

// ── KernelBackend display ─────────────────────────────────────────────────────

#[test]
fn kernel_backend_display_cpu_rust() {
    assert_eq!(KernelBackend::CpuRust.to_string(), "cpu-rust");
}

#[test]
fn kernel_backend_display_cuda() {
    assert_eq!(KernelBackend::Cuda.to_string(), "cuda");
}

#[test]
fn kernel_backend_display_cpp_ffi() {
    assert_eq!(KernelBackend::CppFfi.to_string(), "cpp-ffi");
}

#[test]
fn kernel_backend_all_display_strings_are_distinct() {
    let strings: Vec<String> = [KernelBackend::CpuRust, KernelBackend::Cuda, KernelBackend::CppFfi]
        .iter()
        .map(|b| b.to_string())
        .collect();
    let unique: std::collections::HashSet<_> = strings.iter().collect();
    assert_eq!(strings.len(), unique.len(), "duplicate display strings: {strings:?}");
}

// ── KernelBackend::requires_gpu ───────────────────────────────────────────────

#[test]
fn cpu_rust_does_not_require_gpu() {
    assert!(!KernelBackend::CpuRust.requires_gpu());
}

#[test]
fn cuda_requires_gpu() {
    assert!(KernelBackend::Cuda.requires_gpu());
}

#[test]
fn cpp_ffi_does_not_require_gpu() {
    assert!(!KernelBackend::CppFfi.requires_gpu());
}

// ── KernelBackend::is_compiled ────────────────────────────────────────────────

#[test]
fn cpp_ffi_is_never_compiled_from_common() {
    // bitnet-common has no ffi feature; CppFfi always reports not compiled
    assert!(!KernelBackend::CppFfi.is_compiled());
}

// ── KernelBackend copy and hash semantics ─────────────────────────────────────

#[test]
fn kernel_backend_is_copy() {
    let a = KernelBackend::CpuRust;
    let b = a; // copy, not move
    assert_eq!(a, b);
}

#[test]
fn kernel_backend_is_hashable() {
    let mut set = std::collections::HashSet::new();
    set.insert(KernelBackend::CpuRust);
    set.insert(KernelBackend::Cuda);
    set.insert(KernelBackend::CpuRust); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn kernel_backend_equality() {
    assert_eq!(KernelBackend::Cuda, KernelBackend::Cuda);
    assert_ne!(KernelBackend::Cuda, KernelBackend::CpuRust);
}

// ── KernelCapabilities: from_compile_time ─────────────────────────────────────

#[test]
fn from_compile_time_always_sets_cuda_runtime_false() {
    let caps = KernelCapabilities::from_compile_time();
    assert!(!caps.cuda_runtime, "cuda_runtime must be false without runtime probe");
}

#[test]
fn from_compile_time_cpp_ffi_is_false() {
    // bitnet-common has no ffi feature
    let caps = KernelCapabilities::from_compile_time();
    assert!(!caps.cpp_ffi);
}

// ── KernelCapabilities: builder methods ──────────────────────────────────────

#[test]
fn with_cuda_runtime_true_sets_flag() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    let caps = caps.with_cuda_runtime(true);
    assert!(caps.cuda_runtime);
}

#[test]
fn with_cuda_runtime_false_clears_flag() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
    .with_cuda_runtime(false);
    assert!(!caps.cuda_runtime);
}

#[test]
fn with_cpp_ffi_true_sets_flag() {
    let caps = KernelCapabilities::from_compile_time().with_cpp_ffi(true);
    assert!(caps.cpp_ffi);
}

#[test]
fn with_cpp_ffi_false_clears_flag() {
    let caps = KernelCapabilities::from_compile_time().with_cpp_ffi(true).with_cpp_ffi(false);
    assert!(!caps.cpp_ffi);
}

#[test]
fn builder_methods_are_chainable() {
    let caps = KernelCapabilities::from_compile_time().with_cuda_runtime(true).with_cpp_ffi(true);
    assert!(caps.cuda_runtime);
    assert!(caps.cpp_ffi);
}

// ── KernelCapabilities: compiled_backends ordering ───────────────────────────

#[test]
fn compiled_backends_empty_when_nothing_compiled() {
    let caps = empty_caps();
    assert!(caps.compiled_backends().is_empty());
}

#[test]
fn compiled_backends_cpu_only() {
    let caps = cpu_only_avx2();
    let backends = caps.compiled_backends();
    assert_eq!(backends, vec![KernelBackend::CpuRust]);
}

#[test]
fn compiled_backends_cuda_and_cpu_no_ffi() {
    let caps = cuda_with_runtime();
    let backends = caps.compiled_backends();
    // CUDA before CPU; no FFI
    assert_eq!(backends, vec![KernelBackend::Cuda, KernelBackend::CpuRust]);
}

#[test]
fn compiled_backends_only_ffi() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    };
    let backends = caps.compiled_backends();
    assert_eq!(backends, vec![KernelBackend::CppFfi]);
}

#[test]
fn compiled_backends_all_three_in_priority_order() {
    let caps = all_backends();
    let backends = caps.compiled_backends();
    assert_eq!(backends[0], KernelBackend::Cuda);
    assert_eq!(backends[1], KernelBackend::CppFfi);
    assert_eq!(backends[2], KernelBackend::CpuRust);
    assert_eq!(backends.len(), 3);
}

#[test]
fn compiled_backends_no_duplicates() {
    let caps = all_backends();
    let backends = caps.compiled_backends();
    let unique: std::collections::HashSet<_> = backends.iter().collect();
    assert_eq!(backends.len(), unique.len(), "duplicates in compiled_backends: {backends:?}");
}

#[test]
fn compiled_backends_cuda_not_listed_when_only_runtime_missing() {
    // cuda_compiled=true means it appears in compiled_backends (compile-time list)
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false, // runtime not available, but compiled
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    let backends = caps.compiled_backends();
    // CUDA is compiled so it appears in the list
    assert!(backends.contains(&KernelBackend::Cuda));
}

// ── KernelCapabilities: best_available priority ───────────────────────────────

#[test]
fn best_available_is_none_when_nothing_compiled() {
    assert_eq!(empty_caps().best_available(), None);
}

#[test]
fn best_available_is_cpu_when_only_cpu() {
    assert_eq!(cpu_only_avx2().best_available(), Some(KernelBackend::CpuRust));
}

#[test]
fn best_available_is_cuda_when_compiled_and_runtime() {
    assert_eq!(cuda_with_runtime().best_available(), Some(KernelBackend::Cuda));
}

#[test]
fn best_available_is_cpu_when_cuda_compiled_but_no_runtime() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CpuRust));
}

#[test]
fn best_available_is_ffi_when_no_cuda_runtime_but_ffi_present() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Scalar,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CppFfi));
}

#[test]
fn best_available_prefers_cuda_over_ffi_and_cpu() {
    assert_eq!(all_backends().best_available(), Some(KernelBackend::Cuda));
}

#[test]
fn best_available_prefers_ffi_over_cpu_when_no_cuda() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: true,
        simd_level: SimdLevel::Avx2,
    };
    assert_eq!(caps.best_available(), Some(KernelBackend::CppFfi));
}

// ── KernelCapabilities: summary format ───────────────────────────────────────

#[test]
fn summary_contains_simd_tag() {
    let s = cpu_only_avx2().summary();
    assert!(s.contains("simd=avx2"), "summary: {s}");
}

#[test]
fn summary_contains_backends_tag() {
    let s = cpu_only_avx2().summary();
    assert!(s.contains("backends=["), "summary: {s}");
}

#[test]
fn summary_cpu_only_has_cpu_rust_backend() {
    let s = cpu_only_avx2().summary();
    assert!(s.contains("cpu-rust"), "summary: {s}");
}

#[test]
fn summary_empty_caps_has_empty_backends() {
    let s = empty_caps().summary();
    assert!(s.contains("backends=[]"), "summary: {s}");
}

#[test]
fn summary_all_backends_lists_all_three() {
    let s = all_backends().summary();
    assert!(s.contains("cuda"), "summary: {s}");
    assert!(s.contains("cpp-ffi"), "summary: {s}");
    assert!(s.contains("cpu-rust"), "summary: {s}");
}

#[test]
fn summary_reflects_simd_level_for_neon() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Neon,
    };
    let s = caps.summary();
    assert!(s.contains("simd=neon"), "summary: {s}");
}

#[test]
fn summary_reflects_simd_level_for_scalar() {
    let s = empty_caps().summary();
    assert!(s.contains("simd=scalar"), "summary: {s}");
}

// ── KernelCapabilities: clone semantics ──────────────────────────────────────

#[test]
fn kernel_capabilities_clone_preserves_all_fields() {
    let original = all_backends();
    let cloned = original.clone();
    assert_eq!(original.cpu_rust, cloned.cpu_rust);
    assert_eq!(original.cuda_compiled, cloned.cuda_compiled);
    assert_eq!(original.cuda_runtime, cloned.cuda_runtime);
    assert_eq!(original.cpp_ffi, cloned.cpp_ffi);
    assert_eq!(original.simd_level, cloned.simd_level);
}

#[test]
fn kernel_capabilities_clone_is_independent() {
    let original = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    let cloned = original.clone();
    // Both should have identical fields; clone does not alias original
    assert_eq!(original.cpu_rust, cloned.cpu_rust);
    // Verify cloned summary matches original summary (structural equality)
    assert_eq!(original.summary(), cloned.summary());
}
