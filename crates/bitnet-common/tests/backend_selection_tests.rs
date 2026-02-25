//! Property and snapshot tests for `backend_selection` and `kernel_registry`.
//!
//! These tests run unconditionally (no feature gate) to ensure backend
//! selection contracts are always verified in CI.

use bitnet_common::backend_selection::{BackendRequest, BackendSelectionError, select_backend};
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu_only() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_full() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_compiled_no_runtime() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn empty_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Scalar,
    }
}

// ---------------------------------------------------------------------------
// BackendRequest Display round-trip
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn backend_request_display_is_non_empty(variant in prop::sample::select(vec![
        BackendRequest::Auto,
        BackendRequest::Cpu,
        BackendRequest::Gpu,
        BackendRequest::Cuda,
    ])) {
        let s = variant.to_string();
        prop_assert!(!s.is_empty());
        prop_assert!(s.chars().all(|c| c.is_ascii_lowercase() || c == '-'));
    }
}

proptest! {
    #[test]
    fn backend_request_all_variants_display(variant in prop::sample::select(vec![
        BackendRequest::Auto,
        BackendRequest::Cpu,
        BackendRequest::Gpu,
        BackendRequest::Cuda,
    ])) {
        // Display must produce one of the known strings
        let s = variant.to_string();
        let known = ["auto", "cpu", "gpu", "cuda"];
        prop_assert!(known.contains(&s.as_str()), "unexpected display: {s}");
    }
}

// ---------------------------------------------------------------------------
// BackendSelectionResult.summary() invariants
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn summary_always_contains_requested_tag(variant in prop::sample::select(vec![
        BackendRequest::Auto,
        BackendRequest::Cpu,
    ])) {
        let result = select_backend(variant, &cpu_only()).unwrap();
        let summary = result.summary();
        prop_assert!(summary.contains("requested="), "missing requested= in: {summary}");
        prop_assert!(summary.contains("detected=["), "missing detected=[ in: {summary}");
        prop_assert!(summary.contains("selected="), "missing selected= in: {summary}");
    }
}

proptest! {
    #[test]
    fn summary_selected_matches_result_backend(variant in prop::sample::select(vec![
        BackendRequest::Auto,
        BackendRequest::Cpu,
    ])) {
        let result = select_backend(variant, &cpu_only()).unwrap();
        let summary = result.summary();
        let expected = format!("selected={}", result.selected);
        prop_assert!(summary.contains(&expected), "summary missing {expected}: {summary}");
    }
}

// ---------------------------------------------------------------------------
// select_backend determinism
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn select_backend_is_deterministic(variant in prop::sample::select(vec![
        BackendRequest::Auto,
        BackendRequest::Cpu,
    ])) {
        let r1 = select_backend(variant, &cpu_only()).unwrap();
        let r2 = select_backend(variant, &cpu_only()).unwrap();
        prop_assert_eq!(r1.selected, r2.selected);
    }
}

// ---------------------------------------------------------------------------
// Unit tests: specific selection behavior
// ---------------------------------------------------------------------------

#[test]
fn auto_cpu_only_selects_cpu_rust() {
    let r = select_backend(BackendRequest::Auto, &cpu_only()).unwrap();
    assert_eq!(r.selected, KernelBackend::CpuRust);
}

#[test]
fn auto_cuda_full_selects_cuda() {
    let r = select_backend(BackendRequest::Auto, &cuda_full()).unwrap();
    assert_eq!(r.selected, KernelBackend::Cuda);
}

#[test]
fn cuda_request_strict_fails_without_runtime() {
    let err = select_backend(BackendRequest::Cuda, &cuda_compiled_no_runtime()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

#[test]
fn cuda_request_strict_fails_cpu_only() {
    let err = select_backend(BackendRequest::Cuda, &cpu_only()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not available"), "unexpected error: {msg}");
}

#[test]
fn auto_empty_caps_returns_no_backend_error() {
    let err = select_backend(BackendRequest::Auto, &empty_caps()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::NoBackendAvailable));
}

#[test]
fn gpu_request_with_cuda_compiled_no_runtime_falls_back_to_cpu() {
    let r = select_backend(BackendRequest::Gpu, &cuda_compiled_no_runtime()).unwrap();
    assert_eq!(r.selected, KernelBackend::CpuRust);
    assert!(r.rationale.contains("falling back to CPU"), "rationale: {}", r.rationale);
}

#[test]
fn summary_snapshot_cpu_auto() {
    let r = select_backend(BackendRequest::Auto, &cpu_only()).unwrap();
    insta::assert_snapshot!("backend_summary_cpu_auto", r.summary());
}

#[test]
fn selection_result_debug_snapshot_cpu_auto() {
    let r = select_backend(BackendRequest::Auto, &cpu_only()).unwrap();
    // Snapshot the debug to pin the struct layout
    insta::assert_debug_snapshot!("backend_selection_result_cpu_auto", r);
}
