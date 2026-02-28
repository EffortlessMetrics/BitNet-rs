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
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        vulkan_compiled: false,
        vulkan_runtime: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_full() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        vulkan_compiled: false,
        vulkan_runtime: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn cuda_compiled_no_runtime() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        vulkan_compiled: false,
        vulkan_runtime: false,
        simd_level: SimdLevel::Avx2,
    }
}

fn empty_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        vulkan_compiled: false,
        vulkan_runtime: false,
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
        BackendRequest::OneApi,
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
        BackendRequest::OneApi,
    ])) {
        // Display must produce one of the known strings
        let s = variant.to_string();
        let known = ["auto", "cpu", "gpu", "cuda", "oneapi"];
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
fn oneapi_request_strict_fails_cpu_only() {
    let err = select_backend(BackendRequest::OneApi, &cpu_only()).unwrap_err();
    assert!(matches!(err, BackendSelectionError::RequestedUnavailable { .. }));
}

#[test]
fn gpu_request_prefers_oneapi_when_cuda_unavailable() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: false,
        vulkan_compiled: false,
        vulkan_runtime: false,
        simd_level: SimdLevel::Avx2,
    };
    let r = select_backend(BackendRequest::Gpu, &caps).unwrap();
    assert_eq!(r.selected, KernelBackend::OneApi);
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

// ---------------------------------------------------------------------------
// BackendStartupSummary tests
// ---------------------------------------------------------------------------

use bitnet_common::backend_selection::BackendStartupSummary;

#[test]
fn test_backend_startup_summary_log_line() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".to_string()], "cpu-rust");
    let line = summary.log_line();
    assert!(line.contains("requested="), "missing requested=: {line}");
    assert!(line.contains("detected=["), "missing detected=[: {line}");
    assert!(line.contains("selected="), "missing selected=: {line}");
    assert_eq!(line, "requested=auto detected=[cpu-rust] selected=cpu-rust");
}

#[test]
fn test_backend_startup_summary_log_line_multiple_detected() {
    let summary =
        BackendStartupSummary::new("gpu", vec!["cuda".to_string(), "cpu-rust".to_string()], "cuda");
    let line = summary.log_line();
    assert_eq!(line, "requested=gpu detected=[cuda, cpu-rust] selected=cuda");
}

#[test]
fn test_backend_startup_summary_log_line_empty_detected() {
    let summary = BackendStartupSummary::new("cpu", vec![], "cpu");
    let line = summary.log_line();
    assert_eq!(line, "requested=cpu detected=[] selected=cpu");
}

#[test]
fn test_backend_startup_summary_serde() {
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".to_string()], "cpu-rust");
    let json = serde_json::to_string(&summary).expect("serialization failed");
    let roundtrip: BackendStartupSummary =
        serde_json::from_str(&json).expect("deserialization failed");
    assert_eq!(summary, roundtrip);
}

#[test]
fn test_backend_startup_summary_serde_preserves_fields() {
    let summary =
        BackendStartupSummary::new("gpu", vec!["cuda".to_string(), "cpu-rust".to_string()], "cuda");
    let json = serde_json::to_string(&summary).unwrap();
    assert!(json.contains("\"requested\":\"gpu\""), "json: {json}");
    assert!(json.contains("\"selected\":\"cuda\""), "json: {json}");
    assert!(json.contains("\"detected\""), "json: {json}");
}

proptest! {
    #[test]
    fn log_line_always_contains_all_three_tags(
        requested in "[a-z]{1,8}",
        selected in "[a-z]{1,8}",
    ) {
        let summary = BackendStartupSummary::new(&requested, vec![selected.clone()], &selected);
        let line = summary.log_line();
        prop_assert!(line.contains("requested="), "missing requested= in: {line}");
        prop_assert!(line.contains("detected=["), "missing detected=[ in: {line}");
        prop_assert!(line.contains("selected="), "missing selected= in: {line}");
    }
}
