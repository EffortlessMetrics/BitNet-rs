//! End-to-end backend contract test (always-on, no model required).
//!
//! Exercises the full pipeline:
//!   KernelCapabilities → BackendRequest → BackendSelectionResult
//!       → InferenceReceipt (with backend_summary) → validation gates
//!
//! This test proves that the backend selection and receipt plumbing is wired
//! end-to-end without requiring a real model or inference run.

use bitnet_common::backend_selection::{BackendRequest, select_backend};
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};
use bitnet_receipts::InferenceReceipt;

fn cpu_only_caps() -> KernelCapabilities {
    KernelCapabilities {
        cpu_rust: true,
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

fn cpu_avx2_caps() -> KernelCapabilities {
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

/// Build a minimal valid InferenceReceipt for testing.
fn make_receipt(backend: &str, backend_summary: Option<String>) -> InferenceReceipt {
    InferenceReceipt::generate(
        backend,
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        backend_summary,
    )
    .expect("receipt generation should succeed")
}

#[test]
fn auto_request_cpu_only_selects_cpu_rust() {
    let caps = cpu_only_caps();
    let result =
        select_backend(BackendRequest::Auto, &caps).expect("select_backend should succeed");
    assert_eq!(result.selected, KernelBackend::CpuRust);
    assert!(result.summary().contains("selected=cpu-rust"), "got: {}", result.summary());
}

#[test]
fn backend_summary_propagates_to_receipt() {
    let caps = cpu_only_caps();
    let result = select_backend(BackendRequest::Auto, &caps).unwrap();
    let summary = result.summary();

    let receipt = make_receipt("cpu", Some(summary.clone()));
    assert_eq!(receipt.backend_summary, summary);
    assert!(receipt.backend_summary.contains("selected="), "summary must contain 'selected='");
}

#[test]
fn receipt_with_backend_summary_passes_all_gates() {
    let caps = cpu_avx2_caps();
    let result = select_backend(BackendRequest::Cpu, &caps).unwrap();
    let receipt = make_receipt("cpu", Some(result.summary()));

    // validate() runs all gates: schema, compute_path, kernel_ids, and soft backend_summary check
    receipt.validate().expect("receipt with valid backend_summary should pass all gates");
}

#[test]
fn receipt_without_backend_summary_still_validates() {
    // backward compatibility: existing receipts without backend_summary must still be valid
    let receipt = make_receipt("cpu", None);
    receipt.validate().expect("receipt without backend_summary should still pass");
    assert_eq!(receipt.backend_summary, "", "empty string default when None passed");
}

#[test]
fn receipt_backend_summary_format_check_rejects_garbage() {
    // When backend_summary is non-empty, it must contain "selected=".
    // InferenceReceipt::generate() sets the field; validation checks its format.
    // We test the validation by introducing a bad summary via JSON round-trip.

    let receipt = make_receipt(
        "cpu",
        Some("requested=auto detected=[cpu-rust] selected=cpu-rust".to_string()),
    );

    // Serialize to JSON, corrupt backend_summary in the JSON payload, then deserialize.
    let json = serde_json::to_string(&receipt).expect("receipt must serialize");
    let mut value: serde_json::Value =
        serde_json::from_str(&json).expect("receipt JSON must parse to Value");
    value["backend_summary"] = serde_json::Value::String("bogus_no_selected_key".to_string());
    let corrupted_json = serde_json::to_string(&value).expect("corrupted JSON must serialize");
    let corrupted_receipt: InferenceReceipt =
        serde_json::from_str(&corrupted_json).expect("corrupted JSON must deserialize");

    let err = corrupted_receipt.validate();
    assert!(err.is_err(), "receipt with malformed backend_summary should fail validation");
}

#[test]
fn backend_selection_summary_round_trips_through_receipt_json() {
    let caps = cpu_avx2_caps();
    let result = select_backend(BackendRequest::Auto, &caps).unwrap();
    let original_summary = result.summary();

    let receipt = make_receipt("cpu", Some(original_summary.clone()));

    // Serialize and deserialize
    let json = serde_json::to_string(&receipt).expect("receipt must serialize");
    let reloaded: InferenceReceipt = serde_json::from_str(&json).expect("receipt must deserialize");

    assert_eq!(reloaded.backend_summary, original_summary);
    reloaded.validate().expect("round-tripped receipt must still validate");
}

#[test]
fn simd_level_is_reported_in_kernel_capabilities_summary() {
    let caps = cpu_avx2_caps();
    let summary = caps.summary();
    assert!(summary.contains("avx2"), "AVX2 caps should appear in summary: {summary}");
}

#[test]
fn backend_request_auto_records_full_selection_chain() {
    let caps = cpu_only_caps();
    let result = select_backend(BackendRequest::Auto, &caps).unwrap();
    let summary = result.summary();

    // The summary must have all three components
    assert!(summary.contains("requested=auto"), "summary: {summary}");
    assert!(summary.contains("detected=["), "summary: {summary}");
    assert!(summary.contains("selected="), "summary: {summary}");
}
