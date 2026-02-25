//! Property-based tests for `bitnet-honest-compute`.
//!
//! These tests verify the honest-compute policy invariants across arbitrary
//! kernel ID strings: mock detection, classification, and validation rules.

use bitnet_honest_compute::{
    MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH, classify_compute_path, is_mock_kernel_id,
    validate_compute_path, validate_kernel_ids,
};
use proptest::prelude::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// A proptest strategy producing strings that do NOT contain "mock" (any case).
fn arb_real_kernel_id() -> impl Strategy<Value = String> {
    // Generate short printable ASCII strings; filter out any that contain "mock".
    "[a-zA-Z0-9_]{1,64}".prop_filter("must not contain 'mock'", |s| {
        !s.to_ascii_lowercase().contains("mock")
    })
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    /// Any string containing "mock" (case-insensitive) is detected as mock.
    #[test]
    fn strings_containing_mock_are_detected(
        prefix in "[a-z]{0,16}",
        suffix in "[a-z]{0,16}",
        variant in prop_oneof![
            Just("mock"), Just("MOCK"), Just("Mock"), Just("mOcK"),
        ],
    ) {
        let id = format!("{prefix}{variant}{suffix}");
        prop_assert!(is_mock_kernel_id(&id), "expected mock detection for: {id}");
    }

    /// Real kernel IDs (no "mock") are never flagged as mock.
    #[test]
    fn real_kernel_ids_not_detected_as_mock(id in arb_real_kernel_id()) {
        prop_assert!(!is_mock_kernel_id(&id));
    }

    /// A list of only real kernel IDs classifies as "real".
    #[test]
    fn all_real_kernels_classify_as_real(
        ids in prop::collection::vec(arb_real_kernel_id(), 1..10),
    ) {
        let result = classify_compute_path(ids.iter().map(String::as_str));
        prop_assert_eq!(result, "real");
    }

    /// Any list containing at least one mock kernel ID classifies as "mock".
    #[test]
    fn any_mock_kernel_classifies_as_mock(
        pre in prop::collection::vec(arb_real_kernel_id(), 0..5),
        post in prop::collection::vec(arb_real_kernel_id(), 0..5),
    ) {
        let mut ids: Vec<String> = pre;
        ids.push("mock_cpu_matmul".to_string());
        ids.extend(post);
        let result = classify_compute_path(ids.iter().map(String::as_str));
        prop_assert_eq!(result, "mock");
    }

    /// `validate_compute_path` accepts exactly "real".
    #[test]
    fn validate_compute_path_accepts_real(
        // Generate arbitrary strings that are NOT "real"
        non_real in "[a-z]{1,20}".prop_filter("must not be 'real'", |s| s != "real"),
    ) {
        // Non-"real" values should fail.
        prop_assert!(validate_compute_path(&non_real).is_err());
    }

    /// `validate_kernel_ids` rejects IDs that exceed MAX_KERNEL_ID_LENGTH.
    #[test]
    fn oversized_kernel_id_fails_validation(
        extra in 1usize..=64,
    ) {
        let long_id: String = "x".repeat(MAX_KERNEL_ID_LENGTH + extra);
        let result = validate_kernel_ids([long_id.as_str()]);
        prop_assert!(result.is_err(), "expected error for id of len {}", long_id.len());
    }

    /// Valid real kernel IDs (1..=128 chars, no mock) always pass validation.
    #[test]
    fn valid_real_kernel_ids_pass_validation(
        ids in prop::collection::vec(arb_real_kernel_id(), 1..10),
    ) {
        prop_assert!(validate_kernel_ids(ids.iter().map(String::as_str)).is_ok());
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn validate_compute_path_accepts_literal_real() {
    assert!(validate_compute_path("real").is_ok());
}

#[test]
fn validate_compute_path_rejects_mock() {
    assert!(validate_compute_path("mock").is_err());
}

#[test]
fn empty_kernel_array_fails_validation() {
    let result = validate_kernel_ids([] as [&str; 0]);
    assert!(result.is_err(), "empty kernel array should fail");
}

#[test]
fn kernel_count_at_limit_passes() {
    let ids: Vec<String> = (0..MAX_KERNEL_COUNT).map(|i| format!("kernel_{i}")).collect();
    assert!(validate_kernel_ids(ids.iter().map(String::as_str)).is_ok());
}

#[test]
fn kernel_count_over_limit_fails() {
    let ids: Vec<String> = (0..=MAX_KERNEL_COUNT).map(|i| format!("kernel_{i}")).collect();
    assert!(validate_kernel_ids(ids.iter().map(String::as_str)).is_err());
}
