//! Comprehensive tests for `bitnet-honest-compute` public API.
//!
//! Covers:
//!   - Module constants (`REAL_COMPUTE_PATH`, `MOCK_COMPUTE_PATH`, etc.)
//!   - `ComputePathError` and `KernelValidationError` types (Display, Debug, `Error` impl)
//!   - `validate_compute_path()`: accepts "real", rejects all other strings
//!   - `validate_kernel_ids()`: hygiene rules (empty array, empty IDs, whitespace,
//!     too-long IDs, mock keyword, count limit, boundary values)
//!   - `classify_compute_path()`: returns "real" / "mock" based on kernel list
//!   - `is_mock_kernel_id()`: case-insensitive "mock" detection
//!   - Property tests: arbitrary valid IDs always pass; IDs containing "mock" always fail

use bitnet_honest_compute::{
    ComputePathError, KernelValidationError, MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH,
    MOCK_COMPUTE_PATH, REAL_COMPUTE_PATH, classify_compute_path, is_mock_kernel_id,
    validate_compute_path, validate_kernel_ids,
};
use proptest::prelude::*;
use std::error::Error;

// ── Policy constants ─────────────────────────────────────────────────────────

#[test]
fn real_compute_path_constant_is_real_string() {
    assert_eq!(REAL_COMPUTE_PATH, "real");
}

#[test]
fn mock_compute_path_constant_is_mock_string() {
    assert_eq!(MOCK_COMPUTE_PATH, "mock");
}

#[test]
fn max_kernel_id_length_constant_is_128() {
    assert_eq!(MAX_KERNEL_ID_LENGTH, 128);
}

#[test]
fn max_kernel_count_constant_is_10000() {
    assert_eq!(MAX_KERNEL_COUNT, 10_000);
}

// ── Error type trait implementations ─────────────────────────────────────────

#[test]
fn compute_path_error_implements_std_error_trait() {
    let err = ComputePathError::NotReal { actual: "mock".to_string() };
    // Upcast to &dyn Error — this only compiles when the trait is implemented
    let _: &dyn Error = &err;
    assert!(err.source().is_none(), "ComputePathError has no cause");
}

#[test]
fn kernel_validation_error_implements_std_error_trait() {
    let err = KernelValidationError::EmptyKernelArray;
    let _: &dyn Error = &err;
    assert!(err.source().is_none());
}

#[test]
fn compute_path_error_not_real_display_contains_actual_value() {
    let err = ComputePathError::NotReal { actual: "queue".to_string() };
    let msg = err.to_string();
    assert!(msg.contains("queue"), "Display must include the actual value: {msg}");
    assert!(msg.contains("real"), "Display must mention 'real': {msg}");
}

#[test]
fn kernel_validation_error_variants_all_have_non_empty_display() {
    let cases: Vec<Box<dyn std::fmt::Display>> = vec![
        Box::new(KernelValidationError::EmptyKernelArray),
        Box::new(KernelValidationError::KernelCountExceedsLimit { count: 99_999 }),
        Box::new(KernelValidationError::EmptyKernelId { index: 0 }),
        Box::new(KernelValidationError::WhitespaceOnlyKernelId { index: 2 }),
        Box::new(KernelValidationError::KernelIdTooLong { index: 1, kernel_id: "x".repeat(200) }),
        Box::new(KernelValidationError::MockKernelDetected {
            index: 3,
            kernel_id: "mock_matmul".to_string(),
        }),
    ];

    for case in &cases {
        let msg = case.to_string();
        assert!(!msg.is_empty(), "every KernelValidationError variant must have a message");
    }
}

// ── validate_compute_path ─────────────────────────────────────────────────────

#[test]
fn validate_compute_path_accepts_real() {
    assert!(validate_compute_path("real").is_ok(), "'real' must be accepted");
}

#[test]
fn validate_compute_path_rejects_empty_string() {
    let err = validate_compute_path("").unwrap_err();
    assert_eq!(err, ComputePathError::NotReal { actual: "".to_string() });
}

#[test]
fn validate_compute_path_rejects_real_with_leading_space() {
    let result = validate_compute_path(" real");
    assert!(result.is_err(), "' real' must be rejected (not exact match)");
}

#[test]
fn validate_compute_path_rejects_uppercase_real() {
    assert!(validate_compute_path("Real").is_err(), "'Real' is not 'real'");
    assert!(validate_compute_path("REAL").is_err(), "'REAL' is not 'real'");
}

#[test]
fn validate_compute_path_rejects_mock() {
    let err = validate_compute_path("mock").unwrap_err();
    match err {
        ComputePathError::NotReal { actual } => assert_eq!(actual, "mock"),
    }
}

#[test]
fn validate_compute_path_rejects_common_alternatives() {
    for path in &["queue", "simulated", "fake", "test", "cpu", "gpu", "cuda", "real "] {
        assert!(
            validate_compute_path(path).is_err(),
            "'{path}' must be rejected by validate_compute_path"
        );
    }
}

// ── validate_kernel_ids ───────────────────────────────────────────────────────

#[test]
fn validate_kernel_ids_single_valid_id_passes() {
    assert!(validate_kernel_ids(["i2s_cpu_matmul"]).is_ok());
}

#[test]
fn validate_kernel_ids_multiple_valid_ids_pass() {
    assert!(validate_kernel_ids(["i2s_cpu_matmul", "tl1_avx2", "tl2_neon", "gemm_f32"]).is_ok());
}

#[test]
fn validate_kernel_ids_rejects_empty_array() {
    let err = validate_kernel_ids::<[&str; 0], &str>([]).unwrap_err();
    assert_eq!(err, KernelValidationError::EmptyKernelArray);
}

#[test]
fn validate_kernel_ids_rejects_empty_string_id() {
    let err = validate_kernel_ids([""]).unwrap_err();
    assert_eq!(err, KernelValidationError::EmptyKernelId { index: 0 });
}

#[test]
fn validate_kernel_ids_rejects_tab_only_id() {
    let err = validate_kernel_ids(["\t"]).unwrap_err();
    assert_eq!(err, KernelValidationError::WhitespaceOnlyKernelId { index: 0 });
}

#[test]
fn validate_kernel_ids_rejects_newline_only_id() {
    let err = validate_kernel_ids(["\n"]).unwrap_err();
    assert_eq!(err, KernelValidationError::WhitespaceOnlyKernelId { index: 0 });
}

#[test]
fn validate_kernel_ids_rejects_spaces_only_id() {
    let err = validate_kernel_ids(["   "]).unwrap_err();
    assert_eq!(err, KernelValidationError::WhitespaceOnlyKernelId { index: 0 });
}

#[test]
fn validate_kernel_ids_exactly_max_length_id_passes() {
    let id = "k".repeat(MAX_KERNEL_ID_LENGTH);
    assert!(
        validate_kernel_ids([id.as_str()]).is_ok(),
        "exactly {MAX_KERNEL_ID_LENGTH} chars must pass"
    );
}

#[test]
fn validate_kernel_ids_one_over_max_length_fails() {
    let id = "k".repeat(MAX_KERNEL_ID_LENGTH + 1);
    let err = validate_kernel_ids([id.as_str()]).unwrap_err();
    assert_eq!(err, KernelValidationError::KernelIdTooLong { index: 0, kernel_id: id });
}

#[test]
fn validate_kernel_ids_error_index_points_to_failing_element() {
    // First two are valid; third is empty
    let err = validate_kernel_ids(["gemm_f32", "i2s_cpu", ""]).unwrap_err();
    assert_eq!(err, KernelValidationError::EmptyKernelId { index: 2 });
}

#[test]
fn validate_kernel_ids_mock_detection_is_case_insensitive() {
    for mock_id in &["mock_kernel", "MOCK_KERNEL", "Mock_Kernel", "prefix_mOcK_suffix"] {
        let err = validate_kernel_ids([*mock_id]).unwrap_err();
        match err {
            KernelValidationError::MockKernelDetected { .. } => {}
            other => panic!("expected MockKernelDetected for '{mock_id}', got {other:?}"),
        }
    }
}

#[test]
fn validate_kernel_ids_exactly_max_count_passes() {
    let kernels: Vec<&str> = vec!["i2s_kernel"; MAX_KERNEL_COUNT];
    assert!(validate_kernel_ids(kernels).is_ok(), "exactly {MAX_KERNEL_COUNT} kernels must pass");
}

#[test]
fn validate_kernel_ids_one_over_max_count_fails() {
    let kernels = std::iter::repeat("i2s_kernel").take(MAX_KERNEL_COUNT + 1);
    let err = validate_kernel_ids(kernels).unwrap_err();
    assert_eq!(err, KernelValidationError::KernelCountExceedsLimit { count: MAX_KERNEL_COUNT + 1 });
}

// ── classify_compute_path ─────────────────────────────────────────────────────

#[test]
fn classify_compute_path_single_real_kernel_returns_real() {
    assert_eq!(classify_compute_path(["i2s_cpu_matmul"]), "real");
}

#[test]
fn classify_compute_path_empty_iterator_returns_real() {
    // Empty → no mock kernels found → "real"
    let empty: [&str; 0] = [];
    assert_eq!(classify_compute_path(empty), "real");
}

#[test]
fn classify_compute_path_single_mock_returns_mock() {
    assert_eq!(classify_compute_path(["mock_matmul"]), "mock");
}

#[test]
fn classify_compute_path_mixed_with_mock_returns_mock() {
    let kernels = ["i2s_cpu", "tl1_avx2", "MOCK_gemm", "tl2_neon"];
    assert_eq!(classify_compute_path(kernels), "mock");
}

// ── is_mock_kernel_id ─────────────────────────────────────────────────────────

#[test]
fn is_mock_kernel_id_exact_string() {
    assert!(is_mock_kernel_id("mock"));
    assert!(is_mock_kernel_id("MOCK"));
}

#[test]
fn is_mock_kernel_id_embedded_in_longer_string() {
    assert!(is_mock_kernel_id("prefix_mock_suffix"));
    assert!(is_mock_kernel_id("i2s_MOCK_kernel"));
}

#[test]
fn is_mock_kernel_id_real_kernel_not_detected() {
    assert!(!is_mock_kernel_id("i2s_cpu_matmul"));
    assert!(!is_mock_kernel_id("tl1_avx2_kernel"));
    assert!(!is_mock_kernel_id("gemm_f32_neon"));
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// Any kernel ID matching `[a-z_][a-z0-9_]{0,30}` and not containing
    /// "mock" must pass `validate_kernel_ids`.
    #[test]
    fn arbitrary_real_kernel_ids_always_pass_validation(
        id in "[a-z_][a-z0-9_]{0,30}",
    ) {
        prop_assume!(!id.to_ascii_lowercase().contains("mock"));
        let result = validate_kernel_ids([id.as_str()]);
        prop_assert!(result.is_ok(), "valid kernel id {:?} must pass: {:?}", id, result);
    }

    /// Any kernel ID that contains "mock" (any case) must fail validation.
    #[test]
    fn kernel_ids_containing_mock_always_fail_validation(
        prefix in "[a-z0-9]{0,8}",
        suffix in "[a-z0-9]{0,8}",
        mock_case in prop_oneof![
            Just("mock"), Just("MOCK"), Just("Mock"), Just("mOcK"),
        ],
    ) {
        let id = format!("{prefix}{mock_case}{suffix}");
        let result = validate_kernel_ids([id.as_str()]);
        prop_assert!(result.is_err(), "id containing 'mock' must fail: {:?}", id);
    }

    /// `validate_compute_path` accepts exactly "real" and nothing else
    /// (within lowercase alphabetic strings of length 1–8).
    #[test]
    fn only_real_string_passes_validate_compute_path(path in "[a-z]{1,8}") {
        let result = validate_compute_path(&path);
        if path == "real" {
            prop_assert!(result.is_ok(), "'real' must be accepted");
        } else {
            prop_assert!(result.is_err(), "'{path}' must be rejected");
        }
    }
}
