//! Edge-case tests for bitnet-honest-compute: compute path validation,
//! kernel ID validation, mock detection, classification.

use bitnet_honest_compute::{
    ComputePathError, KernelValidationError, MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH,
    MOCK_COMPUTE_PATH, REAL_COMPUTE_PATH, classify_compute_path, is_mock_kernel_id,
    validate_compute_path, validate_kernel_ids,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn constants_correct() {
    assert_eq!(REAL_COMPUTE_PATH, "real");
    assert_eq!(MOCK_COMPUTE_PATH, "mock");
    assert_eq!(MAX_KERNEL_ID_LENGTH, 128);
    assert_eq!(MAX_KERNEL_COUNT, 10_000);
}

// ---------------------------------------------------------------------------
// is_mock_kernel_id
// ---------------------------------------------------------------------------

#[test]
fn mock_kernel_detected_lowercase() {
    assert!(is_mock_kernel_id("mock_gemv"));
    assert!(is_mock_kernel_id("i2s_mock"));
    assert!(is_mock_kernel_id("a_mock_kernel"));
}

#[test]
fn mock_kernel_detected_uppercase() {
    assert!(is_mock_kernel_id("MOCK_GEMV"));
    assert!(is_mock_kernel_id("I2S_MOCK"));
}

#[test]
fn mock_kernel_detected_mixed_case() {
    assert!(is_mock_kernel_id("Mock_kernel"));
    assert!(is_mock_kernel_id("mOcK_kernel"));
}

#[test]
fn mock_kernel_not_detected() {
    assert!(!is_mock_kernel_id("i2s_gemv"));
    assert!(!is_mock_kernel_id("rope_apply"));
    assert!(!is_mock_kernel_id(""));
    assert!(!is_mock_kernel_id("moc")); // too short
}

#[test]
fn mock_kernel_substring_in_middle() {
    assert!(is_mock_kernel_id("prefix_mock_suffix"));
}

// ---------------------------------------------------------------------------
// classify_compute_path
// ---------------------------------------------------------------------------

#[test]
fn classify_all_real_kernels() {
    let kernels = vec!["i2s_gemv", "rope_apply", "softmax_cpu"];
    assert_eq!(classify_compute_path(kernels), "real");
}

#[test]
fn classify_with_mock_kernel() {
    let kernels = vec!["i2s_gemv", "mock_rope", "softmax_cpu"];
    assert_eq!(classify_compute_path(kernels), "mock");
}

#[test]
fn classify_empty_kernels() {
    let kernels: Vec<&str> = vec![];
    assert_eq!(classify_compute_path(kernels), "real");
}

#[test]
fn classify_single_real() {
    assert_eq!(classify_compute_path(vec!["cpu_matmul"]), "real");
}

#[test]
fn classify_single_mock() {
    assert_eq!(classify_compute_path(vec!["mock_matmul"]), "mock");
}

// ---------------------------------------------------------------------------
// validate_compute_path
// ---------------------------------------------------------------------------

#[test]
fn validate_compute_path_real_ok() {
    assert!(validate_compute_path("real").is_ok());
}

#[test]
fn validate_compute_path_mock_error() {
    let err = validate_compute_path("mock").unwrap_err();
    assert_eq!(err, ComputePathError::NotReal { actual: "mock".to_string() });
}

#[test]
fn validate_compute_path_empty_error() {
    let err = validate_compute_path("").unwrap_err();
    assert_eq!(err, ComputePathError::NotReal { actual: "".to_string() });
}

#[test]
fn validate_compute_path_arbitrary_string_error() {
    let err = validate_compute_path("test").unwrap_err();
    assert_eq!(err, ComputePathError::NotReal { actual: "test".to_string() });
}

#[test]
fn compute_path_error_display() {
    let err = ComputePathError::NotReal { actual: "mock".to_string() };
    let msg = format!("{err}");
    assert!(msg.contains("mock"));
    assert!(msg.contains("real"));
}

#[test]
fn compute_path_error_is_std_error() {
    let err = ComputePathError::NotReal { actual: "x".to_string() };
    let _: &dyn std::error::Error = &err;
}

// ---------------------------------------------------------------------------
// validate_kernel_ids — valid cases
// ---------------------------------------------------------------------------

#[test]
fn validate_kernels_ok() {
    let result = validate_kernel_ids(vec!["i2s_gemv", "rope_apply"]);
    assert!(result.is_ok());
}

#[test]
fn validate_kernels_single_ok() {
    assert!(validate_kernel_ids(vec!["cpu_matmul"]).is_ok());
}

// ---------------------------------------------------------------------------
// validate_kernel_ids — error cases
// ---------------------------------------------------------------------------

#[test]
fn validate_kernels_empty_array() {
    let err = validate_kernel_ids(Vec::<&str>::new()).unwrap_err();
    assert_eq!(err, KernelValidationError::EmptyKernelArray);
}

#[test]
fn validate_kernels_empty_id() {
    let err = validate_kernel_ids(vec!["valid", ""]).unwrap_err();
    assert_eq!(err, KernelValidationError::EmptyKernelId { index: 1 });
}

#[test]
fn validate_kernels_whitespace_only() {
    let err = validate_kernel_ids(vec!["   "]).unwrap_err();
    assert_eq!(err, KernelValidationError::WhitespaceOnlyKernelId { index: 0 });
}

#[test]
fn validate_kernels_too_long() {
    let long_id = "x".repeat(MAX_KERNEL_ID_LENGTH + 1);
    let err = validate_kernel_ids(vec![long_id.as_str()]).unwrap_err();
    assert!(matches!(err, KernelValidationError::KernelIdTooLong { index: 0, .. }));
}

#[test]
fn validate_kernels_exactly_max_length_ok() {
    let id = "x".repeat(MAX_KERNEL_ID_LENGTH);
    assert!(validate_kernel_ids(vec![id.as_str()]).is_ok());
}

#[test]
fn validate_kernels_mock_detected() {
    let err = validate_kernel_ids(vec!["mock_gemv"]).unwrap_err();
    assert!(matches!(err, KernelValidationError::MockKernelDetected { index: 0, .. }));
}

#[test]
fn validate_kernels_mock_case_insensitive() {
    let err = validate_kernel_ids(vec!["MOCK_GEMV"]).unwrap_err();
    assert!(matches!(err, KernelValidationError::MockKernelDetected { index: 0, .. }));
}

// ---------------------------------------------------------------------------
// KernelValidationError Display
// ---------------------------------------------------------------------------

#[test]
fn kernel_error_display_messages() {
    let errors = vec![
        KernelValidationError::EmptyKernelArray,
        KernelValidationError::KernelCountExceedsLimit { count: 20000 },
        KernelValidationError::EmptyKernelId { index: 0 },
        KernelValidationError::WhitespaceOnlyKernelId { index: 1 },
        KernelValidationError::KernelIdTooLong { index: 2, kernel_id: "x".repeat(200) },
        KernelValidationError::MockKernelDetected { index: 3, kernel_id: "mock_x".into() },
    ];
    for err in &errors {
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }
}

#[test]
fn kernel_error_is_std_error() {
    let err = KernelValidationError::EmptyKernelArray;
    let _: &dyn std::error::Error = &err;
}

#[test]
fn kernel_error_clone_eq() {
    let err1 = KernelValidationError::EmptyKernelArray;
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}
