//! Snapshot tests for `bitnet-honest-compute` public API surface.
//!
//! Pins error message formats and constant values to prevent silent breakage
//! of the honest-compute policy contract.

use bitnet_honest_compute::{
    ComputePathError, KernelValidationError, MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH,
    MOCK_COMPUTE_PATH, REAL_COMPUTE_PATH,
};

#[test]
fn constants_snapshot() {
    let summary = format!(
        "real={REAL_COMPUTE_PATH} mock={MOCK_COMPUTE_PATH} \
         max_id_len={MAX_KERNEL_ID_LENGTH} max_count={MAX_KERNEL_COUNT}"
    );
    insta::assert_snapshot!("honest_compute_constants", summary);
}

#[test]
fn compute_path_error_not_real_display() {
    let err = ComputePathError::NotReal { actual: "mock".to_string() };
    insta::assert_snapshot!("compute_path_error_not_real", err.to_string());
}

#[test]
fn kernel_validation_error_empty_array_display() {
    let err = KernelValidationError::EmptyKernelArray;
    insta::assert_snapshot!("kernel_error_empty_array", err.to_string());
}

#[test]
fn kernel_validation_error_count_exceeds_limit_display() {
    let err = KernelValidationError::KernelCountExceedsLimit { count: 20_000 };
    insta::assert_snapshot!("kernel_error_count_exceeds_limit", err.to_string());
}

#[test]
fn kernel_validation_error_mock_kernel_display() {
    let err = KernelValidationError::MockKernelDetected {
        index: 3,
        kernel_id: "mock_cpu_matmul".to_string(),
    };
    insta::assert_snapshot!("kernel_error_mock_detected", err.to_string());
}

#[test]
fn kernel_validation_error_id_too_long_display() {
    let long_id = "x".repeat(200);
    let err = KernelValidationError::KernelIdTooLong { index: 0, kernel_id: long_id };
    // Only snapshot the first ~60 chars to avoid giant snapshot files
    let display = err.to_string();
    let truncated = &display[..display.len().min(80)];
    insta::assert_snapshot!("kernel_error_id_too_long_prefix", truncated);
}
