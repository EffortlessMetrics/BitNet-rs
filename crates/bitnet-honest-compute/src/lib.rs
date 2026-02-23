//! Honest-compute policy helpers shared by receipts and quality gates.
//!
//! This crate intentionally owns only compute-path and kernel ID policy rules.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Required compute path for honest compute.
pub const REAL_COMPUTE_PATH: &str = "real";
/// Derived compute path when mock kernels are observed.
pub const MOCK_COMPUTE_PATH: &str = "mock";
/// Maximum kernel ID length in characters.
pub const MAX_KERNEL_ID_LENGTH: usize = 128;
/// Maximum number of kernels allowed in a receipt.
pub const MAX_KERNEL_COUNT: usize = 10_000;

const MOCK_KEYWORD: &str = "mock";

/// Error returned when compute path violates honest-compute policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputePathError {
    /// Any value other than `"real"` is rejected.
    NotReal { actual: String },
}

impl Display for ComputePathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReal { actual } => {
                write!(f, "Invalid compute_path: {} (expected 'real')", actual)
            }
        }
    }
}

impl Error for ComputePathError {}

/// Error returned when kernel IDs violate honest-compute policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelValidationError {
    EmptyKernelArray,
    KernelCountExceedsLimit { count: usize },
    EmptyKernelId { index: usize },
    WhitespaceOnlyKernelId { index: usize },
    KernelIdTooLong { index: usize, kernel_id: String },
    MockKernelDetected { index: usize, kernel_id: String },
}

impl Display for KernelValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyKernelArray => {
                write!(f, "Kernel array is empty: honest compute requires kernel IDs")
            }
            Self::KernelCountExceedsLimit { count } => {
                write!(f, "Kernel count {} exceeds 10,000 limit", count)
            }
            Self::EmptyKernelId { index } => write!(f, "Empty kernel ID at index {}", index),
            Self::WhitespaceOnlyKernelId { index } => {
                write!(f, "Whitespace-only kernel ID at index {}", index)
            }
            Self::KernelIdTooLong { index, kernel_id } => {
                write!(f, "Kernel ID at index {} exceeds 128 characters: '{}'", index, kernel_id)
            }
            Self::MockKernelDetected { index, kernel_id } => {
                write!(f, "Mock kernel detected at index {}: '{}'", index, kernel_id)
            }
        }
    }
}

impl Error for KernelValidationError {}

/// Returns true when the kernel ID contains "mock" (ASCII case-insensitive).
pub fn is_mock_kernel_id(kernel_id: &str) -> bool {
    kernel_id
        .as_bytes()
        .windows(MOCK_KEYWORD.len())
        .any(|window| window.eq_ignore_ascii_case(MOCK_KEYWORD.as_bytes()))
}

/// Classify compute path from observed kernel IDs.
///
/// Returns `"mock"` when any kernel ID contains `mock`, otherwise `"real"`.
pub fn classify_compute_path<I, S>(kernels: I) -> &'static str
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    if kernels.into_iter().any(|kernel_id| is_mock_kernel_id(kernel_id.as_ref())) {
        MOCK_COMPUTE_PATH
    } else {
        REAL_COMPUTE_PATH
    }
}

/// Validate that compute path is exactly `"real"`.
pub fn validate_compute_path(compute_path: &str) -> Result<(), ComputePathError> {
    if compute_path == REAL_COMPUTE_PATH {
        Ok(())
    } else {
        Err(ComputePathError::NotReal { actual: compute_path.to_string() })
    }
}

/// Validate kernel ID hygiene policy.
///
/// Rules:
/// - At least one kernel ID must be present.
/// - Maximum kernel count is 10,000.
/// - Kernel IDs must be non-empty, non-whitespace, and <= 128 chars.
/// - Kernel IDs must not contain "mock" (ASCII case-insensitive).
pub fn validate_kernel_ids<I, S>(kernels: I) -> Result<(), KernelValidationError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut count = 0usize;

    for (index, kernel_id) in kernels.into_iter().enumerate() {
        let kernel_id = kernel_id.as_ref();
        count += 1;

        if count > MAX_KERNEL_COUNT {
            return Err(KernelValidationError::KernelCountExceedsLimit { count });
        }
        if kernel_id.is_empty() {
            return Err(KernelValidationError::EmptyKernelId { index });
        }
        if kernel_id.trim().is_empty() {
            return Err(KernelValidationError::WhitespaceOnlyKernelId { index });
        }
        if kernel_id.len() > MAX_KERNEL_ID_LENGTH {
            return Err(KernelValidationError::KernelIdTooLong {
                index,
                kernel_id: kernel_id.to_string(),
            });
        }
        if is_mock_kernel_id(kernel_id) {
            return Err(KernelValidationError::MockKernelDetected {
                index,
                kernel_id: kernel_id.to_string(),
            });
        }
    }

    if count == 0 { Err(KernelValidationError::EmptyKernelArray) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::{
        ComputePathError, KernelValidationError, MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH,
        REAL_COMPUTE_PATH, classify_compute_path, is_mock_kernel_id, validate_compute_path,
        validate_kernel_ids,
    };

    #[test]
    fn mock_detection_is_case_insensitive() {
        assert!(is_mock_kernel_id("mock_kernel"));
        assert!(is_mock_kernel_id("MOCK_kernel"));
        assert!(is_mock_kernel_id("kernel_Mock_suffix"));
        assert!(!is_mock_kernel_id("i2s_cpu_quantized_matmul"));
    }

    #[test]
    fn classify_compute_path_returns_mock_when_any_kernel_is_mock() {
        let kernels = ["i2s_kernel", "MOCK_kernel", "tl1_kernel"];
        assert_eq!(classify_compute_path(kernels), "mock");
    }

    #[test]
    fn classify_compute_path_returns_real_for_non_mock_kernels() {
        let kernels = ["i2s_kernel", "tl1_kernel", "tl2_kernel"];
        assert_eq!(classify_compute_path(kernels), REAL_COMPUTE_PATH);
    }

    #[test]
    fn validate_compute_path_rejects_non_real_values() {
        let err = validate_compute_path("mock").unwrap_err();
        assert_eq!(err, ComputePathError::NotReal { actual: "mock".to_string() });
        assert!(err.to_string().contains("Invalid compute_path"));
    }

    #[test]
    fn validate_compute_path_accepts_real() {
        assert!(validate_compute_path(REAL_COMPUTE_PATH).is_ok());
    }

    #[test]
    fn validate_kernel_ids_rejects_empty_array() {
        let kernels: [&str; 0] = [];
        let err = validate_kernel_ids(kernels).unwrap_err();
        assert_eq!(err, KernelValidationError::EmptyKernelArray);
    }

    #[test]
    fn validate_kernel_ids_rejects_empty_or_whitespace() {
        let err = validate_kernel_ids([""]).unwrap_err();
        assert_eq!(err, KernelValidationError::EmptyKernelId { index: 0 });

        let err = validate_kernel_ids(["   "]).unwrap_err();
        assert_eq!(err, KernelValidationError::WhitespaceOnlyKernelId { index: 0 });
    }

    #[test]
    fn validate_kernel_ids_rejects_mock_and_too_long_values() {
        let err = validate_kernel_ids(["i2s_kernel", "mock_kernel"]).unwrap_err();
        assert_eq!(
            err,
            KernelValidationError::MockKernelDetected {
                index: 1,
                kernel_id: "mock_kernel".to_string()
            }
        );

        let too_long = "k".repeat(MAX_KERNEL_ID_LENGTH + 1);
        let err = validate_kernel_ids([too_long.as_str()]).unwrap_err();
        assert_eq!(err, KernelValidationError::KernelIdTooLong { index: 0, kernel_id: too_long });
    }

    #[test]
    fn validate_kernel_ids_rejects_excessive_count() {
        let kernels = std::iter::repeat_n("kernel", MAX_KERNEL_COUNT + 1);
        let err = validate_kernel_ids(kernels).unwrap_err();
        assert_eq!(err, KernelValidationError::KernelCountExceedsLimit { count: 10_001 });
    }

    #[test]
    fn validate_kernel_ids_accepts_boundary_values() {
        let max_len = "k".repeat(MAX_KERNEL_ID_LENGTH);
        assert!(validate_kernel_ids([max_len.as_str()]).is_ok());

        let kernels = std::iter::repeat_n("kernel", MAX_KERNEL_COUNT);
        assert!(validate_kernel_ids(kernels).is_ok());
    }
}
