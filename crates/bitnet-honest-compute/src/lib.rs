//! Honest-compute policy helpers shared by receipts and quality gates.
//!
//! `bitnet-honest-compute` is the façade crate for the policy contracts now
//! centralized in `bitnet-honest-compute-core`.

pub use bitnet_honest_compute_core::{
    ComputePathError, KernelValidationError, MAX_KERNEL_COUNT, MAX_KERNEL_ID_LENGTH,
    MOCK_COMPUTE_PATH, REAL_COMPUTE_PATH, classify_compute_path, is_mock_kernel_id,
    validate_compute_path, validate_kernel_ids,
};
