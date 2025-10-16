//! GPU test utilities for BitNet.rs
//!
//! This module provides utilities for GPU tests to skip cleanly on CPU-only machines.

/// Check if GPU tests are explicitly enabled via environment variable.
///
/// GPU tests should check this at the start and skip if not enabled:
///
/// ```rust,ignore
/// #[cfg(feature = "gpu")]
/// #[test]
/// fn my_gpu_test() {
///     if !tests::common::gpu::gpu_tests_enabled() {
///         eprintln!("Skipping GPU test (set BITNET_ENABLE_GPU_TESTS=1 to run)");
///         return;
///     }
///     // ... GPU test code ...
/// }
/// ```
///
/// This prevents spurious failures on CPU-only CI runners or local machines
/// that don't have CUDA installed.
pub fn gpu_tests_enabled() -> bool {
    std::env::var("BITNET_ENABLE_GPU_TESTS").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tests_disabled_by_default() {
        // Ensure BITNET_ENABLE_GPU_TESTS is not set in this test
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::remove_var("BITNET_ENABLE_GPU_TESTS");
        }
        assert!(!gpu_tests_enabled());
    }

    #[test]
    fn test_gpu_tests_enabled_when_set() {
        // Set the environment variable for this test
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::set_var("BITNET_ENABLE_GPU_TESTS", "1");
        }
        assert!(gpu_tests_enabled());
        // Clean up
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::remove_var("BITNET_ENABLE_GPU_TESTS");
        }
    }

    #[test]
    fn test_gpu_tests_disabled_when_set_to_non_one() {
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::set_var("BITNET_ENABLE_GPU_TESTS", "0");
        }
        assert!(!gpu_tests_enabled());
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::set_var("BITNET_ENABLE_GPU_TESTS", "true");
        }
        assert!(!gpu_tests_enabled());
        // Clean up
        // SAFETY: This is a test environment and we're controlling the environment variable
        unsafe {
            std::env::remove_var("BITNET_ENABLE_GPU_TESTS");
        }
    }
}
