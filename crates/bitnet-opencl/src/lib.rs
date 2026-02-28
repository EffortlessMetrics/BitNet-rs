//! `OpenCL` runtime binding layer for `BitNet` GPU inference.
//!
//! Provides safe wrappers around opencl3 types with dynamic library
//! loading and graceful fallback when no `OpenCL` runtime is installed.

pub mod benchmark_utils;
pub mod runtime;

pub use runtime::{
    OpenClDeviceInfo, OpenClPlatformInfo, enumerate_gpu_devices, enumerate_platforms,
    mock_device_intel_arc, mock_device_nvidia, mock_platform, opencl_available,
};

/// Check whether the `OCL_ICD_VENDORS` environment variable is set.
pub fn ocl_icd_vendors_configured() -> bool {
    std::env::var("OCL_ICD_VENDORS").is_ok()
}

/// Check whether the `VK_ICD_FILENAMES` environment variable is set.
pub fn vulkan_icd_configured() -> bool {
    std::env::var("VK_ICD_FILENAMES").is_ok()
}

/// Return the OpenCL ICD vendors path, if configured.
pub fn ocl_icd_vendors_path() -> Option<String> {
    std::env::var("OCL_ICD_VENDORS").ok()
}

/// Return the Vulkan ICD filenames path, if configured.
pub fn vulkan_icd_path() -> Option<String> {
    std::env::var("VK_ICD_FILENAMES").ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocl_icd_vendors_returns_option() {
        let _ = ocl_icd_vendors_configured();
    }

    #[test]
    fn test_vulkan_icd_returns_option() {
        let _ = vulkan_icd_configured();
    }
}
