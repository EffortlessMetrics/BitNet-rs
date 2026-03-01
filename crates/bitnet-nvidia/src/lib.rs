//! NVIDIA runtime/vendor detection helpers.
//!
//! This crate provides small, focused helpers that can be reused across
//! backend crates without pulling in larger probing modules.

/// NVIDIA PCI vendor ID.
pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

/// Return `true` if the vendor ID/name/driver tuple appears to be NVIDIA.
#[must_use]
pub fn is_nvidia_device(vendor: u32, name: &str, driver: &str) -> bool {
    vendor == NVIDIA_VENDOR_ID
        || name.to_ascii_lowercase().contains("nvidia")
        || driver.to_ascii_lowercase().contains("nvidia")
}

/// Check whether CUDA runtime tooling is available by invoking `nvidia-smi`.
#[must_use]
pub fn cuda_runtime_available() -> bool {
    std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_detected_by_vendor() {
        assert!(is_nvidia_device(NVIDIA_VENDOR_ID, "Unknown", ""));
    }

    #[test]
    fn nvidia_detected_by_name() {
        assert!(is_nvidia_device(0, "NVIDIA RTX 4090", ""));
    }

    #[test]
    fn nvidia_detected_by_driver() {
        assert!(is_nvidia_device(0, "Generic GPU", "nvidia proprietary"));
    }

    #[test]
    fn non_nvidia_not_detected() {
        assert!(!is_nvidia_device(0x1002, "AMD Radeon", "amdgpu"));
    }
}
