//! AMD ROCm/HIP backend for GPU inference.
//!
//! This crate provides a [`RocmBackend`] for running inference on AMD GPUs via
//! the HIP runtime. The runtime is loaded dynamically (like Level-Zero in the
//! Intel backend) so the crate compiles on any platform — GPU functionality is
//! only available when the HIP shared library is present.

pub mod device;
pub mod error;
pub mod ffi;
pub mod kernel;
pub mod kernels;
pub mod memory;
pub mod stream;

pub use device::{enumerate_devices, RocmDeviceInfo};
pub use error::{RocmError, check_hip};
pub use ffi::HipMemcpyKind;
pub use kernel::LaunchConfig;
pub use memory::DeviceBuffer;
pub use stream::Stream;

/// High-level ROCm backend.
pub struct RocmBackend {
    device_index: usize,
    device_info: Option<RocmDeviceInfo>,
}

impl RocmBackend {
    /// Try to initialise the ROCm backend on the given device.
    pub fn new(device_index: usize) -> error::Result<Self> {
        let devices = enumerate_devices()?;
        let info = devices.into_iter().find(|d| d.index == device_index);
        Ok(Self {
            device_index,
            device_info: info,
        })
    }

    /// Backend name for logging / registry.
    pub fn name(&self) -> &'static str {
        "rocm"
    }

    /// Whether the HIP runtime was detected and a device is available.
    pub fn is_available(&self) -> bool {
        self.device_info.is_some()
    }

    /// Device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Device info, if available.
    pub fn device_info(&self) -> Option<&RocmDeviceInfo> {
        self.device_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HipErrorCode;
    use crate::ffi::{HipDeviceProperties, HipMemcpyKind};

    #[test]
    fn hip_error_code_roundtrip() {
        assert_eq!(HipErrorCode::from_raw(0), HipErrorCode::Success);
        assert_eq!(HipErrorCode::from_raw(2), HipErrorCode::OutOfMemory);
        assert_eq!(HipErrorCode::from_raw(9999), HipErrorCode::Unknown);
    }

    #[test]
    fn check_hip_success() {
        assert!(check_hip(0, "test").is_ok());
    }

    #[test]
    fn check_hip_error() {
        let err = check_hip(2, "alloc").unwrap_err();
        assert!(format!("{err}").contains("OutOfMemory"));
    }

    #[test]
    fn rocm_error_display() {
        let e = RocmError::NoDevice;
        assert_eq!(format!("{e}"), "no AMD GPU device found");
    }

    #[test]
    fn rocm_error_allocation() {
        let e = RocmError::Allocation { size: 1024 };
        assert!(format!("{e}").contains("1024"));
    }

    #[test]
    fn device_properties_name() {
        let mut props = HipDeviceProperties::default();
        props.name[..5].copy_from_slice(b"gfx90");
        assert_eq!(props.device_name(), "gfx90");
    }

    #[test]
    fn device_properties_memory() {
        let mut props = HipDeviceProperties::default();
        props.total_global_mem = 16 * 1024 * 1024 * 1024; // 16 GiB
        assert_eq!(props.total_memory_mib(), 16384);
    }

    #[test]
    fn launch_config_linear() {
        let cfg = LaunchConfig::linear(1024, 256);
        assert_eq!(cfg.grid, (4, 1, 1));
        assert_eq!(cfg.block, (256, 1, 1));
    }

    #[test]
    fn launch_config_2d() {
        let cfg = LaunchConfig::grid_2d(64, 128, 16, 16);
        assert_eq!(cfg.grid, (8, 4, 1));
        assert_eq!(cfg.block, (16, 16, 1));
    }

    #[test]
    fn memcpy_kind_values() {
        assert_eq!(
            memory::memcpy_kind_for(false, true),
            HipMemcpyKind::HostToDevice
        );
        assert_eq!(
            memory::memcpy_kind_for(true, false),
            HipMemcpyKind::DeviceToHost
        );
    }

    #[test]
    fn default_stream_synchronize() {
        let s = Stream::default_stream();
        assert!(s.synchronize().is_ok());
    }

    #[test]
    fn backend_new_no_device() {
        let backend = RocmBackend::new(0).unwrap();
        assert_eq!(backend.name(), "rocm");
        assert!(!backend.is_available());
        assert_eq!(backend.device_index(), 0);
    }

    #[test]
    fn device_buffer_zero_size_error() {
        let err = DeviceBuffer::alloc(0).unwrap_err();
        assert!(format!("{err}").contains("zero-size"));
    }

    // --- integration tests (require AMD GPU + ROCm) ---

    #[test]
    #[ignore = "requires AMD GPU with ROCm runtime installed"]
    fn enumerate_real_devices() {
        let devices = enumerate_devices().unwrap();
        assert!(!devices.is_empty(), "expected at least one AMD GPU");
        for d in &devices {
            assert!(!d.name.is_empty());
            assert!(d.total_memory_mib > 0);
        }
    }
}
