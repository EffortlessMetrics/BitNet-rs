//! Device enumeration and selection for OpenCL GPU targets.

use crate::error::{OpenClError, Result};
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::platform::{Platform, get_platforms};
use tracing::{debug, info};

/// An OpenCL GPU device with its parent platform.
#[derive(Debug)]
pub struct OpenClDevice {
    /// The raw opencl3 device handle.
    pub(crate) device: Device,
    /// The platform this device belongs to.
    #[allow(dead_code)]
    pub(crate) platform: Platform,
    /// Human-readable device name.
    pub device_name: String,
    /// Human-readable platform name.
    pub platform_name: String,
    /// Device vendor string.
    pub vendor: String,
}

impl OpenClDevice {
    /// Enumerate all GPU devices across all OpenCL platforms.
    pub fn enumerate_gpus() -> Result<Vec<OpenClDevice>> {
        let platforms = get_platforms().map_err(|_e| OpenClError::NoPlatforms)?;

        if platforms.is_empty() {
            return Err(OpenClError::NoPlatforms);
        }

        let mut devices = Vec::new();
        for platform in platforms {
            let platform_name = platform.name().unwrap_or_default();
            debug!("Scanning OpenCL platform: {}", platform_name);

            let device_ids = platform
                .get_devices(CL_DEVICE_TYPE_GPU)
                .unwrap_or_default();

            for device_id in device_ids {
                let device = Device::new(device_id);
                let device_name = device.name().unwrap_or_default();
                let vendor = device.vendor().unwrap_or_default();
                debug!("Found GPU: {} (vendor: {})", device_name, vendor);

                devices.push(OpenClDevice {
                    device,
                    platform,
                    device_name,
                    platform_name: platform_name.clone(),
                    vendor,
                });
            }
        }

        Ok(devices)
    }

    /// Find the first Intel GPU device.
    pub fn find_intel_gpu() -> Result<OpenClDevice> {
        let gpus = Self::enumerate_gpus()?;

        for gpu in gpus {
            if gpu.vendor.to_lowercase().contains("intel") {
                info!("Selected Intel GPU: {}", gpu.device_name);
                return Ok(gpu);
            }
        }

        Err(OpenClError::NoDevice {
            reason: "no Intel GPU found via OpenCL".into(),
        })
    }

    /// Check whether this device is from Intel.
    pub fn is_intel(&self) -> bool {
        self.vendor.to_lowercase().contains("intel")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_does_not_panic() {
        let _ = OpenClDevice::enumerate_gpus();
    }

    #[test]
    fn find_intel_gpu_graceful_on_missing_hardware() {
        let result = OpenClDevice::find_intel_gpu();
        match result {
            Ok(dev) => assert!(dev.is_intel()),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("no Intel GPU")
                        || msg.contains("no OpenCL platforms"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn is_intel_vendor_matching() {
        assert!(!vendor_matches("NVIDIA Corporation"));
        assert!(vendor_matches("Intel(R) Corporation"));
        assert!(vendor_matches("intel"));
    }

    fn vendor_matches(vendor: &str) -> bool {
        vendor.to_lowercase().contains("intel")
    }
}
