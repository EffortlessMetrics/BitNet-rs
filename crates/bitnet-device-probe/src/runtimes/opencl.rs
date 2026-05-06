//! OpenCL runtime visibility wrapper for platform receipts.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

/// OpenCL device facts needed for hardware-lane identity receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClRuntimeDevice {
    /// OpenCL platform name.
    pub platform_name: Option<String>,
    /// Device name.
    pub device_name: String,
    /// Device vendor.
    pub vendor: String,
    /// Driver version reported by OpenCL.
    pub driver_version: Option<String>,
    /// Whether the device is a GPU.
    pub is_gpu: bool,
}

/// OpenCL runtime visibility result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClRuntimeProbe {
    /// Whether an OpenCL runtime was visible.
    pub runtime_available: bool,
    /// Devices reported by the runtime.
    pub devices: Vec<OpenClRuntimeDevice>,
    /// Non-fatal probe error when the runtime was absent or unusable.
    pub error: Option<String>,
}

impl OpenClRuntimeProbe {
    /// Build an unavailable OpenCL probe result.
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self { runtime_available: false, devices: Vec::new(), error: Some(reason.into()) }
    }
}

/// Probe OpenCL visibility without making an execution claim.
pub fn probe_opencl_runtime() -> OpenClRuntimeProbe {
    #[cfg(feature = "opencl")]
    {
        let result = crate::opencl::probe_opencl();
        let devices = result
            .devices
            .into_iter()
            .map(|device| {
                let is_gpu = device.is_gpu();
                OpenClRuntimeDevice {
                    platform_name: result
                        .platforms
                        .get(device.platform_index)
                        .map(|platform| platform.name.clone()),
                    device_name: device.name,
                    vendor: device.vendor,
                    driver_version: Some(device.driver_version),
                    is_gpu,
                }
            })
            .collect();

        OpenClRuntimeProbe {
            runtime_available: result.runtime_available,
            devices,
            error: result.error,
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        OpenClRuntimeProbe::unavailable("compiled without opencl feature")
    }
}
