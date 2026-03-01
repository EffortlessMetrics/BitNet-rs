//! Core device configuration parsing and runtime resolution.

use anyhow::Result;
use bitnet_common::Device;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Device configuration mode for runtime initialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DeviceConfig {
    /// Automatically select the best available device (prefer GPU if available).
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force GPU execution on specific device ID.
    Gpu(usize),
}

impl FromStr for DeviceConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(DeviceConfig::Auto),
            "cpu" => Ok(DeviceConfig::Cpu),
            "gpu" | "cuda" | "vulkan" | "opencl" | "ocl" | "npu" => Ok(DeviceConfig::Gpu(0)),
            s if s.starts_with("gpu:") => Ok(DeviceConfig::Gpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("cuda:") => Ok(DeviceConfig::Gpu(s[5..].parse::<usize>()?)),
            s if s.starts_with("vulkan:") => Ok(DeviceConfig::Gpu(s[7..].parse::<usize>()?)),
            s if s.starts_with("opencl:") => Ok(DeviceConfig::Gpu(s[7..].parse::<usize>()?)),
            s if s.starts_with("ocl:") => Ok(DeviceConfig::Gpu(s[4..].parse::<usize>()?)),
            _ => anyhow::bail!("Unknown device config: {}", s),
        }
    }
}

impl DeviceConfig {
    /// Resolve configuration to an executable device choice.
    #[must_use]
    pub fn resolve(&self) -> Device {
        match self {
            DeviceConfig::Auto => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use bitnet_kernels::device_features::gpu_available_runtime;
                    if gpu_available_runtime() { Device::Cuda(0) } else { Device::Cpu }
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    Device::Cpu
                }
            }
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Gpu(id) => Device::Cuda(*id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceConfig;

    #[test]
    fn parses_supported_aliases() {
        assert_eq!("cpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
        assert_eq!("auto".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
        assert_eq!("gpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
        assert_eq!("cuda:2".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(2));
        assert_eq!("vulkan:3".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(3));
    }

    #[test]
    fn rejects_invalid_values() {
        assert!("unknown".parse::<DeviceConfig>().is_err());
        assert!("gpu:".parse::<DeviceConfig>().is_err());
        assert!("gpu:abc".parse::<DeviceConfig>().is_err());
    }
}
