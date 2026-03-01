use anyhow::Result;
use bitnet_common::Device;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Device configuration mode for server/CLI initialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DeviceConfig {
    /// Automatically select the best available device (prefer GPU if available)
    #[default]
    Auto,
    /// Force CPU execution
    Cpu,
    /// Force GPU execution on specific device ID
    Gpu(usize),
}

impl FromStr for DeviceConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "cuda" | "vulkan" | "opencl" | "ocl" | "npu" => Ok(Self::Gpu(0)),
            s if s.starts_with("gpu:") => parse_indexed_device(s, "gpu:"),
            s if s.starts_with("cuda:") => parse_indexed_device(s, "cuda:"),
            s if s.starts_with("vulkan:") => parse_indexed_device(s, "vulkan:"),
            s if s.starts_with("opencl:") => parse_indexed_device(s, "opencl:"),
            s if s.starts_with("ocl:") => parse_indexed_device(s, "ocl:"),
            _ => anyhow::bail!("Unknown device config: {s}"),
        }
    }
}

fn parse_indexed_device(raw: &str, prefix: &str) -> Result<DeviceConfig, anyhow::Error> {
    let id = raw[prefix.len()..].parse::<usize>()?;
    Ok(DeviceConfig::Gpu(id))
}

impl DeviceConfig {
    /// Resolve device configuration to actual runtime device.
    pub fn resolve(&self) -> Device {
        match self {
            Self::Auto => {
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
            Self::Cpu => Device::Cpu,
            Self::Gpu(id) => Device::Cuda(*id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases_and_indices() {
        assert_eq!("gpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
        assert_eq!("cuda:2".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(2));
        assert_eq!("VULKAN:3".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(3));
        assert_eq!("OPENCL:4".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(4));
    }

    #[test]
    fn resolves_cpu_and_indexed_gpu() {
        assert_eq!(DeviceConfig::Cpu.resolve(), Device::Cpu);
        assert_eq!(DeviceConfig::Gpu(7).resolve(), Device::Cuda(7));
    }
}
