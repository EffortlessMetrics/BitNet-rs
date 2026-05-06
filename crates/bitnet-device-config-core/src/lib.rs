//! Core device configuration parsing and runtime resolution.

use anyhow::Result;
use bitnet_common::{BackendRequest, Device};
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
    /// Preserve Intel NPU device identity.
    IntelNpu(usize),
    /// Preserve OpenVINO NPU backend identity.
    OpenVinoNpu,
    /// Preserve Intel Arc 140V GPU identity.
    IntelArc140v(usize),
    /// Preserve OpenVINO GPU device identity.
    OpenVinoGpu(usize),
    /// Preserve a native Metal backend identity.
    Metal,
    /// Preserve an MPSGraph graph/reference backend identity.
    MpsGraph,
    /// Preserve the Apple M4 native Metal backend identity.
    AppleM4Metal,
    /// Preserve the Apple M4 MPSGraph graph/reference backend identity.
    AppleM4MpsGraph,
    /// Preserve the Apple M4 CPU/NEON fallback/parity backend identity.
    AppleM4CpuNeon,
}

impl FromStr for DeviceConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(DeviceConfig::Auto),
            "cpu" => Ok(DeviceConfig::Cpu),
            "gpu" | "cuda" | "vulkan" => Ok(DeviceConfig::Gpu(0)),
            "opencl" | "ocl" | "intel-gpu" => Ok(DeviceConfig::OpenVinoGpu(0)),
            "npu" | "intel-npu" => Ok(DeviceConfig::IntelNpu(0)),
            "openvino-npu" => Ok(DeviceConfig::OpenVinoNpu),
            "intel-arc-140v" | "arc-140v" => Ok(DeviceConfig::IntelArc140v(0)),
            "openvino-gpu" | "gpu.0" => Ok(DeviceConfig::OpenVinoGpu(0)),
            "intel-arc-140v-openvino-gpu" | "arc-140v-openvino-gpu" => {
                Ok(DeviceConfig::OpenVinoGpu(0))
            }
            "metal" => Ok(DeviceConfig::Metal),
            "mpsgraph" => Ok(DeviceConfig::MpsGraph),
            "apple-m4-metal" => Ok(DeviceConfig::AppleM4Metal),
            "apple-m4-mpsgraph" => Ok(DeviceConfig::AppleM4MpsGraph),
            "apple-m4-cpu-neon" => Ok(DeviceConfig::AppleM4CpuNeon),
            s if s.starts_with("gpu:") => Ok(DeviceConfig::Gpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("cuda:") => Ok(DeviceConfig::Gpu(s[5..].parse::<usize>()?)),
            s if s.starts_with("vulkan:") => Ok(DeviceConfig::Gpu(s[7..].parse::<usize>()?)),
            s if s.starts_with("opencl:") => {
                Ok(DeviceConfig::OpenVinoGpu(s[7..].parse::<usize>()?))
            }
            s if s.starts_with("ocl:") => Ok(DeviceConfig::OpenVinoGpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("intel-gpu:") => {
                Ok(DeviceConfig::OpenVinoGpu(s[10..].parse::<usize>()?))
            }
            s if s.starts_with("npu:") => Ok(DeviceConfig::IntelNpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("intel-npu:") => {
                Ok(DeviceConfig::IntelNpu(s[10..].parse::<usize>()?))
            }
            s if s.starts_with("intel-arc-140v:") => {
                Ok(DeviceConfig::IntelArc140v(s[15..].parse::<usize>()?))
            }
            s if s.starts_with("openvino-gpu:") => {
                Ok(DeviceConfig::OpenVinoGpu(s[13..].parse::<usize>()?))
            }
            s if s.starts_with("gpu.") => Ok(DeviceConfig::OpenVinoGpu(s[4..].parse::<usize>()?)),
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
            DeviceConfig::IntelNpu(_) | DeviceConfig::OpenVinoNpu => Device::Npu,
            DeviceConfig::IntelArc140v(id) | DeviceConfig::OpenVinoGpu(id) => Device::OpenCL(*id),
            DeviceConfig::Metal | DeviceConfig::AppleM4Metal => Device::Metal,
            // MPSGraph is a separate proof label; runtime execution is introduced in a later item.
            DeviceConfig::MpsGraph | DeviceConfig::AppleM4MpsGraph => Device::Cpu,
            DeviceConfig::AppleM4CpuNeon => Device::Cpu,
        }
    }

    /// Return the backend request identity represented by this config.
    #[must_use]
    pub fn backend_request(&self) -> BackendRequest {
        match self {
            DeviceConfig::Auto => BackendRequest::Auto,
            DeviceConfig::Cpu => BackendRequest::Cpu,
            DeviceConfig::Gpu(_) => BackendRequest::Gpu,
            DeviceConfig::IntelNpu(_) => BackendRequest::IntelNpu,
            DeviceConfig::OpenVinoNpu => BackendRequest::OpenVinoNpu,
            DeviceConfig::IntelArc140v(_) => BackendRequest::IntelArc140v,
            DeviceConfig::OpenVinoGpu(_) => BackendRequest::IntelArc140vOpenVinoGpu,
            DeviceConfig::Metal => BackendRequest::Metal,
            DeviceConfig::MpsGraph => BackendRequest::MpsGraph,
            DeviceConfig::AppleM4Metal => BackendRequest::AppleM4Metal,
            DeviceConfig::AppleM4MpsGraph => BackendRequest::AppleM4MpsGraph,
            DeviceConfig::AppleM4CpuNeon => BackendRequest::AppleM4CpuNeon,
        }
    }

    /// Stable label for logs and planned receipt fields.
    #[must_use]
    pub fn backend_label(&self) -> String {
        self.backend_request().to_string()
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
        assert_eq!("opencl".parse::<DeviceConfig>().unwrap(), DeviceConfig::OpenVinoGpu(0));
        assert_eq!("npu".parse::<DeviceConfig>().unwrap(), DeviceConfig::IntelNpu(0));
        assert_eq!("openvino-npu".parse::<DeviceConfig>().unwrap(), DeviceConfig::OpenVinoNpu);
        assert_eq!(
            "intel-arc-140v".parse::<DeviceConfig>().unwrap(),
            DeviceConfig::IntelArc140v(0)
        );
        assert_eq!("metal".parse::<DeviceConfig>().unwrap(), DeviceConfig::Metal);
        assert_eq!("mpsgraph".parse::<DeviceConfig>().unwrap(), DeviceConfig::MpsGraph);
        assert_eq!("apple-m4-metal".parse::<DeviceConfig>().unwrap(), DeviceConfig::AppleM4Metal);
        assert_eq!(
            "apple-m4-mpsgraph".parse::<DeviceConfig>().unwrap(),
            DeviceConfig::AppleM4MpsGraph
        );
        assert_eq!(
            "apple-m4-cpu-neon".parse::<DeviceConfig>().unwrap(),
            DeviceConfig::AppleM4CpuNeon
        );
    }

    #[test]
    fn rejects_invalid_values() {
        assert!("unknown".parse::<DeviceConfig>().is_err());
        assert!("gpu:".parse::<DeviceConfig>().is_err());
        assert!("gpu:abc".parse::<DeviceConfig>().is_err());
    }

    #[test]
    fn apple_backend_labels_do_not_alias() {
        let metal = "metal".parse::<DeviceConfig>().unwrap();
        let apple_metal = "apple-m4-metal".parse::<DeviceConfig>().unwrap();
        let mpsgraph = "mpsgraph".parse::<DeviceConfig>().unwrap();
        let apple_mpsgraph = "apple-m4-mpsgraph".parse::<DeviceConfig>().unwrap();
        let apple_cpu = "apple-m4-cpu-neon".parse::<DeviceConfig>().unwrap();

        assert_eq!(metal.backend_label(), "metal");
        assert_eq!(apple_metal.backend_label(), "apple-m4-metal");
        assert_eq!(mpsgraph.backend_label(), "mpsgraph");
        assert_eq!(apple_mpsgraph.backend_label(), "apple-m4-mpsgraph");
        assert_eq!(apple_cpu.backend_label(), "apple-m4-cpu-neon");
    }
}
