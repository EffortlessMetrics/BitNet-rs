use anyhow::Result;
use std::str::FromStr;

use crate::DeviceConfig;

impl FromStr for DeviceConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(DeviceConfig::Auto),
            "cpu" => Ok(DeviceConfig::Cpu),
            "gpu" | "cuda" | "vulkan" | "opencl" | "ocl" => Ok(DeviceConfig::Gpu(0)),
            "npu" | "intel-npu" => Ok(DeviceConfig::IntelNpu(0)),
            "openvino-npu" | "intel-npu-openvino" => Ok(DeviceConfig::OpenVinoNpu),
            "nvidia-rtx-5070-ti-cuda" => Ok(DeviceConfig::NvidiaRtx5070TiCuda),
            "nvidia-rtx-5070-ti-wgpu" => Ok(DeviceConfig::NvidiaRtx5070TiWgpu),
            "intel-a770-opencl" | "a770-opencl" => Ok(DeviceConfig::IntelA770OpenCl),
            "metal" => Ok(DeviceConfig::Metal),
            "mpsgraph" => Ok(DeviceConfig::MpsGraph),
            "apple-m4-metal" => Ok(DeviceConfig::AppleM4Metal),
            "apple-m4-mpsgraph" => Ok(DeviceConfig::AppleM4MpsGraph),
            "apple-m4-cpu-neon" => Ok(DeviceConfig::AppleM4CpuNeon),
            "apple-m3-air-metal" => Ok(DeviceConfig::AppleM3AirMetal),
            "apple-m3-air-mpsgraph" => Ok(DeviceConfig::AppleM3AirMpsGraph),
            "apple-m3-air-cpu-neon" => Ok(DeviceConfig::AppleM3AirCpuNeon),
            s if s.starts_with("gpu:") => Ok(DeviceConfig::Gpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("cuda:") => Ok(DeviceConfig::Gpu(s[5..].parse::<usize>()?)),
            s if s.starts_with("vulkan:") => Ok(DeviceConfig::Gpu(s[7..].parse::<usize>()?)),
            s if s.starts_with("opencl:") => Ok(DeviceConfig::Gpu(s[7..].parse::<usize>()?)),
            s if s.starts_with("ocl:") => Ok(DeviceConfig::Gpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("npu:") => Ok(DeviceConfig::IntelNpu(s[4..].parse::<usize>()?)),
            s if s.starts_with("intel-npu:") => {
                Ok(DeviceConfig::IntelNpu(s[10..].parse::<usize>()?))
            }
            _ => anyhow::bail!("Unknown device config: {}", s),
        }
    }
}
