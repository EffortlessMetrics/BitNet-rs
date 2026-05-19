use bitnet_common::Device;

use crate::DeviceConfig;

impl DeviceConfig {
    /// Resolve configuration to an executable device choice.
    #[must_use]
    pub fn resolve(&self) -> Device {
        match self {
            DeviceConfig::Auto => resolve_auto_device(),
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Gpu(id) => Device::Cuda(*id),
            DeviceConfig::IntelNpu(_) | DeviceConfig::OpenVinoNpu => Device::Npu,
            DeviceConfig::NvidiaRtx5070TiCuda => Device::Cuda(0),
            // WGPU is a reference-lane identity; execution lands in a later item.
            DeviceConfig::NvidiaRtx5070TiWgpu => Device::Cpu,
            DeviceConfig::IntelA770OpenCl => Device::OpenCL(0),
            DeviceConfig::Metal | DeviceConfig::AppleM4Metal => Device::Metal,
            // M3 Air Metal is an identity-only request until a receipt-backed runtime item lands.
            DeviceConfig::AppleM3AirMetal => Device::Cpu,
            // MPSGraph is a separate proof label; runtime execution is introduced in a later item.
            DeviceConfig::MpsGraph
            | DeviceConfig::AppleM4MpsGraph
            | DeviceConfig::AppleM3AirMpsGraph => Device::Cpu,
            DeviceConfig::AppleM4CpuNeon | DeviceConfig::AppleM3AirCpuNeon => Device::Cpu,
        }
    }
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn resolve_auto_device() -> Device {
    use bitnet_kernels::device_features::gpu_available_runtime;

    if gpu_available_runtime() { Device::Cuda(0) } else { Device::Cpu }
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn resolve_auto_device() -> Device {
    Device::Cpu
}
