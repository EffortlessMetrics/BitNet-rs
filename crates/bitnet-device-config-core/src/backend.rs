use bitnet_common::BackendRequest;

use crate::DeviceConfig;

impl DeviceConfig {
    /// Return the backend request identity represented by this config.
    #[must_use]
    pub fn backend_request(&self) -> BackendRequest {
        match self {
            DeviceConfig::Auto => BackendRequest::Auto,
            DeviceConfig::Cpu => BackendRequest::Cpu,
            DeviceConfig::Gpu(_) => BackendRequest::Gpu,
            DeviceConfig::IntelNpu(_) => BackendRequest::IntelNpu,
            DeviceConfig::OpenVinoNpu => BackendRequest::OpenVinoNpu,
            DeviceConfig::NvidiaRtx5070TiCuda => BackendRequest::NvidiaRtx5070TiCuda,
            DeviceConfig::NvidiaRtx5070TiWgpu => BackendRequest::NvidiaRtx5070TiWgpu,
            DeviceConfig::IntelA770OpenCl => BackendRequest::IntelA770OpenCl,
            DeviceConfig::Metal => BackendRequest::Metal,
            DeviceConfig::MpsGraph => BackendRequest::MpsGraph,
            DeviceConfig::AppleM4Metal => BackendRequest::AppleM4Metal,
            DeviceConfig::AppleM4MpsGraph => BackendRequest::AppleM4MpsGraph,
            DeviceConfig::AppleM4CpuNeon => BackendRequest::AppleM4CpuNeon,
            DeviceConfig::AppleM3AirMetal => BackendRequest::AppleM3AirMetal,
            DeviceConfig::AppleM3AirMpsGraph => BackendRequest::AppleM3AirMpsGraph,
            DeviceConfig::AppleM3AirCpuNeon => BackendRequest::AppleM3AirCpuNeon,
        }
    }

    /// Stable label for logs and planned receipt fields.
    #[must_use]
    pub fn backend_label(&self) -> String {
        match self {
            DeviceConfig::IntelNpu(0) => "intel-npu".to_string(),
            DeviceConfig::IntelNpu(index) => format!("intel-npu:{index}"),
            _ => self.backend_request().to_string(),
        }
    }
}
