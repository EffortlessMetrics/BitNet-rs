//! wgpu adapter and device enumeration.

use crate::error::{Result, WebGpuError};
use tracing::info;

/// Holds the wgpu instance, adapter, device, and queue.
pub struct WebGpuDevice {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WebGpuDevice {
    /// Create a new device, preferring high-performance adapters.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(WebGpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        info!(
            backend = ?adapter_info.backend,
            device = %adapter_info.name,
            "selected WebGPU adapter"
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bitnet-webgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Return the adapter name.
    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name
    }

    /// Return the wgpu backend in use (Vulkan, Metal, DX12, …).
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter.get_info().backend
    }

    /// Maximum buffer size supported by the device.
    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }
}
