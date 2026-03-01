//! GPU device initialization and adapter discovery.

use crate::error::WgpuError;

/// Configuration for wgpu device creation.
#[derive(Debug, Clone)]
pub struct WgpuDeviceConfig {
    /// Power preference hint for adapter selection.
    pub power_preference: wgpu::PowerPreference,
    /// Backend API filter (e.g. Vulkan-only).
    pub backend_bits: wgpu::Backends,
    /// Features required on the device.
    pub required_features: wgpu::Features,
    /// Limits required on the device.
    pub required_limits: wgpu::Limits,
}

impl Default for WgpuDeviceConfig {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            backend_bits: wgpu::Backends::VULKAN,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }
    }
}

/// Summary information about the selected adapter.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable adapter name (e.g. "NVIDIA RTX 5070 Ti").
    pub name: String,
    /// Backend API in use (Vulkan, Metal, DX12, …).
    pub backend: wgpu::Backend,
    /// Driver description reported by the adapter.
    pub driver: String,
    /// Driver-info string (version, etc.).
    pub driver_info: String,
    /// Device type (discrete GPU, integrated, CPU, …).
    pub device_type: wgpu::DeviceType,
    /// Vendor ID.
    pub vendor: u32,
    /// Negotiated device limits.
    pub limits: wgpu::Limits,
}

/// A wgpu device handle with adapter metadata.
pub struct WgpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
}

impl WgpuDevice {
    /// Create a new device asynchronously.
    pub async fn new(config: &WgpuDeviceConfig) -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: config.backend_bits,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| WgpuError::device("no suitable GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bitnet-wgpu"),
                    required_features: config.required_features,
                    required_limits: config.required_limits.clone(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| WgpuError::device(e))?;

        tracing::info!(
            adapter = %adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "wgpu device created"
        );

        Ok(Self { device, queue, adapter })
    }

    /// Blocking wrapper around [`Self::new`] using `pollster`.
    pub fn new_blocking(config: &WgpuDeviceConfig) -> Result<Self, WgpuError> {
        pollster::block_on(Self::new(config))
    }

    /// Query adapter and device information.
    pub fn info(&self) -> DeviceInfo {
        let info = self.adapter.get_info();
        DeviceInfo {
            name: info.name.clone(),
            backend: info.backend,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            device_type: info.device_type,
            vendor: info.vendor,
            limits: self.device.limits(),
        }
    }

    /// PCI vendor ID for NVIDIA.
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;

    /// Returns `true` if the adapter is an NVIDIA GPU.
    pub fn is_nvidia(&self) -> bool {
        self.adapter.get_info().vendor == Self::NVIDIA_VENDOR_ID
    }

    /// Returns `true` if the device supports subgroup (warp/wave) operations.
    pub fn supports_subgroup_ops(&self) -> bool {
        self.device.features().contains(wgpu::Features::SUBGROUP)
    }

    /// Access the underlying `wgpu::Device`.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the underlying `wgpu::Queue`.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Access the underlying `wgpu::Adapter`.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info = self.adapter.get_info();
        f.debug_struct("WgpuDevice")
            .field("name", &info.name)
            .field("backend", &info.backend)
            .field("vendor", &info.vendor)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPU-gated tests ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_creation_default_config() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default());
        assert!(dev.is_ok(), "device creation failed: {:?}", dev.err());
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_info_populated() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        let info = dev.info();
        assert!(!info.name.is_empty(), "adapter name should not be empty");
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_backend_is_vulkan_by_default() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        let info = dev.info();
        assert_eq!(info.backend, wgpu::Backend::Vulkan);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_is_nvidia_on_nvidia_hw() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        // This may or may not be true depending on HW; just verify no panic.
        let _ = dev.is_nvidia();
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_supports_subgroup_query() {
        let config =
            WgpuDeviceConfig { required_features: wgpu::Features::SUBGROUP, ..Default::default() };
        // This may fail if the adapter doesn't support subgroups — that's OK.
        if let Ok(dev) = WgpuDevice::new_blocking(&config) {
            assert!(dev.supports_subgroup_ops());
        }
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_debug_output() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        let dbg = format!("{dev:?}");
        assert!(dbg.contains("WgpuDevice"), "debug output malformed");
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_accessors() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        // Just verify accessors don't panic.
        let _ = dev.device();
        let _ = dev.queue();
        let _ = dev.adapter();
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn device_limits_populated() {
        let dev = WgpuDevice::new_blocking(&WgpuDeviceConfig::default()).unwrap();
        let info = dev.info();
        assert!(info.limits.max_compute_workgroup_size_x > 0, "max workgroup size x should be > 0");
    }

    // ── Non-GPU tests ────────────────────────────────────────────────

    #[test]
    fn default_config_is_vulkan_high_perf() {
        let cfg = WgpuDeviceConfig::default();
        assert_eq!(cfg.power_preference, wgpu::PowerPreference::HighPerformance);
        assert!(cfg.backend_bits.contains(wgpu::Backends::VULKAN));
    }

    #[test]
    fn device_info_is_clone_and_debug() {
        fn assert_clone_debug<T: Clone + std::fmt::Debug>() {}
        assert_clone_debug::<DeviceInfo>();
    }

    #[test]
    fn config_is_clone_and_debug() {
        fn assert_clone_debug<T: Clone + std::fmt::Debug>() {}
        assert_clone_debug::<WgpuDeviceConfig>();
    }
}
