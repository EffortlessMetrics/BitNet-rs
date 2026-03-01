//! wgpu-based GPU device probing for Vulkan/Metal/DX12 backends.
//!
//! Provides adapter enumeration and capability extraction via the `wgpu` crate,
//! complementing the raw Vulkan probing done via `ash` in the `vulkan` feature.

use std::fmt;

/// Compute-relevant limits extracted from a wgpu adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuLimits {
    /// Maximum size of a single GPU buffer in bytes.
    pub max_buffer_size: u64,
    /// Maximum number of storage buffers per shader stage.
    pub max_storage_buffers: u32,
    /// Maximum compute workgroup size in the X dimension.
    pub max_compute_workgroup_size_x: u32,
    /// Maximum compute workgroup size in the Y dimension.
    pub max_compute_workgroup_size_y: u32,
    /// Maximum compute workgroup size in the Z dimension.
    pub max_compute_workgroup_size_z: u32,
    /// Maximum total invocations per compute workgroup.
    pub max_compute_invocations: u32,
    /// Maximum number of bind groups.
    pub max_bind_groups: u32,
    /// Minimum subgroup size (0 if unavailable).
    pub subgroup_size: u32,
}

impl Default for WgpuLimits {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024 * 1024, // 256 MiB conservative default
            max_storage_buffers: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations: 256,
            max_bind_groups: 4,
            subgroup_size: 0,
        }
    }
}

/// Device type reported by wgpu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuDeviceType {
    /// Discrete GPU (dedicated graphics card).
    DiscreteGpu,
    /// Integrated GPU (shares memory with CPU).
    IntegratedGpu,
    /// Software/CPU renderer.
    Cpu,
    /// Virtual GPU (e.g. in a VM).
    VirtualGpu,
    /// Unknown or unrecognised device type.
    Other,
}

impl fmt::Display for WgpuDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DiscreteGpu => write!(f, "DiscreteGpu"),
            Self::IntegratedGpu => write!(f, "IntegratedGpu"),
            Self::Cpu => write!(f, "Cpu"),
            Self::VirtualGpu => write!(f, "VirtualGpu"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// GPU backend used by the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
    Other,
}

impl fmt::Display for WgpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Dx12 => write!(f, "DX12"),
            Self::Gl => write!(f, "GL"),
            Self::BrowserWebGpu => write!(f, "BrowserWebGpu"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Information about a single wgpu-discovered GPU adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuDeviceInfo {
    /// Human-readable adapter name.
    pub name: String,
    /// Vendor identifier (PCI vendor ID).
    pub vendor: u32,
    /// Backend API used by this adapter.
    pub backend: WgpuBackend,
    /// Device type classification.
    pub device_type: WgpuDeviceType,
    /// Driver name (may be empty on some platforms).
    pub driver: String,
    /// Additional driver information string.
    pub driver_info: String,
    /// Compute-relevant limits.
    pub limits: WgpuLimits,
    /// Whether the adapter reports `shader-f16` feature support.
    pub shader_f16: bool,
}

/// NVIDIA PCI vendor ID.
const NVIDIA_VENDOR_ID: u32 = 0x10DE;

/// Check whether a probed device is an NVIDIA GPU.
pub fn is_nvidia(info: &WgpuDeviceInfo) -> bool {
    info.vendor == NVIDIA_VENDOR_ID
        || info.name.to_ascii_lowercase().contains("nvidia")
        || info.driver.to_ascii_lowercase().contains("nvidia")
}

/// Check whether a probed device uses the Vulkan backend.
pub fn is_vulkan_backend(info: &WgpuDeviceInfo) -> bool {
    info.backend == WgpuBackend::Vulkan
}

/// Check whether a probed device advertises `shader-f16` support.
pub fn supports_f16(info: &WgpuDeviceInfo) -> bool {
    info.shader_f16
}

// ── wgpu backend conversion helpers ──────────────────────────────────────────

fn convert_backend(b: wgpu::Backend) -> WgpuBackend {
    match b {
        wgpu::Backend::Vulkan => WgpuBackend::Vulkan,
        wgpu::Backend::Metal => WgpuBackend::Metal,
        wgpu::Backend::Dx12 => WgpuBackend::Dx12,
        wgpu::Backend::Gl => WgpuBackend::Gl,
        wgpu::Backend::BrowserWebGpu => WgpuBackend::BrowserWebGpu,
        _ => WgpuBackend::Other,
    }
}

fn convert_device_type(dt: wgpu::DeviceType) -> WgpuDeviceType {
    match dt {
        wgpu::DeviceType::DiscreteGpu => WgpuDeviceType::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu => WgpuDeviceType::IntegratedGpu,
        wgpu::DeviceType::Cpu => WgpuDeviceType::Cpu,
        wgpu::DeviceType::VirtualGpu => WgpuDeviceType::VirtualGpu,
        _ => WgpuDeviceType::Other,
    }
}

fn adapter_to_info(adapter: &wgpu::Adapter) -> WgpuDeviceInfo {
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();

    WgpuDeviceInfo {
        name: info.name.clone(),
        vendor: info.vendor,
        backend: convert_backend(info.backend),
        device_type: convert_device_type(info.device_type),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
        limits: WgpuLimits {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_invocations: limits.max_compute_invocations_per_workgroup,
            max_bind_groups: limits.max_bind_groups,
            subgroup_size: if features.contains(wgpu::Features::SUBGROUP) {
                limits.min_subgroup_size
            } else {
                0
            },
        },
        shader_f16: features.contains(wgpu::Features::SHADER_F16),
    }
}

/// Enumerate all wgpu adapters and return their device info.
///
/// Returns an empty `Vec` if no adapters are found or wgpu initialisation fails.
pub fn probe_wgpu_devices() -> Vec<WgpuDeviceInfo> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        adapters.iter().map(|a| adapter_to_info(a)).collect()
    })
}

/// Pick the highest-performance wgpu adapter.
///
/// Preference order: discrete GPU > integrated GPU > virtual GPU > CPU > other.
/// Within the same device type, Vulkan is preferred over other backends.
pub fn probe_best_wgpu_device() -> Option<WgpuDeviceInfo> {
    let mut devices = probe_wgpu_devices();
    if devices.is_empty() {
        return None;
    }

    devices.sort_by(|a, b| {
        fn type_rank(dt: &WgpuDeviceType) -> u32 {
            match dt {
                WgpuDeviceType::DiscreteGpu => 4,
                WgpuDeviceType::IntegratedGpu => 3,
                WgpuDeviceType::VirtualGpu => 2,
                WgpuDeviceType::Cpu => 1,
                WgpuDeviceType::Other => 0,
            }
        }
        fn backend_rank(b: &WgpuBackend) -> u32 {
            match b {
                WgpuBackend::Vulkan => 3,
                WgpuBackend::Metal => 2,
                WgpuBackend::Dx12 => 2,
                WgpuBackend::Gl => 1,
                _ => 0,
            }
        }
        let cmp = type_rank(&b.device_type).cmp(&type_rank(&a.device_type));
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
        backend_rank(&b.backend).cmp(&backend_rank(&a.backend))
    });

    devices.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests (no GPU required) ─────────────────────────────────────

    fn make_info(overrides: impl FnOnce(&mut WgpuDeviceInfo)) -> WgpuDeviceInfo {
        let mut info = WgpuDeviceInfo {
            name: "Test GPU".to_string(),
            vendor: 0,
            backend: WgpuBackend::Vulkan,
            device_type: WgpuDeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            limits: WgpuLimits::default(),
            shader_f16: false,
        };
        overrides(&mut info);
        info
    }

    #[test]
    fn wgpu_device_info_construction() {
        let info = make_info(|i| i.name = "RTX 4090".to_string());
        assert_eq!(info.name, "RTX 4090");
        assert_eq!(info.device_type, WgpuDeviceType::DiscreteGpu);
    }

    #[test]
    fn wgpu_limits_default_values() {
        let limits = WgpuLimits::default();
        assert_eq!(limits.max_buffer_size, 256 * 1024 * 1024);
        assert_eq!(limits.max_storage_buffers, 8);
        assert_eq!(limits.max_compute_workgroup_size_x, 256);
        assert_eq!(limits.max_compute_workgroup_size_y, 256);
        assert_eq!(limits.max_compute_workgroup_size_z, 64);
        assert_eq!(limits.max_compute_invocations, 256);
        assert_eq!(limits.max_bind_groups, 4);
        assert_eq!(limits.subgroup_size, 0);
    }

    #[test]
    fn is_nvidia_by_vendor_id() {
        let info = make_info(|i| i.vendor = NVIDIA_VENDOR_ID);
        assert!(is_nvidia(&info));
    }

    #[test]
    fn is_nvidia_by_name() {
        let info = make_info(|i| i.name = "NVIDIA GeForce RTX 3090".to_string());
        assert!(is_nvidia(&info));
    }

    #[test]
    fn is_nvidia_by_driver() {
        let info = make_info(|i| i.driver = "nvidia proprietary".to_string());
        assert!(is_nvidia(&info));
    }

    #[test]
    fn is_not_nvidia_for_amd() {
        let info = make_info(|i| {
            i.vendor = 0x1002; // AMD
            i.name = "AMD Radeon RX 7900".to_string();
            i.driver = "radv".to_string();
        });
        assert!(!is_nvidia(&info));
    }

    #[test]
    fn is_vulkan_backend_true() {
        let info = make_info(|i| i.backend = WgpuBackend::Vulkan);
        assert!(is_vulkan_backend(&info));
    }

    #[test]
    fn is_vulkan_backend_false_for_metal() {
        let info = make_info(|i| i.backend = WgpuBackend::Metal);
        assert!(!is_vulkan_backend(&info));
    }

    #[test]
    fn supports_f16_true() {
        let info = make_info(|i| i.shader_f16 = true);
        assert!(supports_f16(&info));
    }

    #[test]
    fn supports_f16_false() {
        let info = make_info(|_| {});
        assert!(!supports_f16(&info));
    }

    #[test]
    fn device_type_display() {
        assert_eq!(WgpuDeviceType::DiscreteGpu.to_string(), "DiscreteGpu");
        assert_eq!(WgpuDeviceType::IntegratedGpu.to_string(), "IntegratedGpu");
        assert_eq!(WgpuDeviceType::Cpu.to_string(), "Cpu");
        assert_eq!(WgpuDeviceType::VirtualGpu.to_string(), "VirtualGpu");
        assert_eq!(WgpuDeviceType::Other.to_string(), "Other");
    }

    #[test]
    fn backend_display() {
        assert_eq!(WgpuBackend::Vulkan.to_string(), "Vulkan");
        assert_eq!(WgpuBackend::Metal.to_string(), "Metal");
        assert_eq!(WgpuBackend::Dx12.to_string(), "DX12");
        assert_eq!(WgpuBackend::Gl.to_string(), "GL");
        assert_eq!(WgpuBackend::BrowserWebGpu.to_string(), "BrowserWebGpu");
        assert_eq!(WgpuBackend::Other.to_string(), "Other");
    }

    // ── GPU-requiring tests ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires GPU runtime with wgpu-probe feature"]
    fn probe_wgpu_devices_returns_adapters() {
        let devices = probe_wgpu_devices();
        assert!(!devices.is_empty(), "expected at least one wgpu adapter");
        for d in &devices {
            assert!(!d.name.is_empty(), "adapter name must not be empty");
        }
    }

    #[test]
    #[ignore = "requires GPU runtime with wgpu-probe feature"]
    fn probe_best_wgpu_device_returns_some() {
        let best = probe_best_wgpu_device();
        assert!(best.is_some(), "expected at least one wgpu adapter");
    }

    #[test]
    #[ignore = "requires GPU runtime with wgpu-probe feature"]
    fn probe_best_prefers_discrete() {
        let devices = probe_wgpu_devices();
        if devices.len() < 2 {
            return; // can't test preference with < 2 adapters
        }
        let best = probe_best_wgpu_device().unwrap();
        let has_discrete = devices.iter().any(|d| d.device_type == WgpuDeviceType::DiscreteGpu);
        if has_discrete {
            assert_eq!(best.device_type, WgpuDeviceType::DiscreteGpu);
        }
    }

    #[test]
    #[ignore = "requires GPU runtime with wgpu-probe feature"]
    fn probe_devices_have_valid_limits() {
        let devices = probe_wgpu_devices();
        for d in &devices {
            assert!(d.limits.max_buffer_size > 0);
            assert!(d.limits.max_compute_workgroup_size_x > 0);
            assert!(d.limits.max_compute_invocations > 0);
        }
    }
}
