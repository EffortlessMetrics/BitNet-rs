//! `OpenCL` runtime binding layer.
//!
//! Provides safe wrappers around opencl3 types with
//! dynamic library loading and graceful fallback.

use std::fmt;

/// Check if `OpenCL` runtime is available (dynamically loaded).
///
/// Returns `false` if `libOpenCL.so` / `OpenCL.dll` is not found or
/// no platforms are reported by the ICD loader.
pub fn opencl_available() -> bool {
    cfg!(feature = "opencl-runtime") && check_opencl_icd()
}

fn check_opencl_icd() -> bool {
    std::panic::catch_unwind(|| {
        #[cfg(feature = "opencl-runtime")]
        {
            opencl3::platform::get_platforms().map(|p| !p.is_empty()).unwrap_or(false)
        }
        #[cfg(not(feature = "opencl-runtime"))]
        false
    })
    .unwrap_or(false)
}

// ── Platform info ──────────────────────────────────────────────────

/// `OpenCL` platform information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClPlatformInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub profile: String,
    pub extensions: Vec<String>,
}

impl OpenClPlatformInfo {
    /// Number of advertised extensions.
    pub const fn extension_count(&self) -> usize {
        self.extensions.len()
    }

    /// Whether a specific extension is present (case-insensitive).
    pub fn has_extension(&self, ext: &str) -> bool {
        self.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
    }
}

impl fmt::Display for OpenClPlatformInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) [{}] – {} extension(s)",
            self.name,
            self.vendor,
            self.version,
            self.extension_count()
        )
    }
}

// ── Device info ────────────────────────────────────────────────────

/// `OpenCL` device information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub max_compute_units: u32,
    pub max_work_group_size: usize,
    pub max_clock_frequency: u32,
    pub global_memory: u64,
    pub local_memory: u64,
    pub max_mem_alloc_size: u64,
    pub supports_fp16: bool,
    pub supports_fp64: bool,
    pub driver_version: String,
    pub opencl_version: String,
}

impl OpenClDeviceInfo {
    /// Global memory in MiB.
    pub const fn global_memory_mib(&self) -> u64 {
        self.global_memory / (1024 * 1024)
    }

    /// Local memory in KiB.
    pub const fn local_memory_kib(&self) -> u64 {
        self.local_memory / 1024
    }

    /// Maximum single allocation in MiB.
    pub const fn max_alloc_mib(&self) -> u64 {
        self.max_mem_alloc_size / (1024 * 1024)
    }

    /// Whether the device is a GPU.
    pub fn is_gpu(&self) -> bool {
        self.device_type.eq_ignore_ascii_case("gpu")
    }

    /// Whether the device is a CPU.
    pub fn is_cpu(&self) -> bool {
        self.device_type.eq_ignore_ascii_case("cpu")
    }

    /// Heuristic GFLOPS estimate (single-precision, CUs × clock × 2).
    pub fn estimated_gflops(&self) -> f64 {
        f64::from(self.max_compute_units) * f64::from(self.max_clock_frequency) * 2.0 / 1000.0
    }
}

impl fmt::Display for OpenClDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] – {} CU, {} MiB VRAM ({})",
            self.name,
            self.device_type,
            self.max_compute_units,
            self.global_memory_mib(),
            self.opencl_version,
        )
    }
}

// ── Real enumeration (behind feature gate) ─────────────────────────

#[cfg(feature = "opencl-runtime")]
fn real_enumerate_platforms() -> Vec<OpenClPlatformInfo> {
    use opencl3::platform::get_platforms;

    let Ok(platforms) = get_platforms() else {
        return Vec::new();
    };

    platforms
        .into_iter()
        .filter_map(|p| {
            let name = p.name().ok()?;
            let vendor = p.vendor().ok()?;
            let version = p.version().ok()?;
            let profile = p.profile().ok()?;
            let ext_str = p.extensions().ok().unwrap_or_default();
            let extensions: Vec<String> = ext_str.split_whitespace().map(String::from).collect();
            Some(OpenClPlatformInfo { name, vendor, version, profile, extensions })
        })
        .collect()
}

#[cfg(feature = "opencl-runtime")]
fn real_enumerate_devices(platform_index: usize) -> Vec<OpenClDeviceInfo> {
    use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
    use opencl3::platform::get_platforms;

    let Ok(platforms) = get_platforms() else {
        return Vec::new();
    };
    let Some(platform) = platforms.get(platform_index) else {
        return Vec::new();
    };

    let Ok(device_ids) = get_all_devices(platform.id(), CL_DEVICE_TYPE_ALL) else {
        return Vec::new();
    };

    device_ids
        .into_iter()
        .filter_map(|id| {
            let dev = Device::new(id);
            let name = dev.name().ok()?;
            let vendor = dev.vendor().ok()?;
            let dev_type_bits = dev.dev_type().ok()?;
            let device_type = device_type_string(dev_type_bits);
            let max_compute_units = dev.max_compute_units().ok()? as u32;
            let max_work_group_size = dev.max_work_group_size().ok()?;
            let max_clock_frequency = dev.max_clock_frequency().ok()? as u32;
            let global_memory = dev.global_mem_size().ok()?;
            let local_memory = dev.local_mem_size().ok()?;
            let max_mem_alloc_size = dev.max_mem_alloc_size().ok()?;
            let ext_str = dev.extensions().ok().unwrap_or_default();
            let supports_fp16 = ext_str.contains("cl_khr_fp16");
            let supports_fp64 = ext_str.contains("cl_khr_fp64");
            let driver_version = dev.driver_version().ok().unwrap_or_default();
            let opencl_version = dev.version().ok().unwrap_or_default();

            Some(OpenClDeviceInfo {
                name,
                vendor,
                device_type,
                max_compute_units,
                max_work_group_size,
                max_clock_frequency,
                global_memory,
                local_memory,
                max_mem_alloc_size,
                supports_fp16,
                supports_fp64,
                driver_version,
                opencl_version,
            })
        })
        .collect()
}

#[cfg(feature = "opencl-runtime")]
fn device_type_string(bits: u64) -> String {
    use opencl3::device::{CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
    if bits & CL_DEVICE_TYPE_GPU != 0 {
        "GPU".to_string()
    } else if bits & CL_DEVICE_TYPE_CPU != 0 {
        "CPU".to_string()
    } else if bits & CL_DEVICE_TYPE_ACCELERATOR != 0 {
        "Accelerator".to_string()
    } else {
        format!("Unknown(0x{bits:x})")
    }
}

// ── Public enumeration API ─────────────────────────────────────────

/// Enumerate available `OpenCL` platforms.
///
/// Returns an empty list when the `opencl-runtime` feature is
/// disabled or when no ICD loader is found.
#[allow(clippy::missing_const_for_fn)]
pub fn enumerate_platforms() -> Vec<OpenClPlatformInfo> {
    #[cfg(feature = "opencl-runtime")]
    {
        real_enumerate_platforms()
    }
    #[cfg(not(feature = "opencl-runtime"))]
    {
        Vec::new()
    }
}

/// Enumerate devices on the given platform.
///
/// Returns an empty list when the `opencl-runtime` feature is
/// disabled or when the platform index is out of range.
#[allow(clippy::missing_const_for_fn)]
pub fn enumerate_gpu_devices(platform_index: usize) -> Vec<OpenClDeviceInfo> {
    #[cfg(feature = "opencl-runtime")]
    {
        real_enumerate_devices(platform_index)
    }
    #[cfg(not(feature = "opencl-runtime"))]
    {
        let _ = platform_index;
        Vec::new()
    }
}

// ── Mock helpers ───────────────────────────────────────────────────

/// Create a mock Intel `OpenCL` platform for testing.
pub fn mock_platform() -> OpenClPlatformInfo {
    OpenClPlatformInfo {
        name: "Intel(R) OpenCL Graphics".into(),
        vendor: "Intel(R) Corporation".into(),
        version: "OpenCL 3.0".into(),
        profile: "FULL_PROFILE".into(),
        extensions: vec![
            "cl_khr_fp16".into(),
            "cl_khr_subgroups".into(),
            "cl_intel_subgroups".into(),
            "cl_khr_global_int32_base_atomics".into(),
        ],
    }
}

/// Create a mock Intel Arc A770 device for testing.
pub fn mock_device_intel_arc() -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        name: "Intel(R) Arc(TM) A770 Graphics".into(),
        vendor: "Intel(R) Corporation".into(),
        device_type: "GPU".into(),
        max_compute_units: 512,
        max_work_group_size: 1024,
        max_clock_frequency: 2100,
        global_memory: 16 * 1024 * 1024 * 1024,     // 16 GiB
        local_memory: 64 * 1024,                    // 64 KiB
        max_mem_alloc_size: 4 * 1024 * 1024 * 1024, // 4 GiB
        supports_fp16: true,
        supports_fp64: false,
        driver_version: "24.17.31.10".into(),
        opencl_version: "OpenCL 3.0 NEO".into(),
    }
}

/// Create a mock NVIDIA RTX 4090 device for testing.
pub fn mock_device_nvidia() -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        name: "NVIDIA GeForce RTX 4090".into(),
        vendor: "NVIDIA Corporation".into(),
        device_type: "GPU".into(),
        max_compute_units: 128,
        max_work_group_size: 1024,
        max_clock_frequency: 2520,
        global_memory: 24 * 1024 * 1024 * 1024,     // 24 GiB
        local_memory: 48 * 1024,                    // 48 KiB
        max_mem_alloc_size: 6 * 1024 * 1024 * 1024, // 6 GiB
        supports_fp16: true,
        supports_fp64: true,
        driver_version: "560.35.03".into(),
        opencl_version: "OpenCL 3.0 CUDA".into(),
    }
}

/// Create a mock AMD Instinct MI250X device for testing.
pub fn mock_device_amd() -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        name: "AMD Instinct MI250X".into(),
        vendor: "Advanced Micro Devices, Inc.".into(),
        device_type: "GPU".into(),
        max_compute_units: 220,
        max_work_group_size: 256,
        max_clock_frequency: 1700,
        global_memory: 128 * 1024 * 1024 * 1024, // 128 GiB
        local_memory: 64 * 1024,                 // 64 KiB
        max_mem_alloc_size: 32 * 1024 * 1024 * 1024, // 32 GiB
        supports_fp16: true,
        supports_fp64: true,
        driver_version: "6.0.2".into(),
        opencl_version: "OpenCL 2.0".into(),
    }
}

/// Create a mock CPU-type `OpenCL` device for testing.
pub fn mock_device_cpu() -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        name: "Intel(R) Core(TM) i9-13900K".into(),
        vendor: "Intel(R) Corporation".into(),
        device_type: "CPU".into(),
        max_compute_units: 24,
        max_work_group_size: 8192,
        max_clock_frequency: 5800,
        global_memory: 64 * 1024 * 1024 * 1024,      // 64 GiB
        local_memory: 32 * 1024,                     // 32 KiB
        max_mem_alloc_size: 16 * 1024 * 1024 * 1024, // 16 GiB
        supports_fp16: false,
        supports_fp64: true,
        driver_version: "2024.18.6.0.02".into(),
        opencl_version: "OpenCL 3.0".into(),
    }
}

/// Create a mock NVIDIA platform for testing.
pub fn mock_platform_nvidia() -> OpenClPlatformInfo {
    OpenClPlatformInfo {
        name: "NVIDIA CUDA".into(),
        vendor: "NVIDIA Corporation".into(),
        version: "OpenCL 3.0 CUDA 12.4.131".into(),
        profile: "FULL_PROFILE".into(),
        extensions: vec![
            "cl_khr_fp64".into(),
            "cl_khr_global_int32_base_atomics".into(),
            "cl_nv_device_attribute_query".into(),
        ],
    }
}

/// Parse a whitespace-separated extension string.
pub fn parse_extensions(ext_str: &str) -> Vec<String> {
    ext_str.split_whitespace().map(String::from).collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- opencl_available --

    #[test]
    fn opencl_available_returns_false_without_runtime() {
        // Without the opencl-runtime feature, this must be false.
        assert!(!opencl_available());
    }

    #[test]
    fn check_opencl_icd_returns_false_without_runtime() {
        assert!(!check_opencl_icd());
    }

    // -- Platform info construction --

    #[test]
    fn platform_info_new() {
        let p = mock_platform();
        assert_eq!(p.name, "Intel(R) OpenCL Graphics");
        assert_eq!(p.vendor, "Intel(R) Corporation");
        assert_eq!(p.version, "OpenCL 3.0");
        assert_eq!(p.profile, "FULL_PROFILE");
    }

    #[test]
    fn platform_info_extension_count() {
        let p = mock_platform();
        assert_eq!(p.extension_count(), 4);
    }

    #[test]
    fn platform_info_has_extension_case_insensitive() {
        let p = mock_platform();
        assert!(p.has_extension("cl_khr_fp16"));
        assert!(p.has_extension("CL_KHR_FP16"));
        assert!(!p.has_extension("cl_khr_fp64"));
    }

    #[test]
    fn platform_info_display() {
        let p = mock_platform();
        let s = p.to_string();
        assert!(s.contains("Intel(R) OpenCL Graphics"));
        assert!(s.contains("4 extension(s)"));
    }

    #[test]
    fn platform_info_debug() {
        let p = mock_platform();
        let dbg = format!("{p:?}");
        assert!(dbg.contains("OpenClPlatformInfo"));
    }

    #[test]
    fn platform_info_clone_eq() {
        let p = mock_platform();
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn platform_empty_extensions() {
        let p = OpenClPlatformInfo {
            name: "Minimal".into(),
            vendor: "Test".into(),
            version: "1.0".into(),
            profile: "EMBEDDED_PROFILE".into(),
            extensions: vec![],
        };
        assert_eq!(p.extension_count(), 0);
        assert!(!p.has_extension("anything"));
    }

    // -- Device info construction --

    #[test]
    fn device_info_intel_arc() {
        let d = mock_device_intel_arc();
        assert_eq!(d.name, "Intel(R) Arc(TM) A770 Graphics");
        assert_eq!(d.max_compute_units, 512);
        assert!(d.supports_fp16);
        assert!(!d.supports_fp64);
    }

    #[test]
    fn device_info_nvidia() {
        let d = mock_device_nvidia();
        assert_eq!(d.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(d.max_compute_units, 128);
        assert!(d.supports_fp16);
        assert!(d.supports_fp64);
    }

    #[test]
    fn device_info_amd() {
        let d = mock_device_amd();
        assert_eq!(d.vendor, "Advanced Micro Devices, Inc.");
        assert_eq!(d.max_compute_units, 220);
    }

    #[test]
    fn device_info_cpu_type() {
        let d = mock_device_cpu();
        assert!(d.is_cpu());
        assert!(!d.is_gpu());
    }

    // -- Memory calculations --

    #[test]
    fn device_global_memory_mib() {
        let d = mock_device_intel_arc();
        assert_eq!(d.global_memory_mib(), 16 * 1024); // 16 GiB
    }

    #[test]
    fn device_local_memory_kib() {
        let d = mock_device_intel_arc();
        assert_eq!(d.local_memory_kib(), 64);
    }

    #[test]
    fn device_max_alloc_mib() {
        let d = mock_device_intel_arc();
        assert_eq!(d.max_alloc_mib(), 4 * 1024); // 4 GiB
    }

    #[test]
    fn device_nvidia_memory() {
        let d = mock_device_nvidia();
        assert_eq!(d.global_memory_mib(), 24 * 1024);
        assert_eq!(d.local_memory_kib(), 48);
        assert_eq!(d.max_alloc_mib(), 6 * 1024);
    }

    #[test]
    fn device_amd_memory() {
        let d = mock_device_amd();
        assert_eq!(d.global_memory_mib(), 128 * 1024);
    }

    #[test]
    fn device_zero_memory_edge_case() {
        let mut d = mock_device_intel_arc();
        d.global_memory = 0;
        d.local_memory = 0;
        d.max_mem_alloc_size = 0;
        assert_eq!(d.global_memory_mib(), 0);
        assert_eq!(d.local_memory_kib(), 0);
        assert_eq!(d.max_alloc_mib(), 0);
    }

    // -- Device type queries --

    #[test]
    fn device_is_gpu() {
        let d = mock_device_intel_arc();
        assert!(d.is_gpu());
        assert!(!d.is_cpu());
    }

    #[test]
    fn device_is_gpu_case_insensitive() {
        let mut d = mock_device_intel_arc();
        d.device_type = "gpu".into();
        assert!(d.is_gpu());
        d.device_type = "Gpu".into();
        assert!(d.is_gpu());
    }

    #[test]
    fn device_is_cpu_case_insensitive() {
        let mut d = mock_device_cpu();
        d.device_type = "cpu".into();
        assert!(d.is_cpu());
        d.device_type = "Cpu".into();
        assert!(d.is_cpu());
    }

    #[test]
    fn device_accelerator_is_neither_gpu_nor_cpu() {
        let mut d = mock_device_intel_arc();
        d.device_type = "Accelerator".into();
        assert!(!d.is_gpu());
        assert!(!d.is_cpu());
    }

    // -- GFLOPS estimate --

    #[test]
    fn device_estimated_gflops_intel_arc() {
        let d = mock_device_intel_arc();
        // 512 CU × 2100 MHz × 2 / 1000
        let expected = 512.0 * 2100.0 * 2.0 / 1000.0;
        let diff = (d.estimated_gflops() - expected).abs();
        assert!(diff < 0.01);
    }

    #[test]
    fn device_estimated_gflops_nvidia() {
        let d = mock_device_nvidia();
        let expected = 128.0 * 2520.0 * 2.0 / 1000.0;
        let diff = (d.estimated_gflops() - expected).abs();
        assert!(diff < 0.01);
    }

    #[test]
    fn device_estimated_gflops_zero_cu() {
        let mut d = mock_device_intel_arc();
        d.max_compute_units = 0;
        assert!((d.estimated_gflops()).abs() < f64::EPSILON);
    }

    // -- Display --

    #[test]
    fn device_display_contains_name_and_cu() {
        let d = mock_device_intel_arc();
        let s = d.to_string();
        assert!(s.contains("Arc"));
        assert!(s.contains("512 CU"));
    }

    #[test]
    fn device_display_contains_vram() {
        let d = mock_device_nvidia();
        let s = d.to_string();
        assert!(s.contains("24576 MiB VRAM"));
    }

    #[test]
    fn device_debug_format() {
        let d = mock_device_intel_arc();
        let dbg = format!("{d:?}");
        assert!(dbg.contains("OpenClDeviceInfo"));
    }

    #[test]
    fn device_clone_partial_eq() {
        let d = mock_device_nvidia();
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    // -- Feature-gated enumeration --

    #[test]
    fn enumerate_platforms_empty_without_runtime() {
        let platforms = enumerate_platforms();
        assert!(platforms.is_empty());
    }

    #[test]
    fn enumerate_gpu_devices_empty_without_runtime() {
        let devices = enumerate_gpu_devices(0);
        assert!(devices.is_empty());
    }

    #[test]
    fn enumerate_gpu_devices_out_of_range() {
        let devices = enumerate_gpu_devices(999);
        assert!(devices.is_empty());
    }

    // -- Extensions parsing --

    #[test]
    fn parse_extensions_single() {
        let exts = parse_extensions("cl_khr_fp16");
        assert_eq!(exts, vec!["cl_khr_fp16"]);
    }

    #[test]
    fn parse_extensions_multiple() {
        let exts = parse_extensions("cl_khr_fp16 cl_khr_fp64 cl_khr_subgroups");
        assert_eq!(exts.len(), 3);
        assert_eq!(exts[1], "cl_khr_fp64");
    }

    #[test]
    fn parse_extensions_empty() {
        let exts = parse_extensions("");
        assert!(exts.is_empty());
    }

    #[test]
    fn parse_extensions_whitespace_only() {
        let exts = parse_extensions("   \t\n  ");
        assert!(exts.is_empty());
    }

    #[test]
    fn parse_extensions_extra_whitespace() {
        let exts = parse_extensions("  cl_khr_fp16   cl_khr_fp64  ");
        assert_eq!(exts.len(), 2);
    }

    // -- Mock multi-platform / multi-device --

    #[test]
    fn mock_multi_platform() {
        let platforms = vec![mock_platform(), mock_platform_nvidia()];
        assert_eq!(platforms.len(), 2);
        assert!(platforms[0].name.contains("Intel"));
        assert!(platforms[1].name.contains("NVIDIA"));
    }

    #[test]
    fn mock_multi_device() {
        let devices = vec![mock_device_intel_arc(), mock_device_nvidia(), mock_device_amd()];
        assert_eq!(devices.len(), 3);
        assert!(devices[0].vendor.contains("Intel"));
        assert!(devices[1].vendor.contains("NVIDIA"));
        assert!(devices[2].vendor.contains("Micro Devices"));
    }

    #[test]
    fn mock_mixed_device_types() {
        let devices = vec![mock_device_intel_arc(), mock_device_cpu()];
        let gpus: Vec<_> = devices.iter().filter(|d| d.is_gpu()).collect();
        let cpus: Vec<_> = devices.iter().filter(|d| d.is_cpu()).collect();
        assert_eq!(gpus.len(), 1);
        assert_eq!(cpus.len(), 1);
    }

    // -- Nvidia platform specifics --

    #[test]
    fn nvidia_platform_has_fp64() {
        let p = mock_platform_nvidia();
        assert!(p.has_extension("cl_khr_fp64"));
    }

    #[test]
    fn nvidia_platform_display() {
        let p = mock_platform_nvidia();
        let s = p.to_string();
        assert!(s.contains("NVIDIA CUDA"));
        assert!(s.contains("3 extension(s)"));
    }

    // -- Edge cases --

    #[test]
    fn device_fp16_and_fp64_both_false() {
        let mut d = mock_device_intel_arc();
        d.supports_fp16 = false;
        d.supports_fp64 = false;
        assert!(!d.supports_fp16);
        assert!(!d.supports_fp64);
    }

    #[test]
    fn device_very_large_memory() {
        let mut d = mock_device_amd();
        d.global_memory = 256 * 1024 * 1024 * 1024; // 256 GiB
        assert_eq!(d.global_memory_mib(), 256 * 1024);
    }

    #[test]
    fn device_small_work_group_size() {
        let mut d = mock_device_intel_arc();
        d.max_work_group_size = 1;
        assert_eq!(d.max_work_group_size, 1);
    }
}
