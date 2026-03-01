//! Metal device discovery and Apple Silicon capability detection.
//!
//! Provides mock/stub device probing for Apple Silicon GPUs until real
//! Metal bindings are integrated.  All public types and functions are
//! gated behind `#[cfg(target_os = "macos")]` at the module level in
//! `lib.rs`.

use std::fmt;

// ── GPU family identifiers ──────────────────────────────────────────────────

/// Apple GPU family identifier used for feature-level capability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AppleGpuFamily {
    /// Apple7 — M1, A14 (2020).
    Apple7,
    /// Apple8 — M2, A15/A16 (2022).
    Apple8,
    /// Apple9 — M3, A17 Pro (2023).
    Apple9,
}

impl fmt::Display for AppleGpuFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Apple7 => write!(f, "Apple7"),
            Self::Apple8 => write!(f, "Apple8"),
            Self::Apple9 => write!(f, "Apple9"),
        }
    }
}

// ── Metal device info ───────────────────────────────────────────────────────

/// Information about a single Metal GPU device.
///
/// Field values mirror the Metal API's `MTLDevice` properties.  In stub
/// mode the values are representative of real Apple Silicon hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Human-readable device name (e.g. "Apple M1").
    pub device_name: String,
    /// Unique registry identifier for the GPU.
    pub registry_id: u64,
    /// Whether the GPU shares memory with the CPU (always `true` on
    /// Apple Silicon).
    pub has_unified_memory: bool,
    /// Maximum buffer allocation size in bytes.
    pub max_buffer_length: u64,
    /// Maximum threads per threadgroup (compute).
    pub max_threads_per_threadgroup: u32,
    /// Recommended working-set size in bytes.
    pub recommended_working_set_size: u64,
    /// GPU families this device supports.
    pub supports_family: Vec<AppleGpuFamily>,
}

impl fmt::Display for MetalDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (registry_id={}, unified={}, max_buf={})",
            self.device_name, self.registry_id, self.has_unified_memory, self.max_buffer_length,
        )
    }
}

// ── Metal capabilities ──────────────────────────────────────────────────────

/// Compute-relevant Metal shader and hardware capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalCapabilities {
    /// Metal Shading Language version (major, minor) — e.g. `(3, 1)`.
    pub shader_language_version: (u32, u32),
    /// Maximum threadgroup memory in bytes.
    pub max_total_threadgroup_memory: u32,
    /// SIMD-group matrix multiply/accumulate support (Apple7+).
    pub supports_simdgroup_matrix: bool,
    /// `BFloat16` support (Apple8+).
    pub supports_bfloat16: bool,
    /// Highest Apple GPU family supported by this device.
    pub apple_gpu_family: AppleGpuFamily,
}

impl fmt::Display for MetalCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MSL {}.{}, family={}, simdgroup_matrix={}, bf16={}",
            self.shader_language_version.0,
            self.shader_language_version.1,
            self.apple_gpu_family,
            self.supports_simdgroup_matrix,
            self.supports_bfloat16,
        )
    }
}

// ── Detection helpers ───────────────────────────────────────────────────────

/// Detect whether the current process is running on Apple Silicon.
///
/// Returns `true` on `aarch64` macOS (all M-series chips).  On
/// non-`aarch64` targets this always returns `false`.
///
/// # Examples
///
/// ```
/// # #[cfg(target_os = "macos")]
/// # {
/// use bitnet_device_probe::metal::detect_apple_silicon;
/// let is_as = detect_apple_silicon();
/// // On Apple Silicon hardware this is true.
/// println!("Apple Silicon: {is_as}");
/// # }
/// ```
pub const fn detect_apple_silicon() -> bool {
    cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")
}

// ── Stub device probing ─────────────────────────────────────────────────────

/// Probe for Metal GPU devices.
///
/// **Current implementation**: returns a single hardcoded entry
/// representing an Apple M1 GPU.  This will be replaced with real Metal
/// API calls once `metal-rs` or `wgpu` bindings are integrated.
///
/// Returns an empty `Vec` when not running on Apple Silicon.
///
/// # Examples
///
/// ```
/// # #[cfg(target_os = "macos")]
/// # {
/// use bitnet_device_probe::metal::probe_metal_devices;
/// let devices = probe_metal_devices();
/// // On macOS we always return at least the stub device.
/// assert!(!devices.is_empty());
/// # }
/// ```
pub fn probe_metal_devices() -> Vec<MetalDeviceInfo> {
    if !detect_apple_silicon() {
        return Vec::new();
    }

    vec![MetalDeviceInfo {
        device_name: "Apple M1".to_owned(),
        registry_id: 0x0001_0000_0000_0001,
        has_unified_memory: true,
        // 16 GiB — base M1 default
        max_buffer_length: 16 * 1024 * 1024 * 1024,
        max_threads_per_threadgroup: 1024,
        // 16 GiB recommended working set
        recommended_working_set_size: 16 * 1024 * 1024 * 1024,
        supports_family: vec![AppleGpuFamily::Apple7],
    }]
}

/// Probe Metal capabilities for the default device.
///
/// **Current implementation**: returns hardcoded M1-class capabilities.
/// Returns `None` when not running on Apple Silicon.
pub const fn probe_metal_capabilities() -> Option<MetalCapabilities> {
    if !detect_apple_silicon() {
        return None;
    }

    Some(MetalCapabilities {
        shader_language_version: (3, 0),
        max_total_threadgroup_memory: 32_768,
        supports_simdgroup_matrix: true,
        supports_bfloat16: false,
        apple_gpu_family: AppleGpuFamily::Apple7,
    })
}

/// Build a [`MetalDeviceInfo`] with M2-class values (stub helper).
///
/// Useful for tests that need a second device profile.
pub fn stub_m2_device() -> MetalDeviceInfo {
    MetalDeviceInfo {
        device_name: "Apple M2".to_owned(),
        registry_id: 0x0001_0000_0000_0002,
        has_unified_memory: true,
        // 24 GiB
        max_buffer_length: 24 * 1024 * 1024 * 1024,
        max_threads_per_threadgroup: 1024,
        recommended_working_set_size: 24 * 1024 * 1024 * 1024,
        supports_family: vec![AppleGpuFamily::Apple7, AppleGpuFamily::Apple8],
    }
}

/// Build a [`MetalDeviceInfo`] with M3-class values (stub helper).
pub fn stub_m3_device() -> MetalDeviceInfo {
    MetalDeviceInfo {
        device_name: "Apple M3".to_owned(),
        registry_id: 0x0001_0000_0000_0003,
        has_unified_memory: true,
        // 36 GiB
        max_buffer_length: 36 * 1024 * 1024 * 1024,
        max_threads_per_threadgroup: 1024,
        recommended_working_set_size: 36 * 1024 * 1024 * 1024,
        supports_family: vec![
            AppleGpuFamily::Apple7,
            AppleGpuFamily::Apple8,
            AppleGpuFamily::Apple9,
        ],
    }
}

/// Build [`MetalCapabilities`] for an M2-class device.
pub const fn stub_m2_capabilities() -> MetalCapabilities {
    MetalCapabilities {
        shader_language_version: (3, 1),
        max_total_threadgroup_memory: 32_768,
        supports_simdgroup_matrix: true,
        supports_bfloat16: true,
        apple_gpu_family: AppleGpuFamily::Apple8,
    }
}

/// Build [`MetalCapabilities`] for an M3-class device.
pub const fn stub_m3_capabilities() -> MetalCapabilities {
    MetalCapabilities {
        shader_language_version: (3, 1),
        max_total_threadgroup_memory: 32_768,
        supports_simdgroup_matrix: true,
        supports_bfloat16: true,
        apple_gpu_family: AppleGpuFamily::Apple9,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- detect_apple_silicon -------------------------------------------------

    #[test]
    fn detect_apple_silicon_returns_bool() {
        let result = detect_apple_silicon();
        // On aarch64-macos this is true; on all other targets false.
        if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            assert!(result);
        } else {
            assert!(!result);
        }
    }

    #[test]
    fn detect_apple_silicon_is_deterministic() {
        assert_eq!(detect_apple_silicon(), detect_apple_silicon());
    }

    // -- probe_metal_devices --------------------------------------------------

    #[test]
    fn probe_metal_devices_returns_vec() {
        let devices = probe_metal_devices();
        if detect_apple_silicon() {
            assert!(!devices.is_empty());
        } else {
            assert!(devices.is_empty());
        }
    }

    #[test]
    fn probe_metal_devices_stub_has_valid_name() {
        if !detect_apple_silicon() {
            return;
        }
        let devices = probe_metal_devices();
        assert!(devices[0].device_name.starts_with("Apple"), "device name should start with Apple");
    }

    #[test]
    fn probe_metal_devices_stub_has_unified_memory() {
        if !detect_apple_silicon() {
            return;
        }
        let devices = probe_metal_devices();
        assert!(devices[0].has_unified_memory, "Apple Silicon always has unified memory");
    }

    #[test]
    fn probe_metal_devices_stub_has_nonzero_buffer_length() {
        if !detect_apple_silicon() {
            return;
        }
        let devices = probe_metal_devices();
        assert!(devices[0].max_buffer_length > 0);
    }

    #[test]
    fn probe_metal_devices_stub_has_reasonable_threadgroup() {
        if !detect_apple_silicon() {
            return;
        }
        let devices = probe_metal_devices();
        assert!(devices[0].max_threads_per_threadgroup >= 256);
        assert!(devices[0].max_threads_per_threadgroup <= 2048);
    }

    // -- probe_metal_capabilities ---------------------------------------------

    #[test]
    fn probe_metal_capabilities_matches_silicon() {
        let caps = probe_metal_capabilities();
        if detect_apple_silicon() {
            assert!(caps.is_some());
        } else {
            assert!(caps.is_none());
        }
    }

    #[test]
    fn probe_metal_capabilities_m1_has_simdgroup_matrix() {
        if !detect_apple_silicon() {
            return;
        }
        let caps = probe_metal_capabilities().unwrap();
        assert!(caps.supports_simdgroup_matrix);
    }

    #[test]
    fn probe_metal_capabilities_m1_no_bfloat16() {
        if !detect_apple_silicon() {
            return;
        }
        let caps = probe_metal_capabilities().unwrap();
        assert!(!caps.supports_bfloat16, "M1 stub should not advertise bf16");
    }

    // -- MetalDeviceInfo Display -----------------------------------------------

    #[test]
    fn metal_device_info_display() {
        let dev = stub_m2_device();
        let s = format!("{dev}");
        assert!(s.contains("Apple M2"));
        assert!(s.contains("unified=true"));
    }

    // -- MetalCapabilities Display --------------------------------------------

    #[test]
    fn metal_capabilities_display() {
        let caps = stub_m3_capabilities();
        let s = format!("{caps}");
        assert!(s.contains("MSL 3.1"));
        assert!(s.contains("Apple9"));
        assert!(s.contains("bf16=true"));
    }

    // -- AppleGpuFamily Display -----------------------------------------------

    #[test]
    fn apple_gpu_family_display() {
        assert_eq!(AppleGpuFamily::Apple7.to_string(), "Apple7");
        assert_eq!(AppleGpuFamily::Apple8.to_string(), "Apple8");
        assert_eq!(AppleGpuFamily::Apple9.to_string(), "Apple9");
    }

    // -- Stub helpers ---------------------------------------------------------

    #[test]
    fn stub_m2_device_has_apple8_family() {
        let dev = stub_m2_device();
        assert!(dev.supports_family.contains(&AppleGpuFamily::Apple8));
        assert!(dev.supports_family.contains(&AppleGpuFamily::Apple7));
    }

    #[test]
    fn stub_m3_device_has_apple9_family() {
        let dev = stub_m3_device();
        assert!(dev.supports_family.contains(&AppleGpuFamily::Apple9));
        assert!(dev.supports_family.contains(&AppleGpuFamily::Apple8));
        assert!(dev.supports_family.contains(&AppleGpuFamily::Apple7));
    }

    #[test]
    fn stub_m2_capabilities_has_bfloat16() {
        let caps = stub_m2_capabilities();
        assert!(caps.supports_bfloat16);
        assert_eq!(caps.apple_gpu_family, AppleGpuFamily::Apple8);
    }

    #[test]
    fn stub_m3_capabilities_has_bfloat16() {
        let caps = stub_m3_capabilities();
        assert!(caps.supports_bfloat16);
        assert_eq!(caps.apple_gpu_family, AppleGpuFamily::Apple9);
    }

    #[test]
    fn stub_devices_have_distinct_registry_ids() {
        let m2 = stub_m2_device();
        let m3 = stub_m3_device();
        assert_ne!(m2.registry_id, m3.registry_id);
    }

    #[test]
    fn stub_m3_has_larger_buffer_than_m2() {
        let m2 = stub_m2_device();
        let m3 = stub_m3_device();
        assert!(m3.max_buffer_length > m2.max_buffer_length);
    }
}
