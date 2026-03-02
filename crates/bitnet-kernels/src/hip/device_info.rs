//! ROCm/HIP device detection and capability querying.
//!
//! Provides structs and functions for enumerating AMD GPU devices,
//! querying their GCN/CDNA architecture, and determining feature
//! support (FP16, BF16, matrix cores).
//!
//! # Architecture taxonomy
//!
//! | Family | Example arch | Example GPU |
//! |--------|-------------|-------------|
//! | GCN 5  | gfx900      | Vega 56/64  |
//! | CDNA   | gfx908      | MI100       |
//! | CDNA2  | gfx90a      | MI210/MI250X |
//! | CDNA3  | gfx940/942  | MI300X      |
//! | RDNA 2 | gfx1030     | RX 6800 XT  |
//! | RDNA 3 | gfx1100     | RX 7900 XTX |
//!
//! All device queries return CPU-safe defaults when no HIP runtime is
//! available. GPU-dependent paths are gated behind `#[cfg(feature = "rocm")]`.

use std::fmt;

// ── Architecture family ──────────────────────────────────────────────

/// AMD GPU architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchFamily {
    /// Graphics Core Next generation 5 (Vega).
    Gcn5,
    /// Compute DNA generation 1 (MI100).
    Cdna,
    /// Compute DNA generation 2 (MI200 series).
    Cdna2,
    /// Compute DNA generation 3 (MI300 series).
    Cdna3,
    /// RDNA 2 consumer architecture (RX 6000 series).
    Rdna2,
    /// RDNA 3 consumer architecture (RX 7000 series).
    Rdna3,
    /// Unknown / unrecognised architecture.
    Unknown,
}

impl GpuArchFamily {
    /// Classify a GCN architecture string (e.g. `"gfx90a"`) into a family.
    pub fn from_gcn_arch(arch: &str) -> Self {
        match arch {
            a if a.starts_with("gfx900") => Self::Gcn5,
            a if a.starts_with("gfx906") => Self::Gcn5,
            a if a.starts_with("gfx908") => Self::Cdna,
            a if a.starts_with("gfx90a") => Self::Cdna2,
            a if a.starts_with("gfx940") || a.starts_with("gfx941") || a.starts_with("gfx942") => {
                Self::Cdna3
            }
            a if a.starts_with("gfx103") => Self::Rdna2,
            a if a.starts_with("gfx110") || a.starts_with("gfx115") => Self::Rdna3,
            _ => Self::Unknown,
        }
    }

    /// Whether this architecture supports hardware matrix-multiply-accumulate (MFMA).
    pub fn supports_mfma(&self) -> bool {
        matches!(self, Self::Cdna | Self::Cdna2 | Self::Cdna3)
    }

    /// Whether this architecture supports BF16 natively.
    pub fn supports_bf16(&self) -> bool {
        matches!(self, Self::Cdna2 | Self::Cdna3 | Self::Rdna3)
    }

    /// Wavefront width (number of lanes executed in lock-step).
    pub fn wavefront_size(&self) -> u32 {
        match self {
            Self::Rdna2 | Self::Rdna3 => 32,
            _ => 64,
        }
    }
}

impl fmt::Display for GpuArchFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gcn5 => write!(f, "GCN5"),
            Self::Cdna => write!(f, "CDNA"),
            Self::Cdna2 => write!(f, "CDNA2"),
            Self::Cdna3 => write!(f, "CDNA3"),
            Self::Rdna2 => write!(f, "RDNA2"),
            Self::Rdna3 => write!(f, "RDNA3"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── Device capabilities ──────────────────────────────────────────────

/// Detailed capability snapshot for a single AMD GPU device.
#[derive(Debug, Clone)]
pub struct HipDeviceCapabilities {
    /// Ordinal device index.
    pub device_id: usize,
    /// Marketing name (e.g. `"AMD Instinct MI250X"`).
    pub name: String,
    /// GCN architecture string (e.g. `"gfx90a"`).
    pub gcn_arch: String,
    /// Classified architecture family.
    pub arch_family: GpuArchFamily,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Number of compute units (CUs).
    pub compute_units: u32,
    /// Maximum work-group size (threads).
    pub max_workgroup_size: u32,
    /// Maximum LDS (shared memory) per work-group in bytes.
    pub max_lds_per_workgroup: usize,
    /// Maximum registers per work-group.
    pub max_registers_per_workgroup: u32,
    /// Wavefront (SIMD lane) width.
    pub wavefront_size: u32,
    /// FP16 (half-precision) arithmetic support.
    pub supports_fp16: bool,
    /// BF16 (bfloat16) arithmetic support.
    pub supports_bf16: bool,
    /// Hardware MFMA (matrix-multiply-accumulate) support.
    pub supports_mfma: bool,
    /// Peak FP32 throughput in TFLOPS (theoretical).
    pub peak_tflops_fp32: f64,
    /// Memory bandwidth in GB/s (theoretical).
    pub memory_bandwidth_gbps: f64,
}

impl Default for HipDeviceCapabilities {
    fn default() -> Self {
        Self {
            device_id: 0,
            name: "Unknown AMD GPU".into(),
            gcn_arch: "gfx000".into(),
            arch_family: GpuArchFamily::Unknown,
            total_memory: 0,
            compute_units: 0,
            max_workgroup_size: 1024,
            max_lds_per_workgroup: 65536,
            max_registers_per_workgroup: 256,
            wavefront_size: 64,
            supports_fp16: false,
            supports_bf16: false,
            supports_mfma: false,
            peak_tflops_fp32: 0.0,
            memory_bandwidth_gbps: 0.0,
        }
    }
}

impl HipDeviceCapabilities {
    /// Create a capability set from a known GCN arch string.
    pub fn from_gcn_arch(device_id: usize, name: &str, gcn_arch: &str) -> Self {
        let arch_family = GpuArchFamily::from_gcn_arch(gcn_arch);
        Self {
            device_id,
            name: name.to_string(),
            gcn_arch: gcn_arch.to_string(),
            arch_family,
            wavefront_size: arch_family.wavefront_size(),
            supports_bf16: arch_family.supports_bf16(),
            supports_mfma: arch_family.supports_mfma(),
            supports_fp16: true, // All modern AMD GPUs support FP16
            ..Default::default()
        }
    }

    /// Whether this device can run BitNet INT2 kernels efficiently.
    pub fn supports_bitnet_int2(&self) -> bool {
        self.compute_units > 0 && self.supports_fp16
    }
}

// ── Device enumeration (stubs) ───────────────────────────────────────

/// Enumerate all HIP-visible devices.
///
/// Stub — returns an empty vec until HIP runtime bindings are wired in.
pub fn enumerate_devices() -> Vec<HipDeviceCapabilities> {
    // TODO: call hipGetDeviceCount + hipGetDeviceProperties
    Vec::new()
}

/// Get capabilities for a specific device by ordinal index.
///
/// Stub — always returns `None`.
pub fn get_device(device_id: usize) -> Option<HipDeviceCapabilities> {
    let _ = device_id;
    // TODO: call hipGetDeviceProperties(device_id)
    None
}

/// Return the number of HIP-visible AMD GPU devices.
///
/// Stub — always returns `0`.
pub fn device_count() -> usize {
    // TODO: call hipGetDeviceCount
    0
}

/// Select the best device for BitNet inference based on capabilities.
pub fn select_best_device(devices: &[HipDeviceCapabilities]) -> Option<usize> {
    if devices.is_empty() {
        return None;
    }
    // Prefer CDNA with MFMA, then most CUs, then most memory.
    let mut best_idx = 0;
    let mut best_score: u64 = 0;
    for (i, dev) in devices.iter().enumerate() {
        let mut score: u64 = dev.compute_units as u64 * 1000;
        if dev.supports_mfma {
            score += 100_000;
        }
        score += (dev.total_memory / (1024 * 1024)) as u64; // MB bonus
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    Some(best_idx)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcn_arch_classification_gfx900() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx900"), GpuArchFamily::Gcn5);
    }

    #[test]
    fn gcn_arch_classification_gfx906() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx906"), GpuArchFamily::Gcn5);
    }

    #[test]
    fn gcn_arch_classification_cdna() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx908"), GpuArchFamily::Cdna);
    }

    #[test]
    fn gcn_arch_classification_cdna2() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx90a"), GpuArchFamily::Cdna2);
    }

    #[test]
    fn gcn_arch_classification_cdna3() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx940"), GpuArchFamily::Cdna3);
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx942"), GpuArchFamily::Cdna3);
    }

    #[test]
    fn gcn_arch_classification_rdna2() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx1030"), GpuArchFamily::Rdna2);
    }

    #[test]
    fn gcn_arch_classification_rdna3() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx1100"), GpuArchFamily::Rdna3);
    }

    #[test]
    fn gcn_arch_classification_unknown() {
        assert_eq!(GpuArchFamily::from_gcn_arch("gfx_what"), GpuArchFamily::Unknown);
    }

    #[test]
    fn cdna_supports_mfma() {
        assert!(GpuArchFamily::Cdna.supports_mfma());
        assert!(GpuArchFamily::Cdna2.supports_mfma());
        assert!(GpuArchFamily::Cdna3.supports_mfma());
    }

    #[test]
    fn consumer_gpus_no_mfma() {
        assert!(!GpuArchFamily::Gcn5.supports_mfma());
        assert!(!GpuArchFamily::Rdna2.supports_mfma());
        assert!(!GpuArchFamily::Rdna3.supports_mfma());
    }

    #[test]
    fn bf16_support_cdna2_plus() {
        assert!(!GpuArchFamily::Gcn5.supports_bf16());
        assert!(!GpuArchFamily::Cdna.supports_bf16());
        assert!(GpuArchFamily::Cdna2.supports_bf16());
        assert!(GpuArchFamily::Cdna3.supports_bf16());
        assert!(GpuArchFamily::Rdna3.supports_bf16());
    }

    #[test]
    fn wavefront_size_gcn_is_64() {
        assert_eq!(GpuArchFamily::Gcn5.wavefront_size(), 64);
        assert_eq!(GpuArchFamily::Cdna.wavefront_size(), 64);
        assert_eq!(GpuArchFamily::Cdna2.wavefront_size(), 64);
    }

    #[test]
    fn wavefront_size_rdna_is_32() {
        assert_eq!(GpuArchFamily::Rdna2.wavefront_size(), 32);
        assert_eq!(GpuArchFamily::Rdna3.wavefront_size(), 32);
    }

    #[test]
    fn capabilities_from_gcn_arch() {
        let caps = HipDeviceCapabilities::from_gcn_arch(0, "MI250X", "gfx90a");
        assert_eq!(caps.arch_family, GpuArchFamily::Cdna2);
        assert!(caps.supports_mfma);
        assert!(caps.supports_bf16);
        assert!(caps.supports_fp16);
        assert_eq!(caps.wavefront_size, 64);
    }

    #[test]
    fn capabilities_default_is_unknown() {
        let caps = HipDeviceCapabilities::default();
        assert_eq!(caps.arch_family, GpuArchFamily::Unknown);
        assert!(!caps.supports_mfma);
        assert!(!caps.supports_bf16);
    }

    #[test]
    fn bitnet_int2_support_requires_fp16_and_cus() {
        let mut caps = HipDeviceCapabilities::from_gcn_arch(0, "MI250X", "gfx90a");
        caps.compute_units = 110;
        assert!(caps.supports_bitnet_int2());

        let empty = HipDeviceCapabilities::default();
        assert!(!empty.supports_bitnet_int2());
    }

    #[test]
    fn enumerate_devices_returns_empty() {
        assert!(enumerate_devices().is_empty());
    }

    #[test]
    fn get_device_returns_none() {
        assert!(get_device(0).is_none());
    }

    #[test]
    fn device_count_is_zero() {
        assert_eq!(device_count(), 0);
    }

    #[test]
    fn select_best_device_empty() {
        assert_eq!(select_best_device(&[]), None);
    }

    #[test]
    fn select_best_device_prefers_mfma() {
        let mut cdna2 = HipDeviceCapabilities::from_gcn_arch(0, "MI250X", "gfx90a");
        cdna2.compute_units = 110;
        cdna2.total_memory = 64 * 1024 * 1024 * 1024; // 64 GiB

        let mut rdna3 = HipDeviceCapabilities::from_gcn_arch(1, "RX 7900 XTX", "gfx1100");
        rdna3.compute_units = 96;
        rdna3.total_memory = 24 * 1024 * 1024 * 1024; // 24 GiB

        let devices = vec![rdna3, cdna2];
        assert_eq!(select_best_device(&devices), Some(1)); // CDNA2 wins
    }

    #[test]
    fn select_best_device_tiebreak_by_cus() {
        let mut a = HipDeviceCapabilities::from_gcn_arch(0, "GPU A", "gfx1030");
        a.compute_units = 40;
        let mut b = HipDeviceCapabilities::from_gcn_arch(1, "GPU B", "gfx1030");
        b.compute_units = 80;
        assert_eq!(select_best_device(&[a, b]), Some(1));
    }

    #[test]
    fn arch_family_display() {
        assert_eq!(format!("{}", GpuArchFamily::Cdna2), "CDNA2");
        assert_eq!(format!("{}", GpuArchFamily::Unknown), "Unknown");
    }
}
