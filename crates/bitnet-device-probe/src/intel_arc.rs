//! Intel Arc GPU capability detection and tier classification.
//!
//! Provides hardware-specific capability presets for Intel Arc Alchemist
//! (A-series) and Battlemage (B-series) GPUs.  These capabilities inform
//! dispatch sizing, kernel selection, and performance tuning decisions.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::runtimes::{
    LevelZeroProbe, OpenClRuntimeProbe, OpenVinoProbe, level_zero::probe_level_zero,
    opencl::probe_opencl_runtime, openvino::probe_openvino,
};

// ── PCI Device IDs (Alchemist / Battlemage) ────────────────────────────────

/// PCI device ID for Intel Arc A770 (DG2-512 full die).
pub const PCI_ID_ARC_A770: u32 = 0x56A0;
/// PCI device ID for Intel Arc A750 (DG2-512 cut-down).
pub const PCI_ID_ARC_A750: u32 = 0x56A1;
/// PCI device ID for Intel Arc A580 (DG2-256).
pub const PCI_ID_ARC_A580: u32 = 0x56A5;
/// PCI device ID for Intel Arc A380 (DG2-128).
pub const PCI_ID_ARC_A380: u32 = 0x56A6;
/// PCI device ID for Intel Arc A310 (DG2-128 cut-down).
pub const PCI_ID_ARC_A310: u32 = 0x56A7;

// ── A770 runtime receipt identity ───────────────────────────────────────────

/// Requested backend label for the Intel Arc A770 OpenCL proof lane.
pub const INTEL_ARC_A770_REQUESTED_BACKEND: &str = "intel-a770-opencl";
/// Selected backend label when native OpenCL sees the A770 device.
pub const INTEL_ARC_A770_OPENCL_BACKEND: &str = "intel-a770-opencl";
/// Expected PCI device ID for Intel Arc A770.
pub const INTEL_ARC_A770_PCI_DEVICE_ID: &str = "0x56A0";
/// Universal proof stage for the A770 visibility-only runtime probe.
pub const INTEL_ARC_A770_PROOF_STAGE_RUNTIME_DETECTED: &str = "runtime_detected";

/// Runtime visibility facts for the Intel Arc A770 campaign lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelArcA770RuntimeProbe {
    /// Universal proof stage for this visibility-only probe.
    pub proof_stage: String,
    /// Requested backend identity for A770 receipts.
    pub requested_backend: String,
    /// Selected backend only when native OpenCL reports the A770 device.
    pub selected_backend: Option<String>,
    /// Runtime API associated with the strongest matching runtime evidence.
    pub runtime_api: Option<String>,
    /// Runtime device name selected for native OpenCL proof visibility.
    pub selected_device_name: Option<String>,
    /// Whether any exact A770 identity was visible through the probed runtimes.
    pub available: bool,
    /// Expected PCI device ID when A770 identity is visible.
    pub pci_device_id: Option<String>,
    /// Runtime evidence entries that matched A770 by name or PCI ID.
    pub identity_evidence: Vec<String>,
    /// Whether the OpenCL runtime itself was available.
    pub opencl_runtime_available: bool,
    /// Whether OpenCL reported an exact A770 GPU device.
    pub opencl_available: bool,
    /// OpenCL platform name for the selected A770 device when available.
    pub opencl_platform_name: Option<String>,
    /// OpenCL device name for the selected A770 device when available.
    pub opencl_device_name: Option<String>,
    /// OpenCL vendor for the selected A770 device when available.
    pub opencl_vendor: Option<String>,
    /// OpenCL driver version for the selected A770 device when available.
    pub opencl_driver_version: Option<String>,
    /// Whether Level Zero tooling/runtime visibility was available.
    pub level_zero_runtime_available: bool,
    /// Whether Level Zero reported an A770 name or PCI device ID.
    pub level_zero_available: bool,
    /// Matching Level Zero device names or lines.
    pub level_zero_devices: Vec<String>,
    /// Matching Level Zero PCI/device IDs.
    pub level_zero_device_ids: Vec<String>,
    /// Whether OpenVINO runtime visibility was available.
    pub openvino_runtime_available: bool,
    /// Whether OpenVINO exposes a GPU token on this machine.
    pub openvino_gpu_visible: bool,
    /// First OpenVINO GPU device token when visible.
    pub openvino_gpu_device: Option<String>,
    /// OpenVINO GPU full device name when available.
    pub openvino_gpu_full_name: Option<String>,
    /// Whether a matching OpenVINO GPU identity was recorded as reference-only.
    pub openvino_reference_only: bool,
    /// Expected VRAM from the A770 hardware preset when identity is visible.
    pub expected_vram_bytes: Option<u64>,
    /// Expected VRAM in GiB from the A770 hardware preset when identity is visible.
    pub expected_vram_gib: Option<u64>,
    /// ReBAR context status. This probe records the gap but does not infer it.
    pub rebar_status: String,
    /// Render-node path when available. Windows/command-only probes often cannot provide one.
    pub render_node: Option<String>,
    /// Always false: A770-004 is visibility-only and does not dispatch kernels.
    pub kernel_execution: bool,
    /// Always false: A770-004 is not BitNet inference.
    pub bitnet_inference: bool,
    /// Always false: A770-004 does not prove packed QK256 decode.
    pub qk256_decode: bool,
    /// Always false: CPU or another GPU cannot satisfy A770 runtime proof.
    pub fallback_used: bool,
    /// Non-fatal reason explaining why exact A770 identity was not found.
    pub failure_reason: Option<String>,
}

// ── IntelArcTier ───────────────────────────────────────────────────────────

/// Classification tier for Intel Arc discrete GPUs.
///
/// Each variant carries a preset of hardware capabilities sourced from
/// Intel Xe-HPG architecture documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntelArcTier {
    /// Arc A770 — full DG2-512 die (32 Xe-cores, 512 EUs, 16 GB VRAM).
    A770,
    /// Arc A750 — cut-down DG2-512 (28 Xe-cores, 448 EUs, 8 GB VRAM).
    A750,
    /// Arc A580 — DG2-256 (16 Xe-cores, 256 EUs, 8 GB VRAM).
    A580,
    /// Arc A380 — DG2-128 (8 Xe-cores, 128 EUs, 6 GB VRAM).
    A380,
    /// Arc A310 — cut-down DG2-128 (6 Xe-cores, 96 EUs, 4 GB VRAM).
    A310,
}

impl fmt::Display for IntelArcTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A770 => write!(f, "Arc A770"),
            Self::A750 => write!(f, "Arc A750"),
            Self::A580 => write!(f, "Arc A580"),
            Self::A380 => write!(f, "Arc A380"),
            Self::A310 => write!(f, "Arc A310"),
        }
    }
}

impl IntelArcTier {
    /// All known Alchemist tiers.
    pub const ALL: &[Self] = &[Self::A770, Self::A750, Self::A580, Self::A380, Self::A310];

    /// Build [`IntelArcCapabilities`] from this tier's hardware preset.
    #[must_use]
    pub fn capabilities(self) -> IntelArcCapabilities {
        IntelArcCapabilities::from_tier(self)
    }

    /// Look up a tier by PCI device ID.
    #[must_use]
    pub const fn from_pci_id(device_id: u32) -> Option<Self> {
        match device_id {
            PCI_ID_ARC_A770 => Some(Self::A770),
            PCI_ID_ARC_A750 => Some(Self::A750),
            PCI_ID_ARC_A580 => Some(Self::A580),
            PCI_ID_ARC_A380 => Some(Self::A380),
            PCI_ID_ARC_A310 => Some(Self::A310),
            _ => None,
        }
    }

    /// PCI device ID for this tier.
    #[must_use]
    pub const fn pci_device_id(self) -> u32 {
        match self {
            Self::A770 => PCI_ID_ARC_A770,
            Self::A750 => PCI_ID_ARC_A750,
            Self::A580 => PCI_ID_ARC_A580,
            Self::A380 => PCI_ID_ARC_A380,
            Self::A310 => PCI_ID_ARC_A310,
        }
    }
}

// ── IntelArcCapabilities ───────────────────────────────────────────────────

/// Hardware capabilities for an Intel Arc discrete GPU.
///
/// Values are sourced from the Xe-HPG architecture specification and
/// can be used for dispatch sizing, kernel selection, and SLM tiling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntelArcCapabilities {
    /// Detected or matched [`IntelArcTier`], if any.
    pub tier: Option<IntelArcTier>,
    /// PCI device ID (e.g. `0x56A0` for A770).
    pub device_id: u32,
    /// Number of Execution Units.
    pub eu_count: u32,
    /// Number of Xe-cores (each contains 16 EUs on Xe-HPG).
    pub xe_core_count: u32,
    /// Supported subgroup (SIMD lane) widths on Xe-HPG: typically 8, 16, 32.
    pub subgroup_sizes: Vec<u32>,
    /// Shared Local Memory per sub-slice, in bytes (64 KiB on Xe-HPG).
    pub slm_size: u64,
    /// Maximum work-group size (1024 on Xe-HPG).
    pub max_workgroup_size: u32,
    /// Native FP16 (half-precision) arithmetic support.
    pub fp16_support: bool,
    /// FP64 (double-precision) support. Emulated on Arc consumer SKUs.
    pub fp64_support: bool,
    /// DP4A (INT8 dot-product) hardware support.
    pub int8_dot_product: bool,
    /// Unified Shared Memory — present on Arc but memory is discrete.
    pub unified_memory: bool,
    /// Video RAM in bytes.
    pub vram_bytes: u64,
}

impl IntelArcCapabilities {
    /// Build capabilities from a known tier preset.
    #[must_use]
    pub fn from_tier(tier: IntelArcTier) -> Self {
        // Xe-HPG common: subgroup 8/16/32, SLM 64 KiB, workgroup 1024,
        // FP16 native, FP64 emulated, DP4A yes, USM yes.
        let common = |tier_val, device_id, eu, xe_cores, vram_gb: u64| Self {
            tier: Some(tier_val),
            device_id,
            eu_count: eu,
            xe_core_count: xe_cores,
            subgroup_sizes: vec![8, 16, 32],
            slm_size: 64 * 1024,
            max_workgroup_size: 1024,
            fp16_support: true,
            fp64_support: false, // emulated on consumer Arc
            int8_dot_product: true,
            unified_memory: true,
            vram_bytes: vram_gb * 1024 * 1024 * 1024,
        };

        match tier {
            IntelArcTier::A770 => common(tier, PCI_ID_ARC_A770, 512, 32, 16),
            IntelArcTier::A750 => common(tier, PCI_ID_ARC_A750, 448, 28, 8),
            IntelArcTier::A580 => common(tier, PCI_ID_ARC_A580, 256, 16, 8),
            IntelArcTier::A380 => common(tier, PCI_ID_ARC_A380, 128, 8, 6),
            IntelArcTier::A310 => common(tier, PCI_ID_ARC_A310, 96, 6, 4),
        }
    }

    /// Build a conservative fallback for an unrecognised Intel Arc device.
    ///
    /// Uses the A380 preset as a safe lower bound.
    #[must_use]
    pub fn unknown_arc_fallback() -> Self {
        Self {
            tier: None,
            device_id: 0,
            eu_count: 128,
            xe_core_count: 8,
            subgroup_sizes: vec![8, 16, 32],
            slm_size: 64 * 1024,
            max_workgroup_size: 1024,
            fp16_support: true,
            fp64_support: false,
            int8_dot_product: true,
            unified_memory: true,
            vram_bytes: 6 * 1024 * 1024 * 1024,
        }
    }

    /// VRAM in gibibytes (GiB), rounded down.
    #[must_use]
    pub const fn vram_gib(&self) -> u64 {
        self.vram_bytes / (1024 * 1024 * 1024)
    }
}

// ── Detection helpers ──────────────────────────────────────────────────────

/// Returns `true` if the device name looks like an Intel Arc Alchemist GPU.
///
/// Matches A770, A750, A580, A380, A310 patterns (case-insensitive).
pub fn is_arc_alchemist(device_name: &str) -> bool {
    let lower = device_name.to_ascii_lowercase();
    // Must contain "arc" and an Alchemist model number
    lower.contains("arc")
        && (lower.contains("a770")
            || lower.contains("a750")
            || lower.contains("a580")
            || lower.contains("a380")
            || lower.contains("a310"))
}

/// Detect Intel Arc capabilities from a device name string.
///
/// Attempts to match a known Arc tier from the device name. Returns
/// `None` if the device is not recognised as an Intel Arc GPU.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::intel_arc::detect_intel_arc;
///
/// let caps = detect_intel_arc("Intel(R) Arc(TM) A770 Graphics");
/// assert!(caps.is_some());
/// let caps = caps.unwrap();
/// assert_eq!(caps.eu_count, 512);
/// assert_eq!(caps.vram_gib(), 16);
/// assert!(caps.fp16_support);
/// ```
pub fn detect_intel_arc(device_name: &str) -> Option<IntelArcCapabilities> {
    let lower = device_name.to_ascii_lowercase();
    if !lower.contains("arc") {
        return None;
    }

    // Try to match a specific tier
    let tier = if lower.contains("a770") {
        Some(IntelArcTier::A770)
    } else if lower.contains("a750") {
        Some(IntelArcTier::A750)
    } else if lower.contains("a580") {
        Some(IntelArcTier::A580)
    } else if lower.contains("a380") {
        Some(IntelArcTier::A380)
    } else if lower.contains("a310") {
        Some(IntelArcTier::A310)
    } else {
        None
    };

    tier.map_or_else(
        || Some(IntelArcCapabilities::unknown_arc_fallback()),
        |t| Some(IntelArcCapabilities::from_tier(t)),
    )
}

/// Detect Intel Arc capabilities from a PCI device ID.
///
/// Returns `None` if the device ID does not match a known Arc SKU.
pub fn detect_intel_arc_by_pci_id(device_id: u32) -> Option<IntelArcCapabilities> {
    IntelArcTier::from_pci_id(device_id).map(IntelArcCapabilities::from_tier)
}

/// Probe A770 runtime visibility without compiling kernels or running inference.
#[must_use]
pub fn probe_intel_arc_a770_runtime() -> IntelArcA770RuntimeProbe {
    let opencl = probe_opencl_runtime();
    let level_zero = probe_level_zero();
    let openvino = probe_openvino();
    probe_intel_arc_a770_runtime_from_probes(&opencl, &level_zero, &openvino)
}

/// Build an A770 runtime visibility result from lower-level runtime probes.
#[must_use]
pub fn probe_intel_arc_a770_runtime_from_probes(
    opencl: &OpenClRuntimeProbe,
    level_zero: &LevelZeroProbe,
    openvino: &OpenVinoProbe,
) -> IntelArcA770RuntimeProbe {
    let opencl_device = opencl.devices.iter().find(|device| {
        device.is_gpu
            && vendor_matches_intel(&device.vendor)
            && name_or_id_matches_arc_a770(&device.device_name)
    });

    let level_zero_devices: Vec<String> = level_zero
        .devices
        .iter()
        .filter(|device| name_or_id_matches_arc_a770(device))
        .cloned()
        .collect();
    let level_zero_device_ids: Vec<String> = level_zero
        .device_ids
        .iter()
        .filter(|device_id| device_id_matches_arc_a770(device_id))
        .cloned()
        .collect();

    let openvino_gpu_device = openvino.gpu_device_token();
    let openvino_gpu_full_name =
        openvino_gpu_device.as_deref().and_then(|token| openvino.full_name_for(token));
    let openvino_gpu_visible = openvino_gpu_device.is_some();
    let openvino_name_matches =
        openvino_gpu_full_name.as_deref().is_some_and(name_or_id_matches_arc_a770);

    let opencl_available = opencl_device.is_some();
    let level_zero_available = !level_zero_devices.is_empty() || !level_zero_device_ids.is_empty();
    let available = opencl_available || level_zero_available || openvino_name_matches;

    let mut identity_evidence = Vec::new();
    if let Some(device) = opencl_device {
        identity_evidence.push(format!("opencl:{}", device.device_name));
    }
    identity_evidence
        .extend(level_zero_devices.iter().map(|device| format!("level_zero:{device}")));
    identity_evidence.extend(
        level_zero_device_ids
            .iter()
            .map(|device_id| format!("level_zero_pci_device_id:{device_id}")),
    );
    if let (Some(token), Some(full_name)) = (&openvino_gpu_device, &openvino_gpu_full_name)
        && openvino_name_matches
    {
        identity_evidence.push(format!("openvino_reference:{token}:{full_name}"));
    }

    let selected_backend = opencl_available.then(|| INTEL_ARC_A770_OPENCL_BACKEND.to_owned());
    let runtime_api =
        selected_runtime_api(opencl_available, level_zero_available, openvino_name_matches);
    let selected_device_name = opencl_device.map(|device| device.device_name.clone());
    let caps = available.then(|| IntelArcTier::A770.capabilities());
    let failure_reason = if available {
        None
    } else {
        Some("Intel Arc A770 identity was not visible through OpenCL, Level Zero, or OpenVINO GPU reference visibility".to_owned())
    };

    IntelArcA770RuntimeProbe {
        proof_stage: INTEL_ARC_A770_PROOF_STAGE_RUNTIME_DETECTED.to_owned(),
        requested_backend: INTEL_ARC_A770_REQUESTED_BACKEND.to_owned(),
        selected_backend,
        runtime_api,
        selected_device_name,
        available,
        pci_device_id: available.then_some(INTEL_ARC_A770_PCI_DEVICE_ID.to_owned()),
        identity_evidence,
        opencl_runtime_available: opencl.runtime_available,
        opencl_available,
        opencl_platform_name: opencl_device.and_then(|device| device.platform_name.clone()),
        opencl_device_name: opencl_device.map(|device| device.device_name.clone()),
        opencl_vendor: opencl_device.map(|device| device.vendor.clone()),
        opencl_driver_version: opencl_device.and_then(|device| device.driver_version.clone()),
        level_zero_runtime_available: level_zero.runtime_available,
        level_zero_available,
        level_zero_devices,
        level_zero_device_ids,
        openvino_runtime_available: openvino.runtime_available,
        openvino_gpu_visible,
        openvino_gpu_device,
        openvino_gpu_full_name,
        openvino_reference_only: openvino_name_matches,
        expected_vram_bytes: caps.as_ref().map(|caps| caps.vram_bytes),
        expected_vram_gib: caps.as_ref().map(IntelArcCapabilities::vram_gib),
        rebar_status: "not_probed".to_owned(),
        render_node: None,
        kernel_execution: false,
        bitnet_inference: false,
        qk256_decode: false,
        fallback_used: false,
        failure_reason,
    }
}

fn vendor_matches_intel(value: &str) -> bool {
    value.to_ascii_lowercase().contains("intel")
}

fn name_or_id_matches_arc_a770(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn device_id_matches_arc_a770(value: &str) -> bool {
    let normalized = value.trim().to_ascii_uppercase();
    normalized == INTEL_ARC_A770_PCI_DEVICE_ID
        || normalized.trim_start_matches("0X") == "56A0"
        || normalized.contains("56A0")
}

fn selected_runtime_api(
    opencl_available: bool,
    level_zero_available: bool,
    openvino_name_matches: bool,
) -> Option<String> {
    if opencl_available {
        Some("opencl".to_owned())
    } else if level_zero_available {
        Some("level_zero".to_owned())
    } else if openvino_name_matches {
        Some("openvino".to_owned())
    } else {
        None
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtimes::{
        LevelZeroProbe, OpenClRuntimeDevice, OpenClRuntimeProbe, OpenVinoDeviceProbe, OpenVinoProbe,
    };

    // ── Tier presets ───────────────────────────────────────────────────

    #[test]
    fn a770_capabilities() {
        let caps = IntelArcTier::A770.capabilities();
        assert_eq!(caps.tier, Some(IntelArcTier::A770));
        assert_eq!(caps.device_id, PCI_ID_ARC_A770);
        assert_eq!(caps.eu_count, 512);
        assert_eq!(caps.xe_core_count, 32);
        assert_eq!(caps.subgroup_sizes, vec![8, 16, 32]);
        assert_eq!(caps.slm_size, 64 * 1024);
        assert_eq!(caps.max_workgroup_size, 1024);
        assert!(caps.fp16_support);
        assert!(!caps.fp64_support);
        assert!(caps.int8_dot_product);
        assert!(caps.unified_memory);
        assert_eq!(caps.vram_gib(), 16);
    }

    #[test]
    fn a750_capabilities() {
        let caps = IntelArcTier::A750.capabilities();
        assert_eq!(caps.eu_count, 448);
        assert_eq!(caps.xe_core_count, 28);
        assert_eq!(caps.vram_gib(), 8);
    }

    #[test]
    fn a580_capabilities() {
        let caps = IntelArcTier::A580.capabilities();
        assert_eq!(caps.eu_count, 256);
        assert_eq!(caps.xe_core_count, 16);
        assert_eq!(caps.vram_gib(), 8);
    }

    #[test]
    fn a380_capabilities() {
        let caps = IntelArcTier::A380.capabilities();
        assert_eq!(caps.eu_count, 128);
        assert_eq!(caps.xe_core_count, 8);
        assert_eq!(caps.vram_gib(), 6);
    }

    #[test]
    fn a310_capabilities() {
        let caps = IntelArcTier::A310.capabilities();
        assert_eq!(caps.eu_count, 96);
        assert_eq!(caps.xe_core_count, 6);
        assert_eq!(caps.vram_gib(), 4);
    }

    #[test]
    fn all_tiers_have_xe_hpg_common_traits() {
        for &tier in IntelArcTier::ALL {
            let caps = tier.capabilities();
            assert_eq!(caps.subgroup_sizes, vec![8, 16, 32], "tier {tier}");
            assert_eq!(caps.slm_size, 64 * 1024, "tier {tier}");
            assert_eq!(caps.max_workgroup_size, 1024, "tier {tier}");
            assert!(caps.fp16_support, "tier {tier}");
            assert!(!caps.fp64_support, "tier {tier}");
            assert!(caps.int8_dot_product, "tier {tier}");
            assert!(caps.unified_memory, "tier {tier}");
            assert!(caps.tier.is_some(), "tier {tier}");
        }
    }

    #[test]
    fn eu_counts_are_monotonically_ordered() {
        let ordered = [
            IntelArcTier::A310,
            IntelArcTier::A380,
            IntelArcTier::A580,
            IntelArcTier::A750,
            IntelArcTier::A770,
        ];
        for pair in ordered.windows(2) {
            assert!(
                pair[0].capabilities().eu_count < pair[1].capabilities().eu_count,
                "{} should have fewer EUs than {}",
                pair[0],
                pair[1],
            );
        }
    }

    // ── PCI device ID matching ─────────────────────────────────────────

    #[test]
    fn pci_id_roundtrip_all_tiers() {
        for &tier in IntelArcTier::ALL {
            let id = tier.pci_device_id();
            let recovered = IntelArcTier::from_pci_id(id);
            assert_eq!(recovered, Some(tier), "PCI ID {id:#06X} roundtrip failed");
        }
    }

    #[test]
    fn pci_id_unknown_returns_none() {
        assert_eq!(IntelArcTier::from_pci_id(0x0000), None);
        assert_eq!(IntelArcTier::from_pci_id(0xFFFF), None);
        // NVIDIA GA102 ID — definitely not Arc
        assert_eq!(IntelArcTier::from_pci_id(0x2204), None);
    }

    #[test]
    fn detect_by_pci_id_a770() {
        let caps = detect_intel_arc_by_pci_id(PCI_ID_ARC_A770).unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A770));
        assert_eq!(caps.eu_count, 512);
    }

    #[test]
    fn detect_by_pci_id_unknown() {
        assert!(detect_intel_arc_by_pci_id(0x1234).is_none());
    }

    // ── Device name detection ──────────────────────────────────────────

    #[test]
    fn detect_a770_from_device_string() {
        let caps = detect_intel_arc("Intel(R) Arc(TM) A770 Graphics").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A770));
        assert_eq!(caps.eu_count, 512);
        assert_eq!(caps.vram_gib(), 16);
    }

    #[test]
    fn detect_a750_from_device_string() {
        let caps = detect_intel_arc("Intel Arc A750").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A750));
    }

    #[test]
    fn detect_a580_from_device_string() {
        let caps = detect_intel_arc("Arc A580 Graphics").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A580));
    }

    #[test]
    fn detect_a380_from_device_string() {
        let caps = detect_intel_arc("Intel(R) Arc(TM) A380 Graphics").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A380));
    }

    #[test]
    fn detect_a310_from_device_string() {
        let caps = detect_intel_arc("Arc A310").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A310));
    }

    #[test]
    fn detect_case_insensitive() {
        let caps = detect_intel_arc("INTEL ARC A770 GRAPHICS").unwrap();
        assert_eq!(caps.tier, Some(IntelArcTier::A770));
    }

    #[test]
    fn detect_unknown_arc_gets_fallback() {
        let caps = detect_intel_arc("Intel Arc B999 Future GPU").unwrap();
        assert!(caps.tier.is_none());
        // Fallback uses conservative A380-level values
        assert_eq!(caps.eu_count, 128);
        assert_eq!(caps.vram_gib(), 6);
    }

    #[test]
    fn detect_non_arc_returns_none() {
        assert!(detect_intel_arc("Intel(R) UHD Graphics 770").is_none());
        assert!(detect_intel_arc("NVIDIA GeForce RTX 4090").is_none());
        assert!(detect_intel_arc("AMD Radeon RX 7900 XTX").is_none());
    }

    #[test]
    fn detect_empty_string_returns_none() {
        assert!(detect_intel_arc("").is_none());
    }

    // ── A770 runtime probe ─────────────────────────────────────────────

    #[test]
    fn opencl_a770_identity_selects_native_lane_without_execution_claim()
    -> Result<(), serde_json::Error> {
        let opencl = OpenClRuntimeProbe {
            runtime_available: true,
            devices: vec![OpenClRuntimeDevice {
                platform_name: Some("Intel(R) OpenCL Graphics".to_owned()),
                device_name: "Intel(R) Arc(TM) A770 Graphics".to_owned(),
                vendor: "Intel(R) Corporation".to_owned(),
                driver_version: Some("test-driver".to_owned()),
                is_gpu: true,
            }],
            error: None,
        };
        let level_zero = LevelZeroProbe::unavailable("not installed");
        let openvino = OpenVinoProbe::unavailable("not installed");

        let probe = probe_intel_arc_a770_runtime_from_probes(&opencl, &level_zero, &openvino);

        assert!(probe.available);
        assert!(probe.opencl_available);
        assert_eq!(probe.requested_backend, INTEL_ARC_A770_REQUESTED_BACKEND);
        assert_eq!(probe.selected_backend.as_deref(), Some(INTEL_ARC_A770_OPENCL_BACKEND));
        assert_eq!(probe.runtime_api.as_deref(), Some("opencl"));
        assert_eq!(probe.selected_device_name.as_deref(), Some("Intel(R) Arc(TM) A770 Graphics"));
        assert_eq!(probe.pci_device_id.as_deref(), Some(INTEL_ARC_A770_PCI_DEVICE_ID));
        assert_eq!(probe.expected_vram_gib, Some(16));
        assert!(!probe.kernel_execution);
        assert!(!probe.bitnet_inference);
        assert!(!probe.qk256_decode);
        assert!(!probe.fallback_used);
        assert!(probe.identity_evidence.iter().any(|entry| entry.starts_with("opencl:")));

        let value = serde_json::to_value(&probe)?;
        assert_eq!(value["proof_stage"], "runtime_detected");
        assert_eq!(value["requested_backend"], INTEL_ARC_A770_REQUESTED_BACKEND);
        assert_eq!(value["selected_backend"], INTEL_ARC_A770_OPENCL_BACKEND);
        assert_eq!(value["runtime_api"], "opencl");
        assert_eq!(value["kernel_execution"], false);
        assert_eq!(value["bitnet_inference"], false);
        assert_eq!(value["qk256_decode"], false);
        assert_eq!(value["fallback_used"], false);
        assert_eq!(value["rebar_status"], "not_probed");
        Ok(())
    }

    #[test]
    fn level_zero_a770_pci_id_records_visibility_without_selecting_opencl() {
        let opencl = OpenClRuntimeProbe::unavailable("not installed");
        let level_zero = LevelZeroProbe {
            runtime_available: true,
            devices: Vec::new(),
            device_ids: vec!["0x56A0".to_owned()],
            error: None,
        };
        let openvino = OpenVinoProbe::unavailable("not installed");

        let probe = probe_intel_arc_a770_runtime_from_probes(&opencl, &level_zero, &openvino);

        assert!(probe.available);
        assert!(probe.level_zero_available);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api.as_deref(), Some("level_zero"));
        assert_eq!(probe.level_zero_device_ids, vec!["0x56A0"]);
        assert!(probe.identity_evidence.iter().any(|entry| entry.contains("56A0")));
        assert!(!probe.kernel_execution);
    }

    #[test]
    fn openvino_a770_gpu_is_reference_only_not_native_opencl_proof() {
        let opencl = OpenClRuntimeProbe::unavailable("not installed");
        let level_zero = LevelZeroProbe::unavailable("not installed");
        let openvino = OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["CPU".to_owned(), "GPU.0".to_owned()],
            devices: vec![OpenVinoDeviceProbe {
                device: "GPU.0".to_owned(),
                full_name: Some("Intel(R) Arc(TM) A770 Graphics".to_owned()),
                supported_properties: Vec::new(),
                properties: Vec::new(),
            }],
            error: None,
        };

        let probe = probe_intel_arc_a770_runtime_from_probes(&opencl, &level_zero, &openvino);

        assert!(probe.available);
        assert!(probe.openvino_gpu_visible);
        assert!(probe.openvino_reference_only);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api.as_deref(), Some("openvino"));
        assert!(
            probe.identity_evidence.iter().any(|entry| entry.starts_with("openvino_reference:"))
        );
        assert!(!probe.kernel_execution);
        assert!(!probe.bitnet_inference);
    }

    #[test]
    fn generic_intel_gpu_does_not_count_as_a770_identity() {
        let opencl = OpenClRuntimeProbe {
            runtime_available: true,
            devices: vec![OpenClRuntimeDevice {
                platform_name: Some("Intel(R) OpenCL Graphics".to_owned()),
                device_name: "Intel(R) UHD Graphics 770".to_owned(),
                vendor: "Intel(R) Corporation".to_owned(),
                driver_version: Some("test-driver".to_owned()),
                is_gpu: true,
            }],
            error: None,
        };
        let level_zero = LevelZeroProbe {
            runtime_available: true,
            devices: vec!["Intel(R) UHD Graphics 770".to_owned()],
            device_ids: vec!["0x4680".to_owned()],
            error: None,
        };
        let openvino = OpenVinoProbe {
            runtime_available: true,
            version: Some("2026.1".to_owned()),
            available_devices: vec!["GPU.0".to_owned()],
            devices: vec![OpenVinoDeviceProbe {
                device: "GPU.0".to_owned(),
                full_name: Some("Intel(R) UHD Graphics 770".to_owned()),
                supported_properties: Vec::new(),
                properties: Vec::new(),
            }],
            error: None,
        };

        let probe = probe_intel_arc_a770_runtime_from_probes(&opencl, &level_zero, &openvino);

        assert!(!probe.available);
        assert!(!probe.opencl_available);
        assert!(!probe.level_zero_available);
        assert!(probe.openvino_gpu_visible);
        assert_eq!(probe.selected_backend, None);
        assert_eq!(probe.runtime_api, None);
        assert_eq!(probe.expected_vram_bytes, None);
        assert!(probe.failure_reason.is_some());
    }

    // ── is_arc_alchemist ───────────────────────────────────────────────

    #[test]
    fn is_arc_alchemist_positive_cases() {
        assert!(is_arc_alchemist("Intel(R) Arc(TM) A770 Graphics"));
        assert!(is_arc_alchemist("Intel Arc A750"));
        assert!(is_arc_alchemist("Arc A580 Graphics"));
        assert!(is_arc_alchemist("Arc A380"));
        assert!(is_arc_alchemist("Arc A310"));
    }

    #[test]
    fn is_arc_alchemist_case_insensitive() {
        assert!(is_arc_alchemist("INTEL ARC A770"));
        assert!(is_arc_alchemist("intel arc a750"));
    }

    #[test]
    fn is_arc_alchemist_rejects_non_alchemist() {
        // B-series (Battlemage) is not Alchemist
        assert!(!is_arc_alchemist("Intel Arc B580"));
        // Integrated graphics
        assert!(!is_arc_alchemist("Intel UHD Graphics 770"));
        // Non-Intel
        assert!(!is_arc_alchemist("NVIDIA RTX 4090"));
        // Generic Arc without model
        assert!(!is_arc_alchemist("Intel Arc Graphics"));
    }

    #[test]
    fn is_arc_alchemist_empty_string() {
        assert!(!is_arc_alchemist(""));
    }

    // ── unknown_arc_fallback ───────────────────────────────────────────

    #[test]
    fn unknown_fallback_has_conservative_values() {
        let caps = IntelArcCapabilities::unknown_arc_fallback();
        assert!(caps.tier.is_none());
        assert_eq!(caps.device_id, 0);
        assert_eq!(caps.eu_count, 128);
        assert!(caps.fp16_support);
        assert!(caps.int8_dot_product);
    }

    // ── Display ────────────────────────────────────────────────────────

    #[test]
    fn tier_display() {
        assert_eq!(IntelArcTier::A770.to_string(), "Arc A770");
        assert_eq!(IntelArcTier::A310.to_string(), "Arc A310");
    }

    // ── Clone / Eq ─────────────────────────────────────────────────────

    #[test]
    fn capabilities_clone_eq() {
        let a = IntelArcTier::A770.capabilities();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn tier_copy_eq() {
        let a = IntelArcTier::A770;
        let b = a;
        assert_eq!(a, b);
    }
}
