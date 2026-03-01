//! Intel Arc GPU capability detection and tier classification.
//!
//! Provides hardware-specific capability presets for Intel Arc Alchemist
//! (A-series) and Battlemage (B-series) GPUs.  These capabilities inform
//! dispatch sizing, kernel selection, and performance tuning decisions.

use std::fmt;

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

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
