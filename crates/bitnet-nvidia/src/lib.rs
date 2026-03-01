//! NVIDIA-focused GPU tuning helpers.
//!
//! This crate intentionally stays tiny so NVIDIA-specific heuristics can be
//! shared across backend/runtime crates without pulling in larger dependencies.

/// PCI vendor ID used by NVIDIA GPUs.
pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

/// Threads per warp on NVIDIA hardware.
pub const NVIDIA_WARP_SIZE: u32 = 32;

/// Canonical 1-D workgroup candidate sizes for NVIDIA tuning sweeps.
pub const NVIDIA_1D_WORKGROUP_CANDIDATES: [u32; 4] = [64, 128, 256, 512];

/// Returns `true` when a PCI vendor ID belongs to NVIDIA.
#[must_use]
pub const fn is_nvidia_vendor(vendor_id: u32) -> bool {
    vendor_id == NVIDIA_VENDOR_ID
}

/// Returns `true` if the workgroup size is aligned to warp boundaries.
#[must_use]
pub const fn is_warp_aligned(workgroup_size: u32) -> bool {
    workgroup_size != 0 && workgroup_size.is_multiple_of(NVIDIA_WARP_SIZE)
}

/// Return an NVIDIA-tuned workgroup size for 1-D dispatches.
///
/// Heuristics:
/// - Always warp-aligned (multiple of 32).
/// - Prefer 256 for Blackwell / Ada / Ampere (good occupancy).
/// - Fall back to 128 for small workloads, 64 for tiny workloads.
#[must_use]
pub const fn optimal_workgroup_size_1d(elements: u32) -> u32 {
    if elements == 0 {
        return NVIDIA_WARP_SIZE;
    }
    if elements <= 64 {
        NVIDIA_WARP_SIZE * 2 // 64
    } else if elements <= 256 {
        128
    } else {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_id_detects_nvidia() {
        assert!(is_nvidia_vendor(NVIDIA_VENDOR_ID));
        assert!(!is_nvidia_vendor(0x8086));
        assert!(!is_nvidia_vendor(0x1002));
    }

    #[test]
    fn candidates_are_warp_aligned() {
        for candidate in NVIDIA_1D_WORKGROUP_CANDIDATES {
            assert!(is_warp_aligned(candidate));
        }
    }

    #[test]
    fn heuristic_thresholds() {
        assert_eq!(optimal_workgroup_size_1d(0), 32);
        assert_eq!(optimal_workgroup_size_1d(1), 64);
        assert_eq!(optimal_workgroup_size_1d(64), 64);
        assert_eq!(optimal_workgroup_size_1d(65), 128);
        assert_eq!(optimal_workgroup_size_1d(256), 128);
        assert_eq!(optimal_workgroup_size_1d(257), 256);
    }
}
