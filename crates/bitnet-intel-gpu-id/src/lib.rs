//! Intel GPU identification helpers.
//!
//! This crate contains pure string-based heuristics for identifying Intel
//! GPUs, with a focus on Intel Arc devices used by the OpenCL/oneAPI path.

/// Known Intel Arc device name patterns.
const ARC_PATTERNS: &[&str] = &[
    "arc a",
    "arc b",
    "arc a770",
    "arc a750",
    "arc a580",
    "arc a380",
    "arc a310",
    "arc b580",
    "arc b570",
    "arc pro",
    "arc graphics",
];

/// Returns `true` if `vendor` looks like Intel.
#[must_use]
pub fn is_intel_vendor(vendor: &str) -> bool {
    vendor.to_ascii_lowercase().contains("intel")
}

/// Returns `true` if `device_name` resembles an Intel Arc model.
///
/// This does not check vendor; use [`is_intel_arc_device`] for full matching.
#[must_use]
pub fn is_arc_name(device_name: &str) -> bool {
    let device_lower = device_name.to_ascii_lowercase();
    ARC_PATTERNS.iter().any(|pattern| device_lower.contains(pattern))
        || device_lower.contains("arc")
}

/// Returns `true` if the vendor/device pair looks like an Intel Arc GPU.
#[must_use]
pub fn is_intel_arc_device(vendor: &str, device_name: &str) -> bool {
    is_intel_vendor(vendor) && is_arc_name(device_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intel_vendor_detection_is_case_insensitive() {
        assert!(is_intel_vendor("Intel(R) Corporation"));
        assert!(is_intel_vendor("INTEL"));
        assert!(is_intel_vendor("intel"));
        assert!(!is_intel_vendor("NVIDIA"));
        assert!(!is_intel_vendor("AMD"));
    }

    #[test]
    fn arc_name_detection_accepts_known_models() {
        assert!(is_arc_name("Intel Arc A770"));
        assert!(is_arc_name("Intel Arc B580"));
        assert!(is_arc_name("Intel Arc Pro A60M"));
        assert!(is_arc_name("Some Future Arc Device"));
    }

    #[test]
    fn intel_arc_detection_requires_intel_vendor() {
        assert!(is_intel_arc_device("Intel", "Arc A770"));
        assert!(!is_intel_arc_device("AMD", "Arc A770"));
        assert!(!is_intel_arc_device("Intel", "Intel UHD Graphics 770"));
    }
}
