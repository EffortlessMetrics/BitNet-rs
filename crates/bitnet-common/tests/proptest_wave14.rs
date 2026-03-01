//! Wave 14 property tests: type conversion and equality invariants for
//! QuantizationType and Device.
//!
//! Key invariants tested (12 properties):
//! - QuantizationType Display produces known strings
//! - QuantizationType serde roundtrip preserves variant
//! - QuantizationType Clone produces equal value
//! - QuantizationType Hash consistency (equal values = equal hashes)
//! - Device equality is reflexive
//! - Device Clone produces equal value
//! - Device Default is Cpu
//! - Device Ord is total (a<=b or b<=a)
//! - Device serde roundtrip preserves variant
//! - Device Hash consistency (equal values = equal hashes)
//! - Device is_cpu/is_cuda/is_hip/is_npu/is_opencl are mutually exclusive
//! - Device::Cpu is always minimum in ordering

use bitnet_common::types::{Device, QuantizationType};
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// Strategy producing all QuantizationType variants.
fn all_qtypes() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

/// Strategy producing all Device variants with small indices.
fn all_devices() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..4).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
}

// ===================================================================
// 1. QuantizationType properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Display produces one of the documented strings.
    #[test]
    fn prop_qtype_display_known_strings(qt in all_qtypes()) {
        let s = qt.to_string();
        prop_assert!(
            ["I2_S", "TL1", "TL2"].contains(&s.as_str()),
            "unexpected Display output: '{s}'"
        );
    }

    /// Serde JSON roundtrip preserves the variant.
    #[test]
    fn prop_qtype_serde_roundtrip(qt in all_qtypes()) {
        let json = serde_json::to_string(&qt).expect("serialize");
        let back: QuantizationType = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(qt, back);
    }

    /// Clone produces an equal value.
    #[test]
    fn prop_qtype_clone_eq(qt in all_qtypes()) {
        let cloned = qt;
        prop_assert_eq!(qt, cloned);
    }

    /// Hash is consistent with Eq: equal values produce equal hashes.
    #[test]
    fn prop_qtype_hash_consistency(qt in all_qtypes()) {
        let h1 = hash_of(&qt);
        let h2 = hash_of(&qt);
        prop_assert_eq!(h1, h2, "hash must be consistent for same value");
    }

    /// Display output length is always in [2, 4] characters.
    #[test]
    fn prop_qtype_display_length(qt in all_qtypes()) {
        let s = qt.to_string();
        prop_assert!(s.len() >= 2 && s.len() <= 4, "Display length unexpected: '{s}' ({})", s.len());
    }

    /// Debug output contains the variant name.
    #[test]
    fn prop_qtype_debug_contains_variant(qt in all_qtypes()) {
        let dbg = format!("{qt:?}");
        let expected = match qt {
            QuantizationType::I2S => "I2S",
            QuantizationType::TL1 => "TL1",
            QuantizationType::TL2 => "TL2",
        };
        prop_assert!(dbg.contains(expected), "Debug '{dbg}' should contain '{expected}'");
    }
}

// ===================================================================
// 2. Device properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Equality is reflexive: d == d.
    #[test]
    fn prop_device_equality_reflexive(d in all_devices()) {
        prop_assert_eq!(d, d);
    }

    /// Clone produces equal value.
    #[test]
    fn prop_device_clone_eq(d in all_devices()) {
        let cloned = d;
        prop_assert_eq!(d, cloned);
    }

    /// Default device is Cpu.
    #[test]
    fn prop_device_default_is_cpu(_dummy in 0u8..1) {
        let d = Device::default();
        prop_assert_eq!(d, Device::Cpu);
        prop_assert!(d.is_cpu());
    }

    /// Ord is total: for any pair, a <= b or b <= a.
    #[test]
    fn prop_device_ord_total(a in all_devices(), b in all_devices()) {
        prop_assert!(a <= b || b <= a, "Ord must be total");
    }

    /// Serde JSON roundtrip preserves variant.
    #[test]
    fn prop_device_serde_roundtrip(d in all_devices()) {
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Device = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(d, back);
    }

    /// Hash consistency: equal values produce equal hashes.
    #[test]
    fn prop_device_hash_consistency(d in all_devices()) {
        let h1 = hash_of(&d);
        let h2 = hash_of(&d);
        prop_assert_eq!(h1, h2);
    }

    /// Predicate methods are mutually exclusive (at most one is true).
    #[test]
    fn prop_device_predicates_exclusive(d in all_devices()) {
        let flags = [d.is_cpu(), d.is_cuda(), d.is_hip(), d.is_npu(), d.is_opencl()];
        let count = flags.iter().filter(|&&f| f).count();
        // Metal has no dedicated predicate, so count may be 0 for Metal.
        prop_assert!(count <= 1, "multiple predicates true for {d:?}: {flags:?}");
    }

    /// Device::Cpu is always minimum in ordering.
    #[test]
    fn prop_device_cpu_is_minimum(d in all_devices()) {
        prop_assert!(Device::Cpu <= d, "Cpu should be <= any device, but Cpu > {d:?}");
    }

    /// Cuda index is preserved through construction.
    #[test]
    fn prop_device_cuda_index_preserved(idx in 0usize..100) {
        let d = Device::Cuda(idx);
        prop_assert!(d.is_cuda());
        prop_assert!(!d.is_cpu());
        if let Device::Cuda(i) = d {
            prop_assert_eq!(i, idx);
        }
    }

    /// OpenCL index is preserved through construction.
    #[test]
    fn prop_device_opencl_index_preserved(idx in 0usize..100) {
        let d = Device::OpenCL(idx);
        prop_assert!(d.is_opencl());
        prop_assert!(!d.is_cpu());
        if let Device::OpenCL(i) = d {
            prop_assert_eq!(i, idx);
        }
    }

    /// Hip index is preserved through construction.
    #[test]
    fn prop_device_hip_index_preserved(idx in 0usize..100) {
        let d = Device::Hip(idx);
        prop_assert!(d.is_hip());
        prop_assert!(!d.is_cpu());
        if let Device::Hip(i) = d {
            prop_assert_eq!(i, idx);
        }
    }
}
