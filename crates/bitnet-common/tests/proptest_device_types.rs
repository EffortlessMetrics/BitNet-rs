//! Wave 12 property tests: Device variant coverage, QuantizationType hash
//! consistency, and error context preservation.
//!
//! Key invariants tested (8 properties):
//! - Device variant predicates are mutually exclusive
//! - Device Ord is transitive across variant families
//! - Device Cuda index ordering: Cuda(a) < Cuda(b) when a < b
//! - Device Hash is consistent with Eq
//! - QuantizationType Display produces the documented strings
//! - QuantizationType Hash consistency (same variant = same hash)
//! - Error context: all error variants include their field data in Display
//! - SecurityError ResourceLimit display contains resource, value, and limit

use bitnet_common::{
    error::{
        BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    },
    types::{Device, QuantizationType},
};
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// Strategy producing all Device variants (with small indices for indexed ones).
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

/// Strategy producing all QuantizationType variants.
fn all_qtypes() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

// ===================================================================
// 1. Device variant predicates are mutually exclusive
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Exactly one of is_cpu/is_cuda/is_hip/is_npu/is_opencl is true (or none
    /// for Metal which lacks a dedicated predicate).
    #[test]
    fn prop_device_predicates_exclusive(d in all_devices()) {
        let flags = [d.is_cpu(), d.is_cuda(), d.is_hip(), d.is_npu(), d.is_opencl()];
        let count: usize = flags.iter().filter(|&&f| f).count();
        // Metal has no dedicated predicate, so count may be 0 for Metal
        match d {
            Device::Metal => prop_assert_eq!(count, 0, "Metal should have no true predicates"),
            _ => prop_assert_eq!(count, 1, "exactly one predicate should be true for {:?}", d),
        }
    }
}

// ===================================================================
// 2. Device Ord is transitive across variant families
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Ordering is transitive: if a <= b and b <= c then a <= c.
    #[test]
    fn prop_device_ord_transitive(
        a in all_devices(),
        b in all_devices(),
        c in all_devices(),
    ) {
        if a <= b && b <= c {
            prop_assert!(a <= c, "transitivity violated: {:?} <= {:?} <= {:?}", a, b, c);
        }
    }

    /// Ordering is antisymmetric: if a <= b and b <= a then a == b.
    #[test]
    fn prop_device_ord_antisymmetric(
        a in all_devices(),
        b in all_devices(),
    ) {
        if a <= b && b <= a {
            prop_assert_eq!(a, b, "antisymmetry violated: {:?} vs {:?}", a, b);
        }
    }
}

// ===================================================================
// 3. Cuda index ordering: Cuda(a) < Cuda(b) when a < b
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Cuda devices sort by index.
    #[test]
    fn prop_cuda_index_ordering(a in 0usize..8, b in 0usize..8) {
        let da = Device::Cuda(a);
        let db = Device::Cuda(b);
        prop_assert_eq!(da.cmp(&db), a.cmp(&b),
            "Cuda({}) vs Cuda({}) ordering mismatch", a, b);
    }

    /// Hip devices sort by index (analogous to Cuda).
    #[test]
    fn prop_hip_index_ordering(a in 0usize..8, b in 0usize..8) {
        let da = Device::Hip(a);
        let db = Device::Hip(b);
        prop_assert_eq!(da.cmp(&db), a.cmp(&b),
            "Hip({}) vs Hip({}) ordering mismatch", a, b);
    }

    /// OpenCL devices sort by index.
    #[test]
    fn prop_opencl_index_ordering(a in 0usize..8, b in 0usize..8) {
        let da = Device::OpenCL(a);
        let db = Device::OpenCL(b);
        prop_assert_eq!(da.cmp(&db), a.cmp(&b),
            "OpenCL({}) vs OpenCL({}) ordering mismatch", a, b);
    }
}

// ===================================================================
// 4. Device Hash is consistent with Eq
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Equal devices produce the same hash.
    #[test]
    fn prop_device_hash_eq_consistent(d in all_devices()) {
        let d2 = d;
        prop_assert_eq!(d, d2);
        prop_assert_eq!(hash_of(&d), hash_of(&d2),
            "equal devices must have equal hashes: {:?}", d);
    }
}

// ===================================================================
// 5. QuantizationType Display produces documented strings
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Display output matches the exact documented string.
    #[test]
    fn prop_qtype_display_exact(qt in all_qtypes()) {
        let s = qt.to_string();
        let expected = match qt {
            QuantizationType::I2S => "I2_S",
            QuantizationType::TL1 => "TL1",
            QuantizationType::TL2 => "TL2",
        };
        prop_assert_eq!(s.as_str(), expected,
            "Display for {:?} should be {:?}", qt, expected);
    }

    /// Debug output contains the variant name.
    #[test]
    fn prop_qtype_debug_contains_name(qt in all_qtypes()) {
        let dbg = format!("{:?}", qt);
        let name = match qt {
            QuantizationType::I2S => "I2S",
            QuantizationType::TL1 => "TL1",
            QuantizationType::TL2 => "TL2",
        };
        prop_assert!(dbg.contains(name),
            "Debug {:?} should contain {:?}", dbg, name);
    }
}

// ===================================================================
// 6. QuantizationType Hash consistency
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Same variant always hashes to the same value.
    #[test]
    fn prop_qtype_hash_stable(qt in all_qtypes()) {
        let h1 = hash_of(&qt);
        let h2 = hash_of(&qt);
        prop_assert_eq!(h1, h2, "hash not stable for {:?}", qt);
    }

    /// Different variants produce different hashes (not strictly required,
    /// but a strong signal for enum hash quality).
    #[test]
    fn prop_qtype_distinct_variants_different_hash(
        a in all_qtypes(),
        b in all_qtypes(),
    ) {
        if a != b {
            // This is a probabilistic assertion — hash collisions are
            // theoretically possible but extremely unlikely for 3 enum variants.
            prop_assert_ne!(hash_of(&a), hash_of(&b),
                "different variants {:?} and {:?} should likely have different hashes", a, b);
        }
    }
}

// ===================================================================
// 7. Error context: variants include their field data in Display
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// ModelError::NotFound display includes the path.
    #[test]
    fn prop_model_error_not_found_context(path in "[a-z/]{1,40}") {
        let e = ModelError::NotFound { path: path.clone() };
        let msg = e.to_string();
        prop_assert!(msg.contains(&path),
            "ModelError::NotFound display {:?} should contain path {:?}", msg, path);
    }

    /// QuantizationError::InvalidBlockSize display includes the size.
    #[test]
    fn prop_quant_error_block_size_context(size in 0usize..10_000) {
        let e = QuantizationError::InvalidBlockSize { size };
        let msg = e.to_string();
        prop_assert!(msg.contains(&size.to_string()),
            "QuantizationError display {:?} should contain size {}", msg, size);
    }

    /// KernelError::UnsupportedHardware display includes both required and available.
    #[test]
    fn prop_kernel_error_hw_context(
        required in "[a-z]{2,10}",
        available in "[a-z]{2,10}",
    ) {
        let e = KernelError::UnsupportedHardware {
            required: required.clone(),
            available: available.clone(),
        };
        let msg = e.to_string();
        prop_assert!(msg.contains(&required),
            "KernelError display {:?} should contain required {:?}", msg, required);
        prop_assert!(msg.contains(&available),
            "KernelError display {:?} should contain available {:?}", msg, available);
    }

    /// InferenceError::ContextLengthExceeded display includes the length.
    #[test]
    fn prop_inference_error_context_length(length in 0usize..1_000_000) {
        let e = InferenceError::ContextLengthExceeded { length };
        let msg = e.to_string();
        prop_assert!(msg.contains(&length.to_string()),
            "InferenceError display {:?} should contain length {}", msg, length);
    }
}

// ===================================================================
// 8. SecurityError ResourceLimit contains resource, value, and limit
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// ResourceLimit display includes the resource name, value, and limit.
    #[test]
    fn prop_security_resource_limit_context(
        resource in "[a-z_]{2,16}",
        value in 0u64..1_000_000,
        limit in 0u64..1_000_000,
    ) {
        let e = SecurityError::ResourceLimit {
            resource: resource.clone(),
            value,
            limit,
        };
        let msg = e.to_string();
        prop_assert!(msg.contains(&resource),
            "SecurityError display {:?} should contain resource {:?}", msg, resource);
        prop_assert!(msg.contains(&value.to_string()),
            "SecurityError display {:?} should contain value {}", msg, value);
        prop_assert!(msg.contains(&limit.to_string()),
            "SecurityError display {:?} should contain limit {}", msg, limit);
    }

    /// BitNetError wrapping SecurityError preserves the inner message.
    #[test]
    fn prop_bitnet_error_wraps_security(reason in "[a-z ]{2,30}") {
        let inner = SecurityError::MemoryBomb { reason: reason.clone() };
        let outer: BitNetError = inner.into();
        let msg = outer.to_string();
        prop_assert!(msg.contains(&reason),
            "BitNetError display {:?} should contain reason {:?}", msg, reason);
    }
}
