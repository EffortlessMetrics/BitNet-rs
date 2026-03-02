//! Property-based tests for shape validation, tensor size calculations,
//! and Device enum serde round-trips (proptest wave 31).

use bitnet_common::tensor_validation::{
    broadcast_shape, can_broadcast, validate_matmul_shapes, validate_reshape,
};
use bitnet_common::types::Device;
use proptest::prelude::*;

// ── Helper strategies ─────────────────────────────────────────────────────────

fn small_shape(max_ndim: usize, max_dim: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=max_dim, 0..=max_ndim)
}

fn nonzero_shape(ndim: usize, max_dim: usize) -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=max_dim, ndim)
}

fn device_strategy() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..8).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
}

// ── Broadcasting rules ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Broadcasting a shape with itself always succeeds and returns the same shape.
    #[test]
    fn broadcast_self_is_identity(shape in small_shape(4, 16)) {
        let result = broadcast_shape(&shape, &shape);
        prop_assert!(result.is_ok(), "broadcasting self should succeed: {shape:?}");
        prop_assert_eq!(result.unwrap(), shape);
    }

    /// Broadcasting is commutative: broadcast(a,b) == broadcast(b,a).
    #[test]
    fn broadcast_is_commutative(
        a in small_shape(4, 8),
        b in small_shape(4, 8),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(r1), Ok(r2)) => prop_assert_eq!(r1, r2),
            (Err(_), Err(_)) => {} // both fail — fine
            _ => prop_assert!(false, "broadcast commutativity violated"),
        }
    }

    /// Broadcasting with a scalar (empty shape) always succeeds.
    #[test]
    fn broadcast_with_scalar(shape in small_shape(4, 16)) {
        let result = broadcast_shape(&shape, &[]);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), shape);
    }

    /// Broadcasting with all-ones shape succeeds and returns the other shape.
    #[test]
    fn broadcast_with_ones(shape in nonzero_shape(3, 16)) {
        let ones = vec![1usize; shape.len()];
        let result = broadcast_shape(&shape, &ones);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), shape);
    }

    /// can_broadcast agrees with broadcast_shape.
    #[test]
    fn can_broadcast_consistent(
        a in small_shape(4, 8),
        b in small_shape(4, 8),
    ) {
        let result = broadcast_shape(&a, &b);
        prop_assert_eq!(can_broadcast(&a, &b), result.is_ok());
    }

    /// Output ndim is max(a.ndim, b.ndim) when broadcast succeeds.
    #[test]
    fn broadcast_output_ndim(
        a in nonzero_shape(3, 8),
        b in nonzero_shape(3, 8),
    ) {
        if let Ok(out) = broadcast_shape(&a, &b) {
            let expected_ndim = a.len().max(b.len());
            prop_assert_eq!(out.len(), expected_ndim);
        }
    }

    /// Each output dim is >= max(corresponding input dims).
    #[test]
    fn broadcast_output_dims_ge_inputs(
        a in nonzero_shape(3, 8),
        b in nonzero_shape(3, 8),
    ) {
        if let Ok(out) = broadcast_shape(&a, &b) {
            let max_ndim = a.len().max(b.len());
            for i in 0..max_ndim {
                let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
                let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };
                prop_assert!(out[i] >= da);
                prop_assert!(out[i] >= db);
            }
        }
    }

    /// Incompatible dimensions (neither equal nor 1) must fail.
    #[test]
    fn broadcast_incompatible_fails(
        base in 2usize..16,
        diff in 1usize..8,
    ) {
        let a_dim = base;
        let b_dim = base + diff;
        // Neither is 1, and they differ
        let result = broadcast_shape(&[a_dim], &[b_dim]);
        prop_assert!(result.is_err());
    }
}

// ── Tensor size (element count) calculations ──────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Element count is the product of all dimensions.
    #[test]
    fn elem_count_product(dims in nonzero_shape(4, 32)) {
        let expected: usize = dims.iter().product();
        prop_assert_eq!(dims.iter().product::<usize>(), expected);
        prop_assert!(expected >= 1);
    }

    /// A scalar (empty shape) has 1 element.
    #[test]
    fn scalar_has_one_element(_dummy in 0u8..1) {
        let shape: Vec<usize> = vec![];
        let count: usize = shape.iter().product();
        // Product of empty iterator is 1 by convention
        prop_assert_eq!(count, 1);
    }

    /// Reshape preserves element count.
    #[test]
    fn reshape_preserves_elem_count(
        d1 in 1usize..=16,
        d2 in 1usize..=16,
    ) {
        let from = vec![d1, d2];
        let total = d1 * d2;
        let to = vec![total];
        let result = validate_reshape(&from, &to);
        prop_assert!(result.is_ok(), "flatten reshape should succeed");
    }

    /// Reshape with mismatched element counts must fail.
    #[test]
    fn reshape_mismatched_fails(
        d1 in 2usize..=8,
        d2 in 2usize..=8,
        extra in 1usize..=4,
    ) {
        let from = vec![d1, d2];
        let total = d1 * d2 + extra;
        let to = vec![total];
        let result = validate_reshape(&from, &to);
        prop_assert!(result.is_err());
    }
}

// ── Matmul shape validation ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Standard 2-D matmul with compatible inner dims succeeds.
    #[test]
    fn matmul_compatible_2d(
        m in 1usize..=32,
        k in 1usize..=32,
        n in 1usize..=32,
    ) {
        let a = vec![m, k];
        let b = vec![k, n];
        let result = validate_matmul_shapes(&a, &b);
        prop_assert!(result.is_ok());
        let out = result.unwrap();
        prop_assert_eq!(out, vec![m, n]);
    }

    /// Matmul with mismatched inner dims fails.
    #[test]
    fn matmul_mismatched_inner(
        m in 1usize..=16,
        k1 in 1usize..=16,
        k2 in 1usize..=16,
        n in 1usize..=16,
    ) {
        prop_assume!(k1 != k2);
        let a = vec![m, k1];
        let b = vec![k2, n];
        let result = validate_matmul_shapes(&a, &b);
        prop_assert!(result.is_err());
    }

    /// 1-D dot product succeeds when lengths match.
    #[test]
    fn matmul_1d_dot(k in 1usize..=64) {
        let result = validate_matmul_shapes(&[k], &[k]);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), Vec::<usize>::new());
    }
}

// ── Device serde round-trip ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Device round-trips through JSON serialization.
    #[test]
    fn device_serde_roundtrip(device in device_strategy()) {
        let json = serde_json::to_string(&device).expect("serialize");
        let back: Device = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(device, back);
    }

    /// Serialized Device is always valid JSON.
    #[test]
    fn device_serde_valid_json(device in device_strategy()) {
        let json = serde_json::to_string(&device).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        prop_assert!(!parsed.is_null());
    }

    /// Device ordering is consistent with equality.
    #[test]
    fn device_ord_consistent(d1 in device_strategy(), d2 in device_strategy()) {
        if d1 == d2 {
            prop_assert!(d1.cmp(&d2) == std::cmp::Ordering::Equal);
        }
    }
}

// ── QuantizationType serde round-trip ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// QuantizationType round-trips through JSON.
    #[test]
    fn quantization_type_serde_roundtrip(
        qt in prop_oneof![
            Just(bitnet_common::QuantizationType::I2S),
            Just(bitnet_common::QuantizationType::TL1),
            Just(bitnet_common::QuantizationType::TL2),
        ]
    ) {
        let json = serde_json::to_string(&qt).expect("serialize");
        let back: bitnet_common::QuantizationType = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(qt, back);
    }
}
