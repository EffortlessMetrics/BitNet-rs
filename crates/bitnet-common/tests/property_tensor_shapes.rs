//! Property-based tests for tensor shape invariants.
//!
//! Invariants verified:
//! - Total element count equals the product of all dimensions
//! - Zero in any dimension implies zero total elements
//! - Shape is preserved exactly through MockTensor construction
//! - Dimension ordering is stable (no reordering on construction)
//! - Ceil-div helper produces correct non-negative results

use bitnet_common::math::ceil_div;
use bitnet_common::tensor::{MockTensor, Tensor};
use bitnet_common::types::Device;
use proptest::prelude::*;

// ── Element count invariants ────────────────────────────────────────────────

proptest! {
    /// Total element count equals the product of all dimensions.
    #[test]
    fn element_count_is_product_of_dims(
        dims in proptest::collection::vec(1usize..64, 1..=4),
    ) {
        let expected: usize = dims.iter().product();
        let t = MockTensor::new(dims.clone());
        let count: usize = t.shape().iter().product();
        prop_assert_eq!(
            count, expected,
            "shape {:?}: product {} != expected {}",
            dims, count, expected
        );
    }

    /// A zero dimension anywhere produces zero total elements.
    #[test]
    fn zero_dim_means_zero_elements(
        pre in proptest::collection::vec(1usize..32, 0..=2),
        post in proptest::collection::vec(1usize..32, 0..=2),
    ) {
        let mut dims = pre;
        dims.push(0);
        dims.extend(post);
        let t = MockTensor::new(dims.clone());
        let count: usize = t.shape().iter().product();
        prop_assert_eq!(count, 0, "shape {:?} should have 0 elements", dims);
    }

    /// Scalar tensor (empty dims) has exactly 1 element by convention
    /// (product of empty sequence = 1).
    #[test]
    fn scalar_shape_has_one_element(_dummy in 0u8..1) {
        let t = MockTensor::new(vec![]);
        let count: usize = t.shape().iter().product::<usize>().max(1);
        prop_assert_eq!(count, 1, "scalar tensor should have 1 element");
    }
}

// ── Shape preservation ──────────────────────────────────────────────────────

proptest! {
    /// MockTensor preserves the exact shape passed to the constructor.
    #[test]
    fn shape_preserved_on_construction(
        dims in proptest::collection::vec(1usize..128, 1..=4),
    ) {
        let t = MockTensor::new(dims.clone());
        prop_assert_eq!(t.shape(), dims.as_slice());
    }

    /// Dimension ordering is stable: shape[i] == dims[i] for all i.
    #[test]
    fn dimension_ordering_stable(
        dims in proptest::collection::vec(1usize..128, 2..=4),
    ) {
        let t = MockTensor::new(dims.clone());
        for (i, &d) in dims.iter().enumerate() {
            prop_assert_eq!(t.shape()[i], d);
        }
    }

    /// Number of dimensions (ndim) equals the length of the input shape.
    #[test]
    fn ndim_equals_shape_len(
        ndim in 0usize..=5,
    ) {
        let dims: Vec<usize> = (0..ndim).map(|_| 1).collect();
        let t = MockTensor::new(dims);
        prop_assert_eq!(t.shape().len(), ndim);
    }
}

// ── Device assignment ───────────────────────────────────────────────────────

proptest! {
    /// Default device is CPU.
    #[test]
    fn default_device_is_cpu(
        dims in proptest::collection::vec(1usize..16, 1..=3),
    ) {
        let t = MockTensor::new(dims);
        prop_assert_eq!(t.device(), &Device::Cpu);
    }

    /// with_device overrides the device correctly.
    #[test]
    fn with_device_overrides(
        dims in proptest::collection::vec(1usize..16, 1..=3),
        gpu_idx in 0usize..4,
    ) {
        let t = MockTensor::new(dims.clone()).with_device(Device::Cuda(gpu_idx));
        prop_assert_eq!(t.device(), &Device::Cuda(gpu_idx));
        // Shape is unaffected by device change.
        prop_assert_eq!(t.shape(), dims.as_slice());
    }
}

// ── Ceil-div helper ─────────────────────────────────────────────────────────

proptest! {
    /// ceil_div(a, b) >= a / b for all positive b.
    #[test]
    fn ceil_div_ge_floor_div(
        a in 0usize..10_000,
        b in 1usize..1_000,
    ) {
        let cd = ceil_div(a, b);
        let fd = a / b;
        prop_assert!(cd >= fd, "ceil_div({}, {}) = {} < floor_div = {}", a, b, cd, fd);
    }

    /// ceil_div(a, b) * b >= a (covers all elements).
    #[test]
    fn ceil_div_covers_all_elements(
        a in 0usize..10_000,
        b in 1usize..1_000,
    ) {
        let cd = ceil_div(a, b);
        prop_assert!(
            cd * b >= a,
            "ceil_div({}, {}) * {} = {} < {}",
            a, b, b, cd * b, a
        );
    }

    /// ceil_div(a, b) is exact when a is a multiple of b.
    #[test]
    fn ceil_div_exact_for_multiples(
        quotient in 0usize..1_000,
        b in 1usize..1_000,
    ) {
        let a = quotient * b;
        prop_assert_eq!(ceil_div(a, b), quotient);
    }

    /// ceil_div result is bounded: never exceeds a + b - 1 for non-negative a.
    #[test]
    fn ceil_div_bounded(
        a in 0usize..100_000,
        b in 1usize..10_000,
    ) {
        let cd = ceil_div(a, b);
        prop_assert!(
            cd <= a.saturating_add(b),
            "ceil_div({}, {}) = {} unreasonably large",
            a, b, cd
        );
    }
}
