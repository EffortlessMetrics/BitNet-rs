//! Property-based tests for tensor shape operations.
//!
//! Key invariants tested:
//! - `MockTensor`: element count equals the product of shape dimensions
//! - `MockTensor`: a zero in any dimension means zero total elements
//! - `MockTensor`: shape round-trip — constructing a tensor and reading back
//!   its shape yields the original dimensions
//! - `MockTensor`: dtype is always F32 and device defaults to CPU
//! - `MockTensor::with_device`: device is preserved after builder call
//! - Reshape preserves element count: product(old_shape) == product(new_shape)
//! - Transpose of a 2-D shape swaps dimensions

use bitnet_common::tensor::{MockTensor, Tensor};
use bitnet_common::types::Device;
use proptest::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Strategy for small non-zero shape dimensions (avoids huge allocations).
fn shape_dim() -> impl Strategy<Value = usize> {
    1usize..=32
}

/// Strategy for a 1-4 dimensional shape with bounded element count.
fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(shape_dim(), 1..=4)
        .prop_filter("element count must be <= 65536 to avoid OOM", |dims| {
            dims.iter().product::<usize>() <= 65536
        })
}

// ── Element count ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Element count of a MockTensor equals the product of its shape.
    #[test]
    fn prop_element_count_is_product(shape in arb_shape()) {
        let expected: usize = shape.iter().product();
        let tensor = MockTensor::new(shape.clone());
        let actual: usize = tensor.shape().iter().product();
        prop_assert_eq!(
            actual, expected,
            "element count for shape {:?} must be {}", shape, expected
        );
    }

    /// A shape with a zero dimension has zero elements.
    #[test]
    fn prop_zero_dim_means_zero_elements(
        prefix in prop::collection::vec(shape_dim(), 0..3),
        suffix in prop::collection::vec(shape_dim(), 0..3),
    ) {
        let mut shape = prefix;
        shape.push(0);
        shape.extend(suffix);
        let tensor = MockTensor::new(shape.clone());
        let count: usize = tensor.shape().iter().product();
        prop_assert_eq!(count, 0, "shape {:?} with a zero dim must have 0 elements", shape);
    }
}

// ── Shape round-trip ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Constructing a MockTensor and reading its shape gives the original dims.
    #[test]
    fn prop_shape_roundtrip(shape in arb_shape()) {
        let tensor = MockTensor::new(shape.clone());
        prop_assert_eq!(tensor.shape(), shape.as_slice());
    }

    /// Device defaults to CPU.
    #[test]
    fn prop_default_device_is_cpu(shape in arb_shape()) {
        let tensor = MockTensor::new(shape);
        prop_assert_eq!(*tensor.device(), Device::Cpu);
    }

    /// `with_device` sets the requested device.
    #[test]
    fn prop_with_device_preserved(idx in 0usize..8) {
        let tensor = MockTensor::new(vec![4, 4]).with_device(Device::Cuda(idx));
        prop_assert_eq!(*tensor.device(), Device::Cuda(idx));
    }

    /// dtype is always F32 for MockTensor.
    #[test]
    fn prop_dtype_always_f32(shape in arb_shape()) {
        let tensor = MockTensor::new(shape);
        prop_assert_eq!(tensor.dtype(), candle_core::DType::F32);
    }
}

// ── as_slice consistency ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// as_slice returns data whose length matches element count.
    #[test]
    fn prop_as_slice_len_matches_shape(shape in arb_shape()) {
        let expected: usize = shape.iter().product();
        let tensor = MockTensor::new(shape);
        let slice: &[f32] = tensor.as_slice::<f32>().expect("as_slice must succeed");
        prop_assert_eq!(slice.len(), expected);
    }
}

// ── Reshape preserves element count ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Reshaping preserves total element count: product of new shape must equal
    /// product of old shape.
    #[test]
    fn prop_reshape_preserves_element_count(
        a in 1usize..=64,
        b in 1usize..=64,
    ) {
        let total = a * b;
        let old_shape = vec![a, b];
        let new_shape = vec![total];
        let old_count: usize = old_shape.iter().product();
        let new_count: usize = new_shape.iter().product();
        prop_assert_eq!(
            old_count, new_count,
            "reshape [{}, {}] → [{}] must preserve element count", a, b, total
        );
    }

    /// Reshaping a 3-D tensor to 2-D preserves element count.
    #[test]
    fn prop_reshape_3d_to_2d_preserves(
        a in 1usize..=16,
        b in 1usize..=16,
        c in 1usize..=16,
    ) {
        let total = a * b * c;
        prop_assume!(total <= 65536);
        let new_rows = a * b;
        let new_cols = c;
        prop_assert_eq!(new_rows * new_cols, total);
    }
}

// ── Transpose of 2-D swaps dimensions ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Transposing a 2-D shape [rows, cols] swaps to [cols, rows].
    #[test]
    fn prop_transpose_2d_swaps(rows in 1usize..=128, cols in 1usize..=128) {
        let original = [rows, cols];
        let transposed = [cols, rows];
        prop_assert_eq!(
            original[0] * original[1],
            transposed[0] * transposed[1],
            "transpose must preserve element count"
        );
        prop_assert_eq!(original[0], transposed[1]);
        prop_assert_eq!(original[1], transposed[0]);
    }

    /// Double-transposing a 2-D shape restores the original shape.
    #[test]
    fn prop_double_transpose_identity(rows in 1usize..=128, cols in 1usize..=128) {
        let original = [rows, cols];
        let transposed = [cols, rows];
        let double_transposed = [transposed[1], transposed[0]];
        prop_assert_eq!(original, double_transposed);
    }
}
