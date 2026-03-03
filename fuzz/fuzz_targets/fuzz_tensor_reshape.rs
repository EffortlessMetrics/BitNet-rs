#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_validation::{
    broadcast_shape, validate_matmul_shapes, validate_reshape, validate_transpose_axes,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReshapeInput {
    /// Source dimensions (raw u16 values, clamped to sane range).
    from_dims: Vec<u16>,
    /// Target dimensions (raw u16 values).
    to_dims: Vec<u16>,
    /// Extra shape for chained operations.
    extra_dims: Vec<u16>,
    /// Axes for transpose validation.
    axes: Vec<u8>,
    /// Whether to test element-count-preserving reshapes.
    test_valid_reshape: bool,
}

fn to_shape(raw: &[u16], max_rank: usize) -> Vec<usize> {
    raw.iter().take(max_rank).map(|&d| (d % 128) as usize).collect()
}

fuzz_target!(|input: ReshapeInput| {
    let from = to_shape(&input.from_dims, 6);
    let to = to_shape(&input.to_dims, 6);

    // validate_reshape must not panic for any shape combination
    let result = validate_reshape(&from, &to);

    // If element counts match, reshape must succeed
    if input.test_valid_reshape && !from.is_empty() && !to.is_empty() {
        let from_count: usize = from.iter().product();
        let to_count: usize = to.iter().product();
        if from_count > 0 && from_count == to_count {
            assert!(
                result.is_ok(),
                "reshape should succeed when element counts match: {from:?} -> {to:?}"
            );
        }
    }

    // If element counts differ, reshape must fail
    if !from.is_empty() && !to.is_empty() {
        let from_count: usize = from.iter().product();
        let to_count: usize = to.iter().product();
        if from_count != to_count && from_count > 0 && to_count > 0 {
            assert!(result.is_err(), "reshape should fail when counts differ: {from:?} -> {to:?}");
        }
    }

    // Chained reshapes: from -> to -> extra must preserve element count
    let extra = to_shape(&input.extra_dims, 6);
    if let Ok(()) = validate_reshape(&from, &to) {
        if let Ok(()) = validate_reshape(&to, &extra) {
            // Transitive: from -> extra must also be valid
            assert!(
                validate_reshape(&from, &extra).is_ok(),
                "transitive reshape failed: {from:?} -> {to:?} -> {extra:?}"
            );
        }
    }

    // broadcast_shape must not panic
    let _ = broadcast_shape(&from, &to);

    // validate_matmul_shapes must not panic
    let _ = validate_matmul_shapes(&from, &to);

    // validate_transpose_axes must not panic
    let axes: Vec<usize> = input.axes.iter().take(6).map(|&x| x as usize).collect();
    let _ = validate_transpose_axes(&from, &axes);

    // Identity reshape: same shape must always succeed
    if !from.is_empty() && from.iter().all(|&d| d > 0) {
        assert!(validate_reshape(&from, &from).is_ok(), "identity reshape must succeed: {from:?}");
    }

    // Flatten: any shape to [total_elements] must succeed
    if !from.is_empty() {
        let total: usize = from.iter().product();
        if total > 0 {
            assert!(
                validate_reshape(&from, &[total]).is_ok(),
                "flatten reshape must succeed: {from:?} -> [{total}]"
            );
        }
    }
});
