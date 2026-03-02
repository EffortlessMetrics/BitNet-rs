#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_validation::{
    broadcast_shape, can_broadcast, validate_matmul_shapes, validate_reshape,
    validate_transpose_axes,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReshapeInput {
    from_shape: Vec<u16>,
    to_shape: Vec<u16>,
    other_shape: Vec<u16>,
    axes: Vec<u8>,
}

fuzz_target!(|input: ReshapeInput| {
    let from: Vec<usize> = input.from_shape.iter().take(8).map(|&d| (d % 256) as usize).collect();
    let to: Vec<usize> = input.to_shape.iter().take(8).map(|&d| (d % 256) as usize).collect();
    let other: Vec<usize> = input.other_shape.iter().take(8).map(|&d| (d % 256) as usize).collect();

    // Reshape: must never panic, only return Ok/Err.
    let _ = validate_reshape(&from, &to);

    // Empty shapes must not panic.
    let _ = validate_reshape(&[], &to);
    let _ = validate_reshape(&from, &[]);
    let _ = validate_reshape(&[], &[]);

    // Shapes with zeros must not panic.
    let zero_shape = vec![0usize; from.len()];
    let _ = validate_reshape(&zero_shape, &to);
    let _ = validate_reshape(&from, &zero_shape);

    // Broadcast must not panic.
    let _ = broadcast_shape(&from, &other);
    let _ = can_broadcast(&from, &other);

    // Matmul shapes must not panic.
    let _ = validate_matmul_shapes(&from, &other);

    // Transpose axes must not panic.
    let axes: Vec<usize> = input.axes.iter().take(8).map(|&x| x as usize).collect();
    let _ = validate_transpose_axes(&from, &axes);
});
