#![no_main]
use arbitrary::Arbitrary;
use bitnet_common::tensor_validation::{
    broadcast_shape, can_broadcast, validate_attention_shapes, validate_matmul_shapes,
    validate_reshape, validate_transpose_axes,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TensorValidationInput {
    shape_a: Vec<u16>,
    shape_b: Vec<u16>,
    shape_c: Vec<u16>,
    axes: Vec<u8>,
}

fuzz_target!(|input: TensorValidationInput| {
    let a: Vec<usize> = input.shape_a.iter().take(8).map(|&d| (d % 64) as usize).collect();
    let b: Vec<usize> = input.shape_b.iter().take(8).map(|&d| (d % 64) as usize).collect();
    let c: Vec<usize> = input.shape_c.iter().take(8).map(|&d| (d % 64) as usize).collect();

    // Broadcasting
    let _ = broadcast_shape(&a, &b);
    let _ = can_broadcast(&a, &b);

    // Matmul
    let _ = validate_matmul_shapes(&a, &b);

    // Reshape
    let _ = validate_reshape(&a, &b);

    // Transpose
    let axes: Vec<usize> = input.axes.iter().take(8).map(|&x| x as usize).collect();
    let _ = validate_transpose_axes(&a, &axes);

    // Attention (expects 4-D shapes)
    if a.len() == 4 && b.len() == 4 && c.len() == 4 {
        let _ = validate_attention_shapes(&a, &b, &c);
    }
});
