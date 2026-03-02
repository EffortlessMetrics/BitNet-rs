//! Fuzz shape validation with random shapes and parameters.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ShapeInput {
    shape_a: Vec<u16>,
    shape_b: Vec<u16>,
    dim: u8,
    expected_size: u16,
    expected_rank: u8,
    expected_elements: u32,
    hidden_size: u16,
    num_heads: u8,
}

fuzz_target!(|input: ShapeInput| {
    // Cap dimensions to avoid huge allocations
    let a: Vec<usize> = input.shape_a.iter().take(8).map(|&v| v as usize).collect();
    let b: Vec<usize> = input.shape_b.iter().take(8).map(|&v| v as usize).collect();

    // shape equality — must not panic regardless of input
    let _ = bitnet_common::shape_validator::assert_shape_eq("fuzz", &a, &b);

    // rank check
    let _ = bitnet_common::shape_validator::assert_rank("fuzz", &a, input.expected_rank as usize);

    // dimension check
    let _ = bitnet_common::shape_validator::assert_dim(
        "fuzz",
        &a,
        input.dim as usize,
        input.expected_size as usize,
    );

    // matmul compatibility
    let _ = bitnet_common::shape_validator::assert_matmul_compat("fuzz", &a, &b);

    // broadcastable
    let _ = bitnet_common::shape_validator::assert_broadcastable("fuzz", &a, &b);

    // element count
    let _ = bitnet_common::shape_validator::assert_element_count(
        "fuzz",
        &a,
        input.expected_elements as usize,
    );

    // head divisibility
    let _ = bitnet_common::shape_validator::assert_head_divisible(
        "fuzz",
        input.hidden_size as usize,
        input.num_heads as usize,
    );
});
