#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_validation::{broadcast_shape, can_broadcast, validate_matmul_shapes};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ShapeInput {
    a_dims: Vec<u8>,
    b_dims: Vec<u8>,
}

fuzz_target!(|input: ShapeInput| {
    // Cap rank at 8 dimensions and dimension size at 64 to avoid OOM.
    let a: Vec<usize> = input.a_dims.iter().take(8).map(|&d| (d as usize % 64) + 1).collect();
    let b: Vec<usize> = input.b_dims.iter().take(8).map(|&d| (d as usize % 64) + 1).collect();

    // --- broadcast_shape ---
    let result = broadcast_shape(&a, &b);
    let compat = can_broadcast(&a, &b);

    // Invariant 1: broadcast_shape and can_broadcast agree.
    assert_eq!(result.is_ok(), compat, "broadcast_shape vs can_broadcast disagree");

    if let Ok(ref out) = result {
        // Invariant 2: Output rank is max(rank_a, rank_b).
        assert_eq!(out.len(), a.len().max(b.len()), "output rank should be max of input ranks");

        // Invariant 3: Each output dim >= corresponding input dim (after right-alignment).
        let max_ndim = out.len();
        for i in 0..max_ndim {
            let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
            let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };
            assert!(
                out[i] >= da && out[i] >= db,
                "output dim {i}: {} < input dims ({da}, {db})",
                out[i]
            );
        }

        // Invariant 4: Broadcasting is commutative.
        let reverse = broadcast_shape(&b, &a);
        assert_eq!(
            reverse.as_ref().map(|v| v.as_slice()),
            Ok(out.as_slice()),
            "broadcast_shape should be commutative"
        );

        // Invariant 5: Broadcasting with self is identity.
        let self_broadcast = broadcast_shape(&a, &a).unwrap();
        assert_eq!(self_broadcast, a, "broadcast(a, a) should equal a");
    }

    // --- validate_matmul_shapes ---
    if !a.is_empty() && !b.is_empty() {
        let matmul_result = validate_matmul_shapes(&a, &b);

        if let Ok(ref out_shape) = matmul_result {
            // Invariant 6: Output is non-empty.
            assert!(!out_shape.is_empty(), "matmul output shape should be non-empty");

            // Invariant 7: For 2-D inputs, output is [a_rows, b_cols].
            if a.len() == 2 && b.len() == 2 {
                assert_eq!(out_shape.len(), 2);
                assert_eq!(out_shape[0], a[0], "matmul rows mismatch");
                assert_eq!(out_shape[1], b[1], "matmul cols mismatch");
            }
        }
        // Errors are expected for incompatible shapes — no panic is the invariant.
    }

    // --- Edge cases: empty shapes ---
    let empty: Vec<usize> = vec![];
    let _ = broadcast_shape(&empty, &b);
    let _ = broadcast_shape(&a, &empty);
    let _ = broadcast_shape(&empty, &empty);
    // Matmul with empty shapes should return Err, not panic.
    assert!(validate_matmul_shapes(&empty, &b).is_err());
    assert!(validate_matmul_shapes(&a, &empty).is_err());
});
