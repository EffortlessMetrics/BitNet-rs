#![no_main]

use arbitrary::Arbitrary;
use bitnet_common::tensor_validation::validate_reshape;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ReshapeInput {
    dims_a: Vec<u16>,
    dims_b: Vec<u16>,
}

fuzz_target!(|input: ReshapeInput| {
    // Limit rank to avoid pathological cases.
    if input.dims_a.is_empty()
        || input.dims_b.is_empty()
        || input.dims_a.len() > 6
        || input.dims_b.len() > 6
    {
        return;
    }

    let shape_a: Vec<usize> = input.dims_a.iter().take(6).map(|&d| (d % 32 + 1) as usize).collect();
    let shape_b: Vec<usize> = input.dims_b.iter().take(6).map(|&d| (d % 32 + 1) as usize).collect();

    let product_a: Option<usize> = shape_a.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let product_b: Option<usize> = shape_b.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));

    let (Some(pa), Some(pb)) = (product_a, product_b) else {
        return;
    };

    // validate_reshape must not panic on any input.
    let result = validate_reshape(&shape_a, &shape_b);

    if pa == pb {
        // Same element count → reshape should succeed.
        assert!(result.is_ok(), "reshape should succeed: {shape_a:?} → {shape_b:?} (product={pa})",);

        // Round-trip: reshaping back to original shape should also succeed.
        let round_trip = validate_reshape(&shape_b, &shape_a);
        assert!(round_trip.is_ok(), "round-trip reshape failed: {shape_b:?} → {shape_a:?}",);
    } else {
        // Different element count → reshape should fail.
        assert!(
            result.is_err(),
            "reshape should fail: {shape_a:?} (product={pa}) → {shape_b:?} (product={pb})",
        );
    }
});
