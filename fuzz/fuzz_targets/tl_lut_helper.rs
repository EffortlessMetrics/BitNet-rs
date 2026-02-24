#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz input structure for TL LUT index calculation
#[derive(Arbitrary, Debug)]
struct LutInput {
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
    lut_len: usize,
}

fuzz_target!(|input: LutInput| {
    // Call the TL LUT index calculator with fuzzed inputs
    let result = bitnet_kernels::tl_lut::lut_index(
        input.block_idx,
        input.elem_in_block,
        input.block_bytes,
        input.elems_per_block,
        input.lut_len,
    );

    // Validate properties regardless of success/failure
    match result {
        Ok(idx) => {
            // Property 1: Index must be within LUT bounds
            assert!(idx < input.lut_len, "Index exceeds LUT length");

            // Property 2: If elems_per_block > 0 and elem_in_block < elems_per_block,
            // we should get a valid result
            if input.elems_per_block > 0 && input.elem_in_block < input.elems_per_block {
                // Property 3: Index should match formula (if no overflow)
                if let Some(base) = input.block_idx.checked_mul(input.block_bytes) {
                    let elem_offset = input.elem_in_block / 8;
                    if let Some(expected) = base.checked_add(elem_offset)
                        && expected < input.lut_len
                    {
                        assert_eq!(
                            idx, expected,
                            "Index mismatch: got {}, expected {}",
                            idx, expected
                        );
                    }
                }
            }

            // Property 4: Monotonicity within a block (increasing elem_in_block should not decrease index)
            if input.elem_in_block + 1 < input.elems_per_block
                && let Ok(next_idx) = bitnet_kernels::tl_lut::lut_index(
                    input.block_idx,
                    input.elem_in_block + 1,
                    input.block_bytes,
                    input.elems_per_block,
                    input.lut_len,
                )
            {
                assert!(
                    next_idx >= idx,
                    "Monotonicity violation: next_idx={} < idx={}",
                    next_idx,
                    idx
                );
            }

            // Property 5: Division by 8 granularity - elements 0-7 map to same byte
            let byte_offset = input.elem_in_block / 8;
            let base_elem = byte_offset * 8;
            if base_elem < input.elems_per_block
                && let Ok(base_idx) = bitnet_kernels::tl_lut::lut_index(
                    input.block_idx,
                    base_elem,
                    input.block_bytes,
                    input.elems_per_block,
                    input.lut_len,
                )
            {
                assert_eq!(
                    idx, base_idx,
                    "Elements {} and {} should map to same byte offset",
                    input.elem_in_block, base_elem
                );
            }
        }
        Err(_) => {
            // Error cases should be well-defined
            // Expected failures:
            // 1. elems_per_block == 0
            // 2. elem_in_block >= elems_per_block
            // 3. Overflow in arithmetic
            // 4. Final index >= lut_len

            // We don't assert on error cases since they're expected for invalid inputs,
            // but we verify the function doesn't panic
        }
    }
});
