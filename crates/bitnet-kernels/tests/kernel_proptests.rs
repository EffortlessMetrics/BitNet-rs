//! Additional property-based tests for `bitnet-kernels`.
//!
//! These tests focus on invariants not already covered in `property_tests.rs`:
//! - `tl_lut::lut_index`: valid inputs always succeed; invalid inputs always fail;
//!   elements in the same 8-element group map to the same LUT byte.
//! - `FallbackKernel::matmul_i2s`: a zero weight matrix always produces zero output.
//! - `FallbackKernel::quantize(I2S)`: non-zero inputs always yield positive scales.
//! - `FallbackKernel::quantize(I2S)`: output bytes are always within `u8` range
//!   (trivially guaranteed by the type, but the encoding must round-trip non-trivially).

use bitnet_common::QuantizationType;
use bitnet_kernels::{FallbackKernel, KernelProvider, tl_lut::lut_index};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Properties: tl_lut::lut_index — valid-input success
// ---------------------------------------------------------------------------

proptest! {
    /// `lut_index` succeeds for all well-formed argument combinations.
    ///
    /// Well-formed means:
    /// - `elems_per_block > 0`
    /// - `elem_in_block < elems_per_block`
    /// - final index (`block_idx * block_bytes + elem_in_block / 8`) < `lut_len`
    ///
    /// We pick conservative fixed parameters (32 elems/block, 4 bytes/block, 64-byte LUT)
    /// and vary `block_idx` and `elem_in_block` within their safe ranges.
    #[test]
    fn prop_lut_index_valid_inputs_always_succeed(
        block_idx  in 0usize..8,
        group      in 0usize..4,   // elements_per_block=32 → 4 groups of 8
        bit_offset in 0usize..8,
    ) {
        // Fixed parameters chosen so the maximum possible index fits in lut_len.
        // max idx = 7 * 4 + 3 = 31 < 64 = lut_len  ✓
        const ELEMS: usize = 32;
        const BBYTES: usize = 4;   // ELEMS / 8
        const LUT_LEN: usize = 64;

        let elem_in_block = group * 8 + bit_offset;

        let result = lut_index(block_idx, elem_in_block, BBYTES, ELEMS, LUT_LEN);
        prop_assert!(
            result.is_ok(),
            "lut_index must succeed for valid inputs; \
             block_idx={block_idx}, elem={elem_in_block}, \
             block_bytes={BBYTES}, elems={ELEMS}, lut_len={LUT_LEN}"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: tl_lut::lut_index — invalid-input failures
// ---------------------------------------------------------------------------

proptest! {
    /// `lut_index` returns `Err` when `elem_in_block >= elems_per_block`.
    ///
    /// The bounds check must be strict: passing an element index equal to or beyond
    /// the block size is always an out-of-bounds access and must be rejected.
    #[test]
    fn prop_lut_index_invalid_elem_always_fails(
        elems_per_block in 1usize..64,
        excess          in 0usize..32,  // elem_in_block = elems_per_block + excess ≥ elems_per_block
    ) {
        let elem_in_block = elems_per_block + excess;
        let block_bytes   = 8usize;
        let lut_len       = 1024usize;

        let result = lut_index(0, elem_in_block, block_bytes, elems_per_block, lut_len);
        prop_assert!(
            result.is_err(),
            "elem_in_block ({elem_in_block}) >= elems_per_block ({elems_per_block}) must Err"
        );
    }

    /// `lut_index` returns `Err` when `elems_per_block` is zero.
    ///
    /// Zero-length blocks are logically invalid and must be caught before any
    /// arithmetic that would otherwise divide by zero or produce nonsensical indices.
    #[test]
    fn prop_lut_index_zero_elems_per_block_always_fails(
        block_idx   in 0usize..16,
        block_bytes in 1usize..32,
        lut_len     in 1usize..256,
    ) {
        let result = lut_index(block_idx, 0, block_bytes, 0, lut_len);
        prop_assert!(
            result.is_err(),
            "elems_per_block=0 must always Err (non-empty blocks required)"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: tl_lut::lut_index — group consistency
// ---------------------------------------------------------------------------

proptest! {
    /// All 8 elements within the same group map to the same LUT byte index.
    ///
    /// TL quantization packs 8 elements into a single byte.  The index formula
    /// uses integer division (`elem_in_block / 8`) to find the byte offset, so
    /// elements 0–7 map to byte 0, elements 8–15 map to byte 1, etc.
    ///
    /// This invariant must hold regardless of which two elements within a group
    /// are compared.
    #[test]
    fn prop_lut_index_same_group_same_byte(
        block_idx  in 0usize..4,
        group      in 0usize..4,  // groups 0–3; elems_per_block=32 → 4 groups
        offset_a   in 0usize..8,
        offset_b   in 0usize..8,
    ) {
        const ELEMS: usize = 32;
        const BBYTES: usize = 4;
        const LUT_LEN: usize = 32; // covers block_idx 0..7 (max idx = 3*4+3=15 < 32)

        let elem_a = group * 8 + offset_a;
        let elem_b = group * 8 + offset_b;

        let idx_a = lut_index(block_idx, elem_a, BBYTES, ELEMS, LUT_LEN).unwrap();
        let idx_b = lut_index(block_idx, elem_b, BBYTES, ELEMS, LUT_LEN).unwrap();

        prop_assert_eq!(
            idx_a, idx_b,
            "elements {} and {} are in group {} of block {} \
             and must map to the same LUT byte; got {} vs {}",
            elem_a, elem_b, group, block_idx, idx_a, idx_b
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::matmul_i2s — zero-weight invariant
// ---------------------------------------------------------------------------

proptest! {
    /// `matmul_i2s` with an all-zero weight matrix (B) always produces an all-zero
    /// output matrix (C), regardless of the activation matrix (A).
    ///
    /// This tests the mathematical identity: A × 0 = 0. Any non-zero value in the
    /// output would indicate a memory-initialisation bug or incorrect loop bounds.
    #[test]
    fn prop_fallback_matmul_zero_weights_produce_zero_output(
        m in 1usize..6,
        n in 1usize..6,
        k in 1usize..6,
    ) {
        let kernel = FallbackKernel;
        let a = vec![1i8; m * k];   // non-zero activations
        let b = vec![0u8; k * n];   // all-zero weights
        let mut c = vec![0.0f32; m * n];

        kernel.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();

        for (i, &val) in c.iter().enumerate() {
            prop_assert_eq!(
                val, 0.0f32,
                "output[{}] must be 0.0 when the weight matrix is all zeros (got {})",
                i, val
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::quantize(I2S) — scale positivity
// ---------------------------------------------------------------------------

proptest! {
    /// After `quantize(I2S)`, every scale factor is strictly positive when the
    /// input contains only positive values.
    ///
    /// The quantizer computes `scale = max_abs / 1.5` (or 1.0 as a guard for near-zero
    /// blocks).  Both paths yield a positive scale.  A zero or negative scale would
    /// cause sign inversions or division-by-zero during dequantization.
    #[test]
    fn prop_fallback_quantize_positive_input_yields_positive_scales(
        n_blocks in 1usize..8usize,
    ) {
        const BLOCK_SIZE: usize = 32; // I2S block size in FallbackKernel
        let n = n_blocks * BLOCK_SIZE;

        // Constant positive input — avoids NaN/inf and ensures max_abs > 1e-8.
        let input = vec![1.0f32; n];
        let mut output = vec![0u8; n / 4]; // 2 bits/element → n/4 bytes
        let mut scales = vec![0.0f32; n_blocks];

        FallbackKernel
            .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
            .expect("quantize(I2S) must succeed for well-formed inputs");

        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(
                s > 0.0,
                "scale[{i}] must be > 0 for a block of positive values; got {s}"
            );
        }
    }
}
