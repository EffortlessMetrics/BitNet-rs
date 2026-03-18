//! CPU weight dequantization operations.
//!
//! Converts quantized (1-bit, 2-bit) packed weight representations back to
//! `f32` for computation.  The encoding follows the I2_S convention used
//! elsewhere in the workspace:
//!
//! | 2-bit code | value |
//! |------------|-------|
//! | `0b00`     |  0    |
//! | `0b01`     | +1    |
//! | `0b11`     | -1    |
//! | `0b10`     | (reserved, treated as 0) |

use bitnet_common::{BitNetError, KernelError, Result};

// ── Decoding helpers ───────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Dequantize a single I2_S block of packed 2-bit weights.
///
/// Each byte stores 4 values (LSB-first, 2 bits each). The output length
/// is `block_size`, and every decoded value is multiplied by `scale`.
///
/// # Errors
///
/// Returns an error when `packed` does not contain enough bytes for
/// `block_size` elements (requires `ceil(block_size / 4)` bytes).
pub fn dequant_i2s_block(packed: &[u8], scale: f32, block_size: usize) -> Result<Vec<f32>> {
    let bytes_needed = block_size.div_ceil(4);
    if packed.len() < bytes_needed {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "dequant_i2s_block: need {} bytes for block_size={}, got {}",
                bytes_needed,
                block_size,
                packed.len()
            ),
        }));
    }

    let mut out = Vec::with_capacity(block_size);
    for i in 0..block_size {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        out.push(decode_i2s(bits) as f32 * scale);
    }
    Ok(out)
}

/// Dequantize ternary-packed weights: each decoded value ∈ {-1, 0, +1}
/// is multiplied by `scale`.
///
/// Identical to [`dequant_i2s_block`] with `block_size = packed.len() * 4`,
/// i.e. it decodes every bit in the input.
pub fn dequant_ternary(packed: &[u8], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        for j in 0..4 {
            let bits = (byte >> (j * 2)) & 0x03;
            out.push(decode_i2s(bits) as f32 * scale);
        }
    }
    out
}

/// Quantize f32 values to ternary {-1, 0, +1} packed representation.
///
/// Values whose absolute magnitude is ≤ `threshold` are mapped to 0;
/// positive values above the threshold become +1 and negative values
/// become -1.  The returned scale is the mean absolute value of the
/// non-zero entries (or 1.0 when all entries are zero).
///
/// The input is zero-padded to a multiple of 4 for packing.
pub fn pack_ternary(values: &[f32], threshold: f32) -> (Vec<u8>, f32) {
    let mut ternary = Vec::with_capacity(values.len());
    let mut abs_sum = 0.0_f32;
    let mut nonzero_count = 0u64;

    for &v in values {
        if v.abs() <= threshold {
            ternary.push(0i8);
        } else if v > 0.0 {
            ternary.push(1);
            abs_sum += v.abs();
            nonzero_count += 1;
        } else {
            ternary.push(-1);
            abs_sum += v.abs();
            nonzero_count += 1;
        }
    }

    let scale = if nonzero_count > 0 { abs_sum / nonzero_count as f32 } else { 1.0 };

    // Pad to multiple of 4.
    while ternary.len() % 4 != 0 {
        ternary.push(0);
    }

    let packed: Vec<u8> = ternary
        .chunks_exact(4)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                byte |= code << (i * 2);
            }
            byte
        })
        .collect();

    (packed, scale)
}

/// Dequantize a full row of I2_S packed weights with per-block scales.
///
/// The row is divided into blocks of `block_size` elements, each with its
/// own scale factor.  `scales` must have `ceil(total_elements / block_size)`
/// entries where `total_elements = packed.len() * 4`.
///
/// # Errors
///
/// Returns an error when `scales` is too short for the number of blocks.
pub fn dequant_i2s_row(packed: &[u8], scales: &[f32], block_size: usize) -> Result<Vec<f32>> {
    if block_size == 0 {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: "dequant_i2s_row: block_size must be > 0".to_string(),
        }));
    }

    let total = packed.len() * 4;
    let num_blocks = total.div_ceil(block_size);

    if scales.len() < num_blocks {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "dequant_i2s_row: need {} scales for {} elements (block_size={}), got {}",
                num_blocks,
                total,
                block_size,
                scales.len()
            ),
        }));
    }

    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let blk = i / block_size;
        let scale = scales[blk];
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        out.push(decode_i2s(bits) as f32 * scale);
    }
    Ok(out)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Pack four ternary values into one byte (mirrors quantized_matmul::pack_i2s).
    fn pack4(vals: [i8; 4]) -> u8 {
        let mut byte = 0u8;
        for (i, &v) in vals.iter().enumerate() {
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            byte |= code << (i * 2);
        }
        byte
    }

    // ── dequant_i2s_block ─────────────────────────────────────────────

    #[test]
    fn test_dequant_i2s_block_known_values() {
        // Byte encodes [+1, -1, 0, +1] at scale 2.0
        let packed = vec![pack4([1, -1, 0, 1])];
        let out = dequant_i2s_block(&packed, 2.0, 4).unwrap();
        assert_eq!(out, vec![2.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_dequant_i2s_block_all_zeros() {
        let packed = [0u8; 2]; // 8 zeros
        let out = dequant_i2s_block(&packed, 5.0, 8).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_dequant_i2s_block_all_plus_one() {
        // 0b01_01_01_01 = 0x55
        let packed = vec![0x55];
        let out = dequant_i2s_block(&packed, 3.0, 4).unwrap();
        assert_eq!(out, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_dequant_i2s_block_all_minus_one() {
        // 0b11_11_11_11 = 0xFF
        let packed = vec![0xFF];
        let out = dequant_i2s_block(&packed, 1.5, 4).unwrap();
        assert_eq!(out, vec![-1.5, -1.5, -1.5, -1.5]);
    }

    #[test]
    fn test_dequant_i2s_block_partial_last_byte() {
        // block_size=3, only first 3 of 4 slots used
        let packed = vec![pack4([1, -1, 1, 0])];
        let out = dequant_i2s_block(&packed, 1.0, 3).unwrap();
        assert_eq!(out, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_dequant_i2s_block_insufficient_bytes() {
        let packed = vec![0u8]; // 1 byte = 4 elements max
        let err = dequant_i2s_block(&packed, 1.0, 8);
        assert!(err.is_err());
    }

    // ── dequant_ternary ───────────────────────────────────────────────

    #[test]
    fn test_dequant_ternary_values_in_set() {
        let packed = vec![pack4([1, -1, 0, 1]), pack4([-1, 0, -1, 1])];
        let out = dequant_ternary(&packed, 1.0);
        for &v in &out {
            assert!(v == -1.0 || v == 0.0 || v == 1.0, "value {v} not in {{-1, 0, +1}}");
        }
    }

    #[test]
    fn test_dequant_ternary_with_scale() {
        let packed = vec![pack4([1, -1, 0, 1])];
        let out = dequant_ternary(&packed, 0.5);
        assert_eq!(out, vec![0.5, -0.5, 0.0, 0.5]);
    }

    #[test]
    fn test_dequant_ternary_alternating_pattern() {
        // Alternating +1 -1 +1 -1
        let packed = vec![pack4([1, -1, 1, -1])];
        let out = dequant_ternary(&packed, 1.0);
        assert_eq!(out, vec![1.0, -1.0, 1.0, -1.0]);
    }

    // ── pack_ternary ──────────────────────────────────────────────────

    #[test]
    fn test_pack_ternary_roundtrip() {
        let values = vec![1.0_f32, -0.8, 0.05, 0.9, -1.2, 0.0, 0.7, -0.3];
        let threshold = 0.1;
        let (packed, scale) = pack_ternary(&values, threshold);
        let deq = dequant_ternary(&packed, scale);

        // Signs must match for non-zero originals above threshold.
        for (i, (&orig, &got)) in values.iter().zip(deq.iter()).enumerate() {
            if orig.abs() > threshold {
                assert_eq!(
                    orig.signum(),
                    got.signum(),
                    "sign mismatch at index {i}: orig={orig}, got={got}"
                );
            } else {
                assert_eq!(got, 0.0, "below-threshold value at {i} should be 0");
            }
        }
    }

    #[test]
    fn test_pack_ternary_all_zeros() {
        let values = [0.0; 8];
        let (packed, scale) = pack_ternary(&values, 0.1);
        assert!(packed.iter().all(|&b| b == 0));
        // Scale defaults to 1.0 when all zero.
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_pack_ternary_padding() {
        // 5 values → padded to 8 (2 bytes)
        let values = vec![1.0, -1.0, 0.0, 1.0, -1.0];
        let (packed, _scale) = pack_ternary(&values, 0.0);
        assert_eq!(packed.len(), 2); // ceil(5/4) padded to 2 bytes
    }

    // ── dequant_i2s_row ───────────────────────────────────────────────

    #[test]
    fn test_dequant_i2s_row_multi_block() {
        // 2 blocks of 4: first block scale=2.0, second block scale=3.0
        let packed = vec![
            pack4([1, -1, 0, 1]), // block 0
            pack4([-1, 1, 1, 0]), // block 1
        ];
        let scales = vec![2.0, 3.0];
        let out = dequant_i2s_row(&packed, &scales, 4).unwrap();
        assert_eq!(out, vec![2.0, -2.0, 0.0, 2.0, -3.0, 3.0, 3.0, 0.0]);
    }

    #[test]
    fn test_dequant_i2s_row_block_size_zero() {
        let packed = [0u8; 4];
        let err = dequant_i2s_row(&packed, &[1.0], 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_dequant_i2s_row_insufficient_scales() {
        let packed = [0u8; 4]; // 16 elements
        // block_size=4 → needs 4 scales, only providing 2
        let err = dequant_i2s_row(&packed, &[1.0, 2.0], 4);
        assert!(err.is_err());
    }

    // ── property tests ────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_pack_dequant_preserves_sign(
                values in proptest::collection::vec(-10.0_f32..10.0, 1..64)
            ) {
                let threshold = 0.01;
                let (packed, scale) = pack_ternary(&values, threshold);
                let deq = dequant_ternary(&packed, scale);

                for (i, (&orig, &got)) in values.iter().zip(deq.iter()).enumerate() {
                    if orig.abs() > threshold {
                        prop_assert_eq!(
                            orig.is_sign_positive(),
                            got.is_sign_positive(),
                            "sign mismatch at {}: orig={}, got={}",
                            i, orig, got
                        );
                    }
                }
            }

            #[test]
            fn prop_dequant_ternary_values_in_range(
                bytes in proptest::collection::vec(0u8..=255, 1..32),
                scale in 0.001_f32..100.0
            ) {
                let out = dequant_ternary(&bytes, scale);
                for &v in &out {
                    prop_assert!(
                        (v - scale).abs() < 1e-5
                            || (v + scale).abs() < 1e-5
                            || v.abs() < 1e-5,
                        "value {} not in {{-scale, 0, +scale}} with scale={}",
                        v, scale
                    );
                }
            }

            #[test]
            fn prop_dequant_i2s_block_length(
                bytes in proptest::collection::vec(0u8..=255, 1..32),
                scale in -100.0_f32..100.0
            ) {
                let block_size = bytes.len() * 4;
                let out = dequant_i2s_block(&bytes, scale, block_size).unwrap();
                prop_assert_eq!(out.len(), block_size);
            }

            #[test]
            fn prop_dequant_i2s_row_length(
                bytes in proptest::collection::vec(0u8..=255, 1..32),
                block_size in 1_usize..=64,
            ) {
                let total = bytes.len() * 4;
                let num_blocks = total.div_ceil(block_size);
                let scales: Vec<f32> = vec![1.0; num_blocks];
                let out = dequant_i2s_row(&bytes, &scales, block_size).unwrap();
                prop_assert_eq!(out.len(), total);
            }
        }
    }
}
