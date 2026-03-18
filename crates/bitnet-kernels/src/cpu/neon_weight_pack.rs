//! NEON-optimized weight packing/unpacking for 1-bit and 2-bit
//! quantized neural network weights on Apple Silicon.
//!
//! Provides six operations:
//!
//! 1. `pack_ternary_weights` — pack {-1,0,+1} → 2-bit (4 per byte)
//! 2. `unpack_ternary_weights` — 2-bit → f32
//! 3. `pack_binary_weights` — pack {-1,+1} → 1-bit (8 per byte)
//! 4. `unpack_binary_weights` — 1-bit → f32
//! 5. `repack_for_simd` — reorder packed weights for tile-friendly
//!    SIMD access
//! 6. `pack_with_scale` — pack with per-block scale factors
//!
//! ## Ternary encoding (2 bits per value, LSB-first)
//!
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//!
//! ## Binary encoding (1 bit per value, LSB-first)
//!
//! - `0` → −1
//! - `1` → +1

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar helpers ─────────────────────────────────────────────────

/// Encode a ternary value into 2-bit representation.
#[inline(always)]
fn encode_ternary(v: f32) -> u8 {
    if v > 0.5 {
        0b01
    } else if v < -0.5 {
        0b11
    } else {
        0b00
    }
}

/// Decode a 2-bit code to f32.
#[inline(always)]
fn decode_ternary(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

/// Required packed length for ternary (4 values per byte).
#[inline]
pub fn ternary_packed_len(n: usize) -> usize {
    n.div_ceil(4)
}

/// Required packed length for binary (8 values per byte).
#[inline]
pub fn binary_packed_len(n: usize) -> usize {
    n.div_ceil(8)
}

// ═══════════════════════════════════════════════════════════════════
// 1. pack_ternary_weights
// ═══════════════════════════════════════════════════════════════════

/// Scalar implementation of ternary packing.
fn scalar_pack_ternary(weights: &[f32], packed: &mut [u8]) {
    let required = ternary_packed_len(weights.len());
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );
    weights.chunks(4).zip(packed.iter_mut()).for_each(|(chunk, byte)| {
        let mut val = 0u8;
        chunk.iter().enumerate().for_each(|(j, &w)| {
            val |= encode_ternary(w) << (j * 2);
        });
        *byte = val;
    });
}

/// NEON-accelerated ternary packing.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_pack_ternary(weights: &[f32], packed: &mut [u8]) {
    let required = ternary_packed_len(weights.len());
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );

    let n = weights.len();
    let mut wi = 0;
    let mut pi = 0;
    let pos_thresh = vdupq_n_f32(0.5);
    let neg_thresh = vdupq_n_f32(-0.5);

    // Process 16 weights → 4 packed bytes per iteration.
    while wi + 16 <= n {
        let mut out_bytes = [0u8; 4];
        (0..4).for_each(|k| {
            let off = wi + k * 4;
            // SAFETY: off + 4 <= wi + 16 <= n, in bounds.
            let v = unsafe { vld1q_f32(weights.as_ptr().add(off)) };
            let is_pos = vcgtq_f32(v, pos_thresh);
            let is_neg = vcltq_f32(v, neg_thresh);

            let mut bits = [0u32; 4];
            let mut negs = [0u32; 4];
            // SAFETY: writing to local arrays.
            unsafe {
                vst1q_u32(bits.as_mut_ptr(), is_pos);
                vst1q_u32(negs.as_mut_ptr(), is_neg);
            }

            let mut byte_val = 0u8;
            bits.iter().zip(negs.iter()).enumerate().for_each(|(j, (&p, &ng))| {
                let code = if p != 0 {
                    0b01u8
                } else if ng != 0 {
                    0b11u8
                } else {
                    0b00u8
                };
                byte_val |= code << (j * 2);
            });
            out_bytes[k] = byte_val;
        });
        packed[pi..pi + 4].copy_from_slice(&out_bytes);
        wi += 16;
        pi += 4;
    }

    // Scalar tail.
    if wi < n {
        scalar_pack_ternary(&weights[wi..], &mut packed[pi..]);
    }
}

/// Pack ternary weights {-1, 0, +1} into 2-bit packed format.
///
/// Each output byte stores 4 values (2 bits each, LSB-first).
/// `packed` must have length ≥ `ternary_packed_len(weights.len())`.
pub fn pack_ternary_weights(weights: &[f32], packed: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe { neon_pack_ternary(weights, packed) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_pack_ternary(weights, packed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. unpack_ternary_weights
// ═══════════════════════════════════════════════════════════════════

/// Scalar ternary unpacking.
fn scalar_unpack_ternary(packed: &[u8], count: usize, output: &mut [f32]) {
    assert!(output.len() >= count, "output too small: need {count}, got {}", output.len());
    let required_packed = ternary_packed_len(count);
    assert!(packed.len() >= required_packed, "packed buffer too small for {count} values");

    let mut idx = 0;
    packed.iter().take(required_packed).for_each(|&byte| {
        (0..4).for_each(|j| {
            if idx < count {
                output[idx] = decode_ternary(byte >> (j * 2));
                idx += 1;
            }
        });
    });
}

/// NEON-accelerated ternary unpacking.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_unpack_ternary(packed: &[u8], count: usize, output: &mut [f32]) {
    assert!(output.len() >= count, "output too small: need {count}, got {}", output.len());

    let n = count;
    let mut wi = 0;
    let mut pi = 0;

    // Process 4 packed bytes → 16 floats per iteration.
    while wi + 16 <= n && pi + 4 <= packed.len() {
        let bytes = &packed[pi..pi + 4];
        bytes.iter().enumerate().for_each(|(k, &b)| {
            let base = wi + k * 4;
            let vals: [f32; 4] = [
                decode_ternary(b),
                decode_ternary(b >> 2),
                decode_ternary(b >> 4),
                decode_ternary(b >> 6),
            ];
            // SAFETY: vals is a local array; base+4 <= wi+16 <= n.
            unsafe {
                let v = vld1q_f32(vals.as_ptr());
                vst1q_f32(output.as_mut_ptr().add(base), v);
            }
        });
        wi += 16;
        pi += 4;
    }

    // Scalar tail.
    if wi < n {
        scalar_unpack_ternary(&packed[pi..], n - wi, &mut output[wi..]);
    }
}

/// Unpack 2-bit packed ternary weights to f32.
///
/// Unpacks `count` values from `packed` into `output`.
pub fn unpack_ternary_weights(packed: &[u8], count: usize, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_unpack_ternary(packed, count, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_unpack_ternary(packed, count, output);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. pack_binary_weights
// ═══════════════════════════════════════════════════════════════════

/// Scalar binary packing.
fn scalar_pack_binary(weights: &[f32], packed: &mut [u8]) {
    let required = binary_packed_len(weights.len());
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );
    weights.chunks(8).zip(packed.iter_mut()).for_each(|(chunk, byte)| {
        let mut val = 0u8;
        chunk.iter().enumerate().for_each(|(j, &w)| {
            if w > 0.0 {
                val |= 1 << j;
            }
        });
        *byte = val;
    });
}

/// NEON-accelerated binary packing.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_pack_binary(weights: &[f32], packed: &mut [u8]) {
    let required = binary_packed_len(weights.len());
    assert!(
        packed.len() >= required,
        "packed buffer too small: need {required}, got {}",
        packed.len()
    );

    let n = weights.len();
    let mut wi = 0;
    let mut pi = 0;
    let zero = vdupq_n_f32(0.0);

    // Process 8 weights → 1 packed byte at a time.
    while wi + 8 <= n {
        // SAFETY: wi+8 <= n, so pointer reads are in bounds.
        let (lo, hi, cmp_lo, cmp_hi) = unsafe {
            let lo = vld1q_f32(weights.as_ptr().add(wi));
            let hi = vld1q_f32(weights.as_ptr().add(wi + 4));
            (lo, hi, vcgtq_f32(lo, zero), vcgtq_f32(hi, zero))
        };
        let _ = (lo, hi); // suppress unused binding

        let mut lo_bits = [0u32; 4];
        let mut hi_bits = [0u32; 4];
        // SAFETY: writing to local arrays.
        unsafe {
            vst1q_u32(lo_bits.as_mut_ptr(), cmp_lo);
            vst1q_u32(hi_bits.as_mut_ptr(), cmp_hi);
        }

        let mut byte_val = 0u8;
        lo_bits.iter().enumerate().for_each(|(j, &b)| {
            if b != 0 {
                byte_val |= 1 << j;
            }
        });
        hi_bits.iter().enumerate().for_each(|(j, &b)| {
            if b != 0 {
                byte_val |= 1 << (j + 4);
            }
        });
        packed[pi] = byte_val;
        wi += 8;
        pi += 1;
    }

    // Scalar tail.
    if wi < n {
        scalar_pack_binary(&weights[wi..], &mut packed[pi..]);
    }
}

/// Pack binary weights {-1, +1} into 1-bit packed format.
///
/// Each output byte stores 8 values (1 bit each, LSB-first).
/// Positive → 1, non-positive → 0.
pub fn pack_binary_weights(weights: &[f32], packed: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_pack_binary(weights, packed) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_pack_binary(weights, packed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. unpack_binary_weights
// ═══════════════════════════════════════════════════════════════════

/// Scalar binary unpacking.
fn scalar_unpack_binary(packed: &[u8], count: usize, output: &mut [f32]) {
    assert!(output.len() >= count, "output too small: need {count}, got {}", output.len());
    let required_packed = binary_packed_len(count);
    assert!(packed.len() >= required_packed, "packed buffer too small for {count} values");

    let mut idx = 0;
    packed.iter().take(required_packed).for_each(|&byte| {
        (0..8).for_each(|j| {
            if idx < count {
                output[idx] = if (byte >> j) & 1 == 1 { 1.0 } else { -1.0 };
                idx += 1;
            }
        });
    });
}

/// NEON-accelerated binary unpacking.
///
/// # Safety
/// Requires `neon` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_unpack_binary(packed: &[u8], count: usize, output: &mut [f32]) {
    assert!(output.len() >= count, "output too small: need {count}, got {}", output.len());

    let n = count;
    let mut wi = 0;
    let mut pi = 0;
    let _pos_one = vdupq_n_f32(1.0);
    let _neg_one = vdupq_n_f32(-1.0);

    // Process 1 packed byte → 8 floats per iteration.
    while wi + 8 <= n && pi < packed.len() {
        let byte = packed[pi];

        // Low 4 bits → first 4 floats.
        let lo_vals: [f32; 4] =
            std::array::from_fn(|j| if (byte >> j) & 1 == 1 { 1.0 } else { -1.0 });
        // SAFETY: lo_vals is local; wi+4 <= wi+8 <= n.
        unsafe {
            let lo_v = vld1q_f32(lo_vals.as_ptr());
            vst1q_f32(output.as_mut_ptr().add(wi), lo_v);
        }

        // High 4 bits → next 4 floats.
        let hi_vals: [f32; 4] =
            std::array::from_fn(|j| if (byte >> (j + 4)) & 1 == 1 { 1.0 } else { -1.0 });
        // SAFETY: hi_vals is local; wi+8 <= n.
        unsafe {
            let hi_v = vld1q_f32(hi_vals.as_ptr());
            vst1q_f32(output.as_mut_ptr().add(wi + 4), hi_v);
        }

        wi += 8;
        pi += 1;
    }

    // Scalar tail.
    if wi < n {
        scalar_unpack_binary(&packed[pi..], n - wi, &mut output[wi..]);
    }
}

/// Unpack 1-bit packed binary weights to f32.
///
/// `0` → −1.0, `1` → +1.0. Unpacks `count` values.
pub fn unpack_binary_weights(packed: &[u8], count: usize, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_unpack_binary(packed, count, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_unpack_binary(packed, count, output);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. repack_for_simd
// ═══════════════════════════════════════════════════════════════════

/// Reorder packed ternary weight bytes for optimal SIMD access.
///
/// Rearranges a row-major packed buffer into tiles of `tile_size`
/// bytes so that consecutive SIMD loads access contiguous memory.
/// The total packed length must be a multiple of `tile_size`.
///
/// Falls back to a simple copy when the length is not tile-aligned.
pub fn repack_for_simd(
    packed: &[u8],
    rows: usize,
    cols_packed: usize,
    tile_size: usize,
    output: &mut [u8],
) {
    assert!(tile_size > 0, "tile_size must be > 0");
    let total = rows * cols_packed;
    assert!(packed.len() >= total, "packed buffer too small: need {total}, got {}", packed.len());
    assert!(output.len() >= total, "output buffer too small: need {total}, got {}", output.len());

    if !cols_packed.is_multiple_of(tile_size) {
        // Non-tile-aligned: plain copy preserving row order.
        output[..total].copy_from_slice(&packed[..total]);
        return;
    }

    let tiles_per_row = cols_packed / tile_size;

    // Emit tiles in column-major order within each tile column,
    // so sequential reads walk down rows before advancing columns.
    (0..tiles_per_row).for_each(|tc| {
        (0..rows).for_each(|r| {
            let src_off = r * cols_packed + tc * tile_size;
            let dst_off = (tc * rows + r) * tile_size;
            output[dst_off..dst_off + tile_size]
                .copy_from_slice(&packed[src_off..src_off + tile_size]);
        });
    });
}

// ═══════════════════════════════════════════════════════════════════
// 6. pack_with_scale
// ═══════════════════════════════════════════════════════════════════

/// Pack weights with per-block scale factors.
///
/// Divides `weights` into blocks of `block_size`, computes the
/// absolute-max scale for each block, quantises to ternary
/// {-1,0,+1} using `threshold` (fraction of scale), packs into
/// 2-bit format, and writes scales into `scales`.
///
/// Returns the number of blocks written.
///
/// # Panics
///
/// Panics if `block_size` is 0 or output buffers are too small.
pub fn pack_with_scale(
    weights: &[f32],
    block_size: usize,
    threshold: f32,
    packed: &mut [u8],
    scales: &mut [f32],
) -> usize {
    assert!(block_size > 0, "block_size must be > 0");
    let n = weights.len();
    let num_blocks = n.div_ceil(block_size);
    let packed_per_block = ternary_packed_len(block_size);
    let required_packed = num_blocks * packed_per_block;
    assert!(
        packed.len() >= required_packed,
        "packed buffer too small: need {required_packed}, got {}",
        packed.len()
    );
    assert!(
        scales.len() >= num_blocks,
        "scales buffer too small: need {num_blocks}, got {}",
        scales.len()
    );

    weights.chunks(block_size).enumerate().for_each(|(bi, block)| {
        // Compute absolute-max scale.
        let scale = block.iter().fold(0.0f32, |acc, &w| acc.max(w.abs()));
        scales[bi] = scale;

        let cutoff = scale * threshold;

        // Quantise this block to ternary, then pack.
        let packed_start = bi * packed_per_block;
        let packed_end = packed_start + ternary_packed_len(block.len());
        let dst = &mut packed[packed_start..packed_end];
        dst.iter_mut().for_each(|b| *b = 0);

        block.chunks(4).enumerate().for_each(|(ci, chunk)| {
            let mut byte_val = 0u8;
            chunk.iter().enumerate().for_each(|(j, &w)| {
                let code = if w > cutoff {
                    0b01
                } else if w < -cutoff {
                    0b11
                } else {
                    0b00
                };
                byte_val |= code << (j * 2);
            });
            dst[ci] = byte_val;
        });
    });

    num_blocks
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ternary packing helpers ────────────────────────────────────

    #[test]
    fn test_ternary_packed_len_zero() {
        assert_eq!(ternary_packed_len(0), 0);
    }

    #[test]
    fn test_ternary_packed_len_one() {
        assert_eq!(ternary_packed_len(1), 1);
    }

    #[test]
    fn test_ternary_packed_len_four() {
        assert_eq!(ternary_packed_len(4), 1);
    }

    #[test]
    fn test_ternary_packed_len_five() {
        assert_eq!(ternary_packed_len(5), 2);
    }

    #[test]
    fn test_ternary_packed_len_large() {
        assert_eq!(ternary_packed_len(256), 64);
    }

    #[test]
    fn test_binary_packed_len_zero() {
        assert_eq!(binary_packed_len(0), 0);
    }

    #[test]
    fn test_binary_packed_len_one() {
        assert_eq!(binary_packed_len(1), 1);
    }

    #[test]
    fn test_binary_packed_len_eight() {
        assert_eq!(binary_packed_len(8), 1);
    }

    #[test]
    fn test_binary_packed_len_nine() {
        assert_eq!(binary_packed_len(9), 2);
    }

    #[test]
    fn test_binary_packed_len_large() {
        assert_eq!(binary_packed_len(256), 32);
    }

    // ── encode / decode helpers ────────────────────────────────────

    #[test]
    fn test_encode_ternary_positive() {
        assert_eq!(encode_ternary(1.0), 0b01);
    }

    #[test]
    fn test_encode_ternary_negative() {
        assert_eq!(encode_ternary(-1.0), 0b11);
    }

    #[test]
    fn test_encode_ternary_zero() {
        assert_eq!(encode_ternary(0.0), 0b00);
    }

    #[test]
    fn test_decode_ternary_positive() {
        assert_eq!(decode_ternary(0b01), 1.0);
    }

    #[test]
    fn test_decode_ternary_negative() {
        assert_eq!(decode_ternary(0b11), -1.0);
    }

    #[test]
    fn test_decode_ternary_zero() {
        assert_eq!(decode_ternary(0b00), 0.0);
    }

    #[test]
    fn test_decode_ternary_also_zero() {
        assert_eq!(decode_ternary(0b10), 0.0);
    }

    // ── pack_ternary_weights ───────────────────────────────────────

    #[test]
    fn test_pack_ternary_empty() {
        let w: [f32; 0] = [];
        let mut p = [0u8; 0];
        pack_ternary_weights(&w, &mut p);
    }

    #[test]
    fn test_pack_ternary_single_positive() {
        let w = [1.0f32];
        let mut p = [0u8; 1];
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0] & 0x03, 0b01);
    }

    #[test]
    fn test_pack_ternary_single_negative() {
        let w = [-1.0f32];
        let mut p = [0u8; 1];
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0] & 0x03, 0b11);
    }

    #[test]
    fn test_pack_ternary_single_zero() {
        let w = [0.0f32];
        let mut p = [0u8; 1];
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0] & 0x03, 0b00);
    }

    #[test]
    fn test_pack_ternary_four_values() {
        // [+1, -1, 0, +1] → bits 01_11_00_01 = 0b01_00_11_01
        let w = [1.0, -1.0, 0.0, 1.0f32];
        let mut p = [0u8; 1];
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0], 0b01_00_11_01);
    }

    #[test]
    fn test_pack_ternary_five_values() {
        let w = [1.0, -1.0, 0.0, 1.0, -1.0f32];
        let mut p = [0u8; 2];
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0], 0b01_00_11_01);
        assert_eq!(p[1] & 0x03, 0b11);
    }

    #[test]
    fn test_pack_ternary_all_zeros() {
        let w = [0.0f32; 16];
        let mut p = [0u8; 4];
        pack_ternary_weights(&w, &mut p);
        assert!(p.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pack_ternary_all_positive() {
        let w = [1.0f32; 16];
        let mut p = [0u8; 4];
        pack_ternary_weights(&w, &mut p);
        // Each byte: 01_01_01_01 = 0x55
        assert!(p.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn test_pack_ternary_all_negative() {
        let w = [-1.0f32; 16];
        let mut p = [0u8; 4];
        pack_ternary_weights(&w, &mut p);
        // Each byte: 11_11_11_11 = 0xFF
        assert!(p.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_pack_ternary_17_values() {
        let mut w: Vec<f32> = vec![1.0f32; 16];
        w.push(-1.0);
        let mut p = [0u8; 5];
        pack_ternary_weights(&w, &mut p);
        assert!(p[..4].iter().all(|&b| b == 0x55));
        assert_eq!(p[4] & 0x03, 0b11);
    }

    // ── unpack_ternary_weights ─────────────────────────────────────

    #[test]
    fn test_unpack_ternary_empty() {
        let p: [u8; 0] = [];
        let mut o = [0.0f32; 0];
        unpack_ternary_weights(&p, 0, &mut o);
    }

    #[test]
    fn test_unpack_ternary_single_pos() {
        let p = [0b01u8];
        let mut o = [0.0f32; 1];
        unpack_ternary_weights(&p, 1, &mut o);
        assert_eq!(o[0], 1.0);
    }

    #[test]
    fn test_unpack_ternary_single_neg() {
        let p = [0b11u8];
        let mut o = [0.0f32; 1];
        unpack_ternary_weights(&p, 1, &mut o);
        assert_eq!(o[0], -1.0);
    }

    #[test]
    fn test_unpack_ternary_four_values() {
        let p = [0b01_00_11_01u8]; // +1, -1, 0, +1
        let mut o = [0.0f32; 4];
        unpack_ternary_weights(&p, 4, &mut o);
        assert_eq!(o, [1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_unpack_ternary_16_zeros() {
        let p = [0u8; 4];
        let mut o = [0.0f32; 16];
        unpack_ternary_weights(&p, 16, &mut o);
        assert!(o.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_unpack_ternary_16_positive() {
        let p = [0x55u8; 4]; // all +1
        let mut o = [0.0f32; 16];
        unpack_ternary_weights(&p, 16, &mut o);
        assert!(o.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_unpack_ternary_16_negative() {
        let p = [0xFFu8; 4]; // all -1
        let mut o = [0.0f32; 16];
        unpack_ternary_weights(&p, 16, &mut o);
        assert!(o.iter().all(|&v| v == -1.0));
    }

    // ── ternary round-trip ─────────────────────────────────────────

    #[test]
    fn test_ternary_roundtrip_4() {
        let w = [1.0f32, -1.0, 0.0, 1.0];
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o.as_slice(), &w);
    }

    #[test]
    fn test_ternary_roundtrip_16() {
        let w: Vec<f32> = (0..16)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_17() {
        let w: Vec<f32> = (0..17)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_32() {
        let w: Vec<f32> = (0..32).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_33() {
        let w: Vec<f32> = (0..33).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_64() {
        let w: Vec<f32> = (0..64).map(|i| [1.0, -1.0, 0.0, 1.0][i % 4]).collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_100() {
        let w: Vec<f32> = (0..100)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    // ── pack_binary_weights ────────────────────────────────────────

    #[test]
    fn test_pack_binary_empty() {
        let w: [f32; 0] = [];
        let mut p = [0u8; 0];
        pack_binary_weights(&w, &mut p);
    }

    #[test]
    fn test_pack_binary_single_positive() {
        let w = [1.0f32];
        let mut p = [0u8; 1];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0] & 1, 1);
    }

    #[test]
    fn test_pack_binary_single_negative() {
        let w = [-1.0f32];
        let mut p = [0u8; 1];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0] & 1, 0);
    }

    #[test]
    fn test_pack_binary_eight_positive() {
        let w = [1.0f32; 8];
        let mut p = [0u8; 1];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0], 0xFF);
    }

    #[test]
    fn test_pack_binary_eight_negative() {
        let w = [-1.0f32; 8];
        let mut p = [0u8; 1];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0], 0x00);
    }

    #[test]
    fn test_pack_binary_alternating() {
        // [+1, -1, +1, -1, +1, -1, +1, -1] → bits 10101010
        // LSB-first: bit0=1,bit1=0,... → 0b01010101 = 0x55
        let w: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = [0u8; 1];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0], 0x55);
    }

    #[test]
    fn test_pack_binary_nine_values() {
        let mut w: Vec<f32> = vec![1.0f32; 8];
        w.push(-1.0);
        let mut p = [0u8; 2];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0], 0xFF);
        assert_eq!(p[1] & 1, 0);
    }

    #[test]
    fn test_pack_binary_16_values() {
        let w = [1.0f32; 16];
        let mut p = [0u8; 2];
        pack_binary_weights(&w, &mut p);
        assert!(p.iter().all(|&b| b == 0xFF));
    }

    // ── unpack_binary_weights ──────────────────────────────────────

    #[test]
    fn test_unpack_binary_empty() {
        let p: [u8; 0] = [];
        let mut o = [0.0f32; 0];
        unpack_binary_weights(&p, 0, &mut o);
    }

    #[test]
    fn test_unpack_binary_single_positive() {
        let p = [0b1u8];
        let mut o = [0.0f32; 1];
        unpack_binary_weights(&p, 1, &mut o);
        assert_eq!(o[0], 1.0);
    }

    #[test]
    fn test_unpack_binary_single_negative() {
        let p = [0b0u8];
        let mut o = [0.0f32; 1];
        unpack_binary_weights(&p, 1, &mut o);
        assert_eq!(o[0], -1.0);
    }

    #[test]
    fn test_unpack_binary_eight_positive() {
        let p = [0xFFu8];
        let mut o = [0.0f32; 8];
        unpack_binary_weights(&p, 8, &mut o);
        assert!(o.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_unpack_binary_eight_negative() {
        let p = [0x00u8];
        let mut o = [0.0f32; 8];
        unpack_binary_weights(&p, 8, &mut o);
        assert!(o.iter().all(|&v| v == -1.0));
    }

    // ── binary round-trip ──────────────────────────────────────────

    #[test]
    fn test_binary_roundtrip_8() {
        let w: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_16() {
        let w: Vec<f32> = (0..16).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_9() {
        let w: Vec<f32> = (0..9).map(|i| if i < 5 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_32() {
        let w: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_33() {
        let w: Vec<f32> = (0..33).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_64() {
        let w: Vec<f32> = (0..64).map(|i| if i % 4 == 0 { -1.0 } else { 1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_100() {
        let w: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    // ── repack_for_simd ────────────────────────────────────────────

    #[test]
    fn test_repack_identity_when_non_aligned() {
        let packed = vec![1, 2, 3, 4, 5, 6];
        let mut out = [0u8; 6];
        // 2 rows × 3 cols, tile_size=4 → not aligned → copy
        repack_for_simd(&packed, 2, 3, 4, &mut out);
        assert_eq!(out.to_vec(), packed);
    }

    #[test]
    fn test_repack_single_row_aligned() {
        let packed = vec![10, 20, 30, 40];
        let mut out = [0u8; 4];
        repack_for_simd(&packed, 1, 4, 4, &mut out);
        assert_eq!(out.to_vec(), packed);
    }

    #[test]
    fn test_repack_2x4_tile4() {
        // 2 rows × 4 cols, tile=4 → 1 tile per row
        // Row 0: [A,B,C,D], Row 1: [E,F,G,H]
        // Column-major tile order: tile(0,0), tile(1,0)
        let packed = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = [0u8; 8];
        repack_for_simd(&packed, 2, 4, 4, &mut out);
        assert_eq!(out.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_repack_2x8_tile4() {
        // 2 rows × 8 cols, tile=4 → 2 tiles per row
        // Row0: [1..4][5..8], Row1: [9..12][13..16]
        // Output: tile(col0,row0), tile(col0,row1),
        //         tile(col1,row0), tile(col1,row1)
        let packed: Vec<u8> = (1..=16).collect();
        let mut out = [0u8; 16];
        repack_for_simd(&packed, 2, 8, 4, &mut out);
        assert_eq!(out.to_vec(), vec![1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16]);
    }

    #[test]
    fn test_repack_4x4_tile2() {
        // 4 rows × 4 cols, tile=2 → 2 tiles per row
        let packed: Vec<u8> = (0..16).collect();
        let mut out = [0u8; 16];
        repack_for_simd(&packed, 4, 4, 2, &mut out);
        // tile_col 0: rows 0..3 → [0,1], [4,5], [8,9], [12,13]
        // tile_col 1: rows 0..3 → [2,3], [6,7], [10,11], [14,15]
        assert_eq!(out.to_vec(), vec![0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]);
    }

    #[test]
    fn test_repack_tile_size_one() {
        let packed = vec![10, 20, 30, 40, 50, 60];
        let mut out = [0u8; 6];
        // 2 rows × 3 cols, tile=1 → 3 tiles per row
        repack_for_simd(&packed, 2, 3, 1, &mut out);
        // tile_col 0: [10, 40], tile_col 1: [20, 50],
        // tile_col 2: [30, 60]
        assert_eq!(out.to_vec(), vec![10, 40, 20, 50, 30, 60]);
    }

    #[test]
    #[should_panic(expected = "tile_size must be > 0")]
    fn test_repack_zero_tile_panics() {
        let packed = [0u8; 4];
        let mut out = [0u8; 4];
        repack_for_simd(&packed, 1, 4, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_repack_packed_too_small() {
        let packed = [0u8; 2];
        let mut out = [0u8; 8];
        repack_for_simd(&packed, 2, 4, 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_repack_output_too_small() {
        let packed = [0u8; 8];
        let mut out = [0u8; 4];
        repack_for_simd(&packed, 2, 4, 4, &mut out);
    }

    // ── pack_with_scale ────────────────────────────────────────────

    #[test]
    fn test_pack_with_scale_single_block() {
        let w = [0.5, -0.5, 0.1, 0.9f32];
        let block_size = 4;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; packed_per];
        let mut scales = [0.0f32; 1];
        let nb = pack_with_scale(&w, block_size, 0.3, &mut packed, &mut scales);
        assert_eq!(nb, 1);
        assert!((scales[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_pack_with_scale_two_blocks() {
        let w = vec![1.0, -1.0, 0.0, 0.5, 2.0, -2.0, 0.0, 0.0];
        let block_size = 4;
        let num_blocks = 2;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; num_blocks * packed_per];
        let mut scales = vec![0.0f32; num_blocks];
        let nb = pack_with_scale(&w, block_size, 0.3, &mut packed, &mut scales);
        assert_eq!(nb, 2);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pack_with_scale_partial_last_block() {
        let w = [1.0f32; 5]; // 5 values, block_size=4 → 2 blocks
        let block_size = 4;
        let num_blocks = 2;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; num_blocks * packed_per];
        let mut scales = vec![0.0f32; num_blocks];
        let nb = pack_with_scale(&w, block_size, 0.3, &mut packed, &mut scales);
        assert_eq!(nb, 2);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pack_with_scale_all_zero() {
        let w = [0.0f32; 8];
        let block_size = 4;
        let num_blocks = 2;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; num_blocks * packed_per];
        let mut scales = vec![0.0f32; num_blocks];
        let nb = pack_with_scale(&w, block_size, 0.5, &mut packed, &mut scales);
        assert_eq!(nb, 2);
        assert!(scales.iter().all(|&s| s == 0.0));
        assert!(packed.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pack_with_scale_threshold_effect() {
        // With scale=2.0 and threshold=0.6, cutoff=1.2
        // Values ≤1.2 abs become zero.
        let w = [2.0, -2.0, 1.0, -1.0f32];
        let block_size = 4;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; packed_per];
        let mut scales = [0.0f32; 1];
        let nb = pack_with_scale(&w, block_size, 0.6, &mut packed, &mut scales);
        assert_eq!(nb, 1);
        assert!((scales[0] - 2.0).abs() < 1e-6);
        // 2.0 → +1, -2.0 → -1, 1.0 → 0, -1.0 → 0
        let mut out = [0.0f32; 4];
        unpack_ternary_weights(&packed, 4, &mut out);
        assert_eq!(out, [1.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_pack_with_scale_zero_block_panics() {
        let w = [1.0f32];
        let mut packed = [0u8; 1];
        let mut scales = [0.0f32; 1];
        pack_with_scale(&w, 0, 0.5, &mut packed, &mut scales);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_pack_with_scale_packed_too_small() {
        let w = [1.0f32; 8];
        let mut packed = [0u8; 1]; // need 2
        let mut scales = [0.0f32; 2];
        pack_with_scale(&w, 4, 0.5, &mut packed, &mut scales);
    }

    #[test]
    #[should_panic(expected = "scales buffer too small")]
    fn test_pack_with_scale_scales_too_small() {
        let w = [1.0f32; 8];
        let mut packed = [0u8; 2];
        let mut scales = [0.0f32; 1]; // need 2
        pack_with_scale(&w, 4, 0.5, &mut packed, &mut scales);
    }

    #[test]
    fn test_pack_with_scale_block_size_one() {
        let w = [1.0, -1.0, 0.0f32];
        let mut packed = [0u8; 3]; // 3 blocks × 1 byte
        let mut scales = [0.0f32; 3];
        let nb = pack_with_scale(&w, 1, 0.5, &mut packed, &mut scales);
        assert_eq!(nb, 3);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 1.0).abs() < 1e-6);
        assert_eq!(scales[2], 0.0);
    }

    #[test]
    fn test_pack_with_scale_large_block() {
        let w: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let block_size = 64;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; packed_per];
        let mut scales = [0.0f32; 1];
        let nb = pack_with_scale(&w, block_size, 0.3, &mut packed, &mut scales);
        assert_eq!(nb, 1);
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    // ── Edge cases & panics ────────────────────────────────────────

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_pack_ternary_buffer_too_small() {
        let w = [1.0f32; 8];
        let mut p = [0u8; 1]; // need 2
        pack_ternary_weights(&w, &mut p);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn test_unpack_ternary_output_too_small() {
        let p = [0u8; 1];
        let mut o = [0.0f32; 2]; // need 4 for count=4
        unpack_ternary_weights(&p, 4, &mut o);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_pack_binary_buffer_too_small() {
        let w = [1.0f32; 16];
        let mut p = [0u8; 1]; // need 2
        pack_binary_weights(&w, &mut p);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn test_unpack_binary_output_too_small() {
        let p = [0xFFu8];
        let mut o = [0.0f32; 4]; // need 8 for count=8
        unpack_binary_weights(&p, 8, &mut o);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_unpack_ternary_packed_too_small() {
        let p = [0u8; 1]; // 1 byte = 4 vals, asking for 8
        let mut o = [0.0f32; 8];
        unpack_ternary_weights(&p, 8, &mut o);
    }

    #[test]
    #[should_panic(expected = "packed buffer too small")]
    fn test_unpack_binary_packed_too_small() {
        let p = [0u8; 1]; // 1 byte = 8 vals, asking for 16
        let mut o = [0.0f32; 16];
        unpack_binary_weights(&p, 16, &mut o);
    }

    // ── Large round-trip stress ────────────────────────────────────

    #[test]
    fn test_ternary_roundtrip_256() {
        let w: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_256() {
        let w: Vec<f32> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_ternary_roundtrip_1024() {
        let w: Vec<f32> = (0..1024).map(|i| [-1.0, 0.0, 1.0][i % 3]).collect();
        let mut p = vec![0u8; ternary_packed_len(w.len())];
        pack_ternary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_ternary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    #[test]
    fn test_binary_roundtrip_1024() {
        let w: Vec<f32> = (0..1024).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut p = vec![0u8; binary_packed_len(w.len())];
        pack_binary_weights(&w, &mut p);
        let mut o = vec![0.0f32; w.len()];
        unpack_binary_weights(&p, w.len(), &mut o);
        assert_eq!(o, w);
    }

    // ── Oversized buffer tests ─────────────────────────────────────

    #[test]
    fn test_pack_ternary_oversized_buffer() {
        let w = [1.0, -1.0, 0.0, 1.0f32];
        let mut p = [0u8; 16]; // much larger than needed
        pack_ternary_weights(&w, &mut p);
        assert_eq!(p[0], 0b01_00_11_01);
    }

    #[test]
    fn test_pack_binary_oversized_buffer() {
        let w = [1.0f32; 8];
        let mut p = [0u8; 16];
        pack_binary_weights(&w, &mut p);
        assert_eq!(p[0], 0xFF);
    }

    #[test]
    fn test_unpack_ternary_oversized_output() {
        let p = [0x55u8]; // all +1
        let mut o = [0.0f32; 16];
        unpack_ternary_weights(&p, 4, &mut o);
        assert!(o[..4].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_unpack_binary_oversized_output() {
        let p = [0xFFu8];
        let mut o = [0.0f32; 16];
        unpack_binary_weights(&p, 8, &mut o);
        assert!(o[..8].iter().all(|&v| v == 1.0));
    }

    // ── repack round-trip ──────────────────────────────────────────

    #[test]
    fn test_repack_preserves_data_4x8() {
        let packed: Vec<u8> = (0..32).collect();
        let mut repacked = [0u8; 32];
        repack_for_simd(&packed, 4, 8, 4, &mut repacked);
        // Verify all bytes are present (permutation).
        let mut sorted_in = packed.clone();
        sorted_in.sort();
        let mut sorted_out = repacked.clone();
        sorted_out.sort();
        assert_eq!(sorted_in, sorted_out);
    }

    // ── Mixed-precision scale accuracy ─────────────────────────────

    #[test]
    fn test_pack_with_scale_reconstructs_approximate() {
        let w = [0.8, -0.8, 0.2, 0.0f32];
        let block_size = 4;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; packed_per];
        let mut scales = [0.0f32; 1];
        pack_with_scale(&w, block_size, 0.3, &mut packed, &mut scales);
        // Reconstruct: unpack ternary, multiply by scale.
        let mut quant = [0.0f32; 4];
        unpack_ternary_weights(&packed, 4, &mut quant);
        let reconstructed: Vec<f32> = quant.iter().map(|&q| q * scales[0]).collect();
        // 0.8→+1*0.8=0.8, -0.8→-1*0.8=-0.8, 0.2→0*0.8=0.0
        assert!((reconstructed[0] - 0.8).abs() < 1e-6);
        assert!((reconstructed[1] + 0.8).abs() < 1e-6);
        assert_eq!(reconstructed[2], 0.0);
        assert_eq!(reconstructed[3], 0.0);
    }

    #[test]
    fn test_pack_with_scale_negative_weights() {
        let w = [-3.0, -2.0, -1.0, -0.1f32];
        let block_size = 4;
        let packed_per = ternary_packed_len(block_size);
        let mut packed = vec![0u8; packed_per];
        let mut scales = [0.0f32; 1];
        pack_with_scale(&w, block_size, 0.5, &mut packed, &mut scales);
        assert!((scales[0] - 3.0).abs() < 1e-6);
        let mut quant = [0.0f32; 4];
        unpack_ternary_weights(&packed, 4, &mut quant);
        // -3.0 > cutoff(1.5) in abs → -1
        // -2.0 > cutoff(1.5) in abs → -1
        // -1.0 < cutoff(1.5) in abs → 0
        // -0.1 < cutoff(1.5) in abs → 0
        assert_eq!(quant[0], -1.0);
        assert_eq!(quant[1], -1.0);
        assert_eq!(quant[2], 0.0);
        assert_eq!(quant[3], 0.0);
    }
}
