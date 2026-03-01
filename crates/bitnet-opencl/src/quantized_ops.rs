//! CPU reference implementations for 1.58-bit (ternary) quantized inference.
//!
//! Provides pack/unpack utilities for `I2_S` format (2 bits per weight) and
//! QK256 block dequantization, plus ternary matmul/matvec operations.

// ── I2_S format ──────────────────────────────────────────────────────────────
//
// Each weight is encoded as 2 bits:
//   0b00 → 0, 0b01 → +1, 0b10 → -1, 0b11 → reserved (treated as 0)
//
// Weights are packed LSB-first: 4 weights per byte.

/// Map a 2-bit code to its ternary value.
#[inline]
const fn i2s_decode(code: u8) -> f32 {
    match code & 0b11 {
        0b01 => 1.0,
        0b10 => -1.0,
        _ => 0.0, // 0b00 = zero, 0b11 = reserved → zero
    }
}

/// Unpack `I2_S` packed bytes to `f32` values and apply `scale`.
///
/// Each byte holds 4 weights (2 bits each, LSB-first).
/// Returns `packed.len() * 4` elements.
pub fn dequantize_i2s(packed: &[u8], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        for shift in (0..8).step_by(2) {
            out.push(i2s_decode(byte >> shift) * scale);
        }
    }
    out
}

// ── QK256 format ─────────────────────────────────────────────────────────────
//
// A QK256 block encodes 256 ternary weights:
//   - First 2 bytes: f16 scale factor (little-endian IEEE 754 half)
//   - Next 64 bytes: 256 weights × 2 bits = 512 bits = 64 bytes
//
// Total block size: 66 bytes.

/// Expected byte size of a single QK256 block.
pub const QK256_BLOCK_SIZE: usize = 66;

/// Number of weights per QK256 block.
pub const QK256_WEIGHTS_PER_BLOCK: usize = 256;

/// Decode an f16 value from two little-endian bytes.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u16::from_le_bytes([lo, hi]);
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);

    if exp == 0 {
        // Subnormal or zero
        let val = (frac as f32) * (1.0 / 16_777_216.0); // 2^-24
        if sign == 1 { -val } else { val }
    } else if exp == 0x1F {
        0.0 // Inf / NaN → treat as 0 for safety
    } else {
        let f32_exp = exp + 112; // rebias: -15 + 127
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

/// Dequantize a single QK256 block (66 bytes) into 256 `f32` values.
///
/// Returns an empty vec if the block is too short.
pub fn dequantize_qk256(block: &[u8]) -> Vec<f32> {
    if block.len() < QK256_BLOCK_SIZE {
        return Vec::new();
    }

    let scale = f16_to_f32(block[0], block[1]);
    let weight_bytes = &block[2..QK256_BLOCK_SIZE];

    let mut out = Vec::with_capacity(QK256_WEIGHTS_PER_BLOCK);
    for &byte in weight_bytes {
        for shift in (0..8).step_by(2) {
            out.push(i2s_decode(byte >> shift) * scale);
        }
    }
    out
}

// ── Pack utilities ───────────────────────────────────────────────────────────

/// Map a ternary value to its 2-bit `I2_S` code.
///
/// Rounds: negative → `0b10`, positive → `0b01`, otherwise → `0b00`.
#[inline]
fn i2s_encode(val: f32) -> u8 {
    if val > 0.5 {
        0b01
    } else if val < -0.5 {
        0b10
    } else {
        0b00
    }
}

/// Pack ternary values (-1, 0, +1) into `I2_S` bytes, 4 weights per byte.
///
/// Input length is rounded up to a multiple of 4 (padding with zeros).
pub fn pack_i2s(values: &[f32]) -> Vec<u8> {
    let n_bytes = values.len().div_ceil(4);
    let mut packed = vec![0u8; n_bytes];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= i2s_encode(v) << bit_offset;
    }
    packed
}

// ── Ternary MatMul ───────────────────────────────────────────────────────────

/// Matrix-vector multiply with packed ternary weights.
///
/// `weights_packed` contains `rows * cols_packed` bytes, where
/// `cols_packed = cols.div_ceil(4)`. Each byte holds 4 ternary weights.
/// `input` has length `cols`.
///
/// Returns a vector of length `rows`, each element scaled by `scale`.
pub fn ternary_matvec(
    weights_packed: &[u8],
    input: &[f32],
    scale: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let cols_packed = cols.div_ceil(4);
    assert!(
        weights_packed.len() >= rows * cols_packed,
        "weights buffer too small: need {} bytes, got {}",
        rows * cols_packed,
        weights_packed.len(),
    );

    let mut output = vec![0.0f32; rows];

    for (row, out_val) in output.iter_mut().enumerate() {
        let row_start = row * cols_packed;
        let mut acc = 0.0f32;

        for (byte_idx, &byte) in
            weights_packed[row_start..row_start + cols_packed].iter().enumerate()
        {
            let base_col = byte_idx * 4;
            for sub in 0..4 {
                let col = base_col + sub;
                if col >= cols {
                    break;
                }
                let w = i2s_decode(byte >> (sub * 2));
                acc += w * input[col];
            }
        }

        *out_val = acc * scale;
    }

    output
}

/// Matrix-matrix multiply with packed ternary weights (batch of vectors).
///
/// `input` is a row-major `[batch_size * cols]` matrix.
/// Returns a row-major `[batch_size * rows]` matrix.
pub fn ternary_matmul(
    weights_packed: &[u8],
    input: &[f32],
    scale: f32,
    rows: usize,
    cols: usize,
    batch_size: usize,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        batch_size * cols,
        "input length mismatch: expected {}, got {}",
        batch_size * cols,
        input.len(),
    );

    let mut output = Vec::with_capacity(batch_size * rows);

    for b in 0..batch_size {
        let batch_input = &input[b * cols..(b + 1) * cols];
        let row_out = ternary_matvec(weights_packed, batch_input, scale, rows, cols);
        output.extend_from_slice(&row_out);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i2s_decode_values() {
        assert_eq!(i2s_decode(0b00), 0.0);
        assert_eq!(i2s_decode(0b01), 1.0);
        assert_eq!(i2s_decode(0b10), -1.0);
        assert_eq!(i2s_decode(0b11), 0.0); // reserved
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let values = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let packed = pack_i2s(&values);
        let unpacked = dequantize_i2s(&packed, 1.0);
        assert_eq!(&unpacked[..values.len()], &values[..]);
    }

    #[test]
    fn dequantize_i2s_with_scale() {
        // byte: 0b10_00_01_01 = weights [+1, +1, 0, -1]
        let packed = vec![0b10_00_01_01u8];
        let result = dequantize_i2s(&packed, 2.5);
        assert_eq!(result, vec![2.5, 2.5, 0.0, -2.5]);
    }

    #[test]
    fn f16_to_f32_one() {
        // f16 for 1.0: sign=0, exp=15, frac=0 → bits = 0x3C00
        let val = f16_to_f32(0x00, 0x3C);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_to_f32_negative_two() {
        // f16 for -2.0: sign=1, exp=16, frac=0 → bits = 0xC000
        let val = f16_to_f32(0x00, 0xC0);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn dequantize_qk256_too_short() {
        let result = dequantize_qk256(&[0u8; 10]);
        assert!(result.is_empty());
    }
}
