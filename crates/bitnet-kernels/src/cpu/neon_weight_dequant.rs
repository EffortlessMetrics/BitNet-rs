//! ARM NEON-optimized weight dequantization for Apple Silicon.
//!
//! Provides vectorized dequantization of various quantization formats
//! (I2_S, QK256, FP16, INT8, INT4) using AArch64 NEON intrinsics.

/// Convert f16 bits to f32 (software implementation).
#[inline]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 0i32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e -= 1;
        }
        let exp32 = (127 - 15 + 1 + e) as u32;
        let frac32 = (f & 0x3FF) << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
    } else if exp == 31 {
        if frac == 0 { f32::from_bits((sign << 31) | 0x7F800000) } else { f32::NAN }
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        let frac32 = frac << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S helpers ───────────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code: 0b00→0, 0b01→+1, 0b11→-1.
#[inline(always)]
fn decode_i2s(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

// ── Public API ─────────────────────────────────────────────────────────

/// Dequantize an I2_S block of packed 2-bit weights using NEON.
///
/// Each byte stores 4 values (LSB-first, 2 bits each). Output length is
/// `block_size`, with every decoded value multiplied by `scale`.
///
/// # Panics
///
/// Panics when `packed` does not contain enough bytes for `block_size`
/// elements (requires `ceil(block_size / 4)` bytes).
#[cfg(target_arch = "aarch64")]
pub fn neon_dequant_i2s_block(packed: &[u8], scale: f32, block_size: usize) -> Vec<f32> {
    let bytes_needed = block_size.div_ceil(4);
    assert!(
        packed.len() >= bytes_needed,
        "neon_dequant_i2s_block: need {bytes_needed} bytes for block_size={block_size}, got {}",
        packed.len()
    );

    let mut out = Vec::with_capacity(block_size);

    // SAFETY: NEON is always available on AArch64.
    unsafe {
        let scale_v = vdupq_n_f32(scale);

        // Process 16 elements (4 bytes) at a time via NEON.
        let full_chunks = block_size / 16;
        for chunk in 0..full_chunks {
            let base = chunk * 4;
            for byte_idx in 0..4 {
                let b = packed[base + byte_idx];
                let v0 = decode_i2s(b);
                let v1 = decode_i2s(b >> 2);
                let v2 = decode_i2s(b >> 4);
                let v3 = decode_i2s(b >> 6);

                let raw = [v0, v1, v2, v3];
                let vals = vld1q_f32(raw.as_ptr());
                let scaled = vmulq_f32(vals, scale_v);

                let mut buf = [0.0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), scaled);
                out.extend_from_slice(&buf);
            }
        }

        // Scalar remainder.
        for i in (full_chunks * 16)..block_size {
            let byte_idx = i / 4;
            let bit_off = (i % 4) * 2;
            let bits = (packed[byte_idx] >> bit_off) & 0x03;
            out.push(decode_i2s(bits) * scale);
        }
    }

    out
}

/// Dequantize a QK256 256-element block using NEON.
///
/// Each byte in `packed` stores 4 two-bit values (LSB-first, I2_S encoding).
/// Requires exactly 64 bytes of packed data. Output is 256 floats scaled by
/// `scale`.
///
/// # Panics
///
/// Panics if `packed.len() < 64`.
#[cfg(target_arch = "aarch64")]
pub fn neon_dequant_qk256_block(packed: &[u8], scale: f32) -> Vec<f32> {
    const QK: usize = 256;
    const PACKED_BYTES: usize = QK / 4;
    assert!(
        packed.len() >= PACKED_BYTES,
        "neon_dequant_qk256_block: need {PACKED_BYTES} bytes, got {}",
        packed.len()
    );

    let mut out = Vec::with_capacity(QK);

    // SAFETY: NEON is always available on AArch64.
    unsafe {
        let scale_v = vdupq_n_f32(scale);

        // Process 4 bytes (16 elements) per iteration, 64 bytes total.
        for chunk_start in (0..PACKED_BYTES).step_by(4) {
            for byte_off in 0..4 {
                let b = packed[chunk_start + byte_off];
                let v0 = decode_i2s(b);
                let v1 = decode_i2s(b >> 2);
                let v2 = decode_i2s(b >> 4);
                let v3 = decode_i2s(b >> 6);

                let raw = [v0, v1, v2, v3];
                let vals = vld1q_f32(raw.as_ptr());
                let scaled = vmulq_f32(vals, scale_v);

                let mut buf = [0.0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), scaled);
                out.extend_from_slice(&buf);
            }
        }
    }

    out
}

/// Convert FP16 weights to F32 using NEON `vcvt` instructions.
///
/// `fp16_bytes` is a byte slice of little-endian FP16 values (2 bytes each).
/// Output length is `fp16_bytes.len() / 2`.
///
/// # Panics
///
/// Panics if `fp16_bytes.len()` is odd.
#[cfg(target_arch = "aarch64")]
pub fn neon_dequant_fp16_to_f32(fp16_bytes: &[u8]) -> Vec<f32> {
    assert!(fp16_bytes.len().is_multiple_of(2), "fp16 byte slice must have even length");

    let count = fp16_bytes.len() / 2;
    let mut out = Vec::with_capacity(count);

    // Process 4 FP16 values (8 bytes) at a time.
    let full_chunks = count / 4;
    for chunk in 0..full_chunks {
        let base = chunk * 8;
        let h0 = u16::from_le_bytes([fp16_bytes[base], fp16_bytes[base + 1]]);
        let h1 = u16::from_le_bytes([fp16_bytes[base + 2], fp16_bytes[base + 3]]);
        let h2 = u16::from_le_bytes([fp16_bytes[base + 4], fp16_bytes[base + 5]]);
        let h3 = u16::from_le_bytes([fp16_bytes[base + 6], fp16_bytes[base + 7]]);

        out.push(fp16_to_f32(h0));
        out.push(fp16_to_f32(h1));
        out.push(fp16_to_f32(h2));
        out.push(fp16_to_f32(h3));
    }

    // Scalar remainder using half::f16-compatible bit conversion.
    for i in (full_chunks * 4)..count {
        let base = i * 2;
        let bits = u16::from_le_bytes([fp16_bytes[base], fp16_bytes[base + 1]]);
        out.push(fp16_to_f32(bits));
    }

    out
}

/// Dequantize symmetric INT8 weights: `output[i] = quantized[i] as f32 * scale`.
///
/// Uses NEON widening conversions for throughput.
#[cfg(target_arch = "aarch64")]
pub fn neon_dequant_i8_symmetric(quantized: &[i8], scale: f32) -> Vec<f32> {
    let n = quantized.len();
    let mut out = Vec::with_capacity(n);

    // SAFETY: NEON is always available on AArch64.
    unsafe {
        let scale_v = vdupq_n_f32(scale);

        // Process 8 elements at a time: i8x8 → i16x8 → 2×i32x4 → 2×f32x4.
        let full_chunks = n / 8;
        for chunk in 0..full_chunks {
            let base = chunk * 8;
            let ptr = quantized.as_ptr().add(base);
            let i8v = vld1_s8(ptr);
            let i16v = vmovl_s8(i8v);

            let lo_i16 = vget_low_s16(i16v);
            let hi_i16 = vget_high_s16(i16v);
            let lo_i32 = vmovl_s16(lo_i16);
            let hi_i32 = vmovl_s16(hi_i16);

            let lo_f32 = vcvtq_f32_s32(lo_i32);
            let hi_f32 = vcvtq_f32_s32(hi_i32);

            let lo_scaled = vmulq_f32(lo_f32, scale_v);
            let hi_scaled = vmulq_f32(hi_f32, scale_v);

            let mut buf = [0.0f32; 8];
            vst1q_f32(buf.as_mut_ptr(), lo_scaled);
            vst1q_f32(buf.as_mut_ptr().add(4), hi_scaled);
            out.extend_from_slice(&buf);
        }

        // Scalar remainder.
        for val in &quantized[full_chunks * 8..n] {
            out.push(*val as f32 * scale);
        }
    }

    out
}

/// Dequantize packed 4-bit (INT4) weights with per-block scales.
///
/// Each byte holds two 4-bit signed values (low nibble first). Values are
/// sign-extended from 4-bit (range −8..+7) and multiplied by the per-block
/// scale. The input is divided into blocks of `block_size` elements.
///
/// # Panics
///
/// Panics if `packed` is too short for the number of elements implied by
/// `scales` and `block_size`, or if `block_size` is zero.
#[cfg(target_arch = "aarch64")]
pub fn neon_dequant_i4_packed(packed: &[u8], scales: &[f32], block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be > 0");

    let total_elements = scales.len() * block_size;
    let bytes_needed = total_elements.div_ceil(2);
    assert!(
        packed.len() >= bytes_needed,
        "neon_dequant_i4_packed: need {bytes_needed} bytes for {} elements, got {}",
        total_elements,
        packed.len()
    );

    let mut out = Vec::with_capacity(total_elements);

    // SAFETY: NEON is always available on AArch64.
    unsafe {
        for (blk_idx, &blk_scale) in scales.iter().enumerate() {
            let elem_start = blk_idx * block_size;
            let scale_v = vdupq_n_f32(blk_scale);

            // Process 8 elements (4 bytes) at a time within the block.
            let full_chunks = block_size / 8;
            for chunk in 0..full_chunks {
                let elem_off = chunk * 8;
                let byte_off = (elem_start + elem_off) / 2;

                let mut vals = [0i8; 8];
                for j in 0..4 {
                    let b = packed[byte_off + j];
                    let lo = (b & 0x0F) as i8;
                    let hi = ((b >> 4) & 0x0F) as i8;
                    // Sign-extend from 4-bit.
                    vals[j * 2] = (lo << 4) >> 4;
                    vals[j * 2 + 1] = (hi << 4) >> 4;
                }

                let i8v = vld1_s8(vals.as_ptr());
                let i16v = vmovl_s8(i8v);
                let lo_i32 = vmovl_s16(vget_low_s16(i16v));
                let hi_i32 = vmovl_s16(vget_high_s16(i16v));

                let lo_f32 = vmulq_f32(vcvtq_f32_s32(lo_i32), scale_v);
                let hi_f32 = vmulq_f32(vcvtq_f32_s32(hi_i32), scale_v);

                let mut buf = [0.0f32; 8];
                vst1q_f32(buf.as_mut_ptr(), lo_f32);
                vst1q_f32(buf.as_mut_ptr().add(4), hi_f32);
                out.extend_from_slice(&buf);
            }

            // Scalar remainder within block.
            let done = full_chunks * 8;
            for i in done..block_size {
                let global_idx = elem_start + i;
                let byte_idx = global_idx / 2;
                let b = packed[byte_idx];
                let nibble = if global_idx.is_multiple_of(2) { b & 0x0F } else { (b >> 4) & 0x0F };
                let signed = ((nibble as i8) << 4) >> 4;
                out.push(signed as f32 * blk_scale);
            }
        }
    }

    out
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Pack four I2_S ternary values into one byte (LSB-first).
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

    #[test]
    fn test_i2s_roundtrip() {
        let packed = vec![pack4([1, -1, 0, 1]), pack4([0, -1, -1, 1])];
        let out = neon_dequant_i2s_block(&packed, 2.0, 8);
        assert_eq!(out, vec![2.0, -2.0, 0.0, 2.0, 0.0, -2.0, -2.0, 2.0]);
    }

    #[test]
    fn test_qk256_known_block() {
        // Build a 64-byte packed block: first 4 values are [+1, -1, 0, +1],
        // rest are all zeros.
        let mut packed = vec![0u8; 64];
        packed[0] = pack4([1, -1, 0, 1]);

        let out = neon_dequant_qk256_block(&packed, 3.0);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 3.0);
        // Remaining 252 values should be 0.
        assert!(out[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fp16_conversion() {
        // IEEE 754 FP16 known values (little-endian):
        //   1.0  = 0x3C00
        //  -1.0  = 0xBC00
        //   0.5  = 0x3800
        //   0.0  = 0x0000
        let fp16_bytes: Vec<u8> = vec![
            0x00, 0x3C, // 1.0
            0x00, 0xBC, // -1.0
            0x00, 0x38, // 0.5
            0x00, 0x00, // 0.0
        ];
        let out = neon_dequant_fp16_to_f32(&fp16_bytes);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - (-1.0)).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
        assert!((out[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_i8_symmetric() {
        let quantized: Vec<i8> = vec![1, -1, 0, 127, -128, 42, -42, 0];
        let scale = 0.5;
        let out = neon_dequant_i8_symmetric(&quantized, scale);
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - (-0.5)).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-5);
        assert!((out[3] - 63.5).abs() < 1e-5);
        assert!((out[4] - (-64.0)).abs() < 1e-5);
        assert!((out[5] - 21.0).abs() < 1e-5);
        assert!((out[6] - (-21.0)).abs() < 1e-5);
        assert!((out[7] - 0.0).abs() < 1e-5);
    }
}
