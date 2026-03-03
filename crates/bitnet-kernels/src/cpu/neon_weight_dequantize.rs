//! ARM NEON-optimized weight dequantization kernels for 1-bit/2-bit inference.
//!
//! Provides efficient unpacking and dequantization of quantized weights
//! using NEON SIMD intrinsics on AArch64 (Apple Silicon / ARM servers).
//!
//! Supported formats:
//! - **I2_S ternary**: 2-bit packed encoding (4 values per byte)
//!   with mapping 0b00→0, 0b01→+1, 0b11→−1, 0b10→0 (unused/reserved)
//! - **INT8**: 8-bit symmetric/asymmetric with zero-point offset
//! - **Ternary packing**: f32 → 2-bit quantization for weight compression
//! - **Fused dequant+dot**: avoids materializing intermediate f32 weights

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar helpers ──────────────────────────────────────────────────

/// Decode a single 2-bit code to its ternary value.
/// 0b00→0, 0b01→+1, 0b11→−1, 0b10→0 (reserved)
#[inline(always)]
fn decode_i2(code: u8) -> f32 {
    match code & 0b11 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0, // 0b00 and 0b10 both map to 0
    }
}

/// Unpack one byte into 4 ternary f32 values (LSB-first).
#[inline(always)]
fn unpack_byte_scalar(byte: u8) -> [f32; 4] {
    [decode_i2(byte), decode_i2(byte >> 2), decode_i2(byte >> 4), decode_i2(byte >> 6)]
}

#[inline(always)]
fn map_code_to_f32(code: u8, pos: f32, neg: f32) -> f32 {
    match code & 0b11 {
        0b01 => pos,
        0b11 => neg,
        _ => 0.0,
    }
}

// ── I2_S dequantization (uniform scale) ─────────────────────────────

/// Dequantize 2-bit packed ternary weights to f32 with a uniform scale.
///
/// Each byte in `packed` encodes 4 ternary values (LSB-first, 2 bits each):
/// - `0b00` → 0.0
/// - `0b01` → +scale
/// - `0b11` → −scale
/// - `0b10` → 0.0 (reserved)
///
/// # Panics
///
/// Panics if `output.len() < packed.len() * 4`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_dequantize_i2_to_f32(packed: &[u8], scale: f32, output: &mut [f32]) {
    assert!(
        output.len() >= packed.len() * 4,
        "output too small: need {} but got {}",
        packed.len() * 4,
        output.len()
    );

    let total_values = packed.len() * 4;

    // Process 4 bytes at a time → 16 f32 outputs
    let chunk_bytes = 4;
    let full_chunks = packed.len() / chunk_bytes;

    for i in 0..full_chunks {
        let base_byte = i * chunk_bytes;
        let base_out = base_byte * 4;

        for j in 0..chunk_bytes {
            let byte = packed[base_byte + j];
            let out_idx = base_out + j * 4;

            let c0 = byte & 0x03;
            let c1 = (byte >> 2) & 0x03;
            let c2 = (byte >> 4) & 0x03;
            let c3 = (byte >> 6) & 0x03;

            let vals = [
                map_code_to_f32(c0, scale, -scale),
                map_code_to_f32(c1, scale, -scale),
                map_code_to_f32(c2, scale, -scale),
                map_code_to_f32(c3, scale, -scale),
            ];

            unsafe {
                let v = vld1q_f32(vals.as_ptr());
                vst1q_f32(output.as_mut_ptr().add(out_idx), v);
            }
        }
    }

    // Scalar tail for remaining bytes
    let tail_start = full_chunks * chunk_bytes;
    for i in tail_start..packed.len() {
        let byte = packed[i];
        let out_base = i * 4;
        let unpacked = unpack_byte_scalar(byte);
        for (k, &val) in unpacked.iter().enumerate() {
            if out_base + k < total_values {
                output[out_base + k] = val * scale;
            }
        }
    }
}

// ── I2_S per-block dequantization ───────────────────────────────────

/// Dequantize 2-bit packed ternary weights with per-block scales.
///
/// Weights are divided into blocks of `block_size` values, each scaled
/// by the corresponding entry in `scales`.
///
/// # Panics
///
/// Panics if `output.len() < packed.len() * 4` or if the number of
/// blocks exceeds `scales.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_dequantize_i2_block_f32(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output: &mut [f32],
) {
    let total_values = packed.len() * 4;
    assert!(
        output.len() >= total_values,
        "output too small: need {} but got {}",
        total_values,
        output.len()
    );
    assert!(block_size > 0, "block_size must be > 0");

    let num_blocks = (total_values + block_size - 1) / block_size;
    assert!(
        scales.len() >= num_blocks,
        "not enough scales: need {} but got {}",
        num_blocks,
        scales.len()
    );

    // First pass: scalar unpack with per-element block lookup
    for i in 0..packed.len() {
        let byte = packed[i];
        let out_base = i * 4;
        let unpacked = unpack_byte_scalar(byte);
        for k in 0..4 {
            let val_idx = out_base + k;
            if val_idx < total_values {
                let block_idx = val_idx / block_size;
                let s = scales[block_idx];
                output[val_idx] = unpacked[k] * s;
            }
        }
    }

    // NEON acceleration for aligned blocks (block_size multiple of 4)
    if block_size >= 4 && block_size % 4 == 0 {
        for i in 0..packed.len() {
            let out_base = i * 4;
            let block_idx = out_base / block_size;
            let next_block_start = (block_idx + 1) * block_size;

            // If all 4 values are in the same block, use NEON
            if out_base + 4 <= next_block_start && out_base + 4 <= total_values {
                let s = scales[block_idx];
                let byte = packed[i];

                let vals = [
                    decode_i2(byte),
                    decode_i2(byte >> 2),
                    decode_i2(byte >> 4),
                    decode_i2(byte >> 6),
                ];
                unsafe {
                    let scale_v = vdupq_n_f32(s);
                    let raw_v = vld1q_f32(vals.as_ptr());
                    let result = vmulq_f32(raw_v, scale_v);
                    vst1q_f32(output.as_mut_ptr().add(out_base), result);
                }
            }
        }
    }
}

// ── INT8 dequantization ─────────────────────────────────────────────

/// Dequantize INT8 weights to f32: output = (data - zero_point) * scale.
///
/// Uses NEON to process 8 values at a time through the i8→i16→i32→f32
/// widening pipeline with fused subtract and multiply.
///
/// # Panics
///
/// Panics if `output.len() < data.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_dequantize_i8_to_f32(
    data: &[i8],
    scale: f32,
    zero_point: i8,
    output: &mut [f32],
) {
    assert!(
        output.len() >= data.len(),
        "output too small: need {} but got {}",
        data.len(),
        output.len()
    );

    let n = data.len();
    let zp = zero_point as i32;

    // Process 8 elements at a time using NEON
    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        unsafe {
            let scale_v = vdupq_n_f32(scale);
            let zp_v = vdupq_n_s32(zp);
            let ptr = data.as_ptr().add(base);

            // Load 8 x i8 → widen to i16 → split to 2x i32x4
            let raw8 = vld1_s8(ptr);
            let wide16 = vmovl_s8(raw8);
            let lo16 = vget_low_s16(wide16);
            let hi16 = vget_high_s16(wide16);
            let lo32 = vmovl_s16(lo16);
            let hi32 = vmovl_s16(hi16);
            // Subtract zero point and convert to f32
            let sub_lo = vsubq_s32(lo32, zp_v);
            let sub_hi = vsubq_s32(hi32, zp_v);
            let f_lo = vcvtq_f32_s32(sub_lo);
            let f_hi = vcvtq_f32_s32(sub_hi);
            // Multiply by scale
            let r_lo = vmulq_f32(f_lo, scale_v);
            let r_hi = vmulq_f32(f_hi, scale_v);

            vst1q_f32(output.as_mut_ptr().add(base), r_lo);
            vst1q_f32(output.as_mut_ptr().add(base + 4), r_hi);
        }
    }

    // Scalar tail
    let tail_start = chunks * 8;
    for i in tail_start..n {
        output[i] = (data[i] as f32 - zero_point as f32) * scale;
    }
}

// ── Ternary weight packing ──────────────────────────────────────────

/// Pack f32 weights into 2-bit ternary encoding.
///
/// - `|w| < threshold` → 0b00 (zero)
/// - `w >= threshold`  → 0b01 (+1)
/// - `w <= -threshold` → 0b11 (-1)
///
/// Output is packed LSB-first, 4 values per byte.
///
/// # Panics
///
/// Panics if `output.len() < (weights.len() + 3) / 4`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_pack_ternary_f32(weights: &[f32], threshold: f32, output: &mut [u8]) {
    let needed = (weights.len() + 3) / 4;
    assert!(output.len() >= needed, "output too small: need {} but got {}", needed, output.len());

    let n = weights.len();

    // Process 4 weights at a time → 1 output byte
    let full_groups = n / 4;
    for i in 0..full_groups {
        let base = i * 4;
        unsafe {
            let thresh_v = vdupq_n_f32(threshold);
            let neg_thresh_v = vnegq_f32(thresh_v);
            let w = vld1q_f32(weights.as_ptr().add(base));

            // Compare: w >= threshold → positive
            let pos_mask = vcgeq_f32(w, thresh_v);
            // Compare: w <= -threshold → negative
            let neg_mask = vcleq_f32(w, neg_thresh_v);

            let pos_bits = [
                vgetq_lane_u32(pos_mask, 0),
                vgetq_lane_u32(pos_mask, 1),
                vgetq_lane_u32(pos_mask, 2),
                vgetq_lane_u32(pos_mask, 3),
            ];
            let neg_bits = [
                vgetq_lane_u32(neg_mask, 0),
                vgetq_lane_u32(neg_mask, 1),
                vgetq_lane_u32(neg_mask, 2),
                vgetq_lane_u32(neg_mask, 3),
            ];

            let mut byte: u8 = 0;
            for k in 0..4 {
                let code = if pos_bits[k] != 0 {
                    0b01u8
                } else if neg_bits[k] != 0 {
                    0b11u8
                } else {
                    0b00u8
                };
                byte |= code << (k * 2);
            }
            output[i] = byte;
        }
    }

    // Scalar tail
    let tail_start = full_groups * 4;
    if tail_start < n {
        let mut byte: u8 = 0;
        for k in 0..(n - tail_start) {
            let w = weights[tail_start + k];
            let code = if w >= threshold {
                0b01u8
            } else if w <= -threshold {
                0b11u8
            } else {
                0b00u8
            };
            byte |= code << (k * 2);
        }
        output[full_groups] = byte;
    }
}

// ── Fused dequantize + dot product ──────────────────────────────────

/// Fused dequantize-and-dot-product: computes `sum(dequant(packed) * vector)`
/// without materializing the full f32 weight buffer.
///
/// Since ternary weights are ±1 or 0, the "multiply" reduces to
/// add/subtract/skip, making this significantly faster than a separate
/// dequant followed by dot product.
///
/// # Panics
///
/// Panics if `vector.len() < packed.len() * 4`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_dequant_dot_f32(packed: &[u8], scale: f32, vector: &[f32]) -> f32 {
    let total_values = packed.len() * 4;
    assert!(
        vector.len() >= total_values,
        "vector too small: need {} but got {}",
        total_values,
        vector.len()
    );

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..packed.len() {
        let byte = packed[i];
        let base = i * 4;

        let c0 = byte & 0x03;
        let c1 = (byte >> 2) & 0x03;
        let c2 = (byte >> 4) & 0x03;
        let c3 = (byte >> 6) & 0x03;

        // Build sign vector and multiply-accumulate using NEON
        let signs = [decode_i2(c0), decode_i2(c1), decode_i2(c2), decode_i2(c3)];
        unsafe {
            let v = vld1q_f32(vector.as_ptr().add(base));
            let sign_v = vld1q_f32(signs.as_ptr());
            acc = vmlaq_f32(acc, v, sign_v);
        }
    }

    // Horizontal sum and apply scale
    let sum = vaddvq_f32(acc) * scale;
    sum
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scalar reference implementations for validation ─────────

    fn ref_dequant_i2(packed: &[u8], scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(packed.len() * 4);
        for &byte in packed {
            for shift in [0, 2, 4, 6] {
                let code = (byte >> shift) & 0x03;
                let val = match code {
                    0b01 => 1.0,
                    0b11 => -1.0,
                    _ => 0.0,
                };
                out.push(val * scale);
            }
        }
        out
    }

    fn ref_dequant_i2_block(packed: &[u8], scales: &[f32], block_size: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(packed.len() * 4);
        for (i, &byte) in packed.iter().enumerate() {
            for (k, shift) in [0, 2, 4, 6].iter().enumerate() {
                let idx = i * 4 + k;
                let block = idx / block_size;
                let s = scales[block];
                let code = (byte >> shift) & 0x03;
                let val = match code {
                    0b01 => 1.0,
                    0b11 => -1.0,
                    _ => 0.0,
                };
                out.push(val * s);
            }
        }
        out
    }

    fn ref_dequant_i8(data: &[i8], scale: f32, zp: i8) -> Vec<f32> {
        data.iter().map(|&d| (d as f32 - zp as f32) * scale).collect()
    }

    fn ref_pack_ternary(weights: &[f32], threshold: f32) -> Vec<u8> {
        let mut out = vec![0u8; (weights.len() + 3) / 4];
        for (i, &w) in weights.iter().enumerate() {
            let code = if w >= threshold {
                0b01u8
            } else if w <= -threshold {
                0b11u8
            } else {
                0b00u8
            };
            out[i / 4] |= code << ((i % 4) * 2);
        }
        out
    }

    fn ref_dequant_dot(packed: &[u8], scale: f32, vector: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &byte) in packed.iter().enumerate() {
            for (k, shift) in [0, 2, 4, 6].iter().enumerate() {
                let idx = i * 4 + k;
                let code = (byte >> shift) & 0x03;
                let val = match code {
                    0b01 => 1.0,
                    0b11 => -1.0,
                    _ => 0.0,
                };
                sum += val * vector[idx];
            }
        }
        sum * scale
    }

    fn assert_f32_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {}: {} vs {} (tol={})", i, x, y, tol);
        }
    }

    fn pack_codes(codes: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; (codes.len() + 3) / 4];
        for (i, &c) in codes.iter().enumerate() {
            out[i / 4] |= (c & 0x03) << ((i % 4) * 2);
        }
        out
    }

    // ── dequant_i2 tests ────────────────────────────────────────

    #[test]
    fn dequant_i2_all_zeros() {
        let packed = vec![0x00; 4];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_all_ones() {
        let packed = vec![0x55; 4]; // 0b01010101 → all +1
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_all_neg_ones() {
        let packed = vec![0xFF; 4]; // 0b11111111 → all -1
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_mixed() {
        // byte: 0b11_00_01_00 = 0xC4 → [0, +1, 0, -1]
        let packed = vec![0xC4];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_single_byte() {
        let packed = vec![0x55];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_dequantize_i2_to_f32(&packed, 2.5, &mut output) };
        let expected = ref_dequant_i2(&packed, 2.5);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_multi_byte() {
        let packed = vec![0x55, 0xFF, 0x00, 0xC4, 0x55];
        let mut output = vec![0.0f32; 20];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_with_scale() {
        let packed = vec![0x55; 2];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_to_f32(&packed, 0.125, &mut output) };
        let expected = ref_dequant_i2(&packed, 0.125);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_large() {
        let packed = vec![0x55; 256]; // 1024 values
        let mut output = vec![0.0f32; 1024];
        unsafe { neon_dequantize_i2_to_f32(&packed, 0.5, &mut output) };
        let expected = ref_dequant_i2(&packed, 0.5);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_pattern_0b10() {
        // 0b10101010 = 0xAA → all 0b10 (reserved → 0)
        let packed = vec![0xAA; 4];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_i2_roundtrip() {
        let codes = vec![0b01, 0b11, 0b00, 0b01, 0b11, 0b11, 0b01, 0b00];
        let packed = pack_codes(&codes);
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected: Vec<f32> = codes
            .iter()
            .map(|&c| match c {
                0b01 => 1.0,
                0b11 => -1.0,
                _ => 0.0,
            })
            .collect();
        assert_f32_eq(&output, &expected, 1e-6);
    }

    // ── dequant_i2_block tests ──────────────────────────────────

    #[test]
    fn dequant_i2_block_single_block() {
        let packed = vec![0x55; 2]; // 8 values, all +1
        let scales = vec![3.0];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 8, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 8);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_multi_block() {
        let packed = vec![0x55, 0xFF]; // 8 values
        let scales = vec![2.0, 0.5];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 4, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 4);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_varying_scales() {
        let packed = vec![0x55; 4]; // 16 values, all +1
        let scales = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 4, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 4);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_zero_scale() {
        let packed = vec![0xFF; 2];
        let scales = vec![0.0, 0.0];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 4, &mut output) };
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequant_i2_block_negative_scale() {
        let packed = vec![0x55; 2]; // all +1
        let scales = vec![-1.0, -2.0];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 4, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 4);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_size_32() {
        let packed = vec![0x55; 8]; // 32 values
        let scales = vec![1.5];
        let mut output = vec![0.0f32; 32];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 32, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 32);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_size_256() {
        let packed = vec![0xFF; 64]; // 256 values
        let scales = vec![0.25];
        let mut output = vec![0.0f32; 256];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 256, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 256);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_misaligned() {
        // 3 bytes = 12 values, block_size=8 → 2 blocks (8 + 4)
        let packed = vec![0x55, 0xFF, 0x00];
        let scales = vec![1.0, 2.0];
        let mut output = vec![0.0f32; 12];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 8, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 8);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    // ── dequant_i8 tests ────────────────────────────────────────

    #[test]
    fn dequant_i8_zeros() {
        let data = vec![0i8; 16];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i8_to_f32(&data, 1.0, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 1.0, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_positive() {
        let data: Vec<i8> = (1..=16).map(|x| x as i8).collect();
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.5, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 0.5, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_negative() {
        let data: Vec<i8> = (-16..0).map(|x| x as i8).collect();
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i8_to_f32(&data, 1.0, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 1.0, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_mixed() {
        let data = vec![-128i8, -64, -1, 0, 1, 64, 127, 42, -42, 100];
        let mut output = vec![0.0f32; 10];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.01, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 0.01, 0);
        assert_f32_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn dequant_i8_with_zero_point() {
        let data = vec![10i8, 20, 30, 40, 50, 60, 70, 80];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i8_to_f32(&data, 1.0, 10, &mut output) };
        let expected = ref_dequant_i8(&data, 1.0, 10);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_scale_factor() {
        let data = vec![1i8, 2, 3, 4, 5, 6, 7, 8];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.125, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 0.125, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_large() {
        let data: Vec<i8> = (0..256).map(|x| x as i8).collect();
        let mut output = vec![0.0f32; 256];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.1, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 0.1, 0);
        assert_f32_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn dequant_i8_precision() {
        let data = vec![127i8, -128];
        let mut output = vec![0.0f32; 2];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.00784314, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 0.00784314, 0);
        assert_f32_eq(&output, &expected, 1e-4);
    }

    // ── pack_ternary tests ──────────────────────────────────────

    #[test]
    fn pack_ternary_all_zero() {
        let weights = vec![0.0f32; 8];
        let mut output = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..2], &expected[..]);
    }

    #[test]
    fn pack_ternary_all_positive() {
        let weights = vec![1.0f32; 8];
        let mut output = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..2], &expected[..]);
    }

    #[test]
    fn pack_ternary_all_negative() {
        let weights = vec![-1.0f32; 8];
        let mut output = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..2], &expected[..]);
    }

    #[test]
    fn pack_ternary_mixed() {
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.5, 0.1, -2.0, 3.0];
        let mut output = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..2], &expected[..]);
    }

    #[test]
    fn pack_ternary_threshold_sweep() {
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for threshold in [0.1, 0.25, 0.5, 0.75] {
            let mut output = vec![0u8; 2];
            unsafe { neon_pack_ternary_f32(&weights, threshold, &mut output) };
            let expected = ref_pack_ternary(&weights, threshold);
            assert_eq!(&output[..2], &expected[..], "threshold={threshold}");
        }
    }

    #[test]
    fn pack_ternary_roundtrip_with_dequant() {
        let weights = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let mut packed = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut packed) };
        let mut recovered = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut recovered) };
        assert_f32_eq(&recovered, &weights, 1e-6);
    }

    #[test]
    fn pack_ternary_large() {
        let weights: Vec<f32> = (0..128)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let mut output = vec![0u8; 32];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..32], &expected[..]);
    }

    #[test]
    fn pack_ternary_boundary() {
        // Values exactly at threshold
        let weights = vec![0.5, -0.5, 0.5, -0.5];
        let mut output = vec![0u8; 1];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..1], &expected[..]);
    }

    // ── dequant_dot tests ───────────────────────────────────────

    #[test]
    fn dequant_dot_identity() {
        let packed = vec![0x55]; // 4x +1
        let vector = vec![1.0f32; 4];
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-5, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_zeros() {
        let packed = vec![0x00; 4]; // all zero weights
        let vector = vec![42.0f32; 16];
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        assert!((result - 0.0).abs() < 1e-5);
    }

    #[test]
    fn dequant_dot_ones_vector() {
        let packed = vec![0x55; 4]; // 16x +1
        let vector = vec![1.0f32; 16];
        let result = unsafe { neon_dequant_dot_f32(&packed, 2.0, &vector) };
        let expected = ref_dequant_dot(&packed, 2.0, &vector);
        assert!((result - expected).abs() < 1e-5, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_mixed() {
        let packed = vec![0xC4]; // [0, +1, 0, -1]
        let vector = vec![10.0, 20.0, 30.0, 40.0];
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-5, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_scale_effect() {
        let packed = vec![0x55; 2]; // 8x +1
        let vector = vec![1.0f32; 8];
        let r1 = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let r2 = unsafe { neon_dequant_dot_f32(&packed, 3.0, &vector) };
        assert!((r2 - r1 * 3.0).abs() < 1e-5);
    }

    #[test]
    fn dequant_dot_large() {
        let packed = vec![0x55; 64]; // 256x +1
        let vector: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-2, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_precision() {
        let packed = vec![0xFF; 8]; // 32x -1
        let vector: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-4, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_fused_vs_separate() {
        let packed = vec![0x55, 0xFF, 0xC4, 0x00];
        let vector: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let scale = 2.5;

        let fused = unsafe { neon_dequant_dot_f32(&packed, scale, &vector) };

        // Separate: dequant then dot
        let mut dequant = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut dequant) };
        let separate: f32 =
            dequant.iter().zip(vector.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;

        assert!((fused - separate).abs() < 1e-4, "fused={fused} vs separate={separate}");
    }

    // ── edge_cases ──────────────────────────────────────────────

    #[test]
    fn edge_empty_i2() {
        let packed: Vec<u8> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn edge_empty_i8() {
        let data: Vec<i8> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_dequantize_i8_to_f32(&data, 1.0, 0, &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn edge_empty_dot() {
        let packed: Vec<u8> = vec![];
        let vector: Vec<f32> = vec![];
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        assert_eq!(result, 0.0);
    }

    #[test]
    fn edge_empty_pack() {
        let weights: Vec<f32> = vec![];
        let mut output: Vec<u8> = vec![];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn edge_single_byte_i2() {
        let packed = vec![0b01_11_01_00u8]; // [0, +1, -1, +1]
        let mut output = vec![0.0f32; 4];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn edge_single_element_i8() {
        let data = vec![42i8];
        let mut output = vec![0.0f32; 1];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.5, 0, &mut output) };
        assert!((output[0] - 21.0).abs() < 1e-6);
    }

    #[test]
    fn edge_non_aligned_i8() {
        let data = vec![1i8, 2, 3, 4, 5];
        let mut output = vec![0.0f32; 5];
        unsafe { neon_dequantize_i8_to_f32(&data, 1.0, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 1.0, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn edge_large_scale() {
        let packed = vec![0x55; 4]; // all +1
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1e6, &mut output) };
        let expected = ref_dequant_i2(&packed, 1e6);
        assert_f32_eq(&output, &expected, 1.0);
    }

    #[test]
    fn edge_tiny_scale() {
        let packed = vec![0x55; 4]; // all +1
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1e-7, &mut output) };
        let expected = ref_dequant_i2(&packed, 1e-7);
        assert_f32_eq(&output, &expected, 1e-12);
    }

    #[test]
    fn edge_overflow_safe_i8() {
        let data = vec![127i8, -128];
        let mut output = vec![0.0f32; 2];
        unsafe { neon_dequantize_i8_to_f32(&data, 1e6, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 1e6, 0);
        assert_f32_eq(&output, &expected, 100.0);
    }

    #[test]
    fn dequant_i2_neg_scale() {
        let packed = vec![0x55; 2]; // all +1
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i2_to_f32(&packed, -0.5, &mut output) };
        let expected = ref_dequant_i2(&packed, -0.5);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_negative_zero_point() {
        let data = vec![0i8, 10, -10, 50, -50, 100, -100, 127];
        let mut output = vec![0.0f32; 8];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.1, -5, &mut output) };
        let expected = ref_dequant_i8(&data, 0.1, -5);
        assert_f32_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn pack_ternary_tail_3() {
        let weights = vec![1.0, -1.0, 0.0];
        let mut output = vec![0u8; 1];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..1], &expected[..]);
    }

    #[test]
    fn pack_ternary_tail_5() {
        let weights = vec![1.0, -1.0, 0.0, 1.0, -1.0];
        let mut output = vec![0u8; 2];
        unsafe { neon_pack_ternary_f32(&weights, 0.5, &mut output) };
        let expected = ref_pack_ternary(&weights, 0.5);
        assert_eq!(&output[..2], &expected[..]);
    }

    #[test]
    fn dequant_dot_neg_weights() {
        let packed = vec![0xFF; 4]; // 16x -1
        let vector = vec![2.0f32; 16];
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-4, "{result} vs {expected}");
    }

    #[test]
    fn dequant_dot_alternating() {
        let packed = vec![0b11_01_11_01u8; 4]; // [+1, -1, +1, -1] repeated
        let vector: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = unsafe { neon_dequant_dot_f32(&packed, 1.0, &vector) };
        let expected = ref_dequant_dot(&packed, 1.0, &vector);
        assert!((result - expected).abs() < 1e-4, "{result} vs {expected}");
    }

    #[test]
    fn dequant_i2_all_codes_byte() {
        // One byte with all 4 distinct codes: 0b10_11_01_00
        let packed = vec![0b10_11_01_00u8];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_dequantize_i2_to_f32(&packed, 1.0, &mut output) };
        let expected = ref_dequant_i2(&packed, 1.0);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i2_block_block_size_4() {
        let packed = vec![0x55, 0xFF, 0x00, 0xAA]; // 16 values
        let scales = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i2_block_f32(&packed, &scales, 4, &mut output) };
        let expected = ref_dequant_i2_block(&packed, &scales, 4);
        assert_f32_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn dequant_i8_all_same() {
        let data = vec![42i8; 16];
        let mut output = vec![0.0f32; 16];
        unsafe { neon_dequantize_i8_to_f32(&data, 0.1, 42, &mut output) };
        for &v in &output {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_i8_seven_elements() {
        let data = vec![1i8, -1, 2, -2, 3, -3, 4];
        let mut output = vec![0.0f32; 7];
        unsafe { neon_dequantize_i8_to_f32(&data, 2.0, 0, &mut output) };
        let expected = ref_dequant_i8(&data, 2.0, 0);
        assert_f32_eq(&output, &expected, 1e-6);
    }
}
