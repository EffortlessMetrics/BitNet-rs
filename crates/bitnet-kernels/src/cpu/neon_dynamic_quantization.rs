#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON dynamic quantization kernels for Apple Silicon.
//!
//! Provides NEON-optimized dynamic quantization operations including
//! scale computation, symmetric quantization/dequantization, ternary
//! quantization (I2_S packed), and per-block absolute-max reduction.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Compute the quantization scale as the maximum absolute value of `input`.
///
/// Uses NEON `vabsq_f32` / `vmaxq_f32` for vectorised max-abs reduction,
/// with scalar cleanup for the tail. Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_compute_scale(input: &[f32]) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = input.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let abs_v = vabsq_f32(v);
        acc = vmaxq_f32(acc, abs_v);
    }

    // Horizontal max of the 4-lane accumulator.
    let mut scale = vmaxvq_f32(acc);

    for i in 0..remainder {
        let val = (*ptr.add(chunks * 4 + i)).abs();
        if val > scale {
            scale = val;
        }
    }

    scale
}

/// Symmetric quantization of `f32` values to `i8` using NEON.
///
/// Each element is mapped to `round(x / scale * 127)` and clamped to
/// `[-127, 127]`. If `scale` is zero or subnormal the output is all zeros.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_quantize_symmetric(input: &[f32], scale: f32) -> Vec<i8> {
    let len = input.len();
    let mut output = vec![0i8; len];

    if scale == 0.0 || !scale.is_finite() {
        return output;
    }

    let inv_scale = 127.0 / scale;
    let ptr_in = input.as_ptr();
    let ptr_out = output.as_mut_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let inv_scale_v = vdupq_n_f32(inv_scale);
    let min_v = vdupq_n_f32(-127.0);
    let max_v = vdupq_n_f32(127.0);

    for i in 0..chunks {
        let v = vld1q_f32(ptr_in.add(i * 4));
        // scale, clamp, convert
        let scaled = vmulq_f32(v, inv_scale_v);
        let clamped = vminq_f32(vmaxq_f32(scaled, min_v), max_v);
        let as_i32 = vcvtq_s32_f32(clamped); // round-to-nearest

        // Extract each lane and narrow to i8.
        *ptr_out.add(i * 4) = vgetq_lane_s32(as_i32, 0) as i8;
        *ptr_out.add(i * 4 + 1) = vgetq_lane_s32(as_i32, 1) as i8;
        *ptr_out.add(i * 4 + 2) = vgetq_lane_s32(as_i32, 2) as i8;
        *ptr_out.add(i * 4 + 3) = vgetq_lane_s32(as_i32, 3) as i8;
    }

    // Scalar tail.
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        let val = (*ptr_in.add(idx) * inv_scale).round().clamp(-127.0, 127.0);
        *ptr_out.add(idx) = val as i8;
    }

    output
}

/// Dequantize `i8` values back to `f32` using NEON.
///
/// Each element is mapped to `q * scale / 127.0`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dequantize_symmetric(quantized: &[i8], scale: f32) -> Vec<f32> {
    let len = quantized.len();
    let mut output = vec![0.0f32; len];

    if len == 0 {
        return output;
    }

    let factor = scale / 127.0;
    let factor_v = vdupq_n_f32(factor);
    let ptr_in = quantized.as_ptr();
    let ptr_out = output.as_mut_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        // Load 4 i8 values, widen to i32, convert to f32.
        let b0 = *ptr_in.add(i * 4) as i32;
        let b1 = *ptr_in.add(i * 4 + 1) as i32;
        let b2 = *ptr_in.add(i * 4 + 2) as i32;
        let b3 = *ptr_in.add(i * 4 + 3) as i32;

        let i32x4 =
            vsetq_lane_s32(b3, vsetq_lane_s32(b2, vsetq_lane_s32(b1, vdupq_n_s32(b0), 1), 2), 3);
        let f32x4 = vcvtq_f32_s32(i32x4);
        let result = vmulq_f32(f32x4, factor_v);
        vst1q_f32(ptr_out.add(i * 4), result);
    }

    // Scalar tail.
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        *ptr_out.add(idx) = (*ptr_in.add(idx) as f32) * factor;
    }

    output
}

/// Quantize `f32` values to ternary {-1, 0, +1} packed as I2_S.
///
/// Encoding: `0b00` = 0, `0b01` = +1, `0b11` = -1 (4 values per byte,
/// LSB-first). Values with absolute magnitude below `threshold` map to 0;
/// positive above threshold map to +1; negative to -1.
///
/// NEON is used for the comparison / classification; packing is scalar
/// since the bit-packing pattern is irregular.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_quantize_to_ternary(input: &[f32], threshold: f32) -> Vec<u8> {
    let len = input.len();
    let out_bytes = (len + 3) / 4; // 4 values per byte
    let mut output = vec![0u8; out_bytes];

    let ptr_in = input.as_ptr();
    let pos_thresh = vdupq_n_f32(threshold);
    let neg_thresh = vdupq_n_f32(-threshold);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let v = vld1q_f32(ptr_in.add(i * 4));

        // Compare: pos_mask lanes where v > threshold, neg_mask where v < -threshold.
        let pos_mask = vcgtq_f32(v, pos_thresh);
        let neg_mask = vcltq_f32(v, neg_thresh);

        // Extract each lane (const index required by NEON intrinsics).
        let mut byte: u8 = 0;
        let pos = [
            vgetq_lane_u32(pos_mask, 0),
            vgetq_lane_u32(pos_mask, 1),
            vgetq_lane_u32(pos_mask, 2),
            vgetq_lane_u32(pos_mask, 3),
        ];
        let neg = [
            vgetq_lane_u32(neg_mask, 0),
            vgetq_lane_u32(neg_mask, 1),
            vgetq_lane_u32(neg_mask, 2),
            vgetq_lane_u32(neg_mask, 3),
        ];
        for j in 0..4u32 {
            let bits: u8 = if pos[j as usize] != 0 {
                0b01 // +1
            } else if neg[j as usize] != 0 {
                0b11 // -1
            } else {
                0b00 // 0
            };
            byte |= bits << (j * 2);
        }
        output[i] = byte;
    }

    // Scalar tail.
    if remainder > 0 {
        let mut byte: u8 = 0;
        for j in 0..remainder {
            let val = *ptr_in.add(chunks * 4 + j);
            let bits: u8 = if val > threshold {
                0b01
            } else if val < -threshold {
                0b11
            } else {
                0b00
            };
            byte |= bits << (j * 2);
        }
        output[chunks] = byte;
    }

    output
}

/// Per-block absolute-max reduction using NEON.
///
/// Divides `input` into non-overlapping blocks of `block_size` elements and
/// returns the maximum absolute value within each block. The last block may
/// be smaller than `block_size`.
///
/// Returns an empty vector if `input` is empty or `block_size` is zero.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_absmax_per_block(input: &[f32], block_size: usize) -> Vec<f32> {
    let len = input.len();
    if len == 0 || block_size == 0 {
        return Vec::new();
    }

    let num_blocks = (len + block_size - 1) / block_size;
    let mut output = Vec::with_capacity(num_blocks);
    let ptr = input.as_ptr();

    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(len);
        let block_len = end - start;
        let block_ptr = ptr.add(start);

        let chunks = block_len / 4;
        let remainder = block_len % 4;

        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(block_ptr.add(i * 4));
            let abs_v = vabsq_f32(v);
            acc = vmaxq_f32(acc, abs_v);
        }

        let mut block_max = vmaxvq_f32(acc);

        for i in 0..remainder {
            let val = (*block_ptr.add(chunks * 4 + i)).abs();
            if val > block_max {
                block_max = val;
            }
        }

        output.push(block_max);
    }

    output
}

// ── Convenience wrappers ───────────────────────────────────────────────

/// Safe wrapper around [`neon_compute_scale`].
#[cfg(target_arch = "aarch64")]
pub fn compute_scale(input: &[f32]) -> f32 {
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_compute_scale(input) }
}

/// Safe wrapper around [`neon_quantize_symmetric`].
#[cfg(target_arch = "aarch64")]
pub fn quantize_symmetric(input: &[f32], scale: f32) -> Vec<i8> {
    unsafe { neon_quantize_symmetric(input, scale) }
}

/// Safe wrapper around [`neon_dequantize_symmetric`].
#[cfg(target_arch = "aarch64")]
pub fn dequantize_symmetric(quantized: &[i8], scale: f32) -> Vec<f32> {
    unsafe { neon_dequantize_symmetric(quantized, scale) }
}

/// Safe wrapper around [`neon_quantize_to_ternary`].
#[cfg(target_arch = "aarch64")]
pub fn quantize_to_ternary(input: &[f32], threshold: f32) -> Vec<u8> {
    unsafe { neon_quantize_to_ternary(input, threshold) }
}

/// Safe wrapper around [`neon_absmax_per_block`].
#[cfg(target_arch = "aarch64")]
pub fn absmax_per_block(input: &[f32], block_size: usize) -> Vec<f32> {
    unsafe { neon_absmax_per_block(input, block_size) }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_compute_scale() {
        let input = [1.0f32, -3.5, 2.0, -0.5, 3.0, -1.0, 0.0, 0.25];
        let scale = compute_scale(&input);
        assert!((scale - 3.5).abs() < 1e-6, "expected 3.5, got {scale}");

        // Single-element and empty edge cases.
        assert!((compute_scale(&[-7.0]) - 7.0).abs() < 1e-6);
        assert_eq!(compute_scale(&[]), 0.0);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let input: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -3.0, 0.0, 1.5, -0.25];
        let scale = compute_scale(&input);
        let quantized = quantize_symmetric(&input, scale);
        let recovered = dequantize_symmetric(&quantized, scale);

        assert_eq!(recovered.len(), input.len());
        for (orig, rec) in input.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            // Max quantization error for 8-bit symmetric: scale / 127.
            assert!(
                err < scale / 127.0 + 1e-5,
                "roundtrip error too large: orig={orig}, rec={rec}, err={err}"
            );
        }
    }

    #[test]
    fn test_ternary_quantization() {
        // Threshold = 0.5: values > 0.5 → +1, < -0.5 → -1, else → 0.
        let input = [1.0f32, -1.0, 0.3, 0.0, 2.0, -0.1, -3.0, 0.5];
        let packed = quantize_to_ternary(&input, 0.5);

        // Decode helper.
        let decode = |byte: u8, idx: u32| -> i8 {
            match (byte >> (idx * 2)) & 0x03 {
                0b01 => 1,
                0b11 => -1,
                _ => 0,
            }
        };

        // First 4 values packed in byte 0.
        assert_eq!(decode(packed[0], 0), 1); // 1.0  → +1
        assert_eq!(decode(packed[0], 1), -1); // -1.0 → -1
        assert_eq!(decode(packed[0], 2), 0); // 0.3  → 0
        assert_eq!(decode(packed[0], 3), 0); // 0.0  → 0

        // Next 4 values packed in byte 1.
        assert_eq!(decode(packed[1], 0), 1); // 2.0  → +1
        assert_eq!(decode(packed[1], 1), 0); // -0.1 → 0
        assert_eq!(decode(packed[1], 2), -1); // -3.0 → -1
        assert_eq!(decode(packed[1], 3), 0); // 0.5  → 0 (not strictly > 0.5)
    }

    #[test]
    fn test_absmax_per_block() {
        let input = [1.0f32, -4.0, 2.0, 3.0, -1.0, 5.0, 0.5, -0.5, 7.0];
        let result = absmax_per_block(&input, 4);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-6); // block [1, -4, 2, 3]
        assert!((result[1] - 5.0).abs() < 1e-6); // block [-1, 5, 0.5, -0.5]
        assert!((result[2] - 7.0).abs() < 1e-6); // block [7] (partial)

        // Edge cases.
        assert!(absmax_per_block(&[], 4).is_empty());
        assert!(absmax_per_block(&[1.0], 0).is_empty());
    }
}
