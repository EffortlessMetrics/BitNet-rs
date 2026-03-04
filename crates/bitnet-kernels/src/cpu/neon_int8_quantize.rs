//! NEON-optimized int8 symmetric quantization kernels for Apple Silicon (aarch64).
//!
//! Provides five quantization/matmul operations with NEON intrinsics and scalar
//! fallbacks:
//!
//! 1. `quantize_f32_to_i8_neon` — f32 → int8 with symmetric scale
//! 2. `dequantize_i8_to_f32_neon` — int8 → f32
//! 3. `quantize_per_channel_neon` — per-channel int8 quantization
//! 4. `quantize_dynamic_range_neon` — auto-scale from input absmax
//! 5. `i8_matmul_accumulate_neon` — int8 matmul with i32 accumulation

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Round-half-to-even (banker's rounding), matching NEON `vcvtnq_s32_f32`.
#[inline(always)]
fn round_half_to_even(x: f32) -> f32 {
    // Use the standard rounding mode: round to nearest, ties to even.
    let r = x.round();
    // Check if we're exactly at a half: if so, round to even.
    if (x - r).abs() == 0.0 {
        return r;
    }
    let frac = x - x.floor();
    if (frac - 0.5).abs() < f32::EPSILON {
        let floored = x.floor();
        if floored as i64 % 2 == 0 { floored } else { x.ceil() }
    } else {
        r
    }
}

// ── 1. quantize_f32_to_i8_neon ─────────────────────────────────────

/// Quantize f32 values to int8 using symmetric quantization: `out = clamp(round(x / scale), -128, 127)`.
pub fn quantize_f32_to_i8_neon(input: &[f32], scale: f32) -> Vec<i8> {
    if input.is_empty() || scale == 0.0 {
        return vec![0i8; input.len()];
    }
    let inv_scale = 1.0 / scale;
    let mut output = vec![0i8; input.len()];

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = input.len() / 4;
        let inv_scale_vec = unsafe { vdupq_n_f32(inv_scale) };
        for i in 0..chunks {
            let offset = i * 4;
            let v = unsafe { vld1q_f32(input.as_ptr().add(offset)) };
            let scaled = unsafe { vmulq_f32(v, inv_scale_vec) };
            // Round to nearest via vcvtnq_s32_f32
            let rounded = unsafe { vcvtnq_s32_f32(scaled) };
            // Clamp to [-128, 127]
            let clamped = unsafe {
                let lo = vdupq_n_s32(-128);
                let hi = vdupq_n_s32(127);
                vmaxq_s32(vminq_s32(rounded, hi), lo)
            };
            // Extract 4 clamped i32 values and store as i8
            let mut tmp = [0i32; 4];
            unsafe { vst1q_s32(tmp.as_mut_ptr(), clamped) };
            for j in 0..4 {
                output[offset + j] = tmp[j] as i8;
            }
        }
        // Scalar tail
        for i in (chunks * 4)..input.len() {
            let v = round_half_to_even(input[i] * inv_scale);
            output[i] = v.clamp(-128.0, 127.0) as i8;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..input.len() {
            let v = round_half_to_even(input[i] * inv_scale);
            output[i] = v.clamp(-128.0, 127.0) as i8;
        }
    }

    output
}

// ── 2. dequantize_i8_to_f32_neon ───────────────────────────────────

/// Dequantize int8 values back to f32: `out = x * scale`.
pub fn dequantize_i8_to_f32_neon(input: &[i8], scale: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut output = vec![0.0f32; input.len()];

    #[cfg(target_arch = "aarch64")]
    {
        let chunks = input.len() / 4;
        let scale_vec = unsafe { vdupq_n_f32(scale) };
        for i in 0..chunks {
            let offset = i * 4;
            // Load 4 i8 values, widen to i32, convert to f32
            let vals: [i8; 4] =
                [input[offset], input[offset + 1], input[offset + 2], input[offset + 3]];
            let i32_vals = unsafe {
                let v = vld1_s8([vals[0], vals[1], vals[2], vals[3], 0, 0, 0, 0].as_ptr());
                let wide16 = vmovl_s8(v);
                vmovl_s16(vget_low_s16(wide16))
            };
            let f32_vals = unsafe { vcvtq_f32_s32(i32_vals) };
            let result = unsafe { vmulq_f32(f32_vals, scale_vec) };
            unsafe {
                vst1q_f32(output.as_mut_ptr().add(offset), result);
            }
        }
        // Scalar tail
        for i in (chunks * 4)..input.len() {
            output[i] = input[i] as f32 * scale;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..input.len() {
            output[i] = input[i] as f32 * scale;
        }
    }

    output
}

// ── 3. quantize_per_channel_neon ───────────────────────────────────

/// Per-channel int8 quantization. `input` is laid out as `channels` contiguous
/// blocks each of length `input.len() / channels`. Each block uses its own
/// scale from `scales`.
///
/// # Panics
///
/// Panics if `channels` is 0 or `input.len()` is not divisible by `channels`.
pub fn quantize_per_channel_neon(input: &[f32], scales: &[f32], channels: usize) -> Vec<i8> {
    assert!(channels > 0, "channels must be > 0");
    if input.is_empty() {
        return Vec::new();
    }
    assert_eq!(input.len() % channels, 0, "input length must be divisible by channels");
    assert_eq!(scales.len(), channels, "scales length must equal channels");
    let channel_size = input.len() / channels;
    let mut output = vec![0i8; input.len()];

    for ch in 0..channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let ch_input = &input[start..end];
        let ch_output = quantize_f32_to_i8_neon(ch_input, scales[ch]);
        output[start..end].copy_from_slice(&ch_output);
    }

    output
}

// ── 4. quantize_dynamic_range_neon ─────────────────────────────────

/// Compute the symmetric quantization scale from the input absmax and quantize.
/// Returns `(quantized, scale)` where `scale = absmax / 127.0`.
/// If all values are zero, scale is 1.0.
pub fn quantize_dynamic_range_neon(input: &[f32]) -> (Vec<i8>, f32) {
    if input.is_empty() {
        return (Vec::new(), 1.0);
    }

    let absmax = find_absmax(input);
    let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
    let quantized = quantize_f32_to_i8_neon(input, scale);
    (quantized, scale)
}

/// Find the absolute maximum across a slice.
fn find_absmax(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let mut max_val: f32 = 0.0;
        let chunks = input.len() / 4;
        if chunks > 0 {
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for i in 0..chunks {
                let offset = i * 4;
                let v = unsafe { vld1q_f32(input.as_ptr().add(offset)) };
                let abs_v = unsafe { vabsq_f32(v) };
                acc = unsafe { vmaxq_f32(acc, abs_v) };
            }
            // Horizontal max of the 4 lanes
            max_val = unsafe { vmaxvq_f32(acc) };
        }
        // Scalar tail
        for i in (chunks * 4)..input.len() {
            let a = input[i].abs();
            if a > max_val {
                max_val = a;
            }
        }
        max_val
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        input.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
    }
}

// ── 5. i8_matmul_accumulate_neon ───────────────────────────────────

/// Int8 matrix multiply with i32 accumulation: `C[m,n] = sum_k A[m,k] * B[k,n]`.
/// `a` is row-major `[m, k]`, `b` is row-major `[k, n]`, output is row-major `[m, n]`.
///
/// # Panics
///
/// Panics if `a.len() != m * k` or `b.len() != k * n`.
pub fn i8_matmul_accumulate_neon(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k, "a length must be m * k");
    assert_eq!(b.len(), k * n, "b length must be k * n");
    if m == 0 || n == 0 || k == 0 {
        return vec![0i32; m * n];
    }
    let mut output = vec![0i32; m * n];

    #[cfg(target_arch = "aarch64")]
    {
        for row in 0..m {
            for col in 0..n {
                let mut acc = unsafe { vdupq_n_s32(0) };
                let chunks = k / 8;
                for c in 0..chunks {
                    let offset_a = row * k + c * 8;
                    let offset_b_base = c * 8;
                    // Load 8 elements from row of A
                    let va = unsafe { vld1_s8(a.as_ptr().add(offset_a)) };
                    // Gather 8 elements from column of B (stride = n)
                    let mut b_vals = [0i8; 8];
                    for j in 0..8 {
                        b_vals[j] = b[(offset_b_base + j) * n + col];
                    }
                    let vb = unsafe { vld1_s8(b_vals.as_ptr()) };
                    // Widen to i16 and multiply-accumulate
                    let a_lo = unsafe { vmovl_s8(va) };
                    let b_lo = unsafe { vmovl_s8(vb) };
                    let prod_lo = unsafe { vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo)) };
                    let prod_hi = unsafe { vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo)) };
                    acc = unsafe { vaddq_s32(acc, vaddq_s32(prod_lo, prod_hi)) };
                }
                // Horizontal sum
                let sum = unsafe { vaddvq_s32(acc) };
                let mut scalar_sum = sum;
                // Scalar tail
                for j in (chunks * 8)..k {
                    scalar_sum += a[row * k + j] as i32 * b[j * n + col] as i32;
                }
                output[row * n + col] = scalar_sum;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0i32;
                for j in 0..k {
                    sum += a[row * k + j] as i32 * b[j * n + col] as i32;
                }
                output[row * n + col] = sum;
            }
        }
    }

    output
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── quantize_f32_to_i8_neon tests ──────────────────────────────

    #[test]
    fn test_quantize_empty() {
        let result = quantize_f32_to_i8_neon(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantize_single_element() {
        // 0.5 / 1.0 = 0.5 → banker's rounding → 0 (round to even)
        let result = quantize_f32_to_i8_neon(&[0.5], 1.0);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_quantize_zeros() {
        let input = vec![0.0; 8];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![0i8; 8]);
    }

    #[test]
    fn test_quantize_positive_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_quantize_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![-1, -2, -3, -4]);
    }

    #[test]
    fn test_quantize_mixed_values() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -3.0, 0.5];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        // NEON vcvtnq uses banker's rounding: 0.5 → 0 (round-half-to-even)
        assert_eq!(result, vec![-2, -1, 0, 1, 2, 3, -3, 0]);
    }

    #[test]
    fn test_quantize_with_scale() {
        let input = vec![0.0, 0.1, 0.2, 0.3];
        let scale = 0.1;
        let result = quantize_f32_to_i8_neon(&input, scale);
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_quantize_clamping_upper() {
        let input = vec![200.0, 300.0, 400.0, 500.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![127, 127, 127, 127]);
    }

    #[test]
    fn test_quantize_clamping_lower() {
        let input = vec![-200.0, -300.0, -400.0, -500.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![-128, -128, -128, -128]);
    }

    #[test]
    fn test_quantize_at_i8_boundaries() {
        let input = vec![-128.0, 127.0, -128.0, 127.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![-128, 127, -128, 127]);
    }

    #[test]
    fn test_quantize_zero_scale_returns_zeros() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = quantize_f32_to_i8_neon(&input, 0.0);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_quantize_non_aligned_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements (not multiple of 4)
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_quantize_1_element_tail() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = quantize_f32_to_i8_neon(&input, 10.0);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_quantize_2_element_tail() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_quantize_3_element_tail() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_quantize_rounding() {
        // NEON vcvtnq uses banker's rounding (round-half-to-even):
        // 1.5 → 2 (even), 2.5 → 2 (even), -1.5 → -2 (even), -2.5 → -2 (even)
        let input = vec![1.5, 2.5, -1.5, -2.5];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![2, 2, -2, -2]);
    }

    #[test]
    fn test_quantize_large_scale() {
        let input = vec![100.0, 200.0, 300.0, 400.0];
        let result = quantize_f32_to_i8_neon(&input, 100.0);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_quantize_small_scale() {
        let input = vec![0.001, 0.002, 0.003, 0.004];
        let result = quantize_f32_to_i8_neon(&input, 0.001);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_quantize_negative_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = quantize_f32_to_i8_neon(&input, -1.0);
        assert_eq!(result, vec![-1, -2, -3, -4]);
    }

    // ── dequantize_i8_to_f32_neon tests ────────────────────────────

    #[test]
    fn test_dequantize_empty() {
        let result = dequantize_i8_to_f32_neon(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_single_element() {
        let result = dequantize_i8_to_f32_neon(&[5], 2.0);
        assert_eq!(result, vec![10.0]);
    }

    #[test]
    fn test_dequantize_zeros() {
        let input = vec![0i8; 8];
        let result = dequantize_i8_to_f32_neon(&input, 1.0);
        assert_eq!(result, vec![0.0; 8]);
    }

    #[test]
    fn test_dequantize_positive_values() {
        let input = vec![1i8, 2, 3, 4];
        let result = dequantize_i8_to_f32_neon(&input, 1.0);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dequantize_negative_values() {
        let input = vec![-1i8, -2, -3, -4];
        let result = dequantize_i8_to_f32_neon(&input, 1.0);
        assert_eq!(result, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_dequantize_with_scale() {
        let input = vec![1i8, 2, 3, 4];
        let result = dequantize_i8_to_f32_neon(&input, 0.5);
        assert_eq!(result, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_dequantize_i8_min_max() {
        let input = vec![-128i8, 127, -128, 127];
        let result = dequantize_i8_to_f32_neon(&input, 1.0);
        assert_eq!(result, vec![-128.0, 127.0, -128.0, 127.0]);
    }

    #[test]
    fn test_dequantize_non_aligned_length() {
        let input = vec![1i8, 2, 3, 4, 5];
        let result = dequantize_i8_to_f32_neon(&input, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_dequantize_zero_scale() {
        let input = vec![1i8, 2, 3, 4];
        let result = dequantize_i8_to_f32_neon(&input, 0.0);
        assert_eq!(result, vec![0.0; 4]);
    }

    // ── Round-trip tests ───────────────────────────────────────────

    #[test]
    fn test_roundtrip_identity_scale1() {
        let input = vec![0.0, 1.0, -1.0, 50.0, -50.0, 127.0, -128.0, 0.0];
        let quantized = quantize_f32_to_i8_neon(&input, 1.0);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, 1.0);
        assert_eq!(input, dequantized);
    }

    #[test]
    fn test_roundtrip_small_scale() {
        let scale = 0.1;
        let input: Vec<f32> = (0..8).map(|i| i as f32 * scale).collect();
        let quantized = quantize_f32_to_i8_neon(&input, scale);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, scale);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() < scale + 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_roundtrip_large_input() {
        let n = 1024;
        let scale = 0.05;
        // Keep values within quantizable range: [-128*scale, 127*scale] = [-6.4, 6.35]
        let input: Vec<f32> =
            (0..n).map(|i| ((i as f32 / n as f32) - 0.5) * 2.0 * 127.0 * scale).collect();
        let quantized = quantize_f32_to_i8_neon(&input, scale);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, scale);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() <= scale + 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_roundtrip_preserves_zeros() {
        let input = vec![0.0; 16];
        let quantized = quantize_f32_to_i8_neon(&input, 1.0);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, 1.0);
        assert_eq!(dequantized, input);
    }

    #[test]
    fn test_roundtrip_preserves_sign() {
        let input = vec![-10.0, 10.0, -20.0, 20.0, -5.0, 5.0, -1.0, 1.0];
        let quantized = quantize_f32_to_i8_neon(&input, 1.0);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, 1.0);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert_eq!(a.signum(), b.signum(), "sign mismatch for {a} vs {b}");
        }
    }

    // ── quantize_per_channel_neon tests ────────────────────────────

    #[test]
    fn test_per_channel_empty() {
        let result = quantize_per_channel_neon(&[], &[], 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_per_channel_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![1.0];
        let result = quantize_per_channel_neon(&input, &scales, 1);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_per_channel_two_channels() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let scales = vec![1.0, 10.0];
        let result = quantize_per_channel_neon(&input, &scales, 2);
        // Channel 0 (scale=1.0): [1,2,3,4]
        // Channel 1 (scale=10.0): [1,2,3,4]
        assert_eq!(result, vec![1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn test_per_channel_varying_scales() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 5.0, 10.0, 15.0, 20.0];
        let scales = vec![10.0, 5.0];
        let result = quantize_per_channel_neon(&input, &scales, 2);
        assert_eq!(result, vec![1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn test_per_channel_four_channels() {
        // 4 channels, 2 elements each
        let input = vec![2.0, 4.0, 3.0, 6.0, 5.0, 10.0, 7.0, 14.0];
        let scales = vec![2.0, 3.0, 5.0, 7.0];
        let result = quantize_per_channel_neon(&input, &scales, 4);
        assert_eq!(result, vec![1, 2, 1, 2, 1, 2, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn test_per_channel_zero_channels_panics() {
        quantize_per_channel_neon(&[1.0], &[], 0);
    }

    #[test]
    #[should_panic(expected = "input length must be divisible by channels")]
    fn test_per_channel_misaligned_panics() {
        quantize_per_channel_neon(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2);
    }

    #[test]
    #[should_panic(expected = "scales length must equal channels")]
    fn test_per_channel_wrong_scales_panics() {
        quantize_per_channel_neon(&[1.0, 2.0, 3.0, 4.0], &[1.0], 2);
    }

    #[test]
    fn test_per_channel_with_clamping() {
        let input = vec![200.0, -200.0, 0.5, -0.5];
        let scales = vec![1.0, 0.5];
        let result = quantize_per_channel_neon(&input, &scales, 2);
        assert_eq!(result[0], 127);
        assert_eq!(result[1], -128);
        assert_eq!(result[2], 1);
        assert_eq!(result[3], -1);
    }

    // ── quantize_dynamic_range_neon tests ──────────────────────────

    #[test]
    fn test_dynamic_range_empty() {
        let (quantized, scale) = quantize_dynamic_range_neon(&[]);
        assert!(quantized.is_empty());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_dynamic_range_all_zeros() {
        let input = vec![0.0; 8];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        assert_eq!(scale, 1.0);
        assert_eq!(quantized, vec![0i8; 8]);
    }

    #[test]
    fn test_dynamic_range_positive_only() {
        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let expected_scale = 7.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6, "scale: {scale} expected: {expected_scale}");
        assert_eq!(quantized[7], 127); // 7.0 / scale → 127
        assert_eq!(quantized[0], 0); // 0.0 → 0
    }

    #[test]
    fn test_dynamic_range_negative_only() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let expected_scale = 4.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        assert_eq!(quantized[3], -127);
    }

    #[test]
    fn test_dynamic_range_symmetric() {
        let input = vec![-10.0, -5.0, 0.0, 5.0, 10.0, 0.0, 0.0, 0.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let expected_scale = 10.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        assert_eq!(quantized[0], -127);
        assert_eq!(quantized[4], 127);
    }

    #[test]
    fn test_dynamic_range_single_element() {
        let (quantized, scale) = quantize_dynamic_range_neon(&[5.0]);
        let expected_scale = 5.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        assert_eq!(quantized[0], 127);
    }

    #[test]
    fn test_dynamic_range_large_values() {
        let input = vec![1000.0, -1000.0, 500.0, -500.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let expected_scale = 1000.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -127);
    }

    #[test]
    fn test_dynamic_range_dequantize_roundtrip() {
        let input = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let dequantized = dequantize_i8_to_f32_neon(&quantized, scale);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() <= scale + 1e-5, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_dynamic_range_non_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        let expected_scale = 5.0 / 127.0;
        assert!((scale - expected_scale).abs() < 1e-6);
        assert_eq!(quantized.len(), 5);
        assert_eq!(quantized[4], 127);
    }

    // ── i8_matmul_accumulate_neon tests ────────────────────────────

    #[test]
    fn test_matmul_empty() {
        let result = i8_matmul_accumulate_neon(&[], &[], 0, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_1x1() {
        let a = vec![3i8];
        let b = vec![4i8];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 1);
        assert_eq!(result, vec![12]);
    }

    #[test]
    fn test_matmul_2x2() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let a = vec![1i8, 2, 3, 4];
        let b = vec![5i8, 6, 7, 8];
        let result = i8_matmul_accumulate_neon(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
        // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //   = [[58, 64], [139, 154]]
        let a = vec![1i8, 2, 3, 4, 5, 6];
        let b = vec![7i8, 8, 9, 10, 11, 12];
        let result = i8_matmul_accumulate_neon(&a, &b, 2, 2, 3);
        assert_eq!(result, vec![58, 64, 139, 154]);
    }

    #[test]
    fn test_matmul_identity() {
        // A * I = A (for 2x2)
        let a = vec![3i8, 7, -2, 5];
        let identity = vec![1i8, 0, 0, 1];
        let result = i8_matmul_accumulate_neon(&a, &identity, 2, 2, 2);
        assert_eq!(result, vec![3, 7, -2, 5]);
    }

    #[test]
    fn test_matmul_zeros() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![0i8; 4];
        let result = i8_matmul_accumulate_neon(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_matmul_negative_values() {
        let a = vec![-1i8, -2, -3, -4];
        let b = vec![1i8, 0, 0, 1];
        let result = i8_matmul_accumulate_neon(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![-1, -2, -3, -4]);
    }

    #[test]
    fn test_matmul_row_vector_times_col_vector() {
        // [1, 2, 3] * [[4], [5], [6]] = [32]
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 3);
        assert_eq!(result, vec![32]);
    }

    #[test]
    fn test_matmul_col_vector_times_row_vector() {
        // [[1], [2], [3]] * [4, 5, 6] = [[4,5,6],[8,10,12],[12,15,18]]
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let result = i8_matmul_accumulate_neon(&a, &b, 3, 3, 1);
        assert_eq!(result, vec![4, 5, 6, 8, 10, 12, 12, 15, 18]);
    }

    #[test]
    fn test_matmul_k_larger_than_8() {
        // k=10, m=1, n=1 → dot product of two length-10 vectors
        let a: Vec<i8> = (1..=10).map(|x| x as i8).collect();
        let b: Vec<i8> = (1..=10).map(|x| x as i8).collect();
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 10);
        // 1*1 + 2*2 + ... + 10*10 = 385
        assert_eq!(result, vec![385]);
    }

    #[test]
    fn test_matmul_k_exactly_8() {
        let a: Vec<i8> = (1..=8).map(|x| x as i8).collect();
        let b: Vec<i8> = vec![1i8; 8];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 8);
        // 1+2+...+8 = 36
        assert_eq!(result, vec![36]);
    }

    #[test]
    fn test_matmul_k_16() {
        let a: Vec<i8> = (1..=16).map(|x| x as i8).collect();
        let b: Vec<i8> = vec![1i8; 16];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 16);
        // 1+2+...+16 = 136
        assert_eq!(result, vec![136]);
    }

    #[test]
    fn test_matmul_i8_extremes() {
        let a = vec![127i8, -128];
        let b = vec![1i8, 1];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 2);
        // 127*1 + (-128)*1 = -1
        assert_eq!(result, vec![-1]);
    }

    #[test]
    fn test_matmul_vs_f32_reference() {
        let a = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = vec![9i8, 8, 7, 6, 5, 4, 3, 2, 1];
        let result = i8_matmul_accumulate_neon(&a, &b, 3, 3, 3);
        // Row 0: [1,2,3] . [col0=[9,6,3], col1=[8,5,2], col2=[7,4,1]]
        // C[0,0] = 1*9 + 2*6 + 3*3 = 30
        // C[0,1] = 1*8 + 2*5 + 3*2 = 24
        // C[0,2] = 1*7 + 2*4 + 3*1 = 18
        // C[1,0] = 4*9 + 5*6 + 6*3 = 84
        // C[1,1] = 4*8 + 5*5 + 6*2 = 69
        // C[1,2] = 4*7 + 5*4 + 6*1 = 54
        // C[2,0] = 7*9 + 8*6 + 9*3 = 138
        // C[2,1] = 7*8 + 8*5 + 9*2 = 114
        // C[2,2] = 7*7 + 8*4 + 9*1 = 90
        assert_eq!(result, vec![30, 24, 18, 84, 69, 54, 138, 114, 90]);
    }

    #[test]
    #[should_panic(expected = "a length must be m * k")]
    fn test_matmul_bad_a_length_panics() {
        i8_matmul_accumulate_neon(&[1i8, 2], &[1i8, 2, 3, 4], 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "b length must be k * n")]
    fn test_matmul_bad_b_length_panics() {
        i8_matmul_accumulate_neon(&[1i8, 2, 3, 4], &[1i8, 2], 2, 2, 2);
    }

    // ── Large input tests ──────────────────────────────────────────

    #[test]
    fn test_quantize_large_1000() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.1).collect();
        let result = quantize_f32_to_i8_neon(&input, 0.5);
        assert_eq!(result.len(), 1000);
        // Check a few samples
        // input[500] = 0.0 → 0
        assert_eq!(result[500], 0);
        // input[0] = -50.0, -50/0.5 = -100 → -100
        assert_eq!(result[0], -100);
    }

    #[test]
    fn test_dequantize_large_1000() {
        let input: Vec<i8> = (0..1000).map(|i| (i % 256) as i8).collect();
        let result = dequantize_i8_to_f32_neon(&input, 0.1);
        assert_eq!(result.len(), 1000);
        for (i, &v) in input.iter().enumerate() {
            assert!((result[i] - v as f32 * 0.1).abs() < 1e-5, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_dynamic_range_large_1024() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        assert_eq!(quantized.len(), 1024);
        assert!(scale > 0.0);
        // Dequantize and check roundtrip
        let dequantized = dequantize_i8_to_f32_neon(&quantized, scale);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert!((a - b).abs() <= scale + 1e-5, "roundtrip error too large: {a} vs {b}");
        }
    }

    #[test]
    fn test_matmul_large_k() {
        // 1xK dot K x1 with K=64
        let k = 64;
        let a: Vec<i8> = (0..k).map(|i| (i % 5) as i8).collect();
        let b: Vec<i8> = vec![1i8; k];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, k);
        let expected: i32 = a.iter().map(|&x| x as i32).sum();
        assert_eq!(result, vec![expected]);
    }

    #[test]
    fn test_quantize_large_2048() {
        let input: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.01 - 10.0).collect();
        let result = quantize_f32_to_i8_neon(&input, 0.1);
        assert_eq!(result.len(), 2048);
    }

    // ── Determinism tests ──────────────────────────────────────────

    #[test]
    fn test_quantize_deterministic() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let r1 = quantize_f32_to_i8_neon(&input, 0.1);
        let r2 = quantize_f32_to_i8_neon(&input, 0.1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_dequantize_deterministic() {
        let input: Vec<i8> = (0..256).map(|i| (i % 256) as i8).collect();
        let r1 = dequantize_i8_to_f32_neon(&input, 0.5);
        let r2 = dequantize_i8_to_f32_neon(&input, 0.5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_dynamic_range_deterministic() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.05).collect();
        let (q1, s1) = quantize_dynamic_range_neon(&input);
        let (q2, s2) = quantize_dynamic_range_neon(&input);
        assert_eq!(s1, s2);
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_matmul_deterministic() {
        let a: Vec<i8> = (0..16).map(|i| (i * 3 % 7) as i8).collect();
        let b: Vec<i8> = (0..16).map(|i| (i * 5 % 11) as i8).collect();
        let r1 = i8_matmul_accumulate_neon(&a, &b, 4, 4, 4);
        let r2 = i8_matmul_accumulate_neon(&a, &b, 4, 4, 4);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_per_channel_deterministic() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let scales = vec![0.1, 0.2, 0.5, 1.0];
        let r1 = quantize_per_channel_neon(&input, &scales, 4);
        let r2 = quantize_per_channel_neon(&input, &scales, 4);
        assert_eq!(r1, r2);
    }

    // ── find_absmax tests ──────────────────────────────────────────

    #[test]
    fn test_find_absmax_positive() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(find_absmax(&input), 4.0);
    }

    #[test]
    fn test_find_absmax_negative() {
        let input = vec![-5.0, -2.0, -3.0, -1.0];
        assert_eq!(find_absmax(&input), 5.0);
    }

    #[test]
    fn test_find_absmax_mixed() {
        let input = vec![-10.0, 5.0, 3.0, -7.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(find_absmax(&input), 10.0);
    }

    #[test]
    fn test_find_absmax_single() {
        assert_eq!(find_absmax(&[42.0]), 42.0);
        assert_eq!(find_absmax(&[-42.0]), 42.0);
    }

    #[test]
    fn test_find_absmax_all_zeros() {
        assert_eq!(find_absmax(&[0.0; 8]), 0.0);
    }

    #[test]
    fn test_find_absmax_tail_has_max() {
        // Ensure scalar tail is checked (5 elements, max in last)
        let input = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(find_absmax(&input), 100.0);
    }

    // ── Additional edge-case tests ─────────────────────────────────

    #[test]
    fn test_quantize_exactly_4_elements() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let result = quantize_f32_to_i8_neon(&input, 10.0);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_quantize_exactly_8_elements() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        let expected: Vec<i8> = (1..=8).map(|i| i as i8).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dequantize_exactly_4_elements() {
        let input = vec![1i8, 2, 3, 4];
        let result = dequantize_i8_to_f32_neon(&input, 3.0);
        assert_eq!(result, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_dequantize_exactly_8_elements() {
        let input: Vec<i8> = (1..=8).map(|i| i as i8).collect();
        let result = dequantize_i8_to_f32_neon(&input, 2.0);
        let expected: Vec<f32> = (1..=8).map(|i| i as f32 * 2.0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_4x4() {
        // 4x4 identity
        let a: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let b: Vec<i8> = vec![5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16];
        let result = i8_matmul_accumulate_neon(&a, &b, 4, 4, 4);
        assert_eq!(result, vec![5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_per_channel_many_channels() {
        // 8 channels, 1 element each
        let input: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let scales: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let result = quantize_per_channel_neon(&input, &scales, 8);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_matmul_m_zero() {
        let result = i8_matmul_accumulate_neon(&[], &[], 0, 4, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_n_zero() {
        let result = i8_matmul_accumulate_neon(&[], &[], 4, 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_matmul_k_zero() {
        let result = i8_matmul_accumulate_neon(&[], &[], 4, 4, 0);
        assert_eq!(result, vec![0i32; 16]);
    }

    #[test]
    fn test_quantize_fractional_rounding() {
        // 0.4 / 1.0 → 0, 0.6 / 1.0 → 1
        let input = vec![0.4, 0.6, -0.4, -0.6];
        let result = quantize_f32_to_i8_neon(&input, 1.0);
        assert_eq!(result, vec![0, 1, 0, -1]);
    }

    #[test]
    fn test_dynamic_range_tiny_values() {
        let input = vec![1e-6, -1e-6, 2e-6, -2e-6];
        let (quantized, scale) = quantize_dynamic_range_neon(&input);
        assert!(scale > 0.0);
        assert_eq!(quantized.len(), 4);
    }

    #[test]
    fn test_matmul_accumulation_no_overflow_i32() {
        // 127 * 127 * 16 = 258064, well within i32
        let a = vec![127i8; 16];
        let b = vec![127i8; 16];
        let result = i8_matmul_accumulate_neon(&a, &b, 1, 1, 16);
        assert_eq!(result, vec![127 * 127 * 16]);
    }

    #[test]
    fn test_per_channel_large_channel_size() {
        let ch_size = 128;
        let channels = 2;
        let input: Vec<f32> = (0..(ch_size * channels)).map(|i| (i % ch_size) as f32).collect();
        let scales = vec![1.0, 2.0];
        let result = quantize_per_channel_neon(&input, &scales, channels);
        assert_eq!(result.len(), ch_size * channels);
        // Channel 0 at index 0: 0/1 = 0
        assert_eq!(result[0], 0);
        // Channel 1 at index ch_size: 0/2 = 0
        assert_eq!(result[ch_size], 0);
        // Channel 0 at index 10: 10/1 = 10
        assert_eq!(result[10], 10);
        // Channel 1 at index ch_size+10: 10/2 = 5
        assert_eq!(result[ch_size + 10], 5);
    }
}
