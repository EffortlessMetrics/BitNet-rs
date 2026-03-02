//! ARM NEON padding and clipping kernels for Apple Silicon.
//!
//! Provides zero-padding, reflection-padding, element-wise clipping, and
//! gradient norm clipping using `float32x4` NEON intrinsics with scalar
//! fallback for remainder elements.

use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Padding ─────────────────────────────────────────────────────────────

/// Zero-pad `input` into `output` with `pad_before` zeros prepended and
/// `pad_after` zeros appended.
///
/// # Panics
///
/// Panics if `output.len() != pad_before + input_len + pad_after` or
/// `input_len > input.len()`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_zero_pad(
    input: &[f32],
    output: &mut [f32],
    input_len: usize,
    pad_before: usize,
    pad_after: usize,
) {
    assert!(input_len <= input.len(), "input_len exceeds input slice");
    let total = pad_before + input_len + pad_after;
    assert_eq!(output.len(), total, "output length must equal pad_before + input_len + pad_after");

    let zero = vdupq_n_f32(0.0);

    // Zero-fill the leading pad region.
    unsafe { fill_zero_neon(output, 0, pad_before, zero) };

    // Copy input data.
    unsafe { copy_neon(input, output, input_len, pad_before) };

    // Zero-fill the trailing pad region.
    unsafe { fill_zero_neon(output, pad_before + input_len, pad_after, zero) };
}

/// NEON-accelerated zero-fill helper.
///
/// # Safety
///
/// `zero` must be a valid NEON zero vector; `start + len <= output.len()`.
#[target_feature(enable = "neon")]
unsafe fn fill_zero_neon(output: &mut [f32], start: usize, len: usize, zero: float32x4_t) {
    let chunks = len / LANES;
    let o_ptr = unsafe { output.as_mut_ptr().add(start) };
    for i in 0..chunks {
        unsafe {
            vst1q_f32(o_ptr.add(i * LANES), zero);
        }
    }
    for i in (chunks * LANES)..len {
        unsafe {
            *o_ptr.add(i) = 0.0;
        }
    }
}

/// NEON-accelerated copy helper.
///
/// # Safety
///
/// Requires valid pointers; `input_len` elements are read from `input` and
/// written starting at `output[dst_offset..]`.
#[target_feature(enable = "neon")]
unsafe fn copy_neon(input: &[f32], output: &mut [f32], input_len: usize, dst_offset: usize) {
    let chunks = input_len / LANES;
    let i_ptr = input.as_ptr();
    let o_ptr = unsafe { output.as_mut_ptr().add(dst_offset) };
    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(i_ptr.add(offset));
            vst1q_f32(o_ptr.add(offset), v);
        }
    }
    for i in (chunks * LANES)..input_len {
        unsafe {
            *o_ptr.add(i) = *i_ptr.add(i);
        }
    }
}

/// Reflection-pad `input` into `output`.
///
/// Pads by mirroring elements at the boundaries (excluding the boundary
/// element itself): for `pad_before = 3` on `[a, b, c, d, …]` the prefix
/// is `[d, c, b]`.
///
/// # Panics
///
/// Panics if `pad_before >= input_len`, `pad_after >= input_len`,
/// `input_len > input.len()`, or output length is wrong.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_reflect_pad(
    input: &[f32],
    output: &mut [f32],
    input_len: usize,
    pad_before: usize,
    pad_after: usize,
) {
    assert!(input_len <= input.len(), "input_len exceeds input slice");
    assert!(pad_before < input_len, "pad_before must be < input_len for reflection");
    assert!(pad_after < input_len, "pad_after must be < input_len for reflection");
    let total = pad_before + input_len + pad_after;
    assert_eq!(output.len(), total);

    // Leading reflection: mirror from index pad_before down to 1.
    for i in 0..pad_before {
        output[i] = input[pad_before - i];
    }

    // Copy input data with NEON.
    unsafe { copy_neon(input, output, input_len, pad_before) };

    // Trailing reflection: mirror from input_len-2 downward.
    for i in 0..pad_after {
        output[pad_before + input_len + i] = input[input_len - 2 - i];
    }
}

// ── Clipping ────────────────────────────────────────────────────────────

/// Element-wise clipping: `output[i] = clamp(input[i], min_val, max_val)`.
///
/// Uses NEON `vminq_f32` / `vmaxq_f32` for 4-wide SIMD with scalar
/// remainder.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_clip(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    assert_eq!(input.len(), output.len());
    let n = input.len();
    let chunks = n / LANES;
    let i_ptr = input.as_ptr();
    let o_ptr = output.as_mut_ptr();

    let vmin = vdupq_n_f32(min_val);
    let vmax = vdupq_n_f32(max_val);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(i_ptr.add(offset));
            let clamped = vminq_f32(vmaxq_f32(v, vmin), vmax);
            vst1q_f32(o_ptr.add(offset), clamped);
        }
    }

    for i in (chunks * LANES)..n {
        output[i] = input[i].clamp(min_val, max_val);
    }
}

/// Gradient norm clipping (in-place).
///
/// Computes the L2 norm of `gradients` and, if it exceeds `max_norm`,
/// scales every element by `max_norm / norm`.  Uses NEON for both the
/// norm computation and the scaling pass.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[target_feature(enable = "neon")]
pub unsafe fn neon_gradient_clip(gradients: &mut [f32], max_norm: f32) {
    let n = gradients.len();
    if n == 0 {
        return;
    }

    // ── L2 norm accumulation ────────────────────────────────────────
    let chunks = n / LANES;
    let g_ptr = gradients.as_ptr();
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(g_ptr.add(offset));
            acc = vfmaq_f32(acc, v, v);
        }
    }

    // Horizontal sum of the four accumulator lanes.
    let mut sum_sq: f32 = {
        let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        vget_lane_f32::<0>(vpadd_f32(pair, pair))
    };

    // Scalar tail.
    for &g in &gradients[chunks * LANES..n] {
        sum_sq += g * g;
    }

    let norm = sum_sq.sqrt();
    if norm <= max_norm {
        return;
    }

    // ── Scale gradients ─────────────────────────────────────────────
    let scale = max_norm / norm;
    let vs = vdupq_n_f32(scale);
    let g_mut = gradients.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(g_mut.add(offset));
            vst1q_f32(g_mut.add(offset), vmulq_f32(v, vs));
        }
    }

    for g in &mut gradients[chunks * LANES..n] {
        *g *= scale;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── zero pad ────────────────────────────────────────────────────

    #[test]
    fn test_zero_pad_basic() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 8];
        unsafe { neon_zero_pad(&input, &mut out, 4, 2, 2) };
        assert_eq!(out, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_pad_no_padding() {
        let input = [5.0f32, 6.0, 7.0];
        let mut out = vec![0.0f32; 3];
        unsafe { neon_zero_pad(&input, &mut out, 3, 0, 0) };
        assert_eq!(out, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_zero_pad_only_before() {
        let input = [1.0f32, 2.0];
        let mut out = vec![0.0f32; 6];
        unsafe { neon_zero_pad(&input, &mut out, 2, 4, 0) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_zero_pad_empty_input() {
        let input: &[f32] = &[];
        let mut out = vec![0.0f32; 4];
        unsafe { neon_zero_pad(input, &mut out, 0, 2, 2) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_pad_large() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let pad = 17;
        let mut out = vec![-1.0f32; pad + n + pad];
        unsafe { neon_zero_pad(&input, &mut out, n, pad, pad) };
        for i in 0..pad {
            assert_eq!(out[i], 0.0, "leading pad mismatch at {i}");
        }
        for i in 0..n {
            assert_eq!(out[pad + i], i as f32, "data mismatch at {i}");
        }
        for i in 0..pad {
            assert_eq!(out[pad + n + i], 0.0, "trailing pad mismatch at {i}");
        }
    }

    // ── reflect pad ─────────────────────────────────────────────────

    #[test]
    fn test_reflect_pad_basic() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 9];
        // pad_before=2 → mirror indices 2,1 → [3,2]
        // pad_after=2  → mirror indices 3,2 → [4,3]
        unsafe { neon_reflect_pad(&input, &mut out, 5, 2, 2) };
        assert_eq!(out, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_reflect_pad_single_each_side() {
        let input = [10.0f32, 20.0, 30.0];
        let mut out = vec![0.0f32; 5];
        unsafe { neon_reflect_pad(&input, &mut out, 3, 1, 1) };
        assert_eq!(out, [20.0, 10.0, 20.0, 30.0, 20.0]);
    }

    #[test]
    fn test_reflect_pad_no_padding() {
        let input = [1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        unsafe { neon_reflect_pad(&input, &mut out, 3, 0, 0) };
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    // ── clip ────────────────────────────────────────────────────────

    #[test]
    fn test_clip_basic() {
        let input = [-2.0f32, -0.5, 0.0, 0.5, 1.5, 3.0];
        let mut out = vec![0.0f32; 6];
        unsafe { neon_clip(&input, &mut out, -1.0, 1.0) };
        assert_eq!(out, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_clip_all_within_range() {
        let input = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let mut out = vec![0.0f32; 5];
        unsafe { neon_clip(&input, &mut out, 0.0, 1.0) };
        assert_eq!(out, input);
    }

    #[test]
    fn test_clip_empty() {
        let input: &[f32] = &[];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_clip(input, &mut out, -1.0, 1.0) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_clip_large() {
        let n = 1025;
        let input: Vec<f32> = (0..n).map(|i| i as f32 - 512.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe { neon_clip(&input, &mut out, -100.0, 100.0) };
        for (i, &v) in out.iter().enumerate() {
            let expected = (i as f32 - 512.0).clamp(-100.0, 100.0);
            assert_eq!(v, expected, "mismatch at {i}");
        }
    }

    // ── gradient clip ───────────────────────────────────────────────

    #[test]
    fn test_gradient_clip_within_norm() {
        let mut grads = [1.0f32, 0.0, 0.0, 0.0];
        unsafe { neon_gradient_clip(&mut grads, 5.0) };
        assert_eq!(grads, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gradient_clip_exceeds_norm() {
        let mut grads = [3.0f32, 4.0];
        // norm = 5, max_norm = 2.5 → scale = 0.5
        unsafe { neon_gradient_clip(&mut grads, 2.5) };
        let eps = 1e-6;
        assert!((grads[0] - 1.5).abs() < eps, "got {}", grads[0]);
        assert!((grads[1] - 2.0).abs() < eps, "got {}", grads[1]);
    }

    #[test]
    fn test_gradient_clip_empty() {
        let mut grads: Vec<f32> = vec![];
        unsafe { neon_gradient_clip(&mut grads, 1.0) };
        assert!(grads.is_empty());
    }

    #[test]
    fn test_gradient_clip_large() {
        let n = 1024;
        let mut grads: Vec<f32> = vec![1.0; n];
        // norm = sqrt(1024) = 32.0; max_norm = 16.0 → scale = 0.5
        unsafe { neon_gradient_clip(&mut grads, 16.0) };
        let expected = 0.5f32;
        for (i, &v) in grads.iter().enumerate() {
            assert!((v - expected).abs() < 1e-5, "mismatch at {i}: {v} vs {expected}");
        }
    }

    // ── proptest ────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_zero_pad_length(
                data in proptest::collection::vec(-100.0f32..100.0, 0..128),
                pad_before in 0_usize..32,
                pad_after in 0_usize..32,
            ) {
                let n = data.len();
                let total = pad_before + n + pad_after;
                let mut out = vec![-1.0f32; total];
                unsafe { neon_zero_pad(&data, &mut out, n, pad_before, pad_after) };
                // Verify leading zeros.
                for i in 0..pad_before {
                    prop_assert_eq!(out[i], 0.0);
                }
                // Verify data preserved.
                for i in 0..n {
                    prop_assert_eq!(out[pad_before + i], data[i]);
                }
                // Verify trailing zeros.
                for i in 0..pad_after {
                    prop_assert_eq!(out[pad_before + n + i], 0.0);
                }
            }

            #[test]
            fn prop_clip_bounds(
                data in proptest::collection::vec(-1e6f32..1e6, 1..256),
            ) {
                let lo = -100.0f32;
                let hi = 100.0f32;
                let mut out = vec![0.0f32; data.len()];
                unsafe { neon_clip(&data, &mut out, lo, hi) };
                for (i, &v) in out.iter().enumerate() {
                    prop_assert!(v >= lo, "out[{i}] = {v} < {lo}");
                    prop_assert!(v <= hi, "out[{i}] = {v} > {hi}");
                    prop_assert_eq!(v, data[i].clamp(lo, hi));
                }
            }

            #[test]
            fn prop_gradient_clip_norm(
                data in proptest::collection::vec(-10.0f32..10.0, 1..256),
                max_norm in 0.01f32..100.0,
            ) {
                let mut grads = data.clone();
                unsafe { neon_gradient_clip(&mut grads, max_norm) };
                let norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
                // After clipping, norm should be <= max_norm (with FP tolerance).
                prop_assert!(
                    norm <= max_norm + 1e-3,
                    "norm {norm} exceeded max_norm {max_norm}"
                );
            }
        }
    }
}
