//! NEON-accelerated tensor manipulation operations for Apple Silicon inference.
//!
//! Provides element-wise arithmetic, reductions, and utility operations using
//! ARM NEON SIMD intrinsics with scalar fallback for tail elements.

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    unused_variables,
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_memcpy,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::let_and_return,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::missing_safety_doc,
    clippy::never_loop,
    clippy::while_immutable_condition,
    clippy::manual_abs_diff
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Element-wise binary operations
// ---------------------------------------------------------------------------

/// Element-wise addition: `out[i] = a[i] + b[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let vr = vaddq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, out.len());
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let vr = vmulq_f32(va, vb);
            vst1q_f32(o_ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i];
    }
}

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_fma(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, c.len());
    assert_eq!(n, out.len());
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            let vc = vld1q_f32(c_ptr.add(off));
            let vr = vfmaq_f32(vc, va, vb);
            vst1q_f32(o_ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i] + c[i];
    }
}

// ---------------------------------------------------------------------------
// In-place scalar operations
// ---------------------------------------------------------------------------

/// In-place scalar multiplication: `data[i] *= scale`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_scale(data: &mut [f32], scale: f32) {
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_mut_ptr();

    unsafe {
        let vs = vdupq_n_f32(scale);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let vr = vmulq_f32(v, vs);
            vst1q_f32(ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        data[i] *= scale;
    }
}

/// In-place scalar addition: `data[i] += val`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_add_scalar(data: &mut [f32], val: f32) {
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_mut_ptr();

    unsafe {
        let vv = vdupq_n_f32(val);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let vr = vaddq_f32(v, vv);
            vst1q_f32(ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        data[i] += val;
    }
}

/// In-place clamp: `data[i] = data[i].clamp(min_val, max_val)`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_clamp(data: &mut [f32], min_val: f32, max_val: f32) {
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_mut_ptr();

    unsafe {
        let vmin = vdupq_n_f32(min_val);
        let vmax = vdupq_n_f32(max_val);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let v = vmaxq_f32(v, vmin);
            let v = vminq_f32(v, vmax);
            vst1q_f32(ptr.add(off), v);
        }
    }
    for i in (chunks * 4)..n {
        data[i] = data[i].clamp(min_val, max_val);
    }
}

// ---------------------------------------------------------------------------
// Unary operations
// ---------------------------------------------------------------------------

/// Absolute value: `out[i] = |data[i]|`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_abs(data: &[f32], out: &mut [f32]) {
    let n = data.len();
    assert_eq!(n, out.len());
    let chunks = n / 4;
    let s_ptr = data.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let v = vld1q_f32(s_ptr.add(off));
            let vr = vabsq_f32(v);
            vst1q_f32(o_ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = data[i].abs();
    }
}

/// Negation: `out[i] = -data[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_neg(data: &[f32], out: &mut [f32]) {
    let n = data.len();
    assert_eq!(n, out.len());
    let chunks = n / 4;
    let s_ptr = data.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let v = vld1q_f32(s_ptr.add(off));
            let vr = vnegq_f32(v);
            vst1q_f32(o_ptr.add(off), vr);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = -data[i];
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Find the maximum value in `data`. Returns `f32::NEG_INFINITY` for empty slices.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_max_reduce(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_ptr();

    let mut max_val = f32::NEG_INFINITY;

    if chunks > 0 {
        unsafe {
            let mut vmax = vld1q_f32(ptr);
            for i in 1..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                vmax = vmaxq_f32(vmax, v);
            }
            max_val = vmaxvq_f32(vmax);
        }
    }
    for i in (chunks * 4)..n {
        if data[i] > max_val {
            max_val = data[i];
        }
    }
    max_val
}

/// Sum all elements. Returns `0.0` for empty slices.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_sum_reduce(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_ptr();

    let mut sum: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            let mut vacc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                vacc = vaddq_f32(vacc, v);
            }
            sum = vaddvq_f32(vacc);
        }
    }
    for i in (chunks * 4)..n {
        sum += data[i];
    }
    sum
}

/// Compute the arithmetic mean. Returns `0.0` for empty slices.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    neon_tensor_sum_reduce(data) / data.len() as f32
}

/// Compute the population variance. Returns `0.0` for empty slices.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = neon_tensor_mean(data);
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_ptr();

    let mut var_sum: f32 = 0.0;

    if chunks > 0 {
        unsafe {
            let vmean = vdupq_n_f32(mean);
            let mut vacc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                let diff = vsubq_f32(v, vmean);
                vacc = vfmaq_f32(vacc, diff, diff);
            }
            var_sum = vaddvq_f32(vacc);
        }
    }
    for i in (chunks * 4)..n {
        let diff = data[i] - mean;
        var_sum += diff * diff;
    }
    var_sum / n as f32
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Optimized copy: `dst[i] = src[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_copy(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    assert_eq!(n, dst.len());
    let chunks = n / 4;
    let s_ptr = src.as_ptr();
    let d_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let v = vld1q_f32(s_ptr.add(off));
            vst1q_f32(d_ptr.add(off), v);
        }
    }
    for i in (chunks * 4)..n {
        dst[i] = src[i];
    }
}

/// Fill slice with a constant value.
#[cfg(target_arch = "aarch64")]
pub fn neon_tensor_fill(data: &mut [f32], val: f32) {
    let n = data.len();
    let chunks = n / 4;
    let ptr = data.as_mut_ptr();

    unsafe {
        let vv = vdupq_n_f32(val);
        for i in 0..chunks {
            vst1q_f32(ptr.add(i * 4), vv);
        }
    }
    for i in (chunks * 4)..n {
        data[i] = val;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ---- neon_tensor_add --------------------------------------------------

    #[test]
    fn test_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        neon_tensor_add(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_add_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        neon_tensor_add(&a, &b, &mut out);
    }

    #[test]
    fn test_add_single() {
        let a = vec![3.0];
        let b = vec![7.0];
        let mut out = vec![0.0];
        neon_tensor_add(&a, &b, &mut out);
        assert!(approx_eq(out[0], 10.0));
    }

    #[test]
    fn test_add_lane_boundary() {
        let a = vec![1.0; 4];
        let b = vec![2.0; 4];
        let mut out = vec![0.0; 4];
        neon_tensor_add(&a, &b, &mut out);
        assert!(out.iter().all(|&v| approx_eq(v, 3.0)));
    }

    #[test]
    fn test_add_non_aligned_7() {
        let a: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let b: Vec<f32> = (11..=17).map(|x| x as f32).collect();
        let mut out = vec![0.0; 7];
        neon_tensor_add(&a, &b, &mut out);
        for i in 0..7 {
            assert!(approx_eq(out[i], a[i] + b[i]));
        }
    }

    // ---- neon_tensor_mul --------------------------------------------------

    #[test]
    fn test_mul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 5];
        neon_tensor_mul(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_mul_empty() {
        neon_tensor_mul(&[], &[], &mut []);
    }

    #[test]
    fn test_mul_single() {
        let mut out = vec![0.0];
        neon_tensor_mul(&[3.0], &[4.0], &mut out);
        assert!(approx_eq(out[0], 12.0));
    }

    #[test]
    fn test_mul_non_aligned_13() {
        let a: Vec<f32> = (0..13).map(|x| x as f32).collect();
        let b = vec![2.0; 13];
        let mut out = vec![0.0; 13];
        neon_tensor_mul(&a, &b, &mut out);
        for i in 0..13 {
            assert!(approx_eq(out[i], a[i] * 2.0));
        }
    }

    // ---- neon_tensor_scale ------------------------------------------------

    #[test]
    fn test_scale_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        neon_tensor_scale(&mut data, 3.0);
        assert_eq!(data, vec![3.0, 6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_scale_empty() {
        neon_tensor_scale(&mut [], 5.0);
    }

    #[test]
    fn test_scale_zero() {
        let mut data = vec![1.0, 2.0, 3.0];
        neon_tensor_scale(&mut data, 0.0);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    // ---- neon_tensor_add_scalar -------------------------------------------

    #[test]
    fn test_add_scalar_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        neon_tensor_add_scalar(&mut data, 10.0);
        let expected: Vec<f32> = (11..=17).map(|x| x as f32).collect();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_add_scalar_empty() {
        neon_tensor_add_scalar(&mut [], 42.0);
    }

    #[test]
    fn test_add_scalar_negative() {
        let mut data = vec![5.0; 4];
        neon_tensor_add_scalar(&mut data, -3.0);
        assert!(data.iter().all(|&v| approx_eq(v, 2.0)));
    }

    // ---- neon_tensor_fma --------------------------------------------------

    #[test]
    fn test_fma_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        neon_tensor_fma(&a, &b, &c, &mut out);
        // out = a*b + c
        let expected = vec![12.0, 26.0, 42.0, 60.0, 80.0];
        for i in 0..5 {
            assert!(approx_eq(out[i], expected[i]));
        }
    }

    #[test]
    fn test_fma_empty() {
        neon_tensor_fma(&[], &[], &[], &mut []);
    }

    #[test]
    fn test_fma_accuracy() {
        // Verify FMA is more accurate than separate mul + add for specific cases.
        let a = vec![1.0000001; 4];
        let b = vec![1.0000001; 4];
        let c = vec![-1.0; 4];
        let mut out = vec![0.0; 4];
        neon_tensor_fma(&a, &b, &c, &mut out);
        for &v in &out {
            assert!(v >= 0.0, "FMA result should be non-negative");
            assert!(v < 1e-5, "FMA result should be very small");
        }
    }

    #[test]
    fn test_fma_non_aligned_7() {
        let a: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let b = vec![2.0; 7];
        let c = vec![0.5; 7];
        let mut out = vec![0.0; 7];
        neon_tensor_fma(&a, &b, &c, &mut out);
        for i in 0..7 {
            assert!(approx_eq(out[i], a[i] * 2.0 + 0.5));
        }
    }

    // ---- neon_tensor_clamp ------------------------------------------------

    #[test]
    fn test_clamp_basic() {
        let mut data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0];
        neon_tensor_clamp(&mut data, 0.0, 1.0);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clamp_empty() {
        neon_tensor_clamp(&mut [], 0.0, 1.0);
    }

    #[test]
    fn test_clamp_all_within() {
        let mut data = vec![0.2, 0.5, 0.7, 0.9];
        let expected = data.clone();
        neon_tensor_clamp(&mut data, 0.0, 1.0);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_clamp_boundaries() {
        let mut data = vec![0.0, 1.0];
        neon_tensor_clamp(&mut data, 0.0, 1.0);
        assert_eq!(data, vec![0.0, 1.0]);
    }

    #[test]
    fn test_clamp_negative_range() {
        let mut data = vec![-5.0, 0.0, 5.0];
        neon_tensor_clamp(&mut data, -1.0, 1.0);
        assert_eq!(data, vec![-1.0, 0.0, 1.0]);
    }

    // ---- neon_tensor_abs --------------------------------------------------

    #[test]
    fn test_abs_basic() {
        let data = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let mut out = vec![0.0; 7];
        neon_tensor_abs(&data, &mut out);
        assert_eq!(out, vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_empty() {
        neon_tensor_abs(&[], &mut []);
    }

    #[test]
    fn test_abs_all_positive() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        neon_tensor_abs(&data, &mut out);
        assert_eq!(out, data);
    }

    // ---- neon_tensor_neg --------------------------------------------------

    #[test]
    fn test_neg_basic() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let mut out = vec![0.0; 5];
        neon_tensor_neg(&data, &mut out);
        assert_eq!(out, vec![-1.0, 2.0, -3.0, 4.0, -5.0]);
    }

    #[test]
    fn test_neg_empty() {
        neon_tensor_neg(&[], &mut []);
    }

    #[test]
    fn test_neg_double_neg() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tmp = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_tensor_neg(&data, &mut tmp);
        neon_tensor_neg(&tmp, &mut out);
        for i in 0..4 {
            assert!(approx_eq(out[i], data[i]));
        }
    }

    // ---- neon_tensor_max_reduce -------------------------------------------

    #[test]
    fn test_max_reduce_basic() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert!(approx_eq(neon_tensor_max_reduce(&data), 5.0));
    }

    #[test]
    fn test_max_reduce_empty() {
        assert_eq!(neon_tensor_max_reduce(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn test_max_reduce_single() {
        assert!(approx_eq(neon_tensor_max_reduce(&[42.0]), 42.0));
    }

    #[test]
    fn test_max_reduce_negative() {
        let data = vec![-5.0, -1.0, -3.0, -2.0, -4.0, -0.5, -10.0];
        assert!(approx_eq(neon_tensor_max_reduce(&data), -0.5));
    }

    #[test]
    fn test_max_reduce_lane_boundary() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(neon_tensor_max_reduce(&data), 4.0));
    }

    // ---- neon_tensor_sum_reduce -------------------------------------------

    #[test]
    fn test_sum_reduce_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(neon_tensor_sum_reduce(&data), 15.0));
    }

    #[test]
    fn test_sum_reduce_empty() {
        assert_eq!(neon_tensor_sum_reduce(&[]), 0.0);
    }

    #[test]
    fn test_sum_reduce_single() {
        assert!(approx_eq(neon_tensor_sum_reduce(&[7.5]), 7.5));
    }

    #[test]
    fn test_sum_reduce_non_aligned() {
        let data: Vec<f32> = (1..=13).map(|x| x as f32).collect();
        let expected: f32 = (1..=13).sum::<i32>() as f32;
        assert!(approx_eq(neon_tensor_sum_reduce(&data), expected));
    }

    // ---- neon_tensor_mean -------------------------------------------------

    #[test]
    fn test_mean_basic() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        assert!(approx_eq(neon_tensor_mean(&data), 5.0));
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(neon_tensor_mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert!(approx_eq(neon_tensor_mean(&[99.0]), 99.0));
    }

    // ---- neon_tensor_variance ---------------------------------------------

    #[test]
    fn test_variance_constant() {
        let data = vec![5.0; 8];
        assert!(approx_eq(neon_tensor_variance(&data), 0.0));
    }

    #[test]
    fn test_variance_basic() {
        // [1, 2, 3, 4] → mean=2.5, var = ((1.5²+0.5²+0.5²+1.5²)/4) = 1.25
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(neon_tensor_variance(&data), 1.25));
    }

    #[test]
    fn test_variance_empty() {
        assert_eq!(neon_tensor_variance(&[]), 0.0);
    }

    #[test]
    fn test_variance_single() {
        assert!(approx_eq(neon_tensor_variance(&[10.0]), 0.0));
    }

    #[test]
    fn test_variance_non_aligned() {
        // [1..=7] mean=4.0, var = (9+4+1+0+1+4+9)/7 = 28/7 = 4.0
        let data: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        assert!(approx_eq(neon_tensor_variance(&data), 4.0));
    }

    // ---- neon_tensor_copy -------------------------------------------------

    #[test]
    fn test_copy_basic() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; 5];
        neon_tensor_copy(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_empty() {
        neon_tensor_copy(&[], &mut []);
    }

    #[test]
    fn test_copy_non_aligned() {
        let src: Vec<f32> = (0..13).map(|x| x as f32 * 0.1).collect();
        let mut dst = vec![0.0; 13];
        neon_tensor_copy(&src, &mut dst);
        for i in 0..13 {
            assert!(approx_eq(dst[i], src[i]));
        }
    }

    // ---- neon_tensor_fill -------------------------------------------------

    #[test]
    fn test_fill_basic() {
        let mut data = vec![0.0; 9];
        neon_tensor_fill(&mut data, 7.0);
        assert!(data.iter().all(|&v| approx_eq(v, 7.0)));
    }

    #[test]
    fn test_fill_empty() {
        neon_tensor_fill(&mut [], 1.0);
    }

    #[test]
    fn test_fill_single() {
        let mut data = vec![0.0];
        neon_tensor_fill(&mut data, -3.14);
        assert!(approx_eq(data[0], -3.14));
    }

    #[test]
    fn test_fill_lane_boundary() {
        let mut data = vec![0.0; 4];
        neon_tensor_fill(&mut data, 2.5);
        assert!(data.iter().all(|&v| approx_eq(v, 2.5)));
    }
}
