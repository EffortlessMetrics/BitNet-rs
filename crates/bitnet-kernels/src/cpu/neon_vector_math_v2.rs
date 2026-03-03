//! NEON vector math v2 operations for Apple Silicon.
//!
//! Provides optimized f32 vector arithmetic, reductions, and similarity
//! functions using ARM NEON intrinsics, with scalar fallback for remainders.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Element-wise binary operations
// ---------------------------------------------------------------------------

/// Element-wise addition: `out[i] = a[i] + b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_add_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vaddq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_sub_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vsubq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] - b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mul_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vmulq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i];
    }
}

/// Element-wise division: `out[i] = a[i] / b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_div_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vdivq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] / b[i];
    }
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

/// Dot product using NEON fused multiply-accumulate and horizontal add.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

// ---------------------------------------------------------------------------
// Norms and similarity
// ---------------------------------------------------------------------------

/// L2 (Euclidean) norm of a vector.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_l2_norm_neon(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;
    let ptr = a.as_ptr();

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let v = vld1q_f32(ptr.add(off));
            acc = vfmaq_f32(acc, v, v);
        }
    }

    let mut sum_sq = vaddvq_f32(acc);
    for &val in &a[(chunks * 4)..] {
        sum_sq += val * val;
    }
    sum_sq.sqrt()
}

/// Cosine similarity between two vectors: `dot(a,b) / (‖a‖ * ‖b‖)`.
///
/// Returns `0.0` when either vector has zero norm.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc_dot = vdupq_n_f32(0.0);
    let mut acc_aa = vdupq_n_f32(0.0);
    let mut acc_bb = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            acc_dot = vfmaq_f32(acc_dot, va, vb);
            acc_aa = vfmaq_f32(acc_aa, va, va);
            acc_bb = vfmaq_f32(acc_bb, vb, vb);
        }
    }

    let mut dot = vaddvq_f32(acc_dot);
    let mut norm_a_sq = vaddvq_f32(acc_aa);
    let mut norm_b_sq = vaddvq_f32(acc_bb);

    for i in (chunks * 4)..n {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Element-wise unary / clamping
// ---------------------------------------------------------------------------

/// Element-wise maximum: `out[i] = max(a[i], b[i])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_max_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vmaxq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].max(b[i]);
    }
}

/// Element-wise minimum: `out[i] = min(a[i], b[i])`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_min_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let vb = vld1q_f32(b_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vminq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].min(b[i]);
    }
}

/// Element-wise absolute value: `out[i] = |a[i]|`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_abs_neon(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            vst1q_f32(o_ptr.add(off), vabsq_f32(va));
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].abs();
    }
}

/// Element-wise clamp: `out[i] = clamp(a[i], lo, hi)`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_clamp_neon(a: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    let n = a.len();
    let chunks = n / 4;
    let a_ptr = a.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let v_lo = vdupq_n_f32(lo);
    let v_hi = vdupq_n_f32(hi);

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let va = vld1q_f32(a_ptr.add(off));
            let clamped = vminq_f32(vmaxq_f32(va, v_lo), v_hi);
            vst1q_f32(o_ptr.add(off), clamped);
        }
    }
    for i in (chunks * 4)..n {
        out[i] = a[i].clamp(lo, hi);
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Horizontal sum of all elements.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_reduce_sum_neon(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let chunks = n / 4;
    let ptr = a.as_ptr();

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            acc = vaddq_f32(acc, vld1q_f32(ptr.add(off)));
        }
    }

    let mut sum = vaddvq_f32(acc);
    for &val in &a[(chunks * 4)..] {
        sum += val;
    }
    sum
}

/// Horizontal maximum. Returns `f32::NEG_INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_reduce_max_neon(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }
    let chunks = n / 4;
    let ptr = a.as_ptr();

    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            acc = vmaxq_f32(acc, vld1q_f32(ptr.add(off)));
        }
    }

    let mut max_val = vmaxvq_f32(acc);
    for &val in &a[(chunks * 4)..] {
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// Horizontal minimum. Returns `f32::INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_reduce_min_neon(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let chunks = n / 4;
    let ptr = a.as_ptr();

    let mut acc = vdupq_n_f32(f32::INFINITY);
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            acc = vminq_f32(acc, vld1q_f32(ptr.add(off)));
        }
    }

    let mut min_val = vminvq_f32(acc);
    for &val in &a[(chunks * 4)..] {
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn assert_slices_approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, e), "index {i}: expected {e}, got {a}");
        }
    }

    // -----------------------------------------------------------------------
    // vec_add_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_exact_chunk() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_add_neon(&a, &b, &mut out) };
        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut out = [0.0f32; 7];
        unsafe { vec_add_neon(&a, &b, &mut out) };
        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_add_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_add_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_add_single() {
        let mut out = [0.0f32; 1];
        unsafe { vec_add_neon(&[3.0], &[4.0], &mut out) };
        assert_eq!(out, [7.0]);
    }

    #[test]
    fn test_add_negatives() {
        let a = [-1.0f32, -2.0, -3.0, -4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_add_neon(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // vec_sub_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_sub_exact_chunk() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_sub_neon(&a, &b, &mut out) };
        assert_eq!(out, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_sub_with_remainder() {
        let a = [5.0f32, 10.0, 15.0, 20.0, 25.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_sub_neon(&a, &b, &mut out) };
        assert_eq!(out, [4.0, 8.0, 12.0, 16.0, 20.0]);
    }

    #[test]
    fn test_sub_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_sub_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_sub_self_is_zero() {
        let a = [3.0f32, 7.0, 11.0, 13.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_sub_neon(&a, &a, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sub_negative_result() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_sub_neon(&a, &b, &mut out) };
        assert_eq!(out, [-9.0, -18.0, -27.0, -36.0]);
    }

    // -----------------------------------------------------------------------
    // vec_mul_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_exact_chunk() {
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let b = [0.5f32, 1.0, 2.0, 3.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_mul_neon(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 3.0, 8.0, 15.0]);
    }

    #[test]
    fn test_mul_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = [0.0f32; 6];
        unsafe { vec_mul_neon(&a, &b, &mut out) };
        assert_eq!(out, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_mul_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_mul_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_mul_by_zero() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { vec_mul_neon(&a, &b, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_by_one() {
        let a = [7.0f32, 8.0, 9.0, 10.0];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { vec_mul_neon(&a, &b, &mut out) };
        assert_eq!(out, [7.0, 8.0, 9.0, 10.0]);
    }

    // -----------------------------------------------------------------------
    // vec_div_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_div_exact_chunk() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 4.0, 5.0, 8.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_div_neon(&a, &b, &mut out) };
        assert_eq!(out, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_div_with_remainder() {
        let a = [8.0f32, 9.0, 12.0, 15.0, 20.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 10.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_div_neon(&a, &b, &mut out) };
        assert_slices_approx(&out, &[4.0, 3.0, 3.0, 3.0, 2.0]);
    }

    #[test]
    fn test_div_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_div_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_div_by_one() {
        let a = [3.0f32, 6.0, 9.0, 12.0];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { vec_div_neon(&a, &b, &mut out) };
        assert_eq!(out, [3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_div_self_is_one() {
        let a = [5.0f32, 10.0, 15.0, 20.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_div_neon(&a, &a, &mut out) };
        assert_slices_approx(&out, &[1.0, 1.0, 1.0, 1.0]);
    }

    // -----------------------------------------------------------------------
    // vec_dot_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        // 5 + 12 + 21 + 32 = 70
        let r = unsafe { vec_dot_neon(&a, &b) };
        assert!(approx_eq(r, 70.0), "expected 70.0, got {r}");
    }

    #[test]
    fn test_dot_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        // 5 + 8 + 9 + 8 + 5 = 35
        let r = unsafe { vec_dot_neon(&a, &b) };
        assert!(approx_eq(r, 35.0), "expected 35.0, got {r}");
    }

    #[test]
    fn test_dot_empty() {
        let r = unsafe { vec_dot_neon(&[], &[]) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let r = unsafe { vec_dot_neon(&a, &b) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_dot_single() {
        let r = unsafe { vec_dot_neon(&[3.0], &[4.0]) };
        assert!(approx_eq(r, 12.0), "expected 12.0, got {r}");
    }

    #[test]
    fn test_dot_large() {
        let a: Vec<f32> = (1..=256).map(|x| x as f32).collect();
        let b = vec![1.0f32; 256];
        // sum of 1..=256 = 256*257/2 = 32896
        let r = unsafe { vec_dot_neon(&a, &b) };
        assert!(approx_eq(r, 32896.0), "expected 32896.0, got {r}");
    }

    // -----------------------------------------------------------------------
    // vec_l2_norm_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_norm_3_4() {
        let a = [3.0f32, 4.0];
        let r = unsafe { vec_l2_norm_neon(&a) };
        assert!(approx_eq(r, 5.0), "expected 5.0, got {r}");
    }

    #[test]
    fn test_l2_norm_unit() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let r = unsafe { vec_l2_norm_neon(&a) };
        assert!(approx_eq(r, 1.0), "expected 1.0, got {r}");
    }

    #[test]
    fn test_l2_norm_empty() {
        let r = unsafe { vec_l2_norm_neon(&[]) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_l2_norm_single() {
        let r = unsafe { vec_l2_norm_neon(&[-7.0]) };
        assert!(approx_eq(r, 7.0), "expected 7.0, got {r}");
    }

    #[test]
    fn test_l2_norm_all_ones() {
        let a = [1.0f32; 9];
        let r = unsafe { vec_l2_norm_neon(&a) };
        assert!(approx_eq(r, 3.0), "expected 3.0, got {r}");
    }

    // -----------------------------------------------------------------------
    // vec_cosine_similarity_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_identical() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &a) };
        assert!(approx_eq(r, 1.0), "expected 1.0, got {r}");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [-1.0f32, 0.0, 0.0, 0.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &b) };
        assert!(approx_eq(r, -1.0), "expected -1.0, got {r}");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &b) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0f32; 4];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &b) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_cosine_scaled() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 4.0, 6.0, 8.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &b) };
        assert!(approx_eq(r, 1.0), "expected 1.0 for scaled vectors, got {r}");
    }

    #[test]
    fn test_cosine_with_remainder() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 1.0];
        let b = [0.0f32, 0.0, 0.0, 0.0, 1.0];
        // dot = 1, norm_a = sqrt(2), norm_b = 1 => cos = 1/sqrt(2) ≈ 0.7071
        let r = unsafe { vec_cosine_similarity_neon(&a, &b) };
        assert!(approx_eq(r, std::f32::consts::FRAC_1_SQRT_2), "expected ~0.7071, got {r}");
    }

    // -----------------------------------------------------------------------
    // vec_max_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_basic() {
        let a = [1.0f32, 5.0, 3.0, 7.0];
        let b = [4.0f32, 2.0, 6.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_max_neon(&a, &b, &mut out) };
        assert_eq!(out, [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_max_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 10.0];
        let b = [5.0f32, 0.0, 7.0, 1.0, 2.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_max_neon(&a, &b, &mut out) };
        assert_eq!(out, [5.0, 2.0, 7.0, 4.0, 10.0]);
    }

    #[test]
    fn test_max_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_max_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_max_negative() {
        let a = [-1.0f32, -5.0, -3.0, -7.0];
        let b = [-4.0f32, -2.0, -6.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_max_neon(&a, &b, &mut out) };
        assert_eq!(out, [-1.0, -2.0, -3.0, -1.0]);
    }

    // -----------------------------------------------------------------------
    // vec_min_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_basic() {
        let a = [1.0f32, 5.0, 3.0, 7.0];
        let b = [4.0f32, 2.0, 6.0, 0.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_min_neon(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_min_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 10.0];
        let b = [5.0f32, 0.0, 7.0, 1.0, 2.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_min_neon(&a, &b, &mut out) };
        assert_eq!(out, [1.0, 0.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_min_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_min_neon(&[], &[], &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_min_negative() {
        let a = [-1.0f32, -5.0, -3.0, -7.0];
        let b = [-4.0f32, -2.0, -6.0, -1.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_min_neon(&a, &b, &mut out) };
        assert_eq!(out, [-4.0, -5.0, -6.0, -7.0]);
    }

    // -----------------------------------------------------------------------
    // vec_abs_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_abs_mixed() {
        let a = [-1.0f32, 2.0, -3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_abs_neon(&a, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs_all_negative() {
        let a = [-5.0f32, -10.0, -15.0, -20.0, -25.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_abs_neon(&a, &mut out) };
        assert_eq!(out, [5.0, 10.0, 15.0, 20.0, 25.0]);
    }

    #[test]
    fn test_abs_all_positive() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_abs_neon(&a, &mut out) };
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs_zeros() {
        let a = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe { vec_abs_neon(&a, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_abs_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_abs_neon(&[], &mut out) };
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // vec_clamp_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_clamp_basic() {
        let a = [-2.0f32, 0.5, 1.5, 3.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_clamp_neon(&a, 0.0, 1.0, &mut out) };
        assert_eq!(out, [0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_clamp_all_inside() {
        let a = [0.2f32, 0.4, 0.6, 0.8];
        let mut out = [0.0f32; 4];
        unsafe { vec_clamp_neon(&a, 0.0, 1.0, &mut out) };
        assert_eq!(out, [0.2, 0.4, 0.6, 0.8]);
    }

    #[test]
    fn test_clamp_all_below() {
        let a = [-5.0f32, -4.0, -3.0, -2.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_clamp_neon(&a, 0.0, 10.0, &mut out) };
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clamp_all_above() {
        let a = [50.0f32, 60.0, 70.0, 80.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_clamp_neon(&a, 0.0, 10.0, &mut out) };
        assert_eq!(out, [10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clamp_with_remainder() {
        let a = [-1.0f32, 0.5, 1.5, 0.3, 2.0];
        let mut out = [0.0f32; 5];
        unsafe { vec_clamp_neon(&a, 0.0, 1.0, &mut out) };
        assert_slices_approx(&out, &[0.0, 0.5, 1.0, 0.3, 1.0]);
    }

    #[test]
    fn test_clamp_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { vec_clamp_neon(&[], 0.0, 1.0, &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_clamp_negative_range() {
        let a = [-5.0f32, -1.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        unsafe { vec_clamp_neon(&a, -3.0, -0.5, &mut out) };
        assert_eq!(out, [-3.0, -1.0, -0.5, -0.5]);
    }

    // -----------------------------------------------------------------------
    // vec_reduce_sum_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let r = unsafe { vec_reduce_sum_neon(&a) };
        assert!(approx_eq(r, 15.0), "expected 15.0, got {r}");
    }

    #[test]
    fn test_reduce_sum_empty() {
        let r = unsafe { vec_reduce_sum_neon(&[]) };
        assert!(approx_eq(r, 0.0), "expected 0.0, got {r}");
    }

    #[test]
    fn test_reduce_sum_single() {
        let r = unsafe { vec_reduce_sum_neon(&[42.0]) };
        assert!(approx_eq(r, 42.0), "expected 42.0, got {r}");
    }

    #[test]
    fn test_reduce_sum_large() {
        let data: Vec<f32> = (1..=1024).map(|x| x as f32).collect();
        let expected: f32 = (1..=1024).map(|x| x as f32).sum();
        let r = unsafe { vec_reduce_sum_neon(&data) };
        assert!(approx_eq(r, expected), "expected {expected}, got {r}");
    }

    #[test]
    fn test_reduce_sum_negative() {
        let a = [-1.0f32, -2.0, -3.0, -4.0];
        let r = unsafe { vec_reduce_sum_neon(&a) };
        assert!(approx_eq(r, -10.0), "expected -10.0, got {r}");
    }

    // -----------------------------------------------------------------------
    // vec_reduce_max_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_max_basic() {
        let a = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let r = unsafe { vec_reduce_max_neon(&a) };
        assert!(approx_eq(r, 5.0), "expected 5.0, got {r}");
    }

    #[test]
    fn test_reduce_max_empty() {
        let r = unsafe { vec_reduce_max_neon(&[]) };
        assert_eq!(r, f32::NEG_INFINITY);
    }

    #[test]
    fn test_reduce_max_single() {
        let r = unsafe { vec_reduce_max_neon(&[7.0]) };
        assert!(approx_eq(r, 7.0), "expected 7.0, got {r}");
    }

    #[test]
    fn test_reduce_max_all_negative() {
        let a = [-3.0f32, -1.0, -4.0, -2.0];
        let r = unsafe { vec_reduce_max_neon(&a) };
        assert!(approx_eq(r, -1.0), "expected -1.0, got {r}");
    }

    #[test]
    fn test_reduce_max_with_remainder() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 99.0];
        let r = unsafe { vec_reduce_max_neon(&a) };
        assert!(approx_eq(r, 99.0), "expected 99.0, got {r}");
    }

    // -----------------------------------------------------------------------
    // vec_reduce_min_neon
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_min_basic() {
        let a = [5.0f32, 1.0, 3.0, 2.0, 4.0];
        let r = unsafe { vec_reduce_min_neon(&a) };
        assert!(approx_eq(r, 1.0), "expected 1.0, got {r}");
    }

    #[test]
    fn test_reduce_min_empty() {
        let r = unsafe { vec_reduce_min_neon(&[]) };
        assert_eq!(r, f32::INFINITY);
    }

    #[test]
    fn test_reduce_min_single() {
        let r = unsafe { vec_reduce_min_neon(&[7.0]) };
        assert!(approx_eq(r, 7.0), "expected 7.0, got {r}");
    }

    #[test]
    fn test_reduce_min_all_positive() {
        let a = [10.0f32, 5.0, 8.0, 3.0];
        let r = unsafe { vec_reduce_min_neon(&a) };
        assert!(approx_eq(r, 3.0), "expected 3.0, got {r}");
    }

    #[test]
    fn test_reduce_min_with_remainder() {
        let a = [10.0f32, 20.0, 30.0, 40.0, -5.0];
        let r = unsafe { vec_reduce_min_neon(&a) };
        assert!(approx_eq(r, -5.0), "expected -5.0, got {r}");
    }

    // -----------------------------------------------------------------------
    // Cross-function integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_equals_reduce_sum_of_mul() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let mut prod = [0.0f32; 5];
        unsafe { vec_mul_neon(&a, &b, &mut prod) };
        let sum = unsafe { vec_reduce_sum_neon(&prod) };
        let dot = unsafe { vec_dot_neon(&a, &b) };
        assert!(approx_eq(sum, dot), "sum-of-products {sum} != dot {dot}");
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut sum = [0.0f32; 5];
        let mut roundtrip = [0.0f32; 5];
        unsafe {
            vec_add_neon(&a, &b, &mut sum);
            vec_sub_neon(&sum, &b, &mut roundtrip);
        }
        assert_slices_approx(&roundtrip, &a);
    }

    #[test]
    fn test_mul_div_roundtrip() {
        let a = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let b = [3.0f32, 5.0, 7.0, 9.0, 11.0];
        let mut prod = [0.0f32; 5];
        let mut roundtrip = [0.0f32; 5];
        unsafe {
            vec_mul_neon(&a, &b, &mut prod);
            vec_div_neon(&prod, &b, &mut roundtrip);
        }
        assert_slices_approx(&roundtrip, &a);
    }

    #[test]
    fn test_clamp_then_abs_noop() {
        let a = [0.5f32, 0.3, 0.9, 0.1];
        let mut clamped = [0.0f32; 4];
        let mut absed = [0.0f32; 4];
        unsafe {
            vec_clamp_neon(&a, 0.0, 1.0, &mut clamped);
            vec_abs_neon(&clamped, &mut absed);
        }
        assert_slices_approx(&absed, &clamped);
    }

    #[test]
    fn test_l2_norm_via_dot() {
        let a = [3.0f32, 4.0, 0.0, 0.0];
        let norm = unsafe { vec_l2_norm_neon(&a) };
        let dot_self = unsafe { vec_dot_neon(&a, &a) };
        assert!(
            approx_eq(norm, dot_self.sqrt()),
            "l2_norm {norm} != sqrt(dot_self) {}",
            dot_self.sqrt()
        );
    }

    #[test]
    fn test_reduce_max_min_bounds() {
        let a = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let max = unsafe { vec_reduce_max_neon(&a) };
        let min = unsafe { vec_reduce_min_neon(&a) };
        assert!(max >= min, "max {max} should be >= min {min}");
        assert!(approx_eq(max, 9.0));
        assert!(approx_eq(min, 1.0));
    }

    #[test]
    fn test_cosine_of_unit_vectors() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let r = unsafe { vec_cosine_similarity_neon(&a, &a) };
        assert!(approx_eq(r, 1.0), "expected 1.0, got {r}");
    }
}
