//! ARM NEON vector reduction operations for Apple Silicon (aarch64).
//!
//! Provides NEON-optimized reductions: sum, max, min, argmax, argmin,
//! dot product, L2 norm, mean, variance, and abs-max for `f32` slices.
//! Each function processes four lanes at a time and falls back to scalar
//! for the remaining tail elements.

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

/// Horizontal sum of an `f32` slice using NEON.
///
/// Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sum_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vaddq_f32(acc, v);
    }

    let mut sum = vaddvq_f32(acc);
    for i in 0..remainder {
        sum += *ptr.add(chunks * 4 + i);
    }
    sum
}

// ---------------------------------------------------------------------------
// max
// ---------------------------------------------------------------------------

/// Maximum value of an `f32` slice using NEON.
///
/// Returns `f32::NEG_INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_max_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vmaxq_f32(acc, v);
    }

    let mut max_val = vmaxvq_f32(acc);
    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

// ---------------------------------------------------------------------------
// min
// ---------------------------------------------------------------------------

/// Minimum value of an `f32` slice using NEON.
///
/// Returns `f32::INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_min_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return f32::INFINITY;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(f32::INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vminq_f32(acc, v);
    }

    let mut min_val = vminvq_f32(acc);
    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

/// Index of the maximum value in an `f32` slice.
///
/// Returns `0` for empty slices. First occurrence wins on ties.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmax_f32(data: &[f32]) -> usize {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    let mut best_idx: usize = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in data.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// argmin
// ---------------------------------------------------------------------------

/// Index of the minimum value in an `f32` slice.
///
/// Returns `0` for empty slices. First occurrence wins on ties.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmin_f32(data: &[f32]) -> usize {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    let mut best_idx: usize = 0;
    let mut best_val = f32::INFINITY;

    for (i, &val) in data.iter().enumerate() {
        if val < best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// dot product
// ---------------------------------------------------------------------------

/// Dot product of two equal-length `f32` slices using NEON `vfmaq_f32`.
///
/// Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");

    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(pa.add(i * 4));
        let vb = vld1q_f32(pb.add(i * 4));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut dot = vaddvq_f32(acc);
    for i in 0..remainder {
        dot += *pa.add(chunks * 4 + i) * *pb.add(chunks * 4 + i);
    }
    dot
}

// ---------------------------------------------------------------------------
// L2 norm
// ---------------------------------------------------------------------------

/// L2 (Euclidean) norm of an `f32` slice using NEON.
///
/// Computes `sqrt(sum of squares)`. Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_l2_norm_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vfmaq_f32(acc, v, v);
    }

    let mut sum_sq = vaddvq_f32(acc);
    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        sum_sq += val * val;
    }
    sum_sq.sqrt()
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

/// Arithmetic mean of an `f32` slice using NEON.
///
/// Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_mean_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    neon_sum_f32(data) / len as f32
}

// ---------------------------------------------------------------------------
// variance
// ---------------------------------------------------------------------------

/// Population variance of an `f32` slice given its mean, using NEON.
///
/// Computes `sum((x - mean)^2) / n`. Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_variance_f32(data: &[f32], mean: f32) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let vmean = vdupq_n_f32(mean);
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let diff = vsubq_f32(v, vmean);
        acc = vfmaq_f32(acc, diff, diff);
    }

    let mut sum_sq = vaddvq_f32(acc);
    for i in 0..remainder {
        let diff = *ptr.add(chunks * 4 + i) - mean;
        sum_sq += diff * diff;
    }
    sum_sq / len as f32
}

// ---------------------------------------------------------------------------
// abs_max
// ---------------------------------------------------------------------------

/// Maximum absolute value of an `f32` slice using NEON.
///
/// Useful for quantization scale computation. Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_abs_max_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let abs_v = vabsq_f32(v);
        acc = vmaxq_f32(acc, abs_v);
    }

    let mut abs_max = vmaxvq_f32(acc);
    for i in 0..remainder {
        let val = (*ptr.add(chunks * 4 + i)).abs();
        if val > abs_max {
            abs_max = val;
        }
    }
    abs_max
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    // Scalar reference helpers
    fn scalar_sum(data: &[f32]) -> f32 {
        data.iter().sum()
    }
    fn scalar_max(data: &[f32]) -> f32 {
        data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }
    fn scalar_min(data: &[f32]) -> f32 {
        data.iter().copied().fold(f32::INFINITY, f32::min)
    }
    fn scalar_argmax(data: &[f32]) -> usize {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    fn scalar_argmin(data: &[f32]) -> usize {
        data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    fn scalar_l2_norm(data: &[f32]) -> f32 {
        data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    fn scalar_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        scalar_sum(data) / data.len() as f32
    }
    fn scalar_variance(data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / data.len() as f32
    }
    fn scalar_abs_max(data: &[f32]) -> f32 {
        data.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    }

    // Sizes used across many tests
    const SIZES: &[usize] = &[0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256, 512, 1024];

    fn make_ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.1 + 0.5).collect()
    }

    // -----------------------------------------------------------------------
    // neon_sum_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_empty() {
        let r = unsafe { neon_sum_f32(&[]) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_sum_single() {
        let r = unsafe { neon_sum_f32(&[42.0]) };
        assert!(approx_eq(r, 42.0));
    }

    #[test]
    fn test_sum_exact_chunk() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let r = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(r, 10.0));
    }

    #[test]
    fn test_sum_with_tail() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let r = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(r, 15.0));
    }

    #[test]
    fn test_sum_all_zeros() {
        let data = vec![0.0f32; 33];
        let r = unsafe { neon_sum_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_sum_negatives() {
        let data = [-1.0f32, -2.0, -3.0, -4.0, -5.0];
        let r = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(r, -15.0));
    }

    #[test]
    fn test_sum_all_same() {
        let data = vec![3.0f32; 17];
        let r = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(r, 51.0));
    }

    #[test]
    fn test_sum_sizes_vs_scalar() {
        for &n in SIZES {
            let data = make_ramp(n);
            let neon_r = unsafe { neon_sum_f32(&data) };
            let scalar_r = scalar_sum(&data);
            assert!(
                (neon_r - scalar_r).abs() < 1e-2,
                "sum mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_sum_ones_equals_length() {
        for &n in SIZES {
            if n == 0 {
                continue;
            }
            let data = vec![1.0f32; n];
            let r = unsafe { neon_sum_f32(&data) };
            assert!(approx_eq(r, n as f32), "sum of {n} ones = {r}, expected {n}");
        }
    }

    // -----------------------------------------------------------------------
    // neon_max_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_empty() {
        let r = unsafe { neon_max_f32(&[]) };
        assert_eq!(r, f32::NEG_INFINITY);
    }

    #[test]
    fn test_max_single() {
        let r = unsafe { neon_max_f32(&[7.0]) };
        assert!(approx_eq(r, 7.0));
    }

    #[test]
    fn test_max_basic() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let r = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_max_all_same() {
        let data = vec![2.5f32; 16];
        let r = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(r, 2.5));
    }

    #[test]
    fn test_max_negatives() {
        let data = [-3.0f32, -1.0, -4.0, -2.0, -5.0];
        let r = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(r, -1.0));
    }

    #[test]
    fn test_max_all_zeros() {
        let data = vec![0.0f32; 9];
        let r = unsafe { neon_max_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_max_sizes_vs_scalar() {
        for &n in SIZES {
            if n == 0 {
                continue;
            }
            let data = make_ramp(n);
            let neon_r = unsafe { neon_max_f32(&data) };
            let scalar_r = scalar_max(&data);
            assert!(
                (neon_r - scalar_r).abs() < TOL,
                "max mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_max_ge_any_element() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3 - 15.0).collect();
        let m = unsafe { neon_max_f32(&data) };
        for &v in &data {
            assert!(m >= v, "max {m} < element {v}");
        }
    }

    #[test]
    fn test_max_large_values() {
        let data = [f32::MAX / 2.0, f32::MAX / 4.0, 1.0, -1.0];
        let r = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(r, f32::MAX / 2.0));
    }

    // -----------------------------------------------------------------------
    // neon_min_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_empty() {
        let r = unsafe { neon_min_f32(&[]) };
        assert_eq!(r, f32::INFINITY);
    }

    #[test]
    fn test_min_single() {
        let r = unsafe { neon_min_f32(&[-3.0]) };
        assert!(approx_eq(r, -3.0));
    }

    #[test]
    fn test_min_basic() {
        let data = [5.0f32, 3.0, 1.0, 4.0, 2.0];
        let r = unsafe { neon_min_f32(&data) };
        assert!(approx_eq(r, 1.0));
    }

    #[test]
    fn test_min_all_same() {
        let data = vec![7.0f32; 15];
        let r = unsafe { neon_min_f32(&data) };
        assert!(approx_eq(r, 7.0));
    }

    #[test]
    fn test_min_negatives() {
        let data = [-3.0f32, -1.0, -4.0, -2.0, -5.0];
        let r = unsafe { neon_min_f32(&data) };
        assert!(approx_eq(r, -5.0));
    }

    #[test]
    fn test_min_all_zeros() {
        let data = vec![0.0f32; 11];
        let r = unsafe { neon_min_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_min_sizes_vs_scalar() {
        for &n in SIZES {
            if n == 0 {
                continue;
            }
            let data = make_ramp(n);
            let neon_r = unsafe { neon_min_f32(&data) };
            let scalar_r = scalar_min(&data);
            assert!(
                (neon_r - scalar_r).abs() < TOL,
                "min mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_min_le_any_element() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3 - 15.0).collect();
        let m = unsafe { neon_min_f32(&data) };
        for &v in &data {
            assert!(m <= v, "min {m} > element {v}");
        }
    }

    #[test]
    fn test_min_large_negative() {
        let data = [f32::MIN / 2.0, f32::MIN / 4.0, -1.0, 1.0];
        let r = unsafe { neon_min_f32(&data) };
        assert!(approx_eq(r, f32::MIN / 2.0));
    }

    // -----------------------------------------------------------------------
    // neon_argmax_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_argmax_empty() {
        let r = unsafe { neon_argmax_f32(&[]) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmax_single() {
        let r = unsafe { neon_argmax_f32(&[99.0]) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmax_basic() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let r = unsafe { neon_argmax_f32(&data) };
        assert_eq!(r, 1);
    }

    #[test]
    fn test_argmax_first_occurrence() {
        let data = [3.0f32, 1.0, 3.0, 2.0, 3.0];
        let r = unsafe { neon_argmax_f32(&data) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmax_last_element() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let r = unsafe { neon_argmax_f32(&data) };
        assert_eq!(r, 4);
    }

    #[test]
    fn test_argmax_all_same() {
        let data = vec![4.0f32; 8];
        let r = unsafe { neon_argmax_f32(&data) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmax_negatives() {
        let data = [-5.0f32, -3.0, -4.0, -1.0, -2.0];
        let r = unsafe { neon_argmax_f32(&data) };
        assert_eq!(r, 3);
    }

    #[test]
    fn test_argmax_sizes_vs_scalar() {
        for &n in SIZES {
            if n == 0 {
                continue;
            }
            let data = make_ramp(n);
            let neon_r = unsafe { neon_argmax_f32(&data) };
            let scalar_r = scalar_argmax(&data);
            assert_eq!(neon_r, scalar_r, "argmax mismatch at n={n}");
        }
    }

    #[test]
    fn test_argmax_points_to_max() {
        let data: Vec<f32> = (0..64).map(|i| ((i * 7) % 64) as f32).collect();
        let idx = unsafe { neon_argmax_f32(&data) };
        let max_val = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(data[idx], max_val));
    }

    // -----------------------------------------------------------------------
    // neon_argmin_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_argmin_empty() {
        let r = unsafe { neon_argmin_f32(&[]) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmin_single() {
        let r = unsafe { neon_argmin_f32(&[99.0]) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmin_basic() {
        let data = [5.0f32, 3.0, 1.0, 4.0, 2.0];
        let r = unsafe { neon_argmin_f32(&data) };
        assert_eq!(r, 2);
    }

    #[test]
    fn test_argmin_first_occurrence() {
        let data = [3.0f32, 1.0, 1.0, 2.0, 1.0];
        let r = unsafe { neon_argmin_f32(&data) };
        assert_eq!(r, 1);
    }

    #[test]
    fn test_argmin_last_element() {
        let data = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let r = unsafe { neon_argmin_f32(&data) };
        assert_eq!(r, 4);
    }

    #[test]
    fn test_argmin_all_same() {
        let data = vec![4.0f32; 8];
        let r = unsafe { neon_argmin_f32(&data) };
        assert_eq!(r, 0);
    }

    #[test]
    fn test_argmin_negatives() {
        let data = [-1.0f32, -3.0, -2.0, -5.0, -4.0];
        let r = unsafe { neon_argmin_f32(&data) };
        assert_eq!(r, 3);
    }

    #[test]
    fn test_argmin_sizes_vs_scalar() {
        for &n in SIZES {
            if n == 0 {
                continue;
            }
            let data = make_ramp(n);
            let neon_r = unsafe { neon_argmin_f32(&data) };
            let scalar_r = scalar_argmin(&data);
            assert_eq!(neon_r, scalar_r, "argmin mismatch at n={n}");
        }
    }

    #[test]
    fn test_argmin_points_to_min() {
        let data: Vec<f32> = (0..64).map(|i| ((i * 7) % 64) as f32).collect();
        let idx = unsafe { neon_argmin_f32(&data) };
        let min_val = unsafe { neon_min_f32(&data) };
        assert!(approx_eq(data[idx], min_val));
    }

    // -----------------------------------------------------------------------
    // neon_dot_product_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_empty() {
        let r = unsafe { neon_dot_product_f32(&[], &[]) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_dot_single() {
        let r = unsafe { neon_dot_product_f32(&[3.0], &[4.0]) };
        assert!(approx_eq(r, 12.0));
    }

    #[test]
    fn test_dot_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let r = unsafe { neon_dot_product_f32(&a, &b) };
        assert!(approx_eq(r, 35.0));
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        let r = unsafe { neon_dot_product_f32(&a, &b) };
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_dot_parallel() {
        let a = [2.0f32, 3.0, 4.0, 5.0];
        let b = [4.0f32, 6.0, 8.0, 10.0]; // 2*a
        // dot = 2*4 + 3*6 + 4*8 + 5*10 = 8+18+32+50 = 108
        let r = unsafe { neon_dot_product_f32(&a, &b) };
        assert!(approx_eq(r, 108.0));
    }

    #[test]
    fn test_dot_self_equals_l2_squared() {
        let data = [3.0f32, 4.0];
        let dot = unsafe { neon_dot_product_f32(&data, &data) };
        // 9 + 16 = 25
        assert!(approx_eq(dot, 25.0));
    }

    #[test]
    fn test_dot_all_zeros() {
        let a = vec![0.0f32; 17];
        let b = vec![1.0f32; 17];
        let r = unsafe { neon_dot_product_f32(&a, &b) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_dot_negatives() {
        let a = [-1.0f32, -2.0, -3.0];
        let b = [1.0f32, 2.0, 3.0];
        // -1 + -4 + -9 = -14
        let r = unsafe { neon_dot_product_f32(&a, &b) };
        assert!(approx_eq(r, -14.0));
    }

    #[test]
    fn test_dot_sizes_vs_scalar() {
        for &n in SIZES {
            let a = make_ramp(n);
            let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
            let neon_r = unsafe { neon_dot_product_f32(&a, &b) };
            let scalar_r = scalar_dot(&a, &b);
            assert!(
                (neon_r - scalar_r).abs() < 1e-2,
                "dot mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "equal-length")]
    fn test_dot_length_mismatch() {
        unsafe { neon_dot_product_f32(&[1.0, 2.0], &[1.0]) };
    }

    // -----------------------------------------------------------------------
    // neon_l2_norm_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_norm_empty() {
        let r = unsafe { neon_l2_norm_f32(&[]) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_l2_norm_single() {
        let r = unsafe { neon_l2_norm_f32(&[5.0]) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_l2_norm_3_4_5() {
        let data = [3.0f32, 4.0];
        let r = unsafe { neon_l2_norm_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let inv_sqrt3 = 1.0f32 / 3.0f32.sqrt();
        let data = [inv_sqrt3, inv_sqrt3, inv_sqrt3];
        let r = unsafe { neon_l2_norm_f32(&data) };
        assert!((r - 1.0).abs() < TOL, "unit vector norm = {r}");
    }

    #[test]
    fn test_l2_norm_zero_vector() {
        let data = vec![0.0f32; 16];
        let r = unsafe { neon_l2_norm_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_l2_norm_negative_values() {
        let data = [-3.0f32, -4.0];
        let r = unsafe { neon_l2_norm_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_l2_norm_sizes_vs_scalar() {
        for &n in SIZES {
            let data = make_ramp(n);
            let neon_r = unsafe { neon_l2_norm_f32(&data) };
            let scalar_r = scalar_l2_norm(&data);
            assert!(
                (neon_r - scalar_r).abs() < 1e-2,
                "l2_norm mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_l2_norm_single_negative() {
        let r = unsafe { neon_l2_norm_f32(&[-7.0]) };
        assert!(approx_eq(r, 7.0));
    }

    // -----------------------------------------------------------------------
    // neon_mean_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_empty() {
        let r = unsafe { neon_mean_f32(&[]) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_mean_single() {
        let r = unsafe { neon_mean_f32(&[42.0]) };
        assert!(approx_eq(r, 42.0));
    }

    #[test]
    fn test_mean_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let r = unsafe { neon_mean_f32(&data) };
        assert!(approx_eq(r, 3.0));
    }

    #[test]
    fn test_mean_all_same() {
        let data = vec![5.0f32; 33];
        let r = unsafe { neon_mean_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_mean_all_zeros() {
        let data = vec![0.0f32; 16];
        let r = unsafe { neon_mean_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_mean_negatives() {
        let data = [-2.0f32, -4.0, -6.0];
        let r = unsafe { neon_mean_f32(&data) };
        assert!(approx_eq(r, -4.0));
    }

    #[test]
    fn test_mean_sizes_vs_scalar() {
        for &n in SIZES {
            let data = make_ramp(n);
            let neon_r = unsafe { neon_mean_f32(&data) };
            let scalar_r = scalar_mean(&data);
            assert!(
                (neon_r - scalar_r).abs() < 1e-3,
                "mean mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_mean_symmetric() {
        let data = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let r = unsafe { neon_mean_f32(&data) };
        assert!(approx_eq(r, 0.0));
    }

    // -----------------------------------------------------------------------
    // neon_variance_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_variance_empty() {
        let r = unsafe { neon_variance_f32(&[], 0.0) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_variance_single() {
        let r = unsafe { neon_variance_f32(&[5.0], 5.0) };
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_variance_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        // var = (4+1+0+1+4)/5 = 2.0
        let r = unsafe { neon_variance_f32(&data, mean) };
        assert!(approx_eq(r, 2.0));
    }

    #[test]
    fn test_variance_all_same() {
        let data = vec![7.0f32; 16];
        let r = unsafe { neon_variance_f32(&data, 7.0) };
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_variance_all_zeros() {
        let data = vec![0.0f32; 8];
        let r = unsafe { neon_variance_f32(&data, 0.0) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_variance_sizes_vs_scalar() {
        for &n in SIZES {
            let data = make_ramp(n);
            let mean = scalar_mean(&data);
            let neon_r = unsafe { neon_variance_f32(&data, mean) };
            let scalar_r = scalar_variance(&data, mean);
            assert!(
                (neon_r - scalar_r).abs() < 1e-3,
                "variance mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_variance_known() {
        // [0, 10]: mean=5, var = (25+16+9+4+1+0+1+4+9+16+25)/11 = 10.0
        let data: Vec<f32> = (0..=10).map(|i| i as f32).collect();
        let r = unsafe { neon_variance_f32(&data, 5.0) };
        assert!(approx_eq(r, 10.0));
    }

    #[test]
    fn test_variance_negative_data() {
        let data = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let r = unsafe { neon_variance_f32(&data, 0.0) };
        // var = (4+1+0+1+4)/5 = 2.0
        assert!(approx_eq(r, 2.0));
    }

    // -----------------------------------------------------------------------
    // neon_abs_max_f32
    // -----------------------------------------------------------------------

    #[test]
    fn test_abs_max_empty() {
        let r = unsafe { neon_abs_max_f32(&[]) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_abs_max_single_positive() {
        let r = unsafe { neon_abs_max_f32(&[5.0]) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_abs_max_single_negative() {
        let r = unsafe { neon_abs_max_f32(&[-5.0]) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_abs_max_basic() {
        let data = [1.0f32, -7.0, 3.0, 2.0, -4.0];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert!(approx_eq(r, 7.0));
    }

    #[test]
    fn test_abs_max_all_zeros() {
        let data = vec![0.0f32; 16];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_abs_max_all_same() {
        let data = vec![-3.0f32; 9];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert!(approx_eq(r, 3.0));
    }

    #[test]
    fn test_abs_max_positive_only() {
        let data = [1.0f32, 5.0, 3.0, 2.0];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_abs_max_negative_only() {
        let data = [-1.0f32, -5.0, -3.0, -2.0, -4.0];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert!(approx_eq(r, 5.0));
    }

    #[test]
    fn test_abs_max_sizes_vs_scalar() {
        for &n in SIZES {
            let data: Vec<f32> =
                (0..n).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
            let neon_r = unsafe { neon_abs_max_f32(&data) };
            let scalar_r = scalar_abs_max(&data);
            assert!(
                (neon_r - scalar_r).abs() < TOL,
                "abs_max mismatch at n={n}: neon={neon_r}, scalar={scalar_r}"
            );
        }
    }

    #[test]
    fn test_abs_max_large_values() {
        let data = [1.0f32, -(f32::MAX / 2.0), f32::MAX / 4.0];
        let r = unsafe { neon_abs_max_f32(&data) };
        assert!(approx_eq(r, f32::MAX / 2.0));
    }

    // -----------------------------------------------------------------------
    // Cross-function property tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_equals_sum_div_len() {
        let data = make_ramp(100);
        let sum = unsafe { neon_sum_f32(&data) };
        let mean = unsafe { neon_mean_f32(&data) };
        assert!(approx_eq(mean, sum / data.len() as f32));
    }

    #[test]
    fn test_l2_norm_equals_sqrt_dot_self() {
        let data = make_ramp(64);
        let norm = unsafe { neon_l2_norm_f32(&data) };
        let dot_self = unsafe { neon_dot_product_f32(&data, &data) };
        assert!((norm - dot_self.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn test_max_min_range() {
        let data = make_ramp(128);
        let max = unsafe { neon_max_f32(&data) };
        let min = unsafe { neon_min_f32(&data) };
        assert!(max >= min);
    }

    #[test]
    fn test_argmax_argmin_different_for_distinct() {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let amax = unsafe { neon_argmax_f32(&data) };
        let amin = unsafe { neon_argmin_f32(&data) };
        assert_ne!(amax, amin);
    }

    #[test]
    fn test_abs_max_ge_max_abs_min() {
        let data: Vec<f32> = (0..50).map(|i| (i as f32) - 25.0).collect();
        let abs_max = unsafe { neon_abs_max_f32(&data) };
        let max = unsafe { neon_max_f32(&data) };
        let min = unsafe { neon_min_f32(&data) };
        assert!(abs_max >= max.abs());
        assert!(abs_max >= min.abs());
    }

    #[test]
    fn test_variance_zero_for_constant() {
        for n in [1, 4, 7, 16, 33] {
            let data = vec![42.0f32; n];
            let v = unsafe { neon_variance_f32(&data, 42.0) };
            assert!(approx_eq(v, 0.0), "variance of constant at n={n}: {v}");
        }
    }

    #[test]
    fn test_dot_commutative() {
        let a = make_ramp(50);
        let b: Vec<f32> = a.iter().map(|x| x * 2.0 - 1.0).collect();
        let ab = unsafe { neon_dot_product_f32(&a, &b) };
        let ba = unsafe { neon_dot_product_f32(&b, &a) };
        assert!(approx_eq(ab, ba));
    }

    #[test]
    fn test_sum_negation() {
        let data = make_ramp(33);
        let neg: Vec<f32> = data.iter().map(|x| -x).collect();
        let s1 = unsafe { neon_sum_f32(&data) };
        let s2 = unsafe { neon_sum_f32(&neg) };
        assert!(approx_eq(s1, -s2));
    }
}
