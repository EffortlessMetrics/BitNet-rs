//! ARM NEON reduction operations for Apple Silicon.
//!
//! Provides NEON-optimized horizontal sum, max, argmax, dot product,
//! and L2 norm for `f32` slices.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Horizontal sum of an `f32` slice using NEON.
///
/// Accumulates four lanes at a time with `vaddq_f32`, then reduces
/// with `vaddvq_f32`. Returns `0.0` for empty slices.
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
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        acc = vaddq_f32(acc, v);
    }

    let mut sum = vaddvq_f32(acc);

    for i in 0..remainder {
        sum += unsafe { *ptr.add(chunks * 4 + i) };
    }

    sum
}

/// Horizontal max of an `f32` slice using NEON.
///
/// Uses `vmaxq_f32` across four-lane chunks, then `vmaxvq_f32` for
/// the final reduction. Returns `f32::NEG_INFINITY` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_max_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    let ptr = data.as_ptr();
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        acc = vmaxq_f32(acc, v);
    }

    let mut max_val = vmaxvq_f32(acc);

    for i in 0..remainder {
        let val = unsafe { *ptr.add(chunks * 4 + i) };
        if val > max_val {
            max_val = val;
        }
    }

    max_val
}

/// Index of the maximum value in an `f32` slice.
///
/// Performs a scalar scan tracking the running maximum. Returns `0`
/// for empty slices. When multiple elements share the maximum value
/// the index of the first occurrence is returned.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmax_f32(data: &[f32]) -> usize {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    let mut best_idx: usize = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (i, &val) in data.iter().enumerate().take(len) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }

    best_idx
}

/// Dot product of two equal-length `f32` slices using NEON.
///
/// Accumulates with `vfmaq_f32` (fused multiply-add) four lanes at a
/// time, then reduces the accumulator with `vaddvq_f32`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");

    let len = a.len();
    if len == 0 {
        return 0.0;
    }

    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let va = unsafe { vld1q_f32(pa.add(i * 4)) };
        let vb = unsafe { vld1q_f32(pb.add(i * 4)) };
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut dot = vaddvq_f32(acc);

    for i in 0..remainder {
        dot += unsafe { *pa.add(chunks * 4 + i) * *pb.add(chunks * 4 + i) };
    }

    dot
}

/// L2 norm (Euclidean length) of an `f32` slice using NEON.
///
/// Computes `sqrt(sum of squares)` with NEON-accelerated accumulation.
/// Returns `0.0` for empty slices.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_l2_norm_f32(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        acc = vfmaq_f32(acc, v, v);
    }

    let mut sum_sq = vaddvq_f32(acc);

    for i in 0..remainder {
        let val = unsafe { *ptr.add(chunks * 4 + i) };
        sum_sq += val * val;
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn test_sum_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(result, 15.0), "expected 15.0, got {result}");
    }

    #[test]
    fn test_sum_single() {
        let data = [42.0f32];
        let result = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(result, 42.0), "expected 42.0, got {result}");
    }

    #[test]
    fn test_sum_empty() {
        let data: [f32; 0] = [];
        let result = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(result, 0.0), "expected 0.0, got {result}");
    }

    #[test]
    fn test_max_basic() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(result, 5.0), "expected 5.0, got {result}");
    }

    #[test]
    fn test_max_negative_values() {
        let data = [-3.0f32, -1.0, -4.0, -2.0];
        let result = unsafe { neon_max_f32(&data) };
        assert!(approx_eq(result, -1.0), "expected -1.0, got {result}");
    }

    #[test]
    fn test_argmax_basic() {
        let data = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { neon_argmax_f32(&data) };
        assert_eq!(result, 1, "expected index 1, got {result}");
    }

    #[test]
    fn test_argmax_multiple_max() {
        let data = [3.0f32, 1.0, 3.0, 2.0];
        let result = unsafe { neon_argmax_f32(&data) };
        assert_eq!(result, 0, "expected first occurrence at index 0, got {result}");
    }

    #[test]
    fn test_dot_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5+8+9+8+5 = 35
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert!(approx_eq(result, 35.0), "expected 35.0, got {result}");
    }

    #[test]
    fn test_l2_norm() {
        let data = [3.0f32, 4.0];
        // sqrt(9 + 16) = sqrt(25) = 5.0
        let result = unsafe { neon_l2_norm_f32(&data) };
        assert!(approx_eq(result, 5.0), "expected 5.0, got {result}");
    }

    #[test]
    fn test_large_reduction() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let expected_sum: f32 = (0..1024).map(|i| i as f32).sum();
        let result = unsafe { neon_sum_f32(&data) };
        assert!(approx_eq(result, expected_sum), "expected {expected_sum}, got {result}");
    }
}
