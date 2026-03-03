#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::manual_abs_diff,
    clippy::manual_contains,
    clippy::manual_is_multiple_of,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]
//! ARM NEON reduction v2 operations for Apple Silicon (aarch64).
//!
//! Provides NEON-optimized horizontal sum, max, min, mean, argmax, and
//! variance for `f32` slices. Each operation has an `unsafe` NEON path,
//! a scalar fallback, and a public dispatcher that selects at runtime.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// sum_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated horizontal sum using `vpaddq_f32` pairwise add.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_f32(input: &[f32]) -> f32 {
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
        acc = vaddq_f32(acc, v);
    }

    // Pairwise reduction: [a,b,c,d] -> [a+b, c+d, a+b, c+d] -> scalar
    let pair = vpaddq_f32(acc, acc);
    let mut total = vgetq_lane_f32(pair, 0) + vgetq_lane_f32(pair, 1);

    for i in 0..remainder {
        total += *ptr.add(chunks * 4 + i);
    }

    total
}

fn scalar_sum_f32(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// Horizontal sum of an `f32` slice.
///
/// Uses NEON pairwise add on aarch64, scalar fallback otherwise.
/// Returns `0.0` for empty slices.
pub fn sum_f32(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_sum_f32(input) };
        }
    }
    scalar_sum_f32(input)
}

// ---------------------------------------------------------------------------
// max_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated horizontal max using `vmaxq_f32` tree reduction.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max_f32(input: &[f32]) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    let ptr = input.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vmaxq_f32(acc, v);
    }

    // Tree reduction via pairwise max
    let pair = vpmaxq_f32(acc, acc);
    let mut max_val = f32::max(vgetq_lane_f32(pair, 0), vgetq_lane_f32(pair, 1));

    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        if val > max_val {
            max_val = val;
        }
    }

    max_val
}

fn scalar_max_f32(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Horizontal max of an `f32` slice.
///
/// Uses `vmaxq_f32` tree reduction on aarch64, scalar fallback otherwise.
/// Returns `f32::NEG_INFINITY` for empty slices.
pub fn max_f32(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_max_f32(input) };
        }
    }
    scalar_max_f32(input)
}

// ---------------------------------------------------------------------------
// min_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated horizontal min using `vminq_f32` tree reduction.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_min_f32(input: &[f32]) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::INFINITY;
    }

    let ptr = input.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(f32::INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vminq_f32(acc, v);
    }

    // Tree reduction via pairwise min
    let pair = vpminq_f32(acc, acc);
    let mut min_val = f32::min(vgetq_lane_f32(pair, 0), vgetq_lane_f32(pair, 1));

    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        if val < min_val {
            min_val = val;
        }
    }

    min_val
}

fn scalar_min_f32(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Horizontal min of an `f32` slice.
///
/// Uses `vminq_f32` tree reduction on aarch64, scalar fallback otherwise.
/// Returns `f32::INFINITY` for empty slices.
pub fn min_f32(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_min_f32(input) };
        }
    }
    scalar_min_f32(input)
}

// ---------------------------------------------------------------------------
// mean_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated mean (sum / len).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    neon_sum_f32(input) / input.len() as f32
}

fn scalar_mean_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    scalar_sum_f32(input) / input.len() as f32
}

/// Mean of an `f32` slice.
///
/// Uses NEON-accelerated sum on aarch64, scalar fallback otherwise.
/// Returns `0.0` for empty slices.
pub fn mean_f32(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_mean_f32(input) };
        }
    }
    scalar_mean_f32(input)
}

// ---------------------------------------------------------------------------
// argmax_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated argmax using `vcgtq_f32` comparisons.
///
/// Processes four elements at a time, using NEON comparisons to track
/// the running maximum and its index. Returns the index of the first
/// occurrence of the maximum value.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_argmax_f32(input: &[f32]) -> usize {
    let len = input.len();
    if len == 0 {
        return 0;
    }

    let ptr = input.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    // Track the best value and index across NEON lanes
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx: usize = 0;

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let base = i * 4;

        // Compare each lane against the current best
        let best_vec = vdupq_n_f32(best_val);
        let mask = vcgtq_f32(v, best_vec);

        // Extract lanes and check which beat the current best
        let mask_bits: [u32; 4] = [
            vgetq_lane_u32(mask, 0),
            vgetq_lane_u32(mask, 1),
            vgetq_lane_u32(mask, 2),
            vgetq_lane_u32(mask, 3),
        ];
        let vals: [f32; 4] = [
            vgetq_lane_f32(v, 0),
            vgetq_lane_f32(v, 1),
            vgetq_lane_f32(v, 2),
            vgetq_lane_f32(v, 3),
        ];

        for j in 0..4 {
            if mask_bits[j] != 0 && vals[j] > best_val {
                best_val = vals[j];
                best_idx = base + j;
            }
        }
    }

    // Handle remainder
    for i in 0..remainder {
        let val = *ptr.add(chunks * 4 + i);
        if val > best_val {
            best_val = val;
            best_idx = chunks * 4 + i;
        }
    }

    best_idx
}

fn scalar_argmax_f32(input: &[f32]) -> usize {
    if input.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in input.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Index of the maximum value in an `f32` slice.
///
/// Uses NEON `vcgtq_f32` comparisons on aarch64, scalar fallback otherwise.
/// Returns `0` for empty slices. First occurrence wins on ties.
pub fn argmax_f32(input: &[f32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_argmax_f32(input) };
        }
    }
    scalar_argmax_f32(input)
}

// ---------------------------------------------------------------------------
// variance_f32
// ---------------------------------------------------------------------------

/// NEON-accelerated population variance using two-pass algorithm.
///
/// Pass 1: compute mean via NEON sum. Pass 2: accumulate squared
/// deviations using `vfmaq_f32` (fused multiply-add).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_variance_f32(input: &[f32]) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    // Pass 1: mean
    let mean = neon_mean_f32(input);

    // Pass 2: sum of squared deviations via FMA
    let ptr = input.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;
    let mean_vec = vdupq_n_f32(mean);
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let diff = vsubq_f32(v, mean_vec);
        acc = vfmaq_f32(acc, diff, diff); // acc += diff * diff
    }

    // Reduce accumulator with pairwise add
    let pair = vpaddq_f32(acc, acc);
    let mut sum_sq = vgetq_lane_f32(pair, 0) + vgetq_lane_f32(pair, 1);

    for i in 0..remainder {
        let diff = *ptr.add(chunks * 4 + i) - mean;
        sum_sq += diff * diff;
    }

    sum_sq / len as f32
}

fn scalar_variance_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let mean = scalar_mean_f32(input);
    let sum_sq: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum();
    sum_sq / input.len() as f32
}

/// Population variance of an `f32` slice.
///
/// Two-pass algorithm: compute mean, then accumulate squared deviations
/// with NEON `vfmaq_f32` on aarch64, scalar fallback otherwise.
/// Returns `0.0` for empty slices.
pub fn variance_f32(input: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_variance_f32(input) };
        }
    }
    scalar_variance_f32(input)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // -----------------------------------------------------------------------
    // sum_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_empty() {
        assert_eq!(sum_f32(&[]), 0.0);
    }

    #[test]
    fn test_sum_single() {
        assert!(approx_eq(sum_f32(&[42.0]), 42.0));
    }

    #[test]
    fn test_sum_two() {
        assert!(approx_eq(sum_f32(&[1.0, 2.0]), 3.0));
    }

    #[test]
    fn test_sum_three() {
        assert!(approx_eq(sum_f32(&[1.0, 2.0, 3.0]), 6.0));
    }

    #[test]
    fn test_sum_exact_chunk() {
        assert!(approx_eq(sum_f32(&[1.0, 2.0, 3.0, 4.0]), 10.0));
    }

    #[test]
    fn test_sum_with_remainder() {
        assert!(approx_eq(sum_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]), 15.0));
    }

    #[test]
    fn test_sum_negatives() {
        assert!(approx_eq(sum_f32(&[-1.0, -2.0, -3.0]), -6.0));
    }

    #[test]
    fn test_sum_mixed_sign() {
        assert!(approx_eq(sum_f32(&[-5.0, 3.0, 2.0]), 0.0));
    }

    #[test]
    fn test_sum_zeros() {
        assert_eq!(sum_f32(&[0.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_sum_large_array() {
        let data: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        let expected: f32 = (1..=1024).map(|i| i as f32).sum();
        let result = sum_f32(&data);
        assert!((result - expected).abs() < 1.0, "expected ~{expected}, got {result}");
    }

    #[test]
    fn test_sum_eight_elements() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(approx_eq(sum_f32(&data), 36.0));
    }

    // -----------------------------------------------------------------------
    // max_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_empty() {
        assert_eq!(max_f32(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn test_max_single() {
        assert!(approx_eq(max_f32(&[7.0]), 7.0));
    }

    #[test]
    fn test_max_two() {
        assert!(approx_eq(max_f32(&[3.0, 5.0]), 5.0));
    }

    #[test]
    fn test_max_three() {
        assert!(approx_eq(max_f32(&[1.0, 9.0, 3.0]), 9.0));
    }

    #[test]
    fn test_max_exact_chunk() {
        assert!(approx_eq(max_f32(&[1.0, 5.0, 3.0, 2.0]), 5.0));
    }

    #[test]
    fn test_max_with_remainder() {
        assert!(approx_eq(max_f32(&[1.0, 5.0, 3.0, 2.0, 4.0]), 5.0));
    }

    #[test]
    fn test_max_all_negative() {
        assert!(approx_eq(max_f32(&[-3.0, -1.0, -4.0, -2.0]), -1.0));
    }

    #[test]
    fn test_max_mixed_sign() {
        assert!(approx_eq(max_f32(&[-5.0, 0.0, 3.0, -1.0, 2.0]), 3.0));
    }

    #[test]
    fn test_max_duplicates() {
        assert!(approx_eq(max_f32(&[5.0, 5.0, 5.0, 5.0]), 5.0));
    }

    #[test]
    fn test_max_large_array() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        assert!(approx_eq(max_f32(&data), 1023.0));
    }

    #[test]
    fn test_max_last_element_is_max() {
        assert!(approx_eq(max_f32(&[1.0, 2.0, 3.0, 4.0, 99.0]), 99.0));
    }

    // -----------------------------------------------------------------------
    // min_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_empty() {
        assert_eq!(min_f32(&[]), f32::INFINITY);
    }

    #[test]
    fn test_min_single() {
        assert!(approx_eq(min_f32(&[7.0]), 7.0));
    }

    #[test]
    fn test_min_two() {
        assert!(approx_eq(min_f32(&[3.0, 5.0]), 3.0));
    }

    #[test]
    fn test_min_three() {
        assert!(approx_eq(min_f32(&[9.0, 1.0, 3.0]), 1.0));
    }

    #[test]
    fn test_min_exact_chunk() {
        assert!(approx_eq(min_f32(&[4.0, 2.0, 5.0, 3.0]), 2.0));
    }

    #[test]
    fn test_min_with_remainder() {
        assert!(approx_eq(min_f32(&[4.0, 2.0, 5.0, 3.0, 1.0]), 1.0));
    }

    #[test]
    fn test_min_all_negative() {
        assert!(approx_eq(min_f32(&[-3.0, -1.0, -4.0, -2.0]), -4.0));
    }

    #[test]
    fn test_min_mixed_sign() {
        assert!(approx_eq(min_f32(&[-5.0, 0.0, 3.0, -1.0, 2.0]), -5.0));
    }

    #[test]
    fn test_min_duplicates() {
        assert!(approx_eq(min_f32(&[2.0, 2.0, 2.0, 2.0]), 2.0));
    }

    #[test]
    fn test_min_large_array() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        assert!(approx_eq(min_f32(&data), 0.0));
    }

    #[test]
    fn test_min_last_element_is_min() {
        assert!(approx_eq(min_f32(&[9.0, 8.0, 7.0, 6.0, -1.0]), -1.0));
    }

    // -----------------------------------------------------------------------
    // mean_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean_f32(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert!(approx_eq(mean_f32(&[5.0]), 5.0));
    }

    #[test]
    fn test_mean_two() {
        assert!(approx_eq(mean_f32(&[2.0, 4.0]), 3.0));
    }

    #[test]
    fn test_mean_four() {
        assert!(approx_eq(mean_f32(&[1.0, 2.0, 3.0, 4.0]), 2.5));
    }

    #[test]
    fn test_mean_five() {
        assert!(approx_eq(mean_f32(&[10.0, 20.0, 30.0, 40.0, 50.0]), 30.0));
    }

    #[test]
    fn test_mean_negative() {
        assert!(approx_eq(mean_f32(&[-2.0, -4.0, -6.0]), -4.0));
    }

    #[test]
    fn test_mean_mixed() {
        assert!(approx_eq(mean_f32(&[-1.0, 1.0]), 0.0));
    }

    #[test]
    fn test_mean_identical() {
        assert!(approx_eq(mean_f32(&[7.0, 7.0, 7.0, 7.0, 7.0]), 7.0));
    }

    #[test]
    fn test_mean_large_array() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert!(approx_eq(mean_f32(&data), 50.5));
    }

    // -----------------------------------------------------------------------
    // argmax_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_argmax_empty() {
        assert_eq!(argmax_f32(&[]), 0);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax_f32(&[42.0]), 0);
    }

    #[test]
    fn test_argmax_two() {
        assert_eq!(argmax_f32(&[1.0, 5.0]), 1);
    }

    #[test]
    fn test_argmax_three() {
        assert_eq!(argmax_f32(&[1.0, 9.0, 3.0]), 1);
    }

    #[test]
    fn test_argmax_exact_chunk() {
        assert_eq!(argmax_f32(&[1.0, 2.0, 9.0, 4.0]), 2);
    }

    #[test]
    fn test_argmax_with_remainder() {
        assert_eq!(argmax_f32(&[1.0, 2.0, 3.0, 4.0, 99.0]), 4);
    }

    #[test]
    fn test_argmax_first_is_max() {
        assert_eq!(argmax_f32(&[100.0, 1.0, 2.0, 3.0, 4.0]), 0);
    }

    #[test]
    fn test_argmax_ties_first_wins() {
        assert_eq!(argmax_f32(&[5.0, 3.0, 5.0, 2.0]), 0);
    }

    #[test]
    fn test_argmax_ties_across_chunks() {
        assert_eq!(argmax_f32(&[5.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_argmax_all_negative() {
        assert_eq!(argmax_f32(&[-10.0, -3.0, -7.0, -1.0, -5.0]), 3);
    }

    #[test]
    fn test_argmax_large_array() {
        let mut data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        data[512] = 9999.0;
        assert_eq!(argmax_f32(&data), 512);
    }

    #[test]
    fn test_argmax_max_in_second_chunk() {
        assert_eq!(argmax_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 99.0, 7.0, 8.0]), 5);
    }

    // -----------------------------------------------------------------------
    // variance_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_variance_empty() {
        assert_eq!(variance_f32(&[]), 0.0);
    }

    #[test]
    fn test_variance_single() {
        assert!(approx_eq(variance_f32(&[5.0]), 0.0));
    }

    #[test]
    fn test_variance_identical() {
        assert!(approx_eq(variance_f32(&[3.0, 3.0, 3.0, 3.0]), 0.0));
    }

    #[test]
    fn test_variance_simple() {
        // [1,2,3,4] mean=2.5, deviations=[-1.5,-0.5,0.5,1.5], sq=[2.25,0.25,0.25,2.25], var=5/4=1.25
        assert!(approx_eq(variance_f32(&[1.0, 2.0, 3.0, 4.0]), 1.25));
    }

    #[test]
    fn test_variance_two_elements() {
        // [0, 10] mean=5, sq_dev=[25,25], var=25
        assert!(approx_eq(variance_f32(&[0.0, 10.0]), 25.0));
    }

    #[test]
    fn test_variance_symmetric() {
        // [-1, 1] mean=0, sq_dev=[1,1], var=1
        assert!(approx_eq(variance_f32(&[-1.0, 1.0]), 1.0));
    }

    #[test]
    fn test_variance_five_elements() {
        // [2,4,4,4,5,5,7,9] mean=5, deviations=[-3,-1,-1,-1,0,0,2,4]
        // sq=[9,1,1,1,0,0,4,16] = 32/8 = 4
        assert!(approx_eq(variance_f32(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 4.0));
    }

    #[test]
    fn test_variance_negative() {
        // [-2, -4, -6] mean=-4, dev=[2,0,-2], sq=[4,0,4], var=8/3
        let expected = 8.0 / 3.0;
        assert!(approx_eq(variance_f32(&[-2.0, -4.0, -6.0]), expected));
    }

    #[test]
    fn test_variance_large_values() {
        let data = [1000.0, 1001.0, 1002.0, 1003.0];
        // Same as [0,1,2,3] variance = 1.25
        assert!(approx_eq(variance_f32(&data), 1.25));
    }

    #[test]
    fn test_variance_large_array() {
        let n = 1000;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // Var of 0..n-1 = (n^2 - 1) / 12
        let expected = (n as f32 * n as f32 - 1.0) / 12.0;
        let result = variance_f32(&data);
        assert!((result - expected).abs() < 1.0, "expected ~{expected}, got {result}");
    }

    // -----------------------------------------------------------------------
    // Cross-function consistency tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_equals_sum_div_len() {
        let data = [1.5, 2.5, 3.5, 4.5, 5.5];
        let s = sum_f32(&data);
        let m = mean_f32(&data);
        assert!(approx_eq(m, s / data.len() as f32));
    }

    #[test]
    fn test_max_at_argmax() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let idx = argmax_f32(&data);
        let mx = max_f32(&data);
        assert!(approx_eq(data[idx], mx));
    }

    #[test]
    fn test_min_le_max() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert!(min_f32(&data) <= max_f32(&data));
    }

    #[test]
    fn test_variance_zero_for_constant() {
        let data = [42.0; 16];
        assert!(approx_eq(variance_f32(&data), 0.0));
    }

    #[test]
    fn test_scalar_neon_sum_agree() {
        let data: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
        let s = scalar_sum_f32(&data);
        let d = sum_f32(&data);
        assert!(approx_eq(s, d));
    }

    #[test]
    fn test_scalar_neon_max_agree() {
        let data = [-2.0, 0.5, 3.0, -1.0, 2.0, 4.0, -0.5];
        let s = scalar_max_f32(&data);
        let d = max_f32(&data);
        assert!(approx_eq(s, d));
    }

    #[test]
    fn test_scalar_neon_min_agree() {
        let data = [-2.0, 0.5, 3.0, -1.0, 2.0, 4.0, -0.5];
        let s = scalar_min_f32(&data);
        let d = min_f32(&data);
        assert!(approx_eq(s, d));
    }

    #[test]
    fn test_scalar_neon_mean_agree() {
        let data: Vec<f32> = (1..=33).map(|i| i as f32).collect();
        let s = scalar_mean_f32(&data);
        let d = mean_f32(&data);
        assert!(approx_eq(s, d));
    }

    #[test]
    fn test_scalar_neon_argmax_agree() {
        let data = [1.0, 9.0, 3.0, 7.0, 5.0, 8.0, 2.0];
        let s = scalar_argmax_f32(&data);
        let d = argmax_f32(&data);
        assert_eq!(s, d);
    }

    #[test]
    fn test_scalar_neon_variance_agree() {
        let data: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let s = scalar_variance_f32(&data);
        let d = variance_f32(&data);
        assert!((s - d).abs() < 0.01, "scalar={s}, dispatcher={d}");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_very_small_values() {
        let data = [1e-10_f32; 8];
        assert!(sum_f32(&data) > 0.0);
    }

    #[test]
    fn test_max_single_negative() {
        assert!(approx_eq(max_f32(&[-99.0]), -99.0));
    }

    #[test]
    fn test_min_single_positive() {
        assert!(approx_eq(min_f32(&[99.0]), 99.0));
    }

    #[test]
    fn test_argmax_descending() {
        assert_eq!(argmax_f32(&[5.0, 4.0, 3.0, 2.0, 1.0]), 0);
    }

    #[test]
    fn test_argmax_ascending() {
        assert_eq!(argmax_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]), 4);
    }

    #[test]
    fn test_variance_high_spread() {
        // [-100, 100] mean=0, var=10000
        assert!(approx_eq(variance_f32(&[-100.0, 100.0]), 10000.0));
    }

    #[test]
    fn test_min_max_same_for_single() {
        let data = [3.14];
        assert!(approx_eq(min_f32(&data), max_f32(&data)));
    }

    #[test]
    fn test_sum_alternating() {
        let data = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert!(approx_eq(sum_f32(&data), 0.0));
    }

    #[test]
    fn test_mean_three_elements() {
        assert!(approx_eq(mean_f32(&[3.0, 6.0, 9.0]), 6.0));
    }

    #[test]
    fn test_variance_three_elements() {
        // [3,6,9] mean=6, dev=[-3,0,3], sq=[9,0,9], var=18/3=6
        assert!(approx_eq(variance_f32(&[3.0, 6.0, 9.0]), 6.0));
    }
}
