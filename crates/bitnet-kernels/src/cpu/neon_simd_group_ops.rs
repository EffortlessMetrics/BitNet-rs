#![cfg(target_arch = "aarch64")]
//! ARM NEON SIMD group operations for Apple Silicon.
//!
//! Common parallel reduction and scan patterns used in transformer inference:
//! prefix sums, segmented reductions, broadcast-reduce, fused scale-sum,
//! horizontal adds, pairwise max, and masked sums.
//!
//! Every function is `#[target_feature(enable = "neon")]` and requires
//! AArch64.  The crate-level `#![cfg(target_arch = "aarch64")]` prevents
//! compilation on other architectures entirely.

use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Horizontal sum of one `float32x4_t` register.
///
/// Uses `vaddvq_f32` (single-instruction on ARMv8.1+).
///
/// # Safety
///
/// Requires NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Inclusive prefix sum (scan) of `data` using NEON.
///
/// For each position `i` the output satisfies `out[i] = data[0] + … + data[i]`.
///
/// The implementation processes 4-wide NEON lanes for each chunk and carries
/// the running total across chunks.  Tail elements are handled with scalar
/// arithmetic.
///
/// # Safety
///
/// Requires AArch64 with NEON (always available on Apple Silicon).
#[target_feature(enable = "neon")]
pub unsafe fn parallel_prefix_sum_neon(data: &[f32]) -> Vec<f32> {
    let len = data.len();
    let mut out = Vec::with_capacity(len);
    if len == 0 {
        return out;
    }

    let ptr = data.as_ptr();
    let mut running: f32 = 0.0;
    let chunks = len / 4;
    let remainder = len % 4;

    for c in 0..chunks {
        let base = c * 4;
        // Load 4 elements.
        let v = vld1q_f32(ptr.add(base));

        // Sequential prefix sum within the lane (4 elements only — the
        // optimal Blelloch-style parallel scan doesn't pay off at width 4).
        let a0 = vgetq_lane_f32::<0>(v);
        let a1 = vgetq_lane_f32::<1>(v);
        let a2 = vgetq_lane_f32::<2>(v);
        let a3 = vgetq_lane_f32::<3>(v);

        let s0 = running + a0;
        let s1 = s0 + a1;
        let s2 = s1 + a2;
        let s3 = s2 + a3;

        // Store via a temporary array to keep it simple and correct.
        out.push(s0);
        out.push(s1);
        out.push(s2);
        out.push(s3);

        running = s3;
    }

    // Scalar tail.
    for i in 0..remainder {
        running += *ptr.add(chunks * 4 + i);
        out.push(running);
    }

    out
}

/// Max reduction over fixed-size segments.
///
/// `data.len()` must be a multiple of `segment_len`.  Returns a vector of
/// length `data.len() / segment_len` where each element is the maximum of the
/// corresponding segment.  This is the pattern used to find per-head maximum
/// attention scores.
///
/// # Panics
///
/// Panics if `segment_len == 0` or `data.len() % segment_len != 0`.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn segmented_reduce_max_neon(data: &[f32], segment_len: usize) -> Vec<f32> {
    assert!(segment_len > 0, "segment_len must be > 0");
    assert!(data.len() % segment_len == 0, "data length must be a multiple of segment_len");

    let num_segments = data.len() / segment_len;
    let mut out = Vec::with_capacity(num_segments);
    let ptr = data.as_ptr();

    for seg in 0..num_segments {
        let base = seg * segment_len;
        let chunks = segment_len / 4;
        let remainder = segment_len % 4;

        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vmaxq_f32(acc, v);
        }

        let mut max_val = vmaxvq_f32(acc);

        for r in 0..remainder {
            let val = *ptr.add(base + chunks * 4 + r);
            if val > max_val {
                max_val = val;
            }
        }

        out.push(max_val);
    }

    out
}

/// Sum reduction over fixed-size segments.
///
/// `data.len()` must be a multiple of `segment_len`.  Returns a vector of
/// length `data.len() / segment_len` where each element is the sum of the
/// corresponding segment.
///
/// # Panics
///
/// Panics if `segment_len == 0` or `data.len() % segment_len != 0`.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn segmented_reduce_sum_neon(data: &[f32], segment_len: usize) -> Vec<f32> {
    assert!(segment_len > 0, "segment_len must be > 0");
    assert!(data.len() % segment_len == 0, "data length must be a multiple of segment_len");

    let num_segments = data.len() / segment_len;
    let mut out = Vec::with_capacity(num_segments);
    let ptr = data.as_ptr();

    for seg in 0..num_segments {
        let base = seg * segment_len;
        let chunks = segment_len / 4;
        let remainder = segment_len % 4;

        let mut acc = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);

        for r in 0..remainder {
            sum += *ptr.add(base + chunks * 4 + r);
        }

        out.push(sum);
    }

    out
}

/// Reduce `data` to a single sum, then broadcast to every position.
///
/// Returns a `Vec<f32>` of `data.len()` elements all equal to the total sum.
/// Useful for normalisation denominators in softmax / layer-norm.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn broadcast_reduce_neon(data: &[f32]) -> Vec<f32> {
    let len = data.len();
    if len == 0 {
        return Vec::new();
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for c in 0..chunks {
        let v = vld1q_f32(ptr.add(c * 4));
        acc = vaddq_f32(acc, v);
    }

    let mut total = vaddvq_f32(acc);
    for r in 0..remainder {
        total += *ptr.add(chunks * 4 + r);
    }

    vec![total; len]
}

/// Scale every element by `scale` and return the total sum.
///
/// Equivalent to `data.iter().map(|x| x * scale).sum()` but uses NEON
/// `vmulq_f32` + `vaddq_f32` with a single horizontal reduce at the end.
/// Common in scaled-dot-product attention (`sum(Q·K / √d)`).
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn fused_scale_sum_neon(data: &[f32], scale: f32) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let scale_v = vdupq_n_f32(scale);
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    for c in 0..chunks {
        let v = vld1q_f32(ptr.add(c * 4));
        let scaled = vmulq_f32(v, scale_v);
        acc = vaddq_f32(acc, scaled);
    }

    let mut sum = vaddvq_f32(acc);

    for r in 0..remainder {
        sum += *ptr.add(chunks * 4 + r) * scale;
    }

    sum
}

/// Horizontal sum of exactly 4 `f32` values loaded from `data`.
///
/// This is a thin convenience wrapper around a single `vld1q_f32` +
/// `vaddvq_f32` pair.
///
/// # Panics
///
/// Panics if `data.len() < 4`.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn horizontal_add_f32x4(data: &[f32]) -> f32 {
    assert!(data.len() >= 4, "need at least 4 elements");
    let v = vld1q_f32(data.as_ptr());
    hsum_f32x4(v)
}

/// Element-wise maximum of two equal-length `f32` slices.
///
/// Returns `out[i] = max(a[i], b[i])`.  Used in multi-head attention to merge
/// per-head maxima before softmax.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn pairwise_max_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "slices must have equal length");

    let len = a.len();
    let mut out = Vec::with_capacity(len);
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    // Pre-allocate so we can write via pointer.
    out.set_len(len);
    let po: *mut f32 = out.as_mut_ptr();

    for c in 0..chunks {
        let offset = c * 4;
        let va = vld1q_f32(pa.add(offset));
        let vb = vld1q_f32(pb.add(offset));
        let vm = vmaxq_f32(va, vb);
        vst1q_f32(po.add(offset), vm);
    }

    for r in 0..remainder {
        let idx = chunks * 4 + r;
        let va = *pa.add(idx);
        let vb = *pb.add(idx);
        *po.add(idx) = if va > vb { va } else { vb };
    }

    out
}

/// Sum of `data[i]` where `mask[i]` is `true`.
///
/// Processes four elements at a time: the boolean mask is widened to a NEON
/// bit-mask via `vld1q_u32` and used with `vbslq_f32` to zero-out masked
/// positions before accumulation.
///
/// # Panics
///
/// Panics if `data.len() != mask.len()`.
///
/// # Safety
///
/// Requires AArch64 NEON.
#[target_feature(enable = "neon")]
pub unsafe fn masked_sum_neon(data: &[f32], mask: &[bool]) -> f32 {
    assert_eq!(data.len(), mask.len(), "data and mask must have equal length");

    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    let ptr = data.as_ptr();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = vdupq_n_f32(0.0);
    let zero = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let offset = c * 4;
        let v = vld1q_f32(ptr.add(offset));

        // Build a 32-bit mask from four bools: 0xFFFF_FFFF if true, else 0.
        let m0 = if mask[offset] { !0u32 } else { 0u32 };
        let m1 = if mask[offset + 1] { !0u32 } else { 0u32 };
        let m2 = if mask[offset + 2] { !0u32 } else { 0u32 };
        let m3 = if mask[offset + 3] { !0u32 } else { 0u32 };

        let mask_arr: [u32; 4] = [m0, m1, m2, m3];
        let mask_v = vld1q_u32(mask_arr.as_ptr());

        // Select: true-lane → v, false-lane → 0.0
        let selected = vbslq_f32(mask_v, v, zero);
        acc = vaddq_f32(acc, selected);
    }

    let mut sum = vaddvq_f32(acc);

    for r in 0..remainder {
        let idx = chunks * 4 + r;
        if mask[idx] {
            sum += *ptr.add(idx);
        }
    }

    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // -- prefix sum ----------------------------------------------------------

    #[test]
    fn test_prefix_sum_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { parallel_prefix_sum_neon(&data) };
        let expected: Vec<f32> = vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b), "{a} != {b}");
        }
    }

    #[test]
    fn test_prefix_sum_empty() {
        let result = unsafe { parallel_prefix_sum_neon(&[]) };
        assert!(result.is_empty());
    }

    #[test]
    fn test_prefix_sum_non_aligned() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { parallel_prefix_sum_neon(&data) };
        let expected: Vec<f32> = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b), "{a} != {b}");
        }
    }

    // -- segmented reduce max ------------------------------------------------

    #[test]
    fn test_segmented_reduce_max() {
        // Two segments of 4.
        let data = [1.0f32, 5.0, 3.0, 2.0, 9.0, 0.0, 7.0, 8.0];
        let result = unsafe { segmented_reduce_max_neon(&data, 4) };
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 5.0));
        assert!(approx_eq(result[1], 9.0));
    }

    // -- segmented reduce sum ------------------------------------------------

    #[test]
    fn test_segmented_reduce_sum() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let result = unsafe { segmented_reduce_sum_neon(&data, 4) };
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 10.0));
        assert!(approx_eq(result[1], 100.0));
    }

    // -- broadcast reduce ----------------------------------------------------

    #[test]
    fn test_broadcast_reduce() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { broadcast_reduce_neon(&data) };
        assert_eq!(result.len(), data.len());
        for v in &result {
            assert!(approx_eq(*v, 15.0));
        }
    }

    // -- fused scale sum -----------------------------------------------------

    #[test]
    fn test_fused_scale_sum() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { fused_scale_sum_neon(&data, 2.0) };
        // (1+2+3+4+5)*2 = 30
        assert!(approx_eq(result, 30.0));
    }

    // -- horizontal add f32x4 -----------------------------------------------

    #[test]
    fn test_horizontal_add_f32x4() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = unsafe { horizontal_add_f32x4(&data) };
        assert!(approx_eq(result, 10.0));
    }

    // -- pairwise max --------------------------------------------------------

    #[test]
    fn test_pairwise_max() {
        let a = [1.0f32, 5.0, 3.0, 7.0, 2.0];
        let b = [4.0f32, 2.0, 6.0, 0.0, 9.0];
        let result = unsafe { pairwise_max_neon(&a, &b) };
        let expected = [4.0f32, 5.0, 6.0, 7.0, 9.0];
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b), "{a} != {b}");
        }
    }

    // -- masked sum ----------------------------------------------------------

    #[test]
    fn test_masked_sum() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = [true, false, true, false, true, false, true, false, true];
        let result = unsafe { masked_sum_neon(&data, &mask) };
        // 1 + 3 + 5 + 7 + 9 = 25
        assert!(approx_eq(result, 25.0));
    }

    #[test]
    fn test_masked_sum_all_false() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [false, false, false, false];
        let result = unsafe { masked_sum_neon(&data, &mask) };
        assert!(approx_eq(result, 0.0));
    }

    #[test]
    fn test_masked_sum_all_true() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [true, true, true, true];
        let result = unsafe { masked_sum_neon(&data, &mask) };
        assert!(approx_eq(result, 10.0));
    }
}
