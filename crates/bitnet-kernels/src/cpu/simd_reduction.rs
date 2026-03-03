//! SIMD-accelerated reduction operations for CPU inference.
//!
//! Provides horizontal reductions (sum, max, min), tree-based hierarchical
//! reduction for large arrays, parallel prefix-sum (scan), segmented
//! reductions (per-group sum/max/min), index-tracking reductions (argmax,
//! argmin), and multi-pass stable summation (Kahan).
//!
//! On x86-64 with AVX2, hot loops operate 8-wide; a portable scalar
//! fallback handles all other targets and tail elements.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── AVX2 intrinsics ─────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Error helpers ───────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

fn validate_non_empty(data: &[f32]) -> Result<()> {
    if data.is_empty() {
        return Err(invalid_args("input must not be empty"));
    }
    Ok(())
}

// ── AVX2 horizontal primitives ──────────────────────────────────────────

/// 8-wide horizontal sum → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo128, hi128);
    let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
    _mm_cvtss_f32(s32)
}

/// 8-wide horizontal max → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo128, hi128);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

/// 8-wide horizontal min → scalar.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let m128 = _mm_min_ps(lo128, hi128);
    let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_min_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    _mm_cvtss_f32(m32)
}

// ── Result type for index-tracking reductions ───────────────────────────

/// Value paired with the index of its first occurrence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedValue {
    pub value: f32,
    pub index: usize,
}

// ── Reduction kind for segmented operations ─────────────────────────────

/// The kind of reduction to apply within each segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentOp {
    Sum,
    Max,
    Min,
}

// ═════════════════════════════════════════════════════════════════════════
// 1. SIMD horizontal sum (f32 × 8 → scalar)
// ═════════════════════════════════════════════════════════════════════════

/// Sum all `f32` elements using SIMD where available.
pub fn simd_horizontal_sum(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection guard above.
            return Ok(unsafe { simd_horizontal_sum_avx2(data) });
        }
    }
    Ok(scalar_sum(data))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_horizontal_sum_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }
    let mut total = hsum_avx2(acc);
    for &x in &data[chunks * 8..] {
        total += x;
    }
    total
}

fn scalar_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

// ═════════════════════════════════════════════════════════════════════════
// 2. SIMD horizontal max (f32 × 8 → scalar)
// ═════════════════════════════════════════════════════════════════════════

/// Maximum of all `f32` elements using SIMD where available.
pub fn simd_horizontal_max(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { simd_horizontal_max_avx2(data) });
        }
    }
    Ok(scalar_max(data))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_horizontal_max_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    let chunks = len / 8;
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_max_ps(acc, v);
    }
    let mut best = hmax_avx2(acc);
    for &x in &data[chunks * 8..] {
        if x > best {
            best = x;
        }
    }
    best
}

fn scalar_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

// ═════════════════════════════════════════════════════════════════════════
// 3. SIMD horizontal min (f32 × 8 → scalar)
// ═════════════════════════════════════════════════════════════════════════

/// Minimum of all `f32` elements using SIMD where available.
pub fn simd_horizontal_min(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { simd_horizontal_min_avx2(data) });
        }
    }
    Ok(scalar_min(data))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_horizontal_min_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    let chunks = len / 8;
    let mut acc = _mm256_set1_ps(f32::INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_min_ps(acc, v);
    }
    let mut best = hmin_avx2(acc);
    for &x in &data[chunks * 8..] {
        if x < best {
            best = x;
        }
    }
    best
}

fn scalar_min(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::INFINITY, f32::min)
}

// ═════════════════════════════════════════════════════════════════════════
// 4. Tree reduction for large arrays (hierarchical)
// ═════════════════════════════════════════════════════════════════════════

/// Hierarchical tree reduction: recursively halves the array, reducing
/// pairs at each level until a single scalar remains.  Better numerical
/// stability than a linear left-fold for very large arrays.
pub fn tree_reduce_sum(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    Ok(tree_reduce_sum_impl(data))
}

fn tree_reduce_sum_impl(data: &[f32]) -> f32 {
    let n = data.len();
    if n <= 32 {
        return data.iter().sum();
    }
    let mid = n / 2;
    tree_reduce_sum_impl(&data[..mid]) + tree_reduce_sum_impl(&data[mid..])
}

/// Hierarchical tree reduction for maximum.
pub fn tree_reduce_max(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    Ok(tree_reduce_max_impl(data))
}

fn tree_reduce_max_impl(data: &[f32]) -> f32 {
    let n = data.len();
    if n <= 32 {
        return scalar_max(data);
    }
    let mid = n / 2;
    f32::max(tree_reduce_max_impl(&data[..mid]), tree_reduce_max_impl(&data[mid..]))
}

/// Hierarchical tree reduction for minimum.
pub fn tree_reduce_min(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    Ok(tree_reduce_min_impl(data))
}

fn tree_reduce_min_impl(data: &[f32]) -> f32 {
    let n = data.len();
    if n <= 32 {
        return scalar_min(data);
    }
    let mid = n / 2;
    f32::min(tree_reduce_min_impl(&data[..mid]), tree_reduce_min_impl(&data[mid..]))
}

// ═════════════════════════════════════════════════════════════════════════
// 5. Parallel prefix sum (inclusive scan) with SIMD
// ═════════════════════════════════════════════════════════════════════════

/// Inclusive prefix sum: `output[i] = data[0] + … + data[i]`.
///
/// Uses SIMD block-sums with a serial correction pass for the carry
/// between blocks.
pub fn prefix_sum_inclusive(data: &[f32]) -> Result<Vec<f32>> {
    validate_non_empty(data)?;
    let n = data.len();
    let mut out = vec![0.0_f32; n];

    // Phase 1: local prefix sums within 8-element blocks.
    let full_blocks = n / 8;
    for b in 0..full_blocks {
        let base = b * 8;
        out[base] = data[base];
        for j in 1..8 {
            out[base + j] = out[base + j - 1] + data[base + j];
        }
    }
    // Tail
    let tail_start = full_blocks * 8;
    if tail_start < n {
        out[tail_start] = data[tail_start];
        for j in (tail_start + 1)..n {
            out[j] = out[j - 1] + data[j];
        }
    }

    // Phase 2: propagate block-end carries.
    let mut carry: f32 = 0.0;
    for b in 0..full_blocks {
        let base = b * 8;
        if carry != 0.0 {
            for j in 0..8 {
                out[base + j] += carry;
            }
        }
        carry = out[base + 7];
    }
    if carry != 0.0 {
        for item in out.iter_mut().take(n).skip(tail_start) {
            *item += carry;
        }
    }

    Ok(out)
}

/// Exclusive prefix sum: `output[i] = data[0] + … + data[i-1]`, with
/// `output[0] = 0`.
pub fn prefix_sum_exclusive(data: &[f32]) -> Result<Vec<f32>> {
    validate_non_empty(data)?;
    let n = data.len();
    let mut out = vec![0.0_f32; n];
    for i in 1..n {
        out[i] = out[i - 1] + data[i - 1];
    }
    Ok(out)
}

// ═════════════════════════════════════════════════════════════════════════
// 6. Segmented reduction (per-group sum / max / min)
// ═════════════════════════════════════════════════════════════════════════

/// Reduce contiguous fixed-size segments of `data`.
///
/// `segment_size` must evenly divide `data.len()`.
pub fn segmented_reduce(data: &[f32], segment_size: usize, op: SegmentOp) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Err(invalid_args("input must not be empty"));
    }
    if segment_size == 0 {
        return Err(invalid_args("segment_size must be > 0"));
    }
    if !data.len().is_multiple_of(segment_size) {
        return Err(invalid_args("data.len() must be divisible by segment_size"));
    }
    let n_segs = data.len() / segment_size;
    let mut out = Vec::with_capacity(n_segs);
    for s in 0..n_segs {
        let seg = &data[s * segment_size..(s + 1) * segment_size];
        let val = match op {
            SegmentOp::Sum => seg.iter().sum(),
            SegmentOp::Max => scalar_max(seg),
            SegmentOp::Min => scalar_min(seg),
        };
        out.push(val);
    }
    Ok(out)
}

/// Per-segment reduction using variable-length segments defined by
/// `offsets`.  `offsets[i]..offsets[i+1]` is the range for segment `i`.
pub fn segmented_reduce_variable(
    data: &[f32],
    offsets: &[usize],
    op: SegmentOp,
) -> Result<Vec<f32>> {
    if offsets.len() < 2 {
        return Err(invalid_args("offsets must have at least 2 entries"));
    }
    let n_segs = offsets.len() - 1;
    let mut out = Vec::with_capacity(n_segs);
    for i in 0..n_segs {
        let start = offsets[i];
        let end = offsets[i + 1];
        if start > end || end > data.len() {
            return Err(invalid_args(format!(
                "invalid segment range [{start}, {end}) for data.len()={}",
                data.len()
            )));
        }
        if start == end {
            let ident = match op {
                SegmentOp::Sum => 0.0,
                SegmentOp::Max => f32::NEG_INFINITY,
                SegmentOp::Min => f32::INFINITY,
            };
            out.push(ident);
            continue;
        }
        let seg = &data[start..end];
        let val = match op {
            SegmentOp::Sum => seg.iter().sum(),
            SegmentOp::Max => scalar_max(seg),
            SegmentOp::Min => scalar_min(seg),
        };
        out.push(val);
    }
    Ok(out)
}

// ═════════════════════════════════════════════════════════════════════════
// 7. Reduction with index tracking (argmax / argmin)
// ═════════════════════════════════════════════════════════════════════════

/// Find the maximum value and its first occurrence index.
pub fn simd_argmax(data: &[f32]) -> Result<IndexedValue> {
    validate_non_empty(data)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { simd_argmax_avx2(data) });
        }
    }
    Ok(scalar_argmax(data))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_argmax_avx2(data: &[f32]) -> IndexedValue {
    let len = data.len();
    let chunks = len / 8;

    // First pass: find global max value via SIMD.
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_max_ps(acc, v);
    }
    let mut best_val = hmax_avx2(acc);
    for &x in &data[chunks * 8..] {
        if x > best_val {
            best_val = x;
        }
    }

    // Second pass: find first index.
    let mut best_idx = 0;
    for (i, &x) in data.iter().enumerate() {
        if x == best_val {
            best_idx = i;
            break;
        }
    }
    IndexedValue { value: best_val, index: best_idx }
}

fn scalar_argmax(data: &[f32]) -> IndexedValue {
    let mut best = IndexedValue { value: data[0], index: 0 };
    for (i, &x) in data.iter().enumerate().skip(1) {
        if x > best.value {
            best = IndexedValue { value: x, index: i };
        }
    }
    best
}

/// Find the minimum value and its first occurrence index.
pub fn simd_argmin(data: &[f32]) -> Result<IndexedValue> {
    validate_non_empty(data)?;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { simd_argmin_avx2(data) });
        }
    }
    Ok(scalar_argmin(data))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_argmin_avx2(data: &[f32]) -> IndexedValue {
    let len = data.len();
    let chunks = len / 8;

    let mut acc = _mm256_set1_ps(f32::INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
        acc = _mm256_min_ps(acc, v);
    }
    let mut best_val = hmin_avx2(acc);
    for &x in &data[chunks * 8..] {
        if x < best_val {
            best_val = x;
        }
    }

    let mut best_idx = 0;
    for (i, &x) in data.iter().enumerate() {
        if x == best_val {
            best_idx = i;
            break;
        }
    }
    IndexedValue { value: best_val, index: best_idx }
}

fn scalar_argmin(data: &[f32]) -> IndexedValue {
    let mut best = IndexedValue { value: data[0], index: 0 };
    for (i, &x) in data.iter().enumerate().skip(1) {
        if x < best.value {
            best = IndexedValue { value: x, index: i };
        }
    }
    best
}

// ═════════════════════════════════════════════════════════════════════════
// 8. Multi-pass stable reduction (Kahan compensated summation)
// ═════════════════════════════════════════════════════════════════════════

/// Kahan compensated summation for improved numerical accuracy.
pub fn kahan_sum(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    Ok(kahan_sum_impl(data))
}

fn kahan_sum_impl(data: &[f32]) -> f32 {
    let mut sum: f64 = 0.0;
    let mut comp: f64 = 0.0;
    for &x in data {
        let y = x as f64 - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum as f32
}

/// Stable mean using Kahan summation.
pub fn stable_mean(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    let s = kahan_sum_impl(data);
    Ok(s / data.len() as f32)
}

/// Stable variance (population) using two-pass Kahan summation.
pub fn stable_variance(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    let mean = kahan_sum_impl(data) / data.len() as f32;
    let diffs: Vec<f32> = data.iter().map(|&x| (x - mean) * (x - mean)).collect();
    let var = kahan_sum_impl(&diffs) / data.len() as f32;
    Ok(var)
}

/// Stable L2 norm: sqrt(sum of squares) using Kahan summation.
pub fn stable_l2_norm(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    let sq: Vec<f32> = data.iter().map(|&x| x * x).collect();
    Ok(kahan_sum_impl(&sq).sqrt())
}

/// Pairwise summation for improved accuracy over naive left-fold.
/// Splits the array recursively and uses f64 accumulation at leaves.
pub fn pairwise_sum(data: &[f32]) -> Result<f32> {
    validate_non_empty(data)?;
    Ok(pairwise_sum_impl(data) as f32)
}

fn pairwise_sum_impl(data: &[f32]) -> f64 {
    let n = data.len();
    if n <= 16 {
        return data.iter().map(|&x| x as f64).sum();
    }
    let mid = n / 2;
    pairwise_sum_impl(&data[..mid]) + pairwise_sum_impl(&data[mid..])
}

// ═════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "expected {a} ≈ {b} (delta {})", (a - b).abs());
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn constant(n: usize, v: f32) -> Vec<f32> {
        vec![v; n]
    }

    // ═════════════════════════════════════════════════════════════════════
    // 1. Horizontal sum
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_hsum_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_close(simd_horizontal_sum(&data).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_single() {
        assert_close(simd_horizontal_sum(&[42.0]).unwrap(), 42.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_exact_8() {
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        assert_close(simd_horizontal_sum(&data).unwrap(), 36.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_larger_than_8() {
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        assert_close(simd_horizontal_sum(&data).unwrap(), 136.0, 1e-5);
    }

    #[test]
    fn test_simd_reduction_hsum_with_tail() {
        let data: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        assert_close(simd_horizontal_sum(&data).unwrap(), 66.0, 1e-5);
    }

    #[test]
    fn test_simd_reduction_hsum_negatives() {
        let data = [-1.0, -2.0, -3.0, -4.0, -5.0];
        assert_close(simd_horizontal_sum(&data).unwrap(), -15.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_mixed_signs() {
        let data = [10.0, -5.0, 3.0, -2.0, 1.0];
        assert_close(simd_horizontal_sum(&data).unwrap(), 7.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_zeros() {
        let data = constant(16, 0.0);
        assert_close(simd_horizontal_sum(&data).unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hsum_empty_err() {
        assert!(simd_horizontal_sum(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_hsum_large_array() {
        let data = constant(1024, 1.0);
        assert_close(simd_horizontal_sum(&data).unwrap(), 1024.0, 1e-3);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 2. Horizontal max
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_hmax_basic() {
        let data = [1.0, 4.0, 2.0, 3.0];
        assert_close(simd_horizontal_max(&data).unwrap(), 4.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_single() {
        assert_close(simd_horizontal_max(&[7.0]).unwrap(), 7.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_all_same() {
        let data = constant(16, 3.0);
        assert_close(simd_horizontal_max(&data).unwrap(), 3.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_negatives() {
        let data = [-5.0, -1.0, -10.0, -3.0];
        assert_close(simd_horizontal_max(&data).unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_with_tail() {
        let data: Vec<f32> = (0..11).map(|i| i as f32).collect();
        assert_close(simd_horizontal_max(&data).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_exact_8() {
        let data = [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0];
        assert_close(simd_horizontal_max(&data).unwrap(), 8.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_max_at_end() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        assert_close(simd_horizontal_max(&data).unwrap(), 100.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmax_empty_err() {
        assert!(simd_horizontal_max(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_hmax_large_ramp() {
        let data = ramp(256);
        assert_close(simd_horizontal_max(&data).unwrap(), 255.0, 1e-6);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Horizontal min
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_hmin_basic() {
        let data = [3.0, 1.0, 4.0, 1.5];
        assert_close(simd_horizontal_min(&data).unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_single() {
        assert_close(simd_horizontal_min(&[-9.0]).unwrap(), -9.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_all_same() {
        let data = constant(20, -5.0);
        assert_close(simd_horizontal_min(&data).unwrap(), -5.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_positives() {
        let data = [10.0, 5.0, 20.0, 3.0, 8.0];
        assert_close(simd_horizontal_min(&data).unwrap(), 3.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_with_tail() {
        let data: Vec<f32> = (1..=13).map(|i| i as f32).collect();
        assert_close(simd_horizontal_min(&data).unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_min_in_tail() {
        let mut data = vec![10.0; 9];
        data[8] = -1.0;
        assert_close(simd_horizontal_min(&data).unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_hmin_empty_err() {
        assert!(simd_horizontal_min(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_hmin_large_ramp() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32) - 100.0).collect();
        assert_close(simd_horizontal_min(&data).unwrap(), -100.0, 1e-6);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. Tree reduction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_tree_sum_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_close(tree_reduce_sum(&data).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_sum_single() {
        assert_close(tree_reduce_sum(&[5.0]).unwrap(), 5.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_sum_large() {
        let data = constant(1000, 1.0);
        assert_close(tree_reduce_sum(&data).unwrap(), 1000.0, 1e-3);
    }

    #[test]
    fn test_simd_reduction_tree_sum_matches_linear() {
        let data = ramp(200);
        let tree_val = tree_reduce_sum(&data).unwrap();
        let linear_val: f32 = data.iter().sum();
        assert_close(tree_val, linear_val, 1e-3);
    }

    #[test]
    fn test_simd_reduction_tree_sum_empty_err() {
        assert!(tree_reduce_sum(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_tree_max_basic() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_close(tree_reduce_max(&data).unwrap(), 9.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_max_large() {
        let data = ramp(500);
        assert_close(tree_reduce_max(&data).unwrap(), 499.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_max_negatives() {
        let data = [-10.0, -2.0, -5.0, -1.0];
        assert_close(tree_reduce_max(&data).unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_max_empty_err() {
        assert!(tree_reduce_max(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_tree_min_basic() {
        let data = [3.0, 1.0, 4.0, 0.5, 5.0];
        assert_close(tree_reduce_min(&data).unwrap(), 0.5, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_min_large() {
        let data: Vec<f32> = (1..=500).map(|i| i as f32).collect();
        assert_close(tree_reduce_min(&data).unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_tree_min_empty_err() {
        assert!(tree_reduce_min(&[]).is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 5. Prefix sum (scan)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_prefix_incl_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = prefix_sum_inclusive(&data).unwrap();
        assert_close(out[0], 1.0, 1e-6);
        assert_close(out[1], 3.0, 1e-6);
        assert_close(out[2], 6.0, 1e-6);
        assert_close(out[3], 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_prefix_incl_single() {
        let out = prefix_sum_inclusive(&[42.0]).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_simd_reduction_prefix_incl_exact_8() {
        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = prefix_sum_inclusive(&data).unwrap();
        assert_close(out[7], 36.0, 1e-5);
    }

    #[test]
    fn test_simd_reduction_prefix_incl_multi_block() {
        let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let out = prefix_sum_inclusive(&data).unwrap();
        // Verify last element = sum of 1..=24 = 300
        assert_close(out[23], 300.0, 1e-3);
    }

    #[test]
    fn test_simd_reduction_prefix_incl_with_tail() {
        let data: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        let out = prefix_sum_inclusive(&data).unwrap();
        assert_close(out[10], 66.0, 1e-4);
    }

    #[test]
    fn test_simd_reduction_prefix_incl_monotonic() {
        let data = constant(32, 1.0);
        let out = prefix_sum_inclusive(&data).unwrap();
        for i in 1..out.len() {
            assert!(out[i] >= out[i - 1], "prefix sum should be monotonic for non-negative inputs");
        }
    }

    #[test]
    fn test_simd_reduction_prefix_incl_empty_err() {
        assert!(prefix_sum_inclusive(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_prefix_excl_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = prefix_sum_exclusive(&data).unwrap();
        assert_close(out[0], 0.0, 1e-6);
        assert_close(out[1], 1.0, 1e-6);
        assert_close(out[2], 3.0, 1e-6);
        assert_close(out[3], 6.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_prefix_excl_single() {
        let out = prefix_sum_exclusive(&[99.0]).unwrap();
        assert_eq!(out, vec![0.0]);
    }

    #[test]
    fn test_simd_reduction_prefix_excl_empty_err() {
        assert!(prefix_sum_exclusive(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_prefix_incl_last_equals_sum() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let incl = prefix_sum_inclusive(&data).unwrap();
        let total: f32 = data.iter().sum();
        assert_close(*incl.last().unwrap(), total, 1e-2);
    }

    #[test]
    fn test_simd_reduction_prefix_excl_incl_relationship() {
        let data = [2.0, 3.0, 5.0, 7.0, 11.0];
        let incl = prefix_sum_inclusive(&data).unwrap();
        let excl = prefix_sum_exclusive(&data).unwrap();
        for i in 0..data.len() {
            assert_close(incl[i], excl[i] + data[i], 1e-6);
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. Segmented reduction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_seg_sum_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = segmented_reduce(&data, 3, SegmentOp::Sum).unwrap();
        assert_eq!(out.len(), 2);
        assert_close(out[0], 6.0, 1e-6);
        assert_close(out[1], 15.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_max_basic() {
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = segmented_reduce(&data, 3, SegmentOp::Max).unwrap();
        assert_close(out[0], 5.0, 1e-6);
        assert_close(out[1], 6.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_min_basic() {
        let data = [3.0, 1.0, 5.0, 6.0, 2.0, 4.0];
        let out = segmented_reduce(&data, 3, SegmentOp::Min).unwrap();
        assert_close(out[0], 1.0, 1e-6);
        assert_close(out[1], 2.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_size_1() {
        let data = [10.0, 20.0, 30.0];
        let out = segmented_reduce(&data, 1, SegmentOp::Sum).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_simd_reduction_seg_full() {
        let data = [1.0, 2.0, 3.0];
        let out = segmented_reduce(&data, 3, SegmentOp::Sum).unwrap();
        assert_eq!(out.len(), 1);
        assert_close(out[0], 6.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_empty_err() {
        assert!(segmented_reduce(&[], 2, SegmentOp::Sum).is_err());
    }

    #[test]
    fn test_simd_reduction_seg_zero_size_err() {
        assert!(segmented_reduce(&[1.0], 0, SegmentOp::Sum).is_err());
    }

    #[test]
    fn test_simd_reduction_seg_not_divisible_err() {
        assert!(segmented_reduce(&[1.0, 2.0, 3.0], 2, SegmentOp::Sum).is_err());
    }

    #[test]
    fn test_simd_reduction_seg_var_sum() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let offsets = [0, 2, 5];
        let out = segmented_reduce_variable(&data, &offsets, SegmentOp::Sum).unwrap();
        assert_close(out[0], 3.0, 1e-6);
        assert_close(out[1], 12.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_var_max() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let offsets = [0, 3, 5];
        let out = segmented_reduce_variable(&data, &offsets, SegmentOp::Max).unwrap();
        assert_close(out[0], 5.0, 1e-6);
        assert_close(out[1], 4.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_var_min() {
        let data = [5.0, 1.0, 3.0, 4.0, 2.0];
        let offsets = [0, 2, 5];
        let out = segmented_reduce_variable(&data, &offsets, SegmentOp::Min).unwrap();
        assert_close(out[0], 1.0, 1e-6);
        assert_close(out[1], 2.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_var_empty_segment() {
        let data = [1.0, 2.0, 3.0];
        let offsets = [0, 0, 3]; // first segment is empty
        let out = segmented_reduce_variable(&data, &offsets, SegmentOp::Sum).unwrap();
        assert_close(out[0], 0.0, 1e-6); // identity for sum
        assert_close(out[1], 6.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_seg_var_bad_offsets_err() {
        assert!(segmented_reduce_variable(&[1.0], &[0], SegmentOp::Sum).is_err());
    }

    #[test]
    fn test_simd_reduction_seg_var_out_of_bounds_err() {
        assert!(segmented_reduce_variable(&[1.0], &[0, 5], SegmentOp::Sum).is_err());
    }

    #[test]
    fn test_simd_reduction_seg_var_reversed_err() {
        assert!(segmented_reduce_variable(&[1.0, 2.0], &[2, 0], SegmentOp::Sum).is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. Argmax / Argmin
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_argmax_basic() {
        let data = [1.0, 3.0, 2.0, 5.0, 4.0];
        let r = simd_argmax(&data).unwrap();
        assert_close(r.value, 5.0, 1e-6);
        assert_eq!(r.index, 3);
    }

    #[test]
    fn test_simd_reduction_argmax_single() {
        let r = simd_argmax(&[42.0]).unwrap();
        assert_close(r.value, 42.0, 1e-6);
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_simd_reduction_argmax_first_occurrence() {
        let data = [1.0, 5.0, 5.0, 3.0];
        let r = simd_argmax(&data).unwrap();
        assert_eq!(r.index, 1, "should return first occurrence");
    }

    #[test]
    fn test_simd_reduction_argmax_all_same() {
        let data = constant(16, 3.0);
        let r = simd_argmax(&data).unwrap();
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_simd_reduction_argmax_negatives() {
        let data = [-5.0, -1.0, -10.0, -3.0];
        let r = simd_argmax(&data).unwrap();
        assert_close(r.value, -1.0, 1e-6);
        assert_eq!(r.index, 1);
    }

    #[test]
    fn test_simd_reduction_argmax_larger_than_8() {
        let mut data = vec![0.0; 16];
        data[12] = 99.0;
        let r = simd_argmax(&data).unwrap();
        assert_close(r.value, 99.0, 1e-6);
        assert_eq!(r.index, 12);
    }

    #[test]
    fn test_simd_reduction_argmax_max_in_tail() {
        let mut data = vec![0.0; 10];
        data[9] = 50.0;
        let r = simd_argmax(&data).unwrap();
        assert_close(r.value, 50.0, 1e-6);
        assert_eq!(r.index, 9);
    }

    #[test]
    fn test_simd_reduction_argmax_empty_err() {
        assert!(simd_argmax(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_argmin_basic() {
        let data = [5.0, 3.0, 1.0, 4.0, 2.0];
        let r = simd_argmin(&data).unwrap();
        assert_close(r.value, 1.0, 1e-6);
        assert_eq!(r.index, 2);
    }

    #[test]
    fn test_simd_reduction_argmin_single() {
        let r = simd_argmin(&[-7.0]).unwrap();
        assert_close(r.value, -7.0, 1e-6);
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_simd_reduction_argmin_first_occurrence() {
        let data = [3.0, 1.0, 1.0, 5.0];
        let r = simd_argmin(&data).unwrap();
        assert_eq!(r.index, 1, "should return first occurrence");
    }

    #[test]
    fn test_simd_reduction_argmin_all_same() {
        let data = constant(16, 7.0);
        let r = simd_argmin(&data).unwrap();
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_simd_reduction_argmin_positives() {
        let data = [10.0, 5.0, 20.0, 3.0];
        let r = simd_argmin(&data).unwrap();
        assert_close(r.value, 3.0, 1e-6);
        assert_eq!(r.index, 3);
    }

    #[test]
    fn test_simd_reduction_argmin_larger_than_8() {
        let mut data = vec![100.0; 20];
        data[15] = -1.0;
        let r = simd_argmin(&data).unwrap();
        assert_close(r.value, -1.0, 1e-6);
        assert_eq!(r.index, 15);
    }

    #[test]
    fn test_simd_reduction_argmin_min_in_tail() {
        let mut data = vec![100.0; 11];
        data[10] = 0.5;
        let r = simd_argmin(&data).unwrap();
        assert_close(r.value, 0.5, 1e-6);
        assert_eq!(r.index, 10);
    }

    #[test]
    fn test_simd_reduction_argmin_empty_err() {
        assert!(simd_argmin(&[]).is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 8. Stable summation
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_kahan_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_close(kahan_sum(&data).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_kahan_single() {
        assert_close(kahan_sum(&[3.14]).unwrap(), 3.14, 1e-6);
    }

    #[test]
    fn test_simd_reduction_kahan_accuracy() {
        // Many small values that challenge naive summation.
        let data: Vec<f32> = (0..10_000).map(|_| 1e-7_f32).collect();
        let result = kahan_sum(&data).unwrap();
        assert_close(result, 1e-3, 1e-5);
    }

    #[test]
    fn test_simd_reduction_kahan_empty_err() {
        assert!(kahan_sum(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_stable_mean_basic() {
        let data = [2.0, 4.0, 6.0, 8.0];
        assert_close(stable_mean(&data).unwrap(), 5.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_stable_mean_single() {
        assert_close(stable_mean(&[42.0]).unwrap(), 42.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_stable_mean_empty_err() {
        assert!(stable_mean(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_stable_variance_constant() {
        let data = constant(100, 5.0);
        assert_close(stable_variance(&data).unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_stable_variance_known() {
        // Variance of [1,2,3,4,5] = 2.0
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_close(stable_variance(&data).unwrap(), 2.0, 1e-5);
    }

    #[test]
    fn test_simd_reduction_stable_variance_empty_err() {
        assert!(stable_variance(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_stable_l2_norm_unit() {
        let data = [3.0, 4.0];
        assert_close(stable_l2_norm(&data).unwrap(), 5.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_stable_l2_norm_single() {
        assert_close(stable_l2_norm(&[-7.0]).unwrap(), 7.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_stable_l2_norm_empty_err() {
        assert!(stable_l2_norm(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_pairwise_basic() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_close(pairwise_sum(&data).unwrap(), 10.0, 1e-6);
    }

    #[test]
    fn test_simd_reduction_pairwise_large() {
        let data = constant(10_000, 1.0);
        assert_close(pairwise_sum(&data).unwrap(), 10_000.0, 1e-2);
    }

    #[test]
    fn test_simd_reduction_pairwise_empty_err() {
        assert!(pairwise_sum(&[]).is_err());
    }

    #[test]
    fn test_simd_reduction_pairwise_matches_kahan() {
        let data: Vec<f32> = (0..500).map(|i| (i as f32) * 0.01).collect();
        let pw = pairwise_sum(&data).unwrap();
        let kh = kahan_sum(&data).unwrap();
        assert_close(pw, kh, 1e-2);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Cross-function consistency
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_simd_reduction_sum_agrees_with_tree() {
        let data = ramp(256);
        let s = simd_horizontal_sum(&data).unwrap();
        let t = tree_reduce_sum(&data).unwrap();
        assert_close(s, t, 1e-2);
    }

    #[test]
    fn test_simd_reduction_max_agrees_with_tree() {
        let data = ramp(128);
        let s = simd_horizontal_max(&data).unwrap();
        let t = tree_reduce_max(&data).unwrap();
        assert_close(s, t, 1e-6);
    }

    #[test]
    fn test_simd_reduction_min_agrees_with_tree() {
        let data: Vec<f32> = (0..128).map(|i| 100.0 - i as f32).collect();
        let s = simd_horizontal_min(&data).unwrap();
        let t = tree_reduce_min(&data).unwrap();
        assert_close(s, t, 1e-6);
    }

    #[test]
    fn test_simd_reduction_argmax_value_equals_max() {
        let data = [3.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 4.0, 6.0];
        let max_val = simd_horizontal_max(&data).unwrap();
        let argmax_val = simd_argmax(&data).unwrap();
        assert_close(max_val, argmax_val.value, 1e-6);
    }

    #[test]
    fn test_simd_reduction_argmin_value_equals_min() {
        let data = [3.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 4.0, 6.0];
        let min_val = simd_horizontal_min(&data).unwrap();
        let argmin_val = simd_argmin(&data).unwrap();
        assert_close(min_val, argmin_val.value, 1e-6);
    }

    #[test]
    fn test_simd_reduction_max_ge_min() {
        let data = [-5.0, 10.0, 0.0, -3.0, 7.0, 2.0, -1.0, 8.0, 4.0];
        let mx = simd_horizontal_max(&data).unwrap();
        let mn = simd_horizontal_min(&data).unwrap();
        assert!(mx >= mn, "max {mx} should be >= min {mn}");
    }

    #[test]
    fn test_simd_reduction_sum_matches_kahan() {
        let data: Vec<f32> = (0..200).map(|i| (i as f32) * 0.5).collect();
        let naive = simd_horizontal_sum(&data).unwrap();
        let kh = kahan_sum(&data).unwrap();
        assert_close(naive, kh, 1.0); // wider tol for naive vs compensated
    }

    #[test]
    fn test_simd_reduction_seg_sums_equal_total() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let seg_sums = segmented_reduce(&data, 4, SegmentOp::Sum).unwrap();
        let total = simd_horizontal_sum(&data).unwrap();
        let seg_total: f32 = seg_sums.iter().sum();
        assert_close(seg_total, total, 1e-4);
    }

    #[test]
    fn test_simd_reduction_seg_max_vs_global() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let seg_maxes = segmented_reduce(&data, 4, SegmentOp::Max).unwrap();
        let global_max = simd_horizontal_max(&data).unwrap();
        let seg_best = seg_maxes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_close(seg_best, global_max, 1e-6);
    }

    #[test]
    fn test_simd_reduction_prefix_sum_all_zeros() {
        let data = constant(16, 0.0);
        let out = prefix_sum_inclusive(&data).unwrap();
        for &v in &out {
            assert_close(v, 0.0, 1e-6);
        }
    }

    #[test]
    fn test_simd_reduction_prefix_excl_last_plus_last_data_eq_incl() {
        let data = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 9.0];
        let incl = prefix_sum_inclusive(&data).unwrap();
        let excl = prefix_sum_exclusive(&data).unwrap();
        let n = data.len();
        assert_close(excl[n - 1] + data[n - 1], incl[n - 1], 1e-5);
    }
}
