//! CPU SIMD fast reduction operations.
//!
//! Provides high-performance reduction primitives with AVX2 acceleration
//! and scalar fallbacks: sum, max, min (with index tracking for argmax/argmin),
//! mean, variance/stddev (Welford's algorithm), and L1/L2 norms.
//!
//! All reductions support both contiguous and strided memory layouts.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{KernelError, Result};

// ── AVX2 intrinsics ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Configuration ──────────────────────────────────────────────────────

/// Data type hint for reduction output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceDtype {
    F32,
    F64,
}

/// Configuration for a reduction operation.
#[derive(Debug, Clone)]
pub struct ReduceConfig {
    /// Axis to reduce along. `None` reduces all elements.
    pub axis: Option<usize>,
    /// Whether to keep the reduced dimension as size 1.
    pub keepdims: bool,
    /// Output data type hint.
    pub dtype: ReduceDtype,
}

impl ReduceConfig {
    pub fn new(axis: Option<usize>, keepdims: bool, dtype: ReduceDtype) -> Self {
        Self { axis, keepdims, dtype }
    }

    /// Default config: reduce all, don't keep dims, f32 output.
    pub fn flat() -> Self {
        Self { axis: None, keepdims: false, dtype: ReduceDtype::F32 }
    }
}

impl Default for ReduceConfig {
    fn default() -> Self {
        Self::flat()
    }
}

// ── Result types ───────────────────────────────────────────────────────

/// Result of a max or min reduction with index tracking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueWithIndex {
    pub value: f32,
    pub index: usize,
}

/// Result of Welford's online variance computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelfordResult {
    pub mean: f32,
    pub variance: f32,
    pub stddev: f32,
    pub count: usize,
}

// ── Cache-line chunk size ──────────────────────────────────────────────

/// Elements per processing chunk (fits in L1 cache line).
const CHUNK_SIZE: usize = 1024;

// ── FastReduce ─────────────────────────────────────────────────────────

/// High-performance reduction operations with SIMD acceleration.
pub struct FastReduce;

impl FastReduce {
    // ── Sum ────────────────────────────────────────────────────────────

    /// Horizontal sum of a contiguous f32 slice.
    pub fn sum(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Safety: feature detection guarantees AVX2.
                return unsafe { sum_avx2(data) };
            }
        }
        sum_scalar(data)
    }

    /// Sum over strided elements: data[0], data[stride], data[2*stride], …
    pub fn sum_strided(data: &[f32], stride: usize, count: usize) -> f32 {
        if count == 0 || stride == 0 {
            return 0.0;
        }
        if stride == 1 && count <= data.len() {
            return Self::sum(&data[..count]);
        }
        sum_strided_scalar(data, stride, count)
    }

    // ── Max / Argmax ──────────────────────────────────────────────────

    /// Maximum value and its index in a contiguous slice.
    ///
    /// Returns `None` for empty slices.
    pub fn argmax(data: &[f32]) -> Option<ValueWithIndex> {
        if data.is_empty() {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Safety: feature detection guarantees AVX2.
                return Some(unsafe { argmax_avx2(data) });
            }
        }
        Some(argmax_scalar(data))
    }

    /// Argmax over strided elements.
    pub fn argmax_strided(data: &[f32], stride: usize, count: usize) -> Option<ValueWithIndex> {
        if count == 0 || stride == 0 {
            return None;
        }
        Some(argmax_strided_scalar(data, stride, count))
    }

    // ── Min / Argmin ──────────────────────────────────────────────────

    /// Minimum value and its index in a contiguous slice.
    ///
    /// Returns `None` for empty slices.
    pub fn argmin(data: &[f32]) -> Option<ValueWithIndex> {
        if data.is_empty() {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Safety: feature detection guarantees AVX2.
                return Some(unsafe { argmin_avx2(data) });
            }
        }
        Some(argmin_scalar(data))
    }

    /// Argmin over strided elements.
    pub fn argmin_strided(data: &[f32], stride: usize, count: usize) -> Option<ValueWithIndex> {
        if count == 0 || stride == 0 {
            return None;
        }
        Some(argmin_strided_scalar(data, stride, count))
    }

    // ── Mean ──────────────────────────────────────────────────────────

    /// Arithmetic mean of a contiguous slice.
    pub fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        Self::sum(data) / data.len() as f32
    }

    /// Mean over strided elements.
    pub fn mean_strided(data: &[f32], stride: usize, count: usize) -> f32 {
        if count == 0 {
            return 0.0;
        }
        Self::sum_strided(data, stride, count) / count as f32
    }

    // ── Variance / Stddev (Welford) ───────────────────────────────────

    /// Compute mean, variance, and standard deviation using Welford's
    /// numerically stable online algorithm.
    pub fn welford(data: &[f32]) -> WelfordResult {
        if data.is_empty() {
            return WelfordResult { mean: 0.0, variance: 0.0, stddev: 0.0, count: 0 };
        }
        welford_scalar(data)
    }

    /// Welford's algorithm over strided elements.
    pub fn welford_strided(data: &[f32], stride: usize, count: usize) -> WelfordResult {
        if count == 0 || stride == 0 {
            return WelfordResult { mean: 0.0, variance: 0.0, stddev: 0.0, count: 0 };
        }
        welford_strided_scalar(data, stride, count)
    }

    // ── L1 Norm ───────────────────────────────────────────────────────

    /// L1 norm (sum of absolute values).
    pub fn l1_norm(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Safety: feature detection guarantees AVX2.
                return unsafe { l1_norm_avx2(data) };
            }
        }
        l1_norm_scalar(data)
    }

    /// L1 norm over strided elements.
    pub fn l1_norm_strided(data: &[f32], stride: usize, count: usize) -> f32 {
        if count == 0 || stride == 0 {
            return 0.0;
        }
        l1_norm_strided_scalar(data, stride, count)
    }

    // ── L2 Norm ───────────────────────────────────────────────────────

    /// L2 norm (Euclidean length).
    pub fn l2_norm(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // Safety: feature detection guarantees AVX2.
                return unsafe { l2_norm_avx2(data) };
            }
        }
        l2_norm_scalar(data)
    }

    /// L2 norm over strided elements.
    pub fn l2_norm_strided(data: &[f32], stride: usize, count: usize) -> f32 {
        if count == 0 || stride == 0 {
            return 0.0;
        }
        l2_norm_strided_scalar(data, stride, count)
    }

    // ── Axis-aware dispatch ───────────────────────────────────────────

    /// Reduce a tensor along an axis (or globally) using the specified
    /// operation. Returns a flat result vector.
    pub fn reduce(
        data: &[f32],
        shape: &[usize],
        op: FastReduceOp,
        config: &ReduceConfig,
    ) -> Result<Vec<f32>> {
        validate(data, shape, config.axis)?;

        if data.is_empty() {
            return Ok(vec![op.identity()]);
        }

        match config.axis {
            None => {
                let val = match op {
                    FastReduceOp::Sum => Self::sum(data),
                    FastReduceOp::Max => {
                        Self::argmax(data).map(|v| v.value).unwrap_or(f32::NEG_INFINITY)
                    }
                    FastReduceOp::Min => {
                        Self::argmin(data).map(|v| v.value).unwrap_or(f32::INFINITY)
                    }
                    FastReduceOp::Mean => Self::mean(data),
                    FastReduceOp::L1Norm => Self::l1_norm(data),
                    FastReduceOp::L2Norm => Self::l2_norm(data),
                };
                Ok(vec![val])
            }
            Some(ax) => reduce_axis(data, shape, ax, op),
        }
    }
}

// ── Operation enum ─────────────────────────────────────────────────────

/// Reduction operations supported by [`FastReduce::reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastReduceOp {
    Sum,
    Max,
    Min,
    Mean,
    L1Norm,
    L2Norm,
}

impl FastReduceOp {
    /// Identity element for the operation.
    pub fn identity(self) -> f32 {
        match self {
            Self::Sum | Self::Mean | Self::L1Norm | Self::L2Norm => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }
}

// ── Scalar implementations ─────────────────────────────────────────────

fn sum_scalar(data: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for chunk in data.chunks(CHUNK_SIZE) {
        let partial: f32 = chunk.iter().sum();
        total += partial;
    }
    total
}

fn sum_strided_scalar(data: &[f32], stride: usize, count: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..count {
        let idx = i * stride;
        if idx < data.len() {
            acc += data[idx];
        }
    }
    acc
}

fn argmax_scalar(data: &[f32]) -> ValueWithIndex {
    let mut best = ValueWithIndex { value: f32::NEG_INFINITY, index: 0 };
    for (i, &v) in data.iter().enumerate() {
        if v > best.value {
            best = ValueWithIndex { value: v, index: i };
        }
    }
    best
}

fn argmax_strided_scalar(data: &[f32], stride: usize, count: usize) -> ValueWithIndex {
    let mut best = ValueWithIndex { value: f32::NEG_INFINITY, index: 0 };
    for i in 0..count {
        let idx = i * stride;
        if idx < data.len() && data[idx] > best.value {
            best = ValueWithIndex { value: data[idx], index: i };
        }
    }
    best
}

fn argmin_scalar(data: &[f32]) -> ValueWithIndex {
    let mut best = ValueWithIndex { value: f32::INFINITY, index: 0 };
    for (i, &v) in data.iter().enumerate() {
        if v < best.value {
            best = ValueWithIndex { value: v, index: i };
        }
    }
    best
}

fn argmin_strided_scalar(data: &[f32], stride: usize, count: usize) -> ValueWithIndex {
    let mut best = ValueWithIndex { value: f32::INFINITY, index: 0 };
    for i in 0..count {
        let idx = i * stride;
        if idx < data.len() && data[idx] < best.value {
            best = ValueWithIndex { value: data[idx], index: i };
        }
    }
    best
}

fn welford_scalar(data: &[f32]) -> WelfordResult {
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut n = 0_usize;
    for &x in data {
        n += 1;
        let delta = x as f64 - mean;
        mean += delta / n as f64;
        let delta2 = x as f64 - mean;
        m2 += delta * delta2;
    }
    let variance = if n < 2 { 0.0 } else { m2 / (n - 1) as f64 };
    WelfordResult {
        mean: mean as f32,
        variance: variance as f32,
        stddev: (variance.sqrt()) as f32,
        count: n,
    }
}

fn welford_strided_scalar(data: &[f32], stride: usize, count: usize) -> WelfordResult {
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut n = 0_usize;
    for i in 0..count {
        let idx = i * stride;
        if idx >= data.len() {
            break;
        }
        let x = data[idx] as f64;
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let variance = if n < 2 { 0.0 } else { m2 / (n - 1) as f64 };
    WelfordResult {
        mean: mean as f32,
        variance: variance as f32,
        stddev: (variance.sqrt()) as f32,
        count: n,
    }
}

fn l1_norm_scalar(data: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for chunk in data.chunks(CHUNK_SIZE) {
        let partial: f32 = chunk.iter().map(|x| x.abs()).sum();
        total += partial;
    }
    total
}

fn l1_norm_strided_scalar(data: &[f32], stride: usize, count: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..count {
        let idx = i * stride;
        if idx < data.len() {
            acc += data[idx].abs();
        }
    }
    acc
}

fn l2_norm_scalar(data: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for chunk in data.chunks(CHUNK_SIZE) {
        let partial: f32 = chunk.iter().map(|x| x * x).sum();
        total += partial;
    }
    total.sqrt()
}

fn l2_norm_strided_scalar(data: &[f32], stride: usize, count: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..count {
        let idx = i * stride;
        if idx < data.len() {
            acc += data[idx] * data[idx];
        }
    }
    acc.sqrt()
}

// ── AVX2 implementations ──────────────────────────────────────────────

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(data: &[f32]) -> f32 {
    let n = data.len();
    let ptr = data.as_ptr();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        acc = _mm256_add_ps(acc, v);
    }
    let mut total = hsum_avx2(acc);
    for i in (chunks * 8)..n {
        total += *ptr.add(i);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_avx2(data: &[f32]) -> ValueWithIndex {
    let n = data.len();
    let ptr = data.as_ptr();
    let chunks = n / 8;

    let mut best_val = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut best_chunk_idx = 0usize;

    // Vectorised pass: find max value per 8-wide lane.
    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        best_val = _mm256_max_ps(best_val, v);
    }

    // Horizontal max of the accumulator.
    let hi128 = _mm256_extractf128_ps(best_val, 1);
    let lo128 = _mm256_castps256_ps128(best_val);
    let m128 = _mm_max_ps(lo128, hi128);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    let mut best_chunk_val = _mm_cvtss_f32(m32);

    // Linear scan to find the first index matching the max.
    for i in 0..(chunks * 8) {
        if *ptr.add(i) == best_chunk_val {
            best_chunk_idx = i;
            break;
        }
    }

    // Tail elements.
    for i in (chunks * 8)..n {
        let v = *ptr.add(i);
        if v > best_chunk_val {
            best_chunk_val = v;
            best_chunk_idx = i;
        }
    }

    ValueWithIndex { value: best_chunk_val, index: best_chunk_idx }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmin_avx2(data: &[f32]) -> ValueWithIndex {
    let n = data.len();
    let ptr = data.as_ptr();
    let chunks = n / 8;

    let mut best_val = _mm256_set1_ps(f32::INFINITY);
    let mut best_chunk_idx = 0usize;

    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        best_val = _mm256_min_ps(best_val, v);
    }

    let hi128 = _mm256_extractf128_ps(best_val, 1);
    let lo128 = _mm256_castps256_ps128(best_val);
    let m128 = _mm_min_ps(lo128, hi128);
    let m64 = _mm_min_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_min_ss(m64, _mm_shuffle_ps(m64, m64, 1));
    let mut best_chunk_val = _mm_cvtss_f32(m32);

    for i in 0..(chunks * 8) {
        if *ptr.add(i) == best_chunk_val {
            best_chunk_idx = i;
            break;
        }
    }

    for i in (chunks * 8)..n {
        let v = *ptr.add(i);
        if v < best_chunk_val {
            best_chunk_val = v;
            best_chunk_idx = i;
        }
    }

    ValueWithIndex { value: best_chunk_val, index: best_chunk_idx }
}

/// AVX2 absolute-value mask: clear the sign bit of each f32 lane.
#[cfg(target_arch = "x86_64")]
const ABS_MASK: u32 = 0x7FFF_FFFF;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_norm_avx2(data: &[f32]) -> f32 {
    let n = data.len();
    let ptr = data.as_ptr();
    let abs_mask = _mm256_set1_ps(f32::from_bits(ABS_MASK));
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_v = _mm256_and_ps(v, abs_mask);
        acc = _mm256_add_ps(acc, abs_v);
    }
    let mut total = hsum_avx2(acc);
    for i in (chunks * 8)..n {
        total += (*ptr.add(i)).abs();
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_norm_avx2(data: &[f32]) -> f32 {
    let n = data.len();
    let ptr = data.as_ptr();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    let mut total = hsum_avx2(acc);
    for i in (chunks * 8)..n {
        let v = *ptr.add(i);
        total += v * v;
    }
    total.sqrt()
}

// ── Axis-aware reduction ──────────────────────────────────────────────

fn reduce_axis(data: &[f32], shape: &[usize], axis: usize, op: FastReduceOp) -> Result<Vec<f32>> {
    let axis_len = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);

    let out_len = outer_size * inner_size;
    let mut output = vec![op.identity(); out_len];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = op.identity();
            for k in 0..axis_len {
                let idx = outer * (axis_len * inner_size) + k * inner_size + inner;
                let v = data[idx];
                acc = match op {
                    FastReduceOp::Sum | FastReduceOp::Mean => acc + v,
                    FastReduceOp::Max => acc.max(v),
                    FastReduceOp::Min => acc.min(v),
                    FastReduceOp::L1Norm => acc + v.abs(),
                    FastReduceOp::L2Norm => acc + v * v,
                };
            }
            output[outer * inner_size + inner] = match op {
                FastReduceOp::Mean => {
                    if axis_len == 0 {
                        0.0
                    } else {
                        acc / axis_len as f32
                    }
                }
                FastReduceOp::L2Norm => acc.sqrt(),
                _ => acc,
            };
        }
    }

    Ok(output)
}

// ── Validation ─────────────────────────────────────────────────────────

fn validate(data: &[f32], shape: &[usize], axis: Option<usize>) -> Result<()> {
    if shape.is_empty() {
        return Err(
            KernelError::InvalidArguments { reason: "shape must be non-empty".into() }.into()
        );
    }
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected,
            ),
        }
        .into());
    }
    if let Some(ax) = axis
        && ax >= shape.len()
    {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {} out of bounds for shape {:?}", ax, shape),
        }
        .into());
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // -- Sum tests -------------------------------------------------------

    #[test]
    fn test_sum_basic() {
        assert!(approx(FastReduce::sum(&[1.0, 2.0, 3.0, 4.0]), 10.0, 1e-6));
    }

    #[test]
    fn test_sum_empty() {
        assert_eq!(FastReduce::sum(&[]), 0.0);
    }

    #[test]
    fn test_sum_single() {
        assert!(approx(FastReduce::sum(&[42.0]), 42.0, 1e-6));
    }

    #[test]
    fn test_sum_negative() {
        assert!(approx(FastReduce::sum(&[-1.0, -2.0, -3.0]), -6.0, 1e-6));
    }

    #[test]
    fn test_sum_large() {
        let n = 10_000usize;
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let expected = (n * (n + 1)) as f32 / 2.0;
        let result = FastReduce::sum(&data);
        assert!((result - expected).abs() / expected < 1e-4, "expected {expected}, got {result}");
    }

    #[test]
    fn test_sum_strided() {
        let data = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        assert!(approx(FastReduce::sum_strided(&data, 2, 3), 6.0, 1e-6));
    }

    // -- Argmax tests ----------------------------------------------------

    #[test]
    fn test_argmax_basic() {
        let r = FastReduce::argmax(&[1.0, 5.0, 3.0, 2.0]).unwrap();
        assert!(approx(r.value, 5.0, 1e-6));
        assert_eq!(r.index, 1);
    }

    #[test]
    fn test_argmax_empty() {
        assert!(FastReduce::argmax(&[]).is_none());
    }

    #[test]
    fn test_argmax_single() {
        let r = FastReduce::argmax(&[7.0]).unwrap();
        assert!(approx(r.value, 7.0, 1e-6));
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_argmax_negative() {
        let r = FastReduce::argmax(&[-3.0, -1.0, -4.0]).unwrap();
        assert!(approx(r.value, -1.0, 1e-6));
        assert_eq!(r.index, 1);
    }

    #[test]
    fn test_argmax_first_occurrence() {
        let r = FastReduce::argmax(&[5.0, 3.0, 5.0, 2.0]).unwrap();
        assert_eq!(r.index, 0, "should return first occurrence");
    }

    #[test]
    fn test_argmax_strided() {
        let data = [1.0, 99.0, 3.0, 99.0, 2.0, 99.0];
        let r = FastReduce::argmax_strided(&data, 2, 3).unwrap();
        assert!(approx(r.value, 3.0, 1e-6));
        assert_eq!(r.index, 1); // logical index in strided view
    }

    // -- Argmin tests ----------------------------------------------------

    #[test]
    fn test_argmin_basic() {
        let r = FastReduce::argmin(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert!(approx(r.value, 1.0, 1e-6));
        assert_eq!(r.index, 1);
    }

    #[test]
    fn test_argmin_empty() {
        assert!(FastReduce::argmin(&[]).is_none());
    }

    #[test]
    fn test_argmin_first_occurrence() {
        let r = FastReduce::argmin(&[1.0, 3.0, 1.0, 5.0]).unwrap();
        assert_eq!(r.index, 0);
    }

    // -- Mean tests ------------------------------------------------------

    #[test]
    fn test_mean_basic() {
        assert!(approx(FastReduce::mean(&[2.0, 4.0, 6.0, 8.0]), 5.0, 1e-6));
    }

    #[test]
    fn test_mean_empty() {
        assert_eq!(FastReduce::mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert!(approx(FastReduce::mean(&[42.0]), 42.0, 1e-6));
    }

    // -- Welford variance tests ------------------------------------------

    #[test]
    fn test_welford_basic() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = FastReduce::welford(&data);
        assert_eq!(r.count, 8);
        assert!(approx(r.mean, 5.0, 1e-5));
        // sample variance = 4.571…
        assert!(approx(r.variance, 4.571_428_6, 1e-3));
        assert!(approx(r.stddev, r.variance.sqrt(), 1e-5));
    }

    #[test]
    fn test_welford_empty() {
        let r = FastReduce::welford(&[]);
        assert_eq!(r.count, 0);
        assert_eq!(r.mean, 0.0);
        assert_eq!(r.variance, 0.0);
    }

    #[test]
    fn test_welford_single() {
        let r = FastReduce::welford(&[42.0]);
        assert_eq!(r.count, 1);
        assert!(approx(r.mean, 42.0, 1e-6));
        assert_eq!(r.variance, 0.0);
    }

    #[test]
    fn test_welford_constant() {
        let data = vec![3.0; 100];
        let r = FastReduce::welford(&data);
        assert!(approx(r.mean, 3.0, 1e-6));
        assert!(approx(r.variance, 0.0, 1e-6));
    }

    #[test]
    fn test_welford_numerical_stability() {
        // Large offset with small variance — naive two-pass loses precision.
        let data: Vec<f32> = (0..1000).map(|i| 1e6 + i as f32 * 0.001).collect();
        let r = FastReduce::welford(&data);
        // Exact sample variance for 0,0.001,...,0.999 shifted:
        // Var = (n-1)^{-1} * sum (x_i - mean)^2
        // For uniformly spaced, var ≈ (0.999)^2 / 12 * n/(n-1)
        assert!(r.variance > 0.0, "variance must be positive");
        assert!(r.variance < 1.0, "variance should be small (~0.083)");
    }

    #[test]
    fn test_welford_strided() {
        let data = [1.0, 99.0, 3.0, 99.0, 5.0, 99.0];
        let r = FastReduce::welford_strided(&data, 2, 3);
        assert_eq!(r.count, 3);
        assert!(approx(r.mean, 3.0, 1e-5));
    }

    // -- L1 norm tests ---------------------------------------------------

    #[test]
    fn test_l1_norm_basic() {
        assert!(approx(FastReduce::l1_norm(&[-3.0, 4.0]), 7.0, 1e-6));
    }

    #[test]
    fn test_l1_norm_empty() {
        assert_eq!(FastReduce::l1_norm(&[]), 0.0);
    }

    #[test]
    fn test_l1_norm_all_negative() {
        assert!(approx(FastReduce::l1_norm(&[-1.0, -2.0, -3.0]), 6.0, 1e-6));
    }

    #[test]
    fn test_l1_norm_strided() {
        let data = [-1.0, 99.0, 2.0, 99.0, -3.0, 99.0];
        assert!(approx(FastReduce::l1_norm_strided(&data, 2, 3), 6.0, 1e-6));
    }

    // -- L2 norm tests ---------------------------------------------------

    #[test]
    fn test_l2_norm_basic() {
        assert!(approx(FastReduce::l2_norm(&[3.0, 4.0]), 5.0, 1e-6));
    }

    #[test]
    fn test_l2_norm_empty() {
        assert_eq!(FastReduce::l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let n = 100;
        let val = 1.0 / (n as f32).sqrt();
        let data = vec![val; n];
        assert!(approx(FastReduce::l2_norm(&data), 1.0, 1e-5));
    }

    #[test]
    fn test_l2_norm_strided() {
        let data = [3.0, 99.0, 4.0, 99.0];
        assert!(approx(FastReduce::l2_norm_strided(&data, 2, 2), 5.0, 1e-5));
    }

    // -- Axis-aware reduce tests -----------------------------------------

    #[test]
    fn test_reduce_global_sum() {
        let cfg = ReduceConfig::flat();
        let r = FastReduce::reduce(&[1.0, 2.0, 3.0], &[3], FastReduceOp::Sum, &cfg).unwrap();
        assert!(approx(r[0], 6.0, 1e-6));
    }

    #[test]
    fn test_reduce_axis0_sum_2d() {
        let cfg = ReduceConfig::new(Some(0), false, ReduceDtype::F32);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = FastReduce::reduce(&data, &[2, 3], FastReduceOp::Sum, &cfg).unwrap();
        assert_eq!(r.len(), 3);
        assert!(approx(r[0], 5.0, 1e-6));
        assert!(approx(r[1], 7.0, 1e-6));
        assert!(approx(r[2], 9.0, 1e-6));
    }

    #[test]
    fn test_reduce_axis1_mean_2d() {
        let cfg = ReduceConfig::new(Some(1), false, ReduceDtype::F32);
        let data = vec![2.0, 4.0, 6.0, 1.0, 3.0, 5.0];
        let r = FastReduce::reduce(&data, &[2, 3], FastReduceOp::Mean, &cfg).unwrap();
        assert!(approx(r[0], 4.0, 1e-6));
        assert!(approx(r[1], 3.0, 1e-6));
    }

    #[test]
    fn test_reduce_empty() {
        let cfg = ReduceConfig::flat();
        let r = FastReduce::reduce(&[], &[0], FastReduceOp::Sum, &cfg).unwrap();
        assert_eq!(r[0], 0.0);
    }

    #[test]
    fn test_reduce_validation_shape_mismatch() {
        let cfg = ReduceConfig::flat();
        assert!(FastReduce::reduce(&[1.0, 2.0], &[3], FastReduceOp::Sum, &cfg).is_err());
    }

    #[test]
    fn test_reduce_validation_axis_oob() {
        let cfg = ReduceConfig::new(Some(2), false, ReduceDtype::F32);
        assert!(
            FastReduce::reduce(&[1.0, 2.0, 3.0, 4.0], &[2, 2], FastReduceOp::Sum, &cfg).is_err()
        );
    }

    #[test]
    fn test_reduce_axis_l2norm() {
        let cfg = ReduceConfig::new(Some(1), false, ReduceDtype::F32);
        let data = vec![3.0, 4.0, 5.0, 12.0];
        let r = FastReduce::reduce(&data, &[2, 2], FastReduceOp::L2Norm, &cfg).unwrap();
        assert!(approx(r[0], 5.0, 1e-5));
        assert!(approx(r[1], 13.0, 1e-5));
    }

    #[test]
    fn test_reduce_axis_l1norm() {
        let cfg = ReduceConfig::new(Some(1), false, ReduceDtype::F32);
        let data = vec![-3.0, 4.0, -5.0, 12.0];
        let r = FastReduce::reduce(&data, &[2, 2], FastReduceOp::L1Norm, &cfg).unwrap();
        assert!(approx(r[0], 7.0, 1e-6));
        assert!(approx(r[1], 17.0, 1e-6));
    }

    // -- Property tests --------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_sum_matches_naive(data in proptest::collection::vec(-1e6f32..1e6, 0..512)) {
                let expected: f32 = data.iter().sum();
                let got = FastReduce::sum(&data);
                prop_assert!((got - expected).abs() <= expected.abs() * 1e-4 + 1e-6,
                    "sum mismatch: expected {expected}, got {got}");
            }

            #[test]
            fn prop_argmax_correct(data in proptest::collection::vec(-1e6f32..1e6, 1..256)) {
                let r = FastReduce::argmax(&data).unwrap();
                let naive_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                prop_assert!((r.value - naive_max).abs() < 1e-6);
                prop_assert_eq!(data[r.index], r.value);
            }

            #[test]
            fn prop_argmin_correct(data in proptest::collection::vec(-1e6f32..1e6, 1..256)) {
                let r = FastReduce::argmin(&data).unwrap();
                let naive_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                prop_assert!((r.value - naive_min).abs() < 1e-6);
                prop_assert_eq!(data[r.index], r.value);
            }

            #[test]
            fn prop_l2_norm_nonneg(data in proptest::collection::vec(-1e3f32..1e3, 0..256)) {
                let norm = FastReduce::l2_norm(&data);
                prop_assert!(norm >= 0.0, "L2 norm must be non-negative");
            }

            #[test]
            fn prop_l1_norm_nonneg(data in proptest::collection::vec(-1e3f32..1e3, 0..256)) {
                let norm = FastReduce::l1_norm(&data);
                prop_assert!(norm >= 0.0, "L1 norm must be non-negative");
            }

            #[test]
            fn prop_welford_mean_matches_naive(data in proptest::collection::vec(-1e3f32..1e3, 1..256)) {
                let r = FastReduce::welford(&data);
                let naive_mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                prop_assert!((r.mean - naive_mean).abs() <= naive_mean.abs() * 1e-3 + 1e-5,
                    "mean mismatch: welford={}, naive={}", r.mean, naive_mean);
            }

            #[test]
            fn prop_welford_variance_nonneg(data in proptest::collection::vec(-1e3f32..1e3, 2..256)) {
                let r = FastReduce::welford(&data);
                prop_assert!(r.variance >= 0.0, "variance must be non-negative, got {}", r.variance);
            }
        }
    }
}
