//! CPU SIMD-optimized softmax kernel.
//!
//! Provides numerically stable softmax computation on contiguous `f32`
//! slices with optional temperature scaling.  AVX2 acceleration is used
//! when available on x86_64; a scalar fallback is provided for all
//! platforms.
//!
//! Both 1-D (single vector) and 2-D (batch of equal-length rows)
//! variants are supported, each in allocating and in-place forms.

use bitnet_common::{BitNetError, KernelError, Result};

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Error helper ───────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Scalar implementation ──────────────────────────────────────────

/// Numerically stable softmax on a single slice (scalar path).
fn scalar_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut out: Vec<f32> = input.iter().map(|&x| ((x * inv_t) - max_val * inv_t).exp()).collect();

    let sum: f32 = out.iter().sum();
    let inv_sum = 1.0 / sum;
    for v in &mut out {
        *v *= inv_sum;
    }
    out
}

/// In-place numerically stable softmax (scalar path).
fn scalar_softmax_inplace(data: &mut [f32], temperature: f32) {
    let inv_t = 1.0 / temperature;
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = ((*v * inv_t) - max_val * inv_t).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in data.iter_mut() {
        *v *= inv_sum;
    }
}

// ── AVX2 implementation (x86_64 only) ──────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(hi, lo);
    let hi2 = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, hi2);
    let hi1 = _mm_shuffle_ps::<0x01>(sum2, sum2);
    _mm_cvtss_f32(_mm_add_ss(sum2, hi1))
}

/// Horizontal max of all 8 lanes in a `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let max4 = _mm_max_ps(hi, lo);
    let hi2 = _mm_movehl_ps(max4, max4);
    let max2 = _mm_max_ps(max4, hi2);
    let hi1 = _mm_shuffle_ps::<0x01>(max2, max2);
    _mm_cvtss_f32(_mm_max_ss(max2, hi1))
}

/// AVX2 single-register exp approximation (Cody-Waite + 6th-order
/// Taylor), matching the implementation in `simd_math.rs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_exp_ps(x: __m256) -> __m256 {
    let ln2_hi = _mm256_set1_ps(6.931_457_5e-1);
    let ln2_lo = _mm256_set1_ps(1.428_606_8e-6);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);
    let clamp_lo = _mm256_set1_ps(-87.3);
    let clamp_hi = _mm256_set1_ps(88.3);

    let xc = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);

    let n_i = _mm256_cvtps_epi32(_mm256_mul_ps(xc, log2e));
    let n_f = _mm256_cvtepi32_ps(n_i);
    let r =
        _mm256_sub_ps(_mm256_sub_ps(xc, _mm256_mul_ps(n_f, ln2_hi)), _mm256_mul_ps(n_f, ln2_lo));

    let p = c6;
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c5);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), half);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);

    let pow2n =
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_i, _mm256_set1_epi32(127))));
    _mm256_mul_ps(p, pow2n)
}

/// AVX2-accelerated softmax on a single f32 slice.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    unsafe {
        let len = input.len();
        let chunks = len / 8;
        let inv_t = _mm256_set1_ps(1.0 / temperature);

        // Pass 1: find max(input / temperature)
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let scaled = _mm256_mul_ps(v, inv_t);
            max_vec = _mm256_max_ps(max_vec, scaled);
        }
        let mut max_val = hmax_avx2(max_vec);
        let inv_t_scalar = 1.0 / temperature;
        for &x in &input[chunks * 8..] {
            let s = x * inv_t_scalar;
            if s > max_val {
                max_val = s;
            }
        }

        // Pass 2: compute exp(x/T - max) and accumulate sum
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        let mut out = vec![0.0f32; len];

        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(input.as_ptr().add(off));
            let scaled = _mm256_mul_ps(v, inv_t);
            let shifted = _mm256_sub_ps(scaled, max_broadcast);
            let e = avx2_exp_ps(shifted);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), e);
            sum_vec = _mm256_add_ps(sum_vec, e);
        }
        let mut sum = hsum_avx2(sum_vec);
        for (o, &x) in out[chunks * 8..].iter_mut().zip(&input[chunks * 8..]) {
            let e = (x * inv_t_scalar - max_val).exp();
            *o = e;
            sum += e;
        }

        // Pass 3: normalize
        let inv_sum = _mm256_set1_ps(1.0 / sum);
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(out.as_ptr().add(off));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), _mm256_mul_ps(v, inv_sum));
        }
        let inv_sum_scalar = 1.0 / sum;
        for v in &mut out[chunks * 8..] {
            *v *= inv_sum_scalar;
        }

        out
    }
}

/// AVX2-accelerated in-place softmax.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_softmax_inplace(data: &mut [f32], temperature: f32) {
    unsafe {
        let len = data.len();
        let chunks = len / 8;
        let inv_t = _mm256_set1_ps(1.0 / temperature);
        let inv_t_scalar = 1.0 / temperature;

        // Pass 1: find max
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
            max_vec = _mm256_max_ps(max_vec, _mm256_mul_ps(v, inv_t));
        }
        let mut max_val = hmax_avx2(max_vec);
        for &x in &data[chunks * 8..] {
            let s = x * inv_t_scalar;
            if s > max_val {
                max_val = s;
            }
        }

        // Pass 2: exp + sum
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(data.as_ptr().add(off));
            let scaled = _mm256_mul_ps(v, inv_t);
            let e = avx2_exp_ps(_mm256_sub_ps(scaled, max_broadcast));
            _mm256_storeu_ps(data.as_mut_ptr().add(off), e);
            sum_vec = _mm256_add_ps(sum_vec, e);
        }
        let mut sum = hsum_avx2(sum_vec);
        for v in &mut data[chunks * 8..] {
            let e = (*v * inv_t_scalar - max_val).exp();
            *v = e;
            sum += e;
        }

        // Pass 3: normalize
        let inv_sum = _mm256_set1_ps(1.0 / sum);
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(data.as_ptr().add(off));
            _mm256_storeu_ps(data.as_mut_ptr().add(off), _mm256_mul_ps(v, inv_sum));
        }
        let inv_sum_scalar = 1.0 / sum;
        for v in &mut data[chunks * 8..] {
            *v *= inv_sum_scalar;
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────

/// Compute softmax over a 1-D f32 slice, returning a new vector.
///
/// Uses AVX2 when available, scalar fallback otherwise.
/// Temperature divides logits before exponentiation (must be > 0).
///
/// # Errors
///
/// Returns `InvalidArguments` if `input` is empty or `temperature`
/// is not positive and finite.
pub fn softmax(input: &[f32], temperature: f32) -> Result<Vec<f32>> {
    validate_args(input.len(), temperature)?;
    Ok(dispatch_softmax(input, temperature))
}

/// Compute softmax in-place on a mutable f32 slice.
///
/// # Errors
///
/// Returns `InvalidArguments` if `data` is empty or `temperature`
/// is not positive and finite.
pub fn softmax_inplace(data: &mut [f32], temperature: f32) -> Result<()> {
    validate_args(data.len(), temperature)?;
    dispatch_softmax_inplace(data, temperature);
    Ok(())
}

/// Compute softmax over a batch of equal-length rows (2-D),
/// returning a flattened result vector.
///
/// `input` is a flat buffer of `batch_size * row_len` elements.
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatch, empty rows,
/// or invalid temperature.
pub fn softmax_batch(input: &[f32], row_len: usize, temperature: f32) -> Result<Vec<f32>> {
    validate_batch_args(input.len(), row_len, temperature)?;
    let batch_size = input.len() / row_len;
    let mut out = Vec::with_capacity(input.len());
    for b in 0..batch_size {
        let row = &input[b * row_len..(b + 1) * row_len];
        out.extend_from_slice(&dispatch_softmax(row, temperature));
    }
    Ok(out)
}

/// Compute softmax in-place over a batch of equal-length rows.
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatch, empty rows,
/// or invalid temperature.
pub fn softmax_batch_inplace(data: &mut [f32], row_len: usize, temperature: f32) -> Result<()> {
    validate_batch_args(data.len(), row_len, temperature)?;
    let batch_size = data.len() / row_len;
    for b in 0..batch_size {
        let row = &mut data[b * row_len..(b + 1) * row_len];
        dispatch_softmax_inplace(row, temperature);
    }
    Ok(())
}

// ── Internal dispatch & validation ─────────────────────────────────

fn validate_args(len: usize, temperature: f32) -> Result<()> {
    if len == 0 {
        return Err(invalid_args("softmax requires non-empty input"));
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(invalid_args("temperature must be positive and finite"));
    }
    Ok(())
}

fn validate_batch_args(total_len: usize, row_len: usize, temperature: f32) -> Result<()> {
    if row_len == 0 {
        return Err(invalid_args("row_len must be > 0"));
    }
    if total_len == 0 {
        return Err(invalid_args("softmax_batch requires non-empty input"));
    }
    if !total_len.is_multiple_of(row_len) {
        return Err(invalid_args("input length must be divisible by row_len"));
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(invalid_args("temperature must be positive and finite"));
    }
    Ok(())
}

#[inline]
fn dispatch_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_softmax(input, temperature) };
        }
    }
    scalar_softmax(input, temperature)
}

#[inline]
fn dispatch_softmax_inplace(data: &mut [f32], temperature: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2_softmax_inplace(data, temperature) };
            return;
        }
    }
    scalar_softmax_inplace(data, temperature);
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    /// Reference softmax computed with stdlib math (no SIMD).
    fn reference_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
        let inv_t = 1.0 / temperature;
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x * inv_t - max * inv_t).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── Basic correctness ──────────────────────────────────

    #[test]
    fn softmax_uniform_input() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let out = softmax(&input, 1.0).unwrap();
        assert!(approx_eq(&out, &[0.25, 0.25, 0.25, 0.25], TOL));
    }

    #[test]
    fn softmax_known_values() {
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax(&input, 1.0).unwrap();
        let expected = reference_softmax(&input, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_single_dominant() {
        let input = vec![0.0, 0.0, 10.0, 0.0];
        let out = softmax(&input, 1.0).unwrap();
        assert!(out[2] > 0.99);
    }

    #[test]
    fn softmax_agrees_with_reference() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let out = softmax(&input, 1.0).unwrap();
        let expected = reference_softmax(&input, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn softmax_negative_inputs() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let out = softmax(&input, 1.0).unwrap();
        let expected = reference_softmax(&input, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
        assert!(out[0] > out[1]);
        assert!(out[1] > out[2]);
    }

    // ── Numerical stability ────────────────────────────────

    #[test]
    fn softmax_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = softmax(&input, 1.0).unwrap();
        let expected = reference_softmax(&input, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_very_negative_values() {
        let input = vec![-1000.0, -999.0, -998.0];
        let out = softmax(&input, 1.0).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn softmax_mixed_extreme_values() {
        let input = vec![-500.0, 0.0, 500.0];
        let out = softmax(&input, 1.0).unwrap();
        assert!(out[2] > 0.99);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_no_nan_on_large_spread() {
        let input = vec![f32::MIN / 2.0, 0.0, f32::MAX / 2.0];
        let out = softmax(&input, 1.0).unwrap();
        for &v in &out {
            assert!(!v.is_nan(), "output contains NaN");
        }
    }

    // ── Temperature scaling ────────────────────────────────

    #[test]
    fn softmax_high_temperature_flattens() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out_t1 = softmax(&input, 1.0).unwrap();
        let out_t10 = softmax(&input, 10.0).unwrap();
        let range = |o: &[f32]| {
            o.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - o.iter().copied().fold(f32::INFINITY, f32::min)
        };
        assert!(range(&out_t10) < range(&out_t1), "high temperature should flatten distribution");
    }

    #[test]
    fn softmax_low_temperature_sharpens() {
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax(&input, 0.1).unwrap();
        assert!(out[2] > 0.99);
    }

    #[test]
    fn softmax_temperature_one_is_standard() {
        let input = vec![0.5, 1.5, 2.5];
        let out = softmax(&input, 1.0).unwrap();
        let expected = reference_softmax(&input, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn softmax_very_high_temperature_near_uniform() {
        let input = vec![1.0, 100.0, -50.0];
        let out = softmax(&input, 1e6).unwrap();
        for &v in &out {
            assert!((v - 1.0 / 3.0).abs() < 0.01, "expected near-uniform, got {v}");
        }
    }

    // ── Batch softmax ──────────────────────────────────────

    #[test]
    fn softmax_batch_two_rows() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = softmax_batch(&input, 3, 1.0).unwrap();
        let row0 = reference_softmax(&input[0..3], 1.0);
        let row1 = reference_softmax(&input[3..6], 1.0);
        assert!(approx_eq(&out[0..3], &row0, TOL));
        assert!(approx_eq(&out[3..6], &row1, TOL));
    }

    #[test]
    fn softmax_batch_inplace_matches_allocating() {
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let out = softmax_batch(&input, 3, 0.5).unwrap();
        let mut data = input.clone();
        softmax_batch_inplace(&mut data, 3, 0.5).unwrap();
        assert!(approx_eq(&data, &out, TOL));
    }

    #[test]
    fn softmax_batch_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = softmax_batch(&input, 4, 1.0).unwrap();
        let expected = softmax(&input, 1.0).unwrap();
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn softmax_batch_each_row_sums_to_one() {
        let input: Vec<f32> = (0..12).map(|i| i as f32 * 0.3 - 1.0).collect();
        let out = softmax_batch(&input, 4, 1.0).unwrap();
        for row in out.chunks(4) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < TOL, "row sum = {sum}");
        }
    }

    // ── In-place variants ──────────────────────────────────

    #[test]
    fn softmax_inplace_matches_allocating() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = softmax(&input, 1.0).unwrap();
        let mut data = input.clone();
        softmax_inplace(&mut data, 1.0).unwrap();
        assert!(approx_eq(&data, &out, TOL));
    }

    #[test]
    fn softmax_inplace_with_temperature() {
        let input = vec![0.5, 1.0, 1.5, 2.0];
        let out = softmax(&input, 0.7).unwrap();
        let mut data = input.clone();
        softmax_inplace(&mut data, 0.7).unwrap();
        assert!(approx_eq(&data, &out, TOL));
    }

    // ── Edge cases ─────────────────────────────────────────

    #[test]
    fn softmax_single_element() {
        let out = softmax(&[42.0], 1.0).unwrap();
        assert!(approx_eq(&out, &[1.0], TOL));
    }

    #[test]
    fn softmax_two_elements() {
        let out = softmax(&[0.0, 0.0], 1.0).unwrap();
        assert!(approx_eq(&out, &[0.5, 0.5], TOL));
    }

    #[test]
    fn softmax_empty_returns_error() {
        assert!(softmax(&[], 1.0).is_err());
    }

    #[test]
    fn softmax_inplace_empty_returns_error() {
        assert!(softmax_inplace(&mut [], 1.0).is_err());
    }

    #[test]
    fn softmax_zero_temperature_returns_error() {
        assert!(softmax(&[1.0, 2.0], 0.0).is_err());
    }

    #[test]
    fn softmax_negative_temperature_returns_error() {
        assert!(softmax(&[1.0, 2.0], -1.0).is_err());
    }

    #[test]
    fn softmax_infinite_temperature_returns_error() {
        assert!(softmax(&[1.0], f32::INFINITY).is_err());
    }

    #[test]
    fn softmax_nan_temperature_returns_error() {
        assert!(softmax(&[1.0], f32::NAN).is_err());
    }

    #[test]
    fn softmax_batch_mismatched_length_returns_error() {
        assert!(softmax_batch(&[1.0, 2.0, 3.0], 2, 1.0).is_err());
    }

    #[test]
    fn softmax_batch_zero_row_len_returns_error() {
        assert!(softmax_batch(&[1.0], 0, 1.0).is_err());
    }

    #[test]
    fn softmax_batch_empty_returns_error() {
        assert!(softmax_batch(&[], 3, 1.0).is_err());
    }

    // ── Various lengths (exercises SIMD + scalar tail) ─────

    #[test]
    fn softmax_various_lengths() {
        for &len in &[1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 100, 1024] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 5.0).collect();
            let out = softmax(&input, 1.0).unwrap();
            let expected = reference_softmax(&input, 1.0);
            assert!(approx_eq(&out, &expected, TOL), "mismatch at len={len}");
        }
    }

    // ── Property tests ─────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn finite_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(-100.0f32..100.0f32, min_len..=max_len)
        }

        proptest! {
            #[test]
            fn prop_softmax_sums_to_one(
                input in finite_f32_vec(1, 256)
            ) {
                let has_finite =
                    input.iter().any(|x| x.is_finite());
                prop_assume!(has_finite);
                if let Ok(out) = softmax(&input, 1.0) {
                    let sum: f32 = out.iter().sum();
                    prop_assert!(
                        (sum - 1.0).abs() < 1e-3,
                        "sum = {sum}"
                    );
                }
            }

            #[test]
            fn prop_softmax_outputs_in_unit_interval(
                input in finite_f32_vec(1, 256)
            ) {
                if let Ok(out) = softmax(&input, 1.0) {
                    for (i, &v) in out.iter().enumerate() {
                        prop_assert!(
                            (0.0..=1.0).contains(&v),
                            "out[{i}] = {v} not in [0,1]"
                        );
                    }
                }
            }

            #[test]
            fn prop_softmax_monotonicity(
                input in finite_f32_vec(2, 128)
            ) {
                if let Ok(out) = softmax(&input, 1.0) {
                    for i in 0..input.len() {
                        for j in (i + 1)..input.len() {
                            if input[i] < input[j] {
                                prop_assert!(
                                    out[i] <= out[j] + TOL,
                                    "monotonicity violated"
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn prop_softmax_inplace_matches_allocating(
                input in finite_f32_vec(1, 128)
            ) {
                if let Ok(expected) = softmax(&input, 1.0) {
                    let mut data = input.clone();
                    softmax_inplace(&mut data, 1.0).unwrap();
                    prop_assert!(
                        approx_eq(&data, &expected, TOL),
                        "in-place != allocating"
                    );
                }
            }

            #[test]
            fn prop_softmax_temperature_positive(
                input in finite_f32_vec(1, 64),
                t in 0.01f32..100.0,
            ) {
                if let Ok(out) = softmax(&input, t) {
                    let sum: f32 = out.iter().sum();
                    prop_assert!(
                        (sum - 1.0).abs() < 1e-3,
                        "sum={sum} t={t}"
                    );
                }
            }
        }
    }
}
