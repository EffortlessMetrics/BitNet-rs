//! AVX2-optimized softmax kernels with runtime dispatch.
//!
//! Provides numerically-stable softmax variants (standard, batched,
//! log-softmax, temperature-scaled, masked, in-place) accelerated with
//! AVX2+FMA intrinsics on x86-64.  A scalar fallback is used on all other
//! targets or when AVX2 is unavailable at runtime.
//!
//! Numerical stability is achieved via max-subtraction before `exp` and
//! Kahan compensated summation for large vectors.
#![allow(unsafe_op_in_unsafe_fn)]

// ── Intrinsics imports ─────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Clamp-protected exp to avoid inf/NaN.
#[inline(always)]
fn safe_exp(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

/// Kahan-compensated summation for improved accuracy on large vectors.
#[inline]
fn kahan_sum(values: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in values {
        let y = v as f64 - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum as f32
}

// ── AVX2 horizontal reductions ─────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m = _mm_max_ps(lo, hi);
    let m2 = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    _mm_cvtss_f32(m1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    _mm_cvtss_f32(s1)
}

// ── Scalar core implementations ────────────────────────────────────────

fn softmax_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = safe_exp(x - max_val);
    }
    let sum = kahan_sum(output);
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

fn log_softmax_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| safe_exp(x - max_val)).collect();
    let sum = kahan_sum(&exps);
    let log_sum_exp = max_val + sum.ln();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x - log_sum_exp;
    }
}

fn softmax_temp_scalar(input: &[f32], output: &mut [f32], temperature: f32) {
    if input.is_empty() {
        return;
    }
    if temperature.abs() < 1e-7 {
        // One-hot at argmax
        let (max_idx, _) =
            input.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
        for o in output.iter_mut() {
            *o = 0.0;
        }
        output[max_idx] = 1.0;
        return;
    }
    let inv_t = 1.0 / temperature;
    let scaled: Vec<f32> = input.iter().map(|&x| x * inv_t).collect();
    softmax_scalar(&scaled, output);
}

fn softmax_masked_scalar(input: &[f32], mask: &[bool], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    let max_val = input
        .iter()
        .zip(mask.iter())
        .filter(|&(_, &m)| m)
        .map(|(&x, _)| x)
        .fold(f32::NEG_INFINITY, f32::max);

    for ((o, &x), &m) in output.iter_mut().zip(input.iter()).zip(mask.iter()) {
        *o = if m { safe_exp(x - max_val) } else { 0.0 };
    }
    let sum = kahan_sum(output);
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

// ── AVX2 core implementations ──────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2_inner(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let chunks = n / 8;
    let tail = chunks * 8;

    // Pass 1: find max
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(inp.add(i * 8)));
    }
    let mut max_val = hmax_avx2(vmax);
    for i in tail..n {
        max_val = max_val.max(*inp.add(i));
    }

    // Pass 2: exp(x - max), accumulate sum
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        // Scalar exp per lane for correctness
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), shifted);
        for b in &mut buf {
            *b = safe_exp(*b);
        }
        let exp_v = _mm256_loadu_ps(buf.as_ptr());
        _mm256_storeu_ps(outp.add(i * 8), exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx2(vsum);
    for i in tail..n {
        let e = safe_exp(*inp.add(i) - max_val);
        *outp.add(i) = e;
        sum_exp += e;
    }

    // Pass 3: normalize
    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(outp.add(i * 8));
            _mm256_storeu_ps(outp.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in tail..n {
            *outp.add(i) *= inv_s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_softmax_avx2_inner(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let chunks = n / 8;
    let tail = chunks * 8;

    // Pass 1: find max (AVX2)
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(inp.add(i * 8)));
    }
    let mut max_val = hmax_avx2(vmax);
    for i in tail..n {
        max_val = max_val.max(*inp.add(i));
    }

    // Pass 2: sum of exp(x - max)
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut sum_exp = 0.0f32;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), shifted);
        for b in &mut buf {
            *b = safe_exp(*b);
        }
        sum_exp += kahan_sum(&buf);
    }
    for i in tail..n {
        sum_exp += safe_exp(*inp.add(i) - max_val);
    }

    // Pass 3: x_i - max - ln(sum_exp)
    let log_sum_exp = max_val + sum_exp.ln();
    let vlog = _mm256_set1_ps(log_sum_exp);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        let r = _mm256_sub_ps(v, vlog);
        _mm256_storeu_ps(outp.add(i * 8), r);
    }
    for i in tail..n {
        *outp.add(i) = *inp.add(i) - log_sum_exp;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_temp_avx2_inner(input: &[f32], output: &mut [f32], temperature: f32) {
    if input.is_empty() {
        return;
    }
    if temperature.abs() < 1e-7 {
        softmax_temp_scalar(input, output, temperature);
        return;
    }
    let n = input.len();
    let inv_t = 1.0 / temperature;
    let vinv_t = _mm256_set1_ps(inv_t);
    let chunks = n / 8;
    let tail = chunks * 8;
    let inp = input.as_ptr();

    // Scale input by 1/temperature into a temporary buffer, then softmax
    let mut scaled = vec![0.0f32; n];
    let sp = scaled.as_mut_ptr();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(inp.add(i * 8));
        _mm256_storeu_ps(sp.add(i * 8), _mm256_mul_ps(v, vinv_t));
    }
    for i in tail..n {
        *sp.add(i) = *inp.add(i) * inv_t;
    }
    softmax_avx2_inner(&scaled, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_masked_avx2_inner(input: &[f32], mask: &[bool], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    // Pass 1: find max of unmasked elements
    let mut max_val = f32::NEG_INFINITY;
    for (i, &m) in mask.iter().enumerate().take(n) {
        if m {
            max_val = max_val.max(input[i]);
        }
    }
    if max_val == f32::NEG_INFINITY {
        // All masked out
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    // Pass 2: exp(x - max) for unmasked, 0 for masked
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let chunks = n / 8;
    let tail = chunks * 8;
    let vmax_bc = _mm256_set1_ps(max_val);
    let mut vsum = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let v = _mm256_loadu_ps(inp.add(base));
        let shifted = _mm256_sub_ps(v, vmax_bc);
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), shifted);
        for j in 0..8 {
            buf[j] = if mask[base + j] { safe_exp(buf[j]) } else { 0.0 };
        }
        let exp_v = _mm256_loadu_ps(buf.as_ptr());
        // Zero out masked lanes
        let mut mask_bits = [0.0f32; 8];
        for j in 0..8 {
            mask_bits[j] = if mask[base + j] { f32::from_bits(0xFFFF_FFFF) } else { 0.0 };
        }
        let vmask = _mm256_loadu_ps(mask_bits.as_ptr());
        let masked_exp = _mm256_and_ps(exp_v, vmask);
        _mm256_storeu_ps(outp.add(base), masked_exp);
        vsum = _mm256_add_ps(vsum, masked_exp);
    }
    let mut sum_exp = hsum_avx2(vsum);
    #[allow(clippy::needless_range_loop)]
    for i in tail..n {
        if mask[i] {
            let e = safe_exp(*inp.add(i) - max_val);
            *outp.add(i) = e;
            sum_exp += e;
        } else {
            *outp.add(i) = 0.0;
        }
    }

    // Pass 3: normalize
    if sum_exp > 0.0 {
        let inv = _mm256_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(outp.add(i * 8));
            _mm256_storeu_ps(outp.add(i * 8), _mm256_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in tail..n {
            *outp.add(i) *= inv_s;
        }
    }
}

// ── Public API with runtime dispatch ───────────────────────────────────

/// Numerically-stable softmax over a single row.
///
/// Uses AVX2+FMA when available, otherwise falls back to scalar with
/// Kahan summation.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
pub fn softmax_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "softmax_avx2: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime detection guarantees AVX2+FMA.
            unsafe {
                softmax_avx2_inner(input, output);
            }
            return;
        }
    }

    softmax_scalar(input, output);
}

/// Batched softmax: `rows` independent softmax operations, each of width
/// `cols`.
///
/// # Panics
///
/// Panics if `inputs.len() != rows * cols` or `outputs.len() != rows * cols`.
pub fn softmax_avx2_batch(inputs: &[f32], outputs: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(inputs.len(), rows * cols, "softmax_avx2_batch: inputs length mismatch");
    assert_eq!(outputs.len(), rows * cols, "softmax_avx2_batch: outputs length mismatch");

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        softmax_avx2(&inputs[start..end], &mut outputs[start..end]);
    }
}

/// Log-softmax: `log_softmax(x)_i = x_i − max − ln(Σ exp(x_j − max))`.
///
/// Uses AVX2+FMA for the max and subtraction passes when available.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
pub fn log_softmax_avx2(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "log_softmax_avx2: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                log_softmax_avx2_inner(input, output);
            }
            return;
        }
    }

    log_softmax_scalar(input, output);
}

/// Temperature-scaled softmax: `softmax(x / temperature)`.
///
/// When `temperature ≈ 0` the result is a one-hot vector at the argmax.
///
/// # Panics
///
/// Panics if `input.len() != output.len()`.
pub fn softmax_with_temperature(input: &[f32], output: &mut [f32], temperature: f32) {
    assert_eq!(input.len(), output.len(), "softmax_with_temperature: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                softmax_temp_avx2_inner(input, output, temperature);
            }
            return;
        }
    }

    softmax_temp_scalar(input, output, temperature);
}

/// Masked softmax: masked-out positions receive probability 0.
///
/// `mask[i] == true` means the position participates in the softmax.
///
/// # Panics
///
/// Panics if slice lengths do not match.
pub fn softmax_masked(input: &[f32], mask: &[bool], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "softmax_masked: input/output length mismatch");
    assert_eq!(input.len(), mask.len(), "softmax_masked: input/mask length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                softmax_masked_avx2_inner(input, mask, output);
            }
            return;
        }
    }

    softmax_masked_scalar(input, mask, output);
}

/// In-place softmax (convenience wrapper).
pub fn softmax_inplace(data: &mut [f32]) {
    let copy = data.to_vec();
    softmax_avx2(&copy, data);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    // ── Naive reference implementation ──────────────────────────────────

    fn naive_softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().map(|&e| e as f64).sum();
        exps.iter().map(|&e| e / sum as f32).collect()
    }

    fn naive_log_softmax(input: &[f32]) -> Vec<f32> {
        let sm = naive_softmax(input);
        sm.iter().map(|&p| p.ln()).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    // ── Basic correctness ──────────────────────────────────────────────

    #[test]
    fn test_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        softmax_avx2(&input, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, TOL);
    }

    #[test]
    fn test_softmax_matches_naive_small() {
        let input = [0.5, 1.5, -0.3, 2.1, 0.0];
        let mut output = vec![0.0; 5];
        softmax_avx2(&input, &mut output);
        assert_close(&output, &naive_softmax(&input), TOL);
    }

    #[test]
    fn test_softmax_matches_naive_8_elements() {
        // Exactly one AVX2 chunk
        let input = [1.0, -1.0, 0.5, 2.0, -0.5, 1.5, 0.0, 3.0];
        let mut output = vec![0.0; 8];
        softmax_avx2(&input, &mut output);
        assert_close(&output, &naive_softmax(&input), TOL);
    }

    #[test]
    fn test_softmax_matches_naive_16_elements() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let mut output = vec![0.0; 16];
        softmax_avx2(&input, &mut output);
        assert_close(&output, &naive_softmax(&input), TOL);
    }

    #[test]
    fn test_softmax_matches_naive_17_elements() {
        // Non-power-of-2, tests tail handling
        let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.2 - 1.5).collect();
        let mut output = vec![0.0; 17];
        softmax_avx2(&input, &mut output);
        assert_close(&output, &naive_softmax(&input), TOL);
    }

    #[test]
    fn test_softmax_matches_naive_large() {
        let input: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.01).sin()).collect();
        let mut output = vec![0.0; 1024];
        softmax_avx2(&input, &mut output);
        assert_close(&output, &naive_softmax(&input), TOL);
    }

    // ── Sum-to-one invariant ───────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        softmax_avx2(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
    }

    #[test]
    fn test_softmax_sums_to_one_large() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; 512];
        softmax_avx2(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn test_softmax_all_non_negative() {
        let input = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
        let mut output = vec![0.0; 7];
        softmax_avx2(&input, &mut output);
        for &v in &output {
            assert!(v >= 0.0, "negative probability: {v}");
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        softmax_avx2(&input, &mut output);
    }

    #[test]
    fn test_softmax_single_element() {
        let input = [42.0];
        let mut output = [0.0];
        softmax_avx2(&input, &mut output);
        assert!((output[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_softmax_two_elements() {
        let input = [0.0, 0.0];
        let mut output = [0.0; 2];
        softmax_avx2(&input, &mut output);
        assert!((output[0] - 0.5).abs() < TOL);
        assert!((output[1] - 0.5).abs() < TOL);
    }

    #[test]
    fn test_softmax_all_same_values() {
        let input = [3.0; 10];
        let mut output = vec![0.0; 10];
        softmax_avx2(&input, &mut output);
        for &v in &output {
            assert!((v - 0.1).abs() < TOL, "expected uniform 0.1, got {v}");
        }
    }

    #[test]
    fn test_softmax_all_zeros() {
        let input = [0.0; 5];
        let mut output = vec![0.0; 5];
        softmax_avx2(&input, &mut output);
        for &v in &output {
            assert!((v - 0.2).abs() < TOL, "expected uniform 0.2, got {v}");
        }
    }

    // ── Numerical stability ────────────────────────────────────────────

    #[test]
    fn test_softmax_large_positive_values() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = vec![0.0; 3];
        softmax_avx2(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite value");
    }

    #[test]
    fn test_softmax_large_negative_values() {
        let input = [-1000.0, -999.0, -998.0];
        let mut output = vec![0.0; 3];
        softmax_avx2(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax_mixed_extreme_values() {
        let input = [-100.0, 0.0, 100.0];
        let mut output = vec![0.0; 3];
        softmax_avx2(&input, &mut output);
        // The largest element should dominate
        assert!(output[2] > 0.99, "largest should dominate: {}", output[2]);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
    }

    #[test]
    fn test_softmax_very_large_spread() {
        // Ensure no NaN/inf even with huge spread
        let input = [-500.0, 0.0, 500.0];
        let mut output = vec![0.0; 3];
        softmax_avx2(&input, &mut output);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!((output.iter().sum::<f32>() - 1.0).abs() < TOL);
    }

    // ── Temperature scaling ────────────────────────────────────────────

    #[test]
    fn test_temperature_1_0_equals_standard() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out_std = vec![0.0; 4];
        let mut out_temp = vec![0.0; 4];
        softmax_avx2(&input, &mut out_std);
        softmax_with_temperature(&input, &mut out_temp, 1.0);
        assert_close(&out_std, &out_temp, TOL);
    }

    #[test]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        softmax_with_temperature(&input, &mut output, 0.1);
        // Low temp → distribution more peaked at max
        assert!(output[2] > 0.99, "low temp should sharpen: max = {}", output[2]);
    }

    #[test]
    fn test_temperature_high_flattens() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        softmax_with_temperature(&input, &mut output, 100.0);
        // High temp → near-uniform
        for &v in &output {
            assert!((v - 1.0 / 3.0).abs() < 0.01, "high temp should flatten: {v}");
        }
    }

    #[test]
    fn test_temperature_near_zero_is_onehot() {
        let input = [1.0, 5.0, 3.0];
        let mut output = vec![0.0; 3];
        softmax_with_temperature(&input, &mut output, 1e-8);
        assert!((output[1] - 1.0).abs() < TOL, "should be 1.0 at argmax");
        assert!(output[0].abs() < TOL);
        assert!(output[2].abs() < TOL);
    }

    #[test]
    fn test_temperature_0_5() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        softmax_with_temperature(&input, &mut output, 0.5);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        // More peaked than standard softmax
        let mut std_out = vec![0.0; 4];
        softmax_avx2(&input, &mut std_out);
        assert!(output[3] > std_out[3], "temp=0.5 should be more peaked at max");
    }

    #[test]
    fn test_temperature_2_0() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        softmax_with_temperature(&input, &mut output, 2.0);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        // Flatter than standard softmax
        let mut std_out = vec![0.0; 4];
        softmax_avx2(&input, &mut std_out);
        assert!(output[3] < std_out[3], "temp=2.0 should be flatter");
    }

    #[test]
    fn test_temperature_extreme_high() {
        let input = [0.0, 10.0, -10.0];
        let mut output = vec![0.0; 3];
        softmax_with_temperature(&input, &mut output, 1e6);
        // Effectively uniform
        for &v in &output {
            assert!((v - 1.0 / 3.0).abs() < 1e-4, "extreme high temp: {v}");
        }
    }

    // ── Masked softmax ─────────────────────────────────────────────────

    #[test]
    fn test_masked_all_true() {
        let input = [1.0, 2.0, 3.0];
        let mask = [true, true, true];
        let mut output = vec![0.0; 3];
        softmax_masked(&input, &mask, &mut output);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, TOL);
    }

    #[test]
    fn test_masked_single_unmasked() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, true, false];
        let mut output = vec![0.0; 3];
        softmax_masked(&input, &mask, &mut output);
        assert!(output[0].abs() < TOL);
        assert!((output[1] - 1.0).abs() < TOL);
        assert!(output[2].abs() < TOL);
    }

    #[test]
    fn test_masked_all_false() {
        let input = [1.0, 2.0, 3.0];
        let mask = [false, false, false];
        let mut output = vec![0.0; 3];
        softmax_masked(&input, &mask, &mut output);
        for &v in &output {
            assert!(v.abs() < TOL, "all-masked should give zeros: {v}");
        }
    }

    #[test]
    fn test_masked_alternating() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = [true, false, true, false, true, false];
        let mut output = vec![0.0; 6];
        softmax_masked(&input, &mask, &mut output);
        // Masked positions must be zero
        assert!(output[1].abs() < TOL);
        assert!(output[3].abs() < TOL);
        assert!(output[5].abs() < TOL);
        // Unmasked should sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "unmasked sum = {sum}");
    }

    #[test]
    fn test_masked_large_vector() {
        let n = 100;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mask: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
        let mut output = vec![0.0; n];
        softmax_masked(&input, &mask, &mut output);
        // Masked positions must be zero
        for i in 0..n {
            if !mask[i] {
                assert!(output[i].abs() < TOL, "masked idx {i} = {}", output[i]);
            }
        }
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    // ── Log-softmax ────────────────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        log_softmax_avx2(&input, &mut output);
        let expected = naive_log_softmax(&input);
        assert_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_log_softmax_exp_equals_softmax() {
        let input = [0.5, 1.5, -0.3, 2.1, 0.0];
        let mut log_out = vec![0.0; 5];
        let mut sm_out = vec![0.0; 5];
        log_softmax_avx2(&input, &mut log_out);
        softmax_avx2(&input, &mut sm_out);
        // exp(log_softmax) should equal softmax
        let exp_log: Vec<f32> = log_out.iter().map(|&x| x.exp()).collect();
        assert_close(&exp_log, &sm_out, 1e-5);
    }

    #[test]
    fn test_log_softmax_all_negative() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        log_softmax_avx2(&input, &mut output);
        for &v in &output {
            assert!(v <= 0.0, "log-softmax should be ≤ 0: {v}");
        }
    }

    #[test]
    fn test_log_softmax_stability() {
        let input = [1000.0, 1001.0, 1002.0];
        let mut output = vec![0.0; 3];
        log_softmax_avx2(&input, &mut output);
        assert!(output.iter().all(|v| v.is_finite()), "non-finite log-softmax");
        // exp(log_softmax) should sum to ~1
        let sum: f32 = output.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "exp sum = {sum}");
    }

    #[test]
    fn test_log_softmax_single_element() {
        let input = [5.0];
        let mut output = [0.0];
        log_softmax_avx2(&input, &mut output);
        assert!((output[0] - 0.0).abs() < TOL, "single-elem log-softmax = {}", output[0]);
    }

    #[test]
    fn test_log_softmax_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        log_softmax_avx2(&input, &mut output);
    }

    // ── Batch processing ───────────────────────────────────────────────

    #[test]
    fn test_batch_single_row() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        softmax_avx2_batch(&input, &mut output, 1, 3);
        let expected = naive_softmax(&input);
        assert_close(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_multiple_rows() {
        let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut outputs = vec![0.0; 6];
        softmax_avx2_batch(&inputs, &mut outputs, 2, 3);
        let row1 = naive_softmax(&[1.0, 2.0, 3.0]);
        let row2 = naive_softmax(&[4.0, 5.0, 6.0]);
        assert_close(&outputs[0..3], &row1, TOL);
        assert_close(&outputs[3..6], &row2, TOL);
    }

    #[test]
    fn test_batch_many_rows() {
        let rows = 32;
        let cols = 10;
        let inputs: Vec<f32> = (0..(rows * cols)).map(|i| (i as f32) * 0.05 - 5.0).collect();
        let mut outputs = vec![0.0; rows * cols];
        softmax_avx2_batch(&inputs, &mut outputs, rows, cols);
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let expected = naive_softmax(&inputs[start..end]);
            assert_close(&outputs[start..end], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_non_power_of_2_cols() {
        let rows = 4;
        let cols = 7; // non-power-of-2
        let inputs: Vec<f32> = (0..(rows * cols)).map(|i| (i as f32) * 0.3).collect();
        let mut outputs = vec![0.0; rows * cols];
        softmax_avx2_batch(&inputs, &mut outputs, rows, cols);
        for r in 0..rows {
            let s = r * cols;
            let expected = naive_softmax(&inputs[s..s + cols]);
            assert_close(&outputs[s..s + cols], &expected, TOL);
        }
    }

    #[test]
    fn test_batch_single_element_rows() {
        let inputs = [3.0, 5.0, -1.0];
        let mut outputs = vec![0.0; 3];
        softmax_avx2_batch(&inputs, &mut outputs, 3, 1);
        for &v in &outputs {
            assert!((v - 1.0).abs() < TOL, "single-elem row should be 1.0: {v}");
        }
    }

    // ── In-place softmax ───────────────────────────────────────────────

    #[test]
    fn test_softmax_inplace_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = naive_softmax(&data);
        softmax_inplace(&mut data);
        assert_close(&data, &expected, TOL);
    }

    #[test]
    fn test_softmax_inplace_empty() {
        let mut data: Vec<f32> = vec![];
        softmax_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_softmax_inplace_single() {
        let mut data = vec![99.0];
        softmax_inplace(&mut data);
        assert!((data[0] - 1.0).abs() < TOL);
    }

    // ── Kahan summation accuracy ───────────────────────────────────────

    #[test]
    fn test_kahan_sum_accuracy() {
        // Many small values: naive f32 sum loses precision
        let values: Vec<f32> = (0..10000).map(|_| 1e-4).collect();
        let result = kahan_sum(&values);
        assert!((result - 1.0).abs() < 1e-5, "kahan sum = {result}");
    }

    #[test]
    fn test_kahan_sum_empty() {
        assert!((kahan_sum(&[]) - 0.0).abs() < f32::EPSILON);
    }

    // ── Monotonicity / ordering ────────────────────────────────────────

    #[test]
    fn test_softmax_preserves_ordering() {
        let input = [1.0, 3.0, 2.0, 5.0, 4.0];
        let mut output = vec![0.0; 5];
        softmax_avx2(&input, &mut output);
        // Larger input → larger output
        assert!(output[3] > output[4]); // 5 > 4
        assert!(output[4] > output[1]); // 4 > 3
        assert!(output[1] > output[2]); // 3 > 2
        assert!(output[2] > output[0]); // 2 > 1
    }

    // ── Property-based tests ───────────────────────────────────────────

    #[cfg(test)]
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn softmax_sums_to_one(v in proptest::collection::vec(-50.0f32..50.0, 1..128)) {
                let mut output = vec![0.0; v.len()];
                softmax_avx2(&v, &mut output);
                let sum: f64 = output.iter().map(|&x| x as f64).sum();
                prop_assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
            }

            #[test]
            fn softmax_all_non_negative(v in proptest::collection::vec(-100.0f32..100.0, 1..64)) {
                let mut output = vec![0.0; v.len()];
                softmax_avx2(&v, &mut output);
                for &p in &output {
                    prop_assert!(p >= 0.0, "negative probability: {}", p);
                }
            }

            #[test]
            fn log_softmax_exp_matches_softmax(
                v in proptest::collection::vec(-20.0f32..20.0, 1..64)
            ) {
                let mut log_out = vec![0.0; v.len()];
                let mut sm_out = vec![0.0; v.len()];
                log_softmax_avx2(&v, &mut log_out);
                softmax_avx2(&v, &mut sm_out);
                let exp_log: Vec<f32> = log_out.iter().map(|&x| x.exp()).collect();
                for i in 0..v.len() {
                    prop_assert!(
                        (exp_log[i] - sm_out[i]).abs() < 1e-4,
                        "idx {}: exp(log_sm)={} vs sm={}",
                        i, exp_log[i], sm_out[i]
                    );
                }
            }

            #[test]
            fn temperature_1_equals_standard(
                v in proptest::collection::vec(-10.0f32..10.0, 1..32)
            ) {
                let mut out_std = vec![0.0; v.len()];
                let mut out_temp = vec![0.0; v.len()];
                softmax_avx2(&v, &mut out_std);
                softmax_with_temperature(&v, &mut out_temp, 1.0);
                for i in 0..v.len() {
                    prop_assert!(
                        (out_std[i] - out_temp[i]).abs() < 1e-5,
                        "idx {}: std={} vs temp={}",
                        i, out_std[i], out_temp[i]
                    );
                }
            }

            #[test]
            fn masked_zero_at_false_positions(
                v in proptest::collection::vec(-10.0f32..10.0, 1..32),
                mask_bits in proptest::collection::vec(proptest::bool::ANY, 1..32),
            ) {
                let n = v.len().min(mask_bits.len());
                let input = &v[..n];
                let mask = &mask_bits[..n];
                let mut output = vec![0.0; n];
                softmax_masked(input, mask, &mut output);
                for i in 0..n {
                    if !mask[i] {
                        prop_assert!(
                            output[i].abs() < 1e-7,
                            "masked idx {} = {}",
                            i, output[i]
                        );
                    }
                }
            }

            #[test]
            fn inplace_matches_outofplace(v in proptest::collection::vec(-50.0f32..50.0, 1..64)) {
                let mut inplace = v.clone();
                let mut outplace = vec![0.0; v.len()];
                softmax_inplace(&mut inplace);
                softmax_avx2(&v, &mut outplace);
                for i in 0..v.len() {
                    prop_assert!(
                        (inplace[i] - outplace[i]).abs() < 1e-6,
                        "idx {}: inplace={} vs outplace={}",
                        i, inplace[i], outplace[i]
                    );
                }
            }
        }
    }
}
