//! AVX2-optimized SIMD math operations for the CPU inference path.
//!
//! Provides fast vectorised approximations of common math functions
//! (`exp`, `tanh`, `sigmoid`) and vector arithmetic (`dot_product`,
//! `vector_add`).  Each public function performs runtime AVX2 detection
//! via [`is_x86_feature_detected!`] and falls back to a scalar
//! implementation on platforms without AVX2.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

// ── Scalar fallbacks (all platforms) ────────────────────────────────

fn scalar_exp(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.exp()).collect()
}

fn scalar_tanh(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

fn scalar_sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn scalar_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

fn scalar_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

fn scalar_vector_scale(data: &[f32], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v * scale).collect()
}

fn scalar_l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// ── AVX2 implementations (x86_64 only) ─────────────────────────────

/// Horizontal sum of all 8 lanes in a `__m256`.
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

/// Single-register AVX2 exp approximation using 6th-order Taylor
/// with Cody-Waite range reduction.  Preserves NaN and ±∞ semantics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_exp_ps(x: __m256) -> __m256 {
    // Cody-Waite split of ln(2) for accurate range reduction
    let ln2_hi = _mm256_set1_ps(6.931_457_5e-1);
    let ln2_lo = _mm256_set1_ps(1.428_606_8e-6);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c6 = _mm256_set1_ps(1.0 / 720.0);
    // Safe bounds: n must stay in [-126, 127] for normal f32 exponents
    let clamp_lo = _mm256_set1_ps(-87.3);
    let clamp_hi = _mm256_set1_ps(88.3);

    // Special-value masks (computed before clamping)
    let nan_mask = _mm256_cmp_ps::<{ _CMP_UNORD_Q }>(x, x);
    let pos_inf = _mm256_set1_ps(f32::INFINITY);
    let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
    let inf_mask = _mm256_cmp_ps::<{ _CMP_EQ_OQ }>(x, pos_inf);
    let ninf_mask = _mm256_cmp_ps::<{ _CMP_EQ_OQ }>(x, neg_inf);
    let zero = _mm256_setzero_ps();

    // Clamp to avoid over/underflow in the integer exponent
    let xc = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);

    // Range reduction: x = n·ln2 + r,  |r| ≤ ln2/2
    let n_i = _mm256_cvtps_epi32(_mm256_mul_ps(xc, log2e));
    let n_f = _mm256_cvtepi32_ps(n_i);
    let r =
        _mm256_sub_ps(_mm256_sub_ps(xc, _mm256_mul_ps(n_f, ln2_hi)), _mm256_mul_ps(n_f, ln2_lo));

    // 6th-order Taylor polynomial in Horner form:
    //   exp(r) ≈ 1 + r(1 + r(½ + r(⅙ + r(1/24 + r(1/120 + r/720)))))
    let p = c6;
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c5);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), half);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
    let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);

    // Reconstruct: exp(x) = poly · 2^n  (via float exponent bits)
    let pow2n =
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_i, _mm256_set1_epi32(127))));
    let result = _mm256_mul_ps(p, pow2n);

    // Patch special values: NaN→NaN, +∞→+∞, −∞→0
    let result = _mm256_blendv_ps(result, x, nan_mask);
    let result = _mm256_blendv_ps(result, pos_inf, inf_mask);
    _mm256_blendv_ps(result, zero, ninf_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_exp_f32(x: &[f32]) -> Vec<f32> {
    let len = x.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = avx2_exp_ps(v);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..len {
        out[i] = x[i].exp();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_sigmoid_f32(x: &[f32]) -> Vec<f32> {
    let len = x.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let one = _mm256_set1_ps(1.0);
            let neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
            let exp_neg = avx2_exp_ps(neg_v);
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));

            // Preserve NaN from original input
            let nan_mask = _mm256_cmp_ps::<{ _CMP_UNORD_Q }>(v, v);
            let sig = _mm256_blendv_ps(sig, v, nan_mask);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), sig);
        }
    }
    for i in (chunks * 8)..len {
        out[i] = 1.0 / (1.0 + (-x[i]).exp());
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_tanh_f32(x: &[f32]) -> Vec<f32> {
    let len = x.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let two = _mm256_set1_ps(2.0);
            let one = _mm256_set1_ps(1.0);

            // tanh(x) = 2·sigmoid(2x) − 1
            let neg_2x = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(v, two));
            let exp_neg = avx2_exp_ps(neg_2x);
            let sig2 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
            let tanh = _mm256_sub_ps(_mm256_mul_ps(two, sig2), one);

            let nan_mask = _mm256_cmp_ps::<{ _CMP_UNORD_Q }>(v, v);
            let tanh = _mm256_blendv_ps(tanh, v, nan_mask);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), tanh);
        }
    }
    for i in (chunks * 8)..len {
        out[i] = x[i].tanh();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(off));
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }
        let mut sum = hsum_avx2(acc);
        for i in (chunks * 8)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(off));
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), _mm256_add_ps(va, vb));
        }
    }
    for i in (chunks * 8)..len {
        out[i] = a[i] + b[i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(off));
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), _mm256_mul_ps(va, vb));
        }
    }
    for i in (chunks * 8)..len {
        out[i] = a[i] * b[i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_vector_scale(data: &[f32], scale: f32) -> Vec<f32> {
    let len = data.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / 8;

    unsafe {
        let vs = _mm256_set1_ps(scale);
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(data.as_ptr().add(off));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), _mm256_mul_ps(v, vs));
        }
    }
    for i in (chunks * 8)..len {
        out[i] = data[i] * scale;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_l2_norm(data: &[f32]) -> f32 {
    let len = data.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let off = i * 8;
            let v = _mm256_loadu_ps(data.as_ptr().add(off));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        }
        let mut sum = hsum_avx2(acc);
        for &v in &data[(chunks * 8)..] {
            sum += v * v;
        }
        sum.sqrt()
    }
}

// ── Public dispatch functions ───────────────────────────────────────

/// Fast vectorised exponential.  AVX2 when available, scalar otherwise.
pub fn fast_exp_f32(x: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: AVX2 confirmed available by runtime check.
            return unsafe { avx2_exp_f32(x) };
        }
    }
    scalar_exp(x)
}

/// Fast vectorised hyperbolic tangent.
pub fn fast_tanh_f32(x: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_tanh_f32(x) };
        }
    }
    scalar_tanh(x)
}

/// Fast vectorised sigmoid (logistic function).
pub fn fast_sigmoid_f32(x: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_sigmoid_f32(x) };
        }
    }
    scalar_sigmoid(x)
}

/// SIMD-accelerated dot product of two equal-length slices.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_dot_product(a, b) };
        }
    }
    scalar_dot_product(a, b)
}

/// SIMD-accelerated element-wise vector addition.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn simd_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vector add requires equal-length slices");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_vector_add(a, b) };
        }
    }
    scalar_vector_add(a, b)
}

/// SIMD-accelerated element-wise vector multiplication.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn simd_vector_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vector mul requires equal-length slices");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_vector_mul(a, b) };
        }
    }
    scalar_vector_mul(a, b)
}

/// SIMD-accelerated scalar multiplication of every element.
pub fn simd_vector_scale(data: &[f32], scale: f32) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_vector_scale(data, scale) };
        }
    }
    scalar_vector_scale(data, scale)
}

/// SIMD-accelerated L2 (Euclidean) norm.
pub fn simd_l2_norm(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_l2_norm(data) };
        }
    }
    scalar_l2_norm(data)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ── exp ──

    #[test]
    fn test_exp_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let result = fast_exp_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_exp_accuracy_wide_range() {
        let input: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.1).collect();
        let result = fast_exp_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            if e.is_finite() && *e > 1e-30 {
                let rel = ((r - e) / e).abs();
                assert!(rel < 1e-5, "exp({}) rel error {rel} at index {i}", input[i]);
            }
        }
    }

    #[test]
    fn test_exp_nan_propagation() {
        let input = vec![1.0, f32::NAN, 3.0];
        let result = fast_exp_f32(&input);
        assert!(!result[0].is_nan());
        assert!(result[1].is_nan(), "NaN must propagate through exp");
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_exp_infinity() {
        let input = vec![f32::INFINITY, f32::NEG_INFINITY];
        let result = fast_exp_f32(&input);
        assert!(result[0].is_infinite() && result[0] > 0.0, "exp(+inf) should be +inf");
        assert!(result[1] >= 0.0 && result[1] < 1e-30, "exp(-inf) should be ~0");
    }

    #[test]
    fn test_exp_subnormals() {
        let tiny = f32::MIN_POSITIVE * 0.5; // subnormal
        let input = vec![tiny, -tiny, 0.0];
        let result = fast_exp_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_exp_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let result = fast_exp_f32(&input);
            let expected: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
            assert_eq!(result.len(), len);
            for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
                if e.is_finite() && *e > 1e-30 {
                    let rel = ((r - e) / e).abs();
                    assert!(rel < 1e-5, "len={len} i={i} x={} rel err {rel}", input[i]);
                }
            }
        }
    }

    // ── tanh ──

    #[test]
    fn test_tanh_basic() {
        let input = vec![0.0, 1.0, -1.0, 5.0, -5.0];
        let result = fast_tanh_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_tanh_accuracy() {
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let result = fast_tanh_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        assert!(
            max_abs_error(&result, &expected) < 1e-5,
            "tanh max abs error: {}",
            max_abs_error(&result, &expected)
        );
    }

    #[test]
    fn test_tanh_nan_propagation() {
        let input = vec![0.0, f32::NAN, 1.0];
        let result = fast_tanh_f32(&input);
        assert!(result[1].is_nan(), "NaN must propagate through tanh");
    }

    #[test]
    fn test_tanh_infinity() {
        let input = vec![f32::INFINITY, f32::NEG_INFINITY];
        let result = fast_tanh_f32(&input);
        assert!((result[0] - 1.0).abs() < 1e-5, "tanh(+inf) should be 1");
        assert!((result[1] + 1.0).abs() < 1e-5, "tanh(-inf) should be -1");
    }

    #[test]
    fn test_tanh_subnormals() {
        let tiny = f32::MIN_POSITIVE * 0.5;
        let input = vec![tiny, -tiny];
        let result = fast_tanh_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_tanh_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.02 - 1.0).collect();
            let result = fast_tanh_f32(&input);
            assert_eq!(result.len(), len);
        }
    }

    // ── sigmoid ──

    #[test]
    fn test_sigmoid_basic() {
        let input = vec![0.0, 1.0, -1.0, 10.0, -10.0];
        let result = fast_sigmoid_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_sigmoid_accuracy() {
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let result = fast_sigmoid_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        assert!(
            max_abs_error(&result, &expected) < 1e-5,
            "sigmoid max abs error: {}",
            max_abs_error(&result, &expected)
        );
    }

    #[test]
    fn test_sigmoid_nan_propagation() {
        let input = vec![0.0, f32::NAN, 1.0];
        let result = fast_sigmoid_f32(&input);
        assert!(result[1].is_nan(), "NaN must propagate through sigmoid");
    }

    #[test]
    fn test_sigmoid_infinity() {
        let input = vec![f32::INFINITY, f32::NEG_INFINITY];
        let result = fast_sigmoid_f32(&input);
        assert!((result[0] - 1.0).abs() < 1e-5, "sigmoid(+inf) should be 1");
        assert!(result[1].abs() < 1e-5, "sigmoid(-inf) should be 0");
    }

    #[test]
    fn test_sigmoid_subnormals() {
        let tiny = f32::MIN_POSITIVE * 0.5;
        let input = vec![tiny, -tiny];
        let result = fast_sigmoid_f32(&input);
        let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        assert!(max_abs_error(&result, &expected) < 1e-5);
    }

    #[test]
    fn test_sigmoid_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.02 - 1.0).collect();
            let result = fast_sigmoid_f32(&input);
            assert_eq!(result.len(), len);
        }
    }

    // ── dot product ──

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_dot_product(&a, &b);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-5, "dot product: {result} vs {expected}");
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = vec![0.0; 16];
        let b = vec![1.0; 16];
        assert_eq!(simd_dot_product(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let mut a = vec![0.0; 16];
        let mut b = vec![0.0; 16];
        a[0] = 1.0;
        b[1] = 1.0;
        assert_eq!(simd_dot_product(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let result = simd_dot_product(&a, &b);
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let tol = expected.abs() * 1e-5 + 1e-5;
            assert!((result - expected).abs() < tol, "len={len}: {result} vs {expected}");
        }
    }

    // ── vector add ──

    #[test]
    fn test_vector_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let result = simd_vector_add(&a, &b);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_vector_add_negative() {
        let a = vec![-1.0, -2.0, 3.0];
        let b = vec![1.0, 2.0, -3.0];
        let result = simd_vector_add(&a, &b);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_add_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| -(i as f32)).collect();
            let result = simd_vector_add(&a, &b);
            assert_eq!(result.len(), len);
            assert!(result.iter().all(|&x| x == 0.0), "Failed for len={len}");
        }
    }

    // ── empty inputs ──

    #[test]
    fn test_empty_inputs() {
        assert!(fast_exp_f32(&[]).is_empty());
        assert!(fast_tanh_f32(&[]).is_empty());
        assert!(fast_sigmoid_f32(&[]).is_empty());
        assert_eq!(simd_dot_product(&[], &[]), 0.0);
        assert!(simd_vector_add(&[], &[]).is_empty());
        assert!(simd_vector_mul(&[], &[]).is_empty());
        assert!(simd_vector_scale(&[], 2.0).is_empty());
        assert_eq!(simd_l2_norm(&[]), 0.0);
    }

    // ── vector mul ──

    #[test]
    fn test_vector_mul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_vector_mul(&a, &b);
        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_vector_mul_zeros() {
        let a = vec![0.0; 16];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = simd_vector_mul(&a, &b);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_mul_identity() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let ones = vec![1.0; 16];
        let result = simd_vector_mul(&a, &ones);
        assert_eq!(result, a);
    }

    #[test]
    fn test_vector_mul_negative() {
        let a = vec![1.0, -2.0, 3.0, -4.0];
        let b = vec![-1.0, 2.0, -3.0, 4.0];
        let result = simd_vector_mul(&a, &b);
        assert_eq!(result, vec![-1.0, -4.0, -9.0, -16.0]);
    }

    #[test]
    fn test_vector_mul_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5).collect();
            let result = simd_vector_mul(&a, &b);
            let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
            assert_eq!(result.len(), len);
            assert!(max_abs_error(&result, &expected) < 1e-5, "Failed for len={len}");
        }
    }

    #[test]
    #[should_panic(expected = "vector mul requires equal-length slices")]
    fn test_vector_mul_length_mismatch() {
        simd_vector_mul(&[1.0, 2.0], &[3.0]);
    }

    // ── vector scale ──

    #[test]
    fn test_vector_scale_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = simd_vector_scale(&data, 3.0);
        assert_eq!(result, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_vector_scale_zero() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = simd_vector_scale(&data, 0.0);
        assert!(result.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vector_scale_identity() {
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let result = simd_vector_scale(&data, 1.0);
        assert_eq!(result, data);
    }

    #[test]
    fn test_vector_scale_negative() {
        let data = vec![1.0, -2.0, 3.0, -4.0];
        let result = simd_vector_scale(&data, -2.0);
        assert_eq!(result, vec![-2.0, 4.0, -6.0, 8.0]);
    }

    #[test]
    fn test_vector_scale_various_lengths() {
        for &len in &[0, 1, 7, 8, 15, 16, 100, 1024] {
            let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let result = simd_vector_scale(&data, 2.5);
            let expected: Vec<f32> = data.iter().map(|&v| v * 2.5).collect();
            assert_eq!(result.len(), len);
            assert!(max_abs_error(&result, &expected) < 1e-5, "Failed for len={len}");
        }
    }

    // ── l2 norm ──

    #[test]
    fn test_l2_norm_basic() {
        let data = vec![3.0, 4.0];
        let result = simd_l2_norm(&data);
        assert!((result - 5.0).abs() < 1e-5, "||[3,4]|| should be 5, got {result}");
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let data = vec![1.0, 0.0, 0.0];
        let result = simd_l2_norm(&data);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_zeros() {
        let data = vec![0.0; 16];
        assert_eq!(simd_l2_norm(&data), 0.0);
    }

    #[test]
    fn test_l2_norm_single_element() {
        assert!((simd_l2_norm(&[5.0]) - 5.0).abs() < 1e-5);
        assert!((simd_l2_norm(&[-5.0]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_various_lengths() {
        for &len in &[1, 7, 8, 15, 16, 100, 1024] {
            let data: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01).collect();
            let result = simd_l2_norm(&data);
            let expected = data.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let tol = expected.abs() * 1e-5 + 1e-5;
            assert!(
                (result - expected).abs() < tol,
                "Failed for len={len}: {result} vs {expected}"
            );
        }
    }
}
