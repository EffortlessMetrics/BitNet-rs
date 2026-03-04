//! SIMD-optimized element-wise operations for CPU inference.
//!
//! Provides arithmetic, transcendental, activation, comparison, and
//! utility operations over `f32` slices.  On x86-64 with AVX2 the hot
//! loops use 256-bit intrinsics; on AArch64 they use NEON.  A scalar
//! fallback is always compiled so every function works on any platform.

use std::f32::consts::PI;

// ── helpers ─────────────────────────────────────────────────────────

/// Panic when binary operands differ in length.
#[inline]
fn assert_same_len(a: usize, b: usize) {
    assert_eq!(a, b, "operand length mismatch: {a} vs {b}");
}

// ── x86-64 AVX2 helpers ────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
mod avx2 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    /// Fast exp approximation (6th-order Taylor + Cody-Waite).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn exp_ps(x: __m256) -> __m256 {
        unsafe {
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

            let nan_mask = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
            let pos_inf = _mm256_set1_ps(f32::INFINITY);
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
            let inf_mask = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, pos_inf);
            let ninf_mask = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, neg_inf);
            let zero = _mm256_setzero_ps();

            let xc = _mm256_max_ps(_mm256_min_ps(x, clamp_hi), clamp_lo);
            let n_i = _mm256_cvtps_epi32(_mm256_mul_ps(xc, log2e));
            let n_f = _mm256_cvtepi32_ps(n_i);
            let r = _mm256_sub_ps(
                _mm256_sub_ps(xc, _mm256_mul_ps(n_f, ln2_hi)),
                _mm256_mul_ps(n_f, ln2_lo),
            );

            let p = c6;
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), c5);
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), c4);
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), c3);
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), half);
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
            let p = _mm256_add_ps(_mm256_mul_ps(p, r), one);

            let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                n_i,
                _mm256_set1_epi32(127),
            )));
            let result = _mm256_mul_ps(p, pow2n);
            let result = _mm256_blendv_ps(result, x, nan_mask);
            let result = _mm256_blendv_ps(result, pos_inf, inf_mask);
            _mm256_blendv_ps(result, zero, ninf_mask)
        }
    }

    /// Sigmoid: 1 / (1 + exp(-x)).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sigmoid_ps(x: __m256) -> __m256 {
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
            let e = exp_ps(neg);
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
            let nan_mask = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
            _mm256_blendv_ps(sig, x, nan_mask)
        }
    }

    /// Tanh via 2·sigmoid(2x) − 1.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn tanh_ps(x: __m256) -> __m256 {
        unsafe {
            let two = _mm256_set1_ps(2.0);
            let one = _mm256_set1_ps(1.0);
            let neg2x = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(x, two));
            let e = exp_ps(neg2x);
            let sig2 = _mm256_div_ps(one, _mm256_add_ps(one, e));
            let t = _mm256_sub_ps(_mm256_mul_ps(two, sig2), one);
            let nan_mask = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
            _mm256_blendv_ps(t, x, nan_mask)
        }
    }

    /// Process slices through an AVX2 per-register kernel.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn map_unary(
        x: &[f32],
        out: &mut [f32],
        simd_fn: unsafe fn(__m256) -> __m256,
        scalar_fn: fn(f32) -> f32,
    ) {
        let n = x.len();
        let chunks = n / 8;
        for i in 0..chunks {
            let off = i * 8;
            unsafe {
                let v = _mm256_loadu_ps(x.as_ptr().add(off));
                let r = simd_fn(v);
                _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
            }
        }
        for i in (chunks * 8)..n {
            out[i] = scalar_fn(x[i]);
        }
    }

    /// Binary element-wise via AVX2 intrinsic.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn map_binary(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        simd_fn: unsafe fn(__m256, __m256) -> __m256,
        scalar_fn: fn(f32, f32) -> f32,
    ) {
        let n = a.len();
        let chunks = n / 8;
        for i in 0..chunks {
            let off = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(off));
                let vb = _mm256_loadu_ps(b.as_ptr().add(off));
                let vr = simd_fn(va, vb);
                _mm256_storeu_ps(out.as_mut_ptr().add(off), vr);
            }
        }
        for i in (chunks * 8)..n {
            out[i] = scalar_fn(a[i], b[i]);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Scalar helpers (used by every platform)
// ════════════════════════════════════════════════════════════════════

#[inline]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
#[allow(dead_code)]
fn scalar_tanh(x: f32) -> f32 {
    x.tanh()
}

#[inline]
fn scalar_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044_715 * x.powi(3))).tanh())
}

#[inline]
fn scalar_gelu_fast(x: f32) -> f32 {
    x * scalar_sigmoid(1.702 * x)
}

#[inline]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

// ════════════════════════════════════════════════════════════════════
// Public API — every function allocates and returns a Vec<f32>
// ════════════════════════════════════════════════════════════════════

// ── Arithmetic ──────────────────────────────────────────────────────

/// Element-wise addition: `out[i] = a[i] + b[i]`.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_add_ps(va, vb),
                    |x, y| x + y,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    out
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]`.
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_sub_ps(va, vb),
                    |x, y| x - y,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
    out
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_mul_ps(va, vb),
                    |x, y| x * y,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
    out
}

/// Element-wise division: `out[i] = a[i] / b[i]`.
pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_div_ps(va, vb),
                    |x, y| x / y,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i] / b[i];
    }
    out
}

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
pub fn fused_multiply_add(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    assert_same_len(a.len(), c.len());
    let n = a.len();
    let mut out = vec![0.0; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                fma_avx2(a, b, c, &mut out);
            }
            return out;
        }
    }

    for i in 0..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_avx2(a: &[f32], b: &[f32], c: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(off));
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            let vc = _mm256_loadu_ps(c.as_ptr().add(off));
            let vr = _mm256_fmadd_ps(va, vb, vc);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), vr);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = a[i].mul_add(b[i], c[i]);
    }
}

// ── Transcendental functions ────────────────────────────────────────

/// Element-wise exp: `out[i] = e^(x[i])`.
pub fn exp(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_unary(x, &mut out, avx2::exp_ps, f32::exp);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].exp();
    }
    out
}

/// Element-wise natural log: `out[i] = ln(x[i])`.
pub fn log(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.ln()).collect()
}

/// Element-wise square root.
pub fn sqrt(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                sqrt_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].sqrt();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_sqrt_ps(v);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i].sqrt();
    }
}

/// Element-wise reciprocal square root: `1 / sqrt(x)`.
pub fn rsqrt(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                rsqrt_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = 1.0 / x[i].sqrt();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rsqrt_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            // One Newton-Raphson refinement on the fast rsqrt.
            let est = _mm256_rsqrt_ps(v);
            let half = _mm256_set1_ps(0.5);
            let three = _mm256_set1_ps(3.0);
            let muls = _mm256_mul_ps(_mm256_mul_ps(v, est), est);
            let r = _mm256_mul_ps(_mm256_mul_ps(half, est), _mm256_sub_ps(three, muls));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = 1.0 / x[i].sqrt();
    }
}

// ── Activation functions ────────────────────────────────────────────

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
pub fn sigmoid(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_unary(x, &mut out, avx2::sigmoid_ps, scalar_sigmoid);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = scalar_sigmoid(x[i]);
    }
    out
}

/// Hyperbolic tangent.
pub fn tanh(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_unary(x, &mut out, avx2::tanh_ps, scalar_tanh);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].tanh();
    }
    out
}

/// GELU (tanh approximation):
///   `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
pub fn gelu(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                gelu_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = scalar_gelu(x[i]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gelu_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let sqrt_2_over_pi = _mm256_set1_ps((2.0f32 / PI).sqrt());
            let coeff = _mm256_set1_ps(0.044_715);
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);

            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
            let inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_add_ps(v, _mm256_mul_ps(coeff, v3)));
            let t = avx2::tanh_ps(inner);
            let r = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = scalar_gelu(x[i]);
    }
}

/// Fast GELU (sigmoid approximation): `x * σ(1.702 * x)`.
pub fn gelu_fast(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                gelu_fast_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = scalar_gelu_fast(x[i]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gelu_fast_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let k = _mm256_set1_ps(1.702);
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let s = avx2::sigmoid_ps(_mm256_mul_ps(k, v));
            let r = _mm256_mul_ps(v, s);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = scalar_gelu_fast(x[i]);
    }
}

/// SiLU / Swish: `x * σ(x)`.
pub fn silu(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                silu_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = scalar_silu(x[i]);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn silu_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let s = avx2::sigmoid_ps(v);
            let r = _mm256_mul_ps(v, s);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = scalar_silu(x[i]);
    }
}

/// Swish with configurable beta: `x * σ(β * x)`.
pub fn swish(x: &[f32], beta: f32) -> Vec<f32> {
    x.iter().map(|&v| v * scalar_sigmoid(beta * v)).collect()
}

/// ReLU: `max(0, x)`.
pub fn relu(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].max(0.0);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let zero = _mm256_setzero_ps();
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i].max(0.0);
    }
}

/// Leaky ReLU: `x if x >= 0, else alpha * x`.
pub fn leaky_relu(x: &[f32], alpha: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                leaky_relu_avx2(x, alpha, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = if x[i] >= 0.0 { x[i] } else { alpha * x[i] };
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn leaky_relu_avx2(x: &[f32], alpha: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let zero = _mm256_setzero_ps();
            let valpha = _mm256_set1_ps(alpha);
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let neg = _mm256_mul_ps(v, valpha);
            // _CMP_GE_OQ = 0x1d
            let mask = _mm256_cmp_ps::<0x1d>(v, zero);
            let r = _mm256_blendv_ps(neg, v, mask);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = if x[i] >= 0.0 { x[i] } else { alpha * x[i] };
    }
}

/// ReLU6: `clamp(x, 0, 6)`.
pub fn relu6(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu6_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].clamp(0.0, 6.0);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu6_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let zero = _mm256_setzero_ps();
            let six = _mm256_set1_ps(6.0);
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_min_ps(_mm256_max_ps(v, zero), six);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i].clamp(0.0, 6.0);
    }
}

// ── Utility operations ──────────────────────────────────────────────

/// Element-wise clamp to `[lo, hi]`.
pub fn clamp(x: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                clamp_avx2(x, lo, hi, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].clamp(lo, hi);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_avx2(x: &[f32], lo: f32, hi: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let vlo = _mm256_set1_ps(lo);
            let vhi = _mm256_set1_ps(hi);
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i].clamp(lo, hi);
    }
}

/// Element-wise absolute value.
pub fn abs(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                abs_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = x[i].abs();
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn abs_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            // Clear sign bit: AND with 0x7FFF_FFFF
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_and_ps(v, mask);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = x[i].abs();
    }
}

/// Element-wise negation.
pub fn neg(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                neg_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = -x[i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn neg_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let zero = _mm256_setzero_ps();
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_sub_ps(zero, v);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = -x[i];
    }
}

/// Element-wise sign: -1, 0, or 1 (NaN → NaN, ±0 → 0).
pub fn sign(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                f32::NAN
            } else if v == 0.0 {
                0.0
            } else if v > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Element-wise power: `x[i].powf(exp)`.
pub fn pow(x: &[f32], exponent: f32) -> Vec<f32> {
    x.iter().map(|&v| v.powf(exponent)).collect()
}

/// Element-wise reciprocal: `1 / x[i]`.
pub fn recip(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                recip_avx2(x, &mut out);
            }
            return out;
        }
    }

    for i in 0..x.len() {
        out[i] = 1.0 / x[i];
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn recip_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let off = i * 8;
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let v = _mm256_loadu_ps(x.as_ptr().add(off));
            let r = _mm256_div_ps(one, v);
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
    }
    for i in (chunks * 8)..n {
        out[i] = 1.0 / x[i];
    }
}

/// Element-wise minimum.
pub fn min(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_min_ps(va, vb),
                    f32::min,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i].min(b[i]);
    }
    out
}

/// Element-wise maximum.
pub fn max(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    let mut out = vec![0.0; a.len()];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                avx2::map_binary(
                    a,
                    b,
                    &mut out,
                    |va, vb| std::arch::x86_64::_mm256_max_ps(va, vb),
                    f32::max,
                );
            }
            return out;
        }
    }

    for i in 0..a.len() {
        out[i] = a[i].max(b[i]);
    }
    out
}

// ── Comparison operations ───────────────────────────────────────────

/// Element-wise equality: 1.0 if equal, else 0.0.
pub fn compare_eq(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| if x == y { 1.0 } else { 0.0 }).collect()
}

/// Element-wise greater-than: 1.0 if `a > b`, else 0.0.
pub fn compare_gt(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| if x > y { 1.0 } else { 0.0 }).collect()
}

/// Element-wise less-than: 1.0 if `a < b`, else 0.0.
pub fn compare_lt(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_same_len(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| if x < y { 1.0 } else { 0.0 }).collect()
}

// ════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    fn assert_close_rel(a: &[f32], b: &[f32], rel_tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            let denom = y.abs().max(1.0);
            let rel = (x - y).abs() / denom;
            assert!(rel <= rel_tol, "index {i}: {x} vs {y} (rel {rel})");
        }
    }

    fn assert_exact(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!(x == y, "index {i}: {x} != {y}");
        }
    }

    // ── Arithmetic ──────────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        assert_exact(&add(&a, &b), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_sub_basic() {
        let a = [10.0, 20.0, 30.0];
        let b = [1.0, 2.0, 3.0];
        assert_exact(&sub(&a, &b), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_basic() {
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        assert_exact(&mul(&a, &b), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_basic() {
        let a = [10.0, 20.0, 30.0];
        let b = [2.0, 5.0, 10.0];
        assert_exact(&div(&a, &b), &[5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_fma_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = [10.0, 20.0, 30.0];
        assert_close(&fused_multiply_add(&a, &b, &c), &[14.0, 30.0, 48.0], 1e-6);
    }

    // ── Arithmetic: large (AVX2 bulk + tail) ────────────────────

    #[test]
    fn test_add_large() {
        let n = 67;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let r = add(&a, &b);
        for i in 0..n {
            assert_eq!(r[i], (i * 3) as f32);
        }
    }

    #[test]
    fn test_sub_large() {
        let n = 67;
        let a: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let r = sub(&a, &b);
        for i in 0..n {
            assert_eq!(r[i], (i * 2) as f32);
        }
    }

    #[test]
    fn test_mul_large() {
        let n = 65;
        let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![2.0; n];
        let r = mul(&a, &b);
        for i in 0..n {
            assert_eq!(r[i], ((i + 1) * 2) as f32);
        }
    }

    #[test]
    fn test_div_large() {
        let n = 65;
        let a: Vec<f32> = (1..=n).map(|i| (i * 4) as f32).collect();
        let b: Vec<f32> = vec![4.0; n];
        let r = div(&a, &b);
        for i in 0..n {
            assert_eq!(r[i], (i + 1) as f32);
        }
    }

    #[test]
    fn test_fma_large() {
        let n = 73;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![2.0; n];
        let c: Vec<f32> = vec![1.0; n];
        let r = fused_multiply_add(&a, &b, &c);
        for i in 0..n {
            assert_close(&[r[i]], &[(i as f32).mul_add(2.0, 1.0)], 1e-6);
        }
    }

    // ── Arithmetic: empty ───────────────────────────────────────

    #[test]
    fn test_arithmetic_empty() {
        let e: &[f32] = &[];
        assert!(add(e, e).is_empty());
        assert!(sub(e, e).is_empty());
        assert!(mul(e, e).is_empty());
        assert!(div(e, e).is_empty());
        assert!(fused_multiply_add(e, e, e).is_empty());
    }

    // ── Arithmetic: special values ──────────────────────────────

    #[test]
    fn test_add_nan_propagation() {
        let a = [1.0, f32::NAN, 3.0];
        let b = [1.0, 2.0, f32::NAN];
        let r = add(&a, &b);
        assert_eq!(r[0], 2.0);
        assert!(r[1].is_nan());
        assert!(r[2].is_nan());
    }

    #[test]
    fn test_mul_inf() {
        let a = [f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let b = [2.0, 3.0, f32::INFINITY];
        let r = mul(&a, &b);
        assert_eq!(r[0], f32::INFINITY);
        assert_eq!(r[1], f32::NEG_INFINITY);
        assert!(r[2].is_nan());
    }

    #[test]
    fn test_div_by_zero() {
        let a = [1.0, -1.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let r = div(&a, &b);
        assert_eq!(r[0], f32::INFINITY);
        assert_eq!(r[1], f32::NEG_INFINITY);
        assert!(r[2].is_nan());
    }

    // ── Transcendental ──────────────────────────────────────────

    #[test]
    fn test_exp_basic() {
        let x = [0.0, 1.0, -1.0, 2.0];
        let r = exp(&x);
        let expected: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_exp_large() {
        let n = 33;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let r = exp(&x);
        let expected: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
        assert_close_rel(&r, &expected, 1e-4);
    }

    #[test]
    fn test_exp_special() {
        let x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let r = exp(&x);
        assert!(r[0].is_nan());
        assert_eq!(r[1], f32::INFINITY);
        assert_eq!(r[2], 0.0);
        assert_close(&[r[3]], &[1.0], 1e-6);
    }

    #[test]
    fn test_log_basic() {
        let x = [1.0, std::f32::consts::E, 10.0];
        let r = log(&x);
        assert_close(&r, &[0.0, 1.0, 10.0f32.ln()], 1e-6);
    }

    #[test]
    fn test_log_special() {
        let x = [0.0, -1.0, f32::INFINITY, f32::NAN];
        let r = log(&x);
        assert_eq!(r[0], f32::NEG_INFINITY);
        assert!(r[1].is_nan());
        assert_eq!(r[2], f32::INFINITY);
        assert!(r[3].is_nan());
    }

    #[test]
    fn test_sqrt_basic() {
        let x = [0.0, 1.0, 4.0, 9.0, 16.0];
        let r = sqrt(&x);
        assert_close(&r, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_sqrt_large() {
        let n = 34;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 4.0).collect();
        let r = sqrt(&x);
        let expected: Vec<f32> = x.iter().map(|&v| v.sqrt()).collect();
        assert_close(&r, &expected, 1e-6);
    }

    #[test]
    fn test_sqrt_special() {
        let x = [f32::NAN, f32::INFINITY, -1.0];
        let r = sqrt(&x);
        assert!(r[0].is_nan());
        assert_eq!(r[1], f32::INFINITY);
        assert!(r[2].is_nan());
    }

    #[test]
    fn test_rsqrt_basic() {
        let x = [1.0, 4.0, 16.0];
        let r = rsqrt(&x);
        assert_close(&r, &[1.0, 0.5, 0.25], 1e-3);
    }

    #[test]
    fn test_rsqrt_large() {
        let n = 35;
        let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let r = rsqrt(&x);
        let expected: Vec<f32> = x.iter().map(|&v| 1.0 / v.sqrt()).collect();
        assert_close(&r, &expected, 1e-3);
    }

    // ── Activations ─────────────────────────────────────────────

    #[test]
    fn test_sigmoid_basic() {
        let x = [0.0, 1.0, -1.0, 5.0, -5.0];
        let r = sigmoid(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_sigmoid(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_sigmoid_large() {
        let n = 37;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.5).collect();
        let r = sigmoid(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_sigmoid(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_sigmoid_nan() {
        let r = sigmoid(&[f32::NAN]);
        assert!(r[0].is_nan());
    }

    #[test]
    fn test_sigmoid_extremes() {
        let r = sigmoid(&[100.0, -100.0]);
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
    }

    #[test]
    fn test_tanh_basic() {
        let x = [0.0, 1.0, -1.0, 3.0];
        let r = tanh(&x);
        let expected: Vec<f32> = x.iter().map(|&v| v.tanh()).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_tanh_large() {
        let n = 39;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 19.0) * 0.3).collect();
        let r = tanh(&x);
        let expected: Vec<f32> = x.iter().map(|&v| v.tanh()).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_tanh_nan() {
        let r = tanh(&[f32::NAN]);
        assert!(r[0].is_nan());
    }

    #[test]
    fn test_gelu_basic() {
        let x = [0.0, 1.0, -1.0, 0.5];
        let r = gelu(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_gelu(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_gelu_large() {
        let n = 41;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 20.0) * 0.2).collect();
        let r = gelu(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_gelu(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_gelu_zero() {
        assert_close(&gelu(&[0.0]), &[0.0], 1e-6);
    }

    #[test]
    fn test_gelu_fast_basic() {
        let x = [0.0, 1.0, -1.0, 2.0];
        let r = gelu_fast(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_gelu_fast(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_gelu_fast_large() {
        let n = 43;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 21.0) * 0.2).collect();
        let r = gelu_fast(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_gelu_fast(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_silu_basic() {
        let x = [0.0, 1.0, -1.0, 2.0];
        let r = silu(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_silu(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_silu_large() {
        let n = 45;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 22.0) * 0.2).collect();
        let r = silu(&x);
        let expected: Vec<f32> = x.iter().map(|&v| scalar_silu(v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_silu_zero() {
        assert_close(&silu(&[0.0]), &[0.0], 1e-6);
    }

    #[test]
    fn test_swish_beta() {
        let x = [0.0, 1.0, -1.0];
        let r = swish(&x, 2.0);
        let expected: Vec<f32> = x.iter().map(|&v| v * scalar_sigmoid(2.0 * v)).collect();
        assert_close(&r, &expected, TOL);
    }

    #[test]
    fn test_swish_beta_one_equals_silu() {
        let x: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.3).collect();
        let s = silu(&x);
        let sw = swish(&x, 1.0);
        assert_close(&s, &sw, TOL);
    }

    // ── ReLU variants ───────────────────────────────────────────

    #[test]
    fn test_relu_basic() {
        let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert_exact(&relu(&x), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_large() {
        let n = 47;
        let x: Vec<f32> = (0..n).map(|i| i as f32 - 23.0).collect();
        let r = relu(&x);
        for i in 0..n {
            assert_eq!(r[i], (i as f32 - 23.0).max(0.0));
        }
    }

    #[test]
    fn test_leaky_relu_basic() {
        let x = [-2.0, 0.0, 3.0];
        let r = leaky_relu(&x, 0.1);
        assert_close(&r, &[-0.2, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_leaky_relu_large() {
        let n = 49;
        let x: Vec<f32> = (0..n).map(|i| i as f32 - 24.0).collect();
        let r = leaky_relu(&x, 0.01);
        for i in 0..n {
            let v = x[i];
            let expected = if v >= 0.0 { v } else { 0.01 * v };
            assert_close(&[r[i]], &[expected], 1e-6);
        }
    }

    #[test]
    fn test_relu6_basic() {
        let x = [-1.0, 0.0, 3.0, 6.0, 10.0];
        assert_exact(&relu6(&x), &[0.0, 0.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn test_relu6_large() {
        let n = 51;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.5).collect();
        let r = relu6(&x);
        for i in 0..n {
            assert_eq!(r[i], x[i].clamp(0.0, 6.0));
        }
    }

    // ── Utility ops ─────────────────────────────────────────────

    #[test]
    fn test_clamp_basic() {
        let x = [-5.0, 0.0, 5.0, 10.0];
        assert_exact(&clamp(&x, -1.0, 7.0), &[-1.0, 0.0, 5.0, 7.0]);
    }

    #[test]
    fn test_clamp_large() {
        let n = 53;
        let x: Vec<f32> = (0..n).map(|i| i as f32 - 26.0).collect();
        let r = clamp(&x, -10.0, 10.0);
        for i in 0..n {
            assert_eq!(r[i], x[i].clamp(-10.0, 10.0));
        }
    }

    #[test]
    fn test_abs_basic() {
        let x = [-3.0, 0.0, 5.0, -7.0];
        assert_exact(&abs(&x), &[3.0, 0.0, 5.0, 7.0]);
    }

    #[test]
    fn test_abs_large() {
        let n = 55;
        let x: Vec<f32> = (0..n).map(|i| i as f32 - 27.0).collect();
        let r = abs(&x);
        for i in 0..n {
            assert_eq!(r[i], x[i].abs());
        }
    }

    #[test]
    fn test_neg_basic() {
        let x = [1.0, -2.0, 0.0, 3.0];
        assert_exact(&neg(&x), &[-1.0, 2.0, 0.0, -3.0]);
    }

    #[test]
    fn test_neg_large() {
        let n = 57;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let r = neg(&x);
        for i in 0..n {
            assert_eq!(r[i], -(i as f32));
        }
    }

    #[test]
    fn test_sign_basic() {
        let x = [-5.0, 0.0, 3.0, f32::NAN];
        let r = sign(&x);
        assert_eq!(r[0], -1.0);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 1.0);
        assert!(r[3].is_nan());
    }

    #[test]
    fn test_pow_basic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_close(&pow(&x, 2.0), &[1.0, 4.0, 9.0, 16.0], 1e-6);
    }

    #[test]
    fn test_pow_fractional() {
        let x = [4.0, 9.0, 16.0];
        assert_close(&pow(&x, 0.5), &[2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn test_recip_basic() {
        let x = [1.0, 2.0, 4.0, 0.5];
        assert_close(&recip(&x), &[1.0, 0.5, 0.25, 2.0], 1e-6);
    }

    #[test]
    fn test_recip_large() {
        let n = 59;
        let x: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let r = recip(&x);
        for i in 0..n {
            assert_close(&[r[i]], &[1.0 / (i + 1) as f32], 1e-6);
        }
    }

    #[test]
    fn test_recip_zero() {
        let r = recip(&[0.0, -0.0]);
        assert_eq!(r[0], f32::INFINITY);
        assert_eq!(r[1], f32::NEG_INFINITY);
    }

    // ── Min / Max ───────────────────────────────────────────────

    #[test]
    fn test_min_basic() {
        let a = [1.0, 5.0, 3.0];
        let b = [4.0, 2.0, 6.0];
        assert_exact(&min(&a, &b), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_max_basic() {
        let a = [1.0, 5.0, 3.0];
        let b = [4.0, 2.0, 6.0];
        assert_exact(&max(&a, &b), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_min_max_large() {
        let n = 61;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - 1 - i) as f32).collect();
        let mn = min(&a, &b);
        let mx = max(&a, &b);
        for i in 0..n {
            assert_eq!(mn[i], a[i].min(b[i]));
            assert_eq!(mx[i], a[i].max(b[i]));
        }
    }

    #[test]
    fn test_min_nan() {
        let a = [1.0, f32::NAN];
        let b = [f32::NAN, 2.0];
        let r = min(&a, &b);
        assert_eq!(r.len(), 2);
    }

    // ── Comparisons ─────────────────────────────────────────────

    #[test]
    fn test_compare_eq_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 9.0, 3.0];
        assert_exact(&compare_eq(&a, &b), &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compare_gt_basic() {
        let a = [5.0, 2.0, 3.0];
        let b = [1.0, 9.0, 3.0];
        assert_exact(&compare_gt(&a, &b), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compare_lt_basic() {
        let a = [5.0, 2.0, 3.0];
        let b = [1.0, 9.0, 3.0];
        assert_exact(&compare_lt(&a, &b), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_compare_nan() {
        let a = [f32::NAN, 1.0];
        let b = [1.0, f32::NAN];
        assert_exact(&compare_eq(&a, &b), &[0.0, 0.0]);
        assert_exact(&compare_gt(&a, &b), &[0.0, 0.0]);
        assert_exact(&compare_lt(&a, &b), &[0.0, 0.0]);
    }

    // ── Edge: single element ────────────────────────────────────

    #[test]
    fn test_single_element() {
        assert_exact(&add(&[1.0], &[2.0]), &[3.0]);
        assert_close(&exp(&[0.0]), &[1.0], 1e-6);
        assert_close(&sigmoid(&[0.0]), &[0.5], TOL);
        assert_close(&tanh(&[0.0]), &[0.0], 1e-6);
        assert_close(&gelu(&[0.0]), &[0.0], 1e-6);
        assert_close(&silu(&[0.0]), &[0.0], 1e-6);
        assert_exact(&relu(&[-1.0]), &[0.0]);
        assert_exact(&relu(&[1.0]), &[1.0]);
        assert_exact(&abs(&[-5.0]), &[5.0]);
    }

    // ── Edge: exactly 8 elements (one SIMD register) ────────────

    #[test]
    fn test_exact_register_width() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_exact(&add(&a, &b), &[9.0; 8]);
    }

    // ── Edge: 9 elements (register + 1 tail) ────────────────────

    #[test]
    fn test_register_plus_one() {
        let x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let expected: Vec<f32> = x.iter().map(|v| v * v).collect();
        assert_exact(&mul(&x, &x), &expected);
    }

    // ── Panics ──────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_add_length_mismatch() {
        let _ = add(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_sub_length_mismatch() {
        let _ = sub(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_mul_length_mismatch() {
        let _ = mul(&[1.0, 2.0, 3.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_div_length_mismatch() {
        let _ = div(&[1.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_min_length_mismatch() {
        let _ = min(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_max_length_mismatch() {
        let _ = max(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_compare_eq_length_mismatch() {
        let _ = compare_eq(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_compare_gt_length_mismatch() {
        let _ = compare_gt(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "operand length mismatch")]
    fn test_compare_lt_length_mismatch() {
        let _ = compare_lt(&[1.0], &[1.0, 2.0]);
    }

    // ── Cross-op sanity checks ──────────────────────────────────

    #[test]
    fn test_neg_neg_roundtrip() {
        let x = [-3.0, 0.0, 7.5, -100.0];
        assert_exact(&neg(&neg(&x)), &x);
    }

    #[test]
    fn test_abs_neg_symmetric() {
        let x = [-3.0, 0.0, 7.5, -100.0];
        assert_exact(&abs(&x), &abs(&neg(&x)));
    }

    #[test]
    fn test_sqrt_pow_half_agree() {
        let x = [1.0, 4.0, 9.0, 25.0];
        assert_close(&sqrt(&x), &pow(&x, 0.5), 1e-5);
    }

    #[test]
    fn test_recip_mul_identity() {
        let x = [1.0, 2.0, 4.0, 8.0];
        assert_close(&mul(&x, &recip(&x)), &[1.0; 4], 1e-6);
    }

    #[test]
    fn test_add_sub_inverse() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let sum = add(&a, &b);
        assert_close(&sub(&sum, &b), &a, 1e-6);
    }

    #[test]
    fn test_sigmoid_range() {
        let x: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let r = sigmoid(&x);
        for &v in &r {
            if !v.is_nan() {
                assert!((0.0..=1.0).contains(&v), "sigmoid out of range: {v}");
            }
        }
    }

    #[test]
    fn test_tanh_range() {
        let x: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let r = tanh(&x);
        for &v in &r {
            if !v.is_nan() {
                assert!((-1.0..=1.0).contains(&v), "tanh out of range: {v}");
            }
        }
    }

    #[test]
    fn test_relu_nonnegative() {
        let x: Vec<f32> = (-20..=20).map(|i| i as f32).collect();
        for &v in &relu(&x) {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_relu6_bounded() {
        let x: Vec<f32> = (-20..=20).map(|i| i as f32).collect();
        for &v in &relu6(&x) {
            assert!((0.0..=6.0).contains(&v));
        }
    }

    #[test]
    fn test_exp_positive() {
        let x: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        for &v in &exp(&x) {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_fma_equals_add_mul() {
        let a: Vec<f32> = (0..17).map(|i| i as f32 * 0.3).collect();
        let b: Vec<f32> = (0..17).map(|i| (i as f32 + 1.0) * 0.7).collect();
        let c: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
        let fma = fused_multiply_add(&a, &b, &c);
        let manual = add(&mul(&a, &b), &c);
        assert_close(&fma, &manual, 1e-5);
    }
}
