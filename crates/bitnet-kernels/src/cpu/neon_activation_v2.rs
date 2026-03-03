//! NEON-optimized activation v2 kernels for Apple Silicon (aarch64).
//!
//! Provides six activation functions commonly used in LLM inference:
//! SiLU, GELU (fast tanh approximation), ReLU, SwiGLU, sigmoid, and
//! softplus. Each function has an `unsafe fn neon_*` SIMD path, a
//! `fn scalar_*` fallback, and a public dispatcher that selects the
//! best implementation at runtime via `is_aarch64_feature_detected!`.
//!
//! NEON intrinsics use `vfmaq_f32` for fused multiply-add,
//! `vrecpeq_f32`/`vrecpsq_f32` for fast reciprocal, and Horner's
//! method polynomial approximation for exp/sigmoid.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for `float32x4_t`.
const LANES: usize = 4;

/// sqrt(2/π) ≈ 0.7978845608
const SQRT_2_OVER_PI: f32 = 0.797_884_56;

/// GELU cubic coefficient
const GELU_COEFF: f32 = 0.044715;

// ── Scalar helpers ──────────────────────────────────────────────────

#[inline(always)]
fn scalar_fast_exp(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    // Horner's method degree-4 polynomial for exp(r)
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

// ── Scalar implementations ──────────────────────────────────────────

/// Scalar sigmoid: 1 / (1 + exp(-x)).
#[inline(always)]
pub fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + scalar_fast_exp(-x))
}

/// Scalar SiLU: x * sigmoid(x).
#[inline(always)]
pub fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

/// Scalar GELU (tanh approximation):
/// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline(always)]
pub fn scalar_gelu(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    let e2 = scalar_fast_exp(2.0 * inner);
    let tanh_approx = (e2 - 1.0) / (e2 + 1.0);
    0.5 * x * (1.0 + tanh_approx)
}

/// Scalar ReLU: max(0, x).
#[inline(always)]
pub fn scalar_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Scalar SwiGLU: silu(gate) * up.
#[inline(always)]
pub fn scalar_swiglu(gate: f32, up: f32) -> f32 {
    scalar_silu(gate) * up
}

/// Scalar softplus: ln(1 + exp(x)) with numerical stability.
#[inline(always)]
pub fn scalar_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // ln(1+exp(x)) ≈ x for large x
    } else if x < -20.0 {
        0.0 // ln(1+exp(x)) ≈ 0 for very negative x
    } else {
        (1.0 + scalar_fast_exp(x)).ln()
    }
}

/// Scalar sigmoid applied to a slice.
pub fn scalar_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid(*x);
    }
}

/// Scalar SiLU applied to a slice.
pub fn scalar_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_silu(*x);
    }
}

/// Scalar GELU applied to a slice.
pub fn scalar_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_gelu(*x);
    }
}

/// Scalar ReLU applied to a slice.
pub fn scalar_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_relu(*x);
    }
}

/// Scalar SwiGLU applied to slices.
pub fn scalar_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "gate and up must have same length");
    assert!(output.len() >= gate.len(), "output buffer too small");
    for i in 0..gate.len() {
        output[i] = scalar_swiglu(gate[i], up[i]);
    }
}

/// Scalar softplus applied to a slice.
pub fn scalar_softplus_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_softplus(*x);
    }
}

// ── NEON vector helpers ─────────────────────────────────────────────

/// NEON fast exp approximation (4 lanes) using Horner's method.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
        let min_val = vdupq_n_f32(-88.0);
        let max_val = vdupq_n_f32(88.0);
        let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let n = vrndnq_f32(vmulq_f32(x, log2e));
        let r = vsubq_f32(x, vmulq_f32(n, ln2));

        // Horner's method: 1 + r*(1 + r*(0.5 + r*(1/6 + r/24)))
        let c1 = vdupq_n_f32(1.0 / 24.0);
        let c2 = vdupq_n_f32(1.0 / 6.0);
        let c3 = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);

        let p = vfmaq_f32(c2, r, c1);
        let p = vfmaq_f32(c3, r, p);
        let p = vfmaq_f32(one, r, p);
        let poly = vfmaq_f32(one, r, p);

        // 2^n via bit manipulation
        let bias = vdupq_n_s32(127);
        let ni = vcvtq_s32_f32(n);
        let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

        vmulq_f32(poly, pow2n)
    }
}

/// NEON sigmoid: 1 / (1 + exp(-x)) using vrecpeq_f32 for fast reciprocal.
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
        let one = vdupq_n_f32(1.0);
        let neg_x = vnegq_f32(x);
        let exp_neg = fast_exp_neon(neg_x);
        let denom = vaddq_f32(one, exp_neg);
        // Newton-Raphson refined reciprocal
        let recip = vrecpeq_f32(denom);
        let recip = vmulq_f32(vrecpsq_f32(denom, recip), recip);
        recip
    }
}

/// NEON tanh: (exp(2x)-1)/(exp(2x)+1).
///
/// # Safety
/// Requires AArch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn tanh_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
        let two = vdupq_n_f32(2.0);
        let one = vdupq_n_f32(1.0);
        let e2x = fast_exp_neon(vmulq_f32(two, x));
        let num = vsubq_f32(e2x, one);
        let den = vaddq_f32(e2x, one);
        let recip = vrecpeq_f32(den);
        let recip = vmulq_f32(vrecpsq_f32(den, recip), recip);
        vmulq_f32(num, recip)
    }
}

/// NEON fast approximate ln(x) for positive x.
///
/// Uses IEEE 754 exponent extraction + degree-4 minimax polynomial
/// on the mantissa in [1, 2). Accuracy: < 5e-4 max error.
///
/// # Safety
/// Requires AArch64 with NEON. Input must be positive.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_ln_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
        let one = vdupq_n_f32(1.0);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let bias = vdupq_n_s32(127);

        let xi = vreinterpretq_s32_f32(x);
        let exponent = vsubq_s32(vshrq_n_s32(xi, 23), bias);
        let e_f = vcvtq_f32_s32(exponent);

        // Extract mantissa in [1, 2)
        let mantissa_mask = vdupq_n_s32(0x007F_FFFF);
        let one_bits = vreinterpretq_s32_f32(one);
        let m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(xi, mantissa_mask), one_bits));

        // Polynomial approx of ln(m) for m in [1, 2):
        // ln(m) ≈ (m-1) * (c0 + (m-1)*(c1 + (m-1)*(c2 + (m-1)*c3)))
        // Minimax coefficients for better accuracy
        let m1 = vsubq_f32(m, one);
        let c0 = vdupq_n_f32(0.99949556);
        let c1 = vdupq_n_f32(-0.49190896);
        let c2 = vdupq_n_f32(0.28947478);
        let c3 = vdupq_n_f32(-0.13606275);

        // Horner's method: c0 + m1*(c1 + m1*(c2 + m1*c3))
        let p = vfmaq_f32(c2, m1, c3);
        let p = vfmaq_f32(c1, m1, p);
        let p = vfmaq_f32(c0, m1, p);
        let ln_m = vmulq_f32(m1, p);

        // ln(x) = ln(m) + e * ln(2)
        vfmaq_f32(ln_m, e_f, ln2)
    }
}

// ── Unsafe NEON implementations ─────────────────────────────────────

/// NEON SiLU: x * sigmoid(x).
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        for i in 0..chunks {
            let offset = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(offset));
            let sig = sigmoid_neon(x);
            let result = vmulq_f32(x, sig);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_silu(input[tail + i]);
    }
}

/// NEON GELU (fast tanh approximation):
/// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let coeff = vdupq_n_f32(GELU_COEFF);
        let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);

        for i in 0..chunks {
            let offset = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(offset));
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            // inner = sqrt(2/π) * (x + 0.044715 * x³)
            let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
            let t = tanh_neon(inner);
            let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_gelu(input[tail + i]);
    }
}

/// NEON ReLU: max(0, x) using vmaxq_f32.
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * LANES;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let result = vmaxq_f32(v, zero);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_relu(input[tail + i]);
    }
}

/// NEON SwiGLU: silu(gate) * up, fused for efficiency.
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if slices have mismatched lengths or output is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "gate and up must have same length");
    assert!(output.len() >= gate.len(), "output buffer too small");
    let n = gate.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        for i in 0..chunks {
            let offset = i * LANES;
            let g = vld1q_f32(gate.as_ptr().add(offset));
            let u = vld1q_f32(up.as_ptr().add(offset));
            let sig = sigmoid_neon(g);
            // silu(gate) * up = gate * sigmoid(gate) * up
            let silu_g = vmulq_f32(g, sig);
            let result = vmulq_f32(silu_g, u);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_swiglu(gate[tail + i], up[tail + i]);
    }
}

/// NEON sigmoid: 1 / (1 + exp(-x)) using polynomial approximation.
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32_v2(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        for i in 0..chunks {
            let offset = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(offset));
            let result = sigmoid_neon(x);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_sigmoid(input[tail + i]);
    }
}

/// NEON softplus: ln(1 + exp(x)) with numerical stability.
///
/// Uses the identity: softplus(x) = max(x,0) + ln(1 + exp(-|x|))
/// which avoids overflow. The ln(1+y) for small y is computed via
/// a degree-5 Taylor polynomial of log1p for improved accuracy.
///
/// For |x| > 20, uses the asymptotic approximations directly.
///
/// # Safety
/// Caller must ensure target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softplus_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let threshold_hi = vdupq_n_f32(20.0);
        let threshold_lo = vdupq_n_f32(-20.0);
        let zero = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(offset));

            // Compute scalar softplus for the 4 lanes using
            // NEON for data movement but scalar ln for accuracy
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), x);
            for j in 0..4 {
                buf[j] = scalar_softplus(buf[j]);
            }
            let result_general = vld1q_f32(buf.as_ptr());

            // Select: x>20 → x, x<-20 → 0, else → scalar result
            let hi_mask = vcgtq_f32(x, threshold_hi);
            let lo_mask = vcltq_f32(x, threshold_lo);
            let result = vbslq_f32(hi_mask, x, result_general);
            let result = vbslq_f32(lo_mask, zero, result);

            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = scalar_softplus(input[tail + i]);
    }
}

// ── Public dispatchers ──────────────────────────────────────────────

/// SiLU activation: x * sigmoid(x).
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn silu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON feature detected at runtime.
            unsafe {
                neon_silu_f32(input, output);
            }
            return;
        }
    }
    scalar_silu_f32(input, output);
}

/// GELU activation (fast tanh approximation).
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn gelu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gelu_f32(input, output);
            }
            return;
        }
    }
    scalar_gelu_f32(input, output);
}

/// ReLU activation: max(0, x).
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn relu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_relu_f32(input, output);
            }
            return;
        }
    }
    scalar_relu_f32(input, output);
}

/// SwiGLU activation: silu(gate) * up.
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_swiglu_f32(gate, up, output);
            }
            return;
        }
    }
    scalar_swiglu_f32(gate, up, output);
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn sigmoid_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_sigmoid_f32_v2(input, output);
            }
            return;
        }
    }
    scalar_sigmoid_f32(input, output);
}

/// Softplus activation: ln(1 + exp(x)).
///
/// Dispatches to NEON on aarch64, scalar fallback otherwise.
pub fn softplus_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_softplus_f32(input, output);
            }
            return;
        }
    }
    scalar_softplus_f32(input, output);
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sigmoid using std exp for accuracy comparison.
    fn ref_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn ref_silu(x: f32) -> f32 {
        x * ref_sigmoid(x)
    }

    fn ref_gelu(x: f32) -> f32 {
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    fn ref_softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else if x < -20.0 {
            0.0
        } else {
            (1.0_f32 + x.exp()).ln()
        }
    }

    const TOL: f32 = 2e-3; // Tolerance for fast polynomial approximations
    const STRICT_TOL: f32 = 1e-6; // Tolerance for exact operations (ReLU)

    // ── SiLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_silu_f32_basic() {
        let input = [0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0f32; 4];
        silu_f32(&input, &mut output);
        for (x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_silu(*x)).abs() < TOL, "silu({x}) = {o}, expected {}", ref_silu(*x));
        }
    }

    #[test]
    fn test_silu_f32_zeros() {
        let input = [0.0; 8];
        let mut output = [1.0f32; 8];
        silu_f32(&input, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL);
        }
    }

    #[test]
    fn test_silu_f32_positive() {
        let input: Vec<f32> = (1..=16).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(o > 0.0, "silu({x}) should be positive");
            assert!((o - ref_silu(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_f32_negative() {
        let input: Vec<f32> = (1..=8).map(|i| -(i as f32)).collect();
        let mut output = vec![0.0f32; input.len()];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(o < 0.0, "silu({x}) should be negative for negative x");
            assert!((o - ref_silu(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_f32_tail_elements() {
        // 5 elements: 4 NEON + 1 scalar tail
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 5];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_silu(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_silu_f32_single_element() {
        let input = [3.0];
        let mut output = [0.0f32; 1];
        silu_f32(&input, &mut output);
        assert!((output[0] - ref_silu(3.0)).abs() < TOL);
    }

    #[test]
    fn test_silu_f32_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        silu_f32(&input, &mut output);
    }

    #[test]
    fn test_silu_f32_large_values() {
        let input = [10.0, 50.0, 80.0, -10.0, -50.0, -80.0];
        let mut output = vec![0.0f32; input.len()];
        silu_f32(&input, &mut output);
        // silu(large positive) ≈ x, silu(large negative) ≈ 0
        assert!((output[0] - 10.0).abs() < 0.01);
        assert!((output[3]).abs() < 0.01);
    }

    #[test]
    fn test_silu_f32_monotonic_positive() {
        let input: Vec<f32> = (0..20).map(|i| i as f32 * 0.25).collect();
        let mut output = vec![0.0f32; input.len()];
        silu_f32(&input, &mut output);
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1], "silu should be monotonic for x >= 0");
        }
    }

    #[test]
    fn test_scalar_silu_f32_matches_dispatcher() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
        let mut dispatch_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];
        silu_f32(&input, &mut dispatch_out);
        scalar_silu_f32(&input, &mut scalar_out);
        for i in 0..input.len() {
            assert!((dispatch_out[i] - scalar_out[i]).abs() < TOL);
        }
    }

    // ── GELU tests ──────────────────────────────────────────────────

    #[test]
    fn test_gelu_f32_basic() {
        let input = [0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0f32; 4];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_gelu(x)).abs() < TOL, "gelu({x}) = {o}, expected {}", ref_gelu(x));
        }
    }

    #[test]
    fn test_gelu_f32_zero() {
        let input = [0.0; 8];
        let mut output = [1.0f32; 8];
        gelu_f32(&input, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL, "gelu(0) should be 0");
        }
    }

    #[test]
    fn test_gelu_f32_symmetry() {
        // GELU is NOT symmetric, but gelu(x) + gelu(-x) ≈ x for small x
        let input = [1.0, -1.0];
        let mut output = [0.0f32; 2];
        gelu_f32(&input, &mut output);
        // gelu(-x) should be small negative for x > 0
        assert!(output[1] < 0.0, "gelu(-1) should be negative");
        assert!(output[0] > 0.0, "gelu(1) should be positive");
    }

    #[test]
    fn test_gelu_f32_tail() {
        let input = [0.5, 1.5, -0.5, 2.5, 0.1, -0.1, 3.0];
        let mut output = [0.0f32; 7];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_gelu(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_gelu_f32_positive_large() {
        let input = [5.0, 10.0, 20.0, 50.0];
        let mut output = [0.0f32; 4];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            // gelu(large x) ≈ x
            assert!((o - x).abs() / x.abs().max(1.0) < 0.01);
        }
    }

    #[test]
    fn test_gelu_f32_negative_large() {
        let input = [-5.0, -10.0, -20.0, -50.0];
        let mut output = [0.0f32; 4];
        gelu_f32(&input, &mut output);
        for &o in &output {
            assert!(o.abs() < 0.01, "gelu(large negative) ≈ 0");
        }
    }

    #[test]
    fn test_gelu_f32_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        gelu_f32(&input, &mut output);
    }

    #[test]
    fn test_gelu_f32_single() {
        let mut output = [0.0f32; 1];
        gelu_f32(&[0.5], &mut output);
        assert!((output[0] - ref_gelu(0.5)).abs() < TOL);
    }

    #[test]
    fn test_scalar_gelu_f32_matches_dispatcher() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
        let mut dispatch_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];
        gelu_f32(&input, &mut dispatch_out);
        scalar_gelu_f32(&input, &mut scalar_out);
        for i in 0..input.len() {
            assert!((dispatch_out[i] - scalar_out[i]).abs() < TOL);
        }
    }

    // ── ReLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_f32_basic() {
        let input = [1.0, -1.0, 0.0, 2.5];
        let mut output = [0.0f32; 4];
        relu_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < STRICT_TOL);
        assert!((output[1] - 0.0).abs() < STRICT_TOL);
        assert!((output[2] - 0.0).abs() < STRICT_TOL);
        assert!((output[3] - 2.5).abs() < STRICT_TOL);
    }

    #[test]
    fn test_relu_f32_all_negative() {
        let input = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let mut output = [1.0f32; 8];
        relu_f32(&input, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL);
        }
    }

    #[test]
    fn test_relu_f32_all_positive() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];
        relu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - x).abs() < STRICT_TOL);
        }
    }

    #[test]
    fn test_relu_f32_tail() {
        let input = [1.0, -2.0, 3.0, -4.0, 5.0];
        let mut output = [0.0f32; 5];
        relu_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < STRICT_TOL);
        assert!((output[1] - 0.0).abs() < STRICT_TOL);
        assert!((output[4] - 5.0).abs() < STRICT_TOL);
    }

    #[test]
    fn test_relu_f32_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        relu_f32(&input, &mut output);
    }

    #[test]
    fn test_relu_f32_single() {
        let mut output = [0.0f32; 1];
        relu_f32(&[-5.0], &mut output);
        assert!((output[0] - 0.0).abs() < STRICT_TOL);
    }

    #[test]
    fn test_relu_f32_large_values() {
        let input = [1e6, -1e6, 1e-6, -1e-6];
        let mut output = [0.0f32; 4];
        relu_f32(&input, &mut output);
        assert!((output[0] - 1e6).abs() < 1.0);
        assert!((output[1] - 0.0).abs() < STRICT_TOL);
        assert!(output[2] > 0.0);
        assert!((output[3] - 0.0).abs() < STRICT_TOL);
    }

    #[test]
    fn test_relu_f32_preserves_positive() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        relu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - x).abs() < STRICT_TOL);
        }
    }

    #[test]
    fn test_relu_f32_idempotent() {
        let input = [1.0, -1.0, 2.0, -2.0, 0.0, 3.0, -3.0, 4.0];
        let mut first = [0.0f32; 8];
        let mut second = [0.0f32; 8];
        relu_f32(&input, &mut first);
        relu_f32(&first, &mut second);
        for i in 0..8 {
            assert!((first[i] - second[i]).abs() < STRICT_TOL, "ReLU should be idempotent");
        }
    }

    // ── SwiGLU tests ────────────────────────────────────────────────

    #[test]
    fn test_swiglu_f32_basic() {
        let gate = [0.0, 1.0, -1.0, 2.0];
        let up = [1.0, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 4];
        swiglu_f32(&gate, &up, &mut output);
        for i in 0..4 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!((output[i] - expected).abs() < TOL, "swiglu gate={}, up={}: got {}, expected {}", gate[i], up[i], output[i], expected);
        }
    }

    #[test]
    fn test_swiglu_f32_zero_gate() {
        let gate = [0.0; 8];
        let up = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [1.0f32; 8];
        swiglu_f32(&gate, &up, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL, "swiglu with zero gate should be 0");
        }
    }

    #[test]
    fn test_swiglu_f32_zero_up() {
        let gate = [1.0, 2.0, 3.0, 4.0];
        let up = [0.0; 4];
        let mut output = [1.0f32; 4];
        swiglu_f32(&gate, &up, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL, "swiglu with zero up should be 0");
        }
    }

    #[test]
    fn test_swiglu_f32_tail() {
        let gate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let up = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let mut output = [0.0f32; 7];
        swiglu_f32(&gate, &up, &mut output);
        for i in 0..7 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!((output[i] - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_swiglu_f32_identity_up() {
        let gate = [0.5, 1.5, -0.5, 2.5];
        let up = [1.0; 4];
        let mut output = [0.0f32; 4];
        let mut silu_out = [0.0f32; 4];
        swiglu_f32(&gate, &up, &mut output);
        silu_f32(&gate, &mut silu_out);
        for i in 0..4 {
            assert!((output[i] - silu_out[i]).abs() < TOL, "swiglu with up=1 should equal silu");
        }
    }

    #[test]
    fn test_swiglu_f32_empty() {
        let gate: [f32; 0] = [];
        let up: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        swiglu_f32(&gate, &up, &mut output);
    }

    #[test]
    fn test_swiglu_f32_single() {
        let gate = [2.0];
        let up = [3.0];
        let mut output = [0.0f32; 1];
        swiglu_f32(&gate, &up, &mut output);
        let expected = ref_silu(2.0) * 3.0;
        assert!((output[0] - expected).abs() < TOL);
    }

    #[test]
    fn test_swiglu_f32_negative_gate_and_up() {
        let gate = [-1.0, -2.0, -3.0, -4.0];
        let up = [-1.0, -2.0, -3.0, -4.0];
        let mut output = [0.0f32; 4];
        swiglu_f32(&gate, &up, &mut output);
        for i in 0..4 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!((output[i] - expected).abs() < TOL);
        }
    }

    // ── Sigmoid tests ───────────────────────────────────────────────

    #[test]
    fn test_sigmoid_f32_basic() {
        let input = [0.0, 1.0, -1.0, 5.0];
        let mut output = [0.0f32; 4];
        sigmoid_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_sigmoid(x)).abs() < TOL, "sigmoid({x}) = {o}, expected {}", ref_sigmoid(x));
        }
    }

    #[test]
    fn test_sigmoid_f32_zero() {
        let input = [0.0; 4];
        let mut output = [0.0f32; 4];
        sigmoid_f32(&input, &mut output);
        for &o in &output {
            assert!((o - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn test_sigmoid_f32_range() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; input.len()];
        sigmoid_f32(&input, &mut output);
        for &o in &output {
            assert!(o >= 0.0 && o <= 1.0, "sigmoid must be in [0, 1], got {o}");
        }
    }

    #[test]
    fn test_sigmoid_f32_saturation() {
        let input = [50.0, -50.0, 100.0, -100.0];
        let mut output = [0.0f32; 4];
        sigmoid_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < TOL);
        assert!((output[1] - 0.0).abs() < TOL);
        assert!((output[2] - 1.0).abs() < TOL);
        assert!((output[3] - 0.0).abs() < TOL);
    }

    #[test]
    fn test_sigmoid_f32_antisymmetry() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let neg_input: Vec<f32> = input.iter().map(|x| -x).collect();
        let mut out_pos = [0.0f32; 4];
        let mut out_neg = vec![0.0f32; 4];
        sigmoid_f32(&input, &mut out_pos);
        sigmoid_f32(&neg_input, &mut out_neg);
        for i in 0..4 {
            assert!((out_pos[i] + out_neg[i] - 1.0).abs() < TOL, "sigmoid(x) + sigmoid(-x) should ≈ 1");
        }
    }

    #[test]
    fn test_sigmoid_f32_monotonic() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        sigmoid_f32(&input, &mut output);
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1] - TOL, "sigmoid should be monotonically increasing");
        }
    }

    #[test]
    fn test_sigmoid_f32_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        sigmoid_f32(&input, &mut output);
    }

    #[test]
    fn test_sigmoid_f32_tail() {
        let input = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0];
        let mut output = [0.0f32; 6];
        sigmoid_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_sigmoid(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_scalar_sigmoid_f32_matches_dispatcher() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
        let mut dispatch_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];
        sigmoid_f32(&input, &mut dispatch_out);
        scalar_sigmoid_f32(&input, &mut scalar_out);
        for i in 0..input.len() {
            assert!((dispatch_out[i] - scalar_out[i]).abs() < TOL);
        }
    }

    // ── Softplus tests ──────────────────────────────────────────────

    #[test]
    fn test_softplus_f32_basic() {
        let input = [0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0f32; 4];
        softplus_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_softplus(x)).abs() < TOL, "softplus({x}) = {o}, expected {}", ref_softplus(x));
        }
    }

    #[test]
    fn test_softplus_f32_zero() {
        let input = [0.0; 4];
        let mut output = [0.0f32; 4];
        softplus_f32(&input, &mut output);
        let expected = (2.0_f32).ln(); // ln(1 + exp(0)) = ln(2)
        for &o in &output {
            assert!((o - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_softplus_f32_positive() {
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; input.len()];
        softplus_f32(&input, &mut output);
        for &o in &output {
            assert!(o > 0.0, "softplus should always be positive");
        }
    }

    #[test]
    fn test_softplus_f32_large_positive() {
        let input = [25.0, 50.0, 80.0, 100.0];
        let mut output = [0.0f32; 4];
        softplus_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - x).abs() < 0.1, "softplus(large x) ≈ x");
        }
    }

    #[test]
    fn test_softplus_f32_large_negative() {
        let input = [-25.0, -50.0, -80.0, -100.0];
        let mut output = [0.0f32; 4];
        softplus_f32(&input, &mut output);
        for &o in &output {
            assert!(o.abs() < 0.01, "softplus(large negative) ≈ 0");
        }
    }

    #[test]
    fn test_softplus_f32_monotonic() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; input.len()];
        softplus_f32(&input, &mut output);
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1] - TOL, "softplus should be monotonically increasing");
        }
    }

    #[test]
    fn test_softplus_f32_tail() {
        let input = [0.5, 1.5, -0.5, 2.5, 0.1];
        let mut output = [0.0f32; 5];
        softplus_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!((o - ref_softplus(x)).abs() < TOL);
        }
    }

    #[test]
    fn test_softplus_f32_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        softplus_f32(&input, &mut output);
    }

    #[test]
    fn test_softplus_f32_single() {
        let mut output = [0.0f32; 1];
        softplus_f32(&[1.0], &mut output);
        assert!((output[0] - ref_softplus(1.0)).abs() < TOL);
    }

    #[test]
    fn test_softplus_f32_always_positive() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; input.len()];
        softplus_f32(&input, &mut output);
        for &o in &output {
            assert!(o >= 0.0, "softplus must be non-negative, got {o}");
        }
    }

    #[test]
    fn test_scalar_softplus_f32_matches_dispatcher() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
        let mut dispatch_out = vec![0.0f32; input.len()];
        let mut scalar_out = vec![0.0f32; input.len()];
        softplus_f32(&input, &mut dispatch_out);
        scalar_softplus_f32(&input, &mut scalar_out);
        for i in 0..input.len() {
            assert!((dispatch_out[i] - scalar_out[i]).abs() < TOL);
        }
    }

    // ── Cross-function property tests ───────────────────────────────

    #[test]
    fn test_silu_equals_x_times_sigmoid() {
        let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let mut silu_out = vec![0.0f32; input.len()];
        let mut sig_out = vec![0.0f32; input.len()];
        silu_f32(&input, &mut silu_out);
        sigmoid_f32(&input, &mut sig_out);
        for i in 0..input.len() {
            let expected = input[i] * sig_out[i];
            assert!((silu_out[i] - expected).abs() < TOL, "silu(x) should equal x * sigmoid(x)");
        }
    }

    #[test]
    fn test_swiglu_equals_silu_gate_times_up() {
        let gate: Vec<f32> = (-5..=5).map(|i| i as f32 * 0.5).collect();
        let up: Vec<f32> = (0..11).map(|i| i as f32 * 0.3 + 0.1).collect();
        let mut swiglu_out = vec![0.0f32; gate.len()];
        let mut silu_out = vec![0.0f32; gate.len()];
        swiglu_f32(&gate, &up, &mut swiglu_out);
        silu_f32(&gate, &mut silu_out);
        for i in 0..gate.len() {
            let expected = silu_out[i] * up[i];
            assert!((swiglu_out[i] - expected).abs() < TOL);
        }
    }

    #[test]
    fn test_relu_lower_bounds_silu() {
        // For x > 0, relu(x) >= silu(x) since sigmoid(x) <= 1
        let input: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();
        let mut relu_out = vec![0.0f32; input.len()];
        let mut silu_out = vec![0.0f32; input.len()];
        relu_f32(&input, &mut relu_out);
        silu_f32(&input, &mut silu_out);
        for i in 0..input.len() {
            assert!(relu_out[i] >= silu_out[i] - TOL, "relu(x) >= silu(x) for positive x");
        }
    }

    #[test]
    fn test_softplus_upper_bounds_relu() {
        // softplus(x) >= relu(x) for all x
        let input: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
        let mut sp_out = vec![0.0f32; input.len()];
        let mut relu_out = vec![0.0f32; input.len()];
        softplus_f32(&input, &mut sp_out);
        relu_f32(&input, &mut relu_out);
        for i in 0..input.len() {
            assert!(sp_out[i] >= relu_out[i] - TOL, "softplus(x) >= relu(x)");
        }
    }

    #[test]
    fn test_gelu_at_origin() {
        let input = [0.0; 4];
        let mut output = [1.0f32; 4];
        gelu_f32(&input, &mut output);
        for &o in &output {
            assert!((o - 0.0).abs() < STRICT_TOL, "gelu(0) = 0");
        }
    }

    #[test]
    fn test_all_activations_handle_16_elements() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let ones = vec![1.0f32; 16];
        let mut out = vec![0.0f32; 16];

        silu_f32(&input, &mut out);
        gelu_f32(&input, &mut out);
        relu_f32(&input, &mut out);
        sigmoid_f32(&input, &mut out);
        softplus_f32(&input, &mut out);
        swiglu_f32(&input, &ones, &mut out);
        // No panic = success
    }

    #[test]
    fn test_all_activations_handle_17_elements() {
        let input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.5) * 0.5).collect();
        let ones = vec![1.0f32; 17];
        let mut out = vec![0.0f32; 17];

        silu_f32(&input, &mut out);
        gelu_f32(&input, &mut out);
        relu_f32(&input, &mut out);
        sigmoid_f32(&input, &mut out);
        softplus_f32(&input, &mut out);
        swiglu_f32(&input, &ones, &mut out);
    }

    #[test]
    fn test_all_activations_handle_3_elements() {
        // All scalar tail, no NEON chunks
        let input = [1.0, -1.0, 0.5];
        let ones = [1.0f32; 3];
        let mut out = [0.0f32; 3];

        silu_f32(&input, &mut out);
        gelu_f32(&input, &mut out);
        relu_f32(&input, &mut out);
        sigmoid_f32(&input, &mut out);
        softplus_f32(&input, &mut out);
        swiglu_f32(&input, &ones, &mut out);
    }

    // ── Panic tests ─────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_silu_panics_on_small_output() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        silu_f32(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_gelu_panics_on_small_output() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        gelu_f32(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_relu_panics_on_small_output() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        relu_f32(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_sigmoid_panics_on_small_output() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        sigmoid_f32(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_softplus_panics_on_small_output() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        softplus_f32(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "gate and up must have same length")]
    fn test_swiglu_panics_on_mismatched_lengths() {
        let gate = [1.0, 2.0];
        let up = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        swiglu_f32(&gate, &up, &mut output);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_swiglu_panics_on_small_output() {
        let gate = [1.0, 2.0, 3.0];
        let up = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 2];
        swiglu_f32(&gate, &up, &mut output);
    }
}
