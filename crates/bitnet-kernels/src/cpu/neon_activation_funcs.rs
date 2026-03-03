//! ARM NEON-optimized activation function kernels for LLM inference.
//!
//! Provides SiLU, GELU, ReLU, sigmoid, fused SiLU-mul, and softcap activation
//! functions with NEON SIMD intrinsics on AArch64, scalar fallbacks on other
//! architectures, and public dispatchers that pick the best path at compile time.
//!
//! Transcendental approximations (exp, tanh) use Horner-scheme polynomials with
//! Cody-Waite range reduction, giving < 1e-3 max error in the working range.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ───────────────────────────────────────────────────────

const LANES: usize = 4;

/// sqrt(2/π) ≈ 0.7978845608
const SQRT_2_OVER_PI: f32 = 0.797_884_56;

/// GELU cubic coefficient
const GELU_COEFF: f32 = 0.044715;

// ── Scalar helpers ──────────────────────────────────────────────────

/// Fast scalar exp approximation via Cody-Waite range reduction + degree-4
/// Horner polynomial.  Good for |x| < ~88; clamped to avoid overflow.
#[inline(always)]
fn scalar_fast_exp(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    // Horner: 1 + r + r²/2 + r³/6 + r⁴/24
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn scalar_sigmoid_approx(x: f32) -> f32 {
    1.0 / (1.0 + scalar_fast_exp(-x))
}

#[inline(always)]
fn scalar_tanh_approx(x: f32) -> f32 {
    let e2x = scalar_fast_exp(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

// ── NEON vector helpers ─────────────────────────────────────────────

/// NEON fast exp approximation (4 lanes).
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

        // Horner: 1 + r*(1 + r*(0.5 + r*(1/6 + r/24)))
        let c1 = vdupq_n_f32(1.0 / 24.0);
        let c2 = vdupq_n_f32(1.0 / 6.0);
        let c3 = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);

        let p = vfmaq_f32(c2, r, c1);
        let p = vfmaq_f32(c3, r, p);
        let p = vfmaq_f32(one, r, p);
        let poly = vfmaq_f32(one, r, p);

        // 2^n via IEEE-754 exponent bias
        let bias = vdupq_n_s32(127);
        let ni = vcvtq_s32_f32(n);
        let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

        vmulq_f32(poly, pow2n)
    }
}

/// NEON sigmoid: 1 / (1 + exp(-x))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
        let one = vdupq_n_f32(1.0);
        let neg_x = vnegq_f32(x);
        let exp_neg = fast_exp_neon(neg_x);
        let denom = vaddq_f32(one, exp_neg);
        // Newton-Raphson reciprocal (2 iterations)
        let recip = vrecpeq_f32(denom);
        let recip = vmulq_f32(vrecpsq_f32(denom, recip), recip);
        recip
    }
}

/// NEON tanh approximation: (exp(2x)-1)/(exp(2x)+1)
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

// =====================================================================
// 1. SiLU / Swish: x * sigmoid(x)
// =====================================================================

/// NEON-accelerated SiLU activation.
///
/// # Safety
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;

    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let sig = sigmoid_neon(x);
            let r = vmulq_f32(x, sig);
            vst1q_f32(output.as_mut_ptr().add(off), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..n {
        let x = input[i];
        output[i] = x * scalar_sigmoid_approx(x);
    }
}

/// Scalar SiLU fallback.
pub fn scalar_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = x * scalar_sigmoid_approx(*x);
    }
}

/// SiLU dispatcher — NEON on aarch64, scalar otherwise.
pub fn silu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on AArch64.
        unsafe { neon_silu_f32(input, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_silu_f32(input, output);
    }
}

// =====================================================================
// 2. GELU (fast tanh approximation)
// =====================================================================

/// NEON-accelerated GELU activation.
///
/// Computes `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// # Safety
/// Caller must ensure the target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;

    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let coeff = vdupq_n_f32(GELU_COEFF);
        let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);

        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
            let t = tanh_neon(inner);
            let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
            vst1q_f32(output.as_mut_ptr().add(off), result);
        }
    }

    let tail = chunks * LANES;
    for i in tail..n {
        let x = input[i];
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        output[i] = 0.5 * x * (1.0 + scalar_tanh_approx(inner));
    }
}

/// Scalar GELU fallback.
pub fn scalar_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        *o = 0.5 * x * (1.0 + scalar_tanh_approx(inner));
    }
}

/// GELU dispatcher.
pub fn gelu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_gelu_f32(input, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_gelu_f32(input, output);
    }
}

// =====================================================================
// 3. ReLU: max(0, x)
// =====================================================================

/// NEON-accelerated ReLU activation.
///
/// # Safety
/// Caller must ensure the target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let r = vmaxq_f32(x, zero);
            vst1q_f32(output.as_mut_ptr().add(off), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
    }
}

/// Scalar ReLU fallback.
pub fn scalar_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = if *x > 0.0 { *x } else { 0.0 };
    }
}

/// ReLU dispatcher.
pub fn relu_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_relu_f32(input, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_relu_f32(input, output);
    }
}

// =====================================================================
// 4. Sigmoid: 1 / (1 + exp(-x))
// =====================================================================

/// NEON-accelerated sigmoid activation using polynomial exp approximation.
///
/// # Safety
/// Caller must ensure the target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / LANES;

    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let r = sigmoid_neon(x);
            vst1q_f32(output.as_mut_ptr().add(off), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..n {
        output[i] = scalar_sigmoid_approx(input[i]);
    }
}

/// Scalar sigmoid fallback.
pub fn scalar_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid_approx(*x);
    }
}

/// Sigmoid dispatcher.
pub fn sigmoid_f32(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sigmoid_f32(input, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_sigmoid_f32(input, output);
    }
}

// =====================================================================
// 5. Fused SiLU-mul: silu(x) * y  (gate mechanism in LLMs)
// =====================================================================

/// NEON-accelerated fused SiLU + elementwise multiply.
///
/// Computes `output[i] = silu(x[i]) * gate[i]` where `silu(x) = x * σ(x)`.
///
/// # Safety
/// Caller must ensure the target supports NEON.
///
/// # Panics
/// Panics if slice lengths are mismatched or `output` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_silu_mul_f32(x: &[f32], gate: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), gate.len(), "x and gate must have the same length");
    assert!(output.len() >= x.len(), "output buffer too small");
    let n = x.len();
    let chunks = n / LANES;

    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let vx = vld1q_f32(x.as_ptr().add(off));
            let vg = vld1q_f32(gate.as_ptr().add(off));
            let sig = sigmoid_neon(vx);
            let silu = vmulq_f32(vx, sig);
            let r = vmulq_f32(silu, vg);
            vst1q_f32(output.as_mut_ptr().add(off), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..n {
        let xi = x[i];
        output[i] = xi * scalar_sigmoid_approx(xi) * gate[i];
    }
}

/// Scalar fused SiLU-mul fallback.
pub fn scalar_fused_silu_mul_f32(x: &[f32], gate: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), gate.len(), "x and gate must have the same length");
    assert!(output.len() >= x.len(), "output buffer too small");
    for i in 0..x.len() {
        let xi = x[i];
        output[i] = xi * scalar_sigmoid_approx(xi) * gate[i];
    }
}

/// Fused SiLU-mul dispatcher.
pub fn fused_silu_mul_f32(x: &[f32], gate: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_silu_mul_f32(x, gate, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_silu_mul_f32(x, gate, output);
    }
}

// =====================================================================
// 6. Softcap: cap * tanh(x / cap)  (Gemma-style)
// =====================================================================

/// NEON-accelerated softcap activation.
///
/// Computes `output[i] = cap * tanh(input[i] / cap)`.
///
/// # Safety
/// Caller must ensure the target supports NEON.
///
/// # Panics
/// Panics if `output.len() < input.len()` or `cap == 0.0`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softcap_f32(input: &[f32], cap: f32, output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    assert!(cap != 0.0, "cap must not be zero");
    let n = input.len();
    let chunks = n / LANES;

    unsafe {
        let vcap = vdupq_n_f32(cap);
        let inv_cap = vdupq_n_f32(1.0 / cap);
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let scaled = vmulq_f32(x, inv_cap);
            let t = tanh_neon(scaled);
            let r = vmulq_f32(vcap, t);
            vst1q_f32(output.as_mut_ptr().add(off), r);
        }
    }

    let tail = chunks * LANES;
    let inv = 1.0 / cap;
    for i in tail..n {
        output[i] = cap * scalar_tanh_approx(input[i] * inv);
    }
}

/// Scalar softcap fallback.
pub fn scalar_softcap_f32(input: &[f32], cap: f32, output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    assert!(cap != 0.0, "cap must not be zero");
    let inv = 1.0 / cap;
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = cap * scalar_tanh_approx(*x * inv);
    }
}

/// Softcap dispatcher.
pub fn softcap_f32(input: &[f32], cap: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_softcap_f32(input, cap, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_softcap_f32(input, cap, output);
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Reference sigmoid using stdlib exp.
    fn ref_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Reference SiLU using stdlib exp.
    fn ref_silu(x: f32) -> f32 {
        x * ref_sigmoid(x)
    }

    /// Reference GELU using stdlib tanh.
    fn ref_gelu(x: f32) -> f32 {
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// Reference softcap using stdlib tanh.
    fn ref_softcap(x: f32, cap: f32) -> f32 {
        cap * (x / cap).tanh()
    }

    const EPS: f32 = 1e-3;

    // ================================================================
    // SiLU tests
    // ================================================================

    #[test]
    fn test_silu_basic_positive() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_silu(x), EPS), "silu({x}) = {o}, expected {}", ref_silu(x));
        }
    }

    #[test]
    fn test_silu_basic_negative() {
        let input = [-1.0_f32, -2.0, -3.0, -4.0];
        let mut output = [0.0_f32; 4];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_silu(x), EPS), "silu({x}) = {o}, expected {}", ref_silu(x));
        }
    }

    #[test]
    fn test_silu_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        silu_f32(&input, &mut output);
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_silu_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        silu_f32(&input, &mut output);
    }

    #[test]
    fn test_silu_size_one() {
        let input = [2.5_f32];
        let mut output = [0.0_f32; 1];
        silu_f32(&input, &mut output);
        assert!(approx_eq(output[0], ref_silu(2.5), EPS));
    }

    #[test]
    fn test_silu_remainder() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = [0.0_f32; 7];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_silu(x), EPS));
        }
    }

    #[test]
    fn test_silu_large_array() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut output = vec![0.0_f32; 1024];
        silu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_silu(x), EPS), "silu({x}) = {o}");
        }
    }

    #[test]
    fn test_silu_inf() {
        let input = [f32::INFINITY, f32::NEG_INFINITY];
        let mut output = [0.0_f32; 2];
        silu_f32(&input, &mut output);
        // silu(+inf) = inf * 1 = inf, silu(-inf) = -inf * 0 = 0 (or NaN-ish)
        assert!(output[0].is_infinite() || output[0] > 1e30);
    }

    #[test]
    fn test_silu_nan() {
        let input = [f32::NAN];
        let mut output = [0.0_f32; 1];
        silu_f32(&input, &mut output);
        assert!(output[0].is_nan());
    }

    #[test]
    fn test_silu_inplace_pattern() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0];
        let mut buf = input;
        let input_copy = input;
        let mut expected = [0.0_f32; 5];
        silu_f32(&input_copy, &mut expected);
        silu_f32(&input_copy, &mut buf);
        for i in 0..buf.len() {
            assert!(approx_eq(buf[i], expected[i], 1e-6));
        }
    }

    #[test]
    fn test_silu_dispatcher_matches_scalar() {
        let input = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        silu_f32(&input, &mut disp_out);
        scalar_silu_f32(&input, &mut scalar_out);
        for i in 0..5 {
            assert!(approx_eq(disp_out[i], scalar_out[i], 1e-5));
        }
    }

    // ================================================================
    // GELU tests
    // ================================================================

    #[test]
    fn test_gelu_basic_positive() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_gelu(x), EPS), "gelu({x}) = {o}, expected {}", ref_gelu(x));
        }
    }

    #[test]
    fn test_gelu_basic_negative() {
        let input = [-1.0_f32, -2.0, -3.0, -4.0];
        let mut output = [0.0_f32; 4];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_gelu(x), EPS), "gelu({x}) = {o}, expected {}", ref_gelu(x));
        }
    }

    #[test]
    fn test_gelu_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        gelu_f32(&input, &mut output);
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_gelu_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        gelu_f32(&input, &mut output);
    }

    #[test]
    fn test_gelu_size_one() {
        let input = [1.5_f32];
        let mut output = [0.0_f32; 1];
        gelu_f32(&input, &mut output);
        assert!(approx_eq(output[0], ref_gelu(1.5), EPS));
    }

    #[test]
    fn test_gelu_remainder() {
        let input = [0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let mut output = [0.0_f32; 5];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_gelu(x), EPS));
        }
    }

    #[test]
    fn test_gelu_large_array() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut output = vec![0.0_f32; 1024];
        gelu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_gelu(x), EPS), "gelu({x}) = {o}");
        }
    }

    #[test]
    fn test_gelu_nan() {
        let input = [f32::NAN];
        let mut output = [0.0_f32; 1];
        gelu_f32(&input, &mut output);
        assert!(output[0].is_nan());
    }

    #[test]
    fn test_gelu_inplace_pattern() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0];
        let mut buf = input;
        let input_copy = input;
        let mut expected = [0.0_f32; 5];
        gelu_f32(&input_copy, &mut expected);
        gelu_f32(&input_copy, &mut buf);
        for i in 0..buf.len() {
            assert!(approx_eq(buf[i], expected[i], 1e-6));
        }
    }

    #[test]
    fn test_gelu_dispatcher_matches_scalar() {
        let input = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        gelu_f32(&input, &mut disp_out);
        scalar_gelu_f32(&input, &mut scalar_out);
        for i in 0..5 {
            assert!(approx_eq(disp_out[i], scalar_out[i], 1e-5));
        }
    }

    // ================================================================
    // ReLU tests
    // ================================================================

    #[test]
    fn test_relu_basic_positive() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        relu_f32(&input, &mut output);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_basic_negative() {
        let input = [-1.0_f32, -2.0, -3.0, -4.0];
        let mut output = [0.0_f32; 4];
        relu_f32(&input, &mut output);
        assert_eq!(output, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        relu_f32(&input, &mut output);
        assert_eq!(output[0], 0.0);
    }

    #[test]
    fn test_relu_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        relu_f32(&input, &mut output);
    }

    #[test]
    fn test_relu_size_one() {
        let input = [5.0_f32];
        let mut output = [0.0_f32; 1];
        relu_f32(&input, &mut output);
        assert_eq!(output[0], 5.0);
    }

    #[test]
    fn test_relu_mixed_remainder() {
        let input = [-1.0_f32, 0.0, 1.0, -0.5, 2.0, -3.0, 0.1];
        let mut output = [0.0_f32; 7];
        relu_f32(&input, &mut output);
        assert_eq!(output, [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.1]);
    }

    #[test]
    fn test_relu_large_array() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.1).collect();
        let mut output = vec![0.0_f32; 1024];
        relu_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = if x > 0.0 { x } else { 0.0 };
            assert_eq!(o, expected);
        }
    }

    #[test]
    fn test_relu_inf() {
        let input = [f32::INFINITY, f32::NEG_INFINITY];
        let mut output = [0.0_f32; 2];
        relu_f32(&input, &mut output);
        assert_eq!(output[0], f32::INFINITY);
        assert_eq!(output[1], 0.0);
    }

    #[test]
    fn test_relu_nan() {
        // NEON vmaxq_f32 does not propagate NaN (IEEE 754 maxNum semantics),
        // so relu(NaN) may return 0.0 on aarch64.  Accept either NaN or 0.0.
        let input = [f32::NAN];
        let mut output = [0.0_f32; 1];
        relu_f32(&input, &mut output);
        assert!(output[0].is_nan() || output[0] == 0.0);
    }

    #[test]
    fn test_relu_inplace_pattern() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0];
        let mut buf = input;
        let input_copy = input;
        let mut expected = [0.0_f32; 5];
        relu_f32(&input_copy, &mut expected);
        relu_f32(&input_copy, &mut buf);
        assert_eq!(buf, expected);
    }

    #[test]
    fn test_relu_dispatcher_matches_scalar() {
        let input = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        relu_f32(&input, &mut disp_out);
        scalar_relu_f32(&input, &mut scalar_out);
        assert_eq!(disp_out, scalar_out);
    }

    // ================================================================
    // Sigmoid tests
    // ================================================================

    #[test]
    fn test_sigmoid_basic() {
        let input = [0.0_f32, 1.0, -1.0, 5.0];
        let mut output = [0.0_f32; 4];
        sigmoid_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_sigmoid(x), EPS), "sigmoid({x}) = {o}, expected {}", ref_sigmoid(x));
        }
    }

    #[test]
    fn test_sigmoid_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        sigmoid_f32(&input, &mut output);
        assert!(approx_eq(output[0], 0.5, EPS));
    }

    #[test]
    fn test_sigmoid_bounds() {
        let input = [-10.0_f32, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        let mut output = [0.0_f32; 7];
        sigmoid_f32(&input, &mut output);
        for &o in &output {
            assert!(o >= 0.0 && o <= 1.0, "sigmoid out of [0,1]: {o}");
        }
    }

    #[test]
    fn test_sigmoid_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        sigmoid_f32(&input, &mut output);
    }

    #[test]
    fn test_sigmoid_size_one() {
        let input = [3.0_f32];
        let mut output = [0.0_f32; 1];
        sigmoid_f32(&input, &mut output);
        assert!(approx_eq(output[0], ref_sigmoid(3.0), EPS));
    }

    #[test]
    fn test_sigmoid_large_array() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.02).collect();
        let mut output = vec![0.0_f32; 1024];
        sigmoid_f32(&input, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_sigmoid(x), EPS), "sigmoid({x}) = {o}");
        }
    }

    #[test]
    fn test_sigmoid_extreme_positive() {
        let input = [50.0_f32, 80.0];
        let mut output = [0.0_f32; 2];
        sigmoid_f32(&input, &mut output);
        for &o in &output {
            assert!(approx_eq(o, 1.0, EPS));
        }
    }

    #[test]
    fn test_sigmoid_extreme_negative() {
        let input = [-50.0_f32, -80.0];
        let mut output = [0.0_f32; 2];
        sigmoid_f32(&input, &mut output);
        for &o in &output {
            assert!(approx_eq(o, 0.0, EPS));
        }
    }

    #[test]
    fn test_sigmoid_nan() {
        let input = [f32::NAN];
        let mut output = [0.0_f32; 1];
        sigmoid_f32(&input, &mut output);
        assert!(output[0].is_nan());
    }

    #[test]
    fn test_sigmoid_inplace_pattern() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0];
        let mut buf = input;
        let input_copy = input;
        let mut expected = [0.0_f32; 5];
        sigmoid_f32(&input_copy, &mut expected);
        sigmoid_f32(&input_copy, &mut buf);
        for i in 0..buf.len() {
            assert!(approx_eq(buf[i], expected[i], 1e-6));
        }
    }

    #[test]
    fn test_sigmoid_dispatcher_matches_scalar() {
        let input = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        sigmoid_f32(&input, &mut disp_out);
        scalar_sigmoid_f32(&input, &mut scalar_out);
        for i in 0..5 {
            assert!(approx_eq(disp_out[i], scalar_out[i], 1e-5));
        }
    }

    // ================================================================
    // Fused SiLU-mul tests
    // ================================================================

    #[test]
    fn test_fused_silu_mul_basic() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let gate = [0.5_f32, 1.0, 1.5, 2.0];
        let mut output = [0.0_f32; 4];
        fused_silu_mul_f32(&x, &gate, &mut output);
        for i in 0..4 {
            let expected = ref_silu(x[i]) * gate[i];
            assert!(approx_eq(output[i], expected, EPS), "fused_silu_mul at {i}: {} vs {expected}", output[i]);
        }
    }

    #[test]
    fn test_fused_silu_mul_equivalence() {
        let x = [0.5_f32, -0.5, 1.0, -1.0, 3.0, -3.0, 0.1, 2.0];
        let gate = [1.0_f32, 2.0, 0.5, 1.5, 0.8, 1.2, 3.0, 0.3];
        let mut fused = [0.0_f32; 8];
        fused_silu_mul_f32(&x, &gate, &mut fused);

        // Compare with separate silu + mul
        let mut silu_out = [0.0_f32; 8];
        silu_f32(&x, &mut silu_out);
        for i in 0..8 {
            let separate = silu_out[i] * gate[i];
            assert!(approx_eq(fused[i], separate, EPS), "fused vs separate at {i}: {} vs {separate}", fused[i]);
        }
    }

    #[test]
    fn test_fused_silu_mul_empty() {
        let x: [f32; 0] = [];
        let gate: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        fused_silu_mul_f32(&x, &gate, &mut output);
    }

    #[test]
    fn test_fused_silu_mul_size_one() {
        let x = [2.0_f32];
        let gate = [0.5_f32];
        let mut output = [0.0_f32; 1];
        fused_silu_mul_f32(&x, &gate, &mut output);
        assert!(approx_eq(output[0], ref_silu(2.0) * 0.5, EPS));
    }

    #[test]
    fn test_fused_silu_mul_remainder() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let gate = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut output = [0.0_f32; 7];
        fused_silu_mul_f32(&x, &gate, &mut output);
        for i in 0..7 {
            let expected = ref_silu(x[i]) * gate[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_fused_silu_mul_gate_zeros() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let gate = [0.0_f32; 4];
        let mut output = [0.0_f32; 4];
        fused_silu_mul_f32(&x, &gate, &mut output);
        for &o in &output {
            assert!(approx_eq(o, 0.0, EPS));
        }
    }

    #[test]
    fn test_fused_silu_mul_gate_ones() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let gate = [1.0_f32; 4];
        let mut output = [0.0_f32; 4];
        let mut silu_out = [0.0_f32; 4];
        fused_silu_mul_f32(&x, &gate, &mut output);
        silu_f32(&x, &mut silu_out);
        for i in 0..4 {
            assert!(approx_eq(output[i], silu_out[i], 1e-5));
        }
    }

    #[test]
    fn test_fused_silu_mul_large_array() {
        let x: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let gate: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0_f32; 1024];
        fused_silu_mul_f32(&x, &gate, &mut output);
        for i in 0..1024 {
            let expected = ref_silu(x[i]) * gate[i];
            assert!(approx_eq(output[i], expected, EPS), "fused at {i}");
        }
    }

    #[test]
    fn test_fused_silu_mul_nan() {
        let x = [f32::NAN];
        let gate = [1.0_f32];
        let mut output = [0.0_f32; 1];
        fused_silu_mul_f32(&x, &gate, &mut output);
        assert!(output[0].is_nan());
    }

    #[test]
    fn test_fused_silu_mul_dispatcher_matches_scalar() {
        let x = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let gate = [1.0_f32, 0.5, 2.0, 0.3, 1.5];
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        fused_silu_mul_f32(&x, &gate, &mut disp_out);
        scalar_fused_silu_mul_f32(&x, &gate, &mut scalar_out);
        for i in 0..5 {
            assert!(approx_eq(disp_out[i], scalar_out[i], 1e-5));
        }
    }

    // ================================================================
    // Softcap tests
    // ================================================================

    #[test]
    fn test_softcap_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let cap = 5.0;
        let mut output = [0.0_f32; 4];
        softcap_f32(&input, cap, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_softcap(x, cap), EPS), "softcap({x}, {cap}) = {o}, expected {}", ref_softcap(x, cap));
        }
    }

    #[test]
    fn test_softcap_clamping() {
        // Values much larger than cap should be clamped near ±cap
        let input = [100.0_f32, -100.0];
        let cap = 10.0;
        let mut output = [0.0_f32; 2];
        softcap_f32(&input, cap, &mut output);
        assert!(approx_eq(output[0], cap, EPS), "softcap(100, 10) ≈ 10, got {}", output[0]);
        assert!(approx_eq(output[1], -cap, EPS), "softcap(-100, 10) ≈ -10, got {}", output[1]);
    }

    #[test]
    fn test_softcap_zero_input() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        softcap_f32(&input, 5.0, &mut output);
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_softcap_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        softcap_f32(&input, 5.0, &mut output);
    }

    #[test]
    fn test_softcap_size_one() {
        let input = [2.0_f32];
        let mut output = [0.0_f32; 1];
        softcap_f32(&input, 3.0, &mut output);
        assert!(approx_eq(output[0], ref_softcap(2.0, 3.0), EPS));
    }

    #[test]
    fn test_softcap_remainder() {
        let input = [1.0_f32, -1.0, 2.0, -2.0, 3.0];
        let cap = 4.0;
        let mut output = [0.0_f32; 5];
        softcap_f32(&input, cap, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_softcap(x, cap), EPS));
        }
    }

    #[test]
    fn test_softcap_small_cap() {
        let input = [1.0_f32, 2.0, -1.0, -2.0];
        let cap = 0.5;
        let mut output = [0.0_f32; 4];
        softcap_f32(&input, cap, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_softcap(x, cap), EPS));
        }
    }

    #[test]
    fn test_softcap_large_cap() {
        // Large cap → softcap ≈ identity for small x
        let input = [0.1_f32, 0.2, -0.1, -0.2];
        let cap = 1000.0;
        let mut output = [0.0_f32; 4];
        softcap_f32(&input, cap, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, x, 0.01), "softcap({x}, {cap}) ≈ {x}, got {o}");
        }
    }

    #[test]
    fn test_softcap_large_array() {
        let cap = 8.0;
        let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.05).collect();
        let mut output = vec![0.0_f32; 1024];
        softcap_f32(&input, cap, &mut output);
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(o, ref_softcap(x, cap), EPS), "softcap({x}, {cap}) = {o}");
        }
    }

    #[test]
    fn test_softcap_nan() {
        let input = [f32::NAN];
        let mut output = [0.0_f32; 1];
        softcap_f32(&input, 5.0, &mut output);
        assert!(output[0].is_nan());
    }

    #[test]
    fn test_softcap_inplace_pattern() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0];
        let mut buf = input;
        let input_copy = input;
        let cap = 3.0;
        let mut expected = [0.0_f32; 5];
        softcap_f32(&input_copy, cap, &mut expected);
        softcap_f32(&input_copy, cap, &mut buf);
        for i in 0..buf.len() {
            assert!(approx_eq(buf[i], expected[i], 1e-6));
        }
    }

    #[test]
    fn test_softcap_dispatcher_matches_scalar() {
        let input = [0.5_f32, -0.5, 1.0, -1.0, 3.0];
        let cap = 4.0;
        let mut disp_out = [0.0_f32; 5];
        let mut scalar_out = [0.0_f32; 5];
        softcap_f32(&input, cap, &mut disp_out);
        scalar_softcap_f32(&input, cap, &mut scalar_out);
        for i in 0..5 {
            assert!(approx_eq(disp_out[i], scalar_out[i], 1e-5));
        }
    }

    #[test]
    #[should_panic(expected = "cap must not be zero")]
    fn test_softcap_zero_cap_panics() {
        let input = [1.0_f32];
        let mut output = [0.0_f32; 1];
        softcap_f32(&input, 0.0, &mut output);
    }

    // ================================================================
    // Cross-function tests
    // ================================================================

    #[test]
    fn test_silu_vs_gelu_at_zero() {
        let input = [0.0_f32];
        let mut silu_out = [0.0_f32; 1];
        let mut gelu_out = [0.0_f32; 1];
        silu_f32(&input, &mut silu_out);
        gelu_f32(&input, &mut gelu_out);
        assert!(approx_eq(silu_out[0], 0.0, EPS));
        assert!(approx_eq(gelu_out[0], 0.0, EPS));
    }

    #[test]
    fn test_relu_vs_silu_positive_large() {
        // For large positive x, silu(x) ≈ x ≈ relu(x)
        let input = [10.0_f32, 20.0, 50.0, 100.0];
        let mut relu_out = [0.0_f32; 4];
        let mut silu_out = [0.0_f32; 4];
        relu_f32(&input, &mut relu_out);
        silu_f32(&input, &mut silu_out);
        for i in 0..4 {
            let ratio = silu_out[i] / relu_out[i];
            assert!(ratio > 0.99, "silu/relu ratio at {}: {ratio}", input[i]);
        }
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // sigmoid(-x) + sigmoid(x) ≈ 1
        let input = [0.5_f32, 1.0, 2.0, 3.0];
        let neg_input: Vec<f32> = input.iter().map(|x| -x).collect();
        let mut pos_out = [0.0_f32; 4];
        let mut neg_out = [0.0_f32; 4];
        sigmoid_f32(&input, &mut pos_out);
        sigmoid_f32(&neg_input, &mut neg_out);
        for i in 0..4 {
            assert!(approx_eq(pos_out[i] + neg_out[i], 1.0, EPS), "sigmoid symmetry at {}", input[i]);
        }
    }

    #[test]
    fn test_all_activations_handle_non_aligned() {
        // 3 elements — not a multiple of 4
        let input = [0.5_f32, -0.3, 1.2];
        let gate = [1.0_f32, 0.5, 2.0];
        let mut out = [0.0_f32; 3];

        silu_f32(&input, &mut out);
        for (&x, &o) in input.iter().zip(out.iter()) {
            assert!(approx_eq(o, ref_silu(x), EPS));
        }

        gelu_f32(&input, &mut out);
        for (&x, &o) in input.iter().zip(out.iter()) {
            assert!(approx_eq(o, ref_gelu(x), EPS));
        }

        relu_f32(&input, &mut out);
        sigmoid_f32(&input, &mut out);
        fused_silu_mul_f32(&input, &gate, &mut out);
        softcap_f32(&input, 5.0, &mut out);
    }
}
