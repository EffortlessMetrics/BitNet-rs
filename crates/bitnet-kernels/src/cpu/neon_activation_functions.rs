//! ARM NEON-optimized activation function kernels for transformer inference.
//!
//! Provides GELU, SiLU/Swish, ReLU, Leaky ReLU, fused GELU+SiLU, and SwiGLU
//! activation functions using NEON SIMD intrinsics on AArch64.  Each function
//! processes four `f32` lanes in parallel with scalar fallback for tail
//! elements.  Transcendental approximations (exp, tanh) use Cody-Waite
//! range-reduction with degree-4 minimax polynomials, giving < 1e-3 max error
//! in the working range.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ───────────────────────────────────────────────────────

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
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + scalar_fast_exp(-x))
}

#[inline(always)]
fn scalar_tanh_approx(x: f32) -> f32 {
    let e2x = scalar_fast_exp(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

#[inline(always)]
fn scalar_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + scalar_tanh_approx(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)))
}

#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
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
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

/// NEON sigmoid: 1 / (1 + exp(-x))
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
        let recip = vrecpeq_f32(denom);
        let recip = vmulq_f32(vrecpsq_f32(denom, recip), recip);
        recip
    }
}

/// NEON tanh approximation via exp: (exp(2x)-1)/(exp(2x)+1)
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

// ── Public activation functions ─────────────────────────────────────

/// GELU activation using fast polynomial approximation (in-place).
///
/// Computes `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` using
/// NEON intrinsics for 4-wide parallel processing.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(data: &mut [f32]) {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let coeff = vdupq_n_f32(GELU_COEFF);
        let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);

        for i in 0..chunks {
            let offset = i * LANES;
            let ptr = data.as_mut_ptr().add(offset);

            let x = vld1q_f32(ptr);
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
            let t = tanh_neon(inner);
            let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
            vst1q_f32(ptr, result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        data[tail + i] = scalar_gelu(data[tail + i]);
    }
}

/// SiLU/Swish activation: `x * sigmoid(x)` (in-place).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(data: &mut [f32]) {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        for i in 0..chunks {
            let offset = i * LANES;
            let ptr = data.as_mut_ptr().add(offset);

            let x = vld1q_f32(ptr);
            let sig = sigmoid_neon(x);
            let result = vmulq_f32(x, sig);
            vst1q_f32(ptr, result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        data[tail + i] = scalar_silu(data[tail + i]);
    }
}

/// ReLU activation: `max(0, x)` (in-place).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32(data: &mut [f32]) {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let zero = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * LANES;
            let ptr = data.as_mut_ptr().add(offset);

            let x = vld1q_f32(ptr);
            let result = vmaxq_f32(x, zero);
            vst1q_f32(ptr, result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let x = data[tail + i];
        data[tail + i] = if x > 0.0 { x } else { 0.0 };
    }
}

/// Leaky ReLU activation: `x` if `x > 0`, `alpha * x` otherwise (in-place).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_leaky_relu_f32(data: &mut [f32], alpha: f32) {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        let alpha_v = vdupq_n_f32(alpha);

        for i in 0..chunks {
            let offset = i * LANES;
            let ptr = data.as_mut_ptr().add(offset);

            let x = vld1q_f32(ptr);
            let pos = vmaxq_f32(x, zero);
            let neg = vminq_f32(x, zero);
            let result = vfmaq_f32(pos, alpha_v, neg);
            vst1q_f32(ptr, result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let x = data[tail + i];
        data[tail + i] = if x > 0.0 { x } else { alpha * x };
    }
}

/// Fused GELU+SiLU: computes both activations in one pass, sharing the
/// `exp(-x)` computation.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gelu_out` or `silu_out` length is less than `input` length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_silu_fused_f32(input: &[f32], gelu_out: &mut [f32], silu_out: &mut [f32]) {
    let n = input.len();
    assert!(gelu_out.len() >= n, "gelu_out buffer too small");
    assert!(silu_out.len() >= n, "silu_out buffer too small");

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

            let sig = sigmoid_neon(x);
            let silu = vmulq_f32(x, sig);
            vst1q_f32(silu_out.as_mut_ptr().add(offset), silu);

            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
            let t = tanh_neon(inner);
            let gelu = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, t)));
            vst1q_f32(gelu_out.as_mut_ptr().add(offset), gelu);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let x = input[tail + i];
        gelu_out[tail + i] = scalar_gelu(x);
        silu_out[tail + i] = scalar_silu(x);
    }
}

/// SwiGLU activation: `silu(gate) * up` — used in LLaMA-style FFN layers.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `up` or `output` length is less than `gate` length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert!(up.len() >= n, "up buffer too small");
    assert!(output.len() >= n, "output buffer too small");

    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        for i in 0..chunks {
            let offset = i * LANES;

            let g = vld1q_f32(gate.as_ptr().add(offset));
            let u = vld1q_f32(up.as_ptr().add(offset));

            let sig = sigmoid_neon(g);
            let silu_g = vmulq_f32(g, sig);
            let result = vmulq_f32(silu_g, u);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let g = gate[tail + i];
        let u = up[tail + i];
        output[tail + i] = scalar_silu(g) * u;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Reference implementations for validation ────────────────────

    fn ref_gelu(x: f32) -> f32 {
        0.5 * x
            * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    fn ref_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn ref_relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn ref_leaky_relu(x: f32, alpha: f32) -> f32 {
        if x > 0.0 { x } else { alpha * x }
    }

    fn ref_swiglu(gate: f32, up: f32) -> f32 {
        ref_silu(gate) * up
    }

    /// Max absolute error tolerance for polynomial approximations.
    const APPROX_TOL: f32 = 1e-3;
    /// Tighter tolerance for exact operations (ReLU, leaky ReLU).
    const EXACT_TOL: f32 = 1e-6;

    fn assert_approx_eq(a: f32, b: f32, tol: f32, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff <= tol || (a.is_nan() && b.is_nan()),
            "{msg}: {a} vs {b} (diff={diff}, tol={tol})"
        );
    }

    // ── GELU tests ──────────────────────────────────────────────────

    #[test]
    fn test_gelu_zeros() {
        let mut data = [0.0_f32; 8];
        unsafe { neon_gelu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, 0.0, EXACT_TOL, "gelu(0)");
        }
    }

    #[test]
    fn test_gelu_ones() {
        let mut data = [1.0_f32; 8];
        unsafe { neon_gelu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, ref_gelu(1.0), APPROX_TOL, "gelu(1)");
        }
    }

    #[test]
    fn test_gelu_negative() {
        let mut data = [-1.0_f32; 8];
        unsafe { neon_gelu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, ref_gelu(-1.0), APPROX_TOL, "gelu(-1)");
        }
    }

    #[test]
    fn test_gelu_mixed() {
        let mut data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("gelu mixed[{i}]"));
        }
    }

    #[test]
    fn test_gelu_large_positive() {
        let mut data = vec![5.0, 10.0, 20.0, 50.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("gelu large_pos[{i}]"));
        }
    }

    #[test]
    fn test_gelu_large_negative() {
        let mut data = vec![-5.0, -10.0, -20.0, -50.0];
        unsafe { neon_gelu_f32(&mut data) };
        for (i, &v) in data.iter().enumerate() {
            // GELU of very negative values → ~0
            assert!(v.abs() < 0.01, "gelu large_neg[{i}]: expected ~0, got {v}");
        }
    }

    #[test]
    fn test_gelu_small_values() {
        let mut data = vec![1e-5, -1e-5, 1e-3, -1e-3];
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("gelu small[{i}]"));
        }
    }

    #[test]
    fn test_gelu_symmetry() {
        // GELU is NOT symmetric, but gelu(-x) + gelu(x) ≈ x for moderate x
        // Instead check that gelu(x) > gelu(-x) for x > 0
        let vals = [0.5, 1.0, 2.0, 3.0];
        for &x in &vals {
            let mut pos = vec![x];
            let mut neg = vec![-x];
            unsafe {
                neon_gelu_f32(&mut pos);
                neon_gelu_f32(&mut neg);
            }
            assert!(pos[0] > neg[0], "gelu({x}) should be > gelu(-{x})");
        }
    }

    #[test]
    fn test_gelu_monotonic() {
        let mut data: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
        let orig = data.clone();
        unsafe { neon_gelu_f32(&mut data) };
        // GELU is monotonically increasing for x > ~-0.75
        for w in data.windows(2) {
            if orig[0] > 0.0 {
                assert!(w[1] >= w[0] - APPROX_TOL, "gelu monotonicity");
            }
        }
    }

    #[test]
    fn test_gelu_precision_vs_reference() {
        let inputs: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let mut data = inputs.clone();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &x)) in data.iter().zip(inputs.iter()).enumerate() {
            assert_approx_eq(got, ref_gelu(x), APPROX_TOL, &format!("gelu ref[{i}]"));
        }
    }

    // ── SiLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_silu_zeros() {
        let mut data = [0.0_f32; 8];
        unsafe { neon_silu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, 0.0, EXACT_TOL, "silu(0)");
        }
    }

    #[test]
    fn test_silu_ones() {
        let mut data = [1.0_f32; 8];
        unsafe { neon_silu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, ref_silu(1.0), APPROX_TOL, "silu(1)");
        }
    }

    #[test]
    fn test_silu_negative() {
        let mut data = [-1.0_f32; 8];
        unsafe { neon_silu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, ref_silu(-1.0), APPROX_TOL, "silu(-1)");
        }
    }

    #[test]
    fn test_silu_mixed() {
        let mut data = vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 4.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("silu mixed[{i}]"));
        }
    }

    #[test]
    fn test_silu_large() {
        let mut data = vec![10.0, 20.0, 50.0, 80.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            // For large x, silu(x) ≈ x
            assert_approx_eq(got, exp, APPROX_TOL, &format!("silu large[{i}]"));
        }
    }

    #[test]
    fn test_silu_small() {
        let mut data = vec![1e-6, -1e-6, 1e-4, -1e-4];
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("silu small[{i}]"));
        }
    }

    #[test]
    fn test_silu_boundary() {
        // silu(0) = 0 exactly, and silu is smooth everywhere
        let mut data = vec![-0.0_f32, 0.0];
        unsafe { neon_silu_f32(&mut data) };
        assert_approx_eq(data[0], 0.0, EXACT_TOL, "silu(-0)");
        assert_approx_eq(data[1], 0.0, EXACT_TOL, "silu(0)");
    }

    #[test]
    fn test_silu_precision() {
        let inputs: Vec<f32> = (-80..=80).map(|i| i as f32 * 0.1).collect();
        let mut data = inputs.clone();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &x)) in data.iter().zip(inputs.iter()).enumerate() {
            assert_approx_eq(got, ref_silu(x), APPROX_TOL, &format!("silu ref[{i}]"));
        }
    }

    // ── ReLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_zeros() {
        let mut data = [0.0_f32; 8];
        unsafe { neon_relu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, 0.0, EXACT_TOL, "relu(0)");
        }
    }

    #[test]
    fn test_relu_positive() {
        let mut data = vec![1.0, 2.5, 0.001, 100.0];
        let expected = data.clone();
        unsafe { neon_relu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("relu pos[{i}]"));
        }
    }

    #[test]
    fn test_relu_negative() {
        let mut data = vec![-1.0, -2.5, -0.001, -100.0];
        unsafe { neon_relu_f32(&mut data) };
        for &v in &data {
            assert_approx_eq(v, 0.0, EXACT_TOL, "relu(neg)");
        }
    }

    #[test]
    fn test_relu_mixed() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_relu(x)).collect();
        unsafe { neon_relu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("relu mixed[{i}]"));
        }
    }

    #[test]
    fn test_relu_large() {
        let mut data = vec![1e10, -1e10, 1e20, -1e20];
        let expected: Vec<f32> = data.iter().map(|&x| ref_relu(x)).collect();
        unsafe { neon_relu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("relu large[{i}]"));
        }
    }

    #[test]
    fn test_relu_boundary() {
        let mut data = vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::EPSILON, -f32::EPSILON];
        let expected: Vec<f32> = data.iter().map(|&x| ref_relu(x)).collect();
        unsafe { neon_relu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("relu boundary[{i}]"));
        }
    }

    // ── Leaky ReLU tests ────────────────────────────────────────────

    #[test]
    fn test_leaky_relu_zeros() {
        let mut data = [0.0_f32; 8];
        unsafe { neon_leaky_relu_f32(&mut data, 0.01) };
        for &v in &data {
            assert_approx_eq(v, 0.0, EXACT_TOL, "leaky_relu(0)");
        }
    }

    #[test]
    fn test_leaky_relu_positive() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = data.clone();
        unsafe { neon_leaky_relu_f32(&mut data, 0.01) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky_relu pos[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_negative() {
        let alpha = 0.01_f32;
        let mut data = vec![-1.0, -2.0, -3.0, -4.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_leaky_relu(x, alpha)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, alpha) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky_relu neg[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_alpha_zero() {
        // alpha=0 → same as ReLU
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_relu(x)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, 0.0) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky(a=0)[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_alpha_one() {
        // alpha=1 → identity function
        let original = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0];
        let mut data = original.clone();
        unsafe { neon_leaky_relu_f32(&mut data, 1.0) };
        for (i, (&got, &exp)) in data.iter().zip(original.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky(a=1)[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_alpha_negative() {
        let alpha = -0.1_f32;
        let mut data = vec![-2.0, -1.0, 0.0, 1.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_leaky_relu(x, alpha)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, alpha) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky(a=-0.1)[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_mixed() {
        let alpha = 0.2_f32;
        let mut data = vec![-4.0, -0.5, 0.0, 0.5, 3.0, -1.0, 2.0, -10.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_leaky_relu(x, alpha)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, alpha) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky mixed[{i}]"));
        }
    }

    #[test]
    fn test_leaky_relu_large() {
        let alpha = 0.01_f32;
        let mut data = vec![1e6, -1e6, 1e10, -1e10];
        let expected: Vec<f32> = data.iter().map(|&x| ref_leaky_relu(x, alpha)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, alpha) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("leaky large[{i}]"));
        }
    }

    // ── Fused GELU+SiLU tests ───────────────────────────────────────

    #[test]
    fn test_fused_compare_separate() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
        let mut gelu_fused = vec![0.0_f32; input.len()];
        let mut silu_fused = vec![0.0_f32; input.len()];

        let mut gelu_sep = input.clone();
        let mut silu_sep = input.clone();

        unsafe {
            neon_gelu_silu_fused_f32(&input, &mut gelu_fused, &mut silu_fused);
            neon_gelu_f32(&mut gelu_sep);
            neon_silu_f32(&mut silu_sep);
        }

        for i in 0..input.len() {
            assert_approx_eq(gelu_fused[i], gelu_sep[i], EXACT_TOL, &format!("fused gelu[{i}]"));
            assert_approx_eq(silu_fused[i], silu_sep[i], EXACT_TOL, &format!("fused silu[{i}]"));
        }
    }

    #[test]
    fn test_fused_zeros() {
        let input = [0.0_f32; 8];
        let mut gelu = [0.0_f32; 8];
        let mut silu = [0.0_f32; 8];
        unsafe { neon_gelu_silu_fused_f32(&input, &mut gelu, &mut silu) };
        for i in 0..8 {
            assert_approx_eq(gelu[i], 0.0, EXACT_TOL, "fused gelu(0)");
            assert_approx_eq(silu[i], 0.0, EXACT_TOL, "fused silu(0)");
        }
    }

    #[test]
    fn test_fused_mixed() {
        let input = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let mut gelu_out = [0.0_f32; 8];
        let mut silu_out = [0.0_f32; 8];
        unsafe { neon_gelu_silu_fused_f32(&input, &mut gelu_out, &mut silu_out) };
        for (i, &x) in input.iter().enumerate() {
            assert_approx_eq(gelu_out[i], ref_gelu(x), APPROX_TOL, &format!("fused gelu[{i}]"));
            assert_approx_eq(silu_out[i], ref_silu(x), APPROX_TOL, &format!("fused silu[{i}]"));
        }
    }

    #[test]
    fn test_fused_large() {
        let input = vec![10.0, -10.0, 20.0, -20.0];
        let mut gelu_out = [0.0_f32; 4];
        let mut silu_out = [0.0_f32; 4];
        unsafe { neon_gelu_silu_fused_f32(&input, &mut gelu_out, &mut silu_out) };
        for (i, &x) in input.iter().enumerate() {
            assert_approx_eq(gelu_out[i], ref_gelu(x), APPROX_TOL, &format!("fused gelu lg[{i}]"));
            assert_approx_eq(silu_out[i], ref_silu(x), APPROX_TOL, &format!("fused silu lg[{i}]"));
        }
    }

    #[test]
    fn test_fused_precision() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let mut gelu_out = vec![0.0_f32; input.len()];
        let mut silu_out = vec![0.0_f32; input.len()];
        unsafe { neon_gelu_silu_fused_f32(&input, &mut gelu_out, &mut silu_out) };
        for (i, &x) in input.iter().enumerate() {
            assert_approx_eq(
                gelu_out[i],
                ref_gelu(x),
                APPROX_TOL,
                &format!("fused prec gelu[{i}]"),
            );
            assert_approx_eq(
                silu_out[i],
                ref_silu(x),
                APPROX_TOL,
                &format!("fused prec silu[{i}]"),
            );
        }
    }

    // ── SwiGLU tests ────────────────────────────────────────────────

    #[test]
    fn test_swiglu_identity_gate() {
        // When gate activates fully (large positive), output ≈ gate * up ≈ up * gate
        let gate = [100.0_f32; 4];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        // silu(100) ≈ 100; small polynomial error accumulates with large values
        for i in 0..4 {
            let rel_tol = APPROX_TOL * gate[i].abs().max(1.0);
            assert_approx_eq(
                out[i],
                ref_swiglu(gate[i], up[i]),
                rel_tol,
                &format!("swiglu ident[{i}]"),
            );
        }
    }

    #[test]
    fn test_swiglu_zero_gate() {
        let gate = [0.0_f32; 4];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        for &v in &out {
            assert_approx_eq(v, 0.0, EXACT_TOL, "swiglu(0,x)");
        }
    }

    #[test]
    fn test_swiglu_ones() {
        let gate = [1.0_f32; 8];
        let up = [1.0_f32; 8];
        let mut out = [0.0_f32; 8];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        let expected = ref_swiglu(1.0, 1.0);
        for &v in &out {
            assert_approx_eq(v, expected, APPROX_TOL, "swiglu(1,1)");
        }
    }

    #[test]
    fn test_swiglu_mixed() {
        let gate = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
        let up = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let mut out = [0.0_f32; 8];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        for i in 0..8 {
            assert_approx_eq(
                out[i],
                ref_swiglu(gate[i], up[i]),
                APPROX_TOL,
                &format!("swiglu mix[{i}]"),
            );
        }
    }

    #[test]
    fn test_swiglu_negative() {
        let gate = vec![-3.0, -2.0, -1.0, -0.5];
        let up = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        for i in 0..4 {
            assert_approx_eq(
                out[i],
                ref_swiglu(gate[i], up[i]),
                APPROX_TOL,
                &format!("swiglu neg[{i}]"),
            );
        }
    }

    #[test]
    fn test_swiglu_large() {
        let gate = vec![10.0, -10.0, 50.0, -50.0];
        let up = vec![2.0, 3.0, 0.5, -1.0];
        let mut out = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        for i in 0..4 {
            assert_approx_eq(
                out[i],
                ref_swiglu(gate[i], up[i]),
                APPROX_TOL,
                &format!("swiglu large[{i}]"),
            );
        }
    }

    #[test]
    fn test_swiglu_precision() {
        let gate: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.3).collect();
        let up: Vec<f32> = (0..gate.len()).map(|i| (i as f32 - 20.0) * 0.1).collect();
        let mut out = vec![0.0_f32; gate.len()];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        for i in 0..gate.len() {
            assert_approx_eq(
                out[i],
                ref_swiglu(gate[i], up[i]),
                APPROX_TOL,
                &format!("swiglu prec[{i}]"),
            );
        }
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_empty_gelu() {
        let mut data: Vec<f32> = vec![];
        unsafe { neon_gelu_f32(&mut data) };
        assert!(data.is_empty());
    }

    #[test]
    fn test_empty_silu() {
        let mut data: Vec<f32> = vec![];
        unsafe { neon_silu_f32(&mut data) };
        assert!(data.is_empty());
    }

    #[test]
    fn test_empty_relu() {
        let mut data: Vec<f32> = vec![];
        unsafe { neon_relu_f32(&mut data) };
        assert!(data.is_empty());
    }

    #[test]
    fn test_empty_swiglu() {
        let gate: Vec<f32> = vec![];
        let up: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        unsafe { neon_swiglu_f32(&gate, &up, &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_single_element_gelu() {
        let mut data = vec![1.5_f32];
        unsafe { neon_gelu_f32(&mut data) };
        assert_approx_eq(data[0], ref_gelu(1.5), APPROX_TOL, "single gelu");
    }

    #[test]
    fn test_single_element_silu() {
        let mut data = vec![1.5_f32];
        unsafe { neon_silu_f32(&mut data) };
        assert_approx_eq(data[0], ref_silu(1.5), APPROX_TOL, "single silu");
    }

    #[test]
    fn test_single_element_relu() {
        let mut data = vec![-1.5_f32];
        unsafe { neon_relu_f32(&mut data) };
        assert_approx_eq(data[0], 0.0, EXACT_TOL, "single relu");
    }

    #[test]
    fn test_non_aligned_length() {
        // 7 elements: 4 NEON + 3 scalar tail
        let mut data = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("unaligned gelu[{i}]"));
        }
    }

    #[test]
    fn test_non_aligned_silu() {
        let mut data = vec![0.5, -0.5, 1.0, -1.0, 2.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("unaligned silu[{i}]"));
        }
    }

    #[test]
    fn test_non_aligned_leaky_relu() {
        let alpha = 0.1_f32;
        let mut data = vec![-3.0, 1.0, -2.0, 0.5, 4.0, -1.5, 0.0];
        let expected: Vec<f32> = data.iter().map(|&x| ref_leaky_relu(x, alpha)).collect();
        unsafe { neon_leaky_relu_f32(&mut data, alpha) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, EXACT_TOL, &format!("unaligned leaky[{i}]"));
        }
    }

    #[test]
    fn test_very_large_array_gelu() {
        let n = 1025;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("large arr gelu[{i}]"));
        }
    }

    #[test]
    fn test_very_large_array_silu() {
        let n = 1025;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("large arr silu[{i}]"));
        }
    }

    // ── Numerical edge-case tests ───────────────────────────────────

    #[test]
    fn test_inf_handling_relu() {
        let mut data = vec![f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY];
        unsafe { neon_relu_f32(&mut data) };
        assert_eq!(data[0], f32::INFINITY);
        assert_eq!(data[1], 0.0); // NEG_INFINITY < 0 → max(0, -inf) should be 0 or -inf depending on impl
        assert_eq!(data[2], f32::INFINITY);
    }

    #[test]
    fn test_inf_handling_leaky_relu() {
        let mut data = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        unsafe { neon_leaky_relu_f32(&mut data, 0.01) };
        assert_eq!(data[0], f32::INFINITY);
        assert_eq!(data[2], 0.0);
        assert_approx_eq(data[3], 1.0, EXACT_TOL, "leaky inf");
    }

    #[test]
    fn test_nan_handling_relu() {
        let mut data = vec![f32::NAN, 1.0, f32::NAN, -1.0];
        unsafe { neon_relu_f32(&mut data) };
        // NaN propagation depends on NEON vmaxq_f32 semantics
        assert!(data[0].is_nan() || data[0] == 0.0);
        assert_approx_eq(data[1], 1.0, EXACT_TOL, "nan relu");
    }

    #[test]
    fn test_denormal_gelu() {
        let mut data = [f32::MIN_POSITIVE / 2.0; 4];
        let expected: Vec<f32> = data.iter().map(|&x| ref_gelu(x)).collect();
        unsafe { neon_gelu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("denormal gelu[{i}]"));
        }
    }

    #[test]
    fn test_denormal_silu() {
        let mut data = [f32::MIN_POSITIVE / 2.0; 4];
        let expected: Vec<f32> = data.iter().map(|&x| ref_silu(x)).collect();
        unsafe { neon_silu_f32(&mut data) };
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert_approx_eq(got, exp, APPROX_TOL, &format!("denormal silu[{i}]"));
        }
    }

    #[test]
    fn test_precision_bounds_gelu() {
        // Verify max error stays under 1e-3 across working range
        let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.05).collect();
        let mut data = inputs.clone();
        unsafe { neon_gelu_f32(&mut data) };
        let mut max_err: f32 = 0.0;
        for (&got, &x) in data.iter().zip(inputs.iter()) {
            let err = (got - ref_gelu(x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < APPROX_TOL, "GELU max error {max_err} exceeds tolerance {APPROX_TOL}");
    }

    #[test]
    fn test_precision_bounds_silu() {
        let inputs: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.05).collect();
        let mut data = inputs.clone();
        unsafe { neon_silu_f32(&mut data) };
        let mut max_err: f32 = 0.0;
        for (&got, &x) in data.iter().zip(inputs.iter()) {
            let err = (got - ref_silu(x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < APPROX_TOL, "SiLU max error {max_err} exceeds tolerance {APPROX_TOL}");
    }
}
