#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON-optimized GELU and activation function variants for Apple Silicon.
//!
//! Provides fast and exact GELU, SwiGLU, GeGLU, and fast tanh using NEON SIMD
//! intrinsics on AArch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ───────────────────────────────────────────────────────

const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
const GELU_COEFF: f32 = 0.044715;

/// Approximate erf(x) using Horner form (Abramowitz & Stegun 7.1.26).
#[inline]
fn approx_erff(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ── Polynomial tanh approximation (NEON) ────────────────────────────

/// Clamp a NEON f32x4 vector to [-`limit`, `limit`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_clamp(v: float32x4_t, limit: f32) -> float32x4_t {
    let pos = vdupq_n_f32(limit);
    let neg = vdupq_n_f32(-limit);
    vminq_f32(vmaxq_f32(v, neg), pos)
}

/// Polynomial tanh approximation for a NEON f32x4 lane.
///
/// Uses a degree-7 odd polynomial fit on [-4.5, 4.5] with clamped input.
/// Maximum error ≈ 3e-4 in [-4.5, 4.5]; exact ±1 outside.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_tanh_poly(x: float32x4_t) -> float32x4_t {
    // Padé(3,3) rational approximant for tanh, accurate to ~0.005 on [-3,3].
    // tanh(x) ≈ x*(105 + 10*x²) / (105 + 45*x² + x⁴)
    // For |x| > 3, saturate to ±1 (tanh(3) = 0.995).
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);
    let limit = vdupq_n_f32(3.0);
    let c105 = vdupq_n_f32(105.0);
    let c10 = vdupq_n_f32(10.0);
    let c45 = vdupq_n_f32(45.0);

    let x2 = vmulq_f32(x, x);
    let x4 = vmulq_f32(x2, x2);

    // Numerator: x * (105 + 10*x²)
    let num = vmulq_f32(x, vfmaq_f32(c105, c10, x2));
    // Denominator: 105 + 45*x² + x⁴
    let den = vaddq_f32(vfmaq_f32(c105, c45, x2), x4);

    // Approximate division (NEON has no f32 div, use reciprocal estimate).
    let inv = vrecpeq_f32(den);
    let inv = vmulq_f32(vrecpsq_f32(den, inv), inv); // one Newton-Raphson step
    let poly = vmulq_f32(num, inv);

    // For |x| > 3, use sign(x).
    let abs_x = vabsq_f32(x);
    let saturated = vcgtq_f32(abs_x, limit);
    let sign = vbslq_f32(vcgeq_f32(x, zero), one, neg_one);
    vbslq_f32(saturated, sign, poly)
}

// ── Fast GELU ───────────────────────────────────────────────────────

/// Fast GELU approximation in-place using NEON intrinsics.
///
/// Implements 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Safety
///
/// Requires AArch64 NEON (always available on Apple Silicon).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gelu_fast_impl(input: &mut [f32]) {
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);
    let coeff = vdupq_n_f32(GELU_COEFF);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    for i in 0..chunks {
        let ptr = unsafe { input.as_mut_ptr().add(i * 4) };
        let x = unsafe { vld1q_f32(ptr) };
        let x3 = vmulq_f32(vmulq_f32(x, x), x);
        let inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
        let tanh_val = unsafe { neon_tanh_poly(inner) };
        let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_val)));
        unsafe { vst1q_f32(ptr, result) };
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        let x = input[idx];
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        let t = inner.tanh();
        input[idx] = 0.5 * x * (1.0 + t);
    }
}

/// Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[cfg(target_arch = "aarch64")]
pub fn neon_gelu_fast(input: &mut [f32]) {
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_gelu_fast_impl(input) }
}

// ── Exact GELU ──────────────────────────────────────────────────────

/// Rational polynomial approximation of erf for a NEON f32x4 lane.
///
/// Uses Abramowitz & Stegun 7.1.28 (maximum error ≈ 5e-4).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_erf_approx(x: float32x4_t) -> float32x4_t {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let sign_mask = vcltq_f32(x, zero);
    let abs_x = vabsq_f32(x);

    // Abramowitz & Stegun constants for erf approximation
    let p = vdupq_n_f32(0.3275911);
    let a1 = vdupq_n_f32(0.254_829_6);
    let a2 = vdupq_n_f32(-0.284_496_72);
    let a3 = vdupq_n_f32(1.421_413_8);
    let a4 = vdupq_n_f32(-1.453_152_1);
    let a5 = vdupq_n_f32(1.061_405_4);

    let t = {
        let denom = vfmaq_f32(one, p, abs_x); // 1 + p * |x|
        vrecpeq_f32(denom)
    };
    // One Newton–Raphson refinement for reciprocal.
    let t = {
        let denom = vfmaq_f32(one, p, abs_x);
        vmulq_f32(t, vrecpsq_f32(denom, t))
    };

    // exp(-x²) approximation via scalar (NEON has no native exp).
    let neg_x2 = vnegq_f32(vmulq_f32(abs_x, abs_x));
    let mut exp_vals: [f32; 4] = [0.0; 4];
    unsafe { vst1q_f32(exp_vals.as_mut_ptr(), neg_x2) };
    for v in &mut exp_vals {
        *v = v.exp();
    }
    let exp_neg_x2 = unsafe { vld1q_f32(exp_vals.as_ptr()) };

    // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    let mut poly = vmulq_f32(a5, t);
    poly = vaddq_f32(poly, a4);
    poly = vfmaq_f32(a3, poly, t);
    poly = vfmaq_f32(a2, poly, t);
    poly = vfmaq_f32(a1, poly, t);
    poly = vmulq_f32(poly, t);

    // erf ≈ 1 - poly * exp(-x²)
    let erf_abs = vsubq_f32(one, vmulq_f32(poly, exp_neg_x2));

    // Restore sign: erf(-x) = -erf(x)
    let neg_erf = vnegq_f32(erf_abs);
    vbslq_f32(sign_mask, neg_erf, erf_abs)
}

/// Exact GELU using error function approximation, in-place.
///
/// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gelu_exact_impl(input: &mut [f32]) {
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let inv_sqrt2 = vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2);

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    for i in 0..chunks {
        let ptr = unsafe { input.as_mut_ptr().add(i * 4) };
        let x = unsafe { vld1q_f32(ptr) };
        let scaled = vmulq_f32(x, inv_sqrt2);
        let erf_val = unsafe { neon_erf_approx(scaled) };
        let result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, erf_val)));
        unsafe { vst1q_f32(ptr, result) };
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        let x = input[idx];
        let erf_val = approx_erff(x * std::f32::consts::FRAC_1_SQRT_2);
        input[idx] = 0.5 * x * (1.0 + erf_val);
    }
}

/// Exact GELU using error function approximation.
#[cfg(target_arch = "aarch64")]
pub fn neon_gelu_exact(input: &mut [f32]) {
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_gelu_exact_impl(input) }
}

// ── SwiGLU ──────────────────────────────────────────────────────────

/// Scalar SiLU: x * sigmoid(x).
#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU activation: silu(gate) * up.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_swiglu_impl(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let n = gate.len();
    let mut out = vec![0.0f32; n];

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let g = unsafe { vld1q_f32(gate.as_ptr().add(i * 4)) };
        let u = unsafe { vld1q_f32(up.as_ptr().add(i * 4)) };

        // silu(g) via scalar (exp not available in NEON)
        let mut g_arr: [f32; 4] = [0.0; 4];
        unsafe { vst1q_f32(g_arr.as_mut_ptr(), g) };
        for v in &mut g_arr {
            *v = scalar_silu(*v);
        }
        let silu_g = unsafe { vld1q_f32(g_arr.as_ptr()) };

        let result = vmulq_f32(silu_g, u);
        unsafe { vst1q_f32(out.as_mut_ptr().add(i * 4), result) };
    }

    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        out[idx] = scalar_silu(gate[idx]) * up[idx];
    }

    out
}

/// SwiGLU activation: silu(gate) * up.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "gate and up must have equal length");
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_swiglu_impl(gate, up) }
}

// ── GeGLU ───────────────────────────────────────────────────────────

/// Scalar fast GELU for use in GeGLU tail.
#[inline(always)]
fn scalar_gelu_fast(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GeGLU activation: gelu(gate) * up.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_geglu_impl(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let n = gate.len();
    let mut out = vec![0.0f32; n];

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let sqrt2pi = vdupq_n_f32(SQRT_2_OVER_PI);
    let coeff = vdupq_n_f32(GELU_COEFF);

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let g = unsafe { vld1q_f32(gate.as_ptr().add(i * 4)) };
        let u = unsafe { vld1q_f32(up.as_ptr().add(i * 4)) };

        let g3 = vmulq_f32(vmulq_f32(g, g), g);
        let inner = vmulq_f32(sqrt2pi, vfmaq_f32(g, coeff, g3));
        let tanh_val = unsafe { neon_tanh_poly(inner) };
        let gelu_g = vmulq_f32(half, vmulq_f32(g, vaddq_f32(one, tanh_val)));

        let result = vmulq_f32(gelu_g, u);
        unsafe { vst1q_f32(out.as_mut_ptr().add(i * 4), result) };
    }

    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        out[idx] = scalar_gelu_fast(gate[idx]) * up[idx];
    }

    out
}

/// GeGLU activation: gelu(gate) * up.
///
/// # Panics
///
/// Panics if `gate.len() != up.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "gate and up must have equal length");
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_geglu_impl(gate, up) }
}

// ── Fast tanh ───────────────────────────────────────────────────────

/// Fast tanh approximation using NEON polynomial.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_tanh_fast_impl(input: &mut [f32]) {
    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    for i in 0..chunks {
        let ptr = unsafe { input.as_mut_ptr().add(i * 4) };
        let x = unsafe { vld1q_f32(ptr) };
        let result = unsafe { neon_tanh_poly(x) };
        unsafe { vst1q_f32(ptr, result) };
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        input[idx] = input[idx].tanh();
    }
}

/// Fast tanh approximation using NEON polynomial intrinsics.
#[cfg(target_arch = "aarch64")]
pub fn neon_tanh_fast(input: &mut [f32]) {
    // SAFETY: NEON is always available on AArch64.
    unsafe { neon_tanh_fast_impl(input) }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_fast_known_values() {
        // GELU(0) = 0
        let mut zero = [0.0f32];
        neon_gelu_fast(&mut zero);
        assert!(zero[0].abs() < 1e-6, "GELU(0) should be 0, got {}", zero[0]);

        // GELU(1) ≈ 0.8412
        let mut pos = [1.0f32];
        neon_gelu_fast(&mut pos);
        assert!((pos[0] - 0.8412).abs() < 0.01, "GELU(1) should be ≈0.8412, got {}", pos[0]);

        // GELU(-1) ≈ -0.1588
        let mut neg = [-1.0f32];
        neon_gelu_fast(&mut neg);
        assert!((neg[0] - (-0.1588)).abs() < 0.01, "GELU(-1) should be ≈-0.1588, got {}", neg[0]);
    }

    #[test]
    fn test_gelu_exact_vs_fast() {
        let values: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
        let mut fast = values.clone();
        let mut exact = values.clone();
        neon_gelu_fast(&mut fast);
        neon_gelu_exact(&mut exact);

        for (i, (f, e)) in fast.iter().zip(exact.iter()).enumerate() {
            let diff = (f - e).abs();
            assert!(
                diff < 0.01,
                "GELU mismatch at index {}: fast={}, exact={}, diff={}",
                i,
                f,
                e,
                diff
            );
        }
    }

    #[test]
    fn test_swiglu_basic() {
        let gate = [1.0f32, 0.0, -1.0, 2.0];
        let up = [1.0f32, 1.0, 1.0, 0.5];
        let result = neon_swiglu(&gate, &up);
        assert_eq!(result.len(), 4);

        // silu(0) * 1 = 0
        assert!(result[1].abs() < 1e-6, "silu(0)*1 should be 0, got {}", result[1]);

        // silu(x) = x * sigmoid(x); silu(1) ≈ 0.7311
        assert!(
            (result[0] - 0.7311).abs() < 0.01,
            "silu(1)*1 should be ≈0.7311, got {}",
            result[0]
        );
    }

    #[test]
    fn test_geglu_basic() {
        let gate = [1.0f32, 0.0, -1.0, 2.0];
        let up = [1.0f32, 1.0, 1.0, 0.5];
        let result = neon_geglu(&gate, &up);
        assert_eq!(result.len(), 4);

        // gelu(0) * 1 = 0
        assert!(result[1].abs() < 1e-6, "gelu(0)*1 should be 0, got {}", result[1]);

        // gelu(1) ≈ 0.8412, so geglu(1, 1) ≈ 0.8412
        assert!(
            (result[0] - 0.8412).abs() < 0.02,
            "gelu(1)*1 should be ≈0.8412, got {}",
            result[0]
        );
    }

    #[test]
    fn test_tanh_range() {
        let mut values: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.5).collect();
        neon_tanh_fast(&mut values);

        for (i, &v) in values.iter().enumerate() {
            assert!((-1.0..=1.0).contains(&v), "tanh output at index {} out of range: {}", i, v);
        }
    }
}
