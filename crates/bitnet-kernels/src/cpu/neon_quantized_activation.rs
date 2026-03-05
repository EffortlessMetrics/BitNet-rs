#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON-optimized activation functions for quantized inference on Apple Silicon.
//!
//! Provides GELU, SiLU, ReLU, sigmoid, tanh, SwiGLU, GeGLU, and fast-GELU
//! using NEON SIMD intrinsics with polynomial approximations for transcendentals.
//! Each function processes 4 × f32 lanes at a time with scalar fallback for
//! remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Constants ───────────────────────────────────────────────────────

/// √(2/π) used in GELU tanh approximation.
#[cfg(target_arch = "aarch64")]
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Cubic coefficient in GELU tanh approximation.
#[cfg(target_arch = "aarch64")]
const GELU_COEFF: f32 = 0.044_715;

// ── Polynomial exp approximation (NEON) ─────────────────────────────

/// Fast exp approximation using a degree-6 minimax polynomial on [-87, 87].
///
/// Clamps input to avoid overflow/underflow, then computes exp via
/// range reduction to [0, ln2) and a degree-6 polynomial.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_exp_f32x4(x: float32x4_t) -> float32x4_t {
    unsafe {
        let min_val = vdupq_n_f32(-87.0);
        let max_val = vdupq_n_f32(87.0);
        let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

        // Range reduction: x = n * ln2 + r, where n = round(x / ln2)
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

        let n = vrndnq_f32(vmulq_f32(x, log2e));
        let r = vmlsq_f32(x, n, ln2); // r = x - n * ln2

        // Degree-6 minimax polynomial for exp(r) on [0, ln2)
        let c0 = vdupq_n_f32(1.0);
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.166_666_7);
        let c4 = vdupq_n_f32(0.041_666_67);
        let c5 = vdupq_n_f32(0.008_333_334);
        let c6 = vdupq_n_f32(0.001_388_889);

        // Horner evaluation: p = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
        let mut p = vmlaq_f32(c5, r, c6);
        p = vmlaq_f32(c4, r, p);
        p = vmlaq_f32(c3, r, p);
        p = vmlaq_f32(c2, r, p);
        p = vmlaq_f32(c1, r, p);
        p = vmlaq_f32(c0, r, p);

        // Reconstruct: exp(x) = p * 2^n
        // Use integer bit manipulation for 2^n
        let ni = vcvtq_s32_f32(n);
        let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23));

        vmulq_f32(p, pow2n)
    }
}

/// Scalar fast exp approximation matching the NEON polynomial.
#[inline(always)]
fn scalar_exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let log2e = std::f32::consts::LOG2_E;
    let ln2 = std::f32::consts::LN_2;

    let n = (x * log2e).round();
    let r = x - n * ln2;

    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_7
                    + r * (0.041_666_67 + r * (0.008_333_334 + r * 0.001_388_889)))));

    let ni = n as i32;
    let pow2n = f32::from_bits(((ni + 127) as u32) << 23);
    p * pow2n
}

// ── Sigmoid helpers ─────────────────────────────────────────────────

/// NEON sigmoid: 1 / (1 + exp(-x))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_sigmoid_vec(x: float32x4_t) -> float32x4_t {
    unsafe {
        let one = vdupq_n_f32(1.0);
        let neg_x = vnegq_f32(x);
        let exp_neg = neon_exp_f32x4(neg_x);
        let denom = vaddq_f32(one, exp_neg);
        // reciprocal: use vrecpeq + Newton-Raphson step
        let recip = vrecpeq_f32(denom);

        vmulq_f32(recip, vrecpsq_f32(denom, recip))
    }
}

/// Scalar sigmoid using fast exp.
#[inline(always)]
fn scalar_sigmoid_fast(x: f32) -> f32 {
    1.0 / (1.0 + scalar_exp_fast(-x))
}

// ── Tanh helper ─────────────────────────────────────────────────────

/// NEON tanh: (exp(2x) - 1) / (exp(2x) + 1)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_tanh_vec(x: float32x4_t) -> float32x4_t {
    unsafe {
        let two = vdupq_n_f32(2.0);
        let one = vdupq_n_f32(1.0);
        let two_x = vmulq_f32(x, two);
        let exp2x = neon_exp_f32x4(two_x);
        let num = vsubq_f32(exp2x, one);
        let den = vaddq_f32(exp2x, one);
        let recip = vrecpeq_f32(den);
        let recip = vmulq_f32(recip, vrecpsq_f32(den, recip));
        vmulq_f32(num, recip)
    }
}

/// Scalar tanh using fast exp.
#[inline(always)]
fn scalar_tanh_fast(x: f32) -> f32 {
    let exp2x = scalar_exp_fast(2.0 * x);
    (exp2x - 1.0) / (exp2x + 1.0)
}

// ── GELU helper ─────────────────────────────────────────────────────

/// NEON GELU (tanh approximation): 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_gelu_vec(x: float32x4_t) -> float32x4_t {
    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let coeff = vdupq_n_f32(GELU_COEFF);
        let sqrt_2_pi = vdupq_n_f32(SQRT_2_OVER_PI);

        // x^3
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);

        // inner = sqrt(2/pi) * (x + coeff * x^3)
        let inner = vmulq_f32(sqrt_2_pi, vmlaq_f32(x, coeff, x3));

        // 0.5 * x * (1 + tanh(inner))
        let tanh_val = neon_tanh_vec(inner);
        let scale = vmulq_f32(half, vaddq_f32(one, tanh_val));
        vmulq_f32(x, scale)
    }
}

/// Scalar GELU using tanh approximation.
#[inline(always)]
fn scalar_gelu(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + scalar_tanh_fast(inner))
}

// ── ReLU ────────────────────────────────────────────────────────────

/// Compute ReLU activation (max(0, x)) using NEON intrinsics.
///
/// Processes 4 × f32 lanes at a time with scalar fallback for remainder.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            vst1q_f32(output.as_mut_ptr().add(off), vmaxq_f32(v, zero));
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let x = input[tail + i];
        output[tail + i] = if x > 0.0 { x } else { 0.0 };
    }
}

// ── Sigmoid ─────────────────────────────────────────────────────────

/// Compute sigmoid activation (1/(1+exp(-x))) using NEON polynomial exp.
///
/// Uses a degree-6 minimax polynomial for exp approximation, yielding
/// max error < 1e-4 across the practical range.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            let result = neon_sigmoid_vec(v);
            vst1q_f32(output.as_mut_ptr().add(off), result);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = scalar_sigmoid_fast(input[tail + i]);
    }
}

// ── Tanh ────────────────────────────────────────────────────────────

/// Compute tanh activation using NEON polynomial exp.
///
/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_tanh_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            let result = neon_tanh_vec(v);
            vst1q_f32(output.as_mut_ptr().add(off), result);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = scalar_tanh_fast(input[tail + i]);
    }
}

// ── SiLU / Swish ────────────────────────────────────────────────────

/// Compute SiLU/Swish activation (x * sigmoid(x)) using NEON polynomial exp.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            let sig = neon_sigmoid_vec(v);
            vst1q_f32(output.as_mut_ptr().add(off), vmulq_f32(v, sig));
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let x = input[tail + i];
        output[tail + i] = x * scalar_sigmoid_fast(x);
    }
}

// ── GELU (tanh approximation) ───────────────────────────────────────

/// Compute GELU activation using the tanh approximation:
/// `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            let result = neon_gelu_vec(v);
            vst1q_f32(output.as_mut_ptr().add(off), result);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = scalar_gelu(input[tail + i]);
    }
}

// ── Fast GELU (polynomial approximation) ────────────────────────────

/// Fast GELU approximation using a polynomial fit:
/// `x * sigmoid(1.702 * x)`
///
/// This avoids the cubic term and tanh computation, trading a small
/// accuracy reduction for higher throughput.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `output` must be at least as long as `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fast_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let coeff = vdupq_n_f32(1.702);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(input.as_ptr().add(off));
            let scaled = vmulq_f32(v, coeff);
            let sig = neon_sigmoid_vec(scaled);
            vst1q_f32(output.as_mut_ptr().add(off), vmulq_f32(v, sig));
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let x = input[tail + i];
        output[tail + i] = x * scalar_sigmoid_fast(1.702 * x);
    }
}

/// Scalar fast GELU reference.
#[inline(always)]
fn scalar_fast_gelu(x: f32) -> f32 {
    x * scalar_sigmoid_fast(1.702 * x)
}

// ── SwiGLU ──────────────────────────────────────────────────────────

/// Compute SwiGLU activation: `silu(gate) * up`.
///
/// Commonly used in LLaMA-style FFN blocks where the gate and up
/// projections are separate linear layers.
///
/// # Panics
///
/// Panics if `output.len() < gate.len()` or `up.len() < gate.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `up` and `output` must be at least as long as `gate`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert!(up.len() >= n, "up buffer too small");
    assert!(output.len() >= n, "output buffer too small");
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let g = vld1q_f32(gate.as_ptr().add(off));
            let u = vld1q_f32(up.as_ptr().add(off));
            let sig_g = neon_sigmoid_vec(g);
            let silu_g = vmulq_f32(g, sig_g);
            vst1q_f32(output.as_mut_ptr().add(off), vmulq_f32(silu_g, u));
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let g = gate[tail + i];
        let u = up[tail + i];
        output[tail + i] = g * scalar_sigmoid_fast(g) * u;
    }
}

// ── GeGLU ───────────────────────────────────────────────────────────

/// Compute GeGLU activation: `gelu(gate) * up`.
///
/// Used in some transformer FFN variants where the gate projection
/// passes through GELU before multiplying with the up projection.
///
/// # Panics
///
/// Panics if `output.len() < gate.len()` or `up.len() < gate.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `up` and `output` must be at least as long as `gate`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_geglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert!(up.len() >= n, "up buffer too small");
    assert!(output.len() >= n, "output buffer too small");
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let g = vld1q_f32(gate.as_ptr().add(off));
            let u = vld1q_f32(up.as_ptr().add(off));
            let gelu_g = neon_gelu_vec(g);
            vst1q_f32(output.as_mut_ptr().add(off), vmulq_f32(gelu_g, u));
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let g = gate[tail + i];
        let u = up[tail + i];
        output[tail + i] = scalar_gelu(g) * u;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    const EPS: f32 = 1e-3;
    #[cfg(target_arch = "aarch64")]
    const STRICT_EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        let diff = (a - b).abs();
        // Absolute tolerance OR relative tolerance for larger values
        diff < eps
            || diff < eps * a.abs().max(b.abs()).max(1.0)
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
    }

    /// Reference scalar sigmoid using std exp.
    fn ref_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Reference scalar tanh using std.
    fn ref_tanh(x: f32) -> f32 {
        x.tanh()
    }

    /// Reference scalar GELU (tanh approximation).
    fn ref_gelu(x: f32) -> f32 {
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// Reference scalar SiLU.
    fn ref_silu(x: f32) -> f32 {
        x * ref_sigmoid(x)
    }

    /// Reference scalar fast GELU.
    fn ref_fast_gelu(x: f32) -> f32 {
        x * ref_sigmoid(1.702 * x)
    }

    // ── Helper to run with various sizes ────────────────────────────

    fn make_input(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect()
    }

    // ── ReLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_size_1() {
        let input = [0.5_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output[0], 0.5);
    }

    #[test]
    fn test_relu_size_4() {
        let input = [-1.0, 0.0, 1.0, 2.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_size_8() {
        let input = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0_f32];
        let mut output = [0.0_f32; 8];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output, [0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_relu_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_relu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert_eq!(*o, if *x > 0.0 { *x } else { 0.0 });
        }
    }

    #[test]
    fn test_relu_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_relu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert_eq!(*o, if *x > 0.0 { *x } else { 0.0 });
        }
    }

    #[test]
    fn test_relu_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_relu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert_eq!(*o, if *x > 0.0 { *x } else { 0.0 });
        }
    }

    #[test]
    fn test_relu_zeros() {
        let input = [0.0_f32; 8];
        let mut output = [1.0_f32; 8];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relu_large_positive() {
        let input = [1e6_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert!(output.iter().all(|&v| v == 1e6));
    }

    #[test]
    fn test_relu_large_negative() {
        let input = [-1e6_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relu_near_zero() {
        let input = [-1e-7, 1e-7, -1e-10, 1e-10_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1e-7);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 1e-10);
    }

    // ── Sigmoid tests ───────────────────────────────────────────────

    #[test]
    fn test_sigmoid_size_1() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.5, EPS));
    }

    #[test]
    fn test_sigmoid_size_4() {
        let input = [-2.0, -1.0, 0.0, 1.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_sigmoid(*x), EPS), "sigmoid({x}): got {o}");
        }
    }

    #[test]
    fn test_sigmoid_size_8() {
        let input = make_input(8);
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_sigmoid(*x), EPS), "sigmoid({x}): got {o}");
        }
    }

    #[test]
    fn test_sigmoid_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_sigmoid(*x), EPS));
        }
    }

    #[test]
    fn test_sigmoid_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_sigmoid(*x), EPS));
        }
    }

    #[test]
    fn test_sigmoid_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_sigmoid(*x), EPS));
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        let input = [-100.0, -10.0, 0.0, 10.0, 100.0_f32];
        let mut output = [0.0_f32; 5];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for o in &output {
            assert!(*o >= 0.0 && *o <= 1.0, "sigmoid out of [0,1]: {o}");
        }
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let input = [50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 1.0, EPS));
        }
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let input = [-50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, EPS));
        }
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let input = [-3.0, -1.0, 1.0, 3.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert!(approx_eq(output[0] + output[3], 1.0, EPS));
        assert!(approx_eq(output[1] + output[2], 1.0, EPS));
    }

    // ── Tanh tests ──────────────────────────────────────────────────

    #[test]
    fn test_tanh_size_1() {
        let input = [0.0_f32];
        let mut output = [1.0_f32; 1];
        unsafe { neon_tanh_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, EPS));
    }

    #[test]
    fn test_tanh_size_4() {
        let input = [-2.0, -1.0, 0.0, 1.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_tanh(*x), EPS), "tanh({x}): got {o}");
        }
    }

    #[test]
    fn test_tanh_size_8() {
        let input = make_input(8);
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_tanh(*x), EPS));
        }
    }

    #[test]
    fn test_tanh_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_tanh(*x), EPS));
        }
    }

    #[test]
    fn test_tanh_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_tanh(*x), EPS));
        }
    }

    #[test]
    fn test_tanh_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_tanh(*x), EPS));
        }
    }

    #[test]
    fn test_tanh_bounds() {
        let input = [-100.0, -10.0, 0.0, 10.0, 100.0_f32];
        let mut output = [0.0_f32; 5];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for o in &output {
            assert!(*o >= -1.0 && *o <= 1.0, "tanh out of [-1,1]: {o}");
        }
    }

    #[test]
    fn test_tanh_large_positive() {
        let input = [50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 1.0, EPS));
        }
    }

    #[test]
    fn test_tanh_large_negative() {
        let input = [-50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_tanh_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, -1.0, EPS));
        }
    }

    #[test]
    fn test_tanh_odd_symmetry() {
        let input = [-2.0, -1.0, 1.0, 2.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_tanh_f32(&input, &mut output) };
        assert!(approx_eq(output[0] + output[3], 0.0, EPS));
        assert!(approx_eq(output[1] + output[2], 0.0, EPS));
    }

    // ── SiLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_silu_size_1() {
        let input = [1.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], ref_silu(1.0), EPS));
    }

    #[test]
    fn test_silu_size_4() {
        let input = [-2.0, -1.0, 0.0, 1.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_silu(*x), EPS), "silu({x}): got {o}");
        }
    }

    #[test]
    fn test_silu_size_8() {
        let input = make_input(8);
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_silu(*x), EPS));
        }
    }

    #[test]
    fn test_silu_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_silu(*x), EPS));
        }
    }

    #[test]
    fn test_silu_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_silu(*x), EPS));
        }
    }

    #[test]
    fn test_silu_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_silu(*x), EPS));
        }
    }

    #[test]
    fn test_silu_zero() {
        let input = [0.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_silu_large_positive() {
        let input = [50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 50.0, EPS), "silu(50) ≈ 50: got {o}");
        }
    }

    #[test]
    fn test_silu_large_negative() {
        let input = [-50.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, EPS), "silu(-50) ≈ 0: got {o}");
        }
    }

    // ── GELU tests ──────────────────────────────────────────────────

    #[test]
    fn test_gelu_size_1() {
        let input = [1.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], ref_gelu(1.0), EPS));
    }

    #[test]
    fn test_gelu_size_4() {
        let input = [-2.0, -1.0, 0.0, 1.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(
                approx_eq(*o, ref_gelu(*x), EPS),
                "gelu({x}): got {o}, expected {}",
                ref_gelu(*x)
            );
        }
    }

    #[test]
    fn test_gelu_size_8() {
        let input = make_input(8);
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_gelu_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_gelu_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_gelu_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_gelu_zero() {
        let input = [0.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_gelu_large_positive() {
        let input = [50.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 50.0, EPS));
        }
    }

    #[test]
    fn test_gelu_large_negative() {
        let input = [-50.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, EPS));
        }
    }

    #[test]
    fn test_gelu_near_zero() {
        let input = [-0.001, 0.001, -0.0001, 0.0001_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_gelu_accuracy() {
        // Sweep to verify max error < 1e-3
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
        let mut output = vec![0.0_f32; 200];
        unsafe { neon_gelu_f32(&input, &mut output) };
        let mut max_err: f32 = 0.0;
        for (x, o) in input.iter().zip(output.iter()) {
            let err = (*o - ref_gelu(*x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < EPS, "GELU max error {max_err} exceeds {EPS}");
    }

    // ── Fast GELU tests ─────────────────────────────────────────────

    #[test]
    fn test_fast_gelu_size_1() {
        let input = [1.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], ref_fast_gelu(1.0), EPS));
    }

    #[test]
    fn test_fast_gelu_size_4() {
        let input = [-2.0, -1.0, 0.0, 1.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_fast_gelu(*x), EPS), "fast_gelu({x}): got {o}");
        }
    }

    #[test]
    fn test_fast_gelu_size_8() {
        let input = make_input(8);
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_fast_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_fast_gelu_size_16() {
        let input = make_input(16);
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_fast_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_fast_gelu_size_100() {
        let input = make_input(100);
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_fast_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_fast_gelu_size_1000() {
        let input = make_input(1000);
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for (x, o) in input.iter().zip(output.iter()) {
            assert!(approx_eq(*o, ref_fast_gelu(*x), EPS));
        }
    }

    #[test]
    fn test_fast_gelu_zero() {
        let input = [0.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_fast_gelu_accuracy() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
        let mut output = vec![0.0_f32; 200];
        unsafe { neon_fast_gelu_f32(&input, &mut output) };
        let mut max_err: f32 = 0.0;
        for (x, o) in input.iter().zip(output.iter()) {
            let err = (*o - ref_fast_gelu(*x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < EPS, "fast GELU max error {max_err} exceeds {EPS}");
    }

    // ── SwiGLU tests ────────────────────────────────────────────────

    #[test]
    fn test_swiglu_size_1() {
        let gate = [1.0_f32];
        let up = [2.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        let expected = ref_silu(1.0) * 2.0;
        assert!(approx_eq(output[0], expected, EPS));
    }

    #[test]
    fn test_swiglu_size_4() {
        let gate = [-1.0, 0.0, 1.0, 2.0_f32];
        let up = [1.0, 2.0, 3.0, 4.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for i in 0..4 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!(
                approx_eq(output[i], expected, EPS),
                "swiglu[{i}]: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_swiglu_size_8() {
        let gate = make_input(8);
        let up: Vec<f32> = gate.iter().map(|x| x + 1.0).collect();
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for i in 0..8 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_size_16() {
        let gate = make_input(16);
        let up: Vec<f32> = gate.iter().map(|x| x * 0.5 + 1.0).collect();
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for i in 0..16 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_size_100() {
        let gate = make_input(100);
        let up: Vec<f32> = gate.iter().map(|x| x * 2.0).collect();
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for i in 0..100 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_size_1000() {
        let gate = make_input(1000);
        let up: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for i in 0..1000 {
            let expected = ref_silu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_swiglu_zero_gate() {
        let gate = [0.0_f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0_f32];
        let mut output = [1.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_swiglu_zero_up() {
        let gate = [1.0, 2.0, 3.0, 4.0_f32];
        let up = [0.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_swiglu_gate_up_interaction() {
        // When gate is large positive, silu(gate) ≈ gate, so output ≈ gate * up
        let gate = [10.0_f32; 4];
        let up = [2.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 20.0, 0.1));
        }
    }

    #[test]
    fn test_swiglu_negative_gate() {
        // When gate is large negative, silu(gate) ≈ 0, so output ≈ 0
        let gate = [-10.0_f32; 4];
        let up = [100.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, 0.1));
        }
    }

    // ── GeGLU tests ─────────────────────────────────────────────────

    #[test]
    fn test_geglu_size_1() {
        let gate = [1.0_f32];
        let up = [2.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        let expected = ref_gelu(1.0) * 2.0;
        assert!(approx_eq(output[0], expected, EPS));
    }

    #[test]
    fn test_geglu_size_4() {
        let gate = [-1.0, 0.0, 1.0, 2.0_f32];
        let up = [1.0, 2.0, 3.0, 4.0_f32];
        let mut output = [0.0_f32; 4];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for i in 0..4 {
            let expected = ref_gelu(gate[i]) * up[i];
            assert!(
                approx_eq(output[i], expected, EPS),
                "geglu[{i}]: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_geglu_size_8() {
        let gate = make_input(8);
        let up: Vec<f32> = gate.iter().map(|x| x + 1.0).collect();
        let mut output = vec![0.0_f32; 8];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for i in 0..8 {
            let expected = ref_gelu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_geglu_size_16() {
        let gate = make_input(16);
        let up: Vec<f32> = gate.iter().map(|x| x * 0.5 + 1.0).collect();
        let mut output = vec![0.0_f32; 16];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for i in 0..16 {
            let expected = ref_gelu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_geglu_size_100() {
        let gate = make_input(100);
        let up: Vec<f32> = gate.iter().map(|x| x * 2.0).collect();
        let mut output = vec![0.0_f32; 100];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for i in 0..100 {
            let expected = ref_gelu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_geglu_size_1000() {
        let gate = make_input(1000);
        let up: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0_f32; 1000];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for i in 0..1000 {
            let expected = ref_gelu(gate[i]) * up[i];
            assert!(approx_eq(output[i], expected, EPS));
        }
    }

    #[test]
    fn test_geglu_zero_gate() {
        let gate = [0.0_f32; 4];
        let up = [1.0, 2.0, 3.0, 4.0_f32];
        let mut output = [1.0_f32; 4];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_geglu_zero_up() {
        let gate = [1.0, 2.0, 3.0, 4.0_f32];
        let up = [0.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for o in &output {
            assert!(approx_eq(*o, 0.0, STRICT_EPS));
        }
    }

    #[test]
    fn test_geglu_gate_up_interaction() {
        let gate = [10.0_f32; 4];
        let up = [2.0_f32; 4];
        let mut output = [0.0_f32; 4];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for o in &output {
            // gelu(10) ≈ 10, so output ≈ 20
            assert!(approx_eq(*o, 20.0, 0.1));
        }
    }

    #[test]
    fn test_geglu_negative_gate() {
        let gate = [-10.0_f32; 4];
        let up = [100.0_f32; 4];
        let mut output = [1.0_f32; 4];
        unsafe { neon_geglu_f32(&gate, &up, &mut output) };
        for o in &output {
            // gelu(-10) ≈ 0, so output ≈ 0
            assert!(approx_eq(*o, 0.0, 0.1));
        }
    }

    // ── Sigmoid accuracy ────────────────────────────────────────────

    #[test]
    fn test_sigmoid_accuracy() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.1).collect();
        let mut output = vec![0.0_f32; 200];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        let mut max_err: f32 = 0.0;
        for (x, o) in input.iter().zip(output.iter()) {
            let err = (*o - ref_sigmoid(*x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < EPS, "sigmoid max error {max_err} exceeds {EPS}");
    }

    // ── Tanh accuracy ───────────────────────────────────────────────

    #[test]
    fn test_tanh_accuracy() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
        let mut output = vec![0.0_f32; 200];
        unsafe { neon_tanh_f32(&input, &mut output) };
        let mut max_err: f32 = 0.0;
        for (x, o) in input.iter().zip(output.iter()) {
            let err = (*o - ref_tanh(*x)).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < EPS, "tanh max error {max_err} exceeds {EPS}");
    }

    // ── Empty input ─────────────────────────────────────────────────

    #[test]
    fn test_all_empty() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        unsafe {
            neon_relu_f32(&input, &mut output);
            neon_sigmoid_f32(&input, &mut output);
            neon_tanh_f32(&input, &mut output);
            neon_silu_f32(&input, &mut output);
            neon_gelu_f32(&input, &mut output);
            neon_fast_gelu_f32(&input, &mut output);
        }
    }

    #[test]
    fn test_swiglu_empty() {
        let g: [f32; 0] = [];
        let u: [f32; 0] = [];
        let mut o: [f32; 0] = [];
        unsafe { neon_swiglu_f32(&g, &u, &mut o) };
    }

    #[test]
    fn test_geglu_empty() {
        let g: [f32; 0] = [];
        let u: [f32; 0] = [];
        let mut o: [f32; 0] = [];
        unsafe { neon_geglu_f32(&g, &u, &mut o) };
    }

    // ── Odd sizes (remainder path) ──────────────────────────────────

    #[test]
    fn test_relu_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let input = make_input(n);
            let mut output = vec![0.0_f32; n];
            unsafe { neon_relu_f32(&input, &mut output) };
            for (x, o) in input.iter().zip(output.iter()) {
                assert_eq!(*o, if *x > 0.0 { *x } else { 0.0 });
            }
        }
    }

    #[test]
    fn test_sigmoid_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let input = make_input(n);
            let mut output = vec![0.0_f32; n];
            unsafe { neon_sigmoid_f32(&input, &mut output) };
            for (x, o) in input.iter().zip(output.iter()) {
                assert!(approx_eq(*o, ref_sigmoid(*x), EPS));
            }
        }
    }

    #[test]
    fn test_gelu_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let input = make_input(n);
            let mut output = vec![0.0_f32; n];
            unsafe { neon_gelu_f32(&input, &mut output) };
            for (x, o) in input.iter().zip(output.iter()) {
                assert!(approx_eq(*o, ref_gelu(*x), EPS));
            }
        }
    }

    #[test]
    fn test_silu_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let input = make_input(n);
            let mut output = vec![0.0_f32; n];
            unsafe { neon_silu_f32(&input, &mut output) };
            for (x, o) in input.iter().zip(output.iter()) {
                assert!(approx_eq(*o, ref_silu(*x), EPS));
            }
        }
    }

    #[test]
    fn test_tanh_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let input = make_input(n);
            let mut output = vec![0.0_f32; n];
            unsafe { neon_tanh_f32(&input, &mut output) };
            for (x, o) in input.iter().zip(output.iter()) {
                assert!(approx_eq(*o, ref_tanh(*x), EPS));
            }
        }
    }

    #[test]
    fn test_swiglu_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let gate = make_input(n);
            let up: Vec<f32> = gate.iter().map(|x| x + 1.0).collect();
            let mut output = vec![0.0_f32; n];
            unsafe { neon_swiglu_f32(&gate, &up, &mut output) };
            for i in 0..n {
                let expected = ref_silu(gate[i]) * up[i];
                assert!(approx_eq(output[i], expected, EPS));
            }
        }
    }

    #[test]
    fn test_geglu_odd_sizes() {
        for n in [1, 2, 3, 5, 7, 9, 13, 15, 17] {
            let gate = make_input(n);
            let up: Vec<f32> = gate.iter().map(|x| x + 1.0).collect();
            let mut output = vec![0.0_f32; n];
            unsafe { neon_geglu_f32(&gate, &up, &mut output) };
            for i in 0..n {
                let expected = ref_gelu(gate[i]) * up[i];
                assert!(approx_eq(output[i], expected, EPS));
            }
        }
    }

    // ── Scalar exp fast test ────────────────────────────────────────

    #[test]
    fn test_scalar_exp_fast_accuracy() {
        for i in 0..200 {
            let x = (i as f32 - 100.0) * 0.1;
            let fast = scalar_exp_fast(x);
            let std_exp = x.exp();
            let rel_err = if std_exp.abs() > 1e-10 {
                (fast - std_exp).abs() / std_exp.abs()
            } else {
                (fast - std_exp).abs()
            };
            assert!(rel_err < 1e-3, "exp({x}): fast={fast}, std={std_exp}, rel_err={rel_err}");
        }
    }

    #[test]
    fn test_scalar_exp_fast_clamp() {
        let large_pos = scalar_exp_fast(100.0);
        assert!(large_pos.is_finite());
        let large_neg = scalar_exp_fast(-100.0);
        assert!(large_neg >= 0.0);
        assert!(large_neg.is_finite());
    }
}
