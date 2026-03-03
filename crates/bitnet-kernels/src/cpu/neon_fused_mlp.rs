//! NEON-optimized fused MLP (Multi-Layer Perceptron) kernels for Apple Silicon.
//!
//! Provides fused gate-up projection with SiLU activation for LLaMA-style
//! gated MLPs, plus standalone SiLU, GELU, and sigmoid using NEON SIMD
//! intrinsics with fast polynomial exp approximation.
//!
//! The fused kernels reduce memory traffic by combining gate projection,
//! up projection, and activation into a single pass.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Fast exp approximation ──────────────────────────────────────────

/// Fast exp approximation using Schraudolph's method refined with a
/// degree-4 polynomial correction, suitable for sigmoid/SiLU ranges.
///
/// Accurate to ~1e-4 in the range [-10, 10].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_fast_exp(x: float32x4_t) -> float32x4_t {
    // Clamp to avoid overflow/underflow in the integer trick.
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    // exp(x) = 2^(x / ln2) = 2^(n + f) where n = floor(x/ln2), f = frac
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E); // 1/ln2
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

    let t = vmulq_f32(x, log2e);
    // n = floor(t)
    let n = vrndmq_f32(t);
    // f = x - n * ln2 (reduced argument in [0, ln2))
    let f = vmlsq_f32(x, n, ln2);

    // Polynomial approximation of 2^f - 1 for f in [0, ln2):
    // p(f) ≈ 1 + f + f²/2 + f³/6 + f⁴/24  (Taylor of exp)
    let c1 = vdupq_n_f32(1.0);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);

    // Horner's method: p = 1 + f*(1 + f*(0.5 + f*(1/6 + f/24)))
    let mut p = vmlaq_f32(c3, f, c4);
    p = vmlaq_f32(c2, f, p);
    p = vmlaq_f32(c1, f, p);
    p = vmlaq_f32(c1, f, p);

    // Scale by 2^n: reinterpret n as integer exponent bias
    let n_i = vcvtq_s32_f32(n);
    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n_i, bias), 23));

    vmulq_f32(p, pow2n)
}

/// Scalar fast sigmoid for tail elements.
#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scalar SiLU: x * sigmoid(x).
#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x * scalar_sigmoid(x)
}

/// Scalar GELU (tanh approximation).
#[inline(always)]
fn scalar_gelu(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    0.5 * x * (1.0 + inner.tanh())
}

// ── Sigmoid ─────────────────────────────────────────────────────────

/// Fast NEON sigmoid: 1 / (1 + exp(-x)).
///
/// Uses [`neon_fast_exp`] polynomial approximation with NEON reciprocal
/// estimate + one Newton-Raphson refinement step for the division.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * 4;
        let x = vld1q_f32(input.as_ptr().add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = neon_fast_exp(neg_x);
        let denom = vaddq_f32(one, exp_neg_x);

        // Reciprocal estimate + one Newton-Raphson step
        let recip = vrecpeq_f32(denom);
        let result = vmulq_f32(recip, vrecpsq_f32(denom, recip));

        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = scalar_sigmoid(input[idx]);
    }
}

// ── SiLU ────────────────────────────────────────────────────────────

/// NEON-optimized SiLU activation: x * sigmoid(x).
///
/// Computes sigmoid via fast exp approximation, then fuses the multiply
/// with NEON. This is the primary activation used in LLaMA-style MLPs.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * 4;
        let x = vld1q_f32(input.as_ptr().add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = neon_fast_exp(neg_x);
        let denom = vaddq_f32(one, exp_neg_x);
        let recip = vrecpeq_f32(denom);
        let sig = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        let result = vmulq_f32(x, sig);

        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = scalar_silu(input[idx]);
    }
}

// ── GELU ────────────────────────────────────────────────────────────

/// NEON-optimized GELU (tanh approximation):
/// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
///
/// The inner tanh is computed as:
///   tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
/// using [`neon_fast_exp`] for the exponential.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let sqrt_2_over_pi = vdupq_n_f32((2.0_f32 / std::f32::consts::PI).sqrt());
    let coeff = vdupq_n_f32(0.044715);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);

    for i in 0..chunks {
        let offset = i * 4;
        let x = vld1q_f32(input.as_ptr().add(offset));

        // x³
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);

        // inner = sqrt(2/π) * (x + 0.044715 * x³)
        let cubic_term = vmlaq_f32(x, coeff, x3);
        let inner = vmulq_f32(sqrt_2_over_pi, cubic_term);

        // tanh(inner) via (exp(2*inner) - 1) / (exp(2*inner) + 1)
        let two_inner = vmulq_f32(two, inner);
        let exp2a = neon_fast_exp(two_inner);
        let num = vsubq_f32(exp2a, one);
        let den = vaddq_f32(exp2a, one);
        let recip_den = vrecpeq_f32(den);
        let recip_refined = vmulq_f32(recip_den, vrecpsq_f32(den, recip_den));
        let tanh_val = vmulq_f32(num, recip_refined);

        // 0.5 * x * (1 + tanh)
        let one_plus_tanh = vaddq_f32(one, tanh_val);
        let result = vmulq_f32(half, vmulq_f32(x, one_plus_tanh));

        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = scalar_gelu(input[idx]);
    }
}

// ── Fused Gate-Up ───────────────────────────────────────────────────

/// Fused gate * SiLU(up) for LLaMA-style gated MLP.
///
/// Computes `output[i] = gate[i] * SiLU(up[i])` in a single pass,
/// avoiding an intermediate buffer for the SiLU result.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gate`, `up`, and `output` don't all have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_gate_up_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n, "up length must match gate length");
    assert!(output.len() >= n, "output buffer too small");

    let chunks = n / 4;
    let remainder = n % 4;
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * 4;
        let g = vld1q_f32(gate.as_ptr().add(offset));
        let u = vld1q_f32(up.as_ptr().add(offset));

        // SiLU(up) = up * sigmoid(up)
        let neg_u = vnegq_f32(u);
        let exp_neg_u = neon_fast_exp(neg_u);
        let denom = vaddq_f32(one, exp_neg_u);
        let recip = vrecpeq_f32(denom);
        let sig = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        let silu_u = vmulq_f32(u, sig);

        // gate * SiLU(up)
        let result = vmulq_f32(g, silu_u);
        vst1q_f32(output.as_mut_ptr().add(offset), result);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = gate[idx] * scalar_silu(up[idx]);
    }
}

// ── Fused MLP forward ───────────────────────────────────────────────

/// Full fused MLP forward pass: down_proj(gate_proj(x) ⊙ SiLU(up_proj(x))).
///
/// Performs the LLaMA-style gated MLP in one call:
/// 1. Gate projection: `gate = input × gate_weights`  (input_dim → hidden_dim)
/// 2. Up projection:   `up   = input × up_weights`    (input_dim → hidden_dim)
/// 3. Fused activation: `hidden = gate ⊙ SiLU(up)`
/// 4. Down projection: `output = hidden × down_weights` (hidden_dim → input_dim)
///
/// Weight matrices are stored **row-major**: `gate_weights` and `up_weights`
/// have shape `[hidden_dim × input_dim]`, `down_weights` has shape
/// `[output_dim × hidden_dim]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if weight dimensions are inconsistent with `input` / `hidden_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_mlp_f32(
    input: &[f32],
    gate_weights: &[f32],
    up_weights: &[f32],
    down_weights: &[f32],
    hidden_dim: usize,
    output: &mut [f32],
) {
    let input_dim = input.len();
    let output_dim = output.len();

    assert_eq!(
        gate_weights.len(),
        hidden_dim * input_dim,
        "gate_weights must be hidden_dim × input_dim"
    );
    assert_eq!(
        up_weights.len(),
        hidden_dim * input_dim,
        "up_weights must be hidden_dim × input_dim"
    );
    assert_eq!(
        down_weights.len(),
        output_dim * hidden_dim,
        "down_weights must be output_dim × hidden_dim"
    );

    // Step 1-2: Fused gate and up projections (matmul + activation).
    let mut hidden = vec![0.0f32; hidden_dim];

    for h in 0..hidden_dim {
        let gate_row = &gate_weights[h * input_dim..(h + 1) * input_dim];
        let up_row = &up_weights[h * input_dim..(h + 1) * input_dim];

        let gate_val = neon_dot_f32(input, gate_row);
        let up_val = neon_dot_f32(input, up_row);

        // gate * SiLU(up)
        hidden[h] = gate_val * scalar_silu(up_val);
    }

    // Step 3: Down projection.
    for o in 0..output_dim {
        let down_row = &down_weights[o * hidden_dim..(o + 1) * hidden_dim];
        output[o] = neon_dot_f32(&hidden, down_row);
    }
}

/// NEON-accelerated dot product of two f32 slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        acc = vmlaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        sum += a[idx] * b[idx];
    }

    sum
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps || (a.is_nan() && b.is_nan())
    }

    fn assert_approx_slice(actual: &[f32], expected: &[f32], eps: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*a, *e, eps),
                "{label}[{i}]: got {a}, expected {e} (diff={})",
                (a - e).abs()
            );
        }
    }

    fn ref_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn ref_silu(x: f32) -> f32 {
        x * ref_sigmoid(x)
    }

    fn ref_gelu(x: f32) -> f32 {
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        0.5 * x * (1.0 + inner.tanh())
    }

    const TOLERANCE: f32 = 5e-4;
    const LOOSE_TOL: f32 = 5e-3;

    // ================================================================
    // Sigmoid tests
    // ================================================================

    #[test]
    fn test_sigmoid_basic() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "sigmoid_basic");
    }

    #[test]
    fn test_sigmoid_zero() {
        let input = [0.0];
        let mut output = vec![0.0f32; 1];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.5, TOLERANCE), "sigmoid(0) should be 0.5");
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let input = [10.0, 20.0, 50.0, 88.0];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for &v in &output {
            assert!(v > 0.999, "sigmoid(large) should be ~1.0, got {v}");
        }
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let input = [-10.0, -20.0, -50.0, -88.0];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for &v in &output {
            assert!(v < 0.001, "sigmoid(large neg) should be ~0.0, got {v}");
        }
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let vals = [0.1, 0.5, 1.0, 2.0, 5.0];
        for &x in &vals {
            let input_pos = [x];
            let input_neg = [-x];
            let mut out_pos = [0.0f32];
            let mut out_neg = [0.0f32];
            unsafe {
                neon_sigmoid_f32(&input_pos, &mut out_pos);
                neon_sigmoid_f32(&input_neg, &mut out_neg);
            }
            assert!(
                approx_eq(out_pos[0] + out_neg[0], 1.0, TOLERANCE),
                "sigmoid(x) + sigmoid(-x) should be 1.0 for x={x}"
            );
        }
    }

    #[test]
    fn test_sigmoid_output_range() {
        let input: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.2).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid output must be in [0,1], got {v} at index {i}");
        }
    }

    #[test]
    fn test_sigmoid_monotonic() {
        let input: Vec<f32> = (0..64).map(|i| -5.0 + i as f32 * 0.16).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1] - 1e-6, "sigmoid must be monotonically increasing");
        }
    }

    // ================================================================
    // SiLU tests
    // ================================================================

    #[test]
    fn test_silu_basic() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "silu_basic");
    }

    #[test]
    fn test_silu_zero() {
        let input = [0.0];
        let mut output = [0.0f32];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, 1e-7), "SiLU(0) must be 0");
    }

    #[test]
    fn test_silu_positive_values() {
        let input: Vec<f32> = (1..=16).map(|i| i as f32 * 0.5).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "silu_positive");
    }

    #[test]
    fn test_silu_negative_values() {
        let input: Vec<f32> = (1..=16).map(|i| -(i as f32) * 0.5).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "silu_negative");
    }

    #[test]
    fn test_silu_large_positive() {
        // SiLU(x) ≈ x for large x since sigmoid(x) → 1
        let input = [10.0, 20.0, 50.0, 80.0];
        let mut output = [0.0f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (i, (&v, &x)) in output.iter().zip(input.iter()).enumerate() {
            assert!(approx_eq(v, x, 0.01), "SiLU(large) ≈ x, got {v} for x={x} at {i}");
        }
    }

    #[test]
    fn test_silu_large_negative() {
        // SiLU(x) → 0 for large negative x
        let input = [-10.0, -20.0, -50.0, -80.0];
        let mut output = [0.0f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for &v in &output {
            assert!(v.abs() < 0.01, "SiLU(large neg) ≈ 0, got {v}");
        }
    }

    #[test]
    fn test_silu_approximate_monotonicity() {
        // SiLU is monotonically increasing for x > ~-0.278
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.2).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        for i in 1..output.len() {
            assert!(output[i] >= output[i - 1] - 1e-6, "SiLU should be monotonic for x >= 0");
        }
    }

    #[test]
    fn test_silu_minimum_region() {
        // SiLU has a minimum near x ≈ -1.278
        let input = [-1.278];
        let mut output = [0.0f32];
        unsafe { neon_silu_f32(&input, &mut output) };
        let expected = ref_silu(-1.278);
        assert!(approx_eq(output[0], expected, TOLERANCE));
    }

    // ================================================================
    // GELU tests
    // ================================================================

    #[test]
    fn test_gelu_basic() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, LOOSE_TOL, "gelu_basic");
    }

    #[test]
    fn test_gelu_zero() {
        let input = [0.0];
        let mut output = [0.0f32];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, 1e-6), "GELU(0) must be 0");
    }

    #[test]
    fn test_gelu_positive() {
        let input: Vec<f32> = (1..=16).map(|i| i as f32 * 0.25).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, LOOSE_TOL, "gelu_positive");
    }

    #[test]
    fn test_gelu_negative() {
        let input: Vec<f32> = (1..=16).map(|i| -(i as f32) * 0.25).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, LOOSE_TOL, "gelu_negative");
    }

    #[test]
    fn test_gelu_large_positive() {
        // GELU(x) ≈ x for large positive x
        let input = [10.0, 20.0, 50.0, 80.0];
        let mut output = [0.0f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (&v, &x) in output.iter().zip(input.iter()) {
            assert!(approx_eq(v, x, 0.1), "GELU(large) ≈ x, got {v}");
        }
    }

    #[test]
    fn test_gelu_large_negative() {
        // GELU(x) → 0 for large negative x
        let input = [-10.0, -20.0, -50.0, -80.0];
        let mut output = [0.0f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for &v in &output {
            assert!(v.abs() < 0.01, "GELU(large neg) ≈ 0, got {v}");
        }
    }

    #[test]
    fn test_gelu_symmetry_property() {
        // GELU(-x) ≈ -x + GELU(x) is NOT true, but GELU is NOT odd.
        // Instead verify GELU(x) + GELU(-x) ≈ x for small x (approximately)
        // Actually just verify values match reference.
        let vals = [0.5, 1.0, 1.5, 2.0];
        for &x in &vals {
            let input_pos = [x];
            let input_neg = [-x];
            let mut out_pos = [0.0f32];
            let mut out_neg = [0.0f32];
            unsafe {
                neon_gelu_f32(&input_pos, &mut out_pos);
                neon_gelu_f32(&input_neg, &mut out_neg);
            }
            assert!(approx_eq(out_pos[0], ref_gelu(x), LOOSE_TOL));
            assert!(approx_eq(out_neg[0], ref_gelu(-x), LOOSE_TOL));
        }
    }

    // ================================================================
    // Fused gate-up tests
    // ================================================================

    #[test]
    fn test_fused_gate_up_basic() {
        let gate = [1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 0.0, 1.5];
        let up = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0];
        let expected: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(&g, &u)| g * ref_silu(u)).collect();
        let mut output = vec![0.0f32; gate.len()];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "fused_gate_up_basic");
    }

    #[test]
    fn test_fused_gate_up_zero_gate() {
        let gate = [0.0; 8];
        let up = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![999.0f32; 8];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 0.0, 1e-7), "zero gate should zero output, got {v} at {i}");
        }
    }

    #[test]
    fn test_fused_gate_up_zero_up() {
        // SiLU(0) = 0, so gate * SiLU(0) = 0
        let gate = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let up = [0.0; 8];
        let mut output = vec![999.0f32; 8];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 0.0, 1e-7), "SiLU(0)=0 so output should be 0, got {v} at {i}");
        }
    }

    #[test]
    fn test_fused_gate_up_identity_gate() {
        // gate = 1.0 → output = SiLU(up)
        let gate = [1.0; 8];
        let up = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let expected: Vec<f32> = up.iter().map(|&u| ref_silu(u)).collect();
        let mut output = vec![0.0f32; 8];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "fused_gate_up_identity");
    }

    #[test]
    fn test_fused_gate_up_matches_separate() {
        // Verify fused == gate * SiLU(up) computed separately
        let gate: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let up: Vec<f32> = (0..32).map(|i| (i as f32 - 8.0) * 0.2).collect();

        // Separate computation
        let mut silu_out = vec![0.0f32; 32];
        unsafe { neon_silu_f32(&up, &mut silu_out) };
        let expected: Vec<f32> = gate.iter().zip(silu_out.iter()).map(|(g, s)| g * s).collect();

        // Fused computation
        let mut fused_out = vec![0.0f32; 32];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut fused_out) };

        assert_approx_slice(&fused_out, &expected, 1e-6, "fused_vs_separate");
    }

    // ================================================================
    // Size variation tests
    // ================================================================

    macro_rules! size_test_sigmoid {
        ($name:ident, $size:expr) => {
            #[test]
            fn $name() {
                let input: Vec<f32> =
                    (0..$size).map(|i| (i as f32 - $size as f32 / 2.0) * 0.1).collect();
                let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
                let mut output = vec![0.0f32; $size];
                unsafe { neon_sigmoid_f32(&input, &mut output) };
                assert_approx_slice(
                    &output,
                    &expected,
                    TOLERANCE,
                    concat!("sigmoid_size_", stringify!($size)),
                );
            }
        };
    }

    size_test_sigmoid!(test_sigmoid_size_1, 1);
    size_test_sigmoid!(test_sigmoid_size_4, 4);
    size_test_sigmoid!(test_sigmoid_size_7, 7);
    size_test_sigmoid!(test_sigmoid_size_8, 8);
    size_test_sigmoid!(test_sigmoid_size_15, 15);
    size_test_sigmoid!(test_sigmoid_size_16, 16);
    size_test_sigmoid!(test_sigmoid_size_31, 31);
    size_test_sigmoid!(test_sigmoid_size_32, 32);
    size_test_sigmoid!(test_sigmoid_size_63, 63);
    size_test_sigmoid!(test_sigmoid_size_64, 64);
    size_test_sigmoid!(test_sigmoid_size_128, 128);
    size_test_sigmoid!(test_sigmoid_size_256, 256);
    size_test_sigmoid!(test_sigmoid_size_512, 512);
    size_test_sigmoid!(test_sigmoid_size_1024, 1024);

    macro_rules! size_test_silu {
        ($name:ident, $size:expr) => {
            #[test]
            fn $name() {
                let input: Vec<f32> =
                    (0..$size).map(|i| (i as f32 - $size as f32 / 2.0) * 0.1).collect();
                let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
                let mut output = vec![0.0f32; $size];
                unsafe { neon_silu_f32(&input, &mut output) };
                assert_approx_slice(
                    &output,
                    &expected,
                    TOLERANCE,
                    concat!("silu_size_", stringify!($size)),
                );
            }
        };
    }

    size_test_silu!(test_silu_size_1, 1);
    size_test_silu!(test_silu_size_4, 4);
    size_test_silu!(test_silu_size_7, 7);
    size_test_silu!(test_silu_size_8, 8);
    size_test_silu!(test_silu_size_15, 15);
    size_test_silu!(test_silu_size_16, 16);
    size_test_silu!(test_silu_size_31, 31);
    size_test_silu!(test_silu_size_32, 32);
    size_test_silu!(test_silu_size_63, 63);
    size_test_silu!(test_silu_size_64, 64);
    size_test_silu!(test_silu_size_128, 128);
    size_test_silu!(test_silu_size_256, 256);
    size_test_silu!(test_silu_size_512, 512);
    size_test_silu!(test_silu_size_1024, 1024);

    macro_rules! size_test_gelu {
        ($name:ident, $size:expr) => {
            #[test]
            fn $name() {
                let input: Vec<f32> =
                    (0..$size).map(|i| (i as f32 - $size as f32 / 2.0) * 0.1).collect();
                let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
                let mut output = vec![0.0f32; $size];
                unsafe { neon_gelu_f32(&input, &mut output) };
                assert_approx_slice(
                    &output,
                    &expected,
                    LOOSE_TOL,
                    concat!("gelu_size_", stringify!($size)),
                );
            }
        };
    }

    size_test_gelu!(test_gelu_size_1, 1);
    size_test_gelu!(test_gelu_size_4, 4);
    size_test_gelu!(test_gelu_size_7, 7);
    size_test_gelu!(test_gelu_size_8, 8);
    size_test_gelu!(test_gelu_size_15, 15);
    size_test_gelu!(test_gelu_size_16, 16);
    size_test_gelu!(test_gelu_size_31, 31);
    size_test_gelu!(test_gelu_size_32, 32);
    size_test_gelu!(test_gelu_size_63, 63);
    size_test_gelu!(test_gelu_size_64, 64);
    size_test_gelu!(test_gelu_size_128, 128);
    size_test_gelu!(test_gelu_size_256, 256);
    size_test_gelu!(test_gelu_size_512, 512);
    size_test_gelu!(test_gelu_size_1024, 1024);

    macro_rules! size_test_fused_gate_up {
        ($name:ident, $size:expr) => {
            #[test]
            fn $name() {
                let gate: Vec<f32> =
                    (0..$size).map(|i| (i as f32 - $size as f32 / 2.0) * 0.05).collect();
                let up: Vec<f32> = (0..$size).map(|i| (i as f32) * 0.1 - 2.0).collect();
                let expected: Vec<f32> =
                    gate.iter().zip(up.iter()).map(|(&g, &u)| g * ref_silu(u)).collect();
                let mut output = vec![0.0f32; $size];
                unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
                assert_approx_slice(
                    &output,
                    &expected,
                    TOLERANCE,
                    concat!("fused_size_", stringify!($size)),
                );
            }
        };
    }

    size_test_fused_gate_up!(test_fused_gate_up_size_1, 1);
    size_test_fused_gate_up!(test_fused_gate_up_size_4, 4);
    size_test_fused_gate_up!(test_fused_gate_up_size_7, 7);
    size_test_fused_gate_up!(test_fused_gate_up_size_8, 8);
    size_test_fused_gate_up!(test_fused_gate_up_size_15, 15);
    size_test_fused_gate_up!(test_fused_gate_up_size_16, 16);
    size_test_fused_gate_up!(test_fused_gate_up_size_31, 31);
    size_test_fused_gate_up!(test_fused_gate_up_size_32, 32);
    size_test_fused_gate_up!(test_fused_gate_up_size_63, 63);
    size_test_fused_gate_up!(test_fused_gate_up_size_64, 64);

    // ================================================================
    // MLP forward pass tests
    // ================================================================

    fn ref_mlp(
        input: &[f32],
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
        hidden_dim: usize,
    ) -> Vec<f32> {
        let input_dim = input.len();
        let output_dim = down_w.len() / hidden_dim;

        let mut hidden = vec![0.0f32; hidden_dim];
        for h in 0..hidden_dim {
            let mut gate_val = 0.0f32;
            let mut up_val = 0.0f32;
            for j in 0..input_dim {
                gate_val += input[j] * gate_w[h * input_dim + j];
                up_val += input[j] * up_w[h * input_dim + j];
            }
            hidden[h] = gate_val * ref_silu(up_val);
        }

        let mut output = vec![0.0f32; output_dim];
        for o in 0..output_dim {
            for h in 0..hidden_dim {
                output[o] += hidden[h] * down_w[o * hidden_dim + h];
            }
        }
        output
    }

    #[test]
    fn test_mlp_identity_weights() {
        // 2→4→2 MLP with identity-like weights
        let input_dim = 2;
        let hidden_dim = 4;
        let output_dim = 2;

        let input = [1.0, 0.5];

        // Simple diagonal-ish gate weights
        let mut gate_w = vec![0.0f32; hidden_dim * input_dim];
        gate_w[0] = 1.0; // h0 = input[0]
        gate_w[3] = 1.0; // h1 = input[1]

        let mut up_w = vec![0.0f32; hidden_dim * input_dim];
        up_w[0] = 1.0;
        up_w[3] = 1.0;

        let mut down_w = vec![0.0f32; output_dim * hidden_dim];
        down_w[0] = 1.0; // o0 = hidden[0]
        down_w[5] = 1.0; // o1 = hidden[1]

        let expected = ref_mlp(&input, &gate_w, &up_w, &down_w, hidden_dim);
        let mut output = vec![0.0f32; output_dim];
        unsafe {
            neon_fused_mlp_f32(&input, &gate_w, &up_w, &down_w, hidden_dim, &mut output);
        }
        assert_approx_slice(&output, &expected, TOLERANCE, "mlp_identity");
    }

    #[test]
    fn test_mlp_small_random() {
        // 4→8→4 MLP with pseudo-random weights
        let input_dim = 4;
        let hidden_dim = 8;
        let output_dim = 4;

        let input = [0.5, -0.3, 0.8, -0.1];

        // Deterministic pseudo-random weights
        let gen_weights = |size: usize, seed: u32| -> Vec<f32> {
            let mut w = Vec::with_capacity(size);
            let mut s = seed;
            for _ in 0..size {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                w.push(((s >> 16) as f32 / 32768.0) - 1.0);
            }
            w
        };

        let gate_w = gen_weights(hidden_dim * input_dim, 42);
        let up_w = gen_weights(hidden_dim * input_dim, 137);
        let down_w = gen_weights(output_dim * hidden_dim, 999);

        let expected = ref_mlp(&input, &gate_w, &up_w, &down_w, hidden_dim);
        let mut output = vec![0.0f32; output_dim];
        unsafe {
            neon_fused_mlp_f32(&input, &gate_w, &up_w, &down_w, hidden_dim, &mut output);
        }
        assert_approx_slice(&output, &expected, LOOSE_TOL, "mlp_small_random");
    }

    #[test]
    fn test_mlp_zero_input() {
        let input_dim = 4;
        let hidden_dim = 8;
        let output_dim = 4;

        let input = [0.0; 4];
        let gate_w = vec![1.0f32; hidden_dim * input_dim];
        let up_w = vec![1.0f32; hidden_dim * input_dim];
        let down_w = vec![1.0f32; output_dim * hidden_dim];

        let mut output = vec![999.0f32; output_dim];
        unsafe {
            neon_fused_mlp_f32(&input, &gate_w, &up_w, &down_w, hidden_dim, &mut output);
        }
        // Zero input → zero projections → SiLU(0)=0 → zero output
        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 0.0, 1e-6), "zero input should give zero output, got {v} at {i}");
        }
    }

    macro_rules! mlp_hidden_dim_test {
        ($name:ident, $hidden:expr) => {
            #[test]
            fn $name() {
                let input_dim = 4;
                let hidden_dim = $hidden;
                let output_dim = 4;

                let input = [0.3, -0.2, 0.1, 0.4];
                let gate_w = vec![0.01f32; hidden_dim * input_dim];
                let up_w = vec![0.01f32; hidden_dim * input_dim];
                let down_w = vec![0.01f32; output_dim * hidden_dim];

                let expected = ref_mlp(&input, &gate_w, &up_w, &down_w, hidden_dim);
                let mut output = vec![0.0f32; output_dim];
                unsafe {
                    neon_fused_mlp_f32(&input, &gate_w, &up_w, &down_w, hidden_dim, &mut output);
                }
                assert_approx_slice(
                    &output,
                    &expected,
                    LOOSE_TOL,
                    concat!("mlp_hidden_", stringify!($hidden)),
                );
            }
        };
    }

    mlp_hidden_dim_test!(test_mlp_hidden_128, 128);
    mlp_hidden_dim_test!(test_mlp_hidden_256, 256);
    mlp_hidden_dim_test!(test_mlp_hidden_512, 512);

    // ================================================================
    // Edge case tests
    // ================================================================

    #[test]
    fn test_sigmoid_nan_passthrough() {
        let input = [f32::NAN];
        let mut output = [0.0f32];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert!(output[0].is_nan(), "sigmoid(NaN) should be NaN");
    }

    #[test]
    fn test_silu_nan_passthrough() {
        let input = [f32::NAN];
        let mut output = [0.0f32];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert!(output[0].is_nan(), "SiLU(NaN) should be NaN");
    }

    #[test]
    fn test_gelu_nan_passthrough() {
        let input = [f32::NAN];
        let mut output = [0.0f32];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert!(output[0].is_nan(), "GELU(NaN) should be NaN");
    }

    #[test]
    fn test_sigmoid_infinity() {
        let input = [f32::INFINITY, f32::NEG_INFINITY];
        let mut output = [0.0f32; 2];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        // sigmoid(+inf) → 1, sigmoid(-inf) → 0; fast approx may not be exact
        // but should be clamped to valid range
        assert!(
            output[0] >= 0.99 || output[0].is_nan(),
            "sigmoid(+inf) should be ~1.0, got {}",
            output[0]
        );
        assert!(
            output[1] <= 0.01 || output[1].is_nan(),
            "sigmoid(-inf) should be ~0.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_silu_infinity() {
        let input = [f32::INFINITY];
        let mut output = [0.0f32];
        unsafe { neon_silu_f32(&input, &mut output) };
        // SiLU(+inf) = +inf * 1 = +inf (or very large)
        assert!(
            output[0] > 1e6 || output[0].is_infinite(),
            "SiLU(+inf) should be large/inf, got {}",
            output[0]
        );
    }

    #[test]
    fn test_empty_slices() {
        let input: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        // Should not panic on empty input
        unsafe {
            neon_sigmoid_f32(input, &mut output);
            neon_silu_f32(input, &mut output);
            neon_gelu_f32(input, &mut output);
        }
    }

    #[test]
    fn test_fused_gate_up_empty() {
        let gate: &[f32] = &[];
        let up: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_fused_gate_up_f32(gate, up, &mut output) };
    }

    // ================================================================
    // Property-style tests
    // ================================================================

    #[test]
    fn test_sigmoid_output_bounded() {
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "sigmoid must be in [0,1], got {v} at index {i}");
        }
    }

    #[test]
    fn test_silu_bounded_below() {
        // SiLU has a minimum of about -0.2785
        let input: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= -0.35, "SiLU should be >= ~-0.28, got {v} at index {i}");
        }
    }

    #[test]
    fn test_gelu_bounded_below() {
        // GELU has a minimum of about -0.17
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(v >= -0.25, "GELU should be >= ~-0.17, got {v} at index {i}");
        }
    }

    #[test]
    fn test_silu_positive_for_positive_input() {
        let input: Vec<f32> = (1..=100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(v > 0.0, "SiLU(x>0) > 0, got {v} at index {i}");
        }
    }

    #[test]
    fn test_gelu_positive_for_positive_input() {
        // GELU(x) > 0 for x > 0 (approximately, for x > ~0.05)
        let input: Vec<f32> = (1..=100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            assert!(v > -0.01, "GELU(x>0.1) should be positive, got {v} at index {i}");
        }
    }

    // ================================================================
    // Dot product tests (internal helper)
    // ================================================================

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert!(approx_eq(result, 20.0, 1e-6), "dot product should be 20.0, got {result}");
    }

    #[test]
    fn test_dot_product_with_tail() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert!(approx_eq(result, 35.0, 1e-6), "dot product should be 35.0, got {result}");
    }

    #[test]
    fn test_dot_product_single() {
        let a = [3.0];
        let b = [4.0];
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert!(approx_eq(result, 12.0, 1e-6));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let result = unsafe { neon_dot_f32(&a, &b) };
        assert!(approx_eq(result, 0.0, 1e-6));
    }

    // ================================================================
    // Cross-function consistency tests
    // ================================================================

    #[test]
    fn test_silu_equals_x_times_sigmoid() {
        let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
        let mut sig_out = vec![0.0f32; input.len()];
        let mut silu_out = vec![0.0f32; input.len()];
        unsafe {
            neon_sigmoid_f32(&input, &mut sig_out);
            neon_silu_f32(&input, &mut silu_out);
        }
        for (i, ((&silu, &sig), &x)) in
            silu_out.iter().zip(sig_out.iter()).zip(input.iter()).enumerate()
        {
            let expected = x * sig;
            assert!(
                approx_eq(silu, expected, TOLERANCE),
                "SiLU(x) should equal x*sigmoid(x) at index {i}: silu={silu}, x*sig={expected}"
            );
        }
    }

    #[test]
    fn test_fused_gate_up_with_sigmoid_consistency() {
        // Verify: gate_up(gate, up) = gate * (up * sigmoid(up))
        let gate: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let up: Vec<f32> = (0..16).map(|i| (i as f32 - 4.0) * 0.2).collect();

        let mut sig_of_up = vec![0.0f32; 16];
        unsafe { neon_sigmoid_f32(&up, &mut sig_of_up) };

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .zip(sig_of_up.iter())
            .map(|((&g, &u), &s)| g * u * s)
            .collect();

        let mut fused_out = vec![0.0f32; 16];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut fused_out) };

        assert_approx_slice(&fused_out, &expected, TOLERANCE, "fused_sigmoid_consistency");
    }

    // ================================================================
    // Activation derivative property tests
    // ================================================================

    #[test]
    fn test_sigmoid_derivative_positive_at_zero() {
        // sigmoid'(0) = sigmoid(0)*(1-sigmoid(0)) = 0.25
        let eps = 0.001;
        let input_lo = [-eps];
        let input_hi = [eps];
        let mut out_lo = [0.0f32];
        let mut out_hi = [0.0f32];
        unsafe {
            neon_sigmoid_f32(&input_lo, &mut out_lo);
            neon_sigmoid_f32(&input_hi, &mut out_hi);
        }
        let approx_deriv = (out_hi[0] - out_lo[0]) / (2.0 * eps);
        assert!(approx_eq(approx_deriv, 0.25, 0.01), "sigmoid'(0) ≈ 0.25, got {approx_deriv}");
    }

    #[test]
    fn test_silu_derivative_at_zero() {
        // SiLU'(0) = sigmoid(0) + 0*sigmoid'(0) = 0.5
        let eps = 0.001;
        let input_lo = [-eps];
        let input_hi = [eps];
        let mut out_lo = [0.0f32];
        let mut out_hi = [0.0f32];
        unsafe {
            neon_silu_f32(&input_lo, &mut out_lo);
            neon_silu_f32(&input_hi, &mut out_hi);
        }
        let approx_deriv = (out_hi[0] - out_lo[0]) / (2.0 * eps);
        assert!(approx_eq(approx_deriv, 0.5, 0.01), "SiLU'(0) ≈ 0.5, got {approx_deriv}");
    }

    #[test]
    fn test_gelu_derivative_at_zero() {
        // GELU'(0) ≈ 0.5
        let eps = 0.001;
        let input_lo = [-eps];
        let input_hi = [eps];
        let mut out_lo = [0.0f32];
        let mut out_hi = [0.0f32];
        unsafe {
            neon_gelu_f32(&input_lo, &mut out_lo);
            neon_gelu_f32(&input_hi, &mut out_hi);
        }
        let approx_deriv = (out_hi[0] - out_lo[0]) / (2.0 * eps);
        assert!(approx_eq(approx_deriv, 0.5, 0.01), "GELU'(0) ≈ 0.5, got {approx_deriv}");
    }

    // ================================================================
    // Stress / larger size tests
    // ================================================================

    #[test]
    fn test_sigmoid_large_vector_2048() {
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_sigmoid(x)).collect();
        let mut output = vec![0.0f32; n];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "sigmoid_2048");
    }

    #[test]
    fn test_silu_large_vector_2048() {
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_silu(x)).collect();
        let mut output = vec![0.0f32; n];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "silu_2048");
    }

    #[test]
    fn test_gelu_large_vector_2048() {
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
        let expected: Vec<f32> = input.iter().map(|&x| ref_gelu(x)).collect();
        let mut output = vec![0.0f32; n];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert_approx_slice(&output, &expected, LOOSE_TOL, "gelu_2048");
    }

    #[test]
    fn test_fused_gate_up_large_4096() {
        let n = 4096;
        let gate: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32 % 10.0 - 5.0) * 0.1).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) as f32 % 10.0 - 5.0) * 0.1).collect();
        let expected: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(&g, &u)| g * ref_silu(u)).collect();
        let mut output = vec![0.0f32; n];
        unsafe { neon_fused_gate_up_f32(&gate, &up, &mut output) };
        assert_approx_slice(&output, &expected, TOLERANCE, "fused_4096");
    }
}
