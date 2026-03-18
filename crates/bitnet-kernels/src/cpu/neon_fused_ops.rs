//! NEON-optimized fused compound operations for Apple Silicon.
//!
//! Combines multiple common operations into single passes for better
//! cache efficiency. Each kernel has an ARM NEON path gated behind
//! `#[cfg(target_arch = "aarch64")]` and a scalar fallback for other
//! architectures.
//!
//! # Kernels
//!
//! - [`fused_layernorm_residual`] — LayerNorm + residual add
//! - [`fused_gelu_mul`] — GELU activation × gate (SwiGLU/GeGLU)
//! - [`fused_scale_add`] — α×A + β×B
//! - [`fused_rms_norm_mul`] — RMSNorm + element-wise multiply
//! - [`fused_bias_relu`] — bias add + ReLU
//! - [`fused_softmax_scale`] — scale + softmax

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ─── helpers ───────────────────────────────────────────────────────

/// Scalar GELU approximation (tanh-based).
#[inline(always)]
fn gelu_scalar(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

// ═══════════════════════════════════════════════════════════════════
// 1. fused_layernorm_residual
//    out[i] = ((x[i] - mean) / sqrt(var + eps)) * gamma[i]
//             + beta[i] + residual[i]
// ═══════════════════════════════════════════════════════════════════

/// Fused LayerNorm + residual addition in one pass.
///
/// # Panics
///
/// Panics if slice lengths do not match or are zero.
#[cfg(target_arch = "aarch64")]
pub fn fused_layernorm_residual(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    residual: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, residual.len());
    assert_eq!(n, output.len());

    let chunks = n / LANES;

    // ── mean ──
    let mut sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * LANES));
            acc = vaddq_f32(acc, v);
        }
        vaddvq_f32(acc)
    };
    for val in input.iter().skip(chunks * LANES) {
        sum += val;
    }
    let mean = sum / n as f32;

    // ── variance ──
    let mut var_sum = unsafe {
        let mean_v = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * LANES));
            let d = vsubq_f32(v, mean_v);
            acc = vfmaq_f32(acc, d, d);
        }
        vaddvq_f32(acc)
    };
    for val in input.iter().skip(chunks * LANES) {
        let d = val - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // ── normalize + scale + bias + residual ──
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let g = vld1q_f32(gamma.as_ptr().add(off));
            let b = vld1q_f32(beta.as_ptr().add(off));
            let r = vld1q_f32(residual.as_ptr().add(off));
            let normed = vmulq_f32(vsubq_f32(x, mean_v), inv_v);
            let scaled = vfmaq_f32(b, normed, g); // g*normed + b
            let out = vaddq_f32(scaled, r);
            vst1q_f32(output.as_mut_ptr().add(off), out);
        }
    }
    for (i, val) in output.iter_mut().enumerate().skip(chunks * LANES).take(n - chunks * LANES) {
        let normed = (input[i] - mean) * inv_std;
        *val = normed * gamma[i] + beta[i] + residual[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_layernorm_residual(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    residual: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, residual.len());
    assert_eq!(n, output.len());

    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    for (i, val) in output.iter_mut().enumerate().take(n) {
        let normed = (input[i] - mean) * inv_std;
        *val = normed * gamma[i] + beta[i] + residual[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. fused_gelu_mul
//    out[i] = GELU(x[i]) * gate[i]
// ═══════════════════════════════════════════════════════════════════

/// Fused GELU activation × gate multiply (SwiGLU / GeGLU pattern).
///
/// # Panics
///
/// Panics if slice lengths do not match.
#[cfg(target_arch = "aarch64")]
pub fn fused_gelu_mul(input: &[f32], gate: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, gate.len());
    assert_eq!(n, output.len());

    // GELU uses tanh which has no NEON intrinsic; compute element-wise
    // but fuse the multiply to avoid a second pass.
    unsafe {
        let chunks = n / LANES;
        for i in 0..chunks {
            let off = i * LANES;
            // Compute GELU per-element then load gate via NEON
            let mut tmp = [0.0_f32; 4];
            for (j, t) in tmp.iter_mut().enumerate() {
                *t = gelu_scalar(input[off + j]);
            }
            let gelu_v = vld1q_f32(tmp.as_ptr());
            let gate_v = vld1q_f32(gate.as_ptr().add(off));
            let out_v = vmulq_f32(gelu_v, gate_v);
            vst1q_f32(output.as_mut_ptr().add(off), out_v);
        }
    }
    for (i, val) in
        output.iter_mut().enumerate().skip((n / LANES) * LANES).take(n - (n / LANES) * LANES)
    {
        *val = gelu_scalar(input[i]) * gate[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_gelu_mul(input: &[f32], gate: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, gate.len());
    assert_eq!(n, output.len());

    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = gelu_scalar(input[i]) * gate[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. fused_scale_add
//    out[i] = alpha * a[i] + beta_coeff * b[i]
// ═══════════════════════════════════════════════════════════════════

/// Fused α×A + β×B in one pass.
///
/// # Panics
///
/// Panics if slice lengths do not match.
#[cfg(target_arch = "aarch64")]
pub fn fused_scale_add(a: &[f32], b: &[f32], alpha: f32, beta_coeff: f32, output: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, output.len());

    let chunks = n / LANES;
    unsafe {
        let alpha_v = vdupq_n_f32(alpha);
        let beta_v = vdupq_n_f32(beta_coeff);
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let sa = vmulq_f32(va, alpha_v);
            let out = vfmaq_f32(sa, vb, beta_v); // sa + vb*beta
            vst1q_f32(output.as_mut_ptr().add(off), out);
        }
    }
    for (i, val) in output.iter_mut().enumerate().skip(chunks * LANES).take(n - chunks * LANES) {
        *val = alpha * a[i] + beta_coeff * b[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_scale_add(a: &[f32], b: &[f32], alpha: f32, beta_coeff: f32, output: &mut [f32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    assert_eq!(n, output.len());

    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = alpha * a[i] + beta_coeff * b[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. fused_rms_norm_mul
//    tmp[i] = (x[i] / rms) * weight[i]
//    out[i] = tmp[i] * multiplier[i]
// ═══════════════════════════════════════════════════════════════════

/// Fused RMSNorm + element-wise multiply in one pass.
///
/// # Panics
///
/// Panics if slice lengths do not match or are zero.
#[cfg(target_arch = "aarch64")]
pub fn fused_rms_norm_mul(
    input: &[f32],
    weight: &[f32],
    multiplier: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, weight.len());
    assert_eq!(n, multiplier.len());
    assert_eq!(n, output.len());

    let chunks = n / LANES;

    // ── sum of squares ──
    let mut sq_sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * LANES));
            acc = vfmaq_f32(acc, v, v);
        }
        vaddvq_f32(acc)
    };
    for val in input.iter().skip(chunks * LANES) {
        sq_sum += val * val;
    }
    let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

    // ── normalize × weight × multiplier ──
    unsafe {
        let inv_v = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let w = vld1q_f32(weight.as_ptr().add(off));
            let m = vld1q_f32(multiplier.as_ptr().add(off));
            let normed = vmulq_f32(x, inv_v);
            let scaled = vmulq_f32(normed, w);
            let out = vmulq_f32(scaled, m);
            vst1q_f32(output.as_mut_ptr().add(off), out);
        }
    }
    for (i, val) in output.iter_mut().enumerate().skip(chunks * LANES).take(n - chunks * LANES) {
        *val = input[i] * inv_rms * weight[i] * multiplier[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_rms_norm_mul(
    input: &[f32],
    weight: &[f32],
    multiplier: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, weight.len());
    assert_eq!(n, multiplier.len());
    assert_eq!(n, output.len());

    let sq_sum: f32 = input.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = input[i] * inv_rms * weight[i] * multiplier[i];
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. fused_bias_relu
//    out[i] = max(0, x[i] + bias[i])
// ═══════════════════════════════════════════════════════════════════

/// Fused bias-add + ReLU activation in one pass.
///
/// # Panics
///
/// Panics if slice lengths do not match.
#[cfg(target_arch = "aarch64")]
pub fn fused_bias_relu(input: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, bias.len());
    assert_eq!(n, output.len());

    let chunks = n / LANES;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(bias.as_ptr().add(off));
            let sum = vaddq_f32(x, b);
            let relu = vmaxq_f32(sum, zero);
            vst1q_f32(output.as_mut_ptr().add(off), relu);
        }
    }
    for (i, val) in output.iter_mut().enumerate().skip(chunks * LANES).take(n - chunks * LANES) {
        *val = (input[i] + bias[i]).max(0.0);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_bias_relu(input: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, bias.len());
    assert_eq!(n, output.len());

    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = (input[i] + bias[i]).max(0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. fused_softmax_scale
//    out[i] = softmax(x[i] * scale)
// ═══════════════════════════════════════════════════════════════════

/// Fused scale + softmax in one pass (for attention scores).
///
/// # Panics
///
/// Panics if slice lengths do not match or are zero.
#[cfg(target_arch = "aarch64")]
pub fn fused_softmax_scale(input: &[f32], scale: f32, output: &mut [f32]) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, output.len());

    let chunks = n / LANES;

    // ── find max(x * scale) for numerical stability ──
    let mut max_val = unsafe {
        let scale_v = vdupq_n_f32(scale);
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let x = vld1q_f32(input.as_ptr().add(i * LANES));
            let sx = vmulq_f32(x, scale_v);
            max_v = vmaxq_f32(max_v, sx);
        }
        vmaxvq_f32(max_v)
    };
    for val in input.iter().skip(chunks * LANES) {
        let sv = val * scale;
        if sv > max_val {
            max_val = sv;
        }
    }

    // ── exp(x * scale - max) and sum ──
    let mut exp_sum = 0.0_f32;
    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = (input[i] * scale - max_val).exp();
        exp_sum += *val;
    }

    // ── normalise ──
    if exp_sum > 0.0 {
        let inv_sum = 1.0 / exp_sum;
        let chunks_out = n / LANES;
        unsafe {
            let inv_v = vdupq_n_f32(inv_sum);
            for i in 0..chunks_out {
                let off = i * LANES;
                let v = vld1q_f32(output.as_ptr().add(off));
                let normed = vmulq_f32(v, inv_v);
                vst1q_f32(output.as_mut_ptr().add(off), normed);
            }
        }
        for val in output.iter_mut().skip(chunks_out * LANES) {
            *val *= inv_sum;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn fused_softmax_scale(input: &[f32], scale: f32, output: &mut [f32]) {
    let n = input.len();
    assert!(n > 0, "input must be non-empty");
    assert_eq!(n, output.len());

    let max_val = input.iter().map(|x| x * scale).fold(f32::NEG_INFINITY, f32::max);

    let mut exp_sum = 0.0_f32;
    for (i, val) in output.iter_mut().enumerate().take(n) {
        *val = (input[i] * scale - max_val).exp();
        exp_sum += *val;
    }

    if exp_sum > 0.0 {
        let inv = 1.0 / exp_sum;
        for val in output.iter_mut().take(n) {
            *val *= inv;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    // ── reference implementations ──────────────────────────────────

    fn ref_layernorm_residual(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        residual: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let n = input.len();
        let mean = input.iter().sum::<f32>() / n as f32;
        let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .zip(residual.iter())
            .map(|(((x, g), b), r)| (x - mean) * inv_std * g + b + r)
            .collect()
    }

    fn ref_gelu_mul(input: &[f32], gate: &[f32]) -> Vec<f32> {
        input.iter().zip(gate.iter()).map(|(x, g)| gelu_scalar(*x) * g).collect()
    }

    fn ref_scale_add(a: &[f32], b: &[f32], alpha: f32, beta_c: f32) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| alpha * x + beta_c * y).collect()
    }

    fn ref_rms_norm_mul(input: &[f32], weight: &[f32], multiplier: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sq: f32 = input.iter().map(|x| x * x).sum();
        let inv = 1.0 / (sq / n as f32 + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .zip(multiplier.iter())
            .map(|((x, w), m)| x * inv * w * m)
            .collect()
    }

    fn ref_bias_relu(input: &[f32], bias: &[f32]) -> Vec<f32> {
        input.iter().zip(bias.iter()).map(|(x, b)| (x + b).max(0.0)).collect()
    }

    fn ref_softmax_scale(input: &[f32], scale: f32) -> Vec<f32> {
        let max_v = input.iter().map(|x| x * scale).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|x| (x * scale - max_v).exp()).collect();
        let s: f32 = exps.iter().sum();
        if s > 0.0 { exps.iter().map(|e| e / s).collect() } else { exps }
    }

    // ───────────────────────────────────────────────────────────────
    // 1. fused_layernorm_residual tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_ln_res_aligned_4() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [0.1, 0.2, 0.3, 0.4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_aligned_8() {
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let g = [1.0; 8];
        let b = [0.0; 8];
        let r = [0.5; 8];
        let mut out = [0.0; 8];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_unaligned_5() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g = [1.0; 5];
        let b = [0.0; 5];
        let r = [0.0; 5];
        let mut out = [0.0; 5];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_unaligned_7() {
        let x: Vec<f32> = (1..=7).map(|i| i as f32 * 0.3).collect();
        let g = [0.5; 7];
        let b = [0.1; 7];
        let r = [-0.1; 7];
        let mut out = [0.0; 7];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_single() {
        let x = [5.0];
        let g = [2.0];
        let b = [1.0];
        let r = [0.5];
        let mut out = [0.0];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        // single element ⇒ mean=5, var≈0, normed≈0
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_zero_input() {
        let x = [0.0; 4];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [1.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_constant_input() {
        let x = [3.0; 8];
        let g = [1.0; 8];
        let b = [0.0; 8];
        let r = [0.0; 8];
        let mut out = [0.0; 8];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        // constant ⇒ normed ≈ 0 ⇒ output ≈ beta + residual
        for val in &out {
            assert!(val.abs() < 0.01);
        }
    }

    #[test]
    fn test_ln_res_negative_values() {
        let x = [-1.0, -2.0, -3.0, -4.0];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [0.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_large_residual() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [100.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_non_unit_gamma_beta() {
        let x = [2.0, 4.0, 6.0, 8.0];
        let g = [0.5, 1.0, 1.5, 2.0];
        let b = [0.1, 0.2, 0.3, 0.4];
        let r = [0.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_large_16() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
        let g = [1.0; 16];
        let b = [0.0; 16];
        let r: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut out = [0.0; 16];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_large_eps() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [0.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, 1.0);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_ln_res_large_128() {
        let x: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let g = [1.0; 128];
        let b = [0.0; 128];
        let r = [0.5; 128];
        let mut out = [0.0; 128];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    #[should_panic]
    fn test_ln_res_empty_panics() {
        let mut out: [f32; 0] = [];
        fused_layernorm_residual(&[], &[], &[], &[], &mut out, EPS);
    }

    // ───────────────────────────────────────────────────────────────
    // 2. fused_gelu_mul tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_gelu_mul_aligned_4() {
        let x = [1.0, -1.0, 2.0, -2.0];
        let g = [1.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_aligned_8() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
        let g = [1.0; 8];
        let mut out = [0.0; 8];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_unaligned_5() {
        let x = [0.5, 1.0, -0.5, -1.0, 2.0];
        let g = [1.0; 5];
        let mut out = [0.0; 5];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_unaligned_3() {
        let x = [1.0, 2.0, 3.0];
        let g = [0.5, 1.0, 2.0];
        let mut out = [0.0; 3];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_zero_input() {
        let x = [0.0; 4];
        let g = [1.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        // GELU(0) = 0
        for val in &out {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_gelu_mul_zero_gate() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [0.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        for val in &out {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_gelu_mul_negative_gate() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let g = [-1.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_single() {
        let x = [1.5];
        let g = [2.0];
        let mut out = [0.0];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_empty() {
        let mut out: [f32; 0] = [];
        fused_gelu_mul(&[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gelu_mul_large_positive() {
        let x = [10.0; 4];
        let g = [1.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        // GELU(10) ≈ 10
        for val in &out {
            assert!((val - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_gelu_mul_large_negative() {
        let x = [-10.0; 4];
        let g = [1.0; 4];
        let mut out = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out);
        // GELU(-10) ≈ 0
        for val in &out {
            assert!(val.abs() < 0.01);
        }
    }

    #[test]
    fn test_gelu_mul_large_16() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let g = [1.0; 16];
        let mut out = [0.0; 16];
        fused_gelu_mul(&x, &g, &mut out);
        let exp = ref_gelu_mul(&x, &g);
        assert_approx(&out, &exp, TOL);
    }

    // ───────────────────────────────────────────────────────────────
    // 3. fused_scale_add tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_add_aligned_4() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 4];
        fused_scale_add(&a, &b, 2.0, 3.0, &mut out);
        let exp = ref_scale_add(&a, &b, 2.0, 3.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_aligned_8() {
        let a: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
        let mut out = [0.0; 8];
        fused_scale_add(&a, &b, 1.0, 1.0, &mut out);
        let exp = ref_scale_add(&a, &b, 1.0, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_unaligned_5() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 5];
        fused_scale_add(&a, &b, 0.5, 0.5, &mut out);
        let exp = ref_scale_add(&a, &b, 0.5, 0.5);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_unaligned_3() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        fused_scale_add(&a, &b, 2.0, -1.0, &mut out);
        let exp = ref_scale_add(&a, &b, 2.0, -1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_zero_alpha() {
        let a = [1.0; 4];
        let b = [2.0; 4];
        let mut out = [0.0; 4];
        fused_scale_add(&a, &b, 0.0, 1.0, &mut out);
        let exp = ref_scale_add(&a, &b, 0.0, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_zero_beta() {
        let a = [1.0; 4];
        let b = [2.0; 4];
        let mut out = [0.0; 4];
        fused_scale_add(&a, &b, 1.0, 0.0, &mut out);
        let exp = ref_scale_add(&a, &b, 1.0, 0.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_negative_coeffs() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 4];
        fused_scale_add(&a, &b, -1.0, -1.0, &mut out);
        let exp = ref_scale_add(&a, &b, -1.0, -1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_scale_add_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.0; 4];
        let mut out = [0.0; 4];
        fused_scale_add(&a, &b, 1.0, 0.0, &mut out);
        assert_approx(&out, &a, TOL);
    }

    #[test]
    fn test_scale_add_single() {
        let a = [3.0];
        let b = [7.0];
        let mut out = [0.0];
        fused_scale_add(&a, &b, 2.0, 3.0, &mut out);
        assert!((out[0] - 27.0).abs() < TOL);
    }

    #[test]
    fn test_scale_add_empty() {
        let mut out: [f32; 0] = [];
        fused_scale_add(&[], &[], 1.0, 1.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_scale_add_large_64() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut out = [0.0; 64];
        fused_scale_add(&a, &b, 1.5, 0.3, &mut out);
        let exp = ref_scale_add(&a, &b, 1.5, 0.3);
        assert_approx(&out, &exp, TOL);
    }

    // ───────────────────────────────────────────────────────────────
    // 4. fused_rms_norm_mul tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_rms_mul_aligned_4() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let m = [1.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_aligned_8() {
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let w = [1.0; 8];
        let m = [2.0; 8];
        let mut out = [0.0; 8];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_unaligned_5() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0; 5];
        let m = [1.0; 5];
        let mut out = [0.0; 5];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_unaligned_6() {
        let x = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let w = [0.5; 6];
        let m = [2.0; 6];
        let mut out = [0.0; 6];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_single() {
        let x = [5.0];
        let w = [2.0];
        let m = [3.0];
        let mut out = [0.0];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_zero_multiplier() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let m = [0.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        for val in &out {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_rms_mul_non_unit_weights() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [0.5, 1.0, 1.5, 2.0];
        let m = [1.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_zero_input() {
        let x = [0.0; 4];
        let w = [1.0; 4];
        let m = [1.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        for val in &out {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_rms_mul_negative_values() {
        let x = [-1.0, -2.0, -3.0, -4.0];
        let w = [1.0; 4];
        let m = [1.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_rms_mul_large_32() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
        let w = [1.0; 32];
        let m: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut out = [0.0; 32];
        fused_rms_norm_mul(&x, &w, &m, &mut out, EPS);
        let exp = ref_rms_norm_mul(&x, &w, &m, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    #[should_panic]
    fn test_rms_mul_empty_panics() {
        let mut out: [f32; 0] = [];
        fused_rms_norm_mul(&[], &[], &[], &mut out, EPS);
    }

    #[test]
    fn test_rms_mul_large_eps() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let m = [1.0; 4];
        let mut out = [0.0; 4];
        fused_rms_norm_mul(&x, &w, &m, &mut out, 1.0);
        let exp = ref_rms_norm_mul(&x, &w, &m, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    // ───────────────────────────────────────────────────────────────
    // 5. fused_bias_relu tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_bias_relu_aligned_4() {
        let x = [1.0, -1.0, 2.0, -2.0];
        let b = [0.5; 4];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_aligned_8() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
        let b = [0.0; 8];
        let mut out = [0.0; 8];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_unaligned_5() {
        let x = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let b = [1.0; 5];
        let mut out = [0.0; 5];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_unaligned_3() {
        let x = [-5.0, 0.0, 5.0];
        let b = [2.0, -1.0, -6.0];
        let mut out = [0.0; 3];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_all_negative() {
        let x = [-5.0, -3.0, -1.0, -0.5];
        let b = [0.0; 4];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        for val in &out {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_bias_relu_all_positive() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0; 4];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_zero_bias() {
        let x = [1.0, -1.0, 2.0, -2.0];
        let b = [0.0; 4];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        assert_approx(&out, &[1.0, 0.0, 2.0, 0.0], TOL);
    }

    #[test]
    fn test_bias_relu_negative_bias() {
        let x = [3.0, 3.0, 3.0, 3.0];
        let b = [-1.0, -3.0, -5.0, -7.0];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        assert_approx(&out, &[2.0, 0.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn test_bias_relu_single() {
        let x = [-1.0];
        let b = [2.0];
        let mut out = [0.0];
        fused_bias_relu(&x, &b, &mut out);
        assert!((out[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_bias_relu_empty() {
        let mut out: [f32; 0] = [];
        fused_bias_relu(&[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_bias_relu_large_32() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let b = [1.0; 32];
        let mut out = [0.0; 32];
        fused_bias_relu(&x, &b, &mut out);
        let exp = ref_bias_relu(&x, &b);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_bias_relu_at_boundary() {
        let x = [-0.5, -0.25, 0.0, 0.25];
        let b = [0.5, 0.25, 0.0, -0.25];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        assert_approx(&out, &[0.0, 0.0, 0.0, 0.0], TOL);
    }

    // ───────────────────────────────────────────────────────────────
    // 6. fused_softmax_scale tests
    // ───────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_scale_aligned_4() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 1.0, &mut out);
        let exp = ref_softmax_scale(&x, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_softmax_scale_aligned_8() {
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0; 8];
        fused_softmax_scale(&x, 0.5, &mut out);
        let exp = ref_softmax_scale(&x, 0.5);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_softmax_scale_unaligned_5() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0; 5];
        fused_softmax_scale(&x, 1.0, &mut out);
        let exp = ref_softmax_scale(&x, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_softmax_scale_unaligned_3() {
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        fused_softmax_scale(&x, 2.0, &mut out);
        let exp = ref_softmax_scale(&x, 2.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_softmax_scale_sums_to_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 8];
        fused_softmax_scale(&x, 1.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn test_softmax_scale_uniform() {
        let x = [1.0; 4];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 1.0, &mut out);
        for val in &out {
            assert!((val - 0.25).abs() < TOL);
        }
    }

    #[test]
    fn test_softmax_scale_single() {
        let x = [5.0];
        let mut out = [0.0];
        fused_softmax_scale(&x, 1.0, &mut out);
        assert!((out[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_softmax_scale_zero_scale() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 0.0, &mut out);
        // scale=0 ⇒ all exp(0)=1 ⇒ uniform
        for val in &out {
            assert!((val - 0.25).abs() < TOL);
        }
    }

    #[test]
    fn test_softmax_scale_large_scale() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 10.0, &mut out);
        // largest dominates
        assert!(out[3] > 0.99);
    }

    #[test]
    fn test_softmax_scale_negative_inputs() {
        let x = [-1.0, -2.0, -3.0, -4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 1.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn test_softmax_scale_large_negative_inputs() {
        let x = [-100.0, -200.0, -300.0, -400.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 1.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        // first dominates
        assert!(out[0] > 0.99);
    }

    #[test]
    fn test_softmax_scale_small_scale() {
        let x = [1.0, 100.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 0.001, &mut out);
        let exp = ref_softmax_scale(&x, 0.001);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_softmax_scale_large_16() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.5).sin()).collect();
        let mut out = [0.0; 16];
        fused_softmax_scale(&x, 1.0, &mut out);
        let exp = ref_softmax_scale(&x, 1.0);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    #[should_panic]
    fn test_softmax_scale_empty_panics() {
        let mut out: [f32; 0] = [];
        fused_softmax_scale(&[], 1.0, &mut out);
    }

    #[test]
    fn test_softmax_scale_monotonic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, 1.0, &mut out);
        for pair in out.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn test_softmax_scale_negative_scale() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        fused_softmax_scale(&x, -1.0, &mut out);
        let exp = ref_softmax_scale(&x, -1.0);
        assert_approx(&out, &exp, TOL);
        // reversed: smallest input should dominate
        assert!(out[0] > out[3]);
    }

    // ── cross-kernel consistency tests ─────────────────────────────

    #[test]
    fn test_ln_res_zero_residual_equals_plain_ln() {
        let x = [1.0, 3.0, 5.0, 7.0];
        let g = [1.0; 4];
        let b = [0.0; 4];
        let r = [0.0; 4];
        let mut out = [0.0; 4];
        fused_layernorm_residual(&x, &g, &b, &r, &mut out, EPS);
        let exp = ref_layernorm_residual(&x, &g, &b, &r, EPS);
        assert_approx(&out, &exp, TOL);
    }

    #[test]
    fn test_gelu_mul_symmetry() {
        // GELU(x)*1 vs GELU(-x)*1 — GELU is not symmetric
        let x = [1.0, 2.0, 3.0, 4.0];
        let nx: Vec<f32> = x.iter().map(|v| -v).collect();
        let g = [1.0; 4];
        let mut out_pos = [0.0; 4];
        let mut out_neg = [0.0; 4];
        fused_gelu_mul(&x, &g, &mut out_pos);
        fused_gelu_mul(&nx, &g, &mut out_neg);
        // GELU(-x) ≈ 0 for large x, so they differ
        for (p, n) in out_pos.iter().zip(out_neg.iter()) {
            assert!((p - n).abs() > TOL);
        }
    }

    #[test]
    fn test_bias_relu_large_values() {
        let x = [1e6, -1e6, 1e6, -1e6];
        let b = [0.0; 4];
        let mut out = [0.0; 4];
        fused_bias_relu(&x, &b, &mut out);
        assert_approx(&out, &[1e6, 0.0, 1e6, 0.0], 1.0);
    }

    #[test]
    fn test_scale_add_commutative_swap() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out1 = [0.0; 4];
        let mut out2 = [0.0; 4];
        fused_scale_add(&a, &b, 1.0, 1.0, &mut out1);
        fused_scale_add(&b, &a, 1.0, 1.0, &mut out2);
        assert_approx(&out1, &out2, TOL);
    }
}
