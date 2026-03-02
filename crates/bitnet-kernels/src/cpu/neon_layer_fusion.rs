//! ARM NEON-accelerated fused layer operations for Apple Silicon.
//!
//! Provides fused kernels that combine multiple operations into a single pass,
//! reducing memory bandwidth by avoiding intermediate allocations. Each kernel
//! processes 4 × f32 lanes at a time with scalar fallback for remainder elements.
//!
//! Fused operations:
//! - Residual add + LayerNorm
//! - Scaled add (alpha·x + beta·y)
//! - RMS normalization
//! - Residual add + RMS normalization
//! - GELU activation approximation

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Fused residual add + LayerNorm ─────────────────────────────────

/// Fused residual addition and layer normalization.
///
/// Computes `out[i] = gamma[i] * normalize(residual[i] + input[i]) + beta[i]`
/// in a single pass, avoiding an intermediate buffer for the residual sum.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if any slice length differs from `input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_add_layernorm(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        // Pass 1: compute mean of (input + residual).
        let chunks = n / 4;
        let remainder = n % 4;
        let in_ptr = input.as_ptr();
        let res_ptr = residual.as_ptr();

        let mut sum_vec = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(in_ptr.add(off));
            let vr = vld1q_f32(res_ptr.add(off));
            sum_vec = vaddq_f32(sum_vec, vaddq_f32(vi, vr));
        }
        let mut sum: f32 = vaddvq_f32(sum_vec);
        let tail = chunks * 4;
        for i in 0..remainder {
            sum += input[tail + i] + residual[tail + i];
        }
        let mean = sum / n as f32;

        // Pass 2: compute variance of (input + residual) around mean.
        let mean_vec = vdupq_n_f32(mean);
        let mut var_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(in_ptr.add(off));
            let vr = vld1q_f32(res_ptr.add(off));
            let s = vaddq_f32(vi, vr);
            let diff = vsubq_f32(s, mean_vec);
            var_acc = vfmaq_f32(var_acc, diff, diff);
        }
        let mut var_sum: f32 = vaddvq_f32(var_acc);
        for i in 0..remainder {
            let d = (input[tail + i] + residual[tail + i]) - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

        // Pass 3: normalize and apply affine transform.
        let inv_std_vec = vdupq_n_f32(inv_std);
        let gam_ptr = gamma.as_ptr();
        let bet_ptr = beta.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(in_ptr.add(off));
            let vr = vld1q_f32(res_ptr.add(off));
            let s = vaddq_f32(vi, vr);
            let norm = vmulq_f32(vsubq_f32(s, mean_vec), inv_std_vec);
            let g = vld1q_f32(gam_ptr.add(off));
            let b = vld1q_f32(bet_ptr.add(off));
            let result = vfmaq_f32(b, g, norm);
            vst1q_f32(out_ptr.add(off), result);
        }
        for i in 0..remainder {
            let idx = tail + i;
            let s = input[idx] + residual[idx];
            let norm = (s - mean) * inv_std;
            output[idx] = gamma[idx] * norm + beta[idx];
        }
    }
}

// ── Fused scale + add ──────────────────────────────────────────────

/// Fused scaling and addition: `out[i] = alpha * x[i] + beta * y[i]`.
///
/// Performs the entire axpy-style operation in a single NEON pass,
/// avoiding two separate scale operations and an addition.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `y` or `output` length differs from `x.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_scale_add(
    x: &[f32],
    y: &[f32],
    alpha: f32,
    beta: f32,
    output: &mut [f32],
) {
    let n = x.len();
    assert_eq!(y.len(), n, "y length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / 4;
        let remainder = n % 4;
        let alpha_vec = vdupq_n_f32(alpha);
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(x_ptr.add(off));
            let vy = vld1q_f32(y_ptr.add(off));
            // alpha * x + beta * y = fma(alpha_vec, vx, beta * vy)
            let beta_y = vmulq_n_f32(vy, beta);
            let result = vfmaq_f32(beta_y, alpha_vec, vx);
            vst1q_f32(out_ptr.add(off), result);
        }

        let tail = chunks * 4;
        for i in 0..remainder {
            let idx = tail + i;
            output[idx] = alpha * x[idx] + beta * y[idx];
        }
    }
}

// ── Fused RMS normalization ────────────────────────────────────────

/// Fused RMS normalization: `out[i] = gamma[i] * x[i] / rms(x)`.
///
/// RMS normalization skips mean subtraction, computing only
/// `rms = sqrt(mean(x²) + eps)` then scaling.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gamma` or `output` length differs from `input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_rms_norm(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / 4;
        let remainder = n % 4;
        let in_ptr = input.as_ptr();

        // Pass 1: mean of squares.
        let mut sq_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(in_ptr.add(i * 4));
            sq_acc = vfmaq_f32(sq_acc, v, v);
        }
        let mut sq_sum: f32 = vaddvq_f32(sq_acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let x = input[tail + i];
            sq_sum += x * x;
        }
        let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

        // Pass 2: scale by gamma * inv_rms.
        let inv_rms_vec = vdupq_n_f32(inv_rms);
        let gam_ptr = gamma.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(in_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));
            let scaled = vmulq_f32(v, inv_rms_vec);
            let result = vmulq_f32(g, scaled);
            vst1q_f32(out_ptr.add(off), result);
        }
        for i in 0..remainder {
            let idx = tail + i;
            output[idx] = gamma[idx] * input[idx] * inv_rms;
        }
    }
}

// ── Fused residual add + RMS normalization ─────────────────────────

/// Fused residual addition and RMS normalization.
///
/// Computes `out[i] = gamma[i] * (residual[i] + input[i]) / rms(residual + input)`
/// in a single fused operation.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if any slice length differs from `input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_add_rms_norm(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / 4;
        let remainder = n % 4;
        let in_ptr = input.as_ptr();
        let res_ptr = residual.as_ptr();

        // Pass 1: mean of squares of (input + residual).
        let mut sq_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(in_ptr.add(off));
            let vr = vld1q_f32(res_ptr.add(off));
            let s = vaddq_f32(vi, vr);
            sq_acc = vfmaq_f32(sq_acc, s, s);
        }
        let mut sq_sum: f32 = vaddvq_f32(sq_acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let s = input[tail + i] + residual[tail + i];
            sq_sum += s * s;
        }
        let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

        // Pass 2: scale by gamma * inv_rms.
        let inv_rms_vec = vdupq_n_f32(inv_rms);
        let gam_ptr = gamma.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(in_ptr.add(off));
            let vr = vld1q_f32(res_ptr.add(off));
            let s = vaddq_f32(vi, vr);
            let g = vld1q_f32(gam_ptr.add(off));
            let scaled = vmulq_f32(s, inv_rms_vec);
            let result = vmulq_f32(g, scaled);
            vst1q_f32(out_ptr.add(off), result);
        }
        for i in 0..remainder {
            let idx = tail + i;
            let s = input[idx] + residual[idx];
            output[idx] = gamma[idx] * s * inv_rms;
        }
    }
}

// ── Fused GELU activation ──────────────────────────────────────────

/// Fused approximate GELU activation using NEON.
///
/// Computes `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` with
/// NEON-accelerated polynomial evaluation and scalar tanh.
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
pub unsafe fn neon_fused_gelu(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert!(output.len() >= n, "output buffer too small");

    if n == 0 {
        return;
    }

    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

    unsafe {
        let chunks = n / 4;
        let remainder = n % 4;
        let coeff = vdupq_n_f32(0.044715);
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let s2p = vdupq_n_f32(sqrt_2_over_pi);
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(in_ptr.add(off));
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            // inner = sqrt(2/π) * (x + 0.044715 * x³)
            let cubic = vfmaq_f32(x, coeff, x3);
            let inner = vmulq_f32(s2p, cubic);

            // Scalar tanh for each lane, then reload.
            let mut tanh_buf = [0.0_f32; 4];
            vst1q_f32(tanh_buf.as_mut_ptr(), inner);
            tanh_buf[0] = tanh_buf[0].tanh();
            tanh_buf[1] = tanh_buf[1].tanh();
            tanh_buf[2] = tanh_buf[2].tanh();
            tanh_buf[3] = tanh_buf[3].tanh();
            let tanh_vec = vld1q_f32(tanh_buf.as_ptr());

            // 0.5 * x * (1 + tanh)
            let one_plus_tanh = vaddq_f32(one, tanh_vec);
            let result = vmulq_f32(half, vmulq_f32(x, one_plus_tanh));
            vst1q_f32(out_ptr.add(off), result);
        }

        let tail = chunks * 4;
        for i in 0..remainder {
            let idx = tail + i;
            let x = input[idx];
            let x3 = x * x * x;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
            output[idx] = 0.5 * x * (1.0 + inner.tanh());
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_slices_approx(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(*a, *e, tol),
                "{msg}[{i}]: expected {e}, got {a} (diff={})",
                (a - e).abs()
            );
        }
    }

    /// Scalar reference for LayerNorm.
    fn ref_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter().zip(beta.iter()))
            .map(|(x, (g, b))| g * (x - mean) * inv_std + b)
            .collect()
    }

    /// Scalar reference for RMS norm.
    fn ref_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        input.iter().zip(gamma.iter()).map(|(x, g)| g * x * inv_rms).collect()
    }

    /// Scalar reference for GELU.
    fn ref_gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        0.5 * x * (1.0 + inner.tanh())
    }

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    // ── neon_fused_add_layernorm tests ──────────────────────────────

    #[test]
    fn test_fused_add_layernorm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let residual = [0.1, 0.2, 0.3, 0.4];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_add_layernorm(&input, &residual, &gamma, &beta, EPS, &mut output);
        }

        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let expected = ref_layernorm(&combined, &gamma, &beta, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_add_layernorm_basic");
    }

    #[test]
    fn test_fused_add_layernorm_with_affine() {
        let input = [1.0_f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 1.5];
        let residual = [0.5; 8];
        let gamma = [2.0_f32, 1.0, 0.5, 2.0, 1.0, 0.5, 2.0, 1.0];
        let beta = [0.1_f32, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let mut output = [0.0_f32; 8];

        unsafe {
            neon_fused_add_layernorm(&input, &residual, &gamma, &beta, EPS, &mut output);
        }

        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let expected = ref_layernorm(&combined, &gamma, &beta, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_add_layernorm_affine");
    }

    #[test]
    fn test_fused_add_layernorm_empty() {
        let mut output = [];
        unsafe {
            neon_fused_add_layernorm(&[], &[], &[], &[], EPS, &mut output);
        }
    }

    #[test]
    fn test_fused_add_layernorm_non_aligned() {
        // 7 elements: exercises scalar tail (7 % 4 = 3).
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let residual = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let gamma = [1.0; 7];
        let beta = [0.0; 7];
        let mut output = [0.0_f32; 7];

        unsafe {
            neon_fused_add_layernorm(&input, &residual, &gamma, &beta, EPS, &mut output);
        }

        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let expected = ref_layernorm(&combined, &gamma, &beta, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_add_layernorm_non_aligned");
    }

    // ── neon_fused_scale_add tests ──────────────────────────────────

    #[test]
    fn test_fused_scale_add_basic() {
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let y = [5.0_f32, 6.0, 7.0, 8.0];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_scale_add(&x, &y, 2.0, 3.0, &mut output);
        }

        let expected: Vec<f32> =
            x.iter().zip(y.iter()).map(|(xi, yi)| 2.0 * xi + 3.0 * yi).collect();
        assert_slices_approx(&output, &expected, TOL, "fused_scale_add_basic");
    }

    #[test]
    fn test_fused_scale_add_identity() {
        // alpha=1, beta=0 → copy x.
        let x = [3.0_f32, -1.0, 0.0, 7.0, 2.5];
        let y = [99.0_f32; 5];
        let mut output = [0.0_f32; 5];

        unsafe {
            neon_fused_scale_add(&x, &y, 1.0, 0.0, &mut output);
        }

        assert_slices_approx(&output, &x, TOL, "fused_scale_add_identity");
    }

    #[test]
    fn test_fused_scale_add_negative_alpha() {
        let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [1.0_f32; 6];
        let mut output = [0.0_f32; 6];

        unsafe {
            neon_fused_scale_add(&x, &y, -1.0, 1.0, &mut output);
        }

        let expected: Vec<f32> = x.iter().map(|xi| -xi + 1.0).collect();
        assert_slices_approx(&output, &expected, TOL, "fused_scale_add_negative");
    }

    #[test]
    fn test_fused_scale_add_empty() {
        let mut output = [];
        unsafe {
            neon_fused_scale_add(&[], &[], 1.0, 1.0, &mut output);
        }
    }

    // ── neon_fused_rms_norm tests ───────────────────────────────────

    #[test]
    fn test_fused_rms_norm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_rms_norm(&input, &gamma, EPS, &mut output);
        }

        let expected = ref_rms_norm(&input, &gamma, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_rms_norm_basic");
    }

    #[test]
    fn test_fused_rms_norm_with_gamma() {
        let input = [1.0_f32, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
        let gamma = [2.0_f32, 0.5, 1.0, 3.0, 0.1, 2.0, 0.5];
        let mut output = [0.0_f32; 7];

        unsafe {
            neon_fused_rms_norm(&input, &gamma, EPS, &mut output);
        }

        let expected = ref_rms_norm(&input, &gamma, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_rms_norm_gamma");
    }

    #[test]
    fn test_fused_rms_norm_uniform() {
        // All same value → output = gamma * sign(val).
        let input = [2.0_f32; 8];
        let gamma = [1.0; 8];
        let mut output = [0.0_f32; 8];

        unsafe {
            neon_fused_rms_norm(&input, &gamma, EPS, &mut output);
        }

        let expected = ref_rms_norm(&input, &gamma, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_rms_norm_uniform");
    }

    #[test]
    fn test_fused_rms_norm_empty() {
        let mut output = [];
        unsafe {
            neon_fused_rms_norm(&[], &[], EPS, &mut output);
        }
    }

    // ── neon_fused_add_rms_norm tests ───────────────────────────────

    #[test]
    fn test_fused_add_rms_norm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let residual = [0.5, -0.5, 0.5, -0.5];
        let gamma = [1.0; 4];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_add_rms_norm(&input, &residual, &gamma, EPS, &mut output);
        }

        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let expected = ref_rms_norm(&combined, &gamma, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_add_rms_norm_basic");
    }

    #[test]
    fn test_fused_add_rms_norm_non_aligned() {
        // 9 elements: exercises tail (9 % 4 = 1).
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let residual = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let gamma = [1.0; 9];
        let mut output = [0.0_f32; 9];

        unsafe {
            neon_fused_add_rms_norm(&input, &residual, &gamma, EPS, &mut output);
        }

        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let expected = ref_rms_norm(&combined, &gamma, EPS);
        assert_slices_approx(&output, &expected, TOL, "fused_add_rms_norm_non_aligned");
    }

    #[test]
    fn test_fused_add_rms_norm_empty() {
        let mut output = [];
        unsafe {
            neon_fused_add_rms_norm(&[], &[], &[], EPS, &mut output);
        }
    }

    // ── neon_fused_gelu tests ───────────────────────────────────────

    #[test]
    fn test_fused_gelu_basic() {
        let input = [0.0_f32, 1.0, -1.0, 2.0];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        let expected: Vec<f32> = input.iter().map(|x| ref_gelu(*x)).collect();
        assert_slices_approx(&output, &expected, TOL, "fused_gelu_basic");
    }

    #[test]
    fn test_fused_gelu_zero() {
        // GELU(0) = 0.
        let input = [0.0_f32; 8];
        let mut output = [1.0_f32; 8];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 0.0, TOL), "gelu(0)[{i}] = {v}, expected 0.0");
        }
    }

    #[test]
    fn test_fused_gelu_large_positive() {
        // For large x, GELU(x) ≈ x.
        let input = [10.0_f32, 20.0, 50.0, 100.0];
        let mut output = [0.0_f32; 4];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        for (i, (&o, &x)) in output.iter().zip(input.iter()).enumerate() {
            assert!(approx_eq(o, x, 0.01), "gelu({x})[{i}] = {o}, expected ≈ {x}");
        }
    }

    #[test]
    fn test_fused_gelu_large_negative() {
        // For large negative x, GELU(x) ≈ 0.
        let input = [-10.0_f32, -20.0, -50.0, -100.0];
        let mut output = [1.0_f32; 4];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        for (i, &v) in output.iter().enumerate() {
            assert!(approx_eq(v, 0.0, 0.01), "gelu({})[{i}] = {v}, expected ≈ 0.0", input[i]);
        }
    }

    #[test]
    fn test_fused_gelu_non_aligned() {
        // 6 elements: tail of 2.
        let input = [0.5_f32, -0.5, 1.0, -1.0, 0.25, -0.25];
        let mut output = [0.0_f32; 6];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        let expected: Vec<f32> = input.iter().map(|x| ref_gelu(*x)).collect();
        assert_slices_approx(&output, &expected, TOL, "fused_gelu_non_aligned");
    }

    #[test]
    fn test_fused_gelu_empty() {
        let mut output = [];
        unsafe {
            neon_fused_gelu(&[], &mut output);
        }
    }

    #[test]
    fn test_fused_gelu_single_element() {
        let input = [0.5_f32];
        let mut output = [0.0_f32; 1];

        unsafe {
            neon_fused_gelu(&input, &mut output);
        }

        let expected = ref_gelu(0.5);
        assert!(approx_eq(output[0], expected, TOL), "single element GELU");
    }
}
