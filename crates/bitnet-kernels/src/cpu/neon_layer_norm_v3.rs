//! ARM NEON-optimized LayerNorm v3 kernels for Apple Silicon (aarch64).
//!
//! Improved LayerNorm implementation with five operations:
//!
//! 1. Standard LayerNorm (two-pass: mean → variance + normalize)
//! 2. RMSNorm (used by LLaMA-style models)
//! 3. GroupNorm (channel-grouped normalization)
//! 4. LayerNorm backward pass (gradient computation)
//! 5. Fused LayerNorm + residual addition

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

// ── NEON helper: horizontal sum ────────────────────────────────────────

/// Sum all elements of a slice using NEON vector accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * LANES));
            acc = vaddq_f32(acc, v);
        }

        let mut total: f32 = vaddvq_f32(acc);

        let tail = chunks * LANES;
        for i in 0..remainder {
            total += data[tail + i];
        }
        total
    }
}

/// Sum of squares of all elements using NEON FMA.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_of_squares(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * LANES));
            acc = vfmaq_f32(acc, v, v);
        }

        let mut total: f32 = vaddvq_f32(acc);

        let tail = chunks * LANES;
        for i in 0..remainder {
            let x = data[tail + i];
            total += x * x;
        }
        total
    }
}

/// Sum of squared deviations from `center` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_sq_dev(data: &[f32], center: f32) -> f32 {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    unsafe {
        let c = vdupq_n_f32(center);
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * LANES));
            let diff = vsubq_f32(v, c);
            acc = vfmaq_f32(acc, diff, diff);
        }

        let mut total: f32 = vaddvq_f32(acc);

        let tail = chunks * LANES;
        for i in 0..remainder {
            let d = data[tail + i] - center;
            total += d * d;
        }
        total
    }
}

// ── Public kernels ─────────────────────────────────────────────────────

/// Standard LayerNorm: `output = gamma * ((input - mean) / sqrt(var + eps)) + beta`.
///
/// Two-pass algorithm:
/// - Pass 1: NEON-parallel mean computation.
/// - Pass 2: NEON-parallel variance + normalization + scale/shift.
///
/// # Panics
///
/// Panics if slice lengths are inconsistent.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All slices must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        // Pass 1: mean
        let mean = neon_sum(input) / n as f32;

        // Pass 2: variance, normalize, scale/shift
        let var = neon_sum_sq_dev(input, mean) / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        let chunks = n / LANES;
        let remainder = n % LANES;

        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        let inp = input.as_ptr();
        let gp = gamma.as_ptr();
        let bp = beta.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(inp.add(off));
            let g = vld1q_f32(gp.add(off));
            let b = vld1q_f32(bp.add(off));
            let centered = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(centered, inv_v);
            let result = vfmaq_f32(b, g, normed);
            vst1q_f32(op.add(off), result);
        }

        let tail = chunks * LANES;
        for i in 0..remainder {
            let idx = tail + i;
            let normed = (input[idx] - mean) * inv_std;
            output[idx] = gamma[idx] * normed + beta[idx];
        }
    }
}

/// RMSNorm: `output = gamma * (input / rms)` where `rms = sqrt(mean(input²) + eps)`.
///
/// Used by LLaMA-style transformer models.
///
/// # Panics
///
/// Panics if slice lengths are inconsistent.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All slices must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rms_norm_f32(input: &[f32], gamma: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        let mean_sq = neon_sum_of_squares(input) / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let chunks = n / LANES;
        let remainder = n % LANES;

        let inv_v = vdupq_n_f32(inv_rms);
        let inp = input.as_ptr();
        let gp = gamma.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let x = vld1q_f32(inp.add(off));
            let g = vld1q_f32(gp.add(off));
            let normed = vmulq_f32(x, inv_v);
            let result = vmulq_f32(g, normed);
            vst1q_f32(op.add(off), result);
        }

        let tail = chunks * LANES;
        for i in 0..remainder {
            let idx = tail + i;
            output[idx] = gamma[idx] * (input[idx] * inv_rms);
        }
    }
}

/// GroupNorm: normalizes over groups of channels.
///
/// `input` is treated as `[num_groups, group_size]` where `group_size = n / num_groups`.
/// Each group is independently normalized and then scaled/shifted.
///
/// # Panics
///
/// Panics if `input.len()` is not divisible by `num_groups` or slice lengths mismatch.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All slices must have equal lengths and `input.len()` must be divisible by `num_groups`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_group_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    num_groups: usize,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    assert!(num_groups > 0, "num_groups must be > 0");
    assert_eq!(n % num_groups, 0, "input length must be divisible by num_groups");

    if n == 0 {
        return;
    }

    let group_size = n / num_groups;

    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group = &input[start..end];

        unsafe {
            // Pass 1: mean of this group
            let mean = neon_sum(group) / group_size as f32;

            // Pass 2: variance + normalize + scale/shift
            let var = neon_sum_sq_dev(group, mean) / group_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            let chunks = group_size / LANES;
            let remainder = group_size % LANES;

            let mean_v = vdupq_n_f32(mean);
            let inv_v = vdupq_n_f32(inv_std);
            let inp = group.as_ptr();
            let gp = gamma[start..end].as_ptr();
            let bp = beta[start..end].as_ptr();
            let op = output[start..end].as_mut_ptr();

            for i in 0..chunks {
                let off = i * LANES;
                let x = vld1q_f32(inp.add(off));
                let gc = vld1q_f32(gp.add(off));
                let bc = vld1q_f32(bp.add(off));
                let centered = vsubq_f32(x, mean_v);
                let normed = vmulq_f32(centered, inv_v);
                let result = vfmaq_f32(bc, gc, normed);
                vst1q_f32(op.add(off), result);
            }

            let tail = chunks * LANES;
            for i in 0..remainder {
                let idx = tail + i;
                let normed = (group[idx] - mean) * inv_std;
                output[start + idx] = gamma[start + idx] * normed + beta[start + idx];
            }
        }
    }
}

/// Backward pass for LayerNorm.
///
/// Given the upstream gradient `grad_output`, the original `input`, learned
/// `gamma`, and pre-computed `mean` / `inv_std`, fills:
/// - `grad_input`  — gradient w.r.t. input
/// - `grad_gamma`  — gradient w.r.t. gamma (accumulated)
/// - `grad_beta`   — gradient w.r.t. beta  (accumulated)
///
/// # Panics
///
/// Panics if any slice length differs from `input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All slices must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_backward_f32(
    grad_output: &[f32],
    input: &[f32],
    gamma: &[f32],
    mean: f32,
    inv_std: f32,
    grad_input: &mut [f32],
    grad_gamma: &mut [f32],
    grad_beta: &mut [f32],
) {
    let n = input.len();
    assert_eq!(grad_output.len(), n, "grad_output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(grad_input.len(), n, "grad_input length mismatch");
    assert_eq!(grad_gamma.len(), n, "grad_gamma length mismatch");
    assert_eq!(grad_beta.len(), n, "grad_beta length mismatch");

    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / LANES;
        let remainder = n % LANES;

        // Accumulate grad_gamma and grad_beta, and compute intermediate sums
        // for the input gradient.
        //
        // x_hat[i] = (input[i] - mean) * inv_std
        // grad_gamma[i] += grad_output[i] * x_hat[i]
        // grad_beta[i]  += grad_output[i]
        //
        // For grad_input we need two global sums:
        //   sum1 = sum(grad_output[i] * gamma[i])
        //   sum2 = sum(grad_output[i] * gamma[i] * x_hat[i])

        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        let mut sum1_v = vdupq_n_f32(0.0);
        let mut sum2_v = vdupq_n_f32(0.0);

        let go_ptr = grad_output.as_ptr();
        let in_ptr = input.as_ptr();
        let gam_ptr = gamma.as_ptr();
        let gg_ptr = grad_gamma.as_mut_ptr();
        let gb_ptr = grad_beta.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let go = vld1q_f32(go_ptr.add(off));
            let x = vld1q_f32(in_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));

            let x_hat = vmulq_f32(vsubq_f32(x, mean_v), inv_v);

            // grad_gamma += go * x_hat
            let prev_gg = vld1q_f32(gg_ptr.add(off));
            vst1q_f32(gg_ptr.add(off), vfmaq_f32(prev_gg, go, x_hat));

            // grad_beta += go
            let prev_gb = vld1q_f32(gb_ptr.add(off));
            vst1q_f32(gb_ptr.add(off), vaddq_f32(prev_gb, go));

            // Accumulate sums for grad_input
            let go_g = vmulq_f32(go, g);
            sum1_v = vaddq_f32(sum1_v, go_g);
            sum2_v = vfmaq_f32(sum2_v, go_g, x_hat);
        }

        let mut sum1: f32 = vaddvq_f32(sum1_v);
        let mut sum2: f32 = vaddvq_f32(sum2_v);

        // Scalar tail for accumulation
        let tail = chunks * LANES;
        for i in 0..remainder {
            let idx = tail + i;
            let x_hat = (input[idx] - mean) * inv_std;
            grad_gamma[idx] += grad_output[idx] * x_hat;
            grad_beta[idx] += grad_output[idx];
            let go_g = grad_output[idx] * gamma[idx];
            sum1 += go_g;
            sum2 += go_g * x_hat;
        }

        // grad_input[i] = inv_std * (gamma[i] * go[i] - (sum1 + x_hat[i] * sum2) / n)
        let n_f = n as f32;
        let sum1_n = sum1 / n_f;
        let sum2_n = sum2 / n_f;

        let sum1_n_v = vdupq_n_f32(sum1_n);
        let sum2_n_v = vdupq_n_f32(sum2_n);
        let gi_ptr = grad_input.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let go = vld1q_f32(go_ptr.add(off));
            let x = vld1q_f32(in_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));

            let x_hat = vmulq_f32(vsubq_f32(x, mean_v), inv_v);
            let go_g = vmulq_f32(go, g);
            let correction = vfmaq_f32(sum1_n_v, x_hat, sum2_n_v);
            let result = vmulq_f32(vsubq_f32(go_g, correction), inv_v);
            vst1q_f32(gi_ptr.add(off), result);
        }

        for i in 0..remainder {
            let idx = tail + i;
            let x_hat = (input[idx] - mean) * inv_std;
            let go_g = grad_output[idx] * gamma[idx];
            grad_input[idx] = inv_std * (go_g - sum1_n - x_hat * sum2_n);
        }
    }
}

/// Fused LayerNorm + residual addition.
///
/// Computes `output = LayerNorm(input + residual, gamma, beta, eps)` in a
/// single pass over the combined input, avoiding an extra read/write for
/// the residual addition.
///
/// # Panics
///
/// Panics if any slice length differs from `input.len()`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All slices must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_layer_norm_residual_f32(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
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
        // Pass 1: compute mean of (input + residual) using NEON
        let chunks = n / LANES;
        let remainder = n % LANES;
        let inp = input.as_ptr();
        let res = residual.as_ptr();

        let mut sum_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let a = vld1q_f32(inp.add(off));
            let b = vld1q_f32(res.add(off));
            sum_v = vaddq_f32(sum_v, vaddq_f32(a, b));
        }
        let mut total: f32 = vaddvq_f32(sum_v);
        let tail = chunks * LANES;
        for i in 0..remainder {
            total += input[tail + i] + residual[tail + i];
        }
        let mean = total / n as f32;

        // Pass 2: variance of (input + residual)
        let mean_v = vdupq_n_f32(mean);
        let mut var_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let a = vld1q_f32(inp.add(off));
            let b = vld1q_f32(res.add(off));
            let combined = vaddq_f32(a, b);
            let diff = vsubq_f32(combined, mean_v);
            var_v = vfmaq_f32(var_v, diff, diff);
        }
        let mut var_sum: f32 = vaddvq_f32(var_v);
        for i in 0..remainder {
            let d = (input[tail + i] + residual[tail + i]) - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

        // Normalize + scale/shift
        let inv_v = vdupq_n_f32(inv_std);
        let gp = gamma.as_ptr();
        let bp = beta.as_ptr();
        let op = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let a = vld1q_f32(inp.add(off));
            let b = vld1q_f32(res.add(off));
            let combined = vaddq_f32(a, b);
            let g = vld1q_f32(gp.add(off));
            let bt = vld1q_f32(bp.add(off));
            let centered = vsubq_f32(combined, mean_v);
            let normed = vmulq_f32(centered, inv_v);
            let result = vfmaq_f32(bt, g, normed);
            vst1q_f32(op.add(off), result);
        }

        for i in 0..remainder {
            let idx = tail + i;
            let combined = input[idx] + residual[idx];
            let normed = (combined - mean) * inv_std;
            output[idx] = gamma[idx] * normed + beta[idx];
        }
    }
}

// ── Scalar references (test-only) ──────────────────────────────────────

#[cfg(test)]
fn naive_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * (x - mean) * inv_std + beta[i]).collect()
}

#[cfg(test)]
fn naive_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * x * inv_rms).collect()
}

#[cfg(test)]
fn naive_group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_groups: usize,
    eps: f32,
) -> Vec<f32> {
    let n = input.len();
    let gs = n / num_groups;
    let mut out = vec![0.0f32; n];
    for g in 0..num_groups {
        let start = g * gs;
        let group = &input[start..start + gs];
        let mean: f32 = group.iter().sum::<f32>() / gs as f32;
        let var: f32 = group.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / gs as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..gs {
            let idx = start + j;
            out[idx] = gamma[idx] * (input[idx] - mean) * inv_std + beta[idx];
        }
    }
    out
}

#[cfg(test)]
fn naive_fused_ln_residual(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let combined: Vec<f32> = input.iter().zip(residual).map(|(a, b)| a + b).collect();
    naive_layer_norm(&combined, gamma, beta, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    const EPS: f32 = 1e-5;
    #[cfg(target_arch = "aarch64")]
    const TOL: f32 = 1e-5;
    #[cfg(target_arch = "aarch64")]
    const TOL_LARGE: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn make_input(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect()
    }

    fn make_gamma(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect()
    }

    fn make_beta(n: usize) -> Vec<f32> {
        (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.15).collect()
    }

    // ═══════════════════════════════════════════════════════════════════
    // LayerNorm tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_norm_dim_1() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 1];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_16() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let beta = make_beta(128);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 128];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_1024() {
        let input = make_input(1024);
        let gamma = vec![1.0; 1024];
        let beta = vec![0.0; 1024];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 1024];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_layer_norm_non_aligned_5() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 5];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_non_aligned_13() {
        let input = make_input(13);
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 13];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_non_aligned_137() {
        let input = make_input(137);
        let gamma = make_gamma(137);
        let beta = make_beta(137);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 137];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_zero_variance() {
        let input = vec![3.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_identity_gamma_zero_beta() {
        let input = make_input(32);
        let gamma = vec![1.0; 32];
        let beta = vec![0.0; 32];
        let mut output = vec![0.0; 32];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 1e-3, "normalized output should have ~0 mean, got {sum}");
    }

    #[test]
    fn test_layer_norm_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_layer_norm_small_eps() {
        let input = vec![1e-7, 2e-7, 3e-7, 4e-7];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = naive_layer_norm(&input, &gamma, &beta, 1e-12);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, 1e-12) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_large_values() {
        let input = vec![1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let input = vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_affine_scaling() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![1.0; 4];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    // ═══════════════════════════════════════════════════════════════════
    // RMSNorm tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_rms_norm_dim_1() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 1];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let gamma = vec![1.0; 8];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_16() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 128];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_1024() {
        let input = make_input(1024);
        let gamma = vec![1.0; 1024];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 1024];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_rms_norm_non_aligned_5() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 5];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_non_aligned_13() {
        let input = make_input(13);
        let gamma = vec![1.0; 13];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 13];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_non_aligned_137() {
        let input = make_input(137);
        let gamma = make_gamma(137);
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 137];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_rms_norm_small_eps() {
        let input = vec![1e-7, 2e-7, 3e-7, 4e-7];
        let gamma = vec![1.0; 4];
        let expected = naive_rms_norm(&input, &gamma, 1e-12);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, 1e-12) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_values() {
        let input = vec![1e6, 2e6, 3e6, 4e6];
        let gamma = vec![1.0; 4];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 0.5, 3.0, 0.1];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_equivalence_unit_gamma_zero_mean() {
        // When mean is zero, LayerNorm ≈ RMSNorm (with gamma=1, beta=0).
        // More precisely: for zero-mean input, var == mean_sq, so they coincide.
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 1.0];
        let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        // Create zero-mean input
        let zm: Vec<f32> = input.iter().map(|&x| x - mean).collect();
        let gamma = vec![1.0; zm.len()];
        let beta = vec![0.0; zm.len()];
        let ln = naive_layer_norm(&zm, &gamma, &beta, EPS);
        let rms = naive_rms_norm(&zm, &gamma, EPS);
        approx_eq(&ln, &rms, 1e-4);
    }

    // ═══════════════════════════════════════════════════════════════════
    // GroupNorm tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_group_norm_1_group() {
        // 1 group == LayerNorm
        let input = make_input(8);
        let gamma = make_gamma(8);
        let beta = make_beta(8);
        let expected = naive_group_norm(&input, &gamma, &beta, 1, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 1, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_equals_layer_norm_single_group() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let ln = naive_layer_norm(&input, &gamma, &beta, EPS);
        let gn = naive_group_norm(&input, &gamma, &beta, 1, EPS);
        approx_eq(&ln, &gn, TOL);
        let mut output = vec![0.0; 16];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 1, EPS) };
        approx_eq(&output, &ln, TOL);
    }

    #[test]
    fn test_group_norm_2_groups() {
        let input = make_input(8);
        let gamma = make_gamma(8);
        let beta = make_beta(8);
        let expected = naive_group_norm(&input, &gamma, &beta, 2, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 2, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_4_groups() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let expected = naive_group_norm(&input, &gamma, &beta, 4, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 4, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_8_groups_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let beta = make_beta(128);
        let expected = naive_group_norm(&input, &gamma, &beta, 8, EPS);
        let mut output = vec![0.0; 128];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 8, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_16_groups_dim_1024() {
        let input = make_input(1024);
        let gamma = make_gamma(1024);
        let beta = make_beta(1024);
        let expected = naive_group_norm(&input, &gamma, &beta, 16, EPS);
        let mut output = vec![0.0; 1024];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 16, EPS) };
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_group_norm_n_groups_equals_n() {
        // Each element is its own group → variance=0, output ≈ beta
        let input = make_input(4);
        let gamma = vec![1.0; 4];
        let beta = vec![0.5; 4];
        let mut output = vec![0.0; 4];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 4, EPS) };
        for &v in &output {
            assert!((v - 0.5).abs() < TOL, "single-element groups should yield ~beta");
        }
    }

    #[test]
    fn test_group_norm_non_aligned_groups() {
        // 15 elements, 3 groups of 5 (non NEON-aligned group size)
        let input = make_input(15);
        let gamma = make_gamma(15);
        let beta = make_beta(15);
        let expected = naive_group_norm(&input, &gamma, &beta, 3, EPS);
        let mut output = vec![0.0; 15];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 3, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 1, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_group_norm_32_groups_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let beta = make_beta(128);
        let expected = naive_group_norm(&input, &gamma, &beta, 32, EPS);
        let mut output = vec![0.0; 128];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 32, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_small_eps() {
        let input = vec![1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_group_norm(&input, &gamma, &beta, 2, 1e-12);
        let mut output = vec![0.0; 8];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 2, 1e-12) };
        approx_eq(&output, &expected, TOL);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Backward pass tests
    // ═══════════════════════════════════════════════════════════════════

    fn compute_mean_inv_std(input: &[f32], eps: f32) -> (f32, f32) {
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        (mean, 1.0 / (var + eps).sqrt())
    }

    fn naive_backward(
        grad_output: &[f32],
        input: &[f32],
        gamma: &[f32],
        mean: f32,
        inv_std: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = input.len();
        let n_f = n as f32;
        let x_hat: Vec<f32> = input.iter().map(|&x| (x - mean) * inv_std).collect();
        let grad_gamma: Vec<f32> =
            grad_output.iter().zip(&x_hat).map(|(&go, &xh)| go * xh).collect();
        let grad_beta: Vec<f32> = grad_output.to_vec();

        let sum1: f32 = grad_output.iter().zip(gamma).map(|(&go, &g)| go * g).sum();
        let sum2: f32 =
            grad_output.iter().zip(gamma).zip(&x_hat).map(|((&go, &g), &xh)| go * g * xh).sum();

        let grad_input: Vec<f32> = (0..n)
            .map(|i| inv_std * (grad_output[i] * gamma[i] - (sum1 + x_hat[i] * sum2) / n_f))
            .collect();

        (grad_input, grad_gamma, grad_beta)
    }

    #[test]
    fn test_backward_dim_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let grad_output = vec![0.1, 0.2, 0.3, 0.4];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 4];
        let mut grad_gamma = vec![0.0; 4];
        let mut grad_beta = vec![0.0; 4];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_dim_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
        let gamma = make_gamma(8);
        let grad_output: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 8];
        let mut grad_gamma = vec![0.0; 8];
        let mut grad_beta = vec![0.0; 8];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_dim_16() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let grad_output: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.05).collect();
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 16];
        let mut grad_gamma = vec![0.0; 16];
        let mut grad_beta = vec![0.0; 16];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let grad_output: Vec<f32> = (0..128).map(|i| ((i * 3 + 7) % 50) as f32 * 0.01).collect();
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 128];
        let mut grad_gamma = vec![0.0; 128];
        let mut grad_beta = vec![0.0; 128];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_non_aligned_5() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 2.0];
        let gamma = vec![1.0, 2.0, 0.5, 1.5, 0.8];
        let grad_output = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 5];
        let mut grad_gamma = vec![0.0; 5];
        let mut grad_beta = vec![0.0; 5];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_non_aligned_137() {
        let input = make_input(137);
        let gamma = make_gamma(137);
        let grad_output: Vec<f32> = (0..137).map(|i| ((i * 11 + 5) % 40) as f32 * 0.02).collect();
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 137];
        let mut grad_gamma = vec![0.0; 137];
        let mut grad_beta = vec![0.0; 137];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_accumulates_grad_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let grad_output = vec![0.1, 0.2, 0.3, 0.4];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);

        // Pre-fill grad_gamma to verify accumulation
        let mut grad_gamma = vec![1.0; 4];
        let mut grad_beta = vec![0.0; 4];
        let mut grad_input = vec![0.0; 4];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        // grad_gamma should be 1.0 + the computed gradient
        let (_, exp_gg, _) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);
        let expected: Vec<f32> = exp_gg.iter().map(|&g| g + 1.0).collect();
        approx_eq(&grad_gamma, &expected, TOL);
    }

    #[test]
    fn test_backward_accumulates_grad_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let grad_output = vec![0.1, 0.2, 0.3, 0.4];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);

        let mut grad_gamma = vec![0.0; 4];
        let mut grad_beta = vec![2.0; 4];
        let mut grad_input = vec![0.0; 4];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        let expected: Vec<f32> = grad_output.iter().map(|&g| g + 2.0).collect();
        approx_eq(&grad_beta, &expected, TOL);
    }

    #[test]
    fn test_backward_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let grad_output: Vec<f32> = vec![];
        let mut grad_input: Vec<f32> = vec![];
        let mut grad_gamma: Vec<f32> = vec![];
        let mut grad_beta: Vec<f32> = vec![];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                0.0,
                1.0,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
    }

    #[test]
    fn test_backward_dim_1() {
        let input = vec![5.0];
        let gamma = vec![2.0];
        let grad_output = vec![1.0];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 1];
        let mut grad_gamma = vec![0.0; 1];
        let mut grad_beta = vec![0.0; 1];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }

    #[test]
    fn test_backward_dim_1024() {
        let input = make_input(1024);
        let gamma = make_gamma(1024);
        let grad_output: Vec<f32> = (0..1024).map(|i| ((i * 13 + 1) % 70) as f32 * 0.01).collect();
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 1024];
        let mut grad_gamma = vec![0.0; 1024];
        let mut grad_beta = vec![0.0; 1024];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL_LARGE);
        approx_eq(&grad_gamma, &exp_gg, TOL_LARGE);
        approx_eq(&grad_beta, &exp_gb, TOL_LARGE);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Fused LayerNorm + residual tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fused_residual_dim_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.1, -0.1, 0.2, -0.2];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_dim_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let residual: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.1).collect();
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_dim_16() {
        let input = make_input(16);
        let residual: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3) - 2.0).collect();
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 16];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_dim_128() {
        let input = make_input(128);
        let residual: Vec<f32> = (0..128).map(|i| ((i * 3 + 1) % 20) as f32 * 0.1).collect();
        let gamma = make_gamma(128);
        let beta = make_beta(128);
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 128];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_dim_1024() {
        let input = make_input(1024);
        let residual: Vec<f32> = (0..1024).map(|i| ((i * 7) % 30) as f32 * 0.05).collect();
        let gamma = vec![1.0; 1024];
        let beta = vec![0.0; 1024];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 1024];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_fused_residual_non_aligned_5() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![0.5, -0.5, 0.3, -0.3, 0.1];
        let gamma = vec![1.0, 2.0, 0.5, 1.5, 0.8];
        let beta = vec![0.0, 0.1, -0.1, 0.2, -0.2];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 5];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_non_aligned_13() {
        let input = make_input(13);
        let residual: Vec<f32> = (0..13).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 13];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_non_aligned_137() {
        let input = make_input(137);
        let residual: Vec<f32> = (0..137).map(|i| ((i * 5 + 2) % 30) as f32 * 0.07).collect();
        let gamma = make_gamma(137);
        let beta = make_beta(137);
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 137];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_zero_residual() {
        // Zero residual should equal plain LayerNorm
        let input = make_input(16);
        let residual = vec![0.0; 16];
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 16];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_empty() {
        let input: Vec<f32> = vec![];
        let residual: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        assert!(output.is_empty());
    }

    #[test]
    fn test_fused_residual_dim_1() {
        let input = vec![3.0];
        let residual = vec![2.0];
        let gamma = vec![1.5];
        let beta = vec![0.5];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 1];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_equivalence_to_separate_ops() {
        // Verify fused == separate add + layernorm
        let input = make_input(64);
        let residual: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let gamma = make_gamma(64);
        let beta = make_beta(64);

        // Separate ops
        let combined: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected = naive_layer_norm(&combined, &gamma, &beta, EPS);

        // Fused
        let mut output = vec![0.0; 64];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Numerical stability tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_norm_stability_tiny_eps() {
        let input = vec![0.0001, 0.0002, 0.0003, 0.0004];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = naive_layer_norm(&input, &gamma, &beta, 1e-12);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, 1e-12) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_stability_tiny_eps() {
        let input = vec![0.0001, 0.0002, 0.0003, 0.0004];
        let gamma = vec![1.0; 4];
        let expected = naive_rms_norm(&input, &gamma, 1e-12);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, 1e-12) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_stability_mixed_magnitudes() {
        let input = vec![1e-6, 1e6, -1e-6, -1e6, 0.5, -0.5, 100.0, -100.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, 1e-3);
    }

    #[test]
    fn test_backward_stability_uniform_grad() {
        // Uniform grad_output → grad_input should be ~0 (since we subtract mean contribution)
        let input = make_input(8);
        let gamma = vec![1.0; 8];
        let grad_output = vec![1.0; 8];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);

        let mut grad_input = vec![0.0; 8];
        let mut grad_gamma = vec![0.0; 8];
        let mut grad_beta = vec![0.0; 8];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        let (exp_gi, _, _) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);
        approx_eq(&grad_input, &exp_gi, TOL);
    }

    #[test]
    fn test_fused_residual_stability_large_residual() {
        let input = vec![1.0; 8];
        let residual = vec![1e6; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut output = vec![0.0; 8];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        // All same value → output ≈ 0
        for &v in &output {
            assert!(v.abs() < TOL, "constant combined should yield ~0, got {v}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Cross-function consistency tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_group_norm_1_group_matches_layer_norm() {
        for &n in &[4, 8, 16, 128] {
            let input = make_input(n);
            let gamma = make_gamma(n);
            let beta = make_beta(n);

            let mut ln_out = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut ln_out, EPS) };

            let mut gn_out = vec![0.0; n];
            unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut gn_out, 1, EPS) };

            approx_eq(&ln_out, &gn_out, TOL);
        }
    }

    #[test]
    fn test_layer_norm_output_zero_mean() {
        // With gamma=1, beta=0, output should have approximately zero mean.
        for &n in &[4, 8, 16, 128, 1024] {
            let input = make_input(n);
            let gamma = vec![1.0; n];
            let beta = vec![0.0; n];
            let mut output = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
            let out_mean: f32 = output.iter().sum::<f32>() / n as f32;
            assert!(out_mean.abs() < 1e-3, "n={n}: output mean = {out_mean}");
        }
    }

    #[test]
    fn test_layer_norm_output_unit_variance() {
        // With gamma=1, beta=0, output should have approximately unit variance.
        for &n in &[8, 16, 128, 1024] {
            let input = make_input(n);
            let gamma = vec![1.0; n];
            let beta = vec![0.0; n];
            let mut output = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
            let out_mean: f32 = output.iter().sum::<f32>() / n as f32;
            let out_var: f32 =
                output.iter().map(|x| (x - out_mean).powi(2)).sum::<f32>() / n as f32;
            assert!(
                (out_var - 1.0).abs() < 0.01,
                "n={n}: output variance = {out_var}, expected ~1.0"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Additional dimension and edge-case tests (to reach 90+)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_norm_dim_2() {
        let input = vec![10.0, -10.0];
        let gamma = vec![1.0; 2];
        let beta = vec![0.0; 2];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 2];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_3() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 3];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_7() {
        let input = make_input(7);
        let gamma = make_gamma(7);
        let beta = make_beta(7);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 7];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_33() {
        let input = make_input(33);
        let gamma = make_gamma(33);
        let beta = make_beta(33);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 33];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_dim_64() {
        let input = make_input(64);
        let gamma = make_gamma(64);
        let beta = make_beta(64);
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 64];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_2() {
        let input = vec![3.0, -3.0];
        let gamma = vec![1.0; 2];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 2];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_3() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 3];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_33() {
        let input = make_input(33);
        let gamma = make_gamma(33);
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 33];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_dim_64() {
        let input = make_input(64);
        let gamma = make_gamma(64);
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 64];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let input = vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let gamma = vec![1.0; 8];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_2_groups_dim_16() {
        let input = make_input(16);
        let gamma = make_gamma(16);
        let beta = make_beta(16);
        let expected = naive_group_norm(&input, &gamma, &beta, 2, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 2, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_4_groups_dim_128() {
        let input = make_input(128);
        let gamma = make_gamma(128);
        let beta = make_beta(128);
        let expected = naive_group_norm(&input, &gamma, &beta, 4, EPS);
        let mut output = vec![0.0; 128];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 4, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_64_groups_dim_1024() {
        let input = make_input(1024);
        let gamma = make_gamma(1024);
        let beta = make_beta(1024);
        let expected = naive_group_norm(&input, &gamma, &beta, 64, EPS);
        let mut output = vec![0.0; 1024];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 64, EPS) };
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_backward_zero_grad_output() {
        let input = make_input(8);
        let gamma = make_gamma(8);
        let grad_output = vec![0.0; 8];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);

        let mut grad_input = vec![0.0; 8];
        let mut grad_gamma = vec![0.0; 8];
        let mut grad_beta = vec![0.0; 8];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        for &v in &grad_input {
            assert!(v.abs() < TOL, "zero grad_output should yield zero grad_input");
        }
        for &v in &grad_gamma {
            assert!(v.abs() < TOL, "zero grad_output should yield zero grad_gamma");
        }
        for &v in &grad_beta {
            assert!(v.abs() < TOL, "zero grad_output should yield zero grad_beta");
        }
    }

    #[test]
    fn test_fused_residual_negative_residual() {
        let input = vec![5.0, 6.0, 7.0, 8.0];
        let residual = vec![-5.0, -6.0, -7.0, -8.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_fused_residual_large_dim_with_affine() {
        let input = make_input(256);
        let residual: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.02).collect();
        let gamma = make_gamma(256);
        let beta = make_beta(256);
        let expected = naive_fused_ln_residual(&input, &residual, &gamma, &beta, EPS);
        let mut output = vec![0.0; 256];
        unsafe {
            neon_fused_layer_norm_residual_f32(&input, &residual, &gamma, &beta, &mut output, EPS);
        }
        approx_eq(&output, &expected, TOL_LARGE);
    }

    #[test]
    fn test_layer_norm_all_negative() {
        let input = vec![-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = naive_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_all_ones() {
        let input = vec![1.0; 16];
        let gamma = vec![1.0; 16];
        let expected = naive_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_rms_norm_f32(&input, &gamma, &mut output, EPS) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_large_eps() {
        let input = make_input(8);
        let gamma = make_gamma(8);
        let beta = make_beta(8);
        let expected = naive_group_norm(&input, &gamma, &beta, 2, 1.0);
        let mut output = vec![0.0; 8];
        unsafe { neon_group_norm_f32(&input, &gamma, &beta, &mut output, 2, 1.0) };
        approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_backward_large_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![100.0; 4];
        let grad_output = vec![0.01, 0.02, 0.03, 0.04];
        let (mean, inv_std) = compute_mean_inv_std(&input, EPS);
        let (exp_gi, exp_gg, exp_gb) = naive_backward(&grad_output, &input, &gamma, mean, inv_std);

        let mut grad_input = vec![0.0; 4];
        let mut grad_gamma = vec![0.0; 4];
        let mut grad_beta = vec![0.0; 4];
        unsafe {
            neon_layer_norm_backward_f32(
                &grad_output,
                &input,
                &gamma,
                mean,
                inv_std,
                &mut grad_input,
                &mut grad_gamma,
                &mut grad_beta,
            );
        }
        approx_eq(&grad_input, &exp_gi, TOL);
        approx_eq(&grad_gamma, &exp_gg, TOL);
        approx_eq(&grad_beta, &exp_gb, TOL);
    }
}
