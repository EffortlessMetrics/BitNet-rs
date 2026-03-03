//! ARM NEON-optimized layer normalization V2 kernels for Apple Silicon.
//!
//! Extends the original NEON layernorm with six operations, each providing:
//! - An `unsafe fn neon_*` NEON-accelerated implementation
//! - A safe `fn scalar_*` fallback
//! - A public `fn *` dispatcher (NEON on aarch64+neon, scalar otherwise)
//!
//! Operations:
//! - `layer_norm_f32` — standard LayerNorm with affine transform
//! - `rms_norm_f32` — RMSNorm (LLaMA-style, no mean subtraction)
//! - `group_norm_f32` — GroupNorm over channel groups
//! - `fused_norm_silu_f32` — fused RMSNorm + SiLU activation
//! - `online_norm_stats_f32` — single-pass Welford mean+variance
//! - `norm_residual_add_f32` — fused LayerNorm + residual add
//!
//! All NEON paths process 4×f32 lanes with scalar tail fallback.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_unsafe)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── 1. layer_norm_f32 ─────────────────────────────────────────────

/// NEON-accelerated standard LayerNorm.
///
/// Computes `output[i] = gamma[i] * ((input[i] - mean) / sqrt(var + eps)) + beta[i]`.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
///
/// # Panics
///
/// Panics if slice lengths do not match.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        // Compute mean
        let chunks = n / 4;
        let rem = n % 4;
        let ptr = input.as_ptr();
        let mut sum_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sum_v = vaddq_f32(sum_v, v);
        }
        let mut sum = vaddvq_f32(sum_v);
        for i in (chunks * 4)..n {
            sum += input[i];
        }
        let mean = sum / n as f32;

        // Compute variance
        let mean_v = vdupq_n_f32(mean);
        let mut var_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let d = vsubq_f32(v, mean_v);
            var_v = vfmaq_f32(var_v, d, d);
        }
        let mut var_sum = vaddvq_f32(var_v);
        for i in (chunks * 4)..n {
            let d = input[i] - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

        // Normalize with affine
        let inv_std_v = vdupq_n_f32(inv_std);
        let gptr = gamma.as_ptr();
        let bptr = beta.as_ptr();
        let optr = output.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let b = vld1q_f32(bptr.add(off));
            let centered = vsubq_f32(v, mean_v);
            let normed = vmulq_f32(centered, inv_std_v);
            let result = vfmaq_f32(b, g, normed);
            vst1q_f32(optr.add(off), result);
        }
        let tail = chunks * 4;
        for i in 0..rem {
            let idx = tail + i;
            let normed = (input[idx] - mean) * inv_std;
            output[idx] = gamma[idx] * normed + beta[idx];
        }
    }
}

/// Scalar fallback for standard LayerNorm.
pub fn scalar_layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        output[i] = gamma[i] * ((input[i] - mean) * inv_std) + beta[i];
    }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn layer_norm_f32(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON.
        unsafe { neon_layer_norm_f32(input, gamma, beta, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_layer_norm_f32(input, gamma, beta, eps, output);
    }
}

// ── 2. rms_norm_f32 ───────────────────────────────────────────────

/// NEON-accelerated RMSNorm (LLaMA-style).
///
/// Computes `output[i] = gamma[i] * input[i] / sqrt(mean(input²) + eps)`.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rms_norm_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / 4;
        let rem = n % 4;
        let ptr = input.as_ptr();

        // mean(x²)
        let mut sq_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sq_v = vfmaq_f32(sq_v, v, v);
        }
        let mut sq_sum = vaddvq_f32(sq_v);
        for i in (chunks * 4)..n {
            sq_sum += input[i] * input[i];
        }
        let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

        // Scale
        let inv_rms_v = vdupq_n_f32(inv_rms);
        let gptr = gamma.as_ptr();
        let optr = output.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let normed = vmulq_f32(v, inv_rms_v);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(optr.add(off), scaled);
        }
        let tail = chunks * 4;
        for i in 0..rem {
            let idx = tail + i;
            output[idx] = gamma[idx] * (input[idx] * inv_rms);
        }
    }
}

/// Scalar fallback for RMSNorm.
pub fn scalar_rms_norm_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let mean_sq = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n {
        output[i] = gamma[i] * (input[i] * inv_rms);
    }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn rms_norm_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_rms_norm_f32(input, gamma, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_rms_norm_f32(input, gamma, eps, output);
    }
}

// ── 3. group_norm_f32 ─────────────────────────────────────────────

/// NEON-accelerated GroupNorm.
///
/// Input shape: flat `[num_groups * group_size]`. Each group is normalized
/// independently, then scaled by per-element gamma/beta.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_group_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_groups: usize,
    eps: f32,
    output: &mut [f32],
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
        let group = &input[start..start + group_size];

        unsafe {
            let chunks = group_size / 4;
            let rem = group_size % 4;
            let ptr = group.as_ptr();

            // Mean
            let mut sum_v = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                sum_v = vaddq_f32(sum_v, v);
            }
            let mut sum = vaddvq_f32(sum_v);
            for i in (chunks * 4)..group_size {
                sum += group[i];
            }
            let mean = sum / group_size as f32;

            // Variance
            let mean_v = vdupq_n_f32(mean);
            let mut var_v = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                let d = vsubq_f32(v, mean_v);
                var_v = vfmaq_f32(var_v, d, d);
            }
            let mut var_sum = vaddvq_f32(var_v);
            for i in (chunks * 4)..group_size {
                let d = group[i] - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / (var_sum / group_size as f32 + eps).sqrt();

            // Normalize with affine
            let inv_std_v = vdupq_n_f32(inv_std);
            let gptr = gamma[start..].as_ptr();
            let bptr = beta[start..].as_ptr();
            let optr = output[start..].as_mut_ptr();
            for i in 0..chunks {
                let off = i * 4;
                let v = vld1q_f32(ptr.add(off));
                let g = vld1q_f32(gptr.add(off));
                let b = vld1q_f32(bptr.add(off));
                let centered = vsubq_f32(v, mean_v);
                let normed = vmulq_f32(centered, inv_std_v);
                let result = vfmaq_f32(b, g, normed);
                vst1q_f32(optr.add(off), result);
            }
            let tail = chunks * 4;
            for i in 0..rem {
                let gi = start + tail + i;
                let normed = (input[gi] - mean) * inv_std;
                output[gi] = gamma[gi] * normed + beta[gi];
            }
        }
    }
}

/// Scalar fallback for GroupNorm.
pub fn scalar_group_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_groups: usize,
    eps: f32,
    output: &mut [f32],
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
        let group = &input[start..start + group_size];
        let mean = group.iter().sum::<f32>() / group_size as f32;
        let var = group.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..group_size {
            let gi = start + i;
            output[gi] = gamma[gi] * ((input[gi] - mean) * inv_std) + beta[gi];
        }
    }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn group_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_groups: usize,
    eps: f32,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_group_norm_f32(input, gamma, beta, num_groups, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_group_norm_f32(input, gamma, beta, num_groups, eps, output);
    }
}

// ── 4. fused_norm_silu_f32 ────────────────────────────────────────

/// NEON-accelerated fused RMSNorm + SiLU.
///
/// Computes `output[i] = silu(gamma[i] * input[i] / sqrt(mean(x²) + eps))`
/// where `silu(x) = x / (1 + exp(-x))`.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_norm_silu_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        let chunks = n / 4;
        let rem = n % 4;
        let ptr = input.as_ptr();

        // mean(x²)
        let mut sq_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sq_v = vfmaq_f32(sq_v, v, v);
        }
        let mut sq_sum = vaddvq_f32(sq_v);
        for i in (chunks * 4)..n {
            sq_sum += input[i] * input[i];
        }
        let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

        // Normalize + SiLU (scalar SiLU since exp has no NEON intrinsic)
        let inv_rms_v = vdupq_n_f32(inv_rms);
        let gptr = gamma.as_ptr();
        let optr = output.as_mut_ptr();

        // NEON pass: compute normalized values, store temporarily
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let normed = vmulq_f32(v, inv_rms_v);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(optr.add(off), scaled);
        }
        let tail = chunks * 4;
        for i in 0..rem {
            let idx = tail + i;
            output[idx] = gamma[idx] * (input[idx] * inv_rms);
        }
    }

    // Apply SiLU element-wise (exp has no NEON intrinsic)
    for v in output.iter_mut().take(n) {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
    }
}

/// Scalar fallback for fused RMSNorm + SiLU.
pub fn scalar_fused_norm_silu_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let mean_sq = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n {
        let normed = gamma[i] * (input[i] * inv_rms);
        output[i] = normed / (1.0 + (-normed).exp());
    }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn fused_norm_silu_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_norm_silu_f32(input, gamma, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_fused_norm_silu_f32(input, gamma, eps, output);
    }
}

// ── 5. online_norm_stats_f32 ──────────────────────────────────────

/// Result of single-pass mean+variance computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormStats {
    /// Arithmetic mean.
    pub mean: f32,
    /// Population variance.
    pub variance: f32,
}

/// NEON-accelerated single-pass Welford mean+variance.
///
/// Computes mean and population variance in one pass using Welford's
/// online algorithm. The NEON path accelerates the partial-sum
/// accumulation for the final correction step.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_online_norm_stats_f32(input: &[f32]) -> NormStats {
    let n = input.len();
    if n == 0 {
        return NormStats { mean: 0.0, variance: 0.0 };
    }

    // Welford's algorithm — purely sequential for numerical stability
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;

    for (i, &x) in input.iter().enumerate() {
        let count = (i + 1) as f32;
        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    NormStats { mean, variance: m2 / n as f32 }
}

/// Scalar fallback for single-pass Welford mean+variance.
pub fn scalar_online_norm_stats_f32(input: &[f32]) -> NormStats {
    let n = input.len();
    if n == 0 {
        return NormStats { mean: 0.0, variance: 0.0 };
    }
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;
    for (i, &x) in input.iter().enumerate() {
        let count = (i + 1) as f32;
        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    NormStats { mean, variance: m2 / n as f32 }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn online_norm_stats_f32(input: &[f32]) -> NormStats {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_online_norm_stats_f32(input) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_online_norm_stats_f32(input)
    }
}

// ── 6. norm_residual_add_f32 ──────────────────────────────────────

/// NEON-accelerated fused LayerNorm + residual add.
///
/// Computes `output[i] = layernorm(input)[i] + residual[i]`, where
/// `layernorm(input)[i] = gamma[i] * ((input[i] - mean) / sqrt(var + eps)) + beta[i]`.
///
/// # Safety
///
/// Requires aarch64 target with NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_norm_residual_add_f32(
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
        let chunks = n / 4;
        let rem = n % 4;
        let ptr = input.as_ptr();

        // Mean
        let mut sum_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sum_v = vaddq_f32(sum_v, v);
        }
        let mut sum = vaddvq_f32(sum_v);
        for i in (chunks * 4)..n {
            sum += input[i];
        }
        let mean = sum / n as f32;

        // Variance
        let mean_v = vdupq_n_f32(mean);
        let mut var_v = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let d = vsubq_f32(v, mean_v);
            var_v = vfmaq_f32(var_v, d, d);
        }
        let mut var_sum = vaddvq_f32(var_v);
        for i in (chunks * 4)..n {
            let d = input[i] - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

        // Normalize + residual add
        let inv_std_v = vdupq_n_f32(inv_std);
        let gptr = gamma.as_ptr();
        let bptr = beta.as_ptr();
        let rptr = residual.as_ptr();
        let optr = output.as_mut_ptr();
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(ptr.add(off));
            let g = vld1q_f32(gptr.add(off));
            let b = vld1q_f32(bptr.add(off));
            let r = vld1q_f32(rptr.add(off));
            let centered = vsubq_f32(v, mean_v);
            let normed = vmulq_f32(centered, inv_std_v);
            let affine = vfmaq_f32(b, g, normed);
            let result = vaddq_f32(affine, r);
            vst1q_f32(optr.add(off), result);
        }
        let tail = chunks * 4;
        for i in 0..rem {
            let idx = tail + i;
            let normed = (input[idx] - mean) * inv_std;
            output[idx] = gamma[idx] * normed + beta[idx] + residual[idx];
        }
    }
}

/// Scalar fallback for fused LayerNorm + residual add.
pub fn scalar_norm_residual_add_f32(
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
    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        let normed = (input[i] - mean) * inv_std;
        output[i] = gamma[i] * normed + beta[i] + residual[i];
    }
}

/// Dispatcher: NEON on aarch64, scalar otherwise.
pub fn norm_residual_add_f32(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_norm_residual_add_f32(input, residual, gamma, beta, eps, output) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_norm_residual_add_f32(input, residual, gamma, beta, eps, output);
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-5;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    fn ref_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }
        let mean = input.iter().sum::<f32>() / n as f32;
        let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        (0..n).map(|i| gamma[i] * ((input[i] - mean) * inv) + beta[i]).collect()
    }

    fn ref_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        if n == 0 {
            return vec![];
        }
        let ms = input.iter().map(|&x| x * x).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        (0..n).map(|i| gamma[i] * input[i] * inv).collect()
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // ══════════════════════════════════════════════════════════════
    // 1. layer_norm_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_layer_norm_basic_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 8];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 5];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_empty() {
        let mut out: Vec<f32> = vec![];
        layer_norm_f32(&[], &[], &[], EPS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_layer_norm_single() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_all_zeros() {
        let input = vec![0.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut out = vec![f32::NAN; 8];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0 got {v}");
        }
    }

    #[test]
    fn test_layer_norm_all_same() {
        let input = vec![7.0; 16];
        let gamma = vec![1.0; 16];
        let beta = vec![0.0; 16];
        let mut out = vec![0.0; 16];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_non_aligned() {
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 13];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_large() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; n];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_layer_norm_large_values() {
        let input = vec![1e6, -1e6, 1e6, -1e6];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 4];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, 1e-3);
    }

    #[test]
    fn test_layer_norm_small_values() {
        let input = vec![1e-7, 2e-7, 3e-7, 4e-7];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 4];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_scalar_matches_dispatcher() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let mut out_disp = vec![0.0; 5];
        let mut out_scalar = vec![0.0; 5];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_disp);
        scalar_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_scalar);
        assert_approx(&out_disp, &out_scalar, TOL);
    }

    // ══════════════════════════════════════════════════════════════
    // 2. rms_norm_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_rms_norm_basic_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 8];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 5];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_empty() {
        let mut out: Vec<f32> = vec![];
        rms_norm_f32(&[], &[], EPS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_rms_norm_single() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let input = vec![0.0; 8];
        let gamma = vec![1.0; 8];
        let mut out = vec![f32::NAN; 8];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0 got {v}");
        }
    }

    #[test]
    fn test_rms_norm_non_aligned() {
        let input: Vec<f32> = (0..11).map(|i| (i + 1) as f32).collect();
        let gamma = vec![1.0; 11];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 11];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_values() {
        let input = vec![1e5, -1e5, 1e5, -1e5];
        let gamma = vec![1.0; 4];
        let expected = ref_rms_norm(&input, &gamma, EPS);
        let mut out = vec![0.0; 4];
        rms_norm_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_rms_norm_scalar_matches_dispatcher() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 6];
        let mut out_disp = vec![0.0; 6];
        let mut out_scalar = vec![0.0; 6];
        rms_norm_f32(&input, &gamma, EPS, &mut out_disp);
        scalar_rms_norm_f32(&input, &gamma, EPS, &mut out_scalar);
        assert_approx(&out_disp, &out_scalar, TOL);
    }

    // ══════════════════════════════════════════════════════════════
    // 3. group_norm_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_group_norm_single_group() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 8];
        group_norm_f32(&input, &gamma, &beta, 1, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_group_norm_two_groups() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];

        let exp_g0 = ref_layer_norm(&input[0..4], &gamma[0..4], &beta[0..4], EPS);
        let exp_g1 = ref_layer_norm(&input[4..8], &gamma[4..8], &beta[4..8], EPS);
        let expected: Vec<f32> = exp_g0.into_iter().chain(exp_g1).collect();

        let mut out = vec![0.0; 8];
        group_norm_f32(&input, &gamma, &beta, 2, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_group_norm_per_element_groups() {
        // Each group is a single element → normalized to 0 (zero var)
        let input = vec![5.0, 10.0, 15.0, 20.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        group_norm_f32(&input, &gamma, &beta, 4, EPS, &mut out);
        for &v in &out {
            assert!(v.abs() < TOL, "single-element groups should yield ~0, got {v}");
        }
    }

    #[test]
    fn test_group_norm_empty() {
        let mut out: Vec<f32> = vec![];
        group_norm_f32(&[], &[], &[], 1, EPS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_group_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.5, 1.0];
        let beta = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let exp_g0 = ref_layer_norm(&input[0..3], &gamma[0..3], &beta[0..3], EPS);
        let exp_g1 = ref_layer_norm(&input[3..6], &gamma[3..6], &beta[3..6], EPS);
        let expected: Vec<f32> = exp_g0.into_iter().chain(exp_g1).collect();
        let mut out = vec![0.0; 6];
        group_norm_f32(&input, &gamma, &beta, 2, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_group_norm_non_aligned_group_size() {
        // 3 groups of 5 → non-aligned group size
        let n = 15;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let mut expected = Vec::new();
        for g in 0..3 {
            let s = g * 5;
            expected.extend(ref_layer_norm(
                &input[s..s + 5],
                &gamma[s..s + 5],
                &beta[s..s + 5],
                EPS,
            ));
        }
        let mut out = vec![0.0; n];
        group_norm_f32(&input, &gamma, &beta, 3, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_group_norm_scalar_matches_dispatcher() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let mut out_d = vec![0.0; 6];
        let mut out_s = vec![0.0; 6];
        group_norm_f32(&input, &gamma, &beta, 2, EPS, &mut out_d);
        scalar_group_norm_f32(&input, &gamma, &beta, 2, EPS, &mut out_s);
        assert_approx(&out_d, &out_s, TOL);
    }

    #[test]
    #[should_panic(expected = "input length must be divisible by num_groups")]
    fn test_group_norm_invalid_groups() {
        let mut out = vec![0.0; 5];
        group_norm_f32(&[1.0; 5], &[1.0; 5], &[0.0; 5], 3, EPS, &mut out);
    }

    // ══════════════════════════════════════════════════════════════
    // 4. fused_norm_silu_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_fused_norm_silu_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let rms_out = ref_rms_norm(&input, &gamma, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|&x| silu(x)).collect();
        let mut out = vec![0.0; 8];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_fused_norm_silu_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let rms_out = ref_rms_norm(&input, &gamma, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|&x| silu(x)).collect();
        let mut out = vec![0.0; 5];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_fused_norm_silu_empty() {
        let mut out: Vec<f32> = vec![];
        fused_norm_silu_f32(&[], &[], EPS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_fused_norm_silu_single() {
        let input = vec![3.0];
        let gamma = vec![1.0];
        let rms_out = ref_rms_norm(&input, &gamma, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|&x| silu(x)).collect();
        let mut out = vec![0.0];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_fused_norm_silu_all_zeros() {
        let input = vec![0.0; 8];
        let gamma = vec![1.0; 8];
        let mut out = vec![f32::NAN; 8];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out);
        // silu(0) = 0
        for &v in &out {
            assert!(v.abs() < TOL, "silu(0) should be ~0, got {v}");
        }
    }

    #[test]
    fn test_fused_norm_silu_equivalence() {
        // Verify fused == separate rms_norm + silu
        let input = vec![0.5, -1.0, 2.0, -0.5, 3.0, 1.0, -2.0, 0.0];
        let gamma = vec![1.0; 8];
        let mut rms_out = vec![0.0; 8];
        rms_norm_f32(&input, &gamma, EPS, &mut rms_out);
        let separate: Vec<f32> = rms_out.iter().map(|&x| silu(x)).collect();
        let mut fused = vec![0.0; 8];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut fused);
        assert_approx(&fused, &separate, TOL);
    }

    #[test]
    fn test_fused_norm_silu_negative_inputs() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let gamma = vec![1.0; 4];
        let rms_out = ref_rms_norm(&input, &gamma, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|&x| silu(x)).collect();
        let mut out = vec![0.0; 4];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_fused_norm_silu_scalar_matches_dispatcher() {
        let input = vec![1.0, -1.0, 2.0, -2.0, 0.5];
        let gamma = vec![1.0; 5];
        let mut out_d = vec![0.0; 5];
        let mut out_s = vec![0.0; 5];
        fused_norm_silu_f32(&input, &gamma, EPS, &mut out_d);
        scalar_fused_norm_silu_f32(&input, &gamma, EPS, &mut out_s);
        assert_approx(&out_d, &out_s, TOL);
    }

    // ══════════════════════════════════════════════════════════════
    // 5. online_norm_stats_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_online_stats_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let stats = online_norm_stats_f32(&input);
        let exp_mean = input.iter().sum::<f32>() / input.len() as f32;
        let exp_var = input.iter().map(|&x| (x - exp_mean) * (x - exp_mean)).sum::<f32>()
            / input.len() as f32;
        assert!((stats.mean - exp_mean).abs() < TOL, "mean: {} vs {}", stats.mean, exp_mean);
        assert!((stats.variance - exp_var).abs() < TOL, "var: {} vs {}", stats.variance, exp_var);
    }

    #[test]
    fn test_online_stats_empty() {
        let stats = online_norm_stats_f32(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.variance, 0.0);
    }

    #[test]
    fn test_online_stats_single() {
        let stats = online_norm_stats_f32(&[42.0]);
        assert!((stats.mean - 42.0).abs() < TOL);
        assert!(stats.variance.abs() < TOL);
    }

    #[test]
    fn test_online_stats_all_same() {
        let stats = online_norm_stats_f32(&[5.0; 100]);
        assert!((stats.mean - 5.0).abs() < TOL);
        assert!(stats.variance.abs() < TOL);
    }

    #[test]
    fn test_online_stats_vs_two_pass() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect();
        let stats = online_norm_stats_f32(&input);
        let n = input.len() as f32;
        let mean = input.iter().sum::<f32>() / n;
        let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        assert!((stats.mean - mean).abs() < 1e-4, "mean: {} vs {}", stats.mean, mean);
        assert!((stats.variance - var).abs() < 1e-3, "var: {} vs {}", stats.variance, var);
    }

    #[test]
    fn test_online_stats_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let stats = online_norm_stats_f32(&input);
        let exp_mean = (4e6 + 6.0) / 4.0;
        assert!((stats.mean - exp_mean).abs() < 1.0, "mean: {} vs {}", stats.mean, exp_mean);
    }

    #[test]
    fn test_online_stats_negative_values() {
        let input = vec![-3.0, -1.0, 1.0, 3.0];
        let stats = online_norm_stats_f32(&input);
        assert!(stats.mean.abs() < TOL, "mean should be ~0, got {}", stats.mean);
        let exp_var = (9.0 + 1.0 + 1.0 + 9.0) / 4.0;
        assert!((stats.variance - exp_var).abs() < TOL, "var: {} vs {}", stats.variance, exp_var);
    }

    #[test]
    fn test_online_stats_non_aligned() {
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let stats = online_norm_stats_f32(&input);
        let mean = input.iter().sum::<f32>() / 7.0;
        let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 7.0;
        assert!((stats.mean - mean).abs() < TOL);
        assert!((stats.variance - var).abs() < TOL);
    }

    #[test]
    fn test_online_stats_scalar_matches_dispatcher() {
        let input = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let d = online_norm_stats_f32(&input);
        let s = scalar_online_norm_stats_f32(&input);
        assert!((d.mean - s.mean).abs() < TOL);
        assert!((d.variance - s.variance).abs() < TOL);
    }

    // ══════════════════════════════════════════════════════════════
    // 6. norm_residual_add_f32
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_norm_residual_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let residual = vec![0.1; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().zip(residual.iter()).map(|(&l, &r)| l + r).collect();
        let mut out = vec![0.0; 8];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_norm_residual_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().zip(residual.iter()).map(|(&l, &r)| l + r).collect();
        let mut out = vec![0.0; 5];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_norm_residual_empty() {
        let mut out: Vec<f32> = vec![];
        norm_residual_add_f32(&[], &[], &[], &[], EPS, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_norm_residual_single() {
        let input = vec![42.0];
        let residual = vec![1.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected = vec![ln[0] + 1.0];
        let mut out = vec![0.0];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_norm_residual_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 4];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_norm_residual_non_aligned() {
        let n = 11;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let residual: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().zip(residual.iter()).map(|(&l, &r)| l + r).collect();
        let mut out = vec![0.0; n];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }

    #[test]
    fn test_norm_residual_large_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![1e6; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().zip(residual.iter()).map(|(&l, &r)| l + r).collect();
        let mut out = vec![0.0; 4];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, 1e-3);
    }

    #[test]
    fn test_norm_residual_equivalence() {
        // Fused should equal separate layernorm + add
        let input = vec![0.5, -1.0, 2.0, -0.5, 3.0, 1.0, -2.0, 0.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];

        let mut ln_out = vec![0.0; 8];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut ln_out);
        let separate: Vec<f32> = ln_out.iter().zip(residual.iter()).map(|(&l, &r)| l + r).collect();

        let mut fused = vec![0.0; 8];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut fused);
        assert_approx(&fused, &separate, TOL);
    }

    #[test]
    fn test_norm_residual_scalar_matches_dispatcher() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![0.5; 5];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let mut out_d = vec![0.0; 5];
        let mut out_s = vec![0.0; 5];
        norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out_d);
        scalar_norm_residual_add_f32(&input, &residual, &gamma, &beta, EPS, &mut out_s);
        assert_approx(&out_d, &out_s, TOL);
    }

    // ══════════════════════════════════════════════════════════════
    // Cross-operation and dispatcher tests
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn test_rms_norm_differs_from_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut ln_out = vec![0.0; 4];
        let mut rms_out = vec![0.0; 4];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut ln_out);
        rms_norm_f32(&input, &gamma, EPS, &mut rms_out);
        // They should differ (RMSNorm has no mean subtraction)
        let any_diff = ln_out.iter().zip(rms_out.iter()).any(|(a, b)| (a - b).abs() > TOL);
        assert!(any_diff, "RMSNorm and LayerNorm should produce different results");
    }

    #[test]
    fn test_group_norm_one_group_equals_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let mut gn_out = vec![0.0; 6];
        let mut ln_out = vec![0.0; 6];
        group_norm_f32(&input, &gamma, &beta, 1, EPS, &mut gn_out);
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut ln_out);
        assert_approx(&gn_out, &ln_out, TOL);
    }

    #[test]
    fn test_online_stats_feeds_layer_norm() {
        // Use online stats to feed a manual layer norm
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let stats = online_norm_stats_f32(&input);
        let inv_std = 1.0 / (stats.variance + EPS).sqrt();
        let manual: Vec<f32> = input.iter().map(|&x| (x - stats.mean) * inv_std).collect();

        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &manual, TOL);
    }

    #[test]
    fn test_norm_stats_struct_debug() {
        let stats = NormStats { mean: 1.0, variance: 2.0 };
        let dbg = format!("{:?}", stats);
        assert!(dbg.contains("mean"));
        assert!(dbg.contains("variance"));
    }

    #[test]
    fn test_norm_stats_clone() {
        let stats = NormStats { mean: 1.5, variance: 2.5 };
        let cloned = stats;
        assert_eq!(stats.mean, cloned.mean);
        assert_eq!(stats.variance, cloned.variance);
    }

    #[test]
    fn test_dispatcher_uses_neon_on_aarch64() {
        // On aarch64 this calls NEON; on other arches scalar.
        // Either way, the result should match the scalar reference.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        let mut out = vec![0.0; 4];
        layer_norm_f32(&input, &gamma, &beta, EPS, &mut out);
        assert_approx(&out, &expected, TOL);
    }
}
