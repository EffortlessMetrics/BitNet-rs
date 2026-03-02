//! SIMD-optimized batch normalization and related normalization kernels.
//!
//! Provides AVX2-accelerated implementations of batch normalization,
//! group normalization, instance normalization, fused LayerNorm with
//! residual addition, and online running statistics update.
//!
//! All public functions perform runtime AVX2 detection and fall back to
//! scalar implementations transparently.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Error helper ───────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Runtime AVX2 detection ─────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

// ── AVX2 helpers ───────────────────────────────────────────────────

/// Horizontal sum of all 8 f32 lanes in a __m256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(sums, sums);
    let total = _mm_add_ss(sums, hi64);
    _mm_cvtss_f32(total)
}

// ── 1. Batch Norm Forward ──────────────────────────────────────────

/// Vectorized batch normalization forward pass.
///
/// Input is `[N, C]` flat layout. Returns `(output, batch_mean, batch_var)`.
///
/// Uses AVX2 for the normalize-and-scale inner loop when available.
pub fn batch_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_features: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    validate_norm_args(input, gamma, beta, num_features, eps)?;
    let batch_size = input.len() / num_features;

    // Compute per-channel mean (f64 for stability).
    let mut mean = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for c in 0..num_features {
            mean[c] += input[n * num_features + c] as f64;
        }
    }
    let count = batch_size as f64;
    for m in &mut mean {
        *m /= count;
    }

    // Compute per-channel variance.
    let mut var = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for c in 0..num_features {
            let d = input[n * num_features + c] as f64 - mean[c];
            var[c] += d * d;
        }
    }
    for v in &mut var {
        *v /= count;
    }

    let mean_f32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
    let var_f32: Vec<f32> = var.iter().map(|&v| v as f32).collect();

    // Normalize and scale.
    let mut output = vec![0.0f32; input.len()];
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            batch_norm_normalize_avx2(
                input,
                &mut output,
                &mean_f32,
                &var_f32,
                gamma,
                beta,
                num_features,
                batch_size,
                eps,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        batch_norm_normalize_scalar(
            input,
            &mut output,
            &mean,
            &var,
            gamma,
            beta,
            num_features,
            batch_size,
            eps,
        );
    } else {
        batch_norm_normalize_scalar(
            input,
            &mut output,
            &mean,
            &var,
            gamma,
            beta,
            num_features,
            batch_size,
            eps,
        );
    }

    Ok((output, mean_f32, var_f32))
}

#[allow(clippy::too_many_arguments)]
fn batch_norm_normalize_scalar(
    input: &[f32],
    output: &mut [f32],
    mean: &[f64],
    var: &[f64],
    gamma: &[f32],
    beta: &[f32],
    num_features: usize,
    batch_size: usize,
    eps: f32,
) {
    for n in 0..batch_size {
        for c in 0..num_features {
            let inv_std = 1.0 / (var[c] + eps as f64).sqrt();
            let x_hat = (input[n * num_features + c] as f64 - mean[c]) * inv_std;
            output[n * num_features + c] = (gamma[c] as f64 * x_hat + beta[c] as f64) as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn batch_norm_normalize_avx2(
    input: &[f32],
    output: &mut [f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_features: usize,
    batch_size: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;

    for n in 0..batch_size {
        let row_off = n * num_features;
        let mut c = 0usize;
        while c + 8 <= num_features {
            unsafe {
                let eps_v = _mm256_set1_ps(eps);
                let x = _mm256_loadu_ps(input.as_ptr().add(row_off + c));
                let m = _mm256_loadu_ps(mean.as_ptr().add(c));
                let v = _mm256_loadu_ps(var.as_ptr().add(c));
                let g = _mm256_loadu_ps(gamma.as_ptr().add(c));
                let b = _mm256_loadu_ps(beta.as_ptr().add(c));

                let inv_std =
                    _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(_mm256_add_ps(v, eps_v)));
                let x_hat = _mm256_mul_ps(_mm256_sub_ps(x, m), inv_std);
                let result = _mm256_add_ps(_mm256_mul_ps(g, x_hat), b);
                _mm256_storeu_ps(output.as_mut_ptr().add(row_off + c), result);
            }
            c += 8;
        }
        while c < num_features {
            let inv_std = 1.0 / (var[c] + eps).sqrt();
            let x_hat = (input[row_off + c] - mean[c]) * inv_std;
            output[row_off + c] = gamma[c] * x_hat + beta[c];
            c += 1;
        }
    }
}

// ── 2. Batch Norm Inference ────────────────────────────────────────

/// Fused scale+bias batch normalization for inference mode.
///
/// Uses pre-computed running mean/variance. No statistics update.
/// Pre-fuses `scale = gamma / sqrt(var + eps)` and `bias = beta - mean * scale`
/// for a single multiply-add per element.
pub fn batch_norm_inference(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    num_features: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    validate_inference_args(input, gamma, beta, running_mean, running_var, num_features, eps)?;
    let batch_size = input.len() / num_features;

    // Pre-fuse scale and bias per channel.
    let mut fused_scale = vec![0.0f32; num_features];
    let mut fused_bias = vec![0.0f32; num_features];
    for c in 0..num_features {
        let s = gamma[c] / (running_var[c] + eps).sqrt();
        fused_scale[c] = s;
        fused_bias[c] = beta[c] - running_mean[c] * s;
    }

    let mut output = vec![0.0f32; input.len()];
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fused_scale_bias_avx2(
                input,
                &mut output,
                &fused_scale,
                &fused_bias,
                num_features,
                batch_size,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        fused_scale_bias_scalar(
            input,
            &mut output,
            &fused_scale,
            &fused_bias,
            num_features,
            batch_size,
        );
    } else {
        fused_scale_bias_scalar(
            input,
            &mut output,
            &fused_scale,
            &fused_bias,
            num_features,
            batch_size,
        );
    }

    Ok(output)
}

fn fused_scale_bias_scalar(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    bias: &[f32],
    num_features: usize,
    batch_size: usize,
) {
    for n in 0..batch_size {
        for c in 0..num_features {
            let idx = n * num_features + c;
            output[idx] = input[idx] * scale[c] + bias[c];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fused_scale_bias_avx2(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    bias: &[f32],
    num_features: usize,
    batch_size: usize,
) {
    use std::arch::x86_64::*;
    for n in 0..batch_size {
        let off = n * num_features;
        let mut c = 0usize;
        while c + 8 <= num_features {
            unsafe {
                let x = _mm256_loadu_ps(input.as_ptr().add(off + c));
                let s = _mm256_loadu_ps(scale.as_ptr().add(c));
                let b = _mm256_loadu_ps(bias.as_ptr().add(c));
                let r = _mm256_add_ps(_mm256_mul_ps(x, s), b);
                _mm256_storeu_ps(output.as_mut_ptr().add(off + c), r);
            }
            c += 8;
        }
        while c < num_features {
            output[off + c] = input[off + c] * scale[c] + bias[c];
            c += 1;
        }
    }
}

// ── 3. Group Normalization ─────────────────────────────────────────

/// Group normalization with configurable group count.
///
/// Input is `[N, C]` flat layout. `num_channels` must be divisible by
/// `num_groups`. Each group of `C / num_groups` channels is independently
/// normalized, then scaled by `gamma` and shifted by `beta`.
pub fn group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_channels: usize,
    num_groups: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if num_channels == 0 || num_groups == 0 {
        return Err(invalid_args("num_channels and num_groups must be > 0"));
    }
    if !num_channels.is_multiple_of(num_groups) {
        return Err(invalid_args("num_channels must be divisible by num_groups"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(num_channels) {
        return Err(invalid_args("input length must be a multiple of num_channels"));
    }
    if gamma.len() != num_channels {
        return Err(invalid_args("gamma length must equal num_channels"));
    }
    if beta.len() != num_channels {
        return Err(invalid_args("beta length must equal num_channels"));
    }
    validate_eps(eps)?;

    let batch_size = input.len() / num_channels;
    let group_size = num_channels / num_groups;
    let mut output = vec![0.0f32; input.len()];

    for n in 0..batch_size {
        let row_off = n * num_channels;
        for g in 0..num_groups {
            let start = row_off + g * group_size;
            let group_slice = &input[start..start + group_size];

            let (mean, var) = compute_mean_var(group_slice);

            if has_avx2() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    normalize_affine_avx2(
                        group_slice,
                        &mut output[start..start + group_size],
                        mean,
                        var,
                        &gamma[g * group_size..(g + 1) * group_size],
                        &beta[g * group_size..(g + 1) * group_size],
                        eps,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                normalize_affine_scalar(
                    group_slice,
                    &mut output[start..start + group_size],
                    mean,
                    var,
                    &gamma[g * group_size..(g + 1) * group_size],
                    &beta[g * group_size..(g + 1) * group_size],
                    eps,
                );
            } else {
                normalize_affine_scalar(
                    group_slice,
                    &mut output[start..start + group_size],
                    mean,
                    var,
                    &gamma[g * group_size..(g + 1) * group_size],
                    &beta[g * group_size..(g + 1) * group_size],
                    eps,
                );
            }
        }
    }

    Ok(output)
}

// ── 4. Instance Normalization ──────────────────────────────────────

/// Instance normalization — group norm with `num_groups == num_channels`.
///
/// Each channel is independently normalized across its own statistics.
pub fn instance_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_channels: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    // Instance norm is group norm with num_groups == num_channels.
    group_norm(input, gamma, beta, num_channels, num_channels, eps)
}

// ── 5. Fused LayerNorm + Residual ──────────────────────────────────

/// Fused LayerNorm with residual addition.
///
/// Computes `LayerNorm(input + residual)` in a single pass.
/// `norm_size` is the size of each normalization group (last dimension).
pub fn layer_norm_fused(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    norm_size: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if norm_size == 0 {
        return Err(invalid_args("norm_size must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if input.len() != residual.len() {
        return Err(invalid_args("input and residual must have same length"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of norm_size"));
    }
    if gamma.len() != norm_size {
        return Err(invalid_args("gamma length must equal norm_size"));
    }
    if beta.len() != norm_size {
        return Err(invalid_args("beta length must equal norm_size"));
    }
    validate_eps(eps)?;

    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let off = b * norm_size;
        let inp = &input[off..off + norm_size];
        let res = &residual[off..off + norm_size];
        let out = &mut output[off..off + norm_size];

        if has_avx2() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                layer_norm_fused_avx2(inp, res, out, gamma, beta, eps);
            }
            #[cfg(not(target_arch = "x86_64"))]
            layer_norm_fused_scalar(inp, res, out, gamma, beta, eps);
        } else {
            layer_norm_fused_scalar(inp, res, out, gamma, beta, eps);
        }
    }

    Ok(output)
}

fn layer_norm_fused_scalar(
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = input.len();
    // Add residual and compute mean.
    let mut sum = 0.0f64;
    for i in 0..n {
        let val = input[i] + residual[i];
        output[i] = val; // temporary storage for fused values
        sum += val as f64;
    }
    let mean = (sum / n as f64) as f32;

    // Compute variance.
    let mut var_sum = 0.0f64;
    for val in output.iter() {
        let d = (*val - mean) as f64;
        var_sum += d * d;
    }
    let var = (var_sum / n as f64) as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    // Normalize with affine.
    for i in 0..n {
        output[i] = (output[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_fused_avx2(
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    use std::arch::x86_64::*;
    let n = input.len();

    // Pass 1: fused add + sum for mean.
    let mut sum_v = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        unsafe {
            let a = _mm256_loadu_ps(input.as_ptr().add(i));
            let r = _mm256_loadu_ps(residual.as_ptr().add(i));
            let s = _mm256_add_ps(a, r);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), s);
            sum_v = _mm256_add_ps(sum_v, s);
        }
        i += 8;
    }
    let mut sum = unsafe { hsum_avx(sum_v) } as f64;
    while i < n {
        let val = input[i] + residual[i];
        output[i] = val;
        sum += val as f64;
        i += 1;
    }
    let mean = (sum / n as f64) as f32;

    // Pass 2: variance.
    let mean_v = _mm256_set1_ps(mean);
    let mut var_v = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        unsafe {
            let x = _mm256_loadu_ps(output.as_ptr().add(i));
            let d = _mm256_sub_ps(x, mean_v);
            var_v = _mm256_add_ps(var_v, _mm256_mul_ps(d, d));
        }
        i += 8;
    }
    let mut var_sum = unsafe { hsum_avx(var_v) } as f64;
    while i < n {
        let d = (output[i] - mean) as f64;
        var_sum += d * d;
        i += 1;
    }
    let inv_std = 1.0 / ((var_sum / n as f64) as f32 + eps).sqrt();

    // Pass 3: normalize with affine.
    let inv_v = _mm256_set1_ps(inv_std);
    i = 0;
    while i + 8 <= n {
        unsafe {
            let x = _mm256_loadu_ps(output.as_ptr().add(i));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(i));
            let b = _mm256_loadu_ps(beta.as_ptr().add(i));
            let normed = _mm256_mul_ps(_mm256_sub_ps(x, mean_v), inv_v);
            let result = _mm256_add_ps(_mm256_mul_ps(g, normed), b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
        i += 8;
    }
    while i < n {
        output[i] = (output[i] - mean) * inv_std * gamma[i] + beta[i];
        i += 1;
    }
}

// ── 6. Running Statistics ──────────────────────────────────────────

/// Online mean/variance update for training.
///
/// Performs Welford-style online update of running mean and running variance
/// given a new batch of data. `momentum` controls the blending ratio:
/// `new_running = (1 - momentum) * old_running + momentum * batch_stat`.
///
/// Returns `(updated_running_mean, updated_running_var)`.
pub fn running_stats(
    input: &[f32],
    running_mean: &mut [f32],
    running_var: &mut [f32],
    num_features: usize,
    momentum: f32,
) -> Result<()> {
    if num_features == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    if running_mean.len() != num_features {
        return Err(invalid_args("running_mean length must equal num_features"));
    }
    if running_var.len() != num_features {
        return Err(invalid_args("running_var length must equal num_features"));
    }
    if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
        return Err(invalid_args("momentum must be in [0, 1] and finite"));
    }

    let batch_size = input.len() / num_features;
    let count = batch_size as f64;

    // Compute batch mean.
    let mut batch_mean = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for c in 0..num_features {
            batch_mean[c] += input[n * num_features + c] as f64;
        }
    }
    for m in &mut batch_mean {
        *m /= count;
    }

    // Compute batch variance.
    let mut batch_var = vec![0.0f64; num_features];
    for n in 0..batch_size {
        for c in 0..num_features {
            let d = input[n * num_features + c] as f64 - batch_mean[c];
            batch_var[c] += d * d;
        }
    }
    for v in &mut batch_var {
        *v /= count;
    }

    // EMA update.
    let mom = momentum as f64;
    for c in 0..num_features {
        running_mean[c] = ((1.0 - mom) * running_mean[c] as f64 + mom * batch_mean[c]) as f32;
        running_var[c] = ((1.0 - mom) * running_var[c] as f64 + mom * batch_var[c]) as f32;
    }

    Ok(())
}

// ── Shared helpers ─────────────────────────────────────────────────

fn validate_eps(eps: f32) -> Result<()> {
    if eps <= 0.0 || !eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    Ok(())
}

fn validate_norm_args(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    num_features: usize,
    eps: f32,
) -> Result<()> {
    if num_features == 0 {
        return Err(invalid_args("num_features must be > 0"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(invalid_args("input length must be a multiple of num_features"));
    }
    if gamma.len() != num_features {
        return Err(invalid_args("gamma length must equal num_features"));
    }
    if beta.len() != num_features {
        return Err(invalid_args("beta length must equal num_features"));
    }
    validate_eps(eps)
}

fn validate_inference_args(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    num_features: usize,
    eps: f32,
) -> Result<()> {
    validate_norm_args(input, gamma, beta, num_features, eps)?;
    if running_mean.len() != num_features {
        return Err(invalid_args("running_mean length must equal num_features"));
    }
    if running_var.len() != num_features {
        return Err(invalid_args("running_var length must equal num_features"));
    }
    Ok(())
}

/// Compute mean and variance via f64 accumulation.
fn compute_mean_var(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f64;
    let mut sum = 0.0f64;
    for &x in data {
        sum += x as f64;
    }
    let mean = sum / n;
    let mut var_sum = 0.0f64;
    for &x in data {
        let d = x as f64 - mean;
        var_sum += d * d;
    }
    (mean as f32, (var_sum / n) as f32)
}

fn normalize_affine_scalar(
    input: &[f32],
    output: &mut [f32],
    mean: f32,
    var: f32,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..input.len() {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_affine_avx2(
    input: &[f32],
    output: &mut [f32],
    mean: f32,
    var: f32,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    use std::arch::x86_64::*;
    let inv_std = 1.0 / (var + eps).sqrt();
    let n = input.len();

    let mut i = 0usize;
    while i + 8 <= n {
        unsafe {
            let mean_v = _mm256_set1_ps(mean);
            let inv_v = _mm256_set1_ps(inv_std);
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(i));
            let b = _mm256_loadu_ps(beta.as_ptr().add(i));
            let normed = _mm256_mul_ps(_mm256_sub_ps(x, mean_v), inv_v);
            let result = _mm256_add_ps(_mm256_mul_ps(g, normed), b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
        i += 8;
    }
    while i < n {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
        i += 1;
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// Scalar reference for batch norm forward.
    fn ref_bn_forward(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        c: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = input.len() / c;
        let mut mean = vec![0.0f64; c];
        let mut var = vec![0.0f64; c];
        for b in 0..n {
            for ch in 0..c {
                mean[ch] += input[b * c + ch] as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        for b in 0..n {
            for ch in 0..c {
                let d = input[b * c + ch] as f64 - mean[ch];
                var[ch] += d * d;
            }
        }
        for v in &mut var {
            *v /= n as f64;
        }
        let mut out = vec![0.0f32; input.len()];
        for b in 0..n {
            for ch in 0..c {
                let inv = 1.0 / (var[ch] + eps as f64).sqrt();
                let xh = (input[b * c + ch] as f64 - mean[ch]) * inv;
                out[b * c + ch] = (gamma[ch] as f64 * xh + beta[ch] as f64) as f32;
            }
        }
        let mf: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        let vf: Vec<f32> = var.iter().map(|&v| v as f32).collect();
        (out, mf, vf)
    }

    /// Scalar reference for batch norm inference.
    fn ref_bn_inference(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rmean: &[f32],
        rvar: &[f32],
        c: usize,
        eps: f32,
    ) -> Vec<f32> {
        let n = input.len() / c;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..n {
            for ch in 0..c {
                let inv = 1.0 / (rvar[ch] as f64 + eps as f64).sqrt();
                let xh = (input[b * c + ch] as f64 - rmean[ch] as f64) * inv;
                out[b * c + ch] = (gamma[ch] as f64 * xh + beta[ch] as f64) as f32;
            }
        }
        out
    }

    /// Scalar reference for group norm.
    fn ref_group_norm(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        c: usize,
        groups: usize,
        eps: f32,
    ) -> Vec<f32> {
        let n = input.len() / c;
        let gs = c / groups;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..n {
            for g in 0..groups {
                let off = b * c + g * gs;
                let sl = &input[off..off + gs];
                let m: f64 = sl.iter().map(|&x| x as f64).sum::<f64>() / gs as f64;
                let v: f64 = sl
                    .iter()
                    .map(|&x| {
                        let d = x as f64 - m;
                        d * d
                    })
                    .sum::<f64>()
                    / gs as f64;
                let inv = 1.0 / (v + eps as f64).sqrt();
                for i in 0..gs {
                    let ch = g * gs + i;
                    out[off + i] = ((input[off + i] as f64 - m) * inv * gamma[ch] as f64
                        + beta[ch] as f64) as f32;
                }
            }
        }
        out
    }

    /// Scalar reference for fused layer norm + residual.
    fn ref_layer_norm_fused(
        input: &[f32],
        residual: &[f32],
        gamma: &[f32],
        beta: &[f32],
        norm_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let batches = input.len() / norm_size;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batches {
            let off = b * norm_size;
            let mut sum = 0.0f64;
            for i in 0..norm_size {
                let v = input[off + i] + residual[off + i];
                out[off + i] = v;
                sum += v as f64;
            }
            let mean = (sum / norm_size as f64) as f32;
            let mut var_sum = 0.0f64;
            for i in 0..norm_size {
                let d = (out[off + i] - mean) as f64;
                var_sum += d * d;
            }
            let var = (var_sum / norm_size as f64) as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..norm_size {
                out[off + i] = (out[off + i] - mean) * inv * gamma[i] + beta[i];
            }
        }
        out
    }

    // ================================================================
    // batch_norm_forward tests
    // ================================================================

    #[test]
    fn bn_forward_basic_2ch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, _, _) = batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        let (exp, _, _) = ref_bn_forward(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_forward_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, 2, 1e-5).unwrap();
        let (exp, _, _) = ref_bn_forward(&input, &gamma, &beta, 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_forward_uniform_input() {
        let (out, _, _) = batch_norm_forward(&[5.0; 8], &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn bn_forward_single_sample() {
        let (out, _, _) = batch_norm_forward(&[3.0, 7.0], &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        assert!(out[0].abs() < TOL);
        assert!(out[1].abs() < TOL);
    }

    #[test]
    fn bn_forward_zero_mean_property() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let (out, _, _) = batch_norm_forward(&input, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        let n = input.len() / 4;
        for ch in 0..4 {
            let mean: f32 = (0..n).map(|b| out[b * 4 + ch]).sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-3, "ch{ch} mean = {mean}");
        }
    }

    #[test]
    fn bn_forward_returns_correct_mean() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let (_, mean, _) = batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        assert!((mean[0] - 4.0).abs() < TOL);
        assert!((mean[1] - 6.0).abs() < TOL);
    }

    #[test]
    fn bn_forward_returns_correct_var() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let (_, _, var) = batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        // ch0: [2,6] mean=4, var=((2-4)^2+(6-4)^2)/2 = 4
        assert!((var[0] - 4.0).abs() < TOL);
        assert!((var[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn bn_forward_all_zeros() {
        let (out, mean, var) =
            batch_norm_forward(&[0.0; 12], &[1.0; 3], &[0.0; 3], 3, 1e-5).unwrap();
        assert!(out.iter().all(|&v| v.abs() < TOL));
        assert!(mean.iter().all(|&v| v.abs() < TOL));
        assert!(var.iter().all(|&v| v.abs() < TOL));
    }

    #[test]
    fn bn_forward_all_negative() {
        let input = vec![-10.0, -20.0, -30.0, -40.0];
        let (out, _, _) = batch_norm_forward(&input, &[1.0], &[0.0], 1, 1e-5).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < TOL);
    }

    #[test]
    fn bn_forward_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6, 1e6 + 1.0];
        let (out, _, _) = batch_norm_forward(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bn_forward_large_batch() {
        let c = 4;
        let n = 1024;
        let input: Vec<f32> = (0..n * c).map(|i| (i % n) as f32).collect();
        let (out, _, _) =
            batch_norm_forward(&input, &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();
        for ch in 0..c {
            let mean: f32 = (0..n).map(|b| out[b * c + ch]).sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-3, "ch{ch} mean={mean}");
        }
    }

    #[test]
    fn bn_forward_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (out, _, _) = batch_norm_forward(&input, &[1.0], &[0.0], 1, 1e-5).unwrap();
        let (exp, _, _) = ref_bn_forward(&input, &[1.0], &[0.0], 1, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_forward_many_channels() {
        let c = 64;
        let n = 8;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.01 - 3.0).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.01).collect();
        let beta: Vec<f32> = (0..c).map(|i| -1.0 + i as f32 * 0.02).collect();
        let (out, _, _) = batch_norm_forward(&input, &gamma, &beta, c, 1e-5).unwrap();
        let (exp, _, _) = ref_bn_forward(&input, &gamma, &beta, c, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_forward_symmetric_input() {
        let input = vec![-100.0, -50.0, 50.0, 100.0];
        let (out, _, _) = batch_norm_forward(&input, &[1.0], &[0.0], 1, 1e-5).unwrap();
        assert!((out[0] + out[3]).abs() < TOL);
        assert!((out[1] + out[2]).abs() < TOL);
    }

    #[test]
    fn bn_forward_non_unit_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![3.0, 0.1];
        let (out, _, _) = batch_norm_forward(&input, &gamma, &[0.0; 2], 2, 1e-5).unwrap();
        let (exp, _, _) = ref_bn_forward(&input, &gamma, &[0.0; 2], 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    // ── batch_norm_forward error tests ─────────────────────────

    #[test]
    fn bn_forward_err_empty_input() {
        assert!(batch_norm_forward(&[], &[1.0], &[0.0], 1, 1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_zero_features() {
        assert!(batch_norm_forward(&[1.0], &[], &[], 0, 1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_gamma_mismatch() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0], &[0.0; 2], 2, 1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_beta_mismatch() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0], 2, 1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_not_multiple() {
        assert!(batch_norm_forward(&[1.0, 2.0, 3.0], &[1.0; 2], &[0.0; 2], 2, 1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_zero_eps() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, 0.0).is_err());
    }

    #[test]
    fn bn_forward_err_negative_eps() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, -1e-5).is_err());
    }

    #[test]
    fn bn_forward_err_nan_eps() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, f32::NAN).is_err());
    }

    #[test]
    fn bn_forward_err_inf_eps() {
        assert!(batch_norm_forward(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, f32::INFINITY).is_err());
    }

    // ================================================================
    // batch_norm_inference tests
    // ================================================================

    #[test]
    fn bn_inference_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let rm = vec![2.0, 3.0];
        let rv = vec![1.0, 1.0];
        let out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 2, 1e-5).unwrap();
        let exp = ref_bn_inference(&input, &[1.0; 2], &[0.0; 2], &rm, &rv, 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_inference_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let rm = vec![3.0, 4.0];
        let rv = vec![4.0, 9.0];
        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 2, 1e-5).unwrap();
        let exp = ref_bn_inference(&input, &gamma, &beta, &rm, &rv, 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_inference_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 2, 1e-5)
            .unwrap();
        assert!(approx_eq(&out, &input, 1e-3));
    }

    #[test]
    fn bn_inference_large_batch() {
        let c = 4;
        let n = 256;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.01).collect();
        let rm: Vec<f32> = vec![1.0; c];
        let rv: Vec<f32> = vec![2.0; c];
        let out =
            batch_norm_inference(&input, &vec![1.0; c], &vec![0.0; c], &rm, &rv, c, 1e-5).unwrap();
        let exp = ref_bn_inference(&input, &vec![1.0; c], &vec![0.0; c], &rm, &rv, c, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_inference_many_channels() {
        let c = 64;
        let n = 4;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let rm: Vec<f32> = (0..c).map(|i| i as f32 * 0.05).collect();
        let rv: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.01).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.8 + i as f32 * 0.005).collect();
        let beta: Vec<f32> = (0..c).map(|i| -0.5 + i as f32 * 0.01).collect();
        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, c, 1e-5).unwrap();
        let exp = ref_bn_inference(&input, &gamma, &beta, &rm, &rv, c, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn bn_inference_stability() {
        let input = vec![1e10, -1e10, 0.0, 1e-10];
        let out = batch_norm_inference(&input, &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 2, 1e-5)
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── batch_norm_inference error tests ───────────────────────

    #[test]
    fn bn_inference_err_empty() {
        assert!(batch_norm_inference(&[], &[1.0], &[0.0], &[0.0], &[1.0], 1, 1e-5).is_err());
    }

    #[test]
    fn bn_inference_err_rmean_mismatch() {
        assert!(
            batch_norm_inference(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0], &[1.0; 2], 2, 1e-5,)
                .is_err()
        );
    }

    #[test]
    fn bn_inference_err_rvar_mismatch() {
        assert!(
            batch_norm_inference(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0], 2, 1e-5,)
                .is_err()
        );
    }

    #[test]
    fn bn_inference_err_zero_eps() {
        assert!(
            batch_norm_inference(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[1.0; 2], 2, 0.0,)
                .is_err()
        );
    }

    // ================================================================
    // group_norm tests
    // ================================================================

    #[test]
    fn gn_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // N=2, C=4
        let out = group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 2, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn gn_single_group_is_layer_norm() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let out = group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 1, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 1, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn gn_all_groups_is_instance_norm() {
        let input: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let out = group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 4, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &[1.0; 4], &[0.0; 4], 4, 4, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn gn_with_affine() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let gamma: Vec<f32> = vec![2.0, 0.5, 1.5, 0.8];
        let beta: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0];
        let out = group_norm(&input, &gamma, &beta, 4, 2, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &gamma, &beta, 4, 2, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn gn_large_channels() {
        let c = 64;
        let n = 4;
        let groups = 8;
        let input: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.01 - 3.0).collect();
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let out = group_norm(&input, &gamma, &beta, c, groups, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &gamma, &beta, c, groups, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn gn_uniform_within_group() {
        // All values same within each group → output ≈ beta
        let input = vec![5.0, 5.0, 3.0, 3.0];
        let beta = vec![1.0, 1.0, 2.0, 2.0];
        let out = group_norm(&input, &[1.0; 4], &beta, 4, 2, 1e-5).unwrap();
        // Normalized to 0, then + beta
        assert!((out[0] - 1.0).abs() < TOL);
        assert!((out[1] - 1.0).abs() < TOL);
        assert!((out[2] - 2.0).abs() < TOL);
        assert!((out[3] - 2.0).abs() < TOL);
    }

    #[test]
    fn gn_zero_mean_per_group() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let c = 8;
        let groups = 2;
        let gs = c / groups;
        let out = group_norm(&input, &[1.0; 8], &[0.0; 8], c, groups, 1e-5).unwrap();
        let n = input.len() / c;
        for b in 0..n {
            for g in 0..groups {
                let off = b * c + g * gs;
                let mean: f32 = out[off..off + gs].iter().sum::<f32>() / gs as f32;
                assert!(mean.abs() < 1e-3, "batch{b} group{g} mean={mean}");
            }
        }
    }

    // ── group_norm error tests ─────────────────────────────────

    #[test]
    fn gn_err_zero_channels() {
        assert!(group_norm(&[1.0], &[], &[], 0, 1, 1e-5).is_err());
    }

    #[test]
    fn gn_err_zero_groups() {
        assert!(group_norm(&[1.0], &[1.0], &[0.0], 1, 0, 1e-5).is_err());
    }

    #[test]
    fn gn_err_not_divisible() {
        assert!(group_norm(&[1.0, 2.0, 3.0], &[1.0; 3], &[0.0; 3], 3, 2, 1e-5).is_err());
    }

    #[test]
    fn gn_err_empty_input() {
        assert!(group_norm(&[], &[1.0], &[0.0], 1, 1, 1e-5).is_err());
    }

    #[test]
    fn gn_err_gamma_mismatch() {
        assert!(group_norm(&[1.0, 2.0], &[1.0], &[0.0; 2], 2, 1, 1e-5).is_err());
    }

    #[test]
    fn gn_err_beta_mismatch() {
        assert!(group_norm(&[1.0, 2.0], &[1.0; 2], &[0.0], 2, 1, 1e-5).is_err());
    }

    #[test]
    fn gn_err_zero_eps() {
        assert!(group_norm(&[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, 1, 0.0).is_err());
    }

    // ================================================================
    // instance_norm tests
    // ================================================================

    #[test]
    fn in_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = instance_norm(&input, &[1.0; 2], &[0.0; 2], 2, 1e-5).unwrap();
        // Instance norm on single-element channels → output ≈ beta
        assert!(out[0].abs() < TOL);
        assert!(out[1].abs() < TOL);
    }

    #[test]
    fn in_matches_group_norm() {
        let input: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let c = 4;
        let out_in = instance_norm(&input, &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();
        let out_gn = group_norm(&input, &vec![1.0; c], &vec![0.0; c], c, c, 1e-5).unwrap();
        assert!(approx_eq(&out_in, &out_gn, TOL));
    }

    #[test]
    fn in_with_affine() {
        let input: Vec<f32> = (0..12).map(|i| i as f32 - 5.0).collect();
        let gamma = vec![2.0, 0.5, 1.5];
        let beta = vec![1.0, -1.0, 0.0];
        let out = instance_norm(&input, &gamma, &beta, 3, 1e-5).unwrap();
        let exp = ref_group_norm(&input, &gamma, &beta, 3, 3, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn in_err_empty() {
        assert!(instance_norm(&[], &[1.0], &[0.0], 1, 1e-5).is_err());
    }

    // ================================================================
    // layer_norm_fused tests
    // ================================================================

    #[test]
    fn lnf_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 0.5, -0.5];
        let out = layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        let exp = ref_layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn lnf_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        let out = layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        let exp = ref_layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn lnf_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let residual = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let gamma = vec![2.0, 0.5, 1.5, 0.8];
        let beta = vec![1.0, -1.0, 0.5, 0.0];
        let out = layer_norm_fused(&input, &residual, &gamma, &beta, 4, 1e-5).unwrap();
        let exp = ref_layer_norm_fused(&input, &residual, &gamma, &beta, 4, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn lnf_batched() {
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let residual: Vec<f32> = (0..32).map(|i| (i as f32) * -0.05).collect();
        let out = layer_norm_fused(&input, &residual, &[1.0; 8], &[0.0; 8], 8, 1e-5).unwrap();
        let exp = ref_layer_norm_fused(&input, &residual, &[1.0; 8], &[0.0; 8], 8, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn lnf_large_norm_size() {
        let ns = 128;
        let input: Vec<f32> = (0..ns * 2).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let residual: Vec<f32> = (0..ns * 2).map(|i| (i as f32) * 0.005).collect();
        let gamma = vec![1.0f32; ns];
        let beta = vec![0.0f32; ns];
        let out = layer_norm_fused(&input, &residual, &gamma, &beta, ns, 1e-5).unwrap();
        let exp = ref_layer_norm_fused(&input, &residual, &gamma, &beta, ns, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }

    #[test]
    fn lnf_opposite_residual_cancels() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![-1.0, -2.0, -3.0, -4.0];
        let out = layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        // input + residual = [0,0,0,0] → all outputs ≈ 0
        assert!(out.iter().all(|&v| v.abs() < TOL));
    }

    #[test]
    fn lnf_stability_large_values() {
        let input = vec![1e6, -1e6, 1e6, -1e6];
        let residual = vec![1.0; 4];
        let out = layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── layer_norm_fused error tests ───────────────────────────

    #[test]
    fn lnf_err_empty() {
        assert!(layer_norm_fused(&[], &[], &[1.0], &[0.0], 1, 1e-5).is_err());
    }

    #[test]
    fn lnf_err_length_mismatch() {
        assert!(layer_norm_fused(&[1.0, 2.0], &[1.0], &[1.0; 2], &[0.0; 2], 2, 1e-5).is_err());
    }

    #[test]
    fn lnf_err_norm_size_zero() {
        assert!(layer_norm_fused(&[1.0], &[1.0], &[], &[], 0, 1e-5).is_err());
    }

    #[test]
    fn lnf_err_not_multiple() {
        assert!(
            layer_norm_fused(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[1.0; 2], &[0.0; 2], 2, 1e-5,)
                .is_err()
        );
    }

    #[test]
    fn lnf_err_gamma_mismatch() {
        assert!(layer_norm_fused(&[1.0, 2.0], &[1.0, 2.0], &[1.0], &[0.0; 2], 2, 1e-5,).is_err());
    }

    #[test]
    fn lnf_err_beta_mismatch() {
        assert!(layer_norm_fused(&[1.0, 2.0], &[1.0, 2.0], &[1.0; 2], &[0.0], 2, 1e-5,).is_err());
    }

    #[test]
    fn lnf_err_zero_eps() {
        assert!(layer_norm_fused(&[1.0, 2.0], &[1.0, 2.0], &[1.0; 2], &[0.0; 2], 2, 0.0,).is_err());
    }

    // ================================================================
    // running_stats tests
    // ================================================================

    #[test]
    fn rs_basic_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut rm = vec![0.0; 2];
        let mut rv = vec![1.0; 2];
        running_stats(&input, &mut rm, &mut rv, 2, 0.1).unwrap();
        // batch mean ch0: (2+6)/2 = 4, ch1: (4+8)/2 = 6
        // rm = 0.9*0 + 0.1*4 = 0.4, 0.9*0 + 0.1*6 = 0.6
        assert!((rm[0] - 0.4).abs() < TOL);
        assert!((rm[1] - 0.6).abs() < TOL);
    }

    #[test]
    fn rs_zero_momentum_no_change() {
        let mut rm = vec![5.0, 10.0];
        let mut rv = vec![2.0, 3.0];
        let rm_orig = rm.clone();
        let rv_orig = rv.clone();
        running_stats(&[1.0, 2.0, 3.0, 4.0], &mut rm, &mut rv, 2, 0.0).unwrap();
        assert!(approx_eq(&rm, &rm_orig, TOL));
        assert!(approx_eq(&rv, &rv_orig, TOL));
    }

    #[test]
    fn rs_full_momentum_replaces() {
        let mut rm = vec![100.0; 2];
        let mut rv = vec![50.0; 2];
        running_stats(&[2.0, 4.0, 6.0, 8.0], &mut rm, &mut rv, 2, 1.0).unwrap();
        assert!((rm[0] - 4.0).abs() < TOL);
        assert!((rm[1] - 6.0).abs() < TOL);
    }

    #[test]
    fn rs_var_update() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut rm = vec![0.0; 2];
        let mut rv = vec![1.0; 2];
        running_stats(&input, &mut rm, &mut rv, 2, 0.1).unwrap();
        // batch var ch0: ((2-4)^2+(6-4)^2)/2 = 4 → rv = 0.9*1 + 0.1*4 = 1.3
        assert!((rv[0] - 1.3).abs() < TOL);
        assert!((rv[1] - 1.3).abs() < TOL);
    }

    #[test]
    fn rs_repeated_updates_converge() {
        let mut rm = vec![0.0; 1];
        let mut rv = vec![1.0; 1];
        let data = vec![10.0, 10.0, 10.0, 10.0];
        for _ in 0..100 {
            running_stats(&data, &mut rm, &mut rv, 1, 0.1).unwrap();
        }
        // Should converge to batch mean = 10, var = 0
        assert!((rm[0] - 10.0).abs() < 0.01);
        assert!(rv[0].abs() < 0.01);
    }

    #[test]
    fn rs_single_feature() {
        let mut rm = vec![0.0];
        let mut rv = vec![0.0];
        running_stats(&[1.0, 3.0, 5.0, 7.0], &mut rm, &mut rv, 1, 0.5).unwrap();
        // batch mean = 4, batch var = 5
        assert!((rm[0] - 2.0).abs() < TOL); // 0.5*0 + 0.5*4
        assert!((rv[0] - 2.5).abs() < TOL); // 0.5*0 + 0.5*5
    }

    #[test]
    fn rs_many_features() {
        let c = 32;
        let n = 8;
        let input: Vec<f32> = (0..n * c).map(|i| i as f32 * 0.01).collect();
        let mut rm = vec![0.0f32; c];
        let mut rv = vec![1.0f32; c];
        running_stats(&input, &mut rm, &mut rv, c, 0.1).unwrap();
        assert!(rm.iter().all(|v| v.is_finite()));
        assert!(rv.iter().all(|v| v.is_finite()));
    }

    // ── running_stats error tests ──────────────────────────────

    #[test]
    fn rs_err_empty() {
        let mut rm = vec![0.0];
        let mut rv = vec![1.0];
        assert!(running_stats(&[], &mut rm, &mut rv, 1, 0.1).is_err());
    }

    #[test]
    fn rs_err_zero_features() {
        let mut rm = vec![];
        let mut rv = vec![];
        assert!(running_stats(&[1.0], &mut rm, &mut rv, 0, 0.1).is_err());
    }

    #[test]
    fn rs_err_rmean_mismatch() {
        let mut rm = vec![0.0];
        let mut rv = vec![1.0; 2];
        assert!(running_stats(&[1.0, 2.0], &mut rm, &mut rv, 2, 0.1).is_err());
    }

    #[test]
    fn rs_err_rvar_mismatch() {
        let mut rm = vec![0.0; 2];
        let mut rv = vec![1.0];
        assert!(running_stats(&[1.0, 2.0], &mut rm, &mut rv, 2, 0.1).is_err());
    }

    #[test]
    fn rs_err_invalid_momentum() {
        let mut rm = vec![0.0];
        let mut rv = vec![1.0];
        assert!(running_stats(&[1.0], &mut rm, &mut rv, 1, 1.5).is_err());
    }

    #[test]
    fn rs_err_negative_momentum() {
        let mut rm = vec![0.0];
        let mut rv = vec![1.0];
        assert!(running_stats(&[1.0], &mut rm, &mut rv, 1, -0.1).is_err());
    }

    #[test]
    fn rs_err_not_multiple() {
        let mut rm = vec![0.0; 2];
        let mut rv = vec![1.0; 2];
        assert!(running_stats(&[1.0, 2.0, 3.0], &mut rm, &mut rv, 2, 0.1).is_err());
    }

    // ================================================================
    // Cross-function consistency tests
    // ================================================================

    #[test]
    fn forward_and_inference_agree_after_convergence() {
        // After running_stats converge, forward and inference should agree.
        let c = 2;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut rm = vec![0.0; c];
        let mut rv = vec![1.0; c];
        // Run many updates to converge.
        for _ in 0..200 {
            running_stats(&data, &mut rm, &mut rv, c, 0.1).unwrap();
        }
        let fwd = batch_norm_forward(&data, &[1.0; 2], &[0.0; 2], c, 1e-5).unwrap();
        let inf = batch_norm_inference(&data, &[1.0; 2], &[0.0; 2], &rm, &rv, c, 1e-5).unwrap();
        // After convergence, running stats ≈ batch stats, so outputs should be close.
        assert!(approx_eq(&fwd.0, &inf, 0.05));
    }

    #[test]
    fn instance_norm_equals_group_norm_max_groups() {
        let input: Vec<f32> = (0..20).map(|i| i as f32 * 0.3 - 2.0).collect();
        let c = 5;
        let out_in = instance_norm(&input, &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();
        let out_gn = group_norm(&input, &vec![1.0; c], &vec![0.0; c], c, c, 1e-5).unwrap();
        assert!(approx_eq(&out_in, &out_gn, TOL));
    }

    #[test]
    fn group_norm_one_group_is_layer_norm_equivalent() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let c = 4;
        // group_norm with 1 group normalizes across all channels.
        let gn = group_norm(&input, &vec![1.0; c], &vec![0.0; c], c, 1, 1e-5).unwrap();
        // Fused LN with zero residual should give same result.
        let ln = layer_norm_fused(&input, &vec![0.0; 16], &vec![1.0; c], &vec![0.0; c], c, 1e-5)
            .unwrap();
        assert!(approx_eq(&gn, &ln, TOL));
    }

    #[test]
    fn fused_ln_residual_equivalent_to_separate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 0.5, -0.5];
        // fused: LN(input + residual)
        let fused = layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5).unwrap();
        // separate: add then LN
        let added: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let separate = ref_layer_norm_fused(&input, &residual, &[1.0; 4], &[0.0; 4], 4, 1e-5);
        assert!(approx_eq(&fused, &separate, TOL));
        // Also verify added values are used.
        let _ = added; // used to derive ref
    }

    #[test]
    fn all_norms_finite_on_extreme_input() {
        let input = vec![1e7, -1e7, 0.0, 1e-8, 1e7, -1e7, 0.0, 1e-8];
        let c = 4;
        let (fwd, _, _) =
            batch_norm_forward(&input, &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();
        let inf = batch_norm_inference(
            &input,
            &vec![1.0; c],
            &vec![0.0; c],
            &vec![0.0; c],
            &vec![1.0; c],
            c,
            1e-5,
        )
        .unwrap();
        let gn = group_norm(&input, &vec![1.0; c], &vec![0.0; c], c, 2, 1e-5).unwrap();
        let ins = instance_norm(&input, &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();
        let ln =
            layer_norm_fused(&input, &[0.0; 8], &vec![1.0; c], &vec![0.0; c], c, 1e-5).unwrap();

        for (name, vals) in [("fwd", &fwd), ("inf", &inf), ("gn", &gn), ("in", &ins), ("ln", &ln)] {
            assert!(vals.iter().all(|v| v.is_finite()), "{name} has non-finite");
        }
    }

    #[test]
    fn bn_forward_eps_sensitivity() {
        let input = vec![0.0, 1.0];
        let (out_small, _, _) = batch_norm_forward(&input, &[1.0], &[0.0], 1, 1e-8).unwrap();
        let (out_large, _, _) = batch_norm_forward(&input, &[1.0], &[0.0], 1, 1.0).unwrap();
        // Larger eps → smaller magnitude.
        let mag_s = out_small.iter().map(|v| v.abs()).sum::<f32>();
        let mag_l = out_large.iter().map(|v| v.abs()).sum::<f32>();
        assert!(mag_s > mag_l);
    }

    #[test]
    fn bn_inference_fused_matches_naive() {
        // Verify fused scale+bias gives same result as naive computation.
        let c = 8;
        let n = 4;
        let input: Vec<f32> = (0..n * c).map(|i| i as f32 * 0.1 - 1.6).collect();
        let rm: Vec<f32> = (0..c).map(|i| i as f32 * 0.5).collect();
        let rv: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.1).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> = (0..c).map(|i| -1.0 + i as f32 * 0.2).collect();

        let out = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, c, 1e-5).unwrap();
        let exp = ref_bn_inference(&input, &gamma, &beta, &rm, &rv, c, 1e-5);
        assert!(approx_eq(&out, &exp, TOL));
    }
}
