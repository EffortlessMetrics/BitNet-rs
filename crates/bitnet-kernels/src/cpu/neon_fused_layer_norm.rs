//! NEON-accelerated fused layer normalization kernels.
//!
//! Provides eight normalization variants with ARM NEON SIMD acceleration
//! and scalar fallbacks for non-aarch64 targets.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::excessive_precision, clippy::let_and_return)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    dead_code,
    unused_assignments,
    unused_variables
)]

// ---------------------------------------------------------------------------
// Helper: NEON horizontal sum of a float32x4 vector
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vaddvq_f32_compat(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::*;
    vaddvq_f32(v)
}

// ---------------------------------------------------------------------------
// 1. neon_fused_rms_norm
// ---------------------------------------------------------------------------

/// RMS normalization: `out[i] = (x[i] / rms) * weight[i]`
/// where `rms = sqrt(mean(x^2) + eps)`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());
    assert!(n > 0, "input must be non-empty");

    // Compute sum of squares
    let mut sum_sq = 0.0f32;
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        sum_sq = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        sum_sq += input[i] * input[i];
    }

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    unsafe {
        let inv_v = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let x = vld1q_f32(input.as_ptr().add(i * 4));
            let w = vld1q_f32(weight.as_ptr().add(i * 4));
            let normed = vmulq_f32(x, inv_v);
            let scaled = vmulq_f32(normed, w);
            vst1q_f32(output.as_mut_ptr().add(i * 4), scaled);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = (input[i] * inv_rms) * weight[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());
    assert!(n > 0, "input must be non-empty");

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        output[i] = (input[i] * inv_rms) * weight[i];
    }
}

// ---------------------------------------------------------------------------
// 2. neon_fused_layer_norm
// ---------------------------------------------------------------------------

/// Full layer normalization: `out[i] = ((x[i] - mean) / sqrt(var + eps)) * gamma[i] + beta[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(n > 0, "input must be non-empty");

    let chunks = n / 4;

    // Mean
    let mut sum = 0.0f32;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        sum += input[i];
    }
    let mean = sum / n as f32;

    // Variance
    let mut var_sum = 0.0f32;
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            let d = vsubq_f32(v, mean_v);
            acc = vfmaq_f32(acc, d, d);
        }
        var_sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        let d = input[i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // Normalize, scale, shift
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        for i in 0..chunks {
            let x = vld1q_f32(input.as_ptr().add(i * 4));
            let g = vld1q_f32(gamma.as_ptr().add(i * 4));
            let b = vld1q_f32(beta.as_ptr().add(i * 4));
            let d = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(d, inv_v);
            let scaled = vfmaq_f32(b, normed, g);
            vst1q_f32(output.as_mut_ptr().add(i * 4), scaled);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = ((input[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(n > 0, "input must be non-empty");

    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        output[i] = ((input[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

// ---------------------------------------------------------------------------
// 3. neon_fused_group_norm
// ---------------------------------------------------------------------------

/// Group normalization: normalizes each group of `group_size` elements independently.
/// `gamma` and `beta` are per-element (length == input.len()).
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    num_groups: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(num_groups > 0, "num_groups must be > 0");
    assert_eq!(n % num_groups, 0, "input length must be divisible by num_groups");

    let group_size = n / num_groups;

    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group_in = &input[start..end];
        let chunks = group_size / 4;

        // Mean
        let mut sum = 0.0f32;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(group_in.as_ptr().add(i * 4));
                acc = vaddq_f32(acc, v);
            }
            sum = vaddvq_f32_compat(acc);
        }
        for i in (chunks * 4)..group_size {
            sum += group_in[i];
        }
        let mean = sum / group_size as f32;

        // Variance
        let mut var_sum = 0.0f32;
        unsafe {
            let mean_v = vdupq_n_f32(mean);
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(group_in.as_ptr().add(i * 4));
                let d = vsubq_f32(v, mean_v);
                acc = vfmaq_f32(acc, d, d);
            }
            var_sum = vaddvq_f32_compat(acc);
        }
        for i in (chunks * 4)..group_size {
            let d = group_in[i] - mean;
            var_sum += d * d;
        }
        let inv_std = 1.0 / (var_sum / group_size as f32 + eps).sqrt();

        // Normalize + affine
        unsafe {
            let mean_v = vdupq_n_f32(mean);
            let inv_v = vdupq_n_f32(inv_std);
            for i in 0..chunks {
                let off = start + i * 4;
                let x = vld1q_f32(input.as_ptr().add(off));
                let gv = vld1q_f32(gamma.as_ptr().add(off));
                let bv = vld1q_f32(beta.as_ptr().add(off));
                let d = vsubq_f32(x, mean_v);
                let normed = vmulq_f32(d, inv_v);
                let scaled = vfmaq_f32(bv, normed, gv);
                vst1q_f32(output.as_mut_ptr().add(off), scaled);
            }
        }
        for i in (chunks * 4)..group_size {
            let idx = start + i;
            output[idx] = ((input[idx] - mean) * inv_std) * gamma[idx] + beta[idx];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    num_groups: usize,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(num_groups > 0, "num_groups must be > 0");
    assert_eq!(n % num_groups, 0, "input length must be divisible by num_groups");

    let group_size = n / num_groups;
    for g in 0..num_groups {
        let start = g * group_size;
        let grp = &input[start..start + group_size];
        let mean: f32 = grp.iter().sum::<f32>() / group_size as f32;
        let var: f32 = grp.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..group_size {
            let idx = start + i;
            output[idx] = ((input[idx] - mean) * inv_std) * gamma[idx] + beta[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// 4. neon_fused_instance_norm
// ---------------------------------------------------------------------------

/// Instance normalization: normalizes each channel independently.
/// `input` shape is `[batch_size * channels, channel_size]` flattened.
/// `gamma` and `beta` are per-channel (length == `channels`).
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_instance_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, output.len());
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);
    assert!(batch_size > 0 && channels > 0);
    let channel_size = n / (batch_size * channels);
    assert_eq!(
        batch_size * channels * channel_size,
        n,
        "input length must equal batch_size * channels * channel_size"
    );

    let chunks = channel_size / 4;

    for b in 0..batch_size {
        for c in 0..channels {
            let start = (b * channels + c) * channel_size;
            let slice = &input[start..start + channel_size];

            // Mean
            let mut sum = 0.0f32;
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                for i in 0..chunks {
                    let v = vld1q_f32(slice.as_ptr().add(i * 4));
                    acc = vaddq_f32(acc, v);
                }
                sum = vaddvq_f32_compat(acc);
            }
            for i in (chunks * 4)..channel_size {
                sum += slice[i];
            }
            let mean = sum / channel_size as f32;

            // Variance
            let mut var_sum = 0.0f32;
            unsafe {
                let mean_v = vdupq_n_f32(mean);
                let mut acc = vdupq_n_f32(0.0);
                for i in 0..chunks {
                    let v = vld1q_f32(slice.as_ptr().add(i * 4));
                    let d = vsubq_f32(v, mean_v);
                    acc = vfmaq_f32(acc, d, d);
                }
                var_sum = vaddvq_f32_compat(acc);
            }
            for i in (chunks * 4)..channel_size {
                let d = slice[i] - mean;
                var_sum += d * d;
            }
            let inv_std = 1.0 / (var_sum / channel_size as f32 + eps).sqrt();

            let g = gamma[c];
            let bval = beta[c];

            unsafe {
                let mean_v = vdupq_n_f32(mean);
                let inv_v = vdupq_n_f32(inv_std);
                let gv = vdupq_n_f32(g);
                let bv = vdupq_n_f32(bval);
                for i in 0..chunks {
                    let off = start + i * 4;
                    let x = vld1q_f32(input.as_ptr().add(off));
                    let d = vsubq_f32(x, mean_v);
                    let normed = vmulq_f32(d, inv_v);
                    let scaled = vfmaq_f32(bv, normed, gv);
                    vst1q_f32(output.as_mut_ptr().add(off), scaled);
                }
            }
            for i in (chunks * 4)..channel_size {
                let idx = start + i;
                output[idx] = ((input[idx] - mean) * inv_std) * g + bval;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_instance_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    batch_size: usize,
    channels: usize,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, output.len());
    assert_eq!(gamma.len(), channels);
    assert_eq!(beta.len(), channels);
    assert!(batch_size > 0 && channels > 0);
    let channel_size = n / (batch_size * channels);
    assert_eq!(batch_size * channels * channel_size, n);

    for b in 0..batch_size {
        for c in 0..channels {
            let start = (b * channels + c) * channel_size;
            let slice = &input[start..start + channel_size];
            let mean: f32 = slice.iter().sum::<f32>() / channel_size as f32;
            let var: f32 =
                slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / channel_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let g = gamma[c];
            let bval = beta[c];
            for i in 0..channel_size {
                output[start + i] = ((input[start + i] - mean) * inv_std) * g + bval;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. neon_fused_batch_rms_norm
// ---------------------------------------------------------------------------

/// Batched RMS normalization: applies RMS norm to each row of length `dim`.
/// `input` shape is `[batch_size, dim]` flattened. `weight` has length `dim`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_batch_rms_norm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch_size: usize,
    dim: usize,
    eps: f32,
) {
    use std::arch::aarch64::*;

    assert_eq!(input.len(), batch_size * dim);
    assert_eq!(output.len(), batch_size * dim);
    assert_eq!(weight.len(), dim);
    assert!(dim > 0);

    let chunks = dim / 4;

    for b in 0..batch_size {
        let start = b * dim;
        let row = &input[start..start + dim];

        // Sum of squares
        let mut sum_sq = 0.0f32;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let v = vld1q_f32(row.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, v, v);
            }
            sum_sq = vaddvq_f32_compat(acc);
        }
        for i in (chunks * 4)..dim {
            sum_sq += row[i] * row[i];
        }
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

        unsafe {
            let inv_v = vdupq_n_f32(inv_rms);
            for i in 0..chunks {
                let off = start + i * 4;
                let x = vld1q_f32(input.as_ptr().add(off));
                let w = vld1q_f32(weight.as_ptr().add(i * 4));
                let normed = vmulq_f32(x, inv_v);
                let scaled = vmulq_f32(normed, w);
                vst1q_f32(output.as_mut_ptr().add(off), scaled);
            }
        }
        for i in (chunks * 4)..dim {
            output[start + i] = (input[start + i] * inv_rms) * weight[i];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_batch_rms_norm(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    batch_size: usize,
    dim: usize,
    eps: f32,
) {
    assert_eq!(input.len(), batch_size * dim);
    assert_eq!(output.len(), batch_size * dim);
    assert_eq!(weight.len(), dim);
    assert!(dim > 0);

    for b in 0..batch_size {
        let start = b * dim;
        let row = &input[start..start + dim];
        let sum_sq: f32 = row.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim {
            output[start + i] = (row[i] * inv_rms) * weight[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 6. neon_fused_prenorm_residual
// ---------------------------------------------------------------------------

/// Pre-normalization with residual: `output = layer_norm(input) + residual`.
/// Uses RMS norm internally.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_prenorm_residual(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, residual.len());
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    let chunks = n / 4;

    // Sum of squares for RMS
    let mut sum_sq = 0.0f32;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }
        sum_sq = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        sum_sq += input[i] * input[i];
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // norm(x) * weight + residual
    unsafe {
        let inv_v = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let w = vld1q_f32(weight.as_ptr().add(off));
            let r = vld1q_f32(residual.as_ptr().add(off));
            let normed = vmulq_f32(x, inv_v);
            let scaled = vfmaq_f32(r, normed, w);
            vst1q_f32(output.as_mut_ptr().add(off), scaled);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = (input[i] * inv_rms) * weight[i] + residual[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_prenorm_residual(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, residual.len());
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        output[i] = (input[i] * inv_rms) * weight[i] + residual[i];
    }
}

// ---------------------------------------------------------------------------
// 7. neon_fused_adaptive_layer_norm
// ---------------------------------------------------------------------------

/// Adaptive layer normalization (modulated by conditioning signal):
/// `out = (1 + scale_mod) * layer_norm(x) + shift_mod`
/// where `scale_mod` and `shift_mod` come from a conditioning vector.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_adaptive_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    scale_mod: &[f32],
    shift_mod: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, scale_mod.len());
    assert_eq!(n, shift_mod.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    let chunks = n / 4;

    // Mean
    let mut sum = 0.0f32;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        sum += input[i];
    }
    let mean = sum / n as f32;

    // Variance
    let mut var_sum = 0.0f32;
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            let d = vsubq_f32(v, mean_v);
            acc = vfmaq_f32(acc, d, d);
        }
        var_sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        let d = input[i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // Adaptive: (1 + scale_mod) * (gamma * normed + beta) + shift_mod
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let g = vld1q_f32(gamma.as_ptr().add(off));
            let b = vld1q_f32(beta.as_ptr().add(off));
            let sm = vld1q_f32(scale_mod.as_ptr().add(off));
            let sh = vld1q_f32(shift_mod.as_ptr().add(off));

            let d = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(d, inv_v);
            // gamma * normed + beta
            let ln_out = vfmaq_f32(b, normed, g);
            // (1 + scale_mod) * ln_out + shift_mod
            let scale = vaddq_f32(one, sm);
            let out = vfmaq_f32(sh, ln_out, scale);
            vst1q_f32(output.as_mut_ptr().add(off), out);
        }
    }
    for i in (chunks * 4)..n {
        let normed = (input[i] - mean) * inv_std;
        let ln_out = normed * gamma[i] + beta[i];
        output[i] = (1.0 + scale_mod[i]) * ln_out + shift_mod[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_adaptive_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    scale_mod: &[f32],
    shift_mod: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, scale_mod.len());
    assert_eq!(n, shift_mod.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        let normed = (input[i] - mean) * inv_std;
        let ln_out = normed * gamma[i] + beta[i];
        output[i] = (1.0 + scale_mod[i]) * ln_out + shift_mod[i];
    }
}

// ---------------------------------------------------------------------------
// 8. neon_fused_quantized_layer_norm
// ---------------------------------------------------------------------------

/// Layer normalization on quantized (i8) tensors.
/// Dequantizes via `x_f32 = (x_i8 as f32) * dequant_scale`, normalizes,
/// writes f32 output.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_quantized_layer_norm(
    input: &[i8],
    dequant_scale: f32,
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    // Dequantize into temp buffer
    let mut dequant = vec![0.0f32; n];
    let chunks = n / 4;

    unsafe {
        let scale_v = vdupq_n_f32(dequant_scale);
        for i in 0..chunks {
            let off = i * 4;
            let vals = [
                input[off] as f32,
                input[off + 1] as f32,
                input[off + 2] as f32,
                input[off + 3] as f32,
            ];
            let v = vld1q_f32(vals.as_ptr());
            let scaled = vmulq_f32(v, scale_v);
            vst1q_f32(dequant.as_mut_ptr().add(off), scaled);
        }
    }
    for i in (chunks * 4)..n {
        dequant[i] = input[i] as f32 * dequant_scale;
    }

    // Mean
    let mut sum = 0.0f32;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(dequant.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        sum += dequant[i];
    }
    let mean = sum / n as f32;

    // Variance
    let mut var_sum = 0.0f32;
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(dequant.as_ptr().add(i * 4));
            let d = vsubq_f32(v, mean_v);
            acc = vfmaq_f32(acc, d, d);
        }
        var_sum = vaddvq_f32_compat(acc);
    }
    for i in (chunks * 4)..n {
        let d = dequant[i] - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // Normalize + affine
    unsafe {
        let mean_v = vdupq_n_f32(mean);
        let inv_v = vdupq_n_f32(inv_std);
        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(dequant.as_ptr().add(off));
            let g = vld1q_f32(gamma.as_ptr().add(off));
            let b = vld1q_f32(beta.as_ptr().add(off));
            let d = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(d, inv_v);
            let scaled = vfmaq_f32(b, normed, g);
            vst1q_f32(output.as_mut_ptr().add(off), scaled);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = ((dequant[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_quantized_layer_norm(
    input: &[i8],
    dequant_scale: f32,
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    assert_eq!(n, output.len());
    assert!(n > 0);

    let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * dequant_scale).collect();
    let mean: f32 = dequant.iter().sum::<f32>() / n as f32;
    let var: f32 = dequant.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        output[i] = ((dequant[i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    /// Scalar reference RMS norm for test verification.
    fn ref_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        input.iter().zip(weight.iter()).map(|(x, w)| x * inv_rms * w).collect()
    }

    /// Scalar reference layer norm for test verification.
    fn ref_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        input
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((x, g), b)| ((x - mean) * inv_std) * g + b)
            .collect()
    }

    // -----------------------------------------------------------------------
    // 1. RMS norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_basic_4() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_basic_8() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = [1.0; 8];
        let mut output = [0.0; 8];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_non_unit_weights() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let input = [0.0; 4];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        // With eps, RMS = sqrt(eps), output should be ~0
        for v in &output {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let input = [-1.0, -2.0, -3.0, -4.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_single_element() {
        let input = [3.0];
        let weight = [2.0];
        let mut output = [0.0];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_remainder_elements() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0]; // 5 = 4+1 remainder
        let weight = [1.0; 5];
        let mut output = [0.0; 5];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, 1.0);
        let expected = ref_rms_norm(&input, &weight, 1.0);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_values() {
        let input = [1e6, 2e6, 3e6, 4e6];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, 1e-2);
    }

    #[test]
    fn test_rms_norm_16_elements() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let weight = vec![1.0; 16];
        let mut output = vec![0.0; 16];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic(expected = "input must be non-empty")]
    fn test_rms_norm_empty_panics() {
        let input: [f32; 0] = [];
        let weight: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
    }

    #[test]
    #[should_panic]
    fn test_rms_norm_mismatched_lengths() {
        let input = [1.0; 4];
        let weight = [1.0; 3];
        let mut output = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
    }

    // -----------------------------------------------------------------------
    // 2. Layer norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_norm_basic_4() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_basic_8() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let mut output = [0.0; 8];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.5, -0.5, 1.0, -1.0];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_with_scale() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [2.0, 0.5, 1.0, 3.0];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_constant_input() {
        let input = [5.0; 8];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let mut output = [0.0; 8];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        // Constant input → variance=0 → normed ≈ 0
        for v in &output {
            assert!(v.abs() < 1e-2, "expected ~0 for constant input, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_negative_input() {
        let input = [-3.0, -1.0, 1.0, 3.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_remainder_5() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = [1.0; 5];
        let beta = [0.0; 5];
        let mut output = [0.0; 5];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_single_element() {
        let input = [42.0];
        let gamma = [2.0];
        let beta = [1.0];
        let mut output = [0.0];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        // Single element: mean=42, var=0, normed≈0 → output ≈ beta
        assert!((output[0] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_layer_norm_zero_mean_input() {
        let input = [-2.0, -1.0, 1.0, 2.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_identity_transform() {
        // gamma=1, beta=0 should give standard normalized output
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let gamma = vec![1.0; 12];
        let beta = vec![0.0; 12];
        let mut output = vec![0.0; 12];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        // Sum of normalized values ≈ 0
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 1e-3, "normalized sum should be ~0, got {sum}");
    }

    #[test]
    #[should_panic]
    fn test_layer_norm_mismatched_gamma() {
        let input = [1.0; 4];
        let gamma = [1.0; 3];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
    }

    // -----------------------------------------------------------------------
    // 3. Group norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_norm_single_group() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 1, EPS);
        // Single group = full layer norm
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_two_groups() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let mut output = [0.0; 8];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 2, EPS);
        // Each group of 4 normalized independently
        let exp1 = ref_layer_norm(&input[0..4], &gamma[0..4], &beta[0..4], EPS);
        let exp2 = ref_layer_norm(&input[4..8], &gamma[4..8], &beta[4..8], EPS);
        assert_approx(&output[0..4], &exp1, TOL);
        assert_approx(&output[4..8], &exp2, TOL);
    }

    #[test]
    fn test_group_norm_four_groups() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let gamma = vec![1.0; 16];
        let beta = vec![0.0; 16];
        let mut output = vec![0.0; 16];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 4, EPS);
        for g in 0..4 {
            let s = g * 4;
            let e = s + 4;
            let exp = ref_layer_norm(&input[s..e], &gamma[s..e], &beta[s..e], EPS);
            assert_approx(&output[s..e], &exp, TOL);
        }
    }

    #[test]
    fn test_group_norm_with_bias() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = [2.0; 8];
        let beta = [1.0; 8];
        let mut output = [0.0; 8];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 2, EPS);
        for g in 0..2 {
            let s = g * 4;
            let e = s + 4;
            let exp = ref_layer_norm(&input[s..e], &gamma[s..e], &beta[s..e], EPS);
            assert_approx(&output[s..e], &exp, TOL);
        }
    }

    #[test]
    fn test_group_norm_n_equals_groups() {
        // Each element is its own group
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 4, EPS);
        // Each group has 1 element → var=0 → normed≈0 → output≈beta
        for v in &output {
            assert!(v.abs() < 1e-2);
        }
    }

    #[test]
    fn test_group_norm_remainder_group_size() {
        // Group size 3 (not divisible by 4 → remainder path)
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = [1.0; 6];
        let beta = [0.0; 6];
        let mut output = [0.0; 6];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 2, EPS);
        let exp1 = ref_layer_norm(&input[0..3], &gamma[0..3], &beta[0..3], EPS);
        let exp2 = ref_layer_norm(&input[3..6], &gamma[3..6], &beta[3..6], EPS);
        assert_approx(&output[0..3], &exp1, TOL);
        assert_approx(&output[3..6], &exp2, TOL);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn test_group_norm_indivisible_panics() {
        let input = [1.0; 7];
        let gamma = [1.0; 7];
        let beta = [0.0; 7];
        let mut output = [0.0; 7];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 3, EPS);
    }

    #[test]
    #[should_panic(expected = "num_groups must be > 0")]
    fn test_group_norm_zero_groups_panics() {
        let input = [1.0; 4];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 0, EPS);
    }

    // -----------------------------------------------------------------------
    // 4. Instance norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_norm_single_batch_single_channel() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0];
        let beta = [0.0];
        let mut output = [0.0; 4];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 1, EPS);
        let expected = ref_layer_norm(&input, &[1.0; 4], &[0.0; 4], EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_instance_norm_two_channels() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = [1.0, 2.0];
        let beta = [0.0, 1.0];
        let mut output = [0.0; 8];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 2, EPS);
        // Channel 0: scale=1, shift=0
        let ref0 = ref_layer_norm(&input[0..4], &[1.0; 4], &[0.0; 4], EPS);
        // Channel 1: scale=2, shift=1
        let ref1 = ref_layer_norm(&input[4..8], &[2.0; 4], &[1.0; 4], EPS);
        assert_approx(&output[0..4], &ref0, TOL);
        assert_approx(&output[4..8], &ref1, TOL);
    }

    #[test]
    fn test_instance_norm_two_batches() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = [1.0];
        let beta = [0.0];
        let mut output = [0.0; 8];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 2, 1, EPS);
        let ref0 = ref_layer_norm(&input[0..4], &[1.0; 4], &[0.0; 4], EPS);
        let ref1 = ref_layer_norm(&input[4..8], &[1.0; 4], &[0.0; 4], EPS);
        assert_approx(&output[0..4], &ref0, TOL);
        assert_approx(&output[4..8], &ref1, TOL);
    }

    #[test]
    fn test_instance_norm_non_aligned_channel_size() {
        // channel_size=3, not aligned to 4
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];
        let mut output = [0.0; 6];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 2, EPS);
        let ref0 = ref_layer_norm(&input[0..3], &[1.0; 3], &[0.0; 3], EPS);
        let ref1 = ref_layer_norm(&input[3..6], &[1.0; 3], &[0.0; 3], EPS);
        assert_approx(&output[0..3], &ref0, TOL);
        assert_approx(&output[3..6], &ref1, TOL);
    }

    #[test]
    fn test_instance_norm_with_scale_and_shift() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [0.5];
        let beta = [2.0];
        let mut output = [0.0; 4];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 1, EPS);
        let expected = ref_layer_norm(&input, &[0.5; 4], &[2.0; 4], EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic]
    fn test_instance_norm_bad_shape_panics() {
        let input = [1.0; 7];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut output = [0.0; 7];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 2, EPS);
    }

    // -----------------------------------------------------------------------
    // 5. Batch RMS norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_rms_norm_single_row() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 1, 4, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_rms_norm_two_rows() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 8];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 2, 4, EPS);
        let exp0 = ref_rms_norm(&input[0..4], &weight, EPS);
        let exp1 = ref_rms_norm(&input[4..8], &weight, EPS);
        assert_approx(&output[0..4], &exp0, TOL);
        assert_approx(&output[4..8], &exp1, TOL);
    }

    #[test]
    fn test_batch_rms_norm_non_aligned_dim() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = [1.0; 3];
        let mut output = [0.0; 6];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 2, 3, EPS);
        let exp0 = ref_rms_norm(&input[0..3], &weight, EPS);
        let exp1 = ref_rms_norm(&input[3..6], &weight, EPS);
        assert_approx(&output[0..3], &exp0, TOL);
        assert_approx(&output[3..6], &exp1, TOL);
    }

    #[test]
    fn test_batch_rms_norm_four_rows() {
        let input: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let weight = vec![1.0; 8];
        let mut output = vec![0.0; 32];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 4, 8, EPS);
        for b in 0..4 {
            let s = b * 8;
            let exp = ref_rms_norm(&input[s..s + 8], &weight, EPS);
            assert_approx(&output[s..s + 8], &exp, TOL);
        }
    }

    #[test]
    fn test_batch_rms_norm_with_weights() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let mut output = [0.0; 4];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 1, 4, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic]
    fn test_batch_rms_norm_bad_shape() {
        let input = [1.0; 7];
        let weight = [1.0; 4];
        let mut output = [0.0; 7];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 2, 4, EPS);
    }

    // -----------------------------------------------------------------------
    // 6. Pre-norm residual tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prenorm_residual_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let residual = [0.1, 0.2, 0.3, 0.4];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms_out = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> =
            rms_out.iter().zip(residual.iter()).map(|(r, res)| r + res).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_zero_residual() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let residual = [0.0; 4];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_8_elements() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let residual = [0.5; 8];
        let weight = [1.0; 8];
        let mut output = [0.0; 8];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms_out = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|r| r + 0.5).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_with_weights() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let residual = [1.0, 2.0, 3.0, 4.0];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms_out = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> =
            rms_out.iter().zip(residual.iter()).map(|(r, res)| r + res).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_negative_residual() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let residual = [-10.0; 4];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms_out = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|r| r - 10.0).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_remainder() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = [0.1; 5];
        let weight = [1.0; 5];
        let mut output = [0.0; 5];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms_out = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> = rms_out.iter().map(|r| r + 0.1).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic]
    fn test_prenorm_residual_mismatched_lengths() {
        let input = [1.0; 4];
        let residual = [0.0; 3];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
    }

    // -----------------------------------------------------------------------
    // 7. Adaptive layer norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_ln_zero_modulation() {
        // scale_mod=0, shift_mod=0 → same as regular layer norm
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [0.0; 4];
        let shift_mod = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_scale_modulation() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [1.0; 4]; // (1+1)=2x scale
        let shift_mod = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().map(|x| x * 2.0).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_shift_modulation() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [0.0; 4];
        let shift_mod = [5.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().map(|x| x + 5.0).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_combined_modulation() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [0.5; 4];
        let shift_mod = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().map(|x| 1.5 * x + 1.0).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_with_gamma_beta() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [2.0; 4];
        let beta = [0.5; 4];
        let scale_mod = [0.0; 4];
        let shift_mod = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_8_elements() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let scale_mod = [0.0; 8];
        let shift_mod = [0.0; 8];
        let mut output = [0.0; 8];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_negative_scale_mod() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [-0.5; 4]; // (1-0.5)=0.5x scale
        let shift_mod = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> = ln.iter().map(|x| x * 0.5).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_adaptive_ln_remainder() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = [1.0; 5];
        let beta = [0.0; 5];
        let scale_mod = [0.0; 5];
        let shift_mod = [0.0; 5];
        let mut output = [0.0; 5];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic]
    fn test_adaptive_ln_mismatched_scale_mod() {
        let input = [1.0; 4];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [0.0; 3];
        let shift_mod = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
    }

    // -----------------------------------------------------------------------
    // 8. Quantized layer norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantized_ln_basic() {
        let input: Vec<i8> = vec![10, 20, 30, 40];
        let scale = 0.1;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_negative_values() {
        let input: Vec<i8> = vec![-10, -5, 5, 10];
        let scale = 0.05;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_with_gamma_beta() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let scale = 1.0;
        let gamma = [2.0; 4];
        let beta = [0.5; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_8_elements() {
        let input: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let scale = 0.01;
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut output = vec![0.0; 8];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_extreme_values() {
        let input: Vec<i8> = vec![-128, -1, 0, 127];
        let scale = 0.01;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_all_zeros() {
        let input: Vec<i8> = vec![0; 4];
        let scale = 0.1;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        for v in &output {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn test_quantized_ln_unit_scale() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let scale = 1.0;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let float_input = [1.0f32, 2.0, 3.0, 4.0];
        let mut float_output = [0.0; 4];
        neon_fused_layer_norm(&float_input, &gamma, &beta, &mut float_output, EPS);
        assert_approx(&output, &float_output, TOL);
    }

    #[test]
    fn test_quantized_ln_remainder() {
        let input: Vec<i8> = vec![10, 20, 30, 40, 50];
        let scale = 0.1;
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let mut output = vec![0.0; 5];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    #[should_panic]
    fn test_quantized_ln_mismatched_gamma() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let gamma = [1.0; 3];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, 1.0, &gamma, &beta, &mut output, EPS);
    }

    #[test]
    fn test_quantized_ln_large_scale() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let scale = 100.0;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    // -----------------------------------------------------------------------
    // Cross-variant and property tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_idempotent_with_unit_weights() {
        // Two passes of RMS norm with unit weights should still normalize
        let input = [3.0, 4.0, 5.0, 6.0];
        let weight = [1.0; 4];
        let mut pass1 = [0.0; 4];
        let mut pass2 = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut pass1, EPS);
        neon_fused_rms_norm(&pass1, &weight, &mut pass2, EPS);
        // After second pass, RMS should be close to 1
        let rms: f32 = (pass2.iter().map(|x| x * x).sum::<f32>() / pass2.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "expected rms≈1 after two passes");
    }

    #[test]
    fn test_layer_norm_output_mean_near_beta() {
        // For gamma=1, output mean should be near mean(beta)
        let input: Vec<f32> = (1..=8).map(|x| x as f32 * 10.0).collect();
        let gamma = vec![1.0; 8];
        let beta = vec![3.0; 8];
        let mut output = vec![0.0; 8];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let out_mean: f32 = output.iter().sum::<f32>() / 8.0;
        assert!((out_mean - 3.0).abs() < 1e-3, "output mean {out_mean} should be ≈3.0");
    }

    #[test]
    fn test_batch_rms_matches_individual() {
        let row0 = [1.0, 2.0, 3.0, 4.0];
        let row1 = [5.0, 6.0, 7.0, 8.0];
        let weight = [1.0; 4];
        let mut input = vec![0.0; 8];
        input[..4].copy_from_slice(&row0);
        input[4..].copy_from_slice(&row1);
        let mut batch_out = vec![0.0; 8];
        neon_fused_batch_rms_norm(&input, &weight, &mut batch_out, 2, 4, EPS);

        let mut single0 = [0.0; 4];
        let mut single1 = [0.0; 4];
        neon_fused_rms_norm(&row0, &weight, &mut single0, EPS);
        neon_fused_rms_norm(&row1, &weight, &mut single1, EPS);
        assert_approx(&batch_out[0..4], &single0, TOL);
        assert_approx(&batch_out[4..8], &single1, TOL);
    }

    #[test]
    fn test_group_norm_single_group_matches_layer_norm() {
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut gn_out = vec![0.0; 8];
        let mut ln_out = vec![0.0; 8];
        neon_fused_group_norm(&input, &gamma, &beta, &mut gn_out, 1, EPS);
        neon_fused_layer_norm(&input, &gamma, &beta, &mut ln_out, EPS);
        assert_approx(&gn_out, &ln_out, TOL);
    }

    #[test]
    fn test_prenorm_residual_identity_residual() {
        // residual=input should give norm(input)*weight + input
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut output = [0.0; 4];
        neon_fused_prenorm_residual(&input, &input, &weight, &mut output, EPS);
        let norm = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> = norm.iter().zip(input.iter()).map(|(n, x)| n + x).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_large_vector_rms_norm() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let weight = vec![1.0; n];
        let mut output = vec![0.0; n];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_large_vector_layer_norm() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let mut output = vec![0.0; n];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_scale_invariance() {
        // RMS norm is scale invariant for uniform weights
        let input = [2.0, 4.0, 6.0, 8.0];
        let scaled: Vec<f32> = input.iter().map(|x| x * 10.0).collect();
        let weight = [1.0; 4];
        let mut out1 = [0.0; 4];
        let mut out2 = [0.0; 4];
        neon_fused_rms_norm(&input, &weight, &mut out1, EPS);
        neon_fused_rms_norm(&scaled, &weight, &mut out2, EPS);
        assert_approx(&out1, &out2, 1e-3);
    }

    #[test]
    fn test_layer_norm_translation_invariance() {
        // LayerNorm is invariant to constant shift
        let input = [1.0, 2.0, 3.0, 4.0];
        let shifted: Vec<f32> = input.iter().map(|x| x + 100.0).collect();
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut out1 = [0.0; 4];
        let mut out2 = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut out1, EPS);
        neon_fused_layer_norm(&shifted, &gamma, &beta, &mut out2, EPS);
        assert_approx(&out1, &out2, 1e-3);
    }

    #[test]
    fn test_adaptive_ln_identity_modulation() {
        // scale_mod=-1 should zero the output + shift_mod
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [-1.0; 4]; // (1+(-1))=0
        let shift_mod = [7.0; 4];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        for v in &output {
            assert!((v - 7.0).abs() < TOL, "expected 7.0, got {v}");
        }
    }

    #[test]
    fn test_instance_norm_constant_channels() {
        // Each channel is constant → normed ≈ 0
        let input = [5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0];
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];
        let mut output = [0.0; 8];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 2, EPS);
        for v in &output {
            assert!(v.abs() < 1e-2, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_quantized_ln_symmetric_input() {
        let input: Vec<i8> = vec![-2, -1, 1, 2];
        let scale = 1.0;
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut output = [0.0; 4];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        // Symmetric → mean=0, so output should be proportional to input
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_two_elements() {
        let input = [3.0, 4.0];
        let weight = [1.0, 1.0];
        let mut output = [0.0; 2];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_two_elements() {
        let input = [3.0, 7.0];
        let gamma = [1.0; 2];
        let beta = [0.0; 2];
        let mut output = [0.0; 2];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_three_elements() {
        let input = [1.0, 5.0, 9.0];
        let gamma = [1.0; 3];
        let beta = [0.0; 3];
        let mut output = [0.0; 3];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut output, EPS);
        let expected = ref_layer_norm(&input, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_negative_values() {
        let input = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let mut output = [0.0; 8];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, 2, EPS);
        let exp0 = ref_layer_norm(&input[0..4], &gamma[0..4], &beta[0..4], EPS);
        let exp1 = ref_layer_norm(&input[4..8], &gamma[4..8], &beta[4..8], EPS);
        assert_approx(&output[0..4], &exp0, TOL);
        assert_approx(&output[4..8], &exp1, TOL);
    }

    #[test]
    fn test_instance_norm_single_element_channel() {
        // channel_size=1 → var=0, normed≈0 → output≈beta
        let input = [99.0, 42.0];
        let gamma = [1.0, 1.0];
        let beta = [5.0, 7.0];
        let mut output = [0.0; 2];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, 1, 2, EPS);
        assert!((output[0] - 5.0).abs() < 1e-2);
        assert!((output[1] - 7.0).abs() < 1e-2);
    }

    #[test]
    fn test_batch_rms_norm_single_element_dim() {
        let input = [5.0, 10.0];
        let weight = [1.0];
        let mut output = [0.0; 2];
        neon_fused_batch_rms_norm(&input, &weight, &mut output, 2, 1, EPS);
        // Each row: rms = sqrt(x^2/1 + eps) ≈ |x|, output ≈ sign(x)
        for (i, &x) in input.iter().enumerate() {
            let rms = (x * x + EPS).sqrt();
            let expected = x / rms;
            assert!(
                (output[i] - expected).abs() < TOL,
                "row {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn test_quantized_ln_single_element() {
        let input: Vec<i8> = vec![42];
        let scale = 0.5;
        let gamma = [1.0];
        let beta = [3.0];
        let mut output = [0.0];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        // Single element: var=0, normed≈0, output≈beta
        assert!((output[0] - 3.0).abs() < 1e-2);
    }

    #[test]
    fn test_rms_norm_reproducibility() {
        let input = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let weight = [1.0; 8];
        let mut out1 = [0.0; 8];
        let mut out2 = [0.0; 8];
        neon_fused_rms_norm(&input, &weight, &mut out1, EPS);
        neon_fused_rms_norm(&input, &weight, &mut out2, EPS);
        assert_approx(&out1, &out2, 0.0); // exact match
    }

    #[test]
    fn test_layer_norm_reproducibility() {
        let input = [1.5, 2.5, 3.5, 4.5];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let mut out1 = [0.0; 4];
        let mut out2 = [0.0; 4];
        neon_fused_layer_norm(&input, &gamma, &beta, &mut out1, EPS);
        neon_fused_layer_norm(&input, &gamma, &beta, &mut out2, EPS);
        assert_approx(&out1, &out2, 0.0);
    }

    #[test]
    fn test_rms_norm_mixed_sign() {
        let input = [-3.0, 1.0, -2.0, 4.0, 0.0, -1.0, 5.0, -5.0];
        let weight = [1.0; 8];
        let mut output = [0.0; 8];
        neon_fused_rms_norm(&input, &weight, &mut output, EPS);
        let expected = ref_rms_norm(&input, &weight, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_large() {
        let n = 128;
        let num_groups = 8;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let mut output = vec![0.0; n];
        neon_fused_group_norm(&input, &gamma, &beta, &mut output, num_groups, EPS);
        let gs = n / num_groups;
        for g in 0..num_groups {
            let s = g * gs;
            let e = s + gs;
            let exp = ref_layer_norm(&input[s..e], &gamma[s..e], &beta[s..e], EPS);
            assert_approx(&output[s..e], &exp, TOL);
        }
    }

    #[test]
    fn test_instance_norm_large() {
        let channels = 4;
        let channel_size = 32;
        let batch = 2;
        let n = batch * channels * channel_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
        let gamma = vec![1.0; channels];
        let beta = vec![0.0; channels];
        let mut output = vec![0.0; n];
        neon_fused_instance_norm(&input, &gamma, &beta, &mut output, batch, channels, EPS);
        for b in 0..batch {
            for c in 0..channels {
                let s = (b * channels + c) * channel_size;
                let e = s + channel_size;
                let g_expanded = vec![gamma[c]; channel_size];
                let b_expanded = vec![beta[c]; channel_size];
                let exp = ref_layer_norm(&input[s..e], &g_expanded, &b_expanded, EPS);
                assert_approx(&output[s..e], &exp, TOL);
            }
        }
    }

    #[test]
    fn test_adaptive_ln_per_element_modulation() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let scale_mod = [0.0, 0.5, 1.0, -0.5];
        let shift_mod = [0.0, 1.0, -1.0, 2.0];
        let mut output = [0.0; 4];
        neon_fused_adaptive_layer_norm(
            &input,
            &gamma,
            &beta,
            &scale_mod,
            &shift_mod,
            &mut output,
            EPS,
        );
        let ln = ref_layer_norm(&input, &gamma, &beta, EPS);
        let expected: Vec<f32> =
            ln.iter().enumerate().map(|(i, x)| (1.0 + scale_mod[i]) * x + shift_mod[i]).collect();
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_ln_large() {
        let n = 64;
        let input: Vec<i8> = (0..n).map(|i| ((i as i16 * 3) % 256 - 128) as i8).collect();
        let scale = 0.02;
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let mut output = vec![0.0; n];
        neon_fused_quantized_layer_norm(&input, scale, &gamma, &beta, &mut output, EPS);
        let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * scale).collect();
        let expected = ref_layer_norm(&dequant, &gamma, &beta, EPS);
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn test_prenorm_residual_large() {
        let n = 128;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let residual: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let weight = vec![1.0; n];
        let mut output = vec![0.0; n];
        neon_fused_prenorm_residual(&input, &residual, &weight, &mut output, EPS);
        let rms = ref_rms_norm(&input, &weight, EPS);
        let expected: Vec<f32> = rms.iter().zip(residual.iter()).map(|(r, res)| r + res).collect();
        assert_approx(&output, &expected, TOL);
    }
}
