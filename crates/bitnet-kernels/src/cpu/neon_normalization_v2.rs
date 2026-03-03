//! ARM NEON-optimized normalization v2 kernels for Apple Silicon (aarch64).
//!
//! Provides six normalization operations with NEON SIMD acceleration:
//! - RMS LayerNorm (LLaMA / BitNet)
//! - Standard LayerNorm with affine transform
//! - Group normalization
//! - Batch normalization (inference mode)
//! - L2 normalization
//! - Instance normalization
//!
//! Each operation has an `unsafe fn neon_*` variant using NEON intrinsics,
//! a `fn scalar_*` fallback, and a public dispatcher that selects at runtime
//! via `is_aarch64_feature_detected!("neon")`.
//!
//! Processes 4 × f32 NEON lanes with scalar tail fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── NEON helpers ───────────────────────────────────────────────────

/// Sum of squares of `data` using NEON vfmaq_f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_of_squares(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }

        let mut sum: f32 = vaddvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let x = data[tail + i];
            sum += x * x;
        }
        sum
    }
}

/// Sum of elements using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum: f32 = vaddvq_f32(acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            sum += data[tail + i];
        }
        sum
    }
}

/// Fast inverse sqrt using vrsqrteq_f32 + one Newton–Raphson step.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_fast_inv_sqrt(val: f32) -> f32 {
    unsafe {
        let v = vdupq_n_f32(val);
        let est = vrsqrteq_f32(v);
        // Newton step: est * (3 - val * est * est) / 2
        let step = vrsqrtsq_f32(vmulq_f32(v, est), est);
        let refined = vmulq_f32(est, step);
        vgetq_lane_f32(refined, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. RMS LayerNorm
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated RMS normalization.
///
/// output[i] = weight[i] * input[i] / sqrt(mean(input²) + eps)
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n, "weight length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        let ss = neon_sum_of_squares(input);
        let mean_sq = ss / n as f32;
        let inv_rms = neon_fast_inv_sqrt(mean_sq + eps);

        let chunks = n / 4;
        let remainder = n % 4;
        let inv_rms_v = vdupq_n_f32(inv_rms);
        let in_ptr = input.as_ptr();
        let w_ptr = weight.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(in_ptr.add(off));
            let w = vld1q_f32(w_ptr.add(off));
            let normed = vmulq_f32(x, inv_rms_v);
            let scaled = vmulq_f32(w, normed);
            vst1q_f32(out_ptr.add(off), scaled);
        }

        let tail = chunks * 4;
        for i in 0..remainder {
            let idx = tail + i;
            output[idx] = weight[idx] * (input[idx] * inv_rms);
        }
    }
}

/// Scalar RMS normalization (reference / fallback).
pub fn scalar_rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n, "weight length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n {
        output[i] = weight[i] * input[i] * inv_rms;
    }
}

/// Public dispatcher for RMS normalization.
pub fn rms_norm_f32(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection passed.
            unsafe {
                neon_rms_norm_f32(input, weight, output, eps);
            }
            return;
        }
    }
    scalar_rms_norm_f32(input, weight, output, eps);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Standard LayerNorm
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated layer normalization with affine transform.
///
/// output[i] = weight[i] * (input[i] - mean) / sqrt(var + eps) + bias[i]
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n, "weight length mismatch");
    assert_eq!(bias.len(), n, "bias length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        let mean = neon_sum(input) / n as f32;

        // Variance via sum of (x - mean)²
        let chunks = n / 4;
        let remainder = n % 4;
        let mean_v = vdupq_n_f32(mean);
        let mut var_acc = vdupq_n_f32(0.0);
        let ptr = input.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let d = vsubq_f32(v, mean_v);
            var_acc = vfmaq_f32(var_acc, d, d);
        }
        let mut var_sum: f32 = vaddvq_f32(var_acc);
        let tail = chunks * 4;
        for i in 0..remainder {
            let d = input[tail + i] - mean;
            var_sum += d * d;
        }
        let variance = var_sum / n as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Normalize with affine
        let inv_std_v = vdupq_n_f32(inv_std);
        let w_ptr = weight.as_ptr();
        let b_ptr = bias.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(ptr.add(off));
            let w = vld1q_f32(w_ptr.add(off));
            let b = vld1q_f32(b_ptr.add(off));
            let centered = vsubq_f32(x, mean_v);
            let normed = vmulq_f32(centered, inv_std_v);
            let scaled = vfmaq_f32(b, w, normed); // b + w * normed
            vst1q_f32(out_ptr.add(off), scaled);
        }

        for i in 0..remainder {
            let idx = tail + i;
            let normed = (input[idx] - mean) * inv_std;
            output[idx] = weight[idx] * normed + bias[idx];
        }
    }
}

/// Scalar layer normalization (reference / fallback).
pub fn scalar_layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(weight.len(), n, "weight length mismatch");
    assert_eq!(bias.len(), n, "bias length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 =
        input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        output[i] = weight[i] * (input[i] - mean) * inv_std + bias[i];
    }
}

/// Public dispatcher for layer normalization.
pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_layer_norm_f32(input, weight, bias, output, eps);
            }
            return;
        }
    }
    scalar_layer_norm_f32(input, weight, bias, output, eps);
}

// ═══════════════════════════════════════════════════════════════════
// 3. Group Normalization
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated group normalization.
///
/// Input layout: `[channels * spatial]` (single sample).
/// Groups divide channels evenly. Each group is normalized independently.
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_group_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    num_groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(weight.len(), channels, "weight length mismatch");
    assert_eq!(bias.len(), channels, "bias length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");
    assert!(
        num_groups > 0 && channels % num_groups == 0,
        "channels must be divisible by num_groups"
    );

    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;

    for g in 0..num_groups {
        let c_start = g * channels_per_group;
        let base = c_start * spatial;
        let group_data = &input[base..base + group_size];

        // Compute mean and variance over the group.
        unsafe {
            let sum = neon_sum(group_data);
            let mean = sum / group_size as f32;

            let chunks = group_size / 4;
            let remainder = group_size % 4;
            let mean_v = vdupq_n_f32(mean);
            let mut var_acc = vdupq_n_f32(0.0);
            let ptr = group_data.as_ptr();

            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                let d = vsubq_f32(v, mean_v);
                var_acc = vfmaq_f32(var_acc, d, d);
            }
            let mut var_sum: f32 = vaddvq_f32(var_acc);
            let tail = chunks * 4;
            for i in 0..remainder {
                let d = group_data[tail + i] - mean;
                var_sum += d * d;
            }
            let variance = var_sum / group_size as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();

            // Apply normalization with per-channel affine.
            for c_off in 0..channels_per_group {
                let c = c_start + c_off;
                let w = weight[c];
                let b = bias[c];
                let off = c * spatial;
                for s in 0..spatial {
                    let idx = off + s;
                    output[idx] =
                        w * (input[idx] - mean) * inv_std + b;
                }
            }
        }
    }
}

/// Scalar group normalization (reference / fallback).
pub fn scalar_group_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    num_groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(weight.len(), channels, "weight length mismatch");
    assert_eq!(bias.len(), channels, "bias length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");
    assert!(
        num_groups > 0 && channels % num_groups == 0,
        "channels must be divisible by num_groups"
    );

    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;

    for g in 0..num_groups {
        let c_start = g * channels_per_group;
        let base = c_start * spatial;
        let group_data = &input[base..base + group_size];

        let mean: f32 = group_data.iter().sum::<f32>() / group_size as f32;
        let var: f32 = group_data
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / group_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for c_off in 0..channels_per_group {
            let c = c_start + c_off;
            let w = weight[c];
            let b = bias[c];
            let off = c * spatial;
            for s in 0..spatial {
                let idx = off + s;
                output[idx] = w * (input[idx] - mean) * inv_std + b;
            }
        }
    }
}

/// Public dispatcher for group normalization.
pub fn group_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    num_groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_group_norm_f32(
                    input, weight, bias, output, num_groups, channels,
                    spatial, eps,
                );
            }
            return;
        }
    }
    scalar_group_norm_f32(
        input, weight, bias, output, num_groups, channels, spatial, eps,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 4. Batch Normalization (inference mode)
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated batch normalization (inference mode).
///
/// Input layout: `[channels * spatial]` (single sample, NCHW flattened).
/// output[c, s] = weight[c] * (input[c, s] - mean[c]) / sqrt(var[c] + eps) + bias[c]
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_batch_norm_f32(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(mean.len(), channels, "mean length mismatch");
    assert_eq!(var.len(), channels, "var length mismatch");
    assert_eq!(weight.len(), channels, "weight length mismatch");
    assert_eq!(bias.len(), channels, "bias length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = weight[c] * inv_std;
        let shift = bias[c] - mean[c] * scale;
        let off = c * spatial;
        let chunk_data = &input[off..off + spatial];
        let chunk_out = &mut output[off..off + spatial];

        unsafe {
            let chunks = spatial / 4;
            let remainder = spatial % 4;
            let scale_v = vdupq_n_f32(scale);
            let shift_v = vdupq_n_f32(shift);
            let in_ptr = chunk_data.as_ptr();
            let out_ptr = chunk_out.as_mut_ptr();

            for i in 0..chunks {
                let o = i * 4;
                let x = vld1q_f32(in_ptr.add(o));
                let y = vfmaq_f32(shift_v, scale_v, x); // shift + scale * x
                vst1q_f32(out_ptr.add(o), y);
            }

            let tail = chunks * 4;
            for i in 0..remainder {
                chunk_out[tail + i] =
                    scale * chunk_data[tail + i] + shift;
            }
        }
    }
}

/// Scalar batch normalization (reference / fallback).
pub fn scalar_batch_norm_f32(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(mean.len(), channels, "mean length mismatch");
    assert_eq!(var.len(), channels, "var length mismatch");
    assert_eq!(weight.len(), channels, "weight length mismatch");
    assert_eq!(bias.len(), channels, "bias length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");

    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let scale = weight[c] * inv_std;
        let shift = bias[c] - mean[c] * scale;
        let off = c * spatial;
        for s in 0..spatial {
            output[off + s] = scale * input[off + s] + shift;
        }
    }
}

/// Public dispatcher for batch normalization.
pub fn batch_norm_f32(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_batch_norm_f32(
                    input, mean, var, weight, bias, output, channels,
                    spatial, eps,
                );
            }
            return;
        }
    }
    scalar_batch_norm_f32(
        input, mean, var, weight, bias, output, channels, spatial, eps,
    );
}

// ═══════════════════════════════════════════════════════════════════
// 5. L2 Normalization
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated L2 normalization.
///
/// output[i] = input[i] / sqrt(sum(input²) + eps)
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_l2_normalize_f32(
    input: &[f32],
    output: &mut [f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }

    unsafe {
        let ss = neon_sum_of_squares(input);
        let inv_norm = neon_fast_inv_sqrt(ss + eps);

        let chunks = n / 4;
        let remainder = n % 4;
        let inv_v = vdupq_n_f32(inv_norm);
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(in_ptr.add(off));
            let y = vmulq_f32(x, inv_v);
            vst1q_f32(out_ptr.add(off), y);
        }

        let tail = chunks * 4;
        for i in 0..remainder {
            output[tail + i] = input[tail + i] * inv_norm;
        }
    }
}

/// Scalar L2 normalization (reference / fallback).
pub fn scalar_l2_normalize_f32(input: &[f32], output: &mut [f32], eps: f32) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    if n == 0 {
        return;
    }
    let ss: f32 = input.iter().map(|x| x * x).sum();
    let inv_norm = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        output[i] = input[i] * inv_norm;
    }
}

/// Public dispatcher for L2 normalization.
pub fn l2_normalize_f32(input: &[f32], output: &mut [f32], eps: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_l2_normalize_f32(input, output, eps);
            }
            return;
        }
    }
    scalar_l2_normalize_f32(input, output, eps);
}

// ═══════════════════════════════════════════════════════════════════
// 6. Instance Normalization
// ═══════════════════════════════════════════════════════════════════

/// NEON-accelerated instance normalization.
///
/// Input layout: `[channels * spatial]` (single sample).
/// Each channel is normalized independently over its spatial extent.
///
/// # Safety
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_instance_norm_f32(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");

    if spatial == 0 {
        return;
    }

    for c in 0..channels {
        let off = c * spatial;
        let ch_data = &input[off..off + spatial];

        unsafe {
            let mean = neon_sum(ch_data) / spatial as f32;

            let chunks = spatial / 4;
            let remainder = spatial % 4;
            let mean_v = vdupq_n_f32(mean);
            let mut var_acc = vdupq_n_f32(0.0);
            let ptr = ch_data.as_ptr();

            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * 4));
                let d = vsubq_f32(v, mean_v);
                var_acc = vfmaq_f32(var_acc, d, d);
            }
            let mut var_sum: f32 = vaddvq_f32(var_acc);
            let tail = chunks * 4;
            for i in 0..remainder {
                let d = ch_data[tail + i] - mean;
                var_sum += d * d;
            }
            let variance = var_sum / spatial as f32;
            let inv_std = 1.0 / (variance + eps).sqrt();

            let inv_std_v = vdupq_n_f32(inv_std);
            let out_ptr = output[off..].as_mut_ptr();

            for i in 0..chunks {
                let o = i * 4;
                let v = vld1q_f32(ptr.add(o));
                let d = vsubq_f32(v, mean_v);
                let normed = vmulq_f32(d, inv_std_v);
                vst1q_f32(out_ptr.add(o), normed);
            }

            for i in 0..remainder {
                output[off + tail + i] =
                    (ch_data[tail + i] - mean) * inv_std;
            }
        }
    }
}

/// Scalar instance normalization (reference / fallback).
pub fn scalar_instance_norm_f32(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    let total = channels * spatial;
    assert_eq!(input.len(), total, "input length mismatch");
    assert_eq!(output.len(), total, "output length mismatch");

    if spatial == 0 {
        return;
    }

    for c in 0..channels {
        let off = c * spatial;
        let ch = &input[off..off + spatial];
        let mean: f32 = ch.iter().sum::<f32>() / spatial as f32;
        let var: f32 = ch
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / spatial as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for s in 0..spatial {
            output[off + s] = (ch[s] - mean) * inv_std;
        }
    }
}

/// Public dispatcher for instance normalization.
pub fn instance_norm_f32(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    spatial: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_instance_norm_f32(input, output, channels, spatial, eps);
            }
            return;
        }
    }
    scalar_instance_norm_f32(input, output, channels, spatial, eps);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    /// Reference scalar for RMS norm (independent of the module's scalar fn).
    fn ref_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let ms: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        input.iter().zip(weight).map(|(&x, &w)| w * x * inv).collect()
    }

    fn ref_layer_norm(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 =
            input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        input
            .iter()
            .zip(weight.iter().zip(bias))
            .map(|(&x, (&w, &b))| w * (x - mean) * inv + b)
            .collect()
    }

    fn ref_l2_normalize(input: &[f32], eps: f32) -> Vec<f32> {
        let ss: f32 = input.iter().map(|x| x * x).sum();
        let inv = 1.0 / (ss + eps).sqrt();
        input.iter().map(|&x| x * inv).collect()
    }

    // ── 1. RMS Norm ───────────────────────────────────────────────

    #[test]
    fn test_rms_norm_basic_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let weight = vec![1.0; 8];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 8];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_with_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 5];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_single_element() {
        let input = vec![42.0];
        let weight = vec![2.0];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 1];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_empty() {
        let mut output: Vec<f32> = vec![];
        rms_norm_f32(&[], &[], &mut output, EPS);
        assert!(output.is_empty());
    }

    #[test]
    fn test_rms_norm_non_aligned_13() {
        let input: Vec<f32> = (0..13).map(|i| i as f32 * 0.3 - 2.0).collect();
        let weight = vec![1.0; 13];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 13];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_large_1024() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let weight = vec![1.0; n];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; n];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let input = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 8];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 8];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let input = vec![0.0; 8];
        let weight = vec![1.0; 8];
        let mut output = vec![999.0; 8];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 for zero input, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_scalar_neon_parity() {
        let n = 137;
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let weight: Vec<f32> =
            (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let mut out_scalar = vec![0.0; n];
        let mut out_dispatch = vec![0.0; n];
        scalar_rms_norm_f32(&input, &weight, &mut out_scalar, EPS);
        rms_norm_f32(&input, &weight, &mut out_dispatch, EPS);
        assert_approx_eq(&out_scalar, &out_dispatch, TOL);
    }

    #[test]
    fn test_rms_norm_small_eps() {
        let input = vec![1e-10, 2e-10, 3e-10, 4e-10];
        let weight = vec![1.0; 4];
        let expected = ref_rms_norm(&input, &weight, 1e-12);
        let mut output = vec![0.0; 4];
        rms_norm_f32(&input, &weight, &mut output, 1e-12);
        assert_approx_eq(&output, &expected, 1e-3);
    }

    // ── 2. Layer Norm ─────────────────────────────────────────────

    #[test]
    fn test_layer_norm_basic_8() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 8];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let bias = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 5];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_zero_variance() {
        let input = vec![3.0; 8];
        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];
        let mut output = vec![0.0; 8];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 for const input, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_single_element() {
        let input = vec![42.0];
        let weight = vec![2.0];
        let bias = vec![1.0];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 1];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_empty() {
        let mut output: Vec<f32> = vec![];
        layer_norm_f32(&[], &[], &[], &mut output, EPS);
        assert!(output.is_empty());
    }

    #[test]
    fn test_layer_norm_non_aligned_11() {
        let input: Vec<f32> = (0..11).map(|i| i as f32 * 0.5 - 3.0).collect();
        let weight = vec![1.0; 11];
        let bias = vec![0.0; 11];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 11];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_large_1024() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let weight = vec![1.0; n];
        let bias = vec![0.0; n];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; n];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 8];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_scalar_neon_parity() {
        let n = 137;
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let weight: Vec<f32> =
            (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let bias: Vec<f32> =
            (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();
        let mut out_scalar = vec![0.0; n];
        let mut out_dispatch = vec![0.0; n];
        scalar_layer_norm_f32(&input, &weight, &bias, &mut out_scalar, EPS);
        layer_norm_f32(&input, &weight, &bias, &mut out_dispatch, EPS);
        assert_approx_eq(&out_scalar, &out_dispatch, TOL);
    }

    #[test]
    fn test_layer_norm_identity_transform() {
        // weight=1, bias=0 should give normalized output
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 1e-3, "normalized output should have ~0 mean");
    }

    // ── 3. Group Norm ─────────────────────────────────────────────

    #[test]
    fn test_group_norm_single_group() {
        // 1 group = layer norm over all channels*spatial
        let channels = 4;
        let spatial = 4;
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut output = vec![0.0; 16];
        group_norm_f32(&input, &weight, &bias, &mut output, 1, channels, spatial, EPS);
        // All elements normalized together
        let mean: f32 = input.iter().sum::<f32>() / 16.0;
        let var: f32 =
            input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 16.0;
        let inv = 1.0 / (var + EPS).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| (x - mean) * inv).collect();
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_per_channel() {
        // num_groups == channels → per-channel normalization
        let channels = 4;
        let spatial = 3;
        let input: Vec<f32> = (0..12).map(|i| i as f32 + 1.0).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut out_gn = vec![0.0; 12];
        let mut out_in = vec![0.0; 12];
        group_norm_f32(
            &input, &weight, &bias, &mut out_gn, channels, channels, spatial, EPS,
        );
        // Should match instance norm (weight=1, bias=0)
        scalar_instance_norm_f32(&input, &mut out_in, channels, spatial, EPS);
        assert_approx_eq(&out_gn, &out_in, TOL);
    }

    #[test]
    fn test_group_norm_two_groups() {
        let channels = 4;
        let spatial = 2;
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut output = vec![0.0; 8];
        group_norm_f32(&input, &weight, &bias, &mut output, 2, channels, spatial, EPS);
        // Verify each group normalized independently
        let mut expected = vec![0.0; 8];
        scalar_group_norm_f32(
            &input, &weight, &bias, &mut expected, 2, channels, spatial, EPS,
        );
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_with_affine() {
        let channels = 4;
        let spatial = 3;
        let input: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let weight = vec![2.0, 0.5, 1.0, 3.0];
        let bias = vec![0.1, -0.1, 0.0, 0.5];
        let mut output = vec![0.0; 12];
        let mut expected = vec![0.0; 12];
        scalar_group_norm_f32(
            &input, &weight, &bias, &mut expected, 2, channels, spatial, EPS,
        );
        group_norm_f32(&input, &weight, &bias, &mut output, 2, channels, spatial, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_group_norm_scalar_neon_parity() {
        let channels = 8;
        let spatial = 7;
        let n = channels * spatial;
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 11 + 5) % 200) as f32 * 0.05 - 5.0).collect();
        let weight: Vec<f32> = (0..channels).map(|i| 0.3 + i as f32 * 0.2).collect();
        let bias: Vec<f32> = (0..channels).map(|i| -0.5 + i as f32 * 0.1).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_group_norm_f32(
            &input, &weight, &bias, &mut out_s, 4, channels, spatial, EPS,
        );
        group_norm_f32(&input, &weight, &bias, &mut out_d, 4, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_group_norm_large() {
        let channels = 32;
        let spatial = 16;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2.5).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_group_norm_f32(
            &input, &weight, &bias, &mut out_s, 8, channels, spatial, EPS,
        );
        group_norm_f32(&input, &weight, &bias, &mut out_d, 8, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_group_norm_spatial_1() {
        let channels = 4;
        let spatial = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut out_s = vec![0.0; 4];
        let mut out_d = vec![0.0; 4];
        scalar_group_norm_f32(&input, &weight, &bias, &mut out_s, 2, channels, spatial, EPS);
        group_norm_f32(&input, &weight, &bias, &mut out_d, 2, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    // ── 4. Batch Norm ─────────────────────────────────────────────

    #[test]
    fn test_batch_norm_basic() {
        let channels = 3;
        let spatial = 4;
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mean = vec![0.0; 3];
        let var = vec![1.0; 3];
        let weight = vec![1.0; 3];
        let bias = vec![0.0; 3];
        let mut output = vec![0.0; 12];
        let mut expected = vec![0.0; 12];
        scalar_batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut expected, channels, spatial, EPS,
        );
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut output, channels, spatial, EPS,
        );
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_with_stats() {
        let channels = 2;
        let spatial = 4;
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let mean = vec![1.0, 2.0];
        let var = vec![0.5, 2.0];
        let weight = vec![2.0, 0.5];
        let bias = vec![0.1, -0.1];
        let mut output = vec![0.0; 8];
        let mut expected = vec![0.0; 8];
        scalar_batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut expected, channels, spatial, EPS,
        );
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut output, channels, spatial, EPS,
        );
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_identity() {
        // mean=0, var=1, weight=1, bias=0 → identity
        let channels = 2;
        let spatial = 4;
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mean = vec![0.0; 2];
        let var = vec![1.0; 2];
        let weight = vec![1.0; 2];
        let bias = vec![0.0; 2];
        let mut output = vec![0.0; 8];
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut output, channels, spatial, EPS,
        );
        // With var=1 and mean=0, output ≈ input * inv_sqrt(1+eps)
        let inv = 1.0 / (1.0 + EPS).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| x * inv).collect();
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_scalar_neon_parity() {
        let channels = 6;
        let spatial = 11;
        let n = channels * spatial;
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 13 + 7) % 200) as f32 * 0.05 - 5.0).collect();
        let mean: Vec<f32> = (0..channels).map(|i| i as f32 * 0.5 - 1.0).collect();
        let var: Vec<f32> = (0..channels).map(|i| 0.5 + i as f32 * 0.3).collect();
        let weight: Vec<f32> = (0..channels).map(|i| 0.8 + i as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..channels).map(|i| -0.2 + i as f32 * 0.1).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_s, channels, spatial, EPS,
        );
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_d, channels, spatial, EPS,
        );
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_batch_norm_large() {
        let channels = 16;
        let spatial = 64;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let mean = vec![0.0; channels];
        let var = vec![1.0; channels];
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_s, channels, spatial, EPS,
        );
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_d, channels, spatial, EPS,
        );
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_batch_norm_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mean = vec![2.5];
        let var = vec![1.25];
        let weight = vec![1.0];
        let bias = vec![0.0];
        let mut out_s = vec![0.0; 4];
        let mut out_d = vec![0.0; 4];
        scalar_batch_norm_f32(&input, &mean, &var, &weight, &bias, &mut out_s, 1, 4, EPS);
        batch_norm_f32(&input, &mean, &var, &weight, &bias, &mut out_d, 1, 4, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_batch_norm_high_variance() {
        let channels = 2;
        let spatial = 4;
        let input: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, -100.0, -200.0, -300.0, -400.0];
        let mean = vec![250.0, -250.0];
        let var = vec![12500.0, 12500.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let mut out_s = vec![0.0; 8];
        let mut out_d = vec![0.0; 8];
        scalar_batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_s, channels, spatial, EPS,
        );
        batch_norm_f32(
            &input, &mean, &var, &weight, &bias, &mut out_d, channels, spatial, EPS,
        );
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    // ── 5. L2 Normalize ───────────────────────────────────────────

    #[test]
    fn test_l2_normalize_basic() {
        let input = vec![3.0, 4.0];
        let mut output = vec![0.0; 2];
        l2_normalize_f32(&input, &mut output, EPS);
        // norm = 5.0
        assert_approx_eq(&output, &[0.6, 0.8], TOL);
    }

    #[test]
    fn test_l2_normalize_unit_vector() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0; 4];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &[1.0, 0.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn test_l2_normalize_all_equal() {
        let input = vec![2.0; 4];
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; 4];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_l2_normalize_single() {
        let input = vec![5.0];
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; 1];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_l2_normalize_empty() {
        let mut output: Vec<f32> = vec![];
        l2_normalize_f32(&[], &mut output, EPS);
        assert!(output.is_empty());
    }

    #[test]
    fn test_l2_normalize_non_aligned_7() {
        let input: Vec<f32> = (1..=7).map(|i| i as f32).collect();
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; 7];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_l2_normalize_large_256() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; n];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_l2_normalize_near_zero() {
        let input = vec![1e-20, 2e-20, 3e-20, 4e-20];
        let mut output = vec![0.0; 4];
        l2_normalize_f32(&input, &mut output, EPS);
        // Should not produce NaN/Inf
        for &v in &output {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }

    #[test]
    fn test_l2_normalize_scalar_neon_parity() {
        let n = 137;
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_l2_normalize_f32(&input, &mut out_s, EPS);
        l2_normalize_f32(&input, &mut out_d, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_l2_normalize_negative_values() {
        let input = vec![-3.0, -4.0];
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; 2];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── 6. Instance Norm ──────────────────────────────────────────

    #[test]
    fn test_instance_norm_basic() {
        let channels = 2;
        let spatial = 4;
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out_s = vec![0.0; 8];
        let mut out_d = vec![0.0; 8];
        scalar_instance_norm_f32(&input, &mut out_s, channels, spatial, EPS);
        instance_norm_f32(&input, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_instance_norm_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out_s = vec![0.0; 8];
        let mut out_d = vec![0.0; 8];
        scalar_instance_norm_f32(&input, &mut out_s, 1, 8, EPS);
        instance_norm_f32(&input, &mut out_d, 1, 8, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_instance_norm_constant_channel() {
        // One channel constant, one varying
        let input = vec![5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 8];
        instance_norm_f32(&input, &mut output, 2, 4, EPS);
        // Constant channel should → ~0
        for &v in &output[0..4] {
            assert!(v.abs() < TOL, "expected ~0 for constant channel, got {v}");
        }
    }

    #[test]
    fn test_instance_norm_non_aligned_spatial() {
        let channels = 3;
        let spatial = 5;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_instance_norm_f32(&input, &mut out_s, channels, spatial, EPS);
        instance_norm_f32(&input, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_instance_norm_large() {
        let channels = 8;
        let spatial = 64;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2.5).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_instance_norm_f32(&input, &mut out_s, channels, spatial, EPS);
        instance_norm_f32(&input, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_instance_norm_zero_mean_output() {
        let input = vec![1.0, 3.0, 5.0, 7.0];
        let mut output = vec![0.0; 4];
        instance_norm_f32(&input, &mut output, 1, 4, EPS);
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 1e-3, "instance norm output should sum to ~0");
    }

    #[test]
    fn test_instance_norm_negative_values() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let mut out_s = vec![0.0; 8];
        let mut out_d = vec![0.0; 8];
        scalar_instance_norm_f32(&input, &mut out_s, 2, 4, EPS);
        instance_norm_f32(&input, &mut out_d, 2, 4, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_instance_norm_many_channels() {
        let channels = 16;
        let spatial = 1;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_instance_norm_f32(&input, &mut out_s, channels, spatial, EPS);
        instance_norm_f32(&input, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    // ── Cross-operation parity ────────────────────────────────────

    #[test]
    fn test_group_norm_equals_instance_norm_when_groups_eq_channels() {
        let channels = 4;
        let spatial = 8;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0; channels];
        let bias = vec![0.0; channels];
        let mut out_gn = vec![0.0; n];
        let mut out_in = vec![0.0; n];
        group_norm_f32(
            &input, &weight, &bias, &mut out_gn, channels, channels, spatial, EPS,
        );
        instance_norm_f32(&input, &mut out_in, channels, spatial, EPS);
        assert_approx_eq(&out_gn, &out_in, TOL);
    }

    #[test]
    fn test_rms_norm_scale_invariance() {
        // RMS norm is scale-invariant: rms_norm(alpha * x) == rms_norm(x)
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scaled: Vec<f32> = input.iter().map(|&x| x * 100.0).collect();
        let weight = vec![1.0; 8];
        let mut out1 = vec![0.0; 8];
        let mut out2 = vec![0.0; 8];
        rms_norm_f32(&input, &weight, &mut out1, EPS);
        rms_norm_f32(&scaled, &weight, &mut out2, EPS);
        assert_approx_eq(&out1, &out2, TOL);
    }

    #[test]
    fn test_layer_norm_shift_invariance() {
        // Layer norm is shift-invariant: layer_norm(x + c) == layer_norm(x)
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shifted: Vec<f32> = input.iter().map(|&x| x + 1000.0).collect();
        let weight = vec![1.0; 8];
        let bias = vec![0.0; 8];
        let mut out1 = vec![0.0; 8];
        let mut out2 = vec![0.0; 8];
        layer_norm_f32(&input, &weight, &bias, &mut out1, EPS);
        layer_norm_f32(&shifted, &weight, &bias, &mut out2, EPS);
        assert_approx_eq(&out1, &out2, 1e-2);
    }

    #[test]
    fn test_l2_normalize_output_unit_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        l2_normalize_f32(&input, &mut output, EPS);
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "L2 normalized vector should have unit norm, got {norm}"
        );
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_two_elements() {
        let input = vec![3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 2];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_two_elements() {
        let input = vec![3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 2];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rms_norm_exactly_4_elements() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let expected = ref_rms_norm(&input, &weight, EPS);
        let mut output = vec![0.0; 4];
        rms_norm_f32(&input, &weight, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layer_norm_exactly_4_elements() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let expected = ref_layer_norm(&input, &weight, &bias, EPS);
        let mut output = vec![0.0; 4];
        layer_norm_f32(&input, &weight, &bias, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_spatial_1() {
        let channels = 4;
        let spatial = 1;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mean = vec![0.0; 4];
        let var = vec![1.0; 4];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut out_s = vec![0.0; 4];
        let mut out_d = vec![0.0; 4];
        scalar_batch_norm_f32(&input, &mean, &var, &weight, &bias, &mut out_s, channels, spatial, EPS);
        batch_norm_f32(&input, &mean, &var, &weight, &bias, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }

    #[test]
    fn test_l2_normalize_two_elements() {
        let input = vec![3.0, 4.0];
        let expected = ref_l2_normalize(&input, EPS);
        let mut output = vec![0.0; 2];
        l2_normalize_f32(&input, &mut output, EPS);
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_instance_norm_spatial_8_aligned() {
        let channels = 2;
        let spatial = 8;
        let n = channels * spatial;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut out_s = vec![0.0; n];
        let mut out_d = vec![0.0; n];
        scalar_instance_norm_f32(&input, &mut out_s, channels, spatial, EPS);
        instance_norm_f32(&input, &mut out_d, channels, spatial, EPS);
        assert_approx_eq(&out_s, &out_d, TOL);
    }
}
