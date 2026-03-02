//! Scalar (portable) fallback implementations.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Compute mean of a slice.
#[inline]
pub fn mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    // Kahan summation for numerical stability.
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in data {
        let y = f64::from(v) - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    (sum / data.len() as f64) as f32
}

/// Compute mean of squared values.
#[inline]
pub fn mean_sq_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in data {
        let sq = f64::from(v) * f64::from(v);
        let y = sq - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    (sum / data.len() as f64) as f32
}

/// Compute variance (population) given a pre-computed mean.
#[inline]
pub fn variance_f32(data: &[f32], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let m = f64::from(mean);
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in data {
        let diff = f64::from(v) - m;
        let sq = diff * diff;
        let y = sq - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    (sum / data.len() as f64) as f32
}

/// `LayerNorm`: `(x - mean) / sqrt(var + eps)`
pub fn layer_norm_f32(data: &mut [f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let m = mean_f32(data);
    let v = variance_f32(data, m);
    let inv_std = 1.0 / (v + epsilon).sqrt();
    for x in data.iter_mut() {
        *x = (*x - m) * inv_std;
    }
}

/// `LayerNorm` fused with affine: `gamma * (x - mean) / sqrt(var + eps) + beta`
pub fn layer_norm_affine_f32(data: &mut [f32], gamma: &[f32], beta: &[f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let m = mean_f32(data);
    let v = variance_f32(data, m);
    let inv_std = 1.0 / (v + epsilon).sqrt();
    for (i, x) in data.iter_mut().enumerate() {
        *x = (gamma[i] * (*x - m)).mul_add(inv_std, beta[i]);
    }
}

/// `RMSNorm`: `x / sqrt(mean(x^2) + eps)`
pub fn rms_norm_f32(data: &mut [f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let ms = mean_sq_f32(data);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();
    for x in data.iter_mut() {
        *x *= inv_rms;
    }
}

/// `RMSNorm` fused with scale: `gamma * x / sqrt(mean(x^2) + eps)`
pub fn rms_norm_scale_f32(data: &mut [f32], gamma: &[f32], epsilon: f32) {
    if data.is_empty() {
        return;
    }
    let ms = mean_sq_f32(data);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();
    for (i, x) in data.iter_mut().enumerate() {
        *x = gamma[i] * *x * inv_rms;
    }
}

/// `GroupNorm` on a single sample: `data` has shape `[num_groups * channels_per_group]`.
/// Each group is normalized independently, then affine-transformed.
pub fn group_norm_f32(
    data: &mut [f32],
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    let total = data.len();
    if total == 0 || num_groups == 0 {
        return;
    }
    let cpg = total / num_groups;
    for g in 0..num_groups {
        let start = g * cpg;
        let end = start + cpg;
        let group = &data[start..end];
        let m = mean_f32(group);
        let v = variance_f32(group, m);
        let inv_std = 1.0 / (v + epsilon).sqrt();
        for (j, val) in data[start..end].iter_mut().enumerate() {
            let ch = start + j;
            *val = (gamma[ch] * (*val - m)).mul_add(inv_std, beta[ch]);
        }
    }
}
