//! Scalar fallback implementations for normalization operations.

#![allow(clippy::cast_precision_loss)] // Intentional: dimension lengths fit f64/f32 for our use

/// Compute mean of a slice using Kahan summation.
pub fn mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &val in data {
        let y_val = f64::from(val) - comp;
        let t_val = sum + y_val;
        comp = (t_val - sum) - y_val;
        sum = t_val;
    }
    #[allow(clippy::cast_possible_truncation)]
    let result = (sum / data.len() as f64) as f32;
    result
}

/// Compute variance of a slice given a precomputed mean.
pub fn variance(data: &[f32], mean_val: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean_f64 = f64::from(mean_val);
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &val in data {
        let diff = f64::from(val) - mean_f64;
        let y_val = diff.mul_add(diff, -comp);
        let t_val = sum + y_val;
        comp = (t_val - sum) - y_val;
        sum = t_val;
    }
    #[allow(clippy::cast_possible_truncation)]
    let result = (sum / data.len() as f64) as f32;
    result
}

/// Compute mean of squares for `RMSNorm`.
pub fn mean_of_squares(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &val in data {
        let fv = f64::from(val);
        let sq = fv * fv;
        let y_val = sq - comp;
        let t_val = sum + y_val;
        comp = (t_val - sum) - y_val;
        sum = t_val;
    }
    #[allow(clippy::cast_possible_truncation)]
    let result = (sum / data.len() as f64) as f32;
    result
}

/// Scalar `LayerNorm`: `gamma * (x - mean) / sqrt(var + eps) + beta`
pub fn layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], epsilon: f32, output: &mut [f32]) {
    let mu = mean(input);
    let var = variance(input, mu);
    let inv_std = 1.0 / (var + epsilon).sqrt();
    for idx in 0..input.len() {
        output[idx] = (gamma[idx] * (input[idx] - mu)).mul_add(inv_std, beta[idx]);
    }
}

/// Scalar `RMSNorm`: `gamma * x / sqrt(mean(x^2) + eps)`
pub fn rms_norm(input: &[f32], gamma: &[f32], epsilon: f32, output: &mut [f32]) {
    let ms = mean_of_squares(input);
    let inv_rms = 1.0 / (ms + epsilon).sqrt();
    for idx in 0..input.len() {
        output[idx] = gamma[idx] * input[idx] * inv_rms;
    }
}

/// Scalar `BatchNorm`: `gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
pub fn batch_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    for idx in 0..input.len() {
        let inv_std = 1.0 / (running_var[idx] + epsilon).sqrt();
        output[idx] = (gamma[idx] * (input[idx] - running_mean[idx])).mul_add(inv_std, beta[idx]);
    }
}
