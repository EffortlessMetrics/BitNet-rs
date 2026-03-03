//! Quantized layer normalization kernels for CPU inference.
//!
//! Provides fused quantized layer norm operations that operate directly on
//! INT8 quantized tensors, avoiding unnecessary dequantize→normalize→requantize
//! round-trips. All statistical accumulators use `f64` for numerical stability.
//!
//! # Supported variants
//!
//! - **Fused quantized layer norm**: quantize → normalize → dequantize in one pass
//! - **RMS norm for quantized tensors**: root-mean-square normalization
//! - **INT8 layer norm with scale tracking**: preserves per-tensor scale metadata
//! - **Group norm for quantized tensors**: per-group normalization
//! - **Online (streaming) normalization**: single-pass mean/variance via Welford
//! - **Fused layer norm + residual add**: combined norm and residual connection
//! - **Asymmetric quantized norm**: per-channel scales and zero-points
//! - **Pre-norm / post-norm modes**: transformer block ordering variants

use bitnet_common::{BitNetError, KernelError, Result};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

/// Single-element online accumulator (Welford's algorithm).
#[derive(Debug, Clone, Copy)]
pub struct WelfordAccumulator {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl WelfordAccumulator {
    #[inline]
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    #[inline]
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value as f64 - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value as f64 - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline]
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / self.count as f64) as f32
    }

    #[inline]
    pub fn mean_f32(&self) -> f32 {
        self.mean as f32
    }
}

fn compute_mean(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &x in data {
        sum += x as f64;
    }
    (sum / data.len() as f64) as f32
}

fn compute_variance(data: &[f32], mean: f32) -> f32 {
    let mean_d = mean as f64;
    let mut sum = 0.0f64;
    for &x in data {
        let d = x as f64 - mean_d;
        sum += d * d;
    }
    (sum / data.len() as f64) as f32
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

/// Configuration for quantized layer normalization.
#[derive(Debug, Clone)]
pub struct QuantizedLayerNormConfig {
    /// Size of the normalized dimension.
    pub normalized_size: usize,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Apply learned affine transform (gamma/beta).
    pub elementwise_affine: bool,
}

impl QuantizedLayerNormConfig {
    pub fn new(normalized_size: usize) -> Self {
        Self { normalized_size, eps: 1e-5, elementwise_affine: true }
    }
}

/// Configuration for quantized group normalization.
#[derive(Debug, Clone)]
pub struct QuantizedGroupNormConfig {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub elementwise_affine: bool,
}

impl QuantizedGroupNormConfig {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self { num_groups, num_channels, eps: 1e-5, elementwise_affine: true }
    }
}

/// Normalization ordering within a transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// Normalize *before* the sublayer (Pre-LN).
    Pre,
    /// Normalize *after* the sublayer + residual (Post-LN).
    Post,
}

/// Result of INT8 layer norm that preserves quantization metadata.
#[derive(Debug, Clone)]
pub struct QuantizedNormOutput {
    /// Quantized (INT8) output values.
    pub data: Vec<i8>,
    /// Per-tensor or per-channel scale factors.
    pub scales: Vec<f32>,
    /// Per-channel zero-points (empty for symmetric quantization).
    pub zero_points: Vec<i32>,
}

// ---------------------------------------------------------------------------
// 1. Fused quantized layer norm (float in → float out, fused q/norm/dq)
// ---------------------------------------------------------------------------

/// Fused quantized layer norm: quantize input → normalize in quantized domain
/// → dequantize. Avoids materializing the full quantized tensor separately.
///
/// Returns float output after the fused operation.
pub fn fused_quantized_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        // Quantize to INT8 symmetric
        let qmax = 127.0f32;
        let abs_max = slice.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax };
        let inv_scale = 1.0 / scale;

        // Compute mean/var in quantized domain (f64 accumulators)
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &x in slice {
            let q = (x * inv_scale).round().clamp(-qmax, qmax);
            sum += q as f64;
            sum_sq += (q as f64) * (q as f64);
        }
        let mean_q = sum / n as f64;
        let var_q = sum_sq / n as f64 - mean_q * mean_q;
        let inv_std_q = 1.0 / (var_q + (config.eps / (scale * scale)) as f64).sqrt();

        // Normalize and dequantize in one pass
        for i in 0..n {
            let q = (slice[i] * inv_scale).round().clamp(-qmax, qmax) as f64;
            let normed = ((q - mean_q) * inv_std_q) as f32;
            let out = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
            output[start + i] = out;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// 2. RMS norm for quantized tensors
// ---------------------------------------------------------------------------

/// RMS normalization for quantized tensors. Operates on float input, using
/// a quantize→RMS-norm→dequantize pipeline internally.
pub fn quantized_rms_norm(
    input: &[f32],
    gamma: &[f32],
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        // Quantize and dequantize to introduce quantization noise
        let qmax = 127.0f32;
        let abs_max = slice.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax };
        let inv_scale = 1.0 / scale;

        // Roundtrip through INT8 then compute RMS in float domain
        let mut sum_sq = 0.0f64;
        for &x in slice {
            let q = (x * inv_scale).round().clamp(-qmax, qmax);
            let dq = (q * scale) as f64;
            sum_sq += dq * dq;
        }
        let rms = (sum_sq / n as f64 + config.eps as f64).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            let q = (slice[i] * inv_scale).round().clamp(-qmax, qmax);
            let dq = q * scale;
            let normed = (dq as f64 * inv_rms) as f32;
            output[start + i] = normed * gamma[i];
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// 3. INT8 layer norm with scale tracking
// ---------------------------------------------------------------------------

/// Layer norm that accepts INT8 quantized input with a per-tensor scale and
/// produces INT8 quantized output with an updated scale.
pub fn int8_layer_norm(
    input: &[i8],
    input_scale: f32,
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<QuantizedNormOutput> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(QuantizedNormOutput { data: vec![], scales: vec![], zero_points: vec![] });
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut all_data = Vec::with_capacity(input.len());
    let mut all_scales = Vec::with_capacity(batch);

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        // Dequantize → normalize → requantize
        let mut float_buf: Vec<f32> = slice.iter().map(|&v| v as f32 * input_scale).collect();

        let mean = compute_mean(&float_buf);
        let var = compute_variance(&float_buf, mean);
        let inv_std = 1.0 / (var + config.eps).sqrt();

        for i in 0..n {
            let normed = (float_buf[i] - mean) * inv_std;
            float_buf[i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }

        // Requantize to INT8 symmetric
        let qmax = 127.0f32;
        let abs_max = float_buf.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let out_scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax };
        let inv_out = 1.0 / out_scale;

        for &v in &float_buf {
            all_data.push((v * inv_out).round().clamp(-127.0, 127.0) as i8);
        }
        all_scales.push(out_scale);
    }

    Ok(QuantizedNormOutput { data: all_data, scales: all_scales, zero_points: vec![] })
}

// ---------------------------------------------------------------------------
// 4. Group norm for quantized tensors
// ---------------------------------------------------------------------------

/// Group normalization for quantized tensors. Input is float; normalization
/// happens per-group with quantization-aware statistics.
pub fn quantized_group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedGroupNormConfig,
) -> Result<Vec<f32>> {
    let c = config.num_channels;
    let g = config.num_groups;
    if c == 0 || g == 0 {
        return Err(invalid_args("num_channels and num_groups must be non-zero"));
    }
    if !c.is_multiple_of(g) {
        return Err(invalid_args("num_channels must be divisible by num_groups"));
    }
    if gamma.len() != c {
        return Err(invalid_args("gamma length must equal num_channels"));
    }
    if let Some(b) = beta
        && b.len() != c
    {
        return Err(invalid_args("beta length must equal num_channels"));
    }
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(c) {
        return Err(invalid_args("input length must be a multiple of num_channels"));
    }

    let spatial = input.len() / c;
    let channels_per_group = c / g;
    let group_size = channels_per_group * spatial;
    let mut output = vec![0.0f32; input.len()];

    // Layout: [C, spatial] – for each group, gather elements.
    for group in 0..g {
        let ch_start = group * channels_per_group;

        // Quantize the group to get quantized-domain stats.
        let mut group_vals = Vec::with_capacity(group_size);
        for ch in ch_start..ch_start + channels_per_group {
            for s in 0..spatial {
                group_vals.push(input[ch * spatial + s]);
            }
        }

        let qmax = 127.0f32;
        let abs_max = group_vals.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax };
        let inv_scale = 1.0 / scale;

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &x in &group_vals {
            let q = (x * inv_scale).round().clamp(-qmax, qmax) as f64;
            sum += q;
            sum_sq += q * q;
        }
        let n = group_vals.len() as f64;
        let mean_q = sum / n;
        let var_q = sum_sq / n - mean_q * mean_q;
        let eps_scaled = (config.eps / (scale * scale)) as f64;
        let inv_std = 1.0 / (var_q + eps_scaled).sqrt();

        let mut idx = 0;
        for ch in ch_start..ch_start + channels_per_group {
            for s in 0..spatial {
                let q = (group_vals[idx] * inv_scale).round().clamp(-qmax, qmax) as f64;
                let normed = ((q - mean_q) * inv_std) as f32;
                output[ch * spatial + s] = normed * gamma[ch] + beta.map_or(0.0, |b| b[ch]);
                idx += 1;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// 5. Online (streaming) normalization via Welford's algorithm
// ---------------------------------------------------------------------------

/// Online layer norm: compute mean/variance in a single streaming pass using
/// Welford's algorithm, then normalize. Useful when data arrives
/// incrementally or when minimizing memory passes is important.
pub fn online_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        // Single-pass Welford accumulation
        let mut acc = WelfordAccumulator::new();
        for &x in slice {
            acc.update(x);
        }

        let mean = acc.mean_f32();
        let var = acc.variance();
        let inv_std = 1.0 / (var + config.eps).sqrt();

        for i in 0..n {
            let normed = (slice[i] - mean) * inv_std;
            output[start + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// 6. Fused layer norm + residual add
// ---------------------------------------------------------------------------

/// Fused layer norm + residual addition. Computes:
///   output = LayerNorm(input + residual, gamma, beta)
///
/// The residual is added *before* normalization (pre-norm residual pattern).
pub fn fused_layer_norm_residual(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.len() != residual.len() {
        return Err(invalid_args("input and residual must have the same length"));
    }
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;

        // Fused add + stats in one pass (f64 accumulators)
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut combined = vec![0.0f32; n];
        for i in 0..n {
            let v = input[start + i] + residual[start + i];
            combined[i] = v;
            let vd = v as f64;
            sum += vd;
            sum_sq += vd * vd;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let inv_std = 1.0 / (var + config.eps as f64).sqrt();

        for i in 0..n {
            let normed = ((combined[i] as f64 - mean) * inv_std) as f32;
            output[start + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    Ok(output)
}

/// Fused layer norm + residual addition returning *both* the normalized
/// output and the un-normalized sum (input + residual). This is useful for
/// pre-norm transformer blocks where the residual stream must be carried
/// forward.
pub fn fused_layer_norm_residual_with_sum(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = config.normalized_size;
    if input.len() != residual.len() {
        return Err(invalid_args("input and residual must have the same length"));
    }
    if input.is_empty() {
        return Ok((vec![], vec![]));
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut normed_out = vec![0.0f32; input.len()];
    let mut sum_out = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;

        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let v = input[start + i] + residual[start + i];
            sum_out[start + i] = v;
            let vd = v as f64;
            sum += vd;
            sum_sq += vd * vd;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let inv_std = 1.0 / (var + config.eps as f64).sqrt();

        for i in 0..n {
            let normed = ((sum_out[start + i] as f64 - mean) * inv_std) as f32;
            normed_out[start + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    Ok((normed_out, sum_out))
}

// ---------------------------------------------------------------------------
// 7. Asymmetric quantized norm (per-channel scales + zero-points)
// ---------------------------------------------------------------------------

/// Layer norm for asymmetrically quantized (uint8) tensors with per-channel
/// scale and zero-point arrays. Returns a float output.
pub fn asymmetric_quantized_layer_norm(
    input: &[u8],
    scales: &[f32],
    zero_points: &[i32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if scales.len() != n {
        return Err(invalid_args("scales length must equal normalized_size"));
    }
    if zero_points.len() != n {
        return Err(invalid_args("zero_points length must equal normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        // Dequantize using per-channel params
        let mut float_buf = Vec::with_capacity(n);
        for i in 0..n {
            float_buf.push((slice[i] as i32 - zero_points[i]) as f32 * scales[i]);
        }

        let mean = compute_mean(&float_buf);
        let var = compute_variance(&float_buf, mean);
        let inv_std = 1.0 / (var + config.eps).sqrt();

        for i in 0..n {
            let normed = (float_buf[i] - mean) * inv_std;
            output[start + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// 8. Pre-normalization and post-normalization modes
// ---------------------------------------------------------------------------

/// Applies normalization in either pre-norm or post-norm mode.
///
/// - **Pre-norm**: `output = sublayer(LayerNorm(input)) + input`
/// - **Post-norm**: `output = LayerNorm(sublayer(input) + input)`
///
/// `sublayer_output` is the result of the sublayer applied to the (possibly
/// pre-normalized) input. The caller is responsible for running the sublayer;
/// this function handles only the normalization + residual wiring.
pub fn norm_with_mode(
    input: &[f32],
    sublayer_output: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
    mode: NormMode,
) -> Result<Vec<f32>> {
    if input.len() != sublayer_output.len() {
        return Err(invalid_args("input and sublayer_output must have the same length"));
    }

    match mode {
        NormMode::Pre => {
            // Pre-norm: sublayer was already applied to LayerNorm(input).
            // Final output = sublayer_output + input (residual stream).
            let mut output = vec![0.0f32; input.len()];
            for i in 0..input.len() {
                output[i] = sublayer_output[i] + input[i];
            }
            Ok(output)
        }
        NormMode::Post => {
            // Post-norm: output = LayerNorm(sublayer_output + input)
            fused_layer_norm_residual(sublayer_output, input, gamma, beta, config)
        }
    }
}

/// Pre-norm helper: normalizes input, returns the normalized tensor for the
/// sublayer to consume. The caller should then add the original input as a
/// residual connection after running the sublayer.
pub fn pre_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    let n = config.normalized_size;
    if input.is_empty() {
        return Ok(vec![]);
    }
    if !input.len().is_multiple_of(n) {
        return Err(invalid_args("input length must be a multiple of normalized_size"));
    }
    if gamma.len() != n {
        return Err(invalid_args("gamma length must equal normalized_size"));
    }
    if let Some(b) = beta
        && b.len() != n
    {
        return Err(invalid_args("beta length must equal normalized_size"));
    }

    let batch = input.len() / n;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch {
        let start = b * n;
        let slice = &input[start..start + n];

        let mean = compute_mean(slice);
        let var = compute_variance(slice, mean);
        let inv_std = 1.0 / (var + config.eps).sqrt();

        for i in 0..n {
            let normed = (slice[i] - mean) * inv_std;
            output[start + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
        }
    }

    Ok(output)
}

/// Post-norm helper: adds residual and normalizes. Equivalent to
/// `LayerNorm(sublayer_output + residual)`.
pub fn post_norm(
    sublayer_output: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &QuantizedLayerNormConfig,
) -> Result<Vec<f32>> {
    fused_layer_norm_residual(sublayer_output, residual, gamma, beta, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;
    const STRICT_TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn compute_rms(data: &[f32]) -> f32 {
        let mut sum = 0.0f64;
        for &x in data {
            let v = x as f64;
            sum += v * v;
        }
        (sum / data.len() as f64) as f32
    }

    /// Reference layer norm (pure float, no quantization) for comparison.
    fn reference_layer_norm(
        data: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let n = gamma.len();
        let batch = data.len() / n;
        let mut out = vec![0.0f32; data.len()];
        for b in 0..batch {
            let s = b * n;
            let mean = compute_mean(&data[s..s + n]);
            let var = compute_variance(&data[s..s + n], mean);
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..n {
                let normed = (data[s + i] - mean) * inv;
                out[s + i] = normed * gamma[i] + beta.map_or(0.0, |b| b[i]);
            }
        }
        out
    }

    /// Reference RMS norm (pure float).
    fn reference_rms_norm(data: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = gamma.len();
        let batch = data.len() / n;
        let mut out = vec![0.0f32; data.len()];
        for b in 0..batch {
            let s = b * n;
            let rms = compute_rms(&data[s..s + n]);
            let inv = 1.0 / (rms + eps).sqrt();
            for i in 0..n {
                out[s + i] = data[s + i] * inv * gamma[i];
            }
        }
        out
    }

    // =======================================================================
    // 1. Fused quantized layer norm
    // =======================================================================

    #[test]
    fn fused_qln_identity_gamma_zero_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_quantized_layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        // Quantization introduces some error – use relaxed tolerance
        assert!(approx_eq(&out, &expected, 0.1), "fused_qln mismatch: {out:?} vs {expected:?}");
    }

    #[test]
    fn fused_qln_with_affine() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let beta = vec![0.1, -0.1, 0.0];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_quantized_layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        // Just verify shapes and no NaN/Inf
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fused_qln_batch_two() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_quantized_layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        assert_eq!(out.len(), 6);
        // Each batch element should be independently normalized
        let mean1: f32 = out[0..3].iter().sum::<f32>() / 3.0;
        let mean2: f32 = out[3..6].iter().sum::<f32>() / 3.0;
        assert!(mean1.abs() < 0.2, "batch 0 mean not near zero: {mean1}");
        assert!(mean2.abs() < 0.2, "batch 1 mean not near zero: {mean2}");
    }

    #[test]
    fn fused_qln_empty_input() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_quantized_layer_norm(&[], &[1.0; 3], Some(&[0.0; 3]), &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn fused_qln_no_beta() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fused_qln_constant_input() {
        let input = vec![5.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        // Constant input → all outputs near zero (normalized)
        for &v in &out {
            assert!(v.abs() < 0.1, "constant input should normalize near 0: {v}");
        }
    }

    #[test]
    fn fused_qln_all_zeros() {
        let input = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert!(out.iter().all(|v| v.abs() < STRICT_TOL));
    }

    #[test]
    fn fused_qln_err_mismatched_gamma() {
        let cfg = QuantizedLayerNormConfig::new(4);
        let result = fused_quantized_layer_norm(&[1.0; 4], &[1.0; 3], None, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_qln_err_mismatched_beta() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let result = fused_quantized_layer_norm(&[1.0; 3], &[1.0; 3], Some(&[0.0; 2]), &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_qln_err_bad_length() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let result = fused_quantized_layer_norm(&[1.0; 5], &[1.0; 3], None, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn fused_qln_large_values() {
        let input = vec![1000.0, -1000.0, 500.0, -500.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fused_qln_negative_values() {
        let input = vec![-3.0, -2.0, -1.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        let ref_out = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &ref_out, 0.1));
    }

    #[test]
    fn fused_qln_custom_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut cfg = QuantizedLayerNormConfig::new(3);
        cfg.eps = 1e-3;
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // =======================================================================
    // 2. Quantized RMS norm
    // =======================================================================

    #[test]
    fn qrms_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qrms_matches_reference_approx() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        let ref_out = reference_rms_norm(&input, &gamma, 1e-5);
        // Quantization noise means relaxed tolerance
        assert!(approx_eq(&out, &ref_out, 0.15), "qrms mismatch: {out:?} vs {ref_out:?}");
    }

    #[test]
    fn qrms_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = quantized_rms_norm(&[], &[1.0; 3], &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn qrms_batch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn qrms_all_zeros() {
        let input = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        assert!(out.iter().all(|v| v.abs() < TOL));
    }

    #[test]
    fn qrms_with_gamma_scaling() {
        let input = vec![1.0, 1.0, 1.0];
        let gamma = vec![2.0, 0.5, 3.0];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        // With uniform input, RMS-normed value ≈ 1.0, so output ≈ gamma
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qrms_err_gamma_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(4);
        assert!(quantized_rms_norm(&[1.0; 4], &[1.0; 3], &cfg).is_err());
    }

    #[test]
    fn qrms_err_bad_length() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(quantized_rms_norm(&[1.0; 5], &[1.0; 3], &cfg).is_err());
    }

    #[test]
    fn qrms_large_values() {
        let input = vec![1e4, -1e4, 5e3, -5e3];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // =======================================================================
    // 3. INT8 layer norm with scale tracking
    // =======================================================================

    #[test]
    fn int8_ln_basic() {
        let input: Vec<i8> = vec![10, 20, 30, 40];
        let scale = 0.1;
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = int8_layer_norm(&input, scale, &gamma, None, &cfg).unwrap();
        assert_eq!(out.data.len(), 4);
        assert_eq!(out.scales.len(), 1);
    }

    #[test]
    fn int8_ln_roundtrip_sanity() {
        // Quantize → int8_layer_norm → dequantize should produce reasonable output
        let float_in = vec![1.0, 2.0, 3.0, 4.0];
        let qmax = 127.0f32;
        let abs_max = 4.0f32;
        let scale = abs_max / qmax;
        let inv = 1.0 / scale;
        let qdata: Vec<i8> = float_in.iter().map(|&v| (v * inv).round() as i8).collect();

        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = int8_layer_norm(&qdata, scale, &gamma, None, &cfg).unwrap();

        // Dequantize output
        let deq: Vec<f32> = out.data.iter().map(|&v| v as f32 * out.scales[0]).collect();

        let ref_out = reference_layer_norm(&float_in, &gamma, None, 1e-5);
        // Quantization introduces error, but should be in the right ballpark
        assert!(approx_eq(&deq, &ref_out, 0.3), "int8 roundtrip mismatch: {deq:?} vs {ref_out:?}");
    }

    #[test]
    fn int8_ln_batch_scales() {
        let input: Vec<i8> = vec![10, 20, 30, 40, 50, 60];
        let scale = 0.05;
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = int8_layer_norm(&input, scale, &gamma, None, &cfg).unwrap();
        assert_eq!(out.data.len(), 6);
        assert_eq!(out.scales.len(), 2); // one scale per batch element
    }

    #[test]
    fn int8_ln_with_beta() {
        let input: Vec<i8> = vec![10, -10, 5];
        let scale = 0.1;
        let gamma = vec![1.0; 3];
        let beta = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = int8_layer_norm(&input, scale, &gamma, Some(&beta), &cfg).unwrap();
        assert_eq!(out.data.len(), 3);
    }

    #[test]
    fn int8_ln_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = int8_layer_norm(&[], 1.0, &[1.0; 3], None, &cfg).unwrap();
        assert!(out.data.is_empty());
        assert!(out.scales.is_empty());
    }

    #[test]
    fn int8_ln_zero_input() {
        let input: Vec<i8> = vec![0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = int8_layer_norm(&input, 0.1, &[1.0; 4], None, &cfg).unwrap();
        // All zeros should produce all-zero output
        assert!(out.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn int8_ln_err_gamma_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(4);
        assert!(int8_layer_norm(&[1i8; 4], 1.0, &[1.0; 3], None, &cfg).is_err());
    }

    #[test]
    fn int8_ln_err_bad_length() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(int8_layer_norm(&[1i8; 5], 1.0, &[1.0; 3], None, &cfg).is_err());
    }

    #[test]
    fn int8_ln_scale_positive() {
        let input: Vec<i8> = vec![50, -50, 100, -100];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = int8_layer_norm(&input, 0.01, &[1.0; 4], None, &cfg).unwrap();
        assert!(out.scales.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn int8_ln_output_bounded() {
        let input: Vec<i8> = vec![127, -128, 64, -64];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = int8_layer_norm(&input, 0.05, &[1.0; 4], None, &cfg).unwrap();
        assert!(out.data.iter().all(|&v| v >= -127 && v <= 127));
    }

    // =======================================================================
    // 4. Quantized group norm
    // =======================================================================

    #[test]
    fn qgn_basic() {
        // 4 channels, 2 groups, spatial=1
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let cfg = QuantizedGroupNormConfig::new(2, 4);
        let out = quantized_group_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qgn_single_group_matches_layer_norm() {
        // 1 group = entire layer → should approximate layer norm
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let cfg_gn = QuantizedGroupNormConfig::new(1, 4);
        let cfg_ln = QuantizedLayerNormConfig::new(4);

        let gn_out = quantized_group_norm(&input, &gamma, None, &cfg_gn).unwrap();
        let ln_out = fused_quantized_layer_norm(&input, &gamma, None, &cfg_ln).unwrap();

        assert!(
            approx_eq(&gn_out, &ln_out, 0.15),
            "single-group GN should ≈ LN: {gn_out:?} vs {ln_out:?}"
        );
    }

    #[test]
    fn qgn_groups_independent() {
        // Two groups: changing one group shouldn't affect the other
        let input_a = vec![1.0, 2.0, 3.0, 4.0];
        let input_b = vec![1.0, 2.0, 30.0, 40.0]; // changed group 1
        let gamma = vec![1.0; 4];
        let cfg = QuantizedGroupNormConfig::new(2, 4);

        let out_a = quantized_group_norm(&input_a, &gamma, None, &cfg).unwrap();
        let out_b = quantized_group_norm(&input_b, &gamma, None, &cfg).unwrap();

        // Group 0 (channels 0,1) should be identical
        assert!(approx_eq(&out_a[0..2], &out_b[0..2], STRICT_TOL), "group 0 changed unexpectedly");
    }

    #[test]
    fn qgn_with_spatial() {
        // 2 channels, 1 group, spatial=2 → [C=2, S=2] total 4 elems
        let input = vec![1.0, 2.0, 3.0, 4.0]; // ch0=[1,2], ch1=[3,4]
        let gamma = vec![1.0; 2];
        let cfg = QuantizedGroupNormConfig::new(1, 2);
        let out = quantized_group_norm(&input, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn qgn_empty() {
        let cfg = QuantizedGroupNormConfig::new(2, 4);
        let out = quantized_group_norm(&[], &[1.0; 4], None, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn qgn_err_channels_not_divisible() {
        let cfg = QuantizedGroupNormConfig::new(3, 4);
        assert!(quantized_group_norm(&[1.0; 4], &[1.0; 4], None, &cfg).is_err());
    }

    #[test]
    fn qgn_err_gamma_mismatch() {
        let cfg = QuantizedGroupNormConfig::new(2, 4);
        assert!(quantized_group_norm(&[1.0; 4], &[1.0; 3], None, &cfg).is_err());
    }

    #[test]
    fn qgn_err_beta_mismatch() {
        let cfg = QuantizedGroupNormConfig::new(2, 4);
        assert!(quantized_group_norm(&[1.0; 4], &[1.0; 4], Some(&[0.0; 3]), &cfg).is_err());
    }

    #[test]
    fn qgn_err_zero_groups() {
        let cfg = QuantizedGroupNormConfig::new(0, 4);
        assert!(quantized_group_norm(&[1.0; 4], &[1.0; 4], None, &cfg).is_err());
    }

    #[test]
    fn qgn_err_zero_channels() {
        let cfg = QuantizedGroupNormConfig::new(2, 0);
        assert!(quantized_group_norm(&[1.0; 4], &[1.0; 4], None, &cfg).is_err());
    }

    // =======================================================================
    // 5. Online (streaming) normalization
    // =======================================================================

    #[test]
    fn online_ln_matches_standard() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let online = online_layer_norm(&input, &gamma, None, &cfg).unwrap();
        let standard = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(
            approx_eq(&online, &standard, TOL),
            "online vs standard: {online:?} vs {standard:?}"
        );
    }

    #[test]
    fn online_ln_with_beta() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.5; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let online = online_layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
        let standard = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&online, &standard, TOL));
    }

    #[test]
    fn online_ln_batch() {
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let online = online_layer_norm(&input, &gamma, None, &cfg).unwrap();
        let standard = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&online, &standard, TOL));
    }

    #[test]
    fn online_ln_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = online_layer_norm(&[], &[1.0; 3], None, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn online_ln_constant() {
        let input = vec![7.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = online_layer_norm(&input, &gamma, None, &cfg).unwrap();
        // All same → normalized near zero
        for &v in &out {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn online_ln_err_gamma_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(online_layer_norm(&[1.0; 3], &[1.0; 4], None, &cfg).is_err());
    }

    #[test]
    fn online_ln_err_bad_length() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(online_layer_norm(&[1.0; 5], &[1.0; 3], None, &cfg).is_err());
    }

    // =======================================================================
    // 5b. Welford accumulator unit tests
    // =======================================================================

    #[test]
    fn welford_empty() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.count, 0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn welford_single_value() {
        let mut acc = WelfordAccumulator::new();
        acc.update(5.0);
        assert!((acc.mean_f32() - 5.0).abs() < STRICT_TOL);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn welford_known_sequence() {
        let mut acc = WelfordAccumulator::new();
        for &v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            acc.update(v);
        }
        assert!((acc.mean_f32() - 5.0).abs() < TOL);
        assert!((acc.variance() - 4.0).abs() < TOL);
    }

    #[test]
    fn welford_matches_two_pass() {
        let data = vec![1.3, -2.7, 0.5, 4.1, -1.0, 3.3];
        let mut acc = WelfordAccumulator::new();
        for &v in &data {
            acc.update(v);
        }
        let expected_mean = compute_mean(&data);
        let expected_var = compute_variance(&data, expected_mean);
        assert!((acc.mean_f32() - expected_mean).abs() < STRICT_TOL);
        assert!((acc.variance() - expected_var).abs() < TOL);
    }

    // =======================================================================
    // 6. Fused layer norm + residual
    // =======================================================================

    #[test]
    fn fused_ln_res_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5, 0.5, 0.5];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, None, &cfg).unwrap();
        // Should equal LayerNorm(input + residual)
        let combined: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected = reference_layer_norm(&combined, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn fused_ln_res_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, None, &cfg).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn fused_ln_res_batch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn fused_ln_res_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_layer_norm_residual(&[], &[], &[1.0; 3], None, &cfg).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn fused_ln_res_err_length_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(fused_layer_norm_residual(&[1.0; 3], &[1.0; 4], &[1.0; 3], None, &cfg).is_err());
    }

    #[test]
    fn fused_ln_res_with_beta() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5; 3];
        let gamma = vec![1.0; 3];
        let beta = vec![0.1; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, Some(&beta), &cfg).unwrap();
        let combined: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected = reference_layer_norm(&combined, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn fused_ln_res_with_sum_returns_both() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5; 3];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let (normed, sum) =
            fused_layer_norm_residual_with_sum(&input, &residual, &gamma, None, &cfg).unwrap();

        // sum should be input + residual
        let expected_sum: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        assert!(approx_eq(&sum, &expected_sum, STRICT_TOL));

        // normed should equal LayerNorm(sum)
        let expected_normed = reference_layer_norm(&expected_sum, &gamma, None, 1e-5);
        assert!(approx_eq(&normed, &expected_normed, STRICT_TOL));
    }

    #[test]
    fn fused_ln_res_with_sum_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let (normed, sum) =
            fused_layer_norm_residual_with_sum(&[], &[], &[1.0; 3], None, &cfg).unwrap();
        assert!(normed.is_empty());
        assert!(sum.is_empty());
    }

    #[test]
    fn fused_ln_res_with_sum_err_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(
            fused_layer_norm_residual_with_sum(&[1.0; 3], &[1.0; 6], &[1.0; 3], None, &cfg)
                .is_err()
        );
    }

    // =======================================================================
    // 7. Asymmetric quantized norm
    // =======================================================================

    #[test]
    fn asym_qln_basic() {
        let input: Vec<u8> = vec![128, 140, 160, 200];
        let scales = vec![0.1; 4];
        let zero_points = vec![128; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out =
            asymmetric_quantized_layer_norm(&input, &scales, &zero_points, &gamma, None, &cfg)
                .unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn asym_qln_with_beta() {
        let input: Vec<u8> = vec![100, 150, 200];
        let scales = vec![0.05; 3];
        let zero_points = vec![128; 3];
        let gamma = vec![1.0; 3];
        let beta = vec![0.5; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = asymmetric_quantized_layer_norm(
            &input,
            &scales,
            &zero_points,
            &gamma,
            Some(&beta),
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn asym_qln_batch() {
        let input: Vec<u8> = vec![100, 150, 200, 110, 160, 210];
        let scales = vec![0.1; 3];
        let zero_points = vec![128; 3];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out =
            asymmetric_quantized_layer_norm(&input, &scales, &zero_points, &gamma, None, &cfg)
                .unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn asym_qln_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = asymmetric_quantized_layer_norm(&[], &[0.1; 3], &[128; 3], &[1.0; 3], None, &cfg)
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn asym_qln_err_scales_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(4);
        assert!(
            asymmetric_quantized_layer_norm(
                &[128u8; 4],
                &[0.1; 3],
                &[128; 4],
                &[1.0; 4],
                None,
                &cfg
            )
            .is_err()
        );
    }

    #[test]
    fn asym_qln_err_zp_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(4);
        assert!(
            asymmetric_quantized_layer_norm(
                &[128u8; 4],
                &[0.1; 4],
                &[128; 3],
                &[1.0; 4],
                None,
                &cfg
            )
            .is_err()
        );
    }

    #[test]
    fn asym_qln_err_gamma_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(4);
        assert!(
            asymmetric_quantized_layer_norm(
                &[128u8; 4],
                &[0.1; 4],
                &[128; 4],
                &[1.0; 3],
                None,
                &cfg
            )
            .is_err()
        );
    }

    #[test]
    fn asym_qln_zero_centered_input() {
        // Input at zero-point → dequantized to 0 → output near zero
        let input: Vec<u8> = vec![128; 4];
        let scales = vec![0.1; 4];
        let zero_points = vec![128; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out =
            asymmetric_quantized_layer_norm(&input, &scales, &zero_points, &gamma, None, &cfg)
                .unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "zero-centered should produce near-zero: {v}");
        }
    }

    #[test]
    fn asym_qln_per_channel_scales() {
        // Different scale per channel
        let input: Vec<u8> = vec![178, 153, 228]; // dq: 5.0, 2.5, 10.0
        let scales = vec![1.0, 0.1, 1.0];
        let zero_points = vec![128, 128, 128];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out =
            asymmetric_quantized_layer_norm(&input, &scales, &zero_points, &gamma, None, &cfg)
                .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // =======================================================================
    // 8. Pre-norm / post-norm modes
    // =======================================================================

    #[test]
    fn pre_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = pre_norm(&input, &gamma, None, &cfg).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn pre_norm_empty() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(pre_norm(&[], &[1.0; 3], None, &cfg).unwrap().is_empty());
    }

    #[test]
    fn pre_norm_batch() {
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = pre_norm(&input, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn post_norm_basic() {
        let sublayer = vec![0.5, 1.0, 1.5];
        let residual = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = post_norm(&sublayer, &residual, &gamma, None, &cfg).unwrap();
        let combined: Vec<f32> = sublayer.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected = reference_layer_norm(&combined, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn norm_with_mode_pre() {
        let input = vec![1.0, 2.0, 3.0];
        let sublayer_output = vec![0.5, 0.5, 0.5];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out =
            norm_with_mode(&input, &sublayer_output, &gamma, None, &cfg, NormMode::Pre).unwrap();
        // Pre-norm: output = sublayer_output + input
        let expected: Vec<f32> = input.iter().zip(&sublayer_output).map(|(a, b)| a + b).collect();
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn norm_with_mode_post() {
        let input = vec![1.0, 2.0, 3.0];
        let sublayer_output = vec![0.5, 0.5, 0.5];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out =
            norm_with_mode(&input, &sublayer_output, &gamma, None, &cfg, NormMode::Post).unwrap();
        // Post-norm: LayerNorm(sublayer_output + input)
        let combined: Vec<f32> = input.iter().zip(&sublayer_output).map(|(a, b)| a + b).collect();
        let expected = reference_layer_norm(&combined, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }

    #[test]
    fn norm_with_mode_err_length_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(
            norm_with_mode(&[1.0; 3], &[1.0; 4], &[1.0; 3], None, &cfg, NormMode::Pre).is_err()
        );
    }

    #[test]
    fn pre_norm_err_gamma_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(pre_norm(&[1.0; 3], &[1.0; 4], None, &cfg).is_err());
    }

    #[test]
    fn pre_norm_err_bad_length() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(pre_norm(&[1.0; 5], &[1.0; 3], None, &cfg).is_err());
    }

    // =======================================================================
    // Cross-cutting / integration tests
    // =======================================================================

    #[test]
    fn config_default_eps() {
        let cfg = QuantizedLayerNormConfig::new(10);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(cfg.elementwise_affine);
    }

    #[test]
    fn group_norm_config_default() {
        let cfg = QuantizedGroupNormConfig::new(4, 16);
        assert_eq!(cfg.num_groups, 4);
        assert_eq!(cfg.num_channels, 16);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn norm_mode_eq() {
        assert_eq!(NormMode::Pre, NormMode::Pre);
        assert_ne!(NormMode::Pre, NormMode::Post);
    }

    #[test]
    fn quantized_norm_output_empty() {
        let out = QuantizedNormOutput { data: vec![], scales: vec![], zero_points: vec![] };
        assert!(out.data.is_empty());
    }

    #[test]
    fn fused_qln_single_element() {
        let input = vec![3.0];
        let gamma = vec![1.0];
        let cfg = QuantizedLayerNormConfig::new(1);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), 1);
        // Single element normalizes to 0
        assert!(out[0].abs() < 0.1);
    }

    #[test]
    fn online_vs_fused_consistency() {
        // Online (exact float) vs fused (quantized domain) — should be close
        let input = vec![1.0, 3.0, 5.0, 7.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let online = online_layer_norm(&input, &gamma, None, &cfg).unwrap();
        let fused = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        // Fused has quantization noise, so wider tolerance
        assert!(approx_eq(&online, &fused, 0.15), "online vs fused: {online:?} vs {fused:?}");
    }

    #[test]
    fn pre_norm_then_residual_roundtrip() {
        // Simulate a pre-norm transformer block:
        // normalized = pre_norm(x)
        // sublayer_out = identity(normalized) (trivial sublayer)
        // output = sublayer_out + x
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);

        let normalized = pre_norm(&x, &gamma, None, &cfg).unwrap();
        // Identity sublayer
        let output = norm_with_mode(&x, &normalized, &gamma, None, &cfg, NormMode::Pre).unwrap();

        // output should be normalized + x
        let expected: Vec<f32> = normalized.iter().zip(&x).map(|(n, xi)| n + xi).collect();
        assert!(approx_eq(&output, &expected, STRICT_TOL));
    }

    #[test]
    fn post_norm_after_sublayer() {
        let x = vec![1.0, 2.0, 3.0];
        let sublayer = vec![0.1, 0.2, 0.3];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);

        let via_mode = norm_with_mode(&x, &sublayer, &gamma, None, &cfg, NormMode::Post).unwrap();
        let via_post = post_norm(&sublayer, &x, &gamma, None, &cfg).unwrap();

        assert!(approx_eq(&via_mode, &via_post, STRICT_TOL));
    }

    #[test]
    fn fused_ln_res_equals_manual_add_then_ln() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let residual = vec![1.0, 1.0, 1.0, 1.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);

        let fused =
            fused_layer_norm_residual(&input, &residual, &gamma, Some(&beta), &cfg).unwrap();
        let manual_sum: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let manual = reference_layer_norm(&manual_sum, &gamma, Some(&beta), 1e-5);

        assert!(approx_eq(&fused, &manual, STRICT_TOL));
    }

    #[test]
    fn int8_ln_preserves_batch_independence() {
        // Changing one batch element shouldn't affect another
        let input_a: Vec<i8> = vec![10, 20, 30, 40, 50, 60];
        let input_b: Vec<i8> = vec![10, 20, 30, 100, 100, 100]; // changed batch 1
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);

        let out_a = int8_layer_norm(&input_a, 0.1, &gamma, None, &cfg).unwrap();
        let out_b = int8_layer_norm(&input_b, 0.1, &gamma, None, &cfg).unwrap();

        // Batch 0 should be identical
        assert_eq!(out_a.data[0..3], out_b.data[0..3]);
        assert!((out_a.scales[0] - out_b.scales[0]).abs() < STRICT_TOL);
    }

    #[test]
    fn int8_ln_err_beta_mismatch() {
        let cfg = QuantizedLayerNormConfig::new(3);
        assert!(int8_layer_norm(&[1i8; 3], 1.0, &[1.0; 3], Some(&[0.0; 2]), &cfg).is_err());
    }

    #[test]
    fn fused_qln_symmetric_input() {
        // Symmetric around zero should produce symmetric-ish output
        let input = vec![-2.0, -1.0, 1.0, 2.0];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();
        // Mean of input ≈ 0, so output should preserve symmetry approximately
        assert!((out[0] + out[3]).abs() < 0.1);
        assert!((out[1] + out[2]).abs() < 0.1);
    }

    #[test]
    fn qrms_no_mean_subtraction() {
        // RMS norm doesn't subtract mean — constant positive input should not
        // collapse to zero (unlike layer norm)
        let input = vec![5.0; 4];
        let gamma = vec![1.0; 4];
        let cfg = QuantizedLayerNormConfig::new(4);
        let rms_out = quantized_rms_norm(&input, &gamma, &cfg).unwrap();
        let ln_out = fused_quantized_layer_norm(&input, &gamma, None, &cfg).unwrap();

        // Layer norm of constant → 0; RMS norm of constant → ≈ 1
        assert!(ln_out.iter().all(|v| v.abs() < 0.1));
        // RMS norm output should NOT be all zeros
        assert!(rms_out.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn asym_qln_different_zero_points() {
        let input: Vec<u8> = vec![130, 140, 150];
        let scales = vec![0.1, 0.1, 0.1];
        let zero_points = vec![128, 130, 132]; // different per channel
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out =
            asymmetric_quantized_layer_norm(&input, &scales, &zero_points, &gamma, None, &cfg)
                .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn online_ln_large_batch() {
        let n = 128;
        let batch = 8;
        let input: Vec<f32> = (0..n * batch).map(|i| (i as f32 * 0.01).sin()).collect();
        let gamma = vec![1.0; n];
        let cfg = QuantizedLayerNormConfig::new(n);
        let out = online_layer_norm(&input, &gamma, None, &cfg).unwrap();
        assert_eq!(out.len(), n * batch);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fused_ln_res_negative_residual() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![-0.5, -1.0, -1.5];
        let gamma = vec![1.0; 3];
        let cfg = QuantizedLayerNormConfig::new(3);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, None, &cfg).unwrap();
        let combined: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let expected = reference_layer_norm(&combined, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, STRICT_TOL));
    }
}
