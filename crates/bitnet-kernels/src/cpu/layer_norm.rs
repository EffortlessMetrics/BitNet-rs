//! CPU layer normalization and RMS normalization kernels.
//!
//! Provides numerically stable LayerNorm and RMSNorm on contiguous `f32`
//! slices with optional affine parameters (gamma/beta).  Supports both
//! single-sequence and batched inputs.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Error helper ───────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Shape of the normalized dimensions (product gives the
    /// normalization length per instance).
    pub normalized_shape: Vec<usize>,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub elementwise_affine: bool,
}

impl LayerNormConfig {
    /// Convenience constructor with default eps (1e-5) and affine enabled.
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self { normalized_shape, eps: 1e-5, elementwise_affine: true }
    }

    /// Total number of elements in the normalized dimensions.
    fn norm_size(&self) -> usize {
        self.normalized_shape.iter().product()
    }
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self { normalized_shape: vec![1], eps: 1e-5, elementwise_affine: true }
    }
}

// ── Layer normalization ────────────────────────────────────────────

/// Compute layer normalization over the last dimension(s).
///
/// `input` is a flat buffer of `batch_size * norm_size` elements where
/// `norm_size` is the product of `config.normalized_shape`.  Each
/// contiguous slice of `norm_size` elements is independently normalized.
///
/// When `config.elementwise_affine` is true, `gamma` (scale) is applied
/// and `beta` (shift) is optionally added.  When false, `gamma` and
/// `beta` are ignored.
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatches, empty input,
/// or non-positive/non-finite eps.
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    let norm_size = validate_layer_norm_args(input, gamma, beta, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        let mean = compute_mean(slice);
        let variance = compute_variance(slice, mean);
        let inv_std = 1.0 / (variance + config.eps).sqrt();

        if config.elementwise_affine {
            match beta {
                Some(beta) => {
                    for i in 0..norm_size {
                        out[i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
                    }
                }
                None => {
                    for i in 0..norm_size {
                        out[i] = (slice[i] - mean) * inv_std * gamma[i];
                    }
                }
            }
        } else {
            for i in 0..norm_size {
                out[i] = (slice[i] - mean) * inv_std;
            }
        }
    }

    Ok(output)
}

/// Compute RMS (root mean square) normalization.
///
/// Unlike layer norm, RMS norm does not subtract the mean — it
/// normalizes by the root mean square only:
///
///   rms_norm(x) = x / sqrt(mean(x²) + eps) * gamma
///
/// `input` layout and batching rules are the same as [`layer_norm`].
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatches, empty input,
/// or non-positive/non-finite eps.
pub fn rms_norm(input: &[f32], gamma: &[f32], config: &LayerNormConfig) -> Result<Vec<f32>> {
    let norm_size = validate_rms_norm_args(input, gamma, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        let rms = compute_rms(slice);
        let inv_rms = 1.0 / (rms + config.eps).sqrt();

        for i in 0..norm_size {
            out[i] = slice[i] * inv_rms * gamma[i];
        }
    }

    Ok(output)
}

// ── Internal helpers ───────────────────────────────────────────────

/// Mean via f64 accumulation for numerical stability.
fn compute_mean(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &x in data {
        sum += x as f64;
    }
    (sum / data.len() as f64) as f32
}

/// Variance via f64 accumulation for numerical stability.
fn compute_variance(data: &[f32], mean: f32) -> f32 {
    let mean_d = mean as f64;
    let mut sum = 0.0f64;
    for &x in data {
        let d = x as f64 - mean_d;
        sum += d * d;
    }
    (sum / data.len() as f64) as f32
}

/// Mean of squared values (for RMS norm) via f64 accumulation.
fn compute_rms(data: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &x in data {
        let v = x as f64;
        sum += v * v;
    }
    (sum / data.len() as f64) as f32
}

// ── Validation ─────────────────────────────────────────────────────

fn validate_common(input: &[f32], config: &LayerNormConfig) -> Result<usize> {
    let norm_size = config.norm_size();
    if norm_size == 0 {
        return Err(invalid_args("normalized_shape must have non-zero product"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of normalized_shape product"));
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    Ok(norm_size)
}

fn validate_layer_norm_args(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormConfig,
) -> Result<usize> {
    let norm_size = validate_common(input, config)?;
    if config.elementwise_affine {
        if gamma.len() != norm_size {
            return Err(invalid_args(&format!(
                "gamma length {} != normalized_shape product {norm_size}",
                gamma.len(),
            )));
        }
        if let Some(beta) = beta
            && beta.len() != norm_size
        {
            return Err(invalid_args(&format!(
                "beta length {} != normalized_shape product {norm_size}",
                beta.len(),
            )));
        }
    }
    Ok(norm_size)
}

fn validate_rms_norm_args(input: &[f32], gamma: &[f32], config: &LayerNormConfig) -> Result<usize> {
    let norm_size = validate_common(input, config)?;
    if gamma.len() != norm_size {
        return Err(invalid_args(&format!(
            "gamma length {} != normalized_shape product {norm_size}",
            gamma.len(),
        )));
    }
    Ok(norm_size)
}

// ── Group normalization configuration ──────────────────────────────

/// Configuration for group normalization.
///
/// Input layout is `[batch, channels, spatial…]` flattened to
/// `batch_size * num_channels * spatial_size` elements.
#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    /// Number of groups to divide channels into.
    pub num_groups: usize,
    /// Total number of channels.
    pub num_channels: usize,
    /// Spatial size (product of H×W or sequence length).
    pub spatial_size: usize,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub elementwise_affine: bool,
}

impl GroupNormConfig {
    /// Convenience constructor with default eps (1e-5) and affine enabled.
    pub fn new(num_groups: usize, num_channels: usize, spatial_size: usize) -> Self {
        Self { num_groups, num_channels, spatial_size, eps: 1e-5, elementwise_affine: true }
    }
}

// ── Group normalization ────────────────────────────────────────────

/// Compute group normalization.
///
/// `input` is a flat buffer of `batch_size * num_channels * spatial_size`
/// elements.  Channels are divided into `num_groups` groups, and each
/// group is independently normalized across `channels_per_group *
/// spatial_size` elements.
///
/// `gamma` and `beta` have length `num_channels`.
///
/// # Errors
///
/// Returns `InvalidArguments` on dimension mismatches, empty input,
/// or non-positive/non-finite eps.
pub fn group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &GroupNormConfig,
) -> Result<Vec<f32>> {
    validate_group_norm_args(input, gamma, beta, config)?;

    let nc = config.num_channels;
    let ng = config.num_groups;
    let cpg = nc / ng;
    let sp = config.spatial_size;
    let batch_size = input.len() / (nc * sp);
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        for g in 0..ng {
            let group_size = cpg * sp;

            // Mean via f64 accumulation.
            let mut sum = 0.0f64;
            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * nc * sp + c * sp;
                for s in 0..sp {
                    sum += input[off + s] as f64;
                }
            }
            let mean = (sum / group_size as f64) as f32;

            // Variance via f64 accumulation.
            let mean_d = mean as f64;
            let mut var_sum = 0.0f64;
            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * nc * sp + c * sp;
                for s in 0..sp {
                    let d = input[off + s] as f64 - mean_d;
                    var_sum += d * d;
                }
            }
            let variance = (var_sum / group_size as f64) as f32;
            let inv_std = 1.0 / (variance + config.eps).sqrt();

            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * nc * sp + c * sp;
                if config.elementwise_affine {
                    match beta {
                        Some(beta) => {
                            for s in 0..sp {
                                output[off + s] =
                                    (input[off + s] - mean) * inv_std * gamma[c] + beta[c];
                            }
                        }
                        None => {
                            for s in 0..sp {
                                output[off + s] = (input[off + s] - mean) * inv_std * gamma[c];
                            }
                        }
                    }
                } else {
                    for s in 0..sp {
                        output[off + s] = (input[off + s] - mean) * inv_std;
                    }
                }
            }
        }
    }

    Ok(output)
}

// ── Instance normalization ─────────────────────────────────────────

/// Compute instance normalization.
///
/// Instance normalization is group normalization with
/// `num_groups == num_channels`, i.e., each channel is independently
/// normalized across its spatial dimensions.
///
/// # Errors
///
/// Returns `InvalidArguments` when `num_groups != num_channels` or on
/// any dimension / eps error.
pub fn instance_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &GroupNormConfig,
) -> Result<Vec<f32>> {
    if config.num_groups != config.num_channels {
        return Err(invalid_args("instance_norm requires num_groups == num_channels"));
    }
    group_norm(input, gamma, beta, config)
}

// ── Batched convenience wrappers ───────────────────────────────────

/// Apply layer normalization to each input independently.
///
/// Each element of `inputs` is a separate sequence; they need not have
/// the same length, but each must be a multiple of `config.norm_size()`.
///
/// # Errors
///
/// Returns the first error encountered across the batch.
pub fn batch_layer_norm(
    inputs: &[&[f32]],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormConfig,
) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|inp| layer_norm(inp, gamma, beta, config)).collect()
}

/// Apply RMS normalization to each input independently.
///
/// # Errors
///
/// Returns the first error encountered across the batch.
pub fn batch_rms_norm(
    inputs: &[&[f32]],
    gamma: &[f32],
    config: &LayerNormConfig,
) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|inp| rms_norm(inp, gamma, config)).collect()
}

/// Apply group normalization to each input independently.
///
/// # Errors
///
/// Returns the first error encountered across the batch.
pub fn batch_group_norm(
    inputs: &[&[f32]],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &GroupNormConfig,
) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|inp| group_norm(inp, gamma, beta, config)).collect()
}

/// Apply instance normalization to each input independently.
///
/// # Errors
///
/// Returns the first error encountered across the batch.
pub fn batch_instance_norm(
    inputs: &[&[f32]],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &GroupNormConfig,
) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|inp| instance_norm(inp, gamma, beta, config)).collect()
}

// ── Group norm validation ──────────────────────────────────────────

fn validate_group_norm_args(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &GroupNormConfig,
) -> Result<()> {
    if config.num_groups == 0 {
        return Err(invalid_args("num_groups must be non-zero"));
    }
    if config.num_channels == 0 {
        return Err(invalid_args("num_channels must be non-zero"));
    }
    if config.spatial_size == 0 {
        return Err(invalid_args("spatial_size must be non-zero"));
    }
    if !config.num_channels.is_multiple_of(config.num_groups) {
        return Err(invalid_args("num_channels must be divisible by num_groups"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    let frame = config.num_channels * config.spatial_size;
    if !input.len().is_multiple_of(frame) {
        return Err(invalid_args("input length must be a multiple of num_channels * spatial_size"));
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if config.elementwise_affine {
        if gamma.len() != config.num_channels {
            return Err(invalid_args(&format!(
                "gamma length {} != num_channels {}",
                gamma.len(),
                config.num_channels,
            )));
        }
        if let Some(beta) = beta
            && beta.len() != config.num_channels
        {
            return Err(invalid_args(&format!(
                "beta length {} != num_channels {}",
                beta.len(),
                config.num_channels,
            )));
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// Reference layer norm (pure f64 for verification).
    fn reference_layer_norm(
        input: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let n = gamma.len();
        let batch = input.len() / n;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batch {
            let s = &input[b * n..(b + 1) * n];
            let mean: f64 = s.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            let var: f64 = s
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for i in 0..n {
                let normed = (s[i] as f64 - mean) * inv_std;
                let val = normed * gamma[i] as f64 + beta.map_or(0.0, |b| b[i] as f64);
                out[b * n + i] = val as f32;
            }
        }
        out
    }

    /// Reference RMS norm (pure f64 for verification).
    fn reference_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = gamma.len();
        let batch = input.len() / n;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batch {
            let s = &input[b * n..(b + 1) * n];
            let rms: f64 = s
                .iter()
                .map(|&x| {
                    let v = x as f64;
                    v * v
                })
                .sum::<f64>()
                / n as f64;
            let inv_rms = 1.0 / (rms + eps as f64).sqrt();
            for i in 0..n {
                out[b * n + i] = (s[i] as f64 * inv_rms * gamma[i] as f64) as f32;
            }
        }
        out
    }

    // ── Basic correctness ──────────────────────────────────

    #[test]
    fn layer_norm_uniform_input() {
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn layer_norm_known_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let config = LayerNormConfig::new(vec![5]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let beta = vec![1.0, -1.0, 0.0];
        let config = LayerNormConfig::new(vec![3]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_no_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_affine_disabled() {
        let input = vec![1.0, 3.0, 5.0];
        let gamma = vec![999.0; 3]; // should be ignored
        let mut config = LayerNormConfig::new(vec![3]);
        config.elementwise_affine = false;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let ones = vec![1.0; 3];
        let expected = reference_layer_norm(&input, &ones, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_output_zero_mean() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < TOL, "mean should be ~0, got {mean}");
    }

    #[test]
    fn layer_norm_output_unit_variance() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0; 128];
        let config = LayerNormConfig::new(vec![128]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean = out.iter().sum::<f32>() / 128.0;
        let var = out.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 128.0;
        assert!((var - 1.0).abs() < 0.01, "variance should be ~1, got {var}");
    }

    #[test]
    fn layer_norm_negative_inputs() {
        let input = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let config = LayerNormConfig::new(vec![6]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Numerical stability ────────────────────────────────

    #[test]
    fn layer_norm_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(
            approx_eq(&out, &expected, 1e-3),
            "large values: out={out:?}, expected={expected:?}",
        );
    }

    #[test]
    fn layer_norm_tiny_variance() {
        let input = vec![1.0, 1.0 + 1e-7, 1.0 - 1e-7];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn layer_norm_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    // ── Eps variations ─────────────────────────────────────

    #[test]
    fn layer_norm_custom_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormConfig::new(vec![3]);
        config.eps = 1e-3;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-3);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_large_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormConfig::new(vec![3]);
        config.eps = 1.0;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_tiny_eps() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let mut config = LayerNormConfig::new(vec![4]);
        config.eps = 1e-12;
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-12);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Batched inputs ─────────────────────────────────────

    #[test]
    fn layer_norm_batch_two_sequences() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_batch_independence() {
        let input_a = vec![1.0, 2.0, 3.0];
        let input_b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);

        let out_a = layer_norm(&input_a, &gamma, None, &config).unwrap();
        let out_b = layer_norm(&input_b, &gamma, None, &config).unwrap();

        let combined: Vec<f32> = input_a.iter().chain(input_b.iter()).copied().collect();
        let out_combined = layer_norm(&combined, &gamma, None, &config).unwrap();

        assert!(approx_eq(&out_combined[..3], &out_a, TOL));
        assert!(approx_eq(&out_combined[3..], &out_b, TOL));
    }

    #[test]
    fn layer_norm_batch_four_sequences() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let gamma = vec![1.0; 5];
        let beta = vec![0.5; 5];
        let config = LayerNormConfig::new(vec![5]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_batch_with_affine() {
        let input = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0];
        let gamma = vec![2.0, 1.0, 0.5];
        let beta = vec![0.1, 0.2, 0.3];
        let config = LayerNormConfig::new(vec![3]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── RMS normalization ──────────────────────────────────

    #[test]
    fn rms_norm_known_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn rms_norm_with_gamma() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let config = LayerNormConfig::new(vec![3]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn rms_norm_uniform_input() {
        let input = vec![3.0, 3.0, 3.0, 3.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn rms_norm_batch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn rms_norm_batch_independence() {
        let input_a = vec![1.0, 2.0, 3.0];
        let input_b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);

        let out_a = rms_norm(&input_a, &gamma, &config).unwrap();
        let out_b = rms_norm(&input_b, &gamma, &config).unwrap();

        let combined: Vec<f32> = input_a.iter().chain(input_b.iter()).copied().collect();
        let out_combined = rms_norm(&combined, &gamma, &config).unwrap();

        assert!(approx_eq(&out_combined[..3], &out_a, TOL));
        assert!(approx_eq(&out_combined[3..], &out_b, TOL));
    }

    #[test]
    fn rms_norm_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let out = rms_norm(&input, &gamma, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn rms_norm_custom_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormConfig::new(vec![3]);
        config.eps = 0.1;
        let out = rms_norm(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 0.1);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── layer_norm vs rms_norm difference ──────────────────

    #[test]
    fn layer_norm_and_rms_norm_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);
        let ln = layer_norm(&input, &gamma, None, &config).unwrap();
        let rms = rms_norm(&input, &gamma, &config).unwrap();
        assert!(!approx_eq(&ln, &rms, TOL), "LN and RMS norm should differ");
    }

    // ── Edge cases ─────────────────────────────────────────

    #[test]
    fn layer_norm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let config = LayerNormConfig::new(vec![1]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        // Single element: zero variance, so (42-42)/sqrt(eps)*2 + 1 = 1.0
        assert!((out[0] - 1.0).abs() < TOL, "expected 1.0, got {}", out[0]);
    }

    #[test]
    fn layer_norm_two_elements() {
        let input = vec![0.0, 2.0];
        let gamma = vec![1.0; 2];
        let config = LayerNormConfig::new(vec![2]);
        let out = layer_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn layer_norm_empty_returns_error() {
        let config = LayerNormConfig::new(vec![3]);
        assert!(layer_norm(&[], &[1.0; 3], None, &config).is_err());
    }

    #[test]
    fn rms_norm_empty_returns_error() {
        let config = LayerNormConfig::new(vec![3]);
        assert!(rms_norm(&[], &[1.0; 3], &config).is_err());
    }

    #[test]
    fn layer_norm_zero_eps_returns_error() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormConfig::new(vec![3]);
        config.eps = 0.0;
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn layer_norm_negative_eps_returns_error() {
        let input = vec![1.0, 2.0];
        let gamma = vec![1.0; 2];
        let mut config = LayerNormConfig::new(vec![2]);
        config.eps = -1e-5;
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn layer_norm_inf_eps_returns_error() {
        let input = vec![1.0, 2.0];
        let gamma = vec![1.0; 2];
        let mut config = LayerNormConfig::new(vec![2]);
        config.eps = f32::INFINITY;
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn layer_norm_gamma_length_mismatch() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 2]; // wrong length
        let config = LayerNormConfig::new(vec![3]);
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn layer_norm_beta_length_mismatch() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 2]; // wrong length
        let config = LayerNormConfig::new(vec![3]);
        assert!(layer_norm(&input, &gamma, Some(&beta), &config).is_err());
    }

    #[test]
    fn layer_norm_input_not_multiple_of_norm_size() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 % 3 != 0
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn rms_norm_gamma_length_mismatch() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 4]; // wrong length
        let config = LayerNormConfig::new(vec![3]);
        assert!(rms_norm(&input, &gamma, &config).is_err());
    }

    #[test]
    fn layer_norm_zero_normalized_shape_returns_error() {
        let input = vec![1.0, 2.0];
        let gamma: Vec<f32> = vec![];
        let config =
            LayerNormConfig { normalized_shape: vec![0], eps: 1e-5, elementwise_affine: true };
        assert!(layer_norm(&input, &gamma, None, &config).is_err());
    }

    // ── Multi-dimensional normalized_shape ─────────────────

    #[test]
    fn layer_norm_2d_normalized_shape() {
        // normalized_shape [2, 3] means norm_size = 6
        let input: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let config = LayerNormConfig::new(vec![2, 3]);
        let out = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Config defaults ────────────────────────────────────

    #[test]
    fn layer_norm_config_default() {
        let config = LayerNormConfig::default();
        assert_eq!(config.normalized_shape, vec![1]);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert!(config.elementwise_affine);
    }

    #[test]
    fn layer_norm_config_new() {
        let config = LayerNormConfig::new(vec![64]);
        assert_eq!(config.normalized_shape, vec![64]);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert!(config.elementwise_affine);
    }

    // ── Reference helpers for group/instance norm ──────────

    /// Reference group norm (pure f64 for verification).
    fn reference_group_norm(
        input: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        num_groups: usize,
        num_channels: usize,
        spatial_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let cpg = num_channels / num_groups;
        let batch_size = input.len() / (num_channels * spatial_size);
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batch_size {
            for g in 0..num_groups {
                let gs = cpg * spatial_size;
                let mut sum = 0.0f64;
                for c in (g * cpg)..((g + 1) * cpg) {
                    let off = b * num_channels * spatial_size + c * spatial_size;
                    for s in 0..spatial_size {
                        sum += input[off + s] as f64;
                    }
                }
                let mean = sum / gs as f64;
                let mut var_sum = 0.0f64;
                for c in (g * cpg)..((g + 1) * cpg) {
                    let off = b * num_channels * spatial_size + c * spatial_size;
                    for s in 0..spatial_size {
                        let d = input[off + s] as f64 - mean;
                        var_sum += d * d;
                    }
                }
                let var = var_sum / gs as f64;
                let inv_std = 1.0 / (var + eps as f64).sqrt();
                for c in (g * cpg)..((g + 1) * cpg) {
                    let off = b * num_channels * spatial_size + c * spatial_size;
                    for s in 0..spatial_size {
                        let normed = (input[off + s] as f64 - mean) * inv_std;
                        let val = normed * gamma[c] as f64 + beta.map_or(0.0, |b| b[c] as f64);
                        out[off + s] = val as f32;
                    }
                }
            }
        }
        out
    }

    // ── Group normalization tests ──────────────────────────

    #[test]
    fn group_norm_basic() {
        // 1 batch, 4 channels, 2 groups, spatial=3
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = GroupNormConfig::new(2, 4, 3);
        let out = group_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 2, 4, 3, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_with_affine() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let gamma = vec![2.0, 0.5, 1.0, 3.0];
        let beta = vec![1.0, -1.0, 0.5, 0.0];
        let config = GroupNormConfig::new(2, 4, 2);
        let out = group_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_no_beta() {
        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let gamma = vec![1.0; 2];
        let config = GroupNormConfig::new(1, 2, 3);
        let out = group_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, None, 1, 2, 3, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_affine_disabled() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gamma = vec![999.0; 4]; // ignored
        let mut config = GroupNormConfig::new(2, 4, 3);
        config.elementwise_affine = false;
        let out = group_norm(&input, &gamma, None, &config).unwrap();
        let ones = vec![1.0; 4];
        let expected = reference_group_norm(&input, &ones, None, 2, 4, 3, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_single_group_matches_layer_norm() {
        // 1 group over all channels is equivalent to normalizing all elements
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 6];
        let gn_config = GroupNormConfig::new(1, 6, 1);
        let gn = group_norm(&input, &gamma, None, &gn_config).unwrap();

        let ln_config = LayerNormConfig::new(vec![6]);
        let ln = layer_norm(&input, &gamma, None, &ln_config).unwrap();
        assert!(approx_eq(&gn, &ln, TOL));
    }

    #[test]
    fn group_norm_batched_two() {
        // 2 batches, 4 channels, 2 groups, spatial=2
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = GroupNormConfig::new(2, 4, 2);
        let out = group_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_batch_independence() {
        let a: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..22).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let config = GroupNormConfig::new(2, 4, 3);

        let out_a = group_norm(&a, &gamma, None, &config).unwrap();
        let out_b = group_norm(&b, &gamma, None, &config).unwrap();

        let combined: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
        let out_combined = group_norm(&combined, &gamma, None, &config).unwrap();

        assert!(approx_eq(&out_combined[..12], &out_a, TOL));
        assert!(approx_eq(&out_combined[12..], &out_b, TOL));
    }

    #[test]
    fn group_norm_custom_eps() {
        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let gamma = vec![1.0; 3];
        let mut config = GroupNormConfig::new(1, 3, 2);
        config.eps = 0.1;
        let out = group_norm(&input, &gamma, None, &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, None, 1, 3, 2, 0.1);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let gamma = vec![1.0; 2];
        let config = GroupNormConfig::new(1, 2, 2);
        let out = group_norm(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn group_norm_negative_inputs() {
        let input = vec![-3.0, -1.0, 1.0, 3.0, -2.0, 0.0, 2.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = GroupNormConfig::new(2, 4, 2);
        let out = group_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn group_norm_uniform_within_group() {
        // All values within each group are identical ⇒ normalized to 0
        let input = vec![5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = GroupNormConfig::new(2, 4, 2);
        let out = group_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn group_norm_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10, 42.0, -42.0];
        let gamma = vec![1.0; 3];
        let config = GroupNormConfig::new(1, 3, 2);
        let out = group_norm(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    // ── Group norm error cases ─────────────────────────────

    #[test]
    fn group_norm_empty_input_error() {
        let config = GroupNormConfig::new(1, 2, 3);
        assert!(group_norm(&[], &[1.0; 2], None, &config).is_err());
    }

    #[test]
    fn group_norm_channels_not_divisible_error() {
        let input = vec![1.0; 6];
        let gamma = vec![1.0; 3];
        let config = GroupNormConfig::new(2, 3, 2); // 3 % 2 != 0
        assert!(group_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn group_norm_gamma_length_mismatch_error() {
        let input = vec![1.0; 8];
        let gamma = vec![1.0; 3]; // should be 4
        let config = GroupNormConfig::new(2, 4, 2);
        assert!(group_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn group_norm_beta_length_mismatch_error() {
        let input = vec![1.0; 8];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 3]; // should be 4
        let config = GroupNormConfig::new(2, 4, 2);
        assert!(group_norm(&input, &gamma, Some(&beta), &config).is_err());
    }

    #[test]
    fn group_norm_zero_eps_error() {
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 2];
        let mut config = GroupNormConfig::new(1, 2, 2);
        config.eps = 0.0;
        assert!(group_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn group_norm_input_length_mismatch_error() {
        let input = vec![1.0; 7]; // 7 % (4 * 2) != 0
        let gamma = vec![1.0; 4];
        let config = GroupNormConfig::new(2, 4, 2);
        assert!(group_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn group_norm_zero_groups_error() {
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 2];
        let config = GroupNormConfig {
            num_groups: 0,
            num_channels: 2,
            spatial_size: 2,
            eps: 1e-5,
            elementwise_affine: true,
        };
        assert!(group_norm(&input, &gamma, None, &config).is_err());
    }

    // ── Instance normalization tests ───────────────────────

    #[test]
    fn instance_norm_basic() {
        // 1 batch, 3 channels, spatial=4 (each channel normalized independently)
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let config = GroupNormConfig::new(3, 3, 4);
        let out = instance_norm(&input, &gamma, Some(&beta), &config).unwrap();
        // instance_norm(ng=nc) ≡ group_norm(ng=nc)
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 3, 3, 4, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn instance_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = vec![2.0, 0.5];
        let beta = vec![1.0, -1.0];
        let config = GroupNormConfig::new(2, 2, 4);
        let out = instance_norm(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_group_norm(&input, &gamma, Some(&beta), 2, 2, 4, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn instance_norm_channel_independence() {
        // Each channel normalized independently ⇒ change in one channel
        // doesn't affect another.
        let input_a = vec![1.0, 2.0, 3.0, 100.0, 200.0, 300.0];
        let input_b = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0; 2];
        let config = GroupNormConfig::new(2, 2, 3);
        let out_a = instance_norm(&input_a, &gamma, None, &config).unwrap();
        let out_b = instance_norm(&input_b, &gamma, None, &config).unwrap();
        // Channel 0 (first 3 elements) should be the same.
        assert!(approx_eq(&out_a[..3], &out_b[..3], TOL));
    }

    #[test]
    fn instance_norm_rejects_wrong_groups() {
        let input = vec![1.0; 12];
        let gamma = vec![1.0; 4];
        let config = GroupNormConfig::new(2, 4, 3); // ng != nc
        assert!(instance_norm(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn instance_norm_uniform_channel() {
        // Uniform within each channel ⇒ output is 0 (+ beta)
        let input = vec![7.0, 7.0, 7.0, 3.0, 3.0, 3.0];
        let gamma = vec![1.0; 2];
        let beta = vec![0.0; 2];
        let config = GroupNormConfig::new(2, 2, 3);
        let out = instance_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn instance_norm_single_spatial() {
        // spatial_size=1 ⇒ variance=0, output = beta
        let input = vec![42.0, 99.0];
        let gamma = vec![2.0; 2];
        let beta = vec![5.0; 2];
        let config = GroupNormConfig::new(2, 2, 1);
        let out = instance_norm(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &out {
            assert!((v - 5.0).abs() < TOL, "expected 5.0, got {v}");
        }
    }

    // ── Batched wrapper tests ──────────────────────────────

    #[test]
    fn batch_layer_norm_matches_individual() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);

        let individual_a = layer_norm(&a, &gamma, None, &config).unwrap();
        let individual_b = layer_norm(&b, &gamma, None, &config).unwrap();
        let batched = batch_layer_norm(&[&a, &b], &gamma, None, &config).unwrap();

        assert_eq!(batched.len(), 2);
        assert!(approx_eq(&batched[0], &individual_a, TOL));
        assert!(approx_eq(&batched[1], &individual_b, TOL));
    }

    #[test]
    fn batch_layer_norm_empty_batch() {
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let result: Vec<&[f32]> = vec![];
        let batched = batch_layer_norm(&result, &gamma, None, &config).unwrap();
        assert!(batched.is_empty());
    }

    #[test]
    fn batch_rms_norm_matches_individual() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormConfig::new(vec![4]);

        let individual_a = rms_norm(&a, &gamma, &config).unwrap();
        let individual_b = rms_norm(&b, &gamma, &config).unwrap();
        let batched = batch_rms_norm(&[&a, &b], &gamma, &config).unwrap();

        assert_eq!(batched.len(), 2);
        assert!(approx_eq(&batched[0], &individual_a, TOL));
        assert!(approx_eq(&batched[1], &individual_b, TOL));
    }

    #[test]
    fn batch_rms_norm_empty_batch() {
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        let result: Vec<&[f32]> = vec![];
        let batched = batch_rms_norm(&result, &gamma, &config).unwrap();
        assert!(batched.is_empty());
    }

    #[test]
    fn batch_group_norm_matches_individual() {
        let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..18).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let config = GroupNormConfig::new(2, 4, 2);

        let individual_a = group_norm(&a, &gamma, None, &config).unwrap();
        let individual_b = group_norm(&b, &gamma, None, &config).unwrap();
        let batched = batch_group_norm(&[&a, &b], &gamma, None, &config).unwrap();

        assert_eq!(batched.len(), 2);
        assert!(approx_eq(&batched[0], &individual_a, TOL));
        assert!(approx_eq(&batched[1], &individual_b, TOL));
    }

    #[test]
    fn batch_instance_norm_matches_individual() {
        let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let b: Vec<f32> = (10..16).map(|i| i as f32).collect();
        let gamma = vec![1.0; 3];
        let config = GroupNormConfig::new(3, 3, 2);

        let individual_a = instance_norm(&a, &gamma, None, &config).unwrap();
        let individual_b = instance_norm(&b, &gamma, None, &config).unwrap();
        let batched = batch_instance_norm(&[&a, &b], &gamma, None, &config).unwrap();

        assert_eq!(batched.len(), 2);
        assert!(approx_eq(&batched[0], &individual_a, TOL));
        assert!(approx_eq(&batched[1], &individual_b, TOL));
    }

    #[test]
    fn batch_layer_norm_propagates_error() {
        let good = vec![1.0, 2.0, 3.0];
        let bad = vec![1.0, 2.0]; // not a multiple of norm_size=3
        let gamma = vec![1.0; 3];
        let config = LayerNormConfig::new(vec![3]);
        assert!(batch_layer_norm(&[&good, &bad], &gamma, None, &config).is_err());
    }

    // ── GroupNormConfig defaults ────────────────────────────

    #[test]
    fn group_norm_config_new() {
        let config = GroupNormConfig::new(4, 16, 8);
        assert_eq!(config.num_groups, 4);
        assert_eq!(config.num_channels, 16);
        assert_eq!(config.spatial_size, 8);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert!(config.elementwise_affine);
    }
}
