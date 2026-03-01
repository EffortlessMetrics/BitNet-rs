//! Normalization layer implementations for neural network inference.
//!
//! Provides `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`, and `InstanceNorm`
//! with configurable affine parameters and epsilon.

use crate::HalError;

// ── Types ─────────────────────────────────────────────────────────────────

/// Supported normalization algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormType {
    /// Standard layer normalization (Ba et al., 2016).
    LayerNorm,
    /// Root-mean-square normalization (Zhang & Sennrich, 2019).
    RmsNorm,
    /// Group normalization with a configurable number of groups.
    GroupNorm { num_groups: usize },
    /// Batch normalization using pre-computed running statistics.
    BatchNorm,
    /// Instance normalization (per-instance, per-channel).
    InstanceNorm,
}

/// Configuration for a normalization layer.
#[derive(Debug, Clone)]
pub struct NormConfig {
    /// Which normalization algorithm to use.
    pub norm_type: NormType,
    /// Shape of the normalized dimensions (e.g. `[hidden_dim]`).
    pub normalized_shape: Vec<usize>,
    /// Small constant added to the denominator for numerical stability.
    pub eps: f64,
    /// Whether learnable affine parameters (weight/bias) are applied.
    pub elementwise_affine: bool,
}

/// Learnable affine parameters for a normalization layer.
#[derive(Debug, Clone)]
pub struct NormParams {
    /// Per-element scale (gamma). `None` means identity scaling.
    pub weight: Option<Vec<f32>>,
    /// Per-element shift (beta). `None` means no shift.
    pub bias: Option<Vec<f32>>,
}

/// Output of a normalization forward pass.
#[derive(Debug, Clone)]
pub struct NormOutput {
    /// The normalized (and optionally affine-transformed) data.
    pub normalized: Vec<f32>,
    /// Mean of the input (when computed by the algorithm).
    pub mean: Option<f32>,
    /// Variance of the input (when computed by the algorithm).
    pub variance: Option<f32>,
}

/// A normalization layer combining config and parameters.
#[derive(Debug, Clone)]
pub struct NormLayer {
    pub config: NormConfig,
    pub params: NormParams,
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Compute the arithmetic mean of `data`.
pub fn compute_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = data.len() as f32;
    data.iter().copied().sum::<f32>() / n
}

/// Compute the population variance of `data` given its `mean`.
pub fn compute_variance(data: &[f32], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = data.len() as f32;
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n
}

// ── Standalone functions ──────────────────────────────────────────────────

/// Apply layer normalization to `input`.
///
/// Normalizes so that mean ≈ 0 and variance ≈ 1, then applies
/// optional weight/bias affine transform.
pub fn layer_norm(
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f64,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean = compute_mean(input);
    let var = compute_variance(input, mean);
    #[allow(clippy::cast_possible_truncation)]
    let inv_std = 1.0 / (f64::from(var) + eps).sqrt() as f32;
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = (x - mean) * inv_std;
            let scaled = weight.map_or(normed, |w| normed * w[i]);
            bias.map_or(scaled, |b| scaled + b[i])
        })
        .collect()
}

/// Apply RMS normalization to `input`.
///
/// Scales so that the root-mean-square ≈ 1, then multiplies by `weight`.
pub fn rms_norm(input: &[f32], weight: Option<&[f32]>, eps: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = input.len() as f32;
    let ss: f32 = input.iter().map(|&v| v * v).sum::<f32>() / n;
    #[allow(clippy::cast_possible_truncation)]
    let inv_rms = 1.0 / (f64::from(ss) + eps).sqrt() as f32;
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = x * inv_rms;
            weight.map_or(normed, |w| normed * w[i])
        })
        .collect()
}

/// Apply group normalization to `input`.
///
/// `input` has `C` channels (total length = C × spatial).
/// Channels are split into `num_groups` groups, each normalized
/// independently.
pub fn group_norm(
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    num_groups: usize,
    eps: f64,
) -> Vec<f32> {
    if input.is_empty() || num_groups == 0 {
        return input.to_vec();
    }
    let n = input.len();
    let group_size = n / num_groups;
    if group_size == 0 || !n.is_multiple_of(num_groups) {
        return input.to_vec();
    }
    let mut out = vec![0.0_f32; n];
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group = &input[start..end];
        let mean = compute_mean(group);
        let var = compute_variance(group, mean);
        #[allow(clippy::cast_possible_truncation)]
        let inv_std = 1.0 / (f64::from(var) + eps).sqrt() as f32;
        for i in start..end {
            let normed = (input[i] - mean) * inv_std;
            let scaled = weight.map_or(normed, |w| normed * w[i]);
            out[i] = bias.map_or(scaled, |b| scaled + b[i]);
        }
    }
    out
}

/// Apply batch normalization using pre-computed running statistics.
pub fn batch_norm(
    input: &[f32],
    running_mean: f32,
    running_var: f32,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f64,
) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    #[allow(clippy::cast_possible_truncation)]
    let inv_std = 1.0 / (f64::from(running_var) + eps).sqrt() as f32;
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = (x - running_mean) * inv_std;
            let scaled = weight.map_or(normed, |w| normed * w[i]);
            bias.map_or(scaled, |b| scaled + b[i])
        })
        .collect()
}

/// Apply instance normalization.
///
/// `shape` is `(channels, spatial_size)`. Each channel is normalized
/// independently over the spatial dimension.
pub fn instance_norm(input: &[f32], shape: (usize, usize), eps: f64) -> Vec<f32> {
    let (channels, spatial) = shape;
    if input.is_empty() || channels == 0 || spatial == 0 {
        return input.to_vec();
    }
    let expected = channels * spatial;
    if input.len() != expected {
        return input.to_vec();
    }
    let mut out = vec![0.0_f32; expected];
    for c in 0..channels {
        let start = c * spatial;
        let end = start + spatial;
        let slice = &input[start..end];
        let mean = compute_mean(slice);
        let var = compute_variance(slice, mean);
        #[allow(clippy::cast_possible_truncation)]
        let inv_std = 1.0 / (f64::from(var) + eps).sqrt() as f32;
        for i in start..end {
            out[i] = (input[i] - mean) * inv_std;
        }
    }
    out
}

// ── NormLayer ─────────────────────────────────────────────────────────────

impl NormLayer {
    /// Create a new normalization layer.
    pub fn new(config: NormConfig, params: NormParams) -> Result<Self, HalError> {
        let dim: usize = config.normalized_shape.iter().product();
        if config.elementwise_affine {
            if let Some(ref w) = params.weight
                && w.len() != dim
            {
                return Err(HalError::ShapeMismatch { expected: dim, actual: w.len() });
            }
            if let Some(ref b) = params.bias
                && b.len() != dim
            {
                return Err(HalError::ShapeMismatch { expected: dim, actual: b.len() });
            }
        }
        Ok(Self { config, params })
    }

    /// Run the normalization forward pass on `input`.
    ///
    /// `shape` is only needed for `InstanceNorm` (channels, spatial).
    pub fn forward(
        &self,
        input: &[f32],
        shape: Option<(usize, usize)>,
    ) -> Result<NormOutput, HalError> {
        if input.is_empty() {
            return Err(HalError::EmptyInput);
        }
        let w = self.params.weight.as_deref();
        let b = self.params.bias.as_deref();
        let eps = self.config.eps;

        match &self.config.norm_type {
            NormType::LayerNorm => {
                let mean = compute_mean(input);
                let var = compute_variance(input, mean);
                let normalized = layer_norm(input, w, b, eps);
                Ok(NormOutput { normalized, mean: Some(mean), variance: Some(var) })
            }
            NormType::RmsNorm => {
                let normalized = rms_norm(input, w, eps);
                Ok(NormOutput { normalized, mean: None, variance: None })
            }
            NormType::GroupNorm { num_groups } => {
                let normalized = group_norm(input, w, b, *num_groups, eps);
                Ok(NormOutput { normalized, mean: None, variance: None })
            }
            NormType::BatchNorm => {
                let mean = compute_mean(input);
                let var = compute_variance(input, mean);
                let normalized = batch_norm(input, mean, var, w, b, eps);
                Ok(NormOutput { normalized, mean: Some(mean), variance: Some(var) })
            }
            NormType::InstanceNorm => {
                let s = shape.unwrap_or((1, input.len()));
                let normalized = instance_norm(input, s, eps);
                Ok(NormOutput { normalized, mean: None, variance: None })
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::suboptimal_flops,
        clippy::cast_precision_loss
    )]

    use super::*;

    const EPS: f64 = 1e-5;
    const TOL: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── compute_mean / compute_variance ──────────────────────────────

    #[test]
    fn test_compute_mean_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(compute_mean(&data), 3.0, TOL));
    }

    #[test]
    fn test_compute_mean_empty() {
        assert_eq!(compute_mean(&[]), 0.0);
    }

    #[test]
    fn test_compute_mean_single() {
        assert!(approx_eq(compute_mean(&[42.0]), 42.0, TOL));
    }

    #[test]
    fn test_compute_variance_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = compute_mean(&data);
        assert!(approx_eq(compute_variance(&data, mean), 2.0, TOL));
    }

    #[test]
    fn test_compute_variance_empty() {
        assert_eq!(compute_variance(&[], 0.0), 0.0);
    }

    #[test]
    fn test_compute_variance_constant() {
        let data = [5.0, 5.0, 5.0, 5.0];
        let mean = compute_mean(&data);
        assert!(approx_eq(compute_variance(&data, mean), 0.0, TOL));
    }

    // ── LayerNorm ────────────────────────────────────────────────────

    #[test]
    fn test_layer_norm_zero_mean() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = layer_norm(&input, None, None, EPS);
        let mean = compute_mean(&out);
        assert!(approx_eq(mean, 0.0, 1e-3), "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = layer_norm(&input, None, None, EPS);
        let m = compute_mean(&out);
        let v = compute_variance(&out, m);
        assert!(approx_eq(v, 1.0, 1e-3), "variance should be ~1, got {v}");
    }

    #[test]
    fn test_layer_norm_with_weight() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [2.0, 2.0, 2.0, 2.0];
        let out = layer_norm(&input, Some(&weight), None, EPS);
        let no_w = layer_norm(&input, None, None, EPS);
        for (a, b) in out.iter().zip(no_w.iter()) {
            assert!(approx_eq(*a, *b * 2.0, TOL));
        }
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let bias = [1.0, 1.0, 1.0, 1.0];
        let out = layer_norm(&input, None, Some(&bias), EPS);
        let no_b = layer_norm(&input, None, None, EPS);
        for (a, b) in out.iter().zip(no_b.iter()) {
            assert!(approx_eq(*a, *b + 1.0, TOL));
        }
    }

    #[test]
    fn test_layer_norm_with_weight_and_bias() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [2.0, 2.0, 2.0, 2.0];
        let bias = [0.5, 0.5, 0.5, 0.5];
        let out = layer_norm(&input, Some(&weight), Some(&bias), EPS);
        let no_affine = layer_norm(&input, None, None, EPS);
        for (a, b) in out.iter().zip(no_affine.iter()) {
            assert!(approx_eq(*a, *b * 2.0 + 0.5, TOL));
        }
    }

    #[test]
    fn test_layer_norm_empty_input() {
        let out = layer_norm(&[], None, None, EPS);
        assert!(out.is_empty());
    }

    #[test]
    fn test_layer_norm_single_element() {
        let out = layer_norm(&[5.0], None, None, EPS);
        assert_eq!(out.len(), 1);
        // Single element: (5 - 5) / sqrt(0 + eps) = 0
        assert!(approx_eq(out[0], 0.0, TOL));
    }

    #[test]
    fn test_layer_norm_all_same_values() {
        let input = [3.0; 8];
        let out = layer_norm(&input, None, None, EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }

    #[test]
    fn test_layer_norm_all_zeros() {
        let input = [0.0; 4];
        let out = layer_norm(&input, None, None, EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let input = [-3.0, -1.0, 1.0, 3.0];
        let out = layer_norm(&input, None, None, EPS);
        let mean = compute_mean(&out);
        assert!(approx_eq(mean, 0.0, 1e-3));
    }

    #[test]
    fn test_layer_norm_large_values_stability() {
        let input = [1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let out = layer_norm(&input, None, None, EPS);
        let mean = compute_mean(&out);
        assert!(approx_eq(mean, 0.0, 0.1), "numerical stability: mean={mean}");
    }

    // ── RMSNorm ──────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_unit_rms() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let out = rms_norm(&input, None, EPS);
        #[allow(clippy::cast_precision_loss)]
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!(approx_eq(rms, 1.0, 1e-3), "RMS should be ~1, got {rms}");
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [2.0, 2.0, 2.0, 2.0];
        let out = rms_norm(&input, Some(&weight), EPS);
        let no_w = rms_norm(&input, None, EPS);
        for (a, b) in out.iter().zip(no_w.iter()) {
            assert!(approx_eq(*a, *b * 2.0, TOL));
        }
    }

    #[test]
    fn test_rms_norm_empty_input() {
        let out = rms_norm(&[], None, EPS);
        assert!(out.is_empty());
    }

    #[test]
    fn test_rms_norm_single_element() {
        let out = rms_norm(&[3.0], None, EPS);
        assert_eq!(out.len(), 1);
        // RMS of single value 3.0 is 3.0; normalized = 3/3 = ~1
        assert!(approx_eq(out[0], 1.0, 1e-3));
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let input = [0.0; 4];
        let out = rms_norm(&input, None, EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }

    #[test]
    fn test_rms_norm_all_same_values() {
        let input = [5.0; 4];
        let out = rms_norm(&input, None, EPS);
        // All equal → each output ≈ 1.0
        for &v in &out {
            assert!(approx_eq(v, 1.0, 1e-3));
        }
    }

    #[test]
    fn test_rms_norm_negative_values() {
        let input = [-2.0, -1.0, 1.0, 2.0];
        let out = rms_norm(&input, None, EPS);
        #[allow(clippy::cast_precision_loss)]
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!(approx_eq(rms, 1.0, 1e-3));
    }

    #[test]
    fn test_rms_norm_reference_values() {
        // Hand-computed: input = [1, 2, 3], RMS = sqrt((1+4+9)/3) ≈ 2.1602
        let input = [1.0, 2.0, 3.0];
        let out = rms_norm(&input, None, EPS);
        let expected_rms = (14.0_f32 / 3.0).sqrt();
        assert!(approx_eq(out[0], 1.0 / expected_rms, 1e-3));
        assert!(approx_eq(out[1], 2.0 / expected_rms, 1e-3));
        assert!(approx_eq(out[2], 3.0 / expected_rms, 1e-3));
    }

    #[test]
    fn test_rms_norm_large_values_stability() {
        let input = [1e6, 1e6 + 1.0, 1e6 + 2.0];
        let out = rms_norm(&input, None, EPS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── GroupNorm ────────────────────────────────────────────────────

    #[test]
    fn test_group_norm_two_groups() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let out = group_norm(&input, None, None, 2, EPS);
        // First group [1,2,3,4], second group [10,20,30,40]
        let g1 = &out[0..4];
        let g2 = &out[4..8];
        let m1 = compute_mean(g1);
        let m2 = compute_mean(g2);
        assert!(approx_eq(m1, 0.0, 1e-3));
        assert!(approx_eq(m2, 0.0, 1e-3));
    }

    #[test]
    fn test_group_norm_single_group_equals_layer_norm() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gn = group_norm(&input, None, None, 1, EPS);
        let ln = layer_norm(&input, None, None, EPS);
        for (a, b) in gn.iter().zip(ln.iter()) {
            assert!(approx_eq(*a, *b, TOL));
        }
    }

    #[test]
    fn test_group_norm_each_group_unit_variance() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = group_norm(&input, None, None, 2, EPS);
        for g in 0..2 {
            let start = g * 3;
            let group = &out[start..start + 3];
            let m = compute_mean(group);
            let v = compute_variance(group, m);
            assert!(approx_eq(v, 1.0, 1e-2), "group {g} variance = {v}");
        }
    }

    #[test]
    fn test_group_norm_with_affine() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [2.0, 2.0, 2.0, 2.0];
        let bias = [1.0, 1.0, 1.0, 1.0];
        let out = group_norm(&input, Some(&weight), Some(&bias), 2, EPS);
        let no_affine = group_norm(&input, None, None, 2, EPS);
        for (a, b) in out.iter().zip(no_affine.iter()) {
            assert!(approx_eq(*a, *b * 2.0 + 1.0, TOL));
        }
    }

    #[test]
    fn test_group_norm_empty() {
        let out = group_norm(&[], None, None, 2, EPS);
        assert!(out.is_empty());
    }

    #[test]
    fn test_group_norm_groups_independent() {
        // Changing one group shouldn't affect the other.
        let input_a = [1.0, 2.0, 100.0, 200.0];
        let input_b = [1.0, 2.0, 0.0, 0.0];
        let out_a = group_norm(&input_a, None, None, 2, EPS);
        let out_b = group_norm(&input_b, None, None, 2, EPS);
        // First group should be the same in both.
        assert!(approx_eq(out_a[0], out_b[0], TOL));
        assert!(approx_eq(out_a[1], out_b[1], TOL));
    }

    // ── BatchNorm ────────────────────────────────────────────────────

    #[test]
    fn test_batch_norm_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let var = 2.0;
        let out = batch_norm(&input, mean, var, None, None, EPS);
        let expected_inv_std = 1.0 / (2.0_f64 + EPS).sqrt() as f32;
        for (i, &x) in input.iter().enumerate() {
            let expected = (x - mean) * expected_inv_std;
            assert!(approx_eq(out[i], expected, TOL));
        }
    }

    #[test]
    fn test_batch_norm_with_affine() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [2.0, 2.0, 2.0, 2.0];
        let bias = [0.5, 0.5, 0.5, 0.5];
        let out = batch_norm(&input, 2.5, 1.25, Some(&weight), Some(&bias), EPS);
        let no_affine = batch_norm(&input, 2.5, 1.25, None, None, EPS);
        for (a, b) in out.iter().zip(no_affine.iter()) {
            assert!(approx_eq(*a, *b * 2.0 + 0.5, TOL));
        }
    }

    #[test]
    fn test_batch_norm_empty() {
        let out = batch_norm(&[], 0.0, 1.0, None, None, EPS);
        assert!(out.is_empty());
    }

    #[test]
    fn test_batch_norm_zero_variance() {
        let input = [5.0, 5.0, 5.0];
        let out = batch_norm(&input, 5.0, 0.0, None, None, EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }

    #[test]
    fn test_batch_norm_running_stats() {
        // Simulate running mean/var computed over a batch.
        let batch = [1.0, 2.0, 3.0, 4.0, 5.0];
        let running_mean = compute_mean(&batch);
        let running_var = compute_variance(&batch, running_mean);
        let out = batch_norm(&batch, running_mean, running_var, None, None, EPS);
        let m = compute_mean(&out);
        assert!(approx_eq(m, 0.0, 1e-3));
    }

    // ── InstanceNorm ─────────────────────────────────────────────────

    #[test]
    fn test_instance_norm_per_channel() {
        // 2 channels, 3 spatial elements each
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let out = instance_norm(&input, (2, 3), EPS);
        // Each channel normalized independently.
        let c1 = &out[0..3];
        let c2 = &out[3..6];
        assert!(approx_eq(compute_mean(c1), 0.0, 1e-3));
        assert!(approx_eq(compute_mean(c2), 0.0, 1e-3));
    }

    #[test]
    fn test_instance_norm_single_channel() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = instance_norm(&input, (1, 4), EPS);
        let ln = layer_norm(&input, None, None, EPS);
        for (a, b) in out.iter().zip(ln.iter()) {
            assert!(approx_eq(*a, *b, TOL));
        }
    }

    #[test]
    fn test_instance_norm_empty() {
        let out = instance_norm(&[], (0, 0), EPS);
        assert!(out.is_empty());
    }

    #[test]
    fn test_instance_norm_mismatched_shape() {
        // Shape doesn't match: should return input unchanged.
        let input = [1.0, 2.0, 3.0];
        let out = instance_norm(&input, (2, 3), EPS);
        assert_eq!(out, input);
    }

    #[test]
    fn test_instance_norm_each_channel_unit_variance() {
        let input = [1.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0];
        let out = instance_norm(&input, (2, 4), EPS);
        for c in 0..2 {
            let s = c * 4;
            let ch = &out[s..s + 4];
            let m = compute_mean(ch);
            let v = compute_variance(ch, m);
            assert!(approx_eq(v, 1.0, 1e-2), "ch {c} var = {v}");
        }
    }

    // ── Epsilon ──────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_prevents_div_by_zero_layer_norm() {
        let input = [0.0; 4];
        let out = layer_norm(&input, None, None, EPS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_epsilon_prevents_div_by_zero_rms_norm() {
        let input = [0.0; 4];
        let out = rms_norm(&input, None, EPS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_epsilon_prevents_div_by_zero_batch_norm() {
        let out = batch_norm(&[5.0, 5.0], 5.0, 0.0, None, None, EPS);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_tiny_epsilon() {
        let input = [1.0, 2.0, 3.0];
        let out = layer_norm(&input, None, None, 1e-12);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── NormLayer (unified API) ──────────────────────────────────────

    #[test]
    fn test_norm_layer_layer_norm() {
        let config = NormConfig {
            norm_type: NormType::LayerNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0], None).unwrap();
        assert!(out.mean.is_some());
        assert!(out.variance.is_some());
        assert_eq!(out.normalized.len(), 4);
    }

    #[test]
    fn test_norm_layer_rms_norm() {
        let config = NormConfig {
            norm_type: NormType::RmsNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: true,
        };
        let weight = vec![1.0; 4];
        let params = NormParams { weight: Some(weight), bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0], None).unwrap();
        assert!(out.mean.is_none());
        assert_eq!(out.normalized.len(), 4);
    }

    #[test]
    fn test_norm_layer_group_norm() {
        let config = NormConfig {
            norm_type: NormType::GroupNorm { num_groups: 2 },
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 10.0, 20.0], None).unwrap();
        assert_eq!(out.normalized.len(), 4);
    }

    #[test]
    fn test_norm_layer_batch_norm() {
        let config = NormConfig {
            norm_type: NormType::BatchNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0], None).unwrap();
        assert!(out.mean.is_some());
        assert_eq!(out.normalized.len(), 4);
    }

    #[test]
    fn test_norm_layer_instance_norm() {
        let config = NormConfig {
            norm_type: NormType::InstanceNorm,
            normalized_shape: vec![6],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], Some((2, 3))).unwrap();
        assert_eq!(out.normalized.len(), 6);
    }

    #[test]
    fn test_norm_layer_empty_input_error() {
        let config = NormConfig {
            norm_type: NormType::LayerNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        assert!(layer.forward(&[], None).is_err());
    }

    #[test]
    fn test_norm_layer_weight_shape_mismatch() {
        let config = NormConfig {
            norm_type: NormType::LayerNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: true,
        };
        let params = NormParams { weight: Some(vec![1.0; 3]), bias: None };
        assert!(NormLayer::new(config, params).is_err());
    }

    #[test]
    fn test_norm_layer_bias_shape_mismatch() {
        let config = NormConfig {
            norm_type: NormType::LayerNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: true,
        };
        let params = NormParams { weight: None, bias: Some(vec![1.0; 5]) };
        assert!(NormLayer::new(config, params).is_err());
    }

    // ── Cross-type comparisons ───────────────────────────────────────

    #[test]
    fn test_all_norm_types_same_length() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let ln = layer_norm(&input, None, None, EPS);
        let rn = rms_norm(&input, None, EPS);
        let gn = group_norm(&input, None, None, 2, EPS);
        let bn = batch_norm(&input, 2.5, 1.25, None, None, EPS);
        let inst = instance_norm(&input, (2, 2), EPS);
        assert_eq!(ln.len(), 4);
        assert_eq!(rn.len(), 4);
        assert_eq!(gn.len(), 4);
        assert_eq!(bn.len(), 4);
        assert_eq!(inst.len(), 4);
    }

    #[test]
    fn test_multiple_norm_types_on_same_data() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let ln = layer_norm(&input, None, None, EPS);
        let rn = rms_norm(&input, None, EPS);
        // LayerNorm and RMSNorm should differ (LN subtracts mean).
        assert!(
            !approx_eq(ln[0], rn[0], 1e-3),
            "LayerNorm and RMSNorm should produce different results"
        );
    }

    #[test]
    fn test_layer_norm_preserves_length() {
        for len in [1, 2, 5, 16, 100] {
            let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let out = layer_norm(&input, None, None, EPS);
            assert_eq!(out.len(), len);
        }
    }

    #[test]
    fn test_rms_norm_preserves_length() {
        for len in [1, 2, 5, 16, 100] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
            let out = rms_norm(&input, None, EPS);
            assert_eq!(out.len(), len);
        }
    }

    // ── Identity / no-affine ─────────────────────────────────────────

    #[test]
    fn test_layer_norm_identity_weight() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let ones = [1.0; 4];
        let with_w = layer_norm(&input, Some(&ones), None, EPS);
        let without = layer_norm(&input, None, None, EPS);
        for (a, b) in with_w.iter().zip(without.iter()) {
            assert!(approx_eq(*a, *b, TOL));
        }
    }

    #[test]
    fn test_rms_norm_identity_weight() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let ones = [1.0; 4];
        let with_w = rms_norm(&input, Some(&ones), EPS);
        let without = rms_norm(&input, None, EPS);
        for (a, b) in with_w.iter().zip(without.iter()) {
            assert!(approx_eq(*a, *b, TOL));
        }
    }

    #[test]
    fn test_no_affine_params() {
        let config = NormConfig {
            norm_type: NormType::LayerNorm,
            normalized_shape: vec![4],
            eps: EPS,
            elementwise_affine: false,
        };
        let params = NormParams { weight: None, bias: None };
        let layer = NormLayer::new(config, params).unwrap();
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0], None).unwrap();
        let direct = layer_norm(&[1.0, 2.0, 3.0, 4.0], None, None, EPS);
        for (a, b) in out.normalized.iter().zip(direct.iter()) {
            assert!(approx_eq(*a, *b, TOL));
        }
    }

    // ── Shape / backward-like checks ─────────────────────────────────

    #[test]
    fn test_output_shape_matches_input_layer_norm() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let out = layer_norm(&input, None, None, EPS);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_output_shape_matches_input_rms_norm() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let out = rms_norm(&input, None, EPS);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_gradient_like_backward_shape() {
        // Forward + simulate a "gradient" (just check shapes match).
        let input = [1.0, 2.0, 3.0, 4.0];
        let output = layer_norm(&input, None, None, EPS);
        let grad_output = vec![1.0_f32; output.len()];
        assert_eq!(grad_output.len(), input.len());
    }

    // ── NormType enum ────────────────────────────────────────────────

    #[test]
    fn test_norm_type_eq() {
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_eq!(NormType::RmsNorm, NormType::RmsNorm);
        assert_eq!(NormType::GroupNorm { num_groups: 4 }, NormType::GroupNorm { num_groups: 4 });
        assert_ne!(NormType::LayerNorm, NormType::RmsNorm);
        assert_ne!(NormType::GroupNorm { num_groups: 2 }, NormType::GroupNorm { num_groups: 4 });
    }

    #[test]
    fn test_norm_type_clone() {
        let t = NormType::GroupNorm { num_groups: 8 };
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn test_norm_type_debug() {
        let s = format!("{:?}", NormType::LayerNorm);
        assert!(s.contains("LayerNorm"));
    }

    // ── NormConfig ───────────────────────────────────────────────────

    #[test]
    fn test_norm_config_clone() {
        let c = NormConfig {
            norm_type: NormType::RmsNorm,
            normalized_shape: vec![64],
            eps: 1e-6,
            elementwise_affine: true,
        };
        let c2 = c.clone();
        assert_eq!(c.norm_type, c2.norm_type);
        assert_eq!(c.normalized_shape, c2.normalized_shape);
    }

    // ── Misc edge cases ──────────────────────────────────────────────

    #[test]
    fn test_layer_norm_two_elements() {
        let input = [0.0, 10.0];
        let out = layer_norm(&input, None, None, EPS);
        assert!(approx_eq(out[0], -out[1], TOL));
    }

    #[test]
    fn test_group_norm_num_groups_equals_len() {
        // Each element is its own group → normalized to 0.
        let input = [5.0, 10.0, 15.0, 20.0];
        let out = group_norm(&input, None, None, 4, EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }

    #[test]
    fn test_batch_norm_single_element() {
        let out = batch_norm(&[3.0], 3.0, 1.0, None, None, EPS);
        assert!(approx_eq(out[0], 0.0, TOL));
    }

    #[test]
    fn test_instance_norm_all_same() {
        let input = [7.0; 6];
        let out = instance_norm(&input, (2, 3), EPS);
        for &v in &out {
            assert!(approx_eq(v, 0.0, TOL));
        }
    }
}
