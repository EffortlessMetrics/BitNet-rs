//! OpenCL layer normalization variants optimized for Intel Arc A770 (Xe-HPG).
//!
//! Provides CPU reference implementations and OpenCL kernel sources for:
//!
//! - **LayerNorm**: standard `(x - mean) / sqrt(var + eps) * gamma + beta`
//! - **RMSNorm**: root-mean-square `x / sqrt(mean(x²) + eps) * gamma`
//! - **GroupNorm**: channels split into groups, each normalized independently
//! - **BatchNorm**: normalize across the batch dimension
//! - **InstanceNorm**: per-sample, per-channel normalization
//! - **FusedNormLinear**: norm + linear projection in a single pass
//! - **PreNormResidual**: `x + sublayer(norm(x))`
//!
//! OpenCL kernels use tree reduction within work-groups for numerically
//! stable parallel mean/variance computation.

use std::fmt;
use std::time::Instant;

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for layer normalization variants.
///
/// Uses tree reduction in local memory for computing mean and variance
/// in a single pass over the input, optimized for A770's 64 KB SLM.
pub const LAYER_NORM_CL: &str = r#"
// ── LayerNorm kernel ──────────────────────────────────────────────
// Each work-group normalizes one row of length `norm_size`.
// Tree reduction in local memory for mean and variance.
__kernel void layer_norm(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    const int norm_size,
    const float eps,
    const int use_affine)
{
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    __global const float* x = input + row * norm_size;
    __global float* y = output + row * norm_size;

    // Phase 1: parallel sum for mean
    __local float smem[256];
    float local_sum = 0.0f;
    for (int i = lid; i < norm_size; i += lsize)
        local_sum += x[i];
    smem[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = smem[0] / (float)norm_size;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: parallel sum for variance
    float local_var = 0.0f;
    for (int i = lid; i < norm_size; i += lsize) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    smem[lid] = local_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(smem[0] / (float)norm_size + eps);

    // Phase 3: normalize + optional affine
    for (int i = lid; i < norm_size; i += lsize) {
        float val = (x[i] - mean) * inv_std;
        if (use_affine)
            val = val * gamma[i] + beta[i];
        y[i] = val;
    }
}

// ── RMSNorm kernel ────────────────────────────────────────────────
__kernel void rms_norm(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    const int norm_size,
    const float eps,
    const int use_affine)
{
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    __global const float* x = input + row * norm_size;
    __global float* y = output + row * norm_size;

    __local float smem[256];
    float local_sq = 0.0f;
    for (int i = lid; i < norm_size; i += lsize)
        local_sq += x[i] * x[i];
    smem[lid] = local_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_rms = rsqrt(smem[0] / (float)norm_size + eps);

    for (int i = lid; i < norm_size; i += lsize) {
        float val = x[i] * inv_rms;
        if (use_affine)
            val *= gamma[i];
        y[i] = val;
    }
}

// ── GroupNorm kernel ──────────────────────────────────────────────
// One work-group per (sample, group) pair.
__kernel void group_norm(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    const int channels,
    const int group_size,
    const float eps,
    const int use_affine)
{
    int idx = get_group_id(0);        // flattened (sample, group) index
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int ch_start = (idx % (channels / group_size)) * group_size;
    __global const float* x = input + (idx / (channels / group_size)) * channels;
    __global float* y = output + (idx / (channels / group_size)) * channels;

    __local float smem[256];
    float local_sum = 0.0f;
    for (int i = lid; i < group_size; i += lsize)
        local_sum += x[ch_start + i];
    smem[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = smem[0] / (float)group_size;
    barrier(CLK_LOCAL_MEM_FENCE);

    float local_var = 0.0f;
    for (int i = lid; i < group_size; i += lsize) {
        float d = x[ch_start + i] - mean;
        local_var += d * d;
    }
    smem[lid] = local_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_std = rsqrt(smem[0] / (float)group_size + eps);

    for (int i = lid; i < group_size; i += lsize) {
        int ci = ch_start + i;
        float val = (x[ci] - mean) * inv_std;
        if (use_affine)
            val = val * gamma[ci] + beta[ci];
        y[ci] = val;
    }
}
"#;

// ── Norm type enumeration ────────────────────────────────────────

/// Normalization variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormType {
    /// Standard layer normalization with mean centering.
    LayerNorm,
    /// Root-mean-square normalization (no centering).
    RMSNorm,
    /// Group normalization: channels split into groups.
    GroupNorm,
    /// Batch normalization: normalize across the batch dimension.
    BatchNorm,
    /// Instance normalization: per-sample, per-channel.
    InstanceNorm,
}

impl fmt::Display for NormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::RMSNorm => write!(f, "RMSNorm"),
            Self::GroupNorm => write!(f, "GroupNorm"),
            Self::BatchNorm => write!(f, "BatchNorm"),
            Self::InstanceNorm => write!(f, "InstanceNorm"),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────

/// Configuration for normalization operations.
#[derive(Debug, Clone)]
pub struct NormConfig {
    /// Which normalization variant to use.
    pub norm_type: NormType,
    /// Epsilon added inside the square root for numerical stability.
    pub eps: f32,
    /// Whether to apply a learned affine transform (gamma/beta).
    pub elementwise_affine: bool,
    /// Number of groups for [`NormType::GroupNorm`]. Ignored by others.
    pub num_groups: usize,
}

impl NormConfig {
    /// Create a new config for the given normalization type.
    pub fn new(norm_type: NormType) -> Self {
        Self { norm_type, eps: 1e-5, elementwise_affine: true, num_groups: 1 }
    }

    /// Set epsilon.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Toggle the learned affine transform.
    #[must_use]
    pub fn with_elementwise_affine(mut self, affine: bool) -> Self {
        self.elementwise_affine = affine;
        self
    }

    /// Set the number of groups (only meaningful for GroupNorm).
    #[must_use]
    pub fn with_num_groups(mut self, groups: usize) -> Self {
        self.num_groups = groups;
        self
    }
}

impl Default for NormConfig {
    fn default() -> Self {
        Self::new(NormType::LayerNorm)
    }
}

// ── Statistics ───────────────────────────────────────────────────

/// Statistics collected during normalization.
#[derive(Debug, Clone)]
pub struct NormStats {
    /// Per-row mean (empty for RMSNorm).
    pub mean: Vec<f32>,
    /// Per-row variance (empty for RMSNorm).
    pub variance: Vec<f32>,
    /// Per-row root-mean-square.
    pub rms: Vec<f32>,
    /// Wall-clock time in microseconds.
    pub compute_time_us: u64,
}

impl NormStats {
    fn empty() -> Self {
        Self { mean: Vec::new(), variance: Vec::new(), rms: Vec::new(), compute_time_us: 0 }
    }
}

// ── CPU reference: LayerNorm ─────────────────────────────────────

/// Standard layer normalization.
///
/// `input` is `[num_rows, norm_size]` in row-major order.
/// `gamma` and `beta` each have `norm_size` elements (if affine).
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Learned scale (gamma). Length = `norm_size`.
    pub gamma: Vec<f32>,
    /// Learned shift (beta). Length = `norm_size`.
    pub beta: Vec<f32>,
    /// Configuration.
    pub config: NormConfig,
    /// Normalization dimension.
    pub norm_size: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm with ones for gamma, zeros for beta.
    pub fn new(norm_size: usize, config: NormConfig) -> Self {
        Self { gamma: vec![1.0; norm_size], beta: vec![0.0; norm_size], config, norm_size }
    }

    /// Create with explicit gamma and beta.
    pub fn with_params(gamma: Vec<f32>, beta: Vec<f32>, config: NormConfig) -> Self {
        let norm_size = gamma.len();
        Self { gamma, beta, config, norm_size }
    }

    /// Forward pass (CPU reference).
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormStats {
        let start = Instant::now();
        let num_rows = input.len() / self.norm_size;
        let mut stats = NormStats::empty();
        stats.mean.reserve(num_rows);
        stats.variance.reserve(num_rows);
        stats.rms.reserve(num_rows);

        for row in 0..num_rows {
            let x = &input[row * self.norm_size..(row + 1) * self.norm_size];
            let y = &mut output[row * self.norm_size..(row + 1) * self.norm_size];

            let mean = x.iter().sum::<f32>() / self.norm_size as f32;
            let var =
                x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / self.norm_size as f32;
            let rms = (x.iter().map(|&v| v * v).sum::<f32>() / self.norm_size as f32).sqrt();
            let inv_std = 1.0 / (var + self.config.eps).sqrt();

            for i in 0..self.norm_size {
                let val = (x[i] - mean) * inv_std;
                y[i] = if self.config.elementwise_affine {
                    val * self.gamma[i] + self.beta[i]
                } else {
                    val
                };
            }

            stats.mean.push(mean);
            stats.variance.push(var);
            stats.rms.push(rms);
        }

        stats.compute_time_us = start.elapsed().as_micros() as u64;
        stats
    }
}

// ── CPU reference: RMSNorm ───────────────────────────────────────

/// Root-mean-square normalization (no mean centering).
///
/// `x / sqrt(mean(x²) + eps) * gamma`
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learned scale (gamma). Length = `norm_size`.
    pub gamma: Vec<f32>,
    /// Configuration.
    pub config: NormConfig,
    /// Normalization dimension.
    pub norm_size: usize,
}

impl RmsNorm {
    /// Create a new RmsNorm with ones for gamma.
    pub fn new(norm_size: usize, config: NormConfig) -> Self {
        Self { gamma: vec![1.0; norm_size], config, norm_size }
    }

    /// Create with explicit gamma.
    pub fn with_params(gamma: Vec<f32>, config: NormConfig) -> Self {
        let norm_size = gamma.len();
        Self { gamma, config, norm_size }
    }

    /// Forward pass (CPU reference).
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormStats {
        let start = Instant::now();
        let num_rows = input.len() / self.norm_size;
        let mut stats = NormStats::empty();
        stats.rms.reserve(num_rows);

        for row in 0..num_rows {
            let x = &input[row * self.norm_size..(row + 1) * self.norm_size];
            let y = &mut output[row * self.norm_size..(row + 1) * self.norm_size];

            let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / self.norm_size as f32;
            let rms = mean_sq.sqrt();
            let inv_rms = 1.0 / (mean_sq + self.config.eps).sqrt();

            for i in 0..self.norm_size {
                let val = x[i] * inv_rms;
                y[i] = if self.config.elementwise_affine { val * self.gamma[i] } else { val };
            }

            stats.rms.push(rms);
        }

        stats.compute_time_us = start.elapsed().as_micros() as u64;
        stats
    }
}

// ── CPU reference: GroupNorm ─────────────────────────────────────

/// Group normalization: channels split into equal groups.
///
/// `input` is `[batch, channels]` in row-major order. `channels` must be
/// divisible by `num_groups`.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    /// Learned scale (gamma). Length = `channels`.
    pub gamma: Vec<f32>,
    /// Learned shift (beta). Length = `channels`.
    pub beta: Vec<f32>,
    /// Configuration (must have `num_groups` set).
    pub config: NormConfig,
    /// Total number of channels.
    pub channels: usize,
}

impl GroupNorm {
    /// Create a new GroupNorm.
    pub fn new(channels: usize, config: NormConfig) -> Self {
        Self { gamma: vec![1.0; channels], beta: vec![0.0; channels], config, channels }
    }

    /// Create with explicit gamma and beta.
    pub fn with_params(gamma: Vec<f32>, beta: Vec<f32>, config: NormConfig) -> Self {
        let channels = gamma.len();
        Self { gamma, beta, config, channels }
    }

    /// Forward pass (CPU reference).
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormStats {
        let start = Instant::now();
        let batch = input.len() / self.channels;
        let group_size = self.channels / self.config.num_groups;
        let mut stats = NormStats::empty();

        for b in 0..batch {
            let x = &input[b * self.channels..(b + 1) * self.channels];
            let y = &mut output[b * self.channels..(b + 1) * self.channels];

            for g in 0..self.config.num_groups {
                let start_ch = g * group_size;
                let group = &x[start_ch..start_ch + group_size];

                let mean = group.iter().sum::<f32>() / group_size as f32;
                let var =
                    group.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
                let inv_std = 1.0 / (var + self.config.eps).sqrt();

                for i in 0..group_size {
                    let ci = start_ch + i;
                    let val = (x[ci] - mean) * inv_std;
                    y[ci] = if self.config.elementwise_affine {
                        val * self.gamma[ci] + self.beta[ci]
                    } else {
                        val
                    };
                }

                stats.mean.push(mean);
                stats.variance.push(var);
            }
        }

        stats.compute_time_us = start.elapsed().as_micros() as u64;
        stats
    }
}

// ── CPU reference: BatchNorm ─────────────────────────────────────

/// Batch normalization (inference mode, running stats provided).
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward(
    input: &[f32],
    output: &mut [f32],
    running_mean: &[f32],
    running_var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    eps: f32,
    affine: bool,
) {
    let batch = input.len() / channels;
    for b in 0..batch {
        for c in 0..channels {
            let idx = b * channels + c;
            let val = (input[idx] - running_mean[c]) / (running_var[c] + eps).sqrt();
            output[idx] = if affine { val * gamma[c] + beta[c] } else { val };
        }
    }
}

/// Instance normalization (each sample, each channel independently).
#[allow(clippy::too_many_arguments)]
pub fn instance_norm_forward(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
    affine: bool,
) {
    for b in 0..batch {
        for c in 0..channels {
            let offset = (b * channels + c) * spatial;
            let slice = &input[offset..offset + spatial];

            let mean = slice.iter().sum::<f32>() / spatial as f32;
            let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / spatial as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for s in 0..spatial {
                let val = (input[offset + s] - mean) * inv_std;
                output[offset + s] = if affine { val * gamma[c] + beta[c] } else { val };
            }
        }
    }
}

// ── Fused norm + linear ──────────────────────────────────────────

/// Fused normalization followed by a linear projection in one pass.
///
/// Computes `output = norm(input) @ weight^T + bias` without
/// materializing the full normalized tensor.
#[derive(Debug, Clone)]
pub struct FusedNormLinear {
    /// Normalization layer (LayerNorm or RMSNorm).
    pub norm_type: NormType,
    /// Norm dimension.
    pub norm_size: usize,
    /// Output dimension of the linear projection.
    pub out_features: usize,
    /// Gamma for normalization. Length = `norm_size`.
    pub gamma: Vec<f32>,
    /// Beta for normalization (unused for RMSNorm). Length = `norm_size`.
    pub beta: Vec<f32>,
    /// Weight matrix `[out_features, norm_size]`, row-major.
    pub weight: Vec<f32>,
    /// Optional bias vector. Length = `out_features`.
    pub bias: Option<Vec<f32>>,
    /// Epsilon.
    pub eps: f32,
}

impl FusedNormLinear {
    /// Create a new fused norm+linear with identity-like defaults.
    pub fn new(norm_type: NormType, norm_size: usize, out_features: usize) -> Self {
        Self {
            norm_type,
            norm_size,
            out_features,
            gamma: vec![1.0; norm_size],
            beta: vec![0.0; norm_size],
            weight: vec![0.0; out_features * norm_size],
            bias: None,
            eps: 1e-5,
        }
    }

    /// Set weight matrix.
    #[must_use]
    pub fn with_weight(mut self, weight: Vec<f32>) -> Self {
        self.weight = weight;
        self
    }

    /// Set bias vector.
    #[must_use]
    pub fn with_bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Forward pass (CPU reference).
    ///
    /// `input` is `[num_rows, norm_size]`, `output` is `[num_rows, out_features]`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let num_rows = input.len() / self.norm_size;

        for row in 0..num_rows {
            let x = &input[row * self.norm_size..(row + 1) * self.norm_size];
            let y = &mut output[row * self.out_features..(row + 1) * self.out_features];

            // Compute norm stats inline
            let (inv_scale, mean) = match self.norm_type {
                NormType::RMSNorm => {
                    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / self.norm_size as f32;
                    (1.0 / (mean_sq + self.eps).sqrt(), 0.0)
                }
                _ => {
                    let mean = x.iter().sum::<f32>() / self.norm_size as f32;
                    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>()
                        / self.norm_size as f32;
                    (1.0 / (var + self.eps).sqrt(), mean)
                }
            };

            // Fused: normalize then project
            for o in 0..self.out_features {
                let mut acc = 0.0_f32;
                for (i, (&xi, (&gi, &bi))) in x
                    .iter()
                    .zip(self.gamma.iter().zip(self.beta.iter()))
                    .enumerate()
                    .take(self.norm_size)
                {
                    let normed = (xi - mean) * inv_scale;
                    let affine = normed * gi + bi;
                    acc += affine * self.weight[o * self.norm_size + i];
                }
                if let Some(ref bias) = self.bias {
                    acc += bias[o];
                }
                y[o] = acc;
            }
        }
    }
}

// ── Pre-norm residual ────────────────────────────────────────────

/// Pre-norm residual connection: `output = x + sublayer(norm(x))`.
///
/// The `sublayer` is expressed as a closure so that any transform can
/// be composed (attention, FFN, etc.).
pub struct PreNormResidual {
    /// Normalization layer.
    pub norm: LayerNorm,
}

impl PreNormResidual {
    /// Create a pre-norm residual block.
    pub fn new(norm: LayerNorm) -> Self {
        Self { norm }
    }

    /// Forward pass: `output = x + sublayer(norm(x))`.
    ///
    /// `sublayer_fn` takes normalized input and writes to the provided
    /// output buffer. Input and output have the same shape
    /// `[num_rows, norm_size]`.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        sublayer_fn: &dyn Fn(&[f32], &mut [f32]),
    ) -> NormStats {
        let len = input.len();
        let mut normed = vec![0.0_f32; len];
        let stats = self.norm.forward(input, &mut normed);

        let mut sublayer_out = vec![0.0_f32; len];
        sublayer_fn(&normed, &mut sublayer_out);

        for i in 0..len {
            output[i] = input[i] + sublayer_out[i];
        }

        stats
    }
}

impl fmt::Debug for PreNormResidual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreNormResidual").field("norm", &self.norm).finish()
    }
}

// ── Standalone helpers ───────────────────────────────────────────

/// Standalone LayerNorm forward (no struct needed).
pub fn layer_norm_ref(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    norm_size: usize,
    eps: f32,
    affine: bool,
) {
    let num_rows = input.len() / norm_size;
    for row in 0..num_rows {
        let x = &input[row * norm_size..(row + 1) * norm_size];
        let y = &mut output[row * norm_size..(row + 1) * norm_size];

        let mean = x.iter().sum::<f32>() / norm_size as f32;
        let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..norm_size {
            let val = (x[i] - mean) * inv_std;
            y[i] = if affine { val * gamma[i] + beta[i] } else { val };
        }
    }
}

/// Standalone RMSNorm forward (no struct needed).
pub fn rms_norm_ref(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    norm_size: usize,
    eps: f32,
    affine: bool,
) {
    let num_rows = input.len() / norm_size;
    for row in 0..num_rows {
        let x = &input[row * norm_size..(row + 1) * norm_size];
        let y = &mut output[row * norm_size..(row + 1) * norm_size];

        let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / norm_size as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        for i in 0..norm_size {
            let val = x[i] * inv_rms;
            y[i] = if affine { val * gamma[i] } else { val };
        }
    }
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_near(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b} (tol={tol})");
    }

    fn assert_slices_near(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "{msg} [index {i}]: {x} vs {y} (tol={tol})");
        }
    }

    /// Generate deterministic test data.
    fn make_input(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.1 - 0.5).sin()).collect()
    }

    // ===============================================================
    // NormType / NormConfig
    // ===============================================================

    #[test]
    fn test_norm_type_display() {
        assert_eq!(NormType::LayerNorm.to_string(), "LayerNorm");
        assert_eq!(NormType::RMSNorm.to_string(), "RMSNorm");
        assert_eq!(NormType::GroupNorm.to_string(), "GroupNorm");
        assert_eq!(NormType::BatchNorm.to_string(), "BatchNorm");
        assert_eq!(NormType::InstanceNorm.to_string(), "InstanceNorm");
    }

    #[test]
    fn test_norm_type_equality() {
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_ne!(NormType::LayerNorm, NormType::RMSNorm);
    }

    #[test]
    fn test_norm_config_defaults() {
        let cfg = NormConfig::default();
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
        assert_near(cfg.eps, 1e-5, 1e-10, "default eps");
        assert!(cfg.elementwise_affine);
        assert_eq!(cfg.num_groups, 1);
    }

    #[test]
    fn test_norm_config_builder() {
        let cfg = NormConfig::new(NormType::GroupNorm)
            .with_eps(1e-6)
            .with_elementwise_affine(false)
            .with_num_groups(8);
        assert_eq!(cfg.norm_type, NormType::GroupNorm);
        assert_near(cfg.eps, 1e-6, 1e-10, "custom eps");
        assert!(!cfg.elementwise_affine);
        assert_eq!(cfg.num_groups, 8);
    }

    // ===============================================================
    // LayerNorm
    // ===============================================================

    #[test]
    fn test_layer_norm_basic_4() {
        // PyTorch reference: x = [1, 2, 3, 4], LayerNorm(4, eps=1e-5)
        // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25 + 1e-5) ≈ 0.894427
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let stats = ln.forward(&input, &mut output);

        // Expected: [-1.3416, -0.4472, 0.4472, 1.3416]
        assert_near(output[0], -1.3416, 1e-3, "ln[0]");
        assert_near(output[1], -0.4472, 1e-3, "ln[1]");
        assert_near(output[2], 0.4472, 1e-3, "ln[2]");
        assert_near(output[3], 1.3416, 1e-3, "ln[3]");

        assert_near(stats.mean[0], 2.5, TOL, "mean");
        assert_near(stats.variance[0], 1.25, TOL, "var");
    }

    #[test]
    fn test_layer_norm_output_matches_pytorch_reference() {
        // x = [-1, 0, 1, 2], normalized with eps=1e-5
        // mean = 0.5, var = 1.25
        let input = [-1.0_f32, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 4];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        let mean = 0.5_f32;
        let var = 1.25_f32;
        let inv_std = 1.0 / (var + EPS).sqrt();
        for i in 0..4 {
            let expected = (input[i] - mean) * inv_std;
            assert_near(output[i], expected, TOL, &format!("pytorch ref [{i}]"));
        }
    }

    #[test]
    fn test_layer_norm_with_affine_params() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = vec![2.0, 2.0, 2.0, 2.0];
        let beta = vec![1.0, 1.0, 1.0, 1.0];
        let config = NormConfig::new(NormType::LayerNorm);
        let ln = LayerNorm::with_params(gamma, beta, config);
        let mut output = vec![0.0; 4];
        ln.forward(&input, &mut output);

        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // output = 2 * normalized + 1
        assert_near(output[0], 2.0 * -1.3416 + 1.0, 2e-3, "affine[0]");
        assert_near(output[3], 2.0 * 1.3416 + 1.0, 2e-3, "affine[3]");
    }

    #[test]
    fn test_layer_norm_no_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let cfg = NormConfig::new(NormType::LayerNorm).with_elementwise_affine(false);
        let ln = LayerNorm::new(4, cfg);
        ln.forward(&input, &mut output);

        // Same as basic but without scaling
        assert_near(output[0], -1.3416, 1e-3, "no-affine[0]");
        assert_near(output[3], 1.3416, 1e-3, "no-affine[3]");
    }

    #[test]
    fn test_layer_norm_multiple_rows() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 8];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let stats = ln.forward(&input, &mut output);

        assert_eq!(stats.mean.len(), 2);
        assert_near(stats.mean[0], 2.5, TOL, "row0 mean");
        assert_near(stats.mean[1], 25.0, TOL, "row1 mean");
    }

    #[test]
    fn test_layer_norm_hidden_dim_64() {
        let input = make_input(64);
        let mut output = vec![0.0; 64];
        let ln = LayerNorm::new(64, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        // Output should be approximately zero-mean, unit-variance
        let mean: f32 = output.iter().sum::<f32>() / 64.0;
        let var: f32 = output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 64.0;
        assert_near(mean, 0.0, 1e-3, "dim64 mean");
        assert_near(var, 1.0, 0.05, "dim64 var");
    }

    #[test]
    fn test_layer_norm_hidden_dim_128() {
        let input = make_input(128);
        let mut output = vec![0.0; 128];
        let ln = LayerNorm::new(128, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);
        let mean: f32 = output.iter().sum::<f32>() / 128.0;
        let var: f32 = output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 128.0;
        assert_near(mean, 0.0, 1e-3, "dim128 mean");
        assert_near(var, 1.0, 0.05, "dim128 var");
    }

    #[test]
    fn test_layer_norm_hidden_dim_256() {
        let input = make_input(256);
        let mut output = vec![0.0; 256];
        let ln = LayerNorm::new(256, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);
        let mean: f32 = output.iter().sum::<f32>() / 256.0;
        assert_near(mean, 0.0, 1e-3, "dim256 mean");
    }

    #[test]
    fn test_layer_norm_hidden_dim_512() {
        let input = make_input(512);
        let mut output = vec![0.0; 512];
        let ln = LayerNorm::new(512, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);
        let mean: f32 = output.iter().sum::<f32>() / 512.0;
        assert_near(mean, 0.0, 1e-3, "dim512 mean");
    }

    #[test]
    fn test_layer_norm_hidden_dim_4096() {
        let input = make_input(4096);
        let mut output = vec![0.0; 4096];
        let ln = LayerNorm::new(4096, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);
        let mean: f32 = output.iter().sum::<f32>() / 4096.0;
        let var: f32 = output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4096.0;
        assert_near(mean, 0.0, 1e-3, "dim4096 mean");
        assert_near(var, 1.0, 0.05, "dim4096 var");
    }

    #[test]
    fn test_layer_norm_stats_have_compute_time() {
        let input = make_input(64);
        let mut output = vec![0.0; 64];
        let ln = LayerNorm::new(64, NormConfig::new(NormType::LayerNorm));
        let stats = ln.forward(&input, &mut output);
        // compute_time_us should be non-negative (may be 0 on fast hardware)
        assert!(stats.compute_time_us < 1_000_000, "unreasonable compute time");
    }

    // ===============================================================
    // RMSNorm
    // ===============================================================

    #[test]
    fn test_rms_norm_basic_4() {
        // x = [1, 2, 3, 4], mean(x²) = (1+4+9+16)/4 = 7.5
        // inv_rms = 1/sqrt(7.5 + 1e-5) ≈ 0.36515
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let rms = RmsNorm::new(4, NormConfig::new(NormType::RMSNorm));
        let stats = rms.forward(&input, &mut output);

        let mean_sq = 7.5_f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..4 {
            assert_near(output[i], input[i] * inv, TOL, &format!("rms[{i}]"));
        }
        assert_near(stats.rms[0], mean_sq.sqrt(), TOL, "rms stat");
    }

    #[test]
    fn test_rms_norm_output_matches_reference() {
        let input = [-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        let rms = RmsNorm::new(5, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        let mean_sq = (4.0 + 1.0 + 0.0 + 1.0 + 4.0) / 5.0; // 2.0
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..5 {
            assert_near(output[i], input[i] * inv, TOL, &format!("rms ref [{i}]"));
        }
    }

    #[test]
    fn test_rms_norm_with_gamma() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = vec![0.5, 0.5, 0.5, 0.5];
        let config = NormConfig::new(NormType::RMSNorm);
        let rms = RmsNorm::with_params(gamma, config);
        let mut output = vec![0.0; 4];
        rms.forward(&input, &mut output);

        let mean_sq = 7.5_f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..4 {
            assert_near(output[i], input[i] * inv * 0.5, TOL, &format!("rms gamma[{i}]"));
        }
    }

    #[test]
    fn test_rms_norm_no_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let cfg = NormConfig::new(NormType::RMSNorm).with_elementwise_affine(false);
        let rms = RmsNorm::new(4, cfg);
        rms.forward(&input, &mut output);

        let mean_sq = 7.5_f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..4 {
            assert_near(output[i], input[i] * inv, TOL, &format!("rms no-affine[{i}]"));
        }
    }

    #[test]
    fn test_rms_norm_multiple_rows() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        let rms = RmsNorm::new(4, NormConfig::new(NormType::RMSNorm));
        let stats = rms.forward(&input, &mut output);
        assert_eq!(stats.rms.len(), 2);
    }

    #[test]
    fn test_rms_norm_hidden_dim_64() {
        let input = make_input(64);
        let mut output = vec![0.0; 64];
        let rms = RmsNorm::new(64, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);
        // After RMSNorm (no centering), RMS of output ≈ 1
        let out_rms = (output.iter().map(|v| v * v).sum::<f32>() / 64.0).sqrt();
        assert_near(out_rms, 1.0, 0.05, "rms dim64 output rms");
    }

    #[test]
    fn test_rms_norm_hidden_dim_4096() {
        let input = make_input(4096);
        let mut output = vec![0.0; 4096];
        let rms = RmsNorm::new(4096, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);
        let out_rms = (output.iter().map(|v| v * v).sum::<f32>() / 4096.0).sqrt();
        assert_near(out_rms, 1.0, 0.05, "rms dim4096 output rms");
    }

    // ===============================================================
    // GroupNorm
    // ===============================================================

    #[test]
    fn test_group_norm_single_group() {
        // With 1 group, GroupNorm == LayerNorm
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut gn_out = vec![0.0; 4];
        let mut ln_out = vec![0.0; 4];

        let gn_cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(1);
        let gn = GroupNorm::new(4, gn_cfg);
        gn.forward(&input, &mut gn_out);

        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut ln_out);

        assert_slices_near(&gn_out, &ln_out, TOL, "gn==ln for 1 group");
    }

    #[test]
    fn test_group_norm_two_groups() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(2);
        let gn = GroupNorm::new(4, cfg);
        gn.forward(&input, &mut output);

        // Group 0: [1, 2] → mean=1.5, var=0.25
        // Group 1: [3, 4] → mean=3.5, var=0.25
        let inv = 1.0 / (0.25 + EPS).sqrt();
        assert_near(output[0], (1.0 - 1.5) * inv, TOL, "g0[0]");
        assert_near(output[1], (2.0 - 1.5) * inv, TOL, "g0[1]");
        assert_near(output[2], (3.0 - 3.5) * inv, TOL, "g1[0]");
        assert_near(output[3], (4.0 - 3.5) * inv, TOL, "g1[1]");
    }

    #[test]
    fn test_group_norm_four_groups() {
        // 4 groups of 1 → each element normalized to 0 (zero variance)
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(4);
        let gn = GroupNorm::new(4, cfg);
        gn.forward(&input, &mut output);

        // Each "group" is a single element: (x - x) / sqrt(0 + eps) = 0
        for i in 0..4 {
            assert_near(output[i], 0.0, TOL, &format!("4groups[{i}]"));
        }
    }

    #[test]
    fn test_group_norm_eight_groups_dim_64() {
        let input = make_input(64);
        let mut output = vec![0.0; 64];
        let cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(8);
        let gn = GroupNorm::new(64, cfg);
        let stats = gn.forward(&input, &mut output);
        assert_eq!(stats.mean.len(), 8);
        assert_eq!(stats.variance.len(), 8);
    }

    #[test]
    fn test_group_norm_with_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![0.5; 4];
        let cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(2);
        let gn = GroupNorm::with_params(gamma, beta, cfg);
        let mut output = vec![0.0; 4];
        gn.forward(&input, &mut output);

        let inv = 1.0 / (0.25 + EPS).sqrt();
        let expected_0 = (1.0 - 1.5) * inv * 2.0 + 0.5;
        assert_near(output[0], expected_0, TOL, "gn affine[0]");
    }

    #[test]
    fn test_group_norm_no_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let cfg =
            NormConfig::new(NormType::GroupNorm).with_num_groups(2).with_elementwise_affine(false);
        let gn = GroupNorm::new(4, cfg);
        gn.forward(&input, &mut output);

        let inv = 1.0 / (0.25 + EPS).sqrt();
        assert_near(output[0], (1.0 - 1.5) * inv, TOL, "gn no-affine[0]");
    }

    #[test]
    fn test_group_norm_batch_of_two() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];
        let cfg = NormConfig::new(NormType::GroupNorm).with_num_groups(2);
        let gn = GroupNorm::new(4, cfg);
        let stats = gn.forward(&input, &mut output);
        // 2 batches × 2 groups = 4 mean/var entries
        assert_eq!(stats.mean.len(), 4);
    }

    // ===============================================================
    // BatchNorm
    // ===============================================================

    #[test]
    fn test_batch_norm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];
        let running_mean = [3.0, 4.0, 5.0];
        let running_var = [1.0, 1.0, 1.0];
        let gamma = [1.0, 1.0, 1.0];
        let beta = [0.0, 0.0, 0.0];

        batch_norm_forward(
            &input,
            &mut output,
            &running_mean,
            &running_var,
            &gamma,
            &beta,
            3,
            EPS,
            true,
        );

        // Row 0: (1-3)/sqrt(1+eps), (2-4)/sqrt(1+eps), (3-5)/sqrt(1+eps)
        let inv = 1.0 / (1.0 + EPS).sqrt();
        assert_near(output[0], -2.0 * inv, TOL, "bn[0]");
        assert_near(output[1], -2.0 * inv, TOL, "bn[1]");
        assert_near(output[2], -2.0 * inv, TOL, "bn[2]");
    }

    #[test]
    fn test_batch_norm_with_affine() {
        let input = [0.0_f32; 4];
        let mut output = vec![0.0; 4];
        let running_mean = [0.0, 0.0, 0.0, 0.0];
        let running_var = [1.0, 1.0, 1.0, 1.0];
        let gamma = [2.0, 2.0, 2.0, 2.0];
        let beta = [1.0, 1.0, 1.0, 1.0];

        batch_norm_forward(
            &input,
            &mut output,
            &running_mean,
            &running_var,
            &gamma,
            &beta,
            4,
            EPS,
            true,
        );

        // (0 - 0) / sqrt(1+eps) * 2 + 1 = 1
        for i in 0..4 {
            assert_near(output[i], 1.0, TOL, &format!("bn affine[{i}]"));
        }
    }

    #[test]
    fn test_batch_norm_no_affine() {
        let input = [5.0_f32, 10.0];
        let mut output = vec![0.0; 2];
        let running_mean = [5.0, 10.0];
        let running_var = [4.0, 9.0];
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];

        batch_norm_forward(
            &input,
            &mut output,
            &running_mean,
            &running_var,
            &gamma,
            &beta,
            2,
            EPS,
            false,
        );

        for i in 0..2 {
            assert_near(output[i], 0.0, TOL, &format!("bn no-affine[{i}]"));
        }
    }

    // ===============================================================
    // InstanceNorm
    // ===============================================================

    #[test]
    fn test_instance_norm_basic() {
        // batch=1, channels=1, spatial=4
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let gamma = [1.0];
        let beta = [0.0];

        instance_norm_forward(&input, &mut output, &gamma, &beta, 1, 1, 4, EPS, true);

        // Same as LayerNorm on [1,2,3,4]
        assert_near(output[0], -1.3416, 1e-3, "in[0]");
        assert_near(output[3], 1.3416, 1e-3, "in[3]");
    }

    #[test]
    fn test_instance_norm_two_channels() {
        // batch=1, channels=2, spatial=2
        let input = [1.0_f32, 3.0, 10.0, 20.0];
        let mut output = vec![0.0; 4];
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];

        instance_norm_forward(&input, &mut output, &gamma, &beta, 1, 2, 2, EPS, true);

        // Channel 0: [1,3] → mean=2, var=1
        let inv0 = 1.0 / (1.0 + EPS).sqrt();
        assert_near(output[0], (1.0 - 2.0) * inv0, TOL, "in ch0[0]");
        assert_near(output[1], (3.0 - 2.0) * inv0, TOL, "in ch0[1]");

        // Channel 1: [10,20] → mean=15, var=25
        let inv1 = 1.0 / (25.0 + EPS).sqrt();
        assert_near(output[2], (10.0 - 15.0) * inv1, TOL, "in ch1[0]");
        assert_near(output[3], (20.0 - 15.0) * inv1, TOL, "in ch1[1]");
    }

    // ===============================================================
    // FusedNormLinear
    // ===============================================================

    #[test]
    fn test_fused_norm_linear_matches_sequential_layernorm() {
        let norm_size = 4;
        let out_features = 3;
        let input = [1.0_f32, 2.0, 3.0, 4.0];

        // Build identity-like weight
        let mut weight = vec![0.0_f32; out_features * norm_size];
        for o in 0..out_features.min(norm_size) {
            weight[o * norm_size + o] = 1.0;
        }

        let fused = FusedNormLinear::new(NormType::LayerNorm, norm_size, out_features)
            .with_weight(weight.clone());

        let mut fused_out = vec![0.0; out_features];
        fused.forward(&input, &mut fused_out);

        // Sequential: norm then matmul
        let ln = LayerNorm::new(norm_size, NormConfig::new(NormType::LayerNorm));
        let mut normed = vec![0.0; norm_size];
        ln.forward(&input, &mut normed);

        let mut seq_out = vec![0.0; out_features];
        for o in 0..out_features {
            for i in 0..norm_size {
                seq_out[o] += normed[i] * weight[o * norm_size + i];
            }
        }

        assert_slices_near(&fused_out, &seq_out, TOL, "fused vs sequential LN");
    }

    #[test]
    fn test_fused_norm_linear_matches_sequential_rmsnorm() {
        let norm_size = 4;
        let out_features = 2;
        let input = [1.0_f32, 2.0, 3.0, 4.0];

        let weight = vec![1.0; out_features * norm_size];
        let fused = FusedNormLinear::new(NormType::RMSNorm, norm_size, out_features)
            .with_weight(weight.clone());

        let mut fused_out = vec![0.0; out_features];
        fused.forward(&input, &mut fused_out);

        // Sequential
        let rn = RmsNorm::new(norm_size, NormConfig::new(NormType::RMSNorm));
        let mut normed = vec![0.0; norm_size];
        rn.forward(&input, &mut normed);

        let mut seq_out = vec![0.0; out_features];
        for o in 0..out_features {
            for i in 0..norm_size {
                seq_out[o] += normed[i] * weight[o * norm_size + i];
            }
        }

        assert_slices_near(&fused_out, &seq_out, TOL, "fused vs sequential RMS");
    }

    #[test]
    fn test_fused_norm_linear_with_bias() {
        let norm_size = 4;
        let out_features = 2;
        let input = [1.0_f32, 2.0, 3.0, 4.0];

        let weight = vec![0.0; out_features * norm_size];
        let bias = vec![5.0, 10.0];
        let fused = FusedNormLinear::new(NormType::LayerNorm, norm_size, out_features)
            .with_weight(weight)
            .with_bias(bias);

        let mut output = vec![0.0; out_features];
        fused.forward(&input, &mut output);

        // Zero weight → output = bias
        assert_near(output[0], 5.0, TOL, "bias[0]");
        assert_near(output[1], 10.0, TOL, "bias[1]");
    }

    #[test]
    fn test_fused_norm_linear_multiple_rows() {
        let norm_size = 4;
        let out_features = 2;
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0; out_features * norm_size];

        let fused =
            FusedNormLinear::new(NormType::LayerNorm, norm_size, out_features).with_weight(weight);

        let mut output = vec![0.0; 2 * out_features];
        fused.forward(&input, &mut output);

        // Normalized [1,2,3,4] sums to ~0 → each output row entry ≈ 0
        assert_near(output[0], 0.0, 0.1, "fused row0[0]");
    }

    // ===============================================================
    // PreNormResidual
    // ===============================================================

    #[test]
    fn test_pre_norm_residual_identity_sublayer() {
        // sublayer = identity → output = x + norm(x)
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let norm = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let pre = PreNormResidual::new(norm);

        let identity = |inp: &[f32], out: &mut [f32]| out.copy_from_slice(inp);
        pre.forward(&input, &mut output, &identity);

        // output = x + norm(x)
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let mut normed = vec![0.0; 4];
        ln.forward(&input, &mut normed);

        for i in 0..4 {
            assert_near(output[i], input[i] + normed[i], TOL, &format!("pre-norm id[{i}]"));
        }
    }

    #[test]
    fn test_pre_norm_residual_zero_sublayer() {
        // sublayer = zero → output = x + 0 = x
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let norm = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let pre = PreNormResidual::new(norm);

        let zero = |_inp: &[f32], out: &mut [f32]| {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        };
        pre.forward(&input, &mut output, &zero);

        assert_slices_near(&output, &input, TOL, "pre-norm zero sublayer");
    }

    #[test]
    fn test_pre_norm_residual_returns_stats() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let norm = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let pre = PreNormResidual::new(norm);

        let identity = |inp: &[f32], out: &mut [f32]| out.copy_from_slice(inp);
        let stats = pre.forward(&input, &mut output, &identity);
        assert_eq!(stats.mean.len(), 1);
        assert_near(stats.mean[0], 2.5, TOL, "pre-norm stats mean");
    }

    #[test]
    fn test_pre_norm_residual_debug() {
        let norm = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        let pre = PreNormResidual::new(norm);
        let dbg = format!("{pre:?}");
        assert!(dbg.contains("PreNormResidual"));
    }

    // ===============================================================
    // Standalone helpers
    // ===============================================================

    #[test]
    fn test_standalone_layer_norm_ref() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        layer_norm_ref(&input, &mut output, &gamma, &beta, 4, EPS, true);

        let mut expected = vec![0.0; 4];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut expected);

        assert_slices_near(&output, &expected, TOL, "standalone LN ref");
    }

    #[test]
    fn test_standalone_rms_norm_ref() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        rms_norm_ref(&input, &mut output, &gamma, 4, EPS, true);

        let mut expected = vec![0.0; 4];
        let rms = RmsNorm::new(4, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut expected);

        assert_slices_near(&output, &expected, TOL, "standalone RMS ref");
    }

    #[test]
    fn test_standalone_layer_norm_ref_no_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        layer_norm_ref(&input, &mut out1, &gamma, &beta, 4, EPS, false);

        let cfg = NormConfig::new(NormType::LayerNorm).with_elementwise_affine(false);
        let ln = LayerNorm::new(4, cfg);
        ln.forward(&input, &mut out2);

        assert_slices_near(&out1, &out2, TOL, "standalone LN no-affine");
    }

    // ===============================================================
    // Epsilon sensitivity
    // ===============================================================

    #[test]
    fn test_epsilon_sensitivity_layer_norm() {
        let input = [1e-8_f32, 2e-8, 3e-8, 4e-8];
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];

        let cfg1 = NormConfig::new(NormType::LayerNorm).with_eps(1e-5);
        let cfg2 = NormConfig::new(NormType::LayerNorm).with_eps(1e-12);

        LayerNorm::new(4, cfg1).forward(&input, &mut out1);
        LayerNorm::new(4, cfg2).forward(&input, &mut out2);

        // Both should produce valid output (not NaN/Inf)
        for i in 0..4 {
            assert!(out1[i].is_finite(), "eps=1e-5 produced non-finite at {i}");
            assert!(out2[i].is_finite(), "eps=1e-12 produced non-finite at {i}");
        }
    }

    #[test]
    fn test_epsilon_sensitivity_rms_norm() {
        let input = [1e-8_f32, 2e-8, 3e-8, 4e-8];
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];

        let cfg1 = NormConfig::new(NormType::RMSNorm).with_eps(1e-5);
        let cfg2 = NormConfig::new(NormType::RMSNorm).with_eps(1e-12);

        RmsNorm::new(4, cfg1).forward(&input, &mut out1);
        RmsNorm::new(4, cfg2).forward(&input, &mut out2);

        for i in 0..4 {
            assert!(out1[i].is_finite(), "rms eps=1e-5 non-finite at {i}");
            assert!(out2[i].is_finite(), "rms eps=1e-12 non-finite at {i}");
        }
    }

    #[test]
    fn test_large_epsilon_approaches_zero_output() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        // Very large eps → var+eps dominated by eps → inv_std ≈ 0
        let cfg = NormConfig::new(NormType::LayerNorm).with_eps(1e10);
        LayerNorm::new(4, cfg).forward(&input, &mut output);
        for &v in &output {
            assert!(v.abs() < 0.01, "large eps should squash output, got {v}");
        }
    }

    // ===============================================================
    // Edge cases
    // ===============================================================

    #[test]
    fn test_all_zeros_layer_norm() {
        let input = [0.0_f32; 8];
        let mut output = vec![0.0; 8];
        let ln = LayerNorm::new(8, NormConfig::new(NormType::LayerNorm));
        let stats = ln.forward(&input, &mut output);

        // All zeros → mean=0, var=0, inv_std=1/sqrt(eps)
        // output = (0 - 0) * inv_std = 0
        for &v in &output {
            assert_near(v, 0.0, TOL, "zeros output");
        }
        assert_near(stats.mean[0], 0.0, TOL, "zeros mean");
        assert_near(stats.variance[0], 0.0, TOL, "zeros var");
    }

    #[test]
    fn test_all_zeros_rms_norm() {
        let input = [0.0_f32; 8];
        let mut output = vec![0.0; 8];
        let rms = RmsNorm::new(8, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        for &v in &output {
            assert_near(v, 0.0, TOL, "zeros rms output");
        }
    }

    #[test]
    fn test_all_same_value_layer_norm() {
        let input = [42.0_f32; 4];
        let mut output = vec![0.0; 4];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        // All same → mean=42, var=0 → (x-42)/sqrt(eps) all ≈ 0
        for &v in &output {
            assert_near(v, 0.0, TOL, "same value output");
        }
    }

    #[test]
    fn test_all_same_value_rms_norm() {
        let input = [3.0_f32; 4];
        let mut output = vec![0.0; 4];
        let rms = RmsNorm::new(4, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        // mean(x²)=9, inv_rms=1/sqrt(9+eps)=1/3 → output ≈ 1.0
        for &v in &output {
            assert_near(v, 1.0, TOL, "same value rms output");
        }
    }

    #[test]
    fn test_single_element_layer_norm() {
        let input = [5.0_f32];
        let mut output = vec![0.0; 1];
        let ln = LayerNorm::new(1, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        // mean=5, var=0 → (5-5)/sqrt(eps) = 0
        assert_near(output[0], 0.0, TOL, "single elem LN");
    }

    #[test]
    fn test_single_element_rms_norm() {
        let input = [5.0_f32];
        let mut output = vec![0.0; 1];
        let rms = RmsNorm::new(1, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        // mean(x²)=25, inv=1/sqrt(25+eps) ≈ 0.2 → output ≈ 1.0
        assert_near(output[0], 1.0, TOL, "single elem RMS");
    }

    // ===============================================================
    // Property tests: output has unit variance after LayerNorm
    // ===============================================================

    #[test]
    fn test_output_unit_variance_dim_64() {
        check_unit_variance(64);
    }

    #[test]
    fn test_output_unit_variance_dim_128() {
        check_unit_variance(128);
    }

    #[test]
    fn test_output_unit_variance_dim_256() {
        check_unit_variance(256);
    }

    #[test]
    fn test_output_unit_variance_dim_512() {
        check_unit_variance(512);
    }

    #[test]
    fn test_output_unit_variance_dim_4096() {
        check_unit_variance(4096);
    }

    fn check_unit_variance(dim: usize) {
        let input = make_input(dim);
        let mut output = vec![0.0; dim];
        let ln = LayerNorm::new(dim, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        let mean: f32 = output.iter().sum::<f32>() / dim as f32;
        let var: f32 = output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        assert!((var - 1.0).abs() < 0.05, "dim={dim}: expected var≈1.0, got {var}");
    }

    // ===============================================================
    // Property: RMSNorm output has unit RMS
    // ===============================================================

    #[test]
    fn test_output_unit_rms_dim_64() {
        check_unit_rms(64);
    }

    #[test]
    fn test_output_unit_rms_dim_256() {
        check_unit_rms(256);
    }

    #[test]
    fn test_output_unit_rms_dim_4096() {
        check_unit_rms(4096);
    }

    fn check_unit_rms(dim: usize) {
        let input = make_input(dim);
        let mut output = vec![0.0; dim];
        let rms = RmsNorm::new(dim, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        let out_rms = (output.iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
        assert!((out_rms - 1.0).abs() < 0.05, "dim={dim}: expected RMS≈1.0, got {out_rms}");
    }

    // ===============================================================
    // OpenCL kernel source validation
    // ===============================================================

    #[test]
    fn test_opencl_source_contains_layer_norm_kernel() {
        assert!(LAYER_NORM_CL.contains("__kernel void layer_norm"));
    }

    #[test]
    fn test_opencl_source_contains_rms_norm_kernel() {
        assert!(LAYER_NORM_CL.contains("__kernel void rms_norm"));
    }

    #[test]
    fn test_opencl_source_contains_group_norm_kernel() {
        assert!(LAYER_NORM_CL.contains("__kernel void group_norm"));
    }

    #[test]
    fn test_opencl_source_uses_tree_reduction() {
        // Verify barrier-based tree reduction pattern
        assert!(LAYER_NORM_CL.contains("barrier(CLK_LOCAL_MEM_FENCE)"));
        assert!(LAYER_NORM_CL.contains("smem[lid]"));
    }

    #[test]
    fn test_opencl_source_uses_rsqrt() {
        assert!(LAYER_NORM_CL.contains("rsqrt"));
    }

    // ===============================================================
    // NormStats
    // ===============================================================

    #[test]
    fn test_norm_stats_empty() {
        let stats = NormStats::empty();
        assert!(stats.mean.is_empty());
        assert!(stats.variance.is_empty());
        assert!(stats.rms.is_empty());
        assert_eq!(stats.compute_time_us, 0);
    }

    // ===============================================================
    // Negative input values
    // ===============================================================

    #[test]
    fn test_layer_norm_negative_inputs() {
        let input = [-4.0_f32, -3.0, -2.0, -1.0];
        let mut output = vec![0.0; 4];
        let ln = LayerNorm::new(4, NormConfig::new(NormType::LayerNorm));
        ln.forward(&input, &mut output);

        // Same distribution as [1,2,3,4] but negated
        assert_near(output[0], -1.3416, 1e-3, "neg[0]");
        assert_near(output[3], 1.3416, 1e-3, "neg[3]");
    }

    #[test]
    fn test_rms_norm_negative_inputs() {
        let input = [-1.0_f32, -2.0, -3.0, -4.0];
        let mut output = vec![0.0; 4];
        let rms = RmsNorm::new(4, NormConfig::new(NormType::RMSNorm));
        rms.forward(&input, &mut output);

        // RMS doesn't center, so negatives stay negative
        for &v in &output {
            assert!(v.is_finite(), "negative input produced non-finite");
            assert!(v <= 0.0, "negative input should stay negative in RMS");
        }
    }
}
