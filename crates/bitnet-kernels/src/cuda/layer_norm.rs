//! Extended layer normalization CUDA kernels with CPU reference implementations.
//!
//! This module complements the existing [`super::layernorm`] module with
//! additional normalization variants commonly used in transformer inference:
//!
//! - [`rms_norm_forward`]: RMS normalization (LLaMA / BitNet style)
//! - [`layer_norm_forward`]: Standard LayerNorm with mean subtraction
//! - [`rms_norm_with_residual`]: Fused RMS norm + residual addition
//! - [`group_norm_forward`]: Group normalization (channel-grouped)
//! - [`batch_norm_forward_inference`]: Batch normalization (eval mode)
//!
//! All functions provide a CPU reference implementation that is always compiled.
//! GPU-gated launch stubs are behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to layer normalization kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerNormError {
    /// A dimension or slice length was invalid.
    InvalidShape {
        /// Human-readable explanation.
        reason: String,
    },
    /// Epsilon was not positive-finite.
    InvalidEpsilon {
        /// The invalid value that was provided.
        value: f32,
    },
    /// The operation requires GPU but no runtime is available.
    GpuUnavailable,
}

impl fmt::Display for LayerNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { reason } => write!(f, "LayerNorm invalid shape: {reason}"),
            Self::InvalidEpsilon { value } => {
                write!(f, "LayerNorm eps must be positive and finite, got {value}")
            }
            Self::GpuUnavailable => write!(f, "LayerNorm: GPU runtime unavailable"),
        }
    }
}

impl std::error::Error for LayerNormError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration shared across all normalization variants in this module.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Epsilon for numerical stability inside sqrt.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (weight/bias).
    pub elementwise_affine: bool,
    /// Dimensions to normalise over (e.g. `[hidden_dim]`).
    pub normalized_shape: Vec<usize>,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self { eps: 1e-5, elementwise_affine: true, normalized_shape: Vec::new() }
    }
}

impl LayerNormConfig {
    /// Create a configuration with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LayerNormError::InvalidEpsilon`] if `eps` is not positive-finite.
    pub fn new(
        eps: f32,
        elementwise_affine: bool,
        normalized_shape: Vec<usize>,
    ) -> Result<Self, LayerNormError> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(LayerNormError::InvalidEpsilon { value: eps });
        }
        Ok(Self { eps, elementwise_affine, normalized_shape })
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source (feature-gated)
// ---------------------------------------------------------------------------

/// CUDA kernel source for extended layer normalization operations.
///
/// Contains five kernels:
/// - `layernorm_ext_f32`: Full LayerNorm with mean subtraction + affine
/// - `rmsnorm_ext_f32`: RMS norm (no mean subtraction)
/// - `rmsnorm_residual_f32`: Fused RMS norm + residual add
/// - `groupnorm_f32`: Group normalization
/// - `batchnorm_inference_f32`: Batch normalization in eval mode
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const LAYER_NORM_KERNEL_SRC: &str = r#"
extern "C" {

__global__ void layernorm_ext_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int normalized_shape,
    float eps,
    int elementwise_affine
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in = input + row * normalized_shape;
    float* row_out = output + row * normalized_shape;

    float sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride)
        sum += row_in[i];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float s_mean;
    if (tid == 0) s_mean = sum / (float)normalized_shape;
    __syncthreads();
    float mean = s_mean;

    float var_sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    __shared__ float s_var;
    if (tid == 0) s_var = var_sum / (float)normalized_shape;
    __syncthreads();
    float inv_std = rsqrtf(s_var + eps);

    for (int i = tid; i < normalized_shape; i += stride) {
        float normed = (row_in[i] - mean) * inv_std;
        if (elementwise_affine)
            normed = normed * weight[i] + bias[i];
        row_out[i] = normed;
    }
}

__global__ void rmsnorm_ext_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int normalized_shape,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in = input + row * normalized_shape;
    float* row_out = output + row * normalized_shape;

    float sq_sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        float v = row_in[i];
        sq_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sq_sum += __shfl_down_sync(0xFFFFFFFF, sq_sum, offset);
    __shared__ float s_rms;
    if (tid == 0) s_rms = rsqrtf(sq_sum / (float)normalized_shape + eps);
    __syncthreads();
    float inv_rms = s_rms;

    for (int i = tid; i < normalized_shape; i += stride)
        row_out[i] = row_in[i] * inv_rms * weight[i];
}

__global__ void rmsnorm_residual_f32(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int normalized_shape,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in  = input    + row * normalized_shape;
    const float* row_res = residual + row * normalized_shape;
    float* row_out = output + row * normalized_shape;

    float sq_sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        float v = row_in[i] + row_res[i];
        sq_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sq_sum += __shfl_down_sync(0xFFFFFFFF, sq_sum, offset);
    __shared__ float s_rms;
    if (tid == 0) s_rms = rsqrtf(sq_sum / (float)normalized_shape + eps);
    __syncthreads();
    float inv_rms = s_rms;

    for (int i = tid; i < normalized_shape; i += stride)
        row_out[i] = (row_in[i] + row_res[i]) * inv_rms * weight[i];
}

__global__ void groupnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int num_channels,
    int channels_per_group,
    int spatial_size,
    float eps,
    int elementwise_affine
) {
    int idx = blockIdx.x;
    int batch   = idx / (num_channels / channels_per_group);
    int group   = idx % (num_channels / channels_per_group);
    int g_start = group * channels_per_group;
    int elems   = channels_per_group * spatial_size;
    int tid     = threadIdx.x;
    int stride  = blockDim.x;

    const float* base_in  = input  + batch * num_channels * spatial_size;
    float*       base_out = output + batch * num_channels * spatial_size;

    float sum = 0.0f;
    for (int i = tid; i < elems; i += stride) {
        int c = g_start + i / spatial_size;
        int s = i % spatial_size;
        sum += base_in[c * spatial_size + s];
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    __shared__ float s_mean;
    if (tid == 0) s_mean = sum / (float)elems;
    __syncthreads();
    float mean = s_mean;

    float var_sum = 0.0f;
    for (int i = tid; i < elems; i += stride) {
        int c = g_start + i / spatial_size;
        int s = i % spatial_size;
        float diff = base_in[c * spatial_size + s] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    __shared__ float s_var;
    if (tid == 0) s_var = var_sum / (float)elems;
    __syncthreads();
    float inv_std = rsqrtf(s_var + eps);

    for (int i = tid; i < elems; i += stride) {
        int c = g_start + i / spatial_size;
        int s = i % spatial_size;
        float normed = (base_in[c * spatial_size + s] - mean) * inv_std;
        if (elementwise_affine)
            normed = normed * weight[c] + bias[c];
        base_out[c * spatial_size + s] = normed;
    }
}

__global__ void batchnorm_inference_f32(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int num_features,
    float eps
) {
    int c   = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float inv_std = rsqrtf(var[c] + eps);
    float w = weight[c];
    float b = bias[c];
    float m = mean[c];

    for (int n = tid; n < batch_size; n += stride) {
        int idx = n * num_features + c;
        output[idx] = (input[idx] - m) * inv_std * w + b;
    }
}

} // extern "C"
"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_norm_inputs(
    input_len: usize,
    normalized_shape: usize,
    weight_len: usize,
) -> Result<usize, LayerNormError> {
    if normalized_shape == 0 {
        return Err(LayerNormError::InvalidShape {
            reason: "normalized_shape must be non-zero".into(),
        });
    }
    if input_len == 0 {
        return Ok(0);
    }
    if !input_len.is_multiple_of(normalized_shape) {
        return Err(LayerNormError::InvalidShape {
            reason: format!(
                "input length {input_len} is not a multiple of normalized_shape \
                 {normalized_shape}"
            ),
        });
    }
    if weight_len < normalized_shape {
        return Err(LayerNormError::InvalidShape {
            reason: format!("weight length {weight_len} < normalized_shape {normalized_shape}"),
        });
    }
    Ok(input_len / normalized_shape)
}

// ---------------------------------------------------------------------------
// CPU reference: RMS norm
// ---------------------------------------------------------------------------

/// RMS normalization (LLaMA / BitNet style).
///
/// `y[i] = (x[i] / rms(x)) * weight[i]`
/// where `rms(x) = sqrt(mean(x²) + eps)`.
pub fn rms_norm_forward(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let ns = weight.len();
    let n_rows = validate_norm_inputs(input.len(), ns, ns)?;
    if n_rows == 0 {
        return Ok(());
    }
    if output.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("output length {} < input length {}", output.len(), input.len()),
        });
    }
    for row in 0..n_rows {
        let start = row * ns;
        let end = start + ns;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        let sq_sum: f32 = row_in.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sq_sum / ns as f32 + eps).sqrt();

        for i in 0..ns {
            row_out[i] = row_in[i] * inv_rms * weight[i];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: standard LayerNorm
// ---------------------------------------------------------------------------

/// Standard layer normalization.
///
/// `y[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]`
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let ns = weight.len();
    let n_rows = validate_norm_inputs(input.len(), ns, ns)?;
    if n_rows == 0 {
        return Ok(());
    }
    if bias.len() < ns {
        return Err(LayerNormError::InvalidShape {
            reason: format!("bias length {} < normalized_shape {ns}", bias.len()),
        });
    }
    if output.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("output length {} < input length {}", output.len(), input.len()),
        });
    }
    for row in 0..n_rows {
        let start = row * ns;
        let end = start + ns;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        let mean: f32 = row_in.iter().copied().sum::<f32>() / ns as f32;
        let var: f32 = row_in.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / ns as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..ns {
            row_out[i] = (row_in[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: RMS norm + residual
// ---------------------------------------------------------------------------

/// Fused RMS normalization with residual addition.
///
/// `y[i] = ((input[i] + residual[i]) / rms(input + residual)) * weight[i]`
pub fn rms_norm_with_residual(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let ns = weight.len();
    let n_rows = validate_norm_inputs(input.len(), ns, ns)?;
    if n_rows == 0 {
        return Ok(());
    }
    if residual.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("residual length {} < input length {}", residual.len(), input.len()),
        });
    }
    if output.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("output length {} < input length {}", output.len(), input.len()),
        });
    }
    for row in 0..n_rows {
        let start = row * ns;
        let end = start + ns;
        let row_in = &input[start..end];
        let row_res = &residual[start..end];
        let row_out = &mut output[start..end];

        let sq_sum: f32 = row_in.iter().zip(row_res.iter()).map(|(&a, &b)| (a + b) * (a + b)).sum();
        let inv_rms = 1.0 / (sq_sum / ns as f32 + eps).sqrt();

        for i in 0..ns {
            row_out[i] = (row_in[i] + row_res[i]) * inv_rms * weight[i];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: group norm
// ---------------------------------------------------------------------------

/// Group normalization.
///
/// Input layout: `[batch, num_channels]`. Channels are split into
/// `num_groups` groups and each group is normalised independently.
#[allow(clippy::too_many_arguments)]
pub fn group_norm_forward(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    num_groups: usize,
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let num_channels = weight.len();
    if num_channels == 0 || num_groups == 0 {
        return Err(LayerNormError::InvalidShape {
            reason: "num_channels and num_groups must be non-zero".into(),
        });
    }
    if !num_channels.is_multiple_of(num_groups) {
        return Err(LayerNormError::InvalidShape {
            reason: format!("num_channels {num_channels} not divisible by num_groups {num_groups}"),
        });
    }
    if bias.len() < num_channels {
        return Err(LayerNormError::InvalidShape {
            reason: format!("bias length {} < num_channels {num_channels}", bias.len()),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    if !input.len().is_multiple_of(num_channels) {
        return Err(LayerNormError::InvalidShape {
            reason: format!(
                "input length {} not divisible by num_channels {num_channels}",
                input.len()
            ),
        });
    }
    let batch_size = input.len() / num_channels;
    if output.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("output length {} < input length {}", output.len(), input.len()),
        });
    }

    let channels_per_group = num_channels / num_groups;

    for b in 0..batch_size {
        for g in 0..num_groups {
            let c_start = g * channels_per_group;
            let c_end = c_start + channels_per_group;

            let mut sum = 0.0_f32;
            let mut count = 0usize;
            for c in c_start..c_end {
                let idx = b * num_channels + c;
                sum += input[idx];
                count += 1;
            }
            let mean = sum / count as f32;

            let mut var_sum = 0.0_f32;
            for c in c_start..c_end {
                let idx = b * num_channels + c;
                let diff = input[idx] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / count as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for c in c_start..c_end {
                let idx = b * num_channels + c;
                output[idx] = (input[idx] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: batch norm inference
// ---------------------------------------------------------------------------

/// Batch normalization in inference (eval) mode.
///
/// Uses pre-computed running mean/variance:
/// `y[n,c] = (x[n,c] - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c]`
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward_inference(
    input: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let num_features = weight.len();
    if num_features == 0 {
        return Err(LayerNormError::InvalidShape {
            reason: "num_features must be non-zero".into(),
        });
    }
    if mean.len() < num_features {
        return Err(LayerNormError::InvalidShape {
            reason: format!("mean length {} < num_features {num_features}", mean.len()),
        });
    }
    if var.len() < num_features {
        return Err(LayerNormError::InvalidShape {
            reason: format!("var length {} < num_features {num_features}", var.len()),
        });
    }
    if bias.len() < num_features {
        return Err(LayerNormError::InvalidShape {
            reason: format!("bias length {} < num_features {num_features}", bias.len()),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    if !input.len().is_multiple_of(num_features) {
        return Err(LayerNormError::InvalidShape {
            reason: format!(
                "input length {} not divisible by num_features {num_features}",
                input.len()
            ),
        });
    }
    let batch_size = input.len() / num_features;
    if output.len() < input.len() {
        return Err(LayerNormError::InvalidShape {
            reason: format!("output length {} < input length {}", output.len(), input.len()),
        });
    }

    for n in 0..batch_size {
        for c in 0..num_features {
            let idx = n * num_features + c;
            let inv_std = 1.0 / (var[c] + eps).sqrt();
            output[idx] = (input[idx] - mean[c]) * inv_std * weight[c] + bias[c];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU launch stubs (feature-gated)
// ---------------------------------------------------------------------------

/// Launch stub for the RMS norm CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_rms_norm_cuda(
    _input: &[f32],
    _weight: &[f32],
    _eps: f32,
    _output: &mut [f32],
) -> Result<(), LayerNormError> {
    Err(LayerNormError::GpuUnavailable)
}

/// Launch stub for the standard LayerNorm CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_layer_norm_cuda(
    _input: &[f32],
    _weight: &[f32],
    _bias: &[f32],
    _eps: f32,
    _output: &mut [f32],
) -> Result<(), LayerNormError> {
    Err(LayerNormError::GpuUnavailable)
}

/// Launch stub for the fused RMS norm + residual CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_rms_norm_residual_cuda(
    _input: &[f32],
    _residual: &[f32],
    _weight: &[f32],
    _eps: f32,
    _output: &mut [f32],
) -> Result<(), LayerNormError> {
    Err(LayerNormError::GpuUnavailable)
}

/// Launch stub for the group norm CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_group_norm_cuda(
    _input: &[f32],
    _weight: &[f32],
    _bias: &[f32],
    _num_groups: usize,
    _eps: f32,
    _output: &mut [f32],
) -> Result<(), LayerNormError> {
    Err(LayerNormError::GpuUnavailable)
}

/// Launch stub for the batch norm inference CUDA kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_batch_norm_inference_cuda(
    _input: &[f32],
    _mean: &[f32],
    _var: &[f32],
    _weight: &[f32],
    _bias: &[f32],
    _eps: f32,
    _output: &mut [f32],
) -> Result<(), LayerNormError> {
    Err(LayerNormError::GpuUnavailable)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[allow(unused_mut)]
mod tests {
    use super::*;

    // -- LayerNormError Display/Error impls --------------------------------

    #[test]
    fn test_error_display_invalid_shape() {
        let e = LayerNormError::InvalidShape { reason: "bad dim".into() };
        assert!(e.to_string().contains("bad dim"));
    }

    #[test]
    fn test_error_display_invalid_epsilon() {
        let e = LayerNormError::InvalidEpsilon { value: -1.0 };
        let msg = e.to_string();
        assert!(msg.contains("-1"));
        assert!(msg.contains("positive and finite"));
    }

    #[test]
    fn test_error_display_gpu_unavailable() {
        let e = LayerNormError::GpuUnavailable;
        assert!(e.to_string().contains("GPU"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(LayerNormError::InvalidShape { reason: "x".into() });
        assert!(e.to_string().contains("x"));
    }

    // -- Config tests ------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = LayerNormConfig::default();
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(cfg.elementwise_affine);
        assert!(cfg.normalized_shape.is_empty());
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = LayerNormConfig::new(1e-6, false, vec![128]).unwrap();
        assert!((cfg.eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.elementwise_affine);
        assert_eq!(cfg.normalized_shape, vec![128]);
    }

    #[test]
    fn test_config_rejects_zero_eps() {
        assert!(LayerNormConfig::new(0.0, true, vec![]).is_err());
    }

    #[test]
    fn test_config_rejects_negative_eps() {
        assert!(LayerNormConfig::new(-1e-5, true, vec![]).is_err());
    }

    #[test]
    fn test_config_rejects_nan_eps() {
        assert!(LayerNormConfig::new(f32::NAN, true, vec![]).is_err());
    }

    #[test]
    fn test_config_rejects_inf_eps() {
        assert!(LayerNormConfig::new(f32::INFINITY, true, vec![]).is_err());
    }

    // -- RMS norm tests ----------------------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();

        let sq_sum: f32 = input.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq_sum / 4.0 + 1e-5_f32).sqrt();
        for (i, &v) in output.iter().enumerate() {
            let expected = input[i] * inv_rms;
            assert!((v - expected).abs() < 1e-5, "idx={i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [2.0_f32, 0.5, 1.0, 3.0];
        let mut output = [0.0_f32; 4];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();

        let ones = [1.0_f32; 4];
        let mut ref_out = [0.0_f32; 4];
        rms_norm_forward(&input, &ones, 1e-5, &mut ref_out).unwrap();

        for i in 0..4 {
            let expected = ref_out[i] * weight[i];
            assert!((output[i] - expected).abs() < 1e-5, "idx={i}");
        }
    }

    #[test]
    fn test_rms_norm_multiple_rows() {
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let weight = [1.0; 3];
        let mut output = [0.0_f32; 6];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_empty() {
        let weight = [1.0_f32; 4];
        let mut output = [0.0_f32; 0];
        rms_norm_forward(&[], &weight, 1e-5, &mut output).unwrap();
    }

    #[test]
    fn test_rms_norm_preserves_sign() {
        let input = [-3.0_f32, -1.0, 0.0, 1.0, 3.0];
        let weight = [1.0; 5];
        let mut output = [0.0_f32; 5];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp == 0.0 {
                assert!(out.abs() < 1e-6, "zero at {i}");
            } else {
                assert_eq!(inp.signum(), out.signum(), "sign at {i}");
            }
        }
    }

    #[test]
    fn test_rms_norm_rejects_misaligned() {
        let weight = [1.0; 3];
        let mut output = [0.0_f32; 5];
        let r = rms_norm_forward(&[1.0, 2.0, 3.0, 4.0, 5.0], &weight, 1e-5, &mut output);
        assert!(r.is_err());
    }

    #[test]
    fn test_rms_norm_rejects_short_output() {
        let input = [1.0_f32; 4];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 2];
        assert!(rms_norm_forward(&input, &weight, 1e-5, &mut output).is_err());
    }

    // -- Standard LayerNorm tests ------------------------------------------

    #[test]
    fn test_layer_norm_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();

        let mean = 2.5_f32;
        let var = 1.25_f32;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for (i, &v) in output.iter().enumerate() {
            let expected = (input[i] - mean) * inv_std;
            assert!((v - expected).abs() < 1e-5, "idx={i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [2.0; 4];
        let bias = [0.5; 4];
        let mut output = [0.0_f32; 4];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();

        let w_one = [1.0; 4];
        let b_zero = [0.0; 4];
        let mut ref_out = [0.0_f32; 4];
        layer_norm_forward(&input, &w_one, &b_zero, 1e-5, &mut ref_out).unwrap();
        for i in 0..4 {
            let expected = ref_out[i] * 2.0 + 0.5;
            assert!((output[i] - expected).abs() < 1e-5, "idx={i}");
        }
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_var() {
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0_f32; 256];
        let bias = vec![0.0_f32; 256];
        let mut output = vec![0.0_f32; 256];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();

        let mean: f32 = output.iter().sum::<f32>() / 256.0;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 256.0;
        assert!(mean.abs() < 1e-4, "mean={mean}");
        assert!((var - 1.0).abs() < 0.01, "var={var}");
    }

    #[test]
    fn test_layer_norm_empty() {
        let weight = [1.0_f32; 4];
        let bias = [0.0; 4];
        let mut output = [];
        layer_norm_forward(&[], &weight, &bias, 1e-5, &mut output).unwrap();
    }

    #[test]
    fn test_layer_norm_rejects_short_bias() {
        let input = [1.0_f32; 4];
        let weight = [1.0; 4];
        let bias = [0.0; 2]; // too short
        let mut output = [0.0_f32; 4];
        assert!(layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).is_err());
    }

    #[test]
    fn test_layer_norm_uniform_input() {
        let input = [5.0_f32; 8];
        let weight = [1.0; 8];
        let bias = [0.0; 8];
        let mut output = [0.0_f32; 8];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.abs() < 1e-3));
    }

    // -- RMS norm with residual tests --------------------------------------

    #[test]
    fn test_rms_residual_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let residual = [0.5_f32, 0.5, 0.5, 0.5];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        rms_norm_with_residual(&input, &residual, &weight, 1e-5, &mut output).unwrap();

        // Equivalent to rms_norm_forward(input + residual)
        let combined: Vec<f32> = input.iter().zip(residual.iter()).map(|(a, b)| a + b).collect();
        let mut expected = [0.0_f32; 4];
        rms_norm_forward(&combined, &weight, 1e-5, &mut expected).unwrap();

        for i in 0..4 {
            assert!((output[i] - expected[i]).abs() < 1e-5, "idx={i}");
        }
    }

    #[test]
    fn test_rms_residual_zero_residual() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let residual = [0.0_f32; 4];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        rms_norm_with_residual(&input, &residual, &weight, 1e-5, &mut output).unwrap();

        let mut expected = [0.0_f32; 4];
        rms_norm_forward(&input, &weight, 1e-5, &mut expected).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_rms_residual_rejects_short_residual() {
        let input = [1.0_f32; 4];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        assert!(rms_norm_with_residual(&input, &[0.0; 2], &weight, 1e-5, &mut output,).is_err());
    }

    #[test]
    fn test_rms_residual_empty() {
        let weight = [1.0_f32; 4];
        let mut output = [];
        rms_norm_with_residual(&[], &[], &weight, 1e-5, &mut output).unwrap();
    }

    // -- Group norm tests --------------------------------------------------

    #[test]
    fn test_group_norm_basic() {
        // 1 batch, 4 channels, 2 groups of 2
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        group_norm_forward(&input, &weight, &bias, 2, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_group_norm_single_group_matches_layernorm() {
        // Single group = layer norm
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        group_norm_forward(&input, &weight, &bias, 1, 1e-5, &mut output).unwrap();

        let mut ln_out = [0.0_f32; 4];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut ln_out).unwrap();

        for i in 0..4 {
            assert!((output[i] - ln_out[i]).abs() < 1e-5, "idx={i}");
        }
    }

    #[test]
    fn test_group_norm_rejects_indivisible() {
        let weight = [1.0; 5];
        let bias = [0.0; 5];
        let mut output = [0.0_f32; 5];
        assert!(group_norm_forward(&[1.0; 5], &weight, &bias, 2, 1e-5, &mut output,).is_err());
    }

    #[test]
    fn test_group_norm_rejects_zero_groups() {
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        assert!(group_norm_forward(&[1.0; 4], &weight, &bias, 0, 1e-5, &mut output,).is_err());
    }

    #[test]
    fn test_group_norm_empty() {
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [];
        group_norm_forward(&[], &weight, &bias, 2, 1e-5, &mut output).unwrap();
    }

    #[test]
    fn test_group_norm_multi_batch() {
        // 2 batches, 4 channels, 2 groups
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0_f32];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 8];
        group_norm_forward(&input, &weight, &bias, 2, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // -- Batch norm inference tests ----------------------------------------

    #[test]
    fn test_batch_norm_inference_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mean = [0.0; 2];
        let var = [1.0; 2];
        let weight = [1.0; 2];
        let bias = [0.0; 2];
        let mut output = [0.0_f32; 4];
        batch_norm_forward_inference(&input, &mean, &var, &weight, &bias, 1e-5, &mut output)
            .unwrap();

        // mean=0, var=1, w=1, b=0 → output ≈ input
        for (i, &v) in output.iter().enumerate() {
            let inv_std = 1.0 / (1.0_f32 + 1e-5).sqrt();
            let expected = input[i] * inv_std;
            assert!((v - expected).abs() < 1e-4, "idx={i}: {v} vs {expected}");
        }
    }

    #[test]
    fn test_batch_norm_inference_with_stats() {
        // 2 samples, 3 features
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32];
        let mean = [2.5, 3.5, 4.5];
        let var = [2.25, 2.25, 2.25];
        let weight = [1.0; 3];
        let bias = [0.0; 3];
        let mut output = [0.0_f32; 6];
        batch_norm_forward_inference(&input, &mean, &var, &weight, &bias, 1e-5, &mut output)
            .unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_batch_norm_inference_rejects_short_mean() {
        let input = [1.0_f32; 4];
        let weight = [1.0; 2];
        let bias = [0.0; 2];
        let mut output = [0.0_f32; 4];
        assert!(
            batch_norm_forward_inference(
                &input,
                &[0.0],
                &[1.0; 2],
                &weight,
                &bias,
                1e-5,
                &mut output,
            )
            .is_err()
        );
    }

    #[test]
    fn test_batch_norm_inference_rejects_short_var() {
        let input = [1.0_f32; 4];
        let weight = [1.0; 2];
        let bias = [0.0; 2];
        let mut output = [0.0_f32; 4];
        assert!(
            batch_norm_forward_inference(
                &input,
                &[0.0; 2],
                &[1.0],
                &weight,
                &bias,
                1e-5,
                &mut output,
            )
            .is_err()
        );
    }

    #[test]
    fn test_batch_norm_inference_empty() {
        let weight = [1.0; 2];
        let bias = [0.0; 2];
        let mut output = [];
        batch_norm_forward_inference(&[], &[0.0; 2], &[1.0; 2], &weight, &bias, 1e-5, &mut output)
            .unwrap();
    }

    // -- Numerical stability -----------------------------------------------

    #[test]
    fn test_rms_norm_large_values() {
        let input = [1e6_f32, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_layer_norm_large_values() {
        let input = [1e6_f32, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_tiny_values() {
        let input = [1e-10_f32, 2e-10, 3e-10, 4e-10];
        let weight = [1.0; 4];
        let mut output = [0.0_f32; 4];
        rms_norm_forward(&input, &weight, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_layer_norm_tiny_values() {
        let input = [1e-10_f32, 2e-10, 3e-10, 4e-10];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        let mut output = [0.0_f32; 4];
        layer_norm_forward(&input, &weight, &bias, 1e-5, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // -- GPU stub tests ----------------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_gpu_rms_norm_stub() {
        let mut output = [0.0_f32; 4];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let r = launch_rms_norm_cuda(&[1.0; 4], &[1.0; 4], 1e-5, &mut output);
            assert!(r.is_err());
        }
        let _ = output;
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_gpu_layer_norm_stub() {
        let mut output = [0.0_f32; 4];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let r = launch_layer_norm_cuda(&[1.0; 4], &[1.0; 4], &[0.0; 4], 1e-5, &mut output);
            assert!(r.is_err());
        }
        let _ = output;
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_gpu_rms_residual_stub() {
        let mut output = [0.0_f32; 4];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let r =
                launch_rms_norm_residual_cuda(&[1.0; 4], &[0.5; 4], &[1.0; 4], 1e-5, &mut output);
            assert!(r.is_err());
        }
        let _ = output;
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_gpu_group_norm_stub() {
        let mut output = [0.0_f32; 4];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let r = launch_group_norm_cuda(&[1.0; 4], &[1.0; 4], &[0.0; 4], 2, 1e-5, &mut output);
            assert!(r.is_err());
        }
        let _ = output;
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_gpu_batch_norm_inference_stub() {
        let mut output = [0.0_f32; 4];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let r = launch_batch_norm_inference_cuda(
                &[1.0; 4],
                &[0.0; 2],
                &[1.0; 2],
                &[1.0; 2],
                &[0.0; 2],
                1e-5,
                &mut output,
            );
            assert!(r.is_err());
        }
        let _ = output;
    }
}
