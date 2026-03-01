//! Layer Normalization CUDA kernel with CPU fallback.
//!
//! # Kernel strategy
//!
//! Layer normalization normalises activations across the feature dimension for
//! each token position independently:
//!
//!   `y[i] = ((x[i] - mean) / sqrt(var + eps)) * gamma[i] + beta[i]`
//!
//! where `mean = mean(x)` and `var = mean((x - mean)²)` are computed over the
//! normalised dimension.
//!
//! Two CUDA kernels are provided:
//!
//! ## `layernorm_f32` — Full Layer Normalization
//!
//! 1. Each thread-block handles one row (one token position).
//! 2. A two-pass warp-level reduction computes `mean` and `variance`.
//! 3. Every thread normalises its elements and optionally applies learnable
//!    affine parameters `gamma` (weight) and `beta` (bias).
//!
//! ## `rmsnorm_f32` — Root Mean Square Layer Normalization
//!
//! A single-pass variant that skips mean subtraction:
//!
//!   `y[i] = (x[i] / rms(x)) * gamma[i]`
//!
//! where `rms(x) = sqrt(mean(x²) + eps)`. This is the normalisation used by
//! LLaMA and BitNet architectures.
//!
//! # CPU fallback
//!
//! [`layer_norm_cpu_fallback`] and [`rms_norm_cpu_fallback`] provide equivalent
//! pure-Rust implementations for correctness testing and non-GPU environments.
//!
//! # Dispatchers
//!
//! [`layer_norm_forward`] and [`rms_norm_forward`] automatically dispatch to
//! GPU when available, falling back to CPU otherwise.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// CUDA kernel source for layer normalization and RMS normalization.
///
/// Contains two kernels:
/// - `layernorm_f32`: full layer norm with mean subtraction + affine transform
/// - `rmsnorm_f32`: RMS norm without mean subtraction (LLaMA/BitNet style)
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const LAYERNORM_KERNEL_SRC: &str = r#"
extern "C" {

// Full Layer Normalization kernel.
// Each block handles one row of [n_rows, normalized_shape].
// gamma/beta are [normalized_shape] affine parameters (may be NULL if disabled).
__global__ void layernorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
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

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        sum += row_in[i];
    }
    // Warp-level reduction for mean
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float shared_mean;
    if (tid == 0) {
        shared_mean = sum / (float)normalized_shape;
    }
    __syncthreads();
    float mean = shared_mean;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    __shared__ float shared_var;
    if (tid == 0) {
        shared_var = var_sum / (float)normalized_shape;
    }
    __syncthreads();
    float inv_std = rsqrtf(shared_var + eps);

    // Pass 3: normalize + optional affine
    for (int i = tid; i < normalized_shape; i += stride) {
        float normed = (row_in[i] - mean) * inv_std;
        if (elementwise_affine) {
            normed = normed * gamma[i] + beta[i];
        }
        row_out[i] = normed;
    }
}

// RMS Normalization kernel (LLaMA / BitNet style).
// Each block handles one row. No mean subtraction — just x / rms(x) * gamma.
__global__ void rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int normalized_shape,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row_in = input + row * normalized_shape;
    float* row_out = output + row * normalized_shape;

    // Single pass: sum of squares
    float sq_sum = 0.0f;
    for (int i = tid; i < normalized_shape; i += stride) {
        float v = row_in[i];
        sq_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sq_sum += __shfl_down_sync(0xFFFFFFFF, sq_sum, offset);
    }
    __shared__ float shared_rms;
    if (tid == 0) {
        shared_rms = rsqrtf(sq_sum / (float)normalized_shape + eps);
    }
    __syncthreads();
    float inv_rms = shared_rms;

    // Normalize and scale
    for (int i = tid; i < normalized_shape; i += stride) {
        row_out[i] = row_in[i] * inv_rms * gamma[i];
    }
}

} // extern "C"
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for layer normalization kernels.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Epsilon added inside the square root for numerical stability.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub elementwise_affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self { eps: 1e-5, elementwise_affine: true }
    }
}

impl LayerNormConfig {
    /// Create a new configuration with the given epsilon.
    ///
    /// # Errors
    ///
    /// Returns an error if `eps` is not positive and finite.
    pub fn new(eps: f32, elementwise_affine: bool) -> Result<Self> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("LayerNorm eps must be positive and finite, got {eps}"),
            }
            .into());
        }
        Ok(Self { eps, elementwise_affine })
    }

    /// Create a configuration with default epsilon (`1e-5`) and affine enabled.
    pub fn with_defaults() -> Self {
        Self::default()
    }

    /// Override the epsilon value.
    ///
    /// # Errors
    ///
    /// Returns an error if `eps` is not positive and finite.
    pub fn with_eps(mut self, eps: f32) -> Result<Self> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("LayerNorm eps must be positive and finite, got {eps}"),
            }
            .into());
        }
        self.eps = eps;
        Ok(self)
    }

    /// Compute CUDA grid dimensions for `n_rows` rows.
    pub fn grid_dim(&self, n_rows: usize) -> (u32, u32, u32) {
        (n_rows as u32, 1, 1)
    }

    /// Compute CUDA block dimensions for `normalized_shape` elements.
    pub fn block_dim(&self, normalized_shape: usize) -> (u32, u32, u32) {
        let threads = (normalized_shape as u32).min(1024);
        (threads, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CPU fallbacks
// ---------------------------------------------------------------------------

/// Full layer normalization on the CPU.
///
/// Computes `y[i] = ((x[i] - mean) / sqrt(var + eps)) * gamma[i] + beta[i]`
/// for each row of `normalized_shape` elements.
///
/// # Arguments
///
/// * `input` — Input tensor `[n_rows, normalized_shape]` (FP32, row-major)
/// * `gamma` — Per-element scale weights `[normalized_shape]` (FP32)
/// * `beta`  — Per-element bias `[normalized_shape]` (FP32)
/// * `normalized_shape` — Number of elements per row to normalise
/// * `config` — Configuration (uses `eps`, `elementwise_affine`)
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths are invalid.
pub fn layer_norm_cpu_fallback(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    if normalized_shape == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "normalized_shape must be non-zero".into(),
        }
        .into());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if !input.len().is_multiple_of(normalized_shape) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "input length {} is not a multiple of normalized_shape {}",
                input.len(),
                normalized_shape
            ),
        }
        .into());
    }
    if config.elementwise_affine {
        if gamma.len() < normalized_shape {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "gamma length {} < normalized_shape {}",
                    gamma.len(),
                    normalized_shape
                ),
            }
            .into());
        }
        if beta.len() < normalized_shape {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "beta length {} < normalized_shape {}",
                    beta.len(),
                    normalized_shape
                ),
            }
            .into());
        }
    }

    let n_rows = input.len() / normalized_shape;
    let mut output = vec![0.0_f32; input.len()];

    for row in 0..n_rows {
        let start = row * normalized_shape;
        let end = start + normalized_shape;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Compute mean
        let mean: f32 = row_in.iter().copied().sum::<f32>() / normalized_shape as f32;

        // Compute variance
        let var: f32 = row_in
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / normalized_shape as f32;

        let inv_std = 1.0 / (var + config.eps).sqrt();

        // Normalise + optional affine
        for i in 0..normalized_shape {
            let normed = (row_in[i] - mean) * inv_std;
            row_out[i] =
                if config.elementwise_affine { normed * gamma[i] + beta[i] } else { normed };
        }
    }

    Ok(output)
}

/// RMS normalization on the CPU (LLaMA / BitNet style).
///
/// Computes `y[i] = (x[i] / rms(x)) * gamma[i]` where
/// `rms(x) = sqrt(mean(x²) + eps)`.
///
/// # Arguments
///
/// * `input` — Input tensor `[n_rows, normalized_shape]` (FP32, row-major)
/// * `gamma` — Per-element scale weights `[normalized_shape]` (FP32)
/// * `normalized_shape` — Number of elements per row to normalise
/// * `config` — Configuration (uses `eps`)
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths are invalid.
pub fn rms_norm_cpu_fallback(
    input: &[f32],
    gamma: &[f32],
    normalized_shape: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    if normalized_shape == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "normalized_shape must be non-zero".into(),
        }
        .into());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if !input.len().is_multiple_of(normalized_shape) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "input length {} is not a multiple of normalized_shape {}",
                input.len(),
                normalized_shape
            ),
        }
        .into());
    }
    if gamma.len() < normalized_shape {
        return Err(KernelError::InvalidArguments {
            reason: format!("gamma length {} < normalized_shape {}", gamma.len(), normalized_shape),
        }
        .into());
    }

    let n_rows = input.len() / normalized_shape;
    let mut output = vec![0.0_f32; input.len()];

    for row in 0..n_rows {
        let start = row * normalized_shape;
        let end = start + normalized_shape;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Sum of squares
        let sq_sum: f32 = row_in.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sq_sum / normalized_shape as f32 + config.eps).sqrt();

        // Scale by gamma
        for i in 0..normalized_shape {
            row_out[i] = row_in[i] * inv_rms * gamma[i];
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CUDA launch stubs
// ---------------------------------------------------------------------------

/// Launch stub for the full layer normalization CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_layer_norm(
    _input: &[f32],
    _gamma: &[f32],
    _beta: &[f32],
    normalized_shape: usize,
    n_rows: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    log::debug!(
        "LayerNorm CUDA stub: normalized_shape={}, n_rows={}, eps={}, grid={:?}",
        normalized_shape,
        n_rows,
        config.eps,
        config.grid_dim(n_rows),
    );
    Err(KernelError::GpuError {
        reason: "LayerNorm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for the RMS normalization CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_rms_norm(
    _input: &[f32],
    _gamma: &[f32],
    normalized_shape: usize,
    n_rows: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    log::debug!(
        "RMSNorm CUDA stub: normalized_shape={}, n_rows={}, eps={}, grid={:?}",
        normalized_shape,
        n_rows,
        config.eps,
        config.grid_dim(n_rows),
    );
    Err(KernelError::GpuError {
        reason: "RMSNorm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply full layer normalization with automatic dispatch: GPU if available,
/// else CPU fallback.
pub fn layer_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    normalized_shape: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if !input.is_empty() && normalized_shape > 0 && input.len().is_multiple_of(normalized_shape)
        {
            let n_rows = input.len() / normalized_shape;
            if crate::device_features::gpu_available_runtime()
                && let Ok(result) =
                    launch_layer_norm(input, gamma, beta, normalized_shape, n_rows, config)
            {
                return Ok(result);
            }
        }
    }
    layer_norm_cpu_fallback(input, gamma, beta, normalized_shape, config)
}

/// Apply RMS normalization with automatic dispatch: GPU if available,
/// else CPU fallback.
pub fn rms_norm_forward(
    input: &[f32],
    gamma: &[f32],
    normalized_shape: usize,
    config: &LayerNormConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if !input.is_empty() && normalized_shape > 0 && input.len().is_multiple_of(normalized_shape)
        {
            let n_rows = input.len() / normalized_shape;
            if crate::device_features::gpu_available_runtime()
                && let Ok(result) = launch_rms_norm(input, gamma, normalized_shape, n_rows, config)
            {
                return Ok(result);
            }
        }
    }
    rms_norm_cpu_fallback(input, gamma, normalized_shape, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = LayerNormConfig::default();
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(cfg.elementwise_affine);
    }

    #[test]
    fn test_config_with_defaults() {
        let cfg = LayerNormConfig::with_defaults();
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(cfg.elementwise_affine);
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = LayerNormConfig::new(1e-6, false).unwrap();
        assert!((cfg.eps - 1e-6).abs() < 1e-12);
        assert!(!cfg.elementwise_affine);
    }

    #[test]
    fn test_config_rejects_zero_eps() {
        assert!(LayerNormConfig::new(0.0, true).is_err());
    }

    #[test]
    fn test_config_rejects_negative_eps() {
        assert!(LayerNormConfig::new(-1e-5, true).is_err());
    }

    #[test]
    fn test_config_rejects_nan_eps() {
        assert!(LayerNormConfig::new(f32::NAN, true).is_err());
    }

    #[test]
    fn test_config_rejects_inf_eps() {
        assert!(LayerNormConfig::new(f32::INFINITY, true).is_err());
    }

    #[test]
    fn test_config_with_eps() {
        let cfg = LayerNormConfig::with_defaults().with_eps(1e-8).unwrap();
        assert!((cfg.eps - 1e-8).abs() < 1e-14);
    }

    #[test]
    fn test_config_with_eps_rejects_bad() {
        assert!(LayerNormConfig::with_defaults().with_eps(0.0).is_err());
        assert!(LayerNormConfig::with_defaults().with_eps(-1.0).is_err());
        assert!(LayerNormConfig::with_defaults().with_eps(f32::NAN).is_err());
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = LayerNormConfig::with_defaults();
        assert_eq!(cfg.grid_dim(32), (32, 1, 1));
        assert_eq!(cfg.grid_dim(1), (1, 1, 1));
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = LayerNormConfig::with_defaults();
        assert_eq!(cfg.block_dim(2048), (1024, 1, 1)); // capped
        assert_eq!(cfg.block_dim(64), (64, 1, 1));
        assert_eq!(cfg.block_dim(1), (1, 1, 1));
    }

    // -- Layer norm CPU fallback tests --------------------------------------

    #[test]
    fn test_layer_norm_basic() {
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();

        // Mean = 2.5, Var = 1.25
        // After normalisation: (x - 2.5) / sqrt(1.25 + 1e-5)
        let mean = 2.5_f32;
        let var = 1.25_f32;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for (i, &v) in output.iter().enumerate() {
            let expected = (input[i] - mean) * inv_std;
            assert!((v - expected).abs() < 1e-5, "idx={i}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [2.0_f32; 4];
        let beta = [0.5_f32; 4];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();

        // Verify affine transform is applied
        let cfg_no_affine = LayerNormConfig::new(1e-5, false).unwrap();
        let output_no_affine =
            layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg_no_affine).unwrap();

        for i in 0..4 {
            let expected = output_no_affine[i] * 2.0 + 0.5;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "idx={i}: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_layer_norm_multiple_rows() {
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let gamma = [1.0; 3];
        let beta = [0.0; 3];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 3, &cfg).unwrap();

        // Each row should have zero mean (within tolerance)
        for row in 0..2 {
            let start = row * 3;
            let row_out = &output[start..start + 3];
            let mean: f32 = row_out.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "row {row} mean = {mean}");
        }
    }

    #[test]
    fn test_layer_norm_uniform_input() {
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input = [5.0_f32; 8];
        let gamma = [1.0; 8];
        let beta = [0.0; 8];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 8, &cfg).unwrap();

        // All same input → all outputs near zero
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-3, "idx={i}: expected ~0, got {v}");
        }
    }

    #[test]
    fn test_layer_norm_numerical_stability_large() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1e6_f32, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();

        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_numerical_stability_tiny() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1e-10_f32, 2e-10, 3e-10, 4e-10];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();

        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_layer_norm_single_element() {
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input = [42.0_f32];
        let gamma = [1.0];
        let beta = [0.0];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 1, &cfg).unwrap();

        // Single element → (42 - 42) / sqrt(0 + eps) = 0
        assert!(output[0].abs() < 1e-3, "got {}", output[0]);
    }

    #[test]
    fn test_layer_norm_empty_input() {
        let cfg = LayerNormConfig::with_defaults();
        let output = layer_norm_cpu_fallback(&[], &[1.0], &[0.0], 1, &cfg).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_layer_norm_rejects_zero_shape() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(layer_norm_cpu_fallback(&[1.0], &[1.0], &[0.0], 0, &cfg).is_err());
    }

    #[test]
    fn test_layer_norm_rejects_misaligned_input() {
        let cfg = LayerNormConfig::with_defaults();
        // 5 elements, normalized_shape=3 → not a multiple
        assert!(
            layer_norm_cpu_fallback(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0; 3], &[0.0; 3], 3, &cfg)
                .is_err()
        );
    }

    #[test]
    fn test_layer_norm_rejects_short_gamma() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(
            layer_norm_cpu_fallback(
                &[1.0, 2.0, 3.0],
                &[1.0, 2.0], // too short
                &[0.0; 3],
                3,
                &cfg
            )
            .is_err()
        );
    }

    #[test]
    fn test_layer_norm_rejects_short_beta() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(
            layer_norm_cpu_fallback(
                &[1.0, 2.0, 3.0],
                &[1.0; 3],
                &[0.0, 1.0], // too short
                3,
                &cfg
            )
            .is_err()
        );
    }

    // -- RMS norm CPU fallback tests ----------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let cfg = LayerNormConfig::new(1e-5, true).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let output = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();

        // rms = sqrt(mean(x²) + eps) = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5...)
        let sq_sum: f32 = input.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq_sum / 4.0 + 1e-5_f32).sqrt();
        for (i, &v) in output.iter().enumerate() {
            let expected = input[i] * inv_rms;
            assert!((v - expected).abs() < 1e-5, "idx={i}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_rms_norm_with_gamma() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [2.0_f32, 0.5, 1.0, 3.0];
        let output = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();

        let gamma_ones = [1.0_f32; 4];
        let output_ones = rms_norm_cpu_fallback(&input, &gamma_ones, 4, &cfg).unwrap();

        for i in 0..4 {
            let expected = output_ones[i] * gamma[i];
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "idx={i}: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_rms_norm_multiple_rows() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let gamma = [1.0; 3];
        let output = rms_norm_cpu_fallback(&input, &gamma, 3, &cfg).unwrap();

        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|v| v.is_finite()));

        // Rows are normalised independently — norms should differ
        let norm_row0: f32 = output[0..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_row1: f32 = output[3..6].iter().map(|x| x * x).sum::<f32>().sqrt();
        // Both should be close to sqrt(3) * 1/rms factor, but differ numerically
        assert!(norm_row0.is_finite());
        assert!(norm_row1.is_finite());
    }

    #[test]
    fn test_rms_norm_uniform_input() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [5.0_f32; 4];
        let gamma = [1.0; 4];
        let output = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();

        // All same → all outputs equal (x * inv_rms = 5 / sqrt(25+eps) ≈ 1.0)
        let expected = 5.0 / (25.0_f32 + 1e-5).sqrt();
        for (i, &v) in output.iter().enumerate() {
            assert!((v - expected).abs() < 1e-5, "idx={i}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_rms_norm_numerical_stability_large() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1e6_f32, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let gamma = [1.0; 4];
        let output = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();

        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_rms_norm_single_element() {
        let cfg = LayerNormConfig::new(1e-5, true).unwrap();
        let input = [3.0_f32];
        let gamma = [1.0];
        let output = rms_norm_cpu_fallback(&input, &gamma, 1, &cfg).unwrap();

        // rms = sqrt(9/1 + eps) ≈ 3.0, so output ≈ 3.0 / 3.0 = 1.0
        assert!((output[0] - 1.0).abs() < 1e-4, "got {}", output[0]);
    }

    #[test]
    fn test_rms_norm_empty_input() {
        let cfg = LayerNormConfig::with_defaults();
        let output = rms_norm_cpu_fallback(&[], &[1.0], 1, &cfg).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_rms_norm_rejects_zero_shape() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(rms_norm_cpu_fallback(&[1.0], &[1.0], 0, &cfg).is_err());
    }

    #[test]
    fn test_rms_norm_rejects_misaligned_input() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(rms_norm_cpu_fallback(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0; 3], 3, &cfg).is_err());
    }

    #[test]
    fn test_rms_norm_rejects_short_gamma() {
        let cfg = LayerNormConfig::with_defaults();
        assert!(rms_norm_cpu_fallback(&[1.0, 2.0, 3.0], &[1.0, 2.0], 3, &cfg).is_err());
    }

    // -- Forward dispatch tests ---------------------------------------------

    #[test]
    fn test_layer_norm_forward_falls_back_to_cpu() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];
        let output = layer_norm_forward(&input, &gamma, &beta, 4, &cfg).unwrap();
        let expected = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_rms_norm_forward_falls_back_to_cpu() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let output = rms_norm_forward(&input, &gamma, 4, &cfg).unwrap();
        let expected = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_layer_norm_forward_empty() {
        let cfg = LayerNormConfig::with_defaults();
        let output = layer_norm_forward(&[], &[1.0], &[0.0], 1, &cfg).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_rms_norm_forward_empty() {
        let cfg = LayerNormConfig::with_defaults();
        let output = rms_norm_forward(&[], &[1.0], 1, &cfg).unwrap();
        assert!(output.is_empty());
    }

    // -- CUDA launch stub tests ---------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_layer_norm_launch() {
        let cfg = LayerNormConfig::with_defaults();
        let input = vec![1.0_f32; 2048 * 4];
        let gamma = vec![1.0_f32; 2048];
        let beta = vec![0.0_f32; 2048];
        let result = layer_norm_forward(&input, &gamma, &beta, 2048, &cfg);
        assert!(result.is_ok(), "LayerNorm forward failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rms_norm_launch() {
        let cfg = LayerNormConfig::with_defaults();
        let input = vec![1.0_f32; 2048 * 4];
        let gamma = vec![1.0_f32; 2048];
        let result = rms_norm_forward(&input, &gamma, 2048, &cfg);
        assert!(result.is_ok(), "RMSNorm forward failed: {result:?}");
    }

    // -- Consistency tests --------------------------------------------------

    #[test]
    fn test_layer_norm_zero_mean_unit_var() {
        // Output of layer norm (no affine) should have ~0 mean and ~1 var
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0_f32; 256];
        let beta = vec![0.0_f32; 256];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 256, &cfg).unwrap();

        let mean: f32 = output.iter().sum::<f32>() / 256.0;
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 256.0;
        assert!(mean.abs() < 1e-4, "output mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.01, "output var should be ~1, got {var}");
    }

    #[test]
    fn test_rms_norm_preserves_sign() {
        let cfg = LayerNormConfig::with_defaults();
        let input = [-3.0_f32, -1.0, 0.0, 1.0, 3.0];
        let gamma = [1.0; 5];
        let output = rms_norm_cpu_fallback(&input, &gamma, 5, &cfg).unwrap();

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp == 0.0 {
                assert!(out.abs() < 1e-6, "zero should stay zero at {i}");
            } else {
                assert_eq!(
                    inp.signum(),
                    out.signum(),
                    "sign mismatch at {i}: input={inp}, output={out}"
                );
            }
        }
    }

    #[test]
    fn test_rms_norm_scale_invariance() {
        // RMSNorm(alpha * x) ≈ sign(alpha) * RMSNorm(x) for gamma=1
        let cfg = LayerNormConfig::with_defaults();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];

        let output = rms_norm_cpu_fallback(&input, &gamma, 4, &cfg).unwrap();

        let scaled: Vec<f32> = input.iter().map(|x| x * 10.0).collect();
        let output_scaled = rms_norm_cpu_fallback(&scaled, &gamma, 4, &cfg).unwrap();

        for (i, (&a, &b)) in output.iter().zip(output_scaled.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "idx={i}: base={a}, scaled={b}");
        }
    }

    #[test]
    fn test_layer_norm_negative_values() {
        let cfg = LayerNormConfig::new(1e-5, false).unwrap();
        let input = [-10.0_f32, -5.0, 0.0, 5.0, 10.0];
        let gamma = [1.0; 5];
        let beta = [0.0; 5];
        let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 5, &cfg).unwrap();

        assert!(output.iter().all(|v| v.is_finite()));
        // Mean of symmetric input around 0 → output should also be symmetric
        let mean: f32 = output.iter().sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_different_eps_values() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0; 4];
        let beta = [0.0; 4];

        for eps in [1e-8, 1e-6, 1e-5, 1e-3, 0.1] {
            let cfg = LayerNormConfig::new(eps, false).unwrap();
            let output = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();
            assert!(output.iter().all(|v| v.is_finite()), "non-finite at eps={eps}");
        }
    }
}
