//! Batch Normalization CUDA kernel.
//!
//! # Kernel strategy
//!
//! Batch normalization normalises activations across the batch dimension for
//! each feature channel independently:
//!
//!   `y[n, c] = ((x[n, c] - mean[c]) / sqrt(var[c] + eps)) * weight[c] + bias[c]`
//!
//! **Training mode** computes per-channel mean and variance from the current
//! mini-batch and updates exponential moving averages of running statistics:
//!
//! 1. Each thread-block handles one feature channel.
//! 2. A parallel reduction computes `mean[c]` and `var[c]` over the batch.
//! 3. Running statistics are updated:
//!    `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`
//! 4. Each element is normalised and optionally scaled/shifted by learnable
//!    `weight` (gamma) and `bias` (beta) parameters.
//!
//! **Eval mode** uses pre-computed running mean/variance, skipping the batch
//! reduction entirely.
//!
//! # CPU fallback
//!
//! [`batch_norm_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.
//!
//! # Standalone fallback functions
//!
//! [`batch_norm_cpu_fallback`] and [`batch_norm_inference_cpu_fallback`] provide
//! self-contained pure-Rust implementations that accept explicit parameter
//! slices (gamma, beta, running_mean, running_var) and return a newly allocated
//! `Vec<f32>`, making them convenient for one-shot usage without constructing
//! a [`BatchNormKernel`].

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source strings
// ---------------------------------------------------------------------------

/// CUDA kernel source for batch normalization forward pass (training mode).
///
/// One thread-block per feature channel. Performs a parallel reduction over the
/// batch dimension to compute per-channel mean and variance, updates running
/// statistics via EMA, then normalises and applies affine transform.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCH_NORM_TRAIN_KERNEL_SRC: &str = r#"
extern "C" __global__ void batch_norm_train_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int batch_size,
    int num_features,
    float eps,
    float momentum)
{
    int c = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float* s_sum  = sdata;
    float* s_sum2 = sdata + blockDim.x;

    float local_sum  = 0.0f;
    float local_sum2 = 0.0f;
    for (int b = tid; b < batch_size; b += blockDim.x) {
        float val = input[b * num_features + c];
        local_sum  += val;
        local_sum2 += val * val;
    }
    s_sum[tid]  = local_sum;
    s_sum2[tid] = local_sum2;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]  += s_sum[tid + stride];
            s_sum2[tid] += s_sum2[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)batch_size;
    float var  = s_sum2[0] / (float)batch_size - mean * mean;

    // Update running statistics (only lane 0 writes)
    if (tid == 0) {
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        // Bessel's correction for running variance
        float unbiased_var = (batch_size > 1)
            ? (s_sum2[0] - (float)batch_size * mean * mean) / (float)(batch_size - 1)
            : var;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    }

    float inv_std = rsqrtf(var + eps);
    float g = gamma[c];
    float b_val = beta[c];

    for (int n = tid; n < batch_size; n += blockDim.x) {
        int idx = n * num_features + c;
        output[idx] = (input[idx] - mean) * inv_std * g + b_val;
    }
}
"#;

/// CUDA kernel source for batch normalization forward pass (inference mode).
///
/// Uses pre-computed running mean and variance — no batch reduction needed.
/// Each thread processes multiple elements via grid-stride loop.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const BATCH_NORM_INFERENCE_KERNEL_SRC: &str = r#"
extern "C" __global__ void batch_norm_inference_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int batch_size,
    int num_features,
    float eps)
{
    int c = blockIdx.x;

    float mean    = running_mean[c];
    float inv_std = rsqrtf(running_var[c] + eps);
    float g       = gamma[c];
    float b_val   = beta[c];

    for (int n = threadIdx.x; n < batch_size; n += blockDim.x) {
        int idx = n * num_features + c;
        output[idx] = (input[idx] - mean) * inv_std * g + b_val;
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for batch normalization.
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    /// Number of feature channels (C).
    pub num_features: usize,
    /// Epsilon added inside the square root for numerical stability.
    pub eps: f32,
    /// Momentum for running mean/variance update (default `0.1`).
    pub momentum: f32,
    /// Whether learnable affine parameters (weight, bias) are applied.
    pub affine: bool,
    /// Whether to track running mean and variance across batches.
    pub track_running_stats: bool,
}

impl BatchNormConfig {
    /// Create a configuration for `num_features` channels with sensible defaults.
    ///
    /// Defaults: `eps = 1e-5`, `momentum = 0.1`, `affine = true`,
    /// `track_running_stats = true`.
    pub fn new(num_features: usize) -> Result<Self> {
        if num_features == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "batch norm num_features must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true })
    }

    /// Override the epsilon value.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Override the momentum value.
    ///
    /// # Errors
    ///
    /// Returns an error if `momentum` is not in `[0, 1]`.
    pub fn with_momentum(mut self, momentum: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&momentum) || !momentum.is_finite() {
            return Err(KernelError::InvalidArguments {
                reason: format!("batch norm momentum must be in [0, 1], got {momentum}"),
            }
            .into());
        }
        self.momentum = momentum;
        Ok(self)
    }

    /// Disable the learnable affine parameters (weight and bias).
    #[must_use]
    pub fn without_affine(mut self) -> Self {
        self.affine = false;
        self
    }

    /// Disable running statistics tracking.
    #[must_use]
    pub fn without_running_stats(mut self) -> Self {
        self.track_running_stats = false;
        self
    }

    /// Compute the CUDA grid dimensions `(num_features, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.num_features as u32, 1, 1)
    }

    /// Threads per block — one thread per sample in the batch.
    pub fn block_dim(&self, batch_size: usize) -> (u32, u32, u32) {
        ((batch_size as u32).min(1024), 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CUDA-specific launch configuration
// ---------------------------------------------------------------------------

/// Lightweight CUDA launch configuration for batch normalization.
///
/// This is a simplified config struct suitable for passing to CUDA kernel
/// launches. It carries only the parameters needed by the GPU kernels
/// (num_features, eps, momentum) without the higher-level policy flags
/// (`affine`, `track_running_stats`) that live in [`BatchNormConfig`].
#[derive(Debug, Clone, Copy)]
pub struct CudaBatchNormConfig {
    /// Number of feature channels (C).
    pub num_features: usize,
    /// Epsilon for numerical stability inside `rsqrtf(var + eps)`.
    pub eps: f32,
    /// Momentum for exponential moving average of running statistics.
    pub momentum: f32,
}

impl CudaBatchNormConfig {
    /// Create a new CUDA batch-norm config.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_features` is zero.
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Result<Self> {
        if num_features == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "CudaBatchNormConfig: num_features must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { num_features, eps, momentum })
    }

    /// Build from a full [`BatchNormConfig`].
    pub fn from_config(config: &BatchNormConfig) -> Self {
        Self { num_features: config.num_features, eps: config.eps, momentum: config.momentum }
    }

    /// Grid dimensions for the CUDA launch — one block per channel.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.num_features as u32, 1, 1)
    }

    /// Block dimensions — one thread per sample, capped at 1024.
    pub fn block_dim(&self, batch_size: usize) -> (u32, u32, u32) {
        ((batch_size as u32).min(1024), 1, 1)
    }

    /// Shared memory bytes needed by the training kernel (two float arrays of
    /// `block_dim.0` elements each).
    pub fn shared_mem_bytes(&self, batch_size: usize) -> u32 {
        let threads = (batch_size as u32).min(1024);
        threads * 2 * 4 // 2 arrays × sizeof(f32)
    }
}

// ---------------------------------------------------------------------------
// Running state
// ---------------------------------------------------------------------------

/// Mutable running state for batch normalization.
///
/// Tracks running mean and variance across batches (for eval mode),
/// plus the optional learnable affine parameters.
#[derive(Debug, Clone)]
pub struct BatchNormState {
    /// Running mean per feature channel `[num_features]`.
    pub running_mean: Vec<f32>,
    /// Running variance per feature channel `[num_features]`.
    pub running_var: Vec<f32>,
    /// Learnable scale (gamma) per feature `[num_features]`.
    /// Initialised to `1.0` when `affine = true`.
    pub weight: Vec<f32>,
    /// Learnable shift (beta) per feature `[num_features]`.
    /// Initialised to `0.0` when `affine = true`.
    pub bias: Vec<f32>,
    /// Number of batches observed (for Bessel's correction).
    pub num_batches_tracked: u64,
}

impl BatchNormState {
    /// Create a new state initialised for the given config.
    ///
    /// * `running_mean` and `running_var` start at 0 and 1 respectively.
    /// * `weight` starts at 1 and `bias` at 0 (identity transform).
    pub fn new(config: &BatchNormConfig) -> Self {
        let c = config.num_features;
        Self {
            running_mean: vec![0.0; c],
            running_var: vec![1.0; c],
            weight: if config.affine { vec![1.0; c] } else { vec![] },
            bias: if config.affine { vec![0.0; c] } else { vec![] },
            num_batches_tracked: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Batch norm kernel handle
// ---------------------------------------------------------------------------

/// Batch normalization kernel with CPU fallback and GPU stub.
#[derive(Debug, Clone)]
pub struct BatchNormKernel {
    /// Kernel configuration.
    pub config: BatchNormConfig,
    /// Mutable running state (statistics + affine parameters).
    pub state: BatchNormState,
}

impl BatchNormKernel {
    /// Create a new batch-norm kernel for `num_features` channels.
    pub fn new(num_features: usize) -> Result<Self> {
        let config = BatchNormConfig::new(num_features)?;
        let state = BatchNormState::new(&config);
        Ok(Self { config, state })
    }

    /// Create a kernel with a custom configuration.
    pub fn with_config(config: BatchNormConfig) -> Self {
        let state = BatchNormState::new(&config);
        Self { config, state }
    }

    /// Run the forward pass with automatic dispatch.
    ///
    /// * `input`  — `[batch_size, num_features]` (row-major FP32)
    /// * `output` — `[batch_size, num_features]` (row-major FP32, written)
    /// * `training` — use batch statistics (true) or running stats (false)
    pub fn forward(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        training: bool,
    ) -> Result<()> {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if crate::device_features::gpu_available_runtime()
                && let Ok(()) = launch_batch_norm(
                    input,
                    output,
                    &mut self.state,
                    &self.config,
                    batch_size,
                    training,
                )
            {
                return Ok(());
            }
        }
        batch_norm_cpu(input, output, &mut self.state, &self.config, batch_size, training)
    }
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// Pure-Rust batch normalization forward pass.
///
/// # Arguments
///
/// * `input`      — `[batch_size, num_features]` (FP32, row-major)
/// * `output`     — `[batch_size, num_features]` (FP32, row-major, written)
/// * `state`      — Mutable running statistics and affine parameters
/// * `config`     — Kernel configuration
/// * `batch_size` — Number of samples in the batch (N)
/// * `training`   — If true, compute batch statistics and update running stats
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths do not match
/// `batch_size × num_features`.
pub fn batch_norm_cpu(
    input: &[f32],
    output: &mut [f32],
    state: &mut BatchNormState,
    config: &BatchNormConfig,
    batch_size: usize,
    training: bool,
) -> Result<()> {
    let c = config.num_features;
    let total = batch_size * c;

    if batch_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "batch_size must be non-zero".into() }.into()
        );
    }
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batch_norm input length {} < expected {total}", input.len()),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batch_norm output length {} < expected {total}", output.len()),
        }
        .into());
    }

    let n = batch_size as f32;

    for ch in 0..c {
        // --- Determine mean and variance for this channel ---
        let (mean, var) = if training {
            // Compute batch mean
            let mut sum = 0.0_f32;
            for b in 0..batch_size {
                sum += input[b * c + ch];
            }
            let batch_mean = sum / n;

            // Compute batch variance (biased, matching PyTorch convention)
            let mut var_sum = 0.0_f32;
            for b in 0..batch_size {
                let diff = input[b * c + ch] - batch_mean;
                var_sum += diff * diff;
            }
            let batch_var = var_sum / n;

            // Update running statistics with exponential moving average
            if config.track_running_stats {
                let m = config.momentum;
                state.running_mean[ch] = (1.0 - m) * state.running_mean[ch] + m * batch_mean;
                // Use unbiased variance (Bessel's correction) for running var
                let unbiased_var = if batch_size > 1 { var_sum / (n - 1.0) } else { batch_var };
                state.running_var[ch] = (1.0 - m) * state.running_var[ch] + m * unbiased_var;
            }

            (batch_mean, batch_var)
        } else {
            (state.running_mean[ch], state.running_var[ch])
        };

        // --- Normalise, scale, shift ---
        let inv_std = 1.0 / (var + config.eps).sqrt();

        let (gamma, beta) =
            if config.affine { (state.weight[ch], state.bias[ch]) } else { (1.0, 0.0) };

        for b in 0..batch_size {
            let idx = b * c + ch;
            output[idx] = (input[idx] - mean) * inv_std * gamma + beta;
        }
    }

    if training && config.track_running_stats {
        state.num_batches_tracked += 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Standalone CPU fallback functions
// ---------------------------------------------------------------------------

/// Standalone CPU batch normalization forward pass (training mode).
///
/// Accepts explicit parameter slices and returns a newly allocated output
/// vector. This is a convenience wrapper that does not require constructing
/// a [`BatchNormKernel`] or [`BatchNormState`].
///
/// * `input`        — `[batch_size, num_features]` (FP32, row-major)
/// * `gamma`        — Per-channel scale `[num_features]`
/// * `beta`         — Per-channel shift `[num_features]`
/// * `running_mean` — Running mean `[num_features]` (updated in-place via EMA)
/// * `running_var`  — Running variance `[num_features]` (updated in-place via EMA)
/// * `config`       — CUDA-style launch config carrying `num_features`, `eps`, `momentum`
///
/// Returns `Vec<f32>` of length `input.len()`.
pub fn batch_norm_cpu_fallback(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &mut [f32],
    running_var: &mut [f32],
    config: &CudaBatchNormConfig,
) -> Result<Vec<f32>> {
    let c = config.num_features;
    if c == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "batch_norm_cpu_fallback: num_features must be non-zero".into(),
        }
        .into());
    }
    if input.len() % c != 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_norm_cpu_fallback: input length {} not divisible by num_features {c}",
                input.len()
            ),
        }
        .into());
    }
    let batch_size = input.len() / c;
    if batch_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "batch_norm_cpu_fallback: batch_size must be non-zero".into(),
        }
        .into());
    }
    if gamma.len() < c || beta.len() < c || running_mean.len() < c || running_var.len() < c {
        return Err(KernelError::InvalidArguments {
            reason: format!("batch_norm_cpu_fallback: parameter slices must have length >= {c}"),
        }
        .into());
    }

    let n = batch_size as f32;
    let mut output = vec![0.0_f32; input.len()];

    for ch in 0..c {
        // Compute batch mean
        let mut sum = 0.0_f32;
        for b in 0..batch_size {
            sum += input[b * c + ch];
        }
        let batch_mean = sum / n;

        // Compute batch variance (biased)
        let mut var_sum = 0.0_f32;
        for b in 0..batch_size {
            let diff = input[b * c + ch] - batch_mean;
            var_sum += diff * diff;
        }
        let batch_var = var_sum / n;

        // Update running statistics
        let m = config.momentum;
        running_mean[ch] = (1.0 - m) * running_mean[ch] + m * batch_mean;
        let unbiased_var = if batch_size > 1 { var_sum / (n - 1.0) } else { batch_var };
        running_var[ch] = (1.0 - m) * running_var[ch] + m * unbiased_var;

        // Normalise + affine
        let inv_std = 1.0 / (batch_var + config.eps).sqrt();
        for b in 0..batch_size {
            let idx = b * c + ch;
            output[idx] = (input[idx] - batch_mean) * inv_std * gamma[ch] + beta[ch];
        }
    }

    Ok(output)
}

/// Standalone CPU batch normalization forward pass (inference mode).
///
/// Uses pre-computed running mean and variance — no batch statistics are
/// computed and no state is mutated.
///
/// * `input`        — `[batch_size, num_features]` (FP32, row-major)
/// * `gamma`        — Per-channel scale `[num_features]`
/// * `beta`         — Per-channel shift `[num_features]`
/// * `running_mean` — Pre-computed running mean `[num_features]`
/// * `running_var`  — Pre-computed running variance `[num_features]`
/// * `eps`          — Epsilon for numerical stability
///
/// Returns `Vec<f32>` of length `input.len()`.
pub fn batch_norm_inference_cpu_fallback(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if gamma.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "batch_norm_inference_cpu_fallback: gamma must be non-empty".into(),
        }
        .into());
    }
    let c = gamma.len();
    if input.len() % c != 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_norm_inference_cpu_fallback: input length {} not divisible by \
                 num_features {c}",
                input.len()
            ),
        }
        .into());
    }
    let batch_size = input.len() / c;
    if batch_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "batch_norm_inference_cpu_fallback: batch_size must be non-zero".into(),
        }
        .into());
    }
    if beta.len() < c || running_mean.len() < c || running_var.len() < c {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_norm_inference_cpu_fallback: parameter slices must have length >= {c}"
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; input.len()];

    for ch in 0..c {
        let mean = running_mean[ch];
        let inv_std = 1.0 / (running_var[ch] + eps).sqrt();
        let g = gamma[ch];
        let b = beta[ch];

        for batch in 0..batch_size {
            let idx = batch * c + ch;
            output[idx] = (input[idx] - mean) * inv_std * g + b;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the batch normalization CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_batch_norm(
    _input: &[f32],
    _output: &mut [f32],
    _state: &mut BatchNormState,
    config: &BatchNormConfig,
    batch_size: usize,
    training: bool,
) -> Result<()> {
    log::debug!(
        "batch_norm stub: num_features={}, batch_size={batch_size}, training={training}, \
         grid={:?}",
        config.num_features,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "batch normalization CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = BatchNormConfig::new(64).unwrap();
        assert_eq!(cfg.num_features, 64);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!((cfg.momentum - 0.1).abs() < 1e-10);
        assert!(cfg.affine);
        assert!(cfg.track_running_stats);
    }

    #[test]
    fn test_config_rejects_zero_features() {
        assert!(BatchNormConfig::new(0).is_err());
    }

    #[test]
    fn test_config_with_eps() {
        let cfg = BatchNormConfig::new(32).unwrap().with_eps(1e-8);
        assert!((cfg.eps - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_config_with_momentum() {
        let cfg = BatchNormConfig::new(32).unwrap().with_momentum(0.01).unwrap();
        assert!((cfg.momentum - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_config_rejects_bad_momentum() {
        let cfg = BatchNormConfig::new(32).unwrap();
        assert!(cfg.clone().with_momentum(-0.1).is_err());
        assert!(cfg.clone().with_momentum(1.1).is_err());
        assert!(cfg.clone().with_momentum(f32::NAN).is_err());
        assert!(cfg.with_momentum(f32::INFINITY).is_err());
    }

    #[test]
    fn test_config_without_affine() {
        let cfg = BatchNormConfig::new(16).unwrap().without_affine();
        assert!(!cfg.affine);
    }

    #[test]
    fn test_config_without_running_stats() {
        let cfg = BatchNormConfig::new(16).unwrap().without_running_stats();
        assert!(!cfg.track_running_stats);
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = BatchNormConfig::new(128).unwrap();
        assert_eq!(cfg.grid_dim(), (128, 1, 1));
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = BatchNormConfig::new(64).unwrap();
        assert_eq!(cfg.block_dim(32), (32, 1, 1));
        assert_eq!(cfg.block_dim(2048), (1024, 1, 1)); // capped
    }

    // -- State tests --------------------------------------------------------

    #[test]
    fn test_state_initial_values() {
        let cfg = BatchNormConfig::new(3).unwrap();
        let state = BatchNormState::new(&cfg);
        assert_eq!(state.running_mean, vec![0.0; 3]);
        assert_eq!(state.running_var, vec![1.0; 3]);
        assert_eq!(state.weight, vec![1.0; 3]);
        assert_eq!(state.bias, vec![0.0; 3]);
        assert_eq!(state.num_batches_tracked, 0);
    }

    #[test]
    fn test_state_no_affine() {
        let cfg = BatchNormConfig::new(4).unwrap().without_affine();
        let state = BatchNormState::new(&cfg);
        assert!(state.weight.is_empty());
        assert!(state.bias.is_empty());
    }

    // -- CPU forward pass (training mode) -----------------------------------

    #[test]
    fn test_training_forward_identity() {
        // Input already normalised → output ≈ input when weight=1, bias=0
        let mut kernel = BatchNormKernel::new(2).unwrap();
        // Input: batch=4, features=2, each channel has mean≈0, var≈1
        let input = [
            -1.0, -1.0, //
            -0.33, -0.33, //
            0.33, 0.33, //
            1.0, 1.0,
        ];
        let mut output = [0.0_f32; 8];
        kernel.forward(&input, &mut output, 4, true).unwrap();

        // Output should be roughly centred at 0 with unit variance
        for ch in 0..2 {
            let mut sum = 0.0_f32;
            for b in 0..4 {
                sum += output[b * 2 + ch];
            }
            let mean = sum / 4.0;
            assert!(mean.abs() < 1e-5, "ch {ch}: mean={mean}");
        }
    }

    #[test]
    fn test_training_forward_normalises() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        // batch=4, features=1, values: [2, 4, 6, 8]
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut output = [0.0_f32; 4];
        kernel.forward(&input, &mut output, 4, true).unwrap();

        // Mean of output should be 0
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");

        // Variance of output should be ≈ 1 (with eps correction)
        let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "var={var}");
    }

    #[test]
    fn test_training_forward_multi_channel() {
        let mut kernel = BatchNormKernel::new(3).unwrap();
        // batch=2, features=3
        let input = [
            1.0, 100.0, -5.0, //
            3.0, 200.0, 5.0,
        ];
        let mut output = [0.0_f32; 6];
        kernel.forward(&input, &mut output, 2, true).unwrap();

        // Each channel should be normalised independently
        for ch in 0..3 {
            let vals: Vec<f32> = (0..2).map(|b| output[b * 3 + ch]).collect();
            let mean: f32 = vals.iter().sum::<f32>() / 2.0;
            assert!(mean.abs() < 1e-4, "ch {ch}: mean={mean}");
        }
    }

    // -- Running statistics updates -----------------------------------------

    #[test]
    fn test_running_stats_update() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        let input = [2.0, 4.0, 6.0, 8.0]; // mean=5, var_biased=5
        let mut output = [0.0_f32; 4];

        kernel.forward(&input, &mut output, 4, true).unwrap();

        // running_mean = (1-0.1)*0 + 0.1*5 = 0.5
        assert!(
            (kernel.state.running_mean[0] - 0.5).abs() < 1e-5,
            "running_mean={}",
            kernel.state.running_mean[0]
        );
        // running_var = (1-0.1)*1 + 0.1 * unbiased_var
        // unbiased_var = (20 / 3) ≈ 6.6667
        let expected_var = 0.9 * 1.0 + 0.1 * (20.0 / 3.0);
        assert!(
            (kernel.state.running_var[0] - expected_var).abs() < 1e-4,
            "running_var={}, expected={expected_var}",
            kernel.state.running_var[0]
        );
        assert_eq!(kernel.state.num_batches_tracked, 1);
    }

    #[test]
    fn test_running_stats_accumulate_over_batches() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        let mut output = [0.0_f32; 2];

        let batch1 = [1.0, 3.0];
        kernel.forward(&batch1, &mut output, 2, true).unwrap();
        assert_eq!(kernel.state.num_batches_tracked, 1);

        let batch2 = [5.0, 7.0];
        kernel.forward(&batch2, &mut output, 2, true).unwrap();
        assert_eq!(kernel.state.num_batches_tracked, 2);

        // Running mean should reflect both batches
        assert!(kernel.state.running_mean[0].abs() > 0.0);
    }

    #[test]
    fn test_no_running_stats_tracking() {
        let config = BatchNormConfig::new(1).unwrap().without_running_stats();
        let mut kernel = BatchNormKernel::with_config(config);
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut output = [0.0_f32; 4];
        kernel.forward(&input, &mut output, 4, true).unwrap();

        // Running stats should remain at initial values
        assert!((kernel.state.running_mean[0]).abs() < 1e-10);
        assert!((kernel.state.running_var[0] - 1.0).abs() < 1e-10);
        assert_eq!(kernel.state.num_batches_tracked, 0);
    }

    // -- Eval mode ----------------------------------------------------------

    #[test]
    fn test_eval_uses_running_stats() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        // Set known running stats
        kernel.state.running_mean[0] = 5.0;
        kernel.state.running_var[0] = 4.0;

        let input = [5.0, 7.0, 3.0, 9.0];
        let mut output = [0.0_f32; 4];
        kernel.forward(&input, &mut output, 4, false).unwrap();

        // (5 - 5) / sqrt(4 + 1e-5) ≈ 0
        assert!(output[0].abs() < 1e-4, "output[0]={}", output[0]);
        // (7 - 5) / sqrt(4 + 1e-5) ≈ 1
        assert!((output[1] - 1.0).abs() < 1e-3, "output[1]={}", output[1]);
    }

    #[test]
    fn test_eval_does_not_update_stats() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        kernel.state.running_mean[0] = 2.0;
        kernel.state.running_var[0] = 3.0;
        let orig_mean = kernel.state.running_mean[0];
        let orig_var = kernel.state.running_var[0];

        let input = [10.0, 20.0];
        let mut output = [0.0_f32; 2];
        kernel.forward(&input, &mut output, 2, false).unwrap();

        assert!((kernel.state.running_mean[0] - orig_mean).abs() < 1e-10);
        assert!((kernel.state.running_var[0] - orig_var).abs() < 1e-10);
    }

    // -- Affine transform ---------------------------------------------------

    #[test]
    fn test_affine_scale_and_shift() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        kernel.state.weight[0] = 2.0;
        kernel.state.bias[0] = 3.0;
        kernel.state.running_mean[0] = 0.0;
        kernel.state.running_var[0] = 1.0;

        let input = [0.0, 1.0];
        let mut output = [0.0_f32; 2];
        kernel.forward(&input, &mut output, 2, false).unwrap();

        // y = (x - 0) / sqrt(1 + 1e-5) * 2 + 3 ≈ x * 2 + 3
        assert!((output[0] - 3.0).abs() < 1e-3, "output[0]={}", output[0]);
        assert!((output[1] - 5.0).abs() < 1e-3, "output[1]={}", output[1]);
    }

    #[test]
    fn test_no_affine() {
        let config = BatchNormConfig::new(1).unwrap().without_affine();
        let mut kernel = BatchNormKernel::with_config(config);
        kernel.state.running_mean[0] = 0.0;
        kernel.state.running_var[0] = 1.0;

        let input = [0.0, 1.0];
        let mut output = [0.0_f32; 2];
        kernel.forward(&input, &mut output, 2, false).unwrap();

        // y = (x - 0) / sqrt(1 + 1e-5) * 1 + 0 ≈ x
        assert!(output[0].abs() < 1e-3, "output[0]={}", output[0]);
        assert!((output[1] - 1.0).abs() < 1e-3, "output[1]={}", output[1]);
    }

    // -- Numerical stability ------------------------------------------------

    #[test]
    fn test_numerical_stability_small_eps() {
        let config = BatchNormConfig::new(1).unwrap().with_eps(1e-12);
        let mut kernel = BatchNormKernel::with_config(config);
        // Constant input → zero variance → relies on eps
        let input = [5.0, 5.0, 5.0, 5.0];
        let mut output = [0.0_f32; 4];
        kernel.forward(&input, &mut output, 4, true).unwrap();

        assert!(output.iter().all(|v| v.is_finite()), "non-finite output: {output:?}");
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        let input = [1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
        let mut output = [0.0_f32; 4];
        kernel.forward(&input, &mut output, 4, true).unwrap();

        assert!(output.iter().all(|v| v.is_finite()), "non-finite output: {output:?}");
        let mean: f32 = output.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-2, "mean={mean}");
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_single_sample_batch() {
        let mut kernel = BatchNormKernel::new(2).unwrap();
        let input = [3.0, 7.0]; // batch=1
        let mut output = [0.0_f32; 2];
        kernel.forward(&input, &mut output, 1, true).unwrap();

        // Zero variance case: (x - mean) / sqrt(0 + eps) → ≈ 0
        assert!(output.iter().all(|v| v.is_finite()), "non-finite: {output:?}");
    }

    #[test]
    fn test_single_feature() {
        let mut kernel = BatchNormKernel::new(1).unwrap();
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0_f32; 3];
        kernel.forward(&input, &mut output, 3, true).unwrap();

        let mean: f32 = output.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn test_rejects_zero_batch_size() {
        let mut kernel = BatchNormKernel::new(4).unwrap();
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        assert!(kernel.forward(&input, &mut output, 0, true).is_err());
    }

    #[test]
    fn test_rejects_short_input() {
        let mut kernel = BatchNormKernel::new(4).unwrap();
        let input = [1.0_f32; 4]; // need 8 for batch=2
        let mut output = [0.0_f32; 8];
        assert!(kernel.forward(&input, &mut output, 2, true).is_err());
    }

    #[test]
    fn test_rejects_short_output() {
        let mut kernel = BatchNormKernel::new(4).unwrap();
        let input = [1.0_f32; 8];
        let mut output = [0.0_f32; 4]; // need 8
        assert!(kernel.forward(&input, &mut output, 2, true).is_err());
    }

    // -- Dispatch tests -----------------------------------------------------

    #[test]
    fn test_forward_dispatches_cpu() {
        let mut kernel = BatchNormKernel::new(2).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        let result = kernel.forward(&input, &mut output, 2, true);
        assert!(result.is_ok(), "CPU dispatch should succeed: {result:?}");
    }

    #[test]
    fn test_training_then_eval_consistency() {
        let mut kernel = BatchNormKernel::new(2).unwrap();

        // Train on several batches to build up running stats
        let mut output = [0.0_f32; 4];
        for _ in 0..10 {
            let input = [1.0, 10.0, 3.0, 30.0];
            kernel.forward(&input, &mut output, 2, true).unwrap();
        }

        // Eval with similar data should produce reasonable output
        let eval_input = [2.0, 20.0, 2.0, 20.0];
        let mut eval_output = [0.0_f32; 4];
        kernel.forward(&eval_input, &mut eval_output, 2, false).unwrap();

        assert!(
            eval_output.iter().all(|v| v.is_finite()),
            "non-finite eval output: {eval_output:?}"
        );
    }

    // -- GPU stub tests -----------------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_batch_norm_launch() {
        let config = BatchNormConfig::new(64).unwrap();
        let mut state = BatchNormState::new(&config);
        let input = vec![1.0_f32; 32 * 64];
        let mut output = vec![0.0_f32; 32 * 64];
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            let result = launch_batch_norm(&input, &mut output, &mut state, &config, 32, true);
            assert!(result.is_ok(), "CUDA batch_norm launch failed: {result:?}");
        }
    }

    // -- CudaBatchNormConfig tests ------------------------------------------

    #[test]
    fn test_cuda_config_new() {
        let cfg = CudaBatchNormConfig::new(64, 1e-5, 0.1).unwrap();
        assert_eq!(cfg.num_features, 64);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!((cfg.momentum - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_cuda_config_rejects_zero_features() {
        assert!(CudaBatchNormConfig::new(0, 1e-5, 0.1).is_err());
    }

    #[test]
    fn test_cuda_config_from_batch_norm_config() {
        let cfg = BatchNormConfig::new(128).unwrap().with_eps(1e-6);
        let cuda_cfg = CudaBatchNormConfig::from_config(&cfg);
        assert_eq!(cuda_cfg.num_features, 128);
        assert!((cuda_cfg.eps - 1e-6).abs() < 1e-12);
        assert!((cuda_cfg.momentum - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_cuda_config_grid_and_block_dim() {
        let cfg = CudaBatchNormConfig::new(256, 1e-5, 0.1).unwrap();
        assert_eq!(cfg.grid_dim(), (256, 1, 1));
        assert_eq!(cfg.block_dim(64), (64, 1, 1));
        assert_eq!(cfg.block_dim(2048), (1024, 1, 1)); // capped
    }

    #[test]
    fn test_cuda_config_shared_mem_bytes() {
        let cfg = CudaBatchNormConfig::new(8, 1e-5, 0.1).unwrap();
        // batch_size=32 → threads=32 → 32*2*4 = 256 bytes
        assert_eq!(cfg.shared_mem_bytes(32), 256);
        // batch_size=2048 → threads capped at 1024 → 1024*2*4 = 8192
        assert_eq!(cfg.shared_mem_bytes(2048), 8192);
    }

    // -- batch_norm_cpu_fallback tests --------------------------------------

    #[test]
    fn test_cpu_fallback_training_normalises() {
        let cfg = CudaBatchNormConfig::new(1, 1e-5, 0.1).unwrap();
        let input = [2.0_f32, 4.0, 6.0, 8.0]; // batch=4, features=1
        let gamma = [1.0];
        let beta = [0.0];
        let mut rmean = [0.0_f32];
        let mut rvar = [1.0_f32];

        let out =
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).unwrap();

        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn test_cpu_fallback_training_multi_channel() {
        let cfg = CudaBatchNormConfig::new(3, 1e-5, 0.1).unwrap();
        let input = [1.0, 100.0, -5.0, 3.0, 200.0, 5.0]; // batch=2, features=3
        let gamma = [1.0, 1.0, 1.0];
        let beta = [0.0, 0.0, 0.0];
        let mut rmean = [0.0_f32; 3];
        let mut rvar = [1.0_f32; 3];

        let out =
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).unwrap();
        assert_eq!(out.len(), 6);

        for ch in 0..3 {
            let ch_mean = (out[ch] + out[3 + ch]) / 2.0;
            assert!(ch_mean.abs() < 1e-4, "ch {ch}: mean={ch_mean}");
        }
    }

    #[test]
    fn test_cpu_fallback_updates_running_stats() {
        let cfg = CudaBatchNormConfig::new(1, 1e-5, 0.1).unwrap();
        let input = [2.0_f32, 4.0, 6.0, 8.0]; // mean=5
        let gamma = [1.0];
        let beta = [0.0];
        let mut rmean = [0.0_f32];
        let mut rvar = [1.0_f32];

        let _ =
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).unwrap();

        // running_mean = 0.9*0 + 0.1*5 = 0.5
        assert!((rmean[0] - 0.5).abs() < 1e-5, "rmean={}", rmean[0]);
        assert!(rvar[0] > 0.0, "rvar should be positive");
    }

    #[test]
    fn test_cpu_fallback_with_affine() {
        let cfg = CudaBatchNormConfig::new(1, 1e-5, 0.1).unwrap();
        let input = [0.0_f32, 2.0]; // batch=2
        let gamma = [3.0];
        let beta = [1.0];
        let mut rmean = [0.0_f32];
        let mut rvar = [1.0_f32];

        let out =
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).unwrap();
        // mean=1, var=1, inv_std≈1 → out[0] = (0-1)*1*3+1 = -2, out[1] = (2-1)*1*3+1 = 4
        assert!((out[0] - (-2.0)).abs() < 0.1, "out[0]={}", out[0]);
        assert!((out[1] - 4.0).abs() < 0.1, "out[1]={}", out[1]);
    }

    #[test]
    fn test_cpu_fallback_rejects_empty_input() {
        let cfg = CudaBatchNormConfig::new(2, 1e-5, 0.1).unwrap();
        let input: [f32; 0] = [];
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];
        let mut rmean = [0.0_f32; 2];
        let mut rvar = [1.0_f32; 2];
        assert!(
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).is_err()
        );
    }

    #[test]
    fn test_cpu_fallback_rejects_misaligned_input() {
        let cfg = CudaBatchNormConfig::new(3, 1e-5, 0.1).unwrap();
        let input = [1.0_f32; 5]; // 5 not divisible by 3
        let gamma = [1.0; 3];
        let beta = [0.0; 3];
        let mut rmean = [0.0_f32; 3];
        let mut rvar = [1.0_f32; 3];
        assert!(
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).is_err()
        );
    }

    #[test]
    fn test_cpu_fallback_rejects_short_params() {
        let cfg = CudaBatchNormConfig::new(4, 1e-5, 0.1).unwrap();
        let input = [1.0_f32; 8]; // batch=2, features=4
        let gamma = [1.0; 2]; // too short
        let beta = [0.0; 4];
        let mut rmean = [0.0_f32; 4];
        let mut rvar = [1.0_f32; 4];
        assert!(
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).is_err()
        );
    }

    // -- batch_norm_inference_cpu_fallback tests -----------------------------

    #[test]
    fn test_inference_fallback_basic() {
        let input = [5.0_f32, 7.0, 3.0, 9.0]; // batch=4, features=1
        let gamma = [1.0];
        let beta = [0.0];
        let rmean = [5.0_f32];
        let rvar = [4.0_f32];

        let out =
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).unwrap();

        // (5-5)/sqrt(4+eps) ≈ 0
        assert!(out[0].abs() < 1e-4, "out[0]={}", out[0]);
        // (7-5)/sqrt(4+eps) ≈ 1
        assert!((out[1] - 1.0).abs() < 1e-3, "out[1]={}", out[1]);
    }

    #[test]
    fn test_inference_fallback_multi_channel() {
        // batch=2, features=2
        let input = [10.0_f32, 20.0, 10.0, 20.0];
        let gamma = [1.0, 2.0];
        let beta = [0.0, 1.0];
        let rmean = [10.0, 20.0];
        let rvar = [1.0, 4.0];

        let out =
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).unwrap();
        assert_eq!(out.len(), 4);
        // ch0: (10-10)/sqrt(1+eps)*1+0 ≈ 0
        assert!(out[0].abs() < 1e-3, "out[0]={}", out[0]);
        // ch1: (20-20)/sqrt(4+eps)*2+1 ≈ 1
        assert!((out[1] - 1.0).abs() < 1e-3, "out[1]={}", out[1]);
    }

    #[test]
    fn test_inference_fallback_with_affine() {
        let input = [0.0_f32, 1.0]; // batch=2, features=1
        let gamma = [2.0];
        let beta = [3.0];
        let rmean = [0.0];
        let rvar = [1.0];

        let out =
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).unwrap();
        // y = (x - 0)/sqrt(1+eps)*2+3 ≈ x*2+3
        assert!((out[0] - 3.0).abs() < 1e-3, "out[0]={}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-3, "out[1]={}", out[1]);
    }

    #[test]
    fn test_inference_fallback_numerical_stability() {
        let input = [5.0_f32; 4]; // constant → zero-var case uses running_var
        let gamma = [1.0];
        let beta = [0.0];
        let rmean = [5.0];
        let rvar = [1e-12]; // very small running variance

        let out =
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite: {out:?}");
    }

    #[test]
    fn test_inference_fallback_rejects_empty_gamma() {
        let input = [1.0_f32; 4];
        let gamma: [f32; 0] = [];
        let beta: [f32; 0] = [];
        let rmean: [f32; 0] = [];
        let rvar: [f32; 0] = [];
        assert!(
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).is_err()
        );
    }

    #[test]
    fn test_inference_fallback_rejects_misaligned() {
        let input = [1.0_f32; 5]; // 5 not divisible by 2
        let gamma = [1.0, 1.0];
        let beta = [0.0, 0.0];
        let rmean = [0.0, 0.0];
        let rvar = [1.0, 1.0];
        assert!(
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).is_err()
        );
    }

    // -- Cross-check: standalone fallback vs BatchNormKernel -----------------

    #[test]
    fn test_fallback_matches_kernel_training() {
        let c = 3;
        let batch = 4;
        let input: Vec<f32> = (0..batch * c).map(|i| i as f32 * 0.5 - 3.0).collect();

        // Use BatchNormKernel
        let mut kernel = BatchNormKernel::new(c).unwrap();
        let mut kernel_output = vec![0.0_f32; batch * c];
        kernel.forward(&input, &mut kernel_output, batch, true).unwrap();

        // Use standalone fallback
        let cfg = CudaBatchNormConfig::from_config(&kernel.config);
        let gamma = vec![1.0_f32; c];
        let beta = vec![0.0_f32; c];
        let mut rmean = vec![0.0_f32; c];
        let mut rvar = vec![1.0_f32; c];
        let fallback_output =
            batch_norm_cpu_fallback(&input, &gamma, &beta, &mut rmean, &mut rvar, &cfg).unwrap();

        for i in 0..kernel_output.len() {
            assert!(
                (kernel_output[i] - fallback_output[i]).abs() < 1e-5,
                "mismatch at {i}: kernel={}, fallback={}",
                kernel_output[i],
                fallback_output[i]
            );
        }
    }

    #[test]
    fn test_inference_fallback_matches_kernel_eval() {
        let c = 2;
        let batch = 3;
        let input: Vec<f32> = (0..batch * c).map(|i| i as f32 + 1.0).collect();

        // Use BatchNormKernel in eval mode with known stats
        let mut kernel = BatchNormKernel::new(c).unwrap();
        kernel.state.running_mean = vec![3.0, 4.0];
        kernel.state.running_var = vec![2.0, 5.0];
        let mut kernel_output = vec![0.0_f32; batch * c];
        kernel.forward(&input, &mut kernel_output, batch, false).unwrap();

        // Use standalone inference fallback
        let gamma = vec![1.0_f32; c];
        let beta = vec![0.0_f32; c];
        let rmean = vec![3.0_f32, 4.0];
        let rvar = vec![2.0_f32, 5.0];
        let fallback_output =
            batch_norm_inference_cpu_fallback(&input, &gamma, &beta, &rmean, &rvar, 1e-5).unwrap();

        for i in 0..kernel_output.len() {
            assert!(
                (kernel_output[i] - fallback_output[i]).abs() < 1e-5,
                "mismatch at {i}: kernel={}, fallback={}",
                kernel_output[i],
                fallback_output[i]
            );
        }
    }

    // -- CUDA kernel source string tests ------------------------------------

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_cuda_kernel_source_strings_non_empty() {
        assert!(!BATCH_NORM_TRAIN_KERNEL_SRC.is_empty());
        assert!(!BATCH_NORM_INFERENCE_KERNEL_SRC.is_empty());
        assert!(BATCH_NORM_TRAIN_KERNEL_SRC.contains("batch_norm_train_f32"));
        assert!(BATCH_NORM_INFERENCE_KERNEL_SRC.contains("batch_norm_inference_f32"));
    }
}
