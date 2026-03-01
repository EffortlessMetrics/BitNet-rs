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

use bitnet_common::{KernelError, Result};

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
}
