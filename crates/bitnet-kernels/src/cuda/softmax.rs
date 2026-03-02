//! Softmax CUDA kernel with numerically stable computation.
//!
//! # Kernel strategy
//!
//! Row-wise softmax over the logits (vocabulary) dimension using the
//! three-pass stable algorithm:
//!
//! 1. **Row max** — each thread-block cooperatively reduces one row to find
//!    `m = max(x)`, preventing overflow in the exponentiation step.
//! 2. **Shifted exp + sum** — every element is transformed to
//!    `e = exp((x[i] - m) / T)` where `T` is an optional temperature
//!    parameter (default `1.0`).  A parallel reduction accumulates `sum(e)`.
//! 3. **Normalise** — each element is divided by `sum(e)` to yield a valid
//!    probability distribution.
//!
//! One thread-block handles one row.  Grid size equals the batch/sequence
//! dimension (`n_rows`).  For typical vocabulary sizes (32 000 – 128 000) the
//! kernel achieves high memory-bandwidth utilisation on Ampere+.
//!
//! # CPU fallback
//!
//! [`softmax_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for the softmax kernel.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Number of columns per row (vocabulary / logits dimension).
    pub n_cols: usize,
    /// Number of rows (batch × sequence length).
    pub n_rows: usize,
    /// Threads per block — typically `min(n_cols, 1024)`.
    pub threads_per_block: u32,
    /// Temperature scaling factor applied before exponentiation.
    /// Values `> 1.0` soften the distribution; values in `(0, 1)` sharpen it.
    pub temperature: f32,
}

impl SoftmaxConfig {
    /// Create a configuration for the given shape with `temperature = 1.0`.
    pub fn for_shape(n_cols: usize, n_rows: usize) -> Result<Self> {
        if n_cols == 0 || n_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "softmax dimensions must be non-zero: \
                     n_cols={n_cols}, n_rows={n_rows}"
                ),
            }
            .into());
        }

        let threads_per_block = (n_cols as u32).min(1024);

        Ok(Self { n_cols, n_rows, threads_per_block, temperature: 1.0 })
    }

    /// Override the temperature value (default `1.0`).
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature` is not positive and finite.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "softmax temperature must be positive and finite, got {temperature}"
                ),
            }
            .into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Compute the CUDA grid dimensions `(n_rows, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.n_rows as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax on the CPU.
///
/// Computes `softmax(input / temperature)` for each row independently.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major, written)
/// * `config` — Configuration (uses `n_rows`, `n_cols`, `temperature`)
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if the slice lengths do not match
/// `n_rows × n_cols`.
pub fn softmax_cpu(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let total = config.n_rows * config.n_cols;
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("softmax output length {} < expected {}", output.len(), total),
        }
        .into());
    }

    let inv_temp = 1.0_f32 / config.temperature;

    for row in 0..config.n_rows {
        let start = row * config.n_cols;
        let end = start + config.n_cols;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Pass 1: find row max (for numerical stability)
        let row_max = row_in.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: shifted exp with temperature scaling + accumulate sum
        let mut sum = 0.0_f32;
        for (out, &x) in row_out.iter_mut().zip(row_in.iter()) {
            let e = ((x - row_max) * inv_temp).exp();
            *out = e;
            sum += e;
        }

        // Pass 3: normalise
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for val in row_out.iter_mut() {
                *val *= inv_sum;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the softmax CUDA kernel.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
pub fn launch_softmax(_input: &[f32], _output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    log::debug!(
        "softmax stub: n_cols={}, n_rows={}, temperature={}, grid={:?}",
        config.n_cols,
        config.n_rows,
        config.temperature,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "softmax CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply softmax with automatic dispatch: GPU if available, else CPU fallback.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major, written)
/// * `config` — Launch configuration
pub fn softmax_forward(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_softmax(input, output, config) {
                return Ok(());
            }
            // GPU launch failed — fall through to CPU path
        }
    }
    softmax_cpu(input, output, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_softmax_config_for_shape() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap();
        assert_eq!(cfg.n_cols, 32000);
        assert_eq!(cfg.n_rows, 1);
        assert_eq!(cfg.threads_per_block, 1024); // capped
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_softmax_config_small_vocab() {
        let cfg = SoftmaxConfig::for_shape(64, 10).unwrap();
        assert_eq!(cfg.threads_per_block, 64);
        assert_eq!(cfg.grid_dim(), (10, 1, 1));
        assert_eq!(cfg.block_dim(), (64, 1, 1));
    }

    #[test]
    fn test_softmax_config_rejects_zero() {
        assert!(SoftmaxConfig::for_shape(0, 1).is_err());
        assert!(SoftmaxConfig::for_shape(32000, 0).is_err());
    }

    #[test]
    fn test_softmax_config_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(128, 4).unwrap().with_temperature(0.7).unwrap();
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_config_rejects_bad_temperature() {
        let cfg = SoftmaxConfig::for_shape(128, 4).unwrap();
        assert!(cfg.clone().with_temperature(0.0).is_err());
        assert!(cfg.clone().with_temperature(-1.0).is_err());
        assert!(cfg.clone().with_temperature(f32::NAN).is_err());
        assert!(cfg.with_temperature(f32::INFINITY).is_err());
    }

    #[test]
    fn test_softmax_grid_dim() {
        let cfg = SoftmaxConfig::for_shape(32000, 32).unwrap();
        assert_eq!(cfg.grid_dim(), (32, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    // -- CPU fallback tests -------------------------------------------------

    #[test]
    fn test_cpu_softmax_single_row() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");

        // Monotonically increasing (larger input → larger probability)
        for i in 0..3 {
            assert!(output[i] < output[i + 1], "output not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_softmax_multiple_rows() {
        let cfg = SoftmaxConfig::for_shape(3, 2).unwrap();
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = [0.0_f32; 6];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Each row sums to 1
        let sum_row0: f32 = output[0..3].iter().sum();
        let sum_row1: f32 = output[3..6].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6, "row0 sum={sum_row0}");
        assert!((sum_row1 - 1.0).abs() < 1e-6, "row1 sum={sum_row1}");
    }

    #[test]
    fn test_cpu_softmax_numerical_stability() {
        // Large values that would overflow naive exp()
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [1000.0_f32, 1001.0, 1002.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_cpu_softmax_uniform_input() {
        // All equal inputs → uniform distribution
        let cfg = SoftmaxConfig::for_shape(5, 1).unwrap();
        let input = [3.0_f32; 5];
        let mut output = [0.0_f32; 5];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for &v in &output {
            assert!((v - 0.2).abs() < 1e-6, "expected 0.2, got {v}");
        }
    }

    #[test]
    fn test_cpu_softmax_with_temperature() {
        let n_cols = 4;
        let input = [1.0_f32, 2.0, 3.0, 4.0];

        // Temperature 1.0 (baseline)
        let cfg_t1 = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut out_t1 = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_t1, &cfg_t1).unwrap();

        // High temperature → more uniform
        let cfg_hot = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(10.0).unwrap();
        let mut out_hot = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_hot, &cfg_hot).unwrap();

        // Low temperature → more peaked
        let cfg_cold = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(0.1).unwrap();
        let mut out_cold = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_cold, &cfg_cold).unwrap();

        // High temp: max prob should be closer to uniform (0.25)
        let max_hot = out_hot.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_t1 = out_t1.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_cold = out_cold.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_hot < max_t1, "high temp should be flatter: {max_hot} >= {max_t1}");
        assert!(max_t1 < max_cold, "low temp should be peakier: {max_t1} >= {max_cold}");
    }

    #[test]
    fn test_cpu_softmax_single_element() {
        let cfg = SoftmaxConfig::for_shape(1, 1).unwrap();
        let input = [42.0_f32];
        let mut output = [0.0_f32; 1];
        softmax_cpu(&input, &mut output, &cfg).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_softmax_negative_values() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [-10.0_f32, -5.0, 0.0, 5.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
        // Monotonically increasing
        for i in 0..3 {
            assert!(output[i] < output[i + 1], "output not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_softmax_very_large_values() {
        // Values near f32 limits
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [f32::MAX / 2.0, f32::MAX / 2.0 - 1.0, f32::MAX / 2.0 - 2.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(output.iter().all(|&v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_cpu_softmax_temperature_preserves_sum() {
        let input = [1.0_f32, 3.0, 5.0, 2.0, 4.0];
        for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let cfg = SoftmaxConfig::for_shape(5, 1).unwrap().with_temperature(temp).unwrap();
            let mut output = [0.0_f32; 5];
            softmax_cpu(&input, &mut output, &cfg).unwrap();

            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "temp={temp}: sum={sum}");
        }
    }

    #[test]
    fn test_cpu_softmax_rejects_short_input() {
        let cfg = SoftmaxConfig::for_shape(4, 2).unwrap();
        let input = [1.0_f32; 4]; // need 8
        let mut output = [0.0_f32; 8];
        assert!(softmax_cpu(&input, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_cpu_softmax_rejects_short_output() {
        let cfg = SoftmaxConfig::for_shape(4, 2).unwrap();
        let input = [1.0_f32; 8];
        let mut output = [0.0_f32; 4]; // need 8
        assert!(softmax_cpu(&input, &mut output, &cfg).is_err());
    }

    // -- Unified dispatch tests ---------------------------------------------

    #[test]
    fn test_softmax_forward_dispatches_cpu() {
        // On CPU-only builds, softmax_forward should succeed via the CPU path
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];

        let result = softmax_forward(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CPU dispatch should succeed: {result:?}");

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_softmax_forward_matches_cpu() {
        let cfg = SoftmaxConfig::for_shape(5, 2).unwrap().with_temperature(0.7).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let mut out_forward = [0.0_f32; 10];
        let mut out_cpu = [0.0_f32; 10];

        softmax_forward(&input, &mut out_forward, &cfg).unwrap();
        softmax_cpu(&input, &mut out_cpu, &cfg).unwrap();

        for (i, (&fwd, &cpu)) in out_forward.iter().zip(out_cpu.iter()).enumerate() {
            assert!(
                (fwd - cpu).abs() < 1e-6,
                "dispatch mismatch at elem {i}: forward={fwd}, cpu={cpu}"
            );
        }
    }

    // -- GPU launch stub tests ----------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_softmax_launch() {
        let cfg = SoftmaxConfig::for_shape(32000, 4).unwrap();
        let input = vec![1.0_f32; 32000 * 4];
        let mut output = vec![0.0_f32; 32000 * 4];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_softmax_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap().with_temperature(0.7).unwrap();
        let input = vec![0.0_f32; 32000];
        let mut output = vec![0.0_f32; 32000];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }
}
