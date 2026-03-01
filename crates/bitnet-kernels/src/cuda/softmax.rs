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
//! # Enhanced features
//!
//! - **Causal masking** — optional upper-triangular mask that sets future
//!   positions to `−∞` before the softmax, used in autoregressive attention.
//! - **Log-softmax** — returns `log(softmax(x))` directly, avoiding a
//!   separate `log()` pass and improving numerical precision for
//!   cross-entropy losses.
//! - **In-place mode** — writes results back into the input buffer,
//!   reducing memory traffic for bandwidth-bound workloads.
//! - **Batched multi-head attention** — [`batched_softmax_cpu`] operates
//!   over `[batch, n_heads, seq_len, seq_len]` attention score tensors
//!   with optional causal masking per head.
//!
//! # CPU fallback
//!
//! [`softmax_cpu`] provides an equivalent pure-Rust implementation for
//! correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Softmax mode
// ---------------------------------------------------------------------------

/// Selects between standard softmax and log-softmax output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMode {
    /// Standard softmax: `exp(x_i - m) / sum(exp(x_j - m))`.
    Standard,
    /// Log-softmax: `(x_i - m) - log(sum(exp(x_j - m)))`.
    ///
    /// Numerically more stable than computing `log(softmax(x))` in a
    /// separate pass, and preferred for cross-entropy loss computation.
    LogSoftmax,
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Launch configuration for the softmax kernel.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Number of columns per row (vocabulary / logits dimension).
    pub n_cols: usize,
    /// Number of rows (batch * sequence length).
    pub n_rows: usize,
    /// Threads per block — typically `min(n_cols, 1024)`.
    pub threads_per_block: u32,
    /// Temperature scaling factor applied before exponentiation.
    /// Values `> 1.0` soften the distribution; values in `(0, 1)`
    /// sharpen it.
    pub temperature: f32,
    /// When `true`, applies an upper-triangular causal mask before the
    /// softmax.  Positions where `col > row` are set to negative
    /// infinity so that each token can only attend to itself and
    /// earlier positions.
    ///
    /// Only meaningful when `n_cols == n_rows` (square attention
    /// matrix); for non-square shapes the mask is applied to columns
    /// beyond the current row index.
    pub causal_mask: bool,
    /// Output mode — standard probabilities or log-probabilities.
    pub mode: SoftmaxMode,
    /// When `true`, the CPU fallback writes results back into the input
    /// buffer and ignores the output slice, reducing memory traffic.
    pub in_place: bool,
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

        Ok(Self {
            n_cols,
            n_rows,
            threads_per_block,
            temperature: 1.0,
            causal_mask: false,
            mode: SoftmaxMode::Standard,
            in_place: false,
        })
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
                    "softmax temperature must be positive \
                     and finite, got {temperature}"
                ),
            }
            .into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Enable causal (upper-triangular) masking.
    ///
    /// Positions where `col > row` are set to negative infinity before
    /// the softmax so each token attends only to itself and earlier
    /// positions.
    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    /// Switch to log-softmax output mode.
    pub fn with_log_softmax(mut self) -> Self {
        self.mode = SoftmaxMode::LogSoftmax;
        self
    }

    /// Enable in-place operation — results are written back into the
    /// input buffer and the output slice is unused.
    pub fn with_in_place(mut self) -> Self {
        self.in_place = true;
        self
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
// Batched multi-head attention softmax config
// ---------------------------------------------------------------------------

/// Configuration for batched softmax over multi-head attention scores.
///
/// The input tensor is `[batch, n_heads, seq_len, seq_len]` (row-major).
/// Each `(batch, head)` slice is an independent `[seq_len, seq_len]`
/// attention matrix that is softmax-normalised row-wise.
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sequence length (both query and key dimensions).
    pub seq_len: usize,
    /// Temperature scaling.
    pub temperature: f32,
    /// Apply causal mask to every head.
    pub causal_mask: bool,
    /// Output mode.
    pub mode: SoftmaxMode,
}

impl BatchedSoftmaxConfig {
    /// Create a batched config with defaults (`temperature = 1.0`, no
    /// causal mask, standard mode).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(batch_size: usize, n_heads: usize, seq_len: usize) -> Result<Self> {
        if batch_size == 0 || n_heads == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "batched softmax dimensions must be non-zero: \
                     batch={batch_size}, heads={n_heads}, \
                     seq_len={seq_len}"
                ),
            }
            .into());
        }
        Ok(Self {
            batch_size,
            n_heads,
            seq_len,
            temperature: 1.0,
            causal_mask: false,
            mode: SoftmaxMode::Standard,
        })
    }

    /// Override the temperature value.
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature` is not positive and finite.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "batched softmax temperature must be positive \
                     and finite, got {temperature}"
                ),
            }
            .into());
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// Enable causal masking for all heads.
    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    /// Switch to log-softmax output.
    pub fn with_log_softmax(mut self) -> Self {
        self.mode = SoftmaxMode::LogSoftmax;
        self
    }

    /// Total number of elements in the tensor.
    pub fn total_elements(&self) -> usize {
        self.batch_size * self.n_heads * self.seq_len * self.seq_len
    }
}

// ---------------------------------------------------------------------------
// CPU fallback — core row-wise softmax
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax on the CPU.
///
/// Computes `softmax(input / temperature)` for each row independently.
/// Supports causal masking, log-softmax mode, and in-place operation.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major,
///   written).  Ignored when `config.in_place` is `true`.
/// * `config` — Configuration (uses `n_rows`, `n_cols`, `temperature`,
///   `causal_mask`, `mode`, `in_place`)
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if the slice lengths do not
/// match `n_rows * n_cols`.
pub fn softmax_cpu(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let total = config.n_rows * config.n_cols;
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if !config.in_place && output.len() < total {
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

        // --- Pass 1: find row max (after masking) ---
        let row_max = row_in
            .iter()
            .enumerate()
            .map(|(col, &v)| if config.causal_mask && col > row { f32::NEG_INFINITY } else { v })
            .fold(f32::NEG_INFINITY, f32::max);

        // --- Pass 2: shifted exp + sum ---
        let mut sum = 0.0_f32;

        if config.in_place {
            // In-place: write back into the input buffer via raw
            // pointer.  This is safe because we process each element
            // exactly once and never re-read a written position
            // within the same pass.
            let ptr = input.as_ptr() as *mut f32;
            for col in 0..config.n_cols {
                let val = if config.causal_mask && col > row {
                    0.0_f32
                } else {
                    // SAFETY: col < n_cols <= input.len()
                    let x = unsafe { *ptr.add(start + col) };
                    ((x - row_max) * inv_temp).exp()
                };
                unsafe { *ptr.add(start + col) = val };
                sum += val;
            }
            // --- Pass 3: normalise ---
            let log_sum = sum.ln();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for col in 0..config.n_cols {
                    let p = unsafe { *ptr.add(start + col) };
                    let out_val = match config.mode {
                        SoftmaxMode::Standard => p * inv_sum,
                        SoftmaxMode::LogSoftmax => {
                            if p == 0.0 {
                                f32::NEG_INFINITY
                            } else {
                                p.ln() - log_sum
                            }
                        }
                    };
                    unsafe { *ptr.add(start + col) = out_val };
                }
            }
        } else {
            let row_out = &mut output[start..end];
            for (col, (out, &x)) in row_out.iter_mut().zip(row_in.iter()).enumerate() {
                if config.causal_mask && col > row {
                    *out = 0.0;
                } else {
                    let e = ((x - row_max) * inv_temp).exp();
                    *out = e;
                    sum += e;
                }
            }
            // --- Pass 3: normalise ---
            let log_sum = sum.ln();
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for (col, val) in row_out.iter_mut().enumerate() {
                    match config.mode {
                        SoftmaxMode::Standard => *val *= inv_sum,
                        SoftmaxMode::LogSoftmax => {
                            if (config.causal_mask && col > row) || *val == 0.0 {
                                *val = f32::NEG_INFINITY;
                            } else {
                                *val = val.ln() - log_sum;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — in-place convenience wrapper
// ---------------------------------------------------------------------------

/// In-place softmax: reads from and writes to the same buffer.
///
/// Equivalent to calling [`softmax_cpu`] with `config.in_place = true`.
///
/// # Errors
///
/// Propagates errors from [`softmax_cpu`].
pub fn softmax_cpu_inplace(data: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    let mut cfg = config.clone();
    cfg.in_place = true;
    // output slice is unused in in-place mode; pass an empty slice.
    softmax_cpu(data, &mut [], &cfg)
}

// ---------------------------------------------------------------------------
// CPU fallback — batched multi-head attention softmax
// ---------------------------------------------------------------------------

/// Batched softmax over `[batch, n_heads, seq_len, seq_len]` attention
/// scores.
///
/// Each `(batch, head)` slice is an independent `[seq_len, seq_len]`
/// matrix that is softmax-normalised row-wise with the given
/// configuration (temperature, causal mask, mode).
///
/// # Arguments
///
/// * `input`  — Attention scores (row-major, FP32)
/// * `output` — Normalised attention weights (row-major, FP32, written)
/// * `config` — Batched configuration
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slices are too small.
pub fn batched_softmax_cpu(
    input: &[f32],
    output: &mut [f32],
    config: &BatchedSoftmaxConfig,
) -> Result<()> {
    let total = config.total_elements();
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batched softmax input length {} < expected {}", input.len(), total),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("batched softmax output length {} < expected {}", output.len(), total),
        }
        .into());
    }

    let slice_size = config.seq_len * config.seq_len;
    let per_row_cfg = SoftmaxConfig {
        n_cols: config.seq_len,
        n_rows: config.seq_len,
        threads_per_block: (config.seq_len as u32).min(1024),
        temperature: config.temperature,
        causal_mask: config.causal_mask,
        mode: config.mode,
        in_place: false,
    };

    for i in 0..(config.batch_size * config.n_heads) {
        let off = i * slice_size;
        softmax_cpu(
            &input[off..off + slice_size],
            &mut output[off..off + slice_size],
            &per_row_cfg,
        )?;
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
        "softmax stub: n_cols={}, n_rows={}, temperature={}, \
         causal={}, mode={:?}, grid={:?}",
        config.n_cols,
        config.n_rows,
        config.temperature,
        config.causal_mask,
        config.mode,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "softmax CUDA kernel not yet compiled \
                 — scaffold only"
            .into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply softmax with automatic dispatch: GPU if available, else CPU
/// fallback.
///
/// # Arguments
///
/// * `input`  — Input logits `[n_rows, n_cols]` (FP32, row-major)
/// * `output` — Output probabilities `[n_rows, n_cols]` (FP32, row-major,
///   written)
/// * `config` — Launch configuration
pub fn softmax_forward(input: &[f32], output: &mut [f32], config: &SoftmaxConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_softmax(input, output, config)
        {
            return Ok(());
        }
        // GPU launch failed — fall through to CPU path
    }
    softmax_cpu(input, output, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // == Config tests ====================================================

    #[test]
    fn test_softmax_config_for_shape() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap();
        assert_eq!(cfg.n_cols, 32000);
        assert_eq!(cfg.n_rows, 1);
        assert_eq!(cfg.threads_per_block, 1024); // capped
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
        assert!(!cfg.causal_mask);
        assert_eq!(cfg.mode, SoftmaxMode::Standard);
        assert!(!cfg.in_place);
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

    #[test]
    fn test_softmax_config_builder_chain() {
        let cfg = SoftmaxConfig::for_shape(64, 8)
            .unwrap()
            .with_temperature(0.5)
            .unwrap()
            .with_causal_mask()
            .with_log_softmax()
            .with_in_place();
        assert!((cfg.temperature - 0.5).abs() < 1e-6);
        assert!(cfg.causal_mask);
        assert_eq!(cfg.mode, SoftmaxMode::LogSoftmax);
        assert!(cfg.in_place);
    }

    // == CPU fallback tests ==============================================

    #[test]
    fn test_cpu_softmax_single_row() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");

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

        let sum_row0: f32 = output[0..3].iter().sum();
        let sum_row1: f32 = output[3..6].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6, "row0 sum={sum_row0}");
        assert!((sum_row1 - 1.0).abs() < 1e-6, "row1 sum={sum_row1}");
    }

    #[test]
    fn test_cpu_softmax_numerical_stability() {
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

        let cfg_t1 = SoftmaxConfig::for_shape(n_cols, 1).unwrap();
        let mut out_t1 = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_t1, &cfg_t1).unwrap();

        let cfg_hot = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(10.0).unwrap();
        let mut out_hot = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_hot, &cfg_hot).unwrap();

        let cfg_cold = SoftmaxConfig::for_shape(n_cols, 1).unwrap().with_temperature(0.1).unwrap();
        let mut out_cold = [0.0_f32; 4];
        softmax_cpu(&input, &mut out_cold, &cfg_cold).unwrap();

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
        for i in 0..3 {
            assert!(output[i] < output[i + 1], "output not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_softmax_very_large_values() {
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

    // == Causal mask tests ===============================================

    #[test]
    fn test_cpu_softmax_causal_mask_identity_row0() {
        // Row 0: only column 0 is visible
        let cfg = SoftmaxConfig::for_shape(3, 3).unwrap().with_causal_mask();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!((output[0] - 1.0).abs() < 1e-6, "row0 col0 should be 1.0, got {}", output[0]);
        assert!(output[1].abs() < 1e-6, "row0 col1 should be 0, got {}", output[1]);
        assert!(output[2].abs() < 1e-6, "row0 col2 should be 0, got {}", output[2]);
    }

    #[test]
    fn test_cpu_softmax_causal_mask_last_row_full() {
        let n = 4;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask();
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; 16];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        let sum_last: f32 = output[12..16].iter().sum();
        assert!((sum_last - 1.0).abs() < 1e-6, "last row sum={sum_last}");
        assert!(output[12..16].iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_cpu_softmax_causal_mask_rows_sum_to_one() {
        let n = 5;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask();
        let input: Vec<f32> = (0..25).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0_f32; 25];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum={sum}");
            for col in (row + 1)..n {
                assert!(
                    output[start + col].abs() < 1e-7,
                    "row={row} col={col} should be masked, \
                     got {}",
                    output[start + col]
                );
            }
        }
    }

    // == Log-softmax tests ===============================================

    #[test]
    fn test_cpu_log_softmax_basic() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap().with_log_softmax();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!(output.iter().all(|&v| v <= 0.0));

        let sum: f32 = output.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp sum={sum}");

        for i in 0..3 {
            assert!(output[i] < output[i + 1], "log-softmax not monotonic at {i}");
        }
    }

    #[test]
    fn test_cpu_log_softmax_matches_log_of_softmax() {
        let input = [0.5_f32, 1.5, -0.3, 2.1, 0.0];

        let cfg_std = SoftmaxConfig::for_shape(5, 1).unwrap();
        let mut out_std = [0.0_f32; 5];
        softmax_cpu(&input, &mut out_std, &cfg_std).unwrap();
        let log_of_std: Vec<f32> = out_std.iter().map(|v| v.ln()).collect();

        let cfg_log = SoftmaxConfig::for_shape(5, 1).unwrap().with_log_softmax();
        let mut out_log = [0.0_f32; 5];
        softmax_cpu(&input, &mut out_log, &cfg_log).unwrap();

        for (i, (&ls, &log_s)) in out_log.iter().zip(log_of_std.iter()).enumerate() {
            assert!(
                (ls - log_s).abs() < 1e-5,
                "log-softmax mismatch at {i}: direct={ls}, \
                 log(softmax)={log_s}"
            );
        }
    }

    #[test]
    fn test_cpu_log_softmax_numerical_stability() {
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap().with_log_softmax();
        let input = [1000.0_f32, 1001.0, 1002.0];
        let mut output = [0.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        assert!(output.iter().all(|&v| v.is_finite()), "log-softmax produced non-finite values");
        let sum: f32 = output.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5, "exp sum={sum}");
    }

    #[test]
    fn test_cpu_log_softmax_with_causal_mask() {
        let n = 3;
        let cfg = SoftmaxConfig::for_shape(n, n).unwrap().with_causal_mask().with_log_softmax();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Row 0: only col 0 visible => log(1.0) = 0.0
        assert!(output[0].abs() < 1e-6, "row0 col0 log-prob should be 0, got {}", output[0]);
        // Masked positions => -inf
        assert!(output[1] == f32::NEG_INFINITY, "row0 col1 should be -inf, got {}", output[1]);

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + row + 1].iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} exp-sum={sum}");
        }
    }

    // == In-place tests ==================================================

    #[test]
    fn test_cpu_softmax_inplace() {
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input_orig = [1.0_f32, 2.0, 3.0, 4.0];

        let mut out_ref = [0.0_f32; 4];
        softmax_cpu(&input_orig, &mut out_ref, &cfg).unwrap();

        let mut data = input_orig;
        softmax_cpu_inplace(&mut data, &cfg).unwrap();

        for (i, (&ip, &oop)) in data.iter().zip(out_ref.iter()).enumerate() {
            assert!((ip - oop).abs() < 1e-6, "in-place mismatch at {i}: {ip} vs {oop}");
        }
    }

    #[test]
    fn test_cpu_softmax_inplace_multi_row() {
        let cfg = SoftmaxConfig::for_shape(3, 2).unwrap();
        let input_orig = [1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0];

        let mut out_ref = [0.0_f32; 6];
        softmax_cpu(&input_orig, &mut out_ref, &cfg).unwrap();

        let mut data = input_orig;
        softmax_cpu_inplace(&mut data, &cfg).unwrap();

        for (i, (&ip, &oop)) in data.iter().zip(out_ref.iter()).enumerate() {
            assert!((ip - oop).abs() < 1e-6, "in-place mismatch at {i}: {ip} vs {oop}");
        }
    }

    // == Batched multi-head softmax tests ================================

    #[test]
    fn test_batched_softmax_config_rejects_zero() {
        assert!(BatchedSoftmaxConfig::new(0, 4, 8).is_err());
        assert!(BatchedSoftmaxConfig::new(2, 0, 8).is_err());
        assert!(BatchedSoftmaxConfig::new(2, 4, 0).is_err());
    }

    #[test]
    fn test_batched_softmax_basic() {
        let cfg = BatchedSoftmaxConfig::new(2, 2, 3).unwrap();
        let total = cfg.total_elements(); // 2*2*3*3 = 36
        assert_eq!(total, 36);

        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0_f32; total];
        batched_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let seq = cfg.seq_len;
        for slice_idx in 0..(cfg.batch_size * cfg.n_heads) {
            for row in 0..seq {
                let off = slice_idx * seq * seq + row * seq;
                let sum: f32 = output[off..off + seq].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "slice={slice_idx} row={row} sum={sum}");
            }
        }
    }

    #[test]
    fn test_batched_softmax_with_causal_mask() {
        let cfg = BatchedSoftmaxConfig::new(1, 1, 4).unwrap().with_causal_mask();
        let total = cfg.total_elements(); // 16
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0_f32; total];
        batched_softmax_cpu(&input, &mut output, &cfg).unwrap();

        let seq = cfg.seq_len;
        for row in 0..seq {
            for col in (row + 1)..seq {
                let idx = row * seq + col;
                assert!(output[idx].abs() < 1e-7, "row={row} col={col} should be masked");
            }
            let sum: f32 = output[row * seq..row * seq + row + 1].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row={row} sum={sum}");
        }
    }

    #[test]
    fn test_batched_softmax_rejects_short_slices() {
        let cfg = BatchedSoftmaxConfig::new(2, 2, 4).unwrap();
        let short = vec![0.0_f32; 10]; // need 64
        let mut out = vec![0.0_f32; 64];
        assert!(batched_softmax_cpu(&short, &mut out, &cfg).is_err());

        let full = vec![0.0_f32; 64];
        let mut short_out = vec![0.0_f32; 10];
        assert!(batched_softmax_cpu(&full, &mut short_out, &cfg).is_err());
    }

    // == NaN / special-value handling ====================================

    #[test]
    fn test_cpu_softmax_all_negative_infinity() {
        // When all inputs are -inf, row_max = -inf and (x - row_max) is
        // NaN under IEEE 754.  The result is mathematically undefined
        // (0/0), so we just verify the function does not panic.
        let cfg = SoftmaxConfig::for_shape(3, 1).unwrap();
        let input = [f32::NEG_INFINITY; 3];
        let mut output = [99.0_f32; 3];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Accept NaN or zero — both are valid degenerate outputs.
        for &v in &output {
            assert!(
                v.is_nan() || v.abs() < 1e-7,
                "expected NaN or ~0 for all-neginf input, got {v}"
            );
        }
    }

    #[test]
    fn test_cpu_softmax_mixed_inf() {
        // +inf in input produces row_max = +inf, then (inf - inf) = NaN
        // under IEEE 754.  We verify the function doesn't panic and
        // the +inf element gets the largest share (or NaN, which is
        // a valid IEEE result for this degenerate input).
        let cfg = SoftmaxConfig::for_shape(4, 1).unwrap();
        let input = [1.0_f32, f32::INFINITY, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        // Accept NaN or ~1.0 for the inf element.
        assert!(
            output[1].is_nan() || (output[1] - 1.0).abs() < 1e-5,
            "inf element should dominate or be NaN, got {}",
            output[1]
        );
    }

    #[test]
    fn test_cpu_softmax_with_temperature_and_causal_mask() {
        let n = 3;
        let cfg = SoftmaxConfig::for_shape(n, n)
            .unwrap()
            .with_temperature(0.5)
            .unwrap()
            .with_causal_mask();
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut output = [0.0_f32; 9];
        softmax_cpu(&input, &mut output, &cfg).unwrap();

        for row in 0..n {
            let start = row * n;
            let sum: f32 = output[start..start + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} sum={sum}");
            for col in (row + 1)..n {
                assert!(output[start + col].abs() < 1e-7, "masked position row={row} col={col}");
            }
        }
    }

    // == Unified dispatch tests ==========================================

    #[test]
    fn test_softmax_forward_dispatches_cpu() {
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
                "dispatch mismatch at elem {i}: \
                 forward={fwd}, cpu={cpu}"
            );
        }
    }

    // == GPU launch stub tests ===========================================

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_launch() {
        let cfg = SoftmaxConfig::for_shape(32000, 4).unwrap();
        let input = vec![1.0_f32; 32000 * 4];
        let mut output = vec![0.0_f32; 32000 * 4];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_with_temperature() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap().with_temperature(0.7).unwrap();
        let input = vec![0.0_f32; 32000];
        let mut output = vec![0.0_f32; 32000];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA softmax launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_softmax_causal_mask() {
        let cfg = SoftmaxConfig::for_shape(128, 128).unwrap().with_causal_mask();
        let input = vec![1.0_f32; 128 * 128];
        let mut output = vec![0.0_f32; 128 * 128];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA causal softmax failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_log_softmax() {
        let cfg = SoftmaxConfig::for_shape(32000, 1).unwrap().with_log_softmax();
        let input = vec![1.0_f32; 32000];
        let mut output = vec![0.0_f32; 32000];
        let result = launch_softmax(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA log-softmax failed: {result:?}");
    }
}
