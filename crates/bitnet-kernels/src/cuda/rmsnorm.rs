//! RMSNorm CUDA kernel.
//!
//! # Kernel strategy
//!
//! Root Mean Square Layer Normalization avoids the mean-subtraction step of
//! LayerNorm, making it cheaper and better suited to 1-bit quantised models:
//!
//!   `y[i] = (x[i] / rms(x)) * gamma[i]`
//!
//! where `rms(x) = sqrt(mean(x²) + eps)`.
//!
//! The kernel is a single-pass warp-level reduction:
//!
//! 1. Each thread computes partial `x²` sums for its assigned elements.
//! 2. A warp-shuffle tree reduces partial sums to lane 0.
//! 3. Lane 0 computes `rms = sqrt(sum / n + eps)` and broadcasts `1/rms`.
//! 4. Every thread multiplies its elements by `(1/rms) * gamma[i]` and writes
//!    the normalised output.
//!
//! One thread-block handles one row (one token position). Grid size equals the
//! batch/sequence dimension.
//!
//! Target: full warp utilisation when `hidden_dim ≥ 32`. For typical BitNet
//! hidden dims (2048–4096) each warp processes 64–128 elements, yielding
//! excellent memory-bandwidth utilisation on Ampere+.

use bitnet_common::{KernelError, Result};

/// Launch configuration for the RMSNorm kernel.
#[derive(Debug, Clone)]
pub struct RmsNormConfig {
    /// Hidden dimension (number of elements per row to normalise).
    pub hidden_dim: usize,
    /// Number of rows (batch × sequence length).
    pub n_rows: usize,
    /// Threads per block — typically `min(hidden_dim, 1024)`.
    pub threads_per_block: u32,
    /// Epsilon added inside the square root for numerical stability.
    pub eps: f32,
}

impl RmsNormConfig {
    /// Create a configuration for the given shape.
    pub fn for_shape(hidden_dim: usize, n_rows: usize) -> Result<Self> {
        if hidden_dim == 0 || n_rows == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "RMSNorm dimensions must be non-zero: \
                     hidden_dim={hidden_dim}, n_rows={n_rows}"
                ),
            }
            .into());
        }

        let threads_per_block = (hidden_dim as u32).min(1024);

        Ok(Self { hidden_dim, n_rows, threads_per_block, eps: 1e-6 })
    }

    /// Override the epsilon value (default `1e-6`).
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
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

/// Launch stub for the RMSNorm kernel.
///
/// # Arguments
///
/// * `input`  — Input tensor `[n_rows, hidden_dim]` (FP32)
/// * `gamma`  — Per-element scale weights `[hidden_dim]` (FP32)
/// * `output` — Output buffer `[n_rows, hidden_dim]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_rmsnorm(
    _input: &[f32],
    _gamma: &[f32],
    _output: &mut [f32],
    config: &RmsNormConfig,
) -> Result<()> {
    log::debug!(
        "RMSNorm stub: hidden_dim={}, n_rows={}, eps={}, grid={:?}",
        config.hidden_dim,
        config.n_rows,
        config.eps,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "RMSNorm CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_config_for_shape() {
        let cfg = RmsNormConfig::for_shape(2048, 1).unwrap();
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.n_rows, 1);
        assert_eq!(cfg.threads_per_block, 1024); // capped at 1024
        assert!((cfg.eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_rmsnorm_config_small_hidden() {
        let cfg = RmsNormConfig::for_shape(64, 10).unwrap();
        assert_eq!(cfg.threads_per_block, 64); // hidden_dim < 1024
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 10); // one block per row
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_rmsnorm_config_rejects_zero() {
        assert!(RmsNormConfig::for_shape(0, 1).is_err());
        assert!(RmsNormConfig::for_shape(2048, 0).is_err());
    }

    #[test]
    fn test_rmsnorm_config_with_eps() {
        let cfg = RmsNormConfig::for_shape(128, 4).unwrap().with_eps(1e-5);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_rmsnorm_grid_dim() {
        let cfg = RmsNormConfig::for_shape(4096, 32).unwrap();
        assert_eq!(cfg.grid_dim(), (32, 1, 1));
        assert_eq!(cfg.block_dim(), (1024, 1, 1));
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rmsnorm_launch() {
        let cfg = RmsNormConfig::for_shape(2048, 4).unwrap();
        let input = vec![1.0f32; 2048 * 4];
        let gamma = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 2048 * 4];
        let result = launch_rmsnorm(&input, &gamma, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA RMSNorm launch failed: {result:?}");
    }
}
