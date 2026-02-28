//! RMSNorm forward-pass kernel stubs for ROCm/HIP.
//!
//! Root Mean Square Layer Normalization is used in LLaMA-family
//! architectures (and by extension BitNet) instead of classic LayerNorm.
//! The operation for a vector **x** of dimension *d* with weight **γ** is:
//!
//! ```text
//! rms  = sqrt( (1/d) * Σ x_i² + ε )
//! out_i = (x_i / rms) * γ_i
//! ```
//!
//! # HIP implementation plan
//!
//! 1. Each work-group handles one row (token position).
//! 2. Parallel reduction across the hidden dimension computes Σ x².
//! 3. A single-thread normalisation pass writes the output.
//!
//! Wavefront-level intrinsics (`__shfl_xor`, width 64) are used for the
//! reduction step, followed by LDS if the hidden dim exceeds one wavefront.

use bitnet_common::{KernelError, Result};

/// Configuration for the HIP RMSNorm kernel (stub).
#[derive(Debug, Clone, Copy)]
pub struct HipRmsNormConfig {
    /// Hidden dimension (number of elements per row to normalise).
    pub hidden_dim: usize,
    /// Epsilon added inside the square-root for numerical stability.
    pub eps: f32,
}

impl HipRmsNormConfig {
    /// Create a config with default epsilon (1e-6).
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim, eps: 1e-6 }
    }
}

/// Execute an RMSNorm forward pass via HIP.
///
/// # Arguments
///
/// * `input`  — Input activations `[num_rows, hidden_dim]`.
/// * `gamma`  — Learnable scale weights `[hidden_dim]`.
/// * `output` — Output buffer (same shape as `input`).
///
/// # Errors
///
/// Always returns [`KernelError::ExecutionFailed`] — stub only.
pub fn rmsnorm_hip(
    _input: &[f32],
    _gamma: &[f32],
    _output: &mut [f32],
    _num_rows: usize,
    _config: &HipRmsNormConfig,
) -> Result<()> {
    Err(bitnet_common::BitNetError::Kernel(
        KernelError::ExecutionFailed {
            reason: "ROCm/HIP RMSNorm kernel is not yet implemented".into(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_returns_err() {
        let cfg = HipRmsNormConfig::new(64);
        let input = vec![1.0f32; 64];
        let gamma = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 64];
        assert!(rmsnorm_hip(&input, &gamma, &mut output, 1, &cfg).is_err());
    }

    #[test]
    fn default_eps() {
        let cfg = HipRmsNormConfig::new(128);
        assert!((cfg.eps - 1e-6).abs() < 1e-10);
        assert_eq!(cfg.hidden_dim, 128);
    }
}
