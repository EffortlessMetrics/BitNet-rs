//! Fused multi-head attention kernel stubs for ROCm/HIP.
//!
//! Mirrors planned CUDA attention kernels.  On AMD hardware the optimal
//! strategy is a single fused HIP kernel that performs:
//!
//! 1. Q·K^T scaled dot-product (using LDS for tile-level reduction)
//! 2. Causal mask application
//! 3. Online softmax (numerically stable, streaming)
//! 4. Attention · V output projection
//!
//! # HIP-specific considerations
//!
//! * **Wavefront size**: 64 threads on GCN / CDNA (vs 32 for NVIDIA warps).
//!   Reduction primitives must use `__shfl_xor` with a width of 64.
//! * **LDS budget**: up to 64 KiB per work-group on MI200-series; tiles
//!   should be sized to fit Q/K/V tiles plus the softmax accumulator.
//! * **Matrix cores**: CDNA2 (MI250) exposes MFMA instructions for FP16/BF16
//!   matrix multiply-accumulate — these should be preferred when available.

use bitnet_common::{KernelError, Result};

/// Configuration for the fused attention HIP kernel (stub).
#[derive(Debug, Clone, Copy)]
pub struct HipAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension (d_k).
    pub head_dim: usize,
    /// Whether to apply a causal (triangular) mask.
    pub causal: bool,
    /// Softmax temperature scale factor (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
}

impl HipAttentionConfig {
    /// Create a config with sensible defaults for a given `head_dim`.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            causal: true,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

/// Execute fused multi-head attention via HIP.
///
/// # Arguments
///
/// * `q` — Query tensor `[batch, num_heads, seq_len, head_dim]` (row-major f32).
/// * `k` — Key tensor   (same layout as `q`).
/// * `v` — Value tensor  (same layout as `q`).
/// * `output` — Output buffer (same shape as `q`).
///
/// # Errors
///
/// Always returns [`KernelError::ExecutionFailed`] — stub only.
pub fn fused_attention_hip(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    _seq_len: usize,
    _config: &HipAttentionConfig,
) -> Result<()> {
    Err(bitnet_common::BitNetError::Kernel(
        KernelError::ExecutionFailed {
            reason: "ROCm/HIP fused attention kernel is not yet implemented".into(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_returns_err() {
        let cfg = HipAttentionConfig::new(8, 64);
        let q = vec![0.0f32; 8 * 64];
        let k = vec![0.0f32; 8 * 64];
        let v = vec![0.0f32; 8 * 64];
        let mut out = vec![0.0f32; 8 * 64];
        assert!(fused_attention_hip(&q, &k, &v, &mut out, 1, &cfg).is_err());
    }

    #[test]
    fn config_scale_matches_head_dim() {
        let cfg = HipAttentionConfig::new(8, 64);
        let expected = 1.0 / (64.0f32).sqrt();
        assert!((cfg.scale - expected).abs() < 1e-6);
        assert!(cfg.causal);
    }
}
