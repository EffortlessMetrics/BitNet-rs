//! HIP attention kernel stubs with CPU fallback.
//!
//! Provides scaled dot-product attention mirroring the CUDA interface
//! in [`crate::cuda::attention`]. Includes both causal and non-causal
//! variants, with a pure-Rust CPU fallback for correctness testing.
//!
//! # HIP-specific considerations
//!
//! * Wavefront-level reductions use width-64 `__shfl_xor` on GCN/CDNA.
//! * LDS tiles for Q/K/V should fit within the per-work-group LDS budget.
//! * MFMA instructions (CDNA2+) can accelerate the QK^T matmul when
//!   `head_dim` aligns to 16.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ────────────────────────────────────────────────────

/// Mask mode for attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipAttentionMask {
    /// No masking — full attention.
    None,
    /// Causal (lower-triangular) mask for autoregressive decoding.
    Causal,
}

/// Configuration for the HIP attention kernel.
#[derive(Debug, Clone)]
pub struct HipAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension (d_k = d_v).
    pub head_dim: usize,
    /// Query sequence length.
    pub seq_len_q: usize,
    /// Key/Value sequence length.
    pub seq_len_kv: usize,
    /// Softmax scale factor (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Mask mode.
    pub mask: HipAttentionMask,
    /// Work-group size for the HIP kernel.
    pub workgroup_size: u32,
}

impl HipAttentionConfig {
    /// Create a new config with default scale and causal masking.
    pub fn new(num_heads: usize, head_dim: usize, seq_len_q: usize, seq_len_kv: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            seq_len_q,
            seq_len_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask: HipAttentionMask::Causal,
            workgroup_size: 256,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_heads must be non-zero".into(),
            }
            .into());
        }
        if self.head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim must be non-zero".into(),
            }
            .into());
        }
        if self.seq_len_q == 0 || self.seq_len_kv == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "sequence lengths must be non-zero".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── HIP kernel source (stub) ────────────────────────────────────────

/// HIP C source for scaled dot-product attention.
///
/// Stub — will contain HIP C++ kernel code once implementation begins.
#[cfg(feature = "rocm")]
pub const HIP_ATTENTION_KERNEL_SRC: &str = r#"
// TODO: HIP attention kernel with LDS tiling and online softmax
extern "C" __global__ void sdp_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q, int seq_len_kv, int head_dim, float scale)
{
    // Stub — to be implemented
}
"#;

// ── CPU fallback ─────────────────────────────────────────────────────

/// Single-head scaled dot-product attention (CPU fallback).
///
/// Computes `softmax(Q·K^T / scale) · V` for one attention head.
pub fn hip_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    config: &HipAttentionConfig,
) -> Result<()> {
    config.validate()?;
    let (sq, skv, d) = (config.seq_len_q, config.seq_len_kv, config.head_dim);

    if q.len() < sq * d {
        return Err(KernelError::InvalidArguments {
            reason: format!("Q buffer too small: {} < {}", q.len(), sq * d),
        }
        .into());
    }
    if k.len() < skv * d || v.len() < skv * d {
        return Err(
            KernelError::InvalidArguments { reason: "K or V buffer too small".into() }.into()
        );
    }
    if output.len() < sq * d {
        return Err(
            KernelError::InvalidArguments { reason: "output buffer too small".into() }.into()
        );
    }

    for qi in 0..sq {
        // Compute scores: QK^T * scale
        let mut scores = vec![0.0f32; skv];
        for ki in 0..skv {
            let mut dot = 0.0f32;
            for dd in 0..d {
                dot += q[qi * d + dd] * k[ki * d + dd];
            }
            scores[ki] = dot * config.scale;

            // Apply causal mask
            if config.mask == HipAttentionMask::Causal && ki > qi {
                scores[ki] = f32::NEG_INFINITY;
            }
        }

        // Stable softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        if sum_exp > 0.0 {
            for s in &mut scores {
                *s /= sum_exp;
            }
        }

        // Weighted sum of V
        for dd in 0..d {
            let mut val = 0.0f32;
            for ki in 0..skv {
                val += scores[ki] * v[ki * d + dd];
            }
            output[qi * d + dd] = val;
        }
    }

    Ok(())
}

/// Multi-head attention CPU fallback.
///
/// Processes each head independently using [`hip_attention_cpu`].
pub fn hip_multi_head_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    config: &HipAttentionConfig,
) -> Result<()> {
    config.validate()?;
    let head_size = config.seq_len_q * config.head_dim;
    let kv_head_size = config.seq_len_kv * config.head_dim;

    for h in 0..config.num_heads {
        let q_slice = &q[h * head_size..(h + 1) * head_size];
        let k_slice = &k[h * kv_head_size..(h + 1) * kv_head_size];
        let v_slice = &v[h * kv_head_size..(h + 1) * kv_head_size];
        let o_slice = &mut output[h * head_size..(h + 1) * head_size];

        let single_head_config = HipAttentionConfig { num_heads: 1, ..config.clone() };
        hip_attention_cpu(q_slice, k_slice, v_slice, o_slice, &single_head_config)?;
    }
    Ok(())
}

/// Dispatch attention to HIP GPU or fall back to CPU.
pub fn hip_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    config: &HipAttentionConfig,
) -> Result<()> {
    // TODO: dispatch to HIP kernel when runtime is available
    hip_attention_cpu(q, k, v, output, config)
}

/// Launch the HIP attention kernel on the GPU.
///
/// Stub — returns error until HIP runtime integration.
#[cfg(feature = "rocm")]
pub fn launch_hip_attention(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    _config: &HipAttentionConfig,
) -> Result<()> {
    Err(BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "HIP attention kernel is not yet implemented".into(),
    }))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn config_default_scale() {
        let cfg = HipAttentionConfig::new(8, 64, 10, 10);
        let expected = 1.0 / (64.0f32).sqrt();
        assert!((cfg.scale - expected).abs() < 1e-6);
    }

    #[test]
    fn config_validate_ok() {
        let cfg = HipAttentionConfig::new(4, 32, 8, 8);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_zero_heads() {
        let cfg = HipAttentionConfig::new(0, 32, 8, 8);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_dim() {
        let cfg = HipAttentionConfig::new(4, 0, 8, 8);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_seq() {
        let cfg = HipAttentionConfig::new(4, 32, 0, 8);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn attention_identity_value() {
        // Q=K (uniform), V=identity-like => output ≈ V row average
        let d = 4;
        let seq = 2;
        let q = vec![1.0; seq * d];
        let k = vec![1.0; seq * d];
        // 2×4
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut out = vec![0.0; seq * d];
        let mut cfg = HipAttentionConfig::new(1, d, seq, seq);
        cfg.mask = HipAttentionMask::None;
        hip_attention_cpu(&q, &k, &v, &mut out, &cfg).unwrap();
        // Uniform attention => output = average of V rows
        let expected = vec![0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        assert!(approx_eq(&out, &expected, 1e-5));
    }

    #[test]
    fn causal_mask_first_position() {
        // With causal mask, position 0 can only attend to position 0
        let d = 2;
        let q = vec![1.0, 0.0, 1.0, 0.0]; // 2×2
        let k = vec![1.0, 0.0, 1.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0]; // 2×2
        let mut out = vec![0.0; 4];
        let cfg = HipAttentionConfig::new(1, d, 2, 2);
        hip_attention_cpu(&q, &k, &v, &mut out, &cfg).unwrap();
        // Position 0 attends only to itself => [10, 20]
        assert!((out[0] - 10.0).abs() < 1e-4);
        assert!((out[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        // Verify that attention weights sum to ~1 by checking output
        let d = 2;
        let q = vec![0.5, 0.5];
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2×2
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; d];
        let mut cfg = HipAttentionConfig::new(1, d, 1, 2);
        cfg.mask = HipAttentionMask::None;
        hip_attention_cpu(&q, &k, &v, &mut out, &cfg).unwrap();
        // Output should be a convex combination
        assert!(out[0] >= 0.0 && out[0] <= 1.0);
        assert!(out[1] >= 0.0 && out[1] <= 1.0);
    }

    #[test]
    fn buffer_too_small_q() {
        let q = vec![1.0; 2];
        let k = vec![1.0; 8];
        let v = vec![1.0; 8];
        let mut out = vec![0.0; 8];
        let cfg = HipAttentionConfig::new(1, 4, 2, 2);
        assert!(hip_attention_cpu(&q, &k, &v, &mut out, &cfg).is_err());
    }

    #[test]
    fn buffer_too_small_output() {
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v = vec![1.0; 8];
        let mut out = vec![0.0; 2];
        let cfg = HipAttentionConfig::new(1, 4, 2, 2);
        assert!(hip_attention_cpu(&q, &k, &v, &mut out, &cfg).is_err());
    }

    #[test]
    fn multi_head_attention_two_heads() {
        let d = 2;
        let seq = 1;
        let n_heads = 2;
        let head_size = seq * d;
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 heads, 1×2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; n_heads * head_size];
        let mut cfg = HipAttentionConfig::new(n_heads, d, seq, seq);
        cfg.mask = HipAttentionMask::None;
        hip_multi_head_attention_cpu(&q, &k, &v, &mut out, &cfg).unwrap();
        // Each head: single position → output = V row
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 4.0).abs() < 1e-5);
        assert!((out[2] - 5.0).abs() < 1e-5);
        assert!((out[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn forward_dispatches_to_cpu() {
        let d = 2;
        let q = vec![1.0; d];
        let k = vec![1.0; d];
        let v = vec![1.0; d];
        let mut out = vec![0.0; d];
        let cfg = HipAttentionConfig::new(1, d, 1, 1);
        hip_attention_forward(&q, &k, &v, &mut out, &cfg).unwrap();
        assert!(approx_eq(&out, &[1.0, 1.0], 1e-5));
    }

    #[test]
    fn mask_enum_equality() {
        assert_eq!(HipAttentionMask::Causal, HipAttentionMask::Causal);
        assert_ne!(HipAttentionMask::None, HipAttentionMask::Causal);
    }
}
