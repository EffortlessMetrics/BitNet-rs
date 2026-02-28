//! Fused multi-head attention kernel for ROCm/HIP.
//!
//! On AMD hardware the optimal strategy is a single fused HIP kernel
//! that performs:
//!
//! 1. Q·K^T scaled dot-product (using LDS for tile-level reduction)
//! 2. Causal mask application
//! 3. Online softmax (numerically stable, streaming)
//! 4. Attention · V output projection
//!
//! # HIP-specific considerations
//!
//! * **Wavefront size**: 64 threads on GCN / CDNA (vs 32 for NVIDIA warps).
//! * **LDS budget**: up to 64 KiB per work-group on MI200-series.
//! * **Matrix cores**: CDNA2 (MI250) exposes MFMA instructions for FP16/BF16.

use bitnet_common::{KernelError, Result};

/// Configuration for the fused attention HIP kernel.
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
        Self { num_heads, head_dim, causal: true, scale: 1.0 / (head_dim as f32).sqrt() }
    }

    /// Override the causal mask setting.
    #[must_use]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

// ── CPU fallback ─────────────────────────────────────────────────────

/// CPU scalar fallback for fused multi-head attention.
///
/// Layout (all tensors row-major f32):
///   `q`: `[num_heads, seq_len_q, head_dim]`
///   `k`: `[num_heads, seq_len_kv, head_dim]`
///   `v`: `[num_heads, seq_len_kv, head_dim]`
///   `output`: `[num_heads, seq_len_q, head_dim]`
pub fn fused_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    config: &HipAttentionConfig,
) -> Result<()> {
    let nh = config.num_heads;
    let d = config.head_dim;
    let scale = config.scale;

    if nh == 0 || d == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("attention: num_heads={nh}, head_dim={d} must be non-zero"),
        }
        .into());
    }
    if seq_len_q == 0 || seq_len_kv == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "attention: seq_len_q={seq_len_q}, seq_len_kv={seq_len_kv} \
                 must be non-zero"
            ),
        }
        .into());
    }

    let q_head = seq_len_q * d;
    let kv_head = seq_len_kv * d;

    for h in 0..nh {
        let q_off = h * q_head;
        let k_off = h * kv_head;
        let v_off = h * kv_head;
        let o_off = h * q_head;

        for qi in 0..seq_len_q {
            let kv_limit = if config.causal { (qi + 1).min(seq_len_kv) } else { seq_len_kv };

            // Find max for numerical stability
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = vec![f32::NEG_INFINITY; seq_len_kv];
            for kj in 0..kv_limit {
                let mut dot = 0.0f32;
                for di in 0..d {
                    dot += q[q_off + qi * d + di] * k[k_off + kj * d + di];
                }
                scores[kj] = dot * scale;
                if scores[kj] > max_score {
                    max_score = scores[kj];
                }
            }

            // Softmax
            let mut sum = 0.0f32;
            for s in &mut scores[..kv_limit] {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for s in &mut scores[..kv_limit] {
                    *s *= inv;
                }
            }

            // Weighted sum: output[qi] = Σ_j attn[j] * v[j]
            for di in 0..d {
                let mut acc = 0.0f32;
                for kj in 0..kv_limit {
                    acc += scores[kj] * v[v_off + kj * d + di];
                }
                output[o_off + qi * d + di] = acc;
            }
        }
    }

    Ok(())
}

// ── HIP dispatch ─────────────────────────────────────────────────────

/// Execute fused multi-head attention, dispatching to HIP when available.
pub fn fused_attention_hip(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    config: &HipAttentionConfig,
) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        if super::is_rocm_available() {
            return fused_attention_hip_device(q, k, v, output, seq_len_q, seq_len_kv, config);
        }
    }

    fused_attention_cpu(q, k, v, output, seq_len_q, seq_len_kv, config)
}

/// HIP device-side fused attention launch.
#[cfg(feature = "rocm")]
fn fused_attention_hip_device(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len_q: usize,
    seq_len_kv: usize,
    config: &HipAttentionConfig,
) -> Result<()> {
    use super::hip_ffi;

    let nh = config.num_heads;
    let d = config.head_dim;
    let q_bytes = nh * seq_len_q * d * std::mem::size_of::<f32>();
    let kv_bytes = nh * seq_len_kv * d * std::mem::size_of::<f32>();

    unsafe {
        let stream = hip_ffi::current_stream()?;

        let d_q = hip_ffi::device_malloc(q_bytes)?;
        let d_k = hip_ffi::device_malloc(kv_bytes)?;
        let d_v = hip_ffi::device_malloc(kv_bytes)?;
        let d_out = hip_ffi::device_malloc(q_bytes)?;

        hip_ffi::memcpy_h2d(d_q, q.as_ptr().cast(), q_bytes, stream)?;
        hip_ffi::memcpy_h2d(d_k, k.as_ptr().cast(), kv_bytes, stream)?;
        hip_ffi::memcpy_h2d(d_v, v.as_ptr().cast(), kv_bytes, stream)?;

        let threads = 256u32;
        let tile_q = if seq_len_q <= 32 { seq_len_q as u32 } else { 32 };
        let blocks_x = (seq_len_q as u32).div_ceil(tile_q);
        let blocks_y = nh as u32;

        hip_ffi::launch_fused_attention(
            d_q.cast(),
            d_k.cast(),
            d_v.cast(),
            d_out.cast(),
            nh as u32,
            d as u32,
            seq_len_q as u32,
            seq_len_kv as u32,
            config.scale,
            config.causal,
            threads,
            blocks_x,
            blocks_y,
            stream,
        )?;

        hip_ffi::memcpy_d2h(output.as_mut_ptr().cast(), d_out, q_bytes, stream)?;
        hip_ffi::stream_synchronize(stream)?;

        hip_ffi::device_free(d_q)?;
        hip_ffi::device_free(d_k)?;
        hip_ffi::device_free(d_v)?;
        hip_ffi::device_free(d_out)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_scale_matches_head_dim() {
        let cfg = HipAttentionConfig::new(8, 64);
        let expected = 1.0 / (64.0f32).sqrt();
        assert!((cfg.scale - expected).abs() < 1e-6);
        assert!(cfg.causal);
    }

    #[test]
    fn config_with_causal() {
        let cfg = HipAttentionConfig::new(1, 4).with_causal(false);
        assert!(!cfg.causal);
    }

    #[test]
    fn attention_cpu_single_head_no_causal() {
        let cfg = HipAttentionConfig::new(1, 2).with_causal(false);
        let q = [1.0, 0.0, 0.0, 1.0f32];
        let k = [1.0, 0.0, 0.0, 1.0f32];
        let v = [1.0, 2.0, 3.0, 4.0f32];
        let mut output = [0.0f32; 4];

        fused_attention_cpu(&q, &k, &v, &mut output, 2, 2, &cfg).unwrap();

        for &v in &output {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn attention_cpu_causal_mask() {
        let cfg = HipAttentionConfig::new(1, 2);
        let q = [1.0, 0.0, 0.0, 1.0f32];
        let k = [1.0, 0.0, 0.0, 1.0f32];
        let v = [10.0, 20.0, 30.0, 40.0f32];
        let mut output = [0.0f32; 4];

        fused_attention_cpu(&q, &k, &v, &mut output, 2, 2, &cfg).unwrap();

        // Row 0 can only see V[0] = [10, 20]
        assert!((output[0] - 10.0).abs() < 1e-4, "causal row 0 col 0: {}", output[0]);
        assert!((output[1] - 20.0).abs() < 1e-4, "causal row 0 col 1: {}", output[1]);
    }

    #[test]
    fn attention_cpu_rejects_zero_heads() {
        let cfg = HipAttentionConfig::new(0, 64);
        let q = [0.0f32; 64];
        let mut output = [0.0f32; 64];
        assert!(fused_attention_cpu(&q, &q, &q, &mut output, 1, 1, &cfg).is_err());
    }

    #[test]
    fn attention_cpu_rejects_zero_seq() {
        let cfg = HipAttentionConfig::new(1, 4);
        let q = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        assert!(fused_attention_cpu(&q, &q, &q, &mut out, 0, 1, &cfg).is_err());
    }

    #[test]
    fn attention_hip_falls_back_to_cpu() {
        let cfg = HipAttentionConfig::new(1, 2).with_causal(false);
        let q = [1.0, 0.0f32];
        let k = [1.0, 0.0f32];
        let v = [5.0, 6.0f32];
        let mut output = [0.0f32; 2];

        fused_attention_hip(&q, &k, &v, &mut output, 1, 1, &cfg).unwrap();

        // Single KV position, output = V[0]
        assert!((output[0] - 5.0).abs() < 1e-4);
        assert!((output[1] - 6.0).abs() < 1e-4);
    }

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn attention_hip_device_dispatch() {
        let cfg = HipAttentionConfig::new(8, 64);
        let size_q = 8 * 32 * 64;
        let size_kv = 8 * 32 * 64;
        let q = vec![0.01f32; size_q];
        let k = vec![0.01f32; size_kv];
        let v = vec![0.01f32; size_kv];
        let mut output = vec![0.0f32; size_q];

        fused_attention_hip(&q, &k, &v, &mut output, 32, 32, &cfg).unwrap();

        for &val in &output {
            assert!(val.is_finite(), "HIP attention output not finite");
        }
    }
}
