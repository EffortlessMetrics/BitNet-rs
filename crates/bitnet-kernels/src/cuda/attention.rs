//! Scaled dot-product attention CUDA kernel.
//!
//! # Kernel strategy
//!
//! Implements FlashAttention-style tiled attention to keep the `O(n²)` score
//! matrix in on-chip SRAM rather than HBM:
//!
//! 1. **Q tile** — each thread-block loads a `tile_q × head_dim` slice of Q
//!    into shared memory.
//! 2. **K/V streaming** — K and V blocks are streamed in `tile_kv`-sized
//!    chunks.  For each chunk the partial `softmax(QKᵀ)V` is accumulated
//!    using the online softmax trick (numerically stable running max + sum).
//! 3. **Causal mask** — upper-triangular positions are masked to `-inf` before
//!    the softmax reduction, supporting autoregressive decoding.
//! 4. **Output write-back** — the final `O[tile_q, head_dim]` tile is written
//!    to global memory in a single coalesced store.
//!
//! Target: ≥ 50 % SM occupancy on Ampere (SM 8.0) with 48 KB shared memory
//! per block.  FP16 accumulation is used when `head_dim ≤ 128` and the device
//! supports native FP16 (`compute_capability ≥ 6.0`).

use bitnet_common::{KernelError, Result};

/// Launch configuration for the scaled dot-product attention kernel.
#[derive(Debug, Clone)]
pub struct AttentionKernelConfig {
    /// Tile size along the query (sequence-out) dimension.
    pub tile_q: u32,
    /// Tile size along the key/value (sequence-in) dimension.
    pub tile_kv: u32,
    /// Number of attention heads processed in parallel.
    pub n_heads: usize,
    /// Per-head embedding dimension (typically 64 or 128).
    pub head_dim: usize,
    /// Sequence length of the query tensor.
    pub seq_len_q: usize,
    /// Sequence length of the key/value tensors (may differ during decode).
    pub seq_len_kv: usize,
    /// Threads per block (must be ≥ `tile_q * tile_kv`).
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory for Q/K tiles and running softmax state.
    pub shared_mem_bytes: u32,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
}

impl AttentionKernelConfig {
    /// Create a configuration for the given attention shape.
    pub fn for_shape(
        n_heads: usize,
        head_dim: usize,
        seq_len_q: usize,
        seq_len_kv: usize,
        causal: bool,
    ) -> Result<Self> {
        if n_heads == 0 || head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "attention n_heads={n_heads} and head_dim={head_dim} must be non-zero"
                ),
            }
            .into());
        }
        if seq_len_q == 0 || seq_len_kv == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "attention seq lengths must be non-zero: q={seq_len_q}, kv={seq_len_kv}"
                ),
            }
            .into());
        }

        let tile_q: u32 = if seq_len_q <= 32 { seq_len_q as u32 } else { 32 };
        let tile_kv: u32 = if seq_len_kv <= 64 { seq_len_kv as u32 } else { 64 };
        let threads_per_block = 256u32;

        // Shared memory: Q tile + K tile + running softmax state (max + sum per row)
        let q_tile_bytes = (tile_q as usize) * head_dim * 4; // FP32
        let k_tile_bytes = (tile_kv as usize) * head_dim * 4;
        let softmax_state_bytes = (tile_q as usize) * 2 * 4; // max + sum per row
        let shared_mem_bytes = (q_tile_bytes + k_tile_bytes + softmax_state_bytes) as u32;

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            tile_q,
            tile_kv,
            n_heads,
            head_dim,
            seq_len_q,
            seq_len_kv,
            threads_per_block,
            shared_mem_bytes,
            causal,
            scale,
        })
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, grid_z)`.
    ///
    /// `grid_x` covers query tiles, `grid_y` covers heads, `grid_z` = 1.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.seq_len_q as u32).div_ceil(self.tile_q);
        let grid_y = self.n_heads as u32;
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Launch stub for the scaled dot-product attention kernel.
///
/// # Arguments
///
/// * `q`      — Query tensor `[n_heads, seq_len_q, head_dim]` (FP32)
/// * `k`      — Key tensor   `[n_heads, seq_len_kv, head_dim]` (FP32)
/// * `v`      — Value tensor  `[n_heads, seq_len_kv, head_dim]` (FP32)
/// * `output` — Output buffer `[n_heads, seq_len_q, head_dim]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_attention(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    config: &AttentionKernelConfig,
) -> Result<()> {
    log::debug!(
        "Attention stub: heads={}, head_dim={}, seq_q={}, seq_kv={}, causal={}, grid={:?}",
        config.n_heads,
        config.head_dim,
        config.seq_len_q,
        config.seq_len_kv,
        config.causal,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Attention CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_for_shape() {
        let cfg = AttentionKernelConfig::for_shape(32, 128, 1, 512, true).unwrap();
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert!(cfg.causal);
        assert!((cfg.scale - 1.0 / (128.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_attention_config_grid_dim() {
        let cfg = AttentionKernelConfig::for_shape(8, 64, 100, 100, false).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // ceil(100/32)
        assert_eq!(gy, 8); // n_heads
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_attention_config_rejects_zero_heads() {
        assert!(AttentionKernelConfig::for_shape(0, 128, 1, 512, true).is_err());
    }

    #[test]
    fn test_attention_config_rejects_zero_seq() {
        assert!(AttentionKernelConfig::for_shape(8, 64, 0, 512, true).is_err());
        assert!(AttentionKernelConfig::for_shape(8, 64, 1, 0, true).is_err());
    }

    #[test]
    fn test_attention_config_small_seq() {
        let cfg = AttentionKernelConfig::for_shape(1, 64, 4, 4, false).unwrap();
        assert_eq!(cfg.tile_q, 4); // small seq → tile = seq
        assert_eq!(cfg.tile_kv, 4);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_attention_launch() {
        let cfg = AttentionKernelConfig::for_shape(8, 64, 32, 32, true).unwrap();
        let size_q = 8 * 32 * 64;
        let size_kv = 8 * 32 * 64;
        let q = vec![0.0f32; size_q];
        let k = vec![0.0f32; size_kv];
        let v = vec![0.0f32; size_kv];
        let mut output = vec![0.0f32; size_q];
        let result = launch_attention(&q, &k, &v, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA attention launch failed: {result:?}");
    }
}
