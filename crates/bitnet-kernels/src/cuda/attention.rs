//! Multi-head attention CUDA kernel dispatch with KV cache support.
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
//! # Multi-head attention with KV cache
//!
//! Two dispatch paths are provided for the autoregressive pipeline:
//!
//! - **Prefill** ([`launch_mha_prefill`]): Processes the full prompt in a single
//!   pass, populating the KV cache.  Uses larger `tile_q` for throughput.
//! - **Decode** ([`launch_mha_decode`]): Appends one new KV pair to the cache
//!   and computes attention for the single new query token.  Uses `tile_q = 1`
//!   and streams the entire cached KV range.
//!
//! Both paths support grouped-query attention (GQA) where `n_kv_heads < n_heads`.
//!
//! Target: ≥ 50 % SM occupancy on Ampere (SM 8.0) with 48 KB shared memory
//! per block.  FP16 accumulation is used when `head_dim ≤ 128` and the device
//! supports native FP16 (`compute_capability ≥ 6.0`).

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Supported head dimensions
// ---------------------------------------------------------------------------

/// Supported per-head embedding dimensions.
///
/// Flash-attention tile sizes and shared-memory budgets are tuned per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadDim {
    /// 64-element heads (common in smaller models).
    D64 = 64,
    /// 128-element heads (standard for LLaMA-class models).
    D128 = 128,
}

impl HeadDim {
    /// Try to convert a raw `usize` to a supported [`HeadDim`].
    pub fn from_size(dim: usize) -> Result<Self> {
        match dim {
            64 => Ok(Self::D64),
            128 => Ok(Self::D128),
            other => Err(KernelError::InvalidArguments {
                reason: format!("unsupported head_dim={other}; expected 64 or 128"),
            }
            .into()),
        }
    }

    /// Return the numeric dimension.
    pub fn size(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// KV cache configuration
// ---------------------------------------------------------------------------

/// Describes the layout of a pre-allocated KV cache buffer.
///
/// The cache stores key and value tensors for all layers as contiguous FP32
/// slices laid out as `[n_kv_heads, max_seq_len, head_dim]`.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Number of KV head groups (may differ from query heads in GQA).
    pub n_kv_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: HeadDim,
    /// Current number of valid tokens already written into the cache
    /// (i.e. the write cursor).  Must be `< max_seq_len`.
    pub current_pos: usize,
}

impl KvCacheConfig {
    /// Create a new KV cache configuration.
    pub fn new(
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: HeadDim,
        current_pos: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KV cache max_seq_len must be non-zero".into(),
            }
            .into());
        }
        if n_kv_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "KV cache n_kv_heads must be non-zero".into(),
            }
            .into());
        }
        if current_pos > max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "KV cache current_pos={current_pos} exceeds \
                     max_seq_len={max_seq_len}"
                ),
            }
            .into());
        }
        Ok(Self { max_seq_len, n_kv_heads, head_dim, current_pos })
    }

    /// Number of elements in one head's cache slice: `max_seq_len * head_dim`.
    pub fn head_stride(&self) -> usize {
        self.max_seq_len * self.head_dim.size()
    }

    /// Total elements in the full cache: `n_kv_heads * head_stride()`.
    pub fn total_elements(&self) -> usize {
        self.n_kv_heads * self.head_stride()
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention kernel config
// ---------------------------------------------------------------------------

/// Launch configuration for multi-head attention with optional KV cache.
///
/// Supports both prefill (full-sequence) and decode (single-token) modes.
/// Grouped-query attention (GQA) is enabled when `n_kv_heads < n_heads`.
#[derive(Debug, Clone)]
pub struct MhaConfig {
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value head groups (equals `n_heads` for standard MHA).
    pub n_kv_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: HeadDim,
    /// Sequence length of the query tensor (1 during decode).
    pub seq_len_q: usize,
    /// Total KV sequence length visible to the query (cache + new tokens).
    pub seq_len_kv: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
    /// Flash-attention Q tile size.
    pub tile_q: u32,
    /// Flash-attention KV tile size.
    pub tile_kv: u32,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory per block.
    pub shared_mem_bytes: u32,
    /// GQA repetition factor: how many Q heads share each KV head.
    pub gqa_group_size: usize,
}

/// Maximum shared memory budget per block (48 KB on Ampere).
const MAX_SMEM_BYTES: u32 = 48 * 1024;

impl MhaConfig {
    /// Build a configuration for the given attention shape.
    ///
    /// `n_kv_heads` must evenly divide `n_heads` (GQA constraint).
    pub fn new(
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: HeadDim,
        seq_len_q: usize,
        seq_len_kv: usize,
        causal: bool,
    ) -> Result<Self> {
        if n_heads == 0 || n_kv_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("n_heads={n_heads} and n_kv_heads={n_kv_heads} must be non-zero"),
            }
            .into());
        }
        if n_heads % n_kv_heads != 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads} (GQA)"
                ),
            }
            .into());
        }
        if seq_len_q == 0 || seq_len_kv == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "sequence lengths must be non-zero: q={seq_len_q}, kv={seq_len_kv}"
                ),
            }
            .into());
        }

        let d = head_dim.size();
        let gqa_group_size = n_heads / n_kv_heads;

        // Tile sizes tuned per head dimension
        let tile_q = match head_dim {
            HeadDim::D64 => (seq_len_q as u32).min(64),
            HeadDim::D128 => (seq_len_q as u32).min(32),
        };
        let tile_kv = match head_dim {
            HeadDim::D64 => (seq_len_kv as u32).min(128),
            HeadDim::D128 => (seq_len_kv as u32).min(64),
        };
        let threads_per_block = 256u32;

        // Shared memory: Q tile + K tile + V tile + softmax state (max, sum per row)
        let q_bytes = (tile_q as usize) * d * 4;
        let k_bytes = (tile_kv as usize) * d * 4;
        let v_bytes = (tile_kv as usize) * d * 4;
        let softmax_bytes = (tile_q as usize) * 2 * 4;
        let shared_mem_bytes = (q_bytes + k_bytes + v_bytes + softmax_bytes) as u32;

        if shared_mem_bytes > MAX_SMEM_BYTES {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MHA shared memory {shared_mem_bytes} B exceeds \
                     {MAX_SMEM_BYTES} B limit; reduce tile sizes or head_dim"
                ),
            }
            .into());
        }

        let scale = 1.0 / (d as f32).sqrt();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len_q,
            seq_len_kv,
            causal,
            scale,
            tile_q,
            tile_kv,
            threads_per_block,
            shared_mem_bytes,
            gqa_group_size,
        })
    }

    /// Convenience constructor for standard MHA (`n_kv_heads == n_heads`).
    pub fn standard(
        n_heads: usize,
        head_dim: HeadDim,
        seq_len_q: usize,
        seq_len_kv: usize,
        causal: bool,
    ) -> Result<Self> {
        Self::new(n_heads, n_heads, head_dim, seq_len_q, seq_len_kv, causal)
    }

    /// Override the softmax scale (defaults to `1/sqrt(head_dim)`).
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// CUDA grid: `(q_tiles, n_heads, batch=1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.seq_len_q as u32).div_ceil(self.tile_q);
        let grid_y = self.n_heads as u32;
        (grid_x, grid_y, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible single-shot config (original API)
// ---------------------------------------------------------------------------

/// Launch configuration for the scaled dot-product attention kernel.
///
/// This is the original single-shot API without KV cache support.
/// For new code prefer [`MhaConfig`] with [`launch_mha_prefill`] /
/// [`launch_mha_decode`].
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

// ---------------------------------------------------------------------------
// Launch functions
// ---------------------------------------------------------------------------

/// Launch stub for the scaled dot-product attention kernel (original API).
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

/// Validate buffer lengths for an MHA launch.
fn validate_mha_buffers(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &[f32],
    config: &MhaConfig,
) -> Result<()> {
    let d = config.head_dim.size();
    let expected_q = config.n_heads * config.seq_len_q * d;
    let expected_kv = config.n_kv_heads * config.seq_len_kv * d;
    let expected_out = config.n_heads * config.seq_len_q * d;

    if q.len() < expected_q {
        return Err(KernelError::InvalidArguments {
            reason: format!("Q buffer too small: need {expected_q}, got {}", q.len()),
        }
        .into());
    }
    if k.len() < expected_kv {
        return Err(KernelError::InvalidArguments {
            reason: format!("K buffer too small: need {expected_kv}, got {}", k.len()),
        }
        .into());
    }
    if v.len() < expected_kv {
        return Err(KernelError::InvalidArguments {
            reason: format!("V buffer too small: need {expected_kv}, got {}", v.len()),
        }
        .into());
    }
    if output.len() < expected_out {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: need {expected_out}, got {}", output.len()),
        }
        .into());
    }
    Ok(())
}

/// Prefill-phase multi-head attention dispatch.
///
/// Processes the full prompt sequence in one pass, populating the KV cache
/// (if provided) for subsequent decode steps.
///
/// # Tensor layouts (FP32, row-major)
///
/// * `q`      — `[n_heads,    seq_len_q,  head_dim]`
/// * `k`      — `[n_kv_heads, seq_len_kv, head_dim]`
/// * `v`      — `[n_kv_heads, seq_len_kv, head_dim]`
/// * `output` — `[n_heads,    seq_len_q,  head_dim]`  (written)
///
/// When `kv_cache` is `Some`, K and V are also written into the cache at
/// positions `[0 .. seq_len_kv)`.
///
/// # Errors
///
/// Returns `KernelError::GpuError` (scaffold) or `KernelError::InvalidArguments`
/// on shape / buffer-size mismatch.
pub fn launch_mha_prefill(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    config: &MhaConfig,
    kv_cache: Option<&mut MhaKvCache>,
) -> Result<()> {
    validate_mha_buffers(q, k, v, output, config)?;

    if let Some(cache) = &kv_cache {
        if config.seq_len_kv > cache.config.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "prefill seq_len_kv={} exceeds cache max_seq_len={}",
                    config.seq_len_kv, cache.config.max_seq_len,
                ),
            }
            .into());
        }
        if cache.config.n_kv_heads != config.n_kv_heads {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "cache n_kv_heads={} != config n_kv_heads={}",
                    cache.config.n_kv_heads, config.n_kv_heads,
                ),
            }
            .into());
        }
    }

    log::debug!(
        "MHA prefill: heads={}/{}, head_dim={}, seq_q={}, seq_kv={}, \
         causal={}, gqa_group={}, grid={:?}, smem={} B, cache={}",
        config.n_heads,
        config.n_kv_heads,
        config.head_dim.size(),
        config.seq_len_q,
        config.seq_len_kv,
        config.causal,
        config.gqa_group_size,
        config.grid_dim(),
        config.shared_mem_bytes,
        kv_cache.is_some(),
    );

    Err(KernelError::GpuError {
        reason: "MHA prefill CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Decode-phase multi-head attention dispatch (single new token).
///
/// Appends one new KV pair at `kv_cache.current_pos` and runs attention over
/// the full cached range `[0 .. current_pos + 1)`.
///
/// # Tensor layouts (FP32, row-major)
///
/// * `q`       — `[n_heads,    1, head_dim]`
/// * `k_new`   — `[n_kv_heads, 1, head_dim]`
/// * `v_new`   — `[n_kv_heads, 1, head_dim]`
/// * `output`  — `[n_heads,    1, head_dim]`  (written)
///
/// # Errors
///
/// Returns `KernelError::GpuError` (scaffold) or `KernelError::InvalidArguments`
/// on shape / cache mismatch.
pub fn launch_mha_decode(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    output: &mut [f32],
    config: &MhaConfig,
    kv_cache: &mut MhaKvCache,
) -> Result<()> {
    if config.seq_len_q != 1 {
        return Err(KernelError::InvalidArguments {
            reason: format!("decode expects seq_len_q=1, got {}", config.seq_len_q),
        }
        .into());
    }
    if kv_cache.config.current_pos >= kv_cache.config.max_seq_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "KV cache full: current_pos={} == max_seq_len={}",
                kv_cache.config.current_pos, kv_cache.config.max_seq_len,
            ),
        }
        .into());
    }
    if kv_cache.config.n_kv_heads != config.n_kv_heads {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "cache n_kv_heads={} != config n_kv_heads={}",
                kv_cache.config.n_kv_heads, config.n_kv_heads,
            ),
        }
        .into());
    }

    let d = config.head_dim.size();
    let expected_q = config.n_heads * d;
    let expected_kv_new = config.n_kv_heads * d;
    let expected_out = config.n_heads * d;

    if q.len() < expected_q {
        return Err(KernelError::InvalidArguments {
            reason: format!("decode Q buffer too small: need {expected_q}, got {}", q.len()),
        }
        .into());
    }
    if k_new.len() < expected_kv_new {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "decode K_new buffer too small: need {expected_kv_new}, got {}",
                k_new.len()
            ),
        }
        .into());
    }
    if v_new.len() < expected_kv_new {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "decode V_new buffer too small: need {expected_kv_new}, got {}",
                v_new.len()
            ),
        }
        .into());
    }
    if output.len() < expected_out {
        return Err(KernelError::InvalidArguments {
            reason: format!("decode output too small: need {expected_out}, got {}", output.len()),
        }
        .into());
    }

    let total_kv_len = kv_cache.config.current_pos + 1;

    log::debug!(
        "MHA decode: heads={}/{}, head_dim={}, kv_len={}, \
         causal={}, gqa_group={}, grid={:?}, smem={} B",
        config.n_heads,
        config.n_kv_heads,
        config.head_dim.size(),
        total_kv_len,
        config.causal,
        config.gqa_group_size,
        config.grid_dim(),
        config.shared_mem_bytes,
    );

    Err(KernelError::GpuError {
        reason: "MHA decode CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// KV cache handle
// ---------------------------------------------------------------------------

/// Opaque handle for a KV cache buffer pair.
///
/// Holds host-side (FP32) buffers for keys and values.  In a real CUDA
/// implementation these would be device pointers; the current scaffold keeps
/// host buffers so that config validation and unit tests work without a GPU.
#[derive(Debug)]
pub struct MhaKvCache {
    /// Configuration describing layout and current position.
    pub config: KvCacheConfig,
    /// Key cache `[n_kv_heads, max_seq_len, head_dim]`.
    pub k_cache: Vec<f32>,
    /// Value cache `[n_kv_heads, max_seq_len, head_dim]`.
    pub v_cache: Vec<f32>,
}

impl MhaKvCache {
    /// Allocate a zeroed KV cache.
    pub fn new(config: KvCacheConfig) -> Self {
        let n = config.total_elements();
        Self { config, k_cache: vec![0.0; n], v_cache: vec![0.0; n] }
    }

    /// Reset the cache position to zero (reuse the allocation).
    pub fn reset(&mut self) {
        self.config.current_pos = 0;
    }

    /// Return the current sequence length stored in the cache.
    pub fn cached_len(&self) -> usize {
        self.config.current_pos
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === HeadDim ===========================================================

    #[test]
    fn test_head_dim_from_size_valid() {
        assert_eq!(HeadDim::from_size(64).unwrap(), HeadDim::D64);
        assert_eq!(HeadDim::from_size(128).unwrap(), HeadDim::D128);
    }

    #[test]
    fn test_head_dim_from_size_invalid() {
        assert!(HeadDim::from_size(0).is_err());
        assert!(HeadDim::from_size(96).is_err());
        assert!(HeadDim::from_size(256).is_err());
    }

    #[test]
    fn test_head_dim_size_roundtrip() {
        assert_eq!(HeadDim::D64.size(), 64);
        assert_eq!(HeadDim::D128.size(), 128);
    }

    // === KvCacheConfig =====================================================

    #[test]
    fn test_kv_cache_config_valid() {
        let cfg = KvCacheConfig::new(2048, 8, HeadDim::D128, 0).unwrap();
        assert_eq!(cfg.head_stride(), 2048 * 128);
        assert_eq!(cfg.total_elements(), 8 * 2048 * 128);
    }

    #[test]
    fn test_kv_cache_config_rejects_zero_seq() {
        assert!(KvCacheConfig::new(0, 8, HeadDim::D64, 0).is_err());
    }

    #[test]
    fn test_kv_cache_config_rejects_zero_heads() {
        assert!(KvCacheConfig::new(2048, 0, HeadDim::D64, 0).is_err());
    }

    #[test]
    fn test_kv_cache_config_rejects_pos_overflow() {
        assert!(KvCacheConfig::new(512, 8, HeadDim::D64, 513).is_err());
    }

    #[test]
    fn test_kv_cache_config_allows_pos_equal_max() {
        // current_pos == max_seq_len means cache is full but valid
        let cfg = KvCacheConfig::new(512, 8, HeadDim::D64, 512).unwrap();
        assert_eq!(cfg.current_pos, 512);
    }

    // === MhaConfig =========================================================

    #[test]
    fn test_mha_config_standard() {
        let cfg = MhaConfig::standard(32, HeadDim::D128, 1, 512, true).unwrap();
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 32);
        assert_eq!(cfg.gqa_group_size, 1);
        assert!(cfg.causal);
        assert!((cfg.scale - 1.0 / (128.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_mha_config_gqa() {
        let cfg = MhaConfig::new(32, 8, HeadDim::D128, 1, 512, true).unwrap();
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.gqa_group_size, 4);
    }

    #[test]
    fn test_mha_config_rejects_bad_gqa() {
        // 32 heads with 6 kv_heads → not evenly divisible
        assert!(MhaConfig::new(32, 6, HeadDim::D128, 1, 512, true).is_err());
    }

    #[test]
    fn test_mha_config_rejects_zero_heads() {
        assert!(MhaConfig::new(0, 0, HeadDim::D64, 1, 512, true).is_err());
    }

    #[test]
    fn test_mha_config_rejects_zero_seq() {
        assert!(MhaConfig::new(8, 8, HeadDim::D64, 0, 512, true).is_err());
        assert!(MhaConfig::new(8, 8, HeadDim::D64, 1, 0, true).is_err());
    }

    #[test]
    fn test_mha_config_grid_dim_prefill() {
        let cfg = MhaConfig::standard(8, HeadDim::D64, 100, 100, false).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        // D64 → tile_q=64, ceil(100/64)=2
        assert_eq!(gx, 2);
        assert_eq!(gy, 8);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_mha_config_grid_dim_decode() {
        let cfg = MhaConfig::standard(32, HeadDim::D128, 1, 512, true).unwrap();
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 1); // single query token
        assert_eq!(gy, 32);
    }

    #[test]
    fn test_mha_config_small_seq_tiles() {
        let cfg = MhaConfig::standard(1, HeadDim::D64, 4, 8, false).unwrap();
        assert_eq!(cfg.tile_q, 4);
        assert_eq!(cfg.tile_kv, 8);
    }

    #[test]
    fn test_mha_config_with_scale() {
        let cfg = MhaConfig::standard(8, HeadDim::D64, 1, 1, false).unwrap().with_scale(0.5);
        assert!((cfg.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mha_config_d64_vs_d128_tiles() {
        let d64 = MhaConfig::standard(8, HeadDim::D64, 256, 256, true).unwrap();
        let d128 = MhaConfig::standard(8, HeadDim::D128, 256, 256, true).unwrap();
        // D64 gets larger tiles
        assert!(d64.tile_q >= d128.tile_q);
        assert!(d64.tile_kv >= d128.tile_kv);
    }

    // === MhaKvCache ========================================================

    #[test]
    fn test_mha_kv_cache_alloc() {
        let cfg = KvCacheConfig::new(128, 4, HeadDim::D64, 0).unwrap();
        let cache = MhaKvCache::new(cfg);
        assert_eq!(cache.k_cache.len(), 4 * 128 * 64);
        assert_eq!(cache.v_cache.len(), 4 * 128 * 64);
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn test_mha_kv_cache_reset() {
        let cfg = KvCacheConfig::new(128, 4, HeadDim::D64, 42).unwrap();
        let mut cache = MhaKvCache::new(cfg);
        assert_eq!(cache.cached_len(), 42);
        cache.reset();
        assert_eq!(cache.cached_len(), 0);
    }

    // === Buffer validation =================================================

    #[test]
    fn test_validate_mha_buffers_ok() {
        let cfg = MhaConfig::standard(2, HeadDim::D64, 4, 4, false).unwrap();
        let q = vec![0.0f32; 2 * 4 * 64];
        let k = vec![0.0f32; 2 * 4 * 64];
        let v = vec![0.0f32; 2 * 4 * 64];
        let output = vec![0.0f32; 2 * 4 * 64];
        assert!(validate_mha_buffers(&q, &k, &v, &output, &cfg).is_ok());
    }

    #[test]
    fn test_validate_mha_buffers_q_too_small() {
        let cfg = MhaConfig::standard(2, HeadDim::D64, 4, 4, false).unwrap();
        let q = vec![0.0f32; 1]; // too small
        let k = vec![0.0f32; 2 * 4 * 64];
        let v = vec![0.0f32; 2 * 4 * 64];
        let output = vec![0.0f32; 2 * 4 * 64];
        assert!(validate_mha_buffers(&q, &k, &v, &output, &cfg).is_err());
    }

    #[test]
    fn test_validate_mha_buffers_gqa() {
        // 8 query heads, 2 kv heads → kv buffers are smaller
        let cfg = MhaConfig::new(8, 2, HeadDim::D64, 1, 16, true).unwrap();
        let q = vec![0.0f32; 8 * 1 * 64];
        let k = vec![0.0f32; 2 * 16 * 64];
        let v = vec![0.0f32; 2 * 16 * 64];
        let output = vec![0.0f32; 8 * 1 * 64];
        assert!(validate_mha_buffers(&q, &k, &v, &output, &cfg).is_ok());
    }

    // === Prefill launch ====================================================

    #[test]
    fn test_prefill_buffer_too_small() {
        let cfg = MhaConfig::standard(2, HeadDim::D64, 4, 4, false).unwrap();
        let q = vec![0.0f32; 1]; // too small
        let k = vec![0.0f32; 2 * 4 * 64];
        let v = vec![0.0f32; 2 * 4 * 64];
        let mut output = vec![0.0f32; 2 * 4 * 64];
        let result = launch_mha_prefill(&q, &k, &v, &mut output, &cfg, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefill_cache_overflow() {
        let cfg = MhaConfig::standard(2, HeadDim::D64, 4, 128, false).unwrap();
        let kv_cfg = KvCacheConfig::new(64, 2, HeadDim::D64, 0).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 2 * 4 * 64];
        let k = vec![0.0f32; 2 * 128 * 64];
        let v = vec![0.0f32; 2 * 128 * 64];
        let mut output = vec![0.0f32; 2 * 4 * 64];
        let result = launch_mha_prefill(&q, &k, &v, &mut output, &cfg, Some(&mut cache));
        assert!(result.is_err());
    }

    #[test]
    fn test_prefill_cache_head_mismatch() {
        let cfg = MhaConfig::standard(4, HeadDim::D64, 4, 4, false).unwrap();
        // cache has 2 kv heads, config has 4
        let kv_cfg = KvCacheConfig::new(128, 2, HeadDim::D64, 0).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 4 * 4 * 64];
        let k = vec![0.0f32; 4 * 4 * 64];
        let v = vec![0.0f32; 4 * 4 * 64];
        let mut output = vec![0.0f32; 4 * 4 * 64];
        let result = launch_mha_prefill(&q, &k, &v, &mut output, &cfg, Some(&mut cache));
        assert!(result.is_err());
    }

    // === Decode launch =====================================================

    #[test]
    fn test_decode_rejects_non_unit_seq() {
        let cfg = MhaConfig::standard(8, HeadDim::D64, 4, 4, true).unwrap();
        let kv_cfg = KvCacheConfig::new(128, 8, HeadDim::D64, 3).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 8 * 4 * 64];
        let k = vec![0.0f32; 8 * 64];
        let v = vec![0.0f32; 8 * 64];
        let mut output = vec![0.0f32; 8 * 4 * 64];
        let result = launch_mha_decode(&q, &k, &v, &mut output, &cfg, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_rejects_full_cache() {
        let cfg = MhaConfig::standard(8, HeadDim::D64, 1, 128, true).unwrap();
        // cache is completely full
        let kv_cfg = KvCacheConfig::new(128, 8, HeadDim::D64, 128).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 8 * 64];
        let k = vec![0.0f32; 8 * 64];
        let v = vec![0.0f32; 8 * 64];
        let mut output = vec![0.0f32; 8 * 64];
        let result = launch_mha_decode(&q, &k, &v, &mut output, &cfg, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_rejects_small_q_buffer() {
        let cfg = MhaConfig::standard(8, HeadDim::D128, 1, 64, true).unwrap();
        let kv_cfg = KvCacheConfig::new(128, 8, HeadDim::D128, 63).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 1]; // too small
        let k = vec![0.0f32; 8 * 128];
        let v = vec![0.0f32; 8 * 128];
        let mut output = vec![0.0f32; 8 * 128];
        let result = launch_mha_decode(&q, &k, &v, &mut output, &cfg, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_kv_head_mismatch() {
        let cfg = MhaConfig::new(8, 4, HeadDim::D64, 1, 64, true).unwrap();
        // cache has 8 kv heads, config has 4
        let kv_cfg = KvCacheConfig::new(128, 8, HeadDim::D64, 10).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 8 * 64];
        let k = vec![0.0f32; 4 * 64];
        let v = vec![0.0f32; 4 * 64];
        let mut output = vec![0.0f32; 8 * 64];
        let result = launch_mha_decode(&q, &k, &v, &mut output, &cfg, &mut cache);
        assert!(result.is_err());
    }

    // === Backward-compat: AttentionKernelConfig ============================

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

    // === GPU launch tests (ignored: require CUDA runtime) ==================

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

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_mha_prefill_launch() {
        let cfg = MhaConfig::standard(8, HeadDim::D128, 32, 32, true).unwrap();
        let q = vec![0.0f32; 8 * 32 * 128];
        let k = vec![0.0f32; 8 * 32 * 128];
        let v = vec![0.0f32; 8 * 32 * 128];
        let mut output = vec![0.0f32; 8 * 32 * 128];
        let result = launch_mha_prefill(&q, &k, &v, &mut output, &cfg, None);
        assert!(result.is_ok(), "MHA prefill launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_mha_decode_launch() {
        let cfg = MhaConfig::standard(8, HeadDim::D64, 1, 64, true).unwrap();
        let kv_cfg = KvCacheConfig::new(128, 8, HeadDim::D64, 63).unwrap();
        let mut cache = MhaKvCache::new(kv_cfg);
        let q = vec![0.0f32; 8 * 64];
        let k = vec![0.0f32; 8 * 64];
        let v = vec![0.0f32; 8 * 64];
        let mut output = vec![0.0f32; 8 * 64];
        let result = launch_mha_decode(&q, &k, &v, &mut output, &cfg, &mut cache);
        assert!(result.is_ok(), "MHA decode launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_mha_gqa_launch() {
        // 32 query heads, 8 KV heads (4x GQA)
        let cfg = MhaConfig::new(32, 8, HeadDim::D128, 16, 64, true).unwrap();
        let q = vec![0.0f32; 32 * 16 * 128];
        let k = vec![0.0f32; 8 * 64 * 128];
        let v = vec![0.0f32; 8 * 64 * 128];
        let mut output = vec![0.0f32; 32 * 16 * 128];
        let result = launch_mha_prefill(&q, &k, &v, &mut output, &cfg, None);
        assert!(result.is_ok(), "MHA GQA launch failed: {result:?}");
    }
}
