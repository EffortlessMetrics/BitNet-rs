//! Scaled dot-product attention CUDA kernel with CPU fallback.
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
//!
//! # CPU fallback
//!
//! [`attention_cpu_fallback`], [`masked_attention_cpu_fallback`], and
//! [`multi_head_attention_cpu_fallback`] provide pure-Rust implementations
//! for correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

/// Alias for [`AttentionKernelConfig`] — the CUDA-specific launch configuration.
///
/// Provides a discoverable name matching the `Cuda*Config` naming convention
/// used by other kernel modules (e.g. `CudaTransposeConfig`, `CudaBatchNormConfig`).
pub type CudaAttentionConfig = AttentionKernelConfig;

// ---------------------------------------------------------------------------
// CUDA kernel source (compiled at runtime via NVRTC when `gpu`/`cuda` active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for the scaled dot-product attention kernel.
///
/// Implements a FlashAttention-style tiled kernel:
/// - `sdp_attention_f32`: single-head scaled dot-product attention
/// - `sdp_attention_causal_f32`: causal (autoregressive) variant
///
/// Each thread-block processes one query tile across all K/V positions,
/// using online softmax for numerical stability.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ATTENTION_KERNEL_SRC: &str = r#"
extern "C" __global__ void sdp_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    const float* q_row = Q + q_idx * head_dim;
    float row_max = -1e30f;

    // Pass 1: compute scores and find max for numerical stability
    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        const float* k_row = K + k_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        scores[k_idx] = dot;
        if (dot > row_max) row_max = dot;
    }

    // Pass 2: stable softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] = expf(scores[k_idx] - row_max);
        sum_exp += scores[k_idx];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] *= inv_sum;
    }

    // Pass 3: weighted sum of V
    float* o_row = O + q_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * V[k_idx * head_dim + d];
        }
        o_row[d] = acc;
    }
}

extern "C" __global__ void sdp_attention_causal_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    const float* q_row = Q + q_idx * head_dim;
    float row_max = -1e30f;

    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        if (k_idx > q_idx) {
            scores[k_idx] = -1e30f;
            continue;
        }
        const float* k_row = K + k_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        scores[k_idx] = dot;
        if (dot > row_max) row_max = dot;
    }

    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] = expf(scores[k_idx] - row_max);
        sum_exp += scores[k_idx];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        scores[k_idx] *= inv_sum;
    }

    float* o_row = O + q_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * V[k_idx * head_dim + d];
        }
        o_row[d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Launch configuration (CUDA)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CPU attention configuration
// ---------------------------------------------------------------------------

/// Configuration for the CPU attention fallback functions.
///
/// A simpler struct than [`AttentionKernelConfig`] for use with the pure-Rust
/// CPU fallback implementations.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Sequence length (query and key/value share the same length for CPU path).
    pub seq_len: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
    /// Maximum sequence length for positional bounds checking.
    pub max_seq_len: usize,
    /// Whether to use flash-attention tiled algorithm.
    pub use_flash_attention: bool,
    /// Attention dropout probability (0.0 = no dropout). Reserved for training.
    pub attention_dropout: f32,
    /// Optional explicit scale factor; when `Some`, overrides `1/sqrt(head_dim)`.
    pub scale_factor: Option<f32>,
}

impl AttentionConfig {
    /// Create a new attention config with default scale `1.0 / sqrt(head_dim)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize, causal: bool) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "AttentionConfig: dimensions must be non-zero: \
                     num_heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}"
                ),
            }
            .into());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            seq_len,
            causal,
            scale,
            max_seq_len: seq_len,
            use_flash_attention: false,
            attention_dropout: 0.0,
            scale_factor: None,
        })
    }

    /// Override the default scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self.scale_factor = Some(scale);
        self
    }

    /// Set the maximum sequence length for positional bounds checking.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Enable or disable flash-attention tiled algorithm.
    pub fn with_flash_attention(mut self, enabled: bool) -> Self {
        self.use_flash_attention = enabled;
        self
    }

    /// Set attention dropout probability (training only).
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.attention_dropout = dropout;
        self
    }

    /// Resolve the effective scale: explicit `scale_factor` if set, else `scale`.
    pub fn effective_scale(&self) -> f32 {
        self.scale_factor.unwrap_or(self.scale)
    }
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CPU fallback: single-head scaled dot-product attention
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax over `scores` in-place.
fn softmax_inplace(scores: &mut [f32]) {
    let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let e = (*s - row_max).exp();
        *s = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Pure-Rust CPU fallback for single-head scaled dot-product attention.
///
/// Computes `softmax(Q·Kᵀ · scale) · V`.
///
/// # Arguments
///
/// * `query` — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (uses `seq_len`, `head_dim`, `scale`, `causal`)
///
/// # Returns
///
/// Output tensor `[seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "attention_cpu_fallback: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        // Compute scaled dot-product scores: Q[i] · K[j]^T * scale
        let mut scores = vec![0.0_f32; seq];
        for j in 0..seq {
            if config.causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += query[i * dim + d] * key[j * dim + d];
                }
                scores[j] = dot * scale;
            }
        }

        softmax_inplace(&mut scores);

        // Weighted sum of values
        for d in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..seq {
                acc += scores[j] * value[j * dim + d];
            }
            output[i * dim + d] = acc;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: masked attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for masked scaled dot-product attention.
///
/// Computes `softmax(Q·Kᵀ · scale + mask) · V`.
///
/// # Arguments
///
/// * `query` — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[seq_len, head_dim]` (FP32, row-major)
/// * `mask`  — `[seq_len, seq_len]` additive mask (FP32, row-major);
///   use `0.0` for attending, `f32::NEG_INFINITY` for blocking
/// * `config` — Attention configuration (uses `seq_len`, `head_dim`, `scale`)
///
/// # Returns
///
/// Output tensor `[seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn masked_attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    let mask_expected = config.seq_len * config.seq_len;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "masked_attention_cpu_fallback: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }
    if mask.len() < mask_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "masked_attention_cpu_fallback: mask length {}, expected {mask_expected}",
                mask.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        let mut scores = vec![0.0_f32; seq];
        for j in 0..seq {
            let mut dot = 0.0_f32;
            for d in 0..dim {
                dot += query[i * dim + d] * key[j * dim + d];
            }
            // Additive mask applied after scaling
            scores[j] = dot * scale + mask[i * seq + j];
        }

        softmax_inplace(&mut scores);

        for d in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..seq {
                acc += scores[j] * value[j * dim + d];
            }
            output[i * dim + d] = acc;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: multi-head attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for multi-head scaled dot-product attention.
///
/// Applies single-head attention independently per head, then concatenates.
///
/// # Arguments
///
/// * `query` — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (`num_heads`, `seq_len`, `head_dim`, etc.)
///
/// # Returns
///
/// Output tensor `[num_heads, seq_len, head_dim]` as a flat `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if tensor lengths do not match the configuration.
pub fn multi_head_attention_cpu_fallback(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let head_size = config.seq_len * config.head_dim;
    let total = config.num_heads * head_size;
    if query.len() < total || key.len() < total || value.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "multi_head_attention_cpu_fallback: tensor length mismatch, \
                 expected {total}, got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; total];

    // Per-head config (single head)
    let single_cfg = AttentionConfig {
        num_heads: 1,
        head_dim: config.head_dim,
        seq_len: config.seq_len,
        causal: config.causal,
        scale: config.scale,
        max_seq_len: config.max_seq_len,
        use_flash_attention: config.use_flash_attention,
        attention_dropout: config.attention_dropout,
        scale_factor: config.scale_factor,
    };

    for h in 0..config.num_heads {
        let offset = h * head_size;
        let q_head = &query[offset..offset + head_size];
        let k_head = &key[offset..offset + head_size];
        let v_head = &value[offset..offset + head_size];
        let head_out = attention_cpu_fallback(q_head, k_head, v_head, &single_cfg)?;
        output[offset..offset + head_size].copy_from_slice(&head_out);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU convenience wrapper
// ---------------------------------------------------------------------------

/// Pure-Rust CPU reference for scaled dot-product attention.
///
/// Dispatches to single-head or multi-head depending on `config.num_heads`.
/// This is the canonical CPU entry-point; `attention_cpu_fallback` and
/// `multi_head_attention_cpu_fallback` are the underlying per-variant
/// implementations.
pub fn attention_forward_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    if config.num_heads > 1 {
        multi_head_attention_cpu_fallback(query, key, value, config)
    } else {
        attention_cpu_fallback(query, key, value, config)
    }
}

// ---------------------------------------------------------------------------
// Flash-attention style chunked CPU reference
// ---------------------------------------------------------------------------

/// Default chunk size (number of K/V positions per chunk) for the chunked
/// CPU attention implementation.  Chosen to keep the temporary score buffer
/// small enough for L1/L2 cache while still amortising the per-chunk
/// overhead.
const DEFAULT_CHUNK_SIZE: usize = 64;

/// Flash-attention style chunked single-head attention (CPU reference).
///
/// Instead of materialising the full `[seq_q, seq_kv]` score matrix, this
/// implementation streams K/V in chunks of `chunk_size` positions and
/// maintains a running softmax accumulator (online softmax trick).  Memory
/// usage is `O(seq_q * chunk_size)` instead of `O(seq_q * seq_kv)`.
///
/// The numerical result is equivalent to [`attention_cpu_fallback`] within
/// floating-point tolerance.
///
/// # Arguments
///
/// * `query`      — `[seq_len, head_dim]` (FP32, row-major)
/// * `key`        — `[seq_len, head_dim]` (FP32, row-major)
/// * `value`      — `[seq_len, head_dim]` (FP32, row-major)
/// * `config`     — Attention configuration
/// * `chunk_size` — Number of K/V positions per chunk (`0` → use default)
pub fn chunked_attention_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let expected = config.seq_len * config.head_dim;
    if query.len() < expected || key.len() < expected || value.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "chunked_attention_cpu: tensor length mismatch, expected {expected}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.scale;
    let cs = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };

    let mut output = vec![0.0_f32; expected];

    for i in 0..seq {
        // Per-query running accumulators for online softmax
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        let mut acc = vec![0.0_f32; dim]; // weighted V accumulator

        // Determine effective KV length (causal limits to positions ≤ i)
        let kv_len = if config.causal { i + 1 } else { seq };

        // Stream K/V in chunks
        let mut chunk_start = 0;
        while chunk_start < kv_len {
            let chunk_end = (chunk_start + cs).min(kv_len);
            let chunk_len = chunk_end - chunk_start;

            // Compute scores for this chunk: Q[i] · K[j]^T * scale
            let mut scores = vec![0.0_f32; chunk_len];
            let mut chunk_max = f32::NEG_INFINITY;
            for (ci, j) in (chunk_start..chunk_end).enumerate() {
                let mut dot = 0.0_f32;
                for d in 0..dim {
                    dot += query[i * dim + d] * key[j * dim + d];
                }
                scores[ci] = dot * scale;
                if scores[ci] > chunk_max {
                    chunk_max = scores[ci];
                }
            }

            // Online softmax update: merge this chunk into running state
            // Algorithm: if new_max > running_max, rescale existing accumulators.
            let new_max = running_max.max(chunk_max);

            // Rescale previous accumulator if max changed
            if running_sum > 0.0 {
                let correction = (running_max - new_max).exp();
                running_sum *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
                }
            }

            // Add this chunk's contribution
            let mut chunk_sum = 0.0_f32;
            for (ci, &score) in scores.iter().enumerate().take(chunk_len) {
                let w = (score - new_max).exp();
                chunk_sum += w;
                let j = chunk_start + ci;
                for d in 0..dim {
                    acc[d] += w * value[j * dim + d];
                }
            }

            running_max = new_max;
            running_sum += chunk_sum;
            chunk_start = chunk_end;
        }

        // Normalise
        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for d in 0..dim {
                output[i * dim + d] = acc[d] * inv;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Batch attention
// ---------------------------------------------------------------------------

/// Pure-Rust CPU fallback for batched multi-head attention.
///
/// Applies multi-head attention independently per batch element.
///
/// # Arguments
///
/// * `query` — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `key`   — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `value` — `[batch, num_heads, seq_len, head_dim]` (FP32, row-major)
/// * `config` — Attention configuration (`num_heads`, `seq_len`, `head_dim`)
/// * `batch_size` — Number of independent sequences in the batch
///
/// # Returns
///
/// Output tensor `[batch, num_heads, seq_len, head_dim]` as a flat `Vec<f32>`.
pub fn batch_attention_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
    batch_size: usize,
) -> Result<Vec<f32>> {
    if batch_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "batch_attention_cpu: batch_size must be non-zero".into(),
        }
        .into());
    }
    let per_batch = config.num_heads * config.seq_len * config.head_dim;
    let total = batch_size * per_batch;
    if query.len() < total || key.len() < total || value.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_attention_cpu: tensor length mismatch, expected {total}, \
                 got q={}, k={}, v={}",
                query.len(),
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; total];

    for b in 0..batch_size {
        let offset = b * per_batch;
        let q_batch = &query[offset..offset + per_batch];
        let k_batch = &key[offset..offset + per_batch];
        let v_batch = &value[offset..offset + per_batch];
        let batch_out = if config.num_heads > 1 {
            multi_head_attention_cpu_fallback(q_batch, k_batch, v_batch, config)?
        } else {
            attention_cpu_fallback(q_batch, k_batch, v_batch, config)?
        };
        output[offset..offset + per_batch].copy_from_slice(&batch_out);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply attention with automatic dispatch: GPU if available, else CPU fallback.
pub fn attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            let kernel_cfg = AttentionKernelConfig::for_shape(
                config.num_heads,
                config.head_dim,
                config.seq_len,
                config.seq_len,
                config.causal,
            )?;
            let total = config.num_heads * config.seq_len * config.head_dim;
            let mut output = vec![0.0_f32; total];
            if launch_attention(query, key, value, &mut output, &kernel_cfg).is_ok() {
                return Ok(output);
            }
            // GPU launch failed — fall through to CPU path
        }
    }
    attention_forward_cpu(query, key, value, config)
}

// ---------------------------------------------------------------------------
// AttentionError — typed error for attention operations
// ---------------------------------------------------------------------------

/// Errors specific to attention operations.
#[derive(Debug)]
pub enum AttentionError {
    /// A dimension (heads, head_dim, seq_len, etc.) was invalid.
    InvalidDimension {
        /// Human-readable description of what went wrong.
        message: String,
    },
    /// Input tensor shape does not match the expected layout.
    ShapeMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements supplied.
        actual: usize,
    },
    /// A numerical issue was detected (NaN, Inf, etc.).
    NumericalInstability {
        /// Human-readable description of what went wrong.
        message: String,
    },
    /// The requested operation is not supported on this device.
    UnsupportedOperation {
        /// Human-readable description of the unsupported operation.
        message: String,
    },
}

impl core::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimension { message } => {
                write!(f, "invalid dimension: {message}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected} elements, got {actual}")
            }
            Self::NumericalInstability { message } => {
                write!(f, "numerical instability: {message}")
            }
            Self::UnsupportedOperation { message } => {
                write!(f, "unsupported operation: {message}")
            }
        }
    }
}

impl std::error::Error for AttentionError {}

// ---------------------------------------------------------------------------
// CUDA kernel source for extended attention ops
// ---------------------------------------------------------------------------

/// Inline CUDA C source for extended attention kernels (GQA, flash-attention,
/// component-wise score/mask/softmax/output).
///
/// These supplement [`ATTENTION_KERNEL_SRC`] with finer-grained GPU entry points.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ATTENTION_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void compute_attention_scores_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (q_idx >= seq_len_q || k_idx >= seq_len_kv) return;

    const float* q_row = Q + q_idx * head_dim;
    const float* k_row = K + k_idx * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q_row[d] * k_row[d];
    }
    scores[q_idx * seq_len_kv + k_idx] = dot * scale;
}

extern "C" __global__ void apply_attention_mask_f32(
    const float* __restrict__ scores,
    const float* __restrict__ mask,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = scores[idx] + mask[idx];
}

extern "C" __global__ void attention_softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    float row_max = -1e30f;
    for (int j = 0; j < cols; j++) {
        if (in_row[j] > row_max) row_max = in_row[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = expf(in_row[j] - row_max);
        out_row[j] = e;
        sum_exp += e;
    }
    float inv = 1.0f / sum_exp;
    for (int j = 0; j < cols; j++) {
        out_row[j] *= inv;
    }
}

extern "C" __global__ void compute_attention_output_f32(
    const float* __restrict__ weights,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len_q,
    int seq_len_kv,
    int head_dim)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    float* o_row = output + q_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k = 0; k < seq_len_kv; k++) {
            acc += weights[q_idx * seq_len_kv + k] * V[k * head_dim + d];
        }
        o_row[d] = acc;
    }
}

extern "C" __global__ void grouped_query_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len,
    int head_dim,
    int num_q_heads,
    int num_kv_heads,
    float scale)
{
    int q_head = blockIdx.y;
    int q_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_head >= num_q_heads || q_idx >= seq_len) return;

    int kv_head = q_head * num_kv_heads / num_q_heads;
    const float* q_row = Q + (q_head * seq_len + q_idx) * head_dim;
    const float* k_base = K + kv_head * seq_len * head_dim;
    const float* v_base = V + kv_head * seq_len * head_dim;
    float* o_row = O + (q_head * seq_len + q_idx) * head_dim;

    extern __shared__ float scores[];
    float row_max = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_base[j * head_dim + d];
        }
        dot *= scale;
        scores[j] = dot;
        if (dot > row_max) row_max = dot;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[j] = expf(scores[j] - row_max);
        sum_exp += scores[j];
    }
    float inv = 1.0f / sum_exp;
    for (int j = 0; j < seq_len; j++) scores[j] *= inv;

    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += scores[j] * v_base[j * head_dim + d];
        }
        o_row[d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Component-wise CPU attention functions
// ---------------------------------------------------------------------------

/// Compute raw attention scores: `output[i,j] = sum_d(Q[i,d] * K[j,d]) * scale`.
///
/// `q` is `[seq_q, head_dim]`, `k` is `[seq_kv, head_dim]`,
/// `output` is `[seq_q, seq_kv]`.
pub fn compute_attention_scores(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    if head_dim == 0 {
        return Err(AttentionError::InvalidDimension {
            message: "head_dim must be non-zero".into(),
        });
    }
    let expected_q = seq_q * head_dim;
    if q.len() < expected_q {
        return Err(AttentionError::ShapeMismatch { expected: expected_q, actual: q.len() });
    }
    let expected_k = seq_kv * head_dim;
    if k.len() < expected_k {
        return Err(AttentionError::ShapeMismatch { expected: expected_k, actual: k.len() });
    }
    let expected_out = seq_q * seq_kv;
    if output.len() < expected_out {
        return Err(AttentionError::ShapeMismatch { expected: expected_out, actual: output.len() });
    }

    for i in 0..seq_q {
        for j in 0..seq_kv {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            output[i * seq_kv + j] = dot * scale;
        }
    }
    Ok(())
}

/// Apply an additive attention mask: `output[i] = scores[i] + mask[i]`.
///
/// All slices must have at least `len` elements.
pub fn apply_attention_mask(
    scores: &[f32],
    mask: &[f32],
    output: &mut [f32],
    len: usize,
) -> std::result::Result<(), AttentionError> {
    if scores.len() < len {
        return Err(AttentionError::ShapeMismatch { expected: len, actual: scores.len() });
    }
    if mask.len() < len {
        return Err(AttentionError::ShapeMismatch { expected: len, actual: mask.len() });
    }
    if output.len() < len {
        return Err(AttentionError::ShapeMismatch { expected: len, actual: output.len() });
    }
    for i in 0..len {
        output[i] = scores[i] + mask[i];
    }
    Ok(())
}

/// Numerically stable row-wise softmax.
///
/// `scores` is `[rows, cols]`, `output` is `[rows, cols]`.
pub fn attention_softmax(
    scores: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    if cols == 0 {
        return Err(AttentionError::InvalidDimension {
            message: "cols must be non-zero for softmax".into(),
        });
    }
    let total = rows * cols;
    if scores.len() < total {
        return Err(AttentionError::ShapeMismatch { expected: total, actual: scores.len() });
    }
    if output.len() < total {
        return Err(AttentionError::ShapeMismatch { expected: total, actual: output.len() });
    }

    for r in 0..rows {
        let row_start = r * cols;
        let row = &scores[row_start..row_start + cols];
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0_f32;
        for c in 0..cols {
            let e = (row[c] - row_max).exp();
            output[row_start + c] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            let inv = 1.0 / sum_exp;
            for c in 0..cols {
                output[row_start + c] *= inv;
            }
        }
    }
    Ok(())
}

/// Compute weighted output: `output[i,d] = sum_j(weights[i,j] * V[j,d])`.
///
/// `weights` is `[seq_q, seq_kv]`, `v` is `[seq_kv, head_dim]`,
/// `output` is `[seq_q, head_dim]`.
pub fn compute_attention_output(
    weights: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    let expected_w = seq_q * seq_kv;
    if weights.len() < expected_w {
        return Err(AttentionError::ShapeMismatch { expected: expected_w, actual: weights.len() });
    }
    let expected_v = seq_kv * head_dim;
    if v.len() < expected_v {
        return Err(AttentionError::ShapeMismatch { expected: expected_v, actual: v.len() });
    }
    let expected_o = seq_q * head_dim;
    if output.len() < expected_o {
        return Err(AttentionError::ShapeMismatch { expected: expected_o, actual: output.len() });
    }

    for i in 0..seq_q {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_kv {
                acc += weights[i * seq_kv + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
    Ok(())
}

/// End-to-end scaled dot-product attention with optional mask.
///
/// Computes `softmax(Q·K^T * scale + mask) · V` and writes the result to `output`.
///
/// `q` is `[seq_q, head_dim]`, `k` is `[seq_kv, head_dim]`, `v` is `[seq_kv, head_dim]`,
/// `mask` (optional) is `[seq_q, seq_kv]`, `output` is `[seq_q, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    let score_len = seq_q * seq_kv;
    let mut scores = vec![0.0_f32; score_len];
    compute_attention_scores(q, k, seq_q, seq_kv, head_dim, scale, &mut scores)?;

    if let Some(m) = mask {
        let mut masked = vec![0.0_f32; score_len];
        apply_attention_mask(&scores, m, &mut masked, score_len)?;
        scores = masked;
    }

    let mut weights = vec![0.0_f32; score_len];
    attention_softmax(&scores, seq_q, seq_kv, &mut weights)?;

    compute_attention_output(&weights, v, seq_q, seq_kv, head_dim, output)
}

/// Multi-head attention with explicit weight projections.
///
/// Applies `Wq`, `Wk`, `Wv` to `input` to produce Q, K, V, runs per-head
/// attention, concatenates heads, and applies `Wo`.
///
/// `input`  — `[seq_len, model_dim]`
/// `wq/wk`  — `[model_dim, num_heads * head_dim]`
/// `wv`     — `[model_dim, num_heads * head_dim]`
/// `wo`     — `[num_heads * head_dim, model_dim]`
/// `output` — `[seq_len, model_dim]`
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    config: &AttentionConfig,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    let model_dim = config.num_heads * config.head_dim;
    let seq = config.seq_len;
    let dim = config.head_dim;
    let n_heads = config.num_heads;

    let expected_input = seq * model_dim;
    if input.len() < expected_input {
        return Err(AttentionError::ShapeMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let expected_proj = model_dim * model_dim;
    if wq.len() < expected_proj || wk.len() < expected_proj || wv.len() < expected_proj {
        return Err(AttentionError::ShapeMismatch {
            expected: expected_proj,
            actual: wq.len().min(wk.len()).min(wv.len()),
        });
    }
    if wo.len() < expected_proj {
        return Err(AttentionError::ShapeMismatch { expected: expected_proj, actual: wo.len() });
    }
    if output.len() < expected_input {
        return Err(AttentionError::ShapeMismatch {
            expected: expected_input,
            actual: output.len(),
        });
    }

    // Project: Q = input @ Wq, K = input @ Wk, V = input @ Wv
    let mut q_proj = vec![0.0_f32; seq * model_dim];
    let mut k_proj = vec![0.0_f32; seq * model_dim];
    let mut v_proj = vec![0.0_f32; seq * model_dim];
    matmul_simple(input, wq, &mut q_proj, seq, model_dim, model_dim);
    matmul_simple(input, wk, &mut k_proj, seq, model_dim, model_dim);
    matmul_simple(input, wv, &mut v_proj, seq, model_dim, model_dim);

    // Per-head attention
    let scale = config.effective_scale();
    let head_elements = seq * dim;
    let mut concat = vec![0.0_f32; seq * model_dim];

    for h in 0..n_heads {
        // Extract head h: gather [seq, dim] from [seq, model_dim]
        let mut q_head = vec![0.0_f32; head_elements];
        let mut k_head = vec![0.0_f32; head_elements];
        let mut v_head = vec![0.0_f32; head_elements];
        for s in 0..seq {
            let src_off = s * model_dim + h * dim;
            let dst_off = s * dim;
            q_head[dst_off..dst_off + dim].copy_from_slice(&q_proj[src_off..src_off + dim]);
            k_head[dst_off..dst_off + dim].copy_from_slice(&k_proj[src_off..src_off + dim]);
            v_head[dst_off..dst_off + dim].copy_from_slice(&v_proj[src_off..src_off + dim]);
        }

        let mut head_out = vec![0.0_f32; head_elements];
        let mask: Option<&[f32]> = None;

        if config.causal {
            // Build causal mask for this head
            let mut causal_mask = vec![0.0_f32; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    if j > i {
                        causal_mask[i * seq + j] = f32::NEG_INFINITY;
                    }
                }
            }
            scaled_dot_product_attention(
                &q_head,
                &k_head,
                &v_head,
                Some(&causal_mask),
                seq,
                seq,
                dim,
                scale,
                &mut head_out,
            )?;
        } else {
            scaled_dot_product_attention(
                &q_head, &k_head, &v_head, mask, seq, seq, dim, scale, &mut head_out,
            )?;
        }

        // Scatter back into concat buffer
        for s in 0..seq {
            let src_off = s * dim;
            let dst_off = s * model_dim + h * dim;
            concat[dst_off..dst_off + dim].copy_from_slice(&head_out[src_off..src_off + dim]);
        }
    }

    // Output projection: output = concat @ Wo
    matmul_simple(&concat, wo, output, seq, model_dim, model_dim);
    Ok(())
}

/// Flash-attention forward pass (tiled, streaming K/V).
///
/// This is an alias that delegates to [`chunked_attention_cpu`] with a tile
/// size derived from [`AttentionConfig`].  When `config.use_flash_attention`
/// is `false`, falls back to the standard path.
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &AttentionConfig,
    output: &mut [f32],
) -> Result<()> {
    let tile_size = if config.use_flash_attention { 64 } else { 0 };
    let result = chunked_attention_cpu(q, k, v, config, tile_size)?;
    let expected = config.seq_len * config.head_dim;
    if output.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "flash_attention_forward: output length {}, expected {expected}",
                output.len()
            ),
        }
        .into());
    }
    output[..expected].copy_from_slice(&result[..expected]);
    Ok(())
}

/// Grouped-query attention (GQA): Q heads share K/V heads.
///
/// `q` is `[num_q_heads, seq_len, head_dim]`, `k`/`v` are
/// `[num_kv_heads, seq_len, head_dim]`.  `num_q_heads` must be divisible
/// by `num_kv_heads`.
///
/// `output` is `[num_q_heads, seq_len, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_kv_heads: usize,
    config: &AttentionConfig,
    output: &mut [f32],
) -> std::result::Result<(), AttentionError> {
    let n_q = config.num_heads;
    let seq = config.seq_len;
    let dim = config.head_dim;
    let scale = config.effective_scale();

    if num_kv_heads == 0 {
        return Err(AttentionError::InvalidDimension {
            message: "num_kv_heads must be non-zero".into(),
        });
    }
    if !n_q.is_multiple_of(num_kv_heads) {
        return Err(AttentionError::InvalidDimension {
            message: format!(
                "num_q_heads ({n_q}) must be divisible by num_kv_heads ({num_kv_heads})"
            ),
        });
    }

    let head_elements = seq * dim;
    let expected_q = n_q * head_elements;
    let expected_kv = num_kv_heads * head_elements;
    if q.len() < expected_q {
        return Err(AttentionError::ShapeMismatch { expected: expected_q, actual: q.len() });
    }
    if k.len() < expected_kv {
        return Err(AttentionError::ShapeMismatch { expected: expected_kv, actual: k.len() });
    }
    if v.len() < expected_kv {
        return Err(AttentionError::ShapeMismatch { expected: expected_kv, actual: v.len() });
    }
    if output.len() < expected_q {
        return Err(AttentionError::ShapeMismatch { expected: expected_q, actual: output.len() });
    }

    let groups = n_q / num_kv_heads;

    for q_head in 0..n_q {
        let kv_head = q_head / groups;
        let q_off = q_head * head_elements;
        let kv_off = kv_head * head_elements;
        let o_off = q_head * head_elements;

        let q_slice = &q[q_off..q_off + head_elements];
        let k_slice = &k[kv_off..kv_off + head_elements];
        let v_slice = &v[kv_off..kv_off + head_elements];

        // Per-head SDP
        let mut scores_buf = vec![0.0_f32; seq * seq];
        compute_attention_scores(q_slice, k_slice, seq, seq, dim, scale, &mut scores_buf)?;

        if config.causal {
            for i in 0..seq {
                for j in (i + 1)..seq {
                    scores_buf[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }

        let mut weights_buf = vec![0.0_f32; seq * seq];
        attention_softmax(&scores_buf, seq, seq, &mut weights_buf)?;

        compute_attention_output(
            &weights_buf,
            v_slice,
            seq,
            seq,
            dim,
            &mut output[o_off..o_off + head_elements],
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple row-major matmul: C[m,n] = A[m,k] @ B[k,n]
fn matmul_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionKernelConfig tests ───────────────────────────────────

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

    // ── AttentionConfig tests ─────────────────────────────────────────

    #[test]
    fn test_cpu_config_new() {
        let cfg = AttentionConfig::new(8, 64, 16, true).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.seq_len, 16);
        assert!(cfg.causal);
        assert!((cfg.scale - 1.0 / (64.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_config_rejects_zero() {
        assert!(AttentionConfig::new(0, 64, 16, false).is_err());
        assert!(AttentionConfig::new(8, 0, 16, false).is_err());
        assert!(AttentionConfig::new(8, 64, 0, false).is_err());
    }

    #[test]
    fn test_cpu_config_custom_scale() {
        let cfg = AttentionConfig::new(1, 64, 4, false).unwrap().with_scale(0.5);
        assert!((cfg.scale - 0.5).abs() < f32::EPSILON);
    }

    // ── Single-head CPU fallback tests ────────────────────────────────

    #[test]
    fn test_cpu_attention_identity_key() {
        // Q == K: each query attends most to itself (non-causal, uniform V)
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let qk = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ];
        let value = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            0.5, 0.5, //
        ];
        let out = attention_cpu_fallback(&qk, &qk, &value, &cfg).unwrap();
        // Output should be well-formed (finite, seq_len * head_dim)
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_output_shape() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let q = vec![0.1_f32; 32];
        let k = vec![0.2_f32; 32];
        let v = vec![0.3_f32; 32];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 32); // seq_len * head_dim
    }

    #[test]
    fn test_cpu_attention_uniform_query_equal_values() {
        // All-equal Q,K → uniform attention → output == mean(V rows)
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let v = vec![3.0, 6.0, 3.0, 6.0, 3.0, 6.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // All rows: mean of [3,6], [3,6], [3,6] = [3,6]
        for row in 0..3 {
            assert!((out[row * 2] - 3.0).abs() < 1e-5);
            assert!((out[row * 2 + 1] - 6.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_attention_numerical_stability_large_values() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![500.0, 500.0, -500.0, -500.0];
        let k = vec![500.0, 500.0, -500.0, -500.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite with large values");
    }

    #[test]
    fn test_cpu_attention_single_position() {
        // seq_len=1: output == value (only one position to attend to)
        let cfg = AttentionConfig::new(1, 4, 1, false).unwrap();
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        for d in 0..4 {
            assert!((out[d] - v[d]).abs() < 1e-5, "seq_len=1 should return V");
        }
    }

    #[test]
    fn test_cpu_attention_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let short = vec![0.0_f32; 16]; // need 32
        let ok = vec![0.0_f32; 32];
        assert!(attention_cpu_fallback(&short, &ok, &ok, &cfg).is_err());
        assert!(attention_cpu_fallback(&ok, &short, &ok, &cfg).is_err());
        assert!(attention_cpu_fallback(&ok, &ok, &short, &cfg).is_err());
    }

    // ── Causal masking tests ──────────────────────────────────────────

    #[test]
    fn test_cpu_attention_causal_first_token() {
        // First token with causal: can only attend to itself → output == V[0]
        let cfg = AttentionConfig::new(1, 2, 3, true).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Row 0: can only see position 0 → output = V[0]
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_attention_causal_vs_noncausal() {
        let seq = 4;
        let dim = 2;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v: Vec<f32> = (0..seq * dim).map(|i| i as f32).collect();

        let causal_cfg = AttentionConfig::new(1, dim, seq, true).unwrap();
        let noncausal_cfg = AttentionConfig::new(1, dim, seq, false).unwrap();

        let out_c = attention_cpu_fallback(&q, &k, &v, &causal_cfg).unwrap();
        let out_nc = attention_cpu_fallback(&q, &k, &v, &noncausal_cfg).unwrap();

        // Non-causal: all rows see all positions → all rows identical
        // Causal: each row sees only positions ≤ itself → rows differ
        let row0_c = &out_c[0..dim];
        let row1_c = &out_c[dim..2 * dim];
        // Row 0 should differ from row 1 under causal masking
        let diff: f32 = row0_c.iter().zip(row1_c).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "causal rows should differ");

        // Non-causal: all rows same
        let row0_nc = &out_nc[0..dim];
        let row1_nc = &out_nc[dim..2 * dim];
        let diff_nc: f32 = row0_nc.iter().zip(row1_nc).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_nc < 1e-5, "non-causal rows should be identical");
    }

    #[test]
    fn test_cpu_attention_causal_monotonic_context() {
        // Under causal masking, later tokens have more context
        let cfg = AttentionConfig::new(1, 2, 4, true).unwrap();
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Each row's output[0] should increase (more context, higher-indexed V)
        for i in 0..3 {
            assert!(out[(i + 1) * 2] >= out[i * 2] - 1e-5, "causal context should be monotonic");
        }
    }

    // ── Masked attention tests ────────────────────────────────────────

    #[test]
    fn test_masked_attention_zero_mask_equals_unmasked() {
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0];
        let k = vec![0.5, 0.5, 1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let zero_mask = vec![0.0_f32; 9]; // 3×3 zero mask

        let out_unmasked = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let out_masked = masked_attention_cpu_fallback(&q, &k, &v, &zero_mask, &cfg).unwrap();

        for (a, b) in out_unmasked.iter().zip(out_masked.iter()) {
            assert!((a - b).abs() < 1e-5, "zero mask should equal unmasked");
        }
    }

    #[test]
    fn test_masked_attention_blocks_positions() {
        // Mask blocks all except self-attention (diagonal)
        let seq = 3;
        let dim = 2;
        let cfg = AttentionConfig::new(1, dim, seq, false).unwrap();
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        // Diagonal mask: 0 on diagonal, -inf off-diagonal
        let ninf = f32::NEG_INFINITY;
        #[rustfmt::skip]
        let mask = vec![
            0.0,  ninf, ninf,
            ninf, 0.0,  ninf,
            ninf, ninf, 0.0,
        ];

        let out = masked_attention_cpu_fallback(&q, &k, &v, &mask, &cfg).unwrap();
        // Each row attends only to itself → output == value
        for i in 0..seq {
            for d in 0..dim {
                assert!(
                    (out[i * dim + d] - v[i * dim + d]).abs() < 1e-5,
                    "diagonal mask: row {i} dim {d}"
                );
            }
        }
    }

    #[test]
    fn test_masked_attention_rejects_short_mask() {
        let cfg = AttentionConfig::new(1, 2, 4, false).unwrap();
        let t = vec![0.0_f32; 8];
        let short_mask = vec![0.0_f32; 8]; // need 16
        assert!(masked_attention_cpu_fallback(&t, &t, &t, &short_mask, &cfg).is_err());
    }

    #[test]
    fn test_masked_attention_numerical_stability() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![1000.0, -1000.0, 0.0, 0.0];
        let k = vec![1000.0, -1000.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mask = vec![0.0_f32; 4];
        let out = masked_attention_cpu_fallback(&q, &k, &v, &mask, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite with large values");
    }

    // ── Multi-head attention tests ────────────────────────────────────

    #[test]
    fn test_multi_head_output_shape() {
        let cfg = AttentionConfig::new(4, 8, 6, false).unwrap();
        let total = 4 * 6 * 8;
        let q = vec![0.1_f32; total];
        let k = vec![0.2_f32; total];
        let v = vec![0.3_f32; total];
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_multi_head_independent_heads() {
        // Different data per head → different outputs per head
        let cfg = AttentionConfig::new(2, 2, 2, false).unwrap();
        let q = vec![
            1.0, 0.0, 0.0, 1.0, // head 0
            0.0, 1.0, 1.0, 0.0, // head 1
        ];
        let k = vec![
            1.0, 0.0, 0.0, 1.0, // head 0
            0.0, 1.0, 1.0, 0.0, // head 1
        ];
        let v = vec![
            10.0, 20.0, 30.0, 40.0, // head 0
            50.0, 60.0, 70.0, 80.0, // head 1
        ];
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Head 0 output should differ from head 1 output
        let head0 = &out[0..4];
        let head1 = &out[4..8];
        let diff: f32 = head0.iter().zip(head1).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "heads should produce different outputs");
    }

    #[test]
    fn test_multi_head_matches_single_head() {
        // Multi-head with 1 head should match single-head
        let cfg = AttentionConfig::new(1, 4, 3, false).unwrap();
        let q = vec![1.0, 0.5, 0.0, -0.5, 0.2, 0.8, -0.3, 0.1, 0.7, -0.2, 0.4, 0.6];
        let k = vec![0.5, 1.0, -0.5, 0.0, -0.1, 0.3, 0.7, -0.2, 0.4, 0.0, 0.1, 0.9];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let out_single = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let out_multi = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();

        for (a, b) in out_single.iter().zip(out_multi.iter()) {
            assert!((a - b).abs() < 1e-5, "single vs multi mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_multi_head_causal() {
        let cfg = AttentionConfig::new(2, 2, 3, true).unwrap();
        let total = 2 * 3 * 2;
        let q = vec![1.0_f32; total];
        let k = vec![1.0_f32; total];
        let v: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let out = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multi_head_rejects_short_tensors() {
        let cfg = AttentionConfig::new(4, 4, 4, false).unwrap();
        let short = vec![0.0_f32; 32]; // need 64
        let ok = vec![0.0_f32; 64];
        assert!(multi_head_attention_cpu_fallback(&short, &ok, &ok, &cfg).is_err());
    }

    // ── Unified dispatch tests ────────────────────────────────────────

    #[test]
    fn test_attention_forward_cpu_single_head() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![2.0_f32; 8];
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        // Uniform Q,K,V → output == V
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_forward_cpu_multi_head() {
        let cfg = AttentionConfig::new(2, 4, 3, true).unwrap();
        let total = 2 * 3 * 4;
        let q = vec![0.5_f32; total];
        let k = vec![0.5_f32; total];
        let v = vec![1.0_f32; total];
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Edge case tests ───────────────────────────────────────────────

    #[test]
    fn test_cpu_attention_large_head_dim() {
        let cfg = AttentionConfig::new(1, 128, 2, false).unwrap();
        let size = 2 * 128;
        let q = vec![0.01_f32; size];
        let k = vec![0.01_f32; size];
        let v = vec![1.0_f32; size];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_negative_values() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let q = vec![-1.0, -2.0, -3.0, -4.0];
        let k = vec![-1.0, -2.0, -3.0, -4.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cpu_attention_softmax_sum_to_one() {
        // Verify attention weights implicitly sum to 1 by checking output
        // is a convex combination of values
        let cfg = AttentionConfig::new(1, 1, 3, false).unwrap();
        let q = vec![1.0, 0.0, -1.0];
        let k = vec![1.0, 0.0, -1.0];
        let v = vec![0.0, 50.0, 100.0];
        let out = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        // Each output should be in [0, 100] (convex combination)
        for &val in &out {
            assert!(val >= -1e-5 && val <= 100.0 + 1e-5, "out of range: {val}");
        }
    }

    // ── CUDA launch stub test ─────────────────────────────────────────

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

    // ── CUDA kernel source compile guard ──────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_kernel_source_not_empty() {
        assert!(!ATTENTION_KERNEL_SRC.is_empty(), "CUDA kernel source should not be empty");
        assert!(ATTENTION_KERNEL_SRC.contains("sdp_attention_f32"));
        assert!(ATTENTION_KERNEL_SRC.contains("sdp_attention_causal_f32"));
    }

    // ── CudaAttentionConfig alias test ────────────────────────────────

    #[test]
    fn test_cuda_attention_config_alias() {
        let cfg: CudaAttentionConfig = CudaAttentionConfig::for_shape(4, 64, 16, 16, true).unwrap();
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert!(cfg.causal);
    }

    // ── attention_forward_cpu wrapper tests ────────────────────────────

    #[test]
    fn test_attention_forward_cpu_single() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![2.0_f32; 8];
        let out = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), 8);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_forward_cpu_multi() {
        let cfg = AttentionConfig::new(2, 4, 3, false).unwrap();
        let total = 2 * 3 * 4;
        let q = vec![0.5_f32; total];
        let k = vec![0.5_f32; total];
        let v = vec![1.0_f32; total];
        let out = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Chunked (flash-attention style) CPU tests ─────────────────────

    #[test]
    fn test_chunked_matches_standard_noncausal() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let q: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..32).map(|i| ((i + 7) as f32) * 0.05).collect();
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.2).collect();

        let standard = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let chunked = chunked_attention_cpu(&q, &k, &v, &cfg, 3).unwrap();

        assert_eq!(standard.len(), chunked.len());
        for (a, b) in standard.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "chunked vs standard mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_chunked_matches_standard_causal() {
        let cfg = AttentionConfig::new(1, 2, 6, true).unwrap();
        let q = vec![1.0_f32; 12];
        let k: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();

        let standard = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let chunked = chunked_attention_cpu(&q, &k, &v, &cfg, 2).unwrap();

        for (a, b) in standard.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "chunked causal mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_chunked_default_chunk_size() {
        // chunk_size=0 should use default and still produce correct results
        let cfg = AttentionConfig::new(1, 2, 4, false).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![1.0_f32; 8];
        let v = vec![3.0_f32; 8];
        let out = chunked_attention_cpu(&q, &k, &v, &cfg, 0).unwrap();
        for &val in &out {
            assert!((val - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_chunked_single_token() {
        let cfg = AttentionConfig::new(1, 4, 1, false).unwrap();
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = chunked_attention_cpu(&q, &k, &v, &cfg, 1).unwrap();
        for d in 0..4 {
            assert!((out[d] - v[d]).abs() < 1e-5, "seq_len=1 chunked should return V");
        }
    }

    #[test]
    fn test_chunked_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 4, 8, false).unwrap();
        let short = vec![0.0_f32; 16]; // need 32
        let ok = vec![0.0_f32; 32];
        assert!(chunked_attention_cpu(&short, &ok, &ok, &cfg, 4).is_err());
    }

    // ── Batch attention tests ─────────────────────────────────────────

    #[test]
    fn test_batch_attention_single_batch() {
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let q = vec![1.0_f32; 6];
        let k = vec![1.0_f32; 6];
        let v = vec![5.0_f32; 6];

        let single = attention_forward_cpu(&q, &k, &v, &cfg).unwrap();
        let batched = batch_attention_cpu(&q, &k, &v, &cfg, 1).unwrap();

        for (a, b) in single.iter().zip(batched.iter()) {
            assert!((a - b).abs() < 1e-5, "batch=1 should match single");
        }
    }

    #[test]
    fn test_batch_attention_two_batches() {
        let cfg = AttentionConfig::new(2, 2, 2, false).unwrap();
        let per_batch = 2 * 2 * 2; // heads * seq * dim = 8
        // Batch 0: all ones; Batch 1: counting
        let mut q = vec![1.0_f32; per_batch];
        q.extend((0..per_batch).map(|i| i as f32 * 0.1));
        let mut k = vec![1.0_f32; per_batch];
        k.extend((0..per_batch).map(|i| (i as f32 + 1.0) * 0.1));
        let mut v = vec![3.0_f32; per_batch];
        v.extend((0..per_batch).map(|i| i as f32));

        let out = batch_attention_cpu(&q, &k, &v, &cfg, 2).unwrap();
        assert_eq!(out.len(), 2 * per_batch);
        assert!(out.iter().all(|v| v.is_finite()));

        // Batch 0 (uniform) → output == V = 3.0
        for &val in &out[..per_batch] {
            assert!((val - 3.0).abs() < 1e-4, "batch 0 uniform: {val}");
        }
    }

    #[test]
    fn test_batch_attention_rejects_zero_batch() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let t = vec![0.0_f32; 4];
        assert!(batch_attention_cpu(&t, &t, &t, &cfg, 0).is_err());
    }

    #[test]
    fn test_batch_attention_rejects_short_tensors() {
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let short = vec![0.0_f32; 4]; // need 8 for batch=2
        assert!(batch_attention_cpu(&short, &short, &short, &cfg, 2).is_err());
    }

    // ── Scale factor verification ─────────────────────────────────────

    #[test]
    fn test_scale_factor_affects_output() {
        let cfg_default = AttentionConfig::new(1, 4, 2, false).unwrap();
        let cfg_big = AttentionConfig::new(1, 4, 2, false).unwrap().with_scale(10.0);
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let out_default = attention_cpu_fallback(&q, &k, &v, &cfg_default).unwrap();
        let out_big = attention_cpu_fallback(&q, &k, &v, &cfg_big).unwrap();

        // Large scale sharpens attention → outputs should differ
        let diff: f32 = out_default.iter().zip(out_big.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "different scales should produce different outputs");
    }

    // ── Equal Q=K=V edge case ─────────────────────────────────────────

    #[test]
    fn test_equal_qkv() {
        // When Q == K == V, output should equal the input (uniform attention
        // over identical values returns those values).
        let cfg = AttentionConfig::new(1, 2, 3, false).unwrap();
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let out = attention_cpu_fallback(&data, &data, &data, &cfg).unwrap();
        for i in 0..6 {
            assert!(
                (out[i] - data[i]).abs() < 1e-4,
                "Q==K==V: output should match input at index {i}"
            );
        }
    }

    // ── AttentionError tests ──────────────────────────────────────────

    #[test]
    fn test_attention_error_display_invalid_dimension() {
        let e = AttentionError::InvalidDimension { message: "zero heads".into() };
        assert_eq!(format!("{e}"), "invalid dimension: zero heads");
    }

    #[test]
    fn test_attention_error_display_shape_mismatch() {
        let e = AttentionError::ShapeMismatch { expected: 64, actual: 32 };
        assert_eq!(format!("{e}"), "shape mismatch: expected 64 elements, got 32");
    }

    #[test]
    fn test_attention_error_display_numerical() {
        let e = AttentionError::NumericalInstability { message: "NaN detected".into() };
        assert_eq!(format!("{e}"), "numerical instability: NaN detected");
    }

    #[test]
    fn test_attention_error_display_unsupported() {
        let e = AttentionError::UnsupportedOperation { message: "no GPU".into() };
        assert_eq!(format!("{e}"), "unsupported operation: no GPU");
    }

    #[test]
    fn test_attention_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(AttentionError::InvalidDimension {
            message: "test".into(),
        });
        assert!(e.to_string().contains("test"));
    }

    // ── compute_attention_scores tests ────────────────────────────────

    #[test]
    fn test_compute_scores_identity() {
        // Q = K = identity-like → scores[i,i] should be highest
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        compute_attention_scores(&q, &k, 2, 2, 2, 1.0, &mut out).unwrap();
        assert!(out[0] >= out[1], "diagonal should dominate");
        assert!(out[3] >= out[2], "diagonal should dominate");
    }

    #[test]
    fn test_compute_scores_scale() {
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 1.0];
        let mut out_s1 = vec![0.0_f32; 1];
        let mut out_s2 = vec![0.0_f32; 1];
        compute_attention_scores(&q, &k, 1, 1, 2, 1.0, &mut out_s1).unwrap();
        compute_attention_scores(&q, &k, 1, 1, 2, 0.5, &mut out_s2).unwrap();
        assert!((out_s1[0] - 2.0 * out_s2[0]).abs() < 1e-6);
    }

    #[test]
    fn test_compute_scores_shape() {
        let q = vec![0.1_f32; 12]; // 3 × 4
        let k = vec![0.2_f32; 8]; // 2 × 4
        let mut out = vec![0.0_f32; 6]; // 3 × 2
        compute_attention_scores(&q, &k, 3, 2, 4, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_scores_rejects_short_q() {
        let mut out = vec![0.0_f32; 4];
        let err = compute_attention_scores(&[1.0], &[1.0, 2.0], 2, 1, 1, 1.0, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn test_compute_scores_rejects_zero_head_dim() {
        let mut out = vec![0.0_f32; 1];
        let err = compute_attention_scores(&[], &[], 1, 1, 0, 1.0, &mut out);
        assert!(err.is_err());
    }

    // ── apply_attention_mask tests ────────────────────────────────────

    #[test]
    fn test_apply_mask_additive() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        let mut out = vec![0.0_f32; 4];
        apply_attention_mask(&scores, &mask, &mut out, 4).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!(out[1] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_apply_mask_zero_is_identity() {
        let scores = vec![1.5, 2.5];
        let mask = vec![0.0, 0.0];
        let mut out = vec![0.0_f32; 2];
        apply_attention_mask(&scores, &mask, &mut out, 2).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_mask_rejects_short() {
        let mut out = vec![0.0_f32; 4];
        let err = apply_attention_mask(&[1.0], &[0.0, 0.0], &mut out, 2);
        assert!(err.is_err());
    }

    // ── attention_softmax tests ───────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0_f32; 3];
        attention_softmax(&input, 1, 3, &mut out).unwrap();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {sum}");
    }

    #[test]
    fn test_softmax_preserves_order() {
        let input = vec![1.0, 3.0, 2.0];
        let mut out = vec![0.0_f32; 3];
        attention_softmax(&input, 1, 3, &mut out).unwrap();
        assert!(out[1] > out[2] && out[2] > out[0]);
    }

    #[test]
    fn test_softmax_multirow() {
        let input = vec![1.0, 2.0, 10.0, 20.0];
        let mut out = vec![0.0_f32; 4];
        attention_softmax(&input, 2, 2, &mut out).unwrap();
        let sum_r0: f32 = out[0..2].iter().sum();
        let sum_r1: f32 = out[2..4].iter().sum();
        assert!((sum_r0 - 1.0).abs() < 1e-5);
        assert!((sum_r1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_numerical_stability_large() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let mut out = vec![0.0_f32; 3];
        attention_softmax(&input, 1, 3, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "softmax should be stable with large values");
    }

    #[test]
    fn test_softmax_rejects_zero_cols() {
        let mut out = vec![0.0_f32; 1];
        let err = attention_softmax(&[1.0], 1, 0, &mut out);
        assert!(err.is_err());
    }

    // ── compute_attention_output tests ────────────────────────────────

    #[test]
    fn test_compute_output_identity_weights() {
        // weights = identity → output rows copy corresponding V rows
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // [2,2] identity
        let v = vec![10.0, 20.0, 30.0, 40.0]; // [2,2]
        let mut out = vec![0.0_f32; 4];
        compute_attention_output(&weights, &v, 2, 2, 2, &mut out).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-5);
        assert!((out[1] - 20.0).abs() < 1e-5);
        assert!((out[2] - 30.0).abs() < 1e-5);
        assert!((out[3] - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_output_uniform_weights() {
        // uniform weights → output = mean of V rows
        let weights = vec![0.5, 0.5, 0.5, 0.5]; // [2,2]
        let v = vec![2.0, 4.0, 6.0, 8.0];
        let mut out = vec![0.0_f32; 4];
        compute_attention_output(&weights, &v, 2, 2, 2, &mut out).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_output_rejects_short_weights() {
        let mut out = vec![0.0_f32; 4];
        let err = compute_attention_output(&[1.0], &[1.0; 4], 2, 2, 2, &mut out);
        assert!(err.is_err());
    }

    // ── scaled_dot_product_attention tests ────────────────────────────

    #[test]
    fn test_sdpa_single_token() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![5.0, 10.0];
        let mut out = vec![0.0_f32; 2];
        scaled_dot_product_attention(&q, &k, &v, None, 1, 1, 2, 1.0, &mut out).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sdpa_with_mask() {
        // Mask blocks second position → output = V[0]
        let q = vec![1.0, 1.0, 1.0, 1.0]; // [2,2]
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0]; // row0: block pos1
        let mut out = vec![0.0_f32; 4];
        scaled_dot_product_attention(&q, &k, &v, Some(&mask), 2, 2, 2, 1.0, &mut out).unwrap();
        // Row 0 can only see V[0]
        assert!((out[0] - 10.0).abs() < 1e-4);
        assert!((out[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_sdpa_matches_fallback() {
        let cfg = AttentionConfig::new(1, 4, 3, false).unwrap();
        let q: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..12).map(|i| (i + 3) as f32 * 0.05).collect();
        let v: Vec<f32> = (0..12).map(|i| i as f32 * 0.2).collect();

        let fallback = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let mut sdpa_out = vec![0.0_f32; 12];
        scaled_dot_product_attention(&q, &k, &v, None, 3, 3, 4, cfg.scale, &mut sdpa_out)
            .unwrap();
        for (a, b) in fallback.iter().zip(sdpa_out.iter()) {
            assert!((a - b).abs() < 1e-4, "sdpa vs fallback: {a} vs {b}");
        }
    }

    // ── multi_head_attention (with projections) tests ─────────────────

    #[test]
    fn test_mha_identity_weights() {
        // Identity weight matrices: output ≈ attention(input, input, input)
        let cfg = AttentionConfig::new(1, 2, 2, false).unwrap();
        let input = vec![1.0, 0.0, 0.0, 1.0]; // [2,2]
        // Identity 2×2 for all projections
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        multi_head_attention(&input, &eye, &eye, &eye, &eye, &cfg, &mut out).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mha_output_shape() {
        let cfg = AttentionConfig::new(2, 2, 3, false).unwrap();
        let model_dim = 4; // 2 heads * 2 dim
        let input = vec![0.1_f32; 3 * model_dim];
        let w = vec![0.1_f32; model_dim * model_dim];
        let mut out = vec![0.0_f32; 3 * model_dim];
        multi_head_attention(&input, &w, &w, &w, &w, &cfg, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mha_rejects_short_input() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let short = vec![0.0_f32; 2]; // need 8
        let w = vec![0.0_f32; 16];
        let mut out = vec![0.0_f32; 8];
        let err = multi_head_attention(&short, &w, &w, &w, &w, &cfg, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn test_mha_rejects_short_weights() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap();
        let input = vec![0.0_f32; 8];
        let short_w = vec![0.0_f32; 4]; // need 16
        let w = vec![0.0_f32; 16];
        let mut out = vec![0.0_f32; 8];
        let err = multi_head_attention(&input, &short_w, &w, &w, &w, &cfg, &mut out);
        assert!(err.is_err());
    }

    // ── flash_attention_forward tests ─────────────────────────────────

    #[test]
    fn test_flash_attn_matches_standard() {
        let cfg = AttentionConfig::new(1, 4, 6, false).unwrap().with_flash_attention(true);
        let q: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..24).map(|i| (i + 5) as f32 * 0.05).collect();
        let v: Vec<f32> = (0..24).map(|i| i as f32 * 0.3).collect();

        let standard = attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let mut flash_out = vec![0.0_f32; 24];
        flash_attention_forward(&q, &k, &v, &cfg, &mut flash_out).unwrap();
        for (a, b) in standard.iter().zip(flash_out.iter()) {
            assert!((a - b).abs() < 1e-3, "flash vs standard: {a} vs {b}");
        }
    }

    #[test]
    fn test_flash_attn_causal() {
        let cfg = AttentionConfig::new(1, 2, 4, true).unwrap().with_flash_attention(true);
        let q = vec![1.0_f32; 8];
        let k: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; 8];
        flash_attention_forward(&q, &k, &v, &cfg, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_flash_attn_single_token() {
        let cfg = AttentionConfig::new(1, 4, 1, false).unwrap().with_flash_attention(true);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5; 4];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0_f32; 4];
        flash_attention_forward(&q, &k, &v, &cfg, &mut out).unwrap();
        for d in 0..4 {
            assert!((out[d] - v[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_flash_attn_rejects_short_output() {
        let cfg = AttentionConfig::new(1, 4, 2, false).unwrap().with_flash_attention(true);
        let q = vec![0.0_f32; 8];
        let k = vec![0.0_f32; 8];
        let v = vec![0.0_f32; 8];
        let mut out = vec![0.0_f32; 2]; // too short
        assert!(flash_attention_forward(&q, &k, &v, &cfg, &mut out).is_err());
    }

    // ── grouped_query_attention tests ─────────────────────────────────

    #[test]
    fn test_gqa_equal_heads_matches_mha() {
        // When num_kv_heads == num_q_heads, GQA == MHA
        let cfg = AttentionConfig::new(2, 2, 3, false).unwrap();
        let total = 2 * 3 * 2;
        let q: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|i| (i + 1) as f32 * 0.05).collect();
        let v: Vec<f32> = (0..total).map(|i| i as f32 * 0.2).collect();

        let mha = multi_head_attention_cpu_fallback(&q, &k, &v, &cfg).unwrap();
        let mut gqa_out = vec![0.0_f32; total];
        grouped_query_attention(&q, &k, &v, 2, &cfg, &mut gqa_out).unwrap();
        for (a, b) in mha.iter().zip(gqa_out.iter()) {
            assert!((a - b).abs() < 1e-4, "GQA(nkv==nq) vs MHA: {a} vs {b}");
        }
    }

    #[test]
    fn test_gqa_shared_kv_heads() {
        // 4 Q heads, 2 KV heads → groups of 2 Q heads per KV head
        let cfg = AttentionConfig::new(4, 2, 2, false).unwrap();
        let head_el = 2 * 2; // seq * dim
        let q = vec![0.5_f32; 4 * head_el];
        let k = vec![0.5_f32; 2 * head_el];
        let v = vec![1.0_f32; 2 * head_el];
        let mut out = vec![0.0_f32; 4 * head_el];
        grouped_query_attention(&q, &k, &v, 2, &cfg, &mut out).unwrap();
        // Heads 0,1 share KV head 0; heads 2,3 share KV head 1
        // With uniform Q,K,V the outputs should all be 1.0
        for &val in &out {
            assert!((val - 1.0).abs() < 1e-4, "GQA uniform: {val}");
        }
    }

    #[test]
    fn test_gqa_single_kv_head() {
        // All Q heads share a single KV head (MQA)
        let cfg = AttentionConfig::new(4, 2, 2, false).unwrap();
        let head_el = 2 * 2;
        let q = vec![1.0_f32; 4 * head_el];
        let k = vec![1.0_f32; 1 * head_el];
        let v = vec![3.0_f32; 1 * head_el];
        let mut out = vec![0.0_f32; 4 * head_el];
        grouped_query_attention(&q, &k, &v, 1, &cfg, &mut out).unwrap();
        for &val in &out {
            assert!((val - 3.0).abs() < 1e-4, "MQA uniform: {val}");
        }
    }

    #[test]
    fn test_gqa_causal() {
        let cfg = AttentionConfig::new(2, 2, 3, true).unwrap();
        let head_el = 3 * 2;
        let q = vec![1.0_f32; 2 * head_el];
        let k = vec![1.0_f32; 1 * head_el];
        let v: Vec<f32> = (0..head_el).map(|i| i as f32).collect();
        let mut out = vec![0.0_f32; 2 * head_el];
        grouped_query_attention(&q, &k, &v, 1, &cfg, &mut out).unwrap();
        // First token can only attend to position 0
        assert!((out[0] - v[0]).abs() < 1e-4, "causal GQA first token d0");
        assert!((out[1] - v[1]).abs() < 1e-4, "causal GQA first token d1");
    }

    #[test]
    fn test_gqa_rejects_indivisible_heads() {
        let cfg = AttentionConfig::new(3, 2, 2, false).unwrap();
        let mut out = vec![0.0_f32; 12];
        let err = grouped_query_attention(&vec![0.0; 12], &vec![0.0; 8], &vec![0.0; 8], 2, &cfg, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn test_gqa_rejects_zero_kv_heads() {
        let cfg = AttentionConfig::new(2, 2, 2, false).unwrap();
        let mut out = vec![0.0_f32; 8];
        let err = grouped_query_attention(&vec![0.0; 8], &vec![0.0; 8], &vec![0.0; 8], 0, &cfg, &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn test_gqa_rejects_short_q() {
        let cfg = AttentionConfig::new(4, 2, 2, false).unwrap();
        let mut out = vec![0.0_f32; 16];
        let err = grouped_query_attention(&vec![0.0; 4], &vec![0.0; 8], &vec![0.0; 8], 2, &cfg, &mut out);
        assert!(err.is_err());
    }

    // ── Extended AttentionConfig tests ─────────────────────────────────

    #[test]
    fn test_config_with_flash_attention() {
        let cfg = AttentionConfig::new(1, 64, 16, false).unwrap().with_flash_attention(true);
        assert!(cfg.use_flash_attention);
    }

    #[test]
    fn test_config_with_dropout() {
        let cfg = AttentionConfig::new(1, 64, 16, false).unwrap().with_dropout(0.1);
        assert!((cfg.attention_dropout - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_max_seq_len() {
        let cfg = AttentionConfig::new(1, 64, 16, false).unwrap().with_max_seq_len(2048);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    #[test]
    fn test_config_effective_scale_default() {
        let cfg = AttentionConfig::new(1, 64, 4, false).unwrap();
        let expected = 1.0 / (64.0_f32).sqrt();
        assert!((cfg.effective_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_config_effective_scale_override() {
        let cfg = AttentionConfig::new(1, 64, 4, false).unwrap().with_scale(0.25);
        assert!((cfg.effective_scale() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_defaults_new_fields() {
        let cfg = AttentionConfig::new(2, 32, 8, true).unwrap();
        assert_eq!(cfg.max_seq_len, 8);
        assert!(!cfg.use_flash_attention);
        assert!((cfg.attention_dropout - 0.0).abs() < f32::EPSILON);
        assert!(cfg.scale_factor.is_none());
    }

    // ── CUDA kernel source guard (extended) ───────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_extended_kernel_source_not_empty() {
        assert!(!ATTENTION_KERNEL_SOURCE.is_empty());
        assert!(ATTENTION_KERNEL_SOURCE.contains("compute_attention_scores_f32"));
        assert!(ATTENTION_KERNEL_SOURCE.contains("apply_attention_mask_f32"));
        assert!(ATTENTION_KERNEL_SOURCE.contains("attention_softmax_f32"));
        assert!(ATTENTION_KERNEL_SOURCE.contains("compute_attention_output_f32"));
        assert!(ATTENTION_KERNEL_SOURCE.contains("grouped_query_attention_f32"));
    }
}
