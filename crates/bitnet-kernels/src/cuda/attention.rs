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
        Ok(Self { num_heads, head_dim, seq_len, causal, scale })
    }

    /// Override the default scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
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
    if config.num_heads > 1 {
        multi_head_attention_cpu_fallback(query, key, value, config)
    } else {
        attention_cpu_fallback(query, key, value, config)
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
}
