//! CUDA fused attention kernels with FlashAttention-2 and GQA support.
//!
//! This module provides fused attention operations that combine multiple steps
//! (Q·K^T scoring, masking, softmax, V weighting) into single kernel launches
//! to eliminate intermediate global-memory round-trips.
//!
//! # Supported attention patterns
//!
//! - **Causal**: Autoregressive lower-triangular mask
//! - **Full**: No masking (bidirectional)
//! - **Sliding window**: Fixed-size local attention window
//! - **Sparse**: Block-sparse attention with configurable block size
//!
//! # Algorithms
//!
//! - [`fused_attention_forward`]: Standard QKV → output in one fused pass
//! - [`flash_attention_forward`]: FlashAttention-2 style tiled attention with
//!   online softmax for O(seq) memory
//! - [`grouped_query_attention`]: GQA with KV head expansion (fewer KV heads
//!   than query heads)
//! - [`multi_head_attention`]: Standard MHA wrapper dispatching per-head
//!
//! # Positional bias
//!
//! - [`apply_alibi_bias`]: Attention with Linear Biases (ALiBi) for
//!   length-extrapolatable position encoding
//!
//! All GPU code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// Inline CUDA C source for the fused attention kernel.
///
/// Implements a FlashAttention-2 style tiled kernel with:
/// - `fused_attention_f32`: full fused QKV → output
/// - `fused_attention_causal_f32`: causal variant with lower-triangular mask
/// - `fused_gqa_attention_f32`: grouped query attention with KV head expansion
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const FUSED_ATTENTION_KERNEL_SRC: &str = r#"
extern "C" __global__ void fused_attention_f32(
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

extern "C" __global__ void fused_attention_causal_f32(
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

extern "C" __global__ void fused_gqa_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    int num_kv_heads,
    int heads_per_group,
    float scale)
{
    int q_head = blockIdx.y;
    int kv_head = q_head / heads_per_group;
    if (kv_head >= num_kv_heads) kv_head = num_kv_heads - 1;

    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len_q) return;

    const float* q_row = Q + ((long long)q_head * seq_len_q + q_idx) * head_dim;
    const float* k_base = K + (long long)kv_head * seq_len_kv * head_dim;
    const float* v_base = V + (long long)kv_head * seq_len_kv * head_dim;

    float row_max = -1e30f;
    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        const float* k_row = k_base + k_idx * head_dim;
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

    float* o_row = O + ((long long)q_head * seq_len_q + q_idx) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * inv_sum * v_base[k_idx * head_dim + d];
        }
        o_row[d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for fused attention kernels.
#[derive(Debug, Clone)]
pub struct FusedAttentionConfig {
    /// Per-head embedding dimension (typically 64 or 128).
    pub head_dim: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value attention heads (for GQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
    /// Enable FlashAttention-2 style tiled computation.
    pub flash_attention: bool,
    /// Enable ALiBi (Attention with Linear Biases) positional encoding.
    pub use_alibi: bool,
}

impl FusedAttentionConfig {
    /// Create a new fused attention configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `head_dim` is zero or not a power of two
    /// - `num_heads` is zero
    /// - `num_kv_heads` is zero or does not evenly divide `num_heads`
    /// - `max_seq_len` is zero
    pub fn new(
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "FusedAttentionConfig: head_dim must be a non-zero power of two, got {head_dim}"
                ),
            }
            .into());
        }
        if num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "FusedAttentionConfig: num_heads must be non-zero".into(),
            }
            .into());
        }
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "FusedAttentionConfig: num_kv_heads ({num_kv_heads}) must be non-zero \
                     and evenly divide num_heads ({num_heads})"
                ),
            }
            .into());
        }
        if max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "FusedAttentionConfig: max_seq_len must be non-zero".into(),
            }
            .into());
        }
        Ok(Self {
            head_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            causal: false,
            flash_attention: false,
            use_alibi: false,
        })
    }

    /// Enable causal (autoregressive) masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Enable FlashAttention-2 style tiled computation.
    pub fn with_flash_attention(mut self, flash: bool) -> Self {
        self.flash_attention = flash;
        self
    }

    /// Enable ALiBi positional encoding.
    pub fn with_alibi(mut self, alibi: bool) -> Self {
        self.use_alibi = alibi;
        self
    }

    /// Compute the softmax scale factor: `1.0 / sqrt(head_dim)`.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Number of query heads per KV head group.
    pub fn heads_per_kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Whether this configuration uses grouped query attention.
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Attention patterns
// ---------------------------------------------------------------------------

/// Attention masking pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionPattern {
    /// Causal (autoregressive) lower-triangular mask.
    Causal,
    /// Full bidirectional attention (no mask).
    Full,
    /// Sliding window attention with a fixed window size.
    SlidingWindow {
        /// Number of past positions each token can attend to.
        window_size: usize,
    },
    /// Block-sparse attention with configurable block size.
    Sparse {
        /// Size of each attention block.
        block_size: usize,
    },
}

impl AttentionPattern {
    /// Generate an additive attention mask for the given sequence length.
    ///
    /// Returns a `[seq_len, seq_len]` mask where `0.0` means "attend" and
    /// `f32::NEG_INFINITY` means "block".
    pub fn generate_mask(&self, seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; seq_len * seq_len];
        match self {
            AttentionPattern::Full => {} // all zeros
            AttentionPattern::Causal => {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        mask[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            AttentionPattern::SlidingWindow { window_size } => {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i || (i > *window_size && j < i - *window_size) {
                            mask[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            AttentionPattern::Sparse { block_size } => {
                let bs = (*block_size).max(1);
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let block_i = i / bs;
                        let block_j = j / bs;
                        if block_i != block_j && i != j {
                            mask[i * seq_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
        mask
    }

    /// Check whether position `q_pos` can attend to position `kv_pos`.
    pub fn allows(&self, q_pos: usize, kv_pos: usize) -> bool {
        match self {
            AttentionPattern::Full => true,
            AttentionPattern::Causal => kv_pos <= q_pos,
            AttentionPattern::SlidingWindow { window_size } => {
                kv_pos <= q_pos && (q_pos == 0 || kv_pos >= q_pos.saturating_sub(*window_size))
            }
            AttentionPattern::Sparse { block_size } => {
                let bs = (*block_size).max(1);
                q_pos / bs == kv_pos / bs || q_pos == kv_pos
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to fused attention operations.
#[derive(Debug, Clone)]
pub enum FusedAttentionError {
    /// Invalid configuration parameters.
    InvalidConfig(String),
    /// Tensor shape mismatch.
    ShapeMismatch {
        /// Description of the expected shape.
        expected: String,
        /// Description of the actual shape.
        actual: String,
    },
    /// Sequence length exceeds the configured maximum.
    SequenceTooLong {
        /// The actual sequence length.
        seq_len: usize,
        /// The configured maximum.
        max_seq_len: usize,
    },
    /// GQA head ratio is invalid.
    InvalidGqaRatio {
        /// Number of query heads.
        num_heads: usize,
        /// Number of KV heads.
        num_kv_heads: usize,
    },
}

impl fmt::Display for FusedAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedAttentionError::InvalidConfig(msg) => {
                write!(f, "invalid fused attention config: {msg}")
            }
            FusedAttentionError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            FusedAttentionError::SequenceTooLong { seq_len, max_seq_len } => {
                write!(f, "sequence length {seq_len} exceeds maximum {max_seq_len}")
            }
            FusedAttentionError::InvalidGqaRatio { num_heads, num_kv_heads } => {
                write!(
                    f,
                    "invalid GQA ratio: num_heads ({num_heads}) must be divisible \
                     by num_kv_heads ({num_kv_heads})"
                )
            }
        }
    }
}

impl std::error::Error for FusedAttentionError {}

impl From<FusedAttentionError> for bitnet_common::BitNetError {
    fn from(e: FusedAttentionError) -> Self {
        bitnet_common::BitNetError::Kernel(KernelError::InvalidArguments { reason: e.to_string() })
    }
}

// ---------------------------------------------------------------------------
// Attention metrics
// ---------------------------------------------------------------------------

/// Performance metrics for an attention computation.
#[derive(Debug, Clone, Copy)]
pub struct AttentionMetrics {
    /// Total floating-point operations.
    pub flops: u64,
    /// Total memory bytes accessed (reads + writes).
    pub memory_bytes: u64,
    /// Arithmetic intensity (FLOP/byte).
    pub arithmetic_intensity: f64,
}

impl AttentionMetrics {
    /// Compute metrics for a standard attention operation.
    ///
    /// FLOPs: `2 * num_heads * seq_q * seq_kv * head_dim` (QK^T) +
    ///        `2 * num_heads * seq_q * seq_kv * head_dim` (attn × V)
    ///
    /// Memory: reads of Q, K, V + write of O, all in FP32.
    pub fn compute(num_heads: usize, seq_q: usize, seq_kv: usize, head_dim: usize) -> Self {
        let qk_flops = 2u64 * num_heads as u64 * seq_q as u64 * seq_kv as u64 * head_dim as u64;
        let av_flops = 2u64 * num_heads as u64 * seq_q as u64 * seq_kv as u64 * head_dim as u64;
        let flops = qk_flops + av_flops;

        let q_bytes = (num_heads * seq_q * head_dim * 4) as u64;
        let k_bytes = (num_heads * seq_kv * head_dim * 4) as u64;
        let v_bytes = (num_heads * seq_kv * head_dim * 4) as u64;
        let o_bytes = (num_heads * seq_q * head_dim * 4) as u64;
        let memory_bytes = q_bytes + k_bytes + v_bytes + o_bytes;

        let arithmetic_intensity =
            if memory_bytes > 0 { flops as f64 / memory_bytes as f64 } else { 0.0 };

        Self { flops, memory_bytes, arithmetic_intensity }
    }

    /// Compute metrics for GQA (fewer KV heads).
    pub fn compute_gqa(
        num_heads: usize,
        num_kv_heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
    ) -> Self {
        let qk_flops = 2u64 * num_heads as u64 * seq_q as u64 * seq_kv as u64 * head_dim as u64;
        let av_flops = 2u64 * num_heads as u64 * seq_q as u64 * seq_kv as u64 * head_dim as u64;
        let flops = qk_flops + av_flops;

        let q_bytes = (num_heads * seq_q * head_dim * 4) as u64;
        let k_bytes = (num_kv_heads * seq_kv * head_dim * 4) as u64;
        let v_bytes = (num_kv_heads * seq_kv * head_dim * 4) as u64;
        let o_bytes = (num_heads * seq_q * head_dim * 4) as u64;
        let memory_bytes = q_bytes + k_bytes + v_bytes + o_bytes;

        let arithmetic_intensity =
            if memory_bytes > 0 { flops as f64 / memory_bytes as f64 } else { 0.0 };

        Self { flops, memory_bytes, arithmetic_intensity }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Numerically stable row-wise softmax in-place.
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

/// Apply ALiBi bias to a mutable score slice.
fn apply_alibi_to_scores(scores: &mut [f32], seq_len: usize, i: usize, slope: f32) {
    for (j, score) in scores.iter_mut().enumerate().take(seq_len) {
        if score.is_finite() {
            *score += slope * (j as f32 - i as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// Core attention functions
// ---------------------------------------------------------------------------

/// Compute attention scores: `Q · K^T / sqrt(d_k)`.
///
/// # Arguments
///
/// * `query` — `[seq_q, head_dim]` (FP32, row-major)
/// * `key`   — `[seq_kv, head_dim]` (FP32, row-major)
/// * `seq_q`  — Query sequence length
/// * `seq_kv` — Key sequence length
/// * `head_dim` — Per-head dimension
///
/// # Returns
///
/// Score matrix `[seq_q, seq_kv]` as a flat `Vec<f32>`.
pub fn compute_attention_scores(
    query: &[f32],
    key: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if head_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "compute_attention_scores: head_dim must be non-zero".into(),
        }
        .into());
    }
    let q_expected = seq_q * head_dim;
    let k_expected = seq_kv * head_dim;
    if query.len() < q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "compute_attention_scores: query length {}, expected {q_expected}",
                query.len()
            ),
        }
        .into());
    }
    if key.len() < k_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "compute_attention_scores: key length {}, expected {k_expected}",
                key.len()
            ),
        }
        .into());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0_f32; seq_q * seq_kv];

    for i in 0..seq_q {
        for j in 0..seq_kv {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += query[i * head_dim + d] * key[j * head_dim + d];
            }
            scores[i * seq_kv + j] = dot * scale;
        }
    }

    Ok(scores)
}

/// Apply an attention mask to pre-computed scores.
///
/// Adds the mask values to scores in-place. Use `0.0` for "attend" and
/// `f32::NEG_INFINITY` for "block".
pub fn apply_attention_mask(scores: &mut [f32], mask: &[f32]) -> Result<()> {
    if scores.len() != mask.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "apply_attention_mask: scores length {} != mask length {}",
                scores.len(),
                mask.len()
            ),
        }
        .into());
    }
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        *s += m;
    }
    Ok(())
}

/// Apply ALiBi (Attention with Linear Biases) positional bias to scores.
///
/// ALiBi adds a linear penalty `m * (j - i)` for each head, where `m` is a
/// head-specific slope: `m_h = 2^(-8*(h+1)/n)`.
pub fn apply_alibi_bias(
    scores: &mut [f32],
    seq_q: usize,
    seq_kv: usize,
    head_idx: usize,
    num_heads: usize,
) -> Result<()> {
    if num_heads == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "apply_alibi_bias: num_heads must be non-zero".into(),
        }
        .into());
    }
    let expected = seq_q * seq_kv;
    if scores.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "apply_alibi_bias: scores length {}, expected {expected}",
                scores.len()
            ),
        }
        .into());
    }

    let slope = 2.0_f32.powf(-8.0 * (head_idx as f32 + 1.0) / num_heads as f32);

    for i in 0..seq_q {
        for j in 0..seq_kv {
            let distance = j as f32 - i as f32;
            scores[i * seq_kv + j] += slope * distance;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Fused attention forward
// ---------------------------------------------------------------------------

/// Fused attention forward pass: QKV → attention output in one CPU kernel.
///
/// Computes `softmax(Q·K^T / sqrt(d_k) + mask) · V` with optional causal
/// masking and ALiBi bias.
pub fn fused_attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &FusedAttentionConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    if seq_len > config.max_seq_len {
        return Err(FusedAttentionError::SequenceTooLong {
            seq_len,
            max_seq_len: config.max_seq_len,
        }
        .into());
    }
    if seq_len == 0 {
        return Ok(vec![]);
    }

    let head_size = seq_len * config.head_dim;
    let q_expected = config.num_heads * head_size;
    let kv_expected = config.num_kv_heads * head_size;

    if query.len() < q_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_heads, config.head_dim),
            actual: format!("query length {}", query.len()),
        }
        .into());
    }
    if key.len() < kv_expected || value.len() < kv_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_kv_heads, config.head_dim),
            actual: format!("key length {}, value length {}", key.len(), value.len()),
        }
        .into());
    }

    if config.is_gqa() {
        return grouped_query_attention(query, key, value, config, seq_len);
    }

    let scale = config.scale();
    let mut output = vec![0.0_f32; q_expected];

    for h in 0..config.num_heads {
        let q_offset = h * head_size;
        let kv_offset = h * head_size;

        for i in 0..seq_len {
            let mut scores = vec![0.0_f32; seq_len];

            for j in 0..seq_len {
                if config.causal && j > i {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for d in 0..config.head_dim {
                    dot += query[q_offset + i * config.head_dim + d]
                        * key[kv_offset + j * config.head_dim + d];
                }
                scores[j] = dot * scale;
            }

            if config.use_alibi {
                let slope = 2.0_f32.powf(-8.0 * (h as f32 + 1.0) / config.num_heads as f32);
                apply_alibi_to_scores(&mut scores, seq_len, i, slope);
            }

            softmax_inplace(&mut scores);

            for d in 0..config.head_dim {
                let mut acc = 0.0_f32;
                for j in 0..seq_len {
                    acc += scores[j] * value[kv_offset + j * config.head_dim + d];
                }
                output[q_offset + i * config.head_dim + d] = acc;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Flash attention forward
// ---------------------------------------------------------------------------

/// Default tile size for the flash attention chunked computation.
const FLASH_TILE_SIZE: usize = 64;

/// FlashAttention-2 style tiled attention (CPU reference).
///
/// Uses online softmax to process K/V in tiles, keeping memory usage
/// at `O(seq_q * tile_size)` instead of `O(seq_q * seq_kv)`.
pub fn flash_attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &FusedAttentionConfig,
    seq_len: usize,
    tile_size: usize,
) -> Result<Vec<f32>> {
    if seq_len > config.max_seq_len {
        return Err(FusedAttentionError::SequenceTooLong {
            seq_len,
            max_seq_len: config.max_seq_len,
        }
        .into());
    }
    if seq_len == 0 {
        return Ok(vec![]);
    }

    let head_size = seq_len * config.head_dim;
    let q_expected = config.num_heads * head_size;
    let kv_expected = config.num_kv_heads * head_size;

    if query.len() < q_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_heads, config.head_dim),
            actual: format!("query length {}", query.len()),
        }
        .into());
    }
    if key.len() < kv_expected || value.len() < kv_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_kv_heads, config.head_dim),
            actual: format!("key length {}, value length {}", key.len(), value.len()),
        }
        .into());
    }

    let ts = if tile_size == 0 { FLASH_TILE_SIZE } else { tile_size };
    let scale = config.scale();
    let heads_per_group = config.heads_per_kv_group();
    let mut output = vec![0.0_f32; q_expected];

    for h in 0..config.num_heads {
        let q_offset = h * head_size;
        let kv_head = h / heads_per_group;
        let kv_offset = kv_head * head_size;

        for i in 0..seq_len {
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0_f32;
            let mut acc = vec![0.0_f32; config.head_dim];

            let kv_len = if config.causal { i + 1 } else { seq_len };

            let mut tile_start = 0;
            while tile_start < kv_len {
                let tile_end = (tile_start + ts).min(kv_len);
                let tile_len = tile_end - tile_start;

                let mut tile_scores = vec![0.0_f32; tile_len];
                let mut tile_max = f32::NEG_INFINITY;

                for (ti, j) in (tile_start..tile_end).enumerate() {
                    let mut dot = 0.0_f32;
                    for d in 0..config.head_dim {
                        dot += query[q_offset + i * config.head_dim + d]
                            * key[kv_offset + j * config.head_dim + d];
                    }
                    let mut score = dot * scale;

                    if config.use_alibi {
                        let slope = 2.0_f32.powf(-8.0 * (h as f32 + 1.0) / config.num_heads as f32);
                        score += slope * (j as f32 - i as f32);
                    }

                    tile_scores[ti] = score;
                    if score > tile_max {
                        tile_max = score;
                    }
                }

                // Online softmax merge
                let new_max = running_max.max(tile_max);
                if running_sum > 0.0 {
                    let correction = (running_max - new_max).exp();
                    running_sum *= correction;
                    for a in acc.iter_mut() {
                        *a *= correction;
                    }
                }

                let mut tile_sum = 0.0_f32;
                for (ti, &score) in tile_scores.iter().enumerate().take(tile_len) {
                    let w = (score - new_max).exp();
                    tile_sum += w;
                    let j = tile_start + ti;
                    for d in 0..config.head_dim {
                        acc[d] += w * value[kv_offset + j * config.head_dim + d];
                    }
                }

                running_max = new_max;
                running_sum += tile_sum;
                tile_start = tile_end;
            }

            if running_sum > 0.0 {
                let inv = 1.0 / running_sum;
                for d in 0..config.head_dim {
                    output[q_offset + i * config.head_dim + d] = acc[d] * inv;
                }
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Grouped query attention
// ---------------------------------------------------------------------------

/// Grouped Query Attention (GQA) with KV head expansion.
///
/// When `num_kv_heads < num_heads`, multiple query heads share the same
/// KV head. This reduces KV cache memory by the group ratio.
pub fn grouped_query_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &FusedAttentionConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    if seq_len > config.max_seq_len {
        return Err(FusedAttentionError::SequenceTooLong {
            seq_len,
            max_seq_len: config.max_seq_len,
        }
        .into());
    }
    if seq_len == 0 {
        return Ok(vec![]);
    }

    let head_size = seq_len * config.head_dim;
    let q_expected = config.num_heads * head_size;
    let kv_expected = config.num_kv_heads * head_size;

    if query.len() < q_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_heads, config.head_dim),
            actual: format!("query length {}", query.len()),
        }
        .into());
    }
    if key.len() < kv_expected || value.len() < kv_expected {
        return Err(FusedAttentionError::ShapeMismatch {
            expected: format!("[{}, {seq_len}, {}]", config.num_kv_heads, config.head_dim),
            actual: format!("key length {}, value length {}", key.len(), value.len()),
        }
        .into());
    }

    let scale = config.scale();
    let heads_per_group = config.heads_per_kv_group();
    let mut output = vec![0.0_f32; q_expected];

    for h in 0..config.num_heads {
        let q_offset = h * head_size;
        let kv_head = h / heads_per_group;
        let kv_offset = kv_head * head_size;

        for i in 0..seq_len {
            let mut scores = vec![0.0_f32; seq_len];

            for j in 0..seq_len {
                if config.causal && j > i {
                    scores[j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for d in 0..config.head_dim {
                    dot += query[q_offset + i * config.head_dim + d]
                        * key[kv_offset + j * config.head_dim + d];
                }
                scores[j] = dot * scale;
            }

            if config.use_alibi {
                let slope = 2.0_f32.powf(-8.0 * (h as f32 + 1.0) / config.num_heads as f32);
                apply_alibi_to_scores(&mut scores, seq_len, i, slope);
            }

            softmax_inplace(&mut scores);

            for d in 0..config.head_dim {
                let mut acc = 0.0_f32;
                for j in 0..seq_len {
                    acc += scores[j] * value[kv_offset + j * config.head_dim + d];
                }
                output[q_offset + i * config.head_dim + d] = acc;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Multi-head attention wrapper
// ---------------------------------------------------------------------------

/// Standard multi-head attention wrapper.
///
/// Dispatches to [`fused_attention_forward`] for MHA or
/// [`grouped_query_attention`] for GQA based on the configuration.
pub fn multi_head_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &FusedAttentionConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    fused_attention_forward(query, key, value, config, seq_len)
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the fused attention CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_fused_attention(
    _query: &[f32],
    _key: &[f32],
    _value: &[f32],
    _output: &mut [f32],
    config: &FusedAttentionConfig,
    seq_len: usize,
) -> Result<()> {
    log::debug!(
        "Fused attention stub: heads={}, kv_heads={}, head_dim={}, seq={}, causal={}, gqa={}",
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        seq_len,
        config.causal,
        config.is_gqa(),
    );
    Err(KernelError::GpuError {
        reason: "Fused attention CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── FusedAttentionConfig tests ────────────────────────────────────

    #[test]
    fn test_config_valid_mha() {
        let cfg = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.max_seq_len, 2048);
        assert!(!cfg.causal);
        assert!(!cfg.flash_attention);
        assert!(!cfg.use_alibi);
    }

    #[test]
    fn test_config_valid_gqa() {
        let cfg = FusedAttentionConfig::new(128, 32, 8, 4096).unwrap();
        assert_eq!(cfg.heads_per_kv_group(), 4);
        assert!(cfg.is_gqa());
    }

    #[test]
    fn test_config_rejects_zero_head_dim() {
        assert!(FusedAttentionConfig::new(0, 8, 8, 2048).is_err());
    }

    #[test]
    fn test_config_rejects_non_power_of_two_head_dim() {
        assert!(FusedAttentionConfig::new(48, 8, 8, 2048).is_err());
        assert!(FusedAttentionConfig::new(65, 8, 8, 2048).is_err());
        assert!(FusedAttentionConfig::new(100, 8, 8, 2048).is_err());
    }

    #[test]
    fn test_config_accepts_power_of_two_head_dims() {
        for dim in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            assert!(FusedAttentionConfig::new(dim, 8, 8, 2048).is_ok(), "dim={dim}");
        }
    }

    #[test]
    fn test_config_rejects_zero_num_heads() {
        assert!(FusedAttentionConfig::new(64, 0, 0, 2048).is_err());
    }

    #[test]
    fn test_config_rejects_zero_kv_heads() {
        assert!(FusedAttentionConfig::new(64, 8, 0, 2048).is_err());
    }

    #[test]
    fn test_config_rejects_indivisible_kv_heads() {
        assert!(FusedAttentionConfig::new(64, 8, 3, 2048).is_err());
        assert!(FusedAttentionConfig::new(64, 8, 5, 2048).is_err());
        assert!(FusedAttentionConfig::new(64, 12, 7, 2048).is_err());
    }

    #[test]
    fn test_config_accepts_valid_kv_head_ratios() {
        assert!(FusedAttentionConfig::new(64, 8, 1, 2048).is_ok()); // MQA
        assert!(FusedAttentionConfig::new(64, 8, 2, 2048).is_ok()); // GQA 4:1
        assert!(FusedAttentionConfig::new(64, 8, 4, 2048).is_ok()); // GQA 2:1
        assert!(FusedAttentionConfig::new(64, 8, 8, 2048).is_ok()); // MHA
    }

    #[test]
    fn test_config_rejects_zero_max_seq_len() {
        assert!(FusedAttentionConfig::new(64, 8, 8, 0).is_err());
    }

    #[test]
    fn test_config_scale() {
        let cfg = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        assert!((cfg.scale() - 1.0 / 8.0).abs() < 1e-6); // 1/sqrt(64)=1/8

        let cfg128 = FusedAttentionConfig::new(128, 8, 8, 2048).unwrap();
        assert!((cfg128.scale() - 1.0 / (128.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_config_builder_pattern() {
        let cfg = FusedAttentionConfig::new(64, 8, 8, 2048)
            .unwrap()
            .with_causal(true)
            .with_flash_attention(true)
            .with_alibi(true);
        assert!(cfg.causal);
        assert!(cfg.flash_attention);
        assert!(cfg.use_alibi);
    }

    #[test]
    fn test_config_is_gqa() {
        let mha = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        assert!(!mha.is_gqa());

        let gqa = FusedAttentionConfig::new(64, 8, 2, 2048).unwrap();
        assert!(gqa.is_gqa());

        let mqa = FusedAttentionConfig::new(64, 8, 1, 2048).unwrap();
        assert!(mqa.is_gqa());
    }

    #[test]
    fn test_config_heads_per_kv_group() {
        let cfg = FusedAttentionConfig::new(64, 32, 8, 2048).unwrap();
        assert_eq!(cfg.heads_per_kv_group(), 4);

        let mha = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        assert_eq!(mha.heads_per_kv_group(), 1);

        let mqa = FusedAttentionConfig::new(64, 16, 1, 2048).unwrap();
        assert_eq!(mqa.heads_per_kv_group(), 16);
    }

    // ── AttentionPattern tests ────────────────────────────────────────

    #[test]
    fn test_pattern_full_mask() {
        let mask = AttentionPattern::Full.generate_mask(4);
        assert_eq!(mask.len(), 16);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pattern_causal_mask() {
        let mask = AttentionPattern::Causal.generate_mask(4);
        assert_eq!(mask[0 * 4 + 0], 0.0);
        assert_eq!(mask[0 * 4 + 1], f32::NEG_INFINITY);
        assert_eq!(mask[1 * 4 + 0], 0.0);
        assert_eq!(mask[1 * 4 + 1], 0.0);
        assert_eq!(mask[1 * 4 + 2], f32::NEG_INFINITY);
        assert_eq!(mask[3 * 4 + 3], 0.0);
    }

    #[test]
    fn test_pattern_causal_mask_seq1() {
        let mask = AttentionPattern::Causal.generate_mask(1);
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn test_pattern_sliding_window_mask() {
        let mask = AttentionPattern::SlidingWindow { window_size: 1 }.generate_mask(4);
        assert_eq!(mask[0 * 4 + 0], 0.0);
        assert_eq!(mask[0 * 4 + 1], f32::NEG_INFINITY);
        assert_eq!(mask[2 * 4 + 0], f32::NEG_INFINITY);
        assert_eq!(mask[2 * 4 + 1], 0.0);
        assert_eq!(mask[2 * 4 + 2], 0.0);
        assert_eq!(mask[2 * 4 + 3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_pattern_sparse_mask() {
        let mask = AttentionPattern::Sparse { block_size: 2 }.generate_mask(4);
        assert_eq!(mask[0 * 4 + 0], 0.0);
        assert_eq!(mask[0 * 4 + 1], 0.0);
        assert_eq!(mask[0 * 4 + 2], f32::NEG_INFINITY);
        assert_eq!(mask[0 * 4 + 3], f32::NEG_INFINITY);
        assert_eq!(mask[2 * 4 + 2], 0.0);
        assert_eq!(mask[2 * 4 + 3], 0.0);
    }

    #[test]
    fn test_pattern_allows_full() {
        let p = AttentionPattern::Full;
        assert!(p.allows(0, 5));
        assert!(p.allows(5, 0));
        assert!(p.allows(10, 10));
    }

    #[test]
    fn test_pattern_allows_causal() {
        let p = AttentionPattern::Causal;
        assert!(p.allows(5, 3));
        assert!(p.allows(5, 5));
        assert!(!p.allows(3, 5));
    }

    #[test]
    fn test_pattern_allows_sliding_window() {
        let p = AttentionPattern::SlidingWindow { window_size: 2 };
        assert!(p.allows(5, 5));
        assert!(p.allows(5, 4));
        assert!(p.allows(5, 3));
        assert!(!p.allows(5, 2));
        assert!(!p.allows(5, 6));
    }

    #[test]
    fn test_pattern_allows_sparse() {
        let p = AttentionPattern::Sparse { block_size: 3 };
        assert!(p.allows(0, 2));
        assert!(p.allows(1, 0));
        assert!(!p.allows(0, 3));
        assert!(p.allows(3, 5));
        assert!(p.allows(5, 5));
    }

    #[test]
    fn test_pattern_empty_mask() {
        let mask = AttentionPattern::Full.generate_mask(0);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_pattern_eq() {
        assert_eq!(AttentionPattern::Causal, AttentionPattern::Causal);
        assert_ne!(AttentionPattern::Causal, AttentionPattern::Full);
        assert_eq!(
            AttentionPattern::SlidingWindow { window_size: 5 },
            AttentionPattern::SlidingWindow { window_size: 5 }
        );
        assert_ne!(
            AttentionPattern::SlidingWindow { window_size: 5 },
            AttentionPattern::SlidingWindow { window_size: 3 }
        );
    }

    // ── FusedAttentionError tests ─────────────────────────────────────

    #[test]
    fn test_error_display_invalid_config() {
        let e = FusedAttentionError::InvalidConfig("bad param".into());
        assert!(e.to_string().contains("bad param"));
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = FusedAttentionError::ShapeMismatch {
            expected: "[8,16,64]".into(),
            actual: "[8,16,32]".into(),
        };
        let s = e.to_string();
        assert!(s.contains("[8,16,64]"));
        assert!(s.contains("[8,16,32]"));
    }

    #[test]
    fn test_error_display_seq_too_long() {
        let e = FusedAttentionError::SequenceTooLong { seq_len: 8192, max_seq_len: 4096 };
        let s = e.to_string();
        assert!(s.contains("8192"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn test_error_display_invalid_gqa() {
        let e = FusedAttentionError::InvalidGqaRatio { num_heads: 8, num_kv_heads: 3 };
        let s = e.to_string();
        assert!(s.contains('8'));
        assert!(s.contains('3'));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(FusedAttentionError::InvalidConfig("test".into()));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_conversion_to_bitnet_error() {
        let e = FusedAttentionError::InvalidConfig("test conversion".into());
        let bitnet_err: bitnet_common::BitNetError = e.into();
        assert!(bitnet_err.to_string().contains("test conversion"));
    }

    // ── compute_attention_scores tests ────────────────────────────────

    #[test]
    fn test_scores_identity_keys() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = compute_attention_scores(&q, &k, 2, 2, 2).unwrap();
        assert_eq!(scores.len(), 4);
        let scale = 1.0 / (2.0_f32).sqrt();
        assert!((scores[0] - 1.0 * scale).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scores_shape() {
        let q = vec![0.1_f32; 12];
        let k = vec![0.2_f32; 20];
        let scores = compute_attention_scores(&q, &k, 3, 5, 4).unwrap();
        assert_eq!(scores.len(), 15);
    }

    #[test]
    fn test_scores_scaling() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let scores = compute_attention_scores(&q, &k, 1, 1, 4).unwrap();
        assert!((scores[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scores_rejects_zero_head_dim() {
        assert!(compute_attention_scores(&[1.0], &[1.0], 1, 1, 0).is_err());
    }

    #[test]
    fn test_scores_rejects_short_query() {
        let q = vec![0.0_f32; 2];
        let k = vec![0.0_f32; 4];
        assert!(compute_attention_scores(&q, &k, 2, 2, 2).is_err());
    }

    #[test]
    fn test_scores_rejects_short_key() {
        let q = vec![0.0_f32; 4];
        let k = vec![0.0_f32; 2];
        assert!(compute_attention_scores(&q, &k, 2, 2, 2).is_err());
    }

    #[test]
    fn test_scores_orthogonal_vectors() {
        let q = vec![1.0, 0.0];
        let k = vec![0.0, 1.0];
        let scores = compute_attention_scores(&q, &k, 1, 1, 2).unwrap();
        assert!((scores[0]).abs() < 1e-6);
    }

    // ── apply_attention_mask tests ────────────────────────────────────

    #[test]
    fn test_mask_application() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        apply_attention_mask(&mut scores, &mask).unwrap();
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], 3.0);
        assert_eq!(scores[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_mask_zero_mask_preserves_scores() {
        let mut scores = vec![1.0, 2.0, 3.0];
        let mask = vec![0.0, 0.0, 0.0];
        apply_attention_mask(&mut scores, &mask).unwrap();
        assert_eq!(scores, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mask_rejects_length_mismatch() {
        let mut scores = vec![1.0, 2.0];
        let mask = vec![0.0, 0.0, 0.0];
        assert!(apply_attention_mask(&mut scores, &mask).is_err());
    }

    // ── apply_alibi_bias tests ────────────────────────────────────────

    #[test]
    fn test_alibi_self_position_zero_bias() {
        let mut scores = vec![0.0_f32; 4];
        apply_alibi_bias(&mut scores, 2, 2, 0, 8).unwrap();
        assert!((scores[0]).abs() < 1e-6);
        assert!((scores[3]).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_future_positions_negative() {
        let mut scores = vec![0.0_f32; 4];
        apply_alibi_bias(&mut scores, 2, 2, 0, 8).unwrap();
        assert!((scores[0 * 2 + 1] - 0.5).abs() < 1e-6);
        assert!((scores[1 * 2 + 0] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_different_heads_different_slopes() {
        let mut scores0 = vec![0.0_f32; 4];
        let mut scores1 = vec![0.0_f32; 4];
        apply_alibi_bias(&mut scores0, 2, 2, 0, 4).unwrap();
        apply_alibi_bias(&mut scores1, 2, 2, 1, 4).unwrap();
        assert!((scores0[1] - scores1[1]).abs() > 1e-6);
    }

    #[test]
    fn test_alibi_rejects_zero_heads() {
        let mut scores = vec![0.0_f32; 4];
        assert!(apply_alibi_bias(&mut scores, 2, 2, 0, 0).is_err());
    }

    #[test]
    fn test_alibi_rejects_short_scores() {
        let mut scores = vec![0.0_f32; 2];
        assert!(apply_alibi_bias(&mut scores, 2, 2, 0, 8).is_err());
    }

    #[test]
    fn test_alibi_slopes_decrease_with_head_index() {
        let mut scores0 = vec![0.0_f32; 4];
        let mut scores3 = vec![0.0_f32; 4];
        apply_alibi_bias(&mut scores0, 2, 2, 0, 8).unwrap();
        apply_alibi_bias(&mut scores3, 2, 2, 3, 8).unwrap();
        assert!(scores0[1].abs() > scores3[1].abs());
    }

    // ── fused_attention_forward tests ─────────────────────────────────

    #[test]
    fn test_fused_attn_uniform_qkv() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let size = 2 * 4 * 4;
        let q = vec![1.0_f32; size];
        let k = vec![1.0_f32; size];
        let v = vec![3.0_f32; size];
        let out = fused_attention_forward(&q, &k, &v, &cfg, 4).unwrap();
        assert_eq!(out.len(), size);
        for &val in &out {
            assert!((val - 3.0).abs() < 1e-4, "uniform: {val}");
        }
    }

    #[test]
    fn test_fused_attn_seq_len_1() {
        let cfg = FusedAttentionConfig::new(8, 4, 4, 1024).unwrap();
        let size = 4 * 1 * 8;
        let q = vec![1.0_f32; size];
        let k = vec![0.5_f32; size];
        let v = vec![7.0_f32; size];
        let out = fused_attention_forward(&q, &k, &v, &cfg, 1).unwrap();
        assert_eq!(out.len(), size);
        for &val in &out {
            assert!((val - 7.0).abs() < 1e-5, "seq=1 should return V");
        }
    }

    #[test]
    fn test_fused_attn_seq_len_0() {
        let cfg = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        let out = fused_attention_forward(&[], &[], &[], &cfg, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_fused_attn_causal() {
        let cfg = FusedAttentionConfig::new(2, 1, 1, 128).unwrap().with_causal(true);
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = fused_attention_forward(&q, &k, &v, &cfg, 3).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-4);
        assert!((out[1] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_fused_attn_rejects_seq_too_long() {
        let cfg = FusedAttentionConfig::new(4, 1, 1, 8).unwrap();
        let data = vec![0.0_f32; 1 * 16 * 4];
        assert!(fused_attention_forward(&data, &data, &data, &cfg, 16).is_err());
    }

    #[test]
    fn test_fused_attn_rejects_short_query() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let short = vec![0.0_f32; 4];
        let ok = vec![0.0_f32; 32];
        assert!(fused_attention_forward(&short, &ok, &ok, &cfg, 4).is_err());
    }

    #[test]
    fn test_fused_attn_rejects_short_key() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let ok = vec![0.0_f32; 32];
        let short = vec![0.0_f32; 4];
        assert!(fused_attention_forward(&ok, &short, &ok, &cfg, 4).is_err());
    }

    #[test]
    fn test_fused_attn_output_finite() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let q: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..32).map(|i| (i as f32 + 5.0) * 0.05).collect();
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = fused_attention_forward(&q, &k, &v, &cfg, 4).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fused_attn_with_alibi() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap().with_alibi(true);
        let size = 2 * 4 * 4;
        let q = vec![1.0_f32; size];
        let k = vec![1.0_f32; size];
        let v: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let out = fused_attention_forward(&q, &k, &v, &cfg, 4).unwrap();
        assert_eq!(out.len(), size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fused_attn_alibi_vs_no_alibi() {
        let cfg_no = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let cfg_alibi = FusedAttentionConfig::new(4, 2, 2, 128).unwrap().with_alibi(true);
        let size = 2 * 4 * 4;
        let q: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let out_no = fused_attention_forward(&q, &k, &v, &cfg_no, 4).unwrap();
        let out_alibi = fused_attention_forward(&q, &k, &v, &cfg_alibi, 4).unwrap();

        let diff: f32 = out_no.iter().zip(out_alibi.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "ALiBi should change outputs");
    }

    // ── flash_attention_forward tests ─────────────────────────────────

    #[test]
    fn test_flash_matches_fused_noncausal() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 256).unwrap();
        let size = 2 * 8 * 4;
        let q: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..size).map(|i| ((i + 3) as f32) * 0.05).collect();
        let v: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();

        let fused = fused_attention_forward(&q, &k, &v, &cfg, 8).unwrap();
        let flash = flash_attention_forward(&q, &k, &v, &cfg, 8, 3).unwrap();

        assert_eq!(fused.len(), flash.len());
        for (a, b) in fused.iter().zip(flash.iter()) {
            assert!((a - b).abs() < 1e-3, "flash vs fused mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_flash_matches_fused_causal() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 256).unwrap().with_causal(true);
        let size = 2 * 6 * 4;
        let q = vec![1.0_f32; size];
        let k: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let fused = fused_attention_forward(&q, &k, &v, &cfg, 6).unwrap();
        let flash = flash_attention_forward(&q, &k, &v, &cfg, 6, 2).unwrap();

        for (a, b) in fused.iter().zip(flash.iter()) {
            assert!((a - b).abs() < 1e-3, "flash causal mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_flash_default_tile_size() {
        let cfg = FusedAttentionConfig::new(4, 1, 1, 256).unwrap();
        let size = 1 * 4 * 4;
        let q = vec![1.0_f32; size];
        let k = vec![1.0_f32; size];
        let v = vec![2.0_f32; size];
        let out = flash_attention_forward(&q, &k, &v, &cfg, 4, 0).unwrap();
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_flash_seq_len_0() {
        let cfg = FusedAttentionConfig::new(64, 8, 8, 2048).unwrap();
        let out = flash_attention_forward(&[], &[], &[], &cfg, 0, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_flash_rejects_seq_too_long() {
        let cfg = FusedAttentionConfig::new(4, 1, 1, 4).unwrap();
        let data = vec![0.0_f32; 1 * 8 * 4];
        assert!(flash_attention_forward(&data, &data, &data, &cfg, 8, 0).is_err());
    }

    // ── grouped_query_attention tests ─────────────────────────────────

    #[test]
    fn test_gqa_4_to_1_ratio() {
        let cfg = FusedAttentionConfig::new(4, 8, 2, 128).unwrap();
        let seq = 4;
        let q_size = 8 * seq * 4;
        let kv_size = 2 * seq * 4;
        let q = vec![1.0_f32; q_size];
        let k = vec![1.0_f32; kv_size];
        let v = vec![5.0_f32; kv_size];
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq).unwrap();
        assert_eq!(out.len(), q_size);
        for &val in &out {
            assert!((val - 5.0).abs() < 1e-4, "GQA uniform: {val}");
        }
    }

    #[test]
    fn test_gqa_same_kv_group_same_output() {
        let cfg = FusedAttentionConfig::new(4, 4, 1, 128).unwrap();
        let seq = 2;
        let q_size = 4 * seq * 4;
        let kv_size = 1 * seq * 4;
        let q = vec![1.0_f32; q_size];
        let k = vec![0.5_f32; kv_size];
        let v = vec![3.0_f32; kv_size];
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq).unwrap();
        let head_size = seq * 4;
        for h in 1..4 {
            let h0 = &out[0..head_size];
            let hx = &out[h * head_size..(h + 1) * head_size];
            for (a, b) in h0.iter().zip(hx.iter()) {
                assert!((a - b).abs() < 1e-5, "heads sharing KV group should match");
            }
        }
    }

    #[test]
    fn test_gqa_mha_equivalence() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 128).unwrap();
        let seq = 3;
        let size = 2 * seq * 4;
        let q: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let mha = fused_attention_forward(&q, &k, &v, &cfg, seq).unwrap();
        let gqa = grouped_query_attention(&q, &k, &v, &cfg, seq).unwrap();

        for (a, b) in mha.iter().zip(gqa.iter()) {
            assert!((a - b).abs() < 1e-5, "MHA vs GQA (same heads): {a} vs {b}");
        }
    }

    #[test]
    fn test_gqa_seq_len_0() {
        let cfg = FusedAttentionConfig::new(64, 8, 2, 2048).unwrap();
        let out = grouped_query_attention(&[], &[], &[], &cfg, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_gqa_rejects_short_query() {
        let cfg = FusedAttentionConfig::new(4, 4, 2, 128).unwrap();
        let short = vec![0.0_f32; 4];
        let ok_kv = vec![0.0_f32; 2 * 4 * 4];
        assert!(grouped_query_attention(&short, &ok_kv, &ok_kv, &cfg, 4).is_err());
    }

    #[test]
    fn test_gqa_causal() {
        let cfg = FusedAttentionConfig::new(4, 4, 2, 128).unwrap().with_causal(true);
        let seq = 3;
        let q_size = 4 * seq * 4;
        let kv_size = 2 * seq * 4;
        let q = vec![1.0_f32; q_size];
        let k = vec![1.0_f32; kv_size];
        let v: Vec<f32> = (0..kv_size).map(|i| i as f32).collect();
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq).unwrap();
        assert_eq!(out.len(), q_size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── multi_head_attention wrapper tests ────────────────────────────

    #[test]
    fn test_mha_wrapper_matches_fused() {
        let cfg = FusedAttentionConfig::new(4, 2, 2, 256).unwrap();
        let size = 2 * 4 * 4;
        let q: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..size).map(|i| (i + 2) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let fused = fused_attention_forward(&q, &k, &v, &cfg, 4).unwrap();
        let mha = multi_head_attention(&q, &k, &v, &cfg, 4).unwrap();

        for (a, b) in fused.iter().zip(mha.iter()) {
            assert!((a - b).abs() < 1e-6, "MHA wrapper should match fused");
        }
    }

    #[test]
    fn test_mha_dispatches_to_gqa() {
        let cfg = FusedAttentionConfig::new(4, 8, 2, 128).unwrap();
        let seq = 2;
        let q_size = 8 * seq * 4;
        let kv_size = 2 * seq * 4;
        let q = vec![1.0_f32; q_size];
        let k = vec![1.0_f32; kv_size];
        let v = vec![4.0_f32; kv_size];
        let out = multi_head_attention(&q, &k, &v, &cfg, seq).unwrap();
        assert_eq!(out.len(), q_size);
        for &val in &out {
            assert!((val - 4.0).abs() < 1e-4);
        }
    }

    // ── AttentionMetrics tests ────────────────────────────────────────

    #[test]
    fn test_metrics_basic() {
        let m = AttentionMetrics::compute(8, 16, 16, 64);
        assert_eq!(m.flops, 524288);
        assert_eq!(m.memory_bytes, 131072);
        assert!(m.arithmetic_intensity > 0.0);
    }

    #[test]
    fn test_metrics_arithmetic_intensity() {
        let m = AttentionMetrics::compute(1, 4, 4, 8);
        let expected = m.flops as f64 / m.memory_bytes as f64;
        assert!((m.arithmetic_intensity - expected).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_gqa_less_kv_memory() {
        let mha = AttentionMetrics::compute(8, 16, 16, 64);
        let gqa = AttentionMetrics::compute_gqa(8, 2, 16, 16, 64);
        assert!(gqa.memory_bytes < mha.memory_bytes);
        assert_eq!(gqa.flops, mha.flops);
    }

    #[test]
    fn test_metrics_gqa_same_heads_equals_mha() {
        let mha = AttentionMetrics::compute(8, 16, 16, 64);
        let gqa = AttentionMetrics::compute_gqa(8, 8, 16, 16, 64);
        assert_eq!(mha.flops, gqa.flops);
        assert_eq!(mha.memory_bytes, gqa.memory_bytes);
    }

    #[test]
    fn test_metrics_scales_with_seq_len() {
        let m1 = AttentionMetrics::compute(8, 128, 128, 64);
        let m2 = AttentionMetrics::compute(8, 256, 256, 64);
        assert_eq!(m2.flops, m1.flops * 4);
    }

    #[test]
    fn test_metrics_zero_memory() {
        let m = AttentionMetrics::compute(0, 0, 0, 0);
        assert_eq!(m.flops, 0);
        assert_eq!(m.memory_bytes, 0);
        assert_eq!(m.arithmetic_intensity, 0.0);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_single_head_single_position() {
        let cfg = FusedAttentionConfig::new(8, 1, 1, 1024).unwrap();
        let q = vec![1.0_f32; 8];
        let k = vec![0.5_f32; 8];
        let v = vec![42.0_f32; 8];
        let out = fused_attention_forward(&q, &k, &v, &cfg, 1).unwrap();
        for &val in &out {
            assert!((val - 42.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_max_seq_len_boundary() {
        let cfg = FusedAttentionConfig::new(4, 1, 1, 4).unwrap();
        let size = 1 * 4 * 4;
        let data = vec![1.0_f32; size];
        assert!(fused_attention_forward(&data, &data, &data, &cfg, 4).is_ok());
        let big = vec![1.0_f32; 1 * 5 * 4];
        assert!(fused_attention_forward(&big, &big, &big, &cfg, 5).is_err());
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let cfg = FusedAttentionConfig::new(2, 1, 1, 128).unwrap();
        let q = vec![500.0, 500.0, -500.0, -500.0];
        let k = vec![500.0, 500.0, -500.0, -500.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = fused_attention_forward(&q, &k, &v, &cfg, 2).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "non-finite with large values");
    }

    // ── CUDA kernel source tests ──────────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_fused_kernel_source_not_empty() {
        assert!(!FUSED_ATTENTION_KERNEL_SRC.is_empty());
        assert!(FUSED_ATTENTION_KERNEL_SRC.contains("fused_attention_f32"));
        assert!(FUSED_ATTENTION_KERNEL_SRC.contains("fused_attention_causal_f32"));
        assert!(FUSED_ATTENTION_KERNEL_SRC.contains("fused_gqa_attention_f32"));
    }

    // ── Property tests ────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn head_dim_strategy() -> impl Strategy<Value = usize> {
            prop_oneof![Just(4), Just(8), Just(16), Just(32), Just(64), Just(128),]
        }

        fn head_pair_strategy() -> impl Strategy<Value = (usize, usize)> {
            prop_oneof![
                Just((4, 4)),
                Just((8, 8)),
                Just((8, 4)),
                Just((8, 2)),
                Just((8, 1)),
                Just((16, 4)),
                Just((16, 8)),
                Just((32, 8)),
            ]
        }

        proptest! {
            #[test]
            fn prop_config_valid_head_dims(dim in head_dim_strategy()) {
                let cfg = FusedAttentionConfig::new(dim, 8, 8, 2048);
                prop_assert!(cfg.is_ok(), "dim={dim} should be valid");
            }

            #[test]
            fn prop_config_scale_positive(dim in head_dim_strategy()) {
                let cfg = FusedAttentionConfig::new(dim, 8, 8, 2048).unwrap();
                prop_assert!(cfg.scale() > 0.0);
                prop_assert!(cfg.scale().is_finite());
            }

            #[test]
            fn prop_attention_output_finite(
                seq in 1_usize..=8,
                dim in head_dim_strategy(),
            ) {
                let cfg = FusedAttentionConfig::new(dim, 1, 1, 128).unwrap();
                let size = 1 * seq * dim;
                let q = vec![0.1_f32; size];
                let k = vec![0.2_f32; size];
                let v = vec![0.3_f32; size];
                let out = fused_attention_forward(&q, &k, &v, &cfg, seq).unwrap();
                prop_assert!(out.iter().all(|v| v.is_finite()));
            }

            #[test]
            fn prop_attention_output_shape(
                seq in 1_usize..=8,
                (num_heads, num_kv_heads) in head_pair_strategy(),
            ) {
                let dim = 4;
                let cfg = FusedAttentionConfig::new(dim, num_heads, num_kv_heads, 128).unwrap();
                let q_size = num_heads * seq * dim;
                let kv_size = num_kv_heads * seq * dim;
                let q = vec![0.1_f32; q_size];
                let k = vec![0.2_f32; kv_size];
                let v = vec![0.3_f32; kv_size];
                let out = fused_attention_forward(&q, &k, &v, &cfg, seq).unwrap();
                prop_assert_eq!(out.len(), q_size);
            }

            #[test]
            fn prop_flash_matches_fused(seq in 1_usize..=6) {
                let cfg = FusedAttentionConfig::new(4, 1, 1, 128).unwrap();
                let size = 1 * seq * 4;
                let q: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
                let k: Vec<f32> = (0..size).map(|i| ((i + 3) as f32) * 0.05).collect();
                let v: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();

                let fused = fused_attention_forward(&q, &k, &v, &cfg, seq).unwrap();
                let flash = flash_attention_forward(&q, &k, &v, &cfg, seq, 2).unwrap();

                for (a, b) in fused.iter().zip(flash.iter()) {
                    prop_assert!(
                        (a - b).abs() < 1e-3,
                        "flash vs fused mismatch at seq={}: {} vs {}", seq, a, b
                    );
                }
            }

            #[test]
            fn prop_causal_mask_lower_triangular(seq in 1_usize..=16) {
                let mask = AttentionPattern::Causal.generate_mask(seq);
                for i in 0..seq {
                    for j in 0..seq {
                        if j <= i {
                            prop_assert_eq!(mask[i * seq + j], 0.0,
                                "causal mask [{},{j}] should be 0 (attend)", i, j = j);
                        } else {
                            prop_assert_eq!(mask[i * seq + j], f32::NEG_INFINITY,
                                "causal mask [{},{j}] should be -inf (block)", i, j = j);
                        }
                    }
                }
            }

            #[test]
            fn prop_gqa_heads_per_group(
                (num_heads, num_kv_heads) in head_pair_strategy(),
            ) {
                let cfg = FusedAttentionConfig::new(64, num_heads, num_kv_heads, 2048).unwrap();
                let hpg = cfg.heads_per_kv_group();
                prop_assert_eq!(hpg * num_kv_heads, num_heads);
            }

            #[test]
            fn prop_metrics_flops_positive(
                seq in 1_usize..=32,
                heads in 1_usize..=8,
            ) {
                let m = AttentionMetrics::compute(heads, seq, seq, 64);
                prop_assert!(m.flops > 0);
                prop_assert!(m.memory_bytes > 0);
                prop_assert!(m.arithmetic_intensity > 0.0);
            }

            #[test]
            fn prop_sliding_window_subset_of_causal(
                seq in 2_usize..=12,
                window in 1_usize..=6,
            ) {
                let causal = AttentionPattern::Causal.generate_mask(seq);
                let sw = AttentionPattern::SlidingWindow { window_size: window }.generate_mask(seq);
                for i in 0..seq * seq {
                    if causal[i] == f32::NEG_INFINITY {
                        prop_assert_eq!(sw[i], f32::NEG_INFINITY,
                            "sliding window should block where causal blocks (idx {})", i);
                    }
                }
            }
        }
    }
}
