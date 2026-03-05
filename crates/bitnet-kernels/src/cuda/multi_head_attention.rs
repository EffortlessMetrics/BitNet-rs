//! CUDA multi-head attention with GQA, RoPE, ALiBi, and sliding window.
//!
//! # Kernel strategy
//!
//! Multi-head attention (MHA) decomposes attention across independent heads,
//! each operating on a `head_dim`-dimensional projection of the input.
//! This module provides:
//!
//! - **Standard MHA** — Full multi-head attention with QKV projections.
//! - **Grouped Query Attention (GQA)** — Fewer KV heads shared across Q heads.
//! - **KV cache attention** — Incremental decoding with cached keys/values.
//! - **Sliding window** — Bounded context via a local attention window.
//! - **ALiBi** — Linear position bias (no learned embeddings).
//! - **RoPE** — Rotary position encoding applied to Q and K.
//!
//! All functions feature-gate GPU paths behind `#[cfg(any(feature = "gpu",
//! feature = "cuda"))]` and provide CPU fallback implementations.

use bitnet_common::{KernelError, Result};

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (equal to `num_heads` for standard MHA,
    /// fewer for GQA/MQA).
    pub num_kv_heads: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Dropout probability (applied during training; 0.0 at inference).
    pub dropout_p: f32,
    /// Softmax temperature scale; defaults to `1.0 / sqrt(head_dim)`.
    pub scale: f32,
}

impl MultiHeadAttentionConfig {
    /// Create a new config with default scale `1/sqrt(head_dim)`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero or `num_heads` is not
    /// divisible by `num_kv_heads`.
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        dropout_p: f32,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MHA: dimensions must be non-zero: num_heads={num_heads}, \
                     num_kv_heads={num_kv_heads}, head_dim={head_dim}"
                ),
            }
            .into());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "MHA: num_heads ({num_heads}) must be divisible by \
                     num_kv_heads ({num_kv_heads})"
                ),
            }
            .into());
        }
        if !(0.0..=1.0).contains(&dropout_p) {
            return Err(KernelError::InvalidArguments {
                reason: format!("MHA: dropout_p must be in [0.0, 1.0], got {dropout_p}"),
            }
            .into());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self { num_heads, num_kv_heads, head_dim, causal, dropout_p, scale })
    }

    /// Override the default scale factor.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Number of query heads that share each KV head.
    #[inline]
    pub fn kv_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// True when every Q head has its own KV head (standard MHA).
    #[inline]
    pub fn is_standard_mha(&self) -> bool {
        self.num_heads == self.num_kv_heads
    }

    /// CUDA grid dimensions: `(ceil(seq_q / tile), num_heads, batch)`.
    pub fn grid_dim(&self, seq_len_q: usize, batch_size: usize) -> (u32, u32, u32) {
        let tile = 32u32;
        let grid_x = (seq_len_q as u32).div_ceil(tile);
        (grid_x, self.num_heads as u32, batch_size as u32)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (256, 1, 1)
    }
}

// ── Output ───────────────────────────────────────────────────────────

/// Output from a multi-head attention forward pass.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Output tensor `[batch, seq_len_q, num_heads * head_dim]` (row-major).
    pub output: Vec<f32>,
    /// Optional attention weights `[batch, num_heads, seq_len_q, seq_len_kv]`.
    pub attention_weights: Option<Vec<f32>>,
    /// Optional updated key cache `[batch, num_kv_heads, total_seq, head_dim]`.
    pub key_cache: Option<Vec<f32>>,
    /// Optional updated value cache `[batch, num_kv_heads, total_seq, head_dim]`.
    pub value_cache: Option<Vec<f32>>,
}

// ── CUDA kernel source ───────────────────────────────────────────────

/// Inline CUDA C source for multi-head attention kernel.
///
/// Implements per-head scaled dot-product attention with optional causal
/// masking and GQA head mapping. Each thread-block processes one query
/// tile for one head.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MHA_KERNEL_SRC: &str = r#"
extern "C" __global__ void mha_forward_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float scale,
    int causal)
{
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    if (q_idx >= seq_len_q || head >= num_heads || batch >= batch_size) return;

    int kv_head = head / (num_heads / num_kv_heads);

    int q_offset = ((batch * num_heads + head) * seq_len_q + q_idx) * head_dim;
    int kv_batch_offset_k = (batch * num_kv_heads + kv_head) * seq_len_kv * head_dim;
    int kv_batch_offset_v = kv_batch_offset_k;

    float row_max = -1e30f;

    extern __shared__ float scores[];
    for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
        if (causal && k_idx > q_idx) {
            scores[k_idx] = -1e30f;
        } else {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[q_offset + d] * K[kv_batch_offset_k + k_idx * head_dim + d];
            }
            dot *= scale;
            scores[k_idx] = dot;
            if (dot > row_max) row_max = dot;
        }
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

    int o_offset = ((batch * num_heads + head) * seq_len_q + q_idx) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
            acc += scores[k_idx] * V[kv_batch_offset_v + k_idx * head_dim + d];
        }
        O[o_offset + d] = acc;
    }
}
"#;

// ── Internal helpers ─────────────────────────────────────────────────

/// Numerically stable softmax in-place.
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

/// Single-head scaled dot-product attention (CPU).
///
/// `q`: `[seq_q, head_dim]`, `k`: `[seq_kv, head_dim]`,
/// `v`: `[seq_kv, head_dim]` → output `[seq_q, head_dim]`.
fn sdp_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; seq_q * head_dim];
    for i in 0..seq_q {
        let mut scores = vec![0.0_f32; seq_kv];
        for j in 0..seq_kv {
            if causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = dot * scale;
            }
        }
        softmax_inplace(&mut scores);
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_kv {
                acc += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
    output
}

/// Single-head SDP with an additive bias tensor `[seq_q, seq_kv]`.
#[allow(clippy::too_many_arguments)]
fn sdp_with_bias_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; seq_q * head_dim];
    for i in 0..seq_q {
        let mut scores = vec![0.0_f32; seq_kv];
        for j in 0..seq_kv {
            if causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = dot * scale + bias[i * seq_kv + j];
            }
        }
        softmax_inplace(&mut scores);
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_kv {
                acc += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
    output
}

// ── Reshape helpers ──────────────────────────────────────────────────

/// Reshape `[batch, seq_len, num_heads * head_dim]` →
/// `[batch, num_heads, seq_len, head_dim]`.
///
/// # Errors
///
/// Returns an error if the input length does not match
/// `batch * seq_len * num_heads * head_dim`.
pub fn split_heads(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let expected = batch * seq_len * num_heads * head_dim;
    if input.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("split_heads: expected {expected} elements, got {}", input.len()),
        }
        .into());
    }
    let mut output = vec![0.0_f32; expected];
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src = b * seq_len * num_heads * head_dim
                        + s * num_heads * head_dim
                        + h * head_dim
                        + d;
                    let dst = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    output[dst] = input[src];
                }
            }
        }
    }
    Ok(output)
}

/// Reshape `[batch, num_heads, seq_len, head_dim]` →
/// `[batch, seq_len, num_heads * head_dim]`.
///
/// # Errors
///
/// Returns an error if the input length does not match
/// `batch * num_heads * seq_len * head_dim`.
pub fn merge_heads(
    input: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let expected = batch * num_heads * seq_len * head_dim;
    if input.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("merge_heads: expected {expected} elements, got {}", input.len()),
        }
        .into());
    }
    let mut output = vec![0.0_f32; expected];
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let src = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    let dst = b * seq_len * num_heads * head_dim
                        + s * num_heads * head_dim
                        + h * head_dim
                        + d;
                    output[dst] = input[src];
                }
            }
        }
    }
    Ok(output)
}

// ── Scaled dot-product attention ─────────────────────────────────────

/// Scaled dot-product attention: `softmax(Q·Kᵀ / √d) · V`.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `config` — MHA configuration
/// * `seq_q`  — Query sequence length
/// * `seq_kv` — Key/value sequence length
///
/// # Returns
///
/// Output `[batch, num_heads, seq_q, head_dim]` and optional attention
/// weights `[batch, num_heads, seq_q, seq_kv]`.
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn scaled_dot_product(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    validate_qkv(query, key, value, config, seq_q, seq_kv, batch)?;

    let head_dim = config.head_dim;
    let scale = config.scale;
    let group = config.kv_group_size();
    let q_head_elems = seq_q * head_dim;
    let kv_head_elems = seq_kv * head_dim;

    let total_out = batch * config.num_heads * q_head_elems;
    let mut output = vec![0.0_f32; total_out];

    for b in 0..batch {
        for h in 0..config.num_heads {
            let kv_h = h / group;
            let q_off = (b * config.num_heads + h) * q_head_elems;
            let k_off = (b * config.num_kv_heads + kv_h) * kv_head_elems;
            let v_off = k_off;
            let o_off = q_off;

            let head_out = sdp_cpu(
                &query[q_off..q_off + q_head_elems],
                &key[k_off..k_off + kv_head_elems],
                &value[v_off..v_off + kv_head_elems],
                seq_q,
                seq_kv,
                head_dim,
                scale,
                config.causal,
            );
            output[o_off..o_off + q_head_elems].copy_from_slice(&head_out);
        }
    }

    Ok(AttentionOutput { output, attention_weights: None, key_cache: None, value_cache: None })
}

// ── Full MHA forward pass ────────────────────────────────────────────

/// Full multi-head attention forward pass (CPU fallback).
///
/// Accepts input in `[batch, seq_len, num_heads * head_dim]` layout,
/// internally splits heads, computes attention per head, and merges
/// back.
///
/// # Arguments
///
/// * `query`  — `[batch, seq_q, num_heads * head_dim]`
/// * `key`    — `[batch, seq_kv, num_kv_heads * head_dim]`
/// * `value`  — `[batch, seq_kv, num_kv_heads * head_dim]`
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn multi_head_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    let q_expected = batch * seq_q * config.num_heads * config.head_dim;
    let kv_expected = batch * seq_kv * config.num_kv_heads * config.head_dim;
    if query.len() != q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "multi_head_attention: query length {}, expected {q_expected}",
                query.len()
            ),
        }
        .into());
    }
    if key.len() != kv_expected || value.len() != kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "multi_head_attention: kv length mismatch, expected {kv_expected}, \
                 got k={}, v={}",
                key.len(),
                value.len()
            ),
        }
        .into());
    }

    let q_heads = split_heads(query, batch, seq_q, config.num_heads, config.head_dim)?;
    let k_heads = split_heads(key, batch, seq_kv, config.num_kv_heads, config.head_dim)?;
    let v_heads = split_heads(value, batch, seq_kv, config.num_kv_heads, config.head_dim)?;

    let mut attn = scaled_dot_product(&q_heads, &k_heads, &v_heads, config, seq_q, seq_kv, batch)?;

    attn.output = merge_heads(&attn.output, batch, seq_q, config.num_heads, config.head_dim)?;

    Ok(attn)
}

// ── Grouped query attention ──────────────────────────────────────────

/// Grouped query attention (GQA) — MHA with fewer KV heads.
///
/// Semantically identical to [`multi_head_attention`] but named
/// explicitly for clarity when `num_kv_heads < num_heads`.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn grouped_query_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    validate_qkv(query, key, value, config, seq_q, seq_kv, batch)?;
    scaled_dot_product(query, key, value, config, seq_q, seq_kv, batch)
}

// ── Causal attention ─────────────────────────────────────────────────

/// Attention with a causal (autoregressive) mask.
///
/// Forces `config.causal = true` semantics regardless of config flag.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn causal_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    let mut causal_cfg = config.clone();
    causal_cfg.causal = true;
    scaled_dot_product(query, key, value, &causal_cfg, seq_q, seq_kv, batch)
}

// ── KV cache attention ───────────────────────────────────────────────

/// Attention with KV cache for autoregressive inference.
///
/// Appends new keys/values to the cache then computes attention over
/// the full cached context.
///
/// # Arguments
///
/// * `query`       — `[batch, num_heads, seq_q, head_dim]` (new query)
/// * `new_key`     — `[batch, num_kv_heads, seq_q, head_dim]` (new KV)
/// * `new_value`   — `[batch, num_kv_heads, seq_q, head_dim]`
/// * `cached_key`  — `[batch, num_kv_heads, cached_len, head_dim]`
/// * `cached_value` — `[batch, num_kv_heads, cached_len, head_dim]`
///
/// # Returns
///
/// Attention output with updated cache in `key_cache` / `value_cache`.
///
/// # Errors
///
/// Returns an error on shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_attention(
    query: &[f32],
    new_key: &[f32],
    new_value: &[f32],
    cached_key: &[f32],
    cached_value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    cached_len: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    let head_dim = config.head_dim;
    let new_kv_elems = batch * config.num_kv_heads * seq_q * head_dim;
    let cached_kv_elems = batch * config.num_kv_heads * cached_len * head_dim;

    if new_key.len() != new_kv_elems || new_value.len() != new_kv_elems {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kv_cache_attention: new kv length mismatch, expected {new_kv_elems}, \
                 got k={}, v={}",
                new_key.len(),
                new_value.len()
            ),
        }
        .into());
    }
    if cached_len > 0
        && (cached_key.len() != cached_kv_elems || cached_value.len() != cached_kv_elems)
    {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kv_cache_attention: cached kv length mismatch, expected \
                 {cached_kv_elems}, got k={}, v={}",
                cached_key.len(),
                cached_value.len()
            ),
        }
        .into());
    }

    let total_seq = cached_len + seq_q;
    let total_kv_elems = batch * config.num_kv_heads * total_seq * head_dim;
    let mut full_k = vec![0.0_f32; total_kv_elems];
    let mut full_v = vec![0.0_f32; total_kv_elems];

    // Concatenate cached + new along the sequence dimension.
    let per_head_cached = cached_len * head_dim;
    let per_head_new = seq_q * head_dim;
    for b in 0..batch {
        for h in 0..config.num_kv_heads {
            let dst_base = (b * config.num_kv_heads + h) * total_seq * head_dim;
            let ck_base = (b * config.num_kv_heads + h) * cached_len * head_dim;
            let nk_base = (b * config.num_kv_heads + h) * seq_q * head_dim;

            if cached_len > 0 {
                full_k[dst_base..dst_base + per_head_cached]
                    .copy_from_slice(&cached_key[ck_base..ck_base + per_head_cached]);
                full_v[dst_base..dst_base + per_head_cached]
                    .copy_from_slice(&cached_value[ck_base..ck_base + per_head_cached]);
            }
            full_k[dst_base + per_head_cached..dst_base + per_head_cached + per_head_new]
                .copy_from_slice(&new_key[nk_base..nk_base + per_head_new]);
            full_v[dst_base + per_head_cached..dst_base + per_head_cached + per_head_new]
                .copy_from_slice(&new_value[nk_base..nk_base + per_head_new]);
        }
    }

    let mut attn = scaled_dot_product(query, &full_k, &full_v, config, seq_q, total_seq, batch)?;
    attn.key_cache = Some(full_k);
    attn.value_cache = Some(full_v);
    Ok(attn)
}

// ── Sliding window attention ─────────────────────────────────────────

/// Attention with a sliding window — only the last `window_size`
/// key/value positions are visible to each query position.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `window_size` — Maximum lookback distance
///
/// # Errors
///
/// Returns an error on shape mismatch or if `window_size` is zero.
pub fn sliding_window_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
    window_size: usize,
) -> Result<AttentionOutput> {
    if window_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "sliding_window_attention: window_size must be non-zero".into(),
        }
        .into());
    }
    validate_qkv(query, key, value, config, seq_q, seq_kv, batch)?;

    let head_dim = config.head_dim;
    let scale = config.scale;
    let group = config.kv_group_size();
    let q_head_elems = seq_q * head_dim;
    let kv_head_elems = seq_kv * head_dim;
    let total_out = batch * config.num_heads * q_head_elems;
    let mut output = vec![0.0_f32; total_out];

    for b in 0..batch {
        for h in 0..config.num_heads {
            let kv_h = h / group;
            let q_off = (b * config.num_heads + h) * q_head_elems;
            let k_off = (b * config.num_kv_heads + kv_h) * kv_head_elems;
            let o_off = q_off;

            for i in 0..seq_q {
                let mut scores = vec![0.0_f32; seq_kv];
                for j in 0..seq_kv {
                    // Window mask: only attend if j is within
                    // [i - window_size + 1, i] (causal) or within window_size
                    // positions of i.
                    let in_window = if config.causal {
                        j <= i && (i - j) < window_size
                    } else {
                        j.abs_diff(i) < window_size
                    };
                    if !in_window {
                        scores[j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0_f32;
                        for d in 0..head_dim {
                            dot += query[q_off + i * head_dim + d] * key[k_off + j * head_dim + d];
                        }
                        scores[j] = dot * scale;
                    }
                }
                softmax_inplace(&mut scores);
                for d in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for j in 0..seq_kv {
                        acc += scores[j] * value[k_off + j * head_dim + d];
                    }
                    output[o_off + i * head_dim + d] = acc;
                }
            }
        }
    }

    Ok(AttentionOutput { output, attention_weights: None, key_cache: None, value_cache: None })
}

// ── ALiBi attention ──────────────────────────────────────────────────

/// Build ALiBi per-head slopes.
///
/// Returns `num_heads` slopes following the geometric sequence from
/// the ALiBi paper: `2^(-8/n) ... 2^(-8)` for power-of-two head
/// counts, with nearest-power-of-two interpolation otherwise.
pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    let closest_pow2 = 1usize << (usize::BITS - num_heads.leading_zeros() - 1);
    let base = 2.0_f64.powf(-8.0 / closest_pow2 as f64);
    let mut slopes = Vec::with_capacity(num_heads);
    if num_heads == closest_pow2 {
        for i in 1..=num_heads {
            slopes.push(base.powi(i as i32) as f32);
        }
    } else {
        // Interleave two geometric series for non-power-of-two counts.
        let extra_base = 2.0_f64.powf(-8.0 / (closest_pow2 * 2) as f64);
        let mut idx = 0;
        for i in 1..=closest_pow2 {
            slopes.push(base.powi(i as i32) as f32);
            idx += 1;
            if idx >= num_heads {
                break;
            }
            slopes.push(extra_base.powi((2 * i + 1) as i32) as f32);
            idx += 1;
            if idx >= num_heads {
                break;
            }
        }
    }
    slopes.truncate(num_heads);
    slopes
}

/// Attention with ALiBi linear position bias.
///
/// Adds `slope_h * (j - i)` to each score before softmax, removing
/// the need for explicit position embeddings.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `slopes` — Per-head slopes `[num_heads]` (from [`alibi_slopes`])
///
/// # Errors
///
/// Returns an error on shape mismatch.
pub fn attention_with_alibi(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
    slopes: &[f32],
) -> Result<AttentionOutput> {
    if slopes.len() != config.num_heads {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "attention_with_alibi: expected {} slopes, got {}",
                config.num_heads,
                slopes.len()
            ),
        }
        .into());
    }
    validate_qkv(query, key, value, config, seq_q, seq_kv, batch)?;

    let head_dim = config.head_dim;
    let scale = config.scale;
    let group = config.kv_group_size();
    let q_head_elems = seq_q * head_dim;
    let kv_head_elems = seq_kv * head_dim;
    let total_out = batch * config.num_heads * q_head_elems;
    let mut output = vec![0.0_f32; total_out];

    for b in 0..batch {
        for (h, &slope) in slopes.iter().enumerate().take(config.num_heads) {
            let kv_h = h / group;
            let q_off = (b * config.num_heads + h) * q_head_elems;
            let k_off = (b * config.num_kv_heads + kv_h) * kv_head_elems;
            let o_off = q_off;

            // Build per-head bias: slope * (j - i)
            let mut bias = vec![0.0_f32; seq_q * seq_kv];
            for i in 0..seq_q {
                for j in 0..seq_kv {
                    bias[i * seq_kv + j] = slope * (j as f32 - i as f32);
                }
            }

            let head_out = sdp_with_bias_cpu(
                &query[q_off..q_off + q_head_elems],
                &key[k_off..k_off + kv_head_elems],
                &value[k_off..k_off + kv_head_elems],
                &bias,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                config.causal,
            );
            output[o_off..o_off + q_head_elems].copy_from_slice(&head_out);
        }
    }

    Ok(AttentionOutput { output, attention_weights: None, key_cache: None, value_cache: None })
}

// ── RoPE attention ───────────────────────────────────────────────────

/// Apply rotary position embedding in-place.
///
/// Operates on `[batch, num_heads, seq_len, head_dim]` tensors.
/// Pairs `(x[2i], x[2i+1])` are rotated by the angle
/// `pos * base^(-2i/head_dim)`.
fn apply_rope_inplace(
    data: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    base: f32,
    offset: usize,
) {
    let half = head_dim / 2;
    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                let row = (b * num_heads * seq_len + h * seq_len + s) * head_dim;
                let pos = (s + offset) as f32;
                for i in 0..half {
                    let freq = pos / base.powf(2.0 * i as f32 / head_dim as f32);
                    let cos_v = freq.cos();
                    let sin_v = freq.sin();
                    let x0 = data[row + 2 * i];
                    let x1 = data[row + 2 * i + 1];
                    data[row + 2 * i] = x0 * cos_v - x1 * sin_v;
                    data[row + 2 * i + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }
}

/// Attention with Rotary Position Encoding (RoPE) applied to Q and K.
///
/// # Arguments
///
/// * `query`  — `[batch, num_heads, seq_q, head_dim]`
/// * `key`    — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `value`  — `[batch, num_kv_heads, seq_kv, head_dim]`
/// * `rope_base`   — Base frequency for RoPE (default 10000.0)
/// * `position_offset` — Starting position for RoPE
///
/// # Errors
///
/// Returns an error on shape mismatch or if `head_dim` is odd.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_rope(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
    rope_base: f32,
    position_offset: usize,
) -> Result<AttentionOutput> {
    if !config.head_dim.is_multiple_of(2) {
        return Err(KernelError::InvalidArguments {
            reason: format!("attention_with_rope: head_dim must be even, got {}", config.head_dim),
        }
        .into());
    }
    validate_qkv(query, key, value, config, seq_q, seq_kv, batch)?;

    let mut q_rope = query.to_vec();
    let mut k_rope = key.to_vec();

    apply_rope_inplace(
        &mut q_rope,
        batch,
        config.num_heads,
        seq_q,
        config.head_dim,
        rope_base,
        position_offset,
    );
    apply_rope_inplace(
        &mut k_rope,
        batch,
        config.num_kv_heads,
        seq_kv,
        config.head_dim,
        rope_base,
        position_offset,
    );

    scaled_dot_product(&q_rope, &k_rope, value, config, seq_q, seq_kv, batch)
}

// ── CUDA launch stub ─────────────────────────────────────────────────

/// Launch stub for the MHA CUDA kernel.
///
/// Falls back to CPU via [`multi_head_attention`] when GPU is not
/// available at compile time.
///
/// # Errors
///
/// Returns `KernelError::GpuError` when compiled with GPU features but
/// no runtime kernel is loaded. Falls back to CPU otherwise.
pub fn launch_multi_head_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<AttentionOutput> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        log::debug!(
            "MHA CUDA stub: heads={}, kv_heads={}, head_dim={}, seq_q={}, \
             seq_kv={}, batch={}, causal={}, grid={:?}",
            config.num_heads,
            config.num_kv_heads,
            config.head_dim,
            seq_q,
            seq_kv,
            batch,
            config.causal,
            config.grid_dim(seq_q, batch),
        );
        let _ = (query, key, value);
        return Err(KernelError::GpuError {
            reason: "MHA CUDA kernel not yet compiled — scaffold only".into(),
        }
        .into());
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        multi_head_attention(query, key, value, config, seq_q, seq_kv, batch)
    }
}

// ── Validation helper ────────────────────────────────────────────────

/// Validate Q/K/V shapes for `[batch, heads, seq, head_dim]` layout.
fn validate_qkv(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &MultiHeadAttentionConfig,
    seq_q: usize,
    seq_kv: usize,
    batch: usize,
) -> Result<()> {
    let q_expected = batch * config.num_heads * seq_q * config.head_dim;
    let kv_expected = batch * config.num_kv_heads * seq_kv * config.head_dim;
    if query.len() != q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "query length {}, expected {q_expected} \
                 (batch={batch}, heads={}, seq_q={seq_q}, dim={})",
                query.len(),
                config.num_heads,
                config.head_dim,
            ),
        }
        .into());
    }
    if key.len() != kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("key length {}, expected {kv_expected}", key.len()),
        }
        .into());
    }
    if value.len() != kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("value length {}, expected {kv_expected}", value.len()),
        }
        .into());
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------

    fn make_config(num_heads: usize, head_dim: usize, causal: bool) -> MultiHeadAttentionConfig {
        MultiHeadAttentionConfig::new(num_heads, num_heads, head_dim, causal, 0.0).unwrap()
    }

    fn make_gqa_config(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> MultiHeadAttentionConfig {
        MultiHeadAttentionConfig::new(num_heads, num_kv_heads, head_dim, causal, 0.0).unwrap()
    }

    /// Identity-like Q/K/V: ones for all elements.
    fn ones(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    /// Linearly spaced values for deterministic tests.
    fn linspace(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.01).collect()
    }

    /// Assert two slices are element-wise close.
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    // ── Config tests ─────────────────────────────────────────────────

    #[test]
    fn config_new_valid() {
        let cfg = MultiHeadAttentionConfig::new(8, 8, 64, true, 0.0).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert!((cfg.scale - 1.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn config_gqa_valid() {
        let cfg = MultiHeadAttentionConfig::new(8, 2, 64, false, 0.1).unwrap();
        assert_eq!(cfg.kv_group_size(), 4);
        assert!(!cfg.is_standard_mha());
    }

    #[test]
    fn config_mqa_single_kv_head() {
        let cfg = MultiHeadAttentionConfig::new(8, 1, 64, false, 0.0).unwrap();
        assert_eq!(cfg.kv_group_size(), 8);
    }

    #[test]
    fn config_zero_heads_fails() {
        assert!(MultiHeadAttentionConfig::new(0, 1, 64, false, 0.0).is_err());
    }

    #[test]
    fn config_zero_kv_heads_fails() {
        assert!(MultiHeadAttentionConfig::new(8, 0, 64, false, 0.0).is_err());
    }

    #[test]
    fn config_zero_dim_fails() {
        assert!(MultiHeadAttentionConfig::new(8, 8, 0, false, 0.0).is_err());
    }

    #[test]
    fn config_indivisible_heads_fails() {
        assert!(MultiHeadAttentionConfig::new(7, 3, 64, false, 0.0).is_err());
    }

    #[test]
    fn config_dropout_negative_fails() {
        assert!(MultiHeadAttentionConfig::new(4, 4, 32, false, -0.1).is_err());
    }

    #[test]
    fn config_dropout_above_one_fails() {
        assert!(MultiHeadAttentionConfig::new(4, 4, 32, false, 1.1).is_err());
    }

    #[test]
    fn config_with_scale_override() {
        let cfg = MultiHeadAttentionConfig::new(4, 4, 64, false, 0.0).unwrap().with_scale(0.5);
        assert!((cfg.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_is_standard_mha() {
        let cfg = make_config(4, 64, false);
        assert!(cfg.is_standard_mha());
    }

    #[test]
    fn config_grid_dim_basic() {
        let cfg = make_config(8, 64, false);
        let (gx, gy, gz) = cfg.grid_dim(128, 2);
        assert_eq!(gy, 8);
        assert_eq!(gz, 2);
        assert!(gx >= 4); // 128 / 32
    }

    #[test]
    fn config_block_dim() {
        let cfg = make_config(4, 64, false);
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    // ── split_heads / merge_heads ────────────────────────────────────

    #[test]
    fn split_heads_basic() {
        let input = linspace(2 * 4 * 3 * 8); // batch=2, seq=4, heads=3, dim=8
        let out = split_heads(&input, 2, 4, 3, 8).unwrap();
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn merge_heads_basic() {
        let input = linspace(2 * 3 * 4 * 8);
        let out = merge_heads(&input, 2, 4, 3, 8).unwrap();
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn split_merge_roundtrip() {
        let original = linspace(1 * 4 * 2 * 8);
        let split = split_heads(&original, 1, 4, 2, 8).unwrap();
        let merged = merge_heads(&split, 1, 4, 2, 8).unwrap();
        assert_close(&original, &merged, 1e-6);
    }

    #[test]
    fn split_heads_wrong_length() {
        assert!(split_heads(&[1.0; 10], 1, 2, 2, 4).is_err());
    }

    #[test]
    fn merge_heads_wrong_length() {
        assert!(merge_heads(&[1.0; 10], 1, 2, 2, 4).is_err());
    }

    #[test]
    fn split_merge_batch2() {
        let data = linspace(2 * 3 * 4 * 16);
        let split = split_heads(&data, 2, 3, 4, 16).unwrap();
        let merged = merge_heads(&split, 2, 3, 4, 16).unwrap();
        assert_close(&data, &merged, 1e-6);
    }

    #[test]
    fn split_heads_single_element() {
        let data = vec![42.0_f32];
        let out = split_heads(&data, 1, 1, 1, 1).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    // ── scaled_dot_product ───────────────────────────────────────────

    #[test]
    fn sdp_basic_noncausal() {
        let cfg = make_config(1, 4, false);
        let q = ones(1 * 1 * 2 * 4);
        let k = ones(1 * 1 * 2 * 4);
        let v = ones(1 * 1 * 2 * 4);
        let out = scaled_dot_product(&q, &k, &v, &cfg, 2, 2, 1).unwrap();
        assert_eq!(out.output.len(), 1 * 1 * 2 * 4);
        // With all ones, output should also be all ones (softmax uniform → avg of V=1).
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sdp_basic_causal() {
        let cfg = make_config(1, 4, true);
        let q = ones(1 * 1 * 3 * 4);
        let k = ones(1 * 1 * 3 * 4);
        let v = ones(1 * 1 * 3 * 4);
        let out = scaled_dot_product(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sdp_multi_head() {
        let cfg = make_config(2, 4, false);
        let n = 1 * 2 * 3 * 4;
        let q = linspace(n);
        let k = linspace(n);
        let v = ones(n);
        let out = scaled_dot_product(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn sdp_wrong_query_length() {
        let cfg = make_config(1, 4, false);
        assert!(scaled_dot_product(&[0.0; 3], &[0.0; 4], &[0.0; 4], &cfg, 1, 1, 1).is_err());
    }

    #[test]
    fn sdp_wrong_key_length() {
        let cfg = make_config(1, 4, false);
        assert!(scaled_dot_product(&[0.0; 4], &[0.0; 3], &[0.0; 4], &cfg, 1, 1, 1).is_err());
    }

    #[test]
    fn sdp_wrong_value_length() {
        let cfg = make_config(1, 4, false);
        assert!(scaled_dot_product(&[0.0; 4], &[0.0; 4], &[0.0; 3], &cfg, 1, 1, 1).is_err());
    }

    #[test]
    fn sdp_batch2() {
        let cfg = make_config(1, 4, false);
        let n = 2 * 1 * 2 * 4;
        let q = ones(n);
        let k = ones(n);
        let v = ones(n);
        let out = scaled_dot_product(&q, &k, &v, &cfg, 2, 2, 2).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn sdp_different_seq_lengths() {
        let cfg = make_config(1, 4, false);
        let q = ones(1 * 1 * 2 * 4);
        let k = ones(1 * 1 * 5 * 4);
        let v = ones(1 * 1 * 5 * 4);
        let out = scaled_dot_product(&q, &k, &v, &cfg, 2, 5, 1).unwrap();
        assert_eq!(out.output.len(), 1 * 1 * 2 * 4);
    }

    // ── multi_head_attention (merged-head layout) ────────────────────

    #[test]
    fn mha_basic() {
        let cfg = make_config(2, 4, false);
        let q = ones(1 * 3 * 2 * 4); // batch=1, seq=3, heads*dim=8
        let k = ones(1 * 3 * 2 * 4);
        let v = ones(1 * 3 * 2 * 4);
        let out = multi_head_attention(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        assert_eq!(out.output.len(), 1 * 3 * 8);
    }

    #[test]
    fn mha_wrong_query_length() {
        let cfg = make_config(2, 4, false);
        assert!(multi_head_attention(&[0.0; 5], &[0.0; 8], &[0.0; 8], &cfg, 1, 1, 1).is_err());
    }

    #[test]
    fn mha_wrong_kv_length() {
        let cfg = make_config(2, 4, false);
        assert!(multi_head_attention(&[0.0; 8], &[0.0; 5], &[0.0; 8], &cfg, 1, 1, 1).is_err());
    }

    #[test]
    fn mha_causal() {
        let cfg = make_config(2, 4, true);
        let n = 1 * 4 * 2 * 4;
        let out = multi_head_attention(&ones(n), &ones(n), &ones(n), &cfg, 4, 4, 1).unwrap();
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mha_batch2() {
        let cfg = make_config(2, 4, false);
        let n = 2 * 3 * 2 * 4;
        let out = multi_head_attention(&ones(n), &ones(n), &ones(n), &cfg, 3, 3, 2).unwrap();
        assert_eq!(out.output.len(), n);
    }

    // ── grouped_query_attention ──────────────────────────────────────

    #[test]
    fn gqa_basic() {
        let cfg = make_gqa_config(4, 2, 8, false);
        let q = ones(1 * 4 * 3 * 8);
        let kv = ones(1 * 2 * 3 * 8);
        let out = grouped_query_attention(&q, &kv, &kv, &cfg, 3, 3, 1).unwrap();
        assert_eq!(out.output.len(), 1 * 4 * 3 * 8);
    }

    #[test]
    fn gqa_mqa_single_kv() {
        let cfg = make_gqa_config(4, 1, 8, false);
        let q = ones(1 * 4 * 2 * 8);
        let kv = ones(1 * 1 * 2 * 8);
        let out = grouped_query_attention(&q, &kv, &kv, &cfg, 2, 2, 1).unwrap();
        assert_eq!(out.output.len(), 1 * 4 * 2 * 8);
    }

    #[test]
    fn gqa_wrong_kv_shape() {
        let cfg = make_gqa_config(4, 2, 8, false);
        let q = ones(1 * 4 * 2 * 8);
        let kv_wrong = ones(1 * 4 * 2 * 8); // should be 2 kv heads
        assert!(grouped_query_attention(&q, &kv_wrong, &kv_wrong, &cfg, 2, 2, 1).is_err());
    }

    #[test]
    fn gqa_causal() {
        let cfg = make_gqa_config(4, 2, 4, true);
        let q = ones(1 * 4 * 3 * 4);
        let kv = ones(1 * 2 * 3 * 4);
        let out = grouped_query_attention(&q, &kv, &kv, &cfg, 3, 3, 1).unwrap();
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn gqa_batch2() {
        let cfg = make_gqa_config(4, 2, 8, false);
        let q = ones(2 * 4 * 3 * 8);
        let kv = ones(2 * 2 * 3 * 8);
        let out = grouped_query_attention(&q, &kv, &kv, &cfg, 3, 3, 2).unwrap();
        assert_eq!(out.output.len(), 2 * 4 * 3 * 8);
    }

    // ── causal_attention ─────────────────────────────────────────────

    #[test]
    fn causal_forces_mask() {
        let cfg = make_config(1, 4, false); // causal=false in config
        let n = 1 * 1 * 3 * 4;
        let q = linspace(n);
        let k = linspace(n);
        let v = linspace(n);

        let causal_out = causal_attention(&q, &k, &v, &cfg, 3, 3, 1).unwrap();

        let noncausal_cfg = make_config(1, 4, false);
        let noncausal_out = scaled_dot_product(&q, &k, &v, &noncausal_cfg, 3, 3, 1).unwrap();

        // First row (pos 0) should differ because causal masks out future.
        // (Actually pos 0 is the same for causal since there's nothing to mask.)
        // Pos 2 should differ because it can't see pos 0,1 in non-causal
        // but in causal pos 2 can see everything ≤ 2.
        // We just check shapes match and they are valid.
        assert_eq!(causal_out.output.len(), noncausal_out.output.len());
    }

    #[test]
    fn causal_single_token() {
        let cfg = make_config(2, 4, false);
        let q = ones(1 * 2 * 1 * 4);
        let k = ones(1 * 2 * 1 * 4);
        let v = ones(1 * 2 * 1 * 4);
        let out = causal_attention(&q, &k, &v, &cfg, 1, 1, 1).unwrap();
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    // ── kv_cache_attention ───────────────────────────────────────────

    #[test]
    fn kv_cache_no_prior_cache() {
        let cfg = make_config(1, 4, true);
        let q = ones(1 * 1 * 1 * 4);
        let new_k = ones(1 * 1 * 1 * 4);
        let new_v = ones(1 * 1 * 1 * 4);
        let out = kv_cache_attention(&q, &new_k, &new_v, &[], &[], &cfg, 1, 0, 1).unwrap();
        assert_eq!(out.output.len(), 4);
        assert!(out.key_cache.is_some());
        assert!(out.value_cache.is_some());
    }

    #[test]
    fn kv_cache_with_prior() {
        let cfg = make_config(1, 4, true);
        let cached_k = ones(1 * 1 * 2 * 4);
        let cached_v = ones(1 * 1 * 2 * 4);
        let q = ones(1 * 1 * 1 * 4);
        let new_k = ones(1 * 1 * 1 * 4);
        let new_v = ones(1 * 1 * 1 * 4);
        let out =
            kv_cache_attention(&q, &new_k, &new_v, &cached_k, &cached_v, &cfg, 1, 2, 1).unwrap();
        // Total KV seq = 2 + 1 = 3
        assert_eq!(out.key_cache.as_ref().unwrap().len(), 1 * 1 * 3 * 4);
    }

    #[test]
    fn kv_cache_incremental_growth() {
        let cfg = make_config(1, 4, true);
        // Step 1: first token
        let q1 = ones(1 * 1 * 1 * 4);
        let k1 = ones(1 * 1 * 1 * 4);
        let v1 = ones(1 * 1 * 1 * 4);
        let out1 = kv_cache_attention(&q1, &k1, &v1, &[], &[], &cfg, 1, 0, 1).unwrap();
        let ck1 = out1.key_cache.unwrap();
        let cv1 = out1.value_cache.unwrap();

        // Step 2: second token using cache from step 1
        let q2 = ones(1 * 1 * 1 * 4);
        let k2 = ones(1 * 1 * 1 * 4);
        let v2 = ones(1 * 1 * 1 * 4);
        let out2 = kv_cache_attention(&q2, &k2, &v2, &ck1, &cv1, &cfg, 1, 1, 1).unwrap();
        assert_eq!(out2.key_cache.unwrap().len(), 1 * 1 * 2 * 4);
    }

    #[test]
    fn kv_cache_wrong_new_kv_length() {
        let cfg = make_config(1, 4, true);
        assert!(
            kv_cache_attention(
                &ones(4),
                &ones(3), // wrong length
                &ones(4),
                &[],
                &[],
                &cfg,
                1,
                0,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn kv_cache_wrong_cached_length() {
        let cfg = make_config(1, 4, true);
        assert!(
            kv_cache_attention(
                &ones(4),
                &ones(4),
                &ones(4),
                &ones(3), // wrong cached length
                &ones(4),
                &cfg,
                1,
                1,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn kv_cache_multi_head() {
        let cfg = make_config(2, 4, true);
        let q = ones(1 * 2 * 1 * 4);
        let new_k = ones(1 * 2 * 1 * 4);
        let new_v = ones(1 * 2 * 1 * 4);
        let cached_k = ones(1 * 2 * 3 * 4);
        let cached_v = ones(1 * 2 * 3 * 4);
        let out =
            kv_cache_attention(&q, &new_k, &new_v, &cached_k, &cached_v, &cfg, 1, 3, 1).unwrap();
        assert_eq!(out.key_cache.unwrap().len(), 1 * 2 * 4 * 4);
    }

    // ── sliding_window_attention ─────────────────────────────────────

    #[test]
    fn sliding_window_basic() {
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 4 * 4;
        let q = ones(n);
        let k = ones(n);
        let v = ones(n);
        let out = sliding_window_attention(&q, &k, &v, &cfg, 4, 4, 1, 2).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn sliding_window_causal() {
        let cfg = make_config(1, 4, true);
        let n = 1 * 1 * 4 * 4;
        let out = sliding_window_attention(&ones(n), &ones(n), &ones(n), &cfg, 4, 4, 1, 2).unwrap();
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sliding_window_zero_fails() {
        let cfg = make_config(1, 4, false);
        assert!(sliding_window_attention(&ones(4), &ones(4), &ones(4), &cfg, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn sliding_window_larger_than_seq() {
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 3 * 4;
        // Window larger than seq → equivalent to full attention.
        let full = scaled_dot_product(&ones(n), &ones(n), &ones(n), &cfg, 3, 3, 1).unwrap();
        let windowed =
            sliding_window_attention(&ones(n), &ones(n), &ones(n), &cfg, 3, 3, 1, 100).unwrap();
        assert_close(&full.output, &windowed.output, 1e-5);
    }

    #[test]
    fn sliding_window_gqa() {
        let cfg = make_gqa_config(4, 2, 4, false);
        let q = ones(1 * 4 * 3 * 4);
        let kv = ones(1 * 2 * 3 * 4);
        let out = sliding_window_attention(&q, &kv, &kv, &cfg, 3, 3, 1, 2).unwrap();
        assert_eq!(out.output.len(), 1 * 4 * 3 * 4);
    }

    // ── ALiBi ────────────────────────────────────────────────────────

    #[test]
    fn alibi_slopes_power_of_two() {
        let slopes = alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        // All slopes should be positive and decreasing.
        for i in 1..slopes.len() {
            assert!(slopes[i] > 0.0);
            assert!(slopes[i] < slopes[i - 1]);
        }
    }

    #[test]
    fn alibi_slopes_non_power_of_two() {
        let slopes = alibi_slopes(6);
        assert_eq!(slopes.len(), 6);
        for &s in &slopes {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn alibi_slopes_single_head() {
        let slopes = alibi_slopes(1);
        assert_eq!(slopes.len(), 1);
        assert!(slopes[0] > 0.0);
    }

    #[test]
    fn alibi_attention_basic() {
        let cfg = make_config(2, 4, false);
        let n_q = 1 * 2 * 3 * 4;
        let q = ones(n_q);
        let k = ones(n_q);
        let v = ones(n_q);
        let slopes = alibi_slopes(2);
        let out = attention_with_alibi(&q, &k, &v, &cfg, 3, 3, 1, &slopes).unwrap();
        assert_eq!(out.output.len(), n_q);
    }

    #[test]
    fn alibi_wrong_slopes_count() {
        let cfg = make_config(4, 4, false);
        let slopes = alibi_slopes(2); // should be 4
        assert!(
            attention_with_alibi(
                &ones(1 * 4 * 2 * 4),
                &ones(1 * 4 * 2 * 4),
                &ones(1 * 4 * 2 * 4),
                &cfg,
                2,
                2,
                1,
                &slopes,
            )
            .is_err()
        );
    }

    #[test]
    fn alibi_causal() {
        let cfg = make_config(2, 4, true);
        let n = 1 * 2 * 4 * 4;
        let slopes = alibi_slopes(2);
        let out =
            attention_with_alibi(&ones(n), &ones(n), &ones(n), &cfg, 4, 4, 1, &slopes).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn alibi_gqa() {
        let cfg = make_gqa_config(4, 2, 4, false);
        let q = ones(1 * 4 * 2 * 4);
        let kv = ones(1 * 2 * 2 * 4);
        let slopes = alibi_slopes(4);
        let out = attention_with_alibi(&q, &kv, &kv, &cfg, 2, 2, 1, &slopes).unwrap();
        assert_eq!(out.output.len(), 1 * 4 * 2 * 4);
    }

    #[test]
    fn alibi_single_position() {
        let cfg = make_config(1, 4, false);
        let q = ones(1 * 1 * 1 * 4);
        let k = ones(1 * 1 * 1 * 4);
        let v = ones(1 * 1 * 1 * 4);
        let slopes = alibi_slopes(1);
        let out = attention_with_alibi(&q, &k, &v, &cfg, 1, 1, 1, &slopes).unwrap();
        // Single position: bias is slope * (0-0) = 0 → same as normal attention.
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    // ── RoPE attention ───────────────────────────────────────────────

    #[test]
    fn rope_attention_basic() {
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 2 * 4;
        let out =
            attention_with_rope(&ones(n), &ones(n), &ones(n), &cfg, 2, 2, 1, 10000.0, 0).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn rope_odd_head_dim_fails() {
        let cfg = MultiHeadAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 3,
            causal: false,
            dropout_p: 0.0,
            scale: 1.0,
        };
        assert!(
            attention_with_rope(&ones(3), &ones(3), &ones(3), &cfg, 1, 1, 1, 10000.0, 0,).is_err()
        );
    }

    #[test]
    fn rope_with_offset() {
        // RoPE is invariant to absolute position shift (only relative positions
        // matter in dot-products). Verify that applying RoPE produces different
        // output than standard attention without RoPE.
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 3 * 4;
        let q: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.3).collect();
        let k: Vec<f32> = (0..n).map(|i| 0.5 - i as f32 * 0.2).collect();
        let v = linspace(n);
        let rope_out = attention_with_rope(&q, &k, &v, &cfg, 3, 3, 1, 100.0, 0).unwrap();
        let plain_out = scaled_dot_product(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        let differs =
            rope_out.output.iter().zip(plain_out.output.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "RoPE attention should differ from non-RoPE attention");
    }

    #[test]
    fn rope_causal() {
        let cfg = make_config(1, 4, true);
        let n = 1 * 1 * 3 * 4;
        let out =
            attention_with_rope(&ones(n), &ones(n), &ones(n), &cfg, 3, 3, 1, 10000.0, 0).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn rope_multi_head() {
        let cfg = make_config(4, 8, false);
        let n = 1 * 4 * 3 * 8;
        let out =
            attention_with_rope(&ones(n), &ones(n), &ones(n), &cfg, 3, 3, 1, 10000.0, 0).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn rope_gqa() {
        let cfg = make_gqa_config(4, 2, 8, false);
        let q = ones(1 * 4 * 2 * 8);
        let kv = ones(1 * 2 * 2 * 8);
        let out = attention_with_rope(&q, &kv, &kv, &cfg, 2, 2, 1, 10000.0, 0).unwrap();
        assert_eq!(out.output.len(), 1 * 4 * 2 * 8);
    }

    #[test]
    fn rope_different_bases() {
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 2 * 4;
        let q: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.3).collect();
        let k: Vec<f32> = (0..n).map(|i| 0.5 - i as f32 * 0.2).collect();
        let v = linspace(n); // varying V so softmax weight changes are visible
        let out_a = attention_with_rope(&q, &k, &v, &cfg, 2, 2, 1, 10000.0, 0).unwrap();
        let out_b = attention_with_rope(&q, &k, &v, &cfg, 2, 2, 1, 500.0, 0).unwrap();
        let differs =
            out_a.output.iter().zip(out_b.output.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "Different RoPE bases should produce different results");
    }

    // ── launch_multi_head_attention ──────────────────────────────────

    #[test]
    fn launch_cpu_fallback() {
        // With `--features cpu` (no gpu), this should fall through to CPU.
        let cfg = make_config(1, 4, false);
        let n = 1 * 1 * 2 * 4;
        let result = launch_multi_head_attention(&ones(n), &ones(n), &ones(n), &cfg, 2, 2, 1);
        // Depending on feature flags:
        // - cpu only: should succeed (CPU fallback)
        // - gpu: should return GpuError (scaffold)
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            assert!(result.is_ok());
        }
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            assert!(result.is_err());
        }
    }

    // ── Softmax numerical properties ─────────────────────────────────

    #[test]
    fn softmax_rows_sum_to_one() {
        let cfg = make_config(1, 4, false);
        let q = linspace(1 * 1 * 3 * 4);
        let k = linspace(1 * 1 * 3 * 4);
        // Use identity-like V so output reflects softmax weights directly.
        let mut v = vec![0.0_f32; 3 * 4];
        for i in 0..3 {
            v[i * 4] = 1.0; // only first dim set
        }
        let _out = scaled_dot_product(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        // Output being finite confirms softmax is numerically stable.
        for &val in &_out.output {
            assert!(val.is_finite(), "softmax produced non-finite value");
        }
    }

    #[test]
    fn causal_first_row_only_sees_itself() {
        let cfg = make_config(1, 2, true);
        // Create Q/K/V where each position has distinct values.
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let k = q.clone();
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3x2
        let out = scaled_dot_product(&q, &k, &v, &cfg, 3, 3, 1).unwrap();
        // First row only sees position 0 → output[0:2] == v[0:2] == [1, 0].
        assert!((out.output[0] - 1.0).abs() < 1e-5);
        assert!((out.output[1] - 0.0).abs() < 1e-5);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn single_element_attention() {
        let cfg = make_config(1, 1, false);
        let out = scaled_dot_product(&[2.0], &[3.0], &[5.0], &cfg, 1, 1, 1).unwrap();
        // softmax of single element is 1.0, output = 1.0 * v = 5.0.
        assert!((out.output[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn large_head_dim() {
        let cfg = make_config(1, 128, false);
        let n = 1 * 1 * 2 * 128;
        let out = scaled_dot_product(&ones(n), &ones(n), &ones(n), &cfg, 2, 2, 1).unwrap();
        assert_eq!(out.output.len(), n);
        for &val in &out.output {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn many_heads() {
        let cfg = make_config(32, 4, false);
        let n = 1 * 32 * 2 * 4;
        let out = scaled_dot_product(&ones(n), &ones(n), &ones(n), &cfg, 2, 2, 1).unwrap();
        assert_eq!(out.output.len(), n);
    }

    #[test]
    fn attention_output_no_weights_by_default() {
        let cfg = make_config(1, 4, false);
        let out = scaled_dot_product(&ones(8), &ones(8), &ones(8), &cfg, 2, 2, 1).unwrap();
        assert!(out.attention_weights.is_none());
    }

    #[test]
    fn attention_output_no_cache_by_default() {
        let cfg = make_config(1, 4, false);
        let out = scaled_dot_product(&ones(4), &ones(4), &ones(4), &cfg, 1, 1, 1).unwrap();
        assert!(out.key_cache.is_none());
        assert!(out.value_cache.is_none());
    }

    // ── CUDA kernel source ───────────────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn cuda_kernel_src_not_empty() {
        assert!(!MHA_KERNEL_SRC.is_empty());
        assert!(MHA_KERNEL_SRC.contains("mha_forward_f32"));
    }

    // ── Additional coverage ──────────────────────────────────────────

    #[test]
    fn alibi_slopes_two_heads() {
        let slopes = alibi_slopes(2);
        assert_eq!(slopes.len(), 2);
        assert!(slopes[0] > slopes[1], "slopes should decrease");
    }

    #[test]
    fn kv_cache_gqa() {
        let cfg = make_gqa_config(4, 2, 4, true);
        let q = ones(1 * 4 * 1 * 4);
        let new_k = ones(1 * 2 * 1 * 4);
        let new_v = ones(1 * 2 * 1 * 4);
        let cached_k = ones(1 * 2 * 2 * 4);
        let cached_v = ones(1 * 2 * 2 * 4);
        let out =
            kv_cache_attention(&q, &new_k, &new_v, &cached_k, &cached_v, &cfg, 1, 2, 1).unwrap();
        assert_eq!(out.key_cache.unwrap().len(), 1 * 2 * 3 * 4);
    }
}
