//! OpenCL-optimized scaled dot-product attention with CPU reference.
//!
//! # Overview
//!
//! This module implements the core attention mechanism used in transformer-based
//! LLMs: `softmax(Q @ K^T / sqrt(d_k)) @ V`. It provides:
//!
//! - **CPU reference** — scalar implementations for correctness testing and
//!   non-GPU environments.
//! - **Multi-head attention** — independent per-head computation.
//! - **Grouped-query attention (GQA)** — `n_heads != n_kv_heads` with
//!   key/value sharing across query head groups.
//! - **KV cache** — incremental key/value storage for autoregressive decoding.
//! - **Causal masking** — lower-triangular mask to prevent attending to future
//!   tokens.
//! - **OpenCL kernel source** — ready for GPU dispatch on Intel / AMD / other
//!   OpenCL-capable devices.
//!
//! # OpenCL kernel
//!
//! The embedded OpenCL C source (`ATTENTION_CL`) contains a single-workitem-per-
//! query-row kernel suitable for functional correctness testing. A tiled /
//! flash-attention variant is planned for v0.3.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for the attention mechanism.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (may differ from `num_heads` for GQA).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length (used to pre-allocate KV cache).
    pub max_seq_len: usize,
    /// Whether to apply a causal (lower-triangular) mask.
    pub causal: bool,
    /// Scaling factor applied to dot-product scores.
    /// Typically `1.0 / sqrt(head_dim)`.
    pub scale: f32,
}

impl AttentionConfig {
    /// Create a standard multi-head attention config (num_kv_heads == num_heads).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        causal: bool,
    ) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_heads, head_dim, and max_seq_len must all be > 0".into(),
            }
            .into());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self { num_heads, num_kv_heads: num_heads, head_dim, max_seq_len, causal, scale })
    }

    /// Create a grouped-query attention config where `num_kv_heads` may differ.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if dimensions are zero or if
    /// `num_heads` is not divisible by `num_kv_heads`.
    pub fn new_gqa(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        causal: bool,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all dimensions must be > 0".into(),
            }
            .into());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            }
            .into());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        Ok(Self { num_heads, num_kv_heads, head_dim, max_seq_len, causal, scale })
    }

    /// Number of query heads that share each KV head.
    #[inline]
    pub fn heads_per_kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Returns true when this is a grouped-query configuration.
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Attention mask
// ---------------------------------------------------------------------------

/// Lower-triangular causal mask for autoregressive attention.
///
/// `mask[i][j] == true` means position `i` **may** attend to position `j`.
#[derive(Debug, Clone)]
pub struct AttentionMask {
    /// Row-major `[seq_len, kv_len]` boolean mask.
    mask: Vec<bool>,
    pub seq_len: usize,
    pub kv_len: usize,
}

impl AttentionMask {
    /// Generate a causal mask where each query position can only attend to
    /// itself and earlier key positions.
    ///
    /// With `offset`, the mask accounts for KV cache positions preceding the
    /// current query window (i.e. `query_pos = offset + i`).
    pub fn causal(seq_len: usize, kv_len: usize, offset: usize) -> Self {
        let mut mask = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let query_pos = offset + i;
            for j in 0..kv_len {
                // Allow attending to positions up to and including query_pos.
                mask[i * kv_len + j] = j <= query_pos;
            }
        }
        Self { mask, seq_len, kv_len }
    }

    /// Fully permissive mask (no masking).
    pub fn none(seq_len: usize, kv_len: usize) -> Self {
        Self { mask: vec![true; seq_len * kv_len], seq_len, kv_len }
    }

    /// Check whether query position `i` may attend to key position `j`.
    #[inline]
    pub fn allows(&self, i: usize, j: usize) -> bool {
        self.mask[i * self.kv_len + j]
    }
}

// ---------------------------------------------------------------------------
// Attention scores
// ---------------------------------------------------------------------------

/// Attention score matrix with optional masking applied.
#[derive(Debug, Clone)]
pub struct AttentionScores {
    /// Row-major `[seq_len, kv_len]` scores **after** softmax.
    pub weights: Vec<f32>,
    pub seq_len: usize,
    pub kv_len: usize,
}

impl AttentionScores {
    /// Compute raw (pre-softmax) attention scores: `Q @ K^T * scale`.
    pub fn compute_raw(
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Self {
        let mut weights = vec![0.0f32; seq_len * kv_len];
        for i in 0..seq_len {
            for j in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                weights[i * kv_len + j] = dot * scale;
            }
        }
        Self { weights, seq_len, kv_len }
    }

    /// Apply mask: set disallowed positions to `-inf`.
    pub fn apply_mask(&mut self, mask: &AttentionMask) {
        assert_eq!(self.seq_len, mask.seq_len);
        assert_eq!(self.kv_len, mask.kv_len);
        for i in 0..self.seq_len {
            for j in 0..self.kv_len {
                if !mask.allows(i, j) {
                    self.weights[i * self.kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// In-place row-wise softmax.
    pub fn softmax(&mut self) {
        for i in 0..self.seq_len {
            let row = &mut self.weights[i * self.kv_len..(i + 1) * self.kv_len];
            softmax_row(row);
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// Key-value cache entry for a single attention head.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Cached keys: `[current_len, head_dim]` row-major.
    pub keys: Vec<f32>,
    /// Cached values: `[current_len, head_dim]` row-major.
    pub values: Vec<f32>,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of tokens currently stored.
    pub current_len: usize,
    /// Maximum capacity.
    pub max_len: usize,
}

impl KVCacheEntry {
    /// Allocate an empty cache for one head.
    pub fn new(head_dim: usize, max_len: usize) -> Self {
        Self {
            keys: vec![0.0; max_len * head_dim],
            values: vec![0.0; max_len * head_dim],
            head_dim,
            current_len: 0,
            max_len,
        }
    }

    /// Append new key/value vectors (one per token in the new chunk).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if the cache would overflow.
    pub fn append(&mut self, new_keys: &[f32], new_values: &[f32]) -> Result<()> {
        let num_new = new_keys.len() / self.head_dim;
        if new_keys.len() != num_new * self.head_dim || new_values.len() != num_new * self.head_dim
        {
            return Err(KernelError::InvalidArguments {
                reason: "key/value length must be a multiple of head_dim".into(),
            }
            .into());
        }
        if self.current_len + num_new > self.max_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "KV cache overflow: {} + {} > {}",
                    self.current_len, num_new, self.max_len
                ),
            }
            .into());
        }
        let start = self.current_len * self.head_dim;
        let end = start + num_new * self.head_dim;
        self.keys[start..end].copy_from_slice(new_keys);
        self.values[start..end].copy_from_slice(new_values);
        self.current_len += num_new;
        Ok(())
    }

    /// Return the currently filled key slice.
    #[inline]
    pub fn keys(&self) -> &[f32] {
        &self.keys[..self.current_len * self.head_dim]
    }

    /// Return the currently filled value slice.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values[..self.current_len * self.head_dim]
    }

    /// Reset cache to empty.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

// ---------------------------------------------------------------------------
// Flash attention config (tile-based)
// ---------------------------------------------------------------------------

/// Configuration for tiled / flash-attention style computation.
///
/// Flash attention splits Q, K, V into tiles to improve memory locality and
/// avoid materialising the full `[seq_len, kv_len]` score matrix.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Tile size along the query (row) dimension.
    pub tile_q: usize,
    /// Tile size along the key/value (column) dimension.
    pub tile_kv: usize,
    /// Whether to apply causal masking inside tiles.
    pub causal: bool,
}

impl FlashAttentionConfig {
    /// Create a config with default tile sizes suitable for OpenCL local memory.
    pub fn default_opencl() -> Self {
        Self { tile_q: 32, tile_kv: 32, causal: true }
    }

    /// Create a custom tiled configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if tile sizes are zero.
    pub fn new(tile_q: usize, tile_kv: usize, causal: bool) -> Result<Self> {
        if tile_q == 0 || tile_kv == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "tile sizes must be > 0".into() }.into()
            );
        }
        Ok(Self { tile_q, tile_kv, causal })
    }
}

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------

/// Numerically stable softmax over a single row (in-place).
fn softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        // All masked — distribute uniformly (or zero).
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: scaled dot-product attention (single head)
// ---------------------------------------------------------------------------

/// Scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`
///
/// This is the CPU reference implementation for a **single attention head**.
///
/// # Arguments
///
/// * `q`        — query matrix `[seq_len, head_dim]`, row-major.
/// * `k`        — key matrix   `[kv_len, head_dim]`, row-major.
/// * `v`        — value matrix `[kv_len, head_dim]`, row-major.
/// * `output`   — result       `[seq_len, head_dim]`, row-major.
/// * `seq_len`  — number of query positions.
/// * `kv_len`   — number of key/value positions.
/// * `head_dim` — dimensionality of each head.
/// * `scale`    — scale factor (typically `1/sqrt(head_dim)`).
/// * `causal`   — if true, apply a lower-triangular causal mask.
///
/// # Panics
///
/// Panics if slice lengths are inconsistent with the given dimensions.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    assert_eq!(q.len(), seq_len * head_dim, "Q size mismatch");
    assert_eq!(k.len(), kv_len * head_dim, "K size mismatch");
    assert_eq!(v.len(), kv_len * head_dim, "V size mismatch");
    assert_eq!(output.len(), seq_len * head_dim, "output size mismatch");

    // Score buffer for one query row.
    let mut scores = vec![0.0f32; kv_len];

    for i in 0..seq_len {
        // 1. Compute Q[i] @ K^T => scores[j]
        for j in 0..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }

        // 2. Apply causal mask.
        if causal {
            for score in &mut scores[(i + 1)..kv_len] {
                *score = f32::NEG_INFINITY;
            }
        }

        // 3. Softmax.
        softmax_row(&mut scores);

        // 4. Weighted sum of V.
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..kv_len {
                acc += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention CPU reference
// ---------------------------------------------------------------------------

/// Multi-head attention CPU reference (all heads, MHA or GQA).
///
/// # Layout
///
/// * `q`      — `[seq_len, num_heads * head_dim]` row-major.
/// * `k`      — `[kv_len, num_kv_heads * head_dim]` row-major.
/// * `v`      — `[kv_len, num_kv_heads * head_dim]` row-major.
/// * `output` — `[seq_len, num_heads * head_dim]` row-major.
pub struct MultiHeadAttentionRef;

impl MultiHeadAttentionRef {
    /// Execute multi-head (or grouped-query) attention.
    pub fn forward(
        config: &AttentionConfig,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        kv_len: usize,
    ) {
        let hd = config.head_dim;
        let heads_per_group = config.heads_per_kv_group();

        for h in 0..config.num_heads {
            let kv_head = h / heads_per_group;

            // Extract per-head slices.
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let start = t * config.num_heads * hd + h * hd;
                    q[start..start + hd].iter().copied()
                })
                .collect();

            let k_head: Vec<f32> = (0..kv_len)
                .flat_map(|t| {
                    let start = t * config.num_kv_heads * hd + kv_head * hd;
                    k[start..start + hd].iter().copied()
                })
                .collect();

            let v_head: Vec<f32> = (0..kv_len)
                .flat_map(|t| {
                    let start = t * config.num_kv_heads * hd + kv_head * hd;
                    v[start..start + hd].iter().copied()
                })
                .collect();

            let mut out_head = vec![0.0f32; seq_len * hd];

            scaled_dot_product_attention_ref(
                &q_head,
                &k_head,
                &v_head,
                &mut out_head,
                seq_len,
                kv_len,
                hd,
                config.scale,
                config.causal,
            );

            // Scatter back into the interleaved output.
            for t in 0..seq_len {
                let dst_start = t * config.num_heads * hd + h * hd;
                output[dst_start..dst_start + hd].copy_from_slice(&out_head[t * hd..(t + 1) * hd]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped-query attention helper
// ---------------------------------------------------------------------------

/// Grouped-query attention: a thin wrapper that validates GQA constraints
/// before delegating to [`MultiHeadAttentionRef`].
pub struct GroupedQueryAttention;

impl GroupedQueryAttention {
    /// Execute GQA with explicit head counts.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_heads % num_kv_heads != 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        kv_len: usize,
        causal: bool,
    ) -> Result<()> {
        let config = AttentionConfig::new_gqa(
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len.max(kv_len),
            causal,
        )?;
        MultiHeadAttentionRef::forward(&config, q, k, v, output, seq_len, kv_len);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for scaled dot-product attention.
///
/// One work-item per query row. Each work-item computes the full dot-product
/// against all key positions, applies optional causal masking and softmax,
/// then writes the weighted sum of values to the output.
pub const ATTENTION_CL: &str = r#"
__kernel void scaled_dot_product_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const float scale,
    const int causal)
{
    int seq_idx = get_global_id(0);
    if (seq_idx >= seq_len) return;

    // --- Compute attention scores: Q[seq_idx] dot K[j] * scale -----------
    // We use a local buffer on private memory (OpenCL C VLAs are optional,
    // so we cap at a compile-time maximum and guard at runtime).
    float scores[4096]; // max kv_len supported in this simple kernel
    if (kv_len > 4096) return; // safety guard

    for (int j = 0; j < kv_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[seq_idx * head_dim + d] * K[j * head_dim + d];
        }
        scores[j] = dot * scale;
    }

    // --- Causal mask: set future positions to -inf ---------------------
    if (causal) {
        for (int j = seq_idx + 1; j < kv_len; j++) {
            scores[j] = -1e30f; // approximation of -inf for exp()
        }
    }

    // --- Numerically-stable softmax -----------------------------------
    float max_score = scores[0];
    for (int j = 1; j < kv_len; j++) {
        if (scores[j] > max_score) max_score = scores[j];
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < kv_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }

    if (sum_exp > 0.0f) {
        for (int j = 0; j < kv_len; j++) {
            scores[j] /= sum_exp;
        }
    }

    // --- Weighted sum of values: output = scores @ V -------------------
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            acc += scores[j] * V[j * head_dim + d];
        }
        output[seq_idx * head_dim + d] = acc;
    }
}

__kernel void multi_head_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const float scale,
    const int causal)
{
    int seq_idx = get_global_id(0);
    int head_idx = get_global_id(1);
    if (seq_idx >= seq_len || head_idx >= num_heads) return;

    int heads_per_group = num_heads / num_kv_heads;
    int kv_head = head_idx / heads_per_group;

    float scores[4096];
    if (kv_len > 4096) return;

    // Q offset: [seq_idx, head_idx, :] in [seq_len, num_heads, head_dim]
    int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    // K offset: [j, kv_head, :] in [kv_len, num_kv_heads, head_dim]

    for (int j = 0; j < kv_len; j++) {
        float dot = 0.0f;
        int k_offset = j * num_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[q_offset + d] * K[k_offset + d];
        }
        scores[j] = dot * scale;
    }

    if (causal) {
        for (int j = seq_idx + 1; j < kv_len; j++) {
            scores[j] = -1e30f;
        }
    }

    float max_score = scores[0];
    for (int j = 1; j < kv_len; j++) {
        if (scores[j] > max_score) max_score = scores[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < kv_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }
    if (sum_exp > 0.0f) {
        for (int j = 0; j < kv_len; j++) {
            scores[j] /= sum_exp;
        }
    }

    int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        int v_base = kv_head * head_dim + d;
        for (int j = 0; j < kv_len; j++) {
            acc += scores[j] * V[j * num_kv_heads * head_dim + v_base];
        }
        output[out_offset + d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers --------------------------------------------------------

    /// Absolute tolerance for float comparisons.
    const ATOL: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_slices_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(*x, *y, tol), "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    /// Identity-like values: K == Q, V == identity rows.
    fn make_identity_kv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0f32; seq_len * head_dim];
        let mut v = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                // Use one-hot-ish values so attention-to-self recovers V.
                k[i * head_dim + d] = if d == i % head_dim { 1.0 } else { 0.0 };
                v[i * head_dim + d] = (i * head_dim + d) as f32;
            }
        }
        (k, v)
    }

    // ---- AttentionConfig ------------------------------------------------

    #[test]
    fn test_config_new_valid() {
        let cfg = AttentionConfig::new(8, 64, 512, true).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert!(!cfg.is_gqa());
        assert_eq!(cfg.heads_per_kv_group(), 1);
    }

    #[test]
    fn test_config_new_zero_heads() {
        assert!(AttentionConfig::new(0, 64, 512, true).is_err());
    }

    #[test]
    fn test_config_new_zero_head_dim() {
        assert!(AttentionConfig::new(8, 0, 512, true).is_err());
    }

    #[test]
    fn test_config_new_zero_max_seq() {
        assert!(AttentionConfig::new(8, 64, 0, true).is_err());
    }

    #[test]
    fn test_config_scale_value() {
        let cfg = AttentionConfig::new(1, 64, 32, false).unwrap();
        let expected = 1.0 / (64.0f32).sqrt();
        assert!(approx_eq(cfg.scale, expected, 1e-7));
    }

    #[test]
    fn test_config_gqa_valid() {
        let cfg = AttentionConfig::new_gqa(8, 2, 64, 512, true).unwrap();
        assert!(cfg.is_gqa());
        assert_eq!(cfg.heads_per_kv_group(), 4);
    }

    #[test]
    fn test_config_gqa_indivisible() {
        assert!(AttentionConfig::new_gqa(7, 2, 64, 512, true).is_err());
    }

    #[test]
    fn test_config_gqa_zero_kv_heads() {
        assert!(AttentionConfig::new_gqa(8, 0, 64, 512, true).is_err());
    }

    #[test]
    fn test_config_mha_equals_gqa_with_same_heads() {
        let mha = AttentionConfig::new(4, 64, 128, true).unwrap();
        let gqa = AttentionConfig::new_gqa(4, 4, 64, 128, true).unwrap();
        assert_eq!(mha.num_heads, gqa.num_heads);
        assert_eq!(mha.num_kv_heads, gqa.num_kv_heads);
        assert!(!gqa.is_gqa());
    }

    // ---- AttentionMask --------------------------------------------------

    #[test]
    fn test_causal_mask_basic() {
        let mask = AttentionMask::causal(3, 3, 0);
        // Row 0: can attend to [0] only
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1));
        assert!(!mask.allows(0, 2));
        // Row 1: can attend to [0, 1]
        assert!(mask.allows(1, 0));
        assert!(mask.allows(1, 1));
        assert!(!mask.allows(1, 2));
        // Row 2: can attend to [0, 1, 2]
        assert!(mask.allows(2, 0));
        assert!(mask.allows(2, 1));
        assert!(mask.allows(2, 2));
    }

    #[test]
    fn test_causal_mask_with_offset() {
        // Simulating kv_len=5 cache, seq_len=2 new tokens starting at offset=3.
        let mask = AttentionMask::causal(2, 5, 3);
        // Query 0 (position 3): can attend to [0..3]
        assert!(mask.allows(0, 0));
        assert!(mask.allows(0, 1));
        assert!(mask.allows(0, 2));
        assert!(mask.allows(0, 3));
        assert!(!mask.allows(0, 4));
        // Query 1 (position 4): can attend to [0..4]
        assert!(mask.allows(1, 4));
    }

    #[test]
    fn test_no_mask() {
        let mask = AttentionMask::none(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(mask.allows(i, j));
            }
        }
    }

    #[test]
    fn test_causal_mask_single_token() {
        let mask = AttentionMask::causal(1, 1, 0);
        assert!(mask.allows(0, 0));
    }

    #[test]
    fn test_causal_mask_asymmetric() {
        let mask = AttentionMask::causal(2, 4, 0);
        // Row 0: [true, false, false, false]
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1));
        // Row 1: [true, true, false, false]
        assert!(mask.allows(1, 0));
        assert!(mask.allows(1, 1));
        assert!(!mask.allows(1, 2));
    }

    // ---- softmax --------------------------------------------------------

    #[test]
    fn test_softmax_uniform() {
        let mut row = vec![1.0, 1.0, 1.0, 1.0];
        softmax_row(&mut row);
        for v in &row {
            assert!(approx_eq(*v, 0.25, ATOL));
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, ATOL));
    }

    #[test]
    fn test_softmax_single_element() {
        let mut row = vec![42.0];
        softmax_row(&mut row);
        assert!(approx_eq(row[0], 1.0, ATOL));
    }

    #[test]
    fn test_softmax_empty() {
        let mut row: Vec<f32> = vec![];
        softmax_row(&mut row);
        assert!(row.is_empty());
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let mut row = vec![1.0, f32::NEG_INFINITY, 1.0];
        softmax_row(&mut row);
        assert!(approx_eq(row[1], 0.0, ATOL));
        assert!(approx_eq(row[0], 0.5, ATOL));
        assert!(approx_eq(row[2], 0.5, ATOL));
    }

    #[test]
    fn test_softmax_all_neg_inf() {
        let mut row = vec![f32::NEG_INFINITY; 4];
        softmax_row(&mut row);
        for v in &row {
            assert!(approx_eq(*v, 0.0, ATOL));
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let mut row = vec![1000.0, 1001.0, 1002.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, ATOL));
        // Largest input should get largest probability.
        assert!(row[2] > row[1]);
        assert!(row[1] > row[0]);
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut row = vec![-1.0, -2.0, -3.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, ATOL));
        assert!(row[0] > row[1]);
        assert!(row[1] > row[2]);
    }

    // ---- scaled_dot_product_attention_ref --------------------------------

    #[test]
    fn test_sdpa_single_token_no_causal() {
        // seq_len=1, kv_len=1, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![3.0, 7.0];
        let mut out = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 1, 2, 1.0, false);
        // softmax([1.0]) = [1.0], output = 1.0 * V = V
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_sdpa_identity_attention() {
        // When Q == K with orthogonal one-hot patterns, each query should
        // attend most strongly to the matching key position.
        let head_dim = 4;
        let seq_len = 3;
        let (k, v) = make_identity_kv(seq_len, head_dim);
        let q = k.clone(); // Q == K
        let mut out = vec![0.0; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut out, seq_len, seq_len, head_dim, scale, false,
        );
        // Verify no NaN and outputs are finite.
        for val in &out {
            assert!(!val.is_nan(), "output contains NaN");
            assert!(val.is_finite(), "output contains Inf");
        }
        // The self-matching key has dot=1*scale, others have dot=0.
        // softmax distributes more weight on self → output closer to own V row.
        // Verify the self-match row gets the highest weight by checking that
        // the self-position score (Q[i] · K[i]) is the maximum for each row.
        for i in 0..seq_len {
            let scores: Vec<f32> = (0..seq_len)
                .map(|j| {
                    (0..head_dim).map(|d| q[i * head_dim + d] * k[j * head_dim + d]).sum::<f32>()
                })
                .collect();
            let self_score = scores[i];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (self_score - max_score).abs() < ATOL,
                "row {i}: self-score {self_score} is not the max {max_score}"
            );
        }
    }

    #[test]
    fn test_sdpa_causal_single_token() {
        let q = vec![1.0, 2.0];
        let k = vec![1.0, 2.0];
        let v = vec![5.0, 6.0];
        let mut out = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 1, 2, 1.0, true);
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_sdpa_causal_two_tokens() {
        let head_dim = 2;
        let seq_len = 2;
        let kv_len = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut out, seq_len, kv_len, head_dim, 1.0, true,
        );
        // Token 0 with causal: can only see token 0 → output = V[0]
        assert_slices_close(&out[0..2], &[1.0, 2.0], ATOL);
        // Token 1 with causal: can see tokens 0 and 1 → softmax blend.
    }

    #[test]
    fn test_sdpa_causal_blocks_future() {
        // With causal mask and 3 tokens, token 0 should produce same output
        // regardless of what's in positions 1 and 2.
        let head_dim = 2;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v2 = vec![1.0, 2.0, 99.0, 99.0, 99.0, 99.0];
        let mut out1 = vec![0.0; 6];
        let mut out2 = vec![0.0; 6];
        scaled_dot_product_attention_ref(&q, &k, &v1, &mut out1, 3, 3, head_dim, 1.0, true);
        scaled_dot_product_attention_ref(&q, &k, &v2, &mut out2, 3, 3, head_dim, 1.0, true);
        // Token 0's output should be identical in both cases.
        assert_slices_close(&out1[0..2], &out2[0..2], ATOL);
    }

    #[test]
    fn test_sdpa_scale_factor() {
        // Verify that scaling is applied (different scale → different output).
        let q = vec![2.0, 0.0];
        let k = vec![2.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out_s1 = vec![0.0; 2];
        let mut out_s01 = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out_s1, 1, 2, 2, 1.0, false);
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out_s01, 1, 2, 2, 0.1, false);
        // With scale=1.0, dot with first key is 4.0, with scale=0.1 it's 0.4.
        // The softmax distribution should differ.
        assert!((out_s1[0] - out_s01[0]).abs() > 0.01, "scale should affect output");
    }

    #[test]
    fn test_sdpa_uniform_keys() {
        // All keys identical → uniform attention → output = mean of V rows.
        let head_dim = 3;
        let kv_len = 4;
        let k: Vec<f32> = (0..kv_len).flat_map(|_| [1.0, 0.0, 0.0]).collect();
        let q = vec![1.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0; 3];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, kv_len, head_dim, 1.0, false);
        // Uniform attention → mean of V columns.
        let expected: Vec<f32> = (0..head_dim)
            .map(|d| (0..kv_len).map(|j| v[j * head_dim + d]).sum::<f32>() / kv_len as f32)
            .collect();
        assert_slices_close(&out, &expected, ATOL);
    }

    #[test]
    fn test_sdpa_head_dim_1() {
        let q = vec![1.0];
        let k = vec![1.0, 2.0];
        let v = vec![10.0, 20.0];
        let mut out = vec![0.0];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 2, 1, 1.0, false);
        // scores = [1.0, 2.0], softmax → output is weighted sum of [10, 20].
        let e1 = 1.0f32.exp();
        let e2 = 2.0f32.exp();
        let s = e1 + e2;
        let expected = (e1 / s) * 10.0 + (e2 / s) * 20.0;
        assert!(approx_eq(out[0], expected, ATOL));
    }

    #[test]
    fn test_sdpa_seq_len_1_kv_len_1() {
        let q = vec![0.5, 0.5, 0.5, 0.5];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 1, 4, 0.5, false);
        // Single KV → softmax([score]) = [1.0] → output = V
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_sdpa_zero_query() {
        // Zero query → all dot products equal → uniform attention.
        let head_dim = 2;
        let kv_len = 3;
        let q = vec![0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, kv_len, head_dim, 1.0, false);
        let expected: Vec<f32> = (0..head_dim)
            .map(|d| (0..kv_len).map(|j| v[j * head_dim + d]).sum::<f32>() / kv_len as f32)
            .collect();
        assert_slices_close(&out, &expected, ATOL);
    }

    #[test]
    fn test_sdpa_numerical_stability_large_values() {
        // Large values in Q/K should not produce NaN thanks to stable softmax.
        let q = vec![100.0, 100.0];
        let k = vec![100.0, 100.0, -100.0, -100.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 2, 2, 1.0, false);
        for val in &out {
            assert!(!val.is_nan(), "output contains NaN");
            assert!(!val.is_infinite(), "output contains Inf");
        }
    }

    #[test]
    fn test_sdpa_numerical_stability_very_large() {
        let q = vec![500.0; 4];
        let k = vec![500.0; 4];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 1, 4, 0.01, false);
        for val in &out {
            assert!(!val.is_nan(), "NaN with very large values");
        }
    }

    // ---- AttentionScores ------------------------------------------------

    #[test]
    fn test_scores_compute_raw() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = AttentionScores::compute_raw(&q, &k, 1, 2, 2, 1.0);
        assert!(approx_eq(scores.weights[0], 1.0, ATOL));
        assert!(approx_eq(scores.weights[1], 0.0, ATOL));
    }

    #[test]
    fn test_scores_with_mask_and_softmax() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let mut scores = AttentionScores::compute_raw(&q, &k, 2, 2, 2, 1.0);
        let mask = AttentionMask::causal(2, 2, 0);
        scores.apply_mask(&mask);
        scores.softmax();
        // Row 0: only position 0 allowed → weight = 1.0.
        assert!(approx_eq(scores.weights[0], 1.0, ATOL));
        assert!(approx_eq(scores.weights[1], 0.0, ATOL));
        // Row 1: positions 0 and 1 allowed.
        let sum: f32 = scores.weights[2..4].iter().sum();
        assert!(approx_eq(sum, 1.0, ATOL));
    }

    #[test]
    fn test_scores_scale_applied() {
        let q = vec![2.0, 0.0];
        let k = vec![3.0, 0.0];
        let scores = AttentionScores::compute_raw(&q, &k, 1, 1, 2, 0.5);
        assert!(approx_eq(scores.weights[0], 3.0, ATOL)); // 2*3*0.5 = 3.0
    }

    // ---- KVCacheEntry ---------------------------------------------------

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCacheEntry::new(4, 8);
        assert_eq!(cache.current_len, 0);
        cache.append(&[1.0; 4], &[2.0; 4]).unwrap();
        assert_eq!(cache.current_len, 1);
        assert_eq!(cache.keys().len(), 4);
        assert_eq!(cache.values().len(), 4);
    }

    #[test]
    fn test_kv_cache_multi_append() {
        let mut cache = KVCacheEntry::new(2, 10);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(cache.current_len, 2);
        cache.append(&[9.0, 10.0], &[11.0, 12.0]).unwrap();
        assert_eq!(cache.current_len, 3);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KVCacheEntry::new(2, 2);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(cache.append(&[1.0, 2.0], &[3.0, 4.0]).is_err());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCacheEntry::new(2, 4);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.clear();
        assert_eq!(cache.current_len, 0);
        assert!(cache.keys().is_empty());
    }

    #[test]
    fn test_kv_cache_bad_length() {
        let mut cache = KVCacheEntry::new(4, 8);
        // 3 is not divisible by head_dim=4.
        assert!(cache.append(&[1.0; 3], &[2.0; 3]).is_err());
    }

    #[test]
    fn test_kv_cache_values_preserved() {
        let mut cache = KVCacheEntry::new(2, 4);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
        assert_eq!(cache.keys(), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cache.values(), &[3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_incremental_attention() {
        // Simulate incremental decoding: prefill 2 tokens, then decode 1 more.
        let head_dim = 2;
        let max_len = 8;
        let mut cache = KVCacheEntry::new(head_dim, max_len);

        // Prefill: 2 tokens.
        let k_prefill = vec![1.0, 0.0, 0.0, 1.0];
        let v_prefill = vec![1.0, 2.0, 3.0, 4.0];
        cache.append(&k_prefill, &v_prefill).unwrap();

        let q_prefill = vec![1.0, 0.0, 0.0, 1.0];
        let mut out_prefill = vec![0.0; 4];
        scaled_dot_product_attention_ref(
            &q_prefill,
            cache.keys(),
            cache.values(),
            &mut out_prefill,
            2,
            cache.current_len,
            head_dim,
            1.0,
            true,
        );

        // Decode step: 1 new token.
        let k_new = vec![1.0, 1.0];
        let v_new = vec![5.0, 6.0];
        cache.append(&k_new, &v_new).unwrap();
        assert_eq!(cache.current_len, 3);

        let q_new = vec![1.0, 1.0];
        let mut out_decode = vec![0.0; 2];
        scaled_dot_product_attention_ref(
            &q_new,
            cache.keys(),
            cache.values(),
            &mut out_decode,
            1,
            cache.current_len,
            head_dim,
            1.0,
            false,
        );
        for val in &out_decode {
            assert!(!val.is_nan());
        }
    }

    // ---- FlashAttentionConfig -------------------------------------------

    #[test]
    fn test_flash_config_default() {
        let cfg = FlashAttentionConfig::default_opencl();
        assert_eq!(cfg.tile_q, 32);
        assert_eq!(cfg.tile_kv, 32);
        assert!(cfg.causal);
    }

    #[test]
    fn test_flash_config_custom() {
        let cfg = FlashAttentionConfig::new(16, 64, false).unwrap();
        assert_eq!(cfg.tile_q, 16);
        assert_eq!(cfg.tile_kv, 64);
        assert!(!cfg.causal);
    }

    #[test]
    fn test_flash_config_zero_tile() {
        assert!(FlashAttentionConfig::new(0, 32, true).is_err());
        assert!(FlashAttentionConfig::new(32, 0, true).is_err());
    }

    // ---- MultiHeadAttentionRef ------------------------------------------

    #[test]
    fn test_mha_single_head() {
        let cfg = AttentionConfig::new(1, 2, 4, false).unwrap();
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![5.0, 6.0];
        let mut out = vec![0.0; 2];
        MultiHeadAttentionRef::forward(&cfg, &q, &k, &v, &mut out, 1, 1);
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_mha_two_heads() {
        let cfg = AttentionConfig::new(2, 2, 4, false).unwrap();
        let seq_len = 1;
        let kv_len = 1;
        // Q: [seq_len=1, num_heads=2, head_dim=2] = [1,0, 0,1]
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        MultiHeadAttentionRef::forward(&cfg, &q, &k, &v, &mut out, seq_len, kv_len);
        // Head 0: Q=[1,0], K=[1,0], V=[1,2] → output=[1,2]
        // Head 1: Q=[0,1], K=[0,1], V=[3,4] → output=[3,4]
        assert_slices_close(&out, &[1.0, 2.0, 3.0, 4.0], ATOL);
    }

    #[test]
    fn test_mha_heads_independent() {
        // Changing one head's Q should not affect the other head's output.
        let cfg = AttentionConfig::new(2, 2, 4, false).unwrap();
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        let q1 = vec![1.0, 0.0, 0.0, 1.0];
        let q2 = vec![99.0, 99.0, 0.0, 1.0]; // head 0 changed
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];
        MultiHeadAttentionRef::forward(&cfg, &q1, &k, &v, &mut out1, 1, 1);
        MultiHeadAttentionRef::forward(&cfg, &q2, &k, &v, &mut out2, 1, 1);
        // Head 1 output should be identical.
        assert_slices_close(&out1[2..4], &out2[2..4], ATOL);
    }

    #[test]
    fn test_mha_causal() {
        let cfg = AttentionConfig::new(1, 4, 8, true).unwrap();
        let seq_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; seq_len * head_dim];
        MultiHeadAttentionRef::forward(&cfg, &q, &k, &v, &mut out, seq_len, kv_len);
        // Token 0 with causal: only attends to token 0 → output ≈ V[0].
        assert_slices_close(&out[0..head_dim], &v[0..head_dim], ATOL);
    }

    #[test]
    fn test_mha_multi_token_multi_head() {
        let cfg = AttentionConfig::new(2, 2, 4, false).unwrap();
        let seq_len = 2;
        let kv_len = 2;
        // 2 heads, head_dim=2, seq=2, kv=2
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; seq_len * 2 * 2];
        MultiHeadAttentionRef::forward(&cfg, &q, &k, &v, &mut out, seq_len, kv_len);
        for val in &out {
            assert!(!val.is_nan());
        }
    }

    // ---- GroupedQueryAttention -------------------------------------------

    #[test]
    fn test_gqa_4_heads_2_kv() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let seq_len = 1;
        let kv_len = 1;
        // Q: [1, num_heads=4, head_dim=2]
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        // K: [1, num_kv_heads=2, head_dim=2]
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; num_heads * head_dim];
        GroupedQueryAttention::forward(
            num_heads,
            num_kv_heads,
            head_dim,
            &q,
            &k,
            &v,
            &mut out,
            seq_len,
            kv_len,
            false,
        )
        .unwrap();
        // Heads 0,1 share KV head 0; heads 2,3 share KV head 1.
        // Head 0: Q=[1,0], K=[1,0], V=[1,2] → [1,2]
        // Head 1: Q=[0,1], K=[1,0], V=[1,2] → [1,2] (single KV → softmax=1)
        // Head 2: Q=[1,0], K=[0,1], V=[3,4] → [3,4]
        // Head 3: Q=[0,1], K=[0,1], V=[3,4] → [3,4]
        assert_slices_close(&out[0..2], &[1.0, 2.0], ATOL);
        assert_slices_close(&out[2..4], &[1.0, 2.0], ATOL);
        assert_slices_close(&out[4..6], &[3.0, 4.0], ATOL);
        assert_slices_close(&out[6..8], &[3.0, 4.0], ATOL);
    }

    #[test]
    fn test_gqa_same_as_mha_when_equal_heads() {
        let num_heads = 2;
        let head_dim = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out_mha = vec![0.0; 4];
        let mut out_gqa = vec![0.0; 4];

        let cfg = AttentionConfig::new(num_heads, head_dim, 4, false).unwrap();
        MultiHeadAttentionRef::forward(&cfg, &q, &k, &v, &mut out_mha, 1, 1);
        GroupedQueryAttention::forward(
            num_heads,
            num_heads,
            head_dim,
            &q,
            &k,
            &v,
            &mut out_gqa,
            1,
            1,
            false,
        )
        .unwrap();
        assert_slices_close(&out_mha, &out_gqa, ATOL);
    }

    #[test]
    fn test_gqa_indivisible_error() {
        let mut out = vec![0.0; 6];
        assert!(
            GroupedQueryAttention::forward(
                3, 2, 2, &[0.0; 6], &[0.0; 4], &[0.0; 4], &mut out, 1, 1, false,
            )
            .is_err()
        );
    }

    #[test]
    fn test_gqa_single_kv_head() {
        // Multi-query attention: all heads share 1 KV head.
        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![5.0, 6.0];
        let mut out = vec![0.0; 8];
        GroupedQueryAttention::forward(
            num_heads,
            num_kv_heads,
            head_dim,
            &q,
            &k,
            &v,
            &mut out,
            1,
            1,
            false,
        )
        .unwrap();
        // Single KV → all heads get V regardless of Q.
        for h in 0..num_heads {
            assert_slices_close(&out[h * head_dim..(h + 1) * head_dim], &[5.0, 6.0], ATOL);
        }
    }

    // ---- OpenCL kernel source -------------------------------------------

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!ATTENTION_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_kernel() {
        assert!(ATTENTION_CL.contains("__kernel void scaled_dot_product_attention"));
    }

    #[test]
    fn test_opencl_source_contains_mha_kernel() {
        assert!(ATTENTION_CL.contains("__kernel void multi_head_attention"));
    }

    #[test]
    fn test_opencl_source_contains_softmax() {
        assert!(ATTENTION_CL.contains("exp("));
    }

    #[test]
    fn test_opencl_source_contains_causal_guard() {
        assert!(ATTENTION_CL.contains("causal"));
    }

    #[test]
    fn test_opencl_source_contains_get_global_id() {
        assert!(ATTENTION_CL.contains("get_global_id"));
    }

    // ---- Edge cases & regression ----------------------------------------

    #[test]
    fn test_sdpa_seq_longer_than_kv() {
        // seq_len > kv_len (without causal) — all queries attend to all keys.
        let head_dim = 2;
        let seq_len = 4;
        let kv_len = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; seq_len * head_dim];
        scaled_dot_product_attention_ref(
            &q, &k, &v, &mut out, seq_len, kv_len, head_dim, 1.0, false,
        );
        for val in &out {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_sdpa_output_sums_bounded() {
        // Since softmax weights sum to 1 and V values are bounded, output
        // should be bounded by max(|V|).
        let head_dim = 2;
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut out = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 3, head_dim, 1.0, false);
        for val in &out {
            assert!(*val >= 2.0 && *val <= 7.0, "output {val} out of V range");
        }
    }

    #[test]
    fn test_sdpa_deterministic() {
        // Same input → same output.
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut out1 = vec![0.0; 4];
        let mut out2 = vec![0.0; 4];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out1, 1, 2, 4, 0.5, false);
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out2, 1, 2, 4, 0.5, false);
        assert_slices_close(&out1, &out2, ATOL);
    }

    #[test]
    fn test_sdpa_non_causal_vs_causal_seq1() {
        // With seq_len=1 and kv_len=1, causal and non-causal should be identical.
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let mut out_c = vec![0.0; 2];
        let mut out_nc = vec![0.0; 2];
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out_c, 1, 1, 2, 1.0, true);
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out_nc, 1, 1, 2, 1.0, false);
        assert_slices_close(&out_c, &out_nc, ATOL);
    }

    #[test]
    fn test_mask_causal_diagonal() {
        // The diagonal should always be allowed.
        for n in 1..=8 {
            let mask = AttentionMask::causal(n, n, 0);
            for i in 0..n {
                assert!(mask.allows(i, i), "diagonal position ({i},{i}) not allowed for n={n}");
            }
        }
    }

    #[test]
    fn test_sdpa_large_head_dim() {
        let head_dim = 128;
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let mut out = vec![0.0; head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        scaled_dot_product_attention_ref(&q, &k, &v, &mut out, 1, 1, head_dim, scale, false);
        // Single KV position → output = V.
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_kv_cache_with_attention_consistency() {
        // Full sequence attention should match incremental attention through KV cache.
        let head_dim = 2;
        let full_q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let full_k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let full_v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Full 3-token attention (non-causal, last token).
        let mut full_out = vec![0.0; 6];
        scaled_dot_product_attention_ref(
            &full_q,
            &full_k,
            &full_v,
            &mut full_out,
            3,
            3,
            head_dim,
            1.0,
            false,
        );
        let full_last = &full_out[4..6];

        // KV cache approach: cache all 3 K/V, query with last token only.
        let mut cache = KVCacheEntry::new(head_dim, 8);
        cache.append(&full_k, &full_v).unwrap();
        let last_q = &full_q[4..6];
        let mut cache_out = vec![0.0; 2];
        scaled_dot_product_attention_ref(
            last_q,
            cache.keys(),
            cache.values(),
            &mut cache_out,
            1,
            cache.current_len,
            head_dim,
            1.0,
            false,
        );
        assert_slices_close(full_last, &cache_out, ATOL);
    }
}
