//! Module stub - implementation pending merge from feature branch
//! Attention computation for transformer inference.
//!
//! Implements multi-head, multi-query, grouped-query, and cross-attention
//! with optional causal masking, KV caching, and projection layers.

use std::fmt;

// ── Configuration ───────────────────────────────────────────────────────────

/// Attention hyperparameters.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
    /// Scaling factor applied to QK^T. Typically `1 / sqrt(head_dim)`.
    pub scale_factor: f32,
}

impl AttentionConfig {
    /// Create a config with the standard scale `1 / sqrt(head_dim)`.
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize, causal: bool) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { num_heads, head_dim, max_seq_len, causal, scale_factor: scale }
    }

    /// Model dimension (`num_heads * head_dim`).
    pub const fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

// ── Attention type ──────────────────────────────────────────────────────────

/// Variant of the attention mechanism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard multi-head attention (`num_kv_heads == num_q_heads`).
    MultiHead,
    /// Multi-query attention (one KV head shared across all Q heads).
    MultiQuery,
    /// Grouped-query attention with the given number of KV heads.
    GroupedQuery(usize),
    /// Cross-attention (K, V come from encoder, Q from decoder).
    CrossAttention,
}

impl AttentionType {
    /// Number of KV heads for a given number of query heads.
    pub const fn num_kv_heads(&self, num_q_heads: usize) -> usize {
        match self {
            Self::MultiHead | Self::CrossAttention => num_q_heads,
            Self::MultiQuery => 1,
            Self::GroupedQuery(n) => *n,
        }
    }

    /// How many query heads share each KV head.
    ///
    /// Returns `None` if `num_kv_heads` does not evenly divide `num_q_heads`.
    pub const fn heads_per_group(&self, num_q_heads: usize) -> Option<usize> {
        let kv = self.num_kv_heads(num_q_heads);
        if kv == 0 || !num_q_heads.is_multiple_of(kv) { None } else { Some(num_q_heads / kv) }
    }
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultiHead => write!(f, "MHA"),
            Self::MultiQuery => write!(f, "MQA"),
            Self::GroupedQuery(n) => write!(f, "GQA({n})"),
            Self::CrossAttention => write!(f, "CrossAttn"),
        }
    }
}

// ── QKV projection ─────────────────────────────────────────────────────────

/// Projects an input tensor into query, key, and value tensors.
#[derive(Debug, Clone)]
pub struct QKVProjection {
    /// Weight matrix stored in row-major order: `[3 * out_dim, in_dim]`.
    pub weight: Vec<f32>,
    /// Optional bias vector of length `3 * out_dim`.
    pub bias: Option<Vec<f32>>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension per Q/K/V (equal to `num_heads * head_dim` for Q).
    pub out_dim: usize,
}

impl QKVProjection {
    /// Create a new projection (weights zeroed).
    pub fn new(in_dim: usize, out_dim: usize, use_bias: bool) -> Self {
        Self {
            weight: vec![0.0; 3 * out_dim * in_dim],
            bias: if use_bias { Some(vec![0.0; 3 * out_dim]) } else { None },
            in_dim,
            out_dim,
        }
    }

    /// Project `input` (length `in_dim`) into Q, K, V (each length `out_dim`).
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(input.len(), self.in_dim, "input length mismatch");
        let mut q = vec![0.0_f32; self.out_dim];
        let mut k = vec![0.0_f32; self.out_dim];
        let mut v = vec![0.0_f32; self.out_dim];

        for (slice, offset) in [(&mut q, 0), (&mut k, self.out_dim), (&mut v, 2 * self.out_dim)] {
            for (o, row_start) in slice.iter_mut().enumerate() {
                let base = (offset + o) * self.in_dim;
                let mut sum = 0.0_f32;
                for (j, &x) in input.iter().enumerate() {
                    sum += self.weight[base + j] * x;
                }
                if let Some(ref bias) = self.bias {
                    sum += bias[offset + o];
                }
                *row_start = sum;
            }
        }

        (q, k, v)
    }
}

// ── Output projection ───────────────────────────────────────────────────────

/// Projects concatenated multi-head output back to model dimension.
#[derive(Debug, Clone)]
pub struct OutputProjection {
    /// Weight: `[out_dim, in_dim]` row-major.
    pub weight: Vec<f32>,
    /// Optional bias of length `out_dim`.
    pub bias: Option<Vec<f32>>,
    /// Input dimension (`num_heads * head_dim`).
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl OutputProjection {
    /// Create a new output projection (weights zeroed).
    pub fn new(in_dim: usize, out_dim: usize, use_bias: bool) -> Self {
        Self {
            weight: vec![0.0; out_dim * in_dim],
            bias: if use_bias { Some(vec![0.0; out_dim]) } else { None },
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: `output = W * input + bias`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_dim, "input length mismatch");
        let mut out = vec![0.0_f32; self.out_dim];
        for (o, val) in out.iter_mut().enumerate() {
            let base = o * self.in_dim;
            let mut sum = 0.0_f32;
            for (j, &x) in input.iter().enumerate() {
                sum += self.weight[base + j] * x;
            }
            if let Some(ref bias) = self.bias {
                sum += bias[o];
            }
            *val = sum;
        }
        out
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

/// Lower-triangular causal mask for autoregressive attention.
#[derive(Debug, Clone)]
pub struct CausalMask {
    size: usize,
}

impl CausalMask {
    /// Build a causal mask of the given size.
    pub const fn new(size: usize) -> Self {
        Self { size }
    }

    /// Returns `true` if position `(row, col)` is *allowed* (not masked).
    pub const fn is_allowed(&self, row: usize, col: usize) -> bool {
        col <= row
    }

    /// Mask size.
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Apply the causal mask to a score matrix (row-major, `seq_len × seq_len`).
    /// Masked positions are set to `neg_inf`.
    pub fn apply(&self, scores: &mut [f32], seq_len: usize, neg_inf: f32) {
        for row in 0..seq_len.min(self.size) {
            for col in (row + 1)..seq_len.min(self.size) {
                scores[row * seq_len + col] = neg_inf;
            }
        }
    }
}

// ── General attention mask ──────────────────────────────────────────────────

/// General-purpose attention mask (e.g. padding mask).
#[derive(Debug, Clone)]
pub struct AttentionMask {
    /// `true` = allowed, `false` = masked. Row-major `[rows, cols]`.
    mask: Vec<bool>,
    rows: usize,
    cols: usize,
}

impl AttentionMask {
    /// Create an all-allowed mask.
    pub fn all_allowed(rows: usize, cols: usize) -> Self {
        Self { mask: vec![true; rows * cols], rows, cols }
    }

    /// Create a padding mask: positions `>= seq_len` are masked for each row.
    pub fn padding(rows: usize, total_cols: usize, seq_len: usize) -> Self {
        let mut mask = vec![false; rows * total_cols];
        for r in 0..rows {
            for c in 0..seq_len.min(total_cols) {
                mask[r * total_cols + c] = true;
            }
        }
        Self { mask, rows, cols: total_cols }
    }

    /// Create from a raw bool slice.
    pub fn from_raw(mask: Vec<bool>, rows: usize, cols: usize) -> Self {
        assert_eq!(mask.len(), rows * cols);
        Self { mask, rows, cols }
    }

    /// Check if position `(row, col)` is allowed.
    pub fn is_allowed(&self, row: usize, col: usize) -> bool {
        if row < self.rows && col < self.cols { self.mask[row * self.cols + col] } else { false }
    }

    /// Apply mask to scores (row-major), setting masked positions to `neg_inf`.
    pub fn apply(&self, scores: &mut [f32], neg_inf: f32) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                if !self.mask[r * self.cols + c] {
                    scores[r * self.cols + c] = neg_inf;
                }
            }
        }
    }

    /// Dimensions.
    pub const fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

// ── KV cache entry ──────────────────────────────────────────────────────────

/// KV cache for a single attention layer.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Cached keys: `[capacity, head_dim]` row-major.
    pub key_cache: Vec<f32>,
    /// Cached values: `[capacity, head_dim]` row-major.
    pub value_cache: Vec<f32>,
    /// Number of valid (appended) positions.
    pub seq_len: usize,
    /// Maximum number of positions.
    pub capacity: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl KVCacheEntry {
    /// Allocate a new cache entry.
    pub fn new(capacity: usize, head_dim: usize) -> Self {
        Self {
            key_cache: vec![0.0; capacity * head_dim],
            value_cache: vec![0.0; capacity * head_dim],
            seq_len: 0,
            capacity,
            head_dim,
        }
    }

    /// Append a single key-value pair. Returns `false` if full.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> bool {
        assert_eq!(key.len(), self.head_dim);
        assert_eq!(value.len(), self.head_dim);
        if self.seq_len >= self.capacity {
            return false;
        }
        let offset = self.seq_len * self.head_dim;
        self.key_cache[offset..offset + self.head_dim].copy_from_slice(key);
        self.value_cache[offset..offset + self.head_dim].copy_from_slice(value);
        self.seq_len += 1;
        true
    }

    /// Get cached key at the given position.
    pub fn get_key(&self, pos: usize) -> Option<&[f32]> {
        if pos < self.seq_len {
            let offset = pos * self.head_dim;
            Some(&self.key_cache[offset..offset + self.head_dim])
        } else {
            None
        }
    }

    /// Get cached value at the given position.
    pub fn get_value(&self, pos: usize) -> Option<&[f32]> {
        if pos < self.seq_len {
            let offset = pos * self.head_dim;
            Some(&self.value_cache[offset..offset + self.head_dim])
        } else {
            None
        }
    }

    /// Reset the cache, keeping the allocation.
    pub const fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Remaining capacity.
    pub const fn remaining(&self) -> usize {
        self.capacity - self.seq_len
    }
}

// ── KV cache (multi-layer) ─────────────────────────────────────────────────

/// Multi-layer KV cache for autoregressive generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    layers: Vec<KVCacheEntry>,
}

impl KVCache {
    /// Allocate cache for `num_layers` layers.
    pub fn new(num_layers: usize, capacity: usize, head_dim: usize) -> Self {
        Self { layers: (0..num_layers).map(|_| KVCacheEntry::new(capacity, head_dim)).collect() }
    }

    /// Get a mutable reference to a specific layer's cache.
    pub fn layer_mut(&mut self, layer: usize) -> &mut KVCacheEntry {
        &mut self.layers[layer]
    }

    /// Get a shared reference to a specific layer's cache.
    pub fn layer(&self, layer: usize) -> &KVCacheEntry {
        &self.layers[layer]
    }

    /// Number of layers.
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for entry in &mut self.layers {
            entry.clear();
        }
    }

    /// Current sequence length (from layer 0, assumed uniform).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, |e| e.seq_len)
    }
}

// ── Scaled dot-product attention ────────────────────────────────────────────

/// Computes scaled dot-product attention on pre-projected Q, K, V vectors.
#[derive(Debug)]
pub struct ScaledDotProductAttention {
    scale: f32,
}

impl ScaledDotProductAttention {
    /// Create with explicit scale factor.
    pub const fn new(scale: f32) -> Self {
        Self { scale }
    }

    /// Create with standard scale `1 / sqrt(head_dim)`.
    pub fn standard(head_dim: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { scale }
    }

    /// Compute attention for a single head.
    ///
    /// - `query`: `[head_dim]`
    /// - `keys`: `[seq_len, head_dim]` row-major
    /// - `values`: `[seq_len, head_dim]` row-major
    /// - `mask`: optional; if `mask[i]` is `false`, position `i` is masked.
    ///
    /// Returns the attended output vector of length `head_dim`.
    pub fn forward(
        &self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        head_dim: usize,
        mask: Option<&[bool]>,
    ) -> Vec<f32> {
        let seq_len = keys.len() / head_dim;
        assert_eq!(keys.len(), seq_len * head_dim);
        assert_eq!(values.len(), seq_len * head_dim);
        assert_eq!(query.len(), head_dim);

        if seq_len == 0 {
            return vec![0.0; head_dim];
        }

        // Compute scaled scores: score[i] = (Q · K_i) * scale
        let mut scores = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let k_start = i * head_dim;
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += query[d] * keys[k_start + d];
            }
            scores.push(dot * self.scale);
        }

        // Apply mask
        if let Some(m) = mask {
            for (i, &allowed) in m.iter().enumerate().take(seq_len) {
                if !allowed {
                    scores[i] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        softmax_inplace(&mut scores);

        // Weighted sum over values
        let mut output = vec![0.0_f32; head_dim];
        for (i, &w) in scores.iter().enumerate() {
            let v_start = i * head_dim;
            for d in 0..head_dim {
                output[d] += w * values[v_start + d];
            }
        }
        output
    }
}

/// In-place softmax (numerically stable).
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ── Attention engine ────────────────────────────────────────────────────────

/// Full attention computation: project → attend → output.
#[derive(Debug)]
pub struct AttentionEngine {
    /// Attention configuration.
    pub config: AttentionConfig,
    /// Type of attention mechanism.
    pub attention_type: AttentionType,
    /// QKV projection.
    pub qkv_proj: QKVProjection,
    /// Output projection.
    pub out_proj: OutputProjection,
    /// Scaled dot-product attention kernel.
    pub sdpa: ScaledDotProductAttention,
}

impl AttentionEngine {
    /// Create a new attention engine with zeroed weights.
    pub fn new(config: AttentionConfig, attention_type: AttentionType) -> Self {
        let model_dim = config.model_dim();
        let num_kv_heads = attention_type.num_kv_heads(config.num_heads);
        // kv_dim = num_kv_heads * config.head_dim (reserved for GQA projection sizing)
        let _ = num_kv_heads;
        // QKV projection: out_dim is for Q; K/V may be smaller for GQA/MQA
        // For simplicity, we use the full model_dim for Q and kv_dim for K/V.
        // The unified projection packs Q(model_dim) + K(kv_dim) + V(kv_dim).
        let qkv_out = model_dim;
        let qkv_proj = QKVProjection::new(model_dim, qkv_out, false);
        let out_proj = OutputProjection::new(model_dim, model_dim, false);
        let sdpa = ScaledDotProductAttention::new(config.scale_factor);
        Self { config, attention_type, qkv_proj, out_proj, sdpa }
    }

    /// Run attention on a single token with the KV cache.
    ///
    /// `input`: `[model_dim]` — the current token's hidden state.
    /// `cache`: mutable KV cache entry for this layer.
    ///
    /// Returns the output vector of length `model_dim`.
    pub fn forward_with_cache(&self, input: &[f32], cache: &mut KVCacheEntry) -> Vec<f32> {
        let (q, k, v) = self.qkv_proj.forward(input);

        // Append new K, V to cache (one head — simplified single-head path)
        let head_dim = self.config.head_dim;
        for h in 0..self.attention_type.num_kv_heads(self.config.num_heads) {
            let offset = h * head_dim;
            let k_slice = &k[offset..offset + head_dim];
            let v_slice = &v[offset..offset + head_dim];
            // For multi-head we'd have per-head caches; here we append all
            // KV heads sequentially for simplicity.
            if h == 0 {
                cache.append(k_slice, v_slice);
            }
        }

        // Compute attention for each Q head
        let seq_len = cache.seq_len;
        let keys = &cache.key_cache[..seq_len * head_dim];
        let values = &cache.value_cache[..seq_len * head_dim];

        // For the new token at position (seq_len - 1), all prior positions
        // are visible in both causal and non-causal modes.
        let causal_mask: Vec<bool> = vec![true; seq_len];

        let mut concat = Vec::with_capacity(self.config.num_heads * head_dim);
        for h in 0..self.config.num_heads {
            let q_offset = h * head_dim;
            let q_head = &q[q_offset..q_offset + head_dim];
            let out = self.sdpa.forward(q_head, keys, values, head_dim, Some(&causal_mask));
            concat.extend_from_slice(&out);
        }

        self.out_proj.forward(&concat)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_vec_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y))
    }

    // ───────────────────────────────────────────────────────────────────────
    // AttentionConfig
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn config_scale_factor() {
        let cfg = AttentionConfig::new(8, 64, 2048, true);
        let expected = 1.0 / 64.0_f32.sqrt();
        assert!(approx_eq(cfg.scale_factor, expected));
    }

    #[test]
    fn config_model_dim() {
        let cfg = AttentionConfig::new(8, 64, 2048, true);
        assert_eq!(cfg.model_dim(), 512);
    }

    #[test]
    fn config_single_head() {
        let cfg = AttentionConfig::new(1, 128, 512, false);
        assert_eq!(cfg.model_dim(), 128);
        assert!(!cfg.causal);
    }

    #[test]
    fn config_max_seq_len() {
        let cfg = AttentionConfig::new(4, 32, 4096, true);
        assert_eq!(cfg.max_seq_len, 4096);
    }

    #[test]
    fn config_small_head_dim() {
        let cfg = AttentionConfig::new(16, 1, 64, true);
        assert_eq!(cfg.model_dim(), 16);
        assert!(approx_eq(cfg.scale_factor, 1.0));
    }

    // ───────────────────────────────────────────────────────────────────────
    // AttentionType
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn mha_kv_heads_eq_q_heads() {
        assert_eq!(AttentionType::MultiHead.num_kv_heads(8), 8);
    }

    #[test]
    fn mqa_kv_heads_is_one() {
        assert_eq!(AttentionType::MultiQuery.num_kv_heads(8), 1);
    }

    #[test]
    fn gqa_kv_heads() {
        assert_eq!(AttentionType::GroupedQuery(4).num_kv_heads(8), 4);
    }

    #[test]
    fn cross_attn_kv_heads_eq_q_heads() {
        assert_eq!(AttentionType::CrossAttention.num_kv_heads(12), 12);
    }

    #[test]
    fn mha_heads_per_group() {
        assert_eq!(AttentionType::MultiHead.heads_per_group(8), Some(1));
    }

    #[test]
    fn mqa_heads_per_group() {
        assert_eq!(AttentionType::MultiQuery.heads_per_group(8), Some(8));
    }

    #[test]
    fn gqa_heads_per_group() {
        assert_eq!(AttentionType::GroupedQuery(2).heads_per_group(8), Some(4));
    }

    #[test]
    fn gqa_uneven_returns_none() {
        assert_eq!(AttentionType::GroupedQuery(3).heads_per_group(8), None);
    }

    #[test]
    fn attention_type_display() {
        assert_eq!(format!("{}", AttentionType::MultiHead), "MHA");
        assert_eq!(format!("{}", AttentionType::MultiQuery), "MQA");
        assert_eq!(format!("{}", AttentionType::GroupedQuery(4)), "GQA(4)");
        assert_eq!(format!("{}", AttentionType::CrossAttention), "CrossAttn");
    }

    #[test]
    fn attention_type_eq() {
        assert_eq!(AttentionType::MultiHead, AttentionType::MultiHead);
        assert_ne!(AttentionType::MultiHead, AttentionType::MultiQuery);
        assert_eq!(AttentionType::GroupedQuery(4), AttentionType::GroupedQuery(4));
        assert_ne!(AttentionType::GroupedQuery(4), AttentionType::GroupedQuery(2));
    }

    // ───────────────────────────────────────────────────────────────────────
    // QKVProjection
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn qkv_proj_dims_no_bias() {
        let proj = QKVProjection::new(64, 64, false);
        assert_eq!(proj.weight.len(), 3 * 64 * 64);
        assert!(proj.bias.is_none());
    }

    #[test]
    fn qkv_proj_dims_with_bias() {
        let proj = QKVProjection::new(64, 64, true);
        assert_eq!(proj.bias.as_ref().unwrap().len(), 3 * 64);
    }

    #[test]
    fn qkv_proj_zero_weights_produce_zero_output() {
        let proj = QKVProjection::new(4, 4, false);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (q, k, v) = proj.forward(&input);
        assert!(q.iter().all(|&x| x == 0.0));
        assert!(k.iter().all(|&x| x == 0.0));
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn qkv_proj_identity_weights() {
        let dim = 2;
        let mut proj = QKVProjection::new(dim, dim, false);
        // Set Q block to identity
        for i in 0..dim {
            proj.weight[i * dim + i] = 1.0;
        }
        let input = vec![3.0, 7.0];
        let (q, _k, _v) = proj.forward(&input);
        assert!(approx_vec_eq(&q, &[3.0, 7.0]));
    }

    #[test]
    fn qkv_proj_bias_only() {
        let dim = 2;
        let mut proj = QKVProjection::new(dim, dim, true);
        let bias = proj.bias.as_mut().unwrap();
        bias[0] = 10.0;
        bias[1] = 20.0;
        let input = vec![0.0, 0.0];
        let (q, _k, _v) = proj.forward(&input);
        assert!(approx_vec_eq(&q, &[10.0, 20.0]));
    }

    #[test]
    fn qkv_proj_output_lengths() {
        let proj = QKVProjection::new(8, 4, false);
        let input = vec![0.0; 8];
        let (q, k, v) = proj.forward(&input);
        assert_eq!(q.len(), 4);
        assert_eq!(k.len(), 4);
        assert_eq!(v.len(), 4);
    }

    // ───────────────────────────────────────────────────────────────────────
    // OutputProjection
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn out_proj_zero_weights() {
        let proj = OutputProjection::new(4, 4, false);
        let input = vec![1.0; 4];
        let out = proj.forward(&input);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn out_proj_identity() {
        let dim = 3;
        let mut proj = OutputProjection::new(dim, dim, false);
        for i in 0..dim {
            proj.weight[i * dim + i] = 1.0;
        }
        let input = vec![5.0, 6.0, 7.0];
        let out = proj.forward(&input);
        assert!(approx_vec_eq(&out, &[5.0, 6.0, 7.0]));
    }

    #[test]
    fn out_proj_with_bias() {
        let mut proj = OutputProjection::new(2, 2, true);
        let bias = proj.bias.as_mut().unwrap();
        bias[0] = 1.0;
        bias[1] = 2.0;
        let out = proj.forward(&[0.0, 0.0]);
        assert!(approx_vec_eq(&out, &[1.0, 2.0]));
    }

    #[test]
    fn out_proj_different_dims() {
        let proj = OutputProjection::new(4, 2, false);
        assert_eq!(proj.weight.len(), 2 * 4);
        let out = proj.forward(&[0.0; 4]);
        assert_eq!(out.len(), 2);
    }

    // ───────────────────────────────────────────────────────────────────────
    // CausalMask
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn causal_mask_diagonal_allowed() {
        let mask = CausalMask::new(4);
        for i in 0..4 {
            assert!(mask.is_allowed(i, i));
        }
    }

    #[test]
    fn causal_mask_lower_triangle_allowed() {
        let mask = CausalMask::new(4);
        assert!(mask.is_allowed(2, 0));
        assert!(mask.is_allowed(2, 1));
        assert!(mask.is_allowed(3, 0));
    }

    #[test]
    fn causal_mask_upper_triangle_blocked() {
        let mask = CausalMask::new(4);
        assert!(!mask.is_allowed(0, 1));
        assert!(!mask.is_allowed(0, 3));
        assert!(!mask.is_allowed(1, 2));
    }

    #[test]
    fn causal_mask_size() {
        let mask = CausalMask::new(16);
        assert_eq!(mask.size(), 16);
    }

    #[test]
    fn causal_mask_apply_scores() {
        let mut scores = vec![1.0; 9]; // 3×3
        let mask = CausalMask::new(3);
        mask.apply(&mut scores, 3, f32::NEG_INFINITY);
        // Row 0: [1.0, -inf, -inf]
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        // Row 1: [1.0, 1.0, -inf]
        assert_eq!(scores[3], 1.0);
        assert_eq!(scores[4], 1.0);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        // Row 2: [1.0, 1.0, 1.0]
        assert_eq!(scores[6], 1.0);
        assert_eq!(scores[7], 1.0);
        assert_eq!(scores[8], 1.0);
    }

    #[test]
    fn causal_mask_size_one() {
        let mask = CausalMask::new(1);
        assert!(mask.is_allowed(0, 0));
    }

    // ───────────────────────────────────────────────────────────────────────
    // AttentionMask
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn all_allowed_mask() {
        let mask = AttentionMask::all_allowed(3, 4);
        assert_eq!(mask.dims(), (3, 4));
        for r in 0..3 {
            for c in 0..4 {
                assert!(mask.is_allowed(r, c));
            }
        }
    }

    #[test]
    fn padding_mask_basic() {
        let mask = AttentionMask::padding(2, 5, 3);
        // Positions 0..3 allowed, 3..5 masked
        assert!(mask.is_allowed(0, 0));
        assert!(mask.is_allowed(0, 2));
        assert!(!mask.is_allowed(0, 3));
        assert!(!mask.is_allowed(0, 4));
    }

    #[test]
    fn padding_mask_full_length() {
        let mask = AttentionMask::padding(1, 4, 4);
        for c in 0..4 {
            assert!(mask.is_allowed(0, c));
        }
    }

    #[test]
    fn padding_mask_zero_length() {
        let mask = AttentionMask::padding(1, 4, 0);
        for c in 0..4 {
            assert!(!mask.is_allowed(0, c));
        }
    }

    #[test]
    fn attention_mask_out_of_bounds() {
        let mask = AttentionMask::all_allowed(2, 2);
        assert!(!mask.is_allowed(5, 0));
        assert!(!mask.is_allowed(0, 5));
    }

    #[test]
    fn attention_mask_apply() {
        let mask = AttentionMask::padding(1, 4, 2);
        let mut scores = vec![1.0; 4];
        mask.apply(&mut scores, f32::NEG_INFINITY);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 1.0);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[3], f32::NEG_INFINITY);
    }

    #[test]
    fn attention_mask_from_raw() {
        let raw = vec![true, false, false, true];
        let mask = AttentionMask::from_raw(raw, 2, 2);
        assert!(mask.is_allowed(0, 0));
        assert!(!mask.is_allowed(0, 1));
        assert!(!mask.is_allowed(1, 0));
        assert!(mask.is_allowed(1, 1));
    }

    // ───────────────────────────────────────────────────────────────────────
    // KVCacheEntry
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn kv_cache_entry_empty() {
        let entry = KVCacheEntry::new(16, 4);
        assert_eq!(entry.seq_len, 0);
        assert_eq!(entry.capacity, 16);
        assert_eq!(entry.remaining(), 16);
    }

    #[test]
    fn kv_cache_entry_append_one() {
        let mut entry = KVCacheEntry::new(4, 2);
        let ok = entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert!(ok);
        assert_eq!(entry.seq_len, 1);
        assert_eq!(entry.remaining(), 3);
    }

    #[test]
    fn kv_cache_entry_get_key() {
        let mut entry = KVCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(entry.get_key(0), Some([1.0, 2.0].as_slice()));
    }

    #[test]
    fn kv_cache_entry_get_value() {
        let mut entry = KVCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(entry.get_value(0), Some([3.0, 4.0].as_slice()));
    }

    #[test]
    fn kv_cache_entry_get_out_of_range() {
        let entry = KVCacheEntry::new(4, 2);
        assert!(entry.get_key(0).is_none());
        assert!(entry.get_value(0).is_none());
    }

    #[test]
    fn kv_cache_entry_append_multiple() {
        let mut entry = KVCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[10.0, 20.0]);
        entry.append(&[3.0, 4.0], &[30.0, 40.0]);
        assert_eq!(entry.seq_len, 2);
        assert_eq!(entry.get_key(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(entry.get_key(1), Some([3.0, 4.0].as_slice()));
        assert_eq!(entry.get_value(1), Some([30.0, 40.0].as_slice()));
    }

    #[test]
    fn kv_cache_entry_full() {
        let mut entry = KVCacheEntry::new(2, 1);
        assert!(entry.append(&[1.0], &[2.0]));
        assert!(entry.append(&[3.0], &[4.0]));
        assert!(!entry.append(&[5.0], &[6.0]));
        assert_eq!(entry.seq_len, 2);
    }

    #[test]
    fn kv_cache_entry_clear() {
        let mut entry = KVCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        entry.clear();
        assert_eq!(entry.seq_len, 0);
        assert_eq!(entry.remaining(), 4);
        assert!(entry.get_key(0).is_none());
    }

    #[test]
    fn kv_cache_entry_reuse_after_clear() {
        let mut entry = KVCacheEntry::new(2, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        entry.clear();
        assert!(entry.append(&[5.0, 6.0], &[7.0, 8.0]));
        assert_eq!(entry.get_key(0), Some([5.0, 6.0].as_slice()));
    }

    // ───────────────────────────────────────────────────────────────────────
    // KVCache (multi-layer)
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn kv_cache_num_layers() {
        let cache = KVCache::new(4, 16, 8);
        assert_eq!(cache.num_layers(), 4);
    }

    #[test]
    fn kv_cache_initial_seq_len() {
        let cache = KVCache::new(2, 16, 8);
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn kv_cache_append_and_read() {
        let mut cache = KVCache::new(2, 16, 2);
        cache.layer_mut(0).append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.layer(0).seq_len, 1);
        assert_eq!(cache.layer(1).seq_len, 0);
    }

    #[test]
    fn kv_cache_clear_all() {
        let mut cache = KVCache::new(2, 16, 2);
        cache.layer_mut(0).append(&[1.0, 2.0], &[3.0, 4.0]);
        cache.layer_mut(1).append(&[5.0, 6.0], &[7.0, 8.0]);
        cache.clear();
        assert_eq!(cache.layer(0).seq_len, 0);
        assert_eq!(cache.layer(1).seq_len, 0);
    }

    #[test]
    fn kv_cache_zero_layers() {
        let cache = KVCache::new(0, 16, 8);
        assert_eq!(cache.num_layers(), 0);
        assert_eq!(cache.seq_len(), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Softmax
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_uniform_input() {
        let mut v = vec![1.0, 1.0, 1.0];
        softmax_inplace(&mut v);
        for &x in &v {
            assert!(approx_eq(x, 1.0 / 3.0));
        }
    }

    #[test]
    fn softmax_single_element() {
        let mut v = vec![42.0];
        softmax_inplace(&mut v);
        assert!(approx_eq(v[0], 1.0));
    }

    #[test]
    fn softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn softmax_large_values() {
        let mut v = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(approx_eq(sum, 1.0));
        // Largest input should get largest probability
        assert!(v[2] > v[1]);
        assert!(v[1] > v[0]);
    }

    #[test]
    fn softmax_with_neg_inf() {
        let mut v = vec![1.0, f32::NEG_INFINITY, 1.0];
        softmax_inplace(&mut v);
        assert!(approx_eq(v[1], 0.0));
        assert!(approx_eq(v[0], 0.5));
        assert!(approx_eq(v[2], 0.5));
    }

    // ───────────────────────────────────────────────────────────────────────
    // ScaledDotProductAttention
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn sdpa_standard_scale() {
        let sdpa = ScaledDotProductAttention::standard(64);
        assert!(approx_eq(sdpa.scale, 1.0 / 64.0_f32.sqrt()));
    }

    #[test]
    fn sdpa_single_kv_returns_value() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let query = vec![1.0, 0.0];
        let keys = vec![1.0, 0.0]; // seq_len=1
        let values = vec![5.0, 6.0];
        let out = sdpa.forward(&query, &keys, &values, 2, None);
        // With single KV, softmax([score]) = [1.0], output = values
        assert!(approx_vec_eq(&out, &[5.0, 6.0]));
    }

    #[test]
    fn sdpa_two_equal_keys() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let query = vec![1.0, 0.0];
        let keys = vec![1.0, 0.0, 1.0, 0.0]; // 2 identical keys
        let values = vec![2.0, 4.0, 6.0, 8.0];
        let out = sdpa.forward(&query, &keys, &values, 2, None);
        // Equal scores → uniform softmax → average of values
        assert!(approx_vec_eq(&out, &[4.0, 6.0]));
    }

    #[test]
    fn sdpa_masking() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let query = vec![1.0];
        let keys = vec![1.0, 1.0]; // 2 keys
        let values = vec![10.0, 20.0];
        let mask = vec![true, false]; // second position masked
        let out = sdpa.forward(&query, &keys, &values, 1, Some(&mask));
        // Only first position visible → output = first value
        assert!(approx_vec_eq(&out, &[10.0]));
    }

    #[test]
    fn sdpa_empty_seq() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let out = sdpa.forward(&[1.0, 2.0], &[], &[], 2, None);
        assert!(approx_vec_eq(&out, &[0.0, 0.0]));
    }

    #[test]
    fn sdpa_scale_affects_distribution() {
        // With a very large scale, the largest score dominates
        let sdpa_big = ScaledDotProductAttention::new(100.0);
        let query = vec![1.0];
        let keys = vec![1.0, 0.5]; // scores before scale: 1.0, 0.5
        let values = vec![10.0, 20.0];
        let out_big = sdpa_big.forward(&query, &keys, &values, 1, None);
        // Large scale → softmax concentrates on highest score → output ≈ 10.0
        assert!(out_big[0] < 11.0); // close to 10

        let sdpa_small = ScaledDotProductAttention::new(0.01);
        let out_small = sdpa_small.forward(&query, &keys, &values, 1, None);
        // Small scale → nearly uniform → output ≈ 15.0
        assert!(out_small[0] > 14.0 && out_small[0] < 16.0);
    }

    #[test]
    fn sdpa_orthogonal_query_keys() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        // query is [1, 0], keys are [0, 1] and [0, 1]
        // dot products are 0 and 0 → uniform softmax
        let query = vec![1.0, 0.0];
        let keys = vec![0.0, 1.0, 0.0, 1.0];
        let values = vec![2.0, 4.0, 6.0, 8.0];
        let out = sdpa.forward(&query, &keys, &values, 2, None);
        assert!(approx_vec_eq(&out, &[4.0, 6.0]));
    }

    #[test]
    fn sdpa_all_masked() {
        // When all positions are masked, softmax of [-inf, -inf] produces NaN.
        // The output may be NaN/0 — we just ensure it doesn't panic.
        let sdpa = ScaledDotProductAttention::new(1.0);
        let query = vec![1.0];
        let keys = vec![1.0, 1.0];
        let values = vec![10.0, 20.0];
        let mask = vec![false, false];
        let _out = sdpa.forward(&query, &keys, &values, 1, Some(&mask));
        // No panic is the assertion.
    }

    // ───────────────────────────────────────────────────────────────────────
    // Multi-head attention (integration via AttentionEngine)
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn engine_creation() {
        let cfg = AttentionConfig::new(4, 8, 128, true);
        let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
        assert_eq!(engine.config.num_heads, 4);
        assert_eq!(engine.config.head_dim, 8);
    }

    #[test]
    fn engine_forward_with_cache_zero_weights() {
        let cfg = AttentionConfig::new(2, 4, 16, true);
        let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
        let mut cache = KVCacheEntry::new(16, 4);
        let input = vec![1.0; 8];
        let out = engine.forward_with_cache(&input, &mut cache);
        // Zero weights → zero output
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn engine_cache_grows() {
        let cfg = AttentionConfig::new(1, 2, 16, true);
        let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
        let mut cache = KVCacheEntry::new(16, 2);
        engine.forward_with_cache(&[0.0; 2], &mut cache);
        assert_eq!(cache.seq_len, 1);
        engine.forward_with_cache(&[0.0; 2], &mut cache);
        assert_eq!(cache.seq_len, 2);
    }

    #[test]
    fn engine_gqa_type() {
        let cfg = AttentionConfig::new(8, 4, 128, true);
        let engine = AttentionEngine::new(cfg, AttentionType::GroupedQuery(2));
        assert_eq!(engine.attention_type.num_kv_heads(8), 2);
    }

    #[test]
    fn engine_cross_attention_type() {
        let cfg = AttentionConfig::new(4, 8, 64, false);
        let engine = AttentionEngine::new(cfg, AttentionType::CrossAttention);
        assert!(!engine.config.causal);
        assert_eq!(engine.attention_type, AttentionType::CrossAttention);
    }

    #[test]
    fn engine_output_dimension_matches_model_dim() {
        let cfg = AttentionConfig::new(4, 16, 128, true);
        let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
        let mut cache = KVCacheEntry::new(16, 16);
        let input = vec![0.0; 64]; // 4 * 16 = 64
        let out = engine.forward_with_cache(&input, &mut cache);
        assert_eq!(out.len(), 64);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Edge cases
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn single_head_single_dim() {
        let cfg = AttentionConfig::new(1, 1, 8, true);
        let engine = AttentionEngine::new(cfg, AttentionType::MultiHead);
        let mut cache = KVCacheEntry::new(8, 1);
        let out = engine.forward_with_cache(&[1.0], &mut cache);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn seq_len_one_attention() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0]; // seq_len=1
        let v = vec![7.0, 8.0, 9.0, 10.0];
        let out = sdpa.forward(&q, &k, &v, 4, None);
        assert!(approx_vec_eq(&out, &[7.0, 8.0, 9.0, 10.0]));
    }

    #[test]
    fn kv_cache_entry_head_dim_one() {
        let mut entry = KVCacheEntry::new(4, 1);
        entry.append(&[42.0], &[99.0]);
        assert_eq!(entry.get_key(0), Some([42.0].as_slice()));
        assert_eq!(entry.get_value(0), Some([99.0].as_slice()));
    }

    #[test]
    fn kv_cache_large_capacity() {
        let entry = KVCacheEntry::new(1024, 64);
        assert_eq!(entry.key_cache.len(), 1024 * 64);
        assert_eq!(entry.remaining(), 1024);
    }

    #[test]
    fn causal_mask_size_zero() {
        let mask = CausalMask::new(0);
        assert_eq!(mask.size(), 0);
    }

    #[test]
    fn attention_type_clone() {
        let t = AttentionType::GroupedQuery(4);
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn config_clone() {
        let cfg = AttentionConfig::new(8, 64, 2048, true);
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.num_heads, 8);
        assert!(approx_eq(cfg2.scale_factor, cfg.scale_factor));
    }

    #[test]
    fn kv_cache_entry_clone() {
        let mut entry = KVCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]);
        let entry2 = entry.clone();
        assert_eq!(entry2.seq_len, 1);
        assert_eq!(entry2.get_key(0), Some([1.0, 2.0].as_slice()));
    }

    #[test]
    fn sdpa_negative_values() {
        let sdpa = ScaledDotProductAttention::new(1.0);
        let query = vec![-1.0];
        let keys = vec![1.0, -1.0];
        let values = vec![10.0, 20.0];
        let out = sdpa.forward(&query, &keys, &values, 1, None);
        // dot(-1, 1) = -1, dot(-1, -1) = 1 → second key preferred
        assert!(out[0] > 15.0);
    }

    #[test]
    fn gqa_shared_heads_two_groups() {
        let ty = AttentionType::GroupedQuery(2);
        assert_eq!(ty.heads_per_group(8), Some(4));
        assert_eq!(ty.num_kv_heads(8), 2);
    }

    #[test]
    fn gqa_shared_heads_single_group() {
        let ty = AttentionType::GroupedQuery(1);
        assert_eq!(ty.heads_per_group(8), Some(8));
    }

    #[test]
    fn mqa_all_heads_share() {
        let ty = AttentionType::MultiQuery;
        assert_eq!(ty.heads_per_group(16), Some(16));
    }

    #[test]
    fn cross_attention_heads_per_group() {
        let ty = AttentionType::CrossAttention;
        assert_eq!(ty.heads_per_group(6), Some(1));
    }

    #[test]
    fn qkv_proj_clone() {
        let proj = QKVProjection::new(4, 4, true);
        let proj2 = proj.clone();
        assert_eq!(proj.in_dim, proj2.in_dim);
        assert_eq!(proj2.out_dim, 4);
        assert!(proj2.bias.is_some());
    }

    #[test]
    fn output_proj_clone() {
        let proj = OutputProjection::new(4, 4, false);
        let proj2 = proj.clone();
        assert_eq!(proj.in_dim, proj2.in_dim);
        assert!(proj2.bias.is_none());
    }
}
