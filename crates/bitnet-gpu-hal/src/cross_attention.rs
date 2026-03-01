//! Module stub - implementation pending merge from feature branch
//! Cross-attention module for encoder-decoder architectures.
//!
//! Provides multi-head cross-attention with support for standard, multi-query,
//! grouped-query, linear, and sparse attention variants. Includes KV caching
//! for efficient autoregressive decoding and alignment extraction for
//! visualization/debugging.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration & enums
// ---------------------------------------------------------------------------

/// Configuration for a cross-attention layer.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossAttentionConfig {
    /// Dimension of the decoder hidden states (query source).
    pub query_dim: usize,
    /// Dimension of the encoder hidden states (key/value source).
    pub key_dim: usize,
    /// Dimension of the encoder hidden states used for values.
    pub value_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Dropout probability (0.0–1.0).
    pub dropout: f32,
}

impl CrossAttentionConfig {
    /// Create a new configuration, returning `None` if parameters are invalid.
    pub fn new(
        query_dim: usize,
        key_dim: usize,
        value_dim: usize,
        num_heads: usize,
        head_dim: usize,
        dropout: f32,
    ) -> Option<Self> {
        if num_heads == 0 || head_dim == 0 || query_dim == 0 || key_dim == 0 || value_dim == 0 {
            return None;
        }
        if !(0.0..=1.0).contains(&dropout) {
            return None;
        }
        Some(Self { query_dim, key_dim, value_dim, num_heads, head_dim, dropout })
    }

    /// Total projection dimension (`num_heads * head_dim`).
    pub const fn total_head_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// Attention variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossAttentionType {
    /// Standard multi-head attention (one K/V head per query head).
    Standard,
    /// Multi-query attention (single shared K/V head).
    MultiQuery,
    /// Grouped-query attention (fewer K/V heads than query heads).
    GroupedQuery { num_kv_heads: usize },
    /// Linear attention (kernel-based, no softmax).
    Linear,
    /// Sparse attention with a fixed block size.
    Sparse { block_size: usize },
}

impl fmt::Display for CrossAttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::MultiQuery => write!(f, "MultiQuery"),
            Self::GroupedQuery { num_kv_heads } => {
                write!(f, "GroupedQuery(kv_heads={num_kv_heads})")
            }
            Self::Linear => write!(f, "Linear"),
            Self::Sparse { block_size } => write!(f, "Sparse(block={block_size})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

/// Projects decoder hidden states into query vectors.
#[derive(Debug, Clone)]
pub struct QueryProjection {
    /// Weight matrix of shape `[query_dim, num_heads * head_dim]`.
    pub weights: Vec<f32>,
    /// Bias vector of length `num_heads * head_dim`.
    pub bias: Vec<f32>,
    pub query_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl QueryProjection {
    /// Create a new projection with the given weights and bias.
    pub fn new(
        weights: Vec<f32>,
        bias: Vec<f32>,
        query_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Option<Self> {
        let out_dim = num_heads * head_dim;
        if weights.len() != query_dim * out_dim || bias.len() != out_dim {
            return None;
        }
        Some(Self { weights, bias, query_dim, num_heads, head_dim })
    }

    /// Create a projection initialized to identity-like mapping (for testing).
    pub fn identity(query_dim: usize, num_heads: usize, head_dim: usize) -> Self {
        let out_dim = num_heads * head_dim;
        let mut weights = vec![0.0f32; query_dim * out_dim];
        let copy_dim = query_dim.min(out_dim);
        for i in 0..copy_dim {
            weights[i * out_dim + i] = 1.0;
        }
        Self { weights, bias: vec![0.0; out_dim], query_dim, num_heads, head_dim }
    }

    /// Project a batch of decoder hidden states.
    ///
    /// `input` shape: `[seq_len, query_dim]` (flattened row-major).
    /// Returns shape: `[seq_len, num_heads, head_dim]` (flattened).
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let out_dim = self.num_heads * self.head_dim;
        assert_eq!(input.len(), seq_len * self.query_dim, "input length mismatch");
        let mut output = vec![0.0f32; seq_len * out_dim];
        for t in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = self.bias[j];
                for k in 0..self.query_dim {
                    sum += input[t * self.query_dim + k] * self.weights[k * out_dim + j];
                }
                output[t * out_dim + j] = sum;
            }
        }
        output
    }
}

/// Projects encoder hidden states into key and value vectors.
#[derive(Debug, Clone)]
pub struct KeyValueProjection {
    /// Key weight matrix `[key_dim, num_kv_heads * head_dim]`.
    pub key_weights: Vec<f32>,
    /// Key bias `[num_kv_heads * head_dim]`.
    pub key_bias: Vec<f32>,
    /// Value weight matrix `[value_dim, num_kv_heads * head_dim]`.
    pub value_weights: Vec<f32>,
    /// Value bias `[num_kv_heads * head_dim]`.
    pub value_bias: Vec<f32>,
    pub key_dim: usize,
    pub value_dim: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KeyValueProjection {
    /// Create a new KV projection.
    pub fn new(
        key_weights: Vec<f32>,
        key_bias: Vec<f32>,
        value_weights: Vec<f32>,
        value_bias: Vec<f32>,
        key_dim: usize,
        value_dim: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Option<Self> {
        let kv_out = num_kv_heads * head_dim;
        if key_weights.len() != key_dim * kv_out || key_bias.len() != kv_out {
            return None;
        }
        if value_weights.len() != value_dim * kv_out || value_bias.len() != kv_out {
            return None;
        }
        Some(Self {
            key_weights,
            key_bias,
            value_weights,
            value_bias,
            key_dim,
            value_dim,
            num_kv_heads,
            head_dim,
        })
    }

    /// Create identity-like KV projections (for testing).
    pub fn identity(
        key_dim: usize,
        value_dim: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let kv_out = num_kv_heads * head_dim;

        let mut key_weights = vec![0.0f32; key_dim * kv_out];
        let copy_k = key_dim.min(kv_out);
        for i in 0..copy_k {
            key_weights[i * kv_out + i] = 1.0;
        }

        let mut value_weights = vec![0.0f32; value_dim * kv_out];
        let copy_v = value_dim.min(kv_out);
        for i in 0..copy_v {
            value_weights[i * kv_out + i] = 1.0;
        }

        Self {
            key_weights,
            key_bias: vec![0.0; kv_out],
            value_weights,
            value_bias: vec![0.0; kv_out],
            key_dim,
            value_dim,
            num_kv_heads,
            head_dim,
        }
    }

    /// Project encoder hidden states into keys and values.
    ///
    /// `encoder_input` shape: `[enc_len, key_dim]` (`key_dim` == `value_dim` assumed for
    /// single-source encoders; when they differ, pass separate slices).
    /// Returns `(keys, values)` each of shape `[enc_len, num_kv_heads * head_dim]`.
    pub fn forward(&self, encoder_input: &[f32], enc_len: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_out = self.num_kv_heads * self.head_dim;

        let keys = project(
            encoder_input,
            &self.key_weights,
            &self.key_bias,
            enc_len,
            self.key_dim,
            kv_out,
        );
        let values = project(
            encoder_input,
            &self.value_weights,
            &self.value_bias,
            enc_len,
            self.value_dim,
            kv_out,
        );

        (keys, values)
    }
}

/// Dense matrix-vector multiply with bias: `out[t][j] = bias[j] + sum_k(input[t][k] * W[k][j])`.
fn project(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), seq_len * in_dim);
    let mut output = vec![0.0f32; seq_len * out_dim];
    for t in 0..seq_len {
        for j in 0..out_dim {
            let mut sum = bias[j];
            for k in 0..in_dim {
                sum += input[t * in_dim + k] * weights[k * out_dim + j];
            }
            output[t * out_dim + j] = sum;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// Masks for cross-attention.
#[derive(Debug, Clone)]
pub struct CrossAttentionMask {
    /// Encoder padding mask: `true` = valid, `false` = padded.
    /// Shape: `[enc_len]`.
    pub encoder_padding_mask: Vec<bool>,
    /// Optional decoder-encoder alignment mask.
    /// Shape: `[dec_len, enc_len]` (row-major). `true` = attend.
    pub alignment_mask: Option<Vec<bool>>,
    pub enc_len: usize,
    pub dec_len: usize,
}

impl CrossAttentionMask {
    /// Create a mask with only an encoder padding mask.
    pub const fn encoder_only(encoder_padding_mask: Vec<bool>) -> Self {
        let enc_len = encoder_padding_mask.len();
        Self { encoder_padding_mask, alignment_mask: None, enc_len, dec_len: 0 }
    }

    /// Create a full mask with both encoder padding and alignment masks.
    pub fn full(
        encoder_padding_mask: Vec<bool>,
        alignment_mask: Vec<bool>,
        dec_len: usize,
        enc_len: usize,
    ) -> Option<Self> {
        if encoder_padding_mask.len() != enc_len {
            return None;
        }
        if alignment_mask.len() != dec_len * enc_len {
            return None;
        }
        Some(Self { encoder_padding_mask, alignment_mask: Some(alignment_mask), enc_len, dec_len })
    }

    /// Return the effective mask for position `(dec_pos, enc_pos)`.
    pub fn is_valid(&self, dec_pos: usize, enc_pos: usize) -> bool {
        if enc_pos >= self.enc_len {
            return false;
        }
        if !self.encoder_padding_mask[enc_pos] {
            return false;
        }
        if let Some(ref align) = self.alignment_mask
            && dec_pos < self.dec_len
        {
            return align[dec_pos * self.enc_len + enc_pos];
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Attention scores
// ---------------------------------------------------------------------------

/// Computes scaled dot-product attention scores.
#[derive(Debug)]
pub struct CrossAttentionScores;

impl CrossAttentionScores {
    /// Compute attention weights for a single head.
    ///
    /// `queries`: `[q_len, head_dim]`, `keys`: `[kv_len, head_dim]`.
    /// Returns softmax-normalized weights of shape `[q_len, kv_len]`.
    pub fn compute(
        queries: &[f32],
        keys: &[f32],
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        mask: Option<&CrossAttentionMask>,
    ) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; q_len * kv_len];

        // dot products
        for qi in 0..q_len {
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += queries[qi * head_dim + d] * keys[ki * head_dim + d];
                }
                scores[qi * kv_len + ki] = dot * scale;
            }
        }

        // apply mask (set masked positions to -inf)
        if let Some(m) = mask {
            for qi in 0..q_len {
                for ki in 0..kv_len {
                    if !m.is_valid(qi, ki) {
                        scores[qi * kv_len + ki] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // softmax per query
        for qi in 0..q_len {
            let row = &mut scores[qi * kv_len..(qi + 1) * kv_len];
            softmax_inplace(row);
        }

        scores
    }

    /// Compute attention weights using linear attention (no softmax).
    pub fn compute_linear(
        queries: &[f32],
        keys: &[f32],
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; q_len * kv_len];
        for qi in 0..q_len {
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    // ELU + 1 feature map
                    let q_feat = elu_plus_one(queries[qi * head_dim + d]);
                    let k_feat = elu_plus_one(keys[ki * head_dim + d]);
                    dot += q_feat * k_feat;
                }
                scores[qi * kv_len + ki] = dot;
            }
        }
        // normalize rows
        for qi in 0..q_len {
            let row = &mut scores[qi * kv_len..(qi + 1) * kv_len];
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
        scores
    }
}

fn elu_plus_one(x: f32) -> f32 {
    if x >= 0.0 { x + 1.0 } else { x.exp() }
}

fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        // all masked — uniform
        #[allow(clippy::cast_precision_loss)]
        let val = 1.0 / row.len() as f32;
        row.fill(val);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Caches encoder key/value projections across decoding steps.
#[derive(Debug, Clone)]
pub struct CrossAttentionCache {
    /// Cached keys per layer: `layer_id -> [enc_len, num_kv_heads * head_dim]`.
    keys: HashMap<usize, Vec<f32>>,
    /// Cached values per layer.
    values: HashMap<usize, Vec<f32>>,
    /// Encoder sequence length per layer.
    enc_lens: HashMap<usize, usize>,
    /// KV output dim per layer.
    kv_dims: HashMap<usize, usize>,
}

impl CrossAttentionCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            enc_lens: HashMap::new(),
            kv_dims: HashMap::new(),
        }
    }

    /// Store projected keys and values for a layer.
    pub fn insert(
        &mut self,
        layer_id: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
        enc_len: usize,
        kv_dim: usize,
    ) {
        assert_eq!(keys.len(), enc_len * kv_dim);
        assert_eq!(values.len(), enc_len * kv_dim);
        self.keys.insert(layer_id, keys);
        self.values.insert(layer_id, values);
        self.enc_lens.insert(layer_id, enc_len);
        self.kv_dims.insert(layer_id, kv_dim);
    }

    /// Retrieve cached keys and values for a layer.
    pub fn get(&self, layer_id: usize) -> Option<(&[f32], &[f32], usize, usize)> {
        let k = self.keys.get(&layer_id)?;
        let v = self.values.get(&layer_id)?;
        let enc_len = *self.enc_lens.get(&layer_id)?;
        let kv_dim = *self.kv_dims.get(&layer_id)?;
        Some((k, v, enc_len, kv_dim))
    }

    /// Check if a layer has cached values.
    pub fn contains(&self, layer_id: usize) -> bool {
        self.keys.contains_key(&layer_id)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.enc_lens.clear();
        self.kv_dims.clear();
    }

    /// Number of cached layers.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Default for CrossAttentionCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Alignment extraction
// ---------------------------------------------------------------------------

/// Extracts attention alignments for visualization and debugging.
#[derive(Debug, Clone)]
pub struct AlignmentExtractor {
    /// Stored attention weights per head: `head_id -> [q_len, kv_len]`.
    alignments: HashMap<usize, Vec<f32>>,
    pub q_len: usize,
    pub kv_len: usize,
}

impl AlignmentExtractor {
    pub fn new(q_len: usize, kv_len: usize) -> Self {
        Self { alignments: HashMap::new(), q_len, kv_len }
    }

    /// Record attention weights for a given head.
    pub fn record(&mut self, head_id: usize, weights: Vec<f32>) {
        assert_eq!(weights.len(), self.q_len * self.kv_len, "weight size mismatch");
        self.alignments.insert(head_id, weights);
    }

    /// Retrieve alignment for a specific head.
    pub fn get_head_alignment(&self, head_id: usize) -> Option<&[f32]> {
        self.alignments.get(&head_id).map(Vec::as_slice)
    }

    /// Average alignment across all recorded heads.
    pub fn average_alignment(&self) -> Vec<f32> {
        if self.alignments.is_empty() {
            return vec![0.0; self.q_len * self.kv_len];
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.alignments.len() as f32;
        let size = self.q_len * self.kv_len;
        let mut avg = vec![0.0f32; size];
        for weights in self.alignments.values() {
            for (i, &w) in weights.iter().enumerate() {
                avg[i] += w;
            }
        }
        for v in &mut avg {
            *v /= n;
        }
        avg
    }

    /// For each query position, return the encoder position with the highest
    /// average attention weight.
    pub fn hard_alignment(&self) -> Vec<usize> {
        let avg = self.average_alignment();
        (0..self.q_len)
            .map(|qi| {
                let row = &avg[qi * self.kv_len..(qi + 1) * self.kv_len];
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx)
            })
            .collect()
    }

    /// Number of recorded heads.
    pub fn num_heads_recorded(&self) -> usize {
        self.alignments.len()
    }
}

// ---------------------------------------------------------------------------
// Single-head computation
// ---------------------------------------------------------------------------

/// Computes cross-attention for a single head.
#[derive(Debug)]
pub struct CrossAttentionHead {
    pub head_dim: usize,
}

impl CrossAttentionHead {
    pub const fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    /// Compute attention output for one head.
    ///
    /// `queries`: `[q_len, head_dim]`
    /// `keys`:    `[kv_len, head_dim]`
    /// `values`:  `[kv_len, head_dim]`
    ///
    /// Returns `(output [q_len, head_dim], attn_weights [q_len, kv_len])`.
    pub fn forward(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        q_len: usize,
        kv_len: usize,
        mask: Option<&CrossAttentionMask>,
        attention_type: CrossAttentionType,
    ) -> (Vec<f32>, Vec<f32>) {
        let attn_weights = match attention_type {
            CrossAttentionType::Linear => {
                CrossAttentionScores::compute_linear(queries, keys, q_len, kv_len, self.head_dim)
            }
            _ => CrossAttentionScores::compute(queries, keys, q_len, kv_len, self.head_dim, mask),
        };

        // weighted sum of values
        let mut output = vec![0.0f32; q_len * self.head_dim];
        for qi in 0..q_len {
            for d in 0..self.head_dim {
                let mut sum = 0.0f32;
                for ki in 0..kv_len {
                    sum += attn_weights[qi * kv_len + ki] * values[ki * self.head_dim + d];
                }
                output[qi * self.head_dim + d] = sum;
            }
        }

        (output, attn_weights)
    }
}

// ---------------------------------------------------------------------------
// Multi-head cross-attention module
// ---------------------------------------------------------------------------

/// Multi-head cross-attention: project → compute → concat → output.
#[derive(Debug)]
pub struct CrossAttentionModule {
    pub config: CrossAttentionConfig,
    pub attention_type: CrossAttentionType,
    pub query_proj: QueryProjection,
    pub kv_proj: KeyValueProjection,
    /// Output projection weights `[num_heads * head_dim, query_dim]`.
    pub output_weights: Vec<f32>,
    /// Output projection bias `[query_dim]`.
    pub output_bias: Vec<f32>,
}

impl CrossAttentionModule {
    /// Create a new module with identity-like projections (useful for testing).
    pub fn new_identity(config: &CrossAttentionConfig, attention_type: CrossAttentionType) -> Self {
        let num_kv_heads = match attention_type {
            CrossAttentionType::MultiQuery => 1,
            CrossAttentionType::GroupedQuery { num_kv_heads } => num_kv_heads,
            _ => config.num_heads,
        };

        let query_proj =
            QueryProjection::identity(config.query_dim, config.num_heads, config.head_dim);
        let kv_proj = KeyValueProjection::identity(
            config.key_dim,
            config.value_dim,
            num_kv_heads,
            config.head_dim,
        );

        let total = config.num_heads * config.head_dim;
        let out_dim = config.query_dim;
        let mut output_weights = vec![0.0f32; total * out_dim];
        let copy = total.min(out_dim);
        for i in 0..copy {
            output_weights[i * out_dim + i] = 1.0;
        }

        Self {
            config: config.clone(),
            attention_type,
            query_proj,
            kv_proj,
            output_weights,
            output_bias: vec![0.0; config.query_dim],
        }
    }

    /// Number of KV heads for the configured attention type.
    pub const fn num_kv_heads(&self) -> usize {
        self.kv_proj.num_kv_heads
    }

    /// Forward pass through the cross-attention module.
    ///
    /// `decoder_input`: `[dec_len, query_dim]`
    /// `encoder_input`: `[enc_len, key_dim]` (used only if cache miss)
    /// `dec_len`, `enc_len`: sequence lengths
    /// `mask`: optional attention mask
    /// `cache`: optional KV cache
    /// `layer_id`: layer identifier for caching
    ///
    /// Returns `(output [dec_len, query_dim], alignment_extractor)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        decoder_input: &[f32],
        encoder_input: &[f32],
        dec_len: usize,
        enc_len: usize,
        mask: Option<&CrossAttentionMask>,
        cache: Option<&mut CrossAttentionCache>,
        layer_id: usize,
    ) -> (Vec<f32>, AlignmentExtractor) {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.num_kv_heads();
        let kv_dim = num_kv_heads * head_dim;

        // 1. Project queries
        let q_flat = self.query_proj.forward(decoder_input, dec_len);

        // 2. Get or compute keys/values
        let (keys, values, actual_enc_len) = cache.as_ref().map_or_else(
            || {
                let (k, v) = self.kv_proj.forward(encoder_input, enc_len);
                (k, v, enc_len)
            },
            |c| {
                if let Some((ck, cv, el, _)) = c.get(layer_id) {
                    (ck.to_vec(), cv.to_vec(), el)
                } else {
                    let (k, v) = self.kv_proj.forward(encoder_input, enc_len);
                    (k, v, enc_len)
                }
            },
        );

        // Store in cache if provided and not already cached
        if let Some(cache) = cache
            && !cache.contains(layer_id)
        {
            cache.insert(layer_id, keys.clone(), values.clone(), actual_enc_len, kv_dim);
        }

        // 3. Per-head attention
        let mut concat_output = vec![0.0f32; dec_len * num_heads * head_dim];
        let mut extractor = AlignmentExtractor::new(dec_len, actual_enc_len);

        let heads_per_kv = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };

        for h in 0..num_heads {
            let kv_h = if num_kv_heads == num_heads { h } else { h / heads_per_kv };

            // slice queries for this head
            let q_head: Vec<f32> = (0..dec_len)
                .flat_map(|t| {
                    let start = t * num_heads * head_dim + h * head_dim;
                    q_flat[start..start + head_dim].iter().copied()
                })
                .collect();

            // slice keys/values for the corresponding KV head
            let k_head: Vec<f32> = (0..actual_enc_len)
                .flat_map(|t| {
                    let start = t * kv_dim + kv_h * head_dim;
                    keys[start..start + head_dim].iter().copied()
                })
                .collect();
            let v_head: Vec<f32> = (0..actual_enc_len)
                .flat_map(|t| {
                    let start = t * kv_dim + kv_h * head_dim;
                    values[start..start + head_dim].iter().copied()
                })
                .collect();

            let head = CrossAttentionHead::new(head_dim);
            let (head_out, attn_w) = head.forward(
                &q_head,
                &k_head,
                &v_head,
                dec_len,
                actual_enc_len,
                mask,
                self.attention_type,
            );
            extractor.record(h, attn_w);

            // copy head output into concat buffer
            for t in 0..dec_len {
                let src_start = t * head_dim;
                let dst_start = t * num_heads * head_dim + h * head_dim;
                concat_output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&head_out[src_start..src_start + head_dim]);
            }
        }

        // 4. Output projection
        let total = num_heads * head_dim;
        let out_dim = self.config.query_dim;
        let output = project(
            &concat_output,
            &self.output_weights,
            &self.output_bias,
            dec_len,
            total,
            out_dim,
        );

        (output, extractor)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn sum_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    // --- Config tests ---

    #[test]
    fn test_config_valid() {
        let c = CrossAttentionConfig::new(64, 64, 64, 4, 16, 0.1);
        assert!(c.is_some());
        assert_eq!(c.unwrap().total_head_dim(), 64);
    }

    #[test]
    fn test_config_zero_heads() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 0, 16, 0.1).is_none());
    }

    #[test]
    fn test_config_zero_head_dim() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 4, 0, 0.1).is_none());
    }

    #[test]
    fn test_config_zero_query_dim() {
        assert!(CrossAttentionConfig::new(0, 64, 64, 4, 16, 0.1).is_none());
    }

    #[test]
    fn test_config_zero_key_dim() {
        assert!(CrossAttentionConfig::new(64, 0, 64, 4, 16, 0.1).is_none());
    }

    #[test]
    fn test_config_zero_value_dim() {
        assert!(CrossAttentionConfig::new(64, 64, 0, 4, 16, 0.1).is_none());
    }

    #[test]
    fn test_config_negative_dropout() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 4, 16, -0.1).is_none());
    }

    #[test]
    fn test_config_dropout_above_one() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 4, 16, 1.1).is_none());
    }

    #[test]
    fn test_config_dropout_zero() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 4, 16, 0.0).is_some());
    }

    #[test]
    fn test_config_dropout_one() {
        assert!(CrossAttentionConfig::new(64, 64, 64, 4, 16, 1.0).is_some());
    }

    #[test]
    fn test_config_clone_eq() {
        let c = CrossAttentionConfig::new(32, 48, 48, 2, 8, 0.05).unwrap();
        assert_eq!(c, c.clone());
    }

    // --- CrossAttentionType tests ---

    #[test]
    fn test_attention_type_display_standard() {
        assert_eq!(format!("{}", CrossAttentionType::Standard), "Standard");
    }

    #[test]
    fn test_attention_type_display_multi_query() {
        assert_eq!(format!("{}", CrossAttentionType::MultiQuery), "MultiQuery");
    }

    #[test]
    fn test_attention_type_display_grouped_query() {
        let t = CrossAttentionType::GroupedQuery { num_kv_heads: 2 };
        assert_eq!(format!("{t}"), "GroupedQuery(kv_heads=2)");
    }

    #[test]
    fn test_attention_type_display_linear() {
        assert_eq!(format!("{}", CrossAttentionType::Linear), "Linear");
    }

    #[test]
    fn test_attention_type_display_sparse() {
        let t = CrossAttentionType::Sparse { block_size: 4 };
        assert_eq!(format!("{t}"), "Sparse(block=4)");
    }

    #[test]
    fn test_attention_type_equality() {
        assert_eq!(CrossAttentionType::Standard, CrossAttentionType::Standard);
        assert_ne!(CrossAttentionType::Standard, CrossAttentionType::Linear);
    }

    #[test]
    fn test_attention_type_grouped_query_different_heads() {
        let a = CrossAttentionType::GroupedQuery { num_kv_heads: 2 };
        let b = CrossAttentionType::GroupedQuery { num_kv_heads: 4 };
        assert_ne!(a, b);
    }

    // --- QueryProjection tests ---

    #[test]
    fn test_query_projection_identity() {
        let proj = QueryProjection::identity(4, 1, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = proj.forward(&input, 1);
        assert!(sum_approx_eq(&out, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_query_projection_multi_token() {
        let proj = QueryProjection::identity(2, 1, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        let out = proj.forward(&input, 2);
        assert!(sum_approx_eq(&out, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_query_projection_with_bias() {
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let bias = vec![0.5, 0.5];
        let proj = QueryProjection::new(weights, bias, 2, 1, 2).unwrap();
        let input = vec![1.0, 2.0];
        let out = proj.forward(&input, 1);
        assert!(sum_approx_eq(&out, &[1.5, 2.5]));
    }

    #[test]
    fn test_query_projection_invalid_weights() {
        assert!(QueryProjection::new(vec![1.0], vec![0.0, 0.0], 2, 1, 2).is_none());
    }

    #[test]
    fn test_query_projection_invalid_bias() {
        assert!(QueryProjection::new(vec![1.0; 4], vec![0.0], 2, 1, 2).is_none());
    }

    // --- KeyValueProjection tests ---

    #[test]
    fn test_kv_projection_identity() {
        let kv = KeyValueProjection::identity(4, 4, 1, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (keys, values) = kv.forward(&input, 1);
        assert!(sum_approx_eq(&keys, &[1.0, 2.0, 3.0, 4.0]));
        assert!(sum_approx_eq(&values, &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_kv_projection_multi_token() {
        let kv = KeyValueProjection::identity(2, 2, 1, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens of dim 2
        let (keys, values) = kv.forward(&input, 2);
        assert_eq!(keys.len(), 4);
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn test_kv_projection_invalid_key_weights() {
        assert!(
            KeyValueProjection::new(
                vec![1.0],
                vec![0.0; 2],
                vec![1.0; 4],
                vec![0.0; 2],
                2,
                2,
                1,
                2
            )
            .is_none()
        );
    }

    #[test]
    fn test_kv_projection_invalid_value_weights() {
        assert!(
            KeyValueProjection::new(
                vec![1.0; 4],
                vec![0.0; 2],
                vec![1.0],
                vec![0.0; 2],
                2,
                2,
                1,
                2
            )
            .is_none()
        );
    }

    #[test]
    fn test_kv_projection_multiple_kv_heads() {
        let kv = KeyValueProjection::identity(4, 4, 2, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (keys, _values) = kv.forward(&input, 1);
        assert_eq!(keys.len(), 4); // 1 * 2 * 2
    }

    // --- Mask tests ---

    #[test]
    fn test_mask_encoder_only() {
        let mask = CrossAttentionMask::encoder_only(vec![true, true, false]);
        assert!(mask.is_valid(0, 0));
        assert!(mask.is_valid(0, 1));
        assert!(!mask.is_valid(0, 2));
    }

    #[test]
    fn test_mask_out_of_bounds() {
        let mask = CrossAttentionMask::encoder_only(vec![true, true]);
        assert!(!mask.is_valid(0, 5));
    }

    #[test]
    fn test_mask_full() {
        // 2 decoder positions, 3 encoder positions
        let align = vec![true, false, true, false, true, false];
        let mask = CrossAttentionMask::full(vec![true, true, true], align, 2, 3).unwrap();
        assert!(mask.is_valid(0, 0));
        assert!(!mask.is_valid(0, 1));
        assert!(mask.is_valid(0, 2));
        assert!(!mask.is_valid(1, 0));
        assert!(mask.is_valid(1, 1));
        assert!(!mask.is_valid(1, 2));
    }

    #[test]
    fn test_mask_full_with_padding() {
        let align = vec![true, true, true, true];
        let mask = CrossAttentionMask::full(vec![true, false], align, 2, 2).unwrap();
        // enc_pos=1 is padded
        assert!(mask.is_valid(0, 0));
        assert!(!mask.is_valid(0, 1)); // encoder padding overrides alignment
    }

    #[test]
    fn test_mask_full_invalid_enc_len() {
        assert!(CrossAttentionMask::full(vec![true, true], vec![true; 4], 2, 3).is_none());
    }

    #[test]
    fn test_mask_full_invalid_align_len() {
        assert!(CrossAttentionMask::full(vec![true, true], vec![true; 3], 2, 2).is_none());
    }

    #[test]
    fn test_mask_all_padded() {
        let mask = CrossAttentionMask::encoder_only(vec![false, false, false]);
        assert!(!mask.is_valid(0, 0));
        assert!(!mask.is_valid(0, 1));
        assert!(!mask.is_valid(0, 2));
    }

    // --- Softmax tests ---

    #[test]
    fn test_softmax_uniform() {
        let mut row = vec![0.0, 0.0, 0.0];
        softmax_inplace(&mut row);
        let expected = 1.0 / 3.0;
        for &v in &row {
            assert!(approx_eq(v, expected));
        }
    }

    #[test]
    fn test_softmax_single() {
        let mut row = vec![5.0];
        softmax_inplace(&mut row);
        assert!(approx_eq(row[0], 1.0));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_softmax_ordering_preserved() {
        let mut row = vec![1.0, 3.0, 2.0];
        softmax_inplace(&mut row);
        assert!(row[1] > row[2]);
        assert!(row[2] > row[0]);
    }

    #[test]
    fn test_softmax_all_neg_inf() {
        let mut row = vec![f32::NEG_INFINITY; 3];
        softmax_inplace(&mut row);
        // uniform fallback
        let expected = 1.0 / 3.0;
        for &v in &row {
            assert!(approx_eq(v, expected));
        }
    }

    #[test]
    fn test_softmax_empty() {
        let mut row: Vec<f32> = vec![];
        softmax_inplace(&mut row);
        assert!(row.is_empty());
    }

    // --- CrossAttentionScores tests ---

    #[test]
    fn test_scores_identity_keys() {
        // q = k = [1,0], [0,1] → scores should be identity-like after softmax
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = CrossAttentionScores::compute(&q, &k, 2, 2, 2, None);
        // q0 · k0 > q0 · k1, so scores[0] > scores[1]
        assert!(scores[0] > scores[1]);
        assert!(scores[3] > scores[2]);
    }

    #[test]
    fn test_scores_with_mask() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
        let mask = CrossAttentionMask::encoder_only(vec![true, false]);
        let scores = CrossAttentionScores::compute(&q, &k, 1, 2, 2, Some(&mask));
        // second key is masked, so all weight goes to first
        assert!(approx_eq(scores[0], 1.0));
        assert!(approx_eq(scores[1], 0.0));
    }

    #[test]
    fn test_scores_sum_to_one() {
        let q = vec![0.5, 0.3, 0.2, 0.1];
        let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let scores = CrossAttentionScores::compute(&q, &k, 2, 3, 2, None);
        for qi in 0..2 {
            let row_sum: f32 = scores[qi * 3..(qi + 1) * 3].iter().sum();
            assert!(approx_eq(row_sum, 1.0));
        }
    }

    #[test]
    fn test_scores_linear_positive() {
        let q = vec![1.0, 2.0];
        let k = vec![1.0, 1.0, 0.5, 0.5];
        let scores = CrossAttentionScores::compute_linear(&q, &k, 1, 2, 2);
        for &s in &scores {
            assert!(s >= 0.0);
        }
        let sum: f32 = scores.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_scores_scaling() {
        // With head_dim = 4, scale = 1/2
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let scores = CrossAttentionScores::compute(&q, &k, 1, 1, 4, None);
        assert!(approx_eq(scores[0], 1.0)); // only one key, softmax = 1
    }

    // --- KV Cache tests ---

    #[test]
    fn test_cache_empty() {
        let cache = CrossAttentionCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(0));
    }

    #[test]
    fn test_cache_insert_get() {
        let mut cache = CrossAttentionCache::new();
        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![5.0, 6.0, 7.0, 8.0];
        cache.insert(0, keys.clone(), values.clone(), 2, 2);
        assert!(cache.contains(0));
        assert_eq!(cache.len(), 1);
        let (k, v, el, kd) = cache.get(0).unwrap();
        assert_eq!(k, &keys);
        assert_eq!(v, &values);
        assert_eq!(el, 2);
        assert_eq!(kd, 2);
    }

    #[test]
    fn test_cache_multiple_layers() {
        let mut cache = CrossAttentionCache::new();
        cache.insert(0, vec![1.0; 4], vec![2.0; 4], 2, 2);
        cache.insert(1, vec![3.0; 6], vec![4.0; 6], 3, 2);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(0));
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = CrossAttentionCache::new();
        cache.insert(0, vec![1.0; 4], vec![2.0; 4], 2, 2);
        cache.insert(0, vec![9.0; 4], vec![8.0; 4], 2, 2);
        let (k, _, _, _) = cache.get(0).unwrap();
        assert!(approx_eq(k[0], 9.0));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = CrossAttentionCache::new();
        cache.insert(0, vec![1.0; 4], vec![2.0; 4], 2, 2);
        cache.clear();
        assert!(cache.is_empty());
        assert!(!cache.contains(0));
    }

    #[test]
    fn test_cache_get_missing() {
        let cache = CrossAttentionCache::new();
        assert!(cache.get(42).is_none());
    }

    #[test]
    fn test_cache_default() {
        let cache = CrossAttentionCache::default();
        assert!(cache.is_empty());
    }

    // --- AlignmentExtractor tests ---

    #[test]
    fn test_alignment_empty() {
        let ext = AlignmentExtractor::new(2, 3);
        assert_eq!(ext.num_heads_recorded(), 0);
        let avg = ext.average_alignment();
        assert_eq!(avg.len(), 6);
        assert!(avg.iter().all(|&v| approx_eq(v, 0.0)));
    }

    #[test]
    fn test_alignment_record_and_get() {
        let mut ext = AlignmentExtractor::new(1, 2);
        let w = vec![0.7, 0.3];
        ext.record(0, w.clone());
        assert_eq!(ext.num_heads_recorded(), 1);
        assert_eq!(ext.get_head_alignment(0).unwrap(), &w);
    }

    #[test]
    fn test_alignment_average_single_head() {
        let mut ext = AlignmentExtractor::new(1, 3);
        ext.record(0, vec![0.2, 0.5, 0.3]);
        let avg = ext.average_alignment();
        assert!(sum_approx_eq(&avg, &[0.2, 0.5, 0.3]));
    }

    #[test]
    fn test_alignment_average_two_heads() {
        let mut ext = AlignmentExtractor::new(1, 2);
        ext.record(0, vec![0.8, 0.2]);
        ext.record(1, vec![0.4, 0.6]);
        let avg = ext.average_alignment();
        assert!(sum_approx_eq(&avg, &[0.6, 0.4]));
    }

    #[test]
    fn test_alignment_hard_alignment() {
        let mut ext = AlignmentExtractor::new(2, 3);
        // q0 attends most to enc pos 2, q1 to enc pos 0
        ext.record(0, vec![0.1, 0.2, 0.7, 0.6, 0.3, 0.1]);
        let hard = ext.hard_alignment();
        assert_eq!(hard, vec![2, 0]);
    }

    #[test]
    fn test_alignment_get_missing_head() {
        let ext = AlignmentExtractor::new(1, 1);
        assert!(ext.get_head_alignment(99).is_none());
    }

    // --- CrossAttentionHead tests ---

    #[test]
    fn test_head_basic() {
        let head = CrossAttentionHead::new(2);
        let q = vec![1.0, 0.0]; // 1 query
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
        let v = vec![10.0, 20.0, 30.0, 40.0]; // 2 values
        let (out, weights) = head.forward(&q, &k, &v, 1, 2, None, CrossAttentionType::Standard);
        assert_eq!(out.len(), 2);
        assert_eq!(weights.len(), 2);
        // q aligns more with k0, so output biased toward v0
        assert!(out[0] < 25.0); // closer to 10 than 30
    }

    #[test]
    fn test_head_single_key() {
        let head = CrossAttentionHead::new(2);
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0]; // single key
        let v = vec![5.0, 6.0];
        let (out, weights) = head.forward(&q, &k, &v, 1, 1, None, CrossAttentionType::Standard);
        // single key → weight = 1.0, output = v
        assert!(approx_eq(weights[0], 1.0));
        assert!(sum_approx_eq(&out, &[5.0, 6.0]));
    }

    #[test]
    fn test_head_linear_attention() {
        let head = CrossAttentionHead::new(2);
        let q = vec![1.0, 0.5];
        let k = vec![0.5, 1.0, 1.0, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let (out, weights) = head.forward(&q, &k, &v, 1, 2, None, CrossAttentionType::Linear);
        assert_eq!(out.len(), 2);
        assert_eq!(weights.len(), 2);
        let wsum: f32 = weights.iter().sum();
        assert!(approx_eq(wsum, 1.0));
    }

    #[test]
    fn test_head_with_mask() {
        let head = CrossAttentionHead::new(2);
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mask = CrossAttentionMask::encoder_only(vec![false, true]);
        let (out, weights) =
            head.forward(&q, &k, &v, 1, 2, Some(&mask), CrossAttentionType::Standard);
        // only second key is valid
        assert!(approx_eq(weights[1], 1.0));
        assert!(sum_approx_eq(&out, &[30.0, 40.0]));
    }

    #[test]
    fn test_head_multiple_queries() {
        let head = CrossAttentionHead::new(2);
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 queries
        let k = vec![1.0, 0.0]; // 1 key
        let v = vec![5.0, 10.0];
        let (out, _) = head.forward(&q, &k, &v, 2, 1, None, CrossAttentionType::Standard);
        // both queries attend 100% to single key
        assert!(sum_approx_eq(&out, &[5.0, 10.0, 5.0, 10.0]));
    }

    // --- CrossAttentionModule tests ---

    fn make_config(
        query_dim: usize,
        key_dim: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> CrossAttentionConfig {
        CrossAttentionConfig::new(query_dim, key_dim, key_dim, num_heads, head_dim, 0.0).unwrap()
    }

    #[test]
    fn test_module_standard_basic() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0, 0.0, 0.0, 1.0]; // 1 token, dim=4
        let enc = vec![1.0, 0.0, 0.0, 1.0]; // 1 token, dim=4
        let (out, ext) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 4);
        assert_eq!(ext.num_heads_recorded(), 2);
    }

    #[test]
    fn test_module_multi_query() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::MultiQuery);
        assert_eq!(module.num_kv_heads(), 1);
        let dec = vec![1.0; 4];
        let enc = vec![1.0; 4];
        let (out, _) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_module_grouped_query() {
        let config = make_config(8, 8, 4, 2);
        let module = CrossAttentionModule::new_identity(
            &config,
            CrossAttentionType::GroupedQuery { num_kv_heads: 2 },
        );
        assert_eq!(module.num_kv_heads(), 2);
        let dec = vec![1.0; 8];
        let enc = vec![1.0; 8];
        let (out, ext) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 8);
        assert_eq!(ext.num_heads_recorded(), 4);
    }

    #[test]
    fn test_module_with_cache() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let mut cache = CrossAttentionCache::new();
        let enc = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]; // 2 enc tokens

        // First call: populates cache
        let dec1 = vec![1.0, 0.0, 0.0, 1.0];
        let (out1, _) = module.forward(&dec1, &enc, 1, 2, None, Some(&mut cache), 0);
        assert!(cache.contains(0));

        // Second call with different decoder input: reuses cache
        let dec2 = vec![0.0, 1.0, 1.0, 0.0];
        let (out2, _) = module.forward(&dec2, &[], 1, 0, None, Some(&mut cache), 0);
        assert_eq!(out2.len(), 4);
        // outputs should differ since decoder inputs differ
        assert!(!sum_approx_eq(&out1, &out2));
    }

    #[test]
    fn test_module_cache_reuse_consistency() {
        let config = make_config(4, 4, 1, 4);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let mut cache = CrossAttentionCache::new();
        let enc = vec![1.0, 2.0, 3.0, 4.0];
        let dec = vec![0.5, 0.5, 0.5, 0.5];

        // With cache
        let (out_cached, _) = module.forward(&dec, &enc, 1, 1, None, Some(&mut cache), 0);
        // Without cache
        let (out_no_cache, _) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert!(sum_approx_eq(&out_cached, &out_no_cache));
    }

    #[test]
    fn test_module_alignment_matches_weights() {
        let config = make_config(2, 2, 1, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0, 0.0];
        let enc = vec![1.0, 0.0, 0.0, 1.0]; // 2 enc tokens
        let (_, ext) = module.forward(&dec, &enc, 1, 2, None, None, 0);

        let weights = ext.get_head_alignment(0).unwrap();
        assert_eq!(weights.len(), 2);
        let sum: f32 = weights.iter().sum();
        assert!(approx_eq(sum, 1.0));
        // q=[1,0] aligns more with k0=[1,0]
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_module_mask_zeros_padding() {
        let config = make_config(2, 2, 1, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0, 0.0];
        let enc = vec![1.0, 0.0, 0.0, 1.0]; // 2 enc tokens
        let mask = CrossAttentionMask::encoder_only(vec![true, false]); // mask out enc[1]
        let (_, ext) = module.forward(&dec, &enc, 1, 2, Some(&mask), None, 0);

        let w = ext.get_head_alignment(0).unwrap();
        assert!(approx_eq(w[0], 1.0));
        assert!(approx_eq(w[1], 0.0));
    }

    #[test]
    fn test_module_different_query_key_dims() {
        let config = CrossAttentionConfig::new(6, 4, 4, 2, 3, 0.0).unwrap();
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0; 6]; // query_dim=6
        let enc = vec![1.0; 4]; // key_dim=4
        let (out, _) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_module_multi_decoder_tokens() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0; 12]; // 3 dec tokens
        let enc = vec![1.0; 8]; // 2 enc tokens
        let (out, ext) = module.forward(&dec, &enc, 3, 2, None, None, 0);
        assert_eq!(out.len(), 12);
        assert_eq!(ext.q_len, 3);
        assert_eq!(ext.kv_len, 2);
    }

    #[test]
    fn test_module_single_token_query() {
        let config = make_config(2, 2, 1, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0, 0.5]; // 1 token
        let enc = vec![0.5, 1.0, 1.0, 0.5]; // 2 enc tokens
        let (out, _) = module.forward(&dec, &enc, 1, 2, None, None, 0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_module_linear_attention() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Linear);
        let dec = vec![1.0; 4];
        let enc = vec![1.0; 8]; // 2 tokens
        let (out, ext) = module.forward(&dec, &enc, 1, 2, None, None, 0);
        assert_eq!(out.len(), 4);
        // Linear attention weights should still sum to ~1
        let w = ext.get_head_alignment(0).unwrap();
        let sum: f32 = w.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_single_head_multi_head_consistency() {
        // With 1 head, module should match single head computation
        let config = make_config(2, 2, 1, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);

        let dec = vec![1.0, 0.0];
        let enc = vec![0.5, 0.5, 1.0, 0.0];

        let (mod_out, _) = module.forward(&dec, &enc, 1, 2, None, None, 0);

        // Manual single head
        let head = CrossAttentionHead::new(2);
        let (head_out, _) =
            head.forward(&dec, &enc, &enc, 1, 2, None, CrossAttentionType::Standard);

        assert!(sum_approx_eq(&mod_out, &head_out));
    }

    #[test]
    fn test_grouped_query_fewer_kv_heads() {
        // 4 query heads sharing 2 KV heads
        let config = make_config(8, 8, 4, 2);
        let module = CrossAttentionModule::new_identity(
            &config,
            CrossAttentionType::GroupedQuery { num_kv_heads: 2 },
        );
        let dec = vec![1.0; 8];
        let enc = vec![1.0; 8];
        let (out, ext) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 8);
        // heads 0,1 share kv_head 0; heads 2,3 share kv_head 1
        let w0 = ext.get_head_alignment(0).unwrap();
        let w1 = ext.get_head_alignment(1).unwrap();
        // Same KV head → same attention weights for identity projections
        assert!(sum_approx_eq(w0, w1));
    }

    #[test]
    fn test_module_output_shape_various() {
        for (qd, kd, nh, hd) in [(4, 4, 2, 2), (8, 4, 4, 2), (6, 6, 3, 2), (2, 2, 1, 2)] {
            let config = CrossAttentionConfig::new(qd, kd, kd, nh, hd, 0.0).unwrap();
            let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
            let dec = vec![0.5; qd * 2]; // 2 dec tokens
            let enc = vec![0.5; kd * 3]; // 3 enc tokens
            let (out, _) = module.forward(&dec, &enc, 2, 3, None, None, 0);
            assert_eq!(out.len(), qd * 2, "output shape mismatch for qd={qd}");
        }
    }

    #[test]
    fn test_elu_plus_one_positive() {
        assert!(approx_eq(elu_plus_one(0.0), 1.0));
        assert!(approx_eq(elu_plus_one(1.0), 2.0));
    }

    #[test]
    fn test_elu_plus_one_negative() {
        let val = elu_plus_one(-1.0);
        assert!(val > 0.0);
        assert!(val < 1.0);
    }

    #[test]
    fn test_project_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0];
        let out = project(&input, &weights, &bias, 2, 2, 2);
        assert!(sum_approx_eq(&out, &input));
    }

    #[test]
    fn test_cache_kv_reuse_across_steps() {
        let config = make_config(2, 2, 1, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let mut cache = CrossAttentionCache::new();
        let enc = vec![1.0, 0.0, 0.0, 1.0]; // 2 enc tokens

        // Step 1
        let d1 = vec![1.0, 0.0];
        let (_, e1) = module.forward(&d1, &enc, 1, 2, None, Some(&mut cache), 0);

        // Step 2 (different decoder state, cached encoder)
        let d2 = vec![0.0, 1.0];
        let (_, e2) = module.forward(&d2, &[], 1, 0, None, Some(&mut cache), 0);

        // Both steps should have alignment over 2 encoder positions
        assert_eq!(e1.kv_len, 2);
        assert_eq!(e2.kv_len, 2);
    }

    #[test]
    fn test_mask_combined_encoder_and_alignment() {
        // Encoder has 3 positions, 2nd is padded. Alignment further restricts.
        let enc_mask = vec![true, false, true];
        let align = vec![true, true, false]; // dec[0] can attend to enc[0,1] but not enc[2]
        let mask = CrossAttentionMask::full(enc_mask, align, 1, 3).unwrap();

        assert!(mask.is_valid(0, 0)); // enc valid, align allows
        assert!(!mask.is_valid(0, 1)); // enc padded
        assert!(!mask.is_valid(0, 2)); // align blocks
    }

    #[test]
    fn test_head_output_weighted_sum() {
        let head = CrossAttentionHead::new(1);
        let q = vec![1.0]; // trivial
        let k = vec![1.0, 1.0]; // 2 identical keys
        let v = vec![10.0, 20.0]; // different values
        let (out, weights) = head.forward(&q, &k, &v, 1, 2, None, CrossAttentionType::Standard);
        // Equal keys → equal weights → output = average
        assert!(approx_eq(weights[0], 0.5));
        assert!(approx_eq(weights[1], 0.5));
        assert!(approx_eq(out[0], 15.0));
    }

    #[test]
    fn test_module_no_crash_large_enc() {
        let config = make_config(4, 4, 2, 2);
        let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
        let dec = vec![1.0; 4];
        let enc = vec![0.1; 4 * 50]; // 50 encoder tokens
        let (out, _) = module.forward(&dec, &enc, 1, 50, None, None, 0);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_alignment_hard_alignment_tie() {
        let mut ext = AlignmentExtractor::new(1, 2);
        ext.record(0, vec![0.5, 0.5]);
        let hard = ext.hard_alignment();
        // Tie-breaking: either 0 or 1 is acceptable
        assert!(hard[0] == 0 || hard[0] == 1);
    }

    #[test]
    fn test_config_total_head_dim() {
        let c = CrossAttentionConfig::new(128, 128, 128, 8, 16, 0.0).unwrap();
        assert_eq!(c.total_head_dim(), 128);
    }

    #[test]
    fn test_scores_all_masked() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let mask = CrossAttentionMask::encoder_only(vec![false, false]);
        let scores = CrossAttentionScores::compute(&q, &k, 1, 2, 2, Some(&mask));
        // All masked → uniform fallback
        assert!(approx_eq(scores[0], 0.5));
        assert!(approx_eq(scores[1], 0.5));
    }

    #[test]
    fn test_module_sparse_type_compiles() {
        let config = make_config(4, 4, 2, 2);
        // Sparse still uses standard score computation for now
        let module = CrossAttentionModule::new_identity(
            &config,
            CrossAttentionType::Sparse { block_size: 2 },
        );
        let dec = vec![1.0; 4];
        let enc = vec![1.0; 4];
        let (out, _) = module.forward(&dec, &enc, 1, 1, None, None, 0);
        assert_eq!(out.len(), 4);
    }

    // --- proptest ---

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn arb_dims()(
                query_dim in 1_usize..=16,
                key_dim in 1_usize..=16,
                num_heads in 1_usize..=4,
                head_dim in 1_usize..=8,
            ) -> (usize, usize, usize, usize) {
                (query_dim, key_dim, num_heads, head_dim)
            }
        }

        proptest! {
            #[test]
            fn prop_config_valid_always_some(
                qd in 1_usize..=64,
                kd in 1_usize..=64,
                nh in 1_usize..=8,
                hd in 1_usize..=16,
                dr in 0.0f32..=1.0,
            ) {
                let c = CrossAttentionConfig::new(qd, kd, kd, nh, hd, dr);
                prop_assert!(c.is_some());
            }

            #[test]
            fn prop_softmax_sums_to_one(vals in proptest::collection::vec(-10.0f32..10.0, 1..20)) {
                let mut row = vals;
                softmax_inplace(&mut row);
                let sum: f32 = row.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-4);
            }

            #[test]
            fn prop_softmax_non_negative(vals in proptest::collection::vec(-100.0f32..100.0, 1..20)) {
                let mut row = vals;
                softmax_inplace(&mut row);
                for &v in &row {
                    prop_assert!(v >= 0.0);
                }
            }

            #[test]
            fn prop_scores_sum_to_one(
                q_len in 1_usize..=4,
                kv_len in 1_usize..=4,
                head_dim in 1_usize..=8,
            ) {
                let q = vec![0.5f32; q_len * head_dim];
                let k = vec![0.5f32; kv_len * head_dim];
                let scores = CrossAttentionScores::compute(&q, &k, q_len, kv_len, head_dim, None);
                for qi in 0..q_len {
                    let row_sum: f32 = scores[qi * kv_len..(qi + 1) * kv_len].iter().sum();
                    prop_assert!((row_sum - 1.0).abs() < 1e-4, "row {qi} sum = {row_sum}");
                }
            }

            #[test]
            fn prop_module_output_correct_length(
                (qd, kd, nh, hd) in arb_dims(),
                dec_len in 1_usize..=4,
                enc_len in 1_usize..=4,
            ) {
                let config = CrossAttentionConfig::new(qd, kd, kd, nh, hd, 0.0).unwrap();
                let module = CrossAttentionModule::new_identity(&config, CrossAttentionType::Standard);
                let dec = vec![0.1f32; dec_len * qd];
                let enc = vec![0.1f32; enc_len * kd];
                let (out, ext) = module.forward(&dec, &enc, dec_len, enc_len, None, None, 0);
                prop_assert_eq!(out.len(), dec_len * qd);
                prop_assert_eq!(ext.num_heads_recorded(), nh);
            }

            #[test]
            fn prop_cache_round_trip(
                enc_len in 1_usize..=8,
                kv_dim in 1_usize..=8,
                layer_id in 0_usize..=10,
            ) {
                let mut cache = CrossAttentionCache::new();
                let k = vec![1.0f32; enc_len * kv_dim];
                let v = vec![2.0f32; enc_len * kv_dim];
                cache.insert(layer_id, k.clone(), v.clone(), enc_len, kv_dim);
                let (gk, gv, gel, gkd) = cache.get(layer_id).unwrap();
                prop_assert_eq!(gk, k.as_slice());
                prop_assert_eq!(gv, v.as_slice());
                prop_assert_eq!(gel, enc_len);
                prop_assert_eq!(gkd, kv_dim);
            }

            #[test]
            fn prop_alignment_average_bounded(
                q_len in 1_usize..=4,
                kv_len in 1_usize..=4,
                num_heads in 1_usize..=4,
            ) {
                let mut ext = AlignmentExtractor::new(q_len, kv_len);
                for h in 0..num_heads {
                    // Create valid attention weights (sum to 1 per query)
                    let mut weights = vec![0.0f32; q_len * kv_len];
                    for qi in 0..q_len {
                        let val = 1.0 / kv_len as f32;
                        for ki in 0..kv_len {
                            weights[qi * kv_len + ki] = val;
                        }
                    }
                    ext.record(h, weights);
                }
                let avg = ext.average_alignment();
                for &v in &avg {
                    prop_assert!(v >= 0.0);
                    prop_assert!(v <= 1.0 + 1e-6);
                }
            }

            #[test]
            fn prop_elu_plus_one_positive_result(x in -5.0f32..5.0) {
                let result = elu_plus_one(x);
                prop_assert!(result > 0.0, "elu_plus_one({x}) = {result} should be > 0");
            }

            #[test]
            fn prop_mask_encoder_consistency(len in 1_usize..=20) {
                let mask_vec: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
                let mask = CrossAttentionMask::encoder_only(mask_vec.clone());
                for (i, &valid) in mask_vec.iter().enumerate() {
                    prop_assert_eq!(mask.is_valid(0, i), valid);
                }
            }

            #[test]
            fn prop_query_projection_output_length(
                qd in 1_usize..=8,
                nh in 1_usize..=4,
                hd in 1_usize..=4,
                seq_len in 1_usize..=4,
            ) {
                let proj = QueryProjection::identity(qd, nh, hd);
                let input = vec![0.5f32; seq_len * qd];
                let out = proj.forward(&input, seq_len);
                prop_assert_eq!(out.len(), seq_len * nh * hd);
            }

            #[test]
            fn prop_kv_projection_output_length(
                kd in 1_usize..=8,
                kv_heads in 1_usize..=4,
                hd in 1_usize..=4,
                enc_len in 1_usize..=4,
            ) {
                let kv = KeyValueProjection::identity(kd, kd, kv_heads, hd);
                let input = vec![0.5f32; enc_len * kd];
                let (keys, values) = kv.forward(&input, enc_len);
                let expected = enc_len * kv_heads * hd;
                prop_assert_eq!(keys.len(), expected);
                prop_assert_eq!(values.len(), expected);
            }

            #[test]
            fn prop_linear_scores_non_negative(
                q_len in 1_usize..=3,
                kv_len in 1_usize..=3,
                head_dim in 1_usize..=4,
            ) {
                let q = vec![0.5f32; q_len * head_dim];
                let k = vec![0.5f32; kv_len * head_dim];
                let scores = CrossAttentionScores::compute_linear(&q, &k, q_len, kv_len, head_dim);
                for &s in &scores {
                    prop_assert!(s >= 0.0, "linear score {s} should be >= 0");
                }
            }
        }
    }
}
