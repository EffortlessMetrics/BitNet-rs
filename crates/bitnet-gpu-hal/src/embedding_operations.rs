//! Module stub - implementation pending merge from feature branch
//! Embedding operations for GPU-accelerated inference pipelines.
//!
//! Provides token embeddings, positional encodings (sinusoidal, RoPE, ALiBi),
//! quantized embedding tables, similarity search, aggregation, and normalization.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for an embedding layer.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Vocabulary size (number of unique tokens).
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Optional padding index whose embedding is kept at zero.
    pub padding_idx: Option<usize>,
    /// Optional max-norm constraint for embedding vectors.
    pub max_norm: Option<f32>,
}

impl EmbeddingConfig {
    /// Create a new embedding configuration.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self { vocab_size, embed_dim, padding_idx: None, max_norm: None }
    }

    /// Set the padding index.
    pub fn with_padding_idx(mut self, idx: usize) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Set the max-norm constraint.
    pub fn with_max_norm(mut self, norm: f32) -> Self {
        self.max_norm = Some(norm);
        self
    }
}

// ── Token Embedding ──────────────────────────────────────────────────────────

/// Standard token embedding lookup table.
///
/// Stores a weight matrix of shape `[vocab_size, embed_dim]` and retrieves
/// embedding vectors by token ID.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    config: EmbeddingConfig,
    /// Row-major weight table: `weights[token_id * embed_dim .. (token_id+1) * embed_dim]`.
    weights: Vec<f32>,
}

impl TokenEmbedding {
    /// Create a new token embedding table initialized with small random-like values.
    ///
    /// Uses a deterministic hash-based initialization for reproducibility.
    pub fn new(config: EmbeddingConfig) -> Self {
        let n = config.vocab_size * config.embed_dim;
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic pseudo-random initialization in [-0.1, 0.1].
            let hash = ((i as u64).wrapping_mul(2654435761) ^ 0xDEAD_BEEF) as f32;
            weights.push((hash % 1000.0) / 10000.0);
        }
        // Zero out padding row if specified.
        if let Some(pad) = config.padding_idx {
            let start = pad * config.embed_dim;
            let end = start + config.embed_dim;
            if end <= weights.len() {
                weights[start..end].fill(0.0);
            }
        }
        Self { config, weights }
    }

    /// Create a token embedding from an existing weight table.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != config.vocab_size * config.embed_dim`.
    pub fn from_weights(config: EmbeddingConfig, weights: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            config.vocab_size * config.embed_dim,
            "weight table size mismatch"
        );
        Self { config, weights }
    }

    /// Look up the embedding for a single token.
    ///
    /// Returns `None` if `token_id >= vocab_size`.
    pub fn lookup(&self, token_id: usize) -> Option<Vec<f32>> {
        if token_id >= self.config.vocab_size {
            return None;
        }
        let start = token_id * self.config.embed_dim;
        let end = start + self.config.embed_dim;
        let mut vec = self.weights[start..end].to_vec();
        if let Some(max_norm) = self.config.max_norm {
            renorm_vector(&mut vec, max_norm);
        }
        Some(vec)
    }

    /// Look up embeddings for a batch of token IDs.
    ///
    /// Returns `None` for any out-of-range token.
    pub fn lookup_batch(&self, token_ids: &[usize]) -> Vec<Option<Vec<f32>>> {
        token_ids.iter().map(|&id| self.lookup(id)).collect()
    }

    /// Return the raw weight for a given `(token_id, dim)` coordinate.
    pub fn weight(&self, token_id: usize, dim: usize) -> Option<f32> {
        if token_id >= self.config.vocab_size || dim >= self.config.embed_dim {
            return None;
        }
        Some(self.weights[token_id * self.config.embed_dim + dim])
    }

    /// Return the embedding configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

// ── Positional Embedding ─────────────────────────────────────────────────────

/// Absolute sinusoidal position embeddings (Vaswani et al., 2017).
///
/// For position `pos` and dimension `i`:
/// - `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
/// - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
#[derive(Debug, Clone)]
pub struct PositionalEmbedding {
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Embedding dimension.
    embed_dim: usize,
    /// Pre-computed table: `[max_seq_len, embed_dim]` row-major.
    table: Vec<f32>,
}

impl PositionalEmbedding {
    /// Build the sinusoidal positional embedding table.
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        let mut table = vec![0.0_f32; max_seq_len * embed_dim];
        for pos in 0..max_seq_len {
            for i in 0..embed_dim / 2 {
                let angle = pos as f32 / (10000.0_f32).powf(2.0 * i as f32 / embed_dim as f32);
                table[pos * embed_dim + 2 * i] = angle.sin();
                table[pos * embed_dim + 2 * i + 1] = angle.cos();
            }
            // Handle odd embed_dim: last dimension gets sin only.
            if embed_dim % 2 == 1 {
                let i = embed_dim / 2;
                let angle = pos as f32 / (10000.0_f32).powf(2.0 * i as f32 / embed_dim as f32);
                table[pos * embed_dim + embed_dim - 1] = angle.sin();
            }
        }
        Self { max_seq_len, embed_dim, table }
    }

    /// Get the positional encoding vector for a given position.
    pub fn get(&self, position: usize) -> Option<&[f32]> {
        if position >= self.max_seq_len {
            return None;
        }
        let start = position * self.embed_dim;
        Some(&self.table[start..start + self.embed_dim])
    }

    /// Add positional encoding to a sequence of embeddings in-place.
    ///
    /// `embeddings` is `[seq_len, embed_dim]` row-major; `offset` is the starting
    /// position index (for KV-cache continuation).
    pub fn add_to(&self, embeddings: &mut [f32], seq_len: usize, offset: usize) -> bool {
        if offset + seq_len > self.max_seq_len {
            return false;
        }
        let dim = self.embed_dim;
        for s in 0..seq_len {
            let pos = offset + s;
            let emb_start = s * dim;
            let pe_start = pos * dim;
            for d in 0..dim {
                embeddings[emb_start + d] += self.table[pe_start + d];
            }
        }
        true
    }

    /// Maximum sequence length supported.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ── Rotary Embedding (RoPE) ─────────────────────────────────────────────────

/// Rotary Position Embedding (Su et al., 2021).
///
/// Applies rotation matrices to pairs of dimensions, encoding position
/// information directly into query/key vectors.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Embedding (head) dimension — must be even.
    head_dim: usize,
    /// Maximum sequence length for pre-computed tables.
    max_seq_len: usize,
    /// Base frequency (default 10 000).
    base: f32,
    /// Pre-computed cosine table: `[max_seq_len, head_dim/2]`.
    cos_table: Vec<f32>,
    /// Pre-computed sine table: `[max_seq_len, head_dim/2]`.
    sin_table: Vec<f32>,
}

impl RotaryEmbedding {
    /// Build RoPE tables with the given base frequency.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is odd.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "RoPE head_dim must be even");
        let half = head_dim / 2;
        let mut cos_table = vec![0.0_f32; max_seq_len * half];
        let mut sin_table = vec![0.0_f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table[pos * half + i] = angle.cos();
                sin_table[pos * half + i] = angle.sin();
            }
        }
        Self { head_dim, max_seq_len, base, cos_table, sin_table }
    }

    /// Build RoPE tables with the default base of 10 000.
    pub fn with_default_base(head_dim: usize, max_seq_len: usize) -> Self {
        Self::new(head_dim, max_seq_len, 10000.0)
    }

    /// Apply rotary embedding to a vector of length `head_dim` at position `pos`.
    ///
    /// Returns `None` if `pos >= max_seq_len` or `x.len() != head_dim`.
    pub fn apply(&self, x: &[f32], pos: usize) -> Option<Vec<f32>> {
        if pos >= self.max_seq_len || x.len() != self.head_dim {
            return None;
        }
        let half = self.head_dim / 2;
        let mut out = vec![0.0_f32; self.head_dim];
        let base = pos * half;
        for i in 0..half {
            let cos_val = self.cos_table[base + i];
            let sin_val = self.sin_table[base + i];
            out[2 * i] = x[2 * i] * cos_val - x[2 * i + 1] * sin_val;
            out[2 * i + 1] = x[2 * i] * sin_val + x[2 * i + 1] * cos_val;
        }
        Some(out)
    }

    /// Apply rotary embedding in-place to a `[seq_len, head_dim]` tensor.
    ///
    /// `offset` is the starting position for KV-cache continuation.
    pub fn apply_to_sequence(&self, data: &mut [f32], seq_len: usize, offset: usize) -> bool {
        if data.len() != seq_len * self.head_dim {
            return false;
        }
        if offset + seq_len > self.max_seq_len {
            return false;
        }
        let half = self.head_dim / 2;
        for s in 0..seq_len {
            let pos = offset + s;
            let tbl = pos * half;
            let row = s * self.head_dim;
            for i in 0..half {
                let cos_val = self.cos_table[tbl + i];
                let sin_val = self.sin_table[tbl + i];
                let x0 = data[row + 2 * i];
                let x1 = data[row + 2 * i + 1];
                data[row + 2 * i] = x0 * cos_val - x1 * sin_val;
                data[row + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
        true
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Base frequency.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ── ALiBi Embedding ──────────────────────────────────────────────────────────

/// Attention with Linear Biases (Press et al., 2021).
///
/// Adds a head-specific linear bias `m * (q_pos - k_pos)` to attention scores,
/// replacing explicit positional embeddings.
#[derive(Debug, Clone)]
pub struct ALiBiEmbedding {
    /// Number of attention heads.
    num_heads: usize,
    /// Per-head slope: `m_h = 2^(-8h/H)` where `h ∈ [1..H]`.
    slopes: Vec<f32>,
}

impl ALiBiEmbedding {
    /// Compute ALiBi slopes for the given number of attention heads.
    pub fn new(num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Return the per-head slopes.
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Compute the ALiBi bias matrix for one head.
    ///
    /// Returns a `[query_len, key_len]` row-major bias matrix for head `head_idx`.
    pub fn bias_matrix(
        &self,
        head_idx: usize,
        query_len: usize,
        key_len: usize,
    ) -> Option<Vec<f32>> {
        if head_idx >= self.num_heads {
            return None;
        }
        let slope = self.slopes[head_idx];
        let mut biases = Vec::with_capacity(query_len * key_len);
        for q in 0..query_len {
            for k in 0..key_len {
                // bias = slope * (k - q) — negative for keys before the query.
                let distance = k as f32 - q as f32;
                biases.push(slope * distance);
            }
        }
        Some(biases)
    }

    /// Compute the ALiBi bias for a single (head, query_pos, key_pos) triple.
    pub fn bias(&self, head_idx: usize, query_pos: usize, key_pos: usize) -> Option<f32> {
        if head_idx >= self.num_heads {
            return None;
        }
        let distance = key_pos as f32 - query_pos as f32;
        Some(self.slopes[head_idx] * distance)
    }

    /// Standard ALiBi slope schedule: geometric series `2^(-8/n), 2^(-16/n), …`.
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // Closest power-of-two ≤ num_heads for the geometric series.
        let closest_pow2 = 1_usize << (usize::BITS - 1 - num_heads.leading_zeros() as u32);
        let base = 2.0_f32.powf(-(8.0 / closest_pow2 as f32));
        let mut slopes = Vec::with_capacity(num_heads);
        if num_heads == closest_pow2 {
            for i in 1..=num_heads {
                slopes.push(base.powi(i as i32));
            }
        } else {
            // First half: closest_pow2 slopes with base exponent.
            let extra_base = 2.0_f32.powf(-(8.0 / (2 * closest_pow2) as f32));
            for i in 1..=closest_pow2 {
                slopes.push(base.powi(i as i32));
            }
            // Remaining slots: interleaved from the finer-grained series.
            for i in 1..=(num_heads - closest_pow2) {
                slopes.push(extra_base.powi((2 * i) as i32));
            }
        }
        slopes
    }
}

// ── Embedding Quantizer ──────────────────────────────────────────────────────

/// Quantization format for embedding tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// 8-bit signed integer quantization.
    Int8,
    /// 4-bit signed integer quantization (packed: two values per byte).
    Int4,
}

/// Quantized embedding table for memory-efficient storage.
///
/// Stores per-row scale factors so that `float_value ≈ quantized_value * scale`.
#[derive(Debug, Clone)]
pub struct EmbeddingQuantizer {
    format: QuantFormat,
    vocab_size: usize,
    embed_dim: usize,
    /// Quantized weights — `i8` for INT8; packed nibbles for INT4.
    data: Vec<i8>,
    /// Per-row scale factors.
    scales: Vec<f32>,
}

impl EmbeddingQuantizer {
    /// Quantize a float embedding table.
    ///
    /// `weights` is row-major `[vocab_size, embed_dim]`.
    pub fn quantize(config: &EmbeddingConfig, weights: &[f32], format: QuantFormat) -> Self {
        assert_eq!(
            weights.len(),
            config.vocab_size * config.embed_dim,
            "weight table size mismatch"
        );
        let vocab = config.vocab_size;
        let dim = config.embed_dim;
        let mut scales = Vec::with_capacity(vocab);
        let data = match format {
            QuantFormat::Int8 => {
                let mut buf = Vec::with_capacity(vocab * dim);
                for row in 0..vocab {
                    let slice = &weights[row * dim..(row + 1) * dim];
                    let abs_max = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
                    scales.push(scale);
                    for &v in slice {
                        buf.push((v / scale).round().clamp(-127.0, 127.0) as i8);
                    }
                }
                buf
            }
            QuantFormat::Int4 => {
                // Pack two 4-bit values per byte: low nibble first.
                let packed_dim = (dim + 1) / 2;
                let mut buf = Vec::with_capacity(vocab * packed_dim);
                for row in 0..vocab {
                    let slice = &weights[row * dim..(row + 1) * dim];
                    let abs_max = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
                    scales.push(scale);
                    for pair in (0..dim).step_by(2) {
                        let lo = (slice[pair] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F;
                        let hi = if pair + 1 < dim {
                            ((slice[pair + 1] / scale).round().clamp(-7.0, 7.0) as i8 & 0x0F) << 4
                        } else {
                            0
                        };
                        buf.push(lo | hi);
                    }
                }
                buf
            }
        };
        Self { format, vocab_size: vocab, embed_dim: dim, data, scales }
    }

    /// Dequantize one embedding vector.
    pub fn dequantize(&self, token_id: usize) -> Option<Vec<f32>> {
        if token_id >= self.vocab_size {
            return None;
        }
        let scale = self.scales[token_id];
        let dim = self.embed_dim;
        match self.format {
            QuantFormat::Int8 => {
                let start = token_id * dim;
                let out: Vec<f32> =
                    self.data[start..start + dim].iter().map(|&q| q as f32 * scale).collect();
                Some(out)
            }
            QuantFormat::Int4 => {
                let packed_dim = (dim + 1) / 2;
                let start = token_id * packed_dim;
                let mut out = Vec::with_capacity(dim);
                for j in 0..packed_dim {
                    let byte = self.data[start + j];
                    // Unpack low nibble with sign extension.
                    let lo = sign_extend_4bit(byte & 0x0F);
                    out.push(lo as f32 * scale);
                    if out.len() < dim {
                        let hi = sign_extend_4bit((byte >> 4) & 0x0F);
                        out.push(hi as f32 * scale);
                    }
                }
                Some(out)
            }
        }
    }

    /// Format used.
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// Memory used by the quantized data in bytes (excluding scales).
    pub fn data_bytes(&self) -> usize {
        self.data.len()
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embed dim.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

/// Sign-extend a 4-bit two's complement value stored in the low 4 bits.
fn sign_extend_4bit(nibble: i8) -> i8 {
    if nibble & 0x08 != 0 { nibble | !0x0F } else { nibble & 0x0F }
}

// ── Embedding Aggregator ─────────────────────────────────────────────────────

/// Strategy for aggregating a sequence of embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Element-wise mean.
    Mean,
    /// Element-wise sum.
    Sum,
    /// Weighted mean (requires a weight per token).
    Weighted,
    /// Take the first token's embedding (CLS pooling).
    First,
    /// Take the last token's embedding.
    Last,
    /// Element-wise max across positions.
    Max,
}

/// Aggregates multiple token embeddings into a single vector.
#[derive(Debug, Clone)]
pub struct EmbeddingAggregator {
    strategy: AggregationStrategy,
}

impl EmbeddingAggregator {
    /// Create an aggregator with the specified strategy.
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self { strategy }
    }

    /// Aggregate a sequence of embedding vectors.
    ///
    /// `embeddings` is `[seq_len, embed_dim]` row-major.
    /// `weights` is only required for `Weighted` strategy.
    pub fn aggregate(
        &self,
        embeddings: &[f32],
        seq_len: usize,
        embed_dim: usize,
        weights: Option<&[f32]>,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || embeddings.len() != seq_len * embed_dim {
            return None;
        }
        match self.strategy {
            AggregationStrategy::Mean => {
                let mut acc = vec![0.0_f32; embed_dim];
                for s in 0..seq_len {
                    let row = &embeddings[s * embed_dim..(s + 1) * embed_dim];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a += v;
                    }
                }
                let n = seq_len as f32;
                acc.iter_mut().for_each(|v| *v /= n);
                Some(acc)
            }
            AggregationStrategy::Sum => {
                let mut acc = vec![0.0_f32; embed_dim];
                for s in 0..seq_len {
                    let row = &embeddings[s * embed_dim..(s + 1) * embed_dim];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a += v;
                    }
                }
                Some(acc)
            }
            AggregationStrategy::Weighted => {
                let w = weights?;
                if w.len() != seq_len {
                    return None;
                }
                let total: f32 = w.iter().sum();
                if total == 0.0 {
                    return Some(vec![0.0; embed_dim]);
                }
                let mut acc = vec![0.0_f32; embed_dim];
                for s in 0..seq_len {
                    let row = &embeddings[s * embed_dim..(s + 1) * embed_dim];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a += v * w[s];
                    }
                }
                acc.iter_mut().for_each(|v| *v /= total);
                Some(acc)
            }
            AggregationStrategy::First => Some(embeddings[..embed_dim].to_vec()),
            AggregationStrategy::Last => {
                let start = (seq_len - 1) * embed_dim;
                Some(embeddings[start..start + embed_dim].to_vec())
            }
            AggregationStrategy::Max => {
                let mut acc = vec![f32::NEG_INFINITY; embed_dim];
                for s in 0..seq_len {
                    let row = &embeddings[s * embed_dim..(s + 1) * embed_dim];
                    for (a, &v) in acc.iter_mut().zip(row) {
                        *a = a.max(v);
                    }
                }
                Some(acc)
            }
        }
    }

    /// Strategy in use.
    pub fn strategy(&self) -> AggregationStrategy {
        self.strategy
    }
}

// ── Similarity Search ────────────────────────────────────────────────────────

/// Distance / similarity metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (normalized dot product).
    Cosine,
    /// Raw dot product.
    DotProduct,
    /// Euclidean distance (lower is more similar).
    Euclidean,
}

/// Nearest-neighbour search in embedding space.
#[derive(Debug, Clone)]
pub struct SimilaritySearch {
    metric: SimilarityMetric,
    /// Row-major embedding index: `[num_vectors, dim]`.
    index: Vec<f32>,
    dim: usize,
    num_vectors: usize,
}

impl SimilaritySearch {
    /// Build an index from row-major embeddings.
    ///
    /// # Panics
    ///
    /// Panics if `embeddings.len() % dim != 0`.
    pub fn new(metric: SimilarityMetric, embeddings: Vec<f32>, dim: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert_eq!(embeddings.len() % dim, 0, "embedding length must be a multiple of dim");
        let num_vectors = embeddings.len() / dim;
        Self { metric, index: embeddings, dim, num_vectors }
    }

    /// Search for the `k` most similar vectors to `query`.
    ///
    /// Returns `(index, score)` pairs sorted by descending similarity
    /// (or ascending distance for Euclidean).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let mut scores: Vec<(usize, f32)> = (0..self.num_vectors)
            .map(|i| {
                let row = &self.index[i * self.dim..(i + 1) * self.dim];
                let score = match self.metric {
                    SimilarityMetric::Cosine => cosine_similarity(query, row),
                    SimilarityMetric::DotProduct => dot_product(query, row),
                    SimilarityMetric::Euclidean => {
                        // Negate so that larger = more similar for sorting.
                        -euclidean_distance(query, row)
                    }
                };
                (i, score)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        // For Euclidean, restore positive distance.
        if self.metric == SimilarityMetric::Euclidean {
            for s in &mut scores {
                s.1 = -s.1;
            }
        }
        scores
    }

    /// Compute similarity between two vectors using the configured metric.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != self.dim || b.len() != self.dim {
            return None;
        }
        Some(match self.metric {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::DotProduct => dot_product(a, b),
            SimilarityMetric::Euclidean => euclidean_distance(a, b),
        })
    }

    /// Number of indexed vectors.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Dimension of each vector.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ── Embedding Normalizer ─────────────────────────────────────────────────────

/// Normalization mode for embedding vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// L2 (unit-sphere) normalization.
    L2,
    /// Layer normalization (zero-mean, unit-variance).
    LayerNorm,
}

/// Normalizes embedding vectors.
#[derive(Debug, Clone)]
pub struct EmbeddingNormalizer {
    mode: NormMode,
    /// Small constant for numerical stability.
    eps: f32,
}

impl EmbeddingNormalizer {
    /// Create a normalizer with the given mode and epsilon.
    pub fn new(mode: NormMode, eps: f32) -> Self {
        Self { mode, eps }
    }

    /// Create an L2 normalizer with default epsilon.
    pub fn l2() -> Self {
        Self::new(NormMode::L2, 1e-12)
    }

    /// Create a layer-norm normalizer with default epsilon.
    pub fn layer_norm() -> Self {
        Self::new(NormMode::LayerNorm, 1e-5)
    }

    /// Normalize a single embedding vector in-place.
    pub fn normalize(&self, vec: &mut [f32]) {
        match self.mode {
            NormMode::L2 => {
                let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(self.eps);
                vec.iter_mut().for_each(|v| *v /= norm);
            }
            NormMode::LayerNorm => {
                let n = vec.len() as f32;
                let mean = vec.iter().sum::<f32>() / n;
                let var = vec.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
                let std = (var + self.eps).sqrt();
                vec.iter_mut().for_each(|v| *v = (*v - mean) / std);
            }
        }
    }

    /// Normalize a batch of vectors `[n, dim]` row-major in-place.
    pub fn normalize_batch(&self, data: &mut [f32], n: usize, dim: usize) {
        assert_eq!(data.len(), n * dim, "batch size mismatch");
        for i in 0..n {
            self.normalize(&mut data[i * dim..(i + 1) * dim]);
        }
    }

    /// Return a normalized copy of the input vector.
    pub fn normalized(&self, vec: &[f32]) -> Vec<f32> {
        let mut out = vec.to_vec();
        self.normalize(&mut out);
        out
    }

    /// Normalization mode.
    pub fn mode(&self) -> NormMode {
        self.mode
    }
}

// ── Embedding Engine ─────────────────────────────────────────────────────────

/// Position encoding variant used by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionEncodingKind {
    /// Absolute sinusoidal.
    Sinusoidal,
    /// Rotary (RoPE).
    Rotary,
    /// ALiBi (no additive position encoding; bias is applied at attention).
    ALiBi,
    /// No position encoding.
    None,
}

/// Unified embedding pipeline.
///
/// Combines token lookup, optional positional encoding, optional normalization,
/// and optional aggregation into a single forward pass.
#[derive(Debug, Clone)]
pub struct EmbeddingEngine {
    token_embedding: TokenEmbedding,
    positional: Option<PositionalEmbedding>,
    normalizer: Option<EmbeddingNormalizer>,
    position_kind: PositionEncodingKind,
}

impl EmbeddingEngine {
    /// Create an engine with token embeddings only.
    pub fn new(token_embedding: TokenEmbedding) -> Self {
        Self {
            token_embedding,
            positional: None,
            normalizer: None,
            position_kind: PositionEncodingKind::None,
        }
    }

    /// Add sinusoidal positional encoding.
    pub fn with_sinusoidal(mut self, max_seq_len: usize) -> Self {
        let dim = self.token_embedding.config().embed_dim;
        self.positional = Some(PositionalEmbedding::new(max_seq_len, dim));
        self.position_kind = PositionEncodingKind::Sinusoidal;
        self
    }

    /// Add L2 normalization.
    pub fn with_l2_norm(mut self) -> Self {
        self.normalizer = Some(EmbeddingNormalizer::l2());
        self
    }

    /// Add layer normalization.
    pub fn with_layer_norm(mut self) -> Self {
        self.normalizer = Some(EmbeddingNormalizer::layer_norm());
        self
    }

    /// Run the embedding pipeline for a sequence of token IDs.
    ///
    /// Returns `[seq_len, embed_dim]` row-major or `None` if any token is OOV.
    pub fn forward(&self, token_ids: &[usize], position_offset: usize) -> Option<Vec<f32>> {
        let dim = self.token_embedding.config().embed_dim;
        let seq_len = token_ids.len();
        let mut output = Vec::with_capacity(seq_len * dim);
        for &tid in token_ids {
            output.extend(self.token_embedding.lookup(tid)?);
        }
        // Add sinusoidal positions if configured.
        if let Some(ref pe) = self.positional {
            if !pe.add_to(&mut output, seq_len, position_offset) {
                return None;
            }
        }
        // Normalize each vector.
        if let Some(ref norm) = self.normalizer {
            norm.normalize_batch(&mut output, seq_len, dim);
        }
        Some(output)
    }

    /// Position encoding kind.
    pub fn position_kind(&self) -> PositionEncodingKind {
        self.position_kind
    }

    /// Reference to the underlying token embedding.
    pub fn token_embedding(&self) -> &TokenEmbedding {
        &self.token_embedding
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Dot product of two equal-length vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Euclidean distance.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Renormalize a vector so its L2 norm is at most `max_norm`.
fn renorm_vector(v: &mut [f32], max_norm: f32) {
    let norm = l2_norm(v);
    if norm > max_norm {
        let scale = max_norm / norm;
        v.iter_mut().for_each(|x| *x *= scale);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    // ── EmbeddingConfig ──────────────────────────────────────────────────

    #[test]
    fn config_basic() {
        let cfg = EmbeddingConfig::new(1000, 64);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.embed_dim, 64);
        assert!(cfg.padding_idx.is_none());
        assert!(cfg.max_norm.is_none());
    }

    #[test]
    fn config_with_padding() {
        let cfg = EmbeddingConfig::new(1000, 64).with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    #[test]
    fn config_with_max_norm() {
        let cfg = EmbeddingConfig::new(1000, 64).with_max_norm(1.0);
        assert_eq!(cfg.max_norm, Some(1.0));
    }

    #[test]
    fn config_builder_chain() {
        let cfg = EmbeddingConfig::new(500, 32).with_padding_idx(1).with_max_norm(2.5);
        assert_eq!(cfg.vocab_size, 500);
        assert_eq!(cfg.embed_dim, 32);
        assert_eq!(cfg.padding_idx, Some(1));
        assert_eq!(cfg.max_norm, Some(2.5));
    }

    #[test]
    fn config_clone_debug() {
        let cfg = EmbeddingConfig::new(10, 4);
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.vocab_size, 10);
        let _ = format!("{cfg:?}");
    }

    // ── TokenEmbedding ───────────────────────────────────────────────────

    #[test]
    fn token_embedding_lookup_valid() {
        let cfg = EmbeddingConfig::new(10, 4);
        let emb = TokenEmbedding::new(cfg);
        let v = emb.lookup(0).unwrap();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn token_embedding_lookup_out_of_range() {
        let cfg = EmbeddingConfig::new(10, 4);
        let emb = TokenEmbedding::new(cfg);
        assert!(emb.lookup(10).is_none());
        assert!(emb.lookup(100).is_none());
    }

    #[test]
    fn token_embedding_lookup_deterministic() {
        let cfg = EmbeddingConfig::new(10, 4);
        let emb1 = TokenEmbedding::new(cfg.clone());
        let emb2 = TokenEmbedding::new(cfg);
        assert_eq!(emb1.lookup(3), emb2.lookup(3));
    }

    #[test]
    fn token_embedding_padding_zeroed() {
        let cfg = EmbeddingConfig::new(10, 4).with_padding_idx(0);
        let emb = TokenEmbedding::new(cfg);
        let v = emb.lookup(0).unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn token_embedding_from_weights() {
        let cfg = EmbeddingConfig::new(2, 3);
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        assert_eq!(emb.lookup(0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.lookup(1).unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "weight table size mismatch")]
    fn token_embedding_from_weights_wrong_size() {
        let cfg = EmbeddingConfig::new(2, 3);
        TokenEmbedding::from_weights(cfg, vec![1.0, 2.0]);
    }

    #[test]
    fn token_embedding_max_norm() {
        let cfg = EmbeddingConfig::new(1, 3).with_max_norm(1.0);
        let weights = vec![3.0, 4.0, 0.0]; // norm = 5
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let v = emb.lookup(0).unwrap();
        let norm = l2_norm(&v);
        assert!(approx_eq(norm, 1.0, 1e-5));
    }

    #[test]
    fn token_embedding_max_norm_no_change_when_within() {
        let cfg = EmbeddingConfig::new(1, 3).with_max_norm(10.0);
        let weights = vec![1.0, 0.0, 0.0]; // norm = 1
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let v = emb.lookup(0).unwrap();
        assert_eq!(v, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn token_embedding_batch_lookup() {
        let cfg = EmbeddingConfig::new(3, 2);
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let results = emb.lookup_batch(&[0, 2, 5]);
        assert_eq!(results[0].as_ref().unwrap(), &[1.0, 2.0]);
        assert_eq!(results[1].as_ref().unwrap(), &[5.0, 6.0]);
        assert!(results[2].is_none());
    }

    #[test]
    fn token_embedding_weight_accessor() {
        let cfg = EmbeddingConfig::new(2, 3);
        let weights = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        assert_eq!(emb.weight(0, 0), Some(10.0));
        assert_eq!(emb.weight(1, 2), Some(60.0));
        assert_eq!(emb.weight(2, 0), None);
        assert_eq!(emb.weight(0, 3), None);
    }

    #[test]
    fn token_embedding_different_ids_differ() {
        let cfg = EmbeddingConfig::new(10, 8);
        let emb = TokenEmbedding::new(cfg);
        let v0 = emb.lookup(0).unwrap();
        let v1 = emb.lookup(1).unwrap();
        assert_ne!(v0, v1);
    }

    #[test]
    fn token_embedding_config_accessor() {
        let cfg = EmbeddingConfig::new(42, 16);
        let emb = TokenEmbedding::new(cfg);
        assert_eq!(emb.config().vocab_size, 42);
        assert_eq!(emb.config().embed_dim, 16);
    }

    // ── PositionalEmbedding ──────────────────────────────────────────────

    #[test]
    fn positional_embedding_position_zero() {
        let pe = PositionalEmbedding::new(100, 4);
        let v = pe.get(0).unwrap();
        // sin(0) = 0, cos(0) = 1 for the first pair.
        assert!(approx_eq(v[0], 0.0, EPS));
        assert!(approx_eq(v[1], 1.0, EPS));
    }

    #[test]
    fn positional_embedding_out_of_range() {
        let pe = PositionalEmbedding::new(10, 4);
        assert!(pe.get(10).is_none());
    }

    #[test]
    fn positional_embedding_different_positions_differ() {
        let pe = PositionalEmbedding::new(100, 8);
        let v0 = pe.get(0).unwrap();
        let v1 = pe.get(1).unwrap();
        assert_ne!(v0, v1);
    }

    #[test]
    fn positional_embedding_values_bounded() {
        let pe = PositionalEmbedding::new(100, 16);
        for pos in 0..100 {
            let v = pe.get(pos).unwrap();
            for &x in v {
                assert!((-1.0..=1.0).contains(&x), "PE value {x} out of [-1,1]");
            }
        }
    }

    #[test]
    fn positional_embedding_add_to_sequence() {
        let pe = PositionalEmbedding::new(10, 4);
        let mut embs = vec![0.0_f32; 3 * 4]; // 3 positions, dim 4
        assert!(pe.add_to(&mut embs, 3, 0));
        // After adding to zeros, should equal the PE itself.
        for s in 0..3 {
            let pe_vec = pe.get(s).unwrap();
            let emb_row = &embs[s * 4..(s + 1) * 4];
            assert!(vec_approx_eq(emb_row, pe_vec, EPS));
        }
    }

    #[test]
    fn positional_embedding_add_to_with_offset() {
        let pe = PositionalEmbedding::new(10, 4);
        let mut embs = vec![0.0_f32; 2 * 4];
        assert!(pe.add_to(&mut embs, 2, 5));
        let pe5 = pe.get(5).unwrap();
        assert!(vec_approx_eq(&embs[0..4], pe5, EPS));
    }

    #[test]
    fn positional_embedding_add_to_overflow() {
        let pe = PositionalEmbedding::new(5, 4);
        let mut embs = vec![0.0; 3 * 4];
        assert!(!pe.add_to(&mut embs, 3, 4)); // 4+3=7 > 5
    }

    #[test]
    fn positional_embedding_odd_dim() {
        let pe = PositionalEmbedding::new(10, 5);
        let v = pe.get(1).unwrap();
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn positional_embedding_accessors() {
        let pe = PositionalEmbedding::new(64, 32);
        assert_eq!(pe.max_seq_len(), 64);
        assert_eq!(pe.embed_dim(), 32);
    }

    // ── RotaryEmbedding ──────────────────────────────────────────────────

    #[test]
    fn rope_position_zero_is_identity() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let out = rope.apply(&x, 0).unwrap();
        // At pos=0, angles are 0 → cos=1, sin=0 → identity.
        assert!(vec_approx_eq(&out, &x, EPS));
    }

    #[test]
    fn rope_preserves_norm() {
        let rope = RotaryEmbedding::with_default_base(8, 100);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_norm = l2_norm(&x);
        for pos in 0..50 {
            let out = rope.apply(&x, pos).unwrap();
            let out_norm = l2_norm(&out);
            assert!(
                approx_eq(x_norm, out_norm, 1e-4),
                "norm changed at pos {pos}: {x_norm} vs {out_norm}"
            );
        }
    }

    #[test]
    fn rope_wrong_dim() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        assert!(rope.apply(&[1.0, 2.0, 3.0], 0).is_none());
    }

    #[test]
    fn rope_out_of_range_pos() {
        let rope = RotaryEmbedding::with_default_base(4, 10);
        let x = vec![1.0; 4];
        assert!(rope.apply(&x, 10).is_none());
    }

    #[test]
    #[should_panic(expected = "RoPE head_dim must be even")]
    fn rope_odd_dim_panics() {
        RotaryEmbedding::with_default_base(3, 10);
    }

    #[test]
    fn rope_different_positions_differ() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let a = rope.apply(&x, 1).unwrap();
        let b = rope.apply(&x, 2).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn rope_apply_to_sequence() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let x = vec![1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut data = x.clone();
        assert!(rope.apply_to_sequence(&mut data, 2, 0));
        // First vector at pos=0 should be unchanged.
        let expected0 = rope.apply(&x[0..4], 0).unwrap();
        assert!(vec_approx_eq(&data[0..4], &expected0, EPS));
        // Second vector at pos=1.
        let expected1 = rope.apply(&x[4..8], 1).unwrap();
        assert!(vec_approx_eq(&data[4..8], &expected1, EPS));
    }

    #[test]
    fn rope_apply_to_sequence_with_offset() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = x.clone();
        assert!(rope.apply_to_sequence(&mut data, 1, 5));
        let expected = rope.apply(&x, 5).unwrap();
        assert!(vec_approx_eq(&data, &expected, EPS));
    }

    #[test]
    fn rope_sequence_wrong_len() {
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let mut data = vec![1.0; 5]; // not a multiple of 4
        assert!(!rope.apply_to_sequence(&mut data, 1, 0));
    }

    #[test]
    fn rope_sequence_overflow() {
        let rope = RotaryEmbedding::with_default_base(4, 5);
        let mut data = vec![1.0; 4 * 3]; // 3 vectors
        assert!(!rope.apply_to_sequence(&mut data, 3, 4)); // 4+3=7 > 5
    }

    #[test]
    fn rope_custom_base() {
        let rope = RotaryEmbedding::new(4, 100, 500.0);
        assert_eq!(rope.base(), 500.0);
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let out = rope.apply(&x, 1).unwrap();
        // With lower base, rotation is faster.
        assert_ne!(out, x);
    }

    #[test]
    fn rope_accessors() {
        let rope = RotaryEmbedding::with_default_base(8, 200);
        assert_eq!(rope.head_dim(), 8);
        assert_eq!(rope.max_seq_len(), 200);
        assert_eq!(rope.base(), 10000.0);
    }

    #[test]
    fn rope_relative_position_property() {
        // RoPE encodes relative position: dot(R(x,p1), R(y,p2)) depends on p1-p2.
        let rope = RotaryEmbedding::with_default_base(4, 100);
        let x = vec![1.0, 0.5, 0.3, 0.7];
        let y = vec![0.2, 0.8, 0.4, 0.6];

        let rx_5 = rope.apply(&x, 5).unwrap();
        let ry_8 = rope.apply(&y, 8).unwrap();
        let dot1 = dot_product(&rx_5, &ry_8);

        let rx_10 = rope.apply(&x, 10).unwrap();
        let ry_13 = rope.apply(&y, 13).unwrap();
        let dot2 = dot_product(&rx_10, &ry_13);

        // Same relative distance (3), so dots should be equal.
        assert!(approx_eq(dot1, dot2, 1e-4), "dot1={dot1} dot2={dot2}");
    }

    // ── ALiBiEmbedding ───────────────────────────────────────────────────

    #[test]
    fn alibi_slopes_power_of_two_heads() {
        let alibi = ALiBiEmbedding::new(8);
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 8);
        // First slope should be 2^(-8/8) = 2^(-1) = 0.5.
        assert!(approx_eq(slopes[0], 0.5, EPS));
        // Slopes should be strictly decreasing.
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {i}");
        }
    }

    #[test]
    fn alibi_slopes_non_power_of_two() {
        let alibi = ALiBiEmbedding::new(6);
        assert_eq!(alibi.slopes().len(), 6);
    }

    #[test]
    fn alibi_bias_same_position() {
        let alibi = ALiBiEmbedding::new(4);
        let bias = alibi.bias(0, 5, 5).unwrap();
        // Same position → distance = 0 → bias = 0.
        assert!(approx_eq(bias, 0.0, EPS));
    }

    #[test]
    fn alibi_bias_linearity() {
        let alibi = ALiBiEmbedding::new(4);
        let b1 = alibi.bias(0, 0, 1).unwrap();
        let b2 = alibi.bias(0, 0, 2).unwrap();
        // Linear: b2 should be 2 * b1.
        assert!(approx_eq(b2, 2.0 * b1, EPS));
    }

    #[test]
    fn alibi_bias_matrix_shape() {
        let alibi = ALiBiEmbedding::new(4);
        let mat = alibi.bias_matrix(0, 3, 5).unwrap();
        assert_eq!(mat.len(), 3 * 5);
    }

    #[test]
    fn alibi_bias_matrix_diagonal() {
        let alibi = ALiBiEmbedding::new(4);
        let mat = alibi.bias_matrix(0, 4, 4).unwrap();
        // Diagonal entries (q==k) should be 0.
        for i in 0..4 {
            assert!(approx_eq(mat[i * 4 + i], 0.0, EPS));
        }
    }

    #[test]
    fn alibi_bias_out_of_range_head() {
        let alibi = ALiBiEmbedding::new(4);
        assert!(alibi.bias(4, 0, 0).is_none());
        assert!(alibi.bias_matrix(4, 1, 1).is_none());
    }

    #[test]
    fn alibi_different_heads_different_slopes() {
        let alibi = ALiBiEmbedding::new(8);
        let b0 = alibi.bias(0, 0, 5).unwrap();
        let b1 = alibi.bias(1, 0, 5).unwrap();
        assert_ne!(b0, b1);
    }

    #[test]
    fn alibi_accessors() {
        let alibi = ALiBiEmbedding::new(12);
        assert_eq!(alibi.num_heads(), 12);
    }

    #[test]
    fn alibi_single_head() {
        let alibi = ALiBiEmbedding::new(1);
        assert_eq!(alibi.slopes().len(), 1);
        let b = alibi.bias(0, 0, 3).unwrap();
        assert!(b != 0.0);
    }

    // ── EmbeddingQuantizer ───────────────────────────────────────────────

    #[test]
    fn quantize_int8_roundtrip() {
        let cfg = EmbeddingConfig::new(3, 4);
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.0, 0.5];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int8);
        for token in 0..3 {
            let deq = q.dequantize(token).unwrap();
            let orig = &weights[token * 4..(token + 1) * 4];
            for (a, b) in orig.iter().zip(deq.iter()) {
                assert!(approx_eq(*a, *b, 0.02), "INT8 roundtrip: {a} vs {b}");
            }
        }
    }

    #[test]
    fn quantize_int4_roundtrip() {
        let cfg = EmbeddingConfig::new(2, 4);
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.5, 0.0, 0.3];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int4);
        for token in 0..2 {
            let deq = q.dequantize(token).unwrap();
            let orig = &weights[token * 4..(token + 1) * 4];
            for (a, b) in orig.iter().zip(deq.iter()) {
                assert!(approx_eq(*a, *b, 0.15), "INT4 roundtrip: {a} vs {b}");
            }
        }
    }

    #[test]
    fn quantize_zero_vector() {
        let cfg = EmbeddingConfig::new(1, 4);
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int8);
        let deq = q.dequantize(0).unwrap();
        assert!(deq.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn quantize_out_of_range() {
        let cfg = EmbeddingConfig::new(2, 4);
        let weights = vec![0.0; 8];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int8);
        assert!(q.dequantize(2).is_none());
    }

    #[test]
    fn quantize_int8_memory_savings() {
        let cfg = EmbeddingConfig::new(1000, 64);
        let weights = vec![0.1; 1000 * 64];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int8);
        let float_bytes = 1000 * 64 * 4;
        assert!(q.data_bytes() < float_bytes);
    }

    #[test]
    fn quantize_int4_memory_savings() {
        let cfg = EmbeddingConfig::new(1000, 64);
        let weights = vec![0.1; 1000 * 64];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int4);
        let int8_bytes = 1000 * 64;
        assert!(q.data_bytes() < int8_bytes);
    }

    #[test]
    fn quantize_format_accessor() {
        let cfg = EmbeddingConfig::new(1, 4);
        let w = vec![0.0; 4];
        let q8 = EmbeddingQuantizer::quantize(&cfg, &w, QuantFormat::Int8);
        let q4 = EmbeddingQuantizer::quantize(&cfg, &w, QuantFormat::Int4);
        assert_eq!(q8.format(), QuantFormat::Int8);
        assert_eq!(q4.format(), QuantFormat::Int4);
    }

    #[test]
    fn quantize_dim_accessors() {
        let cfg = EmbeddingConfig::new(5, 8);
        let w = vec![0.0; 40];
        let q = EmbeddingQuantizer::quantize(&cfg, &w, QuantFormat::Int8);
        assert_eq!(q.vocab_size(), 5);
        assert_eq!(q.embed_dim(), 8);
    }

    #[test]
    fn quantize_int4_odd_dim() {
        let cfg = EmbeddingConfig::new(1, 5);
        let weights = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int4);
        let deq = q.dequantize(0).unwrap();
        assert_eq!(deq.len(), 5);
    }

    #[test]
    fn quantize_int8_preserves_sign() {
        let cfg = EmbeddingConfig::new(1, 4);
        let weights = vec![-0.5, 0.5, -1.0, 1.0];
        let q = EmbeddingQuantizer::quantize(&cfg, &weights, QuantFormat::Int8);
        let deq = q.dequantize(0).unwrap();
        assert!(deq[0] < 0.0 && deq[1] > 0.0 && deq[2] < 0.0 && deq[3] > 0.0);
    }

    // ── EmbeddingAggregator ──────────────────────────────────────────────

    #[test]
    fn aggregator_mean() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Mean);
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 vectors of dim 2
        let result = agg.aggregate(&data, 2, 2, None).unwrap();
        assert!(vec_approx_eq(&result, &[2.0, 3.0], EPS));
    }

    #[test]
    fn aggregator_sum() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Sum);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = agg.aggregate(&data, 2, 2, None).unwrap();
        assert!(vec_approx_eq(&result, &[4.0, 6.0], EPS));
    }

    #[test]
    fn aggregator_weighted() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Weighted);
        let data = vec![1.0, 0.0, 0.0, 1.0]; // 2 vectors
        let weights = vec![1.0, 3.0];
        let result = agg.aggregate(&data, 2, 2, Some(&weights)).unwrap();
        // Weighted mean: (1*1 + 0*3)/4, (0*1 + 1*3)/4 = [0.25, 0.75]
        assert!(vec_approx_eq(&result, &[0.25, 0.75], EPS));
    }

    #[test]
    fn aggregator_weighted_no_weights() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Weighted);
        let data = vec![1.0, 2.0];
        assert!(agg.aggregate(&data, 1, 2, None).is_none());
    }

    #[test]
    fn aggregator_first() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::First);
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let result = agg.aggregate(&data, 2, 2, None).unwrap();
        assert_eq!(result, vec![10.0, 20.0]);
    }

    #[test]
    fn aggregator_last() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Last);
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let result = agg.aggregate(&data, 2, 2, None).unwrap();
        assert_eq!(result, vec![30.0, 40.0]);
    }

    #[test]
    fn aggregator_max() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Max);
        let data = vec![1.0, 4.0, 3.0, 2.0]; // 2 vectors of dim 2
        let result = agg.aggregate(&data, 2, 2, None).unwrap();
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn aggregator_empty_sequence() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Mean);
        assert!(agg.aggregate(&[], 0, 2, None).is_none());
    }

    #[test]
    fn aggregator_wrong_size() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Mean);
        assert!(agg.aggregate(&[1.0, 2.0, 3.0], 2, 2, None).is_none());
    }

    #[test]
    fn aggregator_strategy_accessor() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Sum);
        assert_eq!(agg.strategy(), AggregationStrategy::Sum);
    }

    #[test]
    fn aggregator_single_vector_mean() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Mean);
        let data = vec![5.0, 10.0, 15.0];
        let result = agg.aggregate(&data, 1, 3, None).unwrap();
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn aggregator_weighted_zero_weights() {
        let agg = EmbeddingAggregator::new(AggregationStrategy::Weighted);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, 0.0];
        let result = agg.aggregate(&data, 2, 2, Some(&weights)).unwrap();
        assert!(result.iter().all(|&x| x == 0.0));
    }

    // ── SimilaritySearch ─────────────────────────────────────────────────

    #[test]
    fn similarity_cosine_identical() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![1.0, 0.0, 0.0, 1.0], 2);
        let sim = idx.similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!(approx_eq(sim, 1.0, EPS));
    }

    #[test]
    fn similarity_cosine_orthogonal() {
        let _idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![0.0; 0], 2);
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(approx_eq(sim, 0.0, EPS));
    }

    #[test]
    fn similarity_cosine_opposite() {
        let sim = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!(approx_eq(sim, -1.0, EPS));
    }

    #[test]
    fn similarity_dot_product() {
        let idx = SimilaritySearch::new(SimilarityMetric::DotProduct, vec![1.0, 2.0, 3.0, 4.0], 2);
        let sim = idx.similarity(&[1.0, 1.0], &[3.0, 4.0]).unwrap();
        assert!(approx_eq(sim, 7.0, EPS));
    }

    #[test]
    fn similarity_euclidean() {
        let dist = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!(approx_eq(dist, 5.0, EPS));
    }

    #[test]
    fn similarity_search_top_k() {
        let idx = SimilaritySearch::new(
            SimilarityMetric::Cosine,
            vec![
                1.0, 0.0, // vec 0
                0.0, 1.0, // vec 1
                0.7, 0.7, // vec 2 (closest to [1,1])
            ],
            2,
        );
        let results = idx.search(&[1.0, 1.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2); // vec 2 is most similar to [1,1]
    }

    #[test]
    fn similarity_search_euclidean_ordering() {
        let idx = SimilaritySearch::new(
            SimilarityMetric::Euclidean,
            vec![
                0.0, 0.0, // vec 0
                10.0, 10.0, // vec 1
                1.0, 1.0, // vec 2
            ],
            2,
        );
        let results = idx.search(&[0.0, 0.0], 3);
        // Closest first (smallest distance).
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 2);
        assert_eq!(results[2].0, 1);
    }

    #[test]
    fn similarity_search_k_larger_than_index() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![1.0, 0.0, 0.0, 1.0], 2);
        let results = idx.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn similarity_search_empty_query() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![1.0, 0.0], 2);
        assert!(idx.search(&[1.0], 1).is_empty()); // wrong dim
    }

    #[test]
    fn similarity_search_zero_k() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![1.0, 0.0], 2);
        assert!(idx.search(&[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn similarity_wrong_dim() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![1.0, 0.0], 2);
        assert!(idx.similarity(&[1.0], &[1.0, 0.0]).is_none());
    }

    #[test]
    fn similarity_accessors() {
        let idx = SimilaritySearch::new(SimilarityMetric::Cosine, vec![0.0; 6], 3);
        assert_eq!(idx.num_vectors(), 2);
        assert_eq!(idx.dim(), 3);
    }

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn similarity_zero_dim_panics() {
        SimilaritySearch::new(SimilarityMetric::Cosine, vec![], 0);
    }

    // ── EmbeddingNormalizer ──────────────────────────────────────────────

    #[test]
    fn normalizer_l2_unit_norm() {
        let norm = EmbeddingNormalizer::l2();
        let mut v = vec![3.0, 4.0];
        norm.normalize(&mut v);
        let n = l2_norm(&v);
        assert!(approx_eq(n, 1.0, EPS));
    }

    #[test]
    fn normalizer_l2_direction_preserved() {
        let norm = EmbeddingNormalizer::l2();
        let mut v = vec![3.0, 4.0];
        norm.normalize(&mut v);
        // 3/5 = 0.6, 4/5 = 0.8
        assert!(approx_eq(v[0], 0.6, EPS));
        assert!(approx_eq(v[1], 0.8, EPS));
    }

    #[test]
    fn normalizer_l2_zero_vector() {
        let norm = EmbeddingNormalizer::l2();
        let mut v = vec![0.0, 0.0, 0.0];
        norm.normalize(&mut v);
        // Should not blow up; epsilon prevents division by zero.
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn normalizer_layer_norm_mean_zero() {
        let norm = EmbeddingNormalizer::layer_norm();
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        norm.normalize(&mut v);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(approx_eq(mean, 0.0, 1e-4));
    }

    #[test]
    fn normalizer_layer_norm_unit_variance() {
        let norm = EmbeddingNormalizer::layer_norm();
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        norm.normalize(&mut v);
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        assert!(approx_eq(var, 1.0, 1e-3));
    }

    #[test]
    fn normalizer_batch() {
        let norm = EmbeddingNormalizer::l2();
        let mut data = vec![3.0, 4.0, 0.0, 5.0]; // 2 vectors of dim 2
        norm.normalize_batch(&mut data, 2, 2);
        assert!(approx_eq(l2_norm(&data[0..2]), 1.0, EPS));
        assert!(approx_eq(l2_norm(&data[2..4]), 1.0, EPS));
    }

    #[test]
    fn normalizer_normalized_returns_copy() {
        let norm = EmbeddingNormalizer::l2();
        let v = vec![3.0, 4.0];
        let out = norm.normalized(&v);
        assert_ne!(v, out);
        assert!(approx_eq(l2_norm(&out), 1.0, EPS));
    }

    #[test]
    fn normalizer_mode_accessor() {
        assert_eq!(EmbeddingNormalizer::l2().mode(), NormMode::L2);
        assert_eq!(EmbeddingNormalizer::layer_norm().mode(), NormMode::LayerNorm);
    }

    #[test]
    fn normalizer_custom_eps() {
        let norm = EmbeddingNormalizer::new(NormMode::L2, 1e-8);
        let mut v = vec![1.0, 0.0];
        norm.normalize(&mut v);
        assert!(approx_eq(v[0], 1.0, EPS));
    }

    // ── EmbeddingEngine ──────────────────────────────────────────────────

    #[test]
    fn engine_token_only() {
        let cfg = EmbeddingConfig::new(5, 4);
        let weights = vec![0.0; 20];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb);
        let out = engine.forward(&[0, 1, 2], 0).unwrap();
        assert_eq!(out.len(), 3 * 4);
    }

    #[test]
    fn engine_with_sinusoidal() {
        let cfg = EmbeddingConfig::new(5, 4);
        let weights = vec![0.0; 20];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb).with_sinusoidal(100);
        assert_eq!(engine.position_kind(), PositionEncodingKind::Sinusoidal);
        let out = engine.forward(&[0, 1], 0).unwrap();
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn engine_with_l2_norm() {
        let cfg = EmbeddingConfig::new(2, 3);
        let weights = vec![3.0, 4.0, 0.0, 1.0, 0.0, 0.0];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb).with_l2_norm();
        let out = engine.forward(&[0], 0).unwrap();
        let norm = l2_norm(&out);
        assert!(approx_eq(norm, 1.0, EPS));
    }

    #[test]
    fn engine_oov_token() {
        let cfg = EmbeddingConfig::new(2, 3);
        let weights = vec![0.0; 6];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb);
        assert!(engine.forward(&[0, 5], 0).is_none());
    }

    #[test]
    fn engine_position_offset() {
        let cfg = EmbeddingConfig::new(3, 4);
        let weights = vec![0.0; 12];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb).with_sinusoidal(10);
        // Should succeed with offset.
        assert!(engine.forward(&[0], 5).is_some());
        // Should fail if offset + seq_len > max_seq_len.
        assert!(engine.forward(&[0], 10).is_none());
    }

    #[test]
    fn engine_token_embedding_accessor() {
        let cfg = EmbeddingConfig::new(10, 8);
        let emb = TokenEmbedding::new(cfg);
        let engine = EmbeddingEngine::new(emb);
        assert_eq!(engine.token_embedding().config().vocab_size, 10);
    }

    #[test]
    fn engine_with_layer_norm() {
        let cfg = EmbeddingConfig::new(2, 4);
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb).with_layer_norm();
        let out = engine.forward(&[0], 0).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(approx_eq(mean, 0.0, 1e-4));
    }

    #[test]
    fn engine_empty_input() {
        let cfg = EmbeddingConfig::new(5, 4);
        let weights = vec![0.0; 20];
        let emb = TokenEmbedding::from_weights(cfg, weights);
        let engine = EmbeddingEngine::new(emb);
        let out = engine.forward(&[], 0).unwrap();
        assert!(out.is_empty());
    }

    // ── Helper functions ─────────────────────────────────────────────────

    #[test]
    fn helper_l2_norm() {
        assert!(approx_eq(l2_norm(&[3.0, 4.0]), 5.0, EPS));
        assert!(approx_eq(l2_norm(&[0.0]), 0.0, EPS));
    }

    #[test]
    fn helper_dot_product() {
        assert!(approx_eq(dot_product(&[1.0, 2.0], &[3.0, 4.0]), 11.0, EPS));
    }

    #[test]
    fn helper_euclidean_distance() {
        assert!(approx_eq(euclidean_distance(&[0.0], &[5.0]), 5.0, EPS));
    }

    #[test]
    fn helper_renorm_no_change() {
        let mut v = vec![0.3, 0.4];
        renorm_vector(&mut v, 10.0);
        assert!(approx_eq(v[0], 0.3, EPS));
    }

    #[test]
    fn helper_renorm_clamp() {
        let mut v = vec![3.0, 4.0]; // norm = 5
        renorm_vector(&mut v, 1.0);
        assert!(approx_eq(l2_norm(&v), 1.0, EPS));
    }

    #[test]
    fn helper_sign_extend_4bit_positive() {
        assert_eq!(sign_extend_4bit(0x07), 7);
        assert_eq!(sign_extend_4bit(0x00), 0);
    }

    #[test]
    fn helper_sign_extend_4bit_negative() {
        assert_eq!(sign_extend_4bit(0x0F), -1); // 1111 → -1
        assert_eq!(sign_extend_4bit(0x08), -8); // 1000 → -8
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        assert!(approx_eq(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0, EPS));
    }

    // ── Clone / Debug coverage ───────────────────────────────────────────

    #[test]
    fn clone_and_debug_coverage() {
        let _ = format!("{:?}", EmbeddingConfig::new(1, 1));
        let _ = format!("{:?}", TokenEmbedding::new(EmbeddingConfig::new(1, 1)));
        let _ = format!("{:?}", PositionalEmbedding::new(1, 2));
        let _ = format!("{:?}", RotaryEmbedding::with_default_base(2, 1));
        let _ = format!("{:?}", ALiBiEmbedding::new(1));
        let _ = format!(
            "{:?}",
            EmbeddingQuantizer::quantize(&EmbeddingConfig::new(1, 2), &[0.0; 2], QuantFormat::Int8)
        );
        let _ = format!("{:?}", EmbeddingAggregator::new(AggregationStrategy::Mean));
        let _ = format!("{:?}", SimilaritySearch::new(SimilarityMetric::Cosine, vec![0.0; 2], 2));
        let _ = format!("{:?}", EmbeddingNormalizer::l2());
        let _ =
            format!("{:?}", EmbeddingEngine::new(TokenEmbedding::new(EmbeddingConfig::new(1, 1))));
    }

    #[test]
    fn clone_all_types() {
        let cfg = EmbeddingConfig::new(2, 2);
        let _ = cfg.clone();
        let te = TokenEmbedding::new(cfg.clone());
        let _ = te.clone();
        let pe = PositionalEmbedding::new(2, 2);
        let _ = pe.clone();
        let rope = RotaryEmbedding::with_default_base(2, 2);
        let _ = rope.clone();
        let alibi = ALiBiEmbedding::new(2);
        let _ = alibi.clone();
        let quant = EmbeddingQuantizer::quantize(&cfg, &[0.0; 4], QuantFormat::Int8);
        let _ = quant.clone();
        let agg = EmbeddingAggregator::new(AggregationStrategy::Mean);
        let _ = agg.clone();
        let ss = SimilaritySearch::new(SimilarityMetric::Cosine, vec![0.0; 2], 2);
        let _ = ss.clone();
        let norm = EmbeddingNormalizer::l2();
        let _ = norm.clone();
        let engine = EmbeddingEngine::new(TokenEmbedding::new(cfg));
        let _ = engine.clone();
    }
}
