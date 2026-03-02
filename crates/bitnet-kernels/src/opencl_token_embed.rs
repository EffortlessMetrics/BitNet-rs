//! Token embedding manager with GPU-accelerated lookup, tying, and quantized
//! storage for OpenCL backends.
//!
//! # Overview
//!
//! This module manages multiple embedding tables (token, position, segment) with
//! support for:
//!
//! - **`EmbeddingManager`** — orchestrates token, position, and segment tables
//! - **`EmbeddingConfig`** — vocab_size, embed_dim, max_seq_len, quantize,
//!   tie_output
//! - **`QuantizedEmbedding`** — INT8-quantized table with dequantization on
//!   lookup
//! - **`TiedEmbedding`** — shared weights between input embedding and output
//!   projection
//! - **`EmbeddingScaler`** — sqrt(embed_dim) scaling (T5-style)
//! - **`PositionEncoding`** — Learned, Sinusoidal, RoPE, ALiBi
//! - **`EmbedDropout`** — whole-dimension dropout
//! - **`EmbeddingInitializer`** — GGUF or random initialisation
//! - OpenCL kernel sources for batched embedding lookup
//! - CPU reference implementations

use std::fmt;

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL C kernel source for batched embedding lookup and position encoding.
pub const TOKEN_EMBED_CL: &str = r#"
__kernel void batched_embedding_lookup(
    __global const float* weight,   // [vocab_size, embed_dim]
    __global const uint*  token_ids, // [batch_size]
    __global       float* output,   // [batch_size, embed_dim]
    const uint vocab_size,
    const uint embed_dim)
{
    uint gid = get_global_id(0); // token index in batch
    uint tid = token_ids[gid];
    uint out_off = gid * embed_dim;
    if (tid < vocab_size) {
        uint src_off = tid * embed_dim;
        for (uint i = 0; i < embed_dim; ++i) {
            output[out_off + i] = weight[src_off + i];
        }
    } else {
        for (uint i = 0; i < embed_dim; ++i) {
            output[out_off + i] = 0.0f;
        }
    }
}

__kernel void batched_embedding_lookup_i8(
    __global const char*  weight_q, // [vocab_size, embed_dim] INT8
    __global const float* scales,   // [vocab_size]
    __global const uint*  token_ids,
    __global       float* output,
    const uint vocab_size,
    const uint embed_dim)
{
    uint gid = get_global_id(0);
    uint tid = token_ids[gid];
    uint out_off = gid * embed_dim;
    if (tid < vocab_size) {
        float s = scales[tid];
        uint src_off = tid * embed_dim;
        for (uint i = 0; i < embed_dim; ++i) {
            output[out_off + i] = ((float)weight_q[src_off + i]) * s;
        }
    } else {
        for (uint i = 0; i < embed_dim; ++i) {
            output[out_off + i] = 0.0f;
        }
    }
}

__kernel void add_sinusoidal_position(
    __global float* embeddings, // [seq_len, embed_dim] in-place
    const uint embed_dim,
    const uint position_offset)
{
    uint t = get_global_id(0); // token index
    uint pos = position_offset + t;
    uint base = t * embed_dim;
    for (uint i = 0; i < embed_dim; ++i) {
        float angle = (float)pos / pow(10000.0f, (float)(2 * (i / 2)) / (float)embed_dim);
        float pe = (i % 2 == 0) ? sin(angle) : cos(angle);
        embeddings[base + i] += pe;
    }
}

__kernel void scale_embeddings(
    __global float* embeddings,
    const uint total_elements,
    const float factor)
{
    uint gid = get_global_id(0);
    if (gid < total_elements) {
        embeddings[gid] *= factor;
    }
}
"#;

// ── Configuration ────────────────────────────────────────────────

/// Extended embedding configuration for the token embedding manager.
#[derive(Debug, Clone)]
pub struct TokenEmbedConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Maximum sequence length (for position encodings).
    pub max_seq_len: usize,
    /// Whether to use INT8 quantised storage.
    pub quantize: bool,
    /// Whether input embedding and output projection share weights.
    pub tie_output: bool,
}

impl TokenEmbedConfig {
    /// Create a new token embedding configuration.
    pub fn new(vocab_size: usize, embed_dim: usize, max_seq_len: usize) -> Self {
        Self { vocab_size, embed_dim, max_seq_len, quantize: false, tie_output: false }
    }

    /// Enable INT8 quantised storage.
    #[must_use]
    pub fn with_quantize(mut self, quantize: bool) -> Self {
        self.quantize = quantize;
        self
    }

    /// Enable tied output projection weights.
    #[must_use]
    pub fn with_tie_output(mut self, tie: bool) -> Self {
        self.tie_output = tie;
        self
    }
}

// ── Position encoding variants ───────────────────────────────────

/// Position encoding strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum PositionEncoding {
    /// Learned position embeddings stored in a weight table.
    Learned(Vec<f32>),
    /// Fixed sinusoidal encoding (Vaswani et al., 2017).
    Sinusoidal,
    /// Rotary position embeddings — positional info is applied externally.
    RoPE,
    /// Attention with Linear Biases — positional info is applied externally.
    ALiBi,
}

impl fmt::Display for PositionEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Learned(_) => write!(f, "Learned"),
            Self::Sinusoidal => write!(f, "Sinusoidal"),
            Self::RoPE => write!(f, "RoPE"),
            Self::ALiBi => write!(f, "ALiBi"),
        }
    }
}

// ── QuantizedEmbedding ───────────────────────────────────────────

/// INT8-quantized embedding table with per-row scale factors.
///
/// Each row is quantized as `q[i] = round(w[i] / scale)` where
/// `scale = max(abs(row)) / 127`. Dequantization: `w[i] ≈ q[i] * scale`.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// INT8 weight matrix: `[vocab_size, embed_dim]`.
    pub weight_q: Vec<i8>,
    /// Per-row scale factors: `[vocab_size]`.
    pub scales: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl QuantizedEmbedding {
    /// Quantize a full-precision weight table to INT8.
    pub fn from_float(weight: &[f32], vocab_size: usize, embed_dim: usize) -> Result<Self> {
        let expected = vocab_size * embed_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * embed_dim({})",
                    weight.len(),
                    vocab_size,
                    embed_dim,
                ),
            }
            .into());
        }

        let mut weight_q = vec![0i8; expected];
        let mut scales = vec![0.0f32; vocab_size];

        for (row, scale_out) in scales.iter_mut().enumerate() {
            let start = row * embed_dim;
            let row_slice = &weight[start..start + embed_dim];
            let abs_max = row_slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            *scale_out = scale;
            let inv_scale = 1.0 / scale;
            for (i, &v) in row_slice.iter().enumerate() {
                weight_q[start + i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }

        Ok(Self { weight_q, scales, vocab_size, embed_dim })
    }

    /// Look up embeddings with dequantization for a batch of token IDs.
    pub fn lookup(&self, token_ids: &[u32], output: &mut [f32]) -> Result<()> {
        quantized_embedding_lookup_ref(
            token_ids,
            &self.weight_q,
            &self.scales,
            output,
            self.vocab_size,
            self.embed_dim,
        )
    }
}

// ── TiedEmbedding ────────────────────────────────────────────────

/// Shared weights between input embedding and output projection.
///
/// Stores a single `[vocab_size, embed_dim]` matrix used both for token
/// lookup (forward) and logit projection (reverse `hidden @ weight^T`).
#[derive(Debug, Clone)]
pub struct TiedEmbedding {
    /// Shared weight matrix: `[vocab_size, embed_dim]`.
    pub weight: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl TiedEmbedding {
    /// Create a tied embedding with shared weights.
    pub fn new(weight: Vec<f32>, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        let expected = vocab_size * embed_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tied weight length {} != vocab_size({}) * embed_dim({})",
                    weight.len(),
                    vocab_size,
                    embed_dim,
                ),
            }
            .into());
        }
        Ok(Self { weight, vocab_size, embed_dim })
    }

    /// Token lookup (forward direction).
    pub fn lookup(&self, token_ids: &[u32], output: &mut [f32]) -> Result<()> {
        batched_embedding_lookup_ref(
            token_ids,
            &self.weight,
            output,
            self.vocab_size,
            self.embed_dim,
        )
    }

    /// Output projection: `logits = hidden @ weight^T`.
    pub fn project(&self, hidden: &[f32], output: &mut [f32], seq_len: usize) -> Result<()> {
        tied_output_projection_ref(
            hidden,
            &self.weight,
            output,
            seq_len,
            self.embed_dim,
            self.vocab_size,
        )
    }

    /// Get a reference to the shared weight data.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }
}

// ── EmbeddingScaler ──────────────────────────────────────────────

/// Applies `sqrt(embed_dim)` scaling to embeddings (T5-style).
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingScaler {
    /// The scaling factor (typically `sqrt(embed_dim)`).
    pub factor: f32,
}

impl EmbeddingScaler {
    /// Create a scaler with `factor = sqrt(embed_dim)`.
    pub fn new(embed_dim: usize) -> Self {
        Self { factor: (embed_dim as f32).sqrt() }
    }

    /// Create a scaler with a custom factor.
    pub fn with_factor(factor: f32) -> Self {
        Self { factor }
    }

    /// Scale embeddings in-place.
    pub fn apply(&self, data: &mut [f32]) {
        for v in data.iter_mut() {
            *v *= self.factor;
        }
    }
}

// ── EmbedDropout ─────────────────────────────────────────────────

/// Embedding dropout: zeroes out entire embedding dimensions.
///
/// Unlike standard dropout that zeroes individual elements, embedding
/// dropout zeroes the same dimension index across all tokens in a batch,
/// which is more appropriate for embedding regularisation.
#[derive(Debug, Clone)]
pub struct EmbedDropout {
    /// Dropout rate in `[0.0, 1.0)`.
    pub rate: f32,
}

impl EmbedDropout {
    /// Create a new embedding dropout layer.
    pub fn new(rate: f32) -> Self {
        Self { rate: rate.clamp(0.0, 1.0) }
    }

    /// Apply dimension-wise dropout using a pre-computed mask.
    ///
    /// `mask[d]` is `true` if dimension `d` should be kept; `false` → zero.
    /// `data` layout: `[n_tokens, embed_dim]`.
    pub fn apply_mask(
        &self,
        data: &mut [f32],
        n_tokens: usize,
        embed_dim: usize,
        mask: &[bool],
    ) -> Result<()> {
        if mask.len() < embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!("mask length {} < embed_dim({})", mask.len(), embed_dim,),
            }
            .into());
        }
        if data.len() < n_tokens * embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length {} < n_tokens({}) * embed_dim({})",
                    data.len(),
                    n_tokens,
                    embed_dim,
                ),
            }
            .into());
        }
        let scale = if self.rate < 1.0 { 1.0 / (1.0 - self.rate) } else { 0.0 };
        for t in 0..n_tokens {
            let base = t * embed_dim;
            for d in 0..embed_dim {
                if !mask[d] {
                    data[base + d] = 0.0;
                } else {
                    data[base + d] *= scale;
                }
            }
        }
        Ok(())
    }

    /// Generate a deterministic dropout mask from a seed.
    pub fn generate_mask(&self, embed_dim: usize, seed: u64) -> Vec<bool> {
        let mut mask = vec![true; embed_dim];
        if self.rate <= 0.0 {
            return mask;
        }
        // Simple hash-based deterministic pseudo-random
        let threshold = (self.rate * u32::MAX as f32) as u64;
        for (i, m) in mask.iter_mut().enumerate() {
            let hash = simple_hash(seed, i as u64);
            if hash % (u32::MAX as u64) < threshold {
                *m = false;
            }
        }
        mask
    }
}

/// Simple deterministic hash for reproducible dropout masks.
fn simple_hash(seed: u64, idx: u64) -> u64 {
    let mut h = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(idx);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

// ── EmbeddingInitializer ─────────────────────────────────────────

/// Source for initialising embedding weights.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingInitializer {
    /// Load weights from GGUF tensor data.
    Gguf(Vec<f32>),
    /// Random normal initialisation with given std-dev.
    RandomNormal { std_dev: f32, seed: u64 },
    /// All zeros (useful for segment embeddings when unused).
    Zeros,
}

impl EmbeddingInitializer {
    /// Generate a weight vector of length `vocab_size * embed_dim`.
    pub fn initialize(&self, vocab_size: usize, embed_dim: usize) -> Result<Vec<f32>> {
        let n = vocab_size * embed_dim;
        match self {
            Self::Gguf(data) => {
                if data.len() != n {
                    return Err(KernelError::InvalidArguments {
                        reason: format!("GGUF data length {} != expected {}", data.len(), n,),
                    }
                    .into());
                }
                Ok(data.clone())
            }
            Self::RandomNormal { std_dev, seed } => Ok(random_normal_weights(n, *std_dev, *seed)),
            Self::Zeros => Ok(vec![0.0; n]),
        }
    }
}

/// Generate pseudo-random normal weights via Box-Muller transform.
fn random_normal_weights(n: usize, std_dev: f32, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed;
    let mut i = 0;
    while i < n {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = (s >> 33) as f32 / (1u64 << 31) as f32;
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = (s >> 33) as f32 / (1u64 << 31) as f32;
        let u1 = u1.max(1e-10); // avoid log(0)
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(r * theta.cos() * std_dev);
        if i + 1 < n {
            out.push(r * theta.sin() * std_dev);
        }
        i += 2;
    }
    out.truncate(n);
    out
}

// ── EmbeddingManager ─────────────────────────────────────────────

/// Manages token, position, and segment embedding tables.
///
/// Coordinates lookup, position encoding, scaling, and optional dropout
/// into a single forward pass.
#[derive(Debug)]
pub struct EmbeddingManager {
    /// Configuration.
    pub config: TokenEmbedConfig,
    /// Token embedding weights: `[vocab_size, embed_dim]`.
    token_weight: Vec<f32>,
    /// Quantized token embedding (if config.quantize).
    quantized: Option<QuantizedEmbedding>,
    /// Position encoding strategy.
    position_encoding: PositionEncoding,
    /// Optional segment embedding: `[num_segments, embed_dim]`.
    segment_weight: Option<Vec<f32>>,
    /// Number of segment types (e.g. 2 for BERT).
    num_segments: usize,
    /// Optional T5-style scaler.
    scaler: Option<EmbeddingScaler>,
    /// Optional dropout.
    dropout: Option<EmbedDropout>,
}

impl EmbeddingManager {
    /// Create a new embedding manager from token weights and configuration.
    pub fn new(
        token_weight: Vec<f32>,
        config: TokenEmbedConfig,
        position_encoding: PositionEncoding,
    ) -> Result<Self> {
        let expected = config.vocab_size * config.embed_dim;
        if token_weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "token_weight length {} != vocab_size({}) * embed_dim({})",
                    token_weight.len(),
                    config.vocab_size,
                    config.embed_dim,
                ),
            }
            .into());
        }

        let quantized = if config.quantize {
            Some(QuantizedEmbedding::from_float(
                &token_weight,
                config.vocab_size,
                config.embed_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            token_weight,
            quantized,
            position_encoding,
            segment_weight: None,
            num_segments: 0,
            scaler: None,
            dropout: None,
        })
    }

    /// Add segment embeddings (e.g. sentence A / sentence B for BERT).
    pub fn with_segments(mut self, segment_weight: Vec<f32>, num_segments: usize) -> Result<Self> {
        let expected = num_segments * self.config.embed_dim;
        if segment_weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "segment_weight length {} != num_segments({}) * embed_dim({})",
                    segment_weight.len(),
                    num_segments,
                    self.config.embed_dim,
                ),
            }
            .into());
        }
        self.segment_weight = Some(segment_weight);
        self.num_segments = num_segments;
        Ok(self)
    }

    /// Enable T5-style sqrt(embed_dim) scaling.
    #[must_use]
    pub fn with_scaler(mut self) -> Self {
        self.scaler = Some(EmbeddingScaler::new(self.config.embed_dim));
        self
    }

    /// Enable embedding dropout.
    #[must_use]
    pub fn with_dropout(mut self, rate: f32) -> Self {
        self.dropout = Some(EmbedDropout::new(rate));
        self
    }

    /// Run the full embedding forward pass.
    ///
    /// 1. Token lookup (quantized or full-precision)
    /// 2. Add position encoding
    /// 3. Add segment encoding (if provided)
    /// 4. Apply scaler (if enabled)
    /// 5. Apply dropout (if enabled, with optional seed)
    pub fn forward(
        &self,
        token_ids: &[u32],
        position_offset: usize,
        segment_ids: Option<&[u32]>,
        dropout_seed: Option<u64>,
        output: &mut [f32],
    ) -> Result<()> {
        let seq_len = token_ids.len();
        let d = self.config.embed_dim;
        let needed = seq_len * d;
        if output.len() < needed {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * embed_dim({})",
                    output.len(),
                    seq_len,
                    d,
                ),
            }
            .into());
        }

        // 1. Token lookup
        if let Some(ref q) = self.quantized {
            q.lookup(token_ids, output)?;
        } else {
            batched_embedding_lookup_ref(
                token_ids,
                &self.token_weight,
                output,
                self.config.vocab_size,
                d,
            )?;
        }

        // 2. Position encoding
        match &self.position_encoding {
            PositionEncoding::Learned(pos_w) => {
                add_learned_position_ref(output, pos_w, seq_len, d, position_offset)?;
            }
            PositionEncoding::Sinusoidal => {
                add_sinusoidal_position_ref(output, seq_len, d, position_offset);
            }
            PositionEncoding::RoPE | PositionEncoding::ALiBi => {
                // Applied externally in the attention layer.
            }
        }

        // 3. Segment encoding
        if let (Some(seg_w), Some(seg_ids)) = (&self.segment_weight, segment_ids) {
            add_segment_embedding_ref(output, seg_w, seg_ids, d, self.num_segments)?;
        }

        // 4. Scaling
        if let Some(ref scaler) = self.scaler {
            scaler.apply(&mut output[..needed]);
        }

        // 5. Dropout
        if let Some(ref dropout) = self.dropout
            && let Some(seed) = dropout_seed
        {
            let mask = dropout.generate_mask(d, seed);
            dropout.apply_mask(&mut output[..needed], seq_len, d, &mask)?;
        }

        Ok(())
    }

    /// Access the raw token weights.
    pub fn token_weight(&self) -> &[f32] {
        &self.token_weight
    }

    /// Access the position encoding variant.
    pub fn position_encoding(&self) -> &PositionEncoding {
        &self.position_encoding
    }
}

// ── CPU reference: batched embedding lookup ──────────────────────

/// Batched embedding lookup (CPU reference).
///
/// For each token ID copies the corresponding row from `weight` into
/// `output`. OOV IDs (`>= vocab_size`) produce zero vectors.
pub fn batched_embedding_lookup_ref(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embed_dim: usize,
) -> Result<()> {
    let seq_len = token_ids.len();
    if weight.len() < vocab_size * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * embed_dim({})",
                weight.len(),
                vocab_size,
                embed_dim,
            ),
        }
        .into());
    }
    if output.len() < seq_len * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * embed_dim({})",
                output.len(),
                seq_len,
                embed_dim,
            ),
        }
        .into());
    }

    for (t, &tok) in token_ids.iter().enumerate() {
        let tid = tok as usize;
        let out_start = t * embed_dim;
        if tid < vocab_size {
            let src_start = tid * embed_dim;
            output[out_start..out_start + embed_dim]
                .copy_from_slice(&weight[src_start..src_start + embed_dim]);
        } else {
            output[out_start..out_start + embed_dim].fill(0.0);
        }
    }
    Ok(())
}

// ── CPU reference: quantized embedding lookup ────────────────────

/// INT8 quantized embedding lookup with per-row dequantization.
pub fn quantized_embedding_lookup_ref(
    token_ids: &[u32],
    weight_q: &[i8],
    scales: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embed_dim: usize,
) -> Result<()> {
    let seq_len = token_ids.len();
    if weight_q.len() < vocab_size * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight_q length {} < vocab_size({}) * embed_dim({})",
                weight_q.len(),
                vocab_size,
                embed_dim,
            ),
        }
        .into());
    }
    if scales.len() < vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales length {} < vocab_size({})", scales.len(), vocab_size),
        }
        .into());
    }
    if output.len() < seq_len * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * embed_dim({})",
                output.len(),
                seq_len,
                embed_dim,
            ),
        }
        .into());
    }

    for (t, &tok) in token_ids.iter().enumerate() {
        let tid = tok as usize;
        let out_start = t * embed_dim;
        if tid < vocab_size {
            let src_start = tid * embed_dim;
            let scale = scales[tid];
            for i in 0..embed_dim {
                output[out_start + i] = weight_q[src_start + i] as f32 * scale;
            }
        } else {
            output[out_start..out_start + embed_dim].fill(0.0);
        }
    }
    Ok(())
}

// ── CPU reference: position encodings ────────────────────────────

/// Add learned position embeddings in-place.
fn add_learned_position_ref(
    embeddings: &mut [f32],
    pos_weight: &[f32],
    seq_len: usize,
    embed_dim: usize,
    position_offset: usize,
) -> Result<()> {
    let max_pos = pos_weight.len() / embed_dim;
    if position_offset + seq_len > max_pos {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "position_offset({}) + seq_len({}) > max_pos({})",
                position_offset, seq_len, max_pos,
            ),
        }
        .into());
    }
    for t in 0..seq_len {
        let pos = position_offset + t;
        let emb_off = t * embed_dim;
        let pos_off = pos * embed_dim;
        for i in 0..embed_dim {
            embeddings[emb_off + i] += pos_weight[pos_off + i];
        }
    }
    Ok(())
}

/// Add sinusoidal position encoding in-place (Vaswani et al., 2017).
pub fn add_sinusoidal_position_ref(
    embeddings: &mut [f32],
    seq_len: usize,
    embed_dim: usize,
    position_offset: usize,
) {
    for t in 0..seq_len {
        let pos = (position_offset + t) as f32;
        let base = t * embed_dim;
        for i in 0..embed_dim {
            let dim_pair = (i / 2) as f32;
            let angle = pos / 10000.0f32.powf(2.0 * dim_pair / embed_dim as f32);
            if i % 2 == 0 {
                embeddings[base + i] += angle.sin();
            } else {
                embeddings[base + i] += angle.cos();
            }
        }
    }
}

// ── CPU reference: segment embedding ─────────────────────────────

/// Add segment embeddings in-place.
fn add_segment_embedding_ref(
    embeddings: &mut [f32],
    seg_weight: &[f32],
    segment_ids: &[u32],
    embed_dim: usize,
    num_segments: usize,
) -> Result<()> {
    for (t, &seg) in segment_ids.iter().enumerate() {
        let sid = seg as usize;
        if sid >= num_segments {
            return Err(KernelError::InvalidArguments {
                reason: format!("segment_id {} >= num_segments({})", sid, num_segments),
            }
            .into());
        }
        let emb_off = t * embed_dim;
        let seg_off = sid * embed_dim;
        for i in 0..embed_dim {
            embeddings[emb_off + i] += seg_weight[seg_off + i];
        }
    }
    Ok(())
}

// ── CPU reference: tied output projection ────────────────────────

/// Output projection with tied weights: `logits = hidden @ weight^T`.
pub fn tied_output_projection_ref(
    hidden: &[f32],
    weight: &[f32],
    output: &mut [f32],
    seq_len: usize,
    embed_dim: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden.len() < seq_len * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "hidden length {} < seq_len({}) * embed_dim({})",
                hidden.len(),
                seq_len,
                embed_dim,
            ),
        }
        .into());
    }
    if weight.len() < vocab_size * embed_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * embed_dim({})",
                weight.len(),
                vocab_size,
                embed_dim,
            ),
        }
        .into());
    }
    if output.len() < seq_len * vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * vocab_size({})",
                output.len(),
                seq_len,
                vocab_size,
            ),
        }
        .into());
    }

    for s in 0..seq_len {
        for v in 0..vocab_size {
            let mut acc = 0.0f32;
            let h_off = s * embed_dim;
            let w_off = v * embed_dim;
            for k in 0..embed_dim {
                acc += hidden[h_off + k] * weight[w_off + k];
            }
            output[s * vocab_size + v] = acc;
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpenCL kernel source ─────────────────────────────────

    #[test]
    fn opencl_source_is_not_empty() {
        assert!(!TOKEN_EMBED_CL.is_empty());
    }

    #[test]
    fn opencl_source_has_batched_lookup_kernel() {
        assert!(TOKEN_EMBED_CL.contains("batched_embedding_lookup"));
    }

    #[test]
    fn opencl_source_has_i8_lookup_kernel() {
        assert!(TOKEN_EMBED_CL.contains("batched_embedding_lookup_i8"));
    }

    #[test]
    fn opencl_source_has_sinusoidal_kernel() {
        assert!(TOKEN_EMBED_CL.contains("add_sinusoidal_position"));
    }

    #[test]
    fn opencl_source_has_scale_kernel() {
        assert!(TOKEN_EMBED_CL.contains("scale_embeddings"));
    }

    #[test]
    fn opencl_source_has_kernel_keyword() {
        assert!(TOKEN_EMBED_CL.contains("__kernel"));
    }

    // ── TokenEmbedConfig ─────────────────────────────────────

    #[test]
    fn config_basic() {
        let cfg = TokenEmbedConfig::new(32000, 2048, 4096);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embed_dim, 2048);
        assert_eq!(cfg.max_seq_len, 4096);
        assert!(!cfg.quantize);
        assert!(!cfg.tie_output);
    }

    #[test]
    fn config_with_quantize() {
        let cfg = TokenEmbedConfig::new(100, 64, 512).with_quantize(true);
        assert!(cfg.quantize);
    }

    #[test]
    fn config_with_tie_output() {
        let cfg = TokenEmbedConfig::new(100, 64, 512).with_tie_output(true);
        assert!(cfg.tie_output);
    }

    // ── PositionEncoding display ─────────────────────────────

    #[test]
    fn position_encoding_display() {
        assert_eq!(format!("{}", PositionEncoding::Sinusoidal), "Sinusoidal");
        assert_eq!(format!("{}", PositionEncoding::RoPE), "RoPE");
        assert_eq!(format!("{}", PositionEncoding::ALiBi), "ALiBi");
        assert_eq!(format!("{}", PositionEncoding::Learned(vec![1.0])), "Learned");
    }

    // ── batched_embedding_lookup_ref ─────────────────────────

    #[test]
    fn lookup_known_token_returns_correct_vector() {
        let weight = vec![
            0.1, 0.2, // token 0
            0.3, 0.4, // token 1
            0.5, 0.6, // token 2
        ];
        let mut out = vec![0.0; 2];
        batched_embedding_lookup_ref(&[1], &weight, &mut out, 3, 2).unwrap();
        assert_eq!(out, vec![0.3, 0.4]);
    }

    #[test]
    fn lookup_multiple_tokens() {
        let weight = vec![
            1.0, 2.0, 3.0, // tok 0
            4.0, 5.0, 6.0, // tok 1
            7.0, 8.0, 9.0, // tok 2
        ];
        let mut out = vec![0.0; 6];
        batched_embedding_lookup_ref(&[2, 0], &weight, &mut out, 3, 3).unwrap();
        assert_eq!(&out[0..3], &[7.0, 8.0, 9.0]);
        assert_eq!(&out[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn lookup_oov_returns_zero() {
        let weight = vec![1.0; 4]; // vocab=2, dim=2
        let mut out = vec![99.0; 2];
        batched_embedding_lookup_ref(&[5], &weight, &mut out, 2, 2).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn lookup_u32_max_returns_zero() {
        let weight = vec![1.0; 4];
        let mut out = vec![99.0; 2];
        batched_embedding_lookup_ref(&[u32::MAX], &weight, &mut out, 2, 2).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn lookup_empty_batch() {
        let weight = vec![1.0; 4];
        let mut out = vec![];
        batched_embedding_lookup_ref(&[], &weight, &mut out, 2, 2).unwrap();
    }

    #[test]
    fn lookup_repeated_tokens_identical() {
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let mut out = vec![0.0; 6];
        batched_embedding_lookup_ref(&[1, 1, 1], &weight, &mut out, 2, 2).unwrap();
        assert_eq!(&out[0..2], &out[2..4]);
        assert_eq!(&out[0..2], &out[4..6]);
    }

    #[test]
    fn lookup_rejects_short_weight() {
        let mut out = vec![0.0; 2];
        assert!(batched_embedding_lookup_ref(&[0], &[1.0], &mut out, 2, 2).is_err());
    }

    #[test]
    fn lookup_rejects_short_output() {
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 1];
        assert!(batched_embedding_lookup_ref(&[0], &weight, &mut out, 2, 2).is_err());
    }

    #[test]
    fn lookup_embed_dim_one() {
        let weight = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0; 3];
        batched_embedding_lookup_ref(&[2, 0, 1], &weight, &mut out, 3, 1).unwrap();
        assert_eq!(out, vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn lookup_large_batch_matches_weight() {
        let vocab = 50;
        let dim = 32;
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let ids: Vec<u32> = (0..vocab as u32).collect();
        let mut out = vec![0.0; vocab * dim];
        batched_embedding_lookup_ref(&ids, &weight, &mut out, vocab, dim).unwrap();
        assert_eq!(out, weight);
    }

    // ── QuantizedEmbedding ───────────────────────────────────

    #[test]
    fn quantized_from_float_rejects_wrong_size() {
        assert!(QuantizedEmbedding::from_float(&[0.0; 5], 2, 3).is_err());
    }

    #[test]
    fn quantized_lookup_within_tolerance() {
        let weight = vec![
            0.5, -0.3, 0.7, -0.1, // tok 0
            1.0, -1.0, 0.0, 0.5, // tok 1
        ];
        let q = QuantizedEmbedding::from_float(&weight, 2, 4).unwrap();
        let mut out = vec![0.0; 4];
        q.lookup(&[0], &mut out).unwrap();
        for i in 0..4 {
            assert!(
                (out[i] - weight[i]).abs() < 0.02,
                "dim {i}: got {}, expected {}",
                out[i],
                weight[i]
            );
        }
    }

    #[test]
    fn quantized_lookup_multiple_tokens() {
        let weight = vec![
            1.0, 2.0, // tok 0
            3.0, 4.0, // tok 1
        ];
        let q = QuantizedEmbedding::from_float(&weight, 2, 2).unwrap();
        let mut out = vec![0.0; 4];
        q.lookup(&[1, 0], &mut out).unwrap();
        // Allow small quantization error
        assert!((out[0] - 3.0).abs() < 0.1);
        assert!((out[1] - 4.0).abs() < 0.1);
        assert!((out[2] - 1.0).abs() < 0.1);
        assert!((out[3] - 2.0).abs() < 0.1);
    }

    #[test]
    fn quantized_oov_returns_zero() {
        let weight = vec![1.0; 4];
        let q = QuantizedEmbedding::from_float(&weight, 2, 2).unwrap();
        let mut out = vec![99.0; 2];
        q.lookup(&[100], &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn quantized_zero_weight_roundtrip() {
        let weight = vec![0.0; 8]; // vocab=2, dim=4
        let q = QuantizedEmbedding::from_float(&weight, 2, 4).unwrap();
        let mut out = vec![99.0; 4];
        q.lookup(&[0], &mut out).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn quantized_scales_are_positive() {
        let weight: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let q = QuantizedEmbedding::from_float(&weight, 4, 4).unwrap();
        assert!(q.scales.iter().all(|&s| s > 0.0));
    }

    // ── TiedEmbedding ────────────────────────────────────────

    #[test]
    fn tied_rejects_wrong_size() {
        assert!(TiedEmbedding::new(vec![0.0; 5], 2, 3).is_err());
    }

    #[test]
    fn tied_lookup_correct() {
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let tied = TiedEmbedding::new(weight, 2, 2).unwrap();
        let mut out = vec![0.0; 2];
        tied.lookup(&[1], &mut out).unwrap();
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn tied_shares_weight_data() {
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let tied = TiedEmbedding::new(weight.clone(), 2, 2).unwrap();
        assert_eq!(tied.weight(), &weight[..]);
    }

    #[test]
    fn tied_projection_uses_same_weight() {
        let weight = vec![
            1.0, 0.0, // vocab 0
            0.0, 1.0, // vocab 1
        ];
        let tied = TiedEmbedding::new(weight, 2, 2).unwrap();
        let mut emb = vec![0.0; 2];
        tied.lookup(&[0], &mut emb).unwrap();
        assert_eq!(emb, vec![1.0, 0.0]);
        let mut logits = vec![0.0; 2];
        tied.project(&emb, &mut logits, 1).unwrap();
        assert_eq!(logits, vec![1.0, 0.0]);
    }

    #[test]
    fn tied_roundtrip_identity() {
        let weight = vec![
            1.0, 0.0, 0.0, // vocab 0
            0.0, 1.0, 0.0, // vocab 1
            0.0, 0.0, 1.0, // vocab 2
        ];
        let tied = TiedEmbedding::new(weight, 3, 3).unwrap();
        for tok in 0..3u32 {
            let mut emb = vec![0.0; 3];
            tied.lookup(&[tok], &mut emb).unwrap();
            let mut logits = vec![0.0; 3];
            tied.project(&emb, &mut logits, 1).unwrap();
            let argmax =
                logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            assert_eq!(argmax, tok as usize);
        }
    }

    #[test]
    fn tied_project_rejects_short_hidden() {
        let tied = TiedEmbedding::new(vec![1.0; 4], 2, 2).unwrap();
        let mut out = vec![0.0; 2];
        assert!(tied.project(&[1.0], &mut out, 1).is_err());
    }

    #[test]
    fn tied_project_rejects_short_output() {
        let tied = TiedEmbedding::new(vec![1.0; 4], 2, 2).unwrap();
        let mut out = vec![0.0; 1];
        assert!(tied.project(&[1.0, 2.0], &mut out, 1).is_err());
    }

    // ── EmbeddingScaler ──────────────────────────────────────

    #[test]
    fn scaler_sqrt_dim() {
        let s = EmbeddingScaler::new(64);
        assert!((s.factor - 8.0).abs() < 1e-6);
    }

    #[test]
    fn scaler_custom_factor() {
        let s = EmbeddingScaler::with_factor(3.14);
        assert!((s.factor - 3.14).abs() < 1e-6);
    }

    #[test]
    fn scaler_apply_multiplies() {
        let s = EmbeddingScaler::new(4); // factor = 2.0
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        s.apply(&mut data);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn scaler_dim_one() {
        let s = EmbeddingScaler::new(1);
        assert!((s.factor - 1.0).abs() < 1e-6);
    }

    // ── EmbedDropout ─────────────────────────────────────────

    #[test]
    fn dropout_rate_clamped() {
        let d = EmbedDropout::new(-0.5);
        assert!((d.rate - 0.0).abs() < 1e-6);
        let d = EmbedDropout::new(1.5);
        assert!((d.rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dropout_zero_rate_preserves_all() {
        let d = EmbedDropout::new(0.0);
        let mask = d.generate_mask(4, 42);
        assert!(mask.iter().all(|&m| m));
    }

    #[test]
    fn dropout_zeroes_correct_dimensions() {
        let d = EmbedDropout::new(0.5);
        let mask = vec![true, false, true, false];
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 toks
        d.apply_mask(&mut data, 2, 4, &mask).unwrap();
        // Dropped dimensions are zero across all tokens
        assert_eq!(data[1], 0.0);
        assert_eq!(data[3], 0.0);
        assert_eq!(data[5], 0.0);
        assert_eq!(data[7], 0.0);
        // Kept dimensions are scaled by 1/(1-0.5) = 2.0
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        assert!((data[4] - 10.0).abs() < 1e-6);
        assert!((data[6] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn dropout_mask_deterministic() {
        let d = EmbedDropout::new(0.3);
        let m1 = d.generate_mask(16, 999);
        let m2 = d.generate_mask(16, 999);
        assert_eq!(m1, m2);
    }

    #[test]
    fn dropout_mask_different_seeds_differ() {
        let d = EmbedDropout::new(0.5);
        let m1 = d.generate_mask(64, 1);
        let m2 = d.generate_mask(64, 2);
        assert_ne!(m1, m2);
    }

    #[test]
    fn dropout_rejects_short_mask() {
        let d = EmbedDropout::new(0.1);
        let mut data = vec![1.0; 4];
        let mask = vec![true, false]; // too short for embed_dim=4
        assert!(d.apply_mask(&mut data, 1, 4, &mask).is_err());
    }

    #[test]
    fn dropout_rejects_short_data() {
        let d = EmbedDropout::new(0.1);
        let mut data = vec![1.0; 3]; // too short for 2*2
        let mask = vec![true; 2];
        assert!(d.apply_mask(&mut data, 2, 2, &mask).is_err());
    }

    // ── EmbeddingInitializer ─────────────────────────────────

    #[test]
    fn initializer_gguf_correct_size() {
        let data = vec![1.0; 12];
        let init = EmbeddingInitializer::Gguf(data.clone());
        let w = init.initialize(3, 4).unwrap();
        assert_eq!(w, data);
    }

    #[test]
    fn initializer_gguf_wrong_size_rejected() {
        let init = EmbeddingInitializer::Gguf(vec![1.0; 10]);
        assert!(init.initialize(3, 4).is_err());
    }

    #[test]
    fn initializer_zeros() {
        let init = EmbeddingInitializer::Zeros;
        let w = init.initialize(2, 3).unwrap();
        assert_eq!(w, vec![0.0; 6]);
    }

    #[test]
    fn initializer_random_normal_correct_length() {
        let init = EmbeddingInitializer::RandomNormal { std_dev: 0.02, seed: 42 };
        let w = init.initialize(10, 8).unwrap();
        assert_eq!(w.len(), 80);
    }

    #[test]
    fn initializer_random_normal_finite() {
        let init = EmbeddingInitializer::RandomNormal { std_dev: 0.1, seed: 123 };
        let w = init.initialize(50, 64).unwrap();
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn initializer_random_deterministic() {
        let init = EmbeddingInitializer::RandomNormal { std_dev: 0.02, seed: 7 };
        let w1 = init.initialize(4, 4).unwrap();
        let w2 = init.initialize(4, 4).unwrap();
        assert_eq!(w1, w2);
    }

    // ── Sinusoidal position encoding ─────────────────────────

    #[test]
    fn sinusoidal_adds_to_embedding() {
        let mut emb = vec![0.0; 4]; // 1 token, dim=4
        add_sinusoidal_position_ref(&mut emb, 1, 4, 0);
        // Position 0: sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1
        assert!(emb[0].abs() < 1e-6); // sin(0) = 0
        assert!((emb[1] - 1.0).abs() < 1e-6); // cos(0) = 1
    }

    #[test]
    fn sinusoidal_different_positions_differ() {
        let mut emb0 = vec![0.0; 4];
        let mut emb1 = vec![0.0; 4];
        add_sinusoidal_position_ref(&mut emb0, 1, 4, 0);
        add_sinusoidal_position_ref(&mut emb1, 1, 4, 1);
        assert_ne!(emb0, emb1);
    }

    #[test]
    fn sinusoidal_with_offset() {
        let mut emb_a = vec![0.0; 4];
        add_sinusoidal_position_ref(&mut emb_a, 1, 4, 5);
        let mut emb_b = vec![0.0; 8]; // 2 tokens, offset 4
        add_sinusoidal_position_ref(&mut emb_b, 2, 4, 4);
        // Token at offset 5 should match the second token of batch with
        // offset 4
        assert_eq!(&emb_a[..4], &emb_b[4..8]);
    }

    // ── Learned position encoding ────────────────────────────

    #[test]
    fn learned_position_adds_correctly() {
        let pos_w = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
        ];
        let mut emb = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        add_learned_position_ref(&mut emb, &pos_w, 2, 2, 0).unwrap();
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[1] - 2.2).abs() < 1e-6);
        assert!((emb[2] - 3.3).abs() < 1e-6);
        assert!((emb[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn learned_position_rejects_overflow() {
        let pos_w = vec![0.0; 4]; // max_pos = 2
        let mut emb = vec![0.0; 4];
        assert!(add_learned_position_ref(&mut emb, &pos_w, 2, 2, 1).is_err());
    }

    // ── EmbeddingManager ─────────────────────────────────────

    fn make_test_manager(quantize: bool) -> EmbeddingManager {
        let weight = vec![
            1.0, 0.0, // tok 0
            0.0, 1.0, // tok 1
            0.5, 0.5, // tok 2
        ];
        let cfg = TokenEmbedConfig::new(3, 2, 16).with_quantize(quantize);
        EmbeddingManager::new(weight, cfg, PositionEncoding::Sinusoidal).unwrap()
    }

    #[test]
    fn manager_basic_lookup() {
        let mgr = make_test_manager(false);
        let mut out = vec![0.0; 2];
        mgr.forward(&[1], 0, None, None, &mut out).unwrap();
        // Token 1 = [0, 1], plus sinusoidal at pos 0 = [sin(0), cos(0)]
        // = [0+0, 1+1] = [0, 2]
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn manager_quantized_lookup() {
        let mgr = make_test_manager(true);
        let mut out = vec![0.0; 2];
        mgr.forward(&[0], 0, None, None, &mut out).unwrap();
        // Quantized [1, 0] should be close to [1, 0] + sinusoidal[0]
        assert!((out[0] - 1.0).abs() < 0.1);
        assert!(out[1].abs() < 0.1 + 1.0 + 0.01); // cos(0) = 1
    }

    #[test]
    fn manager_with_learned_position() {
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let pos_w = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
        ];
        let cfg = TokenEmbedConfig::new(2, 2, 2);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::Learned(pos_w)).unwrap();
        let mut out = vec![0.0; 4];
        mgr.forward(&[0, 1], 0, None, None, &mut out).unwrap();
        assert!((out[0] - 1.1).abs() < 1e-6);
        assert!((out[1] - 2.2).abs() < 1e-6);
        assert!((out[2] - 3.3).abs() < 1e-6);
        assert!((out[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn manager_rope_no_position_added() {
        let weight = vec![1.0, 2.0]; // vocab=1, dim=2
        let cfg = TokenEmbedConfig::new(1, 2, 8);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::RoPE).unwrap();
        let mut out = vec![0.0; 2];
        mgr.forward(&[0], 0, None, None, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0]); // no position added
    }

    #[test]
    fn manager_alibi_no_position_added() {
        let weight = vec![5.0, 6.0]; // vocab=1, dim=2
        let cfg = TokenEmbedConfig::new(1, 2, 8);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::ALiBi).unwrap();
        let mut out = vec![0.0; 2];
        mgr.forward(&[0], 0, None, None, &mut out).unwrap();
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    fn manager_with_scaler() {
        let weight = vec![1.0, 1.0]; // vocab=1, dim=2
        let cfg = TokenEmbedConfig::new(1, 2, 8);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::RoPE).unwrap().with_scaler();
        let mut out = vec![0.0; 2];
        mgr.forward(&[0], 0, None, None, &mut out).unwrap();
        let expected = (2.0f32).sqrt();
        assert!((out[0] - expected).abs() < 1e-6);
        assert!((out[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn manager_with_dropout() {
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let cfg = TokenEmbedConfig::new(2, 2, 8);
        let mgr =
            EmbeddingManager::new(weight, cfg, PositionEncoding::RoPE).unwrap().with_dropout(0.5);
        let mut out = vec![0.0; 2];
        mgr.forward(&[0], 0, None, Some(42), &mut out).unwrap();
        // At least one dimension may be zeroed
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn manager_with_segments() {
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let seg_w = vec![
            0.1, 0.2, // segment 0
            0.3, 0.4, // segment 1
        ];
        let cfg = TokenEmbedConfig::new(2, 2, 8);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::RoPE)
            .unwrap()
            .with_segments(seg_w, 2)
            .unwrap();
        let mut out = vec![0.0; 4];
        mgr.forward(&[0, 1], 0, Some(&[0, 1]), None, &mut out).unwrap();
        // tok0 + seg0: [1.1, 2.2]
        assert!((out[0] - 1.1).abs() < 1e-6);
        assert!((out[1] - 2.2).abs() < 1e-6);
        // tok1 + seg1: [3.3, 4.4]
        assert!((out[2] - 3.3).abs() < 1e-6);
        assert!((out[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn manager_rejects_wrong_weight_size() {
        let cfg = TokenEmbedConfig::new(2, 3, 8);
        assert!(EmbeddingManager::new(vec![0.0; 5], cfg, PositionEncoding::RoPE).is_err());
    }

    #[test]
    fn manager_rejects_short_output() {
        let mgr = make_test_manager(false);
        let mut out = vec![0.0; 1]; // too small
        assert!(mgr.forward(&[0], 0, None, None, &mut out).is_err());
    }

    #[test]
    fn manager_segment_rejects_wrong_size() {
        let weight = vec![1.0; 4];
        let cfg = TokenEmbedConfig::new(2, 2, 8);
        let mgr = EmbeddingManager::new(weight, cfg, PositionEncoding::RoPE).unwrap();
        assert!(mgr.with_segments(vec![0.0; 3], 2).is_err());
    }

    #[test]
    fn manager_token_weight_accessor() {
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = TokenEmbedConfig::new(2, 2, 8);
        let mgr = EmbeddingManager::new(weight.clone(), cfg, PositionEncoding::RoPE).unwrap();
        assert_eq!(mgr.token_weight(), &weight[..]);
    }

    #[test]
    fn manager_position_encoding_accessor() {
        let mgr = make_test_manager(false);
        assert_eq!(*mgr.position_encoding(), PositionEncoding::Sinusoidal);
    }

    // ── tied_output_projection_ref ───────────────────────────

    #[test]
    fn projection_identity() {
        let hidden = vec![1.0, 0.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity 2×2
        let mut out = vec![0.0; 2];
        tied_output_projection_ref(&hidden, &weight, &mut out, 1, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn projection_matmul_correctness() {
        let hidden = vec![1.0, 2.0, 3.0]; // 1×3
        let weight = vec![
            1.0, 0.0, 0.0, // vocab 0
            0.0, 1.0, 0.0, // vocab 1
            0.0, 0.0, 1.0, // vocab 2
            1.0, 1.0, 1.0, // vocab 3
        ];
        let mut out = vec![0.0; 4];
        tied_output_projection_ref(&hidden, &weight, &mut out, 1, 3, 4).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn projection_rejects_short_hidden() {
        let mut out = vec![0.0; 2];
        assert!(tied_output_projection_ref(&[1.0], &[1.0; 4], &mut out, 1, 2, 2).is_err());
    }

    #[test]
    fn projection_rejects_short_weight() {
        let mut out = vec![0.0; 2];
        assert!(tied_output_projection_ref(&[1.0, 2.0], &[1.0], &mut out, 1, 2, 2).is_err());
    }

    #[test]
    fn projection_zero_hidden_yields_zero() {
        let hidden = vec![0.0; 4]; // seq=2, dim=2
        let weight = vec![1.0; 6]; // vocab=3, dim=2
        let mut out = vec![99.0; 6];
        tied_output_projection_ref(&hidden, &weight, &mut out, 2, 2, 3).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── Property-style tests ─────────────────────────────────

    #[test]
    fn lookup_always_returns_embed_dim_vector() {
        let vocab = 8;
        let dim = 5;
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        for tok in 0..vocab as u32 + 2 {
            let mut out = vec![0.0; dim];
            batched_embedding_lookup_ref(&[tok], &weight, &mut out, vocab, dim).unwrap();
            assert_eq!(out.len(), dim);
        }
    }

    #[test]
    fn quantized_lookup_always_returns_embed_dim() {
        let weight: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let q = QuantizedEmbedding::from_float(&weight, 4, 5).unwrap();
        for tok in 0..6u32 {
            let mut out = vec![0.0; 5];
            q.lookup(&[tok], &mut out).unwrap();
            assert_eq!(out.len(), 5);
        }
    }

    #[test]
    fn sinusoidal_is_finite() {
        let mut emb = vec![0.0; 128]; // 4 tokens, dim=32
        add_sinusoidal_position_ref(&mut emb, 4, 32, 0);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_outputs_finite_full_pipeline() {
        let weight: Vec<f32> = (0..200).map(|i| (i as f32) * 0.01).collect();
        let cfg = TokenEmbedConfig::new(10, 20, 64);
        let mgr =
            EmbeddingManager::new(weight, cfg, PositionEncoding::Sinusoidal).unwrap().with_scaler();
        let ids: Vec<u32> = (0..10).collect();
        let mut out = vec![f32::NAN; 200];
        mgr.forward(&ids, 0, None, None, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batch_position_encoding_consistency() {
        // Running 3 tokens at offset 0 should give the same position
        // values as running 1 token at offsets 0, 1, 2.
        let dim = 8;
        let mut batch = vec![0.0; 3 * dim];
        add_sinusoidal_position_ref(&mut batch, 3, dim, 0);
        for pos in 0..3 {
            let mut single = vec![0.0; dim];
            add_sinusoidal_position_ref(&mut single, 1, dim, pos);
            assert_eq!(
                &batch[pos * dim..(pos + 1) * dim],
                &single[..],
                "mismatch at position {pos}"
            );
        }
    }
}
