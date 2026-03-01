//! Embedding layer for token and position encodings.
//!
//! Provides a composable embedding pipeline for transformer-based inference:
//!
//! - **[`EmbeddingConfig`]** — configuration for vocab size, dimensions, etc.
//! - **[`TokenEmbedding`]** — lookup table mapping token IDs to dense vectors
//! - **[`PositionEmbedding`]** — sinusoidal, learned, `RoPE`, `ALiBi`, or none
//! - **[`EmbeddingTable`]** — weight matrix storage with efficient lookup
//! - **[`EmbeddingCombiner`]** — combines token + position embeddings
//! - **[`EmbeddingNorm`]** — layer normalization on combined embeddings
//! - **[`EmbeddingDropout`]** — dropout (passthrough during inference)
//! - **[`EmbeddingProjection`]** — projects embeddings to a different dimension
//! - **[`EmbeddingLayer`]** — top-level orchestrator

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by embedding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    /// Token ID exceeds the vocabulary size.
    OutOfVocab { token_id: u32, vocab_size: usize },
    /// Weight dimensions do not match the expected shape.
    DimensionMismatch { expected: usize, actual: usize },
    /// Sequence exceeds the maximum allowed length.
    SequenceTooLong { len: usize, max: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfVocab { token_id, vocab_size } => {
                write!(f, "token id {token_id} out of vocabulary (size {vocab_size})")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::SequenceTooLong { len, max } => {
                write!(f, "sequence length {len} exceeds maximum {max}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

// ---------------------------------------------------------------------------
// EmbeddingConfig
// ---------------------------------------------------------------------------

/// Configuration for an embedding layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingConfig {
    /// Vocabulary size (number of distinct tokens).
    pub vocab_size: usize,
    /// Embedding dimension (width of each token vector).
    pub embedding_dim: usize,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Optional padding token index (embeddings are zeroed).
    pub padding_idx: Option<u32>,
    /// Scale factor applied after lookup (often `sqrt(embedding_dim)`).
    pub scale_factor: f32,
    /// Position encoding variant.
    pub position_encoding: PositionEncodingType,
}

impl EmbeddingConfig {
    /// Validate the config, returning an error on nonsensical values.
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.vocab_size == 0 {
            return Err(EmbeddingError::InvalidConfig("vocab_size must be > 0".into()));
        }
        if self.embedding_dim == 0 {
            return Err(EmbeddingError::InvalidConfig("embedding_dim must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(EmbeddingError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        if let Some(pad) = self.padding_idx
            && pad as usize >= self.vocab_size
        {
            return Err(EmbeddingError::InvalidConfig(format!(
                "padding_idx {pad} >= vocab_size {}",
                self.vocab_size
            )));
        }
        Ok(())
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            embedding_dim: 256,
            max_seq_len: 2048,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::Sinusoidal,
        }
    }
}

// ---------------------------------------------------------------------------
// PositionEncodingType
// ---------------------------------------------------------------------------

/// Variants of position encoding supported by the embedding layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionEncodingType {
    /// Learned (trained) position embeddings.
    Learned,
    /// Classic sinusoidal encoding (Vaswani et al., 2017).
    Sinusoidal,
    /// Rotary position embedding — applied during attention, not additive.
    RoPE,
    /// Attention with Linear Biases — bias is added in attention, not here.
    ALiBi,
    /// No positional encoding.
    NoPE,
}

// ---------------------------------------------------------------------------
// EmbeddingTable
// ---------------------------------------------------------------------------

/// Contiguous weight matrix of shape `[rows × cols]` with row-based lookup.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    weights: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl EmbeddingTable {
    /// Create a new table. `weights` must have `rows * cols` elements.
    pub fn new(weights: Vec<f32>, rows: usize, cols: usize) -> Result<Self, EmbeddingError> {
        let expected = rows * cols;
        if weights.len() != expected {
            return Err(EmbeddingError::DimensionMismatch { expected, actual: weights.len() });
        }
        Ok(Self { weights, rows, cols })
    }

    /// Create a zero-initialised table.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { weights: vec![0.0; rows * cols], rows, cols }
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow the raw weight buffer.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Look up a single row by index.
    pub fn row(&self, idx: usize) -> Result<&[f32], EmbeddingError> {
        if idx >= self.rows {
            return Err(EmbeddingError::DimensionMismatch { expected: self.rows, actual: idx });
        }
        let start = idx * self.cols;
        Ok(&self.weights[start..start + self.cols])
    }
}

// ---------------------------------------------------------------------------
// TokenEmbedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table mapping token IDs to dense vectors.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    table: EmbeddingTable,
    padding_idx: Option<u32>,
    scale_factor: f32,
}

impl TokenEmbedding {
    /// Create a new token embedding.
    pub fn new(
        weights: Vec<f32>,
        vocab_size: usize,
        embedding_dim: usize,
        padding_idx: Option<u32>,
        scale_factor: f32,
    ) -> Result<Self, EmbeddingError> {
        let table = EmbeddingTable::new(weights, vocab_size, embedding_dim)?;
        Ok(Self { table, padding_idx, scale_factor })
    }

    /// Vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.table.rows()
    }

    /// Embedding dimension.
    #[must_use]
    pub const fn embedding_dim(&self) -> usize {
        self.table.cols()
    }

    /// Look up a single token, returning a scaled vector (zero for padding).
    pub fn embed_single(&self, token_id: u32) -> Result<Vec<f32>, EmbeddingError> {
        let id = token_id as usize;
        if id >= self.table.rows() {
            return Err(EmbeddingError::OutOfVocab { token_id, vocab_size: self.table.rows() });
        }
        if self.padding_idx == Some(token_id) {
            return Ok(vec![0.0; self.table.cols()]);
        }
        let row = self.table.row(id)?;
        Ok(row.iter().map(|&v| v * self.scale_factor).collect())
    }

    /// Batch embedding lookup. Returns `[seq_len × embedding_dim]` flat.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        let dim = self.table.cols();
        let mut out = Vec::with_capacity(token_ids.len() * dim);
        for &id in token_ids {
            let v = self.embed_single(id)?;
            out.extend_from_slice(&v);
        }
        Ok(out)
    }

    /// Borrow the underlying weight table.
    #[must_use]
    pub const fn table(&self) -> &EmbeddingTable {
        &self.table
    }
}

// ---------------------------------------------------------------------------
// PositionEmbedding
// ---------------------------------------------------------------------------

/// Position embedding that produces per-position vectors to be added to
/// token embeddings.
#[derive(Debug, Clone)]
pub struct PositionEmbedding {
    variant: PositionEncodingType,
    /// Learned weights (only populated for [`PositionEncodingType::Learned`]).
    learned_table: Option<EmbeddingTable>,
    /// `RoPE` base frequency.
    rope_base: f32,
    /// `ALiBi` slope (per-head; stored as a single default here).
    alibi_slope: f32,
}

impl PositionEmbedding {
    /// Create a sinusoidal position embedding (no learned weights needed).
    #[must_use]
    pub const fn sinusoidal() -> Self {
        Self {
            variant: PositionEncodingType::Sinusoidal,
            learned_table: None,
            rope_base: 10_000.0,
            alibi_slope: 0.0,
        }
    }

    /// Create a learned position embedding from pre-trained weights.
    pub fn learned(
        weights: Vec<f32>,
        max_positions: usize,
        embedding_dim: usize,
    ) -> Result<Self, EmbeddingError> {
        let table = EmbeddingTable::new(weights, max_positions, embedding_dim)?;
        Ok(Self {
            variant: PositionEncodingType::Learned,
            learned_table: Some(table),
            rope_base: 10_000.0,
            alibi_slope: 0.0,
        })
    }

    /// Create a `RoPE` position embedding placeholder.
    #[must_use]
    pub const fn rope(base: f32) -> Self {
        Self {
            variant: PositionEncodingType::RoPE,
            learned_table: None,
            rope_base: base,
            alibi_slope: 0.0,
        }
    }

    /// Create an `ALiBi` position embedding placeholder.
    #[must_use]
    pub const fn alibi(slope: f32) -> Self {
        Self {
            variant: PositionEncodingType::ALiBi,
            learned_table: None,
            rope_base: 10_000.0,
            alibi_slope: slope,
        }
    }

    /// Create a no-op position embedding.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            variant: PositionEncodingType::NoPE,
            learned_table: None,
            rope_base: 10_000.0,
            alibi_slope: 0.0,
        }
    }

    /// Which variant this is.
    #[must_use]
    pub const fn variant(&self) -> &PositionEncodingType {
        &self.variant
    }

    /// `RoPE` base frequency.
    #[must_use]
    pub const fn rope_base(&self) -> f32 {
        self.rope_base
    }

    /// `ALiBi` slope.
    #[must_use]
    pub const fn alibi_slope(&self) -> f32 {
        self.alibi_slope
    }

    /// Encode positions, returning a flat buffer of `positions.len() * dim`.
    ///
    /// For `RoPE` and `ALiBi` this returns zeros (they are applied
    /// elsewhere in the attention layer).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn encode(&self, positions: &[usize], dim: usize) -> Vec<f32> {
        match &self.variant {
            PositionEncodingType::Sinusoidal => sinusoidal_encode(positions, dim),
            PositionEncodingType::Learned => self.learned_table.as_ref().map_or_else(
                || vec![0.0; positions.len() * dim],
                |table| learned_encode(table, positions, dim),
            ),
            PositionEncodingType::RoPE
            | PositionEncodingType::ALiBi
            | PositionEncodingType::NoPE => {
                vec![0.0; positions.len() * dim]
            }
        }
    }
}

/// Compute sinusoidal positional encoding.
///
/// `PE(pos, 2i)   = sin(pos / 10000^(2i/d))`
/// `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
#[allow(clippy::cast_precision_loss)]
fn sinusoidal_encode(positions: &[usize], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * dim);
    let d = dim as f32;
    for &pos in positions {
        let p = pos as f32;
        for i in 0..dim {
            let dim_pair = (i / 2) as f32;
            let angle = p / (10_000_f32).powf(2.0 * dim_pair / d);
            if i % 2 == 0 {
                out.push(angle.sin());
            } else {
                out.push(angle.cos());
            }
        }
    }
    out
}

/// Look up learned positional encodings. Positions beyond the table size
/// are clamped to the last row.
fn learned_encode(table: &EmbeddingTable, positions: &[usize], dim: usize) -> Vec<f32> {
    let table_dim = table.cols();
    let copy_dim = dim.min(table_dim);
    let mut out = Vec::with_capacity(positions.len() * dim);
    for &pos in positions {
        let clamped = pos.min(table.rows().saturating_sub(1));
        let row = table.row(clamped).unwrap_or(&[]);
        for i in 0..dim {
            if i < copy_dim && i < row.len() {
                out.push(row[i]);
            } else {
                out.push(0.0);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CombineMode
// ---------------------------------------------------------------------------

/// How token and position embeddings are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMode {
    /// Element-wise addition (most common).
    Add,
    /// Concatenation along the feature dimension.
    Concat,
}

// ---------------------------------------------------------------------------
// EmbeddingCombiner
// ---------------------------------------------------------------------------

/// Combines token embeddings with position embeddings.
#[derive(Debug, Clone)]
pub struct EmbeddingCombiner {
    mode: CombineMode,
}

impl EmbeddingCombiner {
    /// Create a new combiner.
    #[must_use]
    pub const fn new(mode: CombineMode) -> Self {
        Self { mode }
    }

    /// Combine mode in use.
    #[must_use]
    pub const fn mode(&self) -> CombineMode {
        self.mode
    }

    /// Output dimension per position given the input `embedding_dim`.
    #[must_use]
    pub const fn output_dim(&self, embedding_dim: usize) -> usize {
        match self.mode {
            CombineMode::Add => embedding_dim,
            CombineMode::Concat => embedding_dim * 2,
        }
    }

    /// Combine token and position flat buffers.
    ///
    /// Both must have `seq_len * embedding_dim` elements.
    pub fn combine(
        &self,
        token_embeds: &[f32],
        position_embeds: &[f32],
        seq_len: usize,
        embedding_dim: usize,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let expected = seq_len * embedding_dim;
        if token_embeds.len() != expected {
            return Err(EmbeddingError::DimensionMismatch { expected, actual: token_embeds.len() });
        }
        if position_embeds.len() != expected {
            return Err(EmbeddingError::DimensionMismatch {
                expected,
                actual: position_embeds.len(),
            });
        }
        match self.mode {
            CombineMode::Add => {
                Ok(token_embeds.iter().zip(position_embeds.iter()).map(|(a, b)| a + b).collect())
            }
            CombineMode::Concat => {
                let out_dim = embedding_dim * 2;
                let mut out = Vec::with_capacity(seq_len * out_dim);
                for s in 0..seq_len {
                    let offset = s * embedding_dim;
                    out.extend_from_slice(&token_embeds[offset..offset + embedding_dim]);
                    out.extend_from_slice(&position_embeds[offset..offset + embedding_dim]);
                }
                Ok(out)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingNorm
// ---------------------------------------------------------------------------

/// Layer normalization applied to embedding vectors.
///
/// Normalizes each position vector independently:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`
#[derive(Debug, Clone)]
pub struct EmbeddingNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl EmbeddingNorm {
    /// Create a new layer norm with given scale (`gamma`) and bias (`beta`).
    pub fn new(gamma: Vec<f32>, beta: Vec<f32>, eps: f32) -> Result<Self, EmbeddingError> {
        if gamma.len() != beta.len() {
            return Err(EmbeddingError::DimensionMismatch {
                expected: gamma.len(),
                actual: beta.len(),
            });
        }
        Ok(Self { gamma, beta, eps })
    }

    /// Create an identity norm (gamma=1, beta=0).
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        Self { gamma: vec![1.0; dim], beta: vec![0.0; dim], eps: 1e-5 }
    }

    /// The feature dimension this norm operates on.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// Normalise a flat buffer of `seq_len * dim` elements in-place,
    /// returning the result.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn normalize(&self, data: &[f32], dim: usize) -> Vec<f32> {
        let seq_len = data.len() / dim;
        let mut out = Vec::with_capacity(data.len());
        for s in 0..seq_len {
            let offset = s * dim;
            let slice = &data[offset..offset + dim];
            let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for (i, &val) in slice.iter().enumerate().take(dim) {
                let normed = (val - mean) * inv_std;
                let g = if i < self.gamma.len() { self.gamma[i] } else { 1.0 };
                let b = if i < self.beta.len() { self.beta[i] } else { 0.0 };
                out.push(normed.mul_add(g, b));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// EmbeddingDropout
// ---------------------------------------------------------------------------

/// Dropout layer for embeddings.
///
/// During training, randomly zeroes elements with probability `p`.
/// During inference (the default), this is a passthrough.
#[derive(Debug, Clone)]
pub struct EmbeddingDropout {
    p: f32,
    training: bool,
}

impl EmbeddingDropout {
    /// Create a new dropout layer.
    #[must_use]
    pub const fn new(p: f32, training: bool) -> Self {
        Self { p, training }
    }

    /// Inference-mode dropout (always passthrough).
    #[must_use]
    pub const fn inference() -> Self {
        Self { p: 0.0, training: false }
    }

    /// Dropout probability.
    #[must_use]
    pub const fn p(&self) -> f32 {
        self.p
    }

    /// Whether in training mode.
    #[must_use]
    pub const fn is_training(&self) -> bool {
        self.training
    }

    /// Apply dropout. In inference mode, returns the input unchanged.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    pub fn apply(&self, data: &[f32]) -> Vec<f32> {
        if !self.training || self.p == 0.0 {
            return data.to_vec();
        }
        // Deterministic "pseudo-dropout" for reproducibility in tests:
        // zero every element whose index hashes below the threshold.
        let threshold = (self.p * u32::MAX as f32) as u32;
        data.iter()
            .enumerate()
            .map(|(i, &v)| {
                let hash = simple_hash(i as u32);
                if hash < threshold { 0.0 } else { v / (1.0 - self.p) }
            })
            .collect()
    }
}

/// Cheap deterministic hash for reproducible dropout.
const fn simple_hash(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x
}

// ---------------------------------------------------------------------------
// EmbeddingProjection
// ---------------------------------------------------------------------------

/// Linear projection from one dimension to another.
///
/// Computes `y = x @ W^T + bias` per position.
#[derive(Debug, Clone)]
pub struct EmbeddingProjection {
    weights: Vec<f32>,
    bias: Option<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
}

impl EmbeddingProjection {
    /// Create a new projection.
    ///
    /// `weights` has shape `[output_dim × input_dim]` (row-major).
    pub fn new(
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, EmbeddingError> {
        let expected = output_dim * input_dim;
        if weights.len() != expected {
            return Err(EmbeddingError::DimensionMismatch { expected, actual: weights.len() });
        }
        if let Some(ref b) = bias
            && b.len() != output_dim
        {
            return Err(EmbeddingError::DimensionMismatch {
                expected: output_dim,
                actual: b.len(),
            });
        }
        Ok(Self { weights, bias, input_dim, output_dim })
    }

    /// Create an identity-like projection (`input_dim == output_dim`, W = I).
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        let mut weights = vec![0.0; dim * dim];
        for i in 0..dim {
            weights[i * dim + i] = 1.0;
        }
        Self { weights, bias: None, input_dim: dim, output_dim: dim }
    }

    /// Input dimension.
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Output dimension.
    #[must_use]
    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Project a flat buffer of `seq_len * input_dim` to `seq_len * output_dim`.
    pub fn project(&self, data: &[f32], seq_len: usize) -> Result<Vec<f32>, EmbeddingError> {
        let expected = seq_len * self.input_dim;
        if data.len() != expected {
            return Err(EmbeddingError::DimensionMismatch { expected, actual: data.len() });
        }
        let mut out = Vec::with_capacity(seq_len * self.output_dim);
        for s in 0..seq_len {
            let in_offset = s * self.input_dim;
            let input = &data[in_offset..in_offset + self.input_dim];
            for o in 0..self.output_dim {
                let row_start = o * self.input_dim;
                let dot: f32 = self.weights[row_start..row_start + self.input_dim]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                let val = self.bias.as_ref().map_or(dot, |b| dot + b[o]);
                out.push(val);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// EmbeddingLayer (orchestrator)
// ---------------------------------------------------------------------------

/// Top-level embedding layer orchestrator.
///
/// Pipeline: `token_embed → position_embed → combine → norm → dropout → project`
#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    config: EmbeddingConfig,
    token_embedding: TokenEmbedding,
    position_embedding: PositionEmbedding,
    combiner: EmbeddingCombiner,
    norm: EmbeddingNorm,
    dropout: EmbeddingDropout,
    projection: Option<EmbeddingProjection>,
}

impl EmbeddingLayer {
    /// Create a new embedding layer from components.
    #[must_use]
    pub const fn new(
        config: EmbeddingConfig,
        token_embedding: TokenEmbedding,
        position_embedding: PositionEmbedding,
        combiner: EmbeddingCombiner,
        norm: EmbeddingNorm,
        dropout: EmbeddingDropout,
        projection: Option<EmbeddingProjection>,
    ) -> Self {
        Self { config, token_embedding, position_embedding, combiner, norm, dropout, projection }
    }

    /// Borrow the config.
    #[must_use]
    pub const fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Borrow the token embedding.
    #[must_use]
    pub const fn token_embedding(&self) -> &TokenEmbedding {
        &self.token_embedding
    }

    /// Borrow the position embedding.
    #[must_use]
    pub const fn position_embedding(&self) -> &PositionEmbedding {
        &self.position_embedding
    }

    /// Output dimension per position after the full pipeline.
    #[must_use]
    pub const fn output_dim(&self) -> usize {
        match &self.projection {
            Some(proj) => proj.output_dim(),
            None => self.combiner.output_dim(self.config.embedding_dim),
        }
    }

    /// Run the full embedding pipeline on a token sequence.
    ///
    /// Returns a flat `f32` buffer of shape `[seq_len × output_dim]`.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        let seq_len = token_ids.len();
        if seq_len > self.config.max_seq_len {
            return Err(EmbeddingError::SequenceTooLong {
                len: seq_len,
                max: self.config.max_seq_len,
            });
        }
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // 1. Token embeddings
        let tok = self.token_embedding.embed(token_ids)?;

        // 2. Position embeddings
        let positions: Vec<usize> = (0..seq_len).collect();
        let pos = self.position_embedding.encode(&positions, self.config.embedding_dim);

        // 3. Combine
        let combined = self.combiner.combine(&tok, &pos, seq_len, self.config.embedding_dim)?;

        // 4. Layer norm
        let combined_dim = self.combiner.output_dim(self.config.embedding_dim);
        let normed = self.norm.normalize(&combined, combined_dim);

        // 5. Dropout
        let dropped = self.dropout.apply(&normed);

        // 6. Optional projection
        if let Some(ref proj) = self.projection {
            proj.project(&dropped, seq_len)
        } else {
            Ok(dropped)
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

/// Convenience builder for creating a default embedding layer from weights.
pub fn build_default_layer(
    weights: Vec<f32>,
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_len: usize,
) -> Result<EmbeddingLayer, EmbeddingError> {
    let config = EmbeddingConfig {
        vocab_size,
        embedding_dim,
        max_seq_len,
        padding_idx: None,
        scale_factor: 1.0,
        position_encoding: PositionEncodingType::Sinusoidal,
    };
    config.validate()?;
    let token_embedding = TokenEmbedding::new(weights, vocab_size, embedding_dim, None, 1.0)?;
    let position_embedding = PositionEmbedding::sinusoidal();
    let combiner = EmbeddingCombiner::new(CombineMode::Add);
    let norm = EmbeddingNorm::identity(embedding_dim);
    let dropout = EmbeddingDropout::inference();
    Ok(EmbeddingLayer::new(
        config,
        token_embedding,
        position_embedding,
        combiner,
        norm,
        dropout,
        None,
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers ---

    /// Create simple sequential weights `[0.0, 1.0, 2.0, ...]`.
    fn seq_weights(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    /// Create uniform weights.
    fn uniform_weights(n: usize, val: f32) -> Vec<f32> {
        vec![val; n]
    }

    // ===================================================================
    // EmbeddingConfig tests
    // ===================================================================

    #[test]
    fn config_default_is_valid() {
        let cfg = EmbeddingConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_vocab_invalid() {
        let cfg = EmbeddingConfig { vocab_size: 0, ..EmbeddingConfig::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_dim_invalid() {
        let cfg = EmbeddingConfig { embedding_dim: 0, ..EmbeddingConfig::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_seq_len_invalid() {
        let cfg = EmbeddingConfig { max_seq_len: 0, ..EmbeddingConfig::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_padding_out_of_range_invalid() {
        let cfg = EmbeddingConfig {
            vocab_size: 100,
            padding_idx: Some(100),
            ..EmbeddingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_padding_in_range_valid() {
        let cfg = EmbeddingConfig {
            vocab_size: 100,
            padding_idx: Some(99),
            ..EmbeddingConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_default_scale_factor() {
        let cfg = EmbeddingConfig::default();
        assert!((cfg.scale_factor - 1.0).abs() < f32::EPSILON);
    }

    // ===================================================================
    // PositionEncodingType tests
    // ===================================================================

    #[test]
    fn encoding_type_clone_eq() {
        assert_eq!(PositionEncodingType::Sinusoidal, PositionEncodingType::Sinusoidal);
        assert_ne!(PositionEncodingType::RoPE, PositionEncodingType::ALiBi);
        assert_ne!(PositionEncodingType::Learned, PositionEncodingType::NoPE);
    }

    // ===================================================================
    // EmbeddingTable tests
    // ===================================================================

    #[test]
    fn table_creation_valid() {
        let t = EmbeddingTable::new(seq_weights(12), 3, 4).unwrap();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 4);
    }

    #[test]
    fn table_creation_dimension_mismatch() {
        assert!(EmbeddingTable::new(seq_weights(10), 3, 4).is_err());
    }

    #[test]
    fn table_zeros() {
        let t = EmbeddingTable::zeros(2, 3);
        assert_eq!(t.weights(), &[0.0; 6]);
    }

    #[test]
    fn table_row_lookup() {
        let t = EmbeddingTable::new(seq_weights(12), 3, 4).unwrap();
        assert_eq!(t.row(0).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(t.row(1).unwrap(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(t.row(2).unwrap(), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn table_row_out_of_bounds() {
        let t = EmbeddingTable::new(seq_weights(12), 3, 4).unwrap();
        assert!(t.row(3).is_err());
    }

    // ===================================================================
    // TokenEmbedding tests
    // ===================================================================

    #[test]
    fn token_embed_single() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        assert_eq!(te.embed_single(1).unwrap(), vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn token_embed_first_token() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        assert_eq!(te.embed_single(0).unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn token_embed_last_token() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        assert_eq!(te.embed_single(2).unwrap(), vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn token_embed_out_of_vocab() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        assert!(te.embed_single(3).is_err());
    }

    #[test]
    fn token_embed_out_of_vocab_large_id() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        assert!(te.embed_single(u32::MAX).is_err());
    }

    #[test]
    fn token_embed_padding_returns_zeros() {
        let te = TokenEmbedding::new(uniform_weights(12, 5.0), 3, 4, Some(1), 1.0).unwrap();
        assert_eq!(te.embed_single(1).unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn token_embed_non_padding_is_nonzero() {
        let te = TokenEmbedding::new(uniform_weights(12, 5.0), 3, 4, Some(1), 1.0).unwrap();
        assert_eq!(te.embed_single(0).unwrap(), vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn token_embed_scale_factor() {
        let te = TokenEmbedding::new(uniform_weights(12, 2.0), 3, 4, None, 3.0).unwrap();
        assert_eq!(te.embed_single(0).unwrap(), vec![6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn token_embed_batch() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        let result = te.embed(&[0, 2]).unwrap();
        assert_eq!(result.len(), 8);
        assert_eq!(&result[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&result[4..], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn token_embed_empty_batch() {
        let te = TokenEmbedding::new(seq_weights(12), 3, 4, None, 1.0).unwrap();
        let result = te.embed(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn token_embed_dimension_mismatch() {
        assert!(TokenEmbedding::new(seq_weights(10), 3, 4, None, 1.0).is_err());
    }

    #[test]
    fn token_embed_vocab_size_accessor() {
        let te = TokenEmbedding::new(seq_weights(20), 5, 4, None, 1.0).unwrap();
        assert_eq!(te.vocab_size(), 5);
    }

    #[test]
    fn token_embed_dim_accessor() {
        let te = TokenEmbedding::new(seq_weights(20), 5, 4, None, 1.0).unwrap();
        assert_eq!(te.embedding_dim(), 4);
    }

    // ===================================================================
    // PositionEmbedding – sinusoidal tests
    // ===================================================================

    #[test]
    fn sinusoidal_shape() {
        let pe = PositionEmbedding::sinusoidal();
        let out = pe.encode(&[0, 1, 2], 8);
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn sinusoidal_position_zero_starts_with_zero() {
        let pe = PositionEmbedding::sinusoidal();
        let out = pe.encode(&[0], 4);
        // sin(0) = 0 for even dims at position 0
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_position_zero_cos_starts_with_one() {
        let pe = PositionEmbedding::sinusoidal();
        let out = pe.encode(&[0], 4);
        // cos(0) = 1 for odd dims at position 0
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_varies_by_position() {
        let pe = PositionEmbedding::sinusoidal();
        let p0 = pe.encode(&[0], 8);
        let p1 = pe.encode(&[1], 8);
        assert!(p0.iter().zip(p1.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn sinusoidal_deterministic() {
        let pe = PositionEmbedding::sinusoidal();
        let a = pe.encode(&[0, 1, 2], 8);
        let b = pe.encode(&[0, 1, 2], 8);
        assert_eq!(a, b);
    }

    #[test]
    fn sinusoidal_even_odd_pattern() {
        let pe = PositionEmbedding::sinusoidal();
        let out = pe.encode(&[1], 4);
        let d = 4.0_f32;
        // dim 0 (even): sin(1 / 10000^(0/4)) = sin(1.0)
        let expected_0 = (1.0_f32 / 10_000_f32.powf(0.0 / d)).sin();
        assert!((out[0] - expected_0).abs() < 1e-5);
        // dim 1 (odd): cos(1 / 10000^(0/4)) = cos(1.0)
        let expected_1 = (1.0_f32 / 10_000_f32.powf(0.0 / d)).cos();
        assert!((out[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn sinusoidal_higher_dims_change_slower() {
        let pe = PositionEmbedding::sinusoidal();
        let p0 = pe.encode(&[0], 8);
        let p1 = pe.encode(&[1], 8);
        // Low-frequency dims (higher indices) should change less
        let diff_low = (p1[0] - p0[0]).abs();
        let diff_high = (p1[6] - p0[6]).abs();
        assert!(diff_low >= diff_high);
    }

    #[test]
    fn sinusoidal_empty_positions() {
        let pe = PositionEmbedding::sinusoidal();
        let out = pe.encode(&[], 8);
        assert!(out.is_empty());
    }

    #[test]
    fn sinusoidal_math_correctness_dim2() {
        // Verify exact formula: PE(pos,2i)=sin(pos/10000^(2i/d))
        let pe = PositionEmbedding::sinusoidal();
        let pos = 5;
        let dim = 4;
        let out = pe.encode(&[pos], dim);
        let d = dim as f32;
        for i in 0..dim {
            let dim_pair = (i / 2) as f32;
            let angle = pos as f32 / 10_000_f32.powf(2.0 * dim_pair / d);
            let expected = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "dim {i}: got {} expected {expected}",
                out[i]
            );
        }
    }

    // ===================================================================
    // PositionEmbedding – learned tests
    // ===================================================================

    #[test]
    fn learned_encoding_lookup() {
        let weights = seq_weights(8); // 2 positions × 4 dims
        let pe = PositionEmbedding::learned(weights, 2, 4).unwrap();
        let out = pe.encode(&[0], 4);
        assert_eq!(&out, &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn learned_encoding_second_position() {
        let weights = seq_weights(8);
        let pe = PositionEmbedding::learned(weights, 2, 4).unwrap();
        let out = pe.encode(&[1], 4);
        assert_eq!(&out, &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn learned_encoding_clamps_beyond_max() {
        let weights = seq_weights(8);
        let pe = PositionEmbedding::learned(weights, 2, 4).unwrap();
        let out = pe.encode(&[99], 4);
        // Should clamp to last position (1)
        assert_eq!(&out, &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn learned_encoding_dimension_mismatch() {
        assert!(PositionEmbedding::learned(seq_weights(7), 2, 4).is_err());
    }

    // ===================================================================
    // PositionEmbedding – RoPE / ALiBi / NoPE tests
    // ===================================================================

    #[test]
    fn rope_returns_zeros() {
        let pe = PositionEmbedding::rope(10_000.0);
        let out = pe.encode(&[0, 1, 2], 4);
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn rope_base_accessor() {
        let pe = PositionEmbedding::rope(500.0);
        assert!((pe.rope_base() - 500.0).abs() < f32::EPSILON);
    }

    #[test]
    fn alibi_returns_zeros() {
        let pe = PositionEmbedding::alibi(0.5);
        let out = pe.encode(&[0, 1], 4);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn alibi_slope_accessor() {
        let pe = PositionEmbedding::alibi(0.25);
        assert!((pe.alibi_slope() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn nope_returns_zeros() {
        let pe = PositionEmbedding::none();
        let out = pe.encode(&[0, 1, 2], 8);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn position_embedding_variant_accessor() {
        assert_eq!(*PositionEmbedding::sinusoidal().variant(), PositionEncodingType::Sinusoidal);
        assert_eq!(*PositionEmbedding::rope(10_000.0).variant(), PositionEncodingType::RoPE);
        assert_eq!(*PositionEmbedding::alibi(0.5).variant(), PositionEncodingType::ALiBi);
        assert_eq!(*PositionEmbedding::none().variant(), PositionEncodingType::NoPE);
    }

    // ===================================================================
    // EmbeddingCombiner tests
    // ===================================================================

    #[test]
    fn combiner_add_mode() {
        let c = EmbeddingCombiner::new(CombineMode::Add);
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        let out = c.combine(&tok, &pos, 2, 2).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 1.1).abs() < 1e-5);
        assert!((out[1] - 2.2).abs() < 1e-5);
    }

    #[test]
    fn combiner_concat_mode() {
        let c = EmbeddingCombiner::new(CombineMode::Concat);
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        let out = c.combine(&tok, &pos, 2, 2).unwrap();
        // Each position: [tok0, tok1, pos0, pos1]
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..4], &[1.0, 2.0, 0.1, 0.2]);
        assert_eq!(&out[4..], &[3.0, 4.0, 0.3, 0.4]);
    }

    #[test]
    fn combiner_output_dim_add() {
        let c = EmbeddingCombiner::new(CombineMode::Add);
        assert_eq!(c.output_dim(16), 16);
    }

    #[test]
    fn combiner_output_dim_concat() {
        let c = EmbeddingCombiner::new(CombineMode::Concat);
        assert_eq!(c.output_dim(16), 32);
    }

    #[test]
    fn combiner_dimension_mismatch_token() {
        let c = EmbeddingCombiner::new(CombineMode::Add);
        let tok = vec![1.0, 2.0, 3.0]; // wrong size
        let pos = vec![0.1, 0.2, 0.3, 0.4];
        assert!(c.combine(&tok, &pos, 2, 2).is_err());
    }

    #[test]
    fn combiner_dimension_mismatch_position() {
        let c = EmbeddingCombiner::new(CombineMode::Add);
        let tok = vec![1.0, 2.0, 3.0, 4.0];
        let pos = vec![0.1, 0.2, 0.3]; // wrong size
        assert!(c.combine(&tok, &pos, 2, 2).is_err());
    }

    #[test]
    fn combiner_mode_accessor() {
        assert_eq!(EmbeddingCombiner::new(CombineMode::Add).mode(), CombineMode::Add);
        assert_eq!(EmbeddingCombiner::new(CombineMode::Concat).mode(), CombineMode::Concat);
    }

    // ===================================================================
    // EmbeddingNorm tests
    // ===================================================================

    #[test]
    fn norm_identity_preserves_mean_zero_data() {
        let norm = EmbeddingNorm::identity(4);
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let out = norm.normalize(&data, 4);
        // After normalization, mean should be ~0
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn norm_identity_unit_variance() {
        let norm = EmbeddingNorm::identity(4);
        let data = vec![2.0, 0.0, -2.0, 0.0];
        let out = norm.normalize(&data, 4);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        // Variance should be close to 1 (within eps tolerance)
        assert!((var - 1.0).abs() < 0.01, "var = {var}");
    }

    #[test]
    fn norm_with_scale_and_bias() {
        let gamma = vec![2.0, 2.0];
        let beta = vec![1.0, 1.0];
        let norm = EmbeddingNorm::new(gamma, beta, 1e-5).unwrap();
        let data = vec![3.0, 1.0]; // mean=2, var=1
        let out = norm.normalize(&data, 2);
        // (3-2)/1 * 2 + 1 = 3, (1-2)/1 * 2 + 1 = -1
        assert!((out[0] - 3.0).abs() < 0.01);
        assert!((out[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn norm_multiple_positions() {
        let norm = EmbeddingNorm::identity(2);
        let data = vec![4.0, 0.0, 0.0, 4.0];
        let out = norm.normalize(&data, 2);
        assert_eq!(out.len(), 4);
        // Each pair normalised independently
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn norm_gamma_beta_mismatch() {
        assert!(EmbeddingNorm::new(vec![1.0, 1.0], vec![0.0], 1e-5).is_err());
    }

    #[test]
    fn norm_dim_accessor() {
        assert_eq!(EmbeddingNorm::identity(8).dim(), 8);
    }

    // ===================================================================
    // EmbeddingDropout tests
    // ===================================================================

    #[test]
    fn dropout_inference_passthrough() {
        let d = EmbeddingDropout::inference();
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(d.apply(&data), data);
    }

    #[test]
    fn dropout_training_zero_p_passthrough() {
        let d = EmbeddingDropout::new(0.0, true);
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(d.apply(&data), data);
    }

    #[test]
    fn dropout_training_some_zeros() {
        let d = EmbeddingDropout::new(0.5, true);
        let data = vec![1.0; 100];
        let out = d.apply(&data);
        let zero_count = out.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5 and deterministic hash, we expect some zeros
        assert!(zero_count > 0, "Expected some zeros with p=0.5");
        assert!(zero_count < 100, "Not all should be zero");
    }

    #[test]
    fn dropout_training_full_drop() {
        let d = EmbeddingDropout::new(1.0, true);
        let data = vec![1.0; 10];
        let out = d.apply(&data);
        // With p=1.0, all elements should be zero (hash < u32::MAX always)
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dropout_accessors() {
        let d = EmbeddingDropout::new(0.3, true);
        assert!((d.p() - 0.3).abs() < f32::EPSILON);
        assert!(d.is_training());
    }

    #[test]
    fn dropout_empty_input() {
        let d = EmbeddingDropout::new(0.5, true);
        assert!(d.apply(&[]).is_empty());
    }

    // ===================================================================
    // EmbeddingProjection tests
    // ===================================================================

    #[test]
    fn projection_identity() {
        let proj = EmbeddingProjection::identity(3);
        let data = vec![1.0, 2.0, 3.0];
        let out = proj.project(&data, 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn projection_dimension_change() {
        // 2×3 weights: project 3 → 2
        let weights = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let proj = EmbeddingProjection::new(weights, None, 3, 2).unwrap();
        let data = vec![5.0, 7.0, 9.0];
        let out = proj.project(&data, 1).unwrap();
        assert_eq!(out, vec![5.0, 7.0]);
    }

    #[test]
    fn projection_with_bias() {
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let bias = Some(vec![10.0, 20.0]);
        let proj = EmbeddingProjection::new(weights, bias, 2, 2).unwrap();
        let data = vec![3.0, 4.0];
        let out = proj.project(&data, 1).unwrap();
        assert_eq!(out, vec![13.0, 24.0]);
    }

    #[test]
    fn projection_multiple_positions() {
        let proj = EmbeddingProjection::identity(2);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = proj.project(&data, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn projection_weight_mismatch() {
        assert!(EmbeddingProjection::new(vec![1.0; 5], None, 3, 2).is_err());
    }

    #[test]
    fn projection_bias_mismatch() {
        assert!(EmbeddingProjection::new(vec![1.0; 6], Some(vec![1.0]), 3, 2).is_err());
    }

    #[test]
    fn projection_input_size_mismatch() {
        let proj = EmbeddingProjection::identity(3);
        assert!(proj.project(&[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn projection_accessors() {
        let proj = EmbeddingProjection::new(vec![0.0; 6], None, 3, 2).unwrap();
        assert_eq!(proj.input_dim(), 3);
        assert_eq!(proj.output_dim(), 2);
    }

    #[test]
    fn projection_upscale() {
        // Project 2 → 4
        let weights = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
            0.0, 0.0, // row 3
        ];
        let proj = EmbeddingProjection::new(weights, None, 2, 4).unwrap();
        let data = vec![3.0, 5.0];
        let out = proj.project(&data, 1).unwrap();
        assert_eq!(out, vec![3.0, 5.0, 8.0, 0.0]);
    }

    // ===================================================================
    // EmbeddingLayer (orchestrator) tests
    // ===================================================================

    fn make_simple_layer(vocab: usize, dim: usize, max_seq: usize) -> EmbeddingLayer {
        let weights = seq_weights(vocab * dim);
        build_default_layer(weights, vocab, dim, max_seq).unwrap()
    }

    #[test]
    fn layer_forward_basic() {
        let layer = make_simple_layer(4, 4, 16);
        let out = layer.forward(&[0, 1]).unwrap();
        // 2 positions × 4 dims
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn layer_forward_empty_sequence() {
        let layer = make_simple_layer(4, 4, 16);
        let out = layer.forward(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn layer_forward_single_token() {
        let layer = make_simple_layer(4, 4, 16);
        let out = layer.forward(&[0]).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn layer_forward_max_length() {
        let layer = make_simple_layer(8, 2, 4);
        let out = layer.forward(&[0, 1, 2, 3]).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn layer_forward_exceeds_max_length() {
        let layer = make_simple_layer(8, 2, 2);
        assert!(layer.forward(&[0, 1, 2]).is_err());
    }

    #[test]
    fn layer_forward_out_of_vocab() {
        let layer = make_simple_layer(4, 4, 16);
        assert!(layer.forward(&[4]).is_err());
    }

    #[test]
    fn layer_output_dim_no_projection() {
        let layer = make_simple_layer(4, 8, 16);
        assert_eq!(layer.output_dim(), 8);
    }

    #[test]
    fn layer_output_dim_with_projection() {
        let config = EmbeddingConfig {
            vocab_size: 4,
            embedding_dim: 4,
            max_seq_len: 16,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::Sinusoidal,
        };
        let te = TokenEmbedding::new(seq_weights(16), 4, 4, None, 1.0).unwrap();
        let pe = PositionEmbedding::sinusoidal();
        let combiner = EmbeddingCombiner::new(CombineMode::Add);
        let norm = EmbeddingNorm::identity(4);
        let dropout = EmbeddingDropout::inference();
        let proj = EmbeddingProjection::identity(4);
        let layer = EmbeddingLayer::new(config, te, pe, combiner, norm, dropout, Some(proj));
        assert_eq!(layer.output_dim(), 4);
    }

    #[test]
    fn layer_with_rope_positions() {
        let config = EmbeddingConfig {
            vocab_size: 4,
            embedding_dim: 4,
            max_seq_len: 16,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::RoPE,
        };
        let te = TokenEmbedding::new(seq_weights(16), 4, 4, None, 1.0).unwrap();
        let pe = PositionEmbedding::rope(10_000.0);
        let combiner = EmbeddingCombiner::new(CombineMode::Add);
        let norm = EmbeddingNorm::identity(4);
        let dropout = EmbeddingDropout::inference();
        let layer = EmbeddingLayer::new(config, te, pe, combiner, norm, dropout, None);
        // RoPE positions are zeros, so token embeddings pass through norm only
        let out = layer.forward(&[0, 1]).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn layer_with_alibi_positions() {
        let config = EmbeddingConfig {
            vocab_size: 4,
            embedding_dim: 4,
            max_seq_len: 16,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::ALiBi,
        };
        let te = TokenEmbedding::new(seq_weights(16), 4, 4, None, 1.0).unwrap();
        let pe = PositionEmbedding::alibi(0.5);
        let combiner = EmbeddingCombiner::new(CombineMode::Add);
        let norm = EmbeddingNorm::identity(4);
        let dropout = EmbeddingDropout::inference();
        let layer = EmbeddingLayer::new(config, te, pe, combiner, norm, dropout, None);
        let out = layer.forward(&[0]).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn layer_with_nope_positions() {
        let config = EmbeddingConfig {
            vocab_size: 4,
            embedding_dim: 4,
            max_seq_len: 16,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::NoPE,
        };
        let te = TokenEmbedding::new(seq_weights(16), 4, 4, None, 1.0).unwrap();
        let pe = PositionEmbedding::none();
        let combiner = EmbeddingCombiner::new(CombineMode::Add);
        let norm = EmbeddingNorm::identity(4);
        let dropout = EmbeddingDropout::inference();
        let layer = EmbeddingLayer::new(config, te, pe, combiner, norm, dropout, None);
        let out = layer.forward(&[1]).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn layer_config_accessor() {
        let layer = make_simple_layer(4, 4, 16);
        assert_eq!(layer.config().vocab_size, 4);
        assert_eq!(layer.config().embedding_dim, 4);
    }

    #[test]
    fn layer_with_projection_changes_dim() {
        let config = EmbeddingConfig {
            vocab_size: 4,
            embedding_dim: 4,
            max_seq_len: 16,
            padding_idx: None,
            scale_factor: 1.0,
            position_encoding: PositionEncodingType::Sinusoidal,
        };
        let te = TokenEmbedding::new(seq_weights(16), 4, 4, None, 1.0).unwrap();
        let pe = PositionEmbedding::sinusoidal();
        let combiner = EmbeddingCombiner::new(CombineMode::Add);
        let norm = EmbeddingNorm::identity(4);
        let dropout = EmbeddingDropout::inference();
        // Project 4 → 2
        let proj_weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let proj = EmbeddingProjection::new(proj_weights, None, 4, 2).unwrap();
        let layer = EmbeddingLayer::new(config, te, pe, combiner, norm, dropout, Some(proj));
        let out = layer.forward(&[0, 1]).unwrap();
        assert_eq!(out.len(), 4); // 2 positions × 2 dims
        assert_eq!(layer.output_dim(), 2);
    }

    #[test]
    fn layer_deterministic() {
        let layer = make_simple_layer(4, 4, 16);
        let a = layer.forward(&[0, 1, 2]).unwrap();
        let b = layer.forward(&[0, 1, 2]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn layer_different_tokens_different_output() {
        // Use weights where different tokens produce genuinely different
        // normalised embeddings.
        let weights = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 0.0, 0.0, 1.0, // token 1
            0.5, 0.5, 0.0, 0.0, // token 2
            0.0, 0.0, 0.5, 0.5, // token 3
        ];
        let layer = build_default_layer(weights, 4, 4, 16).unwrap();
        let a = layer.forward(&[0]).unwrap();
        let b = layer.forward(&[1]).unwrap();
        // After sinusoidal addition + norm, different weight patterns
        // should still differ.
        assert_ne!(a, b);
    }

    // ===================================================================
    // EmbeddingError tests
    // ===================================================================

    #[test]
    fn error_display_out_of_vocab() {
        let e = EmbeddingError::OutOfVocab { token_id: 5, vocab_size: 3 };
        assert!(e.to_string().contains("5"));
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = EmbeddingError::DimensionMismatch { expected: 10, actual: 7 };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn error_display_sequence_too_long() {
        let e = EmbeddingError::SequenceTooLong { len: 100, max: 50 };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = EmbeddingError::InvalidConfig("bad param".into());
        assert!(e.to_string().contains("bad param"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbeddingError>();
    }

    // ===================================================================
    // build_default_layer tests
    // ===================================================================

    #[test]
    fn build_default_layer_ok() {
        let layer = build_default_layer(seq_weights(16), 4, 4, 32);
        assert!(layer.is_ok());
    }

    #[test]
    fn build_default_layer_bad_weights() {
        assert!(build_default_layer(seq_weights(15), 4, 4, 32).is_err());
    }

    // ===================================================================
    // proptest
    // ===================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn token_embed_never_panics(
                vocab_size in 1_usize..=64,
                dim in 1_usize..=32,
                token_id in 0_u32..=63,
            ) {
                let weights = vec![1.0; vocab_size * dim];
                let te = TokenEmbedding::new(weights, vocab_size, dim, None, 1.0).unwrap();
                let result = te.embed_single(token_id);
                if (token_id as usize) < vocab_size {
                    prop_assert!(result.is_ok());
                    prop_assert_eq!(result.unwrap().len(), dim);
                } else {
                    prop_assert!(result.is_err());
                }
            }

            #[test]
            fn sinusoidal_output_length(
                seq_len in 0_usize..=32,
                dim in 1_usize..=64,
            ) {
                let positions: Vec<usize> = (0..seq_len).collect();
                let pe = PositionEmbedding::sinusoidal();
                let out = pe.encode(&positions, dim);
                prop_assert_eq!(out.len(), seq_len * dim);
            }

            #[test]
            fn sinusoidal_values_bounded(
                pos in 0_usize..=1000,
                dim in 2_usize..=64,
            ) {
                let pe = PositionEmbedding::sinusoidal();
                let out = pe.encode(&[pos], dim);
                for &v in &out {
                    prop_assert!(v >= -1.0 && v <= 1.0,
                        "sinusoidal value {v} out of [-1,1] at pos={pos} dim={dim}");
                }
            }

            #[test]
            fn combiner_add_output_length(
                seq_len in 1_usize..=16,
                dim in 1_usize..=16,
            ) {
                let c = EmbeddingCombiner::new(CombineMode::Add);
                let tok = vec![1.0; seq_len * dim];
                let pos = vec![0.5; seq_len * dim];
                let out = c.combine(&tok, &pos, seq_len, dim).unwrap();
                prop_assert_eq!(out.len(), seq_len * dim);
            }

            #[test]
            fn combiner_concat_output_length(
                seq_len in 1_usize..=16,
                dim in 1_usize..=16,
            ) {
                let c = EmbeddingCombiner::new(CombineMode::Concat);
                let tok = vec![1.0; seq_len * dim];
                let pos = vec![0.5; seq_len * dim];
                let out = c.combine(&tok, &pos, seq_len, dim).unwrap();
                prop_assert_eq!(out.len(), seq_len * dim * 2);
            }

            #[test]
            fn layer_forward_random_tokens(
                vocab in 4_usize..=16,
                dim in 2_usize..=8,
                seq_len in 1_usize..=8,
            ) {
                let weights = vec![0.5; vocab * dim];
                let layer = build_default_layer(weights, vocab, dim, 32).unwrap();
                let tokens: Vec<u32> = (0..seq_len).map(|i| (i % vocab) as u32).collect();
                let result = layer.forward(&tokens);
                prop_assert!(result.is_ok());
                let out = result.unwrap();
                prop_assert_eq!(out.len(), seq_len * dim);
            }

            #[test]
            fn norm_output_length(
                seq_len in 1_usize..=8,
                dim in 1_usize..=16,
            ) {
                let norm = EmbeddingNorm::identity(dim);
                let data: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
                let out = norm.normalize(&data, dim);
                prop_assert_eq!(out.len(), seq_len * dim);
            }

            #[test]
            fn projection_output_length(
                seq_len in 1_usize..=8,
                in_dim in 1_usize..=8,
                out_dim in 1_usize..=8,
            ) {
                let weights = vec![0.0; out_dim * in_dim];
                let proj = EmbeddingProjection::new(weights, None, in_dim, out_dim).unwrap();
                let data = vec![1.0; seq_len * in_dim];
                let out = proj.project(&data, seq_len).unwrap();
                prop_assert_eq!(out.len(), seq_len * out_dim);
            }

            #[test]
            fn dropout_inference_always_passthrough(data in proptest::collection::vec(-100.0_f32..100.0, 0..64)) {
                let d = EmbeddingDropout::inference();
                let out = d.apply(&data);
                prop_assert_eq!(out, data);
            }
        }
    }
}
