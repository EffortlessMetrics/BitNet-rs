//! Multi-modal fusion system for combining text, image, audio, and other
//! modality embeddings into a unified representation.
//!
//! Supports multiple fusion strategies including concatenation, attention-based
//! fusion, gated fusion, cross-attention, and weighted sum.

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names, clippy::unnecessary_wraps)]

use std::collections::HashMap;
use std::fmt;

// ── Error types ──────────────────────────────────────────────────────────────

/// Errors arising from multi-modal fusion operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionError {
    /// Embedding dimension mismatch between modalities.
    DimensionMismatch { expected: usize, got: usize, modality: ModalityType },
    /// No modality inputs provided.
    EmptyInput,
    /// Unsupported modality for the current pipeline configuration.
    UnsupportedModality(ModalityType),
    /// Configuration is invalid.
    InvalidConfig(String),
    /// Projection failed.
    ProjectionError(String),
    /// Zero-dimension embedding not allowed.
    ZeroDimension,
    /// Sequence length exceeded maximum.
    SequenceTooLong { max: usize, got: usize },
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got, modality } => {
                write!(f, "dimension mismatch for {modality:?}: expected {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "no modality inputs provided"),
            Self::UnsupportedModality(m) => write!(f, "unsupported modality: {m:?}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ProjectionError(msg) => write!(f, "projection error: {msg}"),
            Self::ZeroDimension => write!(f, "zero-dimension embedding not allowed"),
            Self::SequenceTooLong { max, got } => {
                write!(f, "sequence too long: max {max}, got {got}")
            }
        }
    }
}

// ── Modality types ───────────────────────────────────────────────────────────

/// Supported input modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Structured,
}

impl ModalityType {
    /// Returns all known modality variants.
    pub const fn all() -> &'static [Self] {
        &[Self::Text, Self::Image, Self::Audio, Self::Video, Self::Structured]
    }

    /// Human-readable label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Structured => "structured",
        }
    }
}

// ── Modality configuration ───────────────────────────────────────────────────

/// Per-modality configuration for embedding dimensions and constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct ModalityConfig {
    pub modality: ModalityType,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub projection_dim: usize,
    pub dropout_rate: f32,
}

impl ModalityConfig {
    pub const fn new(
        modality: ModalityType,
        embedding_dim: usize,
        max_seq_len: usize,
        projection_dim: usize,
    ) -> Result<Self, FusionError> {
        if embedding_dim == 0 || projection_dim == 0 {
            return Err(FusionError::ZeroDimension);
        }
        Ok(Self { modality, embedding_dim, max_seq_len, projection_dim, dropout_rate: 0.0 })
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn with_dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate.clamp(0.0, 1.0);
        self
    }
}

// ── Fusion strategy ──────────────────────────────────────────────────────────

/// Strategy for combining embeddings from multiple modalities.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Concatenate all modality embeddings along the feature dimension.
    Concatenation,
    /// Attention-based weighting across modalities.
    Attention { num_heads: usize },
    /// Learned gating network that produces per-modality weights.
    GatedFusion { hidden_dim: usize },
    /// Cross-attention: one modality attends to another.
    CrossAttention { query_modality: ModalityType, key_modality: ModalityType, num_heads: usize },
    /// Weighted sum with fixed or learnable weights.
    WeightedSum { weights: Vec<f32> },
}

impl FusionStrategy {
    /// Computes the output dimension given input configs and a shared projection dim.
    pub const fn output_dim(&self, num_modalities: usize, projection_dim: usize) -> usize {
        match self {
            Self::Concatenation => num_modalities * projection_dim,
            Self::Attention { .. }
            | Self::GatedFusion { .. }
            | Self::CrossAttention { .. }
            | Self::WeightedSum { .. } => projection_dim,
        }
    }
}

// ── Modality embedding ───────────────────────────────────────────────────────

/// A single modality's embedding tensor (flattened row-major).
#[derive(Debug, Clone, PartialEq)]
pub struct ModalityEmbedding {
    pub data: Vec<f32>,
    pub modality: ModalityType,
    pub seq_len: usize,
    pub dim: usize,
}

impl ModalityEmbedding {
    pub fn new(
        data: Vec<f32>,
        modality: ModalityType,
        seq_len: usize,
        dim: usize,
    ) -> Result<Self, FusionError> {
        if dim == 0 {
            return Err(FusionError::ZeroDimension);
        }
        if data.len() != seq_len * dim {
            return Err(FusionError::DimensionMismatch {
                expected: seq_len * dim,
                got: data.len(),
                modality,
            });
        }
        Ok(Self { data, modality, seq_len, dim })
    }

    /// Returns the element at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.dim + col]
    }

    /// Total number of elements.
    pub const fn numel(&self) -> usize {
        self.data.len()
    }
}

// ── Projection layer ─────────────────────────────────────────────────────────

/// Linearly projects from `input_dim` to `output_dim`: y = x·W + b.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionLayer {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl ProjectionLayer {
    /// Creates an identity-like projection (truncation or zero-padding).
    pub fn identity(input_dim: usize, output_dim: usize) -> Result<Self, FusionError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(FusionError::ZeroDimension);
        }
        let mut weight = vec![0.0f32; input_dim * output_dim];
        let min_dim = input_dim.min(output_dim);
        for idx in 0..min_dim {
            weight[idx * output_dim + idx] = 1.0;
        }
        let bias = vec![0.0f32; output_dim];
        Ok(Self { weight, bias, input_dim, output_dim })
    }

    /// Creates a projection with Xavier-style uniform init scaled by `scale`.
    pub fn scaled(input_dim: usize, output_dim: usize, scale: f32) -> Result<Self, FusionError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(FusionError::ZeroDimension);
        }
        let fan = (input_dim + output_dim) as f32;
        let val = scale * (2.0 / fan).sqrt();
        let weight = vec![val; input_dim * output_dim];
        let bias = vec![0.0f32; output_dim];
        Ok(Self { weight, bias, input_dim, output_dim })
    }

    /// Project a batch of vectors: `input` is `[rows × input_dim]`, returns `[rows × output_dim]`.
    pub fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, FusionError> {
        if input.len() != rows * self.input_dim {
            return Err(FusionError::ProjectionError(format!(
                "input length {} != rows({}) × input_dim({})",
                input.len(),
                rows,
                self.input_dim
            )));
        }
        let mut output = vec![0.0f32; rows * self.output_dim];
        for row in 0..rows {
            for out_idx in 0..self.output_dim {
                let mut sum = self.bias[out_idx];
                for in_idx in 0..self.input_dim {
                    sum += input[row * self.input_dim + in_idx]
                        * self.weight[in_idx * self.output_dim + out_idx];
                }
                output[row * self.output_dim + out_idx] = sum;
            }
        }
        Ok(output)
    }
}

// ── Gating network ──────────────────────────────────────────────────────────

/// Produces per-modality gate values in `[0, 1]` via sigmoid.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingNetwork {
    pub num_modalities: usize,
    pub hidden_dim: usize,
    pub input_to_hidden: Vec<f32>,
    pub hidden_to_gates: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub gate_bias: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

impl GatingNetwork {
    /// Creates a gating network with uniform initialization.
    pub fn new(
        input_dim: usize,
        num_modalities: usize,
        hidden_dim: usize,
    ) -> Result<Self, FusionError> {
        if input_dim == 0 || num_modalities == 0 || hidden_dim == 0 {
            return Err(FusionError::ZeroDimension);
        }
        let fan_in = (input_dim + hidden_dim) as f32;
        let scale = (2.0 / fan_in).sqrt();
        Ok(Self {
            num_modalities,
            hidden_dim,
            input_to_hidden: vec![scale; input_dim * hidden_dim],
            hidden_to_gates: vec![scale; hidden_dim * num_modalities],
            hidden_bias: vec![0.0; hidden_dim],
            gate_bias: vec![0.0; num_modalities],
        })
    }

    /// Computes gate values from concatenated modality embeddings.
    /// `input` is `[input_dim]`, returns `[num_modalities]` values in `[0, 1]`.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let input_dim = self.input_to_hidden.len() / self.hidden_dim;
        // Hidden layer: ReLU(x·W1 + b1)
        let mut hidden: Vec<f32> = self.hidden_bias.clone();
        for (hi, hval) in hidden.iter_mut().enumerate() {
            for in_idx in 0..input_dim {
                *hval += input.get(in_idx).copied().unwrap_or(0.0)
                    * self.input_to_hidden[in_idx * self.hidden_dim + hi];
            }
            *hval = relu(*hval);
        }
        // Gate layer: sigmoid(h·W2 + b2)
        let mut gates: Vec<f32> = self.gate_bias.clone();
        for (gi, gval) in gates.iter_mut().enumerate() {
            for (hi, &hval) in hidden.iter().enumerate() {
                *gval += hval * self.hidden_to_gates[hi * self.num_modalities + gi];
            }
            *gval = sigmoid(*gval);
        }
        gates
    }
}

// ── Cross-attention fuser ────────────────────────────────────────────────────

/// Applies cross-attention: query from one modality, key/value from another.
#[derive(Debug, Clone)]
pub struct CrossAttentionFuser {
    pub num_heads: usize,
    pub head_dim: usize,
    pub model_dim: usize,
    pub query_proj: ProjectionLayer,
    pub key_proj: ProjectionLayer,
    pub value_proj: ProjectionLayer,
    pub output_proj: ProjectionLayer,
}

impl CrossAttentionFuser {
    pub fn new(model_dim: usize, num_heads: usize) -> Result<Self, FusionError> {
        if model_dim == 0 || num_heads == 0 {
            return Err(FusionError::ZeroDimension);
        }
        if !model_dim.is_multiple_of(num_heads) {
            return Err(FusionError::InvalidConfig(format!(
                "model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )));
        }
        let head_dim = model_dim / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            model_dim,
            query_proj: ProjectionLayer::identity(model_dim, model_dim)?,
            key_proj: ProjectionLayer::identity(model_dim, model_dim)?,
            value_proj: ProjectionLayer::identity(model_dim, model_dim)?,
            output_proj: ProjectionLayer::identity(model_dim, model_dim)?,
        })
    }

    /// Compute scaled dot-product attention scores.
    /// `query` is `[q_len × dim]`, `key` is `[k_len × dim]`.
    /// Returns `[q_len × k_len]` attention weights (softmax-normalized per row).
    pub fn attention_weights(
        &self,
        query: &[f32],
        key: &[f32],
        q_len: usize,
        k_len: usize,
        dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut weights = vec![0.0f32; q_len * k_len];
        for qi in 0..q_len {
            for ki in 0..k_len {
                let mut dot = 0.0f32;
                for di in 0..dim {
                    dot += query[qi * dim + di] * key[ki * dim + di];
                }
                weights[qi * k_len + ki] = dot * scale;
            }
            // Softmax over key dimension for this query position.
            let row_start = qi * k_len;
            let row_end = row_start + k_len;
            let max_val =
                weights[row_start..row_end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for val in &mut weights[row_start..row_end] {
                *val = (*val - max_val).exp();
                sum_exp += *val;
            }
            if sum_exp > 0.0 {
                for val in &mut weights[row_start..row_end] {
                    *val /= sum_exp;
                }
            }
        }
        weights
    }

    /// Run cross-attention: `query_emb` attends to `kv_emb`.
    /// Both embeddings must have `dim == self.model_dim`.
    pub fn forward(
        &self,
        query_emb: &ModalityEmbedding,
        kv_emb: &ModalityEmbedding,
    ) -> Result<Vec<f32>, FusionError> {
        if query_emb.dim != self.model_dim {
            return Err(FusionError::DimensionMismatch {
                expected: self.model_dim,
                got: query_emb.dim,
                modality: query_emb.modality,
            });
        }
        if kv_emb.dim != self.model_dim {
            return Err(FusionError::DimensionMismatch {
                expected: self.model_dim,
                got: kv_emb.dim,
                modality: kv_emb.modality,
            });
        }
        let q_data = self.query_proj.forward(&query_emb.data, query_emb.seq_len)?;
        let k_data = self.key_proj.forward(&kv_emb.data, kv_emb.seq_len)?;
        let v_data = self.value_proj.forward(&kv_emb.data, kv_emb.seq_len)?;

        let attn = self.attention_weights(
            &q_data,
            &k_data,
            query_emb.seq_len,
            kv_emb.seq_len,
            self.model_dim,
        );

        // Weighted sum over values: output[q, d] = sum_k attn[q,k] * v[k,d]
        let mut out = vec![0.0f32; query_emb.seq_len * self.model_dim];
        for qi in 0..query_emb.seq_len {
            for di in 0..self.model_dim {
                let mut sum = 0.0f32;
                for ki in 0..kv_emb.seq_len {
                    sum += attn[qi * kv_emb.seq_len + ki] * v_data[ki * self.model_dim + di];
                }
                out[qi * self.model_dim + di] = sum;
            }
        }
        self.output_proj.forward(&out, query_emb.seq_len)
    }
}

// ── Fusion result ────────────────────────────────────────────────────────────

/// Output of a fusion operation.
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Fused embedding data, `[seq_len × fused_dim]`.
    pub fused_data: Vec<f32>,
    pub fused_dim: usize,
    pub seq_len: usize,
    /// Per-modality attention/gate weights (if applicable).
    pub attention_weights: Option<Vec<f32>>,
    /// Gate values from gated fusion (if applicable).
    pub gate_values: Option<Vec<f32>>,
    /// Per-modality contribution norms.
    pub modality_contributions: HashMap<ModalityType, f32>,
}

impl FusionResult {
    pub const fn numel(&self) -> usize {
        self.fused_data.len()
    }

    /// L2 norm of the fused output.
    pub fn norm(&self) -> f32 {
        self.fused_data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// ── Modality encoder (trait) ─────────────────────────────────────────────────

/// Stub encoder interface: each modality provides an encoder.
pub trait ModalityEncoder {
    /// Encode raw input bytes into an embedding.
    fn encode(&self, input: &[u8]) -> Result<ModalityEmbedding, FusionError>;

    /// The native embedding dimension of this encoder.
    fn get_dim(&self) -> usize;

    /// Which modality this encoder handles.
    fn modality(&self) -> ModalityType;

    /// Whether this encoder supports streaming input.
    fn supports_streaming(&self) -> bool;
}

// ── Concrete stub encoders ───────────────────────────────────────────────────

/// A stub text encoder that produces deterministic embeddings from raw bytes.
pub struct StubTextEncoder {
    pub dim: usize,
}

impl ModalityEncoder for StubTextEncoder {
    fn encode(&self, input: &[u8]) -> Result<ModalityEmbedding, FusionError> {
        if input.is_empty() {
            return Err(FusionError::EmptyInput);
        }
        let seq_len = input.len();
        let data: Vec<f32> =
            (0..seq_len * self.dim).map(|idx| (idx % 256) as f32 / 255.0).collect();
        ModalityEmbedding::new(data, ModalityType::Text, seq_len, self.dim)
    }

    fn get_dim(&self) -> usize {
        self.dim
    }

    fn modality(&self) -> ModalityType {
        ModalityType::Text
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

/// A stub image encoder.
pub struct StubImageEncoder {
    pub dim: usize,
    pub patch_size: usize,
}

impl ModalityEncoder for StubImageEncoder {
    fn encode(&self, input: &[u8]) -> Result<ModalityEmbedding, FusionError> {
        if input.is_empty() {
            return Err(FusionError::EmptyInput);
        }
        let seq_len = input.len().div_ceil(self.patch_size);
        let data: Vec<f32> =
            (0..seq_len * self.dim).map(|idx| ((idx * 7 + 13) % 256) as f32 / 255.0).collect();
        ModalityEmbedding::new(data, ModalityType::Image, seq_len, self.dim)
    }

    fn get_dim(&self) -> usize {
        self.dim
    }

    fn modality(&self) -> ModalityType {
        ModalityType::Image
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

/// A stub audio encoder.
pub struct StubAudioEncoder {
    pub dim: usize,
    pub sample_rate: usize,
}

impl ModalityEncoder for StubAudioEncoder {
    fn encode(&self, input: &[u8]) -> Result<ModalityEmbedding, FusionError> {
        if input.is_empty() {
            return Err(FusionError::EmptyInput);
        }
        // Each frame_size bytes = 1 frame.
        let frame_size = self.sample_rate / 100; // ~10ms frames
        let seq_len = if frame_size == 0 { input.len() } else { input.len().div_ceil(frame_size) };
        let seq_len = seq_len.max(1);
        let data: Vec<f32> =
            (0..seq_len * self.dim).map(|idx| ((idx * 3 + 5) % 256) as f32 / 255.0).collect();
        ModalityEmbedding::new(data, ModalityType::Audio, seq_len, self.dim)
    }

    fn get_dim(&self) -> usize {
        self.dim
    }

    fn modality(&self) -> ModalityType {
        ModalityType::Audio
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ── Multi-modal pipeline ─────────────────────────────────────────────────────

/// End-to-end multi-modal fusion pipeline: encode → project → fuse → output.
#[derive(Debug)]
pub struct MultiModalPipeline {
    pub configs: HashMap<ModalityType, ModalityConfig>,
    pub projections: HashMap<ModalityType, ProjectionLayer>,
    pub strategy: FusionStrategy,
    pub shared_dim: usize,
    gating: Option<GatingNetwork>,
    cross_attn: Option<CrossAttentionFuser>,
}

impl MultiModalPipeline {
    /// Build a pipeline from per-modality configs and a fusion strategy.
    pub fn new(configs: &[ModalityConfig], strategy: FusionStrategy) -> Result<Self, FusionError> {
        if configs.is_empty() {
            return Err(FusionError::EmptyInput);
        }
        // All projection dims must agree.
        let shared_dim = configs[0].projection_dim;
        for cfg in configs {
            if cfg.projection_dim != shared_dim {
                return Err(FusionError::InvalidConfig(format!(
                    "mismatched projection dims: {} vs {}",
                    shared_dim, cfg.projection_dim
                )));
            }
        }

        // Validate strategy-specific constraints.
        match &strategy {
            FusionStrategy::WeightedSum { weights } => {
                if weights.len() != configs.len() {
                    return Err(FusionError::InvalidConfig(format!(
                        "WeightedSum needs {} weights, got {}",
                        configs.len(),
                        weights.len()
                    )));
                }
            }
            FusionStrategy::CrossAttention { query_modality, key_modality, .. } => {
                let has_q = configs.iter().any(|c| c.modality == *query_modality);
                let has_k = configs.iter().any(|c| c.modality == *key_modality);
                if !has_q || !has_k {
                    return Err(FusionError::InvalidConfig(
                        "CrossAttention references modalities not in configs".to_string(),
                    ));
                }
            }
            _ => {}
        }

        let mut config_map = HashMap::new();
        let mut proj_map = HashMap::new();
        for cfg in configs {
            let proj = ProjectionLayer::identity(cfg.embedding_dim, cfg.projection_dim)?;
            proj_map.insert(cfg.modality, proj);
            config_map.insert(cfg.modality, cfg.clone());
        }

        let gating = match &strategy {
            FusionStrategy::GatedFusion { hidden_dim } => {
                let total_input = shared_dim * configs.len();
                Some(GatingNetwork::new(total_input, configs.len(), *hidden_dim)?)
            }
            _ => None,
        };

        let cross_attn = match &strategy {
            FusionStrategy::CrossAttention { num_heads, .. } => {
                Some(CrossAttentionFuser::new(shared_dim, *num_heads)?)
            }
            _ => None,
        };

        Ok(Self {
            configs: config_map,
            projections: proj_map,
            strategy,
            shared_dim,
            gating,
            cross_attn,
        })
    }

    /// Project a single modality embedding into the shared space.
    pub fn project(&self, embedding: &ModalityEmbedding) -> Result<ModalityEmbedding, FusionError> {
        let proj = self
            .projections
            .get(&embedding.modality)
            .ok_or(FusionError::UnsupportedModality(embedding.modality))?;
        let config = &self.configs[&embedding.modality];
        if embedding.dim != config.embedding_dim {
            return Err(FusionError::DimensionMismatch {
                expected: config.embedding_dim,
                got: embedding.dim,
                modality: embedding.modality,
            });
        }
        if embedding.seq_len > config.max_seq_len {
            return Err(FusionError::SequenceTooLong {
                max: config.max_seq_len,
                got: embedding.seq_len,
            });
        }
        let projected = proj.forward(&embedding.data, embedding.seq_len)?;
        ModalityEmbedding::new(projected, embedding.modality, embedding.seq_len, self.shared_dim)
    }

    /// Fuse a set of projected embeddings. All must already be in shared dim space.
    pub fn fuse(&self, projected: &[ModalityEmbedding]) -> Result<FusionResult, FusionError> {
        if projected.is_empty() {
            return Err(FusionError::EmptyInput);
        }
        for p_emb in projected {
            if p_emb.dim != self.shared_dim {
                return Err(FusionError::DimensionMismatch {
                    expected: self.shared_dim,
                    got: p_emb.dim,
                    modality: p_emb.modality,
                });
            }
        }

        match &self.strategy {
            FusionStrategy::Concatenation => self.fuse_concat(projected),
            FusionStrategy::Attention { num_heads } => self.fuse_attention(projected, *num_heads),
            FusionStrategy::GatedFusion { .. } => self.fuse_gated(projected),
            FusionStrategy::CrossAttention { query_modality, key_modality, .. } => {
                self.fuse_cross_attention(projected, *query_modality, *key_modality)
            }
            FusionStrategy::WeightedSum { weights } => self.fuse_weighted(projected, weights),
        }
    }

    /// Convenience: project + fuse in one call.
    pub fn run(&self, embeddings: &[ModalityEmbedding]) -> Result<FusionResult, FusionError> {
        let projected: Vec<ModalityEmbedding> =
            embeddings.iter().map(|emb| self.project(emb)).collect::<Result<Vec<_>, _>>()?;
        self.fuse(&projected)
    }

    // ── Private fusion implementations ───────────────────────────────────

    fn fuse_concat(&self, projected: &[ModalityEmbedding]) -> Result<FusionResult, FusionError> {
        // Use the minimum sequence length across modalities.
        let seq_len = projected.iter().map(|p| p.seq_len).min().unwrap();
        let fused_dim = projected.len() * self.shared_dim;
        let mut fused = vec![0.0f32; seq_len * fused_dim];
        let mut contributions = HashMap::new();

        for (mi, p_emb) in projected.iter().enumerate() {
            let mut norm_sq = 0.0f32;
            for si in 0..seq_len {
                for di in 0..self.shared_dim {
                    let val = p_emb.get(si, di);
                    fused[si * fused_dim + mi * self.shared_dim + di] = val;
                    norm_sq += val * val;
                }
            }
            contributions.insert(p_emb.modality, norm_sq.sqrt());
        }

        Ok(FusionResult {
            fused_data: fused,
            fused_dim,
            seq_len,
            attention_weights: None,
            gate_values: None,
            modality_contributions: contributions,
        })
    }

    fn fuse_attention(
        &self,
        projected: &[ModalityEmbedding],
        _num_heads: usize,
    ) -> Result<FusionResult, FusionError> {
        // Simplified: compute per-modality energy and softmax.
        let seq_len = projected.iter().map(|p| p.seq_len).min().unwrap();

        // Compute energy per modality (average L2 norm across sequence).
        let energies: Vec<f32> = projected
            .iter()
            .map(|p_emb| {
                let avg: f32 = (0..seq_len)
                    .map(|si| {
                        (0..self.shared_dim)
                            .map(|di| {
                                let val = p_emb.get(si, di);
                                val * val
                            })
                            .sum::<f32>()
                            .sqrt()
                    })
                    .sum::<f32>()
                    / seq_len as f32;
                avg
            })
            .collect();

        // Softmax over energies.
        let max_e = energies.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_e: Vec<f32> = energies.iter().map(|e| (e - max_e).exp()).collect();
        let sum_exp: f32 = exp_e.iter().sum();
        let attn_w: Vec<f32> = exp_e.iter().map(|e| e / sum_exp).collect();

        // Weighted sum.
        let mut fused = vec![0.0f32; seq_len * self.shared_dim];
        let mut contributions = HashMap::new();
        for (mi, p_emb) in projected.iter().enumerate() {
            let weight = attn_w[mi];
            let mut norm_sq = 0.0f32;
            for si in 0..seq_len {
                for di in 0..self.shared_dim {
                    let val = p_emb.get(si, di) * weight;
                    fused[si * self.shared_dim + di] += val;
                    norm_sq += val * val;
                }
            }
            contributions.insert(p_emb.modality, norm_sq.sqrt());
        }

        Ok(FusionResult {
            fused_data: fused,
            fused_dim: self.shared_dim,
            seq_len,
            attention_weights: Some(attn_w),
            gate_values: None,
            modality_contributions: contributions,
        })
    }

    fn fuse_gated(&self, projected: &[ModalityEmbedding]) -> Result<FusionResult, FusionError> {
        let gating = self.gating.as_ref().ok_or_else(|| {
            FusionError::InvalidConfig("gating network not initialized".to_string())
        })?;
        let seq_len = projected.iter().map(|p| p.seq_len).min().unwrap();
        let num = projected.len();

        // Compute gate input: average pool each modality, concatenate.
        let mut gate_input = Vec::with_capacity(num * self.shared_dim);
        for p_emb in projected {
            let mut avg = vec![0.0f32; self.shared_dim];
            for si in 0..seq_len {
                for (di, avg_val) in avg.iter_mut().enumerate() {
                    *avg_val += p_emb.get(si, di);
                }
            }
            for avg_val in &mut avg {
                *avg_val /= seq_len as f32;
            }
            gate_input.extend_from_slice(&avg);
        }

        let gates = gating.forward(&gate_input);
        // Normalize gates to sum to 1.
        let gate_sum: f32 = gates.iter().sum();
        let norm_gates: Vec<f32> = if gate_sum > 0.0 {
            gates.iter().map(|g| g / gate_sum).collect()
        } else {
            vec![1.0 / num as f32; num]
        };

        let mut fused = vec![0.0f32; seq_len * self.shared_dim];
        let mut contributions = HashMap::new();
        for (mi, p_emb) in projected.iter().enumerate() {
            let weight = norm_gates[mi];
            let mut norm_sq = 0.0f32;
            for si in 0..seq_len {
                for di in 0..self.shared_dim {
                    let val = p_emb.get(si, di) * weight;
                    fused[si * self.shared_dim + di] += val;
                    norm_sq += val * val;
                }
            }
            contributions.insert(p_emb.modality, norm_sq.sqrt());
        }

        Ok(FusionResult {
            fused_data: fused,
            fused_dim: self.shared_dim,
            seq_len,
            attention_weights: None,
            gate_values: Some(norm_gates),
            modality_contributions: contributions,
        })
    }

    fn fuse_cross_attention(
        &self,
        projected: &[ModalityEmbedding],
        query_mod: ModalityType,
        key_mod: ModalityType,
    ) -> Result<FusionResult, FusionError> {
        let cross = self.cross_attn.as_ref().ok_or_else(|| {
            FusionError::InvalidConfig("cross-attention fuser not initialized".to_string())
        })?;
        let q_emb = projected
            .iter()
            .find(|p| p.modality == query_mod)
            .ok_or(FusionError::UnsupportedModality(query_mod))?;
        let k_emb = projected
            .iter()
            .find(|p| p.modality == key_mod)
            .ok_or(FusionError::UnsupportedModality(key_mod))?;

        let fused_data = cross.forward(q_emb, k_emb)?;
        let seq_len = q_emb.seq_len;
        let mut contributions = HashMap::new();
        let norm: f32 = fused_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        contributions.insert(query_mod, norm);
        contributions.insert(key_mod, norm);

        Ok(FusionResult {
            fused_data,
            fused_dim: self.shared_dim,
            seq_len,
            attention_weights: None,
            gate_values: None,
            modality_contributions: contributions,
        })
    }

    fn fuse_weighted(
        &self,
        projected: &[ModalityEmbedding],
        weights: &[f32],
    ) -> Result<FusionResult, FusionError> {
        if weights.len() != projected.len() {
            return Err(FusionError::InvalidConfig(format!(
                "weight count ({}) != modality count ({})",
                weights.len(),
                projected.len()
            )));
        }
        let seq_len = projected.iter().map(|p| p.seq_len).min().unwrap();
        let mut fused = vec![0.0f32; seq_len * self.shared_dim];
        let mut contributions = HashMap::new();

        for (mi, p_emb) in projected.iter().enumerate() {
            let weight = weights[mi];
            let mut norm_sq = 0.0f32;
            for si in 0..seq_len {
                for di in 0..self.shared_dim {
                    let val = p_emb.get(si, di) * weight;
                    fused[si * self.shared_dim + di] += val;
                    norm_sq += val * val;
                }
            }
            contributions.insert(p_emb.modality, norm_sq.sqrt());
        }

        Ok(FusionResult {
            fused_data: fused,
            fused_dim: self.shared_dim,
            seq_len,
            attention_weights: None,
            gate_values: None,
            modality_contributions: contributions,
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModalityType tests ───────────────────────────────────────────────

    #[test]
    fn test_modality_type_all_variants() {
        let all = ModalityType::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&ModalityType::Text));
        assert!(all.contains(&ModalityType::Image));
        assert!(all.contains(&ModalityType::Audio));
        assert!(all.contains(&ModalityType::Video));
        assert!(all.contains(&ModalityType::Structured));
    }

    #[test]
    fn test_modality_type_labels() {
        assert_eq!(ModalityType::Text.label(), "text");
        assert_eq!(ModalityType::Image.label(), "image");
        assert_eq!(ModalityType::Audio.label(), "audio");
        assert_eq!(ModalityType::Video.label(), "video");
        assert_eq!(ModalityType::Structured.label(), "structured");
    }

    #[test]
    fn test_modality_type_equality() {
        assert_eq!(ModalityType::Text, ModalityType::Text);
        assert_ne!(ModalityType::Text, ModalityType::Image);
    }

    #[test]
    fn test_modality_type_hash() {
        let mut map = HashMap::new();
        map.insert(ModalityType::Text, 1);
        map.insert(ModalityType::Image, 2);
        assert_eq!(map[&ModalityType::Text], 1);
        assert_eq!(map[&ModalityType::Image], 2);
    }

    #[test]
    fn test_modality_type_clone() {
        let m = ModalityType::Audio;
        let m2 = m;
        assert_eq!(m, m2);
    }

    // ── ModalityConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = ModalityConfig::new(ModalityType::Text, 768, 512, 256).unwrap();
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.projection_dim, 256);
        assert!((cfg.dropout_rate - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_zero_embedding_dim() {
        let result = ModalityConfig::new(ModalityType::Text, 0, 512, 256);
        assert_eq!(result.unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_config_zero_projection_dim() {
        let result = ModalityConfig::new(ModalityType::Text, 768, 512, 0);
        assert_eq!(result.unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_config_with_dropout() {
        let cfg = ModalityConfig::new(ModalityType::Text, 768, 512, 256).unwrap().with_dropout(0.5);
        assert!((cfg.dropout_rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_dropout_clamp_high() {
        let cfg = ModalityConfig::new(ModalityType::Text, 768, 512, 256).unwrap().with_dropout(2.0);
        assert!((cfg.dropout_rate - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_dropout_clamp_low() {
        let cfg =
            ModalityConfig::new(ModalityType::Text, 768, 512, 256).unwrap().with_dropout(-0.5);
        assert!(cfg.dropout_rate.abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_clone_eq() {
        let cfg = ModalityConfig::new(ModalityType::Image, 1024, 196, 512).unwrap();
        let cfg2 = cfg.clone();
        assert_eq!(cfg, cfg2);
    }

    // ── FusionStrategy tests ─────────────────────────────────────────────

    #[test]
    fn test_concat_output_dim() {
        let strat = FusionStrategy::Concatenation;
        assert_eq!(strat.output_dim(3, 256), 768);
    }

    #[test]
    fn test_attention_output_dim() {
        let strat = FusionStrategy::Attention { num_heads: 8 };
        assert_eq!(strat.output_dim(3, 256), 256);
    }

    #[test]
    fn test_gated_output_dim() {
        let strat = FusionStrategy::GatedFusion { hidden_dim: 128 };
        assert_eq!(strat.output_dim(2, 512), 512);
    }

    #[test]
    fn test_weighted_sum_output_dim() {
        let strat = FusionStrategy::WeightedSum { weights: vec![0.5, 0.5] };
        assert_eq!(strat.output_dim(2, 256), 256);
    }

    #[test]
    fn test_cross_attention_output_dim() {
        let strat = FusionStrategy::CrossAttention {
            query_modality: ModalityType::Text,
            key_modality: ModalityType::Image,
            num_heads: 4,
        };
        assert_eq!(strat.output_dim(2, 256), 256);
    }

    // ── ModalityEmbedding tests ──────────────────────────────────────────

    #[test]
    fn test_embedding_new_valid() {
        let data = vec![1.0; 12];
        let emb = ModalityEmbedding::new(data, ModalityType::Text, 3, 4).unwrap();
        assert_eq!(emb.seq_len, 3);
        assert_eq!(emb.dim, 4);
        assert_eq!(emb.numel(), 12);
    }

    #[test]
    fn test_embedding_size_mismatch() {
        let data = vec![1.0; 10];
        let result = ModalityEmbedding::new(data, ModalityType::Text, 3, 4);
        assert!(matches!(result, Err(FusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_embedding_zero_dim() {
        let result = ModalityEmbedding::new(vec![], ModalityType::Text, 0, 0);
        assert_eq!(result.unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_embedding_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let emb = ModalityEmbedding::new(data, ModalityType::Image, 2, 3).unwrap();
        assert!((emb.get(0, 0) - 1.0).abs() < f32::EPSILON);
        assert!((emb.get(0, 2) - 3.0).abs() < f32::EPSILON);
        assert!((emb.get(1, 0) - 4.0).abs() < f32::EPSILON);
        assert!((emb.get(1, 2) - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_embedding_single_element() {
        let emb = ModalityEmbedding::new(vec![42.0], ModalityType::Audio, 1, 1).unwrap();
        assert_eq!(emb.numel(), 1);
        assert!((emb.get(0, 0) - 42.0).abs() < f32::EPSILON);
    }

    // ── ProjectionLayer tests ────────────────────────────────────────────

    #[test]
    fn test_projection_identity_same_dim() {
        let proj = ProjectionLayer::identity(4, 4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = proj.forward(&input, 1).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_projection_identity_truncation() {
        let proj = ProjectionLayer::identity(4, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = proj.forward(&input, 1).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_projection_identity_padding() {
        let proj = ProjectionLayer::identity(2, 4).unwrap();
        let input = vec![5.0, 7.0];
        let out = proj.forward(&input, 1).unwrap();
        assert_eq!(out.len(), 4);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_projection_zero_dim_input() {
        assert_eq!(ProjectionLayer::identity(0, 4).unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_projection_zero_dim_output() {
        assert_eq!(ProjectionLayer::identity(4, 0).unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_projection_batch() {
        let proj = ProjectionLayer::identity(3, 3).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = proj.forward(&input, 2).unwrap();
        assert_eq!(out.len(), 6);
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_projection_wrong_input_size() {
        let proj = ProjectionLayer::identity(3, 3).unwrap();
        let input = vec![1.0, 2.0];
        assert!(matches!(proj.forward(&input, 1), Err(FusionError::ProjectionError(_))));
    }

    #[test]
    fn test_projection_scaled() {
        let proj = ProjectionLayer::scaled(4, 4, 1.0).unwrap();
        assert_eq!(proj.weight.len(), 16);
        assert_eq!(proj.bias.len(), 4);
        let w0 = proj.weight[0];
        assert!(w0 > 0.0);
        for &weight in &proj.weight {
            assert!((weight - w0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_projection_linearity() {
        // f(ax) = a * f(x) for identity projection without bias
        let proj = ProjectionLayer::identity(4, 4).unwrap();
        let x_in = vec![1.0, 2.0, 3.0, 4.0];
        let fx = proj.forward(&x_in, 1).unwrap();
        let ax: Vec<f32> = x_in.iter().map(|v| v * 3.0).collect();
        let fax = proj.forward(&ax, 1).unwrap();
        let scaled_fx: Vec<f32> = fx.iter().map(|v| v * 3.0).collect();
        for (a, b) in fax.iter().zip(scaled_fx.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_projection_additivity() {
        // f(x + y) = f(x) + f(y) for zero-bias identity
        let proj = ProjectionLayer::identity(3, 3).unwrap();
        let x_in = vec![1.0, 2.0, 3.0];
        let y_in = vec![4.0, 5.0, 6.0];
        let xy: Vec<f32> = x_in.iter().zip(y_in.iter()).map(|(a, b)| a + b).collect();
        let fx = proj.forward(&x_in, 1).unwrap();
        let fy = proj.forward(&y_in, 1).unwrap();
        let fxy = proj.forward(&xy, 1).unwrap();
        let fx_plus_fy: Vec<f32> = fx.iter().zip(fy.iter()).map(|(a, b)| a + b).collect();
        for (a, b) in fxy.iter().zip(fx_plus_fy.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ── GatingNetwork tests ──────────────────────────────────────────────

    #[test]
    fn test_gating_output_range() {
        let gn = GatingNetwork::new(16, 3, 8).unwrap();
        let input = vec![0.5; 16];
        let gates = gn.forward(&input);
        assert_eq!(gates.len(), 3);
        for &gate in &gates {
            assert!((0.0..=1.0).contains(&gate), "gate {gate} not in [0,1]");
        }
    }

    #[test]
    fn test_gating_zero_input() {
        let gn = GatingNetwork::new(8, 2, 4).unwrap();
        let input = vec![0.0; 8];
        let gates = gn.forward(&input);
        assert_eq!(gates.len(), 2);
        for &gate in &gates {
            assert!((0.0..=1.0).contains(&gate));
        }
    }

    #[test]
    fn test_gating_large_input() {
        let gn = GatingNetwork::new(8, 2, 4).unwrap();
        let input = vec![100.0; 8];
        let gates = gn.forward(&input);
        for &gate in &gates {
            assert!((0.0..=1.0).contains(&gate));
        }
    }

    #[test]
    fn test_gating_negative_input() {
        let gn = GatingNetwork::new(8, 2, 4).unwrap();
        let input = vec![-5.0; 8];
        let gates = gn.forward(&input);
        for &gate in &gates {
            assert!((0.0..=1.0).contains(&gate));
        }
    }

    #[test]
    fn test_gating_zero_modalities() {
        let result = GatingNetwork::new(8, 0, 4);
        assert_eq!(result.unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_gating_zero_hidden() {
        let result = GatingNetwork::new(8, 2, 0);
        assert_eq!(result.unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_gating_deterministic() {
        let gn = GatingNetwork::new(8, 3, 4).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let g1 = gn.forward(&input);
        let g2 = gn.forward(&input);
        assert_eq!(g1, g2);
    }

    // ── CrossAttentionFuser tests ────────────────────────────────────────

    #[test]
    fn test_cross_attention_new() {
        let ca = CrossAttentionFuser::new(64, 4).unwrap();
        assert_eq!(ca.num_heads, 4);
        assert_eq!(ca.head_dim, 16);
        assert_eq!(ca.model_dim, 64);
    }

    #[test]
    fn test_cross_attention_indivisible() {
        let result = CrossAttentionFuser::new(65, 4);
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn test_cross_attention_zero_dim() {
        assert_eq!(CrossAttentionFuser::new(0, 4).unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_cross_attention_zero_heads() {
        assert_eq!(CrossAttentionFuser::new(64, 0).unwrap_err(), FusionError::ZeroDimension);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let ca = CrossAttentionFuser::new(8, 2).unwrap();
        let query = vec![1.0; 8]; // 1 query position
        let key = vec![1.0; 24]; // 3 key positions
        let weights = ca.attention_weights(&query, &key, 1, 3, 8);
        assert_eq!(weights.len(), 3);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum = {sum}");
    }

    #[test]
    fn test_attention_weights_shape() {
        let ca = CrossAttentionFuser::new(4, 1).unwrap();
        let query = vec![1.0; 8]; // 2 query positions × 4 dim
        let key = vec![1.0; 12]; // 3 key positions × 4 dim
        let weights = ca.attention_weights(&query, &key, 2, 3, 4);
        assert_eq!(weights.len(), 6); // 2 × 3
    }

    #[test]
    fn test_attention_weights_per_row_sum() {
        let ca = CrossAttentionFuser::new(4, 1).unwrap();
        let query = vec![0.5; 12]; // 3 × 4
        let key = vec![0.5; 8]; // 2 × 4
        let weights = ca.attention_weights(&query, &key, 3, 2, 4);
        for row in 0..3 {
            let sum: f32 = weights[row * 2..(row + 1) * 2].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cross_attention_forward_shape() {
        let ca = CrossAttentionFuser::new(8, 2).unwrap();
        let q_emb = ModalityEmbedding::new(vec![1.0; 16], ModalityType::Text, 2, 8).unwrap();
        let kv_emb = ModalityEmbedding::new(vec![1.0; 24], ModalityType::Image, 3, 8).unwrap();
        let out = ca.forward(&q_emb, &kv_emb).unwrap();
        assert_eq!(out.len(), 16); // 2 × 8
    }

    #[test]
    fn test_cross_attention_dim_mismatch() {
        let ca = CrossAttentionFuser::new(8, 2).unwrap();
        let q_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        let kv_emb = ModalityEmbedding::new(vec![1.0; 24], ModalityType::Image, 3, 8).unwrap();
        assert!(matches!(ca.forward(&q_emb, &kv_emb), Err(FusionError::DimensionMismatch { .. })));
    }

    // ── FusionResult tests ───────────────────────────────────────────────

    #[test]
    fn test_fusion_result_numel() {
        let result = FusionResult {
            fused_data: vec![1.0; 20],
            fused_dim: 4,
            seq_len: 5,
            attention_weights: None,
            gate_values: None,
            modality_contributions: HashMap::new(),
        };
        assert_eq!(result.numel(), 20);
    }

    #[test]
    fn test_fusion_result_norm() {
        let result = FusionResult {
            fused_data: vec![3.0, 4.0],
            fused_dim: 2,
            seq_len: 1,
            attention_weights: None,
            gate_values: None,
            modality_contributions: HashMap::new(),
        };
        assert!((result.norm() - 5.0).abs() < 1e-5);
    }

    // ── Stub encoder tests ───────────────────────────────────────────────

    #[test]
    fn test_stub_text_encoder() {
        let enc = StubTextEncoder { dim: 16 };
        assert_eq!(enc.get_dim(), 16);
        assert_eq!(enc.modality(), ModalityType::Text);
        assert!(enc.supports_streaming());
        let emb = enc.encode(b"hello").unwrap();
        assert_eq!(emb.seq_len, 5);
        assert_eq!(emb.dim, 16);
    }

    #[test]
    fn test_stub_text_encoder_empty() {
        let enc = StubTextEncoder { dim: 16 };
        assert_eq!(enc.encode(b"").unwrap_err(), FusionError::EmptyInput);
    }

    #[test]
    fn test_stub_image_encoder() {
        let enc = StubImageEncoder { dim: 32, patch_size: 4 };
        assert_eq!(enc.get_dim(), 32);
        assert_eq!(enc.modality(), ModalityType::Image);
        assert!(!enc.supports_streaming());
        let emb = enc.encode(&[0u8; 16]).unwrap();
        assert_eq!(emb.seq_len, 4); // 16/4
        assert_eq!(emb.dim, 32);
    }

    #[test]
    fn test_stub_image_encoder_empty() {
        let enc = StubImageEncoder { dim: 32, patch_size: 4 };
        assert_eq!(enc.encode(b"").unwrap_err(), FusionError::EmptyInput);
    }

    #[test]
    fn test_stub_audio_encoder() {
        let enc = StubAudioEncoder { dim: 64, sample_rate: 16000 };
        assert_eq!(enc.get_dim(), 64);
        assert_eq!(enc.modality(), ModalityType::Audio);
        assert!(enc.supports_streaming());
        let emb = enc.encode(&[0u8; 3200]).unwrap();
        assert!(emb.seq_len > 0);
        assert_eq!(emb.dim, 64);
    }

    #[test]
    fn test_stub_audio_encoder_empty() {
        let enc = StubAudioEncoder { dim: 64, sample_rate: 16000 };
        assert_eq!(enc.encode(b"").unwrap_err(), FusionError::EmptyInput);
    }

    // ── Pipeline construction tests ──────────────────────────────────────

    fn text_config(dim: usize, proj: usize) -> ModalityConfig {
        ModalityConfig::new(ModalityType::Text, dim, 512, proj).unwrap()
    }

    fn image_config(dim: usize, proj: usize) -> ModalityConfig {
        ModalityConfig::new(ModalityType::Image, dim, 196, proj).unwrap()
    }

    fn audio_config(dim: usize, proj: usize) -> ModalityConfig {
        ModalityConfig::new(ModalityType::Audio, dim, 1024, proj).unwrap()
    }

    #[test]
    fn test_pipeline_new_concat() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(768, 256), image_config(1024, 256)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        assert_eq!(pipeline.shared_dim, 256);
        assert_eq!(pipeline.configs.len(), 2);
    }

    #[test]
    fn test_pipeline_empty_configs() {
        let result = MultiModalPipeline::new(&[], FusionStrategy::Concatenation);
        assert_eq!(result.unwrap_err(), FusionError::EmptyInput);
    }

    #[test]
    fn test_pipeline_mismatched_projection_dims() {
        let result = MultiModalPipeline::new(
            &[text_config(768, 256), image_config(1024, 512)],
            FusionStrategy::Concatenation,
        );
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn test_pipeline_weighted_wrong_count() {
        let result = MultiModalPipeline::new(
            &[text_config(768, 256), image_config(1024, 256)],
            FusionStrategy::WeightedSum { weights: vec![0.5] },
        );
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn test_pipeline_cross_attention_missing_modality() {
        let result = MultiModalPipeline::new(
            &[text_config(768, 256)],
            FusionStrategy::CrossAttention {
                query_modality: ModalityType::Text,
                key_modality: ModalityType::Image,
                num_heads: 4,
            },
        );
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    // ── Pipeline projection tests ────────────────────────────────────────

    #[test]
    fn test_pipeline_project_valid() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(4, 4)], FusionStrategy::Concatenation).unwrap();
        let emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        let proj = pipeline.project(&emb).unwrap();
        assert_eq!(proj.dim, 4);
        assert_eq!(proj.seq_len, 2);
    }

    #[test]
    fn test_pipeline_project_unsupported_modality() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(4, 4)], FusionStrategy::Concatenation).unwrap();
        let emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Image, 2, 4).unwrap();
        assert!(matches!(pipeline.project(&emb), Err(FusionError::UnsupportedModality(_))));
    }

    #[test]
    fn test_pipeline_project_dim_mismatch() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(8, 4)], FusionStrategy::Concatenation).unwrap();
        let emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        assert!(matches!(pipeline.project(&emb), Err(FusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_pipeline_project_seq_too_long() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(4, 4)], FusionStrategy::Concatenation).unwrap();
        // max_seq_len is 512 for text; create 600 tokens.
        let emb = ModalityEmbedding::new(vec![1.0; 600 * 4], ModalityType::Text, 600, 4).unwrap();
        assert!(matches!(pipeline.project(&emb), Err(FusionError::SequenceTooLong { .. })));
    }

    // ── Concatenation fusion tests ───────────────────────────────────────

    #[test]
    fn test_fuse_concat_two_modalities() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 8], ModalityType::Image, 2, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 8); // 2 × 4
        assert_eq!(result.seq_len, 2);
        assert_eq!(result.fused_data.len(), 16); // 2 × 8
    }

    #[test]
    fn test_fuse_concat_preserves_data() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(2, 2), image_config(2, 2)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0, 2.0], ModalityType::Text, 1, 2).unwrap();
        let img_emb = ModalityEmbedding::new(vec![3.0, 4.0], ModalityType::Image, 1, 2).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert!((result.fused_data[0] - 1.0).abs() < 1e-6);
        assert!((result.fused_data[1] - 2.0).abs() < 1e-6);
        assert!((result.fused_data[2] - 3.0).abs() < 1e-6);
        assert!((result.fused_data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_concat_contributions() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(2, 2), image_config(2, 2)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![3.0, 4.0], ModalityType::Text, 1, 2).unwrap();
        let img_emb = ModalityEmbedding::new(vec![0.0, 0.0], ModalityType::Image, 1, 2).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert!((result.modality_contributions[&ModalityType::Text] - 5.0).abs() < 1e-5);
        assert!(result.modality_contributions[&ModalityType::Image].abs() < 1e-5);
    }

    // ── Weighted sum fusion tests ────────────────────────────────────────

    #[test]
    fn test_fuse_weighted_sum_equal() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::WeightedSum { weights: vec![0.5, 0.5] },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![4.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 4);
        // 0.5*2 + 0.5*4 = 3.0
        for &val in &result.fused_data {
            assert!((val - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fuse_weighted_sum_single_weight_1() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::WeightedSum { weights: vec![1.0, 0.0] },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![7.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![99.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        for &val in &result.fused_data {
            assert!((val - 7.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fuse_weighted_wrong_count() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::WeightedSum { weights: vec![0.5, 0.5] },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        // Only one embedding but pipeline expects 2 weights → internal check.
        let result = pipeline.fuse(&[text_emb]);
        assert!(result.is_err());
    }

    // ── Attention fusion tests ───────────────────────────────────────────

    #[test]
    fn test_fuse_attention_output_dim() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::Attention { num_heads: 2 },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Image, 2, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 4);
        assert!(result.attention_weights.is_some());
        let attn_w = result.attention_weights.unwrap();
        let sum: f32 = attn_w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_attention_weights_sum_to_one() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4), audio_config(4, 4)],
            FusionStrategy::Attention { num_heads: 2 },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let aud_emb = ModalityEmbedding::new(vec![3.0; 4], ModalityType::Audio, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb, aud_emb]).unwrap();
        let attn_w = result.attention_weights.unwrap();
        assert_eq!(attn_w.len(), 3);
        let sum: f32 = attn_w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── Gated fusion tests ───────────────────────────────────────────────

    #[test]
    fn test_fuse_gated_output_dim() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::GatedFusion { hidden_dim: 8 },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 4);
        assert!(result.gate_values.is_some());
    }

    #[test]
    fn test_fuse_gated_gate_values_sum_to_one() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::GatedFusion { hidden_dim: 8 },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        let gates = result.gate_values.unwrap();
        let sum: f32 = gates.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "gate values sum to {sum}, not 1.0");
    }

    #[test]
    fn test_fuse_gated_gate_range() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4), audio_config(4, 4)],
            FusionStrategy::GatedFusion { hidden_dim: 16 },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let aud_emb = ModalityEmbedding::new(vec![3.0; 4], ModalityType::Audio, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb, aud_emb]).unwrap();
        for &gate in result.gate_values.as_ref().unwrap() {
            assert!((0.0..=1.0).contains(&gate));
        }
    }

    // ── Cross-attention fusion tests ─────────────────────────────────────

    #[test]
    fn test_fuse_cross_attention_shape() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(8, 8), image_config(8, 8)],
            FusionStrategy::CrossAttention {
                query_modality: ModalityType::Text,
                key_modality: ModalityType::Image,
                num_heads: 2,
            },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 16], ModalityType::Text, 2, 8).unwrap();
        let img_emb = ModalityEmbedding::new(vec![1.0; 24], ModalityType::Image, 3, 8).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 8);
        assert_eq!(result.seq_len, 2);
        assert_eq!(result.fused_data.len(), 16);
    }

    #[test]
    fn test_fuse_cross_attention_missing_modality() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(8, 8), image_config(8, 8)],
            FusionStrategy::CrossAttention {
                query_modality: ModalityType::Text,
                key_modality: ModalityType::Image,
                num_heads: 2,
            },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 1, 8).unwrap();
        // Only text, no image → should error.
        assert!(matches!(pipeline.fuse(&[text_emb]), Err(FusionError::UnsupportedModality(_))));
    }

    // ── Pipeline end-to-end tests ────────────────────────────────────────

    #[test]
    fn test_pipeline_run_concat() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 8], ModalityType::Text, 2, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 8], ModalityType::Image, 2, 4).unwrap();
        let result = pipeline.run(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.fused_dim, 8);
        assert_eq!(result.seq_len, 2);
    }

    #[test]
    fn test_pipeline_run_weighted() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::WeightedSum { weights: vec![0.3, 0.7] },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![10.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![20.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.run(&[text_emb, img_emb]).unwrap();
        // 0.3*10 + 0.7*20 = 3 + 14 = 17
        for &val in &result.fused_data {
            assert!((val - 17.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_pipeline_run_three_modalities() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4), audio_config(4, 4)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 4], ModalityType::Text, 1, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let aud_emb = ModalityEmbedding::new(vec![3.0; 4], ModalityType::Audio, 1, 4).unwrap();
        let result = pipeline.run(&[text_emb, img_emb, aud_emb]).unwrap();
        assert_eq!(result.fused_dim, 12); // 3 × 4
    }

    #[test]
    fn test_pipeline_single_modality_concat() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(4, 4)], FusionStrategy::Concatenation).unwrap();
        let text_emb = ModalityEmbedding::new(vec![5.0; 8], ModalityType::Text, 2, 4).unwrap();
        let result = pipeline.run(&[text_emb]).unwrap();
        assert_eq!(result.fused_dim, 4);
        for &val in &result.fused_data {
            assert!((val - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pipeline_run_empty() {
        let pipeline =
            MultiModalPipeline::new(&[text_config(4, 4)], FusionStrategy::Concatenation).unwrap();
        assert_eq!(pipeline.run(&[]).unwrap_err(), FusionError::EmptyInput);
    }

    // ── Error display tests ──────────────────────────────────────────────

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = FusionError::DimensionMismatch {
            expected: 256,
            got: 128,
            modality: ModalityType::Text,
        };
        let msg = format!("{err}");
        assert!(msg.contains("256"));
        assert!(msg.contains("128"));
    }

    #[test]
    fn test_error_display_empty_input() {
        let msg = format!("{}", FusionError::EmptyInput);
        assert!(msg.contains("no modality"));
    }

    #[test]
    fn test_error_display_zero_dim() {
        let msg = format!("{}", FusionError::ZeroDimension);
        assert!(msg.contains("zero"));
    }

    #[test]
    fn test_error_display_seq_too_long() {
        let err = FusionError::SequenceTooLong { max: 512, got: 600 };
        let msg = format!("{err}");
        assert!(msg.contains("512"));
        assert!(msg.contains("600"));
    }

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_single_element_embedding_fusion() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(1, 1), image_config(1, 1)],
            FusionStrategy::WeightedSum { weights: vec![0.5, 0.5] },
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![10.0], ModalityType::Text, 1, 1).unwrap();
        let img_emb = ModalityEmbedding::new(vec![20.0], ModalityType::Image, 1, 1).unwrap();
        let result = pipeline.run(&[text_emb, img_emb]).unwrap();
        assert!((result.fused_data[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_large_dimension_projection() {
        let proj = ProjectionLayer::identity(1024, 256).unwrap();
        let input = vec![1.0; 1024];
        let out = proj.forward(&input, 1).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_many_modalities_concat() {
        let configs: Vec<_> = [
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Audio,
            ModalityType::Video,
            ModalityType::Structured,
        ]
        .iter()
        .map(|&m| ModalityConfig::new(m, 4, 128, 4).unwrap())
        .collect();

        let pipeline = MultiModalPipeline::new(&configs, FusionStrategy::Concatenation).unwrap();
        let embeddings: Vec<_> = [
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Audio,
            ModalityType::Video,
            ModalityType::Structured,
        ]
        .iter()
        .map(|&m| ModalityEmbedding::new(vec![1.0; 4], m, 1, 4).unwrap())
        .collect();
        let result = pipeline.run(&embeddings).unwrap();
        assert_eq!(result.fused_dim, 20); // 5 × 4
    }

    #[test]
    fn test_different_seq_lens_use_minimum() {
        let pipeline = MultiModalPipeline::new(
            &[text_config(4, 4), image_config(4, 4)],
            FusionStrategy::Concatenation,
        )
        .unwrap();
        let text_emb = ModalityEmbedding::new(vec![1.0; 12], ModalityType::Text, 3, 4).unwrap();
        let img_emb = ModalityEmbedding::new(vec![2.0; 4], ModalityType::Image, 1, 4).unwrap();
        let result = pipeline.fuse(&[text_emb, img_emb]).unwrap();
        assert_eq!(result.seq_len, 1); // min(3, 1)
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_relu_behavior() {
        assert!((relu(5.0) - 5.0).abs() < f32::EPSILON);
        assert!(relu(-5.0).abs() < f32::EPSILON);
        assert!(relu(0.0).abs() < f32::EPSILON);
    }
}
