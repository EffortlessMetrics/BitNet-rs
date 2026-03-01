//! Model architecture module for GPU HAL.
//!
//! Provides architecture configuration, detection, validation, optimization,
//! and serialization for transformer-based models. Supports LLaMA, GPT,
//! BitNet, Mistral, and Qwen architecture families with a unified DAG-based
//! model graph representation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ─── ArchitectureFamily ──────────────────────────────────────────────────────

/// Supported model architecture families.
///
/// Each variant maps to a known transformer topology with specific structural
/// expectations (attention patterns, normalization placement, FFN layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArchitectureFamily {
    /// Meta LLaMA family — RMSNorm, rotary embeddings, SwiGLU FFN.
    LLaMA,
    /// OpenAI GPT family — LayerNorm, learned positional embeddings, GELU FFN.
    GPT,
    /// Microsoft BitNet family — ternary weights, BitLinear layers.
    BitNet,
    /// Mistral family — sliding-window attention, grouped-query attention.
    Mistral,
    /// Alibaba Qwen family — RMSNorm, SwiGLU, NTK-aware rotary embeddings.
    Qwen,
}

impl fmt::Display for ArchitectureFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LLaMA => write!(f, "LLaMA"),
            Self::GPT => write!(f, "GPT"),
            Self::BitNet => write!(f, "BitNet"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Qwen => write!(f, "Qwen"),
        }
    }
}

impl ArchitectureFamily {
    /// Returns whether this family uses RMSNorm (vs LayerNorm).
    pub fn uses_rms_norm(self) -> bool {
        matches!(self, Self::LLaMA | Self::BitNet | Self::Mistral | Self::Qwen)
    }

    /// Returns whether this family uses rotary positional embeddings.
    pub fn uses_rotary_embeddings(self) -> bool {
        matches!(self, Self::LLaMA | Self::BitNet | Self::Mistral | Self::Qwen)
    }

    /// Returns whether this family uses grouped-query attention.
    pub fn uses_grouped_query_attention(self) -> bool {
        matches!(self, Self::LLaMA | Self::Mistral | Self::Qwen)
    }

    /// Returns the default FFN multiplier relative to hidden dimension.
    pub fn default_ffn_multiplier(self) -> f64 {
        match self {
            Self::GPT => 4.0,
            _ => 2.6875, // 8/3 rounded, typical SwiGLU
        }
    }

    /// All known families.
    pub fn all() -> &'static [ArchitectureFamily] {
        &[Self::LLaMA, Self::GPT, Self::BitNet, Self::Mistral, Self::Qwen]
    }
}

// ─── AttentionType ───────────────────────────────────────────────────────────

/// Type of attention mechanism used by the transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard multi-head attention.
    MultiHead,
    /// Grouped-query attention (fewer KV heads than query heads).
    GroupedQuery,
    /// Multi-query attention (single KV head shared across all query heads).
    MultiQuery,
    /// Sliding-window attention with a fixed window size.
    SlidingWindow { window_size: usize },
}

// ─── NormType ────────────────────────────────────────────────────────────────

/// Normalization layer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormType {
    /// Standard Layer Normalization.
    LayerNorm,
    /// Root Mean Square Layer Normalization.
    RMSNorm,
}

// ─── QuantizationInfo ────────────────────────────────────────────────────────

/// Quantization metadata for a layer or tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization scheme name (e.g. "I2_S", "QK256", "F16").
    pub scheme: String,
    /// Bits per weight element.
    pub bits: u8,
    /// Block size for block-quantized formats (0 if not applicable).
    pub block_size: usize,
    /// Whether this uses symmetric quantization.
    pub symmetric: bool,
}

impl QuantizationInfo {
    /// Creates a new quantization info descriptor.
    pub fn new(scheme: impl Into<String>, bits: u8, block_size: usize, symmetric: bool) -> Self {
        Self {
            scheme: scheme.into(),
            bits,
            block_size,
            symmetric,
        }
    }

    /// Predefined info for BitNet ternary (I2_S, 2-bit symmetric).
    pub fn bitnet_ternary() -> Self {
        Self::new("I2_S", 2, 32, true)
    }

    /// Predefined info for QK256 format.
    pub fn qk256() -> Self {
        Self::new("QK256", 2, 256, true)
    }

    /// Predefined info for FP16.
    pub fn fp16() -> Self {
        Self::new("F16", 16, 0, false)
    }
}

// ─── ArchConfig ──────────────────────────────────────────────────────────────

/// Top-level model architecture configuration.
///
/// Captures the essential structural parameters of a transformer model:
/// layer count, dimensionality, head configuration, vocabulary, and context
/// length. This is the CPU-side reference that the GPU HAL uses to plan
/// memory allocation and kernel dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    /// Human-readable model name (e.g. "bitnet-b1.58-2B-4T").
    pub model_name: String,
    /// Architecture family this model belongs to.
    pub family: ArchitectureFamily,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (may differ from num_heads for GQA).
    pub num_kv_heads: usize,
    /// Hidden dimension (model width).
    pub hidden_dim: usize,
    /// Intermediate/FFN dimension.
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context/sequence length.
    pub max_seq_len: usize,
    /// Head dimension (typically hidden_dim / num_heads).
    pub head_dim: usize,
    /// Optional architecture-specific metadata.
    pub metadata: HashMap<String, String>,
}

impl ArchConfig {
    /// Creates a new architecture configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: impl Into<String>,
        family: ArchitectureFamily,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Self {
        let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };
        Self {
            model_name: model_name.into(),
            family,
            num_layers,
            num_heads,
            num_kv_heads,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            max_seq_len,
            head_dim,
            metadata: HashMap::new(),
        }
    }

    /// Adds a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Total parameter count estimate (excluding embeddings).
    pub fn estimated_param_count(&self) -> u64 {
        let attn_params = self.num_layers as u64
            * (4 * self.hidden_dim as u64 * self.hidden_dim as u64);
        let ffn_params =
            self.num_layers as u64 * (3 * self.hidden_dim as u64 * self.intermediate_dim as u64);
        let embed_params = self.vocab_size as u64 * self.hidden_dim as u64;
        attn_params + ffn_params + embed_params
    }

    /// Memory estimate in bytes for given bits-per-weight.
    pub fn estimated_memory_bytes(&self, bits_per_weight: u8) -> u64 {
        self.estimated_param_count() * u64::from(bits_per_weight) / 8
    }

    /// Creates a standard BitNet-b1.58-2B config.
    pub fn bitnet_2b() -> Self {
        Self::new("bitnet-b1.58-2B-4T", ArchitectureFamily::BitNet, 24, 32, 32, 2560, 6912, 32000, 4096)
    }

    /// Creates a standard LLaMA-7B config.
    pub fn llama_7b() -> Self {
        Self::new("llama-7b", ArchitectureFamily::LLaMA, 32, 32, 32, 4096, 11008, 32000, 4096)
    }

    /// Creates a standard GPT-2 config.
    pub fn gpt2() -> Self {
        Self::new("gpt2", ArchitectureFamily::GPT, 12, 12, 12, 768, 3072, 50257, 1024)
    }
}

// ─── TransformerConfig ───────────────────────────────────────────────────────

/// Transformer-specific configuration details.
///
/// Extends `ArchConfig` with structural information about normalization
/// placement, attention type, activation functions, and positional encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Whether normalization is applied before attention (pre-norm) or after (post-norm).
    pub pre_norm: bool,
    /// Normalization type used.
    pub norm_type: NormType,
    /// Attention mechanism type.
    pub attention_type: AttentionType,
    /// Activation function name (e.g. "silu", "gelu", "relu").
    pub activation: String,
    /// Whether the FFN uses a gated architecture (SwiGLU, GeGLU, etc.).
    pub gated_ffn: bool,
    /// Whether the model ties input and output embeddings.
    pub tie_embeddings: bool,
    /// Rotary embedding base frequency (0.0 if not using RoPE).
    pub rope_base: f64,
    /// RoPE scaling factor (1.0 = no scaling).
    pub rope_scaling: f64,
    /// Layer norm epsilon.
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Creates a new transformer config.
    pub fn new(
        pre_norm: bool,
        norm_type: NormType,
        attention_type: AttentionType,
        activation: impl Into<String>,
    ) -> Self {
        Self {
            pre_norm,
            norm_type,
            attention_type,
            activation: activation.into(),
            gated_ffn: false,
            tie_embeddings: false,
            rope_base: 0.0,
            rope_scaling: 1.0,
            norm_eps: 1e-5,
        }
    }

    /// Sets gated FFN.
    pub fn with_gated_ffn(mut self, gated: bool) -> Self {
        self.gated_ffn = gated;
        self
    }

    /// Sets embedding tying.
    pub fn with_tie_embeddings(mut self, tie: bool) -> Self {
        self.tie_embeddings = tie;
        self
    }

    /// Sets RoPE parameters.
    pub fn with_rope(mut self, base: f64, scaling: f64) -> Self {
        self.rope_base = base;
        self.rope_scaling = scaling;
        self
    }

    /// Sets norm epsilon.
    pub fn with_norm_eps(mut self, eps: f64) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Default config for LLaMA-style models.
    pub fn llama_default() -> Self {
        Self::new(true, NormType::RMSNorm, AttentionType::MultiHead, "silu")
            .with_gated_ffn(true)
            .with_rope(10000.0, 1.0)
            .with_norm_eps(1e-5)
    }

    /// Default config for GPT-style models.
    pub fn gpt_default() -> Self {
        Self::new(true, NormType::LayerNorm, AttentionType::MultiHead, "gelu")
            .with_tie_embeddings(true)
            .with_norm_eps(1e-5)
    }

    /// Default config for BitNet-style models.
    pub fn bitnet_default() -> Self {
        Self::new(true, NormType::RMSNorm, AttentionType::MultiHead, "silu")
            .with_gated_ffn(true)
            .with_rope(10000.0, 1.0)
            .with_norm_eps(1e-6)
    }

    /// Default config for Mistral-style models.
    pub fn mistral_default() -> Self {
        Self::new(
            true,
            NormType::RMSNorm,
            AttentionType::SlidingWindow { window_size: 4096 },
            "silu",
        )
        .with_gated_ffn(true)
        .with_rope(10000.0, 1.0)
        .with_norm_eps(1e-5)
    }

    /// Default config for Qwen-style models.
    pub fn qwen_default() -> Self {
        Self::new(true, NormType::RMSNorm, AttentionType::MultiHead, "silu")
            .with_gated_ffn(true)
            .with_rope(1000000.0, 1.0)
            .with_norm_eps(1e-6)
    }
}

// ─── LayerType ───────────────────────────────────────────────────────────────

/// Type of a layer in the model graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Token + positional embedding lookup.
    Embedding,
    /// Self-attention layer.
    SelfAttention,
    /// Feed-forward network layer.
    FeedForward,
    /// Normalization layer (LayerNorm or RMSNorm).
    Normalization,
    /// Output projection / LM head.
    OutputProjection,
    /// Residual connection.
    Residual,
    /// BitLinear ternary quantized linear.
    BitLinear,
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedding => write!(f, "Embedding"),
            Self::SelfAttention => write!(f, "SelfAttention"),
            Self::FeedForward => write!(f, "FeedForward"),
            Self::Normalization => write!(f, "Normalization"),
            Self::OutputProjection => write!(f, "OutputProjection"),
            Self::Residual => write!(f, "Residual"),
            Self::BitLinear => write!(f, "BitLinear"),
        }
    }
}

// ─── LayerDef ────────────────────────────────────────────────────────────────

/// Definition of a single layer in the model graph.
///
/// Contains the layer type, its position, dimensional information, and
/// optional quantization metadata. The CPU reference implementation uses
/// this to validate that the GPU kernel dispatch plan covers all layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDef {
    /// Unique layer identifier (e.g. "layer_0_attn", "embed").
    pub id: String,
    /// Layer type.
    pub layer_type: LayerType,
    /// Layer index within its block (0-based).
    pub index: usize,
    /// Input dimensions [rows, cols] or [vocab, dim] for embeddings.
    pub input_dims: [usize; 2],
    /// Output dimensions [rows, cols].
    pub output_dims: [usize; 2],
    /// Quantization info, if the layer uses quantized weights.
    pub quantization: Option<QuantizationInfo>,
    /// IDs of layers that feed into this layer.
    pub inputs: Vec<String>,
}

impl LayerDef {
    /// Creates a new layer definition.
    pub fn new(
        id: impl Into<String>,
        layer_type: LayerType,
        index: usize,
        input_dims: [usize; 2],
        output_dims: [usize; 2],
    ) -> Self {
        Self {
            id: id.into(),
            layer_type,
            index,
            input_dims,
            output_dims,
            quantization: None,
            inputs: Vec::new(),
        }
    }

    /// Attaches quantization info.
    pub fn with_quantization(mut self, quant: QuantizationInfo) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Adds an input dependency.
    pub fn with_input(mut self, input_id: impl Into<String>) -> Self {
        self.inputs.push(input_id.into());
        self
    }

    /// Returns estimated parameter count for this layer.
    pub fn param_count(&self) -> u64 {
        self.input_dims[0] as u64 * self.input_dims[1] as u64
    }

    /// Returns true if the layer is quantized.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}

// ─── ModelGraph ──────────────────────────────────────────────────────────────

/// Directed acyclic graph of layers representing the full model architecture.
///
/// Layers are stored in topological order. The graph enables the GPU HAL
/// to plan execution order, identify parallelism opportunities, and allocate
/// intermediate buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    /// Ordered list of layer definitions (topological order).
    pub layers: Vec<LayerDef>,
    /// Mapping from layer ID to index in `layers`.
    layer_index: HashMap<String, usize>,
}

impl ModelGraph {
    /// Creates an empty model graph.
    pub fn new() -> Self {
        Self { layers: Vec::new(), layer_index: HashMap::new() }
    }

    /// Adds a layer to the graph. Returns an error if the ID is duplicate
    /// or any input dependency references a nonexistent layer.
    pub fn add_layer(&mut self, layer: LayerDef) -> Result<(), ArchError> {
        if self.layer_index.contains_key(&layer.id) {
            return Err(ArchError::DuplicateLayer(layer.id.clone()));
        }
        for input in &layer.inputs {
            if !self.layer_index.contains_key(input) {
                return Err(ArchError::MissingDependency {
                    layer: layer.id.clone(),
                    dependency: input.clone(),
                });
            }
        }
        let idx = self.layers.len();
        self.layer_index.insert(layer.id.clone(), idx);
        self.layers.push(layer);
        Ok(())
    }

    /// Returns a layer by ID.
    pub fn get_layer(&self, id: &str) -> Option<&LayerDef> {
        self.layer_index.get(id).map(|&i| &self.layers[i])
    }

    /// Total number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the graph has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Total parameter count across all layers.
    pub fn total_params(&self) -> u64 {
        self.layers.iter().map(LayerDef::param_count).sum()
    }

    /// Returns all layers of a given type.
    pub fn layers_of_type(&self, layer_type: LayerType) -> Vec<&LayerDef> {
        self.layers.iter().filter(|l| l.layer_type == layer_type).collect()
    }

    /// Returns all layer IDs that directly depend on the given layer.
    pub fn dependents(&self, layer_id: &str) -> Vec<&str> {
        self.layers
            .iter()
            .filter(|l| l.inputs.iter().any(|i| i == layer_id))
            .map(|l| l.id.as_str())
            .collect()
    }

    /// Returns the depth (longest path) from any root to each layer.
    pub fn layer_depths(&self) -> HashMap<String, usize> {
        let mut depths = HashMap::new();
        for layer in &self.layers {
            let depth = if layer.inputs.is_empty() {
                0
            } else {
                layer
                    .inputs
                    .iter()
                    .filter_map(|id| depths.get(id))
                    .max()
                    .copied()
                    .unwrap_or(0)
                    + 1
            };
            depths.insert(layer.id.clone(), depth);
        }
        depths
    }

    /// Validates that the graph is a proper DAG with no orphaned references.
    pub fn validate(&self) -> Result<(), ArchError> {
        for layer in &self.layers {
            for input in &layer.inputs {
                if !self.layer_index.contains_key(input) {
                    return Err(ArchError::MissingDependency {
                        layer: layer.id.clone(),
                        dependency: input.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Builds a standard transformer graph from architecture and transformer configs.
    pub fn from_config(arch: &ArchConfig, _transformer: &TransformerConfig) -> Self {
        let mut graph = Self::new();

        // Embedding layer
        let embed =
            LayerDef::new("embed", LayerType::Embedding, 0, [arch.vocab_size, arch.hidden_dim], [1, arch.hidden_dim]);
        graph.add_layer(embed).expect("embed layer");

        let mut prev_id = "embed".to_string();

        for i in 0..arch.num_layers {
            // Pre-attention norm
            let norm_attn_id = format!("layer_{i}_norm_attn");
            let norm_attn = LayerDef::new(
                &norm_attn_id,
                LayerType::Normalization,
                i,
                [1, arch.hidden_dim],
                [1, arch.hidden_dim],
            )
            .with_input(&prev_id);
            graph.add_layer(norm_attn).expect("norm_attn layer");

            // Self-attention
            let attn_id = format!("layer_{i}_attn");
            let attn = LayerDef::new(
                &attn_id,
                LayerType::SelfAttention,
                i,
                [arch.hidden_dim, arch.hidden_dim],
                [arch.hidden_dim, arch.hidden_dim],
            )
            .with_input(&norm_attn_id);
            graph.add_layer(attn).expect("attn layer");

            // Residual after attention
            let res_attn_id = format!("layer_{i}_res_attn");
            let res_attn = LayerDef::new(
                &res_attn_id,
                LayerType::Residual,
                i,
                [1, arch.hidden_dim],
                [1, arch.hidden_dim],
            )
            .with_input(&attn_id)
            .with_input(&prev_id);
            graph.add_layer(res_attn).expect("res_attn layer");

            // Pre-FFN norm
            let norm_ffn_id = format!("layer_{i}_norm_ffn");
            let norm_ffn = LayerDef::new(
                &norm_ffn_id,
                LayerType::Normalization,
                i,
                [1, arch.hidden_dim],
                [1, arch.hidden_dim],
            )
            .with_input(&res_attn_id);
            graph.add_layer(norm_ffn).expect("norm_ffn layer");

            // Feed-forward
            let ffn_id = format!("layer_{i}_ffn");
            let ffn_layer_type = if arch.family == ArchitectureFamily::BitNet {
                LayerType::BitLinear
            } else {
                LayerType::FeedForward
            };
            let ffn = LayerDef::new(
                &ffn_id,
                ffn_layer_type,
                i,
                [arch.hidden_dim, arch.intermediate_dim],
                [arch.intermediate_dim, arch.hidden_dim],
            )
            .with_input(&norm_ffn_id);
            graph.add_layer(ffn).expect("ffn layer");

            // Residual after FFN
            let res_ffn_id = format!("layer_{i}_res_ffn");
            let res_ffn = LayerDef::new(
                &res_ffn_id,
                LayerType::Residual,
                i,
                [1, arch.hidden_dim],
                [1, arch.hidden_dim],
            )
            .with_input(&ffn_id)
            .with_input(&res_attn_id);
            graph.add_layer(res_ffn).expect("res_ffn layer");

            prev_id = res_ffn_id;
        }

        // Final norm
        let final_norm = LayerDef::new(
            "final_norm",
            LayerType::Normalization,
            arch.num_layers,
            [1, arch.hidden_dim],
            [1, arch.hidden_dim],
        )
        .with_input(&prev_id);
        graph.add_layer(final_norm).expect("final_norm layer");

        // Output projection
        let output = LayerDef::new(
            "output",
            LayerType::OutputProjection,
            0,
            [arch.hidden_dim, arch.vocab_size],
            [1, arch.vocab_size],
        )
        .with_input("final_norm");
        graph.add_layer(output).expect("output layer");

        graph
    }
}

impl Default for ModelGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ArchError ───────────────────────────────────────────────────────────────

/// Errors produced by architecture operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchError {
    /// A layer with this ID already exists in the graph.
    DuplicateLayer(String),
    /// A layer references a dependency that does not exist.
    MissingDependency { layer: String, dependency: String },
    /// Architecture validation failed.
    ValidationError(String),
    /// Serialization or deserialization error.
    SerializationError(String),
    /// Architecture detection failed.
    DetectionError(String),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLayer(id) => write!(f, "duplicate layer id: {id}"),
            Self::MissingDependency { layer, dependency } => {
                write!(f, "layer '{layer}' references missing dependency '{dependency}'")
            }
            Self::ValidationError(msg) => write!(f, "validation error: {msg}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
            Self::DetectionError(msg) => write!(f, "detection error: {msg}"),
        }
    }
}

impl std::error::Error for ArchError {}

// ─── ArchDetector ────────────────────────────────────────────────────────────

/// Detects architecture family and configuration from model weight metadata.
///
/// CPU reference implementation that inspects tensor names, shapes, and
/// metadata keys to determine which architecture family a model belongs to
/// and extract its structural parameters.
#[derive(Debug, Clone)]
pub struct ArchDetector {
    /// Known tensor-name patterns per family.
    patterns: HashMap<ArchitectureFamily, Vec<String>>,
}

impl ArchDetector {
    /// Creates a detector with default heuristic patterns.
    pub fn new() -> Self {
        let mut patterns: HashMap<ArchitectureFamily, Vec<String>> = HashMap::new();
        patterns.insert(
            ArchitectureFamily::LLaMA,
            vec![
                "model.layers".into(),
                "model.norm".into(),
                "lm_head".into(),
                "rotary_emb".into(),
            ],
        );
        patterns.insert(
            ArchitectureFamily::GPT,
            vec![
                "transformer.h".into(),
                "transformer.ln_f".into(),
                "wte".into(),
                "wpe".into(),
            ],
        );
        patterns.insert(
            ArchitectureFamily::BitNet,
            vec![
                "bitlinear".into(),
                "ternary".into(),
                "model.layers".into(),
                "bitnet".into(),
            ],
        );
        patterns.insert(
            ArchitectureFamily::Mistral,
            vec![
                "model.layers".into(),
                "sliding_window".into(),
                "mistral".into(),
            ],
        );
        patterns.insert(
            ArchitectureFamily::Qwen,
            vec!["transformer.h".into(), "qwen".into(), "rotary".into()],
        );
        Self { patterns }
    }

    /// Detects the architecture family from a set of tensor names and metadata.
    ///
    /// Scores each family by how many patterns match and returns the best.
    pub fn detect(
        &self,
        tensor_names: &[String],
        metadata: &HashMap<String, String>,
    ) -> Result<ArchitectureFamily, ArchError> {
        // Check metadata for explicit architecture key
        if let Some(arch) = metadata.get("general.architecture") {
            return self.parse_family_name(arch);
        }

        let all_text: Vec<String> = tensor_names
            .iter()
            .chain(metadata.keys())
            .chain(metadata.values())
            .map(|s| s.to_lowercase())
            .collect();

        let mut scores: HashMap<ArchitectureFamily, usize> = HashMap::new();
        for (family, pats) in &self.patterns {
            let score =
                pats.iter().filter(|p| all_text.iter().any(|t| t.contains(p.as_str()))).count();
            if score > 0 {
                scores.insert(*family, score);
            }
        }

        scores
            .into_iter()
            .max_by_key(|&(_, score)| score)
            .map(|(family, _)| family)
            .ok_or_else(|| ArchError::DetectionError("no matching architecture family".into()))
    }

    /// Parses a family name string into an `ArchitectureFamily`.
    pub fn parse_family_name(&self, name: &str) -> Result<ArchitectureFamily, ArchError> {
        match name.to_lowercase().as_str() {
            "llama" => Ok(ArchitectureFamily::LLaMA),
            "gpt" | "gpt2" | "gpt-2" => Ok(ArchitectureFamily::GPT),
            "bitnet" => Ok(ArchitectureFamily::BitNet),
            "mistral" => Ok(ArchitectureFamily::Mistral),
            "qwen" | "qwen2" => Ok(ArchitectureFamily::Qwen),
            _ => Err(ArchError::DetectionError(format!("unknown architecture: {name}"))),
        }
    }

    /// Extracts a `TransformerConfig` heuristic based on detected family.
    pub fn default_transformer_config(
        &self,
        family: ArchitectureFamily,
    ) -> TransformerConfig {
        match family {
            ArchitectureFamily::LLaMA => TransformerConfig::llama_default(),
            ArchitectureFamily::GPT => TransformerConfig::gpt_default(),
            ArchitectureFamily::BitNet => TransformerConfig::bitnet_default(),
            ArchitectureFamily::Mistral => TransformerConfig::mistral_default(),
            ArchitectureFamily::Qwen => TransformerConfig::qwen_default(),
        }
    }
}

impl Default for ArchDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ArchValidator ───────────────────────────────────────────────────────────

/// Validates architecture configuration consistency.
///
/// Checks dimensional agreements, head count divisibility, and
/// family-specific constraints. Reports all validation issues found.
#[derive(Debug, Clone)]
pub struct ArchValidator {
    /// Maximum allowed layer count (sanity bound).
    pub max_layers: usize,
    /// Maximum allowed hidden dimension (sanity bound).
    pub max_hidden_dim: usize,
    /// Whether to enforce strict family-specific rules.
    pub strict: bool,
}

impl ArchValidator {
    /// Creates a validator with default bounds.
    pub fn new() -> Self {
        Self { max_layers: 256, max_hidden_dim: 65536, strict: false }
    }

    /// Enables strict validation mode.
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validates an `ArchConfig` and returns a list of issues.
    pub fn validate(&self, config: &ArchConfig) -> Vec<String> {
        let mut issues = Vec::new();

        if config.num_layers == 0 {
            issues.push("num_layers must be > 0".into());
        }
        if config.num_layers > self.max_layers {
            issues.push(format!("num_layers {} exceeds max {}", config.num_layers, self.max_layers));
        }
        if config.num_heads == 0 {
            issues.push("num_heads must be > 0".into());
        }
        if config.num_kv_heads == 0 {
            issues.push("num_kv_heads must be > 0".into());
        }
        if config.num_kv_heads > config.num_heads {
            issues.push(format!(
                "num_kv_heads ({}) > num_heads ({})",
                config.num_kv_heads, config.num_heads
            ));
        }
        if config.num_heads > 0 && config.num_kv_heads > 0 && config.num_heads % config.num_kv_heads != 0
        {
            issues.push(format!(
                "num_heads ({}) not divisible by num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            ));
        }
        if config.hidden_dim == 0 {
            issues.push("hidden_dim must be > 0".into());
        }
        if config.hidden_dim > self.max_hidden_dim {
            issues.push(format!(
                "hidden_dim {} exceeds max {}",
                config.hidden_dim, self.max_hidden_dim
            ));
        }
        if config.num_heads > 0 && config.hidden_dim % config.num_heads != 0 {
            issues.push(format!(
                "hidden_dim ({}) not divisible by num_heads ({})",
                config.hidden_dim, config.num_heads
            ));
        }
        if config.intermediate_dim == 0 {
            issues.push("intermediate_dim must be > 0".into());
        }
        if config.vocab_size == 0 {
            issues.push("vocab_size must be > 0".into());
        }
        if config.max_seq_len == 0 {
            issues.push("max_seq_len must be > 0".into());
        }

        if self.strict {
            self.validate_family_constraints(config, &mut issues);
        }

        issues
    }

    /// Validates family-specific constraints.
    fn validate_family_constraints(&self, config: &ArchConfig, issues: &mut Vec<String>) {
        match config.family {
            ArchitectureFamily::BitNet => {
                if config.hidden_dim % 256 != 0 {
                    issues.push(format!(
                        "BitNet hidden_dim ({}) should be divisible by 256 for QK256",
                        config.hidden_dim
                    ));
                }
            }
            ArchitectureFamily::Mistral => {
                if config.num_kv_heads >= config.num_heads && config.num_heads > 1 {
                    issues.push(
                        "Mistral typically uses GQA (num_kv_heads < num_heads)".into(),
                    );
                }
            }
            _ => {}
        }
    }

    /// Validates and returns `Ok(())` or the first error.
    pub fn validate_strict(&self, config: &ArchConfig) -> Result<(), ArchError> {
        let issues = self.validate(config);
        if issues.is_empty() {
            Ok(())
        } else {
            Err(ArchError::ValidationError(issues.join("; ")))
        }
    }
}

impl Default for ArchValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── OptimizationHint ────────────────────────────────────────────────────────

/// A suggested architecture optimization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Category of optimization.
    pub category: OptimizationCategory,
    /// Human-readable description.
    pub description: String,
    /// Affected layer IDs (empty if global).
    pub affected_layers: Vec<String>,
    /// Estimated speedup factor (1.0 = no change).
    pub estimated_speedup: u32,
}

/// Category of architecture optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// Fuse adjacent layers into a single kernel.
    LayerFusion,
    /// Candidate for weight pruning.
    Pruning,
    /// Use more efficient quantization.
    Quantization,
    /// Memory layout optimization.
    MemoryLayout,
    /// Attention-specific optimization.
    AttentionOptimization,
}

impl fmt::Display for OptimizationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerFusion => write!(f, "LayerFusion"),
            Self::Pruning => write!(f, "Pruning"),
            Self::Quantization => write!(f, "Quantization"),
            Self::MemoryLayout => write!(f, "MemoryLayout"),
            Self::AttentionOptimization => write!(f, "AttentionOptimization"),
        }
    }
}

// ─── ArchOptimizer ───────────────────────────────────────────────────────────

/// Suggests architecture optimizations based on the model graph and config.
///
/// Analyzes layer adjacency, quantization opportunities, and attention
/// patterns to produce actionable optimization hints. CPU reference
/// implementation — the GPU HAL may apply these automatically.
#[derive(Debug, Clone)]
pub struct ArchOptimizer {
    /// Whether to suggest layer fusion opportunities.
    pub enable_fusion: bool,
    /// Whether to suggest pruning candidates.
    pub enable_pruning: bool,
    /// Whether to suggest quantization changes.
    pub enable_quantization: bool,
    /// Whether to suggest memory layout changes.
    pub enable_memory_layout: bool,
    /// Minimum layer parameter count to consider for pruning.
    pub pruning_min_params: u64,
}

impl ArchOptimizer {
    /// Creates an optimizer with all analyses enabled.
    pub fn new() -> Self {
        Self {
            enable_fusion: true,
            enable_pruning: true,
            enable_quantization: true,
            enable_memory_layout: true,
            pruning_min_params: 1_000_000,
        }
    }

    /// Disables layer fusion suggestions.
    pub fn without_fusion(mut self) -> Self {
        self.enable_fusion = false;
        self
    }

    /// Disables pruning suggestions.
    pub fn without_pruning(mut self) -> Self {
        self.enable_pruning = false;
        self
    }

    /// Analyze the model graph and produce optimization hints.
    pub fn analyze(
        &self,
        graph: &ModelGraph,
        arch: &ArchConfig,
        transformer: &TransformerConfig,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        if self.enable_fusion {
            self.find_fusion_opportunities(graph, &mut hints);
        }
        if self.enable_pruning {
            self.find_pruning_candidates(graph, &mut hints);
        }
        if self.enable_quantization {
            self.find_quantization_hints(graph, arch, &mut hints);
        }
        if self.enable_memory_layout {
            self.find_memory_hints(arch, transformer, &mut hints);
        }

        hints
    }

    fn find_fusion_opportunities(&self, graph: &ModelGraph, hints: &mut Vec<OptimizationHint>) {
        // Detect norm + attention pairs eligible for fusion.
        for (i, layer) in graph.layers.iter().enumerate() {
            if layer.layer_type == LayerType::Normalization {
                if let Some(next) = graph.layers.get(i + 1) {
                    if next.layer_type == LayerType::SelfAttention
                        || next.layer_type == LayerType::FeedForward
                        || next.layer_type == LayerType::BitLinear
                    {
                        hints.push(OptimizationHint {
                            category: OptimizationCategory::LayerFusion,
                            description: format!(
                                "Fuse {} + {} at index {i}",
                                layer.layer_type, next.layer_type
                            ),
                            affected_layers: vec![layer.id.clone(), next.id.clone()],
                            estimated_speedup: 110, // 1.10x encoded as integer
                        });
                    }
                }
            }
        }
    }

    fn find_pruning_candidates(&self, graph: &ModelGraph, hints: &mut Vec<OptimizationHint>) {
        for layer in &graph.layers {
            if layer.param_count() >= self.pruning_min_params && !layer.is_quantized() {
                hints.push(OptimizationHint {
                    category: OptimizationCategory::Pruning,
                    description: format!(
                        "Layer '{}' has {}M unquantized params — pruning candidate",
                        layer.id,
                        layer.param_count() / 1_000_000
                    ),
                    affected_layers: vec![layer.id.clone()],
                    estimated_speedup: 105,
                });
            }
        }
    }

    fn find_quantization_hints(
        &self,
        graph: &ModelGraph,
        arch: &ArchConfig,
        hints: &mut Vec<OptimizationHint>,
    ) {
        let unquantized_attn: Vec<_> = graph
            .layers
            .iter()
            .filter(|l| l.layer_type == LayerType::SelfAttention && !l.is_quantized())
            .collect();

        if !unquantized_attn.is_empty() && arch.family == ArchitectureFamily::BitNet {
            hints.push(OptimizationHint {
                category: OptimizationCategory::Quantization,
                description: format!(
                    "{} attention layers could use BitLinear quantization",
                    unquantized_attn.len()
                ),
                affected_layers: unquantized_attn.iter().map(|l| l.id.clone()).collect(),
                estimated_speedup: 200,
            });
        }
    }

    fn find_memory_hints(
        &self,
        arch: &ArchConfig,
        transformer: &TransformerConfig,
        hints: &mut Vec<OptimizationHint>,
    ) {
        // Suggest KV cache optimization for GQA models.
        match transformer.attention_type {
            AttentionType::GroupedQuery | AttentionType::MultiQuery => {
                let kv_ratio = if arch.num_kv_heads > 0 {
                    arch.num_heads / arch.num_kv_heads
                } else {
                    1
                };
                hints.push(OptimizationHint {
                    category: OptimizationCategory::MemoryLayout,
                    description: format!(
                        "GQA with {kv_ratio}:1 ratio — use shared KV cache layout"
                    ),
                    affected_layers: vec![],
                    estimated_speedup: 100 + (kv_ratio as u32 * 5),
                });
            }
            AttentionType::SlidingWindow { window_size } => {
                hints.push(OptimizationHint {
                    category: OptimizationCategory::AttentionOptimization,
                    description: format!(
                        "Sliding window ({window_size}) — use ring-buffer KV cache"
                    ),
                    affected_layers: vec![],
                    estimated_speedup: 120,
                });
            }
            _ => {}
        }
    }
}

impl Default for ArchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ArchSerializer ──────────────────────────────────────────────────────────

/// Serializes and deserializes architecture configurations.
///
/// Supports JSON format for portability. YAML support is represented
/// by a format enum but delegates to JSON internally (serde-based).
#[derive(Debug, Clone)]
pub struct ArchSerializer;

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format.
    Json,
    /// Pretty-printed JSON.
    JsonPretty,
}

impl ArchSerializer {
    /// Creates a new serializer.
    pub fn new() -> Self {
        Self
    }

    /// Serializes an `ArchConfig` to a string.
    pub fn serialize_config(
        &self,
        config: &ArchConfig,
        format: SerializationFormat,
    ) -> Result<String, ArchError> {
        match format {
            SerializationFormat::Json => serde_json::to_string(config)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
            SerializationFormat::JsonPretty => serde_json::to_string_pretty(config)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
        }
    }

    /// Deserializes an `ArchConfig` from a JSON string.
    pub fn deserialize_config(&self, data: &str) -> Result<ArchConfig, ArchError> {
        serde_json::from_str(data).map_err(|e| ArchError::SerializationError(e.to_string()))
    }

    /// Serializes a `TransformerConfig` to a string.
    pub fn serialize_transformer(
        &self,
        config: &TransformerConfig,
        format: SerializationFormat,
    ) -> Result<String, ArchError> {
        match format {
            SerializationFormat::Json => serde_json::to_string(config)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
            SerializationFormat::JsonPretty => serde_json::to_string_pretty(config)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
        }
    }

    /// Deserializes a `TransformerConfig` from a JSON string.
    pub fn deserialize_transformer(&self, data: &str) -> Result<TransformerConfig, ArchError> {
        serde_json::from_str(data).map_err(|e| ArchError::SerializationError(e.to_string()))
    }

    /// Serializes a `ModelGraph` to a string.
    pub fn serialize_graph(
        &self,
        graph: &ModelGraph,
        format: SerializationFormat,
    ) -> Result<String, ArchError> {
        match format {
            SerializationFormat::Json => serde_json::to_string(graph)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
            SerializationFormat::JsonPretty => serde_json::to_string_pretty(graph)
                .map_err(|e| ArchError::SerializationError(e.to_string())),
        }
    }

    /// Deserializes a `ModelGraph` from a JSON string.
    pub fn deserialize_graph(&self, data: &str) -> Result<ModelGraph, ArchError> {
        serde_json::from_str(data).map_err(|e| ArchError::SerializationError(e.to_string()))
    }
}

impl Default for ArchSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ModelArchitectureEngine ─────────────────────────────────────────────────

/// Unified architecture management engine.
///
/// Combines detection, validation, graph construction, optimization analysis,
/// and serialization into a single entry point. This is the primary interface
/// for the GPU HAL to inspect and plan around model architecture.
#[derive(Debug, Clone)]
pub struct ModelArchitectureEngine {
    /// Architecture detector.
    pub detector: ArchDetector,
    /// Architecture validator.
    pub validator: ArchValidator,
    /// Architecture optimizer.
    pub optimizer: ArchOptimizer,
    /// Serializer.
    pub serializer: ArchSerializer,
}

impl ModelArchitectureEngine {
    /// Creates an engine with default components.
    pub fn new() -> Self {
        Self {
            detector: ArchDetector::new(),
            validator: ArchValidator::new(),
            optimizer: ArchOptimizer::new(),
            serializer: ArchSerializer::new(),
        }
    }

    /// Creates an engine with strict validation.
    pub fn strict() -> Self {
        Self {
            detector: ArchDetector::new(),
            validator: ArchValidator::new().strict(),
            optimizer: ArchOptimizer::new(),
            serializer: ArchSerializer::new(),
        }
    }

    /// Full pipeline: detect family, build default config, validate, build graph,
    /// and analyze optimizations.
    pub fn analyze_model(
        &self,
        tensor_names: &[String],
        metadata: &HashMap<String, String>,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Result<ArchAnalysis, ArchError> {
        let family = self.detector.detect(tensor_names, metadata)?;
        let transformer = self.detector.default_transformer_config(family);

        let model_name = metadata
            .get("general.name")
            .cloned()
            .unwrap_or_else(|| format!("{family}-model"));

        let config = ArchConfig::new(
            model_name,
            family,
            num_layers,
            num_heads,
            num_kv_heads,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            max_seq_len,
        );

        let issues = self.validator.validate(&config);
        if !issues.is_empty() {
            return Err(ArchError::ValidationError(issues.join("; ")));
        }

        let graph = ModelGraph::from_config(&config, &transformer);
        let hints = self.optimizer.analyze(&graph, &config, &transformer);

        Ok(ArchAnalysis { family, config, transformer, graph, hints })
    }

    /// Validates an existing config.
    pub fn validate(&self, config: &ArchConfig) -> Result<(), ArchError> {
        self.validator.validate_strict(config)
    }

    /// Builds a graph from an already-validated config.
    pub fn build_graph(
        &self,
        config: &ArchConfig,
        transformer: &TransformerConfig,
    ) -> ModelGraph {
        ModelGraph::from_config(config, transformer)
    }

    /// Serializes a full analysis to JSON.
    pub fn serialize_analysis(
        &self,
        analysis: &ArchAnalysis,
        pretty: bool,
    ) -> Result<String, ArchError> {
        let format = if pretty {
            SerializationFormat::JsonPretty
        } else {
            SerializationFormat::Json
        };
        self.serializer.serialize_config(&analysis.config, format)
    }
}

impl Default for ModelArchitectureEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a full architecture analysis pipeline.
#[derive(Debug, Clone)]
pub struct ArchAnalysis {
    /// Detected architecture family.
    pub family: ArchitectureFamily,
    /// Architecture configuration.
    pub config: ArchConfig,
    /// Transformer configuration.
    pub transformer: TransformerConfig,
    /// Model graph.
    pub graph: ModelGraph,
    /// Optimization hints.
    pub hints: Vec<OptimizationHint>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ─────────────────────────────────────────────────────────

    fn sample_bitnet_config() -> ArchConfig {
        ArchConfig::bitnet_2b()
    }

    fn sample_llama_config() -> ArchConfig {
        ArchConfig::llama_7b()
    }

    fn sample_gpt_config() -> ArchConfig {
        ArchConfig::gpt2()
    }

    fn small_config(family: ArchitectureFamily) -> ArchConfig {
        ArchConfig::new("test-model", family, 2, 4, 4, 128, 256, 1000, 512)
    }

    fn small_transformer() -> TransformerConfig {
        TransformerConfig::llama_default()
    }

    fn llama_tensor_names() -> Vec<String> {
        vec![
            "model.layers.0.self_attn.q_proj.weight".into(),
            "model.layers.0.mlp.gate_proj.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ]
    }

    fn gpt_tensor_names() -> Vec<String> {
        vec![
            "transformer.h.0.attn.c_attn.weight".into(),
            "transformer.h.0.mlp.c_fc.weight".into(),
            "transformer.ln_f.weight".into(),
            "wte.weight".into(),
            "wpe.weight".into(),
        ]
    }

    fn bitnet_tensor_names() -> Vec<String> {
        vec![
            "model.layers.0.bitlinear.weight".into(),
            "model.layers.0.ternary_proj.weight".into(),
            "bitnet.norm.weight".into(),
        ]
    }

    fn empty_metadata() -> HashMap<String, String> {
        HashMap::new()
    }

    // ─── ArchitectureFamily ──────────────────────────────────────────────

    #[test]
    fn test_family_display() {
        assert_eq!(ArchitectureFamily::LLaMA.to_string(), "LLaMA");
        assert_eq!(ArchitectureFamily::GPT.to_string(), "GPT");
        assert_eq!(ArchitectureFamily::BitNet.to_string(), "BitNet");
        assert_eq!(ArchitectureFamily::Mistral.to_string(), "Mistral");
        assert_eq!(ArchitectureFamily::Qwen.to_string(), "Qwen");
    }

    #[test]
    fn test_family_rms_norm() {
        assert!(ArchitectureFamily::LLaMA.uses_rms_norm());
        assert!(!ArchitectureFamily::GPT.uses_rms_norm());
        assert!(ArchitectureFamily::BitNet.uses_rms_norm());
        assert!(ArchitectureFamily::Mistral.uses_rms_norm());
        assert!(ArchitectureFamily::Qwen.uses_rms_norm());
    }

    #[test]
    fn test_family_rotary() {
        assert!(ArchitectureFamily::LLaMA.uses_rotary_embeddings());
        assert!(!ArchitectureFamily::GPT.uses_rotary_embeddings());
        assert!(ArchitectureFamily::BitNet.uses_rotary_embeddings());
    }

    #[test]
    fn test_family_gqa() {
        assert!(ArchitectureFamily::LLaMA.uses_grouped_query_attention());
        assert!(!ArchitectureFamily::GPT.uses_grouped_query_attention());
        assert!(!ArchitectureFamily::BitNet.uses_grouped_query_attention());
        assert!(ArchitectureFamily::Mistral.uses_grouped_query_attention());
        assert!(ArchitectureFamily::Qwen.uses_grouped_query_attention());
    }

    #[test]
    fn test_family_ffn_multiplier() {
        assert!((ArchitectureFamily::GPT.default_ffn_multiplier() - 4.0).abs() < f64::EPSILON);
        assert!(ArchitectureFamily::LLaMA.default_ffn_multiplier() < 4.0);
        assert!(ArchitectureFamily::BitNet.default_ffn_multiplier() < 4.0);
    }

    #[test]
    fn test_family_all() {
        let all = ArchitectureFamily::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&ArchitectureFamily::LLaMA));
        assert!(all.contains(&ArchitectureFamily::GPT));
        assert!(all.contains(&ArchitectureFamily::BitNet));
        assert!(all.contains(&ArchitectureFamily::Mistral));
        assert!(all.contains(&ArchitectureFamily::Qwen));
    }

    #[test]
    fn test_family_serialize_roundtrip() {
        for family in ArchitectureFamily::all() {
            let json = serde_json::to_string(family).unwrap();
            let back: ArchitectureFamily = serde_json::from_str(&json).unwrap();
            assert_eq!(*family, back);
        }
    }

    #[test]
    fn test_family_eq_and_hash() {
        let mut set = std::collections::HashSet::new();
        set.insert(ArchitectureFamily::LLaMA);
        set.insert(ArchitectureFamily::LLaMA);
        assert_eq!(set.len(), 1);
    }

    // ─── QuantizationInfo ────────────────────────────────────────────────

    #[test]
    fn test_quantization_bitnet_ternary() {
        let q = QuantizationInfo::bitnet_ternary();
        assert_eq!(q.scheme, "I2_S");
        assert_eq!(q.bits, 2);
        assert_eq!(q.block_size, 32);
        assert!(q.symmetric);
    }

    #[test]
    fn test_quantization_qk256() {
        let q = QuantizationInfo::qk256();
        assert_eq!(q.scheme, "QK256");
        assert_eq!(q.bits, 2);
        assert_eq!(q.block_size, 256);
    }

    #[test]
    fn test_quantization_fp16() {
        let q = QuantizationInfo::fp16();
        assert_eq!(q.bits, 16);
        assert_eq!(q.block_size, 0);
        assert!(!q.symmetric);
    }

    #[test]
    fn test_quantization_custom() {
        let q = QuantizationInfo::new("INT8", 8, 128, false);
        assert_eq!(q.scheme, "INT8");
        assert_eq!(q.bits, 8);
        assert_eq!(q.block_size, 128);
        assert!(!q.symmetric);
    }

    #[test]
    fn test_quantization_serialize_roundtrip() {
        let q = QuantizationInfo::qk256();
        let json = serde_json::to_string(&q).unwrap();
        let back: QuantizationInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(q, back);
    }

    // ─── ArchConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_arch_config_bitnet_2b() {
        let c = ArchConfig::bitnet_2b();
        assert_eq!(c.family, ArchitectureFamily::BitNet);
        assert_eq!(c.num_layers, 24);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.hidden_dim, 2560);
        assert_eq!(c.head_dim, 80);
    }

    #[test]
    fn test_arch_config_llama_7b() {
        let c = ArchConfig::llama_7b();
        assert_eq!(c.family, ArchitectureFamily::LLaMA);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.hidden_dim, 4096);
        assert_eq!(c.head_dim, 128);
    }

    #[test]
    fn test_arch_config_gpt2() {
        let c = ArchConfig::gpt2();
        assert_eq!(c.family, ArchitectureFamily::GPT);
        assert_eq!(c.num_layers, 12);
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.vocab_size, 50257);
    }

    #[test]
    fn test_arch_config_head_dim_computation() {
        let c = ArchConfig::new("test", ArchitectureFamily::LLaMA, 4, 8, 8, 512, 1024, 1000, 512);
        assert_eq!(c.head_dim, 64);
    }

    #[test]
    fn test_arch_config_head_dim_zero_heads() {
        let c = ArchConfig::new("test", ArchitectureFamily::LLaMA, 4, 0, 0, 512, 1024, 1000, 512);
        assert_eq!(c.head_dim, 0);
    }

    #[test]
    fn test_arch_config_metadata() {
        let c = sample_bitnet_config()
            .with_metadata("source", "huggingface")
            .with_metadata("quant", "I2_S");
        assert_eq!(c.metadata.get("source").unwrap(), "huggingface");
        assert_eq!(c.metadata.get("quant").unwrap(), "I2_S");
    }

    #[test]
    fn test_arch_config_param_count_nonzero() {
        let c = sample_bitnet_config();
        assert!(c.estimated_param_count() > 0);
    }

    #[test]
    fn test_arch_config_memory_bytes() {
        let c = sample_bitnet_config();
        let mem_2bit = c.estimated_memory_bytes(2);
        let mem_16bit = c.estimated_memory_bytes(16);
        assert_eq!(mem_16bit, mem_2bit * 8);
    }

    #[test]
    fn test_arch_config_serialize_roundtrip() {
        let c = sample_bitnet_config().with_metadata("key", "val");
        let json = serde_json::to_string(&c).unwrap();
        let back: ArchConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, c.model_name);
        assert_eq!(back.family, c.family);
        assert_eq!(back.num_layers, c.num_layers);
        assert_eq!(back.hidden_dim, c.hidden_dim);
        assert_eq!(back.metadata.get("key").unwrap(), "val");
    }

    #[test]
    fn test_arch_config_clone() {
        let c = sample_llama_config();
        let c2 = c.clone();
        assert_eq!(c.model_name, c2.model_name);
        assert_eq!(c.num_layers, c2.num_layers);
    }

    // ─── TransformerConfig ───────────────────────────────────────────────

    #[test]
    fn test_transformer_llama_default() {
        let t = TransformerConfig::llama_default();
        assert!(t.pre_norm);
        assert_eq!(t.norm_type, NormType::RMSNorm);
        assert_eq!(t.attention_type, AttentionType::MultiHead);
        assert_eq!(t.activation, "silu");
        assert!(t.gated_ffn);
        assert!((t.rope_base - 10000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transformer_gpt_default() {
        let t = TransformerConfig::gpt_default();
        assert!(t.pre_norm);
        assert_eq!(t.norm_type, NormType::LayerNorm);
        assert_eq!(t.activation, "gelu");
        assert!(t.tie_embeddings);
        assert!(!t.gated_ffn);
    }

    #[test]
    fn test_transformer_bitnet_default() {
        let t = TransformerConfig::bitnet_default();
        assert_eq!(t.norm_type, NormType::RMSNorm);
        assert!(t.gated_ffn);
        assert!((t.norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_transformer_mistral_default() {
        let t = TransformerConfig::mistral_default();
        assert!(matches!(t.attention_type, AttentionType::SlidingWindow { window_size: 4096 }));
    }

    #[test]
    fn test_transformer_qwen_default() {
        let t = TransformerConfig::qwen_default();
        assert!((t.rope_base - 1_000_000.0).abs() < f64::EPSILON);
        assert!((t.norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_transformer_builder_methods() {
        let t = TransformerConfig::new(false, NormType::LayerNorm, AttentionType::MultiHead, "relu")
            .with_gated_ffn(true)
            .with_tie_embeddings(true)
            .with_rope(5000.0, 2.0)
            .with_norm_eps(1e-4);
        assert!(!t.pre_norm);
        assert!(t.gated_ffn);
        assert!(t.tie_embeddings);
        assert!((t.rope_base - 5000.0).abs() < f64::EPSILON);
        assert!((t.rope_scaling - 2.0).abs() < f64::EPSILON);
        assert!((t.norm_eps - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_transformer_serialize_roundtrip() {
        let t = TransformerConfig::llama_default();
        let json = serde_json::to_string(&t).unwrap();
        let back: TransformerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.norm_type, t.norm_type);
        assert_eq!(back.activation, t.activation);
    }

    // ─── LayerDef ────────────────────────────────────────────────────────

    #[test]
    fn test_layer_def_basic() {
        let l = LayerDef::new("attn_0", LayerType::SelfAttention, 0, [128, 128], [128, 128]);
        assert_eq!(l.id, "attn_0");
        assert_eq!(l.layer_type, LayerType::SelfAttention);
        assert_eq!(l.index, 0);
        assert_eq!(l.param_count(), 128 * 128);
        assert!(!l.is_quantized());
    }

    #[test]
    fn test_layer_def_with_quantization() {
        let l = LayerDef::new("ffn_0", LayerType::BitLinear, 0, [256, 512], [512, 256])
            .with_quantization(QuantizationInfo::bitnet_ternary());
        assert!(l.is_quantized());
        assert_eq!(l.quantization.unwrap().scheme, "I2_S");
    }

    #[test]
    fn test_layer_def_with_input() {
        let l = LayerDef::new("attn_0", LayerType::SelfAttention, 0, [128, 128], [128, 128])
            .with_input("norm_0")
            .with_input("embed");
        assert_eq!(l.inputs.len(), 2);
        assert_eq!(l.inputs[0], "norm_0");
        assert_eq!(l.inputs[1], "embed");
    }

    #[test]
    fn test_layer_def_param_count() {
        let l = LayerDef::new("embed", LayerType::Embedding, 0, [32000, 4096], [1, 4096]);
        assert_eq!(l.param_count(), 32000 * 4096);
    }

    #[test]
    fn test_layer_type_display() {
        assert_eq!(LayerType::Embedding.to_string(), "Embedding");
        assert_eq!(LayerType::SelfAttention.to_string(), "SelfAttention");
        assert_eq!(LayerType::FeedForward.to_string(), "FeedForward");
        assert_eq!(LayerType::BitLinear.to_string(), "BitLinear");
        assert_eq!(LayerType::Residual.to_string(), "Residual");
    }

    #[test]
    fn test_layer_def_serialize_roundtrip() {
        let l = LayerDef::new("test", LayerType::FeedForward, 1, [64, 128], [128, 64])
            .with_quantization(QuantizationInfo::fp16())
            .with_input("prev");
        let json = serde_json::to_string(&l).unwrap();
        let back: LayerDef = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "test");
        assert!(back.is_quantized());
        assert_eq!(back.inputs, vec!["prev"]);
    }

    // ─── ModelGraph ──────────────────────────────────────────────────────

    #[test]
    fn test_graph_empty() {
        let g = ModelGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.total_params(), 0);
    }

    #[test]
    fn test_graph_default() {
        let g = ModelGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn test_graph_add_layer() {
        let mut g = ModelGraph::new();
        let l = LayerDef::new("embed", LayerType::Embedding, 0, [1000, 128], [1, 128]);
        g.add_layer(l).unwrap();
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_graph_duplicate_layer() {
        let mut g = ModelGraph::new();
        let l1 = LayerDef::new("embed", LayerType::Embedding, 0, [1000, 128], [1, 128]);
        let l2 = LayerDef::new("embed", LayerType::Embedding, 0, [1000, 128], [1, 128]);
        g.add_layer(l1).unwrap();
        assert!(matches!(g.add_layer(l2), Err(ArchError::DuplicateLayer(_))));
    }

    #[test]
    fn test_graph_missing_dependency() {
        let mut g = ModelGraph::new();
        let l = LayerDef::new("attn_0", LayerType::SelfAttention, 0, [128, 128], [128, 128])
            .with_input("nonexistent");
        assert!(matches!(g.add_layer(l), Err(ArchError::MissingDependency { .. })));
    }

    #[test]
    fn test_graph_get_layer() {
        let mut g = ModelGraph::new();
        let l = LayerDef::new("embed", LayerType::Embedding, 0, [1000, 128], [1, 128]);
        g.add_layer(l).unwrap();
        assert!(g.get_layer("embed").is_some());
        assert!(g.get_layer("missing").is_none());
    }

    #[test]
    fn test_graph_total_params() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("a", LayerType::Embedding, 0, [100, 10], [1, 10])).unwrap();
        g.add_layer(
            LayerDef::new("b", LayerType::Normalization, 0, [50, 20], [50, 20]).with_input("a"),
        )
        .unwrap();
        assert_eq!(g.total_params(), 100 * 10 + 50 * 20);
    }

    #[test]
    fn test_graph_layers_of_type() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("e", LayerType::Embedding, 0, [100, 10], [1, 10])).unwrap();
        g.add_layer(
            LayerDef::new("n1", LayerType::Normalization, 0, [1, 10], [1, 10]).with_input("e"),
        )
        .unwrap();
        g.add_layer(
            LayerDef::new("n2", LayerType::Normalization, 1, [1, 10], [1, 10]).with_input("n1"),
        )
        .unwrap();
        assert_eq!(g.layers_of_type(LayerType::Normalization).len(), 2);
        assert_eq!(g.layers_of_type(LayerType::Embedding).len(), 1);
        assert_eq!(g.layers_of_type(LayerType::FeedForward).len(), 0);
    }

    #[test]
    fn test_graph_dependents() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("e", LayerType::Embedding, 0, [100, 10], [1, 10])).unwrap();
        g.add_layer(
            LayerDef::new("a", LayerType::SelfAttention, 0, [10, 10], [10, 10]).with_input("e"),
        )
        .unwrap();
        g.add_layer(
            LayerDef::new("b", LayerType::FeedForward, 0, [10, 20], [20, 10]).with_input("e"),
        )
        .unwrap();
        let deps = g.dependents("e");
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&"a"));
        assert!(deps.contains(&"b"));
    }

    #[test]
    fn test_graph_layer_depths() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("e", LayerType::Embedding, 0, [100, 10], [1, 10])).unwrap();
        g.add_layer(
            LayerDef::new("n", LayerType::Normalization, 0, [1, 10], [1, 10]).with_input("e"),
        )
        .unwrap();
        g.add_layer(
            LayerDef::new("a", LayerType::SelfAttention, 0, [10, 10], [10, 10]).with_input("n"),
        )
        .unwrap();
        let depths = g.layer_depths();
        assert_eq!(depths["e"], 0);
        assert_eq!(depths["n"], 1);
        assert_eq!(depths["a"], 2);
    }

    #[test]
    fn test_graph_validate_ok() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("e", LayerType::Embedding, 0, [100, 10], [1, 10])).unwrap();
        g.add_layer(
            LayerDef::new("n", LayerType::Normalization, 0, [1, 10], [1, 10]).with_input("e"),
        )
        .unwrap();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_from_config_bitnet() {
        let arch = small_config(ArchitectureFamily::BitNet);
        let t = TransformerConfig::bitnet_default();
        let g = ModelGraph::from_config(&arch, &t);
        assert!(!g.is_empty());
        // embed + 2 layers * 6 sublayers + final_norm + output = 1 + 12 + 2 = 15
        assert_eq!(g.len(), 15);
        assert!(g.get_layer("embed").is_some());
        assert!(g.get_layer("output").is_some());
        assert!(g.get_layer("final_norm").is_some());
        // BitNet should use BitLinear for FFN
        let ffn = g.get_layer("layer_0_ffn").unwrap();
        assert_eq!(ffn.layer_type, LayerType::BitLinear);
    }

    #[test]
    fn test_graph_from_config_llama() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        // LLaMA should use FeedForward, not BitLinear
        let ffn = g.get_layer("layer_0_ffn").unwrap();
        assert_eq!(ffn.layer_type, LayerType::FeedForward);
    }

    #[test]
    fn test_graph_from_config_residuals() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let residuals = g.layers_of_type(LayerType::Residual);
        // 2 layers * 2 residuals each = 4
        assert_eq!(residuals.len(), 4);
    }

    #[test]
    fn test_graph_from_config_dependencies() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let attn = g.get_layer("layer_0_attn").unwrap();
        assert_eq!(attn.inputs, vec!["layer_0_norm_attn"]);
        let res = g.get_layer("layer_0_res_attn").unwrap();
        assert_eq!(res.inputs.len(), 2);
    }

    #[test]
    fn test_graph_serialize_roundtrip() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let json = serde_json::to_string(&g).unwrap();
        let back: ModelGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), g.len());
    }

    // ─── ArchError ───────────────────────────────────────────────────────

    #[test]
    fn test_error_display_duplicate() {
        let e = ArchError::DuplicateLayer("embed".into());
        assert!(e.to_string().contains("duplicate"));
    }

    #[test]
    fn test_error_display_missing_dep() {
        let e = ArchError::MissingDependency {
            layer: "attn_0".into(),
            dependency: "norm_0".into(),
        };
        assert!(e.to_string().contains("norm_0"));
    }

    #[test]
    fn test_error_display_validation() {
        let e = ArchError::ValidationError("bad config".into());
        assert!(e.to_string().contains("validation"));
    }

    #[test]
    fn test_error_display_serialization() {
        let e = ArchError::SerializationError("parse failed".into());
        assert!(e.to_string().contains("serialization"));
    }

    #[test]
    fn test_error_display_detection() {
        let e = ArchError::DetectionError("no match".into());
        assert!(e.to_string().contains("detection"));
    }

    // ─── ArchDetector ────────────────────────────────────────────────────

    #[test]
    fn test_detector_llama() {
        let d = ArchDetector::new();
        let family = d.detect(&llama_tensor_names(), &empty_metadata()).unwrap();
        assert_eq!(family, ArchitectureFamily::LLaMA);
    }

    #[test]
    fn test_detector_gpt() {
        let d = ArchDetector::new();
        let family = d.detect(&gpt_tensor_names(), &empty_metadata()).unwrap();
        assert_eq!(family, ArchitectureFamily::GPT);
    }

    #[test]
    fn test_detector_bitnet() {
        let d = ArchDetector::new();
        let family = d.detect(&bitnet_tensor_names(), &empty_metadata()).unwrap();
        assert_eq!(family, ArchitectureFamily::BitNet);
    }

    #[test]
    fn test_detector_metadata_override() {
        let d = ArchDetector::new();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".into(), "llama".into());
        let family = d.detect(&gpt_tensor_names(), &meta).unwrap();
        assert_eq!(family, ArchitectureFamily::LLaMA);
    }

    #[test]
    fn test_detector_metadata_bitnet() {
        let d = ArchDetector::new();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".into(), "bitnet".into());
        let family = d.detect(&[], &meta).unwrap();
        assert_eq!(family, ArchitectureFamily::BitNet);
    }

    #[test]
    fn test_detector_empty_fails() {
        let d = ArchDetector::new();
        let result = d.detect(&[], &empty_metadata());
        assert!(matches!(result, Err(ArchError::DetectionError(_))));
    }

    #[test]
    fn test_detector_parse_family_names() {
        let d = ArchDetector::new();
        assert_eq!(d.parse_family_name("llama").unwrap(), ArchitectureFamily::LLaMA);
        assert_eq!(d.parse_family_name("GPT").unwrap(), ArchitectureFamily::GPT);
        assert_eq!(d.parse_family_name("gpt2").unwrap(), ArchitectureFamily::GPT);
        assert_eq!(d.parse_family_name("bitnet").unwrap(), ArchitectureFamily::BitNet);
        assert_eq!(d.parse_family_name("Mistral").unwrap(), ArchitectureFamily::Mistral);
        assert_eq!(d.parse_family_name("qwen").unwrap(), ArchitectureFamily::Qwen);
        assert_eq!(d.parse_family_name("qwen2").unwrap(), ArchitectureFamily::Qwen);
    }

    #[test]
    fn test_detector_parse_unknown_family() {
        let d = ArchDetector::new();
        assert!(d.parse_family_name("unknown_arch").is_err());
    }

    #[test]
    fn test_detector_default_transformer_configs() {
        let d = ArchDetector::new();
        for family in ArchitectureFamily::all() {
            let t = d.default_transformer_config(*family);
            // All should be pre-norm
            assert!(t.pre_norm);
        }
    }

    #[test]
    fn test_detector_default() {
        let d = ArchDetector::default();
        assert!(!d.patterns.is_empty());
    }

    #[test]
    fn test_detector_mistral_via_metadata_value() {
        let d = ArchDetector::new();
        let mut meta = HashMap::new();
        meta.insert("model_type".into(), "mistral".into());
        let names: Vec<String> = vec!["model.layers.0.weight".into()];
        let family = d.detect(&names, &meta).unwrap();
        assert_eq!(family, ArchitectureFamily::Mistral);
    }

    // ─── ArchValidator ───────────────────────────────────────────────────

    #[test]
    fn test_validator_valid_config() {
        let v = ArchValidator::new();
        let issues = v.validate(&sample_llama_config());
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_validator_valid_bitnet() {
        let v = ArchValidator::new();
        let issues = v.validate(&sample_bitnet_config());
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validator_valid_gpt() {
        let v = ArchValidator::new();
        let issues = v.validate(&sample_gpt_config());
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validator_zero_layers() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 0, 32, 32, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("num_layers")));
    }

    #[test]
    fn test_validator_zero_heads() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 0, 0, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("num_heads")));
    }

    #[test]
    fn test_validator_kv_heads_exceeds_heads() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 8, 16, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("num_kv_heads")));
    }

    #[test]
    fn test_validator_heads_not_divisible_by_kv() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 7, 3, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("divisible")));
    }

    #[test]
    fn test_validator_hidden_dim_not_divisible_by_heads() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 7, 7, 100, 200, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("hidden_dim") && i.contains("divisible")));
    }

    #[test]
    fn test_validator_exceeds_max_layers() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("huge", ArchitectureFamily::LLaMA, 999, 32, 32, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("exceeds max")));
    }

    #[test]
    fn test_validator_exceeds_max_hidden() {
        let v = ArchValidator::new();
        let c =
            ArchConfig::new("huge", ArchitectureFamily::LLaMA, 32, 32, 32, 100000, 200000, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("hidden_dim") && i.contains("exceeds")));
    }

    #[test]
    fn test_validator_zero_vocab() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 32, 32, 4096, 11008, 0, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("vocab_size")));
    }

    #[test]
    fn test_validator_zero_seq_len() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 32, 32, 4096, 11008, 32000, 0);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("max_seq_len")));
    }

    #[test]
    fn test_validator_zero_intermediate_dim() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 32, 32, 32, 4096, 0, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("intermediate_dim")));
    }

    #[test]
    fn test_validator_strict_bitnet_hidden_not_256() {
        let v = ArchValidator::new().strict();
        let c = ArchConfig::new("bad", ArchitectureFamily::BitNet, 24, 32, 32, 2500, 6912, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("256")));
    }

    #[test]
    fn test_validator_strict_bitnet_ok() {
        let v = ArchValidator::new().strict();
        let issues = v.validate(&sample_bitnet_config());
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn test_validator_strict_mistral_gqa() {
        let v = ArchValidator::new().strict();
        let c =
            ArchConfig::new("bad-mistral", ArchitectureFamily::Mistral, 32, 32, 32, 4096, 11008, 32000, 4096);
        let issues = v.validate(&c);
        assert!(issues.iter().any(|i| i.contains("GQA")));
    }

    #[test]
    fn test_validator_validate_strict_ok() {
        let v = ArchValidator::new();
        assert!(v.validate_strict(&sample_llama_config()).is_ok());
    }

    #[test]
    fn test_validator_validate_strict_err() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 0, 0, 0, 0, 0, 0, 0);
        assert!(v.validate_strict(&c).is_err());
    }

    #[test]
    fn test_validator_multiple_issues() {
        let v = ArchValidator::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 0, 0, 0, 0, 0, 0, 0);
        let issues = v.validate(&c);
        assert!(issues.len() >= 5); // many things wrong
    }

    #[test]
    fn test_validator_default() {
        let v = ArchValidator::default();
        assert_eq!(v.max_layers, 256);
        assert!(!v.strict);
    }

    // ─── ArchOptimizer ───────────────────────────────────────────────────

    #[test]
    fn test_optimizer_fusion_hints() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new();
        let hints = o.analyze(&g, &arch, &t);
        let fusion: Vec<_> = hints.iter().filter(|h| h.category == OptimizationCategory::LayerFusion).collect();
        assert!(!fusion.is_empty());
    }

    #[test]
    fn test_optimizer_pruning_hints() {
        let arch = sample_llama_config();
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new();
        let hints = o.analyze(&g, &arch, &t);
        let pruning: Vec<_> = hints.iter().filter(|h| h.category == OptimizationCategory::Pruning).collect();
        assert!(!pruning.is_empty());
    }

    #[test]
    fn test_optimizer_bitnet_quantization_hints() {
        let arch = small_config(ArchitectureFamily::BitNet);
        let t = TransformerConfig::bitnet_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new();
        let hints = o.analyze(&g, &arch, &t);
        let quant: Vec<_> =
            hints.iter().filter(|h| h.category == OptimizationCategory::Quantization).collect();
        assert!(!quant.is_empty());
    }

    #[test]
    fn test_optimizer_sliding_window_hint() {
        let arch = small_config(ArchitectureFamily::Mistral);
        let t = TransformerConfig::mistral_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new();
        let hints = o.analyze(&g, &arch, &t);
        let attn_opt: Vec<_> = hints
            .iter()
            .filter(|h| h.category == OptimizationCategory::AttentionOptimization)
            .collect();
        assert!(!attn_opt.is_empty());
        assert!(attn_opt[0].description.contains("ring-buffer"));
    }

    #[test]
    fn test_optimizer_gqa_memory_hint() {
        let arch = ArchConfig::new("gqa-test", ArchitectureFamily::LLaMA, 4, 32, 8, 4096, 11008, 32000, 4096);
        let t = TransformerConfig::new(true, NormType::RMSNorm, AttentionType::GroupedQuery, "silu");
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new();
        let hints = o.analyze(&g, &arch, &t);
        let mem: Vec<_> =
            hints.iter().filter(|h| h.category == OptimizationCategory::MemoryLayout).collect();
        assert!(!mem.is_empty());
        assert!(mem[0].description.contains("GQA"));
    }

    #[test]
    fn test_optimizer_without_fusion() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new().without_fusion();
        let hints = o.analyze(&g, &arch, &t);
        assert!(hints.iter().all(|h| h.category != OptimizationCategory::LayerFusion));
    }

    #[test]
    fn test_optimizer_without_pruning() {
        let arch = sample_llama_config();
        let t = TransformerConfig::llama_default();
        let g = ModelGraph::from_config(&arch, &t);
        let o = ArchOptimizer::new().without_pruning();
        let hints = o.analyze(&g, &arch, &t);
        assert!(hints.iter().all(|h| h.category != OptimizationCategory::Pruning));
    }

    #[test]
    fn test_optimizer_default() {
        let o = ArchOptimizer::default();
        assert!(o.enable_fusion);
        assert!(o.enable_pruning);
        assert!(o.enable_quantization);
        assert!(o.enable_memory_layout);
    }

    #[test]
    fn test_optimization_category_display() {
        assert_eq!(OptimizationCategory::LayerFusion.to_string(), "LayerFusion");
        assert_eq!(OptimizationCategory::Pruning.to_string(), "Pruning");
        assert_eq!(OptimizationCategory::Quantization.to_string(), "Quantization");
        assert_eq!(OptimizationCategory::MemoryLayout.to_string(), "MemoryLayout");
        assert_eq!(OptimizationCategory::AttentionOptimization.to_string(), "AttentionOptimization");
    }

    // ─── ArchSerializer ──────────────────────────────────────────────────

    #[test]
    fn test_serializer_config_json() {
        let s = ArchSerializer::new();
        let c = sample_bitnet_config();
        let json = s.serialize_config(&c, SerializationFormat::Json).unwrap();
        assert!(json.contains("bitnet-b1.58-2B-4T"));
        let back = s.deserialize_config(&json).unwrap();
        assert_eq!(back.model_name, c.model_name);
    }

    #[test]
    fn test_serializer_config_json_pretty() {
        let s = ArchSerializer::new();
        let c = sample_llama_config();
        let json = s.serialize_config(&c, SerializationFormat::JsonPretty).unwrap();
        assert!(json.contains('\n'));
        let back = s.deserialize_config(&json).unwrap();
        assert_eq!(back.num_layers, c.num_layers);
    }

    #[test]
    fn test_serializer_transformer_roundtrip() {
        let s = ArchSerializer::new();
        let t = TransformerConfig::llama_default();
        let json = s.serialize_transformer(&t, SerializationFormat::Json).unwrap();
        let back = s.deserialize_transformer(&json).unwrap();
        assert_eq!(back.norm_type, t.norm_type);
        assert_eq!(back.activation, t.activation);
    }

    #[test]
    fn test_serializer_graph_roundtrip() {
        let s = ArchSerializer::new();
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = small_transformer();
        let g = ModelGraph::from_config(&arch, &t);
        let json = s.serialize_graph(&g, SerializationFormat::Json).unwrap();
        let back = s.deserialize_graph(&json).unwrap();
        assert_eq!(back.len(), g.len());
    }

    #[test]
    fn test_serializer_invalid_json() {
        let s = ArchSerializer::new();
        assert!(s.deserialize_config("not json").is_err());
        assert!(s.deserialize_transformer("{bad}").is_err());
        assert!(s.deserialize_graph("[]").is_err());
    }

    #[test]
    fn test_serializer_default() {
        let _s = ArchSerializer::default();
    }

    // ─── ModelArchitectureEngine ─────────────────────────────────────────

    #[test]
    fn test_engine_analyze_llama() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(&llama_tensor_names(), &empty_metadata(), 32, 32, 32, 4096, 11008, 32000, 4096)
            .unwrap();
        assert_eq!(analysis.family, ArchitectureFamily::LLaMA);
        assert!(!analysis.graph.is_empty());
        assert!(analysis.config.num_layers == 32);
    }

    #[test]
    fn test_engine_analyze_gpt() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(&gpt_tensor_names(), &empty_metadata(), 12, 12, 12, 768, 3072, 50257, 1024)
            .unwrap();
        assert_eq!(analysis.family, ArchitectureFamily::GPT);
    }

    #[test]
    fn test_engine_analyze_bitnet() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(&bitnet_tensor_names(), &empty_metadata(), 24, 32, 32, 2560, 6912, 32000, 4096)
            .unwrap();
        assert_eq!(analysis.family, ArchitectureFamily::BitNet);
    }

    #[test]
    fn test_engine_analyze_with_model_name() {
        let engine = ModelArchitectureEngine::new();
        let mut meta = HashMap::new();
        meta.insert("general.name".into(), "my-custom-model".into());
        meta.insert("general.architecture".into(), "llama".into());
        let analysis = engine
            .analyze_model(&[], &meta, 4, 4, 4, 256, 512, 1000, 512)
            .unwrap();
        assert_eq!(analysis.config.model_name, "my-custom-model");
    }

    #[test]
    fn test_engine_analyze_invalid_config() {
        let engine = ModelArchitectureEngine::new();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".into(), "llama".into());
        let result = engine.analyze_model(&[], &meta, 0, 0, 0, 0, 0, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_analyze_undetectable() {
        let engine = ModelArchitectureEngine::new();
        let result = engine.analyze_model(&[], &empty_metadata(), 4, 4, 4, 256, 512, 1000, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_validate_ok() {
        let engine = ModelArchitectureEngine::new();
        assert!(engine.validate(&sample_llama_config()).is_ok());
    }

    #[test]
    fn test_engine_validate_err() {
        let engine = ModelArchitectureEngine::new();
        let c = ArchConfig::new("bad", ArchitectureFamily::LLaMA, 0, 0, 0, 0, 0, 0, 0);
        assert!(engine.validate(&c).is_err());
    }

    #[test]
    fn test_engine_build_graph() {
        let engine = ModelArchitectureEngine::new();
        let c = small_config(ArchitectureFamily::LLaMA);
        let t = small_transformer();
        let g = engine.build_graph(&c, &t);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_engine_serialize_analysis() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(&llama_tensor_names(), &empty_metadata(), 4, 4, 4, 256, 512, 1000, 512)
            .unwrap();
        let json = engine.serialize_analysis(&analysis, false).unwrap();
        assert!(json.contains("LLaMA"));
        let pretty = engine.serialize_analysis(&analysis, true).unwrap();
        assert!(pretty.contains('\n'));
    }

    #[test]
    fn test_engine_strict_mode() {
        let engine = ModelArchitectureEngine::strict();
        assert!(engine.validator.strict);
    }

    #[test]
    fn test_engine_default() {
        let engine = ModelArchitectureEngine::default();
        assert!(!engine.validator.strict);
    }

    #[test]
    fn test_engine_hints_present_in_analysis() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(
                &llama_tensor_names(),
                &empty_metadata(),
                32,
                32,
                32,
                4096,
                11008,
                32000,
                4096,
            )
            .unwrap();
        // A real LLaMA-7B should have fusion hints at minimum
        assert!(!analysis.hints.is_empty());
    }

    // ─── Cross-component integration ─────────────────────────────────────

    #[test]
    fn test_full_pipeline_bitnet() {
        let engine = ModelArchitectureEngine::new();
        let analysis = engine
            .analyze_model(&bitnet_tensor_names(), &empty_metadata(), 24, 32, 32, 2560, 6912, 32000, 4096)
            .unwrap();

        // Detection
        assert_eq!(analysis.family, ArchitectureFamily::BitNet);

        // Config
        assert_eq!(analysis.config.num_layers, 24);
        assert_eq!(analysis.config.hidden_dim, 2560);

        // Transformer
        assert_eq!(analysis.transformer.norm_type, NormType::RMSNorm);
        assert!(analysis.transformer.gated_ffn);

        // Graph structure
        assert!(analysis.graph.get_layer("embed").is_some());
        assert!(analysis.graph.get_layer("output").is_some());

        // BitLinear FFN
        let ffn = analysis.graph.get_layer("layer_0_ffn").unwrap();
        assert_eq!(ffn.layer_type, LayerType::BitLinear);

        // Serialization
        let s = ArchSerializer::new();
        let json = s.serialize_config(&analysis.config, SerializationFormat::Json).unwrap();
        let back = s.deserialize_config(&json).unwrap();
        assert_eq!(back.family, ArchitectureFamily::BitNet);
    }

    #[test]
    fn test_full_pipeline_all_families() {
        let engine = ModelArchitectureEngine::new();
        let families_and_tensors: Vec<(Vec<String>, &str)> = vec![
            (llama_tensor_names(), "llama"),
            (gpt_tensor_names(), "gpt"),
            (bitnet_tensor_names(), "bitnet"),
        ];

        for (tensors, _label) in &families_and_tensors {
            let result = engine.analyze_model(
                tensors,
                &empty_metadata(),
                4,
                4,
                4,
                256,
                512,
                1000,
                512,
            );
            assert!(result.is_ok(), "failed for tensors matching expected family");
        }
    }

    #[test]
    fn test_graph_depth_consistency() {
        let arch = small_config(ArchitectureFamily::LLaMA);
        let t = small_transformer();
        let g = ModelGraph::from_config(&arch, &t);
        let depths = g.layer_depths();
        // Embed should be at depth 0
        assert_eq!(depths["embed"], 0);
        // Output should be the deepest
        let max_depth = depths.values().max().unwrap();
        assert_eq!(depths["output"], *max_depth);
    }

    #[test]
    fn test_config_param_count_scaling() {
        let small = ArchConfig::new("small", ArchitectureFamily::LLaMA, 4, 4, 4, 256, 512, 1000, 512);
        let large = ArchConfig::new("large", ArchitectureFamily::LLaMA, 32, 32, 32, 4096, 11008, 32000, 4096);
        assert!(large.estimated_param_count() > small.estimated_param_count());
    }

    #[test]
    fn test_optimizer_no_hints_for_trivial_graph() {
        let mut g = ModelGraph::new();
        g.add_layer(LayerDef::new("e", LayerType::Embedding, 0, [100, 10], [1, 10]))
            .unwrap();
        let arch = small_config(ArchitectureFamily::GPT);
        let t = TransformerConfig::gpt_default();
        let o = ArchOptimizer::new().without_fusion().without_pruning();
        let hints = o.analyze(&g, &arch, &t);
        // GPT with MultiHead attention shouldn't trigger GQA or sliding window hints
        assert!(hints.iter().all(|h| h.category != OptimizationCategory::MemoryLayout));
        assert!(hints.iter().all(|h| h.category != OptimizationCategory::AttentionOptimization));
    }
}
