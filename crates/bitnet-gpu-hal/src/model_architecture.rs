//! Model architecture registry for GPU HAL.
//!
//! Provides types for describing, detecting, validating, and estimating
//! resource requirements of neural-network model architectures.

use std::collections::HashMap;
use std::fmt;

// ── Architecture type ────────────────────────────────────────────────────

/// High-level architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ArchitectureType {
    /// Standard decoder-only transformer (GPT-style).
    Transformer,
    /// BitNet 1-bit transformer variant.
    BitNetTransformer,
    /// Mixture-of-Experts transformer.
    MoE,
    /// RWKV (linear-attention RNN-transformer hybrid).
    RWKV,
    /// Mamba (selective state-space model).
    Mamba,
    /// Encoder-only transformer (BERT-style).
    Encoder,
    /// Encoder-decoder transformer (T5-style).
    EncoderDecoder,
}

impl fmt::Display for ArchitectureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transformer => write!(f, "Transformer"),
            Self::BitNetTransformer => write!(f, "BitNetTransformer"),
            Self::MoE => write!(f, "MoE"),
            Self::RWKV => write!(f, "RWKV"),
            Self::Mamba => write!(f, "Mamba"),
            Self::Encoder => write!(f, "Encoder"),
            Self::EncoderDecoder => write!(f, "EncoderDecoder"),
        }
    }
}

// ── Layer configuration ──────────────────────────────────────────────────

/// Configuration for a single transformer-style layer.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayerConfig {
    /// Number of attention heads.
    pub attention_heads: usize,
    /// Number of key-value heads (for GQA; equals `attention_heads` for MHA).
    pub kv_heads: usize,
    /// Hidden (model) dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Per-head dimension (`hidden_dim / attention_heads`).
    pub head_dim: usize,
}

impl LayerConfig {
    /// Create a new layer config, computing `head_dim` automatically.
    ///
    /// Returns `None` if `attention_heads` is zero.
    pub fn new(
        attention_heads: usize,
        kv_heads: usize,
        hidden_dim: usize,
        ffn_dim: usize,
    ) -> Option<Self> {
        if attention_heads == 0 {
            return None;
        }
        Some(Self {
            attention_heads,
            kv_heads,
            hidden_dim,
            ffn_dim,
            head_dim: hidden_dim / attention_heads,
        })
    }
}

// ── Model architecture ───────────────────────────────────────────────────

/// Describes the overall architecture of a model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelArchitecture {
    /// Human-readable name (e.g. "LLaMA-7B").
    pub name: String,
    /// Architecture family.
    pub arch_type: ArchitectureType,
    /// Per-layer configuration (one entry per unique layer shape).
    pub layers: Vec<LayerConfig>,
    /// Total number of transformer layers.
    pub num_layers: usize,
}

// ── Model spec ───────────────────────────────────────────────────────────

/// Data type used for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum WeightDtype {
    F32,
    F16,
    BF16,
    I2S,
    I8,
}

impl WeightDtype {
    /// Bytes per element (approximate for sub-byte types).
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            // 2-bit ≈ 0.25 bytes; we return 1 as a conservative estimate.
            Self::I2S => 1,
        }
    }
}

/// Full model specification.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub max_seq_length: usize,
    pub dtype: WeightDtype,
}

impl ModelSpec {
    /// Approximate total parameter count.
    pub fn param_count(&self) -> u64 {
        let layer = match self.architecture.layers.first() {
            Some(l) => l,
            None => return 0,
        };
        let n = self.architecture.num_layers as u64;
        let h = layer.hidden_dim as u64;
        let ffn = layer.ffn_dim as u64;
        let vocab = self.vocab_size as u64;

        // Embedding + output head.
        let embed = vocab * h * 2;
        // Per-layer: QKV projections + output projection + FFN (gate+up+down).
        let qkv = h * h * 3;
        let out_proj = h * h;
        let ffn_params = h * ffn * 3;
        let per_layer = qkv + out_proj + ffn_params;

        embed + n * per_layer
    }
}

// ── Architecture registry ────────────────────────────────────────────────

/// Registry of known model architectures.
#[derive(Debug, Default)]
pub struct ArchitectureRegistry {
    entries: HashMap<String, ModelSpec>,
}

impl ArchitectureRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Populate with well-known model architectures.
    pub fn with_known_models() -> Self {
        let mut reg = Self::new();
        for spec in known_specs() {
            reg.register(spec);
        }
        reg
    }

    /// Register a model spec. Keyed by `architecture.name`.
    pub fn register(&mut self, spec: ModelSpec) {
        self.entries.insert(spec.architecture.name.clone(), spec);
    }

    /// Look up by name.
    pub fn get(&self, name: &str) -> Option<&ModelSpec> {
        self.entries.get(name)
    }

    /// Number of registered architectures.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all registered specs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ModelSpec)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Filter entries by architecture type.
    pub fn by_type(&self, arch_type: ArchitectureType) -> Vec<&ModelSpec> {
        self.entries.values().filter(|s| s.architecture.arch_type == arch_type).collect()
    }
}

// ── Architecture detector ────────────────────────────────────────────────

/// Detect architecture from GGUF / SafeTensors metadata key-value pairs.
pub struct ArchitectureDetector;

impl ArchitectureDetector {
    /// Detect architecture type from a metadata map.
    ///
    /// Looks for `general.architecture` (GGUF convention) and falls back
    /// to heuristic tensor-name inspection.
    pub fn detect(metadata: &HashMap<String, String>) -> Option<ArchitectureType> {
        if let Some(arch) = metadata.get("general.architecture") {
            return Self::parse_arch_string(arch);
        }
        // Heuristic: check for known tensor name patterns.
        if metadata.keys().any(|k| k.contains("bitnet")) {
            return Some(ArchitectureType::BitNetTransformer);
        }
        if metadata.keys().any(|k| k.contains("moe") || k.contains("expert")) {
            return Some(ArchitectureType::MoE);
        }
        if metadata.keys().any(|k| k.contains("rwkv")) {
            return Some(ArchitectureType::RWKV);
        }
        if metadata.keys().any(|k| k.contains("mamba") || k.contains("ssm")) {
            return Some(ArchitectureType::Mamba);
        }
        None
    }

    /// Parse the GGUF `general.architecture` value.
    pub fn parse_arch_string(s: &str) -> Option<ArchitectureType> {
        match s.to_lowercase().as_str() {
            "llama" | "mistral" | "phi" | "gpt2" | "falcon" | "gemma" | "starcoder" | "qwen2" => {
                Some(ArchitectureType::Transformer)
            }
            "bitnet" => Some(ArchitectureType::BitNetTransformer),
            "moe" | "mixtral" | "dbrx" => Some(ArchitectureType::MoE),
            "rwkv" => Some(ArchitectureType::RWKV),
            "mamba" => Some(ArchitectureType::Mamba),
            "bert" | "roberta" | "nomic-bert" => Some(ArchitectureType::Encoder),
            "t5" | "flan-t5" | "bart" => Some(ArchitectureType::EncoderDecoder),
            _ => None,
        }
    }

    /// Try to extract layer count from metadata.
    pub fn detect_layer_count(metadata: &HashMap<String, String>) -> Option<usize> {
        for key in
            &["llama.block_count", "gpt2.block_count", "bert.block_count", "general.block_count"]
        {
            if let Some(v) = metadata.get(*key) {
                if let Ok(n) = v.parse::<usize>() {
                    return Some(n);
                }
            }
        }
        None
    }
}

// ── Architecture validator ───────────────────────────────────────────────

/// Validation errors for layer configurations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// `hidden_dim` is not divisible by `attention_heads`.
    HeadDimMismatch { hidden_dim: usize, attention_heads: usize },
    /// `kv_heads` exceeds `attention_heads`.
    KvHeadsExceedAttentionHeads { kv_heads: usize, attention_heads: usize },
    /// `attention_heads` is not divisible by `kv_heads` (GQA requirement).
    GqaGroupMismatch { attention_heads: usize, kv_heads: usize },
    /// FFN dimension is zero.
    ZeroFfnDim,
    /// Hidden dimension is zero.
    ZeroHiddenDim,
    /// No layers defined.
    NoLayers,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeadDimMismatch { hidden_dim, attention_heads } => {
                write!(
                    f,
                    "hidden_dim {hidden_dim} not divisible by \
                     attention_heads {attention_heads}"
                )
            }
            Self::KvHeadsExceedAttentionHeads { kv_heads, attention_heads } => {
                write!(
                    f,
                    "kv_heads ({kv_heads}) exceeds \
                     attention_heads ({attention_heads})"
                )
            }
            Self::GqaGroupMismatch { attention_heads, kv_heads } => {
                write!(
                    f,
                    "attention_heads ({attention_heads}) not divisible \
                     by kv_heads ({kv_heads})"
                )
            }
            Self::ZeroFfnDim => write!(f, "ffn_dim must not be zero"),
            Self::ZeroHiddenDim => {
                write!(f, "hidden_dim must not be zero")
            }
            Self::NoLayers => write!(f, "architecture has no layers"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate layer and architecture configurations.
pub struct ArchitectureValidator;

impl ArchitectureValidator {
    /// Validate a single layer config.
    pub fn validate_layer(layer: &LayerConfig) -> Result<(), ValidationError> {
        if layer.hidden_dim == 0 {
            return Err(ValidationError::ZeroHiddenDim);
        }
        if layer.ffn_dim == 0 {
            return Err(ValidationError::ZeroFfnDim);
        }
        if layer.hidden_dim % layer.attention_heads != 0 {
            return Err(ValidationError::HeadDimMismatch {
                hidden_dim: layer.hidden_dim,
                attention_heads: layer.attention_heads,
            });
        }
        if layer.kv_heads > layer.attention_heads {
            return Err(ValidationError::KvHeadsExceedAttentionHeads {
                kv_heads: layer.kv_heads,
                attention_heads: layer.attention_heads,
            });
        }
        if layer.attention_heads % layer.kv_heads != 0 {
            return Err(ValidationError::GqaGroupMismatch {
                attention_heads: layer.attention_heads,
                kv_heads: layer.kv_heads,
            });
        }
        Ok(())
    }

    /// Validate an entire model architecture.
    pub fn validate(arch: &ModelArchitecture) -> Result<(), ValidationError> {
        if arch.layers.is_empty() {
            return Err(ValidationError::NoLayers);
        }
        for layer in &arch.layers {
            Self::validate_layer(layer)?;
        }
        Ok(())
    }
}

// ── Memory estimator ─────────────────────────────────────────────────────

/// Estimate memory requirements for inference.
pub struct MemoryEstimator;

/// Breakdown of estimated memory usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// Weight memory in bytes.
    pub weights_bytes: u64,
    /// KV-cache memory in bytes.
    pub kv_cache_bytes: u64,
    /// Activation scratch memory in bytes.
    pub activation_bytes: u64,
    /// Total estimated bytes.
    pub total_bytes: u64,
}

impl MemoryEstimator {
    /// Estimate memory for a given model spec and runtime parameters.
    pub fn estimate(spec: &ModelSpec, batch_size: usize, context_length: usize) -> MemoryEstimate {
        let bpe = spec.dtype.bytes_per_element() as u64;
        let params = spec.param_count();
        let weights_bytes = params * bpe;

        let layer = spec.architecture.layers.first();
        let (kv_cache_bytes, activation_bytes) = match layer {
            Some(l) => {
                // KV cache: 2 (K+V) × layers × kv_heads × head_dim × ctx × bpe
                let kv = 2u64
                    * spec.architecture.num_layers as u64
                    * l.kv_heads as u64
                    * l.head_dim as u64
                    * context_length as u64
                    * batch_size as u64
                    * 2; // KV stored in f16

                // Activations: batch × ctx × hidden × sizeof(f32)
                let act = batch_size as u64 * context_length as u64 * l.hidden_dim as u64 * 4;
                (kv, act)
            }
            None => (0, 0),
        };

        let total_bytes = weights_bytes + kv_cache_bytes + activation_bytes;
        MemoryEstimate { weights_bytes, kv_cache_bytes, activation_bytes, total_bytes }
    }
}

// ── Compute estimator ────────────────────────────────────────────────────

/// Estimate computational cost.
pub struct ComputeEstimator;

/// Breakdown of estimated FLOPs per token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeEstimate {
    /// Attention FLOPs per token.
    pub attention_flops: u64,
    /// FFN FLOPs per token.
    pub ffn_flops: u64,
    /// Total FLOPs per token.
    pub total_flops: u64,
}

impl ComputeEstimator {
    /// Estimate FLOPs per token for the given model spec.
    ///
    /// Uses the approximation: `2 × params` for a dense matmul-dominated
    /// forward pass, split roughly 40% attention / 60% FFN.
    pub fn estimate(spec: &ModelSpec) -> ComputeEstimate {
        let params = spec.param_count();
        let total_flops = 2 * params;
        let attention_flops = total_flops * 2 / 5;
        let ffn_flops = total_flops - attention_flops;
        ComputeEstimate { attention_flops, ffn_flops, total_flops }
    }
}

// ── Architecture comparator ──────────────────────────────────────────────

/// Summary of differences between two architectures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchDiff {
    pub name_a: String,
    pub name_b: String,
    pub type_matches: bool,
    pub layer_count_diff: i64,
    pub hidden_dim_diff: i64,
    pub ffn_dim_diff: i64,
    pub head_count_diff: i64,
}

/// Compare two architectures.
pub struct ArchitectureComparator;

impl ArchitectureComparator {
    /// Produce a diff summary between two architectures.
    pub fn compare(a: &ModelArchitecture, b: &ModelArchitecture) -> ArchDiff {
        let la = a.layers.first();
        let lb = b.layers.first();

        let (h_a, f_a, hd_a) =
            la.map(|l| (l.hidden_dim, l.ffn_dim, l.attention_heads)).unwrap_or((0, 0, 0));
        let (h_b, f_b, hd_b) =
            lb.map(|l| (l.hidden_dim, l.ffn_dim, l.attention_heads)).unwrap_or((0, 0, 0));

        ArchDiff {
            name_a: a.name.clone(),
            name_b: b.name.clone(),
            type_matches: a.arch_type == b.arch_type,
            layer_count_diff: b.num_layers as i64 - a.num_layers as i64,
            hidden_dim_diff: h_b as i64 - h_a as i64,
            ffn_dim_diff: f_b as i64 - f_a as i64,
            head_count_diff: hd_b as i64 - hd_a as i64,
        }
    }
}

// ── Known architectures ──────────────────────────────────────────────────

fn make_spec(
    name: &str,
    arch_type: ArchitectureType,
    num_layers: usize,
    attention_heads: usize,
    kv_heads: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_seq_length: usize,
    dtype: WeightDtype,
) -> ModelSpec {
    let layer = LayerConfig::new(attention_heads, kv_heads, hidden_dim, ffn_dim)
        .expect("invalid layer config in known spec");
    ModelSpec {
        architecture: ModelArchitecture {
            name: name.to_string(),
            arch_type,
            layers: vec![layer],
            num_layers,
        },
        vocab_size,
        max_seq_length,
        dtype,
    }
}

/// Built-in known model specifications.
pub fn known_specs() -> Vec<ModelSpec> {
    vec![
        // LLaMA-7B
        make_spec(
            "LLaMA-7B",
            ArchitectureType::Transformer,
            32,
            32,
            32,
            4096,
            11008,
            32000,
            4096,
            WeightDtype::F16,
        ),
        // BitNet-3B (BitNet b1.58 2B-4T style)
        make_spec(
            "BitNet-3B",
            ArchitectureType::BitNetTransformer,
            32,
            32,
            32,
            3200,
            8640,
            100352,
            4096,
            WeightDtype::I2S,
        ),
        // Mistral-7B
        make_spec(
            "Mistral-7B",
            ArchitectureType::Transformer,
            32,
            32,
            8,
            4096,
            14336,
            32000,
            32768,
            WeightDtype::F16,
        ),
        // GPT-2 (124M)
        make_spec(
            "GPT-2",
            ArchitectureType::Transformer,
            12,
            12,
            12,
            768,
            3072,
            50257,
            1024,
            WeightDtype::F32,
        ),
        // BERT-base
        make_spec(
            "BERT-base",
            ArchitectureType::Encoder,
            12,
            12,
            12,
            768,
            3072,
            30522,
            512,
            WeightDtype::F32,
        ),
        // T5-small
        make_spec(
            "T5-small",
            ArchitectureType::EncoderDecoder,
            6,
            8,
            8,
            512,
            2048,
            32128,
            512,
            WeightDtype::F32,
        ),
        // Phi-2
        make_spec(
            "Phi-2",
            ArchitectureType::Transformer,
            32,
            32,
            32,
            2560,
            10240,
            51200,
            2048,
            WeightDtype::F16,
        ),
    ]
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- ArchitectureType -------------------------------------------------

    #[test]
    fn architecture_type_display() {
        assert_eq!(ArchitectureType::Transformer.to_string(), "Transformer");
        assert_eq!(ArchitectureType::BitNetTransformer.to_string(), "BitNetTransformer");
        assert_eq!(ArchitectureType::MoE.to_string(), "MoE");
        assert_eq!(ArchitectureType::RWKV.to_string(), "RWKV");
        assert_eq!(ArchitectureType::Mamba.to_string(), "Mamba");
        assert_eq!(ArchitectureType::Encoder.to_string(), "Encoder");
        assert_eq!(ArchitectureType::EncoderDecoder.to_string(), "EncoderDecoder");
    }

    #[test]
    fn architecture_type_equality() {
        assert_eq!(ArchitectureType::Transformer, ArchitectureType::Transformer);
        assert_ne!(ArchitectureType::Transformer, ArchitectureType::MoE);
    }

    #[test]
    fn architecture_type_clone() {
        let a = ArchitectureType::Mamba;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn architecture_type_serde_roundtrip() {
        let t = ArchitectureType::BitNetTransformer;
        let json = serde_json::to_string(&t).unwrap();
        let back: ArchitectureType = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    // -- LayerConfig ------------------------------------------------------

    #[test]
    fn layer_config_new_computes_head_dim() {
        let lc = LayerConfig::new(32, 8, 4096, 11008).unwrap();
        assert_eq!(lc.head_dim, 128);
    }

    #[test]
    fn layer_config_new_zero_heads_returns_none() {
        assert!(LayerConfig::new(0, 0, 4096, 11008).is_none());
    }

    #[test]
    fn layer_config_mha_kv_equals_heads() {
        let lc = LayerConfig::new(12, 12, 768, 3072).unwrap();
        assert_eq!(lc.kv_heads, lc.attention_heads);
    }

    #[test]
    fn layer_config_serde_roundtrip() {
        let lc = LayerConfig::new(32, 8, 4096, 14336).unwrap();
        let json = serde_json::to_string(&lc).unwrap();
        let back: LayerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(lc, back);
    }

    // -- ModelArchitecture ------------------------------------------------

    #[test]
    fn model_architecture_basic() {
        let layer = LayerConfig::new(12, 12, 768, 3072).unwrap();
        let arch = ModelArchitecture {
            name: "test".to_string(),
            arch_type: ArchitectureType::Transformer,
            layers: vec![layer],
            num_layers: 12,
        };
        assert_eq!(arch.num_layers, 12);
        assert_eq!(arch.arch_type, ArchitectureType::Transformer);
    }

    // -- WeightDtype ------------------------------------------------------

    #[test]
    fn weight_dtype_bytes_per_element() {
        assert_eq!(WeightDtype::F32.bytes_per_element(), 4);
        assert_eq!(WeightDtype::F16.bytes_per_element(), 2);
        assert_eq!(WeightDtype::BF16.bytes_per_element(), 2);
        assert_eq!(WeightDtype::I8.bytes_per_element(), 1);
        assert_eq!(WeightDtype::I2S.bytes_per_element(), 1);
    }

    // -- ModelSpec --------------------------------------------------------

    #[test]
    fn model_spec_param_count_nonzero() {
        let specs = known_specs();
        for spec in &specs {
            assert!(spec.param_count() > 0, "{} param count should be > 0", spec.architecture.name);
        }
    }

    #[test]
    fn model_spec_param_count_llama_order_of_magnitude() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let params = llama.param_count();
        // LLaMA-7B: approximately 6-8 billion parameters.
        assert!(params > 5_000_000_000 && params < 10_000_000_000, "LLaMA-7B params = {params}");
    }

    #[test]
    fn model_spec_param_count_gpt2_small() {
        let reg = ArchitectureRegistry::with_known_models();
        let gpt2 = reg.get("GPT-2").unwrap();
        let params = gpt2.param_count();
        // GPT-2 124M: approximately 100-200M parameters.
        assert!(params > 50_000_000 && params < 300_000_000, "GPT-2 params = {params}");
    }

    #[test]
    fn model_spec_no_layers_zero_params() {
        let spec = ModelSpec {
            architecture: ModelArchitecture {
                name: "empty".to_string(),
                arch_type: ArchitectureType::Transformer,
                layers: vec![],
                num_layers: 0,
            },
            vocab_size: 1000,
            max_seq_length: 128,
            dtype: WeightDtype::F32,
        };
        assert_eq!(spec.param_count(), 0);
    }

    #[test]
    fn model_spec_serde_roundtrip() {
        let spec = known_specs().into_iter().next().unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        let back: ModelSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    // -- ArchitectureRegistry ---------------------------------------------

    #[test]
    fn registry_empty() {
        let reg = ArchitectureRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn registry_with_known_models_nonempty() {
        let reg = ArchitectureRegistry::with_known_models();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 7);
    }

    #[test]
    fn registry_get_existing() {
        let reg = ArchitectureRegistry::with_known_models();
        assert!(reg.get("LLaMA-7B").is_some());
        assert!(reg.get("BitNet-3B").is_some());
        assert!(reg.get("Mistral-7B").is_some());
        assert!(reg.get("GPT-2").is_some());
        assert!(reg.get("BERT-base").is_some());
        assert!(reg.get("T5-small").is_some());
        assert!(reg.get("Phi-2").is_some());
    }

    #[test]
    fn registry_get_missing() {
        let reg = ArchitectureRegistry::with_known_models();
        assert!(reg.get("NonExistent").is_none());
    }

    #[test]
    fn registry_register_custom() {
        let mut reg = ArchitectureRegistry::new();
        let spec = known_specs().into_iter().next().unwrap();
        reg.register(spec);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_register_overwrite() {
        let mut reg = ArchitectureRegistry::new();
        let spec = known_specs().into_iter().next().unwrap();
        reg.register(spec.clone());
        reg.register(spec);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_by_type_transformer() {
        let reg = ArchitectureRegistry::with_known_models();
        let transformers = reg.by_type(ArchitectureType::Transformer);
        // LLaMA-7B, Mistral-7B, GPT-2, Phi-2
        assert_eq!(transformers.len(), 4);
    }

    #[test]
    fn registry_by_type_bitnet() {
        let reg = ArchitectureRegistry::with_known_models();
        let bitnet = reg.by_type(ArchitectureType::BitNetTransformer);
        assert_eq!(bitnet.len(), 1);
    }

    #[test]
    fn registry_by_type_encoder() {
        let reg = ArchitectureRegistry::with_known_models();
        let enc = reg.by_type(ArchitectureType::Encoder);
        assert_eq!(enc.len(), 1);
    }

    #[test]
    fn registry_by_type_encoder_decoder() {
        let reg = ArchitectureRegistry::with_known_models();
        let encdec = reg.by_type(ArchitectureType::EncoderDecoder);
        assert_eq!(encdec.len(), 1);
    }

    #[test]
    fn registry_by_type_moe_empty() {
        let reg = ArchitectureRegistry::with_known_models();
        let moe = reg.by_type(ArchitectureType::MoE);
        assert!(moe.is_empty());
    }

    #[test]
    fn registry_iter_count() {
        let reg = ArchitectureRegistry::with_known_models();
        assert_eq!(reg.iter().count(), 7);
    }

    // -- ArchitectureDetector ---------------------------------------------

    #[test]
    fn detect_llama_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "llama".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::Transformer));
    }

    #[test]
    fn detect_bitnet_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "bitnet".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::BitNetTransformer));
    }

    #[test]
    fn detect_bert_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "bert".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::Encoder));
    }

    #[test]
    fn detect_t5_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "t5".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::EncoderDecoder));
    }

    #[test]
    fn detect_moe_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "mixtral".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::MoE));
    }

    #[test]
    fn detect_rwkv_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "rwkv".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::RWKV));
    }

    #[test]
    fn detect_mamba_from_metadata() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "mamba".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::Mamba));
    }

    #[test]
    fn detect_unknown_arch_returns_none() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), "unknown_arch".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), None);
    }

    #[test]
    fn detect_empty_metadata_returns_none() {
        let meta = HashMap::new();
        assert_eq!(ArchitectureDetector::detect(&meta), None);
    }

    #[test]
    fn detect_heuristic_bitnet_key() {
        let mut meta = HashMap::new();
        meta.insert("blk.0.bitnet.weight".to_string(), "tensor".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::BitNetTransformer));
    }

    #[test]
    fn detect_heuristic_moe_expert_key() {
        let mut meta = HashMap::new();
        meta.insert("blk.0.expert.0.weight".to_string(), "t".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::MoE));
    }

    #[test]
    fn detect_heuristic_rwkv_key() {
        let mut meta = HashMap::new();
        meta.insert("blk.0.rwkv.time_mix".to_string(), "t".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::RWKV));
    }

    #[test]
    fn detect_heuristic_mamba_ssm_key() {
        let mut meta = HashMap::new();
        meta.insert("blk.0.ssm.in_proj".to_string(), "t".to_string());
        assert_eq!(ArchitectureDetector::detect(&meta), Some(ArchitectureType::Mamba));
    }

    #[test]
    fn detect_layer_count_llama() {
        let mut meta = HashMap::new();
        meta.insert("llama.block_count".to_string(), "32".to_string());
        assert_eq!(ArchitectureDetector::detect_layer_count(&meta), Some(32));
    }

    #[test]
    fn detect_layer_count_gpt2() {
        let mut meta = HashMap::new();
        meta.insert("gpt2.block_count".to_string(), "12".to_string());
        assert_eq!(ArchitectureDetector::detect_layer_count(&meta), Some(12));
    }

    #[test]
    fn detect_layer_count_missing() {
        let meta = HashMap::new();
        assert_eq!(ArchitectureDetector::detect_layer_count(&meta), None);
    }

    #[test]
    fn parse_arch_string_case_insensitive() {
        assert_eq!(
            ArchitectureDetector::parse_arch_string("LLAMA"),
            Some(ArchitectureType::Transformer)
        );
        assert_eq!(
            ArchitectureDetector::parse_arch_string("Bert"),
            Some(ArchitectureType::Encoder)
        );
    }

    #[test]
    fn parse_arch_string_mistral() {
        assert_eq!(
            ArchitectureDetector::parse_arch_string("mistral"),
            Some(ArchitectureType::Transformer)
        );
    }

    #[test]
    fn parse_arch_string_phi() {
        assert_eq!(
            ArchitectureDetector::parse_arch_string("phi"),
            Some(ArchitectureType::Transformer)
        );
    }

    #[test]
    fn parse_arch_string_bart() {
        assert_eq!(
            ArchitectureDetector::parse_arch_string("bart"),
            Some(ArchitectureType::EncoderDecoder)
        );
    }

    // -- ArchitectureValidator --------------------------------------------

    #[test]
    fn validate_layer_valid() {
        let lc = LayerConfig::new(32, 8, 4096, 14336).unwrap();
        assert!(ArchitectureValidator::validate_layer(&lc).is_ok());
    }

    #[test]
    fn validate_layer_zero_hidden() {
        let lc = LayerConfig {
            attention_heads: 8,
            kv_heads: 8,
            hidden_dim: 0,
            ffn_dim: 1024,
            head_dim: 0,
        };
        assert_eq!(ArchitectureValidator::validate_layer(&lc), Err(ValidationError::ZeroHiddenDim));
    }

    #[test]
    fn validate_layer_zero_ffn() {
        let lc = LayerConfig {
            attention_heads: 8,
            kv_heads: 8,
            hidden_dim: 512,
            ffn_dim: 0,
            head_dim: 64,
        };
        assert_eq!(ArchitectureValidator::validate_layer(&lc), Err(ValidationError::ZeroFfnDim));
    }

    #[test]
    fn validate_layer_head_dim_mismatch() {
        let lc = LayerConfig {
            attention_heads: 7,
            kv_heads: 7,
            hidden_dim: 512,
            ffn_dim: 2048,
            head_dim: 73,
        };
        assert!(matches!(
            ArchitectureValidator::validate_layer(&lc),
            Err(ValidationError::HeadDimMismatch { .. })
        ));
    }

    #[test]
    fn validate_layer_kv_exceeds_heads() {
        let lc = LayerConfig {
            attention_heads: 8,
            kv_heads: 16,
            hidden_dim: 512,
            ffn_dim: 2048,
            head_dim: 64,
        };
        assert!(matches!(
            ArchitectureValidator::validate_layer(&lc),
            Err(ValidationError::KvHeadsExceedAttentionHeads { .. })
        ));
    }

    #[test]
    fn validate_layer_gqa_mismatch() {
        let lc = LayerConfig {
            attention_heads: 8,
            kv_heads: 3,
            hidden_dim: 512,
            ffn_dim: 2048,
            head_dim: 64,
        };
        assert!(matches!(
            ArchitectureValidator::validate_layer(&lc),
            Err(ValidationError::GqaGroupMismatch { .. })
        ));
    }

    #[test]
    fn validate_architecture_no_layers() {
        let arch = ModelArchitecture {
            name: "empty".to_string(),
            arch_type: ArchitectureType::Transformer,
            layers: vec![],
            num_layers: 0,
        };
        assert_eq!(ArchitectureValidator::validate(&arch), Err(ValidationError::NoLayers));
    }

    #[test]
    fn validate_all_known_architectures() {
        for spec in known_specs() {
            assert!(
                ArchitectureValidator::validate(&spec.architecture).is_ok(),
                "known arch {} should be valid",
                spec.architecture.name
            );
        }
    }

    // -- MemoryEstimator --------------------------------------------------

    #[test]
    fn memory_estimate_nonzero() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let est = MemoryEstimator::estimate(llama, 1, 2048);
        assert!(est.total_bytes > 0);
        assert!(est.weights_bytes > 0);
        assert!(est.kv_cache_bytes > 0);
        assert!(est.activation_bytes > 0);
    }

    #[test]
    fn memory_estimate_larger_batch_costs_more() {
        let reg = ArchitectureRegistry::with_known_models();
        let spec = reg.get("LLaMA-7B").unwrap();
        let est1 = MemoryEstimator::estimate(spec, 1, 2048);
        let est4 = MemoryEstimator::estimate(spec, 4, 2048);
        assert!(est4.total_bytes > est1.total_bytes);
    }

    #[test]
    fn memory_estimate_longer_context_costs_more() {
        let reg = ArchitectureRegistry::with_known_models();
        let spec = reg.get("LLaMA-7B").unwrap();
        let short = MemoryEstimator::estimate(spec, 1, 512);
        let long = MemoryEstimator::estimate(spec, 1, 4096);
        assert!(long.total_bytes > short.total_bytes);
    }

    #[test]
    fn memory_estimate_total_is_sum() {
        let reg = ArchitectureRegistry::with_known_models();
        let spec = reg.get("GPT-2").unwrap();
        let est = MemoryEstimator::estimate(spec, 1, 1024);
        assert_eq!(est.total_bytes, est.weights_bytes + est.kv_cache_bytes + est.activation_bytes);
    }

    #[test]
    fn memory_estimate_bitnet_smaller_weights() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let bitnet = reg.get("BitNet-3B").unwrap();
        let est_llama = MemoryEstimator::estimate(llama, 1, 2048);
        let est_bitnet = MemoryEstimator::estimate(bitnet, 1, 2048);
        // BitNet I2S uses fewer bytes per element.
        assert!(est_bitnet.weights_bytes < est_llama.weights_bytes);
    }

    #[test]
    fn memory_estimate_empty_arch() {
        let spec = ModelSpec {
            architecture: ModelArchitecture {
                name: "empty".to_string(),
                arch_type: ArchitectureType::Transformer,
                layers: vec![],
                num_layers: 0,
            },
            vocab_size: 1000,
            max_seq_length: 128,
            dtype: WeightDtype::F32,
        };
        let est = MemoryEstimator::estimate(&spec, 1, 128);
        assert_eq!(est.kv_cache_bytes, 0);
        assert_eq!(est.activation_bytes, 0);
    }

    // -- ComputeEstimator -------------------------------------------------

    #[test]
    fn compute_estimate_nonzero() {
        let reg = ArchitectureRegistry::with_known_models();
        let spec = reg.get("LLaMA-7B").unwrap();
        let est = ComputeEstimator::estimate(spec);
        assert!(est.total_flops > 0);
    }

    #[test]
    fn compute_estimate_total_is_sum() {
        let reg = ArchitectureRegistry::with_known_models();
        let spec = reg.get("GPT-2").unwrap();
        let est = ComputeEstimator::estimate(spec);
        assert_eq!(est.total_flops, est.attention_flops + est.ffn_flops);
    }

    #[test]
    fn compute_estimate_larger_model_more_flops() {
        let reg = ArchitectureRegistry::with_known_models();
        let gpt2 = reg.get("GPT-2").unwrap();
        let llama = reg.get("LLaMA-7B").unwrap();
        let est_gpt2 = ComputeEstimator::estimate(gpt2);
        let est_llama = ComputeEstimator::estimate(llama);
        assert!(est_llama.total_flops > est_gpt2.total_flops);
    }

    // -- ArchitectureComparator -------------------------------------------

    #[test]
    fn compare_same_architecture() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let diff = ArchitectureComparator::compare(&llama.architecture, &llama.architecture);
        assert!(diff.type_matches);
        assert_eq!(diff.layer_count_diff, 0);
        assert_eq!(diff.hidden_dim_diff, 0);
        assert_eq!(diff.ffn_dim_diff, 0);
        assert_eq!(diff.head_count_diff, 0);
    }

    #[test]
    fn compare_different_architectures() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let gpt2 = reg.get("GPT-2").unwrap();
        let diff = ArchitectureComparator::compare(&gpt2.architecture, &llama.architecture);
        assert!(diff.type_matches); // Both Transformer
        assert!(diff.layer_count_diff > 0); // LLaMA has more layers
        assert!(diff.hidden_dim_diff > 0); // LLaMA has larger hidden dim
    }

    #[test]
    fn compare_different_types() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let bert = reg.get("BERT-base").unwrap();
        let diff = ArchitectureComparator::compare(&llama.architecture, &bert.architecture);
        assert!(!diff.type_matches);
    }

    #[test]
    fn compare_llama_vs_mistral() {
        let reg = ArchitectureRegistry::with_known_models();
        let llama = reg.get("LLaMA-7B").unwrap();
        let mistral = reg.get("Mistral-7B").unwrap();
        let diff = ArchitectureComparator::compare(&llama.architecture, &mistral.architecture);
        assert!(diff.type_matches);
        assert_eq!(diff.layer_count_diff, 0); // Both 32 layers
        assert_eq!(diff.hidden_dim_diff, 0); // Both 4096
        assert!(diff.ffn_dim_diff > 0); // Mistral has larger FFN
    }

    // -- ValidationError Display ------------------------------------------

    #[test]
    fn validation_error_display() {
        let e = ValidationError::ZeroFfnDim;
        assert_eq!(e.to_string(), "ffn_dim must not be zero");
    }

    #[test]
    fn validation_error_head_dim_display() {
        let e = ValidationError::HeadDimMismatch { hidden_dim: 512, attention_heads: 7 };
        let s = e.to_string();
        assert!(s.contains("512"));
        assert!(s.contains("7"));
    }

    // -- known_specs consistency ------------------------------------------

    #[test]
    fn known_specs_all_have_layers() {
        for spec in known_specs() {
            assert!(
                !spec.architecture.layers.is_empty(),
                "{} has no layers",
                spec.architecture.name
            );
        }
    }

    #[test]
    fn known_specs_all_have_positive_vocab() {
        for spec in known_specs() {
            assert!(spec.vocab_size > 0, "{} has zero vocab", spec.architecture.name);
        }
    }

    #[test]
    fn known_specs_all_have_positive_seq_length() {
        for spec in known_specs() {
            assert!(spec.max_seq_length > 0, "{} has zero max_seq_length", spec.architecture.name);
        }
    }

    #[test]
    fn known_specs_unique_names() {
        let specs = known_specs();
        let mut names: Vec<_> = specs.iter().map(|s| &s.architecture.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), specs.len());
    }

    #[test]
    fn known_specs_count() {
        assert_eq!(known_specs().len(), 7);
    }
}
