//! GGUF model configuration parser with validation and memory estimation.
//!
//! Extracts and validates model architecture parameters from GGUF metadata,
//! providing structured access to architecture, attention, quantization, and
//! RoPE configuration. Includes memory estimation for deployment planning.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::formats::gguf::GgufValue;

// ---------------------------------------------------------------------------
// GGUF metadata key constants
// ---------------------------------------------------------------------------

const KEY_GENERAL_ARCHITECTURE: &str = "general.architecture";
const KEY_GENERAL_NAME: &str = "general.name";

/// Architecture-prefixed key lookup order for a given field.
/// Tries `{arch}.{field}`, then common fallbacks.
fn arch_keys(arch: &str, field: &str, fallbacks: &[&str]) -> Vec<String> {
    let mut keys = vec![format!("{arch}.{field}")];
    for fb in fallbacks {
        keys.push((*fb).to_string());
    }
    keys
}

// ---------------------------------------------------------------------------
// Helper: extract typed values from metadata map
// ---------------------------------------------------------------------------

fn get_string(map: &HashMap<String, GgufValue>, key: &str) -> Option<String> {
    match map.get(key)? {
        GgufValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn get_u32(map: &HashMap<String, GgufValue>, key: &str) -> Option<u32> {
    match map.get(key)? {
        GgufValue::U32(v) => Some(*v),
        GgufValue::I32(v) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}

fn get_f32(map: &HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    match map.get(key)? {
        GgufValue::F32(v) => Some(*v),
        _ => None,
    }
}

/// Try multiple keys in order and return the first u32 match.
fn get_u32_any(map: &HashMap<String, GgufValue>, keys: &[String]) -> Option<u32> {
    keys.iter().find_map(|k| get_u32(map, k))
}

/// Try multiple keys in order and return the first f32 match.
fn get_f32_any(map: &HashMap<String, GgufValue>, keys: &[String]) -> Option<f32> {
    keys.iter().find_map(|k| get_f32(map, k))
}

/// Try multiple keys in order and return the first string match.
fn get_string_any(map: &HashMap<String, GgufValue>, keys: &[String]) -> Option<String> {
    keys.iter().find_map(|k| get_string(map, k))
}

// ---------------------------------------------------------------------------
// RoPE scaling
// ---------------------------------------------------------------------------

/// RoPE (Rotary Position Embedding) scaling strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RopeScalingType {
    /// No scaling applied.
    None,
    /// Linear interpolation scaling.
    Linear,
    /// NTK-aware scaling (Neural Tangent Kernel).
    Ntk,
    /// YaRN (Yet another RoPE extensioN) scaling.
    YaRn,
}

/// RoPE scaling configuration extracted from GGUF metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: RopeScalingType,
    pub factor: f32,
}

// ---------------------------------------------------------------------------
// Quantization config
// ---------------------------------------------------------------------------

/// Quantization configuration extracted from GGUF metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufQuantizationConfig {
    /// Quantization bit width (e.g., 2 for I2_S).
    pub bit_width: u32,
    /// Block size used for quantized storage.
    pub block_size: usize,
    /// Human-readable format description (e.g., "I2_S", "QK256").
    pub format: String,
}

impl Default for GgufQuantizationConfig {
    fn default() -> Self {
        Self { bit_width: 2, block_size: 64, format: "I2_S".to_string() }
    }
}

// ---------------------------------------------------------------------------
// Memory estimate
// ---------------------------------------------------------------------------

/// Estimated memory requirements for loading and running a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    /// Estimated weight memory in bytes.
    pub weight_bytes: u64,
    /// Estimated KV-cache memory in bytes (single sequence, all layers).
    pub kv_cache_bytes: u64,
    /// Total estimated memory in bytes.
    pub total_bytes: u64,
    /// Human-readable summary.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// GgufModelConfig
// ---------------------------------------------------------------------------

/// Structured model architecture configuration extracted from GGUF metadata.
///
/// Provides typed, validated access to architecture parameters that are
/// otherwise scattered across GGUF key-value pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufModelConfig {
    /// Architecture family (e.g., "llama", "bitnet").
    pub architecture: String,
    /// Optional model name from GGUF metadata.
    pub model_name: Option<String>,
    /// Vocabulary size (token count).
    pub vocab_size: usize,
    /// Hidden dimension / embedding length.
    pub hidden_size: usize,
    /// Number of transformer layers (block count).
    pub num_layers: usize,
    /// Number of attention heads (queries).
    pub num_heads: usize,
    /// Number of key/value heads (for GQA); equals `num_heads` for MHA.
    pub num_kv_heads: usize,
    /// Per-head dimension (`hidden_size / num_heads`).
    pub head_dim: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Maximum sequence length (context length).
    pub max_seq_len: usize,
    /// RoPE base frequency (theta).
    pub rope_theta: f64,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<RopeScaling>,
    /// Quantization parameters.
    pub quantization: GgufQuantizationConfig,
}

impl GgufModelConfig {
    /// Extract model configuration from a GGUF metadata map.
    ///
    /// Keys are tried with the detected architecture prefix first (e.g.,
    /// `llama.embedding_length`), then common fallback names.
    pub fn from_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> Result<Self, ConfigError> {
        let architecture =
            get_string(metadata, KEY_GENERAL_ARCHITECTURE).unwrap_or_else(|| "llama".to_string());
        let model_name = get_string(metadata, KEY_GENERAL_NAME);

        let arch = architecture.as_str();

        // --- vocab_size ---
        let vocab_size =
            get_u32_any(metadata, &arch_keys(arch, "vocab_size", &["tokenizer.ggml.vocab_size"]))
                .unwrap_or(32_000) as usize;

        // --- hidden_size ---
        let hidden_size =
            get_u32_any(metadata, &arch_keys(arch, "embedding_length", &["n_embd", "hidden_size"]))
                .unwrap_or(4096) as usize;

        // --- num_layers ---
        let num_layers = get_u32_any(metadata, &arch_keys(arch, "block_count", &["n_layer"]))
            .unwrap_or(32) as usize;

        // --- num_heads ---
        let num_heads = get_u32_any(
            metadata,
            &arch_keys(
                arch,
                "attention.head_count",
                &["n_head", "attn.n_heads", "num_attention_heads"],
            ),
        )
        .unwrap_or(32) as usize;

        // --- num_kv_heads (GQA) ---
        let num_kv_heads = get_u32_any(
            metadata,
            &arch_keys(
                arch,
                "attention.head_count_kv",
                &["n_head_kv", "n_kv_heads", "num_key_value_heads"],
            ),
        )
        .map(|v| v as usize)
        .unwrap_or(num_heads);

        // --- head_dim ---
        let head_dim = if num_heads > 0 { hidden_size / num_heads } else { 0 };

        // --- intermediate_size ---
        let intermediate_size =
            get_u32_any(metadata, &arch_keys(arch, "feed_forward_length", &["n_ff"]))
                .unwrap_or(11_008) as usize;

        // --- max_seq_len ---
        let max_seq_len =
            get_u32_any(metadata, &arch_keys(arch, "context_length", &[])).unwrap_or(2048) as usize;

        // --- rope_theta ---
        let rope_theta =
            get_f32_any(metadata, &arch_keys(arch, "rope.freq_base", &["rope.freq_base"]))
                .map(|v| v as f64)
                .unwrap_or(10_000.0);

        // --- rope_scaling ---
        let rope_scaling = Self::parse_rope_scaling(metadata, arch);

        // --- quantization ---
        let quantization = Self::parse_quantization(metadata, arch);

        Ok(Self {
            architecture,
            model_name,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            max_seq_len,
            rope_theta,
            rope_scaling,
            quantization,
        })
    }

    /// Validate internal consistency of the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();

        if self.vocab_size == 0 {
            errors.push("vocab_size must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            errors.push("hidden_size must be > 0".to_string());
        }
        if self.num_layers == 0 {
            errors.push("num_layers must be > 0".to_string());
        }
        if self.num_heads == 0 {
            errors.push("num_heads must be > 0".to_string());
        }
        if self.num_kv_heads == 0 {
            errors.push("num_kv_heads must be > 0".to_string());
        }
        if self.intermediate_size == 0 {
            errors.push("intermediate_size must be > 0".to_string());
        }
        if self.max_seq_len == 0 {
            errors.push("max_seq_len must be > 0".to_string());
        }

        // Dimensional consistency
        if self.num_heads > 0 && !self.hidden_size.is_multiple_of(self.num_heads) {
            errors.push(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            ));
        }
        if self.num_heads > 0 && self.num_kv_heads > 0 && self.num_kv_heads > self.num_heads {
            errors.push(format!(
                "num_kv_heads ({}) must be <= num_heads ({})",
                self.num_kv_heads, self.num_heads
            ));
        }
        if self.num_heads > 0
            && self.num_kv_heads > 0
            && !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            errors.push(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.num_heads > 0 && self.head_dim != self.hidden_size / self.num_heads {
            errors.push(format!(
                "head_dim ({}) must equal hidden_size / num_heads ({}/{}={})",
                self.head_dim,
                self.hidden_size,
                self.num_heads,
                self.hidden_size / self.num_heads
            ));
        }

        if errors.is_empty() { Ok(()) } else { Err(ConfigError::Validation(errors.join("; "))) }
    }

    /// Estimate memory requirements for loading and running the model.
    ///
    /// Weight memory is estimated from layer parameters and quantization
    /// bit width. KV-cache is estimated for a single full-length sequence.
    pub fn memory_estimate(&self) -> MemoryEstimate {
        let bits = self.quantization.bit_width as u64;
        let h = self.hidden_size as u64;
        let l = self.num_layers as u64;
        let ff = self.intermediate_size as u64;
        let v = self.vocab_size as u64;
        let kv_heads = self.num_kv_heads as u64;
        let head_dim = self.head_dim as u64;
        let seq = self.max_seq_len as u64;

        // --- weight memory ---
        // Per-layer attention: Q, K, V, O projections
        let attn_params_per_layer = h * h // Q
            + h * (kv_heads * head_dim) // K
            + h * (kv_heads * head_dim) // V
            + h * h; // O
        // Per-layer FFN: gate, up, down (SwiGLU pattern)
        let ffn_params_per_layer = h * ff + h * ff + ff * h;
        // Per-layer norms (2 × hidden, stored as FP32 regardless)
        let norm_bytes_per_layer = 2 * h * 4; // 4 bytes/f32

        let layer_weight_bits = (attn_params_per_layer + ffn_params_per_layer) * bits;
        let layer_weight_bytes = layer_weight_bits / 8 + norm_bytes_per_layer;

        // Embedding + output head (typically FP16 = 16 bits)
        let embed_bytes = v * h * 16 / 8;

        let weight_bytes = l * layer_weight_bytes + embed_bytes;

        // --- KV cache memory (FP16, single sequence) ---
        // Per layer: 2 (K+V) × kv_heads × head_dim × seq_len × 2 bytes
        let kv_cache_bytes = l * 2 * kv_heads * head_dim * seq * 2;

        let total_bytes = weight_bytes + kv_cache_bytes;

        let summary = format!(
            "weights: {:.1} MiB, kv_cache: {:.1} MiB, total: {:.1} MiB",
            weight_bytes as f64 / (1024.0 * 1024.0),
            kv_cache_bytes as f64 / (1024.0 * 1024.0),
            total_bytes as f64 / (1024.0 * 1024.0),
        );

        MemoryEstimate { weight_bytes, kv_cache_bytes, total_bytes, summary }
    }

    /// Returns `true` if the model uses Grouped-Query Attention.
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads > 0 && self.num_kv_heads < self.num_heads
    }

    /// Returns the GQA group size (`num_heads / num_kv_heads`).
    pub fn gqa_group_size(&self) -> usize {
        if self.num_kv_heads > 0 { self.num_heads / self.num_kv_heads } else { 1 }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn parse_rope_scaling(
        metadata: &HashMap<String, GgufValue>,
        arch: &str,
    ) -> Option<RopeScaling> {
        let type_keys = arch_keys(arch, "rope.scaling.type", &["rope.scaling.type"]);
        let factor_keys = arch_keys(arch, "rope.scaling.factor", &["rope.scaling.factor"]);

        let scaling_type_str = get_string_any(metadata, &type_keys)?;
        let factor = get_f32_any(metadata, &factor_keys).unwrap_or(1.0);

        let scaling_type = match scaling_type_str.to_lowercase().as_str() {
            "linear" => RopeScalingType::Linear,
            "ntk" => RopeScalingType::Ntk,
            "yarn" => RopeScalingType::YaRn,
            "none" | "" => return None,
            _ => return None,
        };

        Some(RopeScaling { scaling_type, factor })
    }

    fn parse_quantization(
        metadata: &HashMap<String, GgufValue>,
        arch: &str,
    ) -> GgufQuantizationConfig {
        let bit_keys = arch_keys(arch, "quantization.bit_width", &["general.quantization_version"]);
        let block_keys = arch_keys(arch, "quantization.block_size", &[]);
        let fmt_keys = arch_keys(arch, "quantization.format", &["general.file_type"]);

        let bit_width = get_u32_any(metadata, &bit_keys).unwrap_or(2);
        let block_size = get_u32_any(metadata, &block_keys).unwrap_or(64) as usize;
        let format = get_string_any(metadata, &fmt_keys).unwrap_or_else(|| "I2_S".to_string());

        GgufQuantizationConfig { bit_width, block_size, format }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors arising from GGUF model configuration parsing or validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("missing required metadata key: {0}")]
    MissingKey(String),
    #[error("invalid metadata value for key '{key}': {reason}")]
    InvalidValue { key: String, reason: String },
    #[error("configuration validation failed: {0}")]
    Validation(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a metadata map for a typical LLaMA-style model.
    fn llama_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            (KEY_GENERAL_ARCHITECTURE.into(), GgufValue::String("llama".into())),
            (KEY_GENERAL_NAME.into(), GgufValue::String("TestLlama-7B".into())),
            ("llama.vocab_size".into(), GgufValue::U32(32000)),
            ("llama.embedding_length".into(), GgufValue::U32(4096)),
            ("llama.block_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count_kv".into(), GgufValue::U32(32)),
            ("llama.feed_forward_length".into(), GgufValue::U32(11008)),
            ("llama.context_length".into(), GgufValue::U32(4096)),
            ("llama.rope.freq_base".into(), GgufValue::F32(10000.0)),
        ])
    }

    /// BitNet-style metadata (architecture = "bitnet").
    fn bitnet_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            (KEY_GENERAL_ARCHITECTURE.into(), GgufValue::String("bitnet".into())),
            (KEY_GENERAL_NAME.into(), GgufValue::String("BitNet-2B".into())),
            ("bitnet.vocab_size".into(), GgufValue::U32(100352)),
            ("bitnet.embedding_length".into(), GgufValue::U32(2048)),
            ("bitnet.block_count".into(), GgufValue::U32(24)),
            ("bitnet.attention.head_count".into(), GgufValue::U32(32)),
            ("bitnet.attention.head_count_kv".into(), GgufValue::U32(8)),
            ("bitnet.feed_forward_length".into(), GgufValue::U32(5632)),
            ("bitnet.context_length".into(), GgufValue::U32(2048)),
            ("bitnet.rope.freq_base".into(), GgufValue::F32(500000.0)),
        ])
    }

    /// GQA metadata where kv_heads < heads.
    fn gqa_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            (KEY_GENERAL_ARCHITECTURE.into(), GgufValue::String("llama".into())),
            ("llama.vocab_size".into(), GgufValue::U32(32000)),
            ("llama.embedding_length".into(), GgufValue::U32(4096)),
            ("llama.block_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count_kv".into(), GgufValue::U32(8)),
            ("llama.feed_forward_length".into(), GgufValue::U32(14336)),
            ("llama.context_length".into(), GgufValue::U32(8192)),
            ("llama.rope.freq_base".into(), GgufValue::F32(500000.0)),
        ])
    }

    // =======================================================================
    // Parsing tests
    // =======================================================================

    #[test]
    fn parse_llama_metadata() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        assert_eq!(cfg.architecture, "llama");
        assert_eq!(cfg.model_name.as_deref(), Some("TestLlama-7B"));
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.max_seq_len, 4096);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn parse_bitnet_metadata() {
        let cfg = GgufModelConfig::from_gguf_metadata(&bitnet_metadata()).unwrap();
        assert_eq!(cfg.architecture, "bitnet");
        assert_eq!(cfg.model_name.as_deref(), Some("BitNet-2B"));
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn parse_empty_metadata_uses_defaults() {
        let cfg = GgufModelConfig::from_gguf_metadata(&HashMap::new()).unwrap();
        assert_eq!(cfg.architecture, "llama");
        assert!(cfg.model_name.is_none());
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 32);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    #[test]
    fn parse_fallback_keys() {
        let meta = HashMap::from([
            ("n_embd".into(), GgufValue::U32(1024)),
            ("n_layer".into(), GgufValue::U32(12)),
            ("n_head".into(), GgufValue::U32(16)),
            ("n_head_kv".into(), GgufValue::U32(4)),
            ("n_ff".into(), GgufValue::U32(2816)),
        ]);
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.intermediate_size, 2816);
    }

    #[test]
    fn parse_i32_positive_as_u32() {
        let meta = HashMap::from([
            ("llama.block_count".into(), GgufValue::I32(24)),
            ("llama.embedding_length".into(), GgufValue::I32(2048)),
        ]);
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.hidden_size, 2048);
    }

    // =======================================================================
    // Validation tests
    // =======================================================================

    #[test]
    fn validate_correct_config() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_bitnet_gqa_config() {
        let cfg = GgufModelConfig::from_gguf_metadata(&bitnet_metadata()).unwrap();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_layers() {
        let mut meta = llama_metadata();
        meta.insert("llama.block_count".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("num_layers must be > 0"));
    }

    #[test]
    fn validate_rejects_zero_hidden_size() {
        let mut meta = llama_metadata();
        meta.insert("llama.embedding_length".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hidden_size must be > 0"));
    }

    #[test]
    fn validate_rejects_zero_vocab_size() {
        let mut meta = llama_metadata();
        meta.insert("llama.vocab_size".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("vocab_size must be > 0"));
    }

    #[test]
    fn validate_rejects_zero_num_heads() {
        let mut meta = llama_metadata();
        meta.insert("llama.attention.head_count".into(), GgufValue::U32(0));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("num_heads must be > 0"));
    }

    #[test]
    fn validate_rejects_mismatched_head_dim() {
        // hidden_size=4096 with num_heads=3 → not divisible
        let mut meta = llama_metadata();
        meta.insert("llama.attention.head_count".into(), GgufValue::U32(3));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::U32(3));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hidden_size"));
    }

    #[test]
    fn validate_rejects_kv_heads_greater_than_heads() {
        let mut cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        cfg.num_kv_heads = cfg.num_heads + 1;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("num_kv_heads"));
    }

    #[test]
    fn validate_rejects_kv_heads_not_dividing_heads() {
        // num_heads=32, num_kv_heads=5 → 32 % 5 != 0
        let mut cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        cfg.num_kv_heads = 5;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("divisible by num_kv_heads"));
    }

    #[test]
    fn validate_rejects_zero_intermediate_size() {
        let mut meta = llama_metadata();
        meta.insert("llama.feed_forward_length".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("intermediate_size must be > 0"));
    }

    #[test]
    fn validate_rejects_zero_max_seq_len() {
        let mut meta = llama_metadata();
        meta.insert("llama.context_length".into(), GgufValue::U32(0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("max_seq_len must be > 0"));
    }

    // =======================================================================
    // GQA detection
    // =======================================================================

    #[test]
    fn gqa_detected_correctly() {
        let cfg = GgufModelConfig::from_gguf_metadata(&gqa_metadata()).unwrap();
        assert!(cfg.is_gqa());
        assert_eq!(cfg.gqa_group_size(), 4); // 32 / 8
    }

    #[test]
    fn mha_is_not_gqa() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        assert!(!cfg.is_gqa());
        assert_eq!(cfg.gqa_group_size(), 1);
    }

    #[test]
    fn bitnet_gqa_detected() {
        let cfg = GgufModelConfig::from_gguf_metadata(&bitnet_metadata()).unwrap();
        assert!(cfg.is_gqa());
        assert_eq!(cfg.gqa_group_size(), 4); // 32 / 8
    }

    // =======================================================================
    // Memory estimation
    // =======================================================================

    #[test]
    fn memory_estimate_nonzero() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        let est = cfg.memory_estimate();
        assert!(est.weight_bytes > 0);
        assert!(est.kv_cache_bytes > 0);
        assert_eq!(est.total_bytes, est.weight_bytes + est.kv_cache_bytes);
        assert!(est.summary.contains("MiB"));
    }

    #[test]
    fn memory_estimate_gqa_smaller_kv_cache() {
        let mha = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        let gqa = GgufModelConfig::from_gguf_metadata(&gqa_metadata()).unwrap();
        // GQA has 8 kv_heads vs 32, but also double the context length;
        // normalize by seq_len to compare per-token kv cost.
        let mha_per_token = mha.memory_estimate().kv_cache_bytes / mha.max_seq_len as u64;
        let gqa_per_token = gqa.memory_estimate().kv_cache_bytes / gqa.max_seq_len as u64;
        assert!(gqa_per_token < mha_per_token);
    }

    #[test]
    fn memory_estimate_bitnet_smaller_weights() {
        let llama = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        let bitnet = GgufModelConfig::from_gguf_metadata(&bitnet_metadata()).unwrap();
        // BitNet 2B has fewer layers and smaller hidden_size → smaller weights
        assert!(bitnet.memory_estimate().weight_bytes < llama.memory_estimate().weight_bytes);
    }

    // =======================================================================
    // RoPE scaling
    // =======================================================================

    #[test]
    fn rope_scaling_none_when_absent() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn rope_scaling_linear_parsed() {
        let mut meta = llama_metadata();
        meta.insert("llama.rope.scaling.type".into(), GgufValue::String("linear".into()));
        meta.insert("llama.rope.scaling.factor".into(), GgufValue::F32(2.0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let scaling = cfg.rope_scaling.unwrap();
        assert_eq!(scaling.scaling_type, RopeScalingType::Linear);
        assert!((scaling.factor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn rope_scaling_yarn_parsed() {
        let mut meta = llama_metadata();
        meta.insert("llama.rope.scaling.type".into(), GgufValue::String("yarn".into()));
        meta.insert("llama.rope.scaling.factor".into(), GgufValue::F32(4.0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let scaling = cfg.rope_scaling.unwrap();
        assert_eq!(scaling.scaling_type, RopeScalingType::YaRn);
        assert!((scaling.factor - 4.0).abs() < 1e-6);
    }

    #[test]
    fn rope_scaling_ntk_parsed() {
        let mut meta = llama_metadata();
        meta.insert("llama.rope.scaling.type".into(), GgufValue::String("ntk".into()));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let scaling = cfg.rope_scaling.unwrap();
        assert_eq!(scaling.scaling_type, RopeScalingType::Ntk);
        assert!((scaling.factor - 1.0).abs() < 1e-6); // default factor
    }

    // =======================================================================
    // Quantization config
    // =======================================================================

    #[test]
    fn quantization_defaults() {
        let cfg = GgufModelConfig::from_gguf_metadata(&HashMap::new()).unwrap();
        assert_eq!(cfg.quantization.bit_width, 2);
        assert_eq!(cfg.quantization.block_size, 64);
        assert_eq!(cfg.quantization.format, "I2_S");
    }

    #[test]
    fn quantization_custom_values() {
        let meta = HashMap::from([
            ("llama.quantization.bit_width".into(), GgufValue::U32(4)),
            ("llama.quantization.block_size".into(), GgufValue::U32(256)),
            ("llama.quantization.format".into(), GgufValue::String("QK256".into())),
        ]);
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        assert_eq!(cfg.quantization.bit_width, 4);
        assert_eq!(cfg.quantization.block_size, 256);
        assert_eq!(cfg.quantization.format, "QK256");
    }

    // =======================================================================
    // Serialization round-trip
    // =======================================================================

    #[test]
    fn serde_json_roundtrip() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: GgufModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
    }

    #[test]
    fn serde_json_roundtrip_with_rope_scaling() {
        let mut meta = llama_metadata();
        meta.insert("llama.rope.scaling.type".into(), GgufValue::String("linear".into()));
        meta.insert("llama.rope.scaling.factor".into(), GgufValue::F32(2.0));
        let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let deserialized: GgufModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
    }

    #[test]
    fn serde_json_roundtrip_gqa() {
        let cfg = GgufModelConfig::from_gguf_metadata(&gqa_metadata()).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let deserialized: GgufModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, deserialized);
        assert!(deserialized.is_gqa());
    }

    #[test]
    fn memory_estimate_serializable() {
        let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
        let est = cfg.memory_estimate();
        let json = serde_json::to_string(&est).unwrap();
        let deserialized: MemoryEstimate = serde_json::from_str(&json).unwrap();
        assert_eq!(est, deserialized);
    }
}
