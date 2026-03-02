//! Auto-detection of model configuration from GGUF metadata or HuggingFace
//! `config.json`.
//!
//! Provides a unified pipeline that inspects a model file path and
//! automatically populates a [`ModelConfig`] from whichever metadata
//! source is available — GGUF key-value pairs *or* HuggingFace JSON.
//! Architecture-specific defaults are applied from the
//! [`architecture`](crate::architecture) registry when explicit metadata
//! is missing.

use std::collections::HashMap;
use std::path::Path;

use bitnet_common::config::{ActivationType, ModelConfig, ModelFormat, RopeScaling};
use thiserror::Error;

use crate::architecture::{detect_architecture, get_defaults};
use crate::formats::gguf::GgufValue;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during model configuration detection.
#[derive(Debug, Error)]
pub enum ConfigDetectionError {
    /// A required metadata field was missing and no default could be applied.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// The config.json content could not be parsed.
    #[error("invalid config JSON: {0}")]
    InvalidJson(String),

    /// An I/O error occurred while reading a config file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The model file extension is not recognised.
    #[error("unrecognised model file extension: {0}")]
    UnrecognisedExtension(String),

    /// Validation of the detected configuration failed.
    #[error("config validation failed: {0}")]
    Validation(String),
}

type Result<T> = std::result::Result<T, ConfigDetectionError>;

// ---------------------------------------------------------------------------
// GGUF metadata → ModelConfig
// ---------------------------------------------------------------------------

/// Detect model configuration from GGUF metadata key-value pairs.
///
/// Reads standard GGUF keys (`general.architecture`, `{arch}.context_length`,
/// `{arch}.embedding_length`, etc.) and falls back to architecture defaults
/// from the registry when a key is absent.
pub fn detect_from_gguf(metadata: &HashMap<String, GgufValue>) -> Result<ModelConfig> {
    let arch_str = gguf_string(metadata, "general.architecture")
        .unwrap_or_else(|| "llama".to_string());
    let arch = detect_architecture(&arch_str);
    let defaults = get_defaults(&arch);

    let mut cfg = ModelConfig { format: ModelFormat::Gguf, ..Default::default() };

    // Architecture-prefixed key helper
    let ak = |field: &str| -> String { format!("{arch_str}.{field}") };

    cfg.hidden_size = gguf_usize(metadata, &ak("embedding_length"))
        .unwrap_or(defaults.typical_hidden_size);

    cfg.num_layers = gguf_usize(metadata, &ak("block_count"))
        .unwrap_or(32);

    cfg.num_heads = gguf_usize(metadata, &ak("attention.head_count"))
        .unwrap_or(32);

    cfg.num_key_value_heads = gguf_usize(metadata, &ak("attention.head_count_kv"))
        .unwrap_or(cfg.num_heads);

    cfg.intermediate_size = gguf_usize(metadata, &ak("feed_forward_length"))
        .unwrap_or(cfg.hidden_size * 4);

    cfg.vocab_size = gguf_usize(metadata, &ak("vocab_size"))
        .or_else(|| gguf_usize(metadata, "tokenizer.ggml.vocab_size"))
        .unwrap_or(defaults.vocab_size);

    cfg.max_position_embeddings = gguf_usize(metadata, &ak("context_length"))
        .unwrap_or(defaults.max_context);

    cfg.rope_theta = gguf_f32(metadata, &ak("rope.freq_base"))
        .or(Some(defaults.rope_base));

    // Rope scaling (optional)
    if let Some(st) = gguf_string(metadata, &ak("rope.scaling.type")) {
        let factor = gguf_f32(metadata, &ak("rope.scaling.factor"))
            .unwrap_or(1.0);
        if st != "none" {
            cfg.rope_scaling = Some(RopeScaling { scaling_type: st, factor });
        }
    }

    // Norm / activation from registry
    cfg.norm_type = defaults.normalization;
    cfg.activation_type = defaults.activation;

    // RMS norm epsilon (optional)
    cfg.rms_norm_eps = gguf_f32(metadata, &ak("attention.layer_norm_rms_epsilon"));

    validate_config(&cfg)?;
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// HuggingFace config.json → ModelConfig
// ---------------------------------------------------------------------------

/// Detect model configuration from a HuggingFace `config.json` string.
///
/// Parses the standard HuggingFace transformer config fields
/// (`model_type`, `hidden_size`, `num_hidden_layers`, etc.) and
/// enriches missing values with architecture defaults.
pub fn detect_from_hf_config(config_json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| ConfigDetectionError::InvalidJson(e.to_string()))?;

    let model_type = raw
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let arch = detect_architecture(model_type);
    let defaults = get_defaults(&arch);

    let mut cfg = ModelConfig { format: ModelFormat::SafeTensors, ..Default::default() };

    cfg.hidden_size = json_usize(&raw, "hidden_size")
        .unwrap_or(defaults.typical_hidden_size);

    cfg.num_layers = json_usize(&raw, "num_hidden_layers")
        .unwrap_or(32);

    cfg.num_heads = json_usize(&raw, "num_attention_heads")
        .unwrap_or(32);

    cfg.num_key_value_heads = json_usize(&raw, "num_key_value_heads")
        .unwrap_or(cfg.num_heads);

    cfg.intermediate_size = json_usize(&raw, "intermediate_size")
        .unwrap_or(cfg.hidden_size * 4);

    cfg.vocab_size = json_usize(&raw, "vocab_size")
        .unwrap_or(defaults.vocab_size);

    cfg.max_position_embeddings = json_usize(&raw, "max_position_embeddings")
        .unwrap_or(defaults.max_context);

    cfg.rope_theta = json_f32(&raw, "rope_theta")
        .or(Some(defaults.rope_base));

    // Rope scaling (optional JSON object)
    if let Some(rs) = raw.get("rope_scaling").and_then(|v| v.as_object()) {
        let st = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .to_string();
        let factor = rs
            .get("factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        if st != "none" {
            cfg.rope_scaling = Some(RopeScaling { scaling_type: st, factor });
        }
    }

    // RMS norm epsilon
    cfg.rms_norm_eps = json_f32(&raw, "rms_norm_eps");

    // Norm / activation from registry
    cfg.norm_type = defaults.normalization;
    cfg.activation_type = defaults.activation;

    // Override activation if explicitly specified
    if let Some(act) = raw.get("hidden_act").and_then(|v| v.as_str()) {
        cfg.activation_type = match act.to_lowercase().as_str() {
            "silu" | "swish" => ActivationType::Silu,
            "gelu" | "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => {
                ActivationType::Gelu
            }
            "relu2" | "squared_relu" => ActivationType::Relu2,
            _ => defaults.activation,
        };
    }

    validate_config(&cfg)?;
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Auto-detection pipeline
// ---------------------------------------------------------------------------

/// Try to automatically detect model configuration from a file path.
///
/// The detection strategy depends on the file extension:
/// 1. **`.gguf`** — Opens the file and reads GGUF metadata (requires
///    the file to exist and be a valid GGUF).
/// 2. **`.safetensors`** — Looks for `config.json` in the same directory.
/// 3. **`.json`** — Treats the file itself as a HuggingFace `config.json`.
///
/// Architecture-specific defaults are applied automatically.
pub fn auto_detect_config(model_path: &Path) -> Result<ModelConfig> {
    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gguf" => {
            let metadata = read_gguf_metadata(model_path)?;
            detect_from_gguf(&metadata)
        }
        "safetensors" => {
            // Look for config.json in the same directory
            let dir = model_path.parent().unwrap_or(Path::new("."));
            let config_path = dir.join("config.json");
            if config_path.exists() {
                let json = std::fs::read_to_string(&config_path)?;
                detect_from_hf_config(&json)
            } else {
                Err(ConfigDetectionError::MissingField(
                    "config.json not found next to .safetensors file".into(),
                ))
            }
        }
        "json" => {
            let json = std::fs::read_to_string(model_path)?;
            detect_from_hf_config(&json)
        }
        other => Err(ConfigDetectionError::UnrecognisedExtension(
            other.to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata reader (minimal — just KV pairs)
// ---------------------------------------------------------------------------

/// Read GGUF metadata from a file on disk.
///
/// This is a lightweight read that only extracts the header key-value
/// pairs — it does not load tensor data.
fn read_gguf_metadata(
    path: &Path,
) -> std::result::Result<HashMap<String, GgufValue>, ConfigDetectionError> {
    use memmap2::Mmap;
    use std::fs::File;

    use crate::formats::gguf::GgufReader;

    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let reader = GgufReader::new(&mmap).map_err(|e| {
        ConfigDetectionError::Io(std::io::Error::other(
            e.to_string(),
        ))
    })?;

    // Build a HashMap from the reader's metadata keys and typed getters.
    let mut map = HashMap::new();
    for key in reader.metadata_keys() {
        // Try each typed getter in order of most-common to least-common.
        if let Some(v) = reader.get_string_metadata(key) {
            map.insert(key.to_string(), GgufValue::String(v));
        } else if let Some(v) = reader.get_u32_metadata(key) {
            map.insert(key.to_string(), GgufValue::U32(v));
        } else if let Some(v) = reader.get_f32_metadata(key) {
            map.insert(key.to_string(), GgufValue::F32(v));
        } else if let Some(v) = reader.get_i32_metadata(key) {
            map.insert(key.to_string(), GgufValue::I32(v));
        } else if let Some(v) = reader.get_bool_metadata(key) {
            map.insert(key.to_string(), GgufValue::Bool(v));
        }
        // Arrays and other types are not needed for config detection.
    }

    Ok(map)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_config(cfg: &ModelConfig) -> Result<()> {
    let mut errors = Vec::new();

    if cfg.vocab_size == 0 {
        errors.push("vocab_size must be > 0");
    }
    if cfg.hidden_size == 0 {
        errors.push("hidden_size must be > 0");
    }
    if cfg.num_layers == 0 {
        errors.push("num_layers must be > 0");
    }
    if cfg.num_heads == 0 {
        errors.push("num_heads must be > 0");
    }
    if cfg.max_position_embeddings == 0 {
        errors.push("max_position_embeddings must be > 0");
    }

    if cfg.num_heads > 0 && !cfg.hidden_size.is_multiple_of(cfg.num_heads) {
        errors.push("hidden_size must be divisible by num_heads");
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(ConfigDetectionError::Validation(errors.join("; ")))
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata extraction helpers
// ---------------------------------------------------------------------------

fn gguf_string(map: &HashMap<String, GgufValue>, key: &str) -> Option<String> {
    match map.get(key)? {
        GgufValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn gguf_usize(map: &HashMap<String, GgufValue>, key: &str) -> Option<usize> {
    match map.get(key)? {
        GgufValue::U32(v) => Some(*v as usize),
        GgufValue::I32(v) if *v >= 0 => Some(*v as usize),
        GgufValue::U16(v) => Some(*v as usize),
        GgufValue::I16(v) if *v >= 0 => Some(*v as usize),
        _ => None,
    }
}

fn gguf_f32(map: &HashMap<String, GgufValue>, key: &str) -> Option<f32> {
    match map.get(key)? {
        GgufValue::F32(v) => Some(*v),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JSON extraction helpers
// ---------------------------------------------------------------------------

fn json_usize(val: &serde_json::Value, key: &str) -> Option<usize> {
    val.get(key)?.as_u64().map(|v| v as usize)
}

fn json_f32(val: &serde_json::Value, key: &str) -> Option<f32> {
    val.get(key)?.as_f64().map(|v| v as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::config::NormType;

    // ===================================================================
    // HuggingFace config.json detection
    // ===================================================================

    fn phi4_config_json() -> &'static str {
        r#"{
            "model_type": "phi3",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 10,
            "intermediate_size": 17920,
            "vocab_size": 100352,
            "max_position_embeddings": 16384,
            "rope_theta": 250000.0,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu"
        }"#
    }

    fn llama3_config_json() -> &'static str {
        r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }"#
    }

    fn qwen25_config_json() -> &'static str {
        r#"{
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        }"#
    }

    #[test]
    fn hf_config_phi4() {
        let cfg = detect_from_hf_config(phi4_config_json()).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.num_heads, 40);
        assert_eq!(cfg.num_key_value_heads, 10);
        assert_eq!(cfg.intermediate_size, 17920);
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.max_position_embeddings, 16384);
        assert_eq!(cfg.rope_theta, Some(250000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-5));
        assert_eq!(cfg.activation_type, ActivationType::Silu);
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        assert_eq!(cfg.format, ModelFormat::SafeTensors);
    }

    #[test]
    fn hf_config_llama3() {
        let cfg = detect_from_hf_config(llama3_config_json()).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.rope_theta, Some(500000.0));
        assert_eq!(cfg.activation_type, ActivationType::Silu);
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        // Rope scaling
        let rs = cfg.rope_scaling.as_ref().unwrap();
        assert_eq!(rs.scaling_type, "linear");
        assert_eq!(rs.factor, 2.0);
    }

    #[test]
    fn hf_config_qwen25() {
        let cfg = detect_from_hf_config(qwen25_config_json()).unwrap();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.intermediate_size, 18944);
        assert_eq!(cfg.vocab_size, 152064);
        assert_eq!(cfg.max_position_embeddings, 131072);
        assert_eq!(cfg.rope_theta, Some(1000000.0));
        assert_eq!(cfg.rms_norm_eps, Some(1e-6));
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn hf_config_gemma2() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 24576,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "hidden_act": "gelu"
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.activation_type, ActivationType::Gelu);
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn hf_config_mistral() {
        let json = r#"{
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 32768
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        // No hidden_act → architecture default (Silu for mistral)
        assert_eq!(cfg.activation_type, ActivationType::Silu);
    }

    #[test]
    fn hf_config_bitnet() {
        let json = r#"{
            "model_type": "bitnet",
            "hidden_size": 2560,
            "num_hidden_layers": 26,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 6912,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "hidden_act": "relu2"
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.activation_type, ActivationType::Relu2);
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
    }

    // ===================================================================
    // GGUF metadata detection
    // ===================================================================

    fn llama_gguf_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            ("general.architecture".into(), GgufValue::String("llama".into())),
            ("llama.embedding_length".into(), GgufValue::U32(4096)),
            ("llama.block_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count".into(), GgufValue::U32(32)),
            ("llama.attention.head_count_kv".into(), GgufValue::U32(8)),
            ("llama.feed_forward_length".into(), GgufValue::U32(14336)),
            ("llama.vocab_size".into(), GgufValue::U32(128256)),
            ("llama.context_length".into(), GgufValue::U32(8192)),
            ("llama.rope.freq_base".into(), GgufValue::F32(500000.0)),
        ])
    }

    fn phi_gguf_metadata() -> HashMap<String, GgufValue> {
        HashMap::from([
            ("general.architecture".into(), GgufValue::String("phi3".into())),
            ("phi3.embedding_length".into(), GgufValue::U32(5120)),
            ("phi3.block_count".into(), GgufValue::U32(40)),
            ("phi3.attention.head_count".into(), GgufValue::U32(40)),
            ("phi3.attention.head_count_kv".into(), GgufValue::U32(10)),
            ("phi3.feed_forward_length".into(), GgufValue::U32(17920)),
            ("phi3.vocab_size".into(), GgufValue::U32(100352)),
            ("phi3.context_length".into(), GgufValue::U32(16384)),
            ("phi3.rope.freq_base".into(), GgufValue::F32(250000.0)),
        ])
    }

    #[test]
    fn gguf_detect_llama() {
        let cfg = detect_from_gguf(&llama_gguf_metadata()).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.rope_theta, Some(500000.0));
        assert_eq!(cfg.format, ModelFormat::Gguf);
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
        assert_eq!(cfg.activation_type, ActivationType::Silu);
    }

    #[test]
    fn gguf_detect_phi() {
        let cfg = detect_from_gguf(&phi_gguf_metadata()).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.num_heads, 40);
        assert_eq!(cfg.num_key_value_heads, 10);
        assert_eq!(cfg.intermediate_size, 17920);
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.max_position_embeddings, 16384);
        assert_eq!(cfg.rope_theta, Some(250000.0));
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn gguf_detect_bitnet() {
        let metadata = HashMap::from([
            ("general.architecture".into(), GgufValue::String("bitnet".into())),
            ("bitnet.embedding_length".into(), GgufValue::U32(2560)),
            ("bitnet.block_count".into(), GgufValue::U32(26)),
            ("bitnet.attention.head_count".into(), GgufValue::U32(32)),
            ("bitnet.attention.head_count_kv".into(), GgufValue::U32(32)),
            ("bitnet.feed_forward_length".into(), GgufValue::U32(6912)),
            ("bitnet.vocab_size".into(), GgufValue::U32(32000)),
            ("bitnet.context_length".into(), GgufValue::U32(4096)),
        ]);
        let cfg = detect_from_gguf(&metadata).unwrap();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 26);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.norm_type, NormType::LayerNorm);
        assert_eq!(cfg.activation_type, ActivationType::Relu2);
    }

    #[test]
    fn gguf_rope_scaling() {
        let mut metadata = llama_gguf_metadata();
        metadata.insert(
            "llama.rope.scaling.type".into(),
            GgufValue::String("yarn".into()),
        );
        metadata.insert(
            "llama.rope.scaling.factor".into(),
            GgufValue::F32(4.0),
        );
        let cfg = detect_from_gguf(&metadata).unwrap();
        let rs = cfg.rope_scaling.as_ref().unwrap();
        assert_eq!(rs.scaling_type, "yarn");
        assert_eq!(rs.factor, 4.0);
    }

    // ===================================================================
    // Missing fields → sensible defaults
    // ===================================================================

    #[test]
    fn gguf_missing_fields_uses_defaults() {
        // Only architecture set — everything else falls back to defaults
        let metadata = HashMap::from([
            ("general.architecture".into(), GgufValue::String("llama".into())),
        ]);
        let cfg = detect_from_gguf(&metadata).unwrap();
        // Should get llama defaults from the architecture registry
        assert_eq!(cfg.hidden_size, 4096); // llama typical_hidden_size
        assert_eq!(cfg.vocab_size, 128256); // llama default vocab
        assert_eq!(cfg.max_position_embeddings, 8192); // llama default ctx
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn hf_config_missing_fields_uses_defaults() {
        let json = r#"{ "model_type": "qwen2" }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.hidden_size, 3584); // qwen typical hidden
        assert_eq!(cfg.vocab_size, 152064); // qwen default vocab
        assert_eq!(cfg.max_position_embeddings, 131072); // qwen default ctx
    }

    #[test]
    fn hf_config_unknown_model_type() {
        let json = r#"{
            "model_type": "new_model_xyz",
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "vocab_size": 50000,
            "max_position_embeddings": 4096
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.vocab_size, 50000);
    }

    #[test]
    fn gguf_no_architecture_defaults_to_llama() {
        let metadata = HashMap::from([
            ("llama.embedding_length".into(), GgufValue::U32(2048)),
            ("llama.block_count".into(), GgufValue::U32(16)),
            ("llama.attention.head_count".into(), GgufValue::U32(16)),
            ("llama.feed_forward_length".into(), GgufValue::U32(5504)),
            ("llama.context_length".into(), GgufValue::U32(2048)),
        ]);
        let cfg = detect_from_gguf(&metadata).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_layers, 16);
    }

    // ===================================================================
    // Invalid config → error
    // ===================================================================

    #[test]
    fn hf_config_invalid_json() {
        let result = detect_from_hf_config("not valid json{{{");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConfigDetectionError::InvalidJson(_)));
    }

    #[test]
    fn hf_config_zero_hidden_size_fails_validation() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 0,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096
        }"#;
        let result = detect_from_hf_config(json);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigDetectionError::Validation(_)));
    }

    #[test]
    fn hf_config_misaligned_heads_fails_validation() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 33,
            "vocab_size": 32000,
            "max_position_embeddings": 4096
        }"#;
        let result = detect_from_hf_config(json);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigDetectionError::Validation(_)));
    }

    // ===================================================================
    // Auto-detection from file extension
    // ===================================================================

    #[test]
    fn auto_detect_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, phi4_config_json()).unwrap();

        let cfg = auto_detect_config(&config_path).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_layers, 40);
    }

    #[test]
    fn auto_detect_safetensors_with_config() {
        let dir = tempfile::tempdir().unwrap();
        // Write a config.json next to the safetensors file
        std::fs::write(dir.path().join("config.json"), qwen25_config_json()).unwrap();
        // Create a dummy safetensors file
        std::fs::write(dir.path().join("model.safetensors"), b"dummy").unwrap();

        let cfg = auto_detect_config(&dir.path().join("model.safetensors")).unwrap();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.vocab_size, 152064);
    }

    #[test]
    fn auto_detect_safetensors_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"dummy").unwrap();

        let result = auto_detect_config(&dir.path().join("model.safetensors"));
        assert!(result.is_err());
    }

    #[test]
    fn auto_detect_unknown_extension() {
        let result = auto_detect_config(Path::new("model.bin"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConfigDetectionError::UnrecognisedExtension(_)
        ));
    }

    // ===================================================================
    // Activation override in HF config
    // ===================================================================

    #[test]
    fn hf_config_gelu_activation_override() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "hidden_act": "gelu_new"
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        assert_eq!(cfg.activation_type, ActivationType::Gelu);
    }

    #[test]
    fn hf_config_rope_scaling_with_rope_type_key() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_scaling": {
                "rope_type": "dynamic",
                "factor": 8.0
            }
        }"#;
        let cfg = detect_from_hf_config(json).unwrap();
        let rs = cfg.rope_scaling.as_ref().unwrap();
        assert_eq!(rs.scaling_type, "dynamic");
        assert_eq!(rs.factor, 8.0);
    }
}
