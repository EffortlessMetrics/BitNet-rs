//! Edge-case tests for `bitnet-models` config module:
//! GgufModelConfig, validation, memory estimation, GQA, and RoPE scaling.

use bitnet_models::config::{
    ConfigError, GgufModelConfig, GgufQuantizationConfig, RopeScalingType,
};
use bitnet_models::formats::gguf::GgufValue;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn minimal_metadata(arch: &str) -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".to_string(), GgufValue::String(arch.to_string())),
        (format!("{arch}.vocab_size"), GgufValue::U32(32000)),
        (format!("{arch}.embedding_length"), GgufValue::U32(4096)),
        (format!("{arch}.block_count"), GgufValue::U32(32)),
        (format!("{arch}.attention.head_count"), GgufValue::U32(32)),
        (format!("{arch}.attention.head_count_kv"), GgufValue::U32(32)),
        (format!("{arch}.feed_forward_length"), GgufValue::U32(11008)),
        (format!("{arch}.context_length"), GgufValue::U32(2048)),
    ])
}

fn gqa_metadata() -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".to_string(), GgufValue::String("llama".to_string())),
        ("llama.vocab_size".to_string(), GgufValue::U32(32000)),
        ("llama.embedding_length".to_string(), GgufValue::U32(4096)),
        ("llama.block_count".to_string(), GgufValue::U32(32)),
        ("llama.attention.head_count".to_string(), GgufValue::U32(32)),
        ("llama.attention.head_count_kv".to_string(), GgufValue::U32(8)),
        ("llama.feed_forward_length".to_string(), GgufValue::U32(11008)),
        ("llama.context_length".to_string(), GgufValue::U32(4096)),
    ])
}

// ---------------------------------------------------------------------------
// from_gguf_metadata: basic parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_minimal_llama() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "llama");
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
fn parse_empty_metadata_uses_defaults() {
    let meta = HashMap::new();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "llama"); // default
    assert_eq!(cfg.vocab_size, 32000); // default
    assert_eq!(cfg.hidden_size, 4096); // default
    assert_eq!(cfg.num_layers, 32); // default
    assert_eq!(cfg.max_seq_len, 2048); // default
}

#[test]
fn parse_model_name() {
    let mut meta = minimal_metadata("llama");
    meta.insert("general.name".to_string(), GgufValue::String("TestModel".to_string()));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.model_name.as_deref(), Some("TestModel"));
}

#[test]
fn parse_no_model_name() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.model_name.is_none());
}

// ---------------------------------------------------------------------------
// Architecture variants
// ---------------------------------------------------------------------------

#[test]
fn parse_bitnet_architecture() {
    let meta = minimal_metadata("bitnet");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "bitnet");
}

#[test]
fn parse_phi_architecture() {
    let meta = minimal_metadata("phi");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "phi");
}

// ---------------------------------------------------------------------------
// GQA detection
// ---------------------------------------------------------------------------

#[test]
fn gqa_detected() {
    let meta = gqa_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 4); // 32 / 8
}

#[test]
fn mha_not_gqa() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(!cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 1);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn valid_config_passes_validation() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_ok());
}

#[test]
fn validation_catches_zero_vocab() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.vocab_size".to_string(), GgufValue::U32(0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_catches_zero_hidden_size() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.embedding_length".to_string(), GgufValue::U32(0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_catches_zero_layers() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.block_count".to_string(), GgufValue::U32(0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_catches_kv_heads_gt_heads() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(8));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(16));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_catches_non_divisible_heads() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.attention.head_count".to_string(), GgufValue::U32(7));
    meta.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(3));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn valid_gqa_config_passes() {
    let meta = gqa_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

#[test]
fn memory_estimate_non_zero() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let est = cfg.memory_estimate();
    assert!(est.weight_bytes > 0);
    assert!(est.kv_cache_bytes > 0);
    assert!(est.total_bytes > 0);
    assert_eq!(est.total_bytes, est.weight_bytes + est.kv_cache_bytes);
}

#[test]
fn memory_estimate_summary_contains_mib() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let est = cfg.memory_estimate();
    assert!(est.summary.contains("MiB"));
}

#[test]
fn memory_estimate_larger_model_more_memory() {
    let small = minimal_metadata("llama");
    let mut large = minimal_metadata("llama");
    large.insert("llama.block_count".to_string(), GgufValue::U32(64));
    large.insert("llama.embedding_length".to_string(), GgufValue::U32(8192));
    large.insert("llama.attention.head_count".to_string(), GgufValue::U32(64));
    large.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(64));
    large.insert("llama.feed_forward_length".to_string(), GgufValue::U32(22016));

    let small_cfg = GgufModelConfig::from_gguf_metadata(&small).unwrap();
    let large_cfg = GgufModelConfig::from_gguf_metadata(&large).unwrap();
    assert!(large_cfg.memory_estimate().total_bytes > small_cfg.memory_estimate().total_bytes);
}

#[test]
fn memory_estimate_gqa_less_kv_cache() {
    let mut mha = minimal_metadata("llama");
    mha.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(32));
    let mut gqa = minimal_metadata("llama");
    gqa.insert("llama.attention.head_count_kv".to_string(), GgufValue::U32(8));

    let mha_cfg = GgufModelConfig::from_gguf_metadata(&mha).unwrap();
    let gqa_cfg = GgufModelConfig::from_gguf_metadata(&gqa).unwrap();
    assert!(gqa_cfg.memory_estimate().kv_cache_bytes < mha_cfg.memory_estimate().kv_cache_bytes);
}

// ---------------------------------------------------------------------------
// RoPE theta
// ---------------------------------------------------------------------------

#[test]
fn default_rope_theta() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!((cfg.rope_theta - 10_000.0).abs() < 1.0);
}

#[test]
fn custom_rope_theta() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.freq_base".to_string(), GgufValue::F32(500_000.0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!((cfg.rope_theta - 500_000.0).abs() < 1.0);
}

// ---------------------------------------------------------------------------
// RoPE scaling
// ---------------------------------------------------------------------------

#[test]
fn no_rope_scaling_by_default() {
    let meta = minimal_metadata("llama");
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.rope_scaling.is_none());
}

#[test]
fn linear_rope_scaling() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.scaling.type".to_string(), GgufValue::String("linear".to_string()));
    meta.insert("llama.rope.scaling.factor".to_string(), GgufValue::F32(2.0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let scaling = cfg.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, RopeScalingType::Linear);
    assert!((scaling.factor - 2.0).abs() < f32::EPSILON);
}

#[test]
fn ntk_rope_scaling() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.scaling.type".to_string(), GgufValue::String("ntk".to_string()));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let scaling = cfg.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, RopeScalingType::Ntk);
}

#[test]
fn yarn_rope_scaling() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.scaling.type".to_string(), GgufValue::String("yarn".to_string()));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let scaling = cfg.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, RopeScalingType::YaRn);
}

#[test]
fn unknown_rope_scaling_ignored() {
    let mut meta = minimal_metadata("llama");
    meta.insert(
        "llama.rope.scaling.type".to_string(),
        GgufValue::String("unknown_type".to_string()),
    );
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.rope_scaling.is_none());
}

#[test]
fn none_rope_scaling_ignored() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.scaling.type".to_string(), GgufValue::String("none".to_string()));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.rope_scaling.is_none());
}

// ---------------------------------------------------------------------------
// Quantization config defaults
// ---------------------------------------------------------------------------

#[test]
fn default_quantization_config() {
    let qc = GgufQuantizationConfig::default();
    assert_eq!(qc.bit_width, 2);
    assert_eq!(qc.block_size, 64);
    assert_eq!(qc.format, "I2_S");
}

#[test]
fn custom_quantization_from_metadata() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.quantization.bit_width".to_string(), GgufValue::U32(4));
    meta.insert("llama.quantization.block_size".to_string(), GgufValue::U32(256));
    meta.insert("llama.quantization.format".to_string(), GgufValue::String("QK256".to_string()));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.quantization.bit_width, 4);
    assert_eq!(cfg.quantization.block_size, 256);
    assert_eq!(cfg.quantization.format, "QK256");
}

// ---------------------------------------------------------------------------
// Serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn config_serde_roundtrip() {
    let meta = gqa_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GgufModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, cfg2);
}

#[test]
fn config_with_rope_scaling_serde_roundtrip() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.rope.scaling.type".to_string(), GgufValue::String("yarn".to_string()));
    meta.insert("llama.rope.scaling.factor".to_string(), GgufValue::F32(4.0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GgufModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, cfg2);
}

// ---------------------------------------------------------------------------
// ConfigError variants
// ---------------------------------------------------------------------------

#[test]
fn config_error_display() {
    let err = ConfigError::MissingKey("test_key".to_string());
    assert!(err.to_string().contains("test_key"));

    let err = ConfigError::InvalidValue { key: "k".to_string(), reason: "bad".to_string() };
    assert!(err.to_string().contains("bad"));

    let err = ConfigError::Validation("v=0".to_string());
    assert!(err.to_string().contains("v=0"));
}

// ---------------------------------------------------------------------------
// I32 to U32 coercion in metadata
// ---------------------------------------------------------------------------

#[test]
fn i32_positive_coerces_to_u32() {
    let mut meta = minimal_metadata("llama");
    meta.insert("llama.vocab_size".to_string(), GgufValue::I32(50000));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.vocab_size, 50000);
}
