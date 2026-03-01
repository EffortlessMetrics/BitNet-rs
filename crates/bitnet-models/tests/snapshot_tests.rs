//! Snapshot tests for stable bitnet-models configuration defaults.
//! These test key configuration constants that should never silently change.

use bitnet_models::{
    loader::LoadConfig,
    production_loader::{DeviceStrategy, ProductionLoadConfig},
};

#[test]
fn load_config_default_uses_mmap() {
    let cfg = LoadConfig::default();
    insta::assert_snapshot!(format!(
        "use_mmap={} validate_checksums={}",
        cfg.use_mmap, cfg.validate_checksums
    ));
}

#[test]
fn production_load_config_default_strict_validation() {
    let cfg = ProductionLoadConfig::default();
    insta::assert_snapshot!(format!(
        "strict_validation={} validate_tensor_alignment={}",
        cfg.strict_validation, cfg.validate_tensor_alignment
    ));
}

#[test]
fn production_load_config_default_max_model_size() {
    let cfg = ProductionLoadConfig::default();
    let size_gb = cfg.max_model_size_bytes.unwrap_or(0) / (1024 * 1024 * 1024);
    insta::assert_snapshot!(format!("max_size_gb={size_gb}"));
}

#[test]
fn device_strategy_cpu_only_debug() {
    insta::assert_snapshot!(format!("{:?}", DeviceStrategy::CpuOnly));
}

#[test]
fn device_strategy_hybrid_debug() {
    insta::assert_snapshot!(format!(
        "{:?}",
        DeviceStrategy::Hybrid { cpu_layers: 4, gpu_layers: 8 }
    ));
}

// == Wave 4: GGUF model config ===============================================

use bitnet_models::config::{
    ConfigError, GgufModelConfig, GgufQuantizationConfig, RopeScalingType,
};
use bitnet_models::formats::gguf::GgufValue;
use std::collections::HashMap;

fn llama_metadata() -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".into(), GgufValue::String("llama".into())),
        ("general.name".into(), GgufValue::String("TestLlama-7B".into())),
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

fn bitnet_gqa_metadata() -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".into(), GgufValue::String("bitnet".into())),
        ("general.name".into(), GgufValue::String("BitNet-2B".into())),
        ("bitnet.vocab_size".into(), GgufValue::U32(100000)),
        ("bitnet.embedding_length".into(), GgufValue::U32(2048)),
        ("bitnet.block_count".into(), GgufValue::U32(24)),
        ("bitnet.attention.head_count".into(), GgufValue::U32(16)),
        ("bitnet.attention.head_count_kv".into(), GgufValue::U32(4)),
        ("bitnet.feed_forward_length".into(), GgufValue::U32(5504)),
        ("bitnet.context_length".into(), GgufValue::U32(2048)),
        ("bitnet.rope.freq_base".into(), GgufValue::F32(500000.0)),
    ])
}

#[test]
fn gguf_model_config_llama_snapshot() {
    let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
    insta::assert_debug_snapshot!("gguf_model_config_llama", cfg);
}

#[test]
fn gguf_model_config_bitnet_gqa_snapshot() {
    let cfg = GgufModelConfig::from_gguf_metadata(&bitnet_gqa_metadata()).unwrap();
    insta::assert_debug_snapshot!("gguf_model_config_bitnet_gqa", cfg);
}

#[test]
fn gguf_model_config_validate_ok() {
    let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
    let result = cfg.validate();
    insta::assert_snapshot!("gguf_config_validate_ok", format!("{result:?}"));
}

#[test]
fn gguf_model_config_gqa_detection() {
    let cfg = GgufModelConfig::from_gguf_metadata(&bitnet_gqa_metadata()).unwrap();
    insta::assert_snapshot!(
        "gguf_config_gqa_detection",
        format!(
            "is_gqa={} gqa_group_size={} num_heads={} num_kv_heads={}",
            cfg.is_gqa(),
            cfg.gqa_group_size(),
            cfg.num_heads,
            cfg.num_kv_heads,
        )
    );
}

#[test]
fn gguf_model_config_memory_estimate_snapshot() {
    let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
    let est = cfg.memory_estimate();
    insta::assert_snapshot!(
        "gguf_config_memory_estimate",
        format!(
            "weight_bytes={} kv_cache_bytes={} total_bytes={}\nsummary: {}",
            est.weight_bytes, est.kv_cache_bytes, est.total_bytes, est.summary,
        )
    );
}

#[test]
fn gguf_quantization_config_default_snapshot() {
    let qcfg = GgufQuantizationConfig::default();
    insta::assert_debug_snapshot!("gguf_quantization_config_default", qcfg);
}

#[test]
fn rope_scaling_types_debug() {
    let types = [
        RopeScalingType::None,
        RopeScalingType::Linear,
        RopeScalingType::Ntk,
        RopeScalingType::YaRn,
    ];
    let debug: Vec<String> = types.iter().map(|t| format!("{t:?}")).collect();
    insta::assert_debug_snapshot!("rope_scaling_types", debug);
}

#[test]
fn config_error_variants_display() {
    let errors: Vec<ConfigError> = vec![
        ConfigError::MissingKey("llama.vocab_size".into()),
        ConfigError::InvalidValue {
            key: "llama.block_count".into(),
            reason: "expected u32, got string".into(),
        },
        ConfigError::Validation("hidden_size must be > 0".into()),
    ];
    let displays: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
    insta::assert_debug_snapshot!("config_error_variants", displays);
}

#[test]
fn gguf_model_config_defaults_from_empty_metadata() {
    let metadata = HashMap::new();
    let cfg = GgufModelConfig::from_gguf_metadata(&metadata).unwrap();
    insta::assert_snapshot!(
        "gguf_config_from_empty_metadata",
        format!(
            "arch={} vocab_size={} hidden_size={} num_layers={} num_heads={} head_dim={} max_seq_len={}",
            cfg.architecture,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_layers,
            cfg.num_heads,
            cfg.head_dim,
            cfg.max_seq_len,
        )
    );
}
