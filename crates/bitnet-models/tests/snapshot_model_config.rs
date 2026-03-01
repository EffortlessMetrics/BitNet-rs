//! Wave 6 snapshot tests for bitnet-models configuration types.
//!
//! Pins ModelFormat Debug, GgufModelConfig defaults per architecture,
//! GgufQuantizationConfig defaults, and config JSON serialization.

use bitnet_models::config::{
    GgufModelConfig, GgufQuantizationConfig, RopeScaling, RopeScalingType,
};
use bitnet_models::formats::ModelFormat;
use bitnet_models::loader::LoadConfig;
use bitnet_models::production_loader::ProductionLoadConfig;
use std::collections::HashMap;

// ── ModelFormat Debug / name / extension ─────────────────────────────────────

#[test]
fn snapshot_model_format_debug_safetensors() {
    insta::assert_snapshot!(
        "model_format_debug_safetensors",
        format!("{:?}", ModelFormat::SafeTensors)
    );
}

#[test]
fn snapshot_model_format_debug_gguf() {
    insta::assert_snapshot!("model_format_debug_gguf", format!("{:?}", ModelFormat::Gguf));
}

#[test]
fn snapshot_model_format_names() {
    insta::assert_snapshot!(
        "model_format_names",
        format!(
            "SafeTensors={} Gguf={}",
            ModelFormat::SafeTensors.name(),
            ModelFormat::Gguf.name(),
        )
    );
}

#[test]
fn snapshot_model_format_extensions() {
    insta::assert_snapshot!(
        "model_format_extensions",
        format!(
            "SafeTensors={} Gguf={}",
            ModelFormat::SafeTensors.extension(),
            ModelFormat::Gguf.extension(),
        )
    );
}

#[test]
fn snapshot_model_format_serialization() {
    let formats = vec![ModelFormat::SafeTensors, ModelFormat::Gguf];
    insta::assert_json_snapshot!("model_format_json", formats);
}

// ── GgufQuantizationConfig defaults ─────────────────────────────────────────

#[test]
fn snapshot_gguf_quantization_config_default() {
    insta::assert_debug_snapshot!("gguf_quant_config_default", GgufQuantizationConfig::default());
}

#[test]
fn snapshot_gguf_quantization_config_json() {
    insta::assert_json_snapshot!("gguf_quant_config_json", GgufQuantizationConfig::default());
}

// ── Architecture-specific GgufModelConfig from metadata ─────────────────────

fn make_metadata(
    arch: &str,
    hidden: u32,
    layers: u32,
    heads: u32,
) -> HashMap<String, bitnet_models::formats::gguf::GgufValue> {
    use bitnet_models::formats::gguf::GgufValue;
    let mut m = HashMap::new();
    m.insert("general.architecture".into(), GgufValue::String(arch.into()));
    m.insert(format!("{arch}.embedding_length"), GgufValue::U32(hidden));
    m.insert(format!("{arch}.block_count"), GgufValue::U32(layers));
    m.insert(format!("{arch}.attention.head_count"), GgufValue::U32(heads));
    m.insert(format!("{arch}.attention.head_count_kv"), GgufValue::U32(heads));
    m.insert(format!("{arch}.feed_forward_length"), GgufValue::U32(hidden * 4));
    m.insert(format!("{arch}.context_length"), GgufValue::U32(2048));
    m.insert(format!("{arch}.vocab_size"), GgufValue::U32(32000));
    m.insert(format!("{arch}.rope.freq_base"), GgufValue::F32(10000.0));
    m
}

#[test]
fn snapshot_gguf_config_arch_llama() {
    let meta = make_metadata("llama", 4096, 32, 32);
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    insta::assert_json_snapshot!("gguf_config_arch_llama", config);
}

#[test]
fn snapshot_gguf_config_arch_bitnet() {
    let meta = make_metadata("bitnet", 2048, 24, 16);
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    insta::assert_json_snapshot!("gguf_config_arch_bitnet", config);
}

#[test]
fn snapshot_gguf_config_arch_mistral() {
    let meta = make_metadata("mistral", 4096, 32, 32);
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    insta::assert_json_snapshot!("gguf_config_arch_mistral", config);
}

#[test]
fn snapshot_gguf_config_arch_gpt() {
    let meta = make_metadata("gpt2", 768, 12, 12);
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    insta::assert_json_snapshot!("gguf_config_arch_gpt2", config);
}

// ── Config serialization round-trip ─────────────────────────────────────────

#[test]
fn snapshot_gguf_config_json_roundtrip() {
    let meta = make_metadata("llama", 4096, 32, 32);
    let config = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let json = serde_json::to_string_pretty(&config).unwrap();
    let deserialized: GgufModelConfig = serde_json::from_str(&json).unwrap();
    insta::assert_json_snapshot!("gguf_config_json_roundtrip", deserialized);
}

// ── RopeScaling serialization ───────────────────────────────────────────────

#[test]
fn snapshot_rope_scaling_types() {
    let types = vec![
        RopeScalingType::None,
        RopeScalingType::Linear,
        RopeScalingType::Ntk,
        RopeScalingType::YaRn,
    ];
    insta::assert_json_snapshot!("rope_scaling_types", types);
}

#[test]
fn snapshot_rope_scaling_config() {
    let scaling = RopeScaling { scaling_type: RopeScalingType::Linear, factor: 2.0 };
    insta::assert_json_snapshot!("rope_scaling_config_linear", scaling);
}

// ── LoadConfig / ProductionLoadConfig defaults ──────────────────────────────

#[test]
fn snapshot_load_config_default_debug() {
    insta::assert_debug_snapshot!("load_config_default_debug", LoadConfig::default());
}

#[test]
fn snapshot_production_load_config_default_debug() {
    insta::assert_debug_snapshot!(
        "production_load_config_default_debug",
        ProductionLoadConfig::default()
    );
}
