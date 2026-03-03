//! Snapshot wave 32 — model config debug output, GGUF header parsing,
//! tensor type descriptions, validation types, and I2S flavor formatting.

use bitnet_models::config::{
    GgufModelConfig, GgufQuantizationConfig, RopeScaling, RopeScalingType,
};
use bitnet_models::formats::gguf::{GgufHeader, GgufTensorType, I2SFlavor, I2SLayoutKind};
use bitnet_models::validator::{OverallStatus, Severity, ValidationCheck};

// ── GgufQuantizationConfig defaults ─────────────────────────────────

#[test]
fn gguf_quantization_config_default_debug() {
    let cfg = GgufQuantizationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gguf_quantization_config_default_yaml() {
    let cfg = GgufQuantizationConfig::default();
    insta::assert_yaml_snapshot!(cfg);
}

// ── RopeScaling types ───────────────────────────────────────────────

#[test]
fn rope_scaling_type_debug_all() {
    let types = vec![
        RopeScalingType::None,
        RopeScalingType::Linear,
        RopeScalingType::Ntk,
        RopeScalingType::YaRn,
    ];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn rope_scaling_linear_debug() {
    let scaling = RopeScaling { scaling_type: RopeScalingType::Linear, factor: 2.0 };
    insta::assert_debug_snapshot!(scaling);
}

// ── GgufModelConfig ─────────────────────────────────────────────────

#[test]
fn gguf_model_config_bitnet_2b_debug() {
    let cfg = GgufModelConfig {
        architecture: "llama".to_string(),
        model_name: Some("bitnet-b1.58-2B-4T".to_string()),
        vocab_size: 32000,
        hidden_size: 2048,
        num_layers: 24,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        intermediate_size: 5632,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rope_scaling: None,
        quantization: GgufQuantizationConfig::default(),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn gguf_model_config_with_rope_scaling_yaml() {
    let cfg = GgufModelConfig {
        architecture: "llama".to_string(),
        model_name: Some("test-model".to_string()),
        vocab_size: 128256,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 14336,
        max_seq_len: 8192,
        rope_theta: 500000.0,
        rope_scaling: Some(RopeScaling { scaling_type: RopeScalingType::YaRn, factor: 4.0 }),
        quantization: GgufQuantizationConfig {
            bit_width: 2,
            block_size: 256,
            format: "QK256".to_string(),
        },
    };
    insta::assert_yaml_snapshot!(cfg);
}

// ── GgufHeader ──────────────────────────────────────────────────────

#[test]
fn gguf_header_v3_standard_debug() {
    let header = GgufHeader {
        magic: *b"GGUF",
        version: 3,
        tensor_count: 291,
        metadata_kv_count: 26,
        alignment: 32,
        data_offset: 4096,
    };
    insta::assert_debug_snapshot!(header);
}

#[test]
fn gguf_header_v2_legacy_format_description() {
    let header = GgufHeader {
        magic: *b"GGUF",
        version: 2,
        tensor_count: 100,
        metadata_kv_count: 10,
        alignment: 32,
        data_offset: 0,
    };
    insta::assert_snapshot!(header.format_description());
}

#[test]
fn gguf_header_v3_standard_format_description() {
    let header = GgufHeader {
        magic: *b"GGUF",
        version: 3,
        tensor_count: 291,
        metadata_kv_count: 26,
        alignment: 32,
        data_offset: 4096,
    };
    insta::assert_snapshot!(header.format_description());
}

#[test]
fn gguf_header_v3_early_variant_format_description() {
    let header = GgufHeader {
        magic: *b"GGUF",
        version: 3,
        tensor_count: 200,
        metadata_kv_count: 15,
        alignment: 32,
        data_offset: 0,
    };
    insta::assert_snapshot!(header.format_description());
}

// ── GgufTensorType ──────────────────────────────────────────────────

#[test]
fn gguf_tensor_type_debug_all() {
    let types = vec![
        GgufTensorType::F32,
        GgufTensorType::F16,
        GgufTensorType::F64,
        GgufTensorType::Q4_0,
        GgufTensorType::Q4_1,
        GgufTensorType::Q8_0,
        GgufTensorType::I2_S,
        GgufTensorType::IQ2_S,
    ];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn gguf_tensor_type_from_quant_strings() {
    let pairs: Vec<(&str, Option<GgufTensorType>)> = vec![
        ("i2_s", GgufTensorType::from_quant_string("i2_s")),
        ("iq2_s", GgufTensorType::from_quant_string("iq2_s")),
        ("q4_0", GgufTensorType::from_quant_string("q4_0")),
        ("f16", GgufTensorType::from_quant_string("f16")),
        ("unknown", GgufTensorType::from_quant_string("unknown")),
    ];
    insta::assert_debug_snapshot!(pairs);
}

// ── I2S types ───────────────────────────────────────────────────────

#[test]
fn i2s_layout_kind_debug() {
    let kinds = vec![I2SLayoutKind::GgmlSplit, I2SLayoutKind::InlineF16];
    insta::assert_debug_snapshot!(kinds);
}

#[test]
fn i2s_flavor_debug_all() {
    let flavors =
        vec![I2SFlavor::BitNet32F16, I2SFlavor::Split32WithSibling, I2SFlavor::GgmlQk256NoScale];
    insta::assert_debug_snapshot!(flavors);
}

#[test]
fn i2s_flavor_block_sizes() {
    let output: Vec<String> =
        [I2SFlavor::BitNet32F16, I2SFlavor::Split32WithSibling, I2SFlavor::GgmlQk256NoScale]
            .iter()
            .map(|f| {
                format!(
                    "{:?}: block_size={}, data_bytes={}, total_bytes={}",
                    f,
                    f.block_size(),
                    f.data_bytes_per_block(),
                    f.total_bytes_per_block()
                )
            })
            .collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── Validation types ────────────────────────────────────────────────

#[test]
fn severity_display_all() {
    let output: Vec<String> = [Severity::Info, Severity::Warning, Severity::Error]
        .iter()
        .map(|s| s.to_string())
        .collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn validation_check_display_all() {
    let checks = [
        ValidationCheck::TensorShapes,
        ValidationCheck::WeightDistribution,
        ValidationCheck::LayerNormStats,
        ValidationCheck::VocabSize,
        ValidationCheck::EmbeddingDim,
        ValidationCheck::ArchitectureMatch,
        ValidationCheck::QuantizationFormat,
    ];
    let output: Vec<String> = checks.iter().map(|c| c.to_string()).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn overall_status_display_all() {
    let statuses =
        [OverallStatus::Passed, OverallStatus::PassedWithWarnings, OverallStatus::Failed];
    let output: Vec<String> = statuses.iter().map(|s| s.to_string()).collect();
    insta::assert_snapshot!(output.join("\n"));
}
