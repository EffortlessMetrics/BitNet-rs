//! Weight format detection and ModelFormat configuration tests.
//!
//! Validates that model format parsing, detection, and configuration
//! work correctly for all supported formats (GGUF, SafeTensors, HuggingFace).

use bitnet_common::{BitNetConfig, ConfigBuilder, ModelConfig, ModelFormat};

// ============================================================
// ModelFormat enum tests
// ============================================================

#[test]
fn test_model_format_default_is_gguf() {
    let format = ModelFormat::default();
    assert_eq!(format, ModelFormat::Gguf);
}

#[test]
fn test_model_format_variants_exist() {
    let _gguf = ModelFormat::Gguf;
    let _safetensors = ModelFormat::SafeTensors;
    let _huggingface = ModelFormat::HuggingFace;
}

#[test]
fn test_model_format_equality() {
    assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
    assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::HuggingFace, ModelFormat::HuggingFace);
    assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
    assert_ne!(ModelFormat::SafeTensors, ModelFormat::HuggingFace);
    assert_ne!(ModelFormat::Gguf, ModelFormat::HuggingFace);
}

#[test]
fn test_model_format_clone() {
    let original = ModelFormat::SafeTensors;
    let cloned = original;
    assert_eq!(original, cloned);
}

#[test]
fn test_model_format_debug() {
    let debug_str = format!("{:?}", ModelFormat::Gguf);
    assert!(debug_str.contains("Gguf"));
    let debug_str = format!("{:?}", ModelFormat::SafeTensors);
    assert!(debug_str.contains("SafeTensors"));
    let debug_str = format!("{:?}", ModelFormat::HuggingFace);
    assert!(debug_str.contains("HuggingFace"));
}

// ============================================================
// ModelConfig format field tests
// ============================================================

#[test]
fn test_model_config_default_format() {
    let config = ModelConfig::default();
    assert_eq!(config.format, ModelFormat::Gguf);
}

#[test]
fn test_model_config_set_safetensors() {
    let config = ModelConfig { format: ModelFormat::SafeTensors, ..Default::default() };
    assert_eq!(config.format, ModelFormat::SafeTensors);
}

#[test]
fn test_model_config_set_huggingface() {
    let config = ModelConfig { format: ModelFormat::HuggingFace, ..Default::default() };
    assert_eq!(config.format, ModelFormat::HuggingFace);
}

// ============================================================
// ConfigBuilder format tests
// ============================================================

#[test]
fn test_config_builder_default_format() {
    let config = ConfigBuilder::new().build().unwrap();
    assert_eq!(config.model.format, ModelFormat::Gguf);
}

#[test]
fn test_config_builder_set_safetensors() {
    let config = ConfigBuilder::new().model_format(ModelFormat::SafeTensors).build().unwrap();
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
}

#[test]
fn test_config_builder_set_huggingface() {
    let config = ConfigBuilder::new().model_format(ModelFormat::HuggingFace).build().unwrap();
    assert_eq!(config.model.format, ModelFormat::HuggingFace);
}

#[test]
fn test_config_builder_format_with_model_path() {
    let config = ConfigBuilder::new()
        .model_path("model.safetensors")
        .model_format(ModelFormat::SafeTensors)
        .build()
        .unwrap();
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
    assert!(config.model.path.is_some());
}

// ============================================================
// Environment variable format override tests
// ============================================================

// ============================================================
// Format-specific configuration patterns
// ============================================================

#[test]
fn test_gguf_format_typical_config() {
    // GGUF models: quantized, smaller vocab, BitNet architecture
    let config = ModelConfig {
        format: ModelFormat::Gguf,
        vocab_size: 32000,
        hidden_size: 2560,
        num_layers: 30,
        num_heads: 20,
        num_key_value_heads: 5,
        ..Default::default()
    };
    assert_eq!(config.format, ModelFormat::Gguf);
    assert_eq!(config.vocab_size, 32000);
}

#[test]
fn test_safetensors_format_typical_config() {
    // SafeTensors: Phi-4 style, large vocab, dense weights
    let config = ModelConfig {
        format: ModelFormat::SafeTensors,
        vocab_size: 100352,
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_key_value_heads: 10,
        ..Default::default()
    };
    assert_eq!(config.format, ModelFormat::SafeTensors);
    assert_eq!(config.vocab_size, 100352);
    assert_eq!(config.hidden_size, 5120);
}

#[test]
fn test_huggingface_format_typical_config() {
    // HuggingFace: auto-download, any architecture
    let config = ModelConfig {
        format: ModelFormat::HuggingFace,
        vocab_size: 151936,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_key_value_heads: 8,
        ..Default::default()
    };
    assert_eq!(config.format, ModelFormat::HuggingFace);
    assert_eq!(config.vocab_size, 151936);
}

// ============================================================
// Format serialization tests
// ============================================================

#[test]
fn test_model_format_serialize_json() {
    let gguf_json = serde_json::to_string(&ModelFormat::Gguf).unwrap();
    assert!(gguf_json.contains("Gguf"));

    let st_json = serde_json::to_string(&ModelFormat::SafeTensors).unwrap();
    assert!(st_json.contains("SafeTensors"));

    let hf_json = serde_json::to_string(&ModelFormat::HuggingFace).unwrap();
    assert!(hf_json.contains("HuggingFace"));
}

#[test]
fn test_model_format_deserialize_json() {
    let gguf: ModelFormat = serde_json::from_str("\"Gguf\"").unwrap();
    assert_eq!(gguf, ModelFormat::Gguf);

    let st: ModelFormat = serde_json::from_str("\"SafeTensors\"").unwrap();
    assert_eq!(st, ModelFormat::SafeTensors);

    let hf: ModelFormat = serde_json::from_str("\"HuggingFace\"").unwrap();
    assert_eq!(hf, ModelFormat::HuggingFace);
}

#[test]
fn test_model_format_roundtrip_json() {
    for format in [ModelFormat::Gguf, ModelFormat::SafeTensors, ModelFormat::HuggingFace] {
        let json = serde_json::to_string(&format).unwrap();
        let deserialized: ModelFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, deserialized);
    }
}

// ============================================================
// Format with architecture defaults interaction
// ============================================================

#[test]
fn test_format_preserved_after_architecture_defaults() {
    use bitnet_common::{ActivationType, NormType};

    // SafeTensors format should be preserved when applying architecture defaults
    let mut config = ModelConfig { format: ModelFormat::SafeTensors, ..Default::default() };
    // Simulate what apply_architecture_defaults does for a phi-4 model
    config.norm_type = NormType::RmsNorm;
    config.activation_type = ActivationType::Silu;
    config.max_position_embeddings = 16384;

    assert_eq!(config.format, ModelFormat::SafeTensors);
    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
}

#[test]
fn test_gguf_format_with_bitnet_defaults() {
    use bitnet_common::{ActivationType, NormType};

    let config = ModelConfig {
        format: ModelFormat::Gguf,
        norm_type: NormType::LayerNorm,
        activation_type: ActivationType::Relu2,
        vocab_size: 32000,
        hidden_size: 2560,
        ..Default::default()
    };

    assert_eq!(config.format, ModelFormat::Gguf);
    assert_eq!(config.norm_type, NormType::LayerNorm);
    assert_eq!(config.activation_type, ActivationType::Relu2);
}

// ============================================================
// Edge cases
// ============================================================

#[test]
fn test_format_change_after_creation() {
    let mut config = ModelConfig::default();
    assert_eq!(config.format, ModelFormat::Gguf);

    config.format = ModelFormat::SafeTensors;
    assert_eq!(config.format, ModelFormat::SafeTensors);

    config.format = ModelFormat::HuggingFace;
    assert_eq!(config.format, ModelFormat::HuggingFace);
}

#[test]
fn test_multiple_configs_different_formats() {
    let gguf_config = ModelConfig { format: ModelFormat::Gguf, ..Default::default() };
    let st_config = ModelConfig { format: ModelFormat::SafeTensors, ..Default::default() };
    let hf_config = ModelConfig { format: ModelFormat::HuggingFace, ..Default::default() };

    assert_ne!(gguf_config.format, st_config.format);
    assert_ne!(st_config.format, hf_config.format);
    assert_ne!(gguf_config.format, hf_config.format);
}

#[test]
fn test_config_builder_format_override_order() {
    // Last format set should win
    let config = ConfigBuilder::new()
        .model_format(ModelFormat::Gguf)
        .model_format(ModelFormat::SafeTensors)
        .model_format(ModelFormat::HuggingFace)
        .build()
        .unwrap();
    assert_eq!(config.model.format, ModelFormat::HuggingFace);
}

#[test]
fn test_format_in_full_bitnet_config() {
    let config = BitNetConfig {
        model: ModelConfig {
            format: ModelFormat::SafeTensors,
            vocab_size: 100352,
            hidden_size: 5120,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(config.model.format, ModelFormat::SafeTensors);
    assert_eq!(config.model.vocab_size, 100352);
}
