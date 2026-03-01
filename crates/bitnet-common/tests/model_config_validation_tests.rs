//! Model configuration validation tests for known SLM architectures.
//!
//! Creates `BitNetConfig` instances matching real model specifications
//! (Phi-4, LLaMA 7B, Mistral 7B, etc.) and validates them through the
//! config validation pipeline.

use bitnet_common::QuantizationType;
use bitnet_common::config::{
    ActivationType, BitNetConfig, InferenceConfig, ModelConfig, ModelFormat, NormType,
    PerformanceConfig, QuantizationConfig,
};

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

fn phi4_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::SafeTensors,
        vocab_size: 100352,
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_key_value_heads: 10,
        intermediate_size: 14336,
        max_position_embeddings: 16384,
        rope_theta: Some(250000.0),
        rope_scaling: None,
        rms_norm_eps: Some(1e-5),
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Silu,
        tokenizer: Default::default(),
    }
}

fn llama7b_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_key_value_heads: 32, // MHA
        intermediate_size: 11008,
        max_position_embeddings: 4096,
        rope_theta: Some(10000.0),
        rope_scaling: None,
        rms_norm_eps: Some(1e-6),
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Silu,
        tokenizer: Default::default(),
    }
}

fn mistral7b_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        num_key_value_heads: 8, // GQA 4:1
        intermediate_size: 14336,
        max_position_embeddings: 32768,
        rope_theta: Some(10000.0),
        rope_scaling: None,
        rms_norm_eps: Some(1e-5),
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Silu,
        tokenizer: Default::default(),
    }
}

fn gemma2_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size: 256000,
        hidden_size: 3072,
        num_layers: 28,
        num_heads: 16,
        num_key_value_heads: 16,
        intermediate_size: 24576,
        max_position_embeddings: 8192,
        rope_theta: Some(10000.0),
        rope_scaling: None,
        rms_norm_eps: Some(1e-6),
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Gelu,
        tokenizer: Default::default(),
    }
}

fn qwen25_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::SafeTensors,
        vocab_size: 152064,
        hidden_size: 3584,
        num_layers: 28,
        num_heads: 28,
        num_key_value_heads: 4,
        intermediate_size: 18944,
        max_position_embeddings: 32768,
        rope_theta: Some(1000000.0),
        rope_scaling: None,
        rms_norm_eps: Some(1e-6),
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Silu,
        tokenizer: Default::default(),
    }
}

fn bitnet_2b_config() -> ModelConfig {
    ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size: 32000,
        hidden_size: 2560,
        num_layers: 30,
        num_heads: 20,
        num_key_value_heads: 5,
        intermediate_size: 6912,
        max_position_embeddings: 4096,
        rope_theta: None,
        rope_scaling: None,
        rms_norm_eps: None,
        norm_type: NormType::LayerNorm,
        activation_type: ActivationType::Silu,
        tokenizer: Default::default(),
    }
}

fn make_config(model: ModelConfig) -> BitNetConfig {
    BitNetConfig {
        model,
        inference: InferenceConfig::default(),
        quantization: QuantizationConfig::default(),
        performance: PerformanceConfig::default(),
    }
}

// ────────────────────────────────────────────────────────────────
// 1. Valid model configs pass validation
// ────────────────────────────────────────────────────────────────

#[test]
fn phi4_config_validates() {
    let cfg = make_config(phi4_config());
    assert!(cfg.validate().is_ok(), "Phi-4 config should be valid");
}

#[test]
fn llama7b_config_validates() {
    let cfg = make_config(llama7b_config());
    assert!(cfg.validate().is_ok(), "LLaMA 7B config should be valid");
}

#[test]
fn mistral7b_config_validates() {
    let cfg = make_config(mistral7b_config());
    assert!(cfg.validate().is_ok(), "Mistral 7B config should be valid");
}

#[test]
fn gemma2_config_validates() {
    let cfg = make_config(gemma2_config());
    assert!(cfg.validate().is_ok(), "Gemma 2 config should be valid");
}

#[test]
fn qwen25_config_validates() {
    let cfg = make_config(qwen25_config());
    assert!(cfg.validate().is_ok(), "Qwen 2.5 config should be valid");
}

#[test]
fn bitnet_2b_config_validates() {
    let cfg = make_config(bitnet_2b_config());
    assert!(cfg.validate().is_ok(), "BitNet 2B config should be valid");
}

// ────────────────────────────────────────────────────────────────
// 2. Head dimension calculations
// ────────────────────────────────────────────────────────────────

#[test]
fn phi4_head_dim_is_128() {
    let cfg = phi4_config();
    assert_eq!(cfg.hidden_size / cfg.num_heads, 128);
}

#[test]
fn llama7b_head_dim_is_128() {
    let cfg = llama7b_config();
    assert_eq!(cfg.hidden_size / cfg.num_heads, 128);
}

#[test]
fn mistral7b_head_dim_is_128() {
    let cfg = mistral7b_config();
    assert_eq!(cfg.hidden_size / cfg.num_heads, 128);
}

#[test]
fn gemma2_head_dim_is_192() {
    let cfg = gemma2_config();
    assert_eq!(cfg.hidden_size / cfg.num_heads, 192);
}

// ────────────────────────────────────────────────────────────────
// 3. GQA ratio calculations
// ────────────────────────────────────────────────────────────────

#[test]
fn phi4_gqa_ratio_is_4() {
    let cfg = phi4_config();
    assert_eq!(cfg.num_heads / cfg.num_key_value_heads, 4);
}

#[test]
fn mistral7b_gqa_ratio_is_4() {
    let cfg = mistral7b_config();
    assert_eq!(cfg.num_heads / cfg.num_key_value_heads, 4);
}

#[test]
fn qwen25_gqa_ratio_is_7() {
    let cfg = qwen25_config();
    assert_eq!(cfg.num_heads / cfg.num_key_value_heads, 7);
}

#[test]
fn llama7b_is_mha() {
    let cfg = llama7b_config();
    assert_eq!(cfg.num_heads, cfg.num_key_value_heads, "LLaMA 7B uses MHA");
}

#[test]
fn bitnet_2b_gqa_ratio_is_4() {
    let cfg = bitnet_2b_config();
    assert_eq!(cfg.num_heads / cfg.num_key_value_heads, 4);
}

// ────────────────────────────────────────────────────────────────
// 4. Validation catches invalid configs
// ────────────────────────────────────────────────────────────────

#[test]
fn zero_vocab_size_fails_validation() {
    let mut cfg = make_config(phi4_config());
    cfg.model.vocab_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_hidden_size_fails_validation() {
    let mut cfg = make_config(phi4_config());
    cfg.model.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_num_layers_fails_validation() {
    let mut cfg = make_config(phi4_config());
    cfg.model.num_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_num_heads_fails_validation() {
    let mut cfg = make_config(phi4_config());
    cfg.model.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn hidden_not_divisible_by_heads_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.model.hidden_size = 5121; // not divisible by 40
    assert!(cfg.validate().is_err());
}

#[test]
fn kv_heads_greater_than_heads_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.model.num_key_value_heads = 50; // > 40 heads
    assert!(cfg.validate().is_err());
}

#[test]
fn heads_not_divisible_by_kv_heads_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.model.num_key_value_heads = 7; // 40 not divisible by 7
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_intermediate_size_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.model.intermediate_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_max_position_embeddings_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.model.max_position_embeddings = 0;
    assert!(cfg.validate().is_err());
}

// ────────────────────────────────────────────────────────────────
// 5. Inference config validation
// ────────────────────────────────────────────────────────────────

#[test]
fn zero_temperature_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.inference.temperature = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn negative_temperature_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.inference.temperature = -0.5;
    assert!(cfg.validate().is_err());
}

#[test]
fn top_p_out_of_range_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.inference.top_p = Some(1.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_repetition_penalty_fails() {
    let mut cfg = make_config(phi4_config());
    cfg.inference.repetition_penalty = 0.0;
    assert!(cfg.validate().is_err());
}

// ────────────────────────────────────────────────────────────────
// 6. Apply architecture defaults integration
// ────────────────────────────────────────────────────────────────

#[test]
fn apply_defaults_then_validate_phi4() {
    let mut cfg = BitNetConfig::default();
    cfg.model.apply_architecture_defaults("phi-4");
    assert!(cfg.validate().is_ok(), "Phi-4 defaults should produce valid config");
    assert_eq!(cfg.model.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.model.activation_type, ActivationType::Silu);
    assert_eq!(cfg.model.max_position_embeddings, 16384);
}

#[test]
fn apply_defaults_then_validate_llama() {
    let mut cfg = BitNetConfig::default();
    cfg.model.apply_architecture_defaults("llama");
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.model.norm_type, NormType::RmsNorm);
}

#[test]
fn apply_defaults_then_validate_gpt2() {
    let mut cfg = BitNetConfig::default();
    cfg.model.apply_architecture_defaults("gpt");
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.model.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.model.activation_type, ActivationType::Gelu);
}

// ────────────────────────────────────────────────────────────────
// 7. Model format variants
// ────────────────────────────────────────────────────────────────

#[test]
fn gguf_format_validates() {
    let cfg = make_config(llama7b_config());
    assert_eq!(cfg.model.format, ModelFormat::Gguf);
    assert!(cfg.validate().is_ok());
}

#[test]
fn safetensors_format_validates() {
    let cfg = make_config(phi4_config());
    assert_eq!(cfg.model.format, ModelFormat::SafeTensors);
    assert!(cfg.validate().is_ok());
}

// ────────────────────────────────────────────────────────────────
// 8. Config merge behavior
// ────────────────────────────────────────────────────────────────

#[test]
fn merge_overrides_non_default_values() {
    let mut base = BitNetConfig::default();
    let mut override_cfg = BitNetConfig::default();
    override_cfg.model.vocab_size = 100352;
    override_cfg.model.hidden_size = 5120;
    override_cfg.model.num_layers = 40;
    override_cfg.model.num_heads = 40;
    override_cfg.model.num_key_value_heads = 10;
    override_cfg.model.intermediate_size = 14336;
    override_cfg.model.max_position_embeddings = 16384;

    base.merge_with(override_cfg);

    assert_eq!(base.model.vocab_size, 100352);
    assert_eq!(base.model.hidden_size, 5120);
    assert_eq!(base.model.num_layers, 40);
    assert_eq!(base.model.num_heads, 40);
    assert_eq!(base.model.num_key_value_heads, 10);
    assert_eq!(base.model.max_position_embeddings, 16384);
}

#[test]
fn merge_preserves_defaults_when_other_is_default() {
    let mut base = make_config(phi4_config());
    let default_override = BitNetConfig::default();
    let original_vocab = base.model.vocab_size;

    base.merge_with(default_override);

    // Default values should not override existing non-default values
    assert_eq!(base.model.vocab_size, original_vocab);
}

// ────────────────────────────────────────────────────────────────
// 9. Default config is valid
// ────────────────────────────────────────────────────────────────

#[test]
fn default_config_validates() {
    let cfg = BitNetConfig::default();
    assert!(cfg.validate().is_ok(), "Default config should be valid");
}

// ────────────────────────────────────────────────────────────────
// 10. Quantization config
// ────────────────────────────────────────────────────────────────

#[test]
fn default_quantization_uses_i2s() {
    let cfg = QuantizationConfig::default();
    assert_eq!(cfg.quantization_type, QuantizationType::I2S);
    assert_eq!(cfg.block_size, 64);
}

#[test]
fn non_power_of_two_block_size_fails() {
    let mut cfg = BitNetConfig::default();
    cfg.quantization.block_size = 65; // not a power of 2
    assert!(cfg.validate().is_err());
}
