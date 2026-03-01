//! Integration tests for InferenceConfigBuilder with architecture-aware defaults.
//!
//! Tests that the config builder presets interact correctly with architecture
//! registry defaults, ensuring multi-SLM model configs are properly validated.

use bitnet_common::config::ModelConfig;
use bitnet_common::{ActivationType, NormType};
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};

// --- Builder preset + architecture integration ---

#[test]
fn phi4_config_with_balanced_preset() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("phi-4");

    let inference = InferenceConfigBuilder::new()
        .preset(InferencePreset::Balanced)
        .max_tokens(64)
        .build()
        .unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(model.activation_type, ActivationType::Silu);
    assert_eq!(inference.sampling.temperature, 0.7);
    assert_eq!(inference.generation.max_tokens, 64);
}

#[test]
fn llama_config_with_fast_preset() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("llama");

    let inference = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(model.activation_type, ActivationType::Silu);
    assert_eq!(inference.sampling.temperature, 0.0); // greedy
    assert_eq!(inference.sampling.top_k, 1);
    assert_eq!(inference.hardware.num_threads, 1);
}

#[test]
fn gemma_config_with_quality_preset() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("gemma");

    let inference = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .temperature(0.95)
        .build()
        .unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(model.activation_type, ActivationType::Gelu);
    assert_eq!(inference.sampling.temperature, 0.95);
    assert_eq!(inference.generation.max_tokens, 256);
}

#[test]
fn bitnet_config_with_deterministic_preset() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("bitnet");

    let inference =
        InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();

    assert_eq!(model.norm_type, NormType::LayerNorm);
    assert_eq!(model.activation_type, ActivationType::Silu);
    assert_eq!(inference.sampling.temperature, 0.0);
    assert_eq!(inference.sampling.seed, Some(42));
}

#[test]
fn mistral_config_with_debug_preset() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("mistral");

    let inference = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(model.activation_type, ActivationType::Silu);
    assert_eq!(inference.generation.max_tokens, 8);
    assert_eq!(inference.hardware.memory_limit_mb, 256);
}

// --- Builder override chaining ---

#[test]
fn preset_then_override_preserves_overrides() {
    let config = InferenceConfigBuilder::new()
        .preset(InferencePreset::Fast)
        .temperature(0.5)
        .top_k(10)
        .max_tokens(32)
        .build()
        .unwrap();

    // Overrides should take effect
    assert_eq!(config.sampling.temperature, 0.5);
    assert_eq!(config.sampling.top_k, 10);
    assert_eq!(config.generation.max_tokens, 32);
    // Non-overridden fields should still be from Fast preset
    assert_eq!(config.hardware.num_threads, 1);
}

#[test]
fn multiple_presets_last_wins() {
    let config = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .preset(InferencePreset::Fast)
        .build()
        .unwrap();

    // Fast preset should win
    assert_eq!(config.sampling.temperature, 0.0);
    assert_eq!(config.generation.max_tokens, 64);
}

#[test]
fn stop_sequences_accumulate() {
    let config = InferenceConfigBuilder::new()
        .stop_sequence("<|end|>")
        .stop_sequence("<|im_end|>")
        .stop_token_id(100265)
        .stop_token_id(100257)
        .build()
        .unwrap();

    assert_eq!(config.generation.stop_sequences.len(), 2);
    assert_eq!(config.generation.stop_token_ids.len(), 2);
    assert!(config.generation.stop_sequences.contains(&"<|end|>".to_string()));
    assert!(config.generation.stop_token_ids.contains(&100265));
}

#[test]
fn stop_sequences_replaced_by_setter() {
    let config = InferenceConfigBuilder::new()
        .stop_sequence("a")
        .stop_sequence("b")
        .stop_sequences(vec!["c".to_string()])
        .build()
        .unwrap();

    assert_eq!(config.generation.stop_sequences, vec!["c"]);
}

// --- Validation edge cases ---

#[test]
fn negative_temperature_rejected() {
    let result = InferenceConfigBuilder::new().temperature(-0.1).build();
    assert!(result.is_err());
}

#[test]
fn zero_top_p_rejected() {
    let result = InferenceConfigBuilder::new().top_p(0.0).build();
    assert!(result.is_err());
}

#[test]
fn top_p_above_one_rejected() {
    let result = InferenceConfigBuilder::new().top_p(1.1).build();
    assert!(result.is_err());
}

#[test]
fn zero_repetition_penalty_rejected() {
    let result = InferenceConfigBuilder::new().repetition_penalty(0.0).build();
    assert!(result.is_err());
}

#[test]
fn zero_max_tokens_rejected() {
    let result = InferenceConfigBuilder::new().max_tokens(0).build();
    assert!(result.is_err());
}

#[test]
fn excessive_threads_rejected() {
    let result = InferenceConfigBuilder::new().num_threads(2000).build();
    assert!(result.is_err());
}

#[test]
fn valid_boundary_values() {
    // Exact boundary values should be accepted
    assert!(InferenceConfigBuilder::new().temperature(0.0).build().is_ok());
    assert!(InferenceConfigBuilder::new().top_p(1.0).build().is_ok());
    assert!(InferenceConfigBuilder::new().top_p(0.01).build().is_ok());
    assert!(InferenceConfigBuilder::new().repetition_penalty(0.01).build().is_ok());
    assert!(InferenceConfigBuilder::new().max_tokens(1).build().is_ok());
    assert!(InferenceConfigBuilder::new().num_threads(1024).build().is_ok());
}

// --- JSON serialization ---

#[test]
fn config_roundtrip_json() {
    let config = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .seed(123)
        .stop_sequence("<|end|>")
        .build()
        .unwrap();

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: bitnet_inference::config_builder::InferenceConfig =
        serde_json::from_str(&json).unwrap();

    assert_eq!(config, deserialized);
}

#[test]
fn all_presets_produce_valid_configs() {
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];

    for preset in presets {
        let result = InferenceConfigBuilder::new().preset(preset).build();
        assert!(result.is_ok(), "Preset {preset:?} should produce a valid config");
    }
}

// --- Multi-SLM config scenarios ---

#[test]
fn phi4_16k_streaming_config() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("phi-4");
    model.num_layers = 40;
    model.num_heads = 40;
    model.num_key_value_heads = 10;
    model.hidden_size = 5120;
    model.max_position_embeddings = 16384;

    let inference = InferenceConfigBuilder::new()
        .preset(InferencePreset::Balanced)
        .max_tokens(512)
        .stream(true)
        .stop_token_id(100265) // Phi-4 EOS
        .build()
        .unwrap();

    assert!(inference.generation.stream);
    assert_eq!(inference.generation.max_tokens, 512);
    assert!(inference.generation.stop_token_ids.contains(&100265));
    assert_eq!(model.max_position_embeddings, 16384);
}

#[test]
fn qwen_config_scenario() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("qwen2");

    let inference = InferenceConfigBuilder::new()
        .temperature(0.6)
        .top_k(20)
        .top_p(0.8)
        .max_tokens(256)
        .stop_sequence("<|im_end|>")
        .build()
        .unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(model.activation_type, ActivationType::Silu);
    assert_eq!(inference.sampling.temperature, 0.6);
}

#[test]
fn deepseek_config_scenario() {
    let mut model = ModelConfig::default();
    model.apply_architecture_defaults("deepseek");

    let inference = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .repetition_penalty(1.15)
        .build()
        .unwrap();

    assert_eq!(model.norm_type, NormType::RmsNorm);
    assert_eq!(inference.sampling.repetition_penalty, 1.15);
}
