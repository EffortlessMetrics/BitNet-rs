//! Integration tests for inference configuration serialization.
//!
//! Validates serde round-trips, preset correctness, config validation,
//! and builder ergonomics across `bitnet-inference` config types.

use bitnet_inference::config::GenerationConfig;
use bitnet_inference::config::InferenceConfig;
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};

// ── InferenceConfig serde round-trip ─────────────────────────────────

#[test]
fn integration_inference_config_json_round_trip() {
    let original = InferenceConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: InferenceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(original.max_context_length, restored.max_context_length);
    assert_eq!(original.batch_size, restored.batch_size);
    assert_eq!(original.mixed_precision, restored.mixed_precision);
    assert_eq!(original.memory_pool_size, restored.memory_pool_size);
}

#[test]
fn integration_inference_config_pretty_json_round_trip() {
    let original =
        InferenceConfig::default().with_threads(4).with_batch_size(8).with_mixed_precision(true);
    let json = serde_json::to_string_pretty(&original).unwrap();
    let restored: InferenceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(original.num_threads, restored.num_threads);
    assert_eq!(original.batch_size, restored.batch_size);
    assert!(restored.mixed_precision);
}

// ── GenerationConfig serde round-trip ────────────────────────────────

#[test]
fn integration_generation_config_json_round_trip() {
    let original = GenerationConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(original.max_new_tokens, restored.max_new_tokens);
    assert_eq!(original.temperature, restored.temperature);
    assert_eq!(original.top_k, restored.top_k);
    assert_eq!(original.top_p, restored.top_p);
    assert_eq!(original.repetition_penalty, restored.repetition_penalty);
    assert_eq!(original.seed, restored.seed);
    assert_eq!(original.skip_special_tokens, restored.skip_special_tokens);
    assert_eq!(original.add_bos, restored.add_bos);
}

#[test]
fn integration_generation_config_with_stop_sequences_round_trip() {
    let original = GenerationConfig::greedy()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\nQ:".to_string())
        .with_max_tokens(64);
    let json = serde_json::to_string(&original).unwrap();
    let restored: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(original.stop_sequences, restored.stop_sequences);
    assert_eq!(original.max_new_tokens, restored.max_new_tokens);
}

#[test]
fn integration_generation_config_skipped_fields_reset() {
    // logits_cb and stop_token_ids_set are #[serde(skip)]
    let original = GenerationConfig::default().with_stop_token_ids(vec![128009, 128001]);
    assert!(original.is_stop_token(128009));

    let json = serde_json::to_string(&original).unwrap();
    let restored: GenerationConfig = serde_json::from_str(&json).unwrap();
    // Internal HashSet is skipped; vec is preserved
    assert_eq!(original.stop_token_ids, restored.stop_token_ids);
}

// ── Preset factory methods ───────────────────────────────────────────

#[test]
fn integration_greedy_preset_is_deterministic() {
    let cfg = GenerationConfig::greedy();
    assert_eq!(cfg.temperature, 0.0);
    assert_eq!(cfg.top_k, 1);
    assert_eq!(cfg.top_p, 1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn integration_creative_preset_has_high_temperature() {
    let cfg = GenerationConfig::creative();
    assert!(cfg.temperature > 0.5);
    assert!(cfg.top_k > 50);
    assert!(cfg.repetition_penalty > 1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn integration_balanced_preset_values() {
    let cfg = GenerationConfig::balanced();
    assert_eq!(cfg.temperature, 0.7);
    assert_eq!(cfg.top_k, 50);
    assert_eq!(cfg.top_p, 0.9);
    assert!(cfg.validate().is_ok());
}

// ── Config validation catches mistakes ───────────────────────────────

#[test]
fn integration_validation_rejects_zero_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    let err = cfg.validate().unwrap_err();
    assert!(
        err.contains("max_new_tokens") || err.contains("max_tokens"),
        "error should mention max_tokens: {err}"
    );
}

#[test]
fn integration_validation_rejects_negative_temperature() {
    let cfg = GenerationConfig::default().with_temperature(-1.0);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("temperature"), "error should mention temperature: {err}");
}

#[test]
fn integration_validation_rejects_top_p_zero() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("top_p"), "error should mention top_p: {err}");
}

#[test]
fn integration_validation_rejects_top_p_above_one() {
    let cfg = GenerationConfig::default().with_top_p(1.01);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("top_p"), "error: {err}");
}

#[test]
fn integration_validation_rejects_zero_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("repetition_penalty"), "error: {err}");
}

// ── Builder produces expected defaults ───────────────────────────────

#[test]
fn integration_builder_default_is_balanced_preset() {
    let cfg = InferenceConfigBuilder::new().build().unwrap();
    assert_eq!(cfg.sampling.temperature, 0.7);
    assert_eq!(cfg.sampling.top_k, 50);
    assert_eq!(cfg.generation.max_tokens, 128);
}

#[test]
fn integration_builder_preset_overrides_all_fields() {
    let cfg = InferenceConfigBuilder::new()
        .temperature(0.99)
        .preset(InferencePreset::Deterministic)
        .build()
        .unwrap();
    // Preset resets temperature
    assert_eq!(cfg.sampling.temperature, 0.0);
    assert_eq!(cfg.sampling.seed, Some(42));
}

#[test]
fn integration_builder_chaining_preserves_overrides() {
    let cfg = InferenceConfigBuilder::new()
        .preset(InferencePreset::Fast)
        .temperature(0.5)
        .max_tokens(200)
        .seed(99)
        .build()
        .unwrap();
    assert_eq!(cfg.sampling.temperature, 0.5);
    assert_eq!(cfg.generation.max_tokens, 200);
    assert_eq!(cfg.sampling.seed, Some(99));
    // Hardware from Fast preset
    assert_eq!(cfg.hardware.num_threads, 1);
}

#[test]
fn integration_builder_all_presets_validate() {
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];
    for preset in presets {
        let result = InferenceConfigBuilder::new().preset(preset).build();
        assert!(result.is_ok(), "preset {preset:?} must produce valid config");
    }
}

// ── Builder config serde round-trip ──────────────────────────────────

#[test]
fn integration_builder_config_serde_round_trip() {
    let original = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .seed(77)
        .max_tokens(512)
        .stop_sequence("</s>")
        .stop_token_id(128009)
        .stream(true)
        .num_threads(4)
        .memory_limit_mb(1024)
        .build()
        .unwrap();
    let json = serde_json::to_string(&original).unwrap();
    let restored: bitnet_inference::config_builder::InferenceConfig =
        serde_json::from_str(&json).unwrap();
    assert_eq!(original, restored);
}

#[test]
fn integration_builder_validation_rejects_negative_temp() {
    let err = InferenceConfigBuilder::new().temperature(-0.1).build().unwrap_err();
    assert!(err.contains("temperature"), "error: {err}");
}

#[test]
fn integration_builder_validation_rejects_excess_threads() {
    let err = InferenceConfigBuilder::new().num_threads(2000).build().unwrap_err();
    assert!(err.contains("num_threads"), "error: {err}");
}

#[test]
fn integration_preset_enum_serde_round_trip() {
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];
    for preset in presets {
        let json = serde_json::to_string(&preset).unwrap();
        let restored: InferencePreset = serde_json::from_str(&json).unwrap();
        assert_eq!(preset, restored, "round-trip failed for {preset:?}");
    }
}
