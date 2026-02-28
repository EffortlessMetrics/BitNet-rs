//! Integration tests validating end-to-end architecture defaults flow.
//!
//! These tests verify that ModelConfig correctly picks up architecture
//! defaults from the registry for all known model families.

use bitnet_common::{ActivationType, ArchitectureRegistry, ModelConfig, NormType};

#[test]
fn test_phi4_config_defaults_match_spec() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");
    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.max_position_embeddings, 16384);
}

#[test]
fn test_llama_config_defaults_match_spec() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("llama");
    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
}

#[test]
fn test_gpt_config_defaults_match_spec() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("gpt");
    assert_eq!(config.norm_type, NormType::LayerNorm);
    assert_eq!(config.activation_type, ActivationType::Gelu);
}

#[test]
fn test_bitnet_config_defaults_match_spec() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("bitnet");
    assert_eq!(config.norm_type, NormType::LayerNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
}

#[test]
fn test_all_families_produce_consistent_config() {
    // All known architectures should set config to valid state
    for arch in ArchitectureRegistry::known_architectures() {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);

        assert!(
            matches!(config.norm_type, NormType::LayerNorm | NormType::RmsNorm),
            "Arch '{}' produced invalid norm_type: {:?}",
            arch,
            config.norm_type,
        );
        assert!(
            matches!(
                config.activation_type,
                ActivationType::Silu | ActivationType::Relu2 | ActivationType::Gelu
            ),
            "Arch '{}' produced invalid activation_type: {:?}",
            arch,
            config.activation_type,
        );
        assert!(
            config.max_position_embeddings >= 2048,
            "Arch '{}' has surprisingly small context: {}",
            arch,
            config.max_position_embeddings,
        );
    }
}

#[test]
fn test_template_and_tokenizer_coverage_alignment() {
    // Every architecture family that has a tokenizer entry should
    // also have an architecture registry entry. This is a coherence
    // check across the codebase subsystems.
    let known: Vec<&str> = ArchitectureRegistry::known_architectures().to_vec();

    // At minimum, these key families must be in the registry
    let expected_families = [
        "phi", "llama", "qwen", "gemma", "mistral", "deepseek", "starcoder", "falcon",
        "gpt", "bitnet",
    ];
    for family in &expected_families {
        assert!(
            known.iter().any(|k| k.to_lowercase().contains(family)),
            "Expected family '{}' not found in architecture registry",
            family,
        );
    }
}

#[test]
fn test_config_defaults_idempotent() {
    // Applying defaults twice should not change anything
    for arch in ArchitectureRegistry::known_architectures() {
        let mut config1 = ModelConfig::default();
        config1.apply_architecture_defaults(arch);
        let mut config2 = config1.clone();
        config2.apply_architecture_defaults(arch);
        assert_eq!(config1.norm_type, config2.norm_type);
        assert_eq!(config1.activation_type, config2.activation_type);
        assert_eq!(
            config1.max_position_embeddings,
            config2.max_position_embeddings
        );
    }
}

#[test]
fn test_unknown_architecture_preserves_defaults() {
    let config_before = ModelConfig::default();
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("unknown_model_xyz");
    assert_eq!(config.norm_type, config_before.norm_type);
    assert_eq!(config.activation_type, config_before.activation_type);
    assert_eq!(
        config.max_position_embeddings,
        config_before.max_position_embeddings
    );
}
