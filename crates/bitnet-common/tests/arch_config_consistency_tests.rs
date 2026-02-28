//! Tests verifying consistency between ArchitectureRegistry and ModelConfig.
//!
//! These tests ensure that `apply_architecture_defaults()` correctly wires
//! arch_registry lookups into ModelConfig, and that all known architectures
//! produce valid configurations for the dense model pipeline.

use bitnet_common::arch_registry::ArchitectureRegistry;
use bitnet_common::config::{ActivationType, ModelConfig, NormType};

// ---------------------------------------------------------------------------
// apply_architecture_defaults
// ---------------------------------------------------------------------------

/// Applying Phi-4 defaults should set RMSNorm + SiLU + 16K context.
#[test]
fn apply_defaults_phi4() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");

    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.max_position_embeddings, 16384);
}

/// Applying LLaMA defaults should set RMSNorm + SiLU, leave context at default.
#[test]
fn apply_defaults_llama() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("llama");

    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
    // llama has no specific context length, so default stays
    assert_eq!(config.max_position_embeddings, 2048);
}

/// Gemma uses RMSNorm + GELU.
#[test]
fn apply_defaults_gemma() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("gemma");

    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Gelu);
}

/// GPT/BERT use LayerNorm + GELU.
#[test]
fn apply_defaults_gpt() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("gpt");

    assert_eq!(config.norm_type, NormType::LayerNorm);
    assert_eq!(config.activation_type, ActivationType::Gelu);
}

/// Qwen2.5 should get RMSNorm + SiLU + 32K context.
#[test]
fn apply_defaults_qwen25() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("qwen2.5");

    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.max_position_embeddings, 32768);
}

/// Unknown architectures should not change the config.
#[test]
fn apply_defaults_unknown_is_noop() {
    let before = ModelConfig::default();
    let mut after = ModelConfig::default();
    after.apply_architecture_defaults("completely_unknown_model");

    assert_eq!(before.norm_type, after.norm_type);
    assert_eq!(before.activation_type, after.activation_type);
    assert_eq!(before.max_position_embeddings, after.max_position_embeddings);
}

/// Case-insensitive: "PHI-4" should match "phi-4".
#[test]
fn apply_defaults_case_insensitive() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("PHI-4");

    assert_eq!(config.norm_type, NormType::RmsNorm);
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.max_position_embeddings, 16384);
}

// ---------------------------------------------------------------------------
// Exhaustive: every known architecture produces a valid config
// ---------------------------------------------------------------------------

/// Every known architecture in the registry should produce a ModelConfig
/// with norm_type and activation_type set (not default LayerNorm/Silu combo
/// unless that's what the architecture actually uses).
#[test]
fn all_known_architectures_produce_valid_config() {
    for arch in ArchitectureRegistry::known_architectures() {
        let defaults = ArchitectureRegistry::lookup(arch)
            .unwrap_or_else(|| panic!("known arch '{}' has no defaults", arch));

        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);

        assert_eq!(config.norm_type, defaults.norm_type, "norm_type mismatch for arch '{arch}'");
        assert_eq!(
            config.activation_type, defaults.activation_type,
            "activation_type mismatch for arch '{arch}'"
        );

        if let Some(ctx) = defaults.default_context_length {
            assert_eq!(
                config.max_position_embeddings, ctx,
                "context length mismatch for arch '{arch}'"
            );
        }
    }
}

/// Context length override: if config already has a non-default context,
/// apply_architecture_defaults should NOT overwrite it.
#[test]
fn apply_defaults_preserves_custom_context() {
    let mut config = ModelConfig { max_position_embeddings: 131072, ..Default::default() };
    config.apply_architecture_defaults("phi-4"); // Phi-4 has 16K default

    // Should NOT downgrade to 16K since user explicitly set 128K
    assert_eq!(config.max_position_embeddings, 131072);
}

// ---------------------------------------------------------------------------
// Architecture family coverage
// ---------------------------------------------------------------------------

/// Verify all major model families have at least one entry.
#[test]
fn major_families_covered() {
    let families = [
        "phi",
        "llama",
        "qwen",
        "gemma",
        "mistral",
        "deepseek",
        "falcon",
        "starcoder",
        "codellama",
        "command",
        "internlm",
        "yi",
        "baichuan",
        "chatglm",
        "mpt",
        "rwkv",
        "olmo",
        "bitnet",
        "gpt",
        "mixtral",
        "bloom",
        "jamba",
        "arctic",
        "dbrx",
        "exaone",
        "minicpm",
    ];

    for family in &families {
        assert!(
            ArchitectureRegistry::is_known(family),
            "family '{}' not found in registry",
            family,
        );
    }
}

/// Dense model families should use either RMSNorm or LayerNorm (not Relu2).
#[test]
fn dense_families_use_standard_norms() {
    let dense_families = [
        "phi-4",
        "llama",
        "qwen2.5",
        "gemma2",
        "mistral",
        "deepseek",
        "falcon",
        "starcoder",
        "command",
        "gpt",
    ];

    for arch in &dense_families {
        let defaults = ArchitectureRegistry::lookup(arch)
            .unwrap_or_else(|| panic!("arch '{}' not found", arch));

        assert!(
            matches!(defaults.norm_type, NormType::RmsNorm | NormType::LayerNorm),
            "arch '{}' uses unexpected norm type: {:?}",
            arch,
            defaults.norm_type,
        );
    }
}

/// Dense model families should use SiLU or GELU (not ReLU2).
#[test]
fn dense_families_use_standard_activations() {
    let dense_families = [
        "phi-4",
        "llama",
        "qwen2.5",
        "gemma2",
        "mistral",
        "deepseek",
        "falcon",
        "starcoder",
        "command",
        "gpt",
    ];

    for arch in &dense_families {
        let defaults = ArchitectureRegistry::lookup(arch)
            .unwrap_or_else(|| panic!("arch '{}' not found", arch));

        assert!(
            matches!(defaults.activation_type, ActivationType::Silu | ActivationType::Gelu),
            "arch '{}' uses unexpected activation: {:?}",
            arch,
            defaults.activation_type,
        );
    }
}

// ---------------------------------------------------------------------------
// Context length sanity
// ---------------------------------------------------------------------------

/// No architecture should claim context > 1M tokens (sanity check).
#[test]
fn context_lengths_are_reasonable() {
    for arch in ArchitectureRegistry::known_architectures() {
        if let Some(defaults) = ArchitectureRegistry::lookup(arch)
            && let Some(ctx) = defaults.default_context_length
        {
            assert!(
                ctx <= 1_000_000,
                "arch '{}' claims unreasonable context length: {}",
                arch,
                ctx,
            );
            assert!(ctx >= 512, "arch '{}' has suspiciously small context: {}", arch, ctx,);
        }
    }
}
