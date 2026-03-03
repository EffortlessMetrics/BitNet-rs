//! Integration tests for architecture detection → config defaults pipeline.
//!
//! Verifies that ArchitectureRegistry lookups correctly configure ModelConfig
//! for all supported model families.

use bitnet_common::ArchitectureRegistry;
use bitnet_common::config::{ActivationType, ModelConfig, NormType};

// ---------------------------------------------------------------------------
// Architecture → Config defaults integration
// ---------------------------------------------------------------------------

#[test]
fn phi4_architecture_defaults_applied() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 16384);
}

#[test]
fn phi3_architecture_defaults_applied() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-3");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn phi2_uses_layernorm_gelu() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-2");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn llama_architecture_defaults() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("llama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    // No default context for generic llama, keeps 2048
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn llama32_gets_extended_context() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("llama-3.2");
    assert_eq!(cfg.max_position_embeddings, 131072);
}

#[test]
fn gemma_uses_rmsnorm_gelu() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("gemma");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn qwen25_gets_32k_context() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("qwen2.5");
    assert_eq!(cfg.max_position_embeddings, 32768);
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
}

#[test]
fn bitnet_uses_layernorm_silu() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("bitnet");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn deepseek_v3_gets_64k_context() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("deepseek-v3");
    assert_eq!(cfg.max_position_embeddings, 65536);
}

#[test]
fn unknown_arch_preserves_config_defaults() {
    let mut cfg = ModelConfig::default();
    let original_norm = cfg.norm_type;
    let original_ctx = cfg.max_position_embeddings;
    cfg.apply_architecture_defaults("unknown_model");
    assert_eq!(cfg.norm_type, original_norm);
    assert_eq!(cfg.max_position_embeddings, original_ctx);
}

#[test]
fn case_insensitive_architecture_defaults() {
    let mut cfg1 = ModelConfig::default();
    let mut cfg2 = ModelConfig::default();
    cfg1.apply_architecture_defaults("PHI-4");
    cfg2.apply_architecture_defaults("phi-4");
    assert_eq!(cfg1.norm_type, cfg2.norm_type);
    assert_eq!(cfg1.activation_type, cfg2.activation_type);
    assert_eq!(cfg1.max_position_embeddings, cfg2.max_position_embeddings);
}

// ---------------------------------------------------------------------------
// Architecture families — norm/activation consistency
// ---------------------------------------------------------------------------

#[test]
fn all_phi_variants_use_rmsnorm_silu_except_phi2() {
    for arch in ["phi", "phi-4", "phi-3"] {
        let d = ArchitectureRegistry::lookup(arch).unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm, "{arch} should use RmsNorm");
        assert_eq!(d.activation_type, ActivationType::Silu, "{arch} should use Silu");
    }
    let phi2 = ArchitectureRegistry::lookup("phi-2").unwrap();
    assert_eq!(phi2.norm_type, NormType::LayerNorm);
    assert_eq!(phi2.activation_type, ActivationType::Gelu);
}

#[test]
fn all_llama_variants_use_rmsnorm_silu() {
    for arch in ["llama", "llama2", "llama-3.1", "llama-3.2", "mistral"] {
        let d = ArchitectureRegistry::lookup(arch).unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm, "{arch} should use RmsNorm");
        assert_eq!(d.activation_type, ActivationType::Silu, "{arch} should use Silu");
    }
}

#[test]
fn all_gemma_variants_use_rmsnorm_gelu() {
    for arch in ["gemma", "gemma2", "codegemma"] {
        let d = ArchitectureRegistry::lookup(arch).unwrap();
        assert_eq!(d.norm_type, NormType::RmsNorm, "{arch} should use RmsNorm");
        assert_eq!(d.activation_type, ActivationType::Gelu, "{arch} should use Gelu");
    }
}

#[test]
fn context_lengths_increase_with_version() {
    let phi2_ctx = ArchitectureRegistry::lookup("phi-2").unwrap().default_context_length.unwrap();
    let phi3_ctx = ArchitectureRegistry::lookup("phi-3").unwrap().default_context_length.unwrap();
    let phi4_ctx = ArchitectureRegistry::lookup("phi-4").unwrap().default_context_length.unwrap();
    assert!(phi3_ctx > phi2_ctx, "Phi-3 ctx should exceed Phi-2");
    assert!(phi4_ctx > phi3_ctx, "Phi-4 ctx should exceed Phi-3");
}

// ---------------------------------------------------------------------------
// Architecture count regression guard
// ---------------------------------------------------------------------------

#[test]
fn known_architectures_count_regression() {
    let count = ArchitectureRegistry::known_architectures().len();
    assert!(count >= 70, "Expected at least 70 known architecture strings, got {}", count);
}

// ---------------------------------------------------------------------------
// Config defaults serde roundtrip with architecture applied
// ---------------------------------------------------------------------------

#[test]
fn config_with_arch_defaults_survives_serde() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-4");
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.norm_type, NormType::RmsNorm);
    assert_eq!(cfg2.activation_type, ActivationType::Silu);
    assert_eq!(cfg2.max_position_embeddings, 16384);
}

#[test]
fn config_preserves_custom_fields_after_arch_defaults() {
    let mut cfg = ModelConfig::default();
    cfg.hidden_size = 5120;
    cfg.num_heads = 40;
    cfg.num_key_value_heads = 10;
    cfg.apply_architecture_defaults("phi-4");
    // Architecture defaults should NOT override custom fields
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_heads, 40);
    assert_eq!(cfg.num_key_value_heads, 10);
    // But norm/activation/context should be set
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
}

// ---------------------------------------------------------------------------
// Non-default context preserved for architectures without defaults
// ---------------------------------------------------------------------------

#[test]
fn custom_context_not_overwritten_for_generic_llama() {
    let mut cfg = ModelConfig::default();
    cfg.max_position_embeddings = 8192;
    cfg.apply_architecture_defaults("llama");
    // llama has no default context, so custom should be preserved
    assert_eq!(cfg.max_position_embeddings, 8192);
}

#[test]
fn non_default_context_overwritten_when_arch_has_default() {
    let mut cfg = ModelConfig::default();
    // Default is 2048; phi-4 should override to 16384
    assert_eq!(cfg.max_position_embeddings, 2048);
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.max_position_embeddings, 16384);
}
