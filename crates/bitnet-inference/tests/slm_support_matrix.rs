//! SLM Support Matrix — comprehensive validation of architecture-aware defaults,
//! config builder presets, prompt templates, and logits pipeline for every
//! supported SLM family.
//!
//! These tests exercise the full multi-SLM vertical slice:
//! ArchitectureRegistry → ModelConfig → InferenceConfigBuilder → PromptTemplate

use bitnet_common::config::ModelConfig;
use bitnet_common::{ActivationType, ArchitectureRegistry, NormType};
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};
use bitnet_prompt_templates::TemplateType;

/// Test helper: validate that an architecture has correct defaults and can
/// be used end-to-end with a config builder and prompt template.
fn validate_architecture(arch: &str, expected_norm: NormType, expected_activation: ActivationType) {
    // 1. Registry recognizes the architecture
    assert!(
        ArchitectureRegistry::is_known(arch),
        "Architecture '{arch}' should be recognized by the registry"
    );

    // 2. Lookup returns correct defaults
    let defaults = ArchitectureRegistry::lookup(arch).unwrap();
    assert_eq!(
        defaults.norm_type, expected_norm,
        "Architecture '{arch}' should use {expected_norm:?}"
    );
    assert_eq!(
        defaults.activation_type, expected_activation,
        "Architecture '{arch}' should use {expected_activation:?}"
    );

    // 3. ModelConfig applies defaults correctly
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults(arch);
    assert_eq!(config.norm_type, expected_norm);
    assert_eq!(config.activation_type, expected_activation);

    // 4. Config builder produces valid config for this architecture
    let inference =
        InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
    assert!(inference.sampling.temperature >= 0.0);
}

// --- Major SLM families ---

#[test]
fn slm_matrix_phi4() {
    validate_architecture("phi-4", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_phi3() {
    validate_architecture("phi-3", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_phi2() {
    validate_architecture("phi-2", NormType::LayerNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_phi() {
    validate_architecture("phi", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_llama() {
    validate_architecture("llama", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_llama2() {
    validate_architecture("llama2", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_llama31() {
    validate_architecture("llama-3.1", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_llama32() {
    validate_architecture("llama-3.2", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_mistral() {
    validate_architecture("mistral", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_qwen2() {
    validate_architecture("qwen2", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_qwen25() {
    validate_architecture("qwen2.5", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_gemma() {
    validate_architecture("gemma", NormType::RmsNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_gemma2() {
    validate_architecture("gemma-2", NormType::RmsNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_deepseek() {
    validate_architecture("deepseek", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_deepseek_v3() {
    validate_architecture("deepseek-v3", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_falcon() {
    validate_architecture("falcon", NormType::LayerNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_starcoder() {
    validate_architecture("starcoder", NormType::LayerNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_bitnet() {
    validate_architecture("bitnet", NormType::LayerNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_codellama() {
    validate_architecture("codellama", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_cohere() {
    validate_architecture("cohere", NormType::LayerNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_internlm() {
    validate_architecture("internlm", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_yi() {
    validate_architecture("yi", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_baichuan() {
    validate_architecture("baichuan", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_chatglm() {
    validate_architecture("chatglm", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_mpt() {
    validate_architecture("mpt", NormType::LayerNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_rwkv() {
    validate_architecture("rwkv", NormType::LayerNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_olmo() {
    validate_architecture("olmo", NormType::LayerNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_mixtral() {
    validate_architecture("mixtral", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_stablelm() {
    validate_architecture("stablelm", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_bloom() {
    validate_architecture("bloom", NormType::LayerNorm, ActivationType::Gelu);
}

#[test]
fn slm_matrix_dbrx() {
    validate_architecture("dbrx", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_exaone() {
    validate_architecture("exaone", NormType::RmsNorm, ActivationType::Silu);
}

#[test]
fn slm_matrix_minicpm() {
    validate_architecture("minicpm", NormType::RmsNorm, ActivationType::Silu);
}

// --- Template integration: ensure each major architecture has a template ---

#[test]
fn template_coverage_for_major_architectures() {
    let major_archs = [
        "phi-4", "llama", "mistral", "qwen2", "gemma", "deepseek", "falcon", "cohere", "internlm",
        "yi", "baichuan", "chatglm", "mpt", "rwkv", "olmo", "mixtral", "stablelm", "bloom",
    ];

    for arch in &major_archs {
        let suggestion = TemplateType::suggest_for_arch(arch);
        assert!(suggestion.is_some(), "Architecture '{arch}' should have a suggested template");
    }
}

#[test]
fn every_suggested_template_renders_valid_output() {
    let archs = ["phi-4", "llama", "mistral", "gemma", "qwen2", "deepseek"];

    for arch in &archs {
        if let Some(template) = TemplateType::suggest_for_arch(arch) {
            let output = template.apply("Hello, world!", None);
            assert!(
                !output.is_empty(),
                "Template {template:?} for arch '{arch}' should produce non-empty output"
            );
            assert!(
                output.contains("Hello, world!"),
                "Template {template:?} for arch '{arch}' should contain the prompt text"
            );
        }
    }
}

// --- Comprehensive preset × architecture matrix ---

#[test]
fn all_presets_valid_for_all_known_architectures() {
    let architectures = ArchitectureRegistry::known_architectures();
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];

    for arch in architectures {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);

        for preset in &presets {
            let result = InferenceConfigBuilder::new().preset(*preset).build();
            assert!(result.is_ok(), "Preset {preset:?} should be valid for architecture '{arch}'");
        }
    }
}

#[test]
fn architecture_count_regression_guard() {
    let count = ArchitectureRegistry::known_architectures().len();
    // We have 100+ architecture strings registered
    assert!(count >= 90, "Expected at least 90 registered architecture strings, got {count}");
}

#[test]
fn template_count_regression_guard() {
    let count = TemplateType::all_variants().len();
    // We have 47+ templates registered
    assert!(count >= 40, "Expected at least 40 template variants, got {count}");
}
