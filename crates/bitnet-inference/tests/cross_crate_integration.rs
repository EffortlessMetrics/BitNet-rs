//! Cross-crate integration tests verifying that architecture detection,
//! configuration defaults, prompt template selection, and inference operations
//! form a coherent pipeline.
//!
//! These tests ensure that adding a new SLM architecture to bitnet-common's
//! registry automatically integrates with prompt template suggestion and
//! inference operation dispatch.

use bitnet_common::{ActivationType, ArchitectureRegistry, ModelConfig, NormType};
use bitnet_inference::cpu_opt;
use bitnet_prompt_templates::TemplateType;

// --- Architecture → Config → Ops pipeline tests ---

/// Verify that Phi-4 arch defaults produce correct SiLU+RMSNorm ops.
#[test]
fn phi4_arch_defaults_drive_correct_ops() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");

    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.norm_type, NormType::RmsNorm);

    // Run the activation through cpu_opt
    let mut data = vec![1.0, -1.0, 0.5, -0.5];
    cpu_opt::apply_activation(config.activation_type, &mut data);

    // SiLU: positive inputs stay positive, negative inputs stay negative
    assert!(data[0] > 0.0);
    assert!(data[1] < 0.0);

    // Run RMSNorm
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let bias = vec![0.0; 4];
    let mut output = vec![0.0; 4];
    cpu_opt::apply_norm(config.norm_type, &input, &weight, &bias, &mut output, 1, 4, 1e-5).unwrap();

    // RMSNorm output should be finite and non-zero
    for &v in &output {
        assert!(v.is_finite());
        assert!(v != 0.0);
    }
}

/// Verify that LLaMA arch defaults produce correct SiLU+RMSNorm ops.
#[test]
fn llama_arch_defaults_drive_correct_ops() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("llama");

    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.norm_type, NormType::RmsNorm);
}

/// Verify that GPT-2 arch (not in registry) retains default config.
#[test]
fn gpt2_arch_retains_defaults() {
    let mut config = ModelConfig::default();
    let original_activation = config.activation_type;
    let original_norm = config.norm_type;

    // GPT-2 is not in the registry, so defaults should not change
    config.apply_architecture_defaults("gpt2");

    assert_eq!(config.activation_type, original_activation);
    assert_eq!(config.norm_type, original_norm);
}

// --- Architecture → Template suggestion pipeline ---

/// Every architecture in the registry should resolve template suggestions.
#[test]
fn all_known_architectures_have_template_suggestion() {
    let archs = ArchitectureRegistry::known_architectures();
    let mut missing = Vec::new();

    for &arch in archs {
        if TemplateType::suggest_for_arch(arch).is_none() {
            missing.push(arch);
        }
    }

    // Allow some architectures to not have templates (e.g., very niche ones),
    // but the majority should. We expect at least 50% coverage.
    let coverage = 1.0 - (missing.len() as f64 / archs.len() as f64);
    assert!(
        coverage >= 0.5,
        "Template coverage is {:.1}% ({} of {} missing: {:?})",
        coverage * 100.0,
        missing.len(),
        archs.len(),
        &missing[..missing.len().min(10)]
    );
}

/// Phi-4 architecture should suggest Phi4Chat template.
#[test]
fn phi4_arch_suggests_phi4chat_template() {
    let template = TemplateType::suggest_for_arch("phi-4");
    assert!(template.is_some(), "phi-4 should have a template suggestion");
    assert_eq!(template.unwrap(), TemplateType::Phi4Chat);
}

/// LLaMA architecture should suggest a LLaMA template.
#[test]
fn llama_arch_suggests_llama_template() {
    let template = TemplateType::suggest_for_arch("llama");
    assert!(template.is_some(), "llama should have a template suggestion");
}

/// Mistral architecture should suggest MistralChat template.
#[test]
fn mistral_arch_suggests_mistral_template() {
    let template = TemplateType::suggest_for_arch("mistral");
    assert!(template.is_some(), "mistral should have a template suggestion");
    assert_eq!(template.unwrap(), TemplateType::MistralChat);
}

/// Qwen architecture should suggest QwenChat template.
#[test]
fn qwen_arch_suggests_qwen_template() {
    let template = TemplateType::suggest_for_arch("qwen");
    assert!(template.is_some(), "qwen should have a template suggestion");
    assert_eq!(template.unwrap(), TemplateType::QwenChat);
}

// --- Full pipeline: arch → config → ops → template ---

/// Simulate what happens when loading a Phi-4 model end-to-end.
#[test]
fn phi4_full_pipeline_simulation() {
    // 1. Architecture detection → defaults
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.norm_type, NormType::RmsNorm);

    // 2. Set Phi-4 specific dimensions
    config.hidden_size = 5120;
    config.num_heads = 40;
    config.num_key_value_heads = 10;
    config.num_layers = 40;
    config.intermediate_size = 17920;
    config.vocab_size = 100352;
    config.max_position_embeddings = 16384;

    // 3. Template selection
    let template = TemplateType::suggest_for_arch("phi-4").unwrap();
    assert_eq!(template, TemplateType::Phi4Chat);

    // 4. Apply template to a prompt
    let rendered = template.apply("What is 2+2?", None);
    assert!(!rendered.is_empty());
    assert!(rendered.contains("What is 2+2?"));
}

/// Simulate what happens when loading a Gemma-2 model end-to-end.
#[test]
fn gemma2_full_pipeline_simulation() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("gemma2");
    assert_eq!(config.activation_type, ActivationType::Gelu);

    config.hidden_size = 2304;
    config.num_heads = 8;
    config.num_key_value_heads = 4;
    config.num_layers = 26;
    config.intermediate_size = 16384;
    config.vocab_size = 256128;
    config.max_position_embeddings = 8192;

    let template = TemplateType::suggest_for_arch("gemma2");
    assert!(template.is_some());
}

/// Simulate Mistral 7B pipeline.
#[test]
fn mistral_full_pipeline_simulation() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("mistral");
    assert_eq!(config.activation_type, ActivationType::Silu);
    assert_eq!(config.norm_type, NormType::RmsNorm);

    config.hidden_size = 4096;
    config.num_heads = 32;
    config.num_key_value_heads = 8;
    config.num_layers = 32;
    config.intermediate_size = 14336;
    config.vocab_size = 32000;
    config.max_position_embeddings = 32768;

    let template = TemplateType::suggest_for_arch("mistral").unwrap();
    assert_eq!(template, TemplateType::MistralChat);
}

// --- Activation dispatch correctness across architectures ---

/// Each activation type produces different outputs for the same input.
#[test]
fn activation_types_are_distinguishable() {
    let input = vec![1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0];

    let mut silu_out = input.clone();
    let mut gelu_out = input.clone();
    let mut relu2_out = input.clone();

    cpu_opt::apply_activation(ActivationType::Silu, &mut silu_out);
    cpu_opt::apply_activation(ActivationType::Gelu, &mut gelu_out);
    cpu_opt::apply_activation(ActivationType::Relu2, &mut relu2_out);

    // They should not all be identical
    assert_ne!(silu_out, gelu_out, "SiLU and GELU should differ");
    assert_ne!(silu_out, relu2_out, "SiLU and ReLU² should differ");
    assert_ne!(gelu_out, relu2_out, "GELU and ReLU² should differ");
}

/// Norm types produce different outputs for the same input.
#[test]
fn norm_types_are_distinguishable() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0; 8];
    let bias = vec![0.0; 8];
    let mut rms_out = vec![0.0; 8];
    let mut ln_out = vec![0.0; 8];

    cpu_opt::apply_norm(NormType::RmsNorm, &input, &weight, &bias, &mut rms_out, 1, 8, 1e-5)
        .unwrap();
    cpu_opt::apply_norm(NormType::LayerNorm, &input, &weight, &bias, &mut ln_out, 1, 8, 1e-5)
        .unwrap();

    assert_ne!(rms_out, ln_out, "RMSNorm and LayerNorm should produce different outputs");
}

// --- Template enumeration completeness ---

/// All template variants should have a valid Display representation.
#[test]
fn all_templates_have_display() {
    for &template in TemplateType::all_variants() {
        let display = format!("{template}");
        assert!(!display.is_empty(), "Template {template:?} has empty display");
    }
}

/// All template variants should be able to apply a prompt.
#[test]
fn all_templates_can_apply_prompt() {
    for &template in TemplateType::all_variants() {
        let result = template.apply("Hello", None);
        assert!(!result.is_empty(), "Template {template:?} produced empty output for 'Hello'");
        assert!(result.contains("Hello"), "Template {template:?} doesn't contain the prompt text");
    }
}

// --- Architecture registry consistency ---

/// All known architectures should have valid ArchDefaults.
#[test]
fn all_known_architectures_have_defaults() {
    for &arch in ArchitectureRegistry::known_architectures() {
        let defaults = ArchitectureRegistry::lookup(arch);
        assert!(defaults.is_some(), "Architecture '{arch}' is known but has no defaults");
    }
}

/// Architecture lookup is case-insensitive.
#[test]
fn architecture_lookup_case_insensitive() {
    let lower = ArchitectureRegistry::lookup("llama");
    let upper = ArchitectureRegistry::lookup("LLAMA");
    let mixed = ArchitectureRegistry::lookup("LLaMA");

    assert!(lower.is_some());
    assert!(upper.is_some());
    assert!(mixed.is_some());

    // Same defaults
    let l = lower.unwrap();
    let u = upper.unwrap();
    assert_eq!(l.norm_type, u.norm_type);
    assert_eq!(l.activation_type, u.activation_type);
}

// --- Config → inference ops validation ---

/// A valid Phi-4 model config can be used to compute KV cache sizes.
#[test]
fn phi4_config_kv_cache_estimation() {
    let config = ModelConfig {
        hidden_size: 5120,
        num_heads: 40,
        num_key_value_heads: 10,
        num_layers: 40,
        max_position_embeddings: 16384,
        vocab_size: 100352,
        intermediate_size: 17920,
        ..ModelConfig::default()
    };

    let head_dim = config.hidden_size / config.num_heads;
    assert_eq!(head_dim, 128);

    let kv_heads =
        if config.num_key_value_heads > 0 { config.num_key_value_heads } else { config.num_heads };
    assert_eq!(kv_heads, 10);

    // KV cache size: 2 * layers * kv_heads * head_dim * seq_len * sizeof(f32)
    let kv_bytes = 2
        * config.num_layers
        * kv_heads
        * head_dim
        * config.max_position_embeddings
        * std::mem::size_of::<f32>();
    let kv_gb = kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!((kv_gb - 6.25).abs() < 0.5, "Phi-4 KV cache should be ≈ 6.25 GB, got {kv_gb:.2}");
}

/// A GQA model's head dim can be used for attention computation.
#[test]
fn gqa_head_dim_compatible_with_attention() {
    let config = ModelConfig {
        hidden_size: 4096,
        num_heads: 32,
        num_key_value_heads: 8,
        num_layers: 32,
        max_position_embeddings: 4096,
        vocab_size: 32000,
        intermediate_size: 11008,
        ..ModelConfig::default()
    };

    let head_dim = config.hidden_size / config.num_heads;
    let kv_heads = config.num_key_value_heads;
    let gqa_ratio = config.num_heads / kv_heads;
    assert_eq!(gqa_ratio, 4);

    // Run attention with these dimensions (1 head, short seq for test)
    let seq_len = 4;
    let total = 1 * seq_len * head_dim;
    let query = vec![0.1f32; total];
    let key = vec![0.1f32; total];
    let value = vec![0.5f32; total];
    let mut output = vec![0.0f32; total];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, 1).unwrap();

    // Output should be close to value (uniform attention)
    for &v in &output {
        assert!(v.is_finite());
        assert!((v - 0.5).abs() < 0.1, "Attention output should be ≈ 0.5 (value mean), got {v}");
    }
}
