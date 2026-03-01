//! Architecture defaults validation tests.
//!
//! Ensures `ModelConfig::apply_architecture_defaults()` correctly sets
//! norm_type, activation_type, and default_context_length for every
//! architecture in the registry.

use bitnet_common::ArchitectureRegistry;
use bitnet_common::config::{ActivationType, ModelConfig, NormType};

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

/// Apply defaults for `arch` and return the mutated config.
fn defaults_for(arch: &str) -> ModelConfig {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults(arch);
    cfg
}

// ────────────────────────────────────────────────────────────────
// 1. Every known architecture should be resolvable
// ────────────────────────────────────────────────────────────────

#[test]
fn all_known_architectures_resolve() {
    for arch in ArchitectureRegistry::known_architectures() {
        assert!(ArchitectureRegistry::is_known(arch), "architecture '{}' should be known", arch,);
    }
}

#[test]
fn unknown_architecture_returns_none() {
    assert!(ArchitectureRegistry::lookup("nonexistent-model-xyz").is_none());
    assert!(ArchitectureRegistry::lookup("").is_none());
    assert!(ArchitectureRegistry::lookup("UNKNOWN").is_none());
}

// ────────────────────────────────────────────────────────────────
// 2. Case-insensitivity
// ────────────────────────────────────────────────────────────────

#[test]
fn lookup_is_case_insensitive() {
    let cases = ["PHI", "Phi", "phi", "PHI-4", "Phi-4", "LLAMA", "Llama", "llama"];
    for name in &cases {
        assert!(
            ArchitectureRegistry::is_known(name),
            "'{name}' should be found (case-insensitive)",
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 3. Specific architecture defaults — Phi family
// ────────────────────────────────────────────────────────────────

#[test]
fn phi4_defaults() {
    let cfg = defaults_for("phi-4");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 16384);
}

#[test]
fn phi3_defaults() {
    let cfg = defaults_for("phi-3");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn phi2_defaults() {
    let cfg = defaults_for("phi-2");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

// ────────────────────────────────────────────────────────────────
// 4. LLaMA family
// ────────────────────────────────────────────────────────────────

#[test]
fn llama_defaults() {
    let cfg = defaults_for("llama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    // No default context length for generic "llama"
    assert_eq!(cfg.max_position_embeddings, 2048); // unchanged default
}

#[test]
fn llama2_defaults() {
    let cfg = defaults_for("llama2");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn llama31_defaults() {
    let cfg = defaults_for("llama-3.1");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 131072);
}

#[test]
fn llama32_defaults() {
    let cfg = defaults_for("llama-3.2");
    assert_eq!(cfg.max_position_embeddings, 131072);
}

// ────────────────────────────────────────────────────────────────
// 5. Qwen family
// ────────────────────────────────────────────────────────────────

#[test]
fn qwen_defaults() {
    let cfg = defaults_for("qwen");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn qwen25_defaults() {
    let cfg = defaults_for("qwen2.5");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 32768);
}

// ────────────────────────────────────────────────────────────────
// 6. Gemma family
// ────────────────────────────────────────────────────────────────

#[test]
fn gemma_defaults() {
    let cfg = defaults_for("gemma");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn gemma2_defaults() {
    let cfg = defaults_for("gemma2");
    assert_eq!(cfg.max_position_embeddings, 8192);
}

#[test]
fn codegemma_defaults() {
    let cfg = defaults_for("codegemma");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 8192);
}

// ────────────────────────────────────────────────────────────────
// 7. Mistral family
// ────────────────────────────────────────────────────────────────

#[test]
fn mistral_defaults() {
    let cfg = defaults_for("mistral");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn mistral_nemo_defaults() {
    let cfg = defaults_for("mistral-nemo");
    assert_eq!(cfg.max_position_embeddings, 128000);
}

#[test]
fn mixtral_defaults() {
    let cfg = defaults_for("mixtral");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 32768);
}

// ────────────────────────────────────────────────────────────────
// 8. DeepSeek family
// ────────────────────────────────────────────────────────────────

#[test]
fn deepseek_defaults() {
    let cfg = defaults_for("deepseek");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn deepseekv3_defaults() {
    let cfg = defaults_for("deepseek-v3");
    assert_eq!(cfg.max_position_embeddings, 65536);
}

// ────────────────────────────────────────────────────────────────
// 9. BitNet family
// ────────────────────────────────────────────────────────────────

#[test]
fn bitnet_defaults() {
    let cfg = defaults_for("bitnet");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

// ────────────────────────────────────────────────────────────────
// 10. Code models
// ────────────────────────────────────────────────────────────────

#[test]
fn starcoder_defaults() {
    let cfg = defaults_for("starcoder");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn codellama_defaults() {
    let cfg = defaults_for("codellama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

// ────────────────────────────────────────────────────────────────
// 11. Falcon family
// ────────────────────────────────────────────────────────────────

#[test]
fn falcon_defaults() {
    let cfg = defaults_for("falcon");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn falcon2_defaults() {
    let cfg = defaults_for("falcon-2");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 8192);
}

// ────────────────────────────────────────────────────────────────
// 12. GPT / BERT family
// ────────────────────────────────────────────────────────────────

#[test]
fn gpt_defaults() {
    let cfg = defaults_for("gpt");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
}

#[test]
fn bloom_defaults() {
    let cfg = defaults_for("bloom");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

// ────────────────────────────────────────────────────────────────
// 13. Chinese LLM family
// ────────────────────────────────────────────────────────────────

#[test]
fn chatglm_defaults() {
    let cfg = defaults_for("chatglm");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn baichuan_defaults() {
    let cfg = defaults_for("baichuan");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn yi_defaults() {
    let cfg = defaults_for("yi");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
}

#[test]
fn xverse_defaults() {
    let cfg = defaults_for("xverse");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 8192);
}

// ────────────────────────────────────────────────────────────────
// 14. Instruction-tuned model families
// ────────────────────────────────────────────────────────────────

#[test]
fn zephyr_defaults() {
    let cfg = defaults_for("zephyr");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn vicuna_defaults() {
    let cfg = defaults_for("vicuna");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn orca_defaults() {
    let cfg = defaults_for("orca");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn wizardlm_defaults() {
    let cfg = defaults_for("wizardlm");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn alpaca_defaults() {
    let cfg = defaults_for("alpaca");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

// ────────────────────────────────────────────────────────────────
// 15. Newer / Specialty models
// ────────────────────────────────────────────────────────────────

#[test]
fn jamba_defaults() {
    let cfg = defaults_for("jamba");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 256000);
}

#[test]
fn dbrx_defaults() {
    let cfg = defaults_for("dbrx");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 32768);
}

#[test]
fn persimmon_defaults() {
    let cfg = defaults_for("persimmon");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 16384);
}

#[test]
fn arctic_defaults() {
    let cfg = defaults_for("arctic");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn exaone_defaults() {
    let cfg = defaults_for("exaone");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn minicpm_defaults() {
    let cfg = defaults_for("minicpm");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

// ────────────────────────────────────────────────────────────────
// 16. OLMo family
// ────────────────────────────────────────────────────────────────

#[test]
fn olmo_defaults() {
    let cfg = defaults_for("olmo");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn olmo2_defaults() {
    let cfg = defaults_for("olmo2");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

// ────────────────────────────────────────────────────────────────
// 17. Cohere family
// ────────────────────────────────────────────────────────────────

#[test]
fn cohere_command_defaults() {
    let cfg = defaults_for("command");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 128000);
}

#[test]
fn aya_defaults() {
    let cfg = defaults_for("aya");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 8192);
}

// ────────────────────────────────────────────────────────────────
// 18. RWKV family
// ────────────────────────────────────────────────────────────────

#[test]
fn rwkv_defaults() {
    let cfg = defaults_for("rwkv");
    assert_eq!(cfg.norm_type, NormType::LayerNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

// ────────────────────────────────────────────────────────────────
// 19. ChatML variants
// ────────────────────────────────────────────────────────────────

#[test]
fn stablelm_defaults() {
    let cfg = defaults_for("stablelm");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn tinyllama_defaults() {
    let cfg = defaults_for("tinyllama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 2048);
}

// ────────────────────────────────────────────────────────────────
// 20. Context length override logic
// ────────────────────────────────────────────────────────────────

#[test]
fn context_length_only_overrides_default() {
    // If config already has non-default context, it should NOT be overridden
    let mut cfg = ModelConfig::default();
    cfg.max_position_embeddings = 4096; // non-default (default is 2048)
    cfg.apply_architecture_defaults("phi-4");

    // Should keep 4096 because the code only overrides when == 2048
    assert_eq!(cfg.max_position_embeddings, 4096);
}

#[test]
fn context_length_overrides_when_default() {
    // If config has default context (2048), it should be overridden
    let mut cfg = ModelConfig::default();
    assert_eq!(cfg.max_position_embeddings, 2048); // starts at default
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.max_position_embeddings, 16384); // overridden by Phi-4
}

// ────────────────────────────────────────────────────────────────
// 21. Alias consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn phi_aliases_match() {
    let phi = defaults_for("phi");
    let phi4 = defaults_for("phi-4");
    assert_eq!(phi.norm_type, phi4.norm_type);
    assert_eq!(phi.activation_type, phi4.activation_type);
    assert_eq!(phi.max_position_embeddings, phi4.max_position_embeddings);
}

#[test]
fn llama_version_aliases() {
    let l31a = defaults_for("llama-3.1");
    let l31b = defaults_for("llama3.1");
    let l31c = defaults_for("llama31");
    assert_eq!(l31a.max_position_embeddings, l31b.max_position_embeddings);
    assert_eq!(l31b.max_position_embeddings, l31c.max_position_embeddings);
}

#[test]
fn qwen_aliases() {
    let q = defaults_for("qwen2.5");
    let qa = defaults_for("qwen-2.5");
    assert_eq!(q.max_position_embeddings, qa.max_position_embeddings);
    assert_eq!(q.norm_type, qa.norm_type);
}

#[test]
fn falcon2_aliases() {
    let f2a = defaults_for("falcon-2");
    let f2b = defaults_for("falcon2");
    assert_eq!(f2a.norm_type, f2b.norm_type);
    assert_eq!(f2a.max_position_embeddings, f2b.max_position_embeddings);
}

// ────────────────────────────────────────────────────────────────
// 22. Norm type distribution validation
// ────────────────────────────────────────────────────────────────

#[test]
fn layernorm_architectures_are_older_or_special() {
    // Architectures using LayerNorm tend to be older (GPT-2, BERT, Falcon-1, etc.)
    let layernorm_archs = ["gpt", "bert", "starcoder", "falcon", "bloom", "mpt", "persimmon"];
    for arch in &layernorm_archs {
        let cfg = defaults_for(arch);
        assert_eq!(cfg.norm_type, NormType::LayerNorm, "{arch} should use LayerNorm");
    }
}

#[test]
fn rmsnorm_architectures_are_modern() {
    // Modern architectures use RMSNorm
    let rmsnorm_archs =
        ["phi-4", "llama", "mistral", "qwen", "deepseek", "mixtral", "yi", "baichuan"];
    for arch in &rmsnorm_archs {
        let cfg = defaults_for(arch);
        assert_eq!(cfg.norm_type, NormType::RmsNorm, "{arch} should use RmsNorm");
    }
}
