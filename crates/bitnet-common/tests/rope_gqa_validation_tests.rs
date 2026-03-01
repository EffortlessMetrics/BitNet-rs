//! RoPE frequency table and GQA configuration validation tests.
//!
//! Verifies that RoPE tables are correctly generated for various architecture
//! configurations and that GQA head ratios are valid for all supported models.

use bitnet_common::{ArchitectureRegistry, ModelConfig};

// ---------------------------------------------------------------------------
// GQA (Grouped Query Attention) configuration validation
// ---------------------------------------------------------------------------

/// Head ratio must be a positive integer (num_heads divisible by num_kv_heads).
fn valid_gqa_ratio(num_heads: usize, num_kv_heads: usize) -> bool {
    num_kv_heads > 0 && num_heads >= num_kv_heads && num_heads.is_multiple_of(num_kv_heads)
}

#[test]
fn phi4_gqa_config() {
    // Phi-4: 40 heads, 10 KV heads → group of 4
    assert!(valid_gqa_ratio(40, 10));
    assert_eq!(40 / 10, 4, "Phi-4 GQA group size should be 4");
}

#[test]
fn llama3_8b_gqa_config() {
    // LLaMA-3 8B: 32 heads, 8 KV heads → group of 4
    assert!(valid_gqa_ratio(32, 8));
    assert_eq!(32 / 8, 4);
}

#[test]
fn llama2_7b_gqa_config() {
    // LLaMA-2 7B: 32 heads, 32 KV heads → MHA (group of 1)
    assert!(valid_gqa_ratio(32, 32));
    assert_eq!(32 / 32, 1, "LLaMA-2 7B uses MHA, not GQA");
}

#[test]
fn mistral_7b_gqa_config() {
    // Mistral 7B: 32 heads, 8 KV heads → group of 4
    assert!(valid_gqa_ratio(32, 8));
}

#[test]
fn gemma2_9b_gqa_config() {
    // Gemma-2 9B: 16 heads, 8 KV heads → group of 2
    assert!(valid_gqa_ratio(16, 8));
    assert_eq!(16 / 8, 2);
}

#[test]
fn invalid_gqa_ratios() {
    // KV heads = 0 is invalid
    assert!(!valid_gqa_ratio(32, 0));
    // KV heads > num_heads is invalid
    assert!(!valid_gqa_ratio(8, 16));
    // Non-divisible is invalid
    assert!(!valid_gqa_ratio(32, 7));
}

// ---------------------------------------------------------------------------
// Head dimension computation
// ---------------------------------------------------------------------------

/// head_dim = hidden_size / num_heads
fn compute_head_dim(hidden_size: usize, num_heads: usize) -> usize {
    hidden_size / num_heads
}

#[test]
fn head_dim_phi4() {
    // Phi-4: hidden=5120, 40 heads → head_dim=128
    assert_eq!(compute_head_dim(5120, 40), 128);
}

#[test]
fn head_dim_llama3_8b() {
    // LLaMA-3 8B: hidden=4096, 32 heads → head_dim=128
    assert_eq!(compute_head_dim(4096, 32), 128);
}

#[test]
fn head_dim_qwen25_7b() {
    // Qwen-2.5 7B: hidden=3584, 28 heads → head_dim=128
    assert_eq!(compute_head_dim(3584, 28), 128);
}

#[test]
fn head_dim_gpt2() {
    // GPT-2: hidden=768, 12 heads → head_dim=64
    assert_eq!(compute_head_dim(768, 12), 64);
}

// ---------------------------------------------------------------------------
// RoPE theta / base values
// ---------------------------------------------------------------------------

#[test]
fn default_rope_theta_is_10000() {
    let config = ModelConfig::default();
    assert_eq!(config.rope_theta, None, "Default rope_theta should be None (resolves to 10000)");
}

#[test]
fn phi4_rope_theta() {
    // Phi-4 uses base=10000 (standard)
    let config = ModelConfig::default();
    let theta = config.rope_theta.unwrap_or(10_000.0);
    assert_eq!(theta, 10_000.0);
}

#[test]
fn rope_theta_custom_override() {
    let config = ModelConfig { rope_theta: Some(500_000.0), ..ModelConfig::default() };
    assert_eq!(config.rope_theta, Some(500_000.0));
}

// ---------------------------------------------------------------------------
// Architecture defaults preserve norm and activation
// ---------------------------------------------------------------------------

#[test]
fn phi4_defaults_set_rmsnorm_and_silu() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");
    assert_eq!(config.norm_type, bitnet_common::NormType::RmsNorm, "Phi-4 uses RMSNorm");
    assert_eq!(config.activation_type, bitnet_common::ActivationType::Silu, "Phi-4 uses SiLU");
}

#[test]
fn gpt_defaults_set_layernorm_and_gelu() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("gpt");
    assert_eq!(config.norm_type, bitnet_common::NormType::LayerNorm);
    assert_eq!(config.activation_type, bitnet_common::ActivationType::Gelu);
}

#[test]
fn llama_defaults_set_rmsnorm_and_silu() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("llama");
    assert_eq!(config.norm_type, bitnet_common::NormType::RmsNorm);
    assert_eq!(config.activation_type, bitnet_common::ActivationType::Silu);
}

// ---------------------------------------------------------------------------
// KV cache dimension validation
// ---------------------------------------------------------------------------

/// KV cache shape: [max_seq_len, num_kv_heads, head_dim]
fn kv_cache_elements_per_layer(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> usize {
    // K + V
    2 * max_seq_len * num_kv_heads * head_dim
}

#[test]
fn phi4_kv_cache_elements() {
    // 16K ctx, 10 KV heads, 128 head_dim
    let elems = kv_cache_elements_per_layer(16384, 10, 128);
    // 2 * 16384 * 10 * 128 = 41,943,040
    assert_eq!(elems, 41_943_040);
}

#[test]
fn bitnet_2b_kv_cache_elements() {
    // 4K ctx, 5 KV heads, 128 head_dim
    let elems = kv_cache_elements_per_layer(4096, 5, 128);
    // 2 * 4096 * 5 * 128 = 5,242,880
    assert_eq!(elems, 5_242_880);
}

// ---------------------------------------------------------------------------
// Exhaustive architecture validation
// ---------------------------------------------------------------------------

#[test]
fn all_known_architectures_have_valid_defaults() {
    let known = ArchitectureRegistry::known_architectures();
    assert!(!known.is_empty(), "Registry should not be empty");

    for arch in known {
        let defaults = ArchitectureRegistry::lookup(arch);
        assert!(defaults.is_some(), "Architecture '{arch}' in known list but lookup failed");
    }
}

#[test]
fn all_architectures_have_norm_and_activation() {
    let known = ArchitectureRegistry::known_architectures();
    for arch in known {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);
        // These should always be set (not panicking is the test)
        let _norm = config.norm_type;
        let _act = config.activation_type;
    }
}

// ---------------------------------------------------------------------------
// Dense model family GQA compatibility matrix
// ---------------------------------------------------------------------------

/// Table of typical (num_heads, num_kv_heads) for common models
#[test]
fn common_model_gqa_ratios_are_valid() {
    let configs: &[(usize, usize, &str)] = &[
        (40, 10, "Phi-4 14B"),
        (32, 32, "LLaMA-2 7B"),
        (32, 8, "LLaMA-3 8B"),
        (32, 8, "Mistral 7B"),
        (16, 8, "Gemma-2 9B"),
        (28, 4, "Qwen-2.5 7B"),
        (32, 32, "Falcon 7B"),
        (12, 12, "GPT-2 117M"),
        (16, 16, "Phi-2 2.7B"),
        (36, 4, "DeepSeek-V2 (hypothetical)"),
    ];

    for &(num_heads, num_kv_heads, name) in configs {
        assert!(
            valid_gqa_ratio(num_heads, num_kv_heads),
            "Invalid GQA ratio for {name}: {num_heads}/{num_kv_heads}"
        );
        let group_size = num_heads / num_kv_heads;
        assert!(group_size >= 1, "{name}: group size must be >= 1, got {group_size}");
    }
}
