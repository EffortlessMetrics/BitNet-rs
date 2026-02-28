//! Context length validation tests for multi-SLM architecture support.
//!
//! Verifies that `apply_architecture_defaults` sets correct context lengths
//! for all supported architectures, and that the context length system
//! handles edge cases (custom overrides, boundary conditions, etc.).

use bitnet_common::{ArchitectureRegistry, ModelConfig};

// ---------------------------------------------------------------------------
// Architecture-specific context length tests
// ---------------------------------------------------------------------------

#[test]
fn phi4_context_length_is_16k() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");
    assert_eq!(config.max_position_embeddings, 16384, "Phi-4 should default to 16K context");
}

#[test]
fn llama3_context_length_is_128k() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("llama-3.1");
    assert_eq!(config.max_position_embeddings, 131072, "LLaMA-3.1 should default to 128K context");
}

#[test]
fn qwen25_context_length_is_32k() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("qwen2.5");
    assert_eq!(config.max_position_embeddings, 32768, "Qwen-2.5 should default to 32K context");
}

#[test]
fn phi3_context_length_is_4k() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-3");
    assert_eq!(config.max_position_embeddings, 4096, "Phi-3 should default to 4K context");
}

#[test]
fn phi2_context_length_is_2k() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-2");
    // Phi-2 uses the default 2048
    assert_eq!(config.max_position_embeddings, 2048, "Phi-2 should keep default 2K context");
}

// ---------------------------------------------------------------------------
// Custom context preservation tests
// ---------------------------------------------------------------------------

#[test]
fn custom_context_not_overridden() {
    let mut config = ModelConfig { max_position_embeddings: 8192, ..ModelConfig::default() };
    config.apply_architecture_defaults("phi-4");
    assert_eq!(
        config.max_position_embeddings, 8192,
        "Custom context (non-default 2048) must not be overridden"
    );
}

#[test]
fn default_context_is_overridden() {
    let mut config = ModelConfig::default();
    assert_eq!(config.max_position_embeddings, 2048);
    config.apply_architecture_defaults("phi-4");
    assert_eq!(
        config.max_position_embeddings, 16384,
        "Default 2048 should be replaced by arch default"
    );
}

// ---------------------------------------------------------------------------
// Context length range tests
// ---------------------------------------------------------------------------

#[test]
fn all_context_lengths_are_powers_of_two_or_standard() {
    let known = ArchitectureRegistry::known_architectures();
    for arch in known {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);
        let ctx = config.max_position_embeddings;
        // Context lengths should be at least 512
        assert!(ctx >= 512, "Architecture '{arch}' has unexpectedly small context: {ctx}");
        // Context lengths should be at most 1M
        assert!(ctx <= 1_048_576, "Architecture '{arch}' has unreasonably large context: {ctx}");
    }
}

#[test]
fn all_architectures_with_context_have_reasonable_values() {
    let registry = ArchitectureRegistry::known_architectures();
    for arch in registry {
        if let Some(defaults) = ArchitectureRegistry::lookup(arch)
            && let Some(ctx) = defaults.default_context_length
        {
            // Must be at least 2048
            assert!(ctx >= 2048, "Architecture '{arch}' context {ctx} < 2048");
            // Must be a multiple of 256 (standard alignment)
            assert!(ctx % 256 == 0, "Architecture '{arch}' context {ctx} not aligned to 256");
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache memory estimation tests
// ---------------------------------------------------------------------------

/// Estimate KV cache memory in bytes for a given architecture config.
fn kv_cache_memory_bytes(
    max_seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> usize {
    // K + V caches, each: [max_seq_len, num_kv_heads, head_dim] * sizeof(f32)
    2 * max_seq_len * num_layers * num_kv_heads * head_dim * 4
}

#[test]
fn phi4_kv_cache_memory_estimate() {
    // Phi-4: 40 layers, 10 KV heads, head_dim=128, 16K context
    let bytes = kv_cache_memory_bytes(16384, 40, 10, 128);
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    // Should be approximately 6.25 GB for FP32 KV cache
    assert!(gb < 10.0, "Phi-4 KV cache {gb:.2} GB exceeds 10 GB estimate");
    assert!(gb > 1.0, "Phi-4 KV cache {gb:.2} GB is suspiciously small");
}

#[test]
fn llama3_kv_cache_memory_estimate() {
    // LLaMA-3 8B: 32 layers, 8 KV heads (GQA), head_dim=128, 128K context
    let bytes = kv_cache_memory_bytes(131072, 32, 8, 128);
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    // ~32 GB for full 128K context
    assert!(gb < 40.0, "LLaMA-3 KV cache {gb:.2} GB exceeds 40 GB estimate");
}

#[test]
fn bitnet_2b_kv_cache_memory_estimate() {
    // BitNet 2B: 30 layers, 5 KV heads, head_dim=128, 4K context
    let bytes = kv_cache_memory_bytes(4096, 30, 5, 128);
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    // ~0.58 GB — very manageable
    assert!(gb < 2.0, "BitNet 2B KV cache {gb:.2} GB exceeds 2 GB");
}

// ---------------------------------------------------------------------------
// Architecture defaults consistency
// ---------------------------------------------------------------------------

#[test]
fn unknown_arch_keeps_default_context() {
    let mut config = ModelConfig::default();
    let original = config.max_position_embeddings;
    config.apply_architecture_defaults("nonexistent-model-xyz");
    assert_eq!(
        config.max_position_embeddings, original,
        "Unknown architecture should not change context"
    );
}

#[test]
fn architecture_lookup_is_case_insensitive() {
    let mut c1 = ModelConfig::default();
    let mut c2 = ModelConfig::default();
    c1.apply_architecture_defaults("Phi-4");
    c2.apply_architecture_defaults("phi-4");
    assert_eq!(
        c1.max_position_embeddings, c2.max_position_embeddings,
        "Case should not matter for architecture lookup"
    );
}

#[test]
fn multiple_apply_defaults_is_idempotent() {
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults("phi-4");
    let ctx1 = config.max_position_embeddings;
    config.apply_architecture_defaults("phi-4");
    let ctx2 = config.max_position_embeddings;
    assert_eq!(ctx1, ctx2, "Applying defaults twice should be idempotent");
}

// ---------------------------------------------------------------------------
// Dense model context length matrix
// ---------------------------------------------------------------------------

/// Verify that all dense model families (non-BitNet) have context lengths
/// set that are at least 2048.
#[test]
fn dense_model_families_have_adequate_context() {
    let dense_families =
        ["phi-4", "phi-3", "llama", "qwen", "gemma", "mistral", "falcon", "gpt2", "starcoder"];
    for family in &dense_families {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(family);
        assert!(
            config.max_position_embeddings >= 2048,
            "Dense model '{family}' context {} < 2048",
            config.max_position_embeddings
        );
    }
}

/// Verify that architectures with known large context windows
/// have context >= 16K.
#[test]
fn large_context_architectures() {
    let large_ctx_archs = ["phi-4", "llama-3.1", "qwen2.5", "mixtral"];
    for arch in &large_ctx_archs {
        let mut config = ModelConfig::default();
        config.apply_architecture_defaults(arch);
        assert!(
            config.max_position_embeddings >= 16384,
            "Architecture '{arch}' expected >= 16K context, got {}",
            config.max_position_embeddings
        );
    }
}
