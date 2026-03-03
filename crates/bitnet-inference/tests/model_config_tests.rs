//! Model architecture configuration parsing and validation regression tests.
//!
//! Covers: GenerationConfig builders/validation/serialization, InferenceConfig
//! defaults/overrides, architecture detection via ArchitectureRegistry, GQA
//! head-count validation, context-length configs, hidden-dimension validation,
#![allow(clippy::field_reassign_with_default)]
//! SLM family configs (BitNet, Phi-4, LLaMA, Qwen), and edge cases.

use bitnet_common::ArchitectureRegistry;
use bitnet_common::config::{ActivationType, ModelConfig, NormType};
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_models::config::{GgufModelConfig, GgufQuantizationConfig};

// ===========================================================================
// GenerationConfig builder presets
// ===========================================================================

#[test]
fn generation_config_greedy_preset() {
    let cfg = GenerationConfig::greedy();
    assert_eq!(cfg.temperature, 0.0, "greedy should be deterministic");
    assert_eq!(cfg.top_k, 1, "greedy picks a single token");
    assert_eq!(cfg.top_p, 1.0, "greedy disables nucleus sampling");
    assert_eq!(cfg.max_new_tokens, 100, "default max_new_tokens preserved");
}

#[test]
fn generation_config_creative_preset() {
    let cfg = GenerationConfig::creative();
    assert_eq!(cfg.temperature, 0.9);
    assert_eq!(cfg.top_k, 100);
    assert_eq!(cfg.top_p, 0.95);
    assert_eq!(cfg.repetition_penalty, 1.1);
}

#[test]
fn generation_config_balanced_preset() {
    let cfg = GenerationConfig::balanced();
    assert_eq!(cfg.temperature, 0.7);
    assert_eq!(cfg.top_k, 50);
    assert_eq!(cfg.top_p, 0.9);
    assert_eq!(cfg.repetition_penalty, 1.05);
}

#[test]
fn generation_config_builder_chaining() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(256)
        .with_temperature(0.5)
        .with_top_k(40)
        .with_top_p(0.85)
        .with_repetition_penalty(1.2)
        .with_seed(42)
        .with_stop_sequence("</s>".into())
        .with_eos_token_id(Some(2))
        .with_skip_special_tokens(false)
        .with_add_bos(true);

    assert_eq!(cfg.max_new_tokens, 256);
    assert_eq!(cfg.temperature, 0.5);
    assert_eq!(cfg.top_k, 40);
    assert_eq!(cfg.top_p, 0.85);
    assert_eq!(cfg.repetition_penalty, 1.2);
    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.stop_sequences, vec!["</s>"]);
    assert_eq!(cfg.eos_token_id, Some(2));
    assert!(!cfg.skip_special_tokens);
    assert!(cfg.add_bos);
}

// ===========================================================================
// GenerationConfig field validation
// ===========================================================================

#[test]
fn generation_config_valid_default() {
    assert!(GenerationConfig::default().validate().is_ok());
}

#[test]
fn generation_config_valid_presets() {
    assert!(GenerationConfig::greedy().validate().is_ok());
    assert!(GenerationConfig::creative().validate().is_ok());
    assert!(GenerationConfig::balanced().validate().is_ok());
}

#[test]
fn generation_config_reject_zero_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("max_new_tokens"), "error: {err}");
}

#[test]
fn generation_config_reject_negative_temperature() {
    let cfg = GenerationConfig::default().with_temperature(-0.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn generation_config_accept_zero_temperature() {
    // 0.0 temperature is valid (greedy)
    let cfg = GenerationConfig::default().with_temperature(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn generation_config_reject_top_p_zero() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("top_p"), "error: {err}");
}

#[test]
fn generation_config_reject_top_p_above_one() {
    let cfg = GenerationConfig::default().with_top_p(1.01);
    assert!(cfg.validate().is_err());
}

#[test]
fn generation_config_accept_top_p_one() {
    let cfg = GenerationConfig::default().with_top_p(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn generation_config_reject_zero_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn generation_config_reject_negative_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(-1.0);
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// GenerationConfig serialization round-trip
// ===========================================================================

#[test]
fn generation_config_serde_roundtrip() {
    let original = GenerationConfig::creative()
        .with_max_tokens(512)
        .with_seed(123)
        .with_stop_sequence("<|end|>".into())
        .with_stop_token_ids(vec![128009, 128001])
        .with_eos_token_id(Some(2));

    let json = serde_json::to_string(&original).unwrap();
    let mut restored: GenerationConfig = serde_json::from_str(&json).unwrap();
    restored.rebuild_stop_token_set();

    assert_eq!(restored.max_new_tokens, 512);
    assert_eq!(restored.temperature, original.temperature);
    assert_eq!(restored.top_k, original.top_k);
    assert_eq!(restored.top_p, original.top_p);
    assert_eq!(restored.repetition_penalty, original.repetition_penalty);
    assert_eq!(restored.seed, Some(123));
    assert_eq!(restored.stop_sequences, vec!["<|end|>"]);
    assert_eq!(restored.eos_token_id, Some(2));
    assert!(restored.is_stop_token(128009));
    assert!(restored.is_stop_token(128001));
    assert!(!restored.is_stop_token(999));
}

#[test]
fn inference_config_serde_roundtrip() {
    let original = InferenceConfig::gpu_optimized();
    let json = serde_json::to_string(&original).unwrap();
    let restored: InferenceConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.max_context_length, original.max_context_length);
    assert_eq!(restored.batch_size, original.batch_size);
    assert_eq!(restored.mixed_precision, original.mixed_precision);
    assert_eq!(restored.memory_pool_size, original.memory_pool_size);
}

// ===========================================================================
// InferenceConfig defaults and overrides
// ===========================================================================

#[test]
fn inference_config_defaults() {
    let cfg = InferenceConfig::default();
    assert_eq!(cfg.max_context_length, 2048);
    assert_eq!(cfg.batch_size, 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 512 * 1024 * 1024);
}

#[test]
fn inference_config_cpu_optimized() {
    let cfg = InferenceConfig::cpu_optimized();
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.batch_size, 1);
    assert!(cfg.num_threads > 0);
}

#[test]
fn inference_config_gpu_optimized() {
    let cfg = InferenceConfig::gpu_optimized();
    assert!(cfg.mixed_precision);
    assert_eq!(cfg.batch_size, 4);
    assert_eq!(cfg.memory_pool_size, 1024 * 1024 * 1024);
}

#[test]
fn inference_config_memory_efficient() {
    let cfg = InferenceConfig::memory_efficient();
    assert_eq!(cfg.max_context_length, 1024);
    assert_eq!(cfg.memory_pool_size, 256 * 1024 * 1024);
}

#[test]
fn inference_config_builder_overrides() {
    let cfg = InferenceConfig::default()
        .with_threads(8)
        .with_batch_size(16)
        .with_mixed_precision(true)
        .with_memory_pool_size(2 * 1024 * 1024 * 1024);

    assert_eq!(cfg.num_threads, 8);
    assert_eq!(cfg.batch_size, 16);
    assert!(cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 2 * 1024 * 1024 * 1024);
}

#[test]
fn inference_config_reject_zero_context() {
    let mut cfg = InferenceConfig::default();
    cfg.max_context_length = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_reject_zero_threads() {
    let mut cfg = InferenceConfig::default();
    cfg.num_threads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_config_reject_zero_batch_size() {
    let mut cfg = InferenceConfig::default();
    cfg.batch_size = 0;
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// Architecture detection from strings
// ===========================================================================

#[test]
fn architecture_detection_core_families() {
    let core =
        ["bitnet", "llama", "mistral", "phi", "qwen", "gemma", "deepseek", "falcon", "gpt", "bert"];
    for name in &core {
        assert!(ArchitectureRegistry::is_known(name), "core family '{name}' must be recognised");
    }
}

#[test]
fn architecture_detection_case_insensitive() {
    for name in ["PHI", "Llama", "QWEN2", "BitNet-B1.58", "GEMMA2"] {
        assert!(
            ArchitectureRegistry::is_known(name),
            "case-insensitive lookup failed for '{name}'"
        );
    }
}

#[test]
fn architecture_detection_unknown_returns_none() {
    assert!(!ArchitectureRegistry::is_known("nonexistent"));
    assert!(!ArchitectureRegistry::is_known(""));
    assert!(!ArchitectureRegistry::is_known("gpt-99"));
}

#[test]
fn architecture_defaults_bitnet() {
    let d = ArchitectureRegistry::lookup("bitnet").unwrap();
    assert_eq!(d.norm_type, NormType::LayerNorm);
    assert_eq!(d.activation_type, ActivationType::Silu);
    assert_eq!(d.default_context_length, None);
}

#[test]
fn architecture_defaults_phi4() {
    let d = ArchitectureRegistry::lookup("phi-4").unwrap();
    assert_eq!(d.norm_type, NormType::RmsNorm);
    assert_eq!(d.activation_type, ActivationType::Silu);
    assert_eq!(d.default_context_length, Some(16384));
}

#[test]
fn architecture_defaults_llama() {
    let d = ArchitectureRegistry::lookup("llama").unwrap();
    assert_eq!(d.norm_type, NormType::RmsNorm);
    assert_eq!(d.activation_type, ActivationType::Silu);
}

#[test]
fn architecture_defaults_qwen() {
    let d = ArchitectureRegistry::lookup("qwen").unwrap();
    assert_eq!(d.norm_type, NormType::RmsNorm);
    assert_eq!(d.activation_type, ActivationType::Silu);
}

#[test]
fn architecture_defaults_deepseek_v3() {
    let d = ArchitectureRegistry::lookup("deepseek-v3").unwrap();
    assert_eq!(d.default_context_length, Some(65536));
}

// ===========================================================================
// GQA head count validation
// ===========================================================================

fn make_gguf_config(
    arch: &str,
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    intermediate: usize,
    ctx_len: usize,
) -> GgufModelConfig {
    GgufModelConfig {
        architecture: arch.to_string(),
        model_name: None,
        vocab_size: 32000,
        hidden_size: hidden,
        num_layers: layers,
        num_heads: heads,
        num_kv_heads: kv_heads,
        head_dim: if heads > 0 { hidden / heads } else { 0 },
        intermediate_size: intermediate,
        max_seq_len: ctx_len,
        rope_theta: 10000.0,
        rope_scaling: None,
        quantization: GgufQuantizationConfig::default(),
    }
}

#[test]
fn gqa_valid_divisible_heads() {
    // 32 heads / 8 kv_heads = group size 4
    let cfg = make_gguf_config("llama", 4096, 32, 32, 8, 14336, 8192);
    assert!(cfg.validate().is_ok());
    assert!(cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 4);
}

#[test]
fn gqa_mha_equal_heads() {
    // MHA: num_kv_heads == num_heads
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    assert!(cfg.validate().is_ok());
    assert!(!cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 1);
}

#[test]
fn gqa_reject_kv_heads_greater_than_heads() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 64, 11008, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("num_kv_heads"), "error: {msg}");
}

#[test]
fn gqa_reject_non_divisible_heads() {
    // 32 heads / 5 kv_heads is not divisible
    let cfg = make_gguf_config("llama", 4096, 32, 32, 5, 11008, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("divisible"), "error: {msg}");
}

// ===========================================================================
// Context length configuration
// ===========================================================================

#[test]
fn context_length_4k() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    assert_eq!(cfg.max_seq_len, 4096);
    assert!(cfg.validate().is_ok());
}

#[test]
fn context_length_8k() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 8, 14336, 8192);
    assert_eq!(cfg.max_seq_len, 8192);
    assert!(cfg.validate().is_ok());
}

#[test]
fn context_length_16k() {
    let cfg = make_gguf_config("phi", 5120, 40, 40, 10, 13824, 16384);
    assert_eq!(cfg.max_seq_len, 16384);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// Hidden dimension configuration validation
// ===========================================================================

#[test]
fn hidden_dimension_must_divide_evenly_by_heads() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    assert_eq!(cfg.head_dim, 128); // 4096 / 32
    assert!(cfg.validate().is_ok());
}

#[test]
fn hidden_dimension_mismatch_head_dim() {
    // Manually construct with wrong head_dim
    let mut cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    cfg.head_dim = 64; // should be 128
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("head_dim"), "error: {msg}");
}

// ===========================================================================
// SLM family model configs
// ===========================================================================

#[test]
fn slm_bitnet_2b_config() {
    let cfg = make_gguf_config("bitnet", 2560, 30, 32, 8, 6912, 4096);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.architecture, "bitnet");
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_layers, 30);
    assert!(cfg.is_gqa());

    // Verify architecture defaults
    let defaults = ArchitectureRegistry::lookup("bitnet").unwrap();
    assert_eq!(defaults.norm_type, NormType::LayerNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
}

#[test]
fn slm_phi4_config() {
    let cfg = make_gguf_config("phi", 5120, 40, 40, 10, 13824, 16384);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_layers, 40);
    assert!(cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 4);

    let defaults = ArchitectureRegistry::lookup("phi-4").unwrap();
    assert_eq!(defaults.norm_type, NormType::RmsNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
    assert_eq!(defaults.default_context_length, Some(16384));
}

#[test]
fn slm_llama_7b_config() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);
    assert!(!cfg.is_gqa()); // MHA for 7B

    let defaults = ArchitectureRegistry::lookup("llama").unwrap();
    assert_eq!(defaults.norm_type, NormType::RmsNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
}

#[test]
fn slm_qwen_7b_config() {
    let cfg = make_gguf_config("qwen", 4096, 32, 32, 32, 11008, 4096);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);

    let defaults = ArchitectureRegistry::lookup("qwen").unwrap();
    assert_eq!(defaults.norm_type, NormType::RmsNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
}

// ===========================================================================
// ModelConfig.apply_architecture_defaults
// ===========================================================================

#[test]
fn model_config_apply_defaults_phi4() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 16384);
}

#[test]
fn model_config_apply_defaults_llama() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("llama");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    // llama has no default context length — should keep the default 2048
    assert_eq!(cfg.max_position_embeddings, 2048);
}

#[test]
fn model_config_apply_defaults_unknown_noop() {
    let original = ModelConfig::default();
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("nonexistent_arch");
    assert_eq!(cfg.norm_type, original.norm_type);
    assert_eq!(cfg.activation_type, original.activation_type);
    assert_eq!(cfg.max_position_embeddings, original.max_position_embeddings);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn edge_case_zero_layers() {
    let cfg = make_gguf_config("llama", 4096, 0, 32, 32, 11008, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("num_layers"), "error: {msg}");
}

#[test]
fn edge_case_zero_hidden() {
    let cfg = make_gguf_config("llama", 0, 32, 32, 32, 11008, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("hidden_size"), "error: {msg}");
}

#[test]
fn edge_case_zero_heads() {
    let mut cfg = make_gguf_config("llama", 4096, 32, 0, 0, 11008, 4096);
    cfg.head_dim = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("num_heads"), "error: {msg}");
}

#[test]
fn edge_case_zero_vocab() {
    let cfg = GgufModelConfig {
        vocab_size: 0,
        ..make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096)
    };
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("vocab_size"), "error: {msg}");
}

#[test]
fn edge_case_zero_intermediate_size() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 0, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("intermediate_size"), "error: {msg}");
}

#[test]
fn edge_case_zero_context_length() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 0);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("max_seq_len"), "error: {msg}");
}

#[test]
fn edge_case_hidden_not_divisible_by_heads() {
    // 4097 / 32 won't divide evenly
    let cfg = make_gguf_config("llama", 4097, 32, 32, 32, 11008, 4096);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("divisible"), "error: {msg}");
}

// ===========================================================================
// Memory estimation sanity
// ===========================================================================

#[test]
fn memory_estimate_nonzero() {
    let cfg = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    let est = cfg.memory_estimate();
    assert!(est.weight_bytes > 0);
    assert!(est.kv_cache_bytes > 0);
    assert_eq!(est.total_bytes, est.weight_bytes + est.kv_cache_bytes);
    assert!(!est.summary.is_empty());
}

#[test]
fn memory_estimate_gqa_smaller_kv_cache() {
    let mha = make_gguf_config("llama", 4096, 32, 32, 32, 11008, 4096);
    let gqa = make_gguf_config("llama", 4096, 32, 32, 8, 11008, 4096);
    // GQA with 8 kv_heads should have smaller KV cache than MHA with 32
    assert!(gqa.memory_estimate().kv_cache_bytes < mha.memory_estimate().kv_cache_bytes);
}
