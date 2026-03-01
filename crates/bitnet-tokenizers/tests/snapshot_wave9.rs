//! Snapshot wave 9 — tokenizer config serialization and strategy types.
//!
//! Pins the serialized form of `TokenizerConfig` and the Display output
//! of strategy-related types to catch accidental regressions.

use bitnet_tokenizers::TokenizerConfig;
use bitnet_tokenizers::strategy::LlamaVariant;

// ── TokenizerConfig serialization ──────────────────────────────────

#[test]
fn tokenizer_config_default_yaml() {
    let cfg = TokenizerConfig::default();
    insta::assert_yaml_snapshot!(cfg);
}

#[test]
fn tokenizer_config_llama3_yaml() {
    let cfg = TokenizerConfig {
        model_type: "llama".into(),
        vocab_size: 128256,
        pre_tokenizer: Some("byte_level".into()),
        add_bos: true,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: true,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None,
        bpe_merges: None,
    };
    insta::assert_yaml_snapshot!(cfg);
}

#[test]
fn tokenizer_config_debug_default() {
    let cfg = TokenizerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ── LlamaVariant ───────────────────────────────────────────────────

#[test]
fn llama_variant_vocab_sizes() {
    let variants = [LlamaVariant::Llama2, LlamaVariant::Llama3, LlamaVariant::CodeLlama];
    let output: Vec<String> =
        variants.iter().map(|v| format!("{v:?}: vocab={}", v.expected_vocab_size())).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn llama_variant_gpu_requirements() {
    let variants = [LlamaVariant::Llama2, LlamaVariant::Llama3, LlamaVariant::CodeLlama];
    let output: Vec<String> = variants
        .iter()
        .map(|v| format!("{v:?}: requires_gpu={}", v.requires_gpu_acceleration()))
        .collect();
    insta::assert_snapshot!(output.join("\n"));
}
