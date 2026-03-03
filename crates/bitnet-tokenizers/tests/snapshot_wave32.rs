//! Snapshot wave 32 — tokenizer config serialization, vocabulary
//! statistics output, vocab config formatting.

use bitnet_tokenizers::TokenizerConfig;
use bitnet_tokenizers::vocab_analyzer::VocabStats;
use bitnet_tokenizers::vocabulary::{SpecialTokens, VocabConfig};

// ── TokenizerConfig ─────────────────────────────────────────────────

#[test]
fn tokenizer_config_default_debug() {
    let cfg = TokenizerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn tokenizer_config_default_yaml() {
    let cfg = TokenizerConfig::default();
    insta::assert_yaml_snapshot!(cfg);
}

#[test]
fn tokenizer_config_llama3_yaml() {
    let cfg = TokenizerConfig {
        model_type: "llama".to_string(),
        vocab_size: 128256,
        pre_tokenizer: Some("ByteLevel".to_string()),
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

// ── VocabConfig ─────────────────────────────────────────────────────

#[test]
fn vocab_config_default_debug() {
    let cfg = VocabConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn vocab_config_with_tokens_yaml() {
    let cfg = VocabConfig {
        unk_token: Some("<unk>".to_string()),
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        pad_token: Some("<pad>".to_string()),
        additional_special_tokens: vec!["<|eot_id|>".to_string(), "<|begin_of_text|>".to_string()],
    };
    insta::assert_yaml_snapshot!(cfg);
}

// ── SpecialTokens ───────────────────────────────────────────────────

#[test]
fn special_tokens_default_debug() {
    let tokens = SpecialTokens::default();
    insta::assert_debug_snapshot!(tokens);
}

// ── VocabStats ──────────────────────────────────────────────────────

#[test]
fn vocab_stats_empty_debug() {
    let stats = VocabStats::analyze(&[]);
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn vocab_stats_mixed_tokens() {
    let tokens: Vec<String> = vec![
        "<s>".to_string(),
        "</s>".to_string(),
        "<unk>".to_string(),
        "<0x00>".to_string(),
        "<0xFF>".to_string(),
        "hello".to_string(),
        "world".to_string(),
        "a".to_string(),
        "the".to_string(),
        "<|eot_id|>".to_string(),
    ];
    let stats = VocabStats::analyze(&tokens);
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn vocab_stats_content_ratio() {
    let tokens: Vec<String> =
        vec!["<s>".to_string(), "</s>".to_string(), "hello".to_string(), "world".to_string()];
    let stats = VocabStats::analyze(&tokens);
    insta::assert_snapshot!(format!("content_ratio: {:.4}", stats.content_ratio()));
}
