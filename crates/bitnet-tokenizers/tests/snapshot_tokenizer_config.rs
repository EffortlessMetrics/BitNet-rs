//! Wave 6 snapshot tests for bitnet-tokenizers configuration types.
//!
//! Pins TokenizerConfig defaults, special token configurations,
//! and config serialization.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};

// ── TokenizerConfig defaults ────────────────────────────────────────────────

#[test]
fn snapshot_tokenizer_config_default_debug() {
    insta::assert_debug_snapshot!("tokenizer_config_default_debug", TokenizerConfig::default());
}

#[test]
fn snapshot_tokenizer_config_default_json() {
    insta::assert_json_snapshot!("tokenizer_config_default_json", TokenizerConfig::default());
}

#[test]
fn snapshot_tokenizer_config_new_equals_default() {
    let new = TokenizerConfig::new();
    let default = TokenizerConfig::default();
    // Both should produce identical snapshots
    insta::assert_snapshot!(
        "tokenizer_config_new_eq_default",
        format!(
            "new_model_type={} default_model_type={} equal={}",
            new.model_type,
            default.model_type,
            new.model_type == default.model_type
                && new.vocab_size == default.vocab_size
                && new.add_bos == default.add_bos
                && new.add_eos == default.add_eos,
        )
    );
}

// ── Special token configurations ────────────────────────────────────────────

#[test]
fn snapshot_tokenizer_config_special_tokens_default() {
    let cfg = TokenizerConfig::default();
    insta::assert_snapshot!(
        "tokenizer_config_special_tokens",
        format!(
            "bos={:?} eos={:?} pad={:?} unk={:?}",
            cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id, cfg.unk_token_id,
        )
    );
}

#[test]
fn snapshot_tokenizer_config_with_special_tokens() {
    let cfg = TokenizerConfig {
        model_type: "llama".into(),
        vocab_size: 32000,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        pad_token_id: Some(0),
        unk_token_id: Some(3),
        add_bos: true,
        add_eos: false,
        ..Default::default()
    };
    insta::assert_json_snapshot!("tokenizer_config_llama_special_tokens", cfg);
}

#[test]
fn snapshot_tokenizer_config_gpt2_style() {
    let cfg = TokenizerConfig {
        model_type: "gpt2".into(),
        vocab_size: 50257,
        bos_token_id: None,
        eos_token_id: Some(50256),
        pad_token_id: None,
        unk_token_id: None,
        pre_tokenizer: Some("ByteLevel".into()),
        byte_fallback: false,
        ..Default::default()
    };
    insta::assert_json_snapshot!("tokenizer_config_gpt2_style", cfg);
}

#[test]
fn snapshot_tokenizer_config_bitnet_style() {
    let cfg = TokenizerConfig {
        model_type: "bitnet".into(),
        vocab_size: 32064,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        pad_token_id: None,
        unk_token_id: Some(0),
        add_bos: true,
        add_eos: true,
        add_space_prefix: true,
        byte_fallback: true,
        ..Default::default()
    };
    insta::assert_json_snapshot!("tokenizer_config_bitnet_style", cfg);
}

// ── Config serialization round-trip ─────────────────────────────────────────

#[test]
fn snapshot_tokenizer_config_json_roundtrip() {
    let cfg = TokenizerConfig {
        model_type: "llama".into(),
        vocab_size: 32000,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        add_bos: true,
        ..Default::default()
    };
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    let deserialized: TokenizerConfig = serde_json::from_str(&json).unwrap();
    insta::assert_json_snapshot!("tokenizer_config_json_roundtrip", deserialized);
}

// ── BasicTokenizer special token snapshots ──────────────────────────────────

#[test]
fn snapshot_basic_tokenizer_special_token_ids() {
    let tok = BasicTokenizer::new();
    insta::assert_snapshot!(
        "basic_tokenizer_special_ids",
        format!(
            "bos={:?} eos={:?} pad={:?} vocab_size={}",
            tok.bos_token_id(),
            tok.eos_token_id(),
            tok.pad_token_id(),
            tok.vocab_size(),
        )
    );
}

#[test]
fn snapshot_basic_tokenizer_custom_config() {
    let tok = BasicTokenizer::with_config(128256, Some(128000), Some(128001), Some(128002));
    insta::assert_snapshot!(
        "basic_tokenizer_custom_special_ids",
        format!(
            "bos={:?} eos={:?} pad={:?} vocab_size={}",
            tok.bos_token_id(),
            tok.eos_token_id(),
            tok.pad_token_id(),
            tok.vocab_size(),
        )
    );
}
