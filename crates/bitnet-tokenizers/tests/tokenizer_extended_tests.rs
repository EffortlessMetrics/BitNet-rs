// SPDX-License-Identifier: MIT OR Apache-2.0
//! Extended tests for `bitnet-tokenizers`.
//!
//! Covers areas complementing `tokenizer_comprehensive_tests.rs`:
//! - `TokenizerConfig` construction, field assignment, serde round-trip
//! - `BasicTokenizer` construction variants and default equivalence
//! - Encoding / decoding special-token semantics (BOS/EOS/PAD)
//! - `token_to_piece` for byte and high-ID tokens
//! - `get_family_name` heuristic
//! - `real_vocab_size` default delegation
//! - Legacy shim methods
//! - `MockTokenizer::with_special_tokens` and `token_to_id`
//! - `TokenizerBuilder::from_pretrained` built-in profiles
//! - `TokenizerBuilder::from_file` / `from_path` error paths
//! - Property tests: BOS/EOS opt-in, token_to_piece consistency

use bitnet_tokenizers::{
    BasicTokenizer, MockTokenizer, Tokenizer, TokenizerBuilder, TokenizerConfig,
    from_path,
};
use proptest::prelude::*;
use std::path::Path;

// ── TokenizerConfig defaults and construction ────────────────────────────────

#[test]
fn config_new_has_empty_model_type() {
    let cfg = TokenizerConfig::new();
    assert_eq!(cfg.model_type, "");
}

#[test]
fn config_default_has_zero_vocab_size() {
    let cfg = TokenizerConfig::default();
    assert_eq!(cfg.vocab_size, 0);
}

#[test]
fn config_default_booleans_are_false() {
    let cfg = TokenizerConfig::default();
    assert!(!cfg.add_bos);
    assert!(!cfg.add_eos);
    assert!(!cfg.add_space_prefix);
    assert!(!cfg.byte_fallback);
}

#[test]
fn config_default_option_fields_are_none() {
    let cfg = TokenizerConfig::default();
    assert!(cfg.bos_token_id.is_none());
    assert!(cfg.eos_token_id.is_none());
    assert!(cfg.pad_token_id.is_none());
    assert!(cfg.unk_token_id.is_none());
    assert!(cfg.vocabulary.is_none());
    assert!(cfg.bpe_merges.is_none());
    assert!(cfg.pre_tokenizer.is_none());
}

#[test]
fn config_field_assignment_roundtrip() {
    let mut cfg = TokenizerConfig::new();
    cfg.model_type = "llama".into();
    cfg.vocab_size = 32_000;
    cfg.bos_token_id = Some(1);
    cfg.eos_token_id = Some(2);
    cfg.pad_token_id = Some(0);
    cfg.add_bos = true;

    assert_eq!(cfg.model_type, "llama");
    assert_eq!(cfg.vocab_size, 32_000);
    assert_eq!(cfg.bos_token_id, Some(1));
    assert_eq!(cfg.eos_token_id, Some(2));
    assert_eq!(cfg.pad_token_id, Some(0));
    assert!(cfg.add_bos);
}

#[test]
fn config_serde_roundtrip_preserves_fields() {
    let mut cfg = TokenizerConfig::new();
    cfg.model_type = "gpt2".into();
    cfg.vocab_size = 50_257;
    cfg.eos_token_id = Some(50_256);

    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: TokenizerConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.model_type, cfg.model_type);
    assert_eq!(back.vocab_size, cfg.vocab_size);
    assert_eq!(back.eos_token_id, cfg.eos_token_id);
}

// ── BasicTokenizer construction ───────────────────────────────────────────────

#[test]
fn basic_tokenizer_default_equals_new() {
    let a = BasicTokenizer::new();
    let b = BasicTokenizer::default();
    assert_eq!(a.vocab_size(), b.vocab_size());
    assert_eq!(a.eos_token_id(), b.eos_token_id());
    assert_eq!(a.bos_token_id(), b.bos_token_id());
    assert_eq!(a.pad_token_id(), b.pad_token_id());
}

#[test]
fn basic_tokenizer_new_gpt2_defaults() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.vocab_size(), 50_257);
    assert_eq!(tok.eos_token_id(), Some(50_256));
    assert!(tok.bos_token_id().is_none());
    assert!(tok.pad_token_id().is_none());
}

#[test]
fn basic_tokenizer_with_config_all_fields() {
    let tok = BasicTokenizer::with_config(1_000, Some(1), Some(2), Some(0));
    assert_eq!(tok.vocab_size(), 1_000);
    assert_eq!(tok.bos_token_id(), Some(1));
    assert_eq!(tok.eos_token_id(), Some(2));
    assert_eq!(tok.pad_token_id(), Some(0));
}

// ── Encoding semantics ────────────────────────────────────────────────────────

#[test]
fn encode_add_bos_is_noop_when_bos_token_id_is_none() {
    let tok = BasicTokenizer::new(); // bos_token_id = None
    let without_bos = tok.encode("hi", false, false).unwrap();
    let with_bos_flag = tok.encode("hi", true, false).unwrap();
    assert_eq!(
        without_bos, with_bos_flag,
        "add_bos=true must be a no-op when bos_token_id is None"
    );
}

#[test]
fn encode_bos_prepended_when_configured() {
    let tok = BasicTokenizer::with_config(50_257, Some(99), Some(50_256), None);
    let tokens = tok.encode("A", true, false).unwrap();
    assert_eq!(tokens[0], 99, "BOS token must be first");
}

#[test]
fn encode_eos_appended_when_add_special_true() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, true).unwrap();
    assert!(tokens.contains(&50_256), "EOS must appear when add_special=true");
}

#[test]
fn encode_eos_absent_when_add_special_false() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, false).unwrap();
    assert!(!tokens.contains(&50_256), "EOS must be absent when add_special=false");
}

#[test]
fn encode_small_vocab_rejects_large_bytes() {
    let tok = BasicTokenizer::with_config(10, None, None, None);
    // 'A' = 65, which exceeds vocab_size=10
    let result = tok.encode("A", false, false);
    assert!(result.is_err(), "byte >= vocab_size must be an error");
}

#[test]
fn encode_single_byte_produces_correct_id() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, false).unwrap();
    assert_eq!(tokens, vec![65u32], "'A' (0x41=65) should map to token ID 65");
}

// ── Decoding semantics ────────────────────────────────────────────────────────

#[test]
fn decode_eos_token_is_skipped() {
    let tok = BasicTokenizer::new(); // eos = 50256
    let decoded = tok.decode(&[50_256]).unwrap();
    assert!(decoded.is_empty(), "lone EOS must decode to empty string");
}

#[test]
fn decode_bos_token_is_skipped() {
    let tok = BasicTokenizer::with_config(50_257, Some(1), Some(2), None);
    let decoded = tok.decode(&[1]).unwrap();
    assert!(decoded.is_empty(), "lone BOS must decode to empty string");
}

#[test]
fn decode_pad_token_is_skipped() {
    let tok = BasicTokenizer::with_config(50_257, None, None, Some(0));
    // PAD = 0; byte 0 would normally decode but as a special token it is skipped
    let decoded = tok.decode(&[0]).unwrap();
    assert!(decoded.is_empty(), "lone PAD must decode to empty string");
}

#[test]
fn decode_high_non_special_id_dropped() {
    let tok = BasicTokenizer::new();
    // ID 300 is not a special token and is ≥ 256 (no byte mapping)
    let decoded = tok.decode(&[300]).unwrap();
    assert!(decoded.is_empty(), "non-special high IDs (≥256) must be silently dropped");
}

#[test]
fn decode_ascii_byte_sequence_is_correct() {
    let tok = BasicTokenizer::new();
    // 'h'=104, 'i'=105
    let decoded = tok.decode(&[104, 105]).unwrap();
    assert_eq!(decoded, "hi");
}

// ── token_to_piece ────────────────────────────────────────────────────────────

#[test]
fn token_to_piece_printable_ascii_matches_char() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(65).unwrap(); // 'A'
    assert_eq!(piece, "A");
}

#[test]
fn token_to_piece_high_id_uses_angle_bracket_format() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(9_999).unwrap();
    assert_eq!(piece, "<token_9999>");
}

#[test]
fn token_to_piece_byte_zero_is_some() {
    let tok = BasicTokenizer::new();
    assert!(tok.token_to_piece(0).is_some());
}

// ── real_vocab_size default ───────────────────────────────────────────────────

#[test]
fn real_vocab_size_equals_vocab_size_for_basic_tokenizer() {
    let tok = BasicTokenizer::new();
    assert_eq!(
        tok.real_vocab_size(),
        tok.vocab_size(),
        "BasicTokenizer must delegate real_vocab_size to vocab_size"
    );
}

// ── get_family_name ───────────────────────────────────────────────────────────

#[test]
fn basic_tokenizer_family_name_is_unknown() {
    let tok = BasicTokenizer::new();
    assert_eq!(
        tok.get_family_name(),
        "unknown",
        "BasicTokenizer has no special token mappings so family should be 'unknown'"
    );
}

#[test]
fn mock_tokenizer_with_eot_id_is_llama3_family() {
    let tok = MockTokenizer::with_special_tokens(&[("<|eot_id|>", 128_009)]);
    assert_eq!(tok.get_family_name(), "llama3");
}

#[test]
fn mock_tokenizer_with_start_header_id_is_llama3_family() {
    let tok = MockTokenizer::with_special_tokens(&[("<|start_header_id|>", 128_006)]);
    assert_eq!(tok.get_family_name(), "llama3");
}

#[test]
fn mock_tokenizer_with_inst_token_is_mistral_family() {
    let tok = MockTokenizer::with_special_tokens(&[("[INST]", 3)]);
    assert_eq!(tok.get_family_name(), "mistral-instruct");
}

// ── Legacy shims ──────────────────────────────────────────────────────────────

#[test]
fn encode_legacy_matches_encode_with_same_semantics() {
    let tok = BasicTokenizer::new();
    // encode_legacy(text, add_special) → encode(text, /*add_bos=*/true, add_special)
    let via_legacy = tok.encode_legacy("abc", false).unwrap();
    let via_direct = tok.encode("abc", true, false).unwrap();
    assert_eq!(via_legacy, via_direct);
}

#[test]
fn decode_legacy_matches_decode() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("hello", false, false).unwrap();
    let via_legacy = tok.decode_legacy(&tokens, true).unwrap();
    let via_direct = tok.decode(&tokens).unwrap();
    assert_eq!(via_legacy, via_direct);
}

// ── MockTokenizer::token_to_id ────────────────────────────────────────────────

#[test]
fn mock_tokenizer_token_to_id_returns_mapped_value() {
    let tok = MockTokenizer::with_special_tokens(&[("<|end|>", 32_000)]);
    assert_eq!(tok.token_to_id("<|end|>"), Some(32_000));
}

#[test]
fn mock_tokenizer_token_to_id_returns_none_for_unmapped() {
    let tok = MockTokenizer::new();
    assert!(tok.token_to_id("<|eot_id|>").is_none());
}

// ── TokenizerBuilder::from_pretrained profiles ───────────────────────────────

#[test]
fn builder_from_pretrained_gpt2_has_correct_vocab() {
    let tok = TokenizerBuilder::from_pretrained("gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 50_257);
    assert_eq!(tok.eos_token_id(), Some(50_256));
}

#[test]
fn builder_from_pretrained_bert_has_cls_sep_tokens() {
    let tok = TokenizerBuilder::from_pretrained("bert").unwrap();
    assert_eq!(tok.vocab_size(), 30_522);
    assert_eq!(tok.bos_token_id(), Some(101)); // [CLS]
    assert_eq!(tok.eos_token_id(), Some(102)); // [SEP]
    assert_eq!(tok.pad_token_id(), Some(0));
}

#[test]
fn builder_from_pretrained_tiny_has_small_vocab() {
    let tok = TokenizerBuilder::from_pretrained("tiny").unwrap();
    assert_eq!(tok.vocab_size(), 1_000);
}

#[test]
fn builder_from_pretrained_unknown_falls_back_to_gpt2_defaults() {
    let tok = TokenizerBuilder::from_pretrained("completely_unknown_xyz").unwrap();
    assert_eq!(tok.vocab_size(), 50_257);
}

// ── from_path / from_file error handling ─────────────────────────────────────

#[test]
fn from_path_unsupported_extension_errors() {
    let result = from_path(Path::new("model.safetensors"));
    assert!(result.is_err(), "unsupported extension must return Err");
}

#[test]
fn from_path_bin_extension_errors() {
    let result = from_path(Path::new("tokenizer.bin"));
    assert!(result.is_err(), "'.bin' extension is unsupported and must return Err");
}

#[test]
fn from_path_no_extension_errors() {
    let result = from_path(Path::new("/tmp/tokenizer_no_ext"));
    assert!(result.is_err(), "missing extension must return Err");
}

#[test]
fn from_file_nonexistent_json_errors() {
    let result = TokenizerBuilder::from_file(Path::new("/nonexistent/path/tokenizer.json"));
    assert!(result.is_err(), "nonexistent file must return Err");
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// token_to_piece for IDs 0–127 returns a non-empty string (printable ASCII or control escape).
    #[test]
    fn prop_token_to_piece_always_some_in_byte_range(id in 0u32..256u32) {
        let tok = BasicTokenizer::new();
        prop_assert!(tok.token_to_piece(id).is_some());
    }

    /// For IDs ≥ 256 that are not special tokens, token_to_piece returns the <token_N> format.
    #[test]
    fn prop_token_to_piece_high_id_format(id in 300u32..10_000u32) {
        let tok = BasicTokenizer::new();
        let piece = tok.token_to_piece(id).unwrap();
        prop_assert!(
            piece.starts_with("<token_") && piece.ends_with('>'),
            "expected <token_N> format for id={}, got {:?}",
            id, piece
        );
    }

    /// real_vocab_size always equals vocab_size for BasicTokenizer.
    #[test]
    fn prop_real_vocab_size_equals_vocab_size(vs in 1usize..200_000usize) {
        let tok = BasicTokenizer::with_config(vs, None, None, None);
        prop_assert_eq!(tok.real_vocab_size(), tok.vocab_size());
    }

    /// Encode of a single printable ASCII character produces exactly one token.
    #[test]
    fn prop_single_ascii_byte_encodes_to_one_token(c in b'!'..=b'~') {
        let tok = BasicTokenizer::new();
        let s = String::from_utf8(vec![c]).unwrap();
        let tokens = tok.encode(&s, false, false).unwrap();
        prop_assert_eq!(tokens.len(), 1);
        prop_assert_eq!(tokens[0], c as u32);
    }

    /// decode(encode(text, false, false)) == text for printable ASCII.
    #[test]
    fn prop_encode_decode_roundtrip_no_special(text in "[!-~]{1,80}") {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false).unwrap();
        let decoded = tok.decode(&tokens).unwrap();
        prop_assert_eq!(decoded, text);
    }
}
