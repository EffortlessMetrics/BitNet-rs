//! Property-based tests — wave 16.
//!
//! BasicTokenizer encode/decode roundtrip, config field preservation,
//! EOS/BOS token handling, and vocabulary bounds.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};
use proptest::prelude::*;

// ── Encode/decode roundtrip ─────────────────────────────────────────────────

proptest! {
    /// Encoding then decoding ASCII text recovers the original.
    #[test]
    fn encode_decode_roundtrip_ascii(text in "[a-zA-Z]{1,30}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        let decoded = t.decode(&tokens).expect("decode");
        prop_assert_eq!(decoded.trim(), text.trim(),
            "roundtrip failed for '{}'", text);
    }

    /// Encoding non-empty text produces at least one token.
    #[test]
    fn encode_non_empty_text_non_empty_tokens(text in "[a-zA-Z0-9 ]{1,50}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        prop_assert!(!tokens.is_empty(),
            "encoding '{}' produced empty tokens", text);
    }

    /// Decoding empty tokens produces empty string.
    #[test]
    fn decode_empty_tokens(_dummy in 0u8..1) {
        let t = BasicTokenizer::new();
        let decoded = t.decode(&[]).expect("decode");
        prop_assert!(decoded.is_empty(),
            "decoding empty tokens should give empty string, got '{}'", decoded);
    }
}

// ── BOS / EOS token handling ────────────────────────────────────────────────

proptest! {
    /// add_bos prepends the BOS token ID.
    #[test]
    fn encode_bos_prepends_token(
        text in "[a-z]{1,20}",
        bos_id in 0u32..256,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = t.encode(&text, true, false).expect("encode with bos");
        prop_assert_eq!(tokens[0], bos_id,
            "first token should be BOS {}", bos_id);
    }

    /// add_special appends the EOS token ID.
    #[test]
    fn encode_add_special_appends_eos(
        text in "[a-z]{1,20}",
        eos_id in 256u32..500,
    ) {
        let t = BasicTokenizer::with_config(50257, None, Some(eos_id), None);
        let tokens = t.encode(&text, false, true).expect("encode with eos");
        prop_assert_eq!(*tokens.last().unwrap(), eos_id,
            "last token should be EOS {}", eos_id);
    }

    /// BOS + add_special together add exactly 2 extra tokens (BOS + EOS).
    #[test]
    fn encode_bos_eos_adds_two(
        text in "[a-z]{1,20}",
        bos_id in 0u32..256,
        eos_id in 256u32..500,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), Some(eos_id), None);
        let bare = t.encode(&text, false, false).expect("bare");
        let both = t.encode(&text, true, true).expect("both");
        prop_assert_eq!(both.len(), bare.len() + 2);
        prop_assert_eq!(both[0], bos_id);
        prop_assert_eq!(*both.last().unwrap(), eos_id);
    }

    /// Without bos_token_id set, add_bos is a no-op.
    #[test]
    fn encode_bos_noop_when_unset(text in "[a-z]{1,20}") {
        let t = BasicTokenizer::with_config(50257, None, None, None);
        let without = t.encode(&text, false, false).expect("without bos");
        let with = t.encode(&text, true, false).expect("with bos");
        prop_assert_eq!(without, with);
    }
}

// ── Token ID bounds ─────────────────────────────────────────────────────────

proptest! {
    /// All token IDs from encode are within [0, vocab_size).
    #[test]
    fn encode_ids_within_vocab(text in "[a-zA-Z ]{1,30}") {
        let t = BasicTokenizer::new(); // vocab_size = 50257
        let vocab_size = t.vocab_size();
        let tokens = t.encode(&text, false, false).expect("encode");
        for &id in &tokens {
            prop_assert!((id as usize) < vocab_size,
                "token id {} >= vocab_size {}", id, vocab_size);
        }
    }

    /// vocab_size is preserved by with_config.
    #[test]
    fn vocab_size_preserved(vocab_size in 256usize..200_000) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(t.vocab_size(), vocab_size);
    }

    /// Byte-level encoding fails when vocab_size is too small for the byte.
    #[test]
    fn encode_small_vocab_fails(vocab_size in 1usize..97) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        // 'a' = 97, so any vocab_size < 97 must reject it
        let result = t.encode("a", false, false);
        prop_assert!(result.is_err());
    }
}

// ── TokenizerConfig properties ──────────────────────────────────────────────

proptest! {
    /// Config Default has sane initial values.
    #[test]
    fn config_default_sane(_dummy in 0u8..1) {
        let cfg = TokenizerConfig::new();
        prop_assert_eq!(cfg.vocab_size, 0);
        prop_assert!(!cfg.add_bos);
        prop_assert!(!cfg.add_eos);
    }

    /// Config struct preserves all fields via construction.
    #[test]
    fn config_fields_roundtrip(
        vocab_size in 1usize..100_000,
        add_bos in any::<bool>(),
        add_eos in any::<bool>(),
        bos_id in proptest::option::of(0u32..1000),
        eos_id in proptest::option::of(0u32..1000),
    ) {
        let cfg = TokenizerConfig {
            model_type: "test".to_string(),
            vocab_size,
            add_bos,
            add_eos,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            ..Default::default()
        };
        prop_assert_eq!(cfg.model_type, "test");
        prop_assert_eq!(cfg.vocab_size, vocab_size);
        prop_assert_eq!(cfg.add_bos, add_bos);
        prop_assert_eq!(cfg.add_eos, add_eos);
        prop_assert_eq!(cfg.bos_token_id, bos_id);
        prop_assert_eq!(cfg.eos_token_id, eos_id);
    }

    /// model_type string is preserved.
    #[test]
    fn config_model_type_preserved(model_type in "[a-z]{1,20}") {
        let cfg = TokenizerConfig {
            model_type: model_type.clone(),
            ..Default::default()
        };
        prop_assert_eq!(cfg.model_type, model_type);
    }
}

// ── Encoding idempotency (same input → same output) ────────────────────────

proptest! {
    /// Encoding the same text twice gives the same tokens (deterministic).
    #[test]
    fn encode_deterministic(text in "[a-zA-Z0-9 ]{1,30}") {
        let t = BasicTokenizer::new();
        let t1 = t.encode(&text, false, false).expect("encode 1");
        let t2 = t.encode(&text, false, false).expect("encode 2");
        prop_assert_eq!(t1, t2);
    }

    /// token_to_piece returns Some for all byte-range tokens.
    #[test]
    fn token_to_piece_byte_range(id in 0u32..256) {
        let t = BasicTokenizer::new();
        let piece = t.token_to_piece(id);
        prop_assert!(piece.is_some(),
            "token_to_piece({}) should return Some", id);
    }

    /// token_to_piece is non-empty for all IDs.
    #[test]
    fn token_to_piece_non_empty(id in 0u32..1000) {
        let t = BasicTokenizer::new();
        if let Some(piece) = t.token_to_piece(id) {
            prop_assert!(!piece.is_empty(),
                "token_to_piece({}) returned empty string", id);
        }
    }
}
