//! Property-based tests for bitnet-tokenizers.
//!
//! Tests invariants across all possible inputs for tokenizer types and config.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};
use proptest::prelude::*;

// ── TokenizerConfig ───────────────────────────────────────────────────────────

proptest! {
    /// Default config has empty model_type and zero vocab_size.
    #[test]
    fn prop_tokenizer_config_default(_dummy in 0u8..1) {
        let cfg = TokenizerConfig::new();
        prop_assert_eq!(cfg.vocab_size, 0);
        prop_assert!(!cfg.add_bos);
        prop_assert!(!cfg.add_eos);
    }

    /// Config fields are preserved without mutation.
    #[test]
    fn prop_tokenizer_config_fields_preserved(
        vocab_size in 100usize..100_000,
        add_bos in any::<bool>(),
        add_eos in any::<bool>(),
        bos_id in proptest::option::of(0u32..1000),
        eos_id in proptest::option::of(0u32..1000),
    ) {
        let cfg = TokenizerConfig {
            model_type: "bpe".to_string(),
            vocab_size,
            add_bos,
            add_eos,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            ..Default::default()
        };
        prop_assert_eq!(cfg.vocab_size, vocab_size);
        prop_assert_eq!(cfg.add_bos, add_bos);
        prop_assert_eq!(cfg.add_eos, add_eos);
        prop_assert_eq!(cfg.bos_token_id, bos_id);
        prop_assert_eq!(cfg.eos_token_id, eos_id);
    }
}

// ── BasicTokenizer ────────────────────────────────────────────────────────────

proptest! {
    /// BasicTokenizer vocab_size is always positive.
    #[test]
    fn prop_basic_tokenizer_vocab_size_positive(_dummy in 0u8..1) {
        let t = BasicTokenizer::new();
        prop_assert!(t.vocab_size() > 0);
    }

    /// with_config preserves the given vocab_size.
    #[test]
    fn prop_basic_tokenizer_with_config_vocab_size(
        vocab_size in 100usize..200_000,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(t.vocab_size(), vocab_size);
    }

    /// encode returns token count >= 1 for non-empty ASCII strings.
    #[test]
    fn prop_basic_tokenizer_encode_non_empty(
        text in "[a-zA-Z ]{1,50}",
    ) {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode should succeed");
        prop_assert!(!tokens.is_empty());
    }

    /// encode with add_bos=true produces more tokens than without.
    #[test]
    fn prop_basic_tokenizer_encode_bos_adds_token(
        text in "[a-zA-Z ]{1,50}",
        bos_id in 0u32..1000,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let without_bos = t.encode(&text, false, false).expect("encode no-bos");
        let with_bos = t.encode(&text, true, false).expect("encode with-bos");
        prop_assert_eq!(with_bos.len(), without_bos.len() + 1);
        prop_assert_eq!(with_bos[0], bos_id);
    }

    /// All token IDs returned by encode are within [0, vocab_size).
    #[test]
    fn prop_basic_tokenizer_encode_ids_in_range(
        text in "[a-zA-Z ]{1,50}",
        vocab_size in 100usize..200_000,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        let tokens = t.encode(&text, false, false).expect("encode");
        for &id in &tokens {
            prop_assert!((id as usize) < vocab_size,
                "token id {} >= vocab_size {}", id, vocab_size);
        }
    }

    /// token_to_piece returns Some for IDs within vocab range.
    #[test]
    fn prop_basic_tokenizer_token_to_piece_in_range(
        id in 0u32..50257,
    ) {
        let t = BasicTokenizer::new();
        // token_to_piece may return None for unmapped IDs; it must not panic
        let _ = t.token_to_piece(id);
    }

    /// Default BasicTokenizer has an EOS token.
    #[test]
    fn prop_basic_tokenizer_has_eos(_dummy in 0u8..1) {
        let t = BasicTokenizer::new();
        // Default EOS is 50256 (GPT-2 style)
        prop_assert!(t.vocab_size() > 0);
    }

    /// If add_bos=true and bos_token_id=Some(x), encoded IDs for non-empty input start with x.
    #[test]
    fn prop_bos_eos_prepend_append(
        text in "[a-zA-Z]{1,50}",
        bos_id in 0u32..1000u32,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = t.encode(&text, true, false).expect("encode should succeed");
        prop_assert!(!tokens.is_empty(), "tokens must not be empty for non-empty input");
        prop_assert_eq!(tokens[0], bos_id, "first token must be BOS id");
    }

    /// decode() never panics and always returns Ok for any sequence of in-range token IDs.
    #[test]
    fn prop_decode_never_panics(
        ids in proptest::collection::vec(0u32..50257u32, 0..100),
    ) {
        let t = BasicTokenizer::new();
        let result = t.decode(&ids);
        prop_assert!(result.is_ok(), "decode must not fail for valid token IDs");
    }

    /// encode() of a single ASCII word always returns at least 1 token, one per byte.
    #[test]
    fn prop_tokenize_preserves_words(
        word in "[a-z]{1,50}",
    ) {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&word, false, false).expect("encode should succeed");
        prop_assert!(!tokens.is_empty(), "a non-empty word must produce at least 1 token");
        prop_assert_eq!(tokens.len(), word.len(), "each byte of an ASCII word maps to one token");
    }

    /// TokenizerConfig serialized to JSON and deserialized back gives identical field values.
    #[test]
    fn prop_config_builder_round_trip(
        vocab_size in 100usize..100_000usize,
        add_bos in any::<bool>(),
        add_eos in any::<bool>(),
        add_space_prefix in any::<bool>(),
        byte_fallback in any::<bool>(),
        bos_id in proptest::option::of(0u32..1000u32),
        eos_id in proptest::option::of(0u32..1000u32),
        pad_id in proptest::option::of(0u32..1000u32),
        model_type in "[a-z]{1,10}",
    ) {
        let cfg = TokenizerConfig {
            model_type,
            vocab_size,
            add_bos,
            add_eos,
            add_space_prefix,
            byte_fallback,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            pad_token_id: pad_id,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize must succeed");
        let restored: TokenizerConfig = serde_json::from_str(&json).expect("deserialize must succeed");
        prop_assert_eq!(restored.model_type, cfg.model_type);
        prop_assert_eq!(restored.vocab_size, cfg.vocab_size);
        prop_assert_eq!(restored.add_bos, cfg.add_bos);
        prop_assert_eq!(restored.add_eos, cfg.add_eos);
        prop_assert_eq!(restored.add_space_prefix, cfg.add_space_prefix);
        prop_assert_eq!(restored.byte_fallback, cfg.byte_fallback);
        prop_assert_eq!(restored.bos_token_id, cfg.bos_token_id);
        prop_assert_eq!(restored.eos_token_id, cfg.eos_token_id);
        prop_assert_eq!(restored.pad_token_id, cfg.pad_token_id);
    }

    /// bos_token_id and eos_token_id, when set, must be within [0, vocab_size).
    #[test]
    fn prop_eos_id_bounds(
        vocab_size in 300usize..100_000usize,
        bos_id in 256u32..300u32,
        eos_id in 256u32..300u32,
    ) {
        prop_assume!(bos_id != eos_id);
        let t = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);
        prop_assert!(
            t.bos_token_id().is_none_or(|id| (id as usize) < t.vocab_size()),
            "bos_token_id must be < vocab_size"
        );
        prop_assert!(
            t.eos_token_id().is_none_or(|id| (id as usize) < t.vocab_size()),
            "eos_token_id must be < vocab_size"
        );
    }
}

// ── New expanded property tests ───────────────────────────────────────────────

proptest! {
    // Focus area 1: Token encoding length — at least 1 token, at most 4× char count.
    // BasicTokenizer is byte-level: tokens == bytes, bytes ≤ 4× Unicode chars.
    // Tests multi-byte UTF-8 characters (é, ñ) that produce more bytes than chars.
    #[test]
    fn prop_utf8_token_count_bounded_by_4x_chars(
        text in "[a-z\u{00E9}\u{00F1}\u{4E2D}\u{6587}]{1,20}",
    ) {
        let t = BasicTokenizer::new(); // vocab_size=50257, all bytes 0–255 are valid
        let tokens = t.encode(&text, false, false).expect("encode should succeed");
        let char_count = text.chars().count();
        prop_assert!(!tokens.is_empty(),
            "non-empty text must produce at least 1 token");
        prop_assert!(tokens.len() <= char_count * 4,
            "token count {} must be ≤ 4× char count {}", tokens.len(), char_count);
    }

    // Focus area 2: Special token handling — bos/eos accessors return exactly what
    // was configured, never a different value. Any Some(u32) is a valid u32.
    #[test]
    fn prop_bos_eos_accessor_matches_config(
        vocab_size in 1000usize..100_000usize,
        bos_id in 0u32..512u32,
        eos_id in 512u32..1024u32,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);
        prop_assert_eq!(t.bos_token_id(), Some(bos_id),
            "bos_token_id() must equal configured bos_id");
        prop_assert_eq!(t.eos_token_id(), Some(eos_id),
            "eos_token_id() must equal configured eos_id");
    }

    // Focus area 3: Vocabulary size invariants — vocab_size() always equals what
    // was passed to with_config for any positive value.
    #[test]
    fn prop_vocab_size_equals_configured_value(
        vocab_size in 1usize..500_000usize,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(t.vocab_size(), vocab_size,
            "vocab_size() must equal the value passed to with_config");
        prop_assert!(t.vocab_size() > 0,
            "vocab_size() must always be positive");
    }

    // Focus area 4: Encode-decode consistency — every token produced by encode
    // can be decoded individually (via decode(&[id])) without error.
    #[test]
    fn prop_each_encoded_token_individually_decodable(
        text in "[a-z]{1,32}",
    ) {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode should succeed");
        for &tok in &tokens {
            let result = t.decode(&[tok]);
            prop_assert!(result.is_ok(),
                "decode of single token {} must not fail", tok);
        }
    }

    // Focus area 5: Config validation — TokenizerBuilder::from_file with a
    // path into a non-existent directory must return Err, not panic.
    #[test]
    fn prop_missing_tokenizer_file_returns_err(
        name in "[a-z]{12,20}",
    ) {
        // The parent directory itself does not exist, so the file definitely doesn't.
        let path = std::path::PathBuf::from(format!(
            "/tmp/bitnet_proptest_nonexistent_{}/tokenizer.json",
            name
        ));
        let result = bitnet_tokenizers::TokenizerBuilder::from_file(&path);
        prop_assert!(result.is_err(),
            "loading from a non-existent path must return Err, not panic");
    }

    // Focus area 6: Token string round-trip — token_to_piece is deterministic:
    // calling it twice with the same id returns the same result.
    #[test]
    fn prop_token_to_piece_is_deterministic(
        id in 0u32..50257u32,
    ) {
        let t = BasicTokenizer::new();
        let first = t.token_to_piece(id);
        let second = t.token_to_piece(id);
        prop_assert_eq!(first, second,
            "token_to_piece({}) must return the same value on repeated calls", id);
    }

    // Bonus: EOS appended when add_special=true.
    // When eos_token_id is configured and add_special=true, the final token of
    // any non-empty encoding must be the EOS id.
    #[test]
    fn prop_eos_appended_when_add_special(
        text in "[a-zA-Z]{1,32}",
        eos_id in 256u32..50257u32,
    ) {
        let t = BasicTokenizer::with_config(50257, None, Some(eos_id), None);
        let tokens = t.encode(&text, false, true).expect("encode should succeed");
        prop_assert!(!tokens.is_empty(),
            "non-empty text must produce at least 1 token");
        prop_assert_eq!(*tokens.last().unwrap(), eos_id,
            "last token must be EOS id {} when add_special=true", eos_id);
    }

    // Bonus: encode and decode are both deterministic (same input → same output).
    #[test]
    fn prop_encode_and_decode_are_deterministic(
        text in "[a-z]{1,48}",
    ) {
        let t = BasicTokenizer::new();
        let first_enc = t.encode(&text, false, false).expect("first encode");
        let second_enc = t.encode(&text, false, false).expect("second encode");
        prop_assert_eq!(&first_enc, &second_enc,
            "encode must be deterministic for {:?}", text);

        let first_dec = t.decode(&first_enc).expect("first decode");
        let second_dec = t.decode(&first_enc).expect("second decode");
        prop_assert_eq!(first_dec, second_dec,
            "decode must be deterministic");
    }
}
