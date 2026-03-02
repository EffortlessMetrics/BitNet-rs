//! Property-based tests for tokenizer encode/decode roundtrips and token ID validity
//! (proptest wave 31).

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};
use proptest::prelude::*;

// ── Encode → decode roundtrip ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// ASCII encode→decode roundtrip preserves the original string.
    #[test]
    fn ascii_roundtrip(text in "[a-zA-Z0-9 ]{1,50}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        let decoded = t.decode(&tokens).expect("decode");
        prop_assert_eq!(&decoded, &text, "roundtrip failed");
    }

    /// Single-byte ASCII characters survive roundtrip individually.
    #[test]
    fn single_ascii_roundtrip(ch in 0u8..128) {
        let text = String::from(ch as char);
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        let decoded = t.decode(&tokens).expect("decode");
        prop_assert_eq!(&decoded, &text);
    }

    /// Encode→decode with BOS strips the BOS from decoded output.
    #[test]
    fn roundtrip_with_bos(text in "[a-z]{1,20}", bos_id in 200u32..300) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), Some(50256), None);
        let tokens = t.encode(&text, true, false).expect("encode");
        // First token should be BOS
        prop_assert_eq!(tokens[0], bos_id);
        let decoded = t.decode(&tokens).expect("decode");
        // BOS is skipped during decode (it's a special token)
        prop_assert_eq!(&decoded, &text);
    }

    /// Empty string encodes to empty token list.
    #[test]
    fn empty_string_encode(_dummy in 0u8..1) {
        let t = BasicTokenizer::new();
        let tokens = t.encode("", false, false).expect("encode");
        prop_assert!(tokens.is_empty());
    }

    /// Empty token list decodes to empty string.
    #[test]
    fn empty_tokens_decode(_dummy in 0u8..1) {
        let t = BasicTokenizer::new();
        let decoded = t.decode(&[]).expect("decode");
        prop_assert!(decoded.is_empty());
    }
}

// ── Token IDs in valid range ──────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// All encoded token IDs are < vocab_size (excluding special tokens).
    #[test]
    fn token_ids_in_range(
        text in "[a-zA-Z0-9]{1,40}",
        vocab_size in 256usize..100_000,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        let tokens = t.encode(&text, false, false).expect("encode");
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab_size,
                "token ID {id} >= vocab_size {vocab_size}"
            );
        }
    }

    /// Token IDs from ASCII encoding are all < 128.
    #[test]
    fn ascii_token_ids_bounded(text in "[a-zA-Z0-9 ]{1,50}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        for &id in &tokens {
            prop_assert!(id < 128, "ASCII token ID {id} should be < 128");
        }
    }

    /// BOS token ID, when present, is the first token.
    #[test]
    fn bos_is_first_token(
        text in "[a-z]{1,20}",
        bos_id in 128u32..256,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = t.encode(&text, true, false).expect("encode");
        prop_assert!(!tokens.is_empty());
        prop_assert_eq!(tokens[0], bos_id, "first token should be BOS");
    }

    /// EOS token ID, when added via add_special, is the last token.
    #[test]
    fn eos_is_last_token(
        text in "[a-z]{1,20}",
        eos_id in 128u32..256,
    ) {
        let t = BasicTokenizer::with_config(50257, None, Some(eos_id), None);
        let tokens = t.encode(&text, false, true).expect("encode");
        prop_assert!(!tokens.is_empty());
        prop_assert_eq!(*tokens.last().unwrap(), eos_id, "last token should be EOS");
    }

    /// Token count equals text byte length (for ASCII, no specials).
    #[test]
    fn token_count_equals_byte_len(text in "[a-zA-Z0-9]{1,50}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        prop_assert_eq!(tokens.len(), text.len(),
            "BasicTokenizer should emit one token per byte");
    }
}

// ── Config preservation ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// TokenizerConfig fields survive construction.
    #[test]
    fn config_round_trip(
        vocab_size in 100usize..200_000,
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
    }

    /// BasicTokenizer::with_config preserves all fields.
    #[test]
    fn basic_tokenizer_config_preserved(
        vocab_size in 256usize..100_000,
        bos in proptest::option::of(0u32..256),
        eos in proptest::option::of(0u32..256),
        pad in proptest::option::of(0u32..256),
    ) {
        let t = BasicTokenizer::with_config(vocab_size, bos, eos, pad);
        prop_assert_eq!(t.vocab_size(), vocab_size);
        prop_assert_eq!(t.bos_token_id(), bos);
        prop_assert_eq!(t.eos_token_id(), eos);
    }

    /// Vocab size from with_config matches vocab_size().
    #[test]
    fn vocab_size_consistent(vs in 1usize..200_000) {
        let t = BasicTokenizer::with_config(vs, None, None, None);
        prop_assert_eq!(t.vocab_size(), vs);
    }
}
