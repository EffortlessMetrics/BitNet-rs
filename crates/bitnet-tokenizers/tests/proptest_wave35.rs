//! Property-based tests — wave 35.
//!
//! 10 properties covering token encode/decode invariants, special token
//! handling, and vocabulary bounds.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};
use proptest::prelude::*;

// ===================================================================
// 1–3. Token encode/decode invariants
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Encoding then decoding ASCII text recovers the original string.
    #[test]
    fn prop_encode_decode_ascii_roundtrip(text in "[a-zA-Z]{1,40}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        let decoded = t.decode(&tokens).expect("decode");
        prop_assert_eq!(
            decoded.trim(), text.trim(),
            "roundtrip failed for '{}'", text
        );
    }

    /// Encoding non-empty text always produces at least one token.
    #[test]
    fn prop_encode_nonempty_produces_tokens(text in "[a-zA-Z0-9]{1,50}") {
        let t = BasicTokenizer::new();
        let tokens = t.encode(&text, false, false).expect("encode");
        prop_assert!(
            !tokens.is_empty(),
            "encode('{}') produced empty tokens", text
        );
    }

    /// Encoding is deterministic: same input → same output.
    #[test]
    fn prop_encode_deterministic(text in "[a-zA-Z0-9 ]{1,30}") {
        let t = BasicTokenizer::new();
        let t1 = t.encode(&text, false, false).expect("encode 1");
        let t2 = t.encode(&text, false, false).expect("encode 2");
        prop_assert_eq!(t1, t2);
    }
}

// ===================================================================
// 4–6. Special token handling
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// BOS token is prepended when add_bos is true.
    #[test]
    fn prop_bos_prepended(
        text in "[a-z]{1,20}",
        bos_id in 0u32..256,
    ) {
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = t.encode(&text, true, false).expect("encode bos");
        prop_assert_eq!(
            tokens[0], bos_id,
            "first token should be BOS={}", bos_id
        );
    }

    /// EOS token is appended when add_special is true.
    #[test]
    fn prop_eos_appended(
        text in "[a-z]{1,20}",
        eos_id in 256u32..500,
    ) {
        let t = BasicTokenizer::with_config(50257, None, Some(eos_id), None);
        let tokens = t.encode(&text, false, true).expect("encode eos");
        prop_assert_eq!(
            *tokens.last().unwrap(), eos_id,
            "last token should be EOS={}", eos_id
        );
    }

    /// BOS + EOS together add exactly 2 tokens compared to bare encoding.
    #[test]
    fn prop_bos_eos_add_two(
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
}

// ===================================================================
// 7–10. Vocabulary bounds
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// All encoded token IDs are within [0, vocab_size).
    #[test]
    fn prop_token_ids_within_vocab(text in "[a-zA-Z ]{1,30}") {
        let t = BasicTokenizer::new();
        let vocab = t.vocab_size();
        let tokens = t.encode(&text, false, false).expect("encode");
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab,
                "token id {} >= vocab_size {}", id, vocab
            );
        }
    }

    /// with_config preserves the specified vocab_size.
    #[test]
    fn prop_vocab_size_preserved(vocab in 256usize..200_000) {
        let t = BasicTokenizer::with_config(vocab, None, None, None);
        prop_assert_eq!(t.vocab_size(), vocab);
    }

    /// TokenizerConfig default has vocab_size 0.
    #[test]
    fn prop_config_default_vocab_zero(_dummy in 0u8..1) {
        let cfg = TokenizerConfig::new();
        prop_assert_eq!(cfg.vocab_size, 0);
    }

    /// TokenizerConfig field assignment round-trips.
    #[test]
    fn prop_config_fields_roundtrip(
        vocab_size in 1usize..100_000,
        add_bos in any::<bool>(),
        add_eos in any::<bool>(),
        bos_id in proptest::option::of(0u32..1000),
        eos_id in proptest::option::of(0u32..1000),
    ) {
        let cfg = TokenizerConfig {
            model_type: "wave35".to_string(),
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
        prop_assert_eq!(cfg.model_type, "wave35");
    }
}
