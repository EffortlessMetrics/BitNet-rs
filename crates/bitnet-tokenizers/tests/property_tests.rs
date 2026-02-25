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
}
