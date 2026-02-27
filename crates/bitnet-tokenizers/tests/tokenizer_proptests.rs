//! Property-based tests for `bitnet-tokenizers` – invariant coverage.
//!
//! Invariants tested:
//!   1. MockTokenizer encode -> decode roundtrip recovers original text.
//!   2. All token IDs returned by encode are in [0, vocab_size).
//!   3. Encoding an empty string never panics and returns Ok.
//!   4. Configured BOS / EOS special token IDs are within [0, vocab_size).
//!   5. Registered special tokens resolve correctly via token_to_id.
//!   6. Byte-level token IDs are always in [0, 256).
//!   7. decode never panics for arbitrary valid token ID sequences.

use bitnet_tokenizers::{BasicTokenizer, MockTokenizer, Tokenizer};
use proptest::prelude::*;

proptest! {
    /// MockTokenizer is byte-level: encode maps each UTF-8 byte to its value,
    /// decode reconstructs those bytes. The roundtrip must recover the original text.
    #[test]
    fn mock_tokenizer_encode_decode_roundtrip(
        text in "[a-zA-Z0-9 !?,.:;'-]{1,80}",
    ) {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false)
            .expect("MockTokenizer::encode must not fail");
        prop_assert!(!tokens.is_empty(), "non-empty text must produce at least one token");
        let recovered = tok.decode(&tokens)
            .expect("MockTokenizer::decode must not fail");
        prop_assert_eq!(recovered, text, "encode->decode must recover original text");
    }

    /// Every token ID produced by MockTokenizer::encode must be strictly less
    /// than vocab_size() -- no ID escapes the vocabulary bounds.
    #[test]
    fn mock_tokenizer_token_ids_in_valid_range(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let tokens = tok.encode(&text, false, false)
            .expect("encode must succeed");
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab,
                "token ID {} is outside vocab range [0, {})", id, vocab
            );
        }
    }

    /// Encoding an empty string with any combination of flags must never panic
    /// and must return Ok (not Err).
    #[test]
    fn empty_string_encoding_never_panics(
        add_bos in any::<bool>(),
        add_special in any::<bool>(),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.encode("", add_bos, add_special);
        prop_assert!(
            result.is_ok(),
            "encoding empty string must succeed; got Err: {:?}",
            result.err()
        );
    }

    /// When BOS and EOS token IDs are explicitly configured, both accessors must
    /// return IDs within [0, vocab_size) -- they are valid vocabulary indices.
    #[test]
    fn special_token_ids_are_in_valid_range(
        vocab_size in 512usize..100_000usize,
        bos_id in 0u32..256u32,
        eos_id in 256u32..512u32,
    ) {
        prop_assume!((bos_id as usize) < vocab_size);
        prop_assume!((eos_id as usize) < vocab_size);

        let tok = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);
        if let Some(bos) = tok.bos_token_id() {
            prop_assert!(
                (bos as usize) < tok.vocab_size(),
                "BOS id {} must be < vocab_size {}", bos, tok.vocab_size()
            );
        }
        if let Some(eos) = tok.eos_token_id() {
            prop_assert!(
                (eos as usize) < tok.vocab_size(),
                "EOS id {} must be < vocab_size {}", eos, tok.vocab_size()
            );
        }
    }

    /// A token string registered via MockTokenizer::with_special_tokens must be
    /// retrievable by token_to_id with the exact same ID.
    #[test]
    fn mock_tokenizer_special_token_lookup_matches_registration(
        token_id in 0u32..50257u32,
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("<bos>", token_id)]);
        let resolved = tok.token_to_id("<bos>");
        prop_assert_eq!(
            resolved,
            Some(token_id),
            "token_to_id must return registered ID for <bos>"
        );
    }

    /// BasicTokenizer byte-level encoding (ASCII input, no special tokens)
    /// produces only byte-range IDs in [0, 256).
    #[test]
    fn basic_tokenizer_byte_ids_in_byte_range(
        text in "[a-z]{1,64}",
    ) {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &id in &tokens {
            prop_assert!(id < 256, "byte-level encoding must produce IDs < 256; got {}", id);
        }
    }

    /// MockTokenizer::decode must always return Ok and never panic for any
    /// sequence of arbitrary u32 values within [0, vocab_size).
    #[test]
    fn mock_tokenizer_decode_never_panics(
        ids in prop::collection::vec(0u32..50257u32, 0..100),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&ids);
        prop_assert!(result.is_ok(), "decode must not fail; got {:?}", result.err());
    }
}
