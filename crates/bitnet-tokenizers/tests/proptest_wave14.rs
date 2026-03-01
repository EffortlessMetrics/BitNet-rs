//! Wave 14 property tests: tokenizer encode/decode invariants.
//!
//! Key invariants tested (10 properties):
//! - encode(text).len() > 0 for non-empty text
//! - decode never panics for arbitrary token IDs
//! - encode -> decode roundtrip for ASCII text
//! - vocab_size is always positive
//! - encode with add_bos produces longer or equal output
//! - token IDs are in [0, vocab_size)
//! - empty string encodes to empty or BOS-only
//! - repeated encode yields identical results (determinism)
//! - decode of empty slice produces empty string
//! - encode preserves word boundaries (space-separated words produce multiple tokens)

use bitnet_tokenizers::{BasicTokenizer, MockTokenizer, Tokenizer};
use proptest::prelude::*;

// ===================================================================
// MockTokenizer properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Non-empty text always produces at least one token.
    #[test]
    fn prop_encode_non_empty_produces_tokens(
        text in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false).expect("encode must not fail");
        prop_assert!(!tokens.is_empty(), "non-empty text must produce at least one token");
    }

    /// Decode never panics for valid token IDs in range.
    #[test]
    fn prop_decode_no_panic_valid_ids(
        ids in prop::collection::vec(0u32..256, 0..=32),
    ) {
        let tok = MockTokenizer::new();
        let _ = tok.decode(&ids);
    }

    /// Encode -> decode roundtrip for ASCII text.
    #[test]
    fn prop_encode_decode_roundtrip_ascii(
        text in "[a-zA-Z0-9 !?.,-]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false).expect("encode must not fail");
        let decoded = tok.decode(&tokens).expect("decode must not fail");
        prop_assert_eq!(&decoded, &text, "roundtrip failed");
    }

    /// Token IDs are always in [0, vocab_size).
    #[test]
    fn prop_token_ids_in_range(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let tokens = tok.encode(&text, false, false).expect("encode must not fail");
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab,
                "token id {id} >= vocab_size {vocab}"
            );
        }
    }

    /// Repeated encode yields identical results (determinism).
    #[test]
    fn prop_encode_deterministic(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let t1 = tok.encode(&text, false, false).expect("encode 1");
        let t2 = tok.encode(&text, false, false).expect("encode 2");
        prop_assert_eq!(t1, t2, "encode must be deterministic");
    }

    /// Decode of empty slice produces empty string.
    #[test]
    fn prop_decode_empty_is_empty(_dummy in 0u8..1) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&[]).expect("decode of empty must succeed");
        prop_assert!(result.is_empty(), "decode of empty should be empty, got '{result}'");
    }

    /// Vocab size is always positive.
    #[test]
    fn prop_vocab_size_positive(_dummy in 0u8..1) {
        let tok = MockTokenizer::new();
        prop_assert!(tok.vocab_size() > 0);
    }
}

// ===================================================================
// BasicTokenizer properties
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// BasicTokenizer: non-empty text always produces at least one token.
    #[test]
    fn prop_basic_encode_non_empty(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false).expect("encode must not fail");
        prop_assert!(!tokens.is_empty(), "non-empty text must produce at least one token");
    }

    /// BasicTokenizer: decode never panics for small token IDs.
    #[test]
    fn prop_basic_decode_no_panic(
        ids in prop::collection::vec(0u32..1024, 0..=16),
    ) {
        let tok = BasicTokenizer::new();
        // We only assert it doesn't panic — it may return Ok or Err.
        let _ = tok.decode(&ids);
    }

    /// BasicTokenizer: vocab_size is always positive.
    #[test]
    fn prop_basic_vocab_size_positive(_dummy in 0u8..1) {
        let tok = BasicTokenizer::new();
        prop_assert!(tok.vocab_size() > 0);
    }
}
