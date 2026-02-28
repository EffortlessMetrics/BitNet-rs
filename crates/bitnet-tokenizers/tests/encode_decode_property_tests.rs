//! Property-based tests for tokenizer encode/decode properties.
//!
//! Key invariants tested:
//! - `MockTokenizer`: encode → decode round-trip for ASCII input yields the
//!   original string (byte-level encoding is lossless for ASCII)
//! - All token IDs produced by `encode()` are within `[0, vocab_size)`
//! - `decode()` never panics on arbitrary token ID slices
//! - `vocab_size()` returns a positive value and is stable across calls
//! - `token_to_piece()` returns `Some` for IDs 0–255 (byte tokens)
//! - Encoding an empty string produces an empty token vector

use bitnet_tokenizers::MockTokenizer;
use bitnet_tokenizers::Tokenizer;
use proptest::prelude::*;

// ── Round-trip ───────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Encoding ASCII text and decoding it recovers the original string.
    /// MockTokenizer uses byte-level encoding, so this is lossless for ASCII.
    #[test]
    fn prop_encode_decode_roundtrip_ascii(text in "[a-zA-Z0-9 ,.!?]{0,200}") {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false).unwrap();
        let decoded = tok.decode(&tokens).unwrap();
        prop_assert_eq!(
            &decoded, &text,
            "round-trip must recover ASCII input"
        );
    }

    /// Encoding UTF-8 text and decoding it recovers the original bytes.
    /// MockTokenizer uses byte-level encoding for all UTF-8 input.
    #[test]
    fn prop_encode_decode_roundtrip_utf8(text in "\\PC{0,100}") {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false).unwrap();
        let decoded = tok.decode(&tokens).unwrap();
        prop_assert_eq!(
            &decoded, &text,
            "round-trip must recover UTF-8 input"
        );
    }
}

// ── Token ID range ───────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// All token IDs produced by encode() are < vocab_size.
    #[test]
    fn prop_token_ids_in_range(text in "[a-zA-Z0-9]{1,100}") {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let tokens = tok.encode(&text, false, false).unwrap();
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab,
                "token id {} must be < vocab_size {}", id, vocab
            );
        }
    }
}

// ── decode never panics ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// decode() on arbitrary token IDs never panics (returns Ok).
    #[test]
    fn prop_decode_never_panics(
        tokens in prop::collection::vec(any::<u32>(), 0..64),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&tokens);
        prop_assert!(result.is_ok(), "decode must not panic or error");
    }
}

// ── vocab_size consistency ───────────────────────────────────────────────────

proptest! {
    /// vocab_size is stable across calls and positive.
    #[test]
    fn prop_vocab_size_stable(_seed in 0u8..10) {
        let tok = MockTokenizer::new();
        let v1 = tok.vocab_size();
        let v2 = tok.vocab_size();
        prop_assert!(v1 > 0, "vocab_size must be positive");
        prop_assert_eq!(v1, v2, "vocab_size must be stable");
    }
}

// ── token_to_piece for byte range ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// token_to_piece returns Some for all byte-range IDs (0..256).
    #[test]
    fn prop_token_to_piece_byte_range(id in 0u32..256) {
        let tok = MockTokenizer::new();
        let piece = tok.token_to_piece(id);
        prop_assert!(
            piece.is_some(),
            "token_to_piece({}) must return Some for byte-range token", id
        );
    }
}

// ── Empty input ──────────────────────────────────────────────────────────────

proptest! {
    /// Encoding an empty string produces an empty token vector.
    #[test]
    fn prop_empty_input_empty_tokens(_seed in 0u8..1) {
        let tok = MockTokenizer::new();
        let tokens = tok.encode("", false, false).unwrap();
        prop_assert!(tokens.is_empty(), "empty input must produce empty tokens");
    }
}
