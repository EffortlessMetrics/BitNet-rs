//! Property-based tests for `bitnet-tokenizers` – invariant coverage via
//! `MockTokenizer` and `BasicTokenizer`.
//!
//! Invariants tested:
//!   1. `MockTokenizer` encode → decode roundtrip: original text is recovered.
//!   2. All token IDs returned by `encode` are in `[0, vocab_size)`.
//!   3. Encoding an empty string never panics and returns `Ok`.
//!   4. Configured BOS / EOS special tokens have IDs within `[0, vocab_size)`.

use bitnet_tokenizers::{BasicTokenizer, MockTokenizer, Tokenizer};
use proptest::prelude::*;

proptest! {
    /// `MockTokenizer` is byte-level: `encode` maps each UTF-8 byte to its
    /// numeric value, `decode` reconstructs those bytes into a string.
    /// For valid UTF-8 text the roundtrip must reproduce the original exactly.
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
        prop_assert_eq!(recovered, text, "encode→decode roundtrip must reproduce original text");
    }

    /// Every token ID produced by `MockTokenizer::encode` must be strictly less
    /// than `vocab_size()` — no ID falls outside the vocabulary.
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
                "token ID {} is outside vocab range [0, {})",
                id,
                vocab
            );
        }
    }

    /// Encoding an empty string must **never panic** and must return `Ok`.
    /// The flags `add_bos` and `add_special` may vary freely; the call must
    /// still complete without error.
    #[test]
    fn empty_string_encoding_never_panics(
        add_bos in any::<bool>(),
        add_special in any::<bool>(),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.encode("", add_bos, add_special);
        prop_assert!(
            result.is_ok(),
            "encoding an empty string must succeed; got Err: {:?}",
            result.err()
        );
    }

    /// When BOS and EOS token IDs are configured on a `BasicTokenizer`, both
    /// must be within `[0, vocab_size)` — they are valid indices into the
    /// vocabulary.
    #[test]
    fn special_token_ids_are_in_valid_range(
        vocab_size in 512usize..100_000usize,
        bos_id in 0u32..256u32,
        eos_id in 256u32..512u32,
    ) {
        // Ensure distinct, non-overlapping IDs
        prop_assume!(bos_id != eos_id);
        prop_assume!((bos_id as usize) < vocab_size);
        prop_assume!((eos_id as usize) < vocab_size);

        let tok = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);

        if let Some(bos) = tok.bos_token_id() {
            prop_assert!(
                (bos as usize) < tok.vocab_size(),
                "BOS token ID {} must be < vocab_size {}",
                bos,
                tok.vocab_size()
            );
        }
        if let Some(eos) = tok.eos_token_id() {
            prop_assert!(
                (eos as usize) < tok.vocab_size(),
                "EOS token ID {} must be < vocab_size {}",
                eos,
                tok.vocab_size()
            );
        }
    }

    /// `MockTokenizer::with_special_tokens` resolves registered token strings
    /// to their IDs via `token_to_id`.  The returned ID must be exactly the
    /// one supplied at construction time.
    #[test]
    fn mock_tokenizer_special_token_lookup_matches_registration(
        token_id in 0u32..50257u32,
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("<bos>", token_id)]);
        let resolved = tok.token_to_id("<bos>");
        prop_assert_eq!(
            resolved,
            Some(token_id),
            "token_to_id must return the registered ID for <bos>"
        );
    }

    /// Token IDs produced by `BasicTokenizer::encode` (byte-level, no special
    /// tokens) are always byte values, i.e. in `[0, 256)` ⊆ `[0, vocab_size)`.
    #[test]
    fn basic_tokenizer_byte_ids_in_byte_range(
        text in "[a-z]{1,64}",
    ) {
        let tok = BasicTokenizer::new(); // vocab_size = 50257 ≥ 256
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &id in &tokens {
            prop_assert!(
                id < 256,
                "byte-level encoding must produce IDs < 256; got {}",
                id
            );
        }
    }

    /// `MockTokenizer::decode` must always return `Ok` and never panic for any
    /// sequence of arbitrary u32 values in `[0, vocab_size)`.
    #[test]
    fn mock_tokenizer_decode_never_panics(
        ids in prop::collection::vec(0u32..50257u32, 0..100),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&ids);
        prop_assert!(result.is_ok(), "decode must not fail; got {:?}", result.err());
    }
}
