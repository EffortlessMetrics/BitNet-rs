//! Expanded property-based tests for `bitnet-tokenizers` -- invariant coverage.
//!
//! These tests complement `property_tests.rs` by covering `MockTokenizer`,
//! `TokenizerBuilder::from_pretrained`, the `pad_token_id` accessor,
//! `real_vocab_size`, `get_family_name`, full byte-range `token_to_piece`,
//! `unk_token_id` serialisation, and the legacy encode/decode shims.
//!
//! All tests run without any model files on disk.

use bitnet_tokenizers::{
    BasicTokenizer, MockTokenizer, Tokenizer, TokenizerBuilder, TokenizerConfig,
};
use proptest::prelude::*;

proptest! {
    // 1. MockTokenizer encode -> decode roundtrip
    #[test]
    fn mock_tokenizer_encode_decode_roundtrip(
        text in "[a-zA-Z0-9 !?,.:;-]{1,80}",
    ) {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false)
            .expect("MockTokenizer::encode must not fail");
        prop_assert!(!tokens.is_empty(), "non-empty text must produce at least one token");
        let recovered = tok.decode(&tokens)
            .expect("MockTokenizer::decode must not fail");
        prop_assert_eq!(recovered, text, "encode->decode must recover original text");
    }

    // 2. Token IDs in [0, vocab_size)
    #[test]
    fn mock_tokenizer_token_ids_in_valid_range(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &id in &tokens {
            prop_assert!(
                (id as usize) < vocab,
                "token ID {} is outside vocab range [0, {})",
                id,
                vocab
            );
        }
    }

    // 3. Empty string never panics
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

    // 4. Configured special token IDs are in [0, vocab_size)
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
            prop_assert!((bos as usize) < tok.vocab_size());
        }
        if let Some(eos) = tok.eos_token_id() {
            prop_assert!((eos as usize) < tok.vocab_size());
        }
    }

    // 5. Registered special token lookup
    #[test]
    fn mock_tokenizer_special_token_lookup_matches_registration(
        token_id in 0u32..50257u32,
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("<bos>", token_id)]);
        prop_assert_eq!(tok.token_to_id("<bos>"), Some(token_id));
    }

    // 6. Byte-level IDs are < 256
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

    // 7. decode never panics
    #[test]
    fn mock_tokenizer_decode_never_panics(
        ids in prop::collection::vec(0u32..50257u32, 0..100),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&ids);
        prop_assert!(result.is_ok(), "decode must not fail; got {:?}", result.err());
    }

    // 8. Absent token returns None
    #[test]
    fn mock_tokenizer_absent_token_returns_none(
        suffix in "[a-z]{8,16}",
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("<registered>", 42)]);
        let absent = format!("<absent_{}>", suffix);
        prop_assert!(tok.token_to_id(&absent).is_none());
    }

    // 9. from_pretrained always succeeds
    #[test]
    fn from_pretrained_always_succeeds(
        name in "[a-z]{1,32}",
    ) {
        let result = TokenizerBuilder::from_pretrained(&name);
        prop_assert!(result.is_ok(), "from_pretrained must not fail for any model name");
        prop_assert!(result.unwrap().vocab_size() > 0);
    }

    // 10. pad_token_id accessor
    #[test]
    fn basic_tokenizer_pad_token_id_accessor(
        vocab_size in 1000usize..200_000usize,
        pad_id in 0u32..256u32,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, Some(pad_id));
        prop_assert_eq!(t.pad_token_id(), Some(pad_id));
    }

    // 11. real_vocab_size == vocab_size
    #[test]
    fn basic_tokenizer_real_vocab_size_equals_vocab_size(
        vocab_size in 1usize..500_000usize,
    ) {
        let t = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(t.real_vocab_size(), t.vocab_size());
    }

    // 12. get_family_name == "unknown" for BasicTokenizer
    #[test]
    fn basic_tokenizer_family_name_is_unknown(_dummy in 0u8..1u8) {
        prop_assert_eq!(BasicTokenizer::new().get_family_name(), "unknown");
    }

    // 13. get_family_name == "llama3" with <|eot_id|>
    #[test]
    fn mock_tokenizer_with_eot_id_is_llama3(
        eot_id in 100_000u32..200_000u32,
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("<|eot_id|>", eot_id)]);
        prop_assert_eq!(tok.get_family_name(), "llama3");
    }

    // 14. get_family_name == "mistral-instruct" with [INST]
    #[test]
    fn mock_tokenizer_with_inst_is_mistral(
        inst_id in 50_000u32..100_000u32,
    ) {
        let tok = MockTokenizer::with_special_tokens(&[("[INST]", inst_id)]);
        prop_assert_eq!(tok.get_family_name(), "mistral-instruct");
    }

    // 15. token_to_piece returns Some for bytes 0-255
    #[test]
    fn basic_tokenizer_token_to_piece_some_for_byte_range(
        id in 0u32..256u32,
    ) {
        let t = BasicTokenizer::new();
        prop_assert!(
            t.token_to_piece(id).is_some(),
            "token_to_piece({}) must be Some for byte-range IDs 0-255",
            id
        );
    }

    // 16. unk_token_id JSON round-trip
    #[test]
    fn config_unk_token_id_json_roundtrip(
        vocab_size in 100usize..100_000usize,
        unk_id in proptest::option::of(0u32..1000u32),
    ) {
        let cfg = TokenizerConfig {
            vocab_size,
            unk_token_id: unk_id,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialise must succeed");
        let restored: TokenizerConfig =
            serde_json::from_str(&json).expect("deserialise must succeed");
        prop_assert_eq!(restored.unk_token_id, unk_id);
    }

    // 17. encode_legacy prepends configured BOS
    #[test]
    fn encode_legacy_prepends_configured_bos(
        text in "[a-zA-Z]{1,48}",
        bos_id in 0u32..128u32,
    ) {
        // bos_id < 128 keeps it within the byte-level encoding range
        let t = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = t.encode_legacy(&text, false).expect("encode_legacy must succeed");
        prop_assert!(!tokens.is_empty());
        prop_assert_eq!(tokens[0], bos_id);
    }

    // 18. decode_legacy skip flag has no effect
    #[test]
    fn decode_legacy_skip_flag_has_no_effect(
        ids in prop::collection::vec(0u32..256u32, 1..50),
    ) {
        let t = BasicTokenizer::new();
        let a = t.decode_legacy(&ids, true).expect("decode_legacy(true)");
        let b = t.decode_legacy(&ids, false).expect("decode_legacy(false)");
        prop_assert_eq!(a, b);
    }
}
