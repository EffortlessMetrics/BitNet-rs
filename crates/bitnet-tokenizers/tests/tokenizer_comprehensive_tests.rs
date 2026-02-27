//! Comprehensive property-based and integration tests for `bitnet-tokenizers`.
//!
//! Covers 25+ invariants across:
//! - Encoding/decoding round-trips
//! - Vocabulary consistency and bounds
//! - Special token (BOS/EOS/PAD) validation
//! - Token ID bounds checking
//! - Empty and whitespace input safety
//! - Unicode / multi-byte UTF-8 safety
//! - `TokenizerConfig` field validation
//! - Auto-discovery path pattern matching
//! - Batch encoding length invariants
//! - `MockTokenizer` behaviour
//! - Determinism of encode/decode
//! - `TokenizerBuilder::from_pretrained` built-in profiles
//! - Legacy shim consistency

use bitnet_tokenizers::{
    BasicTokenizer, MockTokenizer, Tokenizer, TokenizerBuilder, TokenizerConfig,
};
use proptest::prelude::*;

// ── 1. Encoding invariants: round-trip ───────────────────────────────────────

proptest! {
    /// ASCII text encodes and then decodes back to the identical string.
    #[test]
    fn prop_ascii_encode_decode_roundtrip(text in "[a-zA-Z0-9 .,!?:;-]{1,80}") {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false)
            .expect("encode must not fail for ASCII input");
        let decoded = tok.decode(&tokens)
            .expect("decode must not fail");
        prop_assert_eq!(decoded, text.clone(), "round-trip failed for {:?}", text);
    }
}

proptest! {
    /// MockTokenizer byte-level encode → decode is loss-less for any printable ASCII.
    #[test]
    fn prop_mock_encode_decode_roundtrip(text in "[a-zA-Z0-9 !?,.:;-]{1,80}") {
        let tok = MockTokenizer::new();
        let tokens = tok.encode(&text, false, false)
            .expect("MockTokenizer encode must not fail");
        prop_assert!(!tokens.is_empty(),
            "non-empty text must produce at least one token");
        let recovered = tok.decode(&tokens)
            .expect("MockTokenizer decode must not fail");
        prop_assert_eq!(recovered, text,
            "MockTokenizer encode→decode must recover original text");
    }
}

// ── 2. Vocabulary consistency ─────────────────────────────────────────────────

proptest! {
    /// `with_config` preserves the exact vocab_size that was supplied.
    #[test]
    fn prop_vocab_size_matches_config(vocab_size in 1usize..500_000usize) {
        let tok = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(tok.vocab_size(), vocab_size,
            "vocab_size() must equal the configured value");
        prop_assert!(tok.vocab_size() > 0, "vocab_size must be positive");
    }
}

proptest! {
    /// Default `BasicTokenizer::new()` always returns a positive vocab_size.
    #[test]
    fn prop_default_vocab_size_is_positive(_dummy in 0u8..2) {
        let tok = BasicTokenizer::new();
        prop_assert!(tok.vocab_size() > 0,
            "default BasicTokenizer must have a positive vocab_size");
    }
}

proptest! {
    /// `real_vocab_size` equals `vocab_size` for `BasicTokenizer` (no padding).
    #[test]
    fn prop_real_vocab_size_equals_vocab_size(vocab_size in 1usize..500_000usize) {
        let tok = BasicTokenizer::with_config(vocab_size, None, None, None);
        prop_assert_eq!(tok.real_vocab_size(), tok.vocab_size(),
            "real_vocab_size must equal vocab_size for BasicTokenizer");
    }
}

// ── 3. Special tokens: BOS / EOS / PAD within vocab range ────────────────────

proptest! {
    /// Configured BOS and EOS IDs are within [0, vocab_size).
    #[test]
    fn prop_special_token_ids_within_vocab_range(
        vocab_size in 512usize..100_000usize,
        bos_id in 0u32..256u32,
        eos_id in 256u32..512u32,
    ) {
        prop_assume!((bos_id as usize) < vocab_size);
        prop_assume!((eos_id as usize) < vocab_size);
        let tok = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);
        if let Some(b) = tok.bos_token_id() {
            prop_assert!((b as usize) < tok.vocab_size(),
                "BOS ID {} must be < vocab_size {}", b, tok.vocab_size());
        }
        if let Some(e) = tok.eos_token_id() {
            prop_assert!((e as usize) < tok.vocab_size(),
                "EOS ID {} must be < vocab_size {}", e, tok.vocab_size());
        }
    }
}

proptest! {
    /// Configured PAD ID is within [0, vocab_size).
    #[test]
    fn prop_pad_token_id_within_vocab_range(
        vocab_size in 1000usize..200_000usize,
        pad_id in 0u32..256u32,
    ) {
        prop_assume!((pad_id as usize) < vocab_size);
        let tok = BasicTokenizer::with_config(vocab_size, None, None, Some(pad_id));
        if let Some(p) = tok.pad_token_id() {
            prop_assert!((p as usize) < tok.vocab_size(),
                "PAD ID {} must be < vocab_size {}", p, tok.vocab_size());
        }
    }
}

proptest! {
    /// `bos_token_id()` and `eos_token_id()` accessors return exactly what was configured.
    #[test]
    fn prop_special_token_accessor_matches_config(
        vocab_size in 1000usize..100_000usize,
        bos_id in 0u32..512u32,
        eos_id in 512u32..1024u32,
    ) {
        let tok = BasicTokenizer::with_config(vocab_size, Some(bos_id), Some(eos_id), None);
        prop_assert_eq!(tok.bos_token_id(), Some(bos_id),
            "bos_token_id() must return exactly the configured value");
        prop_assert_eq!(tok.eos_token_id(), Some(eos_id),
            "eos_token_id() must return exactly the configured value");
    }
}

// ── 4. Token ID bounds: all IDs returned by encode are < vocab_size ──────────

proptest! {
    /// Every token ID produced by `BasicTokenizer::encode` is within [0, vocab_size).
    #[test]
    fn prop_encode_ids_within_vocab_bounds(
        text in "[a-zA-Z ]{1,50}",
        vocab_size in 100usize..200_000usize,
    ) {
        let tok = BasicTokenizer::with_config(vocab_size, None, None, None);
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &id in &tokens {
            prop_assert!((id as usize) < vocab_size,
                "token id {} is >= vocab_size {}", id, vocab_size);
        }
    }
}

proptest! {
    /// Every token ID produced by `MockTokenizer::encode` is within [0, vocab_size).
    #[test]
    fn prop_mock_encode_ids_within_vocab_bounds(text in "[a-zA-Z0-9 ]{1,64}") {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &id in &tokens {
            prop_assert!((id as usize) < vocab,
                "MockTokenizer token ID {} is outside [0, {})", id, vocab);
        }
    }
}

// ── 5. Empty / whitespace input: no panic, returns reasonable output ──────────

proptest! {
    /// Encoding the empty string never panics and always returns Ok.
    #[test]
    fn prop_empty_string_no_panic(
        add_bos in any::<bool>(),
        add_special in any::<bool>(),
    ) {
        let tok = BasicTokenizer::new();
        let result = tok.encode("", add_bos, add_special);
        prop_assert!(result.is_ok(),
            "encoding empty string must succeed; got {:?}", result.err());
    }
}

proptest! {
    /// Encoding a whitespace-only string succeeds and produces valid token IDs.
    #[test]
    fn prop_whitespace_only_encodes_without_panic(spaces in " {1,20}") {
        let tok = BasicTokenizer::new();
        let result = tok.encode(&spaces, false, false);
        prop_assert!(result.is_ok(),
            "whitespace-only input must not cause an error");
        let tokens = result.unwrap();
        let vocab = tok.vocab_size();
        for &id in &tokens {
            prop_assert!((id as usize) < vocab,
                "whitespace token ID {} is >= vocab_size {}", id, vocab);
        }
    }
}

proptest! {
    /// Decoding an empty slice always returns the empty string without panicking.
    #[test]
    fn prop_decode_empty_slice_is_empty_string(_dummy in any::<bool>()) {
        let tok = BasicTokenizer::new();
        let result = tok.decode(&[]);
        prop_assert!(result.is_ok(), "decode of empty slice must succeed");
        prop_assert_eq!(result.unwrap(), "",
            "decode of empty slice must return empty string");
    }
}

// ── 6. Unicode safety: multi-byte UTF-8 encodes without panic ────────────────

proptest! {
    /// Multi-byte UTF-8 characters encode without panicking.
    /// BasicTokenizer uses a 256-byte vocab so only pure-ASCII text round-trips;
    /// the invariant tested here is **no panic, token IDs < 256**.
    #[test]
    fn prop_multibyte_utf8_encodes_without_panic(
        text in "[a-z\u{00E9}\u{00F1}\u{4E2D}\u{6587}]{1,20}",
    ) {
        let tok = BasicTokenizer::new();
        let result = tok.encode(&text, false, false);
        prop_assert!(result.is_ok(),
            "multi-byte UTF-8 encoding must not fail; got {:?}", result.err());
        let tokens = result.unwrap();
        prop_assert!(!tokens.is_empty(),
            "non-empty text must produce at least one token");
        // byte-level IDs must be < 256
        for &id in &tokens {
            prop_assert!(id < 256,
                "byte-level encoding must produce IDs < 256; got {}", id);
        }
    }
}

proptest! {
    /// Token count for multi-byte text is between 1 and 4× the character count.
    #[test]
    fn prop_utf8_token_count_bounded(text in "[a-z\u{00E9}\u{4E2D}]{1,30}") {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false)
            .expect("encode must succeed for valid UTF-8");
        let char_count = text.chars().count();
        prop_assert!(tokens.len() >= 1, "non-empty text must produce ≥1 token");
        prop_assert!(tokens.len() <= char_count * 4,
            "token count {} must be ≤ 4× char count {}", tokens.len(), char_count);
    }
}

// ── 7. TokenizerConfig: field combinations ───────────────────────────────────

proptest! {
    /// `TokenizerConfig` fields survive a JSON serialize/deserialize round-trip.
    #[test]
    fn prop_config_json_roundtrip(
        vocab_size in 100usize..100_000usize,
        add_bos in any::<bool>(),
        add_eos in any::<bool>(),
        add_space_prefix in any::<bool>(),
        byte_fallback in any::<bool>(),
        bos_id in proptest::option::of(0u32..1000u32),
        eos_id in proptest::option::of(0u32..1000u32),
        pad_id in proptest::option::of(0u32..1000u32),
        unk_id in proptest::option::of(0u32..1000u32),
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
            unk_token_id: unk_id,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialization must succeed");
        let restored: TokenizerConfig =
            serde_json::from_str(&json).expect("deserialization must succeed");
        prop_assert_eq!(restored.vocab_size, cfg.vocab_size);
        prop_assert_eq!(restored.add_bos, cfg.add_bos);
        prop_assert_eq!(restored.add_eos, cfg.add_eos);
        prop_assert_eq!(restored.add_space_prefix, cfg.add_space_prefix);
        prop_assert_eq!(restored.byte_fallback, cfg.byte_fallback);
        prop_assert_eq!(restored.bos_token_id, cfg.bos_token_id);
        prop_assert_eq!(restored.eos_token_id, cfg.eos_token_id);
        prop_assert_eq!(restored.pad_token_id, cfg.pad_token_id);
        prop_assert_eq!(restored.unk_token_id, cfg.unk_token_id);
        prop_assert_eq!(restored.model_type, cfg.model_type);
    }
}

proptest! {
    /// Default `TokenizerConfig` has a zero vocab_size and both add_* flags false.
    #[test]
    fn prop_config_default_is_zero(_dummy in 0u8..2) {
        let cfg = TokenizerConfig::new();
        prop_assert_eq!(cfg.vocab_size, 0, "default vocab_size must be 0");
        prop_assert!(!cfg.add_bos, "default add_bos must be false");
        prop_assert!(!cfg.add_eos, "default add_eos must be false");
    }
}

proptest! {
    /// `unk_token_id` survives a JSON serialize/deserialize round-trip.
    #[test]
    fn prop_config_unk_token_id_roundtrip(
        vocab_size in 100usize..100_000usize,
        unk_id in proptest::option::of(0u32..1000u32),
    ) {
        let cfg = TokenizerConfig { vocab_size, unk_token_id: unk_id, ..Default::default() };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: TokenizerConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(restored.unk_token_id, unk_id,
            "unk_token_id must survive JSON round-trip");
    }
}

// ── 8. Auto-discovery: path pattern matching logic ───────────────────────────

proptest! {
    /// Loading a tokenizer from a non-existent path returns Err, never panics.
    #[test]
    fn prop_missing_file_returns_err(name in "[a-z]{12,20}") {
        let path = std::path::PathBuf::from(format!(
            "/tmp/bitnet_proptest_missing_{}/tokenizer.json",
            name
        ));
        let result = TokenizerBuilder::from_file(&path);
        prop_assert!(result.is_err(),
            "loading from non-existent path must return Err, not panic");
    }
}

proptest! {
    /// A path with an unrecognized extension returns Err, not a panic.
    #[test]
    fn prop_unsupported_extension_returns_err(ext in "[a-z]{3,6}") {
        // Avoid accidentally constructing a valid extension
        prop_assume!(ext != "json" && ext != "model");
        let path = std::path::PathBuf::from(format!(
            "/tmp/bitnet_proptest_ext/tokenizer.{}", ext
        ));
        let result = bitnet_tokenizers::from_path(&path);
        prop_assert!(result.is_err(),
            "unsupported extension '.{}' must return Err", ext);
    }
}

proptest! {
    /// `TokenizerBuilder::from_pretrained` succeeds for any arbitrary model name
    /// (returns a fallback tokenizer rather than panicking).
    #[test]
    fn prop_from_pretrained_always_succeeds(name in "[a-z]{1,32}") {
        let result = TokenizerBuilder::from_pretrained(&name);
        prop_assert!(result.is_ok(),
            "from_pretrained must not fail for any model name");
        prop_assert!(result.unwrap().vocab_size() > 0,
            "returned tokenizer must have a positive vocab_size");
    }
}

// ── 9. Batch encoding: multiple sentences ────────────────────────────────────

proptest! {
    /// Encoding each sentence in a batch individually yields the same total token
    /// count as concatenating the individual encoding results.
    #[test]
    fn prop_batch_encoding_length_consistency(
        sentences in prop::collection::vec("[a-z ]{1,30}", 2..8),
    ) {
        let tok = BasicTokenizer::new();
        let individual_totals: usize = sentences.iter()
            .map(|s| tok.encode(s, false, false).expect("encode must succeed").len())
            .sum();
        // Concatenate all sentences and encode as one (no separator)
        let combined: String = sentences.join("");
        let combined_tokens = tok.encode(&combined, false, false)
            .expect("combined encode must succeed");
        // Byte-level tokenization: byte count of the combined string equals the sum
        prop_assert_eq!(combined_tokens.len(), individual_totals,
            "byte-level encoding: combined token count must equal sum of individual counts");
    }
}

proptest! {
    /// Encoding N independent strings never panics and always yields N results.
    #[test]
    fn prop_batch_encoding_no_panic(
        sentences in prop::collection::vec("[a-zA-Z0-9 .,]{1,40}", 1..10),
    ) {
        let tok = MockTokenizer::new();
        for sentence in &sentences {
            let result = tok.encode(sentence, false, false);
            prop_assert!(result.is_ok(),
                "encoding '{}' in batch must not fail", sentence);
        }
    }
}

// ── 10. Determinism of encode / decode ───────────────────────────────────────

proptest! {
    /// `encode` is deterministic: same input always produces the same token IDs.
    #[test]
    fn prop_encode_is_deterministic(text in "[a-z]{1,48}") {
        let tok = BasicTokenizer::new();
        let first = tok.encode(&text, false, false).expect("first encode");
        let second = tok.encode(&text, false, false).expect("second encode");
        prop_assert_eq!(&first, &second,
            "encode must be deterministic for {:?}", text);
    }
}

proptest! {
    /// `decode` is deterministic: same token IDs always produce the same string.
    #[test]
    fn prop_decode_is_deterministic(ids in prop::collection::vec(0u32..256u32, 1..50)) {
        let tok = BasicTokenizer::new();
        let first = tok.decode(&ids).expect("first decode");
        let second = tok.decode(&ids).expect("second decode");
        prop_assert_eq!(first, second, "decode must be deterministic");
    }
}

// ── 11. token_to_piece determinism and completeness ──────────────────────────

proptest! {
    /// `token_to_piece` returns the same value on repeated calls (deterministic).
    #[test]
    fn prop_token_to_piece_is_deterministic(id in 0u32..50257u32) {
        let tok = BasicTokenizer::new();
        let first = tok.token_to_piece(id);
        let second = tok.token_to_piece(id);
        prop_assert_eq!(first, second,
            "token_to_piece({}) must return the same value each time", id);
    }
}

proptest! {
    /// `token_to_piece` returns `Some` for all byte-range IDs (0–255).
    #[test]
    fn prop_token_to_piece_some_for_byte_range(id in 0u32..256u32) {
        let tok = BasicTokenizer::new();
        prop_assert!(tok.token_to_piece(id).is_some(),
            "token_to_piece({}) must be Some for byte-range IDs 0-255", id);
    }
}

// ── 12. BOS prepend / EOS append ─────────────────────────────────────────────

proptest! {
    /// When BOS is configured, encoding with `add_bos=true` prepends the BOS ID.
    #[test]
    fn prop_bos_prepended_when_configured(
        text in "[a-zA-Z]{1,50}",
        bos_id in 0u32..128u32,
    ) {
        let tok = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let with_bos = tok.encode(&text, true, false).expect("encode with BOS");
        prop_assert!(!with_bos.is_empty(),
            "non-empty text must produce at least one token");
        prop_assert_eq!(with_bos[0], bos_id,
            "first token must be BOS id {} when add_bos=true", bos_id);
        let without_bos = tok.encode(&text, false, false).expect("encode without BOS");
        prop_assert_eq!(with_bos.len(), without_bos.len() + 1,
            "with_bos encoding must have exactly one more token than without_bos");
    }
}

proptest! {
    /// When EOS is configured, encoding with `add_special=true` appends the EOS ID.
    #[test]
    fn prop_eos_appended_when_configured(
        text in "[a-zA-Z]{1,32}",
        eos_id in 256u32..50257u32,
    ) {
        let tok = BasicTokenizer::with_config(50257, None, Some(eos_id), None);
        let tokens = tok.encode(&text, false, true).expect("encode with EOS");
        prop_assert!(!tokens.is_empty(),
            "non-empty text must produce at least one token");
        prop_assert_eq!(*tokens.last().unwrap(), eos_id,
            "last token must be EOS id {} when add_special=true", eos_id);
    }
}

// ── 13. MockTokenizer special token lookup ───────────────────────────────────

proptest! {
    /// A registered special token can be looked up by its exact string.
    #[test]
    fn prop_registered_special_token_found(token_id in 0u32..50257u32) {
        let tok = MockTokenizer::with_special_tokens(&[("<test_tok>", token_id)]);
        prop_assert_eq!(tok.token_to_id("<test_tok>"), Some(token_id),
            "registered token must be found by token_to_id");
    }
}

proptest! {
    /// A token string that was never registered returns `None`.
    #[test]
    fn prop_unregistered_special_token_is_none(suffix in "[a-z]{8,16}") {
        let tok = MockTokenizer::with_special_tokens(&[("<registered>", 42)]);
        let absent = format!("<absent_{}>", suffix);
        prop_assert!(tok.token_to_id(&absent).is_none(),
            "absent token must return None");
    }
}

// ── 14. get_family_name classification ───────────────────────────────────────

proptest! {
    /// `BasicTokenizer` (no special tokens) always reports family "unknown".
    #[test]
    fn prop_basic_tokenizer_family_is_unknown(_dummy in 0u8..2) {
        prop_assert_eq!(BasicTokenizer::new().get_family_name(), "unknown",
            "BasicTokenizer must report family 'unknown'");
    }
}

proptest! {
    /// A `MockTokenizer` with `<|eot_id|>` registered reports family "llama3".
    #[test]
    fn prop_llama3_family_detected_via_eot_id(eot_id in 100_000u32..200_000u32) {
        let tok = MockTokenizer::with_special_tokens(&[("<|eot_id|>", eot_id)]);
        prop_assert_eq!(tok.get_family_name(), "llama3",
            "tokenizer with <|eot_id|> must report family 'llama3'");
    }
}

proptest! {
    /// A `MockTokenizer` with `[INST]` (but no LLaMA-3 markers) reports "mistral-instruct".
    #[test]
    fn prop_mistral_family_detected_via_inst(inst_id in 50_000u32..100_000u32) {
        let tok = MockTokenizer::with_special_tokens(&[("[INST]", inst_id)]);
        prop_assert_eq!(tok.get_family_name(), "mistral-instruct",
            "tokenizer with [INST] must report family 'mistral-instruct'");
    }
}

// ── 15. Legacy shims ──────────────────────────────────────────────────────────

proptest! {
    /// `encode_legacy` prepends the configured BOS token.
    #[test]
    fn prop_encode_legacy_prepends_bos(
        text in "[a-zA-Z]{1,48}",
        bos_id in 0u32..128u32,
    ) {
        let tok = BasicTokenizer::with_config(50257, Some(bos_id), None, None);
        let tokens = tok.encode_legacy(&text, false).expect("encode_legacy must succeed");
        prop_assert!(!tokens.is_empty());
        prop_assert_eq!(tokens[0], bos_id,
            "encode_legacy must prepend BOS token id {}", bos_id);
    }
}

proptest! {
    /// `decode_legacy` ignores the `skip_special_tokens` flag: both values yield the same string.
    #[test]
    fn prop_decode_legacy_skip_flag_no_effect(
        ids in prop::collection::vec(0u32..256u32, 1..50),
    ) {
        let tok = BasicTokenizer::new();
        let skip_true = tok.decode_legacy(&ids, true).expect("decode_legacy(true)");
        let skip_false = tok.decode_legacy(&ids, false).expect("decode_legacy(false)");
        prop_assert_eq!(skip_true, skip_false,
            "decode_legacy must return the same result regardless of skip_special_tokens");
    }
}

// ── 16. decode never panics ───────────────────────────────────────────────────

proptest! {
    /// `BasicTokenizer::decode` never fails for any sequence of token IDs in [0, 50256].
    #[test]
    fn prop_decode_never_panics_basic(
        ids in prop::collection::vec(0u32..50257u32, 0..100),
    ) {
        let tok = BasicTokenizer::new();
        let result = tok.decode(&ids);
        prop_assert!(result.is_ok(),
            "BasicTokenizer::decode must not fail for valid token IDs");
    }
}

proptest! {
    /// `MockTokenizer::decode` never fails for any sequence of token IDs.
    #[test]
    fn prop_decode_never_panics_mock(
        ids in prop::collection::vec(0u32..50257u32, 0..100),
    ) {
        let tok = MockTokenizer::new();
        let result = tok.decode(&ids);
        prop_assert!(result.is_ok(),
            "MockTokenizer::decode must not fail for any token IDs");
    }
}

// ── 17. Each encoded token is individually decodable ─────────────────────────

proptest! {
    /// Every token produced by `encode` can be decoded in isolation.
    #[test]
    fn prop_each_token_individually_decodable(text in "[a-z]{1,32}") {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode(&text, false, false).expect("encode must succeed");
        for &t in &tokens {
            let result = tok.decode(&[t]);
            prop_assert!(result.is_ok(),
                "decode of single token {} must not fail", t);
        }
    }
}

// ── Integration tests (no model files required) ───────────────────────────────

/// HF tokenizer loaded from the bundled minimal fixture encodes known words correctly.
#[test]
fn integration_minimal_hf_tokenizer_loads_and_encodes() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("minimal_tokenizer.json");

    // If the fixture is missing we skip gracefully rather than fail
    if !path.exists() {
        eprintln!("skipping: fixture not found at {}", path.display());
        return;
    }

    let tok = bitnet_tokenizers::load_tokenizer(&path).expect("load minimal tokenizer");
    assert!(tok.vocab_size() > 0, "minimal tokenizer must have a positive vocab_size");

    let ids = tok.encode("hello world", false, false).expect("encode 'hello world'");
    assert!(!ids.is_empty(), "encoding 'hello world' must produce at least one token");

    // All returned IDs must be within bounds
    for &id in &ids {
        assert!(
            (id as usize) < tok.vocab_size(),
            "token id {} must be < vocab_size {}",
            id,
            tok.vocab_size()
        );
    }
}

/// `TokenizerBuilder::from_pretrained("gpt2")` returns a tokenizer with the
/// canonical GPT-2 vocab size (50 257).
#[test]
fn integration_from_pretrained_gpt2_has_correct_vocab_size() {
    let tok =
        TokenizerBuilder::from_pretrained("gpt2").expect("from_pretrained('gpt2') must succeed");
    assert_eq!(tok.vocab_size(), 50257, "gpt2 profile must have vocab_size == 50257");
}

/// `TokenizerBuilder::from_pretrained("bert")` returns a tokenizer with the
/// canonical BERT vocab size (30 522) and proper special token configuration.
#[test]
fn integration_from_pretrained_bert_special_tokens() {
    let tok =
        TokenizerBuilder::from_pretrained("bert").expect("from_pretrained('bert') must succeed");
    assert_eq!(tok.vocab_size(), 30522, "bert profile must have vocab_size == 30522");
    // CLS = 101, SEP = 102, PAD = 0 in the BERT preset
    assert_eq!(tok.bos_token_id(), Some(101), "BERT BOS (CLS) must be 101");
    assert_eq!(tok.eos_token_id(), Some(102), "BERT EOS (SEP) must be 102");
    assert_eq!(tok.pad_token_id(), Some(0), "BERT PAD must be 0");
}

/// `BasicTokenizer` correctly reports no special tokens when none are configured.
#[test]
fn integration_no_special_tokens_when_unconfigured() {
    let tok = BasicTokenizer::with_config(1000, None, None, None);
    assert!(tok.bos_token_id().is_none(), "unconfigured BOS must be None");
    assert!(tok.eos_token_id().is_none(), "unconfigured EOS must be None");
    assert!(tok.pad_token_id().is_none(), "unconfigured PAD must be None");
}

/// Unicode text (Japanese, accented Latin) encodes without panic and produces
/// a non-empty, bounds-checked token list.
#[test]
fn integration_unicode_encode_no_panic() {
    let tok = BasicTokenizer::new();
    let samples = ["こんにちは", "café", "مرحبا", "Привет", "αβγ"];
    for &sample in &samples {
        let result = tok.encode(sample, false, false);
        assert!(result.is_ok(), "encoding '{}' must not fail", sample);
        let tokens = result.unwrap();
        assert!(!tokens.is_empty(), "encoding '{}' must produce tokens", sample);
        for &id in &tokens {
            assert!(id < 256, "byte-level token ID {} must be < 256", id);
        }
    }
}

/// Batch of sentences all encode without error and produce non-empty outputs.
#[test]
fn integration_batch_encoding_all_succeed() {
    let tok = MockTokenizer::new();
    let sentences = ["Hello world", "foo bar baz", "42 is the answer", "x"];
    for s in &sentences {
        let tokens = tok.encode(s, false, false).expect("batch item must encode");
        assert!(!tokens.is_empty(), "sentence '{}' must produce tokens", s);
    }
}
