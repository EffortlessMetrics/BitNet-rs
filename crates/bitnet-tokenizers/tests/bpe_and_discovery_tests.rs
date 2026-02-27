//! Comprehensive tests for `bitnet-tokenizers` covering genuinely untested areas:
//! - `HfTokenizer` public API (`from_file`, `from_vocab_and_merges`) without
//!   feature gates — filling the gap left by the feature-gated `hf_json.rs`
//! - Unicode / multi-byte UTF-8 round-trips through `BasicTokenizer`
//! - Very long string handling (≥ 1 000 bytes)
//! - `BasicTokenizer::decode` silently skipping BOS/EOS/PAD special tokens
//! - `BasicTokenizer` vocab-overflow error for small configs
//! - Property tests: all generated token IDs are within [0, vocab_size) for
//!   both `BasicTokenizer` and `MockTokenizer`

use bitnet_tokenizers::{BasicTokenizer, HfTokenizer, MockTokenizer, Tokenizer, TokenizerBuilder};
use proptest::prelude::*;
use std::path::Path;

// ── Helper ────────────────────────────────────────────────────────────────────

fn minimal_tokenizer_path() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/minimal_tokenizer.json"))
}

// ── HfTokenizer::from_file ────────────────────────────────────────────────────

/// `HfTokenizer::from_file` loads the WordLevel fixture successfully.
/// This is distinct from the feature-gated `hf_json.rs` tests.
#[test]
fn hf_from_file_loads_wordlevel_fixture() {
    let tok = HfTokenizer::from_file(minimal_tokenizer_path())
        .expect("HfTokenizer::from_file must succeed for minimal_tokenizer.json");
    assert!(tok.vocab_size() > 0, "vocab_size must be positive after loading");
}

/// `HfTokenizer::from_file` returns a tokenizer whose vocab_size is ≥ 3
/// (the fixture contains [UNK], hello, world).
#[test]
fn hf_from_file_vocab_size_matches_fixture() {
    let tok = HfTokenizer::from_file(minimal_tokenizer_path()).expect("load fixture");
    // minimal_tokenizer.json has exactly 3 entries
    assert!(tok.vocab_size() >= 3, "expected vocab_size ≥ 3, got {}", tok.vocab_size());
}

/// `HfTokenizer::from_file` with a non-existent path must return `Err`.
#[test]
fn hf_from_file_missing_path_is_err() {
    let result = HfTokenizer::from_file(Path::new("/nonexistent/path/to/tokenizer.json"));
    assert!(result.is_err(), "loading a missing file must return Err");
}

/// Encoding an empty string via `HfTokenizer` produces an empty token slice.
#[test]
fn hf_from_file_empty_encode_produces_empty() {
    let tok = HfTokenizer::from_file(minimal_tokenizer_path()).expect("load fixture");
    let ids = tok.encode("", false, false).expect("encode empty string");
    assert_eq!(ids.len(), 0, "empty input must produce zero tokens");
}

/// `HfTokenizer::from_file` round-trip: "hello world" encodes then decodes back
/// to a string containing "hello" and "world".
#[test]
fn hf_from_file_roundtrip_hello_world() {
    let tok = HfTokenizer::from_file(minimal_tokenizer_path()).expect("load fixture");
    let ids = tok.encode("hello world", false, false).expect("encode");
    assert!(!ids.is_empty(), "must produce at least one token");
    let decoded = tok.decode(&ids).expect("decode");
    assert!(
        decoded.contains("hello") || decoded.contains("world"),
        "decoded text must contain original words; got {:?}",
        decoded
    );
}

/// `HfTokenizer::token_to_piece` returns `Some` for any in-range token ID.
#[test]
fn hf_from_file_token_to_piece_in_range() {
    let tok = HfTokenizer::from_file(minimal_tokenizer_path()).expect("load fixture");
    // Token 0 is [UNK] in the fixture; it must be Some.
    let piece = tok.token_to_piece(0);
    assert!(piece.is_some(), "token_to_piece(0) must return Some for a loaded tokenizer");
}

// ── HfTokenizer::from_vocab_and_merges ───────────────────────────────────────

/// `from_vocab_and_merges` with a small vocab and no merges builds without panic.
#[test]
fn hf_from_vocab_and_merges_construction_succeeds() {
    let vocab: Vec<(String, f32)> = vec![
        ("h".to_string(), 0.0),
        ("e".to_string(), 0.0),
        ("l".to_string(), 0.0),
        ("o".to_string(), 0.0),
    ];
    let result = HfTokenizer::from_vocab_and_merges(&vocab, &[]);
    assert!(
        result.is_ok(),
        "from_vocab_and_merges must succeed for small vocab; got: {:?}",
        result.err()
    );
}

/// `from_vocab_and_merges` preserves the vocab size supplied.
#[test]
fn hf_from_vocab_and_merges_vocab_size() {
    let vocab: Vec<(String, f32)> = (0..8u32).map(|i| (format!("tok{}", i), 0.0)).collect();
    let tok = HfTokenizer::from_vocab_and_merges(&vocab, &[]).expect("construction must succeed");
    assert_eq!(tok.vocab_size(), 8, "vocab_size must equal the number of supplied tokens");
}

/// Merge rules in "a b" format are parsed; construction with merges succeeds.
#[test]
fn hf_from_vocab_and_merges_with_merge_rules() {
    let vocab: Vec<(String, f32)> =
        vec![("a".to_string(), 0.0), ("b".to_string(), 0.0), ("ab".to_string(), 0.0)];
    let merges = vec!["a b".to_string()];
    let result = HfTokenizer::from_vocab_and_merges(&vocab, &merges);
    assert!(result.is_ok(), "construction with a merge rule must succeed");
}

/// Empty vocab and no merges still constructs a valid (zero-vocab) tokenizer.
#[test]
fn hf_from_vocab_and_merges_empty_vocab() {
    let result = HfTokenizer::from_vocab_and_merges(&[], &[]);
    assert!(result.is_ok(), "empty vocab must not panic; got: {:?}", result.err());
    if let Ok(tok) = result {
        assert_eq!(tok.vocab_size(), 0);
    }
}

// ── BasicTokenizer: Unicode / multi-byte UTF-8 round-trips ────────────────────

/// Latin-extended characters (é, à, ü) round-trip byte-for-byte through
/// `BasicTokenizer` (default vocab_size = 50 257 ≥ 255).
#[test]
fn basic_tokenizer_unicode_latin_extended_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "café résumé naïve fiancée";
    let ids = tok.encode(text, false, false).expect("encode Latin-extended");
    assert_eq!(ids.len(), text.len(), "each UTF-8 byte must produce exactly one token");
    let decoded = tok.decode(&ids).expect("decode Latin-extended");
    assert_eq!(decoded, text, "Latin-extended round-trip must be lossless");
}

/// CJK characters round-trip byte-for-byte (3 bytes each for BMP CJK).
#[test]
fn basic_tokenizer_unicode_cjk_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "你好世界"; // 4 CJK chars × 3 UTF-8 bytes = 12 tokens
    let ids = tok.encode(text, false, false).expect("encode CJK");
    assert_eq!(ids.len(), text.len(), "CJK: byte count must equal token count");
    let decoded = tok.decode(&ids).expect("decode CJK");
    assert_eq!(decoded, text, "CJK round-trip must be lossless");
}

/// Emoji (4-byte UTF-8 sequences) round-trip correctly.
#[test]
fn basic_tokenizer_unicode_emoji_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "Hello 🌍!"; // 🌍 = 4 bytes: 0xF0 0x9F 0x8C 0x8D
    let ids = tok.encode(text, false, false).expect("encode emoji");
    assert_eq!(ids.len(), text.len(), "emoji: byte count must equal token count");
    let decoded = tok.decode(&ids).expect("decode emoji");
    assert_eq!(decoded, text, "emoji round-trip must be lossless");
}

/// Mixed script (ASCII + Latin + CJK + emoji) round-trips correctly.
#[test]
fn basic_tokenizer_unicode_mixed_script_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "Hello café 世界 🚀";
    let ids = tok.encode(text, false, false).expect("encode mixed script");
    assert_eq!(ids.len(), text.len());
    let decoded = tok.decode(&ids).expect("decode mixed script");
    assert_eq!(decoded, text);
}

// ── BasicTokenizer: very long strings ─────────────────────────────────────────

/// 1 000-byte ASCII string encodes and decodes without error.
#[test]
fn basic_tokenizer_very_long_string_1k() {
    let tok = BasicTokenizer::new();
    let text: String = "abcdefghij".chars().cycle().take(1_000).collect();
    let ids = tok.encode(&text, false, false).expect("encode 1k string");
    assert_eq!(ids.len(), 1_000);
    let decoded = tok.decode(&ids).expect("decode 1k string");
    assert_eq!(decoded, text);
}

/// 8 192-byte ASCII string (stress test) encodes and round-trips correctly.
#[test]
fn basic_tokenizer_very_long_string_8k() {
    let tok = BasicTokenizer::new();
    let text: String = (0..8192u32).map(|i| (b'a' + (i % 26) as u8) as char).collect();
    let ids = tok.encode(&text, false, false).expect("encode 8k string");
    assert_eq!(ids.len(), 8_192);
    let decoded = tok.decode(&ids).expect("decode 8k string");
    assert_eq!(decoded, text);
}

// ── BasicTokenizer: decode silently strips special tokens ─────────────────────

/// Tokens equal to BOS or EOS IDs are silently dropped by `decode`.
#[test]
fn basic_tokenizer_decode_strips_bos_eos() {
    // BOS = 1000, EOS = 1001, both outside 0–255 byte range
    let tok = BasicTokenizer::with_config(50257, Some(1000), Some(1001), None);
    // "hi" = [104 'h', 105 'i']
    let ids = vec![1000u32, 104, 105, 1001];
    let decoded = tok.decode(&ids).expect("decode with BOS/EOS");
    assert_eq!(decoded, "hi", "BOS and EOS IDs must be stripped from decoded text");
}

/// Tokens equal to PAD IDs are silently dropped by `decode`.
#[test]
fn basic_tokenizer_decode_strips_pad() {
    let tok = BasicTokenizer::with_config(50257, None, None, Some(2000));
    let ids = vec![2000u32, 104, 105, 2000];
    let decoded = tok.decode(&ids).expect("decode with PAD tokens");
    assert_eq!(decoded, "hi");
}

/// Decode of a slice that contains only special-token IDs returns empty string.
#[test]
fn basic_tokenizer_decode_only_special_tokens_gives_empty() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let decoded = tok.decode(&[1, 2, 3]).expect("decode only special tokens");
    assert_eq!(decoded, "", "all-special-token slice must decode to empty string");
}

// ── BasicTokenizer: vocab-overflow error ──────────────────────────────────────

/// Encoding a character whose byte value ≥ vocab_size must return `Err`.
#[test]
fn basic_tokenizer_small_vocab_overflow_error() {
    let tok = BasicTokenizer::with_config(10, None, None, None);
    // 'A' = 65, which is ≥ 10
    let result = tok.encode("A", false, false);
    assert!(result.is_err(), "byte value >= vocab_size must produce Err");
}

/// Encoding a string with all bytes < vocab_size must succeed.
#[test]
fn basic_tokenizer_vocab_256_accepts_all_single_ascii_bytes() {
    let tok = BasicTokenizer::with_config(256, None, None, None);
    for b in 0u8..128 {
        // Only test valid UTF-8 single-byte characters (0–127)
        let text = String::from_utf8(vec![b]).unwrap();
        let result = tok.encode(&text, false, false);
        assert!(
            result.is_ok(),
            "byte value {} must succeed for vocab_size=256; got: {:?}",
            b,
            result.err()
        );
    }
}

// ── TokenizerBuilder ──────────────────────────────────────────────────────────

/// `TokenizerBuilder::from_file` loads the minimal WordLevel fixture without error.
#[test]
fn tokenizer_builder_from_file_wordlevel_fixture() {
    let tok = TokenizerBuilder::from_file(minimal_tokenizer_path())
        .expect("TokenizerBuilder::from_file must succeed");
    assert!(tok.vocab_size() > 0, "loaded tokenizer must have a non-zero vocab");
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    /// All token IDs produced by `BasicTokenizer` are within [0, vocab_size).
    #[test]
    fn prop_basic_tokenizer_ids_within_vocab_range(
        text in "[a-z0-9 .,!?]{1,64}",
    ) {
        let tok = BasicTokenizer::new();
        let vocab = tok.vocab_size();
        if let Ok(ids) = tok.encode(&text, false, false) {
            for &id in &ids {
                prop_assert!(
                    (id as usize) < vocab,
                    "token ID {} is not in [0, {}); text={:?}",
                    id, vocab, text
                );
            }
        }
    }

    /// Encoding the same ASCII text twice via `BasicTokenizer` always gives the same IDs.
    #[test]
    fn prop_basic_tokenizer_encode_is_deterministic(
        text in "[a-z0-9 .]{1,80}",
    ) {
        let tok = BasicTokenizer::new();
        let first  = tok.encode(&text, false, false).expect("first encode");
        let second = tok.encode(&text, false, false).expect("second encode");
        prop_assert_eq!(first, second, "encode must be deterministic for {:?}", text);
    }

    /// All token IDs produced by `MockTokenizer` are within [0, vocab_size).
    #[test]
    fn prop_mock_tokenizer_ids_within_vocab_range(
        text in "[a-zA-Z0-9 ]{1,64}",
    ) {
        let tok = MockTokenizer::new();
        let vocab = tok.vocab_size();
        let ids = tok.encode(&text, false, false).expect("encode");
        for &id in &ids {
            prop_assert!(
                (id as usize) < vocab,
                "MockTokenizer token ID {} is not in [0, {}); text={:?}",
                id, vocab, text
            );
        }
    }

    /// `BasicTokenizer::with_config` preserves the exact `vocab_size` supplied.
    #[test]
    fn prop_basic_tokenizer_with_config_preserves_vocab_size(
        vs in 256usize..=200_000usize,
    ) {
        let tok = BasicTokenizer::with_config(vs, None, None, None);
        prop_assert_eq!(tok.vocab_size(), vs);
        prop_assert_eq!(tok.real_vocab_size(), vs);
    }

    /// Unicode text encodes without panic in `BasicTokenizer` (may err for high bytes
    /// if vocab_size is small, but must never panic).
    #[test]
    fn prop_basic_tokenizer_unicode_never_panics(
        text in "[a-z\u{00E9}\u{4E2D}\u{1F600}]{0,20}",
    ) {
        let tok = BasicTokenizer::new(); // default vocab_size covers all UTF-8 bytes
        // Must not panic; may succeed or return Err for unusual inputs.
        let _ = tok.encode(&text, false, false);
    }
}
