//! Encode → decode round-trip and correctness regression tests.
//!
//! All tests use `BasicTokenizer` or `MockTokenizer` so they run without
//! model files.  Both implementations use byte-level encoding (each UTF-8
//! byte maps to a token ID 0–255), which guarantees lossless round-trips
//! for arbitrary UTF-8 input.

use bitnet_tokenizers::{BasicTokenizer, MockTokenizer, Tokenizer};

// ---------------------------------------------------------------------------
// ASCII round-trip
// ---------------------------------------------------------------------------

#[test]
fn ascii_roundtrip_basic() {
    let tok = BasicTokenizer::new();
    let text = "Hello, world! This is a test.";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn ascii_roundtrip_mock() {
    let tok = MockTokenizer::new();
    let text = "Hello, world! This is a test.";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// Unicode round-trip (CJK, emoji, accented chars)
// ---------------------------------------------------------------------------

#[test]
fn unicode_cjk_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "你好世界";
    let tokens = tok.encode(text, false, false).unwrap();
    assert!(!tokens.is_empty());
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn unicode_emoji_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "Hello 🌍🚀✨";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn unicode_accented_roundtrip() {
    let tok = BasicTokenizer::new();
    let text = "café résumé naïve über";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn unicode_mixed_scripts_roundtrip() {
    let tok = MockTokenizer::new();
    let text = "English 日本語 العربية 한국어";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// Empty string handling
// ---------------------------------------------------------------------------

#[test]
fn empty_string_encode_basic() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("", false, false).unwrap();
    assert!(tokens.is_empty());
}

#[test]
fn empty_string_encode_mock() {
    let tok = MockTokenizer::new();
    let tokens = tok.encode("", false, false).unwrap();
    assert!(tokens.is_empty());
}

#[test]
fn empty_string_decode_basic() {
    let tok = BasicTokenizer::new();
    let decoded = tok.decode(&[]).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn empty_string_decode_mock() {
    let tok = MockTokenizer::new();
    let decoded = tok.decode(&[]).unwrap();
    assert_eq!(decoded, "");
}

// ---------------------------------------------------------------------------
// Single character encoding
// ---------------------------------------------------------------------------

#[test]
fn single_ascii_char() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, false).unwrap();
    assert_eq!(tokens, vec![65]);
    assert_eq!(tok.decode(&tokens).unwrap(), "A");
}

#[test]
fn single_multibyte_char() {
    let tok = BasicTokenizer::new();
    // '€' = 3 UTF-8 bytes: 0xE2, 0x82, 0xAC
    let tokens = tok.encode("€", false, false).unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(tok.decode(&tokens).unwrap(), "€");
}

#[test]
fn single_4byte_char() {
    let tok = BasicTokenizer::new();
    // '𝄞' (musical symbol) = 4 UTF-8 bytes
    let text = "𝄞";
    assert_eq!(text.len(), 4);
    let tokens = tok.encode(text, false, false).unwrap();
    assert_eq!(tokens.len(), 4);
    assert_eq!(tok.decode(&tokens).unwrap(), text);
}

// ---------------------------------------------------------------------------
// Whitespace-only text
// ---------------------------------------------------------------------------

#[test]
fn whitespace_only_spaces() {
    let tok = BasicTokenizer::new();
    let text = "   ";
    let tokens = tok.encode(text, false, false).unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(tok.decode(&tokens).unwrap(), text);
}

#[test]
fn whitespace_mixed() {
    let tok = BasicTokenizer::new();
    let text = " \t\n\r ";
    let tokens = tok.encode(text, false, false).unwrap();
    assert_eq!(tokens.len(), text.len());
    assert_eq!(tok.decode(&tokens).unwrap(), text);
}

// ---------------------------------------------------------------------------
// Special token IDs (BOS, EOS) are valid and distinct
// ---------------------------------------------------------------------------

#[test]
fn special_token_ids_distinct() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let bos = tok.bos_token_id().unwrap();
    let eos = tok.eos_token_id().unwrap();
    let pad = tok.pad_token_id().unwrap();
    assert_ne!(bos, eos, "BOS and EOS must differ");
    assert_ne!(bos, pad, "BOS and PAD must differ");
    assert_ne!(eos, pad, "EOS and PAD must differ");
}

#[test]
fn bos_prepended_when_configured() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
    let tokens = tok.encode("Hi", true, false).unwrap();
    assert_eq!(tokens[0], 1, "first token should be BOS");
}

#[test]
fn eos_appended_when_add_special() {
    let tok = BasicTokenizer::with_config(50257, None, Some(2), None);
    let tokens = tok.encode("Hi", false, true).unwrap();
    assert_eq!(*tokens.last().unwrap(), 2, "last token should be EOS");
}

#[test]
fn roundtrip_with_bos_eos_skipped_on_decode() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
    let tokens = tok.encode("AB", true, true).unwrap();
    // tokens: [BOS=1, 65, 66, EOS=2]
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, "AB", "decode should skip BOS/EOS and recover text");
}

// ---------------------------------------------------------------------------
// Long text encoding (4K+ chars) doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn long_text_no_panic() {
    let tok = BasicTokenizer::new();
    let text = "a".repeat(5000);
    let tokens = tok.encode(&text, false, false).unwrap();
    assert_eq!(tokens.len(), 5000);
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn long_unicode_text_no_panic() {
    let tok = BasicTokenizer::new();
    // Each '你' is 3 UTF-8 bytes → 3 tokens
    let text = "你".repeat(1500); // 4500 bytes
    let tokens = tok.encode(&text, false, false).unwrap();
    assert_eq!(tokens.len(), 4500);
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// Determinism: repeated encode gives identical results
// ---------------------------------------------------------------------------

#[test]
fn encode_deterministic_basic() {
    let tok = BasicTokenizer::new();
    let text = "Determinism check: 42 🎲";
    let t1 = tok.encode(text, false, false).unwrap();
    let t2 = tok.encode(text, false, false).unwrap();
    let t3 = tok.encode(text, false, false).unwrap();
    assert_eq!(t1, t2);
    assert_eq!(t2, t3);
}

#[test]
fn encode_deterministic_mock() {
    let tok = MockTokenizer::new();
    let text = "Determinism check: 42 🎲";
    let t1 = tok.encode(text, false, false).unwrap();
    let t2 = tok.encode(text, false, false).unwrap();
    assert_eq!(t1, t2);
}

#[test]
fn encode_deterministic_with_special_tokens() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let text = "test";
    let t1 = tok.encode(text, true, true).unwrap();
    let t2 = tok.encode(text, true, true).unwrap();
    assert_eq!(t1, t2);
}

// ---------------------------------------------------------------------------
// Token count is reasonable
// ---------------------------------------------------------------------------

#[test]
fn token_count_nonzero_for_nonempty() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("x", false, false).unwrap();
    assert!(!tokens.is_empty(), "non-empty input must produce tokens");
}

#[test]
fn token_count_bounded_by_byte_length() {
    let tok = BasicTokenizer::new();
    let text = "Hello, world!";
    let tokens = tok.encode(text, false, false).unwrap();
    // Byte-level tokenizer: exactly one token per UTF-8 byte
    assert_eq!(tokens.len(), text.len());
}

#[test]
fn token_ids_within_vocab() {
    let tok = BasicTokenizer::new();
    let text = "Check all IDs < vocab_size";
    let tokens = tok.encode(text, false, false).unwrap();
    let vs = tok.vocab_size();
    for &id in &tokens {
        assert!((id as usize) < vs, "token ID {} exceeds vocab size {}", id, vs);
    }
}

// ---------------------------------------------------------------------------
// Decode of out-of-range / high token IDs (graceful handling)
// ---------------------------------------------------------------------------

#[test]
fn decode_high_ids_graceful_basic() {
    let tok = BasicTokenizer::new();
    // IDs >= 256 are silently dropped by BasicTokenizer::decode
    let decoded = tok.decode(&[65, 9999, 66]).unwrap();
    assert_eq!(decoded, "AB", "high IDs should be dropped, keeping A and B");
}

#[test]
fn decode_high_ids_graceful_mock() {
    let tok = MockTokenizer::new();
    let decoded = tok.decode(&[65, 50000, 66]).unwrap();
    assert_eq!(decoded, "AB");
}

#[test]
fn decode_all_high_ids_returns_empty() {
    let tok = BasicTokenizer::new();
    let decoded = tok.decode(&[300, 400, 500]).unwrap();
    assert_eq!(decoded, "", "all-high-ID input should decode to empty string");
}

#[test]
fn decode_invalid_utf8_uses_replacement_char() {
    let tok = BasicTokenizer::new();
    // 0xFF 0xFE are not valid UTF-8 start bytes
    let decoded = tok.decode(&[0xFF, 0xFE]).unwrap();
    assert!(decoded.contains('\u{FFFD}'), "invalid UTF-8 should produce replacement chars");
}

// ---------------------------------------------------------------------------
// Vocab size consistency
// ---------------------------------------------------------------------------

#[test]
fn vocab_size_positive() {
    let tok = BasicTokenizer::new();
    assert!(tok.vocab_size() > 0);

    let mock = MockTokenizer::new();
    assert!(mock.vocab_size() > 0);
}

#[test]
fn real_vocab_size_equals_vocab_size_for_basic() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.vocab_size(), tok.real_vocab_size());
}

// ---------------------------------------------------------------------------
// token_to_piece consistency
// ---------------------------------------------------------------------------

#[test]
fn token_to_piece_ascii_range() {
    let tok = BasicTokenizer::new();
    // Printable ASCII
    assert_eq!(tok.token_to_piece(b'A' as u32), Some("A".to_string()));
    assert_eq!(tok.token_to_piece(b'0' as u32), Some("0".to_string()));
    assert_eq!(tok.token_to_piece(b' ' as u32), Some(" ".to_string()));
}

#[test]
fn token_to_piece_beyond_byte_range() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(1000).unwrap();
    assert!(piece.starts_with("<token_"), "out-of-byte-range piece should be a placeholder");
}

// ---------------------------------------------------------------------------
// Builder / pretrained aliases (no model files needed)
// ---------------------------------------------------------------------------

#[test]
fn builder_from_pretrained_smoke() {
    use bitnet_tokenizers::TokenizerBuilder;
    let tok = TokenizerBuilder::from_pretrained("gpt2").unwrap();
    assert_eq!(tok.vocab_size(), 50257);

    let tok2 = TokenizerBuilder::from_pretrained("bert").unwrap();
    assert_eq!(tok2.vocab_size(), 30522);
    assert!(tok2.bos_token_id().is_some());
    assert!(tok2.eos_token_id().is_some());
}

#[test]
fn builder_pretrained_roundtrip() {
    use bitnet_tokenizers::TokenizerBuilder;
    let tok = TokenizerBuilder::from_pretrained("gpt2").unwrap();
    let text = "round trip via builder";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}
