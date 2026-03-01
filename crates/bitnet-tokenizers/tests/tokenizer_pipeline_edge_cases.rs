//! Edge-case tests for the tokenizer pipeline: BasicTokenizer, TokenizerConfig,
//! and the Tokenizer trait interface. Covers encode/decode roundtrips, special
//! token handling, boundary conditions, and multi-SLM configuration scenarios.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer, TokenizerConfig};

// ---------------------------------------------------------------------------
// BasicTokenizer construction
// ---------------------------------------------------------------------------

#[test]
fn basic_tokenizer_default_vocab_size() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.vocab_size(), 50257, "default should be GPT-2 vocab size");
}

#[test]
fn basic_tokenizer_default_eos() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.eos_token_id(), Some(50256));
}

#[test]
fn basic_tokenizer_default_no_bos() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.bos_token_id(), None);
}

#[test]
fn basic_tokenizer_default_no_pad() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.pad_token_id(), None);
}

#[test]
fn basic_tokenizer_with_config() {
    let tok = BasicTokenizer::with_config(100352, Some(1), Some(2), Some(3));
    assert_eq!(tok.vocab_size(), 100352);
    assert_eq!(tok.bos_token_id(), Some(1));
    assert_eq!(tok.eos_token_id(), Some(2));
    assert_eq!(tok.pad_token_id(), Some(3));
}

#[test]
fn basic_tokenizer_zero_vocab() {
    let tok = BasicTokenizer::with_config(0, None, None, None);
    assert_eq!(tok.vocab_size(), 0);
}

// ---------------------------------------------------------------------------
// Encode edge cases
// ---------------------------------------------------------------------------

#[test]
fn encode_empty_string() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("", false, false).unwrap();
    assert!(tokens.is_empty());
}

#[test]
fn encode_single_ascii_char() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, false).unwrap();
    assert_eq!(tokens, vec![65]); // ASCII 'A'
}

#[test]
fn encode_hello_world() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("Hi", false, false).unwrap();
    assert_eq!(tokens, vec![72, 105]); // 'H'=72, 'i'=105
}

#[test]
fn encode_with_bos_token() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(50256), None);
    let tokens = tok.encode("A", true, false).unwrap();
    assert_eq!(tokens[0], 1, "first token should be BOS");
    assert_eq!(tokens[1], 65, "second token should be 'A'");
}

#[test]
fn encode_without_bos_when_none() {
    let tok = BasicTokenizer::new(); // no BOS configured
    let tokens = tok.encode("A", true, false).unwrap();
    // add_bos=true but no bos_token_id, so no BOS prepended
    assert_eq!(tokens, vec![65]);
}

#[test]
fn encode_with_eos_special() {
    let tok = BasicTokenizer::new();
    let tokens = tok.encode("A", false, true).unwrap();
    let last = *tokens.last().unwrap();
    assert_eq!(last, 50256, "last token should be EOS when add_special=true");
}

#[test]
fn encode_with_bos_and_eos() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
    let tokens = tok.encode("A", true, true).unwrap();
    assert_eq!(tokens[0], 1, "BOS");
    assert_eq!(tokens[1], 65, "content");
    assert_eq!(tokens[2], 2, "EOS");
}

#[test]
fn encode_with_pad() {
    let tok = BasicTokenizer::with_config(50257, None, Some(2), Some(3));
    let tokens = tok.encode("A", false, true).unwrap();
    assert!(tokens.contains(&2), "should contain EOS");
    assert!(tokens.contains(&3), "should contain PAD");
}

#[test]
fn encode_multibyte_utf8() {
    let tok = BasicTokenizer::new();
    // 'é' is 0xC3 0xA9 in UTF-8
    let tokens = tok.encode("é", false, false).unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], 0xC3);
    assert_eq!(tokens[1], 0xA9);
}

#[test]
fn encode_emoji_utf8() {
    let tok = BasicTokenizer::new();
    // '🦙' is 4 bytes in UTF-8
    let tokens = tok.encode("🦙", false, false).unwrap();
    assert_eq!(tokens.len(), 4);
}

#[test]
fn encode_with_small_vocab_rejects_high_bytes() {
    let tok = BasicTokenizer::with_config(128, None, None, None);
    // ASCII 'A' (65) should work
    assert!(tok.encode("A", false, false).is_ok());
    // But a high byte (>= 128) should fail with small vocab
    let result = tok.encode("é", false, false);
    assert!(result.is_err(), "should reject bytes >= vocab_size");
}

// ---------------------------------------------------------------------------
// Decode edge cases
// ---------------------------------------------------------------------------

#[test]
fn decode_empty_tokens() {
    let tok = BasicTokenizer::new();
    let text = tok.decode(&[]).unwrap();
    assert!(text.is_empty());
}

#[test]
fn decode_single_byte() {
    let tok = BasicTokenizer::new();
    let text = tok.decode(&[65]).unwrap();
    assert_eq!(text, "A");
}

#[test]
fn decode_skips_special_tokens() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let text = tok.decode(&[1, 65, 66, 2, 3]).unwrap();
    assert_eq!(text, "AB", "special tokens should be skipped");
}

#[test]
fn decode_high_token_ids_dropped() {
    let tok = BasicTokenizer::new();
    let text = tok.decode(&[65, 1000, 66]).unwrap();
    assert_eq!(text, "AB", "IDs >= 256 without vocab are dropped");
}

// ---------------------------------------------------------------------------
// Encode-decode roundtrip
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_ascii() {
    let tok = BasicTokenizer::new();
    let text = "Hello, World!";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn roundtrip_with_special_tokens_skipped() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
    let text = "Test";
    let tokens = tok.encode(text, true, true).unwrap();
    // Decode should skip BOS/EOS
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn roundtrip_multibyte_utf8() {
    let tok = BasicTokenizer::new();
    let text = "café";
    let tokens = tok.encode(text, false, false).unwrap();
    let decoded = tok.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// token_to_piece
// ---------------------------------------------------------------------------

#[test]
fn token_to_piece_ascii() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(65).unwrap();
    assert_eq!(piece, "A");
}

#[test]
fn token_to_piece_high_id() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(1000).unwrap();
    assert_eq!(piece, "<token_1000>");
}

#[test]
fn token_to_piece_zero() {
    let tok = BasicTokenizer::new();
    let piece = tok.token_to_piece(0).unwrap();
    assert_eq!(piece, "\0");
}

// ---------------------------------------------------------------------------
// is_special_token
// ---------------------------------------------------------------------------

#[test]
fn is_special_token_bos() {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    assert!(tok.is_special_token(1));
    assert!(tok.is_special_token(2));
    assert!(tok.is_special_token(3));
    assert!(!tok.is_special_token(65));
}

#[test]
fn is_special_token_none_configured() {
    let tok = BasicTokenizer::with_config(50257, None, None, None);
    assert!(!tok.is_special_token(0));
    assert!(!tok.is_special_token(1));
}

// ---------------------------------------------------------------------------
// get_family_name (default implementation)
// ---------------------------------------------------------------------------

#[test]
fn basic_tokenizer_family_is_unknown() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.get_family_name(), "unknown");
}

// ---------------------------------------------------------------------------
// real_vocab_size (default matches vocab_size)
// ---------------------------------------------------------------------------

#[test]
fn real_vocab_size_matches_vocab_size() {
    let tok = BasicTokenizer::new();
    assert_eq!(tok.real_vocab_size(), tok.vocab_size());
}

// ---------------------------------------------------------------------------
// Legacy shims
// ---------------------------------------------------------------------------

#[test]
fn encode_legacy_calls_encode() {
    let tok = BasicTokenizer::new();
    let legacy = tok.encode_legacy("A", false).unwrap();
    let direct = tok.encode("A", true, false).unwrap();
    assert_eq!(legacy, direct);
}

#[test]
fn decode_legacy_calls_decode() {
    let tok = BasicTokenizer::new();
    let legacy = tok.decode_legacy(&[65, 66], true).unwrap();
    let direct = tok.decode(&[65, 66]).unwrap();
    assert_eq!(legacy, direct);
}

// ---------------------------------------------------------------------------
// TokenizerConfig
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_config_default() {
    let cfg = TokenizerConfig::new();
    assert_eq!(cfg.model_type, "");
    assert_eq!(cfg.vocab_size, 0);
    assert!(!cfg.add_bos);
    assert!(!cfg.add_eos);
    assert_eq!(cfg.bos_token_id, None);
    assert_eq!(cfg.eos_token_id, None);
}

#[test]
fn tokenizer_config_serde_roundtrip() {
    let mut cfg = TokenizerConfig::new();
    cfg.model_type = "phi4".to_string();
    cfg.vocab_size = 100352;
    cfg.add_bos = true;
    cfg.bos_token_id = Some(100257);
    cfg.eos_token_id = Some(100265);
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: TokenizerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.model_type, "phi4");
    assert_eq!(cfg2.vocab_size, 100352);
    assert!(cfg2.add_bos);
    assert_eq!(cfg2.bos_token_id, Some(100257));
}

#[test]
fn tokenizer_config_clone() {
    let mut cfg = TokenizerConfig::new();
    cfg.vocab_size = 32000;
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.vocab_size, 32000);
}

// ---------------------------------------------------------------------------
// Multi-SLM tokenizer config scenarios
// ---------------------------------------------------------------------------

#[test]
fn phi4_tokenizer_config() {
    let tok = BasicTokenizer::with_config(100352, Some(100257), Some(100265), None);
    assert_eq!(tok.vocab_size(), 100352);
    assert_eq!(tok.bos_token_id(), Some(100257));
    assert_eq!(tok.eos_token_id(), Some(100265));
    assert!(tok.is_special_token(100257));
    assert!(tok.is_special_token(100265));
    assert!(!tok.is_special_token(0));
}

#[test]
fn llama3_tokenizer_config() {
    let tok = BasicTokenizer::with_config(128256, Some(128000), Some(128001), None);
    assert_eq!(tok.vocab_size(), 128256);
    assert!(tok.is_special_token(128000));
    assert!(tok.is_special_token(128001));
}

#[test]
fn gemma_tokenizer_config() {
    let tok = BasicTokenizer::with_config(256000, Some(2), Some(1), None);
    assert_eq!(tok.vocab_size(), 256000);
    assert!(tok.is_special_token(2)); // BOS
    assert!(tok.is_special_token(1)); // EOS
}

#[test]
fn qwen_tokenizer_config() {
    let tok = BasicTokenizer::with_config(151936, None, Some(151643), None);
    assert_eq!(tok.vocab_size(), 151936);
    assert!(tok.is_special_token(151643));
}

// ---------------------------------------------------------------------------
// Stress tests
// ---------------------------------------------------------------------------

#[test]
fn encode_long_text() {
    let tok = BasicTokenizer::new();
    let text = "A".repeat(100_000);
    let tokens = tok.encode(&text, false, false).unwrap();
    assert_eq!(tokens.len(), 100_000);
}

#[test]
fn decode_long_token_sequence() {
    let tok = BasicTokenizer::new();
    let tokens: Vec<u32> = (0..256u32).cycle().take(100_000).collect();
    let decoded = tok.decode(&tokens).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn encode_all_ascii_bytes() {
    let tok = BasicTokenizer::new();
    for byte in 0..128u8 {
        let text = String::from(byte as char);
        let tokens = tok.encode(&text, false, false).unwrap();
        assert_eq!(tokens[0], byte as u32);
    }
}
