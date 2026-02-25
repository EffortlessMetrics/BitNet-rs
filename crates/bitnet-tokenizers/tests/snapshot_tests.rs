//! Snapshot tests for `bitnet-tokenizers` public API surface.
//!
//! Pins the byte-level encoding of well-known strings and the default
//! tokenizer configuration to catch accidental regressions.

use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

#[test]
fn basic_tokenizer_encode_hello_snapshot() {
    let tok = BasicTokenizer::new();
    // "Hello" is 5 ASCII bytes: 72 101 108 108 111
    let ids = tok.encode("Hello", false, false).unwrap();
    insta::assert_debug_snapshot!("basic_tokenizer_encode_hello", ids);
}

#[test]
fn basic_tokenizer_encode_empty_is_empty() {
    let tok = BasicTokenizer::new();
    let ids = tok.encode("", false, false).unwrap();
    insta::assert_debug_snapshot!("basic_tokenizer_encode_empty", ids);
}

#[test]
fn basic_tokenizer_vocab_size_snapshot() {
    let tok = BasicTokenizer::new();
    insta::assert_snapshot!("basic_tokenizer_vocab_size", tok.vocab_size().to_string());
}

#[test]
fn basic_tokenizer_eos_token_id_snapshot() {
    let tok = BasicTokenizer::new();
    insta::assert_debug_snapshot!("basic_tokenizer_eos_token_id", tok.eos_token_id());
}

#[test]
fn basic_tokenizer_decode_hello_snapshot() {
    let tok = BasicTokenizer::new();
    let ids: Vec<u32> = "Hello".bytes().map(|b| b as u32).collect();
    let decoded = tok.decode(&ids).unwrap();
    insta::assert_snapshot!("basic_tokenizer_decode_hello", decoded);
}
