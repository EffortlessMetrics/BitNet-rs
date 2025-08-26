#![cfg(feature = "integration-tests")]

use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

#[test]
fn encode_decode_roundtrip() {
    let tokenizer = BasicTokenizer::new();
    let tokens = tokenizer.encode("hello world", false, false).unwrap();
    assert_eq!(tokens, vec![0, 1]);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert!(decoded.contains("2 tokens"));
}

#[test]
fn adds_eos_token_when_requested() {
    let tokenizer = BasicTokenizer::new();
    let without_special = tokenizer.encode("hi", false, false).unwrap();
    let with_special = tokenizer.encode("hi", false, true).unwrap();
    assert_eq!(without_special, vec![0]);
    assert_eq!(with_special, vec![0, tokenizer.eos_token_id().unwrap()]);
}

#[test]
fn vocab_and_special_token_ids() {
    let tokenizer = BasicTokenizer::new();
    assert_eq!(tokenizer.vocab_size(), 50257);
    assert_eq!(tokenizer.eos_token_id(), Some(50256));
    assert_eq!(tokenizer.pad_token_id(), None);
}
