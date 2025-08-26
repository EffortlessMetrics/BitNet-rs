#![cfg(feature = "integration-tests")]
use bitnet_tokenizers::load_tokenizer;
use std::path::PathBuf;

#[test]
fn test_hf_json_encoding_decoding() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("minimal_tokenizer.json");
    let tokenizer = load_tokenizer(&path).expect("load tokenizer");

    // Test basic encoding
    let ids = tokenizer.encode("hello world", false, false).expect("encode");
    assert_eq!(ids, vec![1, 2]);

    // Test decoding
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");

    // Test vocabulary size
    assert_eq!(tokenizer.vocab_size(), 3); // [UNK], hello, world

    // Test unknown token handling
    let unknown_ids = tokenizer.encode("unknown", false, false).expect("encode");
    assert_eq!(unknown_ids, vec![0]); // Should map to [UNK] token

    // Test token to piece conversion
    assert_eq!(tokenizer.token_to_piece(0), Some("[UNK]".to_string()));
    assert_eq!(tokenizer.token_to_piece(1), Some("hello".to_string()));
    assert_eq!(tokenizer.token_to_piece(2), Some("world".to_string()));
}
