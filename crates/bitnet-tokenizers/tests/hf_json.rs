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
    let ids = tokenizer.encode("hello world", false, false).expect("encode");
    assert_eq!(ids, vec![1, 2]);
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");
}
