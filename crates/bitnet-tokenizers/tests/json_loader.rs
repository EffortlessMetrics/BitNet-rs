use bitnet_tokenizers::loader::load_tokenizer;
use std::collections::HashMap;

#[test]
fn load_hf_tokenizer_json() {
    use tokenizers::{Tokenizer as HfTokenizer, models::wordlevel::WordLevel};
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("<unk>".to_string(), 2);

    let vocab_file = tempfile::NamedTempFile::new().expect("vocab file");
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    std::fs::write(vocab_file.path(), vocab_json).unwrap();

    let model = WordLevel::from_file(vocab_file.path().to_str().unwrap(), "<unk>".into())
        .expect("create model");
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
    let json = tokenizer.to_string(true).expect("serialize tokenizer");

    let tmp = tempfile::Builder::new().suffix(".json").tempfile().expect("tmp file");
    std::fs::write(tmp.path(), json).expect("write json");

    let tok = load_tokenizer(tmp.path()).expect("load tokenizer");
    let ids = tok.encode("hello world", false, false).expect("encode");
    assert_eq!(ids, vec![0, 1]);
    let text = tok.decode(&ids).expect("decode");
    assert!(text.contains("hello"));
    assert!(text.contains("world"));
}

#[test]
fn load_invalid_json_fails() {
    let tmp = tempfile::NamedTempFile::new().expect("tmp");
    std::fs::write(tmp.path(), "not json").unwrap();
    assert!(load_tokenizer(tmp.path()).is_err());
}

#[test]
fn load_json_missing_model_type_fails() {
    let tmp = tempfile::Builder::new().suffix(".json").tempfile().expect("tmp file");

    // JSON without model.type field
    let json = r#"{"version": "1.0", "truncation": null}"#;
    std::fs::write(tmp.path(), json).expect("write json");

    let result = load_tokenizer(tmp.path());
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("missing 'model.type' field"));
    }
}

#[test]
fn load_json_with_special_tokens() {
    use tokenizers::{Tokenizer as HfTokenizer, models::wordlevel::WordLevel};
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<s>".to_string(), 0);
    vocab.insert("</s>".to_string(), 1);
    vocab.insert("hello".to_string(), 2);
    vocab.insert("world".to_string(), 3);
    vocab.insert("<unk>".to_string(), 4);

    let vocab_file = tempfile::NamedTempFile::new().expect("vocab file");
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    std::fs::write(vocab_file.path(), vocab_json).unwrap();

    let model = WordLevel::from_file(vocab_file.path().to_str().unwrap(), "<unk>".into())
        .expect("create model");
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
    let json = tokenizer.to_string(true).expect("serialize tokenizer");

    let tmp = tempfile::Builder::new().suffix(".json").tempfile().expect("tmp file");
    std::fs::write(tmp.path(), json).expect("write json");

    let tok = load_tokenizer(tmp.path()).expect("load tokenizer");

    // Test basic encoding
    let ids = tok.encode("hello world", false, false).expect("encode");
    assert_eq!(ids, vec![2, 3]);

    // Test with BOS
    let ids = tok.encode("hello world", true, false).expect("encode with BOS");
    assert_eq!(ids, vec![0, 2, 3]); // <s> hello world

    // Test that special tokens are detected
    assert_eq!(tok.bos_token_id(), Some(0));
    assert_eq!(tok.eos_token_id(), Some(1));
}

#[test]
fn load_json_empty_text_encoding() {
    use tokenizers::{Tokenizer as HfTokenizer, models::wordlevel::WordLevel};
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<unk>".to_string(), 0);

    let vocab_file = tempfile::NamedTempFile::new().expect("vocab file");
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    std::fs::write(vocab_file.path(), vocab_json).unwrap();

    let model = WordLevel::from_file(vocab_file.path().to_str().unwrap(), "<unk>".into())
        .expect("create model");
    let tokenizer = HfTokenizer::new(model);
    let json = tokenizer.to_string(true).expect("serialize tokenizer");

    let tmp = tempfile::Builder::new().suffix(".json").tempfile().expect("tmp file");
    std::fs::write(tmp.path(), json).expect("write json");

    let tok = load_tokenizer(tmp.path()).expect("load tokenizer");

    // Test empty string encoding
    let ids = tok.encode("", false, false).expect("encode empty");
    assert_eq!(ids.len(), 0);
}
