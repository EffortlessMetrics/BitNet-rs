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
    tokenizer.with_pre_tokenizer(tokenizers::pre_tokenizers::whitespace::Whitespace::default());
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
