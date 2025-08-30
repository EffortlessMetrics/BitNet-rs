use bitnet_tokenizers::{Tokenizer, TokenizerConfig, UniversalTokenizer};

#[test]
fn gpt2_bpe_roundtrip() {
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 4,
        vocabulary: Some(vec![
            ("[UNK]".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("ab".to_string(), 0.0),
        ]),
        bpe_merges: Some(vec!["a b".to_string()]),
        unk_token_id: Some(0),
        ..Default::default()
    };
    let tokenizer = UniversalTokenizer::new(config).expect("create tokenizer");
    let ids = tokenizer.encode("ab", false, false).expect("encode");
    assert_eq!(ids, vec![3]);
    let text = tokenizer.decode(&ids).expect("decode");
    assert_eq!(text, "ab");
}
