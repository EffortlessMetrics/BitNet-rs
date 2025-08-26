use bitnet_tokenizers::{Tokenizer, TokenizerConfig, UniversalTokenizer};
use tokenizers::{models::unigram::Unigram, EncodeInput, Tokenizer as HfTokenizer};

#[test]
fn gpt2_parity() {
    let config = TokenizerConfig { model_type: "gpt2".into(), ..Default::default() };
    let tok = UniversalTokenizer::new(config).unwrap();
    let bpe = tiktoken_rs::r50k_base().unwrap();
    let text = "hello world";
    let ours = tok.encode(text, false, false).unwrap();
    let reference = bpe.encode_ordinary(text);
    assert_eq!(ours, reference);
    let dec = tok.decode(&ours).unwrap();
    assert_eq!(dec, bpe.decode(reference).unwrap());
}

#[test]
fn tiktoken_parity() {
    let config = TokenizerConfig { model_type: "tiktoken".into(), ..Default::default() };
    let tok = UniversalTokenizer::new(config).unwrap();
    let bpe = tiktoken_rs::cl100k_base().unwrap();
    let text = "testing";
    let ours = tok.encode(text, false, false).unwrap();
    let reference = bpe.encode_ordinary(text);
    assert_eq!(ours, reference);
    let dec = tok.decode(&ours).unwrap();
    assert_eq!(dec, bpe.decode(reference).unwrap());
}

#[test]
fn falcon_parity() {
    let config = TokenizerConfig { model_type: "falcon".into(), ..Default::default() };
    let tok = UniversalTokenizer::new(config).unwrap();
    let bpe = tiktoken_rs::p50k_base().unwrap();
    let text = "falcon";
    let ours = tok.encode(text, false, false).unwrap();
    let reference = bpe.encode_ordinary(text);
    assert_eq!(ours, reference);
    let dec = tok.decode(&ours).unwrap();
    assert_eq!(dec, bpe.decode(reference).unwrap());
}

#[test]
fn sentencepiece_parity() {
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("▁hello".to_string(), 0.0),
        ("▁world".to_string(), 0.0),
    ];
    let config = TokenizerConfig {
        model_type: "sentencepiece".into(),
        vocabulary: Some(vocab.clone()),
        unk_token_id: Some(0),
        ..Default::default()
    };
    let tok = UniversalTokenizer::new(config.clone()).unwrap();
    let model = Unigram::from(
        vocab.into_iter().map(|(t,s)| (t, s as f64)).collect(),
        Some(0),
        false,
    ).unwrap();
    let reference = HfTokenizer::new(model);
    let text = "hello world";
    let ours = tok.encode(text, false, false).unwrap();
    let ref_tokens = reference.encode(EncodeInput::Single(text.into()), false).unwrap().get_ids().to_vec();
    assert_eq!(ours, ref_tokens);
    let dec = tok.decode(&ours).unwrap();
    assert_eq!(dec, reference.decode(&ref_tokens, true).unwrap());
}

#[test]
fn llama_parity() {
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("▁cat".to_string(), 0.0),
        ("▁sat".to_string(), 0.0),
    ];
    let config = TokenizerConfig {
        model_type: "llama".into(),
        vocabulary: Some(vocab.clone()),
        unk_token_id: Some(0),
        ..Default::default()
    };
    let tok = UniversalTokenizer::new(config.clone()).unwrap();
    let model = Unigram::from(
        vocab.into_iter().map(|(t,s)| (t, s as f64)).collect(),
        Some(0),
        false,
    ).unwrap();
    let reference = HfTokenizer::new(model);
    let text = "cat sat";
    let ours = tok.encode(text, false, false).unwrap();
    let ref_tokens = reference.encode(EncodeInput::Single(text.into()), false).unwrap().get_ids().to_vec();
    assert_eq!(ours, ref_tokens);
    let dec = tok.decode(&ours).unwrap();
    assert_eq!(dec, reference.decode(&ref_tokens, true).unwrap());
}
