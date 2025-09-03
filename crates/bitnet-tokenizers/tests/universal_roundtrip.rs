#![cfg(feature = "integration-tests")]

use bitnet_tokenizers::{Tokenizer, TokenizerConfig, UniversalTokenizer};

#[test]
fn gpt2_bpe_roundtrip() {
    // Minimal BPE tokenizer with vocab and merges using tokenizers crate
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 3,
        pre_tokenizer: None,
        add_bos: false,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: false,
        bos_token_id: None,
        eos_token_id: None,
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: Some(vec![
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("ab".to_string(), 0.0),
        ]),
        bpe_merges: Some(vec!["a b".to_string()]),
    };

    let tokenizer = UniversalTokenizer::new(config).expect("build gpt2 tokenizer");
    let ids = tokenizer.encode("ab", false, false).expect("encode");
    assert_eq!(ids, vec![2]);

    // Test round trip
    let text = tokenizer.decode(&ids).expect("decode");
    assert_eq!(text, "ab");
}

#[cfg(feature = "spm")]
#[test]
fn sentencepiece_roundtrip() {
    let config = TokenizerConfig {
        model_type: "sentencepiece".to_string(),
        vocab_size: 32000, // Standard SentencePiece vocab size
        pre_tokenizer: None,
        add_bos: false,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: false,
        bos_token_id: None,
        eos_token_id: None,
        pad_token_id: None,
        unk_token_id: Some(0),
        vocabulary: None, // Let SentencePiece tokenizer handle this
        bpe_merges: None,
    };

    let tokenizer = UniversalTokenizer::new(config).expect("build sp tokenizer");
    let ids = tokenizer.encode("ab", false, false).expect("encode");
    assert!(!ids.is_empty(), "Should produce tokens");

    // Test that we can decode back
    let text = tokenizer.decode(&ids).expect("decode");
    assert!(!text.is_empty(), "Should decode to non-empty text");
}
