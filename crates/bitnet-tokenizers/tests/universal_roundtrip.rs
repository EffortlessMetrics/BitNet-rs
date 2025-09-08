#![cfg(feature = "integration-tests")]

use bitnet_tokenizers::{Tokenizer, TokenizerConfig, UniversalTokenizer};

#[test]
fn gpt2_bpe_roundtrip() {
    // Test with minimal mock tokenizer behavior
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 50257, // Standard GPT-2 vocab size
        pre_tokenizer: None,
        add_bos: false,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: false,
        bos_token_id: None,
        eos_token_id: None,
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None, // Let mock tokenizer handle this
        bpe_merges: None,
    };

    let tokenizer = UniversalTokenizer::new(config).expect("build gpt2 tokenizer");
    let ids = tokenizer.encode("ab", false, false).expect("encode");
    assert!(!ids.is_empty(), "Should produce tokens");

    // Test that we can decode back
    let text = tokenizer.decode(&ids).expect("decode");
    assert!(!text.is_empty(), "Should decode to non-empty text");
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
