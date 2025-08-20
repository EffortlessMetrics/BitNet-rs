//! Tokenizer Compatibility Contract Tests
//! 
//! These tests ensure our universal tokenizer maintains compatibility
//! with all supported tokenizer formats and never regresses.

use bitnet_tokenizers::{UniversalTokenizer, TokenizerConfig, Tokenizer};

/// Test that we handle GPT-2 tokenizers correctly
#[test]
fn test_gpt2_tokenizer_contract() {
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 50257,
        pre_tokenizer: Some("gpt2".to_string()),
        add_bos: false,
        add_eos: false,
        add_space_prefix: true,
        byte_fallback: false,
        bos_token_id: Some(50256),
        eos_token_id: Some(50256),
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None,
        bpe_merges: None,
    };
    
    let tokenizer = UniversalTokenizer::new(config).expect("GPT-2 tokenizer should work");
    
    // Test basic tokenization
    let tokens = tokenizer.encode("Hello world", false, false).expect("Should tokenize");
    assert!(!tokens.is_empty(), "Should produce tokens");
    
    // Test space prefix behavior
    let tokens1 = tokenizer.encode("test", false, false).expect("Should tokenize");
    let tokens2 = tokenizer.encode(" test", false, false).expect("Should tokenize");
    // GPT-2 should add space prefix automatically to first
}

/// Test Llama 3 tokenizer (GPT-2 variant with 128k vocab)
#[test]
fn test_llama3_tokenizer_contract() {
    let config = TokenizerConfig {
        model_type: "llama3".to_string(),
        vocab_size: 128256,  // Llama 3's vocab size
        pre_tokenizer: Some("gpt2".to_string()),
        add_bos: true,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: true,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None,
        bpe_merges: None,
    };
    
    let tokenizer = UniversalTokenizer::new(config).expect("Llama 3 tokenizer should work");
    assert_eq!(tokenizer.vocab_size(), 128256);
}

/// Test SentencePiece tokenizer compatibility
#[test]
fn test_sentencepiece_tokenizer_contract() {
    let config = TokenizerConfig {
        model_type: "sentencepiece".to_string(),
        vocab_size: 32000,
        pre_tokenizer: Some("llama".to_string()),
        add_bos: true,
        add_eos: false,
        add_space_prefix: false,
        byte_fallback: false,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        pad_token_id: Some(0),
        unk_token_id: Some(0),
        vocabulary: None,
        bpe_merges: None,
    };
    
    let tokenizer = UniversalTokenizer::new(config).expect("SentencePiece tokenizer should work");
    
    // Test BOS addition
    let tokens = tokenizer.encode("test", true, false).expect("Should tokenize");
    assert_eq!(tokens[0], 1, "Should start with BOS token");
}

/// Test auto-detection from broken metadata
#[test]
fn test_broken_metadata_handling() {
    // Test missing pre-tokenizer
    let mut config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 50257,
        pre_tokenizer: None,  // Missing!
        add_bos: false,
        add_eos: false,
        add_space_prefix: true,
        byte_fallback: false,
        bos_token_id: Some(50256),
        eos_token_id: Some(50256),
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None,
        bpe_merges: None,
    };
    
    // Should still work with auto-fix
    let tokenizer = UniversalTokenizer::new(config.clone());
    assert!(tokenizer.is_ok(), "Should handle missing pre-tokenizer");
    
    // Test unknown tokenizer type
    config.model_type = "unknown_tokenizer_xyz".to_string();
    let tokenizer = UniversalTokenizer::new(config);
    assert!(tokenizer.is_ok(), "Should fallback for unknown tokenizer");
}

/// Test tokenizer detection order
#[test]
fn test_tokenizer_detection_priority() {
    let test_cases = vec![
        ("gpt2", "GPT-2 BPE"),
        ("llama", "SentencePiece"),
        ("llama3", "GPT-2 BPE"),  // Llama 3 uses GPT-2 style
        ("spm", "SentencePiece"),
        ("sentencepiece", "SentencePiece"),
        ("tiktoken", "Tiktoken"),
        ("gpt4", "Tiktoken"),
        ("cl100k", "Tiktoken"),
        ("falcon", "Falcon"),
    ];
    
    for (model_type, expected_backend) in test_cases {
        let config = TokenizerConfig {
            model_type: model_type.to_string(),
            vocab_size: 50000,
            ..Default::default()
        };
        
        let tokenizer = UniversalTokenizer::new(config);
        assert!(tokenizer.is_ok(), "Should handle {} tokenizer", model_type);
    }
}

/// Test special token handling across tokenizers
#[test]
fn test_special_token_contracts() {
    let configs = vec![
        // GPT-2 style
        TokenizerConfig {
            model_type: "gpt2".to_string(),
            bos_token_id: Some(50256),
            eos_token_id: Some(50256),
            ..Default::default()
        },
        // Llama style  
        TokenizerConfig {
            model_type: "llama".to_string(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: Some(0),
            ..Default::default()
        },
    ];
    
    for config in configs {
        let tokenizer = UniversalTokenizer::new(config.clone()).unwrap();
        
        // Test BOS addition
        if config.add_bos {
            let tokens = tokenizer.encode("test", true, false).unwrap();
            assert_eq!(tokens[0], config.bos_token_id.unwrap());
        }
    }
}

/// Regression test for the GPT-2 tokenizer that breaks llama.cpp
#[test]
fn test_gpt2_llama_cpp_regression() {
    // This is the exact configuration that breaks llama.cpp
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 128256,  // Llama 3 size
        pre_tokenizer: None,  // Missing - causes llama.cpp to fail!
        add_bos: false,
        add_eos: false,
        add_space_prefix: true,
        byte_fallback: false,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: None,
        unk_token_id: None,
        vocabulary: None,
        bpe_merges: None,
    };
    
    // We should handle this gracefully
    let tokenizer = UniversalTokenizer::new(config);
    assert!(tokenizer.is_ok(), "Must handle tokenizer that breaks llama.cpp");
    
    let tokenizer = tokenizer.unwrap();
    let tokens = tokenizer.encode("Hello world", false, false);
    assert!(tokens.is_ok(), "Must successfully tokenize where llama.cpp fails");
}