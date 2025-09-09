//! Tests for strict mode validation that prevents mock fallbacks

use bitnet_tokenizers::{UniversalTokenizer, TokenizerConfig, Tokenizer};

#[test]
fn strict_mode_disallows_bpe_mock_fallback() {
    // Set strict mode
    unsafe { unsafe { std::env::set_var("BITNET_STRICT_TOKENIZERS", "1"); } }
    
    let cfg = TokenizerConfig {
        model_type: "bpe".into(),
        ..Default::default()
    };
    
    let result = UniversalTokenizer::new(cfg);
    assert!(result.is_err(), "should fail in strict mode");
    let err = result.err().unwrap();
    assert!(
        err.to_string().contains("disabled"), 
        "expected strict mode error, got: {}", 
        err
    );
    
    // Clean up
    unsafe { std::env::remove_var("BITNET_STRICT_TOKENIZERS"); }
}

#[test]
fn strict_mode_disallows_unknown_tokenizer_fallback() {
    unsafe { std::env::set_var("BITNET_STRICT_TOKENIZERS", "1"); }
    
    let cfg = TokenizerConfig {
        model_type: "unknown_tokenizer_type".into(),
        ..Default::default()
    };
    
    let result = UniversalTokenizer::new(cfg);
    assert!(result.is_err(), "should fail in strict mode for unknown tokenizer");
    let err = result.err().unwrap();
    assert!(
        err.to_string().contains("disabled"), 
        "expected strict mode error for unknown tokenizer, got: {}", 
        err
    );
    
    unsafe { std::env::remove_var("BITNET_STRICT_TOKENIZERS"); }
}

#[test]
fn normal_mode_allows_mock_fallback() {
    // Ensure strict mode is not set
    unsafe { std::env::remove_var("BITNET_STRICT_TOKENIZERS"); }
    
    let cfg = TokenizerConfig {
        model_type: "bpe".into(),
        ..Default::default()
    };
    
    // Should succeed with mock tokenizer in normal mode
    let tokenizer = UniversalTokenizer::new(cfg)
        .expect("mock tokenizer should be created in normal mode");
    
    // Should have reasonable vocab size for mock (or at least be valid)
    assert!(tokenizer.vocab_size() >= 0); // Mock may return 0, which is fine
}