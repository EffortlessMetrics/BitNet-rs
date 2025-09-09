//! Tests for strict mode validation that prevents mock fallbacks

use bitnet_tokenizers::{UniversalTokenizer, TokenizerConfig, Tokenizer};
use temp_env::with_var;

#[test]
fn strict_mode_disallows_bpe_mock_fallback() {
    with_var("BITNET_STRICT_TOKENIZERS", Some("1"), || {
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
    });
}

#[test]
fn strict_mode_disallows_unknown_tokenizer_fallback() {
    with_var("BITNET_STRICT_TOKENIZERS", Some("1"), || {
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
    });
}

#[test]
fn normal_mode_allows_mock_fallback() {
    // Ensure strict mode is explicitly not set (None removes the env var)
    with_var("BITNET_STRICT_TOKENIZERS", None::<&str>, || {
        let cfg = TokenizerConfig {
            model_type: "bpe".into(),
            ..Default::default()
        };
        
        // Should succeed with mock tokenizer in normal mode
        let tokenizer = UniversalTokenizer::new(cfg)
            .expect("mock tokenizer should be created in normal mode");
        
        // Test deterministic behavior instead of vacuous assertion
        let a = tokenizer.encode("test", false, false).unwrap();
        let b = tokenizer.encode("test", false, false).unwrap();
        assert_eq!(a, b, "mock tokenizer must be deterministic");

        let empty = tokenizer.encode("", false, false).unwrap();
        assert!(empty.is_empty(), "encoding empty string should be empty");
    });
}