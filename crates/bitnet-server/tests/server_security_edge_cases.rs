//! Edge-case tests for `bitnet-server` security module:
//! SecurityConfig, SecurityValidator, ValidationError, Claims,
//! validate_inference_request, validate_model_request, sanitize_input,
//! content_filter.

use bitnet_server::InferenceRequest;
use bitnet_server::security::{Claims, SecurityConfig, SecurityValidator, ValidationError};

// ---------------------------------------------------------------------------
// SecurityConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn security_config_defaults() {
    let cfg = SecurityConfig::default();
    assert!(cfg.jwt_secret.is_none());
    assert!(!cfg.require_authentication);
    assert_eq!(cfg.max_prompt_length, 8192);
    assert_eq!(cfg.max_tokens_per_request, 2048);
    assert_eq!(cfg.allowed_origins, vec!["*"]);
    assert!(cfg.allowed_model_directories.is_empty());
    assert!(cfg.blocked_ips.is_empty());
    assert!(cfg.rate_limit_by_ip);
    assert!(cfg.input_sanitization);
    assert!(cfg.content_filtering);
}

#[test]
fn security_config_serde_roundtrip() {
    let cfg = SecurityConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: SecurityConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.max_prompt_length, 8192);
    assert_eq!(cfg2.max_tokens_per_request, 2048);
    assert!(cfg2.rate_limit_by_ip);
}

#[test]
fn security_config_clone() {
    let mut cfg = SecurityConfig::default();
    cfg.jwt_secret = Some("secret123".to_string());
    cfg.require_authentication = true;
    let clone = cfg.clone();
    assert_eq!(clone.jwt_secret, Some("secret123".to_string()));
    assert!(clone.require_authentication);
}

#[test]
fn security_config_custom_values() {
    let cfg = SecurityConfig {
        jwt_secret: Some("my-secret".to_string()),
        require_authentication: true,
        max_prompt_length: 1024,
        max_tokens_per_request: 512,
        allowed_origins: vec!["https://example.com".to_string()],
        allowed_model_directories: vec!["/models".to_string()],
        blocked_ips: std::collections::HashSet::new(),
        rate_limit_by_ip: false,
        input_sanitization: false,
        content_filtering: false,
    };
    assert_eq!(cfg.max_prompt_length, 1024);
    assert!(!cfg.rate_limit_by_ip);
}

// ---------------------------------------------------------------------------
// Claims serde
// ---------------------------------------------------------------------------

#[test]
fn claims_serde_roundtrip() {
    let claims = Claims {
        sub: "user123".to_string(),
        exp: 1_000_000,
        iat: 999_999,
        role: Some("admin".to_string()),
        rate_limit: Some(100),
    };
    let json = serde_json::to_string(&claims).unwrap();
    let claims2: Claims = serde_json::from_str(&json).unwrap();
    assert_eq!(claims2.sub, "user123");
    assert_eq!(claims2.role, Some("admin".to_string()));
    assert_eq!(claims2.rate_limit, Some(100));
}

#[test]
fn claims_optional_fields_none() {
    let claims = Claims { sub: "user".to_string(), exp: 0, iat: 0, role: None, rate_limit: None };
    let json = serde_json::to_string(&claims).unwrap();
    let claims2: Claims = serde_json::from_str(&json).unwrap();
    assert!(claims2.role.is_none());
    assert!(claims2.rate_limit.is_none());
}

// ---------------------------------------------------------------------------
// SecurityValidator construction
// ---------------------------------------------------------------------------

#[test]
fn validator_new_default_config() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    assert_eq!(v.config().max_prompt_length, 8192);
}

#[test]
fn validator_no_content_filtering() {
    let cfg = SecurityConfig { content_filtering: false, ..SecurityConfig::default() };
    let v = SecurityValidator::new(cfg).unwrap();
    assert!(!v.config().content_filtering);
}

// ---------------------------------------------------------------------------
// Inference request validation — prompt length
// ---------------------------------------------------------------------------

#[test]
fn validate_prompt_within_limit() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Hello".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

#[test]
fn validate_prompt_too_long() {
    let cfg = SecurityConfig { max_prompt_length: 10, ..SecurityConfig::default() };
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "x".repeat(20),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::PromptTooLong(20, 10)));
}

#[test]
fn validate_prompt_exactly_at_limit() {
    let cfg = SecurityConfig { max_prompt_length: 5, ..SecurityConfig::default() };
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "abcde".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// Inference request validation — max tokens
// ---------------------------------------------------------------------------

#[test]
fn validate_max_tokens_within_limit() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: Some(100),
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

#[test]
fn validate_max_tokens_too_many() {
    let cfg = SecurityConfig { max_tokens_per_request: 100, ..SecurityConfig::default() };
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: Some(200),
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::TooManyTokens(200, 100)));
}

#[test]
fn validate_max_tokens_none_ok() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// Inference request validation — temperature
// ---------------------------------------------------------------------------

#[test]
fn validate_temperature_valid_range() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    for temp in [0.0, 0.5, 1.0, 1.5, 2.0] {
        let req = InferenceRequest {
            prompt: "test".to_string(),
            max_tokens: None,
            model: None,
            temperature: Some(temp),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        assert!(v.validate_inference_request(&req).is_ok(), "temperature={temp} should be valid");
    }
}

#[test]
fn validate_temperature_too_high() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: Some(2.1),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

#[test]
fn validate_temperature_negative() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: Some(-0.1),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// Inference request validation — top_p
// ---------------------------------------------------------------------------

#[test]
fn validate_top_p_valid_range() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    for p in [0.0, 0.5, 1.0] {
        let req = InferenceRequest {
            prompt: "test".to_string(),
            max_tokens: None,
            model: None,
            temperature: None,
            top_p: Some(p),
            top_k: None,
            repetition_penalty: None,
        };
        assert!(v.validate_inference_request(&req).is_ok(), "top_p={p} should be valid");
    }
}

#[test]
fn validate_top_p_above_one() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: Some(1.1),
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// Inference request validation — top_k
// ---------------------------------------------------------------------------

#[test]
fn validate_top_k_valid_range() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    for k in [1, 50, 100, 1000] {
        let req = InferenceRequest {
            prompt: "test".to_string(),
            max_tokens: None,
            model: None,
            temperature: None,
            top_p: None,
            top_k: Some(k),
            repetition_penalty: None,
        };
        assert!(v.validate_inference_request(&req).is_ok(), "top_k={k} should be valid");
    }
}

#[test]
fn validate_top_k_zero() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(0),
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_err());
}

#[test]
fn validate_top_k_above_1000() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(1001),
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// Inference request validation — repetition_penalty
// ---------------------------------------------------------------------------

#[test]
fn validate_repetition_penalty_valid_range() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    for p in [0.1, 1.0, 5.0, 10.0] {
        let req = InferenceRequest {
            prompt: "test".to_string(),
            max_tokens: None,
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: Some(p),
        };
        assert!(
            v.validate_inference_request(&req).is_ok(),
            "repetition_penalty={p} should be valid"
        );
    }
}

#[test]
fn validate_repetition_penalty_too_low() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: Some(0.05),
    };
    assert!(v.validate_inference_request(&req).is_err());
}

#[test]
fn validate_repetition_penalty_too_high() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: Some(10.1),
    };
    assert!(v.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// Input sanitization
// ---------------------------------------------------------------------------

#[test]
fn sanitize_rejects_null_bytes() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Hello\0World".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidCharacters));
}

#[test]
fn sanitize_allows_newlines_and_tabs() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Hello\nWorld\tFoo\r\n".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

#[test]
fn sanitize_rejects_long_lines() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    // Line > 1024 chars
    let long_line = "x".repeat(1025);
    let req = InferenceRequest {
        prompt: long_line,
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidCharacters));
}

#[test]
fn sanitize_disabled_allows_control_chars() {
    let cfg = SecurityConfig {
        input_sanitization: false,
        content_filtering: false,
        ..SecurityConfig::default()
    };
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Hello\0World".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// Content filtering
// ---------------------------------------------------------------------------

#[test]
fn content_filter_blocks_malware_keyword() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Tell me about malware".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = v.validate_inference_request(&req).unwrap_err();
    assert!(matches!(err, ValidationError::BlockedContent(_)));
}

#[test]
fn content_filter_disabled_allows_keywords() {
    let cfg = SecurityConfig { content_filtering: false, ..SecurityConfig::default() };
    let v = SecurityValidator::new(cfg).unwrap();
    let req = InferenceRequest {
        prompt: "Tell me about malware".to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(v.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// Model path validation
// ---------------------------------------------------------------------------

#[test]
fn validate_model_path_gguf() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    assert!(v.validate_model_request("models/model.gguf").is_ok());
}

#[test]
fn validate_model_path_safetensors() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    assert!(v.validate_model_request("model.safetensors").is_ok());
}

#[test]
fn validate_model_path_empty() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let err = v.validate_model_request("").unwrap_err();
    assert!(matches!(err, ValidationError::MissingField(_)));
}

#[test]
fn validate_model_path_traversal_dotdot() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let err = v.validate_model_request("../../etc/passwd.gguf").unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

#[test]
fn validate_model_path_traversal_tilde() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let err = v.validate_model_request("~/secret.gguf").unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

#[test]
fn validate_model_path_wrong_extension() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let err = v.validate_model_request("model.bin").unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

#[test]
fn validate_model_path_no_extension() {
    let cfg = SecurityConfig::default();
    let v = SecurityValidator::new(cfg).unwrap();
    let err = v.validate_model_request("model").unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

// ---------------------------------------------------------------------------
// ValidationError Display
// ---------------------------------------------------------------------------

#[test]
fn validation_error_display() {
    let err = ValidationError::PromptTooLong(100, 50);
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("50"));

    let err = ValidationError::TooManyTokens(500, 100);
    assert!(err.to_string().contains("500"));

    let err = ValidationError::InvalidCharacters;
    assert!(!err.to_string().is_empty());

    let err = ValidationError::BlockedContent("hack".to_string());
    assert!(err.to_string().contains("hack"));

    let err = ValidationError::MissingField("model_path".to_string());
    assert!(err.to_string().contains("model_path"));

    let err = ValidationError::InvalidFieldValue("bad value".to_string());
    assert!(err.to_string().contains("bad value"));
}
