//! Edge-case tests for server security types: SecurityConfig, Claims,
//! AuthState, ValidationError, SecurityValidator, and request validation.

use bitnet_server::InferenceRequest;
use bitnet_server::security::*;
use std::collections::HashSet;
use std::net::IpAddr;

// ── SecurityConfig ───────────────────────────────────────────────

#[test]
fn security_config_default() {
    let cfg = SecurityConfig::default();
    assert!(cfg.jwt_secret.is_none());
    assert!(!cfg.require_authentication);
    assert_eq!(cfg.max_prompt_length, 8192);
    assert_eq!(cfg.max_tokens_per_request, 2048);
    assert_eq!(cfg.allowed_origins, vec!["*".to_string()]);
    assert!(cfg.allowed_model_directories.is_empty());
    assert!(cfg.blocked_ips.is_empty());
    assert!(cfg.rate_limit_by_ip);
    assert!(cfg.input_sanitization);
    assert!(cfg.content_filtering);
}

#[test]
fn security_config_debug() {
    let cfg = SecurityConfig::default();
    let dbg = format!("{:?}", cfg);
    assert!(dbg.contains("SecurityConfig"));
}

#[test]
fn security_config_clone() {
    let cfg = SecurityConfig::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.max_prompt_length, cfg2.max_prompt_length);
    assert_eq!(cfg.require_authentication, cfg2.require_authentication);
}

#[test]
fn security_config_serde_roundtrip() {
    let cfg = SecurityConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: SecurityConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_prompt_length, cfg2.max_prompt_length);
    assert_eq!(cfg.max_tokens_per_request, cfg2.max_tokens_per_request);
}

#[test]
fn security_config_with_blocked_ips() {
    let mut blocked = HashSet::new();
    blocked.insert("192.168.1.1".parse::<IpAddr>().unwrap());
    blocked.insert("10.0.0.1".parse::<IpAddr>().unwrap());

    let cfg = SecurityConfig { blocked_ips: blocked, ..Default::default() };
    assert_eq!(cfg.blocked_ips.len(), 2);

    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: SecurityConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.blocked_ips.len(), 2);
}

#[test]
fn security_config_with_jwt() {
    let cfg = SecurityConfig {
        jwt_secret: Some("super_secret_key".into()),
        require_authentication: true,
        ..Default::default()
    };
    assert_eq!(cfg.jwt_secret.as_deref(), Some("super_secret_key"));
    assert!(cfg.require_authentication);
}

#[test]
fn security_config_custom_limits() {
    let cfg =
        SecurityConfig { max_prompt_length: 100, max_tokens_per_request: 10, ..Default::default() };
    assert_eq!(cfg.max_prompt_length, 100);
    assert_eq!(cfg.max_tokens_per_request, 10);
}

// ── Claims ───────────────────────────────────────────────────────

#[test]
fn claims_debug() {
    let c = Claims {
        sub: "user123".into(),
        exp: 1700000000,
        iat: 1699000000,
        role: Some("admin".into()),
        rate_limit: Some(100),
    };
    let dbg = format!("{:?}", c);
    assert!(dbg.contains("Claims"));
    assert!(dbg.contains("user123"));
}

#[test]
fn claims_clone() {
    let c = Claims { sub: "user".into(), exp: 1000, iat: 500, role: None, rate_limit: None };
    let c2 = c.clone();
    assert_eq!(c.sub, c2.sub);
    assert_eq!(c.exp, c2.exp);
}

#[test]
fn claims_serde_roundtrip() {
    let c = Claims {
        sub: "test_user".into(),
        exp: 1700000000,
        iat: 1699000000,
        role: Some("viewer".into()),
        rate_limit: Some(50),
    };
    let json = serde_json::to_string(&c).unwrap();
    let c2: Claims = serde_json::from_str(&json).unwrap();
    assert_eq!(c.sub, c2.sub);
    assert_eq!(c.role, c2.role);
    assert_eq!(c.rate_limit, c2.rate_limit);
}

#[test]
fn claims_no_role_no_rate_limit() {
    let c = Claims { sub: "anonymous".into(), exp: 0, iat: 0, role: None, rate_limit: None };
    let json = serde_json::to_string(&c).unwrap();
    let c2: Claims = serde_json::from_str(&json).unwrap();
    assert!(c2.role.is_none());
    assert!(c2.rate_limit.is_none());
}

// ── AuthState ────────────────────────────────────────────────────

#[test]
fn auth_state_clone() {
    let state = AuthState { config: SecurityConfig::default(), jwt_secret: Some("secret".into()) };
    let state2 = state.clone();
    assert_eq!(state.config.max_prompt_length, state2.config.max_prompt_length);
    assert_eq!(state.jwt_secret, state2.jwt_secret);
}

#[test]
fn auth_state_no_jwt() {
    let state = AuthState { config: SecurityConfig::default(), jwt_secret: None };
    assert!(state.jwt_secret.is_none());
}

// ── ValidationError ──────────────────────────────────────────────

#[test]
fn validation_error_prompt_too_long() {
    let err = ValidationError::PromptTooLong(10000, 8192);
    let msg = format!("{}", err);
    assert!(msg.contains("10000"));
    assert!(msg.contains("8192"));
}

#[test]
fn validation_error_too_many_tokens() {
    let err = ValidationError::TooManyTokens(5000, 2048);
    let msg = format!("{}", err);
    assert!(msg.contains("5000"));
    assert!(msg.contains("2048"));
}

#[test]
fn validation_error_invalid_characters() {
    let err = ValidationError::InvalidCharacters;
    let msg = format!("{}", err);
    assert!(msg.contains("Invalid characters"));
}

#[test]
fn validation_error_blocked_content() {
    let err = ValidationError::BlockedContent("malware detected".into());
    let msg = format!("{}", err);
    assert!(msg.contains("malware detected"));
}

#[test]
fn validation_error_missing_field() {
    let err = ValidationError::MissingField("prompt".into());
    let msg = format!("{}", err);
    assert!(msg.contains("prompt"));
}

#[test]
fn validation_error_invalid_field_value() {
    let err = ValidationError::InvalidFieldValue("temperature out of range".into());
    let msg = format!("{}", err);
    assert!(msg.contains("temperature out of range"));
}

#[test]
fn validation_error_debug() {
    let err = ValidationError::PromptTooLong(100, 50);
    let dbg = format!("{:?}", err);
    assert!(dbg.contains("PromptTooLong"));
}

// ── SecurityValidator ────────────────────────────────────────────

#[test]
fn validator_new_default_config() {
    let cfg = SecurityConfig::default();
    let validator = SecurityValidator::new(cfg).unwrap();
    assert_eq!(validator.config().max_prompt_length, 8192);
}

#[test]
fn validator_new_no_content_filtering() {
    let cfg = SecurityConfig { content_filtering: false, ..Default::default() };
    let validator = SecurityValidator::new(cfg).unwrap();
    assert!(!validator.config().content_filtering);
}

#[test]
fn validator_config_accessor() {
    let cfg = SecurityConfig {
        max_prompt_length: 500,
        max_tokens_per_request: 100,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    assert_eq!(validator.config().max_prompt_length, 500);
    assert_eq!(validator.config().max_tokens_per_request, 100);
}

#[test]
fn validator_valid_request() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "Hello, world!".into(),
        max_tokens: Some(100),
        model: None,
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(50),
        repetition_penalty: Some(1.1),
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_prompt_too_long() {
    let cfg = SecurityConfig {
        max_prompt_length: 10,
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "a".repeat(20),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = validator.validate_inference_request(&request).unwrap_err();
    assert!(matches!(err, ValidationError::PromptTooLong(20, 10)));
}

#[test]
fn validator_too_many_tokens() {
    let cfg = SecurityConfig {
        max_tokens_per_request: 50,
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: Some(100),
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = validator.validate_inference_request(&request).unwrap_err();
    assert!(matches!(err, ValidationError::TooManyTokens(100, 50)));
}

#[test]
fn validator_temperature_too_high() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: Some(3.0),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let err = validator.validate_inference_request(&request).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
}

#[test]
fn validator_temperature_negative() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: Some(-0.1),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_top_p_out_of_range() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: Some(1.5),
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_top_k_zero() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(0),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_top_k_too_large() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(2000),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_repetition_penalty_too_low() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: Some(0.01),
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_repetition_penalty_too_high() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: Some(11.0),
    };
    assert!(validator.validate_inference_request(&request).is_err());
}

#[test]
fn validator_boundary_temperature_zero() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_boundary_temperature_two() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: Some(2.0),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_boundary_top_k_one() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(1),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_boundary_top_k_1000() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "test".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(1000),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_all_none_params() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&request).is_ok());
}

#[test]
fn validator_empty_prompt_passes() {
    let cfg = SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..Default::default()
    };
    let validator = SecurityValidator::new(cfg).unwrap();
    let request = InferenceRequest {
        prompt: "".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    // Empty prompt is within length limit
    assert!(validator.validate_inference_request(&request).is_ok());
}

// ── extract_client_ip_from_headers ───────────────────────────────

#[test]
fn extract_ip_empty_headers() {
    let headers = axum::http::HeaderMap::new();
    let ip = extract_client_ip_from_headers(&headers);
    assert!(ip.is_none());
}

#[test]
fn extract_ip_from_x_forwarded_for() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
    let ip = extract_client_ip_from_headers(&headers);
    assert_eq!(ip, Some("1.2.3.4".parse::<IpAddr>().unwrap()));
}

#[test]
fn extract_ip_from_x_real_ip() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("x-real-ip", "10.0.0.1".parse().unwrap());
    let ip = extract_client_ip_from_headers(&headers);
    assert_eq!(ip, Some("10.0.0.1".parse::<IpAddr>().unwrap()));
}
