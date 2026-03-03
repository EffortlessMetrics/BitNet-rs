//! Edge-case tests for server security validation, concurrency, and execution router types.
//!
//! Tests cover: SecurityConfig defaults, SecurityValidator (prompt length, token limits,
//! parameter ranges, sanitization, content filtering, model path validation),
//! ValidationError display, ConcurrencyConfig, CircuitBreakerState, RequestAdmission,
//! ConcurrencyManager (admission, circuit breaker), DeviceStats, DeviceCapabilities,
//! DeviceSelectionStrategy, ExecutionRouterConfig, DeviceHealth.

use std::net::{IpAddr, Ipv4Addr};
use std::time::{Duration, Instant};

use bitnet_server::concurrency::{
    CircuitBreakerState, ConcurrencyConfig, ConcurrencyManager, ConcurrencyStats, RequestMetadata,
};
use bitnet_server::execution_router::{
    DeviceCapabilities, DeviceHealth, DeviceSelectionStrategy, DeviceStats, ExecutionRouterConfig,
};
use bitnet_server::security::{Claims, SecurityConfig, SecurityValidator, ValidationError};

// ---------------------------------------------------------------------------
// SecurityConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn security_config_default_values() {
    let config = SecurityConfig::default();
    assert_eq!(config.max_prompt_length, 8192);
    assert_eq!(config.max_tokens_per_request, 2048);
    assert!(config.input_sanitization);
    assert!(config.content_filtering);
    assert!(config.rate_limit_by_ip);
    assert!(!config.require_authentication);
    assert!(config.jwt_secret.is_none());
    assert!(config.blocked_ips.is_empty());
}

#[test]
fn security_config_default_allowed_origins() {
    let config = SecurityConfig::default();
    assert_eq!(config.allowed_origins, vec!["*".to_string()]);
}

#[test]
fn security_config_serde_roundtrip() {
    let config = SecurityConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: SecurityConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_prompt_length, config.max_prompt_length);
    assert_eq!(parsed.max_tokens_per_request, config.max_tokens_per_request);
}

#[test]
fn security_config_debug() {
    let config = SecurityConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("SecurityConfig"));
}

// ---------------------------------------------------------------------------
// SecurityValidator — prompt length
// ---------------------------------------------------------------------------

fn make_request(prompt: &str) -> bitnet_server::InferenceRequest {
    serde_json::from_value(serde_json::json!({
        "prompt": prompt
    }))
    .unwrap()
}

fn make_request_with_params(
    prompt: &str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    rep_penalty: Option<f32>,
) -> bitnet_server::InferenceRequest {
    let mut val = serde_json::json!({ "prompt": prompt });
    if let Some(mt) = max_tokens {
        val["max_tokens"] = serde_json::json!(mt);
    }
    if let Some(t) = temperature {
        val["temperature"] = serde_json::json!(t);
    }
    if let Some(tp) = top_p {
        val["top_p"] = serde_json::json!(tp);
    }
    if let Some(tk) = top_k {
        val["top_k"] = serde_json::json!(tk);
    }
    if let Some(rp) = rep_penalty {
        val["repetition_penalty"] = serde_json::json!(rp);
    }
    serde_json::from_value(val).unwrap()
}

#[test]
fn validator_accepts_short_prompt() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request("Hello, world!");
    assert!(validator.validate_inference_request(&req).is_ok());
}

#[test]
fn validator_rejects_prompt_too_long() {
    let mut config = SecurityConfig::default();
    config.max_prompt_length = 10;
    let validator = SecurityValidator::new(config).unwrap();
    let req = make_request("This prompt is definitely longer than ten characters");
    let result = validator.validate_inference_request(&req);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, ValidationError::PromptTooLong(_, 10)));
}

#[test]
fn validator_accepts_prompt_at_exact_limit() {
    let mut config = SecurityConfig::default();
    config.max_prompt_length = 5;
    let validator = SecurityValidator::new(config).unwrap();
    let req = make_request("12345");
    assert!(validator.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// SecurityValidator — max tokens
// ---------------------------------------------------------------------------

#[test]
fn validator_accepts_tokens_within_limit() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", Some(100), None, None, None, None);
    assert!(validator.validate_inference_request(&req).is_ok());
}

#[test]
fn validator_rejects_tokens_over_limit() {
    let mut config = SecurityConfig::default();
    config.max_tokens_per_request = 50;
    let validator = SecurityValidator::new(config).unwrap();
    let req = make_request_with_params("hi", Some(100), None, None, None, None);
    let result = validator.validate_inference_request(&req);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ValidationError::TooManyTokens(100, 50)));
}

#[test]
fn validator_accepts_no_max_tokens() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, None, None, None);
    assert!(validator.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// SecurityValidator — parameter ranges
// ---------------------------------------------------------------------------

#[test]
fn validator_accepts_valid_temperature() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    for temp in [0.0, 0.5, 1.0, 2.0] {
        let req = make_request_with_params("hi", None, Some(temp), None, None, None);
        assert!(validator.validate_inference_request(&req).is_ok());
    }
}

#[test]
fn validator_rejects_negative_temperature() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, Some(-0.1), None, None, None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_temperature_over_2() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, Some(2.5), None, None, None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_accepts_valid_top_p() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    for tp in [0.0, 0.5, 1.0] {
        let req = make_request_with_params("hi", None, None, Some(tp), None, None);
        assert!(validator.validate_inference_request(&req).is_ok());
    }
}

#[test]
fn validator_rejects_top_p_over_1() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, Some(1.5), None, None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_negative_top_p() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, Some(-0.1), None, None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_accepts_valid_top_k() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    for k in [1, 50, 1000] {
        let req = make_request_with_params("hi", None, None, None, Some(k), None);
        assert!(validator.validate_inference_request(&req).is_ok());
    }
}

#[test]
fn validator_rejects_top_k_zero() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, None, Some(0), None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_top_k_over_1000() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, None, Some(1001), None);
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_accepts_valid_repetition_penalty() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    for rp in [0.1, 1.0, 5.0, 10.0] {
        let req = make_request_with_params("hi", None, None, None, None, Some(rp));
        assert!(validator.validate_inference_request(&req).is_ok());
    }
}

#[test]
fn validator_rejects_repetition_penalty_too_low() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, None, None, Some(0.05));
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_repetition_penalty_too_high() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request_with_params("hi", None, None, None, None, Some(10.5));
    assert!(validator.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// SecurityValidator — input sanitization
// ---------------------------------------------------------------------------

#[test]
fn validator_rejects_null_bytes() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request("Hello\0world");
    let result = validator.validate_inference_request(&req);
    assert!(matches!(result, Err(ValidationError::InvalidCharacters)));
}

#[test]
fn validator_allows_newlines_and_tabs() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request("Hello\nworld\ttab\r\nend");
    assert!(validator.validate_inference_request(&req).is_ok());
}

#[test]
fn validator_rejects_excessively_long_lines() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let long_line = "a".repeat(1025);
    let req = make_request(&long_line);
    let result = validator.validate_inference_request(&req);
    assert!(matches!(result, Err(ValidationError::InvalidCharacters)));
}

#[test]
fn validator_accepts_lines_at_1024() {
    let mut config = SecurityConfig::default();
    config.max_prompt_length = 2000;
    let validator = SecurityValidator::new(config).unwrap();
    let line = "a".repeat(1024);
    let req = make_request(&line);
    assert!(validator.validate_inference_request(&req).is_ok());
}

#[test]
fn validator_skips_sanitization_when_disabled() {
    let mut config = SecurityConfig::default();
    config.input_sanitization = false;
    let validator = SecurityValidator::new(config).unwrap();
    let req = make_request("Hello\0world");
    // Content filtering may still block, but sanitization won't
    // If content is fine, it should pass
    let result = validator.validate_inference_request(&req);
    // null byte passes when sanitization is disabled
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// SecurityValidator — content filtering
// ---------------------------------------------------------------------------

#[test]
fn validator_blocks_content_with_hack() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request("How to hack a system");
    let result = validator.validate_inference_request(&req);
    assert!(matches!(result, Err(ValidationError::BlockedContent(_))));
}

#[test]
fn validator_blocks_sql_injection_mention() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = make_request("sql injection attack");
    let result = validator.validate_inference_request(&req);
    assert!(matches!(result, Err(ValidationError::BlockedContent(_))));
}

#[test]
fn validator_skips_filtering_when_disabled() {
    let mut config = SecurityConfig::default();
    config.content_filtering = false;
    let validator = SecurityValidator::new(config).unwrap();
    let req = make_request("How to hack a system");
    assert!(validator.validate_inference_request(&req).is_ok());
}

// ---------------------------------------------------------------------------
// SecurityValidator — model path validation
// ---------------------------------------------------------------------------

#[test]
fn validator_accepts_valid_gguf_path() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("models/phi4.gguf").is_ok());
}

#[test]
fn validator_accepts_valid_safetensors_path() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("models/model.safetensors").is_ok());
}

#[test]
fn validator_rejects_empty_path() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let result = validator.validate_model_request("");
    assert!(matches!(result, Err(ValidationError::MissingField(_))));
}

#[test]
fn validator_rejects_path_traversal() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let result = validator.validate_model_request("../../etc/passwd.gguf");
    assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
}

#[test]
fn validator_rejects_tilde_in_path() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let result = validator.validate_model_request("~/models/model.gguf");
    assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
}

#[test]
fn validator_rejects_wrong_extension() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let result = validator.validate_model_request("models/model.bin");
    assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
}

#[test]
fn validator_rejects_no_extension() {
    let validator = SecurityValidator::new(SecurityConfig::default()).unwrap();
    let result = validator.validate_model_request("models/model");
    assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
}

// ---------------------------------------------------------------------------
// ValidationError display
// ---------------------------------------------------------------------------

#[test]
fn validation_error_prompt_too_long_display() {
    let err = ValidationError::PromptTooLong(5000, 4096);
    let msg = format!("{err}");
    assert!(msg.contains("5000"));
    assert!(msg.contains("4096"));
}

#[test]
fn validation_error_too_many_tokens_display() {
    let err = ValidationError::TooManyTokens(10000, 2048);
    let msg = format!("{err}");
    assert!(msg.contains("10000"));
    assert!(msg.contains("2048"));
}

#[test]
fn validation_error_debug() {
    let err = ValidationError::InvalidCharacters;
    let dbg = format!("{err:?}");
    assert!(dbg.contains("InvalidCharacters"));
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

#[test]
fn claims_serde_roundtrip() {
    let claims = Claims {
        sub: "user123".into(),
        exp: 9999999999,
        iat: 1000000000,
        role: Some("admin".into()),
        rate_limit: Some(100),
    };
    let json = serde_json::to_string(&claims).unwrap();
    let parsed: Claims = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.sub, "user123");
    assert_eq!(parsed.exp, 9999999999);
    assert_eq!(parsed.role, Some("admin".into()));
    assert_eq!(parsed.rate_limit, Some(100));
}

#[test]
fn claims_optional_fields() {
    let claims = Claims { sub: "user".into(), exp: 0, iat: 0, role: None, rate_limit: None };
    let json = serde_json::to_string(&claims).unwrap();
    assert!(json.contains("null") || !json.contains("role"));
}

// ---------------------------------------------------------------------------
// ConcurrencyConfig
// ---------------------------------------------------------------------------

#[test]
fn concurrency_config_defaults() {
    let config = ConcurrencyConfig::default();
    assert_eq!(config.max_concurrent_requests, 100);
    assert_eq!(config.max_requests_per_second, 50);
    assert_eq!(config.max_requests_per_minute, 1000);
    assert_eq!(config.rate_limit_window, Duration::from_secs(60));
    assert!((config.backpressure_threshold - 0.8).abs() < f64::EPSILON);
    assert!(config.circuit_breaker_enabled);
    assert_eq!(config.circuit_breaker_failure_threshold, 10);
    assert_eq!(config.circuit_breaker_timeout, Duration::from_secs(30));
    assert_eq!(config.per_ip_rate_limit, Some(10));
    assert_eq!(config.global_rate_limit, Some(100));
}

// ---------------------------------------------------------------------------
// CircuitBreakerState
// ---------------------------------------------------------------------------

#[test]
fn circuit_breaker_state_eq() {
    assert_eq!(CircuitBreakerState::Closed, CircuitBreakerState::Closed);
    assert_ne!(CircuitBreakerState::Open, CircuitBreakerState::Closed);
    assert_ne!(CircuitBreakerState::HalfOpen, CircuitBreakerState::Closed);
}

#[test]
fn circuit_breaker_state_debug() {
    assert!(format!("{:?}", CircuitBreakerState::Closed).contains("Closed"));
    assert!(format!("{:?}", CircuitBreakerState::Open).contains("Open"));
    assert!(format!("{:?}", CircuitBreakerState::HalfOpen).contains("HalfOpen"));
}

#[test]
fn circuit_breaker_state_clone() {
    let state = CircuitBreakerState::HalfOpen;
    let cloned = state.clone();
    assert_eq!(state, cloned);
}

#[test]
fn circuit_breaker_state_serialize() {
    let json = serde_json::to_string(&CircuitBreakerState::Open).unwrap();
    assert!(!json.is_empty());
}

// ---------------------------------------------------------------------------
// ConcurrencyManager — admission
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrency_manager_admits_request() {
    let config = ConcurrencyConfig::default();
    let manager = ConcurrencyManager::new(config);
    let metadata = RequestMetadata {
        id: "req-1".into(),
        client_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        user_agent: None,
        start_time: Instant::now(),
        priority: bitnet_server::batch_engine::RequestPriority::Normal,
    };
    let result = manager.should_admit_request(&metadata).await.unwrap();
    assert!(matches!(result, bitnet_server::concurrency::RequestAdmission::Admitted));
}

#[tokio::test]
async fn concurrency_manager_stats_initial() {
    let config = ConcurrencyConfig::default();
    let manager = ConcurrencyManager::new(config);
    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.rejected_requests, 0);
    assert_eq!(stats.max_concurrent_requests, 100);
}

#[tokio::test]
async fn concurrency_manager_health_initial() {
    let config = ConcurrencyConfig::default();
    let manager = ConcurrencyManager::new(config);
    let health = manager.get_health().await;
    assert!(health.healthy);
    assert_eq!(health.circuit_breaker_state, CircuitBreakerState::Closed);
}

#[tokio::test]
async fn concurrency_manager_cleanup_empty() {
    let config = ConcurrencyConfig::default();
    let manager = ConcurrencyManager::new(config);
    // Should not panic with empty rate limiters
    manager.cleanup_rate_limiters(Duration::from_secs(60)).await;
}

// ---------------------------------------------------------------------------
// DeviceStats
// ---------------------------------------------------------------------------

#[test]
fn device_stats_new_zeros() {
    let stats = DeviceStats::new();
    assert_eq!(stats.get_avg_tokens_per_second(), 0.0);
}

#[tokio::test]
async fn device_stats_record_success() {
    let stats = DeviceStats::new();
    stats.record_success(100, Duration::from_secs(1)).await;
    let tps = stats.get_avg_tokens_per_second();
    assert!((tps - 100.0).abs() < 1.0);
}

#[test]
fn device_stats_record_failure_increments() {
    let stats = DeviceStats::new();
    stats.record_failure();
    stats.record_failure();
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[tokio::test]
async fn device_stats_success_resets_failures() {
    let stats = DeviceStats::new();
    stats.record_failure();
    stats.record_failure();
    stats.record_success(10, Duration::from_millis(500)).await;
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn device_stats_clone() {
    let stats = DeviceStats::new();
    stats.record_failure();
    let cloned = stats.clone();
    assert_eq!(cloned.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 1);
}

// ---------------------------------------------------------------------------
// DeviceCapabilities
// ---------------------------------------------------------------------------

#[test]
fn device_capabilities_serde_roundtrip() {
    let cap = DeviceCapabilities {
        device: bitnet_common::Device::Cpu,
        available: true,
        memory_total_mb: 16384,
        memory_free_mb: 8192,
        compute_capability: None,
        simd_support: vec!["AVX2".into()],
        avg_tokens_per_second: 35.0,
        last_benchmark: None,
    };
    let json = serde_json::to_string(&cap).unwrap();
    let parsed: DeviceCapabilities = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.memory_total_mb, 16384);
    assert!(parsed.available);
}

// ---------------------------------------------------------------------------
// DeviceSelectionStrategy
// ---------------------------------------------------------------------------

#[test]
fn device_selection_strategy_debug() {
    let strats = [
        DeviceSelectionStrategy::PreferGpu,
        DeviceSelectionStrategy::CpuOnly,
        DeviceSelectionStrategy::PerformanceBased,
        DeviceSelectionStrategy::LoadBalance,
        DeviceSelectionStrategy::UserPreference(bitnet_common::Device::Cpu),
    ];
    for s in &strats {
        let dbg = format!("{s:?}");
        assert!(!dbg.is_empty());
    }
}

#[test]
fn device_selection_strategy_serde_roundtrip() {
    let strat = DeviceSelectionStrategy::PerformanceBased;
    let json = serde_json::to_string(&strat).unwrap();
    let parsed: DeviceSelectionStrategy = serde_json::from_str(&json).unwrap();
    assert!(format!("{parsed:?}").contains("PerformanceBased"));
}

// ---------------------------------------------------------------------------
// ExecutionRouterConfig
// ---------------------------------------------------------------------------

#[test]
fn execution_router_config_defaults() {
    let config = ExecutionRouterConfig::default();
    assert!(config.fallback_enabled);
    assert_eq!(config.health_check_interval, Duration::from_secs(30));
    assert!((config.performance_threshold_tps - 10.0).abs() < f64::EPSILON);
    assert!((config.memory_threshold_percent - 0.8).abs() < f64::EPSILON);
    assert!(config.benchmark_on_startup);
}

#[test]
fn execution_router_config_serde_roundtrip() {
    let config = ExecutionRouterConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: ExecutionRouterConfig = serde_json::from_str(&json).unwrap();
    assert!(parsed.fallback_enabled);
}

// ---------------------------------------------------------------------------
// DeviceHealth
// ---------------------------------------------------------------------------

#[test]
fn device_health_debug() {
    let h1 = DeviceHealth::Healthy;
    let h2 = DeviceHealth::Degraded { reason: "mem high".into() };
    let h3 = DeviceHealth::Unavailable { reason: "offline".into() };
    assert!(format!("{h1:?}").contains("Healthy"));
    assert!(format!("{h2:?}").contains("Degraded"));
    assert!(format!("{h3:?}").contains("Unavailable"));
}

#[test]
fn device_health_clone() {
    let h = DeviceHealth::Degraded { reason: "test".into() };
    let cloned = h.clone();
    assert!(format!("{cloned:?}").contains("test"));
}

#[test]
fn device_health_serialize() {
    let h = DeviceHealth::Unavailable { reason: "GPU down".into() };
    let json = serde_json::to_string(&h).unwrap();
    assert!(json.contains("GPU down"));
}

// ---------------------------------------------------------------------------
// ConcurrencyStats
// ---------------------------------------------------------------------------

#[test]
fn concurrency_stats_serde() {
    let stats = ConcurrencyStats {
        active_requests: 5,
        max_concurrent_requests: 100,
        current_load: 0.05,
        total_requests: 1000,
        rejected_requests: 10,
        backpressure_activations: 2,
        circuit_breaker_state: "Closed".into(),
        available_permits: 95,
        per_ip_limiter_count: 42,
    };
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("active_requests"));
    assert!(json.contains("95"));
}
