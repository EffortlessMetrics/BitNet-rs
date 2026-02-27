/// Property-based and unit tests for bitnet-server.
///
/// Tests cover:
/// - BatchEngineConfig invariants (valid ranges for all fields)
/// - RequestPriority ordering properties
/// - BatchRequest builder pattern invariants
/// - SecurityConfig field bounds (prompt length, token limits, temperature)
/// - SecurityValidator input validation properties (valid and invalid ranges)
/// - validate_json_payload: valid JSON parses; oversized payloads rejected
/// - ConcurrencyConfig invariants
/// - Health endpoint HTTP 200 response
/// - CORS reflected-origin behaviour
use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
    routing::get,
};
use bitnet_server::InferenceRequest;
use bitnet_server::batch_engine::{BatchEngineConfig, BatchRequest, RequestPriority};
use bitnet_server::concurrency::ConcurrencyConfig;
use bitnet_server::monitoring::{
    MonitoringConfig,
    health::{HealthChecker, create_health_routes},
    metrics::MetricsCollector,
};
use bitnet_server::security::{self, SecurityConfig, SecurityValidator, validate_json_payload};
use proptest::prelude::*;
use tower::ServiceExt;

proptest! {
    /// BatchEngineConfig: max_batch_size > 0 is always valid
    #[test]
    fn prop_batch_engine_config_max_batch_size_is_positive(
        max_batch_size in 1usize..=256usize
    ) {
        let config = BatchEngineConfig {
            max_batch_size,
            ..Default::default()
        };
        prop_assert!(config.max_batch_size > 0);
    }

    /// BatchEngineConfig: max_concurrent_batches > 0 is always valid
    #[test]
    fn prop_batch_engine_config_concurrent_batches_positive(
        max_concurrent_batches in 1usize..=32usize
    ) {
        let config = BatchEngineConfig {
            max_concurrent_batches,
            ..Default::default()
        };
        prop_assert!(config.max_concurrent_batches > 0);
    }

    /// RequestPriority: ordering is transitive
    /// Low < Normal < High < Critical
    #[test]
    fn prop_request_priority_ordering_is_consistent(
        _seed in 0u32..100u32
    ) {
        prop_assert!(RequestPriority::Low < RequestPriority::Normal);
        prop_assert!(RequestPriority::Normal < RequestPriority::High);
        prop_assert!(RequestPriority::High < RequestPriority::Critical);
        prop_assert!(RequestPriority::Low < RequestPriority::Critical);
    }

    /// BatchRequest::new always produces a non-empty ID
    #[test]
    fn prop_batch_request_new_has_nonempty_id(
        prompt in "[a-z ]{1,100}"
    ) {
        use bitnet_inference::GenerationConfig;
        let config = GenerationConfig::default();
        let prompt: String = prompt;
        let request = BatchRequest::new(prompt.clone(), config);
        prop_assert!(!request.id.is_empty(), "BatchRequest id should not be empty");
        prop_assert_eq!(&request.prompt, &prompt);
        prop_assert_eq!(request.priority, RequestPriority::Normal);
    }

    /// BatchRequest::with_priority preserves the priority
    #[test]
    fn prop_batch_request_with_priority_preserves_priority(
        priority_val in 0u8..=3u8
    ) {
        use bitnet_inference::GenerationConfig;
        let priority = match priority_val {
            0 => RequestPriority::Low,
            1 => RequestPriority::Normal,
            2 => RequestPriority::High,
            _ => RequestPriority::Critical,
        };
        let config = GenerationConfig::default();
        let request = BatchRequest::new("test".to_string(), config)
            .with_priority(priority);
        prop_assert_eq!(request.priority, priority);
    }

    /// SecurityConfig: max_prompt_length ≥ 1 is always valid
    #[test]
    fn prop_security_config_prompt_length_positive(
        max_prompt_length in 1usize..=65536usize
    ) {
        let config = SecurityConfig {
            max_prompt_length,
            ..Default::default()
        };
        prop_assert!(config.max_prompt_length >= 1);
    }

    /// SecurityConfig: max_tokens_per_request ≥ 1 is valid
    #[test]
    fn prop_security_config_max_tokens_positive(
        max_tokens in 1u32..=8192u32
    ) {
        let config = SecurityConfig {
            max_tokens_per_request: max_tokens,
            ..Default::default()
        };
        prop_assert!(config.max_tokens_per_request >= 1);
    }

    /// SecurityValidator: prompts within max_prompt_length always pass length check
    #[test]
    fn prop_security_validator_short_prompt_passes(
        prompt in "[a-z ]{1,100}"
    ) {
        let config = SecurityConfig {
            max_prompt_length: 8192,
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: prompt.clone(),
            max_tokens: Some(32),
            model: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "short prompt should pass validation: {:?}",
            prompt
        );
    }

    /// SecurityValidator: prompts exceeding max_prompt_length always fail
    #[test]
    fn prop_security_validator_long_prompt_fails(
        extra in 1usize..=100usize
    ) {
        let max_len = 50usize;
        let config = SecurityConfig {
            max_prompt_length: max_len,
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let long_prompt = "a".repeat(max_len + extra);
        let request = InferenceRequest {
            prompt: long_prompt,
            max_tokens: Some(32),
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_err(),
            "prompt exceeding max_prompt_length should fail validation"
        );
    }

    /// SecurityValidator: temperature in [0.0, 2.0] always passes
    #[test]
    fn prop_security_validator_valid_temperature_passes(
        temp_int in 0u32..=200u32  // represents 0.0..=2.0 in steps of 0.01
    ) {
        let temperature = temp_int as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: Some(temperature),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "temperature {temperature} in [0.0, 2.0] should pass validation"
        );
    }

    /// SecurityValidator: top_p in (0.0, 1.0] always passes
    #[test]
    fn prop_security_validator_valid_top_p_passes(
        top_p_int in 1u32..=100u32  // represents 0.01..=1.0 in steps of 0.01
    ) {
        let top_p = top_p_int as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: None,
            top_p: Some(top_p),
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "top_p {top_p} in (0.0, 1.0] should pass validation"
        );
    }

    /// SecurityValidator: temperature above 2.0 always fails
    #[test]
    fn prop_security_validator_temperature_above_2_fails(
        excess in 1u32..=200u32  // represents 0.01..=2.0 excess above 2.0
    ) {
        let temperature = 2.0 + excess as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: Some(temperature),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_err(),
            "temperature {temperature} above 2.0 should fail validation"
        );
    }

    /// SecurityValidator: top_p above 1.0 always fails
    #[test]
    fn prop_security_validator_top_p_above_1_fails(
        excess in 1u32..=100u32  // represents 0.01..=1.0 excess above 1.0
    ) {
        let top_p = 1.0 + excess as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: None,
            top_p: Some(top_p),
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_err(),
            "top_p {top_p} above 1.0 should fail validation"
        );
    }

    /// validate_json_payload: valid JSON within size limit always parses
    #[test]
    fn prop_validate_json_payload_valid_json_parses(
        prompt in "[a-z ]{1,50}",
        max_tokens in 1u32..=512u32,
    ) {
        let json = format!(
            r#"{{"prompt": "{}", "max_tokens": {}}}"#,
            prompt, max_tokens
        );
        let max_size = json.len() + 1024;
        let result: Result<InferenceRequest, _> = validate_json_payload(&json, max_size);
        prop_assert!(
            result.is_ok(),
            "valid JSON within size limit should parse successfully"
        );
    }

    /// validate_json_payload: payload exceeding size limit is always rejected
    #[test]
    fn prop_validate_json_payload_rejects_oversized(
        extra in 1usize..=64usize
    ) {
        let json = r#"{"prompt": "hello", "max_tokens": 32}"#;
        let max_size = json.len() - extra.min(json.len() - 1);
        let result: Result<InferenceRequest, _> = validate_json_payload(json, max_size);
        prop_assert!(
            result.is_err(),
            "payload exceeding max_size should be rejected"
        );
    }

    /// ConcurrencyConfig: max_concurrent_requests is always the value set
    #[test]
    fn prop_concurrency_config_max_concurrent_requests_preserved(
        n in 1usize..=1024usize
    ) {
        let config = ConcurrencyConfig {
            max_concurrent_requests: n,
            ..Default::default()
        };
        prop_assert_eq!(config.max_concurrent_requests, n);
        prop_assert!(config.max_concurrent_requests > 0);
    }

    /// ConcurrencyConfig: backpressure_threshold in (0.0, 1.0] is preserved
    #[test]
    fn prop_concurrency_config_backpressure_threshold_preserved(
        threshold_int in 1u32..=100u32
    ) {
        let threshold = threshold_int as f64 / 100.0;
        let config = ConcurrencyConfig {
            backpressure_threshold: threshold,
            ..Default::default()
        };
        prop_assert!(
            (0.0..=1.0).contains(&config.backpressure_threshold),
            "backpressure_threshold {threshold} should remain in range"
        );
    }
}

// ── Deterministic HTTP/integration tests ────────────────────────────────────

/// Health endpoint returns HTTP 200 with no model loaded.
#[tokio::test]
async fn test_health_endpoint_returns_200() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).expect("metrics"));
    let checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(checker);

    let req = Request::builder().uri("/health").body(Body::empty()).unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "health endpoint must return 200");
}

/// CORS: a request from an allowed origin gets the origin reflected back.
#[tokio::test]
async fn test_cors_reflects_allowed_origin() {
    let config = SecurityConfig {
        allowed_origins: vec!["http://trusted.example.com".to_string()],
        ..Default::default()
    };
    let app =
        Router::new().route("/", get(|| async { "ok" })).layer(security::configure_cors(&config));

    let req = Request::builder()
        .method(Method::OPTIONS)
        .uri("/")
        .header("Origin", "http://trusted.example.com")
        .header("Access-Control-Request-Method", "GET")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let acao = resp
        .headers()
        .get("access-control-allow-origin")
        .expect("ACAO header must be present for allowed origin");
    assert_eq!(acao, "http://trusted.example.com");
}

/// CORS: a request from a disallowed origin gets no ACAO header.
#[tokio::test]
async fn test_cors_blocks_disallowed_origin() {
    let config = SecurityConfig {
        allowed_origins: vec!["http://trusted.example.com".to_string()],
        ..Default::default()
    };
    let app =
        Router::new().route("/", get(|| async { "ok" })).layer(security::configure_cors(&config));

    let req = Request::builder()
        .method(Method::OPTIONS)
        .uri("/")
        .header("Origin", "http://evil.example.com")
        .header("Access-Control-Request-Method", "GET")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "ACAO header must not be present for disallowed origin"
    );
}

/// validate_json_payload: syntactically invalid JSON returns an error.
#[test]
fn test_validate_json_invalid_syntax_returns_error() {
    let bad_json = r#"{"prompt": "hello", "max_tokens": }"#; // broken JSON
    let result: Result<InferenceRequest, _> = validate_json_payload(bad_json, 1024);
    assert!(result.is_err(), "invalid JSON syntax should return an error");
}
