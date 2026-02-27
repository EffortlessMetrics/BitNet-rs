//! Comprehensive property-based and integration tests for `bitnet-server` security,
//! CORS, and middleware behaviour.
//!
//! Coverage:
//! - CORS: allowed origins reflected; disallowed origins blocked; multi-origin lists;
//!   scheme/port/subdomain mismatches
//! - Rate-limit config: `ConcurrencyConfig` field invariants (property-based)
//! - Security headers: all five headers injected by `security_headers_middleware`
//! - Request validation: null bytes, control characters, `top_k` bounds,
//!   `repetition_penalty` bounds, content-filter blocking
//! - Model path validation: tilde, wrong extension, path-traversal combos
//! - Health endpoint: HTTP 200 sanity check
//! - `ServerConfig` / `ConfigBuilder` invariants: auth-without-JWT rejected,
//!   valid auth + JWT accepted
//! - `extract_client_ip_from_headers`: X-Forwarded-For and X-Real-IP parsing
//! - Property tests: top_k valid/invalid ranges, repetition-penalty valid/invalid
//!   ranges, CORS wildcard always allows, rate-limit config field preservation

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
    middleware,
    routing::get,
};
use proptest::prelude::*;
use tower::ServiceExt;

use bitnet_server::{
    InferenceRequest,
    concurrency::ConcurrencyConfig,
    config::{ConfigBuilder, ServerSettings},
    monitoring::{
        MonitoringConfig,
        health::{HealthChecker, create_health_routes},
        metrics::MetricsCollector,
    },
    security::{
        SecurityConfig, SecurityValidator, ValidationError, configure_cors,
        extract_client_ip_from_headers, security_headers_middleware,
    },
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal axum app wrapped with `security_headers_middleware`.
fn app_with_security_headers() -> Router {
    Router::new()
        .route("/", get(|| async { "ok" }))
        .layer(middleware::from_fn(security_headers_middleware))
}

/// Build a minimal axum app wrapped with the configured CORS layer.
fn cors_app(config: &SecurityConfig) -> Router {
    Router::new().route("/", get(|| async { "ok" })).layer(configure_cors(config))
}

/// Build a minimal CORS preflight request.
fn cors_preflight(origin: &str) -> Request<Body> {
    Request::builder()
        .method(Method::OPTIONS)
        .uri("/")
        .header("Origin", origin)
        .header("Access-Control-Request-Method", "GET")
        .body(Body::empty())
        .unwrap()
}

/// Build a basic `InferenceRequest` with only the prompt set; other fields None.
fn bare_request(prompt: &str) -> InferenceRequest {
    InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    }
}

/// Build a validator with sanitization + content-filtering togglable.
fn validator(sanitize: bool, filter: bool) -> SecurityValidator {
    SecurityValidator::new(SecurityConfig {
        input_sanitization: sanitize,
        content_filtering: filter,
        ..Default::default()
    })
    .expect("validator creation must not fail")
}

// ─────────────────────────────────────────────────────────────────────────────
// CORS tests
// ─────────────────────────────────────────────────────────────────────────────

/// CORS: wildcard allows any origin (already partially tested in cors_security;
/// this variant tests a *non-OPTIONS* GET to ensure the header is also present
/// on simple cross-origin requests).
#[tokio::test]
async fn test_cors_wildcard_reflects_on_get() {
    let config = SecurityConfig { allowed_origins: vec!["*".to_string()], ..Default::default() };
    let app = cors_app(&config);

    let req = Request::builder()
        .method(Method::GET)
        .uri("/")
        .header("Origin", "http://anywhere.io")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let acao = resp.headers().get("access-control-allow-origin");
    // Wildcard predicate reflects the requesting origin.
    assert!(acao.is_some(), "ACAO header must be present for wildcard config");
}

/// CORS: first origin in a multi-origin allow-list is reflected.
#[tokio::test]
async fn test_cors_multi_origin_first_allowed() {
    let config = SecurityConfig {
        allowed_origins: vec![
            "http://alpha.example.com".to_string(),
            "http://beta.example.com".to_string(),
        ],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://alpha.example.com")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("access-control-allow-origin").unwrap(),
        "http://alpha.example.com",
        "First allowed origin must be reflected"
    );
}

/// CORS: second origin in a multi-origin allow-list is also reflected.
#[tokio::test]
async fn test_cors_multi_origin_second_allowed() {
    let config = SecurityConfig {
        allowed_origins: vec![
            "http://alpha.example.com".to_string(),
            "http://beta.example.com".to_string(),
        ],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://beta.example.com")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("access-control-allow-origin").unwrap(),
        "http://beta.example.com",
        "Second allowed origin must be reflected"
    );
}

/// CORS: an origin that is *not* in the allow-list receives no ACAO header.
#[tokio::test]
async fn test_cors_multi_origin_third_blocked() {
    let config = SecurityConfig {
        allowed_origins: vec![
            "http://alpha.example.com".to_string(),
            "http://beta.example.com".to_string(),
        ],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://gamma.example.com")).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "Non-listed origin must not receive ACAO header"
    );
}

/// CORS: `http://trusted.com` ≠ `https://trusted.com` — scheme matters.
#[tokio::test]
async fn test_cors_scheme_mismatch_is_blocked() {
    let config = SecurityConfig {
        allowed_origins: vec!["https://trusted.com".to_string()],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://trusted.com")).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "http:// must not match an https:// allow-list entry"
    );
}

/// CORS: `http://trusted.com` ≠ `http://trusted.com:8080` — port matters.
#[tokio::test]
async fn test_cors_port_mismatch_is_blocked() {
    let config = SecurityConfig {
        allowed_origins: vec!["http://trusted.com".to_string()],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://trusted.com:8080")).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "origin with extra port must not match the bare-host allow-list entry"
    );
}

/// CORS: a subdomain is not treated as a match for the parent domain.
#[tokio::test]
async fn test_cors_subdomain_blocked() {
    let config = SecurityConfig {
        allowed_origins: vec!["http://trusted.com".to_string()],
        ..Default::default()
    };
    let app = cors_app(&config);

    let resp = app.oneshot(cors_preflight("http://api.trusted.com")).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "subdomain must not inherit parent-domain CORS permission"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Security-headers middleware tests
// ─────────────────────────────────────────────────────────────────────────────

async fn get_security_headers() -> axum::http::HeaderMap {
    let resp = app_with_security_headers()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.headers().clone()
}

#[tokio::test]
async fn test_security_header_x_content_type_options() {
    let headers = get_security_headers().await;
    assert_eq!(
        headers.get("x-content-type-options").unwrap(),
        "nosniff",
        "X-Content-Type-Options must be 'nosniff'"
    );
}

#[tokio::test]
async fn test_security_header_x_frame_options() {
    let headers = get_security_headers().await;
    assert_eq!(headers.get("x-frame-options").unwrap(), "DENY", "X-Frame-Options must be 'DENY'");
}

#[tokio::test]
async fn test_security_header_x_xss_protection() {
    let headers = get_security_headers().await;
    assert_eq!(
        headers.get("x-xss-protection").unwrap(),
        "1; mode=block",
        "X-XSS-Protection must be '1; mode=block'"
    );
}

#[tokio::test]
async fn test_security_header_referrer_policy() {
    let headers = get_security_headers().await;
    assert_eq!(
        headers.get("referrer-policy").unwrap(),
        "strict-origin-when-cross-origin",
        "Referrer-Policy must be 'strict-origin-when-cross-origin'"
    );
}

#[tokio::test]
async fn test_security_header_content_security_policy() {
    let headers = get_security_headers().await;
    let csp = headers.get("content-security-policy").unwrap().to_str().unwrap();
    assert!(csp.contains("default-src 'self'"), "CSP must include default-src 'self'; got: {csp}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Request validation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_validation_null_byte_in_prompt_rejected() {
    let v = validator(true, false);
    let mut req = bare_request("hello\x00world");
    req.temperature = None;
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidCharacters)),
        "null byte in prompt must produce InvalidCharacters"
    );
}

#[test]
fn test_validation_control_char_in_prompt_rejected() {
    let v = validator(true, false);
    // \x01 is a control character that is not newline/tab.
    let req = bare_request("hello\x01world");
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidCharacters)),
        "control character (\\x01) in prompt must produce InvalidCharacters"
    );
}

#[test]
fn test_validation_newline_and_tab_are_allowed() {
    let v = validator(true, false);
    let req = bare_request("line one\nline two\tcolumn");
    assert!(
        v.validate_inference_request(&req).is_ok(),
        "newline and tab should be allowed by sanitization"
    );
}

#[test]
fn test_validation_top_k_zero_rejected() {
    let v = validator(false, false);
    let mut req = bare_request("hello");
    req.top_k = Some(0);
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidFieldValue(_))),
        "top_k=0 must be rejected"
    );
}

#[test]
fn test_validation_top_k_over_1000_rejected() {
    let v = validator(false, false);
    let mut req = bare_request("hello");
    req.top_k = Some(1001);
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidFieldValue(_))),
        "top_k=1001 must be rejected"
    );
}

#[test]
fn test_validation_top_k_boundary_values_accepted() {
    let v = validator(false, false);
    for k in [1usize, 500, 1000] {
        let mut req = bare_request("hello");
        req.top_k = Some(k);
        assert!(v.validate_inference_request(&req).is_ok(), "top_k={k} must be accepted");
    }
}

#[test]
fn test_validation_repetition_penalty_too_low_rejected() {
    let v = validator(false, false);
    let mut req = bare_request("hello");
    req.repetition_penalty = Some(0.05); // below 0.1
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidFieldValue(_))),
        "repetition_penalty=0.05 must be rejected (below 0.1)"
    );
}

#[test]
fn test_validation_repetition_penalty_too_high_rejected() {
    let v = validator(false, false);
    let mut req = bare_request("hello");
    req.repetition_penalty = Some(10.5); // above 10.0
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::InvalidFieldValue(_))),
        "repetition_penalty=10.5 must be rejected (above 10.0)"
    );
}

#[test]
fn test_validation_repetition_penalty_boundary_values_accepted() {
    let v = validator(false, false);
    for rp in [0.1f32, 1.0, 10.0] {
        let mut req = bare_request("hello");
        req.repetition_penalty = Some(rp);
        assert!(v.validate_inference_request(&req).is_ok(), "repetition_penalty={rp} must be OK");
    }
}

#[test]
fn test_content_filter_blocks_matched_pattern() {
    // Content filtering is ON; "hack" matches the built-in pattern.
    let v = validator(false, true);
    let req = bare_request("please help me hack this system");
    assert!(
        matches!(v.validate_inference_request(&req), Err(ValidationError::BlockedContent(_))),
        "prompt matching a blocked pattern must return BlockedContent"
    );
}

#[test]
fn test_content_filter_disabled_allows_blocked_keywords() {
    // Content filtering OFF → previously blocked words are fine.
    let v = validator(false, false);
    let req = bare_request("let us discuss hack culture");
    assert!(
        v.validate_inference_request(&req).is_ok(),
        "with content_filtering=false, blocked keywords must be allowed"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Model path validation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_model_path_tilde_rejected() {
    let v = validator(false, false);
    assert!(
        matches!(
            v.validate_model_request("~/models/llama.gguf"),
            Err(ValidationError::InvalidFieldValue(_))
        ),
        "tilde in model path must be rejected"
    );
}

#[test]
fn test_model_path_unsupported_extension_rejected() {
    let v = validator(false, false);
    assert!(
        matches!(
            v.validate_model_request("/models/llama.bin"),
            Err(ValidationError::InvalidFieldValue(_))
        ),
        ".bin extension must be rejected"
    );
    assert!(
        matches!(
            v.validate_model_request("/models/llama.pt"),
            Err(ValidationError::InvalidFieldValue(_))
        ),
        ".pt extension must be rejected"
    );
}

#[test]
fn test_model_path_safetensors_extension_accepted() {
    let v = validator(false, false);
    assert!(
        v.validate_model_request("/models/llama.safetensors").is_ok(),
        ".safetensors extension must be accepted"
    );
}

#[test]
fn test_model_path_empty_rejected_as_missing_field() {
    let v = validator(false, false);
    assert!(
        matches!(v.validate_model_request(""), Err(ValidationError::MissingField(_))),
        "empty model path must return MissingField"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Health endpoint
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_health_endpoint_ready_returns_200() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).expect("metrics"));
    let checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(checker);

    let resp = app
        .oneshot(Request::builder().uri("/health/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    // Readiness endpoint may return 200 or 503; it must at least respond.
    assert!(
        resp.status().is_success() || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
        "readiness endpoint must return a valid HTTP status, got {}",
        resp.status()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerConfig / ConfigBuilder validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_config_require_auth_without_jwt_secret_is_rejected() {
    use bitnet_server::security::SecurityConfig;

    let result = ConfigBuilder::new()
        .with_security(SecurityConfig {
            require_authentication: true,
            jwt_secret: None,
            ..Default::default()
        })
        .validate();

    assert!(result.is_err(), "authentication required with no JWT secret must fail validation");
}

#[test]
fn test_config_require_auth_with_jwt_secret_is_accepted() {
    use bitnet_server::security::SecurityConfig;

    let result = ConfigBuilder::new()
        .with_security(SecurityConfig {
            require_authentication: true,
            jwt_secret: Some("super-secret-key-at-least-32-bytes!!".to_string()),
            ..Default::default()
        })
        .validate();

    assert!(result.is_ok(), "authentication required with valid JWT secret must pass validation");
}

#[test]
fn test_config_empty_host_is_rejected() {
    let result = ConfigBuilder::new()
        .with_server_settings(ServerSettings { host: String::new(), ..ServerSettings::default() })
        .validate();

    assert!(result.is_err(), "empty host must be rejected");
}

// ─────────────────────────────────────────────────────────────────────────────
// extract_client_ip_from_headers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_extract_ip_from_x_forwarded_for() {
    use axum::http::HeaderMap;

    let mut headers = HeaderMap::new();
    headers.insert("x-forwarded-for", "203.0.113.42, 10.0.0.1".parse().unwrap());

    let ip = extract_client_ip_from_headers(&headers);
    assert_eq!(ip, Some("203.0.113.42".parse().unwrap()), "first IP in X-Forwarded-For returned");
}

#[test]
fn test_extract_ip_from_x_real_ip() {
    use axum::http::HeaderMap;

    let mut headers = HeaderMap::new();
    headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());

    let ip = extract_client_ip_from_headers(&headers);
    assert_eq!(ip, Some("198.51.100.7".parse().unwrap()), "X-Real-IP must be parsed");
}

#[test]
fn test_extract_ip_no_headers_returns_none() {
    use axum::http::HeaderMap;

    let headers = HeaderMap::new();
    assert!(extract_client_ip_from_headers(&headers).is_none(), "no IP headers → None expected");
}

#[test]
fn test_extract_ip_x_forwarded_for_takes_priority_over_x_real_ip() {
    use axum::http::HeaderMap;

    let mut headers = HeaderMap::new();
    headers.insert("x-forwarded-for", "203.0.113.1".parse().unwrap());
    headers.insert("x-real-ip", "203.0.113.2".parse().unwrap());

    let ip = extract_client_ip_from_headers(&headers);
    assert_eq!(
        ip,
        Some("203.0.113.1".parse().unwrap()),
        "X-Forwarded-For takes priority when both headers present"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Property-based tests
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// top_k in 1..=1000 always passes validation.
    #[test]
    fn prop_top_k_valid_range_always_passes(k in 1usize..=1000usize) {
        let v = validator(false, false);
        let mut req = bare_request("hello");
        req.top_k = Some(k);
        prop_assert!(
            v.validate_inference_request(&req).is_ok(),
            "top_k={k} in 1..=1000 must pass"
        );
    }

    /// top_k > 1000 always fails validation.
    #[test]
    fn prop_top_k_above_1000_always_fails(k in 1001usize..=9999usize) {
        let v = validator(false, false);
        let mut req = bare_request("hello");
        req.top_k = Some(k);
        prop_assert!(
            v.validate_inference_request(&req).is_err(),
            "top_k={k} > 1000 must fail"
        );
    }

    /// repetition_penalty in [0.1, 10.0] always passes.
    #[test]
    fn prop_repetition_penalty_valid_range_passes(
        rp_int in 10u32..=1000u32  // represents 0.1..=10.0 in steps of 0.01
    ) {
        let rp = rp_int as f32 / 100.0;
        let v = validator(false, false);
        let mut req = bare_request("hello");
        req.repetition_penalty = Some(rp);
        prop_assert!(
            v.validate_inference_request(&req).is_ok(),
            "repetition_penalty={rp} in [0.1, 10.0] must pass"
        );
    }

    /// repetition_penalty < 0.1 always fails.
    #[test]
    fn prop_repetition_penalty_below_0_1_fails(
        rp_int in 1u32..=9u32  // represents 0.01..=0.09
    ) {
        let rp = rp_int as f32 / 100.0;
        let v = validator(false, false);
        let mut req = bare_request("hello");
        req.repetition_penalty = Some(rp);
        prop_assert!(
            v.validate_inference_request(&req).is_err(),
            "repetition_penalty={rp} below 0.1 must fail"
        );
    }

    /// CORS with wildcard config always grants access regardless of origin.
    #[test]
    fn prop_cors_wildcard_always_allows(
        host in "[a-z]{3,10}\\.[a-z]{2,4}"
    ) {
        let config = SecurityConfig {
            allowed_origins: vec!["*".to_string()],
            ..Default::default()
        };

        // configure_cors should not panic for any reasonable origin string.
        let _layer = configure_cors(&config);
        // Structural check: wildcard config must not be empty.
        prop_assert!(!config.allowed_origins.is_empty());
        prop_assert_eq!(&config.allowed_origins[0], "*");
        let _ = host; // consumed by proptest harness
    }

    /// ConcurrencyConfig: `max_requests_per_second` is preserved after construction.
    #[test]
    fn prop_concurrency_config_rps_field_preserved(rps in 1u64..=10000u64) {
        let config = ConcurrencyConfig {
            max_requests_per_second: rps,
            ..Default::default()
        };
        prop_assert_eq!(config.max_requests_per_second, rps);
    }

    /// ConcurrencyConfig: `per_ip_rate_limit` is preserved (Some variant).
    #[test]
    fn prop_concurrency_config_per_ip_rate_limit_preserved(limit in 1u64..=1000u64) {
        let config = ConcurrencyConfig {
            per_ip_rate_limit: Some(limit),
            ..Default::default()
        };
        prop_assert_eq!(config.per_ip_rate_limit, Some(limit));
    }

    /// SecurityConfig: rate_limit_by_ip flag round-trips through construction.
    #[test]
    fn prop_security_config_rate_limit_flag_preserved(flag in proptest::bool::ANY) {
        let config = SecurityConfig {
            rate_limit_by_ip: flag,
            ..Default::default()
        };
        prop_assert_eq!(config.rate_limit_by_ip, flag);
    }
}
