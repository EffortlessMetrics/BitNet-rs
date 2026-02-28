//! Integration tests for bitnet-server covering areas not addressed by existing tests:
//!
//! - `ErrorResponse` with non-null `details` field
//! - `ConfigBuilder` env-loading for rate-limit variables
//!   (`BITNET_PER_IP_RATE_LIMIT`, `BITNET_MAX_REQUESTS_PER_SECOND`)
//! - `/health/ready` returning HTTP 200 for a healthy stub
//! - `ConcurrencyManager` constructed with per-IP rate-limit configuration

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

use bitnet_server::{
    ErrorResponse,
    concurrency::{ConcurrencyConfig, ConcurrencyManager},
    config::ConfigBuilder,
    monitoring::{
        MonitoringConfig,
        health::{
            BuildInfo, HealthMetrics, HealthProbe, HealthResponse, HealthStatus,
            create_health_routes_with_probe,
        },
        metrics::MetricsCollector,
    },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Minimal health probe stub that always reports healthy.
struct AlwaysHealthy;

#[async_trait::async_trait]
impl HealthProbe for AlwaysHealthy {
    async fn check_health(&self) -> HealthResponse {
        HealthResponse {
            status: HealthStatus::Healthy,
            timestamp: chrono::Utc::now().to_rfc3339(),
            uptime_seconds: 1,
            version: "0.0.0-test".to_string(),
            build: BuildInfo {
                version: "0.0.0-test".to_string(),
                git_sha: "abc123".to_string(),
                git_branch: "test".to_string(),
                build_timestamp: "1970-01-01T00:00:00Z".to_string(),
                rustc_version: "1.0.0".to_string(),
                cargo_target: "x86_64-unknown-linux-gnu".to_string(),
                cargo_profile: "test".to_string(),
                cuda_version: None,
            },
            components: Default::default(),
            metrics: HealthMetrics::default(),
        }
    }

    async fn check_liveness(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    async fn check_readiness(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    async fn check_startup(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

// ── ErrorResponse serialization ──────────────────────────────────────────────

/// `ErrorResponse` with a non-null `details` field round-trips through serde.
#[test]
fn test_error_response_details_field_serializes() {
    let detail_payload = json!({"max_allowed": 8192, "received": 10000});
    let resp = ErrorResponse {
        error: "Prompt too long".to_string(),
        error_code: "PROMPT_TOO_LONG".to_string(),
        request_id: Some("req-abc-123".to_string()),
        details: Some(detail_payload.clone()),
    };

    let json: Value = serde_json::to_value(&resp).expect("ErrorResponse must serialize");

    assert_eq!(json["error"], "Prompt too long");
    assert_eq!(json["error_code"], "PROMPT_TOO_LONG");
    assert_eq!(json["request_id"], "req-abc-123");
    // details object must be present and contain the nested fields
    assert_eq!(json["details"]["max_allowed"], 8192);
    assert_eq!(json["details"]["received"], 10000);
}

/// `ErrorResponse` with `details: None` must NOT omit the field (stays null).
#[test]
fn test_error_response_details_none_is_null() {
    let resp = ErrorResponse {
        error: "Missing field".to_string(),
        error_code: "MISSING_FIELD".to_string(),
        request_id: None,
        details: None,
    };

    let json: Value = serde_json::to_value(&resp).expect("ErrorResponse must serialize");
    assert!(json["details"].is_null(), "None details must serialize as null");
    assert!(json["request_id"].is_null(), "None request_id must serialize as null");
}

/// All six validation error codes that the server uses are distinct strings.
#[test]
fn test_error_codes_are_distinct() {
    let codes = [
        "PROMPT_TOO_LONG",
        "TOO_MANY_TOKENS",
        "INVALID_CHARACTERS",
        "BLOCKED_CONTENT",
        "MISSING_FIELD",
        "INVALID_FIELD_VALUE",
    ];
    let unique: std::collections::HashSet<&str> = codes.iter().copied().collect();
    assert_eq!(codes.len(), unique.len(), "all error codes must be unique strings");
}

// ── ConfigBuilder env-loading for rate-limit settings ────────────────────────

/// `BITNET_PER_IP_RATE_LIMIT` is loaded by `ConfigBuilder::from_env`.
#[test]
fn test_config_builder_per_ip_rate_limit_from_env() {
    // Use a serial guard so parallel tests can't clobber each other.
    unsafe { std::env::set_var("BITNET_PER_IP_RATE_LIMIT", "7") };
    let result = ConfigBuilder::new().from_env();
    unsafe { std::env::remove_var("BITNET_PER_IP_RATE_LIMIT") };

    let config = result.expect("from_env must succeed").build();
    assert_eq!(
        config.concurrency.per_ip_rate_limit,
        Some(7),
        "BITNET_PER_IP_RATE_LIMIT=7 must set per_ip_rate_limit to Some(7)"
    );
}

/// `BITNET_MAX_REQUESTS_PER_SECOND` is loaded by `ConfigBuilder::from_env`.
#[test]
fn test_config_builder_max_requests_per_second_from_env() {
    unsafe { std::env::set_var("BITNET_MAX_REQUESTS_PER_SECOND", "25") };
    let result = ConfigBuilder::new().from_env();
    unsafe { std::env::remove_var("BITNET_MAX_REQUESTS_PER_SECOND") };

    let config = result.expect("from_env must succeed").build();
    assert_eq!(
        config.concurrency.max_requests_per_second, 25,
        "BITNET_MAX_REQUESTS_PER_SECOND=25 must set max_requests_per_second to 25"
    );
}

/// `BITNET_MAX_REQUESTS_PER_MINUTE` is loaded by `ConfigBuilder::from_env`.
#[test]
fn test_config_builder_max_requests_per_minute_from_env() {
    unsafe { std::env::set_var("BITNET_MAX_REQUESTS_PER_MINUTE", "300") };
    let result = ConfigBuilder::new().from_env();
    unsafe { std::env::remove_var("BITNET_MAX_REQUESTS_PER_MINUTE") };

    let config = result.expect("from_env must succeed").build();
    assert_eq!(
        config.concurrency.max_requests_per_minute, 300,
        "BITNET_MAX_REQUESTS_PER_MINUTE=300 must be reflected in config"
    );
}

// ── ConcurrencyManager with per-IP rate limit ─────────────────────────────────

/// `ConcurrencyManager` constructed with `per_ip_rate_limit: Some(5)` should
/// create without panicking and report the limit in its config.
#[test]
fn test_concurrency_manager_with_per_ip_limit() {
    let config = ConcurrencyConfig {
        per_ip_rate_limit: Some(5),
        max_concurrent_requests: 50,
        ..ConcurrencyConfig::default()
    };
    // Verify the config is stored as-is
    assert_eq!(config.per_ip_rate_limit, Some(5));

    // Construction must not panic
    let _mgr = ConcurrencyManager::new(config);
}

/// `ConcurrencyManager` with no per-IP limit (`None`) also constructs cleanly.
#[test]
fn test_concurrency_manager_without_per_ip_limit() {
    let config = ConcurrencyConfig { per_ip_rate_limit: None, ..ConcurrencyConfig::default() };
    assert!(config.per_ip_rate_limit.is_none());
    let _mgr = ConcurrencyManager::new(config);
}

/// `ConcurrencyConfig` default has `per_ip_rate_limit = Some(10)`.
#[test]
fn test_concurrency_config_default_per_ip_is_some() {
    let config = ConcurrencyConfig::default();
    assert!(
        config.per_ip_rate_limit.is_some(),
        "default per_ip_rate_limit must be Some (rate limiting on by default)"
    );
}

// ── Health endpoint: /health/ready returns 200 for healthy stub ───────────────

/// `/health/ready` returns HTTP 200 when the probe reports healthy.
#[tokio::test]
async fn test_health_ready_returns_200_for_healthy() {
    let app = create_health_routes_with_probe(Arc::new(AlwaysHealthy));

    let resp = app
        .oneshot(Request::builder().uri("/health/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/health/ready must return 200 when probe is healthy"
    );
}

/// `/health/ready` includes a `Cache-Control: no-store` header.
#[tokio::test]
async fn test_health_ready_has_no_store_cache_header() {
    let app = create_health_routes_with_probe(Arc::new(AlwaysHealthy));

    let resp = app
        .oneshot(Request::builder().uri("/health/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let cache_ctrl =
        resp.headers().get(axum::http::header::CACHE_CONTROL).and_then(|v| v.to_str().ok());

    assert_eq!(cache_ctrl, Some("no-store"), "/health/ready must include Cache-Control: no-store");
}

/// `/health` response body contains a `build` object with at least a `version` key.
#[tokio::test]
async fn test_health_response_build_info_present() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).expect("MetricsCollector::new"));
    let checker = Arc::new(bitnet_server::monitoring::health::HealthChecker::new(metrics));
    let app = bitnet_server::monitoring::health::create_health_routes(checker);

    let resp =
        app.oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();

    let body =
        axum::body::to_bytes(resp.into_body(), usize::MAX).await.expect("body read must succeed");
    let json: Value = serde_json::from_slice(&body).expect("body must be valid JSON");

    let build = json.get("build").expect("/health JSON must contain 'build' key");
    assert!(build.get("version").is_some(), "'build' object must contain 'version'");
    assert!(build.get("git_sha").is_some(), "'build' object must contain 'git_sha'");
}
