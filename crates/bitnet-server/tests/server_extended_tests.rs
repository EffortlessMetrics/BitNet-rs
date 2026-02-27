//! Extended integration tests for `bitnet-server`.
//!
//! Coverage (all NEW — not duplicated from server_security_tests.rs,
//! property_tests.rs, or health_endpoints_integration.rs):
//!
//! - BatchEngine: new, get_stats (initial zeros), get_health (initially healthy)
//! - BatchRequest: unique IDs across instances, priority assignment
//! - ConcurrencyManager: initial stats/health, request admission, global rate-limit rejection
//! - InferenceResponse / ErrorResponse: JSON serialization and field presence
//! - EnhancedInferenceResponse: flattened base fields accessible in JSON
//! - /health/live: HTTP 200 status, Content-Type header, JSON "status" field

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::Value as Json;
use tower::ServiceExt;

use bitnet_inference::GenerationConfig;
use bitnet_server::{
    EnhancedInferenceResponse, ErrorResponse, InferenceResponse,
    batch_engine::{BatchEngine, BatchEngineConfig, BatchRequest, RequestPriority},
    concurrency::{ConcurrencyConfig, ConcurrencyManager, RequestAdmission, RequestMetadata},
    monitoring::{
        MonitoringConfig,
        health::{HealthChecker, create_health_routes},
        metrics::MetricsCollector,
    },
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_metadata(ip: &str) -> RequestMetadata {
    RequestMetadata {
        id: uuid::Uuid::new_v4().to_string(),
        client_ip: ip.parse::<IpAddr>().unwrap(),
        user_agent: None,
        start_time: Instant::now(),
        priority: RequestPriority::Normal,
    }
}

// ── BatchEngine ───────────────────────────────────────────────────────────────

/// BatchEngine::new with default config should not panic.
#[tokio::test]
async fn test_batch_engine_new_creates_instance() {
    let _engine = BatchEngine::new(BatchEngineConfig::default());
}

/// A freshly-created BatchEngine has queue_depth = 0.
#[tokio::test]
async fn test_batch_engine_initial_stats_zero_queue_depth() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    let stats = engine.get_stats().await;
    assert_eq!(stats.queue_depth, 0, "fresh engine must have empty queue");
}

/// A freshly-created BatchEngine has zero processed requests and batches.
#[tokio::test]
async fn test_batch_engine_initial_stats_zero_processed() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    let stats = engine.get_stats().await;
    assert_eq!(stats.total_requests_processed, 0);
    assert_eq!(stats.total_batches_processed, 0);
}

/// A freshly-created BatchEngine reports healthy with no issues.
#[tokio::test]
async fn test_batch_engine_initial_health_healthy() {
    let engine = BatchEngine::new(BatchEngineConfig::default());
    let health = engine.get_health().await;
    assert!(health.healthy, "fresh engine must report healthy");
    assert!(health.issues.is_empty(), "fresh engine must have no issues: {:?}", health.issues);
}

// ── BatchRequest ──────────────────────────────────────────────────────────────

/// Two independently-created BatchRequests must have distinct IDs.
#[test]
fn test_batch_request_ids_are_unique() {
    let r1 = BatchRequest::new("hello".to_string(), GenerationConfig::default());
    let r2 = BatchRequest::new("hello".to_string(), GenerationConfig::default());
    assert_ne!(r1.id, r2.id, "batch request IDs must be unique per request");
}

/// with_priority stores the Critical priority level correctly.
#[test]
fn test_batch_request_priority_critical() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default())
        .with_priority(RequestPriority::Critical);
    assert_eq!(req.priority, RequestPriority::Critical);
}

/// A newly-created BatchRequest has Normal priority by default.
#[test]
fn test_batch_request_default_priority_is_normal() {
    let req = BatchRequest::new("test".to_string(), GenerationConfig::default());
    assert_eq!(req.priority, RequestPriority::Normal, "default priority must be Normal");
}

// ── ConcurrencyManager ────────────────────────────────────────────────────────

/// ConcurrencyManager::new with default config should not panic.
#[tokio::test]
async fn test_concurrency_manager_new_with_config() {
    let _mgr = ConcurrencyManager::new(ConcurrencyConfig::default());
}

/// A fresh ConcurrencyManager has zero active requests and zero total requests.
#[tokio::test]
async fn test_concurrency_manager_initial_stats_zeros() {
    let mgr = ConcurrencyManager::new(ConcurrencyConfig::default());
    let stats = mgr.get_stats().await;
    assert_eq!(stats.active_requests, 0, "no active requests on a fresh manager");
    assert_eq!(stats.total_requests, 0, "no total requests on a fresh manager");
    assert_eq!(stats.rejected_requests, 0, "no rejections on a fresh manager");
}

/// A fresh ConcurrencyManager is healthy and not overloaded.
#[tokio::test]
async fn test_concurrency_manager_initial_health_not_overloaded() {
    let mgr = ConcurrencyManager::new(ConcurrencyConfig::default());
    let health = mgr.get_health().await;
    assert!(health.healthy, "fresh manager must report healthy");
    // current_load should be 0.0 with no active requests
    assert_eq!(health.current_load, 0.0, "load must be 0.0 when no requests active");
}

/// The first request to a manager with plenty of capacity must be Admitted.
#[tokio::test]
async fn test_concurrency_manager_admits_first_request() {
    let config = ConcurrencyConfig {
        global_rate_limit: Some(1000),
        per_ip_rate_limit: None,
        circuit_breaker_enabled: false,
        ..ConcurrencyConfig::default()
    };
    let mgr = ConcurrencyManager::new(config);
    let meta = make_metadata("127.0.0.1");

    let decision = mgr.should_admit_request(&meta).await.unwrap();
    assert!(
        matches!(decision, RequestAdmission::Admitted),
        "first request must be Admitted when capacity is available"
    );
}

/// After exhausting a global rate limit of 1, subsequent requests are Rejected.
#[tokio::test]
async fn test_concurrency_manager_rejects_after_rate_limit_exhausted() {
    let config = ConcurrencyConfig {
        global_rate_limit: Some(1), // only 1 token
        per_ip_rate_limit: None,
        circuit_breaker_enabled: false,
        ..ConcurrencyConfig::default()
    };
    let mgr = ConcurrencyManager::new(config);
    let meta = make_metadata("10.0.0.1");

    // First request consumes the single token.
    let first = mgr.should_admit_request(&meta).await.unwrap();
    assert!(matches!(first, RequestAdmission::Admitted), "first must be admitted");

    // Second request finds no tokens remaining.
    let second = mgr.should_admit_request(&meta).await.unwrap();
    assert!(
        matches!(second, RequestAdmission::Rejected { .. }),
        "second request must be Rejected once the global rate limit token is consumed"
    );
}

// ── JSON serialization ────────────────────────────────────────────────────────

/// InferenceResponse serializes to JSON with a "text" field.
#[test]
fn test_inference_response_json_has_text_field() {
    let resp = InferenceResponse {
        text: "Hello, world!".to_string(),
        tokens_generated: 3,
        inference_time_ms: 50,
        tokens_per_second: 60.0,
    };
    let json: Json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["text"], "Hello, world!");
}

/// InferenceResponse serializes tokens_per_second as a JSON number.
#[test]
fn test_inference_response_tokens_per_second_in_json() {
    let resp = InferenceResponse {
        text: String::new(),
        tokens_generated: 10,
        inference_time_ms: 200,
        tokens_per_second: 50.0,
    };
    let json: Json = serde_json::to_value(&resp).unwrap();
    // JSON numbers from f64 may not compare exactly; check it's numeric and close.
    let tps = json["tokens_per_second"].as_f64().expect("tokens_per_second must be a number");
    assert!((tps - 50.0).abs() < 1e-6);
}

/// ErrorResponse serializes with the required "error" and "error_code" fields.
#[test]
fn test_error_response_json_has_required_fields() {
    let resp = ErrorResponse {
        error: "model not found".to_string(),
        error_code: "ERR_MODEL_404".to_string(),
        request_id: None,
        details: None,
    };
    let json: Json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["error"], "model not found");
    assert_eq!(json["error_code"], "ERR_MODEL_404");
}

/// Optional fields in ErrorResponse serialize as null when None.
#[test]
fn test_error_response_none_fields_serialize_as_null() {
    let resp = ErrorResponse {
        error: "oops".to_string(),
        error_code: "E_OOPS".to_string(),
        request_id: None,
        details: None,
    };
    let json: Json = serde_json::to_value(&resp).unwrap();
    assert!(json["request_id"].is_null(), "None request_id must serialize as JSON null");
    assert!(json["details"].is_null(), "None details must serialize as JSON null");
}

/// EnhancedInferenceResponse: the flattened "text" field from InferenceResponse appears in JSON.
#[test]
fn test_enhanced_response_flattened_text_field() {
    let resp = EnhancedInferenceResponse {
        base: InferenceResponse {
            text: "42".to_string(),
            tokens_generated: 1,
            inference_time_ms: 10,
            tokens_per_second: 100.0,
        },
        device_used: "cpu".to_string(),
        quantization_type: "f16".to_string(),
        batch_id: None,
        batch_size: None,
        queue_time_ms: 0,
    };
    let json: Json = serde_json::to_value(&resp).unwrap();
    // Flattened base fields should appear at the top level.
    assert_eq!(json["text"], "42", "flattened 'text' from base must appear in JSON");
    assert_eq!(json["device_used"], "cpu");
}

// ── /health/live endpoint ─────────────────────────────────────────────────────

/// /health/live returns a valid HTTP status code (200 when healthy, 503 when degraded).
#[tokio::test]
async fn test_health_live_returns_valid_status() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    let checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(checker);

    let resp = app
        .oneshot(Request::builder().uri("/health/live").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::SERVICE_UNAVAILABLE,
        "/health/live must return 200 (healthy) or 503 (degraded/unhealthy), got {status}"
    );
}

/// /health/live response includes a Content-Type that contains "application/json".
#[tokio::test]
async fn test_health_live_content_type_is_json() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    let checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(checker);

    let resp = app
        .oneshot(Request::builder().uri("/health/live").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let ct = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    assert!(
        ct.contains("application/json"),
        "/health/live Content-Type must include 'application/json', got: {ct}"
    );
}

/// /health/live response body is valid JSON with a "status" field.
#[tokio::test]
async fn test_health_live_json_has_status_field() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    let checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(checker);

    let resp = app
        .oneshot(Request::builder().uri("/health/live").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Json = serde_json::from_slice(&body).expect("/health/live body must be valid JSON");
    assert!(json.get("status").is_some(), "liveness response must have a 'status' field");
}
