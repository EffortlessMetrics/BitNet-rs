//! Wave 31 snapshot tests for bitnet-server.
//!
//! Covers: API response format for common operations, error response shapes,
//! health check response format, config serialization.

use bitnet_server::config::ServerConfig;
use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::monitoring::health::{
    BuildInfo, ComponentHealth, HealthMetrics, HealthResponse, HealthStatus,
};
use bitnet_server::sse::SseConfig;
use bitnet_server::{ErrorResponse, InferenceResponse, ModelLoadResponse};
use std::collections::HashMap;

// ── API response format for common operations ───────────────────────────────

#[test]
fn inference_response_minimal_json() {
    let r = InferenceResponse {
        text: "4".into(),
        tokens_generated: 1,
        inference_time_ms: 42,
        tokens_per_second: 23.8,
    };
    insta::assert_json_snapshot!(r);
}

#[test]
fn inference_response_multitoken_json() {
    let r = InferenceResponse {
        text: "The capital of France is Paris.".into(),
        tokens_generated: 8,
        inference_time_ms: 320,
        tokens_per_second: 25.0,
    };
    insta::assert_json_snapshot!(r);
}

#[test]
fn model_load_response_json() {
    let r = ModelLoadResponse {
        model_id: "bitnet-2b-i2s".into(),
        status: "loaded".into(),
        message: "Model loaded successfully on cpu".into(),
    };
    insta::assert_json_snapshot!(r);
}

// ── Error response shapes ───────────────────────────────────────────────────

#[test]
fn error_response_with_details_json() {
    let e = ErrorResponse {
        error: "Invalid request parameters".into(),
        error_code: "INVALID_REQUEST".into(),
        request_id: Some("req-abc-456".into()),
        details: Some(serde_json::json!({
            "field": "max_tokens",
            "reason": "must be between 1 and 4096"
        })),
    };
    insta::assert_json_snapshot!(e);
}

#[test]
fn error_response_minimal_json() {
    let e = ErrorResponse {
        error: "Internal server error".into(),
        error_code: "INTERNAL_ERROR".into(),
        request_id: None,
        details: None,
    };
    insta::assert_json_snapshot!(e);
}

#[test]
fn error_response_model_not_loaded_json() {
    let e = ErrorResponse {
        error: "Model not loaded".into(),
        error_code: "MODEL_NOT_LOADED".into(),
        request_id: Some("req-789".into()),
        details: Some(serde_json::json!({
            "available_models": [],
            "hint": "POST /models/load to load a model first"
        })),
    };
    insta::assert_json_snapshot!(e);
}

// ── Health check response format ────────────────────────────────────────────

#[test]
fn health_status_all_variants_debug() {
    let variants: Vec<String> =
        [HealthStatus::Healthy, HealthStatus::Degraded, HealthStatus::Unhealthy]
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn health_response_healthy_json() {
    let r = HealthResponse {
        status: HealthStatus::Healthy,
        timestamp: "2025-01-15T00:00:00Z".into(),
        uptime_seconds: 3600,
        version: "0.2.1-dev".into(),
        build: BuildInfo {
            version: "0.2.1-dev".into(),
            git_sha: "abc1234".into(),
            git_branch: "main".into(),
            build_timestamp: "2025-01-15T00:00:00Z".into(),
            rustc_version: "1.92.0".into(),
            cargo_target: "x86_64-unknown-linux-gnu".into(),
            cargo_profile: "release".into(),
            cuda_version: None,
        },
        components: {
            let mut m = HashMap::new();
            m.insert(
                "inference_engine".into(),
                ComponentHealth {
                    status: HealthStatus::Healthy,
                    message: "Ready".into(),
                    last_check: "2025-01-15T00:00:00Z".into(),
                    response_time_ms: Some(2),
                },
            );
            m
        },
        metrics: HealthMetrics {
            active_requests: 0,
            total_requests: 1000,
            error_rate_percent: 0.1,
            avg_response_time_ms: 45.0,
            memory_usage_mb: 512.0,
            tokens_per_second: 20.0,
            cpu_usage_percent: Some(35.0),
            gpu_memory_mb: None,
            gpu_memory_leak: None,
        },
    };
    insta::assert_json_snapshot!(r);
}

#[test]
fn health_metrics_default_json() {
    let m = HealthMetrics::default();
    insta::assert_json_snapshot!(m);
}

#[test]
fn component_health_degraded_json() {
    let c = ComponentHealth {
        status: HealthStatus::Degraded,
        message: "High memory pressure".into(),
        last_check: "2025-01-15T01:00:00Z".into(),
        response_time_ms: Some(150),
    };
    insta::assert_json_snapshot!(c);
}

// ── Config serialization ────────────────────────────────────────────────────

#[test]
fn monitoring_config_default_json() {
    let cfg = MonitoringConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn monitoring_config_default_debug() {
    let cfg = MonitoringConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn server_config_settings_debug() {
    let cfg = ServerConfig::default();
    insta::assert_debug_snapshot!(cfg.server);
}

#[test]
fn sse_config_default_snapshot() {
    let cfg = SseConfig::default();
    insta::assert_snapshot!(format!(
        "retry_ms={} keep_alive_secs={}",
        cfg.retry_ms, cfg.keep_alive_secs
    ));
}

#[test]
fn server_config_default_monitoring_path() {
    let cfg = ServerConfig::default();
    insta::assert_snapshot!(format!(
        "health_path={} prometheus_path={} metrics_interval={}",
        cfg.monitoring.health_path, cfg.monitoring.prometheus_path, cfg.monitoring.metrics_interval
    ));
}
