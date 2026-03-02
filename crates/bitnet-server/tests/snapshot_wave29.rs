//! Wave-29 snapshot tests for bitnet-server API surface stability.
//!
//! Pins default server config JSON, health endpoint response format,
//! and error response format for invalid requests.

use bitnet_server::ErrorResponse;
use bitnet_server::config::ServerConfig;
use bitnet_server::monitoring::health::{
    BuildInfo, ComponentHealth, HealthMetrics, HealthResponse, HealthStatus,
};
use std::collections::HashMap;

// ── Default server config (JSON serialization) ────────────────────

#[test]
fn default_server_config_json() {
    let config = ServerConfig::default();
    let json = serde_json::to_string_pretty(&config).expect("serializable");
    insta::assert_snapshot!(json);
}

// ── Health endpoint response format ───────────────────────────────

#[test]
fn health_response_format() {
    let mut components = HashMap::new();
    components.insert(
        "inference_engine".to_string(),
        ComponentHealth {
            status: HealthStatus::Healthy,
            message: "Engine running".to_string(),
            last_check: "2025-01-01T00:00:00Z".to_string(),
            response_time_ms: Some(5),
        },
    );

    let response = HealthResponse {
        status: HealthStatus::Healthy,
        timestamp: "2025-01-01T00:00:00Z".to_string(),
        uptime_seconds: 3600,
        version: "0.2.1-dev".to_string(),
        build: BuildInfo {
            version: "0.2.1-dev".to_string(),
            git_sha: "abc1234".to_string(),
            git_branch: "main".to_string(),
            build_timestamp: "2025-01-01T00:00:00Z".to_string(),
            rustc_version: "1.92.0".to_string(),
            cargo_target: "x86_64-unknown-linux-gnu".to_string(),
            cargo_profile: "release".to_string(),
            cuda_version: None,
        },
        components,
        metrics: HealthMetrics {
            active_requests: 0,
            total_requests: 100,
            error_rate_percent: 0.5,
            avg_response_time_ms: 42.0,
            memory_usage_mb: 256.0,
            tokens_per_second: 15.3,
            cpu_usage_percent: Some(25.0),
            gpu_memory_mb: None,
            gpu_memory_leak: None,
        },
    };

    let json = serde_json::to_string_pretty(&response).expect("serializable");
    insta::assert_snapshot!(json);
}

// ── Error response format for invalid request ─────────────────────

#[test]
fn error_response_format_invalid_request() {
    let error = ErrorResponse {
        error: "Invalid model path".to_string(),
        error_code: "INVALID_REQUEST".to_string(),
        request_id: Some("req-12345".to_string()),
        details: Some(serde_json::json!({
            "field": "model_path",
            "reason": "File not found"
        })),
    };

    let json = serde_json::to_string_pretty(&error).expect("serializable");
    insta::assert_snapshot!(json);
}
