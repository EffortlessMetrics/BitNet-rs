//! Wave 33 snapshot tests for bitnet-server.
//!
//! Covers: ServerConfig defaults, health endpoint response format,
//! error response format, batch engine health, concurrency health,
//! device status, monitoring config, and JSON wire formats.

use bitnet_server::batch_engine::BatchEngineConfig;
use bitnet_server::concurrency::{CircuitBreakerState, ConcurrencyConfig};
use bitnet_server::config::{DeviceConfig, ServerConfig, ServerSettings};
use bitnet_server::execution_router::{
    DeviceHealth, DeviceSelectionStrategy, ExecutionRouterConfig,
};
use bitnet_server::model_manager::ModelManagerConfig;
use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::security::SecurityConfig;
use bitnet_server::{ErrorResponse, InferenceResponse};

// ── ServerConfig defaults ───────────────────────────────────────────────────

#[test]
fn w33_server_config_default_full_json() {
    let cfg = ServerConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn w33_server_settings_default_debug() {
    let settings = ServerSettings::default();
    insta::assert_debug_snapshot!(settings);
}

#[test]
fn w33_server_settings_host_port_display() {
    let s = ServerSettings::default();
    insta::assert_snapshot!(format!("{}:{}", s.host, s.port));
}

#[test]
fn w33_server_settings_timeouts_display() {
    let s = ServerSettings::default();
    insta::assert_snapshot!(format!(
        "keep_alive={}s request_timeout={}s graceful_shutdown={}s",
        s.keep_alive.as_secs(),
        s.request_timeout.as_secs(),
        s.graceful_shutdown_timeout.as_secs()
    ));
}

// ── Health endpoint response format ─────────────────────────────────────────

#[test]
fn w33_error_response_basic_json() {
    let resp = ErrorResponse {
        error: "Model not found".to_string(),
        error_code: "MODEL_NOT_FOUND".to_string(),
        request_id: Some("req-abc-123".to_string()),
        details: None,
    };
    let json = serde_json::to_string_pretty(&resp).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_error_response_with_details_json() {
    let resp = ErrorResponse {
        error: "Validation failed".to_string(),
        error_code: "VALIDATION_ERROR".to_string(),
        request_id: Some("req-def-456".to_string()),
        details: Some(serde_json::json!({
            "field": "prompt",
            "reason": "exceeds maximum length"
        })),
    };
    let json = serde_json::to_string_pretty(&resp).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_error_response_no_request_id_json() {
    let resp = ErrorResponse {
        error: "Internal server error".to_string(),
        error_code: "INTERNAL_ERROR".to_string(),
        request_id: None,
        details: None,
    };
    let json = serde_json::to_string_pretty(&resp).unwrap();
    insta::assert_snapshot!(json);
}

#[test]
fn w33_inference_response_json() {
    let resp = InferenceResponse {
        text: "The capital of France is Paris.".to_string(),
        tokens_generated: 7,
        inference_time_ms: 150,
        tokens_per_second: 46.67,
    };
    let json = serde_json::to_string_pretty(&resp).unwrap();
    insta::assert_snapshot!(json);
}

// ── DeviceConfig variants ───────────────────────────────────────────────────

#[test]
fn w33_device_config_all_variants_debug() {
    let variants = vec![
        format!("{:?}", DeviceConfig::Auto),
        format!("{:?}", DeviceConfig::Cpu),
        format!("{:?}", DeviceConfig::Gpu(0)),
        format!("{:?}", DeviceConfig::Gpu(1)),
    ];
    insta::assert_debug_snapshot!(variants);
}

// ── DeviceSelectionStrategy ─────────────────────────────────────────────────

#[test]
fn w33_device_selection_strategy_all_variants_debug() {
    let variants = vec![
        format!("{:?}", DeviceSelectionStrategy::PreferGpu),
        format!("{:?}", DeviceSelectionStrategy::CpuOnly),
        format!("{:?}", DeviceSelectionStrategy::PerformanceBased),
        format!("{:?}", DeviceSelectionStrategy::LoadBalance),
    ];
    insta::assert_debug_snapshot!(variants);
}

// ── DeviceHealth ────────────────────────────────────────────────────────────

#[test]
fn w33_device_health_healthy_debug() {
    insta::assert_debug_snapshot!(DeviceHealth::Healthy);
}

#[test]
fn w33_device_health_degraded_debug() {
    let h = DeviceHealth::Degraded { reason: "High latency detected".to_string() };
    insta::assert_debug_snapshot!(h);
}

#[test]
fn w33_device_health_unavailable_debug() {
    let h = DeviceHealth::Unavailable { reason: "Device not responding".to_string() };
    insta::assert_debug_snapshot!(h);
}

// ── BatchEngineConfig ───────────────────────────────────────────────────────

#[test]
fn w33_batch_engine_config_summary() {
    let c = BatchEngineConfig::default();
    insta::assert_snapshot!(format!(
        "max_batch={} timeout_ms={} concurrent={} priority={} adaptive={} quant_aware={} simd={}",
        c.max_batch_size,
        c.batch_timeout.as_millis(),
        c.max_concurrent_batches,
        c.priority_queue_enabled,
        c.adaptive_batching,
        c.quantization_aware,
        c.simd_optimization,
    ));
}

// ── ConcurrencyConfig ───────────────────────────────────────────────────────

#[test]
fn w33_concurrency_config_summary() {
    let c = ConcurrencyConfig::default();
    insta::assert_snapshot!(format!(
        "max_concurrent={} rps={} rpm={} backpressure={} cb_enabled={} per_ip={:?}",
        c.max_concurrent_requests,
        c.max_requests_per_second,
        c.max_requests_per_minute,
        c.backpressure_threshold,
        c.circuit_breaker_enabled,
        c.per_ip_rate_limit,
    ));
}

// ── CircuitBreakerState ─────────────────────────────────────────────────────

#[test]
fn w33_circuit_breaker_state_all_variants_debug() {
    let states = vec![
        format!("{:?}", CircuitBreakerState::Closed),
        format!("{:?}", CircuitBreakerState::Open),
        format!("{:?}", CircuitBreakerState::HalfOpen),
    ];
    insta::assert_debug_snapshot!(states);
}

// ── SecurityConfig ──────────────────────────────────────────────────────────

#[test]
fn w33_security_config_default_json() {
    let c = SecurityConfig::default();
    insta::assert_json_snapshot!(c);
}

#[test]
fn w33_security_config_summary() {
    let c = SecurityConfig::default();
    insta::assert_snapshot!(format!(
        "auth={} max_prompt={} max_tokens={} rate_by_ip={} sanitize={} filter={}",
        c.require_authentication,
        c.max_prompt_length,
        c.max_tokens_per_request,
        c.rate_limit_by_ip,
        c.input_sanitization,
        c.content_filtering,
    ));
}

// ── MonitoringConfig ────────────────────────────────────────────────────────

#[test]
fn w33_monitoring_config_default_debug() {
    let c = MonitoringConfig::default();
    insta::assert_debug_snapshot!(c);
}

// ── ModelManagerConfig ──────────────────────────────────────────────────────

#[test]
fn w33_model_manager_config_default_json() {
    let c = ModelManagerConfig::default();
    insta::assert_json_snapshot!(c);
}

// ── ExecutionRouterConfig ───────────────────────────────────────────────────

#[test]
fn w33_execution_router_config_default_json() {
    let c = ExecutionRouterConfig::default();
    insta::assert_json_snapshot!(c);
}
