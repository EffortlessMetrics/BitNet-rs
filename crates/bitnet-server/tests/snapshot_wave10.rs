//! Wave 10 snapshot tests for bitnet-server.
//!
//! Covers: ServerConfig, health response formats, error response formats,
//! monitoring types, security config, execution router config, model manager config.

use bitnet_server::batch_engine::{BatchEngineConfig, RequestPriority};
use bitnet_server::concurrency::{CircuitBreakerState, ConcurrencyConfig};
use bitnet_server::config::{DeviceConfig, ServerConfig, ServerSettings};
use bitnet_server::execution_router::{DeviceSelectionStrategy, ExecutionRouterConfig};
use bitnet_server::model_manager::ModelManagerConfig;
use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::monitoring::ac05_types::{
    Ac05HealthResponse, LivenessResponse, PerformanceIndicators, ReadinessChecks,
    ReadinessResponse, SystemMetrics,
};
use bitnet_server::security::SecurityConfig;

// -- ServerConfig full default -----------------------------------------------

#[test]
fn server_config_default_json() {
    let cfg = ServerConfig::default();
    insta::assert_json_snapshot!(cfg);
}

// -- SecurityConfig defaults -------------------------------------------------

#[test]
fn security_config_default_debug() {
    let cfg = SecurityConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn security_config_default_json() {
    let cfg = SecurityConfig::default();
    insta::assert_json_snapshot!(cfg);
}

// -- MonitoringConfig defaults -----------------------------------------------

#[test]
fn monitoring_config_default_debug() {
    let cfg = MonitoringConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// -- ModelManagerConfig defaults ---------------------------------------------

#[test]
fn model_manager_config_default_debug() {
    let cfg = ModelManagerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn model_manager_config_default_snapshot() {
    let cfg = ModelManagerConfig::default();
    insta::assert_snapshot!(format!(
        "max_concurrent_loads={} model_cache_size={} load_timeout_secs={} validation={} memory_limit_gb={:?}",
        cfg.max_concurrent_loads,
        cfg.model_cache_size,
        cfg.load_timeout.as_secs(),
        cfg.validation_enabled,
        cfg.memory_limit_gb
    ));
}

// -- ExecutionRouterConfig defaults ------------------------------------------

#[test]
fn execution_router_config_default_debug() {
    let cfg = ExecutionRouterConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn device_selection_strategy_variants_debug() {
    let strategies = [
        format!("{:?}", DeviceSelectionStrategy::PreferGpu),
        format!("{:?}", DeviceSelectionStrategy::CpuOnly),
        format!("{:?}", DeviceSelectionStrategy::PerformanceBased),
        format!("{:?}", DeviceSelectionStrategy::LoadBalance),
    ];
    insta::assert_debug_snapshot!(strategies);
}

// -- BatchEngineConfig defaults ----------------------------------------------

#[test]
fn batch_engine_config_default_debug() {
    let cfg = BatchEngineConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn batch_engine_config_default_json() {
    let cfg = BatchEngineConfig::default();
    insta::assert_json_snapshot!(cfg);
}

// -- ConcurrencyConfig defaults ----------------------------------------------

#[test]
fn concurrency_config_default_debug() {
    let cfg = ConcurrencyConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn concurrency_config_default_json() {
    let cfg = ConcurrencyConfig::default();
    insta::assert_json_snapshot!(cfg);
}

// -- CircuitBreakerState variants --------------------------------------------

#[test]
fn circuit_breaker_state_variants_debug() {
    let states = [
        format!("{:?}", CircuitBreakerState::Closed),
        format!("{:?}", CircuitBreakerState::Open),
        format!("{:?}", CircuitBreakerState::HalfOpen),
    ];
    insta::assert_debug_snapshot!(states);
}

// -- RequestPriority variants ------------------------------------------------

#[test]
fn request_priority_variants_debug() {
    let priorities = [
        format!("{:?}", RequestPriority::Low),
        format!("{:?}", RequestPriority::Normal),
        format!("{:?}", RequestPriority::High),
        format!("{:?}", RequestPriority::Critical),
    ];
    insta::assert_debug_snapshot!(priorities);
}

// -- DeviceConfig parsing ----------------------------------------------------

#[test]
fn device_config_parse_auto() {
    let cfg: DeviceConfig = "auto".parse().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn device_config_parse_cpu() {
    let cfg: DeviceConfig = "cpu".parse().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn device_config_parse_gpu() {
    let cfg: DeviceConfig = "gpu".parse().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn device_config_parse_gpu_with_id() {
    let cfg: DeviceConfig = "gpu:1".parse().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn device_config_parse_unknown_error() {
    let err = "quantum".parse::<DeviceConfig>().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// -- Health response formats (AC05 types) ------------------------------------

#[test]
fn liveness_response_json() {
    let resp = LivenessResponse {
        status: "healthy".to_string(),
        timestamp: "2024-01-15T12:00:00Z".to_string(),
    };
    insta::assert_json_snapshot!(resp);
}

#[test]
fn readiness_response_all_ready_json() {
    let resp = ReadinessResponse {
        status: "ready".to_string(),
        timestamp: "2024-01-15T12:00:00Z".to_string(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    insta::assert_json_snapshot!(resp);
}

#[test]
fn readiness_response_not_ready_json() {
    let resp = ReadinessResponse {
        status: "not_ready".to_string(),
        timestamp: "2024-01-15T12:00:00Z".to_string(),
        checks: ReadinessChecks {
            model_loaded: false,
            inference_engine_ready: false,
            device_available: true,
            resources_available: true,
        },
    };
    insta::assert_json_snapshot!(resp);
}

#[test]
fn system_metrics_default_json() {
    let metrics = SystemMetrics::default();
    insta::assert_json_snapshot!(metrics);
}

#[test]
fn performance_indicators_default_json() {
    let indicators = PerformanceIndicators::default();
    insta::assert_json_snapshot!(indicators);
}

#[test]
fn ac05_health_response_healthy_json() {
    let resp = Ac05HealthResponse {
        status: "healthy".to_string(),
        timestamp: "2024-01-15T12:00:00Z".to_string(),
        components: [
            ("model_manager".to_string(), "healthy".to_string()),
            ("execution_router".to_string(), "healthy".to_string()),
            ("batch_engine".to_string(), "healthy".to_string()),
        ]
        .into_iter()
        .collect(),
        system_metrics: SystemMetrics {
            cpu_utilization: 0.45,
            gpu_utilization: 0.0,
            memory_usage_bytes: 4_294_967_296,
            gpu_memory_usage_bytes: None,
            active_requests: 5,
            queue_depth: 2,
        },
        performance_indicators: PerformanceIndicators {
            avg_response_time_ms: 150.0,
            requests_per_second: 10.5,
            error_rate: 0.001,
            sla_compliance: 0.998,
        },
    };
    let mut settings = insta::Settings::clone_current();
    settings.set_sort_maps(true);
    settings.bind(|| {
        insta::assert_json_snapshot!(resp);
    });
}

#[test]
fn ac05_health_response_degraded_json() {
    let resp = Ac05HealthResponse {
        status: "degraded".to_string(),
        timestamp: "2024-01-15T12:05:00Z".to_string(),
        components: [
            ("model_manager".to_string(), "healthy".to_string()),
            ("execution_router".to_string(), "degraded".to_string()),
        ]
        .into_iter()
        .collect(),
        system_metrics: SystemMetrics {
            cpu_utilization: 0.92,
            gpu_utilization: 0.0,
            memory_usage_bytes: 7_516_192_768,
            gpu_memory_usage_bytes: None,
            active_requests: 95,
            queue_depth: 47,
        },
        performance_indicators: PerformanceIndicators {
            avg_response_time_ms: 3500.0,
            requests_per_second: 2.1,
            error_rate: 0.15,
            sla_compliance: 0.62,
        },
    };
    let mut settings = insta::Settings::clone_current();
    settings.set_sort_maps(true);
    settings.bind(|| {
        insta::assert_json_snapshot!(resp);
    });
}

// -- ServerSettings defaults -------------------------------------------------

#[test]
fn server_settings_default_debug() {
    let cfg = ServerSettings::default();
    insta::assert_debug_snapshot!(cfg);
}
