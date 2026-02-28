//! Snapshot tests for stable bitnet-server configuration defaults.
//! These pin key server configuration constants to catch silent regressions.

use bitnet_server::{
    batch_engine::BatchEngineConfig,
    concurrency::ConcurrencyConfig,
    config::{DeviceConfig, ServerSettings},
};

#[test]
fn server_settings_default_host_and_port() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!("host={} port={}", cfg.host, cfg.port));
}

#[test]
fn server_settings_default_timeouts() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!(
        "keep_alive={}s request_timeout={}s shutdown={}s",
        cfg.keep_alive.as_secs(),
        cfg.request_timeout.as_secs(),
        cfg.graceful_shutdown_timeout.as_secs()
    ));
}

#[test]
fn server_settings_default_device_is_auto() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!("{:?}", cfg.default_device));
}

#[test]
fn batch_engine_config_default_batch_size() {
    let cfg = BatchEngineConfig::default();
    insta::assert_snapshot!(format!(
        "max_batch_size={} max_concurrent_batches={}",
        cfg.max_batch_size, cfg.max_concurrent_batches
    ));
}

#[test]
fn concurrency_config_default_max_requests() {
    let cfg = ConcurrencyConfig::default();
    insta::assert_snapshot!(format!("max_concurrent={}", cfg.max_concurrent_requests));
}

#[test]
fn device_config_variants_debug() {
    let variants = [format!("{:?}", DeviceConfig::Cpu), format!("{:?}", DeviceConfig::Auto)];
    insta::assert_snapshot!(variants.join("\n"));
}

// -- Wave 3: server response JSON snapshots ----------------------------------

#[test]
fn device_config_cpu_json() {
    let cfg = DeviceConfig::Cpu;
    insta::assert_json_snapshot!("device_config_cpu", cfg);
}

#[test]
fn device_config_auto_json() {
    let cfg = DeviceConfig::Auto;
    insta::assert_json_snapshot!("device_config_auto", cfg);
}

use bitnet_server::execution_router::{DeviceSelectionStrategy, ExecutionRouterConfig};

#[test]
fn execution_router_config_default_debug() {
    let cfg = ExecutionRouterConfig::default();
    insta::assert_debug_snapshot!("execution_router_config_default", cfg);
}

#[test]
fn device_selection_strategy_variants_debug() {
    let strategies = [
        DeviceSelectionStrategy::PreferGpu,
        DeviceSelectionStrategy::CpuOnly,
        DeviceSelectionStrategy::PerformanceBased,
        DeviceSelectionStrategy::LoadBalance,
    ];
    let debug: Vec<String> = strategies.iter().map(|s| format!("{s:?}")).collect();
    insta::assert_debug_snapshot!("device_selection_strategy_variants", debug);
}

// -- Wave 3: health monitoring JSON snapshots --------------------------------

use bitnet_server::monitoring::ac05_types::{
    LivenessResponse, PerformanceIndicators as Ac05PerformanceIndicators, ReadinessChecks,
    ReadinessResponse, SystemMetrics,
};

#[test]
fn liveness_response_json() {
    let resp = LivenessResponse {
        status: "healthy".to_string(),
        timestamp: "2025-01-01T00:00:00Z".to_string(),
    };
    insta::assert_json_snapshot!("liveness_response", resp);
}

#[test]
fn readiness_response_ready_json() {
    let resp = ReadinessResponse {
        status: "ready".to_string(),
        timestamp: "2025-01-01T00:00:00Z".to_string(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    insta::assert_json_snapshot!("readiness_response_ready", resp);
}

#[test]
fn readiness_response_not_ready_json() {
    let resp = ReadinessResponse {
        status: "not_ready".to_string(),
        timestamp: "2025-01-01T00:00:00Z".to_string(),
        checks: ReadinessChecks {
            model_loaded: false,
            inference_engine_ready: false,
            device_available: true,
            resources_available: true,
        },
    };
    insta::assert_json_snapshot!("readiness_response_not_ready", resp);
}

#[test]
fn system_metrics_default_json() {
    let metrics = SystemMetrics::default();
    insta::assert_json_snapshot!("system_metrics_default", metrics);
}

#[test]
fn ac05_performance_indicators_default_json() {
    let perf = Ac05PerformanceIndicators::default();
    insta::assert_json_snapshot!("ac05_performance_indicators_default", perf);
}
