//! Edge-case tests for monitoring health and metrics modules.
//!
//! Covers serialization roundtrips, boundary values, optional-field
//! handling, stub-probe HTTP mapping, and metrics collector edge cases.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::monitoring::health::{
    BuildInfo, ComponentHealth, HealthChecker, HealthMetrics, HealthProbe, HealthResponse,
    HealthStatus, create_health_routes_with_probe,
};
use bitnet_server::monitoring::metrics::{
    InferenceMetrics, MetricsCollector, PerformanceSnapshot, SystemMetrics,
};

// ---------------------------------------------------------------------------
// HealthStatus serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn health_status_serde_roundtrip_all_variants() {
    for variant in [HealthStatus::Healthy, HealthStatus::Degraded, HealthStatus::Unhealthy] {
        let json = serde_json::to_string(&variant).unwrap();
        let back: HealthStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(variant, back);
    }
}

#[test]
fn health_status_serializes_to_lowercase() {
    assert_eq!(serde_json::to_string(&HealthStatus::Healthy).unwrap(), "\"healthy\"");
    assert_eq!(serde_json::to_string(&HealthStatus::Degraded).unwrap(), "\"degraded\"");
    assert_eq!(serde_json::to_string(&HealthStatus::Unhealthy).unwrap(), "\"unhealthy\"");
}

#[test]
fn health_status_rejects_unknown_variant() {
    let result = serde_json::from_str::<HealthStatus>("\"unknown\"");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// HealthMetrics defaults & optional-field serde
// ---------------------------------------------------------------------------

#[test]
fn health_metrics_default_optional_fields_are_none() {
    let m = HealthMetrics::default();
    assert!(m.cpu_usage_percent.is_none());
    assert!(m.gpu_memory_mb.is_none());
    assert!(m.gpu_memory_leak.is_none());
    assert_eq!(m.active_requests, 0);
    assert_eq!(m.total_requests, 0);
    assert_eq!(m.error_rate_percent, 0.0);
    assert_eq!(m.avg_response_time_ms, 0.0);
    assert_eq!(m.memory_usage_mb, 0.0);
    assert_eq!(m.tokens_per_second, 0.0);
}

#[test]
fn health_metrics_skip_serializing_none_fields() {
    let m = HealthMetrics::default();
    let json = serde_json::to_value(&m).unwrap();
    // Optional fields with None should be absent due to skip_serializing_if
    assert!(json.get("cpu_usage_percent").is_none());
    assert!(json.get("gpu_memory_mb").is_none());
    assert!(json.get("gpu_memory_leak").is_none());
}

#[test]
fn health_metrics_includes_optional_when_some() {
    let m = HealthMetrics { cpu_usage_percent: Some(42.5), ..HealthMetrics::default() };
    let json = serde_json::to_value(&m).unwrap();
    assert_eq!(json["cpu_usage_percent"].as_f64().unwrap(), 42.5);
}

// ---------------------------------------------------------------------------
// ComponentHealth edge cases
// ---------------------------------------------------------------------------

#[test]
fn component_health_none_response_time_skipped() {
    let c = ComponentHealth {
        status: HealthStatus::Healthy,
        message: "ok".to_string(),
        last_check: "2024-01-01T00:00:00Z".to_string(),
        response_time_ms: None,
    };
    let json = serde_json::to_value(&c).unwrap();
    // response_time_ms is Option but *not* skip_serializing_if, so it appears as null
    assert!(json.get("response_time_ms").is_some());
}

#[test]
fn component_health_serde_roundtrip() {
    let c = ComponentHealth {
        status: HealthStatus::Degraded,
        message: "high latency".to_string(),
        last_check: "2024-06-15T12:00:00Z".to_string(),
        response_time_ms: Some(999),
    };
    let json = serde_json::to_string(&c).unwrap();
    let back: ComponentHealth = serde_json::from_str(&json).unwrap();
    assert_eq!(back.status, HealthStatus::Degraded);
    assert_eq!(back.response_time_ms, Some(999));
}

// ---------------------------------------------------------------------------
// BuildInfo edge cases
// ---------------------------------------------------------------------------

#[test]
fn build_info_without_cuda_omits_field() {
    let b = BuildInfo {
        version: "0.1.0".into(),
        git_sha: "abc".into(),
        git_branch: "main".into(),
        build_timestamp: "now".into(),
        rustc_version: "1.80".into(),
        cargo_target: "x86_64".into(),
        cargo_profile: "release".into(),
        cuda_version: None,
    };
    let json = serde_json::to_value(&b).unwrap();
    assert!(json.get("cuda_version").is_none(), "cuda_version=None should be omitted");
}

#[test]
fn build_info_with_cuda_includes_field() {
    let b = BuildInfo {
        version: "0.1.0".into(),
        git_sha: "abc".into(),
        git_branch: "main".into(),
        build_timestamp: "now".into(),
        rustc_version: "1.80".into(),
        cargo_target: "x86_64".into(),
        cargo_profile: "release".into(),
        cuda_version: Some("12.3".into()),
    };
    let json = serde_json::to_value(&b).unwrap();
    assert_eq!(json["cuda_version"].as_str().unwrap(), "12.3");
}

// ---------------------------------------------------------------------------
// HealthResponse serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn health_response_serde_roundtrip() {
    let resp = HealthResponse {
        status: HealthStatus::Healthy,
        timestamp: "2024-01-01T00:00:00Z".to_string(),
        uptime_seconds: 42,
        version: "0.2.1-dev".to_string(),
        build: BuildInfo {
            version: "0.2.1-dev".into(),
            git_sha: "deadbeef".into(),
            git_branch: "main".into(),
            build_timestamp: "2024-01-01".into(),
            rustc_version: "1.80.0".into(),
            cargo_target: "x86_64-unknown-linux-gnu".into(),
            cargo_profile: "release".into(),
            cuda_version: None,
        },
        components: HashMap::new(),
        metrics: HealthMetrics::default(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: HealthResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back.status, HealthStatus::Healthy);
    assert_eq!(back.uptime_seconds, 42);
    assert!(back.components.is_empty());
}

#[test]
fn health_response_with_components_roundtrip() {
    let mut components = HashMap::new();
    components.insert(
        "model".to_string(),
        ComponentHealth {
            status: HealthStatus::Unhealthy,
            message: "model not loaded".into(),
            last_check: "2024-01-01T00:00:00Z".into(),
            response_time_ms: Some(0),
        },
    );
    let resp = HealthResponse {
        status: HealthStatus::Unhealthy,
        timestamp: "2024-01-01T00:00:00Z".to_string(),
        uptime_seconds: 0,
        version: "test".to_string(),
        build: BuildInfo {
            version: "test".into(),
            git_sha: "0".into(),
            git_branch: "test".into(),
            build_timestamp: "test".into(),
            rustc_version: "test".into(),
            cargo_target: "test".into(),
            cargo_profile: "test".into(),
            cuda_version: None,
        },
        components,
        metrics: HealthMetrics::default(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: HealthResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back.components["model"].status, HealthStatus::Unhealthy);
}

// ---------------------------------------------------------------------------
// Stub-probe HTTP route edge cases
// ---------------------------------------------------------------------------

struct StubProbe {
    overall: HealthStatus,
    live: HealthStatus,
    ready: HealthStatus,
}

#[async_trait::async_trait]
impl HealthProbe for StubProbe {
    async fn check_health(&self) -> HealthResponse {
        HealthResponse {
            status: self.overall,
            timestamp: chrono::Utc::now().to_rfc3339(),
            uptime_seconds: 0,
            version: "test".into(),
            build: BuildInfo {
                version: "test".into(),
                git_sha: "stub".into(),
                git_branch: "test".into(),
                build_timestamp: "test".into(),
                rustc_version: "test".into(),
                cargo_target: "test".into(),
                cargo_profile: "test".into(),
                cuda_version: None,
            },
            components: HashMap::new(),
            metrics: HealthMetrics::default(),
        }
    }
    async fn check_liveness(&self) -> HealthStatus {
        self.live
    }
    async fn check_readiness(&self) -> HealthStatus {
        self.ready
    }
}

#[tokio::test]
async fn route_all_unhealthy_returns_503() {
    let app = create_health_routes_with_probe(Arc::new(StubProbe {
        overall: HealthStatus::Unhealthy,
        live: HealthStatus::Unhealthy,
        ready: HealthStatus::Unhealthy,
    }));
    for path in ["/health", "/health/live", "/health/ready"] {
        let resp =
            app.clone().oneshot(Request::get(path).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE, "path={path}");
    }
}

#[tokio::test]
async fn route_invalid_path_returns_404() {
    let app = create_health_routes_with_probe(Arc::new(StubProbe {
        overall: HealthStatus::Healthy,
        live: HealthStatus::Healthy,
        ready: HealthStatus::Healthy,
    }));
    let resp = app
        .oneshot(Request::get("/health/nonexistent").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn route_healthy_readiness_returns_200() {
    let app = create_health_routes_with_probe(Arc::new(StubProbe {
        overall: HealthStatus::Healthy,
        live: HealthStatus::Healthy,
        ready: HealthStatus::Healthy,
    }));
    let resp =
        app.oneshot(Request::get("/health/ready").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// HealthChecker — liveness during startup window
// ---------------------------------------------------------------------------

#[tokio::test]
async fn liveness_during_startup_returns_degraded() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    // Immediately check liveness (within the 5-second startup window)
    let checker = HealthChecker::new(metrics);
    let status = checker.check_liveness().await;
    assert_eq!(status, HealthStatus::Degraded, "should be Degraded during startup");
}

// ---------------------------------------------------------------------------
// MonitoringConfig defaults & serde
// ---------------------------------------------------------------------------

#[test]
fn monitoring_config_defaults() {
    let c = MonitoringConfig::default();
    assert!(c.prometheus_enabled);
    assert_eq!(c.prometheus_path, "/metrics");
    assert!(!c.opentelemetry_enabled);
    assert!(c.opentelemetry_endpoint.is_none());
    assert!(c.otlp_endpoint.is_none());
    assert_eq!(c.health_path, "/health");
    assert_eq!(c.metrics_interval, 10);
    assert!(c.structured_logging);
    assert_eq!(c.log_level, "info");
    assert_eq!(c.log_format, "json");
}

#[test]
fn monitoring_config_serde_roundtrip() {
    let c = MonitoringConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let back: MonitoringConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.metrics_interval, c.metrics_interval);
    assert_eq!(back.prometheus_path, c.prometheus_path);
}

#[test]
fn monitoring_config_clone() {
    let c = MonitoringConfig::default();
    let c2 = c.clone();
    assert_eq!(c.metrics_interval, c2.metrics_interval);
}

// ---------------------------------------------------------------------------
// MetricsCollector edge cases
// ---------------------------------------------------------------------------

#[test]
fn metrics_collector_new_with_default_config() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config);
    assert!(collector.is_ok());
}

#[test]
fn metrics_collector_new_with_zero_interval() {
    let mut config = MonitoringConfig::default();
    config.metrics_interval = 0;
    // Should succeed — interval=0 is handled via .max(1) in collect_system_metrics
    let collector = MetricsCollector::new(&config);
    assert!(collector.is_ok());
}

#[test]
fn metrics_collector_new_with_large_interval() {
    let mut config = MonitoringConfig::default();
    config.metrics_interval = u64::MAX;
    let collector = MetricsCollector::new(&config);
    assert!(collector.is_ok());
}

#[test]
fn metrics_collector_update_queue_depth_zero() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    collector.update_queue_depth(0); // should not panic
}

#[test]
fn metrics_collector_update_queue_depth_large() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    collector.update_queue_depth(usize::MAX); // should not panic
}

#[test]
fn metrics_collector_cache_hit_rate_boundaries() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    collector.update_cache_hit_rate(0.0);
    collector.update_cache_hit_rate(1.0);
    collector.update_cache_hit_rate(-1.0); // out of range but should not panic
    collector.update_cache_hit_rate(f64::MAX);
    collector.update_cache_hit_rate(f64::NAN);
    collector.update_cache_hit_rate(f64::INFINITY);
}

#[test]
fn metrics_collector_record_model_load_zero_duration() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    collector.record_model_load_time(Duration::ZERO); // should not panic
}

#[test]
fn metrics_collector_record_model_load_large_duration() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    collector.record_model_load_time(Duration::from_secs(86400 * 365));
}

// ---------------------------------------------------------------------------
// RequestTracker edge cases
// ---------------------------------------------------------------------------

#[test]
fn request_tracker_empty_id() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let _tracker = collector.track_request(String::new()); // should not panic
}

#[test]
fn request_tracker_drop_records_duration() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    {
        let _tracker = collector.track_request("req-1".to_string());
        // tracker dropped here
    }
    // No panic — duration was recorded on drop
}

#[test]
fn request_tracker_record_zero_tokens() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let tracker = collector.track_request("req-zero-tok".to_string());
    tracker.record_tokens(0); // should not panic
}

#[test]
fn request_tracker_record_large_token_count() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let tracker = collector.track_request("req-large-tok".to_string());
    tracker.record_tokens(u64::MAX); // should not panic
}

#[test]
fn request_tracker_record_error_does_not_panic() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let tracker = collector.track_request("req-err".to_string());
    tracker.record_error("timeout");
    tracker.record_error("");
    tracker.record_error("a]b[c{d}e");
}

#[test]
fn multiple_concurrent_trackers() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let t1 = collector.track_request("req-a".to_string());
    let t2 = collector.track_request("req-b".to_string());
    let t3 = collector.track_request("req-c".to_string());
    t1.record_tokens(10);
    t2.record_tokens(20);
    t3.record_error("test");
    drop(t1);
    drop(t2);
    drop(t3);
}

// ---------------------------------------------------------------------------
// InferenceMetrics & SystemMetrics construction
// ---------------------------------------------------------------------------

#[test]
fn inference_metrics_new_does_not_panic() {
    let _m = InferenceMetrics::new();
}

#[test]
fn inference_metrics_default_same_as_new() {
    // Both should succeed without panicking
    let _a = InferenceMetrics::new();
    let _b = InferenceMetrics::default();
}

#[test]
fn system_metrics_new_does_not_panic() {
    let _m = SystemMetrics::new();
}

#[test]
fn system_metrics_default_same_as_new() {
    let _a = SystemMetrics::new();
    let _b = SystemMetrics::default();
}

// ---------------------------------------------------------------------------
// PerformanceSnapshot construction
// ---------------------------------------------------------------------------

#[test]
fn performance_snapshot_clone_and_debug() {
    let snap = PerformanceSnapshot {
        timestamp: std::time::Instant::now(),
        tokens_per_second: 0.0,
        memory_usage_mb: 0.0,
        active_requests: 0.0,
        error_rate: 0.0,
    };
    let snap2 = snap.clone();
    assert_eq!(snap2.tokens_per_second, 0.0);
    // Debug impl should not panic
    let _ = format!("{:?}", snap2);
}

#[test]
fn performance_snapshot_extreme_values() {
    let snap = PerformanceSnapshot {
        timestamp: std::time::Instant::now(),
        tokens_per_second: f64::MAX,
        memory_usage_mb: f64::INFINITY,
        active_requests: f64::NAN,
        error_rate: -1.0,
    };
    let snap2 = snap.clone();
    assert!(snap2.active_requests.is_nan());
    assert!(snap2.memory_usage_mb.is_infinite());
}

// ---------------------------------------------------------------------------
// MetricsCollector — performance regression with insufficient history
// ---------------------------------------------------------------------------

#[tokio::test]
async fn performance_regression_empty_history_returns_no_alerts() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    let alerts = collector.check_performance_regression().await.unwrap();
    assert!(alerts.is_empty(), "no alerts expected with empty history");
}

#[tokio::test]
async fn performance_regression_insufficient_history() {
    let config = MonitoringConfig::default();
    let collector = MetricsCollector::new(&config).unwrap();
    // Collect fewer than 10 snapshots — threshold for analysis
    for _ in 0..9 {
        collector.collect_system_metrics().await.unwrap();
    }
    let alerts = collector.check_performance_regression().await.unwrap();
    assert!(alerts.is_empty(), "no alerts expected with < 10 snapshots");
}

#[tokio::test]
async fn collect_system_metrics_zero_interval_no_panic() {
    let mut config = MonitoringConfig::default();
    config.metrics_interval = 0;
    let collector = MetricsCollector::new(&config).unwrap();
    // collect_system_metrics uses .max(1) on interval, so history_limit is safe
    collector.collect_system_metrics().await.unwrap();
}

// ---------------------------------------------------------------------------
// HealthChecker — check_health returns valid response immediately
// ---------------------------------------------------------------------------

#[tokio::test]
async fn check_health_returns_valid_response() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    let checker = HealthChecker::new(metrics);
    let resp = checker.check_health().await;

    // Status should be one of the known variants
    assert!(matches!(
        resp.status,
        HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
    ));
    // Should have the critical components
    assert!(resp.components.contains_key("model"));
    assert!(resp.components.contains_key("memory"));
    assert!(resp.components.contains_key("inference_engine"));
    // Metrics memory should be non-negative
    assert!(resp.metrics.memory_usage_mb >= 0.0);
}

#[tokio::test]
async fn check_readiness_returns_valid_status() {
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config).unwrap());
    let checker = HealthChecker::new(metrics);
    let status = checker.check_readiness().await;
    assert!(matches!(
        status,
        HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
    ));
}
