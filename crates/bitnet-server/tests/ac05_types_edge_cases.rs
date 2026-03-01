//! Edge-case tests for AC05 health check types and monitoring metrics types.
//!
//! Tests cover LivenessResponse, SystemMetrics, PerformanceIndicators,
//! ReadinessChecks, ReadinessResponse, Ac05HealthResponse — serialization,
//! deserialization, defaults, edge values, and invariants.

use bitnet_server::monitoring::ac05_types::*;
use std::collections::HashMap;

// ── LivenessResponse ─────────────────────────────────────────────

#[test]
fn liveness_response_debug() {
    let r = LivenessResponse { status: "ok".into(), timestamp: "2024-01-01T00:00:00Z".into() };
    let dbg = format!("{:?}", r);
    assert!(dbg.contains("ok"));
}

#[test]
fn liveness_response_clone() {
    let r = LivenessResponse { status: "ok".into(), timestamp: "2024-01-01T00:00:00Z".into() };
    let r2 = r.clone();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.timestamp, r2.timestamp);
}

#[test]
fn liveness_response_serde_roundtrip() {
    let r = LivenessResponse { status: "healthy".into(), timestamp: "2024-06-15T12:30:00Z".into() };
    let json = serde_json::to_string(&r).unwrap();
    let r2: LivenessResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.timestamp, r2.timestamp);
}

#[test]
fn liveness_response_empty_strings() {
    let r = LivenessResponse { status: "".into(), timestamp: "".into() };
    let json = serde_json::to_string(&r).unwrap();
    let r2: LivenessResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.status, "");
    assert_eq!(r2.timestamp, "");
}

#[test]
fn liveness_response_json_structure() {
    let r = LivenessResponse { status: "ok".into(), timestamp: "2024-01-01T00:00:00Z".into() };
    let v: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["timestamp"], "2024-01-01T00:00:00Z");
}

// ── SystemMetrics ────────────────────────────────────────────────

#[test]
fn system_metrics_default() {
    let m = SystemMetrics::default();
    assert_eq!(m.cpu_utilization, 0.0);
    assert_eq!(m.gpu_utilization, 0.0);
    assert_eq!(m.memory_usage_bytes, 0);
    assert!(m.gpu_memory_usage_bytes.is_none());
    assert_eq!(m.active_requests, 0);
    assert_eq!(m.queue_depth, 0);
}

#[test]
fn system_metrics_debug() {
    let m = SystemMetrics::default();
    let dbg = format!("{:?}", m);
    assert!(dbg.contains("SystemMetrics"));
}

#[test]
fn system_metrics_clone() {
    let m = SystemMetrics {
        cpu_utilization: 0.5,
        gpu_utilization: 0.8,
        memory_usage_bytes: 1024,
        gpu_memory_usage_bytes: Some(2048),
        active_requests: 10,
        queue_depth: 5,
    };
    let m2 = m.clone();
    assert_eq!(m.cpu_utilization, m2.cpu_utilization);
    assert_eq!(m.gpu_memory_usage_bytes, m2.gpu_memory_usage_bytes);
}

#[test]
fn system_metrics_serde_roundtrip() {
    let m = SystemMetrics {
        cpu_utilization: 99.9,
        gpu_utilization: 100.0,
        memory_usage_bytes: i64::MAX,
        gpu_memory_usage_bytes: Some(u32::MAX),
        active_requests: i32::MAX,
        queue_depth: i32::MAX,
    };
    let json = serde_json::to_string(&m).unwrap();
    let m2: SystemMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(m.cpu_utilization, m2.cpu_utilization);
    assert_eq!(m.memory_usage_bytes, m2.memory_usage_bytes);
    assert_eq!(m.gpu_memory_usage_bytes, m2.gpu_memory_usage_bytes);
}

#[test]
fn system_metrics_gpu_memory_none_skipped_in_json() {
    let m = SystemMetrics { gpu_memory_usage_bytes: None, ..Default::default() };
    let json = serde_json::to_string(&m).unwrap();
    assert!(!json.contains("gpu_memory_usage_bytes"));
}

#[test]
fn system_metrics_gpu_memory_some_included_in_json() {
    let m = SystemMetrics { gpu_memory_usage_bytes: Some(1024), ..Default::default() };
    let json = serde_json::to_string(&m).unwrap();
    assert!(json.contains("gpu_memory_usage_bytes"));
}

#[test]
fn system_metrics_negative_memory() {
    let m = SystemMetrics { memory_usage_bytes: -1, ..Default::default() };
    let json = serde_json::to_string(&m).unwrap();
    let m2: SystemMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.memory_usage_bytes, -1);
}

#[test]
fn system_metrics_zero_cpu_and_gpu() {
    let m = SystemMetrics { cpu_utilization: 0.0, gpu_utilization: 0.0, ..Default::default() };
    let json = serde_json::to_string(&m).unwrap();
    let m2: SystemMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.cpu_utilization, 0.0);
    assert_eq!(m2.gpu_utilization, 0.0);
}

// ── PerformanceIndicators ────────────────────────────────────────

#[test]
fn performance_indicators_default() {
    let p = PerformanceIndicators::default();
    assert_eq!(p.avg_response_time_ms, 0.0);
    assert_eq!(p.requests_per_second, 0.0);
    assert_eq!(p.error_rate, 0.0);
    assert_eq!(p.sla_compliance, 0.0);
}

#[test]
fn performance_indicators_debug() {
    let p = PerformanceIndicators::default();
    let dbg = format!("{:?}", p);
    assert!(dbg.contains("PerformanceIndicators"));
}

#[test]
fn performance_indicators_clone() {
    let p = PerformanceIndicators {
        avg_response_time_ms: 1.5,
        requests_per_second: 100.0,
        error_rate: 0.01,
        sla_compliance: 0.999,
    };
    let p2 = p.clone();
    assert_eq!(p.sla_compliance, p2.sla_compliance);
}

#[test]
fn performance_indicators_serde_roundtrip() {
    let p = PerformanceIndicators {
        avg_response_time_ms: 250.5,
        requests_per_second: 42.0,
        error_rate: 0.05,
        sla_compliance: 0.95,
    };
    let json = serde_json::to_string(&p).unwrap();
    let p2: PerformanceIndicators = serde_json::from_str(&json).unwrap();
    assert_eq!(p.avg_response_time_ms, p2.avg_response_time_ms);
    assert_eq!(p.error_rate, p2.error_rate);
}

#[test]
fn performance_indicators_extreme_values() {
    let p = PerformanceIndicators {
        avg_response_time_ms: f64::MAX,
        requests_per_second: f64::MIN_POSITIVE,
        error_rate: 1.0,
        sla_compliance: 0.0,
    };
    let json = serde_json::to_string(&p).unwrap();
    let p2: PerformanceIndicators = serde_json::from_str(&json).unwrap();
    assert_eq!(p.avg_response_time_ms, p2.avg_response_time_ms);
}

// ── ReadinessChecks ──────────────────────────────────────────────

#[test]
fn readiness_checks_all_true() {
    let c = ReadinessChecks {
        model_loaded: true,
        inference_engine_ready: true,
        device_available: true,
        resources_available: true,
    };
    assert!(c.model_loaded);
    assert!(c.inference_engine_ready);
    assert!(c.device_available);
    assert!(c.resources_available);
}

#[test]
fn readiness_checks_all_false() {
    let c = ReadinessChecks {
        model_loaded: false,
        inference_engine_ready: false,
        device_available: false,
        resources_available: false,
    };
    assert!(!c.model_loaded);
    assert!(!c.inference_engine_ready);
}

#[test]
fn readiness_checks_debug() {
    let c = ReadinessChecks {
        model_loaded: true,
        inference_engine_ready: false,
        device_available: true,
        resources_available: false,
    };
    let dbg = format!("{:?}", c);
    assert!(dbg.contains("ReadinessChecks"));
}

#[test]
fn readiness_checks_clone() {
    let c = ReadinessChecks {
        model_loaded: true,
        inference_engine_ready: false,
        device_available: true,
        resources_available: true,
    };
    let c2 = c.clone();
    assert_eq!(c.model_loaded, c2.model_loaded);
    assert_eq!(c.inference_engine_ready, c2.inference_engine_ready);
}

#[test]
fn readiness_checks_serde_roundtrip() {
    let c = ReadinessChecks {
        model_loaded: true,
        inference_engine_ready: true,
        device_available: false,
        resources_available: true,
    };
    let json = serde_json::to_string(&c).unwrap();
    let c2: ReadinessChecks = serde_json::from_str(&json).unwrap();
    assert_eq!(c.model_loaded, c2.model_loaded);
    assert_eq!(c.device_available, c2.device_available);
}

// ── ReadinessResponse ────────────────────────────────────────────

#[test]
fn readiness_response_debug() {
    let r = ReadinessResponse {
        status: "ready".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    let dbg = format!("{:?}", r);
    assert!(dbg.contains("ReadinessResponse"));
}

#[test]
fn readiness_response_clone() {
    let r = ReadinessResponse {
        status: "ready".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    let r2 = r.clone();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.checks.model_loaded, r2.checks.model_loaded);
}

#[test]
fn readiness_response_serde_roundtrip() {
    let r = ReadinessResponse {
        status: "not_ready".into(),
        timestamp: "2024-06-15T12:00:00Z".into(),
        checks: ReadinessChecks {
            model_loaded: false,
            inference_engine_ready: false,
            device_available: true,
            resources_available: true,
        },
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: ReadinessResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.checks.model_loaded, r2.checks.model_loaded);
}

#[test]
fn readiness_response_json_structure() {
    let r = ReadinessResponse {
        status: "ready".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        checks: ReadinessChecks {
            model_loaded: true,
            inference_engine_ready: true,
            device_available: true,
            resources_available: true,
        },
    };
    let v: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(v["status"], "ready");
    assert!(v["checks"]["model_loaded"].as_bool().unwrap());
}

// ── Ac05HealthResponse ───────────────────────────────────────────

#[test]
fn ac05_health_response_debug() {
    let r = Ac05HealthResponse {
        status: "healthy".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        components: HashMap::new(),
        system_metrics: SystemMetrics::default(),
        performance_indicators: PerformanceIndicators::default(),
    };
    let dbg = format!("{:?}", r);
    assert!(dbg.contains("Ac05HealthResponse"));
}

#[test]
fn ac05_health_response_clone() {
    let mut components = HashMap::new();
    components.insert("engine".to_string(), "healthy".to_string());

    let r = Ac05HealthResponse {
        status: "healthy".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        components,
        system_metrics: SystemMetrics::default(),
        performance_indicators: PerformanceIndicators::default(),
    };
    let r2 = r.clone();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.components.len(), r2.components.len());
}

#[test]
fn ac05_health_response_serde_roundtrip() {
    let mut components = HashMap::new();
    components.insert("model_manager".to_string(), "healthy".to_string());
    components.insert("batch_engine".to_string(), "degraded".to_string());

    let r = Ac05HealthResponse {
        status: "degraded".into(),
        timestamp: "2024-06-15T12:00:00Z".into(),
        components,
        system_metrics: SystemMetrics {
            cpu_utilization: 85.5,
            gpu_utilization: 90.0,
            memory_usage_bytes: 8_000_000_000,
            gpu_memory_usage_bytes: Some(4_000_000_000),
            active_requests: 50,
            queue_depth: 20,
        },
        performance_indicators: PerformanceIndicators {
            avg_response_time_ms: 500.0,
            requests_per_second: 25.0,
            error_rate: 0.02,
            sla_compliance: 0.98,
        },
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: Ac05HealthResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r.status, r2.status);
    assert_eq!(r.components["model_manager"], r2.components["model_manager"]);
    assert_eq!(r.system_metrics.cpu_utilization, r2.system_metrics.cpu_utilization);
}

#[test]
fn ac05_health_response_empty_components() {
    let r = Ac05HealthResponse {
        status: "unknown".into(),
        timestamp: "".into(),
        components: HashMap::new(),
        system_metrics: SystemMetrics::default(),
        performance_indicators: PerformanceIndicators::default(),
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: Ac05HealthResponse = serde_json::from_str(&json).unwrap();
    assert!(r2.components.is_empty());
}

#[test]
fn ac05_health_response_many_components() {
    let mut components = HashMap::new();
    for i in 0..100 {
        components.insert(format!("component_{}", i), "healthy".to_string());
    }
    let r = Ac05HealthResponse {
        status: "healthy".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        components,
        system_metrics: SystemMetrics::default(),
        performance_indicators: PerformanceIndicators::default(),
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: Ac05HealthResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.components.len(), 100);
}

#[test]
fn ac05_health_response_json_structure() {
    let mut components = HashMap::new();
    components.insert("engine".to_string(), "healthy".to_string());

    let r = Ac05HealthResponse {
        status: "healthy".into(),
        timestamp: "2024-01-01T00:00:00Z".into(),
        components,
        system_metrics: SystemMetrics { cpu_utilization: 50.0, ..Default::default() },
        performance_indicators: PerformanceIndicators::default(),
    };
    let v: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(v["status"], "healthy");
    assert_eq!(v["components"]["engine"], "healthy");
    assert_eq!(v["system_metrics"]["cpu_utilization"], 50.0);
}

// ── Deserialization from raw JSON ────────────────────────────────

#[test]
fn liveness_from_raw_json() {
    let json = r#"{"status":"alive","timestamp":"2024-01-01T00:00:00Z"}"#;
    let r: LivenessResponse = serde_json::from_str(json).unwrap();
    assert_eq!(r.status, "alive");
}

#[test]
fn system_metrics_from_raw_json_no_gpu() {
    let json = r#"{"cpu_utilization":50.0,"gpu_utilization":0.0,"memory_usage_bytes":1024,"active_requests":5,"queue_depth":2}"#;
    let m: SystemMetrics = serde_json::from_str(json).unwrap();
    assert_eq!(m.cpu_utilization, 50.0);
    assert!(m.gpu_memory_usage_bytes.is_none());
}

#[test]
fn system_metrics_from_raw_json_with_gpu() {
    let json = r#"{"cpu_utilization":50.0,"gpu_utilization":80.0,"memory_usage_bytes":2048,"gpu_memory_usage_bytes":4096,"active_requests":10,"queue_depth":3}"#;
    let m: SystemMetrics = serde_json::from_str(json).unwrap();
    assert_eq!(m.gpu_memory_usage_bytes, Some(4096));
}

#[test]
fn readiness_checks_from_raw_json() {
    let json = r#"{"model_loaded":true,"inference_engine_ready":false,"device_available":true,"resources_available":false}"#;
    let c: ReadinessChecks = serde_json::from_str(json).unwrap();
    assert!(c.model_loaded);
    assert!(!c.inference_engine_ready);
    assert!(c.device_available);
    assert!(!c.resources_available);
}

#[test]
fn ac05_health_response_from_raw_json() {
    let json = r#"{
        "status": "unhealthy",
        "timestamp": "2024-06-15T00:00:00Z",
        "components": {"model": "failed"},
        "system_metrics": {
            "cpu_utilization": 99.0,
            "gpu_utilization": 0.0,
            "memory_usage_bytes": 0,
            "active_requests": 0,
            "queue_depth": 0
        },
        "performance_indicators": {
            "avg_response_time_ms": 0.0,
            "requests_per_second": 0.0,
            "error_rate": 1.0,
            "sla_compliance": 0.0
        }
    }"#;
    let r: Ac05HealthResponse = serde_json::from_str(json).unwrap();
    assert_eq!(r.status, "unhealthy");
    assert_eq!(r.components["model"], "failed");
    assert_eq!(r.performance_indicators.error_rate, 1.0);
}
