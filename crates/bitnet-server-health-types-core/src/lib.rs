//! Reusable AC05 health endpoint response contracts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic health status used by server health endpoints.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual component health state and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub message: String,
    pub last_check: String,
    pub response_time_ms: Option<u64>,
}

/// Liveness probe response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    pub status: String,
    pub timestamp: String,
}

/// System metrics for AC05 health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemMetrics {
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub memory_usage_bytes: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_usage_bytes: Option<u32>,
    pub active_requests: i32,
    pub queue_depth: i32,
}

/// Performance indicators for AC05 SLA tracking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceIndicators {
    pub avg_response_time_ms: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub sla_compliance: f64,
}

/// Readiness checks for Kubernetes readiness probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessChecks {
    pub model_loaded: bool,
    pub inference_engine_ready: bool,
    pub device_available: bool,
    pub resources_available: bool,
}

/// Readiness probe response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub status: String,
    pub timestamp: String,
    pub checks: ReadinessChecks,
}

/// AC05 health response with component status strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ac05HealthResponse {
    pub status: String,
    pub timestamp: String,
    pub components: HashMap<String, ComponentHealth>,
    pub system_metrics: SystemMetrics,
    pub performance_indicators: PerformanceIndicators,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ac05_health_response_serialization_matches_expected_shape() {
        let response = Ac05HealthResponse {
            status: "healthy".to_string(),
            timestamp: "2023-12-01T10:30:00Z".to_string(),
            components: [
                (
                    "model_manager".to_string(),
                    ComponentHealth {
                        status: HealthStatus::Healthy,
                        message: "ok".to_string(),
                        last_check: "2023-12-01T10:30:00Z".to_string(),
                        response_time_ms: Some(4),
                    },
                ),
                (
                    "execution_router".to_string(),
                    ComponentHealth {
                        status: HealthStatus::Healthy,
                        message: "ok".to_string(),
                        last_check: "2023-12-01T10:30:00Z".to_string(),
                        response_time_ms: Some(3),
                    },
                ),
            ]
            .into_iter()
            .collect(),
            system_metrics: SystemMetrics {
                cpu_utilization: 0.65,
                gpu_utilization: 0.78,
                memory_usage_bytes: 6_442_450_944,
                gpu_memory_usage_bytes: Some(2_147_483_648),
                active_requests: 23,
                queue_depth: 5,
            },
            performance_indicators: PerformanceIndicators {
                avg_response_time_ms: 1245.0,
                requests_per_second: 15.2,
                error_rate: 0.0035,
                sla_compliance: 0.995,
            },
        };

        let serialized = serde_json::to_value(&response).unwrap();

        assert_eq!(serialized["status"], "healthy");
        assert_eq!(serialized["timestamp"], "2023-12-01T10:30:00Z");
        assert_eq!(serialized["components"]["model_manager"]["status"], "healthy");
        assert_eq!(serialized["system_metrics"]["cpu_utilization"], 0.65);
        assert_eq!(serialized["performance_indicators"]["sla_compliance"], 0.995);
    }

    #[test]
    fn health_status_serializes_as_lowercase() {
        assert_eq!(serde_json::to_string(&HealthStatus::Degraded).unwrap(), "\"degraded\"");
    }

    #[test]
    fn liveness_response_serialization_matches_schema() {
        let response = LivenessResponse {
            status: "healthy".to_string(),
            timestamp: "2023-12-01T10:30:00Z".to_string(),
        };

        let serialized = serde_json::to_value(&response).unwrap();
        let expected = json!({
            "status": "healthy",
            "timestamp": "2023-12-01T10:30:00Z"
        });

        assert_eq!(serialized, expected);
    }

    #[test]
    fn readiness_response_serialization_contains_checks() {
        let response = ReadinessResponse {
            status: "ready".to_string(),
            timestamp: "2023-12-01T10:30:00Z".to_string(),
            checks: ReadinessChecks {
                model_loaded: true,
                inference_engine_ready: true,
                device_available: true,
                resources_available: true,
            },
        };

        let serialized = serde_json::to_value(&response).unwrap();

        assert_eq!(serialized["status"], "ready");
        assert_eq!(serialized["checks"]["model_loaded"], true);
        assert_eq!(serialized["checks"]["inference_engine_ready"], true);
    }

    #[test]
    fn system_metrics_default_is_zero_initialized() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.cpu_utilization, 0.0);
        assert_eq!(metrics.gpu_utilization, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
        assert_eq!(metrics.active_requests, 0);
        assert_eq!(metrics.queue_depth, 0);
    }

    #[test]
    fn performance_indicators_default_is_zero_initialized() {
        let indicators = PerformanceIndicators::default();
        assert_eq!(indicators.avg_response_time_ms, 0.0);
        assert_eq!(indicators.requests_per_second, 0.0);
        assert_eq!(indicators.error_rate, 0.0);
        assert_eq!(indicators.sla_compliance, 0.0);
    }
}
