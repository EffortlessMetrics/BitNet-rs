//! AC05 health check data structures
//!
//! Types for the AC05 health check response schema as specified in
//! issue-251-production-inference-server-architecture.md#ac5-health-checks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Liveness probe response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    pub status: String,
    pub timestamp: String,
}

/// System metrics for AC05 health monitoring
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

/// Performance indicators for AC05 SLA tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceIndicators {
    pub avg_response_time_ms: f64,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub sla_compliance: f64,
}

/// Readiness checks for Kubernetes readiness probe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessChecks {
    pub model_loaded: bool,
    pub inference_engine_ready: bool,
    pub device_available: bool,
    pub resources_available: bool,
}

/// Readiness probe response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub status: String,
    pub timestamp: String,
    pub checks: ReadinessChecks,
}

/// AC05 health response with component status strings
///
/// This matches the schema expected by AC05 tests:
/// ```json
/// {
///   "status": "healthy|degraded|unhealthy",
///   "timestamp": "ISO8601",
///   "components": {
///     "model_manager": "status",
///     "execution_router": "status",
///     ...
///   },
///   "system_metrics": { ... },
///   "performance_indicators": { ... }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ac05HealthResponse {
    pub status: String,
    pub timestamp: String,
    pub components: HashMap<String, String>,
    pub system_metrics: SystemMetrics,
    pub performance_indicators: PerformanceIndicators,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ac05_health_response_serialization() {
        let response = Ac05HealthResponse {
            status: "healthy".to_string(),
            timestamp: "2023-12-01T10:30:00Z".to_string(),
            components: [
                ("model_manager".to_string(), "healthy".to_string()),
                ("execution_router".to_string(), "healthy".to_string()),
                ("batch_engine".to_string(), "healthy".to_string()),
                ("device_monitor".to_string(), "healthy".to_string()),
                ("quantization_engine".to_string(), "healthy".to_string()),
            ]
            .into_iter()
            .collect(),
            system_metrics: SystemMetrics {
                cpu_utilization: 0.65,
                gpu_utilization: 0.78,
                memory_usage_bytes: 6442450944,
                gpu_memory_usage_bytes: Some(2147483648),
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
        assert_eq!(serialized["components"]["model_manager"], "healthy");
        assert_eq!(serialized["system_metrics"]["cpu_utilization"], 0.65);
        assert_eq!(serialized["performance_indicators"]["sla_compliance"], 0.995);
    }

    #[test]
    fn test_liveness_response_serialization() {
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
    fn test_readiness_response_serialization() {
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
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.cpu_utilization, 0.0);
        assert_eq!(metrics.gpu_utilization, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
        assert_eq!(metrics.active_requests, 0);
        assert_eq!(metrics.queue_depth, 0);
    }

    #[test]
    fn test_performance_indicators_default() {
        let indicators = PerformanceIndicators::default();
        assert_eq!(indicators.avg_response_time_ms, 0.0);
        assert_eq!(indicators.requests_per_second, 0.0);
        assert_eq!(indicators.error_rate, 0.0);
        assert_eq!(indicators.sla_compliance, 0.0);
    }
}
