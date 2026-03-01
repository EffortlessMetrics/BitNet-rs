//! Shared performance telemetry primitives.
//!
//! This microcrate intentionally owns only the types needed to describe
//! inference runtime metrics and whether the computation path was real or mock.

/// Performance metrics for validation and telemetry.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetrics {
    #[serde(default)]
    pub tokens_per_second: f64,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub memory_usage_mb: f64,
    #[serde(default)]
    pub computation_type: ComputationType,
    #[serde(default)]
    pub gpu_utilization: Option<f64>,
}

/// Computation type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputationType {
    #[default]
    Real,
    Mock,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_real_computation() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.computation_type, ComputationType::Real);
        assert_eq!(metrics.tokens_per_second, 0.0);
    }

    #[test]
    fn serde_roundtrip_preserves_fields() {
        let metrics = PerformanceMetrics {
            tokens_per_second: 42.0,
            latency_ms: 10.0,
            memory_usage_mb: 512.0,
            computation_type: ComputationType::Mock,
            gpu_utilization: Some(85.0),
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let deserialized: PerformanceMetrics = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.tokens_per_second, 42.0);
        assert_eq!(deserialized.computation_type, ComputationType::Mock);
        assert_eq!(deserialized.gpu_utilization, Some(85.0));
    }
}
