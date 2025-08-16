//! Health check endpoints for load balancer integration
#![allow(unexpected_cfgs)]

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::metrics::MetricsCollector;

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub message: String,
    pub last_check: String,
    pub response_time_ms: Option<u64>,
}

/// Overall health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: String,
    pub uptime_seconds: u64,
    pub version: String,
    pub components: HashMap<String, ComponentHealth>,
    pub metrics: HealthMetrics,
}

/// Key health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub active_requests: u64,
    pub total_requests: u64,
    pub error_rate_percent: f64,
    pub avg_response_time_ms: f64,
    pub memory_usage_mb: f64,
    pub tokens_per_second: f64,
}

/// Health checker that monitors system components
pub struct HealthChecker {
    start_time: Instant,
    #[allow(dead_code)]
    metrics: Arc<MetricsCollector>,
    #[allow(dead_code)]
    component_checks: Arc<RwLock<HashMap<String, ComponentHealth>>>,
}

impl HealthChecker {
    pub fn new(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            start_time: Instant::now(),
            metrics,
            component_checks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Perform comprehensive health check
    pub async fn check_health(&self) -> HealthResponse {
        let mut components = HashMap::new();

        // Check model availability
        components.insert("model".to_string(), self.check_model_health().await);

        // Check memory usage
        components.insert("memory".to_string(), self.check_memory_health().await);

        // Check inference engine
        components
            .insert("inference_engine".to_string(), self.check_inference_engine_health().await);

        // Check GPU availability (if enabled)
        #[cfg(feature = "cuda")]
        components.insert("gpu".to_string(), self.check_gpu_health().await);

        // Determine overall status
        let overall_status = self.determine_overall_status(&components);

        // Collect metrics
        let metrics = self.collect_health_metrics().await;

        HealthResponse {
            status: overall_status,
            timestamp: chrono::Utc::now().to_rfc3339(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            components,
            metrics,
        }
    }

    /// Quick liveness check (for Kubernetes liveness probe)
    pub async fn check_liveness(&self) -> HealthStatus {
        // Basic checks that indicate the service is running
        if self.start_time.elapsed() < Duration::from_secs(5) {
            return HealthStatus::Degraded; // Still starting up
        }

        // Check if we can perform basic operations
        match self.check_basic_functionality().await {
            Ok(_) => HealthStatus::Healthy,
            Err(_) => HealthStatus::Unhealthy,
        }
    }

    /// Readiness check (for Kubernetes readiness probe)
    pub async fn check_readiness(&self) -> HealthStatus {
        // More comprehensive checks to determine if service can handle traffic
        let model_health = self.check_model_health().await;
        let memory_health = self.check_memory_health().await;
        let inference_health = self.check_inference_engine_health().await;

        let critical_components = [&model_health, &memory_health, &inference_health];

        if critical_components.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if critical_components.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    async fn check_model_health(&self) -> ComponentHealth {
        let start = Instant::now();

        // In a real implementation, this would check if models are loaded and accessible
        let status = HealthStatus::Healthy;
        let message = "Model loaded and ready".to_string();

        ComponentHealth {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
        }
    }

    async fn check_memory_health(&self) -> ComponentHealth {
        let start = Instant::now();

        // Check memory usage - in production, use actual system metrics
        let memory_usage_percent = 45.0; // Placeholder

        let (status, message) = if memory_usage_percent > 90.0 {
            (HealthStatus::Unhealthy, format!("High memory usage: {:.1}%", memory_usage_percent))
        } else if memory_usage_percent > 80.0 {
            (HealthStatus::Degraded, format!("Elevated memory usage: {:.1}%", memory_usage_percent))
        } else {
            (HealthStatus::Healthy, format!("Memory usage normal: {:.1}%", memory_usage_percent))
        };

        ComponentHealth {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
        }
    }

    async fn check_inference_engine_health(&self) -> ComponentHealth {
        let start = Instant::now();

        // Check if inference engine is responsive
        // In production, this might perform a lightweight inference test
        let status = HealthStatus::Healthy;
        let message = "Inference engine responsive".to_string();

        ComponentHealth {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
        }
    }

    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    async fn check_gpu_health(&self) -> ComponentHealth {
        let start = Instant::now();

        // Check GPU availability and memory
        let (status, message) = match self.check_gpu_status().await {
            Ok(gpu_info) => (HealthStatus::Healthy, format!("GPU available: {}", gpu_info)),
            Err(e) => (HealthStatus::Degraded, format!("GPU check failed: {}", e)),
        };

        ComponentHealth {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            response_time_ms: Some(start.elapsed().as_millis() as u64),
        }
    }

    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    async fn check_gpu_status(&self) -> Result<String, String> {
        // Placeholder GPU check - in production, use actual CUDA queries
        Ok("CUDA device 0 available".to_string())
    }

    async fn check_basic_functionality(&self) -> Result<(), String> {
        // Perform basic functionality checks
        // This could include checking if we can access configuration,
        // create basic objects, etc.
        Ok(())
    }

    fn determine_overall_status(
        &self,
        components: &HashMap<String, ComponentHealth>,
    ) -> HealthStatus {
        let critical_components = ["model", "memory", "inference_engine"];

        // Check critical components first
        for component_name in &critical_components {
            if let Some(component) = components.get(*component_name) {
                if component.status == HealthStatus::Unhealthy {
                    return HealthStatus::Unhealthy;
                }
            }
        }

        // If any critical component is degraded, overall status is degraded
        for component_name in &critical_components {
            if let Some(component) = components.get(*component_name) {
                if component.status == HealthStatus::Degraded {
                    return HealthStatus::Degraded;
                }
            }
        }

        // Check non-critical components
        let unhealthy_count =
            components.values().filter(|c| c.status == HealthStatus::Unhealthy).count();

        let degraded_count =
            components.values().filter(|c| c.status == HealthStatus::Degraded).count();

        if unhealthy_count > 0 || degraded_count > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    async fn collect_health_metrics(&self) -> HealthMetrics {
        // In production, these would be collected from actual metrics
        HealthMetrics {
            active_requests: 0,
            total_requests: 0,
            error_rate_percent: 0.0,
            avg_response_time_ms: 0.0,
            memory_usage_mb: 0.0,
            tokens_per_second: 0.0,
        }
    }
}

/// Create health check routes
pub fn create_health_routes(health_checker: Arc<HealthChecker>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/health/live", get(liveness_handler))
        .route("/health/ready", get(readiness_handler))
        .with_state(health_checker)
}

/// Comprehensive health check endpoint
async fn health_handler(
    State(health_checker): State<Arc<HealthChecker>>,
) -> Result<Json<HealthResponse>, StatusCode> {
    let health = health_checker.check_health().await;

    let _status_code = match health.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still serving traffic
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    Ok(Json(health))
}

/// Liveness probe endpoint (Kubernetes)
async fn liveness_handler(State(health_checker): State<Arc<HealthChecker>>) -> StatusCode {
    match health_checker.check_liveness().await {
        HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Readiness probe endpoint (Kubernetes)
async fn readiness_handler(State(health_checker): State<Arc<HealthChecker>>) -> StatusCode {
    match health_checker.check_readiness().await {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded | HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    }
}
