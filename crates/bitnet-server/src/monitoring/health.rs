//! Health check endpoints for load balancer integration

use async_trait::async_trait;
use axum::{
    Router,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::metrics::MetricsCollector;

/// Health check status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

/// Build metadata included in health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub git_sha: String,
    pub git_branch: String,
    pub build_timestamp: String,
    pub rustc_version: String,
    pub cargo_target: String,
    pub cargo_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<String>,
}

/// Overall health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: String,
    pub uptime_seconds: u64,
    pub version: String,
    pub build: BuildInfo,
    pub components: HashMap<String, ComponentHealth>,
    pub metrics: HealthMetrics,
}

/// Key health metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthMetrics {
    pub active_requests: u64,
    pub total_requests: u64,
    pub error_rate_percent: f64,
    pub avg_response_time_ms: f64,
    pub memory_usage_mb: f64,
    pub tokens_per_second: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_mb: Option<f64>,
}

/// Internal memory information
#[derive(Debug, Clone)]
struct MemoryInfo {
    used_bytes: u64,
    total_bytes: u64,
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
        #[cfg(any(feature = "gpu", feature = "cuda"))]
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
            build: BuildInfo {
                version: env!("CARGO_PKG_VERSION").to_string(),
                git_sha: option_env!("VERGEN_GIT_SHA").unwrap_or("unknown").to_string(),
                git_branch: option_env!("VERGEN_GIT_BRANCH").unwrap_or("unknown").to_string(),
                build_timestamp: option_env!("VERGEN_BUILD_TIMESTAMP")
                    .unwrap_or("unknown")
                    .to_string(),
                rustc_version: option_env!("VERGEN_RUSTC_SEMVER").unwrap_or("unknown").to_string(),
                cargo_target: option_env!("VERGEN_CARGO_TARGET_TRIPLE")
                    .unwrap_or("unknown")
                    .to_string(),
                cargo_profile: option_env!("VERGEN_CARGO_OPT_LEVEL")
                    .unwrap_or("unknown")
                    .to_string(),
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                cuda_version: Some(self.get_cuda_version()),
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                cuda_version: None,
            },
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

        // Check memory usage using actual system metrics
        let memory_usage_percent = match self.get_memory_info().await {
            Ok(info) => {
                if info.total_bytes > 0 {
                    (info.used_bytes as f64 / info.total_bytes as f64) * 100.0
                } else {
                    0.0
                }
            }
            Err(_) => 0.0, // Default to 0% if we can't read memory
        };

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
    #[cfg(any(feature = "gpu", feature = "cuda"))]
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
    #[cfg(any(feature = "gpu", feature = "cuda"))]
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
            if let Some(component) = components.get(*component_name)
                && component.status == HealthStatus::Unhealthy
            {
                return HealthStatus::Unhealthy;
            }
        }

        // If any critical component is degraded, overall status is degraded
        for component_name in &critical_components {
            if let Some(component) = components.get(*component_name)
                && component.status == HealthStatus::Degraded
            {
                return HealthStatus::Degraded;
            }
        }

        // Check non-critical components
        let unhealthy_count =
            components.values().filter(|c| c.status == HealthStatus::Unhealthy).count();

        let degraded_count =
            components.values().filter(|c| c.status == HealthStatus::Degraded).count();

        overall_status_from_counts(unhealthy_count, degraded_count)
    }

    async fn collect_health_metrics(&self) -> HealthMetrics {
        // Collect actual metrics from the metrics collector
        let memory_info =
            self.get_memory_info().await.unwrap_or(MemoryInfo { used_bytes: 0, total_bytes: 0 });

        let cpu_usage = self.get_cpu_usage().await.ok();

        HealthMetrics {
            active_requests: 0, // TODO: Wire to actual metrics collector
            total_requests: 0,  // TODO: Wire to actual metrics collector
            error_rate_percent: 0.0,
            avg_response_time_ms: 0.0,
            memory_usage_mb: (memory_info.used_bytes as f64) / (1024.0 * 1024.0),
            tokens_per_second: 0.0,
            cpu_usage_percent: cpu_usage,
            gpu_memory_mb: None, // TODO: Wire to GPU metrics when available
        }
    }

    async fn get_cpu_usage(&self) -> Result<f64, String> {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_cpu_all();
        // Small sleep to get accurate CPU measurement
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        sys.refresh_cpu_all();
        Ok(sys.global_cpu_usage() as f64)
    }

    async fn get_memory_info(&self) -> Result<MemoryInfo, String> {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        Ok(MemoryInfo { used_bytes: sys.used_memory(), total_bytes: sys.total_memory() })
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn get_cuda_version(&self) -> String {
        // In production, query actual CUDA version
        "12.3".to_string()
    }
}

#[inline]
fn overall_status_from_counts(unhealthy_count: usize, degraded_count: usize) -> HealthStatus {
    if unhealthy_count > 0 {
        HealthStatus::Unhealthy
    } else if degraded_count > 0 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

// HTTP mapping is configurable at build-time:
// - Default (fail-fast): Degraded → 503, Unhealthy → 503
// - With `--features degraded-ok`: Degraded → 200, Unhealthy → 503
#[cfg(feature = "degraded-ok")]
#[inline]
fn status_code_for(status: HealthStatus) -> StatusCode {
    match status {
        HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[cfg(not(feature = "degraded-ok"))]
#[inline]
fn status_code_for(status: HealthStatus) -> StatusCode {
    match status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded | HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[inline]
fn with_no_store_headers(res: Response) -> Response {
    let mut res = res;
    res.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    res
}

/// Abstraction for probing health (used to inject a stub in tests)
#[async_trait]
pub trait HealthProbe: Send + Sync + 'static {
    async fn check_health(&self) -> HealthResponse;
    async fn check_liveness(&self) -> HealthStatus;
    async fn check_readiness(&self) -> HealthStatus;
}

#[async_trait]
impl HealthProbe for HealthChecker {
    async fn check_health(&self) -> HealthResponse {
        HealthChecker::check_health(self).await
    }

    async fn check_liveness(&self) -> HealthStatus {
        HealthChecker::check_liveness(self).await
    }

    async fn check_readiness(&self) -> HealthStatus {
        HealthChecker::check_readiness(self).await
    }
}

/// Production constructor keeps the same API
pub fn create_health_routes(health_checker: Arc<HealthChecker>) -> Router {
    create_health_routes_with_probe(health_checker)
}

/// Generic constructor (used by tests to inject a stub)
pub fn create_health_routes_with_probe<T: HealthProbe>(probe: Arc<T>) -> Router {
    Router::new()
        .route("/health", get(health_handler::<T>))
        .route("/health/live", get(liveness_handler::<T>))
        .route("/health/ready", get(readiness_handler::<T>))
        .with_state(probe)
}

/// Comprehensive health check endpoint
async fn health_handler<T: HealthProbe>(State(probe): State<Arc<T>>) -> Response {
    let health = probe.check_health().await;
    let status_code = status_code_for(health.status);
    with_no_store_headers((status_code, Json(health)).into_response())
}

/// Liveness probe endpoint (Kubernetes)
async fn liveness_handler<T: HealthProbe>(State(probe): State<Arc<T>>) -> Response {
    with_no_store_headers(status_code_for(probe.check_liveness().await).into_response())
}

/// Readiness probe endpoint (Kubernetes)
async fn readiness_handler<T: HealthProbe>(State(probe): State<Arc<T>>) -> Response {
    // Readiness always uses strict fail-fast (degraded = not ready)
    match probe.check_readiness().await {
        HealthStatus::Healthy => with_no_store_headers(StatusCode::OK.into_response()),
        HealthStatus::Degraded | HealthStatus::Unhealthy => {
            with_no_store_headers(StatusCode::SERVICE_UNAVAILABLE.into_response())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BuildInfo, HealthMetrics, HealthProbe, HealthResponse, HealthStatus,
        create_health_routes_with_probe, overall_status_from_counts, status_code_for,
    };
    use async_trait::async_trait;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{HeaderValue, Request, StatusCode, header};
    use std::sync::Arc;
    use tower::ServiceExt;

    #[test]
    fn overall_unhealthy_wins() {
        assert!(matches!(overall_status_from_counts(1, 0), HealthStatus::Unhealthy));
        assert!(matches!(overall_status_from_counts(3, 2), HealthStatus::Unhealthy));
    }

    #[test]
    fn overall_degraded_when_no_unhealthy() {
        assert!(matches!(overall_status_from_counts(0, 1), HealthStatus::Degraded));
        assert!(matches!(overall_status_from_counts(0, 7), HealthStatus::Degraded));
    }

    #[test]
    fn overall_healthy_when_none() {
        assert!(matches!(overall_status_from_counts(0, 0), HealthStatus::Healthy));
    }

    // Mapping tests (compile-time conditioned)
    #[cfg(not(feature = "degraded-ok"))]
    #[test]
    fn http_mapping_fail_fast_default() {
        assert_eq!(status_code_for(HealthStatus::Healthy), StatusCode::OK);
        assert_eq!(status_code_for(HealthStatus::Degraded), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_code_for(HealthStatus::Unhealthy), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[cfg(feature = "degraded-ok")]
    #[test]
    fn http_mapping_degraded_ok_feature() {
        assert_eq!(status_code_for(HealthStatus::Healthy), StatusCode::OK);
        assert_eq!(status_code_for(HealthStatus::Degraded), StatusCode::OK);
        assert_eq!(status_code_for(HealthStatus::Unhealthy), StatusCode::SERVICE_UNAVAILABLE);
    }

    // ---- Route-level tests with a stubbed probe ----
    struct StubProbe {
        overall: HealthStatus,
        live: HealthStatus,
        ready: HealthStatus,
    }

    #[async_trait]
    impl HealthProbe for StubProbe {
        async fn check_health(&self) -> HealthResponse {
            HealthResponse {
                status: self.overall,
                timestamp: chrono::Utc::now().to_rfc3339(),
                uptime_seconds: 0,
                version: "test".to_string(),
                build: BuildInfo {
                    version: "test".to_string(),
                    git_sha: "test-sha".to_string(),
                    git_branch: "test".to_string(),
                    build_timestamp: "test".to_string(),
                    rustc_version: "test".to_string(),
                    cargo_target: "test".to_string(),
                    cargo_profile: "test".to_string(),
                    cuda_version: None,
                },
                components: Default::default(),
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

    #[cfg(not(feature = "degraded-ok"))]
    #[tokio::test]
    async fn route_health_fail_fast_mapping() {
        // Default (no feature): Degraded -> 503
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Degraded,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Healthy,
        }));
        let resp = app.oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[cfg(feature = "degraded-ok")]
    #[tokio::test]
    async fn route_health_degraded_ok_mapping() {
        // With feature: Degraded -> 200
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Degraded,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Healthy,
        }));
        let resp = app.oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn route_readiness_always_fail_fast() {
        // Readiness always uses strict fail-fast
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Degraded,
        }));
        let resp =
            app.oneshot(Request::get("/health/ready").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn route_live_uses_mapping() {
        // Degraded should follow mapping (default: 503; with `degraded-ok`: 200)
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Degraded,
            ready: HealthStatus::Healthy,
        }));
        let resp =
            app.oneshot(Request::get("/health/live").body(Body::empty()).unwrap()).await.unwrap();

        #[cfg(not(feature = "degraded-ok"))]
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        #[cfg(feature = "degraded-ok")]
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cache_control_headers_on_all_endpoints() {
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Healthy,
        }));

        // Test /health endpoint
        let resp = app
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );

        // Test /health/live endpoint
        let resp = app
            .clone()
            .oneshot(Request::get("/health/live").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );

        // Test /health/ready endpoint
        let resp =
            app.oneshot(Request::get("/health/ready").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn head_requests_set_no_store() {
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Healthy,
        }));

        for path in ["/health", "/health/live", "/health/ready"] {
            let req = Request::builder().method("HEAD").uri(path).body(Body::empty()).unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            // Healthy stub ⇒ 200 for all three endpoints.
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(
                resp.headers().get(header::CACHE_CONTROL),
                Some(&HeaderValue::from_static("no-store"))
            );
        }
    }

    #[tokio::test]
    async fn head_respects_mapping_on_degraded() {
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Degraded,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Healthy,
        }));
        let req = Request::builder().method("HEAD").uri("/health").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();

        #[cfg(not(feature = "degraded-ok"))]
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        #[cfg(feature = "degraded-ok")]
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn head_readiness_degraded_is_503() {
        // Readiness is strict regardless of mapping.
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Healthy,
            ready: HealthStatus::Degraded,
        }));
        let req =
            Request::builder().method("HEAD").uri("/health/ready").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    #[tokio::test]
    async fn head_live_respects_mapping_on_degraded() {
        // Degraded should follow mapping (default: 503; with `degraded-ok`: 200)
        let app: Router = create_health_routes_with_probe(Arc::new(StubProbe {
            overall: HealthStatus::Healthy,
            live: HealthStatus::Degraded,
            ready: HealthStatus::Healthy,
        }));
        let req =
            Request::builder().method("HEAD").uri("/health/live").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();

        #[cfg(not(feature = "degraded-ok"))]
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        #[cfg(feature = "degraded-ok")]
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }
}
