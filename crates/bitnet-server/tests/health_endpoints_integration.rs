//! Integration tests for health endpoints with real metrics
//!
//! These tests verify that the health endpoints return proper JSON responses
//! with actual system metrics and respond within performance requirements.

use anyhow::Result;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tower::ServiceExt;

// Import from bitnet-server
use bitnet_server::monitoring::{
    MonitoringConfig,
    health::{HealthChecker, create_health_routes},
    metrics::MetricsCollector,
};

#[tokio::test]
async fn test_health_endpoint_returns_json() -> Result<()> {
    // Setup health checker with real metrics
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));

    // Create router with health routes
    let app = create_health_routes(health_checker);

    // Send request to /health
    let request = Request::builder().uri("/health").body(Body::empty())?;

    let response = app.oneshot(request).await?;

    // Verify response
    assert_eq!(response.status(), StatusCode::OK);

    // Parse JSON body
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let json: Value = serde_json::from_slice(&body_bytes)?;

    // Verify required fields exist
    assert!(json.get("status").is_some(), "Missing 'status' field");
    assert!(json.get("timestamp").is_some(), "Missing 'timestamp' field");
    assert!(json.get("uptime_seconds").is_some(), "Missing 'uptime_seconds' field");
    assert!(json.get("version").is_some(), "Missing 'version' field");
    assert!(json.get("build").is_some(), "Missing 'build' field");
    assert!(json.get("components").is_some(), "Missing 'components' field");
    assert!(json.get("metrics").is_some(), "Missing 'metrics' field");

    // Verify status is valid
    let status = json["status"].as_str().unwrap();
    assert!(["healthy", "degraded", "unhealthy"].contains(&status), "Invalid status: {}", status);

    Ok(())
}

#[tokio::test]
async fn test_health_endpoint_includes_real_metrics() -> Result<()> {
    // Setup health checker
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));

    let app = create_health_routes(health_checker);

    // Send request
    let request = Request::builder().uri("/health").body(Body::empty())?;

    let response = app.oneshot(request).await?;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let json: Value = serde_json::from_slice(&body_bytes)?;

    // Verify metrics are populated
    let metrics_obj = &json["metrics"];
    assert!(metrics_obj.is_object(), "Metrics should be an object");

    // Check that memory_usage_mb is a number and reasonable
    let memory_usage = metrics_obj["memory_usage_mb"].as_f64().unwrap();
    assert!(memory_usage >= 0.0, "Memory usage should be non-negative");

    // Check CPU usage if available
    if let Some(cpu_usage) = metrics_obj.get("cpu_usage_percent").and_then(|v| v.as_f64()) {
        assert!((0.0..=100.0).contains(&cpu_usage), "CPU usage should be 0-100%");
    }

    Ok(())
}

#[tokio::test]
async fn test_liveness_probe_responds_quickly() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));

    // Wait for startup period (5 seconds) to complete
    // This is intentional behavior - liveness returns Degraded during startup
    tokio::time::sleep(std::time::Duration::from_secs(6)).await;

    let app = create_health_routes(health_checker);

    // Measure response time
    let start = Instant::now();

    let request = Request::builder().uri("/health/live").body(Body::empty())?;

    let response = app.oneshot(request).await?;
    let elapsed = start.elapsed();

    // Verify response is OK after startup
    assert_eq!(response.status(), StatusCode::OK);

    // Verify response time is under 200ms (allowing for CPU measurement overhead)
    assert!(
        elapsed.as_millis() < 200,
        "Liveness probe took {}ms, should be <200ms",
        elapsed.as_millis()
    );

    // Verify no-store cache header
    let cache_control = response.headers().get("cache-control");
    assert_eq!(cache_control.map(|v| v.to_str().unwrap()), Some("no-store"));

    Ok(())
}

#[tokio::test]
async fn test_readiness_probe_checks_components() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(health_checker);

    // Send request to /health/ready
    let request = Request::builder().uri("/health/ready").body(Body::empty())?;

    let response = app.oneshot(request).await?;

    // Readiness should return 200 or 503
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::SERVICE_UNAVAILABLE,
        "Readiness probe should return 200 or 503, got {}",
        response.status()
    );

    // Verify no-store cache header
    let cache_control = response.headers().get("cache-control");
    assert_eq!(cache_control.map(|v| v.to_str().unwrap()), Some("no-store"));

    Ok(())
}

#[tokio::test]
async fn test_health_components_exist() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(health_checker);

    // Send request
    let request = Request::builder().uri("/health").body(Body::empty())?;

    let response = app.oneshot(request).await?;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let json: Value = serde_json::from_slice(&body_bytes)?;

    // Verify components
    let components = json["components"].as_object().unwrap();

    // Check for critical components
    let required_components = ["model", "memory", "inference_engine"];
    for component_name in required_components {
        assert!(components.contains_key(component_name), "Missing component: {}", component_name);

        let component = &components[component_name];
        assert!(component.get("status").is_some());
        assert!(component.get("message").is_some());
        assert!(component.get("last_check").is_some());
    }

    Ok(())
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
#[tokio::test]
async fn test_health_includes_gpu_component() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(health_checker);

    // Send request
    let request = Request::builder().uri("/health").body(Body::empty())?;

    let response = app.oneshot(request).await?;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let json: Value = serde_json::from_slice(&body_bytes)?;

    // Verify GPU component exists when GPU features are enabled
    let components = json["components"].as_object().unwrap();
    assert!(
        components.contains_key("gpu"),
        "GPU component should be present with GPU features enabled"
    );

    Ok(())
}

#[tokio::test]
async fn test_build_info_populated() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));
    let app = create_health_routes(health_checker);

    // Send request
    let request = Request::builder().uri("/health").body(Body::empty())?;

    let response = app.oneshot(request).await?;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    let json: Value = serde_json::from_slice(&body_bytes)?;

    // Verify build info
    let build = &json["build"];
    assert!(build.is_object(), "Build info should be an object");

    // Check required fields
    assert!(build.get("version").is_some());
    assert!(build.get("git_sha").is_some());
    assert!(build.get("git_branch").is_some());
    assert!(build.get("build_timestamp").is_some());
    assert!(build.get("rustc_version").is_some());
    assert!(build.get("cargo_target").is_some());
    assert!(build.get("cargo_profile").is_some());

    Ok(())
}

#[tokio::test]
async fn test_multiple_concurrent_health_checks() -> Result<()> {
    // Setup
    let config = MonitoringConfig::default();
    let metrics = Arc::new(MetricsCollector::new(&config)?);
    let health_checker = Arc::new(HealthChecker::new(metrics));

    // Wait for startup period to complete
    tokio::time::sleep(std::time::Duration::from_secs(6)).await;

    // Spawn multiple concurrent requests
    let mut handles = vec![];
    for _ in 0..10 {
        let checker = health_checker.clone();
        let handle = tokio::spawn(async move {
            let app = create_health_routes(checker);
            let request = Request::builder().uri("/health/live").body(Body::empty()).unwrap();
            app.oneshot(request).await
        });
        handles.push(handle);
    }

    // Wait for all requests to complete
    let results = futures::future::join_all(handles).await;

    // Verify all succeeded
    for result in results {
        let response = result??;
        assert_eq!(response.status(), StatusCode::OK);
    }

    Ok(())
}
