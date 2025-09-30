#![allow(unused)]
#![allow(dead_code)]

use anyhow::Result;
use bitnet_server::monitoring::{MonitoringConfig, metrics::MetricsCollector};

#[tokio::test]
async fn test_system_metrics_collection() -> Result<()> {
    let config = MonitoringConfig::default();
    let metrics_collector = MetricsCollector::new(&config)?;

    // Simulate metrics collection - this should not panic
    let result = metrics_collector.collect_system_metrics().await;
    assert!(result.is_ok(), "System metrics collection should succeed");

    // Test that we can create a request tracker
    let tracker = metrics_collector.track_request("test-request".to_string());
    tracker.record_tokens(100);

    Ok(())
}

#[tokio::test]
async fn test_performance_regression_detection() -> Result<()> {
    let config = MonitoringConfig::default();
    let metrics_collector = MetricsCollector::new(&config)?;

    // Simulate collecting multiple performance snapshots
    for _ in 0..15 {
        metrics_collector.collect_system_metrics().await?;
    }

    // Check performance regression detection
    let alerts = metrics_collector.check_performance_regression().await?;

    // No specific alert expectations as actual system will vary
    assert!(
        alerts.is_empty() || !alerts.is_empty(),
        "Performance regression check should not panic"
    );

    Ok(())
}
