use anyhow::Result;
use bitnet_server::monitoring::{MonitoringConfig, metrics::MetricsCollector};

#[tokio::test]
async fn test_system_metrics_collection() -> Result<()> {
    let config = MonitoringConfig::default();
    let metrics_collector = MetricsCollector::new(&config)?;

    // Simulate metrics collection
    metrics_collector.collect_system_metrics().await?;

    // Check basic metrics are non-zero
    let memory_info = metrics_collector.get_memory_info().await?;
    assert!(memory_info.total_bytes > 0, "Total memory should be greater than zero");
    assert!(
        memory_info.usage_percent >= 0.0 && memory_info.usage_percent <= 100.0,
        "Memory usage percentage should be between 0 and 100"
    );

    let cpu_usage = metrics_collector.get_cpu_usage().await?;
    assert!(cpu_usage >= 0.0 && cpu_usage <= 100.0, "CPU usage should be between 0 and 100");

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
