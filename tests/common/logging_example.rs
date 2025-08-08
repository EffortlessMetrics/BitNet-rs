/// Example demonstrating the comprehensive logging and debugging infrastructure
///
/// This example shows how to use the logging framework for:
/// - Structured logging with configurable levels
/// - Test execution tracing and debugging support
/// - Error reporting with context and stack traces
/// - Performance monitoring and metrics collection
/// - Artifact collection for failed tests
use super::{
    config::TestConfig,
    errors::TestError,
    logging::{LoggingManager, MetricValue, TraceEventType},
};

/// Example test that demonstrates comprehensive logging
pub async fn example_comprehensive_test() -> Result<(), TestError> {
    // Initialize logging manager with default config
    let config = TestConfig::default();
    let logging_manager = LoggingManager::new(config)?;

    // Create debug context for this test
    let debug_context = logging_manager.create_debug_context("example_test".to_string());

    // Start tracing
    debug_context
        .trace(
            TraceEventType::TestStart,
            "Starting example test".to_string(),
        )
        .await;

    // Start performance profiling
    let mut profiler = logging_manager
        .start_profiling("example_operation".to_string())
        .await;
    profiler.add_metadata("operation_type", "example");
    profiler.add_metadata("complexity", "low");

    // Record some metrics
    logging_manager
        .increment_counter("operations_started", 1)
        .await;
    logging_manager.set_gauge("current_memory_mb", 128.5).await;

    // Simulate some work with tracing
    debug_context
        .trace(
            TraceEventType::Execution,
            "Performing main operation".to_string(),
        )
        .await;

    // Simulate a sub-operation with scoped context
    {
        let scoped_context = debug_context.scope("sub_operation");
        scoped_context
            .trace(TraceEventType::Info, "Starting sub-operation".to_string())
            .await;

        // Simulate work
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        scoped_context
            .trace(TraceEventType::Info, "Sub-operation completed".to_string())
            .await;
    }

    // Record histogram data
    logging_manager
        .record_histogram("operation_duration_ms", 50.0)
        .await;

    // Add some artifacts
    let artifact =
        super::logging::DebugArtifact::text("operation_log", "Operation completed successfully");
    debug_context.add_artifact(artifact).await;

    // Finish profiling
    profiler.finish().await;

    // Record success metrics
    logging_manager
        .increment_counter("operations_completed", 1)
        .await;
    logging_manager
        .record_metric(
            "operation_result",
            MetricValue::String("success".to_string()),
        )
        .await;

    debug_context
        .trace(
            TraceEventType::TestEnd,
            "Example test completed successfully".to_string(),
        )
        .await;

    println!("✅ Example test completed successfully");
    Ok(())
}

/// Example test that demonstrates failure handling
pub async fn example_failure_test() -> Result<(), TestError> {
    let config = TestConfig::default();
    let logging_manager = LoggingManager::new(config)?;

    let debug_context = logging_manager.create_debug_context("example_failure_test".to_string());

    debug_context
        .trace(
            TraceEventType::TestStart,
            "Starting failure example test".to_string(),
        )
        .await;

    // Start profiling
    let profiler = logging_manager
        .start_profiling("failing_operation".to_string())
        .await;

    // Simulate some work before failure
    debug_context
        .trace(
            TraceEventType::Execution,
            "Performing operation that will fail".to_string(),
        )
        .await;

    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    // Create an error
    let error = TestError::execution("Simulated operation failure");

    // Add error context
    debug_context
        .add_error_context(
            &error,
            "This is a simulated failure for demonstration".to_string(),
        )
        .await;

    // Add failure artifacts
    let error_artifact = super::logging::DebugArtifact::json(
        "error_details",
        &serde_json::json!({
            "error_type": "simulation",
            "expected": true,
            "details": "This failure is intentional for demonstration purposes"
        }),
    )?;
    debug_context.add_artifact(error_artifact).await;

    // Handle the failure
    logging_manager
        .handle_test_failure("example_failure_test", &error, &debug_context)
        .await?;

    profiler.finish().await;

    debug_context
        .trace(
            TraceEventType::TestEnd,
            "Failure test completed (with expected failure)".to_string(),
        )
        .await;

    println!("⚠️  Example failure test completed (failure was expected)");
    Err(error)
}

/// Example demonstrating metrics collection and reporting
pub async fn example_metrics_demo() -> Result<(), TestError> {
    let config = TestConfig::default();
    let logging_manager = LoggingManager::new(config)?;

    // Record various types of metrics
    for i in 0..10 {
        logging_manager
            .increment_counter("demo_operations", 1)
            .await;
        logging_manager
            .set_gauge("demo_progress", i as f64 * 10.0)
            .await;
        logging_manager
            .record_histogram("demo_latency_ms", (i * 5 + 10) as f64)
            .await;

        let profiler = logging_manager
            .start_profiling(format!("demo_operation_{}", i))
            .await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        profiler.finish().await;
    }

    // Get summaries
    let performance_summary = logging_manager.get_performance_summary().await;
    let metrics_summary = logging_manager.get_metrics_summary().await;

    println!("📊 Performance Summary:");
    println!(
        "  Total operations: {}",
        performance_summary.total_operations
    );
    println!("  Total duration: {:?}", performance_summary.total_duration);
    println!(
        "  Average duration: {:?}",
        performance_summary.average_duration
    );
    println!(
        "  Peak memory: {} bytes",
        performance_summary.memory_stats.peak_memory_usage
    );

    println!("📈 Metrics Summary:");
    println!("  Counters: {:?}", metrics_summary.counters);
    println!("  Gauges: {:?}", metrics_summary.gauges);

    if let Some(latency_stats) = metrics_summary.histogram_stats.get("demo_latency_ms") {
        println!(
            "  Latency stats: mean={:.2}ms, p95={:.2}ms",
            latency_stats.mean, latency_stats.p95
        );
    }

    Ok(())
}

/// Run all examples
pub async fn run_all_examples() {
    println!("🚀 Running logging infrastructure examples...\n");

    // Example 1: Successful test with comprehensive logging
    println!("1. Running comprehensive test example:");
    match example_comprehensive_test().await {
        Ok(_) => println!("   ✅ Comprehensive test example completed\n"),
        Err(e) => println!("   ❌ Comprehensive test example failed: {}\n", e),
    }

    // Example 2: Failure handling
    println!("2. Running failure handling example:");
    match example_failure_test().await {
        Ok(_) => println!("   ⚠️  Unexpected success in failure test\n"),
        Err(_) => println!("   ✅ Failure handling example completed (expected failure)\n"),
    }

    // Example 3: Metrics demonstration
    println!("3. Running metrics demonstration:");
    match example_metrics_demo().await {
        Ok(_) => println!("   ✅ Metrics demonstration completed\n"),
        Err(e) => println!("   ❌ Metrics demonstration failed: {}\n", e),
    }

    println!("🎉 All logging infrastructure examples completed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_example() {
        let result = example_comprehensive_test().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_failure_example() {
        let result = example_failure_test().await;
        assert!(result.is_err()); // This should fail as expected
    }

    #[tokio::test]
    async fn test_metrics_example() {
        let result = example_metrics_demo().await;
        assert!(result.is_ok());
    }
}
