#![cfg(feature = "integration-tests")]
/// Integration test for the logging and debugging infrastructure
///
/// This test verifies that all components of the logging infrastructure work together:
/// - Structured logging with configurable levels
/// - Test execution tracing and debugging support  
/// - Error reporting with context and stack traces
/// - Performance monitoring and metrics collection
/// - Artifact collection for failed tests
use bitnet_tests::{
    config::TestConfig,
    errors::TestError,
    logging::{LoggingManager, MetricValue, TraceEventType},
};

#[tokio::test]
async fn test_logging_infrastructure_integration() {
    // Create test configuration
    let mut config = TestConfig::default();
    config.log_level = "debug".to_string();
    config.reporting.include_artifacts = true;

    // Initialize logging manager
    let logging_manager = LoggingManager::new(config).expect("Failed to create logging manager");

    // Test 1: Basic logging and tracing
    let debug_context = logging_manager.create_debug_context("integration_test".to_string());

    // Start tracing for this test
    debug_context.get_tracer().start_trace("integration_test").await;

    debug_context.trace(TraceEventType::TestStart, "Starting integration test".to_string()).await;
    debug_context.trace(TraceEventType::Info, "Testing basic functionality".to_string()).await;

    // Test 2: Performance profiling
    let mut profiler = logging_manager.start_profiling("test_operation".to_string()).await;
    profiler.add_metadata("test_type", "integration");

    // Simulate some work
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    profiler.finish().await;

    // Test 3: Metrics collection
    logging_manager.increment_counter("test_operations", 1).await;
    logging_manager.set_gauge("test_progress", 50.0).await;
    logging_manager.record_histogram("test_duration_ms", 10.0).await;
    logging_manager.record_metric("test_status", MetricValue::String("running".to_string())).await;

    // Test 4: Artifact collection
    let artifact =
        bitnet_tests::logging::DebugArtifact::text("test_log", "Integration test log data");
    debug_context.add_artifact(artifact).await;

    // Test 5: Error handling (simulate a failure)
    let test_error = TestError::execution("Simulated test error for integration testing");
    debug_context.add_error_context(&test_error, "This is a test error context".to_string()).await;

    // Verify traces were collected
    let traces = debug_context.get_tracer().get_traces("integration_test").await;
    assert!(!traces.is_empty(), "No traces were collected");
    assert!(traces.iter().any(|t| matches!(t.event_type, TraceEventType::TestStart)));

    // Verify artifacts were collected
    let artifacts = debug_context.get_artifacts().await;
    assert!(!artifacts.is_empty(), "No artifacts were collected");

    // Verify error contexts were collected
    let error_contexts = debug_context.get_error_contexts().await;
    assert!(!error_contexts.is_empty(), "No error contexts were collected");

    // Test 6: Get performance summary
    let performance_summary = logging_manager.get_performance_summary().await;
    assert!(performance_summary.total_operations > 0, "No performance operations recorded");

    // Test 7: Get metrics summary
    let metrics_summary = logging_manager.get_metrics_summary().await;
    assert!(metrics_summary.counters.contains_key("test_operations"), "Counter not recorded");
    assert!(metrics_summary.gauges.contains_key("test_progress"), "Gauge not recorded");
    assert!(
        metrics_summary.histogram_stats.contains_key("test_duration_ms"),
        "Histogram not recorded"
    );

    debug_context
        .trace(TraceEventType::TestEnd, "Integration test completed successfully".to_string())
        .await;

    println!("✅ Logging infrastructure integration test passed");
}

#[tokio::test]
async fn test_failure_artifact_collection() {
    let mut config = TestConfig::default();
    config.reporting.include_artifacts = true;

    // Use a temporary directory for this test
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    config.reporting.output_dir = temp_dir.path().to_path_buf();

    let logging_manager = LoggingManager::new(config).expect("Failed to create logging manager");
    let debug_context = logging_manager.create_debug_context("failure_test".to_string());

    // Start tracing for this test
    debug_context.get_tracer().start_trace("failure_test").await;

    // Add some context and artifacts before failure
    debug_context.trace(TraceEventType::TestStart, "Starting failure test".to_string()).await;

    let artifact =
        bitnet_tests::logging::DebugArtifact::text("pre_failure_log", "Log before failure");
    debug_context.add_artifact(artifact).await;

    // Simulate a failure
    let error = TestError::assertion("Test assertion failed");
    debug_context.add_error_context(&error, "Failure occurred during assertion".to_string()).await;

    // Handle the failure (this should collect artifacts)
    let result = logging_manager.handle_test_failure("failure_test", &error, &debug_context).await;
    assert!(result.is_ok(), "Failed to handle test failure: {:?}", result);

    // Verify that failure artifacts directory was created
    let failure_dir = temp_dir.path().join("failures").join("failure_test");
    assert!(failure_dir.exists(), "Failure artifacts directory was not created");

    // Verify that specific artifact files were created
    assert!(failure_dir.join("error_info.json").exists(), "Error info artifact not created");
    assert!(
        failure_dir.join("execution_trace.json").exists(),
        "Execution trace artifact not created"
    );
    assert!(
        failure_dir.join("error_contexts.json").exists(),
        "Error contexts artifact not created"
    );
    assert!(failure_dir.join("system_info.json").exists(), "System info artifact not created");

    println!("✅ Failure artifact collection test passed");
}

#[tokio::test]
async fn test_performance_monitoring() {
    let config = TestConfig::default();
    let logging_manager = LoggingManager::new(config).expect("Failed to create logging manager");

    // Run multiple operations with different durations
    for i in 0..5 {
        let mut profiler = logging_manager.start_profiling(format!("operation_{}", i)).await;
        profiler.add_metadata("iteration", &i.to_string());

        // Simulate work with varying duration
        tokio::time::sleep(std::time::Duration::from_millis(10 + i * 5)).await;

        profiler.finish().await;
    }

    // Get performance summary
    let summary = logging_manager.get_performance_summary().await;

    assert_eq!(summary.total_operations, 5, "Incorrect number of operations recorded");
    assert!(
        summary.total_duration > std::time::Duration::from_millis(50),
        "Total duration too short"
    );
    assert!(
        summary.average_duration > std::time::Duration::ZERO,
        "Average duration should be positive"
    );
    assert!(summary.min_duration <= summary.max_duration, "Min duration should be <= max duration");

    // Verify percentiles are calculated
    assert!(
        summary.performance_percentiles.p50 > std::time::Duration::ZERO,
        "P50 should be positive"
    );
    assert!(
        summary.performance_percentiles.p95 >= summary.performance_percentiles.p50,
        "P95 should be >= P50"
    );

    println!("✅ Performance monitoring test passed");
}

#[tokio::test]
async fn test_metrics_collection() {
    let config = TestConfig::default();
    let logging_manager = LoggingManager::new(config).expect("Failed to create logging manager");

    // Test counter metrics
    logging_manager.increment_counter("test_counter", 5).await;
    logging_manager.increment_counter("test_counter", 3).await; // Should accumulate to 8

    // Test gauge metrics
    logging_manager.set_gauge("test_gauge", 42.5).await;
    logging_manager.set_gauge("test_gauge", 37.2).await; // Should overwrite to 37.2

    // Test histogram metrics
    for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
        logging_manager.record_histogram("test_histogram", value).await;
    }

    // Test custom metrics
    logging_manager
        .record_metric("test_string", MetricValue::String("test_value".to_string()))
        .await;
    logging_manager.record_metric("test_boolean", MetricValue::Boolean(true)).await;
    logging_manager
        .record_metric("test_duration", MetricValue::Duration(std::time::Duration::from_secs(1)))
        .await;

    // Get metrics summary
    let summary = logging_manager.get_metrics_summary().await;

    // Verify counter
    assert_eq!(summary.counters.get("test_counter"), Some(&8), "Counter not accumulated correctly");

    // Verify gauge
    assert_eq!(summary.gauges.get("test_gauge"), Some(&37.2), "Gauge not set correctly");

    // Verify histogram
    let histogram_stats =
        summary.histogram_stats.get("test_histogram").expect("Histogram not found");
    assert_eq!(histogram_stats.count, 5, "Histogram count incorrect");
    assert_eq!(histogram_stats.mean, 3.0, "Histogram mean incorrect");
    assert_eq!(histogram_stats.min, 1.0, "Histogram min incorrect");
    assert_eq!(histogram_stats.max, 5.0, "Histogram max incorrect");

    // Verify custom metrics
    assert!(summary.custom_metrics.contains_key("test_string"), "Custom string metric not found");
    assert!(summary.custom_metrics.contains_key("test_boolean"), "Custom boolean metric not found");
    assert!(
        summary.custom_metrics.contains_key("test_duration"),
        "Custom duration metric not found"
    );

    println!("✅ Metrics collection test passed");
}
