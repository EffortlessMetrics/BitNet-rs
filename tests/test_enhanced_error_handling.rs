/// Test demonstrating enhanced error handling with actionable debugging information
///
/// This test shows how the enhanced error handler provides:
/// - Comprehensive error analysis
/// - Actionable debugging recommendations
/// - Root cause analysis
/// - Recovery suggestions
/// - Troubleshooting guides
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bitnet_tests::{
    config::TestConfig,
    enhanced_error_handler::{EnhancedErrorHandler, ErrorHandlerConfig, TestExecutionContext},
    errors::TestError,
    logging::LoggingManager,
};

#[tokio::test]
async fn test_enhanced_error_handling_timeout() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging and error handler
    let config = TestConfig::default();
    let logging_manager = Arc::new(LoggingManager::new(config)?);
    let error_handler_config = ErrorHandlerConfig {
        save_analysis_reports: true,
        log_debugging_guides: true,
        enable_retries: true,
        max_retry_attempts: 2,
        ..Default::default()
    };
    let error_handler =
        EnhancedErrorHandler::new(Arc::clone(&logging_manager), error_handler_config);

    // Create a debug context for the test
    let debug_context = logging_manager.create_debug_context("test_timeout_scenario".to_string());

    // Simulate a timeout error
    let timeout_error = TestError::timeout(Duration::from_secs(30));

    // Create execution context
    let execution_context = TestExecutionContext {
        test_suite: "enhanced_error_handling_tests".to_string(),
        execution_time: Duration::from_secs(35),
        concurrent_tests: 4,
        retry_count: 0,
        test_metadata: {
            let mut metadata = HashMap::new();
            metadata.insert("test_type".to_string(), "timeout_simulation".to_string());
            metadata.insert("expected_duration".to_string(), "30s".to_string());
            metadata
        },
    };

    // Handle the error with comprehensive analysis
    let result = error_handler
        .handle_test_error(
            "test_timeout_scenario",
            &timeout_error,
            &debug_context,
            execution_context,
        )
        .await?;

    // Verify error analysis results
    assert_eq!(result.analysis.error_category, "timeout");
    assert!(result.should_retry, "Timeout errors should be retryable");
    assert!(result.retry_delay.is_some(), "Should provide retry delay");
    assert!(!result.recovery_suggestions.is_empty(), "Should provide recovery suggestions");
    assert!(!result.troubleshooting_steps.is_empty(), "Should provide troubleshooting steps");

    // Verify debugging guide is comprehensive
    let debugging_guide = &result.debugging_guide;
    assert!(debugging_guide.contains("DEBUGGING GUIDE"), "Should contain debugging guide header");
    assert!(debugging_guide.contains("RECOMMENDED ACTIONS"), "Should contain recommendations");

    // Print debugging information for manual verification
    println!("=== ENHANCED ERROR HANDLING DEMONSTRATION ===");
    println!("Error Summary: {}", result.generate_summary());
    println!("\nDebugging Guide:\n{}", debugging_guide);

    if let Some(primary_rec) = result.primary_recommendation() {
        println!("\nPrimary Recommendation: {} - {}", primary_rec.title, primary_rec.description);
        println!("Success Probability: {}%", (primary_rec.success_probability * 100.0) as u32);
    }

    println!("\nRecovery Suggestions:");
    for (i, suggestion) in result.recovery_suggestions.iter().enumerate() {
        println!("  {}. {}", i + 1, suggestion);
    }

    println!("\nTroubleshooting Steps:");
    for step in &result.troubleshooting_steps {
        println!("  {}. {} - {}", step.step_number, step.title, step.description);
    }

    Ok(())
}

#[tokio::test]
async fn test_enhanced_error_handling_fixture_failure() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let config = TestConfig::default();
    let logging_manager = Arc::new(LoggingManager::new(config)?);
    let error_handler =
        EnhancedErrorHandler::new(Arc::clone(&logging_manager), ErrorHandlerConfig::default());

    let debug_context = logging_manager.create_debug_context("test_fixture_failure".to_string());

    // Simulate a fixture download error
    let fixture_error = TestError::FixtureError(bitnet_tests::errors::FixtureError::download(
        "https://example.com/test-model.gguf",
        "Connection timeout",
    ));

    let execution_context = TestExecutionContext {
        test_suite: "fixture_tests".to_string(),
        execution_time: Duration::from_secs(10),
        concurrent_tests: 1,
        retry_count: 0,
        test_metadata: HashMap::new(),
    };

    // Handle the fixture error
    let result = error_handler
        .handle_test_error(
            "test_fixture_failure",
            &fixture_error,
            &debug_context,
            execution_context,
        )
        .await?;

    // Verify fixture-specific analysis
    assert_eq!(result.analysis.error_category, "fixture");
    assert!(result.should_retry, "Fixture errors should be retryable");

    // Check for fixture-specific recommendations
    let has_network_recommendation = result.analysis.recommendations.iter().any(|rec| {
        rec.description.to_lowercase().contains("network")
            || rec.description.to_lowercase().contains("connectivity")
    });

    assert!(
        has_network_recommendation,
        "Should provide network-related recommendations for fixture errors"
    );

    println!("\n=== FIXTURE ERROR HANDLING DEMONSTRATION ===");
    println!("Error Analysis Confidence: {:.1}%", result.analysis.confidence_score * 100.0);
    println!("Debugging Priority: {}", result.analysis.debugging_priority);

    if let Some(primary_cause) = &result.analysis.root_cause_analysis.primary_cause {
        println!(
            "Primary Cause: {} ({}% likelihood)",
            primary_cause.cause,
            (primary_cause.likelihood * 100.0) as u32
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_enhanced_error_handling_assertion_failure() -> Result<(), Box<dyn std::error::Error>>
{
    // Initialize components
    let config = TestConfig::default();
    let logging_manager = Arc::new(LoggingManager::new(config)?);
    let error_handler =
        EnhancedErrorHandler::new(Arc::clone(&logging_manager), ErrorHandlerConfig::default());

    let debug_context = logging_manager.create_debug_context("test_assertion_failure".to_string());

    // Simulate an assertion error
    let assertion_error = TestError::assertion(
        "Expected output tokens [1, 2, 3] but got [1, 2, 4] - mismatch at position 2",
    );

    let execution_context = TestExecutionContext {
        test_suite: "inference_tests".to_string(),
        execution_time: Duration::from_secs(5),
        concurrent_tests: 1,
        retry_count: 0,
        test_metadata: HashMap::new(),
    };

    // Handle the assertion error
    let result = error_handler
        .handle_test_error(
            "test_assertion_failure",
            &assertion_error,
            &debug_context,
            execution_context,
        )
        .await?;

    // Verify assertion-specific analysis
    assert_eq!(result.analysis.error_category, "assertion");
    assert!(!result.should_retry, "Assertion errors should not be retryable");
    assert!(result.retry_delay.is_none(), "Should not provide retry delay for assertion errors");

    // Check for assertion-specific recommendations
    let has_debug_recommendation = result.analysis.recommendations.iter().any(|rec| {
        rec.description.to_lowercase().contains("debug")
            || rec.description.to_lowercase().contains("values")
    });

    assert!(
        has_debug_recommendation,
        "Should provide debugging recommendations for assertion errors"
    );

    println!("\n=== ASSERTION ERROR HANDLING DEMONSTRATION ===");
    println!("Should Block Suite: {}", result.should_block_suite());

    // Show troubleshooting steps specific to assertion failures
    println!("Assertion-Specific Troubleshooting:");
    for step in result.troubleshooting_steps.iter().take(3) {
        println!("  Step {}: {}", step.step_number, step.description);
    }

    Ok(())
}

#[tokio::test]
async fn test_error_summary_reporting() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize components
    let config = TestConfig::default();
    let logging_manager = Arc::new(LoggingManager::new(config)?);
    let error_handler =
        EnhancedErrorHandler::new(Arc::clone(&logging_manager), ErrorHandlerConfig::default());

    let debug_context = logging_manager.create_debug_context("test_summary_reporting".to_string());

    // Simulate multiple different errors
    let errors = vec![
        TestError::timeout(Duration::from_secs(30)),
        TestError::assertion("Test assertion failed"),
        TestError::FixtureError(bitnet_tests::errors::FixtureError::not_found("model.gguf")),
        TestError::config("Invalid configuration parameter"),
    ];

    let execution_context = TestExecutionContext {
        test_suite: "summary_tests".to_string(),
        execution_time: Duration::from_secs(10),
        concurrent_tests: 1,
        retry_count: 0,
        test_metadata: HashMap::new(),
    };

    // Handle each error
    for (i, error) in errors.iter().enumerate() {
        let test_name = format!("test_error_{}", i);
        let _result = error_handler
            .handle_test_error(&test_name, error, &debug_context, execution_context.clone())
            .await?;
    }

    // Generate error summary report
    let summary = error_handler.generate_error_summary().await;

    // Verify summary statistics
    assert_eq!(summary.total_errors, 4, "Should track all handled errors");
    assert!(summary.errors_by_category.len() > 1, "Should categorize errors");
    assert!(!summary.most_common_category.is_empty(), "Should identify most common category");

    println!("\n=== ERROR SUMMARY REPORT ===");
    println!("{}", summary.generate_summary());
    println!("Is Concerning: {}", summary.is_concerning());

    println!("\nErrors by Category:");
    for (category, count) in &summary.errors_by_category {
        println!("  {}: {}", category, count);
    }

    println!("\nErrors by Severity:");
    for (severity, count) in &summary.errors_by_severity {
        println!("  {}: {}", severity, count);
    }

    Ok(())
}

#[tokio::test]
async fn test_error_pattern_detection() -> Result<(), Box<dyn std::error::Error>> {
    // This test demonstrates how the error analyzer detects patterns
    // and provides more targeted recommendations based on those patterns

    let config = TestConfig::default();
    let logging_manager = Arc::new(LoggingManager::new(config)?);
    let error_handler =
        EnhancedErrorHandler::new(Arc::clone(&logging_manager), ErrorHandlerConfig::default());

    let debug_context = logging_manager.create_debug_context("test_pattern_detection".to_string());

    // Simulate a timeout error in CI environment (should trigger CI pattern)
    std::env::set_var("CI", "true");

    let timeout_error = TestError::timeout(Duration::from_secs(60));
    let execution_context = TestExecutionContext {
        test_suite: "ci_tests".to_string(),
        execution_time: Duration::from_secs(65),
        concurrent_tests: 8, // High concurrency typical in CI
        retry_count: 0,
        test_metadata: HashMap::new(),
    };

    let result = error_handler
        .handle_test_error("test_ci_timeout", &timeout_error, &debug_context, execution_context)
        .await?;

    // Verify pattern detection
    assert!(!result.analysis.detected_patterns.is_empty(), "Should detect patterns");

    // Check for CI-specific recommendations
    let has_ci_recommendation = result.analysis.recommendations.iter().any(|rec| {
        rec.category.to_lowercase().contains("ci") || rec.description.to_lowercase().contains("ci")
    });

    println!("\n=== PATTERN DETECTION DEMONSTRATION ===");
    println!("Detected {} patterns", result.analysis.detected_patterns.len());

    for pattern in &result.analysis.detected_patterns {
        println!(
            "Pattern: {} (confidence: {:.1}%)",
            pattern.pattern.name,
            pattern.confidence * 100.0
        );
        for evidence in &pattern.evidence {
            println!("  Evidence: {}", evidence);
        }
    }

    // Clean up environment variable
    std::env::remove_var("CI");

    Ok(())
}
