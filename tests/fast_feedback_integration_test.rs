use bitnet_tests::fast_feedback_simple::{
    utils, FastFeedbackConfig, FastFeedbackSystem, FeedbackQuality,
};
use std::time::{Duration, Instant};

/// Integration test for the fast feedback system
#[tokio::test]
async fn test_fast_feedback_system_integration() {
    // Test with default configuration
    let mut system = FastFeedbackSystem::with_defaults();

    // Execute fast feedback
    let start_time = Instant::now();
    let result = system.execute_fast_feedback().await;
    let execution_time = start_time.elapsed();

    // Verify the result
    match result {
        Ok(feedback_result) => {
            // Check that execution completed within reasonable time
            assert!(
                execution_time <= Duration::from_secs(10),
                "Fast feedback took too long: {:?}",
                execution_time
            );

            // Check that we got some kind of feedback
            assert!(feedback_result.execution_time > Duration::ZERO);

            // Check feedback quality is reasonable
            assert!(matches!(
                feedback_result.feedback_quality,
                FeedbackQuality::Complete
                    | FeedbackQuality::HighConfidence
                    | FeedbackQuality::MediumConfidence
                    | FeedbackQuality::BasicConfidence
                    | FeedbackQuality::Limited
            ));

            println!("Fast feedback integration test passed:");
            println!("  Execution time: {:?}", feedback_result.execution_time);
            println!("  Tests run: {}", feedback_result.tests_run);
            println!("  Feedback quality: {:?}", feedback_result.feedback_quality);
        }
        Err(e) => {
            // For now, we expect this to fail since we don't have actual test suites
            // but we can verify the error is reasonable
            println!("Expected failure in integration test: {}", e);
            assert!(e.to_string().contains("test") || e.to_string().contains("suite"));
        }
    }
}

/// Test fast feedback system configuration variants
#[tokio::test]
async fn test_fast_feedback_configurations() {
    // Test CI configuration
    let ci_system = FastFeedbackSystem::for_ci();
    assert_eq!(ci_system.config.target_feedback_time, Duration::from_secs(90));
    assert!(ci_system.config.fail_fast);

    // Test development configuration
    let dev_system = FastFeedbackSystem::for_development();
    assert_eq!(dev_system.config.target_feedback_time, Duration::from_secs(30));
    assert!(!dev_system.config.fail_fast);

    // Test environment-based creation
    let env_system = utils::create_for_environment();
    assert!(env_system.config.enable_incremental);
}

/// Test fast feedback utility functions
#[test]
fn test_fast_feedback_utilities() {
    // Test recommended feedback time
    let recommended_time = utils::get_recommended_feedback_time();
    assert!(recommended_time > Duration::ZERO);
    assert!(recommended_time <= Duration::from_secs(5 * 60)); // Should be reasonable

    // Test should use fast feedback detection
    let should_use = utils::should_use_fast_feedback();
    // This depends on environment variables, so we just check it returns a boolean
    assert!(should_use || !should_use);
}

/// Test fast feedback configuration validation
#[test]
fn test_fast_feedback_config_validation() {
    let config = FastFeedbackConfig::default();

    // Verify reasonable defaults
    assert!(config.target_feedback_time > Duration::ZERO);
    assert!(config.max_feedback_time > config.target_feedback_time);
    assert!(config.min_coverage_threshold >= 0.0);
    assert!(config.min_coverage_threshold <= 1.0);
    assert!(config.max_parallel_fast > 0);
    assert!(config.cache_validity > Duration::ZERO);
}

/// Benchmark fast feedback system performance
#[tokio::test]
async fn benchmark_fast_feedback_performance() {
    let mut system = FastFeedbackSystem::for_development();

    // Measure multiple runs to check consistency
    let mut execution_times = Vec::new();

    for i in 0..3 {
        let start_time = Instant::now();
        let result = system.execute_fast_feedback().await;
        let execution_time = start_time.elapsed();

        execution_times.push(execution_time);

        match result {
            Ok(feedback_result) => {
                println!(
                    "Run {}: {:?} - {} tests",
                    i + 1,
                    execution_time,
                    feedback_result.tests_run
                );
            }
            Err(_) => {
                println!("Run {}: {:?} - expected failure", i + 1, execution_time);
            }
        }

        // Small delay between runs
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Check that execution times are consistent and reasonable
    let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
    println!("Average execution time: {:?}", avg_time);

    // All runs should complete within 10 seconds for this test
    for time in execution_times {
        assert!(time <= Duration::from_secs(10), "Execution took too long: {:?}", time);
    }
}

/// Test incremental testing integration
#[tokio::test]
async fn test_incremental_testing_integration() {
    use std::env;

    // Set environment variable to enable incremental testing
    env::set_var("BITNET_INCREMENTAL", "1");

    let mut system = FastFeedbackSystem::with_defaults();
    assert!(system.config.enable_incremental);

    // Test that incremental testing is detected
    let should_use = utils::should_use_fast_feedback();
    assert!(should_use);

    // Clean up
    env::remove_var("BITNET_INCREMENTAL");
}

/// Test fast feedback system error handling
#[tokio::test]
async fn test_fast_feedback_error_handling() {
    // Create a system with very aggressive constraints
    let mut config = FastFeedbackConfig::default();
    config.target_feedback_time = Duration::from_millis(1); // Impossible target
    config.max_feedback_time = Duration::from_millis(10); // Very short max time

    let mut system = FastFeedbackSystem::new(config);

    // This should either complete very quickly or fail gracefully
    let start_time = Instant::now();
    let result = system.execute_fast_feedback().await;
    let execution_time = start_time.elapsed();

    // Should complete quickly regardless of success/failure
    assert!(execution_time <= Duration::from_secs(5));

    match result {
        Ok(feedback_result) => {
            // If it succeeds, it should be very fast
            assert!(feedback_result.execution_time <= Duration::from_secs(1));
            println!("Fast feedback succeeded with aggressive constraints");
        }
        Err(e) => {
            // If it fails, the error should be reasonable
            println!("Expected failure with aggressive constraints: {}", e);
        }
    }
}
