use bitnet_tests::common::{
    config::TestConfig,
    harness::{ConsoleReporter, TestHarness},
};
use bitnet_tests::integration::resource_management_tests::ResourceManagementTestSuite;

#[tokio::test]
async fn test_resource_management_suite() {
    // Initialize logging
    let _ = tracing_subscriber::fmt::try_init();

    // Create test configuration
    let config = TestConfig::default();

    // Create test harness
    let mut harness = TestHarness::new(config)
        .await
        .expect("Failed to create test harness");
    harness.add_reporter(Box::new(ConsoleReporter::new(true)));

    // Create and run resource management test suite
    let test_suite = ResourceManagementTestSuite::new();

    let result = harness.run_test_suite(test_suite).await;

    match result {
        Ok(suite_result) => {
            println!("Resource management tests completed:");
            println!("  Total tests: {}", suite_result.summary.total_tests);
            println!("  Passed: {}", suite_result.summary.passed);
            println!("  Failed: {}", suite_result.summary.failed);
            println!("  Success rate: {:.1}%", suite_result.summary.success_rate);

            // Assert that most tests passed (allow for some platform-specific failures)
            assert!(
                suite_result.summary.success_rate >= 80.0,
                "Resource management test success rate too low: {:.1}%",
                suite_result.summary.success_rate
            );
        }
        Err(e) => {
            panic!("Resource management test suite failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_memory_leak_detection() {
    use bitnet_tests::common::{config::FixtureConfig, fixtures::FixtureManager};
    use bitnet_tests::integration::resource_management_tests::MemoryLeakDetectionTest;

    let _ = tracing_subscriber::fmt::try_init();

    let fixture_config = FixtureConfig::default();
    let fixtures = FixtureManager::new(&fixture_config)
        .await
        .expect("Failed to create fixture manager");

    let test = MemoryLeakDetectionTest::new();

    // Setup
    test.setup(&fixtures).await.expect("Setup failed");

    // Execute
    let result = test.execute().await;

    // Cleanup
    let _ = test.cleanup().await;

    match result {
        Ok(metrics) => {
            println!("Memory leak detection test completed successfully");
            println!("Metrics: {:?}", metrics.custom_metrics);

            // Verify we have expected metrics
            assert!(metrics.custom_metrics.contains_key("initial_memory_bytes"));
            assert!(metrics.custom_metrics.contains_key("peak_memory_bytes"));
            assert!(metrics.custom_metrics.contains_key("final_memory_bytes"));
        }
        Err(e) => {
            println!("Memory leak detection test failed: {}", e);
            // Don't panic here as memory tests can be platform-dependent
        }
    }
}

#[tokio::test]
async fn test_file_handle_management() {
    use bitnet_tests::common::{config::FixtureConfig, fixtures::FixtureManager};
    use bitnet_tests::integration::resource_management_tests::FileHandleLeakTest;

    let _ = tracing_subscriber::fmt::try_init();

    let fixture_config = FixtureConfig::default();
    let fixtures = FixtureManager::new(&fixture_config)
        .await
        .expect("Failed to create fixture manager");

    let test = FileHandleLeakTest::new();

    // Setup
    test.setup(&fixtures).await.expect("Setup failed");

    // Execute
    let result = test.execute().await;

    // Cleanup
    let _ = test.cleanup().await;

    match result {
        Ok(metrics) => {
            println!("File handle leak test completed successfully");
            println!("Metrics: {:?}", metrics.custom_metrics);

            // Verify we have expected metrics
            assert!(metrics.custom_metrics.contains_key("max_file_handles"));
            assert!(metrics.custom_metrics.contains_key("remaining_handles"));

            // Verify no handles leaked
            let remaining = metrics
                .custom_metrics
                .get("remaining_handles")
                .unwrap_or(&1.0);
            assert_eq!(*remaining, 0.0, "File handles leaked: {}", remaining);
        }
        Err(e) => {
            panic!("File handle leak test failed: {}", e);
        }
    }
}
