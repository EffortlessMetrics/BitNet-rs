use async_trait::async_trait;
use bitnet_tests::common::{
    config::TestConfig,
    errors::{TestError, TestResult},
    fixtures::FixtureManager,
    harness::{TestCase, TestHarness, TestSuite},
    results::TestMetrics,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Simple test case that simulates work and verifies isolation
struct IsolatedTestCase {
    name: String,
    work_duration: Duration,
    should_fail: bool,
}

impl IsolatedTestCase {
    fn new(name: &str, work_duration: Duration, should_fail: bool) -> Self {
        Self { name: name.to_string(), work_duration, should_fail }
    }
}

#[async_trait]
impl TestCase for IsolatedTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        // Verify isolation by checking environment variables
        let test_name = std::env::var("BITNET_TEST_NAME").unwrap_or_default();
        let isolation_flag = std::env::var("BITNET_TEST_ISOLATION").unwrap_or_default();

        if test_name != self.name {
            return Err(TestError::setup(format!(
                "Test name mismatch: expected '{}', got '{}'",
                self.name, test_name
            )));
        }

        if isolation_flag != "true" {
            return Err(TestError::setup("Isolation flag not set".to_string()));
        }

        // Verify temp directory exists
        if std::env::var("BITNET_TEST_TEMP_DIR").is_err() {
            return Err(TestError::setup("Temp directory not set".to_string()));
        }

        println!("Setup completed for test: {}", self.name);
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start = Instant::now();

        // Simulate work
        sleep(self.work_duration).await;

        // Verify we're still in isolated environment
        let test_name = std::env::var("BITNET_TEST_NAME").unwrap_or_default();
        if test_name != self.name {
            return Err(TestError::execution(format!(
                "Isolation broken during execution: expected '{}', got '{}'",
                self.name, test_name
            )));
        }

        if self.should_fail {
            return Err(TestError::execution("Intentional test failure".to_string()));
        }

        let mut metrics = TestMetrics::with_duration(start.elapsed());
        metrics.add_operation();
        metrics.add_assertion();

        println!("Execution completed for test: {}", self.name);
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        // Verify we're still in isolated environment during cleanup
        let test_name = std::env::var("BITNET_TEST_NAME").unwrap_or_default();
        if test_name != self.name {
            return Err(TestError::cleanup(format!(
                "Isolation broken during cleanup: expected '{}', got '{}'",
                self.name, test_name
            )));
        }

        println!("Cleanup completed for test: {}", self.name);
        Ok(())
    }
}

/// Test suite for parallel execution testing
struct ParallelTestSuite {
    test_cases: Vec<Box<dyn TestCase>>,
}

impl ParallelTestSuite {
    fn new() -> Self {
        let mut test_cases: Vec<Box<dyn TestCase>> = Vec::new();

        // Create tests with different durations to verify parallel execution
        test_cases.push(Box::new(IsolatedTestCase::new(
            "fast_test_1",
            Duration::from_millis(100),
            false,
        )));
        test_cases.push(Box::new(IsolatedTestCase::new(
            "fast_test_2",
            Duration::from_millis(150),
            false,
        )));
        test_cases.push(Box::new(IsolatedTestCase::new(
            "medium_test_1",
            Duration::from_millis(300),
            false,
        )));
        test_cases.push(Box::new(IsolatedTestCase::new(
            "medium_test_2",
            Duration::from_millis(250),
            false,
        )));
        test_cases.push(Box::new(IsolatedTestCase::new(
            "slow_test_1",
            Duration::from_millis(500),
            false,
        )));
        test_cases.push(Box::new(IsolatedTestCase::new(
            "failing_test",
            Duration::from_millis(100),
            true,
        )));

        Self { test_cases }
    }
}

impl TestSuite for ParallelTestSuite {
    fn name(&self) -> &str {
        "parallel_execution_test_suite"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        self.test_cases
            .iter()
            .map(|tc| {
                // This is a bit tricky - we need to clone the test cases
                // For now, let's create new instances
                match tc.name() {
                    "fast_test_1" => Box::new(IsolatedTestCase::new(
                        "fast_test_1",
                        Duration::from_millis(100),
                        false,
                    )) as Box<dyn TestCase>,
                    "fast_test_2" => Box::new(IsolatedTestCase::new(
                        "fast_test_2",
                        Duration::from_millis(150),
                        false,
                    )) as Box<dyn TestCase>,
                    "medium_test_1" => Box::new(IsolatedTestCase::new(
                        "medium_test_1",
                        Duration::from_millis(300),
                        false,
                    )) as Box<dyn TestCase>,
                    "medium_test_2" => Box::new(IsolatedTestCase::new(
                        "medium_test_2",
                        Duration::from_millis(250),
                        false,
                    )) as Box<dyn TestCase>,
                    "slow_test_1" => Box::new(IsolatedTestCase::new(
                        "slow_test_1",
                        Duration::from_millis(500),
                        false,
                    )) as Box<dyn TestCase>,
                    "failing_test" => Box::new(IsolatedTestCase::new(
                        "failing_test",
                        Duration::from_millis(100),
                        true,
                    )) as Box<dyn TestCase>,
                    _ => Box::new(IsolatedTestCase::new(
                        "unknown",
                        Duration::from_millis(100),
                        false,
                    )) as Box<dyn TestCase>,
                }
            })
            .collect()
    }
}

#[tokio::test]
async fn test_parallel_execution_with_isolation() {
    // Create test configuration with parallel execution
    let config = TestConfig {
        max_parallel_tests: 3, // Allow 3 tests to run in parallel
        test_timeout: Duration::from_secs(5),
        log_level: "info".to_string(),
        cache_dir: std::path::PathBuf::from("test_cache"),
        fixtures: Default::default(),
    };

    // Create test harness
    let mut harness = TestHarness::new(config).await.expect("Failed to create test harness");

    // Add console reporter
    harness.add_reporter(bitnet_tests::common::harness::ConsoleReporter::new(true));

    // Create test suite
    let suite = ParallelTestSuite::new();

    // Record start time to verify parallel execution
    let start_time = Instant::now();

    // Run the test suite
    let result = harness.run_test_suite(&suite).await.expect("Test suite execution failed");

    let total_duration = start_time.elapsed();

    // Verify results
    assert_eq!(result.test_results.len(), 6, "Should have 6 test results");

    // Count passed and failed tests
    let passed_count = result.test_results.iter().filter(|r| r.is_success()).count();
    let failed_count = result.test_results.iter().filter(|r| r.is_failure()).count();

    assert_eq!(passed_count, 5, "Should have 5 passed tests");
    assert_eq!(failed_count, 1, "Should have 1 failed test");

    // Verify parallel execution efficiency
    // If tests ran sequentially, total time would be ~1400ms
    // With 3 parallel slots, it should be much faster
    let sequential_time = Duration::from_millis(100 + 150 + 300 + 250 + 500 + 100); // 1400ms
    let expected_parallel_time = Duration::from_millis(700); // Rough estimate with 3 parallel slots

    println!("Total execution time: {:?}", total_duration);
    println!("Sequential time would be: {:?}", sequential_time);
    println!("Expected parallel time: ~{:?}", expected_parallel_time);

    // Verify that parallel execution was faster than sequential
    assert!(
        total_duration < sequential_time,
        "Parallel execution should be faster than sequential. Got {:?}, expected less than {:?}",
        total_duration,
        sequential_time
    );

    // Verify that we achieved some parallelism (should be significantly faster)
    let efficiency_threshold = sequential_time.as_millis() as f64 * 0.7; // At least 30% improvement
    assert!(
        (total_duration.as_millis() as f64) < efficiency_threshold,
        "Parallel execution should show significant improvement. Got {:?}, expected less than {}ms",
        total_duration,
        efficiency_threshold
    );

    println!("✅ Parallel execution with isolation test passed!");
    println!("   - All tests executed with proper isolation");
    println!(
        "   - Parallel execution achieved {:.1}% efficiency",
        (sequential_time.as_secs_f64() / total_duration.as_secs_f64() - 1.0) * 100.0
    );
}

#[tokio::test]
async fn test_semaphore_limits_parallelism() {
    // Create test configuration with limited parallelism
    let config = TestConfig {
        max_parallel_tests: 1, // Only allow 1 test at a time
        test_timeout: Duration::from_secs(5),
        log_level: "info".to_string(),
        cache_dir: std::path::PathBuf::from("test_cache"),
        fixtures: Default::default(),
    };

    // Create test harness
    let mut harness = TestHarness::new(config).await.expect("Failed to create test harness");

    // Add console reporter
    harness.add_reporter(bitnet_tests::common::harness::ConsoleReporter::new(false));

    // Create a simple test suite with 3 tests
    let mut test_cases: Vec<Box<dyn TestCase>> = Vec::new();
    test_cases.push(Box::new(IsolatedTestCase::new(
        "sequential_1",
        Duration::from_millis(200),
        false,
    )));
    test_cases.push(Box::new(IsolatedTestCase::new(
        "sequential_2",
        Duration::from_millis(200),
        false,
    )));
    test_cases.push(Box::new(IsolatedTestCase::new(
        "sequential_3",
        Duration::from_millis(200),
        false,
    )));

    struct SequentialTestSuite {
        test_cases: Vec<Box<dyn TestCase>>,
    }

    impl TestSuite for SequentialTestSuite {
        fn name(&self) -> &str {
            "sequential_test_suite"
        }

        fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
            vec![
                Box::new(IsolatedTestCase::new("sequential_1", Duration::from_millis(200), false)),
                Box::new(IsolatedTestCase::new("sequential_2", Duration::from_millis(200), false)),
                Box::new(IsolatedTestCase::new("sequential_3", Duration::from_millis(200), false)),
            ]
        }
    }

    let suite = SequentialTestSuite { test_cases };

    // Record start time
    let start_time = Instant::now();

    // Run the test suite
    let result = harness.run_test_suite(&suite).await.expect("Test suite execution failed");

    let total_duration = start_time.elapsed();

    // Verify results
    assert_eq!(result.test_results.len(), 3, "Should have 3 test results");
    assert_eq!(result.summary.passed, 3, "All tests should pass");

    // With max_parallel_tests = 1, execution should be close to sequential
    let expected_min_time = Duration::from_millis(600); // 3 * 200ms
    assert!(
        total_duration >= expected_min_time.mul_f64(0.8),
        "Sequential execution should take at least ~{:?}, got {:?}",
        expected_min_time,
        total_duration
    );

    println!("✅ Semaphore limiting test passed!");
    println!("   - Execution time: {:?} (expected ~{:?})", total_duration, expected_min_time);
}
