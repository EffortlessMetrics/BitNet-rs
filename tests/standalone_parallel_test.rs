#![cfg(feature = "integration-tests")]
// Standalone test for parallel execution with proper isolation
// This test doesn't depend on the existing test framework to avoid compilation issues

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tempfile;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{sleep, timeout};

/// Simple test result
#[derive(Debug, Clone)]
pub struct SimpleTestResult {
    pub test_name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error: Option<String>,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
}

impl SimpleTestResult {
    pub fn passed(test_name: String, duration: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            test_name,
            passed: true,
            duration,
            error: None,
            start_time: now - duration,
            end_time: now,
        }
    }

    pub fn failed(test_name: String, duration: Duration, error: String) -> Self {
        let now = SystemTime::now();
        Self {
            test_name,
            passed: false,
            duration,
            error: Some(error),
            start_time: now - duration,
            end_time: now,
        }
    }
}

/// Isolated environment for test execution
struct IsolatedEnvironment {
    test_name: String,
    temp_dir: tempfile::TempDir,
    original_env_vars: HashMap<String, String>,
}

impl IsolatedEnvironment {
    fn new(test_name: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let temp_dir = tempfile::tempdir()?;
        let env_vars = std::env::vars().collect();

        Ok(Self { test_name: test_name.to_string(), temp_dir, original_env_vars: env_vars })
    }

    fn setup(&self) {
        // Set test-specific environment variables
        unsafe {
            std::env::set_var("BITNET_TEST_NAME", &self.test_name);
        }
        unsafe {
            std::env::set_var("BITNET_TEST_TEMP_DIR", self.temp_dir.path());
        }
        unsafe {
            std::env::set_var("BITNET_TEST_ISOLATION", "true");
        }
        unsafe {
            std::env::set_var("BITNET_TEST_ID", format!("{}", std::process::id()));
        }
    }

    fn cleanup(self) {
        // Restore original environment variables
        for (key, value) in &self.original_env_vars {
            unsafe {
                std::env::set_var(key, value);
            }
        }

        // Remove test-specific environment variables
        unsafe {
            std::env::remove_var("BITNET_TEST_NAME");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_TEMP_DIR");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_ISOLATION");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_ID");
        }

        // Temp directory is automatically cleaned up when dropped
        drop(self.temp_dir);
    }
}

/// Simple test case trait
trait SimpleTestCase: Send + Sync {
    fn name(&self) -> &str;
    async fn run(&self) -> Result<(), String>;
}

/// Test case that verifies isolation
struct IsolationTestCase {
    name: String,
    work_duration: Duration,
    should_fail: bool,
    unique_value: String,
}

impl IsolationTestCase {
    fn new(name: &str, work_duration: Duration, should_fail: bool) -> Self {
        Self {
            name: name.to_string(),
            work_duration,
            should_fail,
            unique_value: format!("{}_{}", name, std::process::id()),
        }
    }
}

impl SimpleTestCase for IsolationTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    async fn run(&self) -> Result<(), String> {
        // Verify isolation setup
        let test_name = std::env::var("BITNET_TEST_NAME").map_err(|_| "Test name not set")?;
        let isolation_flag =
            std::env::var("BITNET_TEST_ISOLATION").map_err(|_| "Isolation flag not set")?;
        let temp_dir = std::env::var("BITNET_TEST_TEMP_DIR").map_err(|_| "Temp dir not set")?;
        let test_id = std::env::var("BITNET_TEST_ID").map_err(|_| "Test ID not set")?;

        if test_name != self.name {
            return Err(format!(
                "Test name mismatch: expected '{}', got '{}'",
                self.name, test_name
            ));
        }

        if isolation_flag != "true" {
            return Err("Isolation flag not set to true".to_string());
        }

        if !std::path::Path::new(&temp_dir).exists() {
            return Err("Temp directory does not exist".to_string());
        }

        // Set a unique environment variable for this test
        unsafe {
            std::env::set_var("UNIQUE_TEST_VAR", &self.unique_value);
        }

        // Create a unique file in the temp directory
        let unique_file = std::path::Path::new(&temp_dir).join(format!("{}.txt", self.name));
        std::fs::write(&unique_file, &self.unique_value)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        // Simulate work
        sleep(self.work_duration).await;

        // Verify isolation is still intact after work
        let test_name_after =
            std::env::var("BITNET_TEST_NAME").map_err(|_| "Test name lost during execution")?;
        if test_name_after != self.name {
            return Err(format!(
                "Isolation broken during execution: expected '{}', got '{}'",
                self.name, test_name_after
            ));
        }

        // Verify our unique environment variable is still correct
        let unique_var_after =
            std::env::var("UNIQUE_TEST_VAR").map_err(|_| "Unique var lost during execution")?;
        if unique_var_after != self.unique_value {
            return Err(format!(
                "Environment interference detected: expected '{}', got '{}'",
                self.unique_value, unique_var_after
            ));
        }

        // Verify our unique file still exists and has correct content
        let file_content = std::fs::read_to_string(&unique_file)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        if file_content != self.unique_value {
            return Err(format!(
                "File interference detected: expected '{}', got '{}'",
                self.unique_value, file_content
            ));
        }

        if self.should_fail {
            return Err("Intentional test failure".to_string());
        }

        Ok(())
    }
}

/// Simple parallel test harness
pub struct SimpleParallelHarness {
    max_parallel: usize,
    test_timeout: Duration,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<TestStats>>,
}

#[derive(Debug, Default)]
struct TestStats {
    total_tests: usize,
    passed_tests: usize,
    failed_tests: usize,
    total_duration: Duration,
}

impl SimpleParallelHarness {
    pub fn new(max_parallel: usize, test_timeout: Duration) -> Self {
        Self {
            max_parallel,
            test_timeout,
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            stats: Arc::new(RwLock::new(TestStats::default())),
        }
    }

    /// Run tests in parallel with proper isolation
    pub async fn run_tests(
        &self,
        test_cases: Vec<Box<dyn SimpleTestCase>>,
    ) -> Vec<SimpleTestResult> {
        let mut handles = Vec::new();

        for test_case in test_cases {
            let semaphore = Arc::clone(&self.semaphore);
            let stats = Arc::clone(&self.stats);
            let timeout_duration = self.test_timeout;

            let handle = tokio::spawn(async move {
                Self::run_single_test_isolated(test_case, semaphore, stats, timeout_duration).await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Task join error: {}", e);
                    results.push(SimpleTestRecord::failed(
                        "unknown".to_string(),
                        Duration::ZERO,
                        format!("Task join error: {}", e),
                    ));
                }
            }
        }

        results
    }

    async fn run_single_test_isolated(
        test_case: Box<dyn SimpleTestCase>,
        semaphore: Arc<Semaphore>,
        stats: Arc<RwLock<TestStats>>,
        timeout_duration: Duration,
    ) -> SimpleTestResult {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();

        // Acquire semaphore permit for parallel execution control
        let _permit = match semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                return SimpleTestRecord::failed(
                    test_name,
                    Duration::ZERO,
                    format!("Failed to acquire semaphore: {}", e),
                );
            }
        };

        // Create isolated environment
        let isolated_env = match IsolatedEnvironment::new(&test_name) {
            Ok(env) => env,
            Err(e) => {
                return SimpleTestRecord::failed(
                    test_name,
                    start_time.elapsed(),
                    format!("Failed to create isolated environment: {}", e),
                );
            }
        };

        // Setup isolation
        isolated_env.setup();

        // Execute test with timeout
        let execute_result = timeout(timeout_duration, test_case.run()).await;
        let duration = start_time.elapsed();

        // Cleanup isolation
        isolated_env.cleanup();

        // Update stats
        {
            let mut stats_guard = stats.write().await;
            stats_guard.total_tests += 1;
            stats_guard.total_duration += duration;
        }

        // Process result
        match execute_result {
            Ok(Ok(())) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.passed_tests += 1;
                }
                SimpleTestRecord::passed(test_name, duration)
            }
            Ok(Err(e)) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.failed_tests += 1;
                }
                SimpleTestRecord::failed(test_name, duration, e)
            }
            Err(_) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.failed_tests += 1;
                }
                SimpleTestRecord::failed(
                    test_name,
                    timeout_duration,
                    format!("Test timed out after {:?}", timeout_duration),
                )
            }
        }
    }

    pub async fn get_stats(&self) -> TestStats {
        self.stats.read().await.clone()
    }
}

#[tokio::test]
async fn test_parallel_execution_with_proper_isolation() {
    println!("🚀 Testing parallel execution with proper isolation...");

    // Create test harness with 3 parallel slots
    let harness = SimpleParallelHarness::new(3, Duration::from_secs(10));

    // Create test cases with different durations
    let mut test_cases: Vec<Box<dyn SimpleTestCase>> = Vec::new();
    test_cases.push(Box::new(IsolationTestCase::new(
        "fast_test_1",
        Duration::from_millis(100),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "fast_test_2",
        Duration::from_millis(150),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "medium_test_1",
        Duration::from_millis(300),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "medium_test_2",
        Duration::from_millis(250),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "slow_test_1",
        Duration::from_millis(500),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "failing_test",
        Duration::from_millis(100),
        true,
    )));

    // Record start time to verify parallel execution
    let start_time = Instant::now();

    // Run tests
    let results = harness.run_tests(test_cases).await;
    let total_duration = start_time.elapsed();

    // Print results
    println!("📊 Test Results:");
    for result in &results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {} ({:?})", status, result.test_name, result.duration);
        if let Some(error) = &result.error {
            println!("    Error: {}", error);
        }
    }

    // Verify results
    assert_eq!(results.len(), 6, "Should have 6 test results");

    let passed_count = results.iter().filter(|r| r.passed).count();
    let failed_count = results.iter().filter(|r| !r.passed).count();

    assert_eq!(passed_count, 5, "Should have 5 passed tests");
    assert_eq!(failed_count, 1, "Should have 1 failed test");

    // Verify parallel execution efficiency
    let sequential_time = Duration::from_millis(100 + 150 + 300 + 250 + 500 + 100); // 1400ms

    println!("⏱️  Performance Analysis:");
    println!("  Total execution time: {:?}", total_duration);
    println!("  Sequential time would be: {:?}", sequential_time);

    // Verify that parallel execution was faster than sequential
    assert!(
        total_duration < sequential_time,
        "Parallel execution should be faster than sequential. Got {:?}, expected less than {:?}",
        total_duration,
        sequential_time
    );

    // Verify significant parallelism (should be at least 30% faster)
    let efficiency_threshold = sequential_time.as_millis() as f64 * 0.7;
    assert!(
        (total_duration.as_millis() as f64) < efficiency_threshold,
        "Parallel execution should show significant improvement. Got {:?}, expected less than {}ms",
        total_duration,
        efficiency_threshold
    );

    // Verify stats
    let stats = harness.get_stats().await;
    assert_eq!(stats.total_tests, 6);
    assert_eq!(stats.passed_tests, 5);
    assert_eq!(stats.failed_tests, 1);

    let efficiency = (sequential_time.as_secs_f64() / total_duration.as_secs_f64() - 1.0) * 100.0;
    println!("  Parallel efficiency: {:.1}%", efficiency);

    println!("✅ Parallel execution with isolation test passed!");
    println!("   - All tests executed with proper isolation");
    println!("   - Environment variables isolated correctly");
    println!("   - Temporary files isolated correctly");
    println!("   - Semaphore controlled parallelism correctly");
}

#[tokio::test]
async fn test_semaphore_limits_parallelism() {
    println!("🚀 Testing semaphore limits parallelism...");

    // Create test harness with only 1 parallel slot (sequential execution)
    let harness = SimpleParallelHarness::new(1, Duration::from_secs(10));

    // Create test cases
    let mut test_cases: Vec<Box<dyn SimpleTestCase>> = Vec::new();
    test_cases.push(Box::new(IsolationTestCase::new(
        "sequential_1",
        Duration::from_millis(200),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "sequential_2",
        Duration::from_millis(200),
        false,
    )));
    test_cases.push(Box::new(IsolationTestCase::new(
        "sequential_3",
        Duration::from_millis(200),
        false,
    )));

    // Record start time
    let start_time = Instant::now();

    // Run tests
    let results = harness.run_tests(test_cases).await;
    let total_duration = start_time.elapsed();

    // Print results
    println!("📊 Test Results:");
    for result in &results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {} ({:?})", status, result.test_name, result.duration);
    }

    // Verify results
    assert_eq!(results.len(), 3, "Should have 3 test results");
    assert_eq!(results.iter().filter(|r| r.passed).count(), 3, "All tests should pass");

    // With max_parallel = 1, execution should be close to sequential
    let expected_min_time = Duration::from_millis(600); // 3 * 200ms
    assert!(
        total_duration >= expected_min_time.mul_f64(0.8),
        "Sequential execution should take at least ~{:?}, got {:?}",
        expected_min_time,
        total_duration
    );

    println!("⏱️  Performance Analysis:");
    println!("  Execution time: {:?} (expected ~{:?})", total_duration, expected_min_time);
    println!("✅ Semaphore limiting test passed!");
}

#[tokio::test]
async fn test_isolation_prevents_interference() {
    println!("🚀 Testing isolation prevents interference...");

    // Create test harness
    let harness = SimpleParallelHarness::new(3, Duration::from_secs(10));

    // Test case that modifies environment and checks for interference
    struct InterferenceTestCase {
        name: String,
        env_key: String,
        env_value: String,
        work_duration: Duration,
    }

    impl InterferenceTestCase {
        fn new(name: &str, env_key: &str, env_value: &str, work_duration: Duration) -> Self {
            Self {
                name: name.to_string(),
                env_key: env_key.to_string(),
                env_value: env_value.to_string(),
                work_duration,
            }
        }
    }

    impl SimpleTestCase for InterferenceTestCase {
        fn name(&self) -> &str {
            &self.name
        }

        async fn run(&self) -> Result<(), String> {
            // Verify we're in isolated environment
            let test_name =
                std::env::var("BITNET_TEST_NAME").map_err(|_| "Not in isolated environment")?;
            if test_name != self.name {
                return Err(format!(
                    "Wrong test environment: expected '{}', got '{}'",
                    self.name, test_name
                ));
            }

            // Set a test-specific environment variable
            unsafe {
                std::env::set_var(&self.env_key, &self.env_value);
            }

            // Create a file in temp directory
            let temp_dir =
                std::env::var("BITNET_TEST_TEMP_DIR").map_err(|_| "Temp dir not available")?;
            let test_file =
                std::path::Path::new(&temp_dir).join(format!("{}_{}.txt", self.name, self.env_key));
            std::fs::write(&test_file, &self.env_value)
                .map_err(|e| format!("Failed to write file: {}", e))?;

            // Wait a bit to allow other tests to potentially interfere
            sleep(self.work_duration).await;

            // Verify our environment variable is still correct
            let actual_value =
                std::env::var(&self.env_key).map_err(|_| "Environment variable lost")?;
            if actual_value != self.env_value {
                return Err(format!(
                    "Environment interference detected: expected '{}', got '{}'",
                    self.env_value, actual_value
                ));
            }

            // Verify our file still exists and has correct content
            let file_content = std::fs::read_to_string(&test_file)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            if file_content != self.env_value {
                return Err(format!(
                    "File interference detected: expected '{}', got '{}'",
                    self.env_value, file_content
                ));
            }

            // Verify we're still in the correct test environment
            let test_name_after =
                std::env::var("BITNET_TEST_NAME").map_err(|_| "Test name lost")?;
            if test_name_after != self.name {
                return Err(format!(
                    "Test environment changed: expected '{}', got '{}'",
                    self.name, test_name_after
                ));
            }

            Ok(())
        }
    }

    // Create test cases that would interfere if not properly isolated
    let mut test_cases: Vec<Box<dyn SimpleTestCase>> = Vec::new();
    test_cases.push(Box::new(InterferenceTestCase::new(
        "test_a",
        "TEST_VAR",
        "value_a",
        Duration::from_millis(200),
    )));
    test_cases.push(Box::new(InterferenceTestCase::new(
        "test_b",
        "TEST_VAR",
        "value_b",
        Duration::from_millis(250),
    )));
    test_cases.push(Box::new(InterferenceTestCase::new(
        "test_c",
        "ANOTHER_VAR",
        "value_c",
        Duration::from_millis(150),
    )));
    test_cases.push(Box::new(InterferenceTestCase::new(
        "test_d",
        "TEST_VAR",
        "value_d",
        Duration::from_millis(300),
    )));

    // Run tests
    let results = harness.run_tests(test_cases).await;

    // Print results
    println!("📊 Test Results:");
    for result in &results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {} ({:?})", status, result.test_name, result.duration);
        if let Some(error) = &result.error {
            println!("    Error: {}", error);
        }
    }

    // Verify all tests passed (no interference)
    assert_eq!(results.len(), 4, "Should have 4 test results");

    for result in &results {
        if !result.passed {
            panic!(
                "Test '{}' failed: {}",
                result.test_name,
                result.error.as_deref().unwrap_or("Unknown error")
            );
        }
    }

    println!("✅ Isolation prevents interference test passed!");
    println!("   - All tests ran without interfering with each other");
    println!("   - Environment variables properly isolated");
    println!("   - File system properly isolated");
}

#[tokio::test]
async fn test_timeout_handling() {
    println!("🚀 Testing timeout handling...");

    // Create test harness with short timeout
    let harness = SimpleParallelHarness::new(2, Duration::from_millis(500));

    struct TimeoutTestCase {
        name: String,
        work_duration: Duration,
    }

    impl TimeoutTestCase {
        fn new(name: &str, work_duration: Duration) -> Self {
            Self { name: name.to_string(), work_duration }
        }
    }

    impl SimpleTestCase for TimeoutTestCase {
        fn name(&self) -> &str {
            &self.name
        }

        async fn run(&self) -> Result<(), String> {
            // Verify isolation
            let test_name =
                std::env::var("BITNET_TEST_NAME").map_err(|_| "Not in isolated environment")?;
            if test_name != self.name {
                return Err(format!(
                    "Wrong test environment: expected '{}', got '{}'",
                    self.name, test_name
                ));
            }

            // Sleep for the specified duration
            sleep(self.work_duration).await;
            Ok(())
        }
    }

    let mut test_cases: Vec<Box<dyn SimpleTestCase>> = Vec::new();
    test_cases.push(Box::new(TimeoutTestCase::new("fast_test", Duration::from_millis(200)))); // Should pass
    test_cases.push(Box::new(TimeoutTestCase::new("slow_test", Duration::from_millis(800)))); // Should timeout

    let results = harness.run_tests(test_cases).await;

    // Print results
    println!("📊 Test Results:");
    for result in &results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {} ({:?})", status, result.test_name, result.duration);
        if let Some(error) = &result.error {
            println!("    Error: {}", error);
        }
    }

    assert_eq!(results.len(), 2, "Should have 2 test results");

    // Find the fast and slow test results
    let fast_result = results.iter().find(|r| r.test_name == "fast_test").unwrap();
    let slow_result = results.iter().find(|r| r.test_name == "slow_test").unwrap();

    assert!(fast_result.passed, "Fast test should pass");
    assert!(!slow_result.passed, "Slow test should fail due to timeout");
    assert!(
        slow_result.error.as_ref().unwrap().contains("timed out"),
        "Slow test should have timeout error"
    );

    println!("✅ Timeout handling test passed!");
    println!("   - Fast test completed successfully");
    println!("   - Slow test properly timed out");
}
