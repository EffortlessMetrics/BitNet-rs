use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::{
    config::TestConfig,
    errors::{TestError, TestResult},
    fixtures::FixtureManager,
    results::{TestMetrics, TestResult as TestResultData, TestStatus, TestSuiteResult},
};

/// Core test harness for executing tests with parallel support and comprehensive reporting
pub struct TestHarness {
    config: TestConfig,
    fixtures: Arc<FixtureManager>,
    reporters: Vec<Box<dyn TestReporter>>,
    semaphore: Arc<Semaphore>,
    execution_stats: Arc<RwLock<ExecutionStats>>,
}

impl TestHarness {
    /// Create a new test harness with the given configuration
    pub async fn new(config: TestConfig) -> TestResult<Self> {
        let max_parallel = config.max_parallel_tests;
        let fixtures = Arc::new(FixtureManager::new(&config.fixtures).await?);

        info!(
            "Initializing test harness with {} parallel slots",
            max_parallel
        );

        Ok(Self {
            config,
            fixtures,
            reporters: Vec::new(),
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            execution_stats: Arc::new(RwLock::new(ExecutionStats::default())),
        })
    }

    /// Add a test reporter
    pub fn add_reporter(&mut self, reporter: Box<dyn TestReporter>) {
        self.reporters.push(reporter);
    }

    /// Run a complete test suite
    pub async fn run_test_suite<T: TestSuite>(&self, suite: T) -> TestResult<TestSuiteResult> {
        let suite_name = suite.name().to_string();
        let start_time = Instant::now();

        info!("Starting test suite: {}", suite_name);

        // Notify reporters of suite start
        for reporter in &self.reporters {
            reporter.on_suite_start(&suite_name).await;
        }

        // Get all test cases
        let test_cases = suite.test_cases();
        let total_tests = test_cases.len();

        info!("Running {} tests in suite '{}'", total_tests, suite_name);

        // Execute tests with controlled parallelism
        let mut test_results = Vec::with_capacity(total_tests);
        let mut handles = Vec::new();

        for test_case in test_cases {
            let harness_clone = self.clone_for_test();
            let handle =
                tokio::spawn(async move { harness_clone.run_single_test(test_case).await });
            handles.push(handle);
        }

        // Collect results as they complete
        for handle in handles {
            match handle.await {
                Ok(result) => {
                    // Notify reporters of individual test completion
                    for reporter in &self.reporters {
                        reporter.on_test_complete(&result).await;
                    }
                    test_results.push(result);
                }
                Err(e) => {
                    error!("Test execution failed: {}", e);
                    return Err(TestError::execution(format!("Test join failed: {}", e)));
                }
            }
        }

        let total_duration = start_time.elapsed();

        // Create suite result
        let mut suite_result =
            TestSuiteResult::new(suite_name.clone(), test_results, total_duration);

        // Add environment information
        suite_result.add_environment(
            "rust_version",
            &std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        );
        suite_result.add_environment(
            "target_triple",
            &std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
        );
        suite_result.add_environment("test_framework_version", crate::TESTING_FRAMEWORK_VERSION);

        // Add configuration information
        suite_result.add_configuration(
            "max_parallel_tests",
            self.config.max_parallel_tests.to_string(),
        );
        suite_result.add_configuration("test_timeout", format!("{:?}", self.config.test_timeout));

        info!(
            "Completed test suite '{}': {}/{} passed in {:?}",
            suite_name,
            suite_result.summary.passed,
            suite_result.summary.total_tests,
            total_duration
        );

        // Notify reporters of suite completion
        for reporter in &self.reporters {
            reporter.on_suite_complete(&suite_result).await;
        }

        // Update execution stats
        {
            let mut stats = self.execution_stats.write().await;
            stats.total_suites += 1;
            stats.total_tests += suite_result.summary.total_tests;
            stats.total_passed += suite_result.summary.passed;
            stats.total_failed += suite_result.summary.failed;
            stats.total_duration += total_duration;
        }

        Ok(suite_result)
    }

    /// Run a single test case
    async fn run_single_test(&self, test_case: Box<dyn TestCase>) -> TestResultData {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();
        let start_system_time = SystemTime::now();

        debug!("Starting test: {}", test_name);

        // Acquire semaphore permit for parallel execution control
        let _permit = match self.semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                error!("Failed to acquire semaphore permit: {}", e);
                return TestResultData::failed(
                    test_name,
                    TestError::execution(format!("Semaphore error: {}", e)),
                    Duration::ZERO,
                );
            }
        };

        // Setup phase
        debug!("Setting up test: {}", test_name);
        if let Err(e) = test_case.setup(&self.fixtures).await {
            warn!("Test setup failed for '{}': {}", test_name, e);
            return TestResultData::failed(test_name, e, start_time.elapsed());
        }

        // Execute phase with timeout
        debug!("Executing test: {}", test_name);
        let execute_result = timeout(self.config.test_timeout, test_case.execute()).await;

        let duration = start_time.elapsed();

        // Cleanup phase (always run, even if test failed)
        debug!("Cleaning up test: {}", test_name);
        if let Err(e) = test_case.cleanup().await {
            warn!("Test cleanup failed for '{}': {}", test_name, e);
        }

        // Process execution result
        let mut result = match execute_result {
            Ok(Ok(metrics)) => {
                debug!("Test '{}' passed in {:?}", test_name, duration);
                TestResultData::passed(test_name, metrics, duration)
            }
            Ok(Err(e)) => {
                warn!("Test '{}' failed: {}", test_name, e);
                TestResultData::failed(test_name, e, duration)
            }
            Err(_) => {
                warn!(
                    "Test '{}' timed out after {:?}",
                    test_name, self.config.test_timeout
                );
                TestResultData::timeout(test_name, self.config.test_timeout)
            }
        };

        // Set accurate timestamps
        result.start_time = start_system_time;
        result.end_time = SystemTime::now();

        result
    }

    /// Clone harness for test execution (lightweight clone for async tasks)
    fn clone_for_test(&self) -> TestHarnessClone {
        TestHarnessClone {
            config: self.config.clone(),
            fixtures: Arc::clone(&self.fixtures),
            semaphore: Arc::clone(&self.semaphore),
        }
    }

    /// Get execution statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.execution_stats.read().await.clone()
    }
}

/// Lightweight clone of test harness for async execution
#[derive(Clone)]
struct TestHarnessClone {
    config: TestConfig,
    fixtures: Arc<FixtureManager>,
    semaphore: Arc<Semaphore>,
}

impl TestHarnessClone {
    async fn run_single_test(&self, test_case: Box<dyn TestCase>) -> TestResultData {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();
        let start_system_time = SystemTime::now();

        // Acquire semaphore permit
        let _permit = match self.semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                return TestResultData::failed(
                    test_name,
                    TestError::execution(format!("Semaphore error: {}", e)),
                    Duration::ZERO,
                );
            }
        };

        // Setup phase
        if let Err(e) = test_case.setup(&self.fixtures).await {
            return TestResultData::failed(test_name, e, start_time.elapsed());
        }

        // Execute phase with timeout
        let execute_result = timeout(self.config.test_timeout, test_case.execute()).await;

        let duration = start_time.elapsed();

        // Cleanup phase
        let _ = test_case.cleanup().await;

        // Process result
        let mut result = match execute_result {
            Ok(Ok(metrics)) => TestResultData::passed(test_name, metrics, duration),
            Ok(Err(e)) => TestResultData::failed(test_name, e, duration),
            Err(_) => TestResultData::timeout(test_name, self.config.test_timeout),
        };

        result.start_time = start_system_time;
        result.end_time = SystemTime::now();

        result
    }
}

/// Trait for individual test cases
#[async_trait]
pub trait TestCase: Send + Sync {
    /// Get the name of this test case
    fn name(&self) -> &str;

    /// Set up the test case (called before execute)
    async fn setup(&self, fixtures: &FixtureManager) -> TestResult<()>;

    /// Execute the test case
    async fn execute(&self) -> TestResult<TestMetrics>;

    /// Clean up after the test case (always called, even if execute fails)
    async fn cleanup(&self) -> TestResult<()>;

    /// Get test metadata (optional)
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Check if this test should be skipped
    fn should_skip(&self) -> Option<String> {
        None
    }
}

/// Trait for test suites (collections of test cases)
pub trait TestSuite: Send + Sync {
    /// Get the name of this test suite
    fn name(&self) -> &str;

    /// Get all test cases in this suite
    fn test_cases(&self) -> Vec<Box<dyn TestCase>>;

    /// Get suite metadata (optional)
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for test result reporting
#[async_trait]
pub trait TestReporter: Send + Sync {
    /// Called when a test suite starts
    async fn on_suite_start(&self, suite_name: &str);

    /// Called when an individual test completes
    async fn on_test_complete(&self, result: &TestResultData);

    /// Called when a test suite completes
    async fn on_suite_complete(&self, result: &TestSuiteResult);

    /// Called when all testing is complete
    async fn on_all_complete(&self, stats: &ExecutionStats);
}

/// Statistics about test execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    pub total_suites: usize,
    pub total_tests: usize,
    pub total_passed: usize,
    pub total_failed: usize,
    pub total_duration: Duration,
}

impl ExecutionStats {
    /// Calculate success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_tests > 0 {
            (self.total_passed as f64 / self.total_tests as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate average test duration
    pub fn average_duration(&self) -> Duration {
        if self.total_tests > 0 {
            self.total_duration / self.total_tests as u32
        } else {
            Duration::ZERO
        }
    }
}

/// Simple console reporter for test results
pub struct ConsoleReporter {
    verbose: bool,
}

impl ConsoleReporter {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

#[async_trait]
impl TestReporter for ConsoleReporter {
    async fn on_suite_start(&self, suite_name: &str) {
        println!("Running test suite: {}", suite_name);
    }

    async fn on_test_complete(&self, result: &TestResultData) {
        if self.verbose || result.is_failure() {
            let status_symbol = match result.status {
                TestStatus::Passed => "✓",
                TestStatus::Failed => "✗",
                TestStatus::Skipped => "⊝",
                TestStatus::Timeout => "⏱",
                TestStatus::Running => "⋯",
            };

            println!(
                "  {} {} ({:?})",
                status_symbol, result.test_name, result.duration
            );

            if let Some(error) = &result.error {
                println!("    Error: {}", error);
            }
        }
    }

    async fn on_suite_complete(&self, result: &TestSuiteResult) {
        println!(
            "Suite '{}' completed: {}/{} passed ({:.1}%) in {:?}",
            result.suite_name,
            result.summary.passed,
            result.summary.total_tests,
            result.summary.success_rate,
            result.total_duration
        );

        if !result.is_success() {
            println!("Failed tests:");
            for failed_test in result.failed_tests() {
                println!(
                    "  - {}: {}",
                    failed_test.test_name,
                    failed_test.error.as_deref().unwrap_or("Unknown error")
                );
            }
        }
    }

    async fn on_all_complete(&self, stats: &ExecutionStats) {
        println!("\nTest execution complete:");
        println!("  Suites: {}", stats.total_suites);
        println!("  Tests: {}", stats.total_tests);
        println!("  Passed: {}", stats.total_passed);
        println!("  Failed: {}", stats.total_failed);
        println!("  Success rate: {:.1}%", stats.success_rate());
        println!("  Total duration: {:?}", stats.total_duration);
        println!("  Average per test: {:?}", stats.average_duration());
    }
}
