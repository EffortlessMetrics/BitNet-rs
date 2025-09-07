use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tempfile;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;

use super::{
    config::TestConfig,
    errors::{TestError, TestOpResult as TestResultCompat},
    fixtures_facade::Fixtures,
    results::{TestMetrics, TestResult as TestRecord, TestStatus, TestSuiteResult},
};

// Import the stable fixture context type
mod fixture_ctx;
pub use fixture_ctx::FixtureCtx;

/// Core test harness for executing tests with parallel support and proper isolation
pub struct TestHarness {
    config: TestConfig,
    fixtures: Fixtures,
    reporters: Vec<ConsoleReporter>,
    semaphore: Arc<Semaphore>,
    execution_stats: Arc<RwLock<ExecutionStats>>,
}

impl TestHarness {
    /// Create a new test harness with the given configuration
    pub async fn new(config: TestConfig) -> TestResultCompat<Self> {
        let max_parallel = config.max_parallel_tests;
        let fixtures = Fixtures::new(&config).await?;

        println!("Initializing test harness with {} parallel slots", max_parallel);

        Ok(Self {
            config,
            fixtures,
            reporters: Vec::new(),
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            execution_stats: Arc::new(RwLock::new(ExecutionStats::default())),
        })
    }

    /// Add a test reporter
    pub fn add_reporter(&mut self, reporter: ConsoleReporter) {
        self.reporters.push(reporter);
    }

    /// Run a complete test suite with parallel execution and proper isolation
    pub async fn run_test_suite(&self, suite: &dyn TestSuite) -> TestResultCompat<TestSuiteResult> {
        let suite_name = suite.name().to_string();
        let start_time = Instant::now();

        println!("Starting test suite: {}", suite_name);

        // Notify reporters of suite start
        for reporter in &self.reporters {
            reporter.on_suite_start(&suite_name).await;
        }

        // Get all test cases
        let test_cases = suite.test_cases();
        let total_tests = test_cases.len();

        println!("Running {} tests in suite '{}'", total_tests, suite_name);

        // Execute tests with controlled parallelism and proper isolation
        let mut test_results = Vec::with_capacity(total_tests);
        let mut handles = Vec::new();

        for test_case in test_cases {
            let harness_clone = self.clone_for_test();
            let handle =
                tokio::spawn(
                    async move { harness_clone.run_single_test_isolated(test_case).await },
                );
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
                    eprintln!("Test execution failed: {}", e);
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
            std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        );
        suite_result.add_environment(
            "target_triple",
            std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
        );
        suite_result.add_environment("test_framework_version", "0.1.0");

        // Add configuration information
        suite_result
            .add_configuration("max_parallel_tests", self.config.max_parallel_tests.to_string());
        suite_result.add_configuration("test_timeout", format!("{:?}", self.config.test_timeout));

        println!(
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

    /// Run a single test case with proper isolation
    async fn run_single_test_isolated(&self, test_case: Box<dyn TestCase>) -> TestRecord {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();
        let start_system_time = SystemTime::now();

        println!("Starting test: {}", test_name);

        // Acquire semaphore permit for parallel execution control
        let _permit = match self.semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                eprintln!("Failed to acquire semaphore permit: {}", e);
                return TestRecord::failed(
                    test_name,
                    TestError::execution(format!("Semaphore error: {}", e)),
                    Duration::ZERO,
                );
            }
        };

        // Create isolated environment for this test
        let isolated_env = self.create_isolated_environment(&test_name).await;

        // Setup phase with isolation
        println!("Setting up test: {}", test_name);
        if let Err(e) = self.setup_test_with_isolation(test_case.as_ref(), &isolated_env).await {
            eprintln!("Test setup failed for '{}': {}", test_name, e);
            self.cleanup_isolated_environment(isolated_env).await;
            return TestRecord::failed(test_name, e, start_time.elapsed());
        }

        // Execute phase with timeout and isolation
        println!("Executing test: {}", test_name);
        let execute_result = timeout(
            self.config.test_timeout,
            self.execute_test_with_isolation(test_case.as_ref(), &isolated_env),
        )
        .await;

        let duration = start_time.elapsed();

        // Cleanup phase (always run, even if test failed)
        println!("Cleaning up test: {}", test_name);
        if let Err(e) = self.cleanup_test_with_isolation(test_case.as_ref(), &isolated_env).await {
            eprintln!("Test cleanup failed for '{}': {}", test_name, e);
        }

        // Clean up isolated environment
        self.cleanup_isolated_environment(isolated_env).await;

        // Process execution result
        let mut result = match execute_result {
            Ok(Ok(metrics)) => {
                println!("Test '{}' passed in {:?}", test_name, duration);
                TestRecord::passed(test_name, metrics, duration)
            }
            Ok(Err(e)) => {
                eprintln!("Test '{}' failed: {}", test_name, e);
                TestRecord::failed(test_name, e, duration)
            }
            Err(_) => {
                eprintln!("Test '{}' timed out after {:?}", test_name, self.config.test_timeout);
                TestRecord::timeout(test_name, self.config.test_timeout)
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
            fixtures: self.fixtures.clone(),
            semaphore: Arc::clone(&self.semaphore),
        }
    }

    /// Create an isolated environment for a test
    async fn create_isolated_environment(&self, test_name: &str) -> IsolatedEnvironment {
        let temp_dir = tempfile::tempdir()
            .unwrap_or_else(|_| tempfile::tempdir_in("/tmp").expect("Failed to create temp dir"));

        let env_vars = std::env::vars().collect();

        IsolatedEnvironment {
            test_name: test_name.to_string(),
            temp_dir,
            original_env_vars: env_vars,
            isolated_env_vars: HashMap::new(),
        }
    }

    /// Setup test with proper isolation
    async fn setup_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        env: &IsolatedEnvironment,
    ) -> TestResultCompat<()> {
        // Set isolated environment variables
        for (key, value) in &env.isolated_env_vars {
            unsafe {
                std::env::set_var(key, value);
            }
        }

        // Set test-specific environment
        unsafe {
            std::env::set_var("BITNET_TEST_TEMP_DIR", env.temp_dir.path());
        }
        unsafe {
            std::env::set_var("BITNET_TEST_NAME", &env.test_name);
        }
        unsafe {
            std::env::set_var("BITNET_TEST_ISOLATION", "true");
        }

        // Call the test's setup method with stable API
        test_case.setup(self.fixtures.ctx())
        .await?;

        Ok(())
    }

    /// Execute test with proper isolation
    async fn execute_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        _env: &IsolatedEnvironment,
    ) -> TestResultCompat<TestMetrics> {
        // Execute the test in the isolated environment
        test_case.execute().await
    }

    /// Cleanup test with proper isolation
    async fn cleanup_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        _env: &IsolatedEnvironment,
    ) -> TestResultCompat<()> {
        // Call the test's cleanup method
        test_case.cleanup().await
    }

    /// Clean up the isolated environment
    async fn cleanup_isolated_environment(&self, env: IsolatedEnvironment) {
        // Restore original environment variables
        for (key, value) in &env.original_env_vars {
            unsafe {
                std::env::set_var(key, value);
            }
        }

        // Remove test-specific environment variables
        unsafe {
            std::env::remove_var("BITNET_TEST_TEMP_DIR");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_NAME");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_ISOLATION");
        }

        // Temp directory is automatically cleaned up when dropped
        drop(env.temp_dir);
    }

    /// Run a single test case (back-compat wrapper for integration tests)
    pub async fn run_single_test(&self, test_case: Box<dyn TestCase>) -> TestRecord {
        self.run_single_test_isolated(test_case).await
    }

    /// Get execution statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.execution_stats.read().await.clone()
    }
}

/// Isolated environment for test execution
struct IsolatedEnvironment {
    test_name: String,
    temp_dir: tempfile::TempDir,
    original_env_vars: HashMap<String, String>,
    isolated_env_vars: HashMap<String, String>,
}

/// Lightweight clone of test harness for async execution
#[derive(Clone)]
struct TestHarnessClone {
    config: TestConfig,
    fixtures: Fixtures,
    semaphore: Arc<Semaphore>,
}

impl TestHarnessClone {
    async fn run_single_test_isolated(&self, test_case: Box<dyn TestCase>) -> TestRecord {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();
        let start_system_time = SystemTime::now();

        // Acquire semaphore permit for parallel execution control
        let _permit = match self.semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                return TestRecord::failed(
                    test_name,
                    TestError::execution(format!("Semaphore error: {}", e)),
                    Duration::ZERO,
                );
            }
        };

        // Create isolated environment
        let isolated_env = self.create_isolated_environment(&test_name).await;

        // Setup phase with isolation
        if let Err(e) = self.setup_test_with_isolation(test_case.as_ref(), &isolated_env).await {
            self.cleanup_isolated_environment(isolated_env).await;
            return TestRecord::failed(test_name, e, start_time.elapsed());
        }

        // Execute phase with timeout and isolation
        let execute_result = timeout(
            self.config.test_timeout,
            self.execute_test_with_isolation(test_case.as_ref(), &isolated_env),
        )
        .await;

        let duration = start_time.elapsed();

        // Cleanup phase
        let _ = self.cleanup_test_with_isolation(test_case.as_ref(), &isolated_env).await;
        self.cleanup_isolated_environment(isolated_env).await;

        // Process result
        let mut result = match execute_result {
            Ok(Ok(metrics)) => TestRecord::passed(test_name, metrics, duration),
            Ok(Err(e)) => TestRecord::failed(test_name, e, duration),
            Err(_) => TestRecord::timeout(test_name, self.config.test_timeout),
        };

        result.start_time = start_system_time;
        result.end_time = SystemTime::now();

        result
    }

    /// Create an isolated environment for a test
    async fn create_isolated_environment(&self, test_name: &str) -> IsolatedEnvironment {
        let temp_dir = tempfile::tempdir()
            .unwrap_or_else(|_| tempfile::tempdir_in("/tmp").expect("Failed to create temp dir"));

        let env_vars = std::env::vars().collect();

        IsolatedEnvironment {
            test_name: test_name.to_string(),
            temp_dir,
            original_env_vars: env_vars,
            isolated_env_vars: HashMap::new(),
        }
    }

    /// Setup test with proper isolation
    async fn setup_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        env: &IsolatedEnvironment,
    ) -> TestResultCompat<()> {
        // Set isolated environment variables
        for (key, value) in &env.isolated_env_vars {
            unsafe {
                std::env::set_var(key, value);
            }
        }

        // Set test-specific environment
        unsafe {
            std::env::set_var("BITNET_TEST_TEMP_DIR", env.temp_dir.path());
        }
        unsafe {
            std::env::set_var("BITNET_TEST_NAME", &env.test_name);
        }
        unsafe {
            std::env::set_var("BITNET_TEST_ISOLATION", "true");
        }

        // Call the test's setup method with stable API
        test_case.setup(self.fixtures.ctx())
        .await?;

        Ok(())
    }

    /// Execute test with proper isolation
    async fn execute_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        _env: &IsolatedEnvironment,
    ) -> TestResultCompat<TestMetrics> {
        // Execute the test in the isolated environment
        test_case.execute().await
    }

    /// Cleanup test with proper isolation
    async fn cleanup_test_with_isolation(
        &self,
        test_case: &dyn TestCase,
        _env: &IsolatedEnvironment,
    ) -> TestResultCompat<()> {
        // Call the test's cleanup method
        test_case.cleanup().await
    }

    /// Clean up the isolated environment
    async fn cleanup_isolated_environment(&self, env: IsolatedEnvironment) {
        // Restore original environment variables
        for (key, value) in &env.original_env_vars {
            unsafe {
                std::env::set_var(key, value);
            }
        }

        // Remove test-specific environment variables
        unsafe {
            std::env::remove_var("BITNET_TEST_TEMP_DIR");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_NAME");
        }
        unsafe {
            std::env::remove_var("BITNET_TEST_ISOLATION");
        }

        // Temp directory is automatically cleaned up when dropped
        drop(env.temp_dir);
    }
}

/// Trait for individual test cases
#[async_trait]
pub trait TestCase: Send + Sync {
    /// Get the name of this test case
    fn name(&self) -> &str;

    /// Set up the test case (called before execute)
    /// Uses stable API with FixtureCtx type that adapts based on features
    async fn setup(&self, fixtures: FixtureCtx<'_>) -> TestResultCompat<()> {
        let _ = fixtures; // Default no-op implementation
        Ok(())
    }

    /// Execute the test case
    async fn execute(&self) -> TestResultCompat<TestMetrics>;

    /// Clean up after the test case (always called, even if execute fails)
    async fn cleanup(&self) -> TestResultCompat<()> {
        Ok(())
    }

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
    async fn on_test_complete(&self, result: &TestRecord);

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

    async fn on_test_complete(&self, result: &TestRecord) {
        if self.verbose || result.is_failure() {
            let status_symbol = match result.status {
                TestStatus::Passed => "✓",
                TestStatus::Failed => "✗",
                TestStatus::Skipped => "⊝",
                TestStatus::Timeout => "⏱",
                TestStatus::Running => "⋯",
            };

            println!("  {} {} ({:?})", status_symbol, result.test_name, result.duration);

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
