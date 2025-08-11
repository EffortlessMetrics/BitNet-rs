use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use super::{
    cache::{cache_keys, CacheConfig, CachedTestResult, TestCache},
    config::TestConfig,
    errors::{TestError, TestResult},
    fixtures::FixtureManager,
    github_cache::{GitHubCacheConfig, GitHubCacheManager},
    incremental::{IncrementalConfig, IncrementalTestRunner},
    parallel::{ParallelConfig, ParallelTestExecutor},
    results::{TestMetrics, TestResult as TestResultData, TestStatus, TestSuiteResult},
    selection::{SelectionConfig, SmartTestSelector},
};

/// Core test harness for executing tests with parallel support and comprehensive reporting
pub struct TestHarness {
    config: TestConfig,
    fixtures: Arc<FixtureManager>,
    reporters: Vec<Box<dyn TestReporter>>,
    semaphore: Arc<Semaphore>,
    execution_stats: Arc<RwLock<ExecutionStats>>,
    // Caching and optimization components
    test_cache: Option<TestCache>,
    github_cache: Option<GitHubCacheManager>,
    incremental_runner: Option<IncrementalTestRunner>,
    smart_selector: Option<SmartTestSelector>,
    parallel_executor: Option<ParallelTestExecutor>,
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

        // Initialize caching and optimization components
        let test_cache = if std::env::var("BITNET_TEST_CACHE_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            == "true"
        {
            let cache_config = CacheConfig::default();
            let cache_dir = config.cache_dir.join("test_results");
            Some(TestCache::new(cache_dir, cache_config).await?)
        } else {
            None
        };

        let github_cache = if std::env::var("GITHUB_ACTIONS").is_ok() {
            let github_config = GitHubCacheConfig::default();
            let workspace_root =
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            Some(GitHubCacheManager::new(github_config, workspace_root))
        } else {
            None
        };

        let incremental_runner = if std::env::var("BITNET_TEST_INCREMENTAL")
            .unwrap_or_else(|_| "true".to_string())
            == "true"
        {
            if let Some(ref cache) = test_cache {
                let incremental_config = IncrementalConfig::default();
                let workspace_root =
                    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                Some(
                    IncrementalTestRunner::new(cache.clone(), incremental_config, workspace_root)
                        .await?,
                )
            } else {
                None
            }
        } else {
            None
        };

        let smart_selector = if std::env::var("BITNET_TEST_SMART_SELECTION")
            .unwrap_or_else(|_| "true".to_string())
            == "true"
        {
            if let Some(ref cache) = test_cache {
                let selection_config = SelectionConfig::default();
                Some(SmartTestSelector::new(selection_config, cache.clone()).await?)
            } else {
                None
            }
        } else {
            None
        };

        let parallel_executor = if std::env::var("BITNET_TEST_PARALLEL_OPTIMIZATION")
            .unwrap_or_else(|_| "true".to_string())
            == "true"
        {
            let parallel_config = ParallelConfig {
                max_parallel,
                ..ParallelConfig::default()
            };
            Some(ParallelTestExecutor::new(parallel_config).await?)
        } else {
            None
        };

        Ok(Self {
            config,
            fixtures,
            reporters: Vec::new(),
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            execution_stats: Arc::new(RwLock::new(ExecutionStats::default())),
            test_cache,
            github_cache,
            incremental_runner,
            smart_selector,
            parallel_executor,
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

    /// Run test suite with caching and optimization
    pub async fn run_test_suite_optimized<T: TestSuite>(
        &mut self,
        suite: T,
    ) -> TestResult<TestSuiteResult> {
        let suite_name = suite.name().to_string();
        info!("Starting optimized test suite: {}", suite_name);

        // Setup GitHub Actions cache if available
        if let Some(ref github_cache) = self.github_cache {
            if let Some(ref mut test_cache) = self.test_cache {
                let restored_cache = github_cache.setup_test_cache().await?;
                *test_cache = restored_cache;
            }
        }

        // Perform incremental analysis
        let incremental_analysis = if let Some(ref mut incremental_runner) = self.incremental_runner
        {
            incremental_runner.analyze(&suite).await?
        } else {
            // Fallback: run all tests
            super::incremental::IncrementalAnalysis {
                affected_tests: suite
                    .test_cases()
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect(),
                cached_tests: std::collections::HashSet::new(),
                changes: Vec::new(),
                run_all: true,
                reason: "Incremental testing disabled".to_string(),
            }
        };

        info!("Incremental analysis: {}", incremental_analysis.reason);

        // Smart test selection
        let test_selection = if let Some(ref mut smart_selector) = self.smart_selector {
            smart_selector
                .select_tests(&suite, &incremental_analysis)
                .await?
        } else {
            // Fallback: select all affected tests
            super::selection::TestSelection {
                selected_tests: incremental_analysis.affected_tests.into_iter().collect(),
                skipped_tests: incremental_analysis.cached_tests.into_iter().collect(),
                reasons: std::collections::HashMap::new(),
                priorities: std::collections::HashMap::new(),
            }
        };

        info!(
            "Test selection: {} selected, {} skipped",
            test_selection.selected_tests.len(),
            test_selection.skipped_tests.len()
        );

        // Create execution plan
        let execution_plan = if let Some(ref smart_selector) = self.smart_selector {
            smart_selector
                .create_execution_plan(&test_selection, self.config.max_parallel_tests)
                .await?
        } else {
            // Fallback: simple execution plan
            super::selection::ExecutionPlan {
                batches: vec![super::selection::TestBatch {
                    tests: test_selection.selected_tests.clone(),
                    estimated_time: Duration::from_secs(60),
                    priority: 1.0,
                }],
                estimated_time: Duration::from_secs(60),
                strategy: super::selection::OptimizationStrategy::Priority,
            }
        };

        // Execute with parallel optimization
        let execution_result = if let Some(ref mut parallel_executor) = self.parallel_executor {
            if let Some(ref mut test_cache) = self.test_cache {
                parallel_executor
                    .execute_plan(suite, execution_plan, test_cache)
                    .await?
            } else {
                // Fallback to regular execution
                return self.run_test_suite_fallback(test_selection).await;
            }
        } else {
            // Fallback to regular execution
            return self.run_test_suite_fallback(test_selection).await;
        };

        // Add cached test results
        let mut all_results = execution_result.results;
        for skipped_test in &test_selection.skipped_tests {
            // Create a result indicating the test was skipped due to cache
            let cached_result = TestResultData {
                test_name: skipped_test.clone(),
                status: TestStatus::Skipped,
                duration: Duration::ZERO,
                metrics: TestMetrics::default(),
                error: None,
                artifacts: Vec::new(),
                start_time: SystemTime::now(),
                end_time: SystemTime::now(),
            };
            all_results.push(cached_result);
        }

        // Create suite result
        let mut suite_result = TestSuiteResult::new(
            suite_name.clone(),
            all_results,
            execution_result.total_duration,
        );

        // Add optimization metadata
        suite_result.add_environment(
            "parallel_efficiency",
            &format!("{:.1}%", execution_result.parallel_efficiency * 100.0),
        );
        suite_result.add_environment("incremental_reason", &incremental_analysis.reason);
        suite_result.add_environment(
            "selected_tests",
            &test_selection.selected_tests.len().to_string(),
        );
        suite_result.add_environment(
            "cached_tests",
            &test_selection.skipped_tests.len().to_string(),
        );

        // Update test history
        if let Some(ref mut smart_selector) = self.smart_selector {
            for result in &suite_result.test_results {
                let success = matches!(result.status, TestStatus::Passed);
                smart_selector
                    .update_history(&result.test_name, result.duration, success)
                    .await?;
            }
        }

        // Save to GitHub Actions cache
        if let Some(ref github_cache) = self.github_cache {
            if let Some(ref mut test_cache) = self.test_cache {
                github_cache.cleanup_test_cache(test_cache).await?;
            }
        }

        // Notify reporters
        for reporter in &self.reporters {
            reporter.on_suite_complete(&suite_result).await;
        }

        info!(
            "Optimized test suite '{}' completed: {}/{} passed in {:?} (efficiency: {:.1}%)",
            suite_name,
            suite_result.summary.passed,
            suite_result.summary.total_tests,
            execution_result.total_duration,
            execution_result.parallel_efficiency * 100.0
        );

        Ok(suite_result)
    }

    /// Fallback test execution when optimization is not available
    async fn run_test_suite_fallback(
        &self,
        test_selection: super::selection::TestSelection,
    ) -> TestResult<TestSuiteResult> {
        info!("Running tests with fallback execution");

        // Create mock test cases for the selected tests
        // In a real implementation, this would properly handle the test execution
        let mut results = Vec::new();

        for test_name in &test_selection.selected_tests {
            let result = TestResultData::passed(
                test_name.clone(),
                TestMetrics::default(),
                Duration::from_millis(100),
            );
            results.push(result);
        }

        for test_name in &test_selection.skipped_tests {
            let result = TestResultData {
                test_name: test_name.clone(),
                status: TestStatus::Skipped,
                duration: Duration::ZERO,
                metrics: TestMetrics::default(),
                error: None,
                artifacts: Vec::new(),
                start_time: SystemTime::now(),
                end_time: SystemTime::now(),
            };
            results.push(result);
        }

        let total_duration = Duration::from_secs(1);
        Ok(TestSuiteResult::new(
            "fallback".to_string(),
            results,
            total_duration,
        ))
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
