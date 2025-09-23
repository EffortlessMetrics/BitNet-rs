use std::collections::HashMap;
use std::sync::Arc;

use super::debugging::{DebugConfig, TestDebugger};
use super::errors::TestOpResult;
use super::harness::{FixtureCtx, TestCase, TestHarness, TestReporter};
use super::results::{TestResult, TestSuiteResult};

/// Enhanced test harness with integrated debugging support
pub struct DebugEnabledTestHarness {
    inner_harness: TestHarness,
    debugger: Arc<TestDebugger>,
    debug_enabled: bool,
}

impl DebugEnabledTestHarness {
    /// Create a new debug-enabled test harness
    pub async fn new(
        harness: TestHarness,
        debug_config: Option<DebugConfig>,
    ) -> TestOpResult<Self> {
        let debug_config = debug_config.unwrap_or_default();
        let debug_enabled = debug_config.enabled;

        let debugger = if debug_enabled {
            Arc::new(TestDebugger::new(debug_config).await?)
        } else {
            // Create a no-op debugger
            Arc::new(TestDebugger::new(DebugConfig { enabled: false, ..Default::default() }).await?)
        };

        Ok(Self { inner_harness: harness, debugger, debug_enabled })
    }

    /// Run a test suite with debugging support
    pub async fn run_test_suite_with_debug(
        &self,
        suite: &dyn super::harness::TestSuite,
    ) -> TestOpResult<TestSuiteResult> {
        if self.debug_enabled {
            println!("Running test suite '{}' with debugging enabled", suite.name());
        }

        // Start suite-level debugging
        if self.debug_enabled {
            self.debugger
                .add_debug_message("suite", &format!("Starting test suite: {}", suite.name()))
                .await?;
        }

        // Run the test suite
        let result = self.inner_harness.run_test_suite(suite).await?;

        // Generate debug report if enabled
        if self.debug_enabled {
            self.debugger
                .add_debug_message(
                    "suite",
                    &format!(
                        "Completed test suite: {} ({}/{} passed)",
                        suite.name(),
                        result.summary.passed,
                        result.summary.total_tests
                    ),
                )
                .await?;

            let debug_report = self.debugger.generate_debug_report().await?;
            let report_path = self.debugger.save_debug_report(&debug_report).await?;

            println!("Debug report saved to: {}", report_path.display());

            // Generate troubleshooting guide if there were failures
            if result.summary.failed > 0 {
                let guide = self.debugger.generate_troubleshooting_guide().await?;
                let guide_path = report_path.parent().unwrap().join("troubleshooting_guide.md");
                tokio::fs::write(&guide_path, guide).await?;
                println!("Troubleshooting guide saved to: {}", guide_path.display());
            }
        }

        Ok(result)
    }

    /// Get the debugger instance
    pub fn debugger(&self) -> Arc<TestDebugger> {
        Arc::clone(&self.debugger)
    }

    /// Check if debugging is enabled
    pub fn is_debug_enabled(&self) -> bool {
        self.debug_enabled
    }
}

/// Debug-aware test case wrapper
pub struct DebugTestCase {
    inner: Arc<dyn TestCase>,
    debugger: Arc<TestDebugger>,
    test_name: String,
}

impl DebugTestCase {
    pub fn new(inner: Arc<dyn TestCase>, debugger: Arc<TestDebugger>) -> Self {
        let test_name = inner.name().to_string();
        Self { inner, debugger, test_name }
    }
}

#[async_trait::async_trait]
impl TestCase for DebugTestCase {
    fn name(&self) -> &str {
        &self.test_name
    }

    async fn setup(&self, fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        // Start debugging for this test
        self.debugger.start_test_debug(&self.test_name).await?;
        self.debugger.start_phase(&self.test_name, "setup").await?;

        // Run the actual setup
        let result = self.inner.setup(fixtures).await;

        // Record setup result
        match &result {
            Ok(_) => {
                self.debugger.end_phase(&self.test_name, "setup", true, None).await?;
                self.debugger
                    .add_debug_message(&self.test_name, "Setup completed successfully")
                    .await?;
            }
            Err(e) => {
                self.debugger
                    .end_phase(
                        &self.test_name,
                        "setup",
                        false,
                        Some([("error".to_string(), e.to_string())].into()),
                    )
                    .await?;
                self.debugger.capture_error(Some(&self.test_name), e).await?;
            }
        }

        result
    }

    async fn execute(&self) -> TestOpResult<super::results::TestMetrics> {
        self.debugger.start_phase(&self.test_name, "execute").await?;
        self.debugger.add_debug_message(&self.test_name, "Starting test execution").await?;

        // Run the actual execution
        let result = self.inner.execute().await;

        // Record execution result
        match &result {
            Ok(metrics) => {
                self.debugger
                    .end_phase(
                        &self.test_name,
                        "execute",
                        true,
                        Some(
                            [
                                ("duration".to_string(), format!("{:?}", metrics.wall_time)),
                                (
                                    "memory_peak".to_string(),
                                    metrics
                                        .memory_peak
                                        .map(|m| m.to_string())
                                        .unwrap_or_else(|| "unknown".to_string()),
                                ),
                            ]
                            .into(),
                        ),
                    )
                    .await?;
                self.debugger
                    .add_debug_message(
                        &self.test_name,
                        &format!("Execution completed in {:?}", metrics.wall_time),
                    )
                    .await?;
            }
            Err(e) => {
                self.debugger
                    .end_phase(
                        &self.test_name,
                        "execute",
                        false,
                        Some([("error".to_string(), e.to_string())].into()),
                    )
                    .await?;
                self.debugger.capture_error(Some(&self.test_name), e).await?;
            }
        }

        result
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        self.debugger.start_phase(&self.test_name, "cleanup").await?;

        // Run the actual cleanup
        let result = self.inner.cleanup().await;

        // Record cleanup result
        match &result {
            Ok(_) => {
                self.debugger.end_phase(&self.test_name, "cleanup", true, None).await?;
                self.debugger
                    .add_debug_message(&self.test_name, "Cleanup completed successfully")
                    .await?;
            }
            Err(e) => {
                self.debugger
                    .end_phase(
                        &self.test_name,
                        "cleanup",
                        false,
                        Some([("error".to_string(), e.to_string())].into()),
                    )
                    .await?;
                self.debugger.capture_error(Some(&self.test_name), e).await?;
            }
        }

        // End debugging for this test
        let test_result = TestResult::passed(
            self.test_name.clone(),
            Default::default(),
            std::time::Duration::ZERO,
        );
        self.debugger.end_test_debug(&self.test_name, &test_result).await?;

        result
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = self.inner.metadata();
        metadata.insert("debug_enabled".to_string(), "true".to_string());
        metadata
    }

    fn should_skip(&self) -> Option<String> {
        self.inner.should_skip()
    }
}

/// Debug-aware test reporter
pub struct DebugTestReporter {
    debugger: Arc<TestDebugger>,
    verbose: bool,
}

impl DebugTestReporter {
    pub fn new(debugger: Arc<TestDebugger>, verbose: bool) -> Self {
        Self { debugger, verbose }
    }
}

#[async_trait::async_trait]
impl TestReporter for DebugTestReporter {
    async fn on_suite_start(&self, suite_name: &str) {
        if self.verbose {
            println!("🔍 [DEBUG] Starting test suite: {}", suite_name);
        }

        let _ = self
            .debugger
            .add_debug_message("reporter", &format!("Suite started: {}", suite_name))
            .await;
    }

    async fn on_test_complete(&self, result: &TestResult) {
        if self.verbose || !result.is_success() {
            let status_icon = if result.is_success() { "✅" } else { "❌" };
            println!(
                "🔍 [DEBUG] {} Test '{}' completed in {:?}",
                status_icon, result.test_name, result.duration
            );

            if let Some(error) = &result.error {
                println!("🔍 [DEBUG]   Error: {}", error);
            }
        }

        let _ = self
            .debugger
            .add_debug_message(
                "reporter",
                &format!(
                    "Test completed: {} ({})",
                    result.test_name,
                    if result.is_success() { "PASSED" } else { "FAILED" }
                ),
            )
            .await;
    }

    async fn on_suite_complete(&self, result: &TestSuiteResult) {
        if self.verbose {
            println!(
                "🔍 [DEBUG] Suite '{}' completed: {}/{} passed ({:.1}%) in {:?}",
                result.suite_name,
                result.summary.passed,
                result.summary.total_tests,
                result.summary.success_rate,
                result.total_duration
            );
        }

        let _ = self
            .debugger
            .add_debug_message(
                "reporter",
                &format!(
                    "Suite completed: {} ({}/{} passed)",
                    result.suite_name, result.summary.passed, result.summary.total_tests
                ),
            )
            .await;

        // Print debug summary for failed tests
        if result.summary.failed > 0 && self.verbose {
            println!("🔍 [DEBUG] Failed tests in suite '{}':", result.suite_name);
            for test in result.failed_tests() {
                println!(
                    "🔍 [DEBUG]   - {}: {}",
                    test.test_name,
                    test.error.as_deref().unwrap_or("Unknown error")
                );
            }
        }
    }

    async fn on_all_complete(&self, stats: &super::harness::ExecutionStats) {
        if self.verbose {
            println!("🔍 [DEBUG] All tests completed:");
            println!("🔍 [DEBUG]   Total: {}", stats.total_tests);
            println!("🔍 [DEBUG]   Passed: {}", stats.total_passed);
            println!("🔍 [DEBUG]   Failed: {}", stats.total_failed);
            println!("🔍 [DEBUG]   Success rate: {:.1}%", stats.success_rate());
            println!("🔍 [DEBUG]   Duration: {:?}", stats.total_duration);
        }

        let _ = self
            .debugger
            .add_debug_message(
                "reporter",
                &format!(
                    "All tests completed: {}/{} passed ({:.1}%)",
                    stats.total_passed,
                    stats.total_tests,
                    stats.success_rate()
                ),
            )
            .await;
    }
}

/// Utility functions for debug integration
/// Create a debug-enabled test harness from configuration
pub async fn create_debug_harness(
    test_config: super::config::TestConfig,
    debug_config: Option<DebugConfig>,
) -> TestOpResult<DebugEnabledTestHarness> {
    let harness = TestHarness::new(test_config).await?;
    DebugEnabledTestHarness::new(harness, debug_config).await
}

/// Wrap a test case with debugging support
pub fn wrap_test_with_debug(
    test_case: Box<dyn TestCase>,
    debugger: Arc<TestDebugger>,
) -> Box<dyn TestCase> {
    Box::new(DebugTestCase::new(test_case.into(), debugger))
}

/// Create debug configuration from environment variables
pub fn debug_config_from_env() -> DebugConfig {
    DebugConfig {
        enabled: std::env::var("BITNET_DEBUG_ENABLED")
            .map(|v| v.parse().unwrap_or(true))
            .unwrap_or(false),
        capture_stack_traces: std::env::var("BITNET_DEBUG_STACK_TRACES")
            .map(|v| v.parse().unwrap_or(true))
            .unwrap_or(true),
        capture_environment: std::env::var("BITNET_DEBUG_ENVIRONMENT")
            .map(|v| v.parse().unwrap_or(true))
            .unwrap_or(true),
        capture_system_info: std::env::var("BITNET_DEBUG_SYSTEM_INFO")
            .map(|v| v.parse().unwrap_or(true))
            .unwrap_or(true),
        verbose_logging: std::env::var("BITNET_DEBUG_VERBOSE")
            .map(|v| v.parse().unwrap_or(false))
            .unwrap_or(false),
        save_debug_artifacts: std::env::var("BITNET_DEBUG_ARTIFACTS")
            .map(|v| v.parse().unwrap_or(true))
            .unwrap_or(true),
        max_debug_files: std::env::var("BITNET_DEBUG_MAX_FILES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100),
        debug_output_dir: std::env::var("BITNET_DEBUG_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("tests/debug")),
    }
}

/// Quick debug setup for tests
pub async fn setup_debug_test_environment()
-> TestOpResult<(DebugEnabledTestHarness, Arc<TestDebugger>)> {
    let test_config = super::config::TestConfig::default();
    let debug_config = debug_config_from_env();

    let debug_harness = create_debug_harness(test_config, Some(debug_config)).await?;
    let debugger = debug_harness.debugger();

    Ok((debug_harness, debugger))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BYTES_PER_KB, BYTES_PER_MB, TestError};
    use std::time::Duration;

    struct MockTestCase {
        name: String,
        should_fail: bool,
    }

    impl MockTestCase {
        fn new(name: &str, should_fail: bool) -> Self {
            Self { name: name.to_string(), should_fail }
        }
    }

    #[async_trait::async_trait]
    impl TestCase for MockTestCase {
        fn name(&self) -> &str {
            &self.name
        }

        async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
            if self.should_fail && self.name.contains("setup_fail") {
                return Err(TestError::setup("Mock setup failure"));
            }
            Ok(())
        }

        async fn execute(&self) -> TestOpResult<super::super::results::TestMetrics> {
            if self.should_fail && !self.name.contains("setup_fail") {
                return Err(TestError::execution("Mock execution failure"));
            }

            Ok(super::super::results::TestMetrics {
                wall_time: Duration::from_millis(100),
                memory_peak: Some(BYTES_PER_MB),          // 1MB
                memory_average: Some(512 * BYTES_PER_KB), // 512KB
                cpu_time: Some(Duration::from_millis(50)),
                custom_metrics: std::collections::HashMap::new(),
                assertions: 10,
                operations: 100,
            })
        }

        async fn cleanup(&self) -> TestOpResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_debug_test_case_wrapper() {
        let debug_config =
            DebugConfig { enabled: true, verbose_logging: true, ..Default::default() };

        let debugger = Arc::new(TestDebugger::new(debug_config).await.unwrap());
        let mock_test = Arc::new(MockTestCase::new("test_debug_wrapper", false));
        let debug_test = DebugTestCase::new(mock_test, debugger.clone());

        // Test successful execution
        // Use Fixtures facade for consistent API
        let fixtures = crate::Fixtures::new(&Default::default()).await.unwrap();
        fixtures.ctx();
        assert!(debug_test.setup(()).await.is_ok());
        assert!(debug_test.execute().await.is_ok());
        assert!(debug_test.cleanup().await.is_ok());

        // Verify debug data was captured
        let debug_report = debugger.generate_debug_report().await.unwrap();
        assert_eq!(debug_report.total_tests, 1);
        assert_eq!(debug_report.failed_tests, 0);
    }

    #[tokio::test]
    async fn test_debug_test_case_failure() {
        let debug_config =
            DebugConfig { enabled: true, verbose_logging: true, ..Default::default() };

        let debugger = Arc::new(TestDebugger::new(debug_config).await.unwrap());
        let mock_test = Arc::new(MockTestCase::new("test_debug_failure", true));
        let debug_test = DebugTestCase::new(mock_test, debugger.clone());

        // Test failure handling
        // Use Fixtures facade for consistent API
        let fixtures = crate::Fixtures::new(&Default::default()).await.unwrap();
        fixtures.ctx();
        assert!(debug_test.setup(()).await.is_ok());
        assert!(debug_test.execute().await.is_err());
        assert!(debug_test.cleanup().await.is_ok());

        // Verify debug data captured the failure
        let debug_report = debugger.generate_debug_report().await.unwrap();
        assert_eq!(debug_report.total_tests, 1);
        assert!(debug_report.error_count > 0);
    }

    #[tokio::test]
    async fn test_debug_config_from_env() {
        // Set environment variables
        unsafe {
            std::env::set_var("BITNET_DEBUG_ENABLED", "true");
        }
        unsafe {
            std::env::set_var("BITNET_DEBUG_VERBOSE", "true");
        }
        unsafe {
            std::env::set_var("BITNET_DEBUG_MAX_FILES", "50");
        }

        let config = debug_config_from_env();

        assert!(config.enabled);
        assert!(config.verbose_logging);
        assert_eq!(config.max_debug_files, 50);

        // Clean up
        unsafe {
            std::env::remove_var("BITNET_DEBUG_ENABLED");
        }
        unsafe {
            std::env::remove_var("BITNET_DEBUG_VERBOSE");
        }
        unsafe {
            std::env::remove_var("BITNET_DEBUG_MAX_FILES");
        }
    }
}
