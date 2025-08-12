use super::config::TestConfig;
use super::errors::{TestError, TestResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Test group for parallel execution
#[derive(Debug, Clone, Default)]
pub struct TestGroup {
    pub id: usize,
    pub tests: Vec<TestInfo>,
    pub estimated_time: Duration,
}

impl TestGroup {
    pub fn estimated_duration(&self) -> Duration {
        self.estimated_time
    }
}

/// Test information
#[derive(Debug, Clone)]
pub struct TestInfo {
    pub name: String,
    pub crate_name: String,
    pub file_path: std::path::PathBuf,
    pub category: TestCategory,
    pub priority: TestPriority,
}

/// Test categories for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum TestCategory {
    Unit,
    Integration,
    Performance,
    CrossValidation,
}

/// Test priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestPriority {
    Critical = 0,
    High = 1,
    Medium = 2,
    Low = 3,
}

/// Parallel test executor optimized for speed and resource management
pub struct ParallelExecutor {
    config: TestConfig,
    semaphore: Arc<Semaphore>,
    resource_monitor: ResourceMonitor,
    execution_stats: Arc<RwLock<ExecutionStats>>,
}

impl ParallelExecutor {
    pub fn new(config: TestConfig) -> Self {
        let max_parallel = config.max_parallel_tests;

        Self {
            config: config.clone(),
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            resource_monitor: ResourceMonitor::new(config),
            execution_stats: Arc::new(RwLock::new(ExecutionStats::default())),
        }
    }

    /// Execute a test group in parallel
    pub async fn execute_test_group(
        &self,
        group: &TestGroup,
    ) -> Result<ParallelExecutionResult, TestError> {
        let start_time = Instant::now();
        info!(
            "Starting parallel execution of {} tests in group {}",
            group.tests.len(),
            group.id
        );

        // Start resource monitoring
        let _monitor_handle = self.resource_monitor.start_monitoring().await;

        // Execute tests in parallel with semaphore control
        let mut handles = Vec::new();

        for test in &group.tests {
            let test_clone = test.clone();
            let semaphore = Arc::clone(&self.semaphore);
            let config = self.config.clone();
            let stats = Arc::clone(&self.execution_stats);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                Self::execute_single_test(test_clone, config, stats).await
            });

            handles.push(handle);
        }

        // Wait for all tests to complete
        let mut test_results = Vec::new();
        let mut failed_count = 0;

        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => {
                    if !result.passed() {
                        failed_count += 1;
                    }
                    test_results.push(result);
                }
                Ok(Err(e)) => {
                    error!("Test execution failed: {}", e);
                    failed_count += 1;
                    // Create a failed test result
                    test_results.push(TestResultData::failed(
                        "unknown".to_string(),
                        e,
                        Duration::default(),
                    ));
                }
                Err(e) => {
                    error!("Task join failed: {}", e);
                    failed_count += 1;
                }
            }
        }

        let total_duration = start_time.elapsed();
        let success = failed_count == 0;

        // Update execution statistics
        {
            let mut stats = self.execution_stats.write().await;
            stats.total_tests += test_results.len();
            stats.failed_tests += failed_count;
            stats.total_duration += total_duration;
        }

        info!(
            "Group {} completed: {}/{} tests passed in {:.2}s",
            group.id,
            test_results.len() - failed_count,
            test_results.len(),
            total_duration.as_secs_f64()
        );

        Ok(ParallelExecutionResult {
            test_results,
            total_duration,
            success,
            parallel_efficiency: self.calculate_parallel_efficiency(&test_results, total_duration),
        })
    }

    /// Execute a single test with timeout and resource monitoring
    async fn execute_single_test(
        test: TestInfo,
        config: TestConfig,
        stats: Arc<RwLock<ExecutionStats>>,
    ) -> Result<TestResultData, TestError> {
        let start_time = Instant::now();
        debug!("Executing test: {}", test.name);

        // Build cargo test command for specific test
        let mut cmd = Command::new("cargo");
        cmd.arg("test")
            .arg("--package")
            .arg(&test.crate_name)
            .arg(&test.name)
            .arg("--")
            .arg("--exact")
            .arg("--nocapture");

        // Set environment variables for optimization
        cmd.env("RUST_BACKTRACE", "0")
            .env("BITNET_TEST_MODE", "fast")
            .env("BITNET_LOG_LEVEL", &config.log_level);

        // Execute with timeout
        let timeout_duration = config.test_timeout;
        let result = tokio::time::timeout(timeout_duration, cmd.output()).await;

        let duration = start_time.elapsed();

        match result {
            Ok(Ok(output)) => {
                let success = output.status.success();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                // Update stats
                {
                    let mut stats_guard = stats.write().await;
                    if success {
                        stats_guard.passed_tests += 1;
                    } else {
                        stats_guard.failed_tests += 1;
                    }
                }

                if success {
                    debug!(
                        "Test {} passed in {:.2}s",
                        test.name,
                        duration.as_secs_f64()
                    );
                    Ok(TestResultData::passed(
                        test.name,
                        Default::default(),
                        duration,
                    ))
                } else {
                    warn!(
                        "Test {} failed in {:.2}s",
                        test.name,
                        duration.as_secs_f64()
                    );
                    let error_msg = if !stderr.is_empty() {
                        stderr.to_string()
                    } else {
                        stdout.to_string()
                    };
                    Ok(TestResultData::failed(
                        test.name,
                        TestError::execution(error_msg),
                        duration,
                    ))
                }
            }
            Ok(Err(e)) => {
                error!("Failed to execute test {}: {}", test.name, e);
                Ok(TestResultData::failed(
                    test.name,
                    TestError::execution(e.to_string()),
                    duration,
                ))
            }
            Err(_) => {
                warn!(
                    "Test {} timed out after {:.2}s",
                    test.name,
                    timeout_duration.as_secs_f64()
                );
                Ok(TestResultData::failed(
                    test.name,
                    TestError::TimeoutError {
                        timeout: timeout_duration,
                    },
                    timeout_duration,
                ))
            }
        }
    }

    /// Calculate parallel execution efficiency
    fn calculate_parallel_efficiency(
        &self,
        results: &[TestResultData],
        total_duration: Duration,
    ) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let sequential_time: Duration = results.iter().map(|r| r.duration).sum();
        let parallel_time = total_duration;

        if parallel_time.as_secs_f64() > 0.0 {
            sequential_time.as_secs_f64()
                / (parallel_time.as_secs_f64() * self.config.max_parallel_tests as f64)
        } else {
            0.0
        }
    }

    /// Get current execution statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.execution_stats.read().await.clone()
    }
}

/// Resource monitor to prevent system overload during parallel execution
pub struct ResourceMonitor {
    config: TestConfig,
    cpu_threshold: f64,
    memory_threshold: f64,
}

impl ResourceMonitor {
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            cpu_threshold: 0.95,    // 95% CPU usage threshold
            memory_threshold: 0.90, // 90% memory usage threshold
        }
    }

    /// Start resource monitoring (placeholder for now)
    pub async fn start_monitoring(&self) -> ResourceMonitorHandle {
        // In a real implementation, this would start a background task
        // that monitors CPU and memory usage and can signal to reduce
        // parallel execution if resources are constrained
        ResourceMonitorHandle::new()
    }

    /// Check if system resources allow for more parallel execution
    pub async fn can_start_more_tests(&self) -> bool {
        // Placeholder implementation
        // In reality, this would check actual CPU and memory usage
        true
    }
}

/// Handle for resource monitoring
pub struct ResourceMonitorHandle {
    _handle: tokio::task::JoinHandle<()>,
}

impl ResourceMonitorHandle {
    fn new() -> Self {
        let handle = tokio::spawn(async {
            // Placeholder monitoring task
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        Self { _handle: handle }
    }
}

/// Result of parallel execution
#[derive(Debug)]
pub struct ParallelExecutionResult {
    pub test_results: Vec<TestResult>,
    pub total_duration: Duration,
    pub success: bool,
    pub parallel_efficiency: f64,
}

impl ParallelExecutionResult {
    pub fn all_passed(&self) -> bool {
        self.success
    }

    pub fn passed_count(&self) -> usize {
        self.test_results.iter().filter(|r| r.passed()).count()
    }

    pub fn failed_count(&self) -> usize {
        self.test_results.iter().filter(|r| !r.passed()).count()
    }
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub total_duration: Duration,
    pub average_test_duration: Duration,
}

impl ExecutionStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_tests > 0 {
            self.passed_tests as f64 / self.total_tests as f64
        } else {
            0.0
        }
    }

    pub fn update_average_duration(&mut self) {
        if self.total_tests > 0 {
            self.average_test_duration = Duration::from_nanos(
                self.total_duration.as_nanos() as u64 / self.total_tests as u64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::fast_config::fast_config;

    #[tokio::test]
    async fn test_parallel_executor_creation() {
        let config = fast_config();
        let executor = ParallelExecutor::new(config);

        let stats = executor.get_stats().await;
        assert_eq!(stats.total_tests, 0);
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::default();
        stats.total_tests = 10;
        stats.passed_tests = 8;
        stats.failed_tests = 2;

        assert_eq!(stats.success_rate(), 0.8);
    }

    #[tokio::test]
    async fn test_resource_monitor() {
        let config = fast_config();
        let monitor = ResourceMonitor::new(config);

        let can_start = monitor.can_start_more_tests().await;
        assert!(can_start); // Should be true in test environment
    }
}
