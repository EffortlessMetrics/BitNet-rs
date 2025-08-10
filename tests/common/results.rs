use super::errors::{TestError, TestResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Status of a test execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed successfully
    Passed,
    /// Test failed with an error
    Failed,
    /// Test was skipped
    Skipped,
    /// Test exceeded timeout
    Timeout,
    /// Test is currently running
    Running,
}

impl TestStatus {
    /// Check if the test was successful
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Check if the test failed
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed | Self::Timeout)
    }

    /// Check if the test was not executed
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped)
    }
}

/// Metrics collected during test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    /// Peak memory usage in bytes
    pub memory_peak: Option<u64>,
    /// Average memory usage in bytes
    pub memory_average: Option<u64>,
    /// CPU time consumed
    pub cpu_time: Option<Duration>,
    /// Wall clock time
    pub wall_time: Duration,
    /// Custom metrics specific to the test
    pub custom_metrics: HashMap<String, f64>,
    /// Number of assertions made
    pub assertions: usize,
    /// Number of operations performed
    pub operations: usize,
}

impl Default for TestMetrics {
    fn default() -> Self {
        Self {
            memory_peak: None,
            memory_average: None,
            cpu_time: None,
            wall_time: Duration::ZERO,
            custom_metrics: HashMap::new(),
            assertions: 0,
            operations: 0,
        }
    }
}

impl TestMetrics {
    /// Create new metrics with wall time
    pub fn with_duration(duration: Duration) -> Self {
        Self {
            wall_time: duration,
            ..Default::default()
        }
    }

    /// Add a custom metric
    pub fn add_metric<K: Into<String>>(&mut self, key: K, value: f64) {
        self.custom_metrics.insert(key.into(), value);
    }

    /// Get a custom metric
    pub fn get_metric(&self, key: &str) -> Option<f64> {
        self.custom_metrics.get(key).copied()
    }

    /// Increment assertion count
    pub fn add_assertion(&mut self) {
        self.assertions += 1;
    }

    /// Increment operation count
    pub fn add_operation(&mut self) {
        self.operations += 1;
    }

    /// Set memory usage
    pub fn set_memory_usage(&mut self, peak: u64, average: Option<u64>) {
        self.memory_peak = Some(peak);
        self.memory_average = average;
    }

    /// Set CPU time
    pub fn set_cpu_time(&mut self, cpu_time: Duration) {
        self.cpu_time = Some(cpu_time);
    }
}

/// Artifact generated during test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestArtifact {
    /// Name of the artifact
    pub name: String,
    /// Path to the artifact file
    pub path: String,
    /// MIME type of the artifact
    pub mime_type: String,
    /// Size of the artifact in bytes
    pub size: u64,
    /// Description of the artifact
    pub description: Option<String>,
}

impl TestArtifact {
    /// Create a new artifact
    pub fn new<S1: Into<String>, S2: Into<String>, S3: Into<String>>(
        name: S1,
        path: S2,
        mime_type: S3,
        size: u64,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            mime_type: mime_type.into(),
            size,
            description: None,
        }
    }

    /// Set description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Result of a single test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult<T = ()> {
    /// Name of the test
    pub test_name: String,
    /// Status of the test
    pub status: TestStatus,
    /// Duration of the test execution
    pub duration: Duration,
    /// Metrics collected during execution
    pub metrics: TestMetrics,
    /// Error information if the test failed
    pub error: Option<String>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Artifacts generated during test
    pub artifacts: Vec<TestArtifact>,
    /// Timestamp when the test started
    pub start_time: SystemTime,
    /// Timestamp when the test completed
    pub end_time: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TestResult {
    /// Create a passed test result
    pub fn passed<S: Into<String>>(test_name: S, metrics: TestMetrics, duration: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            test_name: test_name.into(),
            status: TestStatus::Passed,
            duration,
            metrics,
            error: None,
            stack_trace: None,
            artifacts: Vec::new(),
            start_time: now - duration,
            end_time: now,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed test result
    pub fn failed<S: Into<String>>(test_name: S, error: TestError, duration: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            test_name: test_name.into(),
            status: TestStatus::Failed,
            duration,
            metrics: TestMetrics::with_duration(duration),
            error: Some(error.to_string()),
            stack_trace: None, // TODO: Extract stack trace from error
            artifacts: Vec::new(),
            start_time: now - duration,
            end_time: now,
            metadata: HashMap::new(),
        }
    }

    /// Create a skipped test result
    pub fn skipped<S: Into<String>>(test_name: S, reason: Option<String>) -> Self {
        let now = SystemTime::now();
        Self {
            test_name: test_name.into(),
            status: TestStatus::Skipped,
            duration: Duration::ZERO,
            metrics: TestMetrics::default(),
            error: reason,
            stack_trace: None,
            artifacts: Vec::new(),
            start_time: now,
            end_time: now,
            metadata: HashMap::new(),
        }
    }

    /// Create a timeout test result
    pub fn timeout<S: Into<String>>(test_name: S, timeout_duration: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            test_name: test_name.into(),
            status: TestStatus::Timeout,
            duration: timeout_duration,
            metrics: TestMetrics::with_duration(timeout_duration),
            error: Some(format!("Test exceeded timeout of {:?}", timeout_duration)),
            stack_trace: None,
            artifacts: Vec::new(),
            start_time: now - timeout_duration,
            end_time: now,
            metadata: HashMap::new(),
        }
    }

    /// Add an artifact to the test result
    pub fn add_artifact(&mut self, artifact: TestArtifact) {
        self.artifacts.push(artifact);
    }

    /// Add metadata to the test result
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Check if the test was successful
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if the test failed
    pub fn is_failure(&self) -> bool {
        self.status.is_failure()
    }
}

/// Summary statistics for a test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Number of skipped tests
    pub skipped: usize,
    /// Number of timed out tests
    pub timeout: usize,
    /// Success rate as a percentage
    pub success_rate: f64,
    /// Total duration of all tests
    pub total_duration: Duration,
    /// Average test duration
    pub average_duration: Duration,
    /// Peak memory usage across all tests
    pub peak_memory: Option<u64>,
    /// Total number of assertions
    pub total_assertions: usize,
}

impl TestSummary {
    /// Calculate summary from test results
    pub fn from_results(results: &[TestResult]) -> Self {
        let total_tests = results.len();
        let passed = results
            .iter()
            .filter(|r| r.status == TestStatus::Passed)
            .count();
        let failed = results
            .iter()
            .filter(|r| r.status == TestStatus::Failed)
            .count();
        let skipped = results
            .iter()
            .filter(|r| r.status == TestStatus::Skipped)
            .count();
        let timeout = results
            .iter()
            .filter(|r| r.status == TestStatus::Timeout)
            .count();

        let success_rate = if total_tests > 0 {
            (passed as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };

        let total_duration = results.iter().map(|r| r.duration).sum();
        let average_duration = if total_tests > 0 {
            total_duration / total_tests as u32
        } else {
            Duration::ZERO
        };

        let peak_memory = results.iter().filter_map(|r| r.metrics.memory_peak).max();

        let total_assertions = results.iter().map(|r| r.metrics.assertions).sum();

        Self {
            total_tests,
            passed,
            failed,
            skipped,
            timeout,
            success_rate,
            total_duration,
            average_duration,
            peak_memory,
            total_assertions,
        }
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.timeout == 0
    }

    /// Get failure rate as a percentage
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate
    }
}

/// Result of a complete test suite execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteResult {
    /// Name of the test suite
    pub suite_name: String,
    /// Total duration of the suite execution
    pub total_duration: Duration,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Summary statistics
    pub summary: TestSummary,
    /// Environment information
    pub environment: HashMap<String, String>,
    /// Configuration used for the test suite
    pub configuration: HashMap<String, String>,
    /// Timestamp when the suite started
    pub start_time: SystemTime,
    /// Timestamp when the suite completed
    pub end_time: SystemTime,
}

impl TestSuiteResult {
    /// Create a new test suite result
    pub fn new<S: Into<String>>(
        suite_name: S,
        test_results: Vec<TestResult>,
        total_duration: Duration,
    ) -> Self {
        let summary = TestSummary::from_results(&test_results);
        let now = SystemTime::now();

        Self {
            suite_name: suite_name.into(),
            total_duration,
            test_results,
            summary,
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: now - total_duration,
            end_time: now,
        }
    }

    /// Add environment information
    pub fn add_environment<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.environment.insert(key.into(), value.into());
    }

    /// Add configuration information
    pub fn add_configuration<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.configuration.insert(key.into(), value.into());
    }

    /// Get all failed tests
    pub fn failed_tests(&self) -> Vec<&TestResult> {
        self.test_results
            .iter()
            .filter(|r| r.is_failure())
            .collect()
    }

    /// Get all passed tests
    pub fn passed_tests(&self) -> Vec<&TestResult> {
        self.test_results
            .iter()
            .filter(|r| r.is_success())
            .collect()
    }

    /// Check if the entire suite passed
    pub fn is_success(&self) -> bool {
        self.summary.all_passed()
    }
}
