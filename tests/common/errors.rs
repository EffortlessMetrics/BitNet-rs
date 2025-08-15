use std::time::Duration;
use thiserror::Error;

// Removed duplicate TestResult type alias - using the one defined later

/// Comprehensive error types for the testing framework
#[derive(Debug, Error)]
pub enum TestError {
    #[error("Test setup failed: {message}")]
    SetupError { message: String },

    #[error("Test execution failed: {message}")]
    ExecutionError { message: String },

    #[error("Test timeout after {timeout:?}")]
    TimeoutError { timeout: Duration },

    #[error("Assertion failed: {message}")]
    AssertionError { message: String },

    #[error("Fixture error: {0}")]
    FixtureError(#[from] FixtureError),

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("IO error: {0}")]
    IoError(std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    #[error("Formatting error: {0}")]
    Fmt(#[from] std::fmt::Error),
}

/// Errors related to fixture management
#[derive(Debug, Clone, Error)]
pub enum FixtureError {
    #[error("Unknown fixture: {name}")]
    UnknownFixture { name: String },

    #[error("Download failed for {url}: {reason}")]
    DownloadError { url: String, reason: String },

    #[error("Checksum mismatch for {filename}: expected {expected}, got {actual}")]
    ChecksumMismatch { filename: String, expected: String, actual: String },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    #[error("Fixture validation failed: {message}")]
    ValidationError { message: String },

    #[error("Fixture not found: {path}")]
    NotFound { path: String },
}

/// Errors related to cross-implementation comparison
#[derive(Debug, Clone, Error)]
pub enum ComparisonError {
    #[error("Implementation error: {0}")]
    ImplementationError(#[from] ImplementationError),

    #[error("Accuracy comparison failed: {message}")]
    AccuracyError { message: String },

    #[error("Performance comparison failed: {message}")]
    PerformanceError { message: String },

    #[error("Tolerance exceeded: {metric} = {value}, threshold = {threshold}")]
    ToleranceExceeded { metric: String, value: f64, threshold: f64 },
}

/// Errors from BitNet implementations
#[derive(Debug, Clone, Error)]
pub enum ImplementationError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Model load error: {message}")]
    ModelLoadError { message: String },

    #[error("Tokenization error: {message}")]
    TokenizationError { message: String },

    #[error("Inference error: {message}")]
    InferenceError { message: String },

    #[error("Implementation not available: {name}")]
    NotAvailable { name: String },

    #[error("FFI error: {message}")]
    FfiError { message: String },
}

/// Result type for test operations (using different name to avoid conflict with results::TestResult)
pub type TestOpResult<T> = Result<T, TestError>;

/// Result type for fixture operations
pub type FixtureResult<T> = Result<T, FixtureError>;

/// Result type for comparison operations
pub type ComparisonResult<T> = Result<T, ComparisonError>;

/// Result type for implementation operations
pub type ImplementationResult<T> = Result<T, ImplementationError>;

/// Error severity levels for prioritizing debugging efforts
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Debugging information for errors
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorDebugInfo {
    pub error_type: String,
    pub category: String,
    pub severity: ErrorSeverity,
    pub recoverable: bool,
    pub recovery_suggestions: Vec<String>,
    pub related_components: Vec<String>,
    pub troubleshooting_steps: Vec<TroubleshootingStep>,
}

/// A single troubleshooting step
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TroubleshootingStep {
    pub step_number: u32,
    pub title: String,
    pub description: String,
    pub estimated_time: Option<String>,
    pub required_tools: Vec<String>,
}

impl TroubleshootingStep {
    pub fn new<S1: Into<String>, S2: Into<String>>(
        step_number: u32,
        title: S1,
        description: S2,
    ) -> Self {
        Self {
            step_number,
            title: title.into(),
            description: description.into(),
            estimated_time: None,
            required_tools: Vec::new(),
        }
    }

    pub fn with_time<S: Into<String>>(mut self, time: S) -> Self {
        self.estimated_time = Some(time.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.required_tools = tools;
        self
    }
}

/// Comprehensive error report for debugging
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorReport {
    pub timestamp: std::time::SystemTime,
    pub error_message: String,
    pub error_category: String,
    pub severity: ErrorSeverity,
    pub debug_info: ErrorDebugInfo,
    pub stack_trace: Option<String>,
    pub environment_info: EnvironmentInfo,
}

impl ErrorReport {
    /// Save error report to file
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        tokio::fs::write(path, json).await
    }

    /// Generate human-readable error summary
    pub fn generate_summary(&self) -> String {
        format!(
            "ERROR REPORT - {} Severity\n\
            =====================================\n\
            Category: {}\n\
            Message: {}\n\
            Timestamp: {:?}\n\n\
            RECOVERY SUGGESTIONS:\n\
            {}\n\n\
            TROUBLESHOOTING STEPS:\n\
            {}\n\n\
            RELATED COMPONENTS:\n\
            - {}\n",
            self.severity,
            self.error_category,
            self.error_message,
            self.timestamp,
            self.debug_info
                .recovery_suggestions
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n"),
            self.debug_info
                .troubleshooting_steps
                .iter()
                .map(|step| format!("{}. {} - {}", step.step_number, step.title, step.description))
                .collect::<Vec<_>>()
                .join("\n"),
            self.debug_info.related_components.join("\n- ")
        )
    }
}

/// Environment information for debugging
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    pub platform: String,
    pub architecture: String,
    pub rust_version: String,
    pub test_framework_version: String,
    pub working_directory: String,
    pub environment_variables: std::collections::HashMap<String, String>,
    pub system_resources: SystemResources,
}

/// System resource information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemResources {
    pub available_memory: u64,
    pub cpu_cores: usize,
    pub disk_space: u64,
    pub load_average: Option<f64>,
}

/// Capture current stack trace for debugging
pub fn capture_current_stack_trace() -> Option<String> {
    // In a production implementation, you would use the `backtrace` crate
    // For now, we'll provide a simple implementation
    Some(format!(
        "Stack trace captured at {}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    ))
}

/// Collect environment information for debugging
pub fn collect_environment_info() -> EnvironmentInfo {
    let mut env_vars = std::collections::HashMap::new();

    // Collect relevant environment variables
    for (key, value) in std::env::vars() {
        if key.starts_with("RUST_")
            || key.starts_with("CARGO_")
            || key.starts_with("BITNET_")
            || key == "CI"
            || key == "PATH"
        {
            env_vars.insert(key, value);
        }
    }

    EnvironmentInfo {
        platform: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
        rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        test_framework_version: env!("CARGO_PKG_VERSION").to_string(),
        working_directory: std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
        environment_variables: env_vars,
        system_resources: collect_system_resources(),
    }
}

/// Collect system resource information
pub fn collect_system_resources() -> SystemResources {
    SystemResources {
        available_memory: get_available_memory(),
        cpu_cores: num_cpus::get(),
        disk_space: get_available_disk_space(),
        load_average: get_load_average(),
    }
}

/// Get available memory in bytes
fn get_available_memory() -> u64 {
    // This is a simplified implementation
    // In production, you'd use a proper system info crate
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    // Fallback: return 0 if we can't determine
    0
}

/// Get available disk space in bytes
fn get_available_disk_space() -> u64 {
    // Simplified implementation - in production use a proper system info crate
    0
}

/// Get system load average
fn get_load_average() -> Option<f64> {
    // Simplified implementation - in production use a proper system info crate
    None
}

impl From<std::io::Error> for TestError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

// Note: tokio::io::Error is an alias for std::io::Error, so we don't need a separate impl

impl From<tokio::io::Error> for FixtureError {
    fn from(err: tokio::io::Error) -> Self {
        Self::CacheError { message: err.to_string() }
    }
}

impl TestError {
    /// Create a setup error with a message
    pub fn setup<S: Into<String>>(message: S) -> Self {
        Self::SetupError { message: message.into() }
    }

    /// Create an execution error with a message
    pub fn execution<S: Into<String>>(message: S) -> Self {
        Self::ExecutionError { message: message.into() }
    }

    /// Create an assertion error with a message
    pub fn assertion<S: Into<String>>(message: S) -> Self {
        Self::AssertionError { message: message.into() }
    }

    /// Create a configuration error with a message
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError { message: message.into() }
    }

    /// Create a timeout error with a duration
    pub fn timeout(timeout: Duration) -> Self {
        Self::TimeoutError { timeout }
    }

    /// Create a cache error (alias for config error for backward compatibility)
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::ConfigError { message: message.into() }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::TimeoutError { .. } => true,
            Self::IoError(_) => true,
            Self::HttpError(_) => true,
            Self::FixtureError(FixtureError::DownloadError { .. }) => true,
            _ => false,
        }
    }

    /// Get error category for reporting
    pub fn category(&self) -> &'static str {
        match self {
            Self::SetupError { .. } => "setup",
            Self::ExecutionError { .. } => "execution",
            Self::TimeoutError { .. } => "timeout",
            Self::AssertionError { .. } => "assertion",
            Self::FixtureError(_) => "fixture",
            Self::ConfigError { .. } => "config",
            Self::IoError(_) => "io",
            Self::SerializationError(_) => "serialization",
            Self::HttpError(_) => "http",
            Self::JoinError(_) => "concurrency",
            Self::Fmt(_) => "formatting",
        }
    }

    /// Get severity level of the error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::AssertionError { .. } => ErrorSeverity::High,
            Self::ExecutionError { .. } => ErrorSeverity::High,
            Self::TimeoutError { .. } => ErrorSeverity::Medium,
            Self::SetupError { .. } => ErrorSeverity::Medium,
            Self::FixtureError(_) => ErrorSeverity::Medium,
            Self::ConfigError { .. } => ErrorSeverity::Low,
            Self::IoError(_) => ErrorSeverity::Low,
            Self::SerializationError(_) => ErrorSeverity::Low,
            Self::HttpError(_) => ErrorSeverity::Low,
            Self::JoinError(_) => ErrorSeverity::Medium,
            Self::Fmt(_) => ErrorSeverity::Low,
        }
    }

    /// Get suggested recovery actions for this error
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            Self::TimeoutError { timeout } => vec![
                format!("Increase timeout from {:?} to a higher value", timeout),
                "Check if the operation is hanging or taking longer than expected".to_string(),
                "Consider running the test in isolation to check for resource contention"
                    .to_string(),
            ],
            Self::IoError(io_err) => match io_err.kind() {
                std::io::ErrorKind::NotFound => vec![
                    "Verify that the required file or directory exists".to_string(),
                    "Check file paths and ensure they are correct".to_string(),
                    "Ensure test fixtures are properly downloaded".to_string(),
                ],
                std::io::ErrorKind::PermissionDenied => vec![
                    "Check file permissions and ensure the test has read/write access".to_string(),
                    "Run tests with appropriate privileges if needed".to_string(),
                ],
                std::io::ErrorKind::AlreadyExists => vec![
                    "Clean up existing files before running the test".to_string(),
                    "Use unique temporary directories for each test run".to_string(),
                ],
                _ => vec![
                    "Check system resources and disk space".to_string(),
                    "Verify file system integrity".to_string(),
                ],
            },
            Self::HttpError(http_err) => {
                let mut suggestions = vec![
                    "Check network connectivity".to_string(),
                    "Verify that the target server is accessible".to_string(),
                ];
                if http_err.is_timeout() {
                    suggestions.push("Increase HTTP timeout settings".to_string());
                }
                if http_err.is_connect() {
                    suggestions.push("Check firewall settings and proxy configuration".to_string());
                }
                suggestions
            }
            Self::FixtureError(fixture_err) => match fixture_err {
                FixtureError::DownloadError { .. } => vec![
                    "Check network connectivity for fixture downloads".to_string(),
                    "Verify fixture URLs are accessible".to_string(),
                    "Consider using cached fixtures if available".to_string(),
                ],
                FixtureError::ChecksumMismatch { .. } => vec![
                    "Re-download the fixture to ensure integrity".to_string(),
                    "Check if the fixture source has been updated".to_string(),
                    "Clear fixture cache and retry".to_string(),
                ],
                FixtureError::NotFound { .. } => vec![
                    "Ensure fixture is properly configured in test setup".to_string(),
                    "Check fixture registry for available fixtures".to_string(),
                    "Verify fixture download completed successfully".to_string(),
                ],
                _ => vec![
                    "Check fixture configuration and availability".to_string(),
                    "Clear fixture cache and retry".to_string(),
                ],
            },
            Self::ConfigError { .. } => vec![
                "Review test configuration file for syntax errors".to_string(),
                "Check environment variables and their values".to_string(),
                "Validate configuration against the schema".to_string(),
                "Use default configuration as a starting point".to_string(),
            ],
            Self::SetupError { .. } => vec![
                "Check test prerequisites and dependencies".to_string(),
                "Verify test environment is properly initialized".to_string(),
                "Review setup steps for missing components".to_string(),
            ],
            Self::ExecutionError { .. } => vec![
                "Review test logic and implementation".to_string(),
                "Check for race conditions in concurrent tests".to_string(),
                "Verify test data and inputs are valid".to_string(),
                "Run test in isolation to identify dependencies".to_string(),
            ],
            Self::AssertionError { .. } => vec![
                "Review expected vs actual values".to_string(),
                "Check test data for correctness".to_string(),
                "Verify implementation matches expected behavior".to_string(),
                "Consider updating test expectations if behavior changed".to_string(),
            ],
            Self::JoinError(_) => vec![
                "Check for panics in concurrent test execution".to_string(),
                "Review thread safety of shared resources".to_string(),
                "Consider reducing parallelism to isolate issues".to_string(),
            ],
            Self::SerializationError(_) => vec![
                "Check data format and structure".to_string(),
                "Verify serialization schema compatibility".to_string(),
                "Review data types and field names".to_string(),
            ],
            Self::Fmt(_) => vec![
                "Check string formatting operations".to_string(),
                "Review format string syntax".to_string(),
                "Verify format arguments match placeholders".to_string(),
            ],
        }
    }

    /// Get debugging information for this error
    pub fn debug_info(&self) -> ErrorDebugInfo {
        ErrorDebugInfo {
            error_type: std::any::type_name::<Self>().to_string(),
            category: self.category().to_string(),
            severity: self.severity(),
            recoverable: self.is_recoverable(),
            recovery_suggestions: self.recovery_suggestions(),
            related_components: self.related_components(),
            troubleshooting_steps: self.troubleshooting_steps(),
        }
    }

    /// Get components related to this error
    pub fn related_components(&self) -> Vec<String> {
        match self {
            Self::FixtureError(_) => vec![
                "fixture_manager".to_string(),
                "download_system".to_string(),
                "cache_system".to_string(),
            ],
            Self::ConfigError { .. } => {
                vec!["config_manager".to_string(), "environment_loader".to_string()]
            }
            Self::TimeoutError { .. } => {
                vec!["test_harness".to_string(), "execution_engine".to_string()]
            }
            Self::HttpError(_) => vec!["http_client".to_string(), "network_layer".to_string()],
            Self::IoError(_) => vec!["file_system".to_string(), "storage_layer".to_string()],
            Self::JoinError(_) => vec!["parallel_executor".to_string(), "thread_pool".to_string()],
            Self::Fmt(_) => vec!["formatting_system".to_string(), "output_writer".to_string()],
            _ => vec!["test_framework".to_string()],
        }
    }

    /// Get step-by-step troubleshooting guide
    pub fn troubleshooting_steps(&self) -> Vec<TroubleshootingStep> {
        match self {
            Self::TimeoutError { timeout } => vec![
                TroubleshootingStep::new(
                    1,
                    "Check Test Duration",
                    format!("Review if the test should complete within {:?}", timeout),
                ),
                TroubleshootingStep::new(
                    2,
                    "Monitor Resource Usage",
                    "Check CPU and memory usage during test execution".to_string(),
                ),
                TroubleshootingStep::new(
                    3,
                    "Run in Isolation",
                    "Execute the test alone to check for external dependencies".to_string(),
                ),
                TroubleshootingStep::new(
                    4,
                    "Increase Timeout",
                    "Temporarily increase timeout to see if test completes".to_string(),
                ),
            ],
            Self::FixtureError(FixtureError::DownloadError { url, .. }) => vec![
                TroubleshootingStep::new(
                    1,
                    "Test Network Connectivity",
                    format!("Verify you can access: {}", url),
                ),
                TroubleshootingStep::new(
                    2,
                    "Check Proxy Settings",
                    "Ensure HTTP proxy is configured correctly if needed".to_string(),
                ),
                TroubleshootingStep::new(
                    3,
                    "Verify URL",
                    "Check if the fixture URL is still valid and accessible".to_string(),
                ),
                TroubleshootingStep::new(
                    4,
                    "Clear Cache",
                    "Remove cached fixtures and retry download".to_string(),
                ),
            ],
            Self::AssertionError { message } => vec![
                TroubleshootingStep::new(
                    1,
                    "Review Assertion",
                    format!("Examine the failed assertion: {}", message),
                ),
                TroubleshootingStep::new(
                    2,
                    "Check Test Data",
                    "Verify input data and expected outcomes are correct".to_string(),
                ),
                TroubleshootingStep::new(
                    3,
                    "Debug Values",
                    "Add debug logging to see actual vs expected values".to_string(),
                ),
                TroubleshootingStep::new(
                    4,
                    "Review Implementation",
                    "Check if the code behavior has changed".to_string(),
                ),
            ],
            _ => vec![
                TroubleshootingStep::new(
                    1,
                    "Review Error Details",
                    "Examine the error message and context carefully".to_string(),
                ),
                TroubleshootingStep::new(
                    2,
                    "Check Logs",
                    "Review test execution logs for additional context".to_string(),
                ),
                TroubleshootingStep::new(
                    3,
                    "Reproduce Locally",
                    "Try to reproduce the error in a local environment".to_string(),
                ),
            ],
        }
    }

    /// Create a detailed error report for debugging
    pub fn create_error_report(&self) -> ErrorReport {
        ErrorReport {
            timestamp: std::time::SystemTime::now(),
            error_message: self.to_string(),
            error_category: self.category().to_string(),
            severity: self.severity(),
            debug_info: self.debug_info(),
            stack_trace: capture_current_stack_trace(),
            environment_info: collect_environment_info(),
        }
    }
}

impl FixtureError {
    /// Create an unknown fixture error
    pub fn unknown<S: Into<String>>(name: S) -> Self {
        Self::UnknownFixture { name: name.into() }
    }

    /// Create a download error
    pub fn download<S1: Into<String>, S2: Into<String>>(url: S1, reason: S2) -> Self {
        Self::DownloadError { url: url.into(), reason: reason.into() }
    }

    /// Create a checksum mismatch error
    pub fn checksum_mismatch<S1: Into<String>, S2: Into<String>, S3: Into<String>>(
        filename: S1,
        expected: S2,
        actual: S3,
    ) -> Self {
        Self::ChecksumMismatch {
            filename: filename.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::CacheError { message: message.into() }
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::ValidationError { message: message.into() }
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(path: S) -> Self {
        Self::NotFound { path: path.into() }
    }
}
