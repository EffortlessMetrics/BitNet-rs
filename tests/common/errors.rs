use std::time::Duration;
use thiserror::Error;

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
}

/// Errors related to fixture management
#[derive(Debug, Clone, Error)]
pub enum FixtureError {
    #[error("Unknown fixture: {name}")]
    UnknownFixture { name: String },

    #[error("Download failed for {url}: {reason}")]
    DownloadError { url: String, reason: String },

    #[error("Checksum mismatch for {filename}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        filename: String,
        expected: String,
        actual: String,
    },

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
    ToleranceExceeded {
        metric: String,
        value: f64,
        threshold: f64,
    },
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

/// Result type for test operations
pub type TestResult<T> = Result<T, TestError>;

/// Result type for fixture operations
pub type FixtureResult<T> = Result<T, FixtureError>;

/// Result type for comparison operations
pub type ComparisonResult<T> = Result<T, ComparisonError>;

/// Result type for implementation operations
pub type ImplementationResult<T> = Result<T, ImplementationError>;

impl From<std::io::Error> for TestError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

// Note: tokio::io::Error is an alias for std::io::Error, so we don't need a separate impl

impl From<tokio::io::Error> for FixtureError {
    fn from(err: tokio::io::Error) -> Self {
        Self::CacheError {
            message: err.to_string(),
        }
    }
}

impl TestError {
    /// Create a setup error with a message
    pub fn setup<S: Into<String>>(message: S) -> Self {
        Self::SetupError {
            message: message.into(),
        }
    }

    /// Create an execution error with a message
    pub fn execution<S: Into<String>>(message: S) -> Self {
        Self::ExecutionError {
            message: message.into(),
        }
    }

    /// Create an assertion error with a message
    pub fn assertion<S: Into<String>>(message: S) -> Self {
        Self::AssertionError {
            message: message.into(),
        }
    }

    /// Create a configuration error with a message
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Create a timeout error with a duration
    pub fn timeout(timeout: Duration) -> Self {
        Self::TimeoutError { timeout }
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
        Self::DownloadError {
            url: url.into(),
            reason: reason.into(),
        }
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
        Self::CacheError {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::ValidationError {
            message: message.into(),
        }
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(path: S) -> Self {
        Self::NotFound { path: path.into() }
    }
}
