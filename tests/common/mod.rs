// Common testing utilities and infrastructure for BitNet.rs
//
// This module provides the core testing framework including:
// - Test harness for parallel execution and result collection
// - Fixture management for test data and models
// - Configuration management for test environments
// - Error handling and reporting utilities

pub mod config;
pub mod config_validator;
pub mod errors;
pub mod fixtures;
pub mod harness;
pub mod logging;
pub mod results;
pub mod utils;

// Re-export commonly used types
pub use config::{ci_config, dev_config, load_test_config, validate_config, TestConfig};
pub use config_validator::{ConfigValidator, ValidationResult};
pub use errors::{ComparisonError, FixtureError, ImplementationError, TestError};
pub use fixtures::{FixtureInfo, FixtureManager, ModelFormat, ModelType};
pub use harness::{ConsoleReporter, TestCase, TestHarness, TestReporter, TestSuite};
pub use logging::{
    init_logging, DebugContext, PerformanceProfiler, TestTracer, TraceEvent, TraceEventType,
};
pub use results::{
    TestArtifact, TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary,
};
pub use utils::{
    format_bytes, format_duration, get_memory_usage, get_peak_memory_usage, measure_time,
};

/// Current version of the testing framework
pub const TESTING_FRAMEWORK_VERSION: &str = "0.1.0";

/// Default test timeout in seconds
pub const DEFAULT_TEST_TIMEOUT_SECS: u64 = 300;

/// Default maximum parallel tests
pub const DEFAULT_MAX_PARALLEL_TESTS: usize = 4;
