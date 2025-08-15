// Common testing utilities and infrastructure for BitNet.rs
//
// This module provides the core testing framework including:
// - Test harness for parallel execution and result collection
// - Fixture management for test data and models
// - Configuration management for test environments
// - Error handling and reporting utilities

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(dead_code)]

// pub mod cache;
// pub mod ci_reporting;
pub mod config;
// pub mod config_validator;
// pub mod enhanced_error_handler;
// pub mod error_analysis;
pub mod errors;
// pub mod execution_optimizer;
// pub mod fast_config;
// pub mod fixtures;
// pub mod github_cache;
// pub mod harness;
// pub mod incremental;
// pub mod logging;
// pub mod logging_example;
// pub mod optimization;
// pub mod parallel;
pub mod reporting;
pub mod results;
// pub mod selection;
// pub mod test_utilities;
// pub mod trend_reporting;
pub mod utils;

// Re-export commonly used functions
pub use utils::{format_bytes, format_duration, get_memory_usage, get_peak_memory_usage};

// Cross-validation module temporarily disabled
// pub mod cross_validation;

// Data module - temporarily disabled due to compilation issues
// #[path = "../data/mod.rs"]
// pub mod data;

// Integration tests module - temporarily disabled due to compilation issues
// #[path = "../integration/mod.rs"]
// pub mod integration;

// Re-export commonly used types
pub use config::TestConfig;
pub use errors::TestError;
pub use results::TestMetrics;

// Cross-validation types temporarily disabled
// pub use cross_validation::{...};

/// Current version of the testing framework
pub const TESTING_FRAMEWORK_VERSION: &str = "0.1.0";

/// Default test timeout in seconds
pub const DEFAULT_TEST_TIMEOUT_SECS: u64 = 300;

/// Default maximum parallel tests
pub const DEFAULT_MAX_PARALLEL_TESTS: usize = 4;

/// Prelude module for tests - re-exports common types for convenience
pub mod prelude {
    // Fallible result alias + common error
    pub use super::errors::{TestError, TestOpResult as TestResultCompat};

    // Structured record (test outcome)
    pub use super::results::TestResult as TestRecord;

    // Common utilities people expect in test code
    pub use super::config::TestConfig;
}
