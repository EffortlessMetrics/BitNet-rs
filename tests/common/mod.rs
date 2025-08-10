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
pub mod logging_example;
pub mod results;
pub mod test_utilities;
pub mod utils;

// Cross-validation module
pub mod cross_validation;

// Data module
#[path = "../data/mod.rs"]
pub mod data;

// Re-export commonly used types
pub use config::TestConfig;
pub use errors::TestError;
pub use fixtures::FixtureManager;
pub use harness::{TestCase, TestHarness, TestSuite};
pub use logging::init_logging;
pub use results::{TestMetrics, TestResult};
pub use utils::{format_bytes, format_duration};

// Re-export cross-validation types
pub use cross_validation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory,
    ImplementationRegistry, InferenceConfig, InferenceResult, ModelFormat, ModelInfo,
    PerformanceMetrics, ResourceInfo, ResourceLimits, ResourceManager, ResourceSummary,
};

/// Current version of the testing framework
pub const TESTING_FRAMEWORK_VERSION: &str = "0.1.0";

/// Default test timeout in seconds
pub const DEFAULT_TEST_TIMEOUT_SECS: u64 = 300;

/// Default maximum parallel tests
pub const DEFAULT_MAX_PARALLEL_TESTS: usize = 4;
