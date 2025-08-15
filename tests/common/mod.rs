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

// Lightweight, always-on modules
pub mod config;
pub mod config_scenarios;
pub mod config_scenarios_simple;
pub mod errors;
pub mod results;
pub mod serde_time;
pub mod utils;
pub mod harness;

// Optional debug modules
pub mod debug_cli;
pub mod debug_integration;
pub mod debugging;

// Heavy/optional modules - only compile when feature is enabled
#[cfg(feature = "fixtures")]
pub mod fixtures;

// Facade for fixtures to reduce cfg scatter
pub mod fixtures_facade;

#[cfg(feature = "fixtures")]
pub mod fast_config;

#[cfg(feature = "fixtures")]
pub mod fast_feedback_simple;

#[cfg(feature = "reporting")]
pub mod reporting;

#[cfg(feature = "reporting")]
pub mod ci_reporting;

pub mod incremental;
pub mod optimization;
pub mod parallel;
pub mod selection;

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
pub use config_scenarios::{ScenarioConfigManager, TestingScenario, EnvironmentType};

// Avoid name collisions: expose both result types clearly
pub use errors::{TestError, TestOpResult};
pub use results::{TestResult, TestSuiteResult, TestMetrics};

// Only re-export FixtureManager when fixtures are enabled
#[cfg(feature = "fixtures")]
pub use fixtures::FixtureManager;

// Heavy/optional modules
#[cfg(feature = "trend")]
pub mod trend_reporting;

#[cfg(feature = "fixtures")]
pub mod config_validator;

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
