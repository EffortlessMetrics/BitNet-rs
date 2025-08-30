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
pub mod concurrency_caps;
pub mod config;
pub mod config_scenarios;
pub mod config_scenarios_simple;
pub mod env;
pub mod error_analysis;
pub mod errors;
pub mod harness;
pub mod results;
pub mod serde_time;
pub mod tensor_helpers;
pub mod units;
pub mod utils;

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
pub mod logging;
pub mod optimization;
pub mod parallel;
pub mod selection;

// Re-export commonly used functions
pub use env::{env_bool, env_duration_secs, env_guard, env_string, env_u64, env_usize};
pub use units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};
pub use utils::{format_bytes, format_duration, get_memory_usage, get_peak_memory_usage};

// Cross-validation module (optional)
#[cfg(feature = "crossval")]
pub mod cross_validation;

// Data module - temporarily disabled due to compilation issues
// #[path = "../data/mod.rs"]
// pub mod data;

// Integration tests module - temporarily disabled due to compilation issues
// #[path = "../integration/mod.rs"]
// pub mod integration;

// Re-export commonly used types
pub use config::TestConfig;
pub use config_scenarios::{EnvironmentType, ScenarioConfigManager, TestingScenario};

// Avoid name collisions: expose both result types clearly
pub use errors::{TestError, TestOpResult};
pub use results::{TestMetrics, TestResult, TestSuiteResult};

// Only re-export FixtureManager when fixtures are enabled
#[cfg(feature = "fixtures")]
pub use fixtures::FixtureManager;

// Heavy/optional modules
#[cfg(feature = "trend")]
pub mod trend_reporting;

#[cfg(feature = "fixtures")]
pub mod config_validator;

#[cfg(feature = "crossval")]
pub use cross_validation::{ComparisonTestRunner, CompleteValidationResult, TestSummaryStatistics};

/// Current version of the testing framework
pub const TESTING_FRAMEWORK_VERSION: &str = "0.1.0";

/// Default test timeout in seconds
pub const DEFAULT_TEST_TIMEOUT_SECS: u64 = 300;

/// Default maximum parallel tests
pub const DEFAULT_MAX_PARALLEL_TESTS: usize = 4;

/// Prelude module for tests - re-exports common types for convenience
pub mod prelude {
    // Error handling types
    pub use super::errors::{TestError, TestOpResult};

    // Result types (ensure clear distinction from TestOpResult)
    pub use super::results::{
        TestMetrics,
        TestResult, // This is TestRecord, not TestOpResult
        TestStatus,
        TestSuiteResult,
    };

    // Core test framework types
    pub use super::harness::{
        FixtureCtx, // Stable fixture context type
        TestCase,
        TestHarness,
    };

    // Configuration
    pub use super::config::TestConfig;

    // Fixtures facade (provides stable API)
    pub use super::fixtures_facade::Fixtures;

    // Tensor helpers
    pub use super::tensor_helpers::ct;
    pub use crate::ctv; // bring the macro into scope

    // CI reporting re-export when available
    #[cfg(feature = "reporting")]
    pub use super::ci_reporting;

    // Concurrency control utilities
    pub use super::concurrency_caps::{init_concurrency_caps, init_and_get_async_limit, get_parallel_limit};
}
