//! Prelude module for integration tests
//!
//! This module re-exports commonly used types and functions to simplify imports
//! in test files. Add `use crate::prelude::*;` at the top of test modules.

// Core test framework types
pub use bitnet_tests::{
    TestCase, TestSuite, TestResult, TestError, TestMetrics,
    PassCheck, FailureReporter,
};

// Fixture management
pub use bitnet_tests::{
    FixtureManager, FixtureCtx,
    harness::{TestRunConfig, TestRunContext},
};

// Error handling
pub use bitnet_tests::errors::{
    ErrorSeverity, FixtureError, collect_environment_info,
};

// Reporting and results
pub use bitnet_tests::results::{
    ResultAggregator, TestSummary,
};

// Logging utilities
pub use bitnet_tests::logging::init_logging;

// Common test utilities
pub use bitnet_tests::common::{
    tensor_helpers::{ct, generate_random_tensor, approx_equal},
    BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB,
};

// Re-export standard library items commonly used in tests
pub use std::sync::Arc;
pub use std::time::{Duration, Instant};

// Re-export async runtime utilities
pub use tokio;

// Common external dependencies
pub use anyhow::{Result as AnyhowResult, Context};
pub use tracing::{debug, info, warn, error};