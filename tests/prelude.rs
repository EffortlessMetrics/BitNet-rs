#![cfg(feature = "integration-tests")]
//! Prelude module for integration tests
//!
//! This module re-exports commonly used types and functions to simplify imports
//! in test files. Add `use crate::prelude::*;` at the top of test modules.

// Core test framework types - import from bitnet-tests lib via Cargo.toml
// Note: When compiling tests/ integration tests, we need to import from the bitnet-tests crate
// But when compiling the bitnet-tests library itself (lib.rs), we use crate::
// The tests/prelude.rs is used by integration tests, so it imports from bitnet-tests
// However, since tests/ is integration tests for the root bitnet crate, not for bitnet-tests,
// we should just use the local common modules

// Re-export from local common modules (tests/ integration tests context)
pub use crate::common::bdd_grid::{
    ActiveProfile, active_features, active_profile, canonical_grid, to_grid_environment,
    to_grid_scenario, validate_active_profile,
};
pub use crate::common::errors::{ErrorSeverity, FixtureError, TestError, collect_environment_info};
pub use crate::common::harness::{FixtureCtx, TestCase, TestHarness};
pub use crate::common::results::{PassCheck, TestMetrics, TestResult, TestStatus, TestSummary};
pub use crate::common::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB, tensor_helpers::ct};

#[cfg(feature = "fixtures")]
pub use crate::common::fixtures::FixtureManager;

// Logging utilities
pub use crate::logging::init_logging;

// TestSuite trait
pub use crate::common::harness::TestSuite;

// Re-export standard library items commonly used in tests
pub use std::sync::Arc;
pub use std::time::{Duration, Instant};

// Re-export async runtime utilities
pub use tokio;

// Common external dependencies
pub use anyhow::{Context, Result as AnyhowResult};
pub use tracing::{debug, error, info, warn};
