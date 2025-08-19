//! Prelude module for integration tests
//!
//! This module re-exports commonly used types and functions to simplify imports
//! in test files. Add `use crate::prelude::*;` at the top of test modules.

// Core test framework types
pub use crate::{TestCase, TestError, TestMetrics, TestResult, TestSuite};

// Fixture management
pub use crate::FixtureCtx;
#[cfg(feature = "fixtures")]
pub use crate::FixtureManager;

pub use crate::common::harness::TestHarness;

// Error handling
pub use crate::common::errors::{ErrorSeverity, FixtureError, collect_environment_info};

// Reporting and results
pub use crate::common::results::{PassCheck, TestStatus, TestSummary};

// Logging utilities
pub use crate::logging::init_logging;

// Common test utilities
pub use crate::common::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB, tensor_helpers::ct};

// Re-export standard library items commonly used in tests
pub use std::sync::Arc;
pub use std::time::{Duration, Instant};

// Re-export async runtime utilities
pub use tokio;

// Common external dependencies
pub use anyhow::{Context, Result as AnyhowResult};
pub use tracing::{debug, error, info, warn};
