//! Prelude module for common test imports

// Bring trait methods & common types into scope for all tests
pub use crate::common::results::PassCheck as _;
pub use crate::common::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult};
pub use crate::common::harness::{FixtureCtx, TestCase, TestHarness};
pub use crate::common::fixtures_facade::Fixtures;
pub use crate::common::config::TestConfig;