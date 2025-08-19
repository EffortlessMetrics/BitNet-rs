// Re-export modules for backward compatibility with existing tests
pub mod common;
pub mod prelude;

// Only compile integration tests when explicitly enabled.
// This keeps crossval work unblocked.
#[cfg(feature = "integration-tests")]
pub mod integration;

// ===== Back-compat shims for legacy imports =====
// Many files still do `crate::results`, `crate::fixtures`, etc.
// These re-export the new locations so those imports work again.

pub mod results  { pub use crate::common::results::*; }
pub mod errors   { pub use crate::common::errors::*; }
#[cfg(feature = "fixtures")]
pub mod fixtures { pub use crate::common::fixtures::*; }
pub mod harness  { pub use crate::common::harness::*; }
pub mod config   { pub use crate::common::config::*; }
pub mod utils    { pub use crate::common::utils::*; }
pub mod units    { pub use crate::common::units::*; }

// Frequently used items at the crate root for older test code
pub use common::prelude::*;                       // e.g., TestCase, etc.
pub use common::harness::TestSuite;               // bring TestSuite to the crate root
pub use common::harness::TestHarness;             // TestHarness from harness module
pub use common::config::TestConfig;               // TestConfig from config module
#[cfg(feature = "fixtures")]
pub use common::fixtures::FixtureManager;         // bring FixtureManager to the crate root
#[cfg(feature = "fixtures")]
pub use common::fixtures_facade::Fixtures;        // Add Fixtures re-export
pub use common::errors::TestError;
pub use common::results::{TestMetrics, TestStatus, TestSuiteResult, TestResult as TestResultStruct};
pub use common::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};

// CRITICAL: Integration tests expect TestResult<T> as a type alias, not the struct
pub type TestResult<T = ()> = Result<T, TestError>;

// Keep the old `crate::fast_config::fast_config` path working:
#[cfg(feature = "fixtures")]
pub mod fast_config {
    pub use crate::common::fast_config::*;
}

// Additional re-exports for bin files
#[cfg(feature = "fixtures")]
pub mod config_validator { pub use crate::common::config_validator::*; }
#[cfg(feature = "fixtures")]
pub mod fast_feedback_simple { pub use crate::common::fast_feedback_simple::*; }
#[cfg(feature = "reporting")]
pub mod reporting { pub use crate::common::reporting::*; }
#[cfg(feature = "reporting")]
pub mod ci_reporting { pub use crate::common::ci_reporting::*; }
#[cfg(feature = "trend")]
pub mod trend_reporting { pub use crate::common::trend_reporting::*; }
pub mod debug_cli { pub use crate::common::debug_cli::*; }
pub mod logging { pub use crate::common::logging::*; }