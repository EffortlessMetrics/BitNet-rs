/// Fixture Infrastructure for Integration Tests
///
/// This module provides test scaffolding for BitNet.cpp auto-configuration integration tests.
/// It enables CI-compatible testing without requiring real C++ installations by using
/// fixture-based directory structures and mock binaries.
///
/// # Architecture
///
/// The fixture infrastructure consists of three main components:
///
/// 1. **DirectoryLayoutBuilder**: Creates temporary directory structures matching
///    real BitNet.cpp/llama.cpp layouts with configurable variants.
///
/// 2. **MockLibrary**: Generates platform-specific stub libraries (.so, .dylib, .lib)
///    with correct file headers for build.rs detection.
///
/// 3. **TestEnvironment**: Provides environment isolation using EnvGuard pattern
///    with automatic cleanup of temp directories and environment variables.
///
/// # Usage
///
/// ```rust,no_run
/// use serial_test::serial;
/// use fixtures::{DirectoryLayoutBuilder, LayoutType, TestEnvironment};
///
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_bitnet_standard_layout() {
///     let mut env = TestEnvironment::new();
///     let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
///         .with_libs(true)
///         .with_headers(true)
///         .build()
///         .unwrap();
///
///     env.set_bitnet_cpp_dir(layout.root_path().to_path_buf());
///
///     // Verify directory structure
///     assert!(layout.root_path().join("build/lib/libbitnet.so").exists());
/// }
/// ```
///
/// # Test Specification
///
/// Tests feature spec: bitnet-integration-tests.md#implementation-plan-phase-1
pub mod directory_layouts;
pub mod env_isolation;
pub mod mock_libraries;

#[cfg(test)]
mod tests;

// Re-export public API
#[allow(unused_imports)]
pub use directory_layouts::{DirectoryLayout, DirectoryLayoutBuilder, LayoutType};
pub use env_isolation::TestEnvironment;
pub use mock_libraries::{LibraryType, MockLibrary, Platform};

use std::error::Error as StdError;
use std::fmt;

/// Errors that can occur during fixture generation
#[derive(Debug)]
pub enum FixtureError {
    /// IO error during directory or file creation
    Io(std::io::Error),
    /// Invalid layout configuration
    #[allow(dead_code)]
    InvalidLayout(String),
    /// Mock library generation failed
    LibraryGeneration(String),
    /// Environment setup failed
    EnvironmentSetup(String),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixtureError::Io(e) => write!(f, "IO error: {}", e),
            FixtureError::InvalidLayout(msg) => write!(f, "Invalid layout: {}", msg),
            FixtureError::LibraryGeneration(msg) => write!(f, "Library generation failed: {}", msg),
            FixtureError::EnvironmentSetup(msg) => write!(f, "Environment setup failed: {}", msg),
        }
    }
}

impl StdError for FixtureError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            FixtureError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FixtureError {
    fn from(error: std::io::Error) -> Self {
        FixtureError::Io(error)
    }
}
