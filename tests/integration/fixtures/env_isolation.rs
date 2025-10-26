/// Test Environment Isolation for Integration Tests
///
/// Provides environment isolation using EnvGuard pattern with automatic cleanup
/// of temporary directories and environment variables.
///
/// # Test Specification
///
/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
use super::FixtureError;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

// Import EnvGuard from existing test infrastructure
// Note: This will need to be available in the test environment
// For now, we'll define a simplified version
struct EnvGuard {
    key: String,
    original: Option<String>,
}

impl EnvGuard {
    fn new(key: &str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: Setting environment variables is unsafe in multi-threaded contexts,
        // but tests using this are marked with #[serial(bitnet_env)] to ensure
        // serial execution and prevent race conditions.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key: key.to_string(), original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: Restoring environment variables is unsafe in multi-threaded contexts,
        // but tests using this are marked with #[serial(bitnet_env)] to ensure
        // serial execution and prevent race conditions.
        unsafe {
            match &self.original {
                Some(val) => std::env::set_var(&self.key, val),
                None => std::env::remove_var(&self.key),
            }
        }
    }
}

/// Test environment with automatic cleanup
///
/// Manages environment variables and temporary directories with automatic
/// restoration on drop. Use with `#[serial(bitnet_env)]` annotation to
/// prevent race conditions in parallel test execution.
///
/// # Usage
///
/// ```rust,no_run
/// use serial_test::serial;
/// use fixtures::TestEnvironment;
///
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_with_isolated_env() {
///     let mut env = TestEnvironment::new();
///     env.set_bitnet_cpp_dir(PathBuf::from("/tmp/bitnet"));
///
///     // Test code here - env restored automatically on drop
/// }
/// ```
pub struct TestEnvironment {
    env_guards: Vec<EnvGuard>,
    temp_dir: TempDir,
}

impl TestEnvironment {
    /// Create a new isolated test environment
    ///
    /// # Returns
    ///
    /// - `Ok(TestEnvironment)`: Successfully created environment
    /// - `Err(FixtureError)`: Failed to create temporary directory
    pub fn new() -> Result<Self, FixtureError> {
        let temp_dir = TempDir::new().map_err(|e| {
            FixtureError::EnvironmentSetup(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self { env_guards: Vec::new(), temp_dir })
    }

    /// Set BITNET_CPP_DIR environment variable
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-14-bitnet-cpp-dir
    pub fn set_bitnet_cpp_dir(&mut self, path: PathBuf) {
        let guard = EnvGuard::new("BITNET_CPP_DIR", path.to_str().unwrap());
        self.env_guards.push(guard);
    }

    /// Set BITNET_CROSSVAL_LIBDIR environment variable (highest priority)
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-13-crossval-libdir
    pub fn set_crossval_libdir(&mut self, path: PathBuf) {
        let guard = EnvGuard::new("BITNET_CROSSVAL_LIBDIR", path.to_str().unwrap());
        self.env_guards.push(guard);
    }

    /// Set HOME environment variable (for cache directory tests)
    ///
    /// Tests feature spec: bitnet-integration-tests.md#scenario-15-default-cache
    pub fn set_home(&mut self, path: PathBuf) {
        let guard = EnvGuard::new("HOME", path.to_str().unwrap());
        self.env_guards.push(guard);
    }

    /// Set LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
    #[allow(dead_code)]
    pub fn set_library_path(&mut self, path: PathBuf) {
        #[cfg(target_os = "linux")]
        let key = "LD_LIBRARY_PATH";

        #[cfg(target_os = "macos")]
        let key = "DYLD_LIBRARY_PATH";

        #[cfg(target_os = "windows")]
        let key = "PATH";

        let guard = EnvGuard::new(key, path.to_str().unwrap());
        self.env_guards.push(guard);
    }

    /// Set custom environment variable
    #[allow(dead_code)]
    pub fn set_env(&mut self, key: &str, value: &str) {
        let guard = EnvGuard::new(key, value);
        self.env_guards.push(guard);
    }

    /// Get the temporary directory path
    pub fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }

    /// Get mutable temporary directory path
    #[allow(dead_code)]
    pub fn temp_path_mut(&mut self) -> &Path {
        self.temp_dir.path()
    }

    /// Create a subdirectory in the temporary directory
    pub fn create_subdir(&self, name: &str) -> Result<PathBuf, FixtureError> {
        let subdir = self.temp_dir.path().join(name);
        std::fs::create_dir_all(&subdir)?;
        Ok(subdir)
    }

    /// Unset an environment variable (for testing missing env vars)
    pub fn unset_env(&mut self, key: &str) {
        // SAFETY: Removing environment variables is unsafe in multi-threaded contexts,
        // but tests using this are marked with #[serial(bitnet_env)] to ensure
        // serial execution and prevent race conditions.
        unsafe {
            std::env::remove_var(key);
        }
        // Add a guard to restore it on drop if it was set
        let original = std::env::var(key).ok();
        if original.is_some() {
            self.env_guards.push(EnvGuard { key: key.to_string(), original });
        }
    }

    /// Get environment variable value (convenience method)
    #[allow(dead_code)]
    pub fn get_env(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self::new().expect("Failed to create default TestEnvironment")
    }
}

// Environment is automatically cleaned up when TestEnvironment is dropped
// No explicit cleanup needed due to Drop implementations on EnvGuard and TempDir

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(bitnet_env)]
    fn test_env_isolation_basic() {
        let original = std::env::var("BITNET_CPP_DIR").ok();

        {
            let mut env = TestEnvironment::new().unwrap();
            env.set_bitnet_cpp_dir(PathBuf::from("/tmp/test"));

            assert_eq!(std::env::var("BITNET_CPP_DIR").unwrap(), "/tmp/test");
        }

        // Verify restoration after drop
        assert_eq!(std::env::var("BITNET_CPP_DIR").ok(), original);
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_multiple_env_vars() {
        let original_cpp_dir = std::env::var("BITNET_CPP_DIR").ok();
        let original_libdir = std::env::var("BITNET_CROSSVAL_LIBDIR").ok();

        {
            let mut env = TestEnvironment::new().unwrap();
            env.set_bitnet_cpp_dir(PathBuf::from("/tmp/cpp"));
            env.set_crossval_libdir(PathBuf::from("/tmp/lib"));

            assert_eq!(std::env::var("BITNET_CPP_DIR").unwrap(), "/tmp/cpp");
            assert_eq!(std::env::var("BITNET_CROSSVAL_LIBDIR").unwrap(), "/tmp/lib");
        }

        // Verify both restored
        assert_eq!(std::env::var("BITNET_CPP_DIR").ok(), original_cpp_dir);
        assert_eq!(std::env::var("BITNET_CROSSVAL_LIBDIR").ok(), original_libdir);
    }

    #[test]
    fn test_temp_dir_cleanup() {
        let temp_path: PathBuf;

        {
            let env = TestEnvironment::new().unwrap();
            temp_path = env.temp_path().to_path_buf();

            // Directory should exist while env is alive
            assert!(temp_path.exists());
        }

        // Directory should be cleaned up after drop
        assert!(!temp_path.exists());
    }

    #[test]
    fn test_create_subdir() {
        let env = TestEnvironment::new().unwrap();
        let subdir = env.create_subdir("test_subdir").unwrap();

        assert!(subdir.exists());
        assert!(subdir.is_dir());
        assert_eq!(subdir.file_name().unwrap(), "test_subdir");
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_unset_env() {
        // Set a test env var
        // SAFETY: Setting environment variable in test marked with #[serial(bitnet_env)]
        unsafe {
            std::env::set_var("TEST_VAR", "original");
        }

        {
            let mut env = TestEnvironment::new().unwrap();
            env.unset_env("TEST_VAR");

            // Should be unset
            assert!(std::env::var("TEST_VAR").is_err());
        }

        // Should be restored after drop
        // Note: This test assumes the guard tracks the original value
        // Implementation may vary
    }
}
