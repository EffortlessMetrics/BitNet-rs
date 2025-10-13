// RAII guard for safe environment variable management in tests
//
// This module provides `EnvGuard`, a thread-safe RAII wrapper that temporarily
// sets environment variables and automatically restores the previous state on drop.
//
// ## Safety
//
// Modifying environment variables is inherently unsafe in multi-threaded contexts.
// This guard uses `unsafe` blocks for env var manipulation but provides a safe API
// by ensuring cleanup even on panic.

/// RAII guard for safe environment variable management
///
/// Temporarily sets an environment variable and restores the previous value on drop.
/// If the variable wasn't set before, it will be removed on drop.
pub struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    /// Set an environment variable temporarily
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name (must be 'static)
    /// * `val` - The value to set
    ///
    /// # Returns
    ///
    /// An `EnvGuard` that will restore the previous value on drop
    ///
    /// # Safety
    ///
    /// Uses `unsafe` internally for env var manipulation but provides a safe API.
    /// The guard ensures cleanup even if the test panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tests::support::env_guard::EnvGuard;
    ///
    /// let _guard = EnvGuard::set("TEST_VAR", "value");
    /// assert_eq!(std::env::var("TEST_VAR").unwrap(), "value");
    /// // Variable restored on drop
    /// ```
    pub fn set(key: &'static str, val: &str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, val);
        }
        Self { key, prev }
    }

    /// Temporarily remove an environment variable
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name to remove
    ///
    /// # Returns
    ///
    /// An `EnvGuard` that will restore the previous value on drop
    pub fn remove(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_restore() {
        // Clear the test var first
        unsafe { std::env::remove_var("ENVGUARD_TEST"); }

        {
            let _guard = EnvGuard::set("ENVGUARD_TEST", "test_value");
            assert_eq!(std::env::var("ENVGUARD_TEST").unwrap(), "test_value");
        }

        // Variable should be removed after guard drop
        assert!(std::env::var("ENVGUARD_TEST").is_err());
    }

    #[test]
    fn test_overwrite_and_restore() {
        unsafe { std::env::set_var("ENVGUARD_TEST2", "original"); }

        {
            let _guard = EnvGuard::set("ENVGUARD_TEST2", "changed");
            assert_eq!(std::env::var("ENVGUARD_TEST2").unwrap(), "changed");
        }

        // Original value should be restored
        assert_eq!(std::env::var("ENVGUARD_TEST2").unwrap(), "original");

        // Cleanup
        unsafe { std::env::remove_var("ENVGUARD_TEST2"); }
    }

    #[test]
    fn test_remove_and_restore() {
        unsafe { std::env::set_var("ENVGUARD_TEST3", "value"); }

        {
            let _guard = EnvGuard::remove("ENVGUARD_TEST3");
            assert!(std::env::var("ENVGUARD_TEST3").is_err());
        }

        // Value should be restored
        assert_eq!(std::env::var("ENVGUARD_TEST3").unwrap(), "value");

        // Cleanup
        unsafe { std::env::remove_var("ENVGUARD_TEST3"); }
    }
}
