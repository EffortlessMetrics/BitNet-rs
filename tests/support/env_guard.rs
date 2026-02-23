// RAII-style environment variable guard for test isolation
//
// This module provides thread-safe environment variable management for tests,
// ensuring automatic restoration of original values and preventing test pollution.
//
/// # Design Philosophy
//
/// The BitNet.rs test suite uses a **two-tiered approach** for environment variable testing:
//
/// 1. **Scoped approach (Preferred)**: Use `temp_env::with_var()` with `#[serial(bitnet_env)]`
///    - Closure-based isolation with automatic cleanup
///    - Clean, idiomatic Rust code
///    - Best for most test scenarios
//
/// 2. **RAII approach (Fallback)**: Use `EnvGuard` when closure-based approach is impractical
///    - Drop-based restoration via RAII pattern
///    - Useful for complex test setups requiring sequential steps
///    - Must still use `#[serial(bitnet_env)]` to prevent process-level races
//
/// # Safety Guarantees
//
/// - **Process-level serialization**: Always use `#[serial(bitnet_env)]` macro to prevent
///   concurrent test execution across cargo test processes
/// - **Thread-level synchronization**: Internal mutex ensures thread safety within a test
/// - **Automatic restoration**: Drop implementation guarantees cleanup even on panic
//
/// # Usage Examples
//
/// ## Preferred: Scoped Approach
//
/// ```rust,ignore
/// use serial_test::serial;
/// use temp_env::with_var;
//
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_strict_mode_enabled() {
///     // Closure-based isolation - automatically restored on scope exit
///     with_var("BITNET_STRICT_MODE", Some("1"), || {
///         let config = StrictModeConfig::from_env();
///         assert!(config.enabled);
///     });
/// }
/// ```
//
/// ## Fallback: RAII Approach
//
/// ```rust,ignore
/// use serial_test::serial;
//
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_strict_mode_with_guard() {
///     let guard = EnvGuard::new("BITNET_STRICT_MODE");
///     guard.set("1");
//
///     let config = StrictModeConfig::from_env();
///     assert!(config.enabled);
///     // Guard drops here, restoring original value
/// }
/// ```
//
/// # Anti-Patterns
//
/// **❌ NEVER do this** (causes test pollution and races):
//
/// ```rust,no_run
/// #[test]
/// fn test_without_serialization() {  // Missing #[serial(bitnet_env)]
///     unsafe { std::env::set_var("BITNET_STRICT_MODE", "1"); }
///     // ❌ Races with other tests!
/// }
/// ```
use std::{
    env,
    sync::{Mutex, OnceLock},
};

/// Global lock to serialize environment variable modifications across threads
///
/// This mutex ensures thread-safe access to environment variables within a single
/// test process. It does NOT prevent races across multiple cargo test processes -
/// use `#[serial(bitnet_env)]` for that.
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn get_env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard for safe environment variable management
///
/// This guard provides automatic restoration of environment variable state via
/// the Drop trait. It holds a global mutex lock to ensure thread safety.
///
/// # Lifecycle
///
/// 1. **Creation**: Captures current environment variable state
/// 2. **Modification**: Provides `set()` and `remove()` methods
/// 3. **Drop**: Automatically restores original state (even on panic)
///
/// # Safety
///
/// - Uses `unsafe { env::set_var/remove_var }` internally (required by std::env API)
/// - Thread-safe via global mutex lock
/// - **Must be used with `#[serial(bitnet_env)]`** for process-level safety
#[derive(Debug)]
pub struct EnvGuard {
    /// The environment variable key being guarded
    key: String,
    /// Original value (None if variable was not set)
    old: Option<String>,
    /// Mutex guard held for the duration of the EnvGuard's lifetime
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Create a new environment variable guard
    ///
    /// This captures the current state of the specified environment variable
    /// and acquires a global lock to ensure thread safety.
    ///
    /// # Arguments
    ///
    /// * `key` - The environment variable name to guard
    ///
    /// # Returns
    ///
    /// An `EnvGuard` instance that will restore the original state on drop
    ///
    /// # Process Safety
    ///
    /// **Important**: This guard only provides thread-level safety. You MUST
    /// use `#[serial(bitnet_env)]` on your test to prevent races across
    /// multiple cargo test processes.
    pub fn new(key: &str) -> Self {
        let lock = get_env_lock().lock().unwrap_or_else(|e| e.into_inner());

        let old = env::var(key).ok();

        Self { key: key.to_string(), old, _lock: lock }
    }

    /// Remove the environment variable temporarily
    ///
    /// This removes the environment variable from the process environment.
    /// The original value (if any) will be restored when the guard is dropped.
    ///
    /// # Safety
    ///
    /// This method uses `unsafe { env::remove_var }` internally, which is
    /// required by the std::env API. The safety is guaranteed by:
    /// - Holding a global mutex lock
    /// - Automatic restoration on drop
    /// - Process-level serialization via `#[serial(bitnet_env)]`
    pub fn remove(&self) {
        // SAFETY: We hold the global ENV_LOCK mutex, ensuring thread-safe
        // access to environment variables. The calling test must use
        // #[serial(bitnet_env)] to ensure process-level safety.
        unsafe {
            env::remove_var(&self.key);
        }
    }

    /// Set the environment variable to a new value
    ///
    /// This sets the environment variable to the specified value. The original
    /// value (if any) will be restored when the guard is dropped.
    ///
    /// # Arguments
    ///
    /// * `val` - The new value to set (must be valid UTF-8)
    ///
    /// # Safety
    ///
    /// This method uses `unsafe { env::set_var }` internally, which is
    /// required by the std::env API. The safety is guaranteed by:
    /// - Holding a global mutex lock
    /// - Automatic restoration on drop
    /// - Process-level serialization via `#[serial(bitnet_env)]`
    pub fn set(&self, val: &str) {
        // SAFETY: We hold the global ENV_LOCK mutex, ensuring thread-safe
        // access to environment variables. The calling test must use
        // #[serial(bitnet_env)] to ensure process-level safety.
        unsafe {
            env::set_var(&self.key, val);
        }
    }

    /// Get the key being guarded
    ///
    /// # Returns
    ///
    /// The environment variable name as a string slice
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the original value (if any)
    ///
    /// # Returns
    ///
    /// - `Some(&str)` if the variable was originally set
    /// - `None` if the variable was not set when the guard was created
    pub fn original_value(&self) -> Option<&str> {
        self.old.as_deref()
    }
}

impl Drop for EnvGuard {
    /// Restore the original environment variable state
    ///
    /// This method is called automatically when the guard goes out of scope,
    /// ensuring cleanup even if the test panics.
    ///
    /// # Safety
    ///
    /// This uses `unsafe { env::set_var/remove_var }` to restore the original
    /// state. The safety is guaranteed by:
    /// - Still holding the global ENV_LOCK mutex (via _lock field)
    /// - Process-level serialization via `#[serial(bitnet_env)]`
    fn drop(&mut self) {
        // SAFETY: We still hold the global ENV_LOCK mutex (via self._lock),
        // ensuring thread-safe access to environment variables. The calling
        // test must use #[serial(bitnet_env)] to ensure process-level safety.
        unsafe {
            if let Some(ref v) = self.old {
                // Restore original value
                env::set_var(&self.key, v);
            } else {
                // Remove variable (it wasn't set originally)
                env::remove_var(&self.key);
            }
        }
    }
}

/// A scope that acquires the env lock **once** and allows setting or removing
/// any number of environment variables under that single lock.
///
/// This is the preferred replacement for creating multiple `EnvGuard` instances
/// in the same test scope, which deadlocks because `EnvGuard` holds a
/// non-reentrant global mutex for its entire lifetime.
///
/// # Usage
///
/// ```rust,ignore
/// #[test]
/// #[serial(bitnet_env)]
/// fn test_multi_env() {
///     let mut scope = EnvScope::new();
///     scope.set("VAR_A", "1");
///     scope.set("VAR_B", "2");
///     // ... test body ...
/// } // all vars restored on drop
/// ```
pub struct EnvScope {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: std::collections::HashMap<String, Option<String>>,
}

impl EnvScope {
    /// Acquire the env lock and return a new scope.
    pub fn new() -> Self {
        let lock = get_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        Self { _lock: lock, saved: std::collections::HashMap::new() }
    }

    /// Set `key` to `value`, saving the original value for restoration.
    pub fn set(&mut self, key: &str, value: &str) {
        self.saved.entry(key.to_string()).or_insert_with(|| env::var(key).ok());
        // SAFETY: We hold the global ENV_LOCK mutex for the duration of this scope.
        unsafe { env::set_var(key, value) };
    }

    /// Remove `key` from the environment, saving the original value for restoration.
    pub fn remove(&mut self, key: &str) {
        self.saved.entry(key.to_string()).or_insert_with(|| env::var(key).ok());
        // SAFETY: We hold the global ENV_LOCK mutex for the duration of this scope.
        unsafe { env::remove_var(key) };
    }
}

impl Default for EnvScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, original) in &self.saved {
            // SAFETY: We still hold the global ENV_LOCK mutex (via self._lock).
            unsafe {
                match original {
                    Some(v) => env::set_var(key, v),
                    None => env::remove_var(key),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_set_and_restore() {
        let test_key = "BITNET_TEST_GUARD_SET";

        // Ensure clean state
        unsafe {
            env::remove_var(test_key);
        }

        {
            let guard = EnvGuard::new(test_key);
            guard.set("test_value");

            assert_eq!(env::var(test_key).unwrap(), "test_value");
        }

        // After drop, variable should be removed (original state)
        assert!(env::var(test_key).is_err());
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_remove_and_restore() {
        let test_key = "BITNET_TEST_GUARD_REMOVE";

        // Set initial value
        unsafe {
            env::set_var(test_key, "original");
        }

        {
            let guard = EnvGuard::new(test_key);
            guard.remove();

            assert!(env::var(test_key).is_err());
        }

        // After drop, original value should be restored
        assert_eq!(env::var(test_key).unwrap(), "original");

        // Cleanup
        unsafe {
            env::remove_var(test_key);
        }
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_multiple_sets() {
        let test_key = "BITNET_TEST_GUARD_MULTI";

        unsafe {
            env::remove_var(test_key);
        }

        {
            let guard = EnvGuard::new(test_key);
            guard.set("value1");
            assert_eq!(env::var(test_key).unwrap(), "value1");

            guard.set("value2");
            assert_eq!(env::var(test_key).unwrap(), "value2");

            guard.set("value3");
            assert_eq!(env::var(test_key).unwrap(), "value3");
        }

        // After drop, should be removed (original state)
        assert!(env::var(test_key).is_err());
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_preserves_original() {
        let test_key = "BITNET_TEST_GUARD_PRESERVE";

        // Set original value
        unsafe {
            env::set_var(test_key, "original_value");
        }

        {
            let guard = EnvGuard::new(test_key);
            assert_eq!(guard.original_value(), Some("original_value"));

            guard.set("temporary");
            assert_eq!(env::var(test_key).unwrap(), "temporary");
        }

        // After drop, original value should be restored
        assert_eq!(env::var(test_key).unwrap(), "original_value");

        // Cleanup
        unsafe {
            env::remove_var(test_key);
        }
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_key_accessor() {
        let test_key = "BITNET_TEST_GUARD_KEY";
        let guard = EnvGuard::new(test_key);

        assert_eq!(guard.key(), test_key);
    }

    #[test]
    #[serial(bitnet_env)]
    #[should_panic(expected = "intentional panic")]
    fn test_env_guard_panic_safety() {
        let test_key = "BITNET_TEST_GUARD_PANIC";

        // Set original value
        unsafe {
            env::set_var(test_key, "original");
        }

        {
            let guard = EnvGuard::new(test_key);
            guard.set("temporary");

            // Panic - guard should still restore on unwind
            panic!("intentional panic");
        }

        // This won't be reached, but guard will still drop and restore
    }

    // Note: This test verifies panic safety by checking restoration after the panic
    #[test]
    #[serial(bitnet_env)]
    fn test_env_guard_panic_safety_verification() {
        let test_key = "BITNET_TEST_GUARD_PANIC_VERIFY";

        // Set original value
        unsafe {
            env::set_var(test_key, "original");
        }

        let result = std::panic::catch_unwind(|| {
            let guard = EnvGuard::new(test_key);
            guard.set("temporary");
            panic!("intentional panic");
        });

        assert!(result.is_err(), "should have panicked");

        // Verify restoration happened despite panic
        assert_eq!(env::var(test_key).unwrap(), "original");

        // Cleanup
        unsafe {
            env::remove_var(test_key);
        }
    }
}
