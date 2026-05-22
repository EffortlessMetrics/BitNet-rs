//! # bitnet-test-env
//!
//! Process-wide environment variable isolation primitives for tests.
//! This microcrate contains only environment locking and restoration logic,
//! so higher-level test-support crates can compose it with policy/fixture helpers.

#![deny(unused_must_use)]

use std::env;

mod env_lock;
mod env_restore;

use env_lock::get_env_lock;
use env_restore::{restore_saved_value, save_original_value};

/// RAII guard for safe environment variable management.
#[derive(Debug)]
pub struct EnvGuard {
    key: String,
    old: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Capture current state of a single variable while holding the global env lock.
    pub fn new(key: &str) -> Self {
        let lock = get_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let old = env::var(key).ok();

        Self { key: key.to_string(), old, _lock: lock }
    }

    /// Remove the environment variable temporarily.
    pub fn remove(&self) {
        // SAFETY: We hold the global ENV_LOCK mutex for this guard's lifetime.
        unsafe {
            env::remove_var(&self.key);
        }
    }

    /// Set the environment variable to a new value.
    pub fn set(&self, val: &str) {
        // SAFETY: We hold the global ENV_LOCK mutex for this guard's lifetime.
        unsafe {
            env::set_var(&self.key, val);
        }
    }

    /// Get the key being guarded.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Get the original value (if any).
    pub fn original_value(&self) -> Option<&str> {
        self.old.as_deref()
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: We still hold the global ENV_LOCK mutex (via self._lock).
        unsafe {
            if let Some(ref v) = self.old {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }
}

/// Scope lock for changing multiple environment variables without deadlock.
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
        save_original_value(&mut self.saved, key);
        // SAFETY: We hold the global ENV_LOCK mutex for the duration of this scope.
        unsafe { env::set_var(key, value) };
    }

    /// Remove `key` from the environment, saving the original value for restoration.
    pub fn remove(&mut self, key: &str) {
        save_original_value(&mut self.saved, key);
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
            restore_saved_value(key, original);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(bitnet_env)]
    fn env_guard_set_and_restore() {
        let test_key = "BITNET_TEST_GUARD_SET";

        unsafe {
            env::remove_var(test_key);
        }

        {
            let guard = EnvGuard::new(test_key);
            guard.set("test_value");
            assert_eq!(env::var(test_key).unwrap(), "test_value");
        }

        assert!(env::var(test_key).is_err());
    }

    #[test]
    #[serial(bitnet_env)]
    fn env_scope_sets_multiple_and_restores() {
        let key_a = "BITNET_TEST_SCOPE_A";
        let key_b = "BITNET_TEST_SCOPE_B";

        unsafe {
            env::remove_var(key_a);
            env::set_var(key_b, "old");
        }

        {
            let mut scope = EnvScope::new();
            scope.set(key_a, "a");
            scope.set(key_b, "new");

            assert_eq!(env::var(key_a).unwrap(), "a");
            assert_eq!(env::var(key_b).unwrap(), "new");
        }

        assert!(env::var(key_a).is_err());
        assert_eq!(env::var(key_b).unwrap(), "old");

        unsafe {
            env::remove_var(key_b);
        }
    }
}
