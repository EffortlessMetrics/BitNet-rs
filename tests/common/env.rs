//! Environment variable parsing utilities for tests
//!
//! Provides typed, consistent parsing of environment variables with proper
//! error handling and case-insensitive boolean parsing.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Global environment lock to prevent race conditions in tests
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// RAII guard for safely setting and restoring environment variables in tests.
///
/// Automatically restores the original value (or removes the variable) when dropped,
/// ensuring cleanup even if the test panics.
///
/// # Example
/// ```rust
/// use bitnet_tests::common::env::EnvGuard;
///
/// fn my_test() {
///     let _guard1 = EnvGuard::set("BITNET_DETERMINISTIC", "1");
///     let _guard2 = EnvGuard::set("BITNET_SEED", "42");
///     // Test code here...
///     // Variables automatically restored when guards drop
/// }
/// ```
#[must_use = "EnvGuard must be held to ensure cleanup"]
pub struct EnvGuard {
    key: String,
    original: Option<String>,
}

impl EnvGuard {
    /// Set an environment variable and return a guard that will restore it on drop.
    pub fn set(key: impl Into<String>, value: impl AsRef<str>) -> Self {
        let key = key.into();
        let original = std::env::var(&key).ok();
        unsafe {
            std::env::set_var(&key, value.as_ref());
        }
        Self { key, original }
    }

    /// Remove an environment variable and return a guard that will restore it on drop.
    #[allow(dead_code)]
    pub fn remove(key: impl Into<String>) -> Self {
        let key = key.into();
        let original = std::env::var(&key).ok();
        unsafe {
            std::env::remove_var(&key);
        }
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref value) = self.original {
                std::env::set_var(&self.key, value);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }
}

/// Acquire a guard for safe environment variable manipulation in tests.
///
/// # Example
/// ```rust
/// use bitnet_tests::common::env::env_guard;
/// fn my_env_test() {
///     let _g = env_guard();
///     unsafe { std::env::set_var("BITNET_NO_NETWORK", "true"); }
///     // Test code here...
/// }
/// ```
#[must_use = "Environment guard must be held to prevent race conditions"]
pub fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.get_or_init(|| Mutex::new(())).lock().expect("env guard poisoned")
}

/// Parse an environment variable as a boolean (case-insensitive).
///
/// Truthy: "1", "true", "yes", "on", "enabled"
/// Falsy: "0", "false", "no", "off", "disabled" (or unset)
///
/// Note: Any other value is treated as false.
pub fn env_bool(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            let v = v.trim().to_lowercase();
            matches!(v.as_str(), "true" | "1" | "yes" | "on" | "enabled")
        })
        .unwrap_or(false)
}

/// Parse an environment variable as a u64.
/// Returns None if not set or not parseable.
#[must_use]
pub fn env_u64(var: &str) -> Option<u64> {
    std::env::var(var).ok().and_then(|v| v.trim().parse().ok())
}

/// Parse an environment variable as a usize.
/// Returns None if not set or not parseable.
#[must_use]
pub fn env_usize(var: &str) -> Option<usize> {
    std::env::var(var).ok().and_then(|v| v.trim().parse().ok())
}

/// Parse an environment variable as a Duration in seconds.
/// Returns None if not set or not parseable.
#[must_use]
pub fn env_duration_secs(var: &str) -> Option<Duration> {
    env_u64(var).map(Duration::from_secs)
}

/// Parse an environment variable as a string, trimming whitespace.
/// Returns None if not set.
#[must_use]
pub fn env_string(var: &str) -> Option<String> {
    std::env::var(var).ok().map(|v| v.trim().to_string()).filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_bool_parsing() {
        let _g = env_guard();

        // Test true values
        for val in &["true", "TRUE", "1", "yes", "YES", "on", "ON", "enabled", "ENABLED"] {
            unsafe {
                std::env::set_var("TEST_BOOL", val);
            }
            assert!(env_bool("TEST_BOOL"), "Failed to parse '{}' as true", val);
        }

        // Test false values
        for val in &["false", "0", "no", "off", "disabled", "random", ""] {
            unsafe {
                std::env::set_var("TEST_BOOL", val);
            }
            assert!(!env_bool("TEST_BOOL"), "Failed to parse '{}' as false", val);
        }

        // Test missing variable
        unsafe {
            std::env::remove_var("TEST_BOOL");
        }
        assert!(!env_bool("TEST_BOOL"));
    }

    #[test]
    fn test_env_numeric_parsing() {
        let _g = env_guard();

        unsafe {
            std::env::set_var("TEST_NUM", "42");
        }
        assert_eq!(env_u64("TEST_NUM"), Some(42));
        assert_eq!(env_usize("TEST_NUM"), Some(42));

        unsafe {
            std::env::set_var("TEST_NUM", "  100  ");
        }
        assert_eq!(env_u64("TEST_NUM"), Some(100));

        unsafe {
            std::env::set_var("TEST_NUM", "not_a_number");
        }
        assert_eq!(env_u64("TEST_NUM"), None);
        assert_eq!(env_usize("TEST_NUM"), None);

        unsafe {
            std::env::remove_var("TEST_NUM");
        }
        assert_eq!(env_u64("TEST_NUM"), None);
    }

    #[test]
    fn test_env_duration_parsing() {
        let _g = env_guard();

        unsafe {
            std::env::set_var("TEST_DUR", "30");
        }
        assert_eq!(env_duration_secs("TEST_DUR"), Some(Duration::from_secs(30)));

        unsafe {
            std::env::set_var("TEST_DUR", "invalid");
        }
        assert_eq!(env_duration_secs("TEST_DUR"), None);
    }

    #[test]
    fn test_env_string_parsing() {
        let _g = env_guard();

        unsafe {
            std::env::set_var("TEST_STR", "  hello world  ");
        }
        assert_eq!(env_string("TEST_STR"), Some("hello world".to_string()));

        unsafe {
            std::env::set_var("TEST_STR", "");
        }
        assert_eq!(env_string("TEST_STR"), None);

        unsafe {
            std::env::remove_var("TEST_STR");
        }
        assert_eq!(env_string("TEST_STR"), None);
    }
}
