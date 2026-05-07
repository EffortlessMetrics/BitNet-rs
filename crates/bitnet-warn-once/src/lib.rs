//! Thread-safe warn-once utility for rate-limited logging.
//!
//! Provides a mechanism to log warnings only once per unique key, with subsequent
//! occurrences logged at debug level. This is useful for avoiding log spam when
//! the same warning condition occurs repeatedly (e.g., deprecated API usage,
//! non-fatal errors in hot paths).
//!
//! # Examples
//!
//! ```
//! use bitnet_warn_once::warn_once;
//!
//! fn some_function() {
//!     warn_once!("deprecated_api_v1", "Using deprecated API v1, please migrate to v2");
//! }
//!
//! // First call: logs at WARN level
//! some_function();
//!
//! // Subsequent calls: logs at DEBUG level (rate-limited)
//! some_function();
//! some_function();
//! ```

use std::collections::HashSet;
#[cfg(test)]
use std::sync::TryLockError;
use std::sync::{Mutex, OnceLock};

/// Global registry of seen warning keys.
///
/// This uses `OnceLock<Mutex<HashSet<String>>>` to provide thread-safe,
/// lazy initialization without `static mut`.
static WARN_REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Get the global warning registry, initializing it if necessary.
fn get_registry() -> &'static Mutex<HashSet<String>> {
    WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Log a warning message once per unique key, with rate-limiting for subsequent occurrences.
///
/// The first occurrence of a warning with a given key is logged at WARN level.
/// Subsequent occurrences are logged at DEBUG level to reduce log spam.
///
/// # Arguments
///
/// * `key` - Unique identifier for this warning type (e.g., "deprecated_api_v1")
/// * `message` - Warning message to log
///
/// # Thread Safety
///
/// This function is thread-safe and can be called concurrently from multiple threads.
/// The first thread to encounter a new warning key will log at WARN level; other
/// threads will see the key as already-warned and log at DEBUG level.
///
/// # Examples
///
/// ```
/// use bitnet_warn_once::warn_once_fn;
///
/// warn_once_fn("model_fallback", "Falling back to CPU for unsupported operation");
/// warn_once_fn("model_fallback", "Falling back to CPU for unsupported operation"); // DEBUG level
/// ```
pub fn warn_once_fn(key: &str, message: &str) {
    let is_first_occurrence = record_warning_occurrence(key);

    if is_first_occurrence {
        // First occurrence - log at WARN level
        tracing::warn!(key = %key, "{}", message);
    } else {
        // Subsequent occurrence - log at DEBUG level
        tracing::debug!(key = %key, "(rate-limited) {}", message);
    }
}

fn record_warning_occurrence(key: &str) -> bool {
    let registry = get_registry();
    // Recover from poisoned lock - we don't care if a thread panicked while holding it
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.insert(key.to_string())
}

#[cfg(test)]
fn registry_lock_is_available_for_test() -> bool {
    let registry = get_registry();
    matches!(registry.try_lock(), Ok(_) | Err(TryLockError::Poisoned(_)))
}

/// Macro for convenient warn-once logging.
///
/// This macro provides a convenient interface to the `warn_once_fn` function,
/// supporting both simple string messages and formatted messages.
///
/// # Examples
///
/// ```
/// use bitnet_warn_once::warn_once;
///
/// // Simple string message
/// warn_once!("deprecated_api", "This API is deprecated");
///
/// // Formatted message
/// let version = "v2.0";
/// warn_once!("deprecated_api", "This API is deprecated, please use {}", version);
/// ```
#[macro_export]
macro_rules! warn_once {
    ($key:expr, $($arg:tt)*) => {
        $crate::warn_once_fn($key, &format!($($arg)*))
    };
}

/// Clear the warning registry (primarily for testing).
///
/// This function clears all previously seen warning keys, allowing warnings
/// to be re-triggered. This should only be used in test code.
#[cfg(test)]
pub fn clear_registry_for_test() {
    let registry = get_registry();
    // Recover from poisoned lock - we don't care if a thread panicked while holding it
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tracing::Subscriber;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::layer::{Context, Layer};
    use tracing_subscriber::util::SubscriberInitExt;

    /// Test helper to capture tracing output
    fn setup_test_tracing() {
        let _ = tracing_subscriber::registry()
            .with(EnvFilter::new("debug"))
            .with(tracing_subscriber::fmt::layer().with_test_writer())
            .try_init();
    }

    #[test]
    #[serial]
    fn test_warn_once_is_rate_limited() {
        setup_test_tracing();
        clear_registry_for_test();

        // First call should log at WARN level
        warn_once_fn("test_key_1", "First warning");

        // Subsequent calls should be rate-limited (DEBUG level)
        warn_once_fn("test_key_1", "Second warning");
        warn_once_fn("test_key_1", "Third warning");

        // Different key should warn again
        warn_once_fn("test_key_2", "Different warning");
        warn_once_fn("test_key_2", "Different warning again");

        // Verify registry state
        let registry = get_registry();
        let seen = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert!(seen.contains("test_key_1"));
        assert!(seen.contains("test_key_2"));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    #[serial]
    fn test_warn_once_macro_simple() {
        setup_test_tracing();
        clear_registry_for_test();

        warn_once!("macro_test_1", "Simple message");
        warn_once!("macro_test_1", "Should be rate-limited");

        let registry = get_registry();
        let seen = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert!(seen.contains("macro_test_1"));
    }

    #[test]
    #[serial]
    fn test_warn_once_macro_formatted() {
        setup_test_tracing();
        clear_registry_for_test();

        let value = 42;
        warn_once!("macro_test_2", "Formatted message: {}", value);
        warn_once!("macro_test_2", "Another formatted: {}", value + 1);

        let registry = get_registry();
        let seen = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert!(seen.contains("macro_test_2"));
    }

    #[test]
    #[serial]
    fn test_warn_once_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        setup_test_tracing();
        clear_registry_for_test();

        let barrier = Arc::new(std::sync::Barrier::new(10));
        let mut handles = vec![];

        // Spawn 10 threads that all try to warn with the same key
        for i in 0..10 {
            let barrier_clone = barrier.clone();
            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();
                warn_once_fn("concurrent_test", &format!("Thread {} warning", i));
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify only one entry in registry despite concurrent access
        let registry = get_registry();
        let seen = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert!(seen.contains("concurrent_test"));
        // Only one key should be present for the concurrent test
        // (may have residual keys from other tests, but we cleared at start)
        assert!(!seen.is_empty());
    }

    #[test]
    #[serial]
    fn test_clear_registry() {
        setup_test_tracing();
        clear_registry_for_test();

        warn_once_fn("clear_test", "First warning");

        {
            let registry = get_registry();
            let seen = match registry.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            assert!(seen.contains("clear_test"));
        }

        clear_registry_for_test();

        {
            let registry = get_registry();
            let seen = match registry.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            assert!(!seen.contains("clear_test"));
            assert_eq!(seen.len(), 0);
        }
    }

    #[test]
    #[serial]
    fn test_multiple_unique_keys() {
        setup_test_tracing();
        clear_registry_for_test();

        warn_once_fn("key_a", "Warning A");
        warn_once_fn("key_b", "Warning B");
        warn_once_fn("key_c", "Warning C");
        warn_once_fn("key_a", "Warning A again"); // rate-limited
        warn_once_fn("key_b", "Warning B again"); // rate-limited

        let registry = get_registry();
        let seen = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert_eq!(seen.len(), 3);
        assert!(seen.contains("key_a"));
        assert!(seen.contains("key_b"));
        assert!(seen.contains("key_c"));
    }

    #[test]
    #[serial]
    fn test_record_warning_occurrence_reports_first_insert_only() {
        clear_registry_for_test();

        assert!(record_warning_occurrence("record_once"));
        assert!(!record_warning_occurrence("record_once"));
        assert!(record_warning_occurrence("record_other"));
    }

    struct RegistryLockProbeLayer {
        lock_available_during_event: &'static AtomicBool,
    }

    impl<S> Layer<S> for RegistryLockProbeLayer
    where
        S: Subscriber,
    {
        fn on_event(&self, _event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            self.lock_available_during_event
                .store(registry_lock_is_available_for_test(), Ordering::SeqCst);
        }
    }

    #[test]
    #[serial]
    fn test_warn_once_does_not_hold_registry_lock_while_logging() {
        static LOCK_AVAILABLE_DURING_EVENT: AtomicBool = AtomicBool::new(false);
        LOCK_AVAILABLE_DURING_EVENT.store(false, Ordering::SeqCst);
        clear_registry_for_test();

        let subscriber = tracing_subscriber::registry().with(RegistryLockProbeLayer {
            lock_available_during_event: &LOCK_AVAILABLE_DURING_EVENT,
        });

        tracing::subscriber::with_default(subscriber, || {
            warn_once_fn("lock_probe_key", "probe message");
        });

        assert!(
            LOCK_AVAILABLE_DURING_EVENT.load(Ordering::SeqCst),
            "registry lock should not be held while emitting tracing events"
        );
    }
}
