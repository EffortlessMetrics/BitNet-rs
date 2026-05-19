//! Thread-safe warn-once utility for rate-limited logging.

mod warn_once;

pub use warn_once::logging::warn_once_fn;

#[macro_export]
macro_rules! warn_once {
    ($key:expr, $($arg:tt)*) => {
        $crate::warn_once_fn($key, &format!($($arg)*))
    };
}

#[cfg(test)]
pub fn clear_registry_for_test() {
    warn_once::registry::clear_registry_for_test();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tracing::Subscriber;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::layer::{Context, Layer};

    #[test]
    #[serial]
    fn test_warn_once_is_rate_limited() {
        clear_registry_for_test();
        warn_once_fn("test_key_1", "First warning");
        warn_once_fn("test_key_1", "Second warning");
        warn_once_fn("test_key_2", "Different warning");
        assert!(warn_once::registry::contains_key_for_test("test_key_1"));
        assert!(warn_once::registry::contains_key_for_test("test_key_2"));
        assert_eq!(warn_once::registry::key_count_for_test(), 2);
    }

    #[test]
    #[serial]
    fn test_warn_once_macro_formatted() {
        clear_registry_for_test();
        let value = 42;
        warn_once!("macro_test_2", "Formatted message: {}", value);
        warn_once!("macro_test_2", "Another formatted: {}", value + 1);
        assert!(warn_once::registry::contains_key_for_test("macro_test_2"));
    }

    #[test]
    #[serial]
    fn test_record_warning_occurrence_reports_first_insert_only() {
        clear_registry_for_test();
        assert!(warn_once::registry::record_warning_occurrence("record_once"));
        assert!(!warn_once::registry::record_warning_occurrence("record_once"));
        assert!(warn_once::registry::record_warning_occurrence("record_other"));
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
                .store(warn_once::registry::lock_available_for_test(), Ordering::SeqCst);
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

        assert!(LOCK_AVAILABLE_DURING_EVENT.load(Ordering::SeqCst));
    }
}
