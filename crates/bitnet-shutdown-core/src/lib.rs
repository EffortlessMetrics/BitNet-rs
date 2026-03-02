//! Shared shutdown signaling primitives.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::info;

/// Process-wide shutdown signal that can be shared across tasks.
#[derive(Clone, Debug)]
pub struct ShutdownSignal {
    shutdown_flag: Arc<AtomicBool>,
}

impl ShutdownSignal {
    /// Create a new shutdown signal in the "not shutting down" state.
    #[must_use]
    pub fn new() -> Self {
        Self { shutdown_flag: Arc::new(AtomicBool::new(false)) }
    }

    /// Returns an [`Arc`] to the underlying atomic flag.
    #[must_use]
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shutdown_flag)
    }

    /// Returns true if shutdown has been initiated.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_flag.load(Ordering::SeqCst)
    }

    /// Marks shutdown as initiated.
    pub fn initiate_shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        info!("Shutdown flag set - new requests will be rejected");
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::ShutdownSignal;

    #[test]
    fn defaults_to_not_shutting_down() {
        let signal = ShutdownSignal::default();
        assert!(!signal.is_shutting_down());
    }

    #[test]
    fn initiate_shutdown_sets_flag() {
        let signal = ShutdownSignal::new();
        signal.initiate_shutdown();
        assert!(signal.is_shutting_down());
    }

    #[test]
    fn cloned_signal_shares_state() {
        let signal = ShutdownSignal::new();
        let clone = signal.clone();

        clone.initiate_shutdown();

        assert!(signal.is_shutting_down());
    }
}
