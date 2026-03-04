//! Engine state machine core.
//!
//! This microcrate isolates state transition contracts for inference engines,
//! keeping transition logic reusable across runtimes and orchestration layers.

use serde::{Deserialize, Serialize};

/// States an inference engine can be in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineState {
    /// Engine is initialised and waiting for work.
    Idle,
    /// Engine is actively generating tokens.
    Running,
    /// Engine has finished generating and is ready to be discarded.
    Done,
}

/// Error produced by invalid [`EngineStateTracker`] transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineStateError(pub String);

impl std::fmt::Display for EngineStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EngineStateError {}

/// Tracks and enforces valid state transitions for an inference engine.
///
/// Valid transitions:
/// - [`EngineState::Idle`] → [`EngineState::Running`] via [`start`](Self::start)
/// - [`EngineState::Running`] → [`EngineState::Done`] via [`finish`](Self::finish)
#[derive(Debug)]
pub struct EngineStateTracker {
    state: EngineState,
}

impl Default for EngineStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineStateTracker {
    /// Create a new tracker in the [`EngineState::Idle`] state.
    pub const fn new() -> Self {
        Self { state: EngineState::Idle }
    }

    /// Return a reference to the current state.
    pub const fn state(&self) -> &EngineState {
        &self.state
    }

    /// Transition `Idle → Running`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineStateError`] if the current state is not [`EngineState::Idle`].
    pub fn start(&mut self) -> Result<(), EngineStateError> {
        if self.state == EngineState::Idle {
            self.state = EngineState::Running;
            Ok(())
        } else {
            Err(EngineStateError(format!("cannot transition to Running from {:?}", self.state)))
        }
    }

    /// Transition `Running → Done`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineStateError`] if the current state is not [`EngineState::Running`].
    pub fn finish(&mut self) -> Result<(), EngineStateError> {
        if self.state == EngineState::Running {
            self.state = EngineState::Done;
            Ok(())
        } else {
            Err(EngineStateError(format!("cannot transition to Done from {:?}", self.state)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_starts_idle() {
        let tracker = EngineStateTracker::new();
        assert_eq!(tracker.state(), &EngineState::Idle);
    }

    #[test]
    fn valid_transition_sequence() {
        let mut tracker = EngineStateTracker::new();
        tracker.start().expect("idle -> running should work");
        assert_eq!(tracker.state(), &EngineState::Running);
        tracker.finish().expect("running -> done should work");
        assert_eq!(tracker.state(), &EngineState::Done);
    }

    #[test]
    fn invalid_transition_reports_error() {
        let mut tracker = EngineStateTracker::new();
        let err = tracker.finish().expect_err("idle -> done should fail");
        assert!(err.0.contains("Done"));
    }

    proptest::proptest! {
        #[test]
        fn only_idle_can_start(started in proptest::bool::ANY) {
            let mut tracker = EngineStateTracker::new();
            if started {
                tracker.start().expect("first start should work");
            }

            let result = tracker.start();
            if started {
                proptest::prop_assert!(result.is_err());
            } else {
                proptest::prop_assert!(result.is_ok());
            }
        }
    }
}
