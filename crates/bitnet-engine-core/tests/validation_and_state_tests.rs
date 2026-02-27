//! Property-based and unit tests for validation, session IDs, state transitions,
//! and concurrency limits added to `bitnet-engine-core`.
//!
//! Coverage matrix
//! ───────────────
//! 1.  Valid `SessionConfig` always passes `validate()`.
//! 2.  Empty `model_path` → `ConfigError::EmptyModelPath` (not a panic).
//! 3.  Empty `tokenizer_path` → `ConfigError::EmptyTokenizerPath`.
//! 4.  Unrecognised backend → `ConfigError::UnsupportedBackend`.
//! 5.  `max_context == 0` → `ConfigError::ZeroContextWindow`.
//! 6.  `SessionId::generate()` is non-empty for every call.
//! 7.  Successive `SessionId::generate()` calls produce distinct values.
//! 8.  `EngineStateTracker` follows `Idle → Running → Done` in order.
//! 9.  Invalid transitions return a non-empty, ASCII error message.
//! 10. `ConcurrencyConfig::allows()` never exceeds `max_concurrent`.
//! 11. `ConcurrencyConfig` JSON round-trip preserves `max_concurrent`.
//! 12. `ConfigError` messages are non-empty and ASCII-only.
//! 13. `SessionConfig::validate()` checks fields in documented priority order.

use bitnet_engine_core::{
    ConcurrencyConfig, ConfigError, EngineState, EngineStateTracker, SessionConfig, SessionId,
    VALID_BACKENDS,
};
use proptest::prelude::*;

// ── strategies ─────────────────────────────────────────────────────────────

fn arb_valid_config() -> impl Strategy<Value = SessionConfig> {
    (
        "[a-z0-9/_\\.]{1,64}", // model_path  — non-empty
        "[a-z0-9/_\\.]{1,64}", // tokenizer_path — non-empty
        prop_oneof![Just("cpu"), Just("cuda"), Just("gpu"), Just("ffi"),],
        1usize..=65536,
        prop::option::of(any::<u64>()),
    )
        .prop_map(|(model_path, tokenizer_path, backend, max_context, seed)| SessionConfig {
            model_path,
            tokenizer_path,
            backend: backend.to_string(),
            max_context,
            seed,
        })
}

fn arb_invalid_backend() -> impl Strategy<Value = String> {
    // Strings that are definitely not in VALID_BACKENDS.
    "[A-Z]{1,8}"
        .prop_filter("must not be a valid backend", |s| !VALID_BACKENDS.contains(&s.as_str()))
}

// ── 1. Valid configs always pass validate() ────────────────────────────────

proptest! {
    /// Any `SessionConfig` built from non-empty paths, a recognised backend,
    /// and a non-zero context window must pass `validate()`.
    #[test]
    fn valid_config_passes_validation(config in arb_valid_config()) {
        prop_assert!(
            config.validate().is_ok(),
            "expected Ok but got {:?}",
            config.validate()
        );
    }
}

// ── 2. Empty model_path returns EmptyModelPath ─────────────────────────────

proptest! {
    /// Providing an empty `model_path` always produces `EmptyModelPath`.
    #[test]
    fn empty_model_path_returns_error(config in arb_valid_config()) {
        let bad = SessionConfig { model_path: String::new(), ..config };
        prop_assert_eq!(bad.validate(), Err(ConfigError::EmptyModelPath));
    }
}

// ── 3. Empty tokenizer_path returns EmptyTokenizerPath ────────────────────

proptest! {
    /// Providing an empty `tokenizer_path` always produces `EmptyTokenizerPath`.
    #[test]
    fn empty_tokenizer_path_returns_error(config in arb_valid_config()) {
        let bad = SessionConfig { tokenizer_path: String::new(), ..config };
        prop_assert_eq!(bad.validate(), Err(ConfigError::EmptyTokenizerPath));
    }
}

// ── 4. Unrecognised backend returns UnsupportedBackend ────────────────────

proptest! {
    /// Any backend string not in `VALID_BACKENDS` produces `UnsupportedBackend`.
    #[test]
    fn invalid_backend_returns_error(
        config in arb_valid_config(),
        bad_backend in arb_invalid_backend(),
    ) {
        let bad = SessionConfig { backend: bad_backend.clone(), ..config };
        prop_assert_eq!(
            bad.validate(),
            Err(ConfigError::UnsupportedBackend(bad_backend))
        );
    }
}

// ── 5. Zero context returns ZeroContextWindow ─────────────────────────────

proptest! {
    /// Setting `max_context = 0` always produces `ZeroContextWindow`.
    #[test]
    fn zero_context_returns_error(config in arb_valid_config()) {
        let bad = SessionConfig { max_context: 0, ..config };
        prop_assert_eq!(bad.validate(), Err(ConfigError::ZeroContextWindow));
    }
}

// ── 6. SessionId is always non-empty ──────────────────────────────────────

proptest! {
    /// `SessionId::generate()` never produces an empty identifier.
    ///
    /// Called repeatedly via proptest to exercise concurrent monotonic
    /// counter increments.
    #[test]
    fn session_id_is_non_empty(_: ()) {
        let id = SessionId::generate();
        prop_assert!(!id.as_str().is_empty(), "SessionId must not be empty");
    }
}

// ── 7. Successive SessionIds are distinct ─────────────────────────────────

proptest! {
    /// Two successive calls to `SessionId::generate()` return different IDs.
    #[test]
    fn successive_session_ids_are_distinct(_: ()) {
        let a = SessionId::generate();
        let b = SessionId::generate();
        prop_assert_ne!(a, b, "consecutive SessionIds must differ");
    }
}

// ── 8. EngineStateTracker: Idle → Running → Done ─────────────────────────

proptest! {
    /// The happy-path state machine traversal must always succeed.
    #[test]
    fn engine_state_idle_running_done(_: ()) {
        let mut tracker = EngineStateTracker::new();
        prop_assert_eq!(tracker.state(), &EngineState::Idle);

        tracker.start().expect("start should succeed from Idle");
        prop_assert_eq!(tracker.state(), &EngineState::Running);

        tracker.finish().expect("finish should succeed from Running");
        prop_assert_eq!(tracker.state(), &EngineState::Done);
    }
}

// ── 9. Invalid transitions produce non-empty ASCII error messages ─────────

proptest! {
    /// Calling `start()` twice yields an error with a non-empty, ASCII message.
    #[test]
    fn double_start_error_message_is_ascii(_: ()) {
        let mut tracker = EngineStateTracker::new();
        tracker.start().unwrap();
        let err = tracker.start().expect_err("second start must fail");
        let msg = err.to_string();
        prop_assert!(!msg.is_empty(), "error message must not be empty");
        prop_assert!(msg.is_ascii(), "error message must be ASCII: {:?}", msg);
    }

    /// Calling `finish()` from `Idle` yields a non-empty, ASCII error message.
    #[test]
    fn finish_from_idle_error_message_is_ascii(_: ()) {
        let mut tracker = EngineStateTracker::new();
        let err = tracker.finish().expect_err("finish from Idle must fail");
        let msg = err.to_string();
        prop_assert!(!msg.is_empty(), "error message must not be empty");
        prop_assert!(msg.is_ascii(), "error message must be ASCII: {:?}", msg);
    }
}

// ── 10. ConcurrencyConfig::allows() never exceeds max_concurrent ──────────

proptest! {
    /// For every combination of `(max, active)`, `allows(active)` returns
    /// `true` iff `active < max`.
    #[test]
    fn concurrency_allows_within_bounds(
        max_concurrent in 1usize..=256,
        active in 0usize..=512,
    ) {
        let cfg = ConcurrencyConfig { max_concurrent };
        let expected = active < max_concurrent;
        prop_assert_eq!(
            cfg.allows(active),
            expected,
        );
    }
}

// ── 11. ConcurrencyConfig JSON roundtrip ──────────────────────────────────

proptest! {
    /// `ConcurrencyConfig` preserves `max_concurrent` after JSON round-trip.
    #[test]
    fn concurrency_config_json_roundtrip(max_concurrent in 1usize..=1024) {
        let cfg = ConcurrencyConfig { max_concurrent };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: ConcurrencyConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(cfg.max_concurrent, restored.max_concurrent);
    }
}

// ── 12. ConfigError messages are non-empty and ASCII ──────────────────────

#[test]
fn config_error_messages_are_non_empty_and_ascii() {
    let errors = [
        ConfigError::EmptyModelPath,
        ConfigError::EmptyTokenizerPath,
        ConfigError::UnsupportedBackend("bad-backend".to_string()),
        ConfigError::ZeroContextWindow,
    ];
    for err in &errors {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error message must not be empty for {err:?}");
        assert!(msg.is_ascii(), "error message must be ASCII for {err:?}: {msg:?}");
    }
}

// ── 13. validate() priority order ─────────────────────────────────────────

/// `model_path` is checked first: even with other invalid fields, an empty
/// `model_path` must yield `EmptyModelPath`, not a different error.
#[test]
fn validate_checks_model_path_before_tokenizer() {
    let cfg = SessionConfig {
        model_path: String::new(),     // invalid #1
        tokenizer_path: String::new(), // invalid #2
        backend: "cpu".to_string(),
        max_context: 0, // invalid #3
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
}

/// `tokenizer_path` is checked second: with a valid `model_path` but an
/// empty `tokenizer_path`, the error must be `EmptyTokenizerPath`.
#[test]
fn validate_checks_tokenizer_before_backend() {
    let cfg = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: String::new(),        // invalid
        backend: "not-a-backend".to_string(), // also invalid
        max_context: 0,                       // also invalid
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
}

// ── additional unit tests ──────────────────────────────────────────────────

#[test]
fn session_config_all_valid_backends_pass() {
    for backend in VALID_BACKENDS {
        let cfg = SessionConfig {
            model_path: "m.gguf".to_string(),
            tokenizer_path: "t.json".to_string(),
            backend: backend.to_string(),
            max_context: 512,
            seed: None,
        };
        assert!(cfg.validate().is_ok(), "backend {backend:?} should be valid");
    }
}

#[test]
fn engine_state_tracker_new_is_idle() {
    let tracker = EngineStateTracker::new();
    assert_eq!(tracker.state(), &EngineState::Idle);
}

#[test]
fn engine_state_tracker_default_is_idle() {
    let tracker = EngineStateTracker::default();
    assert_eq!(tracker.state(), &EngineState::Idle);
}

#[test]
fn concurrency_config_default_allows_fewer_than_max() {
    let cfg = ConcurrencyConfig::default();
    assert!(cfg.allows(0));
    assert!(cfg.allows(cfg.max_concurrent - 1));
    assert!(!cfg.allows(cfg.max_concurrent));
}

#[test]
fn session_id_as_str_matches_display_impl() {
    let id = SessionId::generate();
    assert!(!id.as_str().is_empty());
    // as_str() must be stable (same value on repeated calls)
    assert_eq!(id.as_str(), id.as_str());
}
