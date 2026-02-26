use bitnet_test_support::{EnvGuard, EnvScope, model_path, run_e2e, run_slow_tests};
use proptest::prelude::*;
use serial_test::serial;

fn arb_env_key() -> impl Strategy<Value = String> {
    "[A-Z][A-Z0-9_]{2,15}".prop_map(|s| format!("BITNET_PROPTEST_{}", s))
}

fn arb_env_val() -> impl Strategy<Value = String> {
    "[a-z0-9_/-]{1,32}".prop_filter("no empty", |s| !s.is_empty())
}

// ── EnvGuard properties ─────────────────────────────────────────────────────

proptest! {
    /// key() always returns the key the guard was created with.
    #[test]
    #[serial(bitnet_env)]
    fn env_guard_key_round_trips(key in arb_env_key()) {
        let guard = EnvGuard::new(&key);
        prop_assert_eq!(guard.key(), key.as_str());
    }
}

proptest! {
    /// set() makes the env var readable; remove() makes it unreadable.
    #[test]
    #[serial(bitnet_env)]
    fn env_guard_set_remove_semantics(key in arb_env_key(), val in arb_env_val()) {
        let guard = EnvGuard::new(&key);
        guard.set(&val);
        let read_val = std::env::var(&key).ok();
        prop_assert_eq!(read_val.as_deref(), Some(val.as_str()));
        guard.remove();
        prop_assert!(std::env::var(&key).is_err(), "var should be absent after remove");
    }
}

proptest! {
    /// original_value() is None when the env var was unset before the guard.
    #[test]
    #[serial(bitnet_env)]
    fn env_guard_original_value_unset_when_absent(key in arb_env_key()) {
        // Ensure the key is not set before we create the guard.
        // Safety: test-only env mutation, serial-gated.
        unsafe { std::env::remove_var(&key); }
        let guard = EnvGuard::new(&key);
        prop_assert!(guard.original_value().is_none());
    }
}

proptest! {
    /// original_value() returns the original value when the env var was set before the guard.
    #[test]
    #[serial(bitnet_env)]
    fn env_guard_original_value_preserved(key in arb_env_key(), orig in arb_env_val()) {
        // Safety: test-only env mutation, serial-gated.
        unsafe { std::env::set_var(&key, &orig); }
        let guard = EnvGuard::new(&key);
        prop_assert_eq!(guard.original_value(), Some(orig.as_str()));
        // cleanup
        drop(guard);
        unsafe { std::env::remove_var(&key); }
    }
}

// ── EnvScope properties ─────────────────────────────────────────────────────

proptest! {
    /// EnvScope::set() makes the env var readable.
    #[test]
    #[serial(bitnet_env)]
    fn env_scope_set_is_visible(key in arb_env_key(), val in arb_env_val()) {
        let mut scope = EnvScope::new();
        scope.set(&key, &val);
        let read_val = std::env::var(&key).ok();
        prop_assert_eq!(read_val.as_deref(), Some(val.as_str()));
    }
}

proptest! {
    /// EnvScope::remove() makes the env var absent.
    #[test]
    #[serial(bitnet_env)]
    fn env_scope_remove_clears_var(key in arb_env_key(), val in arb_env_val()) {
        let mut scope = EnvScope::new();
        scope.set(&key, &val);
        scope.remove(&key);
        prop_assert!(std::env::var(&key).is_err());
    }
}

// ── Helper function unit tests ───────────────────────────────────────────────

#[test]
#[serial(bitnet_env)]
fn model_path_absent_by_default() {
    unsafe {
        std::env::remove_var("BITNET_MODEL_PATH");
    }
    assert!(model_path().is_none());
}

#[test]
#[serial(bitnet_env)]
fn model_path_present_when_set() {
    let _guard = EnvGuard::new("BITNET_MODEL_PATH");
    _guard.set("/tmp/test.gguf");
    assert_eq!(model_path().unwrap().to_str(), Some("/tmp/test.gguf"));
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_false_by_default() {
    unsafe {
        std::env::remove_var("BITNET_RUN_SLOW_TESTS");
    }
    assert!(!run_slow_tests());
}

#[test]
#[serial(bitnet_env)]
fn run_e2e_false_by_default() {
    unsafe {
        std::env::remove_var("BITNET_RUN_E2E");
    }
    assert!(!run_e2e());
}
