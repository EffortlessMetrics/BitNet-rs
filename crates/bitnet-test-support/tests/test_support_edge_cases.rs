//! Edge-case tests for bitnet-test-support: EnvGuard, EnvScope, model_path,
//! run_slow_tests, run_e2e.

use bitnet_test_support::{EnvGuard, EnvScope, model_path, run_e2e, run_slow_tests};
use serial_test::serial;
use std::env;

// ---------------------------------------------------------------------------
// model_path
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn model_path_returns_none_when_unset() {
    unsafe { env::remove_var("BITNET_MODEL_PATH") };
    assert!(model_path().is_none());
}

#[test]
#[serial(bitnet_env)]
fn model_path_returns_path_when_set() {
    unsafe { env::set_var("BITNET_MODEL_PATH", "/tmp/model.gguf") };
    let p = model_path();
    assert!(p.is_some());
    assert_eq!(p.unwrap().to_str().unwrap(), "/tmp/model.gguf");
    unsafe { env::remove_var("BITNET_MODEL_PATH") };
}

#[test]
#[serial(bitnet_env)]
fn model_path_handles_empty_string() {
    unsafe { env::set_var("BITNET_MODEL_PATH", "") };
    // Empty string is still Some — it's a valid (empty) path
    assert!(model_path().is_some());
    unsafe { env::remove_var("BITNET_MODEL_PATH") };
}

// ---------------------------------------------------------------------------
// run_slow_tests
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_false_when_unset() {
    unsafe { env::remove_var("BITNET_RUN_SLOW_TESTS") };
    assert!(!run_slow_tests());
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_true_when_one() {
    unsafe { env::set_var("BITNET_RUN_SLOW_TESTS", "1") };
    assert!(run_slow_tests());
    unsafe { env::remove_var("BITNET_RUN_SLOW_TESTS") };
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_false_when_zero() {
    unsafe { env::set_var("BITNET_RUN_SLOW_TESTS", "0") };
    assert!(!run_slow_tests());
    unsafe { env::remove_var("BITNET_RUN_SLOW_TESTS") };
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_false_when_empty() {
    unsafe { env::set_var("BITNET_RUN_SLOW_TESTS", "") };
    assert!(!run_slow_tests());
    unsafe { env::remove_var("BITNET_RUN_SLOW_TESTS") };
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_false_when_true_string() {
    // Only "1" is truthy, not "true"
    unsafe { env::set_var("BITNET_RUN_SLOW_TESTS", "true") };
    assert!(!run_slow_tests());
    unsafe { env::remove_var("BITNET_RUN_SLOW_TESTS") };
}

// ---------------------------------------------------------------------------
// run_e2e
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn run_e2e_false_when_unset() {
    unsafe { env::remove_var("BITNET_RUN_E2E") };
    assert!(!run_e2e());
}

#[test]
#[serial(bitnet_env)]
fn run_e2e_true_when_one() {
    unsafe { env::set_var("BITNET_RUN_E2E", "1") };
    assert!(run_e2e());
    unsafe { env::remove_var("BITNET_RUN_E2E") };
}

#[test]
#[serial(bitnet_env)]
fn run_e2e_false_when_zero() {
    unsafe { env::set_var("BITNET_RUN_E2E", "0") };
    assert!(!run_e2e());
    unsafe { env::remove_var("BITNET_RUN_E2E") };
}

// ---------------------------------------------------------------------------
// EnvGuard: basics
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_guard_key_accessor() {
    let guard = EnvGuard::new("BITNET_EDGE_TEST_KEY_ACCESS");
    assert_eq!(guard.key(), "BITNET_EDGE_TEST_KEY_ACCESS");
}

#[test]
#[serial(bitnet_env)]
fn env_guard_original_value_none_when_unset() {
    unsafe { env::remove_var("BITNET_EDGE_TEST_ORIG") };
    let guard = EnvGuard::new("BITNET_EDGE_TEST_ORIG");
    assert!(guard.original_value().is_none());
}

#[test]
#[serial(bitnet_env)]
fn env_guard_original_value_some_when_set() {
    unsafe { env::set_var("BITNET_EDGE_TEST_ORIG2", "hello") };
    let guard = EnvGuard::new("BITNET_EDGE_TEST_ORIG2");
    assert_eq!(guard.original_value(), Some("hello"));
    drop(guard);
    unsafe { env::remove_var("BITNET_EDGE_TEST_ORIG2") };
}

#[test]
#[serial(bitnet_env)]
fn env_guard_set_changes_value() {
    unsafe { env::remove_var("BITNET_EDGE_TEST_SET") };
    let guard = EnvGuard::new("BITNET_EDGE_TEST_SET");
    guard.set("new_value");
    assert_eq!(env::var("BITNET_EDGE_TEST_SET").unwrap(), "new_value");
    drop(guard);
}

#[test]
#[serial(bitnet_env)]
fn env_guard_remove_unsets_variable() {
    unsafe { env::set_var("BITNET_EDGE_TEST_RM", "to_remove") };
    let guard = EnvGuard::new("BITNET_EDGE_TEST_RM");
    guard.remove();
    assert!(env::var("BITNET_EDGE_TEST_RM").is_err());
    drop(guard);
}

#[test]
#[serial(bitnet_env)]
fn env_guard_restores_on_drop() {
    unsafe { env::set_var("BITNET_EDGE_TEST_RESTORE", "original") };
    {
        let guard = EnvGuard::new("BITNET_EDGE_TEST_RESTORE");
        guard.set("temporary");
        assert_eq!(env::var("BITNET_EDGE_TEST_RESTORE").unwrap(), "temporary");
    }
    assert_eq!(env::var("BITNET_EDGE_TEST_RESTORE").unwrap(), "original");
    unsafe { env::remove_var("BITNET_EDGE_TEST_RESTORE") };
}

#[test]
#[serial(bitnet_env)]
fn env_guard_restores_unset_on_drop() {
    unsafe { env::remove_var("BITNET_EDGE_TEST_RESTORE_UNSET") };
    {
        let guard = EnvGuard::new("BITNET_EDGE_TEST_RESTORE_UNSET");
        guard.set("temporary");
        assert_eq!(env::var("BITNET_EDGE_TEST_RESTORE_UNSET").unwrap(), "temporary");
    }
    assert!(env::var("BITNET_EDGE_TEST_RESTORE_UNSET").is_err());
}

#[test]
#[serial(bitnet_env)]
fn env_guard_multiple_sets_restores_original() {
    unsafe { env::set_var("BITNET_EDGE_TEST_MULTI", "first") };
    {
        let guard = EnvGuard::new("BITNET_EDGE_TEST_MULTI");
        guard.set("second");
        guard.set("third");
        guard.set("fourth");
        assert_eq!(env::var("BITNET_EDGE_TEST_MULTI").unwrap(), "fourth");
    }
    assert_eq!(env::var("BITNET_EDGE_TEST_MULTI").unwrap(), "first");
    unsafe { env::remove_var("BITNET_EDGE_TEST_MULTI") };
}

// ---------------------------------------------------------------------------
// EnvGuard: Debug trait
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_guard_debug_contains_key() {
    let guard = EnvGuard::new("BITNET_EDGE_TEST_DEBUG");
    let d = format!("{:?}", guard);
    assert!(d.contains("BITNET_EDGE_TEST_DEBUG"), "Debug should show key: {d}");
}

// ---------------------------------------------------------------------------
// EnvGuard: panic safety
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_guard_restores_after_panic() {
    unsafe { env::set_var("BITNET_EDGE_TEST_PANIC", "original") };
    let result = std::panic::catch_unwind(|| {
        let guard = EnvGuard::new("BITNET_EDGE_TEST_PANIC");
        guard.set("panicking");
        panic!("test panic");
    });
    assert!(result.is_err());
    assert_eq!(env::var("BITNET_EDGE_TEST_PANIC").unwrap(), "original");
    unsafe { env::remove_var("BITNET_EDGE_TEST_PANIC") };
}

// ---------------------------------------------------------------------------
// EnvScope: basics
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_scope_set_and_restore() {
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_A") };
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_B") };
    {
        let mut scope = EnvScope::new();
        scope.set("BITNET_EDGE_SCOPE_A", "val_a");
        scope.set("BITNET_EDGE_SCOPE_B", "val_b");
        assert_eq!(env::var("BITNET_EDGE_SCOPE_A").unwrap(), "val_a");
        assert_eq!(env::var("BITNET_EDGE_SCOPE_B").unwrap(), "val_b");
    }
    assert!(env::var("BITNET_EDGE_SCOPE_A").is_err());
    assert!(env::var("BITNET_EDGE_SCOPE_B").is_err());
}

#[test]
#[serial(bitnet_env)]
fn env_scope_remove_and_restore() {
    unsafe { env::set_var("BITNET_EDGE_SCOPE_RM", "original") };
    {
        let mut scope = EnvScope::new();
        scope.remove("BITNET_EDGE_SCOPE_RM");
        assert!(env::var("BITNET_EDGE_SCOPE_RM").is_err());
    }
    assert_eq!(env::var("BITNET_EDGE_SCOPE_RM").unwrap(), "original");
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_RM") };
}

#[test]
#[serial(bitnet_env)]
fn env_scope_set_same_key_twice_restores_original() {
    unsafe { env::set_var("BITNET_EDGE_SCOPE_DUP", "original") };
    {
        let mut scope = EnvScope::new();
        scope.set("BITNET_EDGE_SCOPE_DUP", "first");
        scope.set("BITNET_EDGE_SCOPE_DUP", "second");
        assert_eq!(env::var("BITNET_EDGE_SCOPE_DUP").unwrap(), "second");
    }
    // Should restore to original, not "first"
    assert_eq!(env::var("BITNET_EDGE_SCOPE_DUP").unwrap(), "original");
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_DUP") };
}

#[test]
#[serial(bitnet_env)]
fn env_scope_default_trait() {
    let scope = EnvScope::default();
    drop(scope); // just verifying Default works
}

// ---------------------------------------------------------------------------
// EnvScope: mixed set and remove
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_scope_mixed_set_remove() {
    unsafe { env::set_var("BITNET_EDGE_SCOPE_MIX_A", "original_a") };
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_MIX_B") };
    {
        let mut scope = EnvScope::new();
        scope.remove("BITNET_EDGE_SCOPE_MIX_A");
        scope.set("BITNET_EDGE_SCOPE_MIX_B", "new_b");
        assert!(env::var("BITNET_EDGE_SCOPE_MIX_A").is_err());
        assert_eq!(env::var("BITNET_EDGE_SCOPE_MIX_B").unwrap(), "new_b");
    }
    assert_eq!(env::var("BITNET_EDGE_SCOPE_MIX_A").unwrap(), "original_a");
    assert!(env::var("BITNET_EDGE_SCOPE_MIX_B").is_err());
    unsafe { env::remove_var("BITNET_EDGE_SCOPE_MIX_A") };
}
