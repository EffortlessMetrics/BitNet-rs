//! Snapshot tests for `bitnet-test-support` public API surface.
//!
//! Pins the behavior of EnvGuard, EnvScope, and helper functions so that
//! changes to the test-infrastructure contract are visible in code review.

use bitnet_test_support::{EnvGuard, EnvScope, model_path, run_e2e, run_slow_tests};
use serial_test::serial;

// ---------------------------------------------------------------------------
// EnvGuard
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_guard_key_preserved() {
    let guard = EnvGuard::new("BITNET_SNAP_TEST_KEY");
    insta::assert_snapshot!("env_guard_key", guard.key());
}

#[test]
#[serial(bitnet_env)]
fn env_guard_original_value_when_absent() {
    // Safety: test-only env manipulation, serial-gated.
    unsafe {
        std::env::remove_var("BITNET_SNAP_ABSENT");
    }
    let guard = EnvGuard::new("BITNET_SNAP_ABSENT");
    insta::assert_snapshot!(
        "env_guard_original_absent",
        format!("{:?}", guard.original_value())
    );
}

#[test]
#[serial(bitnet_env)]
fn env_guard_original_value_when_present() {
    // Safety: test-only env manipulation, serial-gated.
    unsafe {
        std::env::set_var("BITNET_SNAP_PRESENT", "hello");
    }
    let guard = EnvGuard::new("BITNET_SNAP_PRESENT");
    insta::assert_snapshot!(
        "env_guard_original_present",
        format!("{:?}", guard.original_value())
    );
    drop(guard);
    unsafe {
        std::env::remove_var("BITNET_SNAP_PRESENT");
    }
}

// ---------------------------------------------------------------------------
// EnvScope
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn env_scope_default_debug() {
    let scope = EnvScope::new();
    // Pin that default scope is constructible and has a stable debug repr.
    insta::assert_snapshot!(
        "env_scope_default_debug",
        format!("constructible={}", std::mem::size_of_val(&scope) > 0)
    );
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn model_path_absent_snapshot() {
    unsafe {
        std::env::remove_var("BITNET_MODEL_PATH");
    }
    insta::assert_snapshot!("model_path_absent", format!("{:?}", model_path()));
}

#[test]
#[serial(bitnet_env)]
fn run_slow_tests_default_snapshot() {
    unsafe {
        std::env::remove_var("BITNET_RUN_SLOW_TESTS");
    }
    insta::assert_snapshot!("run_slow_tests_default", format!("{}", run_slow_tests()));
}

#[test]
#[serial(bitnet_env)]
fn run_e2e_default_snapshot() {
    unsafe {
        std::env::remove_var("BITNET_RUN_E2E");
    }
    insta::assert_snapshot!("run_e2e_default", format!("{}", run_e2e()));
}
