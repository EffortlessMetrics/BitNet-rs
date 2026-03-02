use bitnet_test_env::{EnvGuard, EnvScope};
use bitnet_test_gating_core::{env_flag_enabled, model_path, run_e2e, run_slow_tests};
use serial_test::serial;

#[test]
#[serial(bitnet_env)]
fn model_path_none_when_absent() {
    let guard = EnvGuard::new("BITNET_MODEL_PATH");
    guard.remove();
    assert!(model_path().is_none());
}

#[test]
#[serial(bitnet_env)]
fn model_path_present_when_set() {
    let guard = EnvGuard::new("BITNET_MODEL_PATH");
    guard.set("/tmp/model.gguf");
    assert_eq!(
        model_path().and_then(|p| p.into_os_string().into_string().ok()),
        Some(String::from("/tmp/model.gguf"))
    );
}

#[test]
#[serial(bitnet_env)]
fn env_flag_enabled_only_for_one() {
    let guard = EnvGuard::new("BITNET_FLAG_TEST");

    guard.remove();
    assert!(!env_flag_enabled("BITNET_FLAG_TEST"));

    guard.set("0");
    assert!(!env_flag_enabled("BITNET_FLAG_TEST"));

    guard.set("1");
    assert!(env_flag_enabled("BITNET_FLAG_TEST"));
}

#[test]
#[serial(bitnet_env)]
fn run_slow_and_e2e_delegate_to_flag_helper() {
    let mut scope = EnvScope::new();
    scope.set("BITNET_RUN_SLOW_TESTS", "1");
    scope.set("BITNET_RUN_E2E", "0");

    assert!(run_slow_tests());
    assert!(!run_e2e());
}
