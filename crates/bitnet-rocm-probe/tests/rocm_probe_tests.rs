use bitnet_rocm_probe::{
    fake_rocm_available_from_env, rocm_available_runtime, strict_mode_enabled,
};

#[test]
#[serial_test::serial(bitnet_env)]
fn strict_mode_false_by_default() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        assert!(!strict_mode_enabled());
    });
}

#[test]
#[serial_test::serial(bitnet_env)]
fn strict_mode_true_variants() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        assert!(strict_mode_enabled());
    });
    temp_env::with_var("BITNET_STRICT_MODE", Some("TrUe"), || {
        assert!(strict_mode_enabled());
    });
}

#[test]
#[serial_test::serial(bitnet_env)]
fn fake_env_none_forces_unavailable() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
            assert_eq!(fake_rocm_available_from_env(), Some(false));
            assert!(!rocm_available_runtime());
        });
    });
}

#[test]
#[serial_test::serial(bitnet_env)]
fn fake_env_rocm_token_forces_available() {
    temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("cuda,rocm"), || {
            assert_eq!(fake_rocm_available_from_env(), Some(true));
            assert!(rocm_available_runtime());
        });
    });
}

#[test]
#[serial_test::serial(bitnet_env)]
fn strict_mode_ignores_fake_env() {
    temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
        temp_env::with_var("BITNET_GPU_FAKE", Some("rocm"), || {
            assert_eq!(fake_rocm_available_from_env(), None);
            let _ = rocm_available_runtime();
        });
    });
}
