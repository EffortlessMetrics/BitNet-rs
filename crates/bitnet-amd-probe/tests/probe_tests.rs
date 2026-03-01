use bitnet_amd_probe::{detect_amd_driver, rocm_runtime_available};
use serial_test::serial;

#[test]
#[serial]
fn fake_true_forces_detection() {
    temp_env::with_var("BITNET_AMD_FAKE", Some("true"), || {
        let status = detect_amd_driver();
        assert!(status.found);
        assert!(status.description.contains("BITNET_AMD_FAKE"));
        assert!(rocm_runtime_available());
    });
}

#[test]
#[serial]
fn fake_false_forces_absence() {
    temp_env::with_var("BITNET_AMD_FAKE", Some("false"), || {
        let status = detect_amd_driver();
        assert!(!status.found);
        assert_eq!(status.description, "not found");
        assert!(!rocm_runtime_available());
    });
}

#[test]
#[serial]
fn fake_invalid_falls_back_to_real_probe_without_panicking() {
    temp_env::with_var("BITNET_AMD_FAKE", Some("invalid"), || {
        let _ = detect_amd_driver();
        let _ = rocm_runtime_available();
    });
}
