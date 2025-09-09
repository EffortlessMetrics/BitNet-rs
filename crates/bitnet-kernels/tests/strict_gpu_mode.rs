//! Tests for strict GPU mode validation that prevents fake GPU backends

use bitnet_kernels::gpu_utils;
use serial_test::serial;
use temp_env::with_vars;

#[test]
#[should_panic(expected = "BITNET_GPU_FAKE is set but strict mode forbids fake GPU")]
#[serial(bitnet_env)]
fn strict_mode_disallows_fake_gpu() {
    // Set both fake GPU and strict mode
    with_vars(
        [("BITNET_GPU_FAKE", Some("cuda")), ("BITNET_STRICT_NO_FAKE_GPU", Some("1"))],
        || {
            // This should panic due to strict mode
            gpu_utils::get_gpu_info();
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn normal_mode_allows_fake_gpu() {
    with_vars(
        [
            ("BITNET_STRICT_NO_FAKE_GPU", None::<&str>), // Ensure strict mode is not set
            ("BITNET_GPU_FAKE", Some("cuda")),
        ],
        || {
            // Should work fine in normal mode
            let info = gpu_utils::get_gpu_info();
            assert!(info.cuda, "fake cuda should be detected");
        },
    );
}

#[test]
#[serial(bitnet_env)]
fn strict_mode_works_with_real_gpu_detection() {
    with_vars(
        [
            ("BITNET_GPU_FAKE", None::<&str>), // Ensure fake GPU is not set
            ("BITNET_STRICT_NO_FAKE_GPU", Some("1")), // Enable strict mode
        ],
        || {
            // Should work fine with real GPU detection
            let _info = gpu_utils::get_gpu_info();
            // Test passes if no panic occurs
        },
    );
}
