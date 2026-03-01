//! BDD tests for Intel GPU detection logic.

#[cfg(feature = "oneapi")]
use serial_test::serial;

mod oneapi_detection {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn given_no_feature_when_oneapi_compiled_checked_then_false() {
        #[cfg(not(feature = "oneapi"))]
        assert!(!bitnet_device_probe::oneapi_compiled());
    }

    #[cfg(feature = "oneapi")]
    #[test]
    fn given_oneapi_feature_when_compiled_checked_then_true() {
        assert!(bitnet_device_probe::oneapi_compiled());
    }

    #[cfg(feature = "oneapi")]
    #[test]
    #[serial(bitnet_env)]
    fn given_gpu_fake_oneapi_when_runtime_checked_then_available() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("oneapi"), || {
                assert!(bitnet_device_probe::oneapi_available_runtime());
            });
        });
    }

    #[cfg(feature = "oneapi")]
    #[test]
    #[serial(bitnet_env)]
    fn given_gpu_fake_none_when_runtime_checked_then_not_available() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
                assert!(!bitnet_device_probe::oneapi_available_runtime());
            });
        });
    }

    #[cfg(feature = "oneapi")]
    #[test]
    #[serial(bitnet_env)]
    fn given_gpu_fake_gpu_when_runtime_checked_then_oneapi_available() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("gpu"), || {
                assert!(bitnet_device_probe::oneapi_available_runtime());
            });
        });
    }
}
