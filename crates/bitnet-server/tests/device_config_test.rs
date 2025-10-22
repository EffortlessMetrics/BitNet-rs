//! Integration tests for configurable device selection in BitNet server

use bitnet_common::Device;
use bitnet_server::config::{ConfigBuilder, DeviceConfig};
use std::env;

#[test]
fn test_device_config_cpu_mode() {
    let mut config = ConfigBuilder::new().build();
    config.server.default_device = DeviceConfig::Cpu;

    let device = config.server.default_device.resolve();
    assert_eq!(device, Device::Cpu);
}

#[test]
fn test_device_config_gpu_mode() {
    let mut config = ConfigBuilder::new().build();
    config.server.default_device = DeviceConfig::Gpu(1);

    let device = config.server.default_device.resolve();
    assert_eq!(device, Device::Cuda(1));
}

#[test]
fn test_device_config_auto_mode() {
    let mut config = ConfigBuilder::new().build();
    config.server.default_device = DeviceConfig::Auto;

    let device = config.server.default_device.resolve();
    // Auto mode should resolve to CPU or GPU based on feature flags and runtime
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    assert_eq!(device, Device::Cpu);

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        // In GPU builds, device can be CPU or CUDA(0) based on runtime availability
        assert!(device == Device::Cpu || device == Device::Cuda(0));
    }
}

#[test]
fn test_device_config_from_env() {
    // Test CPU config from env
    unsafe {
        env::set_var("BITNET_DEFAULT_DEVICE", "cpu");
    }

    let config = ConfigBuilder::new().from_env().unwrap().build();
    assert_eq!(config.server.default_device, DeviceConfig::Cpu);

    // Test GPU config from env
    unsafe {
        env::set_var("BITNET_DEFAULT_DEVICE", "gpu:2");
    }

    let config = ConfigBuilder::new().from_env().unwrap().build();
    assert_eq!(config.server.default_device, DeviceConfig::Gpu(2));

    // Test Auto config from env
    unsafe {
        env::set_var("BITNET_DEFAULT_DEVICE", "auto");
    }

    let config = ConfigBuilder::new().from_env().unwrap().build();
    assert_eq!(config.server.default_device, DeviceConfig::Auto);

    // Cleanup
    unsafe {
        env::remove_var("BITNET_DEFAULT_DEVICE");
    }
}

#[test]
fn test_device_config_default_is_auto() {
    let config = ConfigBuilder::new().build();
    assert_eq!(config.server.default_device, DeviceConfig::Auto);
}

#[test]
fn test_device_config_parse_variants() {
    // Test all valid string formats
    assert_eq!("cpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
    assert_eq!("CPU".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
    assert_eq!("auto".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
    assert_eq!("AUTO".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
    assert_eq!("gpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("GPU".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("gpu:0".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("gpu:1".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(1));
    assert_eq!("cuda:0".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("cuda:3".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(3));

    // Test invalid formats
    assert!("invalid".parse::<DeviceConfig>().is_err());
    assert!("gpu:abc".parse::<DeviceConfig>().is_err());
    assert!("cuda:".parse::<DeviceConfig>().is_err());
}
