//! Edge-case tests for bitnet-server config, DeviceConfig parsing, and ConfigBuilder validation.

use bitnet_server::ServerConfig;
use bitnet_server::config::{ConfigBuilder, DeviceConfig, ServerSettings};
use std::str::FromStr;
use std::time::Duration;

// ---------------------------------------------------------------------------
// DeviceConfig parsing
// ---------------------------------------------------------------------------

#[test]
fn device_config_parse_auto() {
    let dc: DeviceConfig = "auto".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Auto));
}

#[test]
fn device_config_parse_cpu() {
    let dc: DeviceConfig = "cpu".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Cpu));
}

#[test]
fn device_config_parse_gpu() {
    let dc: DeviceConfig = "gpu".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_cuda() {
    let dc: DeviceConfig = "cuda".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_vulkan() {
    let dc: DeviceConfig = "vulkan".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_opencl() {
    let dc: DeviceConfig = "opencl".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_ocl() {
    let dc: DeviceConfig = "ocl".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_npu() {
    let dc: DeviceConfig = "npu".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_gpu_with_id() {
    let dc: DeviceConfig = "gpu:3".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(3)));
}

#[test]
fn device_config_parse_cuda_with_id() {
    let dc: DeviceConfig = "cuda:1".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(1)));
}

#[test]
fn device_config_parse_vulkan_with_id() {
    let dc: DeviceConfig = "vulkan:2".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(2)));
}

#[test]
fn device_config_parse_opencl_with_id() {
    let dc: DeviceConfig = "opencl:0".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_ocl_with_id() {
    let dc: DeviceConfig = "ocl:5".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(5)));
}

#[test]
fn device_config_parse_case_insensitive() {
    let dc: DeviceConfig = "AUTO".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Auto));
    let dc: DeviceConfig = "CPU".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Cpu));
    let dc: DeviceConfig = "GPU".parse().unwrap();
    assert!(matches!(dc, DeviceConfig::Gpu(0)));
}

#[test]
fn device_config_parse_invalid() {
    assert!(DeviceConfig::from_str("unknown_device").is_err());
}

#[test]
fn device_config_parse_gpu_invalid_id() {
    assert!(DeviceConfig::from_str("gpu:abc").is_err());
}

#[test]
fn device_config_default_is_auto() {
    let dc = DeviceConfig::default();
    assert!(matches!(dc, DeviceConfig::Auto));
}

// ---------------------------------------------------------------------------
// DeviceConfig resolve
// ---------------------------------------------------------------------------

#[test]
fn device_config_cpu_resolves_to_cpu() {
    use bitnet_common::Device;
    let dc = DeviceConfig::Cpu;
    assert!(matches!(dc.resolve(), Device::Cpu));
}

#[test]
fn device_config_gpu_resolves_to_cuda() {
    use bitnet_common::Device;
    let dc = DeviceConfig::Gpu(2);
    assert!(matches!(dc.resolve(), Device::Cuda(2)));
}

// ---------------------------------------------------------------------------
// ServerSettings defaults
// ---------------------------------------------------------------------------

#[test]
fn server_settings_default_host() {
    let ss = ServerSettings::default();
    assert_eq!(ss.host, "0.0.0.0");
}

#[test]
fn server_settings_default_port() {
    let ss = ServerSettings::default();
    assert_eq!(ss.port, 8080);
}

#[test]
fn server_settings_default_workers_none() {
    let ss = ServerSettings::default();
    assert!(ss.workers.is_none());
}

#[test]
fn server_settings_default_keep_alive() {
    let ss = ServerSettings::default();
    assert_eq!(ss.keep_alive, Duration::from_secs(60));
}

#[test]
fn server_settings_default_timeout() {
    let ss = ServerSettings::default();
    assert_eq!(ss.request_timeout, Duration::from_secs(300));
}

// ---------------------------------------------------------------------------
// ConfigBuilder defaults and validation
// ---------------------------------------------------------------------------

#[test]
fn config_builder_default_builds() {
    let config = ConfigBuilder::new().build();
    assert_eq!(config.server.port, 8080);
}

#[test]
fn config_builder_validate_rejects_port_zero() {
    let builder = ConfigBuilder::new()
        .with_server_settings(ServerSettings { port: 0, ..ServerSettings::default() });
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_validate_rejects_empty_host() {
    let builder = ConfigBuilder::new()
        .with_server_settings(ServerSettings { host: String::new(), ..ServerSettings::default() });
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_validate_default_passes() {
    let result = ConfigBuilder::new().validate();
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// ServerConfig serialization
// ---------------------------------------------------------------------------

#[test]
fn server_config_default_serializes() {
    let config = ServerConfig::default();
    let json = serde_json::to_string(&config);
    assert!(json.is_ok());
}

#[test]
fn device_config_serde_roundtrip() {
    let dc = DeviceConfig::Gpu(3);
    let json = serde_json::to_string(&dc).unwrap();
    let dc2: DeviceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(dc, dc2);
}

#[test]
fn device_config_auto_serde_roundtrip() {
    let dc = DeviceConfig::Auto;
    let json = serde_json::to_string(&dc).unwrap();
    let dc2: DeviceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(dc, dc2);
}

#[test]
fn device_config_cpu_serde_roundtrip() {
    let dc = DeviceConfig::Cpu;
    let json = serde_json::to_string(&dc).unwrap();
    let dc2: DeviceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(dc, dc2);
}
