//! Edge-case tests for config module: DeviceConfig, ServerSettings, ServerConfig,
//! ConfigBuilder, validation rules, generate_example_config, and serde roundtrips.

use bitnet_common::Device;
use bitnet_server::config::{ConfigBuilder, DeviceConfig, ServerConfig, ServerSettings};

// ─── DeviceConfig ───────────────────────────────────────────────────

#[test]
fn device_config_default_is_auto() {
    assert_eq!(DeviceConfig::default(), DeviceConfig::Auto);
}

#[test]
fn device_config_from_str_auto() {
    assert_eq!("auto".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
    assert_eq!("AUTO".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
    assert_eq!("Auto".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
}

#[test]
fn device_config_from_str_cpu() {
    assert_eq!("cpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
    assert_eq!("CPU".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
}

#[test]
fn device_config_from_str_gpu_variants() {
    assert_eq!("gpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("cuda".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("vulkan".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("opencl".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("ocl".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("npu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
}

#[test]
fn device_config_from_str_gpu_with_id() {
    assert_eq!("gpu:0".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
    assert_eq!("gpu:1".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(1));
    assert_eq!("gpu:7".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(7));
    assert_eq!("cuda:2".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(2));
    assert_eq!("vulkan:3".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(3));
    assert_eq!("opencl:4".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(4));
    assert_eq!("ocl:5".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(5));
}

#[test]
fn device_config_from_str_invalid() {
    assert!("invalid".parse::<DeviceConfig>().is_err());
    assert!("metal".parse::<DeviceConfig>().is_err());
    assert!("tpu".parse::<DeviceConfig>().is_err());
    assert!("".parse::<DeviceConfig>().is_err());
}

#[test]
fn device_config_from_str_invalid_gpu_id() {
    assert!("gpu:abc".parse::<DeviceConfig>().is_err());
    assert!("cuda:-1".parse::<DeviceConfig>().is_err());
}

#[test]
fn device_config_resolve_cpu() {
    assert_eq!(DeviceConfig::Cpu.resolve(), Device::Cpu);
}

#[test]
fn device_config_resolve_gpu() {
    assert_eq!(DeviceConfig::Gpu(0).resolve(), Device::Cuda(0));
    assert_eq!(DeviceConfig::Gpu(3).resolve(), Device::Cuda(3));
}

#[test]
fn device_config_debug() {
    let d = format!("{:?}", DeviceConfig::Auto);
    assert!(d.contains("Auto"));
    let d = format!("{:?}", DeviceConfig::Gpu(1));
    assert!(d.contains("Gpu"));
    assert!(d.contains("1"));
}

#[test]
fn device_config_clone() {
    let config = DeviceConfig::Gpu(2);
    let cloned = config.clone();
    assert_eq!(cloned, DeviceConfig::Gpu(2));
}

#[test]
fn device_config_serde_roundtrip() {
    for config in
        [DeviceConfig::Auto, DeviceConfig::Cpu, DeviceConfig::Gpu(0), DeviceConfig::Gpu(3)]
    {
        let json = serde_json::to_string(&config).unwrap();
        let deser: DeviceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, config);
    }
}

// ─── ServerSettings ─────────────────────────────────────────────────

#[test]
fn server_settings_defaults() {
    let settings = ServerSettings::default();
    assert_eq!(settings.host, "0.0.0.0");
    assert_eq!(settings.port, 8080);
    assert!(settings.workers.is_none());
    assert_eq!(settings.keep_alive, std::time::Duration::from_secs(60));
    assert_eq!(settings.request_timeout, std::time::Duration::from_secs(300));
    assert_eq!(settings.graceful_shutdown_timeout, std::time::Duration::from_secs(30));
    assert!(settings.default_model_path.is_none());
    assert!(settings.default_tokenizer_path.is_none());
    assert_eq!(settings.default_device, DeviceConfig::Auto);
}

#[test]
fn server_settings_debug() {
    let settings = ServerSettings::default();
    let debug = format!("{:?}", settings);
    assert!(debug.contains("ServerSettings"));
    assert!(debug.contains("0.0.0.0"));
}

#[test]
fn server_settings_clone() {
    let settings = ServerSettings::default();
    let cloned = settings.clone();
    assert_eq!(cloned.port, 8080);
}

#[test]
fn server_settings_serde_roundtrip() {
    let settings = ServerSettings::default();
    let json = serde_json::to_string(&settings).unwrap();
    let deser: ServerSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.host, "0.0.0.0");
    assert_eq!(deser.port, 8080);
}

// ─── ServerConfig ───────────────────────────────────────────────────

#[test]
fn server_config_default() {
    let config = ServerConfig::default();
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8080);
}

#[test]
fn server_config_debug() {
    let config = ServerConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ServerConfig"));
}

#[test]
fn server_config_clone() {
    let config = ServerConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.server.port, config.server.port);
}

#[test]
fn server_config_serde_roundtrip() {
    let config = ServerConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deser: ServerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.server.host, "0.0.0.0");
    assert_eq!(deser.server.port, 8080);
}

// ─── ConfigBuilder ──────────────────────────────────────────────────

#[test]
fn config_builder_new_builds_defaults() {
    let config = ConfigBuilder::new().build();
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8080);
}

#[test]
fn config_builder_default() {
    let config = ConfigBuilder::default().build();
    assert_eq!(config.server.port, 8080);
}

#[test]
fn config_builder_with_server_settings() {
    let settings =
        ServerSettings { host: "127.0.0.1".to_string(), port: 3000, ..ServerSettings::default() };
    let config = ConfigBuilder::new().with_server_settings(settings).build();
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 3000);
}

#[test]
fn config_builder_validate_ok() {
    let result = ConfigBuilder::new().validate();
    assert!(result.is_ok());
    let config = result.ok().unwrap().build();
    assert_eq!(config.server.port, 8080);
}

#[test]
fn config_builder_validate_zero_port() {
    let settings = ServerSettings { port: 0, ..ServerSettings::default() };
    let result = ConfigBuilder::new().with_server_settings(settings).validate();
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("port"));
}

#[test]
fn config_builder_validate_empty_host() {
    let settings = ServerSettings { host: "".to_string(), ..ServerSettings::default() };
    let result = ConfigBuilder::new().with_server_settings(settings).validate();
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("host"));
}

#[test]
fn config_builder_validate_auth_without_jwt_secret() {
    use bitnet_server::security::SecurityConfig;
    let sec = SecurityConfig {
        require_authentication: true,
        jwt_secret: None,
        ..SecurityConfig::default()
    };
    let result = ConfigBuilder::new().with_security(sec).validate();
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("JWT"));
}

#[test]
fn config_builder_validate_auth_with_jwt_secret() {
    use bitnet_server::security::SecurityConfig;
    let sec = SecurityConfig {
        require_authentication: true,
        jwt_secret: Some("my-secret".to_string()),
        ..SecurityConfig::default()
    };
    let result = ConfigBuilder::new().with_security(sec).validate();
    assert!(result.is_ok());
    let config = result.ok().unwrap().build();
    assert!(config.security.require_authentication);
}

#[test]
fn config_builder_validate_zero_max_prompt_length() {
    use bitnet_server::security::SecurityConfig;
    let sec = SecurityConfig { max_prompt_length: 0, ..SecurityConfig::default() };
    let result = ConfigBuilder::new().with_security(sec).validate();
    assert!(result.is_err());
}

#[test]
fn config_builder_validate_zero_max_tokens() {
    use bitnet_server::security::SecurityConfig;
    let sec = SecurityConfig { max_tokens_per_request: 0, ..SecurityConfig::default() };
    let result = ConfigBuilder::new().with_security(sec).validate();
    assert!(result.is_err());
}

#[test]
fn config_builder_validate_backpressure_out_of_range() {
    use bitnet_server::concurrency::ConcurrencyConfig;
    let conc = ConcurrencyConfig { backpressure_threshold: 1.5, ..ConcurrencyConfig::default() };
    let result = ConfigBuilder::new().with_concurrency(conc).validate();
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Backpressure"));
}

#[test]
fn config_builder_validate_negative_backpressure() {
    use bitnet_server::concurrency::ConcurrencyConfig;
    let conc = ConcurrencyConfig { backpressure_threshold: -0.1, ..ConcurrencyConfig::default() };
    let result = ConfigBuilder::new().with_concurrency(conc).validate();
    assert!(result.is_err());
}

#[test]
fn config_builder_chaining() {
    use bitnet_server::concurrency::ConcurrencyConfig;
    use bitnet_server::execution_router::ExecutionRouterConfig;
    let config = ConfigBuilder::new()
        .with_server_settings(ServerSettings { port: 9090, ..ServerSettings::default() })
        .with_concurrency(ConcurrencyConfig {
            max_concurrent_requests: 50,
            ..ConcurrencyConfig::default()
        })
        .with_execution_router(ExecutionRouterConfig::default())
        .build();
    assert_eq!(config.server.port, 9090);
    assert_eq!(config.concurrency.max_concurrent_requests, 50);
}

// ─── generate_example_config ────────────────────────────────────────

#[test]
fn generate_example_config_is_nonempty() {
    let toml = bitnet_server::config::generate_example_config();
    assert!(!toml.is_empty());
    assert!(toml.contains("[server]"));
}

#[test]
fn generate_example_config_contains_defaults() {
    let toml = bitnet_server::config::generate_example_config();
    assert!(toml.contains("host"));
    assert!(toml.contains("port"));
    assert!(toml.contains("8080"));
}

#[test]
fn generate_example_config_is_valid_toml() {
    let toml_str = bitnet_server::config::generate_example_config();
    let parsed: Result<ServerConfig, _> = toml::from_str(&toml_str);
    assert!(parsed.is_ok(), "Generated TOML should parse back: {:?}", parsed.err());
}
