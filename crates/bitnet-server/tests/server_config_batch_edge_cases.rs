//! Edge-case tests for server config, batch engine, concurrency, and API types.
//!
//! Tests cover: DeviceConfig parsing, ConfigBuilder, ServerSettings defaults,
//! BatchEngineConfig, RequestPriority ordering, BatchRequest builder,
//! ConcurrencyConfig defaults, and serde serialization.

use std::time::Duration;

use bitnet_server::batch_engine::{BatchEngineConfig, BatchRequest, RequestPriority};
use bitnet_server::concurrency::ConcurrencyConfig;
use bitnet_server::config::{DeviceConfig, ServerConfig};

// ---------------------------------------------------------------------------
// DeviceConfig parsing
// ---------------------------------------------------------------------------

#[test]
fn device_config_parse_auto() {
    let dc: DeviceConfig = "auto".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Auto);
}

#[test]
fn device_config_parse_auto_case_insensitive() {
    let dc: DeviceConfig = "AUTO".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Auto);
    let dc2: DeviceConfig = "Auto".parse().unwrap();
    assert_eq!(dc2, DeviceConfig::Auto);
}

#[test]
fn device_config_parse_cpu() {
    let dc: DeviceConfig = "cpu".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Cpu);
}

#[test]
fn device_config_parse_gpu() {
    let dc: DeviceConfig = "gpu".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_cuda() {
    let dc: DeviceConfig = "cuda".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_vulkan() {
    let dc: DeviceConfig = "vulkan".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_opencl() {
    let dc: DeviceConfig = "opencl".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_ocl() {
    let dc: DeviceConfig = "ocl".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_npu() {
    let dc: DeviceConfig = "npu".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_gpu_indexed() {
    let dc: DeviceConfig = "gpu:2".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(2));
}

#[test]
fn device_config_parse_cuda_indexed() {
    let dc: DeviceConfig = "cuda:3".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(3));
}

#[test]
fn device_config_parse_vulkan_indexed() {
    let dc: DeviceConfig = "vulkan:1".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(1));
}

#[test]
fn device_config_parse_opencl_indexed() {
    let dc: DeviceConfig = "opencl:5".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(5));
}

#[test]
fn device_config_parse_ocl_indexed() {
    let dc: DeviceConfig = "ocl:4".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(4));
}

#[test]
fn device_config_parse_gpu_zero() {
    let dc: DeviceConfig = "gpu:0".parse().unwrap();
    assert_eq!(dc, DeviceConfig::Gpu(0));
}

#[test]
fn device_config_parse_invalid() {
    let result: Result<DeviceConfig, _> = "unknown_device".parse();
    assert!(result.is_err());
}

#[test]
fn device_config_parse_invalid_index() {
    let result: Result<DeviceConfig, _> = "gpu:abc".parse();
    assert!(result.is_err());
}

#[test]
fn device_config_default_is_auto() {
    let dc = DeviceConfig::default();
    assert_eq!(dc, DeviceConfig::Auto);
}

#[test]
fn device_config_debug() {
    let dbg = format!("{:?}", DeviceConfig::Gpu(7));
    assert!(dbg.contains("Gpu"));
    assert!(dbg.contains("7"));
}

#[test]
fn device_config_clone_eq() {
    let dc = DeviceConfig::Cpu;
    let dc2 = dc.clone();
    assert_eq!(dc, dc2);
}

#[test]
fn device_config_serde_roundtrip() {
    let configs = [DeviceConfig::Auto, DeviceConfig::Cpu, DeviceConfig::Gpu(3)];
    for cfg in &configs {
        let json = serde_json::to_string(cfg).unwrap();
        let parsed: DeviceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(*cfg, parsed);
    }
}

// ---------------------------------------------------------------------------
// ServerConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn server_config_default_serializable() {
    let config = ServerConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.is_empty());
}

#[test]
fn server_config_default_values() {
    let config = ServerConfig::default();
    assert_eq!(config.server.port, 8080);
    assert_eq!(config.server.host, "0.0.0.0");
}

#[test]
fn server_settings_default_timeouts() {
    let config = ServerConfig::default();
    assert_eq!(config.server.keep_alive, Duration::from_secs(60));
    assert_eq!(config.server.request_timeout, Duration::from_secs(300));
    assert_eq!(config.server.graceful_shutdown_timeout, Duration::from_secs(30));
}

#[test]
fn server_settings_default_model_paths_none() {
    let config = ServerConfig::default();
    assert!(config.server.default_model_path.is_none());
    assert!(config.server.default_tokenizer_path.is_none());
}

// ---------------------------------------------------------------------------
// BatchEngineConfig
// ---------------------------------------------------------------------------

#[test]
fn batch_engine_config_default() {
    let config = BatchEngineConfig::default();
    assert_eq!(config.max_batch_size, 16);
    assert_eq!(config.batch_timeout, Duration::from_millis(100));
    assert_eq!(config.max_concurrent_batches, 4);
    assert!(config.priority_queue_enabled);
    assert!(config.adaptive_batching);
    assert!(config.quantization_aware);
    assert!(config.simd_optimization);
}

#[test]
fn batch_engine_config_debug() {
    let config = BatchEngineConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("BatchEngineConfig"));
}

#[test]
fn batch_engine_config_clone() {
    let config = BatchEngineConfig::default();
    let config2 = config.clone();
    assert_eq!(config2.max_batch_size, 16);
}

#[test]
fn batch_engine_config_serde_roundtrip() {
    let config = BatchEngineConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: BatchEngineConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_batch_size, config.max_batch_size);
}

// ---------------------------------------------------------------------------
// RequestPriority
// ---------------------------------------------------------------------------

#[test]
fn request_priority_ordering() {
    assert!(RequestPriority::Low < RequestPriority::Normal);
    assert!(RequestPriority::Normal < RequestPriority::High);
    assert!(RequestPriority::High < RequestPriority::Critical);
}

#[test]
fn request_priority_eq() {
    assert_eq!(RequestPriority::Normal, RequestPriority::Normal);
    assert_ne!(RequestPriority::Low, RequestPriority::High);
}

#[test]
fn request_priority_copy() {
    let p = RequestPriority::Critical;
    let p2 = p;
    assert_eq!(p, p2);
}

#[test]
fn request_priority_debug() {
    let dbg = format!("{:?}", RequestPriority::High);
    assert!(dbg.contains("High"));
}

#[test]
fn request_priority_min_max() {
    let priorities = [
        RequestPriority::High,
        RequestPriority::Low,
        RequestPriority::Critical,
        RequestPriority::Normal,
    ];
    assert_eq!(*priorities.iter().min().unwrap(), RequestPriority::Low);
    assert_eq!(*priorities.iter().max().unwrap(), RequestPriority::Critical);
}

// ---------------------------------------------------------------------------
// BatchRequest
// ---------------------------------------------------------------------------

#[test]
fn batch_request_new() {
    let config = bitnet_inference::GenerationConfig::default();
    let req = BatchRequest::new("Hello".into(), config);
    assert_eq!(req.prompt, "Hello");
    assert_eq!(req.priority, RequestPriority::Normal);
    assert!(req.device_preference.is_none());
    assert!(req.quantization_hint.is_none());
    assert!(req.timeout.is_none());
    assert!(!req.id.is_empty());
}

#[test]
fn batch_request_builder_chain() {
    let config = bitnet_inference::GenerationConfig::default();
    let req = BatchRequest::new("test".into(), config)
        .with_priority(RequestPriority::Critical)
        .with_device_preference(bitnet_common::Device::Cpu)
        .with_timeout(Duration::from_secs(30))
        .with_quantization_hint("int4".into());

    assert_eq!(req.priority, RequestPriority::Critical);
    assert_eq!(req.device_preference, Some(bitnet_common::Device::Cpu));
    assert_eq!(req.timeout, Some(Duration::from_secs(30)));
    assert_eq!(req.quantization_hint, Some("int4".into()));
}

#[test]
fn batch_request_unique_ids() {
    let config = bitnet_inference::GenerationConfig::default();
    let r1 = BatchRequest::new("a".into(), config.clone());
    let r2 = BatchRequest::new("b".into(), config);
    assert_ne!(r1.id, r2.id);
}

// ---------------------------------------------------------------------------
// ConcurrencyConfig
// ---------------------------------------------------------------------------

#[test]
fn concurrency_config_default() {
    let config = ConcurrencyConfig::default();
    assert!(config.max_concurrent_requests > 0);
}

#[test]
fn concurrency_config_serde_roundtrip() {
    let config = ConcurrencyConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: ConcurrencyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_concurrent_requests, config.max_concurrent_requests);
}

#[test]
fn concurrency_config_debug() {
    let config = ConcurrencyConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("ConcurrencyConfig"));
}
