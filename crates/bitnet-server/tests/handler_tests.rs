//! Handler and routing correctness tests for bitnet-server.
//!
//! These tests validate config types, request/response serialization, and
//! handler logic in isolation — no actual server is started.
#![allow(clippy::field_reassign_with_default)]

use std::net::IpAddr;
use std::time::Duration;

use bitnet_server::batch_engine::{BatchEngineConfig, BatchRequest, RequestPriority};
use bitnet_server::concurrency::ConcurrencyConfig;
use bitnet_server::config::{ConfigBuilder, ServerSettings};
use bitnet_server::execution_router::ExecutionRouterConfig;
use bitnet_server::model_manager::ModelManagerConfig;
use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::security::{SecurityConfig, ValidationError};
use bitnet_server::{
    EnhancedInferenceRequest, EnhancedInferenceResponse, ErrorResponse, InferenceRequest,
    InferenceResponse, ModelLoadRequest, ModelLoadResponse, ServerConfig,
};

// ---------------------------------------------------------------------------
// 1. Server config defaults
// ---------------------------------------------------------------------------

#[test]
fn server_config_default_host_and_port() {
    let config = ServerConfig::default();
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8080);
}

#[test]
fn server_config_default_timeouts() {
    let config = ServerConfig::default();
    assert_eq!(config.server.request_timeout, Duration::from_mins(5));
    assert_eq!(config.server.keep_alive, Duration::from_mins(1));
    assert_eq!(config.server.graceful_shutdown_timeout, Duration::from_secs(30));
}

#[test]
fn server_config_default_model_paths_are_none() {
    let config = ServerConfig::default();
    assert!(config.server.default_model_path.is_none());
    assert!(config.server.default_tokenizer_path.is_none());
}

#[test]
fn server_config_default_workers_is_none() {
    let config = ServerConfig::default();
    assert!(config.server.workers.is_none());
}

// ---------------------------------------------------------------------------
// 2. Request parsing — InferenceRequest
// ---------------------------------------------------------------------------

#[test]
fn inference_request_deserializes_minimal() {
    let json = r#"{"prompt": "Hello"}"#;
    let req: InferenceRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Hello");
    assert!(req.max_tokens.is_none());
    assert!(req.temperature.is_none());
    assert!(req.top_p.is_none());
    assert!(req.top_k.is_none());
    assert!(req.repetition_penalty.is_none());
    assert!(req.model.is_none());
}

#[test]
fn inference_request_deserializes_full() {
    let json = r#"{
        "prompt": "Tell me a joke",
        "max_tokens": 128,
        "model": "bitnet-2b",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2
    }"#;
    let req: InferenceRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Tell me a joke");
    assert_eq!(req.max_tokens, Some(128));
    assert_eq!(req.model.as_deref(), Some("bitnet-2b"));
    assert!((req.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
    assert!((req.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
    assert_eq!(req.top_k, Some(50));
    assert!((req.repetition_penalty.unwrap() - 1.2).abs() < f32::EPSILON);
}

#[test]
fn inference_request_rejects_missing_prompt() {
    let json = r#"{"max_tokens": 10}"#;
    let result = serde_json::from_str::<InferenceRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 3. Request parsing — EnhancedInferenceRequest
// ---------------------------------------------------------------------------

#[test]
fn enhanced_request_deserializes_with_extras() {
    let json = r#"{
        "prompt": "Hi",
        "priority": "high",
        "device_preference": "cpu",
        "quantization_hint": "i2_s",
        "timeout_ms": 5000
    }"#;
    let req: EnhancedInferenceRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.base.prompt, "Hi");
    assert_eq!(req.priority.as_deref(), Some("high"));
    assert_eq!(req.device_preference.as_deref(), Some("cpu"));
    assert_eq!(req.quantization_hint.as_deref(), Some("i2_s"));
    assert_eq!(req.timeout_ms, Some(5000));
}

#[test]
fn enhanced_request_extras_default_to_none() {
    let json = r#"{"prompt": "test"}"#;
    let req: EnhancedInferenceRequest = serde_json::from_str(json).unwrap();
    assert!(req.priority.is_none());
    assert!(req.device_preference.is_none());
    assert!(req.quantization_hint.is_none());
    assert!(req.timeout_ms.is_none());
}

// ---------------------------------------------------------------------------
// 4. Response structure — InferenceResponse
// ---------------------------------------------------------------------------

#[test]
fn inference_response_serializes_all_fields() {
    let resp = InferenceResponse {
        text: "Hello, world!".to_string(),
        tokens_generated: 5,
        inference_time_ms: 100,
        tokens_per_second: 50.0,
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["text"], "Hello, world!");
    assert_eq!(json["tokens_generated"], 5);
    assert_eq!(json["inference_time_ms"], 100);
    assert!((json["tokens_per_second"].as_f64().unwrap() - 50.0).abs() < f64::EPSILON);
}

#[test]
fn enhanced_response_flattens_base() {
    let resp = EnhancedInferenceResponse {
        base: InferenceResponse {
            text: "result".to_string(),
            tokens_generated: 3,
            inference_time_ms: 200,
            tokens_per_second: 15.0,
        },
        device_used: "Cpu".to_string(),
        quantization_type: "i2_s".to_string(),
        batch_id: Some("batch-123".to_string()),
        batch_size: Some(4),
        queue_time_ms: 10,
    };
    let json = serde_json::to_value(&resp).unwrap();
    // base fields are flattened
    assert_eq!(json["text"], "result");
    assert_eq!(json["tokens_generated"], 3);
    // enhanced fields present
    assert_eq!(json["device_used"], "Cpu");
    assert_eq!(json["quantization_type"], "i2_s");
    assert_eq!(json["batch_id"], "batch-123");
    assert_eq!(json["batch_size"], 4);
    assert_eq!(json["queue_time_ms"], 10);
}

// ---------------------------------------------------------------------------
// 5. Error response formatting
// ---------------------------------------------------------------------------

#[test]
fn error_response_serializes_with_all_fields() {
    let err = ErrorResponse {
        error: "Model not found".to_string(),
        error_code: "MODEL_NOT_FOUND".to_string(),
        request_id: Some("req-abc".to_string()),
        details: Some(serde_json::json!({"model_id": "missing"})),
    };
    let json = serde_json::to_value(&err).unwrap();
    assert_eq!(json["error"], "Model not found");
    assert_eq!(json["error_code"], "MODEL_NOT_FOUND");
    assert_eq!(json["request_id"], "req-abc");
    assert_eq!(json["details"]["model_id"], "missing");
}

#[test]
fn error_response_serializes_with_null_optionals() {
    let err = ErrorResponse {
        error: "bad request".to_string(),
        error_code: "BAD_REQUEST".to_string(),
        request_id: None,
        details: None,
    };
    let json = serde_json::to_value(&err).unwrap();
    assert!(json["request_id"].is_null());
    assert!(json["details"].is_null());
}

// ---------------------------------------------------------------------------
// 6. ModelLoadRequest / ModelLoadResponse round-trip
// ---------------------------------------------------------------------------

#[test]
fn model_load_request_deserializes() {
    let json = r#"{
        "model_path": "/models/bitnet.gguf",
        "tokenizer_path": "/models/tokenizer.json",
        "device": "cpu",
        "model_id": "my-model"
    }"#;
    let req: ModelLoadRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model_path, "/models/bitnet.gguf");
    assert_eq!(req.tokenizer_path.as_deref(), Some("/models/tokenizer.json"));
    assert_eq!(req.device.as_deref(), Some("cpu"));
    assert_eq!(req.model_id.as_deref(), Some("my-model"));
}

#[test]
fn model_load_request_minimal() {
    let json = r#"{"model_path": "/models/bitnet.gguf"}"#;
    let req: ModelLoadRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model_path, "/models/bitnet.gguf");
    assert!(req.tokenizer_path.is_none());
    assert!(req.device.is_none());
    assert!(req.model_id.is_none());
}

#[test]
fn model_load_response_serializes() {
    let resp = ModelLoadResponse {
        model_id: "m-1".to_string(),
        status: "success".to_string(),
        message: "loaded".to_string(),
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["model_id"], "m-1");
    assert_eq!(json["status"], "success");
    assert_eq!(json["message"], "loaded");
}

// ---------------------------------------------------------------------------
// 7. Batch request handling edge cases
// ---------------------------------------------------------------------------

#[test]
fn batch_request_gets_unique_ids() {
    let cfg = bitnet_inference::GenerationConfig::default();
    let r1 = BatchRequest::new("a".into(), cfg.clone());
    let r2 = BatchRequest::new("b".into(), cfg);
    assert_ne!(r1.id, r2.id);
}

#[test]
fn batch_request_default_priority_is_normal() {
    let cfg = bitnet_inference::GenerationConfig::default();
    let req = BatchRequest::new("test".into(), cfg);
    assert_eq!(req.priority, RequestPriority::Normal);
}

#[test]
fn batch_engine_config_defaults() {
    let cfg = BatchEngineConfig::default();
    assert_eq!(cfg.max_batch_size, 16);
    assert_eq!(cfg.max_concurrent_batches, 4);
    assert!(cfg.priority_queue_enabled);
    assert!(cfg.adaptive_batching);
    assert!(cfg.quantization_aware);
}

// ---------------------------------------------------------------------------
// 8. Health / monitoring config defaults
// ---------------------------------------------------------------------------

#[test]
fn monitoring_config_defaults() {
    let cfg = MonitoringConfig::default();
    assert_eq!(cfg.health_path, "/health");
    assert!(cfg.structured_logging);
    assert_eq!(cfg.log_level, "info");
}

#[test]
fn monitoring_config_prometheus_path() {
    let cfg = MonitoringConfig::default();
    assert_eq!(cfg.prometheus_path, "/metrics");
}

// ---------------------------------------------------------------------------
// 9. Rate limiting config validation
// ---------------------------------------------------------------------------

#[test]
fn concurrency_config_defaults() {
    let cfg = ConcurrencyConfig::default();
    assert_eq!(cfg.max_concurrent_requests, 100);
    assert_eq!(cfg.max_requests_per_second, 50);
    assert_eq!(cfg.max_requests_per_minute, 1000);
    assert!((cfg.backpressure_threshold - 0.8).abs() < f64::EPSILON);
    assert!(cfg.circuit_breaker_enabled);
    assert_eq!(cfg.per_ip_rate_limit, Some(10));
    assert_eq!(cfg.global_rate_limit, Some(100));
}

#[test]
fn config_builder_rejects_zero_max_concurrent_requests() {
    let mut config = ServerConfig::default();
    config.concurrency.max_concurrent_requests = 0;
    let builder = ConfigBuilder::new().with_concurrency(config.concurrency);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_rejects_invalid_backpressure_threshold() {
    let mut config = ServerConfig::default();
    config.concurrency.backpressure_threshold = 1.5;
    let builder = ConfigBuilder::new().with_concurrency(config.concurrency);
    assert!(builder.validate().is_err());

    let mut config2 = ServerConfig::default();
    config2.concurrency.backpressure_threshold = -0.1;
    let builder2 = ConfigBuilder::new().with_concurrency(config2.concurrency);
    assert!(builder2.validate().is_err());
}

// ---------------------------------------------------------------------------
// 10. CORS config validation
// ---------------------------------------------------------------------------

#[test]
fn security_config_default_allows_wildcard_origin() {
    let cfg = SecurityConfig::default();
    assert!(cfg.allowed_origins.contains(&"*".to_string()));
}

#[test]
fn security_config_default_has_sanitization_enabled() {
    let cfg = SecurityConfig::default();
    assert!(cfg.input_sanitization);
    assert!(cfg.content_filtering);
}

#[test]
fn security_config_default_no_auth_required() {
    let cfg = SecurityConfig::default();
    assert!(!cfg.require_authentication);
    assert!(cfg.jwt_secret.is_none());
}

#[test]
fn config_builder_rejects_auth_without_jwt_secret() {
    let mut sec = SecurityConfig::default();
    sec.require_authentication = true;
    sec.jwt_secret = None;
    let builder = ConfigBuilder::new().with_security(sec);
    assert!(builder.validate().is_err());
}

// ---------------------------------------------------------------------------
// 11. Max tokens bounds
// ---------------------------------------------------------------------------

#[test]
fn security_config_default_max_tokens() {
    let cfg = SecurityConfig::default();
    assert_eq!(cfg.max_tokens_per_request, 2048);
}

#[test]
fn config_builder_rejects_zero_max_tokens() {
    let mut sec = SecurityConfig::default();
    sec.max_tokens_per_request = 0;
    let builder = ConfigBuilder::new().with_security(sec);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_rejects_zero_max_prompt_length() {
    let mut sec = SecurityConfig::default();
    sec.max_prompt_length = 0;
    let builder = ConfigBuilder::new().with_security(sec);
    assert!(builder.validate().is_err());
}

// ---------------------------------------------------------------------------
// 12. Temperature / sampling param validation via SecurityValidator
// ---------------------------------------------------------------------------

#[test]
fn validator_accepts_valid_temperature() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: Some(0.7),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_ok());
}

#[test]
fn validator_rejects_temperature_above_2() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: Some(2.5),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_negative_temperature() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: Some(-0.1),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_top_p_out_of_range() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: Some(1.5),
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_top_k_zero() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(0),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_top_k_above_1000() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: Some(1001),
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_repetition_penalty_out_of_range() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    let req = InferenceRequest {
        prompt: "hello".into(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: Some(0.05),
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// 13. Prompt length and max_tokens validation via SecurityValidator
// ---------------------------------------------------------------------------

#[test]
fn validator_rejects_prompt_exceeding_max_length() {
    let mut sec = SecurityConfig::default();
    sec.max_prompt_length = 10;
    // Disable content filtering so the long prompt isn't rejected for other reasons
    sec.content_filtering = false;
    let validator = bitnet_server::security::SecurityValidator::new(sec).unwrap();
    let req = InferenceRequest {
        prompt: "a".repeat(11),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

#[test]
fn validator_rejects_max_tokens_exceeding_limit() {
    let mut sec = SecurityConfig::default();
    sec.max_tokens_per_request = 100;
    let validator = bitnet_server::security::SecurityValidator::new(sec).unwrap();
    let req = InferenceRequest {
        prompt: "hi".into(),
        max_tokens: Some(200),
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    assert!(validator.validate_inference_request(&req).is_err());
}

// ---------------------------------------------------------------------------
// 14. Model path validation
// ---------------------------------------------------------------------------

#[test]
fn validator_rejects_empty_model_path() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("").is_err());
}

#[test]
fn validator_rejects_path_traversal() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("../../etc/passwd.gguf").is_err());
}

#[test]
fn validator_rejects_non_gguf_extension() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("/models/model.bin").is_err());
}

#[test]
fn validator_accepts_gguf_extension() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("models/model.gguf").is_ok());
}

#[test]
fn validator_accepts_safetensors_extension() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(validator.validate_model_request("models/model.safetensors").is_ok());
}

#[test]
fn validator_rejects_absolute_model_path_without_allowlist() {
    let validator =
        bitnet_server::security::SecurityValidator::new(SecurityConfig::default()).unwrap();
    assert!(matches!(
        validator.validate_model_request("/models/model.gguf"),
        Err(ValidationError::InvalidFieldValue(msg))
            if msg == "Absolute paths are not allowed when allowed_model_directories is empty"
    ));
}

// ---------------------------------------------------------------------------
// 15. Config builder validation — batch engine
// ---------------------------------------------------------------------------

#[test]
fn config_builder_rejects_zero_batch_size() {
    let mut cfg = BatchEngineConfig::default();
    cfg.max_batch_size = 0;
    let builder = ConfigBuilder::new().with_batch_engine(cfg);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_rejects_zero_concurrent_batches() {
    let mut cfg = BatchEngineConfig::default();
    cfg.max_concurrent_batches = 0;
    let builder = ConfigBuilder::new().with_batch_engine(cfg);
    assert!(builder.validate().is_err());
}

// ---------------------------------------------------------------------------
// 16. Config builder validation — model manager
// ---------------------------------------------------------------------------

#[test]
fn config_builder_rejects_zero_concurrent_loads() {
    let mut cfg = ModelManagerConfig::default();
    cfg.max_concurrent_loads = 0;
    let builder = ConfigBuilder::new().with_model_manager(cfg);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_rejects_zero_model_cache_size() {
    let mut cfg = ModelManagerConfig::default();
    cfg.model_cache_size = 0;
    let builder = ConfigBuilder::new().with_model_manager(cfg);
    assert!(builder.validate().is_err());
}

// ---------------------------------------------------------------------------
// 17. Config builder validation — server settings
// ---------------------------------------------------------------------------

#[test]
fn config_builder_rejects_port_zero() {
    let mut settings = ServerSettings::default();
    settings.port = 0;
    let builder = ConfigBuilder::new().with_server_settings(settings);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_rejects_empty_host() {
    let mut settings = ServerSettings::default();
    settings.host = String::new();
    let builder = ConfigBuilder::new().with_server_settings(settings);
    assert!(builder.validate().is_err());
}

#[test]
fn config_builder_valid_default_passes() {
    let builder = ConfigBuilder::new();
    assert!(builder.validate().is_ok());
}

// ---------------------------------------------------------------------------
// 18. Execution router config defaults
// ---------------------------------------------------------------------------

#[test]
fn execution_router_config_defaults() {
    let cfg = ExecutionRouterConfig::default();
    assert!(cfg.fallback_enabled);
    assert!(cfg.benchmark_on_startup);
    assert_eq!(cfg.health_check_interval, Duration::from_secs(30));
    assert!((cfg.performance_threshold_tps - 10.0).abs() < f64::EPSILON);
    assert!((cfg.memory_threshold_percent - 0.8).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// 19. Model manager config defaults
// ---------------------------------------------------------------------------

#[test]
fn model_manager_config_defaults() {
    let cfg = ModelManagerConfig::default();
    assert_eq!(cfg.max_concurrent_loads, 2);
    assert_eq!(cfg.model_cache_size, 3);
    assert_eq!(cfg.load_timeout, Duration::from_mins(5));
    assert!(cfg.validation_enabled);
    assert_eq!(cfg.memory_limit_gb, Some(16.0));
}

// ---------------------------------------------------------------------------
// 20. ServerConfig TOML round-trip
// ---------------------------------------------------------------------------

#[test]
fn server_config_toml_round_trip() {
    let original = ServerConfig::default();
    let toml_str = toml::to_string_pretty(&original).unwrap();
    let deserialized: ServerConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(deserialized.server.host, original.server.host);
    assert_eq!(deserialized.server.port, original.server.port);
    assert_eq!(deserialized.batch_engine.max_batch_size, original.batch_engine.max_batch_size);
}

// ---------------------------------------------------------------------------
// 21. SecurityConfig blocked IPs
// ---------------------------------------------------------------------------

#[test]
fn security_config_default_no_blocked_ips() {
    let cfg = SecurityConfig::default();
    assert!(cfg.blocked_ips.is_empty());
}

#[test]
fn security_config_blocked_ips_are_hashset() {
    let mut cfg = SecurityConfig::default();
    let ip: IpAddr = "192.168.1.1".parse().unwrap();
    cfg.blocked_ips.insert(ip);
    assert!(cfg.blocked_ips.contains(&ip));
    assert_eq!(cfg.blocked_ips.len(), 1);
}
