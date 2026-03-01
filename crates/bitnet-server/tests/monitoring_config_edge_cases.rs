//! Edge-case tests for bitnet-server monitoring configuration validation.

use bitnet_server::monitoring::MonitoringConfig;
use bitnet_server::monitoring::config::{
    ConfigValidator, InferenceSettings, ResourceSettings, ServerSettings, ValidatedConfig,
};

// =========================================================================
// MonitoringConfig
// =========================================================================

#[test]
fn monitoring_config_default() {
    let cfg = MonitoringConfig::default();
    assert!(cfg.prometheus_enabled);
    assert_eq!(cfg.prometheus_path, "/metrics");
    assert!(!cfg.opentelemetry_enabled);
    assert!(cfg.opentelemetry_endpoint.is_none());
    assert!(cfg.otlp_endpoint.is_none());
    assert_eq!(cfg.health_path, "/health");
    assert_eq!(cfg.metrics_interval, 10);
    assert!(cfg.structured_logging);
    assert_eq!(cfg.log_level, "info");
    assert_eq!(cfg.log_format, "json");
}

#[test]
fn monitoring_config_clone() {
    let cfg = MonitoringConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cfg.prometheus_path, cloned.prometheus_path);
    assert_eq!(cfg.log_level, cloned.log_level);
}

#[test]
fn monitoring_config_debug() {
    let cfg = MonitoringConfig::default();
    let debug = format!("{:?}", cfg);
    assert!(debug.contains("MonitoringConfig"));
}

#[test]
fn monitoring_config_serde_roundtrip() {
    let cfg = MonitoringConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let deserialized: MonitoringConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.prometheus_path, deserialized.prometheus_path);
    assert_eq!(cfg.log_level, deserialized.log_level);
}

// =========================================================================
// ValidatedConfig
// =========================================================================

#[test]
fn validated_config_default() {
    let cfg = ValidatedConfig::default();
    assert_eq!(cfg.server.host, "0.0.0.0");
    assert_eq!(cfg.server.port, 8080);
    assert!(cfg.server.workers.is_none());
    assert_eq!(cfg.server.max_connections, Some(1000));
    assert_eq!(cfg.server.request_timeout_seconds, 300);
    assert_eq!(cfg.server.graceful_shutdown_timeout_seconds, 30);

    assert_eq!(cfg.inference.default_model, "bitnet-1.58b");
    assert_eq!(cfg.inference.max_tokens_per_request, 2048);
    assert_eq!(cfg.inference.max_concurrent_requests, 100);
    assert!(cfg.inference.enable_batching);

    assert!(cfg.resources.max_memory_mb.is_none());
    assert!(cfg.resources.max_gpu_memory_mb.is_none());
    assert!(cfg.resources.enable_memory_monitoring);
    assert!((cfg.resources.memory_warning_threshold_percent - 80.0).abs() < f64::EPSILON);
    assert!((cfg.resources.memory_critical_threshold_percent - 90.0).abs() < f64::EPSILON);
}

#[test]
fn validated_config_clone() {
    let cfg = ValidatedConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cfg.server.port, cloned.server.port);
    assert_eq!(cfg.inference.default_model, cloned.inference.default_model);
}

#[test]
fn validated_config_debug() {
    let cfg = ValidatedConfig::default();
    let debug = format!("{:?}", cfg);
    assert!(debug.contains("ValidatedConfig"));
}

#[test]
fn validated_config_serde_roundtrip() {
    let cfg = ValidatedConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let deserialized: ValidatedConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.server.port, deserialized.server.port);
    assert_eq!(cfg.inference.default_model, deserialized.inference.default_model);
}

// =========================================================================
// ConfigValidator — valid configurations
// =========================================================================

#[test]
fn config_validator_default_is_valid() {
    let cfg = ValidatedConfig::default();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.is_valid, "Default config should be valid, errors: {:?}", result.errors);
}

#[test]
fn config_validator_default_may_have_warnings() {
    // The default config is valid but may have warnings
    let cfg = ValidatedConfig::default();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.is_valid);
    // Warnings are acceptable for the default config
}

// =========================================================================
// ConfigValidator — server setting errors
// =========================================================================

#[test]
fn config_validator_empty_host() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.host = String::new();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("host")));
}

#[test]
fn config_validator_port_zero() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.port = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("port")));
}

#[test]
fn config_validator_privileged_port_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.port = 80;
    cfg.server.host = "0.0.0.0".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    // Low port on non-localhost should generate a warning
    assert!(result.warnings.iter().any(|w| w.contains("root") || w.contains("privileges")));
}

#[test]
fn config_validator_privileged_port_localhost_no_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.port = 80;
    cfg.server.host = "127.0.0.1".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    // Localhost with low port should NOT warn
    assert!(!result.warnings.iter().any(|w| w.contains("root")));
}

#[test]
fn config_validator_zero_workers() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.workers = Some(0);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Worker") || e.contains("worker")));
}

#[test]
fn config_validator_zero_max_connections() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.max_connections = Some(0);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("connections")));
}

#[test]
fn config_validator_high_max_connections_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.max_connections = Some(20000);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("connections")));
}

#[test]
fn config_validator_zero_request_timeout() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.request_timeout_seconds = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("timeout")));
}

#[test]
fn config_validator_high_request_timeout_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.request_timeout_seconds = 7200;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("timeout")));
}

#[test]
fn config_validator_high_shutdown_timeout_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.server.graceful_shutdown_timeout_seconds = 600;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("shutdown")));
}

// =========================================================================
// ConfigValidator — monitoring errors
// =========================================================================

#[test]
fn config_validator_prometheus_path_no_slash() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.prometheus_path = "metrics".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Prometheus") && e.contains("/")));
}

#[test]
fn config_validator_prometheus_path_dotdot() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.prometheus_path = "/../../etc".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Prometheus") && e.contains("..")));
}

#[test]
fn config_validator_health_path_no_slash() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.health_path = "health".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Health") || e.contains("health")));
}

#[test]
fn config_validator_invalid_log_level() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.log_level = "verbose".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("log level")));
}

#[test]
fn config_validator_valid_log_levels() {
    for level in &["trace", "debug", "info", "warn", "error"] {
        let mut cfg = ValidatedConfig::default();
        cfg.monitoring.log_level = level.to_string();
        let mut validator = ConfigValidator::new();
        let result = validator.validate(&cfg);
        // Should have no log level error
        assert!(
            !result.errors.iter().any(|e| e.contains("log level")),
            "Log level '{}' should be valid",
            level
        );
    }
}

#[test]
fn config_validator_invalid_log_format() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.log_format = "xml".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("log format")));
}

#[test]
fn config_validator_valid_log_formats() {
    for format in &["json", "pretty", "compact"] {
        let mut cfg = ValidatedConfig::default();
        cfg.monitoring.log_format = format.to_string();
        let mut validator = ConfigValidator::new();
        let result = validator.validate(&cfg);
        assert!(
            !result.errors.iter().any(|e| e.contains("log format")),
            "Log format '{}' should be valid",
            format
        );
    }
}

#[test]
fn config_validator_metrics_interval_zero() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.metrics_interval = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Metrics interval")));
}

#[test]
fn config_validator_metrics_interval_low_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.metrics_interval = 2;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("Metrics interval")));
}

#[test]
fn config_validator_otel_enabled_no_endpoint_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.opentelemetry_enabled = true;
    cfg.monitoring.opentelemetry_endpoint = None;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("OpenTelemetry")));
}

#[test]
fn config_validator_otel_invalid_endpoint() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.opentelemetry_enabled = true;
    cfg.monitoring.opentelemetry_endpoint = Some("ftp://bad".to_string());
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("OpenTelemetry")));
}

#[test]
fn config_validator_otel_valid_endpoint() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.opentelemetry_enabled = true;
    cfg.monitoring.opentelemetry_endpoint = Some("http://localhost:4317".to_string());
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.errors.iter().any(|e| e.contains("OpenTelemetry")));
}

// =========================================================================
// ConfigValidator — inference setting errors
// =========================================================================

#[test]
fn config_validator_empty_default_model() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.default_model = String::new();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("model")));
}

#[test]
fn config_validator_zero_max_tokens() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.max_tokens_per_request = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("tokens")));
}

#[test]
fn config_validator_high_max_tokens_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.max_tokens_per_request = 65536;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("tokens")));
}

#[test]
fn config_validator_zero_concurrent_requests() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.max_concurrent_requests = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("concurrent")));
}

#[test]
fn config_validator_high_concurrent_requests_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.max_concurrent_requests = 5000;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("concurrent")));
}

#[test]
fn config_validator_batching_zero_timeout() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.enable_batching = true;
    cfg.inference.batch_timeout_ms = 0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Batch timeout")));
}

#[test]
fn config_validator_high_batch_timeout_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.inference.batch_timeout_ms = 5000;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("Batch timeout")));
}

// =========================================================================
// ConfigValidator — resource setting errors
// =========================================================================

#[test]
fn config_validator_zero_max_memory() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.max_memory_mb = Some(0);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("memory")));
}

#[test]
fn config_validator_low_max_memory_warning() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.max_memory_mb = Some(512);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(result.warnings.iter().any(|w| w.contains("memory")));
}

#[test]
fn config_validator_zero_gpu_memory() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.max_gpu_memory_mb = Some(0);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("GPU memory")));
}

#[test]
fn config_validator_zero_cpu_threads() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.cpu_threads = Some(0);
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("thread")));
}

#[test]
fn config_validator_invalid_warning_threshold() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.memory_warning_threshold_percent = 0.0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("warning threshold")));
}

#[test]
fn config_validator_invalid_critical_threshold() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.memory_critical_threshold_percent = 101.0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("critical threshold")));
}

#[test]
fn config_validator_warning_exceeds_critical() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.memory_warning_threshold_percent = 95.0;
    cfg.resources.memory_critical_threshold_percent = 90.0;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("less than critical")));
}

// =========================================================================
// ConfigValidator — cross-validation
// =========================================================================

#[test]
fn config_validator_duplicate_paths() {
    let mut cfg = ValidatedConfig::default();
    cfg.monitoring.prometheus_path = "/health".to_string();
    cfg.monitoring.health_path = "/health".to_string();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    assert!(!result.is_valid);
    assert!(result.errors.iter().any(|e| e.contains("Duplicate")));
}

#[test]
fn config_validator_memory_insufficient_for_concurrency() {
    let mut cfg = ValidatedConfig::default();
    cfg.resources.max_memory_mb = Some(2048);
    cfg.inference.max_concurrent_requests = 100;
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    // 100 * 100MB = 10GB > 2GB → warning expected
    assert!(result.warnings.iter().any(|w| w.contains("insufficient") || w.contains("Memory")));
}

// =========================================================================
// ConfigValidator — reuse/reset
// =========================================================================

#[test]
fn config_validator_reuse_clears_state() {
    let mut validator = ConfigValidator::new();

    // Validate an invalid config first
    let mut bad_cfg = ValidatedConfig::default();
    bad_cfg.server.port = 0;
    let result1 = validator.validate(&bad_cfg);
    assert!(!result1.is_valid);

    // Then validate a valid config
    let good_cfg = ValidatedConfig::default();
    let result2 = validator.validate(&good_cfg);
    assert!(result2.is_valid, "Reused validator should clear errors, got: {:?}", result2.errors);
}

// =========================================================================
// ValidationResult
// =========================================================================

#[test]
fn validation_result_debug() {
    let cfg = ValidatedConfig::default();
    let mut validator = ConfigValidator::new();
    let result = validator.validate(&cfg);
    let debug = format!("{:?}", result);
    assert!(debug.contains("ValidationResult"));
}

// =========================================================================
// ServerSettings / InferenceSettings / ResourceSettings
// =========================================================================

#[test]
fn server_settings_debug() {
    let cfg = ValidatedConfig::default();
    let debug = format!("{:?}", cfg.server);
    assert!(debug.contains("ServerSettings"));
}

#[test]
fn inference_settings_debug() {
    let cfg = ValidatedConfig::default();
    let debug = format!("{:?}", cfg.inference);
    assert!(debug.contains("InferenceSettings"));
}

#[test]
fn resource_settings_debug() {
    let cfg = ValidatedConfig::default();
    let debug = format!("{:?}", cfg.resources);
    assert!(debug.contains("ResourceSettings"));
}

#[test]
fn server_settings_serde_roundtrip() {
    let settings = ServerSettings {
        host: "192.168.1.1".to_string(),
        port: 9090,
        workers: Some(8),
        max_connections: Some(500),
        request_timeout_seconds: 120,
        graceful_shutdown_timeout_seconds: 15,
    };
    let json = serde_json::to_string(&settings).unwrap();
    let deserialized: ServerSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(settings.host, deserialized.host);
    assert_eq!(settings.port, deserialized.port);
    assert_eq!(settings.workers, deserialized.workers);
}

#[test]
fn inference_settings_serde_roundtrip() {
    let settings = InferenceSettings {
        default_model: "phi-4-14b".to_string(),
        max_tokens_per_request: 8192,
        max_concurrent_requests: 50,
        model_cache_size: 5,
        enable_batching: false,
        batch_timeout_ms: 0,
    };
    let json = serde_json::to_string(&settings).unwrap();
    let deserialized: InferenceSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(settings.default_model, deserialized.default_model);
    assert_eq!(settings.max_tokens_per_request, deserialized.max_tokens_per_request);
}

#[test]
fn resource_settings_serde_roundtrip() {
    let settings = ResourceSettings {
        max_memory_mb: Some(32768),
        max_gpu_memory_mb: Some(16384),
        cpu_threads: Some(16),
        enable_memory_monitoring: true,
        memory_warning_threshold_percent: 70.0,
        memory_critical_threshold_percent: 85.0,
    };
    let json = serde_json::to_string(&settings).unwrap();
    let deserialized: ResourceSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(settings.max_memory_mb, deserialized.max_memory_mb);
    assert_eq!(settings.max_gpu_memory_mb, deserialized.max_gpu_memory_mb);
}
