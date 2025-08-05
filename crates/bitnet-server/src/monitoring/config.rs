//! Configuration validation and management for monitoring

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::MonitoringConfig;

/// Comprehensive server configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedConfig {
    pub server: ServerSettings,
    pub monitoring: MonitoringConfig,
    pub inference: InferenceSettings,
    pub resources: ResourceSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: Option<usize>,
    pub request_timeout_seconds: u64,
    pub graceful_shutdown_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSettings {
    pub default_model: String,
    pub max_tokens_per_request: usize,
    pub max_concurrent_requests: usize,
    pub model_cache_size: usize,
    pub enable_batching: bool,
    pub batch_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSettings {
    pub max_memory_mb: Option<usize>,
    pub max_gpu_memory_mb: Option<usize>,
    pub cpu_threads: Option<usize>,
    pub enable_memory_monitoring: bool,
    pub memory_warning_threshold_percent: f64,
    pub memory_critical_threshold_percent: f64,
}

impl Default for ValidatedConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: None,
                max_connections: Some(1000),
                request_timeout_seconds: 300,
                graceful_shutdown_timeout_seconds: 30,
            },
            monitoring: MonitoringConfig::default(),
            inference: InferenceSettings {
                default_model: "bitnet-1.58b".to_string(),
                max_tokens_per_request: 2048,
                max_concurrent_requests: 100,
                model_cache_size: 3,
                enable_batching: true,
                batch_timeout_ms: 10,
            },
            resources: ResourceSettings {
                max_memory_mb: None,
                max_gpu_memory_mb: None,
                cpu_threads: None,
                enable_memory_monitoring: true,
                memory_warning_threshold_percent: 80.0,
                memory_critical_threshold_percent: 90.0,
            },
        }
    }
}

/// Configuration validator with comprehensive error reporting
pub struct ConfigValidator {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl ConfigValidator {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Validate a configuration and return detailed results
    pub fn validate(&mut self, config: &ValidatedConfig) -> ValidationResult {
        self.errors.clear();
        self.warnings.clear();

        // Validate server settings
        self.validate_server_settings(&config.server);
        
        // Validate monitoring settings
        self.validate_monitoring_settings(&config.monitoring);
        
        // Validate inference settings
        self.validate_inference_settings(&config.inference);
        
        // Validate resource settings
        self.validate_resource_settings(&config.resources);
        
        // Cross-validate settings
        self.cross_validate(config);

        ValidationResult {
            is_valid: self.errors.is_empty(),
            errors: self.errors.clone(),
            warnings: self.warnings.clone(),
        }
    }

    fn validate_server_settings(&mut self, settings: &ServerSettings) {
        // Validate host
        if settings.host.is_empty() {
            self.errors.push("Server host cannot be empty".to_string());
        }

        // Validate port
        if settings.port == 0 {
            self.errors.push("Server port cannot be 0".to_string());
        } else if settings.port < 1024 && settings.host != "127.0.0.1" && settings.host != "localhost" {
            self.warnings.push(format!(
                "Port {} requires root privileges on most systems",
                settings.port
            ));
        }

        // Validate workers
        if let Some(workers) = settings.workers {
            if workers == 0 {
                self.errors.push("Worker count cannot be 0".to_string());
            } else if workers > num_cpus::get() * 2 {
                self.warnings.push(format!(
                    "Worker count {} exceeds 2x CPU cores ({}), may cause performance issues",
                    workers, num_cpus::get()
                ));
            }
        }

        // Validate max connections
        if let Some(max_conn) = settings.max_connections {
            if max_conn == 0 {
                self.errors.push("Max connections cannot be 0".to_string());
            } else if max_conn > 10000 {
                self.warnings.push(format!(
                    "Max connections {} is very high, ensure system limits are configured",
                    max_conn
                ));
            }
        }

        // Validate timeouts
        if settings.request_timeout_seconds == 0 {
            self.errors.push("Request timeout cannot be 0".to_string());
        } else if settings.request_timeout_seconds > 3600 {
            self.warnings.push("Request timeout > 1 hour may cause resource issues".to_string());
        }

        if settings.graceful_shutdown_timeout_seconds > 300 {
            self.warnings.push("Graceful shutdown timeout > 5 minutes is unusually long".to_string());
        }
    }

    fn validate_monitoring_settings(&mut self, settings: &MonitoringConfig) {
        // Validate Prometheus settings
        if settings.prometheus_enabled {
            if !settings.prometheus_path.starts_with('/') {
                self.errors.push("Prometheus path must start with '/'".to_string());
            }
            if settings.prometheus_path.contains("..") {
                self.errors.push("Prometheus path cannot contain '..'".to_string());
            }
        }

        // Validate OpenTelemetry settings
        if settings.opentelemetry_enabled {
            if let Some(endpoint) = &settings.opentelemetry_endpoint {
                if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                    self.errors.push("OpenTelemetry endpoint must be a valid HTTP(S) URL".to_string());
                }
            } else {
                self.warnings.push("OpenTelemetry enabled but no endpoint specified, using stdout".to_string());
            }
        }

        // Validate health check settings
        if !settings.health_path.starts_with('/') {
            self.errors.push("Health path must start with '/'".to_string());
        }

        // Validate metrics interval
        if settings.metrics_interval == 0 {
            self.errors.push("Metrics interval cannot be 0".to_string());
        } else if settings.metrics_interval < 5 {
            self.warnings.push("Metrics interval < 5 seconds may cause high CPU usage".to_string());
        }

        // Validate log settings
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&settings.log_level.as_str()) {
            self.errors.push(format!(
                "Invalid log level '{}', must be one of: {}",
                settings.log_level,
                valid_levels.join(", ")
            ));
        }

        let valid_formats = ["json", "pretty", "compact"];
        if !valid_formats.contains(&settings.log_format.as_str()) {
            self.errors.push(format!(
                "Invalid log format '{}', must be one of: {}",
                settings.log_format,
                valid_formats.join(", ")
            ));
        }
    }

    fn validate_inference_settings(&mut self, settings: &InferenceSettings) {
        // Validate model name
        if settings.default_model.is_empty() {
            self.errors.push("Default model name cannot be empty".to_string());
        }

        // Validate token limits
        if settings.max_tokens_per_request == 0 {
            self.errors.push("Max tokens per request cannot be 0".to_string());
        } else if settings.max_tokens_per_request > 32768 {
            self.warnings.push("Max tokens per request > 32K may cause memory issues".to_string());
        }

        // Validate concurrency
        if settings.max_concurrent_requests == 0 {
            self.errors.push("Max concurrent requests cannot be 0".to_string());
        } else if settings.max_concurrent_requests > 1000 {
            self.warnings.push("Max concurrent requests > 1000 may overwhelm the system".to_string());
        }

        // Validate cache settings
        if settings.model_cache_size == 0 {
            self.warnings.push("Model cache size is 0, models will be reloaded frequently".to_string());
        } else if settings.model_cache_size > 10 {
            self.warnings.push("Large model cache may consume significant memory".to_string());
        }

        // Validate batching settings
        if settings.enable_batching && settings.batch_timeout_ms == 0 {
            self.errors.push("Batch timeout cannot be 0 when batching is enabled".to_string());
        } else if settings.batch_timeout_ms > 1000 {
            self.warnings.push("Batch timeout > 1s may increase latency".to_string());
        }
    }

    fn validate_resource_settings(&mut self, settings: &ResourceSettings) {
        // Validate memory limits
        if let Some(max_mem) = settings.max_memory_mb {
            if max_mem == 0 {
                self.errors.push("Max memory cannot be 0".to_string());
            } else if max_mem < 1024 {
                self.warnings.push("Max memory < 1GB may be insufficient for model loading".to_string());
            }
        }

        if let Some(max_gpu_mem) = settings.max_gpu_memory_mb {
            if max_gpu_mem == 0 {
                self.errors.push("Max GPU memory cannot be 0".to_string());
            }
        }

        // Validate CPU threads
        if let Some(threads) = settings.cpu_threads {
            if threads == 0 {
                self.errors.push("CPU threads cannot be 0".to_string());
            } else if threads > num_cpus::get() * 4 {
                self.warnings.push(format!(
                    "CPU threads {} significantly exceeds available cores ({})",
                    threads, num_cpus::get()
                ));
            }
        }

        // Validate thresholds
        if settings.memory_warning_threshold_percent <= 0.0 || settings.memory_warning_threshold_percent > 100.0 {
            self.errors.push("Memory warning threshold must be between 0 and 100".to_string());
        }

        if settings.memory_critical_threshold_percent <= 0.0 || settings.memory_critical_threshold_percent > 100.0 {
            self.errors.push("Memory critical threshold must be between 0 and 100".to_string());
        }

        if settings.memory_warning_threshold_percent >= settings.memory_critical_threshold_percent {
            self.errors.push("Memory warning threshold must be less than critical threshold".to_string());
        }
    }

    fn cross_validate(&mut self, config: &ValidatedConfig) {
        // Check if monitoring endpoints conflict with server endpoints
        let monitoring_paths = vec![
            &config.monitoring.prometheus_path,
            &config.monitoring.health_path,
        ];

        let mut path_counts = HashMap::new();
        for path in monitoring_paths {
            *path_counts.entry(path).or_insert(0) += 1;
        }

        for (path, count) in path_counts {
            if count > 1 {
                self.errors.push(format!("Duplicate endpoint path: {}", path));
            }
        }

        // Check resource allocation consistency
        if let (Some(max_mem), Some(max_concurrent)) = (
            config.resources.max_memory_mb,
            Some(config.inference.max_concurrent_requests)
        ) {
            let estimated_memory_per_request = 100; // MB, rough estimate
            let total_estimated = max_concurrent * estimated_memory_per_request;
            
            if total_estimated > max_mem {
                self.warnings.push(format!(
                    "Memory limit ({} MB) may be insufficient for max concurrent requests ({}) - estimated need: {} MB",
                    max_mem, max_concurrent, total_estimated
                ));
            }
        }
    }
}

/// Configuration validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Print validation results to console
    pub fn print_results(&self) {
        if !self.errors.is_empty() {
            println!("❌ Configuration Errors:");
            for error in &self.errors {
                println!("   • {}", error);
            }
        }

        if !self.warnings.is_empty() {
            println!("⚠️  Configuration Warnings:");
            for warning in &self.warnings {
                println!("   • {}", warning);
            }
        }

        if self.is_valid && self.warnings.is_empty() {
            println!("✅ Configuration is valid");
        } else if self.is_valid {
            println!("✅ Configuration is valid (with warnings)");
        } else {
            println!("❌ Configuration is invalid");
        }
    }
}

/// Load and validate configuration from file
pub fn load_and_validate_config<P: AsRef<Path>>(path: P) -> Result<ValidatedConfig> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;

    let config: ValidatedConfig = match path.as_ref().extension().and_then(|s| s.to_str()) {
        Some("toml") => toml::from_str(&content)
            .with_context(|| "Failed to parse TOML configuration")?,
        Some("json") => serde_json::from_str(&content)
            .with_context(|| "Failed to parse JSON configuration")?,
        _ => return Err(anyhow::anyhow!("Unsupported config file format, use .toml or .json")),
    };

    let mut validator = ConfigValidator::new();
    let result = validator.validate(&config);

    if !result.is_valid {
        result.print_results();
        return Err(anyhow::anyhow!("Configuration validation failed"));
    }

    if !result.warnings.is_empty() {
        result.print_results();
    }

    Ok(config)
}

/// Generate example configuration file
pub fn generate_example_config(format: &str) -> Result<String> {
    let config = ValidatedConfig::default();
    
    match format {
        "toml" => toml::to_string_pretty(&config)
            .with_context(|| "Failed to serialize config to TOML"),
        "json" => serde_json::to_string_pretty(&config)
            .with_context(|| "Failed to serialize config to JSON"),
        _ => Err(anyhow::anyhow!("Unsupported format, use 'toml' or 'json'")),
    }
}