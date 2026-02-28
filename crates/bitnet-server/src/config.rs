//! Configuration management with environment variable support

use anyhow::Result;
use bitnet_common::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::net::IpAddr;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use crate::batch_engine::BatchEngineConfig;
use crate::concurrency::ConcurrencyConfig;
use crate::execution_router::{DeviceSelectionStrategy, ExecutionRouterConfig};
use crate::model_manager::ModelManagerConfig;
use crate::monitoring::MonitoringConfig;
use crate::security::SecurityConfig;

/// Device configuration mode for server initialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DeviceConfig {
    /// Automatically select the best available device (prefer GPU if available)
    #[default]
    Auto,
    /// Force CPU execution
    Cpu,
    /// Force GPU execution on specific device ID
    Gpu(usize),
    Vulkan(usize),
}

impl FromStr for DeviceConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(DeviceConfig::Auto),
            "cpu" => Ok(DeviceConfig::Cpu),
            "gpu" | "cuda" | "opencl" | "ocl" => Ok(DeviceConfig::Gpu(0)),
            "vulkan" => Ok(DeviceConfig::Vulkan(0)),
            s if s.starts_with("gpu:") => {
                let id_str = &s[4..];
                let id = id_str.parse::<usize>()?;
                Ok(DeviceConfig::Gpu(id))
            }
            s if s.starts_with("cuda:") => {
                let id_str = &s[5..];
                let id = id_str.parse::<usize>()?;
                Ok(DeviceConfig::Gpu(id))
            }
            s if s.starts_with("vulkan:") => {
                let id_str = &s[7..];
                let id = id_str.parse::<usize>()?;
                Ok(DeviceConfig::Vulkan(id))
            }
            s if s.starts_with("opencl:") => {
                let id_str = &s[7..];
                let id = id_str.parse::<usize>()?;
                Ok(DeviceConfig::Gpu(id))
            }
            s if s.starts_with("ocl:") => {
                let id_str = &s[4..];
                let id = id_str.parse::<usize>()?;
                Ok(DeviceConfig::Gpu(id))
            }
            _ => anyhow::bail!("Unknown device config: {}", s),
        }
    }
}

impl DeviceConfig {
    /// Resolve device configuration to actual device
    pub fn resolve(&self) -> Device {
        match self {
            DeviceConfig::Auto => {
                // Auto: prefer GPU if available at runtime
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use bitnet_kernels::device_features::gpu_available_runtime;
                    if gpu_available_runtime() { Device::Cuda(0) } else { Device::Cpu }
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    Device::Cpu
                }
            }
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Gpu(id) => Device::Cuda(*id),
            DeviceConfig::Vulkan(id) => Device::Vulkan(*id),
        }
    }
}

/// Complete server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    pub server: ServerSettings,
    pub model_manager: ModelManagerConfig,
    pub execution_router: ExecutionRouterConfig,
    pub batch_engine: BatchEngineConfig,
    pub concurrency: ConcurrencyConfig,
    pub security: SecurityConfig,
    pub monitoring: MonitoringConfig,
}

/// Basic server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub keep_alive: Duration,
    pub request_timeout: Duration,
    pub graceful_shutdown_timeout: Duration,
    pub default_model_path: Option<String>,
    pub default_tokenizer_path: Option<String>,
    pub default_device: DeviceConfig,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: None, // Use system default
            keep_alive: Duration::from_secs(60),
            request_timeout: Duration::from_secs(300), // 5 minutes
            graceful_shutdown_timeout: Duration::from_secs(30),
            default_model_path: None,
            default_tokenizer_path: None,
            default_device: DeviceConfig::Auto,
        }
    }
}

/// Configuration builder with environment variable support
pub struct ConfigBuilder {
    config: ServerConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self { config: ServerConfig::default() }
    }

    /// Load configuration from environment variables
    pub fn from_env(mut self) -> Result<Self> {
        // Server settings
        if let Ok(host) = env::var("BITNET_SERVER_HOST") {
            self.config.server.host = host;
        }

        if let Ok(port) = env::var("BITNET_SERVER_PORT") {
            self.config.server.port = port.parse()?;
        }

        if let Ok(workers) = env::var("BITNET_SERVER_WORKERS") {
            self.config.server.workers = Some(workers.parse()?);
        }

        if let Ok(timeout) = env::var("BITNET_REQUEST_TIMEOUT") {
            self.config.server.request_timeout = Duration::from_secs(timeout.parse()?);
        }

        if let Ok(model_path) = env::var("BITNET_DEFAULT_MODEL_PATH") {
            self.config.server.default_model_path = Some(model_path);
        }

        if let Ok(tokenizer_path) = env::var("BITNET_DEFAULT_TOKENIZER_PATH") {
            self.config.server.default_tokenizer_path = Some(tokenizer_path);
        }

        if let Ok(device) = env::var("BITNET_DEFAULT_DEVICE") {
            match device.parse::<DeviceConfig>() {
                Ok(device_config) => {
                    self.config.server.default_device = device_config;
                }
                Err(e) => {
                    tracing::warn!("Invalid BITNET_DEFAULT_DEVICE value '{}': {}", device, e);
                }
            }
        }

        // Model manager settings
        if let Ok(max_loads) = env::var("BITNET_MAX_CONCURRENT_LOADS") {
            self.config.model_manager.max_concurrent_loads = max_loads.parse()?;
        }

        if let Ok(cache_size) = env::var("BITNET_MODEL_CACHE_SIZE") {
            self.config.model_manager.model_cache_size = cache_size.parse()?;
        }

        if let Ok(memory_limit) = env::var("BITNET_MEMORY_LIMIT_GB") {
            self.config.model_manager.memory_limit_gb = Some(memory_limit.parse()?);
        }

        if let Ok(validation) = env::var("BITNET_MODEL_VALIDATION") {
            self.config.model_manager.validation_enabled = validation.parse()?;
        }

        // Execution router settings
        if let Ok(strategy) = env::var("BITNET_DEVICE_STRATEGY") {
            self.config.execution_router.strategy = match strategy.to_lowercase().as_str() {
                "prefer_gpu" => DeviceSelectionStrategy::PreferGpu,
                "cpu_only" => DeviceSelectionStrategy::CpuOnly,
                "performance" => DeviceSelectionStrategy::PerformanceBased,
                "load_balance" => DeviceSelectionStrategy::LoadBalance,
                _ => DeviceSelectionStrategy::PerformanceBased,
            };
        }

        if let Ok(fallback) = env::var("BITNET_FALLBACK_ENABLED") {
            self.config.execution_router.fallback_enabled = fallback.parse()?;
        }

        if let Ok(benchmark) = env::var("BITNET_BENCHMARK_ON_STARTUP") {
            self.config.execution_router.benchmark_on_startup = benchmark.parse()?;
        }

        // Batch engine settings
        if let Ok(batch_size) = env::var("BITNET_MAX_BATCH_SIZE") {
            self.config.batch_engine.max_batch_size = batch_size.parse()?;
        }

        if let Ok(timeout) = env::var("BITNET_BATCH_TIMEOUT_MS") {
            self.config.batch_engine.batch_timeout = Duration::from_millis(timeout.parse()?);
        }

        if let Ok(concurrent) = env::var("BITNET_MAX_CONCURRENT_BATCHES") {
            self.config.batch_engine.max_concurrent_batches = concurrent.parse()?;
        }

        if let Ok(adaptive) = env::var("BITNET_ADAPTIVE_BATCHING") {
            self.config.batch_engine.adaptive_batching = adaptive.parse()?;
        }

        if let Ok(quantization) = env::var("BITNET_QUANTIZATION_AWARE") {
            self.config.batch_engine.quantization_aware = quantization.parse()?;
        }

        // Concurrency settings
        if let Ok(max_concurrent) = env::var("BITNET_MAX_CONCURRENT_REQUESTS") {
            self.config.concurrency.max_concurrent_requests = max_concurrent.parse()?;
        }

        if let Ok(rps) = env::var("BITNET_MAX_REQUESTS_PER_SECOND") {
            self.config.concurrency.max_requests_per_second = rps.parse()?;
        }

        if let Ok(rpm) = env::var("BITNET_MAX_REQUESTS_PER_MINUTE") {
            self.config.concurrency.max_requests_per_minute = rpm.parse()?;
        }

        if let Ok(threshold) = env::var("BITNET_BACKPRESSURE_THRESHOLD") {
            self.config.concurrency.backpressure_threshold = threshold.parse()?;
        }

        if let Ok(circuit_breaker) = env::var("BITNET_CIRCUIT_BREAKER_ENABLED") {
            self.config.concurrency.circuit_breaker_enabled = circuit_breaker.parse()?;
        }

        if let Ok(per_ip_limit) = env::var("BITNET_PER_IP_RATE_LIMIT") {
            self.config.concurrency.per_ip_rate_limit = Some(per_ip_limit.parse()?);
        }

        // Security settings
        if let Ok(jwt_secret) = env::var("BITNET_JWT_SECRET") {
            self.config.security.jwt_secret = Some(jwt_secret);
        }

        if let Ok(require_auth) = env::var("BITNET_REQUIRE_AUTHENTICATION") {
            self.config.security.require_authentication = require_auth.parse()?;
        }

        if let Ok(max_prompt) = env::var("BITNET_MAX_PROMPT_LENGTH") {
            self.config.security.max_prompt_length = max_prompt.parse()?;
        }

        if let Ok(max_tokens) = env::var("BITNET_MAX_TOKENS_PER_REQUEST") {
            self.config.security.max_tokens_per_request = max_tokens.parse()?;
        }

        if let Ok(origins) = env::var("BITNET_ALLOWED_ORIGINS") {
            self.config.security.allowed_origins =
                origins.split(',').map(|s| s.trim().to_string()).collect();
        }

        if let Ok(dirs) = env::var("BITNET_ALLOWED_MODEL_DIRECTORIES") {
            self.config.security.allowed_model_directories =
                dirs.split(',').map(|s| s.trim().to_string()).collect();
        }

        if let Ok(blocked_ips) = env::var("BITNET_BLOCKED_IPS") {
            let mut ips = HashSet::new();
            for ip_str in blocked_ips.split(',') {
                if let Ok(ip) = ip_str.trim().parse::<IpAddr>() {
                    ips.insert(ip);
                }
            }
            self.config.security.blocked_ips = ips;
        }

        if let Ok(sanitization) = env::var("BITNET_INPUT_SANITIZATION") {
            self.config.security.input_sanitization = sanitization.parse()?;
        }

        if let Ok(filtering) = env::var("BITNET_CONTENT_FILTERING") {
            self.config.security.content_filtering = filtering.parse()?;
        }

        // Monitoring settings
        if let Ok(prometheus) = env::var("BITNET_PROMETHEUS_ENABLED") {
            self.config.monitoring.prometheus_enabled = prometheus.parse()?;
        }

        if let Ok(opentelemetry) = env::var("BITNET_OPENTELEMETRY_ENABLED") {
            self.config.monitoring.opentelemetry_enabled = opentelemetry.parse()?;
        }

        if let Ok(endpoint) = env::var("BITNET_OTLP_ENDPOINT") {
            self.config.monitoring.otlp_endpoint = Some(endpoint);
        }

        if let Ok(level) = env::var("BITNET_LOG_LEVEL") {
            self.config.monitoring.log_level = level;
        }

        Ok(self)
    }

    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let file_config: ServerConfig = toml::from_str(&content)?;

        // Merge file config with current config (file takes precedence)
        self.config = file_config;

        Ok(self)
    }

    /// Override server settings
    pub fn with_server_settings(mut self, settings: ServerSettings) -> Self {
        self.config.server = settings;
        self
    }

    /// Override model manager config
    pub fn with_model_manager(mut self, config: ModelManagerConfig) -> Self {
        self.config.model_manager = config;
        self
    }

    /// Override execution router config
    pub fn with_execution_router(mut self, config: ExecutionRouterConfig) -> Self {
        self.config.execution_router = config;
        self
    }

    /// Override batch engine config
    pub fn with_batch_engine(mut self, config: BatchEngineConfig) -> Self {
        self.config.batch_engine = config;
        self
    }

    /// Override concurrency config
    pub fn with_concurrency(mut self, config: ConcurrencyConfig) -> Self {
        self.config.concurrency = config;
        self
    }

    /// Override security config
    pub fn with_security(mut self, config: SecurityConfig) -> Self {
        self.config.security = config;
        self
    }

    /// Override monitoring config
    pub fn with_monitoring(mut self, config: MonitoringConfig) -> Self {
        self.config.monitoring = config;
        self
    }

    /// Validate configuration
    pub fn validate(self) -> Result<Self> {
        let config = &self.config;

        // Validate server settings
        if config.server.port == 0 {
            anyhow::bail!("Server port cannot be 0");
        }

        if config.server.host.is_empty() {
            anyhow::bail!("Server host cannot be empty");
        }

        // Validate model manager settings
        if config.model_manager.max_concurrent_loads == 0 {
            anyhow::bail!("Max concurrent loads must be at least 1");
        }

        if config.model_manager.model_cache_size == 0 {
            anyhow::bail!("Model cache size must be at least 1");
        }

        // Validate batch engine settings
        if config.batch_engine.max_batch_size == 0 {
            anyhow::bail!("Max batch size must be at least 1");
        }

        if config.batch_engine.max_concurrent_batches == 0 {
            anyhow::bail!("Max concurrent batches must be at least 1");
        }

        // Validate concurrency settings
        if config.concurrency.max_concurrent_requests == 0 {
            anyhow::bail!("Max concurrent requests must be at least 1");
        }

        if config.concurrency.backpressure_threshold < 0.0
            || config.concurrency.backpressure_threshold > 1.0
        {
            anyhow::bail!("Backpressure threshold must be between 0.0 and 1.0");
        }

        // Validate security settings
        if config.security.max_prompt_length == 0 {
            anyhow::bail!("Max prompt length must be at least 1");
        }

        if config.security.max_tokens_per_request == 0 {
            anyhow::bail!("Max tokens per request must be at least 1");
        }

        if config.security.require_authentication && config.security.jwt_secret.is_none() {
            anyhow::bail!("JWT secret is required when authentication is enabled");
        }

        // Check if default model file exists (if specified)
        if let Some(model_path) = &config.server.default_model_path
            && !Path::new(model_path).exists()
        {
            anyhow::bail!("Default model file not found: {}", model_path);
        }

        Ok(self)
    }

    /// Build the final configuration
    pub fn build(self) -> ServerConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Load configuration from multiple sources with precedence:
/// 1. Environment variables (highest precedence)
/// 2. Configuration file
/// 3. Default values (lowest precedence)
pub fn load_config() -> Result<ServerConfig> {
    let mut builder = ConfigBuilder::new();

    // Load from config file if BITNET_CONFIG_FILE is set
    if let Ok(config_file) = env::var("BITNET_CONFIG_FILE") {
        if Path::new(&config_file).exists() {
            builder = builder.from_file(config_file)?;
        }
    } else {
        // Try default config files
        for default_path in
            &["bitnet-server.toml", "config/bitnet-server.toml", "/etc/bitnet/server.toml"]
        {
            if Path::new(default_path).exists() {
                builder = builder.from_file(default_path)?;
                break;
            }
        }
    }

    // Override with environment variables
    builder = builder.from_env()?;

    // Validate and build
    let config = builder.validate()?.build();

    Ok(config)
}

/// Generate example configuration file
pub fn generate_example_config() -> String {
    let config = ServerConfig::default();
    toml::to_string_pretty(&config).unwrap_or_else(|_| "# Failed to generate config".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert!(config.model_manager.max_concurrent_loads > 0);
        assert_eq!(config.server.default_device, DeviceConfig::Auto);
    }

    #[test]
    fn test_device_config_from_str() {
        assert_eq!("auto".parse::<DeviceConfig>().unwrap(), DeviceConfig::Auto);
        assert_eq!("cpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Cpu);
        assert_eq!("gpu".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
        assert_eq!("vulkan".parse::<DeviceConfig>().unwrap(), DeviceConfig::Vulkan(0));
        assert_eq!("opencl".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(0));
        assert_eq!("gpu:1".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(1));
        assert_eq!("cuda:2".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(2));
        assert_eq!("vulkan:3".parse::<DeviceConfig>().unwrap(), DeviceConfig::Vulkan(3));
        assert_eq!("opencl:4".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(4));
        assert_eq!("ocl:5".parse::<DeviceConfig>().unwrap(), DeviceConfig::Gpu(5));
        assert!("invalid".parse::<DeviceConfig>().is_err());
    }

    #[test]
    fn test_device_config_resolve_cpu() {
        let config = DeviceConfig::Cpu;
        assert_eq!(config.resolve(), Device::Cpu);
    }

    #[test]
    fn test_device_config_resolve_gpu() {
        let config = DeviceConfig::Gpu(1);
        assert_eq!(config.resolve(), Device::Cuda(1));
    }

    #[test]
    fn test_device_config_resolve_auto() {
        let config = DeviceConfig::Auto;
        let device = config.resolve();
        // Auto resolves to CPU or GPU depending on feature flags and runtime detection
        // In CPU-only builds, it should be CPU
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        assert_eq!(device, Device::Cpu);
        // In GPU builds, it depends on runtime GPU availability
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            // Device can be either CPU or CUDA(0) depending on GPU availability
            assert!(device == Device::Cpu || device == Device::Cuda(0));
        }
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new().validate().unwrap().build();

        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_env_override() {
        unsafe {
            env::set_var("BITNET_SERVER_PORT", "9090");
            env::set_var("BITNET_MAX_BATCH_SIZE", "32");
        }

        let config = ConfigBuilder::new().from_env().unwrap().validate().unwrap().build();

        assert_eq!(config.server.port, 9090);
        assert_eq!(config.batch_engine.max_batch_size, 32);

        // Clean up
        unsafe {
            env::remove_var("BITNET_SERVER_PORT");
            env::remove_var("BITNET_MAX_BATCH_SIZE");
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = ServerConfig::default();
        config.server.port = 0; // Invalid

        let builder = ConfigBuilder::new().with_server_settings(config.server);
        assert!(builder.validate().is_err());
    }
}
