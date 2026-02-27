//! Configuration management for BitNet CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::debug;

/// CLI configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default model path
    pub default_model: Option<PathBuf>,
    /// Default device (cpu, cuda, auto)
    pub default_device: String,
    /// Default quantization type
    pub default_quantization: Option<String>,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Model cache directory
    pub model_cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log format (pretty, json, compact)
    pub format: String,
    /// Enable timestamps
    pub timestamps: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for CPU inference
    pub cpu_threads: Option<usize>,
    /// Batch size for inference
    pub batch_size: usize,
    /// Enable memory optimization
    pub memory_optimization: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_device: "auto".to_string(),
            default_quantization: None,
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
            model_cache_dir: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self { level: "info".to_string(), format: "pretty".to_string(), timestamps: true }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self { cpu_threads: None, batch_size: 1, memory_optimization: true }
    }
}

impl CliConfig {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading configuration from: {}", path.display());

        if !path.exists() {
            debug!("Configuration file not found, using defaults");
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        debug!("Configuration loaded successfully");
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        debug!("Saving configuration to: {}", path.display());

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize configuration")?;

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        debug!("Configuration saved successfully");
        Ok(())
    }

    /// Get default configuration file path
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("Failed to get user config directory")?;
        Ok(config_dir.join("bitnet").join("config.toml"))
    }

    /// Merge with environment variables and command line overrides
    pub fn merge_with_env(&mut self) {
        if let Ok(device) = std::env::var("BITNET_DEVICE") {
            self.default_device = device;
        }

        if let Ok(level) = std::env::var("BITNET_LOG_LEVEL") {
            self.logging.level = level;
        }

        if let Ok(threads) = std::env::var("BITNET_CPU_THREADS")
            && let Ok(threads) = threads.parse()
        {
            self.performance.cpu_threads = Some(threads);
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate device
        match self.default_device.as_str() {
            "cpu" | "cuda" | "gpu" | "vulkan" | "opencl" | "ocl" | "auto" => {}
            _ => anyhow::bail!(
                "Invalid device: {}. Must be one of: cpu, cuda, gpu, vulkan, opencl, ocl, auto",
                self.default_device
            ),
        }

        // Validate log level
        match self.logging.level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => anyhow::bail!(
                "Invalid log level: {}. Must be one of: trace, debug, info, warn, error",
                self.logging.level
            ),
        }

        // Validate log format
        match self.logging.format.as_str() {
            "pretty" | "json" | "compact" => {}
            _ => anyhow::bail!(
                "Invalid log format: {}. Must be one of: pretty, json, compact",
                self.logging.format
            ),
        }

        // Validate batch size
        if self.performance.batch_size == 0 {
            anyhow::bail!("Batch size must be greater than 0");
        }

        Ok(())
    }
}

/// Configuration builder for command-line usage
#[derive(Default)]
pub struct ConfigBuilder {
    config: CliConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self { config: CliConfig::load_from_file(path)? })
    }

    pub fn device(mut self, device: Option<String>) -> Self {
        if let Some(device) = device {
            self.config.default_device = device;
        }
        self
    }

    pub fn log_level(mut self, level: Option<String>) -> Self {
        if let Some(level) = level {
            self.config.logging.level = level;
        }
        self
    }

    pub fn cpu_threads(mut self, threads: Option<usize>) -> Self {
        if let Some(threads) = threads {
            self.config.performance.cpu_threads = Some(threads);
        }
        self
    }

    pub fn batch_size(mut self, batch_size: Option<usize>) -> Self {
        if let Some(batch_size) = batch_size {
            self.config.performance.batch_size = batch_size;
        }
        self
    }

    pub fn build(mut self) -> Result<CliConfig> {
        self.config.merge_with_env();
        self.config.validate()?;
        Ok(self.config)
    }
}
