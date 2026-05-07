//! Core configuration contracts and validation for BitNet CLI.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Package-level device/backend labels accepted by CLI configuration.
///
/// These labels name proof lanes and requested backend identities. They do not
/// imply that every subcommand can execute every backend on every host.
pub const SUPPORTED_DEVICE_LABELS: &[&str] = &[
    "cpu",
    "cuda",
    "gpu",
    "vulkan",
    "opencl",
    "ocl",
    "hip",
    "rocm",
    "oneapi",
    "npu",
    "npu:<index>",
    "intel-npu",
    "intel-npu:<index>",
    "openvino-npu",
    "intel-npu-openvino",
    "nvidia-rtx-5070-ti-cuda",
    "nvidia-rtx-5070-ti-wgpu",
    "metal",
    "mpsgraph",
    "apple-m4-metal",
    "apple-m4-mpsgraph",
    "apple-m4-cpu-neon",
    "auto",
];

/// Stable help text for supported package-level device/backend labels.
pub const SUPPORTED_DEVICE_LABELS_TEXT: &str = "cpu, cuda, gpu, vulkan, opencl, ocl, hip, rocm, oneapi, npu, npu:<index>, intel-npu, intel-npu:<index>, openvino-npu, intel-npu-openvino, nvidia-rtx-5070-ti-cuda, nvidia-rtx-5070-ti-wgpu, metal, mpsgraph, apple-m4-metal, apple-m4-mpsgraph, apple-m4-cpu-neon, auto";

/// Stable help text for Apple M4 proof-lane labels.
pub const APPLE_M4_DEVICE_LABELS_TEXT: &str = "apple-m4-metal = native Metal proof lane, apple-m4-mpsgraph = MPSGraph graph/reference lane, apple-m4-cpu-neon = Apple CPU/NEON fallback/parity lane";

/// Top-level `--device` help for package-level backend labels.
pub const DEVICE_HELP: &str = "Device/backend label (cpu, cuda/gpu, hip/rocm, oneapi, npu/openvino-npu, nvidia-rtx-5070-ti-cuda/wgpu, metal/mpsgraph, apple-m4-metal, apple-m4-mpsgraph, apple-m4-cpu-neon, auto). Apple M4 labels are distinct proof lanes";

/// Help for legacy full-cli commands that do not emit Apple proof receipts.
pub const LEGACY_RUNTIME_DEVICE_HELP: &str = "Device for this legacy command (cpu, cuda/gpu aliases, auto). Use `bitnet run` for receipt-backed Apple M4 labels";

/// Runtime labels currently handled by legacy full-cli commands.
pub const LEGACY_RUNTIME_DEVICE_LABELS_TEXT: &str = "cpu, cuda, gpu, vulkan, opencl, ocl, auto";

/// Build a consistent invalid package-level device label error.
pub fn invalid_device_message(device: &str) -> String {
    format!(
        "Invalid device: {device}. Must be one of: {SUPPORTED_DEVICE_LABELS_TEXT}. Apple M4 labels are distinct proof lanes: {APPLE_M4_DEVICE_LABELS_TEXT}. On unavailable or non-M4 hosts, strict mode fails and non-strict receipt paths must record fallback_used and fallback_reason."
    )
}

/// Build a consistent error for legacy commands that do not support a proof lane.
pub fn unsupported_legacy_command_device_message(command: &str, device: &str) -> String {
    format!(
        "{command} does not support device label '{device}'. This legacy command currently supports: {LEGACY_RUNTIME_DEVICE_LABELS_TEXT}. Use `bitnet run` for receipt-backed Apple M4 labels ({APPLE_M4_DEVICE_LABELS_TEXT}); CPU fallback cannot count as Metal execution."
    )
}

/// CLI configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default model path
    pub default_model: Option<PathBuf>,
    /// Default device/backend identity (cpu, cuda, auto, apple-m4-metal, etc.)
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
        } else if let Ok(backend) = std::env::var("BITNET_BACKEND") {
            self.default_device = backend;
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
        if !is_supported_device_label(&self.default_device) {
            anyhow::bail!("{}", invalid_device_message(&self.default_device));
        }

        match self.logging.level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => anyhow::bail!(
                "Invalid log level: {}. Must be one of: trace, debug, info, warn, error",
                self.logging.level
            ),
        }

        match self.logging.format.as_str() {
            "pretty" | "json" | "compact" => {}
            _ => anyhow::bail!(
                "Invalid log format: {}. Must be one of: pretty, json, compact",
                self.logging.format
            ),
        }

        if self.performance.batch_size == 0 {
            anyhow::bail!("Batch size must be greater than 0");
        }

        Ok(())
    }
}

pub fn is_supported_device_label(label: &str) -> bool {
    matches!(
        label,
        "cpu"
            | "cuda"
            | "gpu"
            | "vulkan"
            | "opencl"
            | "ocl"
            | "hip"
            | "rocm"
            | "oneapi"
            | "npu"
            | "intel-npu"
            | "openvino-npu"
            | "intel-npu-openvino"
            | "nvidia-rtx-5070-ti-cuda"
            | "nvidia-rtx-5070-ti-wgpu"
            | "metal"
            | "mpsgraph"
            | "apple-m4-metal"
            | "apple-m4-mpsgraph"
            | "apple-m4-cpu-neon"
            | "auto"
    ) || label.strip_prefix("npu:").is_some_and(|index| index.parse::<usize>().is_ok())
        || label.strip_prefix("intel-npu:").is_some_and(|index| index.parse::<usize>().is_ok())
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

#[cfg(test)]
mod tests {
    use super::{
        APPLE_M4_DEVICE_LABELS_TEXT, CliConfig, ConfigBuilder, SUPPORTED_DEVICE_LABELS,
        invalid_device_message,
    };

    #[test]
    fn supported_device_labels_constant_matches_validation() {
        for device in SUPPORTED_DEVICE_LABELS {
            if device.contains("<index>") {
                continue;
            }
            let config =
                CliConfig { default_device: (*device).to_string(), ..CliConfig::default() };
            config.validate().unwrap_or_else(|err| panic!("{device} should validate: {err}"));
        }
    }

    #[test]
    fn validates_intel_npu_labels_without_aliasing() {
        for device in ["npu", "intel-npu", "intel-npu:1", "openvino-npu", "intel-npu-openvino"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            config.validate().unwrap();
        }
    }

    #[test]
    fn rejects_invalid_intel_npu_index() {
        for device in ["npu:", "npu:abc", "intel-npu:", "intel-npu:abc"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            assert!(config.validate().is_err(), "{device} should be rejected");
        }
    }

    #[test]
    fn builder_preserves_intel_npu_device_label() {
        let config = ConfigBuilder::new().device(Some("intel-npu:2".to_string())).build().unwrap();
        assert_eq!(config.default_device, "intel-npu:2");
    }

    #[test]
    fn validates_apple_m4_labels_without_aliasing() {
        for device in ["apple-m4-metal", "apple-m4-mpsgraph", "apple-m4-cpu-neon"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            config.validate().unwrap();
        }
    }

    #[test]
    fn invalid_device_message_describes_apple_m4_boundaries() {
        let message = invalid_device_message("quantum");
        assert!(message.contains("npu:<index>"), "got: {message}");
        assert!(message.contains("intel-npu-openvino"), "got: {message}");
        assert!(message.contains(APPLE_M4_DEVICE_LABELS_TEXT), "got: {message}");
        assert!(message.contains("strict mode fails"), "got: {message}");
        assert!(message.contains("fallback_used"), "got: {message}");
    }
}
