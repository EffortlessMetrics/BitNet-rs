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
    "intel-a770-opencl",
    "metal",
    "mpsgraph",
    "apple-m4-metal",
    "apple-m4-mpsgraph",
    "apple-m4-cpu-neon",
    "apple-m3-air-metal",
    "apple-m3-air-mpsgraph",
    "apple-m3-air-cpu-neon",
    "auto",
];

/// Stable help text for supported package-level device/backend labels.
pub const SUPPORTED_DEVICE_LABELS_TEXT: &str = "cpu, cuda, gpu, vulkan, opencl, ocl, hip, rocm, oneapi, npu, npu:<index>, intel-npu, intel-npu:<index>, openvino-npu, intel-npu-openvino, nvidia-rtx-5070-ti-cuda, nvidia-rtx-5070-ti-wgpu, intel-a770-opencl, metal, mpsgraph, apple-m4-metal, apple-m4-mpsgraph, apple-m4-cpu-neon, apple-m3-air-metal, apple-m3-air-mpsgraph, apple-m3-air-cpu-neon, auto";

/// Stable help text for Apple M4 proof-lane labels.
pub const APPLE_M4_DEVICE_LABELS_TEXT: &str = "apple-m4-metal = native Metal proof lane, apple-m4-mpsgraph = MPSGraph graph/reference lane, apple-m4-cpu-neon = Apple CPU/NEON fallback/parity lane";

/// Stable help text for Apple M3 MacBook Air proof-lane labels.
pub const APPLE_M3_AIR_DEVICE_LABELS_TEXT: &str = "apple-m3-air-metal = strict request identity for future M3 MacBook Air Metal receipts, apple-m3-air-mpsgraph = strict request identity for future M3 MacBook Air MPSGraph/reference receipts, apple-m3-air-cpu-neon = M3 MacBook Air Apple CPU/NEON lane";

/// Top-level `--device` help for package-level backend labels.
pub const DEVICE_HELP: &str = "Device/backend label (cpu, cuda/gpu, hip/rocm, oneapi, npu/openvino-npu, nvidia-rtx-5070-ti-cuda/wgpu, intel-a770-opencl, metal/mpsgraph, apple-m4-metal, apple-m4-mpsgraph, apple-m4-cpu-neon, apple-m3-air-metal, apple-m3-air-mpsgraph, apple-m3-air-cpu-neon, auto). Apple M4 and M3 Air labels are distinct proof lanes";

/// Help for legacy full-cli commands that do not emit Apple proof receipts.
pub const LEGACY_RUNTIME_DEVICE_HELP: &str = "Device for this legacy command (cpu, cuda/gpu aliases, auto). Use `bitnet run` for receipt-backed Apple proof labels";

/// Runtime labels currently handled by legacy full-cli commands.
pub const LEGACY_RUNTIME_DEVICE_LABELS_TEXT: &str = "cpu, cuda, gpu, vulkan, opencl, ocl, auto";

/// Build a consistent invalid package-level device label error.
pub fn invalid_device_message(device: &str) -> String {
    format!(
        "Invalid device: {device}. Must be one of: {SUPPORTED_DEVICE_LABELS_TEXT}. Apple M4 labels are distinct proof lanes: {APPLE_M4_DEVICE_LABELS_TEXT}. Apple M3 Air labels are distinct proof lanes: {APPLE_M3_AIR_DEVICE_LABELS_TEXT}. On unavailable or non-matching Apple hosts, strict mode fails and non-strict receipt paths must record fallback_used and fallback_reason."
    )
}

/// Build a consistent error for legacy commands that do not support a proof lane.
pub fn unsupported_legacy_command_device_message(command: &str, device: &str) -> String {
    format!(
        "{command} does not support device label '{device}'. This legacy command currently supports: {LEGACY_RUNTIME_DEVICE_LABELS_TEXT}. Use `bitnet run` for receipt-backed Apple proof labels ({APPLE_M4_DEVICE_LABELS_TEXT}; {APPLE_M3_AIR_DEVICE_LABELS_TEXT}); CPU fallback cannot count as Metal execution."
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
            | "intel-a770-opencl"
            | "metal"
            | "mpsgraph"
            | "apple-m4-metal"
            | "apple-m4-mpsgraph"
            | "apple-m4-cpu-neon"
            | "apple-m3-air-metal"
            | "apple-m3-air-mpsgraph"
            | "apple-m3-air-cpu-neon"
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
        APPLE_M3_AIR_DEVICE_LABELS_TEXT, APPLE_M4_DEVICE_LABELS_TEXT, CliConfig, ConfigBuilder,
        DEVICE_HELP, LoggingConfig, PerformanceConfig, SUPPORTED_DEVICE_LABELS,
        SUPPORTED_DEVICE_LABELS_TEXT, invalid_device_message, is_supported_device_label,
        unsupported_legacy_command_device_message,
    };
    use std::path::PathBuf;

    #[test]
    fn default_config_uses_stable_safe_values() -> anyhow::Result<()> {
        let config = CliConfig::default();

        assert_eq!(config.default_model, None);
        assert_eq!(config.default_device, "auto");
        assert_eq!(config.default_quantization, None);
        assert_eq!(config.model_cache_dir, None);
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.format, "pretty");
        assert!(config.logging.timestamps);
        assert_eq!(config.performance.cpu_threads, None);
        assert_eq!(config.performance.batch_size, 1);
        assert!(config.performance.memory_optimization);
        config.validate()?;
        Ok(())
    }

    #[test]
    fn load_from_missing_file_returns_defaults() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let missing_path = temp_dir.path().join("missing").join("config.toml");

        let config = CliConfig::load_from_file(&missing_path)?;

        assert_eq!(config.default_device, CliConfig::default().default_device);
        assert_eq!(config.logging.level, CliConfig::default().logging.level);
        assert_eq!(config.performance.batch_size, CliConfig::default().performance.batch_size);
        Ok(())
    }

    #[test]
    fn save_to_file_creates_parent_directories_and_roundtrips() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let config_path = temp_dir.path().join("nested").join("bitnet").join("config.toml");
        let config = CliConfig {
            default_model: Some(PathBuf::from("models/test.gguf")),
            default_device: "cpu".to_string(),
            default_quantization: Some("i2_s".to_string()),
            logging: LoggingConfig {
                level: "debug".to_string(),
                format: "json".to_string(),
                timestamps: false,
            },
            performance: PerformanceConfig {
                cpu_threads: Some(8),
                batch_size: 16,
                memory_optimization: false,
            },
            model_cache_dir: Some(PathBuf::from("cache/models")),
        };

        config.save_to_file(&config_path)?;
        let loaded = CliConfig::load_from_file(&config_path)?;

        assert_eq!(loaded.default_model, config.default_model);
        assert_eq!(loaded.default_device, config.default_device);
        assert_eq!(loaded.default_quantization, config.default_quantization);
        assert_eq!(loaded.logging.level, config.logging.level);
        assert_eq!(loaded.logging.format, config.logging.format);
        assert_eq!(loaded.logging.timestamps, config.logging.timestamps);
        assert_eq!(loaded.performance.cpu_threads, config.performance.cpu_threads);
        assert_eq!(loaded.performance.batch_size, config.performance.batch_size);
        assert_eq!(loaded.performance.memory_optimization, config.performance.memory_optimization);
        assert_eq!(loaded.model_cache_dir, config.model_cache_dir);
        Ok(())
    }

    #[test]
    fn load_from_invalid_toml_includes_path_context() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let config_path = temp_dir.path().join("config.toml");
        std::fs::write(&config_path, "default_device = [not valid toml")?;

        let err = match CliConfig::load_from_file(&config_path) {
            Ok(_) => anyhow::bail!("invalid TOML should fail to load"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("Failed to parse config file"), "got: {err}");
        assert!(err.contains(&config_path.display().to_string()), "got: {err}");
        Ok(())
    }

    #[test]
    fn validate_rejects_invalid_log_level_format_and_batch_size() -> anyhow::Result<()> {
        let invalid_level = CliConfig {
            logging: LoggingConfig { level: "verbose".to_string(), ..LoggingConfig::default() },
            ..CliConfig::default()
        };
        let invalid_level_err = match invalid_level.validate() {
            Ok(()) => anyhow::bail!("invalid log level should fail validation"),
            Err(err) => err.to_string(),
        };
        assert!(invalid_level_err.contains("Invalid log level"));

        let invalid_format = CliConfig {
            logging: LoggingConfig { format: "yaml".to_string(), ..LoggingConfig::default() },
            ..CliConfig::default()
        };
        let invalid_format_err = match invalid_format.validate() {
            Ok(()) => anyhow::bail!("invalid log format should fail validation"),
            Err(err) => err.to_string(),
        };
        assert!(invalid_format_err.contains("Invalid log format"));

        let invalid_batch = CliConfig {
            performance: PerformanceConfig { batch_size: 0, ..PerformanceConfig::default() },
            ..CliConfig::default()
        };
        let invalid_batch_err = match invalid_batch.validate() {
            Ok(()) => anyhow::bail!("invalid batch size should fail validation"),
            Err(err) => err.to_string(),
        };
        assert!(invalid_batch_err.contains("Batch size"));
        Ok(())
    }

    #[test]
    fn builder_applies_explicit_overrides_before_validation() -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::new()
                    .device(Some("cpu".to_string()))
                    .log_level(Some("warn".to_string()))
                    .cpu_threads(Some(4))
                    .batch_size(Some(32))
                    .build()
            },
        )?;

        assert_eq!(config.default_device, "cpu");
        assert_eq!(config.logging.level, "warn");
        assert_eq!(config.performance.cpu_threads, Some(4));
        assert_eq!(config.performance.batch_size, 32);
        Ok(())
    }

    #[test]
    fn builder_applies_environment_overrides_with_device_precedence() -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Some("cuda")),
                ("BITNET_BACKEND", Some("cpu")),
                ("BITNET_LOG_LEVEL", Some("error")),
                ("BITNET_CPU_THREADS", Some("12")),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::new()
                    .device(Some("auto".to_string()))
                    .log_level(Some("info".to_string()))
                    .cpu_threads(Some(2))
                    .build()
            },
        )?;

        assert_eq!(config.default_device, "cuda");
        assert_eq!(config.logging.level, "error");
        assert_eq!(config.performance.cpu_threads, Some(12));
        Ok(())
    }

    #[test]
    fn builder_uses_backend_when_device_env_is_absent_and_ignores_invalid_threads()
    -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Some("opencl")),
                ("BITNET_CPU_THREADS", Some("many")),
            ],
            || -> anyhow::Result<CliConfig> { ConfigBuilder::new().cpu_threads(Some(3)).build() },
        )?;

        assert_eq!(config.default_device, "opencl");
        assert_eq!(config.performance.cpu_threads, Some(3));
        Ok(())
    }

    #[test]
    fn builder_from_file_merges_file_values_with_overrides() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let config_path = temp_dir.path().join("config.toml");
        std::fs::write(
            &config_path,
            r#"
default_device = "cpu"

[logging]
level = "debug"
format = "compact"
timestamps = false

[performance]
cpu_threads = 6
batch_size = 4
memory_optimization = true
"#,
        )?;

        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::from_file(&config_path)?
                    .device(Some("metal".to_string()))
                    .batch_size(Some(9))
                    .build()
            },
        )?;

        assert_eq!(config.default_device, "metal");
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.logging.format, "compact");
        assert!(!config.logging.timestamps);
        assert_eq!(config.performance.cpu_threads, Some(6));
        assert_eq!(config.performance.batch_size, 9);
        assert!(config.performance.memory_optimization);
        Ok(())
    }

    #[test]
    fn unsupported_legacy_command_message_lists_legacy_labels_and_proof_lanes() {
        let message = unsupported_legacy_command_device_message("bench", "apple-m4-metal");

        assert!(message.contains("bench does not support device label 'apple-m4-metal'"));
        assert!(message.contains("cpu, cuda, gpu, vulkan, opencl, ocl, auto"));
        assert!(message.contains(APPLE_M4_DEVICE_LABELS_TEXT));
        assert!(message.contains(APPLE_M3_AIR_DEVICE_LABELS_TEXT));
        assert!(message.contains("CPU fallback cannot count as Metal execution"));
    }

    #[test]
    fn supported_device_labels_constant_matches_validation() -> anyhow::Result<()> {
        for device in SUPPORTED_DEVICE_LABELS {
            if device.contains("<index>") {
                continue;
            }
            let config =
                CliConfig { default_device: (*device).to_string(), ..CliConfig::default() };
            config.validate()?;
        }
        Ok(())
    }

    #[test]
    fn validates_intel_npu_labels_without_aliasing() -> anyhow::Result<()> {
        for device in ["npu", "intel-npu", "intel-npu:1", "openvino-npu", "intel-npu-openvino"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            config.validate()?;
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_intel_npu_index() {
        for device in ["npu:", "npu:abc", "intel-npu:", "intel-npu:abc"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            assert!(config.validate().is_err(), "{device} should be rejected");
        }
    }

    #[test]
    fn builder_preserves_intel_npu_device_label() -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::new().device(Some("intel-npu:2".to_string())).build()
            },
        )?;
        assert_eq!(config.default_device, "intel-npu:2");
        Ok(())
    }

    #[test]
    fn validates_apple_m4_labels_without_aliasing() -> anyhow::Result<()> {
        for device in ["apple-m4-metal", "apple-m4-mpsgraph", "apple-m4-cpu-neon"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            config.validate()?;
        }
        Ok(())
    }

    #[test]
    fn validates_apple_m3_air_label_without_aliasing() {
        for device in ["apple-m3-air-metal", "apple-m3-air-mpsgraph", "apple-m3-air-cpu-neon"] {
            let config = CliConfig { default_device: device.to_string(), ..CliConfig::default() };
            assert!(config.validate().is_ok(), "{device} should validate");
        }
    }

    #[test]
    fn invalid_device_message_describes_apple_m4_boundaries() {
        let message = invalid_device_message("quantum");
        assert!(message.contains("npu:<index>"), "got: {message}");
        assert!(message.contains("intel-npu-openvino"), "got: {message}");
        assert!(message.contains(APPLE_M4_DEVICE_LABELS_TEXT), "got: {message}");
        assert!(message.contains(APPLE_M3_AIR_DEVICE_LABELS_TEXT), "got: {message}");
        assert!(message.contains("strict mode fails"), "got: {message}");
        assert!(message.contains("fallback_used"), "got: {message}");
    }

    #[test]
    fn unsupported_legacy_command_device_message_names_command_and_device() {
        let msg = unsupported_legacy_command_device_message("infer", "apple-m4-metal");
        assert!(msg.starts_with("infer does not support"), "got: {msg}");
        assert!(msg.contains("apple-m4-metal"), "got: {msg}");
        assert!(msg.contains("bitnet run"), "got: {msg}");
        assert!(msg.contains(APPLE_M4_DEVICE_LABELS_TEXT), "got: {msg}");
        assert!(msg.contains(APPLE_M3_AIR_DEVICE_LABELS_TEXT), "got: {msg}");
    }

    #[test]
    fn is_supported_device_label_accepts_known_labels() {
        for label in ["cpu", "cuda", "auto", "apple-m4-metal", "npu:0", "npu:1234", "intel-npu:0"] {
            assert!(is_supported_device_label(label), "{label} should be supported");
        }
    }

    #[test]
    fn is_supported_device_label_rejects_unknown_or_malformed() {
        for label in ["", "unknown", "npu:", "npu:abc", "intel-npu:", "intel-npu:x", "CPU"] {
            assert!(!is_supported_device_label(label), "{label} should be rejected");
        }
    }

    #[test]
    fn defaults_validate_cleanly() -> anyhow::Result<()> {
        let config = CliConfig::default();
        assert_eq!(config.default_device, "auto");
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.logging.format, "pretty");
        assert!(config.logging.timestamps);
        assert_eq!(config.performance.batch_size, 1);
        assert!(config.performance.memory_optimization);
        assert!(config.performance.cpu_threads.is_none());
        config.validate()?;
        Ok(())
    }

    #[test]
    fn validate_rejects_bad_log_level() {
        let config = CliConfig {
            logging: LoggingConfig { level: "loud".to_string(), ..LoggingConfig::default() },
            ..CliConfig::default()
        };
        let err = config.validate().expect_err("loud is not a level");
        assert!(format!("{err}").contains("Invalid log level"));
    }

    #[test]
    fn validate_accepts_each_log_level() -> anyhow::Result<()> {
        for level in ["trace", "debug", "info", "warn", "error"] {
            let config = CliConfig {
                logging: LoggingConfig { level: level.to_string(), ..LoggingConfig::default() },
                ..CliConfig::default()
            };
            config.validate()?;
        }
        Ok(())
    }

    #[test]
    fn validate_rejects_bad_log_format() {
        let config = CliConfig {
            logging: LoggingConfig { format: "xml".to_string(), ..LoggingConfig::default() },
            ..CliConfig::default()
        };
        let err = config.validate().expect_err("xml is not a format");
        assert!(format!("{err}").contains("Invalid log format"));
    }

    #[test]
    fn validate_accepts_each_log_format() -> anyhow::Result<()> {
        for fmt in ["pretty", "json", "compact"] {
            let config = CliConfig {
                logging: LoggingConfig { format: fmt.to_string(), ..LoggingConfig::default() },
                ..CliConfig::default()
            };
            config.validate()?;
        }
        Ok(())
    }

    #[test]
    fn validate_rejects_zero_batch_size() {
        let config = CliConfig {
            performance: PerformanceConfig { batch_size: 0, ..PerformanceConfig::default() },
            ..CliConfig::default()
        };
        let err = config.validate().expect_err("batch_size 0 must fail");
        assert!(format!("{err}").contains("Batch size"));
    }

    #[test]
    fn load_from_file_returns_defaults_when_missing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("does-not-exist.toml");
        let config = CliConfig::load_from_file(&path)?;
        // Compare to default by checking a few representative fields.
        assert_eq!(config.default_device, CliConfig::default().default_device);
        assert_eq!(config.logging.level, CliConfig::default().logging.level);
        Ok(())
    }

    #[test]
    fn save_and_load_round_trip() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("nested").join("config.toml");
        let original = CliConfig {
            default_device: "cpu".to_string(),
            logging: LoggingConfig {
                level: "debug".to_string(),
                format: "json".to_string(),
                timestamps: false,
            },
            performance: PerformanceConfig {
                cpu_threads: Some(8),
                batch_size: 4,
                memory_optimization: false,
            },
            ..CliConfig::default()
        };
        original.save_to_file(&path)?;
        assert!(path.exists(), "save_to_file must create the file (and any parents)");

        let loaded = CliConfig::load_from_file(&path)?;
        assert_eq!(loaded.default_device, "cpu");
        assert_eq!(loaded.logging.level, "debug");
        assert_eq!(loaded.logging.format, "json");
        assert!(!loaded.logging.timestamps);
        assert_eq!(loaded.performance.cpu_threads, Some(8));
        assert_eq!(loaded.performance.batch_size, 4);
        assert!(!loaded.performance.memory_optimization);
        Ok(())
    }

    #[test]
    fn load_from_file_rejects_malformed_toml() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "this is = not = toml")?;
        let Err(err) = CliConfig::load_from_file(&path) else {
            anyhow::bail!("malformed TOML should fail");
        };
        let message = format!("{err:#}");
        assert!(message.contains("parse"), "expected parse error, got: {message}");
        Ok(())
    }

    #[test]
    fn default_config_path_ends_in_bitnet_config_toml() -> anyhow::Result<()> {
        let path = CliConfig::default_config_path()?;
        assert!(path.ends_with("bitnet/config.toml"), "got: {}", path.display());
        Ok(())
    }

    #[test]
    fn merge_with_env_reads_bitnet_device() {
        temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Some("cuda")),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || {
                let mut config = CliConfig::default();
                config.merge_with_env();
                assert_eq!(config.default_device, "cuda");
            },
        );
    }

    #[test]
    fn merge_with_env_prefers_device_over_backend() {
        temp_env::with_vars(
            vec![("BITNET_DEVICE", Some("cuda")), ("BITNET_BACKEND", Some("rocm"))],
            || {
                let mut config = CliConfig::default();
                config.merge_with_env();
                assert_eq!(config.default_device, "cuda");
            },
        );
    }

    #[test]
    fn merge_with_env_falls_back_to_backend() {
        temp_env::with_vars(
            vec![("BITNET_DEVICE", Option::<&str>::None), ("BITNET_BACKEND", Some("rocm"))],
            || {
                let mut config = CliConfig::default();
                config.merge_with_env();
                assert_eq!(config.default_device, "rocm");
            },
        );
    }

    #[test]
    fn merge_with_env_reads_log_level_and_threads() {
        temp_env::with_vars(
            vec![("BITNET_LOG_LEVEL", Some("warn")), ("BITNET_CPU_THREADS", Some("12"))],
            || {
                let mut config = CliConfig::default();
                config.merge_with_env();
                assert_eq!(config.logging.level, "warn");
                assert_eq!(config.performance.cpu_threads, Some(12));
            },
        );
    }

    #[test]
    fn merge_with_env_ignores_non_numeric_thread_value() {
        temp_env::with_vars(vec![("BITNET_CPU_THREADS", Some("not-a-number"))], || {
            let mut config = CliConfig::default();
            config.merge_with_env();
            assert!(config.performance.cpu_threads.is_none());
        });
    }

    #[test]
    fn builder_log_level_cpu_threads_batch_size_propagate() -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::new()
                    .device(Some("cpu".to_string()))
                    .log_level(Some("debug".to_string()))
                    .cpu_threads(Some(2))
                    .batch_size(Some(8))
                    .build()
            },
        )?;
        assert_eq!(config.default_device, "cpu");
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.performance.cpu_threads, Some(2));
        assert_eq!(config.performance.batch_size, 8);
        Ok(())
    }

    #[test]
    fn builder_none_options_leave_defaults_intact() -> anyhow::Result<()> {
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> {
                ConfigBuilder::new()
                    .device(None)
                    .log_level(None)
                    .cpu_threads(None)
                    .batch_size(None)
                    .build()
            },
        )?;
        let defaults = CliConfig::default();
        assert_eq!(config.default_device, defaults.default_device);
        assert_eq!(config.logging.level, defaults.logging.level);
        assert_eq!(config.performance.cpu_threads, defaults.performance.cpu_threads);
        assert_eq!(config.performance.batch_size, defaults.performance.batch_size);
        Ok(())
    }

    #[test]
    fn builder_fails_validation_for_invalid_device() -> anyhow::Result<()> {
        let err = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || match ConfigBuilder::new().device(Some("nope".to_string())).build() {
                Ok(_) => anyhow::bail!("validate should reject invalid device"),
                Err(err) => Ok(err),
            },
        )?;
        assert!(format!("{err}").contains("Invalid device"));
        Ok(())
    }

    #[test]
    fn validation_accepts_intel_a770_opencl_proof_lane_label() -> anyhow::Result<()> {
        let config =
            CliConfig { default_device: "intel-a770-opencl".to_string(), ..CliConfig::default() };

        config.validate()?;
        assert!(is_supported_device_label("intel-a770-opencl"));
        assert!(SUPPORTED_DEVICE_LABELS_TEXT.contains("intel-a770-opencl"));
        assert!(DEVICE_HELP.contains("intel-a770-opencl"));
        Ok(())
    }

    #[test]
    fn builder_from_file_loads_existing_config() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cfg.toml");
        let written = CliConfig { default_device: "cpu".to_string(), ..CliConfig::default() };
        written.save_to_file(&path)?;
        let config = temp_env::with_vars(
            vec![
                ("BITNET_DEVICE", Option::<&str>::None),
                ("BITNET_BACKEND", Option::<&str>::None),
                ("BITNET_LOG_LEVEL", Option::<&str>::None),
                ("BITNET_CPU_THREADS", Option::<&str>::None),
            ],
            || -> anyhow::Result<CliConfig> { ConfigBuilder::from_file(&path)?.build() },
        )?;
        assert_eq!(config.default_device, "cpu");
        Ok(())
    }
}
