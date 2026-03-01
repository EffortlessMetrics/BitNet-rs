//! GPU inference configuration with TOML, environment variable, and
//! default config sources.

use std::path::Path;
use std::{env, fmt, fs};

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────────

/// Errors produced by configuration loading or validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("unknown backend: {0}")]
    UnknownBackend(String),

    #[error("invalid log level: {0}")]
    InvalidLogLevel(String),

    #[error("invalid environment variable value for {key}: {value}")]
    InvalidEnvVar { key: String, value: String },
}

// ── BackendPreference ───────────────────────────────────────────────

/// Preferred GPU/compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendPreference {
    #[default]
    Auto,
    #[serde(alias = "opencl")]
    OpenCL,
    Vulkan,
    #[serde(alias = "cuda")]
    CUDA,
    #[serde(alias = "cpu")]
    CPU,
}

impl fmt::Display for BackendPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::OpenCL => write!(f, "opencl"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::CUDA => write!(f, "cuda"),
            Self::CPU => write!(f, "cpu"),
        }
    }
}

impl std::str::FromStr for BackendPreference {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "opencl" => Ok(Self::OpenCL),
            "vulkan" => Ok(Self::Vulkan),
            "cuda" => Ok(Self::CUDA),
            "cpu" => Ok(Self::CPU),
            other => Err(ConfigError::UnknownBackend(other.to_string())),
        }
    }
}

// ── LogLevel ────────────────────────────────────────────────────────

/// Log verbosity for GPU subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    #[default]
    Warn,
    Info,
    Debug,
    Trace,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warn => write!(f, "warn"),
            Self::Info => write!(f, "info"),
            Self::Debug => write!(f, "debug"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            other => Err(ConfigError::InvalidLogLevel(other.to_string())),
        }
    }
}

// ── TOML wrapper ────────────────────────────────────────────────────

/// Wrapper used for the `[gpu]` table in TOML files.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TomlWrapper {
    gpu: GpuInferenceConfig,
}

// ── GpuInferenceConfig ──────────────────────────────────────────────

/// Full configuration for GPU-accelerated inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuInferenceConfig {
    pub backend: BackendPreference,
    pub device_id: Option<usize>,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub memory_limit_mb: Option<u64>,
    pub enable_profiling: bool,
    pub kernel_cache_enabled: bool,
    pub workgroup_size_override: Option<usize>,
    pub enable_fp16: bool,
    pub pipeline_depth: u8,
    pub log_level: LogLevel,
}

impl Default for GpuInferenceConfig {
    fn default() -> Self {
        Self {
            backend: BackendPreference::Auto,
            device_id: None,
            max_batch_size: 1,
            max_sequence_length: 2048,
            memory_limit_mb: None,
            enable_profiling: false,
            kernel_cache_enabled: true,
            workgroup_size_override: None,
            enable_fp16: false,
            pipeline_depth: 2,
            log_level: LogLevel::Warn,
        }
    }
}

impl GpuInferenceConfig {
    // ── Constructors ────────────────────────────────────────────

    /// Load configuration from a TOML file at `path`.
    ///
    /// The file is expected to contain a `[gpu]` table.  If the file
    /// does not exist, returns `Ok(Self::default())`.
    pub fn from_toml(path: &Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            log::warn!("Config file not found: {}; using defaults", path.display());
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)?;
        let wrapper: TomlWrapper = toml::from_str(&text)?;
        Ok(wrapper.gpu)
    }

    /// Serialize to a TOML string (wrapped in `[gpu]`).
    pub fn to_toml(&self) -> Result<String, ConfigError> {
        let wrapper = TomlWrapper { gpu: self.clone() };
        Ok(toml::to_string_pretty(&wrapper)?)
    }

    /// Build a *partial* config from `BITNET_GPU_*` environment vars,
    /// then merge on top of `Self::default()`.
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut cfg = Self::default();

        if let Ok(v) = env::var("BITNET_GPU_BACKEND") {
            cfg.backend = v.parse()?;
        }
        if let Ok(v) = env::var("BITNET_GPU_DEVICE_ID") {
            cfg.device_id = Some(Self::parse_env_usize("BITNET_GPU_DEVICE_ID", &v)?);
        }
        if let Ok(v) = env::var("BITNET_GPU_MAX_BATCH_SIZE") {
            cfg.max_batch_size = Self::parse_env_usize("BITNET_GPU_MAX_BATCH_SIZE", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_MAX_SEQ_LEN") {
            cfg.max_sequence_length = Self::parse_env_usize("BITNET_GPU_MAX_SEQ_LEN", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_MEMORY_LIMIT_MB") {
            cfg.memory_limit_mb = Some(Self::parse_env_u64("BITNET_GPU_MEMORY_LIMIT_MB", &v)?);
        }
        if let Ok(v) = env::var("BITNET_GPU_PROFILING") {
            cfg.enable_profiling = Self::parse_env_bool("BITNET_GPU_PROFILING", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_KERNEL_CACHE") {
            cfg.kernel_cache_enabled = Self::parse_env_bool("BITNET_GPU_KERNEL_CACHE", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_WORKGROUP_SIZE") {
            cfg.workgroup_size_override =
                Some(Self::parse_env_usize("BITNET_GPU_WORKGROUP_SIZE", &v)?);
        }
        if let Ok(v) = env::var("BITNET_GPU_FP16") {
            cfg.enable_fp16 = Self::parse_env_bool("BITNET_GPU_FP16", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_PIPELINE_DEPTH") {
            cfg.pipeline_depth = Self::parse_env_u8("BITNET_GPU_PIPELINE_DEPTH", &v)?;
        }
        if let Ok(v) = env::var("BITNET_GPU_LOG_LEVEL") {
            cfg.log_level = v.parse()?;
        }

        Ok(cfg)
    }

    // ── Validation ──────────────────────────────────────────────

    /// Validate the configuration, returning all detected problems.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.pipeline_depth == 0 {
            return Err(ConfigError::Validation("pipeline_depth must be >= 1".into()));
        }
        if self.max_batch_size == 0 {
            return Err(ConfigError::Validation("max_batch_size must be >= 1".into()));
        }
        if self.max_sequence_length == 0 {
            return Err(ConfigError::Validation("max_sequence_length must be >= 1".into()));
        }
        if self.memory_limit_mb == Some(0) {
            return Err(ConfigError::Validation("memory_limit_mb must be > 0 when set".into()));
        }
        if self.workgroup_size_override == Some(0) {
            return Err(ConfigError::Validation(
                "workgroup_size_override must be > 0 when set".into(),
            ));
        }
        if self.enable_fp16 && self.backend == BackendPreference::CPU {
            return Err(ConfigError::Validation(
                "enable_fp16 is not supported with CPU backend".into(),
            ));
        }
        Ok(())
    }

    // ── Merge ───────────────────────────────────────────────────

    /// Overlay `other` on top of `self`. Fields in `other` that differ
    /// from the default take precedence.
    #[must_use]
    pub fn merge_with(&self, other: &Self) -> Self {
        let d = Self::default();
        Self {
            backend: if other.backend == d.backend { self.backend } else { other.backend },
            device_id: other.device_id.or(self.device_id),
            max_batch_size: if other.max_batch_size == d.max_batch_size {
                self.max_batch_size
            } else {
                other.max_batch_size
            },
            max_sequence_length: if other.max_sequence_length == d.max_sequence_length {
                self.max_sequence_length
            } else {
                other.max_sequence_length
            },
            memory_limit_mb: other.memory_limit_mb.or(self.memory_limit_mb),
            enable_profiling: if other.enable_profiling == d.enable_profiling {
                self.enable_profiling
            } else {
                other.enable_profiling
            },
            kernel_cache_enabled: if other.kernel_cache_enabled == d.kernel_cache_enabled {
                self.kernel_cache_enabled
            } else {
                other.kernel_cache_enabled
            },
            workgroup_size_override: other.workgroup_size_override.or(self.workgroup_size_override),
            enable_fp16: if other.enable_fp16 == d.enable_fp16 {
                self.enable_fp16
            } else {
                other.enable_fp16
            },
            pipeline_depth: if other.pipeline_depth == d.pipeline_depth {
                self.pipeline_depth
            } else {
                other.pipeline_depth
            },
            log_level: if other.log_level == d.log_level {
                self.log_level
            } else {
                other.log_level
            },
        }
    }

    // ── Helpers ─────────────────────────────────────────────────

    fn parse_env_usize(key: &str, val: &str) -> Result<usize, ConfigError> {
        val.parse::<usize>().map_err(|_| ConfigError::InvalidEnvVar {
            key: key.to_string(),
            value: val.to_string(),
        })
    }

    fn parse_env_u64(key: &str, val: &str) -> Result<u64, ConfigError> {
        val.parse::<u64>().map_err(|_| ConfigError::InvalidEnvVar {
            key: key.to_string(),
            value: val.to_string(),
        })
    }

    fn parse_env_u8(key: &str, val: &str) -> Result<u8, ConfigError> {
        val.parse::<u8>().map_err(|_| ConfigError::InvalidEnvVar {
            key: key.to_string(),
            value: val.to_string(),
        })
    }

    fn parse_env_bool(key: &str, val: &str) -> Result<bool, ConfigError> {
        match val.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(ConfigError::InvalidEnvVar { key: key.to_string(), value: val.to_string() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        let cfg = GpuInferenceConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn backend_display_roundtrip() {
        for b in [
            BackendPreference::Auto,
            BackendPreference::OpenCL,
            BackendPreference::Vulkan,
            BackendPreference::CUDA,
            BackendPreference::CPU,
        ] {
            let s = b.to_string();
            let parsed: BackendPreference = s.parse().unwrap();
            assert_eq!(b, parsed);
        }
    }

    #[test]
    fn log_level_display_roundtrip() {
        for l in [LogLevel::Error, LogLevel::Warn, LogLevel::Info, LogLevel::Debug, LogLevel::Trace]
        {
            let s = l.to_string();
            let parsed: LogLevel = s.parse().unwrap();
            assert_eq!(l, parsed);
        }
    }

    #[test]
    fn unknown_backend_is_error() {
        let r = "quantum".parse::<BackendPreference>();
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("unknown backend"));
    }

    #[test]
    fn invalid_log_level_is_error() {
        let r = "verbose".parse::<LogLevel>();
        assert!(r.is_err());
    }
}
