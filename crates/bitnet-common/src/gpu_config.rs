//! GPU configuration file format for BitNet inference.
//!
//! Loads [`GpuConfig`] from a TOML file (`gpu.toml`) with environment variable
//! overrides via `BITNET_GPU_*` prefixed variables.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Preferred GPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    OpenCl,
    Vulkan,
    Metal,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::OpenCl => write!(f, "opencl"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Metal => write!(f, "metal"),
        }
    }
}

impl std::str::FromStr for GpuBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cuda" => Ok(Self::Cuda),
            "opencl" => Ok(Self::OpenCl),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            other => Err(format!("unknown GPU backend: {other}")),
        }
    }
}

/// Warmup level controlling how much work is done at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WarmupLevel {
    /// No warmup — fastest startup.
    None,
    /// Minimal warmup — compile kernels only.
    Minimal,
    /// Full warmup — compile kernels and run a small test workload.
    Full,
}

impl std::fmt::Display for WarmupLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Minimal => write!(f, "minimal"),
            Self::Full => write!(f, "full"),
        }
    }
}

impl std::str::FromStr for WarmupLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "minimal" => Ok(Self::Minimal),
            "full" => Ok(Self::Full),
            other => Err(format!("unknown warmup level: {other}")),
        }
    }
}

/// GPU configuration loaded from TOML with environment variable overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Preferred GPU backend (e.g. `cuda`, `opencl`).
    /// Override: `BITNET_GPU_BACKEND`
    pub preferred_backend: GpuBackend,

    /// Zero-based device index.
    /// Override: `BITNET_GPU_DEVICE_INDEX`
    pub device_index: u32,

    /// Memory limit in bytes (0 = unlimited).
    /// Override: `BITNET_GPU_MEMORY_LIMIT`
    pub memory_limit: u64,

    /// Kernel variant name (e.g. `"default"`, `"tiled"`, `"vectorized"`).
    /// Override: `BITNET_GPU_KERNEL_VARIANT`
    pub kernel_variant: String,

    /// Work-group / thread-block size.
    /// Override: `BITNET_GPU_WORK_GROUP_SIZE`
    pub work_group_size: u32,

    /// Enable GPU profiling / event timers.
    /// Override: `BITNET_GPU_ENABLE_PROFILING`
    pub enable_profiling: bool,

    /// Warmup level at startup.
    /// Override: `BITNET_GPU_WARMUP_LEVEL`
    pub warmup_level: WarmupLevel,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: GpuBackend::Cuda,
            device_index: 0,
            memory_limit: 0,
            kernel_variant: "default".to_string(),
            work_group_size: 256,
            enable_profiling: false,
            warmup_level: WarmupLevel::Minimal,
        }
    }
}

/// Errors that can occur when loading or validating a [`GpuConfig`].
#[derive(Debug, thiserror::Error)]
pub enum GpuConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("invalid environment override {key}={value}: {reason}")]
    EnvOverride {
        key: String,
        value: String,
        reason: String,
    },
}

impl GpuConfig {
    /// Generate a default configuration TOML string.
    pub fn default_toml() -> String {
        let cfg = Self::default();
        toml::to_string_pretty(&cfg).expect("default config should serialize")
    }

    /// Load configuration from a TOML file, falling back to defaults for
    /// missing fields, then apply environment variable overrides.
    pub fn load(path: &Path) -> Result<Self, GpuConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let mut cfg: GpuConfig = toml::from_str(&contents)?;
        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load from TOML string (useful for testing).
    pub fn from_toml(toml_str: &str) -> Result<Self, GpuConfigError> {
        let mut cfg: GpuConfig = toml::from_str(toml_str)?;
        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load only from environment variables, starting from defaults.
    pub fn from_env() -> Result<Self, GpuConfigError> {
        let mut cfg = Self::default();
        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration, returning an error with a descriptive
    /// message on failure.
    pub fn validate(&self) -> Result<(), GpuConfigError> {
        if self.work_group_size == 0 {
            return Err(GpuConfigError::Validation(
                "work_group_size must be > 0".into(),
            ));
        }
        if !self.work_group_size.is_power_of_two() {
            return Err(GpuConfigError::Validation(
                format!(
                    "work_group_size must be a power of two, got {}",
                    self.work_group_size
                ),
            ));
        }
        if self.work_group_size > 1024 {
            return Err(GpuConfigError::Validation(
                format!(
                    "work_group_size must be <= 1024, got {}",
                    self.work_group_size
                ),
            ));
        }
        if self.kernel_variant.is_empty() {
            return Err(GpuConfigError::Validation(
                "kernel_variant must not be empty".into(),
            ));
        }
        if self.kernel_variant.len() > 64 {
            return Err(GpuConfigError::Validation(
                "kernel_variant must be <= 64 characters".into(),
            ));
        }
        Ok(())
    }

    /// Apply `BITNET_GPU_*` environment variable overrides.
    pub fn apply_env_overrides(&mut self) -> Result<(), GpuConfigError> {
        if let Ok(val) = std::env::var("BITNET_GPU_BACKEND") {
            self.preferred_backend =
                val.parse::<GpuBackend>().map_err(|reason| {
                    GpuConfigError::EnvOverride {
                        key: "BITNET_GPU_BACKEND".into(),
                        value: val.clone(),
                        reason,
                    }
                })?;
        }

        if let Ok(val) = std::env::var("BITNET_GPU_DEVICE_INDEX") {
            self.device_index =
                val.parse::<u32>().map_err(|e| GpuConfigError::EnvOverride {
                    key: "BITNET_GPU_DEVICE_INDEX".into(),
                    value: val.clone(),
                    reason: e.to_string(),
                })?;
        }

        if let Ok(val) = std::env::var("BITNET_GPU_MEMORY_LIMIT") {
            self.memory_limit =
                val.parse::<u64>().map_err(|e| GpuConfigError::EnvOverride {
                    key: "BITNET_GPU_MEMORY_LIMIT".into(),
                    value: val.clone(),
                    reason: e.to_string(),
                })?;
        }

        if let Ok(val) = std::env::var("BITNET_GPU_KERNEL_VARIANT") {
            self.kernel_variant = val;
        }

        if let Ok(val) = std::env::var("BITNET_GPU_WORK_GROUP_SIZE") {
            self.work_group_size =
                val.parse::<u32>().map_err(|e| GpuConfigError::EnvOverride {
                    key: "BITNET_GPU_WORK_GROUP_SIZE".into(),
                    value: val.clone(),
                    reason: e.to_string(),
                })?;
        }

        if let Ok(val) = std::env::var("BITNET_GPU_ENABLE_PROFILING") {
            self.enable_profiling = matches!(val.as_str(), "1" | "true" | "yes");
        }

        if let Ok(val) = std::env::var("BITNET_GPU_WARMUP_LEVEL") {
            self.warmup_level =
                val.parse::<WarmupLevel>().map_err(|reason| {
                    GpuConfigError::EnvOverride {
                        key: "BITNET_GPU_WARMUP_LEVEL".into(),
                        value: val.clone(),
                        reason,
                    }
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = GpuConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_default_toml_round_trips() {
        let toml_str = GpuConfig::default_toml();
        let cfg: GpuConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(cfg, GpuConfig::default());
    }

    #[test]
    fn test_from_toml_minimal() {
        // All fields have defaults via serde, so an empty table should fail
        // but a complete one should work.
        let toml_str = r#"
preferred_backend = "opencl"
device_index = 1
memory_limit = 4294967296
kernel_variant = "tiled"
work_group_size = 128
enable_profiling = true
warmup_level = "full"
"#;
        let cfg = GpuConfig::from_toml(toml_str).unwrap();
        assert_eq!(cfg.preferred_backend, GpuBackend::OpenCl);
        assert_eq!(cfg.device_index, 1);
        assert_eq!(cfg.memory_limit, 4_294_967_296);
        assert_eq!(cfg.kernel_variant, "tiled");
        assert_eq!(cfg.work_group_size, 128);
        assert!(cfg.enable_profiling);
        assert_eq!(cfg.warmup_level, WarmupLevel::Full);
    }

    #[test]
    fn test_validation_work_group_size_zero() {
        let mut cfg = GpuConfig::default();
        cfg.work_group_size = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("work_group_size must be > 0"));
    }

    #[test]
    fn test_validation_work_group_size_not_power_of_two() {
        let mut cfg = GpuConfig::default();
        cfg.work_group_size = 100;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn test_validation_work_group_size_too_large() {
        let mut cfg = GpuConfig::default();
        cfg.work_group_size = 2048;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("<= 1024"));
    }

    #[test]
    fn test_validation_empty_kernel_variant() {
        let mut cfg = GpuConfig::default();
        cfg.kernel_variant = String::new();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_validation_long_kernel_variant() {
        let mut cfg = GpuConfig::default();
        cfg.kernel_variant = "x".repeat(65);
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("<= 64 characters"));
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_override_backend() {
        temp_env::with_vars(
            [
                ("BITNET_GPU_BACKEND", Some("vulkan")),
                ("BITNET_GPU_DEVICE_INDEX", None::<&str>),
                ("BITNET_GPU_MEMORY_LIMIT", None),
                ("BITNET_GPU_KERNEL_VARIANT", None),
                ("BITNET_GPU_WORK_GROUP_SIZE", None),
                ("BITNET_GPU_ENABLE_PROFILING", None),
                ("BITNET_GPU_WARMUP_LEVEL", None),
            ],
            || {
                let cfg = GpuConfig::from_env().unwrap();
                assert_eq!(cfg.preferred_backend, GpuBackend::Vulkan);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_override_multiple_fields() {
        temp_env::with_vars(
            [
                ("BITNET_GPU_BACKEND", Some("metal")),
                ("BITNET_GPU_DEVICE_INDEX", Some("3")),
                ("BITNET_GPU_MEMORY_LIMIT", Some("8589934592")),
                ("BITNET_GPU_KERNEL_VARIANT", Some("vectorized")),
                ("BITNET_GPU_WORK_GROUP_SIZE", Some("512")),
                ("BITNET_GPU_ENABLE_PROFILING", Some("1")),
                ("BITNET_GPU_WARMUP_LEVEL", Some("full")),
            ],
            || {
                let cfg = GpuConfig::from_env().unwrap();
                assert_eq!(cfg.preferred_backend, GpuBackend::Metal);
                assert_eq!(cfg.device_index, 3);
                assert_eq!(cfg.memory_limit, 8_589_934_592);
                assert_eq!(cfg.kernel_variant, "vectorized");
                assert_eq!(cfg.work_group_size, 512);
                assert!(cfg.enable_profiling);
                assert_eq!(cfg.warmup_level, WarmupLevel::Full);
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_override_invalid_backend() {
        temp_env::with_vars(
            [
                ("BITNET_GPU_BACKEND", Some("directx")),
                ("BITNET_GPU_DEVICE_INDEX", None::<&str>),
                ("BITNET_GPU_MEMORY_LIMIT", None),
                ("BITNET_GPU_KERNEL_VARIANT", None),
                ("BITNET_GPU_WORK_GROUP_SIZE", None),
                ("BITNET_GPU_ENABLE_PROFILING", None),
                ("BITNET_GPU_WARMUP_LEVEL", None),
            ],
            || {
                let err = GpuConfig::from_env().unwrap_err();
                match err {
                    GpuConfigError::EnvOverride { key, .. } => {
                        assert_eq!(key, "BITNET_GPU_BACKEND");
                    }
                    other => panic!("expected EnvOverride, got: {other}"),
                }
            },
        );
    }

    #[test]
    fn test_load_from_tempfile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.toml");
        std::fs::write(&path, GpuConfig::default_toml()).unwrap();
        let cfg = GpuConfig::load(&path).unwrap();
        assert_eq!(cfg, GpuConfig::default());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = GpuConfig::load(Path::new("/nonexistent/gpu.toml"));
        assert!(matches!(result, Err(GpuConfigError::Io(_))));
    }

    #[test]
    fn test_backend_display_roundtrip() {
        for backend in &[
            GpuBackend::Cuda,
            GpuBackend::OpenCl,
            GpuBackend::Vulkan,
            GpuBackend::Metal,
        ] {
            let s = backend.to_string();
            let parsed: GpuBackend = s.parse().unwrap();
            assert_eq!(*backend, parsed);
        }
    }

    #[test]
    fn test_warmup_level_display_roundtrip() {
        for level in &[WarmupLevel::None, WarmupLevel::Minimal, WarmupLevel::Full] {
            let s = level.to_string();
            let parsed: WarmupLevel = s.parse().unwrap();
            assert_eq!(*level, parsed);
        }
    }
}
