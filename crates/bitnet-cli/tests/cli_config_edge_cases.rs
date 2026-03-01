//! Edge-case tests for `bitnet-cli` configuration management:
//! CliConfig, LoggingConfig, PerformanceConfig, ConfigBuilder, validation, and TOML serde.

use bitnet_cli::config::{CliConfig, ConfigBuilder, LoggingConfig, PerformanceConfig};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Default values
// ---------------------------------------------------------------------------

#[test]
fn cli_config_defaults() {
    let cfg = CliConfig::default();
    assert!(cfg.default_model.is_none());
    assert_eq!(cfg.default_device, "auto");
    assert!(cfg.default_quantization.is_none());
    assert!(cfg.model_cache_dir.is_none());
}

#[test]
fn logging_config_defaults() {
    let cfg = LoggingConfig::default();
    assert_eq!(cfg.level, "info");
    assert_eq!(cfg.format, "pretty");
    assert!(cfg.timestamps);
}

#[test]
fn performance_config_defaults() {
    let cfg = PerformanceConfig::default();
    assert!(cfg.cpu_threads.is_none());
    assert_eq!(cfg.batch_size, 1);
    assert!(cfg.memory_optimization);
}

// ---------------------------------------------------------------------------
// Validation — valid configs
// ---------------------------------------------------------------------------

#[test]
fn validate_default_config() {
    let cfg = CliConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_all_valid_devices() {
    for device in &["cpu", "cuda", "gpu", "vulkan", "opencl", "ocl", "npu", "auto"] {
        let mut cfg = CliConfig::default();
        cfg.default_device = device.to_string();
        assert!(cfg.validate().is_ok(), "device '{}' should be valid", device);
    }
}

#[test]
fn validate_all_valid_log_levels() {
    for level in &["trace", "debug", "info", "warn", "error"] {
        let mut cfg = CliConfig::default();
        cfg.logging.level = level.to_string();
        assert!(cfg.validate().is_ok(), "log level '{}' should be valid", level);
    }
}

#[test]
fn validate_all_valid_log_formats() {
    for fmt in &["pretty", "json", "compact"] {
        let mut cfg = CliConfig::default();
        cfg.logging.format = fmt.to_string();
        assert!(cfg.validate().is_ok(), "log format '{}' should be valid", fmt);
    }
}

// ---------------------------------------------------------------------------
// Validation — invalid configs
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_invalid_device() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "tpu".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_empty_device() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_invalid_log_level() {
    let mut cfg = CliConfig::default();
    cfg.logging.level = "verbose".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_invalid_log_format() {
    let mut cfg = CliConfig::default();
    cfg.logging.format = "xml".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_zero_batch_size() {
    let mut cfg = CliConfig::default();
    cfg.performance.batch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_large_batch_size() {
    let mut cfg = CliConfig::default();
    cfg.performance.batch_size = 1024;
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// TOML serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn toml_roundtrip_default() {
    let cfg = CliConfig::default();
    let toml_str = toml::to_string_pretty(&cfg).unwrap();
    let cfg2: CliConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(cfg2.default_device, cfg.default_device);
    assert_eq!(cfg2.logging.level, cfg.logging.level);
    assert_eq!(cfg2.performance.batch_size, cfg.performance.batch_size);
}

#[test]
fn toml_roundtrip_with_model_path() {
    let mut cfg = CliConfig::default();
    cfg.default_model = Some(PathBuf::from("models/phi4.gguf"));
    cfg.model_cache_dir = Some(PathBuf::from("/tmp/cache"));
    cfg.default_quantization = Some("int4".to_string());
    let toml_str = toml::to_string_pretty(&cfg).unwrap();
    let cfg2: CliConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(cfg2.default_model.as_deref(), Some(std::path::Path::new("models/phi4.gguf")));
    assert_eq!(cfg2.default_quantization.as_deref(), Some("int4"));
}

#[test]
fn toml_roundtrip_custom_performance() {
    let mut cfg = CliConfig::default();
    cfg.performance.cpu_threads = Some(8);
    cfg.performance.batch_size = 16;
    cfg.performance.memory_optimization = false;
    let toml_str = toml::to_string_pretty(&cfg).unwrap();
    let cfg2: CliConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(cfg2.performance.cpu_threads, Some(8));
    assert_eq!(cfg2.performance.batch_size, 16);
    assert!(!cfg2.performance.memory_optimization);
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

#[test]
fn load_nonexistent_file_returns_defaults() {
    let cfg = CliConfig::load_from_file("/tmp/nonexistent_bitnet_config.toml").unwrap();
    assert_eq!(cfg.default_device, "auto");
    assert_eq!(cfg.logging.level, "info");
}

#[test]
fn save_and_load_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_config.toml");

    let mut cfg = CliConfig::default();
    cfg.default_device = "cuda".to_string();
    cfg.performance.batch_size = 32;
    cfg.save_to_file(&path).unwrap();

    let cfg2 = CliConfig::load_from_file(&path).unwrap();
    assert_eq!(cfg2.default_device, "cuda");
    assert_eq!(cfg2.performance.batch_size, 32);
}

// ---------------------------------------------------------------------------
// ConfigBuilder
// ---------------------------------------------------------------------------

#[test]
fn builder_defaults_produce_valid_config() {
    // ConfigBuilder::build calls merge_with_env and validate
    // Use temp_env to ensure BITNET_DEVICE doesn't pollute.
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new().build().unwrap();
            assert_eq!(cfg.default_device, "auto");
        },
    );
}

#[test]
fn builder_device_override() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new().device(Some("cpu".to_string())).build().unwrap();
            assert_eq!(cfg.default_device, "cpu");
        },
    );
}

#[test]
fn builder_log_level_override() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new().log_level(Some("debug".to_string())).build().unwrap();
            assert_eq!(cfg.logging.level, "debug");
        },
    );
}

#[test]
fn builder_cpu_threads_override() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new().cpu_threads(Some(4)).build().unwrap();
            assert_eq!(cfg.performance.cpu_threads, Some(4));
        },
    );
}

#[test]
fn builder_batch_size_override() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new().batch_size(Some(64)).build().unwrap();
            assert_eq!(cfg.performance.batch_size, 64);
        },
    );
}

#[test]
fn builder_rejects_invalid_device() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let result = ConfigBuilder::new().device(Some("invalid".to_string())).build();
            assert!(result.is_err());
        },
    );
}

#[test]
fn builder_none_does_not_override() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let cfg = ConfigBuilder::new()
                .device(None)
                .log_level(None)
                .cpu_threads(None)
                .batch_size(None)
                .build()
                .unwrap();
            // Should still have defaults
            assert_eq!(cfg.default_device, "auto");
            assert_eq!(cfg.logging.level, "info");
            assert!(cfg.performance.cpu_threads.is_none());
            assert_eq!(cfg.performance.batch_size, 1);
        },
    );
}

// ---------------------------------------------------------------------------
// Environment variable merging
// ---------------------------------------------------------------------------

#[test]
fn merge_with_env_device() {
    temp_env::with_var("BITNET_DEVICE", Some("cuda"), || {
        let mut cfg = CliConfig::default();
        cfg.merge_with_env();
        assert_eq!(cfg.default_device, "cuda");
    });
}

#[test]
fn merge_with_env_log_level() {
    temp_env::with_var("BITNET_LOG_LEVEL", Some("trace"), || {
        let mut cfg = CliConfig::default();
        cfg.merge_with_env();
        assert_eq!(cfg.logging.level, "trace");
    });
}

#[test]
fn merge_with_env_cpu_threads() {
    temp_env::with_var("BITNET_CPU_THREADS", Some("16"), || {
        let mut cfg = CliConfig::default();
        cfg.merge_with_env();
        assert_eq!(cfg.performance.cpu_threads, Some(16));
    });
}

#[test]
fn merge_with_env_invalid_threads_ignored() {
    temp_env::with_var("BITNET_CPU_THREADS", Some("not_a_number"), || {
        let mut cfg = CliConfig::default();
        cfg.merge_with_env();
        // Invalid parse → threads remain None
        assert!(cfg.performance.cpu_threads.is_none());
    });
}

#[test]
fn merge_with_env_unset_vars_no_change() {
    temp_env::with_vars(
        [
            ("BITNET_DEVICE", None::<&str>),
            ("BITNET_LOG_LEVEL", None::<&str>),
            ("BITNET_CPU_THREADS", None::<&str>),
        ],
        || {
            let mut cfg = CliConfig::default();
            cfg.merge_with_env();
            assert_eq!(cfg.default_device, "auto");
            assert_eq!(cfg.logging.level, "info");
            assert!(cfg.performance.cpu_threads.is_none());
        },
    );
}

// ---------------------------------------------------------------------------
// Edge cases: case sensitivity, whitespace
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_uppercase_device() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "CPU".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_device_with_whitespace() {
    let mut cfg = CliConfig::default();
    cfg.default_device = " cpu ".to_string();
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_uppercase_log_level() {
    let mut cfg = CliConfig::default();
    cfg.logging.level = "INFO".to_string();
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Clone semantics
// ---------------------------------------------------------------------------

#[test]
fn cli_config_clone_is_independent() {
    let mut cfg = CliConfig::default();
    cfg.default_device = "cuda".to_string();
    let mut clone = cfg.clone();
    clone.default_device = "cpu".to_string();
    assert_eq!(cfg.default_device, "cuda");
    assert_eq!(clone.default_device, "cpu");
}
