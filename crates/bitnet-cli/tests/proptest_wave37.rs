//! Property-based tests — wave 37: CLI config round-trips, validation,
//! template detection consistency, and exit code mapping.

use bitnet_cli::config::{CliConfig, ConfigBuilder, LoggingConfig, PerformanceConfig};
use bitnet_cli::exit::*;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// CliConfig serde round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// CliConfig serializes to TOML and deserializes back identically.
    #[test]
    fn cli_config_toml_roundtrip(
        device in prop_oneof![
            Just("cpu".to_string()),
            Just("cuda".to_string()),
            Just("gpu".to_string()),
            Just("auto".to_string()),
        ],
        level in prop_oneof![
            Just("trace".to_string()),
            Just("debug".to_string()),
            Just("info".to_string()),
            Just("warn".to_string()),
            Just("error".to_string()),
        ],
        format in prop_oneof![
            Just("pretty".to_string()),
            Just("json".to_string()),
            Just("compact".to_string()),
        ],
        batch_size in 1usize..=256,
        timestamps in proptest::bool::ANY,
    ) {
        let cfg = CliConfig {
            default_model: None,
            default_device: device.clone(),
            default_quantization: None,
            logging: LoggingConfig { level: level.clone(), format: format.clone(), timestamps },
            performance: PerformanceConfig {
                cpu_threads: None,
                batch_size,
                memory_optimization: true,
            },
            model_cache_dir: None,
        };
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        let back: CliConfig = toml::from_str(&toml_str).unwrap();
        prop_assert_eq!(back.default_device, device);
        prop_assert_eq!(back.logging.level, level);
        prop_assert_eq!(back.logging.format, format);
        prop_assert_eq!(back.performance.batch_size, batch_size);
        prop_assert_eq!(back.logging.timestamps, timestamps);
    }

    /// Configs created with valid values always pass validate().
    #[test]
    fn valid_config_always_validates(
        device in prop_oneof![
            Just("cpu".to_string()),
            Just("cuda".to_string()),
            Just("gpu".to_string()),
            Just("vulkan".to_string()),
            Just("opencl".to_string()),
            Just("ocl".to_string()),
            Just("npu".to_string()),
            Just("auto".to_string()),
        ],
        level in prop_oneof![
            Just("trace".to_string()),
            Just("debug".to_string()),
            Just("info".to_string()),
            Just("warn".to_string()),
            Just("error".to_string()),
        ],
        format in prop_oneof![
            Just("pretty".to_string()),
            Just("json".to_string()),
            Just("compact".to_string()),
        ],
        batch_size in 1usize..=1024,
    ) {
        let cfg = CliConfig {
            default_device: device,
            logging: LoggingConfig { level, format, ..Default::default() },
            performance: PerformanceConfig { batch_size, ..Default::default() },
            ..Default::default()
        };
        prop_assert!(cfg.validate().is_ok());
    }

    /// Invalid device names are rejected by validate().
    #[test]
    fn invalid_device_rejected(name in "[a-z]{6,12}") {
        prop_assume!(!["cpu", "cuda", "gpu", "vulkan", "opencl", "ocl", "npu", "auto"]
            .contains(&name.as_str()));
        let cfg = CliConfig {
            default_device: name,
            ..Default::default()
        };
        prop_assert!(cfg.validate().is_err());
    }

    /// Invalid log levels are rejected.
    #[test]
    fn invalid_log_level_rejected(level in "[a-z]{3,10}") {
        prop_assume!(!["trace", "debug", "info", "warn", "error"].contains(&level.as_str()));
        let cfg = CliConfig {
            logging: LoggingConfig { level, ..Default::default() },
            ..Default::default()
        };
        prop_assert!(cfg.validate().is_err());
    }

    /// Invalid log formats are rejected.
    #[test]
    fn invalid_log_format_rejected(fmt in "[a-z]{3,10}") {
        prop_assume!(!["pretty", "json", "compact"].contains(&fmt.as_str()));
        let cfg = CliConfig {
            logging: LoggingConfig { format: fmt, ..Default::default() },
            ..Default::default()
        };
        prop_assert!(cfg.validate().is_err());
    }

    /// Batch size of 0 is rejected by validate().
    #[test]
    fn zero_batch_size_rejected(_seed in 0u32..10) {
        let cfg = CliConfig {
            performance: PerformanceConfig { batch_size: 0, ..Default::default() },
            ..Default::default()
        };
        prop_assert!(cfg.validate().is_err());
    }
}

// ---------------------------------------------------------------------------
// ConfigBuilder properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// ConfigBuilder.device() overrides the default device.
    #[test]
    fn builder_device_override(device in prop_oneof![
        Just("cpu".to_string()),
        Just("cuda".to_string()),
        Just("auto".to_string()),
    ]) {
        let cfg = ConfigBuilder::new()
            .device(Some(device.clone()))
            .build()
            .unwrap();
        prop_assert_eq!(cfg.default_device, device);
    }

    /// ConfigBuilder.log_level() overrides the default level.
    #[test]
    fn builder_log_level_override(level in prop_oneof![
        Just("trace".to_string()),
        Just("debug".to_string()),
        Just("info".to_string()),
        Just("warn".to_string()),
        Just("error".to_string()),
    ]) {
        let cfg = ConfigBuilder::new()
            .log_level(Some(level.clone()))
            .build()
            .unwrap();
        prop_assert_eq!(cfg.logging.level, level);
    }

    /// ConfigBuilder.cpu_threads() sets the thread count.
    #[test]
    fn builder_cpu_threads(threads in 1usize..=128) {
        let cfg = ConfigBuilder::new()
            .cpu_threads(Some(threads))
            .build()
            .unwrap();
        prop_assert_eq!(cfg.performance.cpu_threads, Some(threads));
    }

    /// ConfigBuilder.batch_size() sets the batch size.
    #[test]
    fn builder_batch_size(bs in 1usize..=256) {
        let cfg = ConfigBuilder::new()
            .batch_size(Some(bs))
            .build()
            .unwrap();
        prop_assert_eq!(cfg.performance.batch_size, bs);
    }

    /// Passing None to builder methods preserves defaults.
    #[test]
    fn builder_none_preserves_defaults(_seed in 0u32..10) {
        let cfg = ConfigBuilder::new()
            .device(None)
            .log_level(None)
            .cpu_threads(None)
            .batch_size(None)
            .build()
            .unwrap();
        let def = CliConfig::default();
        prop_assert_eq!(cfg.default_device, def.default_device);
        prop_assert_eq!(cfg.logging.level, def.logging.level);
        prop_assert_eq!(cfg.performance.cpu_threads, def.performance.cpu_threads);
    }
}

// ---------------------------------------------------------------------------
// CliConfig file round-trip via tempfile
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// save_to_file then load_from_file is identity.
    #[test]
    fn config_file_roundtrip(
        batch_size in 1usize..=256,
        mem_opt in proptest::bool::ANY,
    ) {
        let cfg = CliConfig {
            performance: PerformanceConfig {
                batch_size,
                memory_optimization: mem_opt,
                ..Default::default()
            },
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cfg.toml");
        cfg.save_to_file(&path).unwrap();
        let loaded = CliConfig::load_from_file(&path).unwrap();
        prop_assert_eq!(loaded.performance.batch_size, batch_size);
        prop_assert_eq!(loaded.performance.memory_optimization, mem_opt);
    }
}

// ---------------------------------------------------------------------------
// Exit code properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// All exit codes are distinct.
    #[test]
    fn exit_codes_distinct(_seed in 0u32..10) {
        let codes = [
            EXIT_SUCCESS,
            EXIT_GENERIC_FAIL,
            EXIT_STRICT_MAPPING,
            EXIT_STRICT_TOKENIZER,
            EXIT_NLL_TOO_HIGH,
            EXIT_TAU_TOO_LOW,
            EXIT_ARGMAX_MISMATCH,
            EXIT_LN_SUSPICIOUS,
            EXIT_PERF_FAIL,
            EXIT_RSS_FAIL,
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                prop_assert_ne!(codes[i], codes[j], "codes[{}] == codes[{}]", i, j);
            }
        }
    }

    /// EXIT_SUCCESS is always 0.
    #[test]
    fn exit_success_is_zero(_seed in 0u32..10) {
        prop_assert_eq!(EXIT_SUCCESS, 0);
    }
}
