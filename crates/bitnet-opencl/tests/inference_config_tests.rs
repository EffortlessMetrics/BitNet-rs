//! Integration tests for `GpuInferenceConfig`.

use std::io::Write;

use bitnet_opencl::{BackendPreference, GpuInferenceConfig, LogLevel};
use serial_test::serial;

// ── Default / basic ─────────────────────────────────────────────────

#[test]
fn default_config_has_sensible_values() {
    let cfg = GpuInferenceConfig::default();
    assert_eq!(cfg.backend, BackendPreference::Auto);
    assert_eq!(cfg.max_batch_size, 1);
    assert_eq!(cfg.max_sequence_length, 2048);
    assert!(!cfg.enable_profiling);
    assert!(cfg.kernel_cache_enabled);
    assert_eq!(cfg.pipeline_depth, 2);
    assert_eq!(cfg.log_level, LogLevel::Warn);
    assert!(!cfg.enable_fp16);
    assert!(cfg.device_id.is_none());
    assert!(cfg.memory_limit_mb.is_none());
    assert!(cfg.workgroup_size_override.is_none());
}

#[test]
fn default_config_passes_validation() {
    let cfg = GpuInferenceConfig::default();
    assert!(cfg.validate().is_ok());
}

// ── TOML round-trip ─────────────────────────────────────────────────

#[test]
fn toml_round_trip() {
    let original = GpuInferenceConfig {
        backend: BackendPreference::CUDA,
        device_id: Some(1),
        max_batch_size: 4,
        max_sequence_length: 4096,
        memory_limit_mb: Some(8192),
        enable_profiling: true,
        kernel_cache_enabled: false,
        workgroup_size_override: Some(256),
        enable_fp16: true,
        pipeline_depth: 3,
        log_level: LogLevel::Debug,
    };
    let toml_str = original.to_toml().unwrap();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &toml_str).unwrap();
    let loaded = GpuInferenceConfig::from_toml(tmp.path()).unwrap();
    assert_eq!(original, loaded);
}

#[test]
fn toml_serialize_contains_gpu_table() {
    let cfg = GpuInferenceConfig::default();
    let s = cfg.to_toml().unwrap();
    assert!(s.contains("[gpu]"));
}

#[test]
fn toml_deserialize_minimal_gpu_table() {
    let toml_str = "[gpu]\n";
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), toml_str).unwrap();
    let cfg = GpuInferenceConfig::from_toml(tmp.path()).unwrap();
    assert_eq!(cfg, GpuInferenceConfig::default());
}

#[test]
fn toml_file_not_found_returns_defaults() {
    let path = std::path::Path::new("/nonexistent/gpu.toml");
    let cfg = GpuInferenceConfig::from_toml(path).unwrap();
    assert_eq!(cfg, GpuInferenceConfig::default());
}

#[test]
fn toml_invalid_syntax_is_error() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    write!(tmp, "[gpu]\nbackend = !!!").unwrap();
    let r = GpuInferenceConfig::from_toml(tmp.path());
    assert!(r.is_err());
}

#[test]
fn toml_unknown_backend_value_is_error() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    write!(tmp, "[gpu]\nbackend = \"quantum\"").unwrap();
    let r = GpuInferenceConfig::from_toml(tmp.path());
    assert!(r.is_err());
}

// ── Env vars ────────────────────────────────────────────────────────

#[test]
#[serial(bitnet_env)]
fn env_backend_overrides_default() {
    temp_env::with_var("BITNET_GPU_BACKEND", Some("vulkan"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.backend, BackendPreference::Vulkan);
    });
}

#[test]
#[serial(bitnet_env)]
fn env_device_id() {
    temp_env::with_var("BITNET_GPU_DEVICE_ID", Some("3"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.device_id, Some(3));
    });
}

#[test]
#[serial(bitnet_env)]
fn env_batch_size() {
    temp_env::with_var("BITNET_GPU_MAX_BATCH_SIZE", Some("8"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.max_batch_size, 8);
    });
}

#[test]
#[serial(bitnet_env)]
fn env_profiling_bool_variants() {
    for (val, expected) in [("1", true), ("true", true), ("0", false), ("off", false)] {
        temp_env::with_var("BITNET_GPU_PROFILING", Some(val), || {
            let cfg = GpuInferenceConfig::from_env().unwrap();
            assert_eq!(cfg.enable_profiling, expected, "input={val}");
        });
    }
}

#[test]
#[serial(bitnet_env)]
fn env_invalid_value_is_error() {
    temp_env::with_var("BITNET_GPU_MAX_BATCH_SIZE", Some("not_a_number"), || {
        let r = GpuInferenceConfig::from_env();
        assert!(r.is_err());
    });
}

#[test]
#[serial(bitnet_env)]
fn env_log_level() {
    temp_env::with_var("BITNET_GPU_LOG_LEVEL", Some("trace"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.log_level, LogLevel::Trace);
    });
}

#[test]
#[serial(bitnet_env)]
fn env_pipeline_depth() {
    temp_env::with_var("BITNET_GPU_PIPELINE_DEPTH", Some("4"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.pipeline_depth, 4);
    });
}

#[test]
#[serial(bitnet_env)]
fn env_fp16() {
    temp_env::with_var("BITNET_GPU_FP16", Some("true"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert!(cfg.enable_fp16);
    });
}

#[test]
#[serial(bitnet_env)]
fn env_memory_limit() {
    temp_env::with_var("BITNET_GPU_MEMORY_LIMIT_MB", Some("4096"), || {
        let cfg = GpuInferenceConfig::from_env().unwrap();
        assert_eq!(cfg.memory_limit_mb, Some(4096));
    });
}

// ── Validation ──────────────────────────────────────────────────────

#[test]
fn validation_pipeline_depth_zero() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.pipeline_depth = 0;
    let r = cfg.validate();
    assert!(r.is_err());
    assert!(r.unwrap_err().to_string().contains("pipeline_depth"));
}

#[test]
fn validation_batch_size_zero() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.max_batch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_sequence_length_zero() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.max_sequence_length = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_memory_limit_zero() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.memory_limit_mb = Some(0);
    let r = cfg.validate();
    assert!(r.is_err());
    assert!(r.unwrap_err().to_string().contains("memory_limit_mb"));
}

#[test]
fn validation_workgroup_size_zero() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.workgroup_size_override = Some(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validation_fp16_with_cpu_is_error() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.backend = BackendPreference::CPU;
    cfg.enable_fp16 = true;
    let r = cfg.validate();
    assert!(r.is_err());
    assert!(r.unwrap_err().to_string().contains("fp16"));
}

#[test]
fn validation_fp16_with_cuda_is_ok() {
    let mut cfg = GpuInferenceConfig::default();
    cfg.backend = BackendPreference::CUDA;
    cfg.enable_fp16 = true;
    assert!(cfg.validate().is_ok());
}

// ── Merge ───────────────────────────────────────────────────────────

#[test]
fn merge_env_overrides_file() {
    let file_cfg = GpuInferenceConfig {
        backend: BackendPreference::OpenCL,
        max_batch_size: 4,
        ..Default::default()
    };
    let env_cfg = GpuInferenceConfig { backend: BackendPreference::CUDA, ..Default::default() };
    let merged = file_cfg.merge_with(&env_cfg);
    // env_cfg set backend to CUDA (non-default) → wins
    assert_eq!(merged.backend, BackendPreference::CUDA);
    // env_cfg left batch_size at default → file value preserved
    assert_eq!(merged.max_batch_size, 4);
}

#[test]
fn merge_preserves_base_when_overlay_is_default() {
    let base = GpuInferenceConfig {
        backend: BackendPreference::Vulkan,
        max_batch_size: 8,
        pipeline_depth: 4,
        ..Default::default()
    };
    let overlay = GpuInferenceConfig::default();
    let merged = base.merge_with(&overlay);
    assert_eq!(merged.backend, BackendPreference::Vulkan);
    assert_eq!(merged.max_batch_size, 8);
    assert_eq!(merged.pipeline_depth, 4);
}

#[test]
fn merge_option_fields_use_or() {
    let base = GpuInferenceConfig {
        device_id: Some(0),
        memory_limit_mb: Some(2048),
        ..Default::default()
    };
    let overlay = GpuInferenceConfig { device_id: Some(1), ..Default::default() };
    let merged = base.merge_with(&overlay);
    assert_eq!(merged.device_id, Some(1));
    // overlay had None → base value kept
    assert_eq!(merged.memory_limit_mb, Some(2048));
}

#[test]
fn merge_three_layers() {
    let defaults = GpuInferenceConfig::default();
    let file_cfg = GpuInferenceConfig {
        backend: BackendPreference::OpenCL,
        max_batch_size: 4,
        ..Default::default()
    };
    let env_cfg = GpuInferenceConfig { pipeline_depth: 5, ..Default::default() };
    let merged = defaults.merge_with(&file_cfg).merge_with(&env_cfg);
    assert_eq!(merged.backend, BackendPreference::OpenCL);
    assert_eq!(merged.max_batch_size, 4);
    assert_eq!(merged.pipeline_depth, 5);
}

// ── Backend / LogLevel parsing ──────────────────────────────────────

#[test]
fn backend_unknown_string_is_error() {
    let r = "quantum".parse::<BackendPreference>();
    assert!(r.is_err());
}

#[test]
fn backend_case_insensitive() {
    assert_eq!("CUDA".parse::<BackendPreference>().unwrap(), BackendPreference::CUDA);
    assert_eq!("Vulkan".parse::<BackendPreference>().unwrap(), BackendPreference::Vulkan);
}

#[test]
fn log_level_case_insensitive() {
    assert_eq!("DEBUG".parse::<LogLevel>().unwrap(), LogLevel::Debug);
}
