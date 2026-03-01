//! Edge-case tests for bitnet-models: config parsing, checkpoint management,
//! production loader configuration, device strategy, validation, and memory estimation.

use bitnet_models::LoadConfig;
use bitnet_models::checkpoint::{CheckpointError, CheckpointFormat, CheckpointManager};
use bitnet_models::config::{
    ConfigError, GgufModelConfig, GgufQuantizationConfig, MemoryEstimate, RopeScaling,
    RopeScalingType,
};
use bitnet_models::formats::gguf::GgufValue;
use bitnet_models::production_loader::{
    DeviceConfig, DeviceStrategy, MemoryRequirements, ProductionLoadConfig, ProductionModelLoader,
    ValidationResult,
};
use std::collections::HashMap;
use std::path::Path;

// ===========================================================================
// CheckpointFormat
// ===========================================================================

#[test]
fn checkpoint_format_detect_gguf_extension() {
    let fmt = CheckpointFormat::detect(Path::new("model.gguf"));
    assert_eq!(fmt, CheckpointFormat::Gguf);
}

#[test]
fn checkpoint_format_detect_safetensors_extension() {
    let fmt = CheckpointFormat::detect(Path::new("model.safetensors"));
    assert_eq!(fmt, CheckpointFormat::SafeTensors);
}

#[test]
fn checkpoint_format_detect_pytorch_pt() {
    let fmt = CheckpointFormat::detect(Path::new("model.pt"));
    assert_eq!(fmt, CheckpointFormat::PyTorch);
}

#[test]
fn checkpoint_format_detect_pytorch_bin() {
    let fmt = CheckpointFormat::detect(Path::new("model.bin"));
    assert_eq!(fmt, CheckpointFormat::PyTorch);
}

#[test]
fn checkpoint_format_detect_unknown() {
    let fmt = CheckpointFormat::detect(Path::new("model.xyz"));
    assert_eq!(fmt, CheckpointFormat::Custom);
}

#[test]
fn checkpoint_format_detect_no_extension() {
    let fmt = CheckpointFormat::detect(Path::new("model"));
    assert_eq!(fmt, CheckpointFormat::Custom);
}

#[test]
fn checkpoint_format_as_str() {
    assert_eq!(CheckpointFormat::Gguf.as_str(), "GGUF");
    assert_eq!(CheckpointFormat::SafeTensors.as_str(), "SafeTensors");
    assert_eq!(CheckpointFormat::PyTorch.as_str(), "PyTorch");
    assert_eq!(CheckpointFormat::Custom.as_str(), "Custom");
}

#[test]
fn checkpoint_format_display() {
    assert_eq!(format!("{}", CheckpointFormat::Gguf), "GGUF");
}

#[test]
fn checkpoint_format_serde_roundtrip() {
    let fmt = CheckpointFormat::SafeTensors;
    let json = serde_json::to_string(&fmt).unwrap();
    let back: CheckpointFormat = serde_json::from_str(&json).unwrap();
    assert_eq!(fmt, back);
}

// ===========================================================================
// CheckpointManager
// ===========================================================================

#[test]
fn checkpoint_manager_new_empty() {
    let mgr = CheckpointManager::new();
    assert!(mgr.is_empty());
    assert_eq!(mgr.len(), 0);
}

#[test]
fn checkpoint_manager_list_empty() {
    let mgr = CheckpointManager::new();
    assert!(mgr.list().is_empty());
}

#[test]
fn checkpoint_manager_get_missing() {
    let mgr = CheckpointManager::new();
    assert!(mgr.get(Path::new("nonexistent.gguf")).is_none());
}

#[test]
fn checkpoint_manager_remove_missing() {
    let mgr = CheckpointManager::new();
    let result = mgr.remove(Path::new("nonexistent.gguf"));
    assert!(result.is_err());
}

#[test]
fn checkpoint_manager_search_empty() {
    let mgr = CheckpointManager::new();
    assert!(mgr.search_by_name("model").is_empty());
}

#[test]
fn checkpoint_manager_filter_empty() {
    let mgr = CheckpointManager::new();
    assert!(mgr.filter_by_format(CheckpointFormat::Gguf).is_empty());
}

#[test]
fn checkpoint_manager_default_is_empty() {
    let mgr = CheckpointManager::default();
    assert!(mgr.is_empty());
}

// ===========================================================================
// CheckpointError
// ===========================================================================

#[test]
fn checkpoint_error_not_found_display() {
    let err = CheckpointError::NotFound("model.gguf".into());
    assert!(format!("{err}").contains("not found"));
}

#[test]
fn checkpoint_error_duplicate_display() {
    let err = CheckpointError::Duplicate("model.gguf".into());
    assert!(format!("{err}").contains("duplicate"));
}

#[test]
fn checkpoint_error_hash_mismatch_display() {
    let err = CheckpointError::HashMismatch {
        path: "model.gguf".into(),
        expected: "abc123".into(),
        actual: "def456".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("hash mismatch"));
    assert!(msg.contains("abc123"));
    assert!(msg.contains("def456"));
}

// ===========================================================================
// GgufQuantizationConfig
// ===========================================================================

#[test]
fn quant_config_default() {
    let cfg = GgufQuantizationConfig::default();
    assert_eq!(cfg.bit_width, 2);
    assert_eq!(cfg.block_size, 64);
    assert_eq!(cfg.format, "I2_S");
}

#[test]
fn quant_config_serde_roundtrip() {
    let cfg = GgufQuantizationConfig { bit_width: 4, block_size: 32, format: "Q4_0".into() };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: GgufQuantizationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, back);
}

// ===========================================================================
// RopeScaling
// ===========================================================================

#[test]
fn rope_scaling_serde_roundtrip() {
    let scaling = RopeScaling { scaling_type: RopeScalingType::YaRn, factor: 4.0 };
    let json = serde_json::to_string(&scaling).unwrap();
    let back: RopeScaling = serde_json::from_str(&json).unwrap();
    assert_eq!(scaling, back);
}

#[test]
fn rope_scaling_type_variants() {
    let variants = [
        RopeScalingType::None,
        RopeScalingType::Linear,
        RopeScalingType::Ntk,
        RopeScalingType::YaRn,
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: RopeScalingType = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

// ===========================================================================
// GgufModelConfig — from_gguf_metadata
// ===========================================================================

fn minimal_llama_metadata() -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".into(), GgufValue::String("llama".into())),
        ("llama.vocab_size".into(), GgufValue::U32(32000)),
        ("llama.embedding_length".into(), GgufValue::U32(4096)),
        ("llama.block_count".into(), GgufValue::U32(32)),
        ("llama.attention.head_count".into(), GgufValue::U32(32)),
        ("llama.attention.head_count_kv".into(), GgufValue::U32(8)),
        ("llama.feed_forward_length".into(), GgufValue::U32(11008)),
        ("llama.context_length".into(), GgufValue::U32(4096)),
    ])
}

#[test]
fn config_from_llama_metadata() {
    let meta = minimal_llama_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "llama");
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate_size, 11008);
    assert_eq!(cfg.max_seq_len, 4096);
}

#[test]
fn config_from_empty_metadata_uses_defaults() {
    let meta = HashMap::new();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "llama");
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.num_heads, 32);
    // num_kv_heads defaults to num_heads when not specified
    assert_eq!(cfg.num_kv_heads, 32);
}

#[test]
fn config_gqa_detection() {
    let meta = minimal_llama_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 4);
}

#[test]
fn config_mha_when_kv_heads_equals_heads() {
    let mut meta = minimal_llama_metadata();
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::U32(32));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(!cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 1);
}

#[test]
fn config_validate_valid() {
    let meta = minimal_llama_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validate_zero_vocab() {
    let mut meta = minimal_llama_metadata();
    meta.insert("llama.vocab_size".into(), GgufValue::U32(0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn config_validate_kv_heads_exceeds_heads() {
    let mut meta = minimal_llama_metadata();
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::U32(64));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.validate().is_err());
}

#[test]
fn config_memory_estimate_nonzero() {
    let meta = minimal_llama_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let est = cfg.memory_estimate();
    assert!(est.weight_bytes > 0);
    assert!(est.kv_cache_bytes > 0);
    assert!(est.total_bytes > est.weight_bytes);
    assert!(!est.summary.is_empty());
}

#[test]
fn config_serde_roundtrip() {
    let meta = minimal_llama_metadata();
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: GgufModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, back);
}

#[test]
fn config_rope_scaling_from_metadata() {
    let mut meta = minimal_llama_metadata();
    meta.insert("llama.rope.scaling.type".into(), GgufValue::String("yarn".into()));
    meta.insert("llama.rope.scaling.factor".into(), GgufValue::F32(4.0));
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert!(cfg.rope_scaling.is_some());
    let scaling = cfg.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, RopeScalingType::YaRn);
    assert!((scaling.factor - 4.0).abs() < f32::EPSILON);
}

#[test]
fn config_phi4_like_metadata() {
    let meta = HashMap::from([
        ("general.architecture".into(), GgufValue::String("phi".into())),
        ("general.name".into(), GgufValue::String("Phi-4".into())),
        ("phi.vocab_size".into(), GgufValue::U32(100352)),
        ("phi.embedding_length".into(), GgufValue::U32(5120)),
        ("phi.block_count".into(), GgufValue::U32(40)),
        ("phi.attention.head_count".into(), GgufValue::U32(40)),
        ("phi.attention.head_count_kv".into(), GgufValue::U32(10)),
        ("phi.feed_forward_length".into(), GgufValue::U32(13824)),
        ("phi.context_length".into(), GgufValue::U32(16384)),
    ]);
    let cfg = GgufModelConfig::from_gguf_metadata(&meta).unwrap();
    assert_eq!(cfg.architecture, "phi");
    assert_eq!(cfg.model_name.as_deref(), Some("Phi-4"));
    assert_eq!(cfg.vocab_size, 100352);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_layers, 40);
    assert_eq!(cfg.num_heads, 40);
    assert_eq!(cfg.num_kv_heads, 10);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.max_seq_len, 16384);
    assert!(cfg.is_gqa());
    assert_eq!(cfg.gqa_group_size(), 4);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// ConfigError
// ===========================================================================

#[test]
fn config_error_missing_key_display() {
    let err = ConfigError::MissingKey("vocab_size".into());
    assert!(format!("{err}").contains("vocab_size"));
}

#[test]
fn config_error_invalid_value_display() {
    let err = ConfigError::InvalidValue { key: "n_heads".into(), reason: "expected u32".into() };
    let msg = format!("{err}");
    assert!(msg.contains("n_heads"));
    assert!(msg.contains("expected u32"));
}

#[test]
fn config_error_validation_display() {
    let err = ConfigError::Validation("vocab_size must be > 0".into());
    assert!(format!("{err}").contains("validation"));
}

// ===========================================================================
// MemoryEstimate
// ===========================================================================

#[test]
fn memory_estimate_serde_roundtrip() {
    let est = MemoryEstimate {
        weight_bytes: 1000,
        kv_cache_bytes: 500,
        total_bytes: 1500,
        summary: "test".into(),
    };
    let json = serde_json::to_string(&est).unwrap();
    let back: MemoryEstimate = serde_json::from_str(&json).unwrap();
    assert_eq!(est, back);
}

// ===========================================================================
// LoadConfig
// ===========================================================================

#[test]
fn load_config_default() {
    let cfg = LoadConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("LoadConfig"));
}

// ===========================================================================
// ProductionLoadConfig
// ===========================================================================

#[test]
fn production_load_config_default() {
    let cfg = ProductionLoadConfig::default();
    assert!(cfg.strict_validation);
    assert!(cfg.validate_tensor_alignment);
    assert!(cfg.max_model_size_bytes.is_some());
    assert!(!cfg.profile_memory);
}

#[test]
fn production_load_config_custom() {
    let cfg = ProductionLoadConfig {
        strict_validation: false,
        max_model_size_bytes: None,
        profile_memory: true,
        ..Default::default()
    };
    assert!(!cfg.strict_validation);
    assert!(cfg.max_model_size_bytes.is_none());
    assert!(cfg.profile_memory);
}

// ===========================================================================
// MemoryRequirements
// ===========================================================================

#[test]
fn memory_requirements_construction() {
    let req = MemoryRequirements {
        total_mb: 16000,
        gpu_memory_mb: Some(12000),
        cpu_memory_mb: 4000,
        kv_cache_mb: 2000,
        activation_mb: 500,
        headroom_mb: 1000,
    };
    assert_eq!(req.total_mb, 16000);
    assert_eq!(req.gpu_memory_mb, Some(12000));
}

// ===========================================================================
// DeviceStrategy & DeviceConfig
// ===========================================================================

#[test]
fn device_strategy_cpu_only() {
    let s = DeviceStrategy::CpuOnly;
    let dbg = format!("{s:?}");
    assert!(dbg.contains("CpuOnly"));
}

#[test]
fn device_strategy_hybrid() {
    let s = DeviceStrategy::Hybrid { cpu_layers: 10, gpu_layers: 30 };
    let dbg = format!("{s:?}");
    assert!(dbg.contains("cpu_layers: 10"));
    assert!(dbg.contains("gpu_layers: 30"));
}

#[test]
fn device_config_construction() {
    let cfg = DeviceConfig {
        strategy: Some(DeviceStrategy::GpuOnly),
        cpu_threads: Some(8),
        gpu_memory_fraction: Some(0.9),
        recommended_batch_size: 4,
    };
    assert_eq!(cfg.recommended_batch_size, 4);
    assert!(cfg.strategy.is_some());
}

// ===========================================================================
// ValidationResult
// ===========================================================================

#[test]
fn validation_result_passed() {
    let result = ValidationResult {
        passed: true,
        warnings: vec![],
        errors: vec![],
        alignment_issues: vec![],
        recommendations: vec![],
    };
    assert!(result.passed);
}

#[test]
fn validation_result_failed_with_errors() {
    let result = ValidationResult {
        passed: false,
        warnings: vec!["minor issue".into()],
        errors: vec!["fatal error".into()],
        alignment_issues: vec!["misaligned tensor".into()],
        recommendations: vec!["use mmap".into()],
    };
    assert!(!result.passed);
    assert_eq!(result.errors.len(), 1);
    assert_eq!(result.warnings.len(), 1);
}

// ===========================================================================
// ProductionModelLoader
// ===========================================================================

#[test]
fn production_loader_new() {
    let loader = ProductionModelLoader::new();
    assert!(loader.validation_enabled);
}

#[test]
fn production_loader_strict_validation() {
    let loader = ProductionModelLoader::new_with_strict_validation();
    assert!(loader.config.strict_validation);
    assert!(loader.config.validate_tensor_alignment);
}

#[test]
fn production_loader_with_custom_config() {
    let cfg = ProductionLoadConfig {
        strict_validation: false,
        profile_memory: true,
        ..Default::default()
    };
    let loader = ProductionModelLoader::with_config(cfg);
    assert!(!loader.config.strict_validation);
    assert!(loader.config.profile_memory);
}
