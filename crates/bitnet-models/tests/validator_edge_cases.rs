//! Edge-case tests for validator module: Severity, ValidationCheck, OverallStatus,
//! ValidationResult, ValidationReport, ValidationConfig, TensorInfo, TensorStats,
//! ModelInfo, ModelValidator, and the full validation pipeline.

use bitnet_models::validator::{
    ModelInfo, ModelValidator, OverallStatus, Severity, TensorInfo, TensorStats, ValidationCheck,
    ValidationConfig,
};
use std::collections::{HashMap, HashSet};

// ─── Helper ─────────────────────────────────────────────────────────

fn minimal_model() -> ModelInfo {
    ModelInfo {
        architecture: "llama".to_string(),
        vocab_size: 32000,
        hidden_size: 2560,
        num_layers: 30,
        num_heads: 20,
        intermediate_size: 6912,
        quantization_format: Some("I2_S".to_string()),
        tensors: vec![TensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![32000, 2560],
            stats: Some(TensorStats { mean: 0.0, std_dev: 0.01, min: -0.1, max: 0.1 }),
        }],
    }
}

fn bitnet_model_with_layers(num_layers: usize) -> ModelInfo {
    let mut tensors = vec![TensorInfo {
        name: "token_embd.weight".to_string(),
        shape: vec![32000, 2560],
        stats: Some(TensorStats { mean: 0.0, std_dev: 0.01, min: -0.1, max: 0.1 }),
    }];
    for i in 0..num_layers {
        tensors.push(TensorInfo {
            name: format!("blk.{}.attn_q.weight", i),
            shape: vec![2560, 2560],
            stats: Some(TensorStats { mean: 0.0, std_dev: 0.02, min: -0.2, max: 0.2 }),
        });
    }
    ModelInfo {
        architecture: "bitnet".to_string(),
        vocab_size: 32000,
        hidden_size: 2560,
        num_layers,
        num_heads: 20,
        intermediate_size: 6912,
        quantization_format: Some("I2_S".to_string()),
        tensors,
    }
}

// ─── Severity ───────────────────────────────────────────────────────

#[test]
fn severity_display() {
    assert_eq!(format!("{}", Severity::Info), "INFO");
    assert_eq!(format!("{}", Severity::Warning), "WARNING");
    assert_eq!(format!("{}", Severity::Error), "ERROR");
}

#[test]
fn severity_debug() {
    assert!(format!("{:?}", Severity::Info).contains("Info"));
    assert!(format!("{:?}", Severity::Warning).contains("Warning"));
    assert!(format!("{:?}", Severity::Error).contains("Error"));
}

#[test]
fn severity_clone_copy() {
    let s = Severity::Warning;
    let cloned = s.clone();
    let copied = s;
    assert_eq!(cloned, copied);
}

#[test]
fn severity_eq() {
    assert_eq!(Severity::Info, Severity::Info);
    assert_ne!(Severity::Info, Severity::Warning);
    assert_ne!(Severity::Warning, Severity::Error);
}

#[test]
fn severity_hash() {
    let mut set = HashSet::new();
    set.insert(Severity::Info);
    set.insert(Severity::Warning);
    set.insert(Severity::Info);
    assert_eq!(set.len(), 2);
}

#[test]
fn severity_serde_roundtrip() {
    for s in [Severity::Info, Severity::Warning, Severity::Error] {
        let json = serde_json::to_string(&s).unwrap();
        let deser: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, s);
    }
}

// ─── ValidationCheck ────────────────────────────────────────────────

#[test]
fn validation_check_display() {
    assert_eq!(format!("{}", ValidationCheck::TensorShapes), "tensor_shapes");
    assert_eq!(format!("{}", ValidationCheck::WeightDistribution), "weight_distribution");
    assert_eq!(format!("{}", ValidationCheck::LayerNormStats), "layer_norm_stats");
    assert_eq!(format!("{}", ValidationCheck::VocabSize), "vocab_size");
    assert_eq!(format!("{}", ValidationCheck::EmbeddingDim), "embedding_dim");
    assert_eq!(format!("{}", ValidationCheck::ArchitectureMatch), "architecture_match");
    assert_eq!(format!("{}", ValidationCheck::QuantizationFormat), "quantization_format");
}

#[test]
fn validation_check_serde_roundtrip() {
    let checks = [
        ValidationCheck::TensorShapes,
        ValidationCheck::WeightDistribution,
        ValidationCheck::LayerNormStats,
        ValidationCheck::VocabSize,
        ValidationCheck::EmbeddingDim,
        ValidationCheck::ArchitectureMatch,
        ValidationCheck::QuantizationFormat,
    ];
    for check in checks {
        let json = serde_json::to_string(&check).unwrap();
        let deser: ValidationCheck = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, check);
    }
}

#[test]
fn validation_check_hash() {
    let mut set = HashSet::new();
    set.insert(ValidationCheck::TensorShapes);
    set.insert(ValidationCheck::VocabSize);
    set.insert(ValidationCheck::TensorShapes);
    assert_eq!(set.len(), 2);
}

// ─── OverallStatus ──────────────────────────────────────────────────

#[test]
fn overall_status_display() {
    assert_eq!(format!("{}", OverallStatus::Passed), "PASSED");
    assert_eq!(format!("{}", OverallStatus::PassedWithWarnings), "PASSED_WITH_WARNINGS");
    assert_eq!(format!("{}", OverallStatus::Failed), "FAILED");
}

#[test]
fn overall_status_serde_roundtrip() {
    for status in [OverallStatus::Passed, OverallStatus::PassedWithWarnings, OverallStatus::Failed]
    {
        let json = serde_json::to_string(&status).unwrap();
        let deser: OverallStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, status);
    }
}

#[test]
fn overall_status_eq() {
    assert_eq!(OverallStatus::Passed, OverallStatus::Passed);
    assert_ne!(OverallStatus::Passed, OverallStatus::Failed);
}

// ─── ValidationConfig ───────────────────────────────────────────────

#[test]
fn validation_config_default() {
    let config = ValidationConfig::default();
    assert!(!config.strict_mode);
    assert!(config.skip_checks.is_empty());
    assert!(config.custom_thresholds.is_empty());
}

#[test]
fn validation_config_debug() {
    let config = ValidationConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ValidationConfig"));
}

#[test]
fn validation_config_clone() {
    let mut config = ValidationConfig::default();
    config.strict_mode = true;
    config.skip_checks.insert(ValidationCheck::VocabSize);
    let cloned = config.clone();
    assert!(cloned.strict_mode);
    assert!(cloned.skip_checks.contains(&ValidationCheck::VocabSize));
}

#[test]
fn validation_config_serde_roundtrip() {
    let mut config = ValidationConfig::default();
    config.strict_mode = true;
    config.skip_checks.insert(ValidationCheck::TensorShapes);
    config.custom_thresholds.insert("weight_mean_threshold".to_string(), 0.5);
    let json = serde_json::to_string(&config).unwrap();
    let deser: ValidationConfig = serde_json::from_str(&json).unwrap();
    assert!(deser.strict_mode);
    assert!(deser.skip_checks.contains(&ValidationCheck::TensorShapes));
    assert_eq!(deser.custom_thresholds.get("weight_mean_threshold"), Some(&0.5));
}

// ─── TensorInfo ─────────────────────────────────────────────────────

#[test]
fn tensor_info_construction() {
    let info = TensorInfo { name: "test.weight".to_string(), shape: vec![100, 200], stats: None };
    assert_eq!(info.name, "test.weight");
    assert_eq!(info.shape, vec![100, 200]);
    assert!(info.stats.is_none());
}

#[test]
fn tensor_info_with_stats() {
    let info = TensorInfo {
        name: "layer.0.weight".to_string(),
        shape: vec![512, 512],
        stats: Some(TensorStats { mean: 0.001, std_dev: 0.02, min: -0.1, max: 0.1 }),
    };
    let stats = info.stats.unwrap();
    assert!((stats.mean - 0.001).abs() < 1e-10);
    assert!((stats.std_dev - 0.02).abs() < 1e-10);
}

#[test]
fn tensor_info_clone() {
    let info = TensorInfo {
        name: "a".to_string(),
        shape: vec![1, 2, 3],
        stats: Some(TensorStats { mean: 0.0, std_dev: 1.0, min: -3.0, max: 3.0 }),
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, "a");
    assert_eq!(cloned.shape, vec![1, 2, 3]);
}

// ─── TensorStats ────────────────────────────────────────────────────

#[test]
fn tensor_stats_debug() {
    let stats = TensorStats { mean: 0.0, std_dev: 0.01, min: -1.0, max: 1.0 };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("TensorStats"));
}

#[test]
fn tensor_stats_clone() {
    let stats = TensorStats { mean: 0.5, std_dev: 0.1, min: 0.0, max: 1.0 };
    let cloned = stats.clone();
    assert_eq!(cloned.mean, 0.5);
    assert_eq!(cloned.std_dev, 0.1);
}

// ─── ModelInfo ──────────────────────────────────────────────────────

#[test]
fn model_info_construction() {
    let model = minimal_model();
    assert_eq!(model.architecture, "llama");
    assert_eq!(model.vocab_size, 32000);
    assert_eq!(model.hidden_size, 2560);
    assert_eq!(model.num_layers, 30);
    assert_eq!(model.num_heads, 20);
    assert_eq!(model.tensors.len(), 1);
}

#[test]
fn model_info_clone() {
    let model = minimal_model();
    let cloned = model.clone();
    assert_eq!(cloned.architecture, "llama");
    assert_eq!(cloned.tensors.len(), 1);
}

#[test]
fn model_info_debug() {
    let model = minimal_model();
    let debug = format!("{:?}", model);
    assert!(debug.contains("ModelInfo"));
    assert!(debug.contains("llama"));
}

// ─── ModelValidator — Defaults ──────────────────────────────────────

#[test]
fn model_validator_default() {
    let v = ModelValidator::default_validator();
    let report = v.validate_model(&minimal_model());
    // Should run all 7 checks
    assert!(report.results.len() >= 7);
}

#[test]
fn model_validator_custom_config() {
    let v = ModelValidator::new(ValidationConfig::default());
    let report = v.validate_model(&minimal_model());
    assert!(report.results.len() >= 7);
}

// ─── ModelValidator — Architecture checks ───────────────────────────

#[test]
fn validator_known_architecture_passes() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { architecture: "llama".to_string(), ..minimal_model() };
    let result = v.verify_architecture_match(&model);
    assert!(result.passed);
}

#[test]
fn validator_unknown_architecture_warns() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { architecture: "unknown_arch".to_string(), ..minimal_model() };
    let result = v.verify_architecture_match(&model);
    // Unknown architecture should produce a warning
    assert_eq!(result.severity, Severity::Warning);
}

#[test]
fn validator_bitnet_architecture_passes() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { architecture: "bitnet".to_string(), ..minimal_model() };
    let result = v.verify_architecture_match(&model);
    assert!(result.passed);
}

// ─── ModelValidator — Tensor shapes ─────────────────────────────────

#[test]
fn validator_no_tensors_fails() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { tensors: vec![], ..minimal_model() };
    let result = v.verify_tensor_shapes(&model);
    assert!(!result.passed);
    assert_eq!(result.severity, Severity::Error);
}

#[test]
fn validator_tensors_present_passes() {
    let v = ModelValidator::default_validator();
    let result = v.verify_tensor_shapes(&minimal_model());
    // With at least one tensor, check should pass or warn (not error)
    assert!(result.passed || result.severity != Severity::Error);
}

// ─── ModelValidator — Vocab size ────────────────────────────────────

#[test]
fn validator_zero_vocab_fails() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { vocab_size: 0, ..minimal_model() };
    let result = v.verify_vocab_size(&model);
    assert!(!result.passed);
}

#[test]
fn validator_normal_vocab_passes() {
    let v = ModelValidator::default_validator();
    let result = v.verify_vocab_size(&minimal_model());
    assert!(result.passed);
}

#[test]
fn validator_large_vocab_passes() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { vocab_size: 100352, ..minimal_model() };
    let result = v.verify_vocab_size(&model);
    assert!(result.passed);
}

// ─── ModelValidator — Embedding dim ─────────────────────────────────

#[test]
fn validator_zero_hidden_fails() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { hidden_size: 0, ..minimal_model() };
    let result = v.verify_embedding_dim(&model);
    assert!(!result.passed);
}

#[test]
fn validator_normal_hidden_passes() {
    let v = ModelValidator::default_validator();
    let result = v.verify_embedding_dim(&minimal_model());
    assert!(result.passed);
}

// ─── ModelValidator — Quantization format ───────────────────────────

#[test]
fn validator_known_quant_format_passes() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { quantization_format: Some("I2_S".to_string()), ..minimal_model() };
    let result = v.verify_quantization_format(&model);
    assert!(result.passed);
}

#[test]
fn validator_unknown_quant_format_warns() {
    let v = ModelValidator::default_validator();
    let model =
        ModelInfo { quantization_format: Some("UNKNOWN_FORMAT".to_string()), ..minimal_model() };
    let result = v.verify_quantization_format(&model);
    assert_eq!(result.severity, Severity::Warning);
}

#[test]
fn validator_no_quant_format_passes() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { quantization_format: None, ..minimal_model() };
    let result = v.verify_quantization_format(&model);
    // No quant format should be fine (FP16/BF16 models)
    assert!(result.passed);
}

// ─── ModelValidator — Skip checks ───────────────────────────────────

#[test]
fn validator_skip_checks() {
    let mut skip = HashSet::new();
    skip.insert(ValidationCheck::TensorShapes);
    skip.insert(ValidationCheck::VocabSize);
    let config = ValidationConfig { skip_checks: skip, ..ValidationConfig::default() };
    let v = ModelValidator::new(config);
    let report = v.validate_model(&minimal_model());
    // Should have 5 results (7 total - 2 skipped)
    assert_eq!(report.results.len(), 5);
    assert!(!report.results.iter().any(|r| r.check_name == "tensor_shapes"));
    assert!(!report.results.iter().any(|r| r.check_name == "vocab_size"));
}

#[test]
fn validator_skip_all_checks() {
    let mut skip = HashSet::new();
    skip.insert(ValidationCheck::TensorShapes);
    skip.insert(ValidationCheck::WeightDistribution);
    skip.insert(ValidationCheck::LayerNormStats);
    skip.insert(ValidationCheck::VocabSize);
    skip.insert(ValidationCheck::EmbeddingDim);
    skip.insert(ValidationCheck::ArchitectureMatch);
    skip.insert(ValidationCheck::QuantizationFormat);
    let config = ValidationConfig { skip_checks: skip, ..ValidationConfig::default() };
    let v = ModelValidator::new(config);
    let report = v.validate_model(&minimal_model());
    assert!(report.results.is_empty());
    assert_eq!(report.overall_status, OverallStatus::Passed);
}

// ─── ModelValidator — Strict mode ───────────────────────────────────

#[test]
fn validator_strict_mode_promotes_warnings() {
    let config = ValidationConfig { strict_mode: true, ..ValidationConfig::default() };
    let v = ModelValidator::new(config);
    // Unknown architecture produces warning normally, should become error in strict mode
    let model = ModelInfo { architecture: "unknown_arch".to_string(), ..minimal_model() };
    let report = v.validate_model(&model);
    // Check that warnings were promoted to errors
    let arch_result = report.results.iter().find(|r| r.check_name == "architecture_match");
    if let Some(r) = arch_result {
        assert_eq!(r.severity, Severity::Error);
        assert!(!r.passed);
    }
}

// ─── ModelValidator — Full pipeline ─────────────────────────────────

#[test]
fn validator_full_pipeline_passes() {
    let v = ModelValidator::default_validator();
    let report = v.validate_model(&minimal_model());
    // With a reasonable model, most checks should pass
    assert!(report.passed_count > 0);
    assert!(report.overall_status != OverallStatus::Failed || report.failed_count > 0);
}

#[test]
fn validator_full_pipeline_report_counts() {
    let v = ModelValidator::default_validator();
    let report = v.validate_model(&minimal_model());
    assert_eq!(report.passed_count + report.failed_count, report.results.len());
}

#[test]
fn validator_report_serde_roundtrip() {
    let v = ModelValidator::default_validator();
    let report = v.validate_model(&minimal_model());
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("results"));
    assert!(json.contains("overall_status"));
}

// ─── ModelValidator — Weight distribution ───────────────────────────

#[test]
fn validator_weight_distribution_no_stats() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo {
        tensors: vec![TensorInfo {
            name: "test.weight".to_string(),
            shape: vec![100, 100],
            stats: None,
        }],
        ..minimal_model()
    };
    let result = v.verify_weight_distribution(&model);
    // Without stats, should pass (nothing to check)
    assert!(result.passed);
}

#[test]
fn validator_weight_distribution_normal_stats() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo {
        tensors: vec![TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![2560, 2560],
            stats: Some(TensorStats { mean: 0.001, std_dev: 0.02, min: -0.1, max: 0.1 }),
        }],
        ..minimal_model()
    };
    let result = v.verify_weight_distribution(&model);
    assert!(result.passed);
}

// ─── ModelValidator — Layer norm stats ──────────────────────────────

#[test]
fn validator_layer_norm_stats_no_ln_tensors() {
    let v = ModelValidator::default_validator();
    // Model with no layernorm tensors
    let result = v.verify_layer_norm_stats(&minimal_model());
    // Should pass — nothing to check
    assert!(result.passed);
}

// ─── ModelValidator — Layer count ───────────────────────────────────

#[test]
fn validator_layer_count_normal() {
    let v = ModelValidator::default_validator();
    let model = bitnet_model_with_layers(30);
    let result = v.verify_layer_count(&model);
    assert!(result.passed);
}

#[test]
fn validator_layer_count_zero() {
    let v = ModelValidator::default_validator();
    let model = ModelInfo { num_layers: 0, ..minimal_model() };
    let result = v.verify_layer_count(&model);
    assert!(!result.passed);
}

// ─── dry_run_remap_names ────────────────────────────────────────────

#[test]
fn dry_run_remap_names_empty() {
    let unmapped = bitnet_models::dry_run_remap_names(vec![]);
    assert!(unmapped.is_empty());
}

#[test]
fn dry_run_remap_names_known_patterns() {
    let names = vec![
        "token_embd.weight".to_string(),
        "blk.0.attn_q.weight".to_string(),
        "blk.0.attn_k.weight".to_string(),
    ];
    let unmapped = bitnet_models::dry_run_remap_names(names);
    // Known GGUF patterns should be mapped, not returned as unmapped
    // (dry_run returns unmapped names only)
    for name in &unmapped {
        assert!(
            !name.starts_with("token_embd") && !name.starts_with("blk."),
            "Expected '{}' to be mapped",
            name
        );
    }
}

#[test]
fn dry_run_remap_names_unknown_pattern() {
    let names = vec!["totally_unknown_tensor_name_xyz".to_string()];
    let unmapped = bitnet_models::dry_run_remap_names(names);
    assert_eq!(unmapped.len(), 1);
    assert_eq!(unmapped[0], "totally_unknown_tensor_name_xyz");
}
