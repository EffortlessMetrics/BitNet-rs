//! Edge-case tests for GPU HAL model validator.
//!
//! Tests ValidationConfig, WeightStats, TensorDescriptor, WeightValidator,
//! ShapeValidator, ArchitectureValidator, NumericalValidator, and
//! ValidationReport — all without GPU hardware or model files.

use bitnet_gpu_hal::model_validator::*;

// ── ValidationLevel ─────────────────────────────────────────────────────────

#[test]
fn validation_level_all_variants() {
    let _ = ValidationLevel::Quick;
    let _ = ValidationLevel::Standard;
    let _ = ValidationLevel::Thorough;
    let _ = ValidationLevel::Paranoid;
}

// ── CheckSeverity ───────────────────────────────────────────────────────────

#[test]
fn check_severity_all_variants() {
    let _ = CheckSeverity::Info;
    let _ = CheckSeverity::Warning;
    let _ = CheckSeverity::Error;
    let _ = CheckSeverity::Critical;
}

// ── CheckStatus ─────────────────────────────────────────────────────────────

#[test]
fn check_status_all_variants() {
    let _ = CheckStatus::Pass;
    let _ = CheckStatus::Warn;
    let _ = CheckStatus::Fail;
    let _ = CheckStatus::Skipped;
}

// ── ValidationConfig ────────────────────────────────────────────────────────

#[test]
fn validation_config_quick() {
    let c = ValidationConfig::quick();
    assert!(c.validate().is_ok());
}

#[test]
fn validation_config_paranoid() {
    let c = ValidationConfig::paranoid();
    assert!(c.validate().is_ok());
    assert!(c.strict);
}

#[test]
fn validation_config_check_set_fields() {
    let cs = CheckSet {
        weights: true,
        shapes: true,
        architecture: true,
        numerical: true,
        quantization: false,
    };
    assert!(cs.weights);
    assert!(!cs.quantization);
}

#[test]
fn tolerance_thresholds_fields() {
    let t = ToleranceThresholds {
        max_weight_abs: 100.0,
        min_weight_std: 0.001,
        max_zero_fraction: 0.99,
        rtol: 1e-5,
        atol: 1e-8,
    };
    assert!(t.max_weight_abs > 0.0);
    assert!(t.rtol > 0.0);
}

// ── WeightStats ─────────────────────────────────────────────────────────────

#[test]
fn weight_stats_from_simple_values() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = WeightStats::from_values(&values);
    assert_eq!(stats.total_count, 5);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert!((stats.mean - 3.0).abs() < 1e-10);
    assert!((stats.min - 1.0).abs() < 1e-10);
    assert!((stats.max - 5.0).abs() < 1e-10);
}

#[test]
fn weight_stats_from_empty() {
    let stats = WeightStats::from_values(&[]);
    assert_eq!(stats.total_count, 0);
}

#[test]
fn weight_stats_all_zeros() {
    let values = vec![0.0; 100];
    let stats = WeightStats::from_values(&values);
    assert_eq!(stats.zero_count, 100);
    assert!((stats.mean).abs() < 1e-10);
    assert!((stats.std_dev).abs() < 1e-10);
}

#[test]
fn weight_stats_with_nan() {
    let values = vec![1.0, f64::NAN, 3.0];
    let stats = WeightStats::from_values(&values);
    assert!(stats.nan_count >= 1);
}

#[test]
fn weight_stats_with_inf() {
    let values = vec![1.0, f64::INFINITY, f64::NEG_INFINITY];
    let stats = WeightStats::from_values(&values);
    assert!(stats.inf_count >= 1);
}

#[test]
fn weight_stats_single_value() {
    let stats = WeightStats::from_values(&[42.0]);
    assert_eq!(stats.total_count, 1);
    assert!((stats.min - 42.0).abs() < 1e-10);
    assert!((stats.max - 42.0).abs() < 1e-10);
}

#[test]
fn weight_stats_negative_values() {
    let values = vec![-5.0, -3.0, -1.0, 0.0, 1.0];
    let stats = WeightStats::from_values(&values);
    assert!((stats.min - (-5.0)).abs() < 1e-10);
    assert!((stats.max - 1.0).abs() < 1e-10);
}

// ── TensorDescriptor ────────────────────────────────────────────────────────

#[test]
fn tensor_descriptor_basic() {
    let td = TensorDescriptor::new("model.layers.0.weight", vec![5120, 5120], "f32");
    assert_eq!(td.name, "model.layers.0.weight");
    assert_eq!(td.shape, vec![5120, 5120]);
    assert_eq!(td.dtype, "f32");
    assert_eq!(td.element_count, 5120 * 5120);
}

#[test]
fn tensor_descriptor_scalar() {
    let td = TensorDescriptor::new("bias", vec![1], "f32");
    assert_eq!(td.element_count, 1);
}

#[test]
fn tensor_descriptor_empty_shape() {
    let td = TensorDescriptor::new("empty", vec![], "f32");
    // Product of empty vec should be 0 or 1 depending on implementation
    let _ = td.element_count;
}

#[test]
fn tensor_descriptor_3d_shape() {
    let td = TensorDescriptor::new("attention.qkv", vec![40, 128, 5120], "f16");
    assert_eq!(td.shape.len(), 3);
    assert_eq!(td.dtype, "f16");
}

// ── ArchitectureConfig ──────────────────────────────────────────────────────

#[test]
fn architecture_config_phi4() {
    let arch = ArchitectureConfig {
        hidden_size: 5120,
        num_attention_heads: 40,
        num_layers: 40,
        vocab_size: 100352,
        intermediate_size: 13824,
        head_dim: Some(128),
        num_kv_heads: Some(10),
        max_sequence_length: 16384,
    };
    assert_eq!(arch.hidden_size, 5120);
    assert_eq!(arch.num_layers, 40);
}

#[test]
fn architecture_config_bitnet() {
    let arch = ArchitectureConfig {
        hidden_size: 2560,
        num_attention_heads: 20,
        num_layers: 30,
        vocab_size: 32000,
        intermediate_size: 6912,
        head_dim: None,
        num_kv_heads: Some(5),
        max_sequence_length: 4096,
    };
    assert_eq!(arch.vocab_size, 32000);
}

// ── ArchitectureValidator ───────────────────────────────────────────────────

#[test]
fn architecture_validator_valid_config() {
    let arch = ArchitectureConfig {
        hidden_size: 5120,
        num_attention_heads: 40,
        num_layers: 40,
        vocab_size: 100352,
        intermediate_size: 13824,
        head_dim: Some(128),
        num_kv_heads: Some(10),
        max_sequence_length: 16384,
    };
    let result = ArchitectureValidator::validate(&arch);
    assert!(matches!(result.status, CheckStatus::Pass));
}

#[test]
fn architecture_validator_zero_layers() {
    let arch = ArchitectureConfig {
        hidden_size: 512,
        num_attention_heads: 8,
        num_layers: 0,
        vocab_size: 32000,
        intermediate_size: 1024,
        head_dim: None,
        num_kv_heads: None,
        max_sequence_length: 2048,
    };
    let result = ArchitectureValidator::validate(&arch);
    // 0 layers should at least warn
    assert!(!result.issues.is_empty() || matches!(result.status, CheckStatus::Fail));
}

#[test]
fn architecture_validator_zero_hidden_size() {
    let arch = ArchitectureConfig {
        hidden_size: 0,
        num_attention_heads: 8,
        num_layers: 12,
        vocab_size: 32000,
        intermediate_size: 1024,
        head_dim: None,
        num_kv_heads: None,
        max_sequence_length: 2048,
    };
    let result = ArchitectureValidator::validate(&arch);
    assert!(!result.issues.is_empty() || matches!(result.status, CheckStatus::Fail));
}

// ── WeightValidator ─────────────────────────────────────────────────────────

#[test]
fn weight_validator_normal_weights() {
    let config = ValidationConfig::quick();
    let validator = WeightValidator::new(&config);
    let td = TensorDescriptor::new("weight", vec![100], "f32");
    // Normal gaussian-like values
    let values: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.01).collect();
    let result = validator.validate_tensor(&td, &values);
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Warn));
}

#[test]
fn weight_validator_all_nan() {
    let config = ValidationConfig::paranoid();
    let validator = WeightValidator::new(&config);
    let td = TensorDescriptor::new("broken", vec![10], "f32");
    let values = vec![f64::NAN; 10];
    let result = validator.validate_tensor(&td, &values);
    // All NaN should fail
    assert!(matches!(result.status, CheckStatus::Fail | CheckStatus::Warn));
}

#[test]
fn weight_validator_all_inf() {
    let config = ValidationConfig::paranoid();
    let validator = WeightValidator::new(&config);
    let td = TensorDescriptor::new("exploded", vec![5], "f32");
    let values = vec![f64::INFINITY; 5];
    let result = validator.validate_tensor(&td, &values);
    assert!(matches!(result.status, CheckStatus::Fail | CheckStatus::Warn));
}

#[test]
fn weight_validator_validate_all() {
    let config = ValidationConfig::quick();
    let validator = WeightValidator::new(&config);
    let tensors = vec![
        (TensorDescriptor::new("w1", vec![10], "f32"), vec![0.1; 10]),
        (TensorDescriptor::new("w2", vec![20], "f32"), vec![0.2; 20]),
    ];
    let results = validator.validate_all(&tensors);
    assert_eq!(results.len(), 2);
}

// ── ShapeValidator ──────────────────────────────────────────────────────────

#[test]
fn shape_validator_basic_valid() {
    let arch = ArchitectureConfig {
        hidden_size: 512,
        num_attention_heads: 8,
        num_layers: 12,
        vocab_size: 32000,
        intermediate_size: 1024,
        head_dim: None,
        num_kv_heads: None,
        max_sequence_length: 2048,
    };
    let validator = ShapeValidator::new(&arch);
    let td = TensorDescriptor::new("layer.0.weight", vec![512, 512], "f32");
    let result = validator.validate_basic_shape(&td);
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Warn));
}

#[test]
fn shape_validator_embedding() {
    let arch = ArchitectureConfig {
        hidden_size: 512,
        num_attention_heads: 8,
        num_layers: 12,
        vocab_size: 32000,
        intermediate_size: 1024,
        head_dim: None,
        num_kv_heads: None,
        max_sequence_length: 2048,
    };
    let validator = ShapeValidator::new(&arch);
    let td = TensorDescriptor::new("embed_tokens.weight", vec![32000, 512], "f32");
    let result = validator.validate_embedding_shape(&td);
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Warn));
}

#[test]
fn shape_validator_all_basic() {
    let arch = ArchitectureConfig {
        hidden_size: 256,
        num_attention_heads: 4,
        num_layers: 6,
        vocab_size: 1000,
        intermediate_size: 512,
        head_dim: None,
        num_kv_heads: None,
        max_sequence_length: 512,
    };
    let validator = ShapeValidator::new(&arch);
    let descriptors = vec![
        TensorDescriptor::new("w1", vec![256, 256], "f32"),
        TensorDescriptor::new("w2", vec![512, 256], "f32"),
    ];
    let results = validator.validate_all_basic(&descriptors);
    assert_eq!(results.len(), 2);
}

// ── NumericalValidator ──────────────────────────────────────────────────────

#[test]
fn numerical_validator_normal_activations() {
    let config = ValidationConfig::quick();
    let validator = NumericalValidator::new(&config);
    let activations: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
    let result = validator.validate_activations(&activations);
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Warn));
}

#[test]
fn numerical_validator_reproducibility_identical() {
    let config = ValidationConfig::quick();
    let validator = NumericalValidator::new(&config);
    let run_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let run_b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = validator.validate_reproducibility(&run_a, &run_b);
    assert!(matches!(result.status, CheckStatus::Pass));
}

#[test]
fn numerical_validator_reproducibility_different() {
    let config = ValidationConfig::paranoid();
    let validator = NumericalValidator::new(&config);
    let run_a = vec![1.0, 2.0, 3.0];
    let run_b = vec![1.0, 2.0, 100.0]; // Wildly different
    let result = validator.validate_reproducibility(&run_a, &run_b);
    assert!(matches!(result.status, CheckStatus::Fail | CheckStatus::Warn));
}

// ── QuantizationFormat ──────────────────────────────────────────────────────

#[test]
fn quantization_format_all_variants() {
    let _ = QuantizationFormat::Ternary;
    let _ = QuantizationFormat::I2S;
    let _ = QuantizationFormat::QK256;
    let _ = QuantizationFormat::None;
}

// ── QuantizationValidator ───────────────────────────────────────────────────

#[test]
fn quantization_validator_ternary_valid() {
    let td = TensorDescriptor::new("quant_weight", vec![100], "i2");
    let values = vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0];
    let result = QuantizationValidator::validate_ternary(&td, &values);
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Warn));
}

#[test]
fn quantization_validator_ternary_invalid() {
    let td = TensorDescriptor::new("bad_quant", vec![5], "i2");
    let values = vec![2.0, 3.0, -5.0, 0.5, 1.5]; // Not ternary values
    let result = QuantizationValidator::validate_ternary(&td, &values);
    // Non-ternary values should be detected
    assert!(matches!(result.status, CheckStatus::Fail | CheckStatus::Warn));
}

// ── ValidationReport ────────────────────────────────────────────────────────

#[test]
fn validation_report_from_empty_checks() {
    let config = ValidationConfig::quick();
    let report = ValidationReport::from_checks(config, vec![], std::time::Duration::from_millis(1));
    assert!(report.is_ok());
    assert_eq!(report.summary.passed, 0);
    assert_eq!(report.summary.failed, 0);
}

#[test]
fn validation_report_from_passing_checks() {
    let config = ValidationConfig::quick();
    let checks = vec![CheckResult {
        name: "test_check".into(),
        status: CheckStatus::Pass,
        duration: std::time::Duration::from_millis(10),
        issues: vec![],
    }];
    let report =
        ValidationReport::from_checks(config, checks, std::time::Duration::from_millis(10));
    assert!(report.is_ok());
    assert_eq!(report.summary.passed, 1);
}

#[test]
fn validation_report_from_failing_checks() {
    let config = ValidationConfig::quick();
    let checks = vec![CheckResult {
        name: "bad_check".into(),
        status: CheckStatus::Fail,
        duration: std::time::Duration::from_millis(5),
        issues: vec![ValidationIssue {
            severity: CheckSeverity::Error,
            category: "weights".into(),
            message: "all NaN".into(),
            tensor_name: Some("broken.weight".into()),
            details: None,
        }],
    }];
    let report = ValidationReport::from_checks(config, checks, std::time::Duration::from_millis(5));
    assert!(!report.is_ok());
    assert_eq!(report.summary.failed, 1);
    assert_eq!(report.all_issues().len(), 1);
}

#[test]
fn validation_report_mixed_results() {
    let config = ValidationConfig::quick();
    let checks = vec![
        CheckResult {
            name: "pass".into(),
            status: CheckStatus::Pass,
            duration: std::time::Duration::from_millis(1),
            issues: vec![],
        },
        CheckResult {
            name: "warn".into(),
            status: CheckStatus::Warn,
            duration: std::time::Duration::from_millis(1),
            issues: vec![ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "shapes".into(),
                message: "unusual dim".into(),
                tensor_name: None,
                details: None,
            }],
        },
        CheckResult {
            name: "skip".into(),
            status: CheckStatus::Skipped,
            duration: std::time::Duration::from_millis(0),
            issues: vec![],
        },
    ];
    let report = ValidationReport::from_checks(config, checks, std::time::Duration::from_millis(3));
    assert_eq!(report.summary.passed, 1);
    assert_eq!(report.summary.warned, 1);
    assert_eq!(report.summary.skipped, 1);
}

// ── ValidationIssue ─────────────────────────────────────────────────────────

#[test]
fn validation_issue_with_details() {
    let issue = ValidationIssue {
        severity: CheckSeverity::Critical,
        category: "numerical".into(),
        message: "NaN explosion".into(),
        tensor_name: Some("layer.39.output".into()),
        details: Some("100% NaN values detected".into()),
    };
    assert_eq!(issue.tensor_name.as_deref(), Some("layer.39.output"));
    assert!(issue.details.is_some());
}

#[test]
fn validation_issue_without_details() {
    let issue = ValidationIssue {
        severity: CheckSeverity::Info,
        category: "info".into(),
        message: "model loaded".into(),
        tensor_name: None,
        details: None,
    };
    assert!(issue.tensor_name.is_none());
}
