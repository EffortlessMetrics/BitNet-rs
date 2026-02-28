//! Integration tests for model validation and numerical correctness.

use bitnet_opencl::{
    ComparisonResult, GpuDeviceCapabilities, ModelMetadata, ModelValidator, ModelWeights,
    NumericalValidator, ProjectionWeight, QuickValidator, TransformerConfig, ValidationReport,
    ValidationSeverity,
};

// ── Helpers ─────────────────────────────────────────────────────────

fn good_weights() -> ModelWeights {
    ModelWeights {
        layer_norm_weights: vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.99, 1.01, 1.0, 1.0]],
        projection_weights: vec![
            ProjectionWeight {
                name: "q_proj".into(),
                data: vec![0.01, -0.02, 0.03, -0.01],
                rows: 2,
                cols: 2,
            },
            ProjectionWeight {
                name: "k_proj".into(),
                data: vec![0.02, -0.01, 0.01, -0.03],
                rows: 2,
                cols: 2,
            },
        ],
    }
}

fn good_config() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 2048,
        num_heads: 32,
        num_kv_heads: 8,
        num_layers: 24,
        vocab_size: 32000,
        intermediate_size: 8192,
    }
}

fn small_model_meta() -> ModelMetadata {
    ModelMetadata {
        model_size_bytes: 500 * 1024 * 1024, // 500 MB
        requires_fp16: true,
        requires_fp32: false,
    }
}

fn capable_device() -> GpuDeviceCapabilities {
    GpuDeviceCapabilities {
        total_memory_bytes: 8 * 1024 * 1024 * 1024,
        available_memory_bytes: 6 * 1024 * 1024 * 1024,
        supports_fp16: true,
        supports_fp32: true,
        device_name: "Intel Arc A770".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Weight validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn valid_model_passes_all_weight_checks() {
    let v = ModelValidator::new();
    let report = v.validate_weights(&good_weights());
    assert!(report.passed(), "Expected pass: {report}");
}

#[test]
fn zero_layernorm_weights_produce_warning() {
    let mut w = good_weights();
    w.layer_norm_weights[0] = vec![0.0, 0.0, 0.0, 0.0];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(report.passed(), "Warnings should not fail the report");
    let warnings = report.warnings();
    assert!(!warnings.is_empty(), "Expected at least one warning for zero LN weights");
    assert!(
        warnings.iter().any(|w| w.message.contains("all zero")),
        "Should mention all-zero weights"
    );
}

#[test]
fn empty_layernorm_weights_produce_error() {
    let mut w = good_weights();
    w.layer_norm_weights[0] = vec![];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(!report.passed());
    assert!(report.errors().iter().any(|e| e.message.contains("empty")));
}

#[test]
fn unusual_layernorm_mean_produces_warning() {
    let mut w = good_weights();
    w.layer_norm_weights[0] = vec![5.0, 5.0, 5.0, 5.0];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(report.passed());
    assert!(!report.warnings().is_empty());
}

#[test]
fn vanishing_projection_weights_produce_warning() {
    let mut w = good_weights();
    w.projection_weights[0].data = vec![1e-8, 1e-8, 1e-8, 1e-8];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(report.passed());
    assert!(report.warnings().iter().any(|w| w.message.contains("vanishing")));
}

#[test]
fn exploding_projection_weights_produce_warning() {
    let mut w = good_weights();
    w.projection_weights[0].data = vec![999.0, 999.0, 999.0, 999.0];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(report.passed());
    assert!(report.warnings().iter().any(|w| w.message.contains("exploding")));
}

#[test]
fn empty_projection_data_produces_error() {
    let mut w = good_weights();
    w.projection_weights[0].data = vec![];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(!report.passed());
}

#[test]
fn suspiciously_uniform_layernorm_produces_warning() {
    let mut w = good_weights();
    // All identical non-zero values with zero variance
    w.layer_norm_weights[0] = vec![1.0; 128];
    let v = ModelValidator::new();
    let report = v.validate_weights(&w);
    assert!(report.passed());
    assert!(report.warnings().iter().any(|w| w.message.contains("uniform")));
}

// ═══════════════════════════════════════════════════════════════════
//  Architecture validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn valid_architecture_passes() {
    let v = ModelValidator::new();
    let report = v.validate_architecture(&good_config());
    assert!(report.passed(), "Expected pass: {report}");
}

#[test]
fn hidden_size_not_divisible_by_heads_errors() {
    let mut cfg = good_config();
    cfg.hidden_size = 2049; // Not divisible by 32
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
    assert!(report.errors().iter().any(|e| e.message.contains("not divisible")));
}

#[test]
fn kv_heads_not_dividing_attention_heads_errors() {
    let mut cfg = good_config();
    cfg.num_kv_heads = 5; // 32 % 5 != 0
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
    assert!(report.errors().iter().any(|e| e.message.contains("GQA")));
}

#[test]
fn zero_num_heads_errors() {
    let mut cfg = good_config();
    cfg.num_heads = 0;
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
}

#[test]
fn zero_kv_heads_errors() {
    let mut cfg = good_config();
    cfg.num_kv_heads = 0;
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
}

#[test]
fn zero_layers_errors() {
    let mut cfg = good_config();
    cfg.num_layers = 0;
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
}

#[test]
fn zero_vocab_errors() {
    let mut cfg = good_config();
    cfg.vocab_size = 0;
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    assert!(!report.passed());
}

#[test]
fn head_dim_reported_in_info() {
    let v = ModelValidator::new();
    let report = v.validate_architecture(&good_config());
    let expected_dim = 2048 / 32; // 64
    assert!(
        report.infos().iter().any(|i| i.message.contains(&format!("head_dim = {expected_dim}")))
    );
}

// ═══════════════════════════════════════════════════════════════════
//  GPU compatibility
// ═══════════════════════════════════════════════════════════════════

#[test]
fn model_fits_in_gpu_memory() {
    let v = ModelValidator::new();
    let report = v.validate_gpu_compatibility(&small_model_meta(), &capable_device());
    assert!(report.passed(), "Expected pass: {report}");
}

#[test]
fn model_too_large_for_gpu_errors() {
    let meta = ModelMetadata {
        model_size_bytes: 20 * 1024 * 1024 * 1024, // 20 GB
        requires_fp16: false,
        requires_fp32: true,
    };
    let v = ModelValidator::new();
    let report = v.validate_gpu_compatibility(&meta, &capable_device());
    assert!(!report.passed());
    let errors = report.errors();
    assert!(errors.iter().any(|e| e.message.contains("exceeds")));
    // Should include a suggestion
    assert!(errors.iter().any(|e| e.suggestion.is_some()));
}

#[test]
fn fp16_required_but_unsupported_errors() {
    let meta = small_model_meta(); // requires_fp16 = true
    let mut dev = capable_device();
    dev.supports_fp16 = false;
    let v = ModelValidator::new();
    let report = v.validate_gpu_compatibility(&meta, &dev);
    assert!(!report.passed());
    assert!(report.errors().iter().any(|e| e.message.contains("FP16")));
}

#[test]
fn fp32_required_but_unsupported_errors() {
    let meta = ModelMetadata {
        model_size_bytes: 100 * 1024 * 1024,
        requires_fp16: false,
        requires_fp32: true,
    };
    let mut dev = capable_device();
    dev.supports_fp32 = false;
    let v = ModelValidator::new();
    let report = v.validate_gpu_compatibility(&meta, &dev);
    assert!(!report.passed());
}

#[test]
fn high_memory_usage_produces_warning() {
    let meta = ModelMetadata {
        model_size_bytes: 5_800_000_000, // ~5.4 GB of 6 GB available
        requires_fp16: false,
        requires_fp32: true,
    };
    let v = ModelValidator::new();
    let report = v.validate_gpu_compatibility(&meta, &capable_device());
    assert!(report.passed(), "Should pass with a warning");
    assert!(!report.warnings().is_empty());
}

// ═══════════════════════════════════════════════════════════════════
//  NaN / Inf detection
// ═══════════════════════════════════════════════════════════════════

#[test]
fn clean_tensor_has_no_nan_inf() {
    assert!(!NumericalValidator::check_nan_inf(&[1.0, 2.0, 3.0]));
}

#[test]
fn nan_detected_in_tensor() {
    assert!(NumericalValidator::check_nan_inf(&[1.0, f32::NAN]));
}

#[test]
fn pos_inf_detected_in_tensor() {
    assert!(NumericalValidator::check_nan_inf(&[f32::INFINITY, 1.0]));
}

#[test]
fn neg_inf_detected_in_tensor() {
    assert!(NumericalValidator::check_nan_inf(&[f32::NEG_INFINITY, 1.0,]));
}

#[test]
fn empty_tensor_no_nan_inf() {
    assert!(!NumericalValidator::check_nan_inf(&[]));
}

// ═══════════════════════════════════════════════════════════════════
//  Distribution statistics
// ═══════════════════════════════════════════════════════════════════

#[test]
fn distribution_mean_and_range() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = NumericalValidator::check_distribution(&data);
    assert!((stats.mean - 3.0).abs() < 1e-6);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 5.0);
    assert_eq!(stats.element_count, 5);
}

#[test]
fn distribution_counts_nan_and_inf() {
    let data = vec![1.0, f32::NAN, f32::INFINITY, 2.0];
    let stats = NumericalValidator::check_distribution(&data);
    assert_eq!(stats.nan_count, 1);
    assert_eq!(stats.inf_count, 1);
    assert_eq!(stats.element_count, 4);
    // Mean computed only over finite elements
    assert!((stats.mean - 1.5).abs() < 1e-6);
}

#[test]
fn distribution_empty_tensor() {
    let stats = NumericalValidator::check_distribution(&[]);
    assert_eq!(stats.element_count, 0);
    assert_eq!(stats.mean, 0.0);
}

// ═══════════════════════════════════════════════════════════════════
//  CPU vs GPU comparison
// ═══════════════════════════════════════════════════════════════════

#[test]
fn identical_outputs_match() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = NumericalValidator::compare_outputs(&data, &data, 1e-6);
    assert!(result.matching);
    assert_eq!(result.max_diff, 0.0);
    assert_eq!(result.outlier_count, 0);
}

#[test]
fn outputs_within_tolerance_match() {
    let cpu = vec![1.0, 2.0, 3.0];
    let gpu = vec![1.00001, 2.00001, 3.00001];
    let result = NumericalValidator::compare_outputs(&cpu, &gpu, 1e-4);
    assert!(result.matching);
}

#[test]
fn outputs_beyond_tolerance_mismatch() {
    let cpu = vec![1.0, 2.0, 3.0];
    let gpu = vec![1.0, 2.5, 3.0];
    let result = NumericalValidator::compare_outputs(&cpu, &gpu, 0.1);
    assert!(!result.matching);
    assert_eq!(result.outlier_count, 1);
    assert!((result.max_diff - 0.5).abs() < 1e-6);
}

#[test]
fn empty_outputs_match() {
    let result = NumericalValidator::compare_outputs(&[], &[], 1e-5);
    assert!(result.matching);
    assert_eq!(result.element_count, 0);
}

// ═══════════════════════════════════════════════════════════════════
//  Divergence detection
// ═══════════════════════════════════════════════════════════════════

#[test]
fn stable_sequence_no_divergence() {
    let v = NumericalValidator::new();
    let history = vec![vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], vec![1.0, 1.9, 3.0]];
    assert!(v.detect_divergence(&history).is_none());
}

#[test]
fn diverging_sequence_detected() {
    let v = NumericalValidator { divergence_threshold: 5.0, ..NumericalValidator::new() };
    // Step 0: small std dev; Step 3: huge std dev
    let history = vec![
        vec![1.0, 1.1, 0.9, 1.0],
        vec![1.0, 1.2, 0.8, 1.0],
        vec![1.0, 1.3, 0.7, 1.0],
        vec![1.0, 100.0, -100.0, 1.0], // explosion
    ];
    let d = v.detect_divergence(&history);
    assert!(d.is_some(), "Should detect divergence");
    let point = d.unwrap();
    assert_eq!(point.step, 3);
}

#[test]
fn single_snapshot_no_divergence() {
    let v = NumericalValidator::new();
    let history = vec![vec![1.0, 2.0, 3.0]];
    assert!(v.detect_divergence(&history).is_none());
}

// ═══════════════════════════════════════════════════════════════════
//  QuickValidator
// ═══════════════════════════════════════════════════════════════════

#[test]
fn quick_validator_passes_valid_model() {
    let report = QuickValidator::validate(&good_config(), &small_model_meta(), &capable_device());
    assert!(report.passed(), "Expected pass: {report}");
}

#[test]
fn quick_validator_catches_bad_architecture() {
    let mut cfg = good_config();
    cfg.hidden_size = 100; // Not divisible by 32 heads
    let report = QuickValidator::validate(&cfg, &small_model_meta(), &capable_device());
    assert!(!report.passed());
}

#[test]
fn quick_validator_catches_memory_overflow() {
    let meta = ModelMetadata {
        model_size_bytes: 20 * 1024 * 1024 * 1024,
        requires_fp16: false,
        requires_fp32: true,
    };
    let report = QuickValidator::validate(&good_config(), &meta, &capable_device());
    assert!(!report.passed());
}

// ═══════════════════════════════════════════════════════════════════
//  Report formatting
// ═══════════════════════════════════════════════════════════════════

#[test]
fn report_display_includes_status() {
    let v = ModelValidator::new();
    let report = v.validate_architecture(&good_config());
    let text = format!("{report}");
    assert!(text.contains("PASS"));
}

#[test]
fn failed_report_display_says_fail() {
    let mut cfg = good_config();
    cfg.num_heads = 0;
    let v = ModelValidator::new();
    let report = v.validate_architecture(&cfg);
    let text = format!("{report}");
    assert!(text.contains("FAIL"));
}

#[test]
fn report_merge_combines_findings() {
    let mut r1 = ValidationReport::new();
    r1.add(ValidationSeverity::Info, "first", None);
    let mut r2 = ValidationReport::new();
    r2.add(ValidationSeverity::Warning, "second", None);
    r1.merge(r2);
    assert_eq!(r1.findings.len(), 2);
}

#[test]
fn severity_ordering() {
    assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
    assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
}

#[test]
fn comparison_result_display() {
    let result = ComparisonResult {
        matching: true,
        max_diff: 0.0001,
        mean_diff: 0.00005,
        outlier_count: 0,
        element_count: 100,
    };
    let text = format!("{result}");
    assert!(text.contains("MATCH"));
}

#[test]
fn finding_with_suggestion_displays_it() {
    let mut report = ValidationReport::new();
    report.add(ValidationSeverity::Error, "bad thing", Some("fix it".into()));
    let text = format!("{}", report.findings[0]);
    assert!(text.contains("suggestion: fix it"));
}
