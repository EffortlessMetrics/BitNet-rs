//! Model validation suite for verifying loaded models are correct and complete.
//!
//! Provides tensor-level defect detection that complements the check-level
//! validation in [`crate::validator`]. Each function collects concrete
//! [`ValidationIssue`] variants describing exactly what is wrong, and
//! [`full_validation`] aggregates them into a [`ValidationReport`].

use std::collections::HashSet;
use std::fmt;

use crate::validator::ModelInfo;

// ---------------------------------------------------------------------------
// ValidationIssue
// ---------------------------------------------------------------------------

/// A concrete defect found during model validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationIssue {
    /// A tensor expected by the architecture is missing.
    MissingTensor { name: String, expected_shape: Vec<usize> },
    /// A tensor's shape does not match the expected shape.
    ShapeMismatch { name: String, expected: Vec<usize>, actual: Vec<usize> },
    /// A tensor contains NaN values.
    NanWeights { tensor_name: String, nan_count: usize },
    /// A tensor contains infinite values.
    InfWeights { tensor_name: String, inf_count: usize },
    /// A tensor is entirely zero (degenerate).
    ZeroTensor { tensor_name: String },
    /// A norm tensor has suspicious statistics (far from expected ~1.0).
    SuspiciousNorm { tensor_name: String, mean: f64, std: f64 },
    /// A layer is missing its LayerNorm/RMSNorm weights.
    MissingLayerNormWeights { layer: usize },
    /// The number of layers found does not match the config.
    InconsistentLayerCount { expected: usize, found: usize },
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor { name, expected_shape } => {
                write!(f, "missing tensor '{name}' (expected shape {expected_shape:?})")
            }
            Self::ShapeMismatch { name, expected, actual } => {
                write!(
                    f,
                    "shape mismatch for '{name}': expected {expected:?}, got {actual:?}"
                )
            }
            Self::NanWeights { tensor_name, nan_count } => {
                write!(f, "tensor '{tensor_name}' contains {nan_count} NaN value(s)")
            }
            Self::InfWeights { tensor_name, inf_count } => {
                write!(f, "tensor '{tensor_name}' contains {inf_count} Inf value(s)")
            }
            Self::ZeroTensor { tensor_name } => {
                write!(f, "tensor '{tensor_name}' is entirely zero")
            }
            Self::SuspiciousNorm { tensor_name, mean, std } => {
                write!(
                    f,
                    "norm tensor '{tensor_name}' has suspicious stats \
                     (mean={mean:.6}, std={std:.6})"
                )
            }
            Self::MissingLayerNormWeights { layer } => {
                write!(f, "layer {layer} is missing LayerNorm weights")
            }
            Self::InconsistentLayerCount { expected, found } => {
                write!(f, "expected {expected} layers, found {found}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Comprehensive model validation result.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Architecture string from the model metadata.
    pub architecture: String,
    /// Total parameter count across all tensors.
    pub total_parameters: u64,
    /// Number of transformer layers declared in config.
    pub layer_count: usize,
    /// Vocabulary size declared in config.
    pub vocab_size: usize,
    /// All issues discovered during validation.
    pub issues: Vec<ValidationIssue>,
    /// `true` when no issues were found.
    pub passed: bool,
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Validation Report")?;
        writeln!(f, "  architecture:      {}", self.architecture)?;
        writeln!(f, "  total_parameters:  {}", self.total_parameters)?;
        writeln!(f, "  layer_count:       {}", self.layer_count)?;
        writeln!(f, "  vocab_size:        {}", self.vocab_size)?;
        writeln!(f, "  passed:            {}", self.passed)?;
        if !self.issues.is_empty() {
            writeln!(f, "  issues ({}):", self.issues.len())?;
            for issue in &self.issues {
                writeln!(f, "    - {issue}")?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Expected tensor patterns for a standard transformer block
// ---------------------------------------------------------------------------

/// Suffixes that every transformer block is expected to have.
const BLOCK_TENSOR_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Norm tensor suffixes expected per block.
const BLOCK_NORM_SUFFIXES: &[&str] = &["attn_norm.weight", "ffn_norm.weight"];

// ---------------------------------------------------------------------------
// Validation functions
// ---------------------------------------------------------------------------

/// Check that all tensors have the shapes expected by the architecture config.
///
/// Verifies:
/// - Embedding tensor first dim == `vocab_size`
/// - Attention Q/K tensors contain `hidden_size` in their shape
/// - Output projection first dim == `vocab_size`
pub fn validate_tensor_shapes(model: &ModelInfo) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    for t in &model.tensors {
        // Embedding: dim-0 should be vocab_size
        if t.name.contains("token_embd")
            && !t.shape.is_empty()
            && t.shape[0] != model.vocab_size
        {
            issues.push(ValidationIssue::ShapeMismatch {
                name: t.name.clone(),
                expected: vec![model.vocab_size],
                actual: vec![t.shape[0]],
            });
        }

        // Final output projection (not attn_output): dim-0 should be vocab_size
        if t.name == "output.weight"
            && !t.shape.is_empty()
            && t.shape[0] != model.vocab_size
        {
            issues.push(ValidationIssue::ShapeMismatch {
                name: t.name.clone(),
                expected: vec![model.vocab_size],
                actual: vec![t.shape[0]],
            });
        }

        // Attention Q/K: should reference hidden_size
        if (t.name.contains("attn_q") || t.name.contains("attn_k"))
            && !t.shape.is_empty()
            && t.shape.iter().all(|&d| d != model.hidden_size)
        {
            issues.push(ValidationIssue::ShapeMismatch {
                name: t.name.clone(),
                expected: vec![model.hidden_size, model.hidden_size],
                actual: t.shape.clone(),
            });
        }

        // FFN gate/up: should reference intermediate_size
        if (t.name.contains("ffn_gate") || t.name.contains("ffn_up"))
            && !t.shape.is_empty()
            && model.intermediate_size > 0
            && t.shape.iter().all(|&d| d != model.intermediate_size)
        {
            issues.push(ValidationIssue::ShapeMismatch {
                name: t.name.clone(),
                expected: vec![model.intermediate_size],
                actual: t.shape.clone(),
            });
        }
    }

    issues
}

/// Check tensor values for NaN, Inf, and all-zero tensors.
///
/// Uses the pre-computed [`TensorStats`](crate::validator::TensorStats) when
/// available. A tensor with `std_dev == 0.0` *and* `mean == 0.0` is flagged
/// as all-zero. NaN/Inf are detected via `f64::is_nan` / `f64::is_infinite`
/// on the summary statistics.
pub fn validate_tensor_values(model: &ModelInfo) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    for t in &model.tensors {
        if let Some(ref s) = t.stats {
            // NaN detection — if any stat is NaN, the tensor likely contains NaN.
            let nan_signals =
                [s.mean, s.std_dev, s.min, s.max].iter().filter(|v| v.is_nan()).count();
            if nan_signals > 0 {
                issues.push(ValidationIssue::NanWeights {
                    tensor_name: t.name.clone(),
                    nan_count: nan_signals,
                });
            }

            // Inf detection.
            let inf_signals = [s.mean, s.std_dev, s.min, s.max]
                .iter()
                .filter(|v| v.is_infinite())
                .count();
            if inf_signals > 0 {
                issues.push(ValidationIssue::InfWeights {
                    tensor_name: t.name.clone(),
                    inf_count: inf_signals,
                });
            }

            // All-zero detection.
            if s.mean == 0.0 && s.std_dev == 0.0 && s.min == 0.0 && s.max == 0.0 {
                issues.push(ValidationIssue::ZeroTensor { tensor_name: t.name.clone() });
            }
        }
    }

    issues
}

/// Verify that every expected layer has its required tensors.
///
/// For each layer index `0..num_layers`, checks that the standard block
/// tensor suffixes and norm suffixes are present.
pub fn validate_layer_completeness(model: &ModelInfo) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    if model.num_layers == 0 {
        return issues;
    }

    // Build a set of tensor names for fast lookup.
    let names: HashSet<&str> = model.tensors.iter().map(|t| t.name.as_str()).collect();

    // Count distinct block indices to check overall layer count.
    let layer_indices: HashSet<usize> = model
        .tensors
        .iter()
        .filter_map(|t| parse_block_index(&t.name))
        .collect();

    if !layer_indices.is_empty() && layer_indices.len() != model.num_layers {
        issues.push(ValidationIssue::InconsistentLayerCount {
            expected: model.num_layers,
            found: layer_indices.len(),
        });
    }

    // Per-layer checks.
    for layer in 0..model.num_layers {
        for suffix in BLOCK_TENSOR_SUFFIXES {
            let expected_name = format!("blk.{layer}.{suffix}");
            if !names.contains(expected_name.as_str()) {
                let expected_shape = expected_shape_for_suffix(suffix, model);
                issues.push(ValidationIssue::MissingTensor {
                    name: expected_name,
                    expected_shape,
                });
            }
        }
    }

    issues
}

/// Check that LayerNorm / RMSNorm weights have reasonable values (~1.0).
///
/// For well-initialized models the norm gamma weights should have a mean
/// close to 1.0 and small standard deviation. This function flags tensors
/// whose mean deviates from 1.0 by more than 0.5 or whose std exceeds 0.5.
pub fn validate_norm_weights(model: &ModelInfo) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    let names: HashSet<&str> = model.tensors.iter().map(|t| t.name.as_str()).collect();

    // Check per-layer norm presence.
    for layer in 0..model.num_layers {
        let has_norm = BLOCK_NORM_SUFFIXES
            .iter()
            .any(|s| names.contains(format!("blk.{layer}.{s}").as_str()));
        if !has_norm {
            issues.push(ValidationIssue::MissingLayerNormWeights { layer });
        }
    }

    // Check norm tensor statistics.
    for t in &model.tensors {
        if !is_norm_tensor(&t.name) {
            continue;
        }
        if let Some(ref s) = t.stats {
            let mean_off = (s.mean - 1.0).abs();
            if mean_off > 0.5 || s.std_dev > 0.5 {
                issues.push(ValidationIssue::SuspiciousNorm {
                    tensor_name: t.name.clone(),
                    mean: s.mean,
                    std: s.std_dev,
                });
            }
        }
    }

    issues
}

/// Run all validation checks and produce an aggregated [`ValidationReport`].
pub fn full_validation(model: &ModelInfo) -> ValidationReport {
    let mut issues = Vec::new();
    issues.extend(validate_tensor_shapes(model));
    issues.extend(validate_tensor_values(model));
    issues.extend(validate_layer_completeness(model));
    issues.extend(validate_norm_weights(model));

    let total_parameters: u64 = model
        .tensors
        .iter()
        .map(|t| t.shape.iter().copied().product::<usize>() as u64)
        .sum();

    let passed = issues.is_empty();

    ValidationReport {
        architecture: model.architecture.clone(),
        total_parameters,
        layer_count: model.num_layers,
        vocab_size: model.vocab_size,
        issues,
        passed,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the block index from a tensor name like `blk.3.attn_q.weight`.
fn parse_block_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && parts[0] == "blk" { parts[1].parse().ok() } else { None }
}

/// Return `true` if the tensor name indicates a normalisation weight.
fn is_norm_tensor(name: &str) -> bool {
    name.contains("norm") || name.contains("ln") || name.contains("layer_norm")
}

/// Heuristic expected shape for a block tensor suffix.
fn expected_shape_for_suffix(suffix: &str, model: &ModelInfo) -> Vec<usize> {
    match suffix {
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight" => {
            vec![model.hidden_size, model.hidden_size]
        }
        "ffn_gate.weight" | "ffn_up.weight" => {
            vec![model.intermediate_size, model.hidden_size]
        }
        "ffn_down.weight" => vec![model.hidden_size, model.intermediate_size],
        _ => vec![],
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator::{TensorInfo, TensorStats};

    // -- helpers ----------------------------------------------------------

    fn tensor(name: &str, shape: Vec<usize>, stats: Option<TensorStats>) -> TensorInfo {
        TensorInfo { name: name.to_string(), shape, stats }
    }

    /// A minimal well-formed 2-layer model.
    fn good_model() -> ModelInfo {
        let mut tensors = vec![tensor("token_embd.weight", vec![32000, 256], None)];

        for i in 0..2 {
            for suffix in BLOCK_TENSOR_SUFFIXES {
                let shape = match *suffix {
                    "ffn_gate.weight" | "ffn_up.weight" => vec![512, 256],
                    "ffn_down.weight" => vec![256, 512],
                    _ => vec![256, 256],
                };
                tensors.push(tensor(&format!("blk.{i}.{suffix}"), shape, None));
            }
            for suffix in BLOCK_NORM_SUFFIXES {
                tensors.push(tensor(
                    &format!("blk.{i}.{suffix}"),
                    vec![256],
                    Some(TensorStats { mean: 1.0, std_dev: 0.01, min: 0.98, max: 1.02 }),
                ));
            }
        }

        ModelInfo {
            architecture: "llama".into(),
            vocab_size: 32000,
            hidden_size: 256,
            num_layers: 2,
            num_heads: 8,
            intermediate_size: 512,
            quantization_format: Some("I2_S".into()),
            tensors,
        }
    }

    // -- validate_tensor_shapes -------------------------------------------

    #[test]
    fn tensor_shapes_pass_for_good_model() {
        let issues = validate_tensor_shapes(&good_model());
        assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
    }

    #[test]
    fn tensor_shapes_detect_bad_embedding() {
        let mut m = good_model();
        m.tensors[0] = tensor("token_embd.weight", vec![9999, 256], None);
        let issues = validate_tensor_shapes(&m);
        assert_eq!(issues.len(), 1);
        assert!(matches!(&issues[0], ValidationIssue::ShapeMismatch { name, .. } if name == "token_embd.weight"));
    }

    #[test]
    fn tensor_shapes_detect_bad_attn_hidden_size() {
        let mut m = good_model();
        // Replace attn_q with wrong hidden dim.
        let idx = m
            .tensors
            .iter()
            .position(|t| t.name == "blk.0.attn_q.weight")
            .unwrap();
        m.tensors[idx] = tensor("blk.0.attn_q.weight", vec![128, 128], None);
        let issues = validate_tensor_shapes(&m);
        assert!(!issues.is_empty());
        assert!(matches!(&issues[0], ValidationIssue::ShapeMismatch { name, .. } if name == "blk.0.attn_q.weight"));
    }

    #[test]
    fn tensor_shapes_detect_bad_ffn_intermediate() {
        let mut m = good_model();
        let idx = m
            .tensors
            .iter()
            .position(|t| t.name == "blk.0.ffn_gate.weight")
            .unwrap();
        m.tensors[idx] = tensor("blk.0.ffn_gate.weight", vec![999, 256], None);
        let issues = validate_tensor_shapes(&m);
        assert!(!issues.is_empty());
    }

    // -- validate_tensor_values -------------------------------------------

    #[test]
    fn tensor_values_pass_for_good_model() {
        let issues = validate_tensor_values(&good_model());
        assert!(issues.is_empty());
    }

    #[test]
    fn tensor_values_detect_nan() {
        let mut m = good_model();
        m.tensors.push(tensor(
            "bad_nan",
            vec![10],
            Some(TensorStats { mean: f64::NAN, std_dev: 0.1, min: 0.0, max: 1.0 }),
        ));
        let issues = validate_tensor_values(&m);
        assert_eq!(issues.len(), 1);
        assert!(
            matches!(&issues[0], ValidationIssue::NanWeights { tensor_name, nan_count } if tensor_name == "bad_nan" && *nan_count == 1)
        );
    }

    #[test]
    fn tensor_values_detect_inf() {
        let mut m = good_model();
        m.tensors.push(tensor(
            "bad_inf",
            vec![10],
            Some(TensorStats {
                mean: 0.0,
                std_dev: f64::INFINITY,
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            }),
        ));
        let issues = validate_tensor_values(&m);
        assert_eq!(issues.len(), 1);
        assert!(
            matches!(&issues[0], ValidationIssue::InfWeights { tensor_name, inf_count } if tensor_name == "bad_inf" && *inf_count == 3)
        );
    }

    #[test]
    fn tensor_values_detect_all_zero() {
        let mut m = good_model();
        m.tensors.push(tensor(
            "dead_tensor",
            vec![10],
            Some(TensorStats { mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0 }),
        ));
        let issues = validate_tensor_values(&m);
        assert_eq!(issues.len(), 1);
        assert!(
            matches!(&issues[0], ValidationIssue::ZeroTensor { tensor_name } if tensor_name == "dead_tensor")
        );
    }

    #[test]
    fn tensor_values_skip_without_stats() {
        let m = ModelInfo {
            tensors: vec![tensor("no_stats", vec![10], None)],
            ..good_model()
        };
        let issues = validate_tensor_values(&m);
        assert!(issues.is_empty());
    }

    // -- validate_layer_completeness --------------------------------------

    #[test]
    fn layer_completeness_pass_for_good_model() {
        let issues = validate_layer_completeness(&good_model());
        assert!(issues.is_empty(), "got: {issues:?}");
    }

    #[test]
    fn layer_completeness_detect_missing_tensor() {
        let mut m = good_model();
        m.tensors.retain(|t| t.name != "blk.1.ffn_down.weight");
        let issues = validate_layer_completeness(&m);
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| matches!(
            i,
            ValidationIssue::MissingTensor { name, .. } if name == "blk.1.ffn_down.weight"
        )));
    }

    #[test]
    fn layer_completeness_detect_inconsistent_layer_count() {
        let mut m = good_model();
        m.num_layers = 4; // declared 4 but only 2 block indices exist
        let issues = validate_layer_completeness(&m);
        assert!(issues.iter().any(|i| matches!(
            i,
            ValidationIssue::InconsistentLayerCount { expected: 4, found: 2 }
        )));
    }

    #[test]
    fn layer_completeness_zero_layers_is_fine() {
        let mut m = good_model();
        m.num_layers = 0;
        let issues = validate_layer_completeness(&m);
        assert!(issues.is_empty());
    }

    // -- validate_norm_weights --------------------------------------------

    #[test]
    fn norm_weights_pass_for_good_model() {
        let issues = validate_norm_weights(&good_model());
        assert!(issues.is_empty(), "got: {issues:?}");
    }

    #[test]
    fn norm_weights_detect_suspicious_mean() {
        let mut m = good_model();
        let idx = m
            .tensors
            .iter()
            .position(|t| t.name == "blk.0.attn_norm.weight")
            .unwrap();
        m.tensors[idx] = tensor(
            "blk.0.attn_norm.weight",
            vec![256],
            Some(TensorStats { mean: 0.0, std_dev: 0.01, min: -0.01, max: 0.01 }),
        );
        let issues = validate_norm_weights(&m);
        assert!(issues.iter().any(|i| matches!(
            i,
            ValidationIssue::SuspiciousNorm { tensor_name, .. } if tensor_name == "blk.0.attn_norm.weight"
        )));
    }

    #[test]
    fn norm_weights_detect_missing_layer_norm() {
        let mut m = good_model();
        // Remove all norm tensors for layer 1.
        m.tensors.retain(|t| !(t.name.starts_with("blk.1.") && t.name.contains("norm")));
        let issues = validate_norm_weights(&m);
        assert!(issues.iter().any(|i| matches!(
            i,
            ValidationIssue::MissingLayerNormWeights { layer: 1 }
        )));
    }

    #[test]
    fn norm_weights_high_std() {
        let mut m = good_model();
        m.tensors.push(tensor(
            "output_norm.weight",
            vec![256],
            Some(TensorStats { mean: 1.0, std_dev: 1.5, min: -1.0, max: 3.0 }),
        ));
        let issues = validate_norm_weights(&m);
        assert!(issues.iter().any(|i| matches!(
            i,
            ValidationIssue::SuspiciousNorm { tensor_name, .. } if tensor_name == "output_norm.weight"
        )));
    }

    // -- full_validation --------------------------------------------------

    #[test]
    fn full_validation_passes_for_good_model() {
        let report = full_validation(&good_model());
        assert!(report.passed, "report:\n{report}");
        assert!(report.issues.is_empty());
        assert_eq!(report.architecture, "llama");
        assert_eq!(report.layer_count, 2);
        assert_eq!(report.vocab_size, 32000);
        assert!(report.total_parameters > 0);
    }

    #[test]
    fn full_validation_detects_multiple_defects() {
        let mut m = good_model();
        // Introduce several defects.
        m.tensors[0] = tensor("token_embd.weight", vec![9999, 256], None); // bad shape
        m.tensors.push(tensor(
            "bad_nan",
            vec![10],
            Some(TensorStats { mean: f64::NAN, std_dev: 0.1, min: 0.0, max: 1.0 }),
        ));
        m.tensors.retain(|t| t.name != "blk.1.ffn_gate.weight"); // missing tensor

        let report = full_validation(&m);
        assert!(!report.passed);
        assert!(report.issues.len() >= 3, "expected >=3 issues, got {}", report.issues.len());
    }

    #[test]
    fn full_validation_empty_model() {
        let m = ModelInfo {
            architecture: "llama".into(),
            vocab_size: 32000,
            hidden_size: 256,
            num_layers: 2,
            num_heads: 8,
            intermediate_size: 512,
            quantization_format: None,
            tensors: vec![],
        };
        let report = full_validation(&m);
        assert!(!report.passed);
        // Should detect missing tensors for both layers.
        assert!(!report.issues.is_empty());
        assert_eq!(report.total_parameters, 0);
    }

    #[test]
    fn full_validation_report_display() {
        let report = full_validation(&good_model());
        let text = format!("{report}");
        assert!(text.contains("Model Validation Report"));
        assert!(text.contains("passed:"));
        assert!(text.contains("true"));
    }

    #[test]
    fn full_validation_report_display_with_issues() {
        let mut m = good_model();
        m.tensors[0] = tensor("token_embd.weight", vec![9999, 256], None);
        let report = full_validation(&m);
        let text = format!("{report}");
        assert!(text.contains("issues ("));
    }

    #[test]
    fn validation_issue_display() {
        let issue = ValidationIssue::MissingTensor {
            name: "blk.0.attn_q.weight".into(),
            expected_shape: vec![256, 256],
        };
        let text = format!("{issue}");
        assert!(text.contains("missing tensor"));
        assert!(text.contains("blk.0.attn_q.weight"));
    }

    #[test]
    fn total_parameters_calculated_correctly() {
        let m = ModelInfo {
            architecture: "test".into(),
            vocab_size: 100,
            hidden_size: 10,
            num_layers: 0,
            num_heads: 1,
            intermediate_size: 20,
            quantization_format: None,
            tensors: vec![
                tensor("a", vec![10, 20], None),  // 200
                tensor("b", vec![5, 5, 5], None), // 125
            ],
        };
        let report = full_validation(&m);
        assert_eq!(report.total_parameters, 325);
    }
}
