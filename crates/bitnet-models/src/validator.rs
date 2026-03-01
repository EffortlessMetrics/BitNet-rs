//! Model validation module for verifying model integrity, architecture
//! compatibility, and inference readiness.
//!
//! Provides configurable validation checks that can be run individually or as
//! a full pipeline via [`ModelValidator`]. Each check produces a
//! [`ValidationResult`] with severity and diagnostic details, aggregated into
//! a [`ValidationReport`].

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a validation result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Informational — no action needed.
    Info,
    /// Warning — model may work but results could be suboptimal.
    Warning,
    /// Error — model is likely unusable or will produce incorrect results.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationCheck
// ---------------------------------------------------------------------------

/// Enumeration of the validation checks that can be run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationCheck {
    /// Verify that tensor shapes match the declared architecture.
    TensorShapes,
    /// Check weight distribution statistics (mean ≈ 0, reasonable std‐dev).
    WeightDistribution,
    /// Verify LayerNorm weight statistics are in expected ranges.
    LayerNormStats,
    /// Ensure vocabulary size matches the tokenizer / config.
    VocabSize,
    /// Verify embedding dimension consistency.
    EmbeddingDim,
    /// Confirm the overall architecture string is recognized.
    ArchitectureMatch,
    /// Validate the quantization format metadata.
    QuantizationFormat,
}

impl fmt::Display for ValidationCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationCheck::TensorShapes => write!(f, "tensor_shapes"),
            ValidationCheck::WeightDistribution => write!(f, "weight_distribution"),
            ValidationCheck::LayerNormStats => write!(f, "layer_norm_stats"),
            ValidationCheck::VocabSize => write!(f, "vocab_size"),
            ValidationCheck::EmbeddingDim => write!(f, "embedding_dim"),
            ValidationCheck::ArchitectureMatch => write!(f, "architecture_match"),
            ValidationCheck::QuantizationFormat => write!(f, "quantization_format"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Outcome of a single validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Which check produced this result.
    pub check_name: String,
    /// `true` if the check passed.
    pub passed: bool,
    /// Severity of the finding.
    pub severity: Severity,
    /// Human‐readable summary.
    pub message: String,
    /// Optional structured details (key → value).
    pub details: HashMap<String, String>,
}

impl ValidationResult {
    fn pass(check: ValidationCheck, message: impl Into<String>) -> Self {
        Self {
            check_name: check.to_string(),
            passed: true,
            severity: Severity::Info,
            message: message.into(),
            details: HashMap::new(),
        }
    }

    fn warning(
        check: ValidationCheck,
        message: impl Into<String>,
        details: HashMap<String, String>,
    ) -> Self {
        Self {
            check_name: check.to_string(),
            passed: true,
            severity: Severity::Warning,
            message: message.into(),
            details,
        }
    }

    fn error(
        check: ValidationCheck,
        message: impl Into<String>,
        details: HashMap<String, String>,
    ) -> Self {
        Self {
            check_name: check.to_string(),
            passed: false,
            severity: Severity::Error,
            message: message.into(),
            details,
        }
    }
}

// ---------------------------------------------------------------------------
// OverallStatus
// ---------------------------------------------------------------------------

/// Aggregate pass / fail status of a full validation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverallStatus {
    /// All checks passed (possibly with info messages).
    Passed,
    /// All checks passed but some emitted warnings.
    PassedWithWarnings,
    /// One or more checks failed.
    Failed,
}

impl fmt::Display for OverallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OverallStatus::Passed => write!(f, "PASSED"),
            OverallStatus::PassedWithWarnings => write!(f, "PASSED_WITH_WARNINGS"),
            OverallStatus::Failed => write!(f, "FAILED"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Aggregated report from a full validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Every individual result.
    pub results: Vec<ValidationResult>,
    /// Number of checks that passed.
    pub passed_count: usize,
    /// Number of checks that failed.
    pub failed_count: usize,
    /// Number of checks that produced warnings.
    pub warnings_count: usize,
    /// Aggregate status.
    pub overall_status: OverallStatus,
}

impl ValidationReport {
    /// Build a report from a list of results.
    fn from_results(results: Vec<ValidationResult>) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.iter().filter(|r| !r.passed).count();
        let warnings_count = results.iter().filter(|r| r.severity == Severity::Warning).count();

        let overall_status = if failed_count > 0 {
            OverallStatus::Failed
        } else if warnings_count > 0 {
            OverallStatus::PassedWithWarnings
        } else {
            OverallStatus::Passed
        };

        Self { results, passed_count, failed_count, warnings_count, overall_status }
    }
}

// ---------------------------------------------------------------------------
// ValidationConfig
// ---------------------------------------------------------------------------

/// User‐tunable knobs for the validation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationConfig {
    /// When `true`, warnings are promoted to errors.
    pub strict_mode: bool,
    /// Checks to skip entirely.
    pub skip_checks: HashSet<ValidationCheck>,
    /// Override default thresholds (key → value).
    pub custom_thresholds: HashMap<String, f64>,
}

// `Default` is derived — all fields use their type's default (false, empty set, empty map).

// ---------------------------------------------------------------------------
// TensorInfo — lightweight descriptor used by the validator
// ---------------------------------------------------------------------------

/// Lightweight tensor descriptor consumed by validation checks.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape as a list of dimension sizes.
    pub shape: Vec<usize>,
    /// Optional pre‐computed statistics.
    pub stats: Option<TensorStats>,
}

/// Pre‐computed summary statistics for a tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

// ---------------------------------------------------------------------------
// ModelInfo — everything the validator needs to know about a model
// ---------------------------------------------------------------------------

/// Aggregated model metadata fed into the validator.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Architecture string from GGUF metadata (e.g. `"llama"`, `"bitnet"`).
    pub architecture: String,
    /// Vocabulary size from the model config.
    pub vocab_size: usize,
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of transformer layers (blocks).
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed‐forward intermediate dimension.
    pub intermediate_size: usize,
    /// Quantization format string (e.g. `"I2_S"`, `"TL1"`).
    pub quantization_format: Option<String>,
    /// Per‐tensor metadata.
    pub tensors: Vec<TensorInfo>,
}

// ---------------------------------------------------------------------------
// Default thresholds
// ---------------------------------------------------------------------------

/// Default maximum allowed absolute mean for weight tensors.
const DEFAULT_WEIGHT_MEAN_THRESHOLD: f64 = 0.1;
/// Default maximum allowed standard deviation for weight tensors.
const DEFAULT_WEIGHT_STD_MAX: f64 = 5.0;
/// Default minimum allowed standard deviation for weight tensors.
const DEFAULT_WEIGHT_STD_MIN: f64 = 1e-7;
/// Default LayerNorm gamma RMS lower bound.
const DEFAULT_LN_RMS_MIN: f64 = 0.5;
/// Default LayerNorm gamma RMS upper bound.
const DEFAULT_LN_RMS_MAX: f64 = 2.0;

/// Known / recognized architecture strings.
const KNOWN_ARCHITECTURES: &[&str] = &["llama", "bitnet", "gpt2", "mistral", "phi"];

/// Known quantization format strings.
const KNOWN_QUANT_FORMATS: &[&str] = &["I2_S", "TL1", "TL2", "IQ2_S", "Q4_0", "Q8_0"];

// ---------------------------------------------------------------------------
// ModelValidator
// ---------------------------------------------------------------------------

/// Runs configurable validation checks against a [`ModelInfo`].
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default (non‐strict) settings.
    pub fn default_validator() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Run all enabled checks and return an aggregated report.
    pub fn validate_model(&self, model: &ModelInfo) -> ValidationReport {
        let all_checks = [
            ValidationCheck::TensorShapes,
            ValidationCheck::WeightDistribution,
            ValidationCheck::LayerNormStats,
            ValidationCheck::VocabSize,
            ValidationCheck::EmbeddingDim,
            ValidationCheck::ArchitectureMatch,
            ValidationCheck::QuantizationFormat,
        ];

        let mut results = Vec::new();
        for check in &all_checks {
            if self.config.skip_checks.contains(check) {
                continue;
            }
            let mut result = self.run_check(*check, model);
            // In strict mode, promote warnings to errors.
            if self.config.strict_mode && result.severity == Severity::Warning {
                result.severity = Severity::Error;
                result.passed = false;
            }
            results.push(result);
        }

        ValidationReport::from_results(results)
    }

    // -- individual check dispatch ----------------------------------------

    fn run_check(&self, check: ValidationCheck, model: &ModelInfo) -> ValidationResult {
        match check {
            ValidationCheck::TensorShapes => self.verify_tensor_shapes(model),
            ValidationCheck::WeightDistribution => self.verify_weight_distribution(model),
            ValidationCheck::LayerNormStats => self.verify_layer_norm_stats(model),
            ValidationCheck::VocabSize => self.verify_vocab_size(model),
            ValidationCheck::EmbeddingDim => self.verify_embedding_dim(model),
            ValidationCheck::ArchitectureMatch => self.verify_architecture_match(model),
            ValidationCheck::QuantizationFormat => self.verify_quantization_format(model),
        }
    }

    // -- check implementations --------------------------------------------

    /// Verify that tensors have shapes consistent with the architecture config.
    pub fn verify_tensor_shapes(&self, model: &ModelInfo) -> ValidationResult {
        if model.tensors.is_empty() {
            return ValidationResult::error(
                ValidationCheck::TensorShapes,
                "No tensors found in model",
                HashMap::new(),
            );
        }

        let mut mismatched: Vec<String> = Vec::new();
        for t in &model.tensors {
            // Embedding / output projection: first dim should be vocab_size.
            if (t.name.contains("embed") || t.name.contains("token_embd"))
                && !t.shape.is_empty()
                && t.shape[0] != model.vocab_size
            {
                mismatched.push(format!(
                    "{}: expected dim0={}, got {}",
                    t.name, model.vocab_size, t.shape[0]
                ));
            }
            // Attention Q/K/V: should involve hidden_size.
            if (t.name.contains("attn_q") || t.name.contains("attn_k"))
                && !t.shape.is_empty()
                && t.shape.iter().all(|&d| d != model.hidden_size)
            {
                mismatched.push(format!(
                    "{}: expected hidden_size={} in shape {:?}",
                    t.name, model.hidden_size, t.shape
                ));
            }
        }

        if mismatched.is_empty() {
            ValidationResult::pass(
                ValidationCheck::TensorShapes,
                format!("All {} tensor shapes consistent", model.tensors.len()),
            )
        } else {
            let mut details = HashMap::new();
            details.insert("mismatched_tensors".into(), mismatched.join("; "));
            ValidationResult::error(
                ValidationCheck::TensorShapes,
                format!("{} tensor shape mismatch(es) found", mismatched.len()),
                details,
            )
        }
    }

    /// Check weight distribution: mean ≈ 0 and std‐dev within reasonable bounds.
    pub fn verify_weight_distribution(&self, model: &ModelInfo) -> ValidationResult {
        let mean_threshold = self
            .config
            .custom_thresholds
            .get("weight_mean_threshold")
            .copied()
            .unwrap_or(DEFAULT_WEIGHT_MEAN_THRESHOLD);
        let std_max = self
            .config
            .custom_thresholds
            .get("weight_std_max")
            .copied()
            .unwrap_or(DEFAULT_WEIGHT_STD_MAX);
        let std_min = self
            .config
            .custom_thresholds
            .get("weight_std_min")
            .copied()
            .unwrap_or(DEFAULT_WEIGHT_STD_MIN);

        // Exclude LayerNorm tensors — they have different expected distributions
        // (gamma mean ≈ 1.0, not ≈ 0.0) and are validated separately.
        let tensors_with_stats: Vec<_> = model
            .tensors
            .iter()
            .filter(|t| {
                t.stats.is_some()
                    && !t.name.contains("norm")
                    && !t.name.contains("ln")
                    && !t.name.contains("layer_norm")
            })
            .collect();

        if tensors_with_stats.is_empty() {
            return ValidationResult::pass(
                ValidationCheck::WeightDistribution,
                "No weight statistics available — skipping",
            );
        }

        let mut warnings: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        for t in &tensors_with_stats {
            let s = t.stats.as_ref().unwrap();
            if s.mean.abs() > mean_threshold {
                warnings.push(format!("{}: mean={:.6} exceeds ±{mean_threshold}", t.name, s.mean));
            }
            if s.std_dev > std_max {
                errors.push(format!("{}: std_dev={:.6} exceeds {std_max}", t.name, s.std_dev));
            }
            if s.std_dev < std_min {
                errors.push(format!(
                    "{}: std_dev={:.9} below {std_min} (degenerate)",
                    t.name, s.std_dev
                ));
            }
        }

        if !errors.is_empty() {
            let mut details = HashMap::new();
            details.insert("errors".into(), errors.join("; "));
            if !warnings.is_empty() {
                details.insert("warnings".into(), warnings.join("; "));
            }
            ValidationResult::error(
                ValidationCheck::WeightDistribution,
                format!("{} weight distribution error(s)", errors.len()),
                details,
            )
        } else if !warnings.is_empty() {
            let mut details = HashMap::new();
            details.insert("warnings".into(), warnings.join("; "));
            ValidationResult::warning(
                ValidationCheck::WeightDistribution,
                format!("{} weight distribution warning(s)", warnings.len()),
                details,
            )
        } else {
            ValidationResult::pass(
                ValidationCheck::WeightDistribution,
                format!("Weight distributions OK across {} tensors", tensors_with_stats.len()),
            )
        }
    }

    /// Verify LayerNorm gamma statistics.
    pub fn verify_layer_norm_stats(&self, model: &ModelInfo) -> ValidationResult {
        let rms_min =
            self.config.custom_thresholds.get("ln_rms_min").copied().unwrap_or(DEFAULT_LN_RMS_MIN);
        let rms_max =
            self.config.custom_thresholds.get("ln_rms_max").copied().unwrap_or(DEFAULT_LN_RMS_MAX);

        let ln_tensors: Vec<_> = model
            .tensors
            .iter()
            .filter(|t| {
                t.name.contains("norm") || t.name.contains("ln") || t.name.contains("layer_norm")
            })
            .collect();

        if ln_tensors.is_empty() {
            return ValidationResult::pass(
                ValidationCheck::LayerNormStats,
                "No LayerNorm tensors found — skipping",
            );
        }

        let mut issues: Vec<String> = Vec::new();
        for t in &ln_tensors {
            if let Some(ref s) = t.stats {
                // RMS of gamma should be close to 1.0 for well‐initialized LN.
                let rms = (s.mean * s.mean + s.std_dev * s.std_dev).sqrt();
                if rms < rms_min || rms > rms_max {
                    issues.push(format!(
                        "{}: gamma RMS={rms:.4} outside [{rms_min}, {rms_max}]",
                        t.name
                    ));
                }
            }
        }

        if issues.is_empty() {
            ValidationResult::pass(
                ValidationCheck::LayerNormStats,
                format!("LayerNorm stats OK across {} tensors", ln_tensors.len()),
            )
        } else {
            let mut details = HashMap::new();
            details.insert("issues".into(), issues.join("; "));
            ValidationResult::warning(
                ValidationCheck::LayerNormStats,
                format!("{} LayerNorm issue(s)", issues.len()),
                details,
            )
        }
    }

    /// Ensure vocabulary size is plausible.
    pub fn verify_vocab_size(&self, model: &ModelInfo) -> ValidationResult {
        if model.vocab_size == 0 {
            return ValidationResult::error(
                ValidationCheck::VocabSize,
                "Vocabulary size is 0",
                HashMap::new(),
            );
        }

        // Common vocab sizes for LLM families.
        let typical_min = 256;
        let typical_max = 256_000;

        if model.vocab_size < typical_min {
            let mut details = HashMap::new();
            details.insert("vocab_size".into(), model.vocab_size.to_string());
            details.insert("typical_min".into(), typical_min.to_string());
            return ValidationResult::warning(
                ValidationCheck::VocabSize,
                format!(
                    "Vocabulary size {} is unusually small (typical ≥ {})",
                    model.vocab_size, typical_min
                ),
                details,
            );
        }

        if model.vocab_size > typical_max {
            let mut details = HashMap::new();
            details.insert("vocab_size".into(), model.vocab_size.to_string());
            details.insert("typical_max".into(), typical_max.to_string());
            return ValidationResult::warning(
                ValidationCheck::VocabSize,
                format!(
                    "Vocabulary size {} is unusually large (typical ≤ {})",
                    model.vocab_size, typical_max
                ),
                details,
            );
        }

        ValidationResult::pass(
            ValidationCheck::VocabSize,
            format!("Vocabulary size {} is within expected range", model.vocab_size),
        )
    }

    /// Verify embedding dimension consistency.
    pub fn verify_embedding_dim(&self, model: &ModelInfo) -> ValidationResult {
        if model.hidden_size == 0 {
            return ValidationResult::error(
                ValidationCheck::EmbeddingDim,
                "Hidden size / embedding dimension is 0",
                HashMap::new(),
            );
        }

        // head_dim = hidden_size / num_heads must be integral.
        if model.num_heads > 0 && !model.hidden_size.is_multiple_of(model.num_heads) {
            let mut details = HashMap::new();
            details.insert("hidden_size".into(), model.hidden_size.to_string());
            details.insert("num_heads".into(), model.num_heads.to_string());
            return ValidationResult::error(
                ValidationCheck::EmbeddingDim,
                format!(
                    "hidden_size ({}) not divisible by num_heads ({})",
                    model.hidden_size, model.num_heads
                ),
                details,
            );
        }

        ValidationResult::pass(
            ValidationCheck::EmbeddingDim,
            format!(
                "Embedding dim {} consistent (head_dim={})",
                model.hidden_size,
                if model.num_heads > 0 { model.hidden_size / model.num_heads } else { 0 }
            ),
        )
    }

    /// Confirm the architecture string is recognized.
    pub fn verify_architecture_match(&self, model: &ModelInfo) -> ValidationResult {
        if model.architecture.is_empty() {
            return ValidationResult::error(
                ValidationCheck::ArchitectureMatch,
                "Architecture string is empty",
                HashMap::new(),
            );
        }

        let arch_lower = model.architecture.to_lowercase();
        if KNOWN_ARCHITECTURES.iter().any(|&a| a == arch_lower) {
            ValidationResult::pass(
                ValidationCheck::ArchitectureMatch,
                format!("Architecture '{}' is recognized", model.architecture),
            )
        } else {
            let mut details = HashMap::new();
            details.insert("architecture".into(), model.architecture.clone());
            details.insert("known".into(), KNOWN_ARCHITECTURES.join(", "));
            ValidationResult::warning(
                ValidationCheck::ArchitectureMatch,
                format!("Architecture '{}' is not in the known list", model.architecture),
                details,
            )
        }
    }

    /// Validate quantization format metadata.
    pub fn verify_quantization_format(&self, model: &ModelInfo) -> ValidationResult {
        match &model.quantization_format {
            None => ValidationResult::pass(
                ValidationCheck::QuantizationFormat,
                "No quantization format specified (float model assumed)",
            ),
            Some(fmt) if fmt.is_empty() => ValidationResult::error(
                ValidationCheck::QuantizationFormat,
                "Quantization format string is empty",
                HashMap::new(),
            ),
            Some(fmt) => {
                if KNOWN_QUANT_FORMATS.contains(&fmt.as_str()) {
                    ValidationResult::pass(
                        ValidationCheck::QuantizationFormat,
                        format!("Quantization format '{}' is recognized", fmt),
                    )
                } else {
                    let mut details = HashMap::new();
                    details.insert("format".into(), fmt.clone());
                    details.insert("known".into(), KNOWN_QUANT_FORMATS.join(", "));
                    ValidationResult::warning(
                        ValidationCheck::QuantizationFormat,
                        format!("Quantization format '{}' is not in the known list", fmt),
                        details,
                    )
                }
            }
        }
    }

    /// Verify that the number of block/layer tensors matches `num_layers`.
    pub fn verify_layer_count(&self, model: &ModelInfo) -> ValidationResult {
        if model.num_layers == 0 {
            return ValidationResult::error(
                ValidationCheck::TensorShapes,
                "num_layers is 0",
                HashMap::new(),
            );
        }

        // Count distinct layer indices from tensor names like `blk.{i}.*`.
        let layer_indices: HashSet<usize> = model
            .tensors
            .iter()
            .filter_map(|t| {
                let parts: Vec<&str> = t.name.split('.').collect();
                if parts.len() >= 2 && parts[0] == "blk" {
                    parts[1].parse::<usize>().ok()
                } else {
                    None
                }
            })
            .collect();

        if layer_indices.is_empty() {
            return ValidationResult::pass(
                ValidationCheck::TensorShapes,
                "No block-indexed tensors found — layer count not verifiable",
            );
        }

        if layer_indices.len() != model.num_layers {
            let mut details = HashMap::new();
            details.insert("expected_layers".into(), model.num_layers.to_string());
            details.insert("found_layers".into(), layer_indices.len().to_string());
            return ValidationResult::error(
                ValidationCheck::TensorShapes,
                format!(
                    "Expected {} layers, found {} distinct block indices",
                    model.num_layers,
                    layer_indices.len()
                ),
                details,
            );
        }

        ValidationResult::pass(
            ValidationCheck::TensorShapes,
            format!("Layer count matches: {}", model.num_layers),
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn make_tensor(name: &str, shape: Vec<usize>, stats: Option<TensorStats>) -> TensorInfo {
        TensorInfo { name: name.to_string(), shape, stats }
    }

    fn base_model() -> ModelInfo {
        ModelInfo {
            architecture: "llama".into(),
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 11008,
            quantization_format: Some("I2_S".into()),
            tensors: vec![
                make_tensor("token_embd.weight", vec![32000, 4096], None),
                make_tensor(
                    "blk.0.attn_q.weight",
                    vec![4096, 4096],
                    Some(TensorStats { mean: 0.001, std_dev: 0.02, min: -0.1, max: 0.1 }),
                ),
                make_tensor(
                    "blk.0.attn_norm.weight",
                    vec![4096],
                    Some(TensorStats { mean: 1.0, std_dev: 0.01, min: 0.99, max: 1.01 }),
                ),
            ],
        }
    }

    // -- individual check tests -------------------------------------------

    #[test]
    fn test_tensor_shapes_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_tensor_shapes(&base_model());
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);
    }

    #[test]
    fn test_tensor_shapes_mismatch() {
        let mut m = base_model();
        m.tensors[0] = make_tensor("token_embd.weight", vec![9999, 4096], None);
        let v = ModelValidator::default_validator();
        let r = v.verify_tensor_shapes(&m);
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
        assert!(r.details.contains_key("mismatched_tensors"));
    }

    #[test]
    fn test_tensor_shapes_empty_model() {
        let mut m = base_model();
        m.tensors.clear();
        let v = ModelValidator::default_validator();
        let r = v.verify_tensor_shapes(&m);
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
    }

    #[test]
    fn test_weight_distribution_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_weight_distribution(&base_model());
        assert!(r.passed);
    }

    #[test]
    fn test_weight_distribution_high_mean() {
        let mut m = base_model();
        m.tensors.push(make_tensor(
            "bad_weight",
            vec![100],
            Some(TensorStats { mean: 5.0, std_dev: 0.5, min: -1.0, max: 10.0 }),
        ));
        let v = ModelValidator::default_validator();
        let r = v.verify_weight_distribution(&m);
        assert!(r.passed); // warning, not error
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_weight_distribution_degenerate_std() {
        let mut m = base_model();
        m.tensors.push(make_tensor(
            "dead_weight",
            vec![100],
            Some(TensorStats { mean: 0.0, std_dev: 0.0, min: 0.0, max: 0.0 }),
        ));
        let v = ModelValidator::default_validator();
        let r = v.verify_weight_distribution(&m);
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
    }

    #[test]
    fn test_weight_distribution_no_stats() {
        let mut m = base_model();
        m.tensors = vec![make_tensor("w", vec![10], None)];
        let v = ModelValidator::default_validator();
        let r = v.verify_weight_distribution(&m);
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);
    }

    #[test]
    fn test_layer_norm_stats_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_layer_norm_stats(&base_model());
        assert!(r.passed);
    }

    #[test]
    fn test_layer_norm_stats_bad_rms() {
        let mut m = base_model();
        m.tensors.push(make_tensor(
            "blk.0.attn_norm.bad",
            vec![4096],
            Some(TensorStats { mean: 0.0, std_dev: 0.01, min: -0.01, max: 0.01 }),
        ));
        let v = ModelValidator::default_validator();
        let r = v.verify_layer_norm_stats(&m);
        assert!(r.passed); // warning
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_vocab_size_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_vocab_size(&base_model());
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);
    }

    #[test]
    fn test_vocab_size_zero() {
        let mut m = base_model();
        m.vocab_size = 0;
        let v = ModelValidator::default_validator();
        let r = v.verify_vocab_size(&m);
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
    }

    #[test]
    fn test_vocab_size_too_small() {
        let mut m = base_model();
        m.vocab_size = 100;
        let v = ModelValidator::default_validator();
        let r = v.verify_vocab_size(&m);
        assert!(r.passed); // warning
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_vocab_size_too_large() {
        let mut m = base_model();
        m.vocab_size = 1_000_000;
        let v = ModelValidator::default_validator();
        let r = v.verify_vocab_size(&m);
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_embedding_dim_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_embedding_dim(&base_model());
        assert!(r.passed);
    }

    #[test]
    fn test_embedding_dim_zero() {
        let mut m = base_model();
        m.hidden_size = 0;
        let v = ModelValidator::default_validator();
        let r = v.verify_embedding_dim(&m);
        assert!(!r.passed);
    }

    #[test]
    fn test_embedding_dim_not_divisible() {
        let mut m = base_model();
        m.hidden_size = 4097; // not divisible by 32 heads
        let v = ModelValidator::default_validator();
        let r = v.verify_embedding_dim(&m);
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
    }

    #[test]
    fn test_architecture_match_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_architecture_match(&base_model());
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);
    }

    #[test]
    fn test_architecture_match_unknown() {
        let mut m = base_model();
        m.architecture = "unknown_arch".into();
        let v = ModelValidator::default_validator();
        let r = v.verify_architecture_match(&m);
        assert!(r.passed); // warning
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_architecture_match_empty() {
        let mut m = base_model();
        m.architecture = String::new();
        let v = ModelValidator::default_validator();
        let r = v.verify_architecture_match(&m);
        assert!(!r.passed);
    }

    #[test]
    fn test_quantization_format_pass() {
        let v = ModelValidator::default_validator();
        let r = v.verify_quantization_format(&base_model());
        assert!(r.passed);
    }

    #[test]
    fn test_quantization_format_none() {
        let mut m = base_model();
        m.quantization_format = None;
        let v = ModelValidator::default_validator();
        let r = v.verify_quantization_format(&m);
        assert!(r.passed);
    }

    #[test]
    fn test_quantization_format_empty_string() {
        let mut m = base_model();
        m.quantization_format = Some(String::new());
        let v = ModelValidator::default_validator();
        let r = v.verify_quantization_format(&m);
        assert!(!r.passed);
    }

    #[test]
    fn test_quantization_format_unknown() {
        let mut m = base_model();
        m.quantization_format = Some("FANCY_Q99".into());
        let v = ModelValidator::default_validator();
        let r = v.verify_quantization_format(&m);
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Warning);
    }

    #[test]
    fn test_layer_count_pass() {
        let mut m = base_model();
        m.num_layers = 2;
        m.tensors = vec![
            make_tensor("blk.0.attn_q.weight", vec![4096, 4096], None),
            make_tensor("blk.1.attn_q.weight", vec![4096, 4096], None),
        ];
        let v = ModelValidator::default_validator();
        let r = v.verify_layer_count(&m);
        assert!(r.passed);
    }

    #[test]
    fn test_layer_count_mismatch() {
        let mut m = base_model();
        m.num_layers = 4;
        m.tensors = vec![
            make_tensor("blk.0.attn_q.weight", vec![4096, 4096], None),
            make_tensor("blk.1.attn_q.weight", vec![4096, 4096], None),
        ];
        let v = ModelValidator::default_validator();
        let r = v.verify_layer_count(&m);
        assert!(!r.passed);
    }

    // -- full pipeline tests ----------------------------------------------

    #[test]
    fn test_full_validation_pipeline_pass() {
        let v = ModelValidator::default_validator();
        let report = v.validate_model(&base_model());
        assert_eq!(report.overall_status, OverallStatus::Passed);
        assert_eq!(report.failed_count, 0);
        assert!(report.passed_count > 0);
    }

    #[test]
    fn test_strict_mode_promotes_warnings() {
        let mut m = base_model();
        m.architecture = "exotic_arch".into();
        let config = ValidationConfig { strict_mode: true, ..Default::default() };
        let v = ModelValidator::new(config);
        let report = v.validate_model(&m);
        // The architecture warning should have been promoted to an error.
        assert_eq!(report.overall_status, OverallStatus::Failed);
        assert!(report.failed_count > 0);
    }

    #[test]
    fn test_skip_checks() {
        let mut m = base_model();
        m.architecture = String::new(); // would fail ArchitectureMatch
        let mut skip = HashSet::new();
        skip.insert(ValidationCheck::ArchitectureMatch);
        let config = ValidationConfig { skip_checks: skip, ..Default::default() };
        let v = ModelValidator::new(config);
        let report = v.validate_model(&m);
        // ArchitectureMatch was skipped so no failure from it.
        assert!(!report.results.iter().any(|r| r.check_name == "architecture_match"));
    }

    #[test]
    fn test_warning_vs_error_severity() {
        let mut m = base_model();
        m.vocab_size = 100; // warning
        m.hidden_size = 0; // error
        let v = ModelValidator::default_validator();
        let report = v.validate_model(&m);
        assert_eq!(report.overall_status, OverallStatus::Failed);
        assert!(report.warnings_count > 0);
        assert!(report.failed_count > 0);
    }

    #[test]
    fn test_report_generation_counts() {
        let v = ModelValidator::default_validator();
        let report = v.validate_model(&base_model());
        assert_eq!(report.passed_count + report.failed_count, report.results.len());
    }

    #[test]
    fn test_empty_model_validation() {
        let m = ModelInfo {
            architecture: String::new(),
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            intermediate_size: 0,
            quantization_format: None,
            tensors: vec![],
        };
        let v = ModelValidator::default_validator();
        let report = v.validate_model(&m);
        assert_eq!(report.overall_status, OverallStatus::Failed);
        // Should have errors from multiple checks.
        assert!(report.failed_count >= 3);
    }

    #[test]
    fn test_custom_thresholds() {
        let mut m = base_model();
        m.tensors.push(make_tensor(
            "custom_w",
            vec![100],
            Some(TensorStats { mean: 0.05, std_dev: 0.5, min: -1.0, max: 1.0 }),
        ));
        // Default threshold is 0.1 — mean 0.05 is fine.
        let v = ModelValidator::default_validator();
        let r = v.verify_weight_distribution(&m);
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);

        // Tighten the threshold so 0.05 triggers a warning.
        let mut thresholds = HashMap::new();
        thresholds.insert("weight_mean_threshold".into(), 0.01);
        let config = ValidationConfig { custom_thresholds: thresholds, ..Default::default() };
        let v2 = ModelValidator::new(config);
        let r2 = v2.verify_weight_distribution(&m);
        assert_eq!(r2.severity, Severity::Warning);
    }

    #[test]
    fn test_passed_with_warnings_status() {
        let mut m = base_model();
        m.architecture = "exotic_arch".into(); // triggers warning
        let v = ModelValidator::default_validator();
        let report = v.validate_model(&m);
        assert_eq!(report.overall_status, OverallStatus::PassedWithWarnings);
    }
}
