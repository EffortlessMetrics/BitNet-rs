//! Module stub - implementation pending merge from feature branch
//! Model validation with weight, shape, architecture, and numerical checks.
//!
//! Provides a layered validation pipeline that inspects a loaded model for
//! weight integrity (NaN/Inf, distribution, range), shape consistency,
//! architecture invariants, numerical stability, and quantization formatting.
//! Results are aggregated into a [`ValidationReport`].

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Severity & Status ───────────────────────────────────────────────────────

/// Severity of a single check result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CheckSeverity {
    /// Informational – no action needed.
    Info,
    /// May indicate a problem but does not block inference.
    Warning,
    /// A definite problem that will likely cause incorrect results.
    Error,
    /// A critical issue that prevents inference entirely.
    Critical,
}

impl fmt::Display for CheckSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Outcome of a single validation check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckStatus {
    /// The check passed.
    Pass,
    /// The check passed with warnings.
    Warn,
    /// The check failed.
    Fail,
    /// The check was skipped.
    Skipped,
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Warn => write!(f, "WARN"),
            Self::Fail => write!(f, "FAIL"),
            Self::Skipped => write!(f, "SKIPPED"),
        }
    }
}

// ── Validation Level ────────────────────────────────────────────────────────

/// Controls the thoroughness (and cost) of a validation run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Fast spot checks – suitable for CI.
    Quick,
    /// Default level – covers all categories but samples tensors.
    #[default]
    Standard,
    /// Full scan of every tensor element.
    Thorough,
    /// Exhaustive: full scan, numerical stability, cross-comparisons.
    Paranoid,
}

impl fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quick => write!(f, "Quick"),
            Self::Standard => write!(f, "Standard"),
            Self::Thorough => write!(f, "Thorough"),
            Self::Paranoid => write!(f, "Paranoid"),
        }
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Which categories of checks to run.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct CheckSet {
    pub weights: bool,
    pub shapes: bool,
    pub architecture: bool,
    pub numerical: bool,
    pub quantization: bool,
}

impl Default for CheckSet {
    fn default() -> Self {
        Self {
            weights: true,
            shapes: true,
            architecture: true,
            numerical: true,
            quantization: true,
        }
    }
}

/// Tolerance thresholds for numerical checks.
#[derive(Debug, Clone)]
pub struct ToleranceThresholds {
    /// Maximum absolute value allowed for any weight element.
    pub max_weight_abs: f64,
    /// Minimum expected standard deviation (detects dead layers).
    pub min_weight_std: f64,
    /// Maximum fraction of zero elements before warning.
    pub max_zero_fraction: f64,
    /// Relative tolerance for numerical comparison.
    pub rtol: f64,
    /// Absolute tolerance for numerical comparison.
    pub atol: f64,
}

impl Default for ToleranceThresholds {
    fn default() -> Self {
        Self {
            max_weight_abs: 1e6,
            min_weight_std: 1e-8,
            max_zero_fraction: 0.99,
            rtol: 1e-5,
            atol: 1e-8,
        }
    }
}

/// Top-level configuration for a validation run.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// How thorough the validation should be.
    pub level: ValidationLevel,
    /// Which categories of checks to run.
    pub checks: CheckSet,
    /// Numerical tolerance thresholds.
    pub tolerances: ToleranceThresholds,
    /// Maximum number of issues to record before aborting early.
    pub max_issues: usize,
    /// When true, treat warnings as failures.
    pub strict: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            level: ValidationLevel::Standard,
            checks: CheckSet::default(),
            tolerances: ToleranceThresholds::default(),
            max_issues: 1000,
            strict: false,
        }
    }
}

impl ValidationConfig {
    /// Create a minimal config for fast CI checks.
    pub fn quick() -> Self {
        Self { level: ValidationLevel::Quick, ..Default::default() }
    }

    /// Create an exhaustive config for pre-deployment validation.
    pub fn paranoid() -> Self {
        Self { level: ValidationLevel::Paranoid, strict: true, ..Default::default() }
    }

    /// Validate the configuration itself.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_issues == 0 {
            return Err("max_issues must be > 0".into());
        }
        if self.tolerances.max_weight_abs <= 0.0 {
            return Err("max_weight_abs must be > 0".into());
        }
        if self.tolerances.min_weight_std < 0.0 {
            return Err("min_weight_std must be >= 0".into());
        }
        if !(0.0..=1.0).contains(&self.tolerances.max_zero_fraction) {
            return Err("max_zero_fraction must be in [0.0, 1.0]".into());
        }
        if self.tolerances.rtol < 0.0 {
            return Err("rtol must be >= 0".into());
        }
        if self.tolerances.atol < 0.0 {
            return Err("atol must be >= 0".into());
        }
        Ok(())
    }
}

// ── Tensor / Architecture descriptors ───────────────────────────────────────

/// Lightweight descriptor of a tensor for validation purposes.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
}

impl TensorDescriptor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        let element_count = shape.iter().product();
        Self { name: name.into(), shape, dtype: dtype.into(), element_count }
    }
}

/// Describes the expected model architecture for shape validation.
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub head_dim: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub max_sequence_length: usize,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_layers: 24,
            vocab_size: 32000,
            intermediate_size: 5504,
            head_dim: None,
            num_kv_heads: None,
            max_sequence_length: 2048,
        }
    }
}

/// Statistics about a set of weight values.
#[derive(Debug, Clone)]
pub struct WeightStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub total_count: usize,
}

impl WeightStats {
    /// Compute statistics from a slice of weight values.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                total_count: 0,
            };
        }

        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut zero_count = 0usize;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0f64;

        for &v in values {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }

        let finite_count = values.len() - nan_count - inf_count;
        let mean = if finite_count > 0 { sum / finite_count as f64 } else { 0.0 };

        let variance = if finite_count > 1 {
            values
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| {
                    let d = v - mean;
                    d.mul_add(d, 0.0)
                })
                .sum::<f64>()
                / (finite_count - 1) as f64
        } else {
            0.0
        };

        Self {
            min: if min == f64::INFINITY { 0.0 } else { min },
            max: if max == f64::NEG_INFINITY { 0.0 } else { max },
            mean,
            std_dev: variance.sqrt(),
            nan_count,
            inf_count,
            zero_count,
            total_count: values.len(),
        }
    }
}

// ── Check result / Issue ────────────────────────────────────────────────────

/// A single issue discovered during validation.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: CheckSeverity,
    pub category: String,
    pub message: String,
    pub tensor_name: Option<String>,
    pub details: Option<String>,
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.category, self.message)?;
        if let Some(ref tensor) = self.tensor_name {
            write!(f, " (tensor: {tensor})")?;
        }
        Ok(())
    }
}

/// Result of a single named check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub duration: Duration,
    pub issues: Vec<ValidationIssue>,
}

impl CheckResult {
    fn pass(name: impl Into<String>, duration: Duration) -> Self {
        Self { name: name.into(), status: CheckStatus::Pass, duration, issues: Vec::new() }
    }

    fn fail(name: impl Into<String>, duration: Duration, issues: Vec<ValidationIssue>) -> Self {
        Self { name: name.into(), status: CheckStatus::Fail, duration, issues }
    }

    fn warn(name: impl Into<String>, duration: Duration, issues: Vec<ValidationIssue>) -> Self {
        Self { name: name.into(), status: CheckStatus::Warn, duration, issues }
    }

    fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Skipped,
            duration: Duration::ZERO,
            issues: Vec::new(),
        }
    }
}

// ── Validation Report ───────────────────────────────────────────────────────

/// Aggregated report of all validation checks.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub config: ValidationConfig,
    pub checks: Vec<CheckResult>,
    pub total_duration: Duration,
    pub overall_status: CheckStatus,
    pub summary: ReportSummary,
}

/// Compact summary counts.
#[derive(Debug, Clone, Default)]
pub struct ReportSummary {
    pub passed: usize,
    pub warned: usize,
    pub failed: usize,
    pub skipped: usize,
    pub total_issues: usize,
    pub critical_issues: usize,
}

impl ValidationReport {
    /// Build a report from a set of check results.
    pub fn from_checks(
        config: ValidationConfig,
        checks: Vec<CheckResult>,
        elapsed: Duration,
    ) -> Self {
        let mut summary = ReportSummary::default();
        for c in &checks {
            match c.status {
                CheckStatus::Pass => summary.passed += 1,
                CheckStatus::Warn => summary.warned += 1,
                CheckStatus::Fail => summary.failed += 1,
                CheckStatus::Skipped => summary.skipped += 1,
            }
            summary.total_issues += c.issues.len();
            summary.critical_issues +=
                c.issues.iter().filter(|i| i.severity == CheckSeverity::Critical).count();
        }

        let overall_status = if summary.failed > 0 || summary.critical_issues > 0 {
            CheckStatus::Fail
        } else if summary.warned > 0 {
            if config.strict { CheckStatus::Fail } else { CheckStatus::Warn }
        } else {
            CheckStatus::Pass
        };

        Self { config, checks, total_duration: elapsed, overall_status, summary }
    }

    /// True when the overall validation passed (no failures).
    pub fn is_ok(&self) -> bool {
        self.overall_status == CheckStatus::Pass
    }

    /// All issues across every check, ordered by severity descending.
    pub fn all_issues(&self) -> Vec<&ValidationIssue> {
        let mut issues: Vec<_> = self.checks.iter().flat_map(|c| &c.issues).collect();
        issues.sort_by(|a, b| b.severity.cmp(&a.severity));
        issues
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Model Validation Report ===")?;
        writeln!(f, "Level: {}", self.config.level)?;
        writeln!(f, "Status: {}", self.overall_status)?;
        writeln!(
            f,
            "Checks: {} passed, {} warned, {} failed, {} skipped",
            self.summary.passed, self.summary.warned, self.summary.failed, self.summary.skipped
        )?;
        writeln!(
            f,
            "Issues: {} total ({} critical)",
            self.summary.total_issues, self.summary.critical_issues
        )?;
        writeln!(f, "Duration: {:.2?}", self.total_duration)?;

        for check in &self.checks {
            writeln!(
                f,
                "  [{status}] {name} ({dur:.2?})",
                status = check.status,
                name = check.name,
                dur = check.duration
            )?;
            for issue in &check.issues {
                writeln!(f, "    {issue}")?;
            }
        }
        Ok(())
    }
}

// ── Weight Validator ────────────────────────────────────────────────────────

/// Checks weight tensor integrity: NaN, Inf, distribution, and range.
#[derive(Debug)]
pub struct WeightValidator {
    config: ValidationConfig,
}

impl WeightValidator {
    pub fn new(config: &ValidationConfig) -> Self {
        Self { config: config.clone() }
    }

    /// Validate a single tensor's weight values.
    #[allow(clippy::cast_precision_loss)]
    pub fn validate_tensor(&self, descriptor: &TensorDescriptor, values: &[f64]) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();
        let stats = WeightStats::from_values(values);

        // NaN check
        if stats.nan_count > 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "weights".into(),
                message: format!("{} NaN values detected", stats.nan_count),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        // Inf check
        if stats.inf_count > 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "weights".into(),
                message: format!("{} Inf values detected", stats.inf_count),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        // Range check
        if stats.max.abs() > self.config.tolerances.max_weight_abs
            || stats.min.abs() > self.config.tolerances.max_weight_abs
        {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "weights".into(),
                message: format!(
                    "Weight values out of range: [{:.4}, {:.4}], limit ±{:.0}",
                    stats.min, stats.max, self.config.tolerances.max_weight_abs
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        // Dead-layer check (std-dev too low)
        if stats.std_dev < self.config.tolerances.min_weight_std && stats.total_count > 1 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "weights".into(),
                message: format!(
                    "Standard deviation too low ({:.2e}); possible dead layer",
                    stats.std_dev
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        // Zero-fraction check
        let zero_frac = if stats.total_count > 0 {
            stats.zero_count as f64 / stats.total_count as f64
        } else {
            0.0
        };
        if zero_frac > self.config.tolerances.max_zero_fraction {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "weights".into(),
                message: format!(
                    "High zero fraction ({:.1}%); possible uninitialised tensor",
                    zero_frac * 100.0
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("weight:{}", descriptor.name), elapsed)
        } else if issues
            .iter()
            .any(|i| matches!(i.severity, CheckSeverity::Error | CheckSeverity::Critical))
        {
            CheckResult::fail(format!("weight:{}", descriptor.name), elapsed, issues)
        } else {
            CheckResult::warn(format!("weight:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate a batch of tensors.
    pub fn validate_all(&self, tensors: &[(TensorDescriptor, Vec<f64>)]) -> Vec<CheckResult> {
        tensors.iter().map(|(desc, vals)| self.validate_tensor(desc, vals)).collect()
    }
}

// ── Shape Validator ─────────────────────────────────────────────────────────

/// Validates tensor shapes against an [`ArchitectureConfig`].
#[derive(Debug)]
pub struct ShapeValidator {
    arch: ArchitectureConfig,
}

impl ShapeValidator {
    pub fn new(arch: &ArchitectureConfig) -> Self {
        Self { arch: arch.clone() }
    }

    /// Validate that a tensor's shape is non-empty and has no zero dimensions.
    pub fn validate_basic_shape(&self, descriptor: &TensorDescriptor) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        if descriptor.shape.is_empty() {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "shape".into(),
                message: "Tensor has no dimensions".into(),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        for (i, &dim) in descriptor.shape.iter().enumerate() {
            if dim == 0 {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Error,
                    category: "shape".into(),
                    message: format!("Dimension {i} is zero"),
                    tensor_name: Some(descriptor.name.clone()),
                    details: None,
                });
            }
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("shape:{}", descriptor.name), elapsed)
        } else {
            CheckResult::fail(format!("shape:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate an embedding tensor shape: `[vocab_size, hidden_size]`.
    pub fn validate_embedding_shape(&self, descriptor: &TensorDescriptor) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        if descriptor.shape.len() == 2 {
            if descriptor.shape[0] != self.arch.vocab_size {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Error,
                    category: "shape".into(),
                    message: format!(
                        "Embedding vocab dim {} != expected {}",
                        descriptor.shape[0], self.arch.vocab_size
                    ),
                    tensor_name: Some(descriptor.name.clone()),
                    details: None,
                });
            }
            if descriptor.shape[1] != self.arch.hidden_size {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Error,
                    category: "shape".into(),
                    message: format!(
                        "Embedding hidden dim {} != expected {}",
                        descriptor.shape[1], self.arch.hidden_size
                    ),
                    tensor_name: Some(descriptor.name.clone()),
                    details: None,
                });
            }
        } else {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "shape".into(),
                message: format!("Embedding expected 2 dims, got {}", descriptor.shape.len()),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("shape:{}", descriptor.name), elapsed)
        } else {
            CheckResult::fail(format!("shape:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate an attention projection shape: `[hidden_size, hidden_size]`.
    pub fn validate_attention_shape(&self, descriptor: &TensorDescriptor) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        if descriptor.shape.len() != 2 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "shape".into(),
                message: format!(
                    "Attention weight expected 2 dims, got {}",
                    descriptor.shape.len()
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        } else if descriptor.shape[0] != self.arch.hidden_size
            || descriptor.shape[1] != self.arch.hidden_size
        {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "shape".into(),
                message: format!(
                    "Attention shape {:?} does not match [hidden_size, hidden_size] = [{}, {}]",
                    descriptor.shape, self.arch.hidden_size, self.arch.hidden_size
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("shape:{}", descriptor.name), elapsed)
        } else if issues.iter().any(|i| i.severity >= CheckSeverity::Error) {
            CheckResult::fail(format!("shape:{}", descriptor.name), elapsed, issues)
        } else {
            CheckResult::warn(format!("shape:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate all supplied tensor descriptors for basic shape correctness.
    pub fn validate_all_basic(&self, descriptors: &[TensorDescriptor]) -> Vec<CheckResult> {
        descriptors.iter().map(|d| self.validate_basic_shape(d)).collect()
    }
}

// ── Architecture Validator ──────────────────────────────────────────────────

/// Validates consistency of [`ArchitectureConfig`] parameters.
#[derive(Debug)]
pub struct ArchitectureValidator;

impl ArchitectureValidator {
    /// Run architecture consistency checks.
    #[allow(clippy::too_many_lines)]
    pub fn validate(arch: &ArchitectureConfig) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        Self::check_heads(arch, &mut issues);
        Self::check_kv_heads(arch, &mut issues);
        Self::check_head_dim(arch, &mut issues);
        Self::check_zero_fields(arch, &mut issues);

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass("architecture", elapsed)
        } else if issues
            .iter()
            .any(|i| matches!(i.severity, CheckSeverity::Error | CheckSeverity::Critical))
        {
            CheckResult::fail("architecture", elapsed, issues)
        } else {
            CheckResult::warn("architecture", elapsed, issues)
        }
    }

    fn check_heads(arch: &ArchitectureConfig, issues: &mut Vec<ValidationIssue>) {
        if arch.num_attention_heads == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "architecture".into(),
                message: "num_attention_heads is zero".into(),
                tensor_name: None,
                details: None,
            });
        } else if !arch.hidden_size.is_multiple_of(arch.num_attention_heads) {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "architecture".into(),
                message: format!(
                    "hidden_size ({}) not divisible by num_attention_heads ({})",
                    arch.hidden_size, arch.num_attention_heads
                ),
                tensor_name: None,
                details: None,
            });
        }
    }

    fn check_kv_heads(arch: &ArchitectureConfig, issues: &mut Vec<ValidationIssue>) {
        if let Some(kv_heads) = arch.num_kv_heads {
            if kv_heads == 0 {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Critical,
                    category: "architecture".into(),
                    message: "num_kv_heads is zero".into(),
                    tensor_name: None,
                    details: None,
                });
            } else if !arch.num_attention_heads.is_multiple_of(kv_heads) {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Error,
                    category: "architecture".into(),
                    message: format!(
                        "num_attention_heads ({}) not divisible by num_kv_heads ({kv_heads})",
                        arch.num_attention_heads
                    ),
                    tensor_name: None,
                    details: None,
                });
            }
        }
    }

    fn check_head_dim(arch: &ArchitectureConfig, issues: &mut Vec<ValidationIssue>) {
        if let Some(head_dim) = arch.head_dim {
            let expected = arch.hidden_size / arch.num_attention_heads.max(1);
            if head_dim != expected {
                issues.push(ValidationIssue {
                    severity: CheckSeverity::Warning,
                    category: "architecture".into(),
                    message: format!("head_dim ({head_dim}) != hidden_size/num_heads ({expected})"),
                    tensor_name: None,
                    details: None,
                });
            }
        }
    }

    fn check_zero_fields(arch: &ArchitectureConfig, issues: &mut Vec<ValidationIssue>) {
        if arch.num_layers == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "architecture".into(),
                message: "num_layers is zero".into(),
                tensor_name: None,
                details: None,
            });
        }
        if arch.vocab_size == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "architecture".into(),
                message: "vocab_size is zero".into(),
                tensor_name: None,
                details: None,
            });
        }
        if arch.hidden_size == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "architecture".into(),
                message: "hidden_size is zero".into(),
                tensor_name: None,
                details: None,
            });
        }
        if arch.intermediate_size == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "architecture".into(),
                message: "intermediate_size is zero".into(),
                tensor_name: None,
                details: None,
            });
        }
        if arch.max_sequence_length == 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "architecture".into(),
                message: "max_sequence_length is zero".into(),
                tensor_name: None,
                details: None,
            });
        }
    }
}

// ── Numerical Validator─────────────────────────────────────────────────────

/// Simulated small-scale forward pass for numerical stability checks.
#[derive(Debug)]
pub struct NumericalValidator {
    config: ValidationConfig,
}

impl NumericalValidator {
    pub fn new(config: &ValidationConfig) -> Self {
        Self { config: config.clone() }
    }

    /// Run a simulated forward pass and check for numerical instabilities.
    ///
    /// `activations` should be the output activations from a small probe run.
    pub fn validate_activations(&self, activations: &[f64]) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();
        let stats = WeightStats::from_values(activations);

        if stats.nan_count > 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "numerical".into(),
                message: format!("Forward pass produced {} NaN activations", stats.nan_count),
                tensor_name: None,
                details: None,
            });
        }

        if stats.inf_count > 0 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "numerical".into(),
                message: format!("Forward pass produced {} Inf activations", stats.inf_count),
                tensor_name: None,
                details: None,
            });
        }

        // Exploding activations
        if stats.max.abs() > 1e4 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "numerical".into(),
                message: format!(
                    "Activation magnitudes high: max |activation| = {:.2e}",
                    stats.max.abs().max(stats.min.abs())
                ),
                tensor_name: None,
                details: None,
            });
        }

        // Vanishing activations
        if stats.std_dev < 1e-10 && !activations.is_empty() {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "numerical".into(),
                message: format!(
                    "Activations nearly constant (std = {:.2e}); possible vanishing gradient",
                    stats.std_dev
                ),
                tensor_name: None,
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass("numerical:activations", elapsed)
        } else if issues
            .iter()
            .any(|i| matches!(i.severity, CheckSeverity::Error | CheckSeverity::Critical))
        {
            CheckResult::fail("numerical:activations", elapsed, issues)
        } else {
            CheckResult::warn("numerical:activations", elapsed, issues)
        }
    }

    /// Compare two sets of outputs for numerical closeness (reproducibility check).
    #[allow(clippy::cast_precision_loss)]
    pub fn validate_reproducibility(&self, run_a: &[f64], run_b: &[f64]) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        if run_a.len() != run_b.len() {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "numerical".into(),
                message: format!("Output length mismatch: {} vs {}", run_a.len(), run_b.len()),
                tensor_name: None,
                details: None,
            });
            return CheckResult::fail("numerical:reproducibility", start.elapsed(), issues);
        }

        let mut max_diff = 0.0f64;
        let mut diff_count = 0usize;

        for (i, (&a, &b)) in run_a.iter().zip(run_b.iter()).enumerate() {
            let diff = (a - b).abs();
            let tol = self.config.tolerances.rtol.mul_add(b.abs(), self.config.tolerances.atol);
            if diff > tol {
                diff_count += 1;
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff_count <= 3 {
                    issues.push(ValidationIssue {
                        severity: CheckSeverity::Warning,
                        category: "numerical".into(),
                        message: format!(
                            "Element [{i}]: |{a:.6e} - {b:.6e}| = {diff:.6e} > tol {tol:.6e}"
                        ),
                        tensor_name: None,
                        details: None,
                    });
                }
            }
        }

        if diff_count > 3 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "numerical".into(),
                message: format!(
                    "... and {} more mismatches (max diff = {max_diff:.6e})",
                    diff_count - 3
                ),
                tensor_name: None,
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass("numerical:reproducibility", elapsed)
        } else if diff_count as f64 / run_a.len().max(1) as f64 > 0.1 {
            CheckResult::fail("numerical:reproducibility", elapsed, issues)
        } else {
            CheckResult::warn("numerical:reproducibility", elapsed, issues)
        }
    }
}

// ── Quantization Validator ──────────────────────────────────────────────────

/// Expected quantization format for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationFormat {
    /// Standard 1-bit ternary {-1, 0, +1}.
    Ternary,
    /// 2-bit signed integer.
    I2S,
    /// QK256: 256-element blocks with packed nibbles.
    QK256,
    /// No quantization (full precision).
    None,
}

impl fmt::Display for QuantizationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ternary => write!(f, "Ternary"),
            Self::I2S => write!(f, "I2_S"),
            Self::QK256 => write!(f, "QK256"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Validates that quantized weight values conform to the expected format.
#[derive(Debug)]
pub struct QuantizationValidator;

impl QuantizationValidator {
    /// Validate ternary weights: all values must be in {-1, 0, 1}.
    #[allow(clippy::float_cmp)]
    pub fn validate_ternary(descriptor: &TensorDescriptor, values: &[f64]) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();
        let mut bad_count = 0usize;

        for (i, &v) in values.iter().enumerate() {
            if v != -1.0 && v != 0.0 && v != 1.0 {
                bad_count += 1;
                if bad_count <= 3 {
                    issues.push(ValidationIssue {
                        severity: CheckSeverity::Error,
                        category: "quantization".into(),
                        message: format!("Element [{i}] = {v}; expected {{-1, 0, 1}}"),
                        tensor_name: Some(descriptor.name.clone()),
                        details: None,
                    });
                }
            }
        }

        if bad_count > 3 {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "quantization".into(),
                message: format!("{} more non-ternary values", bad_count - 3),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("quant:ternary:{}", descriptor.name), elapsed)
        } else {
            CheckResult::fail(format!("quant:ternary:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate `I2_S` weights: values in {-1, 0, 1} (2-bit signed).
    pub fn validate_i2s(descriptor: &TensorDescriptor, values: &[f64]) -> CheckResult {
        Self::validate_ternary(descriptor, values)
    }

    /// Validate QK256 block alignment: element count must be divisible by 256.
    pub fn validate_qk256_alignment(descriptor: &TensorDescriptor) -> CheckResult {
        let start = Instant::now();
        let mut issues = Vec::new();

        if !descriptor.element_count.is_multiple_of(256) {
            issues.push(ValidationIssue {
                severity: CheckSeverity::Error,
                category: "quantization".into(),
                message: format!(
                    "QK256 requires element count divisible by 256, got {}",
                    descriptor.element_count
                ),
                tensor_name: Some(descriptor.name.clone()),
                details: None,
            });
        }

        let elapsed = start.elapsed();
        if issues.is_empty() {
            CheckResult::pass(format!("quant:qk256_align:{}", descriptor.name), elapsed)
        } else {
            CheckResult::fail(format!("quant:qk256_align:{}", descriptor.name), elapsed, issues)
        }
    }

    /// Validate quantization for a given format.
    pub fn validate(
        descriptor: &TensorDescriptor,
        values: &[f64],
        format: QuantizationFormat,
    ) -> CheckResult {
        match format {
            QuantizationFormat::Ternary => Self::validate_ternary(descriptor, values),
            QuantizationFormat::I2S => Self::validate_i2s(descriptor, values),
            QuantizationFormat::QK256 => Self::validate_qk256_alignment(descriptor),
            QuantizationFormat::None => CheckResult::skipped("quant:none"),
        }
    }
}

// ── Model Comparator ────────────────────────────────────────────────────────

/// Summary of differences between two models.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub architecture_match: bool,
    pub weight_count_match: bool,
    pub shape_mismatches: Vec<String>,
    pub weight_diffs: HashMap<String, f64>,
    pub max_diff: f64,
    pub status: CheckStatus,
}

/// Compares two models by architecture, shapes, and weight values.
#[derive(Debug)]
pub struct ModelComparator {
    tolerances: ToleranceThresholds,
}

impl ModelComparator {
    #[must_use]
    pub fn new(tolerances: &ToleranceThresholds) -> Self {
        Self { tolerances: tolerances.clone() }
    }

    /// Compare architecture configs.
    #[allow(clippy::missing_const_for_fn, clippy::unused_self)]
    pub fn compare_architectures(&self, a: &ArchitectureConfig, b: &ArchitectureConfig) -> bool {
        a.hidden_size == b.hidden_size
            && a.num_attention_heads == b.num_attention_heads
            && a.num_layers == b.num_layers
            && a.vocab_size == b.vocab_size
            && a.intermediate_size == b.intermediate_size
    }

    /// Compare two sets of tensors by name, shape, and max absolute difference.
    pub fn compare_tensors(
        &self,
        tensors_a: &[(TensorDescriptor, Vec<f64>)],
        tensors_b: &[(TensorDescriptor, Vec<f64>)],
    ) -> ComparisonResult {
        let map_a: HashMap<&str, &(TensorDescriptor, Vec<f64>)> =
            tensors_a.iter().map(|t| (t.0.name.as_str(), t)).collect();
        let map_b: HashMap<&str, &(TensorDescriptor, Vec<f64>)> =
            tensors_b.iter().map(|t| (t.0.name.as_str(), t)).collect();

        let mut shape_mismatches = Vec::new();
        let mut weight_diffs = HashMap::new();
        let mut max_diff = 0.0f64;

        // Check tensors present in A
        for (name, (desc_a, vals_a)) in &map_a {
            if let Some((desc_b, vals_b)) = map_b.get(name) {
                if desc_a.shape == desc_b.shape {
                    let diff = vals_a
                        .iter()
                        .zip(vals_b.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    weight_diffs.insert(name.to_string(), diff);
                    if diff > max_diff {
                        max_diff = diff;
                    }
                } else {
                    shape_mismatches
                        .push(format!("{name}: {:?} vs {:?}", desc_a.shape, desc_b.shape));
                }
            } else {
                shape_mismatches.push(format!("{name}: missing in model B"));
            }
        }

        // Check for tensors only in B
        for name in map_b.keys() {
            if !map_a.contains_key(name) {
                shape_mismatches.push(format!("{name}: missing in model A"));
            }
        }

        let arch_match = shape_mismatches.is_empty();
        let status = if !arch_match {
            CheckStatus::Fail
        } else if max_diff > self.tolerances.atol {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };

        ComparisonResult {
            architecture_match: arch_match,
            weight_count_match: tensors_a.len() == tensors_b.len(),
            shape_mismatches,
            weight_diffs,
            max_diff,
            status,
        }
    }
}

// ── Model Validator (Orchestrator) ──────────────────────────────────────────

/// Orchestrator: load → check weights → check shapes → check arch → check numerics → report.
#[derive(Debug)]
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    #[must_use]
    pub const fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default configuration.
    pub fn default_validator() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Run all applicable checks and produce a [`ValidationReport`].
    pub fn validate(
        &self,
        arch: &ArchitectureConfig,
        tensors: &[(TensorDescriptor, Vec<f64>)],
        activations: Option<&[f64]>,
        quantization_formats: Option<&HashMap<String, QuantizationFormat>>,
    ) -> ValidationReport {
        let overall_start = Instant::now();
        let mut checks = Vec::new();

        // 1. Architecture check
        if self.config.checks.architecture {
            checks.push(ArchitectureValidator::validate(arch));
        }

        // 2. Shape checks
        if self.config.checks.shapes {
            let shape_validator = ShapeValidator::new(arch);
            let descriptors: Vec<_> = tensors.iter().map(|(d, _)| d.clone()).collect();
            checks.extend(shape_validator.validate_all_basic(&descriptors));
        }

        // 3. Weight checks
        if self.config.checks.weights {
            let weight_validator = WeightValidator::new(&self.config);
            let sample_limit = match self.config.level {
                ValidationLevel::Quick => 8,
                ValidationLevel::Standard => 32,
                ValidationLevel::Thorough | ValidationLevel::Paranoid => tensors.len(),
            };
            let to_check = &tensors[..tensors.len().min(sample_limit)];
            checks.extend(weight_validator.validate_all(to_check));
        }

        // 4. Numerical stability checks
        if self.config.checks.numerical {
            if let Some(acts) = activations {
                let numerical = NumericalValidator::new(&self.config);
                checks.push(numerical.validate_activations(acts));
            } else if self.config.level >= ValidationLevel::Thorough {
                checks.push(CheckResult::skipped("numerical:activations"));
            }
        }

        // 5. Quantization checks
        if self.config.checks.quantization {
            if let Some(formats) = quantization_formats {
                for (desc, vals) in tensors {
                    if let Some(&fmt) = formats.get(&desc.name) {
                        checks.push(QuantizationValidator::validate(desc, vals, fmt));
                    }
                }
            } else if self.config.level >= ValidationLevel::Thorough {
                checks.push(CheckResult::skipped("quantization"));
            }
        }

        // Check early-abort
        let issue_count: usize = checks.iter().map(|c| c.issues.len()).sum();
        if issue_count >= self.config.max_issues {
            log::warn!("Validation aborted: reached max_issues limit ({})", self.config.max_issues);
        }

        ValidationReport::from_checks(self.config.clone(), checks, overall_start.elapsed())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper factories ────────────────────────────────────────────────

    fn default_arch() -> ArchitectureConfig {
        ArchitectureConfig::default()
    }

    fn make_tensor(name: &str, shape: Vec<usize>, dtype: &str) -> TensorDescriptor {
        TensorDescriptor::new(name, shape, dtype)
    }

    #[allow(clippy::cast_precision_loss)]
    fn normal_weights(n: usize) -> Vec<f64> {
        // Deterministic pseudo-normal values
        (0..n)
            .map(|i| {
                let x = (i as f64).mul_add(0.618_033_988_749_895, 0.0).fract().mul_add(2.0, -1.0);
                x * 0.1
            })
            .collect()
    }

    fn ternary_weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect()
    }

    // ── CheckSeverity ───────────────────────────────────────────────────

    #[test]
    fn severity_ordering() {
        assert!(CheckSeverity::Info < CheckSeverity::Warning);
        assert!(CheckSeverity::Warning < CheckSeverity::Error);
        assert!(CheckSeverity::Error < CheckSeverity::Critical);
    }

    #[test]
    fn severity_display() {
        assert_eq!(CheckSeverity::Info.to_string(), "INFO");
        assert_eq!(CheckSeverity::Warning.to_string(), "WARN");
        assert_eq!(CheckSeverity::Error.to_string(), "ERROR");
        assert_eq!(CheckSeverity::Critical.to_string(), "CRITICAL");
    }

    // ── CheckStatus ─────────────────────────────────────────────────────

    #[test]
    fn status_display() {
        assert_eq!(CheckStatus::Pass.to_string(), "PASS");
        assert_eq!(CheckStatus::Warn.to_string(), "WARN");
        assert_eq!(CheckStatus::Fail.to_string(), "FAIL");
        assert_eq!(CheckStatus::Skipped.to_string(), "SKIPPED");
    }

    // ── ValidationLevel ─────────────────────────────────────────────────

    #[test]
    fn validation_level_default_is_standard() {
        assert_eq!(ValidationLevel::default(), ValidationLevel::Standard);
    }

    #[test]
    fn validation_level_ordering() {
        assert!(ValidationLevel::Quick < ValidationLevel::Standard);
        assert!(ValidationLevel::Standard < ValidationLevel::Thorough);
        assert!(ValidationLevel::Thorough < ValidationLevel::Paranoid);
    }

    #[test]
    fn validation_level_display() {
        assert_eq!(ValidationLevel::Quick.to_string(), "Quick");
        assert_eq!(ValidationLevel::Paranoid.to_string(), "Paranoid");
    }

    // ── CheckSet ────────────────────────────────────────────────────────

    #[test]
    fn check_set_default_all_enabled() {
        let cs = CheckSet::default();
        assert!(cs.weights);
        assert!(cs.shapes);
        assert!(cs.architecture);
        assert!(cs.numerical);
        assert!(cs.quantization);
    }

    // ── ToleranceThresholds ─────────────────────────────────────────────

    #[test]
    fn tolerance_defaults_are_positive() {
        let t = ToleranceThresholds::default();
        assert!(t.max_weight_abs > 0.0);
        assert!(t.min_weight_std >= 0.0);
        assert!(t.rtol >= 0.0);
        assert!(t.atol >= 0.0);
        assert!((0.0..=1.0).contains(&t.max_zero_fraction));
    }

    // ── ValidationConfig ────────────────────────────────────────────────

    #[test]
    fn config_default_validates() {
        assert!(ValidationConfig::default().validate().is_ok());
    }

    #[test]
    fn config_quick_factory() {
        let c = ValidationConfig::quick();
        assert_eq!(c.level, ValidationLevel::Quick);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_paranoid_factory() {
        let c = ValidationConfig::paranoid();
        assert_eq!(c.level, ValidationLevel::Paranoid);
        assert!(c.strict);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_rejects_zero_max_issues() {
        let c = ValidationConfig { max_issues: 0, ..ValidationConfig::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_max_weight_abs() {
        let c = ValidationConfig {
            tolerances: ToleranceThresholds {
                max_weight_abs: -1.0,
                ..ToleranceThresholds::default()
            },
            ..ValidationConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_min_weight_std() {
        let mut c = ValidationConfig::default();
        c.tolerances.min_weight_std = -0.01;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rejects_out_of_range_zero_fraction() {
        let mut c = ValidationConfig::default();
        c.tolerances.max_zero_fraction = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_rtol() {
        let mut c = ValidationConfig::default();
        c.tolerances.rtol = -1.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_atol() {
        let mut c = ValidationConfig::default();
        c.tolerances.atol = -0.1;
        assert!(c.validate().is_err());
    }

    // ── TensorDescriptor ────────────────────────────────────────────────

    #[test]
    fn tensor_descriptor_element_count() {
        let t = make_tensor("w", vec![3, 4, 5], "f32");
        assert_eq!(t.element_count, 60);
    }

    #[test]
    fn tensor_descriptor_scalar() {
        let t = make_tensor("bias", vec![1], "f32");
        assert_eq!(t.element_count, 1);
    }

    #[test]
    fn tensor_descriptor_empty_shape() {
        let t = make_tensor("empty", vec![], "f32");
        assert_eq!(t.element_count, 1); // product of empty = 1
    }

    // ── WeightStats ─────────────────────────────────────────────────────

    #[test]
    fn weight_stats_empty() {
        let s = WeightStats::from_values(&[]);
        assert_eq!(s.total_count, 0);
        assert_eq!(s.nan_count, 0);
    }

    #[test]
    fn weight_stats_basic() {
        let s = WeightStats::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.total_count, 5);
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min - 1.0).abs() < 1e-10);
        assert!((s.max - 5.0).abs() < 1e-10);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.inf_count, 0);
    }

    #[test]
    fn weight_stats_nan_detection() {
        let s = WeightStats::from_values(&[1.0, f64::NAN, 3.0]);
        assert_eq!(s.nan_count, 1);
        assert_eq!(s.total_count, 3);
    }

    #[test]
    fn weight_stats_inf_detection() {
        let s = WeightStats::from_values(&[1.0, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(s.inf_count, 2);
    }

    #[test]
    fn weight_stats_zero_count() {
        let s = WeightStats::from_values(&[0.0, 0.0, 1.0]);
        assert_eq!(s.zero_count, 2);
    }

    #[test]
    fn weight_stats_all_nan() {
        let s = WeightStats::from_values(&[f64::NAN, f64::NAN]);
        assert_eq!(s.nan_count, 2);
        assert!((s.mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn weight_stats_single_value() {
        let s = WeightStats::from_values(&[42.0]);
        assert!((s.min - 42.0).abs() < 1e-10);
        assert!((s.max - 42.0).abs() < 1e-10);
        assert!((s.std_dev - 0.0).abs() < 1e-10);
    }

    // ── ValidationIssue ─────────────────────────────────────────────────

    #[test]
    fn issue_display_with_tensor() {
        let issue = ValidationIssue {
            severity: CheckSeverity::Error,
            category: "weights".into(),
            message: "NaN found".into(),
            tensor_name: Some("layer.0.q".into()),
            details: None,
        };
        let s = issue.to_string();
        assert!(s.contains("ERROR"));
        assert!(s.contains("NaN found"));
        assert!(s.contains("layer.0.q"));
    }

    #[test]
    fn issue_display_without_tensor() {
        let issue = ValidationIssue {
            severity: CheckSeverity::Warning,
            category: "arch".into(),
            message: "suspicious".into(),
            tensor_name: None,
            details: None,
        };
        let s = issue.to_string();
        assert!(s.contains("WARN"));
        assert!(!s.contains("tensor"));
    }

    // ── CheckResult ─────────────────────────────────────────────────────

    #[test]
    fn check_result_pass() {
        let r = CheckResult::pass("test", Duration::from_millis(10));
        assert_eq!(r.status, CheckStatus::Pass);
        assert!(r.issues.is_empty());
    }

    #[test]
    fn check_result_skipped() {
        let r = CheckResult::skipped("test");
        assert_eq!(r.status, CheckStatus::Skipped);
        assert_eq!(r.duration, Duration::ZERO);
    }

    // ── WeightValidator ─────────────────────────────────────────────────

    #[test]
    fn weight_validator_clean_tensor() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let desc = make_tensor("w", vec![4, 4], "f32");
        let vals = normal_weights(16);
        let r = v.validate_tensor(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn weight_validator_detects_nan() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let desc = make_tensor("w", vec![4], "f32");
        let vals = vec![1.0, f64::NAN, 0.5, -0.5];
        let r = v.validate_tensor(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Fail);
        assert!(r.issues.iter().any(|i| i.severity == CheckSeverity::Critical));
    }

    #[test]
    fn weight_validator_detects_inf() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let desc = make_tensor("w", vec![2], "f32");
        let vals = vec![f64::INFINITY, 0.0];
        let r = v.validate_tensor(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn weight_validator_detects_out_of_range() {
        let mut cfg = ValidationConfig::default();
        cfg.tolerances.max_weight_abs = 1.0;
        let v = WeightValidator::new(&cfg);
        let desc = make_tensor("w", vec![3], "f32");
        let vals = vec![0.5, -1.5, 0.2];
        let r = v.validate_tensor(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn weight_validator_warns_dead_layer() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let desc = make_tensor("w", vec![100], "f32");
        let vals = vec![0.5; 100];
        let r = v.validate_tensor(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn weight_validator_warns_high_zero_fraction() {
        let mut cfg = ValidationConfig::default();
        cfg.tolerances.max_zero_fraction = 0.5;
        let v = WeightValidator::new(&cfg);
        let desc = make_tensor("w", vec![10], "f32");
        let vals = vec![0.0; 10];
        let r = v.validate_tensor(&desc, &vals);
        // Both dead-layer and zero-fraction warnings
        assert!(!r.issues.is_empty());
    }

    #[test]
    fn weight_validator_batch() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let tensors = vec![
            (make_tensor("a", vec![4], "f32"), normal_weights(4)),
            (make_tensor("b", vec![4], "f32"), normal_weights(4)),
        ];
        let results = v.validate_all(&tensors);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.status == CheckStatus::Pass));
    }

    #[test]
    fn weight_validator_empty_tensor() {
        let v = WeightValidator::new(&ValidationConfig::default());
        let desc = make_tensor("w", vec![0], "f32");
        let r = v.validate_tensor(&desc, &[]);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    // ── ShapeValidator ──────────────────────────────────────────────────

    #[test]
    fn shape_basic_valid() {
        let sv = ShapeValidator::new(&default_arch());
        let desc = make_tensor("w", vec![4, 8], "f32");
        let r = sv.validate_basic_shape(&desc);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn shape_basic_empty_dims() {
        let sv = ShapeValidator::new(&default_arch());
        let desc = make_tensor("w", vec![], "f32");
        let r = sv.validate_basic_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_basic_zero_dim() {
        let sv = ShapeValidator::new(&default_arch());
        let desc = make_tensor("w", vec![4, 0, 8], "f32");
        let r = sv.validate_basic_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_embedding_valid() {
        let arch = default_arch();
        let sv = ShapeValidator::new(&arch);
        let desc = make_tensor("embed", vec![arch.vocab_size, arch.hidden_size], "f32");
        let r = sv.validate_embedding_shape(&desc);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn shape_embedding_wrong_vocab() {
        let arch = default_arch();
        let sv = ShapeValidator::new(&arch);
        let desc = make_tensor("embed", vec![999, arch.hidden_size], "f32");
        let r = sv.validate_embedding_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_embedding_wrong_rank() {
        let sv = ShapeValidator::new(&default_arch());
        let desc = make_tensor("embed", vec![32000], "f32");
        let r = sv.validate_embedding_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_embedding_wrong_hidden() {
        let arch = default_arch();
        let sv = ShapeValidator::new(&arch);
        let desc = make_tensor("embed", vec![arch.vocab_size, 1], "f32");
        let r = sv.validate_embedding_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_attention_valid() {
        let arch = default_arch();
        let sv = ShapeValidator::new(&arch);
        let desc = make_tensor("attn", vec![arch.hidden_size, arch.hidden_size], "f32");
        let r = sv.validate_attention_shape(&desc);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn shape_attention_mismatch_warns() {
        let arch = default_arch();
        let sv = ShapeValidator::new(&arch);
        let desc = make_tensor("attn", vec![arch.hidden_size, 512], "f32");
        let r = sv.validate_attention_shape(&desc);
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn shape_attention_wrong_rank() {
        let sv = ShapeValidator::new(&default_arch());
        let desc = make_tensor("attn", vec![2048], "f32");
        let r = sv.validate_attention_shape(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn shape_validate_all_basic() {
        let sv = ShapeValidator::new(&default_arch());
        let descs =
            vec![make_tensor("a", vec![4, 8], "f32"), make_tensor("b", vec![2, 3, 4], "f32")];
        let results = sv.validate_all_basic(&descs);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.status == CheckStatus::Pass));
    }

    // ── ArchitectureValidator ───────────────────────────────────────────

    #[test]
    fn arch_valid_default() {
        let r = ArchitectureValidator::validate(&default_arch());
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn arch_hidden_not_divisible_by_heads() {
        let mut arch = default_arch();
        arch.hidden_size = 2049; // Not divisible by 32
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_zero_heads() {
        let mut arch = default_arch();
        arch.num_attention_heads = 0;
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
        assert!(r.issues.iter().any(|i| i.severity == CheckSeverity::Critical));
    }

    #[test]
    fn arch_zero_layers() {
        let mut arch = default_arch();
        arch.num_layers = 0;
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_zero_vocab() {
        let mut arch = default_arch();
        arch.vocab_size = 0;
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_zero_hidden() {
        let mut arch = default_arch();
        arch.hidden_size = 0;
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_kv_heads_valid_gqa() {
        let mut arch = default_arch();
        arch.num_kv_heads = Some(8); // 32 / 8 = 4 groups
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn arch_kv_heads_invalid() {
        let mut arch = default_arch();
        arch.num_kv_heads = Some(5); // 32 % 5 != 0
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_kv_heads_zero() {
        let mut arch = default_arch();
        arch.num_kv_heads = Some(0);
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn arch_head_dim_correct() {
        let mut arch = default_arch();
        arch.head_dim = Some(64); // 2048 / 32 = 64
        let r = ArchitectureValidator::validate(&arch);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn arch_head_dim_mismatch_warns() {
        let mut arch = default_arch();
        arch.head_dim = Some(128); // != 64
        let r = ArchitectureValidator::validate(&arch);
        assert!(r.issues.iter().any(|i| i.severity == CheckSeverity::Warning));
    }

    #[test]
    fn arch_zero_intermediate_warns() {
        let mut arch = default_arch();
        arch.intermediate_size = 0;
        let r = ArchitectureValidator::validate(&arch);
        // Warning, not error
        assert!(r.issues.iter().any(|i| i.severity == CheckSeverity::Warning));
    }

    #[test]
    fn arch_zero_max_seq_warns() {
        let mut arch = default_arch();
        arch.max_sequence_length = 0;
        let r = ArchitectureValidator::validate(&arch);
        assert!(r.issues.iter().any(|i| i.severity == CheckSeverity::Warning));
    }

    // ── NumericalValidator ──────────────────────────────────────────────

    #[test]
    fn numerical_clean_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let acts = normal_weights(64);
        let r = nv.validate_activations(&acts);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn numerical_nan_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_activations(&[1.0, f64::NAN, 0.5]);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn numerical_inf_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_activations(&[0.0, f64::INFINITY]);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn numerical_exploding_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_activations(&[1e5, -1e5, 0.0]);
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn numerical_vanishing_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_activations(&[0.0; 100]);
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn numerical_empty_activations() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_activations(&[]);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn reproducibility_identical() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let a = vec![1.0, 2.0, 3.0];
        let r = nv.validate_reproducibility(&a, &a);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn reproducibility_within_tolerance() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + 1e-10, 2.0 - 1e-10, 3.0];
        let r = nv.validate_reproducibility(&a, &b);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn reproducibility_length_mismatch() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let r = nv.validate_reproducibility(&[1.0, 2.0], &[1.0]);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn reproducibility_large_diff() {
        let nv = NumericalValidator::new(&ValidationConfig::default());
        let a = vec![1.0; 100];
        let b = vec![2.0; 100];
        let r = nv.validate_reproducibility(&a, &b);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn reproducibility_small_diff_warns() {
        let mut cfg = ValidationConfig::default();
        cfg.tolerances.atol = 0.001;
        let nv = NumericalValidator::new(&cfg);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b: Vec<f64> = a.iter().map(|x| x + 0.005).collect();
        let r = nv.validate_reproducibility(&a, &b);
        // Some elements outside tol but < 10%
        assert!(r.status == CheckStatus::Warn || r.status == CheckStatus::Fail);
    }

    // ── QuantizationFormat ──────────────────────────────────────────────

    #[test]
    fn quant_format_display() {
        assert_eq!(QuantizationFormat::Ternary.to_string(), "Ternary");
        assert_eq!(QuantizationFormat::I2S.to_string(), "I2_S");
        assert_eq!(QuantizationFormat::QK256.to_string(), "QK256");
        assert_eq!(QuantizationFormat::None.to_string(), "None");
    }

    // ── QuantizationValidator ───────────────────────────────────────────

    #[test]
    fn quant_ternary_valid() {
        let desc = make_tensor("w", vec![9], "i2s");
        let vals = ternary_weights(9);
        let r = QuantizationValidator::validate_ternary(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn quant_ternary_invalid() {
        let desc = make_tensor("w", vec![3], "i2s");
        let vals = vec![0.5, -1.0, 1.0];
        let r = QuantizationValidator::validate_ternary(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn quant_i2s_delegates_to_ternary() {
        let desc = make_tensor("w", vec![6], "i2s");
        let vals = ternary_weights(6);
        let r = QuantizationValidator::validate_i2s(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn quant_qk256_aligned() {
        let desc = make_tensor("w", vec![256], "qk256");
        let r = QuantizationValidator::validate_qk256_alignment(&desc);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn quant_qk256_unaligned() {
        let desc = make_tensor("w", vec![100], "qk256");
        let r = QuantizationValidator::validate_qk256_alignment(&desc);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn quant_qk256_multi_block() {
        let desc = make_tensor("w", vec![512], "qk256");
        let r = QuantizationValidator::validate_qk256_alignment(&desc);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn quant_dispatch_none_skipped() {
        let desc = make_tensor("w", vec![4], "f32");
        let r = QuantizationValidator::validate(&desc, &[1.0; 4], QuantizationFormat::None);
        assert_eq!(r.status, CheckStatus::Skipped);
    }

    #[test]
    fn quant_dispatch_ternary() {
        let desc = make_tensor("w", vec![3], "i2s");
        let vals = ternary_weights(3);
        let r = QuantizationValidator::validate(&desc, &vals, QuantizationFormat::Ternary);
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[test]
    fn quant_ternary_many_bad_values() {
        let desc = make_tensor("w", vec![10], "i2s");
        let vals = vec![0.5; 10]; // all non-ternary
        let r = QuantizationValidator::validate_ternary(&desc, &vals);
        assert_eq!(r.status, CheckStatus::Fail);
        // Should have summary issue for extra values
        assert!(r.issues.len() >= 2);
    }

    // ── ModelComparator ─────────────────────────────────────────────────

    #[test]
    fn comparator_identical_models() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let tensors: Vec<(TensorDescriptor, Vec<f64>)> = vec![
            (make_tensor("w1", vec![4], "f32"), vec![1.0, 2.0, 3.0, 4.0]),
            (make_tensor("w2", vec![2], "f32"), vec![0.5, -0.5]),
        ];
        let r = mc.compare_tensors(&tensors, &tensors);
        assert_eq!(r.status, CheckStatus::Pass);
        assert!(r.architecture_match);
        assert!(r.weight_count_match);
        assert!((r.max_diff - 0.0).abs() < 1e-10);
    }

    #[test]
    fn comparator_different_weights() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let a = vec![(make_tensor("w", vec![2], "f32"), vec![1.0, 2.0])];
        let b = vec![(make_tensor("w", vec![2], "f32"), vec![1.0, 3.0])];
        let r = mc.compare_tensors(&a, &b);
        assert!((r.max_diff - 1.0).abs() < 1e-10);
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn comparator_missing_tensor() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let a = vec![(make_tensor("w1", vec![2], "f32"), vec![1.0, 2.0])];
        let b = vec![(make_tensor("w2", vec![2], "f32"), vec![1.0, 2.0])];
        let r = mc.compare_tensors(&a, &b);
        assert!(!r.architecture_match);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn comparator_shape_mismatch() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let a = vec![(make_tensor("w", vec![2], "f32"), vec![1.0, 2.0])];
        let b = vec![(make_tensor("w", vec![3], "f32"), vec![1.0, 2.0, 3.0])];
        let r = mc.compare_tensors(&a, &b);
        assert!(!r.architecture_match);
    }

    #[test]
    fn comparator_arch_match() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let a = default_arch();
        assert!(mc.compare_architectures(&a, &a));
    }

    #[test]
    fn comparator_arch_mismatch() {
        let mc = ModelComparator::new(&ToleranceThresholds::default());
        let a = default_arch();
        let mut b = default_arch();
        b.num_layers = 12;
        assert!(!mc.compare_architectures(&a, &b));
    }

    // ── ValidationReport ────────────────────────────────────────────────

    #[test]
    fn report_all_pass() {
        let checks = vec![
            CheckResult::pass("a", Duration::from_millis(1)),
            CheckResult::pass("b", Duration::from_millis(2)),
        ];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(3),
        );
        assert!(r.is_ok());
        assert_eq!(r.summary.passed, 2);
        assert_eq!(r.summary.total_issues, 0);
    }

    #[test]
    fn report_with_failure() {
        let checks = vec![
            CheckResult::pass("a", Duration::from_millis(1)),
            CheckResult::fail(
                "b",
                Duration::from_millis(2),
                vec![ValidationIssue {
                    severity: CheckSeverity::Error,
                    category: "test".into(),
                    message: "bad".into(),
                    tensor_name: None,
                    details: None,
                }],
            ),
        ];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(3),
        );
        assert!(!r.is_ok());
        assert_eq!(r.summary.failed, 1);
    }

    #[test]
    fn report_strict_treats_warns_as_fail() {
        let checks = vec![CheckResult::warn(
            "a",
            Duration::from_millis(1),
            vec![ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "test".into(),
                message: "hmm".into(),
                tensor_name: None,
                details: None,
            }],
        )];
        let cfg = ValidationConfig { strict: true, ..ValidationConfig::default() };
        let r = ValidationReport::from_checks(cfg, checks, Duration::from_millis(1));
        assert!(!r.is_ok());
    }

    #[test]
    fn report_non_strict_warns_is_ok() {
        let checks = vec![CheckResult::warn(
            "a",
            Duration::from_millis(1),
            vec![ValidationIssue {
                severity: CheckSeverity::Warning,
                category: "test".into(),
                message: "hmm".into(),
                tensor_name: None,
                details: None,
            }],
        )];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(1),
        );
        assert_eq!(r.overall_status, CheckStatus::Warn);
        // is_ok checks for Pass only
        assert!(!r.is_ok());
    }

    #[test]
    fn report_all_issues_sorted_by_severity() {
        let checks = vec![
            CheckResult::fail(
                "a",
                Duration::from_millis(1),
                vec![ValidationIssue {
                    severity: CheckSeverity::Warning,
                    category: "test".into(),
                    message: "w".into(),
                    tensor_name: None,
                    details: None,
                }],
            ),
            CheckResult::fail(
                "b",
                Duration::from_millis(1),
                vec![ValidationIssue {
                    severity: CheckSeverity::Critical,
                    category: "test".into(),
                    message: "c".into(),
                    tensor_name: None,
                    details: None,
                }],
            ),
        ];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(1),
        );
        let issues = r.all_issues();
        assert_eq!(issues[0].severity, CheckSeverity::Critical);
        assert_eq!(issues[1].severity, CheckSeverity::Warning);
    }

    #[test]
    fn report_display_contains_status() {
        let checks = vec![CheckResult::pass("a", Duration::from_millis(1))];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(1),
        );
        let s = r.to_string();
        assert!(s.contains("PASS"));
        assert!(s.contains("Model Validation Report"));
    }

    #[test]
    fn report_skipped_counted() {
        let checks = vec![CheckResult::skipped("x")];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(1),
        );
        assert_eq!(r.summary.skipped, 1);
        assert!(r.is_ok()); // skipped doesn't count as failure
    }

    #[test]
    fn report_critical_issue_counted() {
        let checks = vec![CheckResult::fail(
            "x",
            Duration::from_millis(1),
            vec![ValidationIssue {
                severity: CheckSeverity::Critical,
                category: "t".into(),
                message: "m".into(),
                tensor_name: None,
                details: None,
            }],
        )];
        let r = ValidationReport::from_checks(
            ValidationConfig::default(),
            checks,
            Duration::from_millis(1),
        );
        assert_eq!(r.summary.critical_issues, 1);
    }

    // ── ModelValidator (orchestrator) ───────────────────────────────────

    #[test]
    fn orchestrator_clean_model() {
        let mv = ModelValidator::default_validator();
        let arch = default_arch();
        let tensors =
            vec![(make_tensor("embed", vec![32000, 2048], "f32"), normal_weights(32000 * 2048))];
        let report = mv.validate(&arch, &tensors, None, None);
        assert!(report.overall_status != CheckStatus::Fail || report.summary.critical_issues == 0);
    }

    #[test]
    fn orchestrator_with_activations() {
        let mv = ModelValidator::default_validator();
        let arch = default_arch();
        let tensors = vec![(make_tensor("w", vec![4, 4], "f32"), normal_weights(16))];
        let activations = normal_weights(16);
        let report = mv.validate(&arch, &tensors, Some(&activations), None);
        assert!(report.checks.iter().any(|c| c.name.contains("numerical")));
    }

    #[test]
    fn orchestrator_with_quantization() {
        let mv = ModelValidator::default_validator();
        let arch = default_arch();
        let vals = ternary_weights(256);
        let tensors = vec![(make_tensor("w", vec![256], "i2s"), vals)];
        let mut fmts = HashMap::new();
        fmts.insert("w".into(), QuantizationFormat::Ternary);
        let report = mv.validate(&arch, &tensors, None, Some(&fmts));
        let quant_count = report.checks.iter().filter(|c| c.name.contains("quant")).count();
        assert!(quant_count > 0);
    }

    #[test]
    fn orchestrator_quick_limits_tensor_sampling() {
        let cfg = ValidationConfig::quick();
        let mv = ModelValidator::new(cfg);
        let arch = default_arch();
        let tensors: Vec<_> = (0..20)
            .map(|i| (make_tensor(&format!("w{i}"), vec![4], "f32"), normal_weights(4)))
            .collect();
        let report = mv.validate(&arch, &tensors, None, None);
        // Quick mode only checks 8 tensors for weights
        let weight_checks = report.checks.iter().filter(|c| c.name.starts_with("weight:")).count();
        assert!(weight_checks <= 8);
    }

    #[test]
    fn orchestrator_skips_disabled_checks() {
        let mut cfg = ValidationConfig::default();
        cfg.checks.architecture = false;
        cfg.checks.numerical = false;
        cfg.checks.quantization = false;
        let mv = ModelValidator::new(cfg);
        let arch = default_arch();
        let tensors = vec![(make_tensor("w", vec![4], "f32"), normal_weights(4))];
        let report = mv.validate(&arch, &tensors, Some(&[1.0]), None);
        assert!(!report.checks.iter().any(|c| c.name == "architecture"));
        assert!(!report.checks.iter().any(|c| c.name.contains("numerical")));
    }

    #[test]
    fn orchestrator_bad_arch_reports_failure() {
        let mv = ModelValidator::default_validator();
        let mut arch = default_arch();
        arch.num_attention_heads = 0;
        let tensors = vec![(make_tensor("w", vec![4], "f32"), normal_weights(4))];
        let report = mv.validate(&arch, &tensors, None, None);
        assert_eq!(report.overall_status, CheckStatus::Fail);
    }

    #[test]
    fn orchestrator_nan_weights_reports_failure() {
        let mv = ModelValidator::default_validator();
        let arch = default_arch();
        let tensors = vec![(make_tensor("w", vec![4], "f32"), vec![1.0, f64::NAN, 0.0, -1.0])];
        let report = mv.validate(&arch, &tensors, None, None);
        assert_eq!(report.overall_status, CheckStatus::Fail);
    }

    #[test]
    fn orchestrator_empty_tensors() {
        let mv = ModelValidator::default_validator();
        let arch = default_arch();
        let report = mv.validate(&arch, &[], None, None);
        // Architecture check should still run
        assert!(report.checks.iter().any(|c| c.name == "architecture"));
    }

    #[test]
    fn orchestrator_thorough_skips_noted() {
        let cfg =
            ValidationConfig { level: ValidationLevel::Thorough, ..ValidationConfig::default() };
        let mv = ModelValidator::new(cfg);
        let arch = default_arch();
        let report = mv.validate(&arch, &[], None, None);
        // Without activations, numerical is skipped at Thorough level
        assert!(report.checks.iter().any(|c| c.status == CheckStatus::Skipped));
    }
}
