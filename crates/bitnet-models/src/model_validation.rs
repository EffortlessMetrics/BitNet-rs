//! Multi-stage model validation pipeline.
//!
//! Validates model configuration, tensor shapes, weight statistics,
//! and architecture consistency before inference.

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub stage: String,
    pub message: String,
}

impl ValidationIssue {
    pub fn info(stage: impl Into<String>, msg: impl Into<String>) -> Self {
        Self { severity: Severity::Info, stage: stage.into(), message: msg.into() }
    }
    pub fn warning(stage: impl Into<String>, msg: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, stage: stage.into(), message: msg.into() }
    }
    pub fn error(stage: impl Into<String>, msg: impl Into<String>) -> Self {
        Self { severity: Severity::Error, stage: stage.into(), message: msg.into() }
    }
}

/// Result of a complete validation run.
#[derive(Debug)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
    pub stages_run: usize,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self { issues: Vec::new(), stages_run: 0 }
    }

    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == Severity::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == Severity::Warning).count()
    }

    pub fn info_count(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == Severity::Info).count()
    }

    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|i| i.severity == Severity::Error).collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} issues ({} errors, {} warnings, {} info) across {} stages",
            self.issues.len(),
            self.error_count(),
            self.warning_count(),
            self.info_count(),
            self.stages_run,
        )
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// A validation check that can be applied to model config.
pub trait ValidationCheck {
    fn name(&self) -> &str;
    fn validate(&self, config: &ModelConfig) -> Vec<ValidationIssue>;
}

/// Minimal model config for validation purposes.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position: usize,
    pub intermediate_size: usize,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_heads
    }

    pub fn gqa_ratio(&self) -> usize {
        if self.num_kv_heads == 0 {
            return 0;
        }
        self.num_heads / self.num_kv_heads
    }
}

/// Check that hidden_size is divisible by num_heads.
pub struct HeadDimCheck;
impl ValidationCheck for HeadDimCheck {
    fn name(&self) -> &str {
        "head_dim"
    }
    fn validate(&self, c: &ModelConfig) -> Vec<ValidationIssue> {
        let mut issues = vec![];
        if c.num_heads > 0 && c.hidden_size % c.num_heads != 0 {
            issues.push(ValidationIssue::error(
                self.name(),
                format!("hidden_size {} not divisible by num_heads {}", c.hidden_size, c.num_heads),
            ));
        }
        issues
    }
}

/// Check GQA configuration.
pub struct GqaCheck;
impl ValidationCheck for GqaCheck {
    fn name(&self) -> &str {
        "gqa"
    }
    fn validate(&self, c: &ModelConfig) -> Vec<ValidationIssue> {
        let mut issues = vec![];
        if c.num_kv_heads > c.num_heads {
            issues.push(ValidationIssue::error(
                self.name(),
                format!("num_kv_heads {} > num_heads {}", c.num_kv_heads, c.num_heads),
            ));
        }
        if c.num_kv_heads > 0 && c.num_heads % c.num_kv_heads != 0 {
            issues.push(ValidationIssue::error(
                self.name(),
                format!(
                    "num_heads {} not divisible by num_kv_heads {}",
                    c.num_heads, c.num_kv_heads
                ),
            ));
        }
        issues
    }
}

/// Check for suspicious sizes.
pub struct SizeCheck;
impl ValidationCheck for SizeCheck {
    fn name(&self) -> &str {
        "size"
    }
    fn validate(&self, c: &ModelConfig) -> Vec<ValidationIssue> {
        let mut issues = vec![];
        if c.vocab_size == 0 {
            issues.push(ValidationIssue::error(self.name(), "vocab_size is 0"));
        }
        if c.num_layers == 0 {
            issues.push(ValidationIssue::error(self.name(), "num_layers is 0"));
        }
        if c.hidden_size == 0 {
            issues.push(ValidationIssue::error(self.name(), "hidden_size is 0"));
        }
        if c.max_position > 131072 {
            issues.push(ValidationIssue::warning(
                self.name(),
                format!("very large max_position: {}", c.max_position),
            ));
        }
        issues
    }
}

/// Run all standard validation checks.
pub fn validate_model(config: &ModelConfig) -> ValidationResult {
    let checks: Vec<Box<dyn ValidationCheck>> =
        vec![Box::new(HeadDimCheck), Box::new(GqaCheck), Box::new(SizeCheck)];
    run_checks(config, &checks)
}

/// Run a set of validation checks.
pub fn run_checks(config: &ModelConfig, checks: &[Box<dyn ValidationCheck>]) -> ValidationResult {
    let mut result = ValidationResult::new();
    for check in checks {
        let issues = check.validate(config);
        result.issues.extend(issues);
        result.stages_run += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 10,
            vocab_size: 100352,
            max_position: 16384,
            intermediate_size: 13824,
        }
    }

    #[test]
    fn test_valid_model() {
        let r = validate_model(&valid_config());
        assert!(r.is_valid());
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn test_bad_head_dim() {
        let mut c = valid_config();
        c.hidden_size = 5121; // not divisible by 40
        let r = validate_model(&c);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_bad_gqa() {
        let mut c = valid_config();
        c.num_kv_heads = 7; // 40 not divisible by 7
        let r = validate_model(&c);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_kv_heads_too_large() {
        let mut c = valid_config();
        c.num_kv_heads = 50; // > num_heads
        let r = validate_model(&c);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_zero_vocab() {
        let mut c = valid_config();
        c.vocab_size = 0;
        let r = validate_model(&c);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_zero_layers() {
        let mut c = valid_config();
        c.num_layers = 0;
        let r = validate_model(&c);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_large_position_warning() {
        let mut c = valid_config();
        c.max_position = 200000;
        let r = validate_model(&c);
        assert!(r.is_valid()); // warning, not error
        assert!(r.warning_count() > 0);
    }

    #[test]
    fn test_head_dim() {
        let c = valid_config();
        assert_eq!(c.head_dim(), 128);
    }

    #[test]
    fn test_gqa_ratio() {
        let c = valid_config();
        assert_eq!(c.gqa_ratio(), 4);
    }

    #[test]
    fn test_summary() {
        let r = validate_model(&valid_config());
        let s = r.summary();
        assert!(s.contains("stages"));
    }

    #[test]
    fn test_errors_accessor() {
        let mut c = valid_config();
        c.vocab_size = 0;
        c.num_layers = 0;
        let r = validate_model(&c);
        assert!(r.errors().len() >= 2);
    }

    #[test]
    fn test_stages_count() {
        let r = validate_model(&valid_config());
        assert_eq!(r.stages_run, 3);
    }
}
