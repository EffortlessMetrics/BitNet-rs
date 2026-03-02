//! Model validator.
//!
//! Pre-flight validation of model files and configurations.

/// Validation severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// A validation finding.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    pub category: String,
    pub message: String,
}

/// Validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub passed: bool,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self { findings: Vec::new(), passed: true }
    }

    pub fn add(&mut self, severity: Severity, category: &str, message: &str) {
        if severity == Severity::Error {
            self.passed = false;
        }
        self.findings.push(Finding {
            severity,
            category: category.to_string(),
            message: message.to_string(),
        });
    }

    pub fn errors(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error).collect()
    }

    pub fn warnings(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Warning).collect()
    }

    pub fn error_count(&self) -> usize {
        self.findings.iter().filter(|f| f.severity == Severity::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.findings.iter().filter(|f| f.severity == Severity::Warning).count()
    }

    pub fn merge(&mut self, other: ValidationReport) {
        for f in other.findings {
            if f.severity == Severity::Error {
                self.passed = false;
            }
            self.findings.push(f);
        }
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Model configuration to validate.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub intermediate_size: usize,
}

/// Validate a model configuration.
pub fn validate_config(config: &ModelConfig) -> ValidationReport {
    let mut report = ValidationReport::new();

    // Hidden size must be positive
    if config.hidden_size == 0 {
        report.add(Severity::Error, "shape", "hidden_size must be > 0");
    }

    // Head dimension check
    if config.num_heads > 0 && !config.hidden_size.is_multiple_of(config.num_heads) {
        report.add(Severity::Error, "shape", "hidden_size must be divisible by num_heads");
    }

    // KV heads must divide num_heads
    if config.num_kv_heads > 0 && !config.num_heads.is_multiple_of(config.num_kv_heads) {
        report.add(Severity::Error, "gqa", "num_heads must be divisible by num_kv_heads");
    }

    // Layer count
    if config.num_layers == 0 {
        report.add(Severity::Error, "shape", "num_layers must be > 0");
    } else if config.num_layers > 200 {
        report.add(Severity::Warning, "shape", "unusually high layer count (>200)");
    }

    // Vocab size
    if config.vocab_size == 0 {
        report.add(Severity::Error, "vocab", "vocab_size must be > 0");
    } else if config.vocab_size > 500_000 {
        report.add(Severity::Warning, "vocab", "unusually large vocabulary (>500K)");
    }

    // Context length
    if config.max_context == 0 {
        report.add(Severity::Error, "context", "max_context must be > 0");
    } else if config.max_context > 131_072 {
        report.add(Severity::Warning, "context", "very large context (>128K)");
    }

    // Intermediate size
    if config.intermediate_size == 0 {
        report.add(Severity::Warning, "shape", "intermediate_size is 0");
    }

    report
}

/// Validate tensor shapes against config.
pub fn validate_tensor_shape(
    name: &str,
    shape: &[usize],
    config: &ModelConfig,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    if shape.is_empty() {
        report.add(Severity::Error, "tensor", &format!("{name}: empty shape"));
        return report;
    }

    // Check for zero dimensions
    if shape.contains(&0) {
        report.add(Severity::Error, "tensor", &format!("{name}: shape has zero dimension"));
    }

    // Embedding matrix check
    if (name.contains("embed") || name.contains("wte"))
        && shape.len() == 2
        && shape[0] != config.vocab_size
    {
        report.add(
            Severity::Warning,
            "tensor",
            &format!(
                "{name}: expected vocab_size={} in dim 0, got {}",
                config.vocab_size, shape[0]
            ),
        );
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phi4_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 10,
            vocab_size: 100352,
            max_context: 16384,
            intermediate_size: 13824,
        }
    }

    #[test]
    fn test_valid_config() {
        let r = validate_config(&phi4_config());
        assert!(r.passed);
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn test_zero_hidden() {
        let mut c = phi4_config();
        c.hidden_size = 0;
        let r = validate_config(&c);
        assert!(!r.passed);
    }

    #[test]
    fn test_head_divisibility() {
        let mut c = phi4_config();
        c.hidden_size = 5121; // not divisible by 40
        let r = validate_config(&c);
        assert!(!r.passed);
    }

    #[test]
    fn test_kv_head_divisibility() {
        let mut c = phi4_config();
        c.num_kv_heads = 7; // 40 not divisible by 7
        let r = validate_config(&c);
        assert!(!r.passed);
    }

    #[test]
    fn test_zero_layers() {
        let mut c = phi4_config();
        c.num_layers = 0;
        let r = validate_config(&c);
        assert!(!r.passed);
    }

    #[test]
    fn test_high_layers_warning() {
        let mut c = phi4_config();
        c.num_layers = 300;
        let r = validate_config(&c);
        assert!(r.passed); // warning, not error
        assert!(r.warning_count() > 0);
    }

    #[test]
    fn test_zero_vocab() {
        let mut c = phi4_config();
        c.vocab_size = 0;
        let r = validate_config(&c);
        assert!(!r.passed);
    }

    #[test]
    fn test_tensor_shape_valid() {
        let c = phi4_config();
        let r = validate_tensor_shape("layer.0.weight", &[5120, 5120], &c);
        assert!(r.passed);
    }

    #[test]
    fn test_tensor_shape_empty() {
        let c = phi4_config();
        let r = validate_tensor_shape("bad", &[], &c);
        assert!(!r.passed);
    }

    #[test]
    fn test_tensor_shape_zero_dim() {
        let c = phi4_config();
        let r = validate_tensor_shape("bad", &[0, 5120], &c);
        assert!(!r.passed);
    }

    #[test]
    fn test_embed_shape_warning() {
        let c = phi4_config();
        let r = validate_tensor_shape("token_embed", &[32000, 5120], &c);
        assert!(r.warning_count() > 0);
    }

    #[test]
    fn test_report_merge() {
        let mut r1 = ValidationReport::new();
        r1.add(Severity::Warning, "a", "w1");
        let mut r2 = ValidationReport::new();
        r2.add(Severity::Error, "b", "e1");
        r1.merge(r2);
        assert!(!r1.passed);
        assert_eq!(r1.findings.len(), 2);
    }

    #[test]
    fn test_severity_str() {
        assert_eq!(Severity::Error.as_str(), "error");
        assert_eq!(Severity::Warning.as_str(), "warning");
    }
}
