//! Model validation suite for verifying model integrity and compatibility.

/// Validation severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: Severity,
    pub category: String,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ValidationFinding {
    pub fn info(category: &str, message: &str) -> Self {
        Self {
            severity: Severity::Info,
            category: category.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }
    pub fn warning(category: &str, message: &str) -> Self {
        Self {
            severity: Severity::Warning,
            category: category.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }
    pub fn error(category: &str, message: &str) -> Self {
        Self {
            severity: Severity::Error,
            category: category.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }
    pub fn critical(category: &str, message: &str) -> Self {
        Self {
            severity: Severity::Critical,
            category: category.to_string(),
            message: message.to_string(),
            suggestion: None,
        }
    }
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestion = Some(suggestion.to_string());
        self
    }
}

/// Result of running the validation suite.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<ValidationFinding>,
    pub model_name: String,
    pub passed: bool,
}

impl ValidationReport {
    pub fn new(model_name: &str) -> Self {
        Self { findings: Vec::new(), model_name: model_name.to_string(), passed: true }
    }

    pub fn add(&mut self, finding: ValidationFinding) {
        if finding.severity >= Severity::Error {
            self.passed = false;
        }
        self.findings.push(finding);
    }

    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity >= Severity::Error).collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == Severity::Warning).collect()
    }

    pub fn by_category(&self, category: &str) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.category == category).collect()
    }

    pub fn summary(&self) -> String {
        let errors = self.findings.iter().filter(|f| f.severity >= Severity::Error).count();
        let warnings = self.findings.iter().filter(|f| f.severity == Severity::Warning).count();
        let infos = self.findings.iter().filter(|f| f.severity == Severity::Info).count();
        format!(
            "Validation {}: {} errors, {} warnings, {} info",
            if self.passed { "PASSED" } else { "FAILED" },
            errors,
            warnings,
            infos
        )
    }
}

/// Validation check for tensor shapes.
pub fn validate_tensor_shape(
    name: &str,
    actual_shape: &[usize],
    expected_shape: &[usize],
) -> ValidationFinding {
    if actual_shape == expected_shape {
        ValidationFinding::info("tensor_shape", &format!("{name}: shape {actual_shape:?} OK"))
    } else {
        ValidationFinding::error(
            "tensor_shape",
            &format!("{name}: expected {expected_shape:?}, got {actual_shape:?}"),
        )
    }
}

/// Validate embedding dimensions.
pub fn validate_embedding(
    vocab_size: usize,
    hidden_size: usize,
    actual_shape: &[usize],
) -> ValidationFinding {
    let expected = [vocab_size, hidden_size];
    validate_tensor_shape("embedding", actual_shape, &expected)
}

/// Validate attention projection dimensions.
pub fn validate_attention_proj(
    name: &str,
    hidden_size: usize,
    proj_size: usize,
    actual_shape: &[usize],
) -> ValidationFinding {
    let expected = [proj_size, hidden_size];
    validate_tensor_shape(name, actual_shape, &expected)
}

/// Validate that a model has the expected number of layers.
pub fn validate_layer_count(expected: usize, actual: usize) -> ValidationFinding {
    if actual == expected {
        ValidationFinding::info("structure", &format!("Layer count {actual} OK"))
    } else {
        ValidationFinding::error(
            "structure",
            &format!("Expected {expected} layers, found {actual}"),
        )
    }
}

/// Validate dtype compatibility.
pub fn validate_dtype(dtype: &str) -> ValidationFinding {
    match dtype {
        "f32" | "f16" | "bf16" => {
            ValidationFinding::info("dtype", &format!("dtype {dtype} supported"))
        }
        "i2_s" | "q4_0" | "q4_1" | "q8_0" => {
            ValidationFinding::info("dtype", &format!("dtype {dtype} (quantized) supported"))
        }
        _ => ValidationFinding::warning("dtype", &format!("dtype {dtype} may not be supported"))
            .with_suggestion("Consider converting to f16 or bf16"),
    }
}

/// Validate memory requirements.
pub fn validate_memory_requirements(model_bytes: u64, available_bytes: u64) -> ValidationFinding {
    if model_bytes <= available_bytes {
        ValidationFinding::info(
            "memory",
            &format!("Model fits in memory ({} / {} bytes)", model_bytes, available_bytes),
        )
    } else {
        let deficit_gb = (model_bytes - available_bytes) as f64 / 1e9;
        ValidationFinding::error(
            "memory",
            &format!("Model requires {:.1} GB more than available", deficit_gb),
        )
        .with_suggestion("Use quantization or a smaller model")
    }
}

/// Format a validation report as a human-readable string.
pub fn format_report(report: &ValidationReport) -> String {
    let mut out = format!("=== Validation Report: {} ===\n", report.model_name);
    out.push_str(&report.summary());
    out.push('\n');
    for finding in &report.findings {
        let sev = match finding.severity {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
            Severity::Critical => "CRIT",
        };
        out.push_str(&format!("[{sev}] {}: {}\n", finding.category, finding.message));
        if let Some(ref sugg) = finding.suggestion {
            out.push_str(&format!("  → {sugg}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finding_info() {
        let f = ValidationFinding::info("test", "ok");
        assert_eq!(f.severity, Severity::Info);
        assert_eq!(f.category, "test");
    }

    #[test]
    fn test_finding_warning() {
        let f = ValidationFinding::warning("test", "caution");
        assert_eq!(f.severity, Severity::Warning);
    }

    #[test]
    fn test_finding_error() {
        let f = ValidationFinding::error("test", "bad");
        assert_eq!(f.severity, Severity::Error);
    }

    #[test]
    fn test_finding_critical() {
        let f = ValidationFinding::critical("test", "terrible");
        assert_eq!(f.severity, Severity::Critical);
    }

    #[test]
    fn test_finding_with_suggestion() {
        let f = ValidationFinding::warning("test", "issue").with_suggestion("fix it");
        assert_eq!(f.suggestion, Some("fix it".to_string()));
    }

    #[test]
    fn test_report_new() {
        let r = ValidationReport::new("test_model");
        assert!(r.passed);
        assert!(r.findings.is_empty());
    }

    #[test]
    fn test_report_add_info() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::info("cat", "msg"));
        assert!(r.passed);
        assert_eq!(r.findings.len(), 1);
    }

    #[test]
    fn test_report_add_error_fails() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::error("cat", "msg"));
        assert!(!r.passed);
    }

    #[test]
    fn test_report_errors() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::info("a", "ok"));
        r.add(ValidationFinding::error("b", "bad"));
        r.add(ValidationFinding::warning("c", "meh"));
        assert_eq!(r.errors().len(), 1);
    }

    #[test]
    fn test_report_warnings() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::warning("a", "w1"));
        r.add(ValidationFinding::warning("b", "w2"));
        r.add(ValidationFinding::error("c", "e1"));
        assert_eq!(r.warnings().len(), 2);
    }

    #[test]
    fn test_report_by_category() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::info("shape", "ok1"));
        r.add(ValidationFinding::info("dtype", "ok2"));
        r.add(ValidationFinding::info("shape", "ok3"));
        assert_eq!(r.by_category("shape").len(), 2);
    }

    #[test]
    fn test_report_summary_passed() {
        let r = ValidationReport::new("test");
        let s = r.summary();
        assert!(s.contains("PASSED"));
    }

    #[test]
    fn test_report_summary_failed() {
        let mut r = ValidationReport::new("test");
        r.add(ValidationFinding::error("x", "y"));
        let s = r.summary();
        assert!(s.contains("FAILED"));
    }

    #[test]
    fn test_validate_tensor_shape_ok() {
        let f = validate_tensor_shape("test", &[10, 20], &[10, 20]);
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_tensor_shape_mismatch() {
        let f = validate_tensor_shape("test", &[10, 30], &[10, 20]);
        assert_eq!(f.severity, Severity::Error);
    }

    #[test]
    fn test_validate_embedding_ok() {
        let f = validate_embedding(100352, 5120, &[100352, 5120]);
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_embedding_wrong() {
        let f = validate_embedding(100352, 5120, &[32000, 4096]);
        assert_eq!(f.severity, Severity::Error);
    }

    #[test]
    fn test_validate_layer_count_ok() {
        let f = validate_layer_count(40, 40);
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_layer_count_wrong() {
        let f = validate_layer_count(40, 32);
        assert_eq!(f.severity, Severity::Error);
    }

    #[test]
    fn test_validate_dtype_f16() {
        let f = validate_dtype("f16");
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_dtype_bf16() {
        let f = validate_dtype("bf16");
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_dtype_unknown() {
        let f = validate_dtype("fp8_e4m3");
        assert_eq!(f.severity, Severity::Warning);
        assert!(f.suggestion.is_some());
    }

    #[test]
    fn test_validate_memory_fits() {
        let f = validate_memory_requirements(10_000_000_000, 32_000_000_000);
        assert_eq!(f.severity, Severity::Info);
    }

    #[test]
    fn test_validate_memory_exceeds() {
        let f = validate_memory_requirements(30_000_000_000, 16_000_000_000);
        assert_eq!(f.severity, Severity::Error);
        assert!(f.suggestion.is_some());
    }

    #[test]
    fn test_format_report() {
        let mut r = ValidationReport::new("test_model");
        r.add(ValidationFinding::info("test", "all good"));
        r.add(ValidationFinding::error("test", "bad thing").with_suggestion("fix it"));
        let out = format_report(&r);
        assert!(out.contains("test_model"));
        assert!(out.contains("[INFO]"));
        assert!(out.contains("[ERROR]"));
        assert!(out.contains("fix it"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }
}
