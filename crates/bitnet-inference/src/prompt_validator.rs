//! Prompt template validation.
//!
//! Validate prompt templates for correctness: check placeholders,
//! balanced delimiters, token budget, and structural integrity.

/// Validation severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// A single validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
    pub position: Option<usize>,
}

impl ValidationIssue {
    pub fn error(code: &'static str, message: impl Into<String>) -> Self {
        Self { severity: Severity::Error, code, message: message.into(), position: None }
    }

    pub fn warning(code: &'static str, message: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, code, message: message.into(), position: None }
    }

    pub fn info(code: &'static str, message: impl Into<String>) -> Self {
        Self { severity: Severity::Info, code, message: message.into(), position: None }
    }

    pub fn at(mut self, pos: usize) -> Self {
        self.position = Some(pos);
        self
    }
}

/// Validation result.
#[derive(Debug)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self { issues: Vec::new() }
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
    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Check balanced delimiters in template.
pub fn check_delimiters(template: &str) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let mut brace_depth: i32 = 0;
    let mut angle_depth: i32 = 0;

    for (i, ch) in template.chars().enumerate() {
        match ch {
            '{' => brace_depth += 1,
            '}' => {
                brace_depth -= 1;
                if brace_depth < 0 {
                    issues.push(ValidationIssue::error("E001", "unmatched closing brace").at(i));
                    brace_depth = 0;
                }
            }
            '<' => angle_depth += 1,
            '>' => {
                angle_depth -= 1;
                if angle_depth < 0 {
                    issues.push(
                        ValidationIssue::error("E002", "unmatched closing angle bracket").at(i),
                    );
                    angle_depth = 0;
                }
            }
            _ => {}
        }
    }

    if brace_depth > 0 {
        issues.push(ValidationIssue::error("E001", format!("{brace_depth} unclosed brace(s)")));
    }
    if angle_depth > 0 {
        issues.push(ValidationIssue::error(
            "E002",
            format!("{angle_depth} unclosed angle bracket(s)"),
        ));
    }
    issues
}

/// Known placeholder names.
const KNOWN_PLACEHOLDERS: &[&str] = &[
    "system",
    "user",
    "assistant",
    "content",
    "message",
    "input",
    "output",
    "instruction",
    "response",
    "context",
];

/// Check placeholder usage.
pub fn check_placeholders(template: &str) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Find {{placeholder}} patterns
    let mut i = 0;
    let bytes = template.as_bytes();
    while i + 3 < bytes.len() {
        if bytes[i] == b'{' && bytes[i + 1] == b'{' {
            if let Some(end) = template[i + 2..].find("}}") {
                let name = &template[i + 2..i + 2 + end];
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    issues.push(ValidationIssue::error("E003", "empty placeholder").at(i));
                } else if !KNOWN_PLACEHOLDERS.contains(&trimmed) {
                    issues.push(
                        ValidationIssue::info("I001", format!("custom placeholder: {trimmed}"))
                            .at(i),
                    );
                }
                i = i + 2 + end + 2;
            } else {
                issues.push(ValidationIssue::error("E004", "unclosed placeholder").at(i));
                break;
            }
        } else {
            i += 1;
        }
    }
    issues
}

/// Check template length vs token budget.
pub fn check_token_budget(
    template: &str,
    max_tokens: usize,
    chars_per_token: f64,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    let est_tokens = (template.len() as f64 / chars_per_token).ceil() as usize;

    if est_tokens > max_tokens {
        issues.push(ValidationIssue::error(
            "E005",
            format!("template ~{est_tokens} tokens exceeds budget of {max_tokens}"),
        ));
    } else if est_tokens as f64 > max_tokens as f64 * 0.8 {
        issues.push(ValidationIssue::warning(
            "W001",
            format!("template ~{est_tokens} tokens uses >80% of {max_tokens} budget"),
        ));
    }
    issues
}

/// Check for common structural issues.
pub fn check_structure(template: &str) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    if template.trim().is_empty() {
        issues.push(ValidationIssue::error("E006", "template is empty"));
    }

    if template.len() > 100_000 {
        issues.push(ValidationIssue::warning("W002", "template exceeds 100K chars"));
    }

    // Check for double newlines (excessive spacing)
    if template.contains("\n\n\n\n") {
        issues.push(ValidationIssue::warning("W003", "excessive blank lines in template"));
    }

    issues
}

/// Run all validators on a template.
pub fn validate_template(template: &str, max_tokens: Option<usize>) -> ValidationReport {
    let mut report = ValidationReport::new();
    report.issues.extend(check_structure(template));
    report.issues.extend(check_delimiters(template));
    report.issues.extend(check_placeholders(template));
    if let Some(max) = max_tokens {
        report.issues.extend(check_token_budget(template, max, 4.0));
    }
    report.issues.sort_by(|a, b| b.severity.cmp(&a.severity));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_template() {
        let report = validate_template("<|system|>\n{{system}}\n<|user|>\n{{user}}", None);
        assert!(report.is_valid());
    }

    #[test]
    fn test_unmatched_brace() {
        let issues = check_delimiters("hello { world");
        assert!(issues.iter().any(|i| i.code == "E001"));
    }

    #[test]
    fn test_balanced_delimiters() {
        let issues = check_delimiters("<a>{b}</a>");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_empty_placeholder() {
        let issues = check_placeholders("test {{}} end");
        assert!(issues.iter().any(|i| i.code == "E003"));
    }

    #[test]
    fn test_known_placeholder() {
        let issues = check_placeholders("{{system}} says {{user}}");
        assert!(issues.iter().all(|i| i.severity != Severity::Error));
    }

    #[test]
    fn test_custom_placeholder() {
        let issues = check_placeholders("{{my_custom}}");
        assert!(issues.iter().any(|i| i.code == "I001"));
    }

    #[test]
    fn test_unclosed_placeholder() {
        let issues = check_placeholders("test {{ no end");
        assert!(issues.iter().any(|i| i.code == "E004"));
    }

    #[test]
    fn test_token_budget_exceeded() {
        let long = "x".repeat(10000);
        let issues = check_token_budget(&long, 100, 4.0);
        assert!(issues.iter().any(|i| i.code == "E005"));
    }

    #[test]
    fn test_token_budget_warning() {
        let text = "x".repeat(340);
        let issues = check_token_budget(&text, 100, 4.0);
        assert!(issues.iter().any(|i| i.code == "W001"));
    }

    #[test]
    fn test_empty_template() {
        let issues = check_structure("   ");
        assert!(issues.iter().any(|i| i.code == "E006"));
    }

    #[test]
    fn test_full_validation() {
        let report = validate_template("{{system}}\n{{user}}", Some(1000));
        assert!(report.is_valid());
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn test_report_counts() {
        let report = validate_template("{{ }}", None);
        assert!(report.has_errors());
        assert!(report.error_count() > 0);
    }
}
