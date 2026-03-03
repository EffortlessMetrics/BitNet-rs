//! Rich error context for inference errors.
//!
//! Structured error context with source location, model info,
//! and suggested remediation.

use std::fmt;

/// Error severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Warning,
    Error,
    Fatal,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Fatal => "fatal",
        }
    }
}

/// Error category for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    ModelLoading,
    Tokenization,
    Inference,
    Quantization,
    Memory,
    Configuration,
    Io,
    Internal,
}

impl ErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ModelLoading => "model_loading",
            Self::Tokenization => "tokenization",
            Self::Inference => "inference",
            Self::Quantization => "quantization",
            Self::Memory => "memory",
            Self::Configuration => "configuration",
            Self::Io => "io",
            Self::Internal => "internal",
        }
    }
}

/// Rich error context.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub category: ErrorCategory,
    pub severity: Severity,
    pub message: String,
    pub detail: Option<String>,
    pub suggestion: Option<String>,
    pub source_location: Option<String>,
}

impl ErrorContext {
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            severity: Severity::Error,
            message: message.into(),
            detail: None,
            suggestion: None,
            source_location: None,
        }
    }

    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn with_location(mut self, file: &str, line: u32) -> Self {
        self.source_location = Some(format!("{file}:{line}"));
        self
    }

    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Fatal
    }

    pub fn is_recoverable(&self) -> bool {
        self.severity == Severity::Warning
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity.as_str(), self.category.as_str(), self.message)?;
        if let Some(ref detail) = self.detail {
            write!(f, "\n  detail: {detail}")?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  suggestion: {suggestion}")?;
        }
        if let Some(ref loc) = self.source_location {
            write!(f, "\n  at: {loc}")?;
        }
        Ok(())
    }
}

/// Convenience constructors.
pub fn model_error(msg: impl Into<String>) -> ErrorContext {
    ErrorContext::new(ErrorCategory::ModelLoading, msg)
}

pub fn config_error(msg: impl Into<String>) -> ErrorContext {
    ErrorContext::new(ErrorCategory::Configuration, msg)
}

pub fn memory_error(msg: impl Into<String>) -> ErrorContext {
    ErrorContext::new(ErrorCategory::Memory, msg).with_severity(Severity::Fatal)
}

pub fn inference_warning(msg: impl Into<String>) -> ErrorContext {
    ErrorContext::new(ErrorCategory::Inference, msg).with_severity(Severity::Warning)
}

/// Error accumulator for collecting multiple errors.
#[derive(Debug, Default)]
pub struct ErrorCollector {
    errors: Vec<ErrorContext>,
}

impl ErrorCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, error: ErrorContext) {
        self.errors.push(error);
    }

    pub fn has_errors(&self) -> bool {
        self.errors.iter().any(|e| e.severity >= Severity::Error)
    }

    pub fn has_fatal(&self) -> bool {
        self.errors.iter().any(|e| e.severity == Severity::Fatal)
    }

    pub fn error_count(&self) -> usize {
        self.errors.iter().filter(|e| e.severity >= Severity::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.errors.iter().filter(|e| e.severity == Severity::Warning).count()
    }

    pub fn errors(&self) -> &[ErrorContext] {
        &self.errors
    }

    pub fn into_errors(self) -> Vec<ErrorContext> {
        self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let err = ErrorContext::new(ErrorCategory::ModelLoading, "file not found");
        assert_eq!(err.category, ErrorCategory::ModelLoading);
        assert_eq!(err.severity, Severity::Error);
    }

    #[test]
    fn test_builder_chain() {
        let err = ErrorContext::new(ErrorCategory::Inference, "shape mismatch")
            .with_severity(Severity::Fatal)
            .with_detail("expected [3,4], got [3,5]")
            .with_suggestion("check input dimensions")
            .with_location("model.rs", 42);
        assert!(err.is_fatal());
        assert!(err.detail.is_some());
        assert!(err.suggestion.is_some());
        assert_eq!(err.source_location.as_deref(), Some("model.rs:42"));
    }

    #[test]
    fn test_display() {
        let err = model_error("bad weights").with_detail("corrupt header");
        let s = format!("{err}");
        assert!(s.contains("model_loading"));
        assert!(s.contains("bad weights"));
        assert!(s.contains("corrupt header"));
    }

    #[test]
    fn test_convenience_constructors() {
        assert_eq!(model_error("x").category, ErrorCategory::ModelLoading);
        assert_eq!(config_error("x").category, ErrorCategory::Configuration);
        assert!(memory_error("x").is_fatal());
        assert!(inference_warning("x").is_recoverable());
    }

    #[test]
    fn test_error_collector() {
        let mut collector = ErrorCollector::new();
        collector.push(inference_warning("slow"));
        collector.push(model_error("missing layer"));
        assert!(collector.has_errors());
        assert!(!collector.has_fatal());
        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 1);
    }

    #[test]
    fn test_collector_fatal() {
        let mut collector = ErrorCollector::new();
        collector.push(memory_error("OOM"));
        assert!(collector.has_fatal());
    }

    #[test]
    fn test_empty_collector() {
        let collector = ErrorCollector::new();
        assert!(!collector.has_errors());
        assert_eq!(collector.error_count(), 0);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Fatal > Severity::Error);
        assert!(Severity::Error > Severity::Warning);
    }

    #[test]
    fn test_category_as_str() {
        assert_eq!(ErrorCategory::Inference.as_str(), "inference");
        assert_eq!(ErrorCategory::Memory.as_str(), "memory");
    }

    #[test]
    fn test_into_errors() {
        let mut collector = ErrorCollector::new();
        collector.push(model_error("a"));
        collector.push(model_error("b"));
        let errors = collector.into_errors();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_recoverable() {
        let warn = inference_warning("test");
        assert!(warn.is_recoverable());
        assert!(!warn.is_fatal());
    }

    #[test]
    fn test_severity_as_str() {
        assert_eq!(Severity::Warning.as_str(), "warning");
        assert_eq!(Severity::Error.as_str(), "error");
        assert_eq!(Severity::Fatal.as_str(), "fatal");
    }
}
