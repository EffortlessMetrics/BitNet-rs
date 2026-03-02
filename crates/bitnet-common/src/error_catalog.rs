//! Error catalog.
//!
//! Structured error codes and categories for the BitNet inference stack.

/// Error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    ModelLoading,
    Tokenization,
    Inference,
    Quantization,
    Memory,
    Configuration,
    IO,
    Runtime,
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
            Self::IO => "io",
            Self::Runtime => "runtime",
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            Self::ModelLoading => "ML",
            Self::Tokenization => "TK",
            Self::Inference => "IF",
            Self::Quantization => "QT",
            Self::Memory => "MM",
            Self::Configuration => "CF",
            Self::IO => "IO",
            Self::Runtime => "RT",
        }
    }
}

/// Structured error code.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorCode {
    pub category: ErrorCategory,
    pub number: u32,
    pub name: String,
    pub description: String,
    pub recoverable: bool,
}

impl ErrorCode {
    pub fn code_string(&self) -> String {
        format!("{}{:04}", self.category.prefix(), self.number)
    }
}

/// Error catalog containing all known errors.
#[derive(Debug, Clone)]
pub struct ErrorCatalog {
    errors: Vec<ErrorCode>,
}

impl Default for ErrorCatalog {
    fn default() -> Self {
        Self::standard()
    }
}

impl ErrorCatalog {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn add(&mut self, code: ErrorCode) {
        self.errors.push(code);
    }

    pub fn by_category(&self, cat: ErrorCategory) -> Vec<&ErrorCode> {
        self.errors.iter().filter(|e| e.category == cat).collect()
    }

    pub fn by_code(&self, code_str: &str) -> Option<&ErrorCode> {
        self.errors.iter().find(|e| e.code_string() == code_str)
    }

    pub fn count(&self) -> usize {
        self.errors.len()
    }

    pub fn recoverable(&self) -> Vec<&ErrorCode> {
        self.errors.iter().filter(|e| e.recoverable).collect()
    }

    /// Build the standard error catalog.
    pub fn standard() -> Self {
        let mut cat = Self::new();

        cat.add(ErrorCode {
            category: ErrorCategory::ModelLoading,
            number: 1,
            name: "model_not_found".into(),
            description: "Model file not found at specified path".into(),
            recoverable: false,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::ModelLoading,
            number: 2,
            name: "invalid_format".into(),
            description: "Model file format is invalid or corrupted".into(),
            recoverable: false,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::ModelLoading,
            number: 3,
            name: "unsupported_arch".into(),
            description: "Model architecture is not supported".into(),
            recoverable: false,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Tokenization,
            number: 1,
            name: "tokenizer_not_found".into(),
            description: "Tokenizer file not found".into(),
            recoverable: false,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Tokenization,
            number: 2,
            name: "encoding_failed".into(),
            description: "Failed to encode input text".into(),
            recoverable: true,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Inference,
            number: 1,
            name: "context_exceeded".into(),
            description: "Input exceeds maximum context length".into(),
            recoverable: true,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Inference,
            number: 2,
            name: "generation_failed".into(),
            description: "Token generation failed".into(),
            recoverable: true,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Memory,
            number: 1,
            name: "out_of_memory".into(),
            description: "Insufficient memory for operation".into(),
            recoverable: false,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::Configuration,
            number: 1,
            name: "invalid_config".into(),
            description: "Invalid configuration parameter".into(),
            recoverable: true,
        });
        cat.add(ErrorCode {
            category: ErrorCategory::IO,
            number: 1,
            name: "file_read_error".into(),
            description: "Failed to read file".into(),
            recoverable: false,
        });

        cat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_catalog() {
        let cat = ErrorCatalog::standard();
        assert!(cat.count() >= 10);
    }

    #[test]
    fn test_code_string() {
        let code = ErrorCode {
            category: ErrorCategory::ModelLoading,
            number: 1,
            name: "test".into(),
            description: "test".into(),
            recoverable: false,
        };
        assert_eq!(code.code_string(), "ML0001");
    }

    #[test]
    fn test_by_category() {
        let cat = ErrorCatalog::standard();
        let ml = cat.by_category(ErrorCategory::ModelLoading);
        assert!(ml.len() >= 3);
    }

    #[test]
    fn test_by_code() {
        let cat = ErrorCatalog::standard();
        let err = cat.by_code("ML0001").unwrap();
        assert_eq!(err.name, "model_not_found");
    }

    #[test]
    fn test_recoverable() {
        let cat = ErrorCatalog::standard();
        let rec = cat.recoverable();
        assert!(!rec.is_empty());
    }

    #[test]
    fn test_category_str() {
        assert_eq!(ErrorCategory::Inference.as_str(), "inference");
        assert_eq!(ErrorCategory::Memory.prefix(), "MM");
    }

    #[test]
    fn test_by_code_missing() {
        let cat = ErrorCatalog::standard();
        assert!(cat.by_code("XX9999").is_none());
    }

    #[test]
    fn test_empty_catalog() {
        let cat = ErrorCatalog::new();
        assert_eq!(cat.count(), 0);
    }

    #[test]
    fn test_default() {
        let cat = ErrorCatalog::default();
        assert!(cat.count() > 0);
    }

    #[test]
    fn test_io_category() {
        let cat = ErrorCatalog::standard();
        let io = cat.by_category(ErrorCategory::IO);
        assert!(!io.is_empty());
    }
}
