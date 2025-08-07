use async_trait::async_trait;
use serde_json;
use std::path::Path;

use super::{TestModel, TestPrompt, ValidationLevel};
use crate::common::{TestError, TestResult};

/// Trait for validating test data
#[async_trait]
pub trait DataValidator: Send + Sync {
    /// Validate data from bytes
    async fn validate_bytes(&self, data: &[u8]) -> TestResult<ValidationResult>;

    /// Validate data from file
    async fn validate_file(&self, path: &Path) -> TestResult<ValidationResult> {
        let data = tokio::fs::read(path).await?;
        self.validate_bytes(&data).await
    }

    /// Get the name of this validator
    fn name(&self) -> &str;

    /// Get the validation level
    fn validation_level(&self) -> ValidationLevel;
}

/// Result of data validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the data is valid
    pub is_valid: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Issues found during validation
    pub issues: Vec<ValidationIssue>,
    /// Metadata about the validated data
    pub metadata: ValidationMetadata,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            score: 1.0,
            issues: Vec::new(),
            metadata: ValidationMetadata::default(),
        }
    }

    /// Create a failed validation result
    pub fn failure<S: Into<String>>(reason: S) -> Self {
        Self {
            is_valid: false,
            score: 0.0,
            issues: vec![ValidationIssue::error(reason)],
            metadata: ValidationMetadata::default(),
        }
    }

    /// Add an issue to the validation result
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);

        // Update validity and score based on issue severity
        match issue.severity {
            IssueSeverity::Error => {
                self.is_valid = false;
                self.score = 0.0;
            }
            IssueSeverity::Warning => {
                self.score = (self.score * 0.8).max(0.5);
            }
            IssueSeverity::Info => {
                // Info issues don't affect validity or score
            }
        }
    }

    /// Get all errors
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Error))
            .collect()
    }

    /// Get all warnings
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Warning))
            .collect()
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors().is_empty()
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings().is_empty()
    }
}

/// A validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: IssueSeverity,
    /// Description of the issue
    pub message: String,
    /// Location of the issue (if applicable)
    pub location: Option<String>,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Create an error issue
    pub fn error<S: Into<String>>(message: S) -> Self {
        Self {
            severity: IssueSeverity::Error,
            message: message.into(),
            location: None,
            suggestion: None,
        }
    }

    /// Create a warning issue
    pub fn warning<S: Into<String>>(message: S) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            message: message.into(),
            location: None,
            suggestion: None,
        }
    }

    /// Create an info issue
    pub fn info<S: Into<String>>(message: S) -> Self {
        Self {
            severity: IssueSeverity::Info,
            message: message.into(),
            location: None,
            suggestion: None,
        }
    }

    /// Add location information
    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Add suggestion
    pub fn with_suggestion<S: Into<String>>(mut self, suggestion: S) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Severity of a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Critical error that makes data invalid
    Error,
    /// Warning that indicates potential issues
    Warning,
    /// Informational message
    Info,
}

/// Metadata about validated data
#[derive(Debug, Clone, Default)]
pub struct ValidationMetadata {
    /// Size of the data in bytes
    pub size: usize,
    /// Detected format or type
    pub format: Option<String>,
    /// Encoding information
    pub encoding: Option<String>,
    /// Additional properties
    pub properties: std::collections::HashMap<String, String>,
}

/// Validator for model data
pub struct ModelValidator {
    validation_level: ValidationLevel,
}

impl ModelValidator {
    /// Create a new model validator
    pub fn new(validation_level: ValidationLevel) -> Self {
        Self { validation_level }
    }

    /// Validate a TestModel structure
    pub fn validate_model(&self, model: &TestModel) -> ValidationResult {
        let mut result = ValidationResult::success();
        result.metadata.format = Some("TestModel".to_string());

        // Basic validation
        if model.id.is_empty() {
            result.add_issue(ValidationIssue::error("Model ID cannot be empty"));
        }

        if model.name.is_empty() {
            result.add_issue(ValidationIssue::error("Model name cannot be empty"));
        }

        if model.file_size == 0 {
            result.add_issue(ValidationIssue::warning("Model file size is zero"));
        }

        if model.checksum.is_empty() {
            result.add_issue(ValidationIssue::warning("Model checksum is empty"));
        }

        // Size consistency validation
        if !model.size.fits_size(model.file_size) {
            result.add_issue(
                ValidationIssue::error(format!(
                    "Model file size {} doesn't match size category {:?}",
                    model.file_size, model.size
                ))
                .with_suggestion("Adjust file_size or size category"),
            );
        }

        // Parameter validation (if available)
        if let Some(param_count) = model.parameters.parameter_count {
            if param_count == 0 {
                result.add_issue(ValidationIssue::warning("Parameter count is zero"));
            }

            // Rough validation of parameter count vs file size
            let expected_size_range = match param_count {
                0..=100_000_000 => (0, 500_000_000), // Up to 500MB for small models
                100_000_001..=1_000_000_000 => (100_000_000, 5_000_000_000), // 100MB to 5GB
                _ => (1_000_000_000, u64::MAX),      // 1GB+
            };

            if model.file_size < expected_size_range.0
                || (expected_size_range.1 != u64::MAX && model.file_size > expected_size_range.1)
            {
                result.add_issue(ValidationIssue::warning(format!(
                    "File size {} seems inconsistent with parameter count {}",
                    model.file_size, param_count
                )));
            }
        }

        // Context length validation
        if let Some(context_length) = model.parameters.context_length {
            if context_length == 0 {
                result.add_issue(ValidationIssue::error("Context length cannot be zero"));
            }

            if context_length > 32768 {
                result.add_issue(ValidationIssue::warning("Very large context length"));
            }
        }

        // Vocabulary size validation
        if let Some(vocab_size) = model.parameters.vocab_size {
            if vocab_size == 0 {
                result.add_issue(ValidationIssue::error("Vocabulary size cannot be zero"));
            }

            if vocab_size < 1000 {
                result.add_issue(ValidationIssue::warning("Very small vocabulary size"));
            }
        }

        // Advanced validation for higher levels
        if matches!(self.validation_level, ValidationLevel::Strict) {
            // Strict validation checks
            if model.description.is_empty() {
                result.add_issue(ValidationIssue::warning("Model description is empty"));
            }

            if model.tags.is_empty() {
                result.add_issue(ValidationIssue::info("Model has no tags"));
            }

            if model.download_url.is_none() && model.local_path.is_none() {
                result.add_issue(ValidationIssue::warning(
                    "Model has no download URL or local path",
                ));
            }
        }

        result.metadata.size = std::mem::size_of_val(model);
        result
            .metadata
            .properties
            .insert("model_id".to_string(), model.id.clone());
        result
            .metadata
            .properties
            .insert("model_size".to_string(), format!("{:?}", model.size));

        result
    }
}

#[async_trait]
impl DataValidator for ModelValidator {
    async fn validate_bytes(&self, data: &[u8]) -> TestResult<ValidationResult> {
        // Try to parse as JSON first
        match serde_json::from_slice::<TestModel>(data) {
            Ok(model) => Ok(self.validate_model(&model)),
            Err(e) => {
                let mut result =
                    ValidationResult::failure(format!("Failed to parse model JSON: {}", e));
                result.metadata.size = data.len();
                result.metadata.format = Some("Unknown".to_string());

                // Try to determine if it's valid JSON at least
                if serde_json::from_slice::<serde_json::Value>(data).is_ok() {
                    result.add_issue(
                        ValidationIssue::error("Valid JSON but not a TestModel structure")
                            .with_suggestion("Check the JSON schema"),
                    );
                } else {
                    result.add_issue(ValidationIssue::error("Invalid JSON format"));
                }

                Ok(result)
            }
        }
    }

    fn name(&self) -> &str {
        "ModelValidator"
    }

    fn validation_level(&self) -> ValidationLevel {
        self.validation_level
    }
}

/// Validator for prompt data
pub struct PromptValidator {
    validation_level: ValidationLevel,
}

impl PromptValidator {
    /// Create a new prompt validator
    pub fn new(validation_level: ValidationLevel) -> Self {
        Self { validation_level }
    }

    /// Validate a TestPrompt structure
    pub fn validate_prompt(&self, prompt: &TestPrompt) -> ValidationResult {
        let mut result = ValidationResult::success();
        result.metadata.format = Some("TestPrompt".to_string());

        // Basic validation
        if prompt.id.is_empty() {
            result.add_issue(ValidationIssue::error("Prompt ID cannot be empty"));
        }

        if prompt.text.is_empty() && !matches!(prompt.category, super::PromptCategory::EdgeCase) {
            result.add_issue(ValidationIssue::error(
                "Prompt text cannot be empty (except for edge cases)",
            ));
        }

        // Length validation
        let char_length = prompt.char_length();
        if char_length > 10000 {
            result.add_issue(ValidationIssue::warning(
                "Very long prompt (>10k characters)",
            ));
        }

        // Category-specific validation
        match prompt.category {
            super::PromptCategory::QuestionAnswering => {
                if !prompt.text.contains('?') {
                    result.add_issue(
                        ValidationIssue::warning(
                            "Question answering prompt doesn't contain a question mark",
                        )
                        .with_suggestion("Add a question mark to make it a proper question"),
                    );
                }
            }
            super::PromptCategory::CodeGeneration => {
                let code_keywords = [
                    "write",
                    "create",
                    "implement",
                    "code",
                    "function",
                    "program",
                ];
                if !code_keywords
                    .iter()
                    .any(|&keyword| prompt.text.to_lowercase().contains(keyword))
                {
                    result.add_issue(ValidationIssue::warning(
                        "Code generation prompt doesn't contain typical code keywords",
                    ));
                }
            }
            super::PromptCategory::Mathematics => {
                let math_indicators = ["+", "-", "*", "/", "=", "calculate", "solve", "find"];
                if !math_indicators
                    .iter()
                    .any(|&indicator| prompt.text.to_lowercase().contains(indicator))
                {
                    result.add_issue(ValidationIssue::warning(
                        "Mathematics prompt doesn't contain typical math indicators",
                    ));
                }
            }
            _ => {} // No specific validation for other categories
        }

        // Expected response validation
        if let Some(min_length) = prompt.expected.min_length {
            if let Some(max_length) = prompt.expected.max_length {
                if min_length > max_length {
                    result.add_issue(ValidationIssue::error(
                        "Minimum expected length is greater than maximum",
                    ));
                }
            }
        }

        // Advanced validation for higher levels
        if matches!(self.validation_level, ValidationLevel::Strict) {
            if prompt.tags.is_empty() {
                result.add_issue(ValidationIssue::info("Prompt has no tags"));
            }

            if prompt.metadata.source.is_none() {
                result.add_issue(ValidationIssue::info("Prompt source is not specified"));
            }

            if prompt.expected.required_keywords.is_empty()
                && prompt.expected.quality_indicators.is_empty()
            {
                result.add_issue(ValidationIssue::info(
                    "No expected response criteria specified",
                ));
            }
        }

        result.metadata.size = std::mem::size_of_val(prompt);
        result
            .metadata
            .properties
            .insert("prompt_id".to_string(), prompt.id.clone());
        result
            .metadata
            .properties
            .insert("category".to_string(), format!("{:?}", prompt.category));
        result
            .metadata
            .properties
            .insert("char_length".to_string(), char_length.to_string());

        result
    }
}

#[async_trait]
impl DataValidator for PromptValidator {
    async fn validate_bytes(&self, data: &[u8]) -> TestResult<ValidationResult> {
        // Try to parse as single prompt first
        match serde_json::from_slice::<TestPrompt>(data) {
            Ok(prompt) => Ok(self.validate_prompt(&prompt)),
            Err(_) => {
                // Try to parse as array of prompts
                match serde_json::from_slice::<Vec<TestPrompt>>(data) {
                    Ok(prompts) => {
                        let mut combined_result = ValidationResult::success();
                        combined_result.metadata.size = data.len();
                        combined_result.metadata.format = Some("TestPrompt[]".to_string());

                        if prompts.is_empty() {
                            combined_result
                                .add_issue(ValidationIssue::warning("Empty prompt array"));
                        }

                        for (i, prompt) in prompts.iter().enumerate() {
                            let prompt_result = self.validate_prompt(prompt);
                            for issue in prompt_result.issues {
                                combined_result
                                    .add_issue(issue.with_location(format!("prompt[{}]", i)));
                            }
                        }

                        combined_result
                            .metadata
                            .properties
                            .insert("prompt_count".to_string(), prompts.len().to_string());

                        Ok(combined_result)
                    }
                    Err(e) => {
                        let mut result = ValidationResult::failure(format!(
                            "Failed to parse prompt JSON: {}",
                            e
                        ));
                        result.metadata.size = data.len();
                        result.metadata.format = Some("Unknown".to_string());
                        Ok(result)
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        "PromptValidator"
    }

    fn validation_level(&self) -> ValidationLevel {
        self.validation_level
    }
}

/// Generic JSON validator
pub struct JsonValidator {
    validation_level: ValidationLevel,
}

impl JsonValidator {
    /// Create a new JSON validator
    pub fn new(validation_level: ValidationLevel) -> Self {
        Self { validation_level }
    }
}

#[async_trait]
impl DataValidator for JsonValidator {
    async fn validate_bytes(&self, data: &[u8]) -> TestResult<ValidationResult> {
        let mut result = ValidationResult::success();
        result.metadata.size = data.len();
        result.metadata.format = Some("JSON".to_string());

        match serde_json::from_slice::<serde_json::Value>(data) {
            Ok(value) => {
                // Basic JSON structure validation
                match &value {
                    serde_json::Value::Object(obj) => {
                        if obj.is_empty() {
                            result.add_issue(ValidationIssue::info("Empty JSON object"));
                        }
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "object".to_string());
                        result
                            .metadata
                            .properties
                            .insert("keys".to_string(), obj.len().to_string());
                    }
                    serde_json::Value::Array(arr) => {
                        if arr.is_empty() {
                            result.add_issue(ValidationIssue::info("Empty JSON array"));
                        }
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "array".to_string());
                        result
                            .metadata
                            .properties
                            .insert("length".to_string(), arr.len().to_string());
                    }
                    serde_json::Value::String(s) => {
                        if s.is_empty() {
                            result.add_issue(ValidationIssue::info("Empty JSON string"));
                        }
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "string".to_string());
                    }
                    serde_json::Value::Number(_) => {
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "number".to_string());
                    }
                    serde_json::Value::Bool(_) => {
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "boolean".to_string());
                    }
                    serde_json::Value::Null => {
                        result.add_issue(ValidationIssue::info("JSON value is null"));
                        result
                            .metadata
                            .properties
                            .insert("type".to_string(), "null".to_string());
                    }
                }

                // Advanced validation for strict mode
                if matches!(self.validation_level, ValidationLevel::Strict) {
                    // Check for deeply nested structures
                    fn check_depth(value: &serde_json::Value, current_depth: usize) -> usize {
                        match value {
                            serde_json::Value::Object(obj) => obj
                                .values()
                                .map(|v| check_depth(v, current_depth + 1))
                                .max()
                                .unwrap_or(current_depth),
                            serde_json::Value::Array(arr) => arr
                                .iter()
                                .map(|v| check_depth(v, current_depth + 1))
                                .max()
                                .unwrap_or(current_depth),
                            _ => current_depth,
                        }
                    }

                    let depth = check_depth(&value, 0);
                    if depth > 10 {
                        result.add_issue(ValidationIssue::warning(format!(
                            "Very deep JSON structure (depth: {})",
                            depth
                        )));
                    }
                }
            }
            Err(e) => {
                result.add_issue(ValidationIssue::error(format!("Invalid JSON: {}", e)));

                // Try to provide more specific error information
                let error_msg = e.to_string();
                if error_msg.contains("EOF") {
                    result.add_issue(
                        ValidationIssue::error("Unexpected end of JSON input")
                            .with_suggestion("Check for missing closing brackets or quotes"),
                    );
                } else if error_msg.contains("trailing comma") {
                    result.add_issue(
                        ValidationIssue::error("Trailing comma in JSON")
                            .with_suggestion("Remove trailing commas"),
                    );
                }
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "JsonValidator"
    }

    fn validation_level(&self) -> ValidationLevel {
        self.validation_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{ModelFormat, ModelType};
    use crate::data::{ModelSize, PromptCategory, TestModel, TestPrompt};

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::success();
        assert!(result.is_valid);
        assert_eq!(result.score, 1.0);
        assert!(result.issues.is_empty());

        result.add_issue(ValidationIssue::warning("Test warning"));
        assert!(result.is_valid); // Warnings don't make it invalid
        assert!(result.score < 1.0);
        assert_eq!(result.warnings().len(), 1);

        result.add_issue(ValidationIssue::error("Test error"));
        assert!(!result.is_valid);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.errors().len(), 1);
    }

    #[test]
    fn test_validation_issue() {
        let error = ValidationIssue::error("Test error")
            .with_location("line 5")
            .with_suggestion("Fix the error");

        assert!(matches!(error.severity, IssueSeverity::Error));
        assert_eq!(error.message, "Test error");
        assert_eq!(error.location.as_deref(), Some("line 5"));
        assert_eq!(error.suggestion.as_deref(), Some("Fix the error"));
    }

    #[test]
    fn test_model_validator() {
        let validator = ModelValidator::new(ValidationLevel::Standard);

        // Valid model
        let valid_model = TestModel::new(
            "test-model",
            "Test Model",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(50 * 1024 * 1024); // 50MB fits Tiny category

        let result = validator.validate_model(&valid_model);
        assert!(result.is_valid);
        assert!(result.issues.is_empty());

        // Invalid model (empty ID)
        let invalid_model = TestModel::new(
            "",
            "Test Model",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        );

        let result = validator.validate_model(&invalid_model);
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_prompt_validator() {
        let validator = PromptValidator::new(ValidationLevel::Standard);

        // Valid prompt
        let valid_prompt = TestPrompt::new(
            "test-prompt",
            "What is the capital of France?",
            PromptCategory::QuestionAnswering,
        );

        let result = validator.validate_prompt(&valid_prompt);
        assert!(result.is_valid);

        // Invalid prompt (empty ID)
        let invalid_prompt = TestPrompt::new("", "Test prompt", PromptCategory::Basic);

        let result = validator.validate_prompt(&invalid_prompt);
        assert!(!result.is_valid);
        assert!(result.has_errors());

        // QA prompt without question mark (should generate warning)
        let qa_prompt = TestPrompt::new(
            "qa-no-question",
            "Tell me the capital of France",
            PromptCategory::QuestionAnswering,
        );

        let result = validator.validate_prompt(&qa_prompt);
        assert!(result.is_valid); // Still valid, just a warning
        assert!(result.has_warnings());
    }

    #[tokio::test]
    async fn test_json_validator() {
        let validator = JsonValidator::new(ValidationLevel::Standard);

        // Valid JSON
        let valid_json = r#"{"key": "value", "number": 42}"#;
        let result = validator
            .validate_bytes(valid_json.as_bytes())
            .await
            .unwrap();
        assert!(result.is_valid);
        assert_eq!(result.metadata.format.as_deref(), Some("JSON"));

        // Invalid JSON
        let invalid_json = r#"{"key": "value",}"#; // Trailing comma
        let result = validator
            .validate_bytes(invalid_json.as_bytes())
            .await
            .unwrap();
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[tokio::test]
    async fn test_model_validator_bytes() {
        let validator = ModelValidator::new(ValidationLevel::Standard);

        let model = TestModel::new(
            "test",
            "Test",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        );

        let json_data = serde_json::to_string(&model).unwrap();
        let result = validator
            .validate_bytes(json_data.as_bytes())
            .await
            .unwrap();
        assert!(result.is_valid);

        // Invalid JSON
        let invalid_json = "not json";
        let result = validator
            .validate_bytes(invalid_json.as_bytes())
            .await
            .unwrap();
        assert!(!result.is_valid);
    }
}
