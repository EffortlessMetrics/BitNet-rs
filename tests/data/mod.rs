// Test data definitions and utilities for BitNet.rs testing framework
//
// This module provides:
// - Standard test model definitions
// - Test prompt datasets
// - Data generation utilities
// - Test data validation

pub mod generators;
pub mod models;
pub mod prompts;
pub mod validators;

// Re-export commonly used types
pub use generators::{DataGenerator, ModelDataGenerator, PromptDataGenerator};
pub use models::{ModelSize, TestModel, TestModelRegistry};
pub use prompts::{PromptCategory, TestPrompt, TestPromptRegistry};
pub use validators::{DataValidator, ModelValidator, PromptValidator};

/// Standard test data configuration
pub struct TestDataConfig {
    /// Base directory for test data
    pub base_dir: std::path::PathBuf,
    /// Enable data generation
    pub enable_generation: bool,
    /// Maximum data size in bytes
    pub max_data_size: u64,
    /// Data validation level
    pub validation_level: ValidationLevel,
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            base_dir: std::path::PathBuf::from("tests/data"),
            enable_generation: true,
            max_data_size: 100 * 1024 * 1024, // 100 MB
            validation_level: ValidationLevel::Standard,
        }
    }
}

/// Level of data validation to perform
#[derive(Debug, Clone, Copy)]
pub enum ValidationLevel {
    /// Minimal validation (existence checks only)
    Minimal,
    /// Standard validation (format and basic content checks)
    Standard,
    /// Strict validation (comprehensive content and consistency checks)
    Strict,
}

/// Initialize test data registries
pub async fn init_test_data() -> crate::common::TestResult<(TestModelRegistry, TestPromptRegistry)>
{
    let model_registry = TestModelRegistry::new().await?;
    let prompt_registry = TestPromptRegistry::new().await?;

    Ok((model_registry, prompt_registry))
}
