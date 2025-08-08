use std::collections::HashMap;
use std::path::PathBuf;

use super::{ModelSize, PromptCategory, TestModel, TestPrompt};
use crate::common::{ModelFormat, ModelType, TestResult};

/// Predefined test datasets for common testing scenarios
pub struct TestDatasets;

impl TestDatasets {
    /// Get a comprehensive set of test models covering different sizes and formats
    pub fn comprehensive_models() -> Vec<TestModel> {
        vec![
            // Tiny models for quick testing
            TestModel::new(
                "tiny-bitnet-1b",
                "Tiny BitNet 1B",
                ModelSize::Tiny,
                ModelFormat::Gguf,
                ModelType::BitNet,
            )
            .with_file_size(50 * 1024 * 1024) // 50MB
            .with_checksum("a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456")
            .with_description("Tiny BitNet model for basic functionality testing")
            .with_tags(vec!["tiny".to_string(), "bitnet".to_string(), "quick".to_string()]),
            TestModel::new(
                "tiny-transformer-125m",
                "Tiny Transformer 125M",
                ModelSize::Tiny,
                ModelFormat::SafeTensors,
                ModelType::Transformer,
            )
            .with_file_size(75 * 1024 * 1024) // 75MB
            .with_checksum("b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567")
            .with_description("Tiny transformer for comparison testing")
            .with_tags(vec!["tiny".to_string(), "transformer".to_string(), "comparison".to_string()]),
            // Small models for integration testing
            TestModel::new(
                "small-bitnet-3b",
                "Small BitNet 3B",
                ModelSize::Small,
                ModelFormat::Gguf,
                ModelType::BitNet,
            )
            .with_file_size(400 * 1024 * 1024) // 400MB
            .with_checksum("c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678")
            .with_description("Small BitNet model for integration testing")
            .with_tags(vec!["small".to_string(), "bitnet".to_string(), "integration".to_string()]),
            TestModel::new(
                "small-llama-7b",
                "Small LLaMA 7B",
                ModelSize::Small,
                ModelFormat::Gguf,
                ModelType::Transformer,
            )
            .with_file_size(800 * 1024 * 1024) // 800MB
            .with_checksum("d4e5f6789012345678901234567890abcdef1234567890abcdef123456789")
            .with_description("Small LLaMA model for performance comparison")
            .with_tags(vec!["small".to_string(), "llama".to_string(), "performance".to_string()]),
            // Medium models for performance testing
            TestModel::new(
                "medium-bitnet-13b",
                "Medium BitNet 13B",
                ModelSize::Medium,
                ModelFormat::Gguf,
                ModelType::BitNet,
            )
            .with_file_size(2 * 1024 * 1024 * 1024) // 2GB
            .with_checksum("e5f6789012345678901234567890abcdef1234567890abcdef1234567890")
            .with_description("Medium BitNet model for comprehensive performance testing")
            .with_tags(vec!["medium".to_string(), "bitnet".to_string(), "performance".to_string()]),
            // Edge case models
            TestModel::new(
                "edge-minimal",
                "Minimal Edge Case Model",
                ModelSize::Tiny,
                ModelFormat::Gguf,
                ModelType::BitNet,
            )
            .with_file_size(1024) // 1KB - extremely small
            .with_checksum("f6789012345678901234567890abcdef1234567890abcdef12345678901")
            .with_description("Minimal model for edge case testing")
            .with_tags(vec!["edge".to_string(), "minimal".to_string(), "stress".to_string()]),
        ]
    }

    /// Get a comprehensive set of test prompts covering different categories
    pub fn comprehensive_prompts() -> Vec<TestPrompt> {
        vec![
            // Basic prompts
            TestPrompt::new(
                "basic-greeting",
                "Hello! How are you doing today?",
                PromptCategory::Basic,
            )
            .with_tags(vec!["basic".to_string(), "greeting".to_string()]),

            TestPrompt::new(
                "basic-completion",
                "The quick brown fox jumps over the",
                PromptCategory::Basic,
            )
            .with_tags(vec!["basic".to_string(), "completion".to_string()]),

            // Question answering
            TestPrompt::new(
                "qa-geography",
                "What is the capital city of Japan?",
                PromptCategory::QuestionAnswering,
            )
            .with_tags(vec!["qa".to_string(), "geography".to_string(), "factual".to_string()]),

            TestPrompt::new(
                "qa-science",
                "How does photosynthesis work in plants?",
                PromptCategory::QuestionAnswering,
            )
            .with_tags(vec!["qa".to_string(), "science".to_string(), "biology".to_string()]),

            // Code generation
            TestPrompt::new(
                "code-python-hello",
                "Write a Python function that prints 'Hello, World!' to the console.",
                PromptCategory::CodeGeneration,
            )
            .with_tags(vec!["code".to_string(), "python".to_string(), "beginner".to_string()]),

            TestPrompt::new(
                "code-rust-fibonacci",
                "Implement a Rust function that calculates the nth Fibonacci number using recursion.",
                PromptCategory::CodeGeneration,
            )
            .with_tags(vec!["code".to_string(), "rust".to_string(), "algorithms".to_string()]),

            // Mathematics
            TestPrompt::new(
                "math-arithmetic",
                "Calculate 127 + 384 - 56 * 3",
                PromptCategory::Mathematics,
            )
            .with_tags(vec!["math".to_string(), "arithmetic".to_string()]),

            TestPrompt::new(
                "math-algebra",
                "Solve for x: 3x + 7 = 22",
                PromptCategory::Mathematics,
            )
            .with_tags(vec!["math".to_string(), "algebra".to_string()]),

            // Creative writing
            TestPrompt::new(
                "creative-story",
                "Write a short story about a time traveler who accidentally changes history.",
                PromptCategory::Creative,
            )
            .with_tags(vec!["creative".to_string(), "story".to_string(), "scifi".to_string()]),

            TestPrompt::new(
                "creative-poem",
                "Compose a haiku about the changing seasons.",
                PromptCategory::Creative,
            )
            .with_tags(vec!["creative".to_string(), "poem".to_string(), "haiku".to_string()]),

            // Factual information
            TestPrompt::new(
                "factual-history",
                "Explain the causes and consequences of World War I.",
                PromptCategory::Factual,
            )
            .with_tags(vec!["factual".to_string(), "history".to_string(), "war".to_string()]),

            TestPrompt::new(
                "factual-technology",
                "What is artificial intelligence and how does it work?",
                PromptCategory::Factual,
            )
            .with_tags(vec!["factual".to_string(), "technology".to_string(), "ai".to_string()]),

            // Conversational
            TestPrompt::new(
                "conv-advice",
                "I'm feeling stressed about my upcoming job interview. Do you have any advice?",
                PromptCategory::Conversational,
            )
            .with_tags(vec!["conversational".to_string(), "advice".to_string(), "support".to_string()]),

            TestPrompt::new(
                "conv-casual",
                "What's your favorite type of music and why?",
                PromptCategory::Conversational,
            )
            .with_tags(vec!["conversational".to_string(), "casual".to_string(), "preferences".to_string()]),

            // Edge cases
            TestPrompt::new(
                "edge-empty",
                "",
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["edge".to_string(), "empty".to_string(), "stress".to_string()]),

            TestPrompt::new(
                "edge-very-long",
                &"This is a very long prompt that repeats the same content over and over again to test how the model handles extremely long inputs. ".repeat(50),
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["edge".to_string(), "long".to_string(), "stress".to_string()]),

            TestPrompt::new(
                "edge-special-chars",
                "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["edge".to_string(), "special".to_string(), "characters".to_string()]),

            TestPrompt::new(
                "edge-unicode",
                "🚀 Test with emojis and unicode: 你好世界 🌍 Здравствуй мир 🎉",
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["edge".to_string(), "unicode".to_string(), "emoji".to_string()]),

            // Benchmark prompts
            TestPrompt::new(
                "benchmark-throughput",
                "Generate exactly 100 words about the benefits of renewable energy.",
                PromptCategory::Benchmark,
            )
            .with_tags(vec!["benchmark".to_string(), "throughput".to_string(), "controlled".to_string()]),

            TestPrompt::new(
                "benchmark-reasoning",
                "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain your reasoning step by step.",
                PromptCategory::Benchmark,
            )
            .with_tags(vec!["benchmark".to_string(), "reasoning".to_string(), "logic".to_string()]),

            // Multilingual
            TestPrompt::new(
                "multilingual-translation",
                "Translate the following English sentence to Spanish: 'The weather is beautiful today.'",
                PromptCategory::Multilingual,
            )
            .with_tags(vec!["multilingual".to_string(), "translation".to_string(), "spanish".to_string()]),

            TestPrompt::new(
                "multilingual-mixed",
                "Explain the concept of 'saudade' (Portuguese) and find similar concepts in other languages.",
                PromptCategory::Multilingual,
            )
            .with_tags(vec!["multilingual".to_string(), "cultural".to_string(), "concepts".to_string()]),
        ]
    }

    /// Get prompts suitable for quick smoke testing
    pub fn smoke_test_prompts() -> Vec<TestPrompt> {
        vec![
            TestPrompt::new("smoke-hello", "Hello!", PromptCategory::Basic)
                .with_tags(vec!["smoke".to_string(), "quick".to_string()]),
            TestPrompt::new("smoke-qa", "What is 2+2?", PromptCategory::Mathematics)
                .with_tags(vec!["smoke".to_string(), "quick".to_string()]),
            TestPrompt::new(
                "smoke-code",
                "Write hello world in Python",
                PromptCategory::CodeGeneration,
            )
            .with_tags(vec!["smoke".to_string(), "quick".to_string()]),
        ]
    }

    /// Get prompts for stress testing
    pub fn stress_test_prompts() -> Vec<TestPrompt> {
        vec![
            TestPrompt::new("stress-empty", "", PromptCategory::EdgeCase)
                .with_tags(vec!["stress".to_string(), "edge".to_string()]),
            TestPrompt::new(
                "stress-very-long",
                &"word ".repeat(2000), // 10,000 characters
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["stress".to_string(), "long".to_string()]),
            TestPrompt::new(
                "stress-repetitive",
                &"Please repeat this exact phrase: 'Testing repetitive input handling.' "
                    .repeat(100),
                PromptCategory::EdgeCase,
            )
            .with_tags(vec!["stress".to_string(), "repetitive".to_string()]),
            TestPrompt::new(
                "stress-complex-unicode",
                "🌟🚀💫🎉🔥⭐🌈🎯🎪🎭🎨🎵🎶🎸🎹🎺🎻🥁🎤🎧🎬🎮🎲🎯🎳🎪🎭🎨🎵🎶🎸🎹🎺🎻🥁🎤🎧🎬🎮🎲",
                PromptCategory::EdgeCase,
            )
            .with_tags(vec![
                "stress".to_string(),
                "unicode".to_string(),
                "emoji".to_string(),
            ]),
        ]
    }

    /// Get models suitable for quick testing
    pub fn quick_test_models() -> Vec<TestModel> {
        Self::comprehensive_models()
            .into_iter()
            .filter(|model| model.size == ModelSize::Tiny)
            .collect()
    }

    /// Get models for performance benchmarking
    pub fn benchmark_models() -> Vec<TestModel> {
        Self::comprehensive_models()
            .into_iter()
            .filter(|model| {
                matches!(model.size, ModelSize::Small | ModelSize::Medium)
                    && model.tags.contains(&"performance".to_string())
            })
            .collect()
    }

    /// Get prompts by category
    pub fn prompts_by_category(category: PromptCategory) -> Vec<TestPrompt> {
        Self::comprehensive_prompts()
            .into_iter()
            .filter(|prompt| prompt.category == category)
            .collect()
    }

    /// Get models by format
    pub fn models_by_format(format: ModelFormat) -> Vec<TestModel> {
        Self::comprehensive_models()
            .into_iter()
            .filter(|model| model.format == format)
            .collect()
    }

    /// Get models by type
    pub fn models_by_type(model_type: ModelType) -> Vec<TestModel> {
        Self::comprehensive_models()
            .into_iter()
            .filter(|model| model.model_type == model_type)
            .collect()
    }

    /// Create a test configuration for common scenarios
    pub fn create_test_scenario(scenario: TestScenario) -> TestScenarioConfig {
        match scenario {
            TestScenario::SmokeTest => TestScenarioConfig {
                name: "Smoke Test".to_string(),
                description: "Quick validation that basic functionality works".to_string(),
                models: Self::quick_test_models(),
                prompts: Self::smoke_test_prompts(),
                timeout_seconds: 60,
                max_parallel: 2,
            },

            TestScenario::IntegrationTest => TestScenarioConfig {
                name: "Integration Test".to_string(),
                description: "Comprehensive testing of component interactions".to_string(),
                models: Self::comprehensive_models()
                    .into_iter()
                    .filter(|m| matches!(m.size, ModelSize::Tiny | ModelSize::Small))
                    .collect(),
                prompts: Self::comprehensive_prompts()
                    .into_iter()
                    .filter(|p| !matches!(p.category, PromptCategory::EdgeCase))
                    .collect(),
                timeout_seconds: 300,
                max_parallel: 4,
            },

            TestScenario::PerformanceTest => TestScenarioConfig {
                name: "Performance Test".to_string(),
                description: "Performance benchmarking and optimization validation".to_string(),
                models: Self::benchmark_models(),
                prompts: Self::comprehensive_prompts()
                    .into_iter()
                    .filter(|p| matches!(p.category, PromptCategory::Benchmark))
                    .collect(),
                timeout_seconds: 600,
                max_parallel: 1, // Sequential for accurate performance measurement
            },

            TestScenario::StressTest => TestScenarioConfig {
                name: "Stress Test".to_string(),
                description: "Edge case and stress testing for robustness validation".to_string(),
                models: Self::comprehensive_models(),
                prompts: Self::stress_test_prompts(),
                timeout_seconds: 900,
                max_parallel: 8,
            },

            TestScenario::CrossValidation => TestScenarioConfig {
                name: "Cross Validation".to_string(),
                description: "Comparison between Rust and C++ implementations".to_string(),
                models: Self::comprehensive_models()
                    .into_iter()
                    .filter(|m| m.model_type == ModelType::BitNet)
                    .collect(),
                prompts: Self::comprehensive_prompts()
                    .into_iter()
                    .filter(|p| !matches!(p.category, PromptCategory::EdgeCase))
                    .take(20) // Limit for cross-validation
                    .collect(),
                timeout_seconds: 1200,
                max_parallel: 2,
            },
        }
    }
}

/// Predefined test scenarios
#[derive(Debug, Clone, Copy)]
pub enum TestScenario {
    /// Quick smoke test for basic functionality
    SmokeTest,
    /// Comprehensive integration testing
    IntegrationTest,
    /// Performance benchmarking
    PerformanceTest,
    /// Stress and edge case testing
    StressTest,
    /// Cross-implementation validation
    CrossValidation,
}

/// Configuration for a test scenario
#[derive(Debug, Clone)]
pub struct TestScenarioConfig {
    /// Name of the scenario
    pub name: String,
    /// Description of what this scenario tests
    pub description: String,
    /// Models to use in this scenario
    pub models: Vec<TestModel>,
    /// Prompts to use in this scenario
    pub prompts: Vec<TestPrompt>,
    /// Timeout in seconds for the entire scenario
    pub timeout_seconds: u64,
    /// Maximum number of parallel tests
    pub max_parallel: usize,
}

impl TestScenarioConfig {
    /// Get the estimated runtime for this scenario
    pub fn estimated_runtime_minutes(&self) -> u64 {
        let total_combinations = self.models.len() * self.prompts.len();
        let avg_test_time_seconds = 10; // Rough estimate
        let parallel_factor = self.max_parallel;

        let total_seconds = (total_combinations * avg_test_time_seconds) / parallel_factor;
        (total_seconds as u64 + 59) / 60 // Round up to minutes
    }

    /// Check if this scenario is suitable for CI environments
    pub fn is_ci_suitable(&self) -> bool {
        self.estimated_runtime_minutes() <= 15 && self.max_parallel <= 4
    }

    /// Get a subset of this scenario suitable for CI
    pub fn ci_subset(&self) -> Self {
        let mut subset = self.clone();
        subset.name = format!("{} (CI Subset)", self.name);

        // Limit models and prompts for CI
        subset.models = self.models.iter().take(2).cloned().collect();
        subset.prompts = self.prompts.iter().take(5).cloned().collect();
        subset.timeout_seconds = subset.timeout_seconds.min(300); // Max 5 minutes
        subset.max_parallel = subset.max_parallel.min(2);

        subset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_models() {
        let models = TestDatasets::comprehensive_models();
        assert!(!models.is_empty());

        // Check we have models of different sizes
        let has_tiny = models.iter().any(|m| m.size == ModelSize::Tiny);
        let has_small = models.iter().any(|m| m.size == ModelSize::Small);
        let has_medium = models.iter().any(|m| m.size == ModelSize::Medium);

        assert!(has_tiny);
        assert!(has_small);
        assert!(has_medium);

        // Check we have different formats
        let has_gguf = models.iter().any(|m| m.format == ModelFormat::Gguf);
        let has_safetensors = models.iter().any(|m| m.format == ModelFormat::SafeTensors);

        assert!(has_gguf);
        assert!(has_safetensors);
    }

    #[test]
    fn test_comprehensive_prompts() {
        let prompts = TestDatasets::comprehensive_prompts();
        assert!(!prompts.is_empty());

        // Check we have prompts of different categories
        let categories: std::collections::HashSet<_> = prompts.iter().map(|p| p.category).collect();
        assert!(categories.len() > 5); // Should have multiple categories

        // Check specific categories exist
        assert!(categories.contains(&PromptCategory::Basic));
        assert!(categories.contains(&PromptCategory::QuestionAnswering));
        assert!(categories.contains(&PromptCategory::CodeGeneration));
        assert!(categories.contains(&PromptCategory::EdgeCase));
    }

    #[test]
    fn test_smoke_test_prompts() {
        let prompts = TestDatasets::smoke_test_prompts();
        assert!(!prompts.is_empty());
        assert!(prompts.len() <= 5); // Should be small for quick testing

        // All should be tagged as smoke tests
        for prompt in &prompts {
            assert!(prompt.tags.contains(&"smoke".to_string()));
        }
    }

    #[test]
    fn test_stress_test_prompts() {
        let prompts = TestDatasets::stress_test_prompts();
        assert!(!prompts.is_empty());

        // Should include edge cases
        let has_empty = prompts.iter().any(|p| p.text.is_empty());
        let has_long = prompts.iter().any(|p| p.text.len() > 1000);

        assert!(has_empty);
        assert!(has_long);
    }

    #[test]
    fn test_quick_test_models() {
        let models = TestDatasets::quick_test_models();
        assert!(!models.is_empty());

        // All should be tiny models
        for model in &models {
            assert_eq!(model.size, ModelSize::Tiny);
        }
    }

    #[test]
    fn test_prompts_by_category() {
        let qa_prompts = TestDatasets::prompts_by_category(PromptCategory::QuestionAnswering);
        assert!(!qa_prompts.is_empty());

        // All should be QA prompts
        for prompt in &qa_prompts {
            assert_eq!(prompt.category, PromptCategory::QuestionAnswering);
        }
    }

    #[test]
    fn test_models_by_format() {
        let gguf_models = TestDatasets::models_by_format(ModelFormat::Gguf);
        assert!(!gguf_models.is_empty());

        // All should be GGUF format
        for model in &gguf_models {
            assert_eq!(model.format, ModelFormat::Gguf);
        }
    }

    #[test]
    fn test_test_scenarios() {
        let smoke_config = TestDatasets::create_test_scenario(TestScenario::SmokeTest);
        assert_eq!(smoke_config.name, "Smoke Test");
        assert!(!smoke_config.models.is_empty());
        assert!(!smoke_config.prompts.is_empty());
        assert!(smoke_config.is_ci_suitable());

        let stress_config = TestDatasets::create_test_scenario(TestScenario::StressTest);
        assert_eq!(stress_config.name, "Stress Test");
        assert!(stress_config.timeout_seconds > smoke_config.timeout_seconds);
    }

    #[test]
    fn test_scenario_config_methods() {
        let config = TestDatasets::create_test_scenario(TestScenario::IntegrationTest);

        let runtime = config.estimated_runtime_minutes();
        assert!(runtime > 0);

        let ci_subset = config.ci_subset();
        assert!(ci_subset.models.len() <= config.models.len());
        assert!(ci_subset.prompts.len() <= config.prompts.len());
        assert!(ci_subset.name.contains("CI Subset"));
    }
}
