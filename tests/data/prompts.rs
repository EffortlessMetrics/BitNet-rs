use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::common::TestResult;

/// Definition of a test prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPrompt {
    /// Unique identifier for the prompt
    pub id: String,
    /// The prompt text
    pub text: String,
    /// Category of the prompt
    pub category: PromptCategory,
    /// Expected characteristics of the response
    pub expected: ExpectedResponse,
    /// Metadata about the prompt
    pub metadata: PromptMetadata,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TestPrompt {
    /// Create a new test prompt
    pub fn new<S: Into<String>>(id: S, text: S, category: PromptCategory) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            category,
            expected: ExpectedResponse::default(),
            metadata: PromptMetadata::default(),
            tags: Vec::new(),
        }
    }

    /// Set expected response characteristics
    pub fn with_expected(mut self, expected: ExpectedResponse) -> Self {
        self.expected = expected;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: PromptMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Get the length of the prompt in characters
    pub fn char_length(&self) -> usize {
        self.text.chars().count()
    }

    /// Get the approximate token count (rough estimate)
    pub fn estimated_token_count(&self) -> usize {
        // Rough estimate: 1 token per 4 characters
        (self.char_length() + 3) / 4
    }

    /// Check if this is a long prompt
    pub fn is_long(&self) -> bool {
        self.estimated_token_count() > 1000
    }

    /// Check if this prompt contains special characters or formatting
    pub fn has_special_formatting(&self) -> bool {
        self.text.contains('\n')
            || self.text.contains('\t')
            || self.text.contains("```")
            || self.text.contains("**")
            || self.text.contains("*")
    }
}

/// Categories of test prompts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptCategory {
    /// Basic text generation
    Basic,
    /// Question answering
    QuestionAnswering,
    /// Code generation
    CodeGeneration,
    /// Mathematical reasoning
    Mathematics,
    /// Creative writing
    Creative,
    /// Factual information
    Factual,
    /// Conversational
    Conversational,
    /// Edge cases and stress testing
    EdgeCase,
    /// Performance benchmarking
    Benchmark,
    /// Multilingual content
    Multilingual,
}

impl PromptCategory {
    /// Get a human-readable description of the category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Basic => "Basic text generation and completion",
            Self::QuestionAnswering => "Question answering and comprehension",
            Self::CodeGeneration => "Code generation and programming tasks",
            Self::Mathematics => "Mathematical reasoning and calculations",
            Self::Creative => "Creative writing and storytelling",
            Self::Factual => "Factual information and knowledge retrieval",
            Self::Conversational => "Conversational and dialogue tasks",
            Self::EdgeCase => "Edge cases and stress testing scenarios",
            Self::Benchmark => "Performance benchmarking prompts",
            Self::Multilingual => "Multilingual and translation tasks",
        }
    }

    /// Get typical expected response length for this category
    pub fn typical_response_length(&self) -> (usize, usize) {
        match self {
            Self::Basic => (50, 200),
            Self::QuestionAnswering => (20, 150),
            Self::CodeGeneration => (100, 500),
            Self::Mathematics => (10, 100),
            Self::Creative => (200, 1000),
            Self::Factual => (50, 300),
            Self::Conversational => (20, 200),
            Self::EdgeCase => (0, 100),
            Self::Benchmark => (100, 500),
            Self::Multilingual => (50, 300),
        }
    }
}

/// Expected characteristics of the response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResponse {
    /// Minimum expected response length in tokens
    pub min_length: Option<usize>,
    /// Maximum expected response length in tokens
    pub max_length: Option<usize>,
    /// Expected response time in milliseconds
    pub max_response_time_ms: Option<u64>,
    /// Whether the response should contain specific keywords
    pub required_keywords: Vec<String>,
    /// Whether the response should avoid specific content
    pub forbidden_content: Vec<String>,
    /// Expected response quality indicators
    pub quality_indicators: Vec<QualityIndicator>,
}

impl Default for ExpectedResponse {
    fn default() -> Self {
        Self {
            min_length: None,
            max_length: None,
            max_response_time_ms: None,
            required_keywords: Vec::new(),
            forbidden_content: Vec::new(),
            quality_indicators: Vec::new(),
        }
    }
}

/// Quality indicators for response validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIndicator {
    /// Response should be coherent and well-structured
    Coherent,
    /// Response should be factually accurate
    Accurate,
    /// Response should be relevant to the prompt
    Relevant,
    /// Response should be complete (not cut off)
    Complete,
    /// Response should follow proper grammar and syntax
    WellFormed,
    /// Response should be creative and original
    Creative,
    /// Response should demonstrate reasoning
    Logical,
}

/// Metadata about the prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMetadata {
    /// Source of the prompt (e.g., "HuggingFace", "Custom", "Benchmark")
    pub source: Option<String>,
    /// Language of the prompt
    pub language: Option<String>,
    /// Difficulty level (1-5)
    pub difficulty: Option<u8>,
    /// Domain or subject area
    pub domain: Option<String>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl Default for PromptMetadata {
    fn default() -> Self {
        Self {
            source: None,
            language: Some("en".to_string()),
            difficulty: None,
            domain: None,
            custom: HashMap::new(),
        }
    }
}

/// Registry for managing test prompts
pub struct TestPromptRegistry {
    prompts: HashMap<String, TestPrompt>,
    by_category: HashMap<PromptCategory, Vec<String>>,
}

impl TestPromptRegistry {
    /// Create a new prompt registry with built-in prompts
    pub async fn new() -> TestResult<Self> {
        let mut registry = Self {
            prompts: HashMap::new(),
            by_category: HashMap::new(),
        };

        registry.load_builtin_prompts().await?;
        Ok(registry)
    }

    /// Register a test prompt
    pub fn register(&mut self, prompt: TestPrompt) {
        // Add to category index
        self.by_category
            .entry(prompt.category)
            .or_insert_with(Vec::new)
            .push(prompt.id.clone());

        // Add to main registry
        self.prompts.insert(prompt.id.clone(), prompt);
    }

    /// Get a prompt by ID
    pub fn get(&self, id: &str) -> Option<&TestPrompt> {
        self.prompts.get(id)
    }

    /// Get all prompts
    pub fn all(&self) -> Vec<&TestPrompt> {
        self.prompts.values().collect()
    }

    /// Get prompts by category
    pub fn by_category(&self, category: PromptCategory) -> Vec<&TestPrompt> {
        self.by_category
            .get(&category)
            .map(|ids| ids.iter().filter_map(|id| self.prompts.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get prompts with specific tags
    pub fn by_tags(&self, tags: &[String]) -> Vec<&TestPrompt> {
        self.prompts
            .values()
            .filter(|p| tags.iter().any(|tag| p.tags.contains(tag)))
            .collect()
    }

    /// Get short prompts (for quick testing)
    pub fn short_prompts(&self) -> Vec<&TestPrompt> {
        self.prompts
            .values()
            .filter(|p| p.estimated_token_count() <= 50)
            .collect()
    }

    /// Get long prompts (for stress testing)
    pub fn long_prompts(&self) -> Vec<&TestPrompt> {
        self.prompts.values().filter(|p| p.is_long()).collect()
    }

    /// Get prompts by difficulty level
    pub fn by_difficulty(&self, difficulty: u8) -> Vec<&TestPrompt> {
        self.prompts
            .values()
            .filter(|p| p.metadata.difficulty == Some(difficulty))
            .collect()
    }

    /// Get a random sample of prompts
    pub fn random_sample(&self, count: usize) -> Vec<&TestPrompt> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut prompts: Vec<&TestPrompt> = self.prompts.values().collect();
        prompts.shuffle(&mut rng);
        prompts.into_iter().take(count).collect()
    }

    /// Load built-in test prompts
    async fn load_builtin_prompts(&mut self) -> TestResult<()> {
        // Basic prompts
        let basic_hello = TestPrompt::new(
            "basic-hello",
            "Hello, how are you today?",
            PromptCategory::Basic,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(10),
            max_length: Some(100),
            max_response_time_ms: Some(1000),
            required_keywords: vec!["hello".to_string(), "good".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Coherent, QualityIndicator::Relevant],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(1),
            domain: Some("conversational".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "basic".to_string(),
            "greeting".to_string(),
            "simple".to_string(),
        ]);

        let basic_completion = TestPrompt::new(
            "basic-completion",
            "The quick brown fox",
            PromptCategory::Basic,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(5),
            max_length: Some(50),
            max_response_time_ms: Some(500),
            required_keywords: vec!["jumps".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Complete, QualityIndicator::Relevant],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(1),
            domain: Some("completion".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "basic".to_string(),
            "completion".to_string(),
            "classic".to_string(),
        ]);

        // Question answering prompts
        let qa_simple = TestPrompt::new(
            "qa-simple",
            "What is the capital of France?",
            PromptCategory::QuestionAnswering,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(1),
            max_length: Some(20),
            max_response_time_ms: Some(500),
            required_keywords: vec!["Paris".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Accurate, QualityIndicator::Relevant],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(1),
            domain: Some("geography".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "qa".to_string(),
            "factual".to_string(),
            "geography".to_string(),
        ]);

        // Code generation prompts
        let code_hello = TestPrompt::new(
            "code-hello-world",
            "Write a Python function that prints 'Hello, World!'",
            PromptCategory::CodeGeneration,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(20),
            max_length: Some(200),
            max_response_time_ms: Some(2000),
            required_keywords: vec!["def".to_string(), "print".to_string(), "Hello".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::WellFormed, QualityIndicator::Complete],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(2),
            domain: Some("programming".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "code".to_string(),
            "python".to_string(),
            "beginner".to_string(),
        ]);

        // Mathematical reasoning
        let math_simple = TestPrompt::new(
            "math-simple",
            "What is 15 + 27?",
            PromptCategory::Mathematics,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(1),
            max_length: Some(10),
            max_response_time_ms: Some(500),
            required_keywords: vec!["42".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Accurate, QualityIndicator::Logical],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(1),
            domain: Some("arithmetic".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "math".to_string(),
            "arithmetic".to_string(),
            "simple".to_string(),
        ]);

        // Creative writing
        let creative_story = TestPrompt::new(
            "creative-story-start",
            "Write the beginning of a story about a robot who discovers emotions.",
            PromptCategory::Creative,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(100),
            max_length: Some(500),
            max_response_time_ms: Some(5000),
            required_keywords: vec!["robot".to_string()],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Creative, QualityIndicator::Coherent],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(3),
            domain: Some("creative_writing".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "creative".to_string(),
            "story".to_string(),
            "robot".to_string(),
        ]);

        // Edge case prompts
        let edge_empty = TestPrompt::new("edge-empty", "", PromptCategory::EdgeCase)
            .with_expected(ExpectedResponse {
                min_length: Some(0),
                max_length: Some(100),
                max_response_time_ms: Some(1000),
                required_keywords: vec![],
                forbidden_content: vec![],
                quality_indicators: vec![],
            })
            .with_metadata(PromptMetadata {
                source: Some("builtin".to_string()),
                language: Some("en".to_string()),
                difficulty: Some(5),
                domain: Some("edge_case".to_string()),
                custom: HashMap::new(),
            })
            .with_tags(vec![
                "edge".to_string(),
                "empty".to_string(),
                "stress".to_string(),
            ]);

        let edge_long = TestPrompt::new(
            "edge-very-long",
            &"This is a very long prompt that repeats itself many times. ".repeat(100),
            PromptCategory::EdgeCase,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(10),
            max_length: Some(1000),
            max_response_time_ms: Some(10000),
            required_keywords: vec![],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Complete],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(5),
            domain: Some("edge_case".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "edge".to_string(),
            "long".to_string(),
            "stress".to_string(),
        ]);

        // Benchmark prompts
        let benchmark_throughput = TestPrompt::new(
            "benchmark-throughput",
            "Generate a list of 10 random words.",
            PromptCategory::Benchmark,
        )
        .with_expected(ExpectedResponse {
            min_length: Some(20),
            max_length: Some(100),
            max_response_time_ms: Some(2000),
            required_keywords: vec![],
            forbidden_content: vec![],
            quality_indicators: vec![QualityIndicator::Complete, QualityIndicator::WellFormed],
        })
        .with_metadata(PromptMetadata {
            source: Some("builtin".to_string()),
            language: Some("en".to_string()),
            difficulty: Some(2),
            domain: Some("benchmark".to_string()),
            custom: HashMap::new(),
        })
        .with_tags(vec![
            "benchmark".to_string(),
            "throughput".to_string(),
            "list".to_string(),
        ]);

        // Register all prompts
        self.register(basic_hello);
        self.register(basic_completion);
        self.register(qa_simple);
        self.register(code_hello);
        self.register(math_simple);
        self.register(creative_story);
        self.register(edge_empty);
        self.register(edge_long);
        self.register(benchmark_throughput);

        tracing::debug!("Loaded {} built-in test prompts", self.prompts.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_creation() {
        let prompt = TestPrompt::new(
            "test-prompt",
            "This is a test prompt.",
            PromptCategory::Basic,
        )
        .with_tags(vec!["test".to_string()]);

        assert_eq!(prompt.id, "test-prompt");
        assert_eq!(prompt.text, "This is a test prompt.");
        assert_eq!(prompt.category, PromptCategory::Basic);
        assert_eq!(prompt.tags, vec!["test".to_string()]);

        assert_eq!(prompt.char_length(), 22);
        assert_eq!(prompt.estimated_token_count(), 6); // (22 + 3) / 4
        assert!(!prompt.is_long());
        assert!(!prompt.has_special_formatting());
    }

    #[test]
    fn test_prompt_characteristics() {
        let long_prompt = TestPrompt::new(
            "long",
            &"word ".repeat(300), // 1500 characters, ~375 tokens
            PromptCategory::Basic,
        );

        assert!(long_prompt.is_long());

        let formatted_prompt = TestPrompt::new(
            "formatted",
            "Here's some code:\n```rust\nfn main() {}\n```",
            PromptCategory::CodeGeneration,
        );

        assert!(formatted_prompt.has_special_formatting());
    }

    #[test]
    fn test_prompt_category_descriptions() {
        assert_eq!(
            PromptCategory::Basic.description(),
            "Basic text generation and completion"
        );
        assert_eq!(
            PromptCategory::QuestionAnswering.description(),
            "Question answering and comprehension"
        );

        let (min, max) = PromptCategory::Creative.typical_response_length();
        assert_eq!((min, max), (200, 1000));
    }

    #[tokio::test]
    async fn test_prompt_registry() {
        let registry = TestPromptRegistry::new().await.unwrap();

        // Should have built-in prompts
        assert!(!registry.all().is_empty());

        // Test getting prompts by category
        let basic_prompts = registry.by_category(PromptCategory::Basic);
        assert!(!basic_prompts.is_empty());

        let qa_prompts = registry.by_category(PromptCategory::QuestionAnswering);
        assert!(!qa_prompts.is_empty());

        // Test getting specific prompt
        let hello_prompt = registry.get("basic-hello");
        assert!(hello_prompt.is_some());
        assert_eq!(hello_prompt.unwrap().text, "Hello, how are you today?");

        // Test getting prompts by tags
        let basic_tagged = registry.by_tags(&vec!["basic".to_string()]);
        assert!(!basic_tagged.is_empty());

        // Test getting short and long prompts
        let short_prompts = registry.short_prompts();
        let long_prompts = registry.long_prompts();

        assert!(!short_prompts.is_empty());
        assert!(!long_prompts.is_empty());
    }

    #[tokio::test]
    async fn test_random_sample() {
        let registry = TestPromptRegistry::new().await.unwrap();

        let sample1 = registry.random_sample(3);
        let sample2 = registry.random_sample(3);

        assert_eq!(sample1.len(), 3);
        assert_eq!(sample2.len(), 3);

        // Samples might be different (though not guaranteed due to randomness)
        // Just check that we get valid prompts
        for prompt in sample1 {
            assert!(!prompt.id.is_empty());
            assert!(!prompt.text.is_empty());
        }
    }
}
