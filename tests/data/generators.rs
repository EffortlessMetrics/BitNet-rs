use async_trait::async_trait;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ModelParameters, ModelSize, PromptCategory, TestModel, TestPrompt};
use crate::common::TestResult;

/// Trait for generating test data
#[async_trait]
pub trait DataGenerator: Send + Sync {
    /// Generate data based on the given configuration
    async fn generate(&self, config: &GenerationConfig) -> TestResult<Vec<u8>>;

    /// Validate that the generator can produce the requested data
    fn can_generate(&self, config: &GenerationConfig) -> bool;

    /// Get the name of this generator
    fn name(&self) -> &str;
}

/// Configuration for data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Type of data to generate
    pub data_type: DataType,
    /// Size constraints
    pub size_constraints: SizeConstraints,
    /// Quality requirements
    pub quality: QualityLevel,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            data_type: DataType::Text,
            size_constraints: SizeConstraints::default(),
            quality: QualityLevel::Standard,
            seed: None,
            parameters: HashMap::new(),
        }
    }
}

/// Types of data that can be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    /// Text data (prompts, responses)
    Text,
    /// Model data (synthetic models)
    Model,
    /// Binary data (for testing edge cases)
    Binary,
    /// JSON data (structured test data)
    Json,
    /// Configuration data
    Config,
}

/// Size constraints for generated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeConstraints {
    /// Minimum size in bytes
    pub min_size: usize,
    /// Maximum size in bytes
    pub max_size: usize,
    /// Target size in bytes (if specified)
    pub target_size: Option<usize>,
}

impl Default for SizeConstraints {
    fn default() -> Self {
        Self {
            min_size: 0,
            max_size: 1024 * 1024, // 1MB
            target_size: None,
        }
    }
}

/// Quality level for generated data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Minimal quality (for stress testing)
    Minimal,
    /// Standard quality (for normal testing)
    Standard,
    /// High quality (for validation testing)
    High,
}

/// Generator for model data
pub struct ModelDataGenerator {
    rng: StdRng,
}

impl ModelDataGenerator {
    /// Create a new model data generator
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self { rng }
    }

    /// Generate a synthetic model definition
    pub fn generate_model(&mut self, size: ModelSize) -> TestModel {
        let id = format!("synthetic-{}", self.rng.gen::<u32>());
        let name = format!("Synthetic Model {}", self.rng.gen::<u32>());

        let (min_size, max_size) = size.size_range();
        let file_size = if max_size == u64::MAX {
            min_size + self.rng.gen::<u32>() as u64 * 1024 * 1024 // Add up to 4GB
        } else {
            self.rng.gen_range(min_size..=max_size)
        };

        let parameters = self.generate_model_parameters(file_size);

        TestModel::new(
            id,
            name,
            size,
            crate::common::ModelFormat::Gguf,
            crate::common::ModelType::BitNet,
        )
        .with_file_size(file_size)
        .with_checksum(format!("{:x}", self.rng.gen::<u64>()))
        .with_parameters(parameters)
        .with_description("Synthetically generated test model")
        .with_tags(vec!["synthetic".to_string(), "generated".to_string()])
    }

    /// Generate realistic model parameters based on file size
    fn generate_model_parameters(&mut self, file_size: u64) -> ModelParameters {
        // Rough estimates based on typical model scaling
        let parameter_count = match file_size {
            0..=100_000_000 => self.rng.gen_range(1_000_000..=100_000_000), // 1M-100M
            100_000_001..=1_000_000_000 => self.rng.gen_range(100_000_000..=1_000_000_000), // 100M-1B
            _ => self.rng.gen_range(1_000_000_000..=70_000_000_000), // 1B-70B
        };

        let context_length = *[512, 1024, 2048, 4096, 8192, 16384]
            .choose(&mut self.rng)
            .unwrap();

        let vocab_size = self.rng.gen_range(10000..=100000);

        // Estimate layers and hidden size based on parameter count
        let hidden_size = match parameter_count {
            0..=100_000_000 => *[256, 512, 768].choose(&mut self.rng).unwrap(),
            100_000_001..=1_000_000_000 => *[768, 1024, 1536, 2048].choose(&mut self.rng).unwrap(),
            _ => *[2048, 4096, 8192].choose(&mut self.rng).unwrap(),
        };

        let num_layers = match parameter_count {
            0..=100_000_000 => self.rng.gen_range(6..=24),
            100_000_001..=1_000_000_000 => self.rng.gen_range(12..=48),
            _ => self.rng.gen_range(24..=96),
        };

        ModelParameters::bitnet(
            parameter_count,
            context_length,
            vocab_size,
            num_layers,
            hidden_size,
        )
    }
}

#[async_trait]
impl DataGenerator for ModelDataGenerator {
    async fn generate(&self, config: &GenerationConfig) -> TestResult<Vec<u8>> {
        if !matches!(config.data_type, DataType::Model) {
            return Err(crate::common::TestError::execution(
                "ModelDataGenerator can only generate model data",
            ));
        }

        let mut generator = Self::new(config.seed);

        // Determine model size based on size constraints
        let size = if config.size_constraints.max_size <= 100 * 1024 * 1024 {
            ModelSize::Tiny
        } else if config.size_constraints.max_size <= 1024 * 1024 * 1024 {
            ModelSize::Small
        } else if config.size_constraints.max_size <= 10 * 1024 * 1024 * 1024 {
            ModelSize::Medium
        } else {
            ModelSize::Large
        };

        let model = generator.generate_model(size);
        let json_data = serde_json::to_string_pretty(&model)?;

        Ok(json_data.into_bytes())
    }

    fn can_generate(&self, config: &GenerationConfig) -> bool {
        matches!(config.data_type, DataType::Model)
    }

    fn name(&self) -> &str {
        "ModelDataGenerator"
    }
}

/// Generator for prompt data
pub struct PromptDataGenerator {
    rng: StdRng,
}

impl PromptDataGenerator {
    /// Create a new prompt data generator
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self { rng }
    }

    /// Generate a synthetic prompt
    pub fn generate_prompt(&mut self, category: PromptCategory) -> TestPrompt {
        let id = format!("synthetic-prompt-{}", self.rng.gen::<u32>());
        let text = self.generate_prompt_text(category);

        TestPrompt::new(id, text, category)
            .with_tags(vec!["synthetic".to_string(), "generated".to_string()])
    }

    /// Generate prompt text based on category
    fn generate_prompt_text(&mut self, category: PromptCategory) -> String {
        match category {
            PromptCategory::Basic => {
                let templates = [
                    "Tell me about {}.",
                    "What do you think about {}?",
                    "Explain {} in simple terms.",
                    "Describe {} for me.",
                ];
                let topics = ["technology", "nature", "science", "art", "music", "history"];

                let template = templates.choose(&mut self.rng).unwrap();
                let topic = topics.choose(&mut self.rng).unwrap();

                template.replace("{}", topic)
            }

            PromptCategory::QuestionAnswering => {
                let questions = [
                    "What is the capital of {}?",
                    "Who invented the {}?",
                    "When was {} discovered?",
                    "How does {} work?",
                    "Why is {} important?",
                ];
                let subjects = [
                    "telephone",
                    "computer",
                    "internet",
                    "electricity",
                    "gravity",
                ];

                let question = questions.choose(&mut self.rng).unwrap();
                let subject = subjects.choose(&mut self.rng).unwrap();

                question.replace("{}", subject)
            }

            PromptCategory::CodeGeneration => {
                let tasks = [
                    "Write a {} function that calculates the factorial of a number.",
                    "Create a {} program that sorts an array of integers.",
                    "Implement a {} class for a simple calculator.",
                    "Write {} code to read a file and count the words.",
                ];
                let languages = ["Python", "JavaScript", "Java", "C++", "Rust"];

                let task = tasks.choose(&mut self.rng).unwrap();
                let language = languages.choose(&mut self.rng).unwrap();

                task.replace("{}", language)
            }

            PromptCategory::Mathematics => {
                let problems = [
                    format!(
                        "What is {} + {}?",
                        self.rng.gen_range(1..=100),
                        self.rng.gen_range(1..=100)
                    ),
                    format!(
                        "Calculate {} * {}.",
                        self.rng.gen_range(1..=20),
                        self.rng.gen_range(1..=20)
                    ),
                    format!("Find the square root of {}.", self.rng.gen_range(1..=100)),
                    "Solve for x: 2x + 5 = 15".to_string(),
                ];

                problems.choose(&mut self.rng).unwrap().clone()
            }

            PromptCategory::Creative => {
                let prompts = [
                    "Write a short story about a {} who discovers a magical {}.",
                    "Create a poem about {} in the style of {}.",
                    "Describe a world where {} is the most important thing.",
                    "Tell me about a character who can {}.",
                ];
                let characters = ["wizard", "robot", "detective", "artist", "scientist"];
                let objects = ["book", "key", "mirror", "stone", "flower"];
                let styles = [
                    "Shakespeare",
                    "Dr. Seuss",
                    "Edgar Allan Poe",
                    "Maya Angelou",
                ];
                let abilities = [
                    "read minds",
                    "time travel",
                    "become invisible",
                    "talk to animals",
                ];

                let prompt = prompts.choose(&mut self.rng).unwrap();
                match prompt {
                    p if p.contains("magical") => {
                        let character = characters.choose(&mut self.rng).unwrap();
                        let object = objects.choose(&mut self.rng).unwrap();
                        p.replace("{}", character).replacen("{}", object, 1)
                    }
                    p if p.contains("style") => {
                        let topic = ["love", "nature", "time", "dreams"]
                            .choose(&mut self.rng)
                            .unwrap();
                        let style = styles.choose(&mut self.rng).unwrap();
                        p.replace("{}", topic).replacen("{}", style, 1)
                    }
                    p if p.contains("world") => {
                        let concept = ["kindness", "music", "color", "memory"]
                            .choose(&mut self.rng)
                            .unwrap();
                        p.replace("{}", concept)
                    }
                    p => {
                        let ability = abilities.choose(&mut self.rng).unwrap();
                        p.replace("{}", ability)
                    }
                }
            }

            PromptCategory::Factual => {
                let queries = [
                    "What are the main causes of {}?",
                    "List the benefits of {}.",
                    "Explain the history of {}.",
                    "What are the different types of {}?",
                ];
                let topics = [
                    "climate change",
                    "renewable energy",
                    "artificial intelligence",
                    "space exploration",
                ];

                let query = queries.choose(&mut self.rng).unwrap();
                let topic = topics.choose(&mut self.rng).unwrap();

                query.replace("{}", topic)
            }

            PromptCategory::Conversational => {
                let conversations = [
                    "Hi there! How has your day been?",
                    "I'm feeling a bit stressed today. Any advice?",
                    "What's your favorite way to relax?",
                    "Can you recommend a good book to read?",
                    "What do you think about the weather lately?",
                ];

                conversations.choose(&mut self.rng).unwrap().to_string()
            }

            PromptCategory::EdgeCase => {
                let edge_cases = [
                    "".to_string(),                              // Empty prompt
                    "a".repeat(self.rng.gen_range(1000..=5000)), // Very long prompt
                    "🚀🌟💫🎉🔥".to_string(),                    // Only emojis
                    "1234567890".repeat(100),                    // Only numbers
                    "!@#$%^&*()".repeat(50),                     // Only special characters
                ];

                edge_cases.choose(&mut self.rng).unwrap().clone()
            }

            PromptCategory::Benchmark => {
                let benchmarks = [
                    format!("Generate {} random words.", self.rng.gen_range(5..=50)),
                    format!("Count from 1 to {}.", self.rng.gen_range(10..=100)),
                    "Repeat the word 'test' 20 times.".to_string(),
                    "List the alphabet backwards.".to_string(),
                ];

                benchmarks.choose(&mut self.rng).unwrap().clone()
            }

            PromptCategory::Multilingual => {
                let multilingual = [
                    "Translate 'Hello, how are you?' to Spanish.",
                    "What does 'Bonjour' mean in English?",
                    "Count from 1 to 5 in German.",
                    "Say 'Thank you' in Japanese.",
                ];

                multilingual.choose(&mut self.rng).unwrap().to_string()
            }
        }
    }
}

#[async_trait]
impl DataGenerator for PromptDataGenerator {
    async fn generate(&self, config: &GenerationConfig) -> TestResult<Vec<u8>> {
        if !matches!(config.data_type, DataType::Text) {
            return Err(crate::common::TestError::execution(
                "PromptDataGenerator can only generate text data",
            ));
        }

        let mut generator = Self::new(config.seed);

        // Generate multiple prompts based on size constraints
        let mut prompts = Vec::new();
        let mut total_size = 0;

        while total_size < config.size_constraints.min_size {
            let category = *[
                PromptCategory::Basic,
                PromptCategory::QuestionAnswering,
                PromptCategory::CodeGeneration,
                PromptCategory::Mathematics,
                PromptCategory::Creative,
                PromptCategory::Factual,
                PromptCategory::Conversational,
            ]
            .choose(&mut generator.rng)
            .unwrap();

            let prompt = generator.generate_prompt(category);
            total_size += prompt.text.len();
            prompts.push(prompt);

            if total_size >= config.size_constraints.max_size {
                break;
            }
        }

        let json_data = serde_json::to_string_pretty(&prompts)?;
        Ok(json_data.into_bytes())
    }

    fn can_generate(&self, config: &GenerationConfig) -> bool {
        matches!(config.data_type, DataType::Text)
    }

    fn name(&self) -> &str {
        "PromptDataGenerator"
    }
}

/// Generator for binary test data
pub struct BinaryDataGenerator {
    rng: StdRng,
}

impl BinaryDataGenerator {
    /// Create a new binary data generator
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self { rng }
    }
}

#[async_trait]
impl DataGenerator for BinaryDataGenerator {
    async fn generate(&self, config: &GenerationConfig) -> TestResult<Vec<u8>> {
        if !matches!(config.data_type, DataType::Binary) {
            return Err(crate::common::TestError::execution(
                "BinaryDataGenerator can only generate binary data",
            ));
        }

        let mut generator = Self::new(config.seed);

        let size = config.size_constraints.target_size.unwrap_or(
            generator
                .rng
                .gen_range(config.size_constraints.min_size..=config.size_constraints.max_size),
        );

        let mut data = vec![0u8; size];

        match config.quality {
            QualityLevel::Minimal => {
                // Fill with zeros (minimal quality)
            }
            QualityLevel::Standard => {
                // Fill with random data
                for byte in &mut data {
                    *byte = generator.rng.gen();
                }
            }
            QualityLevel::High => {
                // Fill with structured random data (patterns)
                for (i, byte) in data.iter_mut().enumerate() {
                    *byte = ((i % 256) ^ generator.rng.gen::<u8>()) as u8;
                }
            }
        }

        Ok(data)
    }

    fn can_generate(&self, config: &GenerationConfig) -> bool {
        matches!(config.data_type, DataType::Binary)
    }

    fn name(&self) -> &str {
        "BinaryDataGenerator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_data_generator() {
        let mut generator = ModelDataGenerator::new(Some(42));

        let tiny_model = generator.generate_model(ModelSize::Tiny);
        assert_eq!(tiny_model.size, ModelSize::Tiny);
        assert!(tiny_model.file_size <= 100 * 1024 * 1024);
        assert!(!tiny_model.id.is_empty());
        assert!(!tiny_model.name.is_empty());
        assert!(tiny_model.parameters.parameter_count.is_some());

        let small_model = generator.generate_model(ModelSize::Small);
        assert_eq!(small_model.size, ModelSize::Small);
        assert!(small_model.file_size > 100 * 1024 * 1024);
        assert!(small_model.file_size <= 1024 * 1024 * 1024);
    }

    #[test]
    fn test_prompt_data_generator() {
        let mut generator = PromptDataGenerator::new(Some(42));

        let basic_prompt = generator.generate_prompt(PromptCategory::Basic);
        assert_eq!(basic_prompt.category, PromptCategory::Basic);
        assert!(!basic_prompt.text.is_empty());
        assert!(basic_prompt.tags.contains(&"synthetic".to_string()));

        let qa_prompt = generator.generate_prompt(PromptCategory::QuestionAnswering);
        assert_eq!(qa_prompt.category, PromptCategory::QuestionAnswering);
        assert!(!qa_prompt.text.is_empty());

        let code_prompt = generator.generate_prompt(PromptCategory::CodeGeneration);
        assert_eq!(code_prompt.category, PromptCategory::CodeGeneration);
        assert!(!code_prompt.text.is_empty());
    }

    #[tokio::test]
    async fn test_data_generator_trait() {
        let model_gen = ModelDataGenerator::new(Some(42));
        let prompt_gen = PromptDataGenerator::new(Some(42));
        let binary_gen = BinaryDataGenerator::new(Some(42));

        // Test model generator
        let model_config = GenerationConfig {
            data_type: DataType::Model,
            size_constraints: SizeConstraints {
                min_size: 1000,
                max_size: 10000,
                target_size: None,
            },
            ..Default::default()
        };

        assert!(model_gen.can_generate(&model_config));
        let model_data = model_gen.generate(&model_config).await.unwrap();
        assert!(!model_data.is_empty());

        // Test prompt generator
        let text_config = GenerationConfig {
            data_type: DataType::Text,
            size_constraints: SizeConstraints {
                min_size: 100,
                max_size: 1000,
                target_size: None,
            },
            ..Default::default()
        };

        assert!(prompt_gen.can_generate(&text_config));
        let text_data = prompt_gen.generate(&text_config).await.unwrap();
        assert!(!text_data.is_empty());

        // Test binary generator
        let binary_config = GenerationConfig {
            data_type: DataType::Binary,
            size_constraints: SizeConstraints {
                min_size: 100,
                max_size: 200,
                target_size: Some(150),
            },
            quality: QualityLevel::Standard,
            ..Default::default()
        };

        assert!(binary_gen.can_generate(&binary_config));
        let binary_data = binary_gen.generate(&binary_config).await.unwrap();
        assert_eq!(binary_data.len(), 150);
    }

    #[tokio::test]
    async fn test_generation_config() {
        let config = GenerationConfig::default();
        assert!(matches!(config.data_type, DataType::Text));
        assert_eq!(config.size_constraints.min_size, 0);
        assert_eq!(config.size_constraints.max_size, 1024 * 1024);
        assert!(matches!(config.quality, QualityLevel::Standard));
        assert!(config.seed.is_none());
    }
}
