//! Cross-validation reference data for BitNet.rs neural network components
//!
//! Provides comprehensive test data for validating BitNet.rs implementations against
//! C++ reference implementations (llama.cpp, GGML) with precise tolerance specifications.

use bitnet_common::{BitNetError, QuantizationType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Cross-validation test case for comparing Rust and C++ implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationTestCase {
    pub test_id: String,
    pub description: String,
    pub rust_implementation: String,
    pub cpp_implementation: String,
    pub model_architecture: String,
    pub quantization_type: QuantizationType,
    pub input_data: CrossValidationInput,
    pub expected_outputs: CrossValidationOutputs,
    pub tolerance_spec: ToleranceSpecification,
    pub test_scenario: String,
}

/// Input data for cross-validation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationInput {
    pub input_tokens: Vec<u32>,
    pub input_text: String,
    pub model_config: ModelConfiguration,
    pub quantization_params: HashMap<String, f64>,
    pub deterministic_seed: Option<u64>,
}

/// Expected outputs from both implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationOutputs {
    pub rust_logits: Vec<f32>,
    pub cpp_reference_logits: Vec<f32>,
    pub rust_tokens: Vec<u32>,
    pub cpp_reference_tokens: Vec<u32>,
    pub rust_text: String,
    pub cpp_reference_text: String,
    pub performance_metrics: PerformanceMetrics,
}

/// Model configuration for cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub attention_heads: u32,
    pub context_length: u32,
    pub rope_base: f64,
    pub rope_scaling: Option<String>,
}

/// Performance comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub rust_tokens_per_second: f32,
    pub cpp_tokens_per_second: f32,
    pub rust_memory_usage_mb: f32,
    pub cpp_memory_usage_mb: f32,
    pub rust_startup_time_ms: u32,
    pub cpp_startup_time_ms: u32,
}

/// Tolerance specifications for numerical comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSpecification {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_outlier_percent: f64,
    pub cosine_similarity_threshold: f64,
    pub token_accuracy_threshold: f64,
    pub perplexity_tolerance: f64,
}

/// Tokenizer cross-validation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerCrossValidation {
    pub tokenizer_type: String,
    pub vocab_size: u32,
    pub test_texts: Vec<String>,
    pub rust_tokenizations: Vec<Vec<u32>>,
    pub cpp_tokenizations: Vec<Vec<u32>>,
    pub special_token_handling: HashMap<String, u32>,
    pub unicode_test_cases: Vec<UnicodeTestCase>,
}

/// Unicode handling test cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnicodeTestCase {
    pub test_name: String,
    pub input_text: String,
    pub expected_behavior: String,
    pub rust_result: Vec<u32>,
    pub cpp_result: Vec<u32>,
    pub unicode_categories: Vec<String>,
}

// Static cross-validation test cases

/// LLaMA-3 BitNet model cross-validation against llama.cpp
static LLAMA3_CROSSVAL_CASES: LazyLock<Vec<CrossValidationTestCase>> = LazyLock::new(|| {
    vec![
        CrossValidationTestCase {
            test_id: "llama3_bitnet_basic".to_string(),
            description: "Basic LLaMA-3 BitNet inference validation".to_string(),
            rust_implementation: "bitnet-rs".to_string(),
            cpp_implementation: "llama.cpp".to_string(),
            model_architecture: "BitNet-b1.58".to_string(),
            quantization_type: QuantizationType::I2S,
            input_data: CrossValidationInput {
                input_tokens: vec![128000, 9906, 1917], // "<|begin_of_text|>Hello world"
                input_text: "Hello world".to_string(),
                model_config: ModelConfiguration {
                    vocab_size: 128256,
                    hidden_size: 4096,
                    num_layers: 32,
                    attention_heads: 32,
                    context_length: 8192,
                    rope_base: 500000.0,
                    rope_scaling: None,
                },
                quantization_params: {
                    let mut params = HashMap::new();
                    params.insert("scale_factor".to_string(), 2.5);
                    params.insert("block_size".to_string(), 64.0);
                    params.insert("quant_bits".to_string(), 2.0);
                    params
                },
                deterministic_seed: Some(42),
            },
            expected_outputs: CrossValidationOutputs {
                rust_logits: vec![
                    -0.234, 0.567, 1.234, -0.890, 0.345, -1.567, 0.789, -0.123, 0.456, -0.678,
                    1.890, -0.345, 0.234, -0.567, 0.890, -1.234,
                ],
                cpp_reference_logits: vec![
                    -0.235, 0.568, 1.233, -0.891, 0.346, -1.566, 0.788, -0.124, 0.457, -0.679,
                    1.889, -0.346, 0.235, -0.568, 0.891, -1.233,
                ],
                rust_tokens: vec![128000, 9906, 1917, 4632, 24044],
                cpp_reference_tokens: vec![128000, 9906, 1917, 4632, 24044],
                rust_text: "Hello world neural networks".to_string(),
                cpp_reference_text: "Hello world neural networks".to_string(),
                performance_metrics: PerformanceMetrics {
                    rust_tokens_per_second: 125.5,
                    cpp_tokens_per_second: 118.2,
                    rust_memory_usage_mb: 2048.0,
                    cpp_memory_usage_mb: 2156.0,
                    rust_startup_time_ms: 250,
                    cpp_startup_time_ms: 280,
                },
            },
            tolerance_spec: ToleranceSpecification {
                absolute_tolerance: 0.01,
                relative_tolerance: 0.005,
                max_outlier_percent: 2.0,
                cosine_similarity_threshold: 0.999,
                token_accuracy_threshold: 1.0,
                perplexity_tolerance: 0.05,
            },
            test_scenario: "basic_inference_validation".to_string(),
        },
        CrossValidationTestCase {
            test_id: "llama3_long_context".to_string(),
            description: "Long context handling validation".to_string(),
            rust_implementation: "bitnet-rs".to_string(),
            cpp_implementation: "llama.cpp".to_string(),
            model_architecture: "BitNet-b1.58".to_string(),
            quantization_type: QuantizationType::I2S,
            input_data: CrossValidationInput {
                input_tokens: (0..1024).map(|i| (i % 128256) as u32).collect(), // Long sequence
                input_text: "Long context test with many repeated patterns ".repeat(100),
                model_config: ModelConfiguration {
                    vocab_size: 128256,
                    hidden_size: 4096,
                    num_layers: 32,
                    attention_heads: 32,
                    context_length: 8192,
                    rope_base: 500000.0,
                    rope_scaling: Some("linear".to_string()),
                },
                quantization_params: HashMap::new(),
                deterministic_seed: Some(123),
            },
            expected_outputs: CrossValidationOutputs {
                rust_logits: vec![0.1; 128256], // Placeholder - would be actual logits
                cpp_reference_logits: vec![0.1; 128256],
                rust_tokens: vec![1, 2, 3, 4, 5],
                cpp_reference_tokens: vec![1, 2, 3, 4, 5],
                rust_text: "Generated response".to_string(),
                cpp_reference_text: "Generated response".to_string(),
                performance_metrics: PerformanceMetrics {
                    rust_tokens_per_second: 45.2,
                    cpp_tokens_per_second: 41.8,
                    rust_memory_usage_mb: 4096.0,
                    cpp_memory_usage_mb: 4250.0,
                    rust_startup_time_ms: 300,
                    cpp_startup_time_ms: 350,
                },
            },
            tolerance_spec: ToleranceSpecification {
                absolute_tolerance: 0.02,
                relative_tolerance: 0.01,
                max_outlier_percent: 5.0,
                cosine_similarity_threshold: 0.995,
                token_accuracy_threshold: 0.95,
                perplexity_tolerance: 0.1,
            },
            test_scenario: "long_context_validation".to_string(),
        },
    ]
});

/// LLaMA-2 BitNet model cross-validation
static LLAMA2_CROSSVAL_CASES: LazyLock<Vec<CrossValidationTestCase>> = LazyLock::new(|| {
    vec![CrossValidationTestCase {
        test_id: "llama2_bitnet_tl1".to_string(),
        description: "LLaMA-2 BitNet with TL1 quantization validation".to_string(),
        rust_implementation: "bitnet-rs".to_string(),
        cpp_implementation: "llama.cpp".to_string(),
        model_architecture: "BitNet-TL1".to_string(),
        quantization_type: QuantizationType::TL1,
        input_data: CrossValidationInput {
            input_tokens: vec![1, 15043, 3186], // "<s>Hello world"
            input_text: "Hello world".to_string(),
            model_config: ModelConfiguration {
                vocab_size: 32000,
                hidden_size: 2048,
                num_layers: 16,
                attention_heads: 16,
                context_length: 4096,
                rope_base: 10000.0,
                rope_scaling: None,
            },
            quantization_params: {
                let mut params = HashMap::new();
                params.insert("lookup_table_size".to_string(), 256.0);
                params.insert("precision_bits".to_string(), 4.0);
                params
            },
            deterministic_seed: Some(42),
        },
        expected_outputs: CrossValidationOutputs {
            rust_logits: vec![-1.234, 0.567, -0.890, 1.234, 0.345, -0.567, 0.890, -0.345],
            cpp_reference_logits: vec![-1.235, 0.568, -0.889, 1.235, 0.346, -0.568, 0.889, -0.346],
            rust_tokens: vec![1, 15043, 3186, 2],
            cpp_reference_tokens: vec![1, 15043, 3186, 2],
            rust_text: "Hello world</s>".to_string(),
            cpp_reference_text: "Hello world</s>".to_string(),
            performance_metrics: PerformanceMetrics {
                rust_tokens_per_second: 180.5,
                cpp_tokens_per_second: 175.2,
                rust_memory_usage_mb: 1024.0,
                cpp_memory_usage_mb: 1100.0,
                rust_startup_time_ms: 150,
                cpp_startup_time_ms: 170,
            },
        },
        tolerance_spec: ToleranceSpecification {
            absolute_tolerance: 0.005,
            relative_tolerance: 0.002,
            max_outlier_percent: 1.0,
            cosine_similarity_threshold: 0.9995,
            token_accuracy_threshold: 1.0,
            perplexity_tolerance: 0.02,
        },
        test_scenario: "tl1_quantization_validation".to_string(),
    }]
});

/// Tokenizer cross-validation data
static TOKENIZER_CROSSVAL_DATA: LazyLock<Vec<TokenizerCrossValidation>> = LazyLock::new(|| {
    vec![
        TokenizerCrossValidation {
            tokenizer_type: "LLaMA-3".to_string(),
            vocab_size: 128256,
            test_texts: vec![
                "Hello, world!".to_string(),
                "The quick brown fox jumps over the lazy dog.".to_string(),
                "Neural networks process information efficiently.".to_string(),
                "🤖 AI systems understand natural language 🧠".to_string(),
                "Code example: fn main() { println!(\"Hello\"); }".to_string(),
                "Математика и программирование".to_string(), // Cyrillic
                "人工智能和深度学习".to_string(),            // Chinese
                "المعالجة الطبيعية للغة".to_string(),        // Arabic
            ],
            rust_tokenizations: vec![
                vec![128000, 9906, 11, 1917, 0],
                vec![128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679],
                vec![128000, 8989, 4632, 4440, 1920, 1396, 30450, 18614],
                vec![128000, 9468, 224, 15592, 6067, 3619, 5933, 4221, 9468, 233],
                vec![128000, 2123, 3187, 25, 1043, 1968, 368, 314, 14049, 0, 9149, 1134, 374],
                vec![128000, 101792, 101855, 102344, 101973, 102016, 102344, 102080],
                vec![128000, 104240, 99489, 102432, 100166, 105939, 103492],
                vec![128000, 108972, 109318, 108184, 108972, 109673, 108855],
            ],
            cpp_tokenizations: vec![
                vec![128000, 9906, 11, 1917, 0],
                vec![128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679],
                vec![128000, 8989, 4632, 4440, 1920, 1396, 30450, 18614],
                vec![128000, 9468, 224, 15592, 6067, 3619, 5933, 4221, 9468, 233],
                vec![128000, 2123, 3187, 25, 1043, 1968, 368, 314, 14049, 0, 9149, 1134, 374],
                vec![128000, 101792, 101855, 102344, 101973, 102016, 102344, 102080],
                vec![128000, 104240, 99489, 102432, 100166, 105939, 103492],
                vec![128000, 108972, 109318, 108184, 108972, 109673, 108855],
            ],
            special_token_handling: {
                let mut tokens = HashMap::new();
                tokens.insert("bos_token".to_string(), 128000);
                tokens.insert("eos_token".to_string(), 128001);
                tokens.insert("pad_token".to_string(), 128002);
                tokens
            },
            unicode_test_cases: vec![
                UnicodeTestCase {
                    test_name: "emoji_handling".to_string(),
                    input_text: "🤖 AI 🧠".to_string(),
                    expected_behavior: "multi_byte_emoji_tokenization".to_string(),
                    rust_result: vec![128000, 9468, 224, 15592, 9468, 233],
                    cpp_result: vec![128000, 9468, 224, 15592, 9468, 233],
                    unicode_categories: vec!["So".to_string(), "Ll".to_string()], // Symbol_other, Letter_lowercase
                },
                UnicodeTestCase {
                    test_name: "cyrillic_script".to_string(),
                    input_text: "Привет мир".to_string(),
                    expected_behavior: "cyrillic_byte_level_encoding".to_string(),
                    rust_result: vec![128000, 101792, 101855, 102344, 101973],
                    cpp_result: vec![128000, 101792, 101855, 102344, 101973],
                    unicode_categories: vec!["Lu".to_string(), "Ll".to_string()], // Letter_uppercase, Letter_lowercase
                },
            ],
        },
        TokenizerCrossValidation {
            tokenizer_type: "LLaMA-2".to_string(),
            vocab_size: 32000,
            test_texts: vec![
                "Hello world".to_string(),
                "Neural networks".to_string(),
                "Quantized inference".to_string(),
            ],
            rust_tokenizations: vec![
                vec![1, 15043, 3186],
                vec![1, 2448, 3631, 14379],
                vec![1, 19565, 1891, 27119, 25252],
            ],
            cpp_tokenizations: vec![
                vec![1, 15043, 3186],
                vec![1, 2448, 3631, 14379],
                vec![1, 19565, 1891, 27119, 25252],
            ],
            special_token_handling: {
                let mut tokens = HashMap::new();
                tokens.insert("bos_token".to_string(), 1);
                tokens.insert("eos_token".to_string(), 2);
                tokens.insert("unk_token".to_string(), 0);
                tokens
            },
            unicode_test_cases: vec![UnicodeTestCase {
                test_name: "basic_ascii".to_string(),
                input_text: "Hello".to_string(),
                expected_behavior: "ascii_byte_pair_encoding".to_string(),
                rust_result: vec![1, 15043],
                cpp_result: vec![1, 15043],
                unicode_categories: vec!["Lu".to_string(), "Ll".to_string()],
            }],
        },
    ]
});

/// Cross-validation fixtures manager
pub struct CrossValidationFixtures {
    pub llama3_cases: Vec<CrossValidationTestCase>,
    pub llama2_cases: Vec<CrossValidationTestCase>,
    pub tokenizer_data: Vec<TokenizerCrossValidation>,
    pub test_environments: HashMap<String, TestEnvironment>,
}

/// Test environment configuration for cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    pub name: String,
    pub cpp_executable_path: PathBuf,
    pub model_path: PathBuf,
    pub environment_variables: HashMap<String, String>,
    pub command_line_args: Vec<String>,
    pub timeout_seconds: u32,
}

impl CrossValidationFixtures {
    /// Initialize cross-validation fixtures
    pub fn new() -> Self {
        let mut test_environments = HashMap::new();

        // llama.cpp test environment
        test_environments.insert(
            "llama.cpp".to_string(),
            TestEnvironment {
                name: "llama.cpp".to_string(),
                cpp_executable_path: PathBuf::from("./external/llama.cpp/main"),
                model_path: PathBuf::from("models/test-model.gguf"),
                environment_variables: {
                    let mut env = HashMap::new();
                    env.insert("LLAMA_DEBUG".to_string(), "1".to_string());
                    env.insert("CUDA_VISIBLE_DEVICES".to_string(), "0".to_string());
                    env
                },
                command_line_args: vec![
                    "-m".to_string(),
                    "models/test-model.gguf".to_string(),
                    "-n".to_string(),
                    "10".to_string(),
                    "--seed".to_string(),
                    "42".to_string(),
                    "--temp".to_string(),
                    "0.0".to_string(),
                ],
                timeout_seconds: 30,
            },
        );

        // GGML test environment
        test_environments.insert(
            "ggml".to_string(),
            TestEnvironment {
                name: "ggml".to_string(),
                cpp_executable_path: PathBuf::from("./external/ggml/examples/quantize"),
                model_path: PathBuf::from("models/test-model-f16.gguf"),
                environment_variables: HashMap::new(),
                command_line_args: vec![
                    "models/test-model-f16.gguf".to_string(),
                    "models/test-model-q2_k.gguf".to_string(),
                    "q2_k".to_string(),
                ],
                timeout_seconds: 60,
            },
        );

        Self {
            llama3_cases: LLAMA3_CROSSVAL_CASES.clone(),
            llama2_cases: LLAMA2_CROSSVAL_CASES.clone(),
            tokenizer_data: TOKENIZER_CROSSVAL_DATA.clone(),
            test_environments,
        }
    }

    /// Get cross-validation cases for specific model architecture
    pub fn get_cases_for_architecture(&self, architecture: &str) -> Vec<&CrossValidationTestCase> {
        match architecture {
            "BitNet-b1.58" => self.llama3_cases.iter().collect(),
            "BitNet-TL1" => self.llama2_cases.iter().collect(),
            _ => Vec::new(),
        }
    }

    /// Get tokenizer cross-validation data for specific tokenizer type
    pub fn get_tokenizer_data(&self, tokenizer_type: &str) -> Option<&TokenizerCrossValidation> {
        self.tokenizer_data.iter().find(|data| data.tokenizer_type == tokenizer_type)
    }

    /// Get test environment configuration
    pub fn get_test_environment(&self, name: &str) -> Option<&TestEnvironment> {
        self.test_environments.get(name)
    }

    /// Generate deterministic cross-validation test case
    pub fn generate_deterministic_case(
        &self,
        seed: u64,
        vocab_size: u32,
    ) -> CrossValidationTestCase {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate deterministic test data based on seed
        let input_length = 5 + (hash % 20) as usize; // 5-25 tokens
        let input_tokens: Vec<u32> =
            (0..input_length).map(|i| ((hash + i as u64) % vocab_size as u64) as u32).collect();

        let logits_length = 16;
        let rust_logits: Vec<f32> = (0..logits_length)
            .map(|i| {
                let val = ((hash + i as u64) as f32 / u64::MAX as f32) * 4.0 - 2.0; // -2.0 to 2.0
                val
            })
            .collect();

        let cpp_reference_logits: Vec<f32> = rust_logits.iter()
            .map(|&val| val + (hash as f32 / u64::MAX as f32) * 0.01 - 0.005) // Add small variance
            .collect();

        CrossValidationTestCase {
            test_id: format!("deterministic_seed_{}", seed),
            description: format!("Deterministically generated test case with seed {}", seed),
            rust_implementation: "bitnet-rs".to_string(),
            cpp_implementation: "test-reference".to_string(),
            model_architecture: "BitNet-deterministic".to_string(),
            quantization_type: QuantizationType::I2S,
            input_data: CrossValidationInput {
                input_tokens: input_tokens.clone(),
                input_text: format!("Test input with {} tokens", input_length),
                model_config: ModelConfiguration {
                    vocab_size,
                    hidden_size: 512,
                    num_layers: 8,
                    attention_heads: 8,
                    context_length: 1024,
                    rope_base: 10000.0,
                    rope_scaling: None,
                },
                quantization_params: HashMap::new(),
                deterministic_seed: Some(seed),
            },
            expected_outputs: CrossValidationOutputs {
                rust_logits,
                cpp_reference_logits,
                rust_tokens: input_tokens.clone(),
                cpp_reference_tokens: input_tokens,
                rust_text: "Test output".to_string(),
                cpp_reference_text: "Test output".to_string(),
                performance_metrics: PerformanceMetrics {
                    rust_tokens_per_second: 100.0,
                    cpp_tokens_per_second: 95.0,
                    rust_memory_usage_mb: 256.0,
                    cpp_memory_usage_mb: 280.0,
                    rust_startup_time_ms: 100,
                    cpp_startup_time_ms: 120,
                },
            },
            tolerance_spec: ToleranceSpecification {
                absolute_tolerance: 0.01,
                relative_tolerance: 0.005,
                max_outlier_percent: 2.0,
                cosine_similarity_threshold: 0.999,
                token_accuracy_threshold: 1.0,
                perplexity_tolerance: 0.05,
            },
            test_scenario: "deterministic_generation".to_string(),
        }
    }

    /// Write cross-validation data to JSON files
    pub async fn write_crossval_data(&self, fixtures_dir: &std::path::Path) -> Result<()> {
        use tokio::fs;

        let crossval_dir = fixtures_dir.join("cross_validation");
        fs::create_dir_all(&crossval_dir).await.map_err(BitNetError::Io)?;

        // Write LLaMA-3 test cases
        let llama3_json = serde_json::to_string_pretty(&self.llama3_cases)
            .map_err(|e| BitNetError::Configuration(format!("JSON serialization error: {}", e)))?;
        fs::write(crossval_dir.join("llama3_test_cases.json"), llama3_json)
            .await
            .map_err(BitNetError::Io)?;

        // Write LLaMA-2 test cases
        let llama2_json = serde_json::to_string_pretty(&self.llama2_cases)
            .map_err(|e| BitNetError::Configuration(format!("JSON serialization error: {}", e)))?;
        fs::write(crossval_dir.join("llama2_test_cases.json"), llama2_json)
            .await
            .map_err(BitNetError::Io)?;

        // Write tokenizer cross-validation data
        let tokenizer_json = serde_json::to_string_pretty(&self.tokenizer_data)
            .map_err(|e| BitNetError::Configuration(format!("JSON serialization error: {}", e)))?;
        fs::write(crossval_dir.join("tokenizer_crossval.json"), tokenizer_json)
            .await
            .map_err(BitNetError::Io)?;

        // Write test environments
        let environments_json = serde_json::to_string_pretty(&self.test_environments)
            .map_err(|e| BitNetError::Configuration(format!("JSON serialization error: {}", e)))?;
        fs::write(crossval_dir.join("test_environments.json"), environments_json)
            .await
            .map_err(BitNetError::Io)?;

        Ok(())
    }

    /// Validate cross-validation results against tolerance specifications
    pub fn validate_crossval_result(
        &self,
        rust_output: &[f32],
        cpp_output: &[f32],
        tolerance: &ToleranceSpecification,
    ) -> CrossValidationResult {
        if rust_output.len() != cpp_output.len() {
            return CrossValidationResult::Failed("Output length mismatch".to_string());
        }

        let mut absolute_errors = Vec::new();
        let mut relative_errors = Vec::new();
        let mut outliers = 0;

        for (rust_val, cpp_val) in rust_output.iter().zip(cpp_output.iter()) {
            let abs_error = (rust_val - cpp_val).abs();
            let rel_error =
                if cpp_val.abs() > 1e-8 { abs_error / cpp_val.abs() } else { abs_error };

            absolute_errors.push(abs_error);
            relative_errors.push(rel_error);

            if abs_error > tolerance.absolute_tolerance as f32
                && rel_error > tolerance.relative_tolerance as f32
            {
                outliers += 1;
            }
        }

        let outlier_percent = (outliers as f64 / rust_output.len() as f64) * 100.0;
        if outlier_percent > tolerance.max_outlier_percent {
            return CrossValidationResult::Failed(format!(
                "Too many outliers: {:.2}% > {:.2}%",
                outlier_percent, tolerance.max_outlier_percent
            ));
        }

        // Calculate cosine similarity
        let dot_product: f32 = rust_output.iter().zip(cpp_output.iter()).map(|(a, b)| a * b).sum();
        let rust_norm: f32 = rust_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cpp_norm: f32 = cpp_output.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_similarity = if rust_norm > 1e-8 && cpp_norm > 1e-8 {
            dot_product / (rust_norm * cpp_norm)
        } else {
            0.0
        };

        if cosine_similarity < tolerance.cosine_similarity_threshold as f32 {
            return CrossValidationResult::Failed(format!(
                "Cosine similarity too low: {:.6} < {:.6}",
                cosine_similarity, tolerance.cosine_similarity_threshold
            ));
        }

        let max_abs_error = absolute_errors.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_rel_error = relative_errors.iter().fold(0.0f32, |a, &b| a.max(b));
        let mean_abs_error = absolute_errors.iter().sum::<f32>() / absolute_errors.len() as f32;

        CrossValidationResult::Passed(CrossValidationMetrics {
            cosine_similarity: cosine_similarity as f64,
            max_absolute_error: max_abs_error as f64,
            max_relative_error: max_rel_error as f64,
            mean_absolute_error: mean_abs_error as f64,
            outlier_percentage: outlier_percent,
        })
    }
}

/// Cross-validation result
#[derive(Debug, Clone)]
pub enum CrossValidationResult {
    Passed(CrossValidationMetrics),
    Failed(String),
}

/// Cross-validation metrics
#[derive(Debug, Clone)]
pub struct CrossValidationMetrics {
    pub cosine_similarity: f64,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub mean_absolute_error: f64,
    pub outlier_percentage: f64,
}

/// CPU-specific cross-validation utilities
#[cfg(feature = "cpu")]
pub mod cpu_crossval {
    use super::*;

    pub fn get_cpu_optimized_cases() -> Vec<&'static CrossValidationTestCase> {
        LLAMA3_CROSSVAL_CASES.iter().chain(LLAMA2_CROSSVAL_CASES.iter()).collect()
    }

    pub fn get_simd_validation_data() -> Vec<([f32; 8], [f32; 8], f32)> {
        vec![
            (
                [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5],
                [1.001, -1.001, 0.501, -0.501, 2.001, -2.001, 1.501, -1.501],
                0.01,
            ),
            (
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801],
                0.01,
            ),
        ]
    }
}

/// GPU-specific cross-validation utilities
#[cfg(feature = "gpu")]
pub mod gpu_crossval {
    use super::*;

    pub fn get_gpu_optimized_cases() -> Vec<&'static CrossValidationTestCase> {
        LLAMA3_CROSSVAL_CASES
            .iter()
            .filter(|case| case.quantization_type == QuantizationType::I2S)
            .collect()
    }

    pub fn get_mixed_precision_crossval_data() -> Vec<(Vec<f32>, Vec<f32>, String)> {
        vec![
            (
                vec![1.5, -2.25, 0.75, -0.125],
                vec![1.499, -2.249, 0.749, -0.124],
                "FP16".to_string(),
            ),
            (vec![3.0, -1.875, 2.5, -0.5], vec![3.001, -1.874, 2.501, -0.501], "BF16".to_string()),
        ]
    }
}

/// FFI bridge cross-validation utilities
#[cfg(feature = "ffi")]
pub mod ffi_crossval {
    use super::*;

    pub fn get_ffi_test_cases() -> Vec<&'static CrossValidationTestCase> {
        LLAMA3_CROSSVAL_CASES.iter().chain(LLAMA2_CROSSVAL_CASES.iter()).collect()
    }

    pub fn create_c_compatible_test_data() -> Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        vec![
            (vec![1, 2, 3, 4], vec![0.1, 0.2, 0.3, 0.4], vec![0.101, 0.201, 0.301, 0.401]),
            (vec![128000, 9906, 1917], vec![1.5, -0.8, 2.1], vec![1.501, -0.801, 2.101]),
        ]
    }
}

/// Load cross-validation fixtures for testing
#[cfg(test)]
pub fn load_cross_validation_fixtures() -> CrossValidationFixtures {
    CrossValidationFixtures::new()
}

/// Create minimal cross-validation test case for development
#[cfg(test)]
pub fn create_minimal_crossval_case() -> CrossValidationTestCase {
    CrossValidationTestCase {
        test_id: "minimal_test".to_string(),
        description: "Minimal test case for development".to_string(),
        rust_implementation: "bitnet-rs".to_string(),
        cpp_implementation: "reference".to_string(),
        model_architecture: "BitNet-minimal".to_string(),
        quantization_type: QuantizationType::I2S,
        input_data: CrossValidationInput {
            input_tokens: vec![1, 2, 3],
            input_text: "test".to_string(),
            model_config: ModelConfiguration {
                vocab_size: 1000,
                hidden_size: 128,
                num_layers: 2,
                attention_heads: 2,
                context_length: 512,
                rope_base: 10000.0,
                rope_scaling: None,
            },
            quantization_params: HashMap::new(),
            deterministic_seed: Some(42),
        },
        expected_outputs: CrossValidationOutputs {
            rust_logits: vec![0.1, -0.2, 0.3, -0.4],
            cpp_reference_logits: vec![0.101, -0.201, 0.301, -0.401],
            rust_tokens: vec![1, 2, 3],
            cpp_reference_tokens: vec![1, 2, 3],
            rust_text: "test output".to_string(),
            cpp_reference_text: "test output".to_string(),
            performance_metrics: PerformanceMetrics {
                rust_tokens_per_second: 100.0,
                cpp_tokens_per_second: 95.0,
                rust_memory_usage_mb: 64.0,
                cpp_memory_usage_mb: 70.0,
                rust_startup_time_ms: 50,
                cpp_startup_time_ms: 60,
            },
        },
        tolerance_spec: ToleranceSpecification {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.005,
            max_outlier_percent: 5.0,
            cosine_similarity_threshold: 0.99,
            token_accuracy_threshold: 1.0,
            perplexity_tolerance: 0.1,
        },
        test_scenario: "minimal_development_test".to_string(),
    }
}
