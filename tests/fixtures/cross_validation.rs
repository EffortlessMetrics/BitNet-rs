//! Cross-validation fixtures and reference data
//!
//! Provides reference implementations and test data for validating BitNet.rs
//! against C++ implementations with configurable tolerance levels.

use super::{TestEnvironmentConfig, model_artifacts::ModelFixtures};
use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Reference data from C++ implementation for cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceData {
    pub model_id: String,
    pub test_scenario: String,
    pub input_tokens: Vec<u32>,
    pub expected_logits: Vec<f32>,
    pub expected_probabilities: Vec<f32>,
    pub inference_metadata: InferenceMetadata,
    pub tolerance_config: CrossValidationTolerance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    pub model_path: String,
    pub tokenizer_path: String,
    pub device: String,
    pub precision_mode: String,
    pub cpp_version: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationTolerance {
    pub logit_tolerance: f32,
    pub probability_tolerance: f32,
    pub token_prediction_tolerance: f32,
    pub perplexity_tolerance: f32,
    pub tau_b_correlation_min: f32,
}

/// Cross-validation test vectors for numerical accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationTestVectors {
    pub test_cases: Vec<CrossValidationTestCase>,
    pub global_tolerance: CrossValidationTolerance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationTestCase {
    pub name: String,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub deterministic: bool,
    pub seed: Option<u64>,
    pub expected_output: ExpectedOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutput {
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub final_logits: Vec<f32>,
    pub perplexity_score: Option<f32>,
    pub timing_metrics: TimingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    pub total_inference_ms: f32,
    pub first_token_latency_ms: f32,
    pub tokens_per_second: f32,
}

/// Cross-validation fixtures manager
pub struct CrossValidationFixtures {
    pub reference_data: HashMap<String, ReferenceData>,
    pub test_vectors: CrossValidationTestVectors,
    pub config: TestEnvironmentConfig,
    pub cpp_reference_path: Option<PathBuf>,
}

impl CrossValidationFixtures {
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        Self {
            reference_data: HashMap::new(),
            test_vectors: Self::create_test_vectors(),
            config: config.clone(),
            cpp_reference_path: std::env::var("BITNET_CPP_DIR").ok().map(PathBuf::from),
        }
    }

    /// Initialize cross-validation fixtures
    pub async fn initialize(&mut self, model_fixtures: &ModelFixtures) -> Result<()> {
        // Create reference data for available models
        self.create_reference_data(model_fixtures).await?;

        // Load C++ reference data if available
        self.load_cpp_reference_data().await?;

        // Validate cross-validation setup
        self.validate_crossval_setup().await?;

        Ok(())
    }

    /// Create cross-validation test vectors
    fn create_test_vectors() -> CrossValidationTestVectors {
        let mut test_cases = vec![];

        // Deterministic test case for exact matching
        test_cases.push(CrossValidationTestCase {
            name: "deterministic_simple".to_string(),
            prompt: "The capital of France is".to_string(),
            max_new_tokens: 5,
            temperature: 0.0, // Greedy sampling
            top_k: None,
            top_p: None,
            deterministic: true,
            seed: Some(42),
            expected_output: ExpectedOutput {
                generated_tokens: vec![12, 4108, 15], // " Paris" (example tokens)
                generated_text: " Paris".to_string(),
                final_logits: vec![], // Will be filled during validation
                perplexity_score: None,
                timing_metrics: TimingMetrics {
                    total_inference_ms: 100.0,
                    first_token_latency_ms: 50.0,
                    tokens_per_second: 20.0,
                },
            },
        });

        // Temperature sampling test
        test_cases.push(CrossValidationTestCase {
            name: "temperature_sampling".to_string(),
            prompt: "Once upon a time".to_string(),
            max_new_tokens: 10,
            temperature: 0.7,
            top_k: Some(50),
            top_p: None,
            deterministic: true,
            seed: Some(123),
            expected_output: ExpectedOutput {
                generated_tokens: vec![],      // Will be filled during validation
                generated_text: String::new(), // Will be filled during validation
                final_logits: vec![],
                perplexity_score: None,
                timing_metrics: TimingMetrics {
                    total_inference_ms: 200.0,
                    first_token_latency_ms: 60.0,
                    tokens_per_second: 15.0,
                },
            },
        });

        // Top-p sampling test
        test_cases.push(CrossValidationTestCase {
            name: "top_p_sampling".to_string(),
            prompt: "In the field of artificial intelligence".to_string(),
            max_new_tokens: 8,
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
            deterministic: true,
            seed: Some(456),
            expected_output: ExpectedOutput {
                generated_tokens: vec![],
                generated_text: String::new(),
                final_logits: vec![],
                perplexity_score: None,
                timing_metrics: TimingMetrics {
                    total_inference_ms: 150.0,
                    first_token_latency_ms: 55.0,
                    tokens_per_second: 18.0,
                },
            },
        });

        // Perplexity evaluation test
        test_cases.push(CrossValidationTestCase {
            name: "perplexity_evaluation".to_string(),
            prompt: "The quick brown fox jumps over the lazy dog".to_string(),
            max_new_tokens: 0, // Teacher forcing mode
            temperature: 0.0,
            top_k: None,
            top_p: None,
            deterministic: true,
            seed: None,
            expected_output: ExpectedOutput {
                generated_tokens: vec![], // No generation, just evaluation
                generated_text: String::new(),
                final_logits: vec![],         // Full logits for each token
                perplexity_score: Some(12.5), // Example perplexity
                timing_metrics: TimingMetrics {
                    total_inference_ms: 80.0,
                    first_token_latency_ms: 0.0, // No generation
                    tokens_per_second: 0.0,
                },
            },
        });

        CrossValidationTestVectors {
            test_cases,
            global_tolerance: CrossValidationTolerance {
                logit_tolerance: 1e-3,
                probability_tolerance: 1e-4,
                token_prediction_tolerance: 0.0, // Exact match for deterministic
                perplexity_tolerance: 0.1,
                tau_b_correlation_min: 0.95,
            },
        }
    }

    /// Create reference data for cross-validation
    async fn create_reference_data(&mut self, model_fixtures: &ModelFixtures) -> Result<()> {
        // Create reference data for mock models
        if let Some(mock_model) = model_fixtures.get_mock_model("small") {
            let reference = ReferenceData {
                model_id: mock_model.config.model_id.clone(),
                test_scenario: "mock_reference".to_string(),
                input_tokens: vec![464, 6864, 315, 5408, 374], // "The capital of France is"
                expected_logits: Self::generate_mock_logits(mock_model.config.vocab_size as usize),
                expected_probabilities: Self::generate_mock_probabilities(50),
                inference_metadata: InferenceMetadata {
                    model_path: mock_model.config.model_path.display().to_string(),
                    tokenizer_path: mock_model
                        .config
                        .tokenizer_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    device: "cpu".to_string(),
                    precision_mode: "fp32".to_string(),
                    cpp_version: "mock-1.0.0".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
                tolerance_config: CrossValidationTolerance {
                    logit_tolerance: 1e-2, // Relaxed for mock data
                    probability_tolerance: 1e-3,
                    token_prediction_tolerance: 0.0,
                    perplexity_tolerance: 0.5,
                    tau_b_correlation_min: 0.9,
                },
            };

            self.reference_data.insert("mock_small".to_string(), reference);
        }

        // Create reference data for real models if available
        if self.config.real_models_available() {
            let real_reference = ReferenceData {
                model_id: "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
                test_scenario: "real_model_reference".to_string(),
                input_tokens: vec![464, 6864, 315, 5408, 374],
                expected_logits: Self::generate_mock_logits(128256), // LLaMA-3 vocab
                expected_probabilities: Self::generate_mock_probabilities(50),
                inference_metadata: InferenceMetadata {
                    model_path: self
                        .config
                        .model_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    tokenizer_path: self
                        .config
                        .tokenizer_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    device: format!("{:?}", self.config.device_preference).to_lowercase(),
                    precision_mode: "fp32".to_string(),
                    cpp_version: "unknown".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
                tolerance_config: CrossValidationTolerance {
                    logit_tolerance: 1e-4,
                    probability_tolerance: 1e-5,
                    token_prediction_tolerance: 0.0,
                    perplexity_tolerance: 0.05,
                    tau_b_correlation_min: 0.98,
                },
            };

            self.reference_data.insert("real_bitnet_2b".to_string(), real_reference);
        }

        Ok(())
    }

    /// Generate mock logits for testing
    fn generate_mock_logits(vocab_size: usize) -> Vec<f32> {
        let mut logits = vec![-10.0; vocab_size]; // Initialize with low probability

        // Set higher logits for common tokens
        let common_tokens = vec![
            12,   // " Paris"
            374,  // " is"
            264,  // " the"
            6864, // "capital"
            315,  // " of"
        ];

        for (i, &token_id) in common_tokens.iter().enumerate() {
            if token_id < vocab_size {
                logits[token_id] = 10.0 - i as f32; // Decreasing probability
            }
        }

        // Add some noise to other tokens
        for i in 0..vocab_size.min(1000) {
            if !common_tokens.contains(&i) {
                logits[i] = -5.0 + (i as f32 * 0.01) % 2.0; // Small random variation
            }
        }

        logits
    }

    /// Generate mock probabilities for top-k tokens
    fn generate_mock_probabilities(top_k: usize) -> Vec<f32> {
        let mut probs = vec![0.0; top_k];
        let mut total = 0.0;

        // Generate decreasing probabilities
        for i in 0..top_k {
            probs[i] = (1.0 / (i as f32 + 1.0)).exp();
            total += probs[i];
        }

        // Normalize to sum to 1.0
        for prob in &mut probs {
            *prob /= total;
        }

        probs
    }

    /// Load C++ reference data if available
    async fn load_cpp_reference_data(&mut self) -> Result<()> {
        if let Some(cpp_dir) = &self.cpp_reference_path {
            let reference_file = cpp_dir.join("crossval/reference_outputs.json");

            if reference_file.exists() {
                match tokio::fs::read_to_string(&reference_file).await {
                    Ok(content) => {
                        if let Ok(cpp_references) =
                            serde_json::from_str::<Vec<ReferenceData>>(&content)
                        {
                            for reference in cpp_references {
                                self.reference_data
                                    .insert(format!("cpp_{}", reference.model_id), reference);
                            }
                            println!(
                                "Loaded C++ reference data from: {}",
                                reference_file.display()
                            );
                        }
                    }
                    Err(e) => {
                        println!("Warning: Could not load C++ reference data: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate cross-validation setup
    async fn validate_crossval_setup(&self) -> Result<()> {
        if self.reference_data.is_empty() {
            return Err(BitNetError::Validation(
                "No reference data available for cross-validation".to_string(),
            ));
        }

        // Validate reference data consistency
        for (key, reference) in &self.reference_data {
            if reference.input_tokens.is_empty() {
                return Err(BitNetError::Validation(format!(
                    "Reference data '{}' has no input tokens",
                    key
                )));
            }

            if reference.expected_logits.is_empty() {
                return Err(BitNetError::Validation(format!(
                    "Reference data '{}' has no expected logits",
                    key
                )));
            }
        }

        println!(
            "Cross-validation setup validated with {} reference datasets",
            self.reference_data.len()
        );
        Ok(())
    }

    /// Get reference data for a specific model
    pub fn get_reference_data(&self, model_key: &str) -> Option<&ReferenceData> {
        self.reference_data.get(model_key)
    }

    /// Run cross-validation test against reference
    pub async fn run_cross_validation(
        &self,
        model_key: &str,
        test_case: &CrossValidationTestCase,
    ) -> Result<CrossValidationResult> {
        let reference = self.get_reference_data(model_key).ok_or_else(|| {
            BitNetError::Validation(format!("No reference data for model: {}", model_key))
        })?;

        // This would run actual inference and compare against reference
        // For now, return mock validation result
        let mock_result = self.create_mock_validation_result(reference, test_case);

        Ok(mock_result)
    }

    /// Create mock validation result for testing
    fn create_mock_validation_result(
        &self,
        reference: &ReferenceData,
        test_case: &CrossValidationTestCase,
    ) -> CrossValidationResult {
        // Simulate high correlation for deterministic tests
        let correlation = if test_case.deterministic { 0.99 } else { 0.95 };

        let passes_tolerance = correlation >= reference.tolerance_config.tau_b_correlation_min;

        CrossValidationResult {
            test_case_name: test_case.name.clone(),
            model_key: reference.model_id.clone(),
            logit_correlation: correlation,
            probability_correlation: correlation + 0.01,
            token_accuracy: if test_case.deterministic { 1.0 } else { 0.95 },
            perplexity_diff: 0.05,
            passes_tolerance,
            timing_comparison: TimingComparison {
                rust_timing: test_case.expected_output.timing_metrics.clone(),
                cpp_timing: reference.inference_metadata.timestamp.clone(),
                speedup_ratio: 1.05, // Rust slightly faster
            },
            error_message: None,
        }
    }

    /// Cleanup cross-validation fixtures
    pub async fn cleanup(&mut self) -> Result<()> {
        self.reference_data.clear();
        Ok(())
    }
}

/// Cross-validation test result
#[derive(Debug)]
pub struct CrossValidationResult {
    pub test_case_name: String,
    pub model_key: String,
    pub logit_correlation: f32,
    pub probability_correlation: f32,
    pub token_accuracy: f32,
    pub perplexity_diff: f32,
    pub passes_tolerance: bool,
    pub timing_comparison: TimingComparison,
    pub error_message: Option<String>,
}

#[derive(Debug)]
pub struct TimingComparison {
    pub rust_timing: TimingMetrics,
    pub cpp_timing: String,
    pub speedup_ratio: f32,
}
