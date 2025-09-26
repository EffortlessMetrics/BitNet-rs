//! Autoregressive Generation test fixtures for BitNet.rs neural network components
//!
//! Provides comprehensive test data for autoregressive text generation including
//! token sequences, sampling strategies, KV-cache management, and deterministic
//! generation patterns for comprehensive testing.

use super::{TestEnvironmentConfig, quantization::ToleranceConfig};
use bitnet_common::{BitNetError, Device, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Generation configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub use_cache: bool,
}

impl GenerationConfig {
    /// Deterministic generation (temperature=0, no sampling)
    pub fn deterministic() -> Self {
        Self {
            max_length: 100,
            max_new_tokens: 50,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            do_sample: false,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
            use_cache: true,
        }
    }

    /// Sampling generation with nucleus sampling
    pub fn nucleus_sampling() -> Self {
        Self {
            max_length: 150,
            max_new_tokens: 100,
            temperature: 0.8,
            top_k: None,
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            do_sample: true,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
            use_cache: true,
        }
    }

    /// Top-k sampling generation
    pub fn top_k_sampling() -> Self {
        Self {
            max_length: 120,
            max_new_tokens: 80,
            temperature: 1.0,
            top_k: Some(50),
            top_p: None,
            repetition_penalty: 1.2,
            do_sample: true,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
            use_cache: true,
        }
    }
}

/// Generation test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationTestCase {
    pub test_name: String,
    pub description: String,
    pub config: GenerationConfig,
    pub input_data: GenerationInputData,
    pub expected_outputs: GenerationExpectedOutputs,
    pub deterministic_outputs: Option<DeterministicOutputs>,
    pub performance_targets: PerformanceTargets,
    pub device_variants: HashMap<Device, DeviceGenerationData>,
    pub tolerance: ToleranceConfig,
}

/// Input data for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationInputData {
    pub input_text: String,
    pub input_tokens: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub prompt_length: usize,
    pub model_architecture: String,
    pub vocab_size: u32,
    pub context_length: u32,
}

/// Expected generation outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationExpectedOutputs {
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub generation_logits: Vec<Vec<f32>>, // [seq_len, vocab_size]
    pub attention_scores: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, seq_len]
    pub kv_cache_states: Option<KVCacheStates>,
    pub generation_metadata: GenerationMetadata,
}

/// Deterministic outputs (for BITNET_DETERMINISTIC=1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicOutputs {
    pub seed: u64,
    pub expected_tokens: Vec<u32>,
    pub expected_text: String,
    pub expected_token_probabilities: Vec<Vec<f32>>, // [step, vocab_size]
    pub reproducibility_hash: String,                // Hash for reproducibility verification
}

/// Generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub steps_taken: usize,
    pub tokens_generated: usize,
    pub stopped_reason: String, // "max_length", "eos_token", "user_stop"
    pub generation_time_ms: f32,
    pub tokens_per_second: f32,
    pub memory_usage_mb: f32,
}

/// KV-cache states during generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheStates {
    pub cache_key_states: Vec<Vec<Vec<f32>>>, // [layer, seq_pos, hidden_size]
    pub cache_value_states: Vec<Vec<Vec<f32>>>, // [layer, seq_pos, hidden_size]
    pub cache_positions: Vec<usize>,
    pub cache_length: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Performance targets for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub min_tokens_per_second: f32,
    pub max_latency_ms: f32,
    pub max_memory_usage_mb: f32,
    pub min_accuracy_threshold: f32,
    pub kv_cache_efficiency: f32,
}

/// Device-specific generation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceGenerationData {
    pub device: Device,
    pub optimization_strategy: String,
    pub batch_processing: BatchProcessingConfig,
    pub memory_management: MemoryManagementConfig,
    pub performance_metrics: GenerationPerformanceMetrics,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    pub max_batch_size: usize,
    pub dynamic_batching: bool,
    pub padding_strategy: String, // "left", "right", "dynamic"
    pub batch_optimization: bool,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    pub kv_cache_strategy: String, // "static", "dynamic", "sliding_window"
    pub memory_pooling: bool,
    pub gradient_checkpointing: bool,
    pub memory_optimization_level: u8, // 0-3
}

/// Performance metrics for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPerformanceMetrics {
    pub prefill_latency_ms: f32,
    pub decode_latency_ms: f32,
    pub throughput_tokens_per_second: f32,
    pub memory_bandwidth_utilization: f32,
    pub gpu_utilization: f32,
    pub cache_hit_rate: f32,
}

/// Generation fixtures collection
pub struct GenerationFixtures {
    pub test_cases: Vec<GenerationTestCase>,
    pub sampling_test_data: SamplingTestData,
    pub kv_cache_patterns: KVCachePatterns,
    pub config: TestEnvironmentConfig,
}

/// Sampling algorithm test data
#[derive(Debug, Clone)]
pub struct SamplingTestData {
    pub logits_samples: Vec<Vec<f32>>, // Various logits for testing
    pub temperature_tests: Vec<(f32, Vec<u32>)>, // (temperature, expected_tokens)
    pub top_k_tests: Vec<(usize, Vec<u32>)>, // (k, expected_tokens)
    pub top_p_tests: Vec<(f32, Vec<u32>)>, // (p, expected_tokens)
    pub repetition_penalty_tests: Vec<(f32, Vec<u32>, Vec<u32>)>, // (penalty, context, expected)
}

/// KV-cache patterns for testing
#[derive(Debug, Clone)]
pub struct KVCachePatterns {
    pub sequential_generation: KVCachePattern,
    pub batch_generation: KVCachePattern,
    pub long_context_generation: KVCachePattern,
    pub cache_eviction_patterns: Vec<CacheEvictionTest>,
}

/// KV-cache pattern test case
#[derive(Debug, Clone)]
pub struct KVCachePattern {
    pub pattern_name: String,
    pub sequence_length: usize,
    pub generation_steps: usize,
    pub expected_cache_growth: Vec<usize>, // Cache size at each step
    pub expected_cache_usage: Vec<f32>,    // Memory usage at each step
    pub cache_hit_patterns: Vec<bool>,     // Hit/miss at each step
}

/// Cache eviction test
#[derive(Debug, Clone)]
pub struct CacheEvictionTest {
    pub test_name: String,
    pub max_cache_size: usize,
    pub input_sequence: Vec<u32>,
    pub eviction_strategy: String, // "lru", "fifo", "sliding_window"
    pub expected_evictions: Vec<usize>, // Positions that should be evicted
}

/// Static generation test cases
static GENERATION_TEST_CASES: LazyLock<Vec<GenerationTestCase>> = LazyLock::new(|| {
    vec![
        create_deterministic_generation_test(),
        create_nucleus_sampling_test(),
        create_top_k_sampling_test(),
        create_long_context_generation_test(),
        create_batch_generation_test(),
        create_streaming_generation_test(),
    ]
});

impl GenerationFixtures {
    /// Create new generation fixtures
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        Self {
            test_cases: GENERATION_TEST_CASES.clone(),
            sampling_test_data: create_sampling_test_data(),
            kv_cache_patterns: create_kv_cache_patterns(),
            config: config.clone(),
        }
    }

    /// Initialize generation fixtures
    pub async fn initialize(&mut self) -> Result<()> {
        // Generate deterministic outputs for reproducible testing
        self.generate_deterministic_outputs().await?;

        // Create device-specific optimizations
        self.generate_device_variants().await?;

        // Initialize KV-cache test patterns
        self.initialize_kv_cache_patterns().await?;

        // Precompute sampling test cases
        self.precompute_sampling_tests().await?;

        Ok(())
    }

    /// Generate deterministic outputs for all test cases
    async fn generate_deterministic_outputs(&mut self) -> Result<()> {
        let indices: Vec<usize> = self.test_cases
            .iter()
            .enumerate()
            .filter_map(|(i, test_case)| {
                if test_case.deterministic_outputs.is_none() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        for i in indices {
            let test_case = &self.test_cases[i];
            let deterministic = self
                .create_deterministic_output(
                    &test_case.input_data,
                    &test_case.config,
                    42, // Standard seed
                )
                .await?;
            self.test_cases[i].deterministic_outputs = Some(deterministic);
        }

        Ok(())
    }

    /// Create deterministic output for given input
    async fn create_deterministic_output(
        &self,
        input: &GenerationInputData,
        config: &GenerationConfig,
        seed: u64,
    ) -> Result<DeterministicOutputs> {
        // Mock deterministic generation (replace with real implementation)
        let mut rng_state = seed;
        let mut expected_tokens = input.input_tokens.clone();

        // Generate deterministic token sequence
        for _step in 0..config.max_new_tokens.min(20) {
            // Limit for testing
            // Simple LCG for reproducible "random" generation
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let next_token = (rng_state % input.vocab_size as u64) as u32;

            expected_tokens.push(next_token);

            // Stop on EOS token
            if next_token == config.eos_token_id {
                break;
            }
        }

        // Create mock token probabilities
        let expected_token_probabilities: Vec<Vec<f32>> = (0..expected_tokens.len())
            .map(|i| {
                let mut probs = vec![0.001; input.vocab_size as usize]; // Low baseline probability
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let high_prob_token = (rng_state % input.vocab_size as u64) as usize;
                probs[high_prob_token] = 0.8; // High probability for selected token
                probs
            })
            .collect();

        // Create reproducibility hash
        let reproducibility_hash = format!("seed_{}_tokens_{}", seed, expected_tokens.len());

        // Mock generated text (simplified)
        let expected_text = format!("Generated text with {} tokens", expected_tokens.len());

        Ok(DeterministicOutputs {
            seed,
            expected_tokens,
            expected_text,
            expected_token_probabilities,
            reproducibility_hash,
        })
    }

    /// Generate device-specific variants
    async fn generate_device_variants(&mut self) -> Result<()> {
        for test_case in &mut self.test_cases {
            // CPU variant
            let cpu_data = DeviceGenerationData {
                device: Device::Cpu,
                optimization_strategy: "Sequential_CPU".to_string(),
                batch_processing: BatchProcessingConfig {
                    max_batch_size: 1,
                    dynamic_batching: false,
                    padding_strategy: "right".to_string(),
                    batch_optimization: false,
                },
                memory_management: MemoryManagementConfig {
                    kv_cache_strategy: "static".to_string(),
                    memory_pooling: false,
                    gradient_checkpointing: false,
                    memory_optimization_level: 1,
                },
                performance_metrics: GenerationPerformanceMetrics {
                    prefill_latency_ms: 10.0,
                    decode_latency_ms: 20.0,
                    throughput_tokens_per_second: 25.0,
                    memory_bandwidth_utilization: 0.6,
                    gpu_utilization: 0.0,
                    cache_hit_rate: 0.8,
                },
            };
            test_case.device_variants.insert(Device::Cpu, cpu_data);

            // GPU variant (if available)
            #[cfg(feature = "gpu")]
            {
                let gpu_data = DeviceGenerationData {
                    device: Device::Cuda(0),
                    optimization_strategy: "Batched_GPU_Flash".to_string(),
                    batch_processing: BatchProcessingConfig {
                        max_batch_size: 32,
                        dynamic_batching: true,
                        padding_strategy: "dynamic".to_string(),
                        batch_optimization: true,
                    },
                    memory_management: MemoryManagementConfig {
                        kv_cache_strategy: "dynamic".to_string(),
                        memory_pooling: true,
                        gradient_checkpointing: true,
                        memory_optimization_level: 3,
                    },
                    performance_metrics: GenerationPerformanceMetrics {
                        prefill_latency_ms: 2.0,
                        decode_latency_ms: 5.0,
                        throughput_tokens_per_second: 150.0,
                        memory_bandwidth_utilization: 0.9,
                        gpu_utilization: 0.85,
                        cache_hit_rate: 0.95,
                    },
                };
                test_case.device_variants.insert(Device::Cuda(0), gpu_data);
            }
        }

        Ok(())
    }

    /// Initialize KV-cache patterns
    async fn initialize_kv_cache_patterns(&mut self) -> Result<()> {
        // Sequential generation pattern
        self.kv_cache_patterns.sequential_generation = KVCachePattern {
            pattern_name: "sequential_generation".to_string(),
            sequence_length: 128,
            generation_steps: 50,
            expected_cache_growth: (0..50).map(|i| 128 + i).collect(),
            expected_cache_usage: (0..50).map(|i| (128.0 + i as f32) * 0.5).collect(),
            cache_hit_patterns: vec![true; 50], // All hits for sequential
        };

        // Add other patterns...
        Ok(())
    }

    /// Precompute sampling test cases
    async fn precompute_sampling_tests(&mut self) -> Result<()> {
        // Temperature scaling tests
        let logits = vec![2.0, 1.0, 0.5, 0.1]; // Sample logits
        for &temperature in &[0.1, 0.5, 1.0, 1.5, 2.0] {
            let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
            let expected_token = self.mock_sample_from_logits(&scaled_logits).await?;
            self.sampling_test_data.temperature_tests.push((temperature, vec![expected_token]));
        }

        Ok(())
    }

    /// Mock sampling from logits (replace with real implementation)
    async fn mock_sample_from_logits(&self, logits: &[f32]) -> Result<u32> {
        // Find argmax for deterministic behavior
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx as u32)
    }

    /// Get test case by name
    pub fn get_test_case(&self, name: &str) -> Option<&GenerationTestCase> {
        self.test_cases.iter().find(|case| case.test_name == name)
    }

    /// Get deterministic output for test case
    pub fn get_deterministic_output(&self, test_name: &str) -> Option<&DeterministicOutputs> {
        self.get_test_case(test_name)?.deterministic_outputs.as_ref()
    }

    /// Validate generation output against expected results
    pub async fn validate_generation_output(
        &self,
        test_case_name: &str,
        generated_tokens: &[u32],
        generated_text: &str,
        deterministic_mode: bool,
    ) -> Result<GenerationValidationResult> {
        let test_case = self.get_test_case(test_case_name).ok_or_else(|| {
            BitNetError::Validation(format!("Test case not found: {}", test_case_name))
        })?;

        if deterministic_mode {
            if let Some(det_output) = &test_case.deterministic_outputs {
                // Strict comparison for deterministic mode
                let tokens_match = generated_tokens == det_output.expected_tokens;
                let text_match = generated_text == det_output.expected_text;

                return Ok(GenerationValidationResult {
                    test_case_name: test_case_name.to_string(),
                    passed: tokens_match && text_match,
                    tokens_match,
                    text_similarity: if text_match { 1.0 } else { 0.0 },
                    error_message: if tokens_match && text_match {
                        None
                    } else {
                        Some("Deterministic output mismatch".to_string())
                    },
                });
            }
        }

        // Non-deterministic validation (more lenient)
        let expected_length = test_case.expected_outputs.generated_tokens.len();
        let length_ok = generated_tokens.len() >= expected_length / 2
            && generated_tokens.len() <= expected_length * 2;

        let text_similarity = self
            .compute_text_similarity(generated_text, &test_case.expected_outputs.generated_text)
            .await?;

        Ok(GenerationValidationResult {
            test_case_name: test_case_name.to_string(),
            passed: length_ok && text_similarity > 0.7,
            tokens_match: false, // Not applicable for non-deterministic
            text_similarity,
            error_message: None,
        })
    }

    /// Compute text similarity between generated and expected text
    async fn compute_text_similarity(&self, generated: &str, expected: &str) -> Result<f32> {
        // Simple similarity metric (can be enhanced with more sophisticated methods)
        let generated_words: Vec<&str> = generated.split_whitespace().collect();
        let expected_words: Vec<&str> = expected.split_whitespace().collect();

        if generated_words.is_empty() && expected_words.is_empty() {
            return Ok(1.0);
        }

        if generated_words.is_empty() || expected_words.is_empty() {
            return Ok(0.0);
        }

        // Jaccard similarity
        let generated_set: std::collections::HashSet<_> = generated_words.into_iter().collect();
        let expected_set: std::collections::HashSet<_> = expected_words.into_iter().collect();

        let intersection = generated_set.intersection(&expected_set).count();
        let union = generated_set.union(&expected_set).count();

        Ok(intersection as f32 / union as f32)
    }
}

/// Generation validation result
#[derive(Debug)]
pub struct GenerationValidationResult {
    pub test_case_name: String,
    pub passed: bool,
    pub tokens_match: bool,
    pub text_similarity: f32,
    pub error_message: Option<String>,
}

/// Create deterministic generation test case
fn create_deterministic_generation_test() -> GenerationTestCase {
    GenerationTestCase {
        test_name: "deterministic_generation".to_string(),
        description: "Deterministic text generation with temperature=0".to_string(),
        config: GenerationConfig::deterministic(),
        input_data: GenerationInputData {
            input_text: "The capital of France is".to_string(),
            input_tokens: vec![1, 464, 6287, 315, 4605, 374], // Mock tokenization
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            prompt_length: 6,
            model_architecture: "BitNet-b1.58".to_string(),
            vocab_size: 32000,
            context_length: 2048,
        },
        expected_outputs: GenerationExpectedOutputs {
            generated_tokens: vec![1, 464, 6287, 315, 4605, 374, 12366], // + "Paris"
            generated_text: "The capital of France is Paris".to_string(),
            generation_logits: vec![], // Will be computed during initialization
            attention_scores: vec![],  // Will be computed during initialization
            kv_cache_states: None,
            generation_metadata: GenerationMetadata {
                steps_taken: 1,
                tokens_generated: 1,
                stopped_reason: "max_length".to_string(),
                generation_time_ms: 25.0,
                tokens_per_second: 40.0,
                memory_usage_mb: 256.0,
            },
        },
        deterministic_outputs: None, // Will be generated during initialization
        performance_targets: PerformanceTargets {
            min_tokens_per_second: 15.0,
            max_latency_ms: 100.0,
            max_memory_usage_mb: 512.0,
            min_accuracy_threshold: 0.95,
            kv_cache_efficiency: 0.9,
        },
        device_variants: HashMap::new(),
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.0, // Exact match for deterministic
            dequantization_tolerance: 0.0,
            scale_tolerance: 0.0,
            numerical_accuracy_threshold: 1.0,
        },
    }
}

/// Create nucleus sampling test case
fn create_nucleus_sampling_test() -> GenerationTestCase {
    let mut test_case = create_deterministic_generation_test();
    test_case.test_name = "nucleus_sampling_generation".to_string();
    test_case.description = "Text generation with nucleus sampling (top-p=0.9)".to_string();
    test_case.config = GenerationConfig::nucleus_sampling();
    test_case.tolerance.numerical_accuracy_threshold = 0.7; // More lenient for sampling

    test_case
}

/// Create top-k sampling test case
fn create_top_k_sampling_test() -> GenerationTestCase {
    let mut test_case = create_deterministic_generation_test();
    test_case.test_name = "top_k_sampling_generation".to_string();
    test_case.description = "Text generation with top-k sampling (k=50)".to_string();
    test_case.config = GenerationConfig::top_k_sampling();
    test_case.tolerance.numerical_accuracy_threshold = 0.7;

    test_case
}

/// Create long context generation test case
fn create_long_context_generation_test() -> GenerationTestCase {
    let mut test_case = create_deterministic_generation_test();
    test_case.test_name = "long_context_generation".to_string();
    test_case.description = "Long context text generation (1024+ tokens)".to_string();

    // Extend input for long context
    test_case.input_data.input_tokens = (1..1025).collect(); // 1024 tokens
    test_case.input_data.attention_mask = vec![1; 1024];
    test_case.input_data.prompt_length = 1024;
    test_case.input_data.context_length = 2048;

    test_case.performance_targets.min_tokens_per_second = 5.0; // Slower for long context
    test_case.performance_targets.max_memory_usage_mb = 2048.0; // More memory needed

    test_case
}

/// Create batch generation test case
fn create_batch_generation_test() -> GenerationTestCase {
    let mut test_case = create_deterministic_generation_test();
    test_case.test_name = "batch_generation".to_string();
    test_case.description = "Batched text generation with multiple prompts".to_string();

    test_case.performance_targets.min_tokens_per_second = 50.0; // Higher throughput expected

    test_case
}

/// Create streaming generation test case
fn create_streaming_generation_test() -> GenerationTestCase {
    let mut test_case = create_deterministic_generation_test();
    test_case.test_name = "streaming_generation".to_string();
    test_case.description = "Streaming text generation with incremental output".to_string();

    test_case.performance_targets.max_latency_ms = 50.0; // Lower latency for streaming

    test_case
}

/// Create sampling test data
fn create_sampling_test_data() -> SamplingTestData {
    SamplingTestData {
        logits_samples: vec![
            vec![2.0, 1.5, 1.0, 0.5, 0.1], // Peaked distribution
            vec![1.0, 1.0, 1.0, 1.0, 1.0], // Uniform distribution
            vec![5.0, 0.1, 0.1, 0.1, 0.1], // Very peaked
        ],
        temperature_tests: vec![], // Will be populated during initialization
        top_k_tests: vec![],
        top_p_tests: vec![],
        repetition_penalty_tests: vec![],
    }
}

/// Create KV-cache patterns
fn create_kv_cache_patterns() -> KVCachePatterns {
    KVCachePatterns {
        sequential_generation: KVCachePattern {
            pattern_name: "placeholder".to_string(),
            sequence_length: 0,
            generation_steps: 0,
            expected_cache_growth: vec![],
            expected_cache_usage: vec![],
            cache_hit_patterns: vec![],
        },
        batch_generation: KVCachePattern {
            pattern_name: "placeholder".to_string(),
            sequence_length: 0,
            generation_steps: 0,
            expected_cache_growth: vec![],
            expected_cache_usage: vec![],
            cache_hit_patterns: vec![],
        },
        long_context_generation: KVCachePattern {
            pattern_name: "placeholder".to_string(),
            sequence_length: 0,
            generation_steps: 0,
            expected_cache_growth: vec![],
            expected_cache_usage: vec![],
            cache_hit_patterns: vec![],
        },
        cache_eviction_patterns: vec![],
    }
}

/// Create generation fixtures for testing
#[cfg(test)]
pub fn create_generation_fixtures() -> GenerationFixtures {
    let config = TestEnvironmentConfig::from_env();
    GenerationFixtures::new(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_creation() {
        let config = GenerationConfig::deterministic();
        assert_eq!(config.temperature, 0.0);
        assert!(!config.do_sample);
        assert!(config.use_cache);
    }

    #[tokio::test]
    async fn test_generation_fixtures_initialization() {
        let mut fixtures = create_generation_fixtures();
        fixtures.initialize().await.expect("Initialization failed");

        assert!(!fixtures.test_cases.is_empty());

        // Check deterministic outputs were generated
        let det_test = fixtures.get_test_case("deterministic_generation").unwrap();
        assert!(det_test.deterministic_outputs.is_some());
    }

    #[tokio::test]
    async fn test_text_similarity_computation() {
        let fixtures = create_generation_fixtures();

        let similarity = fixtures
            .compute_text_similarity("The quick brown fox", "The quick brown dog")
            .await
            .expect("Similarity computation failed");

        assert!(similarity > 0.5); // Should be similar
        assert!(similarity < 1.0); // But not identical
    }
}
