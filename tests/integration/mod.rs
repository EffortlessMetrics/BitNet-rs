//! # Integration Tests Module
//!
//! This module contains integration tests that validate complete workflows
//! across the BitNet Rust ecosystem components.

pub mod batch_processing_tests;
pub mod component_interaction_tests;
pub mod model_loading_tests;
pub mod resource_management_tests;
pub mod streaming_tests;
pub mod tokenization_pipeline_tests;
pub mod workflow_tests;

use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device, MockTensor};
use bitnet_inference::{GenerationConfig, InferenceConfig, InferenceEngine};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

/// Mock model implementation for integration tests
pub struct MockModel {
    config: BitNetConfig,
    forward_calls: Arc<std::sync::Mutex<usize>>,
}

impl MockModel {
    pub fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
            forward_calls: Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn forward_call_count(&self) -> usize {
        *self.forward_calls.lock().unwrap()
    }
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &bitnet_common::ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<bitnet_common::ConcreteTensor, BitNetError> {
        // Increment call counter
        *self.forward_calls.lock().unwrap() += 1;

        // Return mock output tensor with vocab size dimensions
        Ok(MockTensor::new(vec![1, 50257]))
    }
}

/// Mock tokenizer implementation for integration tests
pub struct MockTokenizer {
    vocab_size: usize,
    encode_calls: Arc<std::sync::Mutex<usize>>,
    decode_calls: Arc<std::sync::Mutex<usize>>,
}

impl MockTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            encode_calls: Arc::new(std::sync::Mutex::new(0)),
            decode_calls: Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn encode_call_count(&self) -> usize {
        *self.encode_calls.lock().unwrap()
    }

    pub fn decode_call_count(&self) -> usize {
        *self.decode_calls.lock().unwrap()
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>, BitNetError> {
        *self.encode_calls.lock().unwrap() += 1;

        // Simple mock encoding: convert text length to tokens
        let tokens = (0..text.len().min(10)).map(|i| (i + 1) as u32).collect();
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String, BitNetError> {
        *self.decode_calls.lock().unwrap() += 1;

        // Simple mock decoding: create text based on token count
        Ok(format!("generated_text_{}_tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(50257)
    }
}

/// Test data generator for integration tests
pub struct IntegrationTestData;

impl IntegrationTestData {
    /// Generate test prompts of various lengths
    pub fn test_prompts() -> Vec<String> {
        vec![
            "Hello".to_string(),
            "Hello, world!".to_string(),
            "This is a longer test prompt for integration testing.".to_string(),
            "Multi-line\ntest prompt\nwith newlines.".to_string(),
            "Special characters: !@#$%^&*()".to_string(),
        ]
    }

    /// Generate test configurations
    pub fn test_generation_configs() -> Vec<GenerationConfig> {
        vec![
            GenerationConfig::greedy(),
            GenerationConfig::balanced(),
            GenerationConfig::creative(),
            GenerationConfig {
                max_new_tokens: 5,
                temperature: 0.5,
                ..Default::default()
            },
        ]
    }

    /// Generate test inference configurations
    pub fn test_inference_configs() -> Vec<InferenceConfig> {
        vec![
            InferenceConfig::cpu_optimized(),
            InferenceConfig::memory_efficient(),
            InferenceConfig {
                max_context_length: 512,
                batch_size: 1,
                ..Default::default()
            },
        ]
    }
}
