//! Configuration types and utilities

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main BitNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetConfig {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub quantization: QuantizationConfig,
    pub performance: PerformanceConfig,
}

impl Default for BitNetConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            inference: InferenceConfig::default(),
            quantization: QuantizationConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: Option<PathBuf>,
    pub format: ModelFormat,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: None,
            format: ModelFormat::Gguf,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 11008,
            max_position_embeddings: 2048,
        }
    }
}

/// Supported model formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    HuggingFace,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_length: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_length: 2048,
            max_new_tokens: 512,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub block_size: usize,
    pub precision: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::I2S,
            block_size: 64,
            precision: 1e-4,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub num_threads: Option<usize>,
    pub use_gpu: bool,
    pub batch_size: usize,
    pub memory_limit: Option<usize>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            use_gpu: false,
            batch_size: 1,
            memory_limit: None,
        }
    }
}