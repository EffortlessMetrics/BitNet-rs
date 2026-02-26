#![allow(unused)]
#![allow(dead_code)]

//! GGUF Model Test Fixtures for BitNet-rs Inference Server
//!
//! This module provides realistic GGUF model fixtures for testing the production
//! inference server. All fixtures support I2S, TL1, and TL2 quantization formats
//! with proper tensor alignment and device-aware validation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTestModel {
    pub file_path: &'static str,
    pub expected_tensors: usize,
    pub vocab_size: u32,
    pub model_type: &'static str,
    pub alignment: u64,
    pub weight_mapper_compatible: bool,
    pub tensor_alignment_valid: bool,
    pub quantization_type: QuantizationType,
    pub model_size_bytes: u64,
    pub parameter_count: u64,
    pub context_length: u32,
    pub embedding_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    /// I2S (2-bit signed) - Production quantization with 99%+ accuracy
    I2S,
    /// TL1 (Table Lookup 1) - Fast lookup quantization for CPU
    TL1,
    /// TL2 (Table Lookup 2) - Advanced lookup quantization for GPU
    TL2,
    /// IQ2_S (GGML-compatible) - FFI bridge quantization
    IQ2S,
    /// FP16 - Half precision for testing
    FP16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTestMetadata {
    pub description: &'static str,
    pub use_case: &'static str,
    pub expected_accuracy: f32,
    pub min_memory_mb: u64,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub supports_streaming: bool,
    pub test_prompt: &'static str,
    pub expected_response_pattern: &'static str,
}

/// Small test model (I2S quantized, ~25MB) for basic functionality testing
pub static SMALL_I2S_MODEL: LazyLock<GgufTestModel> = LazyLock::new(|| GgufTestModel {
    file_path: "tests/fixtures/models/small_i2s_model.gguf",
    expected_tensors: 48,
    vocab_size: 32000,
    model_type: "bitnet",
    alignment: 32,
    weight_mapper_compatible: true,
    tensor_alignment_valid: true,
    quantization_type: QuantizationType::I2S,
    model_size_bytes: 25 * 1024 * 1024, // 25MB
    parameter_count: 160_000_000,       // 160M parameters
    context_length: 2048,
    embedding_dim: 768,
    num_layers: 12,
    num_heads: 12,
});

/// Medium test model (TL1 quantized, ~100MB) for performance testing
pub static MEDIUM_TL1_MODEL: LazyLock<GgufTestModel> = LazyLock::new(|| GgufTestModel {
    file_path: "tests/fixtures/models/medium_tl1_model.gguf",
    expected_tensors: 96,
    vocab_size: 32000,
    model_type: "bitnet",
    alignment: 32,
    weight_mapper_compatible: true,
    tensor_alignment_valid: true,
    quantization_type: QuantizationType::TL1,
    model_size_bytes: 100 * 1024 * 1024, // 100MB
    parameter_count: 500_000_000,        // 500M parameters
    context_length: 4096,
    embedding_dim: 1024,
    num_layers: 24,
    num_heads: 16,
});

/// Large test model (TL2 quantized, ~500MB) for stress testing
pub static LARGE_TL2_MODEL: LazyLock<GgufTestModel> = LazyLock::new(|| GgufTestModel {
    file_path: "tests/fixtures/models/large_tl2_model.gguf",
    expected_tensors: 192,
    vocab_size: 50257,
    model_type: "bitnet",
    alignment: 32,
    weight_mapper_compatible: true,
    tensor_alignment_valid: true,
    quantization_type: QuantizationType::TL2,
    model_size_bytes: 500 * 1024 * 1024, // 500MB
    parameter_count: 2_000_000_000,      // 2B parameters
    context_length: 8192,
    embedding_dim: 2048,
    num_layers: 48,
    num_heads: 32,
});

/// Invalid model for error handling testing
pub static INVALID_MODEL: LazyLock<GgufTestModel> = LazyLock::new(|| GgufTestModel {
    file_path: "tests/fixtures/models/invalid_model.gguf",
    expected_tensors: 0,
    vocab_size: 0,
    model_type: "corrupted",
    alignment: 0,
    weight_mapper_compatible: false,
    tensor_alignment_valid: false,
    quantization_type: QuantizationType::I2S,
    model_size_bytes: 1024, // 1KB corrupted file
    parameter_count: 0,
    context_length: 0,
    embedding_dim: 0,
    num_layers: 0,
    num_heads: 0,
});

/// Metadata for test models
pub static MODEL_METADATA: LazyLock<std::collections::HashMap<&'static str, ModelTestMetadata>> =
    LazyLock::new(|| {
        let mut metadata = std::collections::HashMap::new();

        metadata.insert("small_i2s", ModelTestMetadata {
        description: "Small BitNet model with I2S quantization for basic functionality testing",
        use_case: "Unit tests, basic inference validation, CI/CD pipelines",
        expected_accuracy: 0.99,
        min_memory_mb: 64,
        supports_cpu: true,
        supports_gpu: true,
        supports_streaming: true,
        test_prompt: "What is neural network quantization?",
        expected_response_pattern: r"(?i)quantization.*neural.*network",
    });

        metadata.insert(
            "medium_tl1",
            ModelTestMetadata {
                description: "Medium BitNet model with TL1 quantization for performance testing",
                use_case: "Performance benchmarks, concurrent request testing, memory validation",
                expected_accuracy: 0.98,
                min_memory_mb: 256,
                supports_cpu: true,
                supports_gpu: true,
                supports_streaming: true,
                test_prompt: "Explain the benefits of 1-bit neural networks.",
                expected_response_pattern: r"(?i)(1-bit|one-bit).*neural.*network",
            },
        );

        metadata.insert("large_tl2", ModelTestMetadata {
        description: "Large BitNet model with TL2 quantization for stress testing",
        use_case: "Load testing, production simulation, GPU memory validation",
        expected_accuracy: 0.98,
        min_memory_mb: 1024,
        supports_cpu: false, // Too large for CPU-only testing
        supports_gpu: true,
        supports_streaming: true,
        test_prompt: "Write a detailed explanation of BitNet architecture and its advantages.",
        expected_response_pattern: r"(?i)bitnet.*architecture.*advantage",
    });

        metadata.insert(
            "invalid",
            ModelTestMetadata {
                description: "Corrupted GGUF file for error handling validation",
                use_case: "Error handling, graceful degradation, security testing",
                expected_accuracy: 0.0,
                min_memory_mb: 0,
                supports_cpu: false,
                supports_gpu: false,
                supports_streaming: false,
                test_prompt: "",
                expected_response_pattern: "",
            },
        );

        metadata
    });

/// Get all available test models
pub fn get_all_test_models() -> Vec<&'static GgufTestModel> {
    vec![&*SMALL_I2S_MODEL, &*MEDIUM_TL1_MODEL, &*LARGE_TL2_MODEL, &*INVALID_MODEL]
}

/// Get test models filtered by quantization type
#[cfg(feature = "cpu")]
pub fn get_cpu_compatible_models() -> Vec<&'static GgufTestModel> {
    vec![&*SMALL_I2S_MODEL, &*MEDIUM_TL1_MODEL]
}

/// Get test models for GPU testing
#[cfg(feature = "gpu")]
pub fn get_gpu_compatible_models() -> Vec<&'static GgufTestModel> {
    vec![&*SMALL_I2S_MODEL, &*MEDIUM_TL1_MODEL, &*LARGE_TL2_MODEL]
}

/// Get models for performance benchmarking
pub fn get_benchmark_models() -> Vec<&'static GgufTestModel> {
    vec![&*MEDIUM_TL1_MODEL, &*LARGE_TL2_MODEL]
}

/// Get model by quantization type
pub fn get_model_by_quantization(quant_type: QuantizationType) -> Option<&'static GgufTestModel> {
    match quant_type {
        QuantizationType::I2S => Some(&*SMALL_I2S_MODEL),
        QuantizationType::TL1 => Some(&*MEDIUM_TL1_MODEL),
        QuantizationType::TL2 => Some(&*LARGE_TL2_MODEL),
        _ => None,
    }
}

/// Get absolute path to fixture file
pub fn get_fixture_path(relative_path: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(relative_path);
    path
}

/// Check if model file exists
pub fn model_file_exists(model: &GgufTestModel) -> bool {
    get_fixture_path(model.file_path).exists()
}

/// Get model metadata
pub fn get_model_metadata(model_key: &str) -> Option<&'static ModelTestMetadata> {
    MODEL_METADATA.get(model_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_fixture_availability() {
        let models = get_all_test_models();
        assert_eq!(models.len(), 4);

        // Verify each model has valid metadata
        for model in models {
            assert!(model.expected_tensors > 0 || model.file_path.contains("invalid"));
            assert!(model.alignment == 32 || model.file_path.contains("invalid"));
        }
    }

    #[test]
    fn test_quantization_type_filtering() {
        let i2s_model = get_model_by_quantization(QuantizationType::I2S);
        assert!(i2s_model.is_some());
        assert_eq!(i2s_model.unwrap().quantization_type, QuantizationType::I2S);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cpu_model_compatibility() {
        let cpu_models = get_cpu_compatible_models();
        assert!(!cpu_models.is_empty());

        for model in cpu_models {
            assert!(model.model_size_bytes <= 200 * 1024 * 1024); // Max 200MB for CPU
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_model_compatibility() {
        let gpu_models = get_gpu_compatible_models();
        assert!(!gpu_models.is_empty());

        for model in gpu_models {
            assert!(model.weight_mapper_compatible);
        }
    }

    #[test]
    fn test_model_metadata_completeness() {
        for (key, metadata) in MODEL_METADATA.iter() {
            assert!(!metadata.description.is_empty());
            assert!(!metadata.use_case.is_empty());
            assert!(metadata.expected_accuracy >= 0.0 && metadata.expected_accuracy <= 1.0);
        }
    }
}
