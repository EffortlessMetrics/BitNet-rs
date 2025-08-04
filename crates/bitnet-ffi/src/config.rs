//! Configuration structures for the C API
//!
//! This module provides C-compatible configuration structures that match
//! the existing C++ API while providing enhanced functionality.

use crate::BitNetCError;
use bitnet_common::{BitNetConfig, ModelFormat, QuantizationType};
use std::os::raw::{c_char, c_int, c_uint, c_float, c_ulong};
use std::ffi::CStr;

/// C API model configuration structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetCConfig {
    /// Model file path (null-terminated string)
    pub model_path: *const c_char,
    /// Model format (0=GGUF, 1=SafeTensors, 2=HuggingFace)
    pub model_format: c_uint,
    /// Vocabulary size
    pub vocab_size: c_uint,
    /// Hidden size
    pub hidden_size: c_uint,
    /// Number of layers
    pub num_layers: c_uint,
    /// Number of attention heads
    pub num_heads: c_uint,
    /// Intermediate size
    pub intermediate_size: c_uint,
    /// Maximum position embeddings
    pub max_position_embeddings: c_uint,
    /// Quantization type (0=I2S, 1=TL1, 2=TL2)
    pub quantization_type: c_uint,
    /// Quantization block size
    pub block_size: c_uint,
    /// Quantization precision
    pub precision: c_float,
    /// Number of threads (0 for auto-detection)
    pub num_threads: c_uint,
    /// Use GPU acceleration (0=false, 1=true)
    pub use_gpu: c_uint,
    /// Batch size
    pub batch_size: c_uint,
    /// Memory limit in bytes (0 for no limit)
    pub memory_limit: c_ulong,
}

impl Default for BitNetCConfig {
    fn default() -> Self {
        Self {
            model_path: std::ptr::null(),
            model_format: 0, // GGUF
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 11008,
            max_position_embeddings: 2048,
            quantization_type: 0, // I2S
            block_size: 64,
            precision: 1e-4,
            num_threads: 0, // Auto-detect
            use_gpu: 0, // False
            batch_size: 1,
            memory_limit: 0, // No limit
        }
    }
}

impl BitNetCConfig {
    /// Convert to Rust BitNetConfig
    pub fn to_bitnet_config(&self) -> Result<BitNetConfig, BitNetCError> {
        let model_format = match self.model_format {
            0 => ModelFormat::Gguf,
            1 => ModelFormat::SafeTensors,
            2 => ModelFormat::HuggingFace,
            _ => return Err(BitNetCError::InvalidArgument(
                format!("Invalid model format: {}", self.model_format)
            )),
        };

        let quantization_type = match self.quantization_type {
            0 => QuantizationType::I2S,
            1 => QuantizationType::TL1,
            2 => QuantizationType::TL2,
            _ => return Err(BitNetCError::InvalidArgument(
                format!("Invalid quantization type: {}", self.quantization_type)
            )),
        };

        let model_path = if self.model_path.is_null() {
            None
        } else {
            let path_str = unsafe { CStr::from_ptr(self.model_path) }
                .to_str()
                .map_err(|e| BitNetCError::InvalidArgument(
                    format!("Invalid UTF-8 in model path: {}", e)
                ))?;
            Some(std::path::PathBuf::from(path_str))
        };

        let num_threads = if self.num_threads == 0 {
            None
        } else {
            Some(self.num_threads as usize)
        };

        let memory_limit = if self.memory_limit == 0 {
            None
        } else {
            Some(self.memory_limit as usize)
        };

        Ok(BitNetConfig {
            model: bitnet_common::ModelConfig {
                path: model_path,
                format: model_format,
                vocab_size: self.vocab_size as usize,
                hidden_size: self.hidden_size as usize,
                num_layers: self.num_layers as usize,
                num_heads: self.num_heads as usize,
                intermediate_size: self.intermediate_size as usize,
                max_position_embeddings: self.max_position_embeddings as usize,
            },
            inference: bitnet_common::InferenceConfig::default(),
            quantization: bitnet_common::QuantizationConfig {
                quantization_type,
                block_size: self.block_size as usize,
                precision: self.precision,
            },
            performance: bitnet_common::PerformanceConfig {
                num_threads,
                use_gpu: self.use_gpu != 0,
                batch_size: self.batch_size as usize,
                memory_limit,
            },
        })
    }

    /// Create from Rust BitNetConfig
    pub fn from_bitnet_config(config: &BitNetConfig) -> Self {
        Self {
            model_path: std::ptr::null(), // Will be set separately if needed
            model_format: match config.model.format {
                ModelFormat::Gguf => 0,
                ModelFormat::SafeTensors => 1,
                ModelFormat::HuggingFace => 2,
            },
            vocab_size: config.model.vocab_size as c_uint,
            hidden_size: config.model.hidden_size as c_uint,
            num_layers: config.model.num_layers as c_uint,
            num_heads: config.model.num_heads as c_uint,
            intermediate_size: config.model.intermediate_size as c_uint,
            max_position_embeddings: config.model.max_position_embeddings as c_uint,
            quantization_type: match config.quantization.quantization_type {
                QuantizationType::I2S => 0,
                QuantizationType::TL1 => 1,
                QuantizationType::TL2 => 2,
            },
            block_size: config.quantization.block_size as c_uint,
            precision: config.quantization.precision,
            num_threads: config.performance.num_threads.unwrap_or(0) as c_uint,
            use_gpu: if config.performance.use_gpu { 1 } else { 0 },
            batch_size: config.performance.batch_size as c_uint,
            memory_limit: config.performance.memory_limit.unwrap_or(0) as c_ulong,
        }
    }
}

/// C API inference configuration structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetCInferenceConfig {
    /// Maximum sequence length
    pub max_length: c_uint,
    /// Maximum new tokens to generate
    pub max_new_tokens: c_uint,
    /// Temperature for sampling
    pub temperature: c_float,
    /// Top-k sampling parameter (0 to disable)
    pub top_k: c_uint,
    /// Top-p sampling parameter (0.0 to disable)
    pub top_p: c_float,
    /// Repetition penalty
    pub repetition_penalty: c_float,
    /// Frequency penalty
    pub frequency_penalty: c_float,
    /// Presence penalty
    pub presence_penalty: c_float,
    /// Random seed (0 for random)
    pub seed: c_ulong,
    /// Enable sampling (0=greedy, 1=sampling)
    pub do_sample: c_uint,
    /// Backend preference (0=auto, 1=cpu, 2=gpu)
    pub backend_preference: c_uint,
    /// Enable streaming output (0=false, 1=true)
    pub enable_streaming: c_uint,
    /// Streaming buffer size
    pub stream_buffer_size: c_uint,
}

impl Default for BitNetCInferenceConfig {
    fn default() -> Self {
        Self {
            max_length: 2048,
            max_new_tokens: 512,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
            do_sample: 1,
            backend_preference: 0, // Auto
            enable_streaming: 0,
            stream_buffer_size: 64,
        }
    }
}

impl BitNetCInferenceConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), BitNetCError> {
        if self.max_length == 0 {
            return Err(BitNetCError::InvalidArgument(
                "max_length must be greater than 0".to_string()
            ));
        }

        if self.max_new_tokens == 0 {
            return Err(BitNetCError::InvalidArgument(
                "max_new_tokens must be greater than 0".to_string()
            ));
        }

        if self.temperature <= 0.0 {
            return Err(BitNetCError::InvalidArgument(
                "temperature must be greater than 0".to_string()
            ));
        }

        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(BitNetCError::InvalidArgument(
                "top_p must be between 0.0 and 1.0".to_string()
            ));
        }

        if self.repetition_penalty <= 0.0 {
            return Err(BitNetCError::InvalidArgument(
                "repetition_penalty must be greater than 0".to_string()
            ));
        }

        if self.backend_preference > 2 {
            return Err(BitNetCError::InvalidArgument(
                "backend_preference must be 0 (auto), 1 (cpu), or 2 (gpu)".to_string()
            ));
        }

        if self.stream_buffer_size == 0 {
            return Err(BitNetCError::InvalidArgument(
                "stream_buffer_size must be greater than 0".to_string()
            ));
        }

        Ok(())
    }

    /// Convert to Rust GenerationConfig
    pub fn to_generation_config(&self) -> bitnet_common::GenerationConfig {
        bitnet_common::GenerationConfig {
            max_new_tokens: self.max_new_tokens as usize,
            temperature: self.temperature,
            top_k: if self.top_k == 0 { None } else { Some(self.top_k as usize) },
            top_p: if self.top_p == 0.0 { None } else { Some(self.top_p) },
            repetition_penalty: self.repetition_penalty,
            do_sample: self.do_sample != 0,
            seed: if self.seed == 0 { None } else { Some(self.seed as u64) },
        }
    }
}

/// C API performance metrics structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetCPerformanceMetrics {
    /// Tokens per second
    pub tokens_per_second: c_float,
    /// Latency in milliseconds
    pub latency_ms: c_float,
    /// Memory usage in MB
    pub memory_usage_mb: c_float,
    /// GPU utilization percentage (0-100, -1 if not available)
    pub gpu_utilization: c_float,
    /// Total inference time in milliseconds
    pub total_inference_time_ms: c_float,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: c_float,
    /// Number of tokens generated
    pub tokens_generated: c_uint,
    /// Number of tokens in prompt
    pub prompt_tokens: c_uint,
}

impl Default for BitNetCPerformanceMetrics {
    fn default() -> Self {
        Self {
            tokens_per_second: 0.0,
            latency_ms: 0.0,
            memory_usage_mb: 0.0,
            gpu_utilization: -1.0,
            total_inference_time_ms: 0.0,
            time_to_first_token_ms: 0.0,
            tokens_generated: 0,
            prompt_tokens: 0,
        }
    }
}

impl BitNetCPerformanceMetrics {
    /// Create from Rust PerformanceMetrics
    pub fn from_performance_metrics(metrics: &bitnet_common::PerformanceMetrics) -> Self {
        Self {
            tokens_per_second: metrics.tokens_per_second as c_float,
            latency_ms: metrics.latency_ms as c_float,
            memory_usage_mb: metrics.memory_usage_mb as c_float,
            gpu_utilization: metrics.gpu_utilization.unwrap_or(-1.0) as c_float,
            total_inference_time_ms: metrics.latency_ms as c_float,
            time_to_first_token_ms: 0.0, // Not available in base metrics
            tokens_generated: 0, // Not available in base metrics
            prompt_tokens: 0, // Not available in base metrics
        }
    }
}

/// C API streaming callback function type
pub type BitNetCStreamCallback = Option<extern "C" fn(
    token: *const c_char,
    user_data: *mut std::ffi::c_void,
) -> c_int>;

/// C API streaming configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetCStreamConfig {
    /// Callback function for streaming tokens
    pub callback: Option<BitNetCStreamCallback>,
    /// User data passed to callback
    pub user_data: *mut std::ffi::c_void,
    /// Buffer size for streaming
    pub buffer_size: c_uint,
    /// Yield interval in tokens
    pub yield_interval: c_uint,
    /// Enable backpressure handling
    pub enable_backpressure: c_uint,
    /// Timeout in milliseconds
    pub timeout_ms: c_uint,
}

impl Default for BitNetCStreamConfig {
    fn default() -> Self {
        Self {
            callback: None,
            user_data: std::ptr::null_mut(),
            buffer_size: 64,
            yield_interval: 1,
            enable_backpressure: 1,
            timeout_ms: 5000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_conversion() {
        let c_config = BitNetCConfig::default();
        let rust_config = c_config.to_bitnet_config().unwrap();
        
        assert_eq!(rust_config.model.vocab_size, 32000);
        assert_eq!(rust_config.model.hidden_size, 4096);
        assert_eq!(rust_config.quantization.quantization_type, QuantizationType::I2S);
    }

    #[test]
    fn test_inference_config_validation() {
        let mut config = BitNetCInferenceConfig::default();
        assert!(config.validate().is_ok());

        config.temperature = 0.0;
        assert!(config.validate().is_err());

        config.temperature = 1.0;
        config.top_p = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_performance_metrics_conversion() {
        let rust_metrics = bitnet_common::PerformanceMetrics {
            tokens_per_second: 100.0,
            latency_ms: 50.0,
            memory_usage_mb: 1024.0,
            gpu_utilization: Some(75.0),
        };

        let c_metrics = BitNetCPerformanceMetrics::from_performance_metrics(&rust_metrics);
        assert_eq!(c_metrics.tokens_per_second, 100.0);
        assert_eq!(c_metrics.latency_ms, 50.0);
        assert_eq!(c_metrics.memory_usage_mb, 1024.0);
        assert_eq!(c_metrics.gpu_utilization, 75.0);
    }

    #[test]
    fn test_generation_config_conversion() {
        let c_config = BitNetCInferenceConfig {
            max_new_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.2,
            do_sample: 1,
            seed: 42,
            ..Default::default()
        };

        let gen_config = c_config.to_generation_config();
        assert_eq!(gen_config.max_new_tokens, 256);
        assert_eq!(gen_config.temperature, 0.8);
        assert_eq!(gen_config.top_k, Some(40));
        assert_eq!(gen_config.top_p, Some(0.95));
        assert_eq!(gen_config.repetition_penalty, 1.2);
        assert_eq!(gen_config.do_sample, true);
        assert_eq!(gen_config.seed, Some(42));
    }
}