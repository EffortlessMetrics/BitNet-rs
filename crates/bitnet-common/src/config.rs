//! Configuration types and utilities

use crate::{BitNetError, QuantizationType};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

/// Main BitNet configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BitNetConfig {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub quantization: QuantizationConfig,
    pub performance: PerformanceConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub path: Option<PathBuf>,
    pub format: ModelFormat,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// GQA/MQA: number of K/V heads (defaults to num_heads for MHA)
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<RopeScaling>,
    /// RMSNorm epsilon for numerical stability
    pub rms_norm_eps: Option<f32>,
    /// Normalization layer type (LayerNorm or RmsNorm)
    pub norm_type: NormType,
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
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
            num_key_value_heads: 0, // will default to num_heads if not set
            intermediate_size: 11008,
            max_position_embeddings: 2048,
            rope_theta: None,
            rope_scaling: None,
            rms_norm_eps: None,
            norm_type: NormType::default(),
            tokenizer: TokenizerConfig::default(),
        }
    }
}

/// Normalization layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormType {
    /// LayerNorm with mean subtraction (BitNet default)
    #[default]
    LayerNorm,
    /// RMSNorm without mean subtraction (LLaMA/Phi/Mistral)
    RmsNorm,
}

/// Supported model formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFormat {
    #[default]
    Gguf,
    SafeTensors,
    HuggingFace,
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub factor: f32,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TokenizerConfig {
    /// Beginning of sequence token ID
    pub bos_id: Option<i32>,
    /// End of sequence token ID
    pub eos_id: Option<i32>,
    /// Unknown token ID
    pub unk_id: Option<i32>,
    /// Padding token ID
    pub pad_id: Option<i32>,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub max_length: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    /// Add BOS token at start of prompt (from tokenizer config)
    pub add_bos: bool,
    /// Append EOS token at end of generation (from tokenizer config)
    pub append_eos: bool,
    /// Mask padding tokens in attention (from tokenizer config)
    pub mask_pad: bool,
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
            add_bos: true,
            append_eos: false,
            mask_pad: true,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub block_size: usize,
    pub precision: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self { quantization_type: QuantizationType::I2S, block_size: 64, precision: 1e-4 }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    pub num_threads: Option<usize>,
    pub use_gpu: bool,
    pub batch_size: usize,
    pub memory_limit: Option<usize>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self { num_threads: None, use_gpu: false, batch_size: 1, memory_limit: None }
    }
}

/// Configuration validation and loading utilities
impl BitNetConfig {
    /// Load configuration from file only (no environment overrides)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BitNetError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            BitNetError::Config(format!("Failed to read config file {}: {}", path.display(), e))
        })?;

        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str::<Self>(&content)
                .map_err(|e| BitNetError::Config(format!("Failed to parse TOML config: {}", e)))?,
            Some("json") => serde_json::from_str::<Self>(&content)
                .map_err(|e| BitNetError::Config(format!("Failed to parse JSON config: {}", e)))?,
            _ => {
                return Err(BitNetError::Config(
                    "Unsupported config file format. Use .toml or .json".to_string(),
                ));
            }
        };

        // Validate the configuration
        config.validate()?;

        Ok(config)
    }

    /// Load configuration from file with environment variable overrides
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self, BitNetError> {
        let mut config = Self::from_file(path)?;
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Create configuration with environment variable overrides applied to defaults
    pub fn from_env() -> Result<Self, BitNetError> {
        let mut config = Self::default();
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Merge this configuration with another, giving precedence to the other
    pub fn merge_with(&mut self, other: Self) {
        // Model config merging
        if other.model.path.is_some() {
            self.model.path = other.model.path;
        }
        if other.model.vocab_size != ModelConfig::default().vocab_size {
            self.model.vocab_size = other.model.vocab_size;
        }
        if other.model.hidden_size != ModelConfig::default().hidden_size {
            self.model.hidden_size = other.model.hidden_size;
        }
        if other.model.num_layers != ModelConfig::default().num_layers {
            self.model.num_layers = other.model.num_layers;
        }
        if other.model.num_heads != ModelConfig::default().num_heads {
            self.model.num_heads = other.model.num_heads;
        }
        if other.model.num_key_value_heads != ModelConfig::default().num_key_value_heads {
            self.model.num_key_value_heads = other.model.num_key_value_heads;
        }
        if other.model.intermediate_size != ModelConfig::default().intermediate_size {
            self.model.intermediate_size = other.model.intermediate_size;
        }
        if other.model.max_position_embeddings != ModelConfig::default().max_position_embeddings {
            self.model.max_position_embeddings = other.model.max_position_embeddings;
        }

        // Inference config merging
        if other.inference.max_length != InferenceConfig::default().max_length {
            self.inference.max_length = other.inference.max_length;
        }
        if other.inference.max_new_tokens != InferenceConfig::default().max_new_tokens {
            self.inference.max_new_tokens = other.inference.max_new_tokens;
        }
        if other.inference.temperature != InferenceConfig::default().temperature {
            self.inference.temperature = other.inference.temperature;
        }
        if other.inference.top_k != InferenceConfig::default().top_k {
            self.inference.top_k = other.inference.top_k;
        }
        if other.inference.top_p != InferenceConfig::default().top_p {
            self.inference.top_p = other.inference.top_p;
        }
        if other.inference.repetition_penalty != InferenceConfig::default().repetition_penalty {
            self.inference.repetition_penalty = other.inference.repetition_penalty;
        }
        if other.inference.seed.is_some() {
            self.inference.seed = other.inference.seed;
        }

        // Quantization config merging
        if other.quantization.quantization_type != QuantizationConfig::default().quantization_type {
            self.quantization.quantization_type = other.quantization.quantization_type;
        }
        if other.quantization.block_size != QuantizationConfig::default().block_size {
            self.quantization.block_size = other.quantization.block_size;
        }
        if other.quantization.precision != QuantizationConfig::default().precision {
            self.quantization.precision = other.quantization.precision;
        }

        // Performance config merging
        if other.performance.num_threads.is_some() {
            self.performance.num_threads = other.performance.num_threads;
        }
        if other.performance.use_gpu != PerformanceConfig::default().use_gpu {
            self.performance.use_gpu = other.performance.use_gpu;
        }
        if other.performance.batch_size != PerformanceConfig::default().batch_size {
            self.performance.batch_size = other.performance.batch_size;
        }
        if other.performance.memory_limit.is_some() {
            self.performance.memory_limit = other.performance.memory_limit;
        }
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) -> Result<(), BitNetError> {
        // Model configuration overrides
        if let Ok(path) = env::var("BITNET_MODEL_PATH") {
            self.model.path = Some(PathBuf::from(path));
        }
        if let Ok(format) = env::var("BITNET_MODEL_FORMAT") {
            self.model.format = match format.to_lowercase().as_str() {
                "gguf" => ModelFormat::Gguf,
                "safetensors" => ModelFormat::SafeTensors,
                "huggingface" => ModelFormat::HuggingFace,
                _ => {
                    return Err(BitNetError::Config(format!(
                        "Invalid model format '{}'. Use 'gguf', 'safetensors', or 'huggingface'",
                        format
                    )));
                }
            };
        }
        if let Ok(vocab_size) = env::var("BITNET_VOCAB_SIZE") {
            self.model.vocab_size = vocab_size.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_VOCAB_SIZE value: '{}'", vocab_size))
            })?;
        }
        if let Ok(hidden_size) = env::var("BITNET_HIDDEN_SIZE") {
            self.model.hidden_size = hidden_size.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_HIDDEN_SIZE value: '{}'", hidden_size))
            })?;
        }
        if let Ok(num_layers) = env::var("BITNET_NUM_LAYERS") {
            self.model.num_layers = num_layers.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_NUM_LAYERS value: '{}'", num_layers))
            })?;
        }
        if let Ok(num_heads) = env::var("BITNET_NUM_HEADS") {
            self.model.num_heads = num_heads.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_NUM_HEADS value: '{}'", num_heads))
            })?;
        }

        // Inference configuration overrides
        if let Ok(max_length) = env::var("BITNET_MAX_LENGTH") {
            self.inference.max_length = max_length.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_MAX_LENGTH value: '{}'", max_length))
            })?;
        }
        if let Ok(max_new_tokens) = env::var("BITNET_MAX_NEW_TOKENS") {
            self.inference.max_new_tokens = max_new_tokens.parse().map_err(|_| {
                BitNetError::Config(format!(
                    "Invalid BITNET_MAX_NEW_TOKENS value: '{}'",
                    max_new_tokens
                ))
            })?;
        }
        if let Ok(temperature) = env::var("BITNET_TEMPERATURE") {
            self.inference.temperature = temperature.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_TEMPERATURE value: '{}'", temperature))
            })?;
        }
        if let Ok(top_k) = env::var("BITNET_TOP_K") {
            if top_k.to_lowercase() == "none" {
                self.inference.top_k = None;
            } else {
                self.inference.top_k = Some(top_k.parse().map_err(|_| {
                    BitNetError::Config(format!("Invalid BITNET_TOP_K value: '{}'", top_k))
                })?);
            }
        }
        if let Ok(top_p) = env::var("BITNET_TOP_P") {
            if top_p.to_lowercase() == "none" {
                self.inference.top_p = None;
            } else {
                self.inference.top_p = Some(top_p.parse().map_err(|_| {
                    BitNetError::Config(format!("Invalid BITNET_TOP_P value: '{}'", top_p))
                })?);
            }
        }
        if let Ok(seed) = env::var("BITNET_SEED") {
            if seed.to_lowercase() == "none" {
                self.inference.seed = None;
            } else {
                self.inference.seed = Some(seed.parse().map_err(|_| {
                    BitNetError::Config(format!("Invalid BITNET_SEED value: '{}'", seed))
                })?);
            }
        }

        // Quantization configuration overrides
        if let Ok(qtype) = env::var("BITNET_QUANTIZATION_TYPE") {
            self.quantization.quantization_type = match qtype.to_uppercase().as_str() {
                "I2S" | "I2_S" => QuantizationType::I2S,
                "TL1" => QuantizationType::TL1,
                "TL2" => QuantizationType::TL2,
                _ => {
                    return Err(BitNetError::Config(format!(
                        "Invalid quantization type '{}'. Use 'I2S', 'TL1', or 'TL2'",
                        qtype
                    )));
                }
            };
        }
        if let Ok(block_size) = env::var("BITNET_BLOCK_SIZE") {
            self.quantization.block_size = block_size.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_BLOCK_SIZE value: '{}'", block_size))
            })?;
        }

        // Performance configuration overrides
        if let Ok(num_threads) = env::var("BITNET_NUM_THREADS") {
            if num_threads.to_lowercase() == "auto" {
                self.performance.num_threads = None;
            } else {
                self.performance.num_threads = Some(num_threads.parse().map_err(|_| {
                    BitNetError::Config(format!(
                        "Invalid BITNET_NUM_THREADS value: '{}'",
                        num_threads
                    ))
                })?);
            }
        }
        if let Ok(use_gpu) = env::var("BITNET_USE_GPU") {
            self.performance.use_gpu = match use_gpu.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => true,
                "false" | "0" | "no" | "off" => false,
                _ => {
                    return Err(BitNetError::Config(format!(
                        "Invalid BITNET_USE_GPU value '{}'. Use 'true' or 'false'",
                        use_gpu
                    )));
                }
            };
        }
        if let Ok(batch_size) = env::var("BITNET_BATCH_SIZE") {
            self.performance.batch_size = batch_size.parse().map_err(|_| {
                BitNetError::Config(format!("Invalid BITNET_BATCH_SIZE value: '{}'", batch_size))
            })?;
        }
        match env::var("BITNET_MEMORY_LIMIT") {
            Ok(memory_limit) => {
                if memory_limit.to_lowercase() == "none" {
                    self.performance.memory_limit = None;
                } else {
                    // Parse memory limit with optional unit suffix (K/KB, M/MB, G/GB, etc.)
                    let memory_limit = memory_limit.trim().to_uppercase().replace('_', "");
                    let (value, multiplier) = if memory_limit.ends_with("GIB") {
                        (memory_limit.trim_end_matches("GIB"), 1024 * 1024 * 1024)
                    } else if memory_limit.ends_with("GB") {
                        (memory_limit.trim_end_matches("GB"), 1024 * 1024 * 1024)
                    } else if memory_limit.ends_with("G") {
                        (memory_limit.trim_end_matches("G"), 1024 * 1024 * 1024)
                    } else if memory_limit.ends_with("MIB") {
                        (memory_limit.trim_end_matches("MIB"), 1024 * 1024)
                    } else if memory_limit.ends_with("MB") {
                        (memory_limit.trim_end_matches("MB"), 1024 * 1024)
                    } else if memory_limit.ends_with("M") {
                        (memory_limit.trim_end_matches("M"), 1024 * 1024)
                    } else if memory_limit.ends_with("KIB") {
                        (memory_limit.trim_end_matches("KIB"), 1024)
                    } else if memory_limit.ends_with("KB") {
                        (memory_limit.trim_end_matches("KB"), 1024)
                    } else if memory_limit.ends_with("K") {
                        (memory_limit.trim_end_matches("K"), 1024)
                    } else if memory_limit.ends_with("B") {
                        (memory_limit.trim_end_matches("B"), 1)
                    } else {
                        (memory_limit.as_str(), 1)
                    };

                    let base_value: usize = value.trim().parse::<usize>().map_err(|_| {
                        BitNetError::Config(format!(
                            "Invalid BITNET_MEMORY_LIMIT value: '{}'",
                            memory_limit
                        ))
                    })?;

                    // Safe multiplication with overflow checking for security
                    let bytes: usize = base_value.checked_mul(multiplier).ok_or_else(|| {
                        BitNetError::Config(format!(
                            "Memory limit value too large and would overflow: '{}' * {}",
                            base_value, multiplier
                        ))
                    })?;
                    self.performance.memory_limit = Some(bytes);
                }
            }
            Err(_) => {
                // Environment variable not set, keep current value
            }
        }

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), BitNetError> {
        let mut errors = Vec::new();

        // Model validation
        if self.model.vocab_size == 0 {
            errors.push("vocab_size must be greater than 0".to_string());
        }
        if self.model.hidden_size == 0 {
            errors.push("hidden_size must be greater than 0".to_string());
        }
        if self.model.num_layers == 0 {
            errors.push("num_layers must be greater than 0".to_string());
        }
        if self.model.num_heads == 0 {
            errors.push("num_heads must be greater than 0".to_string());
        } else if !self.model.hidden_size.is_multiple_of(self.model.num_heads) {
            errors.push("hidden_size must be divisible by num_heads".to_string());
        }
        // Validate num_key_value_heads if set (0 means use num_heads)
        if self.model.num_key_value_heads > 0 {
            if self.model.num_key_value_heads > self.model.num_heads {
                errors.push("num_key_value_heads cannot be greater than num_heads".to_string());
            } else if !self.model.num_heads.is_multiple_of(self.model.num_key_value_heads) {
                errors.push("num_heads must be divisible by num_key_value_heads".to_string());
            }
        }
        if self.model.intermediate_size == 0 {
            errors.push("intermediate_size must be greater than 0".to_string());
        }
        if self.model.max_position_embeddings == 0 {
            errors.push("max_position_embeddings must be greater than 0".to_string());
        }

        // Inference validation
        if self.inference.max_length == 0 {
            errors.push("max_length must be greater than 0".to_string());
        }
        if self.inference.max_new_tokens == 0 {
            errors.push("max_new_tokens must be greater than 0".to_string());
        }
        if self.inference.temperature <= 0.0 {
            errors.push("temperature must be greater than 0".to_string());
        }
        if let Some(top_k) = self.inference.top_k
            && top_k == 0
        {
            errors.push("top_k must be greater than 0 when specified".to_string());
        }
        if let Some(top_p) = self.inference.top_p
            && (top_p <= 0.0 || top_p > 1.0)
        {
            errors.push("top_p must be between 0 and 1 when specified".to_string());
        }
        if self.inference.repetition_penalty <= 0.0 {
            errors.push("repetition_penalty must be greater than 0".to_string());
        }

        // Quantization validation
        if self.quantization.block_size == 0 {
            errors.push("block_size must be greater than 0".to_string());
        }
        if !self.quantization.block_size.is_power_of_two() {
            errors.push("block_size should be a power of 2 for optimal performance".to_string());
        }
        if self.quantization.precision <= 0.0 {
            errors.push("precision must be greater than 0".to_string());
        }

        // Performance validation
        if let Some(num_threads) = self.performance.num_threads
            && num_threads == 0
        {
            errors.push("num_threads must be greater than 0 when specified".to_string());
        }
        if self.performance.batch_size == 0 {
            errors.push("batch_size must be greater than 0".to_string());
        }

        if !errors.is_empty() {
            return Err(BitNetError::Config(format!(
                "Configuration validation failed:\n{}",
                errors.join("\n")
            )));
        }

        Ok(())
    }

    /// Create a configuration builder for fluent configuration
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }
}

/// Configuration builder for fluent configuration creation
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: BitNetConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self { config: BitNetConfig::default() }
    }

    pub fn model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.model.path = Some(path.into());
        self
    }

    pub fn model_format(mut self, format: ModelFormat) -> Self {
        self.config.model.format = format;
        self
    }

    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.model.vocab_size = size;
        self
    }

    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.model.hidden_size = size;
        self
    }

    pub fn num_layers(mut self, layers: usize) -> Self {
        self.config.model.num_layers = layers;
        self
    }

    pub fn num_heads(mut self, heads: usize) -> Self {
        self.config.model.num_heads = heads;
        self
    }

    pub fn num_key_value_heads(mut self, heads: usize) -> Self {
        self.config.model.num_key_value_heads = heads;
        self
    }

    pub fn max_length(mut self, length: usize) -> Self {
        self.config.inference.max_length = length;
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.inference.temperature = temp;
        self
    }

    pub fn top_k(mut self, k: Option<usize>) -> Self {
        self.config.inference.top_k = k;
        self
    }

    pub fn top_p(mut self, p: Option<f32>) -> Self {
        self.config.inference.top_p = p;
        self
    }

    pub fn quantization_type(mut self, qtype: QuantizationType) -> Self {
        self.config.quantization.quantization_type = qtype;
        self
    }

    pub fn use_gpu(mut self, use_gpu: bool) -> Self {
        self.config.performance.use_gpu = use_gpu;
        self
    }

    pub fn num_threads(mut self, threads: Option<usize>) -> Self {
        self.config.performance.num_threads = threads;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.performance.batch_size = size;
        self
    }

    pub fn build(self) -> Result<BitNetConfig, BitNetError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration loading utilities
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration with precedence: env vars > config file > defaults
    pub fn load_with_precedence<P: AsRef<Path>>(
        config_file: Option<P>,
    ) -> Result<BitNetConfig, BitNetError> {
        let mut config = BitNetConfig::default();

        // Apply config file if provided
        if let Some(path) = config_file {
            let file_config = BitNetConfig::from_file(path)?;
            config.merge_with(file_config);
        }

        // Apply environment overrides
        config.apply_env_overrides()?;

        // Final validation
        config.validate()?;

        Ok(config)
    }

    /// Load configuration from multiple sources and merge them
    pub fn load_from_sources(sources: &[ConfigSource]) -> Result<BitNetConfig, BitNetError> {
        let mut config = BitNetConfig::default();

        for source in sources {
            match source {
                ConfigSource::File(path) => {
                    let file_config = BitNetConfig::from_file(path)?;
                    config.merge_with(file_config);
                }
                ConfigSource::Environment => {
                    config.apply_env_overrides()?;
                }
                ConfigSource::Inline(inline_config) => {
                    config.merge_with((**inline_config).clone());
                }
            }
        }

        config.validate()?;
        Ok(config)
    }
}

/// Configuration source types
#[derive(Debug, Clone)]
pub enum ConfigSource {
    File(PathBuf),
    Environment,
    Inline(Box<BitNetConfig>),
}
