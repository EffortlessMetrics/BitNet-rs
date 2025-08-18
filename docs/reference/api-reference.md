# API Reference

This document provides comprehensive API reference for BitNet Rust.

## Core Types

### BitNetModel

The main model interface for loading and running BitNet models.

```rust
pub struct BitNetModel {
    // Internal fields
}

impl BitNetModel {
    /// Load a model from a file path
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BitNetError>;

    /// Load a model from HuggingFace Hub
    pub async fn from_pretrained(model_id: &str) -> Result<Self, BitNetError>;

    /// Load a model with custom configuration
    pub async fn from_pretrained_with_config(
        model_id: &str,
        config: &ModelConfig,
    ) -> Result<Self, BitNetError>;

    /// Generate text from a prompt
    pub async fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String, BitNetError>;

    /// Generate streaming text
    pub fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> impl Stream<Item = Result<String, BitNetError>>;

    /// Get model information
    pub fn model_info(&self) -> &ModelInfo;

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig;
}
```

### ModelConfig

Configuration for model loading and behavior.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Device to run the model on
    pub device: Device,

    /// Data type for model weights
    pub dtype: DType,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Quantization configuration
    pub quantization: QuantizationConfig,

    /// KV cache configuration
    pub kv_cache: KVCacheConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            dtype: DType::F16,
            max_seq_len: 2048,
            vocab_size: 32000,
            num_heads: 32,
            hidden_size: 4096,
            num_layers: 32,
            quantization: QuantizationConfig::default(),
            kv_cache: KVCacheConfig::default(),
        }
    }
}
```

### GenerationConfig

Configuration for text generation.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, higher = more random).
    pub temperature: f32,
    /// Top-k sampling limit (0 = disabled).
    pub top_k: u32,
    /// Top-p (nucleus) sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty, higher = less repetition).
    pub repetition_penalty: f32,
    /// Stop sequences to end generation.
    pub stop_sequences: Vec<String>,
    /// Random seed for reproducible generation.
    pub seed: Option<u64>,
    /// Whether to skip special tokens in output.
    pub skip_special_tokens: bool,
}

impl GenerationConfig {
    /// Creates a greedy generation configuration (deterministic).
    pub fn greedy() -> Self;

    /// Creates a creative generation configuration (high randomness).
    pub fn creative() -> Self;

    /// Creates a balanced generation configuration.
    pub fn balanced() -> Self;

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), String>;

    /// Sets the random seed for reproducible generation.
    pub fn with_seed(mut self, seed: u64) -> Self;

    /// Adds a stop sequence.
    pub fn with_stop_sequence(mut self, stop_seq: String) -> Self;

    /// Sets the maximum number of new tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self;

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self;

    /// Sets the top-k sampling limit.
    pub fn with_top_k(mut self, top_k: u32) -> Self;

    /// Sets the top-p (nucleus) sampling threshold.
    pub fn with_top_p(mut self, top_p: f32) -> Self;
}
```

## Device Management

### Device

Represents compute devices for model execution.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// Automatic device selection
    Auto,
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda(usize), // GPU index
    /// Metal GPU device (macOS)
    Metal,
}

impl Device {
    /// Get available devices
    pub fn available_devices() -> Vec<Device>;

    /// Check if device is available
    pub fn is_available(&self) -> bool;

    /// Get device memory info
    pub fn memory_info(&self) -> Result<DeviceMemoryInfo, BitNetError>;

    /// Get device capabilities
    pub fn capabilities(&self) -> DeviceCapabilities;
}

#[derive(Debug, Clone)]
pub struct DeviceMemoryInfo {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub max_threads: usize,
    pub compute_capability: Option<(u32, u32)>, // For CUDA
}
```

## Quantization

### Quantize Trait

The `Quantize` trait defines the interface for quantization and dequantization operations.

```rust
pub trait Quantize {
    /// Quantize a tensor using the specified quantization type.
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor>;

    /// Dequantize back to a full precision tensor.
    fn dequantize(&self) -> Result<BitNetTensor>;
}
```

### QuantizedTensor

Represents a quantized tensor with compressed data and metadata.

```rust
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Compressed quantized data.
    pub data: Vec<u8>,
    /// Scale factors for dequantization.
    pub scales: Vec<f32>,
    /// Zero points for asymmetric quantization (if needed).
    pub zero_points: Option<Vec<i32>>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Quantization type used.
    pub qtype: QuantizationType,
    /// Block size for grouped quantization.
    pub block_size: usize,
}
```

### QuantizationConfig

Configuration for model quantization.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization type
    pub qtype: QuantizationType,

    /// Block size for quantization
    pub block_size: usize,

    /// Whether to use dynamic quantization
    pub dynamic: bool,

    /// Calibration dataset size
    pub calibration_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (full precision)
    None,
    /// 2-bit signed quantization
    I2S,
    /// Table lookup quantization (ARM optimized)
    TL1,
    /// Table lookup quantization (x86 optimized)
    TL2,
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            qtype: QuantizationType::I2S,
            block_size: 64,
            dynamic: false,
            calibration_size: None,
        }
    }
}
```

## Models

### ModelLoader

The `ModelLoader` is responsible for loading models from disk. It automatically detects the model format and uses the appropriate loader.

```rust
pub struct ModelLoader {
    // ... private fields
}

impl ModelLoader {
    /// Creates a new model loader for the given device.
    pub fn new(device: Device) -> Self;

    /// Loads a model from the given path with default configuration.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn Model>>;

    /// Loads a model from the given path with a custom configuration.
    pub fn load_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        config: &LoadConfig,
    ) -> Result<Box<dyn Model>>;

    /// Returns a list of available format loaders.
    pub fn available_formats(&self) -> Vec<&str>;

    /// Extracts the metadata from a model file without loading the full model.
    pub fn extract_metadata<P: AsRef<Path>>(&self, path: P) -> Result<ModelMetadata>;
}
```

### LoadConfig

Configuration for the model loader.

```rust
#[derive(Clone)]
pub struct LoadConfig {
    /// Whether to use memory-mapped files for loading.
    pub use_mmap: bool,
    /// Whether to validate the checksums of the model files.
    pub validate_checksums: bool,
    /// An optional progress callback for monitoring the loading process.
    pub progress_callback: Option<ProgressCallback>,
}
```

### FormatLoader Trait

The `FormatLoader` trait defines the interface for format-specific model loaders.

```rust
pub trait FormatLoader: Send + Sync {
    /// Returns the name of the format.
    fn name(&self) -> &'static str;

    /// Returns `true` if the loader can load the given file.
    fn can_load(&self, path: &Path) -> bool;

    /// Detects whether the given file is in the correct format.
    fn detect_format(&self, path: &Path) -> Result<bool>;

    /// Extracts the metadata from a model file.
    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata>;

    /// Loads a model from the given file.
    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>>;
}
```

### Model Trait

The `Model` trait defines the interface for all BitNet models. It provides a common set of methods for interacting with different model implementations.

```rust
pub trait Model: Send + Sync {
    /// Returns the configuration of the model.
    fn config(&self) -> &BitNetConfig;

    /// Performs a forward pass through the model.
    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor>;

    /// Embeds a sequence of tokens into a tensor.
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;

    /// Computes the logits for a given hidden state.
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;
}
```

## Inference Engine

### InferenceEngine

The `InferenceEngine` is the main entry point for running inference with a BitNet model. It encapsulates the model, tokenizer, and backend, providing a high-level API for text generation.

```rust
pub struct InferenceEngine {
    // ... private fields
}

impl InferenceEngine {
    /// Creates a new inference engine with the given model, tokenizer, and device.
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self>;

    /// Creates a new inference engine with a custom configuration.
    pub fn with_config(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
        config: InferenceConfig,
    ) -> Result<Self>;

    /// Generates text from a prompt using the default generation configuration.
    pub async fn generate(&self, prompt: &str) -> Result<String>;

    /// Generates text from a prompt with a custom generation configuration.
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String>;

    /// Returns a stream of generated tokens for a given prompt.
    pub fn generate_stream(&self, prompt: &str) -> GenerationStream;

    /// Returns a stream of generated tokens with a custom generation configuration.
    pub fn generate_stream_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> GenerationStream;

    /// Returns the configuration of the underlying model.
    pub fn model_config(&self) -> &BitNetConfig;

    /// Returns statistics about the inference engine, such as cache usage and backend type.
    pub async fn get_stats(&self) -> InferenceStats;

    /// Clears the KV cache of the inference engine.
    pub async fn clear_cache(&self);
}
```

### InferenceResult

Represents the result of an inference operation.

```rust
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The generated text.
    pub generated_text: String,
    /// The number of tokens that were generated.
    pub tokens_generated: usize,
    /// The latency of the generation in milliseconds.
    pub latency_ms: u64,
    /// The number of tokens generated per second.
    pub tokens_per_second: f64,
}
```

### InferenceStats

Statistics about the inference engine.

```rust
#[derive(Debug, Clone)]
pub struct InferenceStats {
    /// The size of the KV cache in bytes.
    pub cache_size: usize,
    /// The usage of the KV cache as a percentage.
    pub cache_usage: f64,
    /// The type of the backend being used (e.g., "cpu", "cuda").
    pub backend_type: String,
}
```

### KVCacheConfig

Configuration for the KV cache.

```rust
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum cache size in bytes.
    pub max_size_bytes: usize,
    /// Maximum sequence length to cache.
    pub max_sequence_length: usize,
    /// Enable cache compression for older entries.
    pub enable_compression: bool,
    /// Eviction policy when cache is full.
    pub eviction_policy: EvictionPolicy,
    /// Block size for memory allocation.
    pub block_size: usize,
}
```

### KVCache

The `KVCache` is a key-value cache used to store intermediate results of the attention mechanism.

```rust
pub struct KVCache {
    // ... private fields
}

impl KVCache {
    /// Creates a new KV cache with the given configuration.
    pub fn new(config: CacheConfig) -> Result<Self>;

    /// Stores a key-value pair in the cache.
    pub fn store(
        &mut self,
        layer: usize,
        position: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()>;

    /// Retrieves a key-value pair from the cache.
    pub fn get(&mut self, layer: usize, position: usize) -> Option<(&Vec<f32>, &Vec<f32>)>;

    /// Returns `true` if the cache contains an entry for the given layer and position.
    pub fn contains(&self, layer: usize, position: usize) -> bool;

    /// Clears all entries from the cache.
    pub fn clear(&mut self);

    /// Clears all entries for a specific layer from the cache.
    pub fn clear_layer(&mut self, layer: usize);

    /// Returns statistics about the cache.
    pub fn stats(&self) -> CacheStats;

    /// Returns the current size of the cache in bytes.
    pub fn size(&self) -> usize;

    /// Returns the usage of the cache as a percentage.
    pub fn usage_percent(&self) -> f64;
}
```

### CacheConfig

Configuration for the KV cache.

```rust
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes.
    pub max_size_bytes: usize,
    /// Maximum sequence length to cache.
    pub max_sequence_length: usize,
    /// Enable cache compression for older entries.
    pub enable_compression: bool,
    /// Eviction policy when cache is full.
    pub eviction_policy: EvictionPolicy,
    /// Block size for memory allocation.
    pub block_size: usize,
}
```

### EvictionPolicy

Cache eviction policies.

```rust
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// First In, First Out
    FIFO,
    /// Least Frequently Used
    LFU,
}
```

### CacheStats

Statistics about the KV cache.

```rust
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub compressed_entries: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_rate: f64,
    pub memory_efficiency: f64,
}
```

### InferenceConfig

Configuration for the inference engine.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum context length to maintain.
    pub max_context_length: usize,
    /// Number of threads for CPU inference.
    pub num_threads: usize,
    /// Batch size for processing.
    pub batch_size: usize,
    /// Enable mixed precision inference.
    pub mixed_precision: bool,
    /// Memory pool size in bytes.
    pub memory_pool_size: usize,
}

impl InferenceConfig {
    /// Creates a configuration optimized for CPU inference.
    pub fn cpu_optimized() -> Self;

    /// Creates a configuration optimized for GPU inference.
    pub fn gpu_optimized() -> Self;

    /// Creates a configuration for memory-constrained environments.
    pub fn memory_efficient() -> Self;

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), String>;

    /// Sets the number of threads for CPU inference.
    pub fn with_threads(mut self, threads: usize) -> Self;

    /// Sets the batch size for processing.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self;

    /// Enables or disables mixed precision inference.
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self;

    /// Sets the memory pool size in bytes.
    pub fn with_memory_pool_size(mut self, size: usize) -> Self;
}
```

## Tokenization

### Tokenizer Trait

The `Tokenizer` trait defines the interface for all tokenizers.

```rust
pub trait Tokenizer: Send + Sync {
    /// Encodes a string into a sequence of token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decodes a sequence of token IDs into a string.
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Returns the size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Returns the ID of the end-of-sentence token, if any.
    fn eos_token_id(&self) -> Option<u32>;

    /// Returns the ID of the padding token, if any.
    fn pad_token_id(&self) -> Option<u32>;
}
```

## Error Handling

### BitNetError

Main error type for BitNet operations.

```rust
#[derive(thiserror::Error, Debug)]
pub enum BitNetError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Tokenization error: {0}")]
    Tokenization(#[from] TokenizerError),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Timeout error: operation timed out after {0:?}")]
    Timeout(Duration),

    #[error("Capacity error: {0}")]
    Capacity(String),
}

pub type Result<T> = std::result::Result<T, BitNetError>;
```

## Utilities

### ModelInfo

Information about a loaded model.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,

    /// Model architecture
    pub architecture: String,

    /// Parameter count
    pub num_parameters: u64,

    /// Model size in bytes
    pub model_size: u64,

    /// Quantization info
    pub quantization: QuantizationInfo,

    /// Supported features
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    pub qtype: QuantizationType,
    pub bits_per_weight: f32,
    pub compression_ratio: f32,
}
```

### Performance Monitoring

```rust
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Tokens generated per second
    pub tokens_per_second: f64,

    /// Time to first token (ms)
    pub time_to_first_token: f64,

    /// Average latency per token (ms)
    pub avg_latency_per_token: f64,

    /// Memory usage (bytes)
    pub memory_usage: usize,

    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: Option<f32>,
}

impl BitNetModel {
    /// Get performance metrics for the last generation
    pub fn last_metrics(&self) -> Option<PerformanceMetrics>;

    /// Reset performance metrics
    pub fn reset_metrics(&self);
}
```

## Feature Flags

BitNet Rust supports conditional compilation with feature flags:

- `gpu`: Enable CUDA GPU support
- `python`: Enable Python bindings
- `wasm`: Enable WebAssembly support
- `cli`: Enable CLI tool
- `serde`: Enable serialization support
- `tracing`: Enable structured logging

Example usage in `Cargo.toml`:

```toml
[dependencies]
bitnet = { version = "0.1.0", features = ["gpu", "serde"] }
```

## Examples

### Basic Text Generation

```rust
use bitnet::{BitNetModel, GenerationConfig};

#[tokio::main]
async fn main() -> bitnet::Result<()> {
    let model = BitNetModel::from_pretrained("microsoft/bitnet-b1_58-large").await?;

    let config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.8,
        ..Default::default()
    };

    let output = model.generate("The future of AI is", &config).await?;
    println!("Generated: {}", output);

    Ok(())
}
```

### Streaming Generation

```rust
use bitnet::{BitNetModel, GenerationConfig};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> bitnet::Result<()> {
    let model = BitNetModel::from_pretrained("microsoft/bitnet-b1_58-large").await?;
    let config = GenerationConfig::default();

    let mut stream = model.generate_stream("Tell me about Rust:", &config);

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => print!("{}", token),
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}
```

### Custom Device Selection

```rust
use bitnet::{BitNetModel, Device, ModelConfig};

#[tokio::main]
async fn main() -> bitnet::Result<()> {
    let config = ModelConfig {
        device: Device::Cuda(0), // Use first GPU
        ..Default::default()
    };

    let model = BitNetModel::from_pretrained_with_config(
        "microsoft/bitnet-b1_58-large",
        &config,
    ).await?;

    // Model will run on GPU 0
    let output = model.generate("Hello", &Default::default()).await?;
    println!("{}", output);

    Ok(())
}
```
