# BitNet.rs API Reference

Complete API documentation for BitNet.rs with examples and usage patterns.

## Table of Contents

- [Core Types](#core-types)
- [Model Loading](#model-loading)
- [Inference Engine](#inference-engine)
- [Configuration](#configuration)
- [Quantization](#quantization)
- [Tokenization](#tokenization)
- [Server API](#server-api)
- [Error Handling](#error-handling)

## Core Types

### Device

Represents the compute device for model execution.

```rust
use bitnet_rs::Device;

// CPU device
let device = Device::Cpu;

// CUDA GPU device
let device = Device::Cuda(0); // GPU index 0

// Auto-select best available device
let device = Device::best_available()?;

// Check device capabilities
if device.supports_quantization(QuantizationType::I2S) {
    println!("Device supports I2S quantization");
}
```

### BitNetConfig

Main configuration structure for BitNet models.

```rust
use bitnet_rs::config::*;

let config = BitNetConfig {
    model: ModelConfig {
        vocab_size: 50257,
        hidden_size: 2048,
        num_layers: 24,
        num_attention_heads: 16,
        max_position_embeddings: 2048,
    },
    inference: InferenceConfig {
        max_new_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
    },
    quantization: QuantizationConfig {
        qtype: QuantizationType::I2S,
        block_size: 64,
    },
};
```

## Model Loading

### BitNetModel

Main model structure for BitNet inference.

```rust
use bitnet_rs::models::*;
use anyhow::Result;

impl BitNetModel {
    /// Load model from GGUF file
    pub async fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self>;
    
    /// Load model from SafeTensors file
    pub async fn from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self>;
    
    /// Load model from HuggingFace checkpoint
    pub async fn from_huggingface<P: AsRef<Path>>(path: P) -> Result<Self>;
    
    /// Auto-detect format and load
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self>;
    
    /// Get model configuration
    pub fn config(&self) -> &BitNetConfig;
    
    /// Get parameter count
    pub fn parameter_count(&self) -> usize;
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize;
}
```

#### Examples

```rust
// Load from different formats
let model = BitNetModel::from_gguf("models/bitnet-1.58b.gguf").await?;
let model = BitNetModel::from_safetensors("models/model.safetensors").await?;
let model = BitNetModel::from_file("models/model.bin").await?; // Auto-detect

// Inspect model
println!("Parameters: {}", model.parameter_count());
println!("Memory usage: {} MB", model.memory_usage() / 1024 / 1024);
println!("Vocab size: {}", model.config().model.vocab_size);
```

### ModelLoader

Advanced model loading with custom options.

```rust
use bitnet_rs::models::ModelLoader;

let loader = ModelLoader::new()
    .device(Device::Cuda(0))
    .dtype(DType::F16)
    .memory_map(true)
    .validate_checksums(true);

let model = loader.load("models/bitnet-1.58b.gguf").await?;
```

## Inference Engine

### InferenceEngine

Main interface for text generation and inference.

```rust
use bitnet_rs::inference::*;

impl InferenceEngine {
    /// Create new inference engine
    pub fn new(
        model: BitNetModel,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self>;
    
    /// Generate text from prompt
    pub async fn generate(&mut self, prompt: &str) -> Result<String>;
    
    /// Generate with custom configuration
    pub async fn generate_with_config(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String>;
    
    /// Generate streaming tokens
    pub fn generate_stream(
        &mut self,
        prompt: &str,
    ) -> impl Stream<Item = Result<String>>;
    
    /// Generate streaming with config
    pub fn generate_stream_with_config(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> impl Stream<Item = Result<String>>;
    
    /// Get model configuration
    pub fn model_config(&self) -> &BitNetConfig;
}
```

#### Basic Usage

```rust
use bitnet_rs::prelude::*;

// Create engine
let model = BitNetModel::from_file("models/bitnet-1.58b.gguf").await?;
let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;
let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

// Simple generation
let response = engine.generate("Hello, world!").await?;
println!("Response: {}", response);
```

#### Advanced Usage

```rust
// Custom generation config
let config = GenerationConfig {
    max_new_tokens: 200,
    temperature: 0.8,
    top_p: 0.95,
    top_k: 40,
    repetition_penalty: 1.1,
    stop_sequences: vec!["<|endoftext|>".to_string()],
    seed: Some(42), // For reproducible generation
};

let response = engine.generate_with_config("The future of AI", &config).await?;
```

#### Streaming Generation

```rust
use futures_util::StreamExt;

let mut stream = engine.generate_stream("Once upon a time");

while let Some(token_result) = stream.next().await {
    match token_result {
        Ok(token) => print!("{}", token),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### GenerationConfig

Configuration for text generation parameters.

```rust
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: u32,
    
    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    
    /// Top-k sampling limit
    pub top_k: u32,
    
    /// Repetition penalty (1.0 = no penalty, higher = less repetition)
    pub repetition_penalty: f32,
    
    /// Stop sequences to end generation
    pub stop_sequences: Vec<String>,
    
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    
    /// Whether to skip special tokens in output
    pub skip_special_tokens: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            seed: None,
            skip_special_tokens: true,
        }
    }
}
```

## Configuration

### Builder Pattern

```rust
use bitnet_rs::config::*;

let config = BitNetConfig::builder()
    .model_config(
        ModelConfig::builder()
            .vocab_size(50257)
            .hidden_size(2048)
            .num_layers(24)
            .build()
    )
    .inference_config(
        InferenceConfig::builder()
            .max_tokens(200)
            .temperature(0.8)
            .build()
    )
    .build();
```

### From File

```rust
// Load from TOML file
let config = BitNetConfig::from_file("bitnet.toml")?;

// Load from JSON file
let config = BitNetConfig::from_json_file("bitnet.json")?;

// Load from environment variables
let config = BitNetConfig::from_env()?;
```

### Environment Variables

```rust
// Supported environment variables
std::env::set_var("BITNET_MODEL_PATH", "models/bitnet-1.58b.gguf");
std::env::set_var("BITNET_DEVICE", "cuda:0");
std::env::set_var("BITNET_MAX_TOKENS", "100");
std::env::set_var("BITNET_TEMPERATURE", "0.7");

let config = BitNetConfig::from_env()?;
```

## Quantization

### QuantizationType

Supported quantization algorithms.

```rust
#[derive(Debug, Clone, Copy)]
pub enum QuantizationType {
    /// 2-bit signed quantization
    I2S,
    /// Table lookup 1 (ARM NEON optimized)
    TL1,
    /// Table lookup 2 (x86 AVX optimized)
    TL2,
}
```

### Quantization Operations

```rust
use bitnet_rs::quantization::*;

// Quantize a tensor
let quantized = tensor.quantize(QuantizationType::I2S)?;
println!("Original size: {} bytes", tensor.size_bytes());
println!("Quantized size: {} bytes", quantized.size_bytes());

// Dequantize back to full precision
let dequantized = quantized.dequantize()?;

// Convert between quantization types
let tl1_quantized = quantized.convert_to(QuantizationType::TL1)?;
```

### Custom Quantization

```rust
// Implement custom quantization
struct CustomQuantizer {
    block_size: usize,
    precision: u8,
}

impl Quantizer for CustomQuantizer {
    fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // Custom quantization logic
        todo!()
    }
    
    fn dequantize(&self, qtensor: &QuantizedTensor) -> Result<Tensor> {
        // Custom dequantization logic
        todo!()
    }
}
```

## Tokenization

### Tokenizer Trait

```rust
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get EOS token ID
    fn eos_token_id(&self) -> Option<u32>;
    
    /// Get PAD token ID
    fn pad_token_id(&self) -> Option<u32>;
}
```

### TokenizerBuilder

```rust
use bitnet_rs::tokenizers::*;

// Load from pretrained
let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;

// Load from file
let tokenizer = TokenizerBuilder::from_file("tokenizer.json")?;

// Create custom tokenizer
let tokenizer = TokenizerBuilder::new()
    .vocab_file("vocab.txt")
    .merges_file("merges.txt")
    .special_tokens(&[("<|endoftext|>", 50256)])
    .build()?;
```

### Usage Examples

```rust
let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;

// Encode text
let tokens = tokenizer.encode("Hello, world!", true)?;
println!("Tokens: {:?}", tokens);

// Decode tokens
let text = tokenizer.decode(&tokens, true)?;
println!("Decoded: {}", text);

// Batch operations
let texts = vec!["Hello", "World", "!"];
let token_batches: Vec<Vec<u32>> = texts
    .iter()
    .map(|text| tokenizer.encode(text, false))
    .collect::<Result<Vec<_>>>()?;
```

## Server API

### HTTP Endpoints

#### POST /api/generate

Generate text from a prompt.

**Request:**
```json
{
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "stream": false
}
```

**Response:**
```json
{
    "generated_text": "Hello, world! How are you doing today?",
    "tokens_generated": 8,
    "latency_ms": 150,
    "model_info": {
        "name": "BitNet-1.58B",
        "quantization": "I2S",
        "device": "Cpu"
    }
}
```

#### POST /api/generate/stream

Stream generated tokens in real-time.

**Request:** Same as `/api/generate`

**Response:** Server-Sent Events stream
```
data: {"token": "Hello", "done": false}
data: {"token": ",", "done": false}
data: {"token": " world", "done": false}
data: {"done": true, "total_tokens": 3}
```

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "Cpu",
    "uptime_seconds": 3600
}
```

### Server Configuration

```rust
use bitnet_rs::server::*;

let server = BitNetServer::builder()
    .model_path("models/bitnet-1.58b.gguf")
    .bind("0.0.0.0:3000")
    .workers(4)
    .max_request_size(1024 * 1024) // 1MB
    .timeout(Duration::from_secs(30))
    .cors(true)
    .auth_tokens(vec!["secret-token".to_string()])
    .build()
    .await?;

server.run().await?;
```

## Error Handling

### Error Types

```rust
#[derive(Error, Debug)]
pub enum BitNetError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),
    
    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),
    
    #[error("Tokenization error: {0}")]
    Tokenization(#[from] TokenizationError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Config(String),
}
```

### Error Handling Patterns

```rust
use bitnet_rs::prelude::*;

// Basic error handling
match engine.generate("Hello").await {
    Ok(response) => println!("Generated: {}", response),
    Err(BitNetError::Model(e)) => eprintln!("Model error: {}", e),
    Err(BitNetError::Inference(e)) => eprintln!("Inference error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}

// Using anyhow for error propagation
use anyhow::Result;

async fn generate_text() -> Result<String> {
    let model = BitNetModel::from_file("model.gguf").await?;
    let tokenizer = TokenizerBuilder::from_pretrained("gpt2")?;
    let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;
    
    let response = engine.generate("Hello, world!").await?;
    Ok(response)
}
```

### Custom Error Context

```rust
use anyhow::{Context, Result};

let model = BitNetModel::from_file("model.gguf")
    .await
    .context("Failed to load model from file")?;

let response = engine
    .generate("Hello")
    .await
    .context("Failed to generate text")?;
```

## Advanced Usage

### Custom Kernels

```rust
use bitnet_rs::kernels::*;

struct CustomKernel;

impl KernelProvider for CustomKernel {
    fn name(&self) -> &'static str {
        "custom"
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Custom matrix multiplication implementation
        todo!()
    }
}

// Register custom kernel
let mut kernel_manager = KernelManager::new();
kernel_manager.register(Box::new(CustomKernel));
```

### Memory Management

```rust
// Monitor memory usage
let memory_usage = engine.memory_usage();
println!("Model memory: {} MB", memory_usage.model_mb);
println!("Cache memory: {} MB", memory_usage.cache_mb);
println!("Total memory: {} MB", memory_usage.total_mb);

// Configure memory limits
let config = InferenceConfig::builder()
    .max_cache_size(1024 * 1024 * 1024) // 1GB cache limit
    .memory_pool_size(512 * 1024 * 1024) // 512MB pool
    .build();
```

### Performance Monitoring

```rust
use bitnet_rs::metrics::*;

// Enable metrics collection
let metrics = MetricsCollector::new();
let mut engine = InferenceEngine::with_metrics(model, tokenizer, device, metrics)?;

// Generate with timing
let start = std::time::Instant::now();
let response = engine.generate("Hello").await?;
let duration = start.elapsed();

println!("Generation took: {:?}", duration);
println!("Tokens per second: {:.2}", response.len() as f64 / duration.as_secs_f64());
```