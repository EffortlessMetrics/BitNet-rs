# BitNet.rs API Reference

## Version: 1.0.0

This document defines the stable public API surface of BitNet.rs. All items listed here are covered by our semantic versioning guarantee.

## Table of Contents

- [Core Types](#core-types)
- [Model API](#model-api)
- [Inference API](#inference-api)
- [Quantization API](#quantization-api)
- [Tokenization API](#tokenization-api)
- [FFI API](#ffi-api)
- [Breaking Change Policy](#breaking-change-policy)

## Core Types

### `Config`

The main configuration struct for BitNet models.

```rust
pub struct Config {
    pub model_type: ModelType,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub quantization: QuantizationType,
}
```

### `ModelType`

Supported model architectures.

```rust
pub enum ModelType {
    BitNetB158,
    BitNet3B,
    Custom(String),
}
```

### `QuantizationType`

Quantization methods available.

```rust
pub enum QuantizationType {
    I2S,     // INT2 symmetric quantization
    TL1,     // Ternary with learned thresholds v1
    TL2,     // Ternary with learned thresholds v2
}
```

## Model API

### Loading Models

```rust
/// Load a model from a file path
pub fn load_model(path: &Path) -> Result<Model, Error>

/// Load a model with custom configuration
pub fn load_model_with_config(path: &Path, config: Config) -> Result<Model, Error>
```

### Model Methods

```rust
impl Model {
    /// Create a new model with the given configuration
    pub fn new(config: Config) -> Result<Self, Error>
    
    /// Run forward pass on input tensors
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, Error>
    
    /// Generate text from a prompt
    pub fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String, Error>
    
    /// Stream generation tokens
    pub fn generate_stream(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> impl Stream<Item = Result<Token, Error>>
    
    /// Get model configuration
    pub fn config(&self) -> &Config
    
    /// Get model device (CPU/CUDA)
    pub fn device(&self) -> Device
}
```

### Generation Parameters

```rust
pub struct GenerationParams {
    pub max_tokens: usize,           // Maximum tokens to generate
    pub temperature: f32,             // Sampling temperature (0.0-2.0)
    pub top_p: f32,                  // Nucleus sampling threshold
    pub top_k: usize,                // Top-k sampling
    pub repetition_penalty: f32,     // Penalty for repeated tokens
    pub seed: Option<u64>,           // Random seed for reproducibility
    pub stop_sequences: Vec<String>, // Stop generation on these sequences
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            seed: None,
            stop_sequences: vec![],
        }
    }
}
```

## Inference API

### InferenceEngine

High-level inference engine with batching and optimization.

```rust
pub struct InferenceEngine {
    // Private fields
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(model: Model) -> Result<Self, Error>

    /// Process a single request
    pub async fn infer(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, Error>
    
    /// Process a batch of requests
    pub async fn infer_batch(
        &self,
        requests: Vec<InferenceRequest>,
        ) -> Result<Vec<InferenceResponse>, Error>

    /// Evaluate a token prefix and return logits for the next token
    pub async fn logits(&self, ids: &[u32]) -> Result<Vec<f32>>

    /// Get engine metrics
    pub fn metrics(&self) -> EngineMetrics
}
```

### Request/Response Types

```rust
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub params: GenerationParams,
}

pub struct InferenceResponse {
    pub id: String,
    pub text: String,
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub finish_reason: FinishReason,
}

pub enum FinishReason {
    MaxTokens,
    StopSequence,
    EndOfText,
    Error(String),
}
```

## Quantization API

### Quantization Functions

```rust
/// Quantize a floating-point tensor to 1-bit
pub fn quantize(tensor: &Tensor, method: QuantizationType) -> Result<QuantizedTensor, Error>

/// Dequantize a 1-bit tensor back to floating-point
pub fn dequantize(tensor: &QuantizedTensor) -> Result<Tensor, Error>

/// Quantize model weights in-place
pub fn quantize_model(model: &mut Model, method: QuantizationType) -> Result<(), Error>
```

### QuantizedTensor

```rust
pub struct QuantizedTensor {
    // Private fields
}

impl QuantizedTensor {
    pub fn shape(&self) -> &[usize]
    pub fn dtype(&self) -> DataType
    pub fn device(&self) -> Device
    pub fn quantization_type(&self) -> QuantizationType
}
```

## Tokenization API

### Tokenizer

```rust
pub struct Tokenizer {
    // Private fields
}

impl Tokenizer {
    /// Load tokenizer from file
    pub fn from_file(path: &Path) -> Result<Self, Error>
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error>
    
    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Error>
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize
    
    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens
}
```

## FFI API

C-compatible API for cross-language integration.

### Core Functions

```c
// Model management
bitnet_model* bitnet_load_model(const char* path);
void bitnet_free_model(bitnet_model* model);

// Inference
bitnet_result* bitnet_generate(
    bitnet_model* model,
    const char* prompt,
    bitnet_params* params
);
void bitnet_free_result(bitnet_result* result);

// Configuration
bitnet_config* bitnet_get_config(bitnet_model* model);
void bitnet_free_config(bitnet_config* config);
```

### Error Handling

```c
// Get last error message
const char* bitnet_get_last_error();

// Clear error state
void bitnet_clear_error();
```

## Breaking Change Policy

### Versioning

BitNet.rs follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### What Constitutes a Breaking Change

The following changes require a MAJOR version bump:

1. **Removing or renaming public items** (functions, types, modules)
2. **Changing function signatures** (parameters, return types)
3. **Changing struct fields** (removing, renaming, changing types)
4. **Changing trait requirements** (adding required methods)
5. **Changing default behavior** in incompatible ways

### What Does NOT Constitute a Breaking Change

The following are NOT considered breaking:

1. **Adding new public items** (functions, types, modules)
2. **Adding optional parameters** with defaults
3. **Adding trait implementations** to existing types
4. **Performance improvements** that don't change behavior
5. **Bug fixes** that restore documented behavior
6. **Internal implementation changes** that don't affect the API

### Deprecation Process

Before removing an API:

1. Mark as `#[deprecated]` with migration guide
2. Keep deprecated API for at least one MINOR version
3. Remove in next MAJOR version

Example:
```rust
#[deprecated(since = "1.2.0", note = "Use `new_function` instead")]
pub fn old_function() { /* ... */ }
```

### Experimental APIs

APIs marked as experimental are not covered by semver:

```rust
#[cfg(feature = "experimental")]
pub mod experimental {
    // These APIs may change without notice
}
```

## Migration Guides

### Migrating from 0.x to 1.0

See [MIGRATION.md](./MIGRATION.md) for detailed migration instructions.

## API Stability Guarantees

The following APIs are guaranteed stable:

- ✅ Core model loading and inference
- ✅ Quantization functions
- ✅ Tokenization interface
- ✅ FFI C API
- ✅ Configuration structures

The following are subject to change:

- ⚠️ Internal optimization details
- ⚠️ Experimental features (requires `experimental` feature flag)
- ⚠️ Benchmark utilities

## Contact

For API questions or breaking change discussions:
- GitHub Issues: [BitNet-rs/issues](https://github.com/microsoft/BitNet/issues)
- Discussions: [BitNet-rs/discussions](https://github.com/microsoft/BitNet/discussions)