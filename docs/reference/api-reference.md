# API Reference

This document provides comprehensive API reference for BitNet Rust library and production inference server.

## Production Inference Server API

The BitNet.rs production inference server provides enterprise-grade HTTP API endpoints for neural network inference with comprehensive model management, monitoring, and security features.

### Base URL and Versioning

```
Base URL: http://localhost:8080
API Version: v1
```

All v1 endpoints are prefixed with `/v1/` for version management and backward compatibility.

### Authentication

JWT authentication is optional and configurable. When enabled, include the JWT token in requests:

```bash
curl -H "Authorization: Bearer <jwt-token>" http://localhost:8080/v1/inference
```

## Inference Endpoints

### POST /v1/inference

Synchronous inference with comprehensive configuration options and performance metrics.

**Request Body**:
```json
{
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "model": "optional_model_id",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.0,
  "stop_sequences": ["string"],
  "seed": 42,
  "priority": "normal",
  "device_preference": "auto",
  "quantization_hint": "i2s",
  "timeout_ms": 30000
}
```

**Request Parameters**:
- `prompt` (string, required): Input text for generation
- `max_tokens` (integer, optional, default: 64): Maximum tokens to generate
- `model` (string, optional): Specific model ID to use for inference
- `temperature` (float, optional, default: 1.0): Sampling temperature (0.0-2.0)
- `top_p` (float, optional, default: 0.9): Nucleus sampling probability
- `top_k` (integer, optional, default: 50): Top-k sampling limit
- `repetition_penalty` (float, optional, default: 1.0): Repetition penalty factor
- `stop_sequences` (array, optional): Sequences to stop generation
- `seed` (integer, optional): Random seed for deterministic generation
- `priority` (string, optional): Request priority ("low", "normal", "high", "critical")
- `device_preference` (string, optional): Device selection ("auto", "cpu", "gpu", "cuda:N")
- `quantization_hint` (string, optional): Preferred quantization ("auto", "i2s", "tl1", "tl2")
- `timeout_ms` (integer, optional, default: 30000): Request timeout in milliseconds

**Response**:
```json
{
  "text": "The future of AI is bright with advances in neural network quantization enabling efficient deployment at scale.",
  "tokens_generated": 17,
  "inference_time_ms": 890,
  "tokens_per_second": 19.1,
  "device_used": "Cpu",
  "quantization_type": "i2s",
  "batch_id": "batch_uuid_123",
  "batch_size": 1,
  "queue_time_ms": 5
}
```

**Status Codes**:
- `200 OK`: Successful inference
- `400 Bad Request`: Invalid request parameters or validation failure
- `429 Too Many Requests`: Rate limit exceeded or server overloaded
- `500 Internal Server Error`: Server error during inference

### POST /v1/inference/stream

Server-Sent Events (SSE) streaming inference for real-time token generation.

**Request**: Same as `/v1/inference` endpoint

**Response Headers**:
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Streaming Response Format**:
```
data: {"type": "token", "text": "The", "position": 0}

data: {"type": "token", "text": " future", "position": 1}

data: {"type": "metrics", "tokens_per_second": 25.3, "device": "Cpu"}

data: {"type": "complete", "total_tokens": 17, "inference_time_ms": 1780, "final_text": "The future of AI is..."}
```

**Event Types**:
- `token`: Individual token generation with text and position
- `metrics`: Real-time performance metrics during generation
- `complete`: Final completion event with summary statistics
- `error`: Error information if generation fails

### POST /inference (Legacy)

Legacy inference endpoint for backward compatibility. Provides simplified interface without enhanced features.

**Request/Response**: Subset of `/v1/inference` without enhanced metadata.

## Model Management Endpoints

### POST /v1/models/load

Load a new model with comprehensive validation and configuration.

**Request Body**:
```json
{
  "model_path": "/path/to/model.gguf",
  "tokenizer_path": "/path/to/tokenizer.json",
  "device": "auto",
  "model_id": "custom_model_name",
  "validation_config": {
    "enable_cross_validation": true,
    "min_accuracy": 0.99,
    "validation_samples": 100
  }
}
```

**Response**:
```json
{
  "model_id": "model_12345",
  "status": "success",
  "message": "Model loaded and activated successfully"
}
```

### GET /v1/models

List all loaded models with metadata and performance information.

**Response**:
```json
{
  "models": [
    {
      "id": "model_12345",
      "path": "/app/models/bitnet-2b.gguf",
      "status": "active",
      "quantization_format": "i2s",
      "device": "Cpu",
      "load_time": "2023-12-01T10:30:00Z",
      "performance_metrics": {
        "avg_tokens_per_second": 28.4,
        "avg_inference_time_ms": 890,
        "accuracy_score": 0.995,
        "total_requests": 1524
      },
      "model_info": {
        "parameter_count": 2000000000,
        "tensor_count": 248,
        "model_size_bytes": 1200000000
      }
    }
  ]
}
```

### GET /v1/models/{model_id}

Get detailed information about a specific model.

**Response**: Single model object from the list above.

### DELETE /v1/models/{model_id}

Unload a specific model from memory.

**Status Codes**:
- `204 No Content`: Model successfully unloaded
- `404 Not Found`: Model ID not found
- `500 Internal Server Error`: Error during unload

### POST /v1/models/swap

Atomic model hot-swapping with rollback capabilities for zero-downtime updates.

**Request Body**:
```json
{
  "new_model_path": "/path/to/new_model.gguf",
  "target_model_id": "existing_model",
  "swap_strategy": "atomic",
  "rollback_on_failure": true,
  "validation_timeout_seconds": 30
}
```

**Response**:
```json
{
  "swap_id": "swap_uuid_456",
  "status": "success",
  "new_model_id": "model_67890",
  "previous_model_id": "model_12345",
  "swap_time_ms": 245
}
```

## Health and Monitoring Endpoints

### GET /health

Comprehensive health check with component status and system metrics.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:30:00Z",
  "components": {
    "model_manager": "healthy",
    "execution_router": "healthy",
    "batch_engine": "healthy",
    "device_monitor": "healthy",
    "concurrency_manager": "healthy"
  },
  "system_metrics": {
    "cpu_utilization": 0.65,
    "gpu_utilization": 0.78,
    "memory_usage_bytes": 6442450944,
    "active_requests": 23,
    "uptime_seconds": 86400
  },
  "build_info": {
    "version": "1.0.0",
    "build_date": "2023-12-01T08:00:00Z",
    "git_commit": "abc1234",
    "features": ["cpu", "gpu", "prometheus"]
  }
}
```

**Status Values**:
- `healthy`: All components functioning normally
- `degraded`: Some components have issues but service available
- `unhealthy`: Critical components failed, service unavailable

### GET /health/live

Kubernetes liveness probe endpoint for container restart decisions.

**Response**:
```json
{
  "status": "live",
  "timestamp": "2023-12-01T10:30:00Z"
}
```

**Status Codes**:
- `200 OK`: Service is alive and responding
- `503 Service Unavailable`: Service is unresponsive and should be restarted

### GET /health/ready

Kubernetes readiness probe endpoint for traffic routing decisions.

**Response**:
```json
{
  "status": "ready",
  "timestamp": "2023-12-01T10:30:00Z",
  "checks": {
    "models_loaded": true,
    "devices_available": true,
    "request_processing": true
  }
}
```

**Status Codes**:
- `200 OK`: Service is ready to accept traffic
- `503 Service Unavailable`: Service not ready for traffic

## Statistics and Performance Endpoints

### GET /v1/stats

Comprehensive server statistics with performance metrics and usage information.

**Response**:
```json
{
  "server_stats": {
    "uptime_seconds": 86400,
    "total_requests": 125430,
    "successful_requests": 124987,
    "error_rate": 0.0035,
    "avg_response_time_ms": 1245
  },
  "inference_stats": {
    "total_tokens_generated": 2847593,
    "avg_tokens_per_second": 28.4,
    "quantization_distribution": {
      "i2s": 0.65,
      "tl1": 0.25,
      "tl2": 0.10
    },
    "device_distribution": {
      "cpu": 0.40,
      "gpu": 0.60
    }
  },
  "batch_engine_stats": {
    "total_batches": 15234,
    "avg_batch_size": 3.2,
    "batch_utilization": 0.78,
    "queue_depth": 12
  },
  "concurrency_stats": {
    "active_requests": 23,
    "max_concurrent_requests": 100,
    "rate_limited_requests": 45,
    "avg_queue_time_ms": 15
  }
}
```

### GET /v1/devices

Device status and utilization information for compute resources.

**Response**:
```json
{
  "devices": [
    {
      "id": "cpu",
      "type": "Cpu",
      "status": "healthy",
      "utilization": 0.65,
      "memory_usage_bytes": 4294967296,
      "memory_total_bytes": 8589934592,
      "active_requests": 15,
      "capabilities": {
        "simd_support": ["avx2", "avx512"],
        "quantization_support": ["i2s", "tl1", "tl2"]
      }
    },
    {
      "id": "cuda:0",
      "type": "Cuda",
      "device_index": 0,
      "status": "healthy",
      "utilization": 0.78,
      "memory_usage_bytes": 6442450944,
      "memory_total_bytes": 10737418240,
      "active_requests": 8,
      "capabilities": {
        "compute_capability": "8.6",
        "mixed_precision": ["fp16", "bf16"],
        "quantization_support": ["i2s", "tl1", "tl2"]
      },
      "gpu_info": {
        "name": "NVIDIA GeForce RTX 3080",
        "driver_version": "525.60.11",
        "cuda_version": "12.0"
      }
    }
  ]
}
```

## Prometheus Metrics Endpoint

### GET /metrics

Prometheus-compatible metrics export for monitoring and alerting.

**Response Format**: Prometheus text exposition format

**Key Metrics**:
```
# HELP bitnet_inference_duration_seconds Time spent processing inference requests
# TYPE bitnet_inference_duration_seconds histogram
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="0.5"} 245
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="1.0"} 892
bitnet_inference_duration_seconds_bucket{quantization_type="i2s",device="cpu",le="2.0"} 1456

# HELP bitnet_tokens_per_second Current token generation rate
# TYPE bitnet_tokens_per_second gauge
bitnet_tokens_per_second{device="cpu",quantization_type="i2s"} 28.4

# HELP bitnet_active_requests Current number of active requests
# TYPE bitnet_active_requests gauge
bitnet_active_requests 23

# HELP bitnet_quantization_accuracy_ratio Quantization accuracy vs reference
# TYPE bitnet_quantization_accuracy_ratio gauge
bitnet_quantization_accuracy_ratio{quantization_type="i2s"} 0.995

# HELP bitnet_gpu_utilization_ratio GPU utilization percentage
# TYPE bitnet_gpu_utilization_ratio gauge
bitnet_gpu_utilization_ratio{device="cuda:0"} 0.78

# HELP bitnet_model_load_duration_seconds Time to load models
# TYPE bitnet_model_load_duration_seconds histogram
bitnet_model_load_duration_seconds_sum{model_size="2b"} 125.6
bitnet_model_load_duration_seconds_count{model_size="2b"} 15
```

## Error Handling and Status Codes

### Standardized Error Response

All API endpoints return standardized error responses:

```json
{
  "error": "Detailed error message",
  "error_code": "ERROR_CODE",
  "request_id": "req_uuid_789",
  "details": {
    "field": "Additional context",
    "suggestion": "How to fix the issue"
  }
}
```

### Common Error Codes

- `PROMPT_TOO_LONG`: Input prompt exceeds maximum length
- `TOO_MANY_TOKENS`: Token count exceeds limit
- `INVALID_CHARACTERS`: Prompt contains invalid characters
- `BLOCKED_CONTENT`: Content blocked by security filters
- `MODEL_NOT_FOUND`: Requested model not available
- `DEVICE_UNAVAILABLE`: Requested device not available
- `QUANTIZATION_NOT_SUPPORTED`: Unsupported quantization format
- `RATE_LIMIT_EXCEEDED`: Request rate limit exceeded
- `SERVER_OVERLOADED`: Server at capacity
- `VALIDATION_FAILED`: Request validation failed

### HTTP Status Code Reference

- `200 OK`: Successful request
- `400 Bad Request`: Client error in request format or content
- `401 Unauthorized`: Authentication required or failed
- `403 Forbidden`: Request not allowed for authenticated user
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: Request body exceeds size limits
- `429 Too Many Requests`: Rate limiting applied
- `500 Internal Server Error`: Unexpected server error
- `503 Service Unavailable`: Server temporarily unavailable

## Real Neural Network Inference (Issue #254)

BitNet.rs implements production-quality neural network inference with real quantized computation, replacing all mock implementations.

### Inference Modes

#### Real Inference (Production)

**compute_path="real"** - Uses actual quantized GEMV operations with trained model weights:

- **Quantized Linear Layers**: I2_S/TL1/TL2 GEMV without FP32 staging in hot path
- **Real Attention**: Q/K/V/O projections with RoPE positional embeddings, GQA, and causal masking
- **Autoregressive Generation**: Deterministic token generation with seeded sampling
- **KV-Cache**: Efficient prefill + decode with cache parity validation
- **Receipt Artifacts**: Generates `ci/inference.json` with verifiable performance metrics

#### Receipt Validation API

Validate inference receipts to ensure real computation paths:

```rust
use std::path::Path;
use serde_json::Value;

/// Validate inference receipt artifact
pub fn validate_inference_receipt(receipt_path: &Path) -> Result<(), String> {
    let receipt: Value = serde_json::from_str(&std::fs::read_to_string(receipt_path)?)?;

    // AC9: Strict validation rules
    let compute_path = receipt["compute_path"].as_str()
        .ok_or("Missing compute_path")?;

    if compute_path != "real" {
        return Err(format!("Invalid compute_path: '{}' (expected 'real')", compute_path));
    }

    let kernels = receipt["kernels"].as_array()
        .ok_or("Missing kernels array")?;

    for kernel in kernels {
        let kernel_name = kernel.as_str().ok_or("Invalid kernel name")?;
        if kernel_name.to_lowercase().contains("mock") {
            return Err(format!("Mock kernel detected: '{}'", kernel_name));
        }
    }

    Ok(())
}

// Example usage in CI/CD
assert!(validate_inference_receipt(Path::new("ci/inference.json")).is_ok());
```

### Deterministic Inference

Enable reproducible generation for testing and validation:

```rust
use bitnet::{BitNetModel, GenerationConfig};

// Set environment variables for determinism
std::env::set_var("BITNET_DETERMINISTIC", "1");
std::env::set_var("BITNET_SEED", "42");
std::env::set_var("RAYON_NUM_THREADS", "1");

let config = GenerationConfig {
    seed: Some(42),
    max_new_tokens: 50,
    ..Default::default()
};

let model = BitNetModel::from_file("model.gguf").await?;

// Two runs produce identical token sequences (AC6)
let tokens1 = model.generate("Test", &config).await?.token_ids;
let tokens2 = model.generate("Test", &config).await?.token_ids;

assert_eq!(tokens1, tokens2); // Deterministic generation
```

### Quantization Accuracy Guarantees

All quantization kernels meet strict accuracy envelopes (AC5):

| Quantization | Accuracy (MSE vs FP32) | Use Case |
|-------------|----------------------|----------|
| **I2_S** | ≤ 1e-5 | Production (default) |
| **TL1** | ≤ 1e-4 | ARM NEON optimized |
| **TL2** | ≤ 1e-4 | x86 AVX optimized |

Validate accuracy with built-in tests:

```bash
# I2S accuracy validation
cargo test --no-default-features --features cpu test_i2s_kernel_accuracy_envelope

# TL1 accuracy validation
cargo test --no-default-features --features cpu test_tl1_kernel_accuracy_envelope

# TL2 accuracy validation
cargo test --no-default-features --features cpu test_tl2_kernel_accuracy_envelope
```

### Performance Metrics with Receipts

All performance claims must be backed by receipt artifacts:

```rust
use bitnet::{BitNetModel, GenerationConfig};

let model = BitNetModel::from_file("model.gguf").await?;
let config = GenerationConfig::default();

let response = model.generate("Prompt", &config).await?;

// Access validated performance metrics
if let Some(metrics) = response.metrics {
    println!("Tokens/sec: {}", metrics.tokens_per_second);
    println!("Latency: {}ms", metrics.total_latency_ms);
    println!("Backend: {}", metrics.backend_type);
}

// Receipt artifact automatically generated at ci/inference.json
// Verify with: cat ci/inference.json | jq '.compute_path'
```

## Core Types

### BitNetModel

The main model interface for loading and running BitNet models with production-ready GGUF weight loading and real neural network inference.

```rust
pub struct BitNetModel {
    // Internal fields for real neural network weights and configuration
}

impl BitNetModel {
    /// Load a model from a local GGUF file with real trained weights
    ///
    /// This method replaces mock tensor initialization with comprehensive GGUF parsing:
    /// - Parses all transformer layer weights (attention, feed-forward, normalization)
    /// - Supports I2_S, TL1, TL2 quantization formats with ≥99% accuracy vs FP32
    /// - Device-aware tensor placement with GPU/CPU support
    /// - Enhanced security validation and bounds checking
    /// - Memory-efficient zero-copy operations where possible
    ///
    /// # Arguments
    /// * `path` - Path to GGUF model file
    ///
    /// # Returns
    /// * `Result<Self, BitNetError>` - Loaded model with real weights or error
    ///
    /// # Errors
    /// Returns `BitNetError::GgufParse` for:
    /// - Invalid GGUF magic bytes or version
    /// - Missing required transformer tensors
    /// - Tensor shape validation failures
    /// - Unsupported quantization formats
    /// - Security limit violations
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BitNetError>;

    /// Load a model from HuggingFace Hub with automatic GGUF detection
    pub async fn from_pretrained(model_id: &str) -> Result<Self, BitNetError>;

    /// Load a model with custom configuration and device placement
    pub async fn from_pretrained_with_config(
        model_id: &str,
        config: &ModelConfig,
    ) -> Result<Self, BitNetError>;

    /// Generate text using real neural network inference (Issue #254)
    ///
    /// Performs production-quality generation with real quantized computation:
    /// - **Real GGUF Weights**: Loaded from trained transformer models
    /// - **Quantized GEMV**: I2_S/TL1/TL2 kernels without FP32 staging (AC1)
    /// - **Real Attention**: Q/K/V/O projections with RoPE + GQA + causal mask (AC2)
    /// - **Deterministic**: Seeded generation with BITNET_DETERMINISTIC=1 (AC3)
    /// - **Receipt Generation**: Creates ci/inference.json with compute_path="real" (AC4)
    /// - **Accuracy Validated**: I2S ≤ 1e-5 MSE, TL1/TL2 ≤ 1e-4 MSE vs FP32 (AC5)
    ///
    /// # Example
    /// ```rust,no_run
    /// use bitnet::{BitNetModel, GenerationConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Load real GGUF model with I2S quantization
    /// let model = BitNetModel::from_file("model.gguf").await?;
    ///
    /// // Deterministic generation configuration
    /// std::env::set_var("BITNET_DETERMINISTIC", "1");
    /// std::env::set_var("BITNET_SEED", "42");
    /// std::env::set_var("RAYON_NUM_THREADS", "1");
    ///
    /// let config = GenerationConfig {
    ///     max_new_tokens: 50,
    ///     temperature: 0.8,
    ///     seed: Some(42),
    ///     ..Default::default()
    /// };
    ///
    /// // Real neural network inference (no mock)
    /// let response = model.generate("The future of AI is", &config).await?;
    ///
    /// // Verify real inference path
    /// assert_eq!(response.metadata.quantization, QuantizationType::I2S);
    /// assert!(response.metrics.is_some());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Returns
    /// * `GenerationResponse` with generated text, token IDs, and performance metrics
    /// * Receipt artifact generated at `ci/inference.json` with compute_path="real"
    pub async fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResponse, BitNetError>;

    /// Generate streaming text with real-time neural network inference
    pub fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> impl Stream<Item = Result<StreamResponse, BitNetError>>;

    /// Prefill the model cache with given tokens for performance optimization
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<(), BitNetError>;

    /// Get model information including real tensor count and parameters
    pub fn model_info(&self) -> &ModelInfo;

    /// Get model configuration extracted from GGUF metadata
    pub fn config(&self) -> &ModelConfig;

    /// Get actual tensor count loaded from GGUF (not mock count)
    pub fn tensor_count(&self) -> usize;

    /// Get total parameter count from real model weights
    pub fn parameter_count(&self) -> u64;

    /// Validate model weights against expected transformer architecture
    pub fn validate_weights(&self) -> Result<ValidationReport, BitNetError>;
}

/// Response from text generation with performance metrics
#[derive(Debug, Clone)]
pub struct GenerationResponse {
    /// Generated text
    pub text: String,
    /// Token IDs generated
    pub token_ids: Vec<u32>,
    /// Performance metrics (if enabled)
    pub metrics: Option<PerformanceMetrics>,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Streaming generation response with token-level access
#[derive(Debug, Clone)]
pub struct StreamResponse {
    /// Generated text for this token
    pub text: String,
    /// Token IDs for this generation step
    pub token_ids: Vec<u32>,
    /// Whether this is the final token
    pub is_final: bool,
    /// Incremental timing metrics
    pub timing: Option<TokenTiming>,
}

/// Generation metadata for analysis and debugging
#[derive(Debug, Clone)]
pub struct GenerationMetadata {
    /// Model used for generation
    pub model_name: String,
    /// Quantization format applied
    pub quantization: QuantizationType,
    /// Device used for inference
    pub device: Device,
    /// Total tokens in prompt
    pub prompt_tokens: usize,
    /// Completion tokens generated
    pub completion_tokens: usize,
}

/// Per-token timing information for streaming
#[derive(Debug, Clone)]
pub struct TokenTiming {
    /// Time to generate this token (ms)
    pub token_time_ms: f64,
    /// Cumulative generation time (ms)
    pub cumulative_time_ms: f64,
    /// Current throughput (tokens/sec)
    pub current_throughput: f64,
}

/// Comprehensive model weight validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Total tensors validated
    pub tensor_count: usize,
    /// Missing required tensors
    pub missing_tensors: Vec<String>,
    /// Tensors with invalid shapes
    pub invalid_shapes: Vec<(String, Vec<usize>, Vec<usize>)>, // (name, expected, actual)
    /// Quantization accuracy vs FP32 baseline
    pub quantization_accuracy: f64,
    /// Memory usage breakdown
    pub memory_breakdown: MemoryBreakdown,
    /// Validation passed all checks
    pub is_valid: bool,
}

/// Memory usage breakdown for model analysis
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Attention layer memory (bytes)
    pub attention_memory: u64,
    /// Feed-forward layer memory (bytes)
    pub feedforward_memory: u64,
    /// Normalization layer memory (bytes)
    pub normalization_memory: u64,
    /// Embedding memory (bytes)
    pub embedding_memory: u64,
    /// Total model memory (bytes)
    pub total_memory: u64,
}
```

### Performance Metrics

BitNet.rs provides structured performance metrics for comprehensive monitoring:

#### TimingMetrics

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Tokenization time in milliseconds
    pub tokenize: f64,
    /// Prefill cache warming time in milliseconds
    pub prefill: f64,
    /// Token decoding time in milliseconds
    pub decode: f64,
    /// Total inference time in milliseconds
    pub total: f64,
}
```

#### ThroughputMetrics

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Prefill throughput (tokens per second)
    pub prefill: f64,
    /// Decode throughput (tokens per second)
    pub decode: f64,
    /// End-to-end throughput (tokens per second)
    pub e2e: f64,
}
```

#### TokenizerInfo

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerInfo {
    /// Tokenizer source description
    pub source: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Beginning of sequence token ID
    pub bos_id: Option<u32>,
    /// End of sequence token ID
    pub eos_id: Option<u32>,
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

Configuration for text generation with enhanced robustness (NaN-safe sampling in PR #184).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: u32,
    
    /// Sampling temperature (0.0 = deterministic)
    /// Note: NaN values automatically sanitized to prevent crashes
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold
    /// Note: Enhanced NaN-safe filtering with graceful degradation
    pub top_p: f32,
    
    /// Top-k sampling limit  
    /// Note: Robust sorting with NaN-safe comparisons
    pub top_k: u32,
    
    /// Repetition penalty
    pub repetition_penalty: f32,
    
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
    
    /// Whether to skip special tokens in output
    pub skip_special_tokens: bool,
    
    /// Whether to stream output
    pub stream: bool,
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
            stream: false,
        }
    }
}
```

## GPU Infrastructure and Kernel Management

### CudaKernel (Enhanced in PR #199)

Low-level CUDA kernel provider with advanced GPU infrastructure access.

```rust
pub struct CudaKernel {
    // Internal fields for CUDA context, streams, modules, and device info
}

impl CudaKernel {
    /// Create a new CUDA kernel provider
    pub fn new() -> Result<Self>;
    
    /// Create a new CUDA kernel provider with specific device
    pub fn new_with_device(device_id: usize) -> Result<Self>;
    
    /// Get device information and capabilities
    pub fn device_info(&self) -> &CudaDeviceInfo;
    
    /// Get access to the CUDA context for advanced operations (New in PR #199)
    /// Enables custom kernel loading and advanced GPU memory management
    pub fn context(&self) -> Arc<CudaContext>;
    
    /// Get access to the CUDA module for loading additional kernels (New in PR #199)
    /// Allows loading of custom PTX kernels for specialized operations
    pub fn module(&self) -> Arc<CudaModule>;
    
    /// Synchronize all CUDA streams
    pub fn synchronize_all(&self) -> Result<()>;
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize);
    
    /// Get performance statistics
    pub fn performance_stats(&self) -> PerformanceStats;
    
    /// Reset performance statistics for benchmarking
    pub fn reset_performance_stats(&self);
    
    /// Batch matrix multiplication for multiple concurrent requests
    pub fn batch_matmul_i2s(&self, batches: &mut [BatchOperation<'_>]) -> Result<()>;
    
    /// Calculate optimal launch parameters based on device capabilities (Enhanced in PR #199)
    /// Now used internally for device-aware optimization instead of hardcoded values
    fn calculate_optimal_launch_params(&self, m: usize, n: usize) -> (usize, usize, usize);
}
```

**Advanced GPU Infrastructure Usage (New in PR #199):**

```rust
use bitnet_kernels::gpu::cuda::CudaKernel;
use cudarc::driver::{CudaModule, LaunchConfig};

// Create CUDA kernel with access to low-level infrastructure
let kernel = CudaKernel::new_with_device(0)?;

// Access CUDA context for advanced memory operations
let context = kernel.context();

// Access CUDA module for loading custom kernels
let module = kernel.module();

// Load custom PTX kernel for specialized operations
let custom_kernel = module.load_function("my_custom_kernel")?;

// Use device-aware launch parameter optimization
let device_info = kernel.device_info();
println!("Using device: {} with {} SMs", 
    device_info.name, device_info.multiprocessor_count);
```

**GPU Infrastructure Sequence (#199 → #202 → #206):**

PR #199 establishes the foundation for the GPU infrastructure enhancement sequence:
- **PR #199**: Exposes CUDA context and module access, integrates optimal launch parameters
- **PR #202**: Advanced GPU memory management and custom kernel loading
- **PR #206**: Multi-GPU support and advanced GPU orchestration

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

#[derive(Debug, Clone)]
pub struct DeviceStats {
    /// Device type description (e.g., "CudaKernel+FallbackKernel")
    pub device_type: String,
    /// Target device for operations
    pub target_device: Device,
    /// Total operations performed
    pub total_operations: u64,
    /// Number of quantization operations
    pub quantization_operations: u64,
    /// Number of matrix multiplication operations
    pub matmul_operations: u64,
    /// Total time spent on operations (ms)
    pub total_time_ms: f64,
    /// Time spent on quantization operations (ms)
    pub quantization_time_ms: f64,
    /// Time spent on matrix multiplication operations (ms)
    pub matmul_time_ms: f64,
    /// Operations performed on GPU
    pub gpu_operations: u64,
    /// Operations performed on CPU
    pub cpu_operations: u64,
    /// Number of GPU->CPU fallbacks
    pub fallback_count: u64,
    /// GPU efficiency ratio (0.0-1.0)
    pub gpu_efficiency: f64,
    /// Last GPU error message
    pub last_gpu_error: Option<String>,
    /// Last CPU error message
    pub last_cpu_error: Option<String>,
    /// Host memory currently used (bytes)
    pub memory_used_bytes: u64,
    /// Total host memory available (bytes)
    pub memory_total_bytes: u64,
}

impl DeviceStats {
    /// Get average quantization time per operation
    pub fn avg_quantization_time_ms(&self) -> f64;
    
    /// Get average matrix multiplication time per operation
    pub fn avg_matmul_time_ms(&self) -> f64;
    
    /// Check if GPU is effectively being used (>80% efficiency)
    pub fn is_gpu_effective(&self) -> bool;
    
    /// Get human-readable summary with memory usage
    pub fn summary(&self) -> String;
}
```

### Device-Aware Quantization

#### DeviceAwareQuantizer

Device-aware quantization provider with automatic GPU/CPU fallback and performance tracking.

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizer;

impl DeviceAwareQuantizer {
    /// Create a new device-aware quantizer for the specified device
    pub fn new(device: Device) -> Result<Self>;
    
    /// Get the currently active provider name
    pub fn active_provider(&self) -> &'static str;
    
    /// Check if GPU acceleration is currently active
    pub fn is_gpu_active(&self) -> bool;
    
    /// Get device information
    pub fn device(&self) -> Device;
    
    /// Perform quantization with automatic fallback
    pub fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()>;
    
    /// Matrix multiplication with device awareness
    pub fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
    
    /// Force fallback to CPU (for testing or reliability)
    pub fn force_cpu_fallback(&mut self);
    
    /// Get comprehensive performance statistics with memory tracking
    pub fn get_stats(&self) -> Option<DeviceStats>;
    
    /// Reset performance statistics (useful for benchmarking)
    pub fn reset_stats(&self);
}
```

#### DeviceAwareQuantizerFactory

Factory for creating device-aware quantizers with automatic device detection.

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizerFactory;

impl DeviceAwareQuantizerFactory {
    /// Create the best quantizer for the given device preference
    pub fn create_best(preferred_device: Option<Device>) -> Result<DeviceAwareQuantizer>;
    
    /// Create a quantizer with automatic GPU detection
    pub fn auto_detect() -> Result<DeviceAwareQuantizer>;
    
    /// List available devices
    pub fn list_available_devices() -> Vec<Device>;
}
```

**Example Usage:**

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizer;
use bitnet_common::{Device, QuantizationType};

// Auto-detect best device and create quantizer
let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

// Perform quantization (automatically falls back to CPU if GPU fails)
let input = vec![1.0f32, -1.0f32, 0.5f32, -0.5f32];
let mut output = vec![0u8; 1];
let mut scales = vec![0.0f32; 1];

quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S)?;

// Get performance statistics with memory tracking
if let Some(stats) = quantizer.get_stats() {
    println!("Device stats: {}", stats.summary());
    println!("GPU efficiency: {:.1}%", stats.gpu_efficiency * 100.0);
    println!("Memory usage: {:.1} MB / {:.1} MB", 
        stats.memory_used_bytes as f64 / (1024.0 * 1024.0),
        stats.memory_total_bytes as f64 / (1024.0 * 1024.0));
}
```

## Quantization

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
    /// IQ2_S quantization with 82-byte block layout (GGML-compatible)
    /// - Block Layout: 82-byte blocks matching GGML specification
    /// - Quantization Mapping: 4-level [-2,-1,1,2] mapping
    /// - Dual implementation: Native Rust + GGML FFI
    IQ2_S,
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

## Convolution Kernels

### Conv2DParams

Configuration parameters for 2D convolution operations.

```rust
#[derive(Clone, Copy, Debug)]
pub struct Conv2DParams {
    /// Stride along (height, width)
    pub stride: (usize, usize),
    
    /// Padding along (height, width)
    pub padding: (usize, usize),
    
    /// Dilation along (height, width)
    pub dilation: (usize, usize),
}

impl Default for Conv2DParams {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }
}
```

### conv2d

Perform 2D convolution with full-precision weights.

```rust
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),  // (N, C, H, W)
    weight_dims: (usize, usize, usize, usize), // (O, I, H, W)
) -> Result<()>
```

**Parameters:**
- `input`: Input tensor data in NCHW format (batch, channels, height, width)
- `weight`: Convolution kernel weights in OIHW format (out_channels, in_channels, height, width)
- `bias`: Optional bias vector with length equal to output channels
- `output`: Output buffer to store convolution results
- `params`: Convolution parameters (stride, padding, dilation)
- `input_dims`: Input tensor dimensions (N, C, H, W)
- `weight_dims`: Weight tensor dimensions (O, I, H, W)

**Features:**
- Supports stride, padding, and dilation operations
- NCHW input format and OIHW weight format
- Optional bias addition
- Comprehensive input validation
- Efficient memory access patterns

### conv2d_quantized

Perform 2D convolution with quantized weights.

```rust
pub fn conv2d_quantized(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()>
```

**Parameters:**
- `input`: Input tensor data in NCHW format
- `weight_quantized`: Quantized convolution kernel weights
- `weight_scales`: Scale factors for dequantizing weights per output channel
- `bias`: Optional bias vector
- `output`: Output buffer to store results
- `params`: Convolution parameters
- `input_dims`: Input tensor dimensions
- `weight_dims`: Weight tensor dimensions
- `qtype`: Quantization type (I2S, TL1, TL2)

**Quantization Support:**
- **I2S**: 2-bit signed quantization with values [-2, -1, 1, 2], packed 4 values per byte
- **TL1**: Table lookup quantization with linear mapping from [0,255] to [-1,1]
- **TL2**: Advanced table lookup quantization with non-linear mapping
- On-the-fly dequantization during convolution
- Per-channel scaling factors

**Example:**
```rust
use bitnet_kernels::convolution::{conv2d, Conv2DParams};

// Basic convolution
let input = vec![1.0, 2.0, 3.0, 4.0]; // 1x1x2x2
let weight = vec![1.0, 0.0, 0.0, 1.0]; // 1x1x2x2
let mut output = vec![0.0; 1]; // 1x1x1x1

let result = conv2d(
    &input,
    &weight,
    None, // No bias
    &mut output,
    Conv2DParams::default(),
    (1, 1, 2, 2), // Input: 1 batch, 1 channel, 2x2
    (1, 1, 2, 2), // Weight: 1 out_ch, 1 in_ch, 2x2
);

assert!(result.is_ok());
```

## Inference Engine

### InferenceEngine

Low-level inference engine for advanced use cases.

```rust
pub struct InferenceEngine {
    // Internal fields
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
    ) -> Result<Self, BitNetError>;
    
    /// Create with custom configuration
    pub fn with_config(
        model: Arc<dyn Model>,
        tokenizer: Arc<dyn Tokenizer>,
        device: Device,
        config: InferenceConfig,
    ) -> Result<Self, BitNetError>;
    
    /// Run inference on token IDs
    pub async fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[bool]>,
    ) -> Result<Tensor, BitNetError>;
    
    /// Generate next token probabilities
    pub async fn next_token_logits(
        &self,
        input_ids: &[u32],
        temperature: f32,
    ) -> Result<Vec<f32>, BitNetError>;
    
    /// Sample next token from logits
    pub fn sample_token(
        &self,
        logits: &[f32],
        config: &SamplingConfig,
    ) -> Result<u32, BitNetError>;
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// KV cache size
    pub kv_cache_size: usize,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Whether to use mixed precision
    pub use_mixed_precision: bool,
}
```

## Sampling API (Enhanced in PR #184)

### Enhanced NaN-Safe Sampling

BitNet.rs provides robust sampling with comprehensive NaN handling to prevent crashes from model output anomalies.

```rust
use bitnet_cli::sampling::Sampler;

/// NaN-safe sampling utilities for text generation
pub struct Sampler {
    // Internal fields for temperature, top-k, top-p, repetition penalty, etc.
}

impl Sampler {
    /// Create a new sampler with given parameters
    pub fn new(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> Self;
    
    /// Sample next token from logits with NaN safety
    /// - Automatically sanitizes NaN logits to negative infinity
    /// - Prevents crashes from numerical edge cases
    /// - Maintains deterministic behavior with proper fallback logic
    pub fn sample(&mut self, logits: &[f32], generated_tokens: &[u32]) -> u32;
    
    /// Apply top-k filtering with NaN awareness
    /// - Filters out NaN values before processing
    /// - Uses safe partial_cmp() with fallback ordering
    /// - Maintains stable sorting for reproducible results
    fn top_k_filter(&self, logits: Vec<f32>) -> Vec<f32>;
    
    /// Apply top-p (nucleus) filtering with NaN safety
    /// - Sanitizes NaN values to negative infinity
    /// - Handles cumulative probability calculation safely
    /// - Graceful degradation for numerical anomalies
    fn top_p_filter(&self, logits: Vec<f32>) -> Vec<f32>;
}

/// Enhanced utility functions with NaN safety
pub fn softmax(logits: &[f32]) -> Vec<f32>;
pub fn argmax(logits: &[f32]) -> u32;

/// NaN-Safe Sampling Configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling temperature (NaN values automatically sanitized)
    pub temperature: f32,
    /// Top-k limit with NaN-safe filtering
    pub top_k: usize,
    /// Top-p threshold with robust numerical handling
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Random seed for reproducible generation
    pub seed: Option<u64>,
}
```

**Key Enhancements (PR #184):**

- **Automatic NaN Sanitization**: Converts NaN logits to `-inf` for predictable behavior
- **Safe Sorting Operations**: Uses `partial_cmp().unwrap_or(Ordering::Equal)` for robust comparisons
- **Graceful Error Recovery**: Maintains sampling functionality even with model output anomalies
- **Deterministic Fallbacks**: Consistent behavior across different hardware and edge cases
- **No Runtime Crashes**: Prevents sampling failures from numerical instabilities

## Tokenization

### Tokenizer Discovery and Auto-Download (Issue #249)

BitNet.rs provides intelligent tokenizer discovery and automatic downloading for seamless neural network model integration.

#### TokenizerDiscovery

Comprehensive tokenizer discovery engine for GGUF metadata parsing and neural network model compatibility.

```rust
use bitnet_tokenizers::{TokenizerDiscovery, TokenizerStrategy};
use std::path::Path;

pub struct TokenizerDiscovery {
    // Internal fields for GGUF reader and model metadata
}

impl TokenizerDiscovery {
    /// Create discovery engine from GGUF model file
    pub fn from_gguf(path: &Path) -> Result<Self>;

    /// Discover optimal tokenizer strategy for the loaded model
    pub fn discover_tokenizer_strategy(&self) -> Result<TokenizerStrategy>;

    /// Get vocabulary size from model metadata
    pub fn vocab_size(&self) -> usize;

    /// Get model architecture type (e.g., "llama", "gpt2")
    pub fn model_type(&self) -> &str;

    /// Check if model requires large vocabulary optimization (>64K tokens)
    /// Large vocabularies require GPU acceleration for efficient embedding lookup
    pub fn requires_large_vocab_optimization(&self) -> bool;

    /// Check for co-located tokenizer files in model directory
    pub fn check_colocated_tokenizers(&self) -> Result<Option<PathBuf>>;

    /// Check standard cache directories for compatible tokenizers
    pub fn check_cache_locations(&self) -> Result<Option<PathBuf>>;

    /// Infer download source based on neural network model patterns
    pub fn infer_download_source(&self) -> Result<Option<TokenizerDownloadInfo>>;

    /// Try to extract embedded tokenizer from GGUF metadata
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>>;
}
```

#### TokenizerStrategy

Comprehensive tokenizer resolution strategy for neural network models.

```rust
#[derive(Clone)]
pub enum TokenizerStrategy {
    /// User explicitly specified tokenizer path
    Exact(PathBuf),
    /// Auto-discovered compatible tokenizer in model directory
    Discovered(PathBuf),
    /// Smart download required from HuggingFace Hub
    NeedsDownload(TokenizerDownloadInfo),
    /// GGUF file contains embedded tokenizer data
    EmbeddedGguf(Arc<dyn Tokenizer>),
    /// Mock tokenizer for testing (non-strict mode only)
    Mock,
}

impl TokenizerStrategy {
    /// Check if strategy requires network access
    pub fn requires_network(&self) -> bool;

    /// Check if strategy uses cached resources
    pub fn uses_cache(&self) -> bool;

    /// Get description for logging and error messages
    pub fn description(&self) -> &'static str;
}
```

#### SmartTokenizerDownload

Intelligent tokenizer downloading with caching, resume capability, and validation.

```rust
use bitnet_tokenizers::{SmartTokenizerDownload, TokenizerDownloadInfo};

pub struct SmartTokenizerDownload {
    // Internal fields for cache management and HTTP client
}

impl SmartTokenizerDownload {
    /// Initialize download system with default cache directory
    pub fn new() -> Result<Self>;

    /// Initialize with custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self>;

    /// Download tokenizer files for given download info
    /// Supports resume capability and automatic validation
    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf>;

    /// Check if tokenizer is already cached
    pub fn find_cached_tokenizer(&self, cache_key: &str) -> Option<PathBuf>;

    /// Clear cache for specific tokenizer or all cached tokenizers
    pub fn clear_cache(&self, cache_key: Option<&str>) -> Result<()>;

    /// Validate downloaded tokenizer against expected metadata
    pub fn validate_downloaded_tokenizer(&self, path: &Path, info: &TokenizerDownloadInfo) -> Result<()>;
}
```

#### ModelCompatibilityMatrix

Neural network model compatibility matrix for tokenizer discovery.

```rust
#[derive(Debug, Clone)]
pub struct ModelCompatibilityMatrix {
    /// LLaMA-3 with 128K vocabulary - requires I2S quantization with GPU acceleration
    pub llama3_128k: TokenizerDownloadInfo,
    /// LLaMA-2 with 32K vocabulary - compatible with TL1/TL2 quantization
    pub llama2_32k: TokenizerDownloadInfo,
    /// GPT-2 with 50K vocabulary - standard BPE tokenization
    pub gpt2_50k: TokenizerDownloadInfo,
    /// BitNet-specific tokenizers for neural network optimization
    pub bitnet_custom: TokenizerDownloadInfo,
}

#[derive(Debug, Clone)]
pub struct TokenizerDownloadInfo {
    /// HuggingFace repository identifier (e.g., "meta-llama/Llama-2-7b-hf")
    pub repo: String,
    /// Required tokenizer files to download (e.g., ["tokenizer.json"])
    pub files: Vec<String>,
    /// Cache identifier for persistent storage (e.g., "llama2-32k")
    pub cache_key: String,
    /// Expected vocabulary size for validation (optional)
    pub expected_vocab: Option<usize>,
}
```

**Example Usage:**

```rust
use bitnet_tokenizers::{TokenizerDiscovery, SmartTokenizerDownload};
use std::path::Path;

// Discover tokenizer strategy from GGUF model
let discovery = TokenizerDiscovery::from_gguf(Path::new("model.gguf"))?;
let strategy = discovery.discover_tokenizer_strategy()?;

match strategy {
    TokenizerStrategy::Discovered(path) => {
        println!("Found tokenizer: {}", path.display());
    },
    TokenizerStrategy::NeedsDownload(info) => {
        let downloader = SmartTokenizerDownload::new()?;
        let tokenizer_path = downloader.download_tokenizer(&info).await?;
        println!("Downloaded tokenizer: {}", tokenizer_path.display());
    },
    TokenizerStrategy::EmbeddedGguf(tokenizer) => {
        println!("Using embedded tokenizer (vocab: {})", tokenizer.vocab_size());
    },
    _ => println!("Strategy: {}", strategy.description()),
}
```

**Environment Variables:**

- `BITNET_STRICT_TOKENIZERS=1`: Prevent mock fallbacks for production use
- `BITNET_OFFLINE=1`: Disable network downloads, use cache only
- `BITNET_DETERMINISTIC=1`: Enable deterministic behavior for testing

### Tokenizer

Interface for text tokenization.

```rust
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs with special token handling
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>, TokenizerError>;

    /// Decode token IDs to text
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Convert token ID to piece string for inspection
    fn token_to_piece(&self, token: u32) -> Option<String>;
}

/// Universal tokenizer with auto-detection and fallback support
pub struct UniversalTokenizer {
    // Internal backend implementation
}

impl UniversalTokenizer {
    /// Create tokenizer with explicit configuration
    pub fn new(config: TokenizerConfig) -> Result<Self>;

    /// Create tokenizer from GGUF model file with embedded metadata
    pub fn from_gguf<P: AsRef<Path>>(path: P) -> Result<Self>;
}

#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Model type (e.g., "llama", "gpt2", "sentencepiece")
    pub model_type: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Path to tokenizer file (for SentencePiece: .model file)
    pub pre_tokenizer: Option<String>,
    /// Add BOS token automatically
    pub add_bos: bool,
    /// Add EOS token automatically
    pub add_eos: bool,
    /// Add space prefix for GPT-style tokenizers
    pub add_space_prefix: bool,
    /// Use byte fallback for unknown characters
    pub byte_fallback: bool,
    /// Special token IDs
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    /// Embedded vocabulary (extracted from GGUF)
    pub vocabulary: Option<Vec<String>>,
    /// BPE merge rules (for BPE tokenizers)
    pub bpe_merges: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}
```

## Error Handling

### BitNetError

Enhanced error type for BitNet operations with comprehensive GGUF and quantization error handling.

```rust
#[derive(thiserror::Error, Debug)]
pub enum BitNetError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Tokenization error: {0}")]
    Tokenization(#[from] TokenizerError),

    #[error("GGUF parsing error: {0}")]
    GgufParse(#[from] GgufParseError),

    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Security error: {0}")]
    Security(String),

    #[error("Timeout error: operation timed out after {0:?}")]
    Timeout(Duration),

    #[error("Capacity error: {0}")]
    Capacity(String),
}

/// Enhanced GGUF parsing errors with detailed context
#[derive(thiserror::Error, Debug)]
pub enum GgufParseError {
    #[error("Invalid GGUF magic bytes: expected 'GGUF', got '{0}'")]
    InvalidMagic(String),

    #[error("Unsupported GGUF version: {0} (supported: 1-3)")]
    UnsupportedVersion(u32),

    #[error("Invalid tensor count: {0} exceeds security limit {1}")]
    InvalidTensorCount(u32, u32),

    #[error("Invalid KV count: {0} exceeds security limit {1}")]
    InvalidKvCount(u32, u32),

    #[error("Missing required tensor: '{0}'")]
    MissingTensor(String),

    #[error("Invalid tensor shape for '{0}': expected {1:?}, got {2:?}")]
    InvalidTensorShape(String, Vec<usize>, Vec<usize>),

    #[error("Tensor data corruption in '{0}': {1}")]
    TensorDataCorruption(String, String),

    #[error("Unsupported tensor type: {0}")]
    UnsupportedTensorType(String),

    #[error("File truncated: expected {0} bytes, got {1}")]
    FileTruncated(u64, u64),

    #[error("Memory mapping failed: {0}")]
    MemoryMappingFailed(String),

    #[error("Tensor alignment error for '{0}': offset {1} not aligned to {2}")]
    TensorAlignment(String, u64, u64),
}

/// Enhanced quantization errors with context and recovery suggestions
#[derive(thiserror::Error, Debug)]
pub enum QuantizationError {
    #[error("Unsupported quantization format: {0}")]
    UnsupportedFormat(String),

    #[error("Quantization accuracy too low: {0:.2}% (minimum: {1:.2}%)")]
    AccuracyTooLow(f64, f64),

    #[error("Input data validation failed: {0}")]
    InputValidation(String),

    #[error("Output buffer too small: need {0} bytes, got {1}")]
    OutputBufferTooSmall(usize, usize),

    #[error("Scale buffer mismatch: expected {0} scales, got {1}")]
    ScaleBufferMismatch(usize, usize),

    #[error("Device quantization failed: {0}")]
    DeviceQuantizationFailed(String),

    #[error("GPU quantization not available: {0}")]
    GpuNotAvailable(String),

    #[error("CUDA error during quantization: {0}")]
    CudaError(String),

    #[error("Memory allocation failed for quantization: {0}")]
    MemoryAllocation(String),

    #[error("Numerical overflow in quantization: input range [{0}, {1}] exceeds format limits")]
    NumericalOverflow(f32, f32),

    #[error("Zero-point calculation failed: {0}")]
    ZeroPointCalculation(String),

    #[error("Scale calculation failed: {0}")]
    ScaleCalculation(String),
}

/// Security validation errors for production deployments
#[derive(thiserror::Error, Debug)]
pub enum SecurityError {
    #[error("Model size exceeds security limit: {0} bytes > {1} bytes")]
    ModelSizeLimit(u64, u64),

    #[error("Tensor count exceeds security limit: {0} > {1}")]
    TensorCountLimit(usize, usize),

    #[error("Memory allocation exceeds security limit: {0} bytes > {1} bytes")]
    MemoryLimit(u64, u64),

    #[error("Unsafe tensor operation detected: {0}")]
    UnsafeTensorOperation(String),

    #[error("Input validation failed: {0}")]
    InputValidation(String),

    #[error("Resource exhaustion detected: {0}")]
    ResourceExhaustion(String),
}

/// Device-specific errors with diagnostic information
#[derive(thiserror::Error, Debug)]
pub enum DeviceError {
    #[error("CUDA device {0} not available")]
    CudaNotAvailable(usize),

    #[error("CUDA out of memory: requested {0} bytes, available {1} bytes")]
    CudaOutOfMemory(u64, u64),

    #[error("CUDA computation failed: {0}")]
    CudaComputation(String),

    #[error("CPU fallback required: {0}")]
    CpuFallbackRequired(String),

    #[error("Device capability insufficient: requires {0}, available {1}")]
    InsufficientCapability(String, String),

    #[error("Multi-device synchronization failed: {0}")]
    SynchronizationFailed(String),
}

pub type Result<T> = std::result::Result<T, BitNetError>;

/// Error context for enhanced debugging and recovery
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation being performed when error occurred
    pub operation: String,

    /// File path if relevant
    pub file_path: Option<PathBuf>,

    /// Model configuration if available
    pub model_config: Option<String>,

    /// Device information
    pub device_info: Option<String>,

    /// Memory usage at time of error
    pub memory_usage: Option<u64>,

    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,

    /// Environment variables that may be relevant
    pub relevant_env_vars: Vec<(String, Option<String>)>,
}

impl ErrorContext {
    /// Create error context for GGUF loading operation
    pub fn gguf_loading(file_path: &Path) -> Self;

    /// Create error context for quantization operation
    pub fn quantization(device: Device, qtype: QuantizationType) -> Self;

    /// Create error context for device operation
    pub fn device_operation(device: Device, operation: &str) -> Self;

    /// Add recovery suggestion to error context
    pub fn with_suggestion(mut self, suggestion: &str) -> Self;

    /// Format error context for display
    pub fn format_for_display(&self) -> String;
}

/// Enhanced error with context and recovery information
#[derive(Debug)]
pub struct ContextualError {
    /// The underlying error
    pub error: BitNetError,

    /// Additional context for debugging
    pub context: ErrorContext,

    /// Error severity level
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorSeverity {
    /// Warning - operation can continue
    Warning,

    /// Error - operation failed but recoverable
    Error,

    /// Critical - system in unstable state
    Critical,

    /// Fatal - immediate termination required
    Fatal,
}

impl ContextualError {
    /// Create a contextual error with suggestions
    pub fn new(error: BitNetError, context: ErrorContext) -> Self;

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool;

    /// Get suggested recovery actions
    pub fn recovery_actions(&self) -> &[String];

    /// Format error for end-user display
    pub fn user_friendly_message(&self) -> String;

    /// Format error for developer debugging
    pub fn debug_message(&self) -> String;
}
```

### Enhanced Error Handling Examples

#### GGUF Loading Error Handling

```rust
use bitnet_models::gguf_simple::load_gguf;
use bitnet_common::{Device, BitNetError, GgufParseError};

fn handle_gguf_loading(path: &Path) -> Result<()> {
    match load_gguf(path, Device::Cpu) {
        Ok((config, tensors)) => {
            println!("Loaded {} tensors successfully", tensors.len());
            Ok(())
        }
        Err(BitNetError::GgufParse(GgufParseError::InvalidMagic(magic))) => {
            eprintln!("Invalid GGUF file: magic bytes '{}'", magic);
            eprintln!("Suggestion: Verify file is a valid GGUF model");
            eprintln!("Command: file {}", path.display());
            Err(BitNetError::GgufParse(GgufParseError::InvalidMagic(magic)))
        }
        Err(BitNetError::GgufParse(GgufParseError::UnsupportedVersion(version))) => {
            eprintln!("Unsupported GGUF version: {}", version);
            eprintln!("Suggestion: Convert model to supported version (1-3)");
            eprintln!("Command: cargo run -p bitnet-cli -- convert --target-version 3");
            Err(BitNetError::GgufParse(GgufParseError::UnsupportedVersion(version)))
        }
        Err(BitNetError::Validation(msg)) => {
            eprintln!("Model validation failed: {}", msg);
            eprintln!("Suggestion: Check model integrity and format");
            eprintln!("Command: cargo run -p bitnet-cli -- compat-check {}", path.display());
            Err(BitNetError::Validation(msg))
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            Err(e)
        }
    }
}
```

#### Quantization Error Handling

```rust
use bitnet_kernels::device_aware::DeviceAwareQuantizer;
use bitnet_common::{Device, QuantizationType, QuantizationError};

fn handle_quantization_errors() -> Result<()> {
    let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;
    let input = vec![1.0; 1024];
    let mut output = vec![0u8; 256];
    let mut scales = vec![0.0f32; 4];

    match quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S) {
        Ok(()) => println!("Quantization successful"),
        Err(BitNetError::Quantization(QuantizationError::GpuNotAvailable(msg))) => {
            eprintln!("GPU quantization failed: {}", msg);
            eprintln!("Suggestion: Check CUDA installation and GPU availability");
            eprintln!("Command: nvidia-smi");
            eprintln!("Fallback: Use CPU quantization with --no-default-features --features cpu");
        }
        Err(BitNetError::Quantization(QuantizationError::AccuracyTooLow(actual, required))) => {
            eprintln!("Quantization accuracy too low: {:.2}% < {:.2}%", actual, required);
            eprintln!("Suggestion: Try different quantization format or adjust input range");
            eprintln!("Alternatives: TL1, TL2, or full precision (F32)");
        }
        Err(e) => eprintln!("Quantization error: {}", e),
    }

    Ok(())
}
```

#### Production Error Handling

```rust
use bitnet_common::{ContextualError, ErrorContext, ErrorSeverity};

fn production_error_handler(error: BitNetError, operation: &str) -> ContextualError {
    let context = match &error {
        BitNetError::GgufParse(_) => {
            ErrorContext::gguf_loading(Path::new("model.gguf"))
                .with_suggestion("Verify model file integrity")
                .with_suggestion("Check available disk space")
                .with_suggestion("Try re-downloading the model")
        }
        BitNetError::Quantization(_) => {
            ErrorContext::quantization(Device::Auto, QuantizationType::I2S)
                .with_suggestion("Check GPU memory availability")
                .with_suggestion("Try CPU fallback")
                .with_suggestion("Reduce batch size")
        }
        BitNetError::Device(_) => {
            ErrorContext::device_operation(Device::Auto, operation)
                .with_suggestion("Check CUDA installation")
                .with_suggestion("Verify device availability")
                .with_suggestion("Enable CPU fallback")
        }
        _ => ErrorContext::default(),
    };

    let severity = match &error {
        BitNetError::Security(_) => ErrorSeverity::Critical,
        BitNetError::Memory(_) => ErrorSeverity::Error,
        BitNetError::Device(_) => ErrorSeverity::Warning,
        _ => ErrorSeverity::Error,
    };

    ContextualError {
        error,
        context,
        severity,
    }
}
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
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Backend type (CPU/GPU)
    pub backend_type: String,
    
    /// Total tokens generated
    pub tokens_generated: u64,
    
    /// Tokens per second throughput
    pub tokens_per_second: f64,
    
    /// Total inference latency (ms)
    pub total_latency_ms: u64,
    
    /// First token latency (ms) - critical for streaming
    pub first_token_latency_ms: Option<u64>,
    
    /// Average per-token latency (ms)
    pub average_token_latency_ms: Option<u64>,
    
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: Option<f64>,
    
    /// Memory usage (bytes)
    pub memory_usage_bytes: Option<u64>,
    
    /// Component timing breakdown
    pub tokenizer_encode_time_ms: Option<u64>,
    pub tokenizer_decode_time_ms: Option<u64>,
    pub forward_pass_time_ms: Option<u64>,
    pub sampling_time_ms: Option<u64>,
    
    /// Error count and rates
    pub error_count: u64,
    pub error_rate: f64,
}

impl PerformanceMetrics {
    /// Validate metrics for consistency
    pub fn validate(&self) -> Result<(), String>;
    
    /// Calculate efficiency ratio (tokens per millisecond)
    pub fn efficiency_ratio(&self) -> f64;
}

#[derive(Debug, Default)]
pub struct PerformanceTracker {
    // Internal fields for tracking metrics
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new() -> Self;
    
    /// Record inference operation
    pub fn record_inference(&mut self, tokens: u64, duration_ms: u64);
    
    /// Record cache hit
    pub fn record_cache_hit(&mut self);
    
    /// Record cache miss
    pub fn record_cache_miss(&mut self);
    
    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> Option<f64>;
    
    /// Get average tokens per second
    pub fn get_average_tokens_per_second(&self) -> f64;
}

impl InferenceEngine {
    /// Get comprehensive performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics, anyhow::Error>;
    
    /// Reset performance tracking for clean benchmarking
    pub fn reset_performance_tracking(&self) -> Result<(), anyhow::Error>;
    
    /// Apply environment variable performance configuration
    pub async fn apply_env_performance_config(&mut self) -> Result<(), anyhow::Error>;
}
```

#### Environment Variables

Performance behavior can be controlled via environment variables:

- `BITNET_DETERMINISTIC=1`: Enable deterministic execution mode
- `BITNET_SEED=<number>`: Set random seed for reproducible results  
- `BITNET_BATCH_SIZE=<number>`: Configure inference batch size
- `BITNET_MEMORY_LIMIT=<size>`: Set memory usage limits
- `BITNET_NUM_THREADS=<number>`: Control inference thread count
- `RAYON_NUM_THREADS=<number>`: Control CPU parallelism

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