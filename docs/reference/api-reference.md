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
    /// Load a model from a local GGUF file, SafeTensors file, or HuggingFace directory
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
    
    /// Prefill the model cache with given tokens for performance optimization
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<(), BitNetError>;
    
    /// Get model information
    pub fn model_info(&self) -> &ModelInfo;
    
    /// Get model configuration
    pub fn config(&self) -> &ModelConfig;
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