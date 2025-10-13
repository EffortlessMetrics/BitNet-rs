# API Contracts Specification

**Component**: Forward pass interfaces and error handling patterns for BitNet.rs inference
**Location**: `bitnet-inference/src/api/contracts.rs`
**Dependencies**: All inference components (transformer, attention, generation)

## Overview

This specification defines the comprehensive API contracts for BitNet.rs neural network inference, establishing clear interfaces for transformer forward passes, generation workflows, and robust error handling patterns. These contracts ensure consistent behavior across CPU/GPU backends, provide type safety for quantized operations, and enable seamless integration with the broader BitNet.rs ecosystem.

## Core API Contracts

### Primary Inference Interface

```rust
/// Core inference trait for all BitNet transformer implementations
pub trait BitNetInference: Send + Sync {
    /// Forward pass through transformer with optional KV cache
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor [batch_size, sequence_length]
    /// * `kv_cache` - Optional KV cache for autoregressive generation
    ///
    /// # Returns
    /// * `InferenceOutput` - Logits and attention outputs
    ///
    /// # Performance Contract
    /// * CPU: <10ms latency per token (single batch)
    /// * GPU: <2ms latency per token (single batch)
    /// * Memory: <8GB peak usage for 2B parameter model
    fn forward(
        &mut self,
        input_ids: &Tensor,
        kv_cache: Option<&mut KVCache>
    ) -> Result<InferenceOutput>;

    /// Generate text autoregressively with sampling
    ///
    /// # Arguments
    /// * `prompt_tokens` - Input prompt as token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `config` - Generation configuration (sampling, penalties)
    ///
    /// # Returns
    /// * `GenerationResult` - Complete generation with metrics
    ///
    /// # Performance Contract
    /// * CPU: 5-15 tok/sec for 2B model
    /// * GPU: 15-45 tok/sec for 2B model
    /// * Accuracy: >99% correlation with FP32 reference
    fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        config: &GenerationConfig
    ) -> Result<GenerationResult>;

    /// Deterministic generation with fixed seed
    ///
    /// # Contract
    /// * Multiple calls with same seed MUST produce identical results
    /// * Thread-safe deterministic generation
    /// * No performance degradation compared to non-deterministic mode
    fn generate_deterministic(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        seed: u64
    ) -> Result<GenerationResult>;

    /// Get model configuration and capabilities
    fn config(&self) -> &ModelConfig;

    /// Get current device (CPU/GPU)
    fn device(&self) -> &Device;

    /// Check if model supports specific quantization type
    fn supports_quantization(&self, qtype: QuantizationType) -> bool;

    /// Get memory usage statistics
    fn memory_usage(&self) -> MemoryUsage;

    /// Reset internal state (clear caches, reset position)
    fn reset(&mut self) -> Result<()>;
}
```

### Model Configuration Contract

```rust
/// Comprehensive model configuration with validation
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    // Architecture parameters
    pub hidden_size: usize,              // Model hidden dimension
    pub num_layers: usize,               // Number of transformer layers
    pub num_attention_heads: usize,      // Number of attention heads
    pub num_key_value_heads: usize,      // Number of KV heads (for GQA)
    pub intermediate_size: usize,        // Feed-forward intermediate size
    pub vocab_size: usize,               // Vocabulary size

    // Sequence configuration
    pub max_position_embeddings: usize, // Maximum sequence length
    pub rope_theta: f32,                 // RoPE base frequency

    // Quantization settings
    pub quantization_type: QuantizationType, // I2S, TL1, TL2
    pub quantization_accuracy: f32,      // Required accuracy threshold

    // Model metadata
    pub model_type: String,              // Model architecture name
    pub model_name: Option<String>,      // Human-readable model name
    pub version: String,                 // Model version/checkpoint
}

impl ModelConfig {
    /// Validate configuration consistency
    pub fn validate(&self) -> Result<()> {
        // Hidden size divisibility
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigurationError::InvalidDimensions {
                hidden_size: self.hidden_size,
                num_heads: self.num_attention_heads,
                reason: "hidden_size must be divisible by num_attention_heads".to_string(),
            });
        }

        // GQA head consistency
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ConfigurationError::InvalidDimensions {
                hidden_size: self.num_attention_heads,
                num_heads: self.num_key_value_heads,
                reason: "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            });
        }

        // Reasonable parameter ranges
        if self.vocab_size == 0 || self.vocab_size > 1_000_000 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "vocab_size".to_string(),
                value: self.vocab_size.to_string(),
                valid_range: "1 to 1,000,000".to_string(),
            });
        }

        if self.max_position_embeddings == 0 || self.max_position_embeddings > 1_000_000 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "max_position_embeddings".to_string(),
                value: self.max_position_embeddings.to_string(),
                valid_range: "1 to 1,000,000".to_string(),
            });
        }

        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get GQA group size
    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Calculate approximate model parameter count
    pub fn parameter_count(&self) -> usize {
        // Embeddings
        let embedding_params = self.vocab_size * self.hidden_size;

        // Transformer layers
        let attention_params = self.num_layers * (
            // QKV projections
            self.hidden_size * (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim()) +
            // Output projection
            self.hidden_size * self.hidden_size
        );

        let ffn_params = self.num_layers * (
            // Gate and up projections
            2 * self.hidden_size * self.intermediate_size +
            // Down projection
            self.intermediate_size * self.hidden_size
        );

        let norm_params = (self.num_layers + 1) * self.hidden_size; // Layer norms

        embedding_params + attention_params + ffn_params + norm_params
    }

    /// Get expected quantized model size in bytes
    pub fn quantized_model_size(&self) -> usize {
        let full_params = self.parameter_count();
        let bits_per_param = match self.quantization_type {
            QuantizationType::I2S => 2,
            QuantizationType::TL1 => 4,
            QuantizationType::TL2 => 8,
        };

        // Quantized weights + scales + metadata
        let quantized_weights = (full_params * bits_per_param) / 8;
        let scales = full_params / 32 * 4; // FP32 scales for each block
        let metadata = 1024 * 1024; // 1MB for metadata

        quantized_weights + scales + metadata
    }
}
```

### Inference Output Contracts

```rust
/// Standardized inference output with performance metrics
#[derive(Debug)]
pub struct InferenceOutput {
    /// Logits tensor [batch_size, sequence_length, vocab_size]
    pub logits: Tensor,

    /// Optional attention weights for analysis [num_layers, batch_size, num_heads, seq_len, seq_len]
    pub attention_weights: Option<Vec<Tensor>>,

    /// Hidden states from final layer [batch_size, sequence_length, hidden_size]
    pub hidden_states: Option<Tensor>,

    /// Performance metrics for this forward pass
    pub metrics: ForwardPassMetrics,

    /// Memory usage during computation
    pub memory_usage: MemoryUsage,
}

impl InferenceOutput {
    /// Validate output tensor shapes and contents
    pub fn validate(&self, expected_batch_size: usize, expected_seq_len: usize, vocab_size: usize) -> Result<()> {
        let logits_dims = self.logits.dims();

        // Check logits shape
        if logits_dims.len() != 3 {
            return Err(OutputValidationError::InvalidShape {
                tensor: "logits".to_string(),
                expected: vec![expected_batch_size, expected_seq_len, vocab_size],
                actual: logits_dims.to_vec(),
            });
        }

        if logits_dims != [expected_batch_size, expected_seq_len, vocab_size] {
            return Err(OutputValidationError::InvalidShape {
                tensor: "logits".to_string(),
                expected: vec![expected_batch_size, expected_seq_len, vocab_size],
                actual: logits_dims.to_vec(),
            });
        }

        // Check for numerical issues
        self.check_numerical_stability()?;

        Ok(())
    }

    /// Check for NaN/Inf values in output tensors
    fn check_numerical_stability(&self) -> Result<()> {
        let logits_data: Vec<f32> = self.logits.flatten_all()?.to_vec1()?;

        let invalid_count = logits_data.iter().filter(|&&x| !x.is_finite()).count();
        if invalid_count > 0 {
            return Err(OutputValidationError::NumericalInstability {
                tensor: "logits".to_string(),
                invalid_values: invalid_count,
                total_values: logits_data.len(),
            });
        }

        // Check for extreme values that might indicate numerical issues
        let max_abs = logits_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 100.0 {
            log::warn!("Logits contain large values (max_abs: {:.2}), may indicate numerical issues", max_abs);
        }

        Ok(())
    }

    /// Extract next token probabilities for generation
    pub fn next_token_logits(&self) -> Result<Tensor> {
        let dims = self.logits.dims();
        if dims.len() != 3 {
            return Err(OutputValidationError::InvalidShape {
                tensor: "logits".to_string(),
                expected: vec![1, 1, dims[2]],
                actual: dims.to_vec(),
            });
        }

        // Get logits for last position: [B, T, V] -> [B, V]
        let last_position = dims[1] - 1;
        self.logits.narrow(1, last_position, 1)?.squeeze(1)
    }
}
```

### Performance Metrics Contracts

```rust
/// Detailed performance metrics for forward pass operations
#[derive(Debug, Clone, Default)]
pub struct ForwardPassMetrics {
    /// Total forward pass time
    pub total_time: Duration,

    /// Time breakdown by component
    pub embedding_time: Duration,
    pub attention_time: Duration,
    pub feedforward_time: Duration,
    pub normalization_time: Duration,
    pub logits_time: Duration,

    /// Quantization-specific metrics
    pub quantization_time: Duration,
    pub dequantization_time: Duration,

    /// Memory metrics
    pub peak_memory_usage: usize,
    pub cache_memory_usage: usize,

    /// Computational metrics
    pub flops_estimate: u64,
    pub memory_bandwidth_usage: f64,
}

impl ForwardPassMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate throughput in tokens per second
    pub fn throughput(&self, num_tokens: usize) -> f32 {
        if self.total_time.as_secs_f32() > 0.0 {
            num_tokens as f32 / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }

    /// Calculate efficiency metrics
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            compute_utilization: self.calculate_compute_utilization(),
            memory_efficiency: self.calculate_memory_efficiency(),
            quantization_overhead: self.calculate_quantization_overhead(),
        }
    }

    fn calculate_compute_utilization(&self) -> f32 {
        // Simplified computation utilization estimate
        let useful_compute_time = self.attention_time + self.feedforward_time;
        let total_compute_time = self.total_time;

        if total_compute_time.as_secs_f32() > 0.0 {
            useful_compute_time.as_secs_f32() / total_compute_time.as_secs_f32()
        } else {
            0.0
        }
    }

    fn calculate_memory_efficiency(&self) -> f32 {
        // Estimate memory efficiency based on bandwidth usage
        if self.memory_bandwidth_usage > 0.0 {
            (self.memory_bandwidth_usage / 1000.0).min(1.0) as f32 // Normalize to theoretical peak
        } else {
            0.0
        }
    }

    fn calculate_quantization_overhead(&self) -> f32 {
        let quant_time = self.quantization_time + self.dequantization_time;
        if self.total_time.as_secs_f32() > 0.0 {
            quant_time.as_secs_f32() / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    pub compute_utilization: f32,    // 0.0 - 1.0
    pub memory_efficiency: f32,      // 0.0 - 1.0
    pub quantization_overhead: f32,  // 0.0 - 1.0
}

/// Memory usage tracking for inference operations
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Current memory usage in bytes
    pub current_usage: usize,

    /// Peak memory usage during operation
    pub peak_usage: usize,

    /// Memory breakdown by component
    pub model_weights: usize,
    pub kv_cache: usize,
    pub activations: usize,
    pub temporary_buffers: usize,

    /// Memory efficiency metrics
    pub memory_pool_hits: usize,
    pub memory_pool_misses: usize,
    pub fragmentation_ratio: f32,
}

impl MemoryUsage {
    /// Calculate total memory footprint
    pub fn total_footprint(&self) -> usize {
        self.model_weights + self.kv_cache + self.activations + self.temporary_buffers
    }

    /// Get memory pool hit rate
    pub fn pool_hit_rate(&self) -> f32 {
        let total_requests = self.memory_pool_hits + self.memory_pool_misses;
        if total_requests > 0 {
            self.memory_pool_hits as f32 / total_requests as f32
        } else {
            0.0
        }
    }

    /// Check if memory usage is within acceptable limits
    pub fn within_limits(&self, max_memory: usize) -> bool {
        self.peak_usage <= max_memory
    }
}
```

## Error Handling Contracts

### Comprehensive Error Type Hierarchy

```rust
/// Root error type for all BitNet inference operations
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// Configuration-related errors
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),

    /// Model loading and initialization errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Forward pass computation errors
    #[error("Forward pass error: {0}")]
    ForwardPass(#[from] ForwardPassError),

    /// Generation-specific errors
    #[error("Generation error: {0}")]
    Generation(#[from] GenerationError),

    /// Quantization-related errors
    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),

    /// Device and hardware errors
    #[error("Device error: {0}")]
    Device(#[from] DeviceError),

    /// Output validation errors
    #[error("Output validation error: {0}")]
    OutputValidation(#[from] OutputValidationError),

    /// Memory-related errors
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(#[from] TokenizationError),
}

/// Configuration validation errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigurationError {
    #[error("Invalid dimensions: hidden_size={hidden_size}, num_heads={num_heads}, reason={reason}")]
    InvalidDimensions {
        hidden_size: usize,
        num_heads: usize,
        reason: String,
    },

    #[error("Invalid parameter {parameter}: value={value}, valid_range={valid_range}")]
    InvalidParameter {
        parameter: String,
        value: String,
        valid_range: String,
    },

    #[error("Incompatible configuration: {reason}")]
    IncompatibleConfig { reason: String },

    #[error("Missing required parameter: {parameter}")]
    MissingParameter { parameter: String },
}

/// Model loading and validation errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model file not found: {path}")]
    FileNotFound { path: String },

    #[error("Invalid model format: {format}, expected: {expected}")]
    InvalidFormat { format: String, expected: String },

    #[error("Model validation failed: {reason}")]
    ValidationFailed { reason: String },

    #[error("Incompatible model version: {version}, supported: {supported_versions:?}")]
    IncompatibleVersion {
        version: String,
        supported_versions: Vec<String>,
    },

    #[error("Model loading failed: {path} - {reason}")]
    LoadingFailed { path: String, reason: String },

    #[error("Model too large: {size_gb:.2}GB, maximum supported: {max_size_gb:.2}GB")]
    ModelTooLarge { size_gb: f64, max_size_gb: f64 },
}

/// Forward pass computation errors
#[derive(Debug, thiserror::Error)]
pub enum ForwardPassError {
    #[error("Input validation failed: {reason}")]
    InvalidInput { reason: String },

    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Computation failed in {component}: {reason}")]
    ComputationFailed { component: String, reason: String },

    #[error("Numerical instability detected: {details}")]
    NumericalInstability { details: String },

    #[error("KV cache error: {operation} failed - {reason}")]
    KVCacheError { operation: String, reason: String },
}

/// Output validation and consistency errors
#[derive(Debug, thiserror::Error)]
pub enum OutputValidationError {
    #[error("Invalid tensor shape for {tensor}: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Numerical instability in {tensor}: {invalid_values}/{total_values} invalid values")]
    NumericalInstability {
        tensor: String,
        invalid_values: usize,
        total_values: usize,
    },

    #[error("Output consistency check failed: {reason}")]
    InconsistentOutput { reason: String },

    #[error("Missing expected output: {output_name}")]
    MissingOutput { output_name: String },
}

/// Memory management errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Out of memory: requested {requested_mb}MB, available {available_mb}MB")]
    OutOfMemory {
        requested_mb: usize,
        available_mb: usize,
    },

    #[error("Memory allocation failed: {size_bytes} bytes - {reason}")]
    AllocationFailed {
        size_bytes: usize,
        reason: String,
    },

    #[error("Memory pool exhausted: {pool_type}")]
    PoolExhausted { pool_type: String },

    #[error("Memory fragmentation too high: {fragmentation_ratio:.2}")]
    HighFragmentation { fragmentation_ratio: f32 },
}

/// Device and hardware-related errors
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Device not available: {device}")]
    DeviceNotAvailable { device: String },

    #[error("Device capability insufficient: required {required}, available {available}")]
    InsufficientCapability {
        required: String,
        available: String,
    },

    #[error("CUDA error: {cuda_error_code} - {description}")]
    CudaError {
        cuda_error_code: i32,
        description: String,
    },

    #[error("Device memory error on {device}: {details}")]
    DeviceMemoryError { device: String, details: String },

    #[error("Device synchronization failed: {reason}")]
    SynchronizationFailed { reason: String },
}
```

### Error Context and Recovery

```rust
/// Error context for better debugging and recovery
pub trait ErrorContext {
    /// Get detailed context about where the error occurred
    fn context(&self) -> ErrorContextInfo;

    /// Check if the error is recoverable
    fn is_recoverable(&self) -> bool;

    /// Get suggested recovery actions
    fn recovery_suggestions(&self) -> Vec<RecoveryAction>;

    /// Get error severity level
    fn severity(&self) -> ErrorSeverity;
}

#[derive(Debug, Clone)]
pub struct ErrorContextInfo {
    pub component: String,           // Component where error occurred
    pub operation: String,           // Operation being performed
    pub input_info: Option<String>,  // Information about inputs
    pub system_state: SystemState,   // System state when error occurred
    pub timestamp: SystemTime,       // When the error occurred
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry,                          // Simple retry may work
    RetryWithDifferentConfig,       // Try with modified configuration
    FallbackToCPU,                 // Switch from GPU to CPU
    ReduceMemoryUsage,             // Reduce batch size or sequence length
    ClearCaches,                   // Clear KV cache and other caches
    RestartModel,                  // Reinitialize the model
    UserIntervention,              // Requires user action
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Low,        // Warning, operation can continue
    Medium,     // Error, but recoverable
    High,       // Critical error, immediate action required
    Fatal,      // Unrecoverable error
}

impl ErrorContext for InferenceError {
    fn context(&self) -> ErrorContextInfo {
        match self {
            InferenceError::Configuration(err) => ErrorContextInfo {
                component: "Configuration".to_string(),
                operation: "Validation".to_string(),
                input_info: Some(format!("{:?}", err)),
                system_state: SystemState::current(),
                timestamp: SystemTime::now(),
            },
            InferenceError::ForwardPass(err) => ErrorContextInfo {
                component: "ForwardPass".to_string(),
                operation: match err {
                    ForwardPassError::ComputationFailed { component, .. } => component.clone(),
                    _ => "Unknown".to_string(),
                },
                input_info: Some(format!("{:?}", err)),
                system_state: SystemState::current(),
                timestamp: SystemTime::now(),
            },
            // ... other error types
            _ => ErrorContextInfo {
                component: "Unknown".to_string(),
                operation: "Unknown".to_string(),
                input_info: None,
                system_state: SystemState::current(),
                timestamp: SystemTime::now(),
            },
        }
    }

    fn is_recoverable(&self) -> bool {
        match self {
            InferenceError::Configuration(_) => false,
            InferenceError::Model(ModelError::FileNotFound { .. }) => false,
            InferenceError::Memory(MemoryError::OutOfMemory { .. }) => true, // May recover with smaller batch
            InferenceError::Device(DeviceError::DeviceNotAvailable { .. }) => true, // Can fallback to CPU
            InferenceError::ForwardPass(ForwardPassError::NumericalInstability { .. }) => true,
            _ => true,
        }
    }

    fn recovery_suggestions(&self) -> Vec<RecoveryAction> {
        match self {
            InferenceError::Memory(MemoryError::OutOfMemory { .. }) => vec![
                RecoveryAction::ReduceMemoryUsage,
                RecoveryAction::ClearCaches,
                RecoveryAction::FallbackToCPU,
            ],
            InferenceError::Device(DeviceError::DeviceNotAvailable { .. }) => vec![
                RecoveryAction::FallbackToCPU,
            ],
            InferenceError::ForwardPass(ForwardPassError::NumericalInstability { .. }) => vec![
                RecoveryAction::RetryWithDifferentConfig,
                RecoveryAction::RestartModel,
            ],
            _ => vec![RecoveryAction::Retry],
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            InferenceError::Configuration(_) => ErrorSeverity::High,
            InferenceError::Model(_) => ErrorSeverity::High,
            InferenceError::Memory(MemoryError::OutOfMemory { .. }) => ErrorSeverity::Medium,
            InferenceError::Device(_) => ErrorSeverity::Medium,
            InferenceError::ForwardPass(ForwardPassError::NumericalInstability { .. }) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}
```

## Input Validation Contracts

### Comprehensive Input Validation

```rust
/// Input validation trait for all inference operations
pub trait InputValidator {
    /// Validate tensor inputs before processing
    fn validate_input_tensor(&self, tensor: &Tensor, expected_shape: &[usize]) -> Result<()>;

    /// Validate token sequences
    fn validate_token_sequence(&self, tokens: &[u32], vocab_size: usize) -> Result<()>;

    /// Validate generation configuration
    fn validate_generation_config(&self, config: &GenerationConfig) -> Result<()>;

    /// Validate memory requirements
    fn validate_memory_requirements(&self, required_memory: usize) -> Result<()>;
}

/// Standard input validator implementation
pub struct StandardInputValidator {
    config: ModelConfig,
    device: Device,
}

impl InputValidator for StandardInputValidator {
    fn validate_input_tensor(&self, tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
        let actual_shape = tensor.dims();

        // Check tensor rank
        if actual_shape.len() != expected_shape.len() {
            return Err(InferenceError::ForwardPass(ForwardPassError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: actual_shape.to_vec(),
            }));
        }

        // Check each dimension (allow flexible batch/sequence dimensions)
        for (i, (&expected, &actual)) in expected_shape.iter().zip(actual_shape.iter()).enumerate() {
            if expected != 0 && expected != actual {
                // 0 means flexible dimension
                return Err(InferenceError::ForwardPass(ForwardPassError::ShapeMismatch {
                    expected: expected_shape.to_vec(),
                    actual: actual_shape.to_vec(),
                }));
            }
        }

        // Check device compatibility
        if tensor.device() != &self.device {
            return Err(InferenceError::Device(DeviceError::DeviceMemoryError {
                device: format!("{:?}", self.device),
                details: format!("Input tensor on {:?}, expected {:?}", tensor.device(), self.device),
            }));
        }

        // Check data type
        if !matches!(tensor.dtype(), DType::F32 | DType::U32 | DType::I64) {
            return Err(InferenceError::ForwardPass(ForwardPassError::InvalidInput {
                reason: format!("Unsupported tensor dtype: {:?}", tensor.dtype()),
            }));
        }

        Ok(())
    }

    fn validate_token_sequence(&self, tokens: &[u32], vocab_size: usize) -> Result<()> {
        if tokens.is_empty() {
            return Err(InferenceError::Tokenization(TokenizationError::EmptySequence));
        }

        // Check sequence length limits
        if tokens.len() > self.config.max_position_embeddings {
            return Err(InferenceError::ForwardPass(ForwardPassError::InvalidInput {
                reason: format!(
                    "Sequence too long: {} tokens, maximum supported: {}",
                    tokens.len(),
                    self.config.max_position_embeddings
                ),
            }));
        }

        // Check token IDs are within vocabulary
        for (i, &token) in tokens.iter().enumerate() {
            if token as usize >= vocab_size {
                return Err(InferenceError::Tokenization(TokenizationError::InvalidToken {
                    position: i,
                    token_id: token,
                    vocab_size: vocab_size as u32,
                }));
            }
        }

        Ok(())
    }

    fn validate_generation_config(&self, config: &GenerationConfig) -> Result<()> {
        // Check token limits
        if config.max_new_tokens == 0 {
            return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                parameter: "max_new_tokens".to_string(),
                value: config.max_new_tokens.to_string(),
                valid_range: "1 to usize::MAX".to_string(),
            }));
        }

        if config.min_new_tokens > config.max_new_tokens {
            return Err(InferenceError::Configuration(ConfigurationError::IncompatibleConfig {
                reason: format!(
                    "min_new_tokens ({}) cannot be greater than max_new_tokens ({})",
                    config.min_new_tokens, config.max_new_tokens
                ),
            }));
        }

        // Check sampling parameters
        if config.temperature < 0.0 {
            return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                parameter: "temperature".to_string(),
                value: config.temperature.to_string(),
                valid_range: "0.0 to positive float".to_string(),
            }));
        }

        if let Some(top_p) = config.top_p {
            if top_p <= 0.0 || top_p > 1.0 {
                return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                    parameter: "top_p".to_string(),
                    value: top_p.to_string(),
                    valid_range: "0.0 < top_p <= 1.0".to_string(),
                }));
            }
        }

        if let Some(top_k) = config.top_k {
            if top_k == 0 {
                return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                    parameter: "top_k".to_string(),
                    value: top_k.to_string(),
                    valid_range: "1 to usize::MAX".to_string(),
                }));
            }
        }

        // Check penalty parameters
        if config.repetition_penalty <= 0.0 {
            return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                parameter: "repetition_penalty".to_string(),
                value: config.repetition_penalty.to_string(),
                valid_range: "positive float".to_string(),
            }));
        }

        // Check stop token validity
        for &stop_token in &config.stop_tokens {
            if stop_token as usize >= self.config.vocab_size {
                return Err(InferenceError::Configuration(ConfigurationError::InvalidParameter {
                    parameter: "stop_tokens".to_string(),
                    value: stop_token.to_string(),
                    valid_range: format!("0 to {}", self.config.vocab_size - 1),
                }));
            }
        }

        Ok(())
    }

    fn validate_memory_requirements(&self, required_memory: usize) -> Result<()> {
        let available_memory = match self.device {
            Device::Cpu => get_available_cpu_memory()?,
            Device::Cuda(_) => get_available_gpu_memory()?,
        };

        if required_memory > available_memory {
            return Err(InferenceError::Memory(MemoryError::OutOfMemory {
                requested_mb: required_memory / 1024 / 1024,
                available_mb: available_memory / 1024 / 1024,
            }));
        }

        Ok(())
    }
}
```

## Thread Safety and Concurrency Contracts

### Safe Concurrent Access

```rust
/// Thread-safe inference trait for concurrent access
pub trait ThreadSafeInference: BitNetInference + Send + Sync {
    /// Generate text with thread-safe access
    /// Multiple threads can call this simultaneously without data races
    fn generate_concurrent(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        config: &GenerationConfig,
    ) -> Result<GenerationResult>;

    /// Get a thread-local copy for exclusive access
    /// Useful for maintaining separate KV caches per thread
    fn thread_local_copy(&self) -> Result<Box<dyn BitNetInference>>;

    /// Check if the implementation supports concurrent generation
    fn supports_concurrent_generation(&self) -> bool;
}

/// Thread-safe wrapper around BitNet inference
pub struct ConcurrentInferenceEngine {
    inner: Arc<Mutex<Box<dyn BitNetInference>>>,
    config: ModelConfig,
}

impl ConcurrentInferenceEngine {
    pub fn new(inference: Box<dyn BitNetInference>) -> Self {
        let config = inference.config().clone();
        Self {
            inner: Arc::new(Mutex::new(inference)),
            config,
        }
    }

    /// Create multiple worker instances for parallel processing
    pub fn create_workers(&self, num_workers: usize) -> Result<Vec<Box<dyn BitNetInference>>> {
        let mut workers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = self.inner.lock().unwrap().thread_local_copy()?;
            workers.push(worker);
        }

        Ok(workers)
    }
}

impl ThreadSafeInference for ConcurrentInferenceEngine {
    fn generate_concurrent(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let mut inference = self.inner.lock().unwrap();
        inference.generate(prompt_tokens, max_tokens, config)
    }

    fn thread_local_copy(&self) -> Result<Box<dyn BitNetInference>> {
        let inference = self.inner.lock().unwrap();
        inference.thread_local_copy()
    }

    fn supports_concurrent_generation(&self) -> bool {
        true // Wrapper always supports concurrency through mutex
    }
}
```

## Performance Contracts

### Service Level Agreements (SLAs)

```rust
/// Performance contract enforcement
pub struct PerformanceContract {
    /// Maximum acceptable latency per token (milliseconds)
    pub max_token_latency_ms: u32,

    /// Minimum acceptable throughput (tokens per second)
    pub min_throughput_tps: f32,

    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,

    /// Minimum accuracy correlation with reference
    pub min_accuracy_correlation: f32,

    /// Performance monitoring configuration
    pub monitoring_enabled: bool,
    pub alert_on_violation: bool,
}

impl Default for PerformanceContract {
    fn default() -> Self {
        Self {
            max_token_latency_ms: 100,     // 100ms per token (CPU)
            min_throughput_tps: 5.0,       // 5 tokens per second minimum
            max_memory_usage: 8 * 1024 * 1024 * 1024, // 8GB maximum
            min_accuracy_correlation: 0.99, // 99% correlation minimum
            monitoring_enabled: true,
            alert_on_violation: true,
        }
    }
}

/// Performance monitoring and enforcement
pub struct PerformanceMonitor {
    contract: PerformanceContract,
    metrics_history: VecDeque<ForwardPassMetrics>,
    violations: Vec<PerformanceViolation>,
}

impl PerformanceMonitor {
    pub fn new(contract: PerformanceContract) -> Self {
        Self {
            contract,
            metrics_history: VecDeque::with_capacity(1000),
            violations: Vec::new(),
        }
    }

    /// Check if performance meets contract requirements
    pub fn validate_performance(&mut self, metrics: &ForwardPassMetrics, num_tokens: usize) -> Result<()> {
        self.metrics_history.push_back(metrics.clone());

        // Check latency contract
        let avg_token_latency = metrics.total_time.as_millis() as u32 / num_tokens.max(1) as u32;
        if avg_token_latency > self.contract.max_token_latency_ms {
            let violation = PerformanceViolation {
                violation_type: ViolationType::LatencyExceeded,
                actual_value: avg_token_latency as f64,
                expected_value: self.contract.max_token_latency_ms as f64,
                timestamp: SystemTime::now(),
            };
            self.violations.push(violation.clone());

            if self.contract.alert_on_violation {
                return Err(InferenceError::ForwardPass(ForwardPassError::ComputationFailed {
                    component: "PerformanceMonitor".to_string(),
                    reason: format!("Latency SLA violated: {}ms > {}ms", avg_token_latency, self.contract.max_token_latency_ms),
                }));
            }
        }

        // Check throughput contract
        let throughput = metrics.throughput(num_tokens);
        if throughput < self.contract.min_throughput_tps {
            let violation = PerformanceViolation {
                violation_type: ViolationType::ThroughputTooLow,
                actual_value: throughput as f64,
                expected_value: self.contract.min_throughput_tps as f64,
                timestamp: SystemTime::now(),
            };
            self.violations.push(violation.clone());

            if self.contract.alert_on_violation {
                return Err(InferenceError::ForwardPass(ForwardPassError::ComputationFailed {
                    component: "PerformanceMonitor".to_string(),
                    reason: format!("Throughput SLA violated: {:.2} tps < {:.2} tps", throughput, self.contract.min_throughput_tps),
                }));
            }
        }

        // Check memory contract
        if metrics.peak_memory_usage > self.contract.max_memory_usage {
            let violation = PerformanceViolation {
                violation_type: ViolationType::MemoryExceeded,
                actual_value: metrics.peak_memory_usage as f64,
                expected_value: self.contract.max_memory_usage as f64,
                timestamp: SystemTime::now(),
            };
            self.violations.push(violation.clone());

            if self.contract.alert_on_violation {
                return Err(InferenceError::Memory(MemoryError::OutOfMemory {
                    requested_mb: metrics.peak_memory_usage / 1024 / 1024,
                    available_mb: self.contract.max_memory_usage / 1024 / 1024,
                }));
            }
        }

        Ok(())
    }

    /// Get performance statistics over time window
    pub fn get_performance_stats(&self, window: Duration) -> PerformanceStats {
        let cutoff_time = SystemTime::now() - window;

        let recent_violations: Vec<_> = self.violations.iter()
            .filter(|v| v.timestamp > cutoff_time)
            .cloned()
            .collect();

        let recent_metrics: Vec<_> = self.metrics_history.iter()
            .rev()
            .take(100) // Last 100 measurements
            .collect();

        PerformanceStats {
            avg_latency_ms: recent_metrics.iter().map(|m| m.total_time.as_millis() as f64).sum::<f64>() / recent_metrics.len() as f64,
            avg_throughput_tps: recent_metrics.iter().map(|m| m.throughput(1) as f64).sum::<f64>() / recent_metrics.len() as f64,
            violation_count: recent_violations.len(),
            sla_compliance_rate: 1.0 - (recent_violations.len() as f64 / recent_metrics.len() as f64),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceViolation {
    pub violation_type: ViolationType,
    pub actual_value: f64,
    pub expected_value: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub enum ViolationType {
    LatencyExceeded,
    ThroughputTooLow,
    MemoryExceeded,
    AccuracyTooLow,
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub avg_latency_ms: f64,
    pub avg_throughput_tps: f64,
    pub violation_count: usize,
    pub sla_compliance_rate: f64,
}
```

## Integration Testing Contracts

### API Contract Validation Tests

```rust
#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn test_inference_api_contract() { // AC:1, AC:10
        let mut inference = create_test_inference_engine().unwrap();
        let input_ids = create_test_tensor([1, 10], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();

        // Test forward pass contract
        let output = inference.forward(&input_ids, None).unwrap();

        // Validate output contract
        output.validate(1, 10, inference.config().vocab_size).unwrap();

        // Check performance contract
        assert!(output.metrics.total_time.as_millis() < 1000); // Should be fast for test
        assert!(output.metrics.throughput(10) > 1.0); // Should have reasonable throughput
    }

    #[test]
    fn test_error_context_contract() { // AC:10
        let error = InferenceError::Configuration(ConfigurationError::InvalidParameter {
            parameter: "test_param".to_string(),
            value: "invalid_value".to_string(),
            valid_range: "valid_range".to_string(),
        });

        let context = error.context();
        assert_eq!(context.component, "Configuration");
        assert!(!error.is_recoverable()); // Config errors should not be recoverable
        assert_eq!(error.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_deterministic_generation_contract() { // AC:7
        let mut inference = create_test_inference_engine().unwrap();
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let seed = 12345;

        let result1 = inference.generate_deterministic(&prompt_tokens, 10, seed).unwrap();
        inference.reset().unwrap(); // Reset state
        let result2 = inference.generate_deterministic(&prompt_tokens, 10, seed).unwrap();

        // Contract: identical results for same seed
        assert_eq!(result1.generated_tokens, result2.generated_tokens);
        assert_eq!(result1.generated_text, result2.generated_text);
    }

    #[test]
    fn test_performance_contract_enforcement() { // AC:5
        let contract = PerformanceContract {
            max_token_latency_ms: 10, // Very strict for testing
            min_throughput_tps: 100.0, // Very high for testing
            ..Default::default()
        };

        let mut monitor = PerformanceMonitor::new(contract);
        let slow_metrics = ForwardPassMetrics {
            total_time: Duration::from_millis(500), // 50ms per token for 10 tokens
            ..Default::default()
        };

        // Should fail performance contract
        let result = monitor.validate_performance(&slow_metrics, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_input_validation_contract() { // AC:10
        let validator = StandardInputValidator {
            config: create_test_model_config(),
            device: Device::Cpu,
        };

        // Test invalid token sequence
        let invalid_tokens = vec![999999]; // Token ID beyond vocabulary
        let result = validator.validate_token_sequence(&invalid_tokens, 1000);
        assert!(result.is_err());

        // Test invalid tensor shape
        let wrong_shape_tensor = create_test_tensor([1, 5], vec![1, 2, 3, 4, 5]).unwrap();
        let result = validator.validate_input_tensor(&wrong_shape_tensor, &[1, 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_contract_compliance() { // AC:5
        let config = create_test_model_config();
        let max_memory = config.quantized_model_size() * 2; // 2x model size should be reasonable

        // Simulate memory usage
        let memory_usage = MemoryUsage {
            peak_usage: max_memory / 2, // Within limits
            model_weights: config.quantized_model_size(),
            kv_cache: max_memory / 8,
            activations: max_memory / 16,
            temporary_buffers: max_memory / 32,
            ..Default::default()
        };

        assert!(memory_usage.within_limits(max_memory));
        assert!(memory_usage.total_footprint() <= max_memory);
    }

    #[test]
    fn test_concurrent_access_contract() { // Thread safety
        let inference = create_test_inference_engine().unwrap();
        let concurrent_engine = ConcurrentInferenceEngine::new(inference);

        // Test concurrent generation
        let handles: Vec<_> = (0..4).map(|i| {
            let engine = concurrent_engine.clone();
            let tokens = vec![1, 2, 3, i as u32 + 10]; // Different prompts

            std::thread::spawn(move || {
                engine.generate_concurrent(&tokens, 5, &GenerationConfig::default())
            })
        }).collect();

        // All threads should complete successfully
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    }

    // Helper functions for testing
    fn create_test_inference_engine() -> Result<Box<dyn BitNetInference>> {
        // Implementation would create a test inference engine
        todo!("Create test inference engine")
    }

    fn create_test_model_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 512,
            num_layers: 6,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            quantization_type: QuantizationType::I2S,
            quantization_accuracy: 0.99,
            model_type: "bitnet".to_string(),
            model_name: Some("test-model".to_string()),
            version: "1.0".to_string(),
        }
    }

    fn create_test_tensor(shape: [usize; 2], data: Vec<u32>) -> Result<Tensor> {
        Tensor::from_vec(data, &shape, &Device::Cpu)
    }
}
```

These comprehensive API contracts ensure robust, consistent, and high-performance neural network inference across the entire BitNet.rs ecosystem, with strong error handling, input validation, and performance guarantees.
