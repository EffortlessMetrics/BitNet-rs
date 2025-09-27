# GGUF Weight Loading API Contracts

## Overview

This document defines the comprehensive API contracts for GGUF model weight loading in BitNet.rs. These contracts ensure type safety, error handling, and device-aware operations for loading quantized neural network weights from GGUF files.

## Core API Contracts

### Primary Loading Interface

```rust
/// Enhanced GGUF weight loader with comprehensive parsing capabilities
pub struct GgufWeightLoader {
    /// Configuration for quantization handling
    pub quantization_config: QuantizationConfig,
    /// Device placement strategy
    pub device_placement: DevicePlacement,
    /// Memory optimization settings
    pub memory_config: MemoryConfig,
    /// Cross-validation settings
    pub validation_config: ValidationConfig,
}

/// Configuration for quantization operations
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Enable I2_S quantization support
    pub enable_i2s: bool,
    /// Enable TL1 quantization support
    pub enable_tl1: bool,
    /// Enable TL2 quantization support
    pub enable_tl2: bool,
    /// Accuracy threshold for cross-validation (default: 0.99)
    pub accuracy_threshold: f32,
    /// Block size for quantization operations
    pub block_size: usize,
}

/// Device placement and memory management strategy
#[derive(Debug, Clone)]
pub struct DevicePlacement {
    /// Target device for tensor placement
    pub target_device: Device,
    /// Enable automatic GPU fallback to CPU
    pub enable_cpu_fallback: bool,
    /// Maximum GPU memory usage percentage (0.0-1.0)
    pub max_gpu_memory_usage: f32,
    /// Enable mixed precision (FP16/BF16) when available
    pub enable_mixed_precision: bool,
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable zero-copy operations when possible
    pub enable_zero_copy: bool,
    /// Enable progressive loading for large models
    pub enable_progressive_loading: bool,
    /// Memory usage limit (bytes)
    pub memory_limit: usize,
    /// Alignment requirement for tensors (bytes)
    pub tensor_alignment: usize,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable cross-validation against C++ reference
    pub enable_cpp_crossval: bool,
    /// Enable deterministic validation
    pub enable_deterministic_validation: bool,
    /// Numerical tolerance for comparisons
    pub numerical_tolerance: f64,
    /// Random seed for deterministic validation
    pub validation_seed: Option<u64>,
}
```

### Core Loading Methods

```rust
impl GgufWeightLoader {
    /// Create a new weight loader with default configuration
    pub fn new() -> Self {
        Self {
            quantization_config: QuantizationConfig::default(),
            device_placement: DevicePlacement::default(),
            memory_config: MemoryConfig::default(),
            validation_config: ValidationConfig::default(),
        }
    }

    /// Create weight loader with custom configuration
    pub fn with_config(
        quantization: QuantizationConfig,
        device: DevicePlacement,
        memory: MemoryConfig,
        validation: ValidationConfig,
    ) -> Self {
        Self {
            quantization_config: quantization,
            device_placement: device,
            memory_config: memory,
            validation_config: validation,
        }
    }

    /// Load complete model with all transformer weights
    ///
    /// # Arguments
    /// * `path` - Path to GGUF file
    /// * `device` - Target device for tensor placement
    ///
    /// # Returns
    /// * `Ok((config, weights))` - Model configuration and weight tensors
    /// * `Err(WeightLoadingError)` - Loading failure with detailed error information
    ///
    /// # Acceptance Criteria Coverage
    /// * AC1: Parse and load all transformer layer weights
    /// * AC2: Support quantization formats with ≥99% accuracy
    /// * AC6: Support CPU/GPU feature flags with device-aware placement
    /// * AC7: Memory-efficient loading with zero-copy operations
    pub fn load_complete_model(
        &self,
        path: &Path,
        device: Device,
    ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>), WeightLoadingError> {
        self.load_complete_model_with_progress(path, device, None)
    }

    /// Load complete model with progress reporting
    pub fn load_complete_model_with_progress(
        &self,
        path: &Path,
        device: Device,
        progress_callback: Option<Box<dyn Fn(LoadingProgress) + Send + Sync>>,
    ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>), WeightLoadingError> {
        // Implementation details in main specification
    }

    /// Validate loaded weights against expected schema
    ///
    /// # Acceptance Criteria Coverage
    /// * AC3: Tensor metadata validation including shape verification
    /// * AC4: Descriptive error messages for validation failures
    pub fn validate_weights(
        &self,
        weights: &HashMap<String, CandleTensor>,
        config: &BitNetConfig,
    ) -> Result<ValidationReport, WeightLoadingError> {
        // Implementation details in main specification
    }

    /// Load specific tensor by name
    pub fn load_tensor(
        &self,
        path: &Path,
        tensor_name: &str,
        device: Device,
    ) -> Result<CandleTensor, WeightLoadingError> {
        // Implementation for individual tensor loading
    }

    /// Load model with backward compatibility (mock fallback)
    ///
    /// # Acceptance Criteria Coverage
    /// * AC9: Maintain backward compatibility with mock tensor loading
    pub fn load_with_mock_fallback(
        &self,
        path: &Path,
        device: Device,
        enable_mock: bool,
    ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>), WeightLoadingError> {
        // Implementation with mock fallback support
    }
}
```

## Error Handling Contracts

### Comprehensive Error Types

```rust
/// Comprehensive error types for GGUF weight loading
#[derive(Debug, thiserror::Error)]
pub enum WeightLoadingError {
    #[error("IO error while reading GGUF file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("GGUF parsing error: {message}")]
    GgufParsingError { message: String },

    #[error("Tensor '{name}' not found in GGUF file")]
    TensorNotFound { name: String },

    #[error("Tensor '{name}' has invalid shape: expected {expected:?}, got {actual:?}")]
    InvalidTensorShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Tensor '{name}' has invalid data type: expected {expected}, got {actual}")]
    InvalidTensorDataType {
        name: String,
        expected: String,
        actual: String,
    },

    #[error("Unsupported quantization type {qtype} for tensor '{name}'. Supported types: {supported:?}")]
    UnsupportedQuantization {
        name: String,
        qtype: String,
        supported: Vec<String>,
    },

    #[error("Quantization accuracy below threshold: {accuracy:.4}% < {required:.4}% for tensor '{name}'")]
    QuantizationAccuracyError {
        name: String,
        accuracy: f32,
        required: f32,
    },

    #[error("Memory allocation failed: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Device error: {message}")]
    DeviceError { message: String },

    #[error("Cross-validation failed: {details}")]
    CrossValidationError { details: String },

    #[error("Tensor alignment error: tensor '{name}' at offset {offset} not aligned to {alignment} bytes")]
    AlignmentError {
        name: String,
        offset: usize,
        alignment: usize,
    },

    #[error("Model architecture mismatch: {details}")]
    ArchitectureMismatch { details: String },

    #[error("Security validation failed: {reason}")]
    SecurityError { reason: String },
}

impl WeightLoadingError {
    /// Get error category for structured error handling
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::IoError(_) => ErrorCategory::Io,
            Self::GgufParsingError { .. } => ErrorCategory::Parsing,
            Self::TensorNotFound { .. } | Self::InvalidTensorShape { .. } => ErrorCategory::Schema,
            Self::UnsupportedQuantization { .. } | Self::QuantizationAccuracyError { .. } => {
                ErrorCategory::Quantization
            }
            Self::OutOfMemory { .. } => ErrorCategory::Memory,
            Self::DeviceError { .. } => ErrorCategory::Device,
            Self::CrossValidationError { .. } => ErrorCategory::Validation,
            Self::AlignmentError { .. } => ErrorCategory::Alignment,
            Self::ArchitectureMismatch { .. } => ErrorCategory::Architecture,
            Self::SecurityError { .. } => ErrorCategory::Security,
        }
    }

    /// Check if error is recoverable with fallback strategy
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::DeviceError { .. } => true,  // Can fallback to CPU
            Self::OutOfMemory { .. } => true,  // Can use progressive loading
            Self::UnsupportedQuantization { .. } => true,  // Can fallback to FP32
            _ => false,
        }
    }

    /// Get suggested recovery action
    pub fn recovery_suggestion(&self) -> Option<String> {
        match self {
            Self::DeviceError { .. } => Some("Try loading with CPU device".to_string()),
            Self::OutOfMemory { .. } => Some("Enable progressive loading or increase memory limit".to_string()),
            Self::UnsupportedQuantization { .. } => Some("Disable quantization or convert model to supported format".to_string()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    Io,
    Parsing,
    Schema,
    Quantization,
    Memory,
    Device,
    Validation,
    Alignment,
    Architecture,
    Security,
}
```

## Tensor Schema Contracts

### Model Architecture Schema

```rust
/// Comprehensive tensor naming and validation schema
#[derive(Debug, Clone)]
pub struct TensorSchema {
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Attention layer specifications
    pub attention_layers: AttentionLayerSchema,
    /// Feed-forward network specifications
    pub feedforward_layers: FeedforwardLayerSchema,
    /// Normalization layer specifications
    pub normalization_layers: NormalizationLayerSchema,
    /// Embedding layer specifications
    pub embedding_layers: EmbeddingLayerSchema,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    BitNet,
    LLaMA,
    GPT,
    Custom(String),
}

/// Attention layer tensor specifications
#[derive(Debug, Clone)]
pub struct AttentionLayerSchema {
    /// Query weight tensor specification
    pub query_weight: TensorSpec,
    /// Key weight tensor specification
    pub key_weight: TensorSpec,
    /// Value weight tensor specification
    pub value_weight: TensorSpec,
    /// Output projection weight specification
    pub output_weight: TensorSpec,
    /// Optional bias tensors
    pub query_bias: Option<TensorSpec>,
    pub key_bias: Option<TensorSpec>,
    pub value_bias: Option<TensorSpec>,
    pub output_bias: Option<TensorSpec>,
}

/// Feed-forward network tensor specifications
#[derive(Debug, Clone)]
pub struct FeedforwardLayerSchema {
    /// Gate projection (or up projection)
    pub gate_weight: TensorSpec,
    /// Up projection (for SwiGLU activation)
    pub up_weight: Option<TensorSpec>,
    /// Down projection
    pub down_weight: TensorSpec,
    /// Optional bias tensors
    pub gate_bias: Option<TensorSpec>,
    pub up_bias: Option<TensorSpec>,
    pub down_bias: Option<TensorSpec>,
}

/// Normalization layer specifications
#[derive(Debug, Clone)]
pub struct NormalizationLayerSchema {
    /// Layer normalization weights
    pub weight: TensorSpec,
    /// Optional bias (some models don't use bias in LayerNorm)
    pub bias: Option<TensorSpec>,
}

/// Embedding layer specifications
#[derive(Debug, Clone)]
pub struct EmbeddingLayerSchema {
    /// Token embedding matrix
    pub token_embeddings: TensorSpec,
    /// Output projection (language model head)
    pub output_projection: TensorSpec,
    /// Position embeddings (if used)
    pub position_embeddings: Option<TensorSpec>,
}

/// Individual tensor specification
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name patterns (supports multiple naming conventions)
    pub name_patterns: Vec<String>,
    /// Expected tensor shape (None means dynamic/inferred)
    pub expected_shape: Option<Vec<Option<usize>>>,
    /// Supported data types
    pub supported_dtypes: Vec<TensorDType>,
    /// Whether tensor is required or optional
    pub required: bool,
    /// Validation constraints
    pub constraints: TensorConstraints,
}

#[derive(Debug, Clone)]
pub struct TensorConstraints {
    /// Minimum and maximum values for tensor elements
    pub value_range: Option<(f32, f32)>,
    /// Expected sparsity (percentage of zero elements)
    pub expected_sparsity: Option<f32>,
    /// Quantization requirements
    pub quantization: QuantizationConstraints,
}

#[derive(Debug, Clone)]
pub struct QuantizationConstraints {
    /// Supported quantization types
    pub supported_qtypes: Vec<QuantizationType>,
    /// Minimum accuracy requirement
    pub min_accuracy: f32,
    /// Block size constraints
    pub block_size_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    I2S,
    TL1,
    TL2,
    IQ2S,
}
```

### Schema Validation Interface

```rust
/// Schema validation implementation
impl TensorSchema {
    /// Create schema for BitNet architecture
    pub fn bitnet_schema(config: &BitNetConfig) -> Self {
        Self {
            architecture: ModelArchitecture::BitNet,
            attention_layers: Self::bitnet_attention_schema(config),
            feedforward_layers: Self::bitnet_feedforward_schema(config),
            normalization_layers: Self::bitnet_normalization_schema(config),
            embedding_layers: Self::bitnet_embedding_schema(config),
        }
    }

    /// Create schema for LLaMA architecture
    pub fn llama_schema(config: &BitNetConfig) -> Self {
        Self {
            architecture: ModelArchitecture::LLaMA,
            attention_layers: Self::llama_attention_schema(config),
            feedforward_layers: Self::llama_feedforward_schema(config),
            normalization_layers: Self::llama_normalization_schema(config),
            embedding_layers: Self::llama_embedding_schema(config),
        }
    }

    /// Validate tensor against schema
    ///
    /// # Acceptance Criteria Coverage
    /// * AC3: Tensor metadata validation including shape verification
    /// * AC10: Document tensor naming conventions and shape expectations
    pub fn validate_tensor(
        &self,
        tensor_name: &str,
        tensor: &CandleTensor,
        layer_index: Option<usize>,
    ) -> Result<(), TensorValidationError> {
        let spec = self.find_tensor_spec(tensor_name, layer_index)
            .ok_or_else(|| TensorValidationError::UnknownTensor {
                name: tensor_name.to_string(),
                architecture: self.architecture.clone(),
            })?;

        // Validate tensor shape
        if let Some(expected_shape) = &spec.expected_shape {
            self.validate_tensor_shape(tensor_name, tensor.shape(), expected_shape)?;
        }

        // Validate data type
        let actual_dtype = TensorDType::from_candle_dtype(tensor.dtype());
        if !spec.supported_dtypes.contains(&actual_dtype) {
            return Err(TensorValidationError::UnsupportedDataType {
                name: tensor_name.to_string(),
                actual: actual_dtype,
                supported: spec.supported_dtypes.clone(),
            });
        }

        // Validate constraints
        self.validate_tensor_constraints(tensor_name, tensor, &spec.constraints)?;

        Ok(())
    }

    /// Find tensor specification by name and layer index
    fn find_tensor_spec(&self, name: &str, layer_index: Option<usize>) -> Option<&TensorSpec> {
        // Implementation for tensor spec lookup based on naming patterns
    }

    /// Validate tensor shape against specification
    fn validate_tensor_shape(
        &self,
        name: &str,
        actual_shape: &[usize],
        expected_shape: &[Option<usize>],
    ) -> Result<(), TensorValidationError> {
        if actual_shape.len() != expected_shape.len() {
            return Err(TensorValidationError::ShapeDimensionMismatch {
                name: name.to_string(),
                actual_dims: actual_shape.len(),
                expected_dims: expected_shape.len(),
            });
        }

        for (i, (&actual, expected)) in actual_shape.iter().zip(expected_shape.iter()).enumerate() {
            if let Some(expected_size) = expected {
                if actual != *expected_size {
                    return Err(TensorValidationError::ShapeSizeMismatch {
                        name: name.to_string(),
                        dimension: i,
                        actual: actual,
                        expected: *expected_size,
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate tensor constraints
    fn validate_tensor_constraints(
        &self,
        name: &str,
        tensor: &CandleTensor,
        constraints: &TensorConstraints,
    ) -> Result<(), TensorValidationError> {
        // Implementation for constraint validation
        Ok(())
    }
}
```

## Progress and Reporting Contracts

### Loading Progress Interface

```rust
/// Progress reporting for model loading operations
#[derive(Debug, Clone)]
pub struct LoadingProgress {
    /// Current loading phase
    pub phase: LoadingPhase,
    /// Overall progress (0.0 to 1.0)
    pub overall_progress: f32,
    /// Phase-specific progress (0.0 to 1.0)
    pub phase_progress: f32,
    /// Number of tensors loaded
    pub tensors_loaded: usize,
    /// Total number of tensors to load
    pub total_tensors: usize,
    /// Bytes loaded
    pub bytes_loaded: u64,
    /// Total bytes to load
    pub total_bytes: u64,
    /// Current operation description
    pub current_operation: String,
    /// Estimated time remaining (if available)
    pub eta_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadingPhase {
    Initializing,
    ParsingHeader,
    ValidatingSchema,
    LoadingEmbeddings,
    LoadingAttentionLayers,
    LoadingFeedforwardLayers,
    LoadingNormalizationLayers,
    DequantizingWeights,
    ValidatingWeights,
    CrossValidating,
    Finalizing,
    Complete,
}

impl LoadingProgress {
    /// Create initial progress state
    pub fn new() -> Self {
        Self {
            phase: LoadingPhase::Initializing,
            overall_progress: 0.0,
            phase_progress: 0.0,
            tensors_loaded: 0,
            total_tensors: 0,
            bytes_loaded: 0,
            total_bytes: 0,
            current_operation: "Initializing".to_string(),
            eta_seconds: None,
        }
    }

    /// Update progress with new values
    pub fn update(&mut self,
                  phase: LoadingPhase,
                  phase_progress: f32,
                  current_operation: String) {
        self.phase = phase;
        self.phase_progress = phase_progress;
        self.current_operation = current_operation;
        self.update_overall_progress();
    }

    /// Calculate overall progress based on phase
    fn update_overall_progress(&mut self) {
        let phase_weight = match self.phase {
            LoadingPhase::Initializing => 0.02,
            LoadingPhase::ParsingHeader => 0.05,
            LoadingPhase::ValidatingSchema => 0.08,
            LoadingPhase::LoadingEmbeddings => 0.15,
            LoadingPhase::LoadingAttentionLayers => 0.30,
            LoadingPhase::LoadingFeedforwardLayers => 0.25,
            LoadingPhase::LoadingNormalizationLayers => 0.05,
            LoadingPhase::DequantizingWeights => 0.15,
            LoadingPhase::ValidatingWeights => 0.10,
            LoadingPhase::CrossValidating => 0.08,
            LoadingPhase::Finalizing => 0.02,
            LoadingPhase::Complete => 1.0,
        };

        // Calculate cumulative progress for completed phases
        let completed_phases_progress: f32 = match self.phase {
            LoadingPhase::Initializing => 0.0,
            LoadingPhase::ParsingHeader => 0.02,
            LoadingPhase::ValidatingSchema => 0.07,
            LoadingPhase::LoadingEmbeddings => 0.15,
            LoadingPhase::LoadingAttentionLayers => 0.30,
            LoadingPhase::LoadingFeedforwardLayers => 0.60,
            LoadingPhase::LoadingNormalizationLayers => 0.85,
            LoadingPhase::DequantizingWeights => 0.90,
            LoadingPhase::ValidatingWeights => 1.05,
            LoadingPhase::CrossValidating => 1.15,
            LoadingPhase::Finalizing => 1.23,
            LoadingPhase::Complete => 1.0,
        };

        self.overall_progress = completed_phases_progress + (phase_weight * self.phase_progress);
        self.overall_progress = self.overall_progress.min(1.0);
    }
}
```

### Validation Reporting

```rust
/// Comprehensive validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Overall validation status
    pub status: ValidationStatus,
    /// Individual tensor validation results
    pub tensor_results: HashMap<String, TensorValidationResult>,
    /// Cross-validation results (if enabled)
    pub cross_validation: Option<CrossValidationResults>,
    /// Performance metrics
    pub performance_metrics: ValidationPerformanceMetrics,
    /// Summary statistics
    pub summary: ValidationSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationStatus {
    Passed,
    Failed,
    PartiallyPassed,
    Skipped,
}

#[derive(Debug, Clone)]
pub struct TensorValidationResult {
    /// Tensor name
    pub name: String,
    /// Validation status for this tensor
    pub status: ValidationStatus,
    /// Shape validation result
    pub shape_validation: ShapeValidationResult,
    /// Data type validation result
    pub dtype_validation: DTypeValidationResult,
    /// Quantization validation result (if applicable)
    pub quantization_validation: Option<QuantizationValidationResult>,
    /// Constraint validation results
    pub constraint_validation: Vec<ConstraintValidationResult>,
    /// Validation errors (if any)
    pub errors: Vec<TensorValidationError>,
}

#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// C++ reference comparison results
    pub cpp_reference_results: Option<CppReferenceResults>,
    /// Deterministic validation results
    pub deterministic_results: Option<DeterministicResults>,
    /// Overall cross-validation status
    pub status: ValidationStatus,
    /// Accuracy metrics
    pub accuracy_metrics: CrossValidationMetrics,
}

#[derive(Debug, Clone)]
pub struct ValidationPerformanceMetrics {
    /// Total validation time
    pub total_time_ms: u64,
    /// Time per validation phase
    pub phase_times: HashMap<String, u64>,
    /// Memory usage during validation
    pub peak_memory_usage: usize,
    /// Number of tensors validated per second
    pub tensors_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total number of tensors validated
    pub total_tensors: usize,
    /// Number of passed validations
    pub passed_count: usize,
    /// Number of failed validations
    pub failed_count: usize,
    /// Number of skipped validations
    pub skipped_count: usize,
    /// Overall pass rate
    pub pass_rate: f32,
    /// Critical errors that require attention
    pub critical_errors: Vec<String>,
    /// Warnings that should be addressed
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.status == ValidationStatus::Passed
    }

    /// Get all critical errors across tensors
    pub fn get_critical_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for result in self.tensor_results.values() {
            for error in &result.errors {
                if error.is_critical() {
                    errors.push(format!("Tensor '{}': {}", result.name, error));
                }
            }
        }

        errors.extend(self.summary.critical_errors.clone());
        errors
    }

    /// Generate human-readable validation summary
    pub fn generate_summary(&self) -> String {
        format!(
            "Validation Summary: {}/{} tensors passed ({:.1}% pass rate)\n\
             Status: {:?}\n\
             Total time: {}ms\n\
             Critical errors: {}\n\
             Warnings: {}",
            self.summary.passed_count,
            self.summary.total_tensors,
            self.summary.pass_rate * 100.0,
            self.status,
            self.performance_metrics.total_time_ms,
            self.summary.critical_errors.len(),
            self.summary.warnings.len()
        )
    }
}
```

## Feature Flag Integration

### Conditional Compilation Contracts

```rust
/// Feature-aware compilation and runtime behavior
impl GgufWeightLoader {
    /// Check if GPU support is available
    #[cfg(feature = "gpu")]
    pub fn gpu_support_available() -> bool {
        cfg!(feature = "cuda") || cfg!(feature = "metal")
    }

    #[cfg(not(feature = "gpu"))]
    pub fn gpu_support_available() -> bool {
        false
    }

    /// Check if CUDA support is specifically available
    #[cfg(feature = "cuda")]
    pub fn cuda_support_available() -> bool {
        bitnet_kernels::gpu::cuda::is_cuda_available()
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_support_available() -> bool {
        false
    }

    /// Load with automatic device selection based on available features
    ///
    /// # Acceptance Criteria Coverage
    /// * AC6: Support CPU/GPU feature flags with graceful fallback
    pub fn load_with_auto_device(
        &self,
        path: &Path,
    ) -> Result<(BitNetConfig, HashMap<String, CandleTensor>), WeightLoadingError> {
        let device = self.select_optimal_device()?;
        self.load_complete_model(path, device)
    }

    /// Select optimal device based on available features and configuration
    fn select_optimal_device(&self) -> Result<Device, WeightLoadingError> {
        #[cfg(feature = "cuda")]
        {
            if self.device_placement.target_device.is_cuda() && Self::cuda_support_available() {
                return Ok(self.device_placement.target_device.clone());
            }
        }

        #[cfg(feature = "metal")]
        {
            if self.device_placement.target_device.is_metal() {
                return Ok(self.device_placement.target_device.clone());
            }
        }

        // Fallback to CPU
        if self.device_placement.enable_cpu_fallback {
            tracing::info!("Falling back to CPU device");
            Ok(Device::Cpu)
        } else {
            Err(WeightLoadingError::DeviceError {
                message: "Requested device not available and CPU fallback disabled".to_string(),
            })
        }
    }
}
```

## Default Implementations and Builders

### Configuration Builders

```rust
/// Builder pattern for creating weight loader configurations
pub struct GgufWeightLoaderBuilder {
    quantization_config: Option<QuantizationConfig>,
    device_placement: Option<DevicePlacement>,
    memory_config: Option<MemoryConfig>,
    validation_config: Option<ValidationConfig>,
}

impl GgufWeightLoaderBuilder {
    pub fn new() -> Self {
        Self {
            quantization_config: None,
            device_placement: None,
            memory_config: None,
            validation_config: None,
        }
    }

    pub fn quantization(mut self, config: QuantizationConfig) -> Self {
        self.quantization_config = Some(config);
        self
    }

    pub fn device_placement(mut self, config: DevicePlacement) -> Self {
        self.device_placement = Some(config);
        self
    }

    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    pub fn validation_config(mut self, config: ValidationConfig) -> Self {
        self.validation_config = Some(config);
        self
    }

    pub fn build(self) -> GgufWeightLoader {
        GgufWeightLoader::with_config(
            self.quantization_config.unwrap_or_default(),
            self.device_placement.unwrap_or_default(),
            self.memory_config.unwrap_or_default(),
            self.validation_config.unwrap_or_default(),
        )
    }
}

/// Default implementations for configurations
impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enable_i2s: true,
            enable_tl1: true,
            enable_tl2: true,
            accuracy_threshold: 0.99,
            block_size: 32,
        }
    }
}

impl Default for DevicePlacement {
    fn default() -> Self {
        Self {
            target_device: Device::Cpu,
            enable_cpu_fallback: true,
            max_gpu_memory_usage: 0.8,
            enable_mixed_precision: false,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_zero_copy: true,
            enable_progressive_loading: false,
            memory_limit: usize::MAX,
            tensor_alignment: 32,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_cpp_crossval: std::env::var("BITNET_CROSSVAL_WEIGHTS").is_ok(),
            enable_deterministic_validation: std::env::var("BITNET_DETERMINISTIC").is_ok(),
            numerical_tolerance: 1e-5,
            validation_seed: std::env::var("BITNET_SEED")
                .ok()
                .and_then(|s| s.parse().ok()),
        }
    }
}
```

## Type Safety and Conversions

### Safe Type Conversions

```rust
/// Safe conversions between tensor types and formats
pub trait TensorConversion {
    /// Convert from Candle tensor to BitNet tensor
    fn from_candle_tensor(tensor: CandleTensor) -> Result<BitNetTensor, ConversionError>;

    /// Convert from BitNet tensor to Candle tensor
    fn to_candle_tensor(tensor: BitNetTensor) -> Result<CandleTensor, ConversionError>;
}

/// Conversion errors
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    #[error("Incompatible tensor shapes: source {source:?}, target {target:?}")]
    IncompatibleShapes { source: Vec<usize>, target: Vec<usize> },

    #[error("Unsupported data type conversion: {from} to {to}")]
    UnsupportedConversion { from: String, to: String },

    #[error("Device mismatch: tensor on {tensor_device}, expected {target_device}")]
    DeviceMismatch { tensor_device: String, target_device: String },
}

/// Utility functions for safe tensor operations
pub mod tensor_utils {
    use super::*;

    /// Safely extract f32 data from tensor with validation
    pub fn extract_f32_data_safe(tensor: &BitNetTensor) -> Result<Vec<f32>, WeightLoadingError> {
        // Implementation with bounds checking and type validation
    }

    /// Create tensor with proper device placement and validation
    pub fn create_tensor_safe(
        data: Vec<f32>,
        shape: &[usize],
        device: &Device,
    ) -> Result<CandleTensor, WeightLoadingError> {
        // Implementation with memory validation and device checks
    }

    /// Validate tensor dimensions against security limits
    pub fn validate_tensor_dimensions(shape: &[usize]) -> Result<(), WeightLoadingError> {
        // Implementation with security checks from bitnet-common
    }
}
```

This comprehensive API contract specification provides:

1. **Type-safe interfaces** with comprehensive error handling
2. **Device-aware operations** supporting CPU/GPU with automatic fallback
3. **Configuration builders** for flexible customization
4. **Progress reporting** for long-running operations
5. **Validation contracts** ensuring data integrity and schema compliance
6. **Feature flag integration** for conditional compilation
7. **Memory safety** with bounds checking and security validations

The contracts align with all 10 acceptance criteria from Issue #159 and provide a solid foundation for implementing real GGUF weight loading in BitNet.rs.