# Real BitNet Model Integration: API Contracts Reference

## Overview

This document defines the comprehensive API contracts for real BitNet model integration across the BitNet.rs neural network inference pipeline. These contracts ensure consistent interfaces for model loading, quantization, inference, and validation while maintaining backward compatibility and supporting device-aware execution.

## Core API Contracts

### Model Loading Interface (`bitnet-models`)

#### Primary Model Loading Contract

```rust
/// Enhanced model loader with real GGUF validation and device-aware capabilities
pub trait RealModelLoader {
    /// Load and validate a real BitNet model from GGUF file
    fn load_with_validation(&self, path: &Path) -> Result<BitNetModel, ModelError>;

    /// Validate GGUF format compatibility without loading full model
    fn validate_gguf_format(&self, path: &Path) -> ValidationResult;

    /// Extract tokenizer metadata from GGUF file headers
    fn extract_tokenizer_metadata(&self, path: &Path) -> Result<TokenizerConfig, ModelError>;

    /// Check device compatibility for model execution
    fn check_device_compatibility(&self, model_path: &Path, device: Device) -> CompatibilityResult;

    /// Get model metadata without full loading (for discovery)
    fn get_model_info(&self, path: &Path) -> Result<ModelInfo, ModelError>;
}

/// Production implementation of real model loader
pub struct ProductionModelLoader {
    pub validation_level: ValidationLevel,
    pub quantization_support: QuantizationSupport,
    pub device_preference: DevicePreference,
    pub memory_config: MemoryConfig,
}

impl ProductionModelLoader {
    /// Create new loader with production-grade validation
    pub fn new(config: LoaderConfig) -> Self;

    /// Create loader optimized for CI testing
    pub fn for_ci_testing() -> Self;

    /// Create loader with maximum validation for debugging
    pub fn with_strict_validation() -> Self;
}
```

#### Model Representation Contract

```rust
/// Real BitNet model with comprehensive metadata and device support
pub struct BitNetModel {
    /// Model metadata extracted from GGUF headers
    pub metadata: ModelMetadata,

    /// Tensor collection with quantization information
    pub tensors: TensorCollection,

    /// Quantization configuration and capabilities
    pub quantization_info: QuantizationInfo,

    /// Device configuration and optimization hints
    pub device_config: DeviceConfig,

    /// Tokenizer configuration extracted from model
    pub tokenizer_config: Option<TokenizerConfig>,
}

impl BitNetModel {
    /// Load model from GGUF file with full validation
    pub fn from_file(path: &Path) -> Result<Self, ModelError>;

    /// Load model with custom validation configuration
    pub fn from_file_with_config(path: &Path, config: LoaderConfig) -> Result<Self, ModelError>;

    /// Validate model format and tensor alignment
    pub fn validate_format(&self) -> ValidationResult;

    /// Get tokenizer configuration if available
    pub fn get_tokenizer_config(&self) -> Option<TokenizerConfig>;

    /// Check if model supports specific device
    pub fn supports_device(&self, device: Device) -> bool;

    /// Get optimal device configuration for this model
    pub fn get_optimal_device_config(&self) -> DeviceConfig;

    /// Get memory requirements for different devices
    pub fn get_memory_requirements(&self, device: Device) -> MemoryRequirements;
}
```

#### Error Handling Contract

```rust
/// Comprehensive error types for model loading operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    /// GGUF file format errors
    #[error("GGUF format error: {message}")]
    GGUFFormatError { message: String, details: GGUFErrorDetails },

    /// Tensor validation errors
    #[error("Tensor validation failed: {tensor_name} - {reason}")]
    TensorValidationError { tensor_name: String, reason: String },

    /// Quantization format not supported
    #[error("Unsupported quantization format: {format}")]
    UnsupportedQuantization { format: String },

    /// Device incompatibility
    #[error("Device incompatible: {device} - {reason}")]
    DeviceIncompatible { device: String, reason: String },

    /// Memory allocation errors
    #[error("Memory allocation failed: {size_mb}MB - {reason}")]
    MemoryAllocationError { size_mb: usize, reason: String },

    /// File I/O errors
    #[error("File I/O error: {path} - {error}")]
    FileIOError { path: PathBuf, error: std::io::Error },
}

/// Detailed validation results with actionable recommendations
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub recommendations: Vec<String>,
    pub metadata: ValidationMetadata,
}

impl ValidationResult {
    /// Create successful validation result
    pub fn success() -> Self;

    /// Create validation result with errors
    pub fn with_errors(errors: Vec<ValidationError>) -> Self;

    /// Add actionable recommendation for fixing issues
    pub fn add_recommendation(&mut self, recommendation: String);

    /// Get summary of validation issues
    pub fn get_summary(&self) -> ValidationSummary;
}
```

### Inference Engine Interface (`bitnet-inference`)

#### Production Inference Engine Contract

```rust
/// Production-grade inference engine with real model support
pub trait ProductionEngine {
    /// Perform inference with comprehensive metrics collection
    async fn infer_with_metrics(&mut self, prompt: &str) -> Result<InferenceResult, InferenceError>;

    /// Batch inference with performance optimization
    async fn infer_batch(&mut self, prompts: &[String]) -> Result<Vec<InferenceResult>, InferenceError>;

    /// Streaming inference with real-time token generation
    async fn infer_streaming(&mut self, prompt: &str) -> Result<TokenStream, InferenceError>;

    /// Validate inference output against reference implementation
    fn validate_against_reference(&self, inputs: &[TokenId], expected: &[TokenId]) -> ValidationResult;

    /// Get comprehensive performance metrics
    fn get_performance_metrics(&self) -> PerformanceMetrics;

    /// Benchmark performance with configurable parameters
    fn benchmark_performance(&mut self, config: BenchmarkConfig) -> PerformanceReport;

    /// Get current engine state and configuration
    fn get_engine_state(&self) -> EngineState;
}

/// Real inference engine implementation
pub struct RealInferenceEngine {
    /// Real BitNet model loaded from GGUF
    model: BitNetModel,

    /// Universal tokenizer with GGUF integration
    tokenizer: UniversalTokenizer,

    /// Device configuration and optimization
    device_config: DeviceConfig,

    /// Performance monitoring and metrics
    performance_monitor: PerformanceMonitor,

    /// Quantization engine for device-aware operations
    quantization_engine: QuantizationEngine,
}

impl RealInferenceEngine {
    /// Create new engine with real model and tokenizer
    pub fn new(
        model: BitNetModel,
        tokenizer: UniversalTokenizer,
        config: EngineConfig
    ) -> Result<Self, InferenceError>;

    /// Create engine optimized for specific device
    pub fn for_device(
        model: BitNetModel,
        tokenizer: UniversalTokenizer,
        device: Device
    ) -> Result<Self, InferenceError>;

    /// Perform explicit prefill operation for cache warming
    pub async fn prefill(&mut self, tokens: &[TokenId]) -> Result<PrefillResult, InferenceError>;

    /// Generate tokens with detailed performance tracking
    pub async fn generate_tokens(
        &mut self,
        prompt_tokens: &[TokenId],
        config: GenerationConfig
    ) -> Result<GenerationResult, InferenceError>;
}
```

#### Inference Result Contract

```rust
/// Comprehensive inference result with metrics and validation
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated tokens
    pub tokens: Vec<TokenId>,

    /// Generated text (decoded)
    pub text: String,

    /// Performance metrics for this inference
    pub metrics: InferenceMetrics,

    /// Validation results if cross-validation enabled
    pub validation: Option<ValidationResult>,

    /// Device information used for inference
    pub device_info: DeviceInfo,
}

/// Detailed performance metrics for inference operations
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total inference time
    pub total_duration: Duration,

    /// Prefill timing (cache warming)
    pub prefill_duration: Duration,

    /// Token generation timing
    pub decode_duration: Duration,

    /// Tokenization timing
    pub tokenization_duration: Duration,

    /// Throughput metrics
    pub tokens_per_second: f64,

    /// Memory usage during inference
    pub memory_usage: MemoryUsage,

    /// GPU utilization if applicable
    pub gpu_metrics: Option<GPUMetrics>,
}

impl InferenceMetrics {
    /// Get throughput for prefill phase
    pub fn get_prefill_throughput(&self) -> f64;

    /// Get throughput for decode phase
    pub fn get_decode_throughput(&self) -> f64;

    /// Get overall efficiency metrics
    pub fn get_efficiency_metrics(&self) -> EfficiencyMetrics;
}
```

### Quantization Interface (`bitnet-quantization`)

#### Real Model Quantization Contract

```rust
/// Device-aware quantization for real model tensors
pub trait RealModelQuantizer {
    /// Quantize real tensor data with device-specific optimization
    fn quantize_real_tensors(
        &self,
        tensors: &[Tensor],
        format: QuantizationFormat
    ) -> Result<QuantizedTensors, QuantizationError>;

    /// Validate quantization accuracy against reference
    fn validate_numerical_accuracy(
        &self,
        original: &[f32],
        quantized: &[f32]
    ) -> AccuracyMetrics;

    /// Get device performance characteristics
    fn get_device_performance(&self) -> DevicePerformanceMetrics;

    /// Perform cross-validation against C++ implementation
    fn cross_validate_with_cpp(
        &self,
        tensors: &[Tensor],
        tolerance: f32
    ) -> CrossValidationResult;

    /// Get quantization capabilities for device
    fn get_quantization_capabilities(&self, device: Device) -> QuantizationCapabilities;
}

/// Production quantization engine with device awareness
pub struct DeviceAwareQuantizer {
    /// Current device configuration
    device_config: DeviceConfig,

    /// Supported quantization formats
    supported_formats: Vec<QuantizationFormat>,

    /// Performance optimization settings
    optimization_config: OptimizationConfig,

    /// Cross-validation configuration
    validation_config: ValidationConfig,
}

impl DeviceAwareQuantizer {
    /// Create quantizer for specific device
    pub fn for_device(device: Device, config: QuantizationConfig) -> Result<Self, QuantizationError>;

    /// Auto-detect optimal device and create quantizer
    pub fn auto_detect() -> Result<Self, QuantizationError>;

    /// Enable strict numerical validation mode
    pub fn with_strict_validation(mut self) -> Self;

    /// Configure cross-validation against C++ reference
    pub fn with_cpp_validation(mut self, cpp_config: CppValidationConfig) -> Self;
}
```

#### Quantization Format Support

```rust
/// Supported quantization formats with device capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationFormat {
    /// 2-bit signed quantization with GPU acceleration
    I2S { gpu_accelerated: bool },

    /// Table lookup quantization Level 1
    TL1 { table_size: usize },

    /// Table lookup quantization Level 2
    TL2 { table_size: usize },

    /// GGML-compatible IQ2_S format
    IQ2S { use_ffi: bool },
}

/// Device-specific quantization capabilities
#[derive(Debug, Clone)]
pub struct QuantizationCapabilities {
    /// Supported formats on this device
    pub supported_formats: Vec<QuantizationFormat>,

    /// Performance characteristics per format
    pub performance_profiles: HashMap<QuantizationFormat, PerformanceProfile>,

    /// Memory requirements per format
    pub memory_requirements: HashMap<QuantizationFormat, MemoryRequirements>,

    /// Accuracy preservation characteristics
    pub accuracy_profiles: HashMap<QuantizationFormat, AccuracyProfile>,
}
```

### Tokenizer Interface (`bitnet-tokenizers`)

#### Universal Tokenizer Contract

```rust
/// Universal tokenizer with GGUF integration and real model support
pub trait UniversalTokenizer {
    /// Tokenize text using real model vocabulary
    fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError>;

    /// Decode tokens to text using real model vocabulary
    fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizerError>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Check if tokenizer supports specific model
    fn supports_model(&self, model_info: &ModelInfo) -> bool;

    /// Validate tokenizer compatibility with model
    fn validate_compatibility(&self, model: &BitNetModel) -> CompatibilityResult;

    /// Get tokenizer metadata and configuration
    fn get_metadata(&self) -> TokenizerMetadata;
}

/// Production tokenizer with GGUF metadata integration
pub struct RealTokenizer {
    /// Tokenizer backend (BPE, SentencePiece, etc.)
    backend: TokenizerBackend,

    /// Vocabulary from GGUF metadata
    vocabulary: Vocabulary,

    /// Special token configuration
    special_tokens: SpecialTokens,

    /// Configuration from model metadata
    config: TokenizerConfig,
}

impl RealTokenizer {
    /// Create tokenizer from GGUF model metadata
    pub fn from_gguf_metadata(model: &BitNetModel) -> Result<Self, TokenizerError>;

    /// Create tokenizer from external file with model validation
    pub fn from_file_with_model(
        tokenizer_path: &Path,
        model: &BitNetModel
    ) -> Result<Self, TokenizerError>;

    /// Create fallback tokenizer for testing (only when explicitly allowed)
    pub fn create_mock_fallback(vocab_size: usize) -> Self;

    /// Validate tokenizer matches model requirements
    pub fn validate_model_compatibility(&self, model: &BitNetModel) -> ValidationResult;
}
```

#### Strict Mode and Fallback Control

```rust
/// Tokenizer provider with strict mode support
pub struct TokenizerProvider {
    strict_mode: bool,
    fallback_enabled: bool,
    cache_dir: Option<PathBuf>,
}

impl TokenizerProvider {
    /// Create provider with strict mode (no mock fallbacks)
    pub fn strict() -> Self;

    /// Create provider with fallback support for development
    pub fn with_fallback() -> Self;

    /// Load tokenizer with automatic backend detection
    pub fn load_for_model(&self, model: &BitNetModel) -> Result<Box<dyn UniversalTokenizer>, TokenizerError>;

    /// Check if mock fallback would be used (for validation)
    pub fn would_use_mock(&self, model: &BitNetModel) -> bool;
}

/// Environment-based tokenizer configuration
pub struct TokenizerEnvironment;

impl TokenizerEnvironment {
    /// Check if strict tokenizers mode is enabled
    pub fn is_strict_mode() -> bool {
        std::env::var("BITNET_STRICT_TOKENIZERS").map(|v| v == "1").unwrap_or(false)
    }

    /// Get configured tokenizer path from environment
    pub fn get_tokenizer_path() -> Option<PathBuf>;

    /// Get tokenizer backend preference from environment
    pub fn get_backend_preference() -> Option<TokenizerBackend>;
}
```

### Cross-Validation Interface (`crossval`)

#### C++ Parity Validation Contract

```rust
/// Cross-validation against C++ reference implementation
pub trait CrossValidator {
    /// Validate inference outputs against C++ reference
    fn validate_inference(
        &self,
        model_path: &Path,
        inputs: &[TokenId],
        rust_outputs: &[TokenId],
        tolerance: f32
    ) -> CrossValidationResult;

    /// Validate quantization accuracy against C++ implementation
    fn validate_quantization(
        &self,
        tensors: &[Tensor],
        rust_quantized: &[f32],
        tolerance: f32
    ) -> QuantizationValidationResult;

    /// Validate perplexity calculations
    fn validate_perplexity(
        &self,
        model_path: &Path,
        corpus: &str,
        rust_perplexity: f64,
        tolerance: f64
    ) -> PerplexityValidationResult;

    /// Run comprehensive validation suite
    fn run_full_validation(
        &self,
        config: ValidationSuiteConfig
    ) -> ComprehensiveValidationResult;
}

/// Production cross-validator with C++ integration
pub struct CppCrossValidator {
    /// Path to C++ implementation
    cpp_binary_path: PathBuf,

    /// Validation configuration
    validation_config: ValidationConfig,

    /// Tolerance settings for different operations
    tolerance_config: ToleranceConfig,

    /// Performance comparison settings
    performance_config: PerformanceComparisonConfig,
}

impl CppCrossValidator {
    /// Create validator with C++ binary path
    pub fn new(cpp_path: PathBuf, config: ValidationConfig) -> Result<Self, ValidationError>;

    /// Auto-detect C++ implementation and create validator
    pub fn auto_detect() -> Result<Self, ValidationError>;

    /// Configure numerical tolerances
    pub fn with_tolerances(mut self, tolerances: ToleranceConfig) -> Self;

    /// Enable performance comparison
    pub fn with_performance_comparison(mut self) -> Self;
}
```

#### Validation Result Types

```rust
/// Comprehensive cross-validation result
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Overall validation status
    pub passed: bool,

    /// Individual test results
    pub test_results: Vec<TestResult>,

    /// Numerical accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,

    /// Performance comparison if enabled
    pub performance_comparison: Option<PerformanceComparison>,

    /// Detailed error information
    pub errors: Vec<ValidationError>,

    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Tolerance configuration for different validation types
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    /// Tolerance for inference outputs
    pub inference_tolerance: f32,

    /// Tolerance for quantization operations
    pub quantization_tolerance: f32,

    /// Tolerance for perplexity calculations
    pub perplexity_tolerance: f64,

    /// Tolerance for performance metrics
    pub performance_tolerance: f32,
}

impl ToleranceConfig {
    /// Default tolerances for production use
    pub fn production() -> Self;

    /// Strict tolerances for development validation
    pub fn strict() -> Self;

    /// Relaxed tolerances for CI environments
    pub fn ci_friendly() -> Self;
}
```

## Command-Line Interface Contracts

### CLI Command Structure

#### Model Management Commands

```bash
# Enhanced model download with validation
cargo run -p xtask -- download-model [OPTIONS] --id <MODEL_ID>

Options:
  --id <MODEL_ID>           Hugging Face model identifier
  --file <FILENAME>         Specific file to download (optional)
  --validate               Validate model after download
  --cache-dir <DIR>        Custom cache directory
  --force                  Force re-download if exists
  --dry-run                Show what would be downloaded

# Model validation and compatibility checking
cargo run -p bitnet-cli -- compat-check <MODEL_PATH> [OPTIONS]

Options:
  --format <FORMAT>         Output format: human, json, yaml
  --strict                 Use strict validation rules
  --fix-output <PATH>      Generate fixed model if possible
  --report-file <PATH>     Save detailed report to file

# Model information and metadata inspection
cargo run -p bitnet-cli -- model-info <MODEL_PATH> [OPTIONS]

Options:
  --format <FORMAT>         Output format: human, json, table
  --show-tensors           Include tensor information
  --show-config            Include model configuration
  --export-metadata <PATH>  Export metadata to file
```

#### Inference Commands

```bash
# Real model inference with performance metrics
cargo run -p bitnet-cli -- run [OPTIONS] --model <MODEL_PATH>

Options:
  --model <PATH>           Path to GGUF model file
  --tokenizer <PATH>       Path to tokenizer file (optional)
  --prompt <TEXT>          Input prompt for generation
  --max-tokens <N>         Maximum tokens to generate
  --temperature <F>        Sampling temperature
  --deterministic          Use deterministic generation
  --metrics                Collect and display performance metrics
  --format <FORMAT>        Output format: human, json
  --device <DEVICE>        Force specific device: auto, gpu, cpu
  --batch-size <N>         Batch size for batch inference

# Batch inference from file
cargo run -p bitnet-cli -- run-batch [OPTIONS] --input-file <FILE>

Options:
  --input-file <PATH>      File containing prompts (one per line)
  --output-file <PATH>     Output file for results
  --batch-size <N>         Number of prompts per batch
  --parallel <N>           Number of parallel workers
  --metrics                Collect performance metrics per batch
```

#### Validation and Testing Commands

```bash
# Cross-validation against C++ reference
cargo run -p xtask -- crossval [OPTIONS] --model <MODEL_PATH>

Options:
  --model <PATH>           Path to GGUF model file
  --cpp-binary <PATH>      Path to C++ implementation
  --tolerance <F>          Numerical tolerance for validation
  --test-suite <NAME>      Specific test suite to run
  --performance            Include performance comparison
  --report-file <PATH>     Save validation report

# Perplexity evaluation
cargo run -p bitnet-cli -- score [OPTIONS] --model <MODEL_PATH>

Options:
  --model <PATH>           Path to GGUF model file
  --file <PATH>            Text file for evaluation
  --tokenizer <PATH>       Custom tokenizer path
  --batch-size <N>         Batch size for evaluation
  --device <DEVICE>        Device for evaluation
  --reference-impl <IMPL>  Reference implementation: cpp, python
  --format <FORMAT>        Output format: human, json
```

### Environment Variable Contracts

#### Model Discovery and Configuration

```bash
# Primary model configuration
export BITNET_GGUF="/path/to/model.gguf"              # Primary model file
export BITNET_TOKENIZER="/path/to/tokenizer.json"     # Tokenizer file
export BITNET_MODEL_CACHE="/path/to/cache"            # Model cache directory

# Device and performance configuration
export BITNET_DEVICE="auto"                           # Device preference: auto, gpu, cpu
export BITNET_GPU_MEMORY_LIMIT="4096"                 # GPU memory limit in MB
export BITNET_CPU_THREADS="16"                        # CPU thread count
export BITNET_MEMORY_POOL_SIZE="8192"                 # Memory pool size in MB

# Testing and validation configuration
export BITNET_STRICT_TOKENIZERS="1"                   # Disable mock tokenizer fallbacks
export BITNET_STRICT_NO_FAKE_GPU="1"                  # Disable fake GPU backends
export BITNET_DETERMINISTIC="1"                       # Enable deterministic mode
export BITNET_SEED="42"                               # Random seed for reproducibility

# Cross-validation configuration
export BITNET_CPP_DIR="/path/to/cpp/implementation"   # C++ reference implementation
export CROSSVAL_GGUF="/path/to/validation/model.gguf" # Model for cross-validation
export BITNET_VALIDATION_TOLERANCE="1e-4"             # Default numerical tolerance

# CI and automation configuration
export BITNET_CI_MODE="1"                             # Enable CI-optimized behavior
export BITNET_FAST_TESTS="1"                          # Skip slow tests in CI
export BITNET_CACHE_MODELS="1"                        # Enable model caching
export BITNET_PARALLEL_DOWNLOADS="4"                  # Parallel download connections
```

### Feature Flag Strategy

#### Build Configuration Contracts

```bash
# Core feature combinations
--no-default-features --features cpu           # CPU-only build (production)
--no-default-features --features gpu           # GPU acceleration enabled
--no-default-features --features inference     # Real inference capability
--no-default-features --features "cpu,gpu"     # Hybrid CPU/GPU support

# Advanced feature combinations
--features "cpu,crossval"                      # CPU with C++ cross-validation
--features "gpu,mixed-precision"               # GPU with FP16/BF16 support
--features "cpu,iq2s-ffi"                      # CPU with GGML FFI quantization
--features "cpu,spm"                           # CPU with SentencePiece tokenizer

# Testing and validation features
--features "crossval,strict-testing"           # Full validation with strict mode
--features "integration-tests,fixtures"        # Integration tests with test data
--features "performance-benchmarks"            # Performance testing suite
--features "ci-optimized"                      # CI-specific optimizations

# WebAssembly features
--target wasm32-unknown-unknown --features browser     # Browser WASM build
--target wasm32-unknown-unknown --features nodejs      # Node.js WASM build
--target wasm32-unknown-unknown --features "browser,debug"  # Debug WASM build
```

## Error Handling and Diagnostics

### Standardized Error Responses

#### Error Classification

```rust
/// Comprehensive error classification for diagnostics
#[derive(Debug, Clone)]
pub enum ErrorCategory {
    /// Model loading and format errors
    ModelLoading {
        error_type: ModelErrorType,
        severity: ErrorSeverity,
        recovery_suggestions: Vec<String>,
    },

    /// Inference execution errors
    InferenceExecution {
        error_type: InferenceErrorType,
        severity: ErrorSeverity,
        device_context: Option<DeviceContext>,
    },

    /// Cross-validation and accuracy errors
    ValidationFailure {
        validation_type: ValidationType,
        tolerance_exceeded: f32,
        reference_info: Option<ReferenceInfo>,
    },

    /// Performance and resource errors
    PerformanceIssue {
        issue_type: PerformanceIssueType,
        metrics: Option<PerformanceMetrics>,
        recommendations: Vec<String>,
    },
}

/// Actionable error information with recovery guidance
#[derive(Debug, Clone)]
pub struct DiagnosticError {
    /// Error category and classification
    pub category: ErrorCategory,

    /// Human-readable error message
    pub message: String,

    /// Technical details for debugging
    pub technical_details: Option<String>,

    /// Suggested recovery actions
    pub recovery_actions: Vec<RecoveryAction>,

    /// Related documentation links
    pub documentation_links: Vec<String>,

    /// Error context and metadata
    pub context: ErrorContext,
}

impl DiagnosticError {
    /// Create user-friendly error summary
    pub fn get_user_summary(&self) -> String;

    /// Get technical details for developers
    pub fn get_technical_details(&self) -> String;

    /// Get prioritized recovery actions
    pub fn get_recovery_plan(&self) -> Vec<RecoveryAction>;
}
```

### Recovery Action Framework

```rust
/// Structured recovery actions with automation support
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Action description
    pub description: String,

    /// Automated command if available
    pub command: Option<String>,

    /// Manual steps required
    pub manual_steps: Vec<String>,

    /// Success criteria
    pub success_criteria: String,

    /// Priority level
    pub priority: ActionPriority,
}

/// Automated recovery execution
pub trait RecoveryExecutor {
    /// Attempt to execute recovery action automatically
    fn execute_recovery(&self, action: &RecoveryAction) -> Result<RecoveryResult, RecoveryError>;

    /// Validate that recovery was successful
    fn validate_recovery(&self, action: &RecoveryAction) -> ValidationResult;

    /// Get recovery recommendations for manual steps
    fn get_manual_recommendations(&self, action: &RecoveryAction) -> Vec<String>;
}
```

## Testing Contracts

### Test Interface Standards

#### Unit Test Contracts

```rust
/// Standard test traits for BitNet.rs components
pub trait ComponentTestSuite {
    /// Test component with mock data (fast)
    fn test_with_mock_data(&self) -> Result<TestResult, TestError>;

    /// Test component with real data (slower)
    fn test_with_real_data(&self) -> Result<TestResult, TestError>;

    /// Performance benchmark test
    fn benchmark_performance(&self) -> Result<BenchmarkResult, TestError>;

    /// Cross-validation test against reference
    fn cross_validate(&self) -> Result<ValidationResult, TestError>;
}

/// Integration test contracts
pub trait IntegrationTestSuite {
    /// End-to-end pipeline test
    fn test_e2e_pipeline(&self) -> Result<TestResult, TestError>;

    /// Multi-device compatibility test
    fn test_device_compatibility(&self) -> Result<TestResult, TestError>;

    /// Performance regression test
    fn test_performance_regression(&self) -> Result<TestResult, TestError>;
}
```

#### Test Data Management

```rust
/// Test data provider with real/mock selection
pub trait TestDataProvider {
    /// Get test data for specific test type
    fn get_test_data(&self, test_type: TestType) -> Result<TestData, TestError>;

    /// Check if real test data is available
    fn has_real_data(&self, test_type: TestType) -> bool;

    /// Download and cache test data
    fn ensure_test_data(&self, test_type: TestType) -> Result<(), TestError>;

    /// Clean up test data cache
    fn cleanup_test_data(&self) -> Result<(), TestError>;
}

/// Test configuration for different environments
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Use real models when available
    pub prefer_real_models: bool,

    /// Enable cross-validation tests
    pub enable_cross_validation: bool,

    /// Performance testing configuration
    pub performance_config: Option<PerformanceTestConfig>,

    /// Timeout configuration
    pub timeout_config: TimeoutConfig,

    /// Resource limits
    pub resource_limits: ResourceLimits,
}
```

## Versioning and Compatibility

### API Version Management

```rust
/// API version information
#[derive(Debug, Clone)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
}

/// Compatibility matrix for API changes
pub trait CompatibilityChecker {
    /// Check if API version is compatible
    fn is_compatible(&self, required_version: &ApiVersion) -> bool;

    /// Get migration guide for version upgrade
    fn get_migration_guide(&self, from_version: &ApiVersion) -> Option<MigrationGuide>;

    /// Check feature compatibility
    fn check_feature_compatibility(&self, features: &[String]) -> CompatibilityResult;
}

/// Backward compatibility guarantees
pub struct BackwardCompatibility;

impl BackwardCompatibility {
    /// Check if change maintains backward compatibility
    pub fn validate_change(old_api: &ApiSignature, new_api: &ApiSignature) -> CompatibilityResult;

    /// Generate compatibility report
    pub fn generate_report(changes: &[ApiChange]) -> CompatibilityReport;
}
```

This comprehensive API contract reference ensures consistent interfaces across the BitNet.rs neural network inference pipeline while supporting real model integration, device-aware execution, and production-grade validation requirements.
