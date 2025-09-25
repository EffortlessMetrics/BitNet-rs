//! Error types and handling

use thiserror::Error;

/// Main BitNet error type
#[derive(Error, Debug)]
pub enum BitNetError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    #[error("Quantization error: {0}")]
    Quantization(#[from] QuantizationError),
    #[error("Kernel error: {0}")]
    Kernel(#[from] KernelError),
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Security error: {0}")]
    Security(#[from] SecurityError),
}

/// Model-related errors
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {path}")]
    NotFound { path: String },
    #[error("Invalid model format: {format}")]
    InvalidFormat { format: String },
    #[error("Model loading failed: {reason}")]
    LoadingFailed { reason: String },
    #[error("Unsupported model version: {version}")]
    UnsupportedVersion { version: String },
    #[error("File I/O error for {path}: {source}")]
    FileIOError {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("GGUF format error: {message}")]
    GGUFFormatError { message: String, details: ValidationErrorDetails },
}

/// Validation error details for enhanced error reporting
#[derive(Debug, Clone)]
pub struct ValidationErrorDetails {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Quantization-related errors
#[derive(Error, Debug)]
pub enum QuantizationError {
    #[error("Unsupported quantization type: {qtype}")]
    UnsupportedType { qtype: String },
    #[error("Quantization failed: {reason}")]
    QuantizationFailed { reason: String },
    #[error("Invalid block size: {size}")]
    InvalidBlockSize { size: usize },
    #[error("Resource limit exceeded: {reason}")]
    ResourceLimit { reason: String },
    #[error("Invalid input dimensions: {reason}")]
    InvalidInput { reason: String },
    #[error("Memory allocation failed: {reason}")]
    MemoryAllocation { reason: String },
}

/// Kernel-related errors
#[derive(Error, Debug)]
pub enum KernelError {
    #[error("No available kernel provider")]
    NoProvider,
    #[error("Kernel execution failed: {reason}")]
    ExecutionFailed { reason: String },
    #[error("Unsupported architecture: {arch}")]
    UnsupportedArchitecture { arch: String },
    #[error("GPU error: {reason}")]
    GpuError { reason: String },
    #[error("Unsupported hardware: required {required}, available {available}")]
    UnsupportedHardware { required: String, available: String },
    #[error("Invalid arguments: {reason}")]
    InvalidArguments { reason: String },
    #[error("Quantization failed: {reason}")]
    QuantizationFailed { reason: String },
    #[error("Matrix multiplication failed: {reason}")]
    MatmulFailed { reason: String },
}

/// Inference-related errors
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Generation failed: {reason}")]
    GenerationFailed { reason: String },
    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },
    #[error("Context length exceeded: {length}")]
    ContextLengthExceeded { length: usize },
    #[error("Tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
}

/// Security-related errors for input validation and resource management
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Input validation failed: {reason}")]
    InputValidation { reason: String },
    #[error("Memory allocation attack detected: {reason}")]
    MemoryBomb { reason: String },
    #[error("Resource limit exceeded: {resource} = {value} exceeds limit {limit}")]
    ResourceLimit { resource: String, value: u64, limit: u64 },
    #[error("Malformed data structure: {reason}")]
    MalformedData { reason: String },
    #[error("Unsafe operation blocked: {operation} - {reason}")]
    UnsafeOperation { operation: String, reason: String },
}

/// Security limits for preventing attacks
pub struct SecurityLimits {
    /// Maximum tensor elements (1 billion)
    pub max_tensor_elements: u64,
    /// Maximum memory allocation (4GB)
    pub max_memory_allocation: usize,
    /// Maximum metadata size (100MB)
    pub max_metadata_size: usize,
    /// Maximum string length (1MB)
    pub max_string_length: usize,
    /// Maximum array length (1M elements)
    pub max_array_length: usize,
}

impl Default for SecurityLimits {
    fn default() -> Self {
        Self {
            max_tensor_elements: 1_000_000_000,            // 1B elements
            max_memory_allocation: 4 * 1024 * 1024 * 1024, // 4GB
            max_metadata_size: 100 * 1024 * 1024,          // 100MB
            max_string_length: 1024 * 1024,                // 1MB
            max_array_length: 1_000_000,                   // 1M elements
        }
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, BitNetError>;
