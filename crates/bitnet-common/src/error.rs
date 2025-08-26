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

/// Result type alias
pub type Result<T> = std::result::Result<T, BitNetError>;
