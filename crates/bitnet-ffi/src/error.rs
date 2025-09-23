//! Error handling for the C API
//!
//! This module provides comprehensive error handling with detailed error codes
//! and messages, thread-safe error state management, and compatibility with
//! the existing C++ API error handling patterns.

use bitnet_common::BitNetError;
use std::cell::RefCell;
use std::fmt;

/// C API error types
#[derive(Debug, Clone)]
pub enum BitNetCError {
    /// Invalid argument provided to function
    InvalidArgument(String),
    /// Model file not found
    ModelNotFound(String),
    /// Model loading failed
    ModelLoadFailed(String),
    /// Inference operation failed
    InferenceFailed(String),
    /// Out of memory error
    OutOfMemory(String),
    /// Thread safety violation
    ThreadSafety(String),
    /// Invalid model ID
    InvalidModelId(String),
    /// Context length exceeded
    ContextLengthExceeded(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Internal error
    Internal(String),
}

impl fmt::Display for BitNetCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitNetCError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            BitNetCError::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
            BitNetCError::ModelLoadFailed(msg) => write!(f, "Model loading failed: {}", msg),
            BitNetCError::InferenceFailed(msg) => write!(f, "Inference failed: {}", msg),
            BitNetCError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            BitNetCError::ThreadSafety(msg) => write!(f, "Thread safety violation: {}", msg),
            BitNetCError::InvalidModelId(msg) => write!(f, "Invalid model ID: {}", msg),
            BitNetCError::ContextLengthExceeded(msg) => {
                write!(f, "Context length exceeded: {}", msg)
            }
            BitNetCError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            BitNetCError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl From<BitNetError> for BitNetCError {
    fn from(error: BitNetError) -> Self {
        match error {
            BitNetError::Model(model_error) => match model_error {
                bitnet_common::ModelError::NotFound { path } => {
                    BitNetCError::ModelNotFound(format!("Model file not found: {}", path))
                }
                bitnet_common::ModelError::InvalidFormat { format } => {
                    BitNetCError::ModelLoadFailed(format!("Invalid model format: {}", format))
                }
                bitnet_common::ModelError::LoadingFailed { reason } => {
                    BitNetCError::ModelLoadFailed(reason)
                }
                bitnet_common::ModelError::UnsupportedVersion { version } => {
                    BitNetCError::ModelLoadFailed(format!("Unsupported model version: {}", version))
                }
                bitnet_common::ModelError::FileIOError { path, source } => {
                    BitNetCError::ModelNotFound(format!(
                        "File I/O error for {}: {}",
                        path.display(),
                        source
                    ))
                }
                bitnet_common::ModelError::GGUFFormatError { message, details: _ } => {
                    BitNetCError::ModelLoadFailed(format!("GGUF format error: {}", message))
                }
            },
            BitNetError::Quantization(quant_error) => {
                BitNetCError::ModelLoadFailed(format!("Quantization error: {}", quant_error))
            }
            BitNetError::Kernel(kernel_error) => match kernel_error {
                bitnet_common::KernelError::NoProvider => {
                    BitNetCError::UnsupportedOperation("No kernel provider available".to_string())
                }
                bitnet_common::KernelError::ExecutionFailed { reason } => {
                    BitNetCError::InferenceFailed(format!("Kernel execution failed: {}", reason))
                }
                bitnet_common::KernelError::UnsupportedArchitecture { arch } => {
                    BitNetCError::UnsupportedOperation(format!(
                        "Unsupported architecture: {}",
                        arch
                    ))
                }
                bitnet_common::KernelError::GpuError { reason } => {
                    BitNetCError::InferenceFailed(format!("GPU error: {}", reason))
                }
                bitnet_common::KernelError::UnsupportedHardware { required, available } => {
                    BitNetCError::UnsupportedOperation(format!(
                        "Unsupported hardware: required {}, available {}",
                        required, available
                    ))
                }
                bitnet_common::KernelError::InvalidArguments { reason } => {
                    BitNetCError::InferenceFailed(format!("Invalid kernel arguments: {}", reason))
                }
                bitnet_common::KernelError::QuantizationFailed { reason } => {
                    BitNetCError::InferenceFailed(format!("Quantization failed: {}", reason))
                }
                bitnet_common::KernelError::MatmulFailed { reason } => {
                    BitNetCError::InferenceFailed(format!(
                        "Matrix multiplication failed: {}",
                        reason
                    ))
                }
            },
            BitNetError::Inference(inference_error) => match inference_error {
                bitnet_common::InferenceError::GenerationFailed { reason } => {
                    BitNetCError::InferenceFailed(reason)
                }
                bitnet_common::InferenceError::InvalidInput { reason } => {
                    BitNetCError::InvalidArgument(reason)
                }
                bitnet_common::InferenceError::ContextLengthExceeded { length } => {
                    BitNetCError::ContextLengthExceeded(format!(
                        "Context length {} exceeded maximum",
                        length
                    ))
                }
                bitnet_common::InferenceError::TokenizationFailed { reason } => {
                    BitNetCError::InferenceFailed(format!("Tokenization failed: {}", reason))
                }
            },
            BitNetError::Io(io_error) => BitNetCError::Internal(format!("IO error: {}", io_error)),
            BitNetError::Candle(candle_error) => {
                BitNetCError::Internal(format!("Candle error: {}", candle_error))
            }
            BitNetError::Config(config_error) => {
                BitNetCError::InvalidArgument(format!("Configuration error: {}", config_error))
            }
            BitNetError::Validation(validation_error) => {
                BitNetCError::InvalidArgument(format!("Validation error: {}", validation_error))
            }
        }
    }
}

// Thread-safe error state management
thread_local! {
    static LAST_ERROR: RefCell<Option<BitNetCError>> = const { RefCell::new(None) };
}

/// Set the last error for the current thread
pub fn set_last_error(error: BitNetCError) {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = Some(error);
    });
}

/// Get the last error for the current thread
pub fn get_last_error() -> Option<BitNetCError> {
    LAST_ERROR.with(|last| last.borrow().clone())
}

/// Clear the last error for the current thread
pub fn clear_last_error() {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = None;
    });
}

/// Convert a Rust Result to a C API result with error handling
pub fn handle_result<T, E>(result: Result<T, E>) -> Result<T, BitNetCError>
where
    E: Into<BitNetCError>,
{
    result.map_err(|e| e.into())
}

/// Macro for safe C string handling
#[macro_export]
macro_rules! safe_cstr {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return Err(BitNetCError::InvalidArgument("String pointer is null".to_string()));
        }
        match unsafe { std::ffi::CStr::from_ptr($ptr) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                return Err(BitNetCError::InvalidArgument(format!("Invalid UTF-8 string: {}", e)));
            }
        }
    };
}

/// Macro for safe pointer validation
#[macro_export]
macro_rules! validate_ptr {
    ($ptr:expr, $name:expr) => {
        if $ptr.is_null() {
            return Err(BitNetCError::InvalidArgument(format!("{} cannot be null", $name)));
        }
    };
}

/// Macro for safe model ID validation
#[macro_export]
macro_rules! validate_model_id {
    ($id:expr) => {
        if $id < 0 {
            return Err(BitNetCError::InvalidArgument("model_id must be non-negative".to_string()));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_common::ModelError;

    #[test]
    fn test_error_conversion() {
        let model_error =
            BitNetError::Model(ModelError::NotFound { path: "test.gguf".to_string() });
        let c_error: BitNetCError = model_error.into();

        match c_error {
            BitNetCError::ModelNotFound(msg) => {
                assert!(msg.contains("test.gguf"));
            }
            _ => panic!("Expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_error_state_management() {
        clear_last_error();
        assert!(get_last_error().is_none());

        let error = BitNetCError::InvalidArgument("test error".to_string());
        set_last_error(error.clone());

        let retrieved_error = get_last_error();
        assert!(retrieved_error.is_some());

        clear_last_error();
        assert!(get_last_error().is_none());
    }

    #[test]
    fn test_error_display() {
        let error = BitNetCError::ModelNotFound("test.gguf".to_string());
        let display_str = format!("{}", error);
        assert!(display_str.contains("Model not found"));
        assert!(display_str.contains("test.gguf"));
    }
}
