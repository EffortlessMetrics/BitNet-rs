//! ROCm/HIP error types.

use thiserror::Error;

/// HIP runtime error codes (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipErrorCode {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    InvalidDevice = 101,
    FileNotFound = 301,
    NotReady = 600,
    Unknown = 999,
}

impl HipErrorCode {
    pub fn from_raw(code: u32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            101 => Self::InvalidDevice,
            301 => Self::FileNotFound,
            600 => Self::NotReady,
            _ => Self::Unknown,
        }
    }
}

/// Errors produced by the ROCm backend.
#[derive(Debug, Error)]
pub enum RocmError {
    #[error("HIP runtime error: {code:?} ({message})")]
    Hip {
        code: HipErrorCode,
        message: String,
    },

    #[error("ROCm runtime not found: {0}")]
    RuntimeNotFound(String),

    #[error("no AMD GPU device found")]
    NoDevice,

    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("memory allocation failed: size={size} bytes")]
    Allocation { size: usize },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("library loading failed: {0}")]
    LibraryLoad(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, RocmError>;

/// Check a HIP status code and return an error if non-zero.
pub fn check_hip(status: u32, context: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(RocmError::Hip {
            code: HipErrorCode::from_raw(status),
            message: context.to_string(),
        })
    }
}
