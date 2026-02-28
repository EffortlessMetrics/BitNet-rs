//! Level-Zero error codes mapped to Rust error types.

use crate::ffi::ZeResult;

/// Errors from Level-Zero operations.
#[derive(Debug, thiserror::Error)]
pub enum LevelZeroError {
    /// Level-Zero runtime library not found.
    #[error("Level-Zero runtime not found: {0}")]
    RuntimeNotFound(String),

    /// A Level-Zero API call returned an error.
    #[error("Level-Zero API error: {result}")]
    ApiError { result: ZeResult },

    /// No compatible GPU device was found.
    #[error("no Level-Zero GPU device found")]
    NoDeviceFound,

    /// Module (SPIR-V) compilation failed.
    #[error("SPIR-V module build failed: {message}")]
    ModuleBuildFailed { message: String },

    /// Kernel not found in loaded module.
    #[error("kernel '{name}' not found in module")]
    KernelNotFound { name: String },

    /// Memory allocation failed.
    #[error("device memory allocation failed: requested {requested_bytes} bytes")]
    AllocationFailed { requested_bytes: usize },

    /// Invalid argument passed to a builder or API wrapper.
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },

    /// The driver version is unsupported.
    #[error("unsupported driver version: {version}")]
    UnsupportedVersion { version: String },
}

impl From<ZeResult> for LevelZeroError {
    fn from(result: ZeResult) -> Self {
        Self::ApiError { result }
    }
}

/// Convenience result type for Level-Zero operations.
pub type Result<T> = std::result::Result<T, LevelZeroError>;

/// Check a `ze_result_t` and convert to `Result<()>`.
pub fn check(result: ZeResult) -> Result<()> {
    if result.is_success() {
        Ok(())
    } else {
        Err(LevelZeroError::ApiError { result })
    }
}
