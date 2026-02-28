//! ROCm/HIP error types.

use thiserror::Error;

/// HIP runtime error codes (subset matching hipError_t enum values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum HipErrorCode {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    FileNotFound = 301,
    NotFound = 500,
    NotReady = 600,
    NoBinaryForGpu = 702,
    InsufficientDriver = 804,
    PeerAccessNotEnabled = 704,
    LaunchFailure = 719,
    CooperativeLaunchTooLarge = 720,
    Unknown = 999,
}

impl HipErrorCode {
    pub fn from_raw(code: u32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            101 => Self::InvalidDevice,
            200 => Self::InvalidImage,
            201 => Self::InvalidContext,
            301 => Self::FileNotFound,
            500 => Self::NotFound,
            600 => Self::NotReady,
            702 => Self::NoBinaryForGpu,
            719 => Self::LaunchFailure,
            720 => Self::CooperativeLaunchTooLarge,
            804 => Self::InsufficientDriver,
            _ => Self::Unknown,
        }
    }

    /// Human-readable description of the error code.
    pub fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::InvalidValue => "invalid value",
            Self::OutOfMemory => "out of memory",
            Self::NotInitialized => "not initialized",
            Self::Deinitialized => "deinitialized",
            Self::InvalidDevice => "invalid device",
            Self::InvalidImage => "invalid image",
            Self::InvalidContext => "invalid context",
            Self::FileNotFound => "file not found",
            Self::NotFound => "not found",
            Self::NotReady => "not ready",
            Self::NoBinaryForGpu => "no binary for GPU",
            Self::LaunchFailure => "launch failure",
            Self::CooperativeLaunchTooLarge => "cooperative launch too large",
            Self::PeerAccessNotEnabled => "peer access not enabled",
            Self::InsufficientDriver => "insufficient driver",
            Self::Unknown => "unknown error",
        }
    }
}

impl std::fmt::Display for HipErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.description(), *self as u32)
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

    #[error("kernel compilation failed: {0}")]
    KernelCompilation(String),
}

/// Generic GPU error type for cross-backend compatibility.
#[derive(Debug, Error)]
#[error("GPU error: {message} (backend: rocm)")]
pub struct GpuError {
    pub message: String,
    pub code: Option<u32>,
}

impl From<RocmError> for GpuError {
    fn from(err: RocmError) -> Self {
        match &err {
            RocmError::Hip { code, .. } => GpuError {
                message: err.to_string(),
                code: Some(*code as u32),
            },
            _ => GpuError {
                message: err.to_string(),
                code: None,
            },
        }
    }
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
