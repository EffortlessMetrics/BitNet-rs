//! OpenCL-specific error types.

use thiserror::Error;

/// Errors that can occur during OpenCL operations.
#[derive(Debug, Error)]
pub enum OpenClError {
    /// No OpenCL platforms found on the system.
    #[error("no OpenCL platforms found")]
    NoPlatforms,

    /// No suitable GPU device found.
    #[error("no suitable GPU device found: {reason}")]
    NoDevice { reason: String },

    /// Failed to create an OpenCL context.
    #[error("context creation failed: {reason}")]
    ContextCreation { reason: String },

    /// Failed to create a command queue.
    #[error("queue creation failed: {reason}")]
    QueueCreation { reason: String },

    /// Failed to compile an OpenCL program.
    #[error("program compilation failed for '{name}': {reason}")]
    ProgramCompilation { name: String, reason: String },

    /// Failed to allocate a buffer on the device.
    #[error("buffer allocation failed ({size} bytes): {reason}")]
    BufferAllocation { size: usize, reason: String },

    /// Failed during a data transfer operation.
    #[error("data transfer failed: {reason}")]
    DataTransfer { reason: String },

    /// Failed during kernel execution.
    #[error("kernel execution failed: {reason}")]
    KernelExecution { reason: String },
}

/// Convenience alias for OpenCL results.
pub type Result<T> = std::result::Result<T, OpenClError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_no_platforms() {
        let err = OpenClError::NoPlatforms;
        assert_eq!(err.to_string(), "no OpenCL platforms found");
    }

    #[test]
    fn error_display_no_device() {
        let err = OpenClError::NoDevice {
            reason: "Intel GPU not detected".into(),
        };
        assert!(err.to_string().contains("Intel GPU not detected"));
    }

    #[test]
    fn error_display_buffer_allocation() {
        let err = OpenClError::BufferAllocation {
            size: 4096,
            reason: "out of memory".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("4096"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn error_display_program_compilation() {
        let err = OpenClError::ProgramCompilation {
            name: "matmul_i2s".into(),
            reason: "syntax error".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("matmul_i2s"));
        assert!(msg.contains("syntax error"));
    }

    #[test]
    fn error_display_kernel_execution() {
        let err = OpenClError::KernelExecution {
            reason: "invalid work group size".into(),
        };
        assert!(err.to_string().contains("invalid work group size"));
    }

    #[test]
    fn error_display_data_transfer() {
        let err = OpenClError::DataTransfer {
            reason: "write failed".into(),
        };
        assert!(err.to_string().contains("write failed"));
    }
}
