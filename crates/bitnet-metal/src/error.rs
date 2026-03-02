//! Metal backend error types.

use std::fmt;

/// Errors produced by the Metal backend.
#[derive(Debug)]
pub enum MetalError {
    /// No Metal device found on this system.
    NoDevice,
    /// Metal is not available (e.g. non-macOS platform).
    NotAvailable,
    /// Failed to create a command queue.
    CommandQueue,
    /// Shader compilation failed.
    ShaderCompilation(String),
    /// Pipeline creation failed.
    PipelineCreation(String),
    /// Buffer allocation failed.
    BufferAllocation {
        /// Requested allocation size in bytes.
        size: usize,
    },
    /// Kernel dispatch failed.
    Dispatch(String),
    /// Command buffer execution failed.
    CommandBuffer(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device found"),
            Self::NotAvailable => write!(f, "Metal is not available on this platform"),
            Self::CommandQueue => write!(f, "failed to create command queue"),
            Self::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
            Self::PipelineCreation(msg) => write!(f, "pipeline creation failed: {msg}"),
            Self::BufferAllocation { size } => {
                write!(f, "buffer allocation failed: size={size} bytes")
            }
            Self::Dispatch(msg) => write!(f, "kernel dispatch failed: {msg}"),
            Self::CommandBuffer(msg) => write!(f, "command buffer execution failed: {msg}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, MetalError>;
