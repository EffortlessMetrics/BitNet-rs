//! Metal backend error types.

use thiserror::Error;

/// Errors produced by the Metal backend.
#[derive(Debug, Error)]
pub enum MetalError {
    #[error("no Metal device found")]
    NoDevice,

    #[error("Metal is not available on this platform")]
    NotAvailable,

    #[error("failed to create command queue")]
    CommandQueue,

    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    #[error("buffer allocation failed: size={size} bytes")]
    BufferAllocation { size: usize },

    #[error("kernel dispatch failed: {0}")]
    Dispatch(String),

    #[error("command buffer execution failed: {0}")]
    CommandBuffer(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, MetalError>;
