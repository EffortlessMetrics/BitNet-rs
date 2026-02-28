//! WebGPU error types.

use thiserror::Error;

/// Errors produced by the WebGPU backend.
#[derive(Debug, Error)]
pub enum WebGpuError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    #[error("failed to request device: {0}")]
    DeviceRequest(String),

    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("buffer mapping failed: {0}")]
    BufferMap(String),

    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    #[error("dispatch failed: {0}")]
    Dispatch(String),

    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::RequestDeviceError),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, WebGpuError>;
