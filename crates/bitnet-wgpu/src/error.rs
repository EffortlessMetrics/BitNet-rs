//! Error types for the bitnet-wgpu backend.

use std::fmt;

/// Errors produced by the wgpu compute backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    /// Failed to create a GPU device or adapter.
    #[error("device creation failed: {0}")]
    DeviceCreation(String),

    /// Buffer allocation failure.
    #[error("buffer allocation failed: {0}")]
    BufferAllocation(String),

    /// Shader module compilation failure.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Compute pipeline creation failure.
    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// Buffer mapping (readback) failure.
    #[error("buffer mapping failed: {0}")]
    BufferMapping(String),

    /// Dispatch or submission error.
    #[error("dispatch failed: {0}")]
    Dispatch(String),

    /// Internal / unexpected error.
    #[error("internal error: {0}")]
    Internal(String),
}

impl WgpuError {
    /// Convenience constructor for device creation errors.
    pub fn device(msg: impl fmt::Display) -> Self {
        Self::DeviceCreation(msg.to_string())
    }

    /// Convenience constructor for shader compilation errors.
    pub fn shader(msg: impl fmt::Display) -> Self {
        Self::ShaderCompilation(msg.to_string())
    }

    /// Convenience constructor for pipeline creation errors.
    pub fn pipeline(msg: impl fmt::Display) -> Self {
        Self::PipelineCreation(msg.to_string())
    }

    /// Convenience constructor for buffer mapping errors.
    pub fn mapping(msg: impl fmt::Display) -> Self {
        Self::BufferMapping(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_device_creation() {
        let err = WgpuError::DeviceCreation("no adapter found".into());
        assert_eq!(err.to_string(), "device creation failed: no adapter found");
    }

    #[test]
    fn error_display_buffer_allocation() {
        let err = WgpuError::BufferAllocation("out of memory".into());
        assert_eq!(err.to_string(), "buffer allocation failed: out of memory");
    }

    #[test]
    fn error_display_shader_compilation() {
        let err = WgpuError::ShaderCompilation("syntax error at line 5".into());
        assert_eq!(err.to_string(), "shader compilation failed: syntax error at line 5");
    }

    #[test]
    fn error_display_pipeline_creation() {
        let err = WgpuError::PipelineCreation("bind group mismatch".into());
        assert_eq!(err.to_string(), "pipeline creation failed: bind group mismatch");
    }

    #[test]
    fn error_display_buffer_mapping() {
        let err = WgpuError::BufferMapping("map async failed".into());
        assert_eq!(err.to_string(), "buffer mapping failed: map async failed");
    }

    #[test]
    fn error_display_dispatch() {
        let err = WgpuError::Dispatch("command encoder error".into());
        assert_eq!(err.to_string(), "dispatch failed: command encoder error");
    }

    #[test]
    fn error_display_internal() {
        let err = WgpuError::Internal("unexpected state".into());
        assert_eq!(err.to_string(), "internal error: unexpected state");
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WgpuError>();
    }

    #[test]
    fn error_source_chain() {
        let err = WgpuError::device("adapter not found");
        // WgpuError has no source — verify the trait impl exists and returns None.
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn convenience_constructors() {
        let e1 = WgpuError::device("d");
        assert!(matches!(e1, WgpuError::DeviceCreation(_)));

        let e2 = WgpuError::shader("s");
        assert!(matches!(e2, WgpuError::ShaderCompilation(_)));

        let e3 = WgpuError::pipeline("p");
        assert!(matches!(e3, WgpuError::PipelineCreation(_)));

        let e4 = WgpuError::mapping("m");
        assert!(matches!(e4, WgpuError::BufferMapping(_)));
    }
}
