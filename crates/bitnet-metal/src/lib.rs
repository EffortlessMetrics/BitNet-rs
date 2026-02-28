//! Metal compute backend with MSL kernels for Apple Silicon GPU inference.
//!
//! This crate provides a [`MetalBackend`] that compiles MSL compute shaders and
//! dispatches them on Apple Silicon GPUs. On non-macOS platforms the backend
//! reports itself as unavailable — compilation succeeds everywhere.

pub mod capabilities;
pub mod command;
pub mod error;
pub mod pipeline;
pub mod shader;

pub use capabilities::{query_device, MetalDeviceInfo};
pub use command::{CommandBuffer, CommandBufferState};
pub use error::MetalError;
pub use pipeline::{PipelineCache, PipelineDescriptor};

use crate::error::Result;

/// High-level Metal compute backend.
///
/// Wraps a Metal device, command queue, and compiled compute pipelines for
/// matmul, softmax, and RMS normalisation.
pub struct MetalBackend {
    device_info: Option<MetalDeviceInfo>,
    #[cfg(target_os = "macos")]
    _device: metal::Device,
}

impl MetalBackend {
    /// Initialise the Metal backend.
    ///
    /// On non-macOS this returns `Err(MetalError::NotAvailable)`.
    #[cfg(target_os = "macos")]
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
        let device_info = query_device();

        tracing::info!(
            name = %device.name(),
            unified_memory = device.has_unified_memory(),
            "Metal backend initialised"
        );

        Ok(Self {
            device_info,
            _device: device,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Result<Self> {
        Err(MetalError::NotAvailable)
    }

    /// Backend name for logging / registry.
    pub fn name(&self) -> &'static str {
        "metal"
    }

    /// Whether the backend is available.
    pub fn is_available(&self) -> bool {
        self.device_info.is_some()
    }

    /// Device information, if available.
    pub fn device_info(&self) -> Option<&MetalDeviceInfo> {
        self.device_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- shader source tests (work everywhere) ---

    #[test]
    fn matmul_shader_non_empty() {
        assert!(!shader::MATMUL_MSL.is_empty());
    }

    #[test]
    fn softmax_shader_non_empty() {
        assert!(!shader::SOFTMAX_MSL.is_empty());
    }

    #[test]
    fn rmsnorm_shader_non_empty() {
        assert!(!shader::RMSNORM_MSL.is_empty());
    }

    #[test]
    fn matmul_shader_has_kernel_function() {
        assert!(shader::MATMUL_MSL.contains("kernel void matmul"));
    }

    #[test]
    fn softmax_shader_has_barrier() {
        assert!(shader::SOFTMAX_MSL.contains("threadgroup_barrier"));
    }

    #[test]
    fn rmsnorm_shader_has_eps() {
        assert!(shader::RMSNORM_MSL.contains("eps"));
    }

    #[test]
    fn rope_shader_non_empty() {
        assert!(!shader::ROPE_MSL.is_empty());
    }

    #[test]
    fn rope_shader_has_kernel_function() {
        assert!(shader::ROPE_MSL.contains("kernel void rope"));
    }

    #[test]
    fn rope_shader_uses_freq_cos_sin() {
        assert!(shader::ROPE_MSL.contains("freq_cos"));
        assert!(shader::ROPE_MSL.contains("freq_sin"));
    }

    #[test]
    fn attention_shader_non_empty() {
        assert!(!shader::ATTENTION_MSL.is_empty());
    }

    #[test]
    fn attention_shader_has_kernel_function() {
        assert!(shader::ATTENTION_MSL.contains("kernel void attention"));
    }

    #[test]
    fn attention_shader_has_softmax_reduction() {
        assert!(shader::ATTENTION_MSL.contains("threadgroup_barrier"));
        assert!(shader::ATTENTION_MSL.contains("shared_max"));
    }

    #[test]
    fn all_kernels_registry_has_five_entries() {
        assert_eq!(shader::ALL_KERNELS.len(), 5);
    }

    #[test]
    fn get_kernel_source_found() {
        assert!(shader::get_kernel_source("matmul").is_some());
        assert!(shader::get_kernel_source("rope").is_some());
        assert!(shader::get_kernel_source("attention").is_some());
    }

    #[test]
    fn get_kernel_source_not_found() {
        assert!(shader::get_kernel_source("nonexistent").is_none());
    }

    #[test]
    fn pipeline_cache_validates_all_builtin_kernels() {
        let cache = PipelineCache::new();
        for (name, _) in shader::ALL_KERNELS {
            assert!(cache.validate(name).is_ok(), "pipeline missing: {name}");
        }
    }

    #[test]
    fn command_buffer_encode_commit_wait_lifecycle() {
        use crate::command::CommandBufferState;
        let mut buf = CommandBuffer::new();
        let cache = PipelineCache::new();
        let pipe = cache.get("softmax").unwrap();
        buf.encode_dispatch(pipe, (32, 1, 1)).unwrap();
        buf.commit().unwrap();
        buf.wait_until_completed().unwrap();
        assert_eq!(buf.state(), CommandBufferState::Completed);
    }

    #[test]
    fn error_display_no_device() {
        let e = MetalError::NoDevice;
        assert_eq!(format!("{e}"), "no Metal device found");
    }

    #[test]
    fn error_display_not_available() {
        let e = MetalError::NotAvailable;
        assert!(format!("{e}").contains("not available"));
    }

    // --- platform-gated tests ---

    #[test]
    #[ignore = "requires macOS with Metal-capable GPU"]
    fn backend_init_macos() {
        let backend = MetalBackend::new().expect("Metal backend init");
        assert!(backend.is_available());
        assert_eq!(backend.name(), "metal");
        let info = backend.device_info().expect("device info");
        assert!(!info.name.is_empty());
    }

    #[test]
    #[ignore = "requires macOS with Metal-capable GPU"]
    fn query_device_info() {
        let info = query_device().expect("should find Metal device on macOS");
        assert!(!info.name.is_empty());
        assert!(info.max_threads_per_threadgroup > 0);
        assert!(info.max_buffer_length > 0);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn non_macos_returns_not_available() {
        assert!(MetalBackend::new().is_err());
        assert!(query_device().is_none());
    }
}
