//! Metal compute backend with MSL kernels for Apple Silicon GPU inference.
//!
//! This crate provides a [`MetalBackend`] that compiles MSL compute shaders and
//! dispatches them on Apple Silicon GPUs. On non-macOS platforms the backend
//! reports itself as unavailable — compilation succeeds everywhere.

pub mod capabilities;
pub mod error;
pub mod shader;

pub use capabilities::{MetalDeviceInfo, query_device};
pub use error::MetalError;

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

        Ok(Self { device_info, _device: device })
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
