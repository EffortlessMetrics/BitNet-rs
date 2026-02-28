//! OpenCL context wrapper with error handling.

use crate::device::OpenClDevice;
use crate::error::{OpenClError, Result};
use crate::queue::OpenClQueue;
use opencl3::context::Context;
use tracing::info;

/// A high-level OpenCL context bound to a single device.
pub struct OpenClContext {
    /// The underlying opencl3 context.
    pub(crate) inner: Context,
    /// The device this context is bound to.
    pub(crate) device: OpenClDevice,
}

impl std::fmt::Debug for OpenClContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClContext")
            .field("device", &self.device.device_name)
            .field("platform", &self.device.platform_name)
            .finish()
    }
}

// SAFETY: OpenCL contexts are reference-counted internally and safe to share
// across threads when command queues serialise access.
unsafe impl Send for OpenClContext {}
unsafe impl Sync for OpenClContext {}

impl OpenClContext {
    /// Create a context from an already-selected device.
    pub fn from_device(device: OpenClDevice) -> Result<Self> {
        let inner =
            Context::from_device(&device.device).map_err(|e| {
                OpenClError::ContextCreation {
                    reason: e.to_string(),
                }
            })?;

        info!(
            "OpenCL context created on {} ({})",
            device.device_name, device.platform_name
        );

        Ok(Self { inner, device })
    }

    /// Create a context by automatically selecting the first Intel GPU.
    pub fn new_intel() -> Result<Self> {
        let device = OpenClDevice::find_intel_gpu()?;
        Self::from_device(device)
    }

    /// Create a command queue on this context.
    pub fn create_queue(&self) -> Result<OpenClQueue> {
        OpenClQueue::new(self)
    }

    /// Get the device name.
    pub fn device_name(&self) -> &str {
        &self.device.device_name
    }

    /// Get the platform name.
    pub fn platform_name(&self) -> &str {
        &self.device.platform_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_intel_graceful_without_hardware() {
        let result = OpenClContext::new_intel();
        match result {
            Ok(ctx) => {
                assert!(!ctx.device_name().is_empty());
                assert!(!ctx.platform_name().is_empty());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("no Intel GPU")
                        || msg.contains("no OpenCL platforms")
                        || msg.contains("context creation failed"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn debug_impl() {
        let result = OpenClContext::new_intel();
        if let Ok(ctx) = result {
            let dbg = format!("{:?}", ctx);
            assert!(dbg.contains("OpenClContext"));
        }
    }
}
