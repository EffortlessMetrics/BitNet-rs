//! HIP kernel launch wrappers.

use crate::error::{Result, RocmError};
use crate::ffi::{HipFunction, HipStream};
use std::ffi::c_void;
use tracing::debug;

/// Grid/block dimensions for a HIP kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub stream: HipStream,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
            stream: crate::ffi::HIP_STREAM_DEFAULT,
        }
    }
}

impl LaunchConfig {
    /// Create a 1-D launch config with `n` total threads.
    pub fn linear(n: u32, block_size: u32) -> Self {
        let grid_x = (n + block_size - 1) / block_size;
        Self {
            grid: (grid_x, 1, 1),
            block: (block_size, 1, 1),
            ..Default::default()
        }
    }

    /// Create a 2-D launch config (e.g. for matrix ops).
    pub fn grid_2d(rows: u32, cols: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (
                (cols + block_x - 1) / block_x,
                (rows + block_y - 1) / block_y,
                1,
            ),
            block: (block_x, block_y, 1),
            ..Default::default()
        }
    }

    /// Create a launch config for softmax/rmsnorm (one workgroup per row).
    pub fn per_row(num_rows: u32, block_size: u32) -> Self {
        Self {
            grid: (num_rows, 1, 1),
            block: (block_size, 1, 1),
            ..Default::default()
        }
    }
}

/// Device context for kernel launches — selects the target GPU.
#[derive(Debug, Clone, Copy)]
pub struct DeviceContext {
    /// HIP device ordinal (0-based).
    pub device_id: i32,
}

impl DeviceContext {
    /// Create a context targeting the given device.
    pub fn new(device_id: i32) -> Self {
        Self { device_id }
    }

    /// Select this device as the current HIP device (stub).
    ///
    /// In a real implementation this calls `hipSetDevice`.
    pub fn activate(&self) -> Result<()> {
        if self.device_id < 0 {
            return Err(RocmError::InvalidArgument(format!(
                "negative device id: {}",
                self.device_id,
            )));
        }
        debug!(device_id = self.device_id, "activating HIP device");
        // Stub: hipSetDevice(self.device_id)
        Ok(())
    }
}

impl Default for DeviceContext {
    fn default() -> Self {
        Self { device_id: 0 }
    }
}

/// Launch a HIP kernel (stub — requires linked runtime).
///
/// # Safety
/// Caller must ensure `function` and `args` are valid.
pub unsafe fn launch_kernel(
    function: HipFunction,
    config: &LaunchConfig,
    args: &[*mut c_void],
) -> Result<()> {
    if function.is_null() {
        return Err(RocmError::KernelLaunch("null function handle".into()));
    }
    debug!(
        grid = ?config.grid,
        block = ?config.block,
        shared_mem = config.shared_mem_bytes,
        "launching HIP kernel"
    );
    // Stub: actual hipLaunchKernel call would go here via dlsym.
    let _ = args;
    Err(RocmError::KernelLaunch(
        "HIP runtime not linked — stub only".into(),
    ))
}

/// Launch a HIP kernel on a specific device (stub).
///
/// Activates the target device before launching the kernel.
///
/// # Safety
/// Caller must ensure `function` and `args` are valid.
pub unsafe fn launch_kernel_on_device(
    device: &DeviceContext,
    function: HipFunction,
    config: &LaunchConfig,
    args: &[*mut c_void],
) -> Result<()> {
    device.activate()?;
    unsafe { launch_kernel(function, config, args) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_context_default_is_zero() {
        let ctx = DeviceContext::default();
        assert_eq!(ctx.device_id, 0);
    }

    #[test]
    fn device_context_activate_succeeds() {
        let ctx = DeviceContext::new(0);
        assert!(ctx.activate().is_ok());
    }

    #[test]
    fn device_context_negative_id_fails() {
        let ctx = DeviceContext::new(-1);
        assert!(ctx.activate().is_err());
    }

    #[test]
    fn per_row_launch_config() {
        let cfg = LaunchConfig::per_row(32, 256);
        assert_eq!(cfg.grid, (32, 1, 1));
        assert_eq!(cfg.block, (256, 1, 1));
    }
}
