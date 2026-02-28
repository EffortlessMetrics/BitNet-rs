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
        Self { grid: (grid_x, 1, 1), block: (block_size, 1, 1), ..Default::default() }
    }

    /// Create a 2-D launch config (e.g. for matrix ops).
    pub fn grid_2d(rows: u32, cols: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: ((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y, 1),
            block: (block_x, block_y, 1),
            ..Default::default()
        }
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
    Err(RocmError::KernelLaunch("HIP runtime not linked — stub only".into()))
}
