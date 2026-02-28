//! Kernel creation and launch for Level-Zero.

use crate::error::{LevelZeroError, Result};
use crate::ffi::ZeKernelHandle;
use crate::module::LevelZeroModule;

/// Dispatch dimensions for kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchDimensions {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

impl DispatchDimensions {
    pub fn new_1d(groups: u32) -> Self {
        Self { group_count_x: groups, group_count_y: 1, group_count_z: 1 }
    }

    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { group_count_x: x, group_count_y: y, group_count_z: 1 }
    }

    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { group_count_x: x, group_count_y: y, group_count_z: z }
    }

    /// Total number of work-groups.
    pub fn total_groups(&self) -> u64 {
        self.group_count_x as u64 * self.group_count_y as u64 * self.group_count_z as u64
    }
}

/// Group size (threads per work-group).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl GroupSize {
    pub fn new_1d(size: u32) -> Self {
        Self { x: size, y: 1, z: 1 }
    }

    pub fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for GroupSize {
    fn default() -> Self {
        Self::new_1d(256)
    }
}

/// Builder for creating and configuring a Level-Zero kernel.
#[derive(Debug)]
pub struct KernelBuilder {
    name: String,
    group_size: GroupSize,
}

impl KernelBuilder {
    /// Create a kernel builder for the given kernel name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), group_size: GroupSize::default() }
    }

    /// Set the work-group size.
    pub fn group_size(mut self, gs: GroupSize) -> Self {
        self.group_size = gs;
        self
    }

    /// The kernel function name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build the kernel from a loaded module.
    ///
    /// Placeholder: real implementation calls `zeKernelCreate`.
    pub fn build(self, _module: &LevelZeroModule) -> Result<LevelZeroKernel> {
        if self.name.is_empty() {
            return Err(LevelZeroError::InvalidArgument { message: "kernel name is empty".into() });
        }
        tracing::debug!(kernel_name = %self.name, "Creating kernel (placeholder)");
        Ok(LevelZeroKernel { name: self.name, group_size: self.group_size, _handle: None })
    }
}

/// An owned Level-Zero kernel handle.
#[derive(Debug)]
pub struct LevelZeroKernel {
    name: String,
    group_size: GroupSize,
    _handle: Option<ZeKernelHandle>,
}

impl LevelZeroKernel {
    /// The kernel function name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The configured work-group size.
    pub fn group_size(&self) -> &GroupSize {
        &self.group_size
    }

    /// Whether this kernel has a live L0 handle.
    pub fn is_initialized(&self) -> bool {
        self._handle.is_some()
    }
}
