//! Device memory allocation for Level-Zero.
//!
//! Wraps `zeMemAllocDevice`, `zeMemAllocHost`, `zeMemAllocShared`.

use crate::context::LevelZeroContext;
use crate::error::{LevelZeroError, Result};
use crate::ffi::ZeMemoryType;

/// Builder for device memory allocations.
#[derive(Debug)]
pub struct MemoryAllocBuilder {
    memory_type: ZeMemoryType,
    size: usize,
    alignment: usize,
}

impl MemoryAllocBuilder {
    /// Allocate device-local memory.
    pub fn device(size: usize) -> Self {
        Self {
            memory_type: ZeMemoryType::Device,
            size,
            alignment: 0,
        }
    }

    /// Allocate host-visible memory.
    pub fn host(size: usize) -> Self {
        Self {
            memory_type: ZeMemoryType::Host,
            size,
            alignment: 0,
        }
    }

    /// Allocate shared (unified) memory.
    pub fn shared(size: usize) -> Self {
        Self {
            memory_type: ZeMemoryType::Shared,
            size,
            alignment: 0,
        }
    }

    /// Set alignment requirement.
    pub fn alignment(mut self, align: usize) -> Self {
        self.alignment = align;
        self
    }

    /// Size of the requested allocation.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Memory type of the requested allocation.
    pub fn memory_type(&self) -> ZeMemoryType {
        self.memory_type
    }

    /// Perform the allocation within the given context.
    ///
    /// Placeholder: real implementation calls zeMemAllocDevice/Host/Shared.
    pub fn allocate(self, _ctx: &LevelZeroContext) -> Result<DeviceBuffer> {
        if self.size == 0 {
            return Err(LevelZeroError::InvalidArgument {
                message: "allocation size must be > 0".into(),
            });
        }
        tracing::debug!(
            mem_type = ?self.memory_type,
            size = self.size,
            alignment = self.alignment,
            "Allocating device memory (placeholder)"
        );
        Ok(DeviceBuffer {
            memory_type: self.memory_type,
            size: self.size,
            _alignment: self.alignment,
            _ptr: std::ptr::null_mut(),
        })
    }
}

/// An allocated device buffer.
///
/// In the real implementation, dropping this calls `zeMemFree`.
#[derive(Debug)]
pub struct DeviceBuffer {
    memory_type: ZeMemoryType,
    size: usize,
    _alignment: usize,
    _ptr: *mut std::ffi::c_void,
}

impl DeviceBuffer {
    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Memory type.
    pub fn memory_type(&self) -> ZeMemoryType {
        self.memory_type
    }

    /// Whether this buffer holds a non-null pointer.
    pub fn is_allocated(&self) -> bool {
        !self._ptr.is_null()
    }
}

// Safety: L0 memory handles are thread-safe per specification.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

/// Estimate total memory needed for a set of tensor sizes.
pub fn estimate_total_memory(tensor_sizes: &[usize]) -> usize {
    tensor_sizes.iter().sum()
}

/// Estimate memory with alignment padding.
pub fn estimate_aligned_memory(tensor_sizes: &[usize], alignment: usize) -> usize {
    tensor_sizes
        .iter()
        .map(|&s| {
            if alignment == 0 {
                s
            } else {
                (s + alignment - 1) / alignment * alignment
            }
        })
        .sum()
}
