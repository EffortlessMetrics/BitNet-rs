//! hipMalloc / hipMemcpy wrappers.

use crate::error::{Result, RocmError};
use crate::ffi::{HipDevicePtr, HipMemcpyKind};
use tracing::debug;

/// A device-side memory allocation.
pub struct DeviceBuffer {
    ptr: HipDevicePtr,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate `size` bytes on the current HIP device (stub).
    pub fn alloc(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(RocmError::InvalidArgument("zero-size allocation".into()));
        }
        debug!(size, "hipMalloc (stub)");
        // Stub: would call hipMalloc via dlsym.
        Err(RocmError::Allocation { size })
    }

    /// Copy host data to this device buffer (stub).
    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        let _ = src;
        debug!(size = self.size, "hipMemcpy H2D (stub)");
        Err(RocmError::KernelLaunch(
            "HIP runtime not linked — stub only".into(),
        ))
    }

    /// Copy device data back to host (stub).
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let _ = dst;
        debug!(size = self.size, "hipMemcpy D2H (stub)");
        Err(RocmError::KernelLaunch(
            "HIP runtime not linked — stub only".into(),
        ))
    }

    /// Raw pointer for kernel arguments.
    pub fn as_ptr(&self) -> HipDevicePtr {
        self.ptr
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            debug!(size = self.size, "hipFree (stub)");
            // Stub: would call hipFree.
        }
    }
}

/// Memory copy direction helper.
pub fn memcpy_kind_for(src_is_device: bool, dst_is_device: bool) -> HipMemcpyKind {
    match (src_is_device, dst_is_device) {
        (false, false) => HipMemcpyKind::HostToHost,
        (false, true) => HipMemcpyKind::HostToDevice,
        (true, false) => HipMemcpyKind::DeviceToHost,
        (true, true) => HipMemcpyKind::DeviceToDevice,
    }
}
