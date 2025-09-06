//! Tensor extension traits for device operations

use bitnet_common::{ConcreteTensor, Device, Result};

/// Extension trait for tensor device operations
#[allow(dead_code)]
pub trait TensorDeviceExt {
    /// Create a new tensor on the specified device
    fn with_device(&self, device: Device) -> Result<ConcreteTensor>;
}

impl TensorDeviceExt for ConcreteTensor {
    fn with_device(&self, _device: Device) -> Result<ConcreteTensor> {
        // For CPU-only builds, moving to a device is a no-op - just clone
        // When GPU support is added, this will actually transfer the tensor
        Ok(self.clone())
    }
}
