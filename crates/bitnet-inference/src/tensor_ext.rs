//! Tensor extension traits for device operations

use bitnet_common::{BitNetError, ConcreteTensor, Device, Result, Tensor};

/// Extension trait for tensor device operations
#[allow(dead_code)]
pub trait TensorDeviceExt {
    /// Create a new tensor on the specified device
    fn with_device(&self, device: Device) -> Result<ConcreteTensor>;
}

impl TensorDeviceExt for ConcreteTensor {
    fn with_device(&self, device: Device) -> Result<ConcreteTensor> {
        let candle_device =
            device.to_candle().map_err(|e| BitNetError::Validation(e.to_string()))?;
        let target_device = Device::from(&candle_device);

        if self.device() == &target_device {
            return Ok(self.clone());
        }

        match self {
            ConcreteTensor::BitNet(t) => {
                let moved = t.as_candle().to_device(&candle_device)?;
                Ok(ConcreteTensor::bitnet(moved))
            }
            ConcreteTensor::Mock(t) => {
                let moved = t.clone().with_device(target_device);
                Ok(ConcreteTensor::Mock(moved))
            }
        }
    }
}
