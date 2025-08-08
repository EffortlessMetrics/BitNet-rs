//! Tensor abstractions and utilities

use crate::{BitNetError, Device, Result};
use candle_core::{DType, Tensor as CandleTensor};

/// Tensor trait for unified tensor operations
pub trait Tensor: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn device(&self) -> &Device;
    fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]>;
    fn to_candle(&self) -> Result<CandleTensor>;
}

/// Concrete tensor type that can be used instead of dyn Tensor
#[derive(Debug, Clone)]
pub enum ConcreteTensor {
    BitNet(BitNetTensor),
    Mock(MockTensor),
}

impl ConcreteTensor {
    pub fn mock(shape: Vec<usize>) -> Self {
        Self::Mock(MockTensor::new(shape))
    }

    pub fn bitnet(tensor: CandleTensor) -> Self {
        Self::BitNet(BitNetTensor::new(tensor))
    }
}

impl Tensor for ConcreteTensor {
    fn shape(&self) -> &[usize] {
        match self {
            Self::BitNet(t) => t.shape(),
            Self::Mock(t) => t.shape(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::BitNet(t) => t.dtype(),
            Self::Mock(t) => t.dtype(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Self::BitNet(t) => t.device(),
            Self::Mock(t) => t.device(),
        }
    }

    fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        match self {
            Self::BitNet(t) => t.as_slice(),
            Self::Mock(t) => t.as_slice(),
        }
    }

    fn to_candle(&self) -> Result<CandleTensor> {
        match self {
            Self::BitNet(t) => t.to_candle(),
            Self::Mock(t) => t.to_candle(),
        }
    }
}

/// Mock tensor for testing
#[derive(Debug, Clone)]
pub struct MockTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
    device: Device,
}

impl MockTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.1; size],
            device: Device::Cpu,
        }
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

/// Tensor implementation wrapping Candle tensors
#[derive(Debug, Clone)]
pub struct BitNetTensor {
    inner: CandleTensor,
}

impl BitNetTensor {
    pub fn new(tensor: CandleTensor) -> Self {
        Self { inner: tensor }
    }

    pub fn from_slice<T: bytemuck::Pod + candle_core::WithDType>(
        data: &[T],
        shape: &[usize],
        device: &Device,
    ) -> Result<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let tensor = CandleTensor::from_slice(data, shape, &candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(Self::new(tensor))
    }

    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        let candle_device = Self::device_to_candle(device)?;
        let tensor = CandleTensor::zeros(shape, dtype, &candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(Self::new(tensor))
    }

    /// Convert our Device to candle Device
    fn device_to_candle(device: &Device) -> Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(_id) => {
                #[cfg(feature = "gpu")]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda_device = candle_core::CudaDevice::new(*_id)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    Ok(candle_core::Device::Cuda(cuda_device))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(BitNetError::Validation("CUDA not available".to_string()))
                }
            }
            Device::Metal => {
                #[cfg(feature = "gpu")]
                {
                    Ok(candle_core::Device::Metal(
                        candle_core::MetalDevice::new(0)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?,
                    ))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(BitNetError::Validation("Metal not available".to_string()))
                }
            }
        }
    }

    pub fn inner(&self) -> &CandleTensor {
        &self.inner
    }

    pub fn into_inner(self) -> CandleTensor {
        self.inner
    }
}

impl Tensor for BitNetTensor {
    fn shape(&self) -> &[usize] {
        self.inner.shape().dims()
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    fn device(&self) -> &Device {
        // This is a simplified implementation - in practice we'd need to store
        // the device or convert properly each time
        &Device::Cpu
    }

    fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        // This is a simplified implementation - in practice, we'd need
        // to handle device transfers and type conversions properly
        Err(BitNetError::Validation(
            "Direct slice access not implemented".to_string(),
        ))
    }

    fn to_candle(&self) -> Result<CandleTensor> {
        Ok(self.inner.clone())
    }
}

impl Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }

    fn to_candle(&self) -> Result<CandleTensor> {
        let candle_device = match &self.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(_id) => {
                #[cfg(feature = "gpu")]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda_device = candle_core::CudaDevice::new(*_id)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    candle_core::Device::Cuda(cuda_device)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(BitNetError::Validation("CUDA not available".to_string()));
                }
            }
            Device::Metal => {
                #[cfg(feature = "gpu")]
                {
                    candle_core::Device::Metal(
                        candle_core::MetalDevice::new(0)
                            .map_err(|e| BitNetError::Validation(e.to_string()))?,
                    )
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(BitNetError::Validation("Metal not available".to_string()));
                }
            }
        };

        CandleTensor::from_slice(&self.data, self.shape.as_slice(), &candle_device)
            .map_err(|e| BitNetError::Validation(e.to_string()))
    }
}
