//! Tensor abstractions and utilities

use crate::{BitNetError, Result};
use candle_core::{Device, DType, Tensor as CandleTensor};

/// Tensor trait for unified tensor operations
pub trait Tensor: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn device(&self) -> &Device;
    fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]>;
    fn to_candle(&self) -> Result<CandleTensor>;
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
        let tensor = CandleTensor::from_slice(data, shape, device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(Self::new(tensor))
    }

    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        let tensor = CandleTensor::zeros(shape, dtype, device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(Self::new(tensor))
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
        self.inner.device()
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