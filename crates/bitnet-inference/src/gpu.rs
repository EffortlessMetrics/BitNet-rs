//! GPU backend implementation

use crate::{Backend, DeviceInfo, DeviceType};
use bitnet_common::{BitNetError, BitNetTensor, Result};
use bitnet_kernels::{KernelProvider, GpuKernel};
use candle_core::Device;

/// GPU backend for inference
pub struct GpuBackend {
    kernel_provider: Box<dyn KernelProvider>,
    device: Device,
    device_id: usize,
}

impl GpuBackend {
    /// Create a new GPU backend
    pub fn new() -> Result<Self> {
        let device_id = 0; // Default to first GPU
        let device = Device::new_cuda(device_id)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        
        let kernel_provider = bitnet_kernels::select_gpu_kernel(device_id)?;
        
        Ok(Self {
            kernel_provider,
            device,
            device_id,
        })
    }
    
    /// Create GPU backend for specific device
    pub fn with_device(device_id: usize) -> Result<Self> {
        let device = Device::new_cuda(device_id)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        
        let kernel_provider = bitnet_kernels::select_gpu_kernel(device_id)?;
        
        Ok(Self {
            kernel_provider,
            device,
            device_id,
        })
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        "GPU"
    }
    
    fn is_available(&self) -> bool {
        // Check if CUDA is available and device exists
        Device::new_cuda(self.device_id).is_ok()
    }
    
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Placeholder implementation
        Ok(text.chars().map(|c| c as u32).collect())
    }
    
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Placeholder implementation
        Ok(tokens.iter().map(|&t| char::from(t as u8)).collect())
    }
    
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<BitNetTensor> {
        BitNetTensor::from_slice(tokens, &[tokens.len()], &self.device)
    }
    
    fn is_eos_token(&self, token: u32) -> bool {
        token == 2 // Placeholder EOS token ID
    }
    
    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self::with_device(self.device_id).unwrap())
    }
    
    fn kernel_provider(&self) -> &dyn KernelProvider {
        self.kernel_provider.as_ref()
    }
    
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cuda(self.device_id),
            memory_total: None, // Would query actual GPU memory
            memory_available: None, // Would query actual GPU memory
            compute_capability: Some(format!("CUDA Device {}", self.device_id)),
        }
    }
}