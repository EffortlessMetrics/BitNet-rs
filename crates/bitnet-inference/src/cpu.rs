//! CPU backend implementation

use crate::{Backend, DeviceInfo, DeviceType};
use bitnet_common::{BitNetTensor, Result};
use bitnet_kernels::KernelProvider;
use candle_core::Device;

/// CPU backend for inference
pub struct CpuBackend {
    kernel_provider: Box<dyn KernelProvider>,
    device: Device,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Result<Self> {
        let kernel_provider = bitnet_kernels::select_cpu_kernel()?;
        let device = Device::Cpu;
        
        Ok(Self {
            kernel_provider,
            device,
        })
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU"
    }
    
    fn is_available(&self) -> bool {
        true // CPU is always available
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
        Box::new(Self::new().unwrap())
    }
    
    fn kernel_provider(&self) -> &dyn KernelProvider {
        self.kernel_provider.as_ref()
    }
    
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cpu,
            memory_total: None,
            memory_available: None,
            compute_capability: Some("CPU".to_string()),
        }
    }
}