//! Backend abstraction for inference engines

use crate::{InferenceConfig, BackendPreference};
use bitnet_common::{BitNetError, BitNetTensor, InferenceError, Result};
use bitnet_kernels::KernelProvider;
use std::sync::Arc;

/// Backend trait for inference operations
pub trait Backend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
    
    /// Tokenize text input
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    
    /// Detokenize tokens to text
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    
    /// Convert tokens to tensor
    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<BitNetTensor>;
    
    /// Check if token is EOS token
    fn is_eos_token(&self, token: u32) -> bool;
    
    /// Clone the backend (for use in streaming)
    fn clone_backend(&self) -> Box<dyn Backend>;
    
    /// Get kernel provider
    fn kernel_provider(&self) -> &dyn KernelProvider;
    
    /// Get device information
    fn device_info(&self) -> DeviceInfo;
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub memory_total: Option<usize>,
    pub memory_available: Option<usize>,
    pub compute_capability: Option<String>,
}

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda(usize),
    Metal,
}

/// Backend selection utilities
pub fn select_best_backend(config: &InferenceConfig) -> Result<Box<dyn Backend>> {
    match config.backend_preference {
        BackendPreference::Auto => select_auto_backend(),
        BackendPreference::Cpu => Ok(Box::new(crate::cpu::CpuBackend::new()?)),
        BackendPreference::Gpu => select_gpu_backend_with_fallback(),
        BackendPreference::CpuOnly => Ok(Box::new(crate::cpu::CpuBackend::new()?)),
        BackendPreference::GpuOnly => select_gpu_backend_only(),
    }
}

fn select_auto_backend() -> Result<Box<dyn Backend>> {
    // Try GPU first, fallback to CPU
    if let Ok(gpu_backend) = crate::gpu::GpuBackend::new() {
        if gpu_backend.is_available() {
            return Ok(Box::new(gpu_backend));
        }
    }
    
    Ok(Box::new(crate::cpu::CpuBackend::new()?))
}

fn select_gpu_backend_with_fallback() -> Result<Box<dyn Backend>> {
    match crate::gpu::GpuBackend::new() {
        Ok(gpu_backend) if gpu_backend.is_available() => Ok(Box::new(gpu_backend)),
        _ => Ok(Box::new(crate::cpu::CpuBackend::new()?)),
    }
}

fn select_gpu_backend_only() -> Result<Box<dyn Backend>> {
    let gpu_backend = crate::gpu::GpuBackend::new()
        .map_err(|_| InferenceError::GenerationFailed {
            reason: "GPU backend not available".to_string()
        })?;
    
    if !gpu_backend.is_available() {
        return Err(InferenceError::GenerationFailed {
            reason: "GPU backend not available".to_string()
        }.into());
    }
    
    Ok(Box::new(gpu_backend))
}

/// Backend registry for managing multiple backends
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }
    
    /// Register a backend
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }
    
    /// Get all available backends
    pub fn available_backends(&self) -> Vec<&dyn Backend> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .map(|b| b.as_ref())
            .collect()
    }
    
    /// Get backend by name
    pub fn get_backend(&self, name: &str) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }
    
    /// Select best backend based on preference
    pub fn select_backend(&self, preference: BackendPreference) -> Result<&dyn Backend> {
        match preference {
            BackendPreference::Auto => {
                // Prefer GPU, fallback to CPU
                self.available_backends()
                    .into_iter()
                    .find(|b| matches!(b.device_info().device_type, DeviceType::Cuda(_)))
                    .or_else(|| {
                        self.available_backends()
                            .into_iter()
                            .find(|b| matches!(b.device_info().device_type, DeviceType::Cpu))
                    })
                    .ok_or_else(|| InferenceError::GenerationFailed {
                        reason: "No available backend found".to_string()
                    }.into())
            }
            BackendPreference::Cpu | BackendPreference::CpuOnly => {
                self.available_backends()
                    .into_iter()
                    .find(|b| matches!(b.device_info().device_type, DeviceType::Cpu))
                    .ok_or_else(|| InferenceError::GenerationFailed {
                        reason: "CPU backend not available".to_string()
                    }.into())
            }
            BackendPreference::Gpu => {
                self.available_backends()
                    .into_iter()
                    .find(|b| matches!(b.device_info().device_type, DeviceType::Cuda(_)))
                    .or_else(|| {
                        self.available_backends()
                            .into_iter()
                            .find(|b| matches!(b.device_info().device_type, DeviceType::Cpu))
                    })
                    .ok_or_else(|| InferenceError::GenerationFailed {
                        reason: "No suitable backend found".to_string()
                    }.into())
            }
            BackendPreference::GpuOnly => {
                self.available_backends()
                    .into_iter()
                    .find(|b| matches!(b.device_info().device_type, DeviceType::Cuda(_)))
                    .ok_or_else(|| InferenceError::GenerationFailed {
                        reason: "GPU backend not available".to_string()
                    }.into())
            }
        }
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        
        // Register CPU backend
        if let Ok(cpu_backend) = crate::cpu::CpuBackend::new() {
            registry.register(Box::new(cpu_backend));
        }
        
        // Register GPU backend if available
        if let Ok(gpu_backend) = crate::gpu::GpuBackend::new() {
            registry.register(Box::new(gpu_backend));
        }
        
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backend_registry() {
        let registry = BackendRegistry::default();
        let available = registry.available_backends();
        
        // Should have at least CPU backend
        assert!(!available.is_empty());
        assert!(available.iter().any(|b| b.name().contains("CPU")));
    }
    
    #[test]
    fn test_backend_selection() {
        let config = InferenceConfig::default();
        let backend = select_best_backend(&config);
        assert!(backend.is_ok());
    }
}