//! # Inference Backends
//!
//! CPU and GPU backend implementations for BitNet inference with
//! automatic backend selection and fallback support.

use anyhow::{Result, Context};
use async_trait::async_trait;
use bitnet_common::{Device, Tensor, ConcreteTensor};
use bitnet_models::Model;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::cache::KVCache;

/// Trait for inference backends
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get backend type name
    fn backend_type(&self) -> String;

    /// Clone the backend (for sharing across threads)
    fn clone_backend(&self) -> Box<dyn Backend>;

    /// Perform forward pass through the model
    async fn forward(
        &self,
        input: &ConcreteTensor,
        cache: Arc<RwLock<KVCache>>,
    ) -> Result<ConcreteTensor>;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    /// Warm up the backend (optional)
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub supports_mixed_precision: bool,
    pub supports_batching: bool,
    pub max_batch_size: usize,
    pub memory_efficient: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_mixed_precision: false,
            supports_batching: true,
            max_batch_size: 1,
            memory_efficient: true,
        }
    }
}

/// CPU backend implementation
pub struct CpuBackend {
    model: Arc<dyn Model>,
    num_threads: usize,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new(model: Arc<dyn Model>) -> Result<Self> {
        let num_threads = num_cpus::get();
        info!("Created CPU backend with {} threads", num_threads);
        
        Ok(Self {
            model,
            num_threads,
        })
    }

    /// Create CPU backend with specific thread count
    pub fn with_threads(model: Arc<dyn Model>, num_threads: usize) -> Result<Self> {
        info!("Created CPU backend with {} threads", num_threads);
        
        Ok(Self {
            model,
            num_threads,
        })
    }
}

#[async_trait]
impl Backend for CpuBackend {
    fn backend_type(&self) -> String {
        "cpu".to_string()
    }

    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self {
            model: self.model.clone(),
            num_threads: self.num_threads,
        })
    }

    async fn forward(
        &self,
        input: &ConcreteTensor,
        cache: Arc<RwLock<KVCache>>,
    ) -> Result<ConcreteTensor> {
        debug!("CPU forward pass with input shape: {:?}", input.shape());
        
        // Set thread count for this operation
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build_global()
            .context("Failed to set thread pool")?;
        
        // Forward pass through model
        let output = tokio::task::spawn_blocking({
            let model = self.model.clone();
            let input_tensor = input.clone();
            let cache = cache.clone();
            move || {
                let mut cache_guard = cache.blocking_write();
                model.forward(&input_tensor, &mut *cache_guard)
            }
        }).await
        .context("CPU forward pass task failed")??;
        
        debug!("CPU forward pass completed");
        Ok(output)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_mixed_precision: false,
            supports_batching: true,
            max_batch_size: 8,
            memory_efficient: true,
        }
    }

    async fn warmup(&self) -> Result<()> {
        debug!("Warming up CPU backend");
        
        // Create a dummy input for warmup
        let dummy_input = ConcreteTensor::mock(vec![1, 512]);
        let dummy_cache = Arc::new(RwLock::new(KVCache::new(Default::default())?));
        
        // Perform a dummy forward pass
        let _ = self.forward(&dummy_input, dummy_cache).await?;
        
        info!("CPU backend warmed up successfully");
        Ok(())
    }
}

/// GPU backend implementation
pub struct GpuBackend {
    model: Arc<dyn Model>,
    device: Device,
    mixed_precision: bool,
}

impl GpuBackend {
    /// Create a new GPU backend
    pub fn new(model: Arc<dyn Model>, device: Device) -> Result<Self> {
        match device {
            Device::Cuda(_) => {
                info!("Created GPU backend with device: {:?}", device);
                Ok(Self {
                    model,
                    device,
                    mixed_precision: false,
                })
            }
            _ => Err(anyhow::anyhow!("GPU backend requires CUDA device")),
        }
    }

    /// Create GPU backend with mixed precision
    pub fn with_mixed_precision(
        model: Arc<dyn Model>,
        device: Device,
        mixed_precision: bool,
    ) -> Result<Self> {
        let mut backend = Self::new(model, device)?;
        backend.mixed_precision = mixed_precision;
        
        if mixed_precision {
            info!("GPU backend configured with mixed precision");
        }
        
        Ok(backend)
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        // In a real implementation, this would check for GPU availability
        cfg!(feature = "gpu")
    }
}

#[async_trait]
impl Backend for GpuBackend {
    fn backend_type(&self) -> String {
        format!("gpu_{:?}", self.device)
    }

    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self {
            model: self.model.clone(),
            device: self.device.clone(),
            mixed_precision: self.mixed_precision,
        })
    }

    async fn forward(
        &self,
        input: &ConcreteTensor,
        cache: Arc<RwLock<KVCache>>,
    ) -> Result<ConcreteTensor> {
        debug!("GPU forward pass with input shape: {:?}", input.shape());
        
        // Ensure input is on the correct device
        let gpu_input = self.ensure_gpu_tensor(input)?;
        
        // Forward pass through model
        let output = tokio::task::spawn_blocking({
            let model = self.model.clone();
            let input_tensor = gpu_input;
            let cache = cache.clone();
            move || {
                let mut cache_guard = cache.blocking_write();
                model.forward(&input_tensor, &mut *cache_guard)
            }
        }).await
        .context("GPU forward pass task failed")??;
        
        debug!("GPU forward pass completed");
        Ok(output)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_mixed_precision: true,
            supports_batching: true,
            max_batch_size: 32,
            memory_efficient: false, // GPU uses more memory but is faster
        }
    }

    async fn warmup(&self) -> Result<()> {
        debug!("Warming up GPU backend");
        
        if !Self::is_available() {
            warn!("CUDA not available, GPU warmup skipped");
            return Ok(());
        }
        
        // Create a dummy input for warmup
        let dummy_input = ConcreteTensor::mock(vec![1, 512]);
        let dummy_cache = Arc::new(RwLock::new(KVCache::new(Default::default())?));
        
        // Perform a dummy forward pass
        let _ = self.forward(&dummy_input, dummy_cache).await?;
        
        info!("GPU backend warmed up successfully");
        Ok(())
    }
}

impl GpuBackend {
    /// Ensure tensor is on GPU device
    fn ensure_gpu_tensor(&self, input: &ConcreteTensor) -> Result<ConcreteTensor> {
        let candle_device = self.device_to_candle(&self.device)?;
        let candle_tensor = input.to_candle()?;
        let gpu_tensor = candle_tensor.to_device(&candle_device)
            .map_err(|e| anyhow::anyhow!("Failed to move tensor to GPU: {}", e))?;
        Ok(ConcreteTensor::bitnet(gpu_tensor))
    }

    fn device_to_candle(&self, device: &Device) -> Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => {
                #[cfg(feature = "gpu")]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda_device = candle_core::CudaDevice::new(*id)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    Ok(candle_core::Device::Cuda(cuda_device))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow::anyhow!("CUDA not available"))
                }
            }
            Device::Metal => {
                #[cfg(feature = "gpu")]
                {
                    Ok(candle_core::Device::Metal(
                        candle_core::MetalDevice::new(0)
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    ))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow::anyhow!("Metal not available"))
                }
            }
        }
    }
}

/// Automatic backend selection
pub fn select_backend(
    model: Arc<dyn Model>,
    preferred_device: Option<Device>,
) -> Result<Box<dyn Backend>> {
    let device = preferred_device.unwrap_or_else(|| {
        if GpuBackend::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    });

    match device {
        Device::Cpu => {
            info!("Selected CPU backend");
            Ok(Box::new(CpuBackend::new(model)?))
        }
        Device::Cuda(_) | Device::Metal => {
            if GpuBackend::is_available() {
                info!("Selected GPU backend");
                Ok(Box::new(GpuBackend::new(model, device)?))
            } else {
                warn!("GPU requested but not available, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
    }
}

// MockTensor is now defined in bitnet_common

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockModel {
        config: bitnet_common::BitNetConfig,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                config: bitnet_common::BitNetConfig::default(),
            }
        }
    }

    impl Model for MockModel {
        fn config(&self) -> &bitnet_common::BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn std::any::Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }
    }

    #[tokio::test]
    async fn test_cpu_backend_creation() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model);
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_cpu_backend_forward() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        let input = ConcreteTensor::mock(vec![1, 512]);
        let cache = Arc::new(RwLock::new(KVCache::new(Default::default()).unwrap()));

        let output = backend.forward(&input, cache).await;
        assert!(output.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_backend_creation() {
        let model = Arc::new(MockModel::new());
        let device = Device::Cuda(0);
        
        // This will fail if CUDA is not available, which is expected
        let backend = GpuBackend::new(model, device);
        
        if GpuBackend::is_available() {
            assert!(backend.is_ok());
        } else {
            assert!(backend.is_err());
        }
    }

    #[test]
    fn test_backend_selection() {
        let model = Arc::new(MockModel::new());
        
        // Test CPU selection
        let backend = select_backend(model.clone(), Some(Device::Cpu));
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().backend_type(), "cpu");
        
        // Test automatic selection
        let backend = select_backend(model, None);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_backend_capabilities() {
        let model = Arc::new(MockModel::new());
        let cpu_backend = CpuBackend::new(model.clone()).unwrap();
        let cpu_caps = cpu_backend.capabilities();
        
        assert!(!cpu_caps.supports_mixed_precision);
        assert!(cpu_caps.supports_batching);
        assert!(cpu_caps.memory_efficient);
        
        if GpuBackend::is_available() {
            let gpu_backend = GpuBackend::new(model, Device::Cuda(0)).unwrap();
            let gpu_caps = gpu_backend.capabilities();
            
            assert!(gpu_caps.supports_mixed_precision);
            assert!(gpu_caps.supports_batching);
            assert!(!gpu_caps.memory_efficient);
        }
    }

    #[tokio::test]
    async fn test_backend_warmup() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        
        let result = backend.warmup().await;
        assert!(result.is_ok());
    }

}