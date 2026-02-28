//! # Inference Backends
//!
//! CPU and GPU backend implementations for BitNet inference with
//! automatic backend selection and fallback support.

use anyhow::Result;
use async_trait::async_trait;
use bitnet_common::{ConcreteTensor, Device, Tensor};
use bitnet_models::Model;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::cache::KVCache;

const NPU_ENABLE_ENV: &str = "BITNET_ENABLE_NPU";
const NPU_FALLBACK_ENV: &str = "BITNET_NPU_ALLOW_FALLBACK";

/// Trait for inference backends
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get backend type name
    fn backend_type(&self) -> String;

    /// Clone the backend (for sharing across threads)
    fn clone_backend(&self) -> Box<dyn Backend>;

    /// Perform forward pass through the model
    async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor>;

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

        Ok(Self { model, num_threads })
    }

    /// Create CPU backend with specific thread count
    pub fn with_threads(model: Arc<dyn Model>, num_threads: usize) -> Result<Self> {
        info!("Created CPU backend with {} threads", num_threads);

        Ok(Self { model, num_threads })
    }
}

#[async_trait]
impl Backend for CpuBackend {
    fn backend_type(&self) -> String {
        "cpu".to_string()
    }

    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self { model: self.model.clone(), num_threads: self.num_threads })
    }

    async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor> {
        debug!("CPU forward pass with input shape: {:?}", input.shape());

        // Set thread count for this operation
        // Ignore errors if the global thread pool has already been initialized
        let _ = rayon::ThreadPoolBuilder::new().num_threads(self.num_threads).build_global();

        let model = self.model.clone();
        let input_tensor = input.clone();

        // Use block_in_place to allow blocking the current thread with computation
        // while properly passing the mutable cache reference.
        // This avoids the 'static lifetime requirement of spawn_blocking.
        let output = tokio::task::block_in_place(move || {
            let cache_any: &mut dyn std::any::Any = cache;
            model.forward(&input_tensor, cache_any)
        })?;

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
        let mut dummy_cache = KVCache::new(Default::default())?;

        // Perform a dummy forward pass
        let _ = self.forward(&dummy_input, &mut dummy_cache).await?;

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

/// NPU backend implementation.
///
/// The current implementation routes execution through the model's Metal path and
/// provides an explicit backend surface for NPU-oriented deployment policy,
/// warm-up behavior, and capability reporting.
pub struct NpuBackend {
    model: Arc<dyn Model>,
    device: Device,
    allow_cpu_fallback: bool,
}

impl NpuBackend {
    /// Create a new NPU backend.
    pub fn new(model: Arc<dyn Model>, device: Device) -> Result<Self> {
        if !Self::is_available() {
            return Err(anyhow::anyhow!(
                "NPU backend unavailable. Set {NPU_ENABLE_ENV}=1 and compile with metal support on macOS"
            ));
        }

        if !matches!(device, Device::Metal) {
            return Err(anyhow::anyhow!("NPU backend currently requires Device::Metal"));
        }

        let allow_cpu_fallback = std::env::var(NPU_FALLBACK_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        info!(
            "Created NPU backend (device={:?}, allow_cpu_fallback={})",
            device, allow_cpu_fallback
        );

        Ok(Self { model, device, allow_cpu_fallback })
    }

    /// Check if NPU backend is available in current build/runtime.
    pub fn is_available() -> bool {
        let enabled = std::env::var(NPU_ENABLE_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        enabled && cfg!(target_os = "macos")
    }

    fn ensure_npu_tensor(&self, input: &ConcreteTensor) -> Result<ConcreteTensor> {
        if input.device() == &self.device {
            return Ok(input.clone());
        }

        // Best-effort transfer placeholder: until dedicated NPU kernels are wired,
        // we keep shape fidelity and allow runtime fallback.
        if self.allow_cpu_fallback {
            debug!(
                "NPU tensor transfer unavailable, using shape-preserving tensor fallback from {:?} to {:?}",
                input.device(),
                self.device
            );
            return Ok(ConcreteTensor::mock(input.shape().to_vec()));
        }

        Err(anyhow::anyhow!(
            "NPU tensor transfer is not available and fallback is disabled via {NPU_FALLBACK_ENV}=0"
        ))
    }
}

impl GpuBackend {
    /// Create a new GPU backend
    pub fn new(model: Arc<dyn Model>, device: Device) -> Result<Self> {
        if !Self::is_available() {
            return Err(anyhow::anyhow!(
                "GPU backend requires gpu or cuda feature to be compiled in"
            ));
        }
        match device {
            Device::Cuda(_) => {
                info!("Created GPU backend with device: {:?}", device);
                Ok(Self { model, device, mixed_precision: false })
            }
            _ => Err(anyhow::anyhow!("GPU backend requires CUDA or Metal device")),
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
        cfg!(any(feature = "gpu", feature = "cuda", all(feature = "metal", target_os = "macos")))
    }
}

#[async_trait]
impl Backend for NpuBackend {
    fn backend_type(&self) -> String {
        "npu_ane".to_string()
    }

    fn clone_backend(&self) -> Box<dyn Backend> {
        Box::new(Self {
            model: self.model.clone(),
            device: self.device,
            allow_cpu_fallback: self.allow_cpu_fallback,
        })
    }

    async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor> {
        debug!("NPU forward pass with input shape: {:?}", input.shape());

        let npu_input = self.ensure_npu_tensor(input)?;
        let model = self.model.clone();

        let output = tokio::task::block_in_place(move || {
            let cache_any: &mut dyn std::any::Any = cache;
            model.forward(&npu_input, cache_any)
        })?;

        debug!("NPU forward pass completed");
        Ok(output)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_mixed_precision: true,
            supports_batching: true,
            max_batch_size: 16,
            memory_efficient: true,
        }
    }

    async fn warmup(&self) -> Result<()> {
        debug!("Warming up NPU backend");
        let dummy_input = ConcreteTensor::mock(vec![1, 512]);
        let mut dummy_cache = KVCache::new(Default::default())?;
        let _ = self.forward(&dummy_input, &mut dummy_cache).await?;
        info!("NPU backend warmed up successfully");
        Ok(())
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
            device: self.device,
            mixed_precision: self.mixed_precision,
        })
    }

    async fn forward(&self, input: &ConcreteTensor, cache: &mut KVCache) -> Result<ConcreteTensor> {
        debug!("GPU forward pass with input shape: {:?}", input.shape());

        // Ensure input is on the correct device
        let gpu_input = self.ensure_gpu_tensor(input)?;

        let model = self.model.clone();
        let input_tensor = gpu_input;

        // Use block_in_place for GPU backend as well to properly handle cache
        let output = tokio::task::block_in_place(move || {
            let cache_any: &mut dyn std::any::Any = cache;
            model.forward(&input_tensor, cache_any)
        })?;

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
        let mut dummy_cache = KVCache::new(Default::default())?;

        // Perform a dummy forward pass
        let _ = self.forward(&dummy_input, &mut dummy_cache).await?;

        info!("GPU backend warmed up successfully");
        Ok(())
    }
}

impl GpuBackend {
    /// Ensure tensor is on GPU device
    fn ensure_gpu_tensor(&self, input: &ConcreteTensor) -> Result<ConcreteTensor> {
        // In a real implementation, this would transfer the tensor to GPU
        // For now, just create a mock GPU tensor
        Ok(ConcreteTensor::mock(input.shape().to_vec()))
    }
}

/// Automatic backend selection
pub fn select_backend(
    model: Arc<dyn Model>,
    preferred_device: Option<Device>,
) -> Result<Box<dyn Backend>> {
    let device = preferred_device
        .unwrap_or_else(|| if GpuBackend::is_available() { Device::Cuda(0) } else { Device::Cpu });

    match device {
        Device::Cpu => {
            info!("Selected CPU backend");
            Ok(Box::new(CpuBackend::new(model)?))
        }
        Device::Metal => {
            if NpuBackend::is_available() {
                info!("Selected NPU backend");
                Ok(Box::new(NpuBackend::new(model, device)?))
            } else if GpuBackend::is_available() {
                info!("Selected GPU backend for Metal device");
                Ok(Box::new(GpuBackend::new(model, device)?))
            } else {
                warn!("Metal/NPU requested but unavailable, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
        Device::Cuda(_) => {
            if GpuBackend::is_available() {
                info!("Selected GPU backend");
                Ok(Box::new(GpuBackend::new(model, device)?))
            } else {
                warn!("GPU requested but not available, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
        Device::Hip(_) => {
            if GpuBackend::is_available() {
                info!("Selected GPU backend (HIP/ROCm)");
                Ok(Box::new(GpuBackend::new(model, device)?))
            } else {
                warn!("HIP requested but not available, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
        Device::Npu => {
            if NpuBackend::is_available() {
                info!("Selected NPU backend");
                Ok(Box::new(NpuBackend::new(model, device)?))
            } else {
                warn!("NPU requested but not available, falling back to CPU");
                Ok(Box::new(CpuBackend::new(model)?))
            }
        }
        Device::OpenCL(_) => {
            if bitnet_kernels::device_features::oneapi_available_runtime() {
                info!(
                    "OpenCL selected; compute kernels dispatched via KernelManager, tensor ops on CPU"
                );
            } else {
                warn!("OpenCL requested but oneapi runtime not available, falling back to CPU");
            }
            // OpenCL compute kernels are dispatched via KernelManager inside
            // quantized linear layers; the high-level backend uses CPU tensors.
            Ok(Box::new(CpuBackend::new(model)?))
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
            Self { config: bitnet_common::BitNetConfig::default() }
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

        fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 10, 768]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }
    }

    #[tokio::test]
    async fn test_cpu_backend_creation() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model);
        assert!(backend.is_ok());
    }

    // Forward pass test using MockModel (no real weights needed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cpu_backend_forward() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();
        let input = ConcreteTensor::mock(vec![1, 512]);
        let mut cache = KVCache::new(Default::default()).unwrap();

        let output = backend
            .forward(&input, &mut cache)
            .await
            .expect("CPU forward should succeed with mock model");
        // Minimal invariant: output tensor has expected shape
        // MockModel::forward returns [1, vocab_size] = [1, 50257] (logits shape)
        assert_eq!(output.shape(), &[1, 50257], "unexpected output shape");
    }

    // Requires a CUDA/Metal/WGPU environment; off by default.
    #[cfg_attr(not(feature = "gpu-tests"), ignore)]
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
    fn test_npu_backend_not_enabled_by_default() {
        assert!(!NpuBackend::is_available());
    }

    #[test]
    fn test_metal_request_falls_back_without_npu() {
        let model = Arc::new(MockModel::new());

        let backend = select_backend(model, Some(Device::Metal)).expect("backend selection");
        let backend_type = backend.backend_type();
        assert!(
            backend_type == "cpu" || backend_type.contains("gpu"),
            "expected CPU/GPU fallback when NPU disabled, got {}",
            backend_type
        );
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_backend_warmup() {
        let model = Arc::new(MockModel::new());
        let backend = CpuBackend::new(model).unwrap();

        let result = backend.warmup().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_tensor() {
        use crate::TensorDeviceExt;
        let tensor = ConcreteTensor::mock(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), candle_core::DType::F32);
        assert_eq!(tensor.device(), &Device::Cpu);

        let tensor_gpu = tensor.with_device(Device::Cuda(0)).unwrap();
        // Determine expected device based on runtime capabilities
        let expected =
            Device::from(&Device::Cuda(0).to_candle().unwrap_or(candle_core::Device::Cpu));
        assert_eq!(tensor_gpu.device(), &expected);
    }

    /// Test that backend_type() returns correct string identifiers
    /// Kills 1 mutation survivor in backends.rs:188 (empty string return for GPU backend)
    #[test]
    fn test_backend_type_identifiers() {
        let model = Arc::new(MockModel::new());

        // Kill survivor: CPU backend should return exactly "cpu" (not empty string)
        let cpu_backend = CpuBackend::new(model.clone()).unwrap();
        let cpu_type = cpu_backend.backend_type();
        assert_eq!(cpu_type, "cpu", "CPU backend should return 'cpu', got '{}'", cpu_type);
        assert!(!cpu_type.is_empty(), "CPU backend type should not be empty");

        // Kill survivor: GPU backend should return valid device identifier (not empty string)
        #[cfg(feature = "gpu")]
        {
            if GpuBackend::is_available() {
                let gpu_backend = GpuBackend::new(model.clone(), Device::Cuda(0)).unwrap();
                let gpu_type = gpu_backend.backend_type();
                assert!(!gpu_type.is_empty(), "GPU backend type should not be empty");
                assert!(
                    gpu_type.contains("gpu"),
                    "GPU backend type should contain 'gpu', got '{}'",
                    gpu_type
                );
                assert!(
                    gpu_type.contains("Cuda"),
                    "GPU backend type should contain device info, got '{}'",
                    gpu_type
                );
            }
        }

        // Validate backend_type for auto-selected backend
        let auto_backend = select_backend(model.clone(), None).unwrap();
        let auto_type = auto_backend.backend_type();
        assert!(!auto_type.is_empty(), "Auto-selected backend type should not be empty");
        assert!(
            auto_type == "cpu" || auto_type.contains("gpu"),
            "Auto-selected backend type should be 'cpu' or contain 'gpu', got '{}'",
            auto_type
        );
    }
}
