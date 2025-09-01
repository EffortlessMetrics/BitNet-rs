//! Device-aware quantization with automatic fallback
//!
//! This module provides device-aware quantization that automatically selects
//! the best available device (GPU vs CPU) and gracefully falls back when needed.

#[cfg(feature = "cuda")]
use crate::gpu;
use crate::{KernelProvider, cpu};
use bitnet_common::{Device, KernelError, QuantizationType, Result};

/// Device-aware quantization provider with automatic fallback
pub struct DeviceAwareQuantizer {
    primary_provider: Option<Box<dyn KernelProvider>>,
    fallback_provider: Box<dyn KernelProvider>,
    target_device: Device,
}

impl DeviceAwareQuantizer {
    /// Create a new device-aware quantizer for the specified device
    pub fn new(device: Device) -> Result<Self> {
        let (primary_provider, fallback_provider) = match device {
            #[cfg(feature = "gpu")]
            Device::Cuda(device_id) => {
                // Try to create GPU provider
                let gpu_provider = match gpu::CudaKernel::new_with_device(device_id) {
                    Ok(kernel) if kernel.is_available() => {
                        log::info!("GPU quantization available on CUDA device {}", device_id);
                        Some(Box::new(kernel) as Box<dyn KernelProvider>)
                    }
                    Ok(_) => {
                        log::warn!("CUDA device {} not available, falling back to CPU", device_id);
                        None
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to create CUDA kernel for device {}: {}, falling back to CPU",
                            device_id,
                            e
                        );
                        None
                    }
                };

                // Create CPU fallback
                let cpu_provider = Self::create_best_cpu_provider()?;
                (gpu_provider, cpu_provider)
            }
            #[cfg(not(feature = "gpu"))]
            Device::Cuda(_device_id) => {
                log::warn!("CUDA support not compiled, falling back to CPU");
                let cpu_provider = Self::create_best_cpu_provider()?;
                (None, cpu_provider)
            }
            Device::Cpu | Device::Metal => {
                // For CPU and Metal, just use CPU provider
                let cpu_provider = Self::create_best_cpu_provider()?;
                (None, cpu_provider)
            }
        };

        Ok(Self { primary_provider, fallback_provider, target_device: device })
    }

    /// Create the best available CPU provider
    fn create_best_cpu_provider() -> Result<Box<dyn KernelProvider>> {
        // Try optimized CPU kernels first
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                log::debug!("Using AVX2 CPU kernel for fallback");
                return Ok(Box::new(cpu::Avx2Kernel));
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                log::debug!("Using NEON CPU kernel for fallback");
                return Ok(Box::new(cpu::NeonKernel));
            }
        }

        log::debug!("Using fallback CPU kernel");
        Ok(Box::new(cpu::FallbackKernel))
    }

    /// Get the currently active provider name
    pub fn active_provider(&self) -> &'static str {
        self.primary_provider
            .as_ref()
            .map(|p| p.name())
            .unwrap_or_else(|| self.fallback_provider.name())
    }

    /// Check if GPU acceleration is currently active
    pub fn is_gpu_active(&self) -> bool {
        self.primary_provider.is_some()
    }

    /// Get device information
    pub fn device(&self) -> Device {
        self.target_device
    }

    /// Attempt quantization with automatic fallback
    pub fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        // Try primary provider (GPU) first
        if let Some(ref primary) = self.primary_provider {
            match primary.quantize(input, output, scales, qtype) {
                Ok(()) => {
                    log::debug!("GPU quantization succeeded");
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("GPU quantization failed, falling back to CPU: {}", e);
                    // Continue to fallback
                }
            }
        }

        // Use CPU fallback
        log::debug!("Using CPU fallback for quantization");
        self.fallback_provider.quantize(input, output, scales, qtype).map_err(|e| {
            KernelError::QuantizationFailed {
                reason: format!("Both GPU and CPU quantization failed: {}", e),
            }
            .into()
        })
    }

    /// Matrix multiplication with device awareness
    pub fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Try primary provider (GPU) first
        if let Some(ref primary) = self.primary_provider {
            match primary.matmul_i2s(a, b, c, m, n, k) {
                Ok(()) => {
                    log::debug!("GPU matrix multiplication succeeded");
                    return Ok(());
                }
                Err(e) => {
                    log::warn!("GPU matrix multiplication failed, falling back to CPU: {}", e);
                    // Continue to fallback
                }
            }
        }

        // Use CPU fallback
        log::debug!("Using CPU fallback for matrix multiplication");
        self.fallback_provider.matmul_i2s(a, b, c, m, n, k).map_err(|e| {
            KernelError::MatmulFailed { reason: format!("Both GPU and CPU matmul failed: {}", e) }
                .into()
        })
    }

    /// Force fallback to CPU (for testing or reliability)
    pub fn force_cpu_fallback(&mut self) {
        if self.primary_provider.is_some() {
            log::info!("Forcing CPU fallback, disabling GPU provider");
            self.primary_provider = None;
        }
    }

    /// Get performance statistics if available
    pub fn get_stats(&self) -> Option<DeviceStats> {
        // For now, return basic stats based on device type
        if self.primary_provider.is_some() {
            Some(DeviceStats {
                device_type: self.primary_provider.as_ref().unwrap().name().to_string(),
                total_operations: 0, // Would need to track this
                total_time_ms: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
            })
        } else {
            Some(DeviceStats {
                device_type: self.fallback_provider.name().to_string(),
                total_operations: 0,
                total_time_ms: 0.0,
                memory_used_bytes: 0,
                memory_total_bytes: 0,
            })
        }
    }
}

/// Performance statistics for device-aware operations
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub device_type: String,
    pub total_operations: u64,
    pub total_time_ms: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
}

/// Factory for creating device-aware quantizers
pub struct DeviceAwareQuantizerFactory;

impl DeviceAwareQuantizerFactory {
    /// Create the best quantizer for the given device preference
    pub fn create_best(preferred_device: Option<Device>) -> Result<DeviceAwareQuantizer> {
        let device = preferred_device.unwrap_or_else(|| {
            // Auto-detect best device
            #[cfg(feature = "gpu")]
            {
                if gpu::is_cuda_available() && gpu::cuda_device_count() > 0 {
                    Device::Cuda(0)
                } else {
                    Device::Cpu
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                Device::Cpu
            }
        });

        DeviceAwareQuantizer::new(device)
    }

    /// Create a quantizer with automatic GPU detection
    pub fn auto_detect() -> Result<DeviceAwareQuantizer> {
        Self::create_best(None)
    }

    /// List available devices
    pub fn list_available_devices() -> Vec<Device> {
        #[cfg(feature = "gpu")]
        {
            let mut devices = vec![Device::Cpu];
            let cuda_count = gpu::cuda_device_count();
            for i in 0..cuda_count {
                devices.push(Device::Cuda(i));
            }
            devices
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            vec![Device::Cpu]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_aware_creation() {
        // Test CPU creation (should always work)
        let quantizer = DeviceAwareQuantizer::new(Device::Cpu);
        assert!(quantizer.is_ok());

        let quantizer = quantizer.unwrap();
        assert_eq!(quantizer.device(), Device::Cpu);
        assert!(!quantizer.is_gpu_active());
    }

    #[test]
    fn test_factory() {
        let quantizer = DeviceAwareQuantizerFactory::auto_detect();
        assert!(quantizer.is_ok());

        let devices = DeviceAwareQuantizerFactory::list_available_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&Device::Cpu));
    }

    #[test]
    fn test_quantization_fallback() {
        let quantizer = DeviceAwareQuantizer::new(Device::Cpu).unwrap();

        // Test with small input
        let input = vec![1.0f32, -1.0f32, 0.5f32, -0.5f32];
        let mut output = vec![0u8; 1]; // 4 values pack into 1 byte
        let mut scales = vec![0.0f32; 1];

        let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_device_aware() {
        if gpu::is_cuda_available() {
            let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0)).unwrap();
            println!("Active provider: {}", quantizer.active_provider());

            // Test quantization
            let input = vec![1.0f32; 1024];
            let mut output = vec![0u8; 256]; // 1024/4 = 256 bytes
            let mut scales = vec![0.0f32; 8]; // 1024/128 = 8 blocks

            let result =
                quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
            assert!(result.is_ok());

            if let Some(stats) = quantizer.get_stats() {
                println!("Device stats: {:?}", stats);
            }
        }
    }
}
