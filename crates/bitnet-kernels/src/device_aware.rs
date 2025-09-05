//! Device-aware quantization with automatic fallback
//!
//! This module provides device-aware quantization that automatically selects
//! the best available device (GPU vs CPU) and gracefully falls back when needed.

#[cfg(feature = "gpu")]
use crate::gpu;
use crate::{KernelProvider, cpu};
use bitnet_common::{Device, KernelError, QuantizationType, Result};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use sysinfo::System;

/// Device-aware quantization provider with automatic fallback
pub struct DeviceAwareQuantizer {
    primary_provider: Option<Box<dyn KernelProvider>>,
    fallback_provider: Box<dyn KernelProvider>,
    target_device: Device,
    stats: Arc<Mutex<DeviceStatsInternal>>,
}

/// Internal performance statistics with thread-safe tracking
#[derive(Debug, Clone, Default)]
struct DeviceStatsInternal {
    quantization_operations: u64,
    matmul_operations: u64,
    total_quantization_time_ms: f64,
    total_matmul_time_ms: f64,
    gpu_operations: u64,
    cpu_operations: u64,
    fallback_count: u64,
    last_gpu_error: Option<String>,
    last_cpu_error: Option<String>,
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

        Ok(Self {
            primary_provider,
            fallback_provider,
            target_device: device,
            stats: Arc::new(Mutex::new(DeviceStatsInternal::default())),
        })
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

    /// Attempt quantization with automatic fallback and performance tracking
    pub fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        let start_time = Instant::now();

        // Try primary provider (GPU) first
        if let Some(ref primary) = self.primary_provider {
            let gpu_start = Instant::now();
            match primary.quantize(input, output, scales, qtype) {
                Ok(()) => {
                    let elapsed = gpu_start.elapsed().as_secs_f64() * 1000.0;

                    if let Ok(mut stats) = self.stats.lock() {
                        stats.quantization_operations += 1;
                        stats.gpu_operations += 1;
                        stats.total_quantization_time_ms += elapsed;
                    }

                    log::trace!(
                        "GPU quantization succeeded for {} elements in {:.3}ms (qtype: {:?})",
                        input.len(),
                        elapsed,
                        qtype
                    );
                    return Ok(());
                }
                Err(e) => {
                    let error_msg = format!(
                        "GPU quantization failed for {} elements: {} (device: {:?}, qtype: {:?})",
                        input.len(),
                        e,
                        self.target_device,
                        qtype
                    );
                    log::warn!("{}", error_msg);

                    if let Ok(mut stats) = self.stats.lock() {
                        stats.fallback_count += 1;
                        stats.last_gpu_error = Some(error_msg);
                    }
                    // Continue to fallback
                }
            }
        }

        // Use CPU fallback
        let cpu_start = Instant::now();
        log::debug!(
            "Using CPU fallback for quantization (input_len: {}, qtype: {:?})",
            input.len(),
            qtype
        );

        match self.fallback_provider.quantize(input, output, scales, qtype) {
            Ok(()) => {
                let elapsed = cpu_start.elapsed().as_secs_f64() * 1000.0;
                let total_elapsed = start_time.elapsed().as_secs_f64() * 1000.0;

                if let Ok(mut stats) = self.stats.lock() {
                    stats.quantization_operations += 1;
                    stats.cpu_operations += 1;
                    stats.total_quantization_time_ms += total_elapsed;
                }

                log::trace!(
                    "CPU quantization succeeded for {} elements in {:.3}ms (total: {:.3}ms)",
                    input.len(),
                    elapsed,
                    total_elapsed
                );
                Ok(())
            }
            Err(e) => {
                let error_msg = format!(
                    "CPU quantization failed for {} elements: {} (qtype: {:?})",
                    input.len(),
                    e,
                    qtype
                );
                log::error!("{}", error_msg);

                if let Ok(mut stats) = self.stats.lock() {
                    stats.last_cpu_error = Some(error_msg.clone());
                }

                Err(KernelError::QuantizationFailed {
                    reason: format!("Both GPU and CPU quantization failed: {}", error_msg),
                }
                .into())
            }
        }
    }

    /// Matrix multiplication with device awareness and performance tracking
    pub fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let start_time = Instant::now();
        let _used_gpu = false;

        // Try primary provider (GPU) first
        if let Some(ref primary) = self.primary_provider {
            let gpu_start = Instant::now();
            match primary.matmul_i2s(a, b, c, m, n, k) {
                Ok(()) => {
                    let elapsed = gpu_start.elapsed().as_secs_f64() * 1000.0;

                    if let Ok(mut stats) = self.stats.lock() {
                        stats.matmul_operations += 1;
                        stats.gpu_operations += 1;
                        stats.total_matmul_time_ms += elapsed;
                    }

                    log::trace!("GPU matmul succeeded for {}x{}x{} in {:.3}ms", m, n, k, elapsed);
                    return Ok(());
                }
                Err(e) => {
                    let error_msg = format!(
                        "GPU matmul failed for {}x{}x{}: {} (device: {:?})",
                        m, n, k, e, self.target_device
                    );
                    log::warn!("{}", error_msg);

                    if let Ok(mut stats) = self.stats.lock() {
                        stats.fallback_count += 1;
                        stats.last_gpu_error = Some(error_msg);
                    }
                    // Continue to fallback
                }
            }
        }

        // Use CPU fallback
        let cpu_start = Instant::now();
        log::debug!("Using CPU fallback for matrix multiplication ({}x{}x{})", m, n, k);

        match self.fallback_provider.matmul_i2s(a, b, c, m, n, k) {
            Ok(()) => {
                let elapsed = cpu_start.elapsed().as_secs_f64() * 1000.0;
                let total_elapsed = start_time.elapsed().as_secs_f64() * 1000.0;

                if let Ok(mut stats) = self.stats.lock() {
                    stats.matmul_operations += 1;
                    stats.cpu_operations += 1;
                    stats.total_matmul_time_ms += total_elapsed;
                }

                log::trace!(
                    "CPU matmul succeeded for {}x{}x{} in {:.3}ms (total: {:.3}ms)",
                    m,
                    n,
                    k,
                    elapsed,
                    total_elapsed
                );
                Ok(())
            }
            Err(e) => {
                let error_msg = format!("CPU matmul failed for {}x{}x{}: {}", m, n, k, e);
                log::error!("{}", error_msg);

                if let Ok(mut stats) = self.stats.lock() {
                    stats.last_cpu_error = Some(error_msg.clone());
                }

                Err(KernelError::MatmulFailed {
                    reason: format!("Both GPU and CPU matmul failed: {}", error_msg),
                }
                .into())
            }
        }
    }

    /// Force fallback to CPU (for testing or reliability)
    pub fn force_cpu_fallback(&mut self) {
        if self.primary_provider.is_some() {
            log::info!("Forcing CPU fallback, disabling GPU provider");
            self.primary_provider = None;
        }
    }

    /// Get comprehensive performance statistics
    pub fn get_stats(&self) -> Option<DeviceStats> {
        if let Ok(stats) = self.stats.lock() {
            let primary_device_type =
                self.primary_provider.as_ref().map(|p| p.name()).unwrap_or("None");
            let fallback_device_type = self.fallback_provider.name();

            let (memory_used_bytes, memory_total_bytes) = match self.target_device {
                #[cfg(feature = "gpu")]
                Device::Cuda(_) => unsafe {
                    use cudarc::driver::sys::cuMemGetInfo_v2;
                    let mut free: usize = 0;
                    let mut total: usize = 0;
                    let result = cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
                    if result != 0 {
                        log::warn!("Failed to get CUDA memory info: {:?}", result);
                        (0, 0)
                    } else {
                        (total.saturating_sub(free) as u64, total as u64)
                    }
                },
                _ => {
                    let sys = System::new_all();
                    let total = sys.total_memory();
                    let used = sys.used_memory();
                    (used as u64, total as u64)
                }
            };

            log::debug!(
                "Memory usage for {:?}: {}/{} bytes",
                self.target_device,
                memory_used_bytes,
                memory_total_bytes
            );

            Some(DeviceStats {
                device_type: format!("{}+{}", primary_device_type, fallback_device_type),
                target_device: self.target_device,
                total_operations: stats.quantization_operations + stats.matmul_operations,
                quantization_operations: stats.quantization_operations,
                matmul_operations: stats.matmul_operations,
                total_time_ms: stats.total_quantization_time_ms + stats.total_matmul_time_ms,
                quantization_time_ms: stats.total_quantization_time_ms,
                matmul_time_ms: stats.total_matmul_time_ms,
                gpu_operations: stats.gpu_operations,
                cpu_operations: stats.cpu_operations,
                fallback_count: stats.fallback_count,
                gpu_efficiency: if stats.gpu_operations + stats.cpu_operations > 0 {
                    stats.gpu_operations as f64
                        / (stats.gpu_operations + stats.cpu_operations) as f64
                } else {
                    0.0
                },
                last_gpu_error: stats.last_gpu_error.clone(),
                last_cpu_error: stats.last_cpu_error.clone(),
                memory_used_bytes,
                memory_total_bytes,
            })
        } else {
            log::warn!("Failed to acquire stats lock");
            None
        }
    }

    /// Reset performance statistics (useful for benchmarking)
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = DeviceStatsInternal::default();
            log::debug!("Performance statistics reset");
        }
    }
}

/// Performance statistics for device-aware operations
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub device_type: String,
    pub target_device: Device,
    pub total_operations: u64,
    pub quantization_operations: u64,
    pub matmul_operations: u64,
    pub total_time_ms: f64,
    pub quantization_time_ms: f64,
    pub matmul_time_ms: f64,
    pub gpu_operations: u64,
    pub cpu_operations: u64,
    pub fallback_count: u64,
    pub gpu_efficiency: f64, // Ratio of GPU operations to total operations
    pub last_gpu_error: Option<String>,
    pub last_cpu_error: Option<String>,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
}

impl DeviceStats {
    /// Get average quantization time per operation
    pub fn avg_quantization_time_ms(&self) -> f64 {
        if self.quantization_operations > 0 {
            self.quantization_time_ms / self.quantization_operations as f64
        } else {
            0.0
        }
    }

    /// Get average matrix multiplication time per operation
    pub fn avg_matmul_time_ms(&self) -> f64 {
        if self.matmul_operations > 0 {
            self.matmul_time_ms / self.matmul_operations as f64
        } else {
            0.0
        }
    }

    /// Check if GPU is effectively being used (low fallback rate)
    pub fn is_gpu_effective(&self) -> bool {
        self.gpu_operations > 0 && self.gpu_efficiency > 0.8
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Device: {} | Total: {} ops ({} quantize, {} matmul) | Time: {:.2}ms | GPU: {:.1}% | Fallbacks: {}",
            self.device_type,
            self.total_operations,
            self.quantization_operations,
            self.matmul_operations,
            self.total_time_ms,
            self.gpu_efficiency * 100.0,
            self.fallback_count
        )
    }
}

/// Factory for creating device-aware quantizers
pub struct DeviceAwareQuantizerFactory;

impl DeviceAwareQuantizerFactory {
    /// Create the best quantizer for the given device preference
    pub fn create_best(preferred_device: Option<Device>) -> Result<DeviceAwareQuantizer> {
        let device = preferred_device.unwrap_or({
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
                println!("Device stats: {}", stats.summary());
                assert!(stats.total_operations > 0);
                assert!(stats.quantization_operations > 0);
                println!("GPU efficiency: {:.2}%", stats.gpu_efficiency * 100.0);
            }

            // Test matrix multiplication
            let a = vec![1i8; 64];
            let b = vec![255u8; 64];
            let mut c = vec![0.0f32; 16];

            let result = quantizer.matmul_i2s(&a, &b, &mut c, 4, 4, 16);
            assert!(result.is_ok());

            if let Some(stats) = quantizer.get_stats() {
                assert!(stats.matmul_operations > 0);
                println!("Final stats: {}", stats.summary());
            }
        }
    }

    #[test]
    fn test_performance_tracking() {
        let quantizer = DeviceAwareQuantizer::new(Device::Cpu).unwrap();

        // Initially no stats
        if let Some(stats) = quantizer.get_stats() {
            assert_eq!(stats.total_operations, 0);
            assert_eq!(stats.total_time_ms, 0.0);
        }

        // Perform some operations
        let input = vec![1.0f32, -1.0f32, 0.5f32, -0.5f32];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];

        let result = quantizer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());

        // Check stats updated
        if let Some(stats) = quantizer.get_stats() {
            assert_eq!(stats.quantization_operations, 1);
            assert!(stats.total_time_ms > 0.0);
            assert!(stats.cpu_operations > 0);
            println!("CPU quantization stats: {}", stats.summary());
        }

        // Reset stats
        quantizer.reset_stats();
        if let Some(stats) = quantizer.get_stats() {
            assert_eq!(stats.total_operations, 0);
        }
    }

    #[test]
    fn test_memory_stats_cpu() {
        let quantizer = DeviceAwareQuantizer::new(Device::Cpu).unwrap();
        if let Some(stats) = quantizer.get_stats() {
            assert!(stats.memory_total_bytes > 0);
            assert!(stats.memory_used_bytes <= stats.memory_total_bytes);
        } else {
            panic!("No stats returned");
        }
    }
}
