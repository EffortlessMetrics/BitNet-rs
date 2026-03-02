//! High-performance compute kernels for BitNet

use bitnet_common::{QuantizationType, Result};
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
use bitnet_cpu_detect::avx2_available;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use bitnet_cpu_detect::avx512_available;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use bitnet_cpu_detect::neon_available;
use std::sync::OnceLock;

pub mod activation_bench;
pub mod activation_registry;
pub mod attention_patterns;
pub mod bench_harness;
pub mod benchmarks;
pub mod capability_matrix;
pub mod convolution;
pub mod cpu;
pub mod cuda;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod cuda_graph_capture;
pub mod device_aware;
pub mod device_features;
pub mod dispatch_planner;
pub mod dispatch_registry;
pub mod dispatch_table;
#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "oneapi"))]
pub mod gpu;
pub mod gpu_utils;
pub mod kernel_select;
pub mod kernels;
pub mod matmul_dispatch;
#[cfg(feature = "metal")]
pub mod metal_compute;
pub mod norm_registry;
#[cfg(feature = "npu-backend")]
pub mod npu;
pub mod opencl_attention;
pub mod opencl_buffer;
pub mod opencl_cache;
pub mod opencl_context;
pub mod opencl_continuous_batch;
#[path = "gpu/opencl_dispatch.rs"]
pub mod opencl_dispatch;
pub mod opencl_embedding;
pub mod opencl_engine_bridge;
pub mod opencl_ffn;
pub mod opencl_graph_compiler;
pub mod opencl_kernel_sources;
pub mod opencl_kv_cache;
pub mod opencl_memory;
pub mod opencl_model_converter;
pub mod opencl_pipeline;
pub mod opencl_profiling;
pub mod opencl_quantized;
pub mod opencl_registry;
pub mod opencl_token_gen;
pub mod opencl_transformer;
pub mod opencl_vocab_trie;
pub mod opencl_work_size;
pub mod perf_tracker;
pub mod reduction;
#[cfg(feature = "rocm")]
pub mod rocm;
pub mod rope_freq;
pub mod scatter_gather;
pub mod shaped_reduction;
pub mod simd_detect;
pub mod simd_diagnostics;
pub mod softmax_utils;
mod stubs;
pub mod tl_lut;

/// Kernel provider trait
pub trait KernelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()>;
}

/// Kernel manager for selecting optimal kernels with cached selection
pub struct KernelManager {
    providers: Vec<Box<dyn KernelProvider>>,
    selected: OnceLock<usize>,
}

impl KernelManager {
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

        // Add GPU kernels first (highest priority)
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if let Ok(cuda_kernel) = gpu::CudaKernel::new() {
                if cuda_kernel.is_available() {
                    log::info!("CUDA kernel available, adding to providers");
                    providers.insert(0, Box::new(cuda_kernel));
                }
            } else {
                log::debug!("CUDA kernel not available");
            }
        }

        #[cfg(feature = "npu-backend")]
        {
            let npu_kernel = npu::NpuKernel::new();
            if npu_kernel.is_available() {
                log::info!("NPU kernel available, adding to providers");
                providers.insert(0, Box::new(npu_kernel));
            } else {
                log::debug!("NPU kernel not available");
            }
        }

        #[cfg(feature = "oneapi")]
        {
            if let Ok(opencl_kernel) = gpu::opencl::OpenClKernel::new() {
                if opencl_kernel.is_available() {
                    log::info!("OpenCL kernel available, adding to providers");
                    providers.insert(0, Box::new(opencl_kernel));
                }
            } else {
                log::debug!("OpenCL kernel not available");
            }
        }

        #[cfg(feature = "rocm")]
        {
            let rocm_kernel = rocm::RocmKernel::new();
            if rocm_kernel.is_available() {
                log::info!("ROCm/HIP kernel available, adding to providers");
                providers.insert(0, Box::new(rocm_kernel));
            } else {
                log::debug!("ROCm/HIP kernel not available");
            }
        }

        // Add optimized CPU kernels in order of preference (best first)
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            if avx512_available() {
                let insert_pos = if providers.is_empty() { 0 } else { providers.len() - 1 };
                providers.insert(insert_pos, Box::new(cpu::Avx512Kernel));
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if avx2_available() {
                let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
                providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if neon_available() {
                let insert_pos = if providers.len() > 1 { providers.len() - 1 } else { 0 };
                providers.insert(insert_pos, Box::new(cpu::NeonKernel));
            }
        }

        // Add FFI kernel as a fallback option (lower priority than optimized kernels)
        #[cfg(feature = "ffi")]
        {
            if let Ok(ffi_kernel) = ffi::FfiKernel::new()
                && ffi_kernel.is_available()
            {
                providers.push(Box::new(ffi_kernel));
            }
        }

        Self { providers, selected: OnceLock::new() }
    }

    /// Select the best available kernel provider with caching
    pub fn select_best(&self) -> Result<&dyn KernelProvider> {
        let selected_idx = self.selected.get_or_init(|| {
            // Find the first available provider (they're ordered by preference)
            for (i, provider) in self.providers.iter().enumerate() {
                if provider.is_available() {
                    log::info!("Selected kernel provider: {}", provider.name());
                    return i;
                }
            }
            log::error!("No available kernel provider found");
            // Return fallback kernel index (should always be last and available)
            self.providers.len() - 1
        });

        if *selected_idx < self.providers.len() {
            Ok(self.providers[*selected_idx].as_ref())
        } else {
            Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
        }
    }

    /// Get the name of the currently selected kernel provider
    pub fn selected_provider_name(&self) -> Option<&'static str> {
        self.selected.get().and_then(|&idx| self.providers.get(idx)).map(|provider| provider.name())
    }

    /// List all available kernel providers
    pub fn list_available_providers(&self) -> Vec<&'static str> {
        self.providers
            .iter()
            .filter(|provider| provider.is_available())
            .map(|provider| provider.name())
            .collect()
    }

    /// Force reselection of kernel provider (for testing)
    #[cfg(test)]
    pub fn reset_selection(&mut self) {
        self.selected = OnceLock::new();
    }
}

impl Default for KernelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Select the best CPU kernel provider
pub fn select_cpu_kernel() -> Result<Box<dyn KernelProvider>> {
    #[allow(unused_mut)]
    let mut providers: Vec<Box<dyn KernelProvider>> = vec![Box::new(cpu::FallbackKernel)];

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        if avx512_available() {
            providers.insert(0, Box::new(cpu::Avx512Kernel));
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        if avx2_available() {
            let insert_pos = if providers.is_empty() { 0 } else { providers.len() - 1 };
            providers.insert(insert_pos, Box::new(cpu::Avx2Kernel));
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if neon_available() {
            providers.insert(0, Box::new(cpu::NeonKernel));
        }
    }

    for provider in providers {
        if provider.is_available() {
            return Ok(provider);
        }
    }

    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}

/// Select the best GPU kernel provider
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn select_gpu_kernel(device_id: usize) -> Result<Box<dyn KernelProvider>> {
    let cuda_kernel = gpu::CudaKernel::new_with_device(device_id)?;
    if cuda_kernel.is_available() {
        Ok(Box::new(cuda_kernel))
    } else {
        Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
    }
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub fn select_gpu_kernel(_device_id: usize) -> Result<Box<dyn KernelProvider>> {
    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}

#[cfg(feature = "npu-backend")]
pub fn select_npu_kernel() -> Result<Box<dyn KernelProvider>> {
    let npu_kernel = npu::NpuKernel::new();
    if npu_kernel.is_available() {
        Ok(Box::new(npu_kernel))
    } else {
        Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
    }
}

#[cfg(not(feature = "npu-backend"))]
pub fn select_npu_kernel() -> Result<Box<dyn KernelProvider>> {
    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}

/// Select the ROCm/HIP kernel provider.
#[cfg(feature = "rocm")]
pub fn select_rocm_kernel() -> Result<Box<dyn KernelProvider>> {
    let rocm_kernel = rocm::RocmKernel::new();
    if rocm_kernel.is_available() {
        Ok(Box::new(rocm_kernel))
    } else {
        Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
    }
}

#[cfg(not(feature = "rocm"))]
pub fn select_rocm_kernel() -> Result<Box<dyn KernelProvider>> {
    Err(bitnet_common::BitNetError::Kernel(bitnet_common::KernelError::NoProvider))
}

// Re-export commonly used types
pub use cpu::FallbackKernel;

// Platform-specific kernel re-exports with stubs
#[cfg(target_arch = "x86_64")]
pub use cpu::{Avx2Kernel, Avx512Kernel};

#[cfg(target_arch = "aarch64")]
pub use cpu::NeonKernel;

// Use stub implementations from stubs module for unavailable kernels
#[cfg(not(target_arch = "x86_64"))]
pub use stubs::Avx2Kernel;

pub use device_aware::{DeviceAwareQuantizer, DeviceAwareQuantizerFactory};
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use gpu::CudaKernel;
#[cfg(feature = "oneapi")]
pub use gpu::opencl::OpenClKernel;
#[cfg(feature = "npu-backend")]
pub use npu::NpuKernel;
#[cfg(feature = "rocm")]
pub use rocm::RocmKernel;
#[cfg(not(target_arch = "aarch64"))]
pub use stubs::NeonKernel;
