//! High-performance compute kernels for BitNet

use bitnet_common::{QuantizationType, Result};
use std::sync::OnceLock;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod ffi;

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
        let providers: Vec<Box<dyn KernelProvider>> = vec![
            Box::new(cpu::FallbackKernel),
        ];
        
        // Add optimized kernels in order of preference (best first)
        // Note: AVX-512 is disabled due to unstable Rust features
        // #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        // {
        //     if is_x86_feature_detected!("avx512f") {
        //         providers.insert(0, Box::new(cpu::Avx512Kernel));
        //     }
        // }
        
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                providers.insert(0, Box::new(cpu::Avx2Kernel));
            }
        }
        
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                providers.insert(0, Box::new(cpu::NeonKernel));
            }
        }
        
        // Add FFI kernel as a fallback option (lower priority than optimized kernels)
        #[cfg(feature = "ffi-bridge")]
        {
            if let Ok(ffi_kernel) = ffi::FfiKernel::new() {
                if ffi_kernel.is_available() {
                    providers.push(Box::new(ffi_kernel));
                }
            }
        }
        
        Self {
            providers,
            selected: OnceLock::new(),
        }
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
            Err(bitnet_common::BitNetError::Kernel(
                bitnet_common::KernelError::NoProvider,
            ))
        }
    }
    
    /// Get the name of the currently selected kernel provider
    pub fn selected_provider_name(&self) -> Option<&'static str> {
        self.selected.get()
            .and_then(|&idx| self.providers.get(idx))
            .map(|provider| provider.name())
    }
    
    /// List all available kernel providers
    pub fn list_available_providers(&self) -> Vec<&'static str> {
        self.providers.iter()
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