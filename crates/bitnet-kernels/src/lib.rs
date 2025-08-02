//! High-performance compute kernels for BitNet

use bitnet_common::{QuantizationType, Result};

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;

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

/// Kernel manager for selecting optimal kernels
pub struct KernelManager {
    providers: Vec<Box<dyn KernelProvider>>,
    selected: Option<usize>,
}

impl KernelManager {
    pub fn new() -> Self {
        let mut providers: Vec<Box<dyn KernelProvider>> = vec![
            Box::new(cpu::FallbackKernel),
        ];
        
        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                providers.push(Box::new(cpu::Avx2Kernel));
            }
        }
        
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                providers.push(Box::new(cpu::NeonKernel));
            }
        }
        
        Self {
            providers,
            selected: None,
        }
    }
    
    pub fn select_best(&mut self) -> Result<&dyn KernelProvider> {
        for (i, provider) in self.providers.iter().enumerate() {
            if provider.is_available() {
                self.selected = Some(i);
                return Ok(provider.as_ref());
            }
        }
        Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::NoProvider,
        ))
    }
}