//! Stub implementations for kernels not available on the current architecture

use crate::KernelProvider;
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

/// Stub implementation of Avx2Kernel for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub struct Avx2Kernel;

#[cfg(not(target_arch = "x86_64"))]
impl KernelProvider for Avx2Kernel {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn matmul_i2s(
        &self,
        _a: &[i8],
        _b: &[u8],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "AVX2 kernel not available on non-x86_64 architectures".to_string(),
        }))
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "AVX2 kernel not available on non-x86_64 architectures".to_string(),
        }))
    }
}

/// Stub implementation of NeonKernel for non-aarch64 architectures
#[cfg(not(target_arch = "aarch64"))]
pub struct NeonKernel;

#[cfg(not(target_arch = "aarch64"))]
impl KernelProvider for NeonKernel {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn matmul_i2s(
        &self,
        _a: &[i8],
        _b: &[u8],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "NEON kernel not available on non-ARM64 architectures".to_string(),
        }))
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "NEON kernel not available on non-ARM64 architectures".to_string(),
        }))
    }
}
