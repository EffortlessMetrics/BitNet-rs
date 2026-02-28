//! ROCm/HIP kernel surface for AMD GPUs.
//!
//! This module provides a [`KernelProvider`] implementation targeting the AMD
//! ROCm stack via HIP.  The structure mirrors `gpu::cuda` so that a future
//! HIP runtime integration can slot in with minimal refactoring.
//!
//! **Current status — stubs only.**  All kernel entry-points return explicit
//! errors until the HIP runtime bindings are wired in.  The provider is gated
//! behind `--features rocm` and requires `BITNET_ENABLE_ROCM=1` at runtime to
//! be selected by the [`KernelManager`](crate::KernelManager).
//!
//! # Sub-modules
//!
//! | Module | CUDA counterpart | Description |
//! |--------|-----------------|-------------|
//! | [`qk256_gemv`] | `gpu::cuda` matmul | QK256 2-bit GEMV via HIP |
//! | [`attention`] | *(planned)* | Fused multi-head attention |
//! | [`rmsnorm`] | *(planned)* | RMSNorm forward pass |

pub mod attention;
pub mod qk256_gemv;
pub mod rmsnorm;

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

use crate::KernelProvider;

// ── Device information ───────────────────────────────────────────────

/// AMD GPU device information and capabilities (mirrors [`super::gpu::cuda::CudaDeviceInfo`]).
#[derive(Debug, Clone)]
pub struct RocmDeviceInfo {
    /// Ordinal index of the HIP device.
    pub device_id: usize,
    /// Device marketing name (e.g. "AMD Instinct MI250X").
    pub name: String,
    /// GCN architecture name (e.g. "gfx90a").
    pub gcn_arch: String,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Number of compute units.
    pub compute_unit_count: i32,
    /// Maximum wavefront (work-group) size.
    pub max_wavefront_size: i32,
    /// Maximum shared (LDS) memory per work-group in bytes.
    pub max_shared_memory_per_workgroup: usize,
    /// FP16 (half-precision) support.
    pub supports_fp16: bool,
    /// BF16 (bfloat16) support — available on CDNA2+.
    pub supports_bf16: bool,
}

// ── Kernel provider ──────────────────────────────────────────────────

/// ROCm/HIP kernel provider.
///
/// Mirrors [`super::gpu::CudaKernel`] but targets the AMD ROCm/HIP runtime.
/// Until the HIP FFI bindings are integrated every operation returns
/// [`KernelError::ExecutionFailed`].
#[derive(Debug, Clone, Default)]
pub struct RocmKernel {
    _private: (),
}

impl RocmKernel {
    /// Create a new ROCm kernel provider (stub).
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Whether `rocm` support was compiled into this build.
    pub fn compiled() -> bool {
        cfg!(feature = "rocm")
    }

    /// Runtime opt-in via `BITNET_ENABLE_ROCM=1`.
    fn rocm_enabled() -> bool {
        std::env::var("BITNET_ENABLE_ROCM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    fn unavailable_err(&self, op: &str) -> BitNetError {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("ROCm/HIP operation '{op}' is not yet wired to the AMD HIP runtime"),
        })
    }
}

impl KernelProvider for RocmKernel {
    fn name(&self) -> &'static str {
        "rocm-hip"
    }

    fn is_available(&self) -> bool {
        Self::compiled() && Self::rocm_enabled()
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
        Err(self.unavailable_err("matmul_i2s"))
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(self.unavailable_err("quantize"))
    }
}

// ── Utility functions ────────────────────────────────────────────────

/// Check whether a ROCm/HIP runtime is available on the system.
///
/// Stub — always returns `false` until HIP device enumeration is wired in.
pub fn is_rocm_available() -> bool {
    false
}

/// Return the number of HIP-visible AMD GPU devices.
///
/// Stub — always returns `0`.
pub fn rocm_device_count() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_kernel_reports_name() {
        let kernel = RocmKernel::new();
        assert_eq!(kernel.name(), "rocm-hip");
    }

    #[test]
    fn rocm_kernel_not_available_without_env() {
        let kernel = RocmKernel::new();
        // Without BITNET_ENABLE_ROCM=1 the provider must not activate.
        assert!(!kernel.is_available());
    }

    #[test]
    fn rocm_matmul_returns_err() {
        let kernel = RocmKernel::new();
        let a = vec![1i8; 16];
        let b = vec![1u8; 16];
        let mut c = vec![0.0f32; 16];
        assert!(kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 4).is_err());
    }

    #[test]
    fn rocm_quantize_returns_err() {
        let kernel = RocmKernel::new();
        let input = vec![1.0f32; 32];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];
        assert!(kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).is_err());
    }

    #[test]
    fn rocm_device_count_is_zero() {
        assert_eq!(rocm_device_count(), 0);
    }

    #[test]
    fn rocm_is_not_available() {
        assert!(!is_rocm_available());
    }
}
