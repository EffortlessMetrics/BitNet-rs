//! Qualcomm-oriented NPU kernel surface.
//!
//! This module introduces a minimal `KernelProvider` implementation that can be
//! selected by the kernel manager when `npu-backend` support is compiled in.
//! The implementation is intentionally conservative: it checks runtime enablement
//! via environment variables and returns explicit errors until Qualcomm QNN/SNPE
//! bindings are wired in.

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};
use bitnet_qualcomm::{QualcommNpuBackend, npu_requested};

use crate::KernelProvider;

/// NPU kernel provider for Qualcomm SDK integration points.
#[derive(Debug, Clone, Default)]
pub struct NpuKernel {
    backend: QualcommNpuBackend,
}

impl NpuKernel {
    /// Create an NPU kernel provider.
    pub fn new() -> Self {
        Self { backend: QualcommNpuBackend::from_env() }
    }

    /// Whether NPU support was enabled for this build.
    pub fn compiled() -> bool {
        cfg!(feature = "npu-backend")
    }

    fn unavailable_err(&self, op: &str) -> BitNetError {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "NPU operation '{op}' is not yet wired to Qualcomm {} runtime",
                self.backend.runtime_name(),
            ),
        })
    }
}

impl KernelProvider for NpuKernel {
    fn name(&self) -> &'static str {
        self.backend.kernel_name()
    }

    fn is_available(&self) -> bool {
        Self::compiled() && npu_requested()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npu_kernel_reports_name() {
        let kernel = NpuKernel::new();
        assert!(kernel.name().starts_with("npu-"));
    }
}
