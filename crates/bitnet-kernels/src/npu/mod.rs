//! Qualcomm-oriented NPU kernel surface.
//!
//! This module introduces a minimal `KernelProvider` implementation that can be
//! selected by the kernel manager when `npu-backend` support is compiled in.
//! The implementation is intentionally conservative: it checks runtime enablement
//! via environment variables and returns explicit errors until Qualcomm QNN/SNPE
//! bindings are wired in.

use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

use crate::KernelProvider;

/// NPU kernel provider for Qualcomm SDK integration points.
#[derive(Debug, Clone, Default)]
pub struct NpuKernel {
    backend: NpuBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpuBackend {
    Qnn,
    Snpe,
}

impl Default for NpuBackend {
    fn default() -> Self {
        match std::env::var("BITNET_NPU_BACKEND") {
            Ok(value) if value.eq_ignore_ascii_case("snpe") => Self::Snpe,
            _ => Self::Qnn,
        }
    }
}

impl NpuKernel {
    /// Create an NPU kernel provider.
    pub fn new() -> Self {
        Self { backend: NpuBackend::default() }
    }

    /// Whether NPU support was enabled for this build.
    pub fn compiled() -> bool {
        cfg!(feature = "npu-backend")
    }

    fn npu_enabled() -> bool {
        std::env::var("BITNET_ENABLE_NPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    fn unavailable_err(&self, op: &str) -> BitNetError {
        BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "NPU operation '{op}' is not yet wired to Qualcomm {} runtime",
                match self.backend {
                    NpuBackend::Qnn => "QNN",
                    NpuBackend::Snpe => "SNPE",
                }
            ),
        })
    }
}

impl KernelProvider for NpuKernel {
    fn name(&self) -> &'static str {
        match self.backend {
            NpuBackend::Qnn => "npu-qnn",
            NpuBackend::Snpe => "npu-snpe",
        }
    }

    fn is_available(&self) -> bool {
        Self::compiled() && Self::npu_enabled()
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
