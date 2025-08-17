//! FFI bridge for gradual C++ kernel migration
//!
//! This module provides a bridge to existing C++ kernel implementations,
//! allowing for gradual migration from C++ to Rust while maintaining
//! functionality throughout the transition period.

#[cfg(feature = "ffi-bridge")]
pub mod bridge;

#[cfg(feature = "ffi-bridge")]
pub use bridge::{FfiKernel, PerformanceComparison};

// Stub implementation when FFI bridge is disabled
#[cfg(not(feature = "ffi-bridge"))]
pub struct FfiKernel;

#[cfg(not(feature = "ffi-bridge"))]
impl crate::KernelProvider for FfiKernel {
    fn name(&self) -> &'static str {
        "ffi"
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
    ) -> bitnet_common::Result<()> {
        Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::UnsupportedArchitecture {
                arch: "FFI bridge not enabled".to_string(),
            },
        ))
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: bitnet_common::QuantizationType,
    ) -> bitnet_common::Result<()> {
        Err(bitnet_common::BitNetError::Kernel(
            bitnet_common::KernelError::UnsupportedArchitecture {
                arch: "FFI bridge not enabled".to_string(),
            },
        ))
    }
}
