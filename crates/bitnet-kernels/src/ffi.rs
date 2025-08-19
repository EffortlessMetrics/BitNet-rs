//! FFI bridge for gradual C++ kernel migration
//!
//! This module provides a bridge to existing C++ kernel implementations,
//! allowing for gradual migration from C++ to Rust while maintaining
//! functionality throughout the transition period.

pub mod bridge;

pub struct FfiKernel;

impl FfiKernel {
    pub fn new() -> Result<Self, &'static str> {
        if crate::ffi::bridge::cpp::is_available() {
            crate::ffi::bridge::cpp::init();
            Ok(Self)
        } else {
            Err("ffi bridge unavailable")
        }
    }

    pub fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), &'static str> {
        crate::ffi::bridge::cpp::matmul_i2s(a, b, c, m, n, k)
    }
}

impl Drop for FfiKernel {
    fn drop(&mut self) {
        // On the stub path this is a no-op; on the real path it calls into C++
        crate::ffi::bridge::cpp::cleanup();
    }
}

impl crate::KernelProvider for FfiKernel {
    fn name(&self) -> &'static str {
        "ffi"
    }

    fn is_available(&self) -> bool {
        crate::ffi::bridge::cpp::is_available()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bitnet_common::Result<()> {
        self.matmul_i2s(a, b, c, m, n, k).map_err(|e| {
            bitnet_common::BitNetError::Kernel(
                bitnet_common::KernelError::UnsupportedArchitecture { arch: e.to_string() },
            )
        })
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
                arch: "FFI quantize not implemented".to_string(),
            },
        ))
    }
}

#[cfg(all(feature = "ffi", have_cpp))]
pub use bridge::PerformanceComparison;

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
pub struct PerformanceComparison;
