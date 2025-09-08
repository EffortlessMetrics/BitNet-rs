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

    pub fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: bitnet_common::QuantizationType,
    ) -> Result<(), &'static str> {
        let qtype = match qtype {
            bitnet_common::QuantizationType::I2S => 0,
            bitnet_common::QuantizationType::TL1 => 1,
            bitnet_common::QuantizationType::TL2 => 2,
        };
        crate::ffi::bridge::cpp::quantize(input, output, scales, qtype)
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
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: bitnet_common::QuantizationType,
    ) -> bitnet_common::Result<()> {
        self.quantize(input, output, scales, qtype).map_err(|e| {
            bitnet_common::BitNetError::Kernel(
                bitnet_common::KernelError::UnsupportedArchitecture { arch: e.to_string() },
            )
        })
    }
}

#[cfg(all(feature = "ffi", have_cpp))]
pub use bridge::PerformanceComparison;

#[cfg(any(not(feature = "ffi"), not(have_cpp)))]
pub struct PerformanceComparison;

#[cfg(test)]
mod tests {
    use super::FfiKernel;
    use crate::KernelProvider;
    use crate::cpu::FallbackKernel;
    use bitnet_common::QuantizationType;

    #[test]
    fn ffi_quantize_matches_rust() {
        let ffi_kernel = match FfiKernel::new() {
            Ok(k) => k,
            Err(_) => return, // skip when FFI bridge unavailable
        };

        let fallback = FallbackKernel;

        let input: Vec<f32> = (0..128).map(|i| ((i as f32) % 32.0 - 16.0) / 8.0).collect();
        let output_len = input.len() / 4;
        let scales_len = input.len() / 32; // fits all qtypes

        for &qtype in &[QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
            let mut ffi_out = vec![0u8; output_len];
            let mut ffi_scales = vec![0.0f32; scales_len];
            let mut rust_out = vec![0u8; output_len];
            let mut rust_scales = vec![0.0f32; scales_len];

            if ffi_kernel.quantize(&input, &mut ffi_out, &mut ffi_scales, qtype).is_err() {
                // skip if quantization not implemented
                return;
            }
            fallback.quantize(&input, &mut rust_out, &mut rust_scales, qtype).unwrap();

            assert_eq!(ffi_out, rust_out, "output mismatch for {:?}", qtype);
            assert_eq!(ffi_scales, rust_scales, "scales mismatch for {:?}", qtype);
        }
    }
}
