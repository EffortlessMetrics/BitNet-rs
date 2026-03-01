//! Fallback CPU kernel implementation
//!
//! This module provides naive but correct implementations of all kernel operations
//! that work on any architecture. These kernels prioritize correctness over performance
//! and serve as a reference implementation and fallback when optimized kernels are
//! not available.

use crate::KernelProvider;
use bitnet_common::{QuantizationType, Result};
use bitnet_scalar as scalar;

/// Fallback CPU kernel that works on any architecture
///
/// This kernel provides basic implementations of all operations without SIMD
/// optimizations. It's always available and serves as a fallback when
/// architecture-specific optimizations are not supported.
///
/// Performance characteristics:
/// - Matrix multiplication: O(m*n*k) with no vectorization
/// - Quantization: Sequential processing with basic bit packing
/// - Memory access: No cache optimization or prefetching
///
/// Expected use cases:
/// - Unsupported architectures (RISC-V, WASM, etc.)
/// - Development and testing environments
/// - Reference implementation for correctness validation
/// - Fallback when SIMD features are disabled
pub struct FallbackKernel;

impl KernelProvider for FallbackKernel {
    fn name(&self) -> &'static str {
        "fallback"
    }

    fn is_available(&self) -> bool {
        // Fallback kernel is always available
        true
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        scalar::matmul_i2s(a, b, c, m, n, k)
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        scalar::quantize(input, output, scales, qtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_kernel_availability() {
        let kernel = FallbackKernel;
        assert!(kernel.is_available());
        assert_eq!(kernel.name(), "fallback");
    }

    #[test]
    fn test_matmul_i2s_basic() {
        let kernel = FallbackKernel;

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4]; // 2x2 matrix
        let b = vec![1u8, 0, 0, 1]; // 2x2 identity matrix
        let mut c = vec![0.0f32; 4]; // 2x2 result

        kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();

        // Expected result: A * I = A
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_i2s_dimension_validation() {
        let kernel = FallbackKernel;

        let a = vec![1i8, 2];
        let b = vec![1u8, 0];
        let mut c = vec![0.0f32; 4];

        // Wrong dimensions should fail
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_i2s() {
        let kernel = FallbackKernel;

        let input = vec![1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1];
        let mut output = vec![0u8; 2]; // 8 values / 4 per byte = 2 bytes
        let mut scales = vec![0.0f32; 1]; // 8 values / 32 per block = 1 block

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

        // Should have computed a scale
        assert!(scales[0] > 0.0);

        // Output should be non-zero (some values quantized)
        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_quantize_buffer_size_validation() {
        let kernel = FallbackKernel;

        let input = vec![1.0; 32];
        let mut output = vec![0u8; 1]; // Too small
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err());
    }
}
