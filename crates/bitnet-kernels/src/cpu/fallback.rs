//! Fallback CPU kernel implementation
//!
//! This module provides naive but correct implementations of all kernel operations
//! that work on any architecture. These kernels prioritize correctness over performance
//! and serve as a reference implementation and fallback when optimized kernels are
//! not available.

use crate::KernelProvider;
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

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
        // Validate input dimensions
        if a.len() != m * k {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix A dimension mismatch: expected {}, got {}", m * k, a.len()),
            }));
        }
        if b.len() != k * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix B dimension mismatch: expected {}, got {}", k * n, b.len()),
            }));
        }
        if c.len() != m * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix C dimension mismatch: expected {}, got {}", m * n, c.len()),
            }));
        }

        // Initialize output matrix to zero
        c.fill(0.0);

        // Naive matrix multiplication: C = A * B
        // A is m x k (i8), B is k x n (u8), C is m x n (f32)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_val = a[i * k + l] as f32;
                    let b_val = b[l * n + j] as f32;
                    sum += a_val * b_val;
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        match qtype {
            QuantizationType::I2S => self.quantize_i2s(input, output, scales),
            QuantizationType::TL1 => self.quantize_tl1(input, output, scales),
            QuantizationType::TL2 => self.quantize_tl2(input, output, scales),
        }
    }
}

impl FallbackKernel {
    /// I2_S quantization: 2-bit signed quantization with scale factors
    fn quantize_i2s(&self, input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
        const BLOCK_SIZE: usize = 32;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for I2_S: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        for (block_idx, scale) in scales.iter_mut().enumerate().take(num_blocks) {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find the maximum absolute value for scaling
            let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Avoid division by zero
            *scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

            // Quantize block to 2-bit signed values (-1, 0, 1)
            for (i, &val) in block.iter().enumerate() {
                let normalized = val / *scale;
                let quantized = if normalized > 0.5 {
                    1i8 // +1
                } else if normalized < -0.5 {
                    3i8 // -1 (represented as 3 in 2-bit)
                } else {
                    0i8 // 0
                };

                // Pack 4 values into one byte (2 bits each)
                let byte_idx = (start + i) / 4;
                let bit_offset = ((start + i) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= (quantized as u8) << bit_offset;
                }
            }
        }

        Ok(())
    }

    /// TL1 quantization: Table lookup optimized for ARM
    fn quantize_tl1(&self, input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
        const BLOCK_SIZE: usize = 64;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL1: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        for (block_idx, scale) in scales.iter_mut().enumerate().take(num_blocks) {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Compute scale for this block
            let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            *scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

            // Create lookup table for this block (simplified version)
            let lut = [-1.0f32, -0.33, 0.33, 1.0];

            // Quantize using lookup table
            for (i, &val) in block.iter().enumerate() {
                let normalized = val / *scale;

                // Find closest value in lookup table
                let mut best_idx = 0;
                let mut best_dist = (normalized - lut[0]).abs();

                for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                    let dist = (normalized - lut_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                // Pack into output (2 bits per value)
                let byte_idx = (start + i) / 4;
                let bit_offset = ((start + i) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= (best_idx as u8) << bit_offset;
                }
            }
        }

        Ok(())
    }

    /// TL2 quantization: Table lookup optimized for x86
    fn quantize_tl2(&self, input: &[f32], output: &mut [u8], scales: &mut [f32]) -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL2: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        for (block_idx, scale) in scales.iter_mut().enumerate().take(num_blocks) {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Compute scale for this block
            let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            *scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

            // Create optimized lookup table for x86 (different from TL1)
            let lut = [-1.2f32, -0.4, 0.4, 1.2];

            // Quantize using lookup table
            for (i, &val) in block.iter().enumerate() {
                let normalized = val / *scale;

                // Find closest value in lookup table
                let mut best_idx = 0;
                let mut best_dist = (normalized - lut[0]).abs();

                for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                    let dist = (normalized - lut_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                // Pack into output (2 bits per value)
                let byte_idx = (start + i) / 4;
                let bit_offset = ((start + i) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= (best_idx as u8) << bit_offset;
                }
            }
        }

        Ok(())
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
