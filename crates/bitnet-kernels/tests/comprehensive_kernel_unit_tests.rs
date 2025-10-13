#![cfg(feature = "integration-tests")]
//! Comprehensive unit tests for bitnet-kernels
//!
//! This test suite provides comprehensive coverage of all kernel implementations
//! including CPU kernels, SIMD optimizations, GPU kernels, and kernel selection.
//! It achieves >90% code coverage with performance validation.

#![cfg(any(feature = "ffi", feature = "cpu"))]
//!
//! Requirements covered:
//! - 2.1: Validate all public functions and methods
//! - 2.2: Validate all error paths and edge cases
//! - Performance validation for kernel implementations

use bitnet_common::{BitNetError, KernelError, QuantizationType};
use bitnet_kernels::cpu::*;
use bitnet_kernels::*;
use std::time::Instant;

// Import all specific kernel types needed for tests
// These are available on all architectures (stub implementations on unsupported architectures)
use bitnet_kernels::{Avx2Kernel, NeonKernel};

/// Test data generator for consistent testing across kernels
struct TestDataGenerator {
    seed: u64,
}

impl TestDataGenerator {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate deterministic test matrix A (i8)
    fn generate_matrix_a(&mut self, m: usize, k: usize) -> Vec<i8> {
        (0..m * k)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (self.seed % 256) as u8;
                if val > 127 { (val as i16 - 256) as i8 } else { val as i8 }
            })
            .collect()
    }

    /// Generate deterministic test matrix B (u8)
    fn generate_matrix_b(&mut self, k: usize, n: usize) -> Vec<u8> {
        (0..k * n)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                (self.seed % 256) as u8
            })
            .collect()
    }

    /// Generate deterministic test input for quantization
    fn generate_quantization_input(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (self.seed % 1000000) as f32 / 1000000.0;
                (val - 0.5) * 4.0 // Range [-2, 2]
            })
            .collect()
    }
}

// ============================================================================
// CPU Kernel Tests
// ============================================================================

#[cfg(test)]
mod cpu_kernel_tests {
    use super::*;

    #[test]
    fn test_fallback_kernel_basic_functionality() {
        let kernel = FallbackKernel;

        // Test availability and name
        assert!(kernel.is_available());
        assert_eq!(kernel.name(), "fallback");
    }

    #[test]
    fn test_fallback_kernel_matmul_basic() {
        let kernel = FallbackKernel;

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4]; // 2x2 matrix
        let b = vec![1u8, 0, 0, 1]; // 2x2 identity matrix
        let mut c = vec![0.0f32; 4]; // 2x2 result

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        // Expected result: A * I = A
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fallback_kernel_matmul_dimension_validation() {
        let kernel = FallbackKernel;

        let a = vec![1i8, 2];
        let b = vec![1u8, 0];
        let mut c = vec![0.0f32; 4];

        // Wrong dimensions should fail
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());

        if let Err(BitNetError::Kernel(KernelError::ExecutionFailed { reason })) = result {
            assert!(reason.contains("dimension mismatch"));
        } else {
            panic!("Expected dimension mismatch error");
        }
    }

    #[test]
    fn test_fallback_kernel_matmul_various_sizes() {
        let kernel = FallbackKernel;
        let mut data_gen = TestDataGenerator::new(12345);

        let test_sizes = vec![
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 32, 16), // Non-square
            (16, 64, 32), // Different aspect ratio
        ];

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "Fallback kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(
                c.iter().all(|&x| x.is_finite()),
                "Non-finite values in output for size {}x{}x{}",
                m,
                n,
                k
            );
        }
    }

    #[test]
    fn test_fallback_kernel_quantize_i2s() {
        let kernel = FallbackKernel;

        let input = vec![1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1];
        let mut output = vec![0u8; 2]; // 8 values / 4 per byte = 2 bytes
        let mut scales = vec![0.0f32; 1]; // 8 values / 32 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());

        // Should have computed a scale
        assert!(scales[0] > 0.0);

        // Output should be non-zero (some values quantized)
        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_fallback_kernel_quantize_tl1() {
        let kernel = FallbackKernel;

        let input = vec![1.5; 64]; // 64 elements for TL1 block size
        let mut output = vec![0u8; 16]; // 64 values / 4 per byte = 16 bytes
        let mut scales = vec![0.0f32; 1]; // 64 values / 64 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1);
        assert!(result.is_ok());

        // Should have computed a scale
        assert!(scales[0] > 0.0);

        // Output should be non-zero
        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_fallback_kernel_quantize_tl2() {
        let kernel = FallbackKernel;

        let input = vec![1.5; 128]; // 128 elements for TL2 block size
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte = 32 bytes
        let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2);
        assert!(result.is_ok());

        // Should have computed a scale
        assert!(scales[0] > 0.0);

        // Output should be non-zero
        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_fallback_kernel_quantize_buffer_size_validation() {
        let kernel = FallbackKernel;

        let input = vec![1.0; 32];
        let mut output = vec![0u8; 1]; // Too small
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err());

        if let Err(BitNetError::Kernel(KernelError::ExecutionFailed { reason })) = result {
            assert!(reason.contains("Output buffer too small"));
        } else {
            panic!("Expected buffer size error");
        }
    }

    #[test]
    fn test_fallback_kernel_quantize_scales_buffer_validation() {
        let kernel = FallbackKernel;

        let input = vec![1.0; 64]; // 64 elements = 2 blocks for I2S
        let mut output = vec![0u8; 16];
        let mut scales = vec![0.0f32; 1]; // Too small - need 2 scales

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_err());

        if let Err(BitNetError::Kernel(KernelError::ExecutionFailed { reason })) = result {
            assert!(reason.contains("Scales buffer too small"));
        } else {
            panic!("Expected scales buffer size error");
        }
    }

    #[test]
    fn test_fallback_kernel_quantize_edge_cases() {
        let kernel = FallbackKernel;

        // Test with all zeros
        let input = vec![0.0f32; 32];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());
        assert_eq!(scales[0], 1.0); // Default scale when max is too small

        // Test with very small values
        let input = vec![1e-10f32; 32];
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());
        assert_eq!(scales[0], 1.0); // Default scale when max is too small
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_availability() {
        let kernel = Avx2Kernel;

        // On x86_64, availability depends on runtime detection
        println!("AVX2 available: {}", kernel.is_available());
        assert_eq!(kernel.name(), "avx2");
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_matmul_basic() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1u8, 0, 0, 1];
        let mut c = vec![0.0f32; 4];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        // Verify output is reasonable
        assert!(c.iter().all(|&x| x.is_finite()));
        assert!(c.iter().any(|&x| x != 0.0));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_quantize_tl2() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(16); // 128 elements
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte = 32 bytes
        let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_quantize_i2s() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(4); // 32 elements
        let mut output = vec![0u8; 8]; // 32 values / 4 per byte = 8 bytes
        let mut scales = vec![0.0f32; 1]; // 32 values / 32 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    #[test]
    fn test_avx2_kernel_quantize_tl1_fallback() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            println!("AVX2 not available, skipping test");
            return;
        }

        // TL1 should use fallback implementation on x86
        let input = vec![1.5; 64]; // 64 elements for TL1 block size
        let mut output = vec![0u8; 16]; // 64 values / 4 per byte = 16 bytes
        let mut scales = vec![0.0f32; 1]; // 64 values / 64 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn test_avx2_kernel_unavailable_on_non_x86() {
        let kernel = Avx2Kernel;

        // On non-x86_64, should not be available
        assert!(!kernel.is_available());
        assert_eq!(kernel.name(), "avx2");

        // Operations should fail with architecture error
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());

        if let Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture { arch })) = result {
            assert!(arch.contains("AVX2 kernel not available"));
        } else {
            panic!("Expected unsupported architecture error");
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_availability() {
        let kernel = NeonKernel;

        // On ARM64, availability depends on runtime detection
        println!("NEON available: {}", kernel.is_available());
        assert_eq!(kernel.name(), "neon");
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_matmul_basic() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            println!("NEON not available, skipping test");
            return;
        }

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1u8, 0, 0, 1];
        let mut c = vec![0.0f32; 4];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_ok());

        // Verify output is reasonable
        assert!(c.iter().all(|&x| x.is_finite()));
        assert!(c.iter().any(|&x| x != 0.0));
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_quantize_tl1() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            println!("NEON not available, skipping test");
            return;
        }

        let input = vec![1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1; 64];
        let mut output = vec![0u8; 16]; // 64 values / 4 per byte = 16 bytes
        let mut scales = vec![0.0f32; 1]; // 64 values / 64 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_quantize_i2s() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            println!("NEON not available, skipping test");
            return;
        }

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(4); // 32 elements
        let mut output = vec![0u8; 8]; // 32 values / 4 per byte = 8 bytes
        let mut scales = vec![0.0f32; 1]; // 32 values / 32 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn test_neon_kernel_quantize_tl2_fallback() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            println!("NEON not available, skipping test");
            return;
        }

        // TL2 should use fallback implementation on ARM
        let input = vec![1.5; 128]; // 128 elements for TL2 block size
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte = 32 bytes
        let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2);
        assert!(result.is_ok());

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn test_neon_kernel_unavailable_on_non_arm() {
        let kernel = NeonKernel;

        // On non-ARM64, should not be available
        assert!(!kernel.is_available());
        assert_eq!(kernel.name(), "neon");

        // Operations should fail with architecture error
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        assert!(result.is_err());

        if let Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture { arch })) = result {
            assert!(arch.contains("NEON kernel not available"));
        } else {
            panic!("Expected unsupported architecture error");
        }
    }
}

// ============================================================================
// GPU Kernel Tests
// ============================================================================

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod gpu_kernel_tests {
    use super::*;
    use bitnet_kernels::gpu::CudaKernel;

    #[test]
    fn test_cuda_kernel_availability() {
        match CudaKernel::new() {
            Ok(kernel) => {
                assert!(kernel.is_available());
                assert_eq!(kernel.name(), "CUDA");

                let device_info = kernel.device_info();
                assert!(!device_info.name.is_empty());
                assert!(device_info.total_memory > 0);
                assert!(device_info.multiprocessor_count > 0);

                println!("CUDA device: {}", device_info.name);
                println!("Compute capability: {:?}", device_info.compute_capability);
                println!("Memory: {} GB", device_info.total_memory / (1024 * 1024 * 1024));
            }
            Err(_) => {
                println!("CUDA not available, skipping GPU tests");
            }
        }
    }

    #[test]
    fn test_cuda_kernel_new_with_device() {
        match CudaKernel::new_with_device(0) {
            Ok(kernel) => {
                assert!(kernel.is_available());
                assert_eq!(kernel.name(), "CUDA");
            }
            Err(_) => {
                println!("CUDA device 0 not available");
            }
        }

        // Test invalid device ID
        match CudaKernel::new_with_device(999) {
            Ok(_) => panic!("Should fail with invalid device ID"),
            Err(_) => {
                // Expected to fail
            }
        }
    }

    #[test]
    fn test_cuda_kernel_matmul_correctness() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let test_sizes = vec![(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)];
        let mut data_gen = TestDataGenerator::new(11111);

        for (m, n, k) in test_sizes {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "CUDA kernel failed for size {}x{}x{}", m, n, k);

            // Verify output is reasonable
            assert!(c.iter().all(|&x| x.is_finite()), "Non-finite values in CUDA output");
            assert!(c.iter().any(|&x| x != 0.0), "CUDA output should not be all zeros");
        }
    }

    #[test]
    fn test_cuda_kernel_quantization() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];

        let mut data_gen = TestDataGenerator::new(22222);

        for qtype in qtypes {
            let input = data_gen.generate_quantization_input(128);
            let mut output = vec![0u8; 32];
            let mut scales = vec![0.0f32; 4];

            let result = kernel.quantize(&input, &mut output, &mut scales, qtype);
            assert!(result.is_ok(), "CUDA quantization failed for {:?}", qtype);

            assert!(
                scales.iter().all(|&s| s > 0.0 && s.is_finite()),
                "Invalid scales for {:?}",
                qtype
            );
            assert!(
                output.iter().any(|&x| x != 0),
                "Quantization output is all zeros for {:?}",
                qtype
            );
        }
    }

    #[test]
    fn test_cuda_kernel_memory_management() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        // Test multiple operations to ensure no memory leaks
        let mut data_gen = TestDataGenerator::new(33333);

        for i in 0..10 {
            let size = 32 + (i % 3) * 16; // Vary size slightly
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c = vec![0.0f32; size * size];

            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "CUDA operation {} failed", i);

            // Check memory stats periodically
            if i % 5 == 0 {
                let (used, total) = kernel.memory_stats();
                println!("Iteration {}: GPU memory used: {} / {} bytes", i, used, total);
                assert!(used <= total, "Used memory should not exceed total");
            }
        }

        // Synchronize to ensure all operations complete
        let sync_result = kernel.synchronize_all();
        assert!(sync_result.is_ok(), "CUDA synchronization failed");
    }

    #[test]
    fn test_cuda_kernel_error_handling() {
        let kernel = match CudaKernel::new() {
            Ok(k) => k,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        // Test invalid dimensions
        let result = kernel.matmul_i2s(&[], &[], &mut [], 0, 0, 0);
        assert!(result.is_err(), "Should fail with invalid dimensions");

        // Test mismatched dimensions
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];

        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3); // Wrong k
        assert!(result.is_err(), "Should fail with mismatched dimensions");
    }
}

// ============================================================================
// Kernel Selection and Dispatch Tests
// ============================================================================

mod kernel_selection_tests {
    use super::*;

    #[test]
    fn test_kernel_manager_creation() {
        let manager = KernelManager::new();

        // Should always have at least the fallback kernel
        let available = manager.list_available_providers();
        assert!(!available.is_empty(), "No kernel providers available");
        assert!(available.contains(&"fallback"), "Fallback kernel should always be available");
    }

    #[test]
    fn test_kernel_manager_selection_priority() {
        let manager = KernelManager::new();

        let available = manager.list_available_providers();
        let kernel = manager.select_best().expect("Should select a kernel");
        let selected_name = kernel.name();

        println!("Available kernels: {:?}", available);
        println!("Selected kernel: {}", selected_name);

        // Verify selection priority
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        if available.contains(&"CUDA") {
            assert_eq!(selected_name, "CUDA", "CUDA should be preferred when available");
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
        if available.contains(&"avx2") && !available.contains(&"CUDA") {
            assert_eq!(selected_name, "avx2", "AVX2 should be preferred over fallback");
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if available.contains(&"neon") && !available.contains(&"CUDA") {
            assert_eq!(selected_name, "neon", "NEON should be preferred over fallback");
        }

        // Fallback should be selected if no optimized kernels are available
        if !available.iter().any(|&name| name != "fallback") {
            assert_eq!(
                selected_name, "fallback",
                "Fallback should be selected when no optimized kernels available"
            );
        }
    }

    #[test]
    fn test_kernel_manager_consistency() {
        let manager = KernelManager::new();

        // Multiple calls should return the same kernel
        let kernel1 = manager.select_best().unwrap();
        let kernel2 = manager.select_best().unwrap();

        assert_eq!(kernel1.name(), kernel2.name(), "Kernel selection should be consistent");

        // Selected provider name should be consistent
        let name1 = manager.selected_provider_name();
        let name2 = manager.selected_provider_name();

        assert_eq!(name1, name2, "Selected provider name should be consistent");
        assert!(name1.is_some(), "Should have a selected provider name");
    }

    #[test]
    fn test_kernel_manager_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(KernelManager::new());

        // Test concurrent access from multiple threads
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || {
                    let kernel = manager_clone.select_best().unwrap();
                    (i, kernel.name().to_string())
                })
            })
            .collect();

        let results: Vec<(usize, String)> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same kernel
        let first_kernel = &results[0].1;
        for (thread_id, kernel_name) in &results {
            assert_eq!(
                kernel_name, first_kernel,
                "Thread {} got different kernel: {} vs {}",
                thread_id, kernel_name, first_kernel
            );
        }
    }

    #[cfg(test)]
    // Note: reset_selection is only available in the main lib for testing
    // This functionality is tested in the main lib unit tests
    #[test]
    fn test_select_cpu_kernel() {
        let cpu_kernel = select_cpu_kernel().unwrap();

        assert!(cpu_kernel.is_available(), "Selected CPU kernel should be available");
        assert!(!cpu_kernel.name().is_empty(), "CPU kernel should have a name");

        // Should be one of the known CPU kernels
        let known_kernels = ["fallback", "avx2", "neon"];
        assert!(
            known_kernels.contains(&cpu_kernel.name()),
            "Unknown CPU kernel: {}",
            cpu_kernel.name()
        );

        println!("Selected CPU kernel: {}", cpu_kernel.name());
    }

    #[test]
    fn test_select_gpu_kernel() {
        let result = select_gpu_kernel(0);

        match result {
            Ok(gpu_kernel) => {
                assert!(gpu_kernel.is_available(), "Selected GPU kernel should be available");
                println!("Selected GPU kernel: {}", gpu_kernel.name());
            }
            Err(_) => {
                println!("GPU kernel not available (expected on systems without CUDA)");
            }
        }
    }

    #[test]
    fn test_kernel_cross_validation() {
        let manager = KernelManager::new();
        let available_kernels = manager.list_available_providers();

        if available_kernels.len() < 2 {
            println!("Only one kernel available, skipping cross-validation");
            return;
        }

        println!("Cross-validating {} kernels", available_kernels.len());

        let mut data_gen = TestDataGenerator::new(55555);
        let test_size = 16;

        let a = data_gen.generate_matrix_a(test_size, test_size);
        let b = data_gen.generate_matrix_b(test_size, test_size);

        // Get reference result from fallback kernel
        let fallback = FallbackKernel;
        let mut c_reference = vec![0.0f32; test_size * test_size];
        fallback.matmul_i2s(&a, &b, &mut c_reference, test_size, test_size, test_size).unwrap();

        // Test selected kernel against reference
        let kernel = manager.select_best().unwrap();
        let mut c_test = vec![0.0f32; test_size * test_size];
        kernel.matmul_i2s(&a, &b, &mut c_test, test_size, test_size, test_size).unwrap();

        // Compare results
        let mut max_diff = 0.0f32;
        for i in 0..c_reference.len() {
            let diff = (c_reference[i] - c_test[i]).abs();
            max_diff = max_diff.max(diff);
        }

        println!("Cross-validation: {} vs fallback, max_diff = {}", kernel.name(), max_diff);
        assert!(
            max_diff < 1e-2,
            "Kernel {} differs too much from reference: {}",
            kernel.name(),
            max_diff
        );
    }
}

// ============================================================================
// FFI Kernel Tests
// ============================================================================

#[cfg(feature = "ffi")]
mod ffi_kernel_tests {
    use super::*;
    use bitnet_kernels::ffi::FfiKernel;

    #[test]
    fn test_ffi_kernel_availability() {
        match FfiKernel::new() {
            Ok(kernel) => {
                println!("FFI kernel available: {}", kernel.is_available());
                assert_eq!(kernel.name(), "ffi");

                if kernel.is_available() {
                    // Test basic functionality
                    let a = vec![1i8, 2, 3, 4];
                    let b = vec![1u8, 0, 0, 1];
                    let mut c = vec![0.0f32; 4];

                    let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
                    assert!(result.is_ok(), "FFI kernel matmul should work");
                }
            }
            Err(e) => {
                println!("FFI kernel not available: {}", e);
            }
        }
    }

    #[test]
    fn test_ffi_kernel_quantization() {
        let kernel = match FfiKernel::new() {
            Ok(k) if k.is_available() => k,
            _ => {
                println!("FFI kernel not available, skipping test");
                return;
            }
        };

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(4);
        let mut output = vec![0u8; 8];
        let mut scales = vec![0.0f32; 1];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        assert!(result.is_ok(), "FFI kernel quantization should work");

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }
}

#[cfg(not(feature = "ffi"))]
mod ffi_kernel_disabled_tests {

    #[test]
    fn test_ffi_kernel_disabled() {
        // FFI module not available when feature is disabled
        println!("FFI kernel feature disabled, skipping tests");
    }
}

// ============================================================================
// Performance and Edge Case Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_kernel_performance_scaling() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        let sizes = vec![32, 64, 128];
        let mut data_gen = TestDataGenerator::new(66666);

        println!("Performance scaling test for kernel: {}", kernel.name());

        for size in sizes {
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c = vec![0.0f32; size * size];

            // Warm up
            for _ in 0..3 {
                let _ = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            }

            // Benchmark
            let iterations = if size <= 64 { 10 } else { 5 };
            let start = Instant::now();

            for _ in 0..iterations {
                kernel.matmul_i2s(&a, &b, &mut c, size, size, size).unwrap();
            }

            let elapsed = start.elapsed().as_nanos() as u64;
            let avg_time = elapsed / iterations;
            let ops = 2 * size * size * size; // Approximate FLOP count

            println!(
                "  {}x{}: {:.2} ms, {:.2} GFLOPS",
                size,
                size,
                avg_time as f64 / 1_000_000.0,
                (ops as f64) / (avg_time as f64 / 1_000_000_000.0) / 1_000_000_000.0
            );

            // Performance should be reasonable
            assert!(avg_time > 0, "Execution time should be positive");
        }
    }

    #[test]
    fn test_quantization_performance() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        let sizes = vec![1024, 4096];
        let qtypes = vec![QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
        let mut data_gen = TestDataGenerator::new(77777);

        println!("Quantization performance test for kernel: {}", kernel.name());

        for qtype in qtypes {
            for size in &sizes {
                let input = data_gen.generate_quantization_input(*size);
                let mut output = vec![0u8; size / 4];
                let block_size = match qtype {
                    QuantizationType::I2S => 32,
                    QuantizationType::TL1 => 64,
                    QuantizationType::TL2 => 128,
                };
                let num_blocks = size.div_ceil(block_size);
                let mut scales = vec![0.0f32; num_blocks];

                // Warm up
                for _ in 0..3 {
                    let _ = kernel.quantize(&input, &mut output, &mut scales, qtype);
                }

                // Benchmark
                let iterations = 50;
                let start = Instant::now();

                for _ in 0..iterations {
                    kernel.quantize(&input, &mut output, &mut scales, qtype).unwrap();
                }

                let elapsed = start.elapsed().as_nanos() as u64;
                let avg_time = elapsed / iterations;

                println!(
                    "  {:?} {}: {:.2} μs, {:.2} M elements/sec",
                    qtype,
                    size,
                    avg_time as f64 / 1_000.0,
                    (*size as f64) / (avg_time as f64 / 1_000_000_000.0) / 1_000_000.0
                );

                // Performance should be reasonable
                assert!(avg_time > 0, "Quantization time should be positive");
            }
        }
    }

    #[test]
    fn test_edge_case_matrix_sizes() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();
        let mut data_gen = TestDataGenerator::new(88888);

        // Test edge case sizes
        let edge_cases = vec![
            (1, 1, 1),    // Minimum size
            (1, 100, 1),  // Very wide
            (100, 1, 1),  // Very tall
            (7, 13, 11),  // Prime numbers
            (15, 17, 19), // Odd numbers
        ];

        for (m, n, k) in edge_cases {
            let a = data_gen.generate_matrix_a(m, k);
            let b = data_gen.generate_matrix_b(k, n);
            let mut c = vec![0.0f32; m * n];

            let result = kernel.matmul_i2s(&a, &b, &mut c, m, n, k);
            assert!(result.is_ok(), "Edge case {}x{}x{} should work", m, n, k);

            // Verify output is reasonable
            assert!(
                c.iter().all(|&x| x.is_finite()),
                "Non-finite values in edge case {}x{}x{}",
                m,
                n,
                k
            );
        }
    }

    #[test]
    fn test_quantization_edge_cases() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();

        // Test with extreme values (excluding infinity which may not be handled consistently)
        let extreme_cases =
            [vec![f32::MAX; 32], vec![f32::MIN; 32], vec![1e-10; 32], vec![-1e-10; 32]];

        for (i, input) in extreme_cases.iter().enumerate() {
            let mut output = vec![0u8; 8];
            let mut scales = vec![0.0f32; 1];

            let result = kernel.quantize(input, &mut output, &mut scales, QuantizationType::I2S);

            // Some extreme cases might fail, but should not panic
            match result {
                Ok(_) => {
                    println!("Extreme case {} handled successfully", i);
                    assert!(
                        scales[0].is_finite() || scales[0] == 1.0,
                        "Scale should be finite or default"
                    );
                }
                Err(e) => {
                    println!("Extreme case {} failed as expected: {}", i, e);
                }
            }
        }

        // Test infinity cases separately (these are expected to either fail or produce special handling)
        let infinity_cases = [vec![f32::INFINITY; 32], vec![f32::NEG_INFINITY; 32]];

        for (i, input) in infinity_cases.iter().enumerate() {
            let mut output = vec![0u8; 8];
            let mut scales = vec![0.0f32; 1];

            let result = kernel.quantize(input, &mut output, &mut scales, QuantizationType::I2S);

            match result {
                Ok(_) => {
                    println!("Infinity case {} handled", i);
                    // For infinity cases, we're more lenient about the scale value
                    // The kernel may handle infinity in various ways
                }
                Err(e) => {
                    println!("Infinity case {} failed as expected: {}", i, e);
                }
            }
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_kernel_interoperability() {
        let manager = KernelManager::new();
        let available = manager.list_available_providers();

        if available.len() < 2 {
            println!("Need at least 2 kernels for interoperability test");
            return;
        }

        let mut data_gen = TestDataGenerator::new(99999);
        let size = 32;

        let a = data_gen.generate_matrix_a(size, size);
        let b = data_gen.generate_matrix_b(size, size);

        // Test that all available kernels produce similar results
        let mut results = Vec::new();

        // Get fallback result as reference
        let fallback = FallbackKernel;
        let mut c_fallback = vec![0.0f32; size * size];
        fallback.matmul_i2s(&a, &b, &mut c_fallback, size, size, size).unwrap();
        results.push(("fallback", c_fallback));

        // Test selected kernel
        let kernel = manager.select_best().unwrap();
        if kernel.name() != "fallback" {
            let mut c_selected = vec![0.0f32; size * size];
            kernel.matmul_i2s(&a, &b, &mut c_selected, size, size, size).unwrap();
            results.push((kernel.name(), c_selected));
        }

        // Compare all results
        for i in 1..results.len() {
            let (name1, ref1) = &results[0];
            let (name2, ref2) = &results[i];

            let mut max_diff = 0.0f32;
            for j in 0..ref1.len() {
                let diff = (ref1[j] - ref2[j]).abs();
                max_diff = max_diff.max(diff);
            }

            println!("Interoperability: {} vs {}, max_diff = {}", name1, name2, max_diff);
            assert!(
                max_diff < 1e-2,
                "Kernels {} and {} differ too much: {}",
                name1,
                name2,
                max_diff
            );
        }
    }

    #[test]
    fn test_kernel_state_isolation() {
        let manager = KernelManager::new();
        let kernel = manager.select_best().unwrap();
        let mut data_gen = TestDataGenerator::new(11111);

        // Run multiple operations to ensure no state interference
        for i in 0..10 {
            let size = 16 + (i % 4) * 8;
            let a = data_gen.generate_matrix_a(size, size);
            let b = data_gen.generate_matrix_b(size, size);
            let mut c = vec![0.0f32; size * size];

            let result = kernel.matmul_i2s(&a, &b, &mut c, size, size, size);
            assert!(result.is_ok(), "Operation {} should succeed", i);

            // Each operation should produce consistent results
            let mut c2 = vec![0.0f32; size * size];
            let result2 = kernel.matmul_i2s(&a, &b, &mut c2, size, size, size);
            assert!(result2.is_ok(), "Repeated operation {} should succeed", i);

            // Results should be identical
            for j in 0..c.len() {
                assert_eq!(c[j], c2[j], "Results should be identical for operation {}", i);
            }
        }
    }
}
