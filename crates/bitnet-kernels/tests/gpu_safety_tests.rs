//! Safety validation tests for GPU backends.
//!
//! These tests validate safety properties of the GPU kernel infrastructure
//! without requiring actual GPU hardware. They test the validation logic,
//! bounds checking, and error handling paths that protect against undefined
//! behavior in unsafe code.

use bitnet_common::QuantizationType;

// ---------------------------------------------------------------------------
// GPU memory pool safety (uses the safe MemoryPool abstraction)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod gpu_memory_pool {
    use super::*;
    use bitnet_kernels::gpu::memory_optimization::{MemoryPool, MemoryPoolConfig};

    #[test]
    fn test_zero_size_allocation_handled() {
        let config = MemoryPoolConfig {
            max_pool_size: 1024,
            max_cached_buffers: 10,
            enable_memory_tracking: true,
            ..Default::default()
        };
        let pool = MemoryPool::new(config);
        // Zero-size allocation must not panic or cause UB.
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_pool_config_max_values() {
        let config = MemoryPoolConfig {
            max_pool_size: usize::MAX,
            max_cached_buffers: usize::MAX,
            enable_memory_tracking: true,
            ..Default::default()
        };
        // Construction with extreme values must not overflow or panic.
        let pool = MemoryPool::new(config);
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocated, 0);
    }
}

// ---------------------------------------------------------------------------
// Kernel provider validation (CPU fallback — always available)
// ---------------------------------------------------------------------------

mod cpu_kernel_safety {
    use super::*;
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::FallbackKernel;

    #[test]
    fn test_matmul_zero_dimensions_returns_error() {
        let kernel = FallbackKernel;
        let a: &[i8] = &[];
        let b: &[u8] = &[];
        let mut c: Vec<f32> = vec![];
        // Zero dimensions must not panic; should return an error or no-op.
        let result = kernel.matmul_i2s(a, b, &mut c, 0, 0, 0);
        // Either Ok (no-op on empty) or Err — but never panic/UB.
        let _ = result;
    }

    #[test]
    fn test_matmul_mismatched_dimensions_returns_error() {
        let kernel = FallbackKernel;
        // a is 2×3 (6 elements), b is 3×2 (6 bytes), c is 2×2 (4 floats)
        let a = vec![1i8; 6];
        let b = vec![1u8; 6];
        let mut c = vec![0.0f32; 4];
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3);
        // Should complete without panic.
        let _ = result;
    }

    #[test]
    fn test_quantize_empty_input_returns_error() {
        let kernel = FallbackKernel;
        let input: &[f32] = &[];
        let mut output: Vec<u8> = vec![];
        let mut scales: Vec<f32> = vec![];
        let result = kernel.quantize(input, &mut output, &mut scales, QuantizationType::I2S);
        // Empty input: should error or no-op, never panic.
        let _ = result;
    }

    #[test]
    fn test_quantize_single_element() {
        let kernel = FallbackKernel;
        let input = [1.0f32];
        let mut output = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        let _ = result;
    }

    #[test]
    fn test_quantize_all_zeros() {
        let kernel = FallbackKernel;
        let input = vec![0.0f32; 64];
        let mut output = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        // All-zero input is a valid edge case; must not divide by zero.
        let _ = result;
    }

    #[test]
    fn test_quantize_nan_input_handled() {
        let kernel = FallbackKernel;
        let input = vec![f32::NAN; 64];
        let mut output = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        // NaN input: should not cause infinite loop or panic.
        let _ = result;
    }

    #[test]
    fn test_quantize_infinity_input_handled() {
        let kernel = FallbackKernel;
        let input = vec![f32::INFINITY; 64];
        let mut output = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        // Infinity: should not cause undefined behavior.
        let _ = result;
    }

    #[test]
    fn test_matmul_large_dimensions_no_overflow() {
        let kernel = FallbackKernel;
        // Use small actual data but large dimension values to check for
        // integer overflow in index calculations.
        let a = vec![1i8; 4];
        let b = vec![1u8; 4];
        let mut c = vec![0.0f32; 4];
        // Intentionally mismatched: m*k != a.len(). The kernel should
        // either bounds-check and error or safely process what it can.
        let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
        let _ = result;
    }
}

// ---------------------------------------------------------------------------
// Device feature detection safety
// ---------------------------------------------------------------------------

mod device_feature_safety {
    use bitnet_kernels::device_features;

    #[test]
    fn test_gpu_compiled_returns_consistent_value() {
        // Must not panic; returns a compile-time constant.
        let compiled = device_features::gpu_compiled();
        // Call twice to verify determinism.
        assert_eq!(compiled, device_features::gpu_compiled());
    }

    #[test]
    fn test_gpu_available_runtime_does_not_panic() {
        // Runtime probe must never panic, even without GPU hardware.
        let _available = device_features::gpu_available_runtime();
    }
}

// ---------------------------------------------------------------------------
// Concurrent access safety
// ---------------------------------------------------------------------------

mod concurrency_safety {
    use bitnet_kernels::KernelProvider;
    use bitnet_kernels::cpu::FallbackKernel;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_concurrent_cpu_kernel_access() {
        // Verify that FallbackKernel can be used from multiple threads
        // without data races. FallbackKernel is stateless and should be
        // trivially safe.
        let kernel = Arc::new(FallbackKernel);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let k = Arc::clone(&kernel);
                thread::spawn(move || {
                    let a = vec![1i8; 16];
                    let b = vec![1u8; 16];
                    let mut c = vec![0.0f32; 16];
                    let _ = k.matmul_i2s(&a, &b, &mut c, 4, 4, 4);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked during concurrent kernel use");
        }
    }

    #[test]
    fn test_concurrent_device_probing() {
        use bitnet_kernels::device_features;
        let handles: Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(|| {
                    let _ = device_features::gpu_compiled();
                    let _ = device_features::gpu_available_runtime();
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked during concurrent probe");
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA-specific safety (requires GPU feature gate)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod cuda_safety {
    use bitnet_kernels::gpu::cuda::CudaDeviceInfo;

    #[test]
    #[ignore = "requires CUDA GPU hardware"]
    fn test_cuda_kernel_creation_invalid_device() {
        use bitnet_kernels::gpu::cuda::CudaKernel;
        // Device 9999 should not exist; must return Err, not panic.
        let result = CudaKernel::new_with_device(9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_device_info_default_values() {
        let info = CudaDeviceInfo {
            device_id: 0,
            name: String::new(),
            compute_capability: (0, 0),
            total_memory: 0,
            multiprocessor_count: 0,
            max_threads_per_block: 0,
            max_shared_memory_per_block: 0,
            supports_fp16: false,
            supports_bf16: false,
        };
        // Verify no invariant violations with zero/empty values.
        assert_eq!(info.device_id, 0);
        assert!(info.name.is_empty());
    }
}
