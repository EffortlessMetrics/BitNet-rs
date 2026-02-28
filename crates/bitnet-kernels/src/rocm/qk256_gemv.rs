//! QK256 2-bit GEMV kernel stubs for ROCm/HIP.
//!
//! Mirrors the CUDA matmul path in [`super::super::gpu::cuda::CudaKernel::launch_matmul`]
//! but targets AMD GPUs via the HIP runtime.
//!
//! # HIP mapping notes
//!
//! | CUDA concept | HIP equivalent |
//! |-------------|----------------|
//! | `cudaMalloc` | `hipMalloc` |
//! | `cudaMemcpy` | `hipMemcpy` |
//! | `cudaStream_t` | `hipStream_t` |
//! | `__shared__` | `__shared__` (identical) |
//! | Thread block | Work-group |
//! | Grid | NDRange |
//! | `blockIdx` / `threadIdx` | `hipBlockIdx_x` / `hipThreadIdx_x` |
//!
//! The QK256 GEMV kernel processes 256-element quantized blocks, each
//! containing 2-bit signed weights packed 4 per byte.  Scale factors are
//! stored separately (one `f32` per block).
//!
//! # Planned launch configuration
//!
//! ```text
//! work-group size : 256 threads (4 wavefronts of 64 on GCN/CDNA)
//! grid            : ceil(N / 256) work-groups
//! LDS (shared)    : 1 KiB for partial-sum reduction
//! ```

use bitnet_common::{KernelError, Result};

/// Launch configuration for the QK256 HIP GEMV kernel (stub).
#[derive(Debug, Clone, Copy)]
pub struct Qk256GemvConfig {
    /// Work-group (block) size — must be a multiple of the wavefront size (64).
    pub workgroup_size: u32,
    /// Bytes of LDS (shared memory) reserved per work-group.
    pub shared_mem_bytes: u32,
}

impl Default for Qk256GemvConfig {
    fn default() -> Self {
        Self { workgroup_size: 256, shared_mem_bytes: 1024 }
    }
}

/// Execute a QK256 2-bit GEMV via HIP.
///
/// # Errors
///
/// Always returns [`KernelError::ExecutionFailed`] — stub only.
pub fn qk256_gemv_hip(
    _weights: &[u8],
    _scales: &[f32],
    _input: &[f32],
    _output: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
    _config: &Qk256GemvConfig,
) -> Result<()> {
    Err(bitnet_common::BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "ROCm/HIP QK256 GEMV kernel is not yet implemented".into(),
    }))
}

/// A single batch GEMV item: (weights, scales, input, output, M, N, K).
pub type GemvBatchItem<'a> = (&'a [u8], &'a [f32], &'a [f32], &'a mut [f32], usize, usize, usize);

/// Batch QK256 GEMV — processes multiple (M,N,K) operations sequentially.
///
/// # Errors
///
/// Always returns [`KernelError::ExecutionFailed`] — stub only.
pub fn qk256_gemv_hip_batch(
    _batches: &[GemvBatchItem<'_>],
    _config: &Qk256GemvConfig,
) -> Result<()> {
    Err(bitnet_common::BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "ROCm/HIP QK256 batch GEMV kernel is not yet implemented".into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qk256_gemv_returns_err() {
        let cfg = Qk256GemvConfig::default();
        let weights = vec![0u8; 64];
        let scales = vec![1.0f32; 1];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        assert!(qk256_gemv_hip(&weights, &scales, &input, &mut output, 1, 1, 256, &cfg).is_err());
    }

    #[test]
    fn default_config_values() {
        let cfg = Qk256GemvConfig::default();
        assert_eq!(cfg.workgroup_size, 256);
        assert_eq!(cfg.shared_mem_bytes, 1024);
    }
}
