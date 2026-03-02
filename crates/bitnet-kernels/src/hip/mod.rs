//! HIP kernel scaffolding for AMD GPU inference operations.
//!
//! This module provides HIP kernel launch configurations, CPU fallback
//! implementations, and stub entry points for AMD GPU acceleration via
//! the ROCm/HIP runtime. The structure mirrors [`crate::cuda`] so that
//! HIP can serve as a drop-in backend alongside CUDA.
//!
//! # Sub-modules
//!
//! | Module | CUDA counterpart | Description |
//! |--------|-----------------|-------------|
//! | [`device_info`] | `gpu::cuda` device | AMD GPU detection, GCN/CDNA arch classification |
//! | [`memory_pool`] | [`crate::cuda::memory_pool`] | HIP memory allocation pool |
//! | [`stream_mgmt`] | [`crate::cuda::stream_mgmt`] | HIP stream and event management |
//! | [`matmul`] | [`crate::cuda::matmul`] | Tiled GEMM kernel stubs |
//! | [`attention`] | [`crate::cuda::attention`] | Scaled dot-product attention stubs |
//! | [`quantize`] | [`crate::cuda::quantize`] | INT2/INT4 quantization stubs |
//!
//! All GPU-dependent code is gated behind `#[cfg(feature = "rocm")]`.
//! CPU fallback implementations are always available.

pub mod attention;
pub mod device_info;
pub mod matmul;
pub mod memory_pool;
pub mod quantize;
pub mod stream_mgmt;

// ── Re-exports: device_info ──────────────────────────────────────────

pub use device_info::{
    GpuArchFamily, HipDeviceCapabilities, device_count, enumerate_devices, get_device,
    select_best_device,
};

// ── Re-exports: memory_pool ──────────────────────────────────────────

pub use memory_pool::{
    HipAllocId, HipAllocation, HipMemoryPool, HipMemoryPoolConfig, HipMemoryStats,
};

// ── Re-exports: stream_mgmt ──────────────────────────────────────────

pub use stream_mgmt::{
    HipStreamConfig, HipStreamEvent, HipStreamHandle, HipStreamPool, HipStreamPriority,
};

// ── Re-exports: matmul ───────────────────────────────────────────────

pub use matmul::{HipMatmulConfig, HipMatmulDtype, hip_matmul_cpu, hip_matmul_forward};

#[cfg(feature = "rocm")]
pub use matmul::{HIP_MATMUL_KERNEL_SRC, launch_hip_matmul};

// ── Re-exports: attention ────────────────────────────────────────────

pub use attention::{
    HipAttentionConfig, HipAttentionMask, hip_attention_cpu, hip_attention_forward,
    hip_multi_head_attention_cpu,
};

#[cfg(feature = "rocm")]
pub use attention::{HIP_ATTENTION_KERNEL_SRC, launch_hip_attention};

// ── Re-exports: quantize ─────────────────────────────────────────────

pub use quantize::{
    HipQuantBits, HipQuantMethod, HipQuantizeConfig, hip_calibrate_scales, hip_dequantize_int2_cpu,
    hip_dequantize_int4_cpu, hip_quantize_int2_cpu, hip_quantize_int4_cpu,
};

#[cfg(feature = "rocm")]
pub use quantize::{
    HIP_DEQUANTIZE_INT2_KERNEL_SRC, HIP_QUANTIZE_INT2_KERNEL_SRC, launch_hip_quantize,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_reexports_device_count() {
        assert_eq!(device_count(), 0);
    }

    #[test]
    fn module_reexports_matmul_cpu() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        hip_matmul_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn module_reexports_attention_cpu() {
        let d = 2;
        let q = vec![1.0; d];
        let k = vec![1.0; d];
        let v = vec![1.0; d];
        let mut out = vec![0.0; d];
        let cfg = HipAttentionConfig::new(1, d, 1, 1);
        hip_attention_cpu(&q, &k, &v, &mut out, &cfg).unwrap();
    }

    #[test]
    fn module_reexports_quantize_calibrate() {
        let input = vec![1.0, -2.0, 0.5, 3.0];
        let cfg = HipQuantizeConfig { block_size: 4, ..Default::default() };
        let scales = hip_calibrate_scales(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 3.0).abs() < 1e-6);
    }
}
