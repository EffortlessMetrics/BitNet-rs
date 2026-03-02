//! HIP matrix multiply kernel stubs with CPU fallback.
//!
//! Mirrors the CUDA matmul interface in [`crate::cuda::matmul`] but targets
//! AMD GPUs via HIP. Provides tiled GEMM configuration and pure-Rust
//! reference implementations for correctness testing.
//!
//! # HIP-specific considerations
//!
//! * Work-group size should be a multiple of the wavefront size (64 for
//!   GCN/CDNA, 32 for RDNA).
//! * LDS (Local Data Share) is used for tile storage — up to 64 KiB on
//!   MI200-series.
//! * MFMA (Matrix Fused Multiply-Add) instructions should be preferred
//!   on CDNA architectures for FP16/BF16 matmul.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Data type enum ───────────────────────────────────────────────────

/// Supported element data types for the HIP matmul kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipMatmulDtype {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754).
    F16,
    /// 16-bit brain floating point.
    Bf16,
}

// ── Launch configuration ─────────────────────────────────────────────

/// Configuration for the HIP tiled GEMM kernel.
///
/// Computes `C = alpha * op(A) · op(B) + beta * C`.
#[derive(Debug, Clone)]
pub struct HipMatmulConfig {
    /// Number of output rows.
    pub m: usize,
    /// Number of output columns.
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Batch count (1 for non-batched).
    pub batch_size: usize,
    /// Transpose A before multiplication.
    pub transpose_a: bool,
    /// Transpose B before multiplication.
    pub transpose_b: bool,
    /// Scalar multiplier for the product.
    pub alpha: f32,
    /// Scalar multiplier for the existing output.
    pub beta: f32,
    /// Element data type.
    pub dtype: HipMatmulDtype,
    /// Tile size in the M dimension.
    pub tile_m: u32,
    /// Tile size in the N dimension.
    pub tile_n: u32,
    /// Tile size in the K dimension (LDS streaming).
    pub tile_k: u32,
    /// Work-group size (threads per work-group).
    pub workgroup_size: u32,
    /// Bytes of LDS for A and B tiles.
    pub lds_bytes: u32,
}

impl Default for HipMatmulConfig {
    fn default() -> Self {
        Self {
            m: 0,
            n: 0,
            k: 0,
            batch_size: 1,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
            dtype: HipMatmulDtype::F32,
            tile_m: 32,
            tile_n: 32,
            tile_k: 16,
            workgroup_size: 256,
            lds_bytes: 8192,
        }
    }
}

impl HipMatmulConfig {
    /// Create a config for a standard (non-batched, non-transposed) matmul.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, ..Default::default() }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "matmul dimensions m, n, k must be non-zero".into(),
            }
            .into());
        }
        if self.batch_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "batch_size must be at least 1".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── HIP kernel source (stub) ────────────────────────────────────────

/// HIP C source for the tiled GEMM kernel.
///
/// Stub — will contain HIP C++ kernel code once implementation begins.
#[cfg(feature = "rocm")]
pub const HIP_MATMUL_KERNEL_SRC: &str = r#"
// TODO: HIP tiled GEMM kernel
// hipLaunchKernelGGL(tiled_gemm_f32, ...)
extern "C" __global__ void tiled_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Stub — to be implemented with LDS tiling
}
"#;

// ── CPU fallback ─────────────────────────────────────────────────────

/// Naive O(n³) CPU reference implementation of matrix multiply.
///
/// Computes `C = alpha * A · B + beta * C` (no transpose support in
/// this fallback).
pub fn hip_matmul_cpu(a: &[f32], b: &[f32], c: &mut [f32], config: &HipMatmulConfig) -> Result<()> {
    config.validate()?;
    let (m, n, k) = (config.m, config.n, config.k);

    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("A buffer too small: {} < {}", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("B buffer too small: {} < {}", b.len(), k * n),
        }
        .into());
    }
    if c.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("C buffer too small: {} < {}", c.len(), m * n),
        }
        .into());
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = config.alpha * sum + config.beta * c[i * n + j];
        }
    }
    Ok(())
}

/// Dispatch matmul to HIP GPU or fall back to CPU.
///
/// Currently always falls back to CPU.
pub fn hip_matmul_forward(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    config: &HipMatmulConfig,
) -> Result<()> {
    // TODO: dispatch to HIP kernel when runtime is available
    hip_matmul_cpu(a, b, c, config)
}

/// Launch the HIP matmul kernel on the GPU.
///
/// Stub — returns an error until HIP runtime integration.
#[cfg(feature = "rocm")]
pub fn launch_hip_matmul(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _config: &HipMatmulConfig,
) -> Result<()> {
    Err(BitNetError::Kernel(KernelError::ExecutionFailed {
        reason: "HIP tiled GEMM kernel is not yet implemented".into(),
    }))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = HipMatmulConfig::default();
        assert_eq!(cfg.tile_m, 32);
        assert_eq!(cfg.tile_n, 32);
        assert_eq!(cfg.workgroup_size, 256);
        assert_eq!(cfg.dtype, HipMatmulDtype::F32);
        assert_eq!(cfg.batch_size, 1);
    }

    #[test]
    fn config_new_sets_dimensions() {
        let cfg = HipMatmulConfig::new(4, 8, 16);
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.n, 8);
        assert_eq!(cfg.k, 16);
    }

    #[test]
    fn config_validate_ok() {
        assert!(HipMatmulConfig::new(4, 4, 4).validate().is_ok());
    }

    #[test]
    fn config_validate_zero_m() {
        let cfg = HipMatmulConfig::new(0, 4, 4);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_batch() {
        let mut cfg = HipMatmulConfig::new(4, 4, 4);
        cfg.batch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cpu_matmul_identity() {
        // 2×2 identity: A = I, B = I => C = I
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        hip_matmul_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn cpu_matmul_3x3() {
        // A = [[1,2,3],[4,5,6],[7,8,9]], B = I => C = A
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 9];
        let cfg = HipMatmulConfig::new(3, 3, 3);
        hip_matmul_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, a);
    }

    #[test]
    fn cpu_matmul_rectangular() {
        // A: 2×3, B: 3×2 => C: 2×2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 3);
        hip_matmul_cpu(&a, &b, &mut c, &cfg).unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_alpha_beta() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        let mut cfg = HipMatmulConfig::new(2, 2, 2);
        cfg.alpha = 2.0;
        cfg.beta = 1.0;
        hip_matmul_cpu(&a, &b, &mut c, &cfg).unwrap();
        // C = 2*A*B + 1*C_old = 2*A + C_old
        assert_eq!(c, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn cpu_matmul_buffer_too_small_a() {
        let a = vec![1.0; 2]; // too small for 2×2
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        assert!(hip_matmul_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn cpu_matmul_buffer_too_small_b() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 2];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        assert!(hip_matmul_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn cpu_matmul_buffer_too_small_c() {
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 2];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        assert!(hip_matmul_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn forward_dispatches_to_cpu() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 4];
        let cfg = HipMatmulConfig::new(2, 2, 2);
        hip_matmul_forward(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn dtype_enum_equality() {
        assert_eq!(HipMatmulDtype::F32, HipMatmulDtype::F32);
        assert_ne!(HipMatmulDtype::F32, HipMatmulDtype::F16);
        assert_ne!(HipMatmulDtype::F16, HipMatmulDtype::Bf16);
    }
}
