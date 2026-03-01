//! General matrix multiplication (GEMM) kernel with CPU fallback and GPU stub.
//!
//! # Operation
//!
//! Computes `C = alpha * op(A) @ op(B) + beta * C` where `op(X)` is either
//! `X` or `X^T` depending on the transpose flags.
//!
//! # Supported variants
//!
//! - Standard GEMM: `C = alpha * A @ B + beta * C`
//! - Transposed A: `C = alpha * A^T @ B + beta * C`
//! - Transposed B: `C = alpha * A @ B^T + beta * C`
//! - Both transposed: `C = alpha * A^T @ B^T + beta * C`
//! - Batched GEMM: operates on 3D tensors `[batch, rows, cols]`
//!
//! # CPU fallback
//!
//! [`gemm_cpu`] provides a naive triple-loop implementation for correctness
//! testing and non-GPU environments. Can be replaced with BLAS later.
//!
//! # GPU stub
//!
//! [`launch_gemm`] is gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`
//! and returns a scaffold error until real PTX kernels are compiled.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a GEMM operation.
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// Rows of `op(A)` and `C`.
    pub m: usize,
    /// Columns of `op(B)` and `C`.
    pub n: usize,
    /// Shared (contraction) dimension: columns of `op(A)`, rows of `op(B)`.
    pub k: usize,
    /// Scaling factor for the product `op(A) @ op(B)`.
    pub alpha: f32,
    /// Scaling factor for the existing contents of `C`.
    pub beta: f32,
    /// Whether `A` is transposed before multiplication.
    pub trans_a: bool,
    /// Whether `B` is transposed before multiplication.
    pub trans_b: bool,
    /// Batch count for batched GEMM (1 = single GEMM).
    pub batch_size: usize,
    /// Threads per block for CUDA launch.
    pub threads_per_block: u32,
}

impl GemmConfig {
    /// Create a configuration for a single GEMM `C[m,n] = A[m,k] @ B[k,n]`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("GEMM dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }

        let threads_per_block = (n as u32).min(1024);

        Ok(Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
            batch_size: 1,
            threads_per_block,
        })
    }

    /// Set alpha/beta scaling: `C = alpha * op(A)@op(B) + beta * C`.
    #[must_use]
    pub fn with_scaling(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set transpose flags for A and/or B.
    #[must_use]
    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.trans_a = trans_a;
        self.trans_b = trans_b;
        self
    }

    /// Set batch size for batched GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `batch_size` is zero.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "batch_size must be non-zero".into(),
            }
            .into());
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Expected number of elements in `A`.
    pub fn a_len(&self) -> usize {
        let (rows, cols) = if self.trans_a { (self.k, self.m) } else { (self.m, self.k) };
        self.batch_size * rows * cols
    }

    /// Expected number of elements in `B`.
    pub fn b_len(&self) -> usize {
        let (rows, cols) = if self.trans_b { (self.n, self.k) } else { (self.k, self.n) };
        self.batch_size * rows * cols
    }

    /// Expected number of elements in `C`.
    pub fn c_len(&self) -> usize {
        self.batch_size * self.m * self.n
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(self.threads_per_block);
        (grid_x, self.m as u32, self.batch_size as u32)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Helper: element access with transpose
// ---------------------------------------------------------------------------

/// Read element `(row, col)` from a row-major matrix, optionally transposed.
#[inline]
fn elem(
    data: &[f32],
    row: usize,
    col: usize,
    ld_rows: usize,
    ld_cols: usize,
    transposed: bool,
) -> f32 {
    if transposed {
        data[col * ld_rows + row]
    } else {
        let _ = ld_cols;
        data[row * ld_cols + col]
    }
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// CPU fallback for GEMM: `C = alpha * op(A) @ op(B) + beta * C`.
///
/// Uses a naive triple-loop implementation. Suitable for correctness testing;
/// replace with BLAS for production throughput.
///
/// All matrices are in row-major order. When `trans_a` is set, `A` is stored
/// as `[K, M]` (transposed layout); similarly `trans_b` means `B` is `[N, K]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if slice lengths do not match the
/// configuration.
pub fn gemm_cpu(a: &[f32], b: &[f32], c: &mut [f32], config: &GemmConfig) -> Result<()> {
    if a.len() < config.a_len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("GEMM: A length {} < expected {}", a.len(), config.a_len()),
        }
        .into());
    }
    if b.len() < config.b_len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("GEMM: B length {} < expected {}", b.len(), config.b_len()),
        }
        .into());
    }
    if c.len() < config.c_len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("GEMM: C length {} < expected {}", c.len(), config.c_len()),
        }
        .into());
    }

    let (m, n, k) = (config.m, config.n, config.k);
    let (alpha, beta) = (config.alpha, config.beta);

    let (a_ld_rows, a_ld_cols) = (m, k);
    let (b_ld_rows, b_ld_cols) = (k, n);

    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_batch_stride;
        let b_off = batch * b_batch_stride;
        let c_off = batch * c_batch_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    let a_val = elem(&a[a_off..], i, p, a_ld_rows, a_ld_cols, config.trans_a);
                    let b_val = elem(&b[b_off..], p, j, b_ld_rows, b_ld_cols, config.trans_b);
                    acc += a_val * b_val;
                }
                let c_idx = c_off + i * n + j;
                c[c_idx] = alpha * acc + beta * c[c_idx];
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA launch stub
// ---------------------------------------------------------------------------

/// Launch stub for the GEMM CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gemm(_a: &[f32], _b: &[f32], _c: &mut [f32], config: &GemmConfig) -> Result<()> {
    log::debug!(
        "GEMM CUDA stub: m={}, n={}, k={}, batch={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.batch_size,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "GEMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply GEMM with automatic dispatch: GPU if available, else CPU fallback.
pub fn gemm_forward(a: &[f32], b: &[f32], c: &mut [f32], config: &GemmConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_gemm(a, b, c, config) {
                return Ok(());
            }
        }
    }
    gemm_cpu(a, b, c, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config tests -------------------------------------------------------

    #[test]
    fn test_matmul_config_new() {
        let cfg = GemmConfig::new(4, 8, 6).unwrap();
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.n, 8);
        assert_eq!(cfg.k, 6);
        assert!((cfg.alpha - 1.0).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.0).abs() < f32::EPSILON);
        assert!(!cfg.trans_a);
        assert!(!cfg.trans_b);
        assert_eq!(cfg.batch_size, 1);
    }

    #[test]
    fn test_matmul_config_rejects_zero_m() {
        assert!(GemmConfig::new(0, 4, 4).is_err());
    }

    #[test]
    fn test_matmul_config_rejects_zero_n() {
        assert!(GemmConfig::new(4, 0, 4).is_err());
    }

    #[test]
    fn test_matmul_config_rejects_zero_k() {
        assert!(GemmConfig::new(4, 4, 0).is_err());
    }

    #[test]
    fn test_matmul_config_rejects_zero_batch() {
        let cfg = GemmConfig::new(4, 4, 4).unwrap();
        assert!(cfg.with_batch_size(0).is_err());
    }

    #[test]
    fn test_matmul_config_scaling() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_scaling(2.0, 0.5);
        assert!((cfg.alpha - 2.0).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matmul_config_transpose() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap().with_transpose(true, false);
        assert!(cfg.trans_a);
        assert!(!cfg.trans_b);
    }

    #[test]
    fn test_matmul_config_grid_dim() {
        let cfg = GemmConfig::new(32, 2048, 512).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gy, 32);
        assert_eq!(gz, 1);
        assert!(gx >= 1);
    }

    #[test]
    fn test_matmul_config_a_b_c_len() {
        let cfg = GemmConfig::new(3, 4, 5).unwrap();
        assert_eq!(cfg.a_len(), 3 * 5);
        assert_eq!(cfg.b_len(), 5 * 4);
        assert_eq!(cfg.c_len(), 3 * 4);
    }

    #[test]
    fn test_matmul_config_len_with_transpose() {
        let cfg = GemmConfig::new(3, 4, 5).unwrap().with_transpose(true, true);
        assert_eq!(cfg.a_len(), 5 * 3);
        assert_eq!(cfg.b_len(), 4 * 5);
        assert_eq!(cfg.c_len(), 3 * 4);
    }

    #[test]
    fn test_matmul_config_len_batched() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap().with_batch_size(5).unwrap();
        assert_eq!(cfg.a_len(), 5 * 2 * 4);
        assert_eq!(cfg.b_len(), 5 * 4 * 3);
        assert_eq!(cfg.c_len(), 5 * 2 * 3);
    }

    // -- Dimension mismatch errors ------------------------------------------

    #[test]
    fn test_matmul_cpu_rejects_short_a() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap();
        let a = vec![0.0_f32; 4]; // need 8
        let b = vec![0.0_f32; 12];
        let mut c = vec![0.0_f32; 6];
        assert!(gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_matmul_cpu_rejects_short_b() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap();
        let a = vec![0.0_f32; 8];
        let b = vec![0.0_f32; 4]; // need 12
        let mut c = vec![0.0_f32; 6];
        assert!(gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_matmul_cpu_rejects_short_c() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap();
        let a = vec![0.0_f32; 8];
        let b = vec![0.0_f32; 12];
        let mut c = vec![0.0_f32; 2]; // need 6
        assert!(gemm_cpu(&a, &b, &mut c, &cfg).is_err());
    }

    // -- Identity multiply --------------------------------------------------

    #[test]
    fn test_matmul_identity_2x2() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap();
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a, &identity, &mut c, &cfg).unwrap();
        assert_eq!(c, a);
    }

    #[test]
    fn test_matmul_identity_3x3() {
        let cfg = GemmConfig::new(3, 3, 3).unwrap();
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut c = vec![0.0_f32; 9];
        gemm_cpu(&a, &identity, &mut c, &cfg).unwrap();
        assert_eq!(c, a);
    }

    // -- Zero matrix --------------------------------------------------------

    #[test]
    fn test_matmul_zero_matrix() {
        let cfg = GemmConfig::new(2, 3, 4).unwrap();
        let a = vec![1.0_f32; 8];
        let zero = vec![0.0_f32; 12];
        let mut c = vec![99.0_f32; 6];
        gemm_cpu(&a, &zero, &mut c, &cfg).unwrap();
        assert!(c.iter().all(|&v| v == 0.0));
    }

    // -- Square matrices (known results) ------------------------------------

    #[test]
    fn test_matmul_square_2x2() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap();
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            5.0, 6.0,
            7.0, 8.0,
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // -- Rectangular matrices -----------------------------------------------

    #[test]
    fn test_matmul_rectangular() {
        // A[2,3] @ B[3,4] = C[2,4]
        let cfg = GemmConfig::new(2, 4, 3).unwrap();
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let mut c = vec![0.0_f32; 8];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 4.0]);
    }

    // -- Transpose variants -------------------------------------------------

    #[test]
    fn test_matmul_trans_a() {
        // A stored as [K=2, M=2] (transposed), logical A = [[1,2],[3,4]]
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_transpose(true, false);
        #[rustfmt::skip]
        let a_stored = vec![
            1.0, 3.0,  // k=0 row
            2.0, 4.0,  // k=1 row
        ];
        #[rustfmt::skip]
        let b = vec![
            5.0, 6.0,
            7.0, 8.0,
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a_stored, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_trans_b() {
        // B stored as [N=2, K=2] (transposed), logical B = [[5,6],[7,8]]
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_transpose(false, true);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        #[rustfmt::skip]
        let b_stored = vec![
            5.0, 7.0,  // n=0 row
            6.0, 8.0,  // n=1 row
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a, &b_stored, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_trans_both() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_transpose(true, true);
        #[rustfmt::skip]
        let a_stored = vec![
            1.0, 3.0,
            2.0, 4.0,
        ];
        #[rustfmt::skip]
        let b_stored = vec![
            5.0, 7.0,
            6.0, 8.0,
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a_stored, &b_stored, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // -- Alpha / Beta scaling -----------------------------------------------

    #[test]
    fn test_matmul_alpha_scaling() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_scaling(2.0, 0.0);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            3.0, 4.0,
            5.0, 6.0,
        ];
        let mut c = vec![0.0_f32; 4];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul_beta_accumulate() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_scaling(1.0, 1.0);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let b = a.clone();
        let mut c = vec![10.0_f32; 4];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![11.0, 10.0, 10.0, 11.0]);
    }

    #[test]
    fn test_matmul_alpha_beta_combined() {
        let cfg = GemmConfig::new(1, 1, 2).unwrap().with_scaling(3.0, 2.0);
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let mut c = vec![5.0];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        // C = 3*(1*3 + 2*4) + 2*5 = 33 + 10 = 43
        assert_eq!(c, vec![43.0]);
    }

    // -- Batched GEMM -------------------------------------------------------

    #[test]
    fn test_matmul_batched() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_batch_size(2).unwrap();
        #[rustfmt::skip]
        let a = vec![
            // batch 0: identity
            1.0, 0.0, 0.0, 1.0,
            // batch 1
            1.0, 2.0, 3.0, 4.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            // batch 0: identity
            1.0, 0.0, 0.0, 1.0,
            // batch 1
            5.0, 6.0, 7.0, 8.0,
        ];
        let mut c = vec![0.0_f32; 8];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0, 1.0,
            19.0, 22.0, 43.0, 50.0,
        ];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_matmul_batched_with_scaling() {
        let cfg =
            GemmConfig::new(1, 1, 1).unwrap().with_scaling(2.0, 0.0).with_batch_size(3).unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0_f32; 3];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![8.0, 20.0, 36.0]);
    }

    // -- Unified dispatch ---------------------------------------------------

    #[test]
    fn test_matmul_forward_dispatches_cpu() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap();
        #[rustfmt::skip]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        #[rustfmt::skip]
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f32; 4];
        gemm_forward(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_forward_matches_cpu() {
        let cfg = GemmConfig::new(3, 2, 4).unwrap();
        let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut out_fwd = vec![0.0_f32; 6];
        let mut out_cpu = vec![0.0_f32; 6];
        gemm_forward(&a, &b, &mut out_fwd, &cfg).unwrap();
        gemm_cpu(&a, &b, &mut out_cpu, &cfg).unwrap();
        assert_eq!(out_fwd, out_cpu);
    }

    // -- Property tests -----------------------------------------------------

    #[test]
    fn test_matmul_property_transpose_identity() {
        // (A @ B)^T = B^T @ A^T  (using explicit transposed data, no flags)
        let m = 3;
        let n = 4;
        let k = 5;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();

        // C = A @ B  (C is [M, N])
        let cfg = GemmConfig::new(m, n, k).unwrap();
        let mut c = vec![0.0_f32; m * n];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();

        // Transpose C: C^T[j, i] = C[i, j], shape [N, M]
        let mut c_t = vec![0.0_f32; n * m];
        for i in 0..m {
            for j in 0..n {
                c_t[j * m + i] = c[i * n + j];
            }
        }

        // Build explicit B^T [N, K] and A^T [K, M]
        let mut bt = vec![0.0_f32; n * k];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let mut at = vec![0.0_f32; k * m];
        for i in 0..m {
            for j in 0..k {
                at[j * m + i] = a[i * k + j];
            }
        }

        // D = B^T @ A^T  (no transpose flags: [N,K] @ [K,M] = [N,M])
        let cfg2 = GemmConfig::new(n, m, k).unwrap();
        let mut result = vec![0.0_f32; n * m];
        gemm_cpu(&bt, &at, &mut result, &cfg2).unwrap();

        for i in 0..n * m {
            assert!(
                (c_t[i] - result[i]).abs() < 1e-4,
                "(AB)^T != B^T A^T at {i}: {} vs {}",
                c_t[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_matmul_property_dimension_output() {
        for (m, n, k, batch) in [(1, 1, 1, 1), (4, 5, 6, 1), (2, 3, 4, 3), (10, 1, 7, 2)] {
            let cfg = GemmConfig::new(m, n, k).unwrap().with_batch_size(batch).unwrap();
            assert_eq!(
                cfg.c_len(),
                batch * m * n,
                "C shape mismatch for m={m}, n={n}, batch={batch}"
            );
        }
    }

    #[test]
    fn test_matmul_property_zero_alpha_ignores_product() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap().with_scaling(0.0, 1.0);
        let a = vec![99.0_f32; 4];
        let b = vec![99.0_f32; 4];
        let mut c = vec![1.0, 2.0, 3.0, 4.0];
        gemm_cpu(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_property_zero_beta_ignores_c() {
        let cfg = GemmConfig::new(2, 2, 2).unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c1 = vec![0.0_f32; 4];
        let mut c2 = vec![999.0_f32; 4];
        gemm_cpu(&a, &b, &mut c1, &cfg).unwrap();
        gemm_cpu(&a, &b, &mut c2, &cfg).unwrap();
        assert_eq!(c1, c2);
    }

    // -- GPU launch stub tests ----------------------------------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_matmul_cuda_gemm_launch() {
        let cfg = GemmConfig::new(128, 128, 128).unwrap();
        let a = vec![1.0_f32; 128 * 128];
        let b = vec![1.0_f32; 128 * 128];
        let mut c = vec![0.0_f32; 128 * 128];
        let result = gemm_forward(&a, &b, &mut c, &cfg);
        assert!(result.is_ok(), "GEMM launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_matmul_cuda_batched_launch() {
        let cfg = GemmConfig::new(64, 64, 64).unwrap().with_batch_size(4).unwrap();
        let a = vec![1.0_f32; 4 * 64 * 64];
        let b = vec![1.0_f32; 4 * 64 * 64];
        let mut c = vec![0.0_f32; 4 * 64 * 64];
        let result = gemm_forward(&a, &b, &mut c, &cfg);
        assert!(result.is_ok(), "Batched GEMM launch failed: {result:?}");
    }
}
