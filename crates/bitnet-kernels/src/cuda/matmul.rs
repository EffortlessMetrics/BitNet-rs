//! CUDA dense matrix multiplication kernel with CPU fallback.
//!
//! # Kernel strategy
//!
//! Tiled GEMM for dense floating-point matrices.  The CUDA path uses a
//! 2-D thread-block tiling scheme where each block computes a
//! `tile_m × tile_n` output sub-matrix, streaming `tile_k`-wide slices
//! of the A and B operands through shared memory.
//!
//! Supported configurations:
//!
//! - **f32 × f32 → f32**: Standard single-precision GEMM
//! - **f16 × f16 → f32**: Mixed-precision with FP16 inputs, FP32 accumulator
//! - **Batched**: Multiple independent matmuls with a batch dimension
//! - **Transpose**: Optional transpose of A and/or B operands
//!
//! # CPU fallback
//!
//! [`matmul_cpu`] provides a naive O(n³) reference implementation for
//! correctness testing and non-GPU environments.  The unified
//! [`matmul_forward`] dispatcher tries the GPU path first and falls
//! back transparently.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Data type enum ────────────────────────────────────────────────────

/// Supported matrix element data types for the matmul kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulDtype {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754 half precision).
    F16,
}

// ── Launch configuration ──────────────────────────────────────────────

/// Launch configuration for the dense matmul CUDA kernel.
///
/// Computes `C = alpha * op(A) · op(B) + beta * C` where `op(X)` is
/// either `X` or `X^T` depending on the transpose flags.
///
/// The grid is 2-D: `(ceil(n / tile_n), ceil(m / tile_m))` with an
/// optional batch dimension in `grid_z`.
#[derive(Debug, Clone)]
pub struct MatmulConfig {
    /// Number of output rows after optional transpose of A.
    pub m: usize,
    /// Number of output columns after optional transpose of B.
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Batch count (1 for non-batched).
    pub batch_size: usize,
    /// Transpose the A operand before multiplication.
    pub transpose_a: bool,
    /// Transpose the B operand before multiplication.
    pub transpose_b: bool,
    /// Scalar multiplier for the product (default 1.0).
    pub alpha: f32,
    /// Scalar multiplier for the existing output (default 0.0).
    pub beta: f32,
    /// Element data type for A and B.
    pub dtype: MatmulDtype,
    /// CUDA tile size in the M dimension.
    pub tile_m: u32,
    /// CUDA tile size in the N dimension.
    pub tile_n: u32,
    /// CUDA tile size in the K dimension (shared-memory streaming).
    pub tile_k: u32,
    /// Number of threads per block.
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory for A and B tiles.
    pub shared_mem_bytes: u32,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            m: 1,
            n: 1,
            k: 1,
            batch_size: 1,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
            dtype: MatmulDtype::F32,
            tile_m: 32,
            tile_n: 32,
            tile_k: 32,
            threads_per_block: 256,
            shared_mem_bytes: 8192,
        }
    }
}

impl MatmulConfig {
    /// Create a config tuned for the given matrix dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn for_shape(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("matmul dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }

        // Shared memory: two tiles of tile_m×tile_k and tile_k×tile_n
        // floats (4 bytes each).
        let tile_m = 32u32;
        let tile_n = 32u32;
        let tile_k = 32u32;
        let shared = (tile_m * tile_k + tile_k * tile_n) * 4;

        Ok(Self { m, n, k, tile_m, tile_n, tile_k, shared_mem_bytes: shared, ..Self::default() })
    }

    /// Create a config with a specific tile size (e.g. 16 for smaller
    /// shared-memory footprint or 32 for higher throughput).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension or tile size is zero.
    pub fn for_shape_tiled(m: usize, n: usize, k: usize, tile_size: u32) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("matmul dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        if tile_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "tile_size must be non-zero".into(),
            }
            .into());
        }
        let shared = (tile_size * tile_size + tile_size * tile_size) * 4;
        Ok(Self {
            m,
            n,
            k,
            tile_m: tile_size,
            tile_n: tile_size,
            tile_k: tile_size,
            threads_per_block: tile_size * tile_size,
            shared_mem_bytes: shared,
            ..Self::default()
        })
    }

    /// Set batch size for batched matmul.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "batch_size must be > 0".into() }.into()
            );
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Set transpose flags.
    pub fn with_transpose(mut self, transpose_a: bool, transpose_b: bool) -> Self {
        self.transpose_a = transpose_a;
        self.transpose_b = transpose_b;
        self
    }

    /// Set alpha and beta scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set the element data type.
    pub fn with_dtype(mut self, dtype: MatmulDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, batch_size)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(self.tile_n);
        let grid_y = (self.m as u32).div_ceil(self.tile_m);
        (grid_x, grid_y, self.batch_size as u32)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ── GemmConfig convenience builder ────────────────────────────────────

/// High-level GEMM configuration: C = α·op(A)·op(B) + β·C.
///
/// This is a convenience builder that produces a [`MatmulConfig`].
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Scalar multiplier for the product.
    pub alpha: f32,
    /// Scalar multiplier for the existing output.
    pub beta: f32,
    /// Transpose A before multiplication.
    pub trans_a: bool,
    /// Transpose B before multiplication.
    pub trans_b: bool,
}

impl GemmConfig {
    /// Create a new GEMM config for the given dimensions.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, alpha: 1.0, beta: 0.0, trans_a: false, trans_b: false }
    }

    /// Set scalar multipliers.
    pub fn with_scalars(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set transpose flags.
    pub fn with_transpose(mut self, trans_a: bool, trans_b: bool) -> Self {
        self.trans_a = trans_a;
        self.trans_b = trans_b;
        self
    }

    /// Convert to a [`MatmulConfig`] with default tile parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn to_matmul_config(&self) -> Result<MatmulConfig> {
        MatmulConfig::for_shape(self.m, self.n, self.k).map(|cfg| {
            cfg.with_transpose(self.trans_a, self.trans_b).with_alpha_beta(self.alpha, self.beta)
        })
    }

    /// Convert to a [`MatmulConfig`] with a specific tile size.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension or tile size is zero.
    pub fn to_matmul_config_tiled(&self, tile_size: u32) -> Result<MatmulConfig> {
        MatmulConfig::for_shape_tiled(self.m, self.n, self.k, tile_size).map(|cfg| {
            cfg.with_transpose(self.trans_a, self.trans_b).with_alpha_beta(self.alpha, self.beta)
        })
    }
}

// ── Tiled CPU matmul ─────────────────────────────────────────────────

/// Cache-friendly tiled matrix multiplication (CPU).
///
/// Uses tile blocking to improve data locality. Each tile of the output
/// is computed by streaming `TILE`-wide slices of A and B through the
/// innermost loops, mimicking the shared-memory tiling strategy used by
/// the CUDA kernel.
///
/// Falls back to [`matmul_cpu`] for transpose or batched cases.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with the config.
pub fn matmul_tiled_cpu(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    config: &MatmulConfig,
) -> Result<()> {
    // Delegate transpose/batched variants to the general implementation
    // since tiling with transpose complicates index math for negligible
    // benefit at the CPU level.
    if config.transpose_a || config.transpose_b || config.batch_size > 1 {
        return matmul_cpu(a, b, out, config);
    }

    validate_matmul_buffers(a, b, out, config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;
    let tile = config.tile_m as usize;

    // Apply beta to the entire output first.
    if beta == 0.0 {
        for v in out.iter_mut().take(m * n) {
            *v = 0.0;
        }
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for v in out.iter_mut().take(m * n) {
            *v *= beta;
        }
    }

    // Tiled accumulation: iterate over tiles of (i, j, l).
    let mut i0 = 0;
    while i0 < m {
        let i_end = (i0 + tile).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j_end = (j0 + tile).min(n);
            let mut l0 = 0;
            while l0 < k {
                let l_end = (l0 + tile).min(k);
                for i in i0..i_end {
                    for j in j0..j_end {
                        let mut acc = 0.0f32;
                        for l in l0..l_end {
                            acc += a[i * k + l] * b[l * n + j];
                        }
                        out[i * n + j] += alpha * acc;
                    }
                }
                l0 += tile;
            }
            j0 += tile;
        }
        i0 += tile;
    }

    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────

fn validate_matmul_buffers(a: &[f32], b: &[f32], out: &[f32], config: &MatmulConfig) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;
    let batch = config.batch_size;

    let a_rows = if config.transpose_a { k } else { m };
    let a_cols = if config.transpose_a { m } else { k };
    let b_rows = if config.transpose_b { n } else { k };
    let b_cols = if config.transpose_b { k } else { n };

    let a_required = batch * a_rows * a_cols;
    let b_required = batch * b_rows * b_cols;
    let out_required = batch * m * n;

    if a.len() < a_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A buffer too small: expected >= {a_required}, got {}", a.len()),
        }));
    }
    if b.len() < b_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B buffer too small: expected >= {b_required}, got {}", b.len()),
        }));
    }
    if out.len() < out_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: expected >= {out_required}, got {}",
                out.len()
            ),
        }));
    }
    Ok(())
}

// ── CPU fallback ──────────────────────────────────────────────────────

/// Naive dense matrix multiplication (CPU fallback).
///
/// Computes `C = alpha * op(A) · op(B) + beta * C` for each batch.
///
/// # Layout
///
/// - `a`: row-major `[batch, a_rows, a_cols]` f32 (where shape depends
///   on `transpose_a`)
/// - `b`: row-major `[batch, b_rows, b_cols]` f32 (where shape depends
///   on `transpose_b`)
/// - `out`: row-major `[batch, m, n]` f32
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent with the config.
pub fn matmul_cpu(a: &[f32], b: &[f32], out: &mut [f32], config: &MatmulConfig) -> Result<()> {
    validate_matmul_buffers(a, b, out, config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;

    let a_rows = if config.transpose_a { k } else { m };
    let a_cols = if config.transpose_a { m } else { k };
    let b_rows = if config.transpose_b { n } else { k };
    let b_cols = if config.transpose_b { k } else { n };

    let a_stride = a_rows * a_cols;
    let b_stride = b_rows * b_cols;
    let out_stride = m * n;

    for batch in 0..config.batch_size {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let o_off = batch * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = if config.transpose_a {
                        a[a_off + l * m + i] // A^T: A stored as [k, m], read [l, i]
                    } else {
                        a[a_off + i * k + l] // A: stored as [m, k], read [i, l]
                    };
                    let b_val = if config.transpose_b {
                        b[b_off + j * k + l] // B^T: B stored as [n, k], read [j, l]
                    } else {
                        b[b_off + l * n + j] // B: stored as [k, n], read [l, j]
                    };
                    acc += a_val * b_val;
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx];
            }
        }
    }
    Ok(())
}

/// Naive f16 matrix multiplication (CPU fallback).
///
/// Inputs are packed as `u16` in IEEE 754 half-precision format.
/// Accumulation is performed in f32 for numerical stability.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn matmul_f16_cpu(a: &[u16], b: &[u16], out: &mut [f32], config: &MatmulConfig) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;
    let alpha = config.alpha;
    let beta = config.beta;
    let batch = config.batch_size;

    let a_rows = if config.transpose_a { k } else { m };
    let a_cols = if config.transpose_a { m } else { k };
    let b_rows = if config.transpose_b { n } else { k };
    let b_cols = if config.transpose_b { k } else { n };

    let a_required = batch * a_rows * a_cols;
    let b_required = batch * b_rows * b_cols;
    let out_required = batch * m * n;

    if a.len() < a_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A f16 buffer too small: expected >= {a_required}, got {}", a.len()),
        }));
    }
    if b.len() < b_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B f16 buffer too small: expected >= {b_required}, got {}", b.len()),
        }));
    }
    if out.len() < out_required {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!(
                "output buffer too small: expected >= {out_required}, got {}",
                out.len()
            ),
        }));
    }

    let a_stride = a_rows * a_cols;
    let b_stride = b_rows * b_cols;
    let out_stride = m * n;

    for b_idx in 0..batch {
        let a_off = b_idx * a_stride;
        let b_off = b_idx * b_stride;
        let o_off = b_idx * out_stride;

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = if config.transpose_a {
                        f16_to_f32(a[a_off + l * m + i])
                    } else {
                        f16_to_f32(a[a_off + i * k + l])
                    };
                    let b_val = if config.transpose_b {
                        f16_to_f32(b[b_off + j * k + l])
                    } else {
                        f16_to_f32(b[b_off + l * n + j])
                    };
                    acc += a_val * b_val;
                }
                let idx = o_off + i * n + j;
                out[idx] = alpha * acc + beta * out[idx];
            }
        }
    }
    Ok(())
}

// ── f16 conversion helpers ────────────────────────────────────────────

/// Convert an IEEE 754 half-precision float (u16) to f32.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            return f32::from_bits(sign << 31);
        }
        // Subnormal: convert to normalised f32
        let mut m = mantissa;
        let mut e: i32 = -14; // bias diff: 127 - 15 - (10 - 0)
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // remove implicit bit
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        // Inf / NaN
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    // Normal
    let f32_exp = exponent + 112; // 127 - 15
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Convert an f32 to IEEE 754 half-precision float (u16).
#[cfg(test)]
#[inline(always)]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0 {
        // ±0 or f32 subnormal → f16 zero
        return sign << 15;
    }
    if exponent == 0xFF {
        // Inf/NaN
        let f16_mantissa = (mantissa >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | f16_mantissa;
    }

    let new_exp = exponent - 112; // 127 - 15
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        // Underflow → zero (no subnormal handling for simplicity)
        return sign << 15;
    }
    let f16_mantissa = (mantissa >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
}

// ── CUDA launch stub ──────────────────────────────────────────────────

/// Launch stub for the dense matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_matmul(
    _a: &[f32],
    _b: &[f32],
    _output: &mut [f32],
    config: &MatmulConfig,
) -> Result<()> {
    log::debug!(
        "matmul CUDA stub: m={}, n={}, k={}, batch={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.batch_size,
        config.grid_dim(),
    );
    // TODO: Load PTX module via cudarc, set up shared memory, and
    // launch the tiled GEMM kernel.  The CPU fallback is used until
    // the real kernel is ready.
    Err(KernelError::GpuError {
        reason: "dense matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for the f16 matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_matmul_f16(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    config: &MatmulConfig,
) -> Result<()> {
    log::debug!(
        "matmul f16 CUDA stub: m={}, n={}, k={}, batch={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.batch_size,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "f16 matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// Dense f32 matmul with automatic dispatch: GPU if available, else CPU
/// fallback.
pub fn matmul_forward(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    config: &MatmulConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_matmul(a, b, output, config)
        {
            return Ok(());
        }
    }
    matmul_cpu(a, b, output, config)
}

/// Dense f16 matmul with automatic dispatch: GPU if available, else CPU
/// fallback.
pub fn matmul_f16_forward(
    a: &[u16],
    b: &[u16],
    output: &mut [f32],
    config: &MatmulConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_matmul_f16(a, b, output, config)
        {
            return Ok(());
        }
    }
    matmul_f16_cpu(a, b, output, config)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Naive reference matmul: C = A · B (no transpose, alpha=1, beta=0).
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    // ── config tests ──────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = MatmulConfig::default();
        assert_eq!(cfg.m, 1);
        assert_eq!(cfg.n, 1);
        assert_eq!(cfg.k, 1);
        assert_eq!(cfg.batch_size, 1);
        assert!(!cfg.transpose_a);
        assert!(!cfg.transpose_b);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
        assert_eq!(cfg.dtype, MatmulDtype::F32);
    }

    #[test]
    fn test_config_for_shape() {
        let cfg = MatmulConfig::for_shape(64, 128, 256).unwrap();
        assert_eq!(cfg.m, 64);
        assert_eq!(cfg.n, 128);
        assert_eq!(cfg.k, 256);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // ceil(128/32)
        assert_eq!(gy, 2); // ceil(64/32)
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_rejects_zero_dims() {
        assert!(MatmulConfig::for_shape(0, 8, 8).is_err());
        assert!(MatmulConfig::for_shape(8, 0, 8).is_err());
        assert!(MatmulConfig::for_shape(8, 8, 0).is_err());
    }

    #[test]
    fn test_config_batch_size() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap().with_batch_size(8).unwrap();
        assert_eq!(cfg.batch_size, 8);
        let (_, _, gz) = cfg.grid_dim();
        assert_eq!(gz, 8);
    }

    #[test]
    fn test_config_rejects_zero_batch() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        assert!(cfg.with_batch_size(0).is_err());
    }

    #[test]
    fn test_config_transpose_flags() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap().with_transpose(true, true);
        assert!(cfg.transpose_a);
        assert!(cfg.transpose_b);
    }

    #[test]
    fn test_config_alpha_beta() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap().with_alpha_beta(2.0, 0.5);
        assert_eq!(cfg.alpha, 2.0);
        assert_eq!(cfg.beta, 0.5);
    }

    #[test]
    fn test_config_dtype() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap().with_dtype(MatmulDtype::F16);
        assert_eq!(cfg.dtype, MatmulDtype::F16);
    }

    // ── identity matrix ───────────────────────────────────────────

    #[test]
    fn test_identity_2x2() {
        let a = vec![3.0, -2.0, 5.0, 7.0];
        #[rustfmt::skip]
        let b = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_identity_4x4() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        #[rustfmt::skip]
        let b = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let mut out = vec![0.0f32; 16];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    // ── zero matrix ───────────────────────────────────────────────

    #[test]
    fn test_zero_a_produces_zero() {
        let a = vec![0.0f32; 12];
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let cfg = MatmulConfig::for_shape(3, 4, 3).unwrap();
        let mut out = vec![0.0f32; 12];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[0.0f32; 12], 1e-6);
    }

    #[test]
    fn test_zero_b_produces_zero() {
        let a: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let b = vec![0.0f32; 12];
        let cfg = MatmulConfig::for_shape(3, 4, 3).unwrap();
        let mut out = vec![0.0f32; 12];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[0.0f32; 12], 1e-6);
    }

    // ── known product ─────────────────────────────────────────────

    #[test]
    fn test_known_2x3_times_3x2() {
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        // Expected: [[58, 64], [139, 154]]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        let cfg = MatmulConfig::for_shape(2, 2, 3).unwrap();
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_known_product_matches_naive() {
        let (m, n, k) = (5, 7, 11);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    // ── 1×1 edge case ─────────────────────────────────────────────

    #[test]
    fn test_1x1_matmul() {
        let cfg = MatmulConfig::for_shape(1, 1, 1).unwrap();
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let mut out = vec![0.0f32; 1];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[15.0], 1e-6);
    }

    // ── alpha / beta scalars ──────────────────────────────────────

    #[test]
    fn test_alpha_scaling() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_alpha_beta(2.0, 0.0);
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        // C = 2 * A * I = 2 * A
        assert_close(&out, &[2.0, 4.0, 6.0, 8.0], 1e-6);
    }

    #[test]
    fn test_beta_accumulate() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_alpha_beta(1.0, 1.0);
        let mut out = vec![10.0, 20.0, 30.0, 40.0];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        // C = I + old_C
        assert_close(&out, &[11.0, 20.0, 30.0, 41.0], 1e-6);
    }

    // ── shape validation ──────────────────────────────────────────

    #[test]
    fn test_a_buffer_too_small() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let a = vec![1.0f32; 8]; // need 16
        let b = vec![1.0f32; 16];
        let mut out = vec![0.0f32; 16];
        assert!(matmul_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_b_buffer_too_small() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 8]; // need 16
        let mut out = vec![0.0f32; 16];
        assert!(matmul_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_output_buffer_too_small() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let mut out = vec![0.0f32; 8]; // need 16
        assert!(matmul_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    // ── transpose ─────────────────────────────────────────────────

    #[test]
    fn test_transpose_a() {
        // A stored as [k=3, m=2] (transposed), B as [k=3, n=2]
        // op(A) = A^T = [2, 3], B = [3, 2] → C = [2, 2]
        #[rustfmt::skip]
        let a = vec![
            1.0, 4.0, // col 0 of A^T → row of A^T
            2.0, 5.0,
            3.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        let cfg = MatmulConfig::for_shape(2, 2, 3).unwrap().with_transpose(true, false);
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        // A^T = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        assert_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn test_transpose_b() {
        // A as [m=2, k=3], B stored as [n=2, k=3] (transposed)
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 9.0, 11.0,  // row 0 of B^T
            8.0, 10.0, 12.0, // row 1 of B^T
        ];
        let cfg = MatmulConfig::for_shape(2, 2, 3).unwrap().with_transpose(false, true);
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    #[test]
    fn test_transpose_both() {
        // A stored as [k=3, m=2], B stored as [n=2, k=3]
        #[rustfmt::skip]
        let a = vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 9.0, 11.0,
            8.0, 10.0, 12.0,
        ];
        let cfg = MatmulConfig::for_shape(2, 2, 3).unwrap().with_transpose(true, true);
        let mut out = vec![0.0f32; 4];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    // ── batch matmul ──────────────────────────────────────────────

    #[test]
    fn test_batch_matmul_2_batches() {
        let (m, n, k) = (2, 2, 2);
        #[rustfmt::skip]
        let a = vec![
            // batch 0
            1.0, 2.0,
            3.0, 4.0,
            // batch 1
            5.0, 6.0,
            7.0, 8.0,
        ];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap().with_batch_size(2).unwrap();
        let mut out = vec![0.0f32; 8];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        // Both batches multiply by identity
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_batch_matmul_different_weights() {
        let (m, n, k) = (2, 2, 2);
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        #[rustfmt::skip]
        let b = vec![
            // batch 0: scale by 2
            2.0, 0.0, 0.0, 2.0,
            // batch 1: scale by 3
            3.0, 0.0, 0.0, 3.0,
        ];
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap().with_batch_size(2).unwrap();
        let mut out = vec![0.0f32; 8];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_batch_matmul_large() {
        let (m, n, k, batch) = (8, 4, 6, 3);
        let a: Vec<f32> = (0..(batch * m * k)).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..(batch * k * n)).map(|i| (i as f32 * 0.02).cos()).collect();
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap().with_batch_size(batch).unwrap();
        let mut out = vec![0.0f32; batch * m * n];
        matmul_cpu(&a, &b, &mut out, &cfg).unwrap();

        // Verify each batch independently
        for bi in 0..batch {
            let a_slice = &a[bi * m * k..(bi + 1) * m * k];
            let b_slice = &b[bi * k * n..(bi + 1) * k * n];
            let expected = naive_matmul(a_slice, b_slice, m, n, k);
            let out_slice = &out[bi * m * n..(bi + 1) * m * n];
            assert_close(out_slice, &expected, 1e-4);
        }
    }

    // ── f16 tests ─────────────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip() {
        for &val in &[0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0] {
            let half = f32_to_f16(val);
            let back = f16_to_f32(half);
            assert!((val - back).abs() < 1.0, "roundtrip failed for {val}: got {back}");
        }
    }

    #[test]
    fn test_f16_matmul_identity() {
        let a: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = [3.0f32, 7.0, -2.0, 5.0].iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_dtype(MatmulDtype::F16);
        let mut out = vec![0.0f32; 4];
        matmul_f16_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[3.0, 7.0, -2.0, 5.0], 0.1);
    }

    #[test]
    fn test_f16_matmul_known() {
        let a_f32 = [1.0f32, 2.0, 3.0, 4.0];
        let b_f32 = [5.0f32, 6.0, 7.0, 8.0];
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_dtype(MatmulDtype::F16);
        let mut out = vec![0.0f32; 4];
        matmul_f16_cpu(&a, &b, &mut out, &cfg).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert_close(&out, &[19.0, 22.0, 43.0, 50.0], 0.5);
    }

    #[test]
    fn test_f16_buffer_too_small() {
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let a = vec![0u16; 8];
        let b = vec![0u16; 16];
        let mut out = vec![0.0f32; 16];
        assert!(matmul_f16_cpu(&a, &b, &mut out, &cfg).is_err());
    }

    // ── unified dispatch ──────────────────────────────────────────

    #[test]
    fn test_forward_dispatches_cpu() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let mut out = vec![0.0f32; 4];
        matmul_forward(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_forward_matches_cpu() {
        let (m, n, k) = (6, 8, 10);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();

        let mut out_cpu = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut out_cpu, &cfg).unwrap();

        let mut out_fwd = vec![0.0f32; m * n];
        matmul_forward(&a, &b, &mut out_fwd, &cfg).unwrap();

        assert_close(&out_fwd, &out_cpu, 1e-6);
    }

    #[test]
    fn test_f16_forward_dispatches_cpu() {
        let a: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = [2.0f32, 3.0, 4.0, 5.0].iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap();
        let mut out = vec![0.0f32; 4];
        matmul_f16_forward(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[2.0, 3.0, 4.0, 5.0], 0.1);
    }

    // ── CUDA launch stubs (require GPU hardware) ──────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_matmul_launch() {
        let cfg = MatmulConfig::for_shape(64, 128, 256).unwrap();
        let a = vec![1.0f32; 64 * 256];
        let b = vec![1.0f32; 256 * 128];
        let mut out = vec![0.0f32; 64 * 128];
        let result = matmul_forward(&a, &b, &mut out, &cfg);
        assert!(result.is_ok(), "CUDA matmul launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_batched_matmul_launch() {
        let cfg = MatmulConfig::for_shape(32, 32, 64).unwrap().with_batch_size(4).unwrap();
        let a = vec![1.0f32; 4 * 32 * 64];
        let b = vec![1.0f32; 4 * 64 * 32];
        let mut out = vec![0.0f32; 4 * 32 * 32];
        let result = matmul_forward(&a, &b, &mut out, &cfg);
        assert!(result.is_ok(), "CUDA batched matmul launch failed: {result:?}");
    }

    // ── GemmConfig tests ──────────────────────────────────────────

    #[test]
    fn test_gemm_config_basic() {
        let gc = GemmConfig::new(16, 32, 64);
        let cfg = gc.to_matmul_config().unwrap();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.n, 32);
        assert_eq!(cfg.k, 64);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
        assert!(!cfg.transpose_a);
        assert!(!cfg.transpose_b);
    }

    #[test]
    fn test_gemm_config_with_scalars() {
        let gc = GemmConfig::new(4, 4, 4).with_scalars(2.0, 0.5);
        let cfg = gc.to_matmul_config().unwrap();
        assert_eq!(cfg.alpha, 2.0);
        assert_eq!(cfg.beta, 0.5);
    }

    #[test]
    fn test_gemm_config_with_transpose() {
        let gc = GemmConfig::new(4, 4, 4).with_transpose(true, false);
        let cfg = gc.to_matmul_config().unwrap();
        assert!(cfg.transpose_a);
        assert!(!cfg.transpose_b);
    }

    #[test]
    fn test_gemm_config_tiled_16() {
        let gc = GemmConfig::new(64, 64, 64);
        let cfg = gc.to_matmul_config_tiled(16).unwrap();
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 16);
        assert_eq!(cfg.threads_per_block, 256);
    }

    #[test]
    fn test_gemm_config_zero_dim_error() {
        let gc = GemmConfig::new(0, 4, 4);
        assert!(gc.to_matmul_config().is_err());
    }

    #[test]
    fn test_gemm_config_zero_tile_error() {
        let gc = GemmConfig::new(4, 4, 4);
        assert!(gc.to_matmul_config_tiled(0).is_err());
    }

    // ── for_shape_tiled tests ─────────────────────────────────────

    #[test]
    fn test_for_shape_tiled_16() {
        let cfg = MatmulConfig::for_shape_tiled(128, 128, 128, 16).unwrap();
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 16);
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 8); // 128/16
        assert_eq!(gy, 8);
    }

    #[test]
    fn test_for_shape_tiled_zero_tile_error() {
        assert!(MatmulConfig::for_shape_tiled(4, 4, 4, 0).is_err());
    }

    #[test]
    fn test_for_shape_tiled_zero_dim_error() {
        assert!(MatmulConfig::for_shape_tiled(0, 4, 4, 16).is_err());
    }

    // ── tiled CPU matmul tests ────────────────────────────────────

    #[test]
    fn test_tiled_identity_4x4() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        #[rustfmt::skip]
        let b = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let cfg = MatmulConfig::for_shape(4, 4, 4).unwrap();
        let mut out = vec![0.0f32; 16];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_tiled_matches_naive() {
        let (m, n, k) = (17, 13, 23);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_tiled_matches_naive_tile16() {
        let (m, n, k) = (33, 17, 45);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.07).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.03).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape_tiled(m, n, k, 16).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_tiled_alpha_scaling() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_alpha_beta(3.0, 0.0);
        let mut out = vec![0.0f32; 4];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[3.0, 6.0, 9.0, 12.0], 1e-6);
    }

    #[test]
    fn test_tiled_beta_accumulate() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(2, 2, 2).unwrap().with_alpha_beta(1.0, 2.0);
        let mut out = vec![5.0, 10.0, 15.0, 20.0];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        // C = 1*I + 2*old = I + [10,20,30,40]
        assert_close(&out, &[11.0, 20.0, 30.0, 41.0], 1e-6);
    }

    #[test]
    fn test_tiled_delegates_transpose() {
        #[rustfmt::skip]
        let a = vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ];
        let cfg = MatmulConfig::for_shape(2, 2, 3).unwrap().with_transpose(true, false);
        let mut out_tiled = vec![0.0f32; 4];
        let mut out_cpu = vec![0.0f32; 4];
        matmul_tiled_cpu(&a, &b, &mut out_tiled, &cfg).unwrap();
        matmul_cpu(&a, &b, &mut out_cpu, &cfg).unwrap();
        assert_close(&out_tiled, &out_cpu, 1e-6);
    }

    #[test]
    fn test_tiled_delegates_batch() {
        let (m, n, k) = (2, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap().with_batch_size(2).unwrap();
        let mut out_tiled = vec![0.0f32; 8];
        let mut out_cpu = vec![0.0f32; 8];
        matmul_tiled_cpu(&a, &b, &mut out_tiled, &cfg).unwrap();
        matmul_cpu(&a, &b, &mut out_cpu, &cfg).unwrap();
        assert_close(&out_tiled, &out_cpu, 1e-6);
    }

    #[test]
    fn test_tiled_1x1() {
        let cfg = MatmulConfig::for_shape(1, 1, 1).unwrap();
        let a = vec![7.0f32];
        let b = vec![3.0f32];
        let mut out = vec![0.0f32; 1];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &[21.0], 1e-6);
    }

    // ── non-square matrices ───────────────────────────────────────

    #[test]
    fn test_very_tall_matrix() {
        let (m, n, k) = (128, 2, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_very_wide_matrix() {
        let (m, n, k) = (2, 128, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    // ── large matrix ──────────────────────────────────────────────

    #[test]
    fn test_large_matrix_64x64() {
        let (m, n, k) = (64, 64, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out = vec![0.0f32; m * n];
        matmul_tiled_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out, &expected, 1e-2);
    }

    // ── numerical precision ───────────────────────────────────────

    #[test]
    fn test_tiled_vs_cpu_precision() {
        let (m, n, k) = (31, 29, 37);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out_cpu = vec![0.0f32; m * n];
        let mut out_tiled = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut out_cpu, &cfg).unwrap();
        matmul_tiled_cpu(&a, &b, &mut out_tiled, &cfg).unwrap();
        assert_close(&out_tiled, &out_cpu, 1e-4);
    }

    // ── property tests ────────────────────────────────────────────

    #[test]
    fn test_property_identity_is_neutral() {
        for sz in [1, 3, 7, 16, 33] {
            let a: Vec<f32> = (0..sz * sz).map(|i| (i as f32) * 0.1).collect();
            let mut eye = vec![0.0f32; sz * sz];
            for i in 0..sz {
                eye[i * sz + i] = 1.0;
            }
            let cfg = MatmulConfig::for_shape(sz, sz, sz).unwrap();
            let mut out = vec![0.0f32; sz * sz];
            matmul_tiled_cpu(&a, &eye, &mut out, &cfg).unwrap();
            assert_close(&out, &a, 1e-4);
        }
    }

    #[test]
    fn test_property_zero_annihilates() {
        for sz in [1, 5, 16, 31] {
            let a: Vec<f32> = (0..sz * sz).map(|i| (i as f32) * 0.1).collect();
            let zero = vec![0.0f32; sz * sz];
            let cfg = MatmulConfig::for_shape(sz, sz, sz).unwrap();
            let mut out = vec![0.0f32; sz * sz];
            matmul_tiled_cpu(&a, &zero, &mut out, &cfg).unwrap();
            assert_close(&out, &zero, 1e-6);
        }
    }

    #[test]
    fn test_property_transpose_commutes() {
        // (A * B)^T == B^T * A^T
        let (m, n, k) = (5, 7, 6);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.2).cos()).collect();

        let cfg = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut c = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut c, &cfg).unwrap();

        // Transpose C → (A*B)^T [n×m]
        let mut c_t = vec![0.0f32; n * m];
        for i in 0..m {
            for j in 0..n {
                c_t[j * m + i] = c[i * n + j];
            }
        }

        // B^T * A^T: A-operand = B [k,n], B-operand = A [m,k]
        let cfg2 = MatmulConfig::for_shape(n, m, k).unwrap().with_transpose(true, true);
        let mut c2 = vec![0.0f32; n * m];
        matmul_cpu(&b, &a, &mut c2, &cfg2).unwrap();
        assert_close(&c_t, &c2, 1e-3);
    }

    #[test]
    fn test_property_scalar_distributive() {
        // alpha * (A * B) == (alpha * A) * B
        let (m, n, k) = (4, 5, 6);
        let alpha = 2.5f32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();

        let cfg1 = MatmulConfig::for_shape(m, n, k).unwrap().with_alpha_beta(alpha, 0.0);
        let mut out1 = vec![0.0f32; m * n];
        matmul_cpu(&a, &b, &mut out1, &cfg1).unwrap();

        let a_scaled: Vec<f32> = a.iter().map(|&v| v * alpha).collect();
        let cfg2 = MatmulConfig::for_shape(m, n, k).unwrap();
        let mut out2 = vec![0.0f32; m * n];
        matmul_cpu(&a_scaled, &b, &mut out2, &cfg2).unwrap();

        assert_close(&out1, &out2, 1e-3);
    }

    // ── f16 additional tests ──────────────────────────────────────

    #[test]
    fn test_f16_batch_matmul() {
        let a_f32 = [1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let b_f32 = [3.0f32, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0];
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = MatmulConfig::for_shape(2, 2, 2)
            .unwrap()
            .with_dtype(MatmulDtype::F16)
            .with_batch_size(2)
            .unwrap();
        let mut out = vec![0.0f32; 8];
        matmul_f16_cpu(&a, &b, &mut out, &cfg).unwrap();
        assert_close(&out[0..4], &[3.0, 4.0, 5.0, 6.0], 0.5);
        assert_close(&out[4..8], &[6.0, 8.0, 10.0, 12.0], 0.5);
    }

    #[test]
    fn test_f16_transpose_a() {
        // A stored as [k=2, m=2] (transposed)
        let a_f32 = [1.0f32, 3.0, 2.0, 4.0];
        let b_f32 = [1.0f32, 0.0, 0.0, 1.0];
        let a: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = MatmulConfig::for_shape(2, 2, 2)
            .unwrap()
            .with_dtype(MatmulDtype::F16)
            .with_transpose(true, false);
        let mut out = vec![0.0f32; 4];
        matmul_f16_cpu(&a, &b, &mut out, &cfg).unwrap();
        // A^T = [[1,2],[3,4]], I → [[1,2],[3,4]]
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.5);
    }
}
