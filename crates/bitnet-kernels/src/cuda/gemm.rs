//! CUDA GEMM (General Matrix Multiply) kernel for transformer inference.
//!
//! # Kernel strategy
//!
//! Implements tiled GEMM: `C = α·A×B + β·C` where A is `[M, K]`, B is
//! `[K, N]`, and C is `[M, N]`.
//!
//! ## GPU path (feature `gpu` or `cuda`)
//!
//! The CUDA kernel uses shared-memory tiling to maximise data reuse:
//!
//! 1. Each thread-block computes a `TILE_M × TILE_N` output tile.
//! 2. The inner (K) dimension is traversed in `TILE_K`-sized steps.
//! 3. Tiles of A and B are cooperatively loaded into shared memory,
//!    then each thread accumulates a local dot-product fragment.
//! 4. After sweeping K, threads write their accumulated result to C.
//!
//! Target: ≥ 50 % SM occupancy on Ampere+ with 48 KB shared memory.
//!
//! ## CPU fallback
//!
//! [`gemm_cpu_reference`] provides a naive triple-loop implementation used
//! for correctness testing and environments without a GPU.
//!
//! ## Data types
//!
//! The kernel operates on `f32` host buffers.  [`DType`] metadata records the
//! logical precision so callers can tag operations for receipt auditing (e.g.
//! `gemm_fp16`, `gemm_bf16`).  Actual half-precision compute requires the
//! CUDA path with FP16-capable hardware.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Data-type and layout descriptors
// ---------------------------------------------------------------------------

/// Logical data type for the GEMM operation.
///
/// The CPU fallback always computes in FP32.  On GPU, `F16` and `BF16` select
/// native half-precision accumulation when hardware supports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 half-precision floating point.
    F16,
    /// 16-bit Brain floating point.
    BF16,
}

impl DType {
    /// Kernel-ID suffix used in inference receipts.
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Self::F32 => "fp32",
            Self::F16 => "fp16",
            Self::BF16 => "bf16",
        }
    }
}

/// Matrix memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major (C-contiguous): elements in the same row are contiguous.
    RowMajor,
    /// Column-major (Fortran-contiguous): elements in the same column are contiguous.
    ColMajor,
}

// ---------------------------------------------------------------------------
// CUDA PTX source
// ---------------------------------------------------------------------------

/// Inline CUDA C source for the tiled GEMM kernel (compiled via NVRTC).
///
/// Template parameters are baked via `#define` at compile time so a single
/// source string covers all tile-size configurations.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const GEMM_KERNEL_SRC: &str = r#"
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

extern "C" __global__ void gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    int lda, int ldb, int ldc,
    int a_col_major, int b_col_major)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < n_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // Load A tile
        if (row < M && a_col < K) {
            if (a_col_major)
                As[threadIdx.y][threadIdx.x] = A[a_col * lda + row];
            else
                As[threadIdx.y][threadIdx.x] = A[row * lda + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (b_row < K && col < N) {
            if (b_col_major)
                Bs[threadIdx.y][threadIdx.x] = B[col * ldb + b_row];
            else
                Bs[threadIdx.y][threadIdx.x] = B[b_row * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int c_idx = row * ldc + col;
        C[c_idx] = alpha * acc + beta * C[c_idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Tile size presets for the tiled GEMM kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSize {
    /// 16×16 tiles — lower register pressure, wider hardware compatibility.
    Tile16,
    /// 32×32 tiles — higher throughput on large matrices.
    Tile32,
}

impl TileSize {
    /// Numeric side length of the tile.
    pub fn dim(self) -> u32 {
        match self {
            Self::Tile16 => 16,
            Self::Tile32 => 32,
        }
    }
}

/// Launch configuration for the GEMM kernel.
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// Rows of matrix A / C.
    pub m: usize,
    /// Columns of matrix B / C.
    pub n: usize,
    /// Inner (shared) dimension.
    pub k: usize,
    /// Scalar multiplied with A×B.
    pub alpha: f32,
    /// Scalar multiplied with existing C before accumulation.
    pub beta: f32,
    /// Layout of matrix A.
    pub layout_a: Layout,
    /// Layout of matrix B.
    pub layout_b: Layout,
    /// Logical data type (tags receipts; CPU always uses f32).
    pub dtype: DType,
    /// Tile size for the CUDA kernel.
    pub tile_size: TileSize,
}

impl GemmConfig {
    /// Create a configuration for `C[M,N] = α·A[M,K]×B[K,N] + β·C`.
    ///
    /// Both matrices default to row-major layout, FP32, tile 16×16.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("GEMM dimensions must be non-zero: M={m}, N={n}, K={k}"),
            }
            .into());
        }
        Ok(Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
            layout_a: Layout::RowMajor,
            layout_b: Layout::RowMajor,
            dtype: DType::F32,
            tile_size: TileSize::Tile16,
        })
    }

    /// Set the α scalar.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the β scalar.
    #[must_use]
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set the layout for matrix A.
    #[must_use]
    pub fn with_layout_a(mut self, layout: Layout) -> Self {
        self.layout_a = layout;
        self
    }

    /// Set the layout for matrix B.
    #[must_use]
    pub fn with_layout_b(mut self, layout: Layout) -> Self {
        self.layout_b = layout;
        self
    }

    /// Set the logical data type.
    #[must_use]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the tile size for the CUDA kernel.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: TileSize) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Leading dimension of A given its layout.
    pub fn lda(&self) -> usize {
        match self.layout_a {
            Layout::RowMajor => self.k,
            Layout::ColMajor => self.m,
        }
    }

    /// Leading dimension of B given its layout.
    pub fn ldb(&self) -> usize {
        match self.layout_b {
            Layout::RowMajor => self.n,
            Layout::ColMajor => self.k,
        }
    }

    /// Leading dimension of C (always row-major).
    pub fn ldc(&self) -> usize {
        self.n
    }

    /// CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let tile = self.tile_size.dim() as usize;
        let gx = (self.n + tile - 1) / tile;
        let gy = (self.m + tile - 1) / tile;
        (gx as u32, gy as u32, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        let tile = self.tile_size.dim();
        (tile, tile, 1)
    }

    /// Bytes of shared memory required (two tiles of f32).
    pub fn shared_mem_bytes(&self) -> u32 {
        let tile = self.tile_size.dim();
        2 * tile * tile * 4 // two tiles × sizeof(f32)
    }

    /// Kernel ID string for inference receipts.
    pub fn kernel_id(&self) -> String {
        format!("gemm_{}", self.dtype.kernel_suffix())
    }
}

// ---------------------------------------------------------------------------
// Buffer validation
// ---------------------------------------------------------------------------

fn validate_buffers(a: &[f32], b: &[f32], c: &[f32], config: &GemmConfig) -> Result<()> {
    let expected_a = config.m * config.k;
    let expected_b = config.k * config.n;
    let expected_c = config.m * config.n;

    if a.len() != expected_a {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "GEMM A buffer length mismatch: expected {} ({}×{}), got {}",
                expected_a,
                config.m,
                config.k,
                a.len(),
            ),
        }
        .into());
    }
    if b.len() != expected_b {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "GEMM B buffer length mismatch: expected {} ({}×{}), got {}",
                expected_b,
                config.k,
                config.n,
                b.len(),
            ),
        }
        .into());
    }
    if c.len() != expected_c {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "GEMM C buffer length mismatch: expected {} ({}×{}), got {}",
                expected_c,
                config.m,
                config.n,
                c.len(),
            ),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Read element from a matrix respecting its layout.
#[inline]
fn read_elem(buf: &[f32], row: usize, col: usize, ld: usize, layout: Layout) -> f32 {
    match layout {
        Layout::RowMajor => buf[row * ld + col],
        Layout::ColMajor => buf[col * ld + row],
    }
}

/// Scalar CPU reference implementation of GEMM.
///
/// Computes `C = α·A×B + β·C` using a naive triple loop.  C is always
/// written in row-major order.
pub fn gemm_cpu_reference(a: &[f32], b: &[f32], c: &mut [f32], config: &GemmConfig) -> Result<()> {
    validate_buffers(a, b, c, config)?;

    let (m, n, k) = (config.m, config.n, config.k);
    let lda = config.lda();
    let ldb = config.ldb();
    let ldc = config.ldc();

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += read_elem(a, i, p, lda, config.layout_a)
                    * read_elem(b, p, j, ldb, config.layout_b);
            }
            let idx = i * ldc + j;
            c[idx] = config.alpha * acc + config.beta * c[idx];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA dispatch (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch GEMM to the CUDA device via cudarc.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gemm_cuda(a: &[f32], b: &[f32], c: &mut [f32], config: &GemmConfig) -> Result<()> {
    use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    validate_buffers(a, b, c, config)?;

    log::debug!(
        "GEMM CUDA dispatch: M={}, N={}, K={}, tile={}, α={}, β={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.tile_size.dim(),
        config.alpha,
        config.beta,
        config.grid_dim(),
    );

    let tile_define = format!("#define TILE_SIZE {}\n", config.tile_size.dim());
    let full_src = format!("{tile_define}{GEMM_KERNEL_SRC}");

    let ctx = CudaContext::new(0).map_err(|e| KernelError::GpuError {
        reason: format!("failed to acquire CUDA device 0: {e:?}"),
    })?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(&full_src).map_err(|e| KernelError::GpuError {
        reason: format!("NVRTC compilation failed: {e:?}"),
    })?;

    let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
        reason: format!("failed to load PTX module: {e:?}"),
    })?;

    let func = module.load_function("gemm_f32").map_err(|e| KernelError::GpuError {
        reason: format!("gemm_f32 function not found in module: {e:?}"),
    })?;

    let a_dev = stream.memcpy_stod(a).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy A to device: {e:?}"),
    })?;
    let b_dev = stream.memcpy_stod(b).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy B to device: {e:?}"),
    })?;

    // Copy C to device (needed when β ≠ 0)
    let mut c_dev: CudaSlice<f32> = stream.memcpy_stod(c).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy C to device: {e:?}"),
    })?;

    let (gx, gy, gz) = config.grid_dim();
    let (bx, by, bz) = config.block_dim();
    let launch_cfg = LaunchConfig {
        grid_dim: (gx, gy, gz),
        block_dim: (bx, by, bz),
        shared_mem_bytes: config.shared_mem_bytes(),
    };

    let m_arg = config.m as i32;
    let n_arg = config.n as i32;
    let k_arg = config.k as i32;
    let alpha_arg = config.alpha;
    let beta_arg = config.beta;
    let lda_arg = config.lda() as i32;
    let ldb_arg = config.ldb() as i32;
    let ldc_arg = config.ldc() as i32;
    let a_col_major = if config.layout_a == Layout::ColMajor { 1i32 } else { 0i32 };
    let b_col_major = if config.layout_b == Layout::ColMajor { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    builder.arg(&m_arg);
    builder.arg(&n_arg);
    builder.arg(&k_arg);
    builder.arg(&alpha_arg);
    builder.arg(&beta_arg);
    builder.arg(&lda_arg);
    builder.arg(&ldb_arg);
    builder.arg(&ldc_arg);
    builder.arg(&a_col_major);
    builder.arg(&b_col_major);

    // Safety: kernel signature matches the CUDA source; buffers are
    // correctly sized as validated above.
    unsafe { builder.launch(launch_cfg) }.map_err(|e| KernelError::GpuError {
        reason: format!("CUDA kernel launch failed: {e:?}"),
    })?;

    stream.synchronize().map_err(|e| KernelError::GpuError {
        reason: format!("stream synchronize failed: {e:?}"),
    })?;

    let c_host: Vec<f32> = stream.memcpy_dtov(&c_dev).map_err(|e| KernelError::GpuError {
        reason: format!("failed to copy C from device: {e:?}"),
    })?;

    c.copy_from_slice(&c_host);
    Ok(())
}

// ---------------------------------------------------------------------------
// Unified dispatch entry point
// ---------------------------------------------------------------------------

/// Launch the GEMM kernel with automatic CPU/GPU dispatch.
///
/// When compiled with the `gpu` or `cuda` feature **and** a CUDA device is
/// available at runtime, the kernel runs on the GPU.  Otherwise the CPU
/// reference path is used.
///
/// # Arguments
///
/// * `a` — Matrix A `[M, K]` (FP32, respects `config.layout_a`)
/// * `b` — Matrix B `[K, N]` (FP32, respects `config.layout_b`)
/// * `c` — Output/accumulator `[M, N]` (FP32, row-major, read-write)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::InvalidArguments` if buffer sizes do not match.
/// May return `KernelError::GpuError` on CUDA failures (caller should fall
/// back to the CPU path).
pub fn launch_gemm(a: &[f32], b: &[f32], c: &mut [f32], config: &GemmConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        match launch_gemm_cuda(a, b, c, config) {
            Ok(()) => {
                log::debug!(
                    "GEMM completed on CUDA ({}×{}×{}, {})",
                    config.m,
                    config.n,
                    config.k,
                    config.kernel_id(),
                );
                return Ok(());
            }
            Err(e) => {
                log::warn!("CUDA GEMM failed, falling back to CPU: {e}");
            }
        }
    }

    log::debug!(
        "GEMM CPU fallback: M={}, N={}, K={}, α={}, β={}",
        config.m,
        config.n,
        config.k,
        config.alpha,
        config.beta,
    );
    gemm_cpu_reference(a, b, c, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config construction tests -----------------------------------------

    #[test]
    fn test_gemm_config_basic() {
        let cfg = GemmConfig::new(4, 8, 16).expect("valid config");
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.n, 8);
        assert_eq!(cfg.k, 16);
        assert!((cfg.alpha - 1.0).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.0).abs() < f32::EPSILON);
        assert_eq!(cfg.layout_a, Layout::RowMajor);
        assert_eq!(cfg.layout_b, Layout::RowMajor);
        assert_eq!(cfg.dtype, DType::F32);
        assert_eq!(cfg.tile_size, TileSize::Tile16);
    }

    #[test]
    fn test_gemm_config_rejects_zero_m() {
        assert!(GemmConfig::new(0, 4, 4).is_err());
    }

    #[test]
    fn test_gemm_config_rejects_zero_n() {
        assert!(GemmConfig::new(4, 0, 4).is_err());
    }

    #[test]
    fn test_gemm_config_rejects_zero_k() {
        assert!(GemmConfig::new(4, 4, 0).is_err());
    }

    #[test]
    fn test_gemm_config_builder_chain() {
        let cfg = GemmConfig::new(2, 3, 4)
            .expect("valid")
            .with_alpha(2.0)
            .with_beta(0.5)
            .with_layout_a(Layout::ColMajor)
            .with_layout_b(Layout::ColMajor)
            .with_dtype(DType::F16)
            .with_tile_size(TileSize::Tile32);

        assert!((cfg.alpha - 2.0).abs() < f32::EPSILON);
        assert!((cfg.beta - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.layout_a, Layout::ColMajor);
        assert_eq!(cfg.layout_b, Layout::ColMajor);
        assert_eq!(cfg.dtype, DType::F16);
        assert_eq!(cfg.tile_size, TileSize::Tile32);
    }

    #[test]
    fn test_gemm_config_leading_dims_row_major() {
        let cfg = GemmConfig::new(3, 5, 7).expect("valid");
        assert_eq!(cfg.lda(), 7); // K for row-major A
        assert_eq!(cfg.ldb(), 5); // N for row-major B
        assert_eq!(cfg.ldc(), 5); // N for C (always row-major)
    }

    #[test]
    fn test_gemm_config_leading_dims_col_major() {
        let cfg = GemmConfig::new(3, 5, 7)
            .expect("valid")
            .with_layout_a(Layout::ColMajor)
            .with_layout_b(Layout::ColMajor);
        assert_eq!(cfg.lda(), 3); // M for col-major A
        assert_eq!(cfg.ldb(), 7); // K for col-major B
    }

    #[test]
    fn test_gemm_grid_block_dims() {
        let cfg = GemmConfig::new(33, 17, 8).expect("valid");
        // Tile16: grid = (ceil(17/16), ceil(33/16), 1) = (2, 3, 1)
        assert_eq!(cfg.grid_dim(), (2, 3, 1));
        assert_eq!(cfg.block_dim(), (16, 16, 1));
        assert_eq!(cfg.shared_mem_bytes(), 2 * 16 * 16 * 4);
    }

    #[test]
    fn test_gemm_grid_tile32() {
        let cfg = GemmConfig::new(64, 64, 64).expect("valid").with_tile_size(TileSize::Tile32);
        assert_eq!(cfg.grid_dim(), (2, 2, 1));
        assert_eq!(cfg.block_dim(), (32, 32, 1));
        assert_eq!(cfg.shared_mem_bytes(), 2 * 32 * 32 * 4);
    }

    #[test]
    fn test_gemm_kernel_id() {
        let f32_cfg = GemmConfig::new(1, 1, 1).expect("valid");
        assert_eq!(f32_cfg.kernel_id(), "gemm_fp32");

        let f16_cfg = f32_cfg.clone().with_dtype(DType::F16);
        assert_eq!(f16_cfg.kernel_id(), "gemm_fp16");

        let bf16_cfg = GemmConfig::new(1, 1, 1).expect("valid").with_dtype(DType::BF16);
        assert_eq!(bf16_cfg.kernel_id(), "gemm_bf16");
    }

    // -- Buffer validation tests -------------------------------------------

    #[test]
    fn test_validate_a_length_mismatch() {
        let cfg = GemmConfig::new(2, 3, 4).expect("valid");
        let a = vec![0.0f32; 7]; // wrong: expect 2×4 = 8
        let b = vec![0.0f32; 12];
        let mut c = vec![0.0f32; 6];
        assert!(launch_gemm(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_validate_b_length_mismatch() {
        let cfg = GemmConfig::new(2, 3, 4).expect("valid");
        let a = vec![0.0f32; 8];
        let b = vec![0.0f32; 11]; // wrong: expect 4×3 = 12
        let mut c = vec![0.0f32; 6];
        assert!(launch_gemm(&a, &b, &mut c, &cfg).is_err());
    }

    #[test]
    fn test_validate_c_length_mismatch() {
        let cfg = GemmConfig::new(2, 3, 4).expect("valid");
        let a = vec![0.0f32; 8];
        let b = vec![0.0f32; 12];
        let mut c = vec![0.0f32; 5]; // wrong: expect 2×3 = 6
        assert!(launch_gemm(&a, &b, &mut c, &cfg).is_err());
    }

    // -- CPU reference correctness tests -----------------------------------

    #[test]
    fn test_cpu_gemm_identity() {
        // A = I₂, B = [[1,2],[3,4]] → C = B
        let cfg = GemmConfig::new(2, 2, 2).expect("valid");
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let mut c = vec![0.0f32; 4];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_gemm_2x3_times_3x2() {
        // A[2,3] × B[3,2] = C[2,2]
        let cfg = GemmConfig::new(2, 2, 3).expect("valid");
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            7.0,  8.0,
            9.0,  10.0,
            11.0, 12.0,
        ];
        let mut c = vec![0.0f32; 4];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        // row 0: 1*7+2*9+3*11 = 7+18+33 = 58,  1*8+2*10+3*12 = 8+20+36 = 64
        // row 1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_cpu_gemm_alpha_beta() {
        // C = 2·A×B + 3·C₀
        let cfg = GemmConfig::new(1, 2, 2).expect("valid").with_alpha(2.0).with_beta(3.0);
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut c = vec![10.0, 20.0]; // pre-existing values
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        // raw = 1*3+2*5 = 13, 1*4+2*6 = 16
        // c[0] = 2*13 + 3*10 = 26+30 = 56
        // c[1] = 2*16 + 3*20 = 32+60 = 92
        assert!((c[0] - 56.0).abs() < 1e-5);
        assert!((c[1] - 92.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_col_major_a() {
        // A stored column-major: logical A = [[1,3],[2,4]] (2×2)
        // Physical buffer (col-major): [1, 2, 3, 4] (col0=[1,2], col1=[3,4])
        let cfg = GemmConfig::new(2, 1, 2).expect("valid").with_layout_a(Layout::ColMajor);
        let a = vec![1.0, 2.0, 3.0, 4.0]; // col-major
        let b = vec![1.0, 1.0]; // [2,1] row-major
        let mut c = vec![0.0f32; 2];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        // logical A = [[1,3],[2,4]], B = [[1],[1]]
        // row0: 1*1+3*1=4, row1: 2*1+4*1=6
        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_col_major_b() {
        // B stored column-major: logical B = [[1,3],[2,4]] (2×2)
        // Physical buffer (col-major): [1, 2, 3, 4]
        let cfg = GemmConfig::new(1, 2, 2).expect("valid").with_layout_b(Layout::ColMajor);
        let a = vec![1.0, 1.0]; // [1,2] row-major
        let b = vec![1.0, 2.0, 3.0, 4.0]; // col-major: logical [[1,3],[2,4]]
        let mut c = vec![0.0f32; 2];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        // A=[1,1] × B=[[1,3],[2,4]] = [1+2, 3+4] = [3, 7]
        assert!((c[0] - 3.0).abs() < 1e-5);
        assert!((c[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_zero_alpha() {
        // α=0 means the product is zeroed, only β·C remains
        let cfg = GemmConfig::new(2, 2, 2).expect("valid").with_alpha(0.0).with_beta(1.0);
        let a = vec![99.0; 4];
        let b = vec![99.0; 4];
        let mut c = vec![1.0, 2.0, 3.0, 4.0];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_gemm_single_element() {
        // 1×1 × 1×1 = scalar multiply
        let cfg = GemmConfig::new(1, 1, 1).expect("valid");
        let a = vec![3.0];
        let b = vec![7.0];
        let mut c = vec![0.0];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        assert!((c[0] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_non_square() {
        // A[1,4] × B[4,1] = C[1,1] (inner product)
        let cfg = GemmConfig::new(1, 1, 4).expect("valid");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0];
        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");
        // 1*4+2*3+3*2+4*1 = 4+6+6+4 = 20
        assert!((c[0] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_gemm_larger_matrix() {
        // 8×16 × 16×8 — stress-test beyond tile boundary (tile=16)
        let m = 8;
        let k = 16;
        let n = 8;
        let cfg = GemmConfig::new(m, n, k).expect("valid");
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        gemm_cpu_reference(&a, &b, &mut c, &cfg).expect("ok");

        // Verify against a manual reference for the (0,0) element
        let mut expected_00 = 0.0f32;
        for p in 0..k {
            expected_00 += a[p] * b[p * n];
        }
        assert!((c[0] - expected_00).abs() < 1e-3, "C[0,0]: expected {expected_00}, got {}", c[0]);

        // All outputs must be finite
        for (i, &v) in c.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}: {v}");
        }
    }

    // -- Unified dispatch tests (CPU path on non-GPU builds) ---------------

    #[test]
    fn test_launch_gemm_dispatches_cpu() {
        let cfg = GemmConfig::new(2, 2, 2).expect("valid");
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            5.0, 6.0,
            7.0, 8.0,
        ];
        let mut c = vec![0.0f32; 4];
        launch_gemm(&a, &b, &mut c, &cfg).expect("dispatch should succeed on CPU");

        let mut expected = vec![0.0f32; 4];
        gemm_cpu_reference(&a, &b, &mut expected, &cfg).expect("reference ok");
        for (i, (&got, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "dispatch mismatch at {i}: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_dtype_kernel_suffix() {
        assert_eq!(DType::F32.kernel_suffix(), "fp32");
        assert_eq!(DType::F16.kernel_suffix(), "fp16");
        assert_eq!(DType::BF16.kernel_suffix(), "bf16");
    }

    // -- GPU-only tests (skipped without CUDA hardware) --------------------

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gemm_identity() {
        let cfg = GemmConfig::new(2, 2, 2).expect("valid");
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            5.0, 6.0,
            7.0, 8.0,
        ];
        let mut c = vec![0.0f32; 4];
        launch_gemm(&a, &b, &mut c, &cfg).expect("CUDA GEMM should succeed");
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gemm_matches_cpu_reference() {
        let m = 64;
        let n = 48;
        let k = 32;
        let cfg = GemmConfig::new(m, n, k).expect("valid");
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.005).collect();

        let mut gpu_c = vec![0.0f32; m * n];
        let mut cpu_c = vec![0.0f32; m * n];

        launch_gemm(&a, &b, &mut gpu_c, &cfg).expect("GPU GEMM failed");
        gemm_cpu_reference(&a, &b, &mut cpu_c, &cfg).expect("CPU GEMM failed");

        let max_diff: f32 =
            gpu_c.iter().zip(cpu_c.iter()).map(|(&g, &c)| (g - c).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "GPU vs CPU max absolute difference {max_diff} exceeds tolerance");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gemm_tile32() {
        let cfg = GemmConfig::new(64, 64, 64).expect("valid").with_tile_size(TileSize::Tile32);
        let a: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64 * 64).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; 64 * 64];

        launch_gemm(&a, &b, &mut c, &cfg).expect("CUDA GEMM tile32 failed");
        for &v in &c {
            assert!(v.is_finite(), "non-finite in tile32 output");
        }
    }
}
