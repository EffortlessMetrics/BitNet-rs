//! CUDA quantized GEMM kernels for INT2/INT4 matrix multiplication.
//!
//! # Kernel strategy
//!
//! Provides fused dequantization + GEMM for quantized weight matrices
//! targeting BitNet 1-bit and low-bit inference workloads:
//!
//! - **INT2 × INT2 → FP32**: Core BitNet ternary GEMM (`quantized_gemm_i2`)
//! - **INT4 × INT4 → FP32**: 4-bit weight GEMM (`quantized_gemm_i4`)
//! - **INT2 × FP16 → FP32**: Mixed-precision activation–weight GEMM
//!   (`quantized_gemm_mixed`)
//! - **Dequant-on-the-fly GEMM**: Streaming dequantization without
//!   materialising the full FP32 weight matrix (`quantized_dequant_gemm`)
//!
//! Each path supports auto-tuned tile strategies (Small / Medium / Large)
//! selected by [`select_tile_strategy`] based on problem dimensions and
//! available shared memory.
//!
//! # Tile strategies
//!
//! | Strategy | Tile M×N×K | Target          |
//! |----------|-----------|-----------------|
//! | Small    | 16×16×16  | Small matrices  |
//! | Medium   | 32×32×32  | General purpose |
//! | Large    | 64×64×32  | Large matrices  |
//! | Auto     | adaptive  | Best heuristic  |
//!
//! # CPU fallback
//!
//! All kernels have CPU-only reference implementations that are always
//! compiled.  The GPU launch functions are feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::fmt;

use bitnet_common::{KernelError, Result};

// ── Accumulator type ──────────────────────────────────────────────────

/// Floating-point accumulator precision for quantized GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorType {
    /// 32-bit float accumulator (default, highest accuracy).
    F32,
    /// 16-bit float accumulator (lower precision, higher throughput on
    /// tensor cores).
    F16,
}

impl fmt::Display for AccumulatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccumulatorType::F32 => write!(f, "f32"),
            AccumulatorType::F16 => write!(f, "f16"),
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────

/// Errors specific to quantized GEMM operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedGemmError {
    /// Matrix dimensions are incompatible for multiplication.
    ShapeMismatch {
        /// A rows.
        m: usize,
        /// A cols / B rows (must agree).
        k_a: usize,
        /// B cols / B rows.
        k_b: usize,
        /// B cols.
        n: usize,
    },
    /// A dimension is zero.
    ZeroDimension {
        /// Which dimension is zero.
        dim_name: &'static str,
    },
    /// Alignment requirements not met.
    AlignmentError {
        /// Required alignment in elements.
        required: usize,
        /// Actual dimension value.
        actual: usize,
        /// Which dimension.
        dim_name: &'static str,
    },
    /// Tile configuration is invalid.
    InvalidTileConfig {
        /// Human-readable reason.
        reason: String,
    },
    /// Accumulator buffer is too small.
    BufferTooSmall {
        /// Required number of elements.
        required: usize,
        /// Actual buffer length.
        actual: usize,
    },
}

impl fmt::Display for QuantizedGemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizedGemmError::ShapeMismatch { m, k_a, k_b, n } => {
                write!(
                    f,
                    "quantized GEMM shape mismatch: A is [{m}×{k_a}], \
                     B is [{k_b}×{n}] — inner dimensions must agree"
                )
            }
            QuantizedGemmError::ZeroDimension { dim_name } => {
                write!(f, "quantized GEMM dimension '{dim_name}' must be non-zero")
            }
            QuantizedGemmError::AlignmentError { required, actual, dim_name } => {
                write!(
                    f,
                    "quantized GEMM alignment: {dim_name}={actual} \
                     must be a multiple of {required}"
                )
            }
            QuantizedGemmError::InvalidTileConfig { reason } => {
                write!(f, "invalid tile config: {reason}")
            }
            QuantizedGemmError::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "output buffer too small: need {required} elements, \
                     got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for QuantizedGemmError {}

impl From<QuantizedGemmError> for KernelError {
    fn from(e: QuantizedGemmError) -> Self {
        KernelError::InvalidArguments { reason: e.to_string() }
    }
}

impl From<QuantizedGemmError> for bitnet_common::BitNetError {
    fn from(e: QuantizedGemmError) -> Self {
        bitnet_common::BitNetError::Kernel(e.into())
    }
}

// ── Tile strategy ─────────────────────────────────────────────────────

/// Tile strategy for CUDA thread-block decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileStrategy {
    /// 16×16×16 tiles — good for small matrices or limited shared memory.
    Small,
    /// 32×32×32 tiles — balanced for most workloads.
    Medium,
    /// 64×64×32 tiles — high-throughput for large matrices.
    Large,
    /// Auto-select based on matrix dimensions.
    Auto,
}

impl TileStrategy {
    /// Resolve `Auto` to a concrete strategy.
    fn resolve(self, m: usize, n: usize, k: usize) -> TileStrategy {
        match self {
            TileStrategy::Auto => select_tile_strategy(m, n, k),
            other => other,
        }
    }

    /// Tile dimensions `(tile_m, tile_n, tile_k)` for this strategy.
    pub fn tile_dims(self) -> (u32, u32, u32) {
        match self {
            TileStrategy::Small => (16, 16, 16),
            TileStrategy::Medium => (32, 32, 32),
            TileStrategy::Large => (64, 64, 32),
            TileStrategy::Auto => (32, 32, 32), // default if not resolved
        }
    }

    /// Threads per block for this strategy.
    pub fn threads_per_block(self) -> u32 {
        match self {
            TileStrategy::Small => 256,
            TileStrategy::Medium => 256,
            TileStrategy::Large => 256,
            TileStrategy::Auto => 256,
        }
    }

    /// Shared memory bytes (two tiles of `tile_m×tile_k` and `tile_k×tile_n`
    /// f32 elements).
    pub fn shared_mem_bytes(self) -> u32 {
        let (tm, tn, tk) = self.tile_dims();
        (tm * tk + tk * tn) * 4
    }
}

/// Select the best tile strategy based on matrix dimensions.
///
/// Heuristic:
/// - M or N ≤ 64 and K ≤ 256 → Small
/// - M or N ≤ 512 → Medium
/// - Otherwise → Large
pub fn select_tile_strategy(m: usize, n: usize, k: usize) -> TileStrategy {
    let max_mn = m.max(n);
    if max_mn <= 64 && k <= 256 {
        TileStrategy::Small
    } else if max_mn <= 512 {
        TileStrategy::Medium
    } else {
        TileStrategy::Large
    }
}

// ── Configuration ─────────────────────────────────────────────────────

/// Launch configuration for quantized GEMM kernels.
#[derive(Debug, Clone)]
pub struct QuantizedGemmConfig {
    /// Number of output rows.
    pub m: usize,
    /// Number of output columns.
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Tile strategy for thread-block decomposition.
    pub tile_strategy: TileStrategy,
    /// CUDA tile size M.
    pub tile_m: u32,
    /// CUDA tile size N.
    pub tile_n: u32,
    /// CUDA tile size K.
    pub tile_k: u32,
    /// Whether to use tensor cores (SM 7.0+).
    pub use_tensor_cores: bool,
    /// Accumulator precision.
    pub accumulator_type: AccumulatorType,
    /// Number of threads per block.
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory.
    pub shared_mem_bytes: u32,
}

impl QuantizedGemmConfig {
    /// Create a configuration for the given matrix dimensions with auto
    /// tile strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self> {
        if m == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "m" }.into());
        }
        if n == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "n" }.into());
        }
        if k == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "k" }.into());
        }
        let strategy = select_tile_strategy(m, n, k);
        let (tile_m, tile_n, tile_k) = strategy.tile_dims();
        Ok(Self {
            m,
            n,
            k,
            tile_strategy: strategy,
            tile_m,
            tile_n,
            tile_k,
            use_tensor_cores: false,
            accumulator_type: AccumulatorType::F32,
            threads_per_block: strategy.threads_per_block(),
            shared_mem_bytes: strategy.shared_mem_bytes(),
        })
    }

    /// Create a configuration with an explicit tile strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn with_strategy(m: usize, n: usize, k: usize, strategy: TileStrategy) -> Result<Self> {
        if m == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "m" }.into());
        }
        if n == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "n" }.into());
        }
        if k == 0 {
            return Err(QuantizedGemmError::ZeroDimension { dim_name: "k" }.into());
        }
        let resolved = strategy.resolve(m, n, k);
        let (tile_m, tile_n, tile_k) = resolved.tile_dims();
        Ok(Self {
            m,
            n,
            k,
            tile_strategy: resolved,
            tile_m,
            tile_n,
            tile_k,
            use_tensor_cores: false,
            accumulator_type: AccumulatorType::F32,
            threads_per_block: resolved.threads_per_block(),
            shared_mem_bytes: resolved.shared_mem_bytes(),
        })
    }

    /// Enable tensor core usage (Volta SM 7.0+).
    #[must_use]
    pub fn with_tensor_cores(mut self, enable: bool) -> Self {
        self.use_tensor_cores = enable;
        self
    }

    /// Set the accumulator type.
    #[must_use]
    pub fn with_accumulator(mut self, acc: AccumulatorType) -> Self {
        self.accumulator_type = acc;
        self
    }

    /// Compute the CUDA grid dimensions for this configuration.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let gx = (self.n as u32).div_ceil(self.tile_n);
        let gy = (self.m as u32).div_ceil(self.tile_m);
        (gx, gy, 1)
    }

    /// Compute the CUDA block dimensions for this configuration.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Total FLOPs for this GEMM (2 × M × N × K).
    pub fn total_flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }
}

// ── Performance metrics ───────────────────────────────────────────────

/// Performance metrics for a completed quantized GEMM operation.
#[derive(Debug, Clone, Copy)]
pub struct GemmPerformanceMetrics {
    /// Achieved GFLOP/s.
    pub gflops: f64,
    /// Memory bandwidth utilisation (0.0–1.0).
    pub bandwidth_utilization: f64,
    /// Estimated SM occupancy (0.0–1.0).
    pub occupancy: f64,
    /// Total floating-point operations performed.
    pub total_flops: u64,
    /// Elapsed wall time in seconds.
    pub elapsed_secs: f64,
}

impl GemmPerformanceMetrics {
    /// Compute metrics from GEMM dimensions and elapsed wall time.
    ///
    /// `peak_bandwidth_gbps` is the theoretical device bandwidth in GB/s.
    ///
    /// Returns `None` if `elapsed_secs` is not positive.
    pub fn compute(
        m: usize,
        n: usize,
        k: usize,
        elapsed_secs: f64,
        peak_bandwidth_gbps: f64,
    ) -> Option<Self> {
        if elapsed_secs <= 0.0 || peak_bandwidth_gbps <= 0.0 {
            return None;
        }
        let total_flops = 2u64 * m as u64 * n as u64 * k as u64;
        let gflops = total_flops as f64 / elapsed_secs / 1e9;

        // Minimum memory traffic: read A (m×k) + read B (k×n) + write C (m×n),
        // all in f32 (4 bytes).  Quantised operands are smaller, but we use
        // the dequantised size as upper-bound estimate.
        let bytes_transferred = ((m * k) as f64 + (k * n) as f64 + (m * n) as f64) * 4.0;
        let bandwidth_gbps = bytes_transferred / elapsed_secs / 1e9;
        let bandwidth_utilization = (bandwidth_gbps / peak_bandwidth_gbps).min(1.0);

        // Rough occupancy estimate based on shared-memory pressure.
        let strategy = select_tile_strategy(m, n, k);
        let smem = strategy.shared_mem_bytes() as f64;
        let max_smem: f64 = 49152.0; // 48 KiB (Ampere default)
        let occupancy = (1.0 - smem / max_smem).clamp(0.1, 1.0);

        Some(Self { gflops, bandwidth_utilization, occupancy, total_flops, elapsed_secs })
    }
}

// ── Input validation ──────────────────────────────────────────────────

/// Validate matrix dimensions and buffer sizes for quantized GEMM.
///
/// # Errors
///
/// Returns [`QuantizedGemmError`] on zero dimensions, inner-dimension
/// mismatch, or output buffer too small.
pub fn validate_gemm_inputs(
    m: usize,
    k_a: usize,
    k_b: usize,
    n: usize,
    output_len: usize,
) -> std::result::Result<(), QuantizedGemmError> {
    if m == 0 {
        return Err(QuantizedGemmError::ZeroDimension { dim_name: "m" });
    }
    if k_a == 0 {
        return Err(QuantizedGemmError::ZeroDimension { dim_name: "k (A cols)" });
    }
    if k_b == 0 {
        return Err(QuantizedGemmError::ZeroDimension { dim_name: "k (B rows)" });
    }
    if n == 0 {
        return Err(QuantizedGemmError::ZeroDimension { dim_name: "n" });
    }
    if k_a != k_b {
        return Err(QuantizedGemmError::ShapeMismatch { m, k_a, k_b, n });
    }
    let required = m * n;
    if output_len < required {
        return Err(QuantizedGemmError::BufferTooSmall { required, actual: output_len });
    }
    Ok(())
}

/// Check that a dimension is aligned to `alignment`.
pub fn check_alignment(
    dim: usize,
    alignment: usize,
    dim_name: &'static str,
) -> std::result::Result<(), QuantizedGemmError> {
    if alignment == 0 {
        return Err(QuantizedGemmError::InvalidTileConfig {
            reason: "alignment must be non-zero".into(),
        });
    }
    if !dim.is_multiple_of(alignment) {
        return Err(QuantizedGemmError::AlignmentError {
            required: alignment,
            actual: dim,
            dim_name,
        });
    }
    Ok(())
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C source for quantized GEMM kernels (INT2, INT4, mixed).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const QUANTIZED_GEMM_KERNEL_SRC: &str = r#"
// Quantized GEMM kernels for BitNet inference.
// Each weight element is packed: 4 values per byte for INT2,
// 2 values per byte for INT4.

/// INT2 quantized GEMM: A (INT2 packed) × B (INT2 packed) → C (FP32).
extern "C" __global__
void quantized_gemm_i2_kernel(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    const float* __restrict__ scales_a,
    const float* __restrict__ scales_b,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k
) {
    int row = blockIdx.y * tile_m + threadIdx.x;
    int col = blockIdx.x * tile_n + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    int k_packed = (K + 3) / 4;
    for (int kb = 0; kb < k_packed; ++kb) {
        unsigned char a_byte = A_packed[row * k_packed + kb];
        unsigned char b_byte = B_packed[col * k_packed + kb];
        for (int sub = 0; sub < 4 && (kb * 4 + sub) < K; ++sub) {
            int a_val = ((int)((a_byte >> (sub * 2)) & 0x3)) - 1;
            int b_val = ((int)((b_byte >> (sub * 2)) & 0x3)) - 1;
            acc += (float)(a_val * b_val);
        }
    }
    float sa = scales_a[row];
    float sb = scales_b[col];
    C[row * N + col] = acc * sa * sb;
}

/// INT4 quantized GEMM: A (INT4 packed) × B (INT4 packed) → C (FP32).
extern "C" __global__
void quantized_gemm_i4_kernel(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ B_packed,
    float* __restrict__ C,
    const float* __restrict__ scales_a,
    const float* __restrict__ scales_b,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k
) {
    int row = blockIdx.y * tile_m + threadIdx.x;
    int col = blockIdx.x * tile_n + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    int k_packed = (K + 1) / 2;
    for (int kb = 0; kb < k_packed; ++kb) {
        unsigned char a_byte = A_packed[row * k_packed + kb];
        unsigned char b_byte = B_packed[col * k_packed + kb];
        int a_lo = ((int)(a_byte & 0xF)) - 8;
        int b_lo = ((int)(b_byte & 0xF)) - 8;
        acc += (float)(a_lo * b_lo);
        if (kb * 2 + 1 < K) {
            int a_hi = ((int)((a_byte >> 4) & 0xF)) - 8;
            int b_hi = ((int)((b_byte >> 4) & 0xF)) - 8;
            acc += (float)(a_hi * b_hi);
        }
    }
    C[row * N + col] = acc * scales_a[row] * scales_b[col];
}

/// Mixed-precision GEMM: A (INT2 packed weights) × B (FP16 activations) → C (FP32).
extern "C" __global__
void quantized_gemm_mixed_kernel(
    const unsigned char* __restrict__ W_packed,
    const __half* __restrict__ X,
    float* __restrict__ C,
    const float* __restrict__ scales_w,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    int col = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    int k_packed = (K + 3) / 4;
    for (int kb = 0; kb < k_packed; ++kb) {
        unsigned char w_byte = W_packed[row * k_packed + kb];
        for (int sub = 0; sub < 4 && (kb * 4 + sub) < K; ++sub) {
            int w_val = ((int)((w_byte >> (sub * 2)) & 0x3)) - 1;
            float x_val = __half2float(X[col * K + kb * 4 + sub]);
            acc += (float)w_val * x_val;
        }
    }
    C[row * N + col] = acc * scales_w[row];
}
"#;

// ── INT2 GEMM (CPU fallback) ─────────────────────────────────────────

/// Decode a 2-bit I2 code to its signed integer value (-1, 0, +1).
#[inline(always)]
fn decode_i2(bits: u8) -> i8 {
    match bits & 0x03 {
        0b00 => 0,
        0b01 => 1,
        0b11 => -1,
        _ => 0, // 0b10 is unused
    }
}

/// CPU reference: INT2 quantized GEMM.
///
/// `a_packed` contains the INT2-packed rows of A (M rows, each
/// `ceil(K/4)` bytes).  `b_packed` contains the INT2-packed columns of
/// B transposed (N rows, each `ceil(K/4)` bytes).  `scales_a` has M
/// entries; `scales_b` has N entries.
///
/// Output is written into `output` which must have at least `M × N`
/// elements.
///
/// # Errors
///
/// Returns an error on dimension or buffer-size mismatch.
pub fn quantized_gemm_i2(
    a_packed: &[u8],
    b_packed: &[u8],
    scales_a: &[f32],
    scales_b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(m, k, k, n, output.len())?;

    let k_packed = k.div_ceil(4);
    let expected_a = m * k_packed;
    let expected_b = n * k_packed;
    if a_packed.len() < expected_a {
        return Err(KernelError::InvalidArguments {
            reason: format!("a_packed too small: need {expected_a}, got {}", a_packed.len()),
        }
        .into());
    }
    if b_packed.len() < expected_b {
        return Err(KernelError::InvalidArguments {
            reason: format!("b_packed too small: need {expected_b}, got {}", b_packed.len()),
        }
        .into());
    }
    if scales_a.len() < m {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales_a too small: need {m}, got {}", scales_a.len()),
        }
        .into());
    }
    if scales_b.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales_b too small: need {n}, got {}", scales_b.len()),
        }
        .into());
    }

    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for kb in 0..k_packed {
                let a_byte = a_packed[row * k_packed + kb];
                let b_byte = b_packed[col * k_packed + kb];
                for sub in 0..4 {
                    if kb * 4 + sub >= k {
                        break;
                    }
                    let a_val = decode_i2((a_byte >> (sub * 2)) & 0x03) as i32;
                    let b_val = decode_i2((b_byte >> (sub * 2)) & 0x03) as i32;
                    acc += a_val * b_val;
                }
            }
            output[row * n + col] = acc as f32 * scales_a[row] * scales_b[col];
        }
    }
    Ok(())
}

// ── INT4 GEMM (CPU fallback) ─────────────────────────────────────────

/// Decode a 4-bit signed value (stored as unsigned 0–15, biased by 8).
#[inline(always)]
fn decode_i4(nibble: u8) -> i8 {
    (nibble & 0x0F) as i8 - 8
}

/// CPU reference: INT4 quantized GEMM.
///
/// `a_packed` contains the INT4-packed rows of A (M rows, each
/// `ceil(K/2)` bytes).  `b_packed` is laid out the same way for B
/// transposed.  Scales have M and N entries respectively.
///
/// # Errors
///
/// Returns an error on dimension or buffer-size mismatch.
pub fn quantized_gemm_i4(
    a_packed: &[u8],
    b_packed: &[u8],
    scales_a: &[f32],
    scales_b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(m, k, k, n, output.len())?;

    let k_packed = k.div_ceil(2);
    let expected_a = m * k_packed;
    let expected_b = n * k_packed;
    if a_packed.len() < expected_a {
        return Err(KernelError::InvalidArguments {
            reason: format!("a_packed too small: need {expected_a}, got {}", a_packed.len()),
        }
        .into());
    }
    if b_packed.len() < expected_b {
        return Err(KernelError::InvalidArguments {
            reason: format!("b_packed too small: need {expected_b}, got {}", b_packed.len()),
        }
        .into());
    }
    if scales_a.len() < m {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales_a too small: need {m}, got {}", scales_a.len()),
        }
        .into());
    }
    if scales_b.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales_b too small: need {n}, got {}", scales_b.len()),
        }
        .into());
    }

    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for kb in 0..k_packed {
                let a_byte = a_packed[row * k_packed + kb];
                let b_byte = b_packed[col * k_packed + kb];
                let a_lo = decode_i4(a_byte & 0x0F) as i32;
                let b_lo = decode_i4(b_byte & 0x0F) as i32;
                acc += a_lo * b_lo;
                if kb * 2 + 1 < k {
                    let a_hi = decode_i4((a_byte >> 4) & 0x0F) as i32;
                    let b_hi = decode_i4((b_byte >> 4) & 0x0F) as i32;
                    acc += a_hi * b_hi;
                }
            }
            output[row * n + col] = acc as f32 * scales_a[row] * scales_b[col];
        }
    }
    Ok(())
}

// ── Mixed-precision GEMM (CPU fallback) ───────────────────────────────

/// CPU reference: mixed-precision GEMM (INT2 weights × FP16 activations).
///
/// `w_packed` is the INT2-packed weight matrix (M rows, each `ceil(K/4)`
/// bytes).  `x` is the FP16 activation matrix stored as `f32` values
/// (N × K row-major).  `scales_w` has M entries.
///
/// # Errors
///
/// Returns an error on dimension or buffer-size mismatch.
pub fn quantized_gemm_mixed(
    w_packed: &[u8],
    x: &[f32],
    scales_w: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(m, k, k, n, output.len())?;

    let k_packed = k.div_ceil(4);
    let expected_w = m * k_packed;
    if w_packed.len() < expected_w {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_packed too small: need {expected_w}, got {}", w_packed.len()),
        }
        .into());
    }
    if x.len() < n * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("x too small: need {}, got {}", n * k, x.len()),
        }
        .into());
    }
    if scales_w.len() < m {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales_w too small: need {m}, got {}", scales_w.len()),
        }
        .into());
    }

    for row in 0..m {
        for col in 0..n {
            let mut acc: f32 = 0.0;
            for kb in 0..k_packed {
                let w_byte = w_packed[row * k_packed + kb];
                for sub in 0..4 {
                    let idx = kb * 4 + sub;
                    if idx >= k {
                        break;
                    }
                    let w_val = decode_i2((w_byte >> (sub * 2)) & 0x03) as f32;
                    let x_val = x[col * k + idx];
                    acc += w_val * x_val;
                }
            }
            output[row * n + col] = acc * scales_w[row];
        }
    }
    Ok(())
}

// ── Dequantize-on-the-fly GEMM (CPU fallback) ─────────────────────────

/// CPU reference: dequantize-on-the-fly GEMM.
///
/// Dequantises INT2 packed weights to FP32 on the fly and multiplies
/// against FP32 activations, avoiding a separate dequantisation pass.
///
/// `w_packed` is INT2-packed (M × ceil(K/4) bytes), `x` is FP32
/// (N × K row-major), `scales` has M entries.
///
/// # Errors
///
/// Returns an error on dimension or buffer-size mismatch.
pub fn quantized_dequant_gemm(
    w_packed: &[u8],
    x: &[f32],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(m, k, k, n, output.len())?;

    let k_packed = k.div_ceil(4);
    let expected_w = m * k_packed;
    if w_packed.len() < expected_w {
        return Err(KernelError::InvalidArguments {
            reason: format!("w_packed too small: need {expected_w}, got {}", w_packed.len()),
        }
        .into());
    }
    if x.len() < n * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("x too small: need {}, got {}", n * k, x.len()),
        }
        .into());
    }
    if scales.len() < m {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales too small: need {m}, got {}", scales.len()),
        }
        .into());
    }

    for row in 0..m {
        let scale = scales[row];
        for col in 0..n {
            let mut acc: f32 = 0.0;
            for kb in 0..k_packed {
                let w_byte = w_packed[row * k_packed + kb];
                for sub in 0..4 {
                    let idx = kb * 4 + sub;
                    if idx >= k {
                        break;
                    }
                    let w_val = decode_i2((w_byte >> (sub * 2)) & 0x03) as f32 * scale;
                    let x_val = x[col * k + idx];
                    acc += w_val * x_val;
                }
            }
            output[row * n + col] = acc;
        }
    }
    Ok(())
}

// ── GPU launch stubs ──────────────────────────────────────────────────

/// Launch the INT2 quantized GEMM kernel on the GPU.
///
/// # Errors
///
/// Returns an error if the GPU is not available or launch fails.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_gemm_i2(
    config: &QuantizedGemmConfig,
    a_packed: &[u8],
    b_packed: &[u8],
    scales_a: &[f32],
    scales_b: &[f32],
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(config.m, config.k, config.k, config.n, output.len())?;
    quantized_gemm_i2(a_packed, b_packed, scales_a, scales_b, config.m, config.n, config.k, output)
}

/// Launch the INT4 quantized GEMM kernel on the GPU.
///
/// # Errors
///
/// Returns an error if the GPU is not available or launch fails.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_gemm_i4(
    config: &QuantizedGemmConfig,
    a_packed: &[u8],
    b_packed: &[u8],
    scales_a: &[f32],
    scales_b: &[f32],
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(config.m, config.k, config.k, config.n, output.len())?;
    quantized_gemm_i4(a_packed, b_packed, scales_a, scales_b, config.m, config.n, config.k, output)
}

/// Launch the mixed-precision GEMM kernel on the GPU.
///
/// # Errors
///
/// Returns an error if the GPU is not available or launch fails.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_gemm_mixed(
    config: &QuantizedGemmConfig,
    w_packed: &[u8],
    x: &[f32],
    scales_w: &[f32],
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(config.m, config.k, config.k, config.n, output.len())?;
    quantized_gemm_mixed(w_packed, x, scales_w, config.m, config.n, config.k, output)
}

/// Launch the dequantize-on-the-fly GEMM kernel on the GPU.
///
/// # Errors
///
/// Returns an error if the GPU is not available or launch fails.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_dequant_gemm(
    config: &QuantizedGemmConfig,
    w_packed: &[u8],
    x: &[f32],
    scales: &[f32],
    output: &mut [f32],
) -> Result<()> {
    validate_gemm_inputs(config.m, config.k, config.k, config.n, output.len())?;
    quantized_dequant_gemm(w_packed, x, scales, config.m, config.n, config.k, output)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────

    /// Pack a flat matrix of signed values into INT2 bytes, row by row.
    /// Each row of `k` values is independently padded to `ceil(k/4)` bytes.
    fn pack_i2_matrix(vals: &[i8], rows: usize, k: usize) -> Vec<u8> {
        assert_eq!(vals.len(), rows * k);
        let k_packed = k.div_ceil(4);
        let mut out = Vec::with_capacity(rows * k_packed);
        for row in 0..rows {
            let row_vals = &vals[row * k..(row + 1) * k];
            let mut padded = row_vals.to_vec();
            padded.resize(k_packed * 4, 0);
            for chunk in padded.chunks(4) {
                let mut byte = 0u8;
                for (i, &v) in chunk.iter().enumerate() {
                    let code: u8 = match v {
                        1 => 0b01,
                        -1 => 0b11,
                        _ => 0b00,
                    };
                    byte |= code << (i * 2);
                }
                out.push(byte);
            }
        }
        out
    }

    /// Pack a single row of signed values into INT2 bytes (4 values per byte).
    fn pack_i2_values(vals: &[i8]) -> Vec<u8> {
        pack_i2_matrix(vals, 1, vals.len())
    }

    /// Pack a flat matrix of signed values into INT4 bytes, row by row.
    /// Each row of `k` values is independently padded to `ceil(k/2)` bytes.
    fn pack_i4_matrix(vals: &[i8], rows: usize, k: usize) -> Vec<u8> {
        assert_eq!(vals.len(), rows * k);
        let k_packed = k.div_ceil(2);
        let mut out = Vec::with_capacity(rows * k_packed);
        for row in 0..rows {
            let row_vals = &vals[row * k..(row + 1) * k];
            let mut padded = row_vals.to_vec();
            padded.resize(k_packed * 2, 0);
            for chunk in padded.chunks(2) {
                let lo = ((chunk[0] + 8) as u8) & 0x0F;
                let hi = ((chunk[1] + 8) as u8) & 0x0F;
                out.push(lo | (hi << 4));
            }
        }
        out
    }

    /// Pack a single row of signed values into INT4 bytes (2 values per byte).
    fn pack_i4_values(vals: &[i8]) -> Vec<u8> {
        pack_i4_matrix(vals, 1, vals.len())
    }

    // ── validate_gemm_inputs tests ────────────────────────────────────

    #[test]
    fn test_validate_inputs_ok() {
        assert!(validate_gemm_inputs(4, 8, 8, 4, 16).is_ok());
    }

    #[test]
    fn test_validate_inputs_zero_m() {
        let err = validate_gemm_inputs(0, 8, 8, 4, 16).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::ZeroDimension { dim_name: "m" }));
    }

    #[test]
    fn test_validate_inputs_zero_k_a() {
        let err = validate_gemm_inputs(4, 0, 8, 4, 16).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::ZeroDimension { .. }));
    }

    #[test]
    fn test_validate_inputs_zero_k_b() {
        let err = validate_gemm_inputs(4, 8, 0, 4, 16).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::ZeroDimension { .. }));
    }

    #[test]
    fn test_validate_inputs_zero_n() {
        let err = validate_gemm_inputs(4, 8, 8, 0, 16).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::ZeroDimension { dim_name: "n" }));
    }

    #[test]
    fn test_validate_inputs_shape_mismatch() {
        let err = validate_gemm_inputs(4, 8, 16, 4, 16).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::ShapeMismatch { k_a: 8, k_b: 16, .. }));
    }

    #[test]
    fn test_validate_inputs_buffer_too_small() {
        let err = validate_gemm_inputs(4, 8, 8, 4, 8).unwrap_err();
        assert!(matches!(err, QuantizedGemmError::BufferTooSmall { required: 16, actual: 8 }));
    }

    #[test]
    fn test_validate_inputs_exact_buffer() {
        assert!(validate_gemm_inputs(2, 3, 3, 5, 10).is_ok());
    }

    #[test]
    fn test_validate_inputs_oversized_buffer() {
        assert!(validate_gemm_inputs(2, 3, 3, 5, 100).is_ok());
    }

    // ── check_alignment tests ─────────────────────────────────────────

    #[test]
    fn test_alignment_ok() {
        assert!(check_alignment(32, 16, "k").is_ok());
    }

    #[test]
    fn test_alignment_fail() {
        let err = check_alignment(33, 16, "k").unwrap_err();
        assert!(matches!(err, QuantizedGemmError::AlignmentError { required: 16, actual: 33, .. }));
    }

    #[test]
    fn test_alignment_zero_alignment() {
        let err = check_alignment(32, 0, "k").unwrap_err();
        assert!(matches!(err, QuantizedGemmError::InvalidTileConfig { .. }));
    }

    #[test]
    fn test_alignment_one() {
        assert!(check_alignment(7, 1, "k").is_ok());
    }

    // ── TileStrategy tests ────────────────────────────────────────────

    #[test]
    fn test_tile_strategy_small_dims() {
        let s = select_tile_strategy(32, 32, 64);
        assert_eq!(s, TileStrategy::Small);
    }

    #[test]
    fn test_tile_strategy_medium_dims() {
        let s = select_tile_strategy(256, 256, 512);
        assert_eq!(s, TileStrategy::Medium);
    }

    #[test]
    fn test_tile_strategy_large_dims() {
        let s = select_tile_strategy(1024, 1024, 2048);
        assert_eq!(s, TileStrategy::Large);
    }

    #[test]
    fn test_tile_strategy_boundary_small_medium() {
        assert_eq!(select_tile_strategy(64, 64, 257), TileStrategy::Medium);
    }

    #[test]
    fn test_tile_strategy_boundary_medium_large() {
        assert_eq!(select_tile_strategy(513, 128, 256), TileStrategy::Large);
    }

    #[test]
    fn test_tile_strategy_auto_resolves() {
        let resolved = TileStrategy::Auto.resolve(32, 32, 64);
        assert_ne!(resolved, TileStrategy::Auto);
    }

    #[test]
    fn test_tile_strategy_explicit_no_resolve() {
        assert_eq!(TileStrategy::Small.resolve(10000, 10000, 10000), TileStrategy::Small);
    }

    #[test]
    fn test_tile_dims_small() {
        assert_eq!(TileStrategy::Small.tile_dims(), (16, 16, 16));
    }

    #[test]
    fn test_tile_dims_medium() {
        assert_eq!(TileStrategy::Medium.tile_dims(), (32, 32, 32));
    }

    #[test]
    fn test_tile_dims_large() {
        assert_eq!(TileStrategy::Large.tile_dims(), (64, 64, 32));
    }

    #[test]
    fn test_tile_shared_mem_small() {
        // (16*16 + 16*16) * 4 = 2048
        assert_eq!(TileStrategy::Small.shared_mem_bytes(), 2048);
    }

    #[test]
    fn test_tile_shared_mem_medium() {
        // (32*32 + 32*32) * 4 = 8192
        assert_eq!(TileStrategy::Medium.shared_mem_bytes(), 8192);
    }

    #[test]
    fn test_tile_shared_mem_large() {
        // (64*32 + 32*64) * 4 = 16384
        assert_eq!(TileStrategy::Large.shared_mem_bytes(), 16384);
    }

    #[test]
    fn test_threads_per_block_all() {
        for s in [TileStrategy::Small, TileStrategy::Medium, TileStrategy::Large] {
            assert_eq!(s.threads_per_block(), 256);
        }
    }

    // ── QuantizedGemmConfig tests ─────────────────────────────────────

    #[test]
    fn test_config_new_ok() {
        let cfg = QuantizedGemmConfig::new(128, 256, 512).unwrap();
        assert_eq!(cfg.m, 128);
        assert_eq!(cfg.n, 256);
        assert_eq!(cfg.k, 512);
        assert_eq!(cfg.accumulator_type, AccumulatorType::F32);
        assert!(!cfg.use_tensor_cores);
    }

    #[test]
    fn test_config_new_zero_m() {
        assert!(QuantizedGemmConfig::new(0, 256, 512).is_err());
    }

    #[test]
    fn test_config_new_zero_n() {
        assert!(QuantizedGemmConfig::new(128, 0, 512).is_err());
    }

    #[test]
    fn test_config_new_zero_k() {
        assert!(QuantizedGemmConfig::new(128, 256, 0).is_err());
    }

    #[test]
    fn test_config_with_strategy_small() {
        let cfg = QuantizedGemmConfig::with_strategy(64, 64, 128, TileStrategy::Small).unwrap();
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 16);
    }

    #[test]
    fn test_config_with_strategy_large() {
        let cfg =
            QuantizedGemmConfig::with_strategy(1024, 1024, 2048, TileStrategy::Large).unwrap();
        assert_eq!(cfg.tile_m, 64);
        assert_eq!(cfg.tile_n, 64);
        assert_eq!(cfg.tile_k, 32);
    }

    #[test]
    fn test_config_with_tensor_cores() {
        let cfg = QuantizedGemmConfig::new(32, 32, 32).unwrap().with_tensor_cores(true);
        assert!(cfg.use_tensor_cores);
    }

    #[test]
    fn test_config_with_accumulator() {
        let cfg =
            QuantizedGemmConfig::new(32, 32, 32).unwrap().with_accumulator(AccumulatorType::F16);
        assert_eq!(cfg.accumulator_type, AccumulatorType::F16);
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = QuantizedGemmConfig::with_strategy(64, 128, 256, TileStrategy::Medium).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 4); // 128 / 32
        assert_eq!(gy, 2); // 64 / 32
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_grid_dim_non_aligned() {
        let cfg = QuantizedGemmConfig::with_strategy(33, 65, 128, TileStrategy::Medium).unwrap();
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 3); // ceil(65/32)
        assert_eq!(gy, 2); // ceil(33/32)
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = QuantizedGemmConfig::new(32, 32, 32).unwrap();
        let (bx, by, bz) = cfg.block_dim();
        assert_eq!(bx, 256);
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_config_total_flops() {
        let cfg = QuantizedGemmConfig::new(4, 8, 16).unwrap();
        assert_eq!(cfg.total_flops(), 2 * 4 * 8 * 16);
    }

    // ── AccumulatorType display ───────────────────────────────────────

    #[test]
    fn test_accumulator_display() {
        assert_eq!(AccumulatorType::F32.to_string(), "f32");
        assert_eq!(AccumulatorType::F16.to_string(), "f16");
    }

    // ── QuantizedGemmError display ────────────────────────────────────

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = QuantizedGemmError::ShapeMismatch { m: 4, k_a: 8, k_b: 16, n: 4 };
        let s = e.to_string();
        assert!(s.contains("shape mismatch"));
        assert!(s.contains("8"));
        assert!(s.contains("16"));
    }

    #[test]
    fn test_error_display_zero_dim() {
        let e = QuantizedGemmError::ZeroDimension { dim_name: "m" };
        assert!(e.to_string().contains("'m'"));
    }

    #[test]
    fn test_error_display_alignment() {
        let e = QuantizedGemmError::AlignmentError { required: 16, actual: 33, dim_name: "k" };
        let s = e.to_string();
        assert!(s.contains("16") && s.contains("33"));
    }

    #[test]
    fn test_error_display_tile_config() {
        let e = QuantizedGemmError::InvalidTileConfig { reason: "bad tile".into() };
        assert!(e.to_string().contains("bad tile"));
    }

    #[test]
    fn test_error_display_buffer_small() {
        let e = QuantizedGemmError::BufferTooSmall { required: 100, actual: 50 };
        let s = e.to_string();
        assert!(s.contains("100") && s.contains("50"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(QuantizedGemmError::ZeroDimension { dim_name: "k" });
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_into_kernel_error() {
        let e = QuantizedGemmError::ZeroDimension { dim_name: "k" };
        let ke: KernelError = e.into();
        assert!(matches!(ke, KernelError::InvalidArguments { .. }));
    }

    // ── INT2 GEMM correctness ─────────────────────────────────────────

    #[test]
    fn test_i2_gemm_identity_like() {
        // A: 2 rows × k=2, B^T: 2 rows × k=2
        let a = pack_i2_matrix(&[1, 0, 0, 1], 2, 2);
        let b = pack_i2_matrix(&[1, 0, 0, 1], 2, 2);
        let scales_a = vec![1.0, 1.0];
        let scales_b = vec![1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, 2, 2, 2, &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_i2_gemm_all_ones() {
        let m = 2;
        let n = 3;
        let k = 4;
        let a = pack_i2_matrix(&vec![1i8; m * k], m, k);
        let b = pack_i2_matrix(&vec![1i8; n * k], n, k);
        let scales_a = vec![1.0; m];
        let scales_b = vec![1.0; n];
        let mut out = vec![0.0f32; m * n];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, m, n, k, &mut out).unwrap();
        for &v in &out {
            assert_eq!(v, k as f32);
        }
    }

    #[test]
    fn test_i2_gemm_with_scales() {
        let a = pack_i2_values(&[1, 1]);
        let b = pack_i2_values(&[1, 1]);
        let scales_a = vec![2.0];
        let scales_b = vec![3.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 2.0 * 2.0 * 3.0);
    }

    #[test]
    fn test_i2_gemm_negative_values() {
        let a = pack_i2_values(&[-1, 1]);
        let b = pack_i2_values(&[1, -1]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], -2.0);
    }

    #[test]
    fn test_i2_gemm_zero_values() {
        let a = pack_i2_values(&[0, 0, 0, 0]);
        let b = pack_i2_values(&[1, 1, 1, 1]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, 1, 1, 4, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn test_i2_gemm_odd_k() {
        let a = pack_i2_values(&[1, 1, 1]);
        let b = pack_i2_values(&[1, 1, 1]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, 1, 1, 3, &mut out).unwrap();
        assert_eq!(out[0], 3.0);
    }

    #[test]
    fn test_i2_gemm_buffer_too_small() {
        let a = pack_i2_matrix(&[1, 1, 1, 1], 2, 2);
        let b = pack_i2_matrix(&[1, 1, 1, 1], 2, 2);
        let mut out = vec![0.0f32; 1]; // need 4
        assert!(quantized_gemm_i2(&a, &b, &[1.0, 1.0], &[1.0, 1.0], 2, 2, 2, &mut out).is_err());
    }

    #[test]
    fn test_i2_gemm_a_packed_too_small() {
        let a = vec![0u8; 1]; // too small for 2×4
        let b = pack_i2_matrix(&[1, 1, 1, 1, 1, 1, 1, 1], 2, 4);
        let mut out = vec![0.0f32; 4];
        assert!(quantized_gemm_i2(&a, &b, &[1.0, 1.0], &[1.0, 1.0], 2, 2, 4, &mut out).is_err());
    }

    // ── INT4 GEMM correctness ─────────────────────────────────────────

    #[test]
    fn test_i4_gemm_simple() {
        let a = pack_i4_values(&[1, 2]);
        let b = pack_i4_values(&[3, 4]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i4(&a, &b, &scales_a, &scales_b, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 11.0);
    }

    #[test]
    fn test_i4_gemm_with_scales() {
        let a = pack_i4_values(&[1, 1]);
        let b = pack_i4_values(&[1, 1]);
        let scales_a = vec![2.0];
        let scales_b = vec![3.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i4(&a, &b, &scales_a, &scales_b, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 12.0);
    }

    #[test]
    fn test_i4_gemm_negative_values() {
        let a = pack_i4_values(&[-3, 2]);
        let b = pack_i4_values(&[4, -1]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i4(&a, &b, &scales_a, &scales_b, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], -14.0);
    }

    #[test]
    fn test_i4_gemm_odd_k() {
        let a = pack_i4_values(&[1, 2, 3]);
        let b = pack_i4_values(&[4, 5, 6]);
        let scales_a = vec![1.0];
        let scales_b = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i4(&a, &b, &scales_a, &scales_b, 1, 1, 3, &mut out).unwrap();
        assert_eq!(out[0], 32.0);
    }

    #[test]
    fn test_i4_gemm_2x2() {
        let a = pack_i4_matrix(&[1, 0, 0, 1], 2, 2);
        let b = pack_i4_matrix(&[1, 0, 0, 1], 2, 2);
        let scales_a = vec![1.0, 1.0];
        let scales_b = vec![1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        quantized_gemm_i4(&a, &b, &scales_a, &scales_b, 2, 2, 2, &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0]);
    }

    // ── Mixed-precision GEMM correctness ──────────────────────────────

    #[test]
    fn test_mixed_gemm_simple() {
        let w = pack_i2_values(&[1, -1]);
        let x = vec![2.0f32, 3.0];
        let scales_w = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_mixed(&w, &x, &scales_w, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn test_mixed_gemm_with_scale() {
        let w = pack_i2_values(&[1, 1]);
        let x = vec![1.0f32, 1.0];
        let scales_w = vec![0.5];
        let mut out = vec![0.0f32; 1];
        quantized_gemm_mixed(&w, &x, &scales_w, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_mixed_gemm_multi_row() {
        let w = pack_i2_matrix(&[1, 0, 0, 1], 2, 2);
        let x = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ];
        let scales_w = vec![1.0, 1.0];
        let mut out = vec![0.0f32; 6];
        quantized_gemm_mixed(&w, &x, &scales_w, 2, 3, 2, &mut out).unwrap();
        assert_eq!(out, [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mixed_gemm_buffer_errors() {
        let w = pack_i2_values(&[1]);
        let x = vec![1.0f32];
        let mut out = vec![0.0f32; 0];
        assert!(quantized_gemm_mixed(&w, &x, &[1.0], 1, 1, 1, &mut out).is_err());
    }

    // ── Dequant GEMM correctness ──────────────────────────────────────

    #[test]
    fn test_dequant_gemm_simple() {
        let w = pack_i2_values(&[1, -1]);
        let x = vec![3.0f32, 4.0];
        let scales = vec![2.0];
        let mut out = vec![0.0f32; 1];
        quantized_dequant_gemm(&w, &x, &scales, 1, 1, 2, &mut out).unwrap();
        assert_eq!(out[0], -2.0);
    }

    #[test]
    fn test_dequant_gemm_all_zeros() {
        let w = pack_i2_values(&[0, 0, 0, 0]);
        let x = vec![1.0f32; 4];
        let scales = vec![1.0];
        let mut out = vec![0.0f32; 1];
        quantized_dequant_gemm(&w, &x, &scales, 1, 1, 4, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn test_dequant_gemm_2x2() {
        let w = pack_i2_matrix(&[1, 1, -1, -1], 2, 2);
        let x = vec![
            1.0, 2.0, // col 0
            3.0, 4.0, // col 1
        ];
        let scales = vec![1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        quantized_dequant_gemm(&w, &x, &scales, 2, 2, 2, &mut out).unwrap();
        assert_eq!(out, [3.0, 7.0, -3.0, -7.0]);
    }

    #[test]
    fn test_dequant_gemm_w_packed_too_small() {
        let w = vec![0u8; 1];
        let x = vec![1.0f32; 8];
        let scales = vec![1.0, 1.0];
        let mut out = vec![0.0f32; 4];
        assert!(quantized_dequant_gemm(&w, &x, &scales, 2, 2, 4, &mut out).is_err());
    }

    // ── GemmPerformanceMetrics tests ──────────────────────────────────

    #[test]
    fn test_perf_metrics_basic() {
        let m = QuantizedGemmConfig::new(128, 128, 128).unwrap();
        assert_eq!(m.total_flops(), 2 * 128 * 128 * 128);
    }

    #[test]
    fn test_perf_metrics_compute() {
        let metrics = GemmPerformanceMetrics::compute(128, 128, 128, 0.001, 900.0).unwrap();
        assert!(metrics.gflops > 0.0);
        assert!(metrics.bandwidth_utilization > 0.0);
        assert!(metrics.bandwidth_utilization <= 1.0);
        assert!(metrics.occupancy > 0.0);
        assert!(metrics.occupancy <= 1.0);
    }

    #[test]
    fn test_perf_metrics_zero_time() {
        assert!(GemmPerformanceMetrics::compute(128, 128, 128, 0.0, 900.0).is_none());
    }

    #[test]
    fn test_perf_metrics_negative_time() {
        assert!(GemmPerformanceMetrics::compute(128, 128, 128, -1.0, 900.0).is_none());
    }

    #[test]
    fn test_perf_metrics_zero_bandwidth() {
        assert!(GemmPerformanceMetrics::compute(128, 128, 128, 0.001, 0.0).is_none());
    }

    #[test]
    fn test_perf_metrics_utilization_capped() {
        let metrics = GemmPerformanceMetrics::compute(128, 128, 128, 1e-12, 1.0).unwrap();
        assert!(metrics.bandwidth_utilization <= 1.0);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_single_element_i2() {
        let a = pack_i2_values(&[1]);
        let b = pack_i2_values(&[1]);
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &[1.0], &[1.0], 1, 1, 1, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_single_element_i4() {
        let a = pack_i4_values(&[3]);
        let b = pack_i4_values(&[4]);
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i4(&a, &b, &[1.0], &[1.0], 1, 1, 1, &mut out).unwrap();
        assert_eq!(out[0], 12.0);
    }

    #[test]
    fn test_large_k_i2() {
        let k = 256;
        let a = pack_i2_values(&vec![1i8; k]);
        let b = pack_i2_values(&vec![1i8; k]);
        let mut out = vec![0.0f32; 1];
        quantized_gemm_i2(&a, &b, &[1.0], &[1.0], 1, 1, k, &mut out).unwrap();
        assert_eq!(out[0], k as f32);
    }

    #[test]
    fn test_rectangular_matrix_i2() {
        let m = 3;
        let n = 2;
        let k = 5;
        let a = pack_i2_matrix(&vec![1i8; m * k], m, k);
        let b = pack_i2_matrix(&vec![1i8; n * k], n, k);
        let scales_a = vec![1.0; m];
        let scales_b = vec![1.0; n];
        let mut out = vec![0.0f32; m * n];
        quantized_gemm_i2(&a, &b, &scales_a, &scales_b, m, n, k, &mut out).unwrap();
        for &v in &out {
            assert_eq!(v, k as f32);
        }
    }

    #[test]
    fn test_decode_i2_values() {
        assert_eq!(decode_i2(0b00), 0);
        assert_eq!(decode_i2(0b01), 1);
        assert_eq!(decode_i2(0b11), -1);
        assert_eq!(decode_i2(0b10), 0);
    }

    #[test]
    fn test_decode_i4_values() {
        assert_eq!(decode_i4(8), 0);
        assert_eq!(decode_i4(15), 7);
        assert_eq!(decode_i4(0), -8);
        assert_eq!(decode_i4(1), -7);
    }

    // ── Property tests ────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn tile_strategy_is_never_auto(
                m in 1usize..2048,
                n in 1usize..2048,
                k in 1usize..2048,
            ) {
                let s = select_tile_strategy(m, n, k);
                prop_assert_ne!(s, TileStrategy::Auto);
            }

            #[test]
            fn auto_resolve_matches_select(
                m in 1usize..2048,
                n in 1usize..2048,
                k in 1usize..2048,
            ) {
                let resolved = TileStrategy::Auto.resolve(m, n, k);
                let selected = select_tile_strategy(m, n, k);
                prop_assert_eq!(resolved, selected);
            }

            #[test]
            fn config_grid_covers_output(
                m in 1usize..1024,
                n in 1usize..1024,
                k in 1usize..1024,
            ) {
                let cfg = QuantizedGemmConfig::new(m, n, k).unwrap();
                let (gx, gy, _) = cfg.grid_dim();
                prop_assert!(gx as usize * cfg.tile_n as usize >= n);
                prop_assert!(gy as usize * cfg.tile_m as usize >= m);
            }

            #[test]
            fn config_total_flops_formula(
                m in 1usize..512,
                n in 1usize..512,
                k in 1usize..512,
            ) {
                let cfg = QuantizedGemmConfig::new(m, n, k).unwrap();
                prop_assert_eq!(cfg.total_flops(), 2 * m as u64 * n as u64 * k as u64);
            }

            #[test]
            fn validate_inputs_symmetric(
                m in 1usize..128,
                n in 1usize..128,
                k in 1usize..128,
            ) {
                let buf = m * n;
                prop_assert!(validate_gemm_inputs(m, k, k, n, buf).is_ok());
            }

            #[test]
            fn shared_mem_positive(strat in prop_oneof![
                Just(TileStrategy::Small),
                Just(TileStrategy::Medium),
                Just(TileStrategy::Large),
            ]) {
                prop_assert!(strat.shared_mem_bytes() > 0);
            }

            #[test]
            fn i2_gemm_ones_equals_k(k in 1usize..64) {
                let a = pack_i2_values(&vec![1i8; k]);
                let b = pack_i2_values(&vec![1i8; k]);
                let mut out = vec![0.0f32; 1];
                quantized_gemm_i2(&a, &b, &[1.0], &[1.0], 1, 1, k, &mut out).unwrap();
                prop_assert!((out[0] - k as f32).abs() < 1e-6);
            }

            #[test]
            fn i2_gemm_zeros_is_zero(k in 1usize..64) {
                let a = pack_i2_values(&vec![0i8; k]);
                let b = pack_i2_values(&vec![1i8; k]);
                let mut out = vec![0.0f32; 1];
                quantized_gemm_i2(&a, &b, &[1.0], &[1.0], 1, 1, k, &mut out).unwrap();
                prop_assert!((out[0]).abs() < 1e-6);
            }

            #[test]
            fn perf_metrics_gflops_positive(
                m in 1usize..256,
                n in 1usize..256,
                k in 1usize..256,
                elapsed in 0.001f64..10.0,
            ) {
                if let Some(metrics) = GemmPerformanceMetrics::compute(m, n, k, elapsed, 900.0) {
                    prop_assert!(metrics.gflops > 0.0);
                    prop_assert!(metrics.total_flops > 0);
                }
            }

            #[test]
            fn validate_mismatch_detected(
                m in 1usize..64,
                n in 1usize..64,
                k_a in 1usize..64,
                k_b in 1usize..64,
            ) {
                if k_a != k_b {
                    let result = validate_gemm_inputs(m, k_a, k_b, n, m * n);
                    prop_assert!(result.is_err());
                }
            }
        }
    }
}
