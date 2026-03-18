//! CUDA Tensor Core (WMMA) operations for accelerated matrix computations.
//!
//! # Kernel strategy
//!
//! Leverages NVIDIA Tensor Cores via WMMA (Warp Matrix Multiply-Accumulate)
//! intrinsics for mixed-precision matrix multiplication. Tensor Cores
//! operate on small matrix fragments (typically 16×16×16) at warp level,
//! delivering significantly higher throughput than scalar FMA pipelines.
//!
//! Supported configurations:
//!
//! - **FP16 × FP16 → FP32**: Mixed-precision with FP16 inputs, FP32 accumulation
//! - **FP16 × FP16 → FP16**: Pure half-precision for bandwidth-bound workloads
//! - **INT8 × INT8 → INT32**: Integer tensor core ops (Turing+ SM 7.5+)
//! - **Batched**: Multiple independent matmuls with a batch dimension
//!
//! # CPU fallback
//!
//! All operations provide pure-Rust CPU simulation for correctness testing
//! and non-GPU environments. The unified dispatch functions try the GPU
//! path first and fall back transparently.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Precision and data type enums ─────────────────────────────────────

/// Precision mode for Tensor Core operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCorePrecision {
    /// FP16 inputs, FP32 accumulation (default, best accuracy).
    Fp16Fp32,
    /// FP16 inputs, FP16 accumulation (faster, less precise).
    Fp16Fp16,
    /// INT8 inputs, INT32 accumulation (Turing+ SM ≥ 7.5).
    Int8Int32,
}

/// Accumulation data type for Tensor Core output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationType {
    /// 32-bit floating point accumulation.
    F32,
    /// 16-bit floating point accumulation.
    F16,
    /// 32-bit integer accumulation (for INT8 inputs).
    I32,
}

/// WMMA fragment layout in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentLayout {
    /// Row-major storage.
    RowMajor,
    /// Column-major storage.
    ColMajor,
}

/// Role of a WMMA fragment in the MMA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentType {
    /// Matrix A operand (left).
    MatrixA,
    /// Matrix B operand (right).
    MatrixB,
    /// Accumulator (C / D).
    Accumulator,
}

// ── Tensor Core configuration ─────────────────────────────────────────

/// Configuration for Tensor Core (WMMA) operations.
///
/// Defines the precision, fragment dimensions, and tiling strategy for
/// warp-level matrix multiply-accumulate.
#[derive(Debug, Clone)]
pub struct TensorCoreConfig {
    /// Precision mode (determines input/output types).
    pub precision: TensorCorePrecision,
    /// Accumulation type for the output fragment.
    pub accumulation_type: AccumulationType,
    /// Fragment M dimension (rows of A / rows of C).
    pub fragment_m: u32,
    /// Fragment N dimension (cols of B / cols of C).
    pub fragment_n: u32,
    /// Fragment K dimension (cols of A / rows of B).
    pub fragment_k: u32,
    /// Number of warps per thread-block in the M dimension.
    pub warps_m: u32,
    /// Number of warps per thread-block in the N dimension.
    pub warps_n: u32,
    /// Warp size (32 threads on NVIDIA GPUs).
    pub warp_size: u32,
    /// Minimum SM version required (7.0 for FP16, 7.5 for INT8).
    pub min_sm_version: u32,
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            precision: TensorCorePrecision::Fp16Fp32,
            accumulation_type: AccumulationType::F32,
            fragment_m: 16,
            fragment_n: 16,
            fragment_k: 16,
            warps_m: 4,
            warps_n: 4,
            warp_size: 32,
            min_sm_version: 70,
        }
    }
}

impl TensorCoreConfig {
    /// Create a config for FP16 input with FP32 accumulation (Volta+).
    pub fn fp16_fp32() -> Self {
        Self::default()
    }

    /// Create a config for FP16 input with FP16 accumulation (Volta+).
    pub fn fp16_fp16() -> Self {
        Self {
            precision: TensorCorePrecision::Fp16Fp16,
            accumulation_type: AccumulationType::F16,
            ..Self::default()
        }
    }

    /// Create a config for INT8 input with INT32 accumulation (Turing+).
    pub fn int8_int32() -> Self {
        Self {
            precision: TensorCorePrecision::Int8Int32,
            accumulation_type: AccumulationType::I32,
            fragment_m: 16,
            fragment_n: 16,
            fragment_k: 16,
            min_sm_version: 75,
            ..Self::default()
        }
    }

    /// Set custom fragment dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn with_fragment_size(mut self, m: u32, n: u32, k: u32) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("WMMA fragment dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        self.fragment_m = m;
        self.fragment_n = n;
        self.fragment_k = k;
        Ok(self)
    }

    /// Set the number of warps per block in M and N dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if either count is zero.
    pub fn with_warp_layout(mut self, warps_m: u32, warps_n: u32) -> Result<Self> {
        if warps_m == 0 || warps_n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "warp layout must be non-zero: warps_m={warps_m}, warps_n={warps_n}"
                ),
            }
            .into());
        }
        self.warps_m = warps_m;
        self.warps_n = warps_n;
        Ok(self)
    }

    /// Total threads per block (warps_m × warps_n × warp_size).
    pub fn threads_per_block(&self) -> u32 {
        self.warps_m * self.warps_n * self.warp_size
    }

    /// Tile size in M dimension covered by one thread-block.
    pub fn block_tile_m(&self) -> u32 {
        self.warps_m * self.fragment_m
    }

    /// Tile size in N dimension covered by one thread-block.
    pub fn block_tile_n(&self) -> u32 {
        self.warps_n * self.fragment_n
    }

    /// Bytes of shared memory for WMMA tile staging.
    pub fn shared_mem_bytes(&self) -> u32 {
        let elem_bytes: u32 = match self.precision {
            TensorCorePrecision::Fp16Fp32 | TensorCorePrecision::Fp16Fp16 => 2,
            TensorCorePrecision::Int8Int32 => 1,
        };
        let a_tile = self.block_tile_m() * self.fragment_k * elem_bytes;
        let b_tile = self.fragment_k * self.block_tile_n() * elem_bytes;
        a_tile + b_tile
    }
}

// ── WMMA Fragment ─────────────────────────────────────────────────────

/// Representation of a WMMA matrix fragment.
///
/// On GPU, this maps to `nvcuda::wmma::fragment<>`. On CPU, it holds
/// the fragment data as a flat f32 vector for simulation.
#[derive(Debug, Clone)]
pub struct WmmaFragment {
    /// Fragment role (A, B, or accumulator).
    pub fragment_type: FragmentType,
    /// Memory layout.
    pub layout: FragmentLayout,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Fragment data (f32 for simulation; actual GPU uses native types).
    pub data: Vec<f32>,
}

impl WmmaFragment {
    /// Create a new zero-initialized fragment.
    pub fn new(
        fragment_type: FragmentType,
        layout: FragmentLayout,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self { fragment_type, layout, rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Number of elements in the fragment.
    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Access element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row or col is out of bounds.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows && col < self.cols, "fragment index out of bounds");
        match self.layout {
            FragmentLayout::RowMajor => self.data[row * self.cols + col],
            FragmentLayout::ColMajor => self.data[col * self.rows + row],
        }
    }

    /// Set element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row or col is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(row < self.rows && col < self.cols, "fragment index out of bounds");
        match self.layout {
            FragmentLayout::RowMajor => self.data[row * self.cols + col] = val,
            FragmentLayout::ColMajor => self.data[col * self.rows + row] = val,
        }
    }
}

// ── WMMA operations (CPU simulation) ──────────────────────────────────

/// Load a matrix fragment from a contiguous buffer (CPU simulation).
///
/// Loads a `rows × cols` sub-matrix from `src` starting at the given
/// offset with the specified leading dimension.
///
/// # Errors
///
/// Returns an error if the source buffer is too small.
pub fn wmma_load(src: &[f32], fragment: &mut WmmaFragment, offset: usize, ld: usize) -> Result<()> {
    let rows = fragment.rows;
    let cols = fragment.cols;

    for r in 0..rows {
        let src_row_start = offset + r * ld;
        if src_row_start + cols > src.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "wmma_load: source too small (need offset {} + {} > len {})",
                    src_row_start,
                    cols,
                    src.len()
                ),
            }
            .into());
        }
        for c in 0..cols {
            fragment.set(r, c, src[src_row_start + c]);
        }
    }
    Ok(())
}

/// Store a matrix fragment to a contiguous buffer (CPU simulation).
///
/// Writes the fragment to `dst` starting at the given offset with the
/// specified leading dimension.
///
/// # Errors
///
/// Returns an error if the destination buffer is too small.
pub fn wmma_store(
    dst: &mut [f32],
    fragment: &WmmaFragment,
    offset: usize,
    ld: usize,
) -> Result<()> {
    let rows = fragment.rows;
    let cols = fragment.cols;

    for r in 0..rows {
        let dst_row_start = offset + r * ld;
        if dst_row_start + cols > dst.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "wmma_store: destination too small (need offset {} + {} > len {})",
                    dst_row_start,
                    cols,
                    dst.len()
                ),
            }
            .into());
        }
        for c in 0..cols {
            dst[dst_row_start + c] = fragment.get(r, c);
        }
    }
    Ok(())
}

/// Warp Matrix Multiply-Accumulate: D = A × B + C (CPU simulation).
///
/// Computes the matrix product of fragments A [m×k] and B [k×n], adding
/// accumulator C [m×n], writing to D [m×n].
///
/// # Errors
///
/// Returns an error if fragment dimensions are incompatible.
pub fn wmma_mma(
    a: &WmmaFragment,
    b: &WmmaFragment,
    c: &WmmaFragment,
    d: &mut WmmaFragment,
) -> Result<()> {
    if a.cols != b.rows {
        return Err(KernelError::InvalidArguments {
            reason: format!("wmma_mma: A cols ({}) != B rows ({})", a.cols, b.rows),
        }
        .into());
    }
    if a.rows != c.rows || b.cols != c.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "wmma_mma: accumulator shape [{}×{}] incompatible with A[{}×{}]·B[{}×{}]",
                c.rows, c.cols, a.rows, a.cols, b.rows, b.cols
            ),
        }
        .into());
    }
    if d.rows != a.rows || d.cols != b.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "wmma_mma: output shape [{}×{}] must be [{}×{}]",
                d.rows, d.cols, a.rows, b.cols
            ),
        }
        .into());
    }

    let m = a.rows;
    let n = b.cols;
    let k = a.cols;

    for i in 0..m {
        for j in 0..n {
            let mut acc = c.get(i, j);
            for l in 0..k {
                acc += a.get(i, l) * b.get(l, j);
            }
            d.set(i, j, acc);
        }
    }
    Ok(())
}

/// Fill a WMMA fragment with a constant value (CPU simulation).
pub fn wmma_fill(fragment: &mut WmmaFragment, value: f32) {
    fragment.data.fill(value);
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
            return f32::from_bits(sign << 31);
        }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    let f32_exp = exponent + 112;
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
        return sign << 15;
    }
    if exponent == 0xFF {
        let f16_mantissa = (mantissa >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | f16_mantissa;
    }

    let new_exp = exponent - 112;
    if new_exp >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let f16_mantissa = (mantissa >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
}

// ── Validation ────────────────────────────────────────────────────────

fn validate_matmul_buffers(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a_len: usize,
    b_len: usize,
    out_len: usize,
) -> Result<()> {
    let a_req = batch * m * k;
    let b_req = batch * k * n;
    let out_req = batch * m * n;
    if a_len < a_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A buffer too small: expected >= {a_req}, got {a_len}"),
        }));
    }
    if b_len < b_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B buffer too small: expected >= {b_req}, got {b_len}"),
        }));
    }
    if out_len < out_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("output buffer too small: expected >= {out_req}, got {out_len}"),
        }));
    }
    Ok(())
}

// ── Tensor Core matmul (CPU simulation) ───────────────────────────────

/// Full matrix multiplication using Tensor Core simulation (CPU fallback).
///
/// Tiles the computation into WMMA-sized fragments (fragment_m × fragment_n × fragment_k),
/// emulating the warp-level MMA execution model.
///
/// # Layout
/// - `a`: row-major `[m, k]` f32
/// - `b`: row-major `[k, n]` f32
/// - `out`: row-major `[m, n]` f32
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent or dimensions are zero.
pub fn tensor_core_matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }
        .into());
    }
    validate_matmul_buffers(m, n, k, 1, a.len(), b.len(), out.len())?;

    let fm = config.fragment_m as usize;
    let fn_ = config.fragment_n as usize;
    let fk = config.fragment_k as usize;

    out[..m * n].fill(0.0);

    // Tile over output in fragment_m × fragment_n blocks, streaming
    // fragment_k-wide slices through the reduction dimension.
    let mut i0 = 0;
    while i0 < m {
        let tile_m = fm.min(m - i0);
        let mut j0 = 0;
        while j0 < n {
            let tile_n = fn_.min(n - j0);

            // Accumulator fragment
            let mut acc = WmmaFragment::new(
                FragmentType::Accumulator,
                FragmentLayout::RowMajor,
                tile_m,
                tile_n,
            );

            let mut l0 = 0;
            while l0 < k {
                let tile_k = fk.min(k - l0);

                // Load A fragment [tile_m × tile_k]
                let mut frag_a = WmmaFragment::new(
                    FragmentType::MatrixA,
                    FragmentLayout::RowMajor,
                    tile_m,
                    tile_k,
                );
                for r in 0..tile_m {
                    for c in 0..tile_k {
                        frag_a.set(r, c, a[(i0 + r) * k + (l0 + c)]);
                    }
                }

                // Load B fragment [tile_k × tile_n]
                let mut frag_b = WmmaFragment::new(
                    FragmentType::MatrixB,
                    FragmentLayout::RowMajor,
                    tile_k,
                    tile_n,
                );
                for r in 0..tile_k {
                    for c in 0..tile_n {
                        frag_b.set(r, c, b[(l0 + r) * n + (j0 + c)]);
                    }
                }

                // MMA: acc = frag_a × frag_b + acc
                let prev_acc = acc.clone();
                wmma_mma(&frag_a, &frag_b, &prev_acc, &mut acc)?;

                l0 += fk;
            }

            // Store accumulator to output
            for r in 0..tile_m {
                for c in 0..tile_n {
                    out[(i0 + r) * n + (j0 + c)] = acc.get(r, c);
                }
            }
            j0 += fn_;
        }
        i0 += fm;
    }
    Ok(())
}

/// Mixed-precision Tensor Core matmul: FP16 inputs → FP32 accumulation.
///
/// Inputs `a` and `b` are packed as `u16` in IEEE 754 half-precision.
/// Accumulation and output are in f32.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent or dimensions are zero.
pub fn mixed_precision_tc_matmul(
    a: &[u16],
    b: &[u16],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }
        .into());
    }
    validate_matmul_buffers(m, n, k, 1, a.len(), b.len(), out.len())?;

    let fm = config.fragment_m as usize;
    let fn_ = config.fragment_n as usize;
    let fk = config.fragment_k as usize;

    out[..m * n].fill(0.0);

    let mut i0 = 0;
    while i0 < m {
        let tile_m = fm.min(m - i0);
        let mut j0 = 0;
        while j0 < n {
            let tile_n = fn_.min(n - j0);
            let mut acc = WmmaFragment::new(
                FragmentType::Accumulator,
                FragmentLayout::RowMajor,
                tile_m,
                tile_n,
            );

            let mut l0 = 0;
            while l0 < k {
                let tile_k = fk.min(k - l0);

                let mut frag_a = WmmaFragment::new(
                    FragmentType::MatrixA,
                    FragmentLayout::RowMajor,
                    tile_m,
                    tile_k,
                );
                for r in 0..tile_m {
                    for c in 0..tile_k {
                        frag_a.set(r, c, f16_to_f32(a[(i0 + r) * k + (l0 + c)]));
                    }
                }

                let mut frag_b = WmmaFragment::new(
                    FragmentType::MatrixB,
                    FragmentLayout::RowMajor,
                    tile_k,
                    tile_n,
                );
                for r in 0..tile_k {
                    for c in 0..tile_n {
                        frag_b.set(r, c, f16_to_f32(b[(l0 + r) * n + (j0 + c)]));
                    }
                }

                let prev_acc = acc.clone();
                wmma_mma(&frag_a, &frag_b, &prev_acc, &mut acc)?;
                l0 += fk;
            }

            for r in 0..tile_m {
                for c in 0..tile_n {
                    out[(i0 + r) * n + (j0 + c)] = acc.get(r, c);
                }
            }
            j0 += fn_;
        }
        i0 += fm;
    }
    Ok(())
}

/// INT8 Tensor Core matmul: INT8 inputs → INT32 accumulation (Turing+).
///
/// Inputs `a` and `b` are signed 8-bit integers. Accumulation and
/// output are in i32.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent or dimensions are zero.
pub fn int8_tc_matmul(
    a: &[i8],
    b: &[i8],
    out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }
        .into());
    }
    let a_req = m * k;
    let b_req = k * n;
    let out_req = m * n;
    if a.len() < a_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("A buffer too small: expected >= {a_req}, got {}", a.len()),
        }));
    }
    if b.len() < b_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("B buffer too small: expected >= {b_req}, got {}", b.len()),
        }));
    }
    if out.len() < out_req {
        return Err(BitNetError::Kernel(KernelError::InvalidArguments {
            reason: format!("output buffer too small: expected >= {out_req}, got {}", out.len()),
        }));
    }

    let fm = config.fragment_m as usize;
    let fn_ = config.fragment_n as usize;
    let fk = config.fragment_k as usize;

    out[..m * n].fill(0);

    let mut i0 = 0;
    while i0 < m {
        let tile_m = fm.min(m - i0);
        let mut j0 = 0;
        while j0 < n {
            let tile_n = fn_.min(n - j0);

            // INT32 accumulator (use Vec<i32> directly)
            let mut acc = vec![0i32; tile_m * tile_n];

            let mut l0 = 0;
            while l0 < k {
                let tile_k = fk.min(k - l0);
                for i in 0..tile_m {
                    for j in 0..tile_n {
                        let mut sum = 0i32;
                        for l in 0..tile_k {
                            sum += a[(i0 + i) * k + (l0 + l)] as i32
                                * b[(l0 + l) * n + (j0 + j)] as i32;
                        }
                        acc[i * tile_n + j] += sum;
                    }
                }
                l0 += fk;
            }

            for i in 0..tile_m {
                for j in 0..tile_n {
                    out[(i0 + i) * n + (j0 + j)] = acc[i * tile_n + j];
                }
            }
            j0 += fn_;
        }
        i0 += fm;
    }
    Ok(())
}

/// Batched Tensor Core matmul: multiple independent matmuls.
///
/// Each batch computes `C[b] = A[b] × B[b]` using Tensor Core tiling.
///
/// # Layout
/// - `a`: row-major `[batch, m, k]` f32
/// - `b`: row-major `[batch, k, n]` f32
/// - `out`: row-major `[batch, m, n]` f32
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent or dimensions are zero.
pub fn batched_tc_matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("dimensions must be non-zero: batch={batch}, m={m}, n={n}, k={k}"),
        }
        .into());
    }
    validate_matmul_buffers(m, n, k, batch, a.len(), b.len(), out.len())?;

    let a_stride = m * k;
    let b_stride = k * n;
    let out_stride = m * n;

    for bi in 0..batch {
        tensor_core_matmul(
            &a[bi * a_stride..(bi + 1) * a_stride],
            &b[bi * b_stride..(bi + 1) * b_stride],
            &mut out[bi * out_stride..(bi + 1) * out_stride],
            m,
            n,
            k,
            config,
        )?;
    }
    Ok(())
}

// ── Tensor Core Scheduler ─────────────────────────────────────────────

/// Schedule and optimize Tensor Core kernel launches.
///
/// Computes grid/block dimensions and shared memory requirements for
/// a given problem size and Tensor Core configuration.
#[derive(Debug, Clone)]
pub struct TensorCoreScheduler {
    /// Total output rows.
    pub m: usize,
    /// Total output columns.
    pub n: usize,
    /// Reduction dimension.
    pub k: usize,
    /// Batch count.
    pub batch: usize,
    /// Tensor Core config.
    pub config: TensorCoreConfig,
}

impl TensorCoreScheduler {
    /// Create a scheduler for the given problem size.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(m: usize, n: usize, k: usize, config: TensorCoreConfig) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("dimensions must be non-zero: m={m}, n={n}, k={k}"),
            }
            .into());
        }
        Ok(Self { m, n, k, batch: 1, config })
    }

    /// Set batch size.
    ///
    /// # Errors
    ///
    /// Returns an error if batch is zero.
    pub fn with_batch(mut self, batch: usize) -> Result<Self> {
        if batch == 0 {
            return Err(KernelError::InvalidArguments { reason: "batch must be > 0".into() }.into());
        }
        self.batch = batch;
        Ok(self)
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, batch)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let block_m = self.config.block_tile_m() as usize;
        let block_n = self.config.block_tile_n() as usize;
        let grid_x = self.n.div_ceil(block_n) as u32;
        let grid_y = self.m.div_ceil(block_m) as u32;
        (grid_x, grid_y, self.batch as u32)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.config.threads_per_block(), 1, 1)
    }

    /// Total number of WMMA operations for the problem.
    pub fn total_wmma_ops(&self) -> usize {
        let fm = self.config.fragment_m as usize;
        let fn_ = self.config.fragment_n as usize;
        let fk = self.config.fragment_k as usize;
        let tiles_m = self.m.div_ceil(fm);
        let tiles_n = self.n.div_ceil(fn_);
        let tiles_k = self.k.div_ceil(fk);
        self.batch * tiles_m * tiles_n * tiles_k
    }

    /// Estimated arithmetic intensity (FLOPs / byte).
    pub fn arithmetic_intensity(&self) -> f64 {
        let flops = 2.0 * self.batch as f64 * self.m as f64 * self.n as f64 * self.k as f64;
        let elem_bytes: f64 = match self.config.precision {
            TensorCorePrecision::Fp16Fp32 | TensorCorePrecision::Fp16Fp16 => 2.0,
            TensorCorePrecision::Int8Int32 => 1.0,
        };
        let bytes_a = self.batch as f64 * self.m as f64 * self.k as f64 * elem_bytes;
        let bytes_b = self.batch as f64 * self.k as f64 * self.n as f64 * elem_bytes;
        let bytes_c = self.batch as f64 * self.m as f64 * self.n as f64 * 4.0; // output always 4 bytes
        flops / (bytes_a + bytes_b + bytes_c)
    }
}

/// Calculate Tensor Core occupancy for a given problem and GPU.
///
/// Returns the estimated occupancy as a fraction [0.0, 1.0].
///
/// # Arguments
///
/// * `scheduler` — Scheduler with problem dimensions
/// * `sm_count` — Number of streaming multiprocessors on the GPU
/// * `max_blocks_per_sm` — Maximum concurrent blocks per SM
pub fn tc_occupancy(scheduler: &TensorCoreScheduler, sm_count: u32, max_blocks_per_sm: u32) -> f32 {
    if sm_count == 0 || max_blocks_per_sm == 0 {
        return 0.0;
    }
    let (grid_x, grid_y, grid_z) = scheduler.grid_dim();
    let total_blocks = grid_x as f32 * grid_y as f32 * grid_z as f32;
    let max_concurrent = sm_count as f32 * max_blocks_per_sm as f32;
    (total_blocks / max_concurrent).min(1.0)
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C kernel using WMMA intrinsics for Tensor Core matmul.
///
/// Requires SM ≥ 7.0 (Volta) for FP16, SM ≥ 7.5 (Turing) for INT8.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TENSOR_CORE_MATMUL_KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void tc_matmul_f16f32(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;

    if (warpM >= M || warpN >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < K; kk += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * K + kk, K);
        wmma::load_matrix_sync(b_frag, B + kk * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}
"#;

// ── CUDA launch stubs ─────────────────────────────────────────────────

/// Launch stub for Tensor Core FP16→FP32 matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_tc_matmul(
    _a: &[u16],
    _b: &[u16],
    _output: &mut [f32],
    scheduler: &TensorCoreScheduler,
) -> Result<()> {
    log::debug!(
        "TC matmul CUDA stub: m={}, n={}, k={}, grid={:?}",
        scheduler.m,
        scheduler.n,
        scheduler.k,
        scheduler.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Tensor Core matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for INT8 Tensor Core matmul CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_int8_tc_matmul(
    _a: &[i8],
    _b: &[i8],
    _output: &mut [i32],
    scheduler: &TensorCoreScheduler,
) -> Result<()> {
    log::debug!(
        "INT8 TC matmul CUDA stub: m={}, n={}, k={}, grid={:?}",
        scheduler.m,
        scheduler.n,
        scheduler.k,
        scheduler.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "INT8 Tensor Core matmul CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// Tensor Core FP16→FP32 matmul with automatic dispatch: GPU if
/// available, else CPU simulation.
pub fn tc_matmul_forward(
    a_f16: &[u16],
    b_f16: &[u16],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let sched = TensorCoreScheduler::new(m, n, k, config.clone())?;
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_tc_matmul(a_f16, b_f16, output, &sched)
        {
            return Ok(());
        }
    }
    mixed_precision_tc_matmul(a_f16, b_f16, output, m, n, k, config)
}

/// INT8 Tensor Core matmul with automatic dispatch: GPU if available,
/// else CPU simulation.
pub fn int8_tc_matmul_forward(
    a: &[i8],
    b: &[i8],
    output: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    config: &TensorCoreConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let sched = TensorCoreScheduler::new(m, n, k, config.clone())?;
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_int8_tc_matmul(a, b, output, &sched)
        {
            return Ok(());
        }
    }
    int8_tc_matmul(a, b, output, m, n, k, config)
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

    fn assert_close_i32(a: &[i32], b: &[i32]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at {i}: {x} vs {y}");
        }
    }

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

    fn naive_i8_matmul(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i32;
                for l in 0..k {
                    s += a[i * k + l] as i32 * b[l * n + j] as i32;
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    // ── TensorCoreConfig tests ────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = TensorCoreConfig::default();
        assert_eq!(cfg.precision, TensorCorePrecision::Fp16Fp32);
        assert_eq!(cfg.accumulation_type, AccumulationType::F32);
        assert_eq!(cfg.fragment_m, 16);
        assert_eq!(cfg.fragment_n, 16);
        assert_eq!(cfg.fragment_k, 16);
        assert_eq!(cfg.warp_size, 32);
        assert_eq!(cfg.min_sm_version, 70);
    }

    #[test]
    fn test_config_fp16_fp16() {
        let cfg = TensorCoreConfig::fp16_fp16();
        assert_eq!(cfg.precision, TensorCorePrecision::Fp16Fp16);
        assert_eq!(cfg.accumulation_type, AccumulationType::F16);
    }

    #[test]
    fn test_config_int8_int32() {
        let cfg = TensorCoreConfig::int8_int32();
        assert_eq!(cfg.precision, TensorCorePrecision::Int8Int32);
        assert_eq!(cfg.accumulation_type, AccumulationType::I32);
        assert_eq!(cfg.min_sm_version, 75);
    }

    #[test]
    fn test_config_with_fragment_size() {
        let cfg = TensorCoreConfig::default().with_fragment_size(8, 8, 4).unwrap();
        assert_eq!(cfg.fragment_m, 8);
        assert_eq!(cfg.fragment_n, 8);
        assert_eq!(cfg.fragment_k, 4);
    }

    #[test]
    fn test_config_rejects_zero_fragment() {
        assert!(TensorCoreConfig::default().with_fragment_size(0, 16, 16).is_err());
        assert!(TensorCoreConfig::default().with_fragment_size(16, 0, 16).is_err());
        assert!(TensorCoreConfig::default().with_fragment_size(16, 16, 0).is_err());
    }

    #[test]
    fn test_config_with_warp_layout() {
        let cfg = TensorCoreConfig::default().with_warp_layout(2, 8).unwrap();
        assert_eq!(cfg.warps_m, 2);
        assert_eq!(cfg.warps_n, 8);
    }

    #[test]
    fn test_config_rejects_zero_warps() {
        assert!(TensorCoreConfig::default().with_warp_layout(0, 4).is_err());
        assert!(TensorCoreConfig::default().with_warp_layout(4, 0).is_err());
    }

    #[test]
    fn test_config_threads_per_block() {
        let cfg = TensorCoreConfig::default(); // 4 × 4 warps × 32 threads
        assert_eq!(cfg.threads_per_block(), 512);
    }

    #[test]
    fn test_config_block_tile_sizes() {
        let cfg = TensorCoreConfig::default(); // warps_m=4, warps_n=4, frag=16
        assert_eq!(cfg.block_tile_m(), 64);
        assert_eq!(cfg.block_tile_n(), 64);
    }

    #[test]
    fn test_config_shared_mem_fp16() {
        let cfg = TensorCoreConfig::fp16_fp32();
        // A tile: 64 × 16 × 2 = 2048, B tile: 16 × 64 × 2 = 2048
        assert_eq!(cfg.shared_mem_bytes(), 4096);
    }

    #[test]
    fn test_config_shared_mem_int8() {
        let cfg = TensorCoreConfig::int8_int32();
        // A tile: 64 × 16 × 1 = 1024, B tile: 16 × 64 × 1 = 1024
        assert_eq!(cfg.shared_mem_bytes(), 2048);
    }

    // ── WmmaFragment tests ────────────────────────────────────────

    #[test]
    fn test_fragment_new_zero_initialized() {
        let frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 4, 4);
        assert_eq!(frag.num_elements(), 16);
        for &v in &frag.data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_fragment_row_major_access() {
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 3);
        frag.set(0, 2, 5.0);
        frag.set(1, 0, 7.0);
        assert_eq!(frag.get(0, 2), 5.0);
        assert_eq!(frag.get(1, 0), 7.0);
        assert_eq!(frag.get(0, 0), 0.0);
    }

    #[test]
    fn test_fragment_col_major_access() {
        let mut frag = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::ColMajor, 3, 2);
        frag.set(2, 1, 9.0);
        frag.set(0, 0, 3.0);
        assert_eq!(frag.get(2, 1), 9.0);
        assert_eq!(frag.get(0, 0), 3.0);
        assert_eq!(frag.get(1, 0), 0.0);
    }

    #[test]
    #[should_panic(expected = "fragment index out of bounds")]
    fn test_fragment_get_out_of_bounds() {
        let frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        frag.get(2, 0);
    }

    #[test]
    #[should_panic(expected = "fragment index out of bounds")]
    fn test_fragment_set_out_of_bounds() {
        let mut frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        frag.set(0, 2, 1.0);
    }

    // ── wmma_fill tests ───────────────────────────────────────────

    #[test]
    fn test_wmma_fill_zero() {
        let mut frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 4, 4);
        frag.data.fill(42.0);
        wmma_fill(&mut frag, 0.0);
        assert!(frag.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_wmma_fill_value() {
        let mut frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 3, 5);
        wmma_fill(&mut frag, 2.5);
        assert!(frag.data.iter().all(|&v| (v - 2.5).abs() < 1e-6));
    }

    // ── wmma_load / wmma_store tests ──────────────────────────────

    #[test]
    fn test_wmma_load_basic() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 4, 4);
        wmma_load(&src, &mut frag, 0, 4).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(frag.get(i, j), (i * 4 + j) as f32);
            }
        }
    }

    #[test]
    fn test_wmma_load_with_offset() {
        let src: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 3);
        wmma_load(&src, &mut frag, 4, 8).unwrap(); // offset=4, ld=8
        assert_eq!(frag.get(0, 0), 4.0);
        assert_eq!(frag.get(0, 2), 6.0);
        assert_eq!(frag.get(1, 0), 12.0);
    }

    #[test]
    fn test_wmma_load_buffer_too_small() {
        let src = [1.0f32; 4];
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 4, 4);
        assert!(wmma_load(&src, &mut frag, 0, 4).is_err());
    }

    #[test]
    fn test_wmma_store_basic() {
        let mut frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 3);
        for i in 0..2 {
            for j in 0..3 {
                frag.set(i, j, (i * 3 + j) as f32);
            }
        }
        let mut dst = [0.0f32; 6];
        wmma_store(&mut dst, &frag, 0, 3).unwrap();
        let expected: Vec<f32> = (0..6).map(|i| i as f32).collect();
        assert_close(&dst, &expected, 1e-6);
    }

    #[test]
    fn test_wmma_store_with_offset() {
        let mut frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        frag.set(0, 0, 1.0);
        frag.set(0, 1, 2.0);
        frag.set(1, 0, 3.0);
        frag.set(1, 1, 4.0);
        let mut dst = [0.0f32; 16];
        wmma_store(&mut dst, &frag, 5, 4).unwrap();
        assert_eq!(dst[5], 1.0);
        assert_eq!(dst[6], 2.0);
        assert_eq!(dst[9], 3.0);
        assert_eq!(dst[10], 4.0);
    }

    #[test]
    fn test_wmma_store_buffer_too_small() {
        let frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 4, 4);
        let mut dst = [0.0f32; 8];
        assert!(wmma_store(&mut dst, &frag, 0, 4).is_err());
    }

    #[test]
    fn test_wmma_load_store_roundtrip() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 4, 4);
        wmma_load(&src, &mut frag, 0, 4).unwrap();
        let mut dst = [0.0f32; 16];
        wmma_store(&mut dst, &frag, 0, 4).unwrap();
        assert_close(&dst, &src, 1e-6);
    }

    // ── wmma_mma tests ────────────────────────────────────────────

    #[test]
    fn test_wmma_mma_identity() {
        let mut a_frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 2);
        a_frag.set(0, 0, 1.0);
        a_frag.set(0, 1, 2.0);
        a_frag.set(1, 0, 3.0);
        a_frag.set(1, 1, 4.0);

        let mut b_frag = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        b_frag.set(0, 0, 1.0);
        b_frag.set(1, 1, 1.0);

        let c_frag = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        let mut d_frag =
            WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);

        wmma_mma(&a_frag, &b_frag, &c_frag, &mut d_frag).unwrap();
        assert_eq!(d_frag.get(0, 0), 1.0);
        assert_eq!(d_frag.get(0, 1), 2.0);
        assert_eq!(d_frag.get(1, 0), 3.0);
        assert_eq!(d_frag.get(1, 1), 4.0);
    }

    #[test]
    fn test_wmma_mma_known_product() {
        // A=[1,2;3,4], B=[5,6;7,8] → A·B = [19,22;43,50]
        let mut a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);

        let mut b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        b.set(0, 0, 5.0);
        b.set(0, 1, 6.0);
        b.set(1, 0, 7.0);
        b.set(1, 1, 8.0);

        let c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);

        wmma_mma(&a, &b, &c, &mut d).unwrap();
        assert_eq!(d.get(0, 0), 19.0);
        assert_eq!(d.get(0, 1), 22.0);
        assert_eq!(d.get(1, 0), 43.0);
        assert_eq!(d.get(1, 1), 50.0);
    }

    #[test]
    fn test_wmma_mma_with_accumulator() {
        let mut a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 2);
        a.set(0, 0, 1.0);
        a.set(1, 1, 1.0);

        let mut b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        b.set(0, 0, 1.0);
        b.set(1, 1, 1.0);

        let mut c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        wmma_fill(&mut c, 10.0);

        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        wmma_mma(&a, &b, &c, &mut d).unwrap();
        // D = I·I + 10 = I + 10
        assert_eq!(d.get(0, 0), 11.0);
        assert_eq!(d.get(0, 1), 10.0);
        assert_eq!(d.get(1, 0), 10.0);
        assert_eq!(d.get(1, 1), 11.0);
    }

    #[test]
    fn test_wmma_mma_dimension_mismatch_a_b() {
        let a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 3);
        let b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        let c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        assert!(wmma_mma(&a, &b, &c, &mut d).is_err());
    }

    #[test]
    fn test_wmma_mma_dimension_mismatch_acc() {
        let a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 2);
        let b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        let c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 3, 3);
        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        assert!(wmma_mma(&a, &b, &c, &mut d).is_err());
    }

    #[test]
    fn test_wmma_mma_dimension_mismatch_output() {
        let a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 2);
        let b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 2, 2);
        let c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 2);
        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 3, 2);
        assert!(wmma_mma(&a, &b, &c, &mut d).is_err());
    }

    #[test]
    fn test_wmma_mma_non_square() {
        // A[2×3] · B[3×4] + C[2×4] = D[2×4]
        let mut a = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::RowMajor, 2, 3);
        for i in 0..2 {
            for j in 0..3 {
                a.set(i, j, (i * 3 + j + 1) as f32);
            }
        }
        let mut b = WmmaFragment::new(FragmentType::MatrixB, FragmentLayout::RowMajor, 3, 4);
        for i in 0..3 {
            for j in 0..4 {
                b.set(i, j, (i * 4 + j + 1) as f32);
            }
        }
        let c = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 4);
        let mut d = WmmaFragment::new(FragmentType::Accumulator, FragmentLayout::RowMajor, 2, 4);

        wmma_mma(&a, &b, &c, &mut d).unwrap();
        // row 0: [1,2,3]·[[1,2,3,4],[5,6,7,8],[9,10,11,12]] = [38,44,50,56]
        assert_eq!(d.get(0, 0), 38.0);
        assert_eq!(d.get(0, 1), 44.0);
        assert_eq!(d.get(0, 2), 50.0);
        assert_eq!(d.get(0, 3), 56.0);
    }

    // ── tensor_core_matmul tests ──────────────────────────────────

    #[test]
    fn test_tc_matmul_identity_2x2() {
        let a = vec![3.0, -2.0, 5.0, 7.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        tensor_core_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_tc_matmul_known_product() {
        #[rustfmt::skip]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        #[rustfmt::skip]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 3).unwrap();
        let mut out = [0.0f32; 4];
        tensor_core_matmul(&a, &b, &mut out, 2, 2, 3, &cfg).unwrap();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn test_tc_matmul_1x1() {
        let cfg = TensorCoreConfig::default().with_fragment_size(1, 1, 1).unwrap();
        let mut out = [0.0f32; 1];
        tensor_core_matmul(&[3.0], &[5.0], &mut out, 1, 1, 1, &cfg).unwrap();
        assert_close(&out, &[15.0], 1e-6);
    }

    #[test]
    fn test_tc_matmul_zero_a() {
        let a = [0.0f32; 9];
        let b: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let cfg = TensorCoreConfig::default().with_fragment_size(4, 4, 4).unwrap();
        let mut out = [0.0f32; 12];
        tensor_core_matmul(&a, &b, &mut out, 3, 4, 3, &cfg).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_tc_matmul_matches_naive() {
        let (m, n, k) = (17, 13, 23);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::default().with_fragment_size(8, 8, 8).unwrap();
        let mut out = vec![0.0f32; m * n];
        tensor_core_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_tc_matmul_large_matrix() {
        let (m, n, k) = (64, 64, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::fp16_fp32();
        let mut out = vec![0.0f32; m * n];
        tensor_core_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close(&out, &expected, 1e-2);
    }

    #[test]
    fn test_tc_matmul_non_aligned_dims() {
        let (m, n, k) = (7, 11, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::fp16_fp32();
        let mut out = vec![0.0f32; m * n];
        tensor_core_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close(&out, &expected, 1e-3);
    }

    #[test]
    fn test_tc_matmul_rejects_zero_dims() {
        let cfg = TensorCoreConfig::default();
        let mut out = [0.0f32; 4];
        assert!(tensor_core_matmul(&[1.0; 4], &[1.0; 4], &mut out, 0, 2, 2, &cfg).is_err());
        assert!(tensor_core_matmul(&[1.0; 4], &[1.0; 4], &mut out, 2, 0, 2, &cfg).is_err());
        assert!(tensor_core_matmul(&[1.0; 4], &[1.0; 4], &mut out, 2, 2, 0, &cfg).is_err());
    }

    #[test]
    fn test_tc_matmul_buffer_too_small_a() {
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        assert!(tensor_core_matmul(&[1.0; 2], &[1.0; 4], &mut out, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_tc_matmul_buffer_too_small_b() {
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        assert!(tensor_core_matmul(&[1.0; 4], &[1.0; 2], &mut out, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_tc_matmul_buffer_too_small_out() {
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 2];
        assert!(tensor_core_matmul(&[1.0; 4], &[1.0; 4], &mut out, 2, 2, 2, &cfg).is_err());
    }

    // ── mixed_precision_tc_matmul tests ───────────────────────────

    #[test]
    fn test_mixed_precision_identity() {
        let a: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = [3.0f32, 7.0, -2.0, 5.0].iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        mixed_precision_tc_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &[3.0, 7.0, -2.0, 5.0], 0.1);
    }

    #[test]
    fn test_mixed_precision_known() {
        let a: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = [5.0f32, 6.0, 7.0, 8.0].iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        mixed_precision_tc_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &[19.0, 22.0, 43.0, 50.0], 0.5);
    }

    #[test]
    fn test_mixed_precision_rejects_zero_dims() {
        let cfg = TensorCoreConfig::fp16_fp32();
        let mut out = [0.0f32; 4];
        assert!(
            mixed_precision_tc_matmul(&[0u16; 4], &[0u16; 4], &mut out, 0, 2, 2, &cfg).is_err()
        );
    }

    #[test]
    fn test_mixed_precision_buffer_too_small() {
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        assert!(
            mixed_precision_tc_matmul(&[0u16; 2], &[0u16; 4], &mut out, 2, 2, 2, &cfg).is_err()
        );
    }

    // ── int8_tc_matmul tests ──────────────────────────────────────

    #[test]
    fn test_int8_identity() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![3, 7, -2, 5];
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0i32; 4];
        int8_tc_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close_i32(&out, &[3, 7, -2, 5]);
    }

    #[test]
    fn test_int8_known_product() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0i32; 4];
        int8_tc_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close_i32(&out, &[19, 22, 43, 50]);
    }

    #[test]
    fn test_int8_matches_naive() {
        let (m, n, k) = (8, 6, 10);
        let a: Vec<i8> = (0..m * k).map(|i| (i % 11) as i8 - 5).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 9) as i8 - 4).collect();
        let expected = naive_i8_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(4, 4, 4).unwrap();
        let mut out = vec![0i32; m * n];
        int8_tc_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close_i32(&out, &expected);
    }

    #[test]
    fn test_int8_zero_weights() {
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b: Vec<i8> = vec![0; 9];
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(3, 3, 3).unwrap();
        let mut out = [0i32; 9];
        int8_tc_matmul(&a, &b, &mut out, 3, 3, 3, &cfg).unwrap();
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_int8_rejects_zero_dims() {
        let cfg = TensorCoreConfig::int8_int32();
        let mut out = [0i32; 4];
        assert!(int8_tc_matmul(&[0i8; 4], &[0i8; 4], &mut out, 0, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_int8_buffer_too_small() {
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0i32; 4];
        assert!(int8_tc_matmul(&[0i8; 2], &[0i8; 4], &mut out, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_int8_1x1() {
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(1, 1, 1).unwrap();
        let mut out = [0i32; 1];
        int8_tc_matmul(&[3i8], &[7i8], &mut out, 1, 1, 1, &cfg).unwrap();
        assert_close_i32(&out, &[21]);
    }

    #[test]
    fn test_int8_negative_values() {
        let a: Vec<i8> = vec![-1, -2, -3, -4];
        let b: Vec<i8> = vec![1, 2, 3, 4];
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0i32; 4];
        int8_tc_matmul(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        // [[-1,-2],[-3,-4]] · [[1,2],[3,4]] = [[-7,-10],[-15,-22]]
        assert_close_i32(&out, &[-7, -10, -15, -22]);
    }

    // ── batched_tc_matmul tests ───────────────────────────────────

    #[test]
    fn test_batched_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 batches × 2×2
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 8];
        batched_tc_matmul(&a, &b, &mut out, 2, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &a, 1e-6);
    }

    #[test]
    fn test_batched_different_data() {
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0];
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 8];
        batched_tc_matmul(&a, &b, &mut out, 2, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &[2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn test_batched_matches_individual() {
        let (batch, m, n, k) = (3, 4, 5, 6);
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let cfg = TensorCoreConfig::default().with_fragment_size(4, 4, 4).unwrap();
        let mut out_batched = vec![0.0f32; batch * m * n];
        batched_tc_matmul(&a, &b, &mut out_batched, batch, m, n, k, &cfg).unwrap();

        for bi in 0..batch {
            let a_sl = &a[bi * m * k..(bi + 1) * m * k];
            let b_sl = &b[bi * k * n..(bi + 1) * k * n];
            let expected = naive_matmul(a_sl, b_sl, m, n, k);
            let out_sl = &out_batched[bi * m * n..(bi + 1) * m * n];
            assert_close(out_sl, &expected, 1e-3);
        }
    }

    #[test]
    fn test_batched_rejects_zero_batch() {
        let cfg = TensorCoreConfig::default();
        let mut out = [0.0f32; 4];
        assert!(batched_tc_matmul(&[1.0; 4], &[1.0; 4], &mut out, 0, 2, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_batched_rejects_zero_dims() {
        let cfg = TensorCoreConfig::default();
        let mut out = [0.0f32; 4];
        assert!(batched_tc_matmul(&[1.0; 4], &[1.0; 4], &mut out, 1, 0, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_batched_buffer_too_small() {
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4]; // need 8 for batch=2
        assert!(batched_tc_matmul(&[1.0; 8], &[1.0; 8], &mut out, 2, 2, 2, 2, &cfg).is_err());
    }

    // ── TensorCoreScheduler tests ─────────────────────────────────

    #[test]
    fn test_scheduler_new() {
        let sched = TensorCoreScheduler::new(128, 256, 64, TensorCoreConfig::default()).unwrap();
        assert_eq!(sched.m, 128);
        assert_eq!(sched.n, 256);
        assert_eq!(sched.k, 64);
        assert_eq!(sched.batch, 1);
    }

    #[test]
    fn test_scheduler_rejects_zero() {
        assert!(TensorCoreScheduler::new(0, 4, 4, TensorCoreConfig::default()).is_err());
        assert!(TensorCoreScheduler::new(4, 0, 4, TensorCoreConfig::default()).is_err());
        assert!(TensorCoreScheduler::new(4, 4, 0, TensorCoreConfig::default()).is_err());
    }

    #[test]
    fn test_scheduler_with_batch() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default())
            .unwrap()
            .with_batch(4)
            .unwrap();
        assert_eq!(sched.batch, 4);
    }

    #[test]
    fn test_scheduler_rejects_zero_batch() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default()).unwrap();
        assert!(sched.with_batch(0).is_err());
    }

    #[test]
    fn test_scheduler_grid_dim() {
        // block_tile_m = 4*16 = 64, block_tile_n = 4*16 = 64
        let sched = TensorCoreScheduler::new(128, 256, 64, TensorCoreConfig::default()).unwrap();
        let (gx, gy, gz) = sched.grid_dim();
        assert_eq!(gx, 4); // ceil(256/64)
        assert_eq!(gy, 2); // ceil(128/64)
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_scheduler_grid_dim_batched() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default())
            .unwrap()
            .with_batch(8)
            .unwrap();
        let (_, _, gz) = sched.grid_dim();
        assert_eq!(gz, 8);
    }

    #[test]
    fn test_scheduler_block_dim() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default()).unwrap();
        let (bx, by, bz) = sched.block_dim();
        assert_eq!(bx, 512); // 4*4*32
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_scheduler_total_wmma_ops() {
        // 128/16=8, 256/16=16, 64/16=4 → 8*16*4 = 512
        let sched = TensorCoreScheduler::new(128, 256, 64, TensorCoreConfig::default()).unwrap();
        assert_eq!(sched.total_wmma_ops(), 512);
    }

    #[test]
    fn test_scheduler_total_wmma_ops_batched() {
        let sched = TensorCoreScheduler::new(16, 16, 16, TensorCoreConfig::default())
            .unwrap()
            .with_batch(3)
            .unwrap();
        assert_eq!(sched.total_wmma_ops(), 3);
    }

    #[test]
    fn test_scheduler_arithmetic_intensity() {
        let sched =
            TensorCoreScheduler::new(1024, 1024, 1024, TensorCoreConfig::fp16_fp32()).unwrap();
        let ai = sched.arithmetic_intensity();
        // 2*1024^3 / (2*1024^2 + 2*1024^2 + 4*1024^2) = 2*1024 / 8 = 256
        assert!((ai - 256.0).abs() < 1.0, "arithmetic intensity: {ai}");
    }

    // ── tc_occupancy tests ────────────────────────────────────────

    #[test]
    fn test_tc_occupancy_full() {
        let sched = TensorCoreScheduler::new(4096, 4096, 64, TensorCoreConfig::default()).unwrap();
        let occ = tc_occupancy(&sched, 80, 16);
        assert!((occ - 1.0).abs() < 1e-4, "occupancy should be capped at 1.0: {occ}");
    }

    #[test]
    fn test_tc_occupancy_partial() {
        // Small problem: grid = (1,1,1) → 1 block
        let sched = TensorCoreScheduler::new(16, 16, 16, TensorCoreConfig::default()).unwrap();
        let occ = tc_occupancy(&sched, 80, 16);
        let expected = 1.0 / (80.0 * 16.0);
        assert!((occ - expected).abs() < 1e-6, "expected {expected}, got {occ}");
    }

    #[test]
    fn test_tc_occupancy_zero_sm() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default()).unwrap();
        assert_eq!(tc_occupancy(&sched, 0, 16), 0.0);
    }

    #[test]
    fn test_tc_occupancy_zero_blocks_per_sm() {
        let sched = TensorCoreScheduler::new(64, 64, 64, TensorCoreConfig::default()).unwrap();
        assert_eq!(tc_occupancy(&sched, 80, 0), 0.0);
    }

    // ── unified dispatch tests ────────────────────────────────────

    #[test]
    fn test_tc_matmul_forward_dispatches_cpu() {
        let a: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0].iter().map(|&v| f32_to_f16(v)).collect();
        let b: Vec<u16> = [2.0f32, 3.0, 4.0, 5.0].iter().map(|&v| f32_to_f16(v)).collect();
        let cfg = TensorCoreConfig::default().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0.0f32; 4];
        tc_matmul_forward(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close(&out, &[2.0, 3.0, 4.0, 5.0], 0.1);
    }

    #[test]
    fn test_int8_tc_matmul_forward_dispatches_cpu() {
        let a: Vec<i8> = vec![1, 0, 0, 1];
        let b: Vec<i8> = vec![2, 3, 4, 5];
        let cfg = TensorCoreConfig::int8_int32().with_fragment_size(2, 2, 2).unwrap();
        let mut out = [0i32; 4];
        int8_tc_matmul_forward(&a, &b, &mut out, 2, 2, 2, &cfg).unwrap();
        assert_close_i32(&out, &[2, 3, 4, 5]);
    }

    // ── CUDA launch stubs (require GPU hardware) ──────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_tc_matmul_launch() {
        let cfg = TensorCoreConfig::fp16_fp32();
        let sched = TensorCoreScheduler::new(256, 256, 256, cfg).unwrap();
        let a = vec![0u16; 256 * 256];
        let b = vec![0u16; 256 * 256];
        let mut out = vec![0.0f32; 256 * 256];
        let config = TensorCoreConfig::fp16_fp32();
        let result = tc_matmul_forward(&a, &b, &mut out, 256, 256, 256, &config);
        let _ = sched; // keep scheduler alive for reference
        assert!(result.is_ok(), "TC matmul launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_int8_tc_matmul_launch() {
        let cfg = TensorCoreConfig::int8_int32();
        let a = vec![0i8; 128 * 128];
        let b = vec![0i8; 128 * 128];
        let mut out = vec![0i32; 128 * 128];
        let result = int8_tc_matmul_forward(&a, &b, &mut out, 128, 128, 128, &cfg);
        assert!(result.is_ok(), "INT8 TC matmul launch failed: {result:?}");
    }

    // ── CUDA kernel source test ───────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_tc_kernel_src_not_empty() {
        assert!(!TENSOR_CORE_MATMUL_KERNEL_SRC.is_empty());
        assert!(TENSOR_CORE_MATMUL_KERNEL_SRC.contains("tc_matmul_f16f32"));
    }

    // ── property-like tests ───────────────────────────────────────

    #[test]
    fn test_property_identity_neutral_tc() {
        for sz in [1, 3, 7, 16, 33] {
            let a: Vec<f32> = (0..sz * sz).map(|i| (i as f32) * 0.1).collect();
            let mut eye = vec![0.0f32; sz * sz];
            for i in 0..sz {
                eye[i * sz + i] = 1.0;
            }
            let cfg = TensorCoreConfig::default().with_fragment_size(4, 4, 4).unwrap();
            let mut out = vec![0.0f32; sz * sz];
            tensor_core_matmul(&a, &eye, &mut out, sz, sz, sz, &cfg).unwrap();
            assert_close(&out, &a, 1e-4);
        }
    }

    #[test]
    fn test_property_zero_annihilates_tc() {
        for sz in [1, 5, 16, 31] {
            let a: Vec<f32> = (0..sz * sz).map(|i| (i as f32) * 0.1).collect();
            let zero = vec![0.0f32; sz * sz];
            let cfg = TensorCoreConfig::default().with_fragment_size(4, 4, 4).unwrap();
            let mut out = vec![0.0f32; sz * sz];
            tensor_core_matmul(&a, &zero, &mut out, sz, sz, sz, &cfg).unwrap();
            assert_close(&out, &zero, 1e-6);
        }
    }

    #[test]
    fn test_property_int8_identity_neutral() {
        for sz in [1, 3, 8, 17] {
            let a: Vec<i8> = (0..sz * sz).map(|i| (i % 7) as i8 - 3).collect();
            let mut eye = vec![0i8; sz * sz];
            for i in 0..sz {
                eye[i * sz + i] = 1;
            }
            let cfg = TensorCoreConfig::int8_int32().with_fragment_size(4, 4, 4).unwrap();
            let mut out = vec![0i32; sz * sz];
            int8_tc_matmul(&a, &eye, &mut out, sz, sz, sz, &cfg).unwrap();
            let expected = naive_i8_matmul(&a, &eye, sz, sz, sz);
            assert_close_i32(&out, &expected);
        }
    }

    #[test]
    fn test_fragment_col_major_roundtrip() {
        let mut frag = WmmaFragment::new(FragmentType::MatrixA, FragmentLayout::ColMajor, 3, 4);
        for i in 0..3 {
            for j in 0..4 {
                frag.set(i, j, (i * 4 + j) as f32);
            }
        }
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(frag.get(i, j), (i * 4 + j) as f32);
            }
        }
    }

    #[test]
    fn test_tc_matmul_tall_matrix() {
        let (m, n, k) = (128, 2, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::default().with_fragment_size(8, 8, 4).unwrap();
        let mut out = vec![0.0f32; m * n];
        tensor_core_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_tc_matmul_wide_matrix() {
        let (m, n, k) = (2, 128, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let expected = naive_matmul(&a, &b, m, n, k);
        let cfg = TensorCoreConfig::default().with_fragment_size(8, 8, 4).unwrap();
        let mut out = vec![0.0f32; m * n];
        tensor_core_matmul(&a, &b, &mut out, m, n, k, &cfg).unwrap();
        assert_close(&out, &expected, 1e-4);
    }
}
