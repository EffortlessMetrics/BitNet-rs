//! CUDA sparse tensor operations for efficient 1-bit model inference.
//!
//! # Sparse formats
//!
//! BitNet models exhibit high sparsity in ternary weight matrices ({-1, 0, +1}),
//! with ~33% zeros in balanced distributions and up to 50%+ after pruning.
//! This module provides sparse storage and compute primitives to exploit that
//! sparsity for reduced memory and faster inference.
//!
//! Supported formats:
//!
//! - **CSR** (Compressed Sparse Row): row-pointer + column-index + values.
//!   Optimal for SpMV and row-slicing.
//! - **CSC** (Compressed Sparse Column): column-pointer + row-index + values.
//!   Optimal for column-slicing and some SpMM layouts.
//! - **COO** (Coordinate): row/col/value triplets.
//!   Easy to construct; converted to CSR/CSC before compute.
//! - **BSR** (Block Sparse Row): CSR with dense sub-blocks.
//!   Exploits spatial locality in weight matrices.
//! - **Block**: Fixed-size dense blocks at known positions.
//!   Used for BitNet block-sparse patterns (block sizes 32/256).
//!
//! # Kernel strategy
//!
//! CUDA kernels use one-thread-per-row for SpMV, and row-parallel tiling for
//! SpMM.  Block-sparse matmul tiles over dense sub-blocks with shared memory.
//! All operations have CPU fallback implementations for correctness testing.
//!
//! # CPU fallback
//!
//! Every public function has a CPU-only implementation that is used when
//! GPU features are not enabled or when GPU launch fails at runtime.

use bitnet_common::{KernelError, Result};

// ── Sparse format enum ───────────────────────────────────────────────

/// Sparse storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Compressed Sparse Row.
    CSR,
    /// Compressed Sparse Column.
    CSC,
    /// Coordinate (triplet) format.
    COO,
    /// Block Sparse Row (CSR with dense sub-blocks).
    BSR,
    /// Fixed-size block layout for BitNet weight patterns.
    Block,
}

impl std::fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSR => write!(f, "CSR"),
            Self::CSC => write!(f, "CSC"),
            Self::COO => write!(f, "COO"),
            Self::BSR => write!(f, "BSR"),
            Self::Block => write!(f, "Block"),
        }
    }
}

// ── Sparse configuration ─────────────────────────────────────────────

/// Configuration for sparse operations.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Sparse storage format.
    pub format: SparseFormat,
    /// Block size for BSR / Block formats (ignored for CSR/CSC/COO).
    pub block_size: usize,
    /// Threshold below which absolute values are treated as zero.
    pub threshold: f32,
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns in the matrix.
    pub cols: usize,
}

impl SparseConfig {
    /// Create a new sparse configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if rows or cols is zero.
    pub fn new(format: SparseFormat, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("sparse dimensions must be non-zero: rows={rows}, cols={cols}"),
            }
            .into());
        }
        Ok(Self { format, block_size: 32, threshold: 0.0, rows, cols })
    }

    /// Set block size (for BSR / Block formats).
    pub fn with_block_size(mut self, block_size: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
            );
        }
        self.block_size = block_size;
        Ok(self)
    }

    /// Set sparsity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

// ── Sparse tensor representation ─────────────────────────────────────

/// A sparse tensor stored in one of the supported formats.
///
/// For CSR: `row_ptrs[i]..row_ptrs[i+1]` indexes into `col_indices` and
/// `values` for row `i`.
///
/// For CSC: `col_ptrs[j]..col_ptrs[j+1]` indexes into `row_indices` and
/// `values` for column `j`.
///
/// For COO: `row_indices[k]`, `col_indices[k]`, `values[k]` form the k-th
/// triplet.
///
/// For BSR/Block: `row_ptrs` and `col_indices` index dense sub-blocks of
/// size `block_size × block_size` stored contiguously in `values`.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// Storage format.
    pub format: SparseFormat,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row pointers (CSR/BSR/Block) or row indices (COO/CSC).
    pub row_ptrs: Vec<usize>,
    /// Column indices (CSR/COO/BSR/Block) or column pointers (CSC).
    pub col_indices: Vec<usize>,
    /// Non-zero values (or dense sub-block values for BSR/Block).
    pub values: Vec<f32>,
    /// Block size (only used for BSR/Block formats).
    pub block_size: usize,
}

impl SparseTensor {
    /// Number of stored non-zero values.
    pub fn nnz(&self) -> usize {
        match self.format {
            SparseFormat::BSR | SparseFormat::Block => {
                // Each stored block contributes block_size² values,
                // but nnz counts the number of non-zero scalar entries.
                self.values.iter().filter(|v| **v != 0.0).count()
            }
            _ => self.values.len(),
        }
    }

    /// Total number of elements in the equivalent dense matrix.
    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }

    /// Sparsity ratio: fraction of elements that are zero.
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.numel();
        if total == 0 {
            return 0.0;
        }
        let nonzeros = match self.format {
            SparseFormat::BSR | SparseFormat::Block => {
                self.values.iter().filter(|v| **v != 0.0).count()
            }
            _ => self.values.len(),
        };
        1.0 - (nonzeros as f64 / total as f64)
    }
}

// ── CUDA kernel source ───────────────────────────────────────────────

/// CUDA C kernel for CSR SpMV: y = A·x.
///
/// One thread per row.  Each thread walks its CSR row segment and
/// accumulates the dot product.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SPARSE_SPMV_CSR_KERNEL_SRC: &str = r#"
extern "C" __global__ void spmv_csr_f32(
    const int*   __restrict__ row_ptrs,
    const int*   __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end   = row_ptrs[row + 1];
    for (int j = start; j < end; j++) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
}
"#;

/// CUDA C kernel for CSR SpMM: C = A·B where A is sparse, B is dense.
///
/// Each thread computes one element of C by walking the sparse row of A
/// and gathering from B.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SPARSE_SPMM_CSR_KERNEL_SRC: &str = r#"
extern "C" __global__ void spmm_csr_f32(
    const int*   __restrict__ row_ptrs,
    const int*   __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int num_rows, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows || col >= B_cols) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end   = row_ptrs[row + 1];
    for (int j = start; j < end; j++) {
        sum += values[j] * B[col_indices[j] * B_cols + col];
    }
    C[row * B_cols + col] = sum;
}
"#;

// ── Conversion: dense → sparse ───────────────────────────────────────

/// Convert a dense row-major matrix to sparse format.
///
/// Elements whose absolute value is ≤ `config.threshold` are treated as
/// zero and excluded from the sparse representation.
///
/// # Errors
///
/// Returns an error if the dense buffer length doesn't match
/// `config.rows × config.cols`.
pub fn dense_to_sparse(dense: &[f32], config: &SparseConfig) -> Result<SparseTensor> {
    let rows = config.rows;
    let cols = config.cols;
    if dense.len() < rows * cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense buffer too small: expected {}, got {}",
                rows * cols,
                dense.len()
            ),
        }
        .into());
    }

    match config.format {
        SparseFormat::CSR => dense_to_csr(dense, rows, cols, config.threshold),
        SparseFormat::CSC => dense_to_csc(dense, rows, cols, config.threshold),
        SparseFormat::COO => dense_to_coo(dense, rows, cols, config.threshold),
        SparseFormat::BSR | SparseFormat::Block => {
            dense_to_bsr(dense, rows, cols, config.block_size, config.threshold)
        }
    }
}

fn dense_to_csr(dense: &[f32], rows: usize, cols: usize, threshold: f32) -> Result<SparseTensor> {
    let mut row_ptrs = Vec::with_capacity(rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0);
    for i in 0..rows {
        for j in 0..cols {
            let v = dense[i * cols + j];
            if v.abs() > threshold {
                col_indices.push(j);
                values.push(v);
            }
        }
        row_ptrs.push(values.len());
    }

    Ok(SparseTensor {
        format: SparseFormat::CSR,
        rows,
        cols,
        row_ptrs,
        col_indices,
        values,
        block_size: 1,
    })
}

fn dense_to_csc(dense: &[f32], rows: usize, cols: usize, threshold: f32) -> Result<SparseTensor> {
    let mut col_ptrs = Vec::with_capacity(cols + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();

    col_ptrs.push(0);
    for j in 0..cols {
        for i in 0..rows {
            let v = dense[i * cols + j];
            if v.abs() > threshold {
                row_indices.push(i);
                values.push(v);
            }
        }
        col_ptrs.push(values.len());
    }

    Ok(SparseTensor {
        format: SparseFormat::CSC,
        rows,
        cols,
        row_ptrs: row_indices,
        col_indices: col_ptrs,
        values,
        block_size: 1,
    })
}

fn dense_to_coo(dense: &[f32], rows: usize, cols: usize, threshold: f32) -> Result<SparseTensor> {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..rows {
        for j in 0..cols {
            let v = dense[i * cols + j];
            if v.abs() > threshold {
                row_indices.push(i);
                col_indices.push(j);
                values.push(v);
            }
        }
    }

    Ok(SparseTensor {
        format: SparseFormat::COO,
        rows,
        cols,
        row_ptrs: row_indices,
        col_indices,
        values,
        block_size: 1,
    })
}

fn dense_to_bsr(
    dense: &[f32],
    rows: usize,
    cols: usize,
    block_size: usize,
    threshold: f32,
) -> Result<SparseTensor> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    let block_rows = rows.div_ceil(block_size);
    let block_cols = cols.div_ceil(block_size);

    let mut row_ptrs = Vec::with_capacity(block_rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0);
    for bi in 0..block_rows {
        for bj in 0..block_cols {
            // Check if block has any non-zero.
            let mut has_nonzero = false;
            for li in 0..block_size {
                let r = bi * block_size + li;
                if r >= rows {
                    break;
                }
                for lj in 0..block_size {
                    let c = bj * block_size + lj;
                    if c >= cols {
                        break;
                    }
                    if dense[r * cols + c].abs() > threshold {
                        has_nonzero = true;
                        break;
                    }
                }
                if has_nonzero {
                    break;
                }
            }

            if has_nonzero {
                col_indices.push(bj);
                for li in 0..block_size {
                    let r = bi * block_size + li;
                    for lj in 0..block_size {
                        let c = bj * block_size + lj;
                        if r < rows && c < cols {
                            values.push(dense[r * cols + c]);
                        } else {
                            values.push(0.0);
                        }
                    }
                }
            }
        }
        row_ptrs.push(col_indices.len());
    }

    Ok(SparseTensor {
        format: SparseFormat::BSR,
        rows,
        cols,
        row_ptrs,
        col_indices,
        values,
        block_size,
    })
}

// ── Conversion: sparse → dense ───────────────────────────────────────

/// Convert a sparse tensor back to dense row-major format.
///
/// # Errors
///
/// Returns an error if the output buffer is too small.
pub fn sparse_to_dense(sparse: &SparseTensor, out: &mut [f32]) -> Result<()> {
    let n = sparse.rows * sparse.cols;
    if out.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("output buffer too small: expected {n}, got {}", out.len()),
        }
        .into());
    }
    out[..n].fill(0.0);

    match sparse.format {
        SparseFormat::CSR => csr_to_dense(sparse, out),
        SparseFormat::CSC => csc_to_dense(sparse, out),
        SparseFormat::COO => coo_to_dense(sparse, out),
        SparseFormat::BSR | SparseFormat::Block => bsr_to_dense(sparse, out),
    }
}

fn csr_to_dense(s: &SparseTensor, out: &mut [f32]) -> Result<()> {
    for i in 0..s.rows {
        let start = s.row_ptrs[i];
        let end = s.row_ptrs[i + 1];
        for idx in start..end {
            let j = s.col_indices[idx];
            out[i * s.cols + j] = s.values[idx];
        }
    }
    Ok(())
}

fn csc_to_dense(s: &SparseTensor, out: &mut [f32]) -> Result<()> {
    // CSC: col_indices stores col_ptrs, row_ptrs stores row_indices.
    let col_ptrs = &s.col_indices;
    let row_indices = &s.row_ptrs;
    for j in 0..s.cols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        for (val, &ri) in s.values[start..end].iter().zip(&row_indices[start..end]) {
            out[ri * s.cols + j] = *val;
        }
    }
    Ok(())
}

fn coo_to_dense(s: &SparseTensor, out: &mut [f32]) -> Result<()> {
    for k in 0..s.values.len() {
        let i = s.row_ptrs[k];
        let j = s.col_indices[k];
        out[i * s.cols + j] = s.values[k];
    }
    Ok(())
}

fn bsr_to_dense(s: &SparseTensor, out: &mut [f32]) -> Result<()> {
    let bs = s.block_size;
    let block_rows = s.rows.div_ceil(bs);
    for bi in 0..block_rows {
        let blk_start = s.row_ptrs[bi];
        let blk_end = s.row_ptrs[bi + 1];
        for blk_idx in blk_start..blk_end {
            let bj = s.col_indices[blk_idx];
            let val_offset = blk_idx * bs * bs;
            for li in 0..bs {
                let r = bi * bs + li;
                if r >= s.rows {
                    break;
                }
                for lj in 0..bs {
                    let c = bj * bs + lj;
                    if c >= s.cols {
                        break;
                    }
                    out[r * s.cols + c] = s.values[val_offset + li * bs + lj];
                }
            }
        }
    }
    Ok(())
}

// ── SpMV: sparse × vector ────────────────────────────────────────────

/// Sparse matrix–vector multiply: y = A·x (CPU fallback).
///
/// Supports CSR, CSC, COO, BSR, and Block formats.
///
/// # Errors
///
/// Returns an error if `x` or `y` have incorrect lengths.
pub fn sparse_matvec(sparse: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    if x.len() < sparse.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("x too small: expected {}, got {}", sparse.cols, x.len()),
        }
        .into());
    }
    if y.len() < sparse.rows {
        return Err(KernelError::InvalidArguments {
            reason: format!("y too small: expected {}, got {}", sparse.rows, y.len()),
        }
        .into());
    }
    y[..sparse.rows].fill(0.0);

    match sparse.format {
        SparseFormat::CSR => spmv_csr(sparse, x, y),
        SparseFormat::CSC => spmv_csc(sparse, x, y),
        SparseFormat::COO => spmv_coo(sparse, x, y),
        SparseFormat::BSR | SparseFormat::Block => spmv_bsr(sparse, x, y),
    }
}

fn spmv_csr(s: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    for (i, y_i) in y.iter_mut().enumerate().take(s.rows) {
        let start = s.row_ptrs[i];
        let end = s.row_ptrs[i + 1];
        let mut acc = 0.0f32;
        for idx in start..end {
            acc += s.values[idx] * x[s.col_indices[idx]];
        }
        *y_i = acc;
    }
    Ok(())
}

fn spmv_csc(s: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    let col_ptrs = &s.col_indices;
    let row_indices = &s.row_ptrs;
    for j in 0..s.cols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        let xj = x[j];
        for idx in start..end {
            y[row_indices[idx]] += s.values[idx] * xj;
        }
    }
    Ok(())
}

fn spmv_coo(s: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    for k in 0..s.values.len() {
        y[s.row_ptrs[k]] += s.values[k] * x[s.col_indices[k]];
    }
    Ok(())
}

fn spmv_bsr(s: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    let bs = s.block_size;
    let block_rows = s.rows.div_ceil(bs);
    for bi in 0..block_rows {
        let blk_start = s.row_ptrs[bi];
        let blk_end = s.row_ptrs[bi + 1];
        for blk_idx in blk_start..blk_end {
            let bj = s.col_indices[blk_idx];
            let val_offset = blk_idx * bs * bs;
            for li in 0..bs {
                let r = bi * bs + li;
                if r >= s.rows {
                    break;
                }
                let mut acc = 0.0f32;
                for lj in 0..bs {
                    let c = bj * bs + lj;
                    if c >= s.cols {
                        break;
                    }
                    acc += s.values[val_offset + li * bs + lj] * x[c];
                }
                y[r] += acc;
            }
        }
    }
    Ok(())
}

// ── SpMM: sparse × dense matrix ─────────────────────────────────────

/// Sparse matrix–matrix multiply: C = A·B (CPU fallback).
///
/// `A` is the sparse tensor (`rows × cols`), `B` is a dense row-major
/// matrix (`cols × b_cols`), and `C` is the dense row-major output
/// (`rows × b_cols`).
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn sparse_matmul(sparse: &SparseTensor, b: &[f32], c: &mut [f32], b_cols: usize) -> Result<()> {
    if b_cols == 0 {
        return Err(KernelError::InvalidArguments { reason: "b_cols must be > 0".into() }.into());
    }
    if b.len() < sparse.cols * b_cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("B too small: expected {}, got {}", sparse.cols * b_cols, b.len()),
        }
        .into());
    }
    if c.len() < sparse.rows * b_cols {
        return Err(KernelError::InvalidArguments {
            reason: format!("C too small: expected {}, got {}", sparse.rows * b_cols, c.len()),
        }
        .into());
    }
    c[..sparse.rows * b_cols].fill(0.0);

    match sparse.format {
        SparseFormat::CSR => spmm_csr(sparse, b, c, b_cols),
        SparseFormat::CSC => spmm_csc(sparse, b, c, b_cols),
        SparseFormat::COO => spmm_coo(sparse, b, c, b_cols),
        SparseFormat::BSR | SparseFormat::Block => spmm_bsr(sparse, b, c, b_cols),
    }
}

fn spmm_csr(s: &SparseTensor, b: &[f32], c: &mut [f32], b_cols: usize) -> Result<()> {
    for i in 0..s.rows {
        let start = s.row_ptrs[i];
        let end = s.row_ptrs[i + 1];
        for idx in start..end {
            let a_val = s.values[idx];
            let col_a = s.col_indices[idx];
            let b_row = &b[col_a * b_cols..(col_a + 1) * b_cols];
            let c_row = &mut c[i * b_cols..(i + 1) * b_cols];
            for k in 0..b_cols {
                c_row[k] += a_val * b_row[k];
            }
        }
    }
    Ok(())
}

fn spmm_csc(s: &SparseTensor, b: &[f32], c: &mut [f32], b_cols: usize) -> Result<()> {
    let col_ptrs = &s.col_indices;
    let row_indices = &s.row_ptrs;
    for j in 0..s.cols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        let b_row = &b[j * b_cols..(j + 1) * b_cols];
        for (&ri, &a_val) in row_indices[start..end].iter().zip(&s.values[start..end]) {
            let c_row = &mut c[ri * b_cols..(ri + 1) * b_cols];
            for k in 0..b_cols {
                c_row[k] += a_val * b_row[k];
            }
        }
    }
    Ok(())
}

fn spmm_coo(s: &SparseTensor, b: &[f32], c: &mut [f32], b_cols: usize) -> Result<()> {
    for k in 0..s.values.len() {
        let i = s.row_ptrs[k];
        let col_a = s.col_indices[k];
        let a_val = s.values[k];
        let b_row = &b[col_a * b_cols..(col_a + 1) * b_cols];
        let c_row = &mut c[i * b_cols..(i + 1) * b_cols];
        for bk in 0..b_cols {
            c_row[bk] += a_val * b_row[bk];
        }
    }
    Ok(())
}

fn spmm_bsr(s: &SparseTensor, b: &[f32], c: &mut [f32], b_cols: usize) -> Result<()> {
    let bs = s.block_size;
    let block_rows = s.rows.div_ceil(bs);
    for bi in 0..block_rows {
        let blk_start = s.row_ptrs[bi];
        let blk_end = s.row_ptrs[bi + 1];
        for blk_idx in blk_start..blk_end {
            let bj = s.col_indices[blk_idx];
            let val_offset = blk_idx * bs * bs;
            for li in 0..bs {
                let r = bi * bs + li;
                if r >= s.rows {
                    break;
                }
                for lj in 0..bs {
                    let col_a = bj * bs + lj;
                    if col_a >= s.cols {
                        break;
                    }
                    let a_val = s.values[val_offset + li * bs + lj];
                    if a_val == 0.0 {
                        continue;
                    }
                    let b_row = &b[col_a * b_cols..(col_a + 1) * b_cols];
                    let c_row = &mut c[r * b_cols..(r + 1) * b_cols];
                    for k in 0..b_cols {
                        c_row[k] += a_val * b_row[k];
                    }
                }
            }
        }
    }
    Ok(())
}

// ── Element-wise sparse operations ───────────────────────────────────

/// Element-wise operation on two sparse tensors (both must be CSR with
/// matching dimensions). Returns a new dense vector.
///
/// # Errors
///
/// Returns an error if dimensions or formats don't match.
pub fn sparse_elementwise(
    a: &SparseTensor,
    b: &SparseTensor,
    op: ElementwiseSpOp,
) -> Result<Vec<f32>> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dimension mismatch: ({},{}) vs ({},{})",
                a.rows, a.cols, b.rows, b.cols
            ),
        }
        .into());
    }
    // Materialise both to dense, apply op, return dense result.
    let n = a.rows * a.cols;
    let mut da = vec![0.0f32; n];
    let mut db = vec![0.0f32; n];
    sparse_to_dense(a, &mut da)?;
    sparse_to_dense(b, &mut db)?;

    let out = da
        .iter()
        .zip(db.iter())
        .map(|(&x, &y)| match op {
            ElementwiseSpOp::Add => x + y,
            ElementwiseSpOp::Sub => x - y,
            ElementwiseSpOp::Mul => x * y,
        })
        .collect();
    Ok(out)
}

/// Supported element-wise sparse operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseSpOp {
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise (Hadamard) multiplication.
    Mul,
}

/// Add two sparse tensors (convenience wrapper).
pub fn sparse_add(a: &SparseTensor, b: &SparseTensor) -> Result<Vec<f32>> {
    sparse_elementwise(a, b, ElementwiseSpOp::Add)
}

/// Subtract two sparse tensors (convenience wrapper).
pub fn sparse_sub(a: &SparseTensor, b: &SparseTensor) -> Result<Vec<f32>> {
    sparse_elementwise(a, b, ElementwiseSpOp::Sub)
}

// ── Utility functions ────────────────────────────────────────────────

/// Count non-zero elements in a dense buffer using a threshold.
pub fn nnz(data: &[f32], threshold: f32) -> usize {
    data.iter().filter(|v| v.abs() > threshold).count()
}

/// Calculate sparsity ratio of a dense buffer.
pub fn sparsity_ratio(data: &[f32], threshold: f32) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let nonzeros = nnz(data, threshold);
    1.0 - (nonzeros as f64 / data.len() as f64)
}

/// Prune values below a threshold, setting them to zero in-place.
/// Returns the number of elements pruned.
pub fn prune_below_threshold(data: &mut [f32], threshold: f32) -> usize {
    let mut count = 0;
    for v in data.iter_mut() {
        if v.abs() <= threshold {
            *v = 0.0;
            count += 1;
        }
    }
    count
}

// ── Block-sparse matmul ──────────────────────────────────────────────

/// Block-sparse matrix multiply optimised for BitNet weight patterns.
///
/// Takes a dense activation matrix `a` (`m × k`, row-major) and a
/// block-sparse weight tensor in BSR/Block format, producing a dense
/// output `c` (`m × n`, row-major).
///
/// This is the key operation for BitNet inference: the ternary weight
/// matrix is block-sparse after pruning, and this routine avoids
/// multiplying with zero blocks entirely.
///
/// # Errors
///
/// Returns an error if dimensions are inconsistent.
pub fn block_sparse_matmul(
    a: &[f32],
    sparse_b: &SparseTensor,
    c: &mut [f32],
    m: usize,
) -> Result<()> {
    if !matches!(sparse_b.format, SparseFormat::BSR | SparseFormat::Block) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "block_sparse_matmul requires BSR or Block format, got {}",
                sparse_b.format
            ),
        }
        .into());
    }
    let k = sparse_b.rows;
    let n = sparse_b.cols;

    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("a too small: expected {}, got {}", m * k, a.len()),
        }
        .into());
    }
    if c.len() < m * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("c too small: expected {}, got {}", m * n, c.len()),
        }
        .into());
    }
    c[..m * n].fill(0.0);

    // A is m×k, sparse_b (transposed view) is k×n in BSR.
    // We compute C = A · B^T where B^T is the BSR tensor.
    // Actually: sparse_b *is* the weight matrix (k×n), stored in BSR.
    // We iterate block-rows of B to compute contributions.
    let bs = sparse_b.block_size;
    let block_rows = k.div_ceil(bs);

    for bi in 0..block_rows {
        let blk_start = sparse_b.row_ptrs[bi];
        let blk_end = sparse_b.row_ptrs[bi + 1];
        for blk_idx in blk_start..blk_end {
            let bj = sparse_b.col_indices[blk_idx];
            let val_offset = blk_idx * bs * bs;

            // For each activation row
            for row in 0..m {
                for li in 0..bs {
                    let r = bi * bs + li;
                    if r >= k {
                        break;
                    }
                    let a_val = a[row * k + r];
                    if a_val == 0.0 {
                        continue;
                    }
                    for lj in 0..bs {
                        let col = bj * bs + lj;
                        if col >= n {
                            break;
                        }
                        c[row * n + col] += a_val * sparse_b.values[val_offset + li * bs + lj];
                    }
                }
            }
        }
    }
    Ok(())
}

// ── CUDA launch stubs ────────────────────────────────────────────────

/// Launch stub for CSR SpMV CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_sparse_spmv(_sparse: &SparseTensor, _x: &[f32], _y: &mut [f32]) -> Result<()> {
    log::debug!("sparse SpMV CUDA stub: rows={}, nnz={}", _sparse.rows, _sparse.nnz(),);
    Err(KernelError::GpuError {
        reason: "sparse SpMV CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for CSR SpMM CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_sparse_spmm(
    _sparse: &SparseTensor,
    _b: &[f32],
    _c: &mut [f32],
    _b_cols: usize,
) -> Result<()> {
    log::debug!(
        "sparse SpMM CUDA stub: rows={}, nnz={}, b_cols={}",
        _sparse.rows,
        _sparse.nnz(),
        _b_cols,
    );
    Err(KernelError::GpuError {
        reason: "sparse SpMM CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unified dispatch ─────────────────────────────────────────────────

/// SpMV with automatic dispatch: GPU if available, else CPU fallback.
pub fn sparse_matvec_forward(sparse: &SparseTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_sparse_spmv(sparse, x, y)
        {
            return Ok(());
        }
    }
    sparse_matvec(sparse, x, y)
}

/// SpMM with automatic dispatch: GPU if available, else CPU fallback.
pub fn sparse_matmul_forward(
    sparse: &SparseTensor,
    b: &[f32],
    c: &mut [f32],
    b_cols: usize,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_sparse_spmm(sparse, b, c, b_cols)
        {
            return Ok(());
        }
    }
    sparse_matmul(sparse, b, c, b_cols)
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

    /// Naive dense matmul: C = A · B (row-major).
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

    /// Naive dense matvec: y = A · x.
    fn naive_matvec(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..k {
                y[i] += a[i * k + j] * x[j];
            }
        }
        y
    }

    // A small test matrix with known sparsity pattern:
    //  [ 1  0  2  0 ]
    //  [ 0  3  0  0 ]
    //  [ 4  0  0  5 ]
    fn test_matrix_3x4() -> Vec<f32> {
        vec![
            1.0, 0.0, 2.0, 0.0, // row 0
            0.0, 3.0, 0.0, 0.0, // row 1
            4.0, 0.0, 0.0, 5.0, // row 2
        ]
    }

    // Identity-like sparse 4×4:
    //  [ 1  0  0  0 ]
    //  [ 0  1  0  0 ]
    //  [ 0  0  1  0 ]
    //  [ 0  0  0  1 ]
    fn identity_4x4() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ]
    }

    fn all_formats() -> Vec<SparseFormat> {
        vec![SparseFormat::CSR, SparseFormat::CSC, SparseFormat::COO, SparseFormat::BSR]
    }

    // ─── SparseFormat Display ─────────────────────────────────────

    #[test]
    fn test_format_display() {
        assert_eq!(SparseFormat::CSR.to_string(), "CSR");
        assert_eq!(SparseFormat::CSC.to_string(), "CSC");
        assert_eq!(SparseFormat::COO.to_string(), "COO");
        assert_eq!(SparseFormat::BSR.to_string(), "BSR");
        assert_eq!(SparseFormat::Block.to_string(), "Block");
    }

    // ─── SparseConfig tests ──────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.threshold, 0.0);
        assert_eq!(cfg.rows, 4);
        assert_eq!(cfg.cols, 4);
    }

    #[test]
    fn test_config_rejects_zero_rows() {
        assert!(SparseConfig::new(SparseFormat::CSR, 0, 4).is_err());
    }

    #[test]
    fn test_config_rejects_zero_cols() {
        assert!(SparseConfig::new(SparseFormat::CSR, 4, 0).is_err());
    }

    #[test]
    fn test_config_with_block_size() {
        let cfg = SparseConfig::new(SparseFormat::BSR, 8, 8).unwrap().with_block_size(4).unwrap();
        assert_eq!(cfg.block_size, 4);
    }

    #[test]
    fn test_config_rejects_zero_block_size() {
        let cfg = SparseConfig::new(SparseFormat::BSR, 8, 8).unwrap();
        assert!(cfg.with_block_size(0).is_err());
    }

    #[test]
    fn test_config_with_threshold() {
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap().with_threshold(0.01);
        assert_eq!(cfg.threshold, 0.01);
    }

    // ─── CSR conversion round-trip ───────────────────────────────

    #[test]
    fn test_csr_roundtrip_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.format, SparseFormat::CSR);
        assert_eq!(sparse.nnz(), 5);

        let mut out = vec![0.0f32; 12];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_csr_roundtrip_identity() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 4);

        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── CSC conversion round-trip ───────────────────────────────

    #[test]
    fn test_csc_roundtrip_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSC, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.format, SparseFormat::CSC);
        assert_eq!(sparse.nnz(), 5);

        let mut out = vec![0.0f32; 12];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_csc_roundtrip_identity() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSC, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 4);

        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── COO conversion round-trip ───────────────────────────────

    #[test]
    fn test_coo_roundtrip_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::COO, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.format, SparseFormat::COO);
        assert_eq!(sparse.nnz(), 5);

        let mut out = vec![0.0f32; 12];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── BSR conversion round-trip ───────────────────────────────

    #[test]
    fn test_bsr_roundtrip_4x4_bs2() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.format, SparseFormat::BSR);

        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_bsr_roundtrip_3x4_bs2() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 3, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();

        let mut out = vec![0.0f32; 12];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_block_roundtrip_alias() {
        // Block format uses BSR path internally.
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::Block, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();

        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── Threshold-based conversion ──────────────────────────────

    #[test]
    fn test_csr_with_threshold() {
        let dense = vec![0.5, 0.01, 0.0, 1.0];
        let cfg = SparseConfig::new(SparseFormat::CSR, 1, 4).unwrap().with_threshold(0.1);
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        // Only 0.5 and 1.0 should survive (0.01 and 0.0 pruned).
        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_coo_with_threshold() {
        let dense = vec![0.05, 0.2, 0.0, 0.3];
        let cfg = SparseConfig::new(SparseFormat::COO, 2, 2).unwrap().with_threshold(0.1);
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 2); // 0.2 and 0.3
    }

    // ─── Error: buffer too small ─────────────────────────────────

    #[test]
    fn test_dense_to_sparse_buffer_too_small() {
        let dense = vec![1.0, 2.0]; // need 4
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        assert!(dense_to_sparse(&dense, &cfg).is_err());
    }

    #[test]
    fn test_sparse_to_dense_buffer_too_small() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let mut out = vec![0.0f32; 8]; // need 16
        assert!(sparse_to_dense(&sparse, &mut out).is_err());
    }

    // ─── All-zero matrix ─────────────────────────────────────────

    #[test]
    fn test_all_zero_matrix_csr() {
        let dense = vec![0.0f32; 12];
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 0);

        let mut out = vec![1.0f32; 12];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_all_zero_matrix_csc() {
        let dense = vec![0.0f32; 6];
        let cfg = SparseConfig::new(SparseFormat::CSC, 2, 3).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 0);
    }

    #[test]
    fn test_all_zero_matrix_coo() {
        let dense = vec![0.0f32; 6];
        let cfg = SparseConfig::new(SparseFormat::COO, 2, 3).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 0);
    }

    // ─── Fully dense matrix ──────────────────────────────────────

    #[test]
    fn test_fully_dense_matrix_csr() {
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 3).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 6);

        let mut out = vec![0.0f32; 6];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── SpMV: all formats ───────────────────────────────────────

    #[test]
    fn test_spmv_csr_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 3];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        let expected = naive_matvec(&dense, &x, 3, 4);
        assert_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_spmv_csc_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSC, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 3];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        let expected = naive_matvec(&dense, &x, 3, 4);
        assert_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_spmv_coo_3x4() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::COO, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 3];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        let expected = naive_matvec(&dense, &x, 3, 4);
        assert_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_spmv_bsr_4x4() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0f32; 4];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        assert_close(&y, &x, 1e-6);
    }

    #[test]
    fn test_spmv_identity_all_formats() {
        let dense = identity_4x4();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        for fmt in all_formats() {
            let mut cfg = SparseConfig::new(fmt, 4, 4).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(2).unwrap();
            }
            let sparse = dense_to_sparse(&dense, &cfg).unwrap();
            let mut y = vec![0.0f32; 4];
            sparse_matvec(&sparse, &x, &mut y).unwrap();
            assert_close(&y, &x, 1e-6);
        }
    }

    #[test]
    fn test_spmv_x_too_small() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![1.0, 2.0]; // too small
        let mut y = vec![0.0f32; 4];
        assert!(sparse_matvec(&sparse, &x, &mut y).is_err());
    }

    #[test]
    fn test_spmv_y_too_small() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 2]; // too small
        assert!(sparse_matvec(&sparse, &x, &mut y).is_err());
    }

    // ─── SpMM: all formats ───────────────────────────────────────

    #[test]
    fn test_spmm_csr_3x4_times_4x2() {
        let dense_a = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4×2
        let mut c = vec![0.0f32; 6]; // 3×2
        sparse_matmul(&sparse, &b, &mut c, 2).unwrap();
        let expected = naive_matmul(&dense_a, &b, 3, 4, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_spmm_csc_3x4_times_4x2() {
        let dense_a = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSC, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 6];
        sparse_matmul(&sparse, &b, &mut c, 2).unwrap();
        let expected = naive_matmul(&dense_a, &b, 3, 4, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_spmm_coo_3x4_times_4x2() {
        let dense_a = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::COO, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 6];
        sparse_matmul(&sparse, &b, &mut c, 2).unwrap();
        let expected = naive_matmul(&dense_a, &b, 3, 4, 2);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_spmm_bsr_4x4_times_4x3() {
        let dense_a = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        #[rustfmt::skip]
        let b = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0f32; 12];
        sparse_matmul(&sparse, &b, &mut c, 3).unwrap();
        // Identity × B = B
        assert_close(&c, &b, 1e-5);
    }

    #[test]
    fn test_spmm_identity_all_formats() {
        let dense_a = identity_4x4();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4×2
        let expected = b.clone(); // Identity × B = B
        for fmt in all_formats() {
            let mut cfg = SparseConfig::new(fmt, 4, 4).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(2).unwrap();
            }
            let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
            let mut c = vec![0.0f32; 8];
            sparse_matmul(&sparse, &b, &mut c, 2).unwrap();
            assert_close(&c, &expected, 1e-5);
        }
    }

    #[test]
    fn test_spmm_b_too_small() {
        let dense_a = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let b = vec![1.0, 2.0]; // too small for 4×2
        let mut c = vec![0.0f32; 8];
        assert!(sparse_matmul(&sparse, &b, &mut c, 2).is_err());
    }

    #[test]
    fn test_spmm_c_too_small() {
        let dense_a = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4]; // need 8
        assert!(sparse_matmul(&sparse, &b, &mut c, 2).is_err());
    }

    #[test]
    fn test_spmm_zero_b_cols() {
        let dense_a = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
        let mut c = vec![0.0f32; 0];
        assert!(sparse_matmul(&sparse, &[], &mut c, 0).is_err());
    }

    // ─── Element-wise operations ─────────────────────────────────

    #[test]
    fn test_sparse_add_csr() {
        let a_dense = vec![1.0, 0.0, 2.0, 0.0];
        let b_dense = vec![0.0, 3.0, 1.0, 0.0];
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        let a = dense_to_sparse(&a_dense, &cfg).unwrap();
        let b = dense_to_sparse(&b_dense, &cfg).unwrap();
        let result = sparse_add(&a, &b).unwrap();
        assert_close(&result, &[1.0, 3.0, 3.0, 0.0], 1e-6);
    }

    #[test]
    fn test_sparse_sub_csr() {
        let a_dense = vec![5.0, 0.0, 3.0, 0.0];
        let b_dense = vec![1.0, 0.0, 1.0, 0.0];
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        let a = dense_to_sparse(&a_dense, &cfg).unwrap();
        let b = dense_to_sparse(&b_dense, &cfg).unwrap();
        let result = sparse_sub(&a, &b).unwrap();
        assert_close(&result, &[4.0, 0.0, 2.0, 0.0], 1e-6);
    }

    #[test]
    fn test_sparse_elementwise_mul() {
        let a_dense = vec![2.0, 0.0, 3.0, 4.0];
        let b_dense = vec![1.0, 5.0, 2.0, 0.0];
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        let a = dense_to_sparse(&a_dense, &cfg).unwrap();
        let b = dense_to_sparse(&b_dense, &cfg).unwrap();
        let result = sparse_elementwise(&a, &b, ElementwiseSpOp::Mul).unwrap();
        assert_close(&result, &[2.0, 0.0, 6.0, 0.0], 1e-6);
    }

    #[test]
    fn test_sparse_add_different_formats() {
        // CSR + COO — both materialise to dense.
        let a_dense = vec![1.0, 2.0, 3.0, 4.0];
        let b_dense = vec![4.0, 3.0, 2.0, 1.0];
        let cfg_a = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        let cfg_b = SparseConfig::new(SparseFormat::COO, 2, 2).unwrap();
        let a = dense_to_sparse(&a_dense, &cfg_a).unwrap();
        let b = dense_to_sparse(&b_dense, &cfg_b).unwrap();
        let result = sparse_add(&a, &b).unwrap();
        assert_close(&result, &[5.0, 5.0, 5.0, 5.0], 1e-6);
    }

    #[test]
    fn test_sparse_elementwise_dimension_mismatch() {
        let cfg_a = SparseConfig::new(SparseFormat::CSR, 2, 2).unwrap();
        let cfg_b = SparseConfig::new(SparseFormat::CSR, 3, 2).unwrap();
        let a = dense_to_sparse(&[1.0, 2.0, 3.0, 4.0], &cfg_a).unwrap();
        let b = dense_to_sparse(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &cfg_b).unwrap();
        assert!(sparse_add(&a, &b).is_err());
    }

    // ─── nnz / sparsity_ratio helpers ────────────────────────────

    #[test]
    fn test_nnz_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(nnz(&data, 0.0), 4);
    }

    #[test]
    fn test_nnz_sparse() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        assert_eq!(nnz(&data, 0.0), 2);
    }

    #[test]
    fn test_nnz_with_threshold() {
        let data = vec![0.5, 0.01, 0.0, 1.0];
        assert_eq!(nnz(&data, 0.1), 2);
    }

    #[test]
    fn test_nnz_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(nnz(&data, 0.0), 0);
    }

    #[test]
    fn test_sparsity_ratio_half() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let ratio = sparsity_ratio(&data, 0.0);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_all_zero() {
        let data = vec![0.0; 10];
        let ratio = sparsity_ratio(&data, 0.0);
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_no_zeros() {
        let data = vec![1.0, 2.0, 3.0];
        let ratio = sparsity_ratio(&data, 0.0);
        assert!(ratio.abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_ratio_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(sparsity_ratio(&data, 0.0), 0.0);
    }

    // ─── prune_below_threshold ───────────────────────────────────

    #[test]
    fn test_prune_basic() {
        let mut data = vec![0.5, 0.01, 0.0, 1.0, -0.05, -2.0];
        let pruned = prune_below_threshold(&mut data, 0.1);
        assert_eq!(pruned, 3);
        assert_eq!(data, vec![0.5, 0.0, 0.0, 1.0, 0.0, -2.0]);
    }

    #[test]
    fn test_prune_nothing() {
        let mut data = vec![1.0, 2.0, 3.0];
        let pruned = prune_below_threshold(&mut data, 0.0);
        // threshold=0 means only exact 0 is pruned
        assert_eq!(pruned, 0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_prune_everything() {
        let mut data = vec![0.01, 0.02, 0.03];
        let pruned = prune_below_threshold(&mut data, 0.1);
        assert_eq!(pruned, 3);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_prune_negative_values() {
        let mut data = vec![-0.5, -0.01, 0.01, 0.5];
        let pruned = prune_below_threshold(&mut data, 0.1);
        assert_eq!(pruned, 2);
        assert_eq!(data, vec![-0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_prune_empty() {
        let mut data: Vec<f32> = vec![];
        let pruned = prune_below_threshold(&mut data, 0.1);
        assert_eq!(pruned, 0);
    }

    // ─── SparseTensor methods ────────────────────────────────────

    #[test]
    fn test_sparse_tensor_nnz() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 5);
    }

    #[test]
    fn test_sparse_tensor_numel() {
        let dense = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.numel(), 12);
    }

    #[test]
    fn test_sparse_tensor_sparsity_ratio() {
        let dense = test_matrix_3x4(); // 5 non-zeros out of 12
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let ratio = sparse.sparsity_ratio();
        let expected = 1.0 - (5.0 / 12.0);
        assert!((ratio - expected).abs() < 1e-10);
    }

    // ─── block_sparse_matmul ─────────────────────────────────────

    #[test]
    fn test_block_sparse_matmul_identity() {
        let dense_w = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse_w = dense_to_sparse(&dense_w, &cfg).unwrap();

        // A is 2×4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 8]; // 2×4
        block_sparse_matmul(&a, &sparse_w, &mut c, 2).unwrap();
        assert_close(&c, &a, 1e-5);
    }

    #[test]
    fn test_block_sparse_matmul_3x4_weights() {
        let dense_w = test_matrix_3x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 3, 4).unwrap().with_block_size(2).unwrap();
        let sparse_w = dense_to_sparse(&dense_w, &cfg).unwrap();

        // A is 1×3 (one row, k=3 = sparse_w.rows)
        let a = vec![1.0, 2.0, 3.0];
        let mut c = vec![0.0f32; 4]; // 1×4
        block_sparse_matmul(&a, &sparse_w, &mut c, 1).unwrap();

        let expected = naive_matmul(&a, &dense_w, 1, 3, 4);
        assert_close(&c, &expected, 1e-5);
    }

    #[test]
    fn test_block_sparse_matmul_rejects_csr() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let a = vec![1.0; 16];
        let mut c = vec![0.0f32; 16];
        assert!(block_sparse_matmul(&a, &sparse, &mut c, 4).is_err());
    }

    #[test]
    fn test_block_sparse_matmul_a_too_small() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let a = vec![1.0, 2.0]; // need 4
        let mut c = vec![0.0f32; 4];
        assert!(block_sparse_matmul(&a, &sparse, &mut c, 1).is_err());
    }

    #[test]
    fn test_block_sparse_matmul_c_too_small() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let a = vec![1.0; 4];
        let mut c = vec![0.0f32; 2]; // need 4
        assert!(block_sparse_matmul(&a, &sparse, &mut c, 1).is_err());
    }

    // ─── Forward dispatch (CPU fallback) ─────────────────────────

    #[test]
    fn test_spmv_forward_fallback() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0f32; 4];
        sparse_matvec_forward(&sparse, &x, &mut y).unwrap();
        assert_close(&y, &x, 1e-6);
    }

    #[test]
    fn test_spmm_forward_fallback() {
        let dense = identity_4x4();
        let cfg = SparseConfig::new(SparseFormat::CSR, 4, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 8];
        sparse_matmul_forward(&sparse, &b, &mut c, 2).unwrap();
        assert_close(&c, &b, 1e-5);
    }

    // ─── Ternary weight patterns (BitNet-relevant) ───────────────

    #[test]
    fn test_ternary_weights_csr() {
        // Simulate a ternary weight matrix: {-1, 0, +1}
        #[rustfmt::skip]
        let dense = vec![
             1.0,  0.0, -1.0,  0.0,
             0.0,  1.0,  0.0, -1.0,
            -1.0,  0.0,  1.0,  0.0,
        ];
        let cfg = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 6);
        let ratio = sparse.sparsity_ratio();
        assert!((ratio - 0.5).abs() < 1e-10);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 3];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        let expected = naive_matvec(&dense, &x, 3, 4);
        assert_close(&y, &expected, 1e-6);
    }

    #[test]
    fn test_ternary_weights_bsr() {
        #[rustfmt::skip]
        let dense = vec![
             1.0,  0.0, -1.0,  0.0,
             0.0,  1.0,  0.0, -1.0,
            -1.0,  0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,  1.0,
        ];
        let cfg = SparseConfig::new(SparseFormat::BSR, 4, 4).unwrap().with_block_size(2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        let expected = naive_matvec(&dense, &x, 4, 4);
        assert_close(&y, &expected, 1e-6);
    }

    // ─── Large-ish matrix correctness ────────────────────────────

    #[test]
    fn test_spmv_large_sparse() {
        // 64×64 matrix with ~25% non-zero.
        let n = 64;
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if (i + j) % 4 == 0 {
                    dense[i * n + j] = ((i * 7 + j * 3) % 19) as f32 - 9.0;
                }
            }
        }
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();

        for fmt in all_formats() {
            let mut cfg = SparseConfig::new(fmt, n, n).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(4).unwrap();
            }
            let sparse = dense_to_sparse(&dense, &cfg).unwrap();
            let mut y = vec![0.0f32; n];
            sparse_matvec(&sparse, &x, &mut y).unwrap();
            let expected = naive_matvec(&dense, &x, n, n);
            assert_close(&y, &expected, 1e-3);
        }
    }

    #[test]
    fn test_spmm_large_sparse() {
        let m = 32;
        let k = 32;
        let n = 16;
        let mut dense_a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                if (i + j) % 3 == 0 {
                    dense_a[i * k + j] = ((i * 5 + j * 2) % 11) as f32 - 5.0;
                }
            }
        }
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5 - 1.5).collect();

        for fmt in all_formats() {
            let mut cfg = SparseConfig::new(fmt, m, k).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(4).unwrap();
            }
            let sparse = dense_to_sparse(&dense_a, &cfg).unwrap();
            let mut c = vec![0.0f32; m * n];
            sparse_matmul(&sparse, &b, &mut c, n).unwrap();
            let expected = naive_matmul(&dense_a, &b, m, k, n);
            assert_close(&c, &expected, 1e-2);
        }
    }

    // ─── 1×1 edge case ──────────────────────────────────────────

    #[test]
    fn test_1x1_matrix_all_formats() {
        for fmt in all_formats() {
            let dense = vec![42.0f32];
            let mut cfg = SparseConfig::new(fmt, 1, 1).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(1).unwrap();
            }
            let sparse = dense_to_sparse(&dense, &cfg).unwrap();

            let mut out = vec![0.0f32; 1];
            sparse_to_dense(&sparse, &mut out).unwrap();
            assert_close(&out, &dense, 0.0);

            // SpMV
            let x = vec![2.0f32];
            let mut y = vec![0.0f32; 1];
            sparse_matvec(&sparse, &x, &mut y).unwrap();
            assert_close(&y, &[84.0], 1e-6);
        }
    }

    #[test]
    fn test_1x1_zero_matrix() {
        let dense = vec![0.0f32];
        let cfg = SparseConfig::new(SparseFormat::CSR, 1, 1).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 0);

        let x = vec![5.0f32];
        let mut y = vec![999.0f32; 1];
        sparse_matvec(&sparse, &x, &mut y).unwrap();
        assert_close(&y, &[0.0], 0.0);
    }

    // ─── Negative values ─────────────────────────────────────────

    #[test]
    fn test_negative_values_roundtrip() {
        let dense = vec![-1.0, -2.0, 0.0, -3.0];
        for fmt in all_formats() {
            let mut cfg = SparseConfig::new(fmt, 2, 2).unwrap();
            if matches!(fmt, SparseFormat::BSR) {
                cfg = cfg.with_block_size(1).unwrap();
            }
            let sparse = dense_to_sparse(&dense, &cfg).unwrap();
            let mut out = vec![0.0f32; 4];
            sparse_to_dense(&sparse, &mut out).unwrap();
            assert_close(&out, &dense, 0.0);
        }
    }

    // ─── BSR block_size == 1 is equivalent to CSR ────────────────

    #[test]
    fn test_bsr_block_size_1_matches_csr() {
        let dense = test_matrix_3x4();
        let cfg_csr = SparseConfig::new(SparseFormat::CSR, 3, 4).unwrap();
        let cfg_bsr =
            SparseConfig::new(SparseFormat::BSR, 3, 4).unwrap().with_block_size(1).unwrap();
        let sparse_csr = dense_to_sparse(&dense, &cfg_csr).unwrap();
        let sparse_bsr = dense_to_sparse(&dense, &cfg_bsr).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y_csr = vec![0.0f32; 3];
        let mut y_bsr = vec![0.0f32; 3];
        sparse_matvec(&sparse_csr, &x, &mut y_csr).unwrap();
        sparse_matvec(&sparse_bsr, &x, &mut y_bsr).unwrap();
        assert_close(&y_csr, &y_bsr, 1e-6);
    }

    // ─── Non-square matrices ─────────────────────────────────────

    #[test]
    fn test_wide_matrix_2x8() {
        let dense: Vec<f32> = (0..16).map(|i| if i % 3 == 0 { i as f32 } else { 0.0 }).collect();
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 8).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    #[test]
    fn test_tall_matrix_8x2() {
        let dense: Vec<f32> = (0..16).map(|i| if i % 5 == 0 { i as f32 } else { 0.0 }).collect();
        let cfg = SparseConfig::new(SparseFormat::COO, 8, 2).unwrap();
        let sparse = dense_to_sparse(&dense, &cfg).unwrap();
        let mut out = vec![0.0f32; 16];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &dense, 0.0);
    }

    // ─── prune then convert ──────────────────────────────────────

    #[test]
    fn test_prune_then_sparse_convert() {
        let mut data = vec![0.5, 0.01, 0.0, 1.0, -0.05, -2.0, 0.0, 0.03];
        prune_below_threshold(&mut data, 0.1);
        let cfg = SparseConfig::new(SparseFormat::CSR, 2, 4).unwrap();
        let sparse = dense_to_sparse(&data, &cfg).unwrap();
        assert_eq!(sparse.nnz(), 3); // 0.5, 1.0, -2.0

        let mut out = vec![0.0f32; 8];
        sparse_to_dense(&sparse, &mut out).unwrap();
        assert_close(&out, &data, 0.0);
    }
}
