//! OpenCL-optimized sparse matrix operations for efficient inference with pruned models.
//!
//! # Overview
//!
//! This module provides sparse matrix representations and operations commonly
//! needed when accelerating inference on pruned / sparse neural network weights.
//! Three classical sparse formats are supported:
//!
//! - **CSR** (Compressed Sparse Row) — fast row slicing and SpMV.
//! - **CSC** (Compressed Sparse Column) — fast column slicing.
//! - **COO** (Coordinate) — simple construction and element-wise ops.
//!
//! Higher-level utilities include:
//!
//! - [`SparseDenseMatmul`] — sparse × dense matrix/vector product.
//! - [`SparseSparseMul`] — element-wise sparse × sparse (union / intersection).
//! - [`SparsityDetector`] — threshold-based sparsity detection on dense tensors.
//! - [`BlockSparse`] — block-sparse format (e.g. 16×16 tiles).
//! - [`PruningMask`] — structured / unstructured pruning mask generation.
//! - [`SparseStats`] — sparsity ratio, NNZ, memory savings, FLOP reduction.
//!
//! All operations have **CPU reference implementations** and compile without any
//! OpenCL runtime. The embedded OpenCL C source ([`SPARSE_MATMUL_CL`]) is
//! provided for future GPU dispatch on Intel / AMD / other OpenCL devices.

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ────────────────────────────────────────────────────

/// OpenCL kernel source for sparse-dense matrix-vector multiply (CSR SpMV).
pub const SPARSE_MATMUL_CL: &str = r#"
__kernel void spmv_csr(
    __global const int   *row_ptrs,   // [rows + 1]
    __global const int   *col_indices, // [nnz]
    __global const float *values,      // [nnz]
    __global const float *x,           // dense input  [cols]
    __global       float *y,           // dense output [rows]
    const int rows)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end   = row_ptrs[row + 1];
    for (int j = start; j < end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
}

__kernel void spmv_csr_blocked(
    __global const int   *row_ptrs,
    __global const int   *col_indices,
    __global const float *values,
    __global const float *x,
    __global       float *y,
    const int rows,
    const int block_size)
{
    int row = get_global_id(0);
    if (row >= rows) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end   = row_ptrs[row + 1];

    // Process in blocks for better cache utilization
    for (int j = start; j < end; j += block_size) {
        int block_end = min(j + block_size, end);
        for (int k = j; k < block_end; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
    }
    y[row] = sum;
}
"#;

// ── Sparse formats ─────────────────────────────────────────────────────────

/// Supported sparse storage formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row — efficient row slicing and SpMV.
    CSR,
    /// Compressed Sparse Column — efficient column slicing.
    CSC,
    /// Coordinate list — simple construction and element-wise ops.
    COO,
}

impl std::fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSR => write!(f, "CSR"),
            Self::CSC => write!(f, "CSC"),
            Self::COO => write!(f, "COO"),
        }
    }
}

// ── SparseMatrix ────────────────────────────────────────────────────────────

/// A generic sparse matrix supporting CSR, CSC, and COO representations.
///
/// Internally stored in COO form (sorted by row, then column). Conversion to
/// CSR / CSC produces the corresponding compressed arrays.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row indices (COO).
    row_indices: Vec<usize>,
    /// Column indices (COO).
    col_indices: Vec<usize>,
    /// Non-zero values.
    values: Vec<f32>,
}

impl SparseMatrix {
    /// Create an empty sparse matrix with the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if either dimension is zero.
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("Matrix dimensions must be non-zero: {rows}×{cols}"),
            }
            .into());
        }
        Ok(Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Insert a value. Duplicates at the same (row, col) are summed on export.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if indices are out of bounds.
    pub fn insert(&mut self, row: usize, col: usize, value: f32) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "Index ({row}, {col}) out of bounds for {r}×{c} matrix",
                    r = self.rows,
                    c = self.cols,
                ),
            }
            .into());
        }
        if value != 0.0 {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
        Ok(())
    }

    /// Build from pre-sorted COO triplets.
    ///
    /// # Errors
    ///
    /// Returns an error if any index is out of bounds or arrays differ in length.
    pub fn from_coo(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("Matrix dimensions must be non-zero: {rows}×{cols}"),
            }
            .into());
        }
        if row_indices.len() != col_indices.len() || col_indices.len() != values.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "COO arrays must have equal length: rows={}, cols={}, vals={}",
                    row_indices.len(),
                    col_indices.len(),
                    values.len(),
                ),
            }
            .into());
        }
        for (i, (&r, &c)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
            if r >= rows || c >= cols {
                return Err(KernelError::InvalidArguments {
                    reason: format!("COO entry {i}: ({r}, {c}) out of bounds for {rows}×{cols}"),
                }
                .into());
            }
        }
        Ok(Self { rows, cols, row_indices, col_indices, values })
    }

    /// Build from a dense row-major matrix, ignoring values whose absolute
    /// value is at or below `threshold`.
    pub fn from_dense(dense: &[f32], rows: usize, cols: usize, threshold: f32) -> Result<Self> {
        if rows * cols != dense.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("Dense length {} != rows*cols = {}", dense.len(), rows * cols),
            }
            .into());
        }
        let mut mat = Self::new(rows, cols)?;
        for r in 0..rows {
            for c in 0..cols {
                let v = dense[r * cols + c];
                if v.abs() > threshold {
                    mat.row_indices.push(r);
                    mat.col_indices.push(c);
                    mat.values.push(v);
                }
            }
        }
        Ok(mat)
    }

    /// Export to dense row-major representation.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        for i in 0..self.nnz() {
            dense[self.row_indices[i] * self.cols + self.col_indices[i]] += self.values[i];
        }
        dense
    }

    // ── COO accessors ───────────────────────────────────────────────────

    /// COO row indices.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// COO column indices.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Non-zero values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    // ── Format conversion ───────────────────────────────────────────────

    /// Convert to CSR arrays: `(row_ptrs, col_indices, values)`.
    pub fn to_csr(&self) -> CsrData {
        let sorted = self.sorted_coo();
        let mut row_ptrs = vec![0usize; self.rows + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut vals = Vec::with_capacity(sorted.len());

        for &(r, c, v) in &sorted {
            row_ptrs[r + 1] += 1;
            col_idx.push(c);
            vals.push(v);
        }
        // Prefix sum.
        for i in 1..=self.rows {
            row_ptrs[i] += row_ptrs[i - 1];
        }
        CsrData { rows: self.rows, cols: self.cols, row_ptrs, col_indices: col_idx, values: vals }
    }

    /// Convert to CSC arrays: `(col_ptrs, row_indices, values)`.
    pub fn to_csc(&self) -> CscData {
        let mut entries: Vec<(usize, usize, f32)> = (0..self.nnz())
            .map(|i| (self.col_indices[i], self.row_indices[i], self.values[i]))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut col_ptrs = vec![0usize; self.cols + 1];
        let mut row_idx = Vec::with_capacity(entries.len());
        let mut vals = Vec::with_capacity(entries.len());

        for &(c, r, v) in &entries {
            col_ptrs[c + 1] += 1;
            row_idx.push(r);
            vals.push(v);
        }
        for i in 1..=self.cols {
            col_ptrs[i] += col_ptrs[i - 1];
        }
        CscData { rows: self.rows, cols: self.cols, col_ptrs, row_indices: row_idx, values: vals }
    }

    /// Reconstruct a `SparseMatrix` from CSR data.
    pub fn from_csr(csr: &CsrData) -> Self {
        let mut row_indices = Vec::with_capacity(csr.values.len());
        let mut col_indices = Vec::with_capacity(csr.values.len());
        let mut values = Vec::with_capacity(csr.values.len());
        for r in 0..csr.rows {
            for j in csr.row_ptrs[r]..csr.row_ptrs[r + 1] {
                row_indices.push(r);
                col_indices.push(csr.col_indices[j]);
                values.push(csr.values[j]);
            }
        }
        Self { rows: csr.rows, cols: csr.cols, row_indices, col_indices, values }
    }

    /// Reconstruct a `SparseMatrix` from CSC data.
    pub fn from_csc(csc: &CscData) -> Self {
        let mut row_indices = Vec::with_capacity(csc.values.len());
        let mut col_indices = Vec::with_capacity(csc.values.len());
        let mut values = Vec::with_capacity(csc.values.len());
        for c in 0..csc.cols {
            for j in csc.col_ptrs[c]..csc.col_ptrs[c + 1] {
                row_indices.push(csc.row_indices[j]);
                col_indices.push(c);
                values.push(csc.values[j]);
            }
        }
        Self { rows: csc.rows, cols: csc.cols, row_indices, col_indices, values }
    }

    /// Return COO entries sorted by (row, col).
    fn sorted_coo(&self) -> Vec<(usize, usize, f32)> {
        let mut entries: Vec<(usize, usize, f32)> = (0..self.nnz())
            .map(|i| (self.row_indices[i], self.col_indices[i], self.values[i]))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        entries
    }
}

// ── CSR / CSC data containers ───────────────────────────────────────────────

/// Compressed Sparse Row storage.
#[derive(Debug, Clone)]
pub struct CsrData {
    pub rows: usize,
    pub cols: usize,
    /// Length = `rows + 1`. `row_ptrs[i]..row_ptrs[i+1]` indexes into
    /// `col_indices` and `values` for row `i`.
    pub row_ptrs: Vec<usize>,
    /// Column index of each non-zero.
    pub col_indices: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f32>,
}

/// Compressed Sparse Column storage.
#[derive(Debug, Clone)]
pub struct CscData {
    pub rows: usize,
    pub cols: usize,
    /// Length = `cols + 1`.
    pub col_ptrs: Vec<usize>,
    /// Row index of each non-zero.
    pub row_indices: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f32>,
}

// ── Sparse-Dense matmul ─────────────────────────────────────────────────────

/// Sparse × dense matrix/vector product.
pub struct SparseDenseMatmul;

impl SparseDenseMatmul {
    /// Compute `y = A * x` where A is sparse (CSR) and x is a dense vector.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are incompatible.
    pub fn spmv_csr(csr: &CsrData, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != csr.cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "SpMV dimension mismatch: A is {}×{} but x has length {}",
                    csr.rows,
                    csr.cols,
                    x.len(),
                ),
            }
            .into());
        }
        if y.len() != csr.rows {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "SpMV output mismatch: expected {} rows, got {}",
                    csr.rows,
                    y.len(),
                ),
            }
            .into());
        }
        for (row, y_val) in y.iter_mut().enumerate().take(csr.rows) {
            let mut sum = 0.0f32;
            for j in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                sum += csr.values[j] * x[csr.col_indices[j]];
            }
            *y_val = sum;
        }
        Ok(())
    }

    /// Compute `C = A * B` where A is sparse (CSR) and B is dense row-major.
    ///
    /// B is `(k × n)`, C is `(m × n)` where A is `(m × k)`.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are incompatible.
    pub fn spmm_csr(csr: &CsrData, b: &[f32], b_cols: usize, c: &mut [f32]) -> Result<()> {
        let k = csr.cols;
        let m = csr.rows;
        if b.len() != k * b_cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "SpMM B dimension mismatch: A is {m}×{k}, B should be {k}×{b_cols} \
                     ({} elems) but got {}",
                    k * b_cols,
                    b.len(),
                ),
            }
            .into());
        }
        if c.len() != m * b_cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "SpMM C dimension mismatch: expected {}×{b_cols} ({} elems) but got {}",
                    m,
                    m * b_cols,
                    c.len(),
                ),
            }
            .into());
        }
        c.fill(0.0);
        for row in 0..m {
            for j in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                let a_val = csr.values[j];
                let a_col = csr.col_indices[j];
                for n in 0..b_cols {
                    c[row * b_cols + n] += a_val * b[a_col * b_cols + n];
                }
            }
        }
        Ok(())
    }

    /// Compute `y = A * x` directly from COO `SparseMatrix`.
    pub fn spmv(mat: &SparseMatrix, x: &[f32], y: &mut [f32]) -> Result<()> {
        let csr = mat.to_csr();
        Self::spmv_csr(&csr, x, y)
    }
}

// ── Sparse-Sparse element-wise multiply ─────────────────────────────────────

/// Mode for element-wise sparse × sparse operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMulMode {
    /// Keep entries present in **both** operands (intersection).
    Intersection,
    /// Keep entries present in **either** operand (union); missing entries
    /// are treated as zero, so union of multiply always equals intersection.
    /// This mode is provided for addition-like ops.
    Union,
}

/// Element-wise sparse × sparse operations.
pub struct SparseSparseMul;

impl SparseSparseMul {
    /// Element-wise multiply of two sparse matrices (intersection semantics).
    ///
    /// Only positions present in **both** matrices contribute to the output;
    /// missing entries are implicitly zero.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions differ.
    pub fn multiply(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
        Self::elementwise(a, b, SparseMulMode::Intersection, |x, y| x * y)
    }

    /// Element-wise addition of two sparse matrices (union semantics).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions differ.
    pub fn add(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
        Self::elementwise(a, b, SparseMulMode::Union, |x, y| x + y)
    }

    /// Generic element-wise operation with configurable mode.
    pub fn elementwise(
        a: &SparseMatrix,
        b: &SparseMatrix,
        mode: SparseMulMode,
        op: impl Fn(f32, f32) -> f32,
    ) -> Result<SparseMatrix> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "Dimension mismatch: {}×{} vs {}×{}",
                    a.rows, a.cols, b.rows, b.cols,
                ),
            }
            .into());
        }
        // Build maps from (row, col) → value for both matrices.
        let map_a = Self::to_map(a);
        let map_b = Self::to_map(b);

        let mut result = SparseMatrix::new(a.rows, a.cols)?;
        match mode {
            SparseMulMode::Intersection => {
                for (&(r, c), &va) in &map_a {
                    if let Some(&vb) = map_b.get(&(r, c)) {
                        let v = op(va, vb);
                        if v != 0.0 {
                            result.row_indices.push(r);
                            result.col_indices.push(c);
                            result.values.push(v);
                        }
                    }
                }
            }
            SparseMulMode::Union => {
                // All keys from A ∪ B.
                let mut all_keys: Vec<(usize, usize)> = map_a.keys().copied().collect();
                for k in map_b.keys() {
                    if !map_a.contains_key(k) {
                        all_keys.push(*k);
                    }
                }
                all_keys.sort();
                for (r, c) in all_keys {
                    let va = map_a.get(&(r, c)).copied().unwrap_or(0.0);
                    let vb = map_b.get(&(r, c)).copied().unwrap_or(0.0);
                    let v = op(va, vb);
                    if v != 0.0 {
                        result.row_indices.push(r);
                        result.col_indices.push(c);
                        result.values.push(v);
                    }
                }
            }
        }
        Ok(result)
    }

    fn to_map(mat: &SparseMatrix) -> std::collections::HashMap<(usize, usize), f32> {
        let mut m = std::collections::HashMap::new();
        for i in 0..mat.nnz() {
            *m.entry((mat.row_indices[i], mat.col_indices[i])).or_insert(0.0) += mat.values[i];
        }
        m
    }
}

// ── Sparsity detector ───────────────────────────────────────────────────────

/// Detect sparsity patterns in dense tensors.
pub struct SparsityDetector;

impl SparsityDetector {
    /// Classify elements as zero / non-zero using an absolute threshold.
    ///
    /// Returns a boolean mask (`true` = non-zero) and the sparsity ratio.
    pub fn detect(data: &[f32], threshold: f32) -> (Vec<bool>, f64) {
        let mask: Vec<bool> = data.iter().map(|&v| v.abs() > threshold).collect();
        let nnz = mask.iter().filter(|&&m| m).count();
        let sparsity = if data.is_empty() { 0.0 } else { 1.0 - (nnz as f64 / data.len() as f64) };
        (mask, sparsity)
    }

    /// Convert a dense row-major matrix to [`SparseMatrix`] using a threshold.
    pub fn to_sparse(
        data: &[f32],
        rows: usize,
        cols: usize,
        threshold: f32,
    ) -> Result<SparseMatrix> {
        SparseMatrix::from_dense(data, rows, cols, threshold)
    }

    /// Detect per-row sparsity ratios.
    pub fn row_sparsity(data: &[f32], rows: usize, cols: usize, threshold: f32) -> Vec<f64> {
        (0..rows)
            .map(|r| {
                let row = &data[r * cols..(r + 1) * cols];
                let nnz = row.iter().filter(|&&v| v.abs() > threshold).count();
                1.0 - (nnz as f64 / cols as f64)
            })
            .collect()
    }

    /// Detect per-column sparsity ratios.
    pub fn col_sparsity(data: &[f32], rows: usize, cols: usize, threshold: f32) -> Vec<f64> {
        (0..cols)
            .map(|c| {
                let nnz = (0..rows).filter(|&r| data[r * cols + c].abs() > threshold).count();
                1.0 - (nnz as f64 / rows as f64)
            })
            .collect()
    }
}

// ── Block-sparse format ─────────────────────────────────────────────────────

/// Block-sparse matrix: stores dense sub-blocks of size `block_size × block_size`
/// only for blocks that contain at least one non-zero value.
#[derive(Debug, Clone)]
pub struct BlockSparse {
    /// Number of rows in the full matrix.
    pub rows: usize,
    /// Number of columns in the full matrix.
    pub cols: usize,
    /// Block size (square blocks).
    pub block_size: usize,
    /// Block-row indices of stored blocks.
    block_rows: Vec<usize>,
    /// Block-column indices of stored blocks.
    block_cols: Vec<usize>,
    /// Flattened dense block data (each block is `block_size * block_size`
    /// elements, stored row-major).
    block_data: Vec<f32>,
}

impl BlockSparse {
    /// Create a block-sparse matrix from a dense row-major array.
    ///
    /// Blocks whose Frobenius norm is at or below `threshold` are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn from_dense(
        data: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        threshold: f32,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 || block_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("Dimensions must be non-zero: {rows}×{cols}, block={block_size}"),
            }
            .into());
        }
        if data.len() != rows * cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("Dense length {} != {rows}×{cols} = {}", data.len(), rows * cols),
            }
            .into());
        }

        let br_count = rows.div_ceil(block_size);
        let bc_count = cols.div_ceil(block_size);
        let bs2 = block_size * block_size;

        let mut block_rows = Vec::new();
        let mut block_cols = Vec::new();
        let mut block_data = Vec::new();

        for br in 0..br_count {
            for bc in 0..bc_count {
                let mut block = vec![0.0f32; bs2];
                let mut norm_sq = 0.0f32;
                for lr in 0..block_size {
                    let gr = br * block_size + lr;
                    if gr >= rows {
                        continue;
                    }
                    for lc in 0..block_size {
                        let gc = bc * block_size + lc;
                        if gc >= cols {
                            continue;
                        }
                        let v = data[gr * cols + gc];
                        block[lr * block_size + lc] = v;
                        norm_sq += v * v;
                    }
                }
                if norm_sq.sqrt() > threshold {
                    block_rows.push(br);
                    block_cols.push(bc);
                    block_data.extend_from_slice(&block);
                }
            }
        }

        Ok(Self { rows, cols, block_size, block_rows, block_cols, block_data })
    }

    /// Number of stored blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_rows.len()
    }

    /// Total number of possible blocks if the matrix were fully dense.
    pub fn total_blocks(&self) -> usize {
        let br = self.rows.div_ceil(self.block_size);
        let bc = self.cols.div_ceil(self.block_size);
        br * bc
    }

    /// Block-level sparsity ratio (fraction of zero blocks).
    pub fn block_sparsity(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.num_blocks() as f64 / total as f64)
    }

    /// Convert back to dense row-major array.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        let bs = self.block_size;
        let bs2 = bs * bs;
        for (idx, (&br, &bc)) in self.block_rows.iter().zip(self.block_cols.iter()).enumerate() {
            let block = &self.block_data[idx * bs2..(idx + 1) * bs2];
            for lr in 0..bs {
                let gr = br * bs + lr;
                if gr >= self.rows {
                    continue;
                }
                for lc in 0..bs {
                    let gc = bc * bs + lc;
                    if gc >= self.cols {
                        continue;
                    }
                    dense[gr * self.cols + gc] = block[lr * bs + lc];
                }
            }
        }
        dense
    }

    /// Multiply this block-sparse matrix (A) by a dense vector x, writing to y.
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != self.cols {
            return Err(KernelError::InvalidArguments {
                reason: format!("BlockSparse SpMV: x.len()={} != cols={}", x.len(), self.cols),
            }
            .into());
        }
        if y.len() != self.rows {
            return Err(KernelError::InvalidArguments {
                reason: format!("BlockSparse SpMV: y.len()={} != rows={}", y.len(), self.rows),
            }
            .into());
        }
        y.fill(0.0);
        let bs = self.block_size;
        let bs2 = bs * bs;
        for (idx, (&br, &bc)) in self.block_rows.iter().zip(self.block_cols.iter()).enumerate() {
            let block = &self.block_data[idx * bs2..(idx + 1) * bs2];
            for lr in 0..bs {
                let gr = br * bs + lr;
                if gr >= self.rows {
                    continue;
                }
                let mut sum = 0.0f32;
                for lc in 0..bs {
                    let gc = bc * bs + lc;
                    if gc >= self.cols {
                        continue;
                    }
                    sum += block[lr * bs + lc] * x[gc];
                }
                y[gr] += sum;
            }
        }
        Ok(())
    }

    /// Access block row/col indices.
    pub fn block_positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.block_rows.iter().copied().zip(self.block_cols.iter().copied())
    }

    /// Access flattened block data.
    pub fn block_data(&self) -> &[f32] {
        &self.block_data
    }
}

// ── Pruning mask ────────────────────────────────────────────────────────────

/// Pruning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningStrategy {
    /// Remove individual weights below a magnitude threshold (unstructured).
    Unstructured,
    /// Remove entire rows whose L2 norm is below a threshold.
    RowStructured,
    /// Remove entire columns whose L2 norm is below a threshold.
    ColumnStructured,
    /// Remove entire blocks whose Frobenius norm is below a threshold.
    BlockStructured { block_size: usize },
}

/// Boolean mask indicating which elements to keep (`true`) or prune (`false`).
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Row-major boolean mask. Same shape as the original weight matrix.
    mask: Vec<bool>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Strategy used to generate this mask.
    pub strategy: PruningStrategy,
}

impl PruningMask {
    /// Generate a pruning mask for a dense row-major weight matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match the data length.
    pub fn generate(
        data: &[f32],
        rows: usize,
        cols: usize,
        threshold: f32,
        strategy: PruningStrategy,
    ) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "Mask data length {} != {rows}×{cols} = {}",
                    data.len(),
                    rows * cols,
                ),
            }
            .into());
        }
        let mask = match strategy {
            PruningStrategy::Unstructured => data.iter().map(|&v| v.abs() > threshold).collect(),
            PruningStrategy::RowStructured => {
                let mut mask = vec![false; rows * cols];
                for r in 0..rows {
                    let row = &data[r * cols..(r + 1) * cols];
                    let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm > threshold {
                        for c in 0..cols {
                            mask[r * cols + c] = true;
                        }
                    }
                }
                mask
            }
            PruningStrategy::ColumnStructured => {
                let mut mask = vec![false; rows * cols];
                for c in 0..cols {
                    let norm: f32 = (0..rows)
                        .map(|r| {
                            let v = data[r * cols + c];
                            v * v
                        })
                        .sum::<f32>()
                        .sqrt();
                    if norm > threshold {
                        for r in 0..rows {
                            mask[r * cols + c] = true;
                        }
                    }
                }
                mask
            }
            PruningStrategy::BlockStructured { block_size } => {
                let mut mask = vec![false; rows * cols];
                let br_count = rows.div_ceil(block_size);
                let bc_count = cols.div_ceil(block_size);
                for br in 0..br_count {
                    for bc in 0..bc_count {
                        let mut norm_sq = 0.0f32;
                        for lr in 0..block_size {
                            let gr = br * block_size + lr;
                            if gr >= rows {
                                continue;
                            }
                            for lc in 0..block_size {
                                let gc = bc * block_size + lc;
                                if gc >= cols {
                                    continue;
                                }
                                let v = data[gr * cols + gc];
                                norm_sq += v * v;
                            }
                        }
                        if norm_sq.sqrt() > threshold {
                            for lr in 0..block_size {
                                let gr = br * block_size + lr;
                                if gr >= rows {
                                    continue;
                                }
                                for lc in 0..block_size {
                                    let gc = bc * block_size + lc;
                                    if gc >= cols {
                                        continue;
                                    }
                                    mask[gr * cols + gc] = true;
                                }
                            }
                        }
                    }
                }
                mask
            }
        };
        Ok(Self { mask, rows, cols, strategy })
    }

    /// Apply the mask to a dense matrix, zeroing out pruned positions.
    pub fn apply(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.mask.len(), "mask/data length mismatch");
        for (v, &keep) in data.iter_mut().zip(self.mask.iter()) {
            if !keep {
                *v = 0.0;
            }
        }
    }

    /// Return the raw boolean mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Number of entries kept.
    pub fn kept(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }

    /// Number of entries pruned.
    pub fn pruned(&self) -> usize {
        self.mask.len() - self.kept()
    }

    /// Sparsity ratio (fraction pruned).
    pub fn sparsity(&self) -> f64 {
        if self.mask.is_empty() {
            return 0.0;
        }
        self.pruned() as f64 / self.mask.len() as f64
    }
}

// ── Sparse statistics ───────────────────────────────────────────────────────

/// Statistics about a sparse representation.
#[derive(Debug, Clone)]
pub struct SparseStats {
    /// Total number of elements in the full dense matrix.
    pub total_elements: usize,
    /// Number of stored non-zero elements.
    pub nnz: usize,
    /// Sparsity ratio (fraction of zeros), in `[0, 1]`.
    pub sparsity_ratio: f64,
    /// Estimated memory in bytes for the dense representation (`total * 4`).
    pub dense_bytes: usize,
    /// Estimated memory in bytes for the sparse (CSR) representation.
    pub sparse_bytes: usize,
    /// Memory savings factor (`dense_bytes / sparse_bytes`).
    pub memory_savings: f64,
    /// FLOP reduction factor (`total_elements / nnz`), i.e. speed-up upper bound.
    pub flop_reduction: f64,
}

impl SparseStats {
    /// Compute statistics for a sparse matrix.
    pub fn from_sparse(mat: &SparseMatrix) -> Self {
        let total = mat.rows * mat.cols;
        let nnz = mat.nnz();
        Self::compute(total, nnz, mat.rows)
    }

    /// Compute statistics from a CSR representation.
    pub fn from_csr(csr: &CsrData) -> Self {
        let total = csr.rows * csr.cols;
        let nnz = csr.values.len();
        Self::compute(total, nnz, csr.rows)
    }

    /// Compute statistics from a block-sparse representation.
    pub fn from_block_sparse(bs: &BlockSparse) -> Self {
        let total = bs.rows * bs.cols;
        let stored = bs.num_blocks() * bs.block_size * bs.block_size;
        // For block-sparse, stored elements include zero padding within blocks.
        let nnz = stored.min(total);
        let dense_bytes = total * 4;
        // Block-sparse overhead: 2 * num_blocks indices + block data.
        let sparse_bytes = bs.num_blocks() * 2 * std::mem::size_of::<usize>()
            + stored * std::mem::size_of::<f32>();
        let sparsity_ratio = if total == 0 { 0.0 } else { 1.0 - (nnz as f64 / total as f64) };
        let memory_savings =
            if sparse_bytes == 0 { 0.0 } else { dense_bytes as f64 / sparse_bytes as f64 };
        let flop_reduction = if nnz == 0 { 0.0 } else { total as f64 / nnz as f64 };
        Self {
            total_elements: total,
            nnz,
            sparsity_ratio,
            dense_bytes,
            sparse_bytes,
            memory_savings,
            flop_reduction,
        }
    }

    /// Compute statistics from a dense tensor with threshold-based sparsity.
    pub fn from_dense(data: &[f32], rows: usize, cols: usize, threshold: f32) -> Self {
        let total = rows * cols;
        let nnz = data.iter().filter(|&&v| v.abs() > threshold).count();
        Self::compute(total, nnz, rows)
    }

    fn compute(total: usize, nnz: usize, rows: usize) -> Self {
        let sparsity_ratio = if total == 0 { 0.0 } else { 1.0 - (nnz as f64 / total as f64) };
        let dense_bytes = total * std::mem::size_of::<f32>();
        // CSR: row_ptrs (rows+1)*8 + col_indices nnz*8 + values nnz*4
        let sparse_bytes = (rows + 1) * std::mem::size_of::<usize>()
            + nnz * std::mem::size_of::<usize>()
            + nnz * std::mem::size_of::<f32>();
        let memory_savings =
            if sparse_bytes == 0 { 0.0 } else { dense_bytes as f64 / sparse_bytes as f64 };
        let flop_reduction = if nnz == 0 { 0.0 } else { total as f64 / nnz as f64 };
        Self {
            total_elements: total,
            nnz,
            sparsity_ratio,
            dense_bytes,
            sparse_bytes,
            memory_savings,
            flop_reduction,
        }
    }
}

impl std::fmt::Display for SparseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NNZ: {}/{} ({:.1}% sparse) | mem: {:.1}× savings | FLOP: {:.1}× reduction",
            self.nnz,
            self.total_elements,
            self.sparsity_ratio * 100.0,
            self.memory_savings,
            self.flop_reduction,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approximate f32 equality.
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn assert_dense_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "lengths differ: {} vs {}", a.len(), b.len());
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(va, vb, eps), "mismatch at index {i}: {va} vs {vb} (eps={eps})");
        }
    }

    // Dense matmul reference: C = A * B (row-major).
    fn dense_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // Dense matvec reference: y = A * x.
    fn dense_matvec(a: &[f32], x: &[f32], m: usize, k: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..k {
                y[i] += a[i * k + j] * x[j];
            }
        }
        y
    }

    // ── SparseFormat Display ────────────────────────────────────────────

    #[test]
    fn test_sparse_format_display() {
        assert_eq!(format!("{}", SparseFormat::CSR), "CSR");
        assert_eq!(format!("{}", SparseFormat::CSC), "CSC");
        assert_eq!(format!("{}", SparseFormat::COO), "COO");
    }

    #[test]
    fn test_sparse_format_eq() {
        assert_eq!(SparseFormat::CSR, SparseFormat::CSR);
        assert_ne!(SparseFormat::CSR, SparseFormat::COO);
    }

    // ── SparseMatrix construction ───────────────────────────────────────

    #[test]
    fn test_new_empty_matrix() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 4);
        assert_eq!(mat.nnz(), 0);
    }

    #[test]
    fn test_new_zero_rows_errors() {
        assert!(SparseMatrix::new(0, 4).is_err());
    }

    #[test]
    fn test_new_zero_cols_errors() {
        assert!(SparseMatrix::new(3, 0).is_err());
    }

    #[test]
    fn test_insert_and_nnz() {
        let mut mat = SparseMatrix::new(3, 3).unwrap();
        mat.insert(0, 0, 1.0).unwrap();
        mat.insert(1, 2, 2.0).unwrap();
        mat.insert(2, 1, 3.0).unwrap();
        assert_eq!(mat.nnz(), 3);
    }

    #[test]
    fn test_insert_zero_is_ignored() {
        let mut mat = SparseMatrix::new(3, 3).unwrap();
        mat.insert(0, 0, 0.0).unwrap();
        assert_eq!(mat.nnz(), 0);
    }

    #[test]
    fn test_insert_out_of_bounds_row() {
        let mut mat = SparseMatrix::new(3, 3).unwrap();
        assert!(mat.insert(3, 0, 1.0).is_err());
    }

    #[test]
    fn test_insert_out_of_bounds_col() {
        let mut mat = SparseMatrix::new(3, 3).unwrap();
        assert!(mat.insert(0, 3, 1.0).is_err());
    }

    // ── from_coo / to_dense roundtrip ───────────────────────────────────

    #[test]
    fn test_from_coo_basic() {
        let mat = SparseMatrix::from_coo(2, 3, vec![0, 1], vec![1, 2], vec![5.0, 7.0]).unwrap();
        assert_eq!(mat.nnz(), 2);
        let dense = mat.to_dense();
        assert_eq!(dense, vec![0.0, 5.0, 0.0, 0.0, 0.0, 7.0]);
    }

    #[test]
    fn test_from_coo_mismatched_lengths() {
        assert!(SparseMatrix::from_coo(2, 2, vec![0], vec![], vec![1.0]).is_err());
    }

    #[test]
    fn test_from_coo_out_of_bounds() {
        assert!(SparseMatrix::from_coo(2, 2, vec![2], vec![0], vec![1.0]).is_err());
    }

    #[test]
    fn test_from_coo_zero_dim() {
        assert!(SparseMatrix::from_coo(0, 3, vec![], vec![], vec![]).is_err());
    }

    // ── from_dense / to_dense roundtrip ─────────────────────────────────

    #[test]
    fn test_from_dense_roundtrip() {
        let dense = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        assert_eq!(mat.nnz(), 4);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_from_dense_with_threshold() {
        let dense = vec![0.5, 0.01, 0.0, 0.3, -0.02, 0.9, 0.0, 0.0, -0.7];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.1).unwrap();
        // Only values with |v| > 0.1 are kept: 0.5, 0.3, 0.9, -0.7
        assert_eq!(mat.nnz(), 4);
    }

    #[test]
    fn test_from_dense_wrong_length() {
        assert!(SparseMatrix::from_dense(&[1.0, 2.0], 3, 3, 0.0).is_err());
    }

    #[test]
    fn test_fully_dense_roundtrip() {
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mat = SparseMatrix::from_dense(&dense, 2, 2, 0.0).unwrap();
        assert_eq!(mat.nnz(), 4);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_all_zeros_sparse() {
        let dense = vec![0.0; 9];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        assert_eq!(mat.nnz(), 0);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_single_nonzero() {
        let mut dense = vec![0.0; 16];
        dense[7] = 42.0;
        let mat = SparseMatrix::from_dense(&dense, 4, 4, 0.0).unwrap();
        assert_eq!(mat.nnz(), 1);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    // ── CSR conversion ──────────────────────────────────────────────────

    #[test]
    fn test_to_csr_basic() {
        let mat = SparseMatrix::from_coo(
            3,
            3,
            vec![0, 0, 1, 2],
            vec![0, 2, 1, 0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let csr = mat.to_csr();
        assert_eq!(csr.row_ptrs, vec![0, 2, 3, 4]);
        assert_eq!(csr.col_indices, vec![0, 2, 1, 0]);
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_csr_empty_rows() {
        // Row 1 is empty.
        let mat = SparseMatrix::from_coo(3, 3, vec![0, 2], vec![0, 2], vec![1.0, 2.0]).unwrap();
        let csr = mat.to_csr();
        assert_eq!(csr.row_ptrs, vec![0, 1, 1, 2]);
    }

    #[test]
    fn test_csr_roundtrip() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        let csr = mat.to_csr();
        let mat2 = SparseMatrix::from_csr(&csr);
        assert_dense_eq(&mat2.to_dense(), &dense, 1e-7);
    }

    // ── CSC conversion ──────────────────────────────────────────────────

    #[test]
    fn test_to_csc_basic() {
        let mat = SparseMatrix::from_coo(3, 3, vec![0, 1, 2], vec![0, 0, 2], vec![1.0, 2.0, 3.0])
            .unwrap();
        let csc = mat.to_csc();
        assert_eq!(csc.col_ptrs, vec![0, 2, 2, 3]);
        assert_eq!(csc.row_indices, vec![0, 1, 2]);
        assert_eq!(csc.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csc_roundtrip() {
        let dense = vec![0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        let csc = mat.to_csc();
        let mat2 = SparseMatrix::from_csc(&csc);
        assert_dense_eq(&mat2.to_dense(), &dense, 1e-7);
    }

    // ── CSR ↔ CSC ↔ COO roundtrip ──────────────────────────────────────

    #[test]
    fn test_csr_csc_coo_roundtrip() {
        let dense = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let original = SparseMatrix::from_dense(&dense, 3, 4, 0.0).unwrap();

        // COO → CSR → COO → dense
        let csr = original.to_csr();
        let from_csr = SparseMatrix::from_csr(&csr);
        assert_dense_eq(&from_csr.to_dense(), &dense, 1e-7);

        // COO → CSC → COO → dense
        let csc = original.to_csc();
        let from_csc = SparseMatrix::from_csc(&csc);
        assert_dense_eq(&from_csc.to_dense(), &dense, 1e-7);

        // CSR → COO → CSC → COO → dense
        let back_to_csc = from_csr.to_csc();
        let back_from_csc = SparseMatrix::from_csc(&back_to_csc);
        assert_dense_eq(&back_from_csc.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_identity_csr_csc_roundtrip() {
        // Identity matrix.
        let n = 5;
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            dense[i * n + i] = 1.0;
        }
        let mat = SparseMatrix::from_dense(&dense, n, n, 0.0).unwrap();
        assert_eq!(mat.nnz(), n);

        let csr = mat.to_csr();
        assert_dense_eq(&SparseMatrix::from_csr(&csr).to_dense(), &dense, 1e-7);

        let csc = mat.to_csc();
        assert_dense_eq(&SparseMatrix::from_csc(&csc).to_dense(), &dense, 1e-7);
    }

    // ── SpMV (sparse × dense vector) ────────────────────────────────────

    #[test]
    fn test_spmv_csr_basic() {
        // A = [[1, 0, 2],
        //      [0, 3, 0],
        //      [4, 0, 5]]
        let dense_a = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let mat = SparseMatrix::from_dense(&dense_a, 3, 3, 0.0).unwrap();
        let csr = mat.to_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        let expected = dense_matvec(&dense_a, &x, 3, 3);
        assert_dense_eq(&y, &expected, 1e-6);
    }

    #[test]
    fn test_spmv_matches_dense() {
        let dense_a = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 0.0, 0.0, 6.0];
        let x = vec![1.0, -1.0, 2.0, 0.5];
        let expected = dense_matvec(&dense_a, &x, 3, 4);

        let mat = SparseMatrix::from_dense(&dense_a, 3, 4, 0.0).unwrap();
        let csr = mat.to_csr();
        let mut y = vec![0.0; 3];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        assert_dense_eq(&y, &expected, 1e-6);
    }

    #[test]
    fn test_spmv_empty_matrix() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        let csr = mat.to_csr();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![99.0; 3];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        assert_dense_eq(&y, &[0.0, 0.0, 0.0], 1e-7);
    }

    #[test]
    fn test_spmv_dimension_mismatch_x() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        let csr = mat.to_csr();
        let x = vec![1.0, 2.0]; // wrong length
        let mut y = vec![0.0; 3];
        assert!(SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).is_err());
    }

    #[test]
    fn test_spmv_dimension_mismatch_y() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        let csr = mat.to_csr();
        let x = vec![1.0; 4];
        let mut y = vec![0.0; 2]; // wrong length
        assert!(SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).is_err());
    }

    #[test]
    fn test_spmv_via_sparse_matrix() {
        let dense_a = vec![1.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        let mat = SparseMatrix::from_dense(&dense_a, 2, 3, 0.0).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];
        SparseDenseMatmul::spmv(&mat, &x, &mut y).unwrap();
        let expected = dense_matvec(&dense_a, &x, 2, 3);
        assert_dense_eq(&y, &expected, 1e-6);
    }

    // ── SpMM (sparse × dense matrix) ───────────────────────────────────

    #[test]
    fn test_spmm_csr_basic() {
        // A(3×3), B(3×2)
        let dense_a = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = SparseMatrix::from_dense(&dense_a, 3, 3, 0.0).unwrap();
        let csr = mat.to_csr();
        let mut c = vec![0.0; 6];
        SparseDenseMatmul::spmm_csr(&csr, &b, 2, &mut c).unwrap();
        let expected = dense_matmul(&dense_a, &b, 3, 3, 2);
        assert_dense_eq(&c, &expected, 1e-6);
    }

    #[test]
    fn test_spmm_dimension_mismatch() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        let csr = mat.to_csr();
        let b = vec![1.0; 6]; // 4×2 would be 8 elements
        let mut c = vec![0.0; 6];
        assert!(SparseDenseMatmul::spmm_csr(&csr, &b, 2, &mut c).is_err());
    }

    #[test]
    fn test_spmm_output_dimension_mismatch() {
        let mat = SparseMatrix::new(3, 4).unwrap();
        let csr = mat.to_csr();
        let b = vec![1.0; 8]; // 4×2
        let mut c = vec![0.0; 4]; // wrong: should be 3×2=6
        assert!(SparseDenseMatmul::spmm_csr(&csr, &b, 2, &mut c).is_err());
    }

    // ── Sparse × Sparse element-wise ────────────────────────────────────

    #[test]
    fn test_sparse_mul_intersection() {
        let a = SparseMatrix::from_coo(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![2.0, 3.0, 4.0])
            .unwrap();
        let b = SparseMatrix::from_coo(3, 3, vec![0, 1, 2], vec![0, 1, 0], vec![5.0, 6.0, 7.0])
            .unwrap();
        let result = SparseSparseMul::multiply(&a, &b).unwrap();
        // Intersection at (0,0) and (1,1). (2,2) in A, (2,0) in B — no overlap at (2,*).
        let dense = result.to_dense();
        assert!(approx_eq(dense[0], 10.0, 1e-6)); // 2*5
        assert!(approx_eq(dense[4], 18.0, 1e-6)); // 3*6
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_sparse_add_union() {
        let a = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![1, 0], vec![3.0, 4.0]).unwrap();
        let result = SparseSparseMul::add(&a, &b).unwrap();
        let dense = result.to_dense();
        assert_dense_eq(&dense, &[1.0, 3.0, 4.0, 2.0], 1e-6);
    }

    #[test]
    fn test_sparse_mul_dimension_mismatch() {
        let a = SparseMatrix::new(2, 3).unwrap();
        let b = SparseMatrix::new(3, 2).unwrap();
        assert!(SparseSparseMul::multiply(&a, &b).is_err());
    }

    #[test]
    fn test_sparse_mul_empty_matrices() {
        let a = SparseMatrix::new(3, 3).unwrap();
        let b = SparseMatrix::new(3, 3).unwrap();
        let result = SparseSparseMul::multiply(&a, &b).unwrap();
        assert_eq!(result.nnz(), 0);
    }

    #[test]
    fn test_sparse_add_overlapping() {
        let a = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![10.0, 20.0]).unwrap();
        let result = SparseSparseMul::add(&a, &b).unwrap();
        let dense = result.to_dense();
        assert_dense_eq(&dense, &[11.0, 0.0, 0.0, 22.0], 1e-6);
    }

    // ── Sparsity detector ───────────────────────────────────────────────

    #[test]
    fn test_detect_all_zeros() {
        let data = vec![0.0; 10];
        let (mask, ratio) = SparsityDetector::detect(&data, 0.0);
        assert!(mask.iter().all(|&m| !m));
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_detect_all_nonzero() {
        let data = vec![1.0; 10];
        let (mask, ratio) = SparsityDetector::detect(&data, 0.0);
        assert!(mask.iter().all(|&m| m));
        assert!(ratio.abs() < 1e-9);
    }

    #[test]
    fn test_detect_half_sparse() {
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let (mask, ratio) = SparsityDetector::detect(&data, 0.0);
        assert_eq!(mask, vec![true, false, true, false, true, false]);
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_detect_with_threshold() {
        let data = vec![0.5, 0.01, 1.0, 0.05, 2.0];
        let (mask, _ratio) = SparsityDetector::detect(&data, 0.1);
        assert_eq!(mask, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_detect_empty() {
        let data: Vec<f32> = vec![];
        let (mask, ratio) = SparsityDetector::detect(&data, 0.0);
        assert!(mask.is_empty());
        assert!(ratio.abs() < 1e-9);
    }

    #[test]
    fn test_detect_negative_values() {
        let data = vec![-1.0, 0.0, -0.5, 0.0];
        let (mask, _) = SparsityDetector::detect(&data, 0.0);
        assert_eq!(mask, vec![true, false, true, false]);
    }

    #[test]
    fn test_to_sparse_via_detector() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let mat = SparsityDetector::to_sparse(&data, 2, 2, 0.0).unwrap();
        assert_eq!(mat.nnz(), 2);
        assert_dense_eq(&mat.to_dense(), &data, 1e-7);
    }

    #[test]
    fn test_row_sparsity() {
        // Row 0: all non-zero, Row 1: all zero.
        let data = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let ratios = SparsityDetector::row_sparsity(&data, 2, 3, 0.0);
        assert!((ratios[0] - 0.0).abs() < 1e-9);
        assert!((ratios[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_col_sparsity() {
        // Col 0: all non-zero, Col 1: all zero.
        let data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let ratios = SparsityDetector::col_sparsity(&data, 3, 2, 0.0);
        assert!((ratios[0] - 0.0).abs() < 1e-9);
        assert!((ratios[1] - 1.0).abs() < 1e-9);
    }

    // ── Block-sparse ────────────────────────────────────────────────────

    #[test]
    fn test_block_sparse_from_dense_roundtrip() {
        let dense =
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
        let bs = BlockSparse::from_dense(&dense, 4, 4, 2, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 2); // top-left and bottom-right 2×2 blocks
        assert_dense_eq(&bs.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_block_sparse_all_zero() {
        let dense = vec![0.0; 16];
        let bs = BlockSparse::from_dense(&dense, 4, 4, 2, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 0);
        assert_dense_eq(&bs.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_block_sparse_fully_dense() {
        let dense: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let bs = BlockSparse::from_dense(&dense, 4, 4, 2, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 4);
        assert_dense_eq(&bs.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_block_sparse_non_aligned() {
        // 3×3 matrix with block_size=2 → 2×2 grid of blocks (last block padded).
        let dense = vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let bs = BlockSparse::from_dense(&dense, 3, 3, 2, 0.0).unwrap();
        assert_dense_eq(&bs.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_block_sparsity_ratio() {
        let dense =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bs = BlockSparse::from_dense(&dense, 4, 4, 2, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 1);
        assert_eq!(bs.total_blocks(), 4);
        assert!((bs.block_sparsity() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_block_sparse_spmv() {
        let dense_a =
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
        let bs = BlockSparse::from_dense(&dense_a, 4, 4, 2, 0.0).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        bs.spmv(&x, &mut y).unwrap();
        let expected = dense_matvec(&dense_a, &x, 4, 4);
        assert_dense_eq(&y, &expected, 1e-6);
    }

    #[test]
    fn test_block_sparse_spmv_dim_mismatch() {
        let dense = vec![1.0; 4];
        let bs = BlockSparse::from_dense(&dense, 2, 2, 2, 0.0).unwrap();
        let x = vec![1.0]; // wrong
        let mut y = vec![0.0; 2];
        assert!(bs.spmv(&x, &mut y).is_err());
    }

    #[test]
    fn test_block_sparse_spmv_y_dim_mismatch() {
        let dense = vec![1.0; 4];
        let bs = BlockSparse::from_dense(&dense, 2, 2, 2, 0.0).unwrap();
        let x = vec![1.0; 2];
        let mut y = vec![0.0; 1]; // wrong
        assert!(bs.spmv(&x, &mut y).is_err());
    }

    #[test]
    fn test_block_sparse_invalid_dims() {
        assert!(BlockSparse::from_dense(&[], 0, 4, 2, 0.0).is_err());
        assert!(BlockSparse::from_dense(&[1.0], 1, 1, 0, 0.0).is_err());
    }

    #[test]
    fn test_block_sparse_wrong_data_length() {
        assert!(BlockSparse::from_dense(&[1.0, 2.0], 3, 3, 2, 0.0).is_err());
    }

    #[test]
    fn test_block_sparse_positions_and_data() {
        let dense = vec![1.0, 0.0, 0.0, 2.0];
        let bs = BlockSparse::from_dense(&dense, 2, 2, 1, 0.0).unwrap();
        let positions: Vec<(usize, usize)> = bs.block_positions().collect();
        assert_eq!(positions.len(), 2);
        assert!(!bs.block_data().is_empty());
    }

    #[test]
    fn test_block_sparse_16x16() {
        // Realistic block size.
        let n = 32;
        let mut dense = vec![0.0f32; n * n];
        // Fill top-left 16×16 block only.
        for r in 0..16 {
            for c in 0..16 {
                dense[r * n + c] = (r * n + c + 1) as f32;
            }
        }
        let bs = BlockSparse::from_dense(&dense, n, n, 16, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 1);
        assert_dense_eq(&bs.to_dense(), &dense, 1e-7);
    }

    // ── Pruning mask ────────────────────────────────────────────────────

    #[test]
    fn test_pruning_unstructured() {
        let data = vec![0.5, 0.01, 0.9, 0.02, 0.8, 0.03];
        let mask = PruningMask::generate(&data, 2, 3, 0.1, PruningStrategy::Unstructured).unwrap();
        assert_eq!(mask.mask(), &[true, false, true, false, true, false]);
        assert_eq!(mask.kept(), 3);
        assert_eq!(mask.pruned(), 3);
        assert!((mask.sparsity() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pruning_apply() {
        let data_orig = vec![0.5, 0.01, 0.9, 0.02];
        let mask =
            PruningMask::generate(&data_orig, 2, 2, 0.1, PruningStrategy::Unstructured).unwrap();
        let mut data = data_orig.clone();
        mask.apply(&mut data);
        assert_dense_eq(&data, &[0.5, 0.0, 0.9, 0.0], 1e-7);
    }

    #[test]
    fn test_pruning_row_structured() {
        // Row 0 norm = sqrt(0.01+0.01) ≈ 0.14, Row 1 norm = sqrt(1+4) ≈ 2.24.
        let data = vec![0.1, 0.1, 1.0, 2.0];
        let mask = PruningMask::generate(&data, 2, 2, 0.5, PruningStrategy::RowStructured).unwrap();
        // Row 0 pruned, row 1 kept.
        assert_eq!(mask.mask(), &[false, false, true, true]);
    }

    #[test]
    fn test_pruning_column_structured() {
        // Col 0: [0.1, 0.1] norm ≈ 0.14. Col 1: [1.0, 2.0] norm ≈ 2.24.
        let data = vec![0.1, 1.0, 0.1, 2.0];
        let mask =
            PruningMask::generate(&data, 2, 2, 0.5, PruningStrategy::ColumnStructured).unwrap();
        assert_eq!(mask.mask(), &[false, true, false, true]);
    }

    #[test]
    fn test_pruning_block_structured() {
        let data = vec![
            0.01, 0.01, 1.0, 2.0, 0.01, 0.01, 3.0, 4.0, 5.0, 6.0, 0.01, 0.01, 7.0, 8.0, 0.01, 0.01,
        ];
        let mask = PruningMask::generate(
            &data,
            4,
            4,
            0.1,
            PruningStrategy::BlockStructured { block_size: 2 },
        )
        .unwrap();
        // Top-left 2×2 block norm ≈ 0.02, pruned.
        // Top-right 2×2 norm ≈ 5.48, kept.
        // Bottom-left norm ≈ 13.2, kept.
        // Bottom-right norm ≈ 0.02, pruned.
        assert_eq!(
            mask.mask(),
            &[
                false, false, true, true, false, false, true, true, true, true, false, false, true,
                true, false, false,
            ]
        );
    }

    #[test]
    fn test_pruning_wrong_data_length() {
        assert!(
            PruningMask::generate(&[1.0, 2.0], 3, 3, 0.1, PruningStrategy::Unstructured).is_err()
        );
    }

    #[test]
    fn test_pruning_all_kept() {
        let data = vec![1.0; 4];
        let mask = PruningMask::generate(&data, 2, 2, 0.0, PruningStrategy::Unstructured).unwrap();
        assert_eq!(mask.kept(), 4);
        assert_eq!(mask.pruned(), 0);
        assert!(mask.sparsity().abs() < 1e-9);
    }

    #[test]
    fn test_pruning_all_pruned() {
        let data = vec![0.0; 4];
        let mask = PruningMask::generate(&data, 2, 2, 0.0, PruningStrategy::Unstructured).unwrap();
        assert_eq!(mask.kept(), 0);
        assert_eq!(mask.pruned(), 4);
        assert!((mask.sparsity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pruning_empty_mask_sparsity() {
        // Edge: generate mask on a 1x1 with zero value.
        let data = vec![0.0];
        let mask = PruningMask::generate(&data, 1, 1, 0.0, PruningStrategy::Unstructured).unwrap();
        assert!((mask.sparsity() - 1.0).abs() < 1e-9);
    }

    // ── Sparse statistics ───────────────────────────────────────────────

    #[test]
    fn test_stats_from_sparse() {
        let mat =
            SparseMatrix::from_coo(4, 4, vec![0, 1, 2, 3], vec![0, 1, 2, 3], vec![1.0; 4]).unwrap();
        let stats = SparseStats::from_sparse(&mat);
        assert_eq!(stats.total_elements, 16);
        assert_eq!(stats.nnz, 4);
        assert!((stats.sparsity_ratio - 0.75).abs() < 1e-9);
        assert!(stats.memory_savings > 0.0);
        assert!((stats.flop_reduction - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_from_csr() {
        let dense = vec![1.0, 0.0, 0.0, 2.0];
        let mat = SparseMatrix::from_dense(&dense, 2, 2, 0.0).unwrap();
        let csr = mat.to_csr();
        let stats = SparseStats::from_csr(&csr);
        assert_eq!(stats.nnz, 2);
        assert!((stats.sparsity_ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_from_block_sparse() {
        let dense =
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bs = BlockSparse::from_dense(&dense, 4, 4, 2, 0.0).unwrap();
        let stats = SparseStats::from_block_sparse(&bs);
        assert_eq!(stats.total_elements, 16);
        // 1 block of 4 elements stored.
        assert_eq!(stats.nnz, 4);
        assert!(stats.memory_savings > 0.0);
    }

    #[test]
    fn test_stats_from_dense() {
        let data = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let stats = SparseStats::from_dense(&data, 3, 3, 0.0);
        assert_eq!(stats.nnz, 3);
        assert!((stats.sparsity_ratio - (6.0 / 9.0)).abs() < 1e-9);
    }

    #[test]
    fn test_stats_display() {
        let mat = SparseMatrix::from_coo(4, 4, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let stats = SparseStats::from_sparse(&mat);
        let display = format!("{stats}");
        assert!(display.contains("NNZ:"));
        assert!(display.contains("sparse"));
    }

    #[test]
    fn test_stats_fully_dense() {
        let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let stats = SparseStats::from_dense(&data, 3, 3, 0.0);
        assert_eq!(stats.nnz, 9);
        assert!(stats.sparsity_ratio.abs() < 1e-9);
    }

    #[test]
    fn test_stats_all_zero() {
        let data = vec![0.0; 9];
        let stats = SparseStats::from_dense(&data, 3, 3, 0.0);
        assert_eq!(stats.nnz, 0);
        assert!((stats.sparsity_ratio - 1.0).abs() < 1e-9);
        assert!(stats.flop_reduction.abs() < 1e-9); // 0 nnz → 0
    }

    // ── Property-style tests ────────────────────────────────────────────

    #[test]
    fn test_sparse_preserves_nonzero_structure() {
        // Dense → Sparse → Dense should preserve all non-zero entries.
        let dense = vec![0.0, 1.5, 0.0, -2.0, 0.0, 3.5, 0.0, 0.0, -4.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        let recovered = mat.to_dense();
        for (i, (&orig, &rec)) in dense.iter().zip(recovered.iter()).enumerate() {
            if orig != 0.0 {
                assert!(
                    approx_eq(orig, rec, 1e-7),
                    "Non-zero entry at {i} differs: {orig} vs {rec}"
                );
            }
        }
    }

    #[test]
    fn test_csr_nnz_matches_values_len() {
        let dense = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        let csr = mat.to_csr();
        assert_eq!(csr.values.len(), mat.nnz());
        assert_eq!(csr.col_indices.len(), mat.nnz());
        assert_eq!(csr.row_ptrs.len(), mat.rows + 1);
        assert_eq!(*csr.row_ptrs.last().unwrap(), mat.nnz());
    }

    #[test]
    fn test_csc_nnz_matches_values_len() {
        let dense = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        let csc = mat.to_csc();
        assert_eq!(csc.values.len(), mat.nnz());
        assert_eq!(csc.row_indices.len(), mat.nnz());
        assert_eq!(csc.col_ptrs.len(), mat.cols + 1);
        assert_eq!(*csc.col_ptrs.last().unwrap(), mat.nnz());
    }

    #[test]
    fn test_spmv_identity_is_passthrough() {
        let n = 5;
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            dense[i * n + i] = 1.0;
        }
        let mat = SparseMatrix::from_dense(&dense, n, n, 0.0).unwrap();
        let csr = mat.to_csr();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![0.0; n];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        assert_dense_eq(&y, &x, 1e-7);
    }

    #[test]
    fn test_block_sparse_spmv_matches_dense() {
        let dense_a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = vec![1.0, -1.0, 0.5, 2.0];
        let expected = dense_matvec(&dense_a, &x, 4, 4);
        let bs = BlockSparse::from_dense(&dense_a, 4, 4, 2, 0.0).unwrap();
        let mut y = vec![0.0; 4];
        bs.spmv(&x, &mut y).unwrap();
        assert_dense_eq(&y, &expected, 1e-5);
    }

    #[test]
    fn test_pruning_then_sparse_roundtrip() {
        let data = vec![0.5, 0.01, 0.9, 0.02, 0.8, 0.03, 0.7, 0.04, 0.6];
        let mut masked = data.clone();
        let mask = PruningMask::generate(&data, 3, 3, 0.1, PruningStrategy::Unstructured).unwrap();
        mask.apply(&mut masked);

        let mat = SparseMatrix::from_dense(&masked, 3, 3, 0.0).unwrap();
        assert_eq!(mat.nnz(), mask.kept());
        let recovered = mat.to_dense();
        assert_dense_eq(&recovered, &masked, 1e-7);
    }

    // ── OpenCL kernel source sanity ─────────────────────────────────────

    #[test]
    fn test_opencl_kernel_source_present() {
        assert!(!SPARSE_MATMUL_CL.is_empty());
        assert!(SPARSE_MATMUL_CL.contains("spmv_csr"));
        assert!(SPARSE_MATMUL_CL.contains("__kernel"));
    }

    #[test]
    fn test_opencl_kernel_source_has_blocked_variant() {
        assert!(SPARSE_MATMUL_CL.contains("spmv_csr_blocked"));
        assert!(SPARSE_MATMUL_CL.contains("block_size"));
    }

    // ── Additional edge-case tests ──────────────────────────────────────

    #[test]
    fn test_1x1_matrix() {
        let mat = SparseMatrix::from_dense(&[42.0], 1, 1, 0.0).unwrap();
        assert_eq!(mat.nnz(), 1);
        let csr = mat.to_csr();
        let csc = mat.to_csc();
        assert_dense_eq(&SparseMatrix::from_csr(&csr).to_dense(), &[42.0], 1e-7);
        assert_dense_eq(&SparseMatrix::from_csc(&csc).to_dense(), &[42.0], 1e-7);
    }

    #[test]
    fn test_wide_matrix() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 2, 5, 0.0).unwrap();
        assert_eq!(mat.nnz(), 5);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_tall_matrix() {
        let dense = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0];
        let mat = SparseMatrix::from_dense(&dense, 5, 2, 0.0).unwrap();
        assert_eq!(mat.nnz(), 5);
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_negative_values_roundtrip() {
        let dense = vec![-1.0, 0.0, -2.0, 0.0, -3.0, 0.0, -4.0, 0.0, -5.0];
        let mat = SparseMatrix::from_dense(&dense, 3, 3, 0.0).unwrap();
        assert_dense_eq(&mat.to_dense(), &dense, 1e-7);
    }

    #[test]
    fn test_spmv_with_negative_values() {
        let dense_a = vec![-1.0, 2.0, 3.0, -4.0];
        let mat = SparseMatrix::from_dense(&dense_a, 2, 2, 0.0).unwrap();
        let csr = mat.to_csr();
        let x = vec![-1.0, 1.0];
        let mut y = vec![0.0; 2];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        let expected = dense_matvec(&dense_a, &x, 2, 2);
        assert_dense_eq(&y, &expected, 1e-6);
    }

    #[test]
    fn test_sparse_matrix_accessors() {
        let mat = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![1, 0], vec![3.0, 4.0]).unwrap();
        assert_eq!(mat.row_indices(), &[0, 1]);
        assert_eq!(mat.col_indices(), &[1, 0]);
        assert_eq!(mat.values(), &[3.0, 4.0]);
    }

    #[test]
    fn test_large_sparse_spmv() {
        // 100×100 diagonal.
        let n = 100;
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            dense[i * n + i] = (i + 1) as f32;
        }
        let mat = SparseMatrix::from_dense(&dense, n, n, 0.0).unwrap();
        assert_eq!(mat.nnz(), n);
        let csr = mat.to_csr();
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut y = vec![0.0; n];
        SparseDenseMatmul::spmv_csr(&csr, &x, &mut y).unwrap();
        let expected = dense_matvec(&dense, &x, n, n);
        assert_dense_eq(&y, &expected, 1e-4);
    }

    #[test]
    fn test_block_sparse_large_16x16() {
        // 32×32 with one 16×16 block filled.
        let n = 32;
        let mut dense = vec![0.0f32; n * n];
        for r in 0..16 {
            for c in 16..32 {
                dense[r * n + c] = ((r + 1) * (c + 1)) as f32;
            }
        }
        let bs = BlockSparse::from_dense(&dense, n, n, 16, 0.0).unwrap();
        assert_eq!(bs.num_blocks(), 1);
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = vec![0.0f32; n];
        bs.spmv(&x, &mut y).unwrap();
        let expected = dense_matvec(&dense, &x, n, n);
        assert_dense_eq(&y, &expected, 1e-2);
    }

    #[test]
    fn test_pruning_strategy_debug() {
        // Ensure all variants are Debug-printable.
        let _ = format!("{:?}", PruningStrategy::Unstructured);
        let _ = format!("{:?}", PruningStrategy::RowStructured);
        let _ = format!("{:?}", PruningStrategy::ColumnStructured);
        let _ = format!("{:?}", PruningStrategy::BlockStructured { block_size: 16 });
    }

    #[test]
    fn test_sparse_mul_mode_eq() {
        assert_eq!(SparseMulMode::Intersection, SparseMulMode::Intersection);
        assert_ne!(SparseMulMode::Intersection, SparseMulMode::Union);
    }
}
