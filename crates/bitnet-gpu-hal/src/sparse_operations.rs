//! Module stub - implementation pending merge from feature branch
//! Sparse matrix operations for GPU-accelerated inference.
//!
//! Provides CSR/CSC/COO/BSR/ELL sparse formats, sparse × dense matmul,
//! sparse softmax, sparse attention patterns, top-k sparsification,
//! block-sparse storage, format conversion, sparsity analysis, and a
//! unified dispatch engine.

use std::fmt;

// ---------------------------------------------------------------------------
// SparseFormat — enum of supported sparse storage layouts
// ---------------------------------------------------------------------------

/// Identifies a sparse storage layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Compressed Sparse Row.
    CSR,
    /// Compressed Sparse Column.
    CSC,
    /// Coordinate list (row, col, value triples).
    COO,
    /// Block Sparse Row with fixed block size.
    BSR,
    /// ELLPACK — padded fixed-width per-row storage.
    ELL,
}

impl fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CSR => write!(f, "CSR"),
            Self::CSC => write!(f, "CSC"),
            Self::COO => write!(f, "COO"),
            Self::BSR => write!(f, "BSR"),
            Self::ELL => write!(f, "ELL"),
        }
    }
}

// ---------------------------------------------------------------------------
// SparseMatrix — generic sparse container
// ---------------------------------------------------------------------------

/// A sparse matrix stored in one of the supported formats.
///
/// All public constructors validate their invariants and return `Result`.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Storage format.
    pub format: SparseFormat,
    /// Row indices (CSR: row_ptr; COO: row indices; CSC: row indices).
    pub row_indices: Vec<usize>,
    /// Column indices (CSR: col indices; COO: col indices; CSC: col_ptr).
    pub col_indices: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Number of stored non-zeros.
    pub nnz: usize,
}

/// Error type for sparse operations.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseError(pub String);

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SparseError: {}", self.0)
    }
}

impl std::error::Error for SparseError {}

impl SparseMatrix {
    // -- CSR constructor --------------------------------------------------

    /// Create a CSR matrix.
    ///
    /// * `row_ptr` – length `rows + 1`, monotonically non-decreasing.
    /// * `col_indices` – column index for each non-zero (length = nnz).
    /// * `values` – value for each non-zero (length = nnz).
    pub fn new_csr(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, SparseError> {
        if row_ptr.len() != rows + 1 {
            return Err(SparseError(format!(
                "row_ptr length {} != rows+1 {}",
                row_ptr.len(),
                rows + 1
            )));
        }
        let nnz = *row_ptr.last().unwrap_or(&0);
        if col_indices.len() != nnz || values.len() != nnz {
            return Err(SparseError(format!(
                "col_indices/values length mismatch: {} / {} vs nnz {}",
                col_indices.len(),
                values.len(),
                nnz
            )));
        }
        for &c in &col_indices {
            if c >= cols {
                return Err(SparseError(format!("col index {c} out of bounds (cols={cols})")));
            }
        }
        Ok(Self {
            rows,
            cols,
            format: SparseFormat::CSR,
            row_indices: row_ptr,
            col_indices,
            values,
            nnz,
        })
    }

    // -- CSC constructor --------------------------------------------------

    /// Create a CSC matrix.
    pub fn new_csc(
        rows: usize,
        cols: usize,
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, SparseError> {
        if col_ptr.len() != cols + 1 {
            return Err(SparseError(format!(
                "col_ptr length {} != cols+1 {}",
                col_ptr.len(),
                cols + 1
            )));
        }
        let nnz = *col_ptr.last().unwrap_or(&0);
        if row_indices.len() != nnz || values.len() != nnz {
            return Err(SparseError(format!(
                "row_indices/values length mismatch: {} / {} vs nnz {}",
                row_indices.len(),
                values.len(),
                nnz
            )));
        }
        for &r in &row_indices {
            if r >= rows {
                return Err(SparseError(format!("row index {r} out of bounds (rows={rows})")));
            }
        }
        Ok(Self {
            rows,
            cols,
            format: SparseFormat::CSC,
            row_indices,
            col_indices: col_ptr,
            values,
            nnz,
        })
    }

    // -- COO constructor --------------------------------------------------

    /// Create a COO matrix.
    pub fn new_coo(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, SparseError> {
        let nnz = values.len();
        if row_indices.len() != nnz || col_indices.len() != nnz {
            return Err(SparseError("index/value length mismatch".into()));
        }
        for (&r, &c) in row_indices.iter().zip(&col_indices) {
            if r >= rows || c >= cols {
                return Err(SparseError(format!("index ({r},{c}) out of bounds ({rows},{cols})")));
            }
        }
        Ok(Self { rows, cols, format: SparseFormat::COO, row_indices, col_indices, values, nnz })
    }

    /// Create an empty sparse matrix in the given format.
    pub fn empty(rows: usize, cols: usize, format: SparseFormat) -> Self {
        match format {
            SparseFormat::CSR => Self {
                rows,
                cols,
                format,
                row_indices: vec![0; rows + 1],
                col_indices: vec![],
                values: vec![],
                nnz: 0,
            },
            SparseFormat::CSC => Self {
                rows,
                cols,
                format,
                row_indices: vec![],
                col_indices: vec![0; cols + 1],
                values: vec![],
                nnz: 0,
            },
            SparseFormat::COO | SparseFormat::BSR | SparseFormat::ELL => Self {
                rows,
                cols,
                format,
                row_indices: vec![],
                col_indices: vec![],
                values: vec![],
                nnz: 0,
            },
        }
    }

    /// Construct a dense matrix (row-major) from this sparse matrix.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        match self.format {
            SparseFormat::CSR => {
                for row in 0..self.rows {
                    let start = self.row_indices[row];
                    let end = self.row_indices[row + 1];
                    for idx in start..end {
                        dense[row * self.cols + self.col_indices[idx]] = self.values[idx];
                    }
                }
            }
            SparseFormat::CSC => {
                for col in 0..self.cols {
                    let start = self.col_indices[col];
                    let end = self.col_indices[col + 1];
                    for idx in start..end {
                        dense[self.row_indices[idx] * self.cols + col] = self.values[idx];
                    }
                }
            }
            SparseFormat::COO => {
                for i in 0..self.nnz {
                    dense[self.row_indices[i] * self.cols + self.col_indices[i]] = self.values[i];
                }
            }
            _ => {}
        }
        dense
    }

    /// Build a CSR matrix from a dense row-major buffer.
    pub fn from_dense_csr(rows: usize, cols: usize, data: &[f32]) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v != 0.0 {
                    col_indices.push(c);
                    values.push(v);
                }
            }
            row_ptr[r + 1] = values.len();
        }
        let nnz = values.len();
        Self {
            rows,
            cols,
            format: SparseFormat::CSR,
            row_indices: row_ptr,
            col_indices,
            values,
            nnz,
        }
    }

    /// Density ratio (nnz / total elements), returns 0.0 for empty dimensions.
    pub fn density(&self) -> f64 {
        let total = self.rows as f64 * self.cols as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.nnz as f64 / total
    }
}

// ---------------------------------------------------------------------------
// SparseMatMul — sparse × dense multiplication
// ---------------------------------------------------------------------------

/// Sparse × dense matrix multiplication engine.
#[derive(Debug, Clone)]
pub struct SparseMatMul;

impl SparseMatMul {
    /// Create a new `SparseMatMul`.
    pub fn new() -> Self {
        Self
    }

    /// Multiply sparse A (rows×K) by dense B (K×N, row-major) → C (rows×N).
    ///
    /// A must be in CSR format.
    pub fn mul_csr_dense(
        &self,
        a: &SparseMatrix,
        b: &[f32],
        b_cols: usize,
    ) -> Result<Vec<f32>, SparseError> {
        if a.format != SparseFormat::CSR {
            return Err(SparseError("SparseMatMul requires CSR input".into()));
        }
        if b.len() != a.cols * b_cols {
            return Err(SparseError(format!(
                "B shape mismatch: {} != {}×{}",
                b.len(),
                a.cols,
                b_cols
            )));
        }
        let mut c = vec![0.0f32; a.rows * b_cols];
        for row in 0..a.rows {
            let start = a.row_indices[row];
            let end = a.row_indices[row + 1];
            for idx in start..end {
                let col_a = a.col_indices[idx];
                let val_a = a.values[idx];
                for j in 0..b_cols {
                    c[row * b_cols + j] += val_a * b[col_a * b_cols + j];
                }
            }
        }
        Ok(c)
    }

    /// Dense reference multiplication for verification.
    pub fn dense_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }
}

impl Default for SparseMatMul {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SparseSoftmax — softmax on sparse attention patterns
// ---------------------------------------------------------------------------

/// Computes softmax over sparse attention patterns.
///
/// Only the stored non-zero positions participate; absent positions
/// are treated as −∞.
#[derive(Debug, Clone)]
pub struct SparseSoftmax;

impl SparseSoftmax {
    /// Create a new `SparseSoftmax`.
    pub fn new() -> Self {
        Self
    }

    /// In-place row-wise softmax on a CSR sparse matrix.
    pub fn softmax_csr_inplace(&self, mat: &mut SparseMatrix) -> Result<(), SparseError> {
        if mat.format != SparseFormat::CSR {
            return Err(SparseError("SparseSoftmax requires CSR format".into()));
        }
        for row in 0..mat.rows {
            let start = mat.row_indices[row];
            let end = mat.row_indices[row + 1];
            if start == end {
                continue;
            }
            // Numerically stable softmax
            let max_val = mat.values[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in &mut mat.values[start..end] {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in &mut mat.values[start..end] {
                    *v /= sum;
                }
            }
        }
        Ok(())
    }

    /// Dense reference softmax (row-wise) for verification.
    pub fn dense_softmax(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(data.len(), rows * cols);
        let mut out = data.to_vec();
        for r in 0..rows {
            let row_start = r * cols;
            let row_end = row_start + cols;
            let max_v = out[row_start..row_end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in &mut out[row_start..row_end] {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in &mut out[row_start..row_end] {
                    *v /= sum;
                }
            }
        }
        out
    }
}

impl Default for SparseSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SparseAttention — sparse attention with local + strided patterns
// ---------------------------------------------------------------------------

/// Configuration for sparse attention patterns.
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Size of the local attention window (each token attends to ±window/2).
    pub local_window: usize,
    /// Stride for the global strided pattern (every `stride`-th token).
    pub stride: usize,
}

/// Sparse attention computation combining local window and strided patterns.
#[derive(Debug, Clone)]
pub struct SparseAttention {
    config: SparseAttentionConfig,
}

impl SparseAttention {
    /// Create with the given configuration.
    pub fn new(config: SparseAttentionConfig) -> Self {
        Self { config }
    }

    /// Build the attention mask as a CSR sparse matrix.
    ///
    /// Entry (i,j) = 1.0 if token i attends to token j.
    pub fn build_mask(&self, seq_len: usize) -> SparseMatrix {
        let mut row_ptr = vec![0usize; seq_len + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let half = self.config.local_window / 2;

        for i in 0..seq_len {
            let mut cols_for_row: Vec<usize> = Vec::new();

            // Local window
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(seq_len);
            for j in lo..hi {
                cols_for_row.push(j);
            }

            // Strided pattern
            if self.config.stride > 0 {
                let mut j = 0;
                while j < seq_len {
                    cols_for_row.push(j);
                    j += self.config.stride;
                }
            }

            cols_for_row.sort_unstable();
            cols_for_row.dedup();

            for &c in &cols_for_row {
                col_indices.push(c);
                values.push(1.0);
            }
            row_ptr[i + 1] = col_indices.len();
        }

        let nnz = values.len();
        SparseMatrix {
            rows: seq_len,
            cols: seq_len,
            format: SparseFormat::CSR,
            row_indices: row_ptr,
            col_indices,
            values,
            nnz,
        }
    }

    /// Apply sparse attention: `softmax(mask ⊙ QK^T / √d) × V`.
    ///
    /// * `q`, `k`, `v` — dense matrices (seq_len × head_dim), row-major.
    /// * Returns dense output (seq_len × head_dim).
    pub fn apply(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, SparseError> {
        if q.len() != seq_len * head_dim
            || k.len() != seq_len * head_dim
            || v.len() != seq_len * head_dim
        {
            return Err(SparseError("q/k/v shape mismatch".into()));
        }

        let mut mask = self.build_mask(seq_len);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute QK^T at sparse positions and scale
        for row in 0..seq_len {
            let start = mask.row_indices[row];
            let end = mask.row_indices[row + 1];
            for idx in start..end {
                let col = mask.col_indices[idx];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[row * head_dim + d] * k[col * head_dim + d];
                }
                mask.values[idx] = dot * scale;
            }
        }

        // Sparse softmax
        let softmax = SparseSoftmax::new();
        softmax.softmax_csr_inplace(&mut mask)?;

        // Multiply attention weights × V
        let mm = SparseMatMul::new();
        mm.mul_csr_dense(&mask, v, head_dim)
    }
}

// ---------------------------------------------------------------------------
// TopKSparsifier — convert dense to sparse by top-k
// ---------------------------------------------------------------------------

/// Converts a dense tensor to sparse by retaining the top-k values per row.
#[derive(Debug, Clone)]
pub struct TopKSparsifier {
    k: usize,
}

impl TopKSparsifier {
    /// Create a sparsifier keeping `k` largest-magnitude values per row.
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Return the configured k.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Sparsify a dense row-major matrix. Returns a CSR `SparseMatrix`.
    pub fn sparsify(&self, data: &[f32], rows: usize, cols: usize) -> SparseMatrix {
        assert_eq!(data.len(), rows * cols);

        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let k = self.k.min(cols);

        for r in 0..rows {
            let row_data = &data[r * cols..(r + 1) * cols];
            // Build (col, |val|) pairs and partial-sort by descending magnitude
            let mut indexed: Vec<(usize, f32)> =
                row_data.iter().enumerate().map(|(c, &v)| (c, v)).collect();
            indexed.sort_by(|a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
            // Sort by column index for CSR ordering
            indexed.sort_by_key(|&(c, _)| c);
            for (c, v) in indexed {
                if v != 0.0 {
                    col_indices.push(c);
                    values.push(v);
                }
            }
            row_ptr[r + 1] = values.len();
        }

        let nnz = values.len();
        SparseMatrix {
            rows,
            cols,
            format: SparseFormat::CSR,
            row_indices: row_ptr,
            col_indices,
            values,
            nnz,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockSparseFormat — block-sparse storage
// ---------------------------------------------------------------------------

/// A block-sparse matrix stored as dense blocks at sparse positions.
#[derive(Debug, Clone)]
pub struct BlockSparseFormat {
    /// Number of rows in the full matrix.
    pub rows: usize,
    /// Number of columns in the full matrix.
    pub cols: usize,
    /// Block height.
    pub block_rows: usize,
    /// Block width.
    pub block_cols: usize,
    /// Block-row indices (length = number of blocks).
    pub block_row_indices: Vec<usize>,
    /// Block-column indices (length = number of blocks).
    pub block_col_indices: Vec<usize>,
    /// Dense data for each block, concatenated (each block is `block_rows * block_cols`).
    pub block_data: Vec<f32>,
}

impl BlockSparseFormat {
    /// Create a new block-sparse matrix.
    pub fn new(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        block_row_indices: Vec<usize>,
        block_col_indices: Vec<usize>,
        block_data: Vec<f32>,
    ) -> Result<Self, SparseError> {
        let n_blocks = block_row_indices.len();
        if block_col_indices.len() != n_blocks {
            return Err(SparseError("block index length mismatch".into()));
        }
        let expected_data = n_blocks * block_rows * block_cols;
        if block_data.len() != expected_data {
            return Err(SparseError(format!(
                "block_data length {} != expected {}",
                block_data.len(),
                expected_data
            )));
        }
        if block_rows == 0 || block_cols == 0 {
            return Err(SparseError("block dimensions must be > 0".into()));
        }
        Ok(Self {
            rows,
            cols,
            block_rows,
            block_cols,
            block_row_indices,
            block_col_indices,
            block_data,
        })
    }

    /// Create an empty block-sparse matrix.
    pub fn empty(rows: usize, cols: usize, block_rows: usize, block_cols: usize) -> Self {
        Self {
            rows,
            cols,
            block_rows,
            block_cols,
            block_row_indices: vec![],
            block_col_indices: vec![],
            block_data: vec![],
        }
    }

    /// Number of stored blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_row_indices.len()
    }

    /// Convert to a dense row-major matrix.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        let bs = self.block_rows * self.block_cols;
        for (blk_idx, (&br, &bc)) in
            self.block_row_indices.iter().zip(&self.block_col_indices).enumerate()
        {
            let data_offset = blk_idx * bs;
            for lr in 0..self.block_rows {
                for lc in 0..self.block_cols {
                    let gr = br * self.block_rows + lr;
                    let gc = bc * self.block_cols + lc;
                    if gr < self.rows && gc < self.cols {
                        dense[gr * self.cols + gc] =
                            self.block_data[data_offset + lr * self.block_cols + lc];
                    }
                }
            }
        }
        dense
    }

    /// Build from a dense matrix, keeping only blocks whose Frobenius norm
    /// exceeds `threshold`.
    pub fn from_dense(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        data: &[f32],
        threshold: f32,
    ) -> Self {
        assert_eq!(data.len(), rows * cols);
        let n_block_rows = (rows + block_rows - 1) / block_rows;
        let n_block_cols = (cols + block_cols - 1) / block_cols;

        let mut bri = Vec::new();
        let mut bci = Vec::new();
        let mut bd = Vec::new();

        for br in 0..n_block_rows {
            for bc in 0..n_block_cols {
                let mut block = vec![0.0f32; block_rows * block_cols];
                let mut norm_sq = 0.0f32;
                for lr in 0..block_rows {
                    for lc in 0..block_cols {
                        let gr = br * block_rows + lr;
                        let gc = bc * block_cols + lc;
                        if gr < rows && gc < cols {
                            let v = data[gr * cols + gc];
                            block[lr * block_cols + lc] = v;
                            norm_sq += v * v;
                        }
                    }
                }
                if norm_sq.sqrt() > threshold {
                    bri.push(br);
                    bci.push(bc);
                    bd.extend_from_slice(&block);
                }
            }
        }

        Self {
            rows,
            cols,
            block_rows,
            block_cols,
            block_row_indices: bri,
            block_col_indices: bci,
            block_data: bd,
        }
    }
}

// ---------------------------------------------------------------------------
// SparseConverter — format conversions
// ---------------------------------------------------------------------------

/// Converts between sparse formats.
#[derive(Debug, Clone)]
pub struct SparseConverter;

impl SparseConverter {
    /// Create a new converter.
    pub fn new() -> Self {
        Self
    }

    /// Convert any supported format to CSR.
    pub fn to_csr(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        match mat.format {
            SparseFormat::CSR => Ok(mat.clone()),
            SparseFormat::COO => self.coo_to_csr(mat),
            SparseFormat::CSC => self.csc_to_csr(mat),
            _ => Err(SparseError(format!("Conversion from {} to CSR not supported", mat.format))),
        }
    }

    /// Convert any supported format to CSC.
    pub fn to_csc(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        match mat.format {
            SparseFormat::CSC => Ok(mat.clone()),
            SparseFormat::COO => self.coo_to_csc(mat),
            SparseFormat::CSR => self.csr_to_csc(mat),
            _ => Err(SparseError(format!("Conversion from {} to CSC not supported", mat.format))),
        }
    }

    /// Convert any supported format to COO.
    pub fn to_coo(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        match mat.format {
            SparseFormat::COO => Ok(mat.clone()),
            SparseFormat::CSR => self.csr_to_coo(mat),
            SparseFormat::CSC => self.csc_to_coo(mat),
            _ => Err(SparseError(format!("Conversion from {} to COO not supported", mat.format))),
        }
    }

    // -- internal conversions ---------------------------------------------

    fn coo_to_csr(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let mut row_ptr = vec![0usize; mat.rows + 1];
        // Count per row
        for &r in &mat.row_indices {
            row_ptr[r + 1] += 1;
        }
        // Prefix sum
        for i in 1..=mat.rows {
            row_ptr[i] += row_ptr[i - 1];
        }
        let nnz = mat.nnz;
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![0.0f32; nnz];
        let mut offset = row_ptr.clone();
        for i in 0..nnz {
            let r = mat.row_indices[i];
            let pos = offset[r];
            col_indices[pos] = mat.col_indices[i];
            values[pos] = mat.values[i];
            offset[r] += 1;
        }
        // Sort within each row by column index
        for r in 0..mat.rows {
            let start = row_ptr[r];
            let end = row_ptr[r + 1];
            if start < end {
                let mut pairs: Vec<(usize, f32)> = col_indices[start..end]
                    .iter()
                    .zip(&values[start..end])
                    .map(|(&c, &v)| (c, v))
                    .collect();
                pairs.sort_by_key(|&(c, _)| c);
                for (i, (c, v)) in pairs.into_iter().enumerate() {
                    col_indices[start + i] = c;
                    values[start + i] = v;
                }
            }
        }
        SparseMatrix::new_csr(mat.rows, mat.cols, row_ptr, col_indices, values)
    }

    fn coo_to_csc(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let mut col_ptr = vec![0usize; mat.cols + 1];
        for &c in &mat.col_indices {
            col_ptr[c + 1] += 1;
        }
        for i in 1..=mat.cols {
            col_ptr[i] += col_ptr[i - 1];
        }
        let nnz = mat.nnz;
        let mut row_indices = vec![0usize; nnz];
        let mut values = vec![0.0f32; nnz];
        let mut offset = col_ptr.clone();
        for i in 0..nnz {
            let c = mat.col_indices[i];
            let pos = offset[c];
            row_indices[pos] = mat.row_indices[i];
            values[pos] = mat.values[i];
            offset[c] += 1;
        }
        // Sort within each column by row index
        for c in 0..mat.cols {
            let start = col_ptr[c];
            let end = col_ptr[c + 1];
            if start < end {
                let mut pairs: Vec<(usize, f32)> = row_indices[start..end]
                    .iter()
                    .zip(&values[start..end])
                    .map(|(&r, &v)| (r, v))
                    .collect();
                pairs.sort_by_key(|&(r, _)| r);
                for (i, (r, v)) in pairs.into_iter().enumerate() {
                    row_indices[start + i] = r;
                    values[start + i] = v;
                }
            }
        }
        SparseMatrix::new_csc(mat.rows, mat.cols, col_ptr, row_indices, values)
    }

    fn csr_to_coo(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let mut row_indices = Vec::with_capacity(mat.nnz);
        for r in 0..mat.rows {
            let start = mat.row_indices[r];
            let end = mat.row_indices[r + 1];
            for _ in start..end {
                row_indices.push(r);
            }
        }
        SparseMatrix::new_coo(
            mat.rows,
            mat.cols,
            row_indices,
            mat.col_indices.clone(),
            mat.values.clone(),
        )
    }

    fn csc_to_coo(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let mut row_indices = Vec::with_capacity(mat.nnz);
        let mut col_indices = Vec::with_capacity(mat.nnz);
        let mut values = Vec::with_capacity(mat.nnz);
        for c in 0..mat.cols {
            let start = mat.col_indices[c];
            let end = mat.col_indices[c + 1];
            for idx in start..end {
                row_indices.push(mat.row_indices[idx]);
                col_indices.push(c);
                values.push(mat.values[idx]);
            }
        }
        SparseMatrix::new_coo(mat.rows, mat.cols, row_indices, col_indices, values)
    }

    fn csr_to_csc(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let coo = self.csr_to_coo(mat)?;
        self.coo_to_csc(&coo)
    }

    fn csc_to_csr(&self, mat: &SparseMatrix) -> Result<SparseMatrix, SparseError> {
        let coo = self.csc_to_coo(mat)?;
        self.coo_to_csr(&coo)
    }
}

impl Default for SparseConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SparseAnalyzer — sparsity pattern analysis
// ---------------------------------------------------------------------------

/// Analyzes sparsity patterns.
#[derive(Debug, Clone)]
pub struct SparseAnalyzer;

/// Result of sparsity analysis.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of non-zeros.
    pub nnz: usize,
    /// Density ratio (nnz / total).
    pub density: f64,
    /// Average non-zeros per row (0.0 if no rows).
    pub avg_nnz_per_row: f64,
    /// Maximum non-zeros in any single row.
    pub max_nnz_per_row: usize,
    /// Minimum non-zeros in any single row.
    pub min_nnz_per_row: usize,
    /// Whether the matrix is structurally symmetric.
    pub is_symmetric: bool,
    /// Whether the matrix has block structure (at given block size).
    pub has_block_structure: bool,
    /// Block size used for block structure detection (0 if not checked).
    pub block_size: usize,
}

impl SparseAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a CSR sparse matrix.
    pub fn analyze(&self, mat: &SparseMatrix) -> Result<SparsityStats, SparseError> {
        if mat.format != SparseFormat::CSR {
            return Err(SparseError("SparseAnalyzer requires CSR format".into()));
        }
        let mut max_nnz = 0usize;
        let mut min_nnz = usize::MAX;
        let mut nnz_per_row = Vec::with_capacity(mat.rows);

        for r in 0..mat.rows {
            let count = mat.row_indices[r + 1] - mat.row_indices[r];
            nnz_per_row.push(count);
            max_nnz = max_nnz.max(count);
            min_nnz = min_nnz.min(count);
        }
        if mat.rows == 0 {
            min_nnz = 0;
        }
        let avg = if mat.rows > 0 { mat.nnz as f64 / mat.rows as f64 } else { 0.0 };

        let is_sym = self.check_symmetry(mat);

        Ok(SparsityStats {
            rows: mat.rows,
            cols: mat.cols,
            nnz: mat.nnz,
            density: mat.density(),
            avg_nnz_per_row: avg,
            max_nnz_per_row: max_nnz,
            min_nnz_per_row: min_nnz,
            is_symmetric: is_sym,
            has_block_structure: false,
            block_size: 0,
        })
    }

    /// Analyze with block structure detection at the given block size.
    pub fn analyze_with_blocks(
        &self,
        mat: &SparseMatrix,
        block_size: usize,
    ) -> Result<SparsityStats, SparseError> {
        let mut stats = self.analyze(mat)?;
        stats.block_size = block_size;
        stats.has_block_structure = self.detect_block_structure(mat, block_size);
        Ok(stats)
    }

    fn check_symmetry(&self, mat: &SparseMatrix) -> bool {
        if mat.rows != mat.cols {
            return false;
        }
        // Build a set of (row, col) pairs
        let mut entries = std::collections::HashSet::new();
        for r in 0..mat.rows {
            let start = mat.row_indices[r];
            let end = mat.row_indices[r + 1];
            for idx in start..end {
                entries.insert((r, mat.col_indices[idx]));
            }
        }
        // Check each (r,c) has a matching (c,r)
        for &(r, c) in &entries {
            if !entries.contains(&(c, r)) {
                return false;
            }
        }
        true
    }

    fn detect_block_structure(&self, mat: &SparseMatrix, block_size: usize) -> bool {
        if block_size == 0 || mat.nnz == 0 {
            return false;
        }
        // Check if all non-zero positions are aligned to block boundaries
        for r in 0..mat.rows {
            let start = mat.row_indices[r];
            let end = mat.row_indices[r + 1];
            for idx in start..end {
                let br = r / block_size;
                let bc = mat.col_indices[idx] / block_size;
                // Check if the entire block is filled
                for lr in 0..block_size {
                    for lc in 0..block_size {
                        let gr = br * block_size + lr;
                        let gc = bc * block_size + lc;
                        if gr < mat.rows && gc < mat.cols {
                            // Check this position exists
                            let row_start = mat.row_indices[gr];
                            let row_end = mat.row_indices[gr + 1];
                            let found =
                                mat.col_indices[row_start..row_end].iter().any(|&c| c == gc);
                            if !found {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

impl Default for SparseAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SparseEngine — unified dispatch
// ---------------------------------------------------------------------------

/// Unified dispatch engine for all sparse operations.
#[derive(Debug, Clone)]
pub struct SparseEngine {
    converter: SparseConverter,
    matmul: SparseMatMul,
    softmax: SparseSoftmax,
    analyzer: SparseAnalyzer,
}

impl SparseEngine {
    /// Create a new engine with default CPU implementations.
    pub fn new() -> Self {
        Self {
            converter: SparseConverter::new(),
            matmul: SparseMatMul::new(),
            softmax: SparseSoftmax::new(),
            analyzer: SparseAnalyzer::new(),
        }
    }

    /// Access the converter.
    pub fn converter(&self) -> &SparseConverter {
        &self.converter
    }

    /// Access the matmul engine.
    pub fn matmul(&self) -> &SparseMatMul {
        &self.matmul
    }

    /// Access the softmax engine.
    pub fn softmax(&self) -> &SparseSoftmax {
        &self.softmax
    }

    /// Access the analyzer.
    pub fn analyzer(&self) -> &SparseAnalyzer {
        &self.analyzer
    }

    /// Sparse × dense matmul, auto-converting to CSR if needed.
    pub fn spmm(
        &self,
        a: &SparseMatrix,
        b: &[f32],
        b_cols: usize,
    ) -> Result<Vec<f32>, SparseError> {
        let csr = self.converter.to_csr(a)?;
        self.matmul.mul_csr_dense(&csr, b, b_cols)
    }

    /// In-place softmax, auto-converting to CSR if needed.
    pub fn softmax_inplace(&self, mat: &mut SparseMatrix) -> Result<(), SparseError> {
        if mat.format != SparseFormat::CSR {
            *mat = self.converter.to_csr(mat)?;
        }
        self.softmax.softmax_csr_inplace(mat)
    }

    /// Analyze a sparse matrix, auto-converting to CSR if needed.
    pub fn analyze(&self, mat: &SparseMatrix) -> Result<SparsityStats, SparseError> {
        let csr = self.converter.to_csr(mat)?;
        self.analyzer.analyze(&csr)
    }

    /// Top-k sparsify a dense matrix.
    pub fn sparsify_topk(&self, data: &[f32], rows: usize, cols: usize, k: usize) -> SparseMatrix {
        TopKSparsifier::new(k).sparsify(data, rows, cols)
    }

    /// Convert format.
    pub fn convert(
        &self,
        mat: &SparseMatrix,
        target: SparseFormat,
    ) -> Result<SparseMatrix, SparseError> {
        match target {
            SparseFormat::CSR => self.converter.to_csr(mat),
            SparseFormat::CSC => self.converter.to_csc(mat),
            SparseFormat::COO => self.converter.to_coo(mat),
            _ => Err(SparseError(format!("Conversion to {target} not supported"))),
        }
    }
}

impl Default for SparseEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check two f32 slices are approximately equal.
    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // -----------------------------------------------------------------------
    // SparseFormat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_format_display() {
        assert_eq!(SparseFormat::CSR.to_string(), "CSR");
        assert_eq!(SparseFormat::CSC.to_string(), "CSC");
        assert_eq!(SparseFormat::COO.to_string(), "COO");
        assert_eq!(SparseFormat::BSR.to_string(), "BSR");
        assert_eq!(SparseFormat::ELL.to_string(), "ELL");
    }

    #[test]
    fn test_sparse_format_clone_eq() {
        let f = SparseFormat::CSR;
        let f2 = f;
        assert_eq!(f, f2);
        assert_ne!(SparseFormat::CSR, SparseFormat::CSC);
    }

    #[test]
    fn test_sparse_format_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SparseFormat::CSR);
        set.insert(SparseFormat::CSC);
        set.insert(SparseFormat::CSR);
        assert_eq!(set.len(), 2);
    }

    // -----------------------------------------------------------------------
    // SparseMatrix CSR tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_csr_basic() {
        // 2×3 matrix: [[1, 0, 2], [0, 3, 0]]
        let m =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.nnz, 3);
        assert_eq!(m.format, SparseFormat::CSR);
    }

    #[test]
    fn test_csr_to_dense() {
        let m =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let dense = m.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_csr_invalid_row_ptr_length() {
        let r = SparseMatrix::new_csr(2, 3, vec![0, 1], vec![0], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csr_invalid_col_length() {
        let r = SparseMatrix::new_csr(2, 3, vec![0, 1, 2], vec![0], vec![1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csr_col_out_of_bounds() {
        let r = SparseMatrix::new_csr(2, 3, vec![0, 1, 2], vec![5, 0], vec![1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csr_empty_matrix() {
        let m = SparseMatrix::new_csr(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(m.nnz, 0);
        let dense = m.to_dense();
        assert_eq!(dense, vec![0.0; 12]);
    }

    #[test]
    fn test_csr_single_element() {
        let m = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![42.0]).unwrap();
        assert_eq!(m.to_dense(), vec![42.0]);
    }

    // -----------------------------------------------------------------------
    // SparseMatrix CSC tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_csc_basic() {
        // 2×3 matrix: [[1, 0, 2], [0, 3, 0]]
        let m = SparseMatrix::new_csc(2, 3, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 3.0, 2.0])
            .unwrap();
        assert_eq!(m.nnz, 3);
        assert_eq!(m.format, SparseFormat::CSC);
    }

    #[test]
    fn test_csc_to_dense() {
        let m = SparseMatrix::new_csc(2, 3, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 3.0, 2.0])
            .unwrap();
        let dense = m.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_csc_invalid_col_ptr_length() {
        let r = SparseMatrix::new_csc(2, 3, vec![0, 1], vec![0], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csc_row_out_of_bounds() {
        let r = SparseMatrix::new_csc(2, 3, vec![0, 1, 1, 1], vec![5], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csc_empty() {
        let m = SparseMatrix::new_csc(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(m.nnz, 0);
    }

    // -----------------------------------------------------------------------
    // SparseMatrix COO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coo_basic() {
        let m =
            SparseMatrix::new_coo(2, 3, vec![0, 0, 1], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m.nnz, 3);
        assert_eq!(m.format, SparseFormat::COO);
    }

    #[test]
    fn test_coo_to_dense() {
        let m =
            SparseMatrix::new_coo(2, 3, vec![0, 0, 1], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let dense = m.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_coo_out_of_bounds() {
        let r = SparseMatrix::new_coo(2, 3, vec![0, 5], vec![0, 0], vec![1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_coo_length_mismatch() {
        let r = SparseMatrix::new_coo(2, 3, vec![0], vec![0, 1], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_coo_empty() {
        let m = SparseMatrix::new_coo(2, 3, vec![], vec![], vec![]).unwrap();
        assert_eq!(m.nnz, 0);
        assert_eq!(m.to_dense(), vec![0.0; 6]);
    }

    #[test]
    fn test_coo_single_element() {
        let m = SparseMatrix::new_coo(3, 3, vec![1], vec![2], vec![7.0]).unwrap();
        let d = m.to_dense();
        assert_eq!(d[1 * 3 + 2], 7.0);
        assert_eq!(d.iter().filter(|&&v| v != 0.0).count(), 1);
    }

    // -----------------------------------------------------------------------
    // SparseMatrix::empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_csr() {
        let m = SparseMatrix::empty(3, 4, SparseFormat::CSR);
        assert_eq!(m.nnz, 0);
        assert_eq!(m.row_indices.len(), 4);
    }

    #[test]
    fn test_empty_csc() {
        let m = SparseMatrix::empty(3, 4, SparseFormat::CSC);
        assert_eq!(m.nnz, 0);
        assert_eq!(m.col_indices.len(), 5);
    }

    #[test]
    fn test_empty_coo() {
        let m = SparseMatrix::empty(3, 4, SparseFormat::COO);
        assert_eq!(m.nnz, 0);
    }

    // -----------------------------------------------------------------------
    // from_dense_csr / density
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_dense_csr() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let m = SparseMatrix::from_dense_csr(3, 3, &data);
        assert_eq!(m.nnz, 3);
        assert_eq!(m.to_dense(), data);
    }

    #[test]
    fn test_from_dense_csr_all_zeros() {
        let data = vec![0.0; 9];
        let m = SparseMatrix::from_dense_csr(3, 3, &data);
        assert_eq!(m.nnz, 0);
    }

    #[test]
    fn test_from_dense_csr_all_nonzero() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = SparseMatrix::from_dense_csr(2, 2, &data);
        assert_eq!(m.nnz, 4);
        assert_eq!(m.to_dense(), data);
    }

    #[test]
    fn test_density() {
        let m = SparseMatrix::from_dense_csr(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!((m.density() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_density_empty() {
        let m = SparseMatrix::empty(0, 0, SparseFormat::COO);
        assert_eq!(m.density(), 0.0);
    }

    #[test]
    fn test_density_full() {
        let m = SparseMatrix::from_dense_csr(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!((m.density() - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // SparseMatMul tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_identity() {
        // Identity × dense column
        let eye = SparseMatrix::new_csr(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0])
            .unwrap();
        let b = vec![1.0, 2.0, 3.0];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&eye, &b, 1).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // A = [[1, 0, 2], [0, 3, 0]]
        let a =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        // B = [[1, 2], [3, 4], [5, 6]]
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&a, &b, 2).unwrap();
        // C = [[1*1+2*5, 1*2+2*6], [3*3, 3*4]] = [[11, 14], [9, 12]]
        assert_eq!(c, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_matmul_vs_dense_reference() {
        let dense_a = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let dense_b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (m, k, n) = (2, 3, 2);
        let ref_c = SparseMatMul::dense_matmul(&dense_a, &dense_b, m, k, n);
        let sparse_a = SparseMatrix::from_dense_csr(m, k, &dense_a);
        let mm = SparseMatMul::new();
        let sparse_c = mm.mul_csr_dense(&sparse_a, &dense_b, n).unwrap();
        assert!(approx_eq(&sparse_c, &ref_c, 1e-6));
    }

    #[test]
    fn test_matmul_empty_sparse() {
        let a = SparseMatrix::empty(2, 3, SparseFormat::CSR);
        let b = vec![1.0; 6];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&a, &b, 2).unwrap();
        assert_eq!(c, vec![0.0; 4]);
    }

    #[test]
    fn test_matmul_wrong_format() {
        let a = SparseMatrix::empty(2, 3, SparseFormat::COO);
        let mm = SparseMatMul::new();
        assert!(mm.mul_csr_dense(&a, &[1.0; 6], 2).is_err());
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = SparseMatrix::new_csr(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
        let mm = SparseMatMul::new();
        assert!(mm.mul_csr_dense(&a, &[1.0; 4], 2).is_err());
    }

    #[test]
    fn test_matmul_single_element() {
        let a = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![5.0]).unwrap();
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&a, &[3.0], 1).unwrap();
        assert_eq!(c, vec![15.0]);
    }

    #[test]
    fn test_matmul_large_sparse() {
        // 10×10 identity × 10×5 dense
        let mut row_ptr = vec![0usize; 11];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        for i in 0..10 {
            col_idx.push(i);
            vals.push(1.0);
            row_ptr[i + 1] = i + 1;
        }
        let a = SparseMatrix::new_csr(10, 10, row_ptr, col_idx, vals).unwrap();
        let b: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&a, &b, 5).unwrap();
        assert_eq!(c, b);
    }

    #[test]
    fn test_dense_matmul_reference() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = SparseMatMul::dense_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // -----------------------------------------------------------------------
    // SparseSoftmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_uniform() {
        let mut m =
            SparseMatrix::new_csr(1, 4, vec![0, 4], vec![0, 1, 2, 3], vec![0.0, 0.0, 0.0, 0.0])
                .unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        for v in &m.values {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let mut m =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        // Row 0
        let sum0: f32 = m.values[0..2].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-6);
        // Row 1
        let sum1: f32 = m.values[2..3].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_large_values() {
        let mut m =
            SparseMatrix::new_csr(1, 3, vec![0, 3], vec![0, 1, 2], vec![1000.0, 1000.0, 1000.0])
                .unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        for v in &m.values {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut m = SparseMatrix::new_csr(1, 2, vec![0, 2], vec![0, 1], vec![-1.0, -2.0]).unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        assert!(m.values[0] > m.values[1]);
        let sum: f32 = m.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_single_element() {
        let mut m = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![5.0]).unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        assert!((m.values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_empty_row() {
        let mut m = SparseMatrix::new_csr(2, 2, vec![0, 0, 1], vec![0], vec![3.0]).unwrap();
        SparseSoftmax::new().softmax_csr_inplace(&mut m).unwrap();
        assert!((m.values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_wrong_format() {
        let mut m = SparseMatrix::empty(2, 2, SparseFormat::COO);
        assert!(SparseSoftmax::new().softmax_csr_inplace(&mut m).is_err());
    }

    #[test]
    fn test_softmax_vs_dense_reference() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dense_result = SparseSoftmax::dense_softmax(&data, 2, 3);
        let mut sparse = SparseMatrix::from_dense_csr(2, 3, &data);
        SparseSoftmax::new().softmax_csr_inplace(&mut sparse).unwrap();
        let sparse_dense = sparse.to_dense();
        assert!(approx_eq(&sparse_dense, &dense_result, 1e-6));
    }

    // -----------------------------------------------------------------------
    // SparseAttention tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_mask_local_window() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 2, stride: 0 });
        let mask = attn.build_mask(4);
        assert_eq!(mask.rows, 4);
        assert_eq!(mask.cols, 4);
        // Token 0 attends to [0,1], token 1 to [0,1,2], etc.
        let d = mask.to_dense();
        // Row 0: [1,1,0,0]
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 0.0);
    }

    #[test]
    fn test_attention_mask_strided() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 0, stride: 2 });
        let mask = attn.build_mask(6);
        let d = mask.to_dense();
        // Every row should attend to columns 0, 2, 4
        for r in 0..6 {
            assert_eq!(d[r * 6 + 0], 1.0);
            assert_eq!(d[r * 6 + 2], 1.0);
            assert_eq!(d[r * 6 + 4], 1.0);
        }
    }

    #[test]
    fn test_attention_mask_combined() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 2, stride: 4 });
        let mask = attn.build_mask(8);
        assert!(mask.nnz > 0);
        // Check mask is structurally correct
        let d = mask.to_dense();
        for v in &d {
            assert!(*v == 0.0 || *v == 1.0);
        }
    }

    #[test]
    fn test_attention_mask_seq_len_1() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 2, stride: 3 });
        let mask = attn.build_mask(1);
        assert_eq!(mask.nnz, 1);
        assert_eq!(mask.to_dense(), vec![1.0]);
    }

    #[test]
    fn test_attention_apply_tiny() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 4, stride: 0 });
        let seq = 2;
        let dim = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = attn.apply(&q, &k, &v, seq, dim).unwrap();
        assert_eq!(out.len(), seq * dim);
        // Output should be a weighted combination of V rows
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_attention_shape_mismatch() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 2, stride: 0 });
        let r = attn.apply(&[1.0], &[1.0, 2.0], &[1.0], 1, 1);
        assert!(r.is_err());
    }

    // -----------------------------------------------------------------------
    // TopKSparsifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_topk_basic() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0];
        let sp = TopKSparsifier::new(2);
        let m = sp.sparsify(&data, 2, 3);
        assert_eq!(m.format, SparseFormat::CSR);
        // Row 0 top-2 by magnitude: 5.0 (col 1), 3.0 (col 2)
        // Row 1 top-2 by magnitude: 4.0 (col 1), 6.0 (col 2)
        assert!(m.nnz <= 4);
    }

    #[test]
    fn test_topk_k_equals_cols() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sp = TopKSparsifier::new(2);
        let m = sp.sparsify(&data, 2, 2);
        // All non-zero elements kept
        assert_eq!(m.nnz, 4);
        assert!(approx_eq(&m.to_dense(), &data, 1e-6));
    }

    #[test]
    fn test_topk_k_greater_than_cols() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sp = TopKSparsifier::new(10);
        let m = sp.sparsify(&data, 2, 2);
        assert_eq!(m.nnz, 4);
    }

    #[test]
    fn test_topk_k_zero() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sp = TopKSparsifier::new(0);
        let m = sp.sparsify(&data, 2, 2);
        assert_eq!(m.nnz, 0);
    }

    #[test]
    fn test_topk_all_zeros() {
        let data = vec![0.0; 6];
        let sp = TopKSparsifier::new(2);
        let m = sp.sparsify(&data, 2, 3);
        assert_eq!(m.nnz, 0);
    }

    #[test]
    fn test_topk_negative_values() {
        let data = vec![-5.0, 1.0, -3.0, 2.0];
        let sp = TopKSparsifier::new(1);
        let m = sp.sparsify(&data, 2, 2);
        // Row 0: -5.0 has largest magnitude
        let d = m.to_dense();
        assert_eq!(d[0], -5.0);
        // Row 1: -3.0 has largest magnitude
        assert_eq!(d[3], 0.0); // col 1 not kept for row 1
    }

    #[test]
    fn test_topk_preserves_values() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let sp = TopKSparsifier::new(1);
        let m = sp.sparsify(&data, 2, 3);
        // Each row keeps only the largest
        let d = m.to_dense();
        assert_eq!(d[2], 30.0); // row 0, col 2
        assert_eq!(d[5], 60.0); // row 1, col 2
    }

    #[test]
    fn test_topk_getter() {
        let sp = TopKSparsifier::new(7);
        assert_eq!(sp.k(), 7);
    }

    // -----------------------------------------------------------------------
    // BlockSparseFormat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_sparse_basic() {
        let bs = BlockSparseFormat::new(
            4,
            4,
            2,
            2,
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        assert_eq!(bs.num_blocks(), 2);
    }

    #[test]
    fn test_block_sparse_to_dense() {
        let bs =
            BlockSparseFormat::new(4, 4, 2, 2, vec![0], vec![0], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let d = bs.to_dense();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[4], 3.0);
        assert_eq!(d[5], 4.0);
        assert_eq!(d[10], 0.0);
    }

    #[test]
    fn test_block_sparse_empty() {
        let bs = BlockSparseFormat::empty(4, 4, 2, 2);
        assert_eq!(bs.num_blocks(), 0);
        assert_eq!(bs.to_dense(), vec![0.0; 16]);
    }

    #[test]
    fn test_block_sparse_from_dense() {
        let data =
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bs = BlockSparseFormat::from_dense(4, 4, 2, 2, &data, 0.0);
        assert_eq!(bs.num_blocks(), 1);
        assert!(approx_eq(&bs.to_dense(), &data, 1e-6));
    }

    #[test]
    fn test_block_sparse_from_dense_threshold() {
        let data = vec![
            0.01, 0.02, 0.0, 0.0, 0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0,
        ];
        let bs = BlockSparseFormat::from_dense(4, 4, 2, 2, &data, 1.0);
        // Only the block at (1,1) with large values should pass
        assert!(bs.num_blocks() >= 1);
    }

    #[test]
    fn test_block_sparse_roundtrip() {
        let data =
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0];
        let bs = BlockSparseFormat::from_dense(4, 4, 2, 2, &data, 0.0);
        let reconstructed = bs.to_dense();
        assert!(approx_eq(&reconstructed, &data, 1e-6));
    }

    #[test]
    fn test_block_sparse_invalid_data_length() {
        let r = BlockSparseFormat::new(4, 4, 2, 2, vec![0], vec![0], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_block_sparse_index_mismatch() {
        let r = BlockSparseFormat::new(4, 4, 2, 2, vec![0, 1], vec![0], vec![1.0; 8]);
        assert!(r.is_err());
    }

    #[test]
    fn test_block_sparse_zero_block_size() {
        let r = BlockSparseFormat::new(4, 4, 0, 2, vec![], vec![], vec![]);
        assert!(r.is_err());
    }

    // -----------------------------------------------------------------------
    // SparseConverter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_coo_to_csr() {
        let coo =
            SparseMatrix::new_coo(2, 3, vec![0, 1, 0], vec![0, 1, 2], vec![1.0, 3.0, 2.0]).unwrap();
        let conv = SparseConverter::new();
        let csr = conv.to_csr(&coo).unwrap();
        assert_eq!(csr.format, SparseFormat::CSR);
        assert_eq!(csr.to_dense(), coo.to_dense());
    }

    #[test]
    fn test_convert_coo_to_csc() {
        let coo =
            SparseMatrix::new_coo(2, 3, vec![0, 1, 0], vec![0, 1, 2], vec![1.0, 3.0, 2.0]).unwrap();
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&coo).unwrap();
        assert_eq!(csc.format, SparseFormat::CSC);
        assert_eq!(csc.to_dense(), coo.to_dense());
    }

    #[test]
    fn test_convert_csr_to_coo() {
        let csr =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let conv = SparseConverter::new();
        let coo = conv.to_coo(&csr).unwrap();
        assert_eq!(coo.format, SparseFormat::COO);
        assert_eq!(coo.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_convert_csr_to_csc() {
        let csr =
            SparseMatrix::new_csr(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&csr).unwrap();
        assert_eq!(csc.format, SparseFormat::CSC);
        assert_eq!(csc.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_convert_csc_to_csr() {
        let csc = SparseMatrix::new_csc(2, 3, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 3.0, 2.0])
            .unwrap();
        let conv = SparseConverter::new();
        let csr = conv.to_csr(&csc).unwrap();
        assert_eq!(csr.format, SparseFormat::CSR);
        assert_eq!(csr.to_dense(), csc.to_dense());
    }

    #[test]
    fn test_convert_csc_to_coo() {
        let csc = SparseMatrix::new_csc(2, 3, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0, 3.0, 2.0])
            .unwrap();
        let conv = SparseConverter::new();
        let coo = conv.to_coo(&csc).unwrap();
        assert_eq!(coo.format, SparseFormat::COO);
        assert_eq!(coo.to_dense(), csc.to_dense());
    }

    #[test]
    fn test_convert_csr_to_csr_noop() {
        let csr = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let conv = SparseConverter::new();
        let csr2 = conv.to_csr(&csr).unwrap();
        assert_eq!(csr2.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_convert_roundtrip_csr_coo_csr() {
        let original = SparseMatrix::new_csr(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let conv = SparseConverter::new();
        let coo = conv.to_coo(&original).unwrap();
        let back = conv.to_csr(&coo).unwrap();
        assert_eq!(back.to_dense(), original.to_dense());
    }

    #[test]
    fn test_convert_roundtrip_csr_csc_csr() {
        let original = SparseMatrix::new_csr(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&original).unwrap();
        let back = conv.to_csr(&csc).unwrap();
        assert_eq!(back.to_dense(), original.to_dense());
    }

    #[test]
    fn test_convert_roundtrip_coo_csc_coo() {
        let original =
            SparseMatrix::new_coo(2, 2, vec![0, 1, 1], vec![0, 0, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&original).unwrap();
        let back = conv.to_coo(&csc).unwrap();
        assert_eq!(back.to_dense(), original.to_dense());
    }

    #[test]
    fn test_convert_empty_coo_to_csr() {
        let coo = SparseMatrix::new_coo(3, 3, vec![], vec![], vec![]).unwrap();
        let conv = SparseConverter::new();
        let csr = conv.to_csr(&coo).unwrap();
        assert_eq!(csr.nnz, 0);
        assert_eq!(csr.to_dense(), vec![0.0; 9]);
    }

    #[test]
    fn test_convert_unsupported_format() {
        let m = SparseMatrix::empty(2, 2, SparseFormat::BSR);
        let conv = SparseConverter::new();
        assert!(conv.to_csr(&m).is_err());
        assert!(conv.to_csc(&m).is_err());
        assert!(conv.to_coo(&m).is_err());
    }

    // -----------------------------------------------------------------------
    // SparseAnalyzer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyzer_basic() {
        let m = SparseMatrix::new_csr(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert_eq!(stats.rows, 3);
        assert_eq!(stats.cols, 3);
        assert_eq!(stats.nnz, 5);
        assert!((stats.density - 5.0 / 9.0).abs() < 1e-9);
        assert_eq!(stats.max_nnz_per_row, 2);
        assert_eq!(stats.min_nnz_per_row, 1);
    }

    #[test]
    fn test_analyzer_symmetric() {
        // Symmetric pattern: (0,1), (1,0)
        let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert!(stats.is_symmetric);
    }

    #[test]
    fn test_analyzer_not_symmetric() {
        let m = SparseMatrix::new_csr(2, 2, vec![0, 1, 1], vec![1], vec![1.0]).unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert!(!stats.is_symmetric);
    }

    #[test]
    fn test_analyzer_rectangular_not_symmetric() {
        let m = SparseMatrix::new_csr(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert!(!stats.is_symmetric);
    }

    #[test]
    fn test_analyzer_empty() {
        let m = SparseMatrix::new_csr(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert_eq!(stats.nnz, 0);
        assert_eq!(stats.density, 0.0);
        assert_eq!(stats.max_nnz_per_row, 0);
        assert_eq!(stats.min_nnz_per_row, 0);
    }

    #[test]
    fn test_analyzer_identity() {
        let mut row_ptr = vec![0usize; 5];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        for i in 0..4 {
            col_idx.push(i);
            vals.push(1.0);
            row_ptr[i + 1] = i + 1;
        }
        let m = SparseMatrix::new_csr(4, 4, row_ptr, col_idx, vals).unwrap();
        let stats = SparseAnalyzer::new().analyze(&m).unwrap();
        assert!(stats.is_symmetric);
        assert_eq!(stats.avg_nnz_per_row, 1.0);
    }

    #[test]
    fn test_analyzer_block_structure() {
        // 4×4 with 2×2 blocks filled at (0,0) and (1,1)
        let data =
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let m = SparseMatrix::from_dense_csr(4, 4, &data);
        let stats = SparseAnalyzer::new().analyze_with_blocks(&m, 2).unwrap();
        assert!(stats.has_block_structure);
        assert_eq!(stats.block_size, 2);
    }

    #[test]
    fn test_analyzer_no_block_structure() {
        let data =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let m = SparseMatrix::from_dense_csr(4, 4, &data);
        let stats = SparseAnalyzer::new().analyze_with_blocks(&m, 2).unwrap();
        assert!(!stats.has_block_structure);
    }

    #[test]
    fn test_analyzer_wrong_format() {
        let m = SparseMatrix::empty(2, 2, SparseFormat::COO);
        assert!(SparseAnalyzer::new().analyze(&m).is_err());
    }

    // -----------------------------------------------------------------------
    // SparseEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_spmm_from_coo() {
        let coo = SparseMatrix::new_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 1.0]).unwrap();
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let engine = SparseEngine::new();
        let c = engine.spmm(&coo, &b, 2).unwrap();
        assert_eq!(c, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_engine_spmm_from_csc() {
        let csc = SparseMatrix::new_csc(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 3.0]).unwrap();
        let b = vec![1.0, 1.0];
        let engine = SparseEngine::new();
        let c = engine.spmm(&csc, &b, 1).unwrap();
        assert_eq!(c, vec![2.0, 3.0]);
    }

    #[test]
    fn test_engine_softmax_auto_convert() {
        let mut coo =
            SparseMatrix::new_coo(1, 3, vec![0, 0, 0], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
        let engine = SparseEngine::new();
        engine.softmax_inplace(&mut coo).unwrap();
        assert_eq!(coo.format, SparseFormat::CSR);
        let sum: f32 = coo.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_engine_analyze_from_coo() {
        let coo = SparseMatrix::new_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let engine = SparseEngine::new();
        let stats = engine.analyze(&coo).unwrap();
        assert_eq!(stats.nnz, 2);
        assert_eq!(stats.rows, 2);
    }

    #[test]
    fn test_engine_sparsify_topk() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0];
        let engine = SparseEngine::new();
        let m = engine.sparsify_topk(&data, 2, 3, 1);
        assert_eq!(m.nnz, 2);
    }

    #[test]
    fn test_engine_convert_csr_to_csc() {
        let csr = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let engine = SparseEngine::new();
        let csc = engine.convert(&csr, SparseFormat::CSC).unwrap();
        assert_eq!(csc.format, SparseFormat::CSC);
        assert_eq!(csc.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_engine_convert_to_coo() {
        let csr = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let engine = SparseEngine::new();
        let coo = engine.convert(&csr, SparseFormat::COO).unwrap();
        assert_eq!(coo.format, SparseFormat::COO);
        assert_eq!(coo.to_dense(), csr.to_dense());
    }

    #[test]
    fn test_engine_convert_unsupported() {
        let csr = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
        let engine = SparseEngine::new();
        assert!(engine.convert(&csr, SparseFormat::BSR).is_err());
    }

    #[test]
    fn test_engine_accessors() {
        let engine = SparseEngine::new();
        let _ = engine.converter();
        let _ = engine.matmul();
        let _ = engine.softmax();
        let _ = engine.analyzer();
    }

    #[test]
    fn test_engine_default() {
        let engine = SparseEngine::default();
        let csr = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let c = engine.spmm(&csr, &[2.0], 1).unwrap();
        assert_eq!(c, vec![2.0]);
    }

    // -----------------------------------------------------------------------
    // Numerical accuracy tests (sparse vs dense reference)
    // -----------------------------------------------------------------------

    #[test]
    fn test_accuracy_matmul_random_like() {
        // A reproducible "pseudo-random" 4×4 sparse matrix
        let dense_a =
            vec![0.5, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0, 0.1, 0.2, 0.0, 0.0, 0.4, 0.0, 0.6, 0.8, 0.0];
        let dense_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ref_c = SparseMatMul::dense_matmul(&dense_a, &dense_b, 4, 4, 2);
        let sparse_a = SparseMatrix::from_dense_csr(4, 4, &dense_a);
        let mm = SparseMatMul::new();
        let sparse_c = mm.mul_csr_dense(&sparse_a, &dense_b, 2).unwrap();
        assert!(approx_eq(&sparse_c, &ref_c, 1e-5));
    }

    #[test]
    fn test_accuracy_softmax_sparse_vs_dense() {
        // Sparse softmax should match dense when all positions are stored
        let data = vec![0.1, 0.5, 0.9, 0.2, 0.8, 0.4];
        let dense_result = SparseSoftmax::dense_softmax(&data, 2, 3);
        let mut sparse = SparseMatrix::from_dense_csr(2, 3, &data);
        SparseSoftmax::new().softmax_csr_inplace(&mut sparse).unwrap();
        assert!(approx_eq(&sparse.to_dense(), &dense_result, 1e-6));
    }

    #[test]
    fn test_accuracy_attention_output_finite() {
        let attn = SparseAttention::new(SparseAttentionConfig { local_window: 4, stride: 2 });
        let seq = 8;
        let dim = 4;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let out = attn.apply(&q, &k, &v, seq, dim).unwrap();
        assert_eq!(out.len(), n);
        for o in &out {
            assert!(o.is_finite(), "non-finite output: {o}");
        }
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_zero_rows() {
        let m = SparseMatrix::new_csr(0, 5, vec![0], vec![], vec![]).unwrap();
        assert_eq!(m.nnz, 0);
        assert!(m.to_dense().is_empty());
    }

    #[test]
    fn test_edge_zero_cols() {
        let m = SparseMatrix::new_csr(5, 0, vec![0, 0, 0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(m.nnz, 0);
        assert!(m.to_dense().is_empty());
    }

    #[test]
    fn test_edge_1x1_operations() {
        let m = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![3.0]).unwrap();
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&m, &[4.0], 1).unwrap();
        assert_eq!(c, vec![12.0]);
    }

    #[test]
    fn test_edge_diagonal_matrix() {
        let n = 5;
        let mut rp = vec![0usize; n + 1];
        let mut ci = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            ci.push(i);
            vals.push((i + 1) as f32);
            rp[i + 1] = i + 1;
        }
        let m = SparseMatrix::new_csr(n, n, rp, ci, vals).unwrap();
        assert_eq!(m.nnz, n);
        assert!((m.density() - n as f64 / (n * n) as f64).abs() < 1e-9);
        let d = m.to_dense();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(d[i * n + j], (i + 1) as f32);
                } else {
                    assert_eq!(d[i * n + j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_edge_full_dense_as_sparse() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = SparseMatrix::from_dense_csr(3, 3, &data);
        assert_eq!(m.nnz, 9);
        assert!((m.density() - 1.0).abs() < 1e-9);
        assert!(approx_eq(&m.to_dense(), &data, 1e-6));
    }

    #[test]
    fn test_edge_large_conversion_roundtrip() {
        let n = 20;
        let mut data = vec![0.0f32; n * n];
        // Diagonal + some off-diagonal
        for i in 0..n {
            data[i * n + i] = (i + 1) as f32;
            if i + 1 < n {
                data[i * n + i + 1] = 0.5;
            }
        }
        let csr = SparseMatrix::from_dense_csr(n, n, &data);
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&csr).unwrap();
        let coo = conv.to_coo(&csc).unwrap();
        let back = conv.to_csr(&coo).unwrap();
        assert!(approx_eq(&back.to_dense(), &data, 1e-6));
    }

    #[test]
    fn test_edge_topk_single_row() {
        let data = vec![10.0, 1.0, 5.0, 3.0, 8.0];
        let sp = TopKSparsifier::new(3);
        let m = sp.sparsify(&data, 1, 5);
        // Top-3 by magnitude: 10.0, 8.0, 5.0
        let d = m.to_dense();
        assert_eq!(d[0], 10.0);
        assert_eq!(d[2], 5.0);
        assert_eq!(d[4], 8.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[3], 0.0);
    }

    #[test]
    fn test_edge_matmul_sparse_times_zero_b() {
        let a = SparseMatrix::new_csr(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0, 3.0]).unwrap();
        let b = vec![0.0; 4];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&a, &b, 2).unwrap();
        assert_eq!(c, vec![0.0; 4]);
    }

    #[test]
    fn test_edge_block_sparse_non_aligned() {
        // 5×5 matrix with 2×2 blocks (not evenly divisible)
        let data = vec![
            1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ];
        let bs = BlockSparseFormat::from_dense(5, 5, 2, 2, &data, 0.0);
        let recon = bs.to_dense();
        assert!(approx_eq(&recon, &data, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Default trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_sparse_matmul() {
        let mm = SparseMatMul::default();
        let eye = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        assert_eq!(mm.mul_csr_dense(&eye, &[7.0], 1).unwrap(), vec![7.0]);
    }

    #[test]
    fn test_default_sparse_softmax() {
        let sm = SparseSoftmax::default();
        let mut m = SparseMatrix::new_csr(1, 1, vec![0, 1], vec![0], vec![5.0]).unwrap();
        sm.softmax_csr_inplace(&mut m).unwrap();
        assert!((m.values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_sparse_converter() {
        let _ = SparseConverter::default();
    }

    #[test]
    fn test_default_sparse_analyzer() {
        let _ = SparseAnalyzer::default();
    }

    // -----------------------------------------------------------------------
    // SparseError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_error_display() {
        let e = SparseError("test error".into());
        assert_eq!(e.to_string(), "SparseError: test error");
    }

    #[test]
    fn test_sparse_error_clone() {
        let e = SparseError("msg".into());
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[test]
    fn test_sparse_error_is_error() {
        let e = SparseError("x".into());
        let _: &dyn std::error::Error = &e;
    }

    // -----------------------------------------------------------------------
    // Additional matmul accuracy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_rectangular_tall() {
        // 3×2 sparse × 2×4 dense
        let dense_a = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0];
        let sparse_a = SparseMatrix::from_dense_csr(3, 2, &dense_a);
        let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&sparse_a, &b, 4).unwrap();
        let ref_c = SparseMatMul::dense_matmul(&dense_a, &b, 3, 2, 4);
        assert!(approx_eq(&c, &ref_c, 1e-6));
    }

    #[test]
    fn test_matmul_rectangular_wide() {
        // 2×4 sparse × 4×1 dense
        let dense_a = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let sparse_a = SparseMatrix::from_dense_csr(2, 4, &dense_a);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mm = SparseMatMul::new();
        let c = mm.mul_csr_dense(&sparse_a, &b, 1).unwrap();
        let ref_c = SparseMatMul::dense_matmul(&dense_a, &b, 2, 4, 1);
        assert!(approx_eq(&c, &ref_c, 1e-6));
    }

    // -----------------------------------------------------------------------
    // Conversion with duplicates / edge patterns
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_single_element_roundtrip() {
        let coo = SparseMatrix::new_coo(3, 3, vec![1], vec![2], vec![42.0]).unwrap();
        let conv = SparseConverter::new();
        let csr = conv.to_csr(&coo).unwrap();
        let csc = conv.to_csc(&csr).unwrap();
        let back = conv.to_coo(&csc).unwrap();
        assert_eq!(back.to_dense(), coo.to_dense());
    }

    #[test]
    fn test_convert_all_in_one_row() {
        let coo =
            SparseMatrix::new_coo(3, 3, vec![0, 0, 0], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
        let conv = SparseConverter::new();
        let csr = conv.to_csr(&coo).unwrap();
        assert_eq!(csr.row_indices[0], 0);
        assert_eq!(csr.row_indices[1], 3);
        assert_eq!(csr.row_indices[2], 3);
        assert_eq!(csr.to_dense(), coo.to_dense());
    }

    #[test]
    fn test_convert_all_in_one_column() {
        let coo =
            SparseMatrix::new_coo(3, 3, vec![0, 1, 2], vec![1, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let conv = SparseConverter::new();
        let csc = conv.to_csc(&coo).unwrap();
        assert_eq!(csc.to_dense(), coo.to_dense());
    }

    // -----------------------------------------------------------------------
    // Integration-style tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_dense_to_sparse_matmul_to_dense() {
        let dense_a = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let dense_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (m, k, n) = (3, 4, 2);

        let sparse_a = SparseMatrix::from_dense_csr(m, k, &dense_a);
        let engine = SparseEngine::new();
        let result = engine.spmm(&sparse_a, &dense_b, n).unwrap();
        let reference = SparseMatMul::dense_matmul(&dense_a, &dense_b, m, k, n);
        assert!(approx_eq(&result, &reference, 1e-5));
    }

    #[test]
    fn test_pipeline_topk_then_matmul() {
        let dense_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let engine = SparseEngine::new();
        let sparse = engine.sparsify_topk(&dense_a, 3, 3, 2);
        let b = vec![1.0; 3];
        let c = engine.spmm(&sparse, &b, 1).unwrap();
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn test_pipeline_convert_analyze() {
        let coo = SparseMatrix::new_coo(
            4,
            4,
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let engine = SparseEngine::new();
        let stats = engine.analyze(&coo).unwrap();
        assert_eq!(stats.nnz, 4);
        assert!(stats.is_symmetric);
    }
}
