//! Sparse tensor operations exploiting BitNet weight sparsity.
//!
//! BitNet ternary weights ({-1, 0, 1}) exhibit high sparsity — typically
//! 30–70% zeros. This module provides CSR (Compressed Sparse Row)
//! representation and sparse matmul kernels that skip zero-weight
//! computations entirely, yielding significant speedups on GPU backends.

use std::fmt;

// ---------------------------------------------------------------------------
// CSR representation
// ---------------------------------------------------------------------------

/// Compressed Sparse Row (CSR) representation for GPU-friendly sparse matrices.
///
/// Stores only non-zero elements, row pointers, and column indices.
/// Optimised for ternary BitNet weights where values ∈ {-1, 0, 1}.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Number of rows in the original dense matrix.
    pub rows: usize,
    /// Number of columns in the original dense matrix.
    pub cols: usize,
    /// Row pointer array of length `rows + 1`.
    /// `row_ptr[i]..row_ptr[i+1]` indexes into `col_indices` / `values`.
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero element.
    pub col_indices: Vec<usize>,
    /// Non-zero values (typically -1.0 or 1.0 for ternary weights).
    pub values: Vec<f32>,
}

impl CsrMatrix {
    /// Build a CSR matrix from a dense row-major `rows × cols` buffer.
    ///
    /// Elements whose absolute value is below `zero_threshold` are treated as
    /// zero and omitted from the sparse representation.
    pub fn from_dense(dense: &[f32], rows: usize, cols: usize, zero_threshold: f32) -> Self {
        assert_eq!(
            dense.len(),
            rows * cols,
            "dense buffer length must equal rows * cols"
        );

        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptr.push(0);
        for r in 0..rows {
            for c in 0..cols {
                let v = dense[r * cols + c];
                if v.abs() > zero_threshold {
                    col_indices.push(c);
                    values.push(v);
                }
            }
            row_ptr.push(col_indices.len());
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    /// Build a CSR matrix from a dense ternary weight tensor.
    ///
    /// This is a convenience wrapper that uses `0.5` as the zero threshold,
    /// which correctly classifies ternary values {-1, 0, 1}.
    pub fn from_ternary_weights(weights: &[f32], rows: usize, cols: usize) -> Self {
        Self::from_dense(weights, rows, cols, 0.5)
    }

    /// Number of stored (non-zero) elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio in `[0.0, 1.0]` — fraction of elements that are zero.
    pub fn sparsity(&self) -> f64 {
        let total = (self.rows * self.cols) as f64;
        if total == 0.0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / total)
    }

    /// Reconstruct a dense row-major buffer from the CSR representation.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                let c = self.col_indices[idx];
                out[r * self.cols + c] = self.values[idx];
            }
        }
        out
    }

    /// Returns `true` when all stored values are in {-1, 1} (ternary).
    pub fn is_ternary(&self) -> bool {
        self.values
            .iter()
            .all(|&v| (v - 1.0).abs() < f32::EPSILON || (v + 1.0).abs() < f32::EPSILON)
    }

    /// Per-row non-zero counts.
    pub fn row_nnz(&self) -> Vec<usize> {
        (0..self.rows)
            .map(|r| self.row_ptr[r + 1] - self.row_ptr[r])
            .collect()
    }
}

impl fmt::Display for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsrMatrix({}×{}, nnz={}, sparsity={:.1}%)",
            self.rows,
            self.cols,
            self.nnz(),
            self.sparsity() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Sparse matrix–vector multiply (SpMV) — ternary-optimised
// ---------------------------------------------------------------------------

/// Sparse matrix–vector multiply: `y = A · x` where `A` is in CSR format.
///
/// For ternary weights the multiply is replaced with addition / subtraction,
/// avoiding FP multiply entirely.
pub fn spmv(csr: &CsrMatrix, x: &[f32]) -> Vec<f32> {
    assert_eq!(
        x.len(),
        csr.cols,
        "input vector length must match matrix columns"
    );

    let mut y = vec![0.0f32; csr.rows];

    if csr.is_ternary() {
        // Fast path: values are ±1 — replace mul with add/sub.
        for r in 0..csr.rows {
            let mut acc = 0.0f32;
            let start = csr.row_ptr[r];
            let end = csr.row_ptr[r + 1];
            for idx in start..end {
                let c = csr.col_indices[idx];
                if csr.values[idx] > 0.0 {
                    acc += x[c];
                } else {
                    acc -= x[c];
                }
            }
            y[r] = acc;
        }
    } else {
        // General sparse path.
        for r in 0..csr.rows {
            let mut acc = 0.0f32;
            let start = csr.row_ptr[r];
            let end = csr.row_ptr[r + 1];
            for idx in start..end {
                let c = csr.col_indices[idx];
                acc += csr.values[idx] * x[c];
            }
            y[r] = acc;
        }
    }

    y
}

/// Sparse matrix–matrix multiply: `C = A · B` (CSR × dense row-major).
///
/// `B` is `(A.cols × b_cols)` row-major dense. Result `C` is `(A.rows × b_cols)`.
pub fn spmm(csr: &CsrMatrix, b: &[f32], b_cols: usize) -> Vec<f32> {
    assert_eq!(
        b.len(),
        csr.cols * b_cols,
        "B dimensions must be compatible with A"
    );

    let mut c = vec![0.0f32; csr.rows * b_cols];

    if csr.is_ternary() {
        for r in 0..csr.rows {
            let start = csr.row_ptr[r];
            let end = csr.row_ptr[r + 1];
            for idx in start..end {
                let col_a = csr.col_indices[idx];
                let sign = csr.values[idx];
                let row_offset_c = r * b_cols;
                let row_offset_b = col_a * b_cols;
                if sign > 0.0 {
                    for j in 0..b_cols {
                        c[row_offset_c + j] += b[row_offset_b + j];
                    }
                } else {
                    for j in 0..b_cols {
                        c[row_offset_c + j] -= b[row_offset_b + j];
                    }
                }
            }
        }
    } else {
        for r in 0..csr.rows {
            let start = csr.row_ptr[r];
            let end = csr.row_ptr[r + 1];
            for idx in start..end {
                let col_a = csr.col_indices[idx];
                let val = csr.values[idx];
                let row_offset_c = r * b_cols;
                let row_offset_b = col_a * b_cols;
                for j in 0..b_cols {
                    c[row_offset_c + j] += val * b[row_offset_b + j];
                }
            }
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Sparsity analysis & reporting
// ---------------------------------------------------------------------------

/// Summary statistics for a weight tensor's sparsity.
#[derive(Debug, Clone)]
pub struct SparsityReport {
    /// Name / label of the tensor (e.g. layer identifier).
    pub name: String,
    /// Total number of elements.
    pub total_elements: usize,
    /// Number of zero elements (below threshold).
    pub zero_elements: usize,
    /// Number of positive-one elements.
    pub positive_one_count: usize,
    /// Number of negative-one elements.
    pub negative_one_count: usize,
    /// Number of non-ternary non-zero elements.
    pub other_nonzero_count: usize,
    /// Sparsity ratio `[0.0, 1.0]`.
    pub sparsity: f64,
    /// Whether all non-zero values are ternary.
    pub is_ternary: bool,
}

impl fmt::Display for SparsityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} elems, {:.1}% sparse, ternary={}",
            self.name,
            self.total_elements,
            self.sparsity * 100.0,
            self.is_ternary,
        )
    }
}

/// Analyse a dense weight tensor and produce a [`SparsityReport`].
pub fn analyze_sparsity(name: &str, weights: &[f32], zero_threshold: f32) -> SparsityReport {
    let total = weights.len();
    let mut zeros = 0usize;
    let mut pos1 = 0usize;
    let mut neg1 = 0usize;
    let mut other = 0usize;

    for &v in weights {
        if v.abs() <= zero_threshold {
            zeros += 1;
        } else if (v - 1.0).abs() < f32::EPSILON {
            pos1 += 1;
        } else if (v + 1.0).abs() < f32::EPSILON {
            neg1 += 1;
        } else {
            other += 1;
        }
    }

    SparsityReport {
        name: name.to_string(),
        total_elements: total,
        zero_elements: zeros,
        positive_one_count: pos1,
        negative_one_count: neg1,
        other_nonzero_count: other,
        sparsity: if total == 0 {
            0.0
        } else {
            zeros as f64 / total as f64
        },
        is_ternary: other == 0,
    }
}

/// Determine whether a sparse representation is beneficial for a tensor.
///
/// Returns `true` when the sparsity exceeds `min_sparsity` (default 30%).
pub fn should_use_sparse(weights: &[f32], zero_threshold: f32, min_sparsity: f64) -> bool {
    let total = weights.len();
    if total == 0 {
        return false;
    }
    let zeros = weights.iter().filter(|v| v.abs() <= zero_threshold).count();
    (zeros as f64 / total as f64) >= min_sparsity
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CSR construction & round-trip --

    #[test]
    fn test_csr_from_dense_identity() {
        // 3×3 identity matrix — two zeros per row
        let dense = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let csr = CsrMatrix::from_dense(&dense, 3, 3, 0.5);
        assert_eq!(csr.nnz(), 3);
        assert!((csr.sparsity() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(csr.to_dense(), dense);
    }

    #[test]
    fn test_csr_from_ternary_weights() {
        let weights = vec![
            1.0, -1.0, 0.0, //
            0.0, 1.0, -1.0, //
        ];
        let csr = CsrMatrix::from_ternary_weights(&weights, 2, 3);
        assert_eq!(csr.nnz(), 4);
        assert!(csr.is_ternary());
        assert_eq!(csr.to_dense(), weights);
    }

    #[test]
    fn test_csr_empty_matrix() {
        let dense = vec![0.0f32; 4];
        let csr = CsrMatrix::from_dense(&dense, 2, 2, 0.5);
        assert_eq!(csr.nnz(), 0);
        assert!((csr.sparsity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_csr_fully_dense() {
        let dense = vec![1.0, -1.0, 1.0, -1.0];
        let csr = CsrMatrix::from_dense(&dense, 2, 2, 0.5);
        assert_eq!(csr.nnz(), 4);
        assert!((csr.sparsity()).abs() < 1e-9);
    }

    #[test]
    fn test_csr_row_nnz() {
        let dense = vec![
            1.0, 0.0, -1.0, //
            0.0, 0.0, 0.0, //
            -1.0, 1.0, 1.0, //
        ];
        let csr = CsrMatrix::from_dense(&dense, 3, 3, 0.5);
        assert_eq!(csr.row_nnz(), vec![2, 0, 3]);
    }

    // -- SpMV --

    #[test]
    fn test_spmv_ternary() {
        // W = [[1, -1, 0], [0, 1, 1]], x = [2, 3, 4]
        // y = [2 - 3, 3 + 4] = [-1, 7]
        let w = vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
        let csr = CsrMatrix::from_ternary_weights(&w, 2, 3);
        let x = vec![2.0, 3.0, 4.0];
        let y = spmv(&csr, &x);
        assert_eq!(y, vec![-1.0, 7.0]);
    }

    #[test]
    fn test_spmv_general() {
        let dense = vec![2.0, 0.0, 0.0, 3.0];
        let csr = CsrMatrix::from_dense(&dense, 2, 2, 0.5);
        let x = vec![1.0, 2.0];
        let y = spmv(&csr, &x);
        assert!((y[0] - 2.0).abs() < 1e-6);
        assert!((y[1] - 6.0).abs() < 1e-6);
    }

    // -- SpMM --

    #[test]
    fn test_spmm_ternary() {
        // A(2×3) * B(3×2)
        let a = vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
        let csr = CsrMatrix::from_ternary_weights(&a, 2, 3);
        let b = vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ];
        let c = spmm(&csr, &b, 2);
        // row0: 1*[1,2] + (-1)*[3,4] = [-2, -2]
        // row1: 1*[3,4] + 1*[5,6]    = [8, 10]
        assert_eq!(c, vec![-2.0, -2.0, 8.0, 10.0]);
    }

    // -- Sparsity analysis --

    #[test]
    fn test_analyze_sparsity_ternary() {
        let w = vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0];
        let report = analyze_sparsity("layer0.weight", &w, 0.5);
        assert_eq!(report.total_elements, 6);
        assert_eq!(report.zero_elements, 3);
        assert_eq!(report.positive_one_count, 2);
        assert_eq!(report.negative_one_count, 1);
        assert_eq!(report.other_nonzero_count, 0);
        assert!(report.is_ternary);
        assert!((report.sparsity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_should_use_sparse_high_sparsity() {
        let w = vec![0.0, 0.0, 0.0, 1.0]; // 75% sparse
        assert!(should_use_sparse(&w, 0.5, 0.3));
    }

    #[test]
    fn test_should_use_sparse_low_sparsity() {
        let w = vec![1.0, -1.0, 1.0, -1.0]; // 0% sparse
        assert!(!should_use_sparse(&w, 0.5, 0.3));
    }

    #[test]
    fn test_display_csr_matrix() {
        let dense = vec![1.0, 0.0, 0.0, -1.0];
        let csr = CsrMatrix::from_dense(&dense, 2, 2, 0.5);
        let s = format!("{csr}");
        assert!(s.contains("CsrMatrix(2×2"));
        assert!(s.contains("nnz=2"));
    }

    #[test]
    fn test_display_sparsity_report() {
        let w = vec![0.0; 8];
        let report = analyze_sparsity("test", &w, 0.5);
        let s = format!("{report}");
        assert!(s.contains("100.0% sparse"));
    }
}
