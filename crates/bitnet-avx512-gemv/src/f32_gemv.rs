//! Dense single-precision GEMV: **y = A · x**.
//!
//! `A` is a row-major `(rows × cols)` matrix stored as a flat `&[f32]` slice,
//! and `x` is a column vector of length `cols`.

use crate::dot;

/// Parameters for a single-precision GEMV operation.
///
/// Borrows the matrix and vector data without copying.
#[derive(Debug, Clone, Copy)]
pub struct GemvParams<'a> {
    rows: usize,
    cols: usize,
    matrix: &'a [f32],
    vector: &'a [f32],
}

impl<'a> GemvParams<'a> {
    /// Create a new parameter set.
    ///
    /// # Panics
    ///
    /// Panics if `matrix.len() != rows * cols` or `vector.len() != cols`.
    #[must_use]
    pub fn new(rows: usize, cols: usize, matrix: &'a [f32], vector: &'a [f32]) -> Self {
        assert_eq!(
            matrix.len(),
            rows * cols,
            "matrix length {} != rows({}) * cols({})",
            matrix.len(),
            rows,
            cols,
        );
        assert_eq!(vector.len(), cols, "vector length {} != cols({})", vector.len(), cols,);
        Self { rows, cols, matrix, vector }
    }

    /// Number of output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (inner dimension).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

/// Compute **y = A · x** using the best available instruction set.
///
/// Returns a `Vec<f32>` of length `params.rows()`.
#[must_use]
pub fn gemv(params: &GemvParams<'_>) -> Vec<f32> {
    let mut out = Vec::with_capacity(params.rows);
    for r in 0..params.rows {
        let row_start = r * params.cols;
        let row = &params.matrix[row_start..row_start + params.cols];
        out.push(dot::dot_f32(row, params.vector));
    }
    out
}

/// Scalar-only GEMV (useful for testing parity).
#[must_use]
pub fn gemv_scalar(params: &GemvParams<'_>) -> Vec<f32> {
    let mut out = Vec::with_capacity(params.rows);
    for r in 0..params.rows {
        let row_start = r * params.cols;
        let row = &params.matrix[row_start..row_start + params.cols];
        out.push(dot::dot_f32_scalar(row, params.vector));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix() {
        #[rustfmt::skip]
        let eye = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let v = vec![7.0, 8.0, 9.0];
        let p = GemvParams::new(3, 3, &eye, &v);
        let r = gemv(&p);
        assert_eq!(r, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn single_row() {
        let m = vec![2.0, 3.0];
        let v = vec![4.0, 5.0];
        let p = GemvParams::new(1, 2, &m, &v);
        assert_eq!(gemv(&p), vec![23.0]);
    }

    #[test]
    fn single_col() {
        let m = vec![2.0, 3.0, 4.0];
        let v = vec![5.0];
        let p = GemvParams::new(3, 1, &m, &v);
        assert_eq!(gemv(&p), vec![10.0, 15.0, 20.0]);
    }

    #[test]
    fn zero_matrix() {
        let m = vec![0.0_f32; 6];
        let v = vec![1.0, 2.0, 3.0];
        let p = GemvParams::new(2, 3, &m, &v);
        assert_eq!(gemv(&p), vec![0.0, 0.0]);
    }

    #[test]
    fn zero_vector() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![0.0, 0.0, 0.0];
        let p = GemvParams::new(2, 3, &m, &v);
        assert_eq!(gemv(&p), vec![0.0, 0.0]);
    }

    #[test]
    fn negative_values() {
        let m = vec![-1.0, -2.0, -3.0, -4.0];
        let v = vec![1.0, -1.0];
        let p = GemvParams::new(2, 2, &m, &v);
        assert_eq!(gemv(&p), vec![1.0, 1.0]);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn large_matrix_32x32() {
        let n = 32;
        let m: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32).collect();
        let v: Vec<f32> = vec![1.0; n];
        let p = GemvParams::new(n, n, &m, &v);
        let r = gemv(&p);
        assert_eq!(r.len(), n);
        // Each row's sum should match the scalar computation.
        let rs = gemv_scalar(&p);
        for (a, b) in r.iter().zip(rs.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn large_matrix_unaligned_33x17() {
        let (rows, cols) = (33, 17);
        let m: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();
        let p = GemvParams::new(rows, cols, &m, &v);
        let r = gemv(&p);
        let rs = gemv_scalar(&p);
        assert_eq!(r.len(), rows);
        for (a, b) in r.iter().zip(rs.iter()) {
            assert!((a - b).abs() < 1e-1, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn scalar_matches_dispatch() {
        let m: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();
        let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3).collect();
        let p = GemvParams::new(8, 16, &m, &v);
        let dispatched = gemv(&p);
        let scalar = gemv_scalar(&p);
        for (a, b) in dispatched.iter().zip(scalar.iter()) {
            assert!((a - b).abs() < 1e-2, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    #[should_panic(expected = "matrix length")]
    fn bad_matrix_length() {
        let m = vec![1.0_f32; 5]; // wrong length for 2×3
        let v = vec![1.0; 3];
        let _ = GemvParams::new(2, 3, &m, &v);
    }

    #[test]
    #[should_panic(expected = "vector length")]
    fn bad_vector_length() {
        let m = vec![1.0_f32; 6];
        let v = vec![1.0; 2]; // wrong length
        let _ = GemvParams::new(2, 3, &m, &v);
    }

    #[test]
    fn zero_rows() {
        let m: Vec<f32> = vec![];
        let v = vec![1.0, 2.0];
        let p = GemvParams::new(0, 2, &m, &v);
        assert!(gemv(&p).is_empty());
    }
}
