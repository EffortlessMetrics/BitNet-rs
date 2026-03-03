//! Quantised 8-bit integer GEMV: **y = A · x** where A and x are `i8`.
//!
//! Accumulates into `i32` to avoid overflow.

use crate::dot;

/// Parameters for an `i8` GEMV operation.
#[derive(Debug, Clone, Copy)]
pub struct I8GemvParams<'a> {
    rows: usize,
    cols: usize,
    matrix: &'a [i8],
    vector: &'a [i8],
}

impl<'a> I8GemvParams<'a> {
    /// Create a new parameter set.
    ///
    /// # Panics
    ///
    /// Panics if `matrix.len() != rows * cols` or `vector.len() != cols`.
    #[must_use]
    pub fn new(rows: usize, cols: usize, matrix: &'a [i8], vector: &'a [i8]) -> Self {
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
/// Returns a `Vec<i32>` of length `params.rows()`.
#[must_use]
pub fn gemv_i8(params: &I8GemvParams<'_>) -> Vec<i32> {
    let mut out = Vec::with_capacity(params.rows);
    for r in 0..params.rows {
        let start = r * params.cols;
        let row = &params.matrix[start..start + params.cols];
        out.push(dot::dot_i8(row, params.vector));
    }
    out
}

/// Scalar-only `i8` GEMV (useful for testing parity).
#[must_use]
pub fn gemv_i8_scalar(params: &I8GemvParams<'_>) -> Vec<i32> {
    let mut out = Vec::with_capacity(params.rows);
    for r in 0..params.rows {
        let start = r * params.cols;
        let row = &params.matrix[start..start + params.cols];
        out.push(dot::dot_i8_scalar(row, params.vector));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_i8() {
        #[rustfmt::skip]
        let eye: Vec<i8> = vec![
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ];
        let v: Vec<i8> = vec![7, 8, 9];
        let p = I8GemvParams::new(3, 3, &eye, &v);
        assert_eq!(gemv_i8(&p), vec![7, 8, 9]);
    }

    #[test]
    fn single_row_i8() {
        let m: Vec<i8> = vec![2, 3];
        let v: Vec<i8> = vec![4, 5];
        let p = I8GemvParams::new(1, 2, &m, &v);
        assert_eq!(gemv_i8(&p), vec![23]);
    }

    #[test]
    fn single_col_i8() {
        let m: Vec<i8> = vec![2, 3, 4];
        let v: Vec<i8> = vec![5];
        let p = I8GemvParams::new(3, 1, &m, &v);
        assert_eq!(gemv_i8(&p), vec![10, 15, 20]);
    }

    #[test]
    fn zero_matrix_i8() {
        let m: Vec<i8> = vec![0; 6];
        let v: Vec<i8> = vec![1, 2, 3];
        let p = I8GemvParams::new(2, 3, &m, &v);
        assert_eq!(gemv_i8(&p), vec![0, 0]);
    }

    #[test]
    fn negative_values_i8() {
        let m: Vec<i8> = vec![-1, -2, -3, -4];
        let v: Vec<i8> = vec![1, -1];
        let p = I8GemvParams::new(2, 2, &m, &v);
        assert_eq!(gemv_i8(&p), vec![1, 1]);
    }

    #[test]
    fn max_values_i8() {
        let m: Vec<i8> = vec![127; 4];
        let v: Vec<i8> = vec![127; 2];
        let p = I8GemvParams::new(2, 2, &m, &v);
        assert_eq!(gemv_i8(&p), vec![127 * 127 * 2; 2]);
    }

    #[test]
    fn min_values_i8() {
        let m: Vec<i8> = vec![-128; 4];
        let v: Vec<i8> = vec![1; 2];
        let p = I8GemvParams::new(2, 2, &m, &v);
        assert_eq!(gemv_i8(&p), vec![-256; 2]);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn large_i8_32x32() {
        let n = 32;
        let m: Vec<i8> = (0..n * n).map(|i| (i % 11) as i8 - 5).collect();
        let v: Vec<i8> = (0..n).map(|i| (i % 7) as i8 - 3).collect();
        let p = I8GemvParams::new(n, n, &m, &v);
        let r = gemv_i8(&p);
        let rs = gemv_i8_scalar(&p);
        assert_eq!(r, rs);
    }

    #[test]
    fn large_unaligned_33x17() {
        let (rows, cols) = (33, 17);
        let m: Vec<i8> = (0..rows * cols).map(|i| (i % 9) as i8 - 4).collect();
        let v: Vec<i8> = (0..cols).map(|i| (i % 5) as i8 - 2).collect();
        let p = I8GemvParams::new(rows, cols, &m, &v);
        assert_eq!(gemv_i8(&p), gemv_i8_scalar(&p));
    }

    #[test]
    fn scalar_matches_dispatch_i8() {
        let m: Vec<i8> = (0..128).map(|i| ((i % 127) - 63) as i8).collect();
        let v: Vec<i8> = (0..16).map(|i| ((i % 11) - 5) as i8).collect();
        let p = I8GemvParams::new(8, 16, &m, &v);
        assert_eq!(gemv_i8(&p), gemv_i8_scalar(&p));
    }

    #[test]
    #[should_panic(expected = "matrix length")]
    fn bad_matrix_length_i8() {
        let m: Vec<i8> = vec![1; 5];
        let v: Vec<i8> = vec![1; 3];
        let _ = I8GemvParams::new(2, 3, &m, &v);
    }

    #[test]
    #[should_panic(expected = "vector length")]
    fn bad_vector_length_i8() {
        let m: Vec<i8> = vec![1; 6];
        let v: Vec<i8> = vec![1; 2];
        let _ = I8GemvParams::new(2, 3, &m, &v);
    }

    #[test]
    fn zero_rows_i8() {
        let m: Vec<i8> = vec![];
        let v: Vec<i8> = vec![1, 2];
        let p = I8GemvParams::new(0, 2, &m, &v);
        assert!(gemv_i8(&p).is_empty());
    }
}
