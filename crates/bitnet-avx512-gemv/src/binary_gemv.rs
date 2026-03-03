//! 1-bit packed binary GEMV: **y = `A_bin` · x**.
//!
//! The matrix `A` is stored in packed binary form — one bit per element, 8
//! elements per byte, MSB-first.  Each bit encodes a weight: `1 → +1`,
//! `0 → −1`.  The vector `x` is dense `f32`.

use crate::detect;

/// Parameters for a binary GEMV operation.
#[derive(Debug, Clone, Copy)]
pub struct BinaryGemvParams<'a> {
    rows: usize,
    cols: usize,
    packed_matrix: &'a [u8],
    vector: &'a [f32],
}

impl<'a> BinaryGemvParams<'a> {
    /// Create a new parameter set.
    ///
    /// `cols` must be a multiple of 8.  `packed_matrix` stores each row as
    /// `cols / 8` bytes.
    ///
    /// # Panics
    ///
    /// Panics if dimensions or buffer sizes are inconsistent.
    #[must_use]
    pub fn new(rows: usize, cols: usize, packed_matrix: &'a [u8], vector: &'a [f32]) -> Self {
        assert_eq!(cols % 8, 0, "cols ({cols}) must be a multiple of 8");
        let bytes_per_row = cols / 8;
        assert_eq!(
            packed_matrix.len(),
            rows * bytes_per_row,
            "packed_matrix length {} != rows({}) * bytes_per_row({})",
            packed_matrix.len(),
            rows,
            bytes_per_row,
        );
        assert_eq!(vector.len(), cols, "vector length {} != cols({})", vector.len(), cols,);
        Self { rows, cols, packed_matrix, vector }
    }

    /// Number of output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (inner dimension, always a multiple of 8).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

/// Compute **y = `A_bin` · x** using the best available instruction set.
///
/// Returns a `Vec<f32>` of length `params.rows()`.
#[must_use]
pub fn gemv_binary(params: &BinaryGemvParams<'_>) -> Vec<f32> {
    if detect::avx512_available() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: runtime detection passed above.
            return unsafe { gemv_binary_avx512(params) };
        }
    }
    gemv_binary_scalar(params)
}

/// Scalar binary GEMV.
#[must_use]
pub fn gemv_binary_scalar(params: &BinaryGemvParams<'_>) -> Vec<f32> {
    let bytes_per_row = params.cols / 8;
    let mut out = Vec::with_capacity(params.rows);
    for r in 0..params.rows {
        let row_start = r * bytes_per_row;
        let row_bytes = &params.packed_matrix[row_start..row_start + bytes_per_row];
        let mut acc = 0.0_f32;
        for (byte_idx, &byte) in row_bytes.iter().enumerate() {
            for bit in 0..8u32 {
                let col = byte_idx * 8 + bit as usize;
                let sign = if (byte >> (7 - bit)) & 1 == 1 { 1.0_f32 } else { -1.0_f32 };
                acc += sign * params.vector[col];
            }
        }
        out.push(acc);
    }
    out
}

/// AVX-512 accelerated binary GEMV.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gemv_binary_avx512(params: &BinaryGemvParams<'_>) -> Vec<f32> {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_setzero_ps,
    };

    unsafe {
        let bytes_per_row = params.cols / 8;
        let mut out = Vec::with_capacity(params.rows);

        let chunks16 = params.cols / 16;
        let remainder = params.cols % 16;

        for r in 0..params.rows {
            let row_start = r * bytes_per_row;
            let row_bytes = &params.packed_matrix[row_start..row_start + bytes_per_row];

            let mut signs = vec![0.0_f32; params.cols];
            for (byte_idx, &byte) in row_bytes.iter().enumerate() {
                for bit in 0..8u32 {
                    let col = byte_idx * 8 + bit as usize;
                    signs[col] = if (byte >> (7 - bit)) & 1 == 1 { 1.0 } else { -1.0 };
                }
            }

            let mut acc = _mm512_setzero_ps();
            let s_ptr = signs.as_ptr();
            let v_ptr = params.vector.as_ptr();

            for c in 0..chunks16 {
                let off = c * 16;
                let vs = _mm512_loadu_ps(s_ptr.add(off));
                let vv = _mm512_loadu_ps(v_ptr.add(off));
                acc = _mm512_fmadd_ps(vs, vv, acc);
            }

            let mut sum = _mm512_reduce_add_ps(acc);
            let tail_start = chunks16 * 16;
            for j in 0..remainder {
                sum += signs[tail_start + j] * params.vector[tail_start + j];
            }

            out.push(sum);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones_binary() {
        // 1 row, 8 cols, all bits = 1 → weights are all +1
        let packed = vec![0xFF_u8];
        let v = vec![1.0_f32; 8];
        let p = BinaryGemvParams::new(1, 8, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn all_zeros_binary() {
        // All bits = 0 → weights are all -1
        let packed = vec![0x00_u8];
        let v = vec![1.0_f32; 8];
        let p = BinaryGemvParams::new(1, 8, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - (-8.0)).abs() < 1e-6);
    }

    #[test]
    fn alternating_bits() {
        // 0b10101010 → +1,-1,+1,-1,+1,-1,+1,-1
        let packed = vec![0xAA_u8];
        let v = vec![1.0_f32; 8];
        let p = BinaryGemvParams::new(1, 8, &packed, &v);
        let r = gemv_binary(&p);
        // Sum of +1,-1,+1,-1,+1,-1,+1,-1 = 0
        assert!((r[0]).abs() < 1e-6);
    }

    #[test]
    fn two_rows_binary() {
        let packed = vec![0xFF_u8, 0x00];
        let v = vec![2.0_f32; 8];
        let p = BinaryGemvParams::new(2, 8, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - 16.0).abs() < 1e-6);
        assert!((r[1] - (-16.0)).abs() < 1e-6);
    }

    #[test]
    fn large_binary_16cols() {
        // 2 rows × 16 cols
        let packed = vec![0xFF, 0xFF, 0x00, 0x00];
        let v = vec![1.0_f32; 16];
        let p = BinaryGemvParams::new(2, 16, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - 16.0).abs() < 1e-6);
        assert!((r[1] - (-16.0)).abs() < 1e-6);
    }

    #[test]
    fn large_binary_64cols() {
        let rows = 4;
        let cols = 64;
        let bytes_per_row = cols / 8;
        let mut packed = vec![0xFF_u8; rows * bytes_per_row];
        // Row 2: all zeros
        for b in &mut packed[2 * bytes_per_row..3 * bytes_per_row] {
            *b = 0x00;
        }
        let v = vec![1.0_f32; cols];
        let p = BinaryGemvParams::new(rows, cols, &packed, &v);
        let r = gemv_binary(&p);
        assert!((r[0] - 64.0).abs() < 1e-5);
        assert!((r[1] - 64.0).abs() < 1e-5);
        assert!((r[2] - (-64.0)).abs() < 1e-5);
        assert!((r[3] - 64.0).abs() < 1e-5);
    }

    #[test]
    fn scalar_matches_dispatch_binary() {
        let rows = 8;
        let cols = 32;
        let bytes_per_row = cols / 8;
        let packed: Vec<u8> = (0..rows * bytes_per_row).map(|i| (i * 37) as u8).collect();
        let v: Vec<f32> = (0..cols).map(|i| i as f32 * 0.1).collect();
        let p = BinaryGemvParams::new(rows, cols, &packed, &v);
        let dispatched = gemv_binary(&p);
        let scalar = gemv_binary_scalar(&p);
        for (a, b) in dispatched.iter().zip(scalar.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn zero_rows_binary() {
        let packed: Vec<u8> = vec![];
        let v = vec![1.0_f32; 8];
        let p = BinaryGemvParams::new(0, 8, &packed, &v);
        assert!(gemv_binary(&p).is_empty());
    }

    #[test]
    fn weighted_vector_binary() {
        // single row: 0b11110000 → first 4 = +1, last 4 = -1
        let packed = vec![0xF0_u8];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = BinaryGemvParams::new(1, 8, &packed, &v);
        let r = gemv_binary(&p);
        // (1+2+3+4) - (5+6+7+8) = 10 - 26 = -16
        assert!((r[0] - (-16.0)).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "cols (7) must be a multiple of 8")]
    fn cols_not_multiple_of_8() {
        let _ = BinaryGemvParams::new(1, 7, &[0u8], &[1.0; 7]);
    }

    #[test]
    #[should_panic(expected = "packed_matrix length")]
    fn bad_packed_length() {
        let _ = BinaryGemvParams::new(2, 8, &[0xFF_u8], &[1.0; 8]);
    }

    #[test]
    #[should_panic(expected = "vector length")]
    fn bad_vector_length_binary() {
        let _ = BinaryGemvParams::new(1, 8, &[0xFF_u8], &[1.0; 4]);
    }

    #[test]
    fn large_binary_128cols() {
        let rows = 3;
        let cols = 128;
        let bytes_per_row = cols / 8;
        let packed = vec![0xAA_u8; rows * bytes_per_row]; // alternating +1/-1
        let v = vec![1.0_f32; cols];
        let p = BinaryGemvParams::new(rows, cols, &packed, &v);
        let r = gemv_binary(&p);
        for val in &r {
            assert!(val.abs() < 1e-5, "alternating should sum to 0, got {val}");
        }
    }
}
