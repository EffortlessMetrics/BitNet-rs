//! ARM NEON-optimized matrix transpose kernels for Apple Silicon.
//!
//! Provides 4×4 block transpose using NEON intrinsics, a general
//! transpose that tiles 4×4 blocks with scalar edges, and an in-place
//! square matrix transpose.

use std::arch::aarch64::*;

// ── 4×4 Block Transpose ────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix using NEON 4×4 block transpose.
///
/// Both dimensions must be exactly 4-aligned. For arbitrary dimensions
/// use [`neon_transpose`] which handles edge elements with a scalar fallback.
///
/// # Panics
///
/// Panics if `rows` or `cols` is not a multiple of 4, or if slice lengths
/// do not match `rows * cols`.
pub fn neon_transpose_4x4(input: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    let numel = rows * cols;
    assert!(rows.is_multiple_of(4), "rows must be a multiple of 4, got {rows}");
    assert!(cols.is_multiple_of(4), "cols must be a multiple of 4, got {cols}");
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    assert!(output.len() >= numel, "output length {} < rows*cols {numel}", output.len());

    for bi in (0..rows).step_by(4) {
        for bj in (0..cols).step_by(4) {
            // Load four rows of 4 floats from the source block.
            let r0;
            let r1;
            let r2;
            let r3;
            unsafe {
                r0 = vld1q_f32(input.as_ptr().add(bi * cols + bj));
                r1 = vld1q_f32(input.as_ptr().add((bi + 1) * cols + bj));
                r2 = vld1q_f32(input.as_ptr().add((bi + 2) * cols + bj));
                r3 = vld1q_f32(input.as_ptr().add((bi + 3) * cols + bj));
            }

            // Stage 1: element-level transpose within pairs using vtrn.
            let t0;
            let t1;
            let t2;
            let t3;
            unsafe {
                t0 = vtrn1q_f32(r0, r1); // [r0[0], r1[0], r0[2], r1[2]]
                t1 = vtrn2q_f32(r0, r1); // [r0[1], r1[1], r0[3], r1[3]]
                t2 = vtrn1q_f32(r2, r3); // [r2[0], r3[0], r2[2], r3[2]]
                t3 = vtrn2q_f32(r2, r3); // [r2[1], r3[1], r2[3], r3[3]]
            }

            // Stage 2: 64-bit half swap via f64 reinterpret + vtrn to complete transpose.
            let o0;
            let o1;
            let o2;
            let o3;
            unsafe {
                let t0_64 = vreinterpretq_f64_f32(t0);
                let t1_64 = vreinterpretq_f64_f32(t1);
                let t2_64 = vreinterpretq_f64_f32(t2);
                let t3_64 = vreinterpretq_f64_f32(t3);
                o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));
            }

            // Store transposed rows into the output (now cols × rows layout).
            unsafe {
                vst1q_f32(output.as_mut_ptr().add(bj * rows + bi), o0);
                vst1q_f32(output.as_mut_ptr().add((bj + 1) * rows + bi), o1);
                vst1q_f32(output.as_mut_ptr().add((bj + 2) * rows + bi), o2);
                vst1q_f32(output.as_mut_ptr().add((bj + 3) * rows + bi), o3);
            }
        }
    }
}

// ── General Transpose ──────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix, using NEON 4×4 blocks
/// for the aligned interior and scalar copy for edge elements.
///
/// # Panics
///
/// Panics if slice lengths do not match `rows * cols`.
pub fn neon_transpose(input: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    let numel = rows * cols;
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    assert!(output.len() >= numel, "output length {} < rows*cols {numel}", output.len());

    if numel == 0 {
        return;
    }

    let block_rows = rows & !3; // largest multiple of 4 ≤ rows
    let block_cols = cols & !3; // largest multiple of 4 ≤ cols

    // Main body: 4×4 NEON blocks.
    for bi in (0..block_rows).step_by(4) {
        for bj in (0..block_cols).step_by(4) {
            let r0;
            let r1;
            let r2;
            let r3;
            unsafe {
                r0 = vld1q_f32(input.as_ptr().add(bi * cols + bj));
                r1 = vld1q_f32(input.as_ptr().add((bi + 1) * cols + bj));
                r2 = vld1q_f32(input.as_ptr().add((bi + 2) * cols + bj));
                r3 = vld1q_f32(input.as_ptr().add((bi + 3) * cols + bj));
            }

            let t0;
            let t1;
            let t2;
            let t3;
            unsafe {
                t0 = vtrn1q_f32(r0, r1);
                t1 = vtrn2q_f32(r0, r1);
                t2 = vtrn1q_f32(r2, r3);
                t3 = vtrn2q_f32(r2, r3);
            }

            let o0;
            let o1;
            let o2;
            let o3;
            unsafe {
                let t0_64 = vreinterpretq_f64_f32(t0);
                let t1_64 = vreinterpretq_f64_f32(t1);
                let t2_64 = vreinterpretq_f64_f32(t2);
                let t3_64 = vreinterpretq_f64_f32(t3);
                o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));
            }

            unsafe {
                vst1q_f32(output.as_mut_ptr().add(bj * rows + bi), o0);
                vst1q_f32(output.as_mut_ptr().add((bj + 1) * rows + bi), o1);
                vst1q_f32(output.as_mut_ptr().add((bj + 2) * rows + bi), o2);
                vst1q_f32(output.as_mut_ptr().add((bj + 3) * rows + bi), o3);
            }
        }
    }

    // Right edge: columns beyond the last full 4-column block.
    for i in 0..block_rows {
        for j in block_cols..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }

    // Bottom edge: rows beyond the last full 4-row block (all columns).
    for i in block_rows..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// ── In-Place Square Transpose ──────────────────────────────────────

/// In-place transpose of a square `dim × dim` matrix stored in row-major
/// order.
///
/// # Panics
///
/// Panics if `data.len() < dim * dim`.
pub fn neon_transpose_inplace(data: &mut [f32], dim: usize) {
    let numel = dim * dim;
    assert!(data.len() >= numel, "data length {} < dim*dim {numel}", data.len());

    if dim <= 1 {
        return;
    }

    // For each pair (i, j) where i < j, swap data[i*dim+j] ↔ data[j*dim+i].
    // We process 4×4 diagonal blocks with NEON where possible.
    let block_dim = dim & !3;

    for bi in (0..block_dim).step_by(4) {
        // Off-diagonal 4×4 blocks: bi < bj.
        for bj in ((bi + 4)..block_dim).step_by(4) {
            // Load two 4×4 blocks: A at (bi, bj) and B at (bj, bi).
            let (a0, a1, a2, a3, b0, b1, b2, b3);
            unsafe {
                a0 = vld1q_f32(data.as_ptr().add(bi * dim + bj));
                a1 = vld1q_f32(data.as_ptr().add((bi + 1) * dim + bj));
                a2 = vld1q_f32(data.as_ptr().add((bi + 2) * dim + bj));
                a3 = vld1q_f32(data.as_ptr().add((bi + 3) * dim + bj));

                b0 = vld1q_f32(data.as_ptr().add(bj * dim + bi));
                b1 = vld1q_f32(data.as_ptr().add((bj + 1) * dim + bi));
                b2 = vld1q_f32(data.as_ptr().add((bj + 2) * dim + bi));
                b3 = vld1q_f32(data.as_ptr().add((bj + 3) * dim + bi));
            }

            // Transpose A → write to (bj, bi) position.
            let (ta0, ta1, ta2, ta3);
            unsafe {
                let at0 = vtrn1q_f32(a0, a1);
                let at1 = vtrn2q_f32(a0, a1);
                let at2 = vtrn1q_f32(a2, a3);
                let at3 = vtrn2q_f32(a2, a3);
                let at0_64 = vreinterpretq_f64_f32(at0);
                let at1_64 = vreinterpretq_f64_f32(at1);
                let at2_64 = vreinterpretq_f64_f32(at2);
                let at3_64 = vreinterpretq_f64_f32(at3);
                ta0 = vreinterpretq_f32_f64(vtrn1q_f64(at0_64, at2_64));
                ta1 = vreinterpretq_f32_f64(vtrn1q_f64(at1_64, at3_64));
                ta2 = vreinterpretq_f32_f64(vtrn2q_f64(at0_64, at2_64));
                ta3 = vreinterpretq_f32_f64(vtrn2q_f64(at1_64, at3_64));
            }

            // Transpose B → write to (bi, bj) position.
            let (tb0, tb1, tb2, tb3);
            unsafe {
                let bt0 = vtrn1q_f32(b0, b1);
                let bt1 = vtrn2q_f32(b0, b1);
                let bt2 = vtrn1q_f32(b2, b3);
                let bt3 = vtrn2q_f32(b2, b3);
                let bt0_64 = vreinterpretq_f64_f32(bt0);
                let bt1_64 = vreinterpretq_f64_f32(bt1);
                let bt2_64 = vreinterpretq_f64_f32(bt2);
                let bt3_64 = vreinterpretq_f64_f32(bt3);
                tb0 = vreinterpretq_f32_f64(vtrn1q_f64(bt0_64, bt2_64));
                tb1 = vreinterpretq_f32_f64(vtrn1q_f64(bt1_64, bt3_64));
                tb2 = vreinterpretq_f32_f64(vtrn2q_f64(bt0_64, bt2_64));
                tb3 = vreinterpretq_f32_f64(vtrn2q_f64(bt1_64, bt3_64));
            }

            // Write transposed A to B's position and vice versa.
            unsafe {
                vst1q_f32(data.as_mut_ptr().add(bj * dim + bi), ta0);
                vst1q_f32(data.as_mut_ptr().add((bj + 1) * dim + bi), ta1);
                vst1q_f32(data.as_mut_ptr().add((bj + 2) * dim + bi), ta2);
                vst1q_f32(data.as_mut_ptr().add((bj + 3) * dim + bi), ta3);

                vst1q_f32(data.as_mut_ptr().add(bi * dim + bj), tb0);
                vst1q_f32(data.as_mut_ptr().add((bi + 1) * dim + bj), tb1);
                vst1q_f32(data.as_mut_ptr().add((bi + 2) * dim + bj), tb2);
                vst1q_f32(data.as_mut_ptr().add((bi + 3) * dim + bj), tb3);
            }
        }

        // Diagonal 4×4 block at (bi, bi): in-place transpose via scalar swaps.
        for i in bi..bi + 4 {
            for j in (i + 1)..bi + 4 {
                data.swap(i * dim + j, j * dim + i);
            }
        }
    }

    // Remainder: rows/columns beyond the last full 4-block boundary.
    for i in 0..dim {
        let j_start = if i < block_dim { block_dim } else { i + 1 };
        for j in j_start..dim {
            data.swap(i * dim + j, j * dim + i);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a row-major matrix with value = row * cols + col + 1 for easy verification.
    fn make_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| (i + 1) as f32).collect()
    }

    /// Scalar reference transpose for verification.
    fn scalar_transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = input[i * cols + j];
            }
        }
        out
    }

    // ── neon_transpose_4x4 ────────────────────────────────────

    #[test]
    fn test_4x4_basic() {
        let input = make_matrix(4, 4);
        let mut output = vec![0.0f32; 16];
        neon_transpose_4x4(&input, 4, 4, &mut output);
        assert_eq!(output, scalar_transpose(&input, 4, 4));
    }

    #[test]
    fn test_4x4_8x8() {
        let input = make_matrix(8, 8);
        let mut output = vec![0.0f32; 64];
        neon_transpose_4x4(&input, 8, 8, &mut output);
        assert_eq!(output, scalar_transpose(&input, 8, 8));
    }

    #[test]
    fn test_4x4_non_square_aligned() {
        let input = make_matrix(4, 8);
        let mut output = vec![0.0f32; 32];
        neon_transpose_4x4(&input, 4, 8, &mut output);
        assert_eq!(output, scalar_transpose(&input, 4, 8));
    }

    #[test]
    #[should_panic(expected = "rows must be a multiple of 4")]
    fn test_4x4_rejects_unaligned_rows() {
        let input = make_matrix(3, 4);
        let mut output = vec![0.0f32; 12];
        neon_transpose_4x4(&input, 3, 4, &mut output);
    }

    #[test]
    #[should_panic(expected = "cols must be a multiple of 4")]
    fn test_4x4_rejects_unaligned_cols() {
        let input = make_matrix(4, 5);
        let mut output = vec![0.0f32; 20];
        neon_transpose_4x4(&input, 4, 5, &mut output);
    }

    // ── neon_transpose (general) ──────────────────────────────

    #[test]
    fn test_general_4x4() {
        let input = make_matrix(4, 4);
        let mut output = vec![0.0f32; 16];
        neon_transpose(&input, 4, 4, &mut output);
        assert_eq!(output, scalar_transpose(&input, 4, 4));
    }

    #[test]
    fn test_general_8x8() {
        let input = make_matrix(8, 8);
        let mut output = vec![0.0f32; 64];
        neon_transpose(&input, 8, 8, &mut output);
        assert_eq!(output, scalar_transpose(&input, 8, 8));
    }

    #[test]
    fn test_general_non_square() {
        let input = make_matrix(5, 7);
        let mut output = vec![0.0f32; 35];
        neon_transpose(&input, 5, 7, &mut output);
        assert_eq!(output, scalar_transpose(&input, 5, 7));
    }

    #[test]
    fn test_general_1x1() {
        let input = [42.0f32];
        let mut output = [0.0f32; 1];
        neon_transpose(&input, 1, 1, &mut output);
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_general_edge_3x5() {
        let input = make_matrix(3, 5);
        let mut output = vec![0.0f32; 15];
        neon_transpose(&input, 3, 5, &mut output);
        assert_eq!(output, scalar_transpose(&input, 3, 5));
    }

    #[test]
    fn test_general_large_matrix() {
        let (rows, cols) = (33, 65);
        let input = make_matrix(rows, cols);
        let mut output = vec![0.0f32; rows * cols];
        neon_transpose(&input, rows, cols, &mut output);
        assert_eq!(output, scalar_transpose(&input, rows, cols));
    }

    #[test]
    fn test_general_empty() {
        let mut output = vec![0.0f32; 0];
        neon_transpose(&[], 0, 0, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_general_involution() {
        let (rows, cols) = (6, 10);
        let input = make_matrix(rows, cols);
        let mut transposed = vec![0.0f32; rows * cols];
        neon_transpose(&input, rows, cols, &mut transposed);
        let mut back = vec![0.0f32; rows * cols];
        neon_transpose(&transposed, cols, rows, &mut back);
        assert_eq!(back, input);
    }

    // ── neon_transpose_inplace ────────────────────────────────

    #[test]
    fn test_inplace_1x1() {
        let mut data = [99.0f32];
        neon_transpose_inplace(&mut data, 1);
        assert_eq!(data[0], 99.0);
    }

    #[test]
    fn test_inplace_4x4() {
        let original = make_matrix(4, 4);
        let expected = scalar_transpose(&original, 4, 4);
        let mut data = original;
        neon_transpose_inplace(&mut data, 4);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_8x8() {
        let original = make_matrix(8, 8);
        let expected = scalar_transpose(&original, 8, 8);
        let mut data = original;
        neon_transpose_inplace(&mut data, 8);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_non_aligned_dim() {
        let dim = 7;
        let original = make_matrix(dim, dim);
        let expected = scalar_transpose(&original, dim, dim);
        let mut data = original;
        neon_transpose_inplace(&mut data, dim);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_involution() {
        let dim = 10;
        let original = make_matrix(dim, dim);
        let mut data = original.clone();
        neon_transpose_inplace(&mut data, dim);
        neon_transpose_inplace(&mut data, dim);
        assert_eq!(data, original);
    }

    #[test]
    fn test_inplace_large() {
        let dim = 35;
        let original = make_matrix(dim, dim);
        let expected = scalar_transpose(&original, dim, dim);
        let mut data = original;
        neon_transpose_inplace(&mut data, dim);
        assert_eq!(data, expected);
    }
}
