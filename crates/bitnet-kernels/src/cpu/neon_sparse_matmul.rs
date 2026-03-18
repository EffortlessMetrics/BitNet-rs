#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! NEON-optimized sparse matrix multiplication kernels for Apple Silicon.
//!
//! Provides CSR, CSC, block-sparse matrix multiply, sparse dot product, and
//! sparse-dense vector addition using NEON intrinsics on aarch64, with
//! scalar fallbacks for other architectures.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::missing_safety_doc,
    clippy::manual_div_ceil
)]

use std::arch::aarch64::*;

// ── CSR sparse × dense matmul ──────────────────────────────────────────

/// Multiply a sparse matrix in CSR format by a dense matrix.
///
/// `values` / `col_indices` / `row_ptrs` describe the sparse matrix (rows × cols).
/// `dense` is column-major with `cols` rows and `dense_cols` columns.
/// `output` is row-major with `rows` rows and `dense_cols` columns.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Index arrays must be in-bounds for the corresponding data slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_matmul_csr_neon(
    values: &[f32],
    col_indices: &[usize],
    row_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    _cols: usize,
    dense_cols: usize,
) {
    assert!(row_ptrs.len() > rows);
    assert!(output.len() >= rows * dense_cols);

    for i in 0..rows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        for dc in 0..dense_cols {
            let mut acc = vdupq_n_f32(0.0);
            let mut scalar_acc: f32 = 0.0;
            let nnz = end - start;
            let chunks = nnz / 4;
            let rem = nnz % 4;

            for c in 0..chunks {
                let base = start + c * 4;
                let v = unsafe {
                    vld1q_f32(
                        [values[base], values[base + 1], values[base + 2], values[base + 3]]
                            .as_ptr(),
                    )
                };
                let d = unsafe {
                    vld1q_f32(
                        [
                            dense[col_indices[base] * dense_cols + dc],
                            dense[col_indices[base + 1] * dense_cols + dc],
                            dense[col_indices[base + 2] * dense_cols + dc],
                            dense[col_indices[base + 3] * dense_cols + dc],
                        ]
                        .as_ptr(),
                    )
                };
                acc = vfmaq_f32(acc, v, d);
            }

            for k in 0..rem {
                let idx = start + chunks * 4 + k;
                scalar_acc += values[idx] * dense[col_indices[idx] * dense_cols + dc];
            }

            // Horizontal add
            let sum = {
                let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
                vget_lane_f32::<0>(vpadd_f32(pair, pair))
            };
            output[i * dense_cols + dc] = sum + scalar_acc;
        }
    }
}

/// Scalar fallback for CSR sparse × dense matmul.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_matmul_csr_neon(
    values: &[f32],
    col_indices: &[usize],
    row_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    _cols: usize,
    dense_cols: usize,
) {
    assert!(row_ptrs.len() > rows);
    assert!(output.len() >= rows * dense_cols);

    for i in 0..rows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        for dc in 0..dense_cols {
            let mut acc: f32 = 0.0;
            for k in start..end {
                acc += values[k] * dense[col_indices[k] * dense_cols + dc];
            }
            output[i * dense_cols + dc] = acc;
        }
    }
}

// ── CSC sparse × dense matmul ──────────────────────────────────────────

/// Multiply a sparse matrix in CSC format by a dense matrix.
///
/// `values` / `row_indices` / `col_ptrs` describe the sparse matrix (rows × cols).
/// `dense` is row-major with `cols` rows and `dense_cols` columns.
/// `output` is row-major with `rows` rows and `dense_cols` columns.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Index arrays must be in-bounds for the corresponding data slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_matmul_csc_neon(
    values: &[f32],
    row_indices: &[usize],
    col_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    dense_cols: usize,
) {
    assert!(col_ptrs.len() > cols);
    assert!(output.len() >= rows * dense_cols);

    // Zero output
    for v in output.iter_mut().take(rows * dense_cols) {
        *v = 0.0;
    }

    for j in 0..cols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        for dc in 0..dense_cols {
            let dense_val = dense[j * dense_cols + dc];
            if dense_val == 0.0 {
                continue;
            }
            let d_vec = vdupq_n_f32(dense_val);
            let nnz = end - start;
            let chunks = nnz / 4;
            let rem = nnz % 4;

            for c in 0..chunks {
                let base = start + c * 4;
                let v = unsafe {
                    vld1q_f32(
                        [values[base], values[base + 1], values[base + 2], values[base + 3]]
                            .as_ptr(),
                    )
                };
                let prod = vmulq_f32(v, d_vec);
                // Scatter-add to output rows
                let r0 = row_indices[base];
                let r1 = row_indices[base + 1];
                let r2 = row_indices[base + 2];
                let r3 = row_indices[base + 3];
                let mut tmp = [0.0f32; 4];
                unsafe { vst1q_f32(tmp.as_mut_ptr(), prod) };
                output[r0 * dense_cols + dc] += tmp[0];
                output[r1 * dense_cols + dc] += tmp[1];
                output[r2 * dense_cols + dc] += tmp[2];
                output[r3 * dense_cols + dc] += tmp[3];
            }
            for k in 0..rem {
                let idx = start + chunks * 4 + k;
                output[row_indices[idx] * dense_cols + dc] += values[idx] * dense_val;
            }
        }
    }
}

/// Scalar fallback for CSC sparse × dense matmul.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_matmul_csc_neon(
    values: &[f32],
    row_indices: &[usize],
    col_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    dense_cols: usize,
) {
    assert!(col_ptrs.len() > cols);
    assert!(output.len() >= rows * dense_cols);

    for v in output.iter_mut().take(rows * dense_cols) {
        *v = 0.0;
    }

    for j in 0..cols {
        let start = col_ptrs[j];
        let end = col_ptrs[j + 1];
        for dc in 0..dense_cols {
            let dense_val = dense[j * dense_cols + dc];
            for k in start..end {
                output[row_indices[k] * dense_cols + dc] += values[k] * dense_val;
            }
        }
    }
}

// ── Sparse-sparse dot product ──────────────────────────────────────────

/// Compute the dot product of two sparse vectors given in (value, index) form.
///
/// Both vectors must be sorted by index. Runs a merge-intersection.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Each values/indices pair must have equal lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_dot_product_neon(
    values_a: &[f32],
    indices_a: &[usize],
    values_b: &[f32],
    indices_b: &[usize],
) -> f32 {
    assert_eq!(values_a.len(), indices_a.len());
    assert_eq!(values_b.len(), indices_b.len());

    let mut ia = 0;
    let mut ib = 0;
    let mut acc = vdupq_n_f32(0.0);
    let mut buf_a = [0.0f32; 4];
    let mut buf_b = [0.0f32; 4];
    let mut buf_count = 0;

    while ia < indices_a.len() && ib < indices_b.len() {
        if indices_a[ia] == indices_b[ib] {
            buf_a[buf_count] = values_a[ia];
            buf_b[buf_count] = values_b[ib];
            buf_count += 1;
            if buf_count == 4 {
                let va = unsafe { vld1q_f32(buf_a.as_ptr()) };
                let vb = unsafe { vld1q_f32(buf_b.as_ptr()) };
                acc = vfmaq_f32(acc, va, vb);
                buf_count = 0;
            }
            ia += 1;
            ib += 1;
        } else if indices_a[ia] < indices_b[ib] {
            ia += 1;
        } else {
            ib += 1;
        }
    }

    // Flush remaining buffer
    let mut scalar_rem: f32 = 0.0;
    for k in 0..buf_count {
        scalar_rem += buf_a[k] * buf_b[k];
    }

    // Horizontal add
    let sum = {
        let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        vget_lane_f32::<0>(vpadd_f32(pair, pair))
    };
    sum + scalar_rem
}

/// Scalar fallback for sparse-sparse dot product.
#[cfg(not(target_arch = "aarch64"))]
pub fn sparse_dot_product_neon(
    values_a: &[f32],
    indices_a: &[usize],
    values_b: &[f32],
    indices_b: &[usize],
) -> f32 {
    assert_eq!(values_a.len(), indices_a.len());
    assert_eq!(values_b.len(), indices_b.len());

    let mut ia = 0;
    let mut ib = 0;
    let mut acc: f32 = 0.0;

    while ia < indices_a.len() && ib < indices_b.len() {
        if indices_a[ia] == indices_b[ib] {
            acc += values_a[ia] * values_b[ib];
            ia += 1;
            ib += 1;
        } else if indices_a[ia] < indices_b[ib] {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    acc
}

// ── Sparse-dense vector addition ───────────────────────────────────────

/// Add a sparse vector to a dense vector in-place: `dense[indices[i]] += values[i]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// All indices must be in-bounds for the `dense` slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_dense_add_neon(values: &[f32], indices: &[usize], dense: &mut [f32]) {
    assert_eq!(values.len(), indices.len());

    let n = values.len();
    let chunks = n / 4;
    let rem = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        let v = unsafe { vld1q_f32(values.as_ptr().add(base)) };
        let mut cur = [0.0f32; 4];
        cur[0] = dense[indices[base]];
        cur[1] = dense[indices[base + 1]];
        cur[2] = dense[indices[base + 2]];
        cur[3] = dense[indices[base + 3]];
        let d = unsafe { vld1q_f32(cur.as_ptr()) };
        let r = vaddq_f32(d, v);
        let mut out = [0.0f32; 4];
        unsafe { vst1q_f32(out.as_mut_ptr(), r) };
        dense[indices[base]] = out[0];
        dense[indices[base + 1]] = out[1];
        dense[indices[base + 2]] = out[2];
        dense[indices[base + 3]] = out[3];
    }

    for k in 0..rem {
        let idx = chunks * 4 + k;
        dense[indices[idx]] += values[idx];
    }
}

/// Scalar fallback for sparse-dense vector addition.
#[cfg(not(target_arch = "aarch64"))]
pub fn sparse_dense_add_neon(values: &[f32], indices: &[usize], dense: &mut [f32]) {
    assert_eq!(values.len(), indices.len());
    for (i, &idx) in indices.iter().enumerate() {
        dense[idx] += values[i];
    }
}

// ── Block-sparse matmul ────────────────────────────────────────────────

/// Block-sparse matrix multiply.
///
/// The sparse matrix is stored as a sequence of dense blocks of size
/// `block_size × block_size`. `blocks` contains the block values in row-major
/// order, `block_indices` the column-block index of each block, and
/// `block_ptrs` the range of blocks per block-row (CSR-of-blocks).
///
/// `dense` is row-major with `cols` rows and `dense_cols` columns.
/// `output` is row-major with `rows` rows and `dense_cols` columns.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// Index arrays must be in-bounds for the corresponding data slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_matmul_block_neon(
    blocks: &[f32],
    block_indices: &[usize],
    block_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    _cols: usize,
    dense_cols: usize,
    block_size: usize,
) {
    let block_rows = rows.div_ceil(block_size);
    assert!(block_ptrs.len() > block_rows);
    assert!(output.len() >= rows * dense_cols);

    for v in output.iter_mut().take(rows * dense_cols) {
        *v = 0.0;
    }

    let bs2 = block_size * block_size;

    for br in 0..block_rows {
        let b_start = block_ptrs[br];
        let b_end = block_ptrs[br + 1];
        for b_idx in b_start..b_end {
            let bc = block_indices[b_idx];
            let block = &blocks[b_idx * bs2..(b_idx + 1) * bs2];

            for bi in 0..block_size {
                let out_row = br * block_size + bi;
                if out_row >= rows {
                    break;
                }
                for dc in 0..dense_cols {
                    let mut acc = vdupq_n_f32(0.0);
                    let mut scalar_acc: f32 = 0.0;
                    let chunks = block_size / 4;
                    let rem = block_size % 4;

                    for c in 0..chunks {
                        let bj = c * 4;
                        let bv = unsafe {
                            vld1q_f32(
                                [
                                    block[bi * block_size + bj],
                                    block[bi * block_size + bj + 1],
                                    block[bi * block_size + bj + 2],
                                    block[bi * block_size + bj + 3],
                                ]
                                .as_ptr(),
                            )
                        };
                        let dense_row = bc * block_size + bj;
                        let dv = unsafe {
                            vld1q_f32(
                                [
                                    dense[dense_row * dense_cols + dc],
                                    dense[(dense_row + 1) * dense_cols + dc],
                                    dense[(dense_row + 2) * dense_cols + dc],
                                    dense[(dense_row + 3) * dense_cols + dc],
                                ]
                                .as_ptr(),
                            )
                        };
                        acc = vfmaq_f32(acc, bv, dv);
                    }

                    for bj in (chunks * 4)..block_size {
                        let dense_row = bc * block_size + bj;
                        if dense_row < _cols {
                            scalar_acc +=
                                block[bi * block_size + bj] * dense[dense_row * dense_cols + dc];
                        }
                    }

                    let sum = {
                        let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
                        vget_lane_f32::<0>(vpadd_f32(pair, pair))
                    };
                    output[out_row * dense_cols + dc] += sum + scalar_acc;
                }
            }
        }
    }
}

/// Scalar fallback for block-sparse matmul.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn sparse_matmul_block_neon(
    blocks: &[f32],
    block_indices: &[usize],
    block_ptrs: &[usize],
    dense: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    dense_cols: usize,
    block_size: usize,
) {
    let block_rows = (rows + block_size - 1) / block_size;
    assert!(block_ptrs.len() > block_rows);
    assert!(output.len() >= rows * dense_cols);

    for v in output.iter_mut().take(rows * dense_cols) {
        *v = 0.0;
    }

    let bs2 = block_size * block_size;

    for br in 0..block_rows {
        let b_start = block_ptrs[br];
        let b_end = block_ptrs[br + 1];
        for b_idx in b_start..b_end {
            let bc = block_indices[b_idx];
            let block = &blocks[b_idx * bs2..(b_idx + 1) * bs2];

            for bi in 0..block_size {
                let out_row = br * block_size + bi;
                if out_row >= rows {
                    break;
                }
                for dc in 0..dense_cols {
                    let mut acc: f32 = 0.0;
                    for bj in 0..block_size {
                        let dense_row = bc * block_size + bj;
                        if dense_row < cols {
                            acc += block[bi * block_size + bj] * dense[dense_row * dense_cols + dc];
                        }
                    }
                    output[out_row * dense_cols + dc] += acc;
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // Helper: call the NEON function (which is `unsafe` on aarch64).
    fn call_csr(
        values: &[f32],
        col_indices: &[usize],
        row_ptrs: &[usize],
        dense: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        dense_cols: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sparse_matmul_csr_neon(
                values,
                col_indices,
                row_ptrs,
                dense,
                output,
                rows,
                cols,
                dense_cols,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        sparse_matmul_csr_neon(
            values,
            col_indices,
            row_ptrs,
            dense,
            output,
            rows,
            cols,
            dense_cols,
        );
    }

    fn call_csc(
        values: &[f32],
        row_indices: &[usize],
        col_ptrs: &[usize],
        dense: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        dense_cols: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sparse_matmul_csc_neon(
                values,
                row_indices,
                col_ptrs,
                dense,
                output,
                rows,
                cols,
                dense_cols,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        sparse_matmul_csc_neon(
            values,
            row_indices,
            col_ptrs,
            dense,
            output,
            rows,
            cols,
            dense_cols,
        );
    }

    fn call_dot(
        values_a: &[f32],
        indices_a: &[usize],
        values_b: &[f32],
        indices_b: &[usize],
    ) -> f32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sparse_dot_product_neon(values_a, indices_a, values_b, indices_b)
        }
        #[cfg(not(target_arch = "aarch64"))]
        sparse_dot_product_neon(values_a, indices_a, values_b, indices_b)
    }

    fn call_add(values: &[f32], indices: &[usize], dense: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sparse_dense_add_neon(values, indices, dense);
        }
        #[cfg(not(target_arch = "aarch64"))]
        sparse_dense_add_neon(values, indices, dense);
    }

    fn call_block(
        blocks: &[f32],
        block_indices: &[usize],
        block_ptrs: &[usize],
        dense: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        dense_cols: usize,
        block_size: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            sparse_matmul_block_neon(
                blocks,
                block_indices,
                block_ptrs,
                dense,
                output,
                rows,
                cols,
                dense_cols,
                block_size,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        sparse_matmul_block_neon(
            blocks,
            block_indices,
            block_ptrs,
            dense,
            output,
            rows,
            cols,
            dense_cols,
            block_size,
        );
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn assert_slice_approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, e), "mismatch at index {i}: got {a}, expected {e}");
        }
    }

    // ── CSR tests ──────────────────────────────────────────────────

    #[test]
    fn test_csr_empty_matrix() {
        let row_ptrs = vec![0, 0, 0];
        let mut out = [0.0f32; 4];
        call_csr(&[], &[], &row_ptrs, &[1.0, 2.0, 3.0, 4.0], &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_csr_single_element() {
        // 1×1 sparse [[5.0]] × dense [[3.0]]
        let values = [5.0];
        let col_indices = [0];
        let row_ptrs = vec![0, 1];
        let dense = [3.0];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 1, 1);
        assert_slice_approx(&out, &[15.0]);
    }

    #[test]
    fn test_csr_identity_2x2() {
        // I₂ × [[1,2],[3,4]] = [[1,2],[3,4]]
        let values = vec![1.0, 1.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_csr_identity_3x3() {
        let values = vec![1.0, 1.0, 1.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = [0.0f32; 9];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 3, 3, 3);
        assert_slice_approx(&out, &dense);
    }

    #[test]
    fn test_csr_zero_values() {
        let values = vec![0.0, 0.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1, 2];
        let dense = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_csr_general_2x3_times_3x2() {
        // Sparse 2×3: row0 = [1, 0, 2], row1 = [0, 3, 0]
        let values = vec![1.0, 2.0, 3.0];
        let col_indices = vec![0, 2, 1];
        let row_ptrs = vec![0, 2, 3];
        // Dense 3×2: [[1,2],[3,4],[5,6]]
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 4];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 2, 3, 2);
        // row0: 1*[1,2] + 2*[5,6] = [11, 14]
        // row1: 3*[3,4] = [9, 12]
        assert_slice_approx(&out, &[11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_csr_negative_values() {
        // 1×2 sparse [-1, 2] × 2×2 dense [[3,4],[5,6]]
        let values = vec![-1.0, 2.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 2];
        let dense = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 2, 2);
        // -1*[3,4] + 2*[5,6] = [7,8]
        assert_slice_approx(&out, &[7.0, 8.0]);
    }

    #[test]
    fn test_csr_single_row_dense_vector() {
        // 1×4 sparse × 4×1 dense = 1×1
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2, 3];
        let row_ptrs = vec![0, 4];
        let dense = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 4, 1);
        assert_slice_approx(&out, &[10.0]);
    }

    #[test]
    fn test_csr_five_nnz_per_row() {
        // Test remainder handling: 5 = 4 (NEON) + 1 (scalar)
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 1, 2, 3, 4];
        let row_ptrs = vec![0, 5];
        let dense = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 5, 1);
        assert_slice_approx(&out, &[15.0]);
    }

    #[test]
    fn test_csr_seven_nnz_per_row() {
        // 7 = 4 + 3 remainder
        let values: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let col_indices: Vec<usize> = (0..7).collect();
        let row_ptrs = vec![0, 7];
        let dense = [1.0; 7];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 7, 1);
        assert_slice_approx(&out, &[28.0]);
    }

    #[test]
    fn test_csr_eight_nnz_exact() {
        // 8 = 2 × 4, no remainder
        let values: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let col_indices: Vec<usize> = (0..8).collect();
        let row_ptrs = vec![0, 8];
        let dense = [2.0; 8];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 8, 1);
        // sum(1..8)*2 = 36*2 = 72
        assert_slice_approx(&out, &[72.0]);
    }

    #[test]
    fn test_csr_multiple_dense_cols() {
        // 2×2 sparse × 2×3 dense
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 0, 1];
        let row_ptrs = vec![0, 2, 4];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 6];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 2, 2, 3);
        // row0: 1*[1,2,3] + 2*[4,5,6] = [9,12,15]
        // row1: 3*[1,2,3] + 4*[4,5,6] = [19,26,33]
        assert_slice_approx(&out, &[9.0, 12.0, 15.0, 19.0, 26.0, 33.0]);
    }

    #[test]
    fn test_csr_sparse_row_with_gap() {
        // 3×4 sparse, middle row empty
        let values = vec![1.0, 2.0];
        let col_indices = vec![0, 3];
        let row_ptrs = vec![0, 1, 1, 2];
        let dense = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 3];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 3, 4, 1);
        assert_slice_approx(&out, &[5.0, 0.0, 16.0]);
    }

    #[test]
    fn test_csr_large_values() {
        let values = vec![1e6, 1e6];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 2];
        let dense = vec![1e6, 1e6];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 2, 1);
        assert_slice_approx(&out, &[2e12]);
    }

    #[test]
    fn test_csr_fractional_values() {
        let values = vec![0.5, 0.25];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 2];
        let dense = vec![4.0, 8.0];
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 2, 1);
        assert_slice_approx(&out, &[4.0]);
    }

    // ── CSC tests ──────────────────────────────────────────────────

    #[test]
    fn test_csc_empty_matrix() {
        let col_ptrs = vec![0, 0, 0];
        let mut out = [99.0f32; 4];
        call_csc(&[], &[], &col_ptrs, &[1.0, 2.0, 3.0, 4.0], &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_csc_single_element() {
        let values = [5.0];
        let row_indices = [0];
        let col_ptrs = vec![0, 1];
        let dense = [3.0];
        let mut out = [0.0f32; 1];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 1, 1, 1);
        assert_slice_approx(&out, &[15.0]);
    }

    #[test]
    fn test_csc_identity_2x2() {
        let values = vec![1.0, 1.0];
        let row_indices = vec![0, 1];
        let col_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_csc_identity_3x3() {
        let values = vec![1.0, 1.0, 1.0];
        let row_indices = vec![0, 1, 2];
        let col_ptrs = vec![0, 1, 2, 3];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = [0.0f32; 9];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 3, 3, 3);
        assert_slice_approx(&out, &dense);
    }

    #[test]
    fn test_csc_zero_values() {
        let values = vec![0.0, 0.0];
        let row_indices = vec![0, 1];
        let col_ptrs = vec![0, 1, 2];
        let dense = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 2, 2, 2);
        assert_slice_approx(&out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_csc_general_2x3_times_3x2() {
        // Same matrix as CSR test: row0 = [1, 0, 2], row1 = [0, 3, 0]
        // CSC: col0: (0,1), col1: (1,3), col2: (0,2)
        let values = vec![1.0, 3.0, 2.0];
        let row_indices = vec![0, 1, 0];
        let col_ptrs = vec![0, 1, 2, 3];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 4];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 2, 3, 2);
        assert_slice_approx(&out, &[11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_csc_negative_values() {
        let values = vec![-1.0, 2.0];
        let row_indices = vec![0, 0];
        let col_ptrs = vec![0, 1, 2];
        let dense = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 1, 2, 2);
        assert_slice_approx(&out, &[7.0, 8.0]);
    }

    #[test]
    fn test_csc_column_with_multiple_entries() {
        // 3×2 sparse, col 0 has entries at rows 0, 1, 2
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![0, 1, 2, 0];
        let col_ptrs = vec![0, 3, 4];
        let dense = vec![1.0, 2.0];
        let mut out = [0.0f32; 3];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 3, 2, 1);
        // col0: 1→row0, 2→row1, 3→row2; dense[0]=1
        // col1: 4→row0; dense[1]=2
        // out[0]=1*1+4*2=9, out[1]=2*1=2, out[2]=3*1=3
        assert_slice_approx(&out, &[9.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csc_five_nnz_in_col() {
        // 5 entries in a single column (test NEON 4+1)
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 1, 2, 3, 4];
        let col_ptrs = vec![0, 5];
        let dense = [2.0];
        let mut out = [0.0f32; 5];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 5, 1, 1);
        assert_slice_approx(&out, &[2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_csc_empty_column() {
        let values = [1.0];
        let row_indices = [0];
        let col_ptrs = vec![0, 0, 1]; // col 0 empty, col 1 has entry
        let dense = vec![5.0, 3.0];
        let mut out = [0.0f32; 1];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 1, 2, 1);
        // col0: empty, col1: 1*3=3
        assert_slice_approx(&out, &[3.0]);
    }

    #[test]
    fn test_csc_multiple_dense_cols() {
        // CSC: col0 = [(row0,1),(row1,2)], col1 = [(row0,3),(row1,4)]
        // Sparse matrix = [[1,3],[2,4]], dense 2×3 = [[1,2,3],[4,5,6]]
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![0, 1, 0, 1];
        let col_ptrs = vec![0, 2, 4];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 6];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 2, 2, 3);
        // row0: 1*[1,2,3]+3*[4,5,6]=[13,17,21]
        // row1: 2*[1,2,3]+4*[4,5,6]=[18,24,30]
        assert_slice_approx(&out, &[13.0, 17.0, 21.0, 18.0, 24.0, 30.0]);
    }

    // ── CSR/CSC consistency tests ──────────────────────────────────

    #[test]
    fn test_csr_csc_consistency_identity() {
        let csr_values = vec![1.0, 1.0, 1.0];
        let csr_col_indices = vec![0, 1, 2];
        let csr_row_ptrs = vec![0, 1, 2, 3];
        let csc_values = vec![1.0, 1.0, 1.0];
        let csc_row_indices = vec![0, 1, 2];
        let csc_col_ptrs = vec![0, 1, 2, 3];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_csr = [0.0f32; 6];
        let mut out_csc = [0.0f32; 6];
        call_csr(&csr_values, &csr_col_indices, &csr_row_ptrs, &dense, &mut out_csr, 3, 3, 2);
        call_csc(&csc_values, &csc_row_indices, &csc_col_ptrs, &dense, &mut out_csc, 3, 3, 2);
        assert_slice_approx(&out_csr, &out_csc);
    }

    #[test]
    fn test_csr_csc_consistency_general() {
        // Sparse 2×3: [[1,0,2],[0,3,0]]
        // CSR
        let csr_v = vec![1.0, 2.0, 3.0];
        let csr_ci = vec![0, 2, 1];
        let csr_rp = vec![0, 2, 3];
        // CSC: col0: (0,1), col1: (1,3), col2: (0,2)
        let csc_v = vec![1.0, 3.0, 2.0];
        let csc_ri = vec![0, 1, 0];
        let csc_cp = vec![0, 1, 2, 3];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out_csr = [0.0f32; 4];
        let mut out_csc = [0.0f32; 4];
        call_csr(&csr_v, &csr_ci, &csr_rp, &dense, &mut out_csr, 2, 3, 2);
        call_csc(&csc_v, &csc_ri, &csc_cp, &dense, &mut out_csc, 2, 3, 2);
        assert_slice_approx(&out_csr, &out_csc);
    }

    #[test]
    fn test_csr_csc_consistency_full_matrix() {
        // Full 2×2: [[1,2],[3,4]]
        let csr_v = vec![1.0, 2.0, 3.0, 4.0];
        let csr_ci = vec![0, 1, 0, 1];
        let csr_rp = vec![0, 2, 4];
        let csc_v = vec![1.0, 3.0, 2.0, 4.0];
        let csc_ri = vec![0, 1, 0, 1];
        let csc_cp = vec![0, 2, 4];
        let dense = vec![5.0, 6.0, 7.0, 8.0];
        let mut out_csr = [0.0f32; 4];
        let mut out_csc = [0.0f32; 4];
        call_csr(&csr_v, &csr_ci, &csr_rp, &dense, &mut out_csr, 2, 2, 2);
        call_csc(&csc_v, &csc_ri, &csc_cp, &dense, &mut out_csc, 2, 2, 2);
        assert_slice_approx(&out_csr, &out_csc);
    }

    #[test]
    fn test_csr_csc_consistency_single_nnz() {
        // Sparse 3×3 with one nonzero at (1,2)=7
        let csr_v = [7.0];
        let csr_ci = [2];
        let csr_rp = vec![0, 0, 1, 1];
        let csc_v = [7.0];
        let csc_ri = [1];
        let csc_cp = vec![0, 0, 0, 1];
        let dense = vec![1.0, 2.0, 3.0];
        let mut out_csr = [0.0f32; 3];
        let mut out_csc = [0.0f32; 3];
        call_csr(&csr_v, &csr_ci, &csr_rp, &dense, &mut out_csr, 3, 3, 1);
        call_csc(&csc_v, &csc_ri, &csc_cp, &dense, &mut out_csc, 3, 3, 1);
        assert_slice_approx(&out_csr, &out_csc);
    }

    // ── Dot product tests ──────────────────────────────────────────

    #[test]
    fn test_dot_empty() {
        let r = call_dot(&[], &[], &[], &[]);
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_dot_no_overlap() {
        let r = call_dot(&[1.0], &[0], &[2.0], &[1]);
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_dot_single_overlap() {
        let r = call_dot(&[3.0], &[5], &[4.0], &[5]);
        assert!(approx_eq(r, 12.0));
    }

    #[test]
    fn test_dot_full_overlap() {
        let r = call_dot(&[1.0, 2.0, 3.0], &[0, 1, 2], &[4.0, 5.0, 6.0], &[0, 1, 2]);
        assert!(approx_eq(r, 32.0));
    }

    #[test]
    fn test_dot_partial_overlap() {
        let r = call_dot(&[1.0, 2.0, 3.0], &[0, 2, 4], &[4.0, 5.0, 6.0], &[1, 2, 3]);
        // only index 2 overlaps: 2*5=10
        assert!(approx_eq(r, 10.0));
    }

    #[test]
    fn test_dot_negative_values() {
        let r = call_dot(&[-1.0, 2.0], &[0, 1], &[3.0, -4.0], &[0, 1]);
        // -1*3 + 2*(-4) = -3 + -8 = -11
        assert!(approx_eq(r, -11.0));
    }

    #[test]
    fn test_dot_four_elements_exact() {
        // Exactly 4 overlapping → one full NEON iteration
        let r =
            call_dot(&[1.0, 2.0, 3.0, 4.0], &[0, 1, 2, 3], &[5.0, 6.0, 7.0, 8.0], &[0, 1, 2, 3]);
        assert!(approx_eq(r, 70.0));
    }

    #[test]
    fn test_dot_five_elements() {
        let r = call_dot(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0, 1, 2, 3, 4],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            &[0, 1, 2, 3, 4],
        );
        assert!(approx_eq(r, 15.0));
    }

    #[test]
    fn test_dot_interleaved_indices() {
        // a: indices [0, 2, 4, 6, 8]
        // b: indices [1, 3, 5, 7, 9]
        let r = call_dot(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[0, 2, 4, 6, 8],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1, 3, 5, 7, 9],
        );
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_dot_one_sided_empty() {
        let r = call_dot(&[1.0, 2.0], &[0, 1], &[], &[]);
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_dot_eight_elements() {
        let va: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let ia: Vec<usize> = (0..8).collect();
        let vb: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let ib: Vec<usize> = (0..8).collect();
        let r = call_dot(&va, &ia, &vb, &ib);
        // sum of squares 1..8 = 204
        assert!(approx_eq(r, 204.0));
    }

    #[test]
    fn test_dot_zero_values() {
        let r = call_dot(&[0.0, 0.0], &[0, 1], &[5.0, 6.0], &[0, 1]);
        assert!(approx_eq(r, 0.0));
    }

    // ── Sparse-dense add tests ─────────────────────────────────────

    #[test]
    fn test_add_empty() {
        let mut dense = vec![1.0, 2.0, 3.0];
        call_add(&[], &[], &mut dense);
        assert_slice_approx(&dense, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_single() {
        let mut dense = vec![1.0, 2.0, 3.0];
        call_add(&[10.0], &[1], &mut dense);
        assert_slice_approx(&dense, &[1.0, 12.0, 3.0]);
    }

    #[test]
    fn test_add_all_indices() {
        let mut dense = [0.0; 4];
        call_add(&[1.0, 2.0, 3.0, 4.0], &[0, 1, 2, 3], &mut dense);
        assert_slice_approx(&dense, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_add_negative() {
        let mut dense = vec![10.0, 20.0, 30.0];
        call_add(&[-5.0, -10.0], &[0, 2], &mut dense);
        assert_slice_approx(&dense, &[5.0, 20.0, 20.0]);
    }

    #[test]
    fn test_add_five_elements() {
        let mut dense = [0.0; 5];
        call_add(&[1.0, 2.0, 3.0, 4.0, 5.0], &[0, 1, 2, 3, 4], &mut dense);
        assert_slice_approx(&dense, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_add_eight_elements() {
        let mut dense = [1.0; 8];
        let values: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices: Vec<usize> = (0..8).collect();
        call_add(&values, &indices, &mut dense);
        let expected: Vec<f32> = (0..8).map(|x| 1.0 + x as f32).collect();
        assert_slice_approx(&dense, &expected);
    }

    #[test]
    fn test_add_zero_values() {
        let mut dense = vec![1.0, 2.0, 3.0];
        call_add(&[0.0, 0.0], &[0, 2], &mut dense);
        assert_slice_approx(&dense, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_non_contiguous_indices() {
        let mut dense = [0.0; 10];
        call_add(&[1.0, 2.0, 3.0], &[0, 5, 9], &mut dense);
        let mut expected = [0.0; 10];
        expected[0] = 1.0;
        expected[5] = 2.0;
        expected[9] = 3.0;
        assert_slice_approx(&dense, &expected);
    }

    #[test]
    fn test_add_duplicate_indices() {
        // Duplicate indices: both add to same slot
        let mut dense = [0.0; 3];
        call_add(&[1.0, 2.0], &[1, 1], &mut dense);
        assert_slice_approx(&dense, &[0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_add_large_values() {
        let mut dense = vec![1e6];
        call_add(&[1e6], &[0], &mut dense);
        assert_slice_approx(&dense, &[2e6]);
    }

    // ── Block-sparse matmul tests ──────────────────────────────────

    #[test]
    fn test_block_empty() {
        let block_ptrs = vec![0, 0];
        let mut out = [99.0f32; 4];
        call_block(&[], &[], &block_ptrs, &[1.0, 2.0, 3.0, 4.0], &mut out, 2, 2, 2, 2);
        assert_slice_approx(&out, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_block_identity_2x2() {
        // 2×2 identity as one block
        let blocks = vec![1.0, 0.0, 0.0, 1.0];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 2, 2);
        assert_slice_approx(&out, &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_block_single_element_block() {
        // Block size 1 = scalar
        let blocks = vec![2.0, 3.0];
        let block_indices = vec![0, 0];
        let block_ptrs = vec![0, 1, 2];
        let dense = [5.0];
        let mut out = [0.0f32; 2];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 1, 1, 1);
        assert_slice_approx(&out, &[10.0, 15.0]);
    }

    #[test]
    fn test_block_size_2() {
        // 2×2 block [[1,2],[3,4]] at block-col 0, dense 2×1 = [5,6]
        let blocks = vec![1.0, 2.0, 3.0, 4.0];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![5.0, 6.0];
        let mut out = [0.0f32; 2];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 1, 2);
        // row0: 1*5+2*6=17, row1: 3*5+4*6=39
        assert_slice_approx(&out, &[17.0, 39.0]);
    }

    #[test]
    fn test_block_size_4() {
        // 4×4 identity block
        let mut blocks = [0.0f32; 16];
        for i in 0..4 {
            blocks[i * 4 + i] = 1.0;
        }
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 4, 1, 4);
        assert_slice_approx(&out, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_block_size_4_general() {
        // 4×4 all-ones block × dense [1,2,3,4] = [10,10,10,10]
        let blocks = [1.0; 16];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 4, 1, 4);
        assert_slice_approx(&out, &[10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_block_size_8() {
        // 8×8 identity
        let mut blocks = [0.0f32; 64];
        for i in 0..8 {
            blocks[i * 8 + i] = 1.0;
        }
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut out = [0.0f32; 8];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 8, 8, 1, 8);
        assert_slice_approx(&out, &dense);
    }

    #[test]
    fn test_block_size_8_ones() {
        let blocks = [1.0; 64];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut out = [0.0f32; 8];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 8, 8, 1, 8);
        // each row: sum(1..8)=36
        assert_slice_approx(&out, &[36.0; 8]);
    }

    #[test]
    fn test_block_multiple_blocks_per_row() {
        // 4×4, block_size=2. Two 2×2 blocks in block-row 0.
        // block 0 at block-col 0: [[1,0],[0,1]]
        // block 1 at block-col 1: [[2,0],[0,2]]
        let blocks = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let block_indices = vec![0, 1];
        let block_ptrs = vec![0, 2, 2]; // block-row 0 has 2 blocks, row 1 has 0
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 4, 1, 2);
        // block-row0, block-col0: I₂ × [1,2]ᵀ = [1,2]
        // block-row0, block-col1: 2I₂ × [3,4]ᵀ = [6,8]
        // out rows 0,1 = [1+6, 2+8] = [7, 10]
        // block-row1: no blocks → [0, 0]
        assert_slice_approx(&out, &[7.0, 10.0, 0.0, 0.0]);
    }

    #[test]
    fn test_block_multiple_dense_cols() {
        let blocks = vec![1.0, 2.0, 3.0, 4.0]; // one 2×2 block
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 2, 2);
        // row0: 1*[1,2]+2*[3,4]=[7,10]
        // row1: 3*[1,2]+4*[3,4]=[15,22]
        assert_slice_approx(&out, &[7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn test_block_zero_block() {
        let blocks = [0.0; 4];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![5.0, 6.0];
        let mut out = [0.0f32; 2];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 1, 2);
        assert_slice_approx(&out, &[0.0, 0.0]);
    }

    #[test]
    fn test_block_negative_values() {
        let blocks = vec![-1.0, 0.0, 0.0, -1.0];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![3.0, 4.0];
        let mut out = [0.0f32; 2];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 1, 2);
        assert_slice_approx(&out, &[-3.0, -4.0]);
    }

    #[test]
    fn test_block_non_square_output() {
        // 4 rows, block_size=2, two block-rows
        // block-row 0: one block at block-col 0 = [[1,1],[1,1]]
        // block-row 1: one block at block-col 0 = [[2,2],[2,2]]
        let blocks = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let block_indices = vec![0, 0];
        let block_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 1.0, 2.0, 2.0]; // 2 rows × 2 dense_cols
        let mut out = [0.0f32; 8];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 2, 2, 2);
        // block-row0: [[1,1],[1,1]] × [[1,1],[2,2]] = [[3,3],[3,3]]
        // block-row1: [[2,2],[2,2]] × [[1,1],[2,2]] = [[6,6],[6,6]]
        assert_slice_approx(&out, &[3.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_block_size_2_diagonal() {
        // 4×4 block-diagonal with block_size=2
        // block-row 0, block-col 0: [[1,0],[0,1]]
        // block-row 1, block-col 1: [[2,0],[0,2]]
        let blocks = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let block_indices = vec![0, 1];
        let block_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 4, 1, 2);
        assert_slice_approx(&out, &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn test_block_size_4_with_remainder() {
        // 4×4 block but only 3 actual rows
        let blocks = [1.0; 16];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 3, 4, 1, 4);
        assert_slice_approx(&out, &[4.0, 4.0, 4.0]);
    }

    // ── Additional edge case tests ─────────────────────────────────

    #[test]
    fn test_csr_1x1() {
        let mut out = [0.0f32; 1];
        call_csr(&[7.0], &[0], &[0, 1], &[3.0], &mut out, 1, 1, 1);
        assert_slice_approx(&out, &[21.0]);
    }

    #[test]
    fn test_csc_1x1() {
        let mut out = [0.0f32; 1];
        call_csc(&[7.0], &[0], &[0, 1], &[3.0], &mut out, 1, 1, 1);
        assert_slice_approx(&out, &[21.0]);
    }

    #[test]
    fn test_csr_diagonal_4x4() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 2, 3];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let dense = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 4, 4, 1);
        assert_slice_approx(&out, &[10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_csc_diagonal_4x4() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![0, 1, 2, 3];
        let col_ptrs = vec![0, 1, 2, 3, 4];
        let dense = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 4, 4, 1);
        assert_slice_approx(&out, &[10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn test_dot_large_sparse() {
        // 9 overlapping elements = 2×4 + 1
        let va: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let ia: Vec<usize> = (0..9).collect();
        let vb = [1.0; 9];
        let ib: Vec<usize> = (0..9).collect();
        let r = call_dot(&va, &ia, &vb, &ib);
        assert!(approx_eq(r, 45.0));
    }

    #[test]
    fn test_add_seven_elements() {
        let mut dense = [0.0; 7];
        let values: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        let indices: Vec<usize> = (0..7).collect();
        call_add(&values, &indices, &mut dense);
        let expected: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        assert_slice_approx(&dense, &expected);
    }

    #[test]
    fn test_csr_all_same_column() {
        // 3×2 sparse, all NNZ in column 1
        let values = vec![1.0, 2.0, 3.0];
        let col_indices = vec![1, 1, 1];
        let row_ptrs = vec![0, 1, 2, 3];
        let dense = vec![10.0, 20.0];
        let mut out = [0.0f32; 3];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 3, 2, 1);
        assert_slice_approx(&out, &[20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_csc_all_same_column() {
        let values = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0, 1, 2];
        let col_ptrs = vec![0, 0, 3]; // col 0 empty, col 1 has all
        let dense = vec![10.0, 20.0];
        let mut out = [0.0f32; 3];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 3, 2, 1);
        assert_slice_approx(&out, &[20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_csr_csc_consistency_diagonal_4x4() {
        let csr_v = vec![1.0, 2.0, 3.0, 4.0];
        let csr_ci = vec![0, 1, 2, 3];
        let csr_rp = vec![0, 1, 2, 3, 4];
        let csc_v = vec![1.0, 2.0, 3.0, 4.0];
        let csc_ri = vec![0, 1, 2, 3];
        let csc_cp = vec![0, 1, 2, 3, 4];
        let dense = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out_csr = [0.0f32; 8];
        let mut out_csc = [0.0f32; 8];
        call_csr(&csr_v, &csr_ci, &csr_rp, &dense, &mut out_csr, 4, 4, 2);
        call_csc(&csc_v, &csc_ri, &csc_cp, &dense, &mut out_csc, 4, 4, 2);
        assert_slice_approx(&out_csr, &out_csc);
    }

    #[test]
    fn test_block_size_2_two_block_rows() {
        // 4×2 with block_size=2
        // block-row 0: block at col 0 = [[1,2],[3,4]]
        // block-row 1: block at col 0 = [[5,6],[7,8]]
        let blocks = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block_indices = vec![0, 0];
        let block_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 1.0];
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 2, 1, 2);
        // row0: 1+2=3, row1: 3+4=7, row2: 5+6=11, row3: 7+8=15
        assert_slice_approx(&out, &[3.0, 7.0, 11.0, 15.0]);
    }

    #[test]
    fn test_dot_sorted_merge_skip() {
        // a at [0,10,20], b at [5,15,25] → no overlap
        let r = call_dot(&[1.0, 2.0, 3.0], &[0, 10, 20], &[4.0, 5.0, 6.0], &[5, 15, 25]);
        assert!(approx_eq(r, 0.0));
    }

    #[test]
    fn test_csr_dense_cols_3() {
        // 1×2 sparse × 2×3 dense
        let values = vec![1.0, 1.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 2];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 2, 3);
        assert_slice_approx(&out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_csc_dense_cols_3() {
        let values = vec![1.0, 1.0];
        let row_indices = vec![0, 0];
        let col_ptrs = vec![0, 1, 2];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 1, 2, 3);
        assert_slice_approx(&out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_to_large_dense() {
        let mut dense = [100.0; 20];
        call_add(&[1.0, 2.0, 3.0], &[0, 10, 19], &mut dense);
        assert!(approx_eq(dense[0], 101.0));
        assert!(approx_eq(dense[10], 102.0));
        assert!(approx_eq(dense[19], 103.0));
        assert!(approx_eq(dense[1], 100.0));
    }

    #[test]
    fn test_block_size_4_two_blocks() {
        // 4×8, block_size=4. One block-row with two blocks.
        // block0 at block-col 0: I₄
        // block1 at block-col 1: 2*I₄
        let mut blocks = [0.0f32; 32]; // 2 blocks × 16
        for i in 0..4 {
            blocks[i * 4 + i] = 1.0; // block 0 = I
            blocks[16 + i * 4 + i] = 2.0; // block 1 = 2I
        }
        let block_indices = vec![0, 1];
        let block_ptrs = vec![0, 2];
        let dense: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut out = [0.0f32; 4];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 4, 8, 1, 4);
        // row0: 1*1 + 2*5 = 11
        // row1: 1*2 + 2*6 = 14
        // row2: 1*3 + 2*7 = 17
        // row3: 1*4 + 2*8 = 20
        assert_slice_approx(&out, &[11.0, 14.0, 17.0, 20.0]);
    }

    #[test]
    fn test_block_size_8_with_dense_cols_2() {
        // 8×8 identity block, dense 8×2
        let mut blocks = [0.0f32; 64];
        for i in 0..8 {
            blocks[i * 8 + i] = 1.0;
        }
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut out = [0.0f32; 16];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 8, 8, 2, 8);
        assert_slice_approx(&out, &dense);
    }

    #[test]
    fn test_csr_nine_nnz() {
        // 9 = 2×4 + 1
        let values: Vec<f32> = vec![1.0; 9];
        let col_indices: Vec<usize> = (0..9).collect();
        let row_ptrs = vec![0, 9];
        let dense: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut out = [0.0f32; 1];
        call_csr(&values, &col_indices, &row_ptrs, &dense, &mut out, 1, 9, 1);
        assert!(approx_eq(out[0], 45.0));
    }

    #[test]
    fn test_csc_nine_nnz_in_col() {
        let values = [1.0; 9];
        let row_indices: Vec<usize> = (0..9).collect();
        let col_ptrs = vec![0, 9];
        let dense = [2.0];
        let mut out = [0.0f32; 9];
        call_csc(&values, &row_indices, &col_ptrs, &dense, &mut out, 9, 1, 1);
        assert_slice_approx(&out, &[2.0; 9]);
    }

    #[test]
    fn test_dot_two_elements() {
        let r = call_dot(&[3.0, 4.0], &[0, 1], &[5.0, 6.0], &[0, 1]);
        assert!(approx_eq(r, 39.0));
    }

    #[test]
    fn test_dot_three_elements() {
        let r = call_dot(&[1.0, 2.0, 3.0], &[0, 1, 2], &[4.0, 5.0, 6.0], &[0, 1, 2]);
        assert!(approx_eq(r, 32.0));
    }

    #[test]
    fn test_add_four_elements_exact() {
        let mut dense = [10.0; 4];
        call_add(&[1.0, 2.0, 3.0, 4.0], &[0, 1, 2, 3], &mut dense);
        assert_slice_approx(&dense, &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_block_size_2_with_dense_cols_3() {
        let blocks = vec![1.0, 0.0, 0.0, 1.0];
        let block_indices = [0];
        let block_ptrs = vec![0, 1];
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 6];
        call_block(&blocks, &block_indices, &block_ptrs, &dense, &mut out, 2, 2, 3, 2);
        assert_slice_approx(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
