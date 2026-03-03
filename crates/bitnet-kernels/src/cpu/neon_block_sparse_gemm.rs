//! ARM NEON block-sparse GEMM kernels for Apple Silicon.
//!
//! Provides NEON-accelerated sparse matrix multiply operations optimised for
//! inference workloads where weight matrices are highly sparse. Supported
//! formats:
//!
//! * **Block-sparse** — dense sub-blocks at sparse locations (4×4, 8×8, 16×16)
//! * **CSR** — compressed sparse row for unstructured sparsity
//! * **2:4 structured** — hardware-friendly 50 % sparsity pattern
//! * **Sparse GEMV** — block-sparse matrix × dense vector
//! * **Sparse-activation GEMM** — dense weights × ReLU-sparse activations
//! * **Sparse transpose multiply** — A^T · B_sparse
//!
//! Every function has a scalar fallback that runs on any target. The NEON
//! variants are gated behind `#[cfg(target_arch = "aarch64")]` +
//! `#[target_feature(enable = "neon")]`.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_unsafe
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Block-sparse storage ────────────────────────────────────────────

/// A non-zero block in a block-sparse matrix.
///
/// `row` / `col` are the **block** indices (not element indices).
/// `data` stores elements in **row-major** order within the block.
#[derive(Clone, Debug)]
pub struct SparseBlock {
    pub row: usize,
    pub col: usize,
    pub data: Vec<f32>,
}

/// Block-sparse matrix: a list of dense sub-blocks of uniform size.
#[derive(Clone, Debug)]
pub struct BlockSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub blocks: Vec<SparseBlock>,
}

/// Compressed Sparse Row (CSR) matrix.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f32>,
}

/// 2:4 structured sparsity — for every group of 4 contiguous values in a
/// row, exactly 2 are non-zero. We store the two non-zero values plus a
/// 4-bit mask indicating which two positions are occupied.
#[derive(Clone, Debug)]
pub struct Structured24Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Two non-zero values per group, stored consecutively per row.
    pub values: Vec<f32>,
    /// One byte per group, low 4 bits = bitmask of non-zero positions.
    pub masks: Vec<u8>,
}

// ── 1. Block-sparse GEMM  C = A · B_sparse ─────────────────────────

/// Scalar block-sparse GEMM: `C[m×n] += A[m×k] · B_sparse[k×n]`.
pub fn block_sparse_gemm_scalar(
    a: &[f32],
    b: &BlockSparseMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for i in 0..m {
            for jb in 0..bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                let mut sum = 0.0f32;
                for kb in 0..bs {
                    let row = br + kb;
                    if row >= k {
                        break;
                    }
                    sum += a[i * k + row] * blk.data[kb * bs + jb];
                }
                c[i * n + col] += sum;
            }
        }
    }
}

/// NEON block-sparse GEMM: `C[m×n] += A[m×k] · B_sparse[k×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn block_sparse_gemm_neon(
    a: &[f32],
    b: &BlockSparseMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for i in 0..m {
            // Process columns in chunks of 4 using NEON.
            let mut jb = 0usize;
            while jb + 4 <= bs && bc + jb + 4 <= n {
                let mut acc = vdupq_n_f32(0.0);
                for kb in 0..bs {
                    let row = br + kb;
                    if row >= k {
                        break;
                    }
                    let a_val = vdupq_n_f32(a[i * k + row]);
                    let b_vals = vld1q_f32(blk.data.as_ptr().add(kb * bs + jb));
                    acc = vfmaq_f32(acc, a_val, b_vals);
                }
                let col = bc + jb;
                let prev = vld1q_f32(c.as_ptr().add(i * n + col));
                vst1q_f32(c.as_mut_ptr().add(i * n + col), vaddq_f32(prev, acc));
                jb += 4;
            }
            // Scalar tail.
            while jb < bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                let mut sum = 0.0f32;
                for kb in 0..bs {
                    let row = br + kb;
                    if row >= k {
                        break;
                    }
                    sum += a[i * k + row] * blk.data[kb * bs + jb];
                }
                c[i * n + col] += sum;
                jb += 1;
            }
        }
    }
}

// ── 2. CSR sparse GEMM  C = A · B_csr ──────────────────────────────

/// Scalar CSR GEMM: `C[m×n] += A[m×k] · B_csr[k×n]`.
pub fn csr_sparse_gemm_scalar(
    a: &[f32],
    b: &CsrMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    for row_b in 0..k {
        let start = b.row_ptr[row_b];
        let end = b.row_ptr[row_b + 1];
        for i in 0..m {
            let a_val = a[i * k + row_b];
            if a_val == 0.0 {
                continue;
            }
            for idx in start..end {
                let col = b.col_idx[idx];
                c[i * n + col] += a_val * b.values[idx];
            }
        }
    }
}

/// NEON CSR GEMM: `C[m×n] += A[m×k] · B_csr[k×n]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn csr_sparse_gemm_neon(
    a: &[f32],
    b: &CsrMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    for row_b in 0..k {
        let start = b.row_ptr[row_b];
        let end = b.row_ptr[row_b + 1];
        if start == end {
            continue;
        }
        for i in 0..m {
            let a_val = a[i * k + row_b];
            if a_val == 0.0 {
                continue;
            }
            let va = vdupq_n_f32(a_val);
            let mut idx = start;
            // Process NNZ in groups of 4 with NEON gather-scatter.
            while idx + 4 <= end {
                let c0 = b.col_idx[idx];
                let c1 = b.col_idx[idx + 1];
                let c2 = b.col_idx[idx + 2];
                let c3 = b.col_idx[idx + 3];
                let bv = vld1q_f32(b.values.as_ptr().add(idx));
                let prod = vmulq_f32(va, bv);
                let mut tmp: [f32; 4] = [0.0; 4];
                vst1q_f32(tmp.as_mut_ptr(), prod);
                c[i * n + c0] += tmp[0];
                c[i * n + c1] += tmp[1];
                c[i * n + c2] += tmp[2];
                c[i * n + c3] += tmp[3];
                idx += 4;
            }
            // Scalar tail.
            while idx < end {
                let col = b.col_idx[idx];
                c[i * n + col] += a_val * b.values[idx];
                idx += 1;
            }
        }
    }
}

// ── 3. 2:4 structured sparse GEMM  C = A · B_24 ────────────────────

/// Scalar 2:4 structured sparse GEMM: `C[m×n] += A[m×k] · B_24[k×n]`.
///
/// `B_24` has `k` rows and `n` columns; every consecutive group of 4
/// columns has exactly 2 non-zero entries.
pub fn structured_24_gemm_scalar(
    a: &[f32],
    b: &Structured24Matrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    let groups_per_row = n.div_ceil(4);
    for row_b in 0..k {
        for g in 0..groups_per_row {
            let mask = b.masks[row_b * groups_per_row + g];
            let val_base = (row_b * groups_per_row + g) * 2;
            let mut vi = 0usize;
            for bit in 0..4u8 {
                if mask & (1 << bit) != 0 {
                    let col = g * 4 + bit as usize;
                    if col >= n {
                        break;
                    }
                    let b_val = b.values[val_base + vi];
                    vi += 1;
                    for i in 0..m {
                        c[i * n + col] += a[i * k + row_b] * b_val;
                    }
                }
            }
        }
    }
}

/// NEON 2:4 structured sparse GEMM.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn structured_24_gemm_neon(
    a: &[f32],
    b: &Structured24Matrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= m * n);

    let groups_per_row = n.div_ceil(4);
    for row_b in 0..k {
        for g in 0..groups_per_row {
            let mask = b.masks[row_b * groups_per_row + g];
            let val_base = (row_b * groups_per_row + g) * 2;
            // Expand 2:4 group into a full 4-wide vector.
            let mut expanded = [0.0f32; 4];
            let mut vi = 0usize;
            for bit in 0..4u8 {
                if mask & (1 << bit) != 0 {
                    expanded[bit as usize] = b.values[val_base + vi];
                    vi += 1;
                }
            }
            let vb = vld1q_f32(expanded.as_ptr());
            let base_col = g * 4;
            for i in 0..m {
                let va = vdupq_n_f32(a[i * k + row_b]);
                let prod = vmulq_f32(va, vb);
                // Accumulate — handle tail columns carefully.
                let remaining = (n - base_col).min(4);
                if remaining == 4 {
                    let prev = vld1q_f32(c.as_ptr().add(i * n + base_col));
                    vst1q_f32(c.as_mut_ptr().add(i * n + base_col), vaddq_f32(prev, prod));
                } else {
                    let mut tmp = [0.0f32; 4];
                    vst1q_f32(tmp.as_mut_ptr(), prod);
                    for j in 0..remaining {
                        c[i * n + base_col + j] += tmp[j];
                    }
                }
            }
        }
    }
}

// ── 4. Block-sparse GEMV  y = B_sparse^T · x ───────────────────────

/// Scalar block-sparse GEMV: `y[n] += B_sparse[k×n]^T · x[k]`.
///
/// This computes `y = B^T · x`, i.e., for each non-zero block we
/// accumulate contributions to the output vector `y`.
pub fn block_sparse_gemv_scalar(
    x: &[f32],
    b: &BlockSparseMatrix,
    y: &mut [f32],
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(x.len() >= k);
    debug_assert!(y.len() >= n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for kb in 0..bs {
            let row = br + kb;
            if row >= k {
                break;
            }
            let xv = x[row];
            for jb in 0..bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                y[col] += xv * blk.data[kb * bs + jb];
            }
        }
    }
}

/// NEON block-sparse GEMV.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn block_sparse_gemv_neon(
    x: &[f32],
    b: &BlockSparseMatrix,
    y: &mut [f32],
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, k);
    debug_assert_eq!(b.cols, n);
    debug_assert!(x.len() >= k);
    debug_assert!(y.len() >= n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for kb in 0..bs {
            let row = br + kb;
            if row >= k {
                break;
            }
            let vx = vdupq_n_f32(x[row]);
            let mut jb = 0usize;
            while jb + 4 <= bs && bc + jb + 4 <= n {
                let vb = vld1q_f32(blk.data.as_ptr().add(kb * bs + jb));
                let col = bc + jb;
                let prev = vld1q_f32(y.as_ptr().add(col));
                vst1q_f32(y.as_mut_ptr().add(col), vfmaq_f32(prev, vx, vb));
                jb += 4;
            }
            while jb < bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                y[col] += x[row] * blk.data[kb * bs + jb];
                jb += 1;
            }
        }
    }
}

// ── 5. Sparse-activation GEMM  C = ReLU-sparse(A) · B ──────────────

/// Scalar sparse-activation GEMM: `C[m×n] += relu_mask(A)[m×k] · B[k×n]`.
///
/// `mask[i*k + j]` is `true` iff `A[i,j]` survived ReLU (> 0). Skips
/// zero-activation rows of the dot product for free.
pub fn sparse_activation_gemm_scalar(
    a: &[f32],
    mask: &[bool],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(mask.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    for i in 0..m {
        for j in 0..k {
            if !mask[i * k + j] {
                continue;
            }
            let a_val = a[i * k + j];
            for col in 0..n {
                c[i * n + col] += a_val * b[j * n + col];
            }
        }
    }
}

/// NEON sparse-activation GEMM.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_activation_gemm_neon(
    a: &[f32],
    mask: &[bool],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(mask.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);

    for i in 0..m {
        for j in 0..k {
            if !mask[i * k + j] {
                continue;
            }
            let va = vdupq_n_f32(a[i * k + j]);
            let mut col = 0usize;
            while col + 4 <= n {
                let vb = vld1q_f32(b.as_ptr().add(j * n + col));
                let prev = vld1q_f32(c.as_ptr().add(i * n + col));
                vst1q_f32(c.as_mut_ptr().add(i * n + col), vfmaq_f32(prev, va, vb));
                col += 4;
            }
            while col < n {
                c[i * n + col] += a[i * k + j] * b[j * n + col];
                col += 1;
            }
        }
    }
}

// ── 6. Sparse transpose multiply  C = A^T · B_sparse ───────────────

/// Scalar sparse transpose multiply: `C[k×n] += A[m×k]^T · B_sparse[m×n]`.
pub fn sparse_transpose_multiply_scalar(
    a: &[f32],
    b: &BlockSparseMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, m);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= k * n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for mb in 0..bs {
            let row_m = br + mb;
            if row_m >= m {
                break;
            }
            for jb in 0..bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                let b_val = blk.data[mb * bs + jb];
                for ki in 0..k {
                    c[ki * n + col] += a[row_m * k + ki] * b_val;
                }
            }
        }
    }
}

/// NEON sparse transpose multiply.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sparse_transpose_multiply_neon(
    a: &[f32],
    b: &BlockSparseMatrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(b.rows, m);
    debug_assert_eq!(b.cols, n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(c.len() >= k * n);

    let bs = b.block_size;
    for blk in &b.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for mb in 0..bs {
            let row_m = br + mb;
            if row_m >= m {
                break;
            }
            // Process block columns in groups of 4.
            let mut jb = 0usize;
            while jb + 4 <= bs && bc + jb + 4 <= n {
                let vb = vld1q_f32(blk.data.as_ptr().add(mb * bs + jb));
                let col_base = bc + jb;
                for ki in 0..k {
                    let va = vdupq_n_f32(a[row_m * k + ki]);
                    let prev = vld1q_f32(c.as_ptr().add(ki * n + col_base));
                    vst1q_f32(c.as_mut_ptr().add(ki * n + col_base), vfmaq_f32(prev, va, vb));
                }
                jb += 4;
            }
            while jb < bs {
                let col = bc + jb;
                if col >= n {
                    break;
                }
                let b_val = blk.data[mb * bs + jb];
                for ki in 0..k {
                    c[ki * n + col] += a[row_m * k + ki] * b_val;
                }
                jb += 1;
            }
        }
    }
}

// ── Helper: dense matmul reference (for tests) ──────────────────────

#[cfg(test)]
fn dense_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

// ── Helper: convert BlockSparseMatrix to dense ──────────────────────

#[cfg(test)]
fn block_sparse_to_dense(bsm: &BlockSparseMatrix) -> Vec<f32> {
    let mut dense = vec![0.0f32; bsm.rows * bsm.cols];
    let bs = bsm.block_size;
    for blk in &bsm.blocks {
        let br = blk.row * bs;
        let bc = blk.col * bs;
        for r in 0..bs {
            for c in 0..bs {
                if br + r < bsm.rows && bc + c < bsm.cols {
                    dense[(br + r) * bsm.cols + bc + c] = blk.data[r * bs + c];
                }
            }
        }
    }
    dense
}

// ── Helper: convert CsrMatrix to dense ──────────────────────────────

#[cfg(test)]
fn csr_to_dense(csr: &CsrMatrix) -> Vec<f32> {
    let mut dense = vec![0.0f32; csr.rows * csr.cols];
    for r in 0..csr.rows {
        for idx in csr.row_ptr[r]..csr.row_ptr[r + 1] {
            dense[r * csr.cols + csr.col_idx[idx]] = csr.values[idx];
        }
    }
    dense
}

// ── Helper: convert Structured24Matrix to dense ─────────────────────

#[cfg(test)]
fn structured_24_to_dense(s: &Structured24Matrix) -> Vec<f32> {
    let mut dense = vec![0.0f32; s.rows * s.cols];
    let groups_per_row = s.cols.div_ceil(4);
    for r in 0..s.rows {
        for g in 0..groups_per_row {
            let mask = s.masks[r * groups_per_row + g];
            let val_base = (r * groups_per_row + g) * 2;
            let mut vi = 0;
            for bit in 0..4u8 {
                if mask & (1 << bit) != 0 {
                    let col = g * 4 + bit as usize;
                    if col < s.cols {
                        dense[r * s.cols + col] = s.values[val_base + vi];
                    }
                    vi += 1;
                }
            }
        }
    }
    dense
}

// ── Helper: make a random BlockSparseMatrix ─────────────────────────

#[cfg(test)]
fn make_block_sparse(
    rows: usize,
    cols: usize,
    block_size: usize,
    density: f64,
    seed: u64,
) -> BlockSparseMatrix {
    let block_rows = rows.div_ceil(block_size);
    let block_cols = cols.div_ceil(block_size);
    let mut blocks = Vec::new();
    let mut rng = seed;
    for br in 0..block_rows {
        for bc in 0..block_cols {
            // Simple LCG PRNG for reproducibility.
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let prob = (rng >> 33) as f64 / (1u64 << 31) as f64;
            if prob < density {
                let mut data = vec![0.0f32; block_size * block_size];
                for v in &mut data {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *v = ((rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32;
                }
                blocks.push(SparseBlock { row: br, col: bc, data });
            }
        }
    }
    BlockSparseMatrix { rows, cols, block_size, blocks }
}

// ── Helper: make a random CsrMatrix ─────────────────────────────────

#[cfg(test)]
fn make_csr(rows: usize, cols: usize, density: f64, seed: u64) -> CsrMatrix {
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut rng = seed;
    for _r in 0..rows {
        for c in 0..cols {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let prob = (rng >> 33) as f64 / (1u64 << 31) as f64;
            if prob < density {
                col_idx.push(c);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32;
                values.push(v);
            }
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix { rows, cols, row_ptr, col_idx, values }
}

// ── Helper: make a 2:4 structured sparse matrix ─────────────────────

#[cfg(test)]
fn make_structured_24(rows: usize, cols: usize, seed: u64) -> Structured24Matrix {
    let groups_per_row = cols.div_ceil(4);
    let mut values = Vec::new();
    let mut masks = Vec::new();
    let mut rng = seed;

    for _r in 0..rows {
        for g in 0..groups_per_row {
            let remaining = cols - g * 4;
            // Pick 2 positions out of min(4, remaining).
            let width = remaining.min(4);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p0 = (rng >> 33) as usize % width;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut p1 = (rng >> 33) as usize % width;
            if p1 == p0 {
                p1 = (p0 + 1) % width;
            }
            let (lo, hi) = if p0 < p1 { (p0, p1) } else { (p1, p0) };
            let mask = (1u8 << lo) | (1u8 << hi);
            masks.push(mask);
            // Values for the two non-zero positions (low index first).
            for _ in 0..2 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32;
                values.push(v);
            }
        }
    }
    Structured24Matrix { rows, cols, values, masks }
}

// ── Helper: random dense matrix ─────────────────────────────────────

#[cfg(test)]
fn rand_dense(len: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut rng = seed;
    for _ in 0..len {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        out.push(((rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32);
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }

    // ── 1. Block-sparse GEMM tests ──────────────────────────────────

    #[test]
    fn test_block_sparse_gemm_scalar_4x4_identity() {
        let bs = 4;
        let n = 4;
        let b = BlockSparseMatrix {
            rows: n,
            cols: n,
            block_size: bs,
            blocks: vec![SparseBlock {
                row: 0,
                col: 0,
                data: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            }],
        };
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 8];
        block_sparse_gemm_scalar(&a, &b, &mut c, 2, 4, 4);
        assert!(approx_eq(&c, &a, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_vs_dense_4x4() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 1);
        let bsm = make_block_sparse(k, n, 4, 0.5, 2);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_vs_dense_8x8() {
        let (m, k, n) = (8, 16, 16);
        let a = rand_dense(m * k, 3);
        let bsm = make_block_sparse(k, n, 8, 0.5, 4);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_vs_dense_16x16() {
        let (m, k, n) = (4, 32, 32);
        let a = rand_dense(m * k, 5);
        let bsm = make_block_sparse(k, n, 16, 0.3, 6);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_empty_blocks() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let a = rand_dense(8, 7);
        let mut c = vec![0.0; 8];
        block_sparse_gemm_scalar(&a, &b, &mut c, 2, 4, 4);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_accumulates() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 8);
        let bsm = make_block_sparse(k, n, 4, 1.0, 9);
        let mut c = vec![1.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c, m, k, n);
        // Should be > 1.0 in at least some entries (accumulated onto 1.0).
        assert!(c.iter().any(|&v| v != 1.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_vs_scalar_4x4() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 10);
        let bsm = make_block_sparse(k, n, 4, 0.6, 11);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_vs_scalar_8x8() {
        let (m, k, n) = (8, 16, 16);
        let a = rand_dense(m * k, 12);
        let bsm = make_block_sparse(k, n, 8, 0.5, 13);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_vs_scalar_16x16() {
        let (m, k, n) = (4, 32, 32);
        let a = rand_dense(m * k, 14);
        let bsm = make_block_sparse(k, n, 16, 0.4, 15);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_vs_dense() {
        let (m, k, n) = (6, 12, 12);
        let a = rand_dense(m * k, 16);
        let bsm = make_block_sparse(k, n, 4, 0.7, 17);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_neon = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_neon, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_empty() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let a = rand_dense(8, 18);
        let mut c = vec![0.0; 8];
        unsafe { block_sparse_gemm_neon(&a, &b, &mut c, 2, 4, 4) };
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_non_square() {
        let (m, k, n) = (3, 12, 8);
        let a = rand_dense(m * k, 19);
        let bsm = make_block_sparse(k, n, 4, 0.5, 20);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_non_square() {
        let (m, k, n) = (3, 12, 8);
        let a = rand_dense(m * k, 21);
        let bsm = make_block_sparse(k, n, 4, 0.5, 22);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_full_density() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 23);
        let bsm = make_block_sparse(k, n, 4, 1.0, 24);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    // ── 2. CSR GEMM tests ───────────────────────────────────────────

    #[test]
    fn test_csr_gemm_scalar_identity() {
        let csr = CsrMatrix {
            rows: 3,
            cols: 3,
            row_ptr: vec![0, 1, 2, 3],
            col_idx: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        };
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0; 6];
        csr_sparse_gemm_scalar(&a, &csr, &mut c, 2, 3, 3);
        assert!(approx_eq(&c, &a, EPS));
    }

    #[test]
    fn test_csr_gemm_scalar_vs_dense() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 30);
        let csr = make_csr(k, n, 0.4, 31);
        let b_dense = csr_to_dense(&csr);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_csr_gemm_scalar_empty() {
        let csr = CsrMatrix {
            rows: 3,
            cols: 3,
            row_ptr: vec![0, 0, 0, 0],
            col_idx: vec![],
            values: vec![],
        };
        let a = rand_dense(6, 32);
        let mut c = vec![0.0; 6];
        csr_sparse_gemm_scalar(&a, &csr, &mut c, 2, 3, 3);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_csr_gemm_scalar_single_element() {
        let csr = CsrMatrix {
            rows: 2,
            cols: 2,
            row_ptr: vec![0, 1, 1],
            col_idx: vec![1],
            values: vec![3.0],
        };
        let a = vec![1.0, 0.0, 0.0, 2.0];
        let mut c = vec![0.0; 4];
        csr_sparse_gemm_scalar(&a, &csr, &mut c, 2, 2, 2);
        // Row 0 of A = [1, 0], B has (0,1)=3. C[0,1] = 1*3 = 3.
        assert!((c[1] - 3.0).abs() < EPS);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_vs_scalar() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 33);
        let csr = make_csr(k, n, 0.5, 34);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_scalar, m, k, n);
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_vs_dense() {
        let (m, k, n) = (6, 12, 12);
        let a = rand_dense(m * k, 35);
        let csr = make_csr(k, n, 0.3, 36);
        let b_dense = csr_to_dense(&csr);
        let mut c_neon = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c_neon, m, k, n) };
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_neon, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_empty() {
        let csr = CsrMatrix {
            rows: 3,
            cols: 3,
            row_ptr: vec![0, 0, 0, 0],
            col_idx: vec![],
            values: vec![],
        };
        let a = rand_dense(6, 37);
        let mut c = vec![0.0; 6];
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c, 2, 3, 3) };
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_csr_gemm_scalar_high_density() {
        let (m, k, n) = (4, 6, 6);
        let a = rand_dense(m * k, 38);
        let csr = make_csr(k, n, 0.9, 39);
        let b_dense = csr_to_dense(&csr);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_non_square() {
        let (m, k, n) = (3, 10, 6);
        let a = rand_dense(m * k, 40);
        let csr = make_csr(k, n, 0.4, 41);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_scalar, m, k, n);
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    // ── 3. 2:4 structured sparse GEMM tests ────────────────────────

    #[test]
    fn test_structured_24_gemm_scalar_vs_dense() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 50);
        let s24 = make_structured_24(k, n, 51);
        let b_dense = structured_24_to_dense(&s24);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_structured_24_gemm_scalar_small() {
        let (m, k, n) = (1, 4, 4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let s24 = make_structured_24(k, n, 52);
        let b_dense = structured_24_to_dense(&s24);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_structured_24_sparsity_ratio() {
        let s24 = make_structured_24(8, 8, 53);
        let dense = structured_24_to_dense(&s24);
        let nnz = dense.iter().filter(|&&v| v != 0.0).count();
        let total = 8 * 8;
        let ratio = nnz as f64 / total as f64;
        assert!((ratio - 0.5).abs() < 0.1, "Expected ~50% sparsity, got {ratio}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_structured_24_gemm_neon_vs_scalar() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 54);
        let s24 = make_structured_24(k, n, 55);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_scalar, m, k, n);
        unsafe { structured_24_gemm_neon(&a, &s24, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_structured_24_gemm_neon_vs_dense() {
        let (m, k, n) = (6, 12, 12);
        let a = rand_dense(m * k, 56);
        let s24 = make_structured_24(k, n, 57);
        let b_dense = structured_24_to_dense(&s24);
        let mut c_neon = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        unsafe { structured_24_gemm_neon(&a, &s24, &mut c_neon, m, k, n) };
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_neon, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_structured_24_gemm_neon_large() {
        let (m, k, n) = (8, 32, 32);
        let a = rand_dense(m * k, 58);
        let s24 = make_structured_24(k, n, 59);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_scalar, m, k, n);
        unsafe { structured_24_gemm_neon(&a, &s24, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_structured_24_gemm_scalar_non_multiple_of_4_cols() {
        let (m, k, n) = (2, 4, 6);
        let a = rand_dense(m * k, 60);
        let s24 = make_structured_24(k, n, 61);
        let b_dense = structured_24_to_dense(&s24);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_structured_24_gemm_neon_non_multiple_of_4_cols() {
        let (m, k, n) = (2, 4, 6);
        let a = rand_dense(m * k, 62);
        let s24 = make_structured_24(k, n, 63);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c_scalar, m, k, n);
        unsafe { structured_24_gemm_neon(&a, &s24, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    // ── 4. Block-sparse GEMV tests ──────────────────────────────────

    #[test]
    fn test_block_sparse_gemv_scalar_identity() {
        let b = BlockSparseMatrix {
            rows: 4,
            cols: 4,
            block_size: 4,
            blocks: vec![SparseBlock {
                row: 0,
                col: 0,
                data: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            }],
        };
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        block_sparse_gemv_scalar(&x, &b, &mut y, 4, 4);
        assert!(approx_eq(&y, &x, EPS));
    }

    #[test]
    fn test_block_sparse_gemv_scalar_vs_dense() {
        let (k, n) = (8, 8);
        let x = rand_dense(k, 70);
        let bsm = make_block_sparse(k, n, 4, 0.5, 71);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut y_sparse = vec![0.0; n];
        let mut y_dense = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_sparse, k, n);
        // Dense: y = B^T * x
        for j in 0..n {
            for i in 0..k {
                y_dense[j] += b_dense[i * n + j] * x[i];
            }
        }
        assert!(approx_eq(&y_sparse, &y_dense, EPS));
    }

    #[test]
    fn test_block_sparse_gemv_scalar_empty() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let x = rand_dense(4, 72);
        let mut y = vec![0.0; 4];
        block_sparse_gemv_scalar(&x, &b, &mut y, 4, 4);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemv_neon_vs_scalar() {
        let (k, n) = (8, 8);
        let x = rand_dense(k, 73);
        let bsm = make_block_sparse(k, n, 4, 0.6, 74);
        let mut y_scalar = vec![0.0; n];
        let mut y_neon = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_scalar, k, n);
        unsafe { block_sparse_gemv_neon(&x, &bsm, &mut y_neon, k, n) };
        assert!(approx_eq(&y_neon, &y_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemv_neon_vs_scalar_8x8() {
        let (k, n) = (16, 16);
        let x = rand_dense(k, 75);
        let bsm = make_block_sparse(k, n, 8, 0.5, 76);
        let mut y_scalar = vec![0.0; n];
        let mut y_neon = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_scalar, k, n);
        unsafe { block_sparse_gemv_neon(&x, &bsm, &mut y_neon, k, n) };
        assert!(approx_eq(&y_neon, &y_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemv_neon_empty() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let x = rand_dense(4, 77);
        let mut y = vec![0.0; 4];
        unsafe { block_sparse_gemv_neon(&x, &b, &mut y, 4, 4) };
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_block_sparse_gemv_scalar_accumulates() {
        let (k, n) = (4, 4);
        let x = rand_dense(k, 78);
        let bsm = make_block_sparse(k, n, 4, 1.0, 79);
        let mut y = vec![1.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y, k, n);
        assert!(y.iter().any(|&v| v != 1.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemv_neon_non_square() {
        let (k, n) = (12, 8);
        let x = rand_dense(k, 80);
        let bsm = make_block_sparse(k, n, 4, 0.5, 81);
        let mut y_scalar = vec![0.0; n];
        let mut y_neon = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_scalar, k, n);
        unsafe { block_sparse_gemv_neon(&x, &bsm, &mut y_neon, k, n) };
        assert!(approx_eq(&y_neon, &y_scalar, EPS));
    }

    // ── 5. Sparse-activation GEMM tests ─────────────────────────────

    #[test]
    fn test_sparse_activation_gemm_scalar_vs_dense_all_active() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 90);
        let b = rand_dense(k * n, 91);
        let mask = vec![true; m * k];
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_sparse_activation_gemm_scalar_all_masked() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 92);
        let b = rand_dense(k * n, 93);
        let mask = vec![false; m * k];
        let mut c = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c, m, k, n);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sparse_activation_gemm_scalar_half_masked() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 94);
        let b = rand_dense(k * n, 95);
        let mut mask = vec![false; m * k];
        for i in (0..m * k).step_by(2) {
            mask[i] = true;
        }
        let mut c_sparse = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_sparse, m, k, n);
        // Build masked-A and do dense matmul for reference.
        let mut a_masked = a.clone();
        for i in 0..m * k {
            if !mask[i] {
                a_masked[i] = 0.0;
            }
        }
        let mut c_ref = vec![0.0; m * n];
        dense_matmul(&a_masked, &b, &mut c_ref, m, k, n);
        assert!(approx_eq(&c_sparse, &c_ref, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_activation_gemm_neon_vs_scalar() {
        let (m, k, n) = (4, 8, 8);
        let a = rand_dense(m * k, 96);
        let b = rand_dense(k * n, 97);
        let mask: Vec<bool> = (0..m * k).map(|i| i % 3 != 0).collect();
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_scalar, m, k, n);
        unsafe { sparse_activation_gemm_neon(&a, &mask, &b, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_activation_gemm_neon_vs_dense_all_active() {
        let (m, k, n) = (6, 12, 12);
        let a = rand_dense(m * k, 98);
        let b = rand_dense(k * n, 99);
        let mask = vec![true; m * k];
        let mut c_neon = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        unsafe { sparse_activation_gemm_neon(&a, &mask, &b, &mut c_neon, m, k, n) };
        dense_matmul(&a, &b, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_neon, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_activation_gemm_neon_all_masked() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 100);
        let b = rand_dense(k * n, 101);
        let mask = vec![false; m * k];
        let mut c = vec![0.0; m * n];
        unsafe { sparse_activation_gemm_neon(&a, &mask, &b, &mut c, m, k, n) };
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_activation_gemm_neon_non_multiple_of_4() {
        let (m, k, n) = (3, 5, 7);
        let a = rand_dense(m * k, 102);
        let b = rand_dense(k * n, 103);
        let mask: Vec<bool> = (0..m * k).map(|i| i % 2 == 0).collect();
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_scalar, m, k, n);
        unsafe { sparse_activation_gemm_neon(&a, &mask, &b, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    // ── 6. Sparse transpose multiply tests ──────────────────────────

    #[test]
    fn test_sparse_transpose_multiply_scalar_vs_dense() {
        let (m, k, n) = (8, 4, 8);
        let a = rand_dense(m * k, 110);
        let bsm = make_block_sparse(m, n, 4, 0.5, 111);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; k * n];
        let mut c_dense = vec![0.0; k * n];
        sparse_transpose_multiply_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        // Dense: C = A^T * B, where A is [m×k], A^T is [k×m], B is [m×n].
        for ki in 0..k {
            for j in 0..n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += a[i * k + ki] * b_dense[i * n + j];
                }
                c_dense[ki * n + j] += s;
            }
        }
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_sparse_transpose_multiply_scalar_empty() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let a = rand_dense(16, 112);
        let mut c = vec![0.0; 16];
        sparse_transpose_multiply_scalar(&a, &b, &mut c, 4, 4, 4);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sparse_transpose_multiply_scalar_identity_block() {
        let b = BlockSparseMatrix {
            rows: 4,
            cols: 4,
            block_size: 4,
            blocks: vec![SparseBlock {
                row: 0,
                col: 0,
                data: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            }],
        };
        // A is 4×2, B (identity) is 4×4 → C = A^T · I = A^T which is 2×4.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 8];
        sparse_transpose_multiply_scalar(&a, &b, &mut c, 4, 2, 4);
        // C[ki,j] = sum_i A[i,ki] * B[i,j]. B=I so C[ki,j] = A[j,ki].
        let expected = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        assert!(approx_eq(&c, &expected, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_transpose_multiply_neon_vs_scalar() {
        let (m, k, n) = (8, 4, 8);
        let a = rand_dense(m * k, 113);
        let bsm = make_block_sparse(m, n, 4, 0.5, 114);
        let mut c_scalar = vec![0.0; k * n];
        let mut c_neon = vec![0.0; k * n];
        sparse_transpose_multiply_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { sparse_transpose_multiply_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_transpose_multiply_neon_vs_dense() {
        let (m, k, n) = (12, 6, 12);
        let a = rand_dense(m * k, 115);
        let bsm = make_block_sparse(m, n, 4, 0.6, 116);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_neon = vec![0.0; k * n];
        let mut c_dense = vec![0.0; k * n];
        unsafe { sparse_transpose_multiply_neon(&a, &bsm, &mut c_neon, m, k, n) };
        for ki in 0..k {
            for j in 0..n {
                for i in 0..m {
                    c_dense[ki * n + j] += a[i * k + ki] * b_dense[i * n + j];
                }
            }
        }
        assert!(approx_eq(&c_neon, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_transpose_multiply_neon_empty() {
        let b = BlockSparseMatrix { rows: 4, cols: 4, block_size: 4, blocks: vec![] };
        let a = rand_dense(16, 117);
        let mut c = vec![0.0; 16];
        unsafe { sparse_transpose_multiply_neon(&a, &b, &mut c, 4, 4, 4) };
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_transpose_multiply_neon_8x8_blocks() {
        let (m, k, n) = (16, 4, 16);
        let a = rand_dense(m * k, 118);
        let bsm = make_block_sparse(m, n, 8, 0.5, 119);
        let mut c_scalar = vec![0.0; k * n];
        let mut c_neon = vec![0.0; k * n];
        sparse_transpose_multiply_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { sparse_transpose_multiply_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    // ── Edge-case / stress tests ────────────────────────────────────

    #[test]
    fn test_block_sparse_gemm_scalar_single_block() {
        let (m, k, n) = (1, 4, 4);
        let bsm = make_block_sparse(k, n, 4, 1.0, 130);
        let a = rand_dense(m * k, 131);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[test]
    fn test_csr_gemm_scalar_large() {
        let (m, k, n) = (16, 32, 16);
        let a = rand_dense(m * k, 132);
        let csr = make_csr(k, n, 0.2, 133);
        let b_dense = csr_to_dense(&csr);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_large() {
        let (m, k, n) = (16, 32, 16);
        let a = rand_dense(m * k, 134);
        let csr = make_csr(k, n, 0.2, 135);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c_scalar, m, k, n);
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_block_sparse_gemm_scalar_very_sparse() {
        let (m, k, n) = (8, 32, 32);
        let a = rand_dense(m * k, 136);
        let bsm = make_block_sparse(k, n, 4, 0.05, 137);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_sparse, m, k, n);
        dense_matmul(&a, &b_dense, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_very_sparse() {
        let (m, k, n) = (8, 32, 32);
        let a = rand_dense(m * k, 138);
        let bsm = make_block_sparse(k, n, 4, 0.05, 139);
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        block_sparse_gemm_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_sparse_activation_gemm_scalar_relu_pattern() {
        let (m, k, n) = (4, 8, 4);
        let a = rand_dense(m * k, 140);
        let b = rand_dense(k * n, 141);
        // Simulate ReLU: mask = a > 0.
        let mask: Vec<bool> = a.iter().map(|&v| v > 0.0).collect();
        let mut a_relu = a.clone();
        for i in 0..a_relu.len() {
            if !mask[i] {
                a_relu[i] = 0.0;
            }
        }
        let mut c_sparse = vec![0.0; m * n];
        let mut c_dense = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_sparse, m, k, n);
        dense_matmul(&a_relu, &b, &mut c_dense, m, k, n);
        assert!(approx_eq(&c_sparse, &c_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_activation_gemm_neon_relu_pattern() {
        let (m, k, n) = (4, 8, 4);
        let a = rand_dense(m * k, 142);
        let b = rand_dense(k * n, 143);
        let mask: Vec<bool> = a.iter().map(|&v| v > 0.0).collect();
        let mut c_scalar = vec![0.0; m * n];
        let mut c_neon = vec![0.0; m * n];
        sparse_activation_gemm_scalar(&a, &mask, &b, &mut c_scalar, m, k, n);
        unsafe { sparse_activation_gemm_neon(&a, &mask, &b, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_block_sparse_gemv_scalar_large() {
        let (k, n) = (32, 32);
        let x = rand_dense(k, 144);
        let bsm = make_block_sparse(k, n, 4, 0.3, 145);
        let b_dense = block_sparse_to_dense(&bsm);
        let mut y_sparse = vec![0.0; n];
        let mut y_dense = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_sparse, k, n);
        for j in 0..n {
            for i in 0..k {
                y_dense[j] += b_dense[i * n + j] * x[i];
            }
        }
        assert!(approx_eq(&y_sparse, &y_dense, EPS));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemv_neon_large() {
        let (k, n) = (32, 32);
        let x = rand_dense(k, 146);
        let bsm = make_block_sparse(k, n, 4, 0.3, 147);
        let mut y_scalar = vec![0.0; n];
        let mut y_neon = vec![0.0; n];
        block_sparse_gemv_scalar(&x, &bsm, &mut y_scalar, k, n);
        unsafe { block_sparse_gemv_neon(&x, &bsm, &mut y_neon, k, n) };
        assert!(approx_eq(&y_neon, &y_scalar, EPS));
    }

    #[test]
    fn test_structured_24_gemm_scalar_accumulates() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 148);
        let s24 = make_structured_24(k, n, 149);
        let mut c = vec![1.0; m * n];
        structured_24_gemm_scalar(&a, &s24, &mut c, m, k, n);
        assert!(c.iter().any(|&v| v != 1.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_block_sparse_gemm_neon_accumulates() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 150);
        let bsm = make_block_sparse(k, n, 4, 1.0, 151);
        let mut c = vec![1.0; m * n];
        unsafe { block_sparse_gemm_neon(&a, &bsm, &mut c, m, k, n) };
        assert!(c.iter().any(|&v| v != 1.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sparse_transpose_multiply_neon_non_square() {
        let (m, k, n) = (12, 3, 8);
        let a = rand_dense(m * k, 152);
        let bsm = make_block_sparse(m, n, 4, 0.5, 153);
        let mut c_scalar = vec![0.0; k * n];
        let mut c_neon = vec![0.0; k * n];
        sparse_transpose_multiply_scalar(&a, &bsm, &mut c_scalar, m, k, n);
        unsafe { sparse_transpose_multiply_neon(&a, &bsm, &mut c_neon, m, k, n) };
        assert!(approx_eq(&c_neon, &c_scalar, EPS));
    }

    #[test]
    fn test_csr_gemm_scalar_accumulates() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 154);
        let csr = make_csr(k, n, 0.5, 155);
        let mut c = vec![1.0; m * n];
        csr_sparse_gemm_scalar(&a, &csr, &mut c, m, k, n);
        assert!(c.iter().any(|&v| v != 1.0));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_csr_gemm_neon_accumulates() {
        let (m, k, n) = (2, 4, 4);
        let a = rand_dense(m * k, 156);
        let csr = make_csr(k, n, 0.5, 157);
        let mut c = vec![1.0; m * n];
        unsafe { csr_sparse_gemm_neon(&a, &csr, &mut c, m, k, n) };
        assert!(c.iter().any(|&v| v != 1.0));
    }
}
