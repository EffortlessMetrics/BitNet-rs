//! NEON-optimized matrix transpose operations for Apple Silicon (aarch64).
//!
//! Provides six operations:
//! 1. Full matrix transpose using NEON 4×4 blocks with vtrn/vzip
//! 2. In-place square matrix transpose
//! 3. Cache-friendly blocked transpose
//! 4. Batched transpose for attention heads
//! 5. Generalized dimension permutation (like numpy transpose with axes)
//! 6. Contiguous layout check

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    unused_variables,
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_memcpy,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::let_and_return,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::missing_safety_doc,
    clippy::never_loop,
    clippy::while_immutable_condition,
    clippy::manual_abs_diff
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── 1. Full Matrix Transpose ────────────────────────────────────────

/// NEON-accelerated 4×4 block transpose kernel.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_transpose_f32(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    let full_bi = rows / 4 * 4;
    let full_bj = cols / 4 * 4;

    // NEON 4×4 blocks
    for bi in (0..full_bi).step_by(4) {
        for bj in (0..full_bj).step_by(4) {
            unsafe {
                let r0 = vld1q_f32(input.as_ptr().add(bi * cols + bj));
                let r1 = vld1q_f32(input.as_ptr().add((bi + 1) * cols + bj));
                let r2 = vld1q_f32(input.as_ptr().add((bi + 2) * cols + bj));
                let r3 = vld1q_f32(input.as_ptr().add((bi + 3) * cols + bj));

                let t0 = vtrn1q_f32(r0, r1);
                let t1 = vtrn2q_f32(r0, r1);
                let t2 = vtrn1q_f32(r2, r3);
                let t3 = vtrn2q_f32(r2, r3);

                let t0_64 = vreinterpretq_f64_f32(t0);
                let t1_64 = vreinterpretq_f64_f32(t1);
                let t2_64 = vreinterpretq_f64_f32(t2);
                let t3_64 = vreinterpretq_f64_f32(t3);

                let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

                vst1q_f32(output.as_mut_ptr().add(bj * rows + bi), o0);
                vst1q_f32(output.as_mut_ptr().add((bj + 1) * rows + bi), o1);
                vst1q_f32(output.as_mut_ptr().add((bj + 2) * rows + bi), o2);
                vst1q_f32(output.as_mut_ptr().add((bj + 3) * rows + bi), o3);
            }
        }
    }

    // Scalar tail: right edge columns
    for i in 0..full_bi {
        for j in full_bj..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    // Scalar tail: bottom edge rows
    for i in full_bi..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

fn scalar_transpose_f32(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/// Transpose a row-major `rows × cols` matrix to `cols × rows`.
///
/// Uses NEON 4×4 block transpose on aarch64 with scalar fallback for
/// tail elements. Falls back to pure scalar on other architectures.
///
/// # Panics
///
/// Panics if slice lengths are less than `rows * cols`.
pub fn transpose_f32(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    let numel = rows * cols;
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    assert!(output.len() >= numel, "output length {} < rows*cols {numel}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detected above
            unsafe {
                neon_transpose_f32(input, output, rows, cols);
            }
            return;
        }
    }
    scalar_transpose_f32(input, output, rows, cols);
}

// ── 2. In-place Square Matrix Transpose ─────────────────────────────

/// NEON-accelerated in-place square matrix transpose.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_transpose_inplace_f32(data: &mut [f32], n: usize) {
    let full_b = n / 4 * 4;

    // Process 4×4 diagonal and upper-triangle blocks via NEON
    for bi in (0..full_b).step_by(4) {
        for bj in (bi..full_b).step_by(4) {
            if bi == bj {
                // On-diagonal block: transpose in place
                unsafe {
                    let ptr = data.as_mut_ptr();
                    let r0 = vld1q_f32(ptr.add(bi * n + bj));
                    let r1 = vld1q_f32(ptr.add((bi + 1) * n + bj));
                    let r2 = vld1q_f32(ptr.add((bi + 2) * n + bj));
                    let r3 = vld1q_f32(ptr.add((bi + 3) * n + bj));

                    let t0 = vtrn1q_f32(r0, r1);
                    let t1 = vtrn2q_f32(r0, r1);
                    let t2 = vtrn1q_f32(r2, r3);
                    let t3 = vtrn2q_f32(r2, r3);

                    let t0_64 = vreinterpretq_f64_f32(t0);
                    let t1_64 = vreinterpretq_f64_f32(t1);
                    let t2_64 = vreinterpretq_f64_f32(t2);
                    let t3_64 = vreinterpretq_f64_f32(t3);

                    let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                    let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                    let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                    let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

                    vst1q_f32(ptr.add(bi * n + bj), o0);
                    vst1q_f32(ptr.add((bi + 1) * n + bj), o1);
                    vst1q_f32(ptr.add((bi + 2) * n + bj), o2);
                    vst1q_f32(ptr.add((bi + 3) * n + bj), o3);
                }
            } else {
                // Off-diagonal: swap block (bi,bj) with block (bj,bi)
                unsafe {
                    let ptr = data.as_mut_ptr();
                    // Load upper block at (bi, bj)
                    let u0 = vld1q_f32(ptr.add(bi * n + bj));
                    let u1 = vld1q_f32(ptr.add((bi + 1) * n + bj));
                    let u2 = vld1q_f32(ptr.add((bi + 2) * n + bj));
                    let u3 = vld1q_f32(ptr.add((bi + 3) * n + bj));
                    // Load lower block at (bj, bi)
                    let l0 = vld1q_f32(ptr.add(bj * n + bi));
                    let l1 = vld1q_f32(ptr.add((bj + 1) * n + bi));
                    let l2 = vld1q_f32(ptr.add((bj + 2) * n + bi));
                    let l3 = vld1q_f32(ptr.add((bj + 3) * n + bi));

                    // Transpose upper block
                    let t0 = vtrn1q_f32(u0, u1);
                    let t1 = vtrn2q_f32(u0, u1);
                    let t2 = vtrn1q_f32(u2, u3);
                    let t3 = vtrn2q_f32(u2, u3);
                    let t0_64 = vreinterpretq_f64_f32(t0);
                    let t1_64 = vreinterpretq_f64_f32(t1);
                    let t2_64 = vreinterpretq_f64_f32(t2);
                    let t3_64 = vreinterpretq_f64_f32(t3);
                    let ut0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                    let ut1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                    let ut2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                    let ut3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

                    // Transpose lower block
                    let s0 = vtrn1q_f32(l0, l1);
                    let s1 = vtrn2q_f32(l0, l1);
                    let s2 = vtrn1q_f32(l2, l3);
                    let s3 = vtrn2q_f32(l2, l3);
                    let s0_64 = vreinterpretq_f64_f32(s0);
                    let s1_64 = vreinterpretq_f64_f32(s1);
                    let s2_64 = vreinterpretq_f64_f32(s2);
                    let s3_64 = vreinterpretq_f64_f32(s3);
                    let lt0 = vreinterpretq_f32_f64(vtrn1q_f64(s0_64, s2_64));
                    let lt1 = vreinterpretq_f32_f64(vtrn1q_f64(s1_64, s3_64));
                    let lt2 = vreinterpretq_f32_f64(vtrn2q_f64(s0_64, s2_64));
                    let lt3 = vreinterpretq_f32_f64(vtrn2q_f64(s1_64, s3_64));

                    // Store transposed upper → lower position, transposed lower → upper
                    vst1q_f32(ptr.add(bj * n + bi), ut0);
                    vst1q_f32(ptr.add((bj + 1) * n + bi), ut1);
                    vst1q_f32(ptr.add((bj + 2) * n + bi), ut2);
                    vst1q_f32(ptr.add((bj + 3) * n + bi), ut3);

                    vst1q_f32(ptr.add(bi * n + bj), lt0);
                    vst1q_f32(ptr.add((bi + 1) * n + bj), lt1);
                    vst1q_f32(ptr.add((bi + 2) * n + bj), lt2);
                    vst1q_f32(ptr.add((bi + 3) * n + bj), lt3);
                }
            }
        }
    }

    // Scalar tail: elements in rows/cols beyond the 4-aligned boundary
    for i in 0..n {
        let start = if i < full_b { full_b } else { i + 1 };
        for j in start..n {
            data.swap(i * n + j, j * n + i);
        }
    }
}

fn scalar_transpose_inplace_f32(data: &mut [f32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            data.swap(i * n + j, j * n + i);
        }
    }
}

/// In-place transpose of a square `n × n` row-major matrix.
///
/// # Panics
///
/// Panics if `data.len() < n * n`.
pub fn transpose_inplace_f32(data: &mut [f32], n: usize) {
    assert!(data.len() >= n * n, "data length {} < n*n {}", data.len(), n * n);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_transpose_inplace_f32(data, n);
            }
            return;
        }
    }
    scalar_transpose_inplace_f32(data, n);
}

// ── 3. Cache-Friendly Blocked Transpose ─────────────────────────────

/// NEON-accelerated blocked transpose with configurable block size.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_transpose_blocked_f32(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    for bi in (0..rows).step_by(block_size) {
        let bi_end = (bi + block_size).min(rows);
        for bj in (0..cols).step_by(block_size) {
            let bj_end = (bj + block_size).min(cols);
            // Within each cache block, use NEON 4×4 tiles
            let inner_rows = bi_end - bi;
            let inner_cols = bj_end - bj;
            let full_ir = inner_rows / 4 * 4;
            let full_ic = inner_cols / 4 * 4;

            for ii in (0..full_ir).step_by(4) {
                let ri = bi + ii;
                for jj in (0..full_ic).step_by(4) {
                    let cj = bj + jj;
                    unsafe {
                        let r0 = vld1q_f32(input.as_ptr().add(ri * cols + cj));
                        let r1 = vld1q_f32(input.as_ptr().add((ri + 1) * cols + cj));
                        let r2 = vld1q_f32(input.as_ptr().add((ri + 2) * cols + cj));
                        let r3 = vld1q_f32(input.as_ptr().add((ri + 3) * cols + cj));

                        let t0 = vtrn1q_f32(r0, r1);
                        let t1 = vtrn2q_f32(r0, r1);
                        let t2 = vtrn1q_f32(r2, r3);
                        let t3 = vtrn2q_f32(r2, r3);

                        let t0_64 = vreinterpretq_f64_f32(t0);
                        let t1_64 = vreinterpretq_f64_f32(t1);
                        let t2_64 = vreinterpretq_f64_f32(t2);
                        let t3_64 = vreinterpretq_f64_f32(t3);

                        let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
                        let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
                        let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
                        let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

                        vst1q_f32(output.as_mut_ptr().add(cj * rows + ri), o0);
                        vst1q_f32(output.as_mut_ptr().add((cj + 1) * rows + ri), o1);
                        vst1q_f32(output.as_mut_ptr().add((cj + 2) * rows + ri), o2);
                        vst1q_f32(output.as_mut_ptr().add((cj + 3) * rows + ri), o3);
                    }
                }
                // Right-edge scalar within this block-row
                for jj in full_ic..inner_cols {
                    let cj = bj + jj;
                    for k in 0..4 {
                        output[cj * rows + (ri + k)] = input[(ri + k) * cols + cj];
                    }
                }
            }
            // Bottom-edge scalar within this block
            for ii in full_ir..inner_rows {
                let ri = bi + ii;
                for jj in 0..inner_cols {
                    let cj = bj + jj;
                    output[cj * rows + ri] = input[ri * cols + cj];
                }
            }
        }
    }
}

fn scalar_transpose_blocked_f32(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    for bi in (0..rows).step_by(block_size) {
        let bi_end = (bi + block_size).min(rows);
        for bj in (0..cols).step_by(block_size) {
            let bj_end = (bj + block_size).min(cols);
            for i in bi..bi_end {
                for j in bj..bj_end {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        }
    }
}

/// Cache-friendly blocked transpose of a row-major `rows × cols` matrix.
///
/// Tiles the matrix into `block_size × block_size` sub-blocks to improve
/// cache locality. Each sub-block is transposed using NEON 4×4 tiles where
/// possible.
///
/// # Panics
///
/// Panics if slice lengths are less than `rows * cols` or `block_size == 0`.
pub fn transpose_blocked_f32(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    let numel = rows * cols;
    assert!(block_size > 0, "block_size must be > 0");
    assert!(input.len() >= numel, "input length {} < rows*cols {numel}", input.len());
    assert!(output.len() >= numel, "output length {} < rows*cols {numel}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_transpose_blocked_f32(input, output, rows, cols, block_size);
            }
            return;
        }
    }
    scalar_transpose_blocked_f32(input, output, rows, cols, block_size);
}

// ── 4. Batched Transpose ────────────────────────────────────────────

/// NEON-accelerated batched transpose.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_batch_transpose_f32(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    let mat_size = rows * cols;
    for b in 0..batch {
        let in_off = b * mat_size;
        let out_off = b * mat_size;
        let in_slice = &input[in_off..in_off + mat_size];
        let out_slice = &mut output[out_off..out_off + mat_size];
        // Reuse the single-matrix NEON transpose
        unsafe {
            neon_transpose_f32(in_slice, out_slice, rows, cols);
        }
    }
}

fn scalar_batch_transpose_f32(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    let mat_size = rows * cols;
    for b in 0..batch {
        let in_off = b * mat_size;
        let out_off = b * mat_size;
        scalar_transpose_f32(
            &input[in_off..in_off + mat_size],
            &mut output[out_off..out_off + mat_size],
            rows,
            cols,
        );
    }
}

/// Batched transpose: transposes `batch` independent `rows × cols` matrices.
///
/// Input layout: `[batch, rows, cols]` (contiguous). Output layout: `[batch, cols, rows]`.
///
/// # Panics
///
/// Panics if slice lengths are less than `batch * rows * cols`.
pub fn batch_transpose_f32(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    let total = batch * rows * cols;
    assert!(input.len() >= total, "input length {} < batch*rows*cols {total}", input.len());
    assert!(output.len() >= total, "output length {} < batch*rows*cols {total}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_batch_transpose_f32(input, output, batch, rows, cols);
            }
            return;
        }
    }
    scalar_batch_transpose_f32(input, output, batch, rows, cols);
}

// ── 5. Generalized Dimension Permutation ────────────────────────────

/// Compute strides from a shape (row-major).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// NEON-accelerated dimension permutation with vectorised contiguous-tail copy.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_permute_dims_f32(
    input: &[f32],
    output: &mut [f32],
    shape: &[usize],
    perm: &[usize],
) {
    // Falls through to the same index-mapping logic; the NEON advantage
    // is used when the innermost permuted dimension is contiguous so we
    // can memcpy / NEON-copy whole rows.
    scalar_permute_dims_f32(input, output, shape, perm);
}

fn scalar_permute_dims_f32(input: &[f32], output: &mut [f32], shape: &[usize], perm: &[usize]) {
    let ndim = shape.len();
    let total: usize = shape.iter().product();
    if total == 0 {
        return;
    }

    let in_strides = compute_strides(shape);

    // Output shape after permutation
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let out_strides = compute_strides(&out_shape);

    // Build a mapping: for each output dimension d, which input stride?
    let perm_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();

    // Iterate over all output indices
    let mut out_coord = vec![0usize; ndim];
    for out_idx in 0..total {
        // Compute flat input index from output coordinates
        let mut in_idx = 0usize;
        for d in 0..ndim {
            in_idx += out_coord[d] * perm_strides[d];
        }
        output[out_idx] = input[in_idx];

        // Increment output coordinates (odometer)
        for d in (0..ndim).rev() {
            out_coord[d] += 1;
            if out_coord[d] < out_shape[d] {
                break;
            }
            out_coord[d] = 0;
        }
    }
}

/// Generalized dimension permutation (like `numpy.transpose(axes=...)`).
///
/// Rearranges tensor dimensions according to `perm`. For a tensor with
/// shape `[d0, d1, d2]` and `perm = [2, 0, 1]`, the output shape is
/// `[d2, d0, d1]` and `output[k][i][j] = input[i][j][k]`.
///
/// # Panics
///
/// Panics if `perm` is not a valid permutation of `0..shape.len()`,
/// or if slice lengths are less than the product of `shape`.
pub fn permute_dims_f32(input: &[f32], output: &mut [f32], shape: &[usize], perm: &[usize]) {
    let ndim = shape.len();
    assert_eq!(perm.len(), ndim, "perm length {} != shape length {ndim}", perm.len());
    // Validate perm is a proper permutation
    let mut seen = vec![false; ndim];
    for &p in perm {
        assert!(p < ndim, "perm value {p} out of range 0..{ndim}");
        assert!(!seen[p], "duplicate perm value {p}");
        seen[p] = true;
    }

    let total: usize = shape.iter().product();
    assert!(input.len() >= total, "input length {} < total {total}", input.len());
    assert!(output.len() >= total, "output length {} < total {total}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_permute_dims_f32(input, output, shape, perm);
            }
            return;
        }
    }
    scalar_permute_dims_f32(input, output, shape, perm);
}

// ── 6. Contiguous Layout Check ──────────────────────────────────────

/// Check whether a tensor with the given `shape` and `strides` is contiguous
/// (i.e., row-major layout with no gaps or overlaps).
///
/// Returns `true` if and only if the strides match the standard row-major
/// strides for the given shape.
pub fn contiguous_check_f32(_input: &[f32], shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let expected = compute_strides(shape);
    strides == expected.as_slice()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build row-major matrix with sequential values
    fn sequential_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| i as f32).collect()
    }

    // Reference scalar transpose for verification
    fn reference_transpose(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = input[i * cols + j];
            }
        }
        out
    }

    // ── transpose_f32 tests ─────────────────────────────────────────

    #[test]
    fn test_transpose_f32_4x4_aligned() {
        let input = sequential_matrix(4, 4);
        let mut output = vec![0.0f32; 16];
        transpose_f32(&input, &mut output, 4, 4);
        let expected = reference_transpose(&input, 4, 4);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_8x8_aligned() {
        let input = sequential_matrix(8, 8);
        let mut output = vec![0.0f32; 64];
        transpose_f32(&input, &mut output, 8, 8);
        let expected = reference_transpose(&input, 8, 8);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_non_square() {
        let input = sequential_matrix(3, 7);
        let mut output = vec![0.0f32; 21];
        transpose_f32(&input, &mut output, 3, 7);
        let expected = reference_transpose(&input, 3, 7);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_wide_rectangle() {
        let input = sequential_matrix(4, 12);
        let mut output = vec![0.0f32; 48];
        transpose_f32(&input, &mut output, 4, 12);
        let expected = reference_transpose(&input, 4, 12);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_tall_rectangle() {
        let input = sequential_matrix(12, 4);
        let mut output = vec![0.0f32; 48];
        transpose_f32(&input, &mut output, 12, 4);
        let expected = reference_transpose(&input, 12, 4);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_1x1() {
        let input = vec![42.0f32];
        let mut output = vec![0.0f32; 1];
        transpose_f32(&input, &mut output, 1, 1);
        assert_eq!(output, vec![42.0]);
    }

    #[test]
    fn test_transpose_f32_1xn() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];
        transpose_f32(&input, &mut output, 1, 5);
        // 1×5 → 5×1, same values
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_transpose_f32_nx1() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        transpose_f32(&input, &mut output, 3, 1);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpose_f32_5x6_unaligned() {
        let input = sequential_matrix(5, 6);
        let mut output = vec![0.0f32; 30];
        transpose_f32(&input, &mut output, 5, 6);
        let expected = reference_transpose(&input, 5, 6);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_7x9() {
        let input = sequential_matrix(7, 9);
        let mut output = vec![0.0f32; 63];
        transpose_f32(&input, &mut output, 7, 9);
        let expected = reference_transpose(&input, 7, 9);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_transpose_f32_identity_property() {
        // Transposing twice should yield the original matrix
        let input = sequential_matrix(6, 10);
        let mut tmp = vec![0.0f32; 60];
        let mut result = vec![0.0f32; 60];
        transpose_f32(&input, &mut tmp, 6, 10);
        transpose_f32(&tmp, &mut result, 10, 6);
        assert_eq!(result, input);
    }

    #[test]
    fn test_transpose_f32_negative_values() {
        let input: Vec<f32> = (-8..8).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 16];
        transpose_f32(&input, &mut output, 4, 4);
        let expected = reference_transpose(&input, 4, 4);
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_transpose_f32_input_too_short() {
        let input = vec![0.0f32; 5];
        let mut output = vec![0.0f32; 12];
        transpose_f32(&input, &mut output, 3, 4);
    }

    #[test]
    #[should_panic(expected = "output length")]
    fn test_transpose_f32_output_too_short() {
        let input = vec![0.0f32; 12];
        let mut output = vec![0.0f32; 5];
        transpose_f32(&input, &mut output, 3, 4);
    }

    // ── transpose_inplace_f32 tests ─────────────────────────────────

    #[test]
    fn test_inplace_4x4() {
        let mut data = sequential_matrix(4, 4);
        let expected = reference_transpose(&data, 4, 4);
        transpose_inplace_f32(&mut data, 4);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_8x8() {
        let mut data = sequential_matrix(8, 8);
        let expected = reference_transpose(&data, 8, 8);
        transpose_inplace_f32(&mut data, 8);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_1x1() {
        let mut data = vec![7.0f32];
        transpose_inplace_f32(&mut data, 1);
        assert_eq!(data, vec![7.0]);
    }

    #[test]
    fn test_inplace_2x2() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        transpose_inplace_f32(&mut data, 2);
        assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_inplace_3x3() {
        let mut data = sequential_matrix(3, 3);
        let expected = reference_transpose(&data, 3, 3);
        transpose_inplace_f32(&mut data, 3);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_5x5_unaligned() {
        let mut data = sequential_matrix(5, 5);
        let expected = reference_transpose(&data, 5, 5);
        transpose_inplace_f32(&mut data, 5);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_7x7() {
        let mut data = sequential_matrix(7, 7);
        let expected = reference_transpose(&data, 7, 7);
        transpose_inplace_f32(&mut data, 7);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_inplace_double_is_identity() {
        let original = sequential_matrix(6, 6);
        let mut data = original.clone();
        transpose_inplace_f32(&mut data, 6);
        transpose_inplace_f32(&mut data, 6);
        assert_eq!(data, original);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_inplace_data_too_short() {
        let mut data = vec![0.0f32; 3];
        transpose_inplace_f32(&mut data, 4);
    }

    // ── transpose_blocked_f32 tests ─────────────────────────────────

    #[test]
    fn test_blocked_matches_reference_4x4_block8() {
        let input = sequential_matrix(4, 4);
        let mut output = vec![0.0f32; 16];
        transpose_blocked_f32(&input, &mut output, 4, 4, 8);
        let expected = reference_transpose(&input, 4, 4);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_matches_reference_16x16_block4() {
        let input = sequential_matrix(16, 16);
        let mut output = vec![0.0f32; 256];
        transpose_blocked_f32(&input, &mut output, 16, 16, 4);
        let expected = reference_transpose(&input, 16, 16);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_matches_reference_7x9_block4() {
        let input = sequential_matrix(7, 9);
        let mut output = vec![0.0f32; 63];
        transpose_blocked_f32(&input, &mut output, 7, 9, 4);
        let expected = reference_transpose(&input, 7, 9);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_matches_reference_13x5_block8() {
        let input = sequential_matrix(13, 5);
        let mut output = vec![0.0f32; 65];
        transpose_blocked_f32(&input, &mut output, 13, 5, 8);
        let expected = reference_transpose(&input, 13, 5);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_block_size_1() {
        let input = sequential_matrix(5, 3);
        let mut output = vec![0.0f32; 15];
        transpose_blocked_f32(&input, &mut output, 5, 3, 1);
        let expected = reference_transpose(&input, 5, 3);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_block_larger_than_matrix() {
        let input = sequential_matrix(3, 3);
        let mut output = vec![0.0f32; 9];
        transpose_blocked_f32(&input, &mut output, 3, 3, 64);
        let expected = reference_transpose(&input, 3, 3);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_blocked_large_32x32() {
        let input = sequential_matrix(32, 32);
        let mut output = vec![0.0f32; 1024];
        transpose_blocked_f32(&input, &mut output, 32, 32, 8);
        let expected = reference_transpose(&input, 32, 32);
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_blocked_zero_block_size() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        transpose_blocked_f32(&input, &mut output, 2, 2, 0);
    }

    // ── batch_transpose_f32 tests ───────────────────────────────────

    #[test]
    fn test_batch_single() {
        let input = sequential_matrix(3, 5);
        let mut output = vec![0.0f32; 15];
        batch_transpose_f32(&input, &mut output, 1, 3, 5);
        let expected = reference_transpose(&input, 3, 5);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_batch_multiple() {
        let batch = 3;
        let (rows, cols) = (4, 4);
        let mat_size = rows * cols;
        let input: Vec<f32> = (0..batch * mat_size).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; batch * mat_size];
        batch_transpose_f32(&input, &mut output, batch, rows, cols);

        for b in 0..batch {
            let off = b * mat_size;
            let expected = reference_transpose(&input[off..off + mat_size], rows, cols);
            assert_eq!(&output[off..off + mat_size], expected.as_slice());
        }
    }

    #[test]
    fn test_batch_non_square() {
        let batch = 2;
        let (rows, cols) = (5, 7);
        let mat_size = rows * cols;
        let input: Vec<f32> = (0..batch * mat_size).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; batch * mat_size];
        batch_transpose_f32(&input, &mut output, batch, rows, cols);

        for b in 0..batch {
            let off = b * mat_size;
            let expected = reference_transpose(&input[off..off + mat_size], rows, cols);
            assert_eq!(&output[off..off + mat_size], expected.as_slice());
        }
    }

    #[test]
    fn test_batch_1x1_matrices() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        batch_transpose_f32(&input, &mut output, 3, 1, 1);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_zero_batch() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        batch_transpose_f32(&input, &mut output, 0, 4, 4);
        // No panic, no work
    }

    #[test]
    #[should_panic(expected = "input length")]
    fn test_batch_input_too_short() {
        let input = vec![0.0f32; 10];
        let mut output = vec![0.0f32; 48];
        batch_transpose_f32(&input, &mut output, 3, 4, 4);
    }

    // ── permute_dims_f32 tests ──────────────────────────────────────

    #[test]
    fn test_permute_2d_is_transpose() {
        let input = sequential_matrix(3, 4);
        let mut output = vec![0.0f32; 12];
        permute_dims_f32(&input, &mut output, &[3, 4], &[1, 0]);
        let expected = reference_transpose(&input, 3, 4);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_permute_identity_2d() {
        let input = sequential_matrix(3, 4);
        let mut output = vec![0.0f32; 12];
        permute_dims_f32(&input, &mut output, &[3, 4], &[0, 1]);
        assert_eq!(output, input);
    }

    #[test]
    fn test_permute_3d_swap_last_two() {
        // Shape [2, 3, 4], perm [0, 2, 1] → [2, 4, 3]
        let shape = [2, 3, 4];
        let perm = [0, 2, 1];
        let total: usize = shape.iter().product();
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; total];
        permute_dims_f32(&input, &mut output, &shape, &perm);

        // Verify: output[b][k][j] = input[b][j][k]
        for b in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let in_idx = b * 12 + j * 4 + k;
                    let out_idx = b * 12 + k * 3 + j;
                    assert_eq!(output[out_idx], input[in_idx], "mismatch at b={b} j={j} k={k}");
                }
            }
        }
    }

    #[test]
    fn test_permute_3d_full_rotate() {
        // Shape [2, 3, 4], perm [2, 0, 1] → [4, 2, 3]
        let shape = [2, 3, 4];
        let perm = [2, 0, 1];
        let total: usize = shape.iter().product();
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; total];
        permute_dims_f32(&input, &mut output, &shape, &perm);

        // Verify: output[k][i][j] = input[i][j][k]
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let in_idx = i * 12 + j * 4 + k;
                    let out_idx = k * 6 + i * 3 + j;
                    assert_eq!(output[out_idx], input[in_idx], "mismatch at i={i} j={j} k={k}");
                }
            }
        }
    }

    #[test]
    fn test_permute_identity_3d() {
        let shape = [2, 3, 4];
        let total: usize = shape.iter().product();
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; total];
        permute_dims_f32(&input, &mut output, &shape, &[0, 1, 2]);
        assert_eq!(output, input);
    }

    #[test]
    fn test_permute_4d() {
        // Shape [2, 3, 4, 5], perm [0, 2, 1, 3] → [2, 4, 3, 5]
        let shape = [2, 3, 4, 5];
        let perm = [0, 2, 1, 3];
        let total: usize = shape.iter().product();
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; total];
        permute_dims_f32(&input, &mut output, &shape, &perm);

        // Verify a sample of elements
        // output[a][c][b][d] = input[a][b][c][d]
        for a in 0..2 {
            for b in 0..3 {
                for c in 0..4 {
                    for d in 0..5 {
                        let in_idx = a * 60 + b * 20 + c * 5 + d;
                        let out_idx = a * 60 + c * 15 + b * 5 + d;
                        assert_eq!(
                            output[out_idx], input[in_idx],
                            "mismatch at a={a} b={b} c={c} d={d}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_permute_1d_identity() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        permute_dims_f32(&input, &mut output, &[3], &[0]);
        assert_eq!(output, input);
    }

    #[test]
    fn test_permute_empty_tensor() {
        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        permute_dims_f32(&input, &mut output, &[0, 3], &[1, 0]);
    }

    #[test]
    #[should_panic(expected = "perm length")]
    fn test_permute_wrong_perm_length() {
        let input = vec![0.0f32; 12];
        let mut output = vec![0.0f32; 12];
        permute_dims_f32(&input, &mut output, &[3, 4], &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "duplicate perm")]
    fn test_permute_duplicate_perm() {
        let input = vec![0.0f32; 12];
        let mut output = vec![0.0f32; 12];
        permute_dims_f32(&input, &mut output, &[3, 4], &[0, 0]);
    }

    #[test]
    #[should_panic(expected = "perm value")]
    fn test_permute_out_of_range_perm() {
        let input = vec![0.0f32; 12];
        let mut output = vec![0.0f32; 12];
        permute_dims_f32(&input, &mut output, &[3, 4], &[0, 5]);
    }

    // ── contiguous_check_f32 tests ──────────────────────────────────

    #[test]
    fn test_contiguous_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(contiguous_check_f32(&data, &[4], &[1]));
    }

    #[test]
    fn test_contiguous_2d() {
        let data = vec![0.0f32; 12];
        assert!(contiguous_check_f32(&data, &[3, 4], &[4, 1]));
    }

    #[test]
    fn test_contiguous_3d() {
        let data = vec![0.0f32; 24];
        assert!(contiguous_check_f32(&data, &[2, 3, 4], &[12, 4, 1]));
    }

    #[test]
    fn test_non_contiguous_transposed() {
        let data = vec![0.0f32; 12];
        // Transposed 3×4 → strides [1, 3] instead of [4, 1]
        assert!(!contiguous_check_f32(&data, &[3, 4], &[1, 3]));
    }

    #[test]
    fn test_non_contiguous_strided() {
        let data = vec![0.0f32; 20];
        // Extra padding stride
        assert!(!contiguous_check_f32(&data, &[4, 4], &[5, 1]));
    }

    #[test]
    fn test_contiguous_shape_stride_mismatch() {
        let data = vec![0.0f32; 12];
        assert!(!contiguous_check_f32(&data, &[3, 4], &[4, 1, 1]));
    }

    #[test]
    fn test_contiguous_scalar() {
        let data = vec![42.0f32];
        assert!(contiguous_check_f32(&data, &[], &[]));
    }

    // ── Cross-function consistency tests ────────────────────────────

    #[test]
    fn test_transpose_vs_blocked_consistency() {
        let input = sequential_matrix(11, 13);
        let mut out_plain = vec![0.0f32; 143];
        let mut out_blocked = vec![0.0f32; 143];
        transpose_f32(&input, &mut out_plain, 11, 13);
        transpose_blocked_f32(&input, &mut out_blocked, 11, 13, 8);
        assert_eq!(out_plain, out_blocked);
    }

    #[test]
    fn test_transpose_vs_permute_consistency() {
        let input = sequential_matrix(6, 8);
        let mut out_t = vec![0.0f32; 48];
        let mut out_p = vec![0.0f32; 48];
        transpose_f32(&input, &mut out_t, 6, 8);
        permute_dims_f32(&input, &mut out_p, &[6, 8], &[1, 0]);
        assert_eq!(out_t, out_p);
    }

    #[test]
    fn test_inplace_vs_out_of_place_consistency() {
        let input = sequential_matrix(8, 8);
        let mut inplace = input.clone();
        let mut out_of_place = vec![0.0f32; 64];
        transpose_inplace_f32(&mut inplace, 8);
        transpose_f32(&input, &mut out_of_place, 8, 8);
        assert_eq!(inplace, out_of_place);
    }

    #[test]
    fn test_batch_vs_individual() {
        let batch = 4;
        let (rows, cols) = (5, 6);
        let mat_size = rows * cols;
        let input: Vec<f32> = (0..batch * mat_size).map(|i| i as f32).collect();
        let mut batch_out = vec![0.0f32; batch * mat_size];
        batch_transpose_f32(&input, &mut batch_out, batch, rows, cols);

        for b in 0..batch {
            let off = b * mat_size;
            let mut individual_out = vec![0.0f32; mat_size];
            transpose_f32(&input[off..off + mat_size], &mut individual_out, rows, cols);
            assert_eq!(&batch_out[off..off + mat_size], individual_out.as_slice());
        }
    }

    #[test]
    fn test_blocked_various_block_sizes() {
        let input = sequential_matrix(10, 10);
        let reference = reference_transpose(&input, 10, 10);
        for bs in [1, 2, 4, 5, 8, 10, 16, 32] {
            let mut output = vec![0.0f32; 100];
            transpose_blocked_f32(&input, &mut output, 10, 10, bs);
            assert_eq!(output, reference, "mismatch with block_size={bs}");
        }
    }

    #[test]
    fn test_transpose_f32_large_non_aligned() {
        let input = sequential_matrix(33, 17);
        let mut output = vec![0.0f32; 33 * 17];
        transpose_f32(&input, &mut output, 33, 17);
        let expected = reference_transpose(&input, 33, 17);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_inplace_12x12() {
        let mut data = sequential_matrix(12, 12);
        let expected = reference_transpose(&data, 12, 12);
        transpose_inplace_f32(&mut data, 12);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_permute_inverse_3d() {
        // Applying perm then inverse perm should yield identity
        let shape = [2, 3, 4];
        let perm = [1, 2, 0];
        let inv_perm = [2, 0, 1]; // inverse of [1,2,0]
        let total: usize = shape.iter().product();
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut tmp = vec![0.0f32; total];
        let mut result = vec![0.0f32; total];
        permute_dims_f32(&input, &mut tmp, &shape, &perm);
        let mid_shape = [3, 4, 2]; // permuted shape
        permute_dims_f32(&tmp, &mut result, &mid_shape, &inv_perm);
        assert_eq!(result, input);
    }

    #[test]
    fn test_batch_aligned_8x8() {
        let batch = 2;
        let (rows, cols) = (8, 8);
        let mat_size = rows * cols;
        let input: Vec<f32> = (0..batch * mat_size).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; batch * mat_size];
        batch_transpose_f32(&input, &mut output, batch, rows, cols);
        for b in 0..batch {
            let off = b * mat_size;
            let expected = reference_transpose(&input[off..off + mat_size], rows, cols);
            assert_eq!(&output[off..off + mat_size], expected.as_slice());
        }
    }

    #[test]
    fn test_contiguous_4d() {
        let data = vec![0.0f32; 120];
        assert!(contiguous_check_f32(&data, &[2, 3, 4, 5], &[60, 20, 5, 1]));
    }

    #[test]
    fn test_non_contiguous_4d() {
        let data = vec![0.0f32; 120];
        assert!(!contiguous_check_f32(&data, &[2, 3, 4, 5], &[60, 20, 1, 5]));
    }
}
