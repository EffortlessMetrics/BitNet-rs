//! ARM NEON-optimized memory layout operations for Apple Silicon.
//!
//! Provides transpose, interleave/deinterleave, alignment padding, and
//! tiled-access helpers that exploit `float32x4_t` NEON intrinsics for
//! SIMD-friendly data organisation.
//!
//! All hot paths use NEON vector loads/stores with scalar fallback for
//! remainder elements that do not fill a full 4-wide lane.

use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

/// Byte alignment required by NEON `float32x4_t` loads/stores (16 bytes).
#[cfg(target_arch = "aarch64")]
const NEON_ALIGN_BYTES: usize = 16;

/// Number of `f32` values that fit in one NEON-aligned chunk.
#[cfg(target_arch = "aarch64")]
const NEON_ALIGN_F32: usize = NEON_ALIGN_BYTES / size_of::<f32>();

// ── 4×4 Block Transpose ────────────────────────────────────────────────

/// Transpose a row-major 4×4 `f32` matrix in-place using NEON `vtrn`
/// and 64-bit reinterpret shuffles.
///
/// Both `input` and `output` must have length ≥ 16. The first 16 elements
/// are treated as a row-major 4×4 matrix.
///
/// # Panics
///
/// Panics if either slice is shorter than 16 elements.
pub fn transpose_4x4_neon(input: &[f32], output: &mut [f32]) {
    assert!(input.len() >= 16, "input must have >= 16 elements, got {}", input.len());
    assert!(output.len() >= 16, "output must have >= 16 elements, got {}", output.len());

    unsafe {
        // Load four rows.
        let r0 = vld1q_f32(input.as_ptr());
        let r1 = vld1q_f32(input.as_ptr().add(4));
        let r2 = vld1q_f32(input.as_ptr().add(8));
        let r3 = vld1q_f32(input.as_ptr().add(12));

        // Stage 1 – element-level transpose within pairs.
        let t0 = vtrn1q_f32(r0, r1);
        let t1 = vtrn2q_f32(r0, r1);
        let t2 = vtrn1q_f32(r2, r3);
        let t3 = vtrn2q_f32(r2, r3);

        // Stage 2 – 64-bit half-swap via f64 reinterpret.
        let t0_64 = vreinterpretq_f64_f32(t0);
        let t1_64 = vreinterpretq_f64_f32(t1);
        let t2_64 = vreinterpretq_f64_f32(t2);
        let t3_64 = vreinterpretq_f64_f32(t3);

        let o0 = vreinterpretq_f32_f64(vtrn1q_f64(t0_64, t2_64));
        let o1 = vreinterpretq_f32_f64(vtrn1q_f64(t1_64, t3_64));
        let o2 = vreinterpretq_f32_f64(vtrn2q_f64(t0_64, t2_64));
        let o3 = vreinterpretq_f32_f64(vtrn2q_f64(t1_64, t3_64));

        // Store four transposed rows.
        vst1q_f32(output.as_mut_ptr(), o0);
        vst1q_f32(output.as_mut_ptr().add(4), o1);
        vst1q_f32(output.as_mut_ptr().add(8), o2);
        vst1q_f32(output.as_mut_ptr().add(12), o3);
    }
}

// ── Arbitrary 2-D Transpose ────────────────────────────────────────────

/// Transpose a row-major `rows × cols` matrix using NEON 4×4 tiling with
/// scalar fallback for remainder rows/columns.
///
/// # Panics
///
/// Panics if `input.len() < rows * cols` or `output.len() < rows * cols`.
pub fn transpose_2d_neon(input: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    let numel = rows * cols;
    assert!(input.len() >= numel, "input too short: {} < {numel}", input.len());
    assert!(output.len() >= numel, "output too short: {} < {numel}", output.len());

    let row_blocks = rows / LANES;
    let col_blocks = cols / LANES;

    // Full 4×4 NEON blocks.
    for bi in 0..row_blocks {
        for bj in 0..col_blocks {
            let ri = bi * LANES;
            let cj = bj * LANES;
            transpose_4x4_block(input, cols, ri, cj, output, rows);
        }
    }

    // Remainder columns (right edge).
    let col_tail = col_blocks * LANES;
    for r in 0..rows {
        for c in col_tail..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }

    // Remainder rows (bottom edge, only full-block columns).
    let row_tail = row_blocks * LANES;
    for r in row_tail..rows {
        for c in 0..col_tail {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

/// Transpose a single 4×4 block from `(ri, cj)` in a `rows × cols` matrix.
#[inline(always)]
fn transpose_4x4_block(
    input: &[f32],
    cols: usize,
    ri: usize,
    cj: usize,
    output: &mut [f32],
    rows: usize,
) {
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

// ── Interleave / Deinterleave ──────────────────────────────────────────

/// Interleave two `f32` slices (AOS-style): `[a0, b0, a1, b1, …]`.
///
/// `output` must have length ≥ `2 * count` where `count = min(a.len(), b.len())`.
///
/// # Panics
///
/// Panics if `output` is too short.
pub fn interleave_neon(a: &[f32], b: &[f32], output: &mut [f32]) {
    let count = a.len().min(b.len());
    assert!(output.len() >= count * 2, "output too short: {} < {}", output.len(), count * 2);

    let full = count / LANES;
    for i in 0..full {
        let off = i * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let lo = vzip1q_f32(va, vb);
            let hi = vzip2q_f32(va, vb);
            vst1q_f32(output.as_mut_ptr().add(off * 2), lo);
            vst1q_f32(output.as_mut_ptr().add(off * 2 + LANES), hi);
        }
    }

    // Scalar tail.
    for i in (full * LANES)..count {
        output[i * 2] = a[i];
        output[i * 2 + 1] = b[i];
    }
}

/// Deinterleave an AOS `f32` stream into two separate slices (SOA).
///
/// `input` has pairs `[a0, b0, a1, b1, …]`; `a` and `b` each receive
/// `count` values where `count = input.len() / 2`.
///
/// # Panics
///
/// Panics if output slices are too short.
pub fn deinterleave_neon(input: &[f32], a: &mut [f32], b: &mut [f32]) {
    let count = input.len() / 2;
    assert!(a.len() >= count, "a too short: {} < {count}", a.len());
    assert!(b.len() >= count, "b too short: {} < {count}", b.len());

    let full = count / LANES;
    for i in 0..full {
        let off = i * LANES;
        unsafe {
            let lo = vld1q_f32(input.as_ptr().add(off * 2));
            let hi = vld1q_f32(input.as_ptr().add(off * 2 + LANES));
            let va = vuzp1q_f32(lo, hi);
            let vb = vuzp2q_f32(lo, hi);
            vst1q_f32(a.as_mut_ptr().add(off), va);
            vst1q_f32(b.as_mut_ptr().add(off), vb);
        }
    }

    // Scalar tail.
    for i in (full * LANES)..count {
        a[i] = input[i * 2];
        b[i] = input[i * 2 + 1];
    }
}

// ── Multi-channel interleave/deinterleave ──────────────────────────────

/// Interleave `channels` separate slices into a single AOS stream.
///
/// Each input slice must have at least `count` elements. The output
/// receives `count * channels` values laid out as
/// `[ch0[0], ch1[0], …, ch0[1], ch1[1], …]`.
///
/// NEON is used for the 2-channel special case; the general path is scalar.
///
/// # Panics
///
/// Panics if `inputs.len() != channels` or any slice is too short.
pub fn interleave_multi_neon(inputs: &[&[f32]], channels: usize, count: usize, output: &mut [f32]) {
    assert_eq!(inputs.len(), channels, "inputs.len() must equal channels");
    assert!(
        output.len() >= count * channels,
        "output too short: {} < {}",
        output.len(),
        count * channels,
    );
    for inp in inputs {
        assert!(inp.len() >= count, "input slice too short");
    }

    if channels == 2 {
        interleave_neon(inputs[0], inputs[1], output);
        return;
    }

    for i in 0..count {
        for ch in 0..channels {
            output[i * channels + ch] = inputs[ch][i];
        }
    }
}

/// Deinterleave a multi-channel AOS stream into separate per-channel slices.
///
/// NEON is used for the 2-channel special case; the general path is scalar.
///
/// # Panics
///
/// Panics if `outputs.len() != channels` or any slice is too short.
pub fn deinterleave_multi_neon(
    input: &[f32],
    channels: usize,
    count: usize,
    outputs: &mut [&mut [f32]],
) {
    assert_eq!(outputs.len(), channels, "outputs.len() must equal channels");
    assert!(
        input.len() >= count * channels,
        "input too short: {} < {}",
        input.len(),
        count * channels,
    );
    for out in outputs.iter() {
        assert!(out.len() >= count, "output slice too short");
    }

    if channels == 2 {
        let (first, rest) = outputs.split_at_mut(1);
        deinterleave_neon(input, first[0], rest[0]);
        return;
    }

    for i in 0..count {
        for ch in 0..channels {
            outputs[ch][i] = input[i * channels + ch];
        }
    }
}

// ── Alignment / Padding ────────────────────────────────────────────────

/// Pad `data` to the next NEON-aligned `f32` count (multiple of 4).
///
/// Returns a new `Vec<f32>` whose length is a multiple of
/// [`NEON_ALIGN_F32`] (4). Trailing elements are filled with `pad_value`.
pub fn pad_to_neon_alignment(data: &[f32], pad_value: f32) -> Vec<f32> {
    let aligned_len = data.len().div_ceil(NEON_ALIGN_F32) * NEON_ALIGN_F32;
    let mut out = Vec::with_capacity(aligned_len);
    out.extend_from_slice(data);
    out.resize(aligned_len, pad_value);
    out
}

/// Pad a 2-D row-major matrix so that both dimensions are multiples of 4.
///
/// Returns `(padded_data, new_rows, new_cols)`.
pub fn pad_matrix_neon(
    data: &[f32],
    rows: usize,
    cols: usize,
    pad_value: f32,
) -> (Vec<f32>, usize, usize) {
    assert!(data.len() >= rows * cols, "data too short");
    let new_rows = rows.div_ceil(NEON_ALIGN_F32) * NEON_ALIGN_F32;
    let new_cols = cols.div_ceil(NEON_ALIGN_F32) * NEON_ALIGN_F32;
    let mut out = vec![pad_value; new_rows * new_cols];
    for r in 0..rows {
        out[r * new_cols..r * new_cols + cols].copy_from_slice(&data[r * cols..r * cols + cols]);
    }
    (out, new_rows, new_cols)
}

// ── Tiled Memory Access ────────────────────────────────────────────────

/// Reorganise a row-major `rows × cols` matrix into `tile_rows × tile_cols`
/// tiles laid out contiguously in memory.
///
/// Tiles are emitted in row-major tile order (left-to-right, top-to-bottom).
/// Partial tiles at the right/bottom edge are zero-padded.
///
/// Returns `(tiled_data, n_tile_rows, n_tile_cols)`.
pub fn tile_data_neon(
    data: &[f32],
    rows: usize,
    cols: usize,
    tile_rows: usize,
    tile_cols: usize,
) -> (Vec<f32>, usize, usize) {
    assert!(tile_rows > 0 && tile_cols > 0, "tile dimensions must be > 0");
    assert!(data.len() >= rows * cols, "data too short");

    let n_tr = rows.div_ceil(tile_rows);
    let n_tc = cols.div_ceil(tile_cols);
    let tile_size = tile_rows * tile_cols;
    let mut out = vec![0.0f32; n_tr * n_tc * tile_size];

    // NEON-accelerated copy for tiles whose columns align to LANES.
    let use_neon = tile_cols >= LANES && tile_cols.is_multiple_of(LANES);

    for tr in 0..n_tr {
        for tc in 0..n_tc {
            let tile_idx = tr * n_tc + tc;
            let base_out = tile_idx * tile_size;
            let src_row_start = tr * tile_rows;
            let src_col_start = tc * tile_cols;

            for lr in 0..tile_rows {
                let sr = src_row_start + lr;
                if sr >= rows {
                    break;
                }
                let dst_off = base_out + lr * tile_cols;
                let valid_cols = (cols - src_col_start).min(tile_cols);

                if use_neon && valid_cols == tile_cols {
                    neon_copy_row(
                        &data[sr * cols + src_col_start..],
                        &mut out[dst_off..],
                        tile_cols,
                    );
                } else {
                    out[dst_off..dst_off + valid_cols].copy_from_slice(
                        &data[sr * cols + src_col_start..sr * cols + src_col_start + valid_cols],
                    );
                }
            }
        }
    }

    (out, n_tr, n_tc)
}

/// Reverse of [`tile_data_neon`]: reconstruct a row-major matrix from tiles.
///
/// `tiled` must have at least `n_tile_rows * n_tile_cols * tile_rows * tile_cols`
/// elements. The returned `Vec` has exactly `rows * cols` elements.
pub fn untile_data_neon(
    tiled: &[f32],
    rows: usize,
    cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    n_tile_rows: usize,
    n_tile_cols: usize,
) -> Vec<f32> {
    let tile_size = tile_rows * tile_cols;
    assert!(tiled.len() >= n_tile_rows * n_tile_cols * tile_size, "tiled data too short");

    let mut out = vec![0.0f32; rows * cols];
    let use_neon = tile_cols >= LANES && tile_cols.is_multiple_of(LANES);

    for tr in 0..n_tile_rows {
        for tc in 0..n_tile_cols {
            let tile_idx = tr * n_tile_cols + tc;
            let base_in = tile_idx * tile_size;
            let dst_row_start = tr * tile_rows;
            let dst_col_start = tc * tile_cols;

            for lr in 0..tile_rows {
                let dr = dst_row_start + lr;
                if dr >= rows {
                    break;
                }
                let src_off = base_in + lr * tile_cols;
                let valid_cols = (cols - dst_col_start).min(tile_cols);

                if use_neon && valid_cols == tile_cols {
                    neon_copy_row(
                        &tiled[src_off..],
                        &mut out[dr * cols + dst_col_start..],
                        tile_cols,
                    );
                } else {
                    out[dr * cols + dst_col_start..dr * cols + dst_col_start + valid_cols]
                        .copy_from_slice(&tiled[src_off..src_off + valid_cols]);
                }
            }
        }
    }

    out
}

// ── Cache-Line Aware Copy ──────────────────────────────────────────────

/// Copy `len` f32 values using NEON vector loads/stores.
///
/// Falls back to scalar for the tail elements that don't fill a full lane.
#[inline]
fn neon_copy_row(src: &[f32], dst: &mut [f32], len: usize) {
    let full = len / LANES;
    for i in 0..full {
        let off = i * LANES;
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(off));
            vst1q_f32(dst.as_mut_ptr().add(off), v);
        }
    }
    let tail = full * LANES;
    dst[tail..len].copy_from_slice(&src[tail..len]);
}

/// Copy `n` f32 values from `src` to `dst` using NEON, prefetching the
/// next cache line (64 bytes = 16 f32) ahead.
pub fn cache_aware_copy_neon(src: &[f32], dst: &mut [f32], n: usize) {
    assert!(src.len() >= n, "src too short");
    assert!(dst.len() >= n, "dst too short");

    let full = n / LANES;
    for i in 0..full {
        let off = i * LANES;
        // Prefetch 16 elements ahead (one cache line).
        if off + 16 < n {
            unsafe {
                // Use a volatile read as a software prefetch hint.
                let _ = std::ptr::read_volatile(src.as_ptr().add(off + 16));
            }
        }
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(off));
            vst1q_f32(dst.as_mut_ptr().add(off), v);
        }
    }
    let tail = full * LANES;
    dst[tail..n].copy_from_slice(&src[tail..n]);
}

/// Stripe-copy: copy every `stride`-th element into a contiguous output,
/// using NEON gather when `stride` is small.
pub fn gather_stride_neon(src: &[f32], stride: usize, count: usize, dst: &mut [f32]) {
    assert!(stride > 0, "stride must be > 0");
    assert!(src.len() > (count.saturating_sub(1)) * stride || count == 0, "src too short");
    assert!(dst.len() >= count, "dst too short");

    // Scalar gather – NEON gather is not available for f32 on NEON
    // (no native vgatherq equivalent), so we use scalar with prefetch.
    for i in 0..count {
        dst[i] = src[i * stride];
    }
}

/// Scatter-store: write contiguous `src` elements to every `stride`-th
/// position in `dst`.
pub fn scatter_stride_neon(src: &[f32], stride: usize, count: usize, dst: &mut [f32]) {
    assert!(stride > 0, "stride must be > 0");
    assert!(src.len() >= count, "src too short");
    assert!(dst.len() > (count.saturating_sub(1)) * stride || count == 0, "dst too short");

    for i in 0..count {
        dst[i * stride] = src[i];
    }
}

// ── Block-Copy Utilities ───────────────────────────────────────────────

/// Copy a sub-matrix (block) from a larger row-major matrix using NEON.
///
/// Copies `block_rows × block_cols` starting at `(start_row, start_col)`.
pub fn copy_block_neon(
    src: &[f32],
    src_cols: usize,
    start_row: usize,
    start_col: usize,
    block_rows: usize,
    block_cols: usize,
    dst: &mut [f32],
) {
    assert!(
        dst.len() >= block_rows * block_cols,
        "dst too short: {} < {}",
        dst.len(),
        block_rows * block_cols,
    );
    for r in 0..block_rows {
        let sr = start_row + r;
        let src_off = sr * src_cols + start_col;
        let dst_off = r * block_cols;
        neon_copy_row(&src[src_off..], &mut dst[dst_off..], block_cols);
    }
}

/// Write a block back into a larger row-major matrix.
pub fn write_block_neon(
    block: &[f32],
    block_cols: usize,
    block_rows: usize,
    dst: &mut [f32],
    dst_cols: usize,
    start_row: usize,
    start_col: usize,
) {
    assert!(block.len() >= block_rows * block_cols, "block too short");
    for r in 0..block_rows {
        let dr = start_row + r;
        let dst_off = dr * dst_cols + start_col;
        let src_off = r * block_cols;
        neon_copy_row(&block[src_off..], &mut dst[dst_off..], block_cols);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────

    /// Build a sequential f32 vector: [1.0, 2.0, …, n].
    fn seq(n: usize) -> Vec<f32> {
        (1..=n).map(|x| x as f32).collect()
    }

    /// Assert two f32 slices are approximately equal.
    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (tol={tol})");
        }
    }

    // ── transpose_4x4_neon ─────────────────────────────────────────────

    #[test]
    fn test_transpose_4x4_identity_like() {
        // Row-major 4×4: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        let input = seq(16);
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);

        #[rustfmt::skip]
        let expected: Vec<f32> = vec![
            1.0, 5.0,  9.0, 13.0,
            2.0, 6.0, 10.0, 14.0,
            3.0, 7.0, 11.0, 15.0,
            4.0, 8.0, 12.0, 16.0,
        ];
        assert_approx(&output, &expected, 0.0);
    }

    #[test]
    fn test_transpose_4x4_double_is_identity() {
        let input = seq(16);
        let mut mid = vec![0.0f32; 16];
        let mut back = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut mid);
        transpose_4x4_neon(&mid, &mut back);
        assert_approx(&back, &input, 0.0);
    }

    #[test]
    fn test_transpose_4x4_all_zeros() {
        let input = vec![0.0f32; 16];
        let mut output = vec![1.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
        assert_approx(&output, &input, 0.0);
    }

    #[test]
    fn test_transpose_4x4_all_same() {
        let input = vec![42.0f32; 16];
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
        assert_approx(&output, &input, 0.0);
    }

    #[test]
    fn test_transpose_4x4_negative_values() {
        let input: Vec<f32> = (-8..8).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
        // Double-transpose should recover.
        let mut back = vec![0.0f32; 16];
        transpose_4x4_neon(&output, &mut back);
        assert_approx(&back, &input, 0.0);
    }

    #[test]
    fn test_transpose_4x4_diagonal() {
        #[rustfmt::skip]
        let input = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
        ];
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
        // Diagonal matrix is its own transpose.
        assert_approx(&output, &input, 0.0);
    }

    #[test]
    fn test_transpose_4x4_longer_slices() {
        // Extra elements beyond 16 should be untouched.
        let mut input = seq(20);
        input[16..].fill(99.0);
        let mut output = vec![0.0f32; 20];
        transpose_4x4_neon(&input, &mut output);
        // Only first 16 should be transposed.
        assert_eq!(output[16], 0.0);
    }

    #[test]
    #[should_panic(expected = "input must have >= 16")]
    fn test_transpose_4x4_short_input_panics() {
        let input = vec![0.0f32; 10];
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "output must have >= 16")]
    fn test_transpose_4x4_short_output_panics() {
        let input = vec![0.0f32; 16];
        let mut output = vec![0.0f32; 8];
        transpose_4x4_neon(&input, &mut output);
    }

    // ── transpose_2d_neon ──────────────────────────────────────────────

    #[test]
    fn test_transpose_2d_4x4() {
        let input = seq(16);
        let mut output = vec![0.0f32; 16];
        transpose_2d_neon(&input, 4, 4, &mut output);

        let mut expected = vec![0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                expected[c * 4 + r] = input[r * 4 + c];
            }
        }
        assert_approx(&output, &expected, 0.0);
    }

    #[test]
    fn test_transpose_2d_8x8() {
        let input = seq(64);
        let mut output = vec![0.0f32; 64];
        transpose_2d_neon(&input, 8, 8, &mut output);

        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(output[c * 8 + r], input[r * 8 + c], "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_transpose_2d_non_square() {
        let rows = 3;
        let cols = 7;
        let input = seq(rows * cols);
        let mut output = vec![0.0f32; rows * cols];
        transpose_2d_neon(&input, rows, cols, &mut output);

        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(output[c * rows + r], input[r * cols + c]);
            }
        }
    }

    #[test]
    fn test_transpose_2d_wide() {
        let rows = 2;
        let cols = 12;
        let input = seq(rows * cols);
        let mut output = vec![0.0f32; rows * cols];
        transpose_2d_neon(&input, rows, cols, &mut output);

        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(output[c * rows + r], input[r * cols + c]);
            }
        }
    }

    #[test]
    fn test_transpose_2d_tall() {
        let rows = 11;
        let cols = 3;
        let input = seq(rows * cols);
        let mut output = vec![0.0f32; rows * cols];
        transpose_2d_neon(&input, rows, cols, &mut output);

        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(output[c * rows + r], input[r * cols + c]);
            }
        }
    }

    #[test]
    fn test_transpose_2d_single_row() {
        let input = seq(5);
        let mut output = vec![0.0f32; 5];
        transpose_2d_neon(&input, 1, 5, &mut output);
        // 1×5 transposed is 5×1 (same data, column-major).
        assert_approx(&output, &input, 0.0);
    }

    #[test]
    fn test_transpose_2d_single_col() {
        let input = seq(5);
        let mut output = vec![0.0f32; 5];
        transpose_2d_neon(&input, 5, 1, &mut output);
        assert_approx(&output, &input, 0.0);
    }

    #[test]
    fn test_transpose_2d_double_is_identity() {
        let rows = 6;
        let cols = 10;
        let input = seq(rows * cols);
        let mut mid = vec![0.0f32; rows * cols];
        let mut back = vec![0.0f32; rows * cols];
        transpose_2d_neon(&input, rows, cols, &mut mid);
        transpose_2d_neon(&mid, cols, rows, &mut back);
        assert_approx(&back, &input, 0.0);
    }

    #[test]
    fn test_transpose_2d_1x1() {
        let input = vec![7.0f32];
        let mut output = vec![0.0f32; 1];
        transpose_2d_neon(&input, 1, 1, &mut output);
        assert_eq!(output[0], 7.0);
    }

    // ── interleave_neon ────────────────────────────────────────────────

    #[test]
    fn test_interleave_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 8];
        interleave_neon(&a, &b, &mut out);
        assert_approx(&out, &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], 0.0);
    }

    #[test]
    fn test_interleave_8_elements() {
        let a: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let b: Vec<f32> = (100..108).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 16];
        interleave_neon(&a, &b, &mut out);
        for i in 0..8 {
            assert_eq!(out[i * 2], a[i]);
            assert_eq!(out[i * 2 + 1], b[i]);
        }
    }

    #[test]
    fn test_interleave_non_multiple_of_4() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut out = vec![0.0f32; 14];
        interleave_neon(&a, &b, &mut out);
        for i in 0..7 {
            assert_eq!(out[i * 2], a[i]);
            assert_eq!(out[i * 2 + 1], b[i]);
        }
    }

    #[test]
    fn test_interleave_single_element() {
        let a = vec![3.14];
        let b = vec![2.71];
        let mut out = vec![0.0f32; 2];
        interleave_neon(&a, &b, &mut out);
        assert_approx(&out, &[3.14, 2.71], 1e-6);
    }

    #[test]
    fn test_interleave_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        interleave_neon(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_interleave_mismatched_uses_min() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0];
        let mut out = vec![0.0f32; 4];
        interleave_neon(&a, &b, &mut out);
        // count = min(3,2) = 2
        assert_approx(&out, &[1.0, 10.0, 2.0, 20.0], 0.0);
    }

    // ── deinterleave_neon ──────────────────────────────────────────────

    #[test]
    fn test_deinterleave_basic() {
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let mut a = vec![0.0f32; 4];
        let mut b = vec![0.0f32; 4];
        deinterleave_neon(&input, &mut a, &mut b);
        assert_approx(&a, &[1.0, 2.0, 3.0, 4.0], 0.0);
        assert_approx(&b, &[10.0, 20.0, 30.0, 40.0], 0.0);
    }

    #[test]
    fn test_deinterleave_non_multiple_of_4() {
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut a = vec![0.0f32; 3];
        let mut b = vec![0.0f32; 3];
        deinterleave_neon(&input, &mut a, &mut b);
        assert_approx(&a, &[1.0, 2.0, 3.0], 0.0);
        assert_approx(&b, &[10.0, 20.0, 30.0], 0.0);
    }

    #[test]
    fn test_deinterleave_single_pair() {
        let input = vec![5.0, 6.0];
        let mut a = vec![0.0f32; 1];
        let mut b = vec![0.0f32; 1];
        deinterleave_neon(&input, &mut a, &mut b);
        assert_eq!(a[0], 5.0);
        assert_eq!(b[0], 6.0);
    }

    #[test]
    fn test_deinterleave_empty() {
        let input: Vec<f32> = vec![];
        let mut a: Vec<f32> = vec![];
        let mut b: Vec<f32> = vec![];
        deinterleave_neon(&input, &mut a, &mut b);
    }

    // ── interleave / deinterleave round-trip ───────────────────────────

    #[test]
    fn test_interleave_deinterleave_roundtrip_4() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut interleaved = vec![0.0f32; 8];
        interleave_neon(&a, &b, &mut interleaved);
        let mut a2 = vec![0.0f32; 4];
        let mut b2 = vec![0.0f32; 4];
        deinterleave_neon(&interleaved, &mut a2, &mut b2);
        assert_approx(&a2, &a, 0.0);
        assert_approx(&b2, &b, 0.0);
    }

    #[test]
    fn test_interleave_deinterleave_roundtrip_9() {
        let a: Vec<f32> = (0..9).map(|x| x as f32).collect();
        let b: Vec<f32> = (100..109).map(|x| x as f32).collect();
        let mut interleaved = vec![0.0f32; 18];
        interleave_neon(&a, &b, &mut interleaved);
        let mut a2 = vec![0.0f32; 9];
        let mut b2 = vec![0.0f32; 9];
        deinterleave_neon(&interleaved, &mut a2, &mut b2);
        assert_approx(&a2, &a, 0.0);
        assert_approx(&b2, &b, 0.0);
    }

    // ── multi-channel interleave/deinterleave ──────────────────────────

    #[test]
    fn test_interleave_multi_2ch() {
        let ch0 = vec![1.0, 2.0, 3.0, 4.0];
        let ch1 = vec![10.0, 20.0, 30.0, 40.0];
        let inputs: Vec<&[f32]> = vec![&ch0, &ch1];
        let mut out = vec![0.0f32; 8];
        interleave_multi_neon(&inputs, 2, 4, &mut out);
        assert_approx(&out, &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], 0.0);
    }

    #[test]
    fn test_interleave_multi_3ch() {
        let ch0 = vec![1.0, 2.0];
        let ch1 = vec![10.0, 20.0];
        let ch2 = vec![100.0, 200.0];
        let inputs: Vec<&[f32]> = vec![&ch0, &ch1, &ch2];
        let mut out = vec![0.0f32; 6];
        interleave_multi_neon(&inputs, 3, 2, &mut out);
        assert_approx(&out, &[1.0, 10.0, 100.0, 2.0, 20.0, 200.0], 0.0);
    }

    #[test]
    fn test_deinterleave_multi_3ch() {
        let input = vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
        let mut ch0 = vec![0.0f32; 2];
        let mut ch1 = vec![0.0f32; 2];
        let mut ch2 = vec![0.0f32; 2];
        let mut outputs: Vec<&mut [f32]> = vec![&mut ch0, &mut ch1, &mut ch2];
        deinterleave_multi_neon(&input, 3, 2, &mut outputs);
        assert_approx(&ch0, &[1.0, 2.0], 0.0);
        assert_approx(&ch1, &[10.0, 20.0], 0.0);
        assert_approx(&ch2, &[100.0, 200.0], 0.0);
    }

    #[test]
    fn test_multi_roundtrip_2ch() {
        let ch0: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let ch1: Vec<f32> = (100..108).map(|x| x as f32).collect();
        let inputs: Vec<&[f32]> = vec![&ch0, &ch1];
        let mut interleaved = vec![0.0f32; 16];
        interleave_multi_neon(&inputs, 2, 8, &mut interleaved);

        let mut out0 = vec![0.0f32; 8];
        let mut out1 = vec![0.0f32; 8];
        let mut outputs: Vec<&mut [f32]> = vec![&mut out0, &mut out1];
        deinterleave_multi_neon(&interleaved, 2, 8, &mut outputs);
        assert_approx(&out0, &ch0, 0.0);
        assert_approx(&out1, &ch1, 0.0);
    }

    // ── pad_to_neon_alignment ──────────────────────────────────────────

    #[test]
    fn test_pad_already_aligned() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_to_neon_alignment(&data, 0.0);
        assert_eq!(padded.len(), 4);
        assert_approx(&padded, &data, 0.0);
    }

    #[test]
    fn test_pad_needs_padding() {
        let data = vec![1.0, 2.0, 3.0];
        let padded = pad_to_neon_alignment(&data, 0.0);
        assert_eq!(padded.len(), 4);
        assert_approx(&padded, &[1.0, 2.0, 3.0, 0.0], 0.0);
    }

    #[test]
    fn test_pad_empty() {
        let data: Vec<f32> = vec![];
        let padded = pad_to_neon_alignment(&data, -1.0);
        assert!(padded.is_empty());
    }

    #[test]
    fn test_pad_one_element() {
        let data = vec![42.0];
        let padded = pad_to_neon_alignment(&data, 0.0);
        assert_eq!(padded.len(), 4);
        assert_approx(&padded, &[42.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_pad_custom_value() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = pad_to_neon_alignment(&data, -999.0);
        assert_eq!(padded.len(), 8);
        assert_approx(&padded, &[1.0, 2.0, 3.0, 4.0, 5.0, -999.0, -999.0, -999.0], 0.0);
    }

    #[test]
    fn test_pad_preserves_original() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let padded = pad_to_neon_alignment(&data, 0.0);
        assert_eq!(padded.len(), 8);
        assert_approx(&padded[..7], &data, 0.0);
    }

    // ── pad_matrix_neon ────────────────────────────────────────────────

    #[test]
    fn test_pad_matrix_already_aligned() {
        let data = seq(16);
        let (padded, nr, nc) = pad_matrix_neon(&data, 4, 4, 0.0);
        assert_eq!(nr, 4);
        assert_eq!(nc, 4);
        assert_approx(&padded, &data, 0.0);
    }

    #[test]
    fn test_pad_matrix_3x3() {
        let data = seq(9);
        let (padded, nr, nc) = pad_matrix_neon(&data, 3, 3, 0.0);
        assert_eq!(nr, 4);
        assert_eq!(nc, 4);
        assert_eq!(padded.len(), 16);
        // Row 0
        assert_approx(&padded[0..4], &[1.0, 2.0, 3.0, 0.0], 0.0);
        // Row 3 should be all pad.
        assert_approx(&padded[12..16], &[0.0; 4], 0.0);
    }

    #[test]
    fn test_pad_matrix_preserves_data() {
        let data = seq(6);
        let (padded, nr, nc) = pad_matrix_neon(&data, 2, 3, -1.0);
        assert_eq!(nr, 4);
        assert_eq!(nc, 4);
        // Original rows preserved.
        assert_approx(&padded[0..3], &[1.0, 2.0, 3.0], 0.0);
        assert_eq!(padded[3], -1.0); // pad col
        assert_approx(&padded[4..7], &[4.0, 5.0, 6.0], 0.0);
    }

    // ── tile_data_neon / untile_data_neon ──────────────────────────────

    #[test]
    fn test_tile_4x4_exact() {
        let input = seq(16);
        let (tiled, ntr, ntc) = tile_data_neon(&input, 4, 4, 4, 4);
        assert_eq!(ntr, 1);
        assert_eq!(ntc, 1);
        assert_approx(&tiled, &input, 0.0);
    }

    #[test]
    fn test_tile_untile_roundtrip_exact() {
        let rows = 8;
        let cols = 8;
        let input = seq(rows * cols);
        let (tiled, ntr, ntc) = tile_data_neon(&input, rows, cols, 4, 4);
        let recovered = untile_data_neon(&tiled, rows, cols, 4, 4, ntr, ntc);
        assert_approx(&recovered, &input, 0.0);
    }

    #[test]
    fn test_tile_untile_roundtrip_non_aligned() {
        let rows = 7;
        let cols = 5;
        let input = seq(rows * cols);
        let (tiled, ntr, ntc) = tile_data_neon(&input, rows, cols, 4, 4);
        let recovered = untile_data_neon(&tiled, rows, cols, 4, 4, ntr, ntc);
        assert_approx(&recovered, &input, 0.0);
    }

    #[test]
    fn test_tile_untile_roundtrip_wide() {
        let rows = 3;
        let cols = 16;
        let input = seq(rows * cols);
        let (tiled, ntr, ntc) = tile_data_neon(&input, rows, cols, 4, 8);
        let recovered = untile_data_neon(&tiled, rows, cols, 4, 8, ntr, ntc);
        assert_approx(&recovered, &input, 0.0);
    }

    #[test]
    fn test_tile_dimensions() {
        let (_, ntr, ntc) = tile_data_neon(&seq(35), 5, 7, 4, 4);
        assert_eq!(ntr, 2); // ceil(5/4)
        assert_eq!(ntc, 2); // ceil(7/4)
    }

    #[test]
    fn test_tile_1x1_tiles() {
        let input = seq(6);
        let (tiled, ntr, ntc) = tile_data_neon(&input, 2, 3, 1, 1);
        assert_eq!(ntr, 2);
        assert_eq!(ntc, 3);
        assert_approx(&tiled, &input, 0.0);
    }

    #[test]
    #[should_panic(expected = "tile dimensions must be > 0")]
    fn test_tile_zero_size_panics() {
        tile_data_neon(&[1.0], 1, 1, 0, 4);
    }

    // ── cache_aware_copy_neon ──────────────────────────────────────────

    #[test]
    fn test_cache_copy_basic() {
        let src = seq(32);
        let mut dst = vec![0.0f32; 32];
        cache_aware_copy_neon(&src, &mut dst, 32);
        assert_approx(&dst, &src, 0.0);
    }

    #[test]
    fn test_cache_copy_non_aligned() {
        let src = seq(7);
        let mut dst = vec![0.0f32; 7];
        cache_aware_copy_neon(&src, &mut dst, 7);
        assert_approx(&dst, &src, 0.0);
    }

    #[test]
    fn test_cache_copy_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        cache_aware_copy_neon(&src, &mut dst, 0);
    }

    #[test]
    fn test_cache_copy_single() {
        let src = vec![3.14];
        let mut dst = vec![0.0f32; 1];
        cache_aware_copy_neon(&src, &mut dst, 1);
        assert_approx(&dst, &src, 1e-6);
    }

    // ── gather_stride_neon / scatter_stride_neon ───────────────────────

    #[test]
    fn test_gather_stride_2() {
        let src = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dst = vec![0.0f32; 4];
        gather_stride_neon(&src, 2, 4, &mut dst);
        assert_approx(&dst, &[0.0, 2.0, 4.0, 6.0], 0.0);
    }

    #[test]
    fn test_gather_stride_3() {
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut dst = vec![0.0f32; 3];
        gather_stride_neon(&src, 3, 3, &mut dst);
        assert_approx(&dst, &[10.0, 40.0, 70.0], 0.0);
    }

    #[test]
    fn test_gather_stride_1() {
        let src = seq(5);
        let mut dst = vec![0.0f32; 5];
        gather_stride_neon(&src, 1, 5, &mut dst);
        assert_approx(&dst, &src, 0.0);
    }

    #[test]
    fn test_scatter_stride_2() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0f32; 8];
        scatter_stride_neon(&src, 2, 4, &mut dst);
        assert_approx(&dst, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], 0.0);
    }

    #[test]
    fn test_gather_scatter_roundtrip() {
        let original = vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0, 0.0];
        let mut gathered = vec![0.0f32; 4];
        gather_stride_neon(&original, 2, 4, &mut gathered);
        let mut scattered = vec![0.0f32; 8];
        scatter_stride_neon(&gathered, 2, 4, &mut scattered);
        // Even-index elements should match.
        for i in 0..4 {
            assert_eq!(scattered[i * 2], original[i * 2]);
        }
    }

    #[test]
    fn test_gather_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        gather_stride_neon(&src, 1, 0, &mut dst);
    }

    #[test]
    fn test_scatter_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        scatter_stride_neon(&src, 1, 0, &mut dst);
    }

    // ── copy_block_neon / write_block_neon ──────────────────────────────

    #[test]
    fn test_copy_block_basic() {
        #[rustfmt::skip]
        let src = vec![
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut block = vec![0.0f32; 4];
        copy_block_neon(&src, 4, 1, 1, 2, 2, &mut block);
        assert_approx(&block, &[6.0, 7.0, 10.0, 11.0], 0.0);
    }

    #[test]
    fn test_copy_block_full() {
        let src = seq(16);
        let mut block = vec![0.0f32; 16];
        copy_block_neon(&src, 4, 0, 0, 4, 4, &mut block);
        assert_approx(&block, &src, 0.0);
    }

    #[test]
    fn test_write_block_basic() {
        let block = vec![99.0, 98.0, 97.0, 96.0];
        let mut dst = vec![0.0f32; 16];
        write_block_neon(&block, 2, 2, &mut dst, 4, 1, 1);
        assert_eq!(dst[5], 99.0);
        assert_eq!(dst[6], 98.0);
        assert_eq!(dst[9], 97.0);
        assert_eq!(dst[10], 96.0);
    }

    #[test]
    fn test_copy_write_roundtrip() {
        let src = seq(64);
        let mut block = vec![0.0f32; 16];
        copy_block_neon(&src, 8, 2, 3, 4, 4, &mut block);
        let mut dst = vec![0.0f32; 64];
        write_block_neon(&block, 4, 4, &mut dst, 8, 2, 3);
        // Verify the block region matches.
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dst[(2 + r) * 8 + (3 + c)], src[(2 + r) * 8 + (3 + c)],);
            }
        }
    }

    #[test]
    fn test_copy_block_single_row() {
        let src = seq(8);
        let mut block = vec![0.0f32; 4];
        copy_block_neon(&src, 8, 0, 2, 1, 4, &mut block);
        assert_approx(&block, &[3.0, 4.0, 5.0, 6.0], 0.0);
    }

    // ── neon_copy_row (indirect via cache_aware_copy_neon) ─────────────

    #[test]
    fn test_neon_copy_large() {
        let src: Vec<f32> = (0..256).map(|x| x as f32).collect();
        let mut dst = vec![0.0f32; 256];
        cache_aware_copy_neon(&src, &mut dst, 256);
        assert_approx(&dst, &src, 0.0);
    }

    #[test]
    fn test_neon_copy_13_elements() {
        let src = seq(13);
        let mut dst = vec![0.0f32; 13];
        cache_aware_copy_neon(&src, &mut dst, 13);
        assert_approx(&dst, &src, 0.0);
    }

    // ── edge cases & stress tests ──────────────────────────────────────

    #[test]
    fn test_transpose_2d_large() {
        let rows = 33;
        let cols = 17;
        let input = seq(rows * cols);
        let mut output = vec![0.0f32; rows * cols];
        transpose_2d_neon(&input, rows, cols, &mut output);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(output[c * rows + r], input[r * cols + c]);
            }
        }
    }

    #[test]
    fn test_interleave_large() {
        let n = 129;
        let a: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b: Vec<f32> = (1000..1000 + n as i32).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; n * 2];
        interleave_neon(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i * 2], a[i]);
            assert_eq!(out[i * 2 + 1], b[i]);
        }
    }

    #[test]
    fn test_pad_alignment_property() {
        for len in 0..20 {
            let data: Vec<f32> = (0..len).map(|x| x as f32).collect();
            let padded = pad_to_neon_alignment(&data, 0.0);
            assert_eq!(padded.len() % NEON_ALIGN_F32, 0, "len={len} not aligned");
            assert!(padded.len() >= data.len());
        }
    }

    #[test]
    fn test_tile_single_element() {
        let input = vec![7.0];
        let (tiled, ntr, ntc) = tile_data_neon(&input, 1, 1, 4, 4);
        assert_eq!(ntr, 1);
        assert_eq!(ntc, 1);
        assert_eq!(tiled[0], 7.0);
        let recovered = untile_data_neon(&tiled, 1, 1, 4, 4, ntr, ntc);
        assert_eq!(recovered[0], 7.0);
    }

    #[test]
    fn test_cache_copy_partial() {
        let src = seq(32);
        let mut dst = vec![0.0f32; 32];
        cache_aware_copy_neon(&src, &mut dst, 10);
        assert_approx(&dst[..10], &src[..10], 0.0);
    }

    #[test]
    fn test_transpose_4x4_fractional_values() {
        let input: Vec<f32> = (0..16).map(|x| x as f32 * 0.1).collect();
        let mut output = vec![0.0f32; 16];
        transpose_4x4_neon(&input, &mut output);
        let mut back = vec![0.0f32; 16];
        transpose_4x4_neon(&output, &mut back);
        assert_approx(&back, &input, 1e-6);
    }

    #[test]
    fn test_interleave_deinterleave_roundtrip_large() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|x| x as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|x| -(x as f32)).collect();
        let mut interleaved = vec![0.0f32; n * 2];
        interleave_neon(&a, &b, &mut interleaved);
        let mut a2 = vec![0.0f32; n];
        let mut b2 = vec![0.0f32; n];
        deinterleave_neon(&interleaved, &mut a2, &mut b2);
        assert_approx(&a2, &a, 1e-6);
        assert_approx(&b2, &b, 1e-6);
    }
}
