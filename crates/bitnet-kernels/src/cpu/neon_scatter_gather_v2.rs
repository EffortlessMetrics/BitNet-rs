//! ARM NEON scatter-gather v2 operations for Apple Silicon.
//!
//! Provides NEON-optimized gather, scatter, scatter-add, gather-rows,
//! index-select, and masked-scatter for `f32` tensor indexing paths.
//!
//! Each operation is split into three functions:
//! - `unsafe fn neon_<op>` — uses NEON intrinsics (aarch64 only)
//! - `fn scalar_<op>` — pure Rust scalar fallback
//! - `pub fn <op>` — public dispatcher that runtime-detects NEON
//!
//! NEON load/store (`vld1q_f32`, `vst1q_f32`) are **unsafe**; pure
//! arithmetic (`vaddq_f32`) is safe on AArch64 Rust 2024.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ────────────────────────────────────────────────────────────────────
// helpers
// ────────────────────────────────────────────────────────────────────

/// Validate that every index in `indices` is `< bound`.
#[inline]
fn check_bounds(indices: &[usize], bound: usize) {
    for (pos, &idx) in indices.iter().enumerate() {
        assert!(
            idx < bound,
            "neon_scatter_gather_v2: index {idx} at position {pos} \
             is out of bounds for length {bound}",
        );
    }
}

// ════════════════════════════════════════════════════════════════════
// 1. gather_f32  —  output[i] = src[indices[i]]
// ════════════════════════════════════════════════════════════════════

/// NEON gather: `output[i] = src[indices[i]]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gather_f32(src: &[f32], indices: &[usize], output: &mut [f32]) {
    let n = indices.len();
    let chunks = n / LANES;
    let s_ptr = src.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let v = vsetq_lane_f32::<3>(
                *s_ptr.add(indices[base + 3]),
                vsetq_lane_f32::<2>(
                    *s_ptr.add(indices[base + 2]),
                    vsetq_lane_f32::<1>(
                        *s_ptr.add(indices[base + 1]),
                        vdupq_n_f32(*s_ptr.add(indices[base])),
                    ),
                ),
            );
            vst1q_f32(o_ptr.add(base), v);
        }
    }

    for i in (chunks * LANES)..n {
        output[i] = src[indices[i]];
    }
}

/// Scalar fallback for gather.
#[inline(always)]
fn scalar_gather_f32(src: &[f32], indices: &[usize], output: &mut [f32]) {
    for (i, &idx) in indices.iter().enumerate() {
        output[i] = src[idx];
    }
}

/// Gather elements from `src` by index: `output[i] = src[indices[i]]`.
///
/// Uses NEON on aarch64 when available, otherwise falls back to scalar.
///
/// # Panics
///
/// * `indices.len() != output.len()`
/// * Any index in `indices` ≥ `src.len()`
pub fn gather_f32(src: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        indices.len(),
        output.len(),
        "gather_f32: indices.len() ({}) != output.len() ({})",
        indices.len(),
        output.len(),
    );
    if indices.is_empty() {
        return;
    }
    check_bounds(indices, src.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gather_f32(src, indices, output);
            }
            return;
        }
    }

    scalar_gather_f32(src, indices, output);
}

// ════════════════════════════════════════════════════════════════════
// 2. scatter_f32  —  output[indices[i]] = input[i]
// ════════════════════════════════════════════════════════════════════

/// NEON scatter: `output[indices[i]] = input[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scatter_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    let n = input.len();
    let chunks = n / LANES;
    let s_ptr = input.as_ptr();

    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let v = vld1q_f32(s_ptr.add(base));
            let mut buf = [0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), v);

            output[indices[base]] = buf[0];
            output[indices[base + 1]] = buf[1];
            output[indices[base + 2]] = buf[2];
            output[indices[base + 3]] = buf[3];
        }
    }

    for i in (chunks * LANES)..n {
        output[indices[i]] = input[i];
    }
}

/// Scalar fallback for scatter.
#[inline(always)]
fn scalar_scatter_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    for (i, &idx) in indices.iter().enumerate() {
        output[idx] = input[i];
    }
}

/// Scatter elements to positions: `output[indices[i]] = input[i]`.
///
/// Uses NEON on aarch64 when available, otherwise falls back to scalar.
///
/// # Panics
///
/// * `input.len() != indices.len()`
/// * Any index in `indices` ≥ `output.len()`
pub fn scatter_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        indices.len(),
        "scatter_f32: input.len() ({}) != indices.len() ({})",
        input.len(),
        indices.len(),
    );
    if input.is_empty() {
        return;
    }
    check_bounds(indices, output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_scatter_f32(input, indices, output);
            }
            return;
        }
    }

    scalar_scatter_f32(input, indices, output);
}

// ════════════════════════════════════════════════════════════════════
// 3. gather_rows_f32  —  gather entire rows from a matrix
// ════════════════════════════════════════════════════════════════════

/// NEON gather-rows: copy selected rows from a row-major matrix.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gather_rows_f32(
    matrix: &[f32],
    cols: usize,
    row_indices: &[usize],
    output: &mut [f32],
) {
    let s_ptr = matrix.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for (out_row, &row_idx) in row_indices.iter().enumerate() {
        let src_off = row_idx * cols;
        let dst_off = out_row * cols;
        let chunks = cols / LANES;

        for c in 0..chunks {
            let off = c * LANES;
            unsafe {
                let v = vld1q_f32(s_ptr.add(src_off + off));
                vst1q_f32(o_ptr.add(dst_off + off), v);
            }
        }

        let tail_start = chunks * LANES;
        output[dst_off + tail_start..dst_off + cols]
            .copy_from_slice(&matrix[src_off + tail_start..src_off + cols]);
    }
}

/// Scalar fallback for gather-rows.
#[inline(always)]
fn scalar_gather_rows_f32(matrix: &[f32], cols: usize, row_indices: &[usize], output: &mut [f32]) {
    for (out_row, &row_idx) in row_indices.iter().enumerate() {
        let src_off = row_idx * cols;
        let dst_off = out_row * cols;
        output[dst_off..dst_off + cols].copy_from_slice(&matrix[src_off..src_off + cols]);
    }
}

/// Gather entire rows from a row-major matrix.
///
/// `matrix` is `rows × cols`; for each `row_indices[i]` the corresponding
/// row is copied contiguously into `output`.
///
/// # Panics
///
/// * `cols == 0`
/// * `matrix.len() % cols != 0`
/// * `output.len() != row_indices.len() * cols`
/// * Any `row_indices[i]` ≥ number of rows
pub fn gather_rows_f32(matrix: &[f32], cols: usize, row_indices: &[usize], output: &mut [f32]) {
    assert!(cols > 0, "gather_rows_f32: cols must be > 0");
    assert_eq!(
        matrix.len() % cols,
        0,
        "gather_rows_f32: matrix.len() ({}) not divisible by cols ({cols})",
        matrix.len(),
    );
    let num_rows = matrix.len() / cols;
    assert_eq!(
        output.len(),
        row_indices.len() * cols,
        "gather_rows_f32: output.len() ({}) != row_indices.len() ({}) * cols ({cols})",
        output.len(),
        row_indices.len(),
    );
    if row_indices.is_empty() {
        return;
    }
    check_bounds(row_indices, num_rows);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_gather_rows_f32(matrix, cols, row_indices, output);
            }
            return;
        }
    }

    scalar_gather_rows_f32(matrix, cols, row_indices, output);
}

// ════════════════════════════════════════════════════════════════════
// 4. scatter_add_f32  —  output[indices[i]] += input[i]
// ════════════════════════════════════════════════════════════════════

/// NEON scatter-add: `output[indices[i]] += input[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scatter_add_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    let n = input.len();
    let chunks = n / LANES;
    let s_ptr = input.as_ptr();

    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let vs = vld1q_f32(s_ptr.add(base));
            let mut buf = [0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), vs);

            output[indices[base]] += buf[0];
            output[indices[base + 1]] += buf[1];
            output[indices[base + 2]] += buf[2];
            output[indices[base + 3]] += buf[3];
        }
    }

    for i in (chunks * LANES)..n {
        output[indices[i]] += input[i];
    }
}

/// Scalar fallback for scatter-add.
#[inline(always)]
fn scalar_scatter_add_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    for (i, &idx) in indices.iter().enumerate() {
        output[idx] += input[i];
    }
}

/// Scatter with addition: `output[indices[i]] += input[i]`.
///
/// When multiple indices map to the same position the additions
/// accumulate in iteration order (deterministic).
///
/// # Panics
///
/// * `input.len() != indices.len()`
/// * Any index in `indices` ≥ `output.len()`
pub fn scatter_add_f32(input: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        indices.len(),
        "scatter_add_f32: input.len() ({}) != indices.len() ({})",
        input.len(),
        indices.len(),
    );
    if input.is_empty() {
        return;
    }
    check_bounds(indices, output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_scatter_add_f32(input, indices, output);
            }
            return;
        }
    }

    scalar_scatter_add_f32(input, indices, output);
}

// ════════════════════════════════════════════════════════════════════
// 5. index_select_f32  —  select slices along first dimension
// ════════════════════════════════════════════════════════════════════

/// NEON index-select: select rows of width `dim_size` from `src`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_index_select_f32(
    src: &[f32],
    dim_size: usize,
    indices: &[usize],
    output: &mut [f32],
) {
    let s_ptr = src.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for (out_row, &idx) in indices.iter().enumerate() {
        let src_off = idx * dim_size;
        let dst_off = out_row * dim_size;
        let chunks = dim_size / LANES;

        for c in 0..chunks {
            let off = c * LANES;
            unsafe {
                let v = vld1q_f32(s_ptr.add(src_off + off));
                vst1q_f32(o_ptr.add(dst_off + off), v);
            }
        }

        let tail_start = chunks * LANES;
        output[dst_off + tail_start..dst_off + dim_size]
            .copy_from_slice(&src[src_off + tail_start..src_off + dim_size]);
    }
}

/// Scalar fallback for index-select.
#[inline(always)]
fn scalar_index_select_f32(src: &[f32], dim_size: usize, indices: &[usize], output: &mut [f32]) {
    for (out_row, &idx) in indices.iter().enumerate() {
        let src_off = idx * dim_size;
        let dst_off = out_row * dim_size;
        output[dst_off..dst_off + dim_size].copy_from_slice(&src[src_off..src_off + dim_size]);
    }
}

/// Select slices of size `dim_size` along the first dimension.
///
/// `src` is treated as `(src.len() / dim_size)` rows of width `dim_size`.
/// For each `indices[k]`, row `k` of `src` is copied into `output`.
///
/// # Panics
///
/// * `dim_size == 0`
/// * `src.len() % dim_size != 0`
/// * `output.len() != indices.len() * dim_size`
/// * Any `indices[i]` ≥ number of rows
pub fn index_select_f32(src: &[f32], dim_size: usize, indices: &[usize], output: &mut [f32]) {
    assert!(dim_size > 0, "index_select_f32: dim_size must be > 0");
    assert_eq!(
        src.len() % dim_size,
        0,
        "index_select_f32: src.len() ({}) not divisible by dim_size ({dim_size})",
        src.len(),
    );
    let num_rows = src.len() / dim_size;
    assert_eq!(
        output.len(),
        indices.len() * dim_size,
        "index_select_f32: output.len() ({}) != indices.len() ({}) * dim_size ({dim_size})",
        output.len(),
        indices.len(),
    );
    if indices.is_empty() {
        return;
    }
    check_bounds(indices, num_rows);

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_index_select_f32(src, dim_size, indices, output);
            }
            return;
        }
    }

    scalar_index_select_f32(src, dim_size, indices, output);
}

// ════════════════════════════════════════════════════════════════════
// 6. masked_scatter_f32  —  scatter where mask is true
// ════════════════════════════════════════════════════════════════════

/// NEON masked-scatter: where `mask[i]` is true, consume next `values`
/// element into `output[i]`; otherwise copy `input[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_masked_scatter_f32(
    input: &[f32],
    mask: &[bool],
    values: &[f32],
    output: &mut [f32],
) {
    let n = input.len();
    let chunks = n / LANES;
    let i_ptr = input.as_ptr();
    let o_ptr = output.as_mut_ptr();
    let mut val_idx: usize = 0;

    for c in 0..chunks {
        let base = c * LANES;
        unsafe {
            let v_input = vld1q_f32(i_ptr.add(base));

            // Build replacement values: consume from `values` where mask is true.
            let mut buf = [0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), v_input);

            for lane in 0..LANES {
                if mask[base + lane] {
                    buf[lane] = values[val_idx];
                    val_idx += 1;
                }
            }

            let v_out = vld1q_f32(buf.as_ptr());
            vst1q_f32(o_ptr.add(base), v_out);
        }
    }

    // Scalar tail
    for i in (chunks * LANES)..n {
        if mask[i] {
            output[i] = values[val_idx];
            val_idx += 1;
        } else {
            output[i] = input[i];
        }
    }
}

/// Scalar fallback for masked-scatter.
#[inline(always)]
fn scalar_masked_scatter_f32(input: &[f32], mask: &[bool], values: &[f32], output: &mut [f32]) {
    let mut val_idx: usize = 0;
    for i in 0..input.len() {
        if mask[i] {
            output[i] = values[val_idx];
            val_idx += 1;
        } else {
            output[i] = input[i];
        }
    }
}

/// Masked scatter: where `mask[i]` is true, consume the next element
/// from `values` into `output[i]`; otherwise copy `input[i]`.
///
/// # Panics
///
/// * `input.len() != mask.len()` or `input.len() != output.len()`
/// * `values.len() < count of true entries in mask`
pub fn masked_scatter_f32(input: &[f32], mask: &[bool], values: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(
        n,
        mask.len(),
        "masked_scatter_f32: input.len() ({n}) != mask.len() ({})",
        mask.len(),
    );
    assert_eq!(
        n,
        output.len(),
        "masked_scatter_f32: input.len() ({n}) != output.len() ({})",
        output.len(),
    );
    let true_count = mask.iter().filter(|&&b| b).count();
    assert!(
        values.len() >= true_count,
        "masked_scatter_f32: values.len() ({}) < true count in mask ({true_count})",
        values.len(),
    );
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_masked_scatter_f32(input, mask, values, output);
            }
            return;
        }
    }

    scalar_masked_scatter_f32(input, mask, values, output);
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── gather_f32 tests ───────────────────────────────────────────

    #[test]
    fn test_gather_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let idx = [4, 2, 0, 3, 1];
        let mut out = vec![0.0; 5];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, vec![50.0, 30.0, 10.0, 40.0, 20.0]);
    }

    #[test]
    fn test_gather_empty() {
        let src: [f32; 0] = [];
        let idx: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        gather_f32(&src, &idx, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gather_single() {
        let src = [42.0];
        let idx = [0];
        let mut out = vec![0.0; 1];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_gather_identity() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..16).collect();
        let mut out = vec![0.0; 16];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn test_gather_reverse() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..16).rev().collect();
        let mut out = vec![0.0; 16];
        gather_f32(&src, &idx, &mut out);
        let expected: Vec<f32> = (0..16).rev().map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_gather_len1() {
        let src = [99.0, 88.0];
        let idx = [1];
        let mut out = vec![0.0; 1];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, vec![88.0]);
    }

    #[test]
    fn test_gather_len15() {
        let src: Vec<f32> = (0..15).map(|i| i as f32 * 10.0).collect();
        let idx: Vec<usize> = (0..15).rev().collect();
        let mut out = vec![0.0; 15];
        gather_f32(&src, &idx, &mut out);
        for i in 0..15 {
            assert_eq!(out[i], (14 - i) as f32 * 10.0);
        }
    }

    #[test]
    fn test_gather_len16() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..16).rev().collect();
        let mut out = vec![0.0; 16];
        gather_f32(&src, &idx, &mut out);
        for i in 0..16 {
            assert_eq!(out[i], (15 - i) as f32);
        }
    }

    #[test]
    fn test_gather_len17() {
        let src: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..17).rev().collect();
        let mut out = vec![0.0; 17];
        gather_f32(&src, &idx, &mut out);
        for i in 0..17 {
            assert_eq!(out[i], (16 - i) as f32);
        }
    }

    #[test]
    fn test_gather_len31() {
        let src: Vec<f32> = (0..31).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..31).collect();
        let mut out = vec![0.0; 31];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn test_gather_len32() {
        let src: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..32).rev().collect();
        let mut out = vec![0.0; 32];
        gather_f32(&src, &idx, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], (31 - i) as f32);
        }
    }

    #[test]
    fn test_gather_len33() {
        let src: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..33).collect();
        let mut out = vec![0.0; 33];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn test_gather_duplicate_indices() {
        let src = [1.0, 2.0, 3.0];
        let idx = [0, 0, 1, 1, 2, 2];
        let mut out = vec![0.0; 6];
        gather_f32(&src, &idx, &mut out);
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_oob() {
        let src = [1.0, 2.0];
        let idx = [5];
        let mut out = vec![0.0; 1];
        gather_f32(&src, &idx, &mut out);
    }

    #[test]
    fn test_gather_large() {
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..100).rev().collect();
        let mut out = vec![0.0; 100];
        gather_f32(&src, &idx, &mut out);
        let expected: Vec<f32> = (0..100).rev().map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    // ── scatter_f32 tests ──────────────────────────────────────────

    #[test]
    fn test_scatter_basic() {
        let input = [10.0, 20.0, 30.0, 40.0, 50.0];
        let idx = [4, 3, 2, 1, 0];
        let mut out = vec![0.0; 5];
        scatter_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_scatter_empty() {
        let input: [f32; 0] = [];
        let idx: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        scatter_f32(&input, &idx, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_scatter_single() {
        let input = [7.0];
        let idx = [0];
        let mut out = vec![0.0; 1];
        scatter_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![7.0]);
    }

    #[test]
    fn test_scatter_identity() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..16).collect();
        let mut out = vec![0.0; 16];
        scatter_f32(&input, &idx, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn test_scatter_len15() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..15).rev().collect();
        let mut out = vec![0.0; 15];
        scatter_f32(&input, &idx, &mut out);
        for i in 0..15 {
            assert_eq!(out[14 - i], i as f32);
        }
    }

    #[test]
    fn test_scatter_len17() {
        let input: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..17).collect();
        let mut out = vec![0.0; 17];
        scatter_f32(&input, &idx, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_oob() {
        let input = [1.0];
        let idx = [10];
        let mut out = vec![0.0; 2];
        scatter_f32(&input, &idx, &mut out);
    }

    #[test]
    fn test_scatter_overwrites_last() {
        // When duplicate indices exist, last write wins.
        let input = [1.0, 2.0, 3.0, 4.0];
        let idx = [0, 0, 0, 0];
        let mut out = vec![0.0; 1];
        scatter_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![4.0]);
    }

    // ── gather_rows_f32 tests ──────────────────────────────────────

    #[test]
    fn test_gather_rows_basic() {
        // 4 rows × 3 cols
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let row_idx = [3, 0];
        let mut out = vec![0.0; 6];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
        assert_eq!(out, vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_rows_empty() {
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let row_idx: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gather_rows_single_row() {
        let matrix: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let row_idx = [1]; // 2×4 matrix, grab row 1
        let mut out = vec![0.0; 4];
        gather_rows_f32(&matrix, 4, &row_idx, &mut out);
        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_gather_rows_wide() {
        // Exercises NEON 4-wide copy path.
        let matrix: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let row_idx = [1]; // 2×16
        let mut out = vec![0.0; 16];
        gather_rows_f32(&matrix, 16, &row_idx, &mut out);
        let expected: Vec<f32> = (16..32).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_gather_rows_all() {
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let row_idx = [0, 1, 2, 3];
        let mut out = vec![0.0; 12];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
        assert_eq!(out, matrix);
    }

    #[test]
    fn test_gather_rows_reverse() {
        let matrix: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let row_idx = [3, 2, 1, 0];
        let mut out = vec![0.0; 12];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
        assert_eq!(out, vec![9.0, 10.0, 11.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_rows_large_matrix() {
        // 100 rows × 8 cols
        let matrix: Vec<f32> = (0..800).map(|i| i as f32).collect();
        let row_idx = [99, 0, 50];
        let mut out = vec![0.0; 24];
        gather_rows_f32(&matrix, 8, &row_idx, &mut out);
        // Row 99 starts at 792
        assert_eq!(out[0], 792.0);
        assert_eq!(out[7], 799.0);
        // Row 0
        assert_eq!(out[8], 0.0);
        assert_eq!(out[15], 7.0);
        // Row 50 starts at 400
        assert_eq!(out[16], 400.0);
        assert_eq!(out[23], 407.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_rows_oob() {
        let matrix = vec![0.0; 12]; // 4 rows × 3 cols
        let row_idx = [5];
        let mut out = vec![0.0; 3];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
    }

    // ── scatter_add_f32 tests ──────────────────────────────────────

    #[test]
    fn test_scatter_add_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let idx = [0, 1, 2, 3, 4];
        let mut out = vec![10.0; 5];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_scatter_add_empty() {
        let input: [f32; 0] = [];
        let idx: [usize; 0] = [];
        let mut out = vec![1.0; 3];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![1.0; 3]);
    }

    #[test]
    fn test_scatter_add_single() {
        let input = [5.0];
        let idx = [0];
        let mut out = vec![10.0];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![15.0]);
    }

    #[test]
    fn test_scatter_add_duplicate_indices() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let idx = [0, 0, 1, 1];
        let mut out = vec![0.0; 2];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![3.0, 7.0]); // 1+2, 3+4
    }

    #[test]
    fn test_scatter_add_all_same_index() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let idx = [0, 0, 0, 0, 0];
        let mut out = vec![0.0; 1];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![15.0]);
    }

    #[test]
    fn test_scatter_add_len15() {
        let input: Vec<f32> = (1..=15).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..15).collect();
        let mut out = vec![0.0; 15];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn test_scatter_add_len17() {
        let input: Vec<f32> = (1..=17).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..17).collect();
        let mut out = vec![100.0; 17];
        scatter_add_f32(&input, &idx, &mut out);
        for i in 0..17 {
            assert_eq!(out[i], 100.0 + (i + 1) as f32);
        }
    }

    #[test]
    fn test_scatter_add_preserves_existing() {
        let input = [1.0, 2.0];
        let idx = [0, 2];
        let mut out = vec![10.0, 20.0, 30.0];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![11.0, 20.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_add_oob() {
        let input = [1.0];
        let idx = [10];
        let mut out = vec![0.0; 2];
        scatter_add_f32(&input, &idx, &mut out);
    }

    // ── index_select_f32 tests ─────────────────────────────────────

    #[test]
    fn test_index_select_basic() {
        // 4 rows × 3 cols
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let idx = [3, 0];
        let mut out = vec![0.0; 6];
        index_select_f32(&src, 3, &idx, &mut out);
        assert_eq!(out, vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_index_select_empty() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let idx: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        index_select_f32(&src, 3, &idx, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_index_select_wide_row() {
        // dim_size ≥ 4, exercises NEON copy path.
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx = [1]; // 2×8 matrix, row 1
        let mut out = vec![0.0; 8];
        index_select_f32(&src, 8, &idx, &mut out);
        let expected: Vec<f32> = (8..16).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_index_select_all_rows() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let idx = [0, 1, 2, 3];
        let mut out = vec![0.0; 12];
        index_select_f32(&src, 3, &idx, &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn test_index_select_reverse() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let idx = [3, 2, 1, 0]; // 4×2 matrix
        let mut out = vec![0.0; 8];
        index_select_f32(&src, 2, &idx, &mut out);
        assert_eq!(out, vec![6.0, 7.0, 4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn test_index_select_single() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx = [2]; // 3×2, row 2
        let mut out = vec![0.0; 2];
        index_select_f32(&src, 2, &idx, &mut out);
        assert_eq!(out, vec![5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_index_select_oob() {
        let src = vec![0.0; 8]; // 2 rows × 4
        let idx = [5];
        let mut out = vec![0.0; 4];
        index_select_f32(&src, 4, &idx, &mut out);
    }

    // ── masked_scatter_f32 tests ───────────────────────────────────

    #[test]
    fn test_masked_scatter_basic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [true, false, true, false, true];
        let values = [10.0, 30.0, 50.0];
        let mut out = vec![0.0; 5];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![10.0, 2.0, 30.0, 4.0, 50.0]);
    }

    #[test]
    fn test_masked_scatter_empty() {
        let input: [f32; 0] = [];
        let mask: [bool; 0] = [];
        let values: [f32; 0] = [];
        let mut out: Vec<f32> = vec![];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_masked_scatter_single_true() {
        let input = [1.0];
        let mask = [true];
        let values = [99.0];
        let mut out = vec![0.0; 1];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![99.0]);
    }

    #[test]
    fn test_masked_scatter_single_false() {
        let input = [1.0];
        let mask = [false];
        let values: [f32; 0] = [];
        let mut out = vec![0.0; 1];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_masked_scatter_all_true() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true; 4];
        let values = [10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_masked_scatter_all_false() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false; 4];
        let values: [f32; 0] = [];
        let mut out = vec![0.0; 4];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_masked_scatter_len15() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let mask: Vec<bool> = (0..15).map(|i| i % 2 == 0).collect();
        let values: Vec<f32> = (100..108).map(|i| i as f32).collect(); // 8 true entries
        let mut out = vec![0.0; 15];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        let mut vi = 0;
        for i in 0..15 {
            if i % 2 == 0 {
                assert_eq!(out[i], values[vi]);
                vi += 1;
            } else {
                assert_eq!(out[i], i as f32);
            }
        }
    }

    #[test]
    fn test_masked_scatter_len17() {
        let input: Vec<f32> = (0..17).map(|i| i as f32).collect();
        // All true
        let mask = vec![true; 17];
        let values: Vec<f32> = (100..117).map(|i| i as f32).collect();
        let mut out = vec![0.0; 17];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, values);
    }

    #[test]
    fn test_masked_scatter_alternating() {
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mask = [true, false, true, false, true, false, true, false];
        let values = [100.0, 200.0, 300.0, 400.0];
        let mut out = vec![0.0; 8];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![100.0, 1.0, 200.0, 3.0, 300.0, 5.0, 400.0, 7.0]);
    }

    // ── NEON vs scalar consistency ─────────────────────────────────

    #[test]
    fn test_gather_neon_scalar_consistency() {
        let src: Vec<f32> = (0..33).map(|i| i as f32 * 1.5).collect();
        let idx: Vec<usize> = (0..33).rev().collect();
        let mut out_pub = vec![0.0; 33];
        let mut out_scalar = vec![0.0; 33];
        gather_f32(&src, &idx, &mut out_pub);
        scalar_gather_f32(&src, &idx, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_scatter_neon_scalar_consistency() {
        let input: Vec<f32> = (0..33).map(|i| i as f32 * 2.0).collect();
        let idx: Vec<usize> = (0..33).rev().collect();
        let mut out_pub = vec![0.0; 33];
        let mut out_scalar = vec![0.0; 33];
        scatter_f32(&input, &idx, &mut out_pub);
        scalar_scatter_f32(&input, &idx, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_scatter_add_neon_scalar_consistency() {
        let input: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..33).map(|i| i % 10).collect();
        let mut out_pub = vec![100.0; 10];
        let mut out_scalar = vec![100.0; 10];
        scatter_add_f32(&input, &idx, &mut out_pub);
        scalar_scatter_add_f32(&input, &idx, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_gather_rows_neon_scalar_consistency() {
        let matrix: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let row_idx = [4, 0, 2, 1, 3];
        let mut out_pub = vec![0.0; 40];
        let mut out_scalar = vec![0.0; 40];
        gather_rows_f32(&matrix, 8, &row_idx, &mut out_pub);
        scalar_gather_rows_f32(&matrix, 8, &row_idx, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_index_select_neon_scalar_consistency() {
        let src: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let idx = [4, 2, 0, 1, 3]; // 5 rows × 6 cols
        let mut out_pub = vec![0.0; 30];
        let mut out_scalar = vec![0.0; 30];
        index_select_f32(&src, 6, &idx, &mut out_pub);
        scalar_index_select_f32(&src, 6, &idx, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    #[test]
    fn test_masked_scatter_neon_scalar_consistency() {
        let input: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mask: Vec<bool> = (0..33).map(|i| i % 3 == 0).collect();
        let true_count = mask.iter().filter(|&&b| b).count();
        let values: Vec<f32> = (0..true_count).map(|i| 1000.0 + i as f32).collect();
        let mut out_pub = vec![0.0; 33];
        let mut out_scalar = vec![0.0; 33];
        masked_scatter_f32(&input, &mask, &values, &mut out_pub);
        scalar_masked_scatter_f32(&input, &mask, &values, &mut out_scalar);
        assert_eq!(out_pub, out_scalar);
    }

    // ── scatter-add accumulation correctness ───────────────────────

    #[test]
    fn test_scatter_add_triple_accumulation() {
        // 3 values all mapping to the same slot.
        let input = [1.0, 2.0, 3.0];
        let idx = [0, 0, 0];
        let mut out = vec![10.0; 1];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![16.0]); // 10 + 1 + 2 + 3
    }

    #[test]
    fn test_scatter_add_large_accumulation() {
        let n = 32;
        let input = vec![1.0; n];
        let idx = vec![0; n];
        let mut out = vec![0.0; 1];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, vec![n as f32]);
    }

    // ── edge cases for gather_rows ─────────────────────────────────

    #[test]
    fn test_gather_rows_single_col() {
        let matrix = [1.0, 2.0, 3.0, 4.0]; // 4×1
        let row_idx = [2, 0];
        let mut out = vec![0.0; 2];
        gather_rows_f32(&matrix, 1, &row_idx, &mut out);
        assert_eq!(out, vec![3.0, 1.0]);
    }

    #[test]
    fn test_gather_rows_duplicate_rows() {
        let matrix: Vec<f32> = (0..6).map(|i| i as f32).collect(); // 2×3
        let row_idx = [1, 1, 0, 0];
        let mut out = vec![0.0; 12];
        gather_rows_f32(&matrix, 3, &row_idx, &mut out);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    // ── scatter roundtrip ──────────────────────────────────────────

    #[test]
    fn test_scatter_gather_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let perm: Vec<usize> = (0..16).rev().collect();
        // scatter into permuted positions
        let mut scattered = vec![0.0; 16];
        scatter_f32(&data, &perm, &mut scattered);
        // gather back using same permutation
        let mut gathered = vec![0.0; 16];
        gather_f32(&scattered, &perm, &mut gathered);
        assert_eq!(gathered, data);
    }

    #[test]
    fn test_scatter_add_gather_roundtrip() {
        // scatter_add with identity indices from zeros is same as scatter.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..16).collect();
        let mut out = vec![0.0; 16];
        scatter_add_f32(&data, &idx, &mut out);
        assert_eq!(out, data);
    }

    // ── len 32/33 for scatter_add ──────────────────────────────────

    #[test]
    fn test_scatter_add_len32() {
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..32).collect();
        let mut out = vec![0.0; 32];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn test_scatter_add_len33() {
        let input: Vec<f32> = (1..=33).map(|i| i as f32).collect();
        let idx: Vec<usize> = (0..33).collect();
        let mut out = vec![0.0; 33];
        scatter_add_f32(&input, &idx, &mut out);
        assert_eq!(out, input);
    }

    // ── masked_scatter_f32 additional tests ────────────────────────

    #[test]
    fn test_masked_scatter_len32_all_true() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mask = vec![true; 32];
        let values: Vec<f32> = (100..132).map(|i| i as f32).collect();
        let mut out = vec![0.0; 32];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, values);
    }

    #[test]
    fn test_masked_scatter_len33_all_false() {
        let input: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mask = vec![false; 33];
        let values: [f32; 0] = [];
        let mut out = vec![0.0; 33];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn test_masked_scatter_values_consumed_in_order() {
        let input = vec![0.0; 8];
        let mask = [false, true, false, true, false, true, false, true];
        let values = [10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 8];
        masked_scatter_f32(&input, &mask, &values, &mut out);
        assert_eq!(out, vec![0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0]);
    }
}
