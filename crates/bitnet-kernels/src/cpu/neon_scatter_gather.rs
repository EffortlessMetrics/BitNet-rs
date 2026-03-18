//! ARM NEON scatter-gather operations for Apple Silicon.
//!
//! Provides vectorised gather, scatter-add, index-select, and masked-fill
//! using NEON intrinsics, with scalar fallback for remainder elements.
//! Every public function validates indices before touching memory.

use std::arch::aarch64::*;

// ── helpers ────────────────────────────────────────────────────────

/// Validate that every index in `indices` is `< bound`.
#[inline]
fn check_bounds(indices: &[usize], bound: usize) {
    for (pos, &idx) in indices.iter().enumerate() {
        assert!(
            idx < bound,
            "neon_scatter_gather: index {idx} at position {pos} \
             is out of bounds for length {bound}",
        );
    }
}

// ── gather ─────────────────────────────────────────────────────────

/// Gather elements from `src` by index: `output[i] = src[indices[i]]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * `indices.len() != output.len()`
/// * Any index in `indices` ≥ `src.len()`
#[target_feature(enable = "neon")]
pub unsafe fn neon_gather(src: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        indices.len(),
        output.len(),
        "neon_gather: indices.len() ({}) != output.len() ({})",
        indices.len(),
        output.len(),
    );
    check_bounds(indices, src.len());

    let n = indices.len();
    let chunks = n / 4;
    let o_ptr = output.as_mut_ptr();
    let s_ptr = src.as_ptr();

    // NEON doesn't have a native gather, but we can load four gathered
    // scalars into a register and store as a contiguous vector.
    for i in 0..chunks {
        let base = i * 4;
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

    // Scalar tail
    for i in (chunks * 4)..n {
        output[i] = src[indices[i]];
    }
}

// ── scatter-add ────────────────────────────────────────────────────

/// Scatter-add: `output[indices[i]] += src[i]` for every `i`.
///
/// When multiple `indices` map to the same position the additions
/// accumulate in iteration order (deterministic).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * `src.len() != indices.len()`
/// * Any index in `indices` ≥ `output.len()`
#[target_feature(enable = "neon")]
pub unsafe fn neon_scatter_add(src: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        src.len(),
        indices.len(),
        "neon_scatter_add: src.len() ({}) != indices.len() ({})",
        src.len(),
        indices.len(),
    );
    check_bounds(indices, output.len());

    let n = src.len();
    let chunks = n / 4;
    let s_ptr = src.as_ptr();

    for i in 0..chunks {
        let base = i * 4;
        // Load 4 contiguous source values with NEON, then scatter them
        // back. Because destination indices may alias we must read-add-
        // write each lane individually.
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

    // Scalar tail
    for i in (chunks * 4)..n {
        output[indices[i]] += src[i];
    }
}

// ── index-select ───────────────────────────────────────────────────

/// Select rows (or columns) from a 2-D logical view of `src`.
///
/// `src` is treated as a matrix with `src.len() / dim_size` rows, each
/// of width `dim_size`. For every index `k` in `indices` the
/// corresponding row `src[k*dim_size .. (k+1)*dim_size]` is copied into
/// `output`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * `src.len() % dim_size != 0`
/// * `output.len() != indices.len() * dim_size`
/// * Any index in `indices` ≥ `src.len() / dim_size`
#[target_feature(enable = "neon")]
pub unsafe fn neon_index_select(
    src: &[f32],
    dim_size: usize,
    indices: &[usize],
    output: &mut [f32],
) {
    assert!(dim_size > 0, "neon_index_select: dim_size must be > 0");
    assert_eq!(
        src.len() % dim_size,
        0,
        "neon_index_select: src.len() ({}) not divisible by dim_size ({dim_size})",
        src.len(),
    );
    let num_rows = src.len() / dim_size;
    assert_eq!(
        output.len(),
        indices.len() * dim_size,
        "neon_index_select: output.len() ({}) != indices.len() ({}) * dim_size ({dim_size})",
        output.len(),
        indices.len(),
    );
    check_bounds(indices, num_rows);

    let s_ptr = src.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for (out_row, &idx) in indices.iter().enumerate() {
        let src_off = idx * dim_size;
        let dst_off = out_row * dim_size;
        let chunks = dim_size / 4;

        // Vectorised copy in 4-wide chunks
        for c in 0..chunks {
            let off = c * 4;
            unsafe {
                let v = vld1q_f32(s_ptr.add(src_off + off));
                vst1q_f32(o_ptr.add(dst_off + off), v);
            }
        }

        // Scalar tail
        let tail_start = chunks * 4;
        output[dst_off + tail_start..dst_off + dim_size]
            .copy_from_slice(&src[src_off + tail_start..src_off + dim_size]);
    }
}

// ── masked fill ────────────────────────────────────────────────────

/// Fill positions where `mask[i]` is `true` with `value`.
///
/// Uses NEON bitwise-select (`vbslq_f32`) to blend `value` into `data`
/// without branching in the hot loop.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// * `data.len() != mask.len()`
#[target_feature(enable = "neon")]
pub unsafe fn neon_masked_fill(data: &mut [f32], mask: &[bool], value: f32) {
    assert_eq!(
        data.len(),
        mask.len(),
        "neon_masked_fill: data.len() ({}) != mask.len() ({})",
        data.len(),
        mask.len(),
    );

    let n = data.len();
    let chunks = n / 4;
    let d_ptr = data.as_mut_ptr();
    let v_fill = vdupq_n_f32(value);

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            // Build a NEON mask: 0xFFFFFFFF where true, 0 where false.
            let m0 = if mask[base] { !0u32 } else { 0u32 };
            let m1 = if mask[base + 1] { !0u32 } else { 0u32 };
            let m2 = if mask[base + 2] { !0u32 } else { 0u32 };
            let m3 = if mask[base + 3] { !0u32 } else { 0u32 };
            let vmask = vld1q_u32([m0, m1, m2, m3].as_ptr());

            let vdata = vld1q_f32(d_ptr.add(base));
            // vbslq: for each bit, pick fill where mask=1, data where 0
            let vresult = vbslq_f32(vmask, v_fill, vdata);
            vst1q_f32(d_ptr.add(base), vresult);
        }
    }

    // Scalar tail
    for i in (chunks * 4)..n {
        if mask[i] {
            data[i] = value;
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: call gather through the target_feature wrapper.
    fn gather(src: &[f32], indices: &[usize], out: &mut [f32]) {
        unsafe { neon_gather(src, indices, out) };
    }

    fn scatter_add(src: &[f32], indices: &[usize], out: &mut [f32]) {
        unsafe { neon_scatter_add(src, indices, out) };
    }

    fn index_select(src: &[f32], dim: usize, idx: &[usize], out: &mut [f32]) {
        unsafe { neon_index_select(src, dim, idx, out) };
    }

    fn masked_fill(data: &mut [f32], mask: &[bool], value: f32) {
        unsafe { neon_masked_fill(data, mask, value) };
    }

    // ── gather tests ───────────────────────────────────────────────

    #[test]
    fn test_gather_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 2, 0, 3, 1];
        let mut out = [0.0; 5];
        gather(&src, &indices, &mut out);
        assert_eq!(out, vec![50.0, 30.0, 10.0, 40.0, 20.0]);
    }

    #[test]
    fn test_gather_tail() {
        // Length not divisible by 4 — exercises scalar tail.
        let src = [1.0, 2.0, 3.0];
        let indices = [2, 0, 1];
        let mut out = [0.0; 3];
        gather(&src, &indices, &mut out);
        assert_eq!(out, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_oob() {
        let src = [1.0, 2.0];
        let indices = [5];
        let mut out = [0.0; 1];
        gather(&src, &indices, &mut out);
    }

    // ── scatter-add tests ──────────────────────────────────────────

    #[test]
    fn test_scatter_add_basic() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = [0, 1, 2, 3, 4];
        let mut out = [10.0; 5];
        scatter_add(&src, &indices, &mut out);
        assert_eq!(out, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_scatter_add_duplicate_indices() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 0, 1, 1];
        let mut out = [0.0; 2];
        scatter_add(&src, &indices, &mut out);
        assert_eq!(out, vec![3.0, 7.0]); // 1+2, 3+4
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_add_oob() {
        let src = [1.0];
        let indices = [10];
        let mut out = [0.0; 2];
        scatter_add(&src, &indices, &mut out);
    }

    // ── index-select tests ─────────────────────────────────────────

    #[test]
    fn test_index_select_rows() {
        // 4 rows × 3 cols
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let indices = [3, 0];
        let mut out = [0.0; 6];
        index_select(&src, 3, &indices, &mut out);
        assert_eq!(out, vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_index_select_wide_row() {
        // Exercises the NEON copy path (dim_size ≥ 4).
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let indices = [1]; // row 1 of 2×8
        let mut out = [0.0; 8];
        index_select(&src, 8, &indices, &mut out);
        let expected: Vec<f32> = (8..16).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_index_select_oob() {
        let src = [0.0; 8]; // 2 rows × 4
        let indices = [5]; // invalid
        let mut out = [0.0; 4];
        index_select(&src, 4, &indices, &mut out);
    }

    // ── masked-fill tests ──────────────────────────────────────────

    #[test]
    fn test_masked_fill_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, false, true, false, true];
        masked_fill(&mut data, &mask, -1.0);
        assert_eq!(data, vec![-1.0, 2.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn test_masked_fill_all_true() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let mask = [true; 4];
        masked_fill(&mut data, &mask, 0.0);
        assert_eq!(data, vec![0.0; 4]);
    }

    #[test]
    fn test_masked_fill_all_false() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        let mask = [false; 4];
        masked_fill(&mut data, &mask, 99.0);
        assert_eq!(data, original);
    }

    #[test]
    fn test_gather_large() {
        // Verify multi-chunk + tail on a larger input.
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..100).rev().collect();
        let mut out = [0.0; 100];
        gather(&src, &indices, &mut out);
        let expected: Vec<f32> = (0..100).rev().map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }
}
