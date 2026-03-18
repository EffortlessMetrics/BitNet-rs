//! NEON-optimized gather and scatter operations for Apple Silicon.
//!
//! Provides vectorized gather (flat and batched), scatter-add with NEON
//! accumulation, indexed copy with stride support, masked gather/scatter,
//! and multi-dimensional index mapping.
//!
//! All NEON intrinsics are gated behind `#[cfg(target_arch = "aarch64")]`
//! with scalar fallbacks for other architectures. Public functions are safe;
//! internal `unsafe` blocks cover only the NEON intrinsic calls.

// ── helpers ────────────────────────────────────────────────────────

/// Validate that every index in `indices` is `< bound`.
#[inline]
fn check_bounds(indices: &[usize], bound: usize) {
    for (pos, &idx) in indices.iter().enumerate() {
        assert!(
            idx < bound,
            "neon_gather_scatter: index {idx} at position {pos} \
             is out of bounds for length {bound}",
        );
    }
}

/// Validate that masked-true indices are `< bound`.
#[inline]
fn check_bounds_masked(indices: &[usize], mask: &[bool], bound: usize) {
    for (pos, (&idx, &m)) in indices.iter().zip(mask.iter()).enumerate() {
        if m {
            assert!(
                idx < bound,
                "neon_gather_scatter: masked index {idx} at position \
                 {pos} is out of bounds for length {bound}",
            );
        }
    }
}

/// Compute a flat offset from multi-dimensional coordinates (row-major).
#[inline]
fn flat_offset(coords: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&c, &s) in coords.iter().zip(shape.iter()).rev() {
        offset += c * stride;
        stride *= s;
    }
    offset
}

/// Total number of elements implied by a shape.
#[inline]
fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

// ═══════════════════════════════════════════════════════════════════
// 1. Vectorized gather (flat + batched)
// ═══════════════════════════════════════════════════════════════════

/// Gather elements: `output[i] = src[indices[i]]` for all `i`.
///
/// Uses NEON to batch-load four gathered scalars at a time on aarch64.
///
/// # Panics
///
/// - `indices.len() != output.len()`
/// - Any index in `indices` ≥ `src.len()`
pub fn gather_f32(src: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        indices.len(),
        output.len(),
        "gather_f32: indices.len() ({}) != output.len() ({})",
        indices.len(),
        output.len(),
    );
    check_bounds(indices, src.len());

    let n = indices.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let o_ptr = output.as_mut_ptr();
        let s_ptr = src.as_ptr();
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
        for i in (chunks * 4)..n {
            output[i] = src[indices[i]];
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[i] = src[indices[i]];
        }
    }
}

/// Batched gather: process `batch_count` independent gathers.
///
/// `src` contains `batch_count` contiguous segments of `segment_len`
/// elements. `all_indices` has `batch_count * indices_per_batch`
/// entries (each index is relative to its segment). `output` has the
/// same total length as `all_indices`.
///
/// # Panics
///
/// - `all_indices.len() % indices_per_batch != 0`
/// - `output.len() != all_indices.len()`
/// - `src.len() < batch_count * segment_len`
/// - Any per-segment index ≥ `segment_len`
pub fn gather_batched(
    src: &[f32],
    segment_len: usize,
    all_indices: &[usize],
    indices_per_batch: usize,
    output: &mut [f32],
) {
    if indices_per_batch == 0 {
        return;
    }
    assert_eq!(
        all_indices.len() % indices_per_batch,
        0,
        "gather_batched: all_indices.len() not divisible by indices_per_batch"
    );
    let batch_count = all_indices.len() / indices_per_batch;
    assert_eq!(output.len(), all_indices.len());
    assert!(
        src.len() >= batch_count * segment_len,
        "gather_batched: src too small for {batch_count} batches"
    );

    for b in 0..batch_count {
        let s = &src[b * segment_len..(b + 1) * segment_len];
        let idx = &all_indices[b * indices_per_batch..(b + 1) * indices_per_batch];
        let out = &mut output[b * indices_per_batch..(b + 1) * indices_per_batch];
        gather_f32(s, idx, out);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Scatter-add with NEON accumulation
// ═══════════════════════════════════════════════════════════════════

/// Scatter-add: `output[indices[i]] += src[i]` for all `i`.
///
/// Duplicate indices accumulate in iteration order (deterministic).
///
/// # Panics
///
/// - `src.len() != indices.len()`
/// - Any index in `indices` ≥ `output.len()`
pub fn scatter_add_f32(src: &[f32], indices: &[usize], output: &mut [f32]) {
    assert_eq!(
        src.len(),
        indices.len(),
        "scatter_add_f32: src.len() ({}) != indices.len() ({})",
        src.len(),
        indices.len(),
    );
    check_bounds(indices, output.len());

    let n = src.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let s_ptr = src.as_ptr();
        for i in 0..chunks {
            let base = i * 4;
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
        for i in (chunks * 4)..n {
            output[indices[i]] += src[i];
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[indices[i]] += src[i];
        }
    }
}

/// Scatter-add with scaling: `output[indices[i]] += alpha * src[i]`.
///
/// # Panics
///
/// - `src.len() != indices.len()`
/// - Any index in `indices` ≥ `output.len()`
pub fn scatter_add_scaled(src: &[f32], indices: &[usize], alpha: f32, output: &mut [f32]) {
    assert_eq!(src.len(), indices.len(), "scatter_add_scaled: length mismatch");
    check_bounds(indices, output.len());

    let n = src.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let s_ptr = src.as_ptr();
        let valpha = unsafe { vdupq_n_f32(alpha) };
        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let vs = vmulq_f32(vld1q_f32(s_ptr.add(base)), valpha);
                let mut buf = [0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), vs);
                output[indices[base]] += buf[0];
                output[indices[base + 1]] += buf[1];
                output[indices[base + 2]] += buf[2];
                output[indices[base + 3]] += buf[3];
            }
        }
        for i in (chunks * 4)..n {
            output[indices[i]] += alpha * src[i];
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[indices[i]] += alpha * src[i];
        }
    }
}

/// Batched scatter-add across independent output segments.
///
/// `src` has `batch_count * values_per_batch` elements.
/// `all_indices` has the same length (indices within each output segment).
/// `output` has `batch_count * output_segment_len` elements.
pub fn scatter_add_batched(
    src: &[f32],
    all_indices: &[usize],
    values_per_batch: usize,
    output: &mut [f32],
    output_segment_len: usize,
) {
    if values_per_batch == 0 {
        return;
    }
    assert_eq!(src.len(), all_indices.len());
    assert_eq!(
        src.len() % values_per_batch,
        0,
        "scatter_add_batched: src not divisible by values_per_batch"
    );
    let batch_count = src.len() / values_per_batch;
    assert!(output.len() >= batch_count * output_segment_len);

    for b in 0..batch_count {
        let s = &src[b * values_per_batch..(b + 1) * values_per_batch];
        let idx = &all_indices[b * values_per_batch..(b + 1) * values_per_batch];
        let out = &mut output[b * output_segment_len..(b + 1) * output_segment_len];
        scatter_add_f32(s, idx, out);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Indexed copy with stride support
// ═══════════════════════════════════════════════════════════════════

/// Indexed copy (flat): `dst[i] = src[indices[i]]`.
///
/// Semantically identical to [`gather_f32`]; provided as a distinct
/// entry-point for clarity in call-sites that perform a copy.
pub fn indexed_copy_f32(src: &[f32], indices: &[usize], dst: &mut [f32]) {
    gather_f32(src, indices, dst);
}

/// Strided indexed copy.
///
/// For each index `k` in `indices`, copies `count` contiguous elements
/// from `src[indices[k] * src_stride ..]` to `dst[k * dst_stride ..]`.
///
/// # Panics
///
/// - `src_stride < count` or `dst_stride < count`
/// - Any index would read past `src` or write past `dst`
pub fn indexed_copy_strided(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
    indices: &[usize],
    count: usize,
) {
    assert!(src_stride >= count, "src_stride < count");
    assert!(dst_stride >= count, "dst_stride < count");
    if count == 0 || indices.is_empty() {
        return;
    }
    let max_src_rows = src.len() / src_stride;
    check_bounds(indices, max_src_rows);
    assert!(
        dst.len() >= indices.len() * dst_stride,
        "dst too small for {} rows × stride {dst_stride}",
        indices.len(),
    );

    for (out_row, &idx) in indices.iter().enumerate() {
        let s_off = idx * src_stride;
        let d_off = out_row * dst_stride;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let chunks = count / 4;
            let s_ptr = src.as_ptr();
            let d_ptr = dst.as_mut_ptr();
            for c in 0..chunks {
                let off = c * 4;
                unsafe {
                    let v = vld1q_f32(s_ptr.add(s_off + off));
                    vst1q_f32(d_ptr.add(d_off + off), v);
                }
            }
            let tail = chunks * 4;
            dst[d_off + tail..d_off + count].copy_from_slice(&src[s_off + tail..s_off + count]);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            dst[d_off..d_off + count].copy_from_slice(&src[s_off..s_off + count]);
        }
    }
}

/// Gather entire rows from a 2-D matrix (row-major).
///
/// `src` has `src.len() / cols` rows, each of width `cols`.
/// `output` must hold `indices.len() * cols` elements.
pub fn gather_rows(src: &[f32], cols: usize, indices: &[usize], output: &mut [f32]) {
    assert!(cols > 0, "gather_rows: cols must be > 0");
    assert_eq!(src.len() % cols, 0, "gather_rows: src.len() not divisible by cols");
    assert_eq!(output.len(), indices.len() * cols);
    indexed_copy_strided(src, cols, output, cols, indices, cols);
}

/// Scatter-add entire rows into a 2-D matrix.
///
/// For each `i`, `output[indices[i], :] += src[i, :]`.
pub fn scatter_rows_add(src: &[f32], cols: usize, indices: &[usize], output: &mut [f32]) {
    assert!(cols > 0, "scatter_rows_add: cols must be > 0");
    let num_src_rows = src.len() / cols;
    assert_eq!(src.len(), num_src_rows * cols);
    assert_eq!(indices.len(), num_src_rows);
    let num_dst_rows = output.len() / cols;
    assert_eq!(output.len(), num_dst_rows * cols);
    check_bounds(indices, num_dst_rows);

    for (src_row, &dst_idx) in indices.iter().enumerate() {
        let s_off = src_row * cols;
        let d_off = dst_idx * cols;

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let chunks = cols / 4;
            let s_ptr = src.as_ptr();
            let d_ptr = output.as_mut_ptr();
            for c in 0..chunks {
                let off = c * 4;
                unsafe {
                    let vs = vld1q_f32(s_ptr.add(s_off + off));
                    let vd = vld1q_f32(d_ptr.add(d_off + off));
                    vst1q_f32(d_ptr.add(d_off + off), vaddq_f32(vd, vs));
                }
            }
            for j in (chunks * 4)..cols {
                output[d_off + j] += src[s_off + j];
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            for j in 0..cols {
                output[d_off + j] += src[s_off + j];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Masked gather / scatter operations
// ═══════════════════════════════════════════════════════════════════

/// Masked gather: where `mask[i]` is true, `output[i] = src[indices[i]]`;
/// otherwise `output[i] = default_val`.
///
/// Only indices at masked-true positions must be in-bounds.
pub fn masked_gather(
    src: &[f32],
    indices: &[usize],
    mask: &[bool],
    output: &mut [f32],
    default_val: f32,
) {
    let n = indices.len();
    assert_eq!(n, mask.len(), "masked_gather: mask length mismatch");
    assert_eq!(n, output.len(), "masked_gather: output length mismatch");
    check_bounds_masked(indices, mask, src.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let o_ptr = output.as_mut_ptr();
        let s_ptr = src.as_ptr();
        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let g0 = if mask[base] { *s_ptr.add(indices[base]) } else { default_val };
                let g1 = if mask[base + 1] { *s_ptr.add(indices[base + 1]) } else { default_val };
                let g2 = if mask[base + 2] { *s_ptr.add(indices[base + 2]) } else { default_val };
                let g3 = if mask[base + 3] { *s_ptr.add(indices[base + 3]) } else { default_val };
                let m0: u32 = if mask[base] { !0 } else { 0 };
                let m1: u32 = if mask[base + 1] { !0 } else { 0 };
                let m2: u32 = if mask[base + 2] { !0 } else { 0 };
                let m3: u32 = if mask[base + 3] { !0 } else { 0 };
                let vmask = vld1q_u32([m0, m1, m2, m3].as_ptr());
                let vdefault = vdupq_n_f32(default_val);
                let vgathered = vsetq_lane_f32::<3>(
                    g3,
                    vsetq_lane_f32::<2>(g2, vsetq_lane_f32::<1>(g1, vdupq_n_f32(g0))),
                );
                let vresult = vbslq_f32(vmask, vgathered, vdefault);
                vst1q_f32(o_ptr.add(base), vresult);
            }
        }
        for i in (chunks * 4)..n {
            output[i] = if mask[i] { src[indices[i]] } else { default_val };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            output[i] = if mask[i] { src[indices[i]] } else { default_val };
        }
    }
}

/// Masked scatter (assign): `output[indices[i]] = src[i]` only where
/// `mask[i]` is true. Positions where mask is false are untouched.
pub fn masked_scatter(src: &[f32], indices: &[usize], mask: &[bool], output: &mut [f32]) {
    let n = src.len();
    assert_eq!(n, indices.len(), "masked_scatter: length mismatch");
    assert_eq!(n, mask.len(), "masked_scatter: mask length mismatch");
    check_bounds_masked(indices, mask, output.len());

    for i in 0..n {
        if mask[i] {
            output[indices[i]] = src[i];
        }
    }
}

/// Masked scatter-add: `output[indices[i]] += src[i]` only where
/// `mask[i]` is true.
pub fn masked_scatter_add(src: &[f32], indices: &[usize], mask: &[bool], output: &mut [f32]) {
    let n = src.len();
    assert_eq!(n, indices.len(), "masked_scatter_add: length mismatch");
    assert_eq!(n, mask.len(), "masked_scatter_add: mask length mismatch");
    check_bounds_masked(indices, mask, output.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let s_ptr = src.as_ptr();
        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let vs = vld1q_f32(s_ptr.add(base));
                let mut buf = [0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), vs);
                for lane in 0..4 {
                    if mask[base + lane] {
                        output[indices[base + lane]] += buf[lane];
                    }
                }
            }
        }
        for i in (chunks * 4)..n {
            if mask[i] {
                output[indices[i]] += src[i];
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            if mask[i] {
                output[indices[i]] += src[i];
            }
        }
    }
}

/// Masked fill: set `data[i] = value` wherever `mask[i]` is true.
///
/// Uses NEON `vbslq_f32` to blend without branching in the hot loop.
pub fn masked_fill(data: &mut [f32], mask: &[bool], value: f32) {
    assert_eq!(
        data.len(),
        mask.len(),
        "masked_fill: data.len() ({}) != mask.len() ({})",
        data.len(),
        mask.len(),
    );

    let n = data.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = n / 4;
        let d_ptr = data.as_mut_ptr();
        let vfill = unsafe { vdupq_n_f32(value) };
        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let m0: u32 = if mask[base] { !0 } else { 0 };
                let m1: u32 = if mask[base + 1] { !0 } else { 0 };
                let m2: u32 = if mask[base + 2] { !0 } else { 0 };
                let m3: u32 = if mask[base + 3] { !0 } else { 0 };
                let vmask = vld1q_u32([m0, m1, m2, m3].as_ptr());
                let vdata = vld1q_f32(d_ptr.add(base));
                let vresult = vbslq_f32(vmask, vfill, vdata);
                vst1q_f32(d_ptr.add(base), vresult);
            }
        }
        for i in (chunks * 4)..n {
            if mask[i] {
                data[i] = value;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            if mask[i] {
                data[i] = value;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Multi-dimensional index mapping
// ═══════════════════════════════════════════════════════════════════

/// Convert N-dimensional coordinates to a flat (row-major) index.
///
/// # Panics
///
/// - `coords.len() != shape.len()`
/// - Any coordinate ≥ its corresponding dimension size
pub fn ravel_multi_index(coords: &[usize], shape: &[usize]) -> usize {
    assert_eq!(
        coords.len(),
        shape.len(),
        "ravel_multi_index: coords.len() ({}) != shape.len() ({})",
        coords.len(),
        shape.len(),
    );
    for (&c, &s) in coords.iter().zip(shape.iter()) {
        assert!(c < s, "ravel_multi_index: coordinate {c} out of range for dimension {s}");
    }
    flat_offset(coords, shape)
}

/// Convert a flat index to N-dimensional coordinates (row-major).
///
/// # Panics
///
/// - `shape` is empty
/// - `flat_idx` ≥ total number of elements
pub fn unravel_index(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    assert!(!shape.is_empty(), "unravel_index: shape must not be empty");
    let numel = shape_numel(shape);
    assert!(flat_idx < numel, "unravel_index: index {flat_idx} >= numel {numel}");
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    for d in (0..ndim).rev() {
        coords[d] = flat_idx % shape[d];
        flat_idx /= shape[d];
    }
    coords
}

/// Batch-convert N-dimensional coordinate tuples to flat indices.
///
/// `coords_flat` contains `n * ndim` coordinates packed row-major:
/// `[c0_d0, c0_d1, …, c0_dN, c1_d0, …]`. Returns `n` flat indices.
pub fn multi_index_map(coords_flat: &[usize], shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    assert!(ndim > 0, "multi_index_map: shape must not be empty");
    assert_eq!(coords_flat.len() % ndim, 0, "multi_index_map: coords length not divisible by ndim");
    let n = coords_flat.len() / ndim;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let c = &coords_flat[i * ndim..(i + 1) * ndim];
        result.push(flat_offset(c, shape));
    }
    result
}

/// N-dimensional gather.
///
/// `src` is a flat buffer with logical shape `shape`.
/// `indices` are flat offsets; `output[i] = src[indices[i]]`.
pub fn gather_nd(src: &[f32], shape: &[usize], indices: &[usize], output: &mut [f32]) {
    let numel = shape_numel(shape);
    assert!(src.len() >= numel, "gather_nd: src shorter than shape implies");
    assert_eq!(indices.len(), output.len());
    check_bounds(indices, numel);
    gather_f32(src, indices, output);
}

/// N-dimensional scatter-add.
///
/// `src[i]` is added to `output[indices[i]]`.
/// `output` has logical shape `shape`.
pub fn scatter_nd_add(src: &[f32], indices: &[usize], shape: &[usize], output: &mut [f32]) {
    let numel = shape_numel(shape);
    assert!(output.len() >= numel, "scatter_nd_add: output shorter than shape implies");
    assert_eq!(src.len(), indices.len());
    check_bounds(indices, numel);
    scatter_add_f32(src, indices, output);
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── gather_f32 ─────────────────────────────────────────────────

    #[test]
    fn test_gather_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 2, 0, 3, 1];
        let mut out = [0.0; 5];
        gather_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![50.0, 30.0, 10.0, 40.0, 20.0]);
    }

    #[test]
    fn test_gather_identity() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..8).collect();
        let mut out = [0.0; 8];
        gather_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), src);
    }

    #[test]
    fn test_gather_reverse() {
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..8).rev().collect();
        let mut out = [0.0; 8];
        gather_f32(&src, &indices, &mut out);
        let expected: Vec<f32> = (0..8).rev().map(|i| i as f32).collect();
        assert_eq!(out.to_vec(), expected);
    }

    #[test]
    fn test_gather_tail_elements() {
        let src = [1.0, 2.0, 3.0];
        let indices = [2, 0, 1];
        let mut out = [0.0; 3];
        gather_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_single() {
        let src = [42.0, 99.0];
        let indices = [1];
        let mut out = [0.0; 1];
        gather_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![99.0]);
    }

    #[test]
    fn test_gather_empty() {
        let src = [1.0, 2.0];
        let indices: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        gather_f32(&src, &indices, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gather_duplicate_indices() {
        let src = [10.0, 20.0, 30.0];
        let indices = [1, 1, 1, 1];
        let mut out = [0.0; 4];
        gather_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_oob() {
        let src = [1.0, 2.0];
        let indices = [5];
        let mut out = [0.0; 1];
        gather_f32(&src, &indices, &mut out);
    }

    #[test]
    fn test_gather_large() {
        let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..100).rev().collect();
        let mut out = [0.0; 100];
        gather_f32(&src, &indices, &mut out);
        let expected: Vec<f32> = (0..100).rev().map(|i| i as f32).collect();
        assert_eq!(out.to_vec(), expected);
    }

    // ── gather_batched ─────────────────────────────────────────────

    #[test]
    fn test_gather_batched_basic() {
        // 2 batches × 4 elements each, gather 2 per batch
        let src = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let indices = [3, 1, 0, 2]; // batch0: [3,1], batch1: [0,2]
        let mut out = [0.0; 4];
        gather_batched(&src, 4, &indices, 2, &mut out);
        assert_eq!(out.to_vec(), vec![40.0, 20.0, 50.0, 70.0]);
    }

    #[test]
    fn test_gather_batched_single_batch() {
        let src = [1.0, 2.0, 3.0];
        let indices = [2, 0];
        let mut out = [0.0; 2];
        gather_batched(&src, 3, &indices, 2, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 1.0]);
    }

    #[test]
    fn test_gather_batched_empty() {
        let src = [1.0, 2.0];
        let indices: [usize; 0] = [];
        let mut out: Vec<f32> = vec![];
        gather_batched(&src, 2, &indices, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_batched_oob() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 5]; // index 5 out of segment_len=2
        let mut out = [0.0; 2];
        gather_batched(&src, 2, &indices, 1, &mut out);
    }

    // ── scatter_add_f32 ────────────────────────────────────────────

    #[test]
    fn test_scatter_add_basic() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = [0, 1, 2, 3, 4];
        let mut out = [10.0; 5];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_scatter_add_duplicate_indices() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 0, 1, 1];
        let mut out = [0.0; 2];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 7.0]);
    }

    #[test]
    fn test_scatter_add_tail() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 1, 0];
        let mut out = [0.0; 2];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 2.0]);
    }

    #[test]
    fn test_scatter_add_single() {
        let src = [7.0];
        let indices = [0];
        let mut out = [3.0];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![10.0]);
    }

    #[test]
    fn test_scatter_add_empty() {
        let src: [f32; 0] = [];
        let indices: [usize; 0] = [];
        let mut out = [5.0; 3];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![5.0; 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_add_oob() {
        let src = [1.0];
        let indices = [10];
        let mut out = [0.0; 2];
        scatter_add_f32(&src, &indices, &mut out);
    }

    #[test]
    fn test_scatter_add_all_same_index() {
        let src = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let indices = [0; 8];
        let mut out = [0.0; 1];
        scatter_add_f32(&src, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![8.0]);
    }

    // ── scatter_add_scaled ─────────────────────────────────────────

    #[test]
    fn test_scatter_add_scaled_basic() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 1, 2, 3];
        let mut out = [0.0; 4];
        scatter_add_scaled(&src, &indices, 2.0, &mut out);
        assert_eq!(out.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scatter_add_scaled_zero_alpha() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 1, 2, 3];
        let mut out = [10.0; 4];
        scatter_add_scaled(&src, &indices, 0.0, &mut out);
        assert_eq!(out.to_vec(), vec![10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_scatter_add_scaled_negative() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 1, 0];
        let mut out = [10.0; 2];
        scatter_add_scaled(&src, &indices, -1.0, &mut out);
        assert_eq!(out.to_vec(), vec![6.0, 8.0]); // 10-1-3, 10-2
    }

    #[test]
    fn test_scatter_add_scaled_tail() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = [0, 1, 2, 3, 4];
        let mut out = [0.0; 5];
        scatter_add_scaled(&src, &indices, 0.5, &mut out);
        assert_eq!(out.to_vec(), vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    // ── scatter_add_batched ────────────────────────────────────────

    #[test]
    fn test_scatter_add_batched_basic() {
        // 2 batches, 2 values each, output segments of 3
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 2, 1, 0];
        let mut out = [0.0; 6]; // 2 × 3
        scatter_add_batched(&src, &indices, 2, &mut out, 3);
        assert_eq!(out.to_vec(), vec![1.0, 0.0, 2.0, 4.0, 3.0, 0.0]);
    }

    #[test]
    fn test_scatter_add_batched_empty() {
        let src: [f32; 0] = [];
        let indices: [usize; 0] = [];
        let mut out = [0.0; 4];
        scatter_add_batched(&src, &indices, 0, &mut out, 4);
        assert_eq!(out.to_vec(), vec![0.0; 4]);
    }

    // ── indexed_copy_f32 ───────────────────────────────────────────

    #[test]
    fn test_indexed_copy_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 0, 2];
        let mut dst = [0.0; 3];
        indexed_copy_f32(&src, &indices, &mut dst);
        assert_eq!(dst.to_vec(), vec![50.0, 10.0, 30.0]);
    }

    #[test]
    fn test_indexed_copy_single() {
        let src = [42.0];
        let indices = [0];
        let mut dst = [0.0; 1];
        indexed_copy_f32(&src, &indices, &mut dst);
        assert_eq!(dst.to_vec(), vec![42.0]);
    }

    #[test]
    fn test_indexed_copy_empty() {
        let src = [1.0];
        let indices: [usize; 0] = [];
        let mut dst: Vec<f32> = vec![];
        indexed_copy_f32(&src, &indices, &mut dst);
        assert!(dst.is_empty());
    }

    // ── indexed_copy_strided ───────────────────────────────────────

    #[test]
    fn test_strided_copy_basic() {
        // 4 rows × stride 4, copy 3 elements per row
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let indices = [2, 0]; // rows 2 and 0
        let mut dst = [0.0; 8]; // 2 × stride 4
        indexed_copy_strided(&src, 4, &mut dst, 4, &indices, 3);
        // row 2: [8,9,10,...], row 0: [0,1,2,...]
        assert_eq!(dst[0..3], [8.0, 9.0, 10.0]);
        assert_eq!(dst[4..7], [0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_strided_copy_wide() {
        // NEON path: 8 elements per row (2 chunks)
        let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let indices = [1]; // row 1 of 3×8
        let mut dst = [0.0; 8];
        indexed_copy_strided(&src, 8, &mut dst, 8, &indices, 8);
        let expected: Vec<f32> = (8..16).map(|i| i as f32).collect();
        assert_eq!(dst.to_vec(), expected);
    }

    #[test]
    fn test_strided_copy_count_one() {
        let src = [10.0, 20.0, 30.0, 40.0];
        let indices = [3, 1];
        let mut dst = [0.0; 2];
        indexed_copy_strided(&src, 1, &mut dst, 1, &indices, 1);
        assert_eq!(dst.to_vec(), vec![40.0, 20.0]);
    }

    #[test]
    fn test_strided_copy_empty_indices() {
        let src = [1.0, 2.0];
        let indices: [usize; 0] = [];
        let mut dst: Vec<f32> = vec![];
        indexed_copy_strided(&src, 2, &mut dst, 2, &indices, 2);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_strided_copy_oob() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2 rows × 2
        let indices = [5]; // oob
        let mut dst = [0.0; 2];
        indexed_copy_strided(&src, 2, &mut dst, 2, &indices, 2);
    }

    // ── gather_rows ────────────────────────────────────────────────

    #[test]
    fn test_gather_rows_basic() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect(); // 4×3
        let indices = [3, 0];
        let mut out = [0.0; 6];
        gather_rows(&src, 3, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_rows_single() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [1];
        let mut out = [0.0; 2];
        gather_rows(&src, 2, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_gather_rows_wide() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect(); // 2×8
        let indices = [1, 0];
        let mut out = [0.0; 16];
        gather_rows(&src, 8, &indices, &mut out);
        let expected: Vec<f32> = (8..16).chain(0..8).map(|i| i as f32).collect();
        assert_eq!(out.to_vec(), expected);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_rows_oob() {
        let src = [0.0; 8]; // 2×4
        let indices = [5];
        let mut out = [0.0; 4];
        gather_rows(&src, 4, &indices, &mut out);
    }

    // ── scatter_rows_add ───────────────────────────────────────────

    #[test]
    fn test_scatter_rows_add_basic() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3
        let indices = [1, 0]; // row 0→dst 1, row 1→dst 0
        let mut out = [10.0; 6]; // 2×3
        scatter_rows_add(&src, 3, &indices, &mut out);
        // dst row 0: [10+4, 10+5, 10+6], dst row 1: [10+1, 10+2, 10+3]
        assert_eq!(out.to_vec(), vec![14.0, 15.0, 16.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_scatter_rows_add_duplicate() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2 rows × 2
        let indices = [0, 0]; // both scatter to row 0
        let mut out = [0.0; 2]; // 1×2
        scatter_rows_add(&src, 2, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 6.0]); // 1+3, 2+4
    }

    #[test]
    fn test_scatter_rows_add_wide() {
        // 8 cols → exercises NEON path
        let src: Vec<f32> = vec![1.0; 8]; // 1 row × 8
        let indices = [0];
        let mut out = [10.0; 8];
        scatter_rows_add(&src, 8, &indices, &mut out);
        assert_eq!(out.to_vec(), vec![11.0; 8]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_rows_add_oob() {
        let src = [1.0; 4]; // 2×2
        let indices = [5, 0];
        let mut out = [0.0; 4]; // 2×2
        scatter_rows_add(&src, 2, &indices, &mut out);
    }

    // ── masked_gather ──────────────────────────────────────────────

    #[test]
    fn test_masked_gather_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 3, 2, 1, 0];
        let mask = [true, false, true, false, true];
        let mut out = [0.0; 5];
        masked_gather(&src, &indices, &mask, &mut out, -1.0);
        assert_eq!(out.to_vec(), vec![50.0, -1.0, 30.0, -1.0, 10.0]);
    }

    #[test]
    fn test_masked_gather_all_true() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [3, 2, 1, 0];
        let mask = [true; 4];
        let mut out = [0.0; 4];
        masked_gather(&src, &indices, &mask, &mut out, -1.0);
        assert_eq!(out.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_masked_gather_all_false() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 0, 0, 0]; // indices don't matter
        let mask = [false; 4];
        let mut out = [0.0; 4];
        masked_gather(&src, &indices, &mask, &mut out, 99.0);
        assert_eq!(out.to_vec(), vec![99.0; 4]);
    }

    #[test]
    fn test_masked_gather_tail() {
        let src = [10.0, 20.0, 30.0];
        let indices = [2, 1, 0];
        let mask = [true, false, true];
        let mut out = [0.0; 3];
        masked_gather(&src, &indices, &mask, &mut out, -1.0);
        assert_eq!(out.to_vec(), vec![30.0, -1.0, 10.0]);
    }

    #[test]
    fn test_masked_gather_oob_masked_out() {
        // Out-of-bounds index at a masked-out position should not panic.
        let src = [1.0, 2.0];
        let indices = [999]; // oob, but mask is false
        let mask = [false];
        let mut out = [0.0; 1];
        masked_gather(&src, &indices, &mask, &mut out, -1.0);
        assert_eq!(out.to_vec(), vec![-1.0]);
    }

    // ── masked_scatter ─────────────────────────────────────────────

    #[test]
    fn test_masked_scatter_basic() {
        let src = [10.0, 20.0, 30.0];
        let indices = [2, 1, 0];
        let mask = [true, false, true];
        let mut out = [0.0; 3];
        masked_scatter(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![30.0, 0.0, 10.0]);
    }

    #[test]
    fn test_masked_scatter_all_true() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [3, 2, 1, 0];
        let mask = [true; 4];
        let mut out = [0.0; 4];
        masked_scatter(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_masked_scatter_all_false() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 1, 2];
        let mask = [false; 3];
        let mut out = [99.0; 3];
        masked_scatter(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![99.0; 3]); // unchanged
    }

    // ── masked_scatter_add ─────────────────────────────────────────

    #[test]
    fn test_masked_scatter_add_basic() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = [0, 1, 2, 0, 1];
        let mask = [true, false, true, true, false];
        let mut out = [10.0; 3];
        masked_scatter_add(&src, &indices, &mask, &mut out);
        // out[0] += 1 + 4 = 15, out[2] += 3 = 13
        assert_eq!(out.to_vec(), vec![15.0, 10.0, 13.0]);
    }

    #[test]
    fn test_masked_scatter_add_all_true() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 1, 0, 1];
        let mask = [true; 4];
        let mut out = [0.0; 2];
        masked_scatter_add(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_masked_scatter_add_all_false() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 1, 2, 3];
        let mask = [false; 4];
        let mut out = [10.0; 4];
        masked_scatter_add(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![10.0; 4]);
    }

    #[test]
    fn test_masked_scatter_add_tail() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 0, 0];
        let mask = [true, false, true];
        let mut out = [0.0; 1];
        masked_scatter_add(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![4.0]); // 1 + 3
    }

    #[test]
    fn test_masked_scatter_add_oob_masked_out() {
        let src = [1.0];
        let indices = [999]; // oob but masked out
        let mask = [false];
        let mut out = [5.0; 1];
        masked_scatter_add(&src, &indices, &mask, &mut out);
        assert_eq!(out.to_vec(), vec![5.0]);
    }

    // ── masked_fill ────────────────────────────────────────────────

    #[test]
    fn test_masked_fill_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [true, false, true, false, true];
        masked_fill(&mut data, &mask, -1.0);
        assert_eq!(data, vec![-1.0, 2.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn test_masked_fill_all_true() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        masked_fill(&mut data, &[true; 4], 0.0);
        assert_eq!(data, vec![0.0; 4]);
    }

    #[test]
    fn test_masked_fill_all_false() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let orig = data.clone();
        masked_fill(&mut data, &[false; 4], 99.0);
        assert_eq!(data, orig);
    }

    #[test]
    fn test_masked_fill_tail() {
        let mut data = vec![1.0, 2.0, 3.0];
        masked_fill(&mut data, &[false, true, false], 0.0);
        assert_eq!(data, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_masked_fill_neg_inf() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        masked_fill(&mut data, &[true; 4], f32::NEG_INFINITY);
        assert!(data.iter().all(|&v| v == f32::NEG_INFINITY));
    }

    // ── ravel_multi_index ──────────────────────────────────────────

    #[test]
    fn test_ravel_1d() {
        assert_eq!(ravel_multi_index(&[3], &[10]), 3);
    }

    #[test]
    fn test_ravel_2d() {
        // shape [3,4], coord [1,2] → 1*4+2 = 6
        assert_eq!(ravel_multi_index(&[1, 2], &[3, 4]), 6);
    }

    #[test]
    fn test_ravel_3d() {
        // shape [2,3,4], coord [1,2,3] → 1*12+2*4+3 = 23
        assert_eq!(ravel_multi_index(&[1, 2, 3], &[2, 3, 4]), 23);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_ravel_oob() {
        ravel_multi_index(&[5], &[3]);
    }

    // ── unravel_index ──────────────────────────────────────────────

    #[test]
    fn test_unravel_1d() {
        assert_eq!(unravel_index(3, &[10]), vec![3]);
    }

    #[test]
    fn test_unravel_2d() {
        assert_eq!(unravel_index(6, &[3, 4]), vec![1, 2]);
    }

    #[test]
    fn test_unravel_3d() {
        assert_eq!(unravel_index(23, &[2, 3, 4]), vec![1, 2, 3]);
    }

    #[test]
    fn test_ravel_unravel_roundtrip() {
        let shape = [5, 4, 3];
        for idx in 0..60 {
            let coords = unravel_index(idx, &shape);
            assert_eq!(ravel_multi_index(&coords, &shape), idx);
        }
    }

    #[test]
    #[should_panic(expected = "numel")]
    fn test_unravel_oob() {
        unravel_index(100, &[5, 4]);
    }

    // ── multi_index_map ────────────────────────────────────────────

    #[test]
    fn test_multi_index_map_2d() {
        let shape = [3, 4];
        let coords = [0, 0, 1, 2, 2, 3]; // 3 coord-pairs
        let flat = multi_index_map(&coords, &shape);
        assert_eq!(flat, vec![0, 6, 11]);
    }

    #[test]
    fn test_multi_index_map_single() {
        let flat = multi_index_map(&[2, 1], &[3, 4]);
        assert_eq!(flat, vec![9]);
    }

    #[test]
    fn test_multi_index_map_empty() {
        let flat = multi_index_map(&[], &[3, 4]);
        assert!(flat.is_empty());
    }

    // ── gather_nd ──────────────────────────────────────────────────

    #[test]
    fn test_gather_nd_2d() {
        // Logical shape [2,3], flat: [0,1,2,3,4,5]
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let indices = [5, 0, 3]; // flat indices
        let mut out = [0.0; 3];
        gather_nd(&src, &[2, 3], &indices, &mut out);
        assert_eq!(out.to_vec(), vec![5.0, 0.0, 3.0]);
    }

    #[test]
    fn test_gather_nd_3d() {
        let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shape = [2, 3, 4];
        let idx = ravel_multi_index(&[1, 2, 3], &shape); // = 23
        let mut out = [0.0; 1];
        gather_nd(&src, &shape, &[idx], &mut out);
        assert_eq!(out.to_vec(), vec![23.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_nd_oob() {
        let src = [0.0; 6];
        let mut out = [0.0; 1];
        gather_nd(&src, &[2, 3], &[6], &mut out); // 6 >= numel=6
    }

    // ── scatter_nd_add ─────────────────────────────────────────────

    #[test]
    fn test_scatter_nd_add_2d() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 3, 5];
        let mut out = [10.0; 6];
        scatter_nd_add(&src, &indices, &[2, 3], &mut out);
        assert_eq!(out.to_vec(), vec![11.0, 10.0, 10.0, 12.0, 10.0, 13.0]);
    }

    #[test]
    fn test_scatter_nd_add_duplicate() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 0, 0];
        let mut out = [0.0; 4];
        scatter_nd_add(&src, &indices, &[4], &mut out);
        assert_eq!(out[0], 6.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scatter_nd_add_oob() {
        let src = [1.0];
        let indices = [10];
        let mut out = [0.0; 6];
        scatter_nd_add(&src, &indices, &[2, 3], &mut out);
    }

    // ── NEON-specific tests (aarch64 only) ─────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod neon_specific {
        use super::super::*;

        #[test]
        fn test_gather_exact_chunk() {
            // Exactly 4 elements → pure NEON, no scalar tail.
            let src = [10.0, 20.0, 30.0, 40.0];
            let indices = [3, 2, 1, 0];
            let mut out = [0.0; 4];
            gather_f32(&src, &indices, &mut out);
            assert_eq!(out.to_vec(), vec![40.0, 30.0, 20.0, 10.0]);
        }

        #[test]
        fn test_gather_two_chunks() {
            let src: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let indices = [9, 8, 7, 6, 5, 4, 3, 2];
            let mut out = [0.0; 8];
            gather_f32(&src, &indices, &mut out);
            let expected: Vec<f32> = (2..10).rev().map(|i| i as f32).collect();
            assert_eq!(out.to_vec(), expected);
        }

        #[test]
        fn test_scatter_add_exact_chunk() {
            let src = [1.0, 2.0, 3.0, 4.0];
            let indices = [0, 1, 2, 3];
            let mut out = [0.0; 4];
            scatter_add_f32(&src, &indices, &mut out);
            assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_masked_fill_exact_chunk() {
            let mut data = vec![1.0, 2.0, 3.0, 4.0];
            masked_fill(&mut data, &[true, false, true, false], 0.0);
            assert_eq!(data, vec![0.0, 2.0, 0.0, 4.0]);
        }

        #[test]
        fn test_scatter_add_scaled_exact_chunk() {
            let src = [2.0, 4.0, 6.0, 8.0];
            let indices = [0, 1, 2, 3];
            let mut out = [0.0; 4];
            scatter_add_scaled(&src, &indices, 0.5, &mut out);
            assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_strided_copy_neon_path() {
            // 8 cols → 2 NEON chunks per row
            let src: Vec<f32> = (0..32).map(|i| i as f32).collect();
            let indices = [3, 1];
            let mut dst = [0.0; 16];
            indexed_copy_strided(&src, 8, &mut dst, 8, &indices, 8);
            let expected: Vec<f32> = (24..32).chain(8..16).map(|i| i as f32).collect();
            assert_eq!(dst.to_vec(), expected);
        }

        #[test]
        fn test_scatter_rows_add_neon_path() {
            let src: Vec<f32> = [2.0; 8]; // 1×8
            let indices = [0];
            let mut out = [1.0; 8]; // 1×8
            scatter_rows_add(&src, 8, &indices, &mut out);
            assert_eq!(out.to_vec(), vec![3.0; 8]);
        }

        #[test]
        fn test_masked_gather_exact_chunk() {
            let src = [10.0, 20.0, 30.0, 40.0];
            let indices = [0, 1, 2, 3];
            let mask = [true, false, true, false];
            let mut out = [0.0; 4];
            masked_gather(&src, &indices, &mask, &mut out, -1.0);
            assert_eq!(out.to_vec(), vec![10.0, -1.0, 30.0, -1.0]);
        }

        #[test]
        fn test_masked_scatter_add_neon_path() {
            let src = [1.0, 2.0, 3.0, 4.0];
            let indices = [0, 0, 0, 0];
            let mask = [true, false, true, false];
            let mut out = [0.0; 1];
            masked_scatter_add(&src, &indices, &mask, &mut out);
            assert_eq!(out.to_vec(), vec![4.0]); // 1 + 3
        }

        #[test]
        fn test_gather_large_multi_chunk() {
            let src: Vec<f32> = (0..256).map(|i| i as f32).collect();
            let indices: Vec<usize> = (0..256).rev().collect();
            let mut out = [0.0; 256];
            gather_f32(&src, &indices, &mut out);
            for i in 0..256 {
                assert_eq!(out[i], (255 - i) as f32);
            }
        }
    }
}
