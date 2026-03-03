//! ARM NEON-optimized tensor reshape and view operations for Apple Silicon.
//!
//! Provides fast contiguous reshape, strided-to-contiguous copy, squeeze,
//! unsqueeze, broadcast expand, and dimension permute — all using NEON
//! `float32x4_t` intrinsics with scalar fallback for tail elements.

use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

// ── Shape helpers ──────────────────────────────────────────────────────

/// Product of all elements in a shape slice.
#[inline]
fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Compute default (C-contiguous / row-major) strides for a shape.
#[inline]
fn default_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Check whether given strides represent a C-contiguous layout for `shape`.
#[inline]
fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let expected = default_strides(shape);
    strides == expected.as_slice()
}

// ── Contiguous Reshape ─────────────────────────────────────────────────

/// Reshape a contiguous tensor to a new shape with the same element count.
///
/// Uses NEON 128-bit loads/stores for the bulk copy, falling back to scalar
/// for the tail (0–3 elements). This is semantically a `memcpy` + metadata
/// change, but the NEON path keeps the copy inside SIMD registers which
/// benefits from the ARM store-buffer pipeline.
///
/// # Panics
///
/// * `new_shape` element count differs from `data.len()`.
/// * `output.len() < data.len()`.
pub fn reshape_contiguous_neon(
    data: &[f32],
    _old_shape: &[usize],
    new_shape: &[usize],
    output: &mut [f32],
) {
    let n = data.len();
    let new_n = numel(new_shape);
    assert_eq!(n, new_n, "reshape_contiguous_neon: element count mismatch ({n} vs {new_n})");
    assert!(
        output.len() >= n,
        "reshape_contiguous_neon: output too short ({} < {n})",
        output.len()
    );

    let chunks = n / LANES;
    let src = data.as_ptr();
    let dst = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let v = vld1q_f32(src.add(off));
            vst1q_f32(dst.add(off), v);
        }
    }
    let tail_start = chunks * LANES;
    output[tail_start..n].copy_from_slice(&data[tail_start..n]);
}

/// Copy a non-contiguous (strided) tensor into a contiguous buffer with a
/// new shape.
///
/// Walks the source tensor via its `old_strides` and writes elements in
/// row-major order to `output`. When four consecutive source elements are
/// contiguous (stride-1 in the innermost dimension) a NEON vector
/// load/store is used; otherwise elements are gathered one by one.
///
/// # Panics
///
/// * `old_shape` and `old_strides` length mismatch.
/// * `new_shape` element count differs from `old_shape` element count.
/// * `output` too short.
pub fn reshape_strided_neon(
    data: &[f32],
    old_shape: &[usize],
    old_strides: &[usize],
    new_shape: &[usize],
    output: &mut [f32],
) {
    let ndim = old_shape.len();
    assert_eq!(ndim, old_strides.len(), "reshape_strided_neon: shape/strides ndim mismatch");
    let n = numel(old_shape);
    let new_n = numel(new_shape);
    assert_eq!(n, new_n, "reshape_strided_neon: element count mismatch ({n} vs {new_n})");
    assert!(output.len() >= n, "reshape_strided_neon: output too short ({} < {n})", output.len());

    if is_contiguous(old_shape, old_strides) {
        // Fast path: already contiguous — delegate to NEON memcpy.
        reshape_contiguous_neon(data, old_shape, new_shape, output);
        return;
    }

    // General strided gather into contiguous output.
    let mut coord = vec![0usize; ndim];
    let src = data.as_ptr();
    let dst = output.as_mut_ptr();
    let innermost_stride = if ndim > 0 { old_strides[ndim - 1] } else { 1 };

    let mut out_idx = 0usize;
    while out_idx < n {
        // Compute flat source offset for current coord.
        let src_off: usize = coord.iter().zip(old_strides.iter()).map(|(&c, &s)| c * s).sum();

        // If the innermost dimension has stride 1, try a NEON bulk copy
        // for as many contiguous elements as remain in that dimension.
        let inner_remaining = old_shape[ndim - 1] - coord[ndim - 1];
        if innermost_stride == 1 && inner_remaining >= LANES {
            let vec_count = inner_remaining / LANES;
            for v in 0..vec_count {
                let s = src_off + v * LANES;
                let d = out_idx + v * LANES;
                unsafe {
                    let val = vld1q_f32(src.add(s));
                    vst1q_f32(dst.add(d), val);
                }
            }
            let copied = vec_count * LANES;
            out_idx += copied;
            coord[ndim - 1] += copied;
        } else {
            // Scalar gather for non-unit stride or short tail.
            output[out_idx] = data[src_off];
            out_idx += 1;
            coord[ndim - 1] += 1;
        }

        // Carry: propagate coordinate overflow from inner to outer dims.
        let mut d = ndim - 1;
        while d > 0 && coord[d] >= old_shape[d] {
            coord[d] = 0;
            coord[d - 1] += 1;
            d -= 1;
        }
        if coord[0] >= old_shape[0] {
            break;
        }
    }
}

// ── Squeeze ────────────────────────────────────────────────────────────

/// Remove all size-1 dimensions from `shape`, returning the new shape and
/// the corresponding strides. Data is unchanged (squeeze is a view-only op)
/// but we still provide a NEON-accelerated copy if the caller supplies an
/// output buffer.
///
/// Returns `(new_shape, new_strides)`.
///
/// # Panics
///
/// * `shape.len() != strides.len()`
pub fn squeeze_neon(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    output: Option<&mut [f32]>,
) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(shape.len(), strides.len(), "squeeze_neon: shape/strides ndim mismatch");

    let new_shape: Vec<usize> = shape.iter().copied().filter(|&s| s != 1).collect();
    let new_strides: Vec<usize> =
        shape.iter().zip(strides.iter()).filter(|(s, _)| **s != 1).map(|(_, st)| *st).collect();

    // If all dims are 1 we keep a scalar shape.
    let (new_shape, new_strides) =
        if new_shape.is_empty() { (vec![1], vec![1]) } else { (new_shape, new_strides) };

    if let Some(out) = output {
        let n = data.len();
        assert!(out.len() >= n, "squeeze_neon: output too short ({} < {n})", out.len());
        neon_copy(data, out);
    }

    (new_shape, new_strides)
}

// ── Unsqueeze ──────────────────────────────────────────────────────────

/// Insert a size-1 dimension at position `dim`, returning the new shape and
/// strides. As with squeeze this is a view-only metadata change; the
/// optional output buffer receives a NEON-accelerated copy.
///
/// # Panics
///
/// * `dim > shape.len()`
/// * `shape.len() != strides.len()`
pub fn unsqueeze_neon(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
    output: Option<&mut [f32]>,
) -> (Vec<usize>, Vec<usize>) {
    let ndim = shape.len();
    assert_eq!(ndim, strides.len(), "unsqueeze_neon: ndim mismatch");
    assert!(dim <= ndim, "unsqueeze_neon: dim {dim} out of range for ndim {ndim}");

    let mut new_shape = Vec::with_capacity(ndim + 1);
    let mut new_strides = Vec::with_capacity(ndim + 1);

    // Stride for the new size-1 dim: product of subsequent dims' sizes
    // times their stride, or 1 if appended at the end.
    let stride_for_new = if dim < ndim { shape[dim] * strides[dim] } else { 1 };

    for i in 0..=ndim {
        if i == dim {
            new_shape.push(1);
            new_strides.push(stride_for_new);
        }
        if i < ndim {
            new_shape.push(shape[i]);
            new_strides.push(strides[i]);
        }
    }

    if let Some(out) = output {
        let n = data.len();
        assert!(out.len() >= n, "unsqueeze_neon: output too short ({} < {n})", out.len());
        neon_copy(data, out);
    }

    (new_shape, new_strides)
}

// ── Broadcast Expand ───────────────────────────────────────────────────

/// Expand (broadcast) a tensor along size-1 dimensions to match
/// `target_shape`.
///
/// For each dimension that is 1 in the source and > 1 in the target, the
/// single value is replicated. NEON `vdupq_n_f32` is used when the
/// broadcast runs along the innermost dimension, giving 4-wide replication
/// per store.
///
/// # Panics
///
/// * `src_shape.len() != target_shape.len()`
/// * A source dimension is neither 1 nor equal to the target dimension.
/// * `output` too short.
pub fn broadcast_expand_neon(
    data: &[f32],
    src_shape: &[usize],
    target_shape: &[usize],
    output: &mut [f32],
) {
    let ndim = src_shape.len();
    assert_eq!(ndim, target_shape.len(), "broadcast_expand_neon: ndim mismatch");
    for d in 0..ndim {
        assert!(
            src_shape[d] == 1 || src_shape[d] == target_shape[d],
            "broadcast_expand_neon: dim {d} src={} target={} incompatible",
            src_shape[d],
            target_shape[d],
        );
    }

    let target_n = numel(target_shape);
    assert!(
        output.len() >= target_n,
        "broadcast_expand_neon: output too short ({} < {target_n})",
        output.len()
    );

    let src_strides = default_strides(src_shape);
    let tgt_strides = default_strides(target_shape);

    let mut coord = vec![0usize; ndim];
    let mut out_idx = 0usize;

    while out_idx < target_n {
        // Compute source flat index (clamped by broadcast dims).
        let src_off: usize = coord
            .iter()
            .enumerate()
            .map(|(d, &c)| if src_shape[d] == 1 { 0 } else { c * src_strides[d] })
            .sum();

        // Check if innermost dim is a broadcast (src=1, tgt>1).
        let inner_remaining = target_shape[ndim - 1] - coord[ndim - 1];
        if ndim > 0 && src_shape[ndim - 1] == 1 && inner_remaining >= LANES {
            // Broadcast single value across a NEON register.
            let val = data[src_off];
            let chunks = inner_remaining / LANES;
            let dst = output.as_mut_ptr();
            unsafe {
                let v = vdupq_n_f32(val);
                for c in 0..chunks {
                    vst1q_f32(dst.add(out_idx + c * LANES), v);
                }
            }
            let written = chunks * LANES;
            out_idx += written;
            coord[ndim - 1] += written;
        } else if ndim > 0
            && src_shape[ndim - 1] == target_shape[ndim - 1]
            && inner_remaining >= LANES
        {
            // Non-broadcast innermost: NEON copy from source.
            let chunks = inner_remaining / LANES;
            let s = data.as_ptr();
            let d = output.as_mut_ptr();
            for c in 0..chunks {
                unsafe {
                    let v = vld1q_f32(s.add(src_off + c * LANES));
                    vst1q_f32(d.add(out_idx + c * LANES), v);
                }
            }
            let written = chunks * LANES;
            out_idx += written;
            coord[ndim - 1] += written;
        } else {
            output[out_idx] = data[src_off];
            out_idx += 1;
            coord[ndim - 1] += 1;
        }

        // Carry from inner to outer.
        if ndim > 0 {
            let mut d = ndim - 1;
            while d > 0 && coord[d] >= target_shape[d] {
                coord[d] = 0;
                coord[d - 1] += 1;
                d -= 1;
            }
            if coord[0] >= target_shape[0] {
                break;
            }
        }
    }

    // Tail (scalar) for the final partial LANES chunk of innermost.
    // Already handled element-by-element in the loop above via the else branch.
    let _ = tgt_strides; // suppress unused warning
}

// ── Permute Dimensions ─────────────────────────────────────────────────

/// Reorder tensor dimensions according to `perm`.
///
/// `perm` must be a valid permutation of `0..ndim`. The output is written
/// in row-major order of the permuted shape, with NEON bulk copy when the
/// innermost source stride is 1 and the run length is ≥ 4.
///
/// # Panics
///
/// * `perm.len() != shape.len()`
/// * `perm` is not a permutation of `0..ndim`.
/// * `output` too short.
pub fn permute_dims_neon(data: &[f32], shape: &[usize], perm: &[usize], output: &mut [f32]) {
    let ndim = shape.len();
    assert_eq!(perm.len(), ndim, "permute_dims_neon: perm.len() ({}) != ndim ({ndim})", perm.len());

    // Validate permutation.
    let mut seen = vec![false; ndim];
    for &p in perm {
        assert!(p < ndim, "permute_dims_neon: perm value {p} out of range for ndim {ndim}");
        assert!(!seen[p], "permute_dims_neon: duplicate perm value {p}");
        seen[p] = true;
    }

    let n = numel(shape);
    assert!(output.len() >= n, "permute_dims_neon: output too short ({} < {n})", output.len());

    let src_strides = default_strides(shape);

    // Build permuted shape and the strides into the *source* buffer when
    // walking in permuted order.
    let perm_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let perm_src_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();

    // Walk the permuted coordinate space.
    let mut coord = vec![0usize; ndim];
    let src = data.as_ptr();
    let dst = output.as_mut_ptr();

    let innermost_src_stride = if ndim > 0 { perm_src_strides[ndim - 1] } else { 1 };

    let mut out_idx = 0usize;
    while out_idx < n {
        let src_off: usize = coord.iter().zip(perm_src_strides.iter()).map(|(&c, &s)| c * s).sum();

        let inner_remaining = if ndim > 0 { perm_shape[ndim - 1] - coord[ndim - 1] } else { 0 };

        if innermost_src_stride == 1 && inner_remaining >= LANES {
            let chunks = inner_remaining / LANES;
            for c in 0..chunks {
                unsafe {
                    let v = vld1q_f32(src.add(src_off + c * LANES));
                    vst1q_f32(dst.add(out_idx + c * LANES), v);
                }
            }
            let written = chunks * LANES;
            out_idx += written;
            coord[ndim - 1] += written;
        } else {
            output[out_idx] = data[src_off];
            out_idx += 1;
            if ndim > 0 {
                coord[ndim - 1] += 1;
            }
        }

        // Carry.
        if ndim > 0 {
            let mut d = ndim - 1;
            while d > 0 && coord[d] >= perm_shape[d] {
                coord[d] = 0;
                coord[d - 1] += 1;
                d -= 1;
            }
            if coord[0] >= perm_shape[0] {
                break;
            }
        } else {
            break;
        }
    }
}

// ── Internal helper ────────────────────────────────────────────────────

/// NEON-accelerated `memcpy` for `f32` slices.
#[inline]
fn neon_copy(src: &[f32], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let chunks = n / LANES;
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        unsafe {
            let v = vld1q_f32(s.add(off));
            vst1q_f32(d.add(off), v);
        }
    }
    let tail_start = chunks * LANES;
    dst[tail_start..n].copy_from_slice(&src[tail_start..n]);
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── reshape_contiguous_neon ────────────────────────────────────────

    #[test]
    fn test_reshape_contiguous_1d_to_2d() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 12];
        reshape_contiguous_neon(&data, &[12], &[3, 4], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_2d_to_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 24];
        reshape_contiguous_neon(&data, &[4, 6], &[2, 3, 4], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_3d_to_1d() {
        let data: Vec<f32> = (0..60).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 60];
        reshape_contiguous_neon(&data, &[3, 4, 5], &[60], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_identity() {
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 16];
        reshape_contiguous_neon(&data, &[4, 4], &[4, 4], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_single_element() {
        let data = vec![42.0f32];
        let mut out = vec![0.0f32; 1];
        reshape_contiguous_neon(&data, &[1], &[1, 1, 1], &mut out);
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_reshape_contiguous_tail_elements() {
        // 5 elements: 1 NEON chunk of 4 + 1 scalar tail.
        let data: Vec<f32> = (0..5).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 5];
        reshape_contiguous_neon(&data, &[5], &[1, 5], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_exact_neon_multiple() {
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 16];
        reshape_contiguous_neon(&data, &[16], &[2, 8], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_contiguous_large() {
        let n = 1024;
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; n as usize];
        reshape_contiguous_neon(&data, &[n as usize], &[32, 32], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    #[should_panic(expected = "element count mismatch")]
    fn test_reshape_contiguous_size_mismatch() {
        let data = vec![1.0f32; 12];
        let mut out = vec![0.0f32; 12];
        reshape_contiguous_neon(&data, &[12], &[3, 5], &mut out);
    }

    #[test]
    #[should_panic(expected = "output too short")]
    fn test_reshape_contiguous_output_too_short() {
        let data = vec![1.0f32; 12];
        let mut out = vec![0.0f32; 6];
        reshape_contiguous_neon(&data, &[12], &[3, 4], &mut out);
    }

    #[test]
    fn test_reshape_contiguous_oversized_output() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![99.0f32; 16];
        reshape_contiguous_neon(&data, &[8], &[2, 4], &mut out);
        assert_eq!(&out[..8], &data[..]);
        // Trailing elements untouched.
        assert_eq!(&out[8..], &[99.0; 8]);
    }

    // ── reshape_strided_neon ──────────────────────────────────────────

    #[test]
    fn test_strided_contiguous_fast_path() {
        // Contiguous strides → should take the fast path.
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 12];
        reshape_strided_neon(&data, &[3, 4], &[4, 1], &[12], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_strided_transpose_2d() {
        // 3×4 row-major, read with transposed strides → 4×3 column-major read.
        // data[r][c] = r*4+c
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        // Read with strides [1, 3] = column-major of a 4×3 view.
        let mut out = vec![0.0f32; 12];
        reshape_strided_neon(&data, &[4, 3], &[1, 4], &[12], &mut out);
        // Expected: column-major traversal of 3×4 matrix.
        let expected: Vec<f32> = vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_strided_skip_rows() {
        // Stride 2 in innermost dim → gather every other element.
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 4];
        reshape_strided_neon(&data, &[2, 2], &[4, 2], &[4], &mut out);
        assert_eq!(out, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_strided_3d() {
        // 2×2×2 with strides [4,2,1] (contiguous) → should fast-path.
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 8];
        reshape_strided_neon(&data, &[2, 2, 2], &[4, 2, 1], &[8], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_strided_inner_non_unit() {
        // 2×3 with inner stride=2, outer stride=1 (a peculiar view).
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 6];
        reshape_strided_neon(&data, &[2, 3], &[1, 2], &[6], &mut out);
        // Reads: (0,0)→0, (0,1)→2, (0,2)→4, (1,0)→1, (1,1)→3, (1,2)→5
        assert_eq!(out, vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_strided_single_element() {
        let data = vec![7.0f32];
        let mut out = vec![0.0f32; 1];
        reshape_strided_neon(&data, &[1], &[1], &[1, 1], &mut out);
        assert_eq!(out, vec![7.0]);
    }

    #[test]
    fn test_strided_large_contiguous() {
        let n = 256;
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; n as usize];
        reshape_strided_neon(&data, &[16, 16], &[16, 1], &[n as usize], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    #[should_panic(expected = "element count mismatch")]
    fn test_strided_count_mismatch() {
        let data = vec![0.0f32; 12];
        let mut out = vec![0.0f32; 6];
        reshape_strided_neon(&data, &[3, 4], &[4, 1], &[3, 3], &mut out);
    }

    #[test]
    #[should_panic(expected = "shape/strides ndim mismatch")]
    fn test_strided_ndim_mismatch() {
        let data = vec![0.0f32; 6];
        let mut out = vec![0.0f32; 6];
        reshape_strided_neon(&data, &[2, 3], &[3], &[6], &mut out);
    }

    // ── squeeze_neon ──────────────────────────────────────────────────

    #[test]
    fn test_squeeze_removes_ones() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (s, st) = squeeze_neon(&data, &[1, 2, 1, 3], &[6, 3, 3, 1], None);
        assert_eq!(s, vec![2, 3]);
        assert_eq!(st, vec![3, 1]);
    }

    #[test]
    fn test_squeeze_no_ones() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let (s, st) = squeeze_neon(&data, &[3, 4], &[4, 1], None);
        assert_eq!(s, vec![3, 4]);
        assert_eq!(st, vec![4, 1]);
    }

    #[test]
    fn test_squeeze_all_ones() {
        let data = vec![5.0f32];
        let (s, st) = squeeze_neon(&data, &[1, 1, 1], &[1, 1, 1], None);
        assert_eq!(s, vec![1]);
        assert_eq!(st, vec![1]);
    }

    #[test]
    fn test_squeeze_with_copy() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 8];
        let (s, _st) = squeeze_neon(&data, &[1, 8], &[8, 1], Some(&mut out));
        assert_eq!(s, vec![8]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_squeeze_leading_ones() {
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let (s, st) = squeeze_neon(&data, &[1, 1, 4], &[4, 4, 1], None);
        assert_eq!(s, vec![4]);
        assert_eq!(st, vec![1]);
    }

    #[test]
    fn test_squeeze_trailing_ones() {
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let (s, st) = squeeze_neon(&data, &[4, 1, 1], &[1, 1, 1], None);
        assert_eq!(s, vec![4]);
        assert_eq!(st, vec![1]);
    }

    #[test]
    fn test_squeeze_interleaved_ones() {
        let data = vec![0.0f32; 24];
        let (s, st) = squeeze_neon(&data, &[2, 1, 3, 1, 4], &[12, 12, 4, 4, 1], None);
        assert_eq!(s, vec![2, 3, 4]);
        assert_eq!(st, vec![12, 4, 1]);
    }

    #[test]
    #[should_panic(expected = "ndim mismatch")]
    fn test_squeeze_ndim_mismatch() {
        let data = vec![0.0f32; 6];
        squeeze_neon(&data, &[2, 3], &[3, 1, 1], None);
    }

    #[test]
    fn test_squeeze_neon_copy_tail() {
        // 5 elements: ensures the tail scalar copy in neon_copy works.
        let data: Vec<f32> = (0..5).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 5];
        let _ = squeeze_neon(&data, &[1, 5], &[5, 1], Some(&mut out));
        assert_eq!(out, data);
    }

    // ── unsqueeze_neon ────────────────────────────────────────────────

    #[test]
    fn test_unsqueeze_dim_0() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (s, st) = unsqueeze_neon(&data, &[2, 3], &[3, 1], 0, None);
        assert_eq!(s, vec![1, 2, 3]);
        assert_eq!(st, vec![6, 3, 1]);
    }

    #[test]
    fn test_unsqueeze_dim_1() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (s, st) = unsqueeze_neon(&data, &[2, 3], &[3, 1], 1, None);
        assert_eq!(s, vec![2, 1, 3]);
        assert_eq!(st, vec![3, 3, 1]);
    }

    #[test]
    fn test_unsqueeze_dim_last() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (s, st) = unsqueeze_neon(&data, &[2, 3], &[3, 1], 2, None);
        assert_eq!(s, vec![2, 3, 1]);
        assert_eq!(st, vec![3, 1, 1]);
    }

    #[test]
    fn test_unsqueeze_scalar() {
        let data = vec![42.0f32];
        let (s, st) = unsqueeze_neon(&data, &[1], &[1], 0, None);
        assert_eq!(s, vec![1, 1]);
        assert_eq!(st, vec![1, 1]);
    }

    #[test]
    fn test_unsqueeze_with_copy() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 8];
        let (s, _) = unsqueeze_neon(&data, &[8], &[1], 0, Some(&mut out));
        assert_eq!(s, vec![1, 8]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_unsqueeze_roundtrip() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let (us_shape, us_strides) = unsqueeze_neon(&data, &[3, 4], &[4, 1], 1, None);
        assert_eq!(us_shape, vec![3, 1, 4]);
        let (sq_shape, sq_strides) = squeeze_neon(&data, &us_shape, &us_strides, None);
        assert_eq!(sq_shape, vec![3, 4]);
        assert_eq!(sq_strides, vec![4, 1]);
    }

    #[test]
    #[should_panic(expected = "dim 3 out of range")]
    fn test_unsqueeze_out_of_range() {
        let data = vec![0.0f32; 6];
        unsqueeze_neon(&data, &[2, 3], &[3, 1], 3, None);
    }

    #[test]
    fn test_unsqueeze_multiple() {
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let (s1, st1) = unsqueeze_neon(&data, &[4], &[1], 0, None);
        assert_eq!(s1, vec![1, 4]);
        let (s2, st2) = unsqueeze_neon(&data, &s1, &st1, 2, None);
        assert_eq!(s2, vec![1, 4, 1]);
        assert_eq!(st2[0], s1[1] * st1[1]); // outer stride
    }

    // ── broadcast_expand_neon ─────────────────────────────────────────

    #[test]
    fn test_broadcast_scalar_to_vector() {
        let data = vec![3.0f32];
        let mut out = vec![0.0f32; 8];
        broadcast_expand_neon(&data, &[1], &[8], &mut out);
        assert_eq!(out, vec![3.0; 8]);
    }

    #[test]
    fn test_broadcast_row_to_matrix() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 12];
        broadcast_expand_neon(&data, &[1, 4], &[3, 4], &mut out);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_broadcast_col_to_matrix() {
        let data = vec![10.0f32, 20.0, 30.0];
        let mut out = vec![0.0f32; 12];
        broadcast_expand_neon(&data, &[3, 1], &[3, 4], &mut out);
        let expected = vec![10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_broadcast_no_expansion() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 6];
        broadcast_expand_neon(&data, &[2, 3], &[2, 3], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_broadcast_3d() {
        // (1,1,4) → (2,3,4)
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 24];
        broadcast_expand_neon(&data, &[1, 1, 4], &[2, 3, 4], &mut out);
        for chunk in out.chunks(4) {
            assert_eq!(chunk, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_broadcast_inner_dim_only() {
        // (2,1) → (2,4): broadcast along dim-1 only.
        let data = vec![5.0f32, 7.0];
        let mut out = vec![0.0f32; 8];
        broadcast_expand_neon(&data, &[2, 1], &[2, 4], &mut out);
        assert_eq!(out, vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_broadcast_outer_dim_only() {
        // (1,4) → (3,4): broadcast along dim-0.
        let data: Vec<f32> = (1..=4).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 12];
        broadcast_expand_neon(&data, &[1, 4], &[3, 4], &mut out);
        for chunk in out.chunks(4) {
            assert_eq!(chunk, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_broadcast_non_aligned_tail() {
        // (1,5) → (2,5): inner dim not a multiple of 4.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 10];
        broadcast_expand_neon(&data, &[1, 5], &[2, 5], &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "ndim mismatch")]
    fn test_broadcast_ndim_mismatch() {
        let data = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 12];
        broadcast_expand_neon(&data, &[4], &[3, 4], &mut out);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn test_broadcast_incompatible_dim() {
        let data = vec![1.0f32; 6];
        let mut out = vec![0.0f32; 8];
        broadcast_expand_neon(&data, &[2, 3], &[2, 4], &mut out);
    }

    #[test]
    fn test_broadcast_scalar_to_3d() {
        let data = vec![1.0f32];
        let mut out = vec![0.0f32; 24];
        broadcast_expand_neon(&data, &[1, 1, 1], &[2, 3, 4], &mut out);
        assert_eq!(out, vec![1.0; 24]);
    }

    #[test]
    fn test_broadcast_large_inner() {
        // (1,64) → (4,64): broadcast 64-element row.
        let data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 256];
        broadcast_expand_neon(&data, &[1, 64], &[4, 64], &mut out);
        for row in 0..4 {
            let start = row * 64;
            assert_eq!(&out[start..start + 64], &data[..]);
        }
    }

    // ── permute_dims_neon ─────────────────────────────────────────────

    #[test]
    fn test_permute_identity() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[0, 1], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_permute_transpose_2d() {
        // 3×4 → 4×3
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[1, 0], &mut out);
        // Expected column-major read of 3×4.
        let expected = vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_permute_3d_021() {
        // (2,3,4) → (2,4,3): swap last two dims.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 24];
        permute_dims_neon(&data, &[2, 3, 4], &[0, 2, 1], &mut out);

        // Verify by computing expected output element by element.
        let mut expected = vec![0.0f32; 24];
        for i in 0..2 {
            for j in 0..4 {
                for k in 0..3 {
                    let src_idx = i * 12 + k * 4 + j;
                    let dst_idx = i * 12 + j * 3 + k;
                    expected[dst_idx] = data[src_idx];
                }
            }
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn test_permute_3d_120() {
        // (2,3,4) → (4,2,3): full cycle permutation.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 24];
        permute_dims_neon(&data, &[2, 3, 4], &[1, 2, 0], &mut out);

        let mut expected = vec![0.0f32; 24];
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..2 {
                    let src_idx = k * 12 + i * 4 + j;
                    let dst_idx = i * 8 + j * 2 + k;
                    expected[dst_idx] = data[src_idx];
                }
            }
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn test_permute_3d_210() {
        // (2,3,4) → (4,3,2): full reversal.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 24];
        permute_dims_neon(&data, &[2, 3, 4], &[2, 1, 0], &mut out);

        let mut expected = vec![0.0f32; 24];
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    let src_idx = k * 12 + j * 4 + i;
                    let dst_idx = i * 6 + j * 2 + k;
                    expected[dst_idx] = data[src_idx];
                }
            }
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn test_permute_1d() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 8];
        permute_dims_neon(&data, &[8], &[0], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_permute_single_element() {
        let data = vec![99.0f32];
        let mut out = vec![0.0f32; 1];
        permute_dims_neon(&data, &[1, 1], &[1, 0], &mut out);
        assert_eq!(out, vec![99.0]);
    }

    #[test]
    fn test_permute_large_aligned() {
        // 4×16 → 16×4: NEON-aligned inner dimension.
        let data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 64];
        permute_dims_neon(&data, &[4, 16], &[1, 0], &mut out);

        let mut expected = vec![0.0f32; 64];
        for i in 0..16 {
            for j in 0..4 {
                expected[i * 4 + j] = data[j * 16 + i];
            }
        }
        assert_eq!(out, expected);
    }

    #[test]
    #[should_panic(expected = "perm.len()")]
    fn test_permute_wrong_perm_len() {
        let data = vec![0.0f32; 12];
        let mut out = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[0, 1, 2], &mut out);
    }

    #[test]
    #[should_panic(expected = "duplicate perm")]
    fn test_permute_duplicate_perm() {
        let data = vec![0.0f32; 12];
        let mut out = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[0, 0], &mut out);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_permute_perm_out_of_range() {
        let data = vec![0.0f32; 12];
        let mut out = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[0, 5], &mut out);
    }

    // ── helper tests ──────────────────────────────────────────────────

    #[test]
    fn test_default_strides_1d() {
        assert_eq!(default_strides(&[8]), vec![1]);
    }

    #[test]
    fn test_default_strides_2d() {
        assert_eq!(default_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_default_strides_3d() {
        assert_eq!(default_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_default_strides_empty() {
        let empty: Vec<usize> = vec![];
        assert_eq!(default_strides(&empty), Vec::<usize>::new());
    }

    #[test]
    fn test_is_contiguous_true() {
        assert!(is_contiguous(&[3, 4], &[4, 1]));
    }

    #[test]
    fn test_is_contiguous_false() {
        assert!(!is_contiguous(&[3, 4], &[1, 3]));
    }

    #[test]
    fn test_numel() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[5]), 5);
    }

    #[test]
    fn test_neon_copy_exact() {
        let src: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut dst = vec![0.0f32; 16];
        neon_copy(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_neon_copy_with_tail() {
        let src: Vec<f32> = (0..7).map(|x| x as f32).collect();
        let mut dst = vec![0.0f32; 7];
        neon_copy(&src, &mut dst);
        assert_eq!(dst, src);
    }

    // ── cross-operation integration ───────────────────────────────────

    #[test]
    fn test_squeeze_then_reshape() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let (sq_shape, _) = squeeze_neon(&data, &[1, 3, 1, 4], &[12, 4, 4, 1], None);
        assert_eq!(sq_shape, vec![3, 4]);
        let mut out = vec![0.0f32; 12];
        reshape_contiguous_neon(&data, &sq_shape, &[6, 2], &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn test_unsqueeze_then_broadcast() {
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        let (us_shape, _) = unsqueeze_neon(&data, &[4], &[1], 0, None);
        assert_eq!(us_shape, vec![1, 4]);
        let mut out = vec![0.0f32; 12];
        broadcast_expand_neon(&data, &us_shape, &[3, 4], &mut out);
        for row in out.chunks(4) {
            assert_eq!(row, &[0.0, 1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_permute_then_reshape() {
        // Permute (3,4) → (4,3), then reshape to (12,).
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut permuted = vec![0.0f32; 12];
        permute_dims_neon(&data, &[3, 4], &[1, 0], &mut permuted);
        let mut flat = vec![0.0f32; 12];
        reshape_contiguous_neon(&permuted, &[4, 3], &[12], &mut flat);
        assert_eq!(flat, permuted);
    }

    #[test]
    fn test_broadcast_then_permute() {
        // (1,4) → (2,4) → permute (4,2)
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut broad = vec![0.0f32; 8];
        broadcast_expand_neon(&data, &[1, 4], &[2, 4], &mut broad);

        let mut perm_out = vec![0.0f32; 8];
        permute_dims_neon(&broad, &[2, 4], &[1, 0], &mut perm_out);

        let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        assert_eq!(perm_out, expected);
    }

    #[test]
    fn test_reshape_preserves_data_integrity() {
        // Chain of reshapes should preserve all data exactly.
        let data: Vec<f32> = (0..120).map(|x| x as f32).collect();
        let mut a = vec![0.0f32; 120];
        reshape_contiguous_neon(&data, &[120], &[2, 3, 4, 5], &mut a);
        let mut b = vec![0.0f32; 120];
        reshape_contiguous_neon(&a, &[2, 3, 4, 5], &[10, 12], &mut b);
        let mut c = vec![0.0f32; 120];
        reshape_contiguous_neon(&b, &[10, 12], &[120], &mut c);
        assert_eq!(c, data);
    }

    #[test]
    fn test_strided_vs_contiguous_consistency() {
        // A contiguous tensor should produce identical output whether
        // copied via reshape_contiguous_neon or reshape_strided_neon.
        let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let mut out_c = vec![0.0f32; 32];
        let mut out_s = vec![0.0f32; 32];
        reshape_contiguous_neon(&data, &[4, 8], &[32], &mut out_c);
        reshape_strided_neon(&data, &[4, 8], &[8, 1], &[32], &mut out_s);
        assert_eq!(out_c, out_s);
    }

    #[test]
    fn test_broadcast_col_non_aligned() {
        // (3,1) → (3,5): inner broadcast with non-aligned size.
        let data = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 15];
        broadcast_expand_neon(&data, &[3, 1], &[3, 5], &mut out);
        let expected =
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_permute_4d() {
        // (2,3,2,2) with perm [0,2,1,3] — swap dims 1 and 2.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 24];
        permute_dims_neon(&data, &[2, 3, 2, 2], &[0, 2, 1, 3], &mut out);

        let shape = [2usize, 3, 2, 2];
        let mut expected = vec![0.0f32; 24];
        for a in 0..shape[0] {
            for b in 0..shape[2] {
                for c in 0..shape[1] {
                    for d in 0..shape[3] {
                        let src_idx = a * 12 + c * 4 + b * 2 + d;
                        let dst_idx = a * 12 + b * 6 + c * 2 + d;
                        expected[dst_idx] = data[src_idx];
                    }
                }
            }
        }
        assert_eq!(out, expected);
    }
}
