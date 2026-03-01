//! OpenCL tensor manipulation operations for Intel Arc A770 (Xe-HPG).
//!
//! Provides CPU reference implementations and OpenCL kernel sources for
//! common tensor operations: transpose, permute, reshape, concat, split,
//! slice, pad, broadcast, gather, and repeat.
//!
//! CPU implementations serve as ground-truth for validating OpenCL kernels.

use std::fmt;

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during tensor manipulation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorOpsError {
    /// Shape dimensions do not match the expected sizes.
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Axis index exceeds the tensor's number of dimensions.
    InvalidAxis { axis: usize, ndim: usize },
    /// Index or range exceeds the dimension's extent.
    OutOfBounds {
        index: usize,
        size: usize,
    },
    /// Permutation vector is not a valid permutation of `0..ndim`.
    InvalidPermutation { perm: Vec<usize>, ndim: usize },
}

impl fmt::Display for TensorOpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "invalid axis {axis} for {ndim}-D tensor")
            }
            Self::OutOfBounds { index, size } => {
                write!(f, "index {index} out of bounds for size {size}")
            }
            Self::InvalidPermutation { perm, ndim } => {
                write!(
                    f,
                    "invalid permutation {perm:?} for {ndim}-D tensor"
                )
            }
        }
    }
}

impl std::error::Error for TensorOpsError {}

// -----------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------

/// N-dimensional tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Row-major strides for this shape.
    pub fn strides(&self) -> Vec<usize> {
        compute_strides(&self.0)
    }

    /// Stride for a specific dimension.
    pub fn stride_for(&self, dim: usize) -> usize {
        self.strides()[dim]
    }
}

/// Memory layout descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorLayout {
    /// Elements are stored contiguously in row-major order.
    Contiguous,
    /// Elements are stored with explicit strides per dimension.
    Strided(Vec<usize>),
    /// View formed by transposing dimensions according to a permutation.
    Transposed(Vec<usize>),
}

/// A view into a contiguous buffer described by offset, shape, and strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSlice {
    /// Byte-element offset into the backing buffer.
    pub offset: usize,
    /// Shape of the slice.
    pub shape: Vec<usize>,
    /// Strides of the slice (element counts, not bytes).
    pub strides: Vec<usize>,
}

// -----------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------

/// Compute row-major strides from a shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
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

/// Convert a multi-dimensional index to a flat offset using strides.
pub fn flat_index(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Convert a flat offset to a multi-dimensional index.
pub fn unravel_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = vec![0usize; shape.len()];
    let mut remaining = flat;
    for (i, s) in strides.iter().enumerate() {
        if *s > 0 {
            indices[i] = remaining / s;
            remaining %= s;
        }
    }
    indices
}

/// Check whether `perm` is a valid permutation of `0..ndim`.
pub fn is_valid_permutation(perm: &[usize], ndim: usize) -> bool {
    if perm.len() != ndim {
        return false;
    }
    let mut seen = vec![false; ndim];
    for &p in perm {
        if p >= ndim || seen[p] {
            return false;
        }
        seen[p] = true;
    }
    true
}

/// Compute the broadcast-compatible shape for two shapes (NumPy rules).
pub fn broadcast_shapes(
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, TensorOpsError> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        let da =
            if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db =
            if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return Err(TensorOpsError::ShapeMismatch {
                expected: a.to_vec(),
                actual: b.to_vec(),
            });
        }
    }
    Ok(result)
}

// -----------------------------------------------------------------------
// CPU reference implementations
// -----------------------------------------------------------------------

/// Transpose a 2-D matrix stored in row-major order.
pub fn cpu_transpose_2d(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Generalized permute for an N-D tensor.
pub fn cpu_permute(
    data: &[f32],
    shape: &[usize],
    perm: &[usize],
) -> Vec<f32> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let src_strides = compute_strides(shape);
    let dst_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let dst_strides = compute_strides(&dst_shape);
    let mut out = vec![0.0f32; numel];

    for flat in 0..numel {
        let src_idx = unravel_index(flat, shape);
        let mut dst_idx = vec![0usize; ndim];
        for (d, &p) in perm.iter().enumerate() {
            dst_idx[d] = src_idx[p];
        }
        out[flat_index(&dst_idx, &dst_strides)] =
            data[flat_index(&src_idx, &src_strides)];
    }
    out
}

/// Reshape a tensor. The total element count must match.
pub fn cpu_reshape(
    data: &[f32],
    old_shape: &[usize],
    new_shape: &[usize],
) -> Result<Vec<f32>, TensorOpsError> {
    let old_numel: usize = old_shape.iter().product();
    let new_numel: usize = new_shape.iter().product();
    if old_numel != new_numel {
        return Err(TensorOpsError::ShapeMismatch {
            expected: old_shape.to_vec(),
            actual: new_shape.to_vec(),
        });
    }
    Ok(data.to_vec())
}

/// Concatenate tensors along `axis`.
pub fn cpu_concat(
    tensors: &[&[f32]],
    shapes: &[&[usize]],
    axis: usize,
) -> Result<(Vec<f32>, Vec<usize>), TensorOpsError> {
    if tensors.is_empty() {
        return Ok((vec![], vec![]));
    }
    let ndim = shapes[0].len();
    if axis >= ndim {
        return Err(TensorOpsError::InvalidAxis { axis, ndim });
    }
    // Validate that all non-axis dimensions match.
    for (i, shape) in shapes.iter().enumerate().skip(1) {
        for d in 0..ndim {
            if d != axis && shape[d] != shapes[0][d] {
                return Err(TensorOpsError::ShapeMismatch {
                    expected: shapes[0].to_vec(),
                    actual: shapes[i].to_vec(),
                });
            }
        }
    }

    let mut out_shape = shapes[0].to_vec();
    out_shape[axis] = shapes.iter().map(|s| s[axis]).sum();

    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_numel];
    let out_strides = compute_strides(&out_shape);

    let mut axis_offset = 0usize;
    for (t, shape) in tensors.iter().zip(shapes.iter()) {
        let src_strides = compute_strides(shape);
        let numel: usize = shape.iter().product();
        for flat in 0..numel {
            let mut idx = unravel_index(flat, shape);
            idx[axis] += axis_offset;
            out[flat_index(&idx, &out_strides)] =
                t[flat_index(
                    &unravel_index(flat, shape),
                    &src_strides,
                )];
        }
        axis_offset += shape[axis];
    }
    Ok((out, out_shape))
}

/// Split a tensor along `axis` into chunks of the given sizes.
pub fn cpu_split(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<f32>>, TensorOpsError> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(TensorOpsError::InvalidAxis { axis, ndim });
    }
    let total: usize = sizes.iter().sum();
    if total != shape[axis] {
        return Err(TensorOpsError::ShapeMismatch {
            expected: shape.to_vec(),
            actual: vec![total],
        });
    }

    let src_strides = compute_strides(shape);
    let mut results = Vec::with_capacity(sizes.len());
    let mut axis_offset = 0usize;

    for &sz in sizes {
        let mut chunk_shape = shape.to_vec();
        chunk_shape[axis] = sz;
        let chunk_strides = compute_strides(&chunk_shape);
        let chunk_numel: usize = chunk_shape.iter().product();
        let mut chunk = vec![0.0f32; chunk_numel];

        for flat in 0..chunk_numel {
            let mut idx = unravel_index(flat, &chunk_shape);
            let dst_flat = flat_index(&idx, &chunk_strides);
            idx[axis] += axis_offset;
            chunk[dst_flat] = data[flat_index(&idx, &src_strides)];
        }
        results.push(chunk);
        axis_offset += sz;
    }
    Ok(results)
}

/// Slice a tensor with a range per dimension.
pub fn cpu_slice(
    data: &[f32],
    shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<(Vec<f32>, Vec<usize>), TensorOpsError> {
    let ndim = shape.len();
    if ranges.len() != ndim {
        return Err(TensorOpsError::ShapeMismatch {
            expected: shape.to_vec(),
            actual: ranges.iter().map(|r| r.1 - r.0).collect(),
        });
    }
    for (d, &(start, end)) in ranges.iter().enumerate() {
        if end > shape[d] {
            return Err(TensorOpsError::OutOfBounds {
                index: end,
                size: shape[d],
            });
        }
        if start > end {
            return Err(TensorOpsError::OutOfBounds {
                index: start,
                size: end,
            });
        }
    }

    let out_shape: Vec<usize> =
        ranges.iter().map(|&(s, e)| e - s).collect();
    let out_numel: usize = out_shape.iter().product();
    let src_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0.0f32; out_numel];

    for flat in 0..out_numel {
        let local = unravel_index(flat, &out_shape);
        let global: Vec<usize> =
            local.iter().zip(ranges.iter()).map(|(l, r)| l + r.0).collect();
        out[flat_index(&local, &out_strides)] =
            data[flat_index(&global, &src_strides)];
    }
    Ok((out, out_shape))
}

/// Pad a tensor with `pad_value` on each side of each dimension.
pub fn cpu_pad(
    data: &[f32],
    shape: &[usize],
    padding: &[(usize, usize)],
    pad_value: f32,
) -> (Vec<f32>, Vec<usize>) {
    let ndim = shape.len();
    let out_shape: Vec<usize> = (0..ndim)
        .map(|d| shape[d] + padding[d].0 + padding[d].1)
        .collect();
    let out_numel: usize = out_shape.iter().product();
    let src_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![pad_value; out_numel];

    let numel: usize = shape.iter().product();
    for flat in 0..numel {
        let src_idx = unravel_index(flat, shape);
        let dst_idx: Vec<usize> = src_idx
            .iter()
            .zip(padding.iter())
            .map(|(i, p)| i + p.0)
            .collect();
        out[flat_index(&dst_idx, &out_strides)] =
            data[flat_index(&src_idx, &src_strides)];
    }
    (out, out_shape)
}

/// Broadcast a tensor from `from_shape` to `to_shape` (NumPy semantics).
pub fn cpu_broadcast(
    data: &[f32],
    from_shape: &[usize],
    to_shape: &[usize],
) -> Result<Vec<f32>, TensorOpsError> {
    // Validate broadcast compatibility.
    let _ = broadcast_shapes(from_shape, to_shape)?;

    let ndim = to_shape.len();
    let from_ndim = from_shape.len();
    let out_numel: usize = to_shape.iter().product();
    let mut out = vec![0.0f32; out_numel];

    // Compute padded source shape (left-pad with 1s).
    let padded_from: Vec<usize> = (0..ndim)
        .map(|i| {
            if i < ndim - from_ndim {
                1
            } else {
                from_shape[i - (ndim - from_ndim)]
            }
        })
        .collect();
    let src_strides = compute_strides(&padded_from);
    let dst_strides = compute_strides(to_shape);

    for flat in 0..out_numel {
        let dst_idx = unravel_index(flat, to_shape);
        let src_idx: Vec<usize> = dst_idx
            .iter()
            .zip(padded_from.iter())
            .map(|(&di, &fs)| if fs == 1 { 0 } else { di })
            .collect();
        out[flat_index(&dst_idx, &dst_strides)] =
            data[flat_index(&src_idx, &src_strides)];
    }
    Ok(out)
}

/// Gather slices along `axis` at the given indices.
pub fn cpu_gather(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    indices: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), TensorOpsError> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(TensorOpsError::InvalidAxis { axis, ndim });
    }
    for &idx in indices {
        if idx >= shape[axis] {
            return Err(TensorOpsError::OutOfBounds {
                index: idx,
                size: shape[axis],
            });
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let out_numel: usize = out_shape.iter().product();
    let src_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0.0f32; out_numel];

    for flat in 0..out_numel {
        let dst_idx = unravel_index(flat, &out_shape);
        let mut src_idx = dst_idx.clone();
        src_idx[axis] = indices[dst_idx[axis]];
        out[flat_index(&dst_idx, &out_strides)] =
            data[flat_index(&src_idx, &src_strides)];
    }
    Ok((out, out_shape))
}

/// Tile / repeat a tensor along each dimension.
pub fn cpu_repeat(
    data: &[f32],
    shape: &[usize],
    repeats: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    let ndim = shape.len();
    let out_shape: Vec<usize> =
        (0..ndim).map(|d| shape[d] * repeats[d]).collect();
    let out_numel: usize = out_shape.iter().product();
    let src_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0.0f32; out_numel];

    for flat in 0..out_numel {
        let dst_idx = unravel_index(flat, &out_shape);
        let src_idx: Vec<usize> =
            dst_idx.iter().zip(shape.iter()).map(|(d, s)| d % s).collect();
        out[flat_index(&dst_idx, &out_strides)] =
            data[flat_index(&src_idx, &src_strides)];
    }
    (out, out_shape)
}

// -----------------------------------------------------------------------
// OpenCL kernel source
// -----------------------------------------------------------------------

/// OpenCL C kernel source for tensor manipulation operations.
///
/// Contains kernels for:
/// - `transpose_2d`: 2-D matrix transpose using local memory tiling
/// - `permute_nd`:   N-D permute via stride mapping
/// - `gather_axis`:  gather along a single axis
/// - `pad_tensor`:   pad with a constant value
pub const TENSOR_OPS_SRC: &str = r#"
// ----- transpose_2d -----
// Work-group: (TILE, TILE) where TILE is typically 16.
// Global size: (ceil(cols/TILE)*TILE, ceil(rows/TILE)*TILE).
#ifndef TILE
#define TILE 16
#endif

__kernel void transpose_2d(
    __global const float* src,
    __global       float* dst,
    const uint rows,
    const uint cols)
{
    __local float tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    uint gx = get_global_id(0); // col in src
    uint gy = get_global_id(1); // row in src
    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    if (gx < cols && gy < rows) {
        tile[ly][lx] = src[gy * cols + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Transposed coordinates
    uint new_gx = get_group_id(1) * TILE + lx;
    uint new_gy = get_group_id(0) * TILE + ly;

    if (new_gx < rows && new_gy < cols) {
        dst[new_gy * rows + new_gx] = tile[lx][ly];
    }
}

// ----- permute_nd -----
// Generic N-D permute via per-element stride remapping.
// `src_strides` and `dst_strides` are arrays of length `ndim`.
// `perm` maps destination dimension d to source dimension perm[d].
__kernel void permute_nd(
    __global const float* src,
    __global       float* dst,
    __global const uint*  src_strides,
    __global const uint*  dst_strides,
    __global const uint*  perm,
    __global const uint*  shape,       // source shape
    const uint ndim,
    const uint numel)
{
    uint gid = get_global_id(0);
    if (gid >= numel) return;

    // Unravel flat index into source multi-index.
    uint remaining = gid;
    uint src_idx[16]; // max 16 dims
    for (uint d = 0; d < ndim; d++) {
        src_idx[d] = remaining / src_strides[d];
        remaining  = remaining % src_strides[d];
    }

    // Map to destination multi-index via permutation.
    uint dst_flat = 0;
    for (uint d = 0; d < ndim; d++) {
        dst_flat += src_idx[perm[d]] * dst_strides[d];
    }

    dst[dst_flat] = src[gid];
}

// ----- gather_axis -----
// Gather elements along a single axis.
// `indices` has length `num_indices`.
// Output shape equals input shape with shape[axis] replaced by num_indices.
__kernel void gather_axis(
    __global const float* src,
    __global       float* dst,
    __global const uint*  indices,
    __global const uint*  src_strides,
    __global const uint*  dst_strides,
    __global const uint*  dst_shape,
    const uint axis,
    const uint ndim,
    const uint numel)
{
    uint gid = get_global_id(0);
    if (gid >= numel) return;

    // Unravel destination flat index.
    uint remaining = gid;
    uint dst_idx[16];
    for (uint d = 0; d < ndim; d++) {
        dst_idx[d] = remaining / dst_strides[d];
        remaining  = remaining % dst_strides[d];
    }

    // Replace axis index with gathered index.
    uint src_flat = 0;
    for (uint d = 0; d < ndim; d++) {
        uint idx_d = dst_idx[d];
        if (d == axis) {
            idx_d = indices[dst_idx[d]];
        }
        src_flat += idx_d * src_strides[d];
    }

    dst[gid] = src[src_flat];
}

// ----- pad_tensor -----
// Pads an N-D tensor with a constant value.
// `pad_before` has length `ndim` — number of elements to prepend per dim.
__kernel void pad_tensor(
    __global const float* src,
    __global       float* dst,
    __global const uint*  src_shape,
    __global const uint*  src_strides,
    __global const uint*  dst_strides,
    __global const uint*  pad_before,
    const float pad_value,
    const uint ndim,
    const uint dst_numel)
{
    uint gid = get_global_id(0);
    if (gid >= dst_numel) return;

    // Unravel into destination indices.
    uint remaining = gid;
    uint dst_idx[16];
    for (uint d = 0; d < ndim; d++) {
        dst_idx[d] = remaining / dst_strides[d];
        remaining  = remaining % dst_strides[d];
    }

    // Check if this element is inside the padded source region.
    uint src_flat = 0;
    bool inside = true;
    for (uint d = 0; d < ndim; d++) {
        int si = (int)dst_idx[d] - (int)pad_before[d];
        if (si < 0 || (uint)si >= src_shape[d]) {
            inside = false;
            break;
        }
        src_flat += (uint)si * src_strides[d];
    }

    dst[gid] = inside ? src[src_flat] : pad_value;
}
"#;

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === TensorShape ==================================================

    #[test]
    fn test_tensor_shape_ndim() {
        let s = TensorShape(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
    }

    #[test]
    fn test_tensor_shape_numel() {
        let s = TensorShape(vec![2, 3, 4]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn test_tensor_shape_stride_for() {
        let s = TensorShape(vec![2, 3, 4]);
        assert_eq!(s.stride_for(0), 12);
        assert_eq!(s.stride_for(1), 4);
        assert_eq!(s.stride_for(2), 1);
    }

    #[test]
    fn test_tensor_shape_empty() {
        let s = TensorShape(vec![]);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1); // product of empty = 1
    }

    // === compute_strides / flat_index / unravel_index =================

    #[test]
    fn test_compute_strides_3d() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_compute_strides_1d() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_compute_strides_empty() {
        let s: Vec<usize> = compute_strides(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn test_flat_index_basic() {
        let strides = compute_strides(&[2, 3]);
        assert_eq!(flat_index(&[1, 2], &strides), 5);
    }

    #[test]
    fn test_unravel_index_basic() {
        assert_eq!(unravel_index(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_flat_unravel_roundtrip() {
        let shape = [3, 4, 5];
        let strides = compute_strides(&shape);
        for flat in 0..60 {
            let idx = unravel_index(flat, &shape);
            assert_eq!(flat_index(&idx, &strides), flat);
        }
    }

    // === is_valid_permutation =========================================

    #[test]
    fn test_valid_permutation() {
        assert!(is_valid_permutation(&[2, 0, 1], 3));
    }

    #[test]
    fn test_invalid_permutation_wrong_len() {
        assert!(!is_valid_permutation(&[0, 1], 3));
    }

    #[test]
    fn test_invalid_permutation_duplicate() {
        assert!(!is_valid_permutation(&[0, 0, 1], 3));
    }

    #[test]
    fn test_invalid_permutation_out_of_range() {
        assert!(!is_valid_permutation(&[0, 1, 5], 3));
    }

    // === broadcast_shapes =============================================

    #[test]
    fn test_broadcast_shapes_scalar_to_vec() {
        assert_eq!(broadcast_shapes(&[1], &[5]).unwrap(), vec![5]);
    }

    #[test]
    fn test_broadcast_shapes_vec_to_matrix() {
        assert_eq!(
            broadcast_shapes(&[3], &[2, 3]).unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        assert!(broadcast_shapes(&[3], &[4]).is_err());
    }

    // === cpu_transpose_2d =============================================

    #[test]
    fn test_transpose_1x1() {
        assert_eq!(cpu_transpose_2d(&[42.0], 1, 1), vec![42.0]);
    }

    #[test]
    fn test_transpose_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = cpu_transpose_2d(&data, 2, 3);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_4x4_identity() {
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let t1 = cpu_transpose_2d(&data, 4, 4);
        let t2 = cpu_transpose_2d(&t1, 4, 4);
        assert_eq!(t2, data);
    }

    #[test]
    fn test_transpose_square_symmetric() {
        // Symmetric matrix should be unchanged by transpose.
        let data = vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0];
        assert_eq!(cpu_transpose_2d(&data, 3, 3), data);
    }

    #[test]
    fn test_transpose_large_128x256() {
        let rows = 128;
        let cols = 256;
        let data: Vec<f32> =
            (0..rows * cols).map(|x| x as f32).collect();
        let out = cpu_transpose_2d(&data, rows, cols);
        assert_eq!(out.len(), rows * cols);
        // Spot-check: src[0][1] == 1.0 should be at dst[1][0] == dst[1*128+0]
        assert_eq!(out[1 * rows + 0], 1.0);
        // src[2][5] = 2*256+5 = 517 should be at dst[5][2] = 5*128+2 = 642
        assert_eq!(out[5 * rows + 2], 517.0);
    }

    // === cpu_permute ==================================================

    #[test]
    fn test_permute_identity() {
        let shape = [2, 3, 4];
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let out = cpu_permute(&data, &shape, &[0, 1, 2]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_permute_reverse() {
        // (2,3,4) -> (4,3,2)
        let shape = [2, 3, 4];
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let out = cpu_permute(&data, &shape, &[2, 1, 0]);
        // element at src[1][2][3] = 1*12+2*4+3 = 23
        // goes to dst[3][2][1] = 3*6+2*2+1 = 23
        assert_eq!(out[23], 23.0);
    }

    #[test]
    fn test_permute_rotation() {
        let shape = [2, 3, 4];
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let out = cpu_permute(&data, &shape, &[1, 2, 0]);
        // src[1][0][0] = 12.0 -> dst[0][0][1], dst shape (3,4,2)
        // dst strides = [8,2,1], dst[0][0][1] = 1
        assert_eq!(out[1], 12.0);
    }

    #[test]
    fn test_permute_roundtrip() {
        let shape = [2, 3, 4];
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let perm = [2, 0, 1];
        let inv_perm = [1, 2, 0];
        let permuted = cpu_permute(&data, &shape, &perm);
        let perm_shape: Vec<usize> =
            perm.iter().map(|&p| shape[p]).collect();
        let recovered = cpu_permute(&permuted, &perm_shape, &inv_perm);
        assert_eq!(recovered, data);
    }

    // === cpu_reshape ==================================================

    #[test]
    fn test_reshape_same_size() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let out =
            cpu_reshape(&data, &[3, 4], &[4, 3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_to_flat() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let out = cpu_reshape(&data, &[3, 4], &[12]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_size_mismatch() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        assert!(cpu_reshape(&data, &[3, 4], &[5, 3]).is_err());
    }

    // === cpu_concat ===================================================

    #[test]
    fn test_concat_axis0() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let (out, shape) = cpu_concat(
            &[&a, &b],
            &[&[2, 2], &[2, 2]],
            0,
        )
        .unwrap();
        assert_eq!(shape, vec![4, 2]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_axis1() {
        // [[1,2],[3,4]] ++ [[5],[6]] along axis 1 = [[1,2,5],[3,4,6]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0];
        let (out, shape) = cpu_concat(
            &[&a, &b],
            &[&[2, 2], &[2, 1]],
            1,
        )
        .unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_concat_shape_mismatch() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!(cpu_concat(
            &[&a, &b],
            &[&[2, 2], &[2, 3]],
            0,
        )
        .is_err());
    }

    // === cpu_split ====================================================

    #[test]
    fn test_split_basic() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let parts =
            cpu_split(&data, &[4, 3], 0, &[2, 2]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(parts[1], vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape = [2, 3, 4];
        let parts =
            cpu_split(&data, &shape, 1, &[1, 2]).unwrap();
        let part_shapes: Vec<Vec<usize>> = parts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut s = shape.to_vec();
                s[1] = [1, 2][i];
                s
            })
            .collect();
        let refs: Vec<&[f32]> = parts.iter().map(|v| v.as_slice()).collect();
        let shape_refs: Vec<&[usize]> =
            part_shapes.iter().map(|v| v.as_slice()).collect();
        let (recovered, rshape) =
            cpu_concat(&refs, &shape_refs, 1).unwrap();
        assert_eq!(rshape, shape.to_vec());
        assert_eq!(recovered, data);
    }

    // === cpu_slice ====================================================

    #[test]
    fn test_slice_basic() {
        // 3x4 matrix, take rows 0..2, cols 1..3
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let (out, shape) =
            cpu_slice(&data, &[3, 4], &[(0, 2), (1, 3)]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_full_range() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (out, shape) =
            cpu_slice(&data, &[2, 3], &[(0, 2), (0, 3)]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        assert!(
            cpu_slice(&data, &[2, 3], &[(0, 2), (0, 5)]).is_err()
        );
    }

    // === cpu_pad ======================================================

    #[test]
    fn test_pad_symmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (out, shape) =
            cpu_pad(&data, &[2, 2], &[(1, 1), (1, 1)], 0.0);
        assert_eq!(shape, vec![4, 4]);
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_pad_asymmetric() {
        let data = vec![1.0, 2.0];
        let (out, shape) = cpu_pad(&data, &[2], &[(2, 1)], -1.0);
        assert_eq!(shape, vec![5]);
        assert_eq!(out, vec![-1.0, -1.0, 1.0, 2.0, -1.0]);
    }

    #[test]
    fn test_pad_zero_padding() {
        let data = vec![1.0, 2.0, 3.0];
        let (out, shape) = cpu_pad(&data, &[3], &[(0, 0)], 0.0);
        assert_eq!(shape, vec![3]);
        assert_eq!(out, data);
    }

    // === cpu_broadcast ================================================

    #[test]
    fn test_broadcast_scalar_to_vec() {
        let out = cpu_broadcast(&[5.0], &[1], &[4]).unwrap();
        assert_eq!(out, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_broadcast_vec_to_matrix() {
        let out =
            cpu_broadcast(&[1.0, 2.0, 3.0], &[3], &[2, 3]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(cpu_broadcast(&[1.0, 2.0, 3.0], &[3], &[4]).is_err());
    }

    #[test]
    fn test_broadcast_col_vec_to_matrix() {
        // (2,1) -> (2,3)
        let out =
            cpu_broadcast(&[1.0, 2.0], &[2, 1], &[2, 3]).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    // === cpu_gather ===================================================

    #[test]
    fn test_gather_basic() {
        // 3x2 matrix, gather rows [0, 2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) =
            cpu_gather(&data, &[3, 2], 0, &[0, 2]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(cpu_gather(&data, &[2, 2], 0, &[3]).is_err());
    }

    #[test]
    fn test_gather_axis1() {
        // 2x3, gather cols [2, 0]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) =
            cpu_gather(&data, &[2, 3], 1, &[2, 0]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(out, vec![3.0, 1.0, 6.0, 4.0]);
    }

    // === cpu_repeat ===================================================

    #[test]
    fn test_repeat_1d() {
        let (out, shape) = cpu_repeat(&[1.0, 2.0], &[2], &[3]);
        assert_eq!(shape, vec![6]);
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (out, shape) = cpu_repeat(&data, &[2, 2], &[2, 3]);
        assert_eq!(shape, vec![4, 6]);
        #[rustfmt::skip]
        let expected = vec![
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
            3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
        ];
        assert_eq!(out, expected);
    }

    // === TensorLayout / TensorSlice (construction) ====================

    #[test]
    fn test_tensor_layout_variants() {
        let _c = TensorLayout::Contiguous;
        let _s = TensorLayout::Strided(vec![12, 4, 1]);
        let _t = TensorLayout::Transposed(vec![1, 0]);
        // Ensure they compare correctly.
        assert_eq!(TensorLayout::Contiguous, TensorLayout::Contiguous);
    }

    #[test]
    fn test_tensor_slice_construction() {
        let ts = TensorSlice {
            offset: 8,
            shape: vec![2, 3],
            strides: vec![3, 1],
        };
        assert_eq!(ts.offset, 8);
        assert_eq!(ts.shape, vec![2, 3]);
        assert_eq!(ts.strides, vec![3, 1]);
    }

    // === Error display ================================================

    #[test]
    fn test_error_display() {
        let e = TensorOpsError::InvalidAxis { axis: 5, ndim: 3 };
        let msg = format!("{e}");
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    // === OpenCL kernel source smoke ===================================

    #[test]
    fn test_kernel_source_contains_transpose() {
        assert!(TENSOR_OPS_SRC.contains("transpose_2d"));
    }

    #[test]
    fn test_kernel_source_contains_permute() {
        assert!(TENSOR_OPS_SRC.contains("permute_nd"));
    }

    #[test]
    fn test_kernel_source_contains_gather() {
        assert!(TENSOR_OPS_SRC.contains("gather_axis"));
    }

    #[test]
    fn test_kernel_source_contains_pad() {
        assert!(TENSOR_OPS_SRC.contains("pad_tensor"));
    }
}
