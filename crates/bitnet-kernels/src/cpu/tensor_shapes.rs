//! CPU tensor shape manipulation kernel.
//!
//! Provides a `TensorShape` descriptor and free-standing shape operations:
//! reshape, transpose (2-D and N-D), permute, squeeze, unsqueeze,
//! NumPy-style broadcasting, flatten, and broadcast-expand.
//!
//! All functions operate on contiguous row-major `f32` slices and never
//! allocate beyond the output buffer.

use core::fmt;

// ── Error type ─────────────────────────────────────────────────────

/// Error returned by shape manipulation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    msg: String,
}

impl ShapeError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShapeError: {}", self.msg)
    }
}

impl std::error::Error for ShapeError {}

// ── TensorShape ────────────────────────────────────────────────────

/// Lightweight shape descriptor for a contiguous tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new `TensorShape` from a dimension list.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (product of all dimensions).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// True when the shape has zero dimensions (scalar).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// True when the shape has exactly one dimension.
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }

    /// True when the shape has exactly two dimensions.
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }

    /// Return a reference to the underlying dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

// ── Free-standing operations ───────────────────────────────────────

/// Reshape `data` from `old_shape` to `new_shape`.
///
/// Validates that element counts match.  Because the data is already
/// contiguous, reshape is a zero-copy logical operation — we clone the
/// buffer to give the caller ownership of a new `Vec`.
pub fn reshape(
    data: &[f32],
    old_shape: &[usize],
    new_shape: &[usize],
) -> Result<Vec<f32>, ShapeError> {
    let old_numel: usize = old_shape.iter().product();
    let new_numel: usize = new_shape.iter().product();

    if data.len() != old_numel {
        return Err(ShapeError::new("data length does not match old_shape"));
    }
    if old_numel != new_numel {
        return Err(ShapeError::new(format!(
            "cannot reshape: element count {old_numel} != {new_numel}"
        )));
    }
    Ok(data.to_vec())
}

/// 2-D matrix transpose (row-major).
///
/// Given `rows × cols` input, produces `cols × rows` output where
/// `out[j * rows + i] = data[i * cols + j]`.
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let numel = rows * cols;
    assert_eq!(data.len(), numel, "data length must equal rows * cols");
    if numel == 0 {
        return vec![];
    }
    let mut out = vec![0.0f32; numel];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// N-dimensional transpose with arbitrary axis permutation.
///
/// `axes` must be a permutation of `0..shape.len()`.
pub fn transpose_nd(data: &[f32], shape: &[usize], axes: &[usize]) -> Result<Vec<f32>, ShapeError> {
    let ndim = shape.len();
    if axes.len() != ndim {
        return Err(ShapeError::new("axes length must equal ndim"));
    }

    let mut seen = vec![false; ndim];
    for &a in axes {
        if a >= ndim {
            return Err(ShapeError::new("axis out of range"));
        }
        if seen[a] {
            return Err(ShapeError::new("duplicate axis"));
        }
        seen[a] = true;
    }

    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return Err(ShapeError::new("data length must match product of shape"));
    }
    if numel == 0 {
        return Ok(vec![]);
    }

    let out_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();
    let dst_strides = c_contiguous_strides(&out_shape);

    let mut out = vec![0.0f32; numel];
    let mut src_idx = vec![0usize; ndim];

    for &val in &data[..numel] {
        let mut dst_flat = 0usize;
        for d in 0..ndim {
            dst_flat += src_idx[axes[d]] * dst_strides[d];
        }
        out[dst_flat] = val;

        // Increment source multi-index (odometer).
        for d in (0..ndim).rev() {
            src_idx[d] += 1;
            if src_idx[d] < shape[d] {
                break;
            }
            src_idx[d] = 0;
        }
    }
    Ok(out)
}

/// Alias for [`transpose_nd`] — permute dimensions according to `axes`.
pub fn permute(data: &[f32], shape: &[usize], axes: &[usize]) -> Result<Vec<f32>, ShapeError> {
    transpose_nd(data, shape, axes)
}

/// Remove all size-1 dimensions.
pub fn squeeze(shape: &[usize]) -> Vec<usize> {
    shape.iter().copied().filter(|&d| d != 1).collect()
}

/// Insert a size-1 dimension at position `dim` (0 ≤ dim ≤ shape.len()).
pub fn unsqueeze(shape: &[usize], dim: usize) -> Result<Vec<usize>, ShapeError> {
    if dim > shape.len() {
        return Err(ShapeError::new(format!(
            "dim {dim} out of range for shape with {} dims",
            shape.len()
        )));
    }
    let mut out = Vec::with_capacity(shape.len() + 1);
    out.extend_from_slice(&shape[..dim]);
    out.push(1);
    out.extend_from_slice(&shape[dim..]);
    Ok(out)
}

/// Compute the broadcast-compatible output shape for shapes `a` and `b`
/// using NumPy-style broadcasting rules.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(ShapeError::new(format!(
                "incompatible broadcast dimensions: {da} vs {db}"
            )));
        }
    }
    result.reverse();
    Ok(result)
}

/// Flatten contiguous dimensions `[start_dim, end_dim]` (inclusive).
///
/// Returns the (cloned) data and the new logical shape.
pub fn flatten(
    data: &[f32],
    shape: &[usize],
    start_dim: usize,
    end_dim: usize,
) -> Result<(Vec<f32>, Vec<usize>), ShapeError> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(ShapeError::new("cannot flatten a scalar shape"));
    }
    if start_dim >= ndim || end_dim >= ndim {
        return Err(ShapeError::new("dim out of range"));
    }
    if start_dim > end_dim {
        return Err(ShapeError::new("start_dim must be <= end_dim"));
    }
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return Err(ShapeError::new("data length does not match shape"));
    }

    let flat_size: usize = shape[start_dim..=end_dim].iter().product();
    let mut new_shape = Vec::with_capacity(ndim - (end_dim - start_dim));
    new_shape.extend_from_slice(&shape[..start_dim]);
    new_shape.push(flat_size);
    new_shape.extend_from_slice(&shape[end_dim + 1..]);

    Ok((data.to_vec(), new_shape))
}

/// Broadcast-expand `data` with `shape` to `target` shape.
///
/// Each dimension in `shape` must be either equal to the corresponding
/// `target` dimension, or be 1 (in which case the data is repeated).
/// `shape` is right-aligned against `target` (NumPy semantics).
pub fn expand(data: &[f32], shape: &[usize], target: &[usize]) -> Result<Vec<f32>, ShapeError> {
    if target.len() < shape.len() {
        return Err(ShapeError::new("target must have at least as many dims as shape"));
    }

    // Right-align shape against target, padding with 1.
    let pad = target.len() - shape.len();
    let padded: Vec<usize> =
        (0..target.len()).map(|i| if i < pad { 1 } else { shape[i - pad] }).collect();

    for (i, (&s, &t)) in padded.iter().zip(target.iter()).enumerate() {
        if s != 1 && s != t {
            return Err(ShapeError::new(format!(
                "cannot expand dim {i}: source {s} != target {t} and source != 1"
            )));
        }
    }

    let src_numel: usize = shape.iter().product();
    if data.len() != src_numel {
        return Err(ShapeError::new("data length does not match shape"));
    }

    let dst_numel: usize = target.iter().product();
    if dst_numel == 0 {
        return Ok(vec![]);
    }

    let src_strides = broadcast_strides(&padded, target);
    let dst_strides = c_contiguous_strides(target);
    let ndim = target.len();

    let mut out = vec![0.0f32; dst_numel];
    let mut idx = vec![0usize; ndim];

    for item in out.iter_mut() {
        let mut src_flat = 0usize;
        for d in 0..ndim {
            src_flat += idx[d] * src_strides[d];
        }
        *item = data[src_flat];

        // Increment multi-index.
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < target[d] {
                break;
            }
            idx[d] = 0;
        }
    }

    let _ = dst_strides; // suppress unused warning

    Ok(out)
}

// ── Internal helpers ───────────────────────────────────────────────

/// Row-major strides for the given shape.
fn c_contiguous_strides(shape: &[usize]) -> Vec<usize> {
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

/// Compute strides for a source tensor that is being broadcast to `target`.
///
/// Where `padded[d] == 1` and `target[d] > 1`, the stride is 0 (repeat).
fn broadcast_strides(padded: &[usize], target: &[usize]) -> Vec<usize> {
    let ndim = target.len();
    let base = c_contiguous_strides(padded);
    (0..ndim).map(|d| if padded[d] == 1 && target[d] > 1 { 0 } else { base[d] }).collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── TensorShape ────────────────────────────────────────────

    #[test]
    fn shape_scalar() {
        let s = TensorShape::new(vec![]);
        assert!(s.is_scalar());
        assert!(!s.is_vector());
        assert!(!s.is_matrix());
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1); // empty product
    }

    #[test]
    fn shape_vector() {
        let s = TensorShape::new(vec![5]);
        assert!(!s.is_scalar());
        assert!(s.is_vector());
        assert!(!s.is_matrix());
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.numel(), 5);
    }

    #[test]
    fn shape_matrix() {
        let s = TensorShape::new(vec![3, 4]);
        assert!(s.is_matrix());
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.numel(), 12);
    }

    #[test]
    fn shape_3d() {
        let s = TensorShape::new(vec![2, 3, 4]);
        assert!(!s.is_scalar());
        assert!(!s.is_vector());
        assert!(!s.is_matrix());
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn shape_dims_accessor() {
        let s = TensorShape::new(vec![2, 3]);
        assert_eq!(s.dims(), &[2, 3]);
    }

    #[test]
    fn shape_clone_eq() {
        let a = TensorShape::new(vec![2, 3]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── reshape ────────────────────────────────────────────────

    #[test]
    fn reshape_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = reshape(&data, &[2, 3], &[3, 2]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_to_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = reshape(&data, &[2, 2], &[4]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_to_3d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = reshape(&data, &[6], &[1, 2, 3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_identity() {
        let data = vec![1.0, 2.0, 3.0];
        let out = reshape(&data, &[3], &[3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_element_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(reshape(&data, &[3], &[2, 2]).is_err());
    }

    #[test]
    fn reshape_data_length_mismatch() {
        assert!(reshape(&[1.0, 2.0], &[3], &[3]).is_err());
    }

    // ── transpose_2d ──────────────────────────────────────────

    #[test]
    fn transpose_2d_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_2d(&data, 2, 3);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_square() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = transpose_2d(&data, 2, 2);
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_2d_single_element() {
        let out = transpose_2d(&[42.0], 1, 1);
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn transpose_2d_involution() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose_2d(&data, 2, 3);
        let tt = transpose_2d(&t, 3, 2);
        assert_eq!(tt, data);
    }

    #[test]
    fn transpose_2d_empty() {
        let out = transpose_2d(&[], 0, 5);
        assert!(out.is_empty());
    }

    // ── transpose_nd / permute ────────────────────────────────

    #[test]
    fn transpose_nd_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_nd(&data, &[2, 3], &[0, 1]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn transpose_nd_swap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let nd = transpose_nd(&data, &[2, 3], &[1, 0]).unwrap();
        let flat = transpose_2d(&data, 2, 3);
        assert_eq!(nd, flat);
    }

    #[test]
    fn transpose_nd_3d() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 2], &[1, 0, 2]).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn transpose_nd_bad_axes_length() {
        assert!(transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0]).is_err());
    }

    #[test]
    fn transpose_nd_duplicate_axis() {
        assert!(transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0, 0]).is_err());
    }

    #[test]
    fn transpose_nd_axis_out_of_range() {
        assert!(transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0, 5]).is_err());
    }

    #[test]
    fn transpose_nd_data_mismatch() {
        assert!(transpose_nd(&[1.0, 2.0], &[2, 3], &[1, 0]).is_err());
    }

    #[test]
    fn transpose_nd_empty() {
        let out = transpose_nd(&[], &[0, 3], &[1, 0]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn permute_is_transpose_nd() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a = transpose_nd(&data, &[2, 3, 4], &[2, 0, 1]).unwrap();
        let b = permute(&data, &[2, 3, 4], &[2, 0, 1]).unwrap();
        assert_eq!(a, b);
    }

    // ── squeeze ────────────────────────────────────────────────

    #[test]
    fn squeeze_removes_ones() {
        assert_eq!(squeeze(&[1, 3, 1, 4, 1]), vec![3, 4]);
    }

    #[test]
    fn squeeze_no_ones() {
        assert_eq!(squeeze(&[2, 3, 4]), vec![2, 3, 4]);
    }

    #[test]
    fn squeeze_all_ones() {
        assert!(squeeze(&[1, 1, 1]).is_empty());
    }

    #[test]
    fn squeeze_empty() {
        assert!(squeeze(&[]).is_empty());
    }

    // ── unsqueeze ──────────────────────────────────────────────

    #[test]
    fn unsqueeze_front() {
        assert_eq!(unsqueeze(&[2, 3], 0).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn unsqueeze_middle() {
        assert_eq!(unsqueeze(&[2, 3], 1).unwrap(), vec![2, 1, 3]);
    }

    #[test]
    fn unsqueeze_end() {
        assert_eq!(unsqueeze(&[2, 3], 2).unwrap(), vec![2, 3, 1]);
    }

    #[test]
    fn unsqueeze_out_of_range() {
        assert!(unsqueeze(&[2, 3], 5).is_err());
    }

    #[test]
    fn unsqueeze_empty_shape() {
        assert_eq!(unsqueeze(&[], 0).unwrap(), vec![1]);
    }

    #[test]
    fn unsqueeze_then_squeeze_roundtrip() {
        let orig = vec![2, 3, 4];
        let expanded = unsqueeze(&orig, 1).unwrap();
        assert_eq!(expanded, vec![2, 1, 3, 4]);
        let back = squeeze(&expanded);
        assert_eq!(back, orig);
    }

    // ── broadcast_shapes ──────────────────────────────────────

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[3, 4], &[]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_expand_1() {
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_different_ndim() {
        assert_eq!(broadcast_shapes(&[3, 1], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn broadcast_both_expand() {
        assert_eq!(broadcast_shapes(&[1, 3], &[4, 1]).unwrap(), vec![4, 3]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(broadcast_shapes(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn broadcast_empty_shapes() {
        assert_eq!(broadcast_shapes(&[], &[]).unwrap(), Vec::<usize>::new());
    }

    // ── flatten ────────────────────────────────────────────────

    #[test]
    fn flatten_middle() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let (out, shape) = flatten(&data, &[2, 3, 4], 1, 2).unwrap();
        assert_eq!(shape, vec![2, 12]);
        assert_eq!(out, data);
    }

    #[test]
    fn flatten_all() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let (_, shape) = flatten(&data, &[2, 3, 4], 0, 2).unwrap();
        assert_eq!(shape, vec![24]);
    }

    #[test]
    fn flatten_noop() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) = flatten(&data, &[2, 3], 0, 0).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn flatten_bad_range() {
        assert!(flatten(&[1.0; 6], &[2, 3], 1, 0).is_err());
    }

    #[test]
    fn flatten_dim_out_of_range() {
        assert!(flatten(&[1.0; 6], &[2, 3], 0, 5).is_err());
    }

    #[test]
    fn flatten_scalar_shape() {
        assert!(flatten(&[], &[], 0, 0).is_err());
    }

    // ── expand ─────────────────────────────────────────────────

    #[test]
    fn expand_no_broadcast() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = expand(&data, &[2, 3], &[2, 3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn expand_row_broadcast() {
        // [1, 3] → [2, 3]
        let data = vec![1.0, 2.0, 3.0];
        let out = expand(&data, &[1, 3], &[2, 3]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn expand_col_broadcast() {
        // [2, 1] → [2, 3]
        let data = vec![1.0, 2.0];
        let out = expand(&data, &[2, 1], &[2, 3]).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn expand_add_leading_dim() {
        // [3] → [2, 3]
        let data = vec![1.0, 2.0, 3.0];
        let out = expand(&data, &[3], &[2, 3]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn expand_scalar() {
        // [] → [2, 3]  (scalar = single element)
        let data = vec![7.0];
        let out = expand(&data, &[], &[2, 3]).unwrap();
        assert_eq!(out, vec![7.0; 6]);
    }

    #[test]
    fn expand_incompatible() {
        // [2, 3] → [2, 4]  (3 != 4 and 3 != 1)
        assert!(expand(&[1.0; 6], &[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn expand_target_fewer_dims() {
        // target must have >= source dims
        assert!(expand(&[1.0; 6], &[2, 3], &[6]).is_err());
    }

    #[test]
    fn expand_empty_target() {
        let out = expand(&[1.0], &[], &[0]).unwrap();
        assert!(out.is_empty());
    }

    // ── ShapeError Display ────────────────────────────────────

    #[test]
    fn shape_error_display() {
        let e = ShapeError::new("test message");
        assert_eq!(format!("{e}"), "ShapeError: test message");
    }

    #[test]
    fn shape_error_is_error_trait() {
        let e = ShapeError::new("x");
        // Prove it implements std::error::Error by calling .source().
        let _: Option<&dyn std::error::Error> = std::error::Error::source(&e);
    }
}
