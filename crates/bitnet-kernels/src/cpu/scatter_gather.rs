//! CPU-optimized scatter/gather operations for indexed tensor access.
//!
//! This module provides pure-Rust CPU implementations of scatter and
//! gather operations on multi-dimensional tensors, complementing the
//! CUDA stubs in [`crate::scatter_gather`].
//!
//! # Operations
//!
//! - [`cpu_gather`] — Collect elements from a source tensor by index.
//! - [`cpu_scatter`] — Distribute elements into a destination tensor by
//!   index with an optional reduction ([`ScatterReduce`]).
//! - [`cpu_scatter_add`] — Convenience wrapper for additive scatter.
//! - [`cpu_gather_nd`] / [`cpu_scatter_nd`] — N-dimensional variants
//!   that operate on arbitrary-rank tensors described by shape slices.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Reduction mode ─────────────────────────────────────────────────

/// Reduction applied at each destination index during scatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScatterReduce {
    /// Overwrite (last write wins).
    Assign,
    /// Accumulate (addition).
    Add,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
    /// Multiply the existing value by the source value.
    Mul,
}

impl ScatterReduce {
    /// Identity element for the reduction so that
    /// `apply(identity, x) == x` for all `x`.
    #[must_use]
    pub fn identity(self) -> f32 {
        match self {
            Self::Assign | Self::Add => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
            Self::Mul => 1.0,
        }
    }

    /// Combine `dst` and `src` according to the reduction.
    #[inline]
    fn apply(self, dst: f32, src: f32) -> f32 {
        match self {
            Self::Assign => src,
            Self::Add => dst + src,
            Self::Max => dst.max(src),
            Self::Min => dst.min(src),
            Self::Mul => dst * src,
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

/// Compute the flat-buffer length implied by a shape.
fn shape_len(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

/// Convert a multi-dimensional coordinate to a flat offset.
#[cfg(test)]
fn coord_to_offset(coord: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (dim, &size) in coord.iter().zip(shape.iter()).rev() {
        offset += dim * stride;
        stride *= size;
    }
    offset
}

/// Compute strides (row-major) for a shape.
fn strides_for(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ── 2-D gather ─────────────────────────────────────────────────────

/// Gather elements from `src` along `axis` using `indices`.
///
/// `src` is a 2-D tensor `[rows, cols]` stored in row-major order.
/// `indices` has shape `[idx_rows, idx_cols]`.
///
/// - **axis 0**: `output[i][j] = src[indices[i][j]][j]`
///   (requires `idx_cols == cols`).
/// - **axis 1**: `output[i][j] = src[i][indices[i][j]]`
///   (requires `idx_rows == rows`).
///
/// # Errors
///
/// Returns an error on shape mismatch or out-of-bounds indices when
/// `bounds_check` is `true`.
pub fn cpu_gather(
    src: &[f32],
    src_shape: [usize; 2],
    indices: &[usize],
    idx_shape: [usize; 2],
    axis: usize,
    bounds_check: bool,
    output: &mut [f32],
) -> Result<()> {
    let [s_rows, s_cols] = src_shape;
    let [i_rows, i_cols] = idx_shape;
    let out_len = i_rows * i_cols;

    if axis > 1 {
        return Err(invalid_args(format!("cpu_gather: axis must be 0 or 1, got {axis}")));
    }
    if src.len() < s_rows * s_cols {
        return Err(invalid_args(format!(
            "cpu_gather: src length {} < expected {}",
            src.len(),
            s_rows * s_cols
        )));
    }
    if indices.len() < out_len {
        return Err(invalid_args(format!(
            "cpu_gather: indices length {} < expected {out_len}",
            indices.len()
        )));
    }
    if output.len() < out_len {
        return Err(invalid_args(format!(
            "cpu_gather: output length {} < expected {out_len}",
            output.len()
        )));
    }
    if axis == 0 && i_cols != s_cols {
        return Err(invalid_args(format!(
            "cpu_gather axis 0: idx cols ({i_cols}) != src cols ({s_cols})"
        )));
    }
    if axis == 1 && i_rows != s_rows {
        return Err(invalid_args(format!(
            "cpu_gather axis 1: idx rows ({i_rows}) != src rows ({s_rows})"
        )));
    }

    let bound = if axis == 0 { s_rows } else { s_cols };

    for i in 0..i_rows {
        for j in 0..i_cols {
            let idx = indices[i * i_cols + j];
            if bounds_check && idx >= bound {
                return Err(invalid_args(format!(
                    "cpu_gather: index {idx} out of bounds \
                     for axis {axis} with size {bound}"
                )));
            }
            let clamped = idx.min(bound.saturating_sub(1));
            let src_off = if axis == 0 { clamped * s_cols + j } else { i * s_cols + clamped };
            output[i * i_cols + j] = src[src_off];
        }
    }
    Ok(())
}

// ── 2-D scatter ────────────────────────────────────────────────────

/// Scatter elements from `src` into `dst` along `axis` using
/// `indices`, applying `reduce` at each destination slot.
///
/// Layout conventions mirror [`cpu_gather`].
///
/// # Errors
///
/// Returns an error on shape mismatch or out-of-bounds indices when
/// `bounds_check` is `true`.
pub fn cpu_scatter(
    src: &[f32],
    indices: &[usize],
    idx_shape: [usize; 2],
    dst: &mut [f32],
    dst_shape: [usize; 2],
    axis: usize,
    reduce: ScatterReduce,
    bounds_check: bool,
) -> Result<()> {
    let [d_rows, d_cols] = dst_shape;
    let [i_rows, i_cols] = idx_shape;
    let elem_count = i_rows * i_cols;

    if axis > 1 {
        return Err(invalid_args(format!("cpu_scatter: axis must be 0 or 1, got {axis}")));
    }
    if src.len() < elem_count {
        return Err(invalid_args(format!(
            "cpu_scatter: src length {} < expected {elem_count}",
            src.len()
        )));
    }
    if indices.len() < elem_count {
        return Err(invalid_args(format!(
            "cpu_scatter: indices length {} < expected {elem_count}",
            indices.len()
        )));
    }
    if dst.len() < d_rows * d_cols {
        return Err(invalid_args(format!(
            "cpu_scatter: dst length {} < expected {}",
            dst.len(),
            d_rows * d_cols
        )));
    }
    if axis == 0 && i_cols != d_cols {
        return Err(invalid_args(format!(
            "cpu_scatter axis 0: idx cols ({i_cols}) != dst cols ({d_cols})"
        )));
    }
    if axis == 1 && i_rows != d_rows {
        return Err(invalid_args(format!(
            "cpu_scatter axis 1: idx rows ({i_rows}) != dst rows ({d_rows})"
        )));
    }

    let bound = if axis == 0 { d_rows } else { d_cols };

    for i in 0..i_rows {
        for j in 0..i_cols {
            let idx = indices[i * i_cols + j];
            if bounds_check && idx >= bound {
                return Err(invalid_args(format!(
                    "cpu_scatter: index {idx} out of bounds \
                     for axis {axis} with size {bound}"
                )));
            }
            let clamped = idx.min(bound.saturating_sub(1));
            let dst_off = if axis == 0 { clamped * d_cols + j } else { i * d_cols + clamped };
            let src_val = src[i * i_cols + j];
            dst[dst_off] = reduce.apply(dst[dst_off], src_val);
        }
    }
    Ok(())
}

// ── scatter-add convenience ────────────────────────────────────────

/// Additive scatter — accumulates `src` values into `dst` at the
/// given `indices` along `axis`.
///
/// Equivalent to [`cpu_scatter`] with [`ScatterReduce::Add`].
pub fn cpu_scatter_add(
    src: &[f32],
    indices: &[usize],
    idx_shape: [usize; 2],
    dst: &mut [f32],
    dst_shape: [usize; 2],
    axis: usize,
    bounds_check: bool,
) -> Result<()> {
    cpu_scatter(src, indices, idx_shape, dst, dst_shape, axis, ScatterReduce::Add, bounds_check)
}

// ── N-dimensional gather ───────────────────────────────────────────

/// N-dimensional gather along `axis`.
///
/// `src_shape` and `idx_shape` describe arbitrary-rank tensors.  The
/// gather semantics generalise from the 2-D case:
///
/// ```text
/// output[d0, .., d_{axis-1}, i, d_{axis+1}, ..] =
///     src[d0, .., d_{axis-1}, indices[d0, .., i, ..], d_{axis+1}, ..]
/// ```
///
/// # Errors
///
/// Returns an error if shapes are incompatible or an index exceeds the
/// source dimension along `axis` (when `bounds_check` is `true`).
pub fn cpu_gather_nd(
    src: &[f32],
    src_shape: &[usize],
    indices: &[usize],
    idx_shape: &[usize],
    axis: usize,
    bounds_check: bool,
    output: &mut [f32],
) -> Result<()> {
    let rank = src_shape.len();
    if rank == 0 {
        return Err(invalid_args("cpu_gather_nd: empty src_shape"));
    }
    if idx_shape.len() != rank {
        return Err(invalid_args(format!(
            "cpu_gather_nd: src rank ({rank}) != idx rank ({})",
            idx_shape.len()
        )));
    }
    if axis >= rank {
        return Err(invalid_args(format!("cpu_gather_nd: axis {axis} >= rank {rank}")));
    }
    // All dims except `axis` must match between src and idx.
    for (d, (&s, &i)) in src_shape.iter().zip(idx_shape.iter()).enumerate() {
        if d != axis && s != i {
            return Err(invalid_args(format!("cpu_gather_nd: dim {d}: src ({s}) != idx ({i})")));
        }
    }

    let total = shape_len(idx_shape);
    if indices.len() < total {
        return Err(invalid_args(format!(
            "cpu_gather_nd: indices len {} < {total}",
            indices.len()
        )));
    }
    if output.len() < total {
        return Err(invalid_args(format!("cpu_gather_nd: output len {} < {total}", output.len())));
    }
    if src.len() < shape_len(src_shape) {
        return Err(invalid_args("cpu_gather_nd: src too short"));
    }

    let src_strides = strides_for(src_shape);
    let idx_strides = strides_for(idx_shape);
    let bound = src_shape[axis];

    let mut coord = vec![0usize; rank];
    for flat in 0..total {
        // Decode flat → coord in idx_shape.
        {
            let mut rem = flat;
            for d in 0..rank {
                coord[d] = rem / idx_strides[d];
                rem %= idx_strides[d];
            }
        }

        let idx = indices[flat];
        if bounds_check && idx >= bound {
            return Err(invalid_args(format!(
                "cpu_gather_nd: index {idx} OOB for axis {axis} \
                 (size {bound})"
            )));
        }
        let clamped = idx.min(bound.saturating_sub(1));

        // Build source offset — same as coord but with axis
        // dimension replaced by the looked-up index.
        let mut src_off = 0usize;
        for d in 0..rank {
            let c = if d == axis { clamped } else { coord[d] };
            src_off += c * src_strides[d];
        }
        output[flat] = src[src_off];
    }
    Ok(())
}

// ── N-dimensional scatter ──────────────────────────────────────────

/// N-dimensional scatter along `axis` with reduction.
///
/// Generalises [`cpu_scatter`] to arbitrary rank.
///
/// # Errors
///
/// Returns an error if shapes are incompatible or an index exceeds
/// `dst_shape[axis]` (when `bounds_check` is `true`).
pub fn cpu_scatter_nd(
    src: &[f32],
    indices: &[usize],
    idx_shape: &[usize],
    dst: &mut [f32],
    dst_shape: &[usize],
    axis: usize,
    reduce: ScatterReduce,
    bounds_check: bool,
) -> Result<()> {
    let rank = dst_shape.len();
    if rank == 0 {
        return Err(invalid_args("cpu_scatter_nd: empty dst_shape"));
    }
    if idx_shape.len() != rank {
        return Err(invalid_args(format!(
            "cpu_scatter_nd: dst rank ({rank}) != idx rank ({})",
            idx_shape.len()
        )));
    }
    if axis >= rank {
        return Err(invalid_args(format!("cpu_scatter_nd: axis {axis} >= rank {rank}")));
    }
    for (d, (&ds, &is)) in dst_shape.iter().zip(idx_shape.iter()).enumerate() {
        if d != axis && ds != is {
            return Err(invalid_args(format!("cpu_scatter_nd: dim {d}: dst ({ds}) != idx ({is})")));
        }
    }

    let total = shape_len(idx_shape);
    if src.len() < total {
        return Err(invalid_args(format!("cpu_scatter_nd: src len {} < {total}", src.len())));
    }
    if indices.len() < total {
        return Err(invalid_args(format!(
            "cpu_scatter_nd: indices len {} < {total}",
            indices.len()
        )));
    }
    if dst.len() < shape_len(dst_shape) {
        return Err(invalid_args("cpu_scatter_nd: dst too short"));
    }

    let dst_strides = strides_for(dst_shape);
    let idx_strides = strides_for(idx_shape);
    let bound = dst_shape[axis];

    let mut coord = vec![0usize; rank];
    for flat in 0..total {
        {
            let mut rem = flat;
            for d in 0..rank {
                coord[d] = rem / idx_strides[d];
                rem %= idx_strides[d];
            }
        }

        let idx = indices[flat];
        if bounds_check && idx >= bound {
            return Err(invalid_args(format!(
                "cpu_scatter_nd: index {idx} OOB for axis {axis} \
                 (size {bound})"
            )));
        }
        let clamped = idx.min(bound.saturating_sub(1));

        let mut dst_off = 0usize;
        for d in 0..rank {
            let c = if d == axis { clamped } else { coord[d] };
            dst_off += c * dst_strides[d];
        }
        dst[dst_off] = reduce.apply(dst[dst_off], src[flat]);
    }
    Ok(())
}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for scatter/gather operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterGatherConfig {
    /// When `true`, return an error on out-of-bounds indices.
    /// When `false`, out-of-bounds indices are clamped silently.
    pub bounds_check: bool,
    /// Reduction mode for scatter operations.
    pub reduce: ScatterReduce,
}

impl Default for ScatterGatherConfig {
    fn default() -> Self {
        Self { bounds_check: true, reduce: ScatterReduce::Assign }
    }
}

impl ScatterGatherConfig {
    /// Create a config with bounds checking and the given reduction.
    #[must_use]
    pub fn with_reduce(reduce: ScatterReduce) -> Self {
        Self { bounds_check: true, reduce }
    }
}

// ── 1-D scatter ────────────────────────────────────────────────────

/// Scatter `values` into `data` at the given `indices`.
///
/// `indices` and `values` must have the same length.  Each
/// `data[indices[i]]` is overwritten with `values[i]` (last write wins
/// for duplicate indices).
///
/// # Errors
///
/// Returns an error when `indices` and `values` differ in length, or
/// an index is out of bounds.
pub fn scatter_1d(data: &mut [f32], indices: &[usize], values: &[f32]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(invalid_args(format!(
            "scatter_1d: indices len ({}) != values len ({})",
            indices.len(),
            values.len()
        )));
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "scatter_1d: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        data[idx] = values[i];
    }
    Ok(())
}

// ── 1-D gather ─────────────────────────────────────────────────────

/// Gather elements from `data` at the given `indices`, returning a
/// new vector.
///
/// # Errors
///
/// Returns an error if any index is out of bounds.
pub fn gather_1d(data: &[f32], indices: &[usize]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "gather_1d: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        out.push(data[idx]);
    }
    Ok(out)
}

// ── 2-D scatter (Vec-of-Vec) ───────────────────────────────────────

/// Scatter rows from `values` into `data` at the given row `indices`.
///
/// `indices` and `values` must have the same length.  Each row
/// `data[indices[i]]` is overwritten with `values[i]`.
///
/// # Errors
///
/// Returns an error when lengths mismatch, an index is out of bounds,
/// or inner row widths differ.
pub fn scatter_2d(data: &mut [Vec<f32>], indices: &[usize], values: &[Vec<f32>]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(invalid_args(format!(
            "scatter_2d: indices len ({}) != values len ({})",
            indices.len(),
            values.len()
        )));
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "scatter_2d: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        if data[idx].len() != values[i].len() {
            return Err(invalid_args(format!(
                "scatter_2d: row width mismatch at index {idx}: dst {} vs src {}",
                data[idx].len(),
                values[i].len()
            )));
        }
        data[idx].copy_from_slice(&values[i]);
    }
    Ok(())
}

// ── 2-D gather (Vec-of-Vec) ───────────────────────────────────────

/// Gather rows from `data` at the given row `indices`, returning
/// cloned rows.
///
/// # Errors
///
/// Returns an error if any index is out of bounds.
pub fn gather_2d(data: &[Vec<f32>], indices: &[usize]) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "gather_2d: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        out.push(data[idx].clone());
    }
    Ok(out)
}

// ── scatter_add (1-D convenience) ──────────────────────────────────

/// Scatter with addition: `data[indices[i]] += values[i]`.
///
/// Unlike [`scatter_1d`] this accumulates rather than overwrites.
///
/// # Errors
///
/// Returns an error when lengths mismatch or an index is out of bounds.
pub fn scatter_add(data: &mut [f32], indices: &[usize], values: &[f32]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(invalid_args(format!(
            "scatter_add: indices len ({}) != values len ({})",
            indices.len(),
            values.len()
        )));
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "scatter_add: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        data[idx] += values[i];
    }
    Ok(())
}

// ── scatter_max (1-D convenience) ──────────────────────────────────

/// Scatter with maximum: `data[indices[i]] = max(data[indices[i]], values[i])`.
///
/// # Errors
///
/// Returns an error when lengths mismatch or an index is out of bounds.
pub fn scatter_max(data: &mut [f32], indices: &[usize], values: &[f32]) -> Result<()> {
    if indices.len() != values.len() {
        return Err(invalid_args(format!(
            "scatter_max: indices len ({}) != values len ({})",
            indices.len(),
            values.len()
        )));
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= data.len() {
            return Err(invalid_args(format!(
                "scatter_max: index {idx} out of bounds for data len {}",
                data.len()
            )));
        }
        data[idx] = data[idx].max(values[i]);
    }
    Ok(())
}

// ── index_select ───────────────────────────────────────────────────

/// Select slices along the first dimension of a flat tensor.
///
/// `data` is a flat buffer representing a tensor whose first dimension
/// has `dim_size` elements; the remaining (inner) elements per slice
/// are inferred as `data.len() / dim_size`.
///
/// Returns the selected slices concatenated into a new `Vec`.
///
/// # Errors
///
/// Returns an error when `dim_size` is zero, `data.len()` is not
/// divisible by `dim_size`, or any index is out of bounds.
pub fn index_select(data: &[f32], dim_size: usize, indices: &[usize]) -> Result<Vec<f32>> {
    if dim_size == 0 {
        return Err(invalid_args("index_select: dim_size must be > 0"));
    }
    if !data.len().is_multiple_of(dim_size) {
        return Err(invalid_args(format!(
            "index_select: data len ({}) not divisible by dim_size ({dim_size})",
            data.len()
        )));
    }
    let inner = data.len() / dim_size;
    let mut out = Vec::with_capacity(indices.len() * inner);
    for &idx in indices {
        if idx >= dim_size {
            return Err(invalid_args(format!(
                "index_select: index {idx} out of bounds for dim_size {dim_size}"
            )));
        }
        let start = idx * inner;
        out.extend_from_slice(&data[start..start + inner]);
    }
    Ok(out)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScatterReduce ──────────────────────────────────────────────

    #[test]
    fn reduce_identity_values() {
        assert_eq!(ScatterReduce::Assign.identity(), 0.0);
        assert_eq!(ScatterReduce::Add.identity(), 0.0);
        assert_eq!(ScatterReduce::Max.identity(), f32::NEG_INFINITY);
        assert_eq!(ScatterReduce::Min.identity(), f32::INFINITY);
        assert_eq!(ScatterReduce::Mul.identity(), 1.0);
    }

    #[test]
    fn reduce_apply_all_modes() {
        let r = ScatterReduce::Assign;
        assert_eq!(r.apply(10.0, 5.0), 5.0);
        assert_eq!(ScatterReduce::Add.apply(10.0, 5.0), 15.0);
        assert_eq!(ScatterReduce::Max.apply(3.0, 7.0), 7.0);
        assert_eq!(ScatterReduce::Min.apply(3.0, 7.0), 3.0);
        assert_eq!(ScatterReduce::Mul.apply(4.0, 3.0), 12.0);
    }

    // ── helpers ────────────────────────────────────────────────────

    #[test]
    fn helper_shape_len() {
        assert_eq!(shape_len(&[2, 3, 4]), 24);
        assert_eq!(shape_len(&[1]), 1);
        assert_eq!(shape_len(&[]), 1); // empty product = 1
    }

    #[test]
    fn helper_coord_to_offset() {
        // shape [2, 3]: row-major strides [3, 1]
        assert_eq!(coord_to_offset(&[0, 0], &[2, 3]), 0);
        assert_eq!(coord_to_offset(&[1, 2], &[2, 3]), 5);
    }

    #[test]
    fn helper_strides() {
        assert_eq!(strides_for(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides_for(&[5]), vec![1]);
    }

    // ── cpu_gather 2-D ─────────────────────────────────────────────

    #[test]
    fn gather_axis0_basic() {
        // src 3×2:  [[10,11],[20,21],[30,31]]
        let src = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let indices = [2, 0]; // row2-col0, row0-col1
        let mut out = [0.0f32; 2];
        cpu_gather(&src, [3, 2], &indices, [1, 2], 0, true, &mut out).unwrap();
        assert_eq!(out, [30.0, 11.0]);
    }

    #[test]
    fn gather_axis1_basic() {
        let src: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let indices = [3, 1, 0, 2]; // 2×2
        let mut out = [0.0f32; 4];
        cpu_gather(&src, [2, 4], &indices, [2, 2], 1, true, &mut out).unwrap();
        assert_eq!(out, [3.0, 1.0, 4.0, 6.0]);
    }

    #[test]
    fn gather_bounds_check_error() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [5, 0]; // OOB
        let mut out = [0.0f32; 2];
        let r = cpu_gather(&src, [2, 2], &indices, [1, 2], 0, true, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn gather_clamp_on_oob_when_unchecked() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [99, 0]; // OOB, clamped to row 1
        let mut out = [0.0f32; 2];
        cpu_gather(&src, [2, 2], &indices, [1, 2], 0, false, &mut out).unwrap();
        assert_eq!(out[0], 3.0); // row 1, col 0
        assert_eq!(out[1], 2.0); // row 0, col 1
    }

    #[test]
    fn gather_single_element() {
        let src = [42.0];
        let mut out = [0.0f32; 1];
        cpu_gather(&src, [1, 1], &[0], [1, 1], 0, true, &mut out).unwrap();
        assert_eq!(out[0], 42.0);
    }

    #[test]
    fn gather_shape_mismatch_axis0() {
        let src = [0.0f32; 6]; // 3×2
        let indices = [0, 1, 2]; // 1×3, cols mismatch
        let mut out = [0.0f32; 3];
        let r = cpu_gather(&src, [3, 2], &indices, [1, 3], 0, true, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn gather_invalid_axis() {
        let src = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let r = cpu_gather(&src, [2, 2], &[0; 4], [2, 2], 2, true, &mut out);
        assert!(r.is_err());
    }

    // ── cpu_scatter 2-D ────────────────────────────────────────────

    #[test]
    fn scatter_assign_axis0() {
        let src = [10.0, 11.0];
        let indices = [2, 2]; // put into row 2
        let mut dst = [0.0f32; 6]; // 3×2
        cpu_scatter(&src, &indices, [1, 2], &mut dst, [3, 2], 0, ScatterReduce::Assign, true)
            .unwrap();
        assert_eq!(dst, [0.0, 0.0, 0.0, 0.0, 10.0, 11.0]);
    }

    #[test]
    fn scatter_add_accumulates() {
        let src = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let indices = [0, 0, 0, 0]; // both rows → row 0
        let mut dst = [0.0f32; 4]; // 2×2
        cpu_scatter(&src, &indices, [2, 2], &mut dst, [2, 2], 0, ScatterReduce::Add, true).unwrap();
        assert_eq!(dst[0], 4.0); // 1 + 3
        assert_eq!(dst[1], 6.0); // 2 + 4
    }

    #[test]
    fn scatter_max_keeps_maximum() {
        let src = [5.0, 1.0, 3.0, 9.0];
        let indices = [0, 0, 0, 0];
        let mut dst = [0.0f32; 4];
        cpu_scatter(&src, &indices, [2, 2], &mut dst, [2, 2], 0, ScatterReduce::Max, true).unwrap();
        assert_eq!(dst[0], 5.0);
        assert_eq!(dst[1], 9.0);
    }

    #[test]
    fn scatter_min_keeps_minimum() {
        let src = [5.0, 1.0, 3.0, 9.0];
        let indices = [0, 0, 0, 0];
        let mut dst = [f32::INFINITY; 4];
        cpu_scatter(&src, &indices, [2, 2], &mut dst, [2, 2], 0, ScatterReduce::Min, true).unwrap();
        assert_eq!(dst[0], 3.0);
        assert_eq!(dst[1], 1.0);
    }

    #[test]
    fn scatter_mul_multiplies() {
        let src = [2.0, 3.0];
        let indices = [0, 0];
        let mut dst = [5.0, 7.0, 0.0, 0.0]; // 2×2
        cpu_scatter(&src, &indices, [1, 2], &mut dst, [2, 2], 0, ScatterReduce::Mul, true).unwrap();
        assert_eq!(dst[0], 10.0); // 5 * 2
        assert_eq!(dst[1], 21.0); // 7 * 3
    }

    #[test]
    fn scatter_oob_checked() {
        let src = [1.0, 2.0];
        let indices = [99, 0];
        let mut dst = [0.0f32; 4];
        let r =
            cpu_scatter(&src, &indices, [1, 2], &mut dst, [2, 2], 0, ScatterReduce::Assign, true);
        assert!(r.is_err());
    }

    #[test]
    fn scatter_axis1() {
        let src = [10.0, 20.0]; // 2×1
        let indices = [2, 0]; // r0→c2, r1→c0
        let mut dst = [0.0f32; 6]; // 2×3
        cpu_scatter(&src, &indices, [2, 1], &mut dst, [2, 3], 1, ScatterReduce::Assign, true)
            .unwrap();
        assert_eq!(dst, [0.0, 0.0, 10.0, 20.0, 0.0, 0.0]);
    }

    // ── cpu_scatter_add ────────────────────────────────────────────

    #[test]
    fn scatter_add_convenience() {
        let src = [1.0, 2.0, 3.0, 4.0];
        let indices = [0, 0, 0, 0];
        let mut dst = [0.0f32; 4];
        cpu_scatter_add(&src, &indices, [2, 2], &mut dst, [2, 2], 0, true).unwrap();
        assert_eq!(dst[0], 4.0);
        assert_eq!(dst[1], 6.0);
    }

    // ── cpu_gather_nd ──────────────────────────────────────────────

    #[test]
    fn gather_nd_3d_axis0() {
        // src shape [3, 2, 2] = 12 elements
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        // idx shape [2, 2, 2], gather along axis 0
        let indices = [2, 1, 0, 0, 1, 2, 0, 1];
        let mut out = [0.0f32; 8];
        cpu_gather_nd(&src, &[3, 2, 2], &indices, &[2, 2, 2], 0, true, &mut out).unwrap();
        // output[0,0,0] = src[2,0,0] = 8
        assert_eq!(out[0], 8.0);
        // output[0,0,1] = src[1,0,1] = 5
        assert_eq!(out[1], 5.0);
    }

    #[test]
    fn gather_nd_rank_mismatch() {
        let src = [0.0f32; 6];
        let mut out = [0.0f32; 6];
        let r = cpu_gather_nd(&src, &[2, 3], &[0; 6], &[2, 3, 1], 0, true, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn gather_nd_axis_oob() {
        let src = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        let r = cpu_gather_nd(&src, &[2, 2], &[0; 4], &[2, 2], 5, true, &mut out);
        assert!(r.is_err());
    }

    // ── cpu_scatter_nd ─────────────────────────────────────────────

    #[test]
    fn scatter_nd_3d_add() {
        // dst shape [2, 2, 2] = 8 elements
        let mut dst = [0.0f32; 8];
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // All scatter into dst axis-0 index 0
        let indices = [0, 0, 0, 0, 0, 0, 0, 0];
        cpu_scatter_nd(
            &src,
            &indices,
            &[2, 2, 2],
            &mut dst,
            &[2, 2, 2],
            0,
            ScatterReduce::Add,
            true,
        )
        .unwrap();
        // dst[0,0,0] = 1 + 5 = 6
        assert_eq!(dst[0], 6.0);
        // dst[0,1,1] = 4 + 8 = 12
        assert_eq!(dst[3], 12.0);
    }

    #[test]
    fn scatter_nd_bounds_error() {
        let mut dst = [0.0f32; 4];
        let src = [1.0; 4];
        let indices = [99, 0, 0, 0]; // OOB
        let r = cpu_scatter_nd(
            &src,
            &indices,
            &[2, 2],
            &mut dst,
            &[2, 2],
            0,
            ScatterReduce::Assign,
            true,
        );
        assert!(r.is_err());
    }

    // ── Round-trip: scatter then gather ─────────────────────────────

    #[test]
    fn scatter_then_gather_roundtrip() {
        // Scatter src into dst, then gather back — should recover
        // the original values for non-overlapping indices.
        let src = [10.0, 20.0, 30.0, 40.0]; // 2×2
        // Row 0 → dst row 1, row 1 → dst row 0
        let indices = [1, 1, 0, 0];
        let mut dst = [0.0f32; 6]; // 3×2
        cpu_scatter(&src, &indices, [2, 2], &mut dst, [3, 2], 0, ScatterReduce::Assign, true)
            .unwrap();

        // Gather back: row 1 then row 0 from dst
        let gather_idx = [1, 1, 0, 0];
        let mut recovered = [0.0f32; 4];
        cpu_gather(&dst, [3, 2], &gather_idx, [2, 2], 0, true, &mut recovered).unwrap();
        assert_eq!(recovered, src);
    }

    // ── Large tensor stress test ───────────────────────────────────

    #[test]
    fn gather_large_tensor() {
        let rows = 128;
        let cols = 64;
        let src: Vec<f32> = (0..(rows * cols) as u32).map(|x| x as f32).collect();
        let n_sel = 64;
        let indices: Vec<usize> = (0..n_sel).flat_map(|i| vec![i * 2; cols]).collect();
        let mut out = vec![0.0f32; n_sel * cols];
        cpu_gather(&src, [rows, cols], &indices, [n_sel, cols], 0, true, &mut out).unwrap();
        for j in 0..cols {
            assert_eq!(out[j], src[j]); // row 0
        }
        for j in 0..cols {
            assert_eq!(out[cols + j], src[2 * cols + j]); // row 2
        }
    }

    // ── ScatterGatherConfig ────────────────────────────────────────

    #[test]
    fn config_default() {
        let cfg = ScatterGatherConfig::default();
        assert!(cfg.bounds_check);
        assert_eq!(cfg.reduce, ScatterReduce::Assign);
    }

    #[test]
    fn config_with_reduce() {
        let cfg = ScatterGatherConfig::with_reduce(ScatterReduce::Add);
        assert!(cfg.bounds_check);
        assert_eq!(cfg.reduce, ScatterReduce::Add);
    }

    // ── scatter_1d ─────────────────────────────────────────────────

    #[test]
    fn scatter_1d_basic() {
        let mut data = [0.0f32; 5];
        scatter_1d(&mut data, &[1, 3], &[10.0, 30.0]).unwrap();
        assert_eq!(data, [0.0, 10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn scatter_1d_overwrites_duplicate_indices() {
        let mut data = [0.0f32; 3];
        scatter_1d(&mut data, &[1, 1], &[10.0, 20.0]).unwrap();
        assert_eq!(data[1], 20.0); // last write wins
    }

    #[test]
    fn scatter_1d_length_mismatch() {
        let mut data = [0.0f32; 3];
        assert!(scatter_1d(&mut data, &[0, 1], &[1.0]).is_err());
    }

    #[test]
    fn scatter_1d_oob() {
        let mut data = [0.0f32; 3];
        assert!(scatter_1d(&mut data, &[5], &[1.0]).is_err());
    }

    // ── gather_1d ──────────────────────────────────────────────────

    #[test]
    fn gather_1d_basic() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let out = gather_1d(&data, &[4, 0, 2]).unwrap();
        assert_eq!(out, vec![50.0, 10.0, 30.0]);
    }

    #[test]
    fn gather_1d_empty_indices() {
        let data = [1.0, 2.0];
        let out = gather_1d(&data, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn gather_1d_oob() {
        let data = [1.0, 2.0];
        assert!(gather_1d(&data, &[5]).is_err());
    }

    #[test]
    fn gather_1d_duplicate_indices() {
        let data = [10.0, 20.0, 30.0];
        let out = gather_1d(&data, &[1, 1, 1]).unwrap();
        assert_eq!(out, vec![20.0, 20.0, 20.0]);
    }

    // ── scatter_2d ─────────────────────────────────────────────────

    #[test]
    fn scatter_2d_basic() {
        let mut data = vec![vec![0.0; 3]; 4];
        let values = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        scatter_2d(&mut data, &[1, 3], &values).unwrap();
        assert_eq!(data[1], vec![1.0, 2.0, 3.0]);
        assert_eq!(data[3], vec![4.0, 5.0, 6.0]);
        assert_eq!(data[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_2d_oob() {
        let mut data = vec![vec![0.0; 2]; 2];
        assert!(scatter_2d(&mut data, &[5], &[vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn scatter_2d_width_mismatch() {
        let mut data = vec![vec![0.0; 3]; 2];
        assert!(scatter_2d(&mut data, &[0], &[vec![1.0, 2.0]]).is_err());
    }

    // ── gather_2d ──────────────────────────────────────────────────

    #[test]
    fn gather_2d_basic() {
        let data = vec![vec![10.0, 11.0], vec![20.0, 21.0], vec![30.0, 31.0]];
        let out = gather_2d(&data, &[2, 0]).unwrap();
        assert_eq!(out, vec![vec![30.0, 31.0], vec![10.0, 11.0]]);
    }

    #[test]
    fn gather_2d_oob() {
        let data = vec![vec![1.0]; 2];
        assert!(gather_2d(&data, &[3]).is_err());
    }

    #[test]
    fn scatter_2d_then_gather_2d_roundtrip() {
        let mut data = vec![vec![0.0; 2]; 4];
        let values = vec![vec![7.0, 8.0], vec![9.0, 10.0]];
        scatter_2d(&mut data, &[1, 3], &values).unwrap();
        let recovered = gather_2d(&data, &[1, 3]).unwrap();
        assert_eq!(recovered, values);
    }

    // ── scatter_add (1-D) ──────────────────────────────────────────

    #[test]
    fn scatter_add_1d_basic() {
        let mut data = [0.0f32; 4];
        scatter_add(&mut data, &[1, 1, 2], &[10.0, 5.0, 3.0]).unwrap();
        assert_eq!(data, [0.0, 15.0, 3.0, 0.0]);
    }

    #[test]
    fn scatter_add_1d_oob() {
        let mut data = [0.0f32; 2];
        assert!(scatter_add(&mut data, &[5], &[1.0]).is_err());
    }

    // ── scatter_max (1-D) ──────────────────────────────────────────

    #[test]
    fn scatter_max_1d_basic() {
        let mut data = [0.0f32; 3];
        scatter_max(&mut data, &[1, 1, 1], &[5.0, 3.0, 9.0]).unwrap();
        assert_eq!(data[1], 9.0);
    }

    #[test]
    fn scatter_max_1d_preserves_existing() {
        let mut data = [100.0, 0.0];
        scatter_max(&mut data, &[0], &[50.0]).unwrap();
        assert_eq!(data[0], 100.0); // existing is larger
    }

    #[test]
    fn scatter_max_1d_oob() {
        let mut data = [0.0f32; 2];
        assert!(scatter_max(&mut data, &[5], &[1.0]).is_err());
    }

    #[test]
    fn scatter_max_1d_length_mismatch() {
        let mut data = [0.0f32; 2];
        assert!(scatter_max(&mut data, &[0, 1], &[1.0]).is_err());
    }

    // ── index_select ───────────────────────────────────────────────

    #[test]
    fn index_select_basic() {
        // 3 rows of 2 elements: [[0,1],[2,3],[4,5]]
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = index_select(&data, 3, &[2, 0]).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn index_select_single_inner() {
        let data = [10.0, 20.0, 30.0];
        let out = index_select(&data, 3, &[1]).unwrap();
        assert_eq!(out, vec![20.0]);
    }

    #[test]
    fn index_select_oob() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!(index_select(&data, 2, &[5]).is_err());
    }

    #[test]
    fn index_select_zero_dim() {
        let data = [1.0];
        assert!(index_select(&data, 0, &[0]).is_err());
    }

    #[test]
    fn index_select_indivisible() {
        let data = [1.0, 2.0, 3.0];
        assert!(index_select(&data, 2, &[0]).is_err()); // 3 % 2 != 0
    }

    #[test]
    fn index_select_empty_indices() {
        let data = [1.0, 2.0];
        let out = index_select(&data, 2, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn index_select_duplicate_indices() {
        let data = [10.0, 20.0, 30.0];
        let out = index_select(&data, 3, &[1, 1, 1]).unwrap();
        assert_eq!(out, vec![20.0, 20.0, 20.0]);
    }

    // ── 1-D scatter/gather roundtrip ───────────────────────────────

    #[test]
    fn scatter_1d_then_gather_1d_roundtrip() {
        let mut data = [0.0f32; 5];
        let indices = [0, 2, 4];
        let values = [10.0, 20.0, 30.0];
        scatter_1d(&mut data, &indices, &values).unwrap();
        let recovered = gather_1d(&data, &indices).unwrap();
        assert_eq!(recovered, values.to_vec());
    }
}
