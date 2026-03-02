//! GPU-accelerated tensor reshape, transpose, permute, and view operations.
//!
//! Provides [`TensorShape`] for tracking dimensions/strides/contiguity,
//! [`ReshapeOp`] for declarative reshape operations, and kernel structs
//! ([`TransposeKernel`], [`PermuteKernel`], [`ConcatKernel`], [`SplitKernel`])
//! with CPU reference implementations and OpenCL kernel sources for A770.

use std::fmt;

// ── OpenCL kernel sources ────────────────────────────────────────

/// OpenCL kernel source for shared-memory tiled transpose.
pub const TRANSPOSE_CL: &str = r#"
__kernel void transpose_2d(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    __local float tile[16][17]; // +1 to avoid bank conflicts
    int bx = get_group_id(0) * 16;
    int by = get_group_id(1) * 16;
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int ix = bx + tx;
    int iy = by + ty;
    if (ix < cols && iy < rows)
        tile[ty][tx] = input[iy * cols + ix];
    barrier(CLK_LOCAL_MEM_FENCE);

    int ox = by + tx;
    int oy = bx + ty;
    if (ox < rows && oy < cols)
        output[oy * rows + ox] = tile[tx][ty];
}
"#;

/// OpenCL kernel source for arbitrary dimension permutation.
pub const PERMUTE_CL: &str = r#"
__kernel void permute_nd(
    __global const float* input,
    __global float* output,
    __global const int* in_strides,
    __global const int* out_strides,
    __global const int* perm,
    const int ndim,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;

    int remaining = gid;
    int in_offset = 0;
    for (int d = 0; d < ndim; d++) {
        int idx = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        in_offset += idx * in_strides[perm[d]];
    }
    output[gid] = input[in_offset];
}
"#;

/// OpenCL kernel source for tensor concatenation along a dimension.
pub const CONCAT_CL: &str = r#"
__kernel void concat_1d(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int a_len,
    const int b_len)
{
    int gid = get_global_id(0);
    if (gid < a_len)
        output[gid] = a[gid];
    else if (gid < a_len + b_len)
        output[gid] = b[gid - a_len];
}
"#;

// ── TensorShape ──────────────────────────────────────────────────

/// Describes the shape, strides, offset, and contiguity of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes (e.g. `[2, 3, 4]`).
    pub dims: Vec<usize>,
    /// Element strides per dimension (row-major by default).
    pub strides: Vec<usize>,
    /// Element offset into the underlying buffer.
    pub offset: usize,
    /// Whether elements are laid out contiguously in memory.
    pub contiguous: bool,
}

impl TensorShape {
    /// Create a new contiguous shape with row-major strides.
    pub fn new(dims: Vec<usize>) -> Self {
        let strides = Self::compute_row_major_strides(&dims);
        Self { dims, strides, offset: 0, contiguous: true }
    }

    /// Create a shape with explicit strides.
    pub fn with_strides(dims: Vec<usize>, strides: Vec<usize>) -> Self {
        let contiguous = Self::check_contiguous(&dims, &strides);
        Self { dims, strides, offset: 0, contiguous }
    }

    /// Create a scalar (0-dimensional) shape.
    pub fn scalar() -> Self {
        Self { dims: vec![], strides: vec![], offset: 0, contiguous: true }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            return 1; // scalar
        }
        self.dims.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Compute row-major strides for the given dimensions.
    pub fn compute_row_major_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![0usize; dims.len()];
        let mut stride = 1usize;
        for i in (0..dims.len()).rev() {
            strides[i] = stride;
            stride = stride.saturating_mul(dims[i]);
        }
        strides
    }

    /// Check whether the given dims/strides combination is contiguous.
    fn check_contiguous(dims: &[usize], strides: &[usize]) -> bool {
        if dims.len() != strides.len() {
            return false;
        }
        let expected = Self::compute_row_major_strides(dims);
        strides == expected.as_slice()
    }

    /// Recompute the `contiguous` flag from current dims/strides.
    pub fn refresh_contiguity(&mut self) {
        self.contiguous = Self::check_contiguous(&self.dims, &self.strides);
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorShape({:?}, strides={:?})", self.dims, self.strides)
    }
}

// ── ReshapeOp ────────────────────────────────────────────────────

/// Declarative tensor reshape operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReshapeOp {
    /// Change shape without changing element order (view).
    Reshape(Vec<usize>),
    /// Swap two dimensions.
    Transpose(usize, usize),
    /// Arbitrary dimension permutation.
    Permute(Vec<usize>),
    /// Remove a size-1 dimension.
    Squeeze(usize),
    /// Insert a size-1 dimension.
    Unsqueeze(usize),
    /// Flatten dimensions `start..=end` into one.
    Flatten { start: usize, end: usize },
    /// Split a dimension into chunks of `chunk_size`.
    Split { dim: usize, chunk_size: usize },
    /// Concatenate along a dimension.
    Concat { dim: usize },
}

impl fmt::Display for ReshapeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reshape(dims) => write!(f, "Reshape({dims:?})"),
            Self::Transpose(a, b) => write!(f, "Transpose({a}, {b})"),
            Self::Permute(p) => write!(f, "Permute({p:?})"),
            Self::Squeeze(d) => write!(f, "Squeeze({d})"),
            Self::Unsqueeze(d) => write!(f, "Unsqueeze({d})"),
            Self::Flatten { start, end } => write!(f, "Flatten({start}..={end})"),
            Self::Split { dim, chunk_size } => {
                write!(f, "Split(dim={dim}, chunk={chunk_size})")
            }
            Self::Concat { dim } => write!(f, "Concat(dim={dim})"),
        }
    }
}

// ── StrideComputer ───────────────────────────────────────────────

/// Computes output strides from input shape + operation.
pub struct StrideComputer;

impl StrideComputer {
    /// Compute the output shape produced by applying `op` to `input`.
    ///
    /// Returns `None` when the operation is invalid for the given input.
    pub fn compute(input: &TensorShape, op: &ReshapeOp) -> Option<TensorShape> {
        match op {
            ReshapeOp::Reshape(new_dims) => {
                let new_numel: usize = new_dims.iter().product();
                if new_numel != input.numel() {
                    return None;
                }
                Some(TensorShape::new(new_dims.clone()))
            }
            ReshapeOp::Transpose(a, b) => {
                let ndim = input.ndim();
                if *a >= ndim || *b >= ndim {
                    return None;
                }
                let mut dims = input.dims.clone();
                let mut strides = input.strides.clone();
                dims.swap(*a, *b);
                strides.swap(*a, *b);
                let contiguous =
                    TensorShape::check_contiguous(&dims, &strides);
                Some(TensorShape { dims, strides, offset: input.offset, contiguous })
            }
            ReshapeOp::Permute(perm) => {
                if perm.len() != input.ndim() {
                    return None;
                }
                let mut seen = vec![false; perm.len()];
                for &p in perm {
                    if p >= perm.len() {
                        return None;
                    }
                    seen[p] = true;
                }
                if seen.iter().any(|s| !s) {
                    return None;
                }
                let dims: Vec<usize> = perm.iter().map(|&p| input.dims[p]).collect();
                let strides: Vec<usize> =
                    perm.iter().map(|&p| input.strides[p]).collect();
                let contiguous =
                    TensorShape::check_contiguous(&dims, &strides);
                Some(TensorShape { dims, strides, offset: input.offset, contiguous })
            }
            ReshapeOp::Squeeze(d) => {
                if *d >= input.ndim() || input.dims[*d] != 1 {
                    return None;
                }
                let mut dims = input.dims.clone();
                let mut strides = input.strides.clone();
                dims.remove(*d);
                strides.remove(*d);
                let contiguous =
                    TensorShape::check_contiguous(&dims, &strides);
                Some(TensorShape { dims, strides, offset: input.offset, contiguous })
            }
            ReshapeOp::Unsqueeze(d) => {
                if *d > input.ndim() {
                    return None;
                }
                let mut dims = input.dims.clone();
                let mut strides = input.strides.clone();
                let stride_val =
                    if *d < input.strides.len() { input.strides[*d] } else { 1 };
                dims.insert(*d, 1);
                strides.insert(*d, stride_val);
                let contiguous =
                    TensorShape::check_contiguous(&dims, &strides);
                Some(TensorShape { dims, strides, offset: input.offset, contiguous })
            }
            ReshapeOp::Flatten { start, end } => {
                if *start > *end || *end >= input.ndim() {
                    return None;
                }
                let flat_dim: usize = input.dims[*start..=*end].iter().product();
                let mut dims = Vec::with_capacity(
                    input.ndim() - (end - start),
                );
                dims.extend_from_slice(&input.dims[..*start]);
                dims.push(flat_dim);
                dims.extend_from_slice(&input.dims[end + 1..]);
                Some(TensorShape::new(dims))
            }
            ReshapeOp::Split { dim, chunk_size } => {
                if *dim >= input.ndim() || *chunk_size == 0 {
                    return None;
                }
                let dim_size = input.dims[*dim];
                if !dim_size.is_multiple_of(*chunk_size) {
                    return None;
                }
                let num_chunks = dim_size / chunk_size;
                let mut dims = Vec::with_capacity(input.ndim() + 1);
                dims.extend_from_slice(&input.dims[..*dim]);
                dims.push(num_chunks);
                dims.push(*chunk_size);
                dims.extend_from_slice(&input.dims[dim + 1..]);
                Some(TensorShape::new(dims))
            }
            ReshapeOp::Concat { dim } => {
                // Concat metadata only — actual data merge via ConcatKernel.
                if *dim >= input.ndim() {
                    return None;
                }
                Some(input.clone())
            }
        }
    }
}

// ── TensorReshaper ───────────────────────────────────────────────

/// Performs reshape operations with contiguity tracking.
///
/// Applies [`ReshapeOp`] to data vectors using CPU reference
/// implementations. The data is always `f32` in row-major order.
pub struct TensorReshaper;

impl TensorReshaper {
    /// Apply a reshape operation, returning the new data and shape.
    ///
    /// For operations that are purely metadata (Reshape, Squeeze,
    /// Unsqueeze, Flatten), data is returned unchanged and only the
    /// shape is updated. For Transpose and Permute, data is physically
    /// rearranged.
    pub fn apply(
        data: &[f32],
        shape: &TensorShape,
        op: &ReshapeOp,
    ) -> Option<(Vec<f32>, TensorShape)> {
        let new_shape = StrideComputer::compute(shape, op)?;
        match op {
            ReshapeOp::Reshape(_)
            | ReshapeOp::Squeeze(_)
            | ReshapeOp::Unsqueeze(_)
            | ReshapeOp::Flatten { .. } => {
                // Metadata-only: data order unchanged.
                Some((data.to_vec(), new_shape))
            }
            ReshapeOp::Transpose(_, _) | ReshapeOp::Permute(_) => {
                let out = permute_ref(data, shape, &new_shape, op);
                Some((out, TensorShape::new(new_shape.dims.clone())))
            }
            ReshapeOp::Split { .. } => {
                Some((data.to_vec(), new_shape))
            }
            ReshapeOp::Concat { .. } => {
                // Single-tensor concat is identity.
                Some((data.to_vec(), new_shape))
            }
        }
    }
}

// ── CPU reference: permute ───────────────────────────────────────

/// CPU reference implementation for permute / transpose.
fn permute_ref(
    data: &[f32],
    in_shape: &TensorShape,
    out_shape: &TensorShape,
    op: &ReshapeOp,
) -> Vec<f32> {
    let perm: Vec<usize> = match op {
        ReshapeOp::Transpose(a, b) => {
            let mut p: Vec<usize> = (0..in_shape.ndim()).collect();
            p.swap(*a, *b);
            p
        }
        ReshapeOp::Permute(p) => p.clone(),
        _ => return data.to_vec(),
    };

    let numel = in_shape.numel();
    let mut output = vec![0.0f32; numel];
    let ndim = in_shape.ndim();
    let out_strides = TensorShape::compute_row_major_strides(&out_shape.dims);

    for (linear, out_val) in output.iter_mut().enumerate() {
        // Decompose linear index into output coordinates.
        let mut remaining = linear;
        let mut in_offset = 0usize;
        for d in 0..ndim {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            in_offset += coord * in_shape.strides[perm[d]];
        }
        *out_val = data[in_offset];
    }
    output
}

// ── TransposeKernel ──────────────────────────────────────────────

/// Efficient 2D matrix transpose using shared-memory tiling.
///
/// The OpenCL implementation uses a 16×17 tile (padded column to avoid
/// bank conflicts) stored in `__local` memory.
pub struct TransposeKernel;

impl TransposeKernel {
    /// OpenCL source for this kernel.
    pub const SOURCE: &'static str = TRANSPOSE_CL;

    /// CPU reference: transpose a `[rows, cols]` matrix to `[cols, rows]`.
    pub fn execute_ref(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Option<()> {
        if input.len() < rows * cols || output.len() < rows * cols {
            return None;
        }
        for r in 0..rows {
            for c in 0..cols {
                output[c * rows + r] = input[r * cols + c];
            }
        }
        Some(())
    }
}

// ── PermuteKernel ────────────────────────────────────────────────

/// Arbitrary dimension permutation kernel.
///
/// Reorders tensor data according to a permutation vector.
pub struct PermuteKernel;

impl PermuteKernel {
    /// OpenCL source for this kernel.
    pub const SOURCE: &'static str = PERMUTE_CL;

    /// CPU reference: permute `data` from `in_shape` according to `perm`.
    pub fn execute_ref(
        data: &[f32],
        in_shape: &TensorShape,
        perm: &[usize],
    ) -> Option<Vec<f32>> {
        if perm.len() != in_shape.ndim() {
            return None;
        }
        let op = ReshapeOp::Permute(perm.to_vec());
        let out_shape = StrideComputer::compute(in_shape, &op)?;
        Some(permute_ref(data, in_shape, &out_shape, &op))
    }
}

// ── ConcatKernel ─────────────────────────────────────────────────

/// Concatenates tensors along a dimension.
pub struct ConcatKernel;

impl ConcatKernel {
    /// OpenCL source for this kernel.
    pub const SOURCE: &'static str = CONCAT_CL;

    /// CPU reference: concatenate `tensors` along `dim`.
    ///
    /// All tensors must have the same shape except for dimension `dim`.
    pub fn execute_ref(
        tensors: &[(&[f32], &TensorShape)],
        dim: usize,
    ) -> Option<(Vec<f32>, TensorShape)> {
        if tensors.is_empty() {
            return None;
        }
        let ndim = tensors[0].1.ndim();
        if dim >= ndim {
            return None;
        }
        // Validate shapes match on non-concat dims.
        for (_, shape) in tensors.iter().skip(1) {
            if shape.ndim() != ndim {
                return None;
            }
            for d in 0..ndim {
                if d != dim && shape.dims[d] != tensors[0].1.dims[d] {
                    return None;
                }
            }
        }

        let mut out_dims = tensors[0].1.dims.clone();
        let total_dim: usize =
            tensors.iter().map(|(_, s)| s.dims[dim]).sum();
        out_dims[dim] = total_dim;
        let out_shape = TensorShape::new(out_dims);

        let numel = out_shape.numel();
        let mut output = vec![0.0f32; numel];

        // Copy block-by-block along the concat dimension.
        let outer: usize = out_shape.dims[..dim].iter().product();
        let inner: usize = out_shape.dims[dim + 1..].iter().product();

        let mut dim_offset = 0usize;
        for (data, shape) in tensors {
            let src_dim = shape.dims[dim];
            for o in 0..outer {
                for d in 0..src_dim {
                    let src_start = (o * src_dim + d) * inner;
                    let dst_start =
                        (o * out_shape.dims[dim] + dim_offset + d) * inner;
                    output[dst_start..dst_start + inner]
                        .copy_from_slice(&data[src_start..src_start + inner]);
                }
            }
            dim_offset += src_dim;
        }

        Some((output, out_shape))
    }
}

// ── SplitKernel ──────────────────────────────────────────────────

/// Splits a tensor along a dimension into equal-sized chunks.
pub struct SplitKernel;

impl SplitKernel {
    /// CPU reference: split `data` with `shape` along `dim` into
    /// chunks of `chunk_size`.
    pub fn execute_ref(
        data: &[f32],
        shape: &TensorShape,
        dim: usize,
        chunk_size: usize,
    ) -> Option<Vec<(Vec<f32>, TensorShape)>> {
        if dim >= shape.ndim() || chunk_size == 0 {
            return None;
        }
        let dim_size = shape.dims[dim];
        if !dim_size.is_multiple_of(chunk_size) {
            return None;
        }
        let num_chunks = dim_size / chunk_size;

        let outer: usize = shape.dims[..dim].iter().product();
        let inner: usize = shape.dims[dim + 1..].iter().product();

        let mut chunk_dims = shape.dims.clone();
        chunk_dims[dim] = chunk_size;
        let chunk_numel: usize = chunk_dims.iter().product();

        let mut results = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let mut chunk_data = vec![0.0f32; chunk_numel];
            for o in 0..outer {
                for d in 0..chunk_size {
                    let src_start =
                        (o * dim_size + c * chunk_size + d) * inner;
                    let dst_start = (o * chunk_size + d) * inner;
                    chunk_data[dst_start..dst_start + inner]
                        .copy_from_slice(&data[src_start..src_start + inner]);
                }
            }
            results.push((chunk_data, TensorShape::new(chunk_dims.clone())));
        }
        Some(results)
    }
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── TensorShape basics ───────────────────────────────────────

    #[test]
    fn shape_new_computes_row_major_strides() {
        let s = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(s.strides, vec![12, 4, 1]);
        assert!(s.contiguous);
    }

    #[test]
    fn shape_numel_product_of_dims() {
        assert_eq!(TensorShape::new(vec![2, 3, 4]).numel(), 24);
    }

    #[test]
    fn shape_scalar_numel_is_one() {
        let s = TensorShape::scalar();
        assert_eq!(s.numel(), 1);
        assert_eq!(s.ndim(), 0);
        assert!(s.contiguous);
    }

    #[test]
    fn shape_display_formatting() {
        let s = TensorShape::new(vec![2, 3]);
        let text = format!("{s}");
        assert!(text.contains("[2, 3]"));
    }

    #[test]
    fn shape_with_strides_contiguous() {
        let s = TensorShape::with_strides(vec![2, 3], vec![3, 1]);
        assert!(s.contiguous);
    }

    #[test]
    fn shape_with_strides_non_contiguous() {
        let s = TensorShape::with_strides(vec![2, 3], vec![6, 1]);
        assert!(!s.contiguous);
    }

    #[test]
    fn shape_ndim_matches_dims_len() {
        assert_eq!(TensorShape::new(vec![5, 10, 15]).ndim(), 3);
    }

    #[test]
    fn shape_1d() {
        let s = TensorShape::new(vec![8]);
        assert_eq!(s.strides, vec![1]);
        assert_eq!(s.numel(), 8);
    }

    #[test]
    fn shape_single_element_dims() {
        let s = TensorShape::new(vec![1, 1, 1]);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.strides, vec![1, 1, 1]);
    }

    #[test]
    fn shape_refresh_contiguity_updates_flag() {
        let mut s = TensorShape::new(vec![2, 3]);
        s.strides = vec![10, 1];
        s.refresh_contiguity();
        assert!(!s.contiguous);
    }

    #[test]
    fn shape_offset_default_zero() {
        assert_eq!(TensorShape::new(vec![4, 5]).offset, 0);
    }

    // ── ReshapeOp display ────────────────────────────────────────

    #[test]
    fn reshape_op_display() {
        assert_eq!(
            ReshapeOp::Reshape(vec![6]).to_string(),
            "Reshape([6])"
        );
        assert_eq!(
            ReshapeOp::Transpose(0, 1).to_string(),
            "Transpose(0, 1)"
        );
        assert_eq!(
            ReshapeOp::Squeeze(2).to_string(),
            "Squeeze(2)"
        );
        assert_eq!(
            ReshapeOp::Flatten { start: 1, end: 2 }.to_string(),
            "Flatten(1..=2)"
        );
    }

    // ── StrideComputer: Reshape ──────────────────────────────────

    #[test]
    fn stride_reshape_valid() {
        let s = TensorShape::new(vec![2, 3]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Reshape(vec![6]))
            .unwrap();
        assert_eq!(out.dims, vec![6]);
        assert_eq!(out.numel(), 6);
        assert!(out.contiguous);
    }

    #[test]
    fn stride_reshape_invalid_numel() {
        let s = TensorShape::new(vec![2, 3]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Reshape(vec![5])).is_none()
        );
    }

    #[test]
    fn stride_reshape_to_higher_rank() {
        let s = TensorShape::new(vec![12]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Reshape(vec![2, 2, 3]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![2, 2, 3]);
    }

    #[test]
    fn stride_reshape_preserves_numel() {
        let s = TensorShape::new(vec![4, 6]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Reshape(vec![8, 3]),
        )
        .unwrap();
        assert_eq!(out.numel(), 24);
    }

    // ── StrideComputer: Transpose ────────────────────────────────

    #[test]
    fn stride_transpose_2d() {
        let s = TensorShape::new(vec![2, 3]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 1))
            .unwrap();
        assert_eq!(out.dims, vec![3, 2]);
    }

    #[test]
    fn stride_transpose_out_of_bounds() {
        let s = TensorShape::new(vec![2, 3]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 2)).is_none()
        );
    }

    #[test]
    fn stride_transpose_same_dim() {
        let s = TensorShape::new(vec![2, 3]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 0))
            .unwrap();
        assert_eq!(out.dims, vec![2, 3]);
        assert!(out.contiguous);
    }

    #[test]
    fn stride_transpose_3d_dims_01() {
        let s = TensorShape::new(vec![2, 3, 4]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 1))
            .unwrap();
        assert_eq!(out.dims, vec![3, 2, 4]);
    }

    #[test]
    fn stride_transpose_marks_non_contiguous() {
        let s = TensorShape::new(vec![2, 3]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 1))
            .unwrap();
        assert!(!out.contiguous);
    }

    // ── StrideComputer: Permute ──────────────────────────────────

    #[test]
    fn stride_permute_identity() {
        let s = TensorShape::new(vec![2, 3, 4]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Permute(vec![0, 1, 2]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![2, 3, 4]);
        assert!(out.contiguous);
    }

    #[test]
    fn stride_permute_reverse() {
        let s = TensorShape::new(vec![2, 3, 4]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Permute(vec![2, 1, 0]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![4, 3, 2]);
    }

    #[test]
    fn stride_permute_wrong_length() {
        let s = TensorShape::new(vec![2, 3]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Permute(vec![0, 1, 2])
            )
            .is_none()
        );
    }

    #[test]
    fn stride_permute_duplicate_index() {
        let s = TensorShape::new(vec![2, 3, 4]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Permute(vec![0, 0, 1])
            )
            .is_none()
        );
    }

    #[test]
    fn stride_permute_4d() {
        let s = TensorShape::new(vec![2, 3, 4, 5]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Permute(vec![3, 2, 1, 0]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![5, 4, 3, 2]);
    }

    // ── StrideComputer: Squeeze / Unsqueeze ──────────────────────

    #[test]
    fn stride_squeeze_valid() {
        let s = TensorShape::new(vec![1, 3, 4]);
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Squeeze(0)).unwrap();
        assert_eq!(out.dims, vec![3, 4]);
    }

    #[test]
    fn stride_squeeze_non_one_dim_fails() {
        let s = TensorShape::new(vec![2, 3, 4]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Squeeze(0)).is_none()
        );
    }

    #[test]
    fn stride_squeeze_out_of_bounds() {
        let s = TensorShape::new(vec![1, 3]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Squeeze(5)).is_none()
        );
    }

    #[test]
    fn stride_unsqueeze_front() {
        let s = TensorShape::new(vec![3, 4]);
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Unsqueeze(0)).unwrap();
        assert_eq!(out.dims, vec![1, 3, 4]);
    }

    #[test]
    fn stride_unsqueeze_back() {
        let s = TensorShape::new(vec![3, 4]);
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Unsqueeze(2)).unwrap();
        assert_eq!(out.dims, vec![3, 4, 1]);
    }

    #[test]
    fn stride_unsqueeze_out_of_bounds() {
        let s = TensorShape::new(vec![3, 4]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Unsqueeze(5)).is_none()
        );
    }

    #[test]
    fn stride_squeeze_then_unsqueeze_roundtrip() {
        let s = TensorShape::new(vec![1, 4, 1]);
        let squeezed =
            StrideComputer::compute(&s, &ReshapeOp::Squeeze(0)).unwrap();
        assert_eq!(squeezed.dims, vec![4, 1]);
        let restored =
            StrideComputer::compute(&squeezed, &ReshapeOp::Unsqueeze(0))
                .unwrap();
        assert_eq!(restored.dims, vec![1, 4, 1]);
    }

    // ── StrideComputer: Flatten ──────────────────────────────────

    #[test]
    fn stride_flatten_middle() {
        let s = TensorShape::new(vec![2, 3, 4, 5]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Flatten { start: 1, end: 2 },
        )
        .unwrap();
        assert_eq!(out.dims, vec![2, 12, 5]);
    }

    #[test]
    fn stride_flatten_all() {
        let s = TensorShape::new(vec![2, 3, 4]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Flatten { start: 0, end: 2 },
        )
        .unwrap();
        assert_eq!(out.dims, vec![24]);
    }

    #[test]
    fn stride_flatten_single_dim() {
        let s = TensorShape::new(vec![2, 3, 4]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Flatten { start: 1, end: 1 },
        )
        .unwrap();
        assert_eq!(out.dims, vec![2, 3, 4]);
    }

    #[test]
    fn stride_flatten_invalid_range() {
        let s = TensorShape::new(vec![2, 3]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Flatten { start: 1, end: 5 }
            )
            .is_none()
        );
    }

    #[test]
    fn stride_flatten_start_gt_end() {
        let s = TensorShape::new(vec![2, 3, 4]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Flatten { start: 2, end: 1 }
            )
            .is_none()
        );
    }

    // ── StrideComputer: Split ────────────────────────────────────

    #[test]
    fn stride_split_even() {
        let s = TensorShape::new(vec![2, 6, 4]);
        let out = StrideComputer::compute(
            &s,
            &ReshapeOp::Split { dim: 1, chunk_size: 3 },
        )
        .unwrap();
        assert_eq!(out.dims, vec![2, 2, 3, 4]);
    }

    #[test]
    fn stride_split_uneven_fails() {
        let s = TensorShape::new(vec![2, 7]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Split { dim: 1, chunk_size: 3 }
            )
            .is_none()
        );
    }

    #[test]
    fn stride_split_zero_chunk_fails() {
        let s = TensorShape::new(vec![4, 4]);
        assert!(
            StrideComputer::compute(
                &s,
                &ReshapeOp::Split { dim: 0, chunk_size: 0 }
            )
            .is_none()
        );
    }

    // ── StrideComputer: Concat ───────────────────────────────────

    #[test]
    fn stride_concat_metadata() {
        let s = TensorShape::new(vec![2, 3]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Concat { dim: 0 })
            .unwrap();
        assert_eq!(out.dims, vec![2, 3]);
    }

    #[test]
    fn stride_concat_out_of_bounds() {
        let s = TensorShape::new(vec![2, 3]);
        assert!(
            StrideComputer::compute(&s, &ReshapeOp::Concat { dim: 5 })
                .is_none()
        );
    }

    // ── TransposeKernel ──────────────────────────────────────────

    #[test]
    fn transpose_2x3() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];
        TransposeKernel::execute_ref(&input, &mut output, 2, 3).unwrap();
        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_1x1() {
        let input = vec![42.0];
        let mut output = vec![0.0; 1];
        TransposeKernel::execute_ref(&input, &mut output, 1, 1).unwrap();
        assert_eq!(output, vec![42.0]);
    }

    #[test]
    fn transpose_square_3x3() {
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut output = vec![0.0; 9];
        TransposeKernel::execute_ref(&input, &mut output, 3, 3).unwrap();
        assert_eq!(output, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn transpose_buffer_too_small() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        assert!(
            TransposeKernel::execute_ref(&input, &mut output, 3, 3).is_none()
        );
    }

    #[test]
    fn transpose_double_is_identity() {
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut mid = vec![0.0; 12];
        let mut out = vec![0.0; 12];
        TransposeKernel::execute_ref(&input, &mut mid, 3, 4).unwrap();
        TransposeKernel::execute_ref(&mid, &mut out, 4, 3).unwrap();
        assert_eq!(input, out);
    }

    #[test]
    fn transpose_has_opencl_source() {
        assert!(TransposeKernel::SOURCE.contains("transpose_2d"));
    }

    // ── PermuteKernel ────────────────────────────────────────────

    #[test]
    fn permute_identity_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![2, 3, 4]);
        let out =
            PermuteKernel::execute_ref(&data, &shape, &[0, 1, 2]).unwrap();
        assert_eq!(data, out);
    }

    #[test]
    fn permute_reverse_3d() {
        // [2, 3, 4] -> [4, 3, 2]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![2, 3, 4]);
        let out =
            PermuteKernel::execute_ref(&data, &shape, &[2, 1, 0]).unwrap();
        // Verify element at [i,j,k] in input == element at [k,j,i] in output
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let src_idx = i * 12 + j * 4 + k;
                    let dst_idx = k * 6 + j * 2 + i;
                    assert_eq!(
                        out[dst_idx], data[src_idx],
                        "mismatch at [{i},{j},{k}]"
                    );
                }
            }
        }
    }

    #[test]
    fn permute_4d_batch_head_swap() {
        // [batch=2, heads=3, seq=2, dim=2] -> [batch=2, seq=2, heads=3, dim=2]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![2, 3, 2, 2]);
        let out =
            PermuteKernel::execute_ref(&data, &shape, &[0, 2, 1, 3]).unwrap();
        assert_eq!(out.len(), 24);
        // spot-check: input[0,0,0,0]=0 -> output[0,0,0,0]=0
        assert_eq!(out[0], 0.0);
        // input[0,1,0,0]=4 -> output[0,0,1,0]=2 (new layout)
        assert_eq!(out[2], 4.0);
    }

    #[test]
    fn permute_wrong_perm_length() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![2, 3]);
        assert!(PermuteKernel::execute_ref(&data, &shape, &[0, 1, 2]).is_none());
    }

    #[test]
    fn permute_has_opencl_source() {
        assert!(PermuteKernel::SOURCE.contains("permute_nd"));
    }

    // ── ConcatKernel ─────────────────────────────────────────────

    #[test]
    fn concat_1d_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let sa = TensorShape::new(vec![3]);
        let sb = TensorShape::new(vec![2]);
        let (out, shape) = ConcatKernel::execute_ref(
            &[(&a, &sa), (&b, &sb)],
            0,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(shape.dims, vec![5]);
    }

    #[test]
    fn concat_2d_along_dim0() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0];
        let sa = TensorShape::new(vec![2, 2]);
        let sb = TensorShape::new(vec![1, 2]);
        let (out, shape) = ConcatKernel::execute_ref(
            &[(&a, &sa), (&b, &sb)],
            0,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(shape.dims, vec![3, 2]);
    }

    #[test]
    fn concat_2d_along_dim1() {
        // [2,2] ++ [2,3] along dim=1 -> [2,5]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sa = TensorShape::new(vec![2, 2]);
        let sb = TensorShape::new(vec![2, 3]);
        let (out, shape) = ConcatKernel::execute_ref(
            &[(&a, &sa), (&b, &sb)],
            1,
        )
        .unwrap();
        assert_eq!(shape.dims, vec![2, 5]);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn concat_empty_tensors() {
        let result: Option<(Vec<f32>, TensorShape)> =
            ConcatKernel::execute_ref(&[], 0);
        assert!(result.is_none());
    }

    #[test]
    fn concat_mismatched_non_concat_dims() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0, 7.0];
        let sa = TensorShape::new(vec![3]);
        let sb = TensorShape::new(vec![4]);
        // 1D concat along dim=0 should work since there is no
        // non-concat dimension to mismatch.
        let result = ConcatKernel::execute_ref(&[(&a, &sa), (&b, &sb)], 0);
        assert!(result.is_some());
    }

    #[test]
    fn concat_dim_mismatch_fails() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let sa = TensorShape::new(vec![2, 3]);
        let sb = TensorShape::new(vec![3, 2]);
        // Non-concat dims differ: concat along dim=0 requires dim1 match
        assert!(
            ConcatKernel::execute_ref(&[(&a, &sa), (&b, &sb)], 0).is_none()
        );
    }

    #[test]
    fn concat_has_opencl_source() {
        assert!(ConcatKernel::SOURCE.contains("concat_1d"));
    }

    #[test]
    fn concat_three_tensors() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        let c = vec![4.0, 5.0, 6.0];
        let sa = TensorShape::new(vec![2]);
        let sb = TensorShape::new(vec![1]);
        let sc = TensorShape::new(vec![3]);
        let (out, shape) = ConcatKernel::execute_ref(
            &[(&a, &sa), (&b, &sb), (&c, &sc)],
            0,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(shape.dims, vec![6]);
    }

    // ── SplitKernel ──────────────────────────────────────────────

    #[test]
    fn split_1d_equal() {
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![6]);
        let chunks =
            SplitKernel::execute_ref(&data, &shape, 0, 2).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].0, vec![1.0, 2.0]);
        assert_eq!(chunks[1].0, vec![3.0, 4.0]);
        assert_eq!(chunks[2].0, vec![5.0, 6.0]);
    }

    #[test]
    fn split_2d_along_dim0() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![4, 3]);
        let chunks =
            SplitKernel::execute_ref(&data, &shape, 0, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(chunks[0].1.dims, vec![2, 3]);
    }

    #[test]
    fn split_2d_along_dim1() {
        // [2, 4] -> split dim=1 chunk=2 -> two [2,2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = TensorShape::new(vec![2, 4]);
        let chunks =
            SplitKernel::execute_ref(&data, &shape, 1, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(chunks[1].0, vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn split_uneven_fails() {
        let data: Vec<f32> = (0..7).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![7]);
        assert!(SplitKernel::execute_ref(&data, &shape, 0, 3).is_none());
    }

    #[test]
    fn split_zero_chunk_fails() {
        let data = vec![1.0, 2.0];
        let shape = TensorShape::new(vec![2]);
        assert!(SplitKernel::execute_ref(&data, &shape, 0, 0).is_none());
    }

    #[test]
    fn split_and_concat_roundtrip() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![4, 3]);
        let chunks =
            SplitKernel::execute_ref(&data, &shape, 0, 2).unwrap();
        let refs: Vec<(&[f32], &TensorShape)> =
            chunks.iter().map(|(d, s)| (d.as_slice(), s)).collect();
        let (reconstructed, _) = ConcatKernel::execute_ref(&refs, 0).unwrap();
        assert_eq!(data, reconstructed);
    }

    // ── TensorReshaper ───────────────────────────────────────────

    #[test]
    fn reshaper_view_preserves_data() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![3, 4]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Reshape(vec![4, 3]),
        )
        .unwrap();
        assert_eq!(out, data);
        assert_eq!(new_shape.dims, vec![4, 3]);
    }

    #[test]
    fn reshaper_transpose_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new(vec![2, 3]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Transpose(0, 1),
        )
        .unwrap();
        assert_eq!(new_shape.dims, vec![3, 2]);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn reshaper_squeeze() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = TensorShape::new(vec![1, 3]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Squeeze(0),
        )
        .unwrap();
        assert_eq!(out, data);
        assert_eq!(new_shape.dims, vec![3]);
    }

    #[test]
    fn reshaper_unsqueeze() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = TensorShape::new(vec![3]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Unsqueeze(0),
        )
        .unwrap();
        assert_eq!(out, data);
        assert_eq!(new_shape.dims, vec![1, 3]);
    }

    #[test]
    fn reshaper_flatten() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![2, 3, 4]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Flatten { start: 0, end: 2 },
        )
        .unwrap();
        assert_eq!(out, data);
        assert_eq!(new_shape.dims, vec![24]);
    }

    #[test]
    fn reshaper_invalid_op_returns_none() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = TensorShape::new(vec![3]);
        assert!(
            TensorReshaper::apply(
                &data,
                &shape,
                &ReshapeOp::Reshape(vec![5])
            )
            .is_none()
        );
    }

    #[test]
    fn reshaper_split_preserves_data() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let shape = TensorShape::new(vec![8]);
        let (out, new_shape) = TensorReshaper::apply(
            &data,
            &shape,
            &ReshapeOp::Split { dim: 0, chunk_size: 4 },
        )
        .unwrap();
        assert_eq!(out, data);
        assert_eq!(new_shape.dims, vec![2, 4]);
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn edge_scalar_reshape_to_1d() {
        let s = TensorShape::scalar();
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Reshape(vec![1])).unwrap();
        assert_eq!(out.dims, vec![1]);
    }

    #[test]
    fn edge_1d_single_element() {
        let s = TensorShape::new(vec![1]);
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Reshape(vec![1])).unwrap();
        assert_eq!(out.numel(), 1);
    }

    #[test]
    fn edge_zero_dim_tensor() {
        let s = TensorShape::new(vec![0, 3]);
        assert_eq!(s.numel(), 0);
        // Reshape to another zero-element shape is valid.
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Reshape(vec![0, 1]))
                .unwrap();
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn edge_unsqueeze_scalar() {
        let s = TensorShape::scalar();
        let out =
            StrideComputer::compute(&s, &ReshapeOp::Unsqueeze(0)).unwrap();
        assert_eq!(out.dims, vec![1]);
    }

    // ── Property-style tests ─────────────────────────────────────

    #[test]
    fn property_reshape_preserves_numel() {
        let shapes: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![2, 3], vec![6]),
            (vec![4, 3], vec![2, 6]),
            (vec![2, 3, 4], vec![24]),
            (vec![2, 3, 4], vec![6, 4]),
            (vec![5, 4], vec![4, 5]),
            (vec![1, 12], vec![3, 4]),
        ];
        for (from, to) in &shapes {
            let s = TensorShape::new(from.clone());
            let out = StrideComputer::compute(
                &s,
                &ReshapeOp::Reshape(to.clone()),
            )
            .unwrap();
            assert_eq!(
                s.numel(),
                out.numel(),
                "numel mismatch: {from:?} -> {to:?}"
            );
        }
    }

    #[test]
    fn property_transpose_involution() {
        // Transposing the same pair twice yields the original shape.
        let s = TensorShape::new(vec![3, 5, 7]);
        let t1 = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 2))
            .unwrap();
        let t2 = StrideComputer::compute(&t1, &ReshapeOp::Transpose(0, 2))
            .unwrap();
        assert_eq!(t2.dims, s.dims);
    }

    #[test]
    fn property_permute_preserves_numel() {
        let s = TensorShape::new(vec![2, 3, 4, 5]);
        let perms = vec![
            vec![0, 1, 2, 3],
            vec![3, 2, 1, 0],
            vec![1, 0, 3, 2],
            vec![2, 3, 0, 1],
        ];
        for p in &perms {
            let out = StrideComputer::compute(
                &s,
                &ReshapeOp::Permute(p.clone()),
            )
            .unwrap();
            assert_eq!(out.numel(), s.numel(), "perm {p:?}");
        }
    }

    #[test]
    fn property_flatten_preserves_numel() {
        let s = TensorShape::new(vec![2, 3, 4, 5]);
        for start in 0..4 {
            for end in start..4 {
                let out = StrideComputer::compute(
                    &s,
                    &ReshapeOp::Flatten { start, end },
                )
                .unwrap();
                assert_eq!(
                    out.numel(),
                    s.numel(),
                    "flatten {start}..={end}"
                );
            }
        }
    }

    #[test]
    fn property_squeeze_unsqueeze_preserves_numel() {
        let s = TensorShape::new(vec![1, 4, 1, 3, 1]);
        let squeezable = [0, 2, 4];
        for &d in &squeezable {
            let out =
                StrideComputer::compute(&s, &ReshapeOp::Squeeze(d)).unwrap();
            assert_eq!(out.numel(), s.numel());
        }
    }

    // ── Non-contiguous tensor handling ───────────────────────────

    #[test]
    fn non_contiguous_after_transpose() {
        let s = TensorShape::new(vec![4, 5]);
        let out = StrideComputer::compute(&s, &ReshapeOp::Transpose(0, 1))
            .unwrap();
        assert!(!out.contiguous);
        assert_eq!(out.strides, vec![1, 5]);
    }

    #[test]
    fn non_contiguous_custom_strides() {
        let s = TensorShape::with_strides(vec![3, 3], vec![4, 1]);
        assert!(!s.contiguous);
        assert_eq!(s.numel(), 9);
    }
}
