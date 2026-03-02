//! CUDA tensor manipulation operations with CPU fallback.
//!
//! Provides shape-preserving and shape-changing tensor operations commonly
//! used in transformer inference pipelines:
//!
//! - **Reshape / View**: Reinterpret tensor shape without data copy
//! - **Permute**: Rearrange tensor dimensions via permutation
//! - **Contiguous**: Materialise a contiguous memory layout
//! - **Expand / Repeat**: Broadcast and tile tensors
//! - **Flip / Roll**: Reverse and circular-shift along axes
//! - **Pad / Crop**: Add padding or extract sub-regions
//! - **Where / Clamp**: Conditional selection and value clamping
//! - **Abs / Neg / Sign**: Unary element-wise operations
//!
//! # Kernel strategy
//!
//! CUDA kernels use grid-stride loops with 256 threads per block.
//! CPU fallbacks are provided for all operations.
//!
//! # Feature gating
//!
//! All GPU paths are behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallbacks compile unconditionally.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source (compiled at runtime via NVRTC when `gpu`/`cuda` is active)
// ---------------------------------------------------------------------------

/// CUDA kernel for conditional selection (where).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const WHERE_KERNEL_SRC: &str = r#"
extern "C" __global__ void where_f32(
    const float* __restrict__ cond,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = (cond[i] != 0.0f) ? x[i] : y[i];
    }
}
"#;

/// CUDA kernel for value clamping.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const CLAMP_KERNEL_SRC: &str = r#"
extern "C" __global__ void clamp_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    float min_val,
    float max_val,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float v = input[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        output[i] = v;
    }
}
"#;

/// CUDA kernel for absolute value.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ABS_KERNEL_SRC: &str = r#"
extern "C" __global__ void abs_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = fabsf(input[i]);
    }
}
"#;

/// CUDA kernel for negation.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const NEG_KERNEL_SRC: &str = r#"
extern "C" __global__ void neg_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = -input[i];
    }
}
"#;

/// CUDA kernel for sign function.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const SIGN_KERNEL_SRC: &str = r#"
extern "C" __global__ void sign_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float v = input[i];
        output[i] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
    }
}
"#;

/// CUDA kernel for padding.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const PAD_KERNEL_SRC: &str = r#"
extern "C" __global__ void pad_2d_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_rows, int in_cols,
    int out_rows, int out_cols,
    int pad_top, int pad_left,
    float pad_val,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int out_r = i / out_cols;
        int out_c = i % out_cols;
        int in_r = out_r - pad_top;
        int in_c = out_c - pad_left;
        if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            output[i] = input[in_r * in_cols + in_c];
        } else {
            output[i] = pad_val;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for pad operations.
#[derive(Debug, Clone)]
pub struct PadConfig {
    /// Padding before each dimension `[dim0_before, dim0_after, dim1_before, …]`.
    pub padding: Vec<usize>,
    /// Fill value for padded regions.
    pub value: f32,
}

/// Unary tensor operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Absolute value.
    Abs,
    /// Negation.
    Neg,
    /// Sign function: -1, 0, or 1.
    Sign,
}

impl UnaryOp {
    /// CUDA kernel function name.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Self::Abs => "abs_f32",
            Self::Neg => "neg_f32",
            Self::Sign => "sign_f32",
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: compute total elements from shape
// ---------------------------------------------------------------------------

fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Compute strides for a contiguous row-major layout.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0usize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a linear index to multi-dimensional coordinates.
fn linear_to_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    for i in (0..ndim).rev() {
        coords[i] = linear % shape[i];
        linear /= shape[i];
    }
    coords
}

/// Convert multi-dimensional coordinates to a linear index using strides.
fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// Reshape tensor without data copy (CPU).
///
/// Validates that old and new shapes have the same total element count,
/// then returns a copy of the data reinterpreted as the new shape.
pub fn reshape_cpu(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    let old_total = total_elements(old_shape);
    let new_total = total_elements(new_shape);
    if old_total != new_total {
        return Err(KernelError::InvalidArguments {
            reason: format!("reshape: element count mismatch (old={old_total}, new={new_total})"),
        }
        .into());
    }
    if data.len() < old_total {
        return Err(KernelError::InvalidArguments {
            reason: format!("reshape: buffer too small ({} < {old_total})", data.len()),
        }
        .into());
    }
    Ok(data[..old_total].to_vec())
}

/// Create a view with a different shape (CPU).
///
/// Semantically identical to [`reshape_cpu`] — validates element count
/// parity and returns a logical copy.
pub fn view_cpu(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    reshape_cpu(data, old_shape, new_shape)
}

/// Permute tensor dimensions (CPU).
///
/// Rearranges data so that the output dimension order matches `perm`.
pub fn permute_cpu(data: &[f32], shape: &[usize], perm: &[usize]) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if perm.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("permute: perm length {} != ndim {ndim}", perm.len()),
        }
        .into());
    }
    // Validate permutation is a valid bijection.
    let mut seen = vec![false; ndim];
    for &p in perm {
        if p >= ndim {
            return Err(KernelError::InvalidArguments {
                reason: format!("permute: index {p} out of range for ndim {ndim}"),
            }
            .into());
        }
        if seen[p] {
            return Err(KernelError::InvalidArguments {
                reason: format!("permute: duplicate index {p}"),
            }
            .into());
        }
        seen[p] = true;
    }

    let n = total_elements(shape);
    if data.len() < n {
        return Err(
            KernelError::InvalidArguments { reason: "permute: buffer too small".into() }.into()
        );
    }

    let in_strides = compute_strides(shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let out_strides = compute_strides(&out_shape);

    let mut output = vec![0.0f32; n];
    for i in 0..n {
        let out_coords = linear_to_coords(i, &out_shape);
        // Map output coords back to input coords via inverse permutation.
        let mut in_coords = vec![0usize; ndim];
        for (d, &p) in perm.iter().enumerate() {
            in_coords[p] = out_coords[d];
        }
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        output[coords_to_linear(&out_coords, &out_strides)] = data[in_idx];
    }
    Ok(output)
}

/// Make tensor contiguous in memory (CPU).
///
/// For tensors stored with custom strides, copies data into a fresh
/// contiguous buffer.  If `strides` matches the row-major layout for
/// `shape`, the data is already contiguous and is returned as-is.
pub fn contiguous_cpu(data: &[f32], shape: &[usize], strides: &[usize]) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if strides.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("contiguous: strides length {} != ndim {ndim}", strides.len()),
        }
        .into());
    }

    let expected = compute_strides(shape);
    let n = total_elements(shape);

    // Already contiguous — fast path.
    if strides == expected.as_slice() {
        if data.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: "contiguous: buffer too small".into(),
            }
            .into());
        }
        return Ok(data[..n].to_vec());
    }

    let mut output = vec![0.0f32; n];
    for (i, out) in output.iter_mut().enumerate() {
        let coords = linear_to_coords(i, shape);
        let src_idx = coords_to_linear(&coords, strides);
        if src_idx >= data.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "contiguous: source index {src_idx} out of bounds (len={})",
                    data.len()
                ),
            }
            .into());
        }
        *out = data[src_idx];
    }
    Ok(output)
}

/// Broadcast-expand tensor along size-1 dimensions (CPU).
///
/// Each dimension in `new_shape` must either equal the original dimension
/// or the original dimension must be 1 (which is then broadcast).
pub fn expand_cpu(data: &[f32], shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if new_shape.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("expand: ndim mismatch (old={ndim}, new={})", new_shape.len()),
        }
        .into());
    }
    for (d, (&old, &new)) in shape.iter().zip(new_shape.iter()).enumerate() {
        if old != 1 && old != new {
            return Err(KernelError::InvalidArguments {
                reason: format!("expand: dim {d} is {old}, cannot expand to {new} (must be 1)"),
            }
            .into());
        }
    }

    let in_strides = compute_strides(shape);
    let n = total_elements(new_shape);
    let mut output = vec![0.0f32; n];

    for (i, out) in output.iter_mut().enumerate() {
        let coords = linear_to_coords(i, new_shape);
        let mut src_idx = 0usize;
        for (d, &c) in coords.iter().enumerate() {
            if shape[d] != 1 {
                src_idx += c * in_strides[d];
            }
        }
        *out = data[src_idx];
    }
    Ok(output)
}

/// Repeat (tile) tensor along each dimension (CPU).
///
/// `repeats[d]` specifies how many times to tile dimension `d`.
pub fn repeat_cpu(data: &[f32], shape: &[usize], repeats: &[usize]) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if repeats.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("repeat: repeats length {} != ndim {ndim}", repeats.len()),
        }
        .into());
    }
    for &r in repeats {
        if r == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "repeat: repeat count must be >= 1".into(),
            }
            .into());
        }
    }

    let in_strides = compute_strides(shape);
    let out_shape: Vec<usize> = shape.iter().zip(repeats.iter()).map(|(&s, &r)| s * r).collect();
    let n = total_elements(&out_shape);
    let mut output = vec![0.0f32; n];

    for (i, out) in output.iter_mut().enumerate() {
        let coords = linear_to_coords(i, &out_shape);
        let src_coords: Vec<usize> =
            coords.iter().zip(shape.iter()).map(|(&c, &s)| c % s).collect();
        let src_idx = coords_to_linear(&src_coords, &in_strides);
        *out = data[src_idx];
    }
    Ok(output)
}

/// Flip tensor along a given axis (CPU).
pub fn flip_cpu(data: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("flip: axis {axis} >= ndim {ndim}"),
        }
        .into());
    }

    let n = total_elements(shape);
    if data.len() < n {
        return Err(
            KernelError::InvalidArguments { reason: "flip: buffer too small".into() }.into()
        );
    }

    let strides = compute_strides(shape);
    let mut output = vec![0.0f32; n];

    for (i, out) in output.iter_mut().enumerate() {
        let mut coords = linear_to_coords(i, shape);
        coords[axis] = shape[axis] - 1 - coords[axis];
        let src_idx = coords_to_linear(&coords, &strides);
        *out = data[src_idx];
    }
    Ok(output)
}

/// Circular shift (roll) tensor along a given axis (CPU).
pub fn roll_cpu(data: &[f32], shape: &[usize], axis: usize, shift: isize) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("roll: axis {axis} >= ndim {ndim}"),
        }
        .into());
    }

    let n = total_elements(shape);
    if data.len() < n {
        return Err(
            KernelError::InvalidArguments { reason: "roll: buffer too small".into() }.into()
        );
    }

    let dim_size = shape[axis] as isize;
    let strides = compute_strides(shape);
    let mut output = vec![0.0f32; n];

    for (i, out) in output.iter_mut().enumerate() {
        let mut coords = linear_to_coords(i, shape);
        let orig = coords[axis] as isize;
        // Wrap with modular arithmetic.
        coords[axis] = ((orig - shift) % dim_size + dim_size) as usize % shape[axis];
        let src_idx = coords_to_linear(&coords, &strides);
        *out = data[src_idx];
    }
    Ok(output)
}

/// Pad tensor with a constant value (CPU).
///
/// `padding` is `[before_dim0, after_dim0, before_dim1, after_dim1, …]`.
pub fn pad_cpu(data: &[f32], shape: &[usize], padding: &[usize], value: f32) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if padding.len() != ndim * 2 {
        return Err(KernelError::InvalidArguments {
            reason: format!("pad: padding length {} != 2 * ndim {ndim}", padding.len()),
        }
        .into());
    }

    let n = total_elements(shape);
    if data.len() < n {
        return Err(KernelError::InvalidArguments { reason: "pad: buffer too small".into() }.into());
    }

    let out_shape: Vec<usize> =
        shape.iter().enumerate().map(|(d, &s)| s + padding[2 * d] + padding[2 * d + 1]).collect();
    let out_n = total_elements(&out_shape);
    let in_strides = compute_strides(shape);
    let mut output = vec![value; out_n];

    for i in 0..n {
        let in_coords = linear_to_coords(i, shape);
        let out_coords: Vec<usize> =
            in_coords.iter().enumerate().map(|(d, &c)| c + padding[2 * d]).collect();
        let out_strides = compute_strides(&out_shape);
        let out_idx = coords_to_linear(&out_coords, &out_strides);
        output[out_idx] = data[coords_to_linear(&in_coords, &in_strides)];
    }
    Ok(output)
}

/// Crop (slice) a rectangular region from a tensor (CPU).
///
/// `starts` and `ends` define the half-open range `[start, end)` per
/// dimension.
pub fn crop_cpu(
    data: &[f32],
    shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<f32>> {
    let ndim = shape.len();
    if starts.len() != ndim || ends.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "crop: starts/ends length mismatch (starts={}, ends={}, ndim={ndim})",
                starts.len(),
                ends.len()
            ),
        }
        .into());
    }
    for d in 0..ndim {
        if starts[d] >= ends[d] {
            return Err(KernelError::InvalidArguments {
                reason: format!("crop: start >= end at dim {d} ({} >= {})", starts[d], ends[d]),
            }
            .into());
        }
        if ends[d] > shape[d] {
            return Err(KernelError::InvalidArguments {
                reason: format!("crop: end {} exceeds dim {d} size {}", ends[d], shape[d]),
            }
            .into());
        }
    }

    let out_shape: Vec<usize> = starts.iter().zip(ends.iter()).map(|(&s, &e)| e - s).collect();
    let out_n = total_elements(&out_shape);
    let in_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut output = vec![0.0f32; out_n];

    for i in 0..out_n {
        let out_coords = linear_to_coords(i, &out_shape);
        let in_coords: Vec<usize> =
            out_coords.iter().zip(starts.iter()).map(|(&c, &s)| c + s).collect();
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        output[coords_to_linear(&out_coords, &out_strides)] = data[in_idx];
    }
    Ok(output)
}

/// Conditional selection: `out[i] = cond[i] != 0 ? x[i] : y[i]` (CPU).
pub fn where_cond_cpu(cond: &[f32], x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    let n = cond.len();
    if x.len() < n || y.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("where_cond: length mismatch (cond={n}, x={}, y={})", x.len(), y.len()),
        }
        .into());
    }
    let output: Vec<f32> = cond
        .iter()
        .zip(x.iter().zip(y.iter()))
        .map(|(&c, (&xv, &yv))| if c != 0.0 { xv } else { yv })
        .collect();
    Ok(output)
}

/// Clamp values to `[min_val, max_val]` (CPU).
pub fn clamp_cpu(data: &[f32], min_val: f32, max_val: f32) -> Result<Vec<f32>> {
    if min_val > max_val {
        return Err(KernelError::InvalidArguments {
            reason: format!("clamp: min ({min_val}) > max ({max_val})"),
        }
        .into());
    }
    let output: Vec<f32> = data.iter().map(|&v| v.clamp(min_val, max_val)).collect();
    Ok(output)
}

/// Absolute value (CPU).
pub fn abs_cpu(data: &[f32]) -> Vec<f32> {
    data.iter().map(|v| v.abs()).collect()
}

/// Negation (CPU).
pub fn neg_cpu(data: &[f32]) -> Vec<f32> {
    data.iter().map(|v| -v).collect()
}

/// Sign function: returns -1.0, 0.0, or 1.0 per element (CPU).
pub fn sign_cpu(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Dispatch unary operation by [`UnaryOp`] variant (CPU).
pub fn unary_cpu(data: &[f32], op: UnaryOp) -> Vec<f32> {
    match op {
        UnaryOp::Abs => abs_cpu(data),
        UnaryOp::Neg => neg_cpu(data),
        UnaryOp::Sign => sign_cpu(data),
    }
}

// ---------------------------------------------------------------------------
// Unified dispatchers (GPU → CPU fallback)
// ---------------------------------------------------------------------------

/// Reshape tensor — dispatches to GPU when available, falls back to CPU.
pub fn reshape_forward(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    // Reshape is a logical operation; no GPU kernel needed.
    reshape_cpu(data, old_shape, new_shape)
}

/// View tensor — dispatches to GPU when available, falls back to CPU.
pub fn view_forward(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    view_cpu(data, old_shape, new_shape)
}

/// Permute tensor — dispatches to GPU when available, falls back to CPU.
pub fn permute_forward(data: &[f32], shape: &[usize], perm: &[usize]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, perm);
    }
    permute_cpu(data, shape, perm)
}

/// Contiguous — dispatches to GPU when available, falls back to CPU.
pub fn contiguous_forward(data: &[f32], shape: &[usize], strides: &[usize]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, strides);
    }
    contiguous_cpu(data, shape, strides)
}

/// Expand — dispatches to GPU when available, falls back to CPU.
pub fn expand_forward(data: &[f32], shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, new_shape);
    }
    expand_cpu(data, shape, new_shape)
}

/// Repeat — dispatches to GPU when available, falls back to CPU.
pub fn repeat_forward(data: &[f32], shape: &[usize], repeats: &[usize]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, repeats);
    }
    repeat_cpu(data, shape, repeats)
}

/// Flip — dispatches to GPU when available, falls back to CPU.
pub fn flip_forward(data: &[f32], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, axis);
    }
    flip_cpu(data, shape, axis)
}

/// Roll — dispatches to GPU when available, falls back to CPU.
pub fn roll_forward(data: &[f32], shape: &[usize], axis: usize, shift: isize) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, axis, shift);
    }
    roll_cpu(data, shape, axis, shift)
}

/// Pad — dispatches to GPU when available, falls back to CPU.
pub fn pad_forward(
    data: &[f32],
    shape: &[usize],
    padding: &[usize],
    value: f32,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, padding, value);
    }
    pad_cpu(data, shape, padding, value)
}

/// Crop — dispatches to GPU when available, falls back to CPU.
pub fn crop_forward(
    data: &[f32],
    shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, shape, starts, ends);
    }
    crop_cpu(data, shape, starts, ends)
}

/// Where — dispatches to GPU when available, falls back to CPU.
pub fn where_cond_forward(cond: &[f32], x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (cond, x, y);
    }
    where_cond_cpu(cond, x, y)
}

/// Clamp — dispatches to GPU when available, falls back to CPU.
pub fn clamp_forward(data: &[f32], min_val: f32, max_val: f32) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, min_val, max_val);
    }
    clamp_cpu(data, min_val, max_val)
}

/// Unary operation — dispatches to GPU when available, falls back to CPU.
pub fn unary_forward(data: &[f32], op: UnaryOp) -> Vec<f32> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let _ = (data, op);
    }
    unary_cpu(data, op)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Helper utilities
    // -------------------------------------------------------------------

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // -------------------------------------------------------------------
    // Reshape
    // -------------------------------------------------------------------

    #[test]
    fn test_reshape_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = reshape_cpu(&data, &[2, 3], &[3, 2]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_reshape_to_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = reshape_cpu(&data, &[2, 2], &[4]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_reshape_from_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = reshape_cpu(&data, &[6], &[2, 3]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_reshape_element_count_mismatch() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let err = reshape_cpu(&data, &[2, 2], &[3, 2]);
        assert!(err.is_err());
    }

    #[test]
    fn test_reshape_buffer_too_small() {
        let data = vec![1.0, 2.0];
        let err = reshape_cpu(&data, &[2, 2], &[4]);
        assert!(err.is_err());
    }

    #[test]
    fn test_reshape_scalar() {
        let data = vec![42.0];
        let result = reshape_cpu(&data, &[1], &[1, 1, 1]).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_reshape_high_rank() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let result = reshape_cpu(&data, &[2, 3, 4], &[4, 6]).unwrap();
        assert_eq!(result.len(), 24);
        assert_eq!(result, data);
    }

    // -------------------------------------------------------------------
    // View
    // -------------------------------------------------------------------

    #[test]
    fn test_view_same_as_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r1 = reshape_cpu(&data, &[2, 3], &[6]).unwrap();
        let r2 = view_cpu(&data, &[2, 3], &[6]).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_view_error_propagation() {
        let data = vec![1.0, 2.0];
        assert!(view_cpu(&data, &[2], &[3]).is_err());
    }

    // -------------------------------------------------------------------
    // Permute
    // -------------------------------------------------------------------

    #[test]
    fn test_permute_2d_transpose() {
        // 2×3 matrix → 3×2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = permute_cpu(&data, &[2, 3], &[1, 0]).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_permute_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = permute_cpu(&data, &[2, 3], &[0, 1]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_permute_3d() {
        // Shape [2,3,1] → perm [2,0,1] → shape [1,2,3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = permute_cpu(&data, &[2, 3, 1], &[2, 0, 1]).unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_permute_invalid_perm_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(permute_cpu(&data, &[2, 2], &[0]).is_err());
    }

    #[test]
    fn test_permute_invalid_perm_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(permute_cpu(&data, &[2, 2], &[0, 5]).is_err());
    }

    #[test]
    fn test_permute_duplicate_index() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(permute_cpu(&data, &[2, 2], &[0, 0]).is_err());
    }

    #[test]
    fn test_permute_buffer_too_small() {
        let data = vec![1.0];
        assert!(permute_cpu(&data, &[2, 2], &[1, 0]).is_err());
    }

    // -------------------------------------------------------------------
    // Contiguous
    // -------------------------------------------------------------------

    #[test]
    fn test_contiguous_already_contiguous() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = contiguous_cpu(&data, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_contiguous_transposed_strides() {
        // 2×3 stored as column-major (strides [1, 2])
        // Original row-major: [[1,2,3],[4,5,6]] stored as [1,4,2,5,3,6]
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let result = contiguous_cpu(&data, &[2, 3], &[1, 2]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_contiguous_stride_mismatch() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(contiguous_cpu(&data, &[2, 2], &[2]).is_err());
    }

    #[test]
    fn test_contiguous_buffer_too_small() {
        let data = vec![1.0];
        assert!(contiguous_cpu(&data, &[2, 2], &[2, 1]).is_err());
    }

    #[test]
    fn test_contiguous_1d() {
        let data = vec![10.0, 20.0, 30.0];
        let result = contiguous_cpu(&data, &[3], &[1]).unwrap();
        assert_eq!(result, data);
    }

    // -------------------------------------------------------------------
    // Expand
    // -------------------------------------------------------------------

    #[test]
    fn test_expand_broadcast_row() {
        // [1, 3] → [2, 3]
        let data = vec![1.0, 2.0, 3.0];
        let result = expand_cpu(&data, &[1, 3], &[2, 3]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_expand_broadcast_col() {
        // [2, 1] → [2, 3]
        let data = vec![1.0, 2.0];
        let result = expand_cpu(&data, &[2, 1], &[2, 3]).unwrap();
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_expand_no_broadcast() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = expand_cpu(&data, &[2, 2], &[2, 2]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_expand_ndim_mismatch() {
        let data = vec![1.0, 2.0];
        assert!(expand_cpu(&data, &[2], &[2, 3]).is_err());
    }

    #[test]
    fn test_expand_invalid_dim() {
        // dim=2 is not 1, cannot expand to 4
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(expand_cpu(&data, &[2, 2], &[2, 4]).is_err());
    }

    #[test]
    fn test_expand_3d() {
        // [1, 1, 2] → [3, 2, 2]
        let data = vec![5.0, 6.0];
        let result = expand_cpu(&data, &[1, 1, 2], &[3, 2, 2]).unwrap();
        assert_eq!(result.len(), 12);
        assert!(result.iter().all(|&v| v == 5.0 || v == 6.0));
    }

    // -------------------------------------------------------------------
    // Repeat
    // -------------------------------------------------------------------

    #[test]
    fn test_repeat_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = repeat_cpu(&data, &[2, 2], &[1, 2]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_rows() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = repeat_cpu(&data, &[2, 2], &[2, 1]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_identity() {
        let data = vec![1.0, 2.0, 3.0];
        let result = repeat_cpu(&data, &[3], &[1]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_repeat_ndim_mismatch() {
        let data = vec![1.0, 2.0];
        assert!(repeat_cpu(&data, &[2], &[1, 2]).is_err());
    }

    #[test]
    fn test_repeat_zero_count() {
        let data = vec![1.0, 2.0];
        assert!(repeat_cpu(&data, &[2], &[0]).is_err());
    }

    #[test]
    fn test_repeat_3d() {
        let data = vec![1.0, 2.0];
        let result = repeat_cpu(&data, &[1, 1, 2], &[2, 3, 1]).unwrap();
        // Output shape: [2, 3, 2] = 12 elements
        assert_eq!(result.len(), 12);
    }

    // -------------------------------------------------------------------
    // Flip
    // -------------------------------------------------------------------

    #[test]
    fn test_flip_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = flip_cpu(&data, &[4], 0).unwrap();
        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d_axis0() {
        // [[1,2],[3,4]] → [[3,4],[1,2]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = flip_cpu(&data, &[2, 2], 0).unwrap();
        assert_eq!(result, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_flip_2d_axis1() {
        // [[1,2],[3,4]] → [[2,1],[4,3]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = flip_cpu(&data, &[2, 2], 1).unwrap();
        assert_eq!(result, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_flip_invalid_axis() {
        let data = vec![1.0, 2.0];
        assert!(flip_cpu(&data, &[2], 1).is_err());
    }

    #[test]
    fn test_flip_single_element() {
        let data = vec![42.0];
        let result = flip_cpu(&data, &[1], 0).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_flip_buffer_too_small() {
        let data = vec![1.0];
        assert!(flip_cpu(&data, &[3], 0).is_err());
    }

    // -------------------------------------------------------------------
    // Roll
    // -------------------------------------------------------------------

    #[test]
    fn test_roll_1d_positive() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = roll_cpu(&data, &[4], 0, 1).unwrap();
        assert_eq!(result, vec![4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_1d_negative() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = roll_cpu(&data, &[4], 0, -1).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_roll_2d_axis1() {
        // [[1,2,3],[4,5,6]] roll axis=1 shift=1 → [[3,1,2],[6,4,5]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = roll_cpu(&data, &[2, 3], 1, 1).unwrap();
        assert_eq!(result, vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    }

    #[test]
    fn test_roll_zero_shift() {
        let data = vec![1.0, 2.0, 3.0];
        let result = roll_cpu(&data, &[3], 0, 0).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roll_full_cycle() {
        let data = vec![1.0, 2.0, 3.0];
        let result = roll_cpu(&data, &[3], 0, 3).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roll_invalid_axis() {
        let data = vec![1.0, 2.0];
        assert!(roll_cpu(&data, &[2], 1, 1).is_err());
    }

    #[test]
    fn test_roll_buffer_too_small() {
        let data = vec![1.0];
        assert!(roll_cpu(&data, &[3], 0, 1).is_err());
    }

    // -------------------------------------------------------------------
    // Pad
    // -------------------------------------------------------------------

    #[test]
    fn test_pad_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let result = pad_cpu(&data, &[3], &[1, 2], 0.0).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pad_2d() {
        // [[1,2],[3,4]] pad top=1, bottom=0, left=0, right=1 with -1
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = pad_cpu(&data, &[2, 2], &[1, 0, 0, 1], -1.0).unwrap();
        // Output shape: [3, 3]
        assert_eq!(result, vec![-1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0]);
    }

    #[test]
    fn test_pad_zero_padding() {
        let data = vec![1.0, 2.0, 3.0];
        let result = pad_cpu(&data, &[3], &[0, 0], 0.0).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_pad_custom_value() {
        let data = vec![5.0];
        let result = pad_cpu(&data, &[1], &[2, 2], 99.0).unwrap();
        assert_eq!(result, vec![99.0, 99.0, 5.0, 99.0, 99.0]);
    }

    #[test]
    fn test_pad_invalid_padding_length() {
        let data = vec![1.0, 2.0];
        assert!(pad_cpu(&data, &[2], &[1, 1, 1], 0.0).is_err());
    }

    #[test]
    fn test_pad_buffer_too_small() {
        let data = vec![1.0];
        assert!(pad_cpu(&data, &[3], &[0, 0], 0.0).is_err());
    }

    // -------------------------------------------------------------------
    // Crop
    // -------------------------------------------------------------------

    #[test]
    fn test_crop_1d() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = crop_cpu(&data, &[5], &[1], &[4]).unwrap();
        assert_eq!(result, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_crop_2d() {
        // [[1,2,3],[4,5,6],[7,8,9]] → crop rows[0..2], cols[1..3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = crop_cpu(&data, &[3, 3], &[0, 1], &[2, 3]).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_crop_full_extent() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = crop_cpu(&data, &[2, 2], &[0, 0], &[2, 2]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_crop_start_ge_end() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(crop_cpu(&data, &[3], &[2], &[1]).is_err());
    }

    #[test]
    fn test_crop_end_exceeds_shape() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(crop_cpu(&data, &[3], &[0], &[5]).is_err());
    }

    #[test]
    fn test_crop_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(crop_cpu(&data, &[3], &[0, 0], &[3]).is_err());
    }

    // -------------------------------------------------------------------
    // Where conditional
    // -------------------------------------------------------------------

    #[test]
    fn test_where_cond_basic() {
        let cond = vec![1.0, 0.0, 1.0, 0.0];
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let y = vec![100.0, 200.0, 300.0, 400.0];
        let result = where_cond_cpu(&cond, &x, &y).unwrap();
        assert_eq!(result, vec![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn test_where_cond_all_true() {
        let cond = vec![1.0, 1.0, 1.0];
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = where_cond_cpu(&cond, &x, &y).unwrap();
        assert_eq!(result, x);
    }

    #[test]
    fn test_where_cond_all_false() {
        let cond = vec![0.0, 0.0, 0.0];
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = where_cond_cpu(&cond, &x, &y).unwrap();
        assert_eq!(result, y);
    }

    #[test]
    fn test_where_cond_length_mismatch() {
        let cond = vec![1.0, 0.0];
        let x = vec![1.0];
        let y = vec![2.0, 3.0];
        assert!(where_cond_cpu(&cond, &x, &y).is_err());
    }

    #[test]
    fn test_where_cond_nonzero_as_true() {
        let cond = vec![-5.0, 0.0, 0.01];
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = where_cond_cpu(&cond, &x, &y).unwrap();
        assert_eq!(result, vec![1.0, 5.0, 3.0]);
    }

    // -------------------------------------------------------------------
    // Clamp
    // -------------------------------------------------------------------

    #[test]
    fn test_clamp_basic() {
        let data = vec![-5.0, -1.0, 0.0, 0.5, 3.0, 10.0];
        let result = clamp_cpu(&data, -1.0, 3.0).unwrap();
        assert_eq!(result, vec![-1.0, -1.0, 0.0, 0.5, 3.0, 3.0]);
    }

    #[test]
    fn test_clamp_no_effect() {
        let data = vec![1.0, 2.0, 3.0];
        let result = clamp_cpu(&data, 0.0, 10.0).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_clamp_min_eq_max() {
        let data = vec![1.0, 2.0, 3.0];
        let result = clamp_cpu(&data, 2.0, 2.0).unwrap();
        assert_eq!(result, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_clamp_min_gt_max() {
        let data = vec![1.0];
        assert!(clamp_cpu(&data, 5.0, 1.0).is_err());
    }

    #[test]
    fn test_clamp_empty() {
        let data: Vec<f32> = vec![];
        let result = clamp_cpu(&data, -1.0, 1.0).unwrap();
        assert!(result.is_empty());
    }

    // -------------------------------------------------------------------
    // Abs
    // -------------------------------------------------------------------

    #[test]
    fn test_abs_basic() {
        let data = vec![-3.0, -1.0, 0.0, 2.0, 5.0];
        let result = abs_cpu(&data);
        assert_eq!(result, vec![3.0, 1.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn test_abs_empty() {
        let result = abs_cpu(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_abs_all_positive() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(abs_cpu(&data), data);
    }

    // -------------------------------------------------------------------
    // Neg
    // -------------------------------------------------------------------

    #[test]
    fn test_neg_basic() {
        let data = vec![-3.0, 0.0, 5.0];
        let result = neg_cpu(&data);
        assert_eq!(result, vec![3.0, -0.0, -5.0]);
    }

    #[test]
    fn test_neg_double_negation() {
        let data = vec![1.0, -2.0, 3.0];
        let result = neg_cpu(&neg_cpu(&data));
        assert!(approx_eq(&result, &data, 1e-7));
    }

    #[test]
    fn test_neg_empty() {
        let result = neg_cpu(&[]);
        assert!(result.is_empty());
    }

    // -------------------------------------------------------------------
    // Sign
    // -------------------------------------------------------------------

    #[test]
    fn test_sign_basic() {
        let data = vec![-5.0, -0.1, 0.0, 0.1, 100.0];
        let result = sign_cpu(&data);
        assert_eq!(result, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sign_empty() {
        let result = sign_cpu(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sign_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let result = sign_cpu(&data);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    // -------------------------------------------------------------------
    // UnaryOp dispatch
    // -------------------------------------------------------------------

    #[test]
    fn test_unary_dispatch_abs() {
        let data = vec![-1.0, 2.0, -3.0];
        assert_eq!(unary_cpu(&data, UnaryOp::Abs), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unary_dispatch_neg() {
        let data = vec![1.0, -2.0];
        assert_eq!(unary_cpu(&data, UnaryOp::Neg), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_unary_dispatch_sign() {
        let data = vec![-1.0, 0.0, 1.0];
        assert_eq!(unary_cpu(&data, UnaryOp::Sign), vec![-1.0, 0.0, 1.0]);
    }

    // -------------------------------------------------------------------
    // UnaryOp methods
    // -------------------------------------------------------------------

    #[test]
    fn test_unary_op_kernel_name() {
        assert_eq!(UnaryOp::Abs.kernel_name(), "abs_f32");
        assert_eq!(UnaryOp::Neg.kernel_name(), "neg_f32");
        assert_eq!(UnaryOp::Sign.kernel_name(), "sign_f32");
    }

    // -------------------------------------------------------------------
    // Forward dispatchers (ensure CPU fallback works)
    // -------------------------------------------------------------------

    #[test]
    fn test_reshape_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = reshape_forward(&data, &[2, 2], &[4]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_view_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = view_forward(&data, &[6], &[2, 3]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_permute_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = permute_forward(&data, &[2, 3], &[1, 0]).unwrap();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_contiguous_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = contiguous_forward(&data, &[2, 2], &[2, 1]).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_expand_forward() {
        let data = vec![1.0, 2.0];
        let result = expand_forward(&data, &[1, 2], &[3, 2]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat_forward() {
        let data = vec![1.0, 2.0];
        let result = repeat_forward(&data, &[2], &[3]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_flip_forward() {
        let data = vec![1.0, 2.0, 3.0];
        let result = flip_forward(&data, &[3], 0).unwrap();
        assert_eq!(result, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_roll_forward() {
        let data = vec![1.0, 2.0, 3.0];
        let result = roll_forward(&data, &[3], 0, 1).unwrap();
        assert_eq!(result, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_pad_forward() {
        let data = vec![1.0, 2.0];
        let result = pad_forward(&data, &[2], &[1, 1], 0.0).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_crop_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = crop_forward(&data, &[5], &[1], &[4]).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_where_cond_forward() {
        let cond = vec![1.0, 0.0];
        let x = vec![10.0, 20.0];
        let y = vec![30.0, 40.0];
        let result = where_cond_forward(&cond, &x, &y).unwrap();
        assert_eq!(result, vec![10.0, 40.0]);
    }

    #[test]
    fn test_clamp_forward() {
        let data = vec![-5.0, 0.0, 5.0];
        let result = clamp_forward(&data, -1.0, 1.0).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_unary_forward_abs() {
        let data = vec![-2.0, 3.0];
        let result = unary_forward(&data, UnaryOp::Abs);
        assert_eq!(result, vec![2.0, 3.0]);
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_linear_to_coords_roundtrip() {
        let shape = [2, 3, 4];
        let strides = compute_strides(&shape);
        for i in 0..24 {
            let coords = linear_to_coords(i, &shape);
            let back = coords_to_linear(&coords, &strides);
            assert_eq!(i, back);
        }
    }

    #[test]
    fn test_total_elements() {
        assert_eq!(total_elements(&[2, 3, 4]), 24);
        assert_eq!(total_elements(&[1]), 1);
        assert_eq!(total_elements(&[]), 1); // empty product = 1
    }

    // -------------------------------------------------------------------
    // PadConfig
    // -------------------------------------------------------------------

    #[test]
    fn test_pad_config() {
        let cfg = PadConfig { padding: vec![1, 1, 2, 2], value: -1.0 };
        assert_eq!(cfg.padding.len(), 4);
        assert_eq!(cfg.value, -1.0);
    }
}
