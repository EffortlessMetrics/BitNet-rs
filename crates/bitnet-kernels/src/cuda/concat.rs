//! CUDA tensor concatenation and split kernels with CPU fallback.
//!
//! # Kernel strategy
//!
//! Provides tensor concatenation and splitting operations along arbitrary axes:
//!
//! - [`concat_along_axis`]: Concatenate multiple tensors along a specified axis.
//! - [`split_along_axis`]: Split a tensor into parts along an axis.
//! - [`stack`]: Stack tensors along a new dimension.
//! - [`unstack`]: Remove a dimension, producing multiple tensors.
//! - [`chunk`]: Split a tensor into equal-sized chunks.
//! - [`interleave`]: Interleave elements from multiple tensors.
//! - [`narrow`]: Select a contiguous slice along a dimension.
//!
//! # CPU fallback
//!
//! Pure-Rust implementations are provided for correctness testing and
//! non-GPU environments.

use bitnet_common::{KernelError, Result};

// ── Helper: compute strides from shape ────────────────────────────────

/// Compute the total number of elements from a shape.
fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ── Concat along axis ─────────────────────────────────────────────────

/// Concatenate multiple tensors along a specified axis.
///
/// All tensors must have the same shape except along the concatenation axis.
///
/// # Arguments
///
/// - `tensors`: Slice of (data, shape) pairs.
/// - `axis`: The dimension along which to concatenate.
///
/// # Returns
///
/// A tuple of (concatenated data, output shape).
///
/// # Errors
///
/// Returns an error if shapes are incompatible or axis is out of bounds.
pub fn concat_along_axis(
    tensors: &[(&[f32], &[usize])],
    axis: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    if tensors.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "concat requires at least one tensor".into(),
        }
        .into());
    }

    let ndim = tensors[0].1.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for {ndim}-D tensor"),
        }
        .into());
    }

    // Validate shapes match on all non-concat axes.
    let ref_shape = tensors[0].1;
    for (i, (_, shape)) in tensors.iter().enumerate().skip(1) {
        if shape.len() != ndim {
            return Err(KernelError::InvalidArguments {
                reason: format!("tensor {i} has {}-D shape, expected {ndim}-D", shape.len()),
            }
            .into());
        }
        for d in 0..ndim {
            if d != axis && shape[d] != ref_shape[d] {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "tensor {i} shape mismatch on dim {d}: {} vs {}",
                        shape[d], ref_shape[d]
                    ),
                }
                .into());
            }
        }
    }

    // Validate data lengths.
    for (i, (data, shape)) in tensors.iter().enumerate() {
        let expected = total_elements(shape);
        if data.len() < expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor {i} data too small: expected {expected}, got {}",
                    data.len()
                ),
            }
            .into());
        }
    }

    // Build output shape.
    let mut out_shape = ref_shape.to_vec();
    out_shape[axis] = tensors.iter().map(|(_, s)| s[axis]).sum();

    let out_total = total_elements(&out_shape);
    let mut output = vec![0.0f32; out_total];

    // Copy data block by block.
    // For axis concatenation: outer dims = product of dims before axis,
    // inner dims = product of dims after axis.
    let outer: usize = ref_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = ref_shape[axis + 1..].iter().product::<usize>().max(1);
    let out_axis_stride = out_shape[axis] * inner;

    let mut axis_offset = 0;
    for (data, shape) in tensors {
        let src_axis_size = shape[axis];
        let src_axis_stride = src_axis_size * inner;
        for o in 0..outer {
            let src_start = o * src_axis_stride;
            let dst_start = o * out_axis_stride + axis_offset * inner;
            let copy_len = src_axis_size * inner;
            output[dst_start..dst_start + copy_len]
                .copy_from_slice(&data[src_start..src_start + copy_len]);
        }
        axis_offset += src_axis_size;
    }

    Ok((output, out_shape))
}

/// Dispatch concat: GPU if available, else CPU fallback.
pub fn concat_along_axis_forward(
    tensors: &[(&[f32], &[usize])],
    axis: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_concat_along_axis(tensors, axis)
        {
            return Ok(result);
        }
    }
    concat_along_axis(tensors, axis)
}

/// CUDA launch stub for concat along axis.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_concat_along_axis(
    _tensors: &[(&[f32], &[usize])],
    _axis: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    Err(KernelError::GpuError {
        reason: "concat CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Split along axis ──────────────────────────────────────────────────

/// Split a tensor along an axis into parts of given sizes.
///
/// # Arguments
///
/// - `data`: Flat tensor data in row-major order.
/// - `shape`: Shape of the input tensor.
/// - `axis`: The dimension along which to split.
/// - `split_sizes`: Size of each split along the axis. Must sum to `shape[axis]`.
///
/// # Returns
///
/// A vector of (data, shape) pairs for each split.
///
/// # Errors
///
/// Returns an error if sizes don't sum to the axis dimension.
pub fn split_along_axis(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    split_sizes: &[usize],
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for {ndim}-D tensor"),
        }
        .into());
    }

    let total_split: usize = split_sizes.iter().sum();
    if total_split != shape[axis] {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "split sizes sum to {total_split}, but axis {axis} has size {}",
                shape[axis]
            ),
        }
        .into());
    }

    let expected = total_elements(shape);
    if data.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("data too small: expected {expected}, got {}", data.len()),
        }
        .into());
    }

    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let src_axis_stride = shape[axis] * inner;

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut axis_offset = 0;

    for &sz in split_sizes {
        let mut out_shape = shape.to_vec();
        out_shape[axis] = sz;
        let out_total = total_elements(&out_shape);
        let mut out_data = vec![0.0f32; out_total];

        let dst_axis_stride = sz * inner;
        for o in 0..outer {
            let src_start = o * src_axis_stride + axis_offset * inner;
            let dst_start = o * dst_axis_stride;
            let copy_len = sz * inner;
            out_data[dst_start..dst_start + copy_len]
                .copy_from_slice(&data[src_start..src_start + copy_len]);
        }

        results.push((out_data, out_shape));
        axis_offset += sz;
    }

    Ok(results)
}

/// Dispatch split: GPU if available, else CPU fallback.
pub fn split_along_axis_forward(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    split_sizes: &[usize],
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_split_along_axis(data, shape, axis, split_sizes)
        {
            return Ok(result);
        }
    }
    split_along_axis(data, shape, axis, split_sizes)
}

/// CUDA launch stub for split along axis.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_split_along_axis(
    _data: &[f32],
    _shape: &[usize],
    _axis: usize,
    _split_sizes: &[usize],
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    Err(KernelError::GpuError {
        reason: "split CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Stack ─────────────────────────────────────────────────────────────

/// Stack tensors along a new dimension.
///
/// All tensors must have the same shape. A new dimension of size N is
/// inserted at position `axis`.
///
/// # Arguments
///
/// - `tensors`: Slice of (data, shape) pairs with identical shapes.
/// - `axis`: Position for the new dimension (0 ≤ axis ≤ ndim).
///
/// # Returns
///
/// A tuple of (stacked data, output shape).
///
/// # Errors
///
/// Returns an error if shapes differ or axis is out of bounds.
pub fn stack(tensors: &[(&[f32], &[usize])], axis: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    if tensors.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "stack requires at least one tensor".into(),
        }
        .into());
    }

    let ref_shape = tensors[0].1;
    let ndim = ref_shape.len();
    if axis > ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for stack on {ndim}-D tensors"),
        }
        .into());
    }

    let elem_count = total_elements(ref_shape);
    for (i, (data, shape)) in tensors.iter().enumerate() {
        if *shape != ref_shape {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor {i} shape {shape:?} differs from tensor 0 shape {ref_shape:?}"
                ),
            }
            .into());
        }
        if data.len() < elem_count {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor {i} data too small: expected {elem_count}, got {}",
                    data.len()
                ),
            }
            .into());
        }
    }

    let n = tensors.len();
    // Output shape: insert n at position axis.
    let mut out_shape = Vec::with_capacity(ndim + 1);
    out_shape.extend_from_slice(&ref_shape[..axis]);
    out_shape.push(n);
    out_shape.extend_from_slice(&ref_shape[axis..]);

    let out_total = total_elements(&out_shape);
    let mut output = vec![0.0f32; out_total];

    // outer = product of dims before axis, inner = product of dims from axis onward.
    let outer: usize = ref_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = ref_shape[axis..].iter().product::<usize>().max(1);

    for (t_idx, (data, _)) in tensors.iter().enumerate() {
        for o in 0..outer {
            let src_off = o * inner;
            let dst_off = o * n * inner + t_idx * inner;
            output[dst_off..dst_off + inner].copy_from_slice(&data[src_off..src_off + inner]);
        }
    }

    Ok((output, out_shape))
}

/// Dispatch stack: GPU if available, else CPU fallback.
pub fn stack_forward(
    tensors: &[(&[f32], &[usize])],
    axis: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_stack(tensors, axis)
        {
            return Ok(result);
        }
    }
    stack(tensors, axis)
}

/// CUDA launch stub for stack.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_stack(
    _tensors: &[(&[f32], &[usize])],
    _axis: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    Err(KernelError::GpuError {
        reason: "stack CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Unstack ───────────────────────────────────────────────────────────

/// Unstack a tensor along a dimension, producing one tensor per index.
///
/// Reverses [`stack`]: removes the dimension at `axis` and returns
/// `shape[axis]` tensors.
///
/// # Errors
///
/// Returns an error if axis is out of bounds.
pub fn unstack(data: &[f32], shape: &[usize], axis: usize) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "cannot unstack a 0-D tensor".into() }.into()
        );
    }
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for {ndim}-D tensor"),
        }
        .into());
    }

    let expected = total_elements(shape);
    if data.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("data too small: expected {expected}, got {}", data.len()),
        }
        .into());
    }

    let n = shape[axis];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);

    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let elem_count = total_elements(&out_shape);

    let mut results = Vec::with_capacity(n);
    for t in 0..n {
        let mut out_data = vec![0.0f32; elem_count];
        for o in 0..outer {
            let src_off = o * n * inner + t * inner;
            let dst_off = o * inner;
            out_data[dst_off..dst_off + inner].copy_from_slice(&data[src_off..src_off + inner]);
        }
        results.push((out_data, out_shape.clone()));
    }

    Ok(results)
}

/// Dispatch unstack: GPU if available, else CPU fallback.
pub fn unstack_forward(
    data: &[f32],
    shape: &[usize],
    axis: usize,
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_unstack(data, shape, axis)
        {
            return Ok(result);
        }
    }
    unstack(data, shape, axis)
}

/// CUDA launch stub for unstack.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_unstack(
    _data: &[f32],
    _shape: &[usize],
    _axis: usize,
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    Err(KernelError::GpuError {
        reason: "unstack CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Chunk ─────────────────────────────────────────────────────────────

/// Split a tensor into equal-sized chunks along an axis.
///
/// If the axis dimension is not evenly divisible by `num_chunks`,
/// the last chunk will be smaller.
///
/// # Errors
///
/// Returns an error if `num_chunks` is zero or axis is out of bounds.
pub fn chunk(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    num_chunks: usize,
) -> Result<Vec<(Vec<f32>, Vec<usize>)>> {
    if num_chunks == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "num_chunks must be > 0".into() }.into()
        );
    }
    let ndim = shape.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for {ndim}-D tensor"),
        }
        .into());
    }

    let axis_size = shape[axis];
    let chunk_size = axis_size.div_ceil(num_chunks);
    let mut split_sizes = Vec::with_capacity(num_chunks);
    let mut remaining = axis_size;
    for _ in 0..num_chunks {
        if remaining == 0 {
            break;
        }
        let sz = chunk_size.min(remaining);
        split_sizes.push(sz);
        remaining -= sz;
    }

    split_along_axis(data, shape, axis, &split_sizes)
}

// ── Interleave ────────────────────────────────────────────────────────

/// Interleave elements from multiple 1-D tensors.
///
/// Given N tensors of equal length L, produces a tensor of length N*L
/// where elements are taken round-robin from each input tensor.
///
/// # Errors
///
/// Returns an error if tensors have different lengths or are empty.
pub fn interleave(tensors: &[&[f32]]) -> Result<Vec<f32>> {
    if tensors.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "interleave requires at least one tensor".into(),
        }
        .into());
    }

    let len = tensors[0].len();
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.len() != len {
            return Err(KernelError::InvalidArguments {
                reason: format!("tensor {i} has length {}, expected {len}", t.len()),
            }
            .into());
        }
    }

    let n = tensors.len();
    let mut output = vec![0.0f32; n * len];
    for i in 0..len {
        for (t, tensor) in tensors.iter().enumerate() {
            output[i * n + t] = tensor[i];
        }
    }
    Ok(output)
}

/// Dispatch interleave: GPU if available, else CPU fallback.
pub fn interleave_forward(tensors: &[&[f32]]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_interleave(tensors)
        {
            return Ok(result);
        }
    }
    interleave(tensors)
}

/// CUDA launch stub for interleave.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_interleave(_tensors: &[&[f32]]) -> Result<Vec<f32>> {
    Err(KernelError::GpuError {
        reason: "interleave CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── Narrow ────────────────────────────────────────────────────────────

/// Select a contiguous slice along a dimension.
///
/// Returns the sub-tensor `data[..., start:start+length, ...]` along
/// the given dimension.
///
/// # Errors
///
/// Returns an error if the range is out of bounds.
pub fn narrow(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    start: usize,
    length: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("axis {axis} out of bounds for {ndim}-D tensor"),
        }
        .into());
    }
    if start + length > shape[axis] {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "narrow range [{start}, {}) exceeds axis size {}",
                start + length,
                shape[axis]
            ),
        }
        .into());
    }

    let expected = total_elements(shape);
    if data.len() < expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("data too small: expected {expected}, got {}", data.len()),
        }
        .into());
    }

    let mut out_shape = shape.to_vec();
    out_shape[axis] = length;

    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let src_axis_stride = shape[axis] * inner;

    let out_total = total_elements(&out_shape);
    let mut output = vec![0.0f32; out_total];
    let dst_axis_stride = length * inner;

    for o in 0..outer {
        let src_start = o * src_axis_stride + start * inner;
        let dst_start = o * dst_axis_stride;
        let copy_len = length * inner;
        output[dst_start..dst_start + copy_len]
            .copy_from_slice(&data[src_start..src_start + copy_len]);
    }

    Ok((output, out_shape))
}

/// Dispatch narrow: GPU if available, else CPU fallback.
pub fn narrow_forward(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    start: usize,
    length: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(result) = launch_narrow(data, shape, axis, start, length)
        {
            return Ok(result);
        }
    }
    narrow(data, shape, axis, start, length)
}

/// CUDA launch stub for narrow.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_narrow(
    _data: &[f32],
    _shape: &[usize],
    _axis: usize,
    _start: usize,
    _length: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    Err(KernelError::GpuError {
        reason: "narrow CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ── CUDA kernel source (feature-gated) ───────────────────────────────

/// Inline CUDA C source for tensor concatenation kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const CONCAT_KERNEL_SRC: &str = r#"
extern "C" __global__ void concat_f32(
    const float* __restrict__* inputs,
    const int* __restrict__ axis_sizes,
    float* __restrict__ output,
    int num_inputs,
    int outer,
    int inner,
    int total_axis)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * total_axis * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / (total_axis * inner);
        int rem = i % (total_axis * inner);
        int a = rem / inner;
        int in_idx = rem % inner;

        // Find which input tensor this axis position belongs to.
        int acc = 0;
        int src_tensor = 0;
        for (int t = 0; t < num_inputs; t++) {
            if (a < acc + axis_sizes[t]) {
                src_tensor = t;
                break;
            }
            acc += axis_sizes[t];
        }
        int local_a = a - acc;
        int src_idx = o * axis_sizes[src_tensor] * inner + local_a * inner + in_idx;
        output[i] = inputs[src_tensor][src_idx];
    }
}
"#;

/// Inline CUDA C source for interleave kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const INTERLEAVE_KERNEL_SRC: &str = r#"
extern "C" __global__ void interleave_f32(
    const float* __restrict__* inputs,
    float* __restrict__ output,
    int num_tensors,
    int tensor_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tensors * tensor_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int elem = i / num_tensors;
        int t = i % num_tensors;
        output[i] = inputs[t][elem];
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    // ── concat_along_axis tests ───────────────────────────────────

    #[test]
    fn test_concat_dim0_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (out, shape) = concat_along_axis(&[(&a, &[3][..]), (&b, &[3][..])], 0).unwrap();
        assert_eq!(shape, &[6]);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_concat_dim0_matrices() {
        // [2,3] + [1,3] → [3,3]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0];
        let (out, shape) = concat_along_axis(&[(&a, &[2, 3][..]), (&b, &[1, 3][..])], 0).unwrap();
        assert_eq!(shape, &[3, 3]);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
    }

    #[test]
    fn test_concat_dim1_matrices() {
        // [2,2] + [2,1] → [2,3]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0];
        let (out, shape) = concat_along_axis(&[(&a, &[2, 2][..]), (&b, &[2, 1][..])], 1).unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_close(&out, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_concat_dim2_3d() {
        // [1,2,2] + [1,2,3] → [1,2,5]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (out, shape) =
            concat_along_axis(&[(&a, &[1, 2, 2][..]), (&b, &[1, 2, 3][..])], 2).unwrap();
        assert_eq!(shape, &[1, 2, 5]);
        assert_close(&out, &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0], 1e-6);
    }

    #[test]
    fn test_concat_three_tensors() {
        let a = vec![1.0];
        let b = vec![2.0];
        let c = vec![3.0];
        let (out, shape) =
            concat_along_axis(&[(&a, &[1][..]), (&b, &[1][..]), (&c, &[1][..])], 0).unwrap();
        assert_eq!(shape, &[3]);
        assert_close(&out, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_concat_empty_error() {
        let tensors: Vec<(&[f32], &[usize])> = vec![];
        assert!(concat_along_axis(&tensors, 0).is_err());
    }

    #[test]
    fn test_concat_axis_oob() {
        let a = vec![1.0, 2.0];
        assert!(concat_along_axis(&[(&a, &[2][..])], 1).is_err());
    }

    #[test]
    fn test_concat_shape_mismatch() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(concat_along_axis(&[(&a, &[2, 2][..]), (&b, &[2, 3][..])], 0).is_err());
    }

    #[test]
    fn test_concat_ndim_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(concat_along_axis(&[(&a, &[2][..]), (&b, &[1, 2][..])], 0).is_err());
    }

    #[test]
    fn test_concat_forward_cpu() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let (out, shape) = concat_along_axis_forward(&[(&a, &[2][..]), (&b, &[2][..])], 0).unwrap();
        assert_eq!(shape, &[4]);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    // ── split_along_axis tests ────────────────────────────────────

    #[test]
    fn test_split_dim0_equal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let parts = split_along_axis(&data, &[6], 0, &[3, 3]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &[1.0, 2.0, 3.0], 1e-6);
        assert_close(&parts[1].0, &[4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_split_dim0_unequal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let parts = split_along_axis(&data, &[5], 0, &[2, 3]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &[1.0, 2.0], 1e-6);
        assert_close(&parts[1].0, &[3.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_split_dim1() {
        // [2, 4] → [2, 1] + [2, 3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let parts = split_along_axis(&data, &[2, 4], 1, &[1, 3]).unwrap();
        assert_eq!(parts[0].1, &[2, 1]);
        assert_eq!(parts[1].1, &[2, 3]);
        assert_close(&parts[0].0, &[1.0, 5.0], 1e-6);
        assert_close(&parts[1].0, &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_split_sum_mismatch() {
        let data = vec![1.0; 6];
        assert!(split_along_axis(&data, &[6], 0, &[2, 2]).is_err());
    }

    #[test]
    fn test_split_axis_oob() {
        let data = vec![1.0; 4];
        assert!(split_along_axis(&data, &[4], 1, &[4]).is_err());
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let shape = [3, 3];
        let parts = split_along_axis(&data, &shape, 0, &[1, 2]).unwrap();

        let tensors: Vec<(&[f32], &[usize])> =
            parts.iter().map(|(d, s)| (d.as_slice(), s.as_slice())).collect();
        let (reconstructed, out_shape) = concat_along_axis(&tensors, 0).unwrap();
        assert_eq!(out_shape, &[3, 3]);
        assert_close(&reconstructed, &data, 1e-6);
    }

    #[test]
    fn test_split_concat_roundtrip_dim1() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let parts = split_along_axis(&data, &shape, 1, &[1, 2]).unwrap();

        let tensors: Vec<(&[f32], &[usize])> =
            parts.iter().map(|(d, s)| (d.as_slice(), s.as_slice())).collect();
        let (reconstructed, out_shape) = concat_along_axis(&tensors, 1).unwrap();
        assert_eq!(out_shape, &[2, 3]);
        assert_close(&reconstructed, &data, 1e-6);
    }

    #[test]
    fn test_split_forward_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let parts = split_along_axis_forward(&data, &[4], 0, &[2, 2]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &[1.0, 2.0], 1e-6);
    }

    // ── stack tests ───────────────────────────────────────────────

    #[test]
    fn test_stack_dim0_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (out, shape) = stack(&[(&a, &[3][..]), (&b, &[3][..])], 0).unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_stack_dim1_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (out, shape) = stack(&[(&a, &[3][..]), (&b, &[3][..])], 1).unwrap();
        assert_eq!(shape, &[3, 2]);
        // Interleaved: [1, 4, 2, 5, 3, 6]
        assert_close(&out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    #[test]
    fn test_stack_dim0_matrices() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [2, 2]
        let (out, shape) = stack(&[(&a, &[2, 2][..]), (&b, &[2, 2][..])], 0).unwrap();
        assert_eq!(shape, &[2, 2, 2]);
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_stack_single_tensor() {
        let a = vec![1.0, 2.0];
        let (out, shape) = stack(&[(&a, &[2][..])], 0).unwrap();
        assert_eq!(shape, &[1, 2]);
        assert_close(&out, &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_stack_empty_error() {
        let tensors: Vec<(&[f32], &[usize])> = vec![];
        assert!(stack(&tensors, 0).is_err());
    }

    #[test]
    fn test_stack_shape_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(stack(&[(&a, &[2][..]), (&b, &[3][..])], 0).is_err());
    }

    #[test]
    fn test_stack_axis_oob() {
        let a = vec![1.0, 2.0];
        // For a 1-D tensor, axis can be 0 or 1, but not 2.
        assert!(stack(&[(&a, &[2][..])], 2).is_err());
    }

    #[test]
    fn test_stack_forward_cpu() {
        let a = vec![1.0];
        let b = vec![2.0];
        let (out, shape) = stack_forward(&[(&a, &[1][..]), (&b, &[1][..])], 0).unwrap();
        assert_eq!(shape, &[2, 1]);
        assert_close(&out, &[1.0, 2.0], 1e-6);
    }

    // ── unstack tests ─────────────────────────────────────────────

    #[test]
    fn test_unstack_dim0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let parts = unstack(&data, &[2, 3], 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1, &[3]);
        assert_close(&parts[0].0, &[1.0, 2.0, 3.0], 1e-6);
        assert_close(&parts[1].0, &[4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_unstack_dim1() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let parts = unstack(&data, &[2, 2], 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1, &[2]);
        assert_close(&parts[0].0, &[1.0, 3.0], 1e-6);
        assert_close(&parts[1].0, &[2.0, 4.0], 1e-6);
    }

    #[test]
    fn test_unstack_axis_oob() {
        let data = vec![1.0; 4];
        assert!(unstack(&data, &[4], 1).is_err());
    }

    #[test]
    fn test_unstack_0d_error() {
        assert!(unstack(&[], &[], 0).is_err());
    }

    #[test]
    fn test_stack_unstack_roundtrip() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let (stacked, stacked_shape) = stack(&[(&a, &[3][..]), (&b, &[3][..])], 0).unwrap();
        let parts = unstack(&stacked, &stacked_shape, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &a, 1e-6);
        assert_close(&parts[1].0, &b, 1e-6);
    }

    #[test]
    fn test_stack_unstack_roundtrip_dim1() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let (stacked, stacked_shape) = stack(&[(&a, &[2][..]), (&b, &[2][..])], 1).unwrap();
        let parts = unstack(&stacked, &stacked_shape, 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &a, 1e-6);
        assert_close(&parts[1].0, &b, 1e-6);
    }

    #[test]
    fn test_unstack_forward_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let parts = unstack_forward(&data, &[2, 2], 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &[1.0, 2.0], 1e-6);
    }

    // ── chunk tests ───────────────────────────────────────────────

    #[test]
    fn test_chunk_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let parts = chunk(&data, &[6], 0, 3).unwrap();
        assert_eq!(parts.len(), 3);
        assert_close(&parts[0].0, &[1.0, 2.0], 1e-6);
        assert_close(&parts[1].0, &[3.0, 4.0], 1e-6);
        assert_close(&parts[2].0, &[5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_chunk_uneven() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let parts = chunk(&data, &[5], 0, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_close(&parts[0].0, &[1.0, 2.0, 3.0], 1e-6);
        assert_close(&parts[1].0, &[4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_chunk_single() {
        let data = vec![1.0, 2.0, 3.0];
        let parts = chunk(&data, &[3], 0, 1).unwrap();
        assert_eq!(parts.len(), 1);
        assert_close(&parts[0].0, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_chunk_more_than_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let parts = chunk(&data, &[3], 0, 5).unwrap();
        // Each chunk is size 1, only 3 chunks produced.
        assert_eq!(parts.len(), 3);
        for (i, (d, _)) in parts.iter().enumerate() {
            assert_close(d, &[data[i]], 1e-6);
        }
    }

    #[test]
    fn test_chunk_zero_error() {
        let data = vec![1.0; 4];
        assert!(chunk(&data, &[4], 0, 0).is_err());
    }

    #[test]
    fn test_chunk_2d() {
        // [4, 2] chunked along dim 0 into 2 chunks
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let parts = chunk(&data, &[4, 2], 0, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1, &[2, 2]);
        assert_close(&parts[0].0, &[1.0, 2.0, 3.0, 4.0], 1e-6);
        assert_close(&parts[1].0, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    // ── interleave tests ──────────────────────────────────────────

    #[test]
    fn test_interleave_basic() {
        let a = vec![1.0, 3.0, 5.0];
        let b = vec![2.0, 4.0, 6.0];
        let out = interleave(&[&a, &b]).unwrap();
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_interleave_three_tensors() {
        let a = vec![1.0, 4.0];
        let b = vec![2.0, 5.0];
        let c = vec![3.0, 6.0];
        let out = interleave(&[&a, &b, &c]).unwrap();
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_interleave_single() {
        let a = vec![1.0, 2.0, 3.0];
        let out = interleave(&[&a]).unwrap();
        assert_close(&out, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_interleave_empty_error() {
        let tensors: Vec<&[f32]> = vec![];
        assert!(interleave(&tensors).is_err());
    }

    #[test]
    fn test_interleave_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(interleave(&[&a[..], &b[..]]).is_err());
    }

    #[test]
    fn test_interleave_single_element() {
        let a = vec![10.0];
        let b = vec![20.0];
        let out = interleave(&[&a, &b]).unwrap();
        assert_close(&out, &[10.0, 20.0], 1e-6);
    }

    #[test]
    fn test_interleave_forward_cpu() {
        let a = vec![1.0, 3.0];
        let b = vec![2.0, 4.0];
        let out = interleave_forward(&[&a, &b]).unwrap();
        assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    // ── narrow tests ──────────────────────────────────────────────

    #[test]
    fn test_narrow_dim0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [6]
        let (out, shape) = narrow(&data, &[6], 0, 1, 3).unwrap();
        assert_eq!(shape, &[3]);
        assert_close(&out, &[2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_narrow_dim0_matrix() {
        // [3, 2], narrow dim 0 from index 1, length 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, shape) = narrow(&data, &[3, 2], 0, 1, 2).unwrap();
        assert_eq!(shape, &[2, 2]);
        assert_close(&out, &[3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_narrow_dim1_matrix() {
        // [2, 4], narrow dim 1 from index 1, length 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (out, shape) = narrow(&data, &[2, 4], 1, 1, 2).unwrap();
        assert_eq!(shape, &[2, 2]);
        assert_close(&out, &[2.0, 3.0, 6.0, 7.0], 1e-6);
    }

    #[test]
    fn test_narrow_full_range() {
        let data = vec![1.0, 2.0, 3.0];
        let (out, shape) = narrow(&data, &[3], 0, 0, 3).unwrap();
        assert_eq!(shape, &[3]);
        assert_close(&out, &data, 1e-6);
    }

    #[test]
    fn test_narrow_single_element() {
        let data = vec![10.0, 20.0, 30.0];
        let (out, shape) = narrow(&data, &[3], 0, 1, 1).unwrap();
        assert_eq!(shape, &[1]);
        assert_close(&out, &[20.0], 1e-6);
    }

    #[test]
    fn test_narrow_oob() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(narrow(&data, &[3], 0, 2, 3).is_err());
    }

    #[test]
    fn test_narrow_axis_oob() {
        let data = vec![1.0, 2.0];
        assert!(narrow(&data, &[2], 1, 0, 1).is_err());
    }

    #[test]
    fn test_narrow_3d() {
        // [2, 3, 2], narrow dim 1 from index 1, length 1
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0,  3.0, 4.0,  5.0, 6.0,
            7.0, 8.0,  9.0, 10.0, 11.0, 12.0,
        ];
        let (out, shape) = narrow(&data, &[2, 3, 2], 1, 1, 1).unwrap();
        assert_eq!(shape, &[2, 1, 2]);
        assert_close(&out, &[3.0, 4.0, 9.0, 10.0], 1e-6);
    }

    #[test]
    fn test_narrow_forward_cpu() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (out, shape) = narrow_forward(&data, &[5], 0, 1, 3).unwrap();
        assert_eq!(shape, &[3]);
        assert_close(&out, &[2.0, 3.0, 4.0], 1e-6);
    }

    // ── helper tests ──────────────────────────────────────────────

    #[test]
    fn test_total_elements() {
        assert_eq!(total_elements(&[2, 3, 4]), 24);
        assert_eq!(total_elements(&[1]), 1);
        assert_eq!(total_elements(&[]), 1);
    }
}
