//! OpenCL tensor concatenation and splitting operations with CPU reference.
//!
//! # Overview
//!
//! This module provides operations for combining and decomposing tensors
//! along arbitrary axes. Every operation has a CPU reference implementation
//! that works without any OpenCL runtime, plus an embedded OpenCL C kernel
//! source for GPU dispatch.
//!
//! # Operations
//!
//! - **Concat** — join tensors along an existing axis
//! - **Split** — divide a tensor into N equal parts
//! - **Chunk** — divide a tensor into parts of a given size
//! - **Stack** — join tensors along a *new* axis (adds a dimension)
//! - **Unstack** — remove a dimension, producing a list of tensors
//! - **Slice** — extract a sub-tensor with start:stop:step per dimension
//! - **PaddedConcat** — concatenate with zero/constant padding to match shapes
//!
//! # OpenCL kernel
//!
//! The embedded OpenCL C source (`TENSOR_CONCAT_CL`) contains kernels for
//! axis-0 and general-axis concatenation, splitting, and padded copy.

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for tensor concatenation and splitting.
pub const TENSOR_CONCAT_CL: &str = include_str!("gpu/kernels/tensor_concat.cl");

// ── Tensor descriptor ────────────────────────────────────────────

/// Lightweight shape descriptor for dense, row-major tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes, outermost first (e.g. `[batch, seq, hidden]`).
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new shape from the given dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Product of dimensions before `axis`.
    fn outer_size(&self, axis: usize) -> usize {
        self.dims[..axis].iter().product()
    }

    /// Product of dimensions after `axis`.
    fn inner_size(&self, axis: usize) -> usize {
        if axis + 1 >= self.dims.len() { 1 } else { self.dims[axis + 1..].iter().product() }
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

// ── TensorConcat ─────────────────────────────────────────────────

/// Concatenate multiple tensors along an existing axis (CPU reference).
///
/// All input tensors must have the same number of dimensions and agree
/// on every dimension except `axis`.
///
/// # Arguments
///
/// * `inputs`  — slices of f32 data for each input tensor
/// * `shapes`  — corresponding shapes
/// * `axis`    — the axis along which to concatenate
/// * `output`  — pre-allocated buffer for the result
///
/// # Returns
///
/// The shape of the concatenated output tensor.
pub fn tensor_concat_ref(
    inputs: &[&[f32]],
    shapes: &[&TensorShape],
    axis: usize,
    output: &mut [f32],
) -> Result<TensorShape> {
    if inputs.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "tensor_concat: inputs must not be empty".into(),
        }
        .into());
    }
    if inputs.len() != shapes.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_concat: inputs.len()={} != shapes.len()={}",
                inputs.len(),
                shapes.len()
            ),
        }
        .into());
    }

    let ndim = shapes[0].ndim();
    if ndim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "tensor_concat: cannot concat scalar (0-d) tensors".into(),
        }
        .into());
    }
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("tensor_concat: axis {} out of range for {}-d tensor", axis, ndim),
        }
        .into());
    }

    // Validate shapes match on non-concat dims
    for (i, shape) in shapes.iter().enumerate() {
        if shape.ndim() != ndim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor_concat: tensor {} has {} dims, expected {}",
                    i,
                    shape.ndim(),
                    ndim
                ),
            }
            .into());
        }
        for d in 0..ndim {
            if d != axis && shape.dims[d] != shapes[0].dims[d] {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "tensor_concat: tensor {} dim {} is {}, expected {}",
                        i, d, shape.dims[d], shapes[0].dims[d]
                    ),
                }
                .into());
            }
        }
        if inputs[i].len() != shape.numel() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor_concat: tensor {} data len {} != shape numel {}",
                    i,
                    inputs[i].len(),
                    shape.numel()
                ),
            }
            .into());
        }
    }

    // Compute output shape
    let concat_dim: usize = shapes.iter().map(|s| s.dims[axis]).sum();
    let mut out_dims = shapes[0].dims.clone();
    out_dims[axis] = concat_dim;
    let out_shape = TensorShape::new(out_dims);

    let expected_len = out_shape.numel();
    if output.len() < expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_concat: output buffer len {} < required {}",
                output.len(),
                expected_len
            ),
        }
        .into());
    }

    // Copy data
    let inner_size = shapes[0].inner_size(axis);
    let outer_size = shapes[0].outer_size(axis);
    let dst_axis = concat_dim;

    let mut axis_offset = 0usize;
    for (tensor_data, shape) in inputs.iter().zip(shapes.iter()) {
        let src_axis = shape.dims[axis];
        for o in 0..outer_size {
            for a in 0..src_axis {
                let src_start = (o * src_axis + a) * inner_size;
                let dst_start = (o * dst_axis + (axis_offset + a)) * inner_size;
                output[dst_start..dst_start + inner_size]
                    .copy_from_slice(&tensor_data[src_start..src_start + inner_size]);
            }
        }
        axis_offset += src_axis;
    }

    Ok(out_shape)
}

// ── TensorSplit ──────────────────────────────────────────────────

/// Split a tensor into `n` equal parts along `axis` (CPU reference).
///
/// # Errors
///
/// Returns an error if the axis dimension is not evenly divisible by `n`.
pub fn tensor_split_ref(
    input: &[f32],
    shape: &TensorShape,
    axis: usize,
    n: usize,
    outputs: &mut Vec<Vec<f32>>,
) -> Result<Vec<TensorShape>> {
    validate_axis(shape, axis, "tensor_split")?;
    if n == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "tensor_split: n must be > 0".into() }.into()
        );
    }
    if !shape.dims[axis].is_multiple_of(n) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_split: axis {} size {} not divisible by {}",
                axis, shape.dims[axis], n
            ),
        }
        .into());
    }
    if input.len() != shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_split: data len {} != shape numel {}",
                input.len(),
                shape.numel()
            ),
        }
        .into());
    }

    let chunk = shape.dims[axis] / n;
    let mut part_dims = shape.dims.clone();
    part_dims[axis] = chunk;
    let part_shape = TensorShape::new(part_dims);
    let part_numel = part_shape.numel();

    outputs.clear();
    outputs.resize_with(n, || vec![0.0; part_numel]);

    let outer = shape.outer_size(axis);
    let inner = shape.inner_size(axis);
    let src_axis = shape.dims[axis];

    for (part_idx, part_buf) in outputs.iter_mut().enumerate() {
        let axis_offset = part_idx * chunk;
        for o in 0..outer {
            for a in 0..chunk {
                let src_start = (o * src_axis + (axis_offset + a)) * inner;
                let dst_start = (o * chunk + a) * inner;
                part_buf[dst_start..dst_start + inner]
                    .copy_from_slice(&input[src_start..src_start + inner]);
            }
        }
    }

    Ok(vec![part_shape; n])
}

// ── TensorChunk ──────────────────────────────────────────────────

/// Split a tensor into chunks of `chunk_size` along `axis`.
///
/// The last chunk may be smaller than `chunk_size`.
pub fn tensor_chunk_ref(
    input: &[f32],
    shape: &TensorShape,
    axis: usize,
    chunk_size: usize,
    outputs: &mut Vec<Vec<f32>>,
) -> Result<Vec<TensorShape>> {
    validate_axis(shape, axis, "tensor_chunk")?;
    if chunk_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "tensor_chunk: chunk_size must be > 0".into(),
        }
        .into());
    }
    if input.len() != shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_chunk: data len {} != shape numel {}",
                input.len(),
                shape.numel()
            ),
        }
        .into());
    }

    let axis_dim = shape.dims[axis];
    let n_full = axis_dim / chunk_size;
    let remainder = axis_dim % chunk_size;
    let n_chunks = n_full + if remainder > 0 { 1 } else { 0 };

    let outer = shape.outer_size(axis);
    let inner = shape.inner_size(axis);
    let src_axis = axis_dim;

    outputs.clear();
    let mut out_shapes = Vec::with_capacity(n_chunks);

    let mut axis_offset = 0usize;
    for c in 0..n_chunks {
        let this_chunk = if c < n_full { chunk_size } else { remainder };
        let mut part_dims = shape.dims.clone();
        part_dims[axis] = this_chunk;
        let part_shape = TensorShape::new(part_dims);
        let part_numel = part_shape.numel();
        let mut buf = vec![0.0f32; part_numel];

        for o in 0..outer {
            for a in 0..this_chunk {
                let src_start = (o * src_axis + (axis_offset + a)) * inner;
                let dst_start = (o * this_chunk + a) * inner;
                buf[dst_start..dst_start + inner]
                    .copy_from_slice(&input[src_start..src_start + inner]);
            }
        }

        axis_offset += this_chunk;
        outputs.push(buf);
        out_shapes.push(part_shape);
    }

    Ok(out_shapes)
}

// ── TensorStack ──────────────────────────────────────────────────

/// Stack tensors along a *new* axis, adding a dimension (CPU reference).
///
/// All tensors must have identical shapes. The resulting tensor has
/// `ndim + 1` dimensions with `dims[axis] == inputs.len()`.
pub fn tensor_stack_ref(
    inputs: &[&[f32]],
    shapes: &[&TensorShape],
    axis: usize,
    output: &mut [f32],
) -> Result<TensorShape> {
    if inputs.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "tensor_stack: inputs must not be empty".into(),
        }
        .into());
    }
    if inputs.len() != shapes.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_stack: inputs.len()={} != shapes.len()={}",
                inputs.len(),
                shapes.len()
            ),
        }
        .into());
    }

    let ndim = shapes[0].ndim();
    if axis > ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_stack: axis {} out of range for {}-d tensors (max {})",
                axis, ndim, ndim
            ),
        }
        .into());
    }

    // All shapes must be identical
    for (i, shape) in shapes.iter().enumerate() {
        if shape.dims != shapes[0].dims {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor_stack: tensor {} shape {} != expected {}",
                    i, shape, shapes[0]
                ),
            }
            .into());
        }
        if inputs[i].len() != shape.numel() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor_stack: tensor {} data len {} != numel {}",
                    i,
                    inputs[i].len(),
                    shape.numel()
                ),
            }
            .into());
        }
    }

    // Build output shape: insert new dim at `axis`
    let n = inputs.len();
    let mut out_dims = shapes[0].dims.clone();
    out_dims.insert(axis, n);
    let out_shape = TensorShape::new(out_dims);

    if output.len() < out_shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_stack: output len {} < required {}",
                output.len(),
                out_shape.numel()
            ),
        }
        .into());
    }

    // outer = product of dims before axis in *input* shape
    let outer: usize = if axis == 0 { 1 } else { shapes[0].dims[..axis].iter().product() };
    // inner = product of dims from axis onward in *input* shape
    let inner: usize = if axis >= ndim { 1 } else { shapes[0].dims[axis..].iter().product() };

    for (t, tensor) in inputs.iter().enumerate() {
        for o in 0..outer {
            let src_start = o * inner;
            let dst_start = (o * n + t) * inner;
            output[dst_start..dst_start + inner]
                .copy_from_slice(&tensor[src_start..src_start + inner]);
        }
    }

    Ok(out_shape)
}

// ── TensorUnstack ────────────────────────────────────────────────

/// Remove a dimension from a tensor, producing one sub-tensor per index
/// along that axis (CPU reference).
pub fn tensor_unstack_ref(
    input: &[f32],
    shape: &TensorShape,
    axis: usize,
    outputs: &mut Vec<Vec<f32>>,
) -> Result<Vec<TensorShape>> {
    validate_axis(shape, axis, "tensor_unstack")?;
    if input.len() != shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!("tensor_unstack: data len {} != numel {}", input.len(), shape.numel()),
        }
        .into());
    }

    let n = shape.dims[axis];
    let mut sub_dims = shape.dims.clone();
    sub_dims.remove(axis);
    // For scalar result when unstacking a 1-d tensor
    if sub_dims.is_empty() {
        sub_dims.push(1);
    }
    let sub_shape = TensorShape::new(sub_dims);
    let sub_numel = sub_shape.numel();

    // outer = product of dims before axis
    let outer = shape.outer_size(axis);
    // inner = product of dims after axis
    let inner = shape.inner_size(axis);

    outputs.clear();
    outputs.resize_with(n, || vec![0.0; sub_numel]);

    for (t, out_buf) in outputs.iter_mut().enumerate() {
        for o in 0..outer {
            let src_start = (o * n + t) * inner;
            let dst_start = o * inner;
            out_buf[dst_start..dst_start + inner]
                .copy_from_slice(&input[src_start..src_start + inner]);
        }
    }

    Ok(vec![sub_shape; n])
}

// ── TensorSlice ──────────────────────────────────────────────────

/// Per-dimension slice specification: `start:stop:step`.
#[derive(Debug, Clone, Copy)]
pub struct SliceSpec {
    /// Start index (inclusive). Default 0.
    pub start: usize,
    /// Stop index (exclusive). Must be <= dim size.
    pub stop: usize,
    /// Step (must be >= 1). Default 1.
    pub step: usize,
}

impl SliceSpec {
    /// Create a new slice specification.
    pub fn new(start: usize, stop: usize, step: usize) -> Self {
        Self { start, stop, step }
    }

    /// Full slice covering the entire dimension of size `dim`.
    pub fn full(dim: usize) -> Self {
        Self { start: 0, stop: dim, step: 1 }
    }

    /// Number of elements selected by this slice.
    pub fn output_size(&self) -> usize {
        if self.stop <= self.start || self.step == 0 {
            0
        } else {
            (self.stop - self.start).div_ceil(self.step)
        }
    }
}

/// Slice a tensor with per-dimension start:stop:step (CPU reference).
///
/// `specs` must have exactly one entry per dimension.
pub fn tensor_slice_ref(
    input: &[f32],
    shape: &TensorShape,
    specs: &[SliceSpec],
    output: &mut [f32],
) -> Result<TensorShape> {
    let ndim = shape.ndim();
    if specs.len() != ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("tensor_slice: specs.len()={} != ndim={}", specs.len(), ndim),
        }
        .into());
    }
    if input.len() != shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!("tensor_slice: data len {} != numel {}", input.len(), shape.numel()),
        }
        .into());
    }

    // Validate specs
    for (d, spec) in specs.iter().enumerate() {
        if spec.step == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("tensor_slice: step=0 at dim {d}"),
            }
            .into());
        }
        if spec.stop > shape.dims[d] {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tensor_slice: stop={} > dim size={} at dim {}",
                    spec.stop, shape.dims[d], d
                ),
            }
            .into());
        }
    }

    let out_dims: Vec<usize> = specs.iter().map(|s| s.output_size()).collect();
    let out_shape = TensorShape::new(out_dims.clone());

    if output.len() < out_shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_slice: output len {} < required {}",
                output.len(),
                out_shape.numel()
            ),
        }
        .into());
    }

    if out_shape.numel() == 0 {
        return Ok(out_shape);
    }

    // Compute source strides (row-major)
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape.dims[d + 1];
    }

    // Iterate over all output elements via multi-index
    let total = out_shape.numel();
    let mut out_idx_vec = vec![0usize; ndim];

    for (flat_out, out_elem) in output.iter_mut().enumerate().take(total) {
        // Compute multi-index in output
        let mut rem = flat_out;
        for d in 0..ndim {
            let dim_size = out_dims[d];
            if dim_size > 0 {
                out_idx_vec[d] = rem / out_shape.inner_size_from(d + 1);
                rem %= out_shape.inner_size_from(d + 1);
            }
        }

        // Map to source multi-index
        let mut src_flat = 0usize;
        for d in 0..ndim {
            let src_idx = specs[d].start + out_idx_vec[d] * specs[d].step;
            src_flat += src_idx * strides[d];
        }

        *out_elem = input[src_flat];
    }

    Ok(out_shape)
}

// ── PaddedConcat ─────────────────────────────────────────────────

/// Concatenate tensors along `axis`, padding shorter tensors in the
/// non-concat dimensions with `pad_value` so all shapes match.
///
/// Tensors must have the same number of dimensions. For each non-concat
/// dimension, the output size is the *maximum* across all inputs.
pub fn padded_concat_ref(
    inputs: &[&[f32]],
    shapes: &[&TensorShape],
    axis: usize,
    pad_value: f32,
    output: &mut [f32],
) -> Result<TensorShape> {
    if inputs.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "padded_concat: inputs must not be empty".into(),
        }
        .into());
    }
    if inputs.len() != shapes.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "padded_concat: inputs.len()={} != shapes.len()={}",
                inputs.len(),
                shapes.len()
            ),
        }
        .into());
    }

    let ndim = shapes[0].ndim();
    if ndim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "padded_concat: cannot concat scalar tensors".into(),
        }
        .into());
    }
    if axis >= ndim {
        return Err(KernelError::InvalidArguments {
            reason: format!("padded_concat: axis {} >= ndim {}", axis, ndim),
        }
        .into());
    }

    for (i, shape) in shapes.iter().enumerate() {
        if shape.ndim() != ndim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "padded_concat: tensor {} ndim {} != expected {}",
                    i,
                    shape.ndim(),
                    ndim
                ),
            }
            .into());
        }
        if inputs[i].len() != shape.numel() {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "padded_concat: tensor {} data len {} != numel {}",
                    i,
                    inputs[i].len(),
                    shape.numel()
                ),
            }
            .into());
        }
    }

    // Output shape: max over each non-concat dim, sum over concat dim
    let mut out_dims = vec![0usize; ndim];
    for (d, dim_val) in out_dims.iter_mut().enumerate() {
        if d == axis {
            *dim_val = shapes.iter().map(|s| s.dims[d]).sum();
        } else {
            *dim_val = shapes.iter().map(|s| s.dims[d]).max().unwrap();
        }
    }
    let out_shape = TensorShape::new(out_dims.clone());

    if output.len() < out_shape.numel() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "padded_concat: output len {} < required {}",
                output.len(),
                out_shape.numel()
            ),
        }
        .into());
    }

    // Fill with pad_value
    output[..out_shape.numel()].fill(pad_value);

    // Compute output strides (row-major)
    let mut out_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_dims[d + 1];
    }

    // Copy each input tensor into the padded output
    let mut axis_offset = 0usize;
    for (tensor_data, shape) in inputs.iter().zip(shapes.iter()) {
        // Compute source strides
        let mut src_strides = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            src_strides[d] = src_strides[d + 1] * shape.dims[d + 1];
        }

        // Iterate over all source elements
        let src_numel = shape.numel();
        let mut src_multi = vec![0usize; ndim];
        for flat in 0..src_numel {
            // Decompose flat index into multi-index
            let mut rem = flat;
            for d in 0..ndim {
                src_multi[d] = rem / src_strides[d];
                rem %= src_strides[d];
            }

            // Map to output index
            let mut dst_flat = 0usize;
            for d in 0..ndim {
                let idx = if d == axis { src_multi[d] + axis_offset } else { src_multi[d] };
                dst_flat += idx * out_strides[d];
            }
            output[dst_flat] = tensor_data[flat];
        }
        axis_offset += shape.dims[axis];
    }

    Ok(out_shape)
}

// ── ConcatStats ──────────────────────────────────────────────────

/// Performance statistics for concatenation operations.
#[derive(Debug, Clone)]
pub struct ConcatStats {
    /// Number of elements processed.
    pub elements: usize,
    /// Number of input tensors.
    pub num_inputs: usize,
    /// Effective copy bandwidth in elements per second (if timed externally).
    pub bandwidth_elem_per_sec: Option<f64>,
    /// Wall-clock time in seconds (if timed externally).
    pub elapsed_secs: Option<f64>,
}

impl ConcatStats {
    /// Create stats for a completed operation.
    pub fn new(elements: usize, num_inputs: usize) -> Self {
        Self { elements, num_inputs, bandwidth_elem_per_sec: None, elapsed_secs: None }
    }

    /// Record timing after the operation.
    #[must_use]
    pub fn with_timing(mut self, elapsed_secs: f64) -> Self {
        self.elapsed_secs = Some(elapsed_secs);
        if elapsed_secs > 0.0 {
            self.bandwidth_elem_per_sec = Some(self.elements as f64 / elapsed_secs);
        }
        self
    }

    /// Throughput in MB/s assuming 4 bytes per element (f32).
    pub fn throughput_mb_per_sec(&self) -> Option<f64> {
        self.bandwidth_elem_per_sec.map(|bw| bw * 4.0 / (1024.0 * 1024.0))
    }
}

impl std::fmt::Display for ConcatStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConcatStats {{ elements: {}, inputs: {}", self.elements, self.num_inputs)?;
        if let Some(t) = self.elapsed_secs {
            write!(f, ", time: {t:.6}s")?;
        }
        if let Some(tp) = self.throughput_mb_per_sec() {
            write!(f, ", throughput: {tp:.1} MB/s")?;
        }
        write!(f, " }}")
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn validate_axis(shape: &TensorShape, axis: usize, fn_name: &str) -> Result<()> {
    if shape.ndim() == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("{fn_name}: cannot operate on scalar tensor"),
        }
        .into());
    }
    if axis >= shape.ndim() {
        return Err(KernelError::InvalidArguments {
            reason: format!("{}: axis {} >= ndim {}", fn_name, axis, shape.ndim()),
        }
        .into());
    }
    Ok(())
}

impl TensorShape {
    /// Product of dims from index `from` onward (used for flat-index decomposition).
    fn inner_size_from(&self, from: usize) -> usize {
        if from >= self.dims.len() { 1 } else { self.dims[from..].iter().product() }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────

    fn shape(dims: &[usize]) -> TensorShape {
        TensorShape::new(dims.to_vec())
    }

    fn iota(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn iota_offset(n: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| i as f32 + offset).collect()
    }

    // ── kernel source ───────────────────────────────────────

    #[test]
    fn kernel_source_not_empty() {
        assert!(!TENSOR_CONCAT_CL.is_empty());
    }

    #[test]
    fn kernel_source_contains_concat_kernels() {
        assert!(TENSOR_CONCAT_CL.contains("concat_axis0"));
        assert!(TENSOR_CONCAT_CL.contains("concat_general"));
    }

    #[test]
    fn kernel_source_contains_split_kernels() {
        assert!(TENSOR_CONCAT_CL.contains("split_axis0"));
        assert!(TENSOR_CONCAT_CL.contains("split_general"));
    }

    #[test]
    fn kernel_source_contains_padded_copy() {
        assert!(TENSOR_CONCAT_CL.contains("padded_copy"));
    }

    #[test]
    fn kernel_source_has_kernel_keyword() {
        assert!(TENSOR_CONCAT_CL.contains("__kernel"));
    }

    // ── TensorShape ─────────────────────────────────────────

    #[test]
    fn shape_numel() {
        assert_eq!(shape(&[2, 3, 4]).numel(), 24);
        assert_eq!(shape(&[1]).numel(), 1);
        assert_eq!(shape(&[0, 5]).numel(), 0);
    }

    #[test]
    fn shape_ndim() {
        assert_eq!(shape(&[2, 3]).ndim(), 2);
        assert_eq!(shape(&[5]).ndim(), 1);
    }

    #[test]
    fn shape_display() {
        let s = shape(&[2, 3, 4]);
        assert_eq!(format!("{s}"), "[2, 3, 4]");
    }

    // ── Concat axis 0 ───────────────────────────────────────

    #[test]
    fn concat_axis0_two_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let sa = shape(&[3]);
        let sb = shape(&[2]);
        let mut out = [0.0; 5];
        let result = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 0, &mut out).unwrap();
        assert_eq!(result.dims, vec![5]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_axis0_2d() {
        // [2,3] + [1,3] -> [3,3]
        let a = iota(6); // 0..6
        let b = iota_offset(3, 100.0); // 100,101,102
        let sa = shape(&[2, 3]);
        let sb = shape(&[1, 3]);
        let mut out = [0.0; 9];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![3, 3]);
        assert_eq!(out.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0]);
    }

    #[test]
    fn concat_axis0_3d() {
        // [1,2,3] + [2,2,3] -> [3,2,3]
        let a = iota(6);
        let b = iota_offset(12, 10.0);
        let sa = shape(&[1, 2, 3]);
        let sb = shape(&[2, 2, 3]);
        let mut out = [0.0; 18];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![3, 2, 3]);
        assert_eq!(&out[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(out[6], 10.0);
    }

    // ── Concat axis 1 ───────────────────────────────────────

    #[test]
    fn concat_axis1_2d() {
        // [2,2] + [2,3] -> [2,5]
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // [[10,20,30],[40,50,60]]
        let sa = shape(&[2, 2]);
        let sb = shape(&[2, 3]);
        let mut out = [0.0; 10];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 1, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 5]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0]);
    }

    // ── Concat axis 2 ───────────────────────────────────────

    #[test]
    fn concat_axis2_3d() {
        // [2,2,1] + [2,2,2] -> [2,2,3]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let sa = shape(&[2, 2, 1]);
        let sb = shape(&[2, 2, 2]);
        let mut out = [0.0; 12];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 2, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 2, 3]);
        // Row 0,0: [1, 10, 20]
        assert_eq!(&out[0..3], &[1.0, 10.0, 20.0]);
        // Row 0,1: [2, 30, 40]
        assert_eq!(&out[3..6], &[2.0, 30.0, 40.0]);
        // Row 1,0: [3, 50, 60]
        assert_eq!(&out[6..9], &[3.0, 50.0, 60.0]);
        // Row 1,1: [4, 70, 80]
        assert_eq!(&out[9..12], &[4.0, 70.0, 80.0]);
    }

    // ── Concat multiple tensors ─────────────────────────────

    #[test]
    fn concat_three_tensors_axis0() {
        let a = [1.0];
        let b = [2.0];
        let c = [3.0];
        let s = shape(&[1]);
        let mut out = [0.0; 3];
        let r = tensor_concat_ref(&[&a, &b, &c], &[&s, &s, &s], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![3]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_four_tensors_axis1() {
        // 4 × [1,1] concat along axis 1 -> [1,4]
        let data: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32]).collect();
        let s = shape(&[1, 1]);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let shapes: Vec<&TensorShape> = [&s; 4].to_vec();
        let mut out = [0.0; 4];
        let r = tensor_concat_ref(&refs, &shapes, 1, &mut out).unwrap();
        assert_eq!(r.dims, vec![1, 4]);
        assert_eq!(out.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    // ── Concat single tensor ────────────────────────────────

    #[test]
    fn concat_single_tensor() {
        let a = vec![1.0, 2.0];
        let s = shape(&[2]);
        let mut out = [0.0; 2];
        let r = tensor_concat_ref(&[&a], &[&s], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0]);
    }

    // ── Concat errors ───────────────────────────────────────

    #[test]
    fn concat_empty_inputs_error() {
        let mut out = [0.0; 1];
        let r = tensor_concat_ref(&[], &[], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_mismatched_dims_error() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let sa = shape(&[2]);
        let sb = shape(&[2, 2]);
        let mut out = [0.0; 6];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_mismatched_non_axis_dim_error() {
        // [2,3] + [2,4] concat on axis 0 should fail (dim 1 differs)
        let a = iota(6);
        let b = iota(8);
        let sa = shape(&[2, 3]);
        let sb = shape(&[2, 4]);
        let mut out = [0.0; 20];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_axis_out_of_range_error() {
        let a = [1.0];
        let s = shape(&[1]);
        let mut out = [0.0; 1];
        let r = tensor_concat_ref(&[&a], &[&s], 5, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_output_too_small_error() {
        let a = vec![1.0, 2.0];
        let s = shape(&[2]);
        let mut out = [0.0; 1]; // too small
        let r = tensor_concat_ref(&[&a, &a], &[&s, &s], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_data_shape_mismatch_error() {
        let a = vec![1.0, 2.0, 3.0]; // len 3 but shape says [2]
        let s = shape(&[2]);
        let mut out = [0.0; 4];
        let r = tensor_concat_ref(&[&a], &[&s], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_scalar_tensors_error() {
        let a = [1.0];
        let s = TensorShape::new(vec![]);
        let mut out = [0.0; 1];
        let r = tensor_concat_ref(&[&a], &[&s], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn concat_inputs_shapes_len_mismatch_error() {
        let a = [1.0];
        let s = shape(&[1]);
        let mut out = [0.0; 2];
        let r = tensor_concat_ref(&[&a, &a], &[&s], 0, &mut out);
        assert!(r.is_err());
    }

    // ── Split ───────────────────────────────────────────────

    #[test]
    fn split_axis0_into_two() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let s = shape(&[4]);
        let mut outs = Vec::new();
        let shapes = tensor_split_ref(&data, &s, 0, 2, &mut outs).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].dims, vec![2]);
        assert_eq!(outs[0], vec![1.0, 2.0]);
        assert_eq!(outs[1], vec![3.0, 4.0]);
    }

    #[test]
    fn split_axis1_2d() {
        // [2,4] split along axis 1 into 2 => 2×[2,2]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let s = shape(&[2, 4]);
        let mut outs = Vec::new();
        let shapes = tensor_split_ref(&data, &s, 1, 2, &mut outs).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].dims, vec![2, 2]);
        assert_eq!(outs[0], vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(outs[1], vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn split_not_divisible_error() {
        let data = iota(5);
        let s = shape(&[5]);
        let mut outs = Vec::new();
        let r = tensor_split_ref(&data, &s, 0, 2, &mut outs);
        assert!(r.is_err());
    }

    #[test]
    fn split_n_zero_error() {
        let data = iota(4);
        let s = shape(&[4]);
        let mut outs = Vec::new();
        let r = tensor_split_ref(&data, &s, 0, 0, &mut outs);
        assert!(r.is_err());
    }

    #[test]
    fn split_axis_out_of_range_error() {
        let data = iota(4);
        let s = shape(&[4]);
        let mut outs = Vec::new();
        let r = tensor_split_ref(&data, &s, 5, 2, &mut outs);
        assert!(r.is_err());
    }

    #[test]
    fn split_data_mismatch_error() {
        let data = iota(5); // len 5 but shape [4]
        let s = shape(&[4]);
        let mut outs = Vec::new();
        let r = tensor_split_ref(&data, &s, 0, 2, &mut outs);
        assert!(r.is_err());
    }

    // ── Split + Concat roundtrip ────────────────────────────

    #[test]
    fn split_concat_roundtrip_axis0() {
        let data = iota(12);
        let s = shape(&[12]);
        let mut parts = Vec::new();
        let part_shapes = tensor_split_ref(&data, &s, 0, 3, &mut parts).unwrap();
        let refs: Vec<&[f32]> = parts.iter().map(|v| v.as_slice()).collect();
        let shape_refs: Vec<&TensorShape> = part_shapes.iter().collect();
        let mut recon = [0.0; 12];
        tensor_concat_ref(&refs, &shape_refs, 0, &mut recon).unwrap();
        assert_eq!(data, recon);
    }

    #[test]
    fn split_concat_roundtrip_axis1() {
        let data = iota(24); // [4,6]
        let s = shape(&[4, 6]);
        let mut parts = Vec::new();
        let part_shapes = tensor_split_ref(&data, &s, 1, 3, &mut parts).unwrap();
        let refs: Vec<&[f32]> = parts.iter().map(|v| v.as_slice()).collect();
        let shape_refs: Vec<&TensorShape> = part_shapes.iter().collect();
        let mut recon = [0.0; 24];
        tensor_concat_ref(&refs, &shape_refs, 1, &mut recon).unwrap();
        assert_eq!(data, recon);
    }

    #[test]
    fn split_concat_roundtrip_3d_axis2() {
        let data = iota(24); // [2,3,4]
        let s = shape(&[2, 3, 4]);
        let mut parts = Vec::new();
        let part_shapes = tensor_split_ref(&data, &s, 2, 2, &mut parts).unwrap();
        let refs: Vec<&[f32]> = parts.iter().map(|v| v.as_slice()).collect();
        let shape_refs: Vec<&TensorShape> = part_shapes.iter().collect();
        let mut recon = [0.0; 24];
        tensor_concat_ref(&refs, &shape_refs, 2, &mut recon).unwrap();
        assert_eq!(data, recon);
    }

    // ── Chunk ───────────────────────────────────────────────

    #[test]
    fn chunk_even_split() {
        let data = iota(6);
        let s = shape(&[6]);
        let mut outs = Vec::new();
        let shapes = tensor_chunk_ref(&data, &s, 0, 2, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(outs[0], vec![0.0, 1.0]);
        assert_eq!(outs[1], vec![2.0, 3.0]);
        assert_eq!(outs[2], vec![4.0, 5.0]);
    }

    #[test]
    fn chunk_with_remainder() {
        let data = iota(7);
        let s = shape(&[7]);
        let mut outs = Vec::new();
        let shapes = tensor_chunk_ref(&data, &s, 0, 3, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(shapes[0].dims, vec![3]);
        assert_eq!(shapes[2].dims, vec![1]);
        assert_eq!(outs[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(outs[1], vec![3.0, 4.0, 5.0]);
        assert_eq!(outs[2], vec![6.0]);
    }

    #[test]
    fn chunk_axis1_2d() {
        // [2,5] chunk axis 1 size 2 => [2,2], [2,2], [2,1]
        let data = iota(10);
        let s = shape(&[2, 5]);
        let mut outs = Vec::new();
        let shapes = tensor_chunk_ref(&data, &s, 1, 2, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(shapes[0].dims, vec![2, 2]);
        assert_eq!(shapes[2].dims, vec![2, 1]);
        // First chunk: cols 0,1 from each row
        assert_eq!(outs[0], vec![0.0, 1.0, 5.0, 6.0]);
        // Last chunk: col 4 from each row
        assert_eq!(outs[2], vec![4.0, 9.0]);
    }

    #[test]
    fn chunk_size_zero_error() {
        let data = iota(4);
        let s = shape(&[4]);
        let mut outs = Vec::new();
        let r = tensor_chunk_ref(&data, &s, 0, 0, &mut outs);
        assert!(r.is_err());
    }

    #[test]
    fn chunk_single_element() {
        let data = [42.0];
        let s = shape(&[1]);
        let mut outs = Vec::new();
        let shapes = tensor_chunk_ref(&data, &s, 0, 1, &mut outs).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_eq!(outs[0], vec![42.0]);
    }

    // ── Stack ───────────────────────────────────────────────

    #[test]
    fn stack_axis0_vectors() {
        // stack [3],[3] along axis 0 -> [2,3]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let s = shape(&[3]);
        let mut out = [0.0; 6];
        let r = tensor_stack_ref(&[&a, &b], &[&s, &s], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 3]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_axis1_vectors() {
        // stack [3],[3] along axis 1 -> [3,2]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let s = shape(&[3]);
        let mut out = [0.0; 6];
        let r = tensor_stack_ref(&[&a, &b], &[&s, &s], 1, &mut out).unwrap();
        assert_eq!(r.dims, vec![3, 2]);
        assert_eq!(out.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn stack_axis0_2d() {
        // stack [2,3],[2,3] along axis 0 -> [2,2,3]
        let a = iota(6);
        let b = iota_offset(6, 10.0);
        let s = shape(&[2, 3]);
        let mut out = [0.0; 12];
        let r = tensor_stack_ref(&[&a, &b], &[&s, &s], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 2, 3]);
        assert_eq!(&out[..6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&out[6..12], &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn stack_three_tensors() {
        let a = [1.0];
        let b = [2.0];
        let c = [3.0];
        let s = shape(&[1]);
        let mut out = [0.0; 3];
        let r = tensor_stack_ref(&[&a, &b, &c], &[&s, &s, &s], 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![3, 1]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn stack_empty_inputs_error() {
        let mut out = [0.0; 1];
        let r: Result<TensorShape> = tensor_stack_ref(&[], &[], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn stack_mismatched_shapes_error() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let sa = shape(&[2]);
        let sb = shape(&[3]);
        let mut out = [0.0; 5];
        let r = tensor_stack_ref(&[&a, &b], &[&sa, &sb], 0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn stack_axis_out_of_range_error() {
        let a = [1.0];
        let s = shape(&[1]);
        let mut out = [0.0; 1];
        // axis=2 for 1-d tensor (max allowed is 1)
        let r = tensor_stack_ref(&[&a], &[&s], 2, &mut out);
        assert!(r.is_err());
    }

    // ── Unstack ─────────────────────────────────────────────

    #[test]
    fn unstack_axis0_2d() {
        // [2,3] -> two [3] tensors
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = shape(&[2, 3]);
        let mut outs = Vec::new();
        let shapes = tensor_unstack_ref(&data, &s, 0, &mut outs).unwrap();
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].dims, vec![3]);
        assert_eq!(outs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(outs[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn unstack_axis1_2d() {
        // [2,3] -> three [2] tensors
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = shape(&[2, 3]);
        let mut outs = Vec::new();
        let shapes = tensor_unstack_ref(&data, &s, 1, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(shapes[0].dims, vec![2]);
        assert_eq!(outs[0], vec![1.0, 4.0]);
        assert_eq!(outs[1], vec![2.0, 5.0]);
        assert_eq!(outs[2], vec![3.0, 6.0]);
    }

    #[test]
    fn unstack_1d() {
        // [3] -> three scalars (represented as [1])
        let data = vec![10.0, 20.0, 30.0];
        let s = shape(&[3]);
        let mut outs = Vec::new();
        let shapes = tensor_unstack_ref(&data, &s, 0, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(shapes[0].dims, vec![1]);
        assert_eq!(outs[0], vec![10.0]);
        assert_eq!(outs[1], vec![20.0]);
        assert_eq!(outs[2], vec![30.0]);
    }

    #[test]
    fn unstack_scalar_error() {
        let data = [1.0];
        let s = TensorShape::new(vec![]);
        let mut outs = Vec::new();
        let r = tensor_unstack_ref(&data, &s, 0, &mut outs);
        assert!(r.is_err());
    }

    // ── Stack + Unstack roundtrip ───────────────────────────

    #[test]
    fn stack_unstack_roundtrip_axis0() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let s = shape(&[3]);
        let mut stacked = [0.0; 6];
        let stacked_shape = tensor_stack_ref(&[&a, &b], &[&s, &s], 0, &mut stacked).unwrap();
        let mut unstacked = Vec::new();
        tensor_unstack_ref(&stacked, &stacked_shape, 0, &mut unstacked).unwrap();
        assert_eq!(unstacked[0], a);
        assert_eq!(unstacked[1], b);
    }

    #[test]
    fn stack_unstack_roundtrip_axis1() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let s = shape(&[2]);
        let mut stacked = [0.0; 4];
        let stacked_shape = tensor_stack_ref(&[&a, &b], &[&s, &s], 1, &mut stacked).unwrap();
        let mut unstacked = Vec::new();
        tensor_unstack_ref(&stacked, &stacked_shape, 1, &mut unstacked).unwrap();
        assert_eq!(unstacked[0], a);
        assert_eq!(unstacked[1], b);
    }

    // ── Slice ───────────────────────────────────────────────

    #[test]
    fn slice_full() {
        let data = iota(6);
        let s = shape(&[2, 3]);
        let specs = [SliceSpec::full(2), SliceSpec::full(3)];
        let mut out = [0.0; 6];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 3]);
        assert_eq!(out.to_vec(), data);
    }

    #[test]
    fn slice_subrange() {
        // [6] slice [1:4:1] -> [3]
        let data = iota(6);
        let s = shape(&[6]);
        let specs = [SliceSpec::new(1, 4, 1)];
        let mut out = [0.0; 3];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![3]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn slice_with_step() {
        // [6] slice [0:6:2] -> [3]
        let data = iota(6);
        let s = shape(&[6]);
        let specs = [SliceSpec::new(0, 6, 2)];
        let mut out = [0.0; 3];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![3]);
        assert_eq!(out.to_vec(), vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn slice_2d_submatrix() {
        // [3,4] -> [2,2] via rows [0:2], cols [1:3]
        let data = iota(12);
        let s = shape(&[3, 4]);
        let specs = [SliceSpec::new(0, 2, 1), SliceSpec::new(1, 3, 1)];
        let mut out = [0.0; 4];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 2]);
        // row 0: [0,1,2,3] -> cols 1,2 -> [1,2]
        // row 1: [4,5,6,7] -> cols 1,2 -> [5,6]
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_step_2_on_2d() {
        // [4,4] slice rows [0:4:2], cols [0:4:2] -> [2,2]
        let data = iota(16);
        let s = shape(&[4, 4]);
        let specs = [SliceSpec::new(0, 4, 2), SliceSpec::new(0, 4, 2)];
        let mut out = [0.0; 4];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 2]);
        // (0,0)=0, (0,2)=2, (2,0)=8, (2,2)=10
        assert_eq!(out.to_vec(), vec![0.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn slice_empty_range() {
        let data = iota(4);
        let s = shape(&[4]);
        let specs = [SliceSpec::new(2, 2, 1)]; // empty range
        let mut out = vec![];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![0]);
        assert_eq!(r.numel(), 0);
    }

    #[test]
    fn slice_step_zero_error() {
        let data = iota(4);
        let s = shape(&[4]);
        let specs = [SliceSpec::new(0, 4, 0)];
        let mut out = [0.0; 4];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn slice_stop_exceeds_dim_error() {
        let data = iota(4);
        let s = shape(&[4]);
        let specs = [SliceSpec::new(0, 5, 1)];
        let mut out = [0.0; 5];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn slice_wrong_specs_len_error() {
        let data = iota(6);
        let s = shape(&[2, 3]);
        let specs = [SliceSpec::full(2)]; // only 1 spec for 2-d tensor
        let mut out = [0.0; 6];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out);
        assert!(r.is_err());
    }

    // ── PaddedConcat ────────────────────────────────────────

    #[test]
    fn padded_concat_same_shapes() {
        // Same shapes = same as regular concat
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sa = shape(&[1, 3]);
        let sb = shape(&[1, 3]);
        let mut out = [0.0; 6];
        let r = padded_concat_ref(&[&a, &b], &[&sa, &sb], 0, 0.0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 3]);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn padded_concat_different_non_axis_dims() {
        // [1,2] + [1,3] along axis 0 -> [2, max(2,3)=3] with padding
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let sa = shape(&[1, 2]);
        let sb = shape(&[1, 3]);
        let mut out = [-1.0; 6];
        let r = padded_concat_ref(&[&a, &b], &[&sa, &sb], 0, 0.0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 3]);
        // row 0: [1, 2, pad=0]
        // row 1: [3, 4, 5]
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 0.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn padded_concat_custom_pad_value() {
        let a = [1.0];
        let b = vec![2.0, 3.0];
        let sa = shape(&[1, 1]);
        let sb = shape(&[1, 2]);
        let mut out = [0.0; 4];
        let r = padded_concat_ref(&[&a, &b], &[&sa, &sb], 0, -999.0, &mut out).unwrap();
        assert_eq!(r.dims, vec![2, 2]);
        assert_eq!(out.to_vec(), vec![1.0, -999.0, 2.0, 3.0]);
    }

    #[test]
    fn padded_concat_empty_inputs_error() {
        let mut out = [0.0; 1];
        let r = padded_concat_ref(&[], &[], 0, 0.0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn padded_concat_axis_out_of_range_error() {
        let a = [1.0];
        let s = shape(&[1]);
        let mut out = [0.0; 1];
        let r = padded_concat_ref(&[&a], &[&s], 5, 0.0, &mut out);
        assert!(r.is_err());
    }

    // ── ConcatStats ─────────────────────────────────────────

    #[test]
    fn stats_new() {
        let stats = ConcatStats::new(1000, 3);
        assert_eq!(stats.elements, 1000);
        assert_eq!(stats.num_inputs, 3);
        assert!(stats.bandwidth_elem_per_sec.is_none());
    }

    #[test]
    fn stats_with_timing() {
        let stats = ConcatStats::new(1_000_000, 2).with_timing(0.5);
        assert!((stats.bandwidth_elem_per_sec.unwrap() - 2_000_000.0).abs() < 1.0);
        assert!((stats.elapsed_secs.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stats_throughput_mb() {
        let stats = ConcatStats::new(1_000_000, 1).with_timing(1.0);
        let tp = stats.throughput_mb_per_sec().unwrap();
        // 1M * 4 bytes / 1MB = ~3.81 MB/s
        assert!(tp > 3.0 && tp < 4.0);
    }

    #[test]
    fn stats_display() {
        let stats = ConcatStats::new(100, 2);
        let s = format!("{stats}");
        assert!(s.contains("100"));
        assert!(s.contains("2"));
    }

    #[test]
    fn stats_zero_time() {
        let stats = ConcatStats::new(100, 1).with_timing(0.0);
        assert!(stats.bandwidth_elem_per_sec.is_none());
    }

    // ── SliceSpec ───────────────────────────────────────────

    #[test]
    fn slice_spec_output_size() {
        assert_eq!(SliceSpec::new(0, 10, 1).output_size(), 10);
        assert_eq!(SliceSpec::new(0, 10, 3).output_size(), 4);
        assert_eq!(SliceSpec::new(2, 8, 2).output_size(), 3);
        assert_eq!(SliceSpec::new(5, 5, 1).output_size(), 0);
        assert_eq!(SliceSpec::new(0, 0, 1).output_size(), 0);
    }

    #[test]
    fn slice_spec_full() {
        let s = SliceSpec::full(10);
        assert_eq!(s.start, 0);
        assert_eq!(s.stop, 10);
        assert_eq!(s.step, 1);
        assert_eq!(s.output_size(), 10);
    }

    // ── Property tests ──────────────────────────────────────

    #[test]
    fn concat_output_size_equals_sum_of_inputs_axis0() {
        // For axis 0: output numel = sum of input numels
        for n_tensors in 1..=5 {
            let data: Vec<Vec<f32>> = (0..n_tensors).map(|i| iota((i + 1) * 3)).collect();
            let shapes: Vec<TensorShape> = (0..n_tensors).map(|i| shape(&[(i + 1) * 3])).collect();
            let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
            let shape_refs: Vec<&TensorShape> = shapes.iter().collect();
            let total: usize = shapes.iter().map(|s| s.numel()).sum();
            let mut out = vec![0.0; total];
            let r = tensor_concat_ref(&refs, &shape_refs, 0, &mut out).unwrap();
            assert_eq!(r.numel(), total);
        }
    }

    #[test]
    fn concat_output_axis_size_equals_sum_axis1() {
        // axis 1: output dim[1] = sum of input dim[1]
        let a = iota(6); // [2,3]
        let b = iota(4); // [2,2]
        let sa = shape(&[2, 3]);
        let sb = shape(&[2, 2]);
        let mut out = [0.0; 10];
        let r = tensor_concat_ref(&[&a, &b], &[&sa, &sb], 1, &mut out).unwrap();
        assert_eq!(r.dims[1], 3 + 2);
        assert_eq!(r.dims[0], 2);
    }

    #[test]
    fn split_produces_correct_count() {
        for n in 1..=6 {
            let data = iota(12);
            let s = shape(&[12]);
            if 12 % n != 0 {
                continue;
            }
            let mut outs = Vec::new();
            let shapes = tensor_split_ref(&data, &s, 0, n, &mut outs).unwrap();
            assert_eq!(shapes.len(), n);
            assert_eq!(outs.len(), n);
            for part in &outs {
                assert_eq!(part.len(), 12 / n);
            }
        }
    }

    #[test]
    fn chunk_total_elements_preserved() {
        for chunk_size in 1..=5 {
            let data = iota(10);
            let s = shape(&[10]);
            let mut outs = Vec::new();
            tensor_chunk_ref(&data, &s, 0, chunk_size, &mut outs).unwrap();
            let total: usize = outs.iter().map(|v| v.len()).sum();
            assert_eq!(total, 10);
        }
    }

    #[test]
    fn stack_adds_dimension() {
        let a = iota(4); // [4]
        let s = shape(&[4]);
        let mut out = [0.0; 8];
        let r = tensor_stack_ref(&[&a, &a], &[&s, &s], 0, &mut out).unwrap();
        assert_eq!(r.ndim(), 2);
        assert_eq!(r.dims[0], 2);
        assert_eq!(r.dims[1], 4);
    }

    #[test]
    fn unstack_removes_dimension() {
        let data = iota(6); // [2,3]
        let s = shape(&[2, 3]);
        let mut outs = Vec::new();
        let shapes = tensor_unstack_ref(&data, &s, 0, &mut outs).unwrap();
        assert_eq!(shapes[0].ndim(), 1);
        assert_eq!(shapes[0].dims, vec![3]);
    }

    // ── Edge cases ──────────────────────────────────────────

    #[test]
    fn concat_large_number_of_tensors() {
        let n = 100;
        let data: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32]).collect();
        let s = shape(&[1]);
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let shapes: Vec<&TensorShape> = vec![&s; n];
        let mut out = vec![0.0; n];
        let r = tensor_concat_ref(&refs, &shapes, 0, &mut out).unwrap();
        assert_eq!(r.dims, vec![n]);
        for i in 0..n {
            assert_eq!(out[i], i as f32);
        }
    }

    #[test]
    fn split_into_one() {
        let data = iota(6);
        let s = shape(&[6]);
        let mut outs = Vec::new();
        let shapes = tensor_split_ref(&data, &s, 0, 1, &mut outs).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_eq!(outs[0], data);
    }

    #[test]
    fn chunk_larger_than_dim() {
        let data = iota(3);
        let s = shape(&[3]);
        let mut outs = Vec::new();
        let shapes = tensor_chunk_ref(&data, &s, 0, 10, &mut outs).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_eq!(outs[0], data);
    }

    #[test]
    fn unstack_3d_axis1() {
        // [2,3,2] unstack axis 1 -> three [2,2]
        let data = iota(12);
        let s = shape(&[2, 3, 2]);
        let mut outs = Vec::new();
        let shapes = tensor_unstack_ref(&data, &s, 1, &mut outs).unwrap();
        assert_eq!(shapes.len(), 3);
        assert_eq!(shapes[0].dims, vec![2, 2]);
        // tensor 0: elements at axis1=0 -> [0,1, 6,7]
        assert_eq!(outs[0], vec![0.0, 1.0, 6.0, 7.0]);
    }

    #[test]
    fn padded_concat_three_different_sizes() {
        // [1,1] + [1,2] + [1,3] along axis 0 -> [3, 3] with padding
        let a = [1.0];
        let b = vec![2.0, 3.0];
        let c = vec![4.0, 5.0, 6.0];
        let sa = shape(&[1, 1]);
        let sb = shape(&[1, 2]);
        let sc = shape(&[1, 3]);
        let mut out = [0.0; 9];
        let r = padded_concat_ref(&[&a, &b, &c], &[&sa, &sb, &sc], 0, 0.0, &mut out).unwrap();
        assert_eq!(r.dims, vec![3, 3]);
        // row 0: [1, 0, 0]
        assert_eq!(&out[0..3], &[1.0, 0.0, 0.0]);
        // row 1: [2, 3, 0]
        assert_eq!(&out[3..6], &[2.0, 3.0, 0.0]);
        // row 2: [4, 5, 6]
        assert_eq!(&out[6..9], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn padded_concat_ndim_mismatch_error() {
        let a = [1.0];
        let b = vec![2.0, 3.0];
        let sa = shape(&[1]);
        let sb = shape(&[1, 2]);
        let mut out = [0.0; 4];
        let r = padded_concat_ref(&[&a, &b], &[&sa, &sb], 0, 0.0, &mut out);
        assert!(r.is_err());
    }

    #[test]
    fn slice_3d_tensor() {
        // [2,3,4] slice [0:1, 1:3, 0:4:2] -> [1,2,2]
        let data = iota(24);
        let s = shape(&[2, 3, 4]);
        let specs = [SliceSpec::new(0, 1, 1), SliceSpec::new(1, 3, 1), SliceSpec::new(0, 4, 2)];
        let mut out = [0.0; 4];
        let r = tensor_slice_ref(&data, &s, &specs, &mut out).unwrap();
        assert_eq!(r.dims, vec![1, 2, 2]);
        // row(0,1): [4,5,6,7] -> cols 0,2 -> [4,6]
        // row(0,2): [8,9,10,11] -> cols 0,2 -> [8,10]
        assert_eq!(out.to_vec(), vec![4.0, 6.0, 8.0, 10.0]);
    }
}
