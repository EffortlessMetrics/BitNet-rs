//! OpenCL-accelerated elementwise operations for Intel Arc A770 (Xe-HPG).
//!
//! Provides GPU-optimized elementwise operations (add, mul, scale, residual,
//! broadcast) with vectorized `float4` kernels. CPU reference implementations
//! are included for correctness testing and non-GPU environments.
//!
//! # Components
//!
//! - **`ElemOp`** — operation discriminant (add, sub, mul, div, etc.)
//! - **`BroadcastRule`** — NumPy-style shape broadcasting with validation
//! - **`ElemwiseKernel`** — dispatches elementwise operations on tensors
//! - **`FusedElemwise`** — chains 2–3 ops in a single kernel launch
//! - **`ScalarBroadcast`** — efficient scalar-to-tensor broadcast (no copy)
//! - **`InPlaceOps`** — in-place variants for memory efficiency
//! - **`ElemwiseStats`** — throughput and bandwidth measurements

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to elementwise operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ElemwiseError {
    /// Tensor shapes are incompatible for the requested operation.
    ShapeMismatch { a: Vec<usize>, b: Vec<usize> },
    /// A dimension is zero, which is not permitted.
    ZeroDimension,
    /// Division by zero detected.
    DivisionByZero,
    /// Clamp bounds are invalid (min > max).
    InvalidClampBounds { min: f32, max: f32 },
    /// Data length does not match the declared shape.
    DataShapeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ElemwiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { a, b } => {
                write!(f, "shape mismatch: {a:?} vs {b:?}")
            }
            Self::ZeroDimension => write!(f, "zero-sized dimension"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::InvalidClampBounds { min, max } => {
                write!(f, "invalid clamp bounds: min={min} > max={max}")
            }
            Self::DataShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "data length {actual} does not match shape \
                     (expected {expected} elements)"
                )
            }
        }
    }
}

impl std::error::Error for ElemwiseError {}

// ---------------------------------------------------------------------------
// ElemOp — operation discriminant
// ---------------------------------------------------------------------------

/// Elementwise operation to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElemOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Uniform scale: `x * scalar`.
    Scale,
    /// Residual connection: semantically identical to `Add` but named for
    /// clarity in transformer pipelines.
    Residual,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
    /// Absolute value (unary).
    Abs,
    /// Negation (unary).
    Neg,
    /// Clamp to `[min, max]` (unary with parameters).
    Clamp,
}

impl ElemOp {
    /// Returns `true` for operations that take only one input tensor.
    pub fn is_unary(self) -> bool {
        matches!(self, Self::Abs | Self::Neg | Self::Clamp)
    }

    /// Apply the operation element-wise on two scalars.
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Add | Self::Residual => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            Self::Scale => a * b,
            Self::Max => a.max(b),
            Self::Min => a.min(b),
            Self::Abs => a.abs(),
            Self::Neg => -a,
            Self::Clamp => a, // handled separately with bounds
        }
    }
}

impl fmt::Display for ElemOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Scale => "scale",
            Self::Residual => "residual",
            Self::Max => "max",
            Self::Min => "min",
            Self::Abs => "abs",
            Self::Neg => "neg",
            Self::Clamp => "clamp",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// BroadcastRule — NumPy-style broadcasting
// ---------------------------------------------------------------------------

/// NumPy-style broadcasting rules with shape validation.
///
/// Two shapes are broadcast-compatible when, starting from the trailing
/// dimensions and working forward, for each pair of dimensions either:
/// - The dimensions are equal, **or**
/// - One of them is 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastRule {
    /// The output shape after broadcasting.
    pub output_shape: Vec<usize>,
    /// Per-dimension stride multiplier for tensor A (0 = broadcast, 1 = use).
    pub strides_a: Vec<usize>,
    /// Per-dimension stride multiplier for tensor B (0 = broadcast, 1 = use).
    pub strides_b: Vec<usize>,
}

impl BroadcastRule {
    /// Compute the broadcast rule for two shapes.
    ///
    /// Returns `Err` if shapes are not broadcast-compatible.
    pub fn new(shape_a: &[usize], shape_b: &[usize]) -> Result<Self, ElemwiseError> {
        let ndim = shape_a.len().max(shape_b.len());
        let mut output_shape = Vec::with_capacity(ndim);
        let mut strides_a = Vec::with_capacity(ndim);
        let mut strides_b = Vec::with_capacity(ndim);

        for i in 0..ndim {
            let da = if i < ndim - shape_a.len() { 1 } else { shape_a[i - (ndim - shape_a.len())] };
            let db = if i < ndim - shape_b.len() { 1 } else { shape_b[i - (ndim - shape_b.len())] };

            if da == db {
                output_shape.push(da);
                strides_a.push(1);
                strides_b.push(1);
            } else if da == 1 {
                output_shape.push(db);
                strides_a.push(0);
                strides_b.push(1);
            } else if db == 1 {
                output_shape.push(da);
                strides_a.push(1);
                strides_b.push(0);
            } else {
                return Err(ElemwiseError::ShapeMismatch {
                    a: shape_a.to_vec(),
                    b: shape_b.to_vec(),
                });
            }
        }

        Ok(Self { output_shape, strides_a, strides_b })
    }

    /// Total number of elements in the broadcast output.
    pub fn output_len(&self) -> usize {
        self.output_shape.iter().product()
    }

    /// Map a flat output index back to flat indices into A and B.
    pub fn map_indices(
        &self,
        out_idx: usize,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> (usize, usize) {
        let ndim = self.output_shape.len();
        let mut remaining = out_idx;
        let mut coords = vec![0usize; ndim];

        // Decompose flat index into multi-dimensional coordinates.
        for i in (0..ndim).rev() {
            coords[i] = remaining % self.output_shape[i];
            remaining /= self.output_shape[i];
        }

        let flat_a = Self::coords_to_flat(&coords, &self.strides_a, shape_a, ndim);
        let flat_b = Self::coords_to_flat(&coords, &self.strides_b, shape_b, ndim);
        (flat_a, flat_b)
    }

    fn coords_to_flat(
        coords: &[usize],
        broadcast_strides: &[usize],
        orig_shape: &[usize],
        ndim: usize,
    ) -> usize {
        let orig_ndim = orig_shape.len();
        let offset = ndim - orig_ndim;
        let mut flat = 0usize;
        let mut stride = 1usize;
        for i in (0..orig_ndim).rev() {
            let coord = if broadcast_strides[i + offset] == 0 { 0 } else { coords[i + offset] };
            flat += coord * stride;
            stride *= orig_shape[i];
        }
        flat
    }
}

// ---------------------------------------------------------------------------
// ElemwiseKernel — dispatch elementwise operations
// ---------------------------------------------------------------------------

/// Dispatches elementwise operations on flat f32 tensors.
///
/// All public methods use CPU reference implementations. The companion
/// OpenCL kernel source (see [`ELEMENTWISE_CL`]) targets GPU dispatch.
pub struct ElemwiseKernel;

impl ElemwiseKernel {
    /// Apply a binary elementwise operation on two tensors of equal length.
    pub fn apply_binary(
        op: ElemOp,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
    ) -> Result<(), ElemwiseError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(ElemwiseError::ShapeMismatch { a: vec![a.len()], b: vec![b.len()] });
        }
        if op == ElemOp::Div && b.contains(&0.0) {
            return Err(ElemwiseError::DivisionByZero);
        }
        for (i, out_v) in out.iter_mut().enumerate() {
            *out_v = op.apply(a[i], b[i]);
        }
        Ok(())
    }

    /// Apply a unary elementwise operation.
    pub fn apply_unary(op: ElemOp, a: &[f32], out: &mut [f32]) -> Result<(), ElemwiseError> {
        if a.len() != out.len() {
            return Err(ElemwiseError::DataShapeMismatch { expected: a.len(), actual: out.len() });
        }
        for (i, out_v) in out.iter_mut().enumerate() {
            *out_v = op.apply(a[i], 0.0);
        }
        Ok(())
    }

    /// Apply a binary operation with NumPy-style broadcasting.
    pub fn apply_broadcast(
        op: ElemOp,
        a: &[f32],
        shape_a: &[usize],
        b: &[f32],
        shape_b: &[usize],
        out: &mut [f32],
    ) -> Result<(), ElemwiseError> {
        let rule = BroadcastRule::new(shape_a, shape_b)?;
        let out_len = rule.output_len();
        if out.len() < out_len {
            return Err(ElemwiseError::DataShapeMismatch { expected: out_len, actual: out.len() });
        }
        if op == ElemOp::Div && b.contains(&0.0) {
            return Err(ElemwiseError::DivisionByZero);
        }
        for (idx, out_v) in out.iter_mut().enumerate().take(out_len) {
            let (ia, ib) = rule.map_indices(idx, shape_a, shape_b);
            *out_v = op.apply(a[ia], b[ib]);
        }
        Ok(())
    }

    /// Clamp each element of `a` to `[min_val, max_val]`.
    pub fn clamp(
        a: &[f32],
        min_val: f32,
        max_val: f32,
        out: &mut [f32],
    ) -> Result<(), ElemwiseError> {
        if min_val > max_val {
            return Err(ElemwiseError::InvalidClampBounds { min: min_val, max: max_val });
        }
        if a.len() != out.len() {
            return Err(ElemwiseError::DataShapeMismatch { expected: a.len(), actual: out.len() });
        }
        for i in 0..a.len() {
            out[i] = a[i].clamp(min_val, max_val);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FusedElemwise — chain 2–3 ops in a single kernel
// ---------------------------------------------------------------------------

/// Chains 2–3 elementwise operations in a single pass, avoiding intermediate
/// memory traffic. Typical use: `scale + add` = residual connection.
#[derive(Debug, Clone)]
pub struct FusedElemwise {
    /// Ordered list of operations to apply.
    ops: Vec<FusedStep>,
}

/// A single step in a fused elementwise pipeline.
#[derive(Debug, Clone)]
pub enum FusedStep {
    /// Binary op with a second tensor.
    Binary { op: ElemOp, operand: Vec<f32> },
    /// Scalar op (broadcast scalar to all elements).
    Scalar { op: ElemOp, value: f32 },
    /// Unary op (no extra operand).
    Unary { op: ElemOp },
}

impl FusedElemwise {
    /// Create a new fused pipeline.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append a binary step.
    pub fn then_binary(mut self, op: ElemOp, operand: Vec<f32>) -> Self {
        self.ops.push(FusedStep::Binary { op, operand });
        self
    }

    /// Append a scalar step.
    pub fn then_scalar(mut self, op: ElemOp, value: f32) -> Self {
        self.ops.push(FusedStep::Scalar { op, value });
        self
    }

    /// Append a unary step.
    pub fn then_unary(mut self, op: ElemOp) -> Self {
        self.ops.push(FusedStep::Unary { op });
        self
    }

    /// Number of fused steps.
    pub fn step_count(&self) -> usize {
        self.ops.len()
    }

    /// Execute the fused pipeline on `input`, writing to `out`.
    pub fn execute(&self, input: &[f32], out: &mut [f32]) -> Result<(), ElemwiseError> {
        if input.len() != out.len() {
            return Err(ElemwiseError::DataShapeMismatch {
                expected: input.len(),
                actual: out.len(),
            });
        }
        // Start from input.
        out.copy_from_slice(input);

        for step in &self.ops {
            match step {
                FusedStep::Binary { op, operand } => {
                    if operand.len() != out.len() {
                        return Err(ElemwiseError::ShapeMismatch {
                            a: vec![out.len()],
                            b: vec![operand.len()],
                        });
                    }
                    if *op == ElemOp::Div && operand.contains(&0.0) {
                        return Err(ElemwiseError::DivisionByZero);
                    }
                    for (i, out_v) in out.iter_mut().enumerate() {
                        *out_v = op.apply(*out_v, operand[i]);
                    }
                }
                FusedStep::Scalar { op, value } => {
                    if *op == ElemOp::Div && *value == 0.0 {
                        return Err(ElemwiseError::DivisionByZero);
                    }
                    for out_v in out.iter_mut() {
                        *out_v = op.apply(*out_v, *value);
                    }
                }
                FusedStep::Unary { op } => {
                    for out_v in out.iter_mut() {
                        *out_v = op.apply(*out_v, 0.0);
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for FusedElemwise {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScalarBroadcast — efficient scalar-to-tensor broadcast
// ---------------------------------------------------------------------------

/// Broadcast a scalar value across a tensor without allocating a full copy.
pub struct ScalarBroadcast;

impl ScalarBroadcast {
    /// Apply `tensor <op> scalar` element-wise.
    pub fn apply(
        op: ElemOp,
        tensor: &[f32],
        scalar: f32,
        out: &mut [f32],
    ) -> Result<(), ElemwiseError> {
        if tensor.len() != out.len() {
            return Err(ElemwiseError::DataShapeMismatch {
                expected: tensor.len(),
                actual: out.len(),
            });
        }
        if op == ElemOp::Div && scalar == 0.0 {
            return Err(ElemwiseError::DivisionByZero);
        }
        for i in 0..tensor.len() {
            out[i] = op.apply(tensor[i], scalar);
        }
        Ok(())
    }

    /// Scale every element: `out[i] = tensor[i] * scalar`.
    pub fn scale(tensor: &[f32], scalar: f32, out: &mut [f32]) -> Result<(), ElemwiseError> {
        Self::apply(ElemOp::Scale, tensor, scalar, out)
    }

    /// Add scalar: `out[i] = tensor[i] + scalar`.
    pub fn add_scalar(tensor: &[f32], scalar: f32, out: &mut [f32]) -> Result<(), ElemwiseError> {
        Self::apply(ElemOp::Add, tensor, scalar, out)
    }
}

// ---------------------------------------------------------------------------
// InPlaceOps — in-place elementwise operations
// ---------------------------------------------------------------------------

/// In-place elementwise operations for memory efficiency.
///
/// These modify the target tensor directly, avoiding allocation of a
/// separate output buffer.
pub struct InPlaceOps;

impl InPlaceOps {
    /// `a[i] += b[i]`
    pub fn add_assign(a: &mut [f32], b: &[f32]) -> Result<(), ElemwiseError> {
        if a.len() != b.len() {
            return Err(ElemwiseError::ShapeMismatch { a: vec![a.len()], b: vec![b.len()] });
        }
        for i in 0..a.len() {
            a[i] += b[i];
        }
        Ok(())
    }

    /// `a[i] -= b[i]`
    pub fn sub_assign(a: &mut [f32], b: &[f32]) -> Result<(), ElemwiseError> {
        if a.len() != b.len() {
            return Err(ElemwiseError::ShapeMismatch { a: vec![a.len()], b: vec![b.len()] });
        }
        for i in 0..a.len() {
            a[i] -= b[i];
        }
        Ok(())
    }

    /// `a[i] *= b[i]`
    pub fn mul_assign(a: &mut [f32], b: &[f32]) -> Result<(), ElemwiseError> {
        if a.len() != b.len() {
            return Err(ElemwiseError::ShapeMismatch { a: vec![a.len()], b: vec![b.len()] });
        }
        for i in 0..a.len() {
            a[i] *= b[i];
        }
        Ok(())
    }

    /// `a[i] *= scalar`
    pub fn scale_assign(a: &mut [f32], scalar: f32) {
        for v in a.iter_mut() {
            *v *= scalar;
        }
    }

    /// Residual connection in-place: `a[i] += residual[i]`.
    pub fn residual_assign(a: &mut [f32], residual: &[f32]) -> Result<(), ElemwiseError> {
        Self::add_assign(a, residual)
    }

    /// Clamp in-place: `a[i] = a[i].clamp(min, max)`.
    pub fn clamp_assign(a: &mut [f32], min_val: f32, max_val: f32) -> Result<(), ElemwiseError> {
        if min_val > max_val {
            return Err(ElemwiseError::InvalidClampBounds { min: min_val, max: max_val });
        }
        for v in a.iter_mut() {
            *v = v.clamp(min_val, max_val);
        }
        Ok(())
    }

    /// Negate in-place: `a[i] = -a[i]`.
    pub fn neg_assign(a: &mut [f32]) {
        for v in a.iter_mut() {
            *v = -*v;
        }
    }

    /// Absolute value in-place: `a[i] = |a[i]|`.
    pub fn abs_assign(a: &mut [f32]) {
        for v in a.iter_mut() {
            *v = v.abs();
        }
    }
}

// ---------------------------------------------------------------------------
// ElemwiseStats — performance measurement
// ---------------------------------------------------------------------------

/// Performance statistics for an elementwise kernel invocation.
#[derive(Debug, Clone)]
pub struct ElemwiseStats {
    /// Effective throughput in GB/s.
    pub throughput_gb_s: f64,
    /// Kernel execution time in microseconds.
    pub kernel_time_us: f64,
    /// Fraction of peak memory bandwidth utilised (0.0–1.0).
    pub bandwidth_utilization: f64,
    /// Number of elements processed.
    pub elements: usize,
    /// Operation that was measured.
    pub op: ElemOp,
}

impl ElemwiseStats {
    /// Compute stats from a timed kernel run.
    ///
    /// `bytes_transferred` = total bytes read + written.
    /// `peak_bandwidth_gb_s` = theoretical peak of the device (A770 ≈ 560 GB/s).
    pub fn from_timing(
        op: ElemOp,
        elements: usize,
        bytes_transferred: usize,
        elapsed: std::time::Duration,
        peak_bandwidth_gb_s: f64,
    ) -> Self {
        let secs = elapsed.as_secs_f64();
        let throughput_gb_s = if secs > 0.0 { bytes_transferred as f64 / secs / 1e9 } else { 0.0 };
        let utilization =
            if peak_bandwidth_gb_s > 0.0 { throughput_gb_s / peak_bandwidth_gb_s } else { 0.0 };
        Self {
            throughput_gb_s,
            kernel_time_us: secs * 1e6,
            bandwidth_utilization: utilization,
            elements,
            op,
        }
    }

    /// Convenience: measure a binary operation and return (result, stats).
    pub fn measure_binary(
        op: ElemOp,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        peak_bw: f64,
    ) -> Result<Self, ElemwiseError> {
        let start = Instant::now();
        ElemwiseKernel::apply_binary(op, a, b, out)?;
        let elapsed = start.elapsed();
        // Binary: reads 2 inputs + writes 1 output = 3 × n × 4 bytes.
        let bytes = 3 * a.len() * std::mem::size_of::<f32>();
        Ok(Self::from_timing(op, a.len(), bytes, elapsed, peak_bw))
    }
}

impl fmt::Display for ElemwiseStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} elems, {:.1} µs, {:.2} GB/s ({:.1}% BW)",
            self.op,
            self.elements,
            self.kernel_time_us,
            self.throughput_gb_s,
            self.bandwidth_utilization * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source — vectorised float4 elementwise operations
// ---------------------------------------------------------------------------

/// Embedded OpenCL kernel source for A770-optimised elementwise operations.
///
/// Uses `float4` vectorisation for 4× throughput on the Xe-HPG ALUs and
/// coalesced global memory access patterns.
pub const ELEMENTWISE_CL: &str = r#"
// Elementwise binary operations with float4 vectorisation.
// Each work-item processes 4 consecutive elements.

// Op codes matching ElemOp enum ordinals:
//   0=Add, 1=Sub, 2=Mul, 3=Div, 4=Scale, 5=Residual, 6=Max, 7=Min

inline float4 apply_op(float4 a, float4 b, int op) {
    switch (op) {
        case 0: return a + b;          // Add
        case 1: return a - b;          // Sub
        case 2: return a * b;          // Mul
        case 3: return a / b;          // Div
        case 4: return a * b;          // Scale
        case 5: return a + b;          // Residual
        case 6: return fmax(a, b);     // Max
        case 7: return fmin(a, b);     // Min
        default: return a;
    }
}

__kernel void elemwise_binary(
    __global const float4* a,
    __global const float4* b,
    __global       float4* out,
    const int n_vec4,
    const int op
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        out[gid] = apply_op(a[gid], b[gid], op);
    }
}

// Scalar broadcast: apply op(tensor[i], scalar) for each element.
__kernel void elemwise_scalar(
    __global const float4* tensor,
    __global       float4* out,
    const int n_vec4,
    const float scalar,
    const int op
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        float4 s = (float4)(scalar, scalar, scalar, scalar);
        out[gid] = apply_op(tensor[gid], s, op);
    }
}

// Unary operations: abs, neg.
__kernel void elemwise_unary(
    __global const float4* in_data,
    __global       float4* out,
    const int n_vec4,
    const int op  // 8=Abs, 9=Neg
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        float4 v = in_data[gid];
        if (op == 8) {
            out[gid] = fabs(v);
        } else if (op == 9) {
            out[gid] = -v;
        }
    }
}

// Clamp: out[i] = clamp(in[i], min_val, max_val).
__kernel void elemwise_clamp(
    __global const float4* in_data,
    __global       float4* out,
    const int n_vec4,
    const float min_val,
    const float max_val
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        out[gid] = clamp(in_data[gid], (float4)(min_val), (float4)(max_val));
    }
}

// Fused scale + add (residual connection): out = x * scale + residual.
__kernel void fused_scale_add(
    __global const float4* x,
    __global const float4* residual,
    __global       float4* out,
    const int n_vec4,
    const float scale
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        out[gid] = x[gid] * (float4)(scale) + residual[gid];
    }
}

// In-place add: a[i] += b[i].
__kernel void inplace_add(
    __global float4* a,
    __global const float4* b,
    const int n_vec4
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        a[gid] += b[gid];
    }
}

// In-place scale: a[i] *= scalar.
__kernel void inplace_scale(
    __global float4* a,
    const int n_vec4,
    const float scalar
) {
    int gid = get_global_id(0);
    if (gid < n_vec4) {
        a[gid] *= (float4)(scalar);
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a vec of f32 from a range.
    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        if n <= 1 {
            return vec![start];
        }
        let step = (end - start) / (n - 1) as f32;
        (0..n).map(|i| start + step * i as f32).collect()
    }

    // ===== ElemOp basic properties =====

    #[test]
    fn test_elem_op_display() {
        assert_eq!(ElemOp::Add.to_string(), "add");
        assert_eq!(ElemOp::Residual.to_string(), "residual");
        assert_eq!(ElemOp::Clamp.to_string(), "clamp");
    }

    #[test]
    fn test_elem_op_is_unary() {
        assert!(ElemOp::Abs.is_unary());
        assert!(ElemOp::Neg.is_unary());
        assert!(ElemOp::Clamp.is_unary());
        assert!(!ElemOp::Add.is_unary());
        assert!(!ElemOp::Mul.is_unary());
        assert!(!ElemOp::Scale.is_unary());
    }

    // ===== Element-by-element binary ops =====

    #[test]
    fn test_add_elementwise() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_sub_elementwise() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Sub, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 2.0, 0.25, 10.0];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_binary(ElemOp::Mul, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 6.0, 1.0, 50.0]);
    }

    #[test]
    fn test_div_elementwise() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![2.0, 5.0, 10.0];
        let mut out = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Div, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_div_by_zero_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        let err = ElemwiseKernel::apply_binary(ElemOp::Div, &a, &b, &mut out);
        assert!(matches!(err, Err(ElemwiseError::DivisionByZero)));
    }

    #[test]
    fn test_max_elementwise() {
        let a = vec![1.0, 5.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let mut out = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Max, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_min_elementwise() {
        let a = vec![1.0, 5.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let mut out = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Min, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_residual_elementwise() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_binary(ElemOp::Residual, &x, &residual, &mut out).unwrap();
        for i in 0..4 {
            assert!((out[i] - (x[i] + residual[i])).abs() < 1e-6, "residual mismatch at {i}");
        }
    }

    // ===== Unary ops =====

    #[test]
    fn test_abs_unary() {
        let a = vec![-1.0, 0.0, 3.0, -5.5];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_unary(ElemOp::Abs, &a, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 0.0, 3.0, 5.5]);
    }

    #[test]
    fn test_neg_unary() {
        let a = vec![1.0, -2.0, 0.0, 3.5];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_unary(ElemOp::Neg, &a, &mut out).unwrap();
        assert_eq!(out, vec![-1.0, 2.0, 0.0, -3.5]);
    }

    // ===== Clamp =====

    #[test]
    fn test_clamp_basic() {
        let a = vec![-2.0, 0.5, 1.5, 3.0];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::clamp(&a, 0.0, 2.0, &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.5, 1.5, 2.0]);
    }

    #[test]
    fn test_clamp_all_within() {
        let a = vec![0.1, 0.5, 0.9];
        let mut out = vec![0.0; 3];
        ElemwiseKernel::clamp(&a, 0.0, 1.0, &mut out).unwrap();
        assert_eq!(out, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn test_clamp_invalid_bounds() {
        let a = vec![1.0];
        let mut out = vec![0.0];
        let err = ElemwiseKernel::clamp(&a, 5.0, 2.0, &mut out);
        assert!(matches!(err, Err(ElemwiseError::InvalidClampBounds { .. })));
    }

    #[test]
    fn test_clamp_equal_bounds() {
        let a = vec![-1.0, 0.0, 1.0, 5.0];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::clamp(&a, 2.0, 2.0, &mut out).unwrap();
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
    }

    // ===== Scalar broadcast =====

    #[test]
    fn test_scalar_add() {
        let t = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        ScalarBroadcast::add_scalar(&t, 10.0, &mut out).unwrap();
        assert_eq!(out, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_scalar_scale() {
        let t = vec![2.0, 4.0, 6.0];
        let mut out = vec![0.0; 3];
        ScalarBroadcast::scale(&t, 0.5, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let t = vec![3.0, 5.0];
        let mut out = vec![0.0; 2];
        ScalarBroadcast::apply(ElemOp::Mul, &t, 2.0, &mut out).unwrap();
        assert_eq!(out, vec![6.0, 10.0]);
    }

    #[test]
    fn test_scalar_div() {
        let t = vec![10.0, 20.0];
        let mut out = vec![0.0; 2];
        ScalarBroadcast::apply(ElemOp::Div, &t, 5.0, &mut out).unwrap();
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn test_scalar_div_by_zero() {
        let t = vec![1.0];
        let mut out = vec![0.0];
        let err = ScalarBroadcast::apply(ElemOp::Div, &t, 0.0, &mut out);
        assert!(matches!(err, Err(ElemwiseError::DivisionByZero)));
    }

    #[test]
    fn test_scalar_sub() {
        let t = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0; 3];
        ScalarBroadcast::apply(ElemOp::Sub, &t, 5.0, &mut out).unwrap();
        assert_eq!(out, vec![5.0, 15.0, 25.0]);
    }

    #[test]
    fn test_scalar_max() {
        let t = vec![-1.0, 0.0, 1.0, 5.0];
        let mut out = vec![0.0; 4];
        ScalarBroadcast::apply(ElemOp::Max, &t, 0.0, &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 1.0, 5.0]);
    }

    #[test]
    fn test_scalar_min() {
        let t = vec![-1.0, 0.0, 1.0, 5.0];
        let mut out = vec![0.0; 4];
        ScalarBroadcast::apply(ElemOp::Min, &t, 1.0, &mut out).unwrap();
        assert_eq!(out, vec![-1.0, 0.0, 1.0, 1.0]);
    }

    // ===== BroadcastRule (NumPy broadcasting) =====

    #[test]
    fn test_broadcast_same_shape() {
        let rule = BroadcastRule::new(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(rule.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_scalar_to_vector() {
        let rule = BroadcastRule::new(&[4], &[1]).unwrap();
        assert_eq!(rule.output_shape, vec![4]);
        assert_eq!(rule.strides_a, vec![1]);
        assert_eq!(rule.strides_b, vec![0]);
    }

    #[test]
    fn test_broadcast_row_to_matrix() {
        let rule = BroadcastRule::new(&[3, 4], &[1, 4]).unwrap();
        assert_eq!(rule.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_col_to_matrix() {
        let rule = BroadcastRule::new(&[3, 1], &[3, 4]).unwrap();
        assert_eq!(rule.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_different_ndim() {
        let rule = BroadcastRule::new(&[4], &[3, 4]).unwrap();
        assert_eq!(rule.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes() {
        let err = BroadcastRule::new(&[3], &[4]);
        assert!(matches!(err, Err(ElemwiseError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_broadcast_3d() {
        let rule = BroadcastRule::new(&[2, 1, 4], &[1, 3, 4]).unwrap();
        assert_eq!(rule.output_shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_apply_row_vector() {
        // [2, 3] + [1, 3] → broadcast row across rows.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2,3]
        let b = vec![10.0, 20.0, 30.0]; // shape [1,3]
        let mut out = vec![0.0; 6];
        ElemwiseKernel::apply_broadcast(ElemOp::Add, &a, &[2, 3], &b, &[1, 3], &mut out).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_apply_col_vector() {
        // [2, 3] * [2, 1] → broadcast column across columns.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2,3]
        let b = vec![10.0, 100.0]; // shape [2,1]
        let mut out = vec![0.0; 6];
        ElemwiseKernel::apply_broadcast(ElemOp::Mul, &a, &[2, 3], &b, &[2, 1], &mut out).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0, 400.0, 500.0, 600.0]);
    }

    #[test]
    fn test_broadcast_scalar_to_matrix() {
        // [2, 2] + [1] → scalar broadcast.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![100.0];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_broadcast(ElemOp::Add, &a, &[2, 2], &b, &[1], &mut out).unwrap();
        assert_eq!(out, vec![101.0, 102.0, 103.0, 104.0]);
    }

    #[test]
    fn test_broadcast_incompatible_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let mut out = vec![0.0; 6];
        let err = ElemwiseKernel::apply_broadcast(ElemOp::Add, &a, &[3], &b, &[2], &mut out);
        assert!(matches!(err, Err(ElemwiseError::ShapeMismatch { .. })));
    }

    // ===== FusedElemwise =====

    #[test]
    fn test_fused_scale_add() {
        // fused: scale(2.0) + add(bias) should equal sequential.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        let mut fused_out = vec![0.0; 4];
        let pipeline = FusedElemwise::new()
            .then_scalar(ElemOp::Scale, 2.0)
            .then_binary(ElemOp::Add, bias.clone());
        pipeline.execute(&input, &mut fused_out).unwrap();

        // Sequential reference.
        let mut seq_out = vec![0.0; 4];
        ScalarBroadcast::scale(&input, 2.0, &mut seq_out).unwrap();
        InPlaceOps::add_assign(&mut seq_out, &bias).unwrap();

        for i in 0..4 {
            assert!(
                (fused_out[i] - seq_out[i]).abs() < 1e-6,
                "fused vs seq mismatch at {i}: {} vs {}",
                fused_out[i],
                seq_out[i],
            );
        }
    }

    #[test]
    fn test_fused_matches_sequential() {
        let input = vec![1.0, 2.0, 3.0];
        let other = vec![10.0, 20.0, 30.0];

        // Fused: add(other) then neg.
        let mut fused_out = vec![0.0; 3];
        let pipeline =
            FusedElemwise::new().then_binary(ElemOp::Add, other.clone()).then_unary(ElemOp::Neg);
        pipeline.execute(&input, &mut fused_out).unwrap();

        // Sequential: add, then negate.
        let mut seq_out = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Add, &input, &other, &mut seq_out).unwrap();
        InPlaceOps::neg_assign(&mut seq_out);

        assert_eq!(fused_out, seq_out);
    }

    #[test]
    fn test_fused_three_steps() {
        let input = vec![4.0, 9.0, 16.0];
        let mut out = vec![0.0; 3];

        // scale(0.5), add([1,1,1]), abs.
        let pipeline = FusedElemwise::new()
            .then_scalar(ElemOp::Scale, 0.5)
            .then_binary(ElemOp::Add, vec![1.0, 1.0, 1.0])
            .then_unary(ElemOp::Abs);
        pipeline.execute(&input, &mut out).unwrap();

        // 4*0.5+1=3, 9*0.5+1=5.5, 16*0.5+1=9
        assert_eq!(out, vec![3.0, 5.5, 9.0]);
    }

    #[test]
    fn test_fused_step_count() {
        let p = FusedElemwise::new().then_scalar(ElemOp::Scale, 2.0).then_unary(ElemOp::Neg);
        assert_eq!(p.step_count(), 2);
    }

    #[test]
    fn test_fused_empty_pipeline() {
        let input = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        FusedElemwise::new().execute(&input, &mut out).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn test_fused_div_by_zero() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0; 2];
        let pipeline = FusedElemwise::new().then_scalar(ElemOp::Div, 0.0);
        let err = pipeline.execute(&input, &mut out);
        assert!(matches!(err, Err(ElemwiseError::DivisionByZero)));
    }

    #[test]
    fn test_fused_shape_mismatch() {
        let input = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        let pipeline = FusedElemwise::new().then_binary(ElemOp::Add, vec![1.0, 2.0]); // wrong len
        let err = pipeline.execute(&input, &mut out);
        assert!(matches!(err, Err(ElemwiseError::ShapeMismatch { .. })));
    }

    // ===== InPlaceOps =====

    #[test]
    fn test_inplace_add_assign() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        InPlaceOps::add_assign(&mut a, &b).unwrap();
        assert_eq!(a, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_inplace_sub_assign() {
        let mut a = vec![10.0, 20.0, 30.0];
        let b = vec![1.0, 2.0, 3.0];
        InPlaceOps::sub_assign(&mut a, &b).unwrap();
        assert_eq!(a, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_inplace_mul_assign() {
        let mut a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 10.0, 0.5];
        InPlaceOps::mul_assign(&mut a, &b).unwrap();
        assert_eq!(a, vec![10.0, 30.0, 2.0]);
    }

    #[test]
    fn test_inplace_scale_assign() {
        let mut a = vec![2.0, 4.0, 6.0];
        InPlaceOps::scale_assign(&mut a, 0.5);
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_inplace_residual_assign() {
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4];
        InPlaceOps::residual_assign(&mut hidden, &residual).unwrap();
        for i in 0..4 {
            assert!((hidden[i] - (1.0 + i as f32 + 0.1 * (i as f32 + 1.0))).abs() < 1e-6);
        }
    }

    #[test]
    fn test_inplace_clamp_assign() {
        let mut a = vec![-5.0, 0.0, 0.5, 1.0, 10.0];
        InPlaceOps::clamp_assign(&mut a, 0.0, 1.0).unwrap();
        assert_eq!(a, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_inplace_neg_assign() {
        let mut a = vec![1.0, -2.0, 0.0, 3.5];
        InPlaceOps::neg_assign(&mut a);
        assert_eq!(a, vec![-1.0, 2.0, 0.0, -3.5]);
    }

    #[test]
    fn test_inplace_abs_assign() {
        let mut a = vec![-1.0, -2.5, 0.0, 3.0];
        InPlaceOps::abs_assign(&mut a);
        assert_eq!(a, vec![1.0, 2.5, 0.0, 3.0]);
    }

    #[test]
    fn test_inplace_shape_mismatch() {
        let mut a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let err = InPlaceOps::add_assign(&mut a, &b);
        assert!(matches!(err, Err(ElemwiseError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_inplace_clamp_invalid_bounds() {
        let mut a = vec![1.0];
        let err = InPlaceOps::clamp_assign(&mut a, 5.0, 2.0);
        assert!(matches!(err, Err(ElemwiseError::InvalidClampBounds { .. })));
    }

    // ===== Various tensor sizes =====

    #[test]
    fn test_small_tensor_single_element() {
        let a = vec![42.0];
        let b = vec![8.0];
        let mut out = vec![0.0];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut out).unwrap();
        assert_eq!(out, vec![50.0]);
    }

    #[test]
    fn test_medium_tensor_256() {
        let n = 256;
        let a = linspace(0.0, 1.0, n);
        let b = linspace(1.0, 0.0, n);
        let mut out = vec![0.0; n];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut out).unwrap();
        // a[i] + b[i] should be approximately 1.0 for all i.
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_large_tensor_4096() {
        let n = 4096;
        let a = vec![1.5; n];
        let b = vec![2.5; n];
        let mut out = vec![0.0; n];
        ElemwiseKernel::apply_binary(ElemOp::Mul, &a, &b, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v - 3.75).abs() < 1e-6));
    }

    #[test]
    fn test_large_tensor_16384() {
        let n = 16384;
        let a = vec![0.5; n];
        let mut out = vec![0.0; n];
        ScalarBroadcast::scale(&a, 4.0, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // ===== Edge cases =====

    #[test]
    fn test_zero_length_tensor() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let mut out = vec![0.0; 3];
        let err = ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut out);
        assert!(matches!(err, Err(ElemwiseError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_unary_length_mismatch() {
        let a = vec![1.0, 2.0];
        let mut out = vec![0.0; 3];
        let err = ElemwiseKernel::apply_unary(ElemOp::Abs, &a, &mut out);
        assert!(matches!(err, Err(ElemwiseError::DataShapeMismatch { .. })));
    }

    #[test]
    fn test_scalar_broadcast_length_mismatch() {
        let t = vec![1.0, 2.0];
        let mut out = vec![0.0; 3];
        let err = ScalarBroadcast::apply(ElemOp::Add, &t, 1.0, &mut out);
        assert!(matches!(err, Err(ElemwiseError::DataShapeMismatch { .. })));
    }

    // ===== Property-style tests =====

    #[test]
    fn test_property_add_sub_roundtrip() {
        // a + b - b ≈ a (within f32 precision).
        let a = vec![1.0, -2.5, std::f32::consts::PI, 0.0, 100.0];
        let b = vec![10.0, 20.0, -5.0, 0.0, 1e-6];
        let mut sum = vec![0.0; 5];
        let mut result = vec![0.0; 5];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut sum).unwrap();
        ElemwiseKernel::apply_binary(ElemOp::Sub, &sum, &b, &mut result).unwrap();
        for i in 0..5 {
            assert!(
                (result[i] - a[i]).abs() < 1e-4,
                "roundtrip failed at {i}: {} vs {}",
                result[i],
                a[i],
            );
        }
    }

    #[test]
    fn test_property_mul_div_roundtrip() {
        // (a * b) / b ≈ a for non-zero b.
        let a = vec![1.0, 2.5, 0.1, 100.0];
        let b = vec![2.0, 0.5, 10.0, 0.01];
        let mut prod = vec![0.0; 4];
        let mut result = vec![0.0; 4];
        ElemwiseKernel::apply_binary(ElemOp::Mul, &a, &b, &mut prod).unwrap();
        ElemwiseKernel::apply_binary(ElemOp::Div, &prod, &b, &mut result).unwrap();
        for i in 0..4 {
            assert!(
                (result[i] - a[i]).abs() < 1e-4,
                "mul/div roundtrip at {i}: {} vs {}",
                result[i],
                a[i],
            );
        }
    }

    #[test]
    fn test_property_add_commutative() {
        let a = vec![1.0, 3.0, -2.0];
        let b = vec![4.0, -1.0, 7.0];
        let mut ab = vec![0.0; 3];
        let mut ba = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &b, &mut ab).unwrap();
        ElemwiseKernel::apply_binary(ElemOp::Add, &b, &a, &mut ba).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_property_mul_commutative() {
        let a = vec![2.0, 0.5, -3.0];
        let b = vec![4.0, -1.0, 7.0];
        let mut ab = vec![0.0; 3];
        let mut ba = vec![0.0; 3];
        ElemwiseKernel::apply_binary(ElemOp::Mul, &a, &b, &mut ab).unwrap();
        ElemwiseKernel::apply_binary(ElemOp::Mul, &b, &a, &mut ba).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_property_neg_involution() {
        // neg(neg(a)) == a.
        let a = vec![1.0, -2.0, 0.0, 3.5];
        let mut neg1 = vec![0.0; 4];
        let mut neg2 = vec![0.0; 4];
        ElemwiseKernel::apply_unary(ElemOp::Neg, &a, &mut neg1).unwrap();
        ElemwiseKernel::apply_unary(ElemOp::Neg, &neg1, &mut neg2).unwrap();
        assert_eq!(neg2, a);
    }

    #[test]
    fn test_property_abs_idempotent() {
        // abs(abs(a)) == abs(a).
        let a = vec![-1.0, 2.0, -3.5, 0.0];
        let mut abs1 = vec![0.0; 4];
        let mut abs2 = vec![0.0; 4];
        ElemwiseKernel::apply_unary(ElemOp::Abs, &a, &mut abs1).unwrap();
        ElemwiseKernel::apply_unary(ElemOp::Abs, &abs1, &mut abs2).unwrap();
        assert_eq!(abs1, abs2);
    }

    #[test]
    fn test_property_scale_one_identity() {
        let a = vec![1.0, -2.0, std::f32::consts::PI, 0.0];
        let mut out = vec![0.0; 4];
        ScalarBroadcast::scale(&a, 1.0, &mut out).unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn test_property_add_zero_identity() {
        let a = vec![1.0, -2.0, std::f32::consts::PI, 0.0];
        let zeros = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        ElemwiseKernel::apply_binary(ElemOp::Add, &a, &zeros, &mut out).unwrap();
        assert_eq!(out, a);
    }

    // ===== ElemwiseStats =====

    #[test]
    fn test_stats_from_timing() {
        let stats = ElemwiseStats::from_timing(
            ElemOp::Add,
            1024,
            1024 * 3 * 4, // 3 tensors × 4 bytes
            std::time::Duration::from_micros(100),
            560.0,
        );
        assert_eq!(stats.elements, 1024);
        assert_eq!(stats.op, ElemOp::Add);
        assert!(stats.throughput_gb_s > 0.0);
        assert!(stats.kernel_time_us > 0.0);
        assert!(stats.bandwidth_utilization > 0.0);
        assert!(stats.bandwidth_utilization <= 1.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = ElemwiseStats::from_timing(
            ElemOp::Mul,
            256,
            256 * 3 * 4,
            std::time::Duration::from_micros(50),
            560.0,
        );
        let text = stats.to_string();
        assert!(text.contains("mul"));
        assert!(text.contains("256 elems"));
    }

    #[test]
    fn test_stats_measure_binary() {
        let a = vec![1.0; 128];
        let b = vec![2.0; 128];
        let mut out = vec![0.0; 128];
        let stats = ElemwiseStats::measure_binary(ElemOp::Add, &a, &b, &mut out, 560.0).unwrap();
        assert_eq!(stats.elements, 128);
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_stats_zero_duration() {
        let stats = ElemwiseStats::from_timing(ElemOp::Add, 0, 0, std::time::Duration::ZERO, 0.0);
        assert_eq!(stats.throughput_gb_s, 0.0);
        assert_eq!(stats.bandwidth_utilization, 0.0);
    }

    // ===== OpenCL kernel source =====

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!ELEMENTWISE_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(ELEMENTWISE_CL.contains("elemwise_binary"));
        assert!(ELEMENTWISE_CL.contains("elemwise_scalar"));
        assert!(ELEMENTWISE_CL.contains("elemwise_unary"));
        assert!(ELEMENTWISE_CL.contains("elemwise_clamp"));
        assert!(ELEMENTWISE_CL.contains("fused_scale_add"));
        assert!(ELEMENTWISE_CL.contains("inplace_add"));
        assert!(ELEMENTWISE_CL.contains("inplace_scale"));
    }

    #[test]
    fn test_opencl_source_uses_float4() {
        assert!(ELEMENTWISE_CL.contains("float4"));
    }

    // ===== ElemwiseError display =====

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = ElemwiseError::ShapeMismatch { a: vec![3], b: vec![4] };
        assert!(e.to_string().contains("[3]"));
        assert!(e.to_string().contains("[4]"));
    }

    #[test]
    fn test_error_display_div_zero() {
        let e = ElemwiseError::DivisionByZero;
        assert!(e.to_string().contains("division by zero"));
    }

    #[test]
    fn test_error_display_clamp_bounds() {
        let e = ElemwiseError::InvalidClampBounds { min: 5.0, max: 2.0 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("2"));
    }

    #[test]
    fn test_error_display_data_shape() {
        let e = ElemwiseError::DataShapeMismatch { expected: 10, actual: 5 };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("5"));
    }

    // ===== BroadcastRule output_len =====

    #[test]
    fn test_broadcast_output_len() {
        let rule = BroadcastRule::new(&[2, 1, 4], &[1, 3, 4]).unwrap();
        assert_eq!(rule.output_len(), 24);
    }
}
