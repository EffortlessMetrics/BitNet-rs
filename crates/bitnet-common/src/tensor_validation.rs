//! Tensor shape validation and broadcasting utilities for neural network operations.
//!
//! Provides compile-time-style shape checking for matmul, broadcasting,
//! attention, reshape, and transpose — following NumPy broadcasting semantics.

use std::fmt;

use thiserror::Error;

/// Errors arising from tensor shape validation.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    #[error("matmul shape mismatch: a has {a_inner} columns but b has {b_inner} rows")]
    MatmulMismatch { a_inner: usize, b_inner: usize },

    #[error("matmul requires at least 1-D tensors, got a={a_ndim}-D and b={b_ndim}-D")]
    MatmulRank { a_ndim: usize, b_ndim: usize },

    #[error(
        "matmul batch dimensions incompatible: \
         a batch {a_batch:?} vs b batch {b_batch:?}"
    )]
    MatmulBatchMismatch { a_batch: Vec<usize>, b_batch: Vec<usize> },

    #[error("broadcast incompatible: dimension {dim} has sizes {a} and {b}")]
    BroadcastIncompatible { dim: usize, a: usize, b: usize },

    #[error("attention shape mismatch: Q={q:?}, K={k:?}, V={v:?} — {reason}")]
    AttentionShape { q: Vec<usize>, k: Vec<usize>, v: Vec<usize>, reason: String },

    #[error(
        "reshape element count mismatch: source has {from_count} elements \
         but target shape {to:?} requires {to_count}"
    )]
    ReshapeElementCount { from_count: usize, to: Vec<usize>, to_count: usize },

    #[error("transpose axis {axis} out of range for {ndim}-D tensor")]
    TransposeAxisOutOfRange { axis: usize, ndim: usize },

    #[error("transpose axes {axes:?} do not form a permutation of 0..{ndim}")]
    TransposeNotPermutation { axes: Vec<usize>, ndim: usize },

    #[error("empty shape is not valid for this operation")]
    EmptyShape,
}

/// Convenience alias used throughout this module.
pub type Result<T> = std::result::Result<T, ShapeError>;

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// Compute the broadcast-compatible output shape (NumPy rules).
///
/// Two dimensions are compatible when they are equal **or** one of them is 1.
/// Shapes are right-aligned before comparison; missing leading dimensions are
/// treated as 1.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
        let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(ShapeError::BroadcastIncompatible { dim: i, a: da, b: db });
        }
    }
    Ok(result)
}

/// Returns `true` when the two shapes are broadcast-compatible.
pub fn can_broadcast(a: &[usize], b: &[usize]) -> bool {
    broadcast_shape(a, b).is_ok()
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

/// Validate shapes for matrix multiplication `a @ b` and return the output shape.
///
/// Supports batched matmul: the last two dimensions are the matrix dims while
/// leading dimensions must be broadcast-compatible.
pub fn validate_matmul_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    if a.is_empty() || b.is_empty() {
        return Err(ShapeError::MatmulRank { a_ndim: a.len(), b_ndim: b.len() });
    }

    // 1-D × 1-D → dot product (scalar)
    if a.len() == 1 && b.len() == 1 {
        if a[0] != b[0] {
            return Err(ShapeError::MatmulMismatch { a_inner: a[0], b_inner: b[0] });
        }
        return Ok(vec![]);
    }

    // 1-D × N-D → treat `a` as (1, K), then squeeze leading 1
    if a.len() == 1 {
        let k = a[0];
        if b[b.len() - 2] != k {
            return Err(ShapeError::MatmulMismatch { a_inner: k, b_inner: b[b.len() - 2] });
        }
        let mut out: Vec<usize> = b[..b.len() - 2].to_vec();
        out.push(b[b.len() - 1]);
        return Ok(out);
    }

    // N-D × 1-D → treat `b` as (K, 1), then squeeze trailing 1
    if b.len() == 1 {
        let k = b[0];
        if a[a.len() - 1] != k {
            return Err(ShapeError::MatmulMismatch { a_inner: a[a.len() - 1], b_inner: k });
        }
        return Ok(a[..a.len() - 1].to_vec());
    }

    // General N-D × M-D
    let a_inner = a[a.len() - 1];
    let b_inner = b[b.len() - 2];
    if a_inner != b_inner {
        return Err(ShapeError::MatmulMismatch { a_inner, b_inner });
    }

    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let batch = broadcast_shape(a_batch, b_batch).map_err(|_| ShapeError::MatmulBatchMismatch {
        a_batch: a_batch.to_vec(),
        b_batch: b_batch.to_vec(),
    })?;

    let mut out = batch;
    out.push(a[a.len() - 2]);
    out.push(b[b.len() - 1]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Validate Q, K, V shapes for multi-head attention.
///
/// Expected layout: `[batch, heads, seq_len, head_dim]`.
/// K and V must share `seq_len`; Q and K must share `head_dim`.
pub fn validate_attention_shapes(q: &[usize], k: &[usize], v: &[usize]) -> Result<()> {
    let err = |reason: &str| ShapeError::AttentionShape {
        q: q.to_vec(),
        k: k.to_vec(),
        v: v.to_vec(),
        reason: reason.to_string(),
    };

    if q.len() != 4 {
        return Err(err(&format!(
            "Q must be 4-D [batch, heads, seq_len, head_dim], got {}-D",
            q.len()
        )));
    }
    if k.len() != 4 {
        return Err(err(&format!(
            "K must be 4-D [batch, heads, kv_len, head_dim], got {}-D",
            k.len()
        )));
    }
    if v.len() != 4 {
        return Err(err(&format!(
            "V must be 4-D [batch, heads, kv_len, v_dim], got {}-D",
            v.len()
        )));
    }

    // Batch must match
    if q[0] != k[0] || q[0] != v[0] {
        return Err(err("batch dimensions must match across Q, K, V"));
    }

    // Head count: K and V must match; Q heads must be a multiple (GQA)
    if k[1] != v[1] {
        return Err(err("K and V must have the same number of heads"));
    }
    if !q[1].is_multiple_of(k[1]) {
        return Err(err("Q head count must be a multiple of K/V head count (for GQA)"));
    }

    // Q and K share head_dim
    if q[3] != k[3] {
        return Err(err("Q head_dim must match K head_dim"));
    }

    // K and V share kv_len
    if k[2] != v[2] {
        return Err(err("K and V must have the same sequence length"));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

/// Validate that `from` can be reshaped to `to` (element counts must match).
pub fn validate_reshape(from: &[usize], to: &[usize]) -> Result<()> {
    let from_count: usize = from.iter().product();
    let to_count: usize = to.iter().product();
    if from_count != to_count {
        return Err(ShapeError::ReshapeElementCount { from_count, to: to.to_vec(), to_count });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

/// Validate `axes` as a permutation of `0..shape.len()` and return the
/// resulting shape.
pub fn validate_transpose_axes(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    let ndim = shape.len();
    if axes.len() != ndim {
        return Err(ShapeError::TransposeNotPermutation { axes: axes.to_vec(), ndim });
    }

    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim {
            return Err(ShapeError::TransposeAxisOutOfRange { axis, ndim });
        }
        seen[axis] = true;
    }
    if seen.iter().any(|&s| !s) {
        return Err(ShapeError::TransposeNotPermutation { axes: axes.to_vec(), ndim });
    }

    Ok(axes.iter().map(|&ax| shape[ax]).collect())
}

// ---------------------------------------------------------------------------
// Display helpers (kept private; errors use thiserror Display)
// ---------------------------------------------------------------------------

fn _fmt_shape(shape: &[usize], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[")?;
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{d}")?;
    }
    write!(f, "]")
}

// =========================================================================
// Production Tensor Data Validation
// =========================================================================

/// Lightweight trait for tensor data validation.
///
/// Implement this for any tensor-like type to enable production safety
/// checks (NaN/Inf detection, value range, alignment) without coupling
/// to heavy framework dependencies.
pub trait TensorLike: Send + Sync {
    /// Returns the tensor shape dimensions.
    fn shape(&self) -> &[usize];

    /// Returns f32 data if available (for NaN/Inf/range checks).
    /// Return `None` to skip data-level validation.
    fn data_f32(&self) -> Option<&[f32]>;

    /// Returns stride information for memory layout validation.
    /// Return `None` to skip stride validation.
    fn strides(&self) -> Option<&[usize]>;

    /// Returns the byte alignment of the underlying data pointer.
    fn data_alignment(&self) -> usize;
}

/// Configuration for tensor validation with builder pattern.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    max_total_elements: usize,
    max_dimensions: usize,
    check_nan: bool,
    check_inf: bool,
    value_range: Option<(f32, f32)>,
    required_alignment: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_total_elements: 1 << 30, // ~1 billion elements
            max_dimensions: 8,
            check_nan: true,
            check_inf: true,
            value_range: None,
            required_alignment: 1, // no alignment requirement
        }
    }
}

impl ValidationConfig {
    /// Create a new configuration with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum total number of elements allowed.
    pub fn max_total_elements(mut self, max: usize) -> Self {
        self.max_total_elements = max;
        self
    }

    /// Set the maximum number of dimensions allowed.
    pub fn max_dimensions(mut self, max: usize) -> Self {
        self.max_dimensions = max;
        self
    }

    /// Enable or disable NaN checking.
    pub fn check_nan(mut self, check: bool) -> Self {
        self.check_nan = check;
        self
    }

    /// Enable or disable infinity checking.
    pub fn check_inf(mut self, check: bool) -> Self {
        self.check_inf = check;
        self
    }

    /// Set the allowed value range `[min, max]` (inclusive).
    pub fn value_range(mut self, min: f32, max: f32) -> Self {
        self.value_range = Some((min, max));
        self
    }

    /// Set the required data pointer alignment in bytes.
    pub fn required_alignment(mut self, align: usize) -> Self {
        self.required_alignment = align;
        self
    }
}

/// Detailed validation error information.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TensorValidationError {
    #[error("zero dimension at axis {axis} in shape {shape:?}")]
    ZeroDimension { axis: usize, shape: Vec<usize> },

    #[error("total elements {total} exceeds maximum {max}")]
    TotalElementsExceeded { total: usize, max: usize },

    #[error("{ndim} dimensions exceeds maximum {max}")]
    DimensionsExceeded { ndim: usize, max: usize },

    #[error("NaN detected at index {index}")]
    NanDetected { index: usize },

    #[error("Inf detected at index {index}, value={value}")]
    InfDetected { index: usize, value: f32 },

    #[error("value {value} at index {index} outside range [{min}, {max}]")]
    ValueOutOfRange { index: usize, value: f32, min: f32, max: f32 },

    #[error("stride mismatch at axis {axis}: expected {expected} (C-contiguous), got {actual}")]
    StrideInconsistent { axis: usize, expected: usize, actual: usize },

    #[error("alignment {actual} does not meet required {required}")]
    AlignmentViolation { required: usize, actual: usize },
}

/// Thread-safe tensor validator with configurable rules.
///
/// Zero-allocation on the common (valid) path — every check returns
/// `Ok(())` without heap activity when the tensor is well-formed.
#[derive(Debug, Clone)]
pub struct TensorValidator {
    config: ValidationConfig,
}

impl TensorValidator {
    /// Create a validator from the given configuration.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a single tensor against all configured rules.
    pub fn validate<T: TensorLike>(
        &self,
        tensor: &T,
    ) -> std::result::Result<(), TensorValidationError> {
        self.validate_shape(tensor)?;
        self.validate_data(tensor)?;
        self.validate_strides(tensor)?;
        self.validate_alignment(tensor)?;
        Ok(())
    }

    /// Validate a batch of tensors, returning one result per tensor.
    pub fn validate_batch<T: TensorLike>(
        &self,
        tensors: &[T],
    ) -> Vec<std::result::Result<(), TensorValidationError>> {
        tensors.iter().map(|t| self.validate(t)).collect()
    }

    fn validate_shape<T: TensorLike>(
        &self,
        tensor: &T,
    ) -> std::result::Result<(), TensorValidationError> {
        let shape = tensor.shape();

        if shape.len() > self.config.max_dimensions {
            return Err(TensorValidationError::DimensionsExceeded {
                ndim: shape.len(),
                max: self.config.max_dimensions,
            });
        }

        for (axis, &dim) in shape.iter().enumerate() {
            if dim == 0 {
                return Err(TensorValidationError::ZeroDimension { axis, shape: shape.to_vec() });
            }
        }

        let mut total: usize = 1;
        for &dim in shape {
            match total.checked_mul(dim) {
                Some(t) if t <= self.config.max_total_elements => total = t,
                Some(t) => {
                    return Err(TensorValidationError::TotalElementsExceeded {
                        total: t,
                        max: self.config.max_total_elements,
                    });
                }
                None => {
                    return Err(TensorValidationError::TotalElementsExceeded {
                        total: usize::MAX,
                        max: self.config.max_total_elements,
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_data<T: TensorLike>(
        &self,
        tensor: &T,
    ) -> std::result::Result<(), TensorValidationError> {
        let data = match tensor.data_f32() {
            Some(d) => d,
            None => return Ok(()),
        };

        for (i, &val) in data.iter().enumerate() {
            if self.config.check_nan && val.is_nan() {
                return Err(TensorValidationError::NanDetected { index: i });
            }
            if self.config.check_inf && val.is_infinite() {
                return Err(TensorValidationError::InfDetected { index: i, value: val });
            }
            if let Some((min, max)) = self.config.value_range
                && (val < min || val > max)
            {
                return Err(TensorValidationError::ValueOutOfRange {
                    index: i,
                    value: val,
                    min,
                    max,
                });
            }
        }

        Ok(())
    }

    fn validate_strides<T: TensorLike>(
        &self,
        tensor: &T,
    ) -> std::result::Result<(), TensorValidationError> {
        let strides = match tensor.strides() {
            Some(s) => s,
            None => return Ok(()),
        };
        let shape = tensor.shape();
        if shape.is_empty() {
            return Ok(());
        }

        let expected = c_contiguous_strides(shape);
        for (axis, (&exp, &act)) in expected.iter().zip(strides.iter()).enumerate() {
            if exp != act {
                return Err(TensorValidationError::StrideInconsistent {
                    axis,
                    expected: exp,
                    actual: act,
                });
            }
        }

        Ok(())
    }

    fn validate_alignment<T: TensorLike>(
        &self,
        tensor: &T,
    ) -> std::result::Result<(), TensorValidationError> {
        let actual = tensor.data_alignment();
        let required = self.config.required_alignment;
        if required > 1 && !actual.is_multiple_of(required) {
            return Err(TensorValidationError::AlignmentViolation { required, actual });
        }
        Ok(())
    }
}

/// Compute C-contiguous (row-major) strides for the given shape.
pub fn c_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- broadcast_shape / can_broadcast --------------------------------

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_scalar_and_tensor() {
        assert_eq!(broadcast_shape(&[], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[3, 4], &[]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_one_expands() {
        assert_eq!(broadcast_shape(&[1, 4], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[3, 1], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_different_ranks() {
        assert_eq!(broadcast_shape(&[4], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 4]).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(broadcast_shape(&[3], &[4]).is_err());
        assert!(broadcast_shape(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn can_broadcast_returns_bool() {
        assert!(can_broadcast(&[1, 3], &[2, 3]));
        assert!(!can_broadcast(&[2], &[3]));
    }

    #[test]
    fn broadcast_both_scalars() {
        assert_eq!(broadcast_shape(&[], &[]).unwrap(), Vec::<usize>::new());
    }

    // ----- validate_matmul_shapes ----------------------------------------

    #[test]
    fn matmul_2d_valid() {
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3, 4]).unwrap(), vec![2, 4]);
    }

    #[test]
    fn matmul_2d_mismatch() {
        let err = validate_matmul_shapes(&[2, 3], &[4, 5]).unwrap_err();
        assert!(matches!(err, ShapeError::MatmulMismatch { a_inner: 3, b_inner: 4 }));
    }

    #[test]
    fn matmul_batched() {
        assert_eq!(validate_matmul_shapes(&[8, 2, 3], &[8, 3, 4]).unwrap(), vec![8, 2, 4]);
    }

    #[test]
    fn matmul_batched_broadcast() {
        assert_eq!(validate_matmul_shapes(&[1, 2, 3], &[8, 3, 4]).unwrap(), vec![8, 2, 4]);
    }

    #[test]
    fn matmul_batch_mismatch() {
        assert!(validate_matmul_shapes(&[7, 2, 3], &[8, 3, 4]).is_err());
    }

    #[test]
    fn matmul_1d_dot() {
        assert_eq!(validate_matmul_shapes(&[3], &[3]).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn matmul_1d_dot_mismatch() {
        assert!(validate_matmul_shapes(&[3], &[4]).is_err());
    }

    #[test]
    fn matmul_1d_times_2d() {
        // (K,) × (K, N) → (N,)
        assert_eq!(validate_matmul_shapes(&[3], &[3, 4]).unwrap(), vec![4]);
    }

    #[test]
    fn matmul_2d_times_1d() {
        // (M, K) × (K,) → (M,)
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3]).unwrap(), vec![2]);
    }

    #[test]
    fn matmul_empty_shape() {
        assert!(validate_matmul_shapes(&[], &[3, 4]).is_err());
        assert!(validate_matmul_shapes(&[3, 4], &[]).is_err());
    }

    // ----- validate_attention_shapes -------------------------------------

    #[test]
    fn attention_valid_mha() {
        // Standard multi-head attention
        let q = [1, 8, 32, 64];
        let k = [1, 8, 32, 64];
        let v = [1, 8, 32, 64];
        assert!(validate_attention_shapes(&q, &k, &v).is_ok());
    }

    #[test]
    fn attention_valid_gqa() {
        // Grouped query attention: Q has 8 heads, K/V have 2
        let q = [1, 8, 32, 64];
        let k = [1, 2, 32, 64];
        let v = [1, 2, 32, 48];
        assert!(validate_attention_shapes(&q, &k, &v).is_ok());
    }

    #[test]
    fn attention_kv_different_seq_len() {
        // K and V must share seq_len
        let q = [1, 8, 32, 64];
        let k = [1, 8, 16, 64];
        let v = [1, 8, 32, 64];
        let err = validate_attention_shapes(&q, &k, &v).unwrap_err();
        match &err {
            ShapeError::AttentionShape { reason, .. } => {
                assert!(reason.contains("sequence length"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn attention_head_dim_mismatch() {
        let q = [1, 8, 32, 64];
        let k = [1, 8, 32, 128];
        let v = [1, 8, 32, 64];
        let err = validate_attention_shapes(&q, &k, &v).unwrap_err();
        match &err {
            ShapeError::AttentionShape { reason, .. } => {
                assert!(reason.contains("head_dim"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn attention_wrong_ndim() {
        assert!(validate_attention_shapes(&[8, 32, 64], &[8, 32, 64], &[8, 32, 64]).is_err());
    }

    #[test]
    fn attention_batch_mismatch() {
        let q = [2, 8, 32, 64];
        let k = [4, 8, 32, 64];
        let v = [4, 8, 32, 64];
        let err = validate_attention_shapes(&q, &k, &v).unwrap_err();
        match &err {
            ShapeError::AttentionShape { reason, .. } => {
                assert!(reason.contains("batch"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn attention_gqa_non_divisible_heads() {
        // Q=7 heads, K/V=3 heads → 7 % 3 ≠ 0
        let q = [1, 7, 32, 64];
        let k = [1, 3, 32, 64];
        let v = [1, 3, 32, 64];
        assert!(validate_attention_shapes(&q, &k, &v).is_err());
    }

    // ----- validate_reshape ----------------------------------------------

    #[test]
    fn reshape_valid() {
        assert!(validate_reshape(&[2, 3, 4], &[6, 4]).is_ok());
        assert!(validate_reshape(&[24], &[2, 3, 4]).is_ok());
    }

    #[test]
    fn reshape_element_mismatch() {
        let err = validate_reshape(&[2, 3], &[2, 4]).unwrap_err();
        assert!(matches!(err, ShapeError::ReshapeElementCount { from_count: 6, to_count: 8, .. }));
    }

    #[test]
    fn reshape_to_scalar() {
        // [1] → [] both have 1 element (product of empty shape is 1)
        assert!(validate_reshape(&[1], &[]).is_ok());
    }

    #[test]
    fn reshape_with_zero_dim() {
        // Zero-element tensors: [0, 3] has 0 elements, [0] also has 0
        assert!(validate_reshape(&[0, 3], &[0]).is_ok());
    }

    // ----- validate_transpose_axes ---------------------------------------

    #[test]
    fn transpose_valid_2d() {
        assert_eq!(validate_transpose_axes(&[3, 4], &[1, 0]).unwrap(), vec![4, 3]);
    }

    #[test]
    fn transpose_valid_4d() {
        assert_eq!(
            validate_transpose_axes(&[1, 8, 32, 64], &[0, 2, 1, 3]).unwrap(),
            vec![1, 32, 8, 64]
        );
    }

    #[test]
    fn transpose_axis_out_of_range() {
        let err = validate_transpose_axes(&[3, 4], &[0, 5]).unwrap_err();
        assert!(matches!(err, ShapeError::TransposeAxisOutOfRange { axis: 5, ndim: 2 }));
    }

    #[test]
    fn transpose_duplicate_axes() {
        let err = validate_transpose_axes(&[3, 4, 5], &[0, 0, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::TransposeNotPermutation { .. }));
    }

    #[test]
    fn transpose_wrong_length() {
        let err = validate_transpose_axes(&[3, 4], &[0]).unwrap_err();
        assert!(matches!(err, ShapeError::TransposeNotPermutation { .. }));
    }

    #[test]
    fn transpose_identity() {
        assert_eq!(validate_transpose_axes(&[2, 3, 4], &[0, 1, 2]).unwrap(), vec![2, 3, 4]);
    }

    // ----- edge cases ----------------------------------------------------

    #[test]
    fn broadcast_high_rank() {
        assert_eq!(broadcast_shape(&[1, 1, 1, 5], &[2, 3, 4, 5]).unwrap(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn matmul_high_rank_batched() {
        // [2, 1, 4, 3] × [2, 5, 3, 6] → broadcast batch [2, 5], out [2, 5, 4, 6]
        assert_eq!(validate_matmul_shapes(&[2, 1, 4, 3], &[2, 5, 3, 6]).unwrap(), vec![2, 5, 4, 6]);
    }

    // =================================================================
    // Production data-validation tests
    // =================================================================

    mod data_validation {
        use super::super::*;
        use proptest::prelude::*;

        // ----- Test helper -----------------------------------------------

        struct TestTensor {
            shape: Vec<usize>,
            data: Option<Vec<f32>>,
            strides: Option<Vec<usize>>,
            alignment: usize,
        }

        impl TestTensor {
            fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
                Self { shape, data: Some(data), strides: None, alignment: 64 }
            }

            fn with_strides(mut self, strides: Vec<usize>) -> Self {
                self.strides = Some(strides);
                self
            }

            fn with_alignment(mut self, alignment: usize) -> Self {
                self.alignment = alignment;
                self
            }

            fn no_data(shape: Vec<usize>) -> Self {
                Self { shape, data: None, strides: None, alignment: 64 }
            }
        }

        impl TensorLike for TestTensor {
            fn shape(&self) -> &[usize] {
                &self.shape
            }
            fn data_f32(&self) -> Option<&[f32]> {
                self.data.as_deref()
            }
            fn strides(&self) -> Option<&[usize]> {
                self.strides.as_deref()
            }
            fn data_alignment(&self) -> usize {
                self.alignment
            }
        }

        fn default_validator() -> TensorValidator {
            TensorValidator::new(ValidationConfig::new())
        }

        // ----- Valid tensors pass ----------------------------------------

        #[test]
        fn valid_tensor_passes() {
            let t = TestTensor::new(vec![2, 3], vec![1.0; 6]);
            assert!(default_validator().validate(&t).is_ok());
        }

        #[test]
        fn scalar_tensor_passes() {
            let t = TestTensor::new(vec![], vec![42.0]);
            assert!(default_validator().validate(&t).is_ok());
        }

        #[test]
        fn valid_tensor_with_strides_passes() {
            let t = TestTensor::new(vec![2, 3], vec![1.0; 6]).with_strides(vec![3, 1]);
            assert!(default_validator().validate(&t).is_ok());
        }

        // ----- Zero-dimension detection ----------------------------------

        #[test]
        fn zero_dimension_first_axis() {
            let t = TestTensor::no_data(vec![0, 3]);
            let err = default_validator().validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::ZeroDimension { axis: 0, .. }));
        }

        #[test]
        fn zero_dimension_middle_axis() {
            let t = TestTensor::no_data(vec![3, 0, 5]);
            let err = default_validator().validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::ZeroDimension { axis: 1, .. }));
        }

        #[test]
        fn zero_dimension_last_axis() {
            let t = TestTensor::no_data(vec![3, 5, 0]);
            let err = default_validator().validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::ZeroDimension { axis: 2, .. }));
        }

        // ----- NaN detection ---------------------------------------------

        #[test]
        fn nan_detection_first_element() {
            let t = TestTensor::new(vec![4], vec![f32::NAN, 1.0, 2.0, 3.0]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(err, TensorValidationError::NanDetected { index: 0 });
        }

        #[test]
        fn nan_detection_middle() {
            let t = TestTensor::new(vec![4], vec![1.0, 2.0, f32::NAN, 3.0]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(err, TensorValidationError::NanDetected { index: 2 });
        }

        #[test]
        fn nan_detection_last() {
            let t = TestTensor::new(vec![3], vec![1.0, 2.0, f32::NAN]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(err, TensorValidationError::NanDetected { index: 2 });
        }

        #[test]
        fn nan_check_disabled_passes() {
            let cfg = ValidationConfig::new().check_nan(false);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![2], vec![f32::NAN, 1.0]);
            assert!(v.validate(&t).is_ok());
        }

        // ----- Inf detection ---------------------------------------------

        #[test]
        fn inf_positive_detected() {
            let t = TestTensor::new(vec![3], vec![1.0, 2.0, f32::INFINITY]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(err, TensorValidationError::InfDetected { index: 2, value: f32::INFINITY });
        }

        #[test]
        fn inf_negative_detected() {
            let t = TestTensor::new(vec![3], vec![1.0, f32::NEG_INFINITY, 2.0]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(
                err,
                TensorValidationError::InfDetected { index: 1, value: f32::NEG_INFINITY }
            );
        }

        #[test]
        fn inf_check_disabled_passes() {
            let cfg = ValidationConfig::new().check_inf(false);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![1], vec![f32::INFINITY]);
            assert!(v.validate(&t).is_ok());
        }

        // ----- Value range -----------------------------------------------

        #[test]
        fn value_above_range() {
            let cfg = ValidationConfig::new().value_range(-1.0, 1.0);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![3], vec![0.0, 0.5, 1.5]);
            let err = v.validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::ValueOutOfRange { index: 2, .. }));
        }

        #[test]
        fn value_below_range() {
            let cfg = ValidationConfig::new().value_range(0.0, 1.0);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![2], vec![-0.1, 0.5]);
            let err = v.validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::ValueOutOfRange { index: 0, .. }));
        }

        #[test]
        fn value_at_bounds_passes() {
            let cfg = ValidationConfig::new().value_range(-1.0, 1.0);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![3], vec![-1.0, 0.0, 1.0]);
            assert!(v.validate(&t).is_ok());
        }

        #[test]
        fn no_range_check_by_default() {
            let t = TestTensor::new(vec![2], vec![-1e30, 1e30]);
            assert!(default_validator().validate(&t).is_ok());
        }

        // ----- Stride consistency ----------------------------------------

        #[test]
        fn stride_consistent_passes() {
            // C-contiguous strides for [4, 3] are [3, 1]
            let t = TestTensor::new(vec![4, 3], vec![0.0; 12]).with_strides(vec![3, 1]);
            assert!(default_validator().validate(&t).is_ok());
        }

        #[test]
        fn stride_inconsistent_detected() {
            // [4, 3] expects strides [3, 1], not [4, 1]
            let t = TestTensor::new(vec![4, 3], vec![0.0; 12]).with_strides(vec![4, 1]);
            let err = default_validator().validate(&t).unwrap_err();
            assert_eq!(
                err,
                TensorValidationError::StrideInconsistent { axis: 0, expected: 3, actual: 4 }
            );
        }

        #[test]
        fn no_strides_skips_check() {
            let t = TestTensor::new(vec![2, 3], vec![0.0; 6]);
            assert!(default_validator().validate(&t).is_ok());
        }

        // ----- Alignment -------------------------------------------------

        #[test]
        fn alignment_satisfied() {
            let cfg = ValidationConfig::new().required_alignment(32);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![2], vec![1.0, 2.0]).with_alignment(64);
            assert!(v.validate(&t).is_ok());
        }

        #[test]
        fn alignment_violation() {
            let cfg = ValidationConfig::new().required_alignment(32);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::new(vec![2], vec![1.0, 2.0]).with_alignment(16);
            let err = v.validate(&t).unwrap_err();
            assert_eq!(err, TensorValidationError::AlignmentViolation { required: 32, actual: 16 });
        }

        #[test]
        fn alignment_default_passes() {
            // Default required_alignment is 1, so any alignment works.
            let t = TestTensor::new(vec![1], vec![1.0]).with_alignment(1);
            assert!(default_validator().validate(&t).is_ok());
        }

        // ----- Total elements / dimensions --------------------------------

        #[test]
        fn total_elements_exceeded() {
            let cfg = ValidationConfig::new().max_total_elements(100);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::no_data(vec![11, 10]); // 110 > 100
            let err = v.validate(&t).unwrap_err();
            assert!(matches!(
                err,
                TensorValidationError::TotalElementsExceeded { total: 110, max: 100 }
            ));
        }

        #[test]
        fn dimensions_exceeded() {
            let cfg = ValidationConfig::new().max_dimensions(3);
            let v = TensorValidator::new(cfg);
            let t = TestTensor::no_data(vec![2, 3, 4, 5]);
            let err = v.validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::DimensionsExceeded { ndim: 4, max: 3 }));
        }

        #[test]
        fn very_large_dimension_overflow() {
            let cfg = ValidationConfig::new();
            let v = TensorValidator::new(cfg);
            let t = TestTensor::no_data(vec![usize::MAX, 2]);
            let err = v.validate(&t).unwrap_err();
            assert!(matches!(err, TensorValidationError::TotalElementsExceeded { .. }));
        }

        // ----- Batch validation ------------------------------------------

        #[test]
        fn batch_all_valid() {
            let tensors = vec![
                TestTensor::new(vec![2], vec![1.0, 2.0]),
                TestTensor::new(vec![3], vec![1.0, 2.0, 3.0]),
            ];
            let results = default_validator().validate_batch(&tensors);
            assert!(results.iter().all(|r| r.is_ok()));
        }

        #[test]
        fn batch_mixed_valid_invalid() {
            let tensors = vec![
                TestTensor::new(vec![2], vec![1.0, 2.0]),
                TestTensor::new(vec![2], vec![f32::NAN, 1.0]),
                TestTensor::new(vec![3], vec![1.0, 2.0, 3.0]),
            ];
            let results = default_validator().validate_batch(&tensors);
            assert!(results[0].is_ok());
            assert!(results[1].is_err());
            assert!(results[2].is_ok());
        }

        #[test]
        fn batch_all_invalid() {
            let tensors = vec![
                TestTensor::new(vec![1], vec![f32::NAN]),
                TestTensor::new(vec![1], vec![f32::INFINITY]),
            ];
            let results = default_validator().validate_batch(&tensors);
            assert!(results.iter().all(|r| r.is_err()));
        }

        #[test]
        fn batch_empty() {
            let tensors: Vec<TestTensor> = vec![];
            let results = default_validator().validate_batch(&tensors);
            assert!(results.is_empty());
        }

        // ----- Config builder --------------------------------------------

        #[test]
        fn config_builder_defaults() {
            let cfg = ValidationConfig::new();
            assert_eq!(cfg.max_total_elements, 1 << 30);
            assert_eq!(cfg.max_dimensions, 8);
            assert!(cfg.check_nan);
            assert!(cfg.check_inf);
            assert!(cfg.value_range.is_none());
            assert_eq!(cfg.required_alignment, 1);
        }

        #[test]
        fn config_builder_custom() {
            let cfg = ValidationConfig::new()
                .max_total_elements(500)
                .max_dimensions(4)
                .check_nan(false)
                .check_inf(false)
                .value_range(-2.0, 2.0)
                .required_alignment(16);
            assert_eq!(cfg.max_total_elements, 500);
            assert_eq!(cfg.max_dimensions, 4);
            assert!(!cfg.check_nan);
            assert!(!cfg.check_inf);
            assert_eq!(cfg.value_range, Some((-2.0, 2.0)));
            assert_eq!(cfg.required_alignment, 16);
        }

        // ----- Miscellaneous edge cases ----------------------------------

        #[test]
        fn no_data_skips_value_checks() {
            let t = TestTensor::no_data(vec![2, 3]);
            assert!(default_validator().validate(&t).is_ok());
        }

        #[test]
        fn c_contiguous_strides_helper() {
            assert_eq!(c_contiguous_strides(&[4, 3, 2]), vec![6, 2, 1]);
            assert_eq!(c_contiguous_strides(&[5]), vec![1]);
            assert!(c_contiguous_strides(&[]).is_empty());
        }

        #[test]
        fn single_element_tensor_passes() {
            let t = TestTensor::new(vec![1, 1, 1], vec![0.5]).with_strides(vec![1, 1, 1]);
            assert!(default_validator().validate(&t).is_ok());
        }

        // ----- Property tests --------------------------------------------

        proptest! {
            #[test]
            fn prop_valid_tensors_always_pass(len in 1..64usize) {
                let data: Vec<f32> =
                    (0..len).map(|i| (i as f32) * 0.01).collect();
                let t = TestTensor::new(vec![len], data);
                prop_assert!(default_validator().validate(&t).is_ok());
            }

            #[test]
            fn prop_nan_always_fails(len in 1..64usize) {
                let pos = len / 2;
                let mut data = vec![1.0f32; len];
                data[pos] = f32::NAN;
                let t = TestTensor::new(vec![len], data);
                let result = default_validator().validate(&t);
                let is_nan_err = matches!(
                    result,
                    Err(TensorValidationError::NanDetected { .. })
                );
                prop_assert!(is_nan_err, "expected NanDetected error");
            }

            #[test]
            fn prop_inf_always_fails(len in 1..64usize) {
                let pos = len / 2;
                let mut data = vec![1.0f32; len];
                data[pos] = f32::INFINITY;
                let t = TestTensor::new(vec![len], data);
                let result = default_validator().validate(&t);
                let is_inf_err = matches!(
                    result,
                    Err(TensorValidationError::InfDetected { .. })
                );
                prop_assert!(is_inf_err, "expected InfDetected error");
            }
        }
    }
}
