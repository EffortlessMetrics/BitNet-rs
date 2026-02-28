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
}
