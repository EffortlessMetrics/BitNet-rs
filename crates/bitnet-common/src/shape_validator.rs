//! Tensor shape validation utilities.
//!
//! Validates tensor shapes against expected constraints before
//! computation, providing clear error messages.

use std::fmt;

/// Error from shape validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError {
    pub context: String,
    pub expected: String,
    pub actual: String,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Shape mismatch in {}: expected {}, got {}",
            self.context, self.expected, self.actual
        )
    }
}

impl std::error::Error for ShapeError {}

/// Validate that two shapes are identical.
pub fn assert_shape_eq(ctx: &str, actual: &[usize], expected: &[usize]) -> Result<(), ShapeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        })
    }
}

/// Validate tensor rank (number of dimensions).
pub fn assert_rank(ctx: &str, shape: &[usize], expected_rank: usize) -> Result<(), ShapeError> {
    if shape.len() == expected_rank {
        Ok(())
    } else {
        Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("rank {expected_rank}"),
            actual: format!("rank {}", shape.len()),
        })
    }
}

/// Validate that a specific dimension has the expected size.
pub fn assert_dim(
    ctx: &str,
    shape: &[usize],
    dim: usize,
    expected_size: usize,
) -> Result<(), ShapeError> {
    match shape.get(dim) {
        Some(&s) if s == expected_size => Ok(()),
        Some(&s) => Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("dim[{dim}]={expected_size}"),
            actual: format!("dim[{dim}]={s}"),
        }),
        None => Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("dim[{dim}]={expected_size}"),
            actual: format!("rank {} (no dim {dim})", shape.len()),
        }),
    }
}

/// Validate shapes are compatible for matrix multiplication (A @ B).
pub fn assert_matmul_compat(
    ctx: &str,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(), ShapeError> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(ShapeError {
            context: ctx.to_string(),
            expected: "both tensors rank >= 2".to_string(),
            actual: format!("ranks {} and {}", a_shape.len(), b_shape.len()),
        });
    }
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = b_shape[b_shape.len() - 2];
    if a_cols != b_rows {
        return Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("A cols ({a_cols}) == B rows ({b_rows})"),
            actual: format!("A={a_shape:?}, B={b_shape:?}"),
        });
    }
    Ok(())
}

/// Validate shapes are broadcastable (element-wise ops).
pub fn assert_broadcastable(
    ctx: &str,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(), ShapeError> {
    let max_rank = a_shape.len().max(b_shape.len());
    for i in 0..max_rank {
        let a_dim = if i < a_shape.len() { a_shape[a_shape.len() - 1 - i] } else { 1 };
        let b_dim = if i < b_shape.len() { b_shape[b_shape.len() - 1 - i] } else { 1 };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(ShapeError {
                context: ctx.to_string(),
                expected: "broadcastable shapes".to_string(),
                actual: format!("{a_shape:?} vs {b_shape:?}"),
            });
        }
    }
    Ok(())
}

/// Validate total element count matches expected.
pub fn assert_element_count(ctx: &str, shape: &[usize], expected: usize) -> Result<(), ShapeError> {
    let actual: usize = shape.iter().product();
    if actual == expected {
        Ok(())
    } else {
        Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("{expected} elements"),
            actual: format!("{actual} elements (shape={shape:?})"),
        })
    }
}

/// Validate hidden size is divisible by number of heads.
pub fn assert_head_divisible(
    ctx: &str,
    hidden_size: usize,
    num_heads: usize,
) -> Result<(), ShapeError> {
    if num_heads == 0 {
        return Err(ShapeError {
            context: ctx.to_string(),
            expected: "num_heads > 0".to_string(),
            actual: "num_heads = 0".to_string(),
        });
    }
    if !hidden_size.is_multiple_of(num_heads) {
        return Err(ShapeError {
            context: ctx.to_string(),
            expected: format!("hidden_size ({hidden_size}) divisible by num_heads ({num_heads})"),
            actual: format!("remainder = {}", hidden_size % num_heads),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_eq_pass() {
        assert!(assert_shape_eq("test", &[2, 3], &[2, 3]).is_ok());
    }

    #[test]
    fn test_shape_eq_fail() {
        let err = assert_shape_eq("linear", &[2, 3], &[2, 4]).unwrap_err();
        assert!(err.to_string().contains("linear"));
    }

    #[test]
    fn test_rank_pass() {
        assert!(assert_rank("weight", &[768, 768], 2).is_ok());
    }

    #[test]
    fn test_rank_fail() {
        let err = assert_rank("bias", &[768], 2).unwrap_err();
        assert!(err.to_string().contains("rank 2"));
    }

    #[test]
    fn test_dim_pass() {
        assert!(assert_dim("embed", &[32000, 2560], 0, 32000).is_ok());
    }

    #[test]
    fn test_dim_fail_value() {
        let err = assert_dim("embed", &[32000, 2560], 1, 4096).unwrap_err();
        assert!(err.to_string().contains("dim[1]"));
    }

    #[test]
    fn test_dim_fail_missing() {
        let err = assert_dim("test", &[10], 2, 5).unwrap_err();
        assert!(err.to_string().contains("rank 1"));
    }

    #[test]
    fn test_matmul_compat_pass() {
        assert!(assert_matmul_compat("mm", &[2, 3], &[3, 4]).is_ok());
    }

    #[test]
    fn test_matmul_compat_fail() {
        let err = assert_matmul_compat("mm", &[2, 3], &[4, 5]).unwrap_err();
        assert!(err.to_string().contains("cols"));
    }

    #[test]
    fn test_matmul_low_rank() {
        let err = assert_matmul_compat("mm", &[5], &[5, 3]).unwrap_err();
        assert!(err.to_string().contains("rank"));
    }

    #[test]
    fn test_broadcastable_same() {
        assert!(assert_broadcastable("add", &[2, 3], &[2, 3]).is_ok());
    }

    #[test]
    fn test_broadcastable_broadcast() {
        assert!(assert_broadcastable("add", &[2, 3], &[1, 3]).is_ok());
        assert!(assert_broadcastable("add", &[2, 3], &[3]).is_ok());
    }

    #[test]
    fn test_broadcastable_fail() {
        let err = assert_broadcastable("add", &[2, 3], &[2, 4]).unwrap_err();
        assert!(err.to_string().contains("broadcastable"));
    }

    #[test]
    fn test_element_count_pass() {
        assert!(assert_element_count("test", &[2, 3, 4], 24).is_ok());
    }

    #[test]
    fn test_element_count_fail() {
        let err = assert_element_count("test", &[2, 3], 10).unwrap_err();
        assert!(err.to_string().contains("6 elements"));
    }

    #[test]
    fn test_head_divisible_pass() {
        assert!(assert_head_divisible("attn", 5120, 40).is_ok());
        assert!(assert_head_divisible("attn", 4096, 32).is_ok());
    }

    #[test]
    fn test_head_divisible_fail() {
        let err = assert_head_divisible("attn", 100, 7).unwrap_err();
        assert!(err.to_string().contains("remainder"));
    }

    #[test]
    fn test_head_divisible_zero() {
        let err = assert_head_divisible("attn", 100, 0).unwrap_err();
        assert!(err.to_string().contains("num_heads > 0"));
    }

    #[test]
    fn test_shape_error_display() {
        let err = ShapeError {
            context: "test_op".to_string(),
            expected: "rank 2".to_string(),
            actual: "rank 3".to_string(),
        };
        assert_eq!(err.to_string(), "Shape mismatch in test_op: expected rank 2, got rank 3");
    }

    #[test]
    fn test_matmul_3d() {
        assert!(assert_matmul_compat("batched", &[4, 2, 3], &[4, 3, 5]).is_ok());
    }

    #[test]
    fn test_empty_shape() {
        assert!(assert_element_count("empty", &[], 1).is_ok());
    }

    #[test]
    fn test_broadcastable_scalar() {
        assert!(assert_broadcastable("scale", &[2, 3], &[]).is_ok());
    }
}
