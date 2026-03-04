//! Tensor shape validation utilities.
//!
//! Validate shapes for common neural network operations.

/// Shape validation error.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeError {
    pub op: String,
    pub expected: String,
    pub got: String,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: expected {}, got {}", self.op, self.expected, self.got)
    }
}

/// Validate matmul compatibility: \[M,K\] × \[K,N\] → \[M,N\].
pub fn validate_matmul(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if a.len() != 2 || b.len() != 2 {
        return Err(ShapeError {
            op: "matmul".into(),
            expected: "2D tensors".into(),
            got: format!("{}D × {}D", a.len(), b.len()),
        });
    }
    if a[1] != b[0] {
        return Err(ShapeError {
            op: "matmul".into(),
            expected: format!("a[1]={} == b[0]", a[1]),
            got: format!("b[0]={}", b[0]),
        });
    }
    Ok(vec![a[0], b[1]])
}

/// Validate batched matmul: \[B,M,K\] × \[B,K,N\] → \[B,M,N\].
pub fn validate_batched_matmul(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if a.len() != 3 || b.len() != 3 {
        return Err(ShapeError {
            op: "batched_matmul".into(),
            expected: "3D tensors".into(),
            got: format!("{}D × {}D", a.len(), b.len()),
        });
    }
    if a[0] != b[0] {
        return Err(ShapeError {
            op: "batched_matmul".into(),
            expected: format!("batch={}", a[0]),
            got: format!("batch={}", b[0]),
        });
    }
    if a[2] != b[1] {
        return Err(ShapeError {
            op: "batched_matmul".into(),
            expected: format!("a[2]={} == b[1]", a[2]),
            got: format!("b[1]={}", b[1]),
        });
    }
    Ok(vec![a[0], a[1], b[2]])
}

/// Validate element-wise ops (shapes must match).
pub fn validate_elementwise(a: &[usize], b: &[usize]) -> Result<(), ShapeError> {
    if a != b {
        return Err(ShapeError {
            op: "elementwise".into(),
            expected: format!("{a:?}"),
            got: format!("{b:?}"),
        });
    }
    Ok(())
}

/// Validate layer norm input: last dim must equal normalized_shape.
pub fn validate_layer_norm(input: &[usize], norm_size: usize) -> Result<(), ShapeError> {
    if input.is_empty() {
        return Err(ShapeError {
            op: "layer_norm".into(),
            expected: "non-empty shape".into(),
            got: "empty".into(),
        });
    }
    if *input.last().unwrap() != norm_size {
        return Err(ShapeError {
            op: "layer_norm".into(),
            expected: format!("last_dim={norm_size}"),
            got: format!("last_dim={}", input.last().unwrap()),
        });
    }
    Ok(())
}

/// Validate embedding lookup: indices in [0, vocab_size).
pub fn validate_embedding(vocab_size: usize, max_index: usize) -> Result<(), ShapeError> {
    if max_index >= vocab_size {
        return Err(ShapeError {
            op: "embedding".into(),
            expected: format!("index < {vocab_size}"),
            got: format!("index={max_index}"),
        });
    }
    Ok(())
}

/// Validate reshape: total elements must match.
pub fn validate_reshape(old: &[usize], new: &[usize]) -> Result<(), ShapeError> {
    let old_total: usize = old.iter().product();
    let new_total: usize = new.iter().product();
    if old_total != new_total {
        return Err(ShapeError {
            op: "reshape".into(),
            expected: format!("{old_total} elements"),
            got: format!("{new_total} elements"),
        });
    }
    Ok(())
}

/// Asserts that a shape's rank matches the expected rank.
pub fn assert_rank(op: &str, shape: &[usize], expected: usize) -> Result<(), ShapeError> {
    if shape.len() != expected {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("rank {expected}"),
            got: format!("rank {}", shape.len()),
        });
    }
    Ok(())
}

/// Asserts that two shapes are exactly equal.
pub fn assert_shape_eq(op: &str, expected: &[usize], got: &[usize]) -> Result<(), ShapeError> {
    if expected != got {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("{expected:?}"),
            got: format!("{got:?}"),
        });
    }
    Ok(())
}

/// Asserts that a specific dimension has the expected size.
pub fn assert_dim(
    op: &str,
    shape: &[usize],
    dim: usize,
    expected: usize,
) -> Result<(), ShapeError> {
    if dim >= shape.len() {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("rank > {dim}"),
            got: format!("rank {}", shape.len()),
        });
    }
    if shape[dim] != expected {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("dim {dim} = {expected}"),
            got: format!("dim {dim} = {}", shape[dim]),
        });
    }
    Ok(())
}

/// Asserts that the total element count matches the expected count.
pub fn assert_element_count(op: &str, shape: &[usize], expected: usize) -> Result<(), ShapeError> {
    let count: usize = shape.iter().product();
    if count != expected {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("{expected} elements"),
            got: format!("{count} elements"),
        });
    }
    Ok(())
}

/// Asserts that a dimension is divisible by the number of heads.
pub fn assert_head_divisible(
    op: &str,
    dim_size: usize,
    num_heads: usize,
) -> Result<(), ShapeError> {
    if !dim_size.is_multiple_of(num_heads) {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("divisible by {num_heads}"),
            got: format!("{dim_size}"),
        });
    }
    Ok(())
}

/// Asserts that two shapes are matmul compatible (inner dimensions match).
pub fn assert_matmul_compat(op: &str, a: &[usize], b: &[usize]) -> Result<(), ShapeError> {
    if a.is_empty() || b.is_empty() {
        return Err(ShapeError {
            op: op.to_string(),
            expected: "non-empty shapes".to_string(),
            got: "empty shape".to_string(),
        });
    }
    let a_inner = *a.last().unwrap();
    let b_inner = if b.len() > 1 { b[b.len() - 2] } else { b[0] };
    if a_inner != b_inner {
        return Err(ShapeError {
            op: op.to_string(),
            expected: format!("inner dim match ({a_inner})"),
            got: format!("{b_inner}"),
        });
    }
    Ok(())
}

/// Asserts that two shapes are broadcastable.
pub fn assert_broadcastable(op: &str, a: &[usize], b: &[usize]) -> Result<(), ShapeError> {
    let max_len = std::cmp::max(a.len(), b.len());
    let mut a_pad = vec![1; max_len - a.len()];
    a_pad.extend_from_slice(a);
    let mut b_pad = vec![1; max_len - b.len()];
    b_pad.extend_from_slice(b);

    for i in 0..max_len {
        if a_pad[i] != b_pad[i] && a_pad[i] != 1 && b_pad[i] != 1 {
            return Err(ShapeError {
                op: op.to_string(),
                expected: "broadcastable shapes".to_string(),
                got: format!("{a:?} and {b:?}"),
            });
        }
    }
    Ok(())
}

/// Validate attention shapes: Q\[B,H,S,D\], K\[B,H,S,D\], V\[B,H,S,D\].
pub fn validate_attention(q: &[usize], k: &[usize], v: &[usize]) -> Result<Vec<usize>, ShapeError> {
    if q.len() != 4 || k.len() != 4 || v.len() != 4 {
        return Err(ShapeError {
            op: "attention".into(),
            expected: "4D tensors [B,H,S,D]".into(),
            got: format!("{}D, {}D, {}D", q.len(), k.len(), v.len()),
        });
    }
    // Q and K must have same head_dim for dot product
    if q[3] != k[3] {
        return Err(ShapeError {
            op: "attention".into(),
            expected: format!("q_head_dim={}", q[3]),
            got: format!("k_head_dim={}", k[3]),
        });
    }
    // K and V must have same seq_len
    if k[2] != v[2] {
        return Err(ShapeError {
            op: "attention".into(),
            expected: format!("k_seq={}", k[2]),
            got: format!("v_seq={}", v[2]),
        });
    }
    Ok(vec![q[0], q[1], q[2], v[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_valid() {
        let r = validate_matmul(&[3, 4], &[4, 5]).unwrap();
        assert_eq!(r, vec![3, 5]);
    }

    #[test]
    fn test_matmul_mismatch() {
        assert!(validate_matmul(&[3, 4], &[5, 6]).is_err());
    }

    #[test]
    fn test_matmul_wrong_dims() {
        assert!(validate_matmul(&[3], &[3, 4]).is_err());
    }

    #[test]
    fn test_batched_matmul_valid() {
        let r = validate_batched_matmul(&[2, 3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(r, vec![2, 3, 5]);
    }

    #[test]
    fn test_batched_matmul_batch_mismatch() {
        assert!(validate_batched_matmul(&[2, 3, 4], &[3, 4, 5]).is_err());
    }

    #[test]
    fn test_elementwise_ok() {
        assert!(validate_elementwise(&[2, 3], &[2, 3]).is_ok());
    }

    #[test]
    fn test_elementwise_mismatch() {
        assert!(validate_elementwise(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn test_layer_norm_ok() {
        assert!(validate_layer_norm(&[2, 512], 512).is_ok());
    }

    #[test]
    fn test_layer_norm_mismatch() {
        assert!(validate_layer_norm(&[2, 512], 256).is_err());
    }

    #[test]
    fn test_embedding_ok() {
        assert!(validate_embedding(32000, 31999).is_ok());
    }

    #[test]
    fn test_embedding_oob() {
        assert!(validate_embedding(32000, 32000).is_err());
    }

    #[test]
    fn test_reshape_ok() {
        assert!(validate_reshape(&[2, 3, 4], &[24]).is_ok());
    }

    #[test]
    fn test_reshape_mismatch() {
        assert!(validate_reshape(&[2, 3], &[7]).is_err());
    }

    #[test]
    fn test_attention_valid() {
        let r = validate_attention(&[1, 8, 32, 64], &[1, 8, 32, 64], &[1, 8, 32, 64]).unwrap();
        assert_eq!(r, vec![1, 8, 32, 64]);
    }

    #[test]
    fn test_attention_head_dim_mismatch() {
        assert!(validate_attention(&[1, 8, 32, 64], &[1, 8, 32, 128], &[1, 8, 32, 64]).is_err());
    }

    #[test]
    fn test_shape_error_display() {
        let e = ShapeError { op: "test".into(), expected: "A".into(), got: "B".into() };
        assert_eq!(format!("{e}"), "test: expected A, got B");
    }
}
