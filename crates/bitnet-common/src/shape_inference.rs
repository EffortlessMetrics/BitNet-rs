//! Tensor shape inference and broadcasting.
//!
//! Shape compatibility checking, broadcasting rules, and
//! output shape computation for common operations.

/// A tensor shape (list of dimension sizes).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    pub fn scalar() -> Self {
        Self(vec![])
    }

    pub fn vector(len: usize) -> Self {
        Self(vec![len])
    }

    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self(vec![rows, cols])
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Get dimension size, supporting negative indexing.
    pub fn dim(&self, idx: isize) -> Option<usize> {
        let i = if idx < 0 { (self.ndim() as isize + idx) as usize } else { idx as usize };
        self.0.get(i).copied()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Check if two shapes are broadcast-compatible (NumPy rules).
pub fn broadcast_compatible(a: &Shape, b: &Shape) -> bool {
    broadcast_shape(a, b).is_some()
}

/// Compute the broadcast output shape. Returns None if incompatible.
pub fn broadcast_shape(a: &Shape, b: &Shape) -> Option<Shape> {
    let max_ndim = a.ndim().max(b.ndim());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < a.ndim() { a.0[a.ndim() - 1 - i] } else { 1 };
        let db = if i < b.ndim() { b.0[b.ndim() - 1 - i] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return None;
        }
    }

    result.reverse();
    Some(Shape(result))
}

/// Compute the output shape for matrix multiplication (A @ B).
/// A: [..., M, K], B: [..., K, N] -> [..., M, N]
pub fn matmul_shape(a: &Shape, b: &Shape) -> Option<Shape> {
    let dims = matmul_srp::extract_dims(a, b)?;
    let batch = matmul_srp::broadcast_batch_dims(a, b)?;
    Some(matmul_srp::build_output_shape(batch, dims.m, dims.n))
}

mod matmul_srp {
    use super::{Shape, broadcast_shape};

    pub(super) struct MatmulDims {
        pub(super) m: usize,
        pub(super) n: usize,
    }

    pub(super) fn extract_dims(a: &Shape, b: &Shape) -> Option<MatmulDims> {
        if a.ndim() < 2 || b.ndim() < 2 {
            return None;
        }

        let m = a.0[a.ndim() - 2];
        let k1 = a.0[a.ndim() - 1];
        let k2 = b.0[b.ndim() - 2];
        let n = b.0[b.ndim() - 1];
        if k1 != k2 {
            return None;
        }

        Some(MatmulDims { m, n })
    }

    pub(super) fn broadcast_batch_dims(a: &Shape, b: &Shape) -> Option<Vec<usize>> {
        let a_batch = Shape(a.0[..a.ndim() - 2].to_vec());
        let b_batch = Shape(b.0[..b.ndim() - 2].to_vec());

        if a_batch.ndim() == 0 && b_batch.ndim() == 0 {
            return Some(vec![]);
        }

        broadcast_shape(&a_batch, &b_batch).map(|shape| shape.0)
    }

    pub(super) fn build_output_shape(mut batch: Vec<usize>, m: usize, n: usize) -> Shape {
        batch.push(m);
        batch.push(n);
        Shape(batch)
    }
}

/// Compute output shape for a transpose of the last two dimensions.
pub fn transpose_shape(s: &Shape) -> Option<Shape> {
    if s.ndim() < 2 {
        return None;
    }
    let mut result = s.0.clone();
    let n = result.len();
    result.swap(n - 1, n - 2);
    Some(Shape(result))
}

/// Check if a reshape is valid (same number of elements).
pub fn reshape_valid(from: &Shape, to: &Shape) -> bool {
    from.numel() == to.numel()
}

/// Compute shape after concatenation along an axis.
pub fn concat_shape(shapes: &[Shape], axis: usize) -> Option<Shape> {
    if shapes.is_empty() {
        return None;
    }
    let ndim = shapes[0].ndim();
    if axis >= ndim {
        return None;
    }
    // All shapes must have same ndim and match on non-concat dims
    for s in &shapes[1..] {
        if s.ndim() != ndim {
            return None;
        }
        for (i, (&d1, &d2)) in shapes[0].0.iter().zip(s.0.iter()).enumerate() {
            if i != axis && d1 != d2 {
                return None;
            }
        }
    }
    let mut result = shapes[0].0.clone();
    result[axis] = shapes.iter().map(|s| s.0[axis]).sum();
    Some(Shape(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::matrix(3, 4);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.numel(), 12);
        assert!(!s.is_scalar());
        assert_eq!(s.dim(0), Some(3));
        assert_eq!(s.dim(-1), Some(4));
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn test_broadcast_same() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![3, 4]);
        assert_eq!(broadcast_shape(&a, &b), Some(Shape::new(vec![3, 4])));
    }

    #[test]
    fn test_broadcast_expand() {
        let a = Shape::new(vec![3, 1]);
        let b = Shape::new(vec![1, 4]);
        assert_eq!(broadcast_shape(&a, &b), Some(Shape::new(vec![3, 4])));
    }

    #[test]
    fn test_broadcast_incompatible() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![3, 5]);
        assert!(broadcast_shape(&a, &b).is_none());
    }

    #[test]
    fn test_broadcast_different_ndim() {
        let a = Shape::new(vec![2, 3, 4]);
        let b = Shape::new(vec![4]);
        assert_eq!(broadcast_shape(&a, &b), Some(Shape::new(vec![2, 3, 4])));
    }

    #[test]
    fn test_matmul_shape() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![4, 5]);
        assert_eq!(matmul_shape(&a, &b), Some(Shape::new(vec![3, 5])));
    }

    #[test]
    fn test_matmul_batch() {
        let a = Shape::new(vec![2, 3, 4]);
        let b = Shape::new(vec![2, 4, 5]);
        assert_eq!(matmul_shape(&a, &b), Some(Shape::new(vec![2, 3, 5])));
    }

    #[test]
    fn test_matmul_batch_broadcast() {
        let a = Shape::new(vec![2, 1, 3, 4]);
        let b = Shape::new(vec![1, 7, 4, 5]);
        assert_eq!(matmul_shape(&a, &b), Some(Shape::new(vec![2, 7, 3, 5])));
    }

    #[test]
    fn test_matmul_incompatible() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![5, 6]);
        assert!(matmul_shape(&a, &b).is_none());
    }

    #[test]
    fn test_transpose() {
        let s = Shape::new(vec![3, 4]);
        assert_eq!(transpose_shape(&s), Some(Shape::new(vec![4, 3])));
    }

    #[test]
    fn test_reshape_valid() {
        let from = Shape::new(vec![2, 3, 4]);
        let to = Shape::new(vec![6, 4]);
        assert!(reshape_valid(&from, &to));
    }

    #[test]
    fn test_concat_shape() {
        let shapes = vec![Shape::new(vec![2, 3]), Shape::new(vec![2, 5])];
        assert_eq!(concat_shape(&shapes, 1), Some(Shape::new(vec![2, 8])));
    }

    #[test]
    fn test_shape_display() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(format!("{s}"), "[2, 3, 4]");
    }
}
