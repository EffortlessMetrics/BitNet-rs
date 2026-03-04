//! Tensor memory layout and stride calculation.
//!
//! Row-major/column-major layout, stride computation, contiguity
//! checks, and offset calculation for multi-dimensional tensors.

/// Memory layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutOrder {
    /// Row-major (C-style): last dimension varies fastest.
    RowMajor,
    /// Column-major (Fortran-style): first dimension varies fastest.
    ColMajor,
}

/// Tensor memory layout with shape, strides, and offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub order: LayoutOrder,
}

impl TensorLayout {
    /// Create a contiguous layout with the given shape and order.
    pub fn contiguous(shape: Vec<usize>, order: LayoutOrder) -> Self {
        let strides = compute_strides(&shape, order);
        Self { shape, strides, offset: 0, order }
    }

    /// Create a row-major contiguous layout.
    pub fn row_major(shape: Vec<usize>) -> Self {
        Self::contiguous(shape, LayoutOrder::RowMajor)
    }

    /// Create a column-major contiguous layout.
    pub fn col_major(shape: Vec<usize>) -> Self {
        Self::contiguous(shape, LayoutOrder::ColMajor)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the layout is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape, self.order);
        self.strides == expected && self.offset == 0
    }

    /// Compute the linear offset for a multi-dimensional index.
    pub fn linear_offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return None;
            }
        }
        let offset: usize = indices.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum();
        Some(self.offset + offset)
    }

    /// Transpose the last two dimensions.
    pub fn transpose(&self) -> Option<Self> {
        if self.ndim() < 2 {
            return None;
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        let n = new_shape.len();
        new_shape.swap(n - 1, n - 2);
        new_strides.swap(n - 1, n - 2);
        Some(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            order: self.order,
        })
    }

    /// Create a view with an offset (slice at first dimension).
    pub fn slice_first(&self, start: usize, len: usize) -> Option<Self> {
        if self.shape.is_empty() || start + len > self.shape[0] {
            return None;
        }
        let mut new_shape = self.shape.clone();
        new_shape[0] = len;
        Some(Self {
            shape: new_shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[0],
            order: self.order,
        })
    }

    /// Size in bytes assuming f32 elements.
    pub fn size_bytes_f32(&self) -> usize {
        self.numel() * 4
    }
}

/// Compute strides for a contiguous layout.
pub fn compute_strides(shape: &[usize], order: LayoutOrder) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0usize; shape.len()];
    match order {
        LayoutOrder::RowMajor => {
            strides[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        LayoutOrder::ColMajor => {
            strides[0] = 1;
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_strides() {
        let layout = TensorLayout::row_major(vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_col_major_strides() {
        let layout = TensorLayout::col_major(vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_numel() {
        let layout = TensorLayout::row_major(vec![2, 3, 4]);
        assert_eq!(layout.numel(), 24);
    }

    #[test]
    fn test_contiguous() {
        let layout = TensorLayout::row_major(vec![3, 4]);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn test_linear_offset() {
        let layout = TensorLayout::row_major(vec![2, 3]);
        assert_eq!(layout.linear_offset(&[0, 0]), Some(0));
        assert_eq!(layout.linear_offset(&[0, 2]), Some(2));
        assert_eq!(layout.linear_offset(&[1, 0]), Some(3));
        assert_eq!(layout.linear_offset(&[1, 2]), Some(5));
    }

    #[test]
    fn test_linear_offset_bounds() {
        let layout = TensorLayout::row_major(vec![2, 3]);
        assert!(layout.linear_offset(&[2, 0]).is_none());
        assert!(layout.linear_offset(&[0]).is_none());
    }

    #[test]
    fn test_transpose() {
        let layout = TensorLayout::row_major(vec![3, 4]);
        let transposed = layout.transpose().unwrap();
        assert_eq!(transposed.shape, vec![4, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_slice_first() {
        let layout = TensorLayout::row_major(vec![10, 4]);
        let sliced = layout.slice_first(2, 3).unwrap();
        assert_eq!(sliced.shape, vec![3, 4]);
        assert_eq!(sliced.offset, 8); // 2 * stride[0] = 2 * 4
    }

    #[test]
    fn test_slice_bounds() {
        let layout = TensorLayout::row_major(vec![5, 3]);
        assert!(layout.slice_first(3, 5).is_none());
    }

    #[test]
    fn test_size_bytes() {
        let layout = TensorLayout::row_major(vec![2, 3]);
        assert_eq!(layout.size_bytes_f32(), 24); // 6 * 4
    }

    #[test]
    fn test_1d_layout() {
        let layout = TensorLayout::row_major(vec![10]);
        assert_eq!(layout.strides, vec![1]);
        assert_eq!(layout.linear_offset(&[5]), Some(5));
    }

    #[test]
    fn test_col_major_offset() {
        let layout = TensorLayout::col_major(vec![3, 4]);
        assert_eq!(layout.linear_offset(&[1, 0]), Some(1));
        assert_eq!(layout.linear_offset(&[0, 1]), Some(3));
    }
}
