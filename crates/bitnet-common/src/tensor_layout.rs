//! Tensor layout analysis.
//!
//! Determines memory layout, strides, and alignment for tensors.

/// Memory layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutOrder {
    RowMajor, // C-style, last dim varies fastest
    ColMajor, // Fortran-style, first dim varies fastest
}

/// Tensor layout descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub order: LayoutOrder,
    pub element_size: usize,
}

impl TensorLayout {
    /// Create a contiguous row-major layout.
    pub fn contiguous(shape: &[usize], element_size: usize) -> Self {
        let strides = compute_strides(shape, LayoutOrder::RowMajor);
        Self { shape: shape.to_vec(), strides, order: LayoutOrder::RowMajor, element_size }
    }

    /// Create a column-major layout.
    pub fn col_major(shape: &[usize], element_size: usize) -> Self {
        let strides = compute_strides(shape, LayoutOrder::ColMajor);
        Self { shape: shape.to_vec(), strides, order: LayoutOrder::ColMajor, element_size }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size.
    pub fn byte_size(&self) -> usize {
        self.numel() * self.element_size
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Whether the layout is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides(&self.shape, self.order);
        self.strides == expected
    }

    /// Byte offset for a given multi-dimensional index.
    pub fn offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
        }
        let elem_offset: usize =
            indices.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum();
        Some(elem_offset * self.element_size)
    }

    /// Transpose: swap two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Option<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return None;
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Some(Self { shape, strides, order: self.order, element_size: self.element_size })
    }

    /// Check if dimensions are aligned to a given boundary.
    pub fn is_aligned(&self, alignment: usize) -> bool {
        if alignment == 0 {
            return false;
        }
        self.shape.last().is_some_and(|&last| last.is_multiple_of(alignment))
    }

    /// Reshape (only valid for contiguous tensors).
    pub fn reshape(&self, new_shape: &[usize]) -> Option<Self> {
        if !self.is_contiguous() {
            return None;
        }
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return None;
        }
        Some(Self::contiguous(new_shape, self.element_size))
    }
}

/// Compute strides for a given shape and order.
pub fn compute_strides(shape: &[usize], order: LayoutOrder) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    match order {
        LayoutOrder::RowMajor => {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        LayoutOrder::ColMajor => {
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

/// Check if a shape is broadcastable with another.
pub fn broadcastable(a: &[usize], b: &[usize]) -> bool {
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_layout() {
        let l = TensorLayout::contiguous(&[2, 3, 4], 4);
        assert_eq!(l.strides, vec![12, 4, 1]);
        assert_eq!(l.numel(), 24);
        assert_eq!(l.byte_size(), 96);
    }

    #[test]
    fn test_col_major() {
        let l = TensorLayout::col_major(&[2, 3, 4], 4);
        assert_eq!(l.strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_is_contiguous() {
        let l = TensorLayout::contiguous(&[2, 3], 4);
        assert!(l.is_contiguous());
    }

    #[test]
    fn test_offset() {
        let l = TensorLayout::contiguous(&[2, 3], 4);
        assert_eq!(l.offset(&[0, 0]), Some(0));
        assert_eq!(l.offset(&[1, 2]), Some(20)); // (1*3 + 2) * 4 = 20
    }

    #[test]
    fn test_offset_out_of_bounds() {
        let l = TensorLayout::contiguous(&[2, 3], 4);
        assert!(l.offset(&[2, 0]).is_none());
        assert!(l.offset(&[0, 3]).is_none());
    }

    #[test]
    fn test_transpose() {
        let l = TensorLayout::contiguous(&[2, 3, 4], 4);
        let t = l.transpose(0, 2).unwrap();
        assert_eq!(t.shape, vec![4, 3, 2]);
    }

    #[test]
    fn test_reshape() {
        let l = TensorLayout::contiguous(&[2, 3], 4);
        let r = l.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.numel(), 6);
    }

    #[test]
    fn test_reshape_mismatch() {
        let l = TensorLayout::contiguous(&[2, 3], 4);
        assert!(l.reshape(&[4, 4]).is_none());
    }

    #[test]
    fn test_ndim() {
        let l = TensorLayout::contiguous(&[2, 3, 4], 4);
        assert_eq!(l.ndim(), 3);
    }

    #[test]
    fn test_alignment() {
        let l = TensorLayout::contiguous(&[2, 32], 4);
        assert!(l.is_aligned(16));
        assert!(!l.is_aligned(64));
    }

    #[test]
    fn test_broadcastable() {
        assert!(broadcastable(&[1, 3], &[2, 3]));
        assert!(broadcastable(&[1], &[2, 3, 4]));
        assert!(!broadcastable(&[2, 3], &[4, 3]));
    }

    #[test]
    fn test_compute_strides_empty() {
        assert!(compute_strides(&[], LayoutOrder::RowMajor).is_empty());
    }

    #[test]
    fn test_scalar() {
        let l = TensorLayout::contiguous(&[1], 4);
        assert_eq!(l.numel(), 1);
        assert_eq!(l.byte_size(), 4);
    }
}
