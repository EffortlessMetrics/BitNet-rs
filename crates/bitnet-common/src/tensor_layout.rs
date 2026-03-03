//! Tensor memory layout validation and utilities.
//!
//! Validate strides, contiguity, alignment, and compute offsets
//! for multi-dimensional tensor layouts.

/// Memory layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutOrder {
    RowMajor,    // C-style
    ColumnMajor, // Fortran-style
}

/// Tensor layout descriptor.
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub order: LayoutOrder,
}

impl TensorLayout {
    /// Create a contiguous row-major layout.
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let strides = compute_strides_row_major(&shape);
        Self { shape, strides, offset: 0, order: LayoutOrder::RowMajor }
    }

    /// Create a contiguous column-major layout.
    pub fn column_major(shape: Vec<usize>) -> Self {
        let strides = compute_strides_col_major(&shape);
        Self { shape, strides, offset: 0, order: LayoutOrder::ColumnMajor }
    }

    /// Create with custom strides.
    pub fn with_strides(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self { shape, strides, offset: 0, order: LayoutOrder::RowMajor }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let expected = compute_strides_row_major(&self.shape);
        self.strides == expected
    }

    pub fn is_column_contiguous(&self) -> bool {
        let expected = compute_strides_col_major(&self.shape);
        self.strides == expected
    }

    /// Check if the layout is valid (strides match shape).
    pub fn is_valid(&self) -> bool {
        self.shape.len() == self.strides.len()
    }

    /// Compute flat offset for given indices.
    pub fn flat_offset(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                let _ = i; // suppress unused warning
                return None;
            }
        }
        let offset: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        Some(self.offset + offset)
    }

    /// Total bytes needed (assumes f32 = 4 bytes per element).
    pub fn size_bytes(&self, elem_size: usize) -> usize {
        self.numel() * elem_size
    }

    /// Check alignment.
    pub fn is_aligned(&self, alignment: usize) -> bool {
        if alignment == 0 {
            return true;
        }
        self.offset.is_multiple_of(alignment)
    }

    /// Transpose (swap last two dims).
    pub fn transpose(&self) -> Option<Self> {
        if self.ndim() < 2 {
            return None;
        }
        let n = self.ndim();
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(n - 2, n - 1);
        new_strides.swap(n - 2, n - 1);
        Some(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            order: self.order,
        })
    }

    /// Reshape (only valid for contiguous layouts).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<Self> {
        if !self.is_contiguous() {
            return None;
        }
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return None;
        }
        Some(Self::contiguous(new_shape))
    }
}

fn compute_strides_row_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn compute_strides_col_major(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_2d() {
        let layout = TensorLayout::contiguous(vec![3, 4]);
        assert_eq!(layout.strides, vec![4, 1]);
        assert!(layout.is_contiguous());
        assert_eq!(layout.numel(), 12);
    }

    #[test]
    fn test_column_major() {
        let layout = TensorLayout::column_major(vec![3, 4]);
        assert_eq!(layout.strides, vec![1, 3]);
        assert!(layout.is_column_contiguous());
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn test_flat_offset() {
        let layout = TensorLayout::contiguous(vec![2, 3]);
        assert_eq!(layout.flat_offset(&[0, 0]), Some(0));
        assert_eq!(layout.flat_offset(&[1, 2]), Some(5));
    }

    #[test]
    fn test_flat_offset_oob() {
        let layout = TensorLayout::contiguous(vec![2, 3]);
        assert_eq!(layout.flat_offset(&[2, 0]), None);
        assert_eq!(layout.flat_offset(&[0]), None);
    }

    #[test]
    fn test_3d_strides() {
        let layout = TensorLayout::contiguous(vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![12, 4, 1]);
        assert_eq!(layout.flat_offset(&[1, 2, 3]), Some(23));
    }

    #[test]
    fn test_transpose() {
        let layout = TensorLayout::contiguous(vec![2, 3]);
        let transposed = layout.transpose().unwrap();
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.strides, vec![1, 3]);
    }

    #[test]
    fn test_reshape() {
        let layout = TensorLayout::contiguous(vec![2, 6]);
        let reshaped = layout.reshape(vec![3, 4]).unwrap();
        assert_eq!(reshaped.numel(), 12);
        assert!(reshaped.is_contiguous());
    }

    #[test]
    fn test_reshape_mismatch() {
        let layout = TensorLayout::contiguous(vec![2, 3]);
        assert!(layout.reshape(vec![2, 4]).is_none());
    }

    #[test]
    fn test_size_bytes() {
        let layout = TensorLayout::contiguous(vec![100, 200]);
        assert_eq!(layout.size_bytes(4), 100 * 200 * 4);
    }

    #[test]
    fn test_alignment() {
        let layout = TensorLayout::contiguous(vec![4, 4]);
        assert!(layout.is_aligned(16));
        let mut layout2 = layout;
        layout2.offset = 7;
        assert!(!layout2.is_aligned(16));
    }

    #[test]
    fn test_ndim() {
        assert_eq!(TensorLayout::contiguous(vec![2, 3, 4]).ndim(), 3);
        assert_eq!(TensorLayout::contiguous(vec![10]).ndim(), 1);
    }

    #[test]
    fn test_custom_strides() {
        let layout = TensorLayout::with_strides(vec![2, 3], vec![6, 2]);
        assert!(!layout.is_contiguous());
        assert!(layout.is_valid());
    }
}
