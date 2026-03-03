//! Transpose descriptors and CPU reference implementations.

// ---------------------------------------------------------------------------
// TransposeDesc
// ---------------------------------------------------------------------------

/// Describes a 2-D transpose (rows ↔ cols).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposeDesc {
    rows: usize,
    cols: usize,
}

impl TransposeDesc {
    /// Create a new descriptor for a `rows × cols` matrix.
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Number of rows in the source matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns in the source matrix.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Total number of elements (`rows * cols`).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether the described matrix is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Whether the matrix is square (`rows == cols`).
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

// ---------------------------------------------------------------------------
// BatchTransposeDesc
// ---------------------------------------------------------------------------

/// Describes a batch of identically-shaped 2-D transposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTransposeDesc {
    batch_size: usize,
    inner: TransposeDesc,
}

impl BatchTransposeDesc {
    /// Create a batched descriptor for `batch_size` matrices of shape
    /// `rows × cols`.
    #[must_use]
    pub const fn new(batch_size: usize, rows: usize, cols: usize) -> Self {
        Self { batch_size, inner: TransposeDesc::new(rows, cols) }
    }

    /// Number of matrices in the batch.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The inner per-matrix descriptor.
    #[must_use]
    pub const fn inner(&self) -> &TransposeDesc {
        &self.inner
    }

    /// Total element count across all batches.
    #[must_use]
    pub const fn total_len(&self) -> usize {
        self.batch_size * self.inner.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total_len() == 0
    }
}

// ---------------------------------------------------------------------------
// 2-D transpose (CPU reference)
// ---------------------------------------------------------------------------

/// Transpose a row-major `rows × cols` matrix into a `cols × rows` result.
///
/// # Panics
///
/// Panics if `src.len() != desc.len()`.
#[must_use]
pub fn transpose_2d(src: &[f32], desc: &TransposeDesc) -> Vec<f32> {
    assert_eq!(src.len(), desc.len(), "source length {} != rows*cols {}", src.len(), desc.len(),);
    if desc.is_empty() {
        return Vec::new();
    }
    let (rows, cols) = (desc.rows, desc.cols);
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

/// In-place transpose for **square** matrices.
///
/// # Panics
///
/// Panics if `desc` is not square or `data.len() != desc.len()`.
pub fn transpose_2d_in_place(data: &mut [f32], desc: &TransposeDesc) {
    assert!(desc.is_square(), "in-place transpose requires a square matrix");
    assert_eq!(data.len(), desc.len(), "data length {} != n*n {}", data.len(), desc.len(),);
    let n = desc.rows;
    for r in 0..n {
        for c in (r + 1)..n {
            data.swap(r * n + c, c * n + r);
        }
    }
}

/// Transpose each matrix in a batch independently.
///
/// # Panics
///
/// Panics if `src.len() != desc.total_len()`.
#[must_use]
pub fn batched_transpose_2d(src: &[f32], desc: &BatchTransposeDesc) -> Vec<f32> {
    assert_eq!(
        src.len(),
        desc.total_len(),
        "source length {} != batch total {}",
        src.len(),
        desc.total_len(),
    );
    if desc.is_empty() {
        return Vec::new();
    }
    let mat_len = desc.inner().len();
    let inner = desc.inner();
    let mut dst = Vec::with_capacity(src.len());
    for b in 0..desc.batch_size() {
        let offset = b * mat_len;
        let slice = &src[offset..offset + mat_len];
        dst.extend_from_slice(&transpose_2d(slice, inner));
    }
    dst
}
