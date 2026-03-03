//! N-dimensional permutation and contiguous conversion.

/// Descriptor for an N-dimensional axis permutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermuteDesc {
    shape: Vec<usize>,
    perm: Vec<usize>,
}

impl PermuteDesc {
    /// Create a permutation descriptor.
    ///
    /// `perm[i]` gives the source axis that maps to output axis `i`.
    ///
    /// # Panics
    ///
    /// Panics if `perm` is not a valid permutation of `0..shape.len()`.
    #[must_use]
    pub fn new(shape: Vec<usize>, perm: Vec<usize>) -> Self {
        assert_eq!(shape.len(), perm.len(), "perm length must equal number of dims");
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in &perm {
            assert!(p < n, "perm index {p} out of range 0..{n}");
            assert!(!seen[p], "duplicate axis {p} in perm");
            seen[p] = true;
        }
        Self { shape, perm }
    }

    /// The source tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The axis permutation.
    #[must_use]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Output shape after applying the permutation.
    #[must_use]
    pub fn output_shape(&self) -> Vec<usize> {
        self.perm.iter().map(|&p| self.shape[p]).collect()
    }

    /// Total number of elements (unchanged by permutation).
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the permutation is the identity (no-op).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.perm.iter().enumerate().all(|(i, &p)| i == p)
    }

    /// Number of dimensions.
    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute row-major strides for a given shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a flat index to an N-D coordinate given `strides`.
fn flat_to_nd(mut idx: usize, strides: &[usize]) -> Vec<usize> {
    strides
        .iter()
        .map(|&s| {
            let coord = idx / s;
            idx %= s;
            coord
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Permute dimensions
// ---------------------------------------------------------------------------

/// Permute the axes of an N-D tensor stored in row-major order.
///
/// Returns a new `Vec<f32>` with elements reordered so that the output
/// has shape `desc.output_shape()`.
///
/// # Panics
///
/// Panics if `src.len() != desc.total_len()`.
#[must_use]
pub fn permute_dims(src: &[f32], desc: &PermuteDesc) -> Vec<f32> {
    let total = desc.total_len();
    assert_eq!(src.len(), total, "source length {} != total elements {}", src.len(), total,);
    if total == 0 {
        return Vec::new();
    }
    if desc.is_identity() {
        return src.to_vec();
    }

    let src_strides = row_major_strides(&desc.shape);
    let out_shape = desc.output_shape();
    let out_strides = row_major_strides(&out_shape);

    let mut dst = vec![0.0f32; total];

    for (src_flat, &val) in src.iter().enumerate() {
        let src_coord = flat_to_nd(src_flat, &src_strides);
        // dst_coord[i] = src_coord[perm[i]]
        let dst_flat: usize = desc
            .perm
            .iter()
            .enumerate()
            .map(|(out_axis, &src_axis)| src_coord[src_axis] * out_strides[out_axis])
            .sum();
        dst[dst_flat] = val;
    }
    dst
}

// ---------------------------------------------------------------------------
// Contiguous copy
// ---------------------------------------------------------------------------

/// Copy a strided tensor into a fresh contiguous (row-major) buffer.
///
/// `strides` are element-strides (not byte-strides). The function
/// iterates over the logical shape in row-major order and gathers
/// from `src` using the given strides.
///
/// # Panics
///
/// Panics if `shape` and `strides` have different lengths, or if any
/// computed offset falls outside `src`.
#[must_use]
pub fn contiguous_copy(src: &[f32], shape: &[usize], strides: &[usize]) -> Vec<f32> {
    assert_eq!(shape.len(), strides.len(), "shape and strides must have the same length");
    let total: usize = shape.iter().product();
    if total == 0 {
        return Vec::new();
    }

    let row_strides = row_major_strides(shape);
    let mut dst = vec![0.0f32; total];

    for (flat, dst_elem) in dst.iter_mut().enumerate() {
        let coord = flat_to_nd(flat, &row_strides);
        let src_offset: usize = coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        assert!(
            src_offset < src.len(),
            "strided offset {src_offset} out of bounds (len {})",
            src.len(),
        );
        *dst_elem = src[src_offset];
    }
    dst
}
