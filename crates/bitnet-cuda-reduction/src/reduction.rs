//! Reduction kernels: sum, max, min, mean, softmax, argmax.
//!
//! Every public function accepts data as a flat `&[f32]` in row-major order
//! together with `rows` × `cols` dimensions and a [`ReductionDim`] that
//! selects the reduction axis.

use std::fmt;

// ── Reduction axis ──────────────────────────────────────────────────────────

/// Axis along which a reduction is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionDim {
    /// Reduce each row to a single value (output length = `rows`).
    Row,
    /// Reduce each column to a single value (output length = `cols`).
    Column,
    /// Reduce the entire tensor to one scalar (output length = 1).
    Full,
}

impl fmt::Display for ReductionDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Row => write!(f, "Row"),
            Self::Column => write!(f, "Column"),
            Self::Full => write!(f, "Full"),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Validate that `data.len() == rows * cols` and that dimensions are
/// internally consistent. Returns `true` when the tensor is *empty* (i.e.
/// either dimension is zero), which callers treat as an early-return case.
///
/// # Panics
///
/// Panics when `data.len() != rows * cols`.
fn validate_shape(data: &[f32], rows: usize, cols: usize) -> bool {
    assert_eq!(
        data.len(),
        rows * cols,
        "data length ({}) must equal rows ({}) × cols ({})",
        data.len(),
        rows,
        cols,
    );
    rows == 0 || cols == 0
}

// ── Kernel implementations ──────────────────────────────────────────────────

/// Collection of stateless reduction kernels.
///
/// All functions are associated (no `self` receiver) and operate purely on
/// the provided slice.  On a real GPU these would launch CUDA kernels; the
/// host-side implementations here serve as the reference / fallback and are
/// used for testing.
pub struct ReductionKernels;

impl ReductionKernels {
    // -- sum -----------------------------------------------------------------

    /// Element-wise sum reduction.
    ///
    /// ```
    /// use bitnet_cuda_reduction::{ReductionDim, ReductionKernels};
    ///
    /// let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    /// assert_eq!(
    ///     ReductionKernels::sum(&m, 2, 3, ReductionDim::Row),
    ///     vec![6.0, 15.0],
    /// );
    /// ```
    #[must_use]
    pub fn sum(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<f32> {
        if validate_shape(data, rows, cols) {
            return match dim {
                ReductionDim::Row | ReductionDim::Column => vec![],
                ReductionDim::Full => vec![0.0],
            };
        }
        match dim {
            ReductionDim::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().sum()
                })
                .collect(),
            ReductionDim::Column => {
                let mut out = vec![0.0_f32; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] += data[start + c];
                    }
                }
                out
            }
            ReductionDim::Full => vec![data.iter().sum()],
        }
    }

    // -- max -----------------------------------------------------------------

    /// Element-wise maximum reduction.
    ///
    /// Returns `f32::NEG_INFINITY` for empty reductions.
    ///
    /// ```
    /// use bitnet_cuda_reduction::{ReductionDim, ReductionKernels};
    ///
    /// let m = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]; // 2×3
    /// assert_eq!(
    ///     ReductionKernels::max(&m, 2, 3, ReductionDim::Row),
    ///     vec![5.0, 6.0],
    /// );
    /// ```
    #[must_use]
    pub fn max(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<f32> {
        if validate_shape(data, rows, cols) {
            return match dim {
                ReductionDim::Row | ReductionDim::Column => vec![],
                ReductionDim::Full => vec![f32::NEG_INFINITY],
            };
        }
        match dim {
            ReductionDim::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().copied().fold(f32::NEG_INFINITY, f32::max)
                })
                .collect(),
            ReductionDim::Column => {
                let mut out = vec![f32::NEG_INFINITY; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] = f32::max(out[c], data[start + c]);
                    }
                }
                out
            }
            ReductionDim::Full => {
                vec![data.iter().copied().fold(f32::NEG_INFINITY, f32::max)]
            }
        }
    }

    // -- min -----------------------------------------------------------------

    /// Element-wise minimum reduction.
    ///
    /// Returns `f32::INFINITY` for empty reductions.
    #[must_use]
    pub fn min(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<f32> {
        if validate_shape(data, rows, cols) {
            return match dim {
                ReductionDim::Row | ReductionDim::Column => vec![],
                ReductionDim::Full => vec![f32::INFINITY],
            };
        }
        match dim {
            ReductionDim::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().copied().fold(f32::INFINITY, f32::min)
                })
                .collect(),
            ReductionDim::Column => {
                let mut out = vec![f32::INFINITY; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] = f32::min(out[c], data[start + c]);
                    }
                }
                out
            }
            ReductionDim::Full => {
                vec![data.iter().copied().fold(f32::INFINITY, f32::min)]
            }
        }
    }

    // -- mean ----------------------------------------------------------------

    /// Arithmetic mean reduction.
    ///
    /// Returns `0.0` when the reduction window is empty (matching the
    /// convention used by most tensor frameworks for a zero-element mean).
    ///
    /// ```
    /// use bitnet_cuda_reduction::{ReductionDim, ReductionKernels};
    ///
    /// let m = vec![2.0, 4.0, 6.0, 8.0]; // 2×2
    /// let row_means = ReductionKernels::mean(&m, 2, 2, ReductionDim::Row);
    /// assert!((row_means[0] - 3.0).abs() < 1e-6);
    /// assert!((row_means[1] - 7.0).abs() < 1e-6);
    /// ```
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<f32> {
        if validate_shape(data, rows, cols) {
            return match dim {
                ReductionDim::Row | ReductionDim::Column => vec![],
                ReductionDim::Full => vec![0.0],
            };
        }
        match dim {
            ReductionDim::Row => {
                let sums = Self::sum(data, rows, cols, ReductionDim::Row);
                sums.into_iter().map(|s| s / cols as f32).collect()
            }
            ReductionDim::Column => {
                let sums = Self::sum(data, rows, cols, ReductionDim::Column);
                sums.into_iter().map(|s| s / rows as f32).collect()
            }
            ReductionDim::Full => {
                let total: f32 = data.iter().sum();
                vec![total / data.len() as f32]
            }
        }
    }

    // -- softmax reduction ---------------------------------------------------

    /// Numerically-stable softmax reduction (per-row when `Row`, per-column
    /// when `Column`, or over the full tensor when `Full`).
    ///
    /// The output has the **same shape** as the input: each element is
    /// replaced by `exp(x - max) / sum(exp(x - max))` within its reduction
    /// window.
    ///
    /// ```
    /// use bitnet_cuda_reduction::{ReductionDim, ReductionKernels};
    ///
    /// let logits = vec![1.0, 2.0, 3.0];
    /// let probs = ReductionKernels::softmax(&logits, 1, 3, ReductionDim::Row);
    /// let sum: f32 = probs.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn softmax(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<f32> {
        if validate_shape(data, rows, cols) {
            return vec![];
        }
        let mut out = data.to_vec();
        match dim {
            ReductionDim::Row => {
                for r in 0..rows {
                    let start = r * cols;
                    let slice = &mut out[start..start + cols];
                    softmax_in_place(slice);
                }
            }
            ReductionDim::Column => {
                // Transpose → per-row softmax → transpose back.
                let mut transposed = vec![0.0_f32; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        transposed[c * rows + r] = out[r * cols + c];
                    }
                }
                for c in 0..cols {
                    let start = c * rows;
                    softmax_in_place(&mut transposed[start..start + rows]);
                }
                for r in 0..rows {
                    for c in 0..cols {
                        out[r * cols + c] = transposed[c * rows + r];
                    }
                }
            }
            ReductionDim::Full => {
                softmax_in_place(&mut out);
            }
        }
        out
    }

    // -- argmax --------------------------------------------------------------

    /// Index of the maximum element along the chosen dimension.
    ///
    /// For `Row` the output has one `usize` per row (column index of the
    /// row-max).  For `Column` one per column (row index).  For `Full` a
    /// single flat index.
    ///
    /// Ties are broken in favour of the *first* occurrence (lowest index).
    /// NaN values are treated as less-than any finite number (skipped).
    ///
    /// ```
    /// use bitnet_cuda_reduction::{ReductionDim, ReductionKernels};
    ///
    /// let m = vec![1.0, 3.0, 2.0, 6.0, 5.0, 4.0]; // 2×3
    /// assert_eq!(
    ///     ReductionKernels::argmax(&m, 2, 3, ReductionDim::Row),
    ///     vec![1, 0],
    /// );
    /// ```
    #[must_use]
    pub fn argmax(data: &[f32], rows: usize, cols: usize, dim: ReductionDim) -> Vec<usize> {
        if validate_shape(data, rows, cols) {
            return match dim {
                ReductionDim::Row | ReductionDim::Column => vec![],
                ReductionDim::Full => vec![0],
            };
        }
        match dim {
            ReductionDim::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    argmax_slice(&data[start..start + cols])
                })
                .collect(),
            ReductionDim::Column => {
                let mut indices = vec![0_usize; cols];
                let mut best = vec![f32::NEG_INFINITY; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        let v = data[start + c];
                        if !v.is_nan() && v > best[c] {
                            best[c] = v;
                            indices[c] = r;
                        }
                    }
                }
                indices
            }
            ReductionDim::Full => vec![argmax_slice(data)],
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

/// Numerically-stable softmax applied in-place to `slice`.
fn softmax_in_place(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in slice.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in slice.iter_mut() {
            *v /= sum;
        }
    }
}

/// Argmax over a flat slice, skipping NaN values. Returns 0 for empty slices.
fn argmax_slice(data: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if !v.is_nan() && v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

// ── GPU kernel stubs ────────────────────────────────────────────────────────

/// GPU-accelerated reduction kernels (requires `gpu` or `cuda` feature).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod gpu {
    use super::ReductionDim;

    /// Configuration for launching a CUDA reduction kernel.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ReductionLaunchConfig {
        /// Reduction axis.
        pub dim: ReductionDim,
        /// Number of threads per block.
        pub block_size: u32,
        /// Bytes of shared memory required per block.
        pub shared_mem_bytes: u32,
    }

    impl ReductionLaunchConfig {
        /// Build launch configuration for a `rows × cols` reduction.
        #[must_use]
        pub fn for_shape(rows: u32, cols: u32, dim: ReductionDim) -> Self {
            let n = match dim {
                ReductionDim::Row => cols,
                ReductionDim::Column => rows,
                ReductionDim::Full => rows.saturating_mul(cols),
            };
            let block_size = n.min(256).max(32);
            let block_size = (block_size + 31) / 32 * 32;
            Self { dim, block_size, shared_mem_bytes: block_size * 4 }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // -- sum ---------------------------------------------------------------

    #[test]
    fn sum_row_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ReductionKernels::sum(&data, 2, 3, ReductionDim::Row);
        assert_eq!(r, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_col_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ReductionKernels::sum(&data, 2, 3, ReductionDim::Column);
        assert_eq!(r, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_full_2x3() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ReductionKernels::sum(&data, 2, 3, ReductionDim::Full);
        assert!((r[0] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn sum_single_element() {
        let r = ReductionKernels::sum(&[42.0], 1, 1, ReductionDim::Full);
        assert!((r[0] - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sum_empty_row() {
        let r = ReductionKernels::sum(&[], 0, 0, ReductionDim::Row);
        assert!(r.is_empty());
    }

    #[test]
    fn sum_empty_full() {
        let r = ReductionKernels::sum(&[], 0, 0, ReductionDim::Full);
        assert!((r[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn sum_row_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let r = ReductionKernels::sum(&data, 1, 3, ReductionDim::Row);
        assert_eq!(r, vec![6.0]);
    }

    #[test]
    fn sum_col_single_col() {
        let data = vec![1.0, 2.0, 3.0];
        let r = ReductionKernels::sum(&data, 3, 1, ReductionDim::Column);
        assert!((r[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn sum_with_negatives() {
        let data = vec![-1.0, 2.0, -3.0, 4.0];
        let r = ReductionKernels::sum(&data, 2, 2, ReductionDim::Row);
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-5);
    }

    // -- max ---------------------------------------------------------------

    #[test]
    fn max_row_2x3() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::max(&data, 2, 3, ReductionDim::Row);
        assert_eq!(r, vec![5.0, 6.0]);
    }

    #[test]
    fn max_col_2x3() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::max(&data, 2, 3, ReductionDim::Column);
        assert_eq!(r, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn max_full() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::max(&data, 2, 3, ReductionDim::Full);
        assert!((r[0] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn max_empty_full() {
        let r = ReductionKernels::max(&[], 0, 0, ReductionDim::Full);
        assert_eq!(r[0], f32::NEG_INFINITY);
    }

    #[test]
    fn max_single_element() {
        let r = ReductionKernels::max(&[7.0], 1, 1, ReductionDim::Full);
        assert!((r[0] - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn max_all_negative() {
        let data = vec![-5.0, -3.0, -8.0, -1.0];
        let r = ReductionKernels::max(&data, 2, 2, ReductionDim::Full);
        assert!((r[0] - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn max_with_nan() {
        let data = vec![f32::NAN, 1.0, 2.0, 3.0];
        let r = ReductionKernels::max(&data, 1, 4, ReductionDim::Full);
        // f32::max propagates NaN
        assert!(r[0].is_nan() || (r[0] - 3.0).abs() < f32::EPSILON);
    }

    // -- min ---------------------------------------------------------------

    #[test]
    fn min_row_2x3() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::min(&data, 2, 3, ReductionDim::Row);
        assert_eq!(r, vec![1.0, 2.0]);
    }

    #[test]
    fn min_col_2x3() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::min(&data, 2, 3, ReductionDim::Column);
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn min_full() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::min(&data, 2, 3, ReductionDim::Full);
        assert!((r[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn min_empty_full() {
        let r = ReductionKernels::min(&[], 0, 0, ReductionDim::Full);
        assert_eq!(r[0], f32::INFINITY);
    }

    #[test]
    fn min_all_positive() {
        let data = vec![5.0, 3.0, 8.0, 1.0];
        let r = ReductionKernels::min(&data, 2, 2, ReductionDim::Full);
        assert!((r[0] - 1.0).abs() < f32::EPSILON);
    }

    // -- mean --------------------------------------------------------------

    #[test]
    fn mean_row_2x2() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let r = ReductionKernels::mean(&data, 2, 2, ReductionDim::Row);
        assert!((r[0] - 3.0).abs() < 1e-5);
        assert!((r[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn mean_col_2x2() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let r = ReductionKernels::mean(&data, 2, 2, ReductionDim::Column);
        assert!((r[0] - 4.0).abs() < 1e-5);
        assert!((r[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn mean_full() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let r = ReductionKernels::mean(&data, 2, 2, ReductionDim::Full);
        assert!((r[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn mean_empty_full() {
        let r = ReductionKernels::mean(&[], 0, 0, ReductionDim::Full);
        assert!((r[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn mean_single_element() {
        let r = ReductionKernels::mean(&[10.0], 1, 1, ReductionDim::Full);
        assert!((r[0] - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mean_uniform_values() {
        let data = vec![5.0; 9];
        let r = ReductionKernels::mean(&data, 3, 3, ReductionDim::Full);
        assert!((r[0] - 5.0).abs() < 1e-5);
    }

    // -- softmax -----------------------------------------------------------

    #[test]
    fn softmax_row_sums_to_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ReductionKernels::softmax(&data, 2, 3, ReductionDim::Row);
        let s1: f32 = r[..3].iter().sum();
        let s2: f32 = r[3..].iter().sum();
        assert!((s1 - 1.0).abs() < 1e-5);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_col_sums_to_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = ReductionKernels::softmax(&data, 2, 3, ReductionDim::Column);
        for c in 0..3 {
            let col_sum: f32 = (0..2).map(|row| r[row * 3 + c]).sum();
            assert!((col_sum - 1.0).abs() < 1e-5, "col {c} sum = {col_sum}");
        }
    }

    #[test]
    fn softmax_full_sums_to_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let r = ReductionKernels::softmax(&data, 2, 2, ReductionDim::Full);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_empty_returns_empty() {
        let r = ReductionKernels::softmax(&[], 0, 0, ReductionDim::Row);
        assert!(r.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let r = ReductionKernels::softmax(&[100.0], 1, 1, ReductionDim::Row);
        assert!((r[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_all_equal() {
        let data = vec![1.0; 4];
        let r = ReductionKernels::softmax(&data, 1, 4, ReductionDim::Row);
        for &v in &r {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_large_values_no_overflow() {
        let data = vec![1000.0, 1001.0, 1002.0];
        let r = ReductionKernels::softmax(&data, 1, 3, ReductionDim::Row);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // largest input → largest probability
        assert!(r[2] > r[1]);
        assert!(r[1] > r[0]);
    }

    #[test]
    fn softmax_negative_values() {
        let data = vec![-1.0, -2.0, -3.0];
        let r = ReductionKernels::softmax(&data, 1, 3, ReductionDim::Row);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_preserves_ordering() {
        let data = vec![1.0, 3.0, 2.0];
        let r = ReductionKernels::softmax(&data, 1, 3, ReductionDim::Row);
        assert!(r[1] > r[2]);
        assert!(r[2] > r[0]);
    }

    // -- argmax ------------------------------------------------------------

    #[test]
    fn argmax_row_2x3() {
        let data = vec![1.0, 3.0, 2.0, 6.0, 5.0, 4.0];
        let r = ReductionKernels::argmax(&data, 2, 3, ReductionDim::Row);
        assert_eq!(r, vec![1, 0]);
    }

    #[test]
    fn argmax_col_2x3() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::argmax(&data, 2, 3, ReductionDim::Column);
        assert_eq!(r, vec![1, 0, 1]);
    }

    #[test]
    fn argmax_full() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let r = ReductionKernels::argmax(&data, 2, 3, ReductionDim::Full);
        assert_eq!(r, vec![5]);
    }

    #[test]
    fn argmax_empty() {
        let r = ReductionKernels::argmax(&[], 0, 0, ReductionDim::Full);
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn argmax_tie_first_occurrence() {
        let data = vec![5.0, 5.0, 5.0];
        let r = ReductionKernels::argmax(&data, 1, 3, ReductionDim::Row);
        assert_eq!(r[0], 0); // first occurrence wins
    }

    #[test]
    fn argmax_single_element() {
        let r = ReductionKernels::argmax(&[42.0], 1, 1, ReductionDim::Full);
        assert_eq!(r[0], 0);
    }

    #[test]
    fn argmax_nan_skipped() {
        let data = vec![f32::NAN, 1.0, 2.0];
        let r = ReductionKernels::argmax(&data, 1, 3, ReductionDim::Row);
        assert_eq!(r[0], 2); // NaN skipped, 2.0 is max
    }

    #[test]
    fn argmax_all_nan_returns_zero() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let r = ReductionKernels::argmax(&data, 1, 3, ReductionDim::Row);
        assert_eq!(r[0], 0); // no valid element, default to 0
    }

    // -- shape validation --------------------------------------------------

    #[test]
    #[should_panic(expected = "data length")]
    fn sum_mismatched_shape_panics() {
        let _ = ReductionKernels::sum(&[1.0, 2.0], 2, 3, ReductionDim::Full);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn max_mismatched_shape_panics() {
        let _ = ReductionKernels::max(&[1.0], 2, 2, ReductionDim::Full);
    }

    // -- ReductionDim display ----------------------------------------------

    #[test]
    fn reduction_dim_display() {
        assert_eq!(format!("{}", ReductionDim::Row), "Row");
        assert_eq!(format!("{}", ReductionDim::Column), "Column");
        assert_eq!(format!("{}", ReductionDim::Full), "Full");
    }

    #[test]
    fn reduction_dim_clone_eq() {
        let d = ReductionDim::Row;
        let d2 = d;
        assert_eq!(d, d2);
    }

    // -- large matrix stress -----------------------------------------------

    #[test]
    fn sum_large_matrix() {
        let rows = 100;
        let cols = 200;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let r = ReductionKernels::sum(&data, rows, cols, ReductionDim::Full);
        let expected: f32 = (0..rows * cols).map(|i| i as f32).sum();
        assert!((r[0] - expected).abs() / expected < 1e-4);
    }

    #[test]
    fn softmax_large_row() {
        let cols = 1024;
        let data: Vec<f32> = (0..cols).map(|i| i as f32 / 100.0).collect();
        let r = ReductionKernels::softmax(&data, 1, cols, ReductionDim::Row);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }

    // -- edge cases --------------------------------------------------------

    #[test]
    fn sum_row_1x1() {
        let r = ReductionKernels::sum(&[5.0], 1, 1, ReductionDim::Row);
        assert_eq!(r, vec![5.0]);
    }

    #[test]
    fn min_single_element() {
        let r = ReductionKernels::min(&[3.0], 1, 1, ReductionDim::Full);
        assert!((r[0] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mean_row_1x4() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let r = ReductionKernels::mean(&data, 1, 4, ReductionDim::Row);
        assert!((r[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn argmax_col_single_col() {
        let data = vec![1.0, 3.0, 2.0];
        let r = ReductionKernels::argmax(&data, 3, 1, ReductionDim::Column);
        assert_eq!(r, vec![1]);
    }

    #[test]
    fn sum_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let r = ReductionKernels::sum(&data, 1, 3, ReductionDim::Full);
        assert_eq!(r[0], f32::INFINITY);
    }

    #[test]
    fn max_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let r = ReductionKernels::max(&data, 1, 3, ReductionDim::Full);
        assert_eq!(r[0], f32::INFINITY);
    }

    #[test]
    fn min_with_neg_inf() {
        let data = vec![1.0, f32::NEG_INFINITY, 3.0];
        let r = ReductionKernels::min(&data, 1, 3, ReductionDim::Full);
        assert_eq!(r[0], f32::NEG_INFINITY);
    }

    #[test]
    fn argmax_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let r = ReductionKernels::argmax(&data, 1, 3, ReductionDim::Row);
        assert_eq!(r[0], 1);
    }

    // -- GPU cfg tests (compile only without feature) ----------------------

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod gpu_tests {
        use super::super::ReductionDim;
        use super::super::gpu::ReductionLaunchConfig;

        #[test]
        fn launch_config_row() {
            let c = ReductionLaunchConfig::for_shape(4, 256, ReductionDim::Row);
            assert_eq!(c.dim, ReductionDim::Row);
            assert_eq!(c.block_size, 256);
        }

        #[test]
        fn launch_config_full_small() {
            let c = ReductionLaunchConfig::for_shape(2, 8, ReductionDim::Full);
            assert_eq!(c.block_size, 32); // clamped to min warp
        }

        #[test]
        fn launch_config_shared_mem() {
            let c = ReductionLaunchConfig::for_shape(4, 128, ReductionDim::Column);
            assert_eq!(c.shared_mem_bytes, c.block_size * 4);
        }
    }
}

// ── Property tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy that generates a (rows, cols, data) triple with valid shape.
    fn matrix_strategy() -> impl Strategy<Value = (usize, usize, Vec<f32>)> {
        (1_usize..=20, 1_usize..=20).prop_flat_map(|(rows, cols)| {
            let len = rows * cols;
            (Just(rows), Just(cols), proptest::collection::vec(-1e6_f32..1e6, len..=len))
        })
    }

    proptest! {
        #[test]
        fn sum_row_length((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::sum(&data, rows, cols, ReductionDim::Row);
            prop_assert_eq!(r.len(), rows);
        }

        #[test]
        fn sum_col_length((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::sum(&data, rows, cols, ReductionDim::Column);
            prop_assert_eq!(r.len(), cols);
        }

        #[test]
        fn sum_full_length((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::sum(&data, rows, cols, ReductionDim::Full);
            prop_assert_eq!(r.len(), 1);
        }

        #[test]
        fn sum_full_equals_row_sum_sum((rows, cols, data) in matrix_strategy()) {
            let full = ReductionKernels::sum(&data, rows, cols, ReductionDim::Full);
            let row_sums = ReductionKernels::sum(&data, rows, cols, ReductionDim::Row);
            let row_total: f32 = row_sums.iter().sum();
            prop_assert!((full[0] - row_total).abs() < 1e-1 * (1.0 + full[0].abs()));
        }

        #[test]
        fn max_row_le_full_max((rows, cols, data) in matrix_strategy()) {
            let full_max = ReductionKernels::max(&data, rows, cols, ReductionDim::Full)[0];
            let row_maxes = ReductionKernels::max(&data, rows, cols, ReductionDim::Row);
            for &rm in &row_maxes {
                prop_assert!(rm <= full_max + f32::EPSILON);
            }
        }

        #[test]
        fn min_row_ge_full_min((rows, cols, data) in matrix_strategy()) {
            let full_min = ReductionKernels::min(&data, rows, cols, ReductionDim::Full)[0];
            let row_mins = ReductionKernels::min(&data, rows, cols, ReductionDim::Row);
            for &rm in &row_mins {
                prop_assert!(rm >= full_min - f32::EPSILON);
            }
        }

        #[test]
        fn mean_between_min_and_max((rows, cols, data) in matrix_strategy()) {
            let mean = ReductionKernels::mean(&data, rows, cols, ReductionDim::Full)[0];
            let min_val = ReductionKernels::min(&data, rows, cols, ReductionDim::Full)[0];
            let max_val = ReductionKernels::max(&data, rows, cols, ReductionDim::Full)[0];
            prop_assert!(mean >= min_val - 1e-3);
            prop_assert!(mean <= max_val + 1e-3);
        }

        #[test]
        fn softmax_row_sums_to_one((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::softmax(&data, rows, cols, ReductionDim::Row);
            for row in 0..rows {
                let start = row * cols;
                let s: f32 = r[start..start + cols].iter().sum();
                prop_assert!((s - 1.0).abs() < 1e-4, "row {row} sum = {s}");
            }
        }

        #[test]
        fn softmax_all_non_negative((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::softmax(&data, rows, cols, ReductionDim::Row);
            for (i, &v) in r.iter().enumerate() {
                prop_assert!(v >= 0.0, "softmax[{i}] = {v} < 0");
            }
        }

        #[test]
        fn argmax_row_in_bounds((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::argmax(&data, rows, cols, ReductionDim::Row);
            prop_assert_eq!(r.len(), rows);
            for &idx in &r {
                prop_assert!(idx < cols);
            }
        }

        #[test]
        fn argmax_col_in_bounds((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::argmax(&data, rows, cols, ReductionDim::Column);
            prop_assert_eq!(r.len(), cols);
            for &idx in &r {
                prop_assert!(idx < rows);
            }
        }

        #[test]
        fn argmax_full_in_bounds((rows, cols, data) in matrix_strategy()) {
            let r = ReductionKernels::argmax(&data, rows, cols, ReductionDim::Full);
            prop_assert_eq!(r.len(), 1);
            prop_assert!(r[0] < rows * cols);
        }

        #[test]
        fn argmax_row_value_is_max((rows, cols, data) in matrix_strategy()) {
            let argmaxes = ReductionKernels::argmax(&data, rows, cols, ReductionDim::Row);
            let maxes = ReductionKernels::max(&data, rows, cols, ReductionDim::Row);
            for (r, (&idx, &mx)) in argmaxes.iter().zip(maxes.iter()).enumerate() {
                let val = data[r * cols + idx];
                prop_assert!(
                    (val - mx).abs() < 1e-5,
                    "row {r}: data[argmax={idx}]={val} != max={mx}"
                );
            }
        }
    }
}
