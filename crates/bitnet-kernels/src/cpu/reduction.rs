//! CPU reduction operations kernel.
//!
//! Provides reduction operations (sum, mean, max, min, product, norms)
//! on contiguous `f32` slices and 2-D row-major matrices.  All
//! reductions support both full (1-D) and axis-wise (row / column)
//! modes via [`ReductionAxis`].

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

fn validate_non_empty(data: &[f32]) -> Result<()> {
    if data.is_empty() {
        return Err(invalid_args("input must not be empty"));
    }
    Ok(())
}

fn validate_matrix(data: &[f32], rows: usize, cols: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(invalid_args("rows and cols must be > 0"));
    }
    if data.len() != rows * cols {
        return Err(invalid_args("data length must equal rows * cols"));
    }
    Ok(())
}

// ── Types ──────────────────────────────────────────────────────────

/// Axis along which a 2-D reduction is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionAxis {
    /// Reduce each row to a single value (output length = `rows`).
    Row,
    /// Reduce each column to a single value (output length = `cols`).
    Column,
}

/// Result of a max or min reduction that also tracks the index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueWithIndex {
    /// The extreme value.
    pub value: f32,
    /// Index of the first occurrence of that value.
    pub index: usize,
}

// ── Kernel ─────────────────────────────────────────────────────────

/// Stateless dispatcher for CPU reduction operations.
pub struct ReductionKernel;

impl ReductionKernel {
    // ── Sum ────────────────────────────────────────────────────

    /// Sum all elements.
    pub fn sum(data: &[f32]) -> Result<f32> {
        validate_non_empty(data)?;
        Ok(data.iter().sum())
    }

    /// Sum along an axis of a row-major matrix.
    pub fn sum_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<f32>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().sum()
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out = vec![0.0_f32; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] += data[start + c];
                    }
                }
                out
            }
        })
    }

    // ── Mean ───────────────────────────────────────────────────

    /// Arithmetic mean of all elements.
    pub fn mean(data: &[f32]) -> Result<f32> {
        validate_non_empty(data)?;
        Ok(data.iter().sum::<f32>() / data.len() as f32)
    }

    /// Mean along an axis of a row-major matrix.
    pub fn mean_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<f32>> {
        let sums = Self::sum_axis(data, rows, cols, axis)?;
        let divisor = match axis {
            ReductionAxis::Row => cols as f32,
            ReductionAxis::Column => rows as f32,
        };
        Ok(sums.into_iter().map(|s| s / divisor).collect())
    }

    // ── Max / Argmax ───────────────────────────────────────────

    /// Maximum value and its index.
    pub fn max(data: &[f32]) -> Result<ValueWithIndex> {
        validate_non_empty(data)?;
        let (index, &value) =
            data.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap(); // safe: non-empty
        Ok(ValueWithIndex { value, index })
    }

    /// Per-axis maximum values.
    pub fn max_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<ValueWithIndex>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    let row = &data[start..start + cols];
                    let (ci, &value) =
                        row.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();
                    ValueWithIndex { value, index: ci }
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out: Vec<ValueWithIndex> =
                    (0..cols).map(|c| ValueWithIndex { value: data[c], index: 0 }).collect();
                for r in 1..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        let v = data[start + c];
                        if v > out[c].value {
                            out[c] = ValueWithIndex { value: v, index: r };
                        }
                    }
                }
                out
            }
        })
    }

    // ── Min / Argmin ───────────────────────────────────────────

    /// Minimum value and its index.
    pub fn min(data: &[f32]) -> Result<ValueWithIndex> {
        validate_non_empty(data)?;
        let (index, &value) =
            data.iter().enumerate().min_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();
        Ok(ValueWithIndex { value, index })
    }

    /// Per-axis minimum values.
    pub fn min_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<ValueWithIndex>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    let row = &data[start..start + cols];
                    let (ci, &value) =
                        row.iter().enumerate().min_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();
                    ValueWithIndex { value, index: ci }
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out: Vec<ValueWithIndex> =
                    (0..cols).map(|c| ValueWithIndex { value: data[c], index: 0 }).collect();
                for r in 1..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        let v = data[start + c];
                        if v < out[c].value {
                            out[c] = ValueWithIndex { value: v, index: r };
                        }
                    }
                }
                out
            }
        })
    }

    // ── Product ────────────────────────────────────────────────

    /// Product of all elements.
    pub fn product(data: &[f32]) -> Result<f32> {
        validate_non_empty(data)?;
        Ok(data.iter().product())
    }

    /// Product along an axis of a row-major matrix.
    pub fn product_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<f32>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().product()
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out = vec![1.0_f32; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] *= data[start + c];
                    }
                }
                out
            }
        })
    }

    // ── Norms ──────────────────────────────────────────────────

    /// L1 norm (sum of absolute values).
    pub fn l1_norm(data: &[f32]) -> Result<f32> {
        validate_non_empty(data)?;
        Ok(data.iter().map(|x| x.abs()).sum())
    }

    /// L1 norm along an axis of a row-major matrix.
    pub fn l1_norm_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<f32>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().map(|x| x.abs()).sum()
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out = vec![0.0_f32; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] += data[start + c].abs();
                    }
                }
                out
            }
        })
    }

    /// L2 norm (Euclidean norm).
    pub fn l2_norm(data: &[f32]) -> Result<f32> {
        validate_non_empty(data)?;
        Ok(data.iter().map(|x| x * x).sum::<f32>().sqrt())
    }

    /// L2 norm along an axis of a row-major matrix.
    pub fn l2_norm_axis(
        data: &[f32],
        rows: usize,
        cols: usize,
        axis: ReductionAxis,
    ) -> Result<Vec<f32>> {
        validate_matrix(data, rows, cols)?;
        Ok(match axis {
            ReductionAxis::Row => (0..rows)
                .map(|r| {
                    let start = r * cols;
                    data[start..start + cols].iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect(),
            ReductionAxis::Column => {
                let mut out = vec![0.0_f32; cols];
                for r in 0..rows {
                    let start = r * cols;
                    for c in 0..cols {
                        out[c] += data[start + c] * data[start + c];
                    }
                }
                out.iter_mut().for_each(|v| *v = v.sqrt());
                out
            }
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn approx_vec(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx(*x, *y))
    }

    // ── Sum ────────────────────────────────────────────────────

    #[test]
    fn sum_basic() {
        assert!(approx(ReductionKernel::sum(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 10.0,));
    }

    #[test]
    fn sum_single() {
        assert!(approx(ReductionKernel::sum(&[42.0]).unwrap(), 42.0,));
    }

    #[test]
    fn sum_empty_rejected() {
        assert!(ReductionKernel::sum(&[]).is_err());
    }

    #[test]
    fn sum_axis_row() {
        // [[1,2,3],[4,5,6]]  row sums → [6, 15]
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ReductionKernel::sum_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&out, &[6.0, 15.0]));
    }

    #[test]
    fn sum_axis_column() {
        // [[1,2,3],[4,5,6]]  col sums → [5, 7, 9]
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ReductionKernel::sum_axis(&data, 2, 3, ReductionAxis::Column).unwrap();
        assert!(approx_vec(&out, &[5.0, 7.0, 9.0]));
    }

    #[test]
    fn sum_axis_dimension_mismatch() {
        assert!(ReductionKernel::sum_axis(&[1.0, 2.0], 2, 3, ReductionAxis::Row,).is_err());
    }

    // ── Mean ───────────────────────────────────────────────────

    #[test]
    fn mean_basic() {
        assert!(approx(ReductionKernel::mean(&[2.0, 4.0, 6.0]).unwrap(), 4.0,));
    }

    #[test]
    fn mean_axis_row() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ReductionKernel::mean_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&out, &[2.0, 5.0]));
    }

    #[test]
    fn mean_axis_column() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = ReductionKernel::mean_axis(&data, 2, 3, ReductionAxis::Column).unwrap();
        assert!(approx_vec(&out, &[2.5, 3.5, 4.5]));
    }

    // ── Max / Argmax ───────────────────────────────────────────

    #[test]
    fn max_basic() {
        let r = ReductionKernel::max(&[3.0, 1.0, 5.0, 2.0]).unwrap();
        assert!(approx(r.value, 5.0));
        assert_eq!(r.index, 2);
    }

    #[test]
    fn max_negative() {
        let r = ReductionKernel::max(&[-10.0, -3.0, -7.0]).unwrap();
        assert!(approx(r.value, -3.0));
        assert_eq!(r.index, 1);
    }

    #[test]
    fn max_empty_rejected() {
        assert!(ReductionKernel::max(&[]).is_err());
    }

    #[test]
    fn max_axis_row() {
        // [[1,5,3],[4,2,6]]
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = ReductionKernel::max_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
        assert!(approx(out[0].value, 5.0));
        assert_eq!(out[0].index, 1);
        assert!(approx(out[1].value, 6.0));
        assert_eq!(out[1].index, 2);
    }

    #[test]
    fn max_axis_column() {
        // [[1,5,3],[4,2,6]] col maxes → (4@r1, 5@r0, 6@r1)
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = ReductionKernel::max_axis(&data, 2, 3, ReductionAxis::Column).unwrap();
        assert!(approx(out[0].value, 4.0));
        assert_eq!(out[0].index, 1);
        assert!(approx(out[1].value, 5.0));
        assert_eq!(out[1].index, 0);
        assert!(approx(out[2].value, 6.0));
        assert_eq!(out[2].index, 1);
    }

    // ── Min / Argmin ───────────────────────────────────────────

    #[test]
    fn min_basic() {
        let r = ReductionKernel::min(&[3.0, 1.0, 5.0, 2.0]).unwrap();
        assert!(approx(r.value, 1.0));
        assert_eq!(r.index, 1);
    }

    #[test]
    fn min_axis_row() {
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = ReductionKernel::min_axis(&data, 2, 3, ReductionAxis::Row).unwrap();
        assert!(approx(out[0].value, 1.0));
        assert_eq!(out[0].index, 0);
        assert!(approx(out[1].value, 2.0));
        assert_eq!(out[1].index, 1);
    }

    #[test]
    fn min_axis_column() {
        let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = ReductionKernel::min_axis(&data, 2, 3, ReductionAxis::Column).unwrap();
        assert!(approx(out[0].value, 1.0));
        assert_eq!(out[0].index, 0);
        assert!(approx(out[1].value, 2.0));
        assert_eq!(out[1].index, 1);
        assert!(approx(out[2].value, 3.0));
        assert_eq!(out[2].index, 0);
    }

    // ── Product ────────────────────────────────────────────────

    #[test]
    fn product_basic() {
        assert!(approx(ReductionKernel::product(&[2.0, 3.0, 4.0]).unwrap(), 24.0,));
    }

    #[test]
    fn product_with_zero() {
        assert!(approx(ReductionKernel::product(&[5.0, 0.0, 3.0]).unwrap(), 0.0,));
    }

    #[test]
    fn product_axis_row() {
        // [[1,2],[3,4]]  row products → [2, 12]
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = ReductionKernel::product_axis(&data, 2, 2, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&out, &[2.0, 12.0]));
    }

    #[test]
    fn product_axis_column() {
        // [[1,2],[3,4]]  col products → [3, 8]
        let data = [1.0, 2.0, 3.0, 4.0];
        let out = ReductionKernel::product_axis(&data, 2, 2, ReductionAxis::Column).unwrap();
        assert!(approx_vec(&out, &[3.0, 8.0]));
    }

    // ── L1 norm ────────────────────────────────────────────────

    #[test]
    fn l1_norm_basic() {
        assert!(approx(ReductionKernel::l1_norm(&[-1.0, 2.0, -3.0]).unwrap(), 6.0,));
    }

    #[test]
    fn l1_norm_axis_row() {
        // [[-1,2],[-3,4]]  row L1 → [3, 7]
        let data = [-1.0, 2.0, -3.0, 4.0];
        let out = ReductionKernel::l1_norm_axis(&data, 2, 2, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&out, &[3.0, 7.0]));
    }

    #[test]
    fn l1_norm_axis_column() {
        // [[-1,2],[-3,4]]  col L1 → [4, 6]
        let data = [-1.0, 2.0, -3.0, 4.0];
        let out = ReductionKernel::l1_norm_axis(&data, 2, 2, ReductionAxis::Column).unwrap();
        assert!(approx_vec(&out, &[4.0, 6.0]));
    }

    // ── L2 norm ────────────────────────────────────────────────

    #[test]
    fn l2_norm_basic() {
        // sqrt(3^2 + 4^2) = 5
        assert!(approx(ReductionKernel::l2_norm(&[3.0, 4.0]).unwrap(), 5.0,));
    }

    #[test]
    fn l2_norm_axis_row() {
        // [[3,4],[5,12]]  row L2 → [5, 13]
        let data = [3.0, 4.0, 5.0, 12.0];
        let out = ReductionKernel::l2_norm_axis(&data, 2, 2, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&out, &[5.0, 13.0]));
    }

    #[test]
    fn l2_norm_axis_column() {
        // [[3,4],[5,12]]  col L2 → [sqrt(34), sqrt(160)]
        let data = [3.0, 4.0, 5.0, 12.0];
        let out = ReductionKernel::l2_norm_axis(&data, 2, 2, ReductionAxis::Column).unwrap();
        assert!(approx(out[0], 34.0_f32.sqrt()));
        assert!(approx(out[1], 160.0_f32.sqrt()));
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn single_element_all_ops() {
        let d = [7.0_f32];
        assert!(approx(ReductionKernel::sum(&d).unwrap(), 7.0));
        assert!(approx(ReductionKernel::mean(&d).unwrap(), 7.0));
        assert!(approx(ReductionKernel::product(&d).unwrap(), 7.0,));
        assert!(approx(ReductionKernel::l1_norm(&d).unwrap(), 7.0,));
        assert!(approx(ReductionKernel::l2_norm(&d).unwrap(), 7.0,));
        let mx = ReductionKernel::max(&d).unwrap();
        assert!(approx(mx.value, 7.0));
        assert_eq!(mx.index, 0);
    }

    #[test]
    fn zero_rows_rejected() {
        assert!(ReductionKernel::sum_axis(&[], 0, 3, ReductionAxis::Row,).is_err());
    }

    #[test]
    fn zero_cols_rejected() {
        assert!(ReductionKernel::sum_axis(&[], 3, 0, ReductionAxis::Row,).is_err());
    }

    #[test]
    fn large_input_sum() {
        let data: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
        let expected = 1024.0 * 1025.0 / 2.0;
        assert!(approx(ReductionKernel::sum(&data).unwrap(), expected,));
    }

    #[test]
    fn negative_values_l2() {
        // L2 norm is sign-independent
        let a = ReductionKernel::l2_norm(&[3.0, -4.0]).unwrap();
        let b = ReductionKernel::l2_norm(&[-3.0, 4.0]).unwrap();
        assert!(approx(a, b));
        assert!(approx(a, 5.0));
    }
}
