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

    // ── All-same values ───────────────────────────────────────

    #[test]
    fn reduce_all_same_values() {
        let d = [5.0_f32; 8];
        assert!(approx(ReductionKernel::sum(&d).unwrap(), 40.0));
        assert!(approx(ReductionKernel::mean(&d).unwrap(), 5.0));
        let mx = ReductionKernel::max(&d).unwrap();
        assert!(approx(mx.value, 5.0));
        assert_eq!(mx.index, 7); // max_by returns last for ties
        let mn = ReductionKernel::min(&d).unwrap();
        assert!(approx(mn.value, 5.0));
        assert_eq!(mn.index, 0); // min_by returns first for ties
    }

    #[test]
    fn reduce_all_zeros() {
        let d = [0.0_f32; 4];
        assert!(approx(ReductionKernel::sum(&d).unwrap(), 0.0));
        assert!(approx(ReductionKernel::mean(&d).unwrap(), 0.0));
        assert!(approx(ReductionKernel::product(&d).unwrap(), 0.0));
        assert!(approx(ReductionKernel::l1_norm(&d).unwrap(), 0.0));
        assert!(approx(ReductionKernel::l2_norm(&d).unwrap(), 0.0));
    }

    // ── NaN handling ──────────────────────────────────────────

    #[test]
    fn reduce_nan_sum_propagates() {
        let d = [1.0, f32::NAN, 3.0];
        assert!(ReductionKernel::sum(&d).unwrap().is_nan());
    }

    #[test]
    fn reduce_nan_mean_propagates() {
        let d = [1.0, f32::NAN, 3.0];
        assert!(ReductionKernel::mean(&d).unwrap().is_nan());
    }

    #[test]
    fn reduce_nan_max_uses_total_cmp() {
        // total_cmp places NaN above all finite values
        let d = [1.0, f32::NAN, 3.0];
        let r = ReductionKernel::max(&d).unwrap();
        assert!(r.value.is_nan());
        assert_eq!(r.index, 1);
    }

    #[test]
    fn reduce_nan_min_uses_total_cmp() {
        // total_cmp: -NaN < -Inf < ... < Inf < NaN
        let d = [1.0, f32::NAN, 3.0];
        let r = ReductionKernel::min(&d).unwrap();
        assert!(approx(r.value, 1.0));
        assert_eq!(r.index, 0);
    }

    #[test]
    fn reduce_nan_product_propagates() {
        let d = [2.0, f32::NAN, 3.0];
        assert!(ReductionKernel::product(&d).unwrap().is_nan());
    }

    #[test]
    fn reduce_nan_l1_propagates() {
        let d = [1.0, f32::NAN, 3.0];
        assert!(ReductionKernel::l1_norm(&d).unwrap().is_nan());
    }

    #[test]
    fn reduce_nan_l2_propagates() {
        let d = [1.0, f32::NAN, 3.0];
        assert!(ReductionKernel::l2_norm(&d).unwrap().is_nan());
    }

    // ── Infinity handling ─────────────────────────────────────

    #[test]
    fn reduce_inf_max() {
        let d = [1.0, f32::INFINITY, 3.0];
        let r = ReductionKernel::max(&d).unwrap();
        assert_eq!(r.value, f32::INFINITY);
        assert_eq!(r.index, 1);
    }

    #[test]
    fn reduce_neg_inf_min() {
        let d = [1.0, f32::NEG_INFINITY, 3.0];
        let r = ReductionKernel::min(&d).unwrap();
        assert_eq!(r.value, f32::NEG_INFINITY);
        assert_eq!(r.index, 1);
    }

    // ── Empty-slice errors for all ops ────────────────────────

    #[test]
    fn reduce_empty_rejected_all_ops() {
        let e: &[f32] = &[];
        assert!(ReductionKernel::sum(e).is_err());
        assert!(ReductionKernel::mean(e).is_err());
        assert!(ReductionKernel::max(e).is_err());
        assert!(ReductionKernel::min(e).is_err());
        assert!(ReductionKernel::product(e).is_err());
        assert!(ReductionKernel::l1_norm(e).is_err());
        assert!(ReductionKernel::l2_norm(e).is_err());
    }

    // ── Single-element min/argmin ─────────────────────────────

    #[test]
    fn reduce_single_element_min() {
        let d = [42.0_f32];
        let r = ReductionKernel::min(&d).unwrap();
        assert!(approx(r.value, 42.0));
        assert_eq!(r.index, 0);
    }

    // ── Axis reduction shape invariants ───────────────────────

    #[test]
    fn reduce_axis_row_output_length() {
        let rows = 5;
        let cols = 3;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let out = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
        assert_eq!(out.len(), rows);
        let max_out = ReductionKernel::max_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
        assert_eq!(max_out.len(), rows);
        let min_out = ReductionKernel::min_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
        assert_eq!(min_out.len(), rows);
    }

    #[test]
    fn reduce_axis_col_output_length() {
        let rows = 4;
        let cols = 7;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let out = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Column).unwrap();
        assert_eq!(out.len(), cols);
        let max_out = ReductionKernel::max_axis(&data, rows, cols, ReductionAxis::Column).unwrap();
        assert_eq!(max_out.len(), cols);
        let min_out = ReductionKernel::min_axis(&data, rows, cols, ReductionAxis::Column).unwrap();
        assert_eq!(min_out.len(), cols);
    }

    #[test]
    fn reduce_axis_1x1_matrix() {
        let data = [99.0_f32];
        let row_sum = ReductionKernel::sum_axis(&data, 1, 1, ReductionAxis::Row).unwrap();
        assert!(approx_vec(&row_sum, &[99.0]));
        let col_sum = ReductionKernel::sum_axis(&data, 1, 1, ReductionAxis::Column).unwrap();
        assert!(approx_vec(&col_sum, &[99.0]));
    }

    #[test]
    fn reduce_axis_mean_consistent_with_sum() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (rows, cols) = (2, 3);
        let sums = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
        let means = ReductionKernel::mean_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
        for (s, m) in sums.iter().zip(means.iter()) {
            assert!(approx(*m, *s / cols as f32));
        }
    }

    // ── Duplicate max/min returns first index ─────────────────

    #[test]
    fn reduce_max_tie_returns_last() {
        // max_by returns last element for ties
        let d = [1.0, 5.0, 5.0, 2.0];
        let r = ReductionKernel::max(&d).unwrap();
        assert!(approx(r.value, 5.0));
        assert_eq!(r.index, 2);
    }

    #[test]
    fn reduce_min_tie_returns_first() {
        // min_by returns first element for ties
        let d = [3.0, 1.0, 1.0, 5.0];
        let r = ReductionKernel::min(&d).unwrap();
        assert!(approx(r.value, 1.0));
        assert_eq!(r.index, 1);
    }
}

// ── Property tests ────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1e6_f32..1e6_f32, min_len..=max_len)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        // sum >= min * len for non-negative data
        #[test]
        fn prop_reduce_sum_ge_min_times_len(data in finite_vec(1, 128)) {
            let s = ReductionKernel::sum(&data).unwrap();
            let mn = ReductionKernel::min(&data).unwrap().value;
            let n = data.len() as f32;
            // sum >= min * n  (with tolerance for floating point)
            prop_assert!(s >= mn * n - 1e-3);
        }

        // max >= mean >= min
        #[test]
        fn prop_reduce_max_ge_mean_ge_min(data in finite_vec(1, 128)) {
            let mx = ReductionKernel::max(&data).unwrap().value;
            let avg = ReductionKernel::mean(&data).unwrap();
            let mn = ReductionKernel::min(&data).unwrap().value;
            prop_assert!(mx >= avg - 1e-5, "max={mx} < mean={avg}");
            prop_assert!(avg >= mn - 1e-5, "mean={avg} < min={mn}");
        }

        // argmax index is valid and points to the max value
        #[test]
        fn prop_reduce_argmax_valid(data in finite_vec(1, 128)) {
            let r = ReductionKernel::max(&data).unwrap();
            prop_assert!(r.index < data.len());
            prop_assert_eq!(data[r.index].to_bits(), r.value.to_bits());
        }

        // argmin index is valid and points to the min value
        #[test]
        fn prop_reduce_argmin_valid(data in finite_vec(1, 128)) {
            let r = ReductionKernel::min(&data).unwrap();
            prop_assert!(r.index < data.len());
            prop_assert_eq!(data[r.index].to_bits(), r.value.to_bits());
        }

        // mean equals sum / len
        #[test]
        fn prop_reduce_mean_equals_sum_div_len(data in finite_vec(1, 128)) {
            let s = ReductionKernel::sum(&data).unwrap();
            let m = ReductionKernel::mean(&data).unwrap();
            let expected = s / data.len() as f32;
            prop_assert!((m - expected).abs() < 1e-4,
                "mean={m} vs sum/n={expected}");
        }

        // L1 norm >= L2 norm is NOT always true, but L2 <= L1 for any vector
        // Actually: L2 <= L1 (triangle inequality in component form)
        #[test]
        fn prop_reduce_l2_le_l1(data in finite_vec(1, 64)) {
            let l1 = ReductionKernel::l1_norm(&data).unwrap();
            let l2 = ReductionKernel::l2_norm(&data).unwrap();
            prop_assert!(l2 <= l1 + 1e-4,
                "L2={l2} > L1={l1}");
        }

        // L1 norm is non-negative
        #[test]
        fn prop_reduce_l1_non_negative(data in finite_vec(1, 64)) {
            let l1 = ReductionKernel::l1_norm(&data).unwrap();
            prop_assert!(l1 >= 0.0);
        }

        // L2 norm is non-negative
        #[test]
        fn prop_reduce_l2_non_negative(data in finite_vec(1, 64)) {
            let l2 = ReductionKernel::l2_norm(&data).unwrap();
            prop_assert!(l2 >= 0.0);
        }

        // Row-axis sum output length equals number of rows
        #[test]
        fn prop_reduce_axis_row_shape(
            rows in 1_usize..=8,
            cols in 1_usize..=8,
        ) {
            let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let out = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
            prop_assert_eq!(out.len(), rows);
        }

        // Column-axis sum output length equals number of columns
        #[test]
        fn prop_reduce_axis_col_shape(
            rows in 1_usize..=8,
            cols in 1_usize..=8,
        ) {
            let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let out = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Column).unwrap();
            prop_assert_eq!(out.len(), cols);
        }

        // Full sum equals sum of row sums
        #[test]
        fn prop_reduce_row_sums_equal_total(
            rows in 1_usize..=8,
            cols in 1_usize..=8,
        ) {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
            let total = ReductionKernel::sum(&data).unwrap();
            let row_sums = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
            let recomposed: f32 = row_sums.iter().sum();
            prop_assert!((total - recomposed).abs() < 1e-3,
                "total={total} vs row_sums.sum()={recomposed}");
        }

        // Full sum equals sum of column sums
        #[test]
        fn prop_reduce_col_sums_equal_total(
            rows in 1_usize..=8,
            cols in 1_usize..=8,
        ) {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
            let total = ReductionKernel::sum(&data).unwrap();
            let col_sums = ReductionKernel::sum_axis(&data, rows, cols, ReductionAxis::Column).unwrap();
            let recomposed: f32 = col_sums.iter().sum();
            prop_assert!((total - recomposed).abs() < 1e-3,
                "total={total} vs col_sums.sum()={recomposed}");
        }

        // Per-row max >= per-row mean >= per-row min
        #[test]
        fn prop_reduce_axis_row_max_ge_mean_ge_min(
            rows in 1_usize..=6,
            cols in 1_usize..=6,
        ) {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) - 5.0).collect();
            let maxes = ReductionKernel::max_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
            let means = ReductionKernel::mean_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
            let mins = ReductionKernel::min_axis(&data, rows, cols, ReductionAxis::Row).unwrap();
            for r in 0..rows {
                prop_assert!(maxes[r].value >= means[r] - 1e-5);
                prop_assert!(means[r] >= mins[r].value - 1e-5);
            }
        }

        // sum of single element equals that element
        #[test]
        fn sum_of_single_element(x in -1e6f32..1e6) {
            let result = ReductionKernel::sum(&[x]).unwrap();
            prop_assert!(
                (result - x).abs() < 1e-6,
                "sum([{x}]) = {result}, expected {x}"
            );
        }

        // mean is bounded by min and max of input
        #[test]
        fn mean_between_min_and_max(
            xs in prop::collection::vec(-1e4f32..1e4, 2..256)
        ) {
            let mean = ReductionKernel::mean(&xs).unwrap();
            let min_val = xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val =
                xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                mean >= min_val && mean <= max_val,
                "mean={mean} not in [{min_val}, {max_val}]"
            );
        }

        // max is >= every element
        #[test]
        fn max_geq_all_elements(
            xs in prop::collection::vec(-1e6f32..1e6, 1..256)
        ) {
            let result = ReductionKernel::max(&xs).unwrap();
            for (i, &x) in xs.iter().enumerate() {
                prop_assert!(
                    result.value >= x,
                    "max={} < xs[{i}]={x}",
                    result.value
                );
            }
        }

        // min is <= every element
        #[test]
        fn min_leq_all_elements(
            xs in prop::collection::vec(-1e6f32..1e6, 1..256)
        ) {
            let result = ReductionKernel::min(&xs).unwrap();
            for (i, &x) in xs.iter().enumerate() {
                prop_assert!(
                    result.value <= x,
                    "min={} > xs[{i}]={x}",
                    result.value
                );
            }
        }

        // L2 norm >= L_inf norm
        #[test]
        fn l2_norm_geq_l_inf(
            xs in prop::collection::vec(-1e3f32..1e3, 1..128)
        ) {
            let l2 = ReductionKernel::l2_norm(&xs).unwrap();
            let l_inf = xs.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            prop_assert!(l2 >= l_inf - 1e-5, "l2={l2} < l_inf={l_inf}");
        }
    }
}
