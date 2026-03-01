//! CPU tensor concatenation, split, and stack operations.
//!
//! Provides `concat`, `split`, and `stack` on contiguous row-major `f32`
//! slices.  These are the building blocks for multi-head attention
//! (splitting/concatenating heads) and residual connections.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ── Kernel ─────────────────────────────────────────────────────────

/// Stateless dispatcher for CPU concat / split / stack operations.
pub struct ConcatKernel;

impl ConcatKernel {
    // ── concat ─────────────────────────────────────────────────

    /// Concatenate tensors along `axis`.
    ///
    /// Each element of `inputs` is a flat row-major buffer whose logical
    /// shape matches on every dimension **except** `axis`.  The `shapes`
    /// slice carries the full shape of each input so that the `axis`
    /// dimension may differ across inputs.
    ///
    /// Returns the concatenated buffer in row-major order.
    pub fn concat(inputs: &[&[f32]], shapes: &[&[usize]], axis: usize) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        if inputs.len() != shapes.len() {
            return Err(invalid_args("inputs and shapes must have the same length"));
        }

        let ndim = shapes[0].len();
        if ndim == 0 {
            return Err(invalid_args("shape must not be empty"));
        }
        if axis >= ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for {ndim}-dimensional tensor"
            )));
        }

        for (i, (&input, &shape)) in inputs.iter().zip(shapes.iter()).enumerate() {
            if shape.len() != ndim {
                return Err(invalid_args(&format!(
                    "input {i} has {} dimensions, expected {ndim}",
                    shape.len()
                )));
            }
            if input.len() != shape_numel(shape) {
                return Err(invalid_args(&format!(
                    "input {i} data length {} != product of shape {:?}",
                    input.len(),
                    shape
                )));
            }
            for d in 0..ndim {
                if d != axis && shape[d] != shapes[0][d] {
                    return Err(invalid_args(&format!(
                        "input {i} dim {d} is {}, expected {}",
                        shape[d], shapes[0][d]
                    )));
                }
            }
        }

        let total: usize = inputs.iter().map(|s| s.len()).sum();
        let mut out = Vec::with_capacity(total);

        let outer_size: usize = shapes[0][..axis].iter().product();
        let inner_size: usize = shapes[0][axis + 1..].iter().product();

        for outer in 0..outer_size {
            for (input, shape) in inputs.iter().zip(shapes.iter()) {
                let axis_len = shape[axis];
                let chunk = axis_len * inner_size;
                let start = outer * chunk;
                out.extend_from_slice(&input[start..start + chunk]);
            }
        }

        Ok(out)
    }

    // ── split ──────────────────────────────────────────────────

    /// Split a tensor along `axis` into `num_splits` equal parts.
    ///
    /// `shape[axis]` must be evenly divisible by `num_splits`.
    pub fn split(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        num_splits: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(invalid_args("shape must not be empty"));
        }
        if axis >= ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for {ndim}-dimensional tensor"
            )));
        }
        if num_splits == 0 {
            return Err(invalid_args("num_splits must be > 0"));
        }
        if !shape[axis].is_multiple_of(num_splits) {
            return Err(invalid_args(&format!(
                "shape[{axis}] = {} is not divisible by num_splits = {num_splits}",
                shape[axis]
            )));
        }
        if data.len() != shape_numel(shape) {
            return Err(invalid_args(&format!(
                "data length {} != product of shape {:?}",
                data.len(),
                shape
            )));
        }

        let split_axis_len = shape[axis] / num_splits;
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();
        let split_chunk = split_axis_len * inner_size;

        let elems_per_split = data.len() / num_splits;
        let mut results: Vec<Vec<f32>> =
            (0..num_splits).map(|_| Vec::with_capacity(elems_per_split)).collect();

        for outer in 0..outer_size {
            let row_start = outer * shape[axis] * inner_size;
            for (s, result) in results.iter_mut().enumerate().take(num_splits) {
                let start = row_start + s * split_chunk;
                result.extend_from_slice(&data[start..start + split_chunk]);
            }
        }

        Ok(results)
    }

    // ── split_sizes ────────────────────────────────────────────

    /// Split a tensor along `axis` into chunks whose axis-lengths are
    /// given by `sizes`.  The sum of `sizes` must equal `shape[axis]`.
    pub fn split_sizes(
        data: &[f32],
        shape: &[usize],
        axis: usize,
        sizes: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(invalid_args("shape must not be empty"));
        }
        if axis >= ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for {ndim}-dimensional tensor"
            )));
        }
        let size_sum: usize = sizes.iter().sum();
        if size_sum != shape[axis] {
            return Err(invalid_args(&format!(
                "sum of sizes ({size_sum}) != shape[{axis}] ({})",
                shape[axis]
            )));
        }
        if data.len() != shape_numel(shape) {
            return Err(invalid_args(&format!(
                "data length {} != product of shape {:?}",
                data.len(),
                shape
            )));
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        let mut results: Vec<Vec<f32>> =
            sizes.iter().map(|&s| Vec::with_capacity(outer_size * s * inner_size)).collect();

        for outer in 0..outer_size {
            let row_start = outer * shape[axis] * inner_size;
            let mut offset = 0;
            for (i, &sz) in sizes.iter().enumerate() {
                let chunk = sz * inner_size;
                let start = row_start + offset;
                results[i].extend_from_slice(&data[start..start + chunk]);
                offset += chunk;
            }
        }

        Ok(results)
    }

    // ── stack ──────────────────────────────────────────────────

    /// Stack tensors along a new dimension inserted at `axis`.
    ///
    /// All inputs must have identical shape.  The result has `ndim + 1`
    /// dimensions with the new axis of size `inputs.len()` at position
    /// `axis`.
    pub fn stack(inputs: &[&[f32]], shape: &[usize], axis: usize) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        let ndim = shape.len();
        if axis > ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for stack on {ndim}-dimensional tensor (max {ndim})"
            )));
        }

        let numel = shape_numel(shape);
        for (i, &input) in inputs.iter().enumerate() {
            if input.len() != numel {
                return Err(invalid_args(&format!(
                    "input {i} length {} != expected {numel}",
                    input.len()
                )));
            }
        }

        let total = inputs.len() * numel;
        let mut out = Vec::with_capacity(total);

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis..].iter().product();

        for outer in 0..outer_size {
            for input in inputs {
                let start = outer * inner_size;
                out.extend_from_slice(&input[start..start + inner_size]);
            }
        }

        Ok(out)
    }

    /// Compute the output shape of `stack` without performing it.
    pub fn stack_output_shape(shape: &[usize], axis: usize, n: usize) -> Result<Vec<usize>> {
        if axis > shape.len() {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for stack on {}-dimensional tensor",
                shape.len()
            )));
        }
        let mut out = Vec::with_capacity(shape.len() + 1);
        out.extend_from_slice(&shape[..axis]);
        out.push(n);
        out.extend_from_slice(&shape[axis..]);
        Ok(out)
    }

    /// Compute the output shape of `concat`.
    pub fn concat_output_shape(shapes: &[&[usize]], axis: usize) -> Result<Vec<usize>> {
        if shapes.is_empty() {
            return Ok(vec![]);
        }
        let ndim = shapes[0].len();
        if axis >= ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for {ndim}-dimensional tensor"
            )));
        }
        let axis_total: usize = shapes.iter().map(|s| s[axis]).sum();
        let mut out = shapes[0].to_vec();
        out[axis] = axis_total;
        Ok(out)
    }

    /// Compute the output shapes of `split`.
    pub fn split_output_shapes(
        shape: &[usize],
        axis: usize,
        num_splits: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let ndim = shape.len();
        if axis >= ndim {
            return Err(invalid_args(&format!(
                "axis {axis} out of range for {ndim}-dimensional tensor"
            )));
        }
        if num_splits == 0 || !shape[axis].is_multiple_of(num_splits) {
            return Err(invalid_args("shape[axis] must be divisible by num_splits"));
        }
        let mut s = shape.to_vec();
        s[axis] = shape[axis] / num_splits;
        Ok(vec![s; num_splits])
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert `&[Vec<usize>]` → `Vec<&[usize]>` for shape args.
    fn s(vecs: &[Vec<usize>]) -> Vec<&[usize]> {
        vecs.iter().map(|v| v.as_slice()).collect()
    }

    // ── concat ─────────────────────────────────────────────────

    #[test]
    fn concat_1d() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let shapes = [vec![3usize], vec![2]];
        let out = ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn concat_2d_axis0() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0];
        let shapes = [vec![2usize, 2], vec![1, 2]];
        let out = ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn concat_2d_axis1() {
        let a = [1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = [5.0, 6.0]; // [2,1]
        let shapes = [vec![2usize, 2], vec![2, 1]];
        let out = ConcatKernel::concat(&[&a, &b], &s(&shapes), 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn concat_3d_axis1() {
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // [2,1,3]
        let b: Vec<f32> = (7..=18).map(|x| x as f32).collect(); // [2,2,3]
        let shapes = [vec![2usize, 1, 3], vec![2, 2, 3]];
        let out = ConcatKernel::concat(&[&a, &b], &s(&shapes), 1).unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0
            ]
        );
    }

    #[test]
    fn concat_single_tensor() {
        let a = [1.0, 2.0, 3.0];
        let shapes = [vec![3usize]];
        let out = ConcatKernel::concat(&[&a], &s(&shapes), 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn concat_empty_inputs() {
        let out = ConcatKernel::concat(&[], &[], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn concat_axis_out_of_range() {
        let a = [1.0, 2.0];
        let shapes = [vec![2usize]];
        assert!(ConcatKernel::concat(&[&a], &s(&shapes), 1).is_err());
    }

    #[test]
    fn concat_shape_mismatch() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let shapes = [vec![2usize, 2], vec![1, 3]];
        assert!(ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).is_err());
    }

    #[test]
    fn concat_ndim_mismatch() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0, 6.0];
        let shapes = [vec![2usize], vec![2, 2]];
        assert!(ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).is_err());
    }

    #[test]
    fn concat_data_length_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let shapes = [vec![2usize, 2]];
        assert!(ConcatKernel::concat(&[&a], &s(&shapes), 0).is_err());
    }

    // ── split ──────────────────────────────────────────────────

    #[test]
    fn split_1d() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let parts = ConcatKernel::split(&data, &[6], 0, 3).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], vec![1.0, 2.0]);
        assert_eq!(parts[1], vec![3.0, 4.0]);
        assert_eq!(parts[2], vec![5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis0() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3,2]
        let parts = ConcatKernel::split(&data, &[3, 2], 0, 3).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], vec![1.0, 2.0]);
        assert_eq!(parts[1], vec![3.0, 4.0]);
        assert_eq!(parts[2], vec![5.0, 6.0]);
    }

    #[test]
    fn split_2d_axis1() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2,4]
        let parts = ConcatKernel::split(&data, &[2, 4], 1, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(parts[1], vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn split_single() {
        let data = [1.0, 2.0, 3.0];
        let parts = ConcatKernel::split(&data, &[3], 0, 1).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_not_divisible() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(ConcatKernel::split(&data, &[5], 0, 2).is_err());
    }

    #[test]
    fn split_zero_splits() {
        let data = [1.0, 2.0];
        assert!(ConcatKernel::split(&data, &[2], 0, 0).is_err());
    }

    #[test]
    fn split_axis_out_of_range() {
        let data = [1.0, 2.0];
        assert!(ConcatKernel::split(&data, &[2], 1, 1).is_err());
    }

    #[test]
    fn split_data_length_mismatch() {
        let data = [1.0, 2.0, 3.0];
        assert!(ConcatKernel::split(&data, &[2, 2], 0, 2).is_err());
    }

    // ── split_sizes ────────────────────────────────────────────

    #[test]
    fn split_sizes_1d() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let parts = ConcatKernel::split_sizes(&data, &[5], 0, &[2, 3]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], vec![1.0, 2.0]);
        assert_eq!(parts[1], vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn split_sizes_2d_axis1() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let parts = ConcatKernel::split_sizes(&data, &[2, 3], 1, &[1, 2]).unwrap();
        assert_eq!(parts[0], vec![1.0, 4.0]);
        assert_eq!(parts[1], vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn split_sizes_sum_mismatch() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(ConcatKernel::split_sizes(&data, &[5], 0, &[2, 2]).is_err());
    }

    // ── concat / split roundtrip ───────────────────────────────

    #[test]
    fn concat_split_roundtrip_1d() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let shapes = [vec![3usize], vec![3]];
        let cat = ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).unwrap();
        let parts = ConcatKernel::split(&cat, &[6], 0, 2).unwrap();
        assert_eq!(parts[0], a);
        assert_eq!(parts[1], b);
    }

    #[test]
    fn concat_split_roundtrip_2d_axis0() {
        let a = [1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = [5.0, 6.0, 7.0, 8.0]; // [2,2]
        let shapes = [vec![2usize, 2], vec![2, 2]];
        let cat = ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).unwrap();
        let parts = ConcatKernel::split(&cat, &[4, 2], 0, 2).unwrap();
        assert_eq!(parts[0], a);
        assert_eq!(parts[1], b);
    }

    #[test]
    fn concat_split_roundtrip_2d_axis1() {
        let a = [1.0, 2.0, 5.0, 6.0]; // [2,2]
        let b = [3.0, 4.0, 7.0, 8.0]; // [2,2]
        let shapes = [vec![2usize, 2], vec![2, 2]];
        let cat = ConcatKernel::concat(&[&a, &b], &s(&shapes), 1).unwrap();
        assert_eq!(cat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let parts = ConcatKernel::split(&cat, &[2, 4], 1, 2).unwrap();
        assert_eq!(parts[0], a);
        assert_eq!(parts[1], b);
    }

    // ── stack ──────────────────────────────────────────────────

    #[test]
    fn stack_1d_axis0() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let out = ConcatKernel::stack(&[&a, &b], &[3], 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_1d_axis1() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let out = ConcatKernel::stack(&[&a, &b], &[3], 1).unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn stack_2d_axis0() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = ConcatKernel::stack(&[&a, &b], &[2, 2], 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn stack_2d_axis1() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = ConcatKernel::stack(&[&a, &b], &[2, 2], 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn stack_2d_axis2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let out = ConcatKernel::stack(&[&a, &b], &[2, 2], 2).unwrap();
        assert_eq!(out, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn stack_single_tensor() {
        let a = [1.0, 2.0];
        let out = ConcatKernel::stack(&[&a], &[2], 0).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn stack_empty_inputs() {
        let out = ConcatKernel::stack(&[], &[3], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn stack_axis_out_of_range() {
        let a = [1.0, 2.0];
        assert!(ConcatKernel::stack(&[&a], &[2], 2).is_err());
    }

    #[test]
    fn stack_length_mismatch() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        assert!(ConcatKernel::stack(&[&a, &b], &[2], 0).is_err());
    }

    // ── output shape helpers ───────────────────────────────────

    #[test]
    fn concat_output_shape_basic() {
        let shapes = [vec![2usize, 3], vec![2, 5]];
        let refs = s(&shapes);
        let out = ConcatKernel::concat_output_shape(&refs, 1).unwrap();
        assert_eq!(out, vec![2, 8]);
    }

    #[test]
    fn stack_output_shape_basic() {
        let out = ConcatKernel::stack_output_shape(&[3, 4], 1, 5).unwrap();
        assert_eq!(out, vec![3, 5, 4]);
    }

    #[test]
    fn split_output_shapes_basic() {
        let out = ConcatKernel::split_output_shapes(&[4, 6], 1, 3).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], vec![4, 2]);
    }

    // ── element count preservation ─────────────────────────────

    #[test]
    fn concat_preserves_element_count() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0];
        let shapes = [vec![2usize, 2], vec![1, 2]];
        let out = ConcatKernel::concat(&[&a, &b], &s(&shapes), 0).unwrap();
        assert_eq!(out.len(), a.len() + b.len());
    }

    #[test]
    fn split_preserves_element_count() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let parts = ConcatKernel::split(&data, &[2, 3, 4], 1, 3).unwrap();
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn stack_preserves_element_count() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = [7.0, 8.0, 9.0];
        let out = ConcatKernel::stack(&[&a, &b, &c], &[3], 0).unwrap();
        assert_eq!(out.len(), 9);
    }

    // ── property tests ─────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_2d_pair() -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
            (1..8usize, 1..8usize).prop_flat_map(|(r, c)| {
                let n = r * c;
                (proptest::collection::vec(-100.0f32..100.0, n), Just(r), Just(c))
            })
        }

        proptest! {
            #[test]
            fn concat_split_roundtrip_axis0(
                (a_data, a_rows, cols) in arb_2d_pair(),
                b_rows in 1..8usize,
            ) {
                let b_data: Vec<f32> = (0..b_rows * cols).map(|i| i as f32).collect();
                let sa = vec![a_rows, cols];
                let sb = vec![b_rows, cols];
                let cat = ConcatKernel::concat(
                    &[a_data.as_slice(), b_data.as_slice()],
                    &[sa.as_slice(), sb.as_slice()],
                    0,
                ).unwrap();
                let total_rows = a_rows + b_rows;
                let parts = ConcatKernel::split_sizes(
                    &cat, &[total_rows, cols], 0, &[a_rows, b_rows],
                ).unwrap();
                prop_assert_eq!(&parts[0], &a_data);
                prop_assert_eq!(&parts[1], &b_data);
            }

            #[test]
            fn concat_split_roundtrip_axis1(
                (a_data, rows, a_cols) in arb_2d_pair(),
                b_cols in 1..8usize,
            ) {
                let b_data: Vec<f32> = (0..rows * b_cols).map(|i| i as f32).collect();
                let sa = vec![rows, a_cols];
                let sb = vec![rows, b_cols];
                let cat = ConcatKernel::concat(
                    &[a_data.as_slice(), b_data.as_slice()],
                    &[sa.as_slice(), sb.as_slice()],
                    1,
                ).unwrap();
                let total_cols = a_cols + b_cols;
                let parts = ConcatKernel::split_sizes(
                    &cat, &[rows, total_cols], 1, &[a_cols, b_cols],
                ).unwrap();
                prop_assert_eq!(&parts[0], &a_data);
                prop_assert_eq!(&parts[1], &b_data);
            }

            #[test]
            fn split_preserves_total_elements(
                rows in 1..8usize,
                cols in 1..8usize,
                num_splits in 1..5usize,
            ) {
                let adjusted_rows = rows * num_splits;
                let data: Vec<f32> = (0..adjusted_rows * cols)
                    .map(|i| i as f32).collect();
                let parts = ConcatKernel::split(
                    &data, &[adjusted_rows, cols], 0, num_splits,
                ).unwrap();
                let total: usize = parts.iter().map(|p| p.len()).sum();
                prop_assert_eq!(total, data.len());
                prop_assert_eq!(parts.len(), num_splits);
            }

            #[test]
            fn stack_element_count(
                (data, rows, cols) in arb_2d_pair(),
                n_extra in 1..5usize,
            ) {
                let n = n_extra + 1;
                let inputs: Vec<&[f32]> =
                    std::iter::repeat_n(data.as_slice(), n).collect();
                let out = ConcatKernel::stack(&inputs, &[rows, cols], 0).unwrap();
                prop_assert_eq!(out.len(), data.len() * n);
            }
        }
    }
}
