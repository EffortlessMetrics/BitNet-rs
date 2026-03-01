//! CPU tensor transpose and reshape operations kernel.
//!
//! Provides transpose, reshape, and shape manipulation operations on
//! contiguous `f32` slices.  Supports 2-D matrix transpose, N-dimensional
//! transpose with arbitrary permutation, reshape, dimension flattening,
//! squeeze/unsqueeze, and stride computation.

use bitnet_common::tensor_validation::c_contiguous_strides;
use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ── Kernel ─────────────────────────────────────────────────────────

/// Stateless dispatcher for CPU transpose and reshape operations.
pub struct TransposeKernel;

impl TransposeKernel {
    // ── 2-D transpose ──────────────────────────────────────────

    /// Standard row-major 2-D matrix transpose.
    ///
    /// Given a `rows × cols` matrix stored in row-major order, returns
    /// a `cols × rows` matrix where `out[j * rows + i] = data[i * cols + j]`.
    ///
    /// Returns `Ok(vec![])` when `rows * cols == 0`.
    pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let numel = rows * cols;
        if data.len() != numel {
            return Err(invalid_args("data length must equal rows * cols"));
        }
        if numel == 0 {
            return Ok(vec![]);
        }

        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = data[i * cols + j];
            }
        }
        Ok(out)
    }

    // ── N-dimensional transpose ────────────────────────────────

    /// N-dimensional transpose with an arbitrary axis permutation.
    ///
    /// `perm` must be a permutation of `0..shape.len()`.  The output
    /// tensor has shape `[shape[perm[0]], shape[perm[1]], ...]` and its
    /// elements are rearranged accordingly.
    pub fn transpose_nd(data: &[f32], shape: &[usize], perm: &[usize]) -> Result<Vec<f32>> {
        let ndim = shape.len();
        if perm.len() != ndim {
            return Err(invalid_args("perm length must equal number of dimensions"));
        }

        // Validate perm is a valid permutation of 0..ndim.
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim {
                return Err(invalid_args("perm contains out-of-range index"));
            }
            if seen[p] {
                return Err(invalid_args("perm contains duplicate index"));
            }
            seen[p] = true;
        }

        let numel = shape_numel(shape);
        if data.len() != numel {
            return Err(invalid_args("data length must equal product of shape dimensions"));
        }
        if numel == 0 {
            return Ok(vec![]);
        }

        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let dst_strides = c_contiguous_strides(&out_shape);

        let mut out = vec![0.0f32; numel];
        let mut src_idx = vec![0usize; ndim];

        // Iterate using an incrementing multi-index to avoid per-element div/mod.
        for &val in &data[..numel] {
            let mut dst_flat = 0usize;
            for d in 0..ndim {
                dst_flat += src_idx[perm[d]] * dst_strides[d];
            }

            out[dst_flat] = val;

            // Increment source multi-index like an odometer.
            for d in (0..ndim).rev() {
                src_idx[d] += 1;
                if src_idx[d] < shape[d] {
                    break;
                }
                src_idx[d] = 0;
            }
        }
        Ok(out)
    }

    // ── Reshape ────────────────────────────────────────────────

    /// Reshape with validation: checks that old and new shapes have the
    /// same number of elements, then returns a cloned buffer with the
    /// same memory layout and a new logical shape.
    pub fn reshape(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
        let old_numel = shape_numel(old_shape);
        let new_numel = shape_numel(new_shape);

        if data.len() != old_numel {
            return Err(invalid_args("data length must match product of old_shape"));
        }
        if old_numel != new_numel {
            return Err(invalid_args("old and new shapes must have the same number of elements"));
        }

        Ok(data.to_vec())
    }

    // ── Flatten ────────────────────────────────────────────────

    /// Flatten contiguous dimensions `[start_dim, end_dim]` (inclusive)
    /// into a single dimension.
    ///
    /// Returns a cloned `Vec<f32>` containing the same elements in the
    /// same memory order as `data`, together with the new logical shape.
    pub fn flatten(
        data: &[f32],
        shape: &[usize],
        start_dim: usize,
        end_dim: usize,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(invalid_args("shape must not be empty"));
        }
        if start_dim >= ndim || end_dim >= ndim {
            return Err(invalid_args("start_dim and end_dim must be < number of dimensions"));
        }
        if start_dim > end_dim {
            return Err(invalid_args("start_dim must be <= end_dim"));
        }
        if data.len() != shape_numel(shape) {
            return Err(invalid_args("data length must match product of shape"));
        }

        let flat_size: usize = shape[start_dim..=end_dim].iter().product();
        let mut new_shape = Vec::with_capacity(ndim - (end_dim - start_dim));
        new_shape.extend_from_slice(&shape[..start_dim]);
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&shape[end_dim + 1..]);

        Ok((data.to_vec(), new_shape))
    }

    // ── Squeeze / Unsqueeze ────────────────────────────────────

    /// Remove all size-1 dimensions from `shape`.
    ///
    /// If all dimensions are 1 (or `shape` is empty), the result is `[]` (a scalar shape).
    pub fn squeeze(shape: &[usize]) -> Vec<usize> {
        shape.iter().copied().filter(|&d| d != 1).collect()
    }

    /// Insert a size-1 dimension at position `dim`.
    ///
    /// `dim` may range from `0` to `shape.len()` (inclusive).
    pub fn unsqueeze(shape: &[usize], dim: usize) -> Result<Vec<usize>> {
        if dim > shape.len() {
            return Err(invalid_args("dim must be <= number of dimensions"));
        }
        let mut out = Vec::with_capacity(shape.len() + 1);
        out.extend_from_slice(&shape[..dim]);
        out.push(1);
        out.extend_from_slice(&shape[dim..]);
        Ok(out)
    }

    // ── Stride utilities ───────────────────────────────────────

    /// Compute row-major (C-contiguous) strides for the given shape.
    pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        c_contiguous_strides(shape)
    }

    /// Check whether `strides` matches the contiguous layout for `shape`.
    pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
        if shape.len() != strides.len() {
            return false;
        }
        let expected = c_contiguous_strides(shape);
        strides == expected.as_slice()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── transpose_2d ───────────────────────────────────────────

    #[test]
    fn transpose_2d_basic() {
        // [[1,2,3],[4,5,6]]  →  [[1,4],[2,5],[3,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = TransposeKernel::transpose_2d(&data, 2, 3).unwrap();
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_square() {
        // [[1,2],[3,4]]  →  [[1,3],[2,4]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = TransposeKernel::transpose_2d(&data, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_2d_single_row() {
        let data = vec![1.0, 2.0, 3.0];
        let out = TransposeKernel::transpose_2d(&data, 1, 3).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn transpose_2d_single_col() {
        let data = vec![1.0, 2.0, 3.0];
        let out = TransposeKernel::transpose_2d(&data, 3, 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn transpose_2d_single_element() {
        let out = TransposeKernel::transpose_2d(&[42.0], 1, 1).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn transpose_2d_zero_rows() {
        let out = TransposeKernel::transpose_2d(&[], 0, 3).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn transpose_2d_zero_cols() {
        let out = TransposeKernel::transpose_2d(&[], 3, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn transpose_2d_length_mismatch() {
        assert!(TransposeKernel::transpose_2d(&[1.0, 2.0], 2, 3).is_err());
    }

    #[test]
    fn transpose_2d_involution() {
        // Transposing twice yields the original.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = TransposeKernel::transpose_2d(&data, 2, 3).unwrap();
        let tt = TransposeKernel::transpose_2d(&t, 3, 2).unwrap();
        assert_eq!(tt, data);
    }

    // ── transpose_nd ───────────────────────────────────────────

    #[test]
    fn transpose_nd_2d_matches_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let nd = TransposeKernel::transpose_nd(&data, &[2, 3], &[1, 0]).unwrap();
        let flat = TransposeKernel::transpose_2d(&data, 2, 3).unwrap();
        assert_eq!(nd, flat);
    }

    #[test]
    fn transpose_nd_identity_perm() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = TransposeKernel::transpose_nd(&data, &[2, 3], &[0, 1]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn transpose_nd_3d() {
        // Shape [2,3,1], perm [2,0,1] → shape [1,2,3]
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let out = TransposeKernel::transpose_nd(&data, &[2, 3, 1], &[2, 0, 1]).unwrap();
        // With trailing dim=1, the data order is preserved under this perm.
        assert_eq!(out, data);
    }

    #[test]
    fn transpose_nd_3d_swap_first_two() {
        // Shape [2,3,2], perm [1,0,2] swaps first two axes.
        // Input as 2×3×2:
        //   [[[0,1],[2,3],[4,5]],
        //    [[6,7],[8,9],[10,11]]]
        // Output 3×2×2:
        //   [[[0,1],[6,7]],
        //    [[2,3],[8,9]],
        //    [[4,5],[10,11]]]
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let out = TransposeKernel::transpose_nd(&data, &[2, 3, 2], &[1, 0, 2]).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn transpose_nd_empty_tensor() {
        let out = TransposeKernel::transpose_nd(&[], &[0, 3], &[1, 0]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn transpose_nd_perm_wrong_length() {
        assert!(
            TransposeKernel::transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0, 1, 2],)
                .is_err()
        );
    }

    #[test]
    fn transpose_nd_perm_duplicate() {
        assert!(
            TransposeKernel::transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0, 0],)
                .is_err()
        );
    }

    #[test]
    fn transpose_nd_perm_out_of_range() {
        assert!(
            TransposeKernel::transpose_nd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &[0, 5],)
                .is_err()
        );
    }

    #[test]
    fn transpose_nd_data_length_mismatch() {
        assert!(TransposeKernel::transpose_nd(&[1.0, 2.0], &[2, 3], &[1, 0],).is_err());
    }

    // ── reshape ────────────────────────────────────────────────

    #[test]
    fn reshape_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = TransposeKernel::reshape(&data, &[2, 3], &[3, 2]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_flatten_to_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = TransposeKernel::reshape(&data, &[2, 2], &[4]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_expand_to_3d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = TransposeKernel::reshape(&data, &[6], &[1, 2, 3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reshape_incompatible_sizes() {
        assert!(TransposeKernel::reshape(&[1.0, 2.0, 3.0], &[3], &[2, 2],).is_err());
    }

    #[test]
    fn reshape_data_length_mismatch() {
        assert!(TransposeKernel::reshape(&[1.0, 2.0], &[3], &[3],).is_err());
    }

    // ── flatten ────────────────────────────────────────────────

    #[test]
    fn flatten_middle_dims() {
        // [2, 3, 4] flatten(1,2) → [2, 12]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let (out, new_shape) = TransposeKernel::flatten(&data, &[2, 3, 4], 1, 2).unwrap();
        assert_eq!(new_shape, vec![2, 12]);
        assert_eq!(out, data);
    }

    #[test]
    fn flatten_all_dims() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let (_, new_shape) = TransposeKernel::flatten(&data, &[2, 3, 4], 0, 2).unwrap();
        assert_eq!(new_shape, vec![24]);
    }

    #[test]
    fn flatten_single_dim_noop() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, new_shape) = TransposeKernel::flatten(&data, &[2, 3], 0, 0).unwrap();
        assert_eq!(new_shape, vec![2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn flatten_start_gt_end() {
        assert!(TransposeKernel::flatten(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], 1, 0,).is_err());
    }

    #[test]
    fn flatten_dim_out_of_range() {
        assert!(TransposeKernel::flatten(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], 0, 5,).is_err());
    }

    #[test]
    fn flatten_empty_shape() {
        assert!(TransposeKernel::flatten(&[], &[], 0, 0).is_err());
    }

    // ── squeeze ────────────────────────────────────────────────

    #[test]
    fn squeeze_removes_ones() {
        assert_eq!(TransposeKernel::squeeze(&[1, 3, 1, 4, 1]), vec![3, 4]);
    }

    #[test]
    fn squeeze_no_ones() {
        assert_eq!(TransposeKernel::squeeze(&[2, 3, 4]), vec![2, 3, 4]);
    }

    #[test]
    fn squeeze_all_ones() {
        assert_eq!(TransposeKernel::squeeze(&[1, 1, 1]), Vec::<usize>::new());
    }

    #[test]
    fn squeeze_empty_shape() {
        assert_eq!(TransposeKernel::squeeze(&[]), Vec::<usize>::new());
    }

    // ── unsqueeze ──────────────────────────────────────────────

    #[test]
    fn unsqueeze_front() {
        assert_eq!(TransposeKernel::unsqueeze(&[2, 3], 0).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn unsqueeze_middle() {
        assert_eq!(TransposeKernel::unsqueeze(&[2, 3], 1).unwrap(), vec![2, 1, 3]);
    }

    #[test]
    fn unsqueeze_end() {
        assert_eq!(TransposeKernel::unsqueeze(&[2, 3], 2).unwrap(), vec![2, 3, 1]);
    }

    #[test]
    fn unsqueeze_out_of_range() {
        assert!(TransposeKernel::unsqueeze(&[2, 3], 5).is_err());
    }

    #[test]
    fn unsqueeze_empty_shape() {
        assert_eq!(TransposeKernel::unsqueeze(&[], 0).unwrap(), vec![1]);
    }

    // ── contiguous_strides ─────────────────────────────────────

    #[test]
    fn strides_2d() {
        assert_eq!(TransposeKernel::contiguous_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn strides_3d() {
        assert_eq!(TransposeKernel::contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn strides_1d() {
        assert_eq!(TransposeKernel::contiguous_strides(&[5]), vec![1]);
    }

    #[test]
    fn strides_empty_shape() {
        let s = TransposeKernel::contiguous_strides(&[]);
        assert!(s.is_empty());
    }

    // ── is_contiguous ──────────────────────────────────────────

    #[test]
    fn is_contiguous_true() {
        assert!(TransposeKernel::is_contiguous(&[2, 3, 4], &[12, 4, 1],));
    }

    #[test]
    fn is_contiguous_false() {
        assert!(!TransposeKernel::is_contiguous(&[2, 3, 4], &[12, 1, 3],));
    }

    #[test]
    fn is_contiguous_length_mismatch() {
        assert!(!TransposeKernel::is_contiguous(&[2, 3], &[6, 3, 1]));
    }

    #[test]
    fn is_contiguous_empty() {
        assert!(TransposeKernel::is_contiguous(&[], &[]));
    }
}
