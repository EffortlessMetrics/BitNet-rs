//! CPU transpose and reshape operations.
//!
//! Pure-Rust implementations of 2D/ND transpose and reshape for
//! correctness testing and non-GPU environments.  The CUDA-accelerated
//! versions live in [`crate::cuda::transpose`] (feature-gated behind
//! `gpu`/`cuda`).

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a transpose operation.
#[derive(Debug, Clone)]
pub struct TransposeConfig {
    /// Shape of the input tensor (e.g. `[rows, cols]` for 2D).
    pub shape: Vec<usize>,
    /// Permutation of axes.  For a 2D transpose this is `[1, 0]`.
    pub permutation: Vec<usize>,
}

impl TransposeConfig {
    /// Total number of elements described by `shape`.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Compute the output shape after applying the permutation.
    pub fn output_shape(&self) -> Vec<usize> {
        self.permutation.iter().map(|&p| self.shape[p]).collect()
    }

    /// Validate that the permutation is a valid bijection over `[0, ndim)`.
    pub fn validate(&self) -> Result<()> {
        let ndim = self.shape.len();
        if self.permutation.len() != ndim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "permutation length ({}) must equal shape rank ({})",
                    self.permutation.len(),
                    ndim
                ),
            }
            .into());
        }
        let mut seen = vec![false; ndim];
        for &p in &self.permutation {
            if p >= ndim {
                return Err(KernelError::InvalidArguments {
                    reason: format!("permutation index {p} out of range for rank {ndim}"),
                }
                .into());
            }
            if seen[p] {
                return Err(KernelError::InvalidArguments {
                    reason: format!("duplicate index {p} in permutation"),
                }
                .into());
            }
            seen[p] = true;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CPU implementations
// ---------------------------------------------------------------------------

/// Transpose a 2D row-major matrix (rows × cols) → (cols × rows).
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols, "data length {} != rows*cols {}", data.len(), rows * cols,);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Transpose an N-dimensional tensor according to a permutation.
///
/// `shape` describes the input dimensions and `perm` is a permutation of
/// `[0, 1, …, ndim-1]`.  Returns a new `Vec<f32>` with the transposed
/// data in row-major order.
pub fn transpose_nd(data: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    assert_eq!(perm.len(), ndim, "permutation rank mismatch");
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "data length mismatch");

    let in_strides = compute_strides(shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let out_strides = compute_strides(&out_shape);

    let mut out = vec![0.0f32; total];
    for (i, out_elem) in out.iter_mut().enumerate() {
        let mut remaining = i;
        let mut in_idx = 0usize;
        for d in 0..ndim {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            in_idx += coord * in_strides[perm[d]];
        }
        *out_elem = data[in_idx];
    }
    out
}

/// Reshape `data` from `old_shape` to `new_shape` (both must have the same
/// total element count).  This is a logical operation — the data is simply
/// returned in a new `Vec` with identical contents.
pub fn reshape(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Result<Vec<f32>> {
    let old_total: usize = old_shape.iter().product();
    let new_total: usize = new_shape.iter().product();
    if old_total != new_total {
        return Err(KernelError::InvalidArguments {
            reason: format!("reshape total mismatch: old={old_total}, new={new_total}"),
        }
        .into());
    }
    if data.len() != old_total {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} != old shape total {old_total}", data.len()),
        }
        .into());
    }
    Ok(data.to_vec())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute row-major strides for the given shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- 2D transpose -------------------------------------------------------

    #[test]
    fn test_transpose_2d_square() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = transpose_2d(&data, 3, 3);
        assert_eq!(out, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_transpose_2d_rect_wide() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_2d(&data, 2, 3);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_2d_rect_tall() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_2d(&data, 3, 2);
        assert_eq!(out, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_2d_single_row() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let out = transpose_2d(&data, 1, 4);
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_transpose_2d_single_col() {
        let data = vec![10.0, 20.0, 30.0];
        let out = transpose_2d(&data, 3, 1);
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_transpose_2d_1x1() {
        let out = transpose_2d(&[42.0], 1, 1);
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn test_transpose_2d_double_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = transpose_2d(&data, 2, 3);
        let t2 = transpose_2d(&t1, 3, 2);
        assert_eq!(t2, data);
    }

    // -- ND transpose -------------------------------------------------------

    #[test]
    fn test_transpose_nd_2d_equiv() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let nd = transpose_nd(&data, &[2, 3], &[1, 0]);
        let tw = transpose_2d(&data, 2, 3);
        assert_eq!(nd, tw);
    }

    #[test]
    fn test_transpose_nd_identity_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_nd(&data, &[2, 3], &[0, 1]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_transpose_nd_identity_3d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4], &[0, 1, 2]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_transpose_nd_3d_swap_last_two() {
        // (2,3,4) perm [0,2,1] → (2,4,3)
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4], &[0, 2, 1]);
        assert_eq!(out[0], 0.0);
        // out shape [2,4,3]; out[0][1][0] at linear index 3
        // → input coords: dim0=0, dim2=1, dim1=0 → data[0*12+0*4+1]=1.0
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn test_transpose_nd_3d_full_reverse() {
        // (2,3,4) perm [2,1,0] → (4,3,2)
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4], &[2, 1, 0]);
        assert_eq!(out[0], 0.0);
        // out[0][0][1] == data[1][0][0] == 12.0
        assert_eq!(out[1], 12.0);
    }

    #[test]
    fn test_transpose_nd_3d_roundtrip() {
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let perm = [2, 0, 1];
        let inv = [1, 2, 0];
        let t1 = transpose_nd(&data, &[3, 4, 5], &perm);
        let mid_shape: Vec<usize> = perm.iter().map(|&p| [3, 4, 5][p]).collect();
        let t2 = transpose_nd(&t1, &mid_shape, &inv);
        assert_eq!(t2, data);
    }

    #[test]
    fn test_transpose_nd_4d_identity() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4, 5], &[0, 1, 2, 3]);
        assert_eq!(out, data);
    }

    #[test]
    fn test_transpose_nd_4d_swap_middle() {
        // (2,3,4,5) perm [0,2,1,3] → (2,4,3,5)
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4, 5], &[0, 2, 1, 3]);
        assert_eq!(out[0], 0.0);
        assert_eq!(out.len(), 120);
        // Self-inverse roundtrip.
        let back = transpose_nd(&out, &[2, 4, 3, 5], &[0, 2, 1, 3]);
        assert_eq!(back, data);
    }

    #[test]
    fn test_transpose_nd_4d_full_reverse() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let out = transpose_nd(&data, &[2, 3, 4, 5], &[3, 2, 1, 0]);
        assert_eq!(out.len(), 120);
        let back = transpose_nd(&out, &[5, 4, 3, 2], &[3, 2, 1, 0]);
        assert_eq!(back, data);
    }

    // -- config validation --------------------------------------------------

    #[test]
    fn test_config_validate_ok() {
        let cfg = TransposeConfig { shape: vec![2, 3, 4], permutation: vec![2, 0, 1] };
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_validate_length_mismatch() {
        let cfg = TransposeConfig { shape: vec![2, 3], permutation: vec![0, 1, 2] };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_out_of_range() {
        let cfg = TransposeConfig { shape: vec![2, 3], permutation: vec![0, 5] };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_duplicate() {
        let cfg = TransposeConfig { shape: vec![2, 3], permutation: vec![0, 0] };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_output_shape() {
        let cfg = TransposeConfig { shape: vec![2, 3, 4], permutation: vec![2, 0, 1] };
        assert_eq!(cfg.output_shape(), vec![4, 2, 3]);
    }

    #[test]
    fn test_config_num_elements() {
        let cfg = TransposeConfig { shape: vec![2, 3, 4], permutation: vec![0, 1, 2] };
        assert_eq!(cfg.num_elements(), 24);
    }

    // -- reshape ------------------------------------------------------------

    #[test]
    fn test_reshape_same_total() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = reshape(&data, &[3, 4], &[2, 6]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_flatten() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let out = reshape(&data, &[2, 3, 4], &[24]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_expand_dims() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = reshape(&data, &[6], &[1, 2, 3]).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_reshape_mismatch_total() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert!(reshape(&data, &[3, 4], &[5, 3]).is_err());
    }

    #[test]
    fn test_reshape_data_length_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(reshape(&data, &[2, 2], &[4]).is_err());
    }

    // -- helper tests -------------------------------------------------------

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }
}
