//! Comprehensive test suite for CUDA transpose operations.

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use crate::{
    BatchTransposeDesc, PermuteDesc, TileConfig, TransposeDesc, batched_transpose_2d,
    contiguous_copy, permute_dims, transpose_2d, transpose_2d_in_place,
};

// ===================================================================
// TransposeDesc
// ===================================================================

#[test]
fn desc_basic_accessors() {
    let d = TransposeDesc::new(4, 5);
    assert_eq!(d.rows(), 4);
    assert_eq!(d.cols(), 5);
    assert_eq!(d.len(), 20);
    assert!(!d.is_empty());
    assert!(!d.is_square());
}

#[test]
fn desc_square() {
    let d = TransposeDesc::new(3, 3);
    assert!(d.is_square());
}

#[test]
fn desc_empty_zero_rows() {
    let d = TransposeDesc::new(0, 5);
    assert!(d.is_empty());
    assert_eq!(d.len(), 0);
}

#[test]
fn desc_empty_zero_cols() {
    let d = TransposeDesc::new(5, 0);
    assert!(d.is_empty());
}

#[test]
fn desc_empty_both_zero() {
    let d = TransposeDesc::new(0, 0);
    assert!(d.is_empty());
    assert!(d.is_square());
}

#[test]
fn desc_single_element() {
    let d = TransposeDesc::new(1, 1);
    assert_eq!(d.len(), 1);
    assert!(d.is_square());
}

// ===================================================================
// 2-D transpose correctness
// ===================================================================

#[test]
fn transpose_2x3() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let desc = TransposeDesc::new(2, 3);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_3x2() {
    let src = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let desc = TransposeDesc::new(3, 2);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_1x1() {
    let src = vec![42.0];
    let desc = TransposeDesc::new(1, 1);
    assert_eq!(transpose_2d(&src, &desc), vec![42.0]);
}

#[test]
fn transpose_1xn_row_to_col() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let desc = TransposeDesc::new(1, 4);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn transpose_nx1_col_to_row() {
    let src = vec![1.0, 2.0, 3.0, 4.0];
    let desc = TransposeDesc::new(4, 1);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn transpose_empty() {
    let src: Vec<f32> = vec![];
    let desc = TransposeDesc::new(0, 5);
    assert!(transpose_2d(&src, &desc).is_empty());
}

#[test]
fn transpose_4x4_square() {
    #[rustfmt::skip]
    let src = vec![
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let desc = TransposeDesc::new(4, 4);
    let dst = transpose_2d(&src, &desc);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 5.0, 9.0,  13.0,
        2.0, 6.0, 10.0, 14.0,
        3.0, 7.0, 11.0, 15.0,
        4.0, 8.0, 12.0, 16.0,
    ];
    assert_eq!(dst, expected);
}

#[test]
fn transpose_large_8x8() {
    let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(8, 8);
    let dst = transpose_2d(&src, &desc);
    for r in 0..8 {
        for c in 0..8 {
            assert_eq!(dst[c * 8 + r], src[r * 8 + c], "mismatch at ({r},{c})");
        }
    }
}

// ===================================================================
// Identity & double transpose
// ===================================================================

#[test]
fn transpose_identity_square_3x3() {
    #[rustfmt::skip]
    let src = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    let desc = TransposeDesc::new(3, 3);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, src, "identity matrix should be its own transpose");
}

#[test]
fn double_transpose_is_original_2x5() {
    let src: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let desc_fwd = TransposeDesc::new(2, 5);
    let t1 = transpose_2d(&src, &desc_fwd);
    let desc_rev = TransposeDesc::new(5, 2);
    let t2 = transpose_2d(&t1, &desc_rev);
    assert_eq!(t2, src);
}

#[test]
fn double_transpose_is_original_7x3() {
    let src: Vec<f32> = (0..21).map(|i| i as f32 * 0.5).collect();
    let fwd = TransposeDesc::new(7, 3);
    let rev = TransposeDesc::new(3, 7);
    assert_eq!(transpose_2d(&transpose_2d(&src, &fwd), &rev), src);
}

#[test]
fn double_transpose_is_original_1x1() {
    let src = vec![99.0];
    let d = TransposeDesc::new(1, 1);
    assert_eq!(transpose_2d(&transpose_2d(&src, &d), &d), src);
}

// ===================================================================
// In-place transpose
// ===================================================================

#[test]
fn in_place_3x3() {
    #[rustfmt::skip]
    let mut data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let desc = TransposeDesc::new(3, 3);
    transpose_2d_in_place(&mut data, &desc);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 4.0, 7.0,
        2.0, 5.0, 8.0,
        3.0, 6.0, 9.0,
    ];
    assert_eq!(data, expected);
}

#[test]
fn in_place_1x1() {
    let mut data = vec![42.0];
    let desc = TransposeDesc::new(1, 1);
    transpose_2d_in_place(&mut data, &desc);
    assert_eq!(data, vec![42.0]);
}

#[test]
fn in_place_matches_out_of_place() {
    let src: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(5, 5);
    let out_of_place = transpose_2d(&src, &desc);
    let mut in_place = src;
    transpose_2d_in_place(&mut in_place, &desc);
    assert_eq!(in_place, out_of_place);
}

#[test]
fn in_place_double_transpose_is_original() {
    let original: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(4, 4);
    let mut data = original.clone();
    transpose_2d_in_place(&mut data, &desc);
    transpose_2d_in_place(&mut data, &desc);
    assert_eq!(data, original);
}

#[test]
#[should_panic(expected = "square")]
fn in_place_non_square_panics() {
    let mut data = vec![1.0; 6];
    let desc = TransposeDesc::new(2, 3);
    transpose_2d_in_place(&mut data, &desc);
}

// ===================================================================
// Batched transpose
// ===================================================================

#[test]
fn batched_single_matrix() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let desc = BatchTransposeDesc::new(1, 2, 3);
    let dst = batched_transpose_2d(&src, &desc);
    let single = transpose_2d(&src, &TransposeDesc::new(2, 3));
    assert_eq!(dst, single);
}

#[test]
fn batched_two_matrices() {
    #[rustfmt::skip]
    let src = vec![
        // batch 0: 2×3
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // batch 1: 2×3
        7.0,  8.0,  9.0,
        10.0, 11.0, 12.0,
    ];
    let desc = BatchTransposeDesc::new(2, 2, 3);
    let dst = batched_transpose_2d(&src, &desc);
    #[rustfmt::skip]
    let expected = vec![
        1.0, 4.0, 2.0, 5.0, 3.0, 6.0,
        7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
    ];
    assert_eq!(dst, expected);
}

#[test]
fn batched_empty_batch() {
    let src: Vec<f32> = vec![];
    let desc = BatchTransposeDesc::new(0, 2, 3);
    assert!(batched_transpose_2d(&src, &desc).is_empty());
}

#[test]
fn batched_desc_accessors() {
    let desc = BatchTransposeDesc::new(4, 3, 5);
    assert_eq!(desc.batch_size(), 4);
    assert_eq!(desc.inner().rows(), 3);
    assert_eq!(desc.inner().cols(), 5);
    assert_eq!(desc.total_len(), 60);
    assert!(!desc.is_empty());
}

#[test]
fn batched_desc_empty() {
    let desc = BatchTransposeDesc::new(0, 3, 5);
    assert!(desc.is_empty());
}

#[test]
fn batched_three_1x1() {
    let src = vec![1.0, 2.0, 3.0];
    let desc = BatchTransposeDesc::new(3, 1, 1);
    assert_eq!(batched_transpose_2d(&src, &desc), src);
}

// ===================================================================
// Permute dimensions
// ===================================================================

#[test]
fn permute_identity_2d() {
    let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let desc = PermuteDesc::new(vec![2, 3], vec![0, 1]);
    let dst = permute_dims(&src, &desc);
    assert_eq!(dst, src);
}

#[test]
fn permute_2d_swap_equals_transpose() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let perm = PermuteDesc::new(vec![2, 3], vec![1, 0]);
    let permuted = permute_dims(&src, &perm);
    let transposed = transpose_2d(&src, &TransposeDesc::new(2, 3));
    assert_eq!(permuted, transposed);
}

#[test]
fn permute_3d_identity() {
    let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let desc = PermuteDesc::new(vec![2, 3, 4], vec![0, 1, 2]);
    assert_eq!(permute_dims(&src, &desc), src);
}

#[test]
fn permute_3d_swap_last_two() {
    // shape [2, 3, 4] → perm [0, 2, 1] → shape [2, 4, 3]
    let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let desc = PermuteDesc::new(vec![2, 3, 4], vec![0, 2, 1]);
    let dst = permute_dims(&src, &desc);
    assert_eq!(dst.len(), 24);
    // Verify element [0,0,0] stays at 0
    assert_eq!(dst[0], 0.0);
    // src[0,1,0] = 4.0 should map to dst[0,0,1] = index 1 in [2,4,3]
    assert_eq!(dst[1], 4.0);
}

#[test]
fn permute_3d_full_rotation() {
    // perm [2, 0, 1]: shape [2,3,4] → [4,2,3]
    let src: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let desc = PermuteDesc::new(vec![2, 3, 4], vec![2, 0, 1]);
    let dst = permute_dims(&src, &desc);
    assert_eq!(dst.len(), 24);
    // src[0,0,0]=0 → dst[0,0,0]=0
    assert_eq!(dst[0], 0.0);
    // src[1,2,3]=23 → dst[3,1,2] = 3*6+1*3+2 = 23 in [4,2,3]
    assert_eq!(dst[23], 23.0);
}

#[test]
fn permute_double_is_original() {
    let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let fwd = PermuteDesc::new(vec![3, 4], vec![1, 0]);
    let t1 = permute_dims(&src, &fwd);
    let rev = PermuteDesc::new(vec![4, 3], vec![1, 0]);
    let t2 = permute_dims(&t1, &rev);
    assert_eq!(t2, src);
}

#[test]
fn permute_empty() {
    let src: Vec<f32> = vec![];
    let desc = PermuteDesc::new(vec![0, 3], vec![1, 0]);
    assert!(permute_dims(&src, &desc).is_empty());
}

#[test]
fn permute_desc_output_shape() {
    let desc = PermuteDesc::new(vec![2, 3, 4], vec![2, 0, 1]);
    assert_eq!(desc.output_shape(), vec![4, 2, 3]);
}

#[test]
fn permute_desc_is_identity() {
    assert!(PermuteDesc::new(vec![2, 3], vec![0, 1]).is_identity());
    assert!(!PermuteDesc::new(vec![2, 3], vec![1, 0]).is_identity());
}

#[test]
fn permute_desc_ndim() {
    let desc = PermuteDesc::new(vec![2, 3, 4, 5], vec![3, 2, 1, 0]);
    assert_eq!(desc.ndim(), 4);
}

#[test]
fn permute_desc_total_len() {
    let desc = PermuteDesc::new(vec![2, 3, 4], vec![0, 1, 2]);
    assert_eq!(desc.total_len(), 24);
}

#[test]
#[should_panic(expected = "perm length")]
fn permute_desc_bad_perm_length() {
    let _ = PermuteDesc::new(vec![2, 3], vec![0]);
}

#[test]
#[should_panic(expected = "duplicate")]
fn permute_desc_duplicate_axis() {
    let _ = PermuteDesc::new(vec![2, 3], vec![0, 0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn permute_desc_out_of_range() {
    let _ = PermuteDesc::new(vec![2, 3], vec![0, 2]);
}

// ===================================================================
// Contiguous copy
// ===================================================================

#[test]
fn contiguous_row_major_is_noop() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // shape [2,3], row-major strides [3,1]
    let dst = contiguous_copy(&src, &[2, 3], &[3, 1]);
    assert_eq!(dst, src);
}

#[test]
fn contiguous_col_major_to_row_major() {
    // 2×3 in column-major: strides [1, 2]
    // Logical: [[1,3,5],[2,4,6]] → row-major [1,3,5,2,4,6]
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dst = contiguous_copy(&src, &[2, 3], &[1, 2]);
    assert_eq!(dst, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn contiguous_empty() {
    let dst = contiguous_copy(&[], &[0, 3], &[3, 1]);
    assert!(dst.is_empty());
}

#[test]
fn contiguous_1d() {
    let src = vec![10.0, 20.0, 30.0];
    let dst = contiguous_copy(&src, &[3], &[1]);
    assert_eq!(dst, src);
}

#[test]
fn contiguous_1d_strided() {
    // Pick every other element
    let src = vec![10.0, 99.0, 20.0, 99.0, 30.0];
    let dst = contiguous_copy(&src, &[3], &[2]);
    assert_eq!(dst, vec![10.0, 20.0, 30.0]);
}

// ===================================================================
// TileConfig
// ===================================================================

#[test]
fn tile_default() {
    let t = TileConfig::default();
    assert_eq!(t.tile_dim(), 32);
    assert_eq!(t.block_rows(), 8);
    assert_eq!(t.threads_per_block(), 256);
}

#[test]
fn tile_custom() {
    let t = TileConfig::new(16, 4);
    assert_eq!(t.tile_dim(), 16);
    assert_eq!(t.block_rows(), 4);
    assert_eq!(t.threads_per_block(), 64);
}

#[test]
fn tile_grid_dims_exact() {
    let t = TileConfig::new(32, 8);
    assert_eq!(t.grid_dims(64, 128), (4, 2));
}

#[test]
fn tile_grid_dims_non_exact() {
    let t = TileConfig::new(32, 8);
    // 33 rows → ceil(33/32) = 2, 65 cols → ceil(65/32) = 3
    assert_eq!(t.grid_dims(33, 65), (3, 2));
}

#[test]
fn tile_shared_mem_bytes() {
    let t = TileConfig::new(32, 8);
    // 32 * (32+1) * 4 = 32 * 33 * 4 = 4224
    assert_eq!(t.shared_mem_bytes(), 4224);
}

#[test]
#[should_panic(expected = "tile_dim must be > 0")]
fn tile_zero_dim_panics() {
    let _ = TileConfig::new(0, 8);
}

#[test]
#[should_panic(expected = "block_rows must be > 0")]
fn tile_zero_block_rows_panics() {
    let _ = TileConfig::new(32, 0);
}

#[test]
#[should_panic(expected = "divisible")]
fn tile_indivisible_panics() {
    let _ = TileConfig::new(32, 5);
}

// ===================================================================
// Panic / edge-case tests
// ===================================================================

#[test]
#[should_panic(expected = "source length")]
fn transpose_2d_length_mismatch() {
    let _ = transpose_2d(&[1.0, 2.0], &TransposeDesc::new(2, 3));
}

#[test]
#[should_panic(expected = "source length")]
fn batched_length_mismatch() {
    let _ = batched_transpose_2d(&[1.0], &BatchTransposeDesc::new(2, 2, 3));
}

#[test]
#[should_panic(expected = "source length")]
fn permute_length_mismatch() {
    let _ = permute_dims(&[1.0], &PermuteDesc::new(vec![2, 3], vec![1, 0]));
}

#[test]
#[should_panic(expected = "shape and strides")]
fn contiguous_shape_stride_mismatch() {
    let _ = contiguous_copy(&[1.0], &[2, 3], &[1]);
}

// ===================================================================
// Various shapes
// ===================================================================

#[test]
fn transpose_wide_1x100() {
    let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(1, 100);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, src);
}

#[test]
fn transpose_tall_100x1() {
    let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(100, 1);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, src);
}

#[test]
fn transpose_prime_dims_7x11() {
    let src: Vec<f32> = (0..77).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(7, 11);
    let dst = transpose_2d(&src, &desc);
    for r in 0..7 {
        for c in 0..11 {
            assert_eq!(dst[c * 7 + r], src[r * 11 + c]);
        }
    }
}

#[test]
fn transpose_prime_dims_13x17() {
    let src: Vec<f32> = (0..221).map(|i| i as f32).collect();
    let desc = TransposeDesc::new(13, 17);
    let dst = transpose_2d(&src, &desc);
    for r in 0..13 {
        for c in 0..17 {
            assert_eq!(dst[c * 13 + r], src[r * 17 + c]);
        }
    }
}

#[test]
fn transpose_negative_values() {
    let src = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let desc = TransposeDesc::new(2, 3);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst, vec![-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
}

#[test]
fn transpose_special_floats() {
    let src = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    let desc = TransposeDesc::new(2, 2);
    let dst = transpose_2d(&src, &desc);
    assert_eq!(dst[0], f32::INFINITY);
    assert_eq!(dst[1], 0.0);
    assert_eq!(dst[2], f32::NEG_INFINITY);
    assert!(dst[3].is_sign_negative() && dst[3] == 0.0);
}

#[test]
fn transpose_nan_preserved() {
    let src = vec![f32::NAN, 1.0, 2.0, 3.0];
    let desc = TransposeDesc::new(2, 2);
    let dst = transpose_2d(&src, &desc);
    assert!(dst[0].is_nan());
    assert_eq!(dst[1], 2.0);
    assert_eq!(dst[2], 1.0);
    assert_eq!(dst[3], 3.0);
}

// ===================================================================
// GPU module (feature-gated)
// ===================================================================

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn cuda_launch_transpose_returns_err_without_runtime() {
    let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut dst = vec![0.0; 6];
    let desc = TransposeDesc::new(2, 3);
    let tile = TileConfig::default();
    let result = crate::cuda::launch_transpose_2d(&src, &mut dst, &desc, &tile);
    assert!(result.is_err());
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn cuda_launch_batched_returns_err_without_runtime() {
    let src = vec![1.0; 12];
    let mut dst = vec![0.0; 12];
    let desc = BatchTransposeDesc::new(2, 2, 3);
    let tile = TileConfig::default();
    let result = crate::cuda::launch_batched_transpose_2d(&src, &mut dst, &desc, &tile);
    assert!(result.is_err());
}

// ===================================================================
// Property tests (proptest)
// ===================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_dims() -> impl Strategy<Value = (usize, usize)> {
        (1..=32_usize, 1..=32_usize)
    }

    proptest! {
        #[test]
        fn double_transpose_roundtrip((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let fwd = TransposeDesc::new(rows, cols);
            let rev = TransposeDesc::new(cols, rows);
            let roundtrip = transpose_2d(&transpose_2d(&src, &fwd), &rev);
            prop_assert_eq!(roundtrip, src);
        }

        #[test]
        fn transpose_preserves_length((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let desc = TransposeDesc::new(rows, cols);
            let dst = transpose_2d(&src, &desc);
            prop_assert_eq!(dst.len(), n);
        }

        #[test]
        fn transpose_preserves_sum((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let desc = TransposeDesc::new(rows, cols);
            let dst = transpose_2d(&src, &desc);
            let src_sum: f32 = src.iter().sum();
            let dst_sum: f32 = dst.iter().sum();
            prop_assert!((src_sum - dst_sum).abs() < 1e-3);
        }

        #[test]
        fn transpose_preserves_sorted_multiset((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let desc = TransposeDesc::new(rows, cols);
            let dst = transpose_2d(&src, &desc);
            let mut src_sorted = src;
            let mut dst_sorted = dst;
            src_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dst_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            prop_assert_eq!(src_sorted, dst_sorted);
        }

        #[test]
        fn in_place_matches_out_of_place_prop(n in 1..=16_usize) {
            let src: Vec<f32> = (0..n*n).map(|i| i as f32).collect();
            let desc = TransposeDesc::new(n, n);
            let oop = transpose_2d(&src, &desc);
            let mut ip = src;
            transpose_2d_in_place(&mut ip, &desc);
            prop_assert_eq!(ip, oop);
        }

        #[test]
        fn batched_matches_individual((rows, cols) in arb_dims(), batch in 1..=8_usize) {
            let mat_len = rows * cols;
            let src: Vec<f32> = (0..batch * mat_len).map(|i| i as f32).collect();
            let batch_desc = BatchTransposeDesc::new(batch, rows, cols);
            let batched = batched_transpose_2d(&src, &batch_desc);

            let inner = TransposeDesc::new(rows, cols);
            let mut individual = Vec::with_capacity(src.len());
            for b in 0..batch {
                let off = b * mat_len;
                individual.extend_from_slice(&transpose_2d(&src[off..off + mat_len], &inner));
            }
            prop_assert_eq!(batched, individual);
        }

        #[test]
        fn permute_2d_swap_matches_transpose((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let perm = PermuteDesc::new(vec![rows, cols], vec![1, 0]);
            let td = TransposeDesc::new(rows, cols);
            prop_assert_eq!(permute_dims(&src, &perm), transpose_2d(&src, &td));
        }

        #[test]
        fn permute_identity_is_noop((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let desc = PermuteDesc::new(vec![rows, cols], vec![0, 1]);
            prop_assert_eq!(permute_dims(&src, &desc), src);
        }

        #[test]
        fn contiguous_row_major_is_identity((rows, cols) in arb_dims()) {
            let n = rows * cols;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let dst = contiguous_copy(&src, &[rows, cols], &[cols, 1]);
            prop_assert_eq!(dst, src);
        }

        #[test]
        fn tile_grid_covers_matrix(
            (rows, cols) in (1..=1024_u32, 1..=1024_u32),
        ) {
            let t = TileConfig::default();
            let (gx, gy) = t.grid_dims(rows, cols);
            prop_assert!(gx * t.tile_dim() >= cols);
            prop_assert!(gy * t.tile_dim() >= rows);
        }

        #[test]
        fn tile_threads_equals_dim_times_block_rows(
            tile_dim in prop::sample::select(vec![8u32, 16, 32, 64]),
        ) {
            let block_rows = tile_dim / 4;
            let t = TileConfig::new(tile_dim, block_rows);
            prop_assert_eq!(t.threads_per_block(), tile_dim * block_rows);
        }
    }
}
