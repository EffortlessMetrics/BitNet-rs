//! Edge-case integration tests for `bitnet_kernels::cpu::pooling` module.
//!
//! Covers:
//! - PoolConfig validation and PoolType variants
//! - PoolingKernel::apply for 1-D max and average pooling
//! - PoolingKernel::adaptive_config
//! - pool_1d, pool_2d free functions
//! - adaptive_avg_pool_1d, adaptive_avg_pool_2d
//! - global_avg_pool, global_max_pool with spatial dims
//! - Padding, stride, and kernel size edge cases
//! - Error paths: empty inputs, invalid configs

use bitnet_kernels::cpu::pooling::{
    PoolConfig, PoolType, PoolingKernel, adaptive_avg_pool_1d, adaptive_avg_pool_2d,
    global_avg_pool, global_max_pool, pool_1d, pool_2d,
};

const TOL: f32 = 1e-5;

// =========================================================================
// PoolConfig validation
// =========================================================================

#[test]
fn pool_config_valid_max() {
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
    assert!(cfg.validate().is_ok());
}

#[test]
fn pool_config_zero_kernel_error() {
    let cfg = PoolConfig { pool_type: PoolType::Average, kernel_size: 0, stride: 1, padding: 0 };
    assert!(cfg.validate().is_err());
}

#[test]
fn pool_config_zero_stride_error() {
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 0, padding: 0 };
    assert!(cfg.validate().is_err());
}

#[test]
fn pool_config_global_max_skips_validation() {
    let cfg = PoolConfig {
        pool_type: PoolType::GlobalMax,
        kernel_size: 0, // would fail for non-global
        stride: 0,
        padding: 0,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn pool_config_global_avg_skips_validation() {
    let cfg =
        PoolConfig { pool_type: PoolType::GlobalAverage, kernel_size: 0, stride: 0, padding: 0 };
    assert!(cfg.validate().is_ok());
}

// =========================================================================
// 1-D max pooling
// =========================================================================

#[test]
fn max_pool_1d_basic() {
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    // Windows: [1,3,2]=3, [3,2,5]=5, [2,5,4]=5
    assert_eq!(out.len(), 3);
    assert!((out[0] - 3.0).abs() < TOL);
    assert!((out[1] - 5.0).abs() < TOL);
    assert!((out[2] - 5.0).abs() < TOL);
}

#[test]
fn max_pool_1d_stride_2() {
    let input = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    // Windows: [1,5]=5, [2,8]=8, [3,7]=7
    assert_eq!(out.len(), 3);
    assert!((out[0] - 5.0).abs() < TOL);
    assert!((out[1] - 8.0).abs() < TOL);
    assert!((out[2] - 7.0).abs() < TOL);
}

#[test]
fn max_pool_1d_with_padding() {
    let input = vec![3.0, 1.0, 4.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 1 };
    let out = pool_1d(&input, &cfg).unwrap();
    // Padded: [-inf, 3, 1, 4, -inf]
    // Windows: [-inf,3,1]=3, [3,1,4]=4, [1,4,-inf]=4
    assert_eq!(out.len(), 3);
    assert!((out[0] - 3.0).abs() < TOL);
    assert!((out[1] - 4.0).abs() < TOL);
    assert!((out[2] - 4.0).abs() < TOL);
}

#[test]
fn max_pool_1d_kernel_equals_input() {
    let input = vec![2.0, 5.0, 1.0, 8.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 4, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 8.0).abs() < TOL);
}

#[test]
fn max_pool_1d_single_element() {
    let input = vec![42.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 1, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 42.0).abs() < TOL);
}

#[test]
fn max_pool_1d_negative_values() {
    let input = vec![-5.0, -3.0, -8.0, -1.0];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 3);
    assert!((out[0] - (-3.0)).abs() < TOL);
    assert!((out[1] - (-3.0)).abs() < TOL);
    assert!((out[2] - (-1.0)).abs() < TOL);
}

// =========================================================================
// 1-D average pooling
// =========================================================================

#[test]
fn avg_pool_1d_basic() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let cfg = PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 3.0).abs() < TOL);
    assert!((out[1] - 7.0).abs() < TOL);
}

#[test]
fn avg_pool_count_include_pad_same_as_avg() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cfg_avg =
        PoolConfig { pool_type: PoolType::Average, kernel_size: 3, stride: 1, padding: 1 };
    let cfg_cip = PoolConfig {
        pool_type: PoolType::AvgPoolCountIncludePad,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    };
    let out_avg = pool_1d(&input, &cfg_avg).unwrap();
    let out_cip = pool_1d(&input, &cfg_cip).unwrap();
    assert_eq!(out_avg.len(), out_cip.len());
    for (a, c) in out_avg.iter().zip(out_cip.iter()) {
        assert!((a - c).abs() < TOL);
    }
}

// =========================================================================
// Global pooling
// =========================================================================

#[test]
fn global_max_basic() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let cfg = PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 5.0).abs() < TOL);
}

#[test]
fn global_avg_basic() {
    let input = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let cfg =
        PoolConfig { pool_type: PoolType::GlobalAverage, kernel_size: 0, stride: 0, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 6.0).abs() < TOL);
}

#[test]
fn global_max_single_element() {
    let cfg = PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
    let out = pool_1d(&[7.0], &cfg).unwrap();
    assert!((out[0] - 7.0).abs() < TOL);
}

#[test]
fn global_pool_empty_error() {
    let cfg = PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
    assert!(pool_1d(&[], &cfg).is_err());
}

// =========================================================================
// PoolingKernel::adaptive_config
// =========================================================================

#[test]
fn adaptive_config_output_1_gives_global() {
    let cfg = PoolingKernel::adaptive_config(PoolType::Max, 100, 1).unwrap();
    assert_eq!(cfg.pool_type, PoolType::GlobalMax);
}

#[test]
fn adaptive_config_avg_output_1_gives_global_avg() {
    let cfg = PoolingKernel::adaptive_config(PoolType::Average, 50, 1).unwrap();
    assert_eq!(cfg.pool_type, PoolType::GlobalAverage);
}

#[test]
fn adaptive_config_output_equals_input() {
    let cfg = PoolingKernel::adaptive_config(PoolType::Average, 10, 10).unwrap();
    // stride=1, kernel_size=1 → identity
    assert_eq!(cfg.kernel_size, 1);
    assert_eq!(cfg.stride, 1);
}

#[test]
fn adaptive_config_output_zero_error() {
    assert!(PoolingKernel::adaptive_config(PoolType::Max, 10, 0).is_err());
}

#[test]
fn adaptive_config_input_zero_error() {
    assert!(PoolingKernel::adaptive_config(PoolType::Max, 0, 1).is_err());
}

#[test]
fn adaptive_config_output_larger_than_input_error() {
    assert!(PoolingKernel::adaptive_config(PoolType::Max, 5, 10).is_err());
}

// =========================================================================
// pool_2d
// =========================================================================

#[test]
fn pool_2d_max_basic() {
    // 3x3 input, 2x2 kernel, stride 1
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
    let (out, oh, ow) = pool_2d(&input, 3, 3, &cfg).unwrap();
    assert_eq!(oh, 2);
    assert_eq!(ow, 2);
    assert!((out[0] - 5.0).abs() < TOL); // max of [1,2,4,5]
    assert!((out[1] - 6.0).abs() < TOL); // max of [2,3,5,6]
    assert!((out[2] - 8.0).abs() < TOL); // max of [4,5,7,8]
    assert!((out[3] - 9.0).abs() < TOL); // max of [5,6,8,9]
}

#[test]
fn pool_2d_avg_basic() {
    #[rustfmt::skip]
    let input = vec![
        1.0, 3.0,
        5.0, 7.0,
    ];
    let cfg = PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 1, padding: 0 };
    let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
    assert_eq!(oh, 1);
    assert_eq!(ow, 1);
    assert!((out[0] - 4.0).abs() < TOL); // mean of [1,3,5,7]
}

#[test]
fn pool_2d_global_max() {
    let input = vec![1.0, 5.0, 3.0, 9.0];
    let cfg = PoolConfig { pool_type: PoolType::GlobalMax, kernel_size: 0, stride: 0, padding: 0 };
    let (out, oh, ow) = pool_2d(&input, 2, 2, &cfg).unwrap();
    assert_eq!(oh, 1);
    assert_eq!(ow, 1);
    assert!((out[0] - 9.0).abs() < TOL);
}

#[test]
fn pool_2d_input_size_mismatch_error() {
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 1, padding: 0 };
    assert!(pool_2d(&[1.0, 2.0, 3.0], 2, 2, &cfg).is_err());
}

// =========================================================================
// adaptive_avg_pool_1d
// =========================================================================

#[test]
fn adaptive_avg_pool_1d_halve() {
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let out = adaptive_avg_pool_1d(&input, 2).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 3.0).abs() < TOL); // mean(2,4)
    assert!((out[1] - 7.0).abs() < TOL); // mean(6,8)
}

#[test]
fn adaptive_avg_pool_1d_identity() {
    let input = vec![1.0, 2.0, 3.0];
    let out = adaptive_avg_pool_1d(&input, 3).unwrap();
    assert_eq!(out.len(), 3);
    for (i, v) in out.iter().enumerate() {
        assert!((v - (i as f32 + 1.0)).abs() < TOL);
    }
}

#[test]
fn adaptive_avg_pool_1d_to_one() {
    let input = vec![10.0, 20.0, 30.0];
    let out = adaptive_avg_pool_1d(&input, 1).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 20.0).abs() < TOL);
}

#[test]
fn adaptive_avg_pool_1d_empty_error() {
    assert!(adaptive_avg_pool_1d(&[], 1).is_err());
}

#[test]
fn adaptive_avg_pool_1d_output_zero_error() {
    assert!(adaptive_avg_pool_1d(&[1.0], 0).is_err());
}

#[test]
fn adaptive_avg_pool_1d_output_larger_error() {
    assert!(adaptive_avg_pool_1d(&[1.0, 2.0], 5).is_err());
}

// =========================================================================
// adaptive_avg_pool_2d
// =========================================================================

#[test]
fn adaptive_avg_pool_2d_basic() {
    #[rustfmt::skip]
    let input = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let out = adaptive_avg_pool_2d(&input, 4, 4, 2, 2).unwrap();
    assert_eq!(out.len(), 4);
    // Top-left 2x2: mean(1,2,5,6) = 3.5
    assert!((out[0] - 3.5).abs() < TOL);
    // Top-right 2x2: mean(3,4,7,8) = 5.5
    assert!((out[1] - 5.5).abs() < TOL);
}

#[test]
fn adaptive_avg_pool_2d_to_1x1() {
    let input = vec![2.0, 4.0, 6.0, 8.0]; // 2x2
    let out = adaptive_avg_pool_2d(&input, 2, 2, 1, 1).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 5.0).abs() < TOL);
}

#[test]
fn adaptive_avg_pool_2d_size_mismatch_error() {
    assert!(adaptive_avg_pool_2d(&[1.0, 2.0], 2, 2, 1, 1).is_err());
}

#[test]
fn adaptive_avg_pool_2d_output_larger_error() {
    assert!(adaptive_avg_pool_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2, 3, 3).is_err());
}

// =========================================================================
// global_avg_pool / global_max_pool with spatial dims
// =========================================================================

#[test]
fn global_avg_pool_multi_channel() {
    // 2 channels, spatial 3
    let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let out = global_avg_pool(&input, &[3]).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 2.0).abs() < TOL);
    assert!((out[1] - 20.0).abs() < TOL);
}

#[test]
fn global_max_pool_multi_channel() {
    let input = vec![1.0, 5.0, 3.0, 10.0, 20.0, 15.0];
    let out = global_max_pool(&input, &[3]).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 5.0).abs() < TOL);
    assert!((out[1] - 20.0).abs() < TOL);
}

#[test]
fn global_avg_pool_2d_spatial() {
    // 1 channel, 2x3 spatial
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = global_avg_pool(&input, &[2, 3]).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 3.5).abs() < TOL);
}

#[test]
fn global_max_pool_2d_spatial() {
    let input = vec![1.0, 9.0, 3.0, 4.0, 5.0, 6.0];
    let out = global_max_pool(&input, &[2, 3]).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 9.0).abs() < TOL);
}

#[test]
fn global_avg_pool_zero_spatial_error() {
    assert!(global_avg_pool(&[1.0], &[0]).is_err());
}

#[test]
fn global_max_pool_not_divisible_error() {
    assert!(global_max_pool(&[1.0, 2.0, 3.0], &[2]).is_err());
}

// =========================================================================
// Numerical edge cases
// =========================================================================

#[test]
fn max_pool_all_same_values() {
    let input = vec![5.0; 10];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    for &v in &out {
        assert!((v - 5.0).abs() < TOL);
    }
}

#[test]
fn avg_pool_large_values() {
    let input = vec![1e8, 1e8, 1e8, 1e8];
    let cfg = PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    for &v in &out {
        assert!((v - 1e8).abs() < 1.0);
    }
}

#[test]
fn max_pool_with_neg_infinity() {
    let input = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    let cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
    let out = pool_1d(&input, &cfg).unwrap();
    assert!((out[0] - 1.0).abs() < TOL);
}
