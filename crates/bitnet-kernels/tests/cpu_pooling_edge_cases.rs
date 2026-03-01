//! Edge-case tests for CPU pooling operations.
//!
//! Tests cover 1-D/2-D pooling (max, avg, min), adaptive pooling,
//! and global average/max pool variants.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::pooling::{
    PoolConfig, PoolType, PoolingKernel, adaptive_avg_pool_1d, adaptive_avg_pool_2d,
    global_avg_pool, global_max_pool, pool_1d, pool_2d,
};

// ── 1-D pooling ──────────────────────────────────────────────────────

#[test]
fn pool_1d_max_basic() {
    let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let result = pool_1d(&input, &config).unwrap();
    assert_eq!(result, vec![3.0, 5.0, 6.0]);
}

#[test]
fn pool_1d_avg_basic() {
    let config = PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
    let input = vec![2.0, 4.0, 6.0, 8.0];
    let result = pool_1d(&input, &config).unwrap();
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[1] - 7.0).abs() < 1e-6);
}

#[test]
fn pool_1d_stride_1() {
    let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 3, stride: 1, padding: 0 };
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let result = pool_1d(&input, &config).unwrap();
    // Windows: [1,3,2]→3, [3,2,5]→5, [2,5,4]→5
    assert_eq!(result, vec![3.0, 5.0, 5.0]);
}

#[test]
fn pool_1d_single_element_window() {
    let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 1, stride: 1, padding: 0 };
    let input = vec![1.0, 2.0, 3.0];
    let result = pool_1d(&input, &config).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

// ── 2-D pooling ──────────────────────────────────────────────────────

#[test]
fn pool_2d_max_basic() {
    let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    // 4x4 input
    let input =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let (result, out_h, out_w) = pool_2d(&input, 4, 4, &config).unwrap();
    assert_eq!(out_h, 2);
    assert_eq!(out_w, 2);
    assert_eq!(result, vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn pool_2d_avg_basic() {
    let config = PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
    let input = vec![1.0, 3.0, 5.0, 7.0]; // 2x2
    let (result, out_h, out_w) = pool_2d(&input, 2, 2, &config).unwrap();
    assert_eq!(out_h, 1);
    assert_eq!(out_w, 1);
    assert!((result[0] - 4.0).abs() < 1e-6);
}

// ── Adaptive pooling ─────────────────────────────────────────────────

#[test]
fn adaptive_avg_pool_1d_downsample() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = adaptive_avg_pool_1d(&input, 3).unwrap();
    assert_eq!(result.len(), 3);
    // Each bin covers 2 elements
    assert!((result[0] - 1.5).abs() < 1e-6);
    assert!((result[1] - 3.5).abs() < 1e-6);
    assert!((result[2] - 5.5).abs() < 1e-6);
}

#[test]
fn adaptive_avg_pool_1d_identity() {
    let input = vec![1.0, 2.0, 3.0];
    let result = adaptive_avg_pool_1d(&input, 3).unwrap();
    assert_eq!(result.len(), 3);
}

#[test]
fn adaptive_avg_pool_2d_basic() {
    // 4x4 → 2x2
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let result = adaptive_avg_pool_2d(&input, 4, 4, 2, 2).unwrap();
    assert_eq!(result.len(), 4);
}

// ── Global pooling ───────────────────────────────────────────────────

#[test]
fn global_avg_pool_1d() {
    let input = vec![2.0, 4.0, 6.0];
    let result = global_avg_pool(&input, &[3]).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 4.0).abs() < 1e-6);
}

#[test]
fn global_max_pool_1d() {
    let input = vec![2.0, 8.0, 4.0];
    let result = global_max_pool(&input, &[3]).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 8.0).abs() < 1e-6);
}

#[test]
fn global_avg_pool_negative() {
    let input = vec![-2.0, -4.0, -6.0];
    let result = global_avg_pool(&input, &[3]).unwrap();
    assert!((result[0] - (-4.0)).abs() < 1e-6);
}

// ── PoolingKernel apply ──────────────────────────────────────────────

#[test]
fn pooling_kernel_apply_max() {
    let config = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let input = vec![1.0, 5.0, 3.0, 7.0];
    let result = PoolingKernel::apply(&input, &config).unwrap();
    assert_eq!(result, vec![5.0, 7.0]);
}

#[test]
fn pooling_kernel_adaptive_config() {
    let config = PoolingKernel::adaptive_config(PoolType::Average, 10, 5).unwrap();
    assert_eq!(config.pool_type, PoolType::Average);
    // Adaptive should produce output_size=5 from input_size=10
    assert!(config.kernel_size > 0);
    assert!(config.stride > 0);
}
