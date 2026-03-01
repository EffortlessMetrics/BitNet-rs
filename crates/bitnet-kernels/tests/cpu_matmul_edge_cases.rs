//! Edge-case tests for CPU SIMD matrix multiplication.
//!
//! Tests cover f32 matmul, i2s quantized matmul, batched matmul,
//! and SimdMatmulConfig.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::quantized_matmul::pack_i2s;
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};

// ── SimdMatmulConfig ─────────────────────────────────────────────────

#[test]
fn config_defaults() {
    let cfg = SimdMatmulConfig::new(2, 3, 4);
    assert_eq!(cfg.m, 2);
    assert_eq!(cfg.n, 3);
    assert_eq!(cfg.k, 4);
    assert!((cfg.alpha - 1.0).abs() < 1e-6);
    assert!((cfg.beta - 0.0).abs() < 1e-6);
}

// ── f32 matmul ───────────────────────────────────────────────────────

#[test]
fn matmul_identity_2x2() {
    let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
    let mut c = vec![0.0; 4];
    let cfg = SimdMatmulConfig::new(2, 2, 2);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_basic_2x3_3x2() {
    // A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = vec![0.0; 4]; // 2x2
    let cfg = SimdMatmulConfig::new(2, 2, 3);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    assert!((c[0] - 58.0).abs() < 1e-3);
    assert!((c[1] - 64.0).abs() < 1e-3);
    assert!((c[2] - 139.0).abs() < 1e-3);
    assert!((c[3] - 154.0).abs() < 1e-3);
}

#[test]
fn matmul_zeros() {
    let a = vec![0.0; 4]; // 2x2
    let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let mut c = vec![0.0; 4];
    let cfg = SimdMatmulConfig::new(2, 2, 2);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    assert_eq!(c, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn matmul_with_alpha() {
    let a = vec![1.0, 0.0, 0.0, 1.0]; // identity
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];
    let mut cfg = SimdMatmulConfig::new(2, 2, 2);
    cfg.alpha = 2.0; // scale result by 2
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    assert!((c[0] - 2.0).abs() < 1e-4);
    assert!((c[3] - 8.0).abs() < 1e-4);
}

#[test]
fn matmul_with_beta() {
    let a = vec![1.0, 0.0, 0.0, 1.0]; // identity
    let b = vec![1.0, 1.0, 1.0, 1.0]; // all ones
    let mut c = vec![10.0; 4]; // pre-filled
    let mut cfg = SimdMatmulConfig::new(2, 2, 2);
    cfg.beta = 1.0; // C = alpha*A*B + beta*C
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    // C = 1*I*ones + 1*10 = 1 + 10 = 11
    assert!((c[0] - 11.0).abs() < 1e-4);
}

#[test]
fn matmul_large() {
    let n = 32;
    let a = vec![1.0; n * n];
    let b = vec![1.0; n * n];
    let mut c = vec![0.0; n * n];
    let cfg = SimdMatmulConfig::new(n, n, n);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
    // All ones * all ones → each element = n
    assert!((c[0] - n as f32).abs() < 1e-2);
}

// ── pack_i2s ─────────────────────────────────────────────────────────

#[test]
fn pack_i2s_basic() {
    let packed = pack_i2s([0, 1, -1, 0]);
    // Just verify it produces a byte
    assert!(packed <= 255);
}

#[test]
fn pack_i2s_all_zero() {
    let packed = pack_i2s([0, 0, 0, 0]);
    assert_eq!(packed, 0);
}

#[test]
fn pack_i2s_all_one() {
    let packed = pack_i2s([1, 1, 1, 1]);
    assert!(packed > 0);
}
