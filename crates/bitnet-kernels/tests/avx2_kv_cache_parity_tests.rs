#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for KV cache SIMD operations.
//!
//! Each test exercises dot-product and scale kernels that have an
//! explicit AVX2 fast-path (via `is_x86_feature_detected!("avx2")`
//! runtime dispatch) and compares the output against a pure-scalar
//! reference within defined tolerances.
//!
//! Test dimensions are chosen to exercise both the 8-wide SIMD body
//! and the scalar tail (lengths not divisible by 8).
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::kv_cache_simd::{scalar_dot_f32, simd_dot_f32, simd_scale_f32};

// Cache management API — available for integration tests.
#[allow(unused_imports)]
use bitnet_kernels::cpu::kv_cache_simd::{
    EvictionPolicy, KVCacheConfig, append_kv, create_kv_cache, lookup_kv,
};

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, close, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// FMA vs multiply-then-add can differ by up to ~0.5 ULP per op; with
// hundreds of accumulations the absolute error can reach ~1e-5 for
// dot products on moderate vectors.

/// Absolute tolerance for dot-product kernels.
const DOT_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for dot-product kernels.
const DOT_REL_TOL: f32 = 1e-5;

/// Absolute tolerance for scale kernels (multiplication is exact in
/// IEEE 754 for the scale factors used here).
const SCALE_ABS_TOL: f32 = 1e-7;
/// Relative tolerance for scale kernels.
const SCALE_REL_TOL: f32 = 1e-7;

// ════════════════════════════════════════════════════════════════════════
// 1. DOT PRODUCT PARITY
// ════════════════════════════════════════════════════════════════════════

/// Dot-product parity: 128 elements (8-aligned, exercises SIMD body only).
#[test]
fn test_simd_dot_parity_aligned() {
    let a = pseudo_rand(128, 42);
    let b = pseudo_rand(128, 137);
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert!(
        close(simd_result, scalar_result, DOT_ABS_TOL, DOT_REL_TOL),
        "aligned(128): scalar={scalar_result}, simd={simd_result} (diff={})",
        (simd_result - scalar_result).abs()
    );
}

/// Dot-product parity: 63 elements (exercises scalar tail).
#[test]
fn test_simd_dot_parity_unaligned() {
    let a = pseudo_rand(63, 200);
    let b = pseudo_rand(63, 300);
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert!(
        close(simd_result, scalar_result, DOT_ABS_TOL, DOT_REL_TOL),
        "unaligned(63): scalar={scalar_result}, simd={simd_result} (diff={})",
        (simd_result - scalar_result).abs()
    );
}

/// Dot-product parity: 7 elements (all scalar, below SIMD threshold).
#[test]
fn test_simd_dot_parity_small() {
    let a = pseudo_rand(7, 400);
    let b = pseudo_rand(7, 500);
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert!(
        close(simd_result, scalar_result, DOT_ABS_TOL, DOT_REL_TOL),
        "small(7): scalar={scalar_result}, simd={simd_result} (diff={})",
        (simd_result - scalar_result).abs()
    );
}

/// Dot-product parity: 1024 elements (large, many SIMD iterations).
#[test]
fn test_simd_dot_parity_large() {
    let a = pseudo_rand(1024, 600);
    let b = pseudo_rand(1024, 700);
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert!(
        close(simd_result, scalar_result, DOT_ABS_TOL, DOT_REL_TOL),
        "large(1024): scalar={scalar_result}, simd={simd_result} (diff={})",
        (simd_result - scalar_result).abs()
    );
}

/// Dot product of zero vectors must be exactly 0.0.
#[test]
fn test_simd_dot_zero_vectors() {
    let a = vec![0.0f32; 128];
    let b = vec![0.0f32; 128];
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert_eq!(simd_result, 0.0, "simd dot of zeros should be 0.0");
    assert_eq!(scalar_result, 0.0, "scalar dot of zeros should be 0.0");
}

/// Orthogonal vectors: interleaved [1,0,1,0...] × [0,1,0,1...] = 0.0.
#[test]
fn test_simd_dot_orthogonal() {
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let b: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let simd_result = simd_dot_f32(&a, &b);
    let scalar_result = scalar_dot_f32(&a, &b);
    assert_eq!(simd_result, 0.0, "simd dot of orthogonal vectors should be 0.0");
    assert_eq!(scalar_result, 0.0, "scalar dot of orthogonal vectors should be 0.0");
}

// ════════════════════════════════════════════════════════════════════════
// 2. SCALE PARITY
// ════════════════════════════════════════════════════════════════════════

/// Scale parity: 128 elements, factor 2.5 (8-aligned, SIMD body only).
#[test]
fn test_simd_scale_parity_aligned() {
    let original = pseudo_rand(128, 800);
    let expected: Vec<f32> = original.iter().map(|&v| v * 2.5).collect();
    let mut actual = original.clone();
    simd_scale_f32(&mut actual, 2.5);
    assert_vec_parity(&actual, &expected, SCALE_ABS_TOL, SCALE_REL_TOL, "scale_aligned(128)");
}

/// Scale parity: 63 elements, factor 2.5 (exercises scalar tail).
#[test]
fn test_simd_scale_parity_unaligned() {
    let original = pseudo_rand(63, 900);
    let expected: Vec<f32> = original.iter().map(|&v| v * 2.5).collect();
    let mut actual = original.clone();
    simd_scale_f32(&mut actual, 2.5);
    assert_vec_parity(&actual, &expected, SCALE_ABS_TOL, SCALE_REL_TOL, "scale_unaligned(63)");
}

/// Scaling by 0.0 must produce all zeros.
#[test]
fn test_simd_scale_by_zero() {
    let mut data = pseudo_rand(128, 1000);
    simd_scale_f32(&mut data, 0.0);
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, 0.0, "scale_by_zero[{i}] should be 0.0, got {v}");
    }
}

/// Scaling by 1.0 must preserve original values exactly.
#[test]
fn test_simd_scale_by_one() {
    let original = pseudo_rand(128, 1100);
    let mut data = original.clone();
    simd_scale_f32(&mut data, 1.0);
    assert_vec_parity(&data, &original, 0.0, 0.0, "scale_by_one(128)");
}
