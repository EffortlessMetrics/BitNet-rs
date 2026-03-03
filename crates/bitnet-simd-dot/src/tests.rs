//! Comprehensive tests for bitnet-simd-dot.
//!
//! 70+ tests covering SIMD-vs-scalar parity, edge cases, alignment, and
//! property-based testing with proptest.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::suboptimal_flops
)]

use crate::dispatch::{
    SimdLevel, batched_dot_f32, binary_dot, dot_f32, dot_i8, fma_dot_f32, strided_dot_f32,
};
use crate::scalar;

// ════════════════════════════════════════════════════════════════════
// Helper: approximate equality for f32
// ════════════════════════════════════════════════════════════════════

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// ════════════════════════════════════════════════════════════════════
// SimdLevel
// ════════════════════════════════════════════════════════════════════

#[test]
fn simd_level_detect_does_not_panic() {
    let _level = SimdLevel::detect();
}

#[test]
fn simd_level_display() {
    assert_eq!(SimdLevel::Scalar.to_string(), "scalar");
    assert_eq!(SimdLevel::Sse41.to_string(), "sse4.1");
    assert_eq!(SimdLevel::Avx2.to_string(), "avx2");
    assert_eq!(SimdLevel::Avx512.to_string(), "avx512");
    assert_eq!(SimdLevel::Neon.to_string(), "neon");
}

#[test]
fn simd_level_equality() {
    assert_eq!(SimdLevel::Scalar, SimdLevel::Scalar);
    assert_ne!(SimdLevel::Scalar, SimdLevel::Avx2);
}

#[test]
fn simd_level_clone() {
    let level = SimdLevel::detect();
    let cloned = level;
    assert_eq!(level, cloned);
}

#[test]
fn simd_level_debug() {
    let s = format!("{:?}", SimdLevel::Avx2);
    assert!(s.contains("Avx2"));
}

#[test]
fn simd_level_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SimdLevel::Scalar);
    set.insert(SimdLevel::Avx2);
    assert_eq!(set.len(), 2);
}

// ════════════════════════════════════════════════════════════════════
// f32 dot product
// ════════════════════════════════════════════════════════════════════

#[test]
fn dot_f32_empty() {
    assert_eq!(dot_f32(&[], &[]), 0.0);
}

#[test]
fn dot_f32_single() {
    assert!(approx_eq(dot_f32(&[3.0], &[4.0]), 12.0, 1e-6));
}

#[test]
fn dot_f32_small() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-5));
}

#[test]
fn dot_f32_matches_scalar_len7() {
    let a: Vec<f32> = (0..7).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..7).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_f32_matches_scalar_len8() {
    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..8).map(|i| (i as f32) * 2.0).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_f32_matches_scalar_len15() {
    let a: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..15).map(|i| (i as f32 + 0.5) * 0.2).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_f32_matches_scalar_len16() {
    let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 * 0.05).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-3));
}

#[test]
fn dot_f32_matches_scalar_len31() {
    let a: Vec<f32> = (0..31).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..31).map(|i| (i as f32).cos()).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_f32_matches_scalar_len64() {
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..64).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_f32_matches_scalar_len100() {
    let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32) * 0.2 - 10.0).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-2));
}

#[test]
fn dot_f32_matches_scalar_len256() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-1));
}

#[test]
fn dot_f32_negative_values() {
    let a = [-1.0, -2.0, -3.0, -4.0];
    let b = [1.0, 2.0, 3.0, 4.0];
    let expected = -(1.0 + 4.0 + 9.0 + 16.0);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-5));
}

#[test]
fn dot_f32_zeros() {
    let a = [0.0f32; 32];
    let b = [1.0f32; 32];
    assert_eq!(dot_f32(&a, &b), 0.0);
}

#[test]
fn dot_f32_ones() {
    let n = 128;
    let a = vec![1.0f32; n];
    let b = vec![1.0f32; n];
    assert!(approx_eq(dot_f32(&a, &b), n as f32, 1e-5));
}

#[test]
#[should_panic(expected = "length mismatch")]
fn dot_f32_panics_on_mismatch() {
    let _ = dot_f32(&[1.0, 2.0], &[1.0]);
}

// ════════════════════════════════════════════════════════════════════
// i8 dot product
// ════════════════════════════════════════════════════════════════════

#[test]
fn dot_i8_empty() {
    assert_eq!(dot_i8(&[], &[]), 0);
}

#[test]
fn dot_i8_single() {
    assert_eq!(dot_i8(&[3], &[4]), 12);
}

#[test]
fn dot_i8_small() {
    let a: Vec<i8> = vec![1, 2, 3, 4, 5];
    let b: Vec<i8> = vec![5, 4, 3, 2, 1];
    let expected = 5 + 8 + 9 + 8 + 5;
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len16() {
    let a: Vec<i8> = (0..16).map(|i| (i * 7 % 127) as i8).collect();
    let b: Vec<i8> = (0..16).map(|i| ((i + 3) * 11 % 127) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len17() {
    let a: Vec<i8> = (0..17).map(|i| (i % 10) as i8).collect();
    let b: Vec<i8> = (0..17).map(|i| ((i + 1) % 10) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len32() {
    let a: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
    let b: Vec<i8> = (0..32).map(|i| 16 - (i as i8)).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len33() {
    let a: Vec<i8> = (0..33).map(|i| ((i * 3) % 127) as i8).collect();
    let b: Vec<i8> = (0..33).map(|i| ((i * 5) % 127) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len64() {
    let a: Vec<i8> = (0..64).map(|i| ((i * 7) % 256 - 128) as i8).collect();
    let b: Vec<i8> = (0..64).map(|i| ((i * 11) % 256 - 128) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_matches_scalar_len100() {
    let a: Vec<i8> = (0..100).map(|i| ((i * 3) % 256 - 128) as i8).collect();
    let b: Vec<i8> = (0..100).map(|i| ((i * 7) % 256 - 128) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

#[test]
fn dot_i8_negative_values() {
    let a: Vec<i8> = vec![-1, -2, -3, -4];
    let b: Vec<i8> = vec![1, 2, 3, 4];
    assert_eq!(dot_i8(&a, &b), -(1 + 4 + 9 + 16));
}

#[test]
fn dot_i8_max_values() {
    let a = vec![127i8; 4];
    let b = vec![127i8; 4];
    assert_eq!(dot_i8(&a, &b), 4 * 127 * 127);
}

#[test]
fn dot_i8_min_max() {
    let a = vec![-128i8; 4];
    let b = vec![127i8; 4];
    assert_eq!(dot_i8(&a, &b), 4 * (-128) * 127);
}

#[test]
fn dot_i8_zeros() {
    let a = vec![0i8; 64];
    let b: Vec<i8> = (0..64).map(|i| i as i8).collect();
    assert_eq!(dot_i8(&a, &b), 0);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn dot_i8_panics_on_mismatch() {
    let _ = dot_i8(&[1, 2], &[1]);
}

// ════════════════════════════════════════════════════════════════════
// Binary (popcount) dot product
// ════════════════════════════════════════════════════════════════════

#[test]
fn binary_dot_empty() {
    assert_eq!(binary_dot(&[], &[]), 0);
}

#[test]
fn binary_dot_identical() {
    let a = vec![0xFFFF_FFFF_FFFF_FFFFu64; 4];
    let b = a.clone();
    assert_eq!(binary_dot(&a, &b), 4 * 64);
}

#[test]
fn binary_dot_opposite() {
    let a = vec![0u64; 4];
    let b = vec![0xFFFF_FFFF_FFFF_FFFFu64; 4];
    assert_eq!(binary_dot(&a, &b), 0);
}

#[test]
fn binary_dot_half_match() {
    let a = vec![0x00FF_00FF_00FF_00FFu64];
    let b = vec![0xFFFF_FFFF_FFFF_FFFFu64];
    // 32 bits differ, 32 match
    assert_eq!(binary_dot(&a, &b), 32);
}

#[test]
fn binary_dot_single_bit() {
    let a = vec![1u64];
    let b = vec![1u64];
    // 63 zero-match + 1 one-match = 64
    assert_eq!(binary_dot(&a, &b), 64);
}

#[test]
fn binary_dot_xor_check() {
    let a = vec![0b1010_1010u64];
    let b = vec![0b0101_0101u64];
    // All 8 low bits differ; rest are zeros (matching)
    // diff bits = 8, total bits = 64, matching = 56
    assert_eq!(binary_dot(&a, &b), 56);
}

#[test]
fn binary_dot_matches_scalar() {
    let a: Vec<u64> = vec![0xDEAD_BEEF_CAFE_BABEu64, 0x1234_5678_9ABC_DEF0u64];
    let b: Vec<u64> = vec![0xBEEF_DEAD_BABE_CAFEu64, 0xFEDC_BA98_7654_3210u64];
    let expected = scalar::binary_dot(&a, &b);
    assert_eq!(binary_dot(&a, &b), expected);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn binary_dot_panics_on_mismatch() {
    let _ = binary_dot(&[1], &[1, 2]);
}

// ════════════════════════════════════════════════════════════════════
// Fused multiply-accumulate
// ════════════════════════════════════════════════════════════════════

#[test]
fn fma_dot_f32_basic() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let c = [0.5, 1.0];
    let d = [2.0, 3.0];
    let expected = (4.0 + 10.0 + 18.0) + (1.0 + 3.0);
    assert!(approx_eq(fma_dot_f32(&a, &b, &c, &d), expected, 1e-5));
}

#[test]
fn fma_dot_f32_empty_ab() {
    let c = [1.0, 2.0];
    let d = [3.0, 4.0];
    assert!(approx_eq(fma_dot_f32(&[], &[], &c, &d), 11.0, 1e-5));
}

#[test]
fn fma_dot_f32_empty_cd() {
    let a = [1.0, 2.0];
    let b = [3.0, 4.0];
    assert!(approx_eq(fma_dot_f32(&a, &b, &[], &[]), 11.0, 1e-5));
}

#[test]
fn fma_dot_f32_both_empty() {
    assert_eq!(fma_dot_f32(&[], &[], &[], &[]), 0.0);
}

#[test]
fn fma_dot_f32_matches_scalar_large() {
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.2).collect();
    let c: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3).collect();
    let d: Vec<f32> = (0..32).map(|i| (i as f32) * 0.4).collect();
    let expected = scalar::fma_dot_f32(&a, &b, &c, &d);
    assert!(approx_eq(fma_dot_f32(&a, &b, &c, &d), expected, 1e-2));
}

#[test]
#[should_panic(expected = "a/b length mismatch")]
fn fma_dot_f32_panics_ab_mismatch() {
    let _ = fma_dot_f32(&[1.0], &[1.0, 2.0], &[], &[]);
}

#[test]
#[should_panic(expected = "c/d length mismatch")]
fn fma_dot_f32_panics_cd_mismatch() {
    let _ = fma_dot_f32(&[], &[], &[1.0], &[1.0, 2.0]);
}

// ════════════════════════════════════════════════════════════════════
// Strided dot product
// ════════════════════════════════════════════════════════════════════

#[test]
fn strided_dot_f32_stride1() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [4.0, 3.0, 2.0, 1.0];
    // stride=1 is a normal dot product
    let expected = dot_f32(&a, &b);
    assert!(approx_eq(strided_dot_f32(&a, &b, 1), expected, 1e-5));
}

#[test]
fn strided_dot_f32_stride2() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    // indices 0, 2, 4 → 1*6 + 3*4 + 5*2 = 6 + 12 + 10 = 28
    assert!(approx_eq(strided_dot_f32(&a, &b, 2), 28.0, 1e-5));
}

#[test]
fn strided_dot_f32_stride3() {
    let a = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0];
    let b = [10.0, 0.0, 0.0, 20.0, 0.0, 0.0, 30.0];
    // indices 0, 3, 6 → 10 + 40 + 90 = 140
    assert!(approx_eq(strided_dot_f32(&a, &b, 3), 140.0, 1e-5));
}

#[test]
fn strided_dot_f32_empty() {
    assert_eq!(strided_dot_f32(&[], &[], 1), 0.0);
}

#[test]
#[should_panic(expected = "stride must be > 0")]
fn strided_dot_f32_panics_on_zero_stride() {
    let _ = strided_dot_f32(&[1.0], &[1.0], 0);
}

#[test]
fn strided_dot_f32_large_stride() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    // stride=10 → only index 0: 1*10 = 10
    assert!(approx_eq(strided_dot_f32(&a, &b, 10), 10.0, 1e-5));
}

// ════════════════════════════════════════════════════════════════════
// Batched dot product
// ════════════════════════════════════════════════════════════════════

#[test]
fn batched_dot_f32_single_row() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    let result = batched_dot_f32(&a, &b, 1, 3);
    assert_eq!(result.len(), 1);
    assert!(approx_eq(result[0], 32.0, 1e-5));
}

#[test]
fn batched_dot_f32_two_rows() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let result = batched_dot_f32(&a, &b, 2, 3);
    assert_eq!(result.len(), 2);
    assert!(approx_eq(result[0], 6.0 + 10.0 + 12.0, 1e-5));
    assert!(approx_eq(result[1], 12.0 + 10.0 + 6.0, 1e-5));
}

#[test]
fn batched_dot_f32_many_rows() {
    let rows = 10;
    let cols = 8;
    let a = vec![1.0f32; rows * cols];
    let b = vec![1.0f32; rows * cols];
    let result = batched_dot_f32(&a, &b, rows, cols);
    assert_eq!(result.len(), rows);
    for r in &result {
        assert!(approx_eq(*r, cols as f32, 1e-5));
    }
}

#[test]
fn batched_dot_f32_empty() {
    let result = batched_dot_f32(&[], &[], 0, 0);
    assert!(result.is_empty());
}

#[test]
#[should_panic(expected = "a length mismatch")]
fn batched_dot_f32_panics_a_mismatch() {
    let _ = batched_dot_f32(&[1.0, 2.0], &[1.0, 2.0, 3.0], 1, 3);
}

#[test]
#[should_panic(expected = "b length mismatch")]
fn batched_dot_f32_panics_b_mismatch() {
    let _ = batched_dot_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0], 1, 3);
}

// ════════════════════════════════════════════════════════════════════
// Alignment edge-cases (odd sizes that don't evenly divide SIMD widths)
// ════════════════════════════════════════════════════════════════════

#[test]
fn dot_f32_len1_alignment() {
    assert!(approx_eq(dot_f32(&[42.0], &[0.5]), 21.0, 1e-6));
}

#[test]
fn dot_f32_len3_alignment() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 1.0, 1.0];
    assert!(approx_eq(dot_f32(&a, &b), 6.0, 1e-5));
}

#[test]
fn dot_f32_len5_alignment() {
    let a: Vec<f32> = (1..=5).map(|i| i as f32).collect();
    let b = vec![1.0f32; 5];
    assert!(approx_eq(dot_f32(&a, &b), 15.0, 1e-5));
}

#[test]
fn dot_f32_len9_alignment() {
    let a: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let b = vec![1.0f32; 9];
    assert!(approx_eq(dot_f32(&a, &b), 36.0, 1e-5));
}

#[test]
fn dot_f32_len17_alignment() {
    let a: Vec<f32> = (0..17).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..17).map(|i| (i as f32) * 0.1).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 1e-4));
}

#[test]
fn dot_i8_len1_alignment() {
    assert_eq!(dot_i8(&[7], &[3]), 21);
}

#[test]
fn dot_i8_len3_alignment() {
    assert_eq!(dot_i8(&[1, 2, 3], &[4, 5, 6]), 32);
}

#[test]
fn dot_i8_len15_alignment() {
    let a: Vec<i8> = (1..=15).map(|i| i as i8).collect();
    let b = vec![1i8; 15];
    let expected: i32 = (1..=15).sum();
    assert_eq!(dot_i8(&a, &b), expected);
}

// ════════════════════════════════════════════════════════════════════
// Large vectors (stress-test SIMD paths)
// ════════════════════════════════════════════════════════════════════

#[test]
fn dot_f32_len1024() {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..1024).map(|i| 1.0 - (i as f32) * 0.001).collect();
    let expected = scalar::dot_f32(&a, &b);
    assert!(approx_eq(dot_f32(&a, &b), expected, 0.5));
}

#[test]
fn dot_i8_len1024() {
    let a: Vec<i8> = (0..1024).map(|i| ((i * 7) % 256 - 128) as i8).collect();
    let b: Vec<i8> = (0..1024).map(|i| ((i * 11) % 256 - 128) as i8).collect();
    let expected = scalar::dot_i8(&a, &b);
    assert_eq!(dot_i8(&a, &b), expected);
}

// ════════════════════════════════════════════════════════════════════
// Property tests with proptest
// ════════════════════════════════════════════════════════════════════

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn dot_f32_matches_scalar(
            a in proptest::collection::vec(-100.0f32..100.0, 0..512),
        ) {
            let b: Vec<f32> = a.iter().map(|x| x * 0.5 + 1.0).collect();
            let expected = scalar::dot_f32(&a, &b);
            let actual = dot_f32(&a, &b);
            let tol = expected.abs() * 1e-4 + 1e-4;
            prop_assert!((actual - expected).abs() <= tol,
                "len={}, expected={expected}, actual={actual}", a.len());
        }

        #[test]
        fn dot_i8_matches_scalar(
            a in proptest::collection::vec(-128i8..=127, 0..512),
        ) {
            let b: Vec<i8> = a.iter().map(|&x| x.wrapping_add(1)).collect();
            let expected = scalar::dot_i8(&a, &b);
            let actual = dot_i8(&a, &b);
            prop_assert_eq!(actual, expected, "len={}", a.len());
        }

        #[test]
        fn dot_f32_commutative(
            a in proptest::collection::vec(-10.0f32..10.0, 1..128),
        ) {
            let b: Vec<f32> = a.iter().map(|x| x + 1.0).collect();
            let ab = dot_f32(&a, &b);
            let ba = dot_f32(&b, &a);
            let tol = ab.abs() * 1e-5 + 1e-5;
            prop_assert!((ab - ba).abs() <= tol,
                "dot(a,b)={ab}, dot(b,a)={ba}");
        }

        #[test]
        fn dot_f32_self_non_negative(
            a in proptest::collection::vec(-10.0f32..10.0, 0..256),
        ) {
            let result = dot_f32(&a, &a);
            prop_assert!(result >= -1e-6, "self-dot should be >= 0, got {result}");
        }

        #[test]
        fn dot_i8_commutative(
            a in proptest::collection::vec(-128i8..=127, 1..128),
        ) {
            let b: Vec<i8> = a.iter().map(|&x| x.wrapping_add(3)).collect();
            let ab = dot_i8(&a, &b);
            let ba = dot_i8(&b, &a);
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn binary_dot_self_is_total_bits(
            a in proptest::collection::vec(any::<u64>(), 1..32),
        ) {
            let result = binary_dot(&a, &a);
            let expected = (a.len() as u32) * 64;
            prop_assert_eq!(result, expected);
        }

        #[test]
        fn binary_dot_commutative(
            a in proptest::collection::vec(any::<u64>(), 1..32),
        ) {
            let b: Vec<u64> = a.iter().map(|x| x.wrapping_add(1)).collect();
            let ab = binary_dot(&a, &b);
            let ba = binary_dot(&b, &a);
            prop_assert_eq!(ab, ba);
        }

        #[test]
        fn binary_dot_bounded(
            a in proptest::collection::vec(any::<u64>(), 1..32),
            b in proptest::collection::vec(any::<u64>(), 1..32),
        ) {
            let len = a.len().min(b.len());
            let a = &a[..len];
            let b = &b[..len];
            let result = binary_dot(a, b);
            let max = (len as u32) * 64;
            prop_assert!(result <= max, "result={result} > max={max}");
        }

        #[test]
        fn fma_dot_f32_matches_scalar(
            a in proptest::collection::vec(-10.0f32..10.0, 0..128),
            c in proptest::collection::vec(-10.0f32..10.0, 0..128),
        ) {
            let b: Vec<f32> = a.iter().map(|x| x * 0.5).collect();
            let d: Vec<f32> = c.iter().map(|x| x * 0.3).collect();
            let expected = scalar::fma_dot_f32(&a, &b, &c, &d);
            let actual = fma_dot_f32(&a, &b, &c, &d);
            let tol = expected.abs() * 1e-4 + 1e-3;
            prop_assert!((actual - expected).abs() <= tol,
                "expected={expected}, actual={actual}");
        }

        #[test]
        fn strided_matches_manual(
            a in proptest::collection::vec(-10.0f32..10.0, 1..128),
            stride in 1usize..8,
        ) {
            let b: Vec<f32> = a.iter().map(|x| x + 0.5).collect();
            let result = strided_dot_f32(&a, &b, stride);
            let expected = scalar::strided_dot_f32(&a, &b, stride);
            let tol = expected.abs() * 1e-5 + 1e-5;
            prop_assert!((result - expected).abs() <= tol,
                "stride={stride}, expected={expected}, got={result}");
        }

        #[test]
        fn batched_row_count(
            rows in 1usize..16,
            cols in 1usize..64,
        ) {
            let a = vec![1.0f32; rows * cols];
            let b = vec![1.0f32; rows * cols];
            let result = batched_dot_f32(&a, &b, rows, cols);
            prop_assert_eq!(result.len(), rows);
            for &val in &result {
                let tol = (cols as f32) * 1e-5 + 1e-5;
                prop_assert!((val - cols as f32).abs() <= tol);
            }
        }
    }
}
