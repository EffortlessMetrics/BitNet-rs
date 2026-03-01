//! Edge-case integration tests for `bitnet_kernels::cpu::quantized_matmul`.
//!
//! These tests exercise the public API: `i2s_matmul_f32`, `i2s_matmul_blocked`,
//! `dequantize_and_matmul`, and `pack_i2s` from outside the crate boundary.

use bitnet_kernels::cpu::quantized_matmul::{
    dequantize_and_matmul, i2s_matmul_blocked, i2s_matmul_f32, pack_i2s,
};

const TOL: f32 = 1e-5;

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (tol={tol})");
    }
}

/// Helper: pack a ternary weight matrix (k x n, row-major) into I2_S column-major
/// packed bytes and uniform scale=1.0 vectors.
fn pack_weight_matrix(
    weights: &[i8],
    k: usize,
    n: usize,
    block_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);
    let mut packed = vec![0u8; packed_k * n];
    for col in 0..n {
        for row in 0..k {
            let val = weights[row * n + col];
            let code: u8 = match val {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = col * packed_k + row / 4;
            let bit_off = (row % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    let scales = vec![1.0f32; n * num_blocks_k];
    (packed, scales)
}

/// Naive reference matmul for verification
fn naive_matmul(a: &[f32], w_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * w_f32[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Run all three kernel functions and verify they match expected output
fn run_all(
    act: &[f32],
    packed: &[u8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
    bs: usize,
    expected: &[f32],
    tol: f32,
) {
    let mut out = vec![0.0f32; m * n];
    i2s_matmul_f32(act, packed, scales, &mut out, m, n, k, bs).unwrap();
    assert_close(&out, expected, tol);

    i2s_matmul_blocked(act, packed, scales, &mut out, m, n, k, bs).unwrap();
    assert_close(&out, expected, tol);

    dequantize_and_matmul(act, packed, scales, &mut out, m, n, k, bs).unwrap();
    assert_close(&out, expected, tol);
}

// =========================================================================
// pack_i2s
// =========================================================================

#[test]
fn pack_roundtrip_all_combos() {
    // Test all 81 possible ternary 4-tuples
    for a in [-1i8, 0, 1] {
        for b in [-1i8, 0, 1] {
            for c in [-1i8, 0, 1] {
                for d in [-1i8, 0, 1] {
                    let vals = [a, b, c, d];
                    let byte = pack_i2s(vals);
                    // Verify round-trip
                    for (i, &expected) in vals.iter().enumerate() {
                        let bits = (byte >> (i * 2)) & 0x03;
                        let decoded = match bits {
                            0b00 => 0,
                            0b01 => 1,
                            0b11 => -1,
                            _ => 0,
                        };
                        assert_eq!(
                            decoded, expected,
                            "pack_i2s roundtrip failed for {vals:?} at pos {i}"
                        );
                    }
                }
            }
        }
    }
}

// =========================================================================
// Correctness: identity matrix
// =========================================================================

#[test]
fn identity_4x4_block32() {
    let m = 4;
    let n = 4;
    let k = 4;
    let bs = 32;
    #[rustfmt::skip]
    let w: Vec<i8> = vec![
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ];
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act: Vec<f32> =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    // A * I = A
    run_all(&act, &packed, &scales, m, n, k, bs, &act, TOL);
}

// =========================================================================
// Correctness: all-ones weights
// =========================================================================

#[test]
fn all_ones_weight_sums_each_row() {
    let m = 3;
    let n = 2;
    let k = 4;
    let bs = 32;
    let w = vec![1i8; k * n];
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
    let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    let expected = naive_matmul(&act, &w_f32, m, n, k);
    run_all(&act, &packed, &scales, m, n, k, bs, &expected, TOL);
}

// =========================================================================
// Correctness: all -1 weights (negation)
// =========================================================================

#[test]
fn all_neg_ones_negate_row_sum() {
    let m = 2;
    let n = 1;
    let k = 4;
    let bs = 32;
    let w = vec![-1i8; k];
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Row sums negated: -(1+2+3+4) = -10, -(5+6+7+8) = -26
    run_all(&act, &packed, &scales, m, n, k, bs, &[-10.0, -26.0], TOL);
}

// =========================================================================
// Edge case: 1x1
// =========================================================================

#[test]
fn scalar_1x1_block32() {
    let w = vec![1i8];
    let (packed, scales) = pack_weight_matrix(&w, 1, 1, 32);
    run_all(&[42.0], &packed, &scales, 1, 1, 1, 32, &[42.0], TOL);
}

#[test]
fn scalar_1x1_block256() {
    let w = vec![-1i8];
    let (packed, scales) = pack_weight_matrix(&w, 1, 1, 256);
    run_all(&[7.0], &packed, &scales, 1, 1, 1, 256, &[-7.0], TOL);
}

// =========================================================================
// Edge case: k not multiple of 4
// =========================================================================

#[test]
fn k_not_multiple_of_4() {
    let m = 2;
    let n = 2;
    let k = 5;
    let bs = 32;
    let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act: Vec<f32> = (0..m * k).map(|i| i as f32 + 0.5).collect();
    let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    let expected = naive_matmul(&act, &w_f32, m, n, k);
    run_all(&act, &packed, &scales, m, n, k, bs, &expected, TOL);
}

// =========================================================================
// Edge case: k exactly equals block_size
// =========================================================================

#[test]
fn k_equals_block_size_32() {
    let m = 2;
    let n = 2;
    let k = 32;
    let bs = 32;
    let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
    let expected = naive_matmul(&act, &w_f32, m, n, k);
    run_all(&act, &packed, &scales, m, n, k, bs, &expected, TOL);
}

// =========================================================================
// Non-unit scales
// =========================================================================

#[test]
fn non_unit_scale_amplifies_output() {
    let m: usize = 1;
    let n: usize = 1;
    let k: usize = 4;
    let bs: usize = 32;
    let packed_k = k.div_ceil(4); // 1 byte
    // All +1 weights
    let packed = vec![0b01_01_01_01u8; packed_k];
    let scales = vec![3.0f32]; // Scale = 3
    let act = vec![1.0f32; k];
    let mut out = vec![0.0f32; 1];
    i2s_matmul_f32(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
    // 4 * 1.0 * 1 * 3.0 = 12.0
    assert_close(&out, &[12.0], TOL);
}

// =========================================================================
// Multi-block with different scales
// =========================================================================

#[test]
fn multi_block_different_scales() {
    let m: usize = 1;
    let n: usize = 1;
    let k: usize = 64;
    let bs: usize = 32;
    let packed_k = k.div_ceil(4); // 16 bytes
    // All +1 weights
    let packed = vec![0b01_01_01_01u8; packed_k];
    let mut scales = vec![0.0f32; 2]; // 2 blocks
    scales[0] = 1.0; // block 0
    scales[1] = 5.0; // block 1
    let act = vec![1.0f32; k];
    let mut out = vec![0.0f32; 1];
    i2s_matmul_f32(&act, &packed, &scales, &mut out, m, n, k, bs).unwrap();
    // block0: 32 * 1.0 = 32, block1: 32 * 5.0 = 160 → 192
    assert_close(&out, &[192.0], TOL);
}

// =========================================================================
// All three kernels agree on larger random-ish pattern
// =========================================================================

#[test]
fn three_kernels_agree_16x8_block32() {
    let m = 16;
    let n = 8;
    let k = 48;
    let bs = 32;
    let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();

    let mut out1 = vec![0.0f32; m * n];
    let mut out2 = vec![0.0f32; m * n];
    let mut out3 = vec![0.0f32; m * n];

    i2s_matmul_f32(&act, &packed, &scales, &mut out1, m, n, k, bs).unwrap();
    i2s_matmul_blocked(&act, &packed, &scales, &mut out2, m, n, k, bs).unwrap();
    dequantize_and_matmul(&act, &packed, &scales, &mut out3, m, n, k, bs).unwrap();

    assert_close(&out1, &out2, 1e-4);
    assert_close(&out1, &out3, 1e-4);
}

#[test]
fn three_kernels_agree_block256() {
    let m = 4;
    let n = 4;
    let k = 256;
    let bs = 256;
    let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
    let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.002).cos()).collect();

    let mut out1 = vec![0.0f32; m * n];
    let mut out2 = vec![0.0f32; m * n];
    let mut out3 = vec![0.0f32; m * n];

    i2s_matmul_f32(&act, &packed, &scales, &mut out1, m, n, k, bs).unwrap();
    i2s_matmul_blocked(&act, &packed, &scales, &mut out2, m, n, k, bs).unwrap();
    dequantize_and_matmul(&act, &packed, &scales, &mut out3, m, n, k, bs).unwrap();

    assert_close(&out1, &out2, 1e-3);
    assert_close(&out1, &out3, 1e-3);
}

// =========================================================================
// Error paths
// =========================================================================

#[test]
fn error_zero_dimensions() {
    let act = vec![1.0f32; 4];
    let packed = vec![0u8; 4];
    let scales = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 4];

    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 0, 2, 2, 32).is_err());
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 0, 2, 32).is_err());
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 0, 32).is_err());

    assert!(i2s_matmul_blocked(&act, &packed, &scales, &mut out, 0, 2, 2, 32).is_err());
    assert!(dequantize_and_matmul(&act, &packed, &scales, &mut out, 0, 2, 2, 32).is_err());
}

#[test]
fn error_block_size_zero() {
    let act = vec![1.0f32; 4];
    let packed = vec![0u8; 2];
    let scales = vec![1.0f32; 2];
    let mut out = vec![0.0f32; 4];

    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 2, 0).is_err());
    assert!(i2s_matmul_blocked(&act, &packed, &scales, &mut out, 2, 2, 2, 0).is_err());
    assert!(dequantize_and_matmul(&act, &packed, &scales, &mut out, 2, 2, 2, 0).is_err());
}

#[test]
fn error_activation_too_small() {
    let act = vec![1.0f32; 2];
    let packed = vec![0u8; 4];
    let scales = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 4];
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 4, 32).is_err());
}

#[test]
fn error_output_too_small() {
    let act = vec![1.0f32; 4];
    let packed = vec![0u8; 2];
    let scales = vec![1.0f32; 2];
    let mut out = vec![0.0f32; 1];
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 2, 32).is_err());
}

#[test]
fn error_weights_too_small() {
    let act = vec![1.0f32; 8]; // 2x4
    let packed = vec![0u8; 1]; // Too small for 4x2 packed
    let scales = vec![1.0f32; 2];
    let mut out = vec![0.0f32; 4];
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 2, 2, 4, 32).is_err());
}

#[test]
fn error_scales_too_small() {
    let act = vec![1.0f32; 128]; // 1x128
    let packed = vec![0u8; 32]; // 128/4 = 32 bytes for 1 col
    let scales = vec![1.0f32; 1]; // Need 4 blocks for block_size=32 with k=128
    let mut out = vec![0.0f32; 1];
    assert!(i2s_matmul_f32(&act, &packed, &scales, &mut out, 1, 1, 128, 32).is_err());
}
