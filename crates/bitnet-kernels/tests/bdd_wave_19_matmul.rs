//! BDD Wave 19 — Matrix multiplication tests.
//!
//! Covers batched f32 matmul, identity/zero matrices, rectangular shapes,
//! I2_S quantized matmul, and cross-method parity between scalar/blocked paths.

use bitnet_kernels::cpu::batch::batched_matmul;
use bitnet_kernels::cpu::quantized_matmul::{
    dequantize_and_matmul, i2s_matmul_blocked, i2s_matmul_f32,
};

const TOL: f32 = 1e-5;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
    }
}

// ── F32 Batched MatMul ─────────────────────────────────────────────

#[test]
fn given_identity_matrix_when_batched_matmul_then_output_equals_input() {
    // I₂ × A = A
    #[rustfmt::skip]
    let identity = vec![
        1.0, 0.0,
        0.0, 1.0,
    ];
    let a = vec![3.0, 7.0, 5.0, 11.0]; // 2×2
    let result = batched_matmul(&identity, &a, 1, 2, 2, 2).unwrap();
    approx_eq(&result, &a, TOL);
}

#[test]
fn given_zero_matrix_when_batched_matmul_then_output_all_zeros() {
    let zero = vec![0.0; 4]; // 2×2
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let result = batched_matmul(&zero, &a, 1, 2, 2, 2).unwrap();
    approx_eq(&result, &[0.0; 4], TOL);
}

#[test]
fn given_small_matrices_when_batched_matmul_then_correct_product() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A*B = [[19,22],[43,50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = batched_matmul(&a, &b, 1, 2, 2, 2).unwrap();
    approx_eq(&result, &[19.0, 22.0, 43.0, 50.0], TOL);
}

#[test]
fn given_rectangular_matrices_when_batched_matmul_then_correct_shape() {
    // A: 2×3, B: 3×1 → C: 2×1
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 1.0, 1.0];
    let result = batched_matmul(&a, &b, 1, 2, 3, 1).unwrap();
    assert_eq!(result.len(), 2);
    approx_eq(&result, &[6.0, 15.0], TOL);
}

#[test]
fn given_batch_of_two_when_batched_matmul_then_independent_products() {
    // Batch 0: [[1,0],[0,1]] × [[2,3],[4,5]] = [[2,3],[4,5]]
    // Batch 1: [[0,1],[1,0]] × [[2,3],[4,5]] = [[4,5],[2,3]]
    #[rustfmt::skip]
    let a = vec![
        1.0, 0.0, 0.0, 1.0,   // batch 0: identity
        0.0, 1.0, 1.0, 0.0,   // batch 1: swap rows
    ];
    let b = vec![2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0];
    let result = batched_matmul(&a, &b, 2, 2, 2, 2).unwrap();
    approx_eq(&result, &[2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 2.0, 3.0], TOL);
}

#[test]
fn given_dimension_mismatch_when_batched_matmul_then_error() {
    let a = vec![1.0; 6]; // 2×3
    let b = vec![1.0; 4]; // should be 3×?, but is too short
    let result = batched_matmul(&a, &b, 1, 2, 3, 2);
    assert!(result.is_err());
}

#[test]
fn given_single_element_when_batched_matmul_then_scalar_product() {
    let a = vec![3.0];
    let b = vec![7.0];
    let result = batched_matmul(&a, &b, 1, 1, 1, 1).unwrap();
    approx_eq(&result, &[21.0], TOL);
}

#[test]
fn given_column_times_row_when_batched_matmul_then_outer_product() {
    // col [1,2,3] × row [4,5] → 3×2 outer product
    let a = vec![1.0, 2.0, 3.0]; // 3×1
    let b = vec![4.0, 5.0]; // 1×2
    let result = batched_matmul(&a, &b, 1, 3, 1, 2).unwrap();
    approx_eq(&result, &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], TOL);
}

// ── I2_S Quantized MatMul ──────────────────────────────────────────

fn make_i2s_weights(values: &[i8], n_cols: usize, k: usize) -> (Vec<u8>, usize) {
    let packed_k = k.div_ceil(4);
    let mut packed = vec![0u8; n_cols * packed_k];
    for col in 0..n_cols {
        for idx in 0..k {
            let v = values[col * k + idx];
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            };
            let byte_idx = col * packed_k + idx / 4;
            let bit_off = (idx % 4) * 2;
            packed[byte_idx] |= code << bit_off;
        }
    }
    (packed, packed_k)
}

#[test]
fn given_identity_i2s_weights_when_matmul_then_scales_applied() {
    // k=4, n=1, block_size=4; weight column = [1,0,0,0] with scale=1.0
    let (packed, _) = make_i2s_weights(&[1, 0, 0, 0], 1, 4);
    let activations = vec![5.0, 3.0, 2.0, 1.0];
    let scales = vec![1.0];
    let mut out = vec![0.0];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 1, 4, 4).unwrap();
    approx_eq(&out, &[5.0], TOL);
}

#[test]
fn given_all_ones_i2s_when_matmul_then_sum_of_activations() {
    let k = 4;
    let (packed, _) = make_i2s_weights(&[1, 1, 1, 1], 1, k);
    let activations = vec![1.0, 2.0, 3.0, 4.0];
    let scales = vec![1.0];
    let mut out = vec![0.0];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 1, k, k).unwrap();
    approx_eq(&out, &[10.0], TOL);
}

#[test]
fn given_minus_one_weights_when_matmul_then_negated_sum() {
    let k = 4;
    let (packed, _) = make_i2s_weights(&[-1, -1, -1, -1], 1, k);
    let activations = vec![1.0, 2.0, 3.0, 4.0];
    let scales = vec![1.0];
    let mut out = vec![0.0];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 1, k, k).unwrap();
    approx_eq(&out, &[-10.0], TOL);
}

#[test]
fn given_scale_factor_when_i2s_matmul_then_output_scaled() {
    let k = 4;
    let (packed, _) = make_i2s_weights(&[1, 1, 1, 1], 1, k);
    let activations = vec![1.0, 1.0, 1.0, 1.0];
    let scales = vec![0.5];
    let mut out = vec![0.0];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 1, k, k).unwrap();
    approx_eq(&out, &[2.0], TOL); // 4 * 1.0 * 0.5
}

#[test]
fn given_two_output_columns_when_i2s_matmul_then_both_correct() {
    let k = 4;
    // col0 = [1,1,0,0], col1 = [0,0,1,1]
    let weights: Vec<i8> = vec![1, 1, 0, 0, 0, 0, 1, 1];
    let (packed, _) = make_i2s_weights(&weights, 2, k);
    let activations = vec![1.0, 2.0, 3.0, 4.0];
    let scales = vec![1.0, 1.0]; // one per column
    let mut out = vec![0.0; 2];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 1, 2, k, k).unwrap();
    approx_eq(&out, &[3.0, 7.0], TOL);
}

#[test]
fn given_multiple_rows_when_i2s_matmul_then_each_row_independent() {
    let k = 4;
    let (packed, _) = make_i2s_weights(&[1, 1, 1, 1], 1, k);
    // Two activation rows
    let activations = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let scales = vec![1.0];
    let mut out = vec![0.0; 2];
    i2s_matmul_f32(&activations, &packed, &scales, &mut out, 2, 1, k, k).unwrap();
    approx_eq(&out, &[4.0, 8.0], TOL);
}

// ── Cross-Method Parity ────────────────────────────────────────────

#[test]
fn given_same_inputs_when_scalar_and_blocked_i2s_then_identical_output() {
    let m = 2;
    let n = 2;
    let k = 8;
    let block_size = 4;
    let weights: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0];
    let (packed, _) = make_i2s_weights(&weights, n, k);
    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let num_blocks = k.div_ceil(block_size);
    let scales: Vec<f32> = (0..n * num_blocks).map(|i| 0.5 + i as f32 * 0.1).collect();

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_blocked = vec![0.0f32; m * n];

    i2s_matmul_f32(&activations, &packed, &scales, &mut out_scalar, m, n, k, block_size).unwrap();
    i2s_matmul_blocked(&activations, &packed, &scales, &mut out_blocked, m, n, k, block_size)
        .unwrap();

    approx_eq(&out_scalar, &out_blocked, TOL);
}

#[test]
fn given_same_inputs_when_scalar_and_dequant_matmul_then_identical_output() {
    let m = 2;
    let n = 2;
    let k = 8;
    let block_size = 4;
    let weights: Vec<i8> = vec![1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
    let (packed, _) = make_i2s_weights(&weights, n, k);
    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let num_blocks = k.div_ceil(block_size);
    let scales: Vec<f32> = vec![1.0; n * num_blocks];

    let mut out_scalar = vec![0.0f32; m * n];
    let mut out_dequant = vec![0.0f32; m * n];

    i2s_matmul_f32(&activations, &packed, &scales, &mut out_scalar, m, n, k, block_size).unwrap();
    dequantize_and_matmul(&activations, &packed, &scales, &mut out_dequant, m, n, k, block_size)
        .unwrap();

    approx_eq(&out_scalar, &out_dequant, TOL);
}

#[test]
fn given_larger_matrix_when_all_three_methods_agree_then_parity_confirmed() {
    let m = 4;
    let n = 3;
    let k = 16;
    let block_size = 8;

    // Deterministic ternary pattern
    let weights: Vec<i8> = (0..n * k).map(|i| [0, 1, -1, 1, 0, -1][i % 6]).collect();
    let (packed, _) = make_i2s_weights(&weights, n, k);
    let activations: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let num_blocks = k.div_ceil(block_size);
    let scales: Vec<f32> = (0..n * num_blocks).map(|i| 0.8 + (i % 3) as f32 * 0.1).collect();

    let mut out_a = vec![0.0f32; m * n];
    let mut out_b = vec![0.0f32; m * n];
    let mut out_c = vec![0.0f32; m * n];

    i2s_matmul_f32(&activations, &packed, &scales, &mut out_a, m, n, k, block_size).unwrap();
    i2s_matmul_blocked(&activations, &packed, &scales, &mut out_b, m, n, k, block_size).unwrap();
    dequantize_and_matmul(&activations, &packed, &scales, &mut out_c, m, n, k, block_size).unwrap();

    approx_eq(&out_a, &out_b, TOL);
    approx_eq(&out_a, &out_c, TOL);
}
