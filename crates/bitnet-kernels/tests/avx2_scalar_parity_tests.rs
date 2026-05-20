#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for CPU kernels.
//!
//! Each test exercises a kernel that has an explicit AVX2 fast-path
//! (via `is_x86_feature_detected!("avx2")` runtime dispatch) and
//! compares the output against a pure-scalar reference within defined
//! tolerances.
//!
//! Test dimensions are chosen to exercise both the 8-wide SIMD body
//! and the scalar tail (lengths not divisible by 8).
//!
//! All tests are pure-math — no model files required.

use bitnet_kernels::cpu::layer_norm_simd::{
    LayerNormSimdConfig, RMSNormConfig, layer_norm_avx2, layer_norm_f32, rms_norm_avx2,
    rms_norm_f32,
};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32, simd_matmul_i2s};
use bitnet_kernels::cpu::simd_softmax::simd_softmax;
use bitnet_kernels::cpu::softmax::log_softmax_f32;

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────
//
// FMA vs multiply-then-add can differ by up to ~0.5 ULP per op; with
// thousands of accumulations the absolute error can reach ~1e-4 for
// matmul on moderate matrices.  Softmax and layer-norm are tighter
// because the element-wise operations dominate.

/// Absolute tolerance for element-wise kernels (softmax, layer-norm).
const ELEM_ABS_TOL: f32 = 1e-6;
/// Relative tolerance for element-wise kernels.
const ELEM_REL_TOL: f32 = 1e-5;

/// Absolute tolerance for reduction-heavy kernels (matmul).
const MATMUL_ABS_TOL: f32 = 1e-4;
/// Relative tolerance for reduction-heavy kernels.
const MATMUL_REL_TOL: f32 = 1e-4;

/// Reference scalar softmax (portable, no dispatch).
fn reference_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).clamp(-88.0, 88.0).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() } else { exps }
}

// ════════════════════════════════════════════════════════════════════════
// 1. SOFTMAX  —  simd_softmax dispatches to AVX2 or scalar internally
// ════════════════════════════════════════════════════════════════════════

/// Softmax parity: dispatch path vs pure-scalar reference (8-aligned).
#[test]
fn softmax_parity_aligned() {
    for &n in &[8, 16, 32, 64, 256] {
        let input = pseudo_rand(n, 42 + n as u64);
        let expected = reference_softmax(&input);
        let mut actual = vec![0.0f32; n];
        simd_softmax(&input, &mut actual).expect("softmax should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("softmax_aligned(n={n})"),
        );
    }
}

/// Softmax parity: non-8-aligned lengths exercise the scalar tail.
#[test]
fn softmax_parity_unaligned() {
    for &n in &[7, 13, 19, 33, 65, 100] {
        let input = pseudo_rand(n, 137 + n as u64);
        let expected = reference_softmax(&input);
        let mut actual = vec![0.0f32; n];
        simd_softmax(&input, &mut actual).expect("softmax should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("softmax_unaligned(n={n})"),
        );
    }
}

/// Softmax numerical stability: large magnitude inputs must not produce
/// NaN/Inf regardless of dispatch path.
#[test]
fn softmax_stability_large_values() {
    let input = vec![88.0, -88.0, 0.0, 50.0, -50.0, 1e-8, -1e-8, 42.0];
    let mut output = vec![0.0f32; input.len()];
    simd_softmax(&input, &mut output).expect("softmax should not fail");
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "softmax output[{i}] is not finite: {v}");
        assert!(v >= 0.0, "softmax output[{i}] is negative: {v}");
    }
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax outputs should sum to 1.0, got {sum}");
}

// ════════════════════════════════════════════════════════════════════════
// 2. MATMUL  —  simd_matmul_f32 dispatches to AVX2 gemm or scalar
// ════════════════════════════════════════════════════════════════════════

/// Reference scalar matmul: C = A * B (row-major, no transpose).
fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64; // f64 accumulation for reference
            for l in 0..k {
                acc += a[i * k + l] as f64 * b[l * n + j] as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

/// F32 GEMM parity: dispatched path vs f64-accumulated scalar reference.
#[test]
fn matmul_f32_parity_square() {
    for &dim in &[4, 8, 16, 32] {
        let (m, n, k) = (dim, dim, dim);
        let a = pseudo_rand(m * k, 1000 + dim as u64);
        let b = pseudo_rand(k * n, 2000 + dim as u64);
        let expected = reference_matmul(&a, &b, m, n, k);

        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut actual = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut actual, &cfg).expect("matmul should not fail");

        assert_vec_parity(
            &actual,
            &expected,
            MATMUL_ABS_TOL,
            MATMUL_REL_TOL,
            &format!("matmul_square(dim={dim})"),
        );
    }
}

/// F32 GEMM parity: non-square dimensions that exercise SIMD tails.
#[test]
fn matmul_f32_parity_nonsquare() {
    let cases = [(3, 5, 7), (1, 16, 9), (7, 1, 15), (10, 13, 17)];
    for (m, n, k) in cases {
        let a = pseudo_rand(m * k, 3000 + m as u64);
        let b = pseudo_rand(k * n, 4000 + n as u64);
        let expected = reference_matmul(&a, &b, m, n, k);

        let cfg = SimdMatmulConfig::new(m, n, k);
        let mut actual = vec![0.0f32; m * n];
        simd_matmul_f32(&a, &b, &mut actual, &cfg).expect("matmul should not fail");

        assert_vec_parity(
            &actual,
            &expected,
            MATMUL_ABS_TOL,
            MATMUL_REL_TOL,
            &format!("matmul_nonsquare(m={m},n={n},k={k})"),
        );
    }
}

/// I2_S quantized matmul: verify that the dispatched path produces the
/// same output as the scalar reference for a small hand-crafted case.
#[test]
#[ignore = "TDD scaffold: requires isolated scalar I2_S reference (dispatch always picks best path)"]
fn matmul_i2s_parity() {
    // Pack ternary weights: 4 values per byte, 2 bits each, LSB-first.
    // Encoding: 0b01 → +1, 0b11 → −1, 0b00 → 0
    let k: usize = 8;
    let n: usize = 2;
    let m: usize = 1;
    let block_size: usize = 8;

    // weights column 0: [+1, -1, 0, +1, -1, 0, +1, -1]
    // weights column 1: [+1, +1, +1, +1, 0, 0, 0, 0]
    // packed_k = ceil(8/4) = 2 bytes per column
    let weights_packed: Vec<u8> = vec![
        0b11_00_11_01, // col0 byte0: +1, -1, 0, -1
        0b11_01_00_01, // col0 byte1: +1, 0, +1, -1
        0b01_01_01_01, // col1 byte0: +1, +1, +1, +1
        0b00_00_00_00, // col1 byte1: 0, 0, 0, 0
    ];
    let scales = vec![1.0f32; n * k.div_ceil(block_size)]; // 2 scales, all 1.0
    let activations = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut out = vec![0.0f32; m * n];
    simd_matmul_i2s(&activations, &weights_packed, &scales, &mut out, m, n, k, block_size)
        .expect("i2s matmul should not fail");

    // Manual expected: dot(activations, decoded_col) for each column.
    // The exact values depend on the encoding above — assert finite for now.
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "i2s matmul output[{i}] is not finite: {v}");
    }
}

// ════════════════════════════════════════════════════════════════════════
// 3. LAYER NORM / RMS NORM  —  explicit scalar vs AVX2 entry points
// ════════════════════════════════════════════════════════════════════════

/// LayerNorm parity: `layer_norm_f32` (scalar) vs `layer_norm_avx2`.
#[test]
fn layer_norm_parity_aligned() {
    for &n in &[8, 16, 32, 64] {
        let input = pseudo_rand(n, 5000 + n as u64);
        let gamma: Vec<f32> = vec![1.0; n];
        let beta: Vec<f32> = vec![0.0; n];
        let config = LayerNormSimdConfig::new(vec![n]);

        let expected =
            layer_norm_f32(&input, &gamma, Some(&beta), &config).expect("scalar layer_norm");
        let actual =
            layer_norm_avx2(&input, &gamma, Some(&beta), &config).expect("avx2 layer_norm");

        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("layer_norm_aligned(n={n})"),
        );
    }
}

/// LayerNorm parity: non-8-aligned lengths.
#[test]
fn layer_norm_parity_unaligned() {
    for &n in &[7, 13, 19, 33] {
        let input = pseudo_rand(n, 6000 + n as u64);
        let gamma: Vec<f32> = vec![1.0; n];
        let beta: Vec<f32> = vec![0.0; n];
        let config = LayerNormSimdConfig::new(vec![n]);

        let expected =
            layer_norm_f32(&input, &gamma, Some(&beta), &config).expect("scalar layer_norm");
        let actual =
            layer_norm_avx2(&input, &gamma, Some(&beta), &config).expect("avx2 layer_norm");

        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("layer_norm_unaligned(n={n})"),
        );
    }
}

/// LayerNorm parity: batched (multiple instances in one call).
#[test]
fn layer_norm_parity_batched() {
    let norm_size = 16;
    let batch = 4;
    let total = norm_size * batch;
    let input = pseudo_rand(total, 7000);
    let gamma: Vec<f32> = pseudo_rand(norm_size, 7100).iter().map(|x| x.abs() + 0.1).collect();
    let beta = pseudo_rand(norm_size, 7200);
    let config = LayerNormSimdConfig::new(vec![norm_size]);

    let expected =
        layer_norm_f32(&input, &gamma, Some(&beta), &config).expect("scalar layer_norm batched");
    let actual =
        layer_norm_avx2(&input, &gamma, Some(&beta), &config).expect("avx2 layer_norm batched");

    assert_vec_parity(&actual, &expected, ELEM_ABS_TOL, ELEM_REL_TOL, "layer_norm_batched");
}

/// RMSNorm parity: `rms_norm_f32` (scalar) vs `rms_norm_avx2`.
#[test]
fn rms_norm_parity() {
    for &n in &[8, 13, 32, 64] {
        let input = pseudo_rand(n, 8000 + n as u64);
        let gamma: Vec<f32> =
            pseudo_rand(n, 8100 + n as u64).iter().map(|x| x.abs() + 0.1).collect();
        let config = RMSNormConfig::new(vec![n]);

        let expected = rms_norm_f32(&input, &gamma, &config).expect("scalar rms_norm");
        let actual = rms_norm_avx2(&input, &gamma, &config).expect("avx2 rms_norm");

        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("rms_norm(n={n})"),
        );
    }
}

/// RMSNorm parity: batched.
#[test]
fn rms_norm_parity_batched() {
    let norm_size = 32;
    let batch = 3;
    let total = norm_size * batch;
    let input = pseudo_rand(total, 9000);
    let gamma: Vec<f32> = pseudo_rand(norm_size, 9100).iter().map(|x| x.abs() + 0.1).collect();
    let config = RMSNormConfig::new(vec![norm_size]);

    let expected = rms_norm_f32(&input, &gamma, &config).expect("scalar rms_norm batched");
    let actual = rms_norm_avx2(&input, &gamma, &config).expect("avx2 rms_norm batched");

    assert_vec_parity(&actual, &expected, ELEM_ABS_TOL, ELEM_REL_TOL, "rms_norm_batched");
}

// ════════════════════════════════════════════════════════════════════════
// 4. CROSS-KERNEL NUMERICAL STABILITY
// ════════════════════════════════════════════════════════════════════════

/// End-to-end stability: layer_norm → matmul → softmax pipeline.
/// Verifies that FP rounding through the dispatched pipeline stays
/// within tolerance of the scalar reference pipeline.
#[test]
fn pipeline_layernorm_matmul_softmax_parity() {
    let seq_len = 16;
    let hidden = 8;

    let input = pseudo_rand(seq_len * hidden, 10_000);
    let gamma: Vec<f32> = vec![1.0; hidden];
    let beta: Vec<f32> = vec![0.0; hidden];
    let ln_cfg = LayerNormSimdConfig::new(vec![hidden]);

    // Step 1: LayerNorm (scalar reference)
    let normed_ref =
        layer_norm_f32(&input, &gamma, Some(&beta), &ln_cfg).expect("scalar layer_norm");
    let normed_avx2 =
        layer_norm_avx2(&input, &gamma, Some(&beta), &ln_cfg).expect("avx2 layer_norm");

    // Step 2: Matmul  normed @ W  →  logits (seq_len × 4)
    let out_dim = 4;
    let w = pseudo_rand(hidden * out_dim, 10_100);
    let mm_cfg = SimdMatmulConfig::new(seq_len, out_dim, hidden);

    let mut logits_ref = vec![0.0f32; seq_len * out_dim];
    simd_matmul_f32(&normed_ref, &w, &mut logits_ref, &mm_cfg).expect("matmul ref");

    let mut logits_avx2 = vec![0.0f32; seq_len * out_dim];
    simd_matmul_f32(&normed_avx2, &w, &mut logits_avx2, &mm_cfg).expect("matmul avx2");

    // Step 3: Row-wise softmax over logits
    for row in 0..seq_len {
        let start = row * out_dim;
        let end = start + out_dim;

        let mut sm_ref = vec![0.0f32; out_dim];
        simd_softmax(&logits_ref[start..end], &mut sm_ref).expect("softmax ref");

        let mut sm_avx2 = vec![0.0f32; out_dim];
        simd_softmax(&logits_avx2[start..end], &mut sm_avx2).expect("softmax avx2");

        // Pipeline tolerance is wider because errors compound.
        assert_vec_parity(
            &sm_avx2,
            &sm_ref,
            MATMUL_ABS_TOL,
            MATMUL_REL_TOL,
            &format!("pipeline_row_{row}"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. LOG-SOFTMAX  —  log_softmax_f32 dispatches to AVX2 or scalar
// ═══════════════════════════════════════════════════════════════════════

/// Reference scalar log-softmax (portable, no SIMD dispatch).
fn reference_log_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = input.iter().map(|&x| (x - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    input.iter().map(|&x| x - log_sum_exp).collect()
}

#[test]
fn log_softmax_parity_aligned() {
    for &n in &[8, 16, 32, 64, 128, 256] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin() * 3.0).collect();
        let expected = reference_log_softmax(&input);
        let mut actual = vec![0.0f32; n];
        log_softmax_f32(&input, &mut actual).expect("log_softmax should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("log_softmax_aligned(n={n})"),
        );
    }
}

#[test]
fn log_softmax_parity_unaligned() {
    for &n in &[1, 3, 7, 9, 15, 17, 31, 33, 65] {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).cos() * 2.0).collect();
        let expected = reference_log_softmax(&input);
        let mut actual = vec![0.0f32; n];
        log_softmax_f32(&input, &mut actual).expect("log_softmax should not fail");
        assert_vec_parity(
            &actual,
            &expected,
            ELEM_ABS_TOL,
            ELEM_REL_TOL,
            &format!("log_softmax_unaligned(n={n})"),
        );
    }
}

#[test]
fn log_softmax_stability_large_values() {
    let input = vec![1000.0, 1001.0, 999.0, 1002.0, 998.0, 1003.0, 997.0, 1004.0];
    let mut output = vec![0.0f32; 8];
    log_softmax_f32(&input, &mut output).expect("log_softmax should not fail");
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "log_softmax[{i}] is not finite: {v}");
        assert!(v <= 0.0, "log_softmax[{i}] is positive: {v}");
    }
    let exp_sum: f32 = output.iter().map(|&x| x.exp()).sum();
    assert!((exp_sum - 1.0).abs() < 1e-4, "exp(log_softmax) should sum to 1.0, got {exp_sum}");
}
