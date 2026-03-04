#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! CPU determinism regression tests — verifies identical outputs for identical inputs.
//!
//! Core invariant: same weights + same input + same seed = same output, bit-for-bit.
//! Each kernel is run 10 times with identical inputs and the outputs are compared
//! byte-by-byte to detect any non-determinism.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::layer_norm_simd::{LayerNormSimdConfig, layer_norm_f32};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cpu::softmax::softmax_f32;

const RUNS: usize = 10;

/// Assert two `f32` slices are bit-for-bit identical via their byte representations.
fn assert_bitwise_identical(a: &[f32], b: &[f32], label: &str, run: usize) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch on run {run}");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: bit mismatch on run {run}: {:#010x} vs {:#010x}",
            x.to_bits(),
            y.to_bits(),
        );
    }
}

// ── Hardcoded test vectors ────────────────────────────────────────────────

/// 8-element vector with mixed positive, negative, zero, and near-zero values.
const INPUT_8: [f32; 8] = [0.5, -1.2, 0.0, 3.14, -0.001, 2.71, 1.0, -0.5];

/// 4×4 matrix A (row-major) for matmul tests.
const MAT_A_4X4: [f32; 16] = [
    1.0, 0.5, -0.3, 0.7, //
    -1.2, 2.0, 0.1, -0.4, //
    0.3, -0.8, 1.5, 0.2, //
    0.0, 0.6, -1.0, 0.9, //
];

/// 4×4 matrix B (row-major) for matmul tests.
const MAT_B_4X4: [f32; 16] = [
    0.2, -0.5, 1.1, 0.3, //
    0.8, 0.0, -0.7, 1.4, //
    -0.1, 0.9, 0.4, -0.6, //
    1.0, -0.3, 0.2, 0.5, //
];

/// Gamma weights for normalization (8 elements).
const GAMMA_8: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Beta bias for normalization (8 elements).
const BETA_8: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Gamma weights for 4-element normalization in the chain test.
const GAMMA_4: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

// ── Individual kernel determinism tests ───────────────────────────────────

#[test]
fn test_softmax_deterministic_across_runs() {
    let mut reference = vec![0.0f32; INPUT_8.len()];
    softmax_f32(&INPUT_8, &mut reference).expect("softmax reference run");

    for run in 1..RUNS {
        let mut output = vec![0.0f32; INPUT_8.len()];
        softmax_f32(&INPUT_8, &mut output).expect("softmax run");
        assert_bitwise_identical(&reference, &output, "softmax_f32", run);
    }
}

#[test]
fn test_layer_norm_deterministic_across_runs() {
    let config = LayerNormConfig::new(vec![INPUT_8.len()]);
    let reference =
        layer_norm(&INPUT_8, &GAMMA_8, Some(&BETA_8), &config).expect("layer_norm reference run");

    for run in 1..RUNS {
        let output =
            layer_norm(&INPUT_8, &GAMMA_8, Some(&BETA_8), &config).expect("layer_norm run");
        assert_bitwise_identical(&reference, &output, "layer_norm", run);
    }
}

#[test]
fn test_layer_norm_simd_deterministic_across_runs() {
    let config = LayerNormSimdConfig::new(vec![INPUT_8.len()]);
    let reference = layer_norm_f32(&INPUT_8, &GAMMA_8, Some(&BETA_8), &config)
        .expect("layer_norm_simd reference run");

    for run in 1..RUNS {
        let output = layer_norm_f32(&INPUT_8, &GAMMA_8, Some(&BETA_8), &config)
            .expect("layer_norm_simd run");
        assert_bitwise_identical(&reference, &output, "layer_norm_f32 (simd)", run);
    }
}

#[test]
fn test_rms_norm_deterministic_across_runs() {
    let config = LayerNormConfig::new(vec![INPUT_8.len()]);
    let reference = rms_norm(&INPUT_8, &GAMMA_8, &config).expect("rms_norm reference run");

    for run in 1..RUNS {
        let output = rms_norm(&INPUT_8, &GAMMA_8, &config).expect("rms_norm run");
        assert_bitwise_identical(&reference, &output, "rms_norm", run);
    }
}

#[test]
fn test_matmul_deterministic_across_runs() {
    let cfg = SimdMatmulConfig::new(4, 4, 4);
    let mut reference = vec![0.0f32; 16];
    simd_matmul_f32(&MAT_A_4X4, &MAT_B_4X4, &mut reference, &cfg).expect("matmul reference run");

    for run in 1..RUNS {
        let mut output = vec![0.0f32; 16];
        simd_matmul_f32(&MAT_A_4X4, &MAT_B_4X4, &mut output, &cfg).expect("matmul run");
        assert_bitwise_identical(&reference, &output, "simd_matmul_f32", run);
    }
}

#[test]
fn test_chain_rms_norm_matmul_softmax_deterministic_across_runs() {
    // Chain: rms_norm → matmul → softmax on a 4×4 workload.
    let norm_config = LayerNormConfig::new(vec![4]);

    let run_chain = || -> Vec<f32> {
        // Step 1: RMS-norm each row of MAT_A_4X4 (4 batches of 4 elements).
        let normed = rms_norm(&MAT_A_4X4, &GAMMA_4, &norm_config).expect("chain: rms_norm failed");

        // Step 2: matmul normed(4×4) × B(4×4) → C(4×4).
        let cfg = SimdMatmulConfig::new(4, 4, 4);
        let mut matmul_out = vec![0.0f32; 16];
        simd_matmul_f32(&normed, &MAT_B_4X4, &mut matmul_out, &cfg).expect("chain: matmul failed");

        // Step 3: softmax each row independently.
        let mut final_out = vec![0.0f32; 16];
        for row in 0..4 {
            let start = row * 4;
            let end = start + 4;
            softmax_f32(&matmul_out[start..end], &mut final_out[start..end])
                .expect("chain: softmax failed");
        }
        final_out
    };

    let reference = run_chain();
    for run in 1..RUNS {
        let output = run_chain();
        assert_bitwise_identical(&reference, &output, "chain(rms_norm→matmul→softmax)", run);
    }
}
