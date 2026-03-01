//! Numerical correctness tests for BitNet CPU kernels.
//!
//! Validates mathematical properties: softmax sum-to-one, layernorm
//! zero-mean, attention symmetry, quantization roundtrip fidelity,
//! and activation function boundary behaviour.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{gelu, sigmoid, silu};
use bitnet_kernels::cpu::attention::scaled_dot_product_attention;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_asymmetric_u8, dequantize_symmetric_i8,
    quantize_asymmetric_u8, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};

// ── Helpers ────────────────────────────────────────────────────────

fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Inline softmax for testing expected outputs.
fn reference_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ── Softmax properties ─────────────────────────────────────────────

#[test]
fn softmax_outputs_sum_to_one() {
    // Use attention to exercise the internal softmax: for seq_len=1 the
    // output should simply be V (softmax over a single element → 1.0).
    let head_dim = 8;
    let v = pseudo_random(10, head_dim);
    let q = pseudo_random(11, head_dim);
    let k = pseudo_random(12, head_dim);

    let out = scaled_dot_product_attention(&q, &k, &v, 1, 1, head_dim, false).unwrap();
    // With seq_len=1, softmax of a single score is 1.0, so output == v
    for (o, expected) in out.iter().zip(&v) {
        assert!(
            (o - expected).abs() < 1e-5,
            "single-token attention should pass through V: {o} vs {expected}"
        );
    }
}

#[test]
fn softmax_reference_sums_to_one_various_sizes() {
    for size in [2, 4, 8, 16, 64, 256] {
        let input = pseudo_random(size as u64, size);
        let sm = reference_softmax(&input);
        let sum: f32 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum} for size={size}, expected ≈ 1.0");
    }
}

#[test]
fn softmax_all_equal_gives_uniform() {
    let input = vec![1.0f32; 8];
    let sm = reference_softmax(&input);
    for &v in &sm {
        assert!((v - 0.125).abs() < 1e-6, "uniform input should give 1/n: got {v}");
    }
}

// ── LayerNorm properties ───────────────────────────────────────────

#[test]
fn layernorm_output_has_zero_mean() {
    let dim = 64;
    let input = pseudo_random(20, dim);
    let gamma = vec![1.0f32; dim];
    let config =
        LayerNormConfig { normalized_shape: vec![dim], eps: 1e-5, elementwise_affine: false };
    let output = layer_norm(&input, &gamma, None, &config).unwrap();

    let mean: f32 = output.iter().sum::<f32>() / dim as f32;
    assert!(mean.abs() < 1e-5, "layernorm output mean={mean}, expected ≈ 0.0");
}

#[test]
fn layernorm_output_has_unit_variance() {
    let dim = 128;
    let input = pseudo_random(30, dim);
    let gamma = vec![1.0f32; dim];
    let config =
        LayerNormConfig { normalized_shape: vec![dim], eps: 1e-5, elementwise_affine: false };
    let output = layer_norm(&input, &gamma, None, &config).unwrap();

    let mean: f32 = output.iter().sum::<f32>() / dim as f32;
    let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
    assert!((var - 1.0).abs() < 0.05, "layernorm output variance={var}, expected ≈ 1.0");
}

#[test]
fn rms_norm_output_has_unit_rms() {
    let dim = 64;
    let input = pseudo_random(40, dim);
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);
    let output = rms_norm(&input, &gamma, &config).unwrap();

    let rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
    assert!((rms - 1.0).abs() < 0.05, "rms_norm output RMS={rms}, expected ≈ 1.0");
}

// ── Attention symmetry ─────────────────────────────────────────────

#[test]
fn attention_symmetric_input_symmetric_output() {
    // Non-causal self-attention on identical Q=K=V rows should produce
    // identical output rows (uniform softmax → avg = same row).
    let seq_len = 4;
    let head_dim = 8;
    let row = pseudo_random(50, head_dim);
    let data: Vec<f32> = (0..seq_len).flat_map(|_| row.iter().copied()).collect();

    let output =
        scaled_dot_product_attention(&data, &data, &data, seq_len, seq_len, head_dim, false)
            .unwrap();

    for i in 1..seq_len {
        for d in 0..head_dim {
            let first = output[d];
            let current = output[i * head_dim + d];
            assert!((first - current).abs() < 1e-5, "row0[{d}]={first} vs row{i}[{d}]={current}");
        }
    }
}

// ── Quantization roundtrip ─────────────────────────────────────────

#[test]
fn quantize_i8_roundtrip_preserves_signal() {
    let input = pseudo_random(60, 256);
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let reconstructed = dequantize_symmetric_i8(&quantized, scale);
    let err = compute_quantization_error(&input, &reconstructed);

    assert!(err.mse < 0.01, "i8 roundtrip MSE={} too high (expected < 0.01)", err.mse);
    assert!(err.snr > 10.0, "i8 roundtrip SNR={} too low (expected > 10 dB)", err.snr);
}

#[test]
fn quantize_u8_asymmetric_roundtrip() {
    let input = pseudo_random(70, 128);
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let reconstructed = dequantize_asymmetric_u8(&quantized, scale, zero_point);
    let err = compute_quantization_error(&input, &reconstructed);

    assert!(err.mse < 0.01, "u8 asymmetric roundtrip MSE={} too high", err.mse);
}

#[test]
fn quantize_ternary_preserves_sign() {
    let input = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let threshold = 0.3;
    let ternary = quantize_ternary(&input, threshold);

    assert_eq!(ternary, vec![-1, -1, 0, 1, 1]);
}

#[test]
fn quantize_constant_input_produces_zero_scale() {
    let input = vec![2.71f32; 64];
    let (quantized, scale, _zp) = quantize_asymmetric_u8(&input);
    assert_eq!(scale, 0.0);
    assert!(quantized.iter().all(|&q| q == 0));
}

// ── Matmul identity ────────────────────────────────────────────────

#[test]
fn matmul_identity_matrix() {
    let n = 4;
    let mut identity = vec![0.0f32; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let input = pseudo_random(80, n * n);
    let mut output = vec![0.0f32; n * n];
    let cfg = SimdMatmulConfig::new(n, n, n);
    simd_matmul_f32(&input, &identity, &mut output, &cfg).unwrap();

    for (o, i) in output.iter().zip(&input) {
        assert!((o - i).abs() < 1e-5, "A * I should equal A: {o} vs {i}");
    }
}

// ── Activation boundary correctness ────────────────────────────────

#[test]
fn sigmoid_boundary_values() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");
    assert!(sigmoid(100.0) > 0.999, "sigmoid(large) should be ≈ 1");
    assert!(sigmoid(-100.0) < 0.001, "sigmoid(−large) should be ≈ 0");
}

#[test]
fn silu_at_zero() {
    assert!((silu(0.0) - 0.0).abs() < 1e-6, "silu(0) should be 0");
}

#[test]
fn gelu_at_zero() {
    assert!((gelu(0.0) - 0.0).abs() < 1e-6, "gelu(0) should be 0");
}

// ── RoPE numerical correctness ─────────────────────────────────────

#[test]
fn rope_position_zero_is_identity() {
    let head_dim = 8;
    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let original = pseudo_random(90, head_dim);
    let mut data = original.clone();
    apply_rope(&mut data, 0, head_dim, &freqs);

    // At position 0, angle = 0 → cos=1, sin=0 → rotation is identity
    for (o, &e) in data.iter().zip(&original) {
        assert!((o - e).abs() < 1e-5, "RoPE at position 0 should be identity: {o} vs {e}");
    }
}
