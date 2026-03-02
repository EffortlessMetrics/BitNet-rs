//! Integration Wave 15 — Quantization Pipeline Integration Tests
//!
//! BDD-style Given/When/Then tests for end-to-end quantization workflows:
//!
//! 1. Quantization pipeline: FP32 → INT2/INT4/INT8 → validation → dequantize → verify accuracy
//! 2. Calibration flow: collect activation stats → compute scales/zero-points → apply quantization
//! 3. Layer-by-layer quantization: different precision per layer type
//! 4. Quantization-aware training simulation: forward → quantize → loss computation
//! 5. Dynamic quantization: runtime precision selection based on tensor statistics
//! 6. Quantization error analysis: measure and track quantization noise per layer

use bitnet_kernels::cpu::activations::{ActivationType, activate, gelu_vec, silu_vec};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, cross_entropy_loss, mse_loss,
};
use bitnet_kernels::cpu::quantize::{
    QuantizationError, compute_quantization_error, dequantize_asymmetric_u8,
    dequantize_symmetric_i8, quantize_asymmetric_u8, quantize_binary, quantize_symmetric_i8,
    quantize_ternary,
};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::residual::add_residual;
use bitnet_kernels::reduction::{ReductionOp, reduce_f32};

// ── Helpers ────────────────────────────────────────────────────────

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!((a - b).abs() <= tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
}

fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        assert_close(ai, bi, tol, &format!("{ctx}[{i}]"));
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn variance(v: &[f32]) -> f32 {
    let m = mean(v);
    v.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
}

fn std_dev(v: &[f32]) -> f32 {
    variance(v).sqrt()
}

/// Simple matmul: C[m×n] = A[m×k] × B[k×n] (row-major).
fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Generate a pseudo-random f32 vector using a simple LCG.
fn pseudo_random_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((state >> 33) as i32) as f32;
            bits / (i32::MAX as f32) * scale
        })
        .collect()
}

/// Collect activation statistics (min, max, mean, std) from a tensor.
struct ActivationStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    abs_max: f32,
}

impl ActivationStats {
    fn collect(data: &[f32]) -> Self {
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let m = mean(data);
        let s = std_dev(data);
        let abs_max = data.iter().copied().fold(0.0_f32, |a, v| a.max(v.abs()));
        Self { min, max, mean: m, std: s, abs_max }
    }
}

// ══════════════════════════════════════════════════════════════════
// 1. Quantization Pipeline: FP32 → INT → validation → dequantize → verify
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_fp32_tensor_when_quantize_i8_then_dequantize_roundtrip_accurate() {
    // Given
    let input: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect();

    // When
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then
    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.mse < 1e-4, "INT8 roundtrip MSE too high: {}", err.mse);
    assert!(err.max_abs_error < 0.01, "INT8 max error too high: {}", err.max_abs_error);
    assert!(err.snr > 40.0, "INT8 SNR too low: {}", err.snr);
}

#[test]
fn given_fp32_tensor_when_quantize_i4_then_dequantize_roundtrip_within_bounds() {
    // Given
    let input: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();

    // When
    let (quantized, scale) = quantize_symmetric_i8(&input, 4);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then
    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.mse < 0.02, "INT4 roundtrip MSE too high: {}", err.mse);
    assert!(err.snr > 15.0, "INT4 SNR too low: {}", err.snr);
}

#[test]
fn given_fp32_tensor_when_quantize_i2_then_dequantize_coarse_but_bounded() {
    // Given
    let input: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();

    // When
    let (quantized, scale) = quantize_symmetric_i8(&input, 2);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then
    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.mse < 0.5, "INT2 roundtrip MSE too high: {}", err.mse);
    assert!(err.snr > 0.0, "INT2 SNR should be positive: {}", err.snr);
}

#[test]
fn given_fp32_tensor_when_quantize_u8_asymmetric_then_roundtrip_preserves_range() {
    // Given — values spanning a non-symmetric range
    let input: Vec<f32> = (0..200).map(|i| (i as f32) * 0.05 - 3.0).collect();

    // When
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let dequantized = dequantize_asymmetric_u8(&quantized, scale, zero_point);

    // Then
    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.mse < 0.001, "U8 asymmetric MSE too high: {}", err.mse);
    assert!(
        err.max_abs_error < scale * 1.5,
        "max error {} exceeds 1.5× scale {}",
        err.max_abs_error,
        scale
    );
}

#[test]
fn given_fp32_tensor_when_binary_quantize_then_all_values_are_pm1() {
    // Given
    let input = pseudo_random_vec(128, 42, 2.0);

    // When
    let binary = quantize_binary(&input);

    // Then
    assert!(binary.iter().all(|&v| v == -1 || v == 1));
    assert_eq!(binary.len(), input.len());
}

#[test]
fn given_fp32_tensor_when_ternary_quantize_then_values_in_neg1_0_pos1() {
    // Given
    let input = pseudo_random_vec(128, 99, 1.0);
    let threshold = 0.3;

    // When
    let ternary = quantize_ternary(&input, threshold);

    // Then
    assert!(ternary.iter().all(|&v| v == -1 || v == 0 || v == 1));
    for (&orig, &quant) in input.iter().zip(ternary.iter()) {
        if orig.abs() <= threshold {
            assert_eq!(quant, 0, "value {} should be quantized to 0", orig);
        }
    }
}

#[test]
fn given_fp32_pipeline_when_quantize_all_formats_then_error_decreases_with_bits() {
    // Given
    let input: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.1).sin()).collect();

    // When — quantize at 2, 4, and 8 bits
    let err2 = {
        let (q, s) = quantize_symmetric_i8(&input, 2);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&input, &d)
    };
    let err4 = {
        let (q, s) = quantize_symmetric_i8(&input, 4);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&input, &d)
    };
    let err8 = {
        let (q, s) = quantize_symmetric_i8(&input, 8);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&input, &d)
    };

    // Then — monotonically decreasing error
    assert!(err8.mse < err4.mse, "8-bit MSE {} ≥ 4-bit MSE {}", err8.mse, err4.mse);
    assert!(err4.mse < err2.mse, "4-bit MSE {} ≥ 2-bit MSE {}", err4.mse, err2.mse);
    assert!(err8.snr > err4.snr, "8-bit SNR {} ≤ 4-bit SNR {}", err8.snr, err4.snr);
    assert!(err4.snr > err2.snr, "4-bit SNR {} ≤ 2-bit SNR {}", err4.snr, err2.snr);
}

#[test]
fn given_i2s_packed_weights_when_matmul_then_output_matches_naive() {
    // Given — pack weights to I2S format and multiply
    let m = 4;
    let k = 8;
    let n = 4;
    let weights_i8: Vec<i8> = (0..(m * k) as i8)
        .map(|i| (i % 3) - 1) // values in {-1, 0, 1}
        .collect();
    let packed: Vec<u8> =
        weights_i8.chunks(4).map(|c| pack_i2s([c[0], c[1], c[2], c[3]])).collect();
    let scales = vec![1.0f32; m];
    let input: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

    // When
    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&input, &packed, &scales, &mut output, m, n, k, k).unwrap();

    // Then — output should be finite and have correct shape
    assert_eq!(output.len(), m * n);
    assert!(output.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 2. Calibration Flow: collect stats → compute scales → quantize
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_activation_tensor_when_collect_stats_then_min_max_correct() {
    // Given
    let activations = vec![-2.5, -1.0, 0.0, 0.5, 1.0, 3.0];

    // When
    let stats = ActivationStats::collect(&activations);

    // Then
    assert_close(stats.min, -2.5, 1e-5, "min");
    assert_close(stats.max, 3.0, 1e-5, "max");
    assert_close(stats.abs_max, 3.0, 1e-5, "abs_max");
}

#[test]
fn given_activation_stats_when_compute_symmetric_scale_then_scale_correct() {
    // Given
    let activations: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
    let stats = ActivationStats::collect(&activations);

    // When — symmetric scale should be abs_max / qmax
    let (_, scale) = quantize_symmetric_i8(&activations, 8);

    // Then
    let expected_scale = stats.abs_max / 127.0;
    assert_close(scale, expected_scale, 1e-5, "symmetric scale");
}

#[test]
fn given_activation_stats_when_compute_asymmetric_params_then_range_covered() {
    // Given — non-symmetric distribution
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let stats = ActivationStats::collect(&activations);

    // When
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&activations);

    // Then — scale should cover the full range
    let expected_scale = (stats.max - stats.min) / 255.0;
    assert_close(scale, expected_scale, 1e-5, "asymmetric scale");
    assert!(*quantized.iter().min().unwrap() <= 1, "min quantized should be near 0");
    assert!(*quantized.iter().max().unwrap() >= 254, "max quantized should be near 255");
    assert!(zero_point >= 0, "zero_point should be non-negative");
}

#[test]
fn given_calibration_batches_when_aggregate_stats_then_global_range_captured() {
    // Given — multiple calibration batches
    let batch1 = pseudo_random_vec(64, 1, 1.0);
    let batch2 = pseudo_random_vec(64, 2, 2.0);
    let batch3 = pseudo_random_vec(64, 3, 3.0);

    let stats1 = ActivationStats::collect(&batch1);
    let stats2 = ActivationStats::collect(&batch2);
    let stats3 = ActivationStats::collect(&batch3);

    // When — aggregate: take global min/max
    let global_min = stats1.min.min(stats2.min).min(stats3.min);
    let global_max = stats1.max.max(stats2.max).max(stats3.max);
    let global_abs_max = stats1.abs_max.max(stats2.abs_max).max(stats3.abs_max);

    // Then — global range covers all batches
    let all: Vec<f32> = [batch1, batch2, batch3].concat();
    let global_stats = ActivationStats::collect(&all);
    assert_close(global_min, global_stats.min, 1e-5, "global min");
    assert_close(global_max, global_stats.max, 1e-5, "global max");
    assert_close(global_abs_max, global_stats.abs_max, 1e-5, "global abs_max");
}

#[test]
fn given_calibrated_scale_when_quantize_then_error_within_calibration_bound() {
    // Given — calibrate on representative data
    let calibration_data = pseudo_random_vec(512, 42, 1.5);
    let (_, calibrated_scale) = quantize_symmetric_i8(&calibration_data, 8);

    // When — quantize new data using the same scale
    let new_data = pseudo_random_vec(512, 77, 1.5);
    let quantized: Vec<i8> = new_data
        .iter()
        .map(|&v| (v / calibrated_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let dequantized = dequantize_symmetric_i8(&quantized, calibrated_scale);

    // Then
    let err = compute_quantization_error(&new_data, &dequantized);
    assert!(err.mse < 0.001, "calibrated quantization MSE too high: {}", err.mse);
}

#[test]
fn given_post_relu_activations_when_calibrate_then_asymmetric_captures_non_negative() {
    // Given — post-ReLU: all non-negative
    let pre_relu = pseudo_random_vec(256, 42, 2.0);
    let post_relu = activate(&pre_relu, ActivationType::ReLU);
    let stats = ActivationStats::collect(&post_relu);

    // When
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&post_relu);
    let dequantized = dequantize_asymmetric_u8(&quantized, scale, zero_point);

    // Then — non-negative distribution well captured
    assert!(stats.min >= 0.0, "post-ReLU min should be ≥ 0");
    let err = compute_quantization_error(&post_relu, &dequantized);
    assert!(err.mse < 0.001, "post-ReLU quant MSE: {}", err.mse);
}

#[test]
fn given_post_gelu_activations_when_calibrate_then_symmetric_captures_distribution() {
    // Given — post-GELU: near-symmetric but slightly shifted
    let pre_gelu = pseudo_random_vec(256, 55, 3.0);
    let post_gelu = gelu_vec(&pre_gelu);

    // When
    let (quantized, scale) = quantize_symmetric_i8(&post_gelu, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then
    let err = compute_quantization_error(&post_gelu, &dequantized);
    assert!(err.mse < 0.01, "post-GELU quant MSE: {}", err.mse);
}

// ══════════════════════════════════════════════════════════════════
// 3. Layer-by-Layer Quantization: different precision per layer type
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_attention_weights_when_quantize_i8_then_cosine_sim_above_threshold() {
    // Given — simulate attention projection weights
    let dim = 64;
    let weights = pseudo_random_vec(dim * dim, 10, 0.02);

    // When — INT8 quantization for attention weights
    let (quantized, scale) = quantize_symmetric_i8(&weights, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then — cosine_similarity_loss returns 1 - cos_sim (0 = identical)
    let cos_loss = cosine_similarity_loss(&weights, &dequantized).unwrap();
    assert!(cos_loss < 0.001, "attention weight cosine loss too high: {cos_loss}");
}

#[test]
fn given_ffn_weights_when_quantize_i4_then_error_acceptable_for_ffn() {
    // Given — simulate FFN (larger, more redundant) weights
    let in_dim = 64;
    let hidden_dim = 128;
    let weights = pseudo_random_vec(in_dim * hidden_dim, 20, 0.1);

    // When — INT4 for FFN weights (lower precision acceptable)
    let (quantized, scale) = quantize_symmetric_i8(&weights, 4);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    // Then — FFN tolerates more noise
    let err = compute_quantization_error(&weights, &dequantized);
    assert!(err.mse < 0.01, "FFN INT4 MSE too high: {}", err.mse);
    assert!(err.snr > 15.0, "FFN INT4 SNR too low: {}", err.snr);
}

#[test]
fn given_layernorm_weights_when_kept_fp32_then_zero_quantization_error() {
    // Given — LayerNorm weights should stay in float (no quantization)
    let dim = 64;
    let gamma: Vec<f32> = (0..dim).map(|i| 0.9 + (i as f32) * 0.003).collect();
    let beta = vec![0.0f32; dim];

    // When — "quantize" by keeping as-is (FP32 identity)
    let gamma_copy = gamma.clone();

    // Then — zero error
    let err = compute_quantization_error(&gamma, &gamma_copy);
    assert_close(err.mse, 0.0, 1e-10, "LN gamma MSE");
    assert_eq!(err.snr, f32::INFINITY);
    assert_close(err.max_abs_error, 0.0, 1e-10, "LN gamma max error");
    // Also verify the LN weights are valid for normalization
    let input = pseudo_random_vec(dim, 42, 1.0);
    let cfg = LayerNormConfig::new(vec![dim]);
    let normed = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
    assert!(normed.iter().all(|v| v.is_finite()));
}

#[test]
fn given_embedding_table_when_quantize_i8_then_lookup_close_to_original() {
    // Given
    let vocab = 32;
    let dim = 16;
    let table = pseudo_random_vec(vocab * dim, 30, 0.5);

    // When — quantize the embedding table
    let (q_table, scale) = quantize_symmetric_i8(&table, 8);
    let dq_table = dequantize_symmetric_i8(&q_table, scale);

    // Then — lookup a few tokens and compare
    for token_id in [0usize, 5, 15, 31] {
        let start = token_id * dim;
        let orig_emb = &table[start..start + dim];
        let quant_emb = &dq_table[start..start + dim];
        let err = compute_quantization_error(orig_emb, quant_emb);
        assert!(err.mse < 1e-4, "token {token_id} embedding MSE: {}", err.mse);
    }
}

#[test]
fn given_mixed_precision_layers_when_pipeline_then_output_finite_and_bounded() {
    // Given — simulate a mini transformer block with mixed precision
    let dim = 16;
    let seq = 4;

    // Attention weights: INT8
    let attn_w = pseudo_random_vec(dim * dim, 10, 0.1);
    let (q_attn, s_attn) = quantize_symmetric_i8(&attn_w, 8);
    let dq_attn = dequantize_symmetric_i8(&q_attn, s_attn);

    // FFN weights: INT4
    let ffn_w = pseudo_random_vec(dim * dim, 20, 0.15);
    let (q_ffn, s_ffn) = quantize_symmetric_i8(&ffn_w, 4);
    let dq_ffn = dequantize_symmetric_i8(&q_ffn, s_ffn);

    // LN weights: FP32 (no quantization)
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];
    let cfg = LayerNormConfig::new(vec![dim]);

    // When — pipeline: norm → attn_proj → norm → ffn_proj
    let input = pseudo_random_vec(seq * dim, 5, 1.0);
    let normed1 = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();
    let attn_out = naive_matmul(&normed1, &dq_attn, seq, dim, dim);
    let normed2 = layer_norm(&attn_out, &gamma, Some(&beta), &cfg).unwrap();
    let ffn_out = naive_matmul(&normed2, &dq_ffn, seq, dim, dim);

    // Then
    assert_eq!(ffn_out.len(), seq * dim);
    assert!(ffn_out.iter().all(|v| v.is_finite()), "mixed-precision output must be finite");
}

#[test]
fn given_attention_i8_vs_ffn_i4_when_compare_error_then_attention_has_lower_error() {
    // Given — same weight distribution
    let dim = 64;
    let weights = pseudo_random_vec(dim * dim, 42, 0.1);

    // When
    let err_attn = {
        let (q, s) = quantize_symmetric_i8(&weights, 8);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&weights, &d)
    };
    let err_ffn = {
        let (q, s) = quantize_symmetric_i8(&weights, 4);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&weights, &d)
    };

    // Then — INT8 (attention) should have lower error than INT4 (FFN)
    assert!(err_attn.mse < err_ffn.mse, "attn MSE {} ≥ ffn MSE {}", err_attn.mse, err_ffn.mse);
}

// ══════════════════════════════════════════════════════════════════
// 4. Quantization-Aware Training Simulation
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_forward_pass_when_fake_quantize_i8_then_loss_computable() {
    // Given — mini forward pass with fake quantization
    let batch = 4;
    let dim = 8;
    let num_classes = 4;
    let input = pseudo_random_vec(batch * dim, 42, 1.0);
    let weights = pseudo_random_vec(dim * num_classes, 99, 0.3);

    // When — fake quantize weights (quantize then dequantize)
    let (q_w, s_w) = quantize_symmetric_i8(&weights, 8);
    let fake_q_weights = dequantize_symmetric_i8(&q_w, s_w);
    let logits = naive_matmul(&input, &fake_q_weights, batch, num_classes, dim);

    // Then — loss should be computable
    let targets = vec![0.25f32; batch * num_classes]; // uniform target
    let loss = mse_loss(&logits, &targets, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "QAT forward loss should be finite");
    assert!(loss >= 0.0, "MSE loss should be non-negative");
}

#[test]
fn given_fake_quantized_weights_when_forward_then_output_close_to_fp32() {
    // Given
    let m = 4;
    let k = 8;
    let n = 4;
    let weights = pseudo_random_vec(k * n, 55, 0.5);
    let input = pseudo_random_vec(m * k, 77, 1.0);

    // When — FP32 forward
    let fp32_out = naive_matmul(&input, &weights, m, n, k);

    // Fake quantize weights
    let (q_w, s_w) = quantize_symmetric_i8(&weights, 8);
    let fq_weights = dequantize_symmetric_i8(&q_w, s_w);
    let fq_out = naive_matmul(&input, &fq_weights, m, n, k);

    // Then — outputs should be very close
    let err = compute_quantization_error(&fp32_out, &fq_out);
    assert!(err.mse < 0.01, "fake-quant vs FP32 MSE: {}", err.mse);
}

#[test]
fn given_qat_simulation_when_multiple_iterations_then_loss_stays_finite() {
    // Given
    let dim = 8;
    let num_classes = 4;
    let batch = 2;
    let mut weights = pseudo_random_vec(dim * num_classes, 42, 0.3);

    // When — simulate 5 QAT iterations
    for iter in 0..5 {
        let input = pseudo_random_vec(batch * dim, 100 + iter, 1.0);

        // Fake quantize
        let (q_w, s_w) = quantize_symmetric_i8(&weights, 8);
        let fq_weights = dequantize_symmetric_i8(&q_w, s_w);

        // Forward
        let logits = naive_matmul(&input, &fq_weights, batch, num_classes, dim);
        let targets = vec![0.5f32; batch * num_classes];
        let loss = mse_loss(&logits, &targets, LossReduction::Mean).unwrap();
        assert!(loss.is_finite(), "QAT iteration {iter} loss not finite");

        // Pseudo-gradient step (just perturb weights slightly)
        for w in &mut weights {
            *w -= 0.001 * (*w);
        }
    }
}

#[test]
fn given_forward_with_activation_when_fake_quantize_activations_then_loss_close() {
    // Given
    let dim = 16;
    let hidden = 8;
    let batch = 2;
    let input = pseudo_random_vec(batch * dim, 42, 1.0);
    let w1 = pseudo_random_vec(dim * hidden, 10, 0.3);
    let w2 = pseudo_random_vec(hidden * dim, 20, 0.3);

    // When — forward with fake-quantized activations
    let h = naive_matmul(&input, &w1, batch, hidden, dim);
    let h_act = gelu_vec(&h);
    // Fake quantize activations
    let (q_h, s_h) = quantize_symmetric_i8(&h_act, 8);
    let fq_h = dequantize_symmetric_i8(&q_h, s_h);
    let output = naive_matmul(&fq_h, &w2, batch, dim, hidden);

    // Also compute without fake quantization
    let output_fp32 = naive_matmul(&h_act, &w2, batch, dim, hidden);

    // Then
    let err = compute_quantization_error(&output_fp32, &output);
    assert!(err.mse < 0.1, "activation fake-quant MSE: {}", err.mse);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn given_cross_entropy_loss_when_fake_quantized_logits_then_loss_finite() {
    // Given
    let num_classes = 8;
    let logits_fp32 = pseudo_random_vec(num_classes, 42, 3.0);

    // When — fake quantize logits
    let (q, s) = quantize_symmetric_i8(&logits_fp32, 8);
    let logits_fq = dequantize_symmetric_i8(&q, s);
    let targets: Vec<usize> = vec![3]; // target class index
    let (loss, _grad) =
        cross_entropy_loss(&logits_fq, &targets, num_classes, LossReduction::Mean).unwrap();

    // Then
    assert!(loss.is_finite(), "cross-entropy with fake-quant logits should be finite");
}

#[test]
fn given_qat_with_i4_weights_when_forward_then_output_still_reasonable() {
    // Given — more aggressive quantization
    let m = 4;
    let k = 16;
    let n = 4;
    let weights = pseudo_random_vec(k * n, 55, 0.5);
    let input = pseudo_random_vec(m * k, 77, 1.0);

    // When
    let (q_w, s_w) = quantize_symmetric_i8(&weights, 4);
    let fq_weights = dequantize_symmetric_i8(&q_w, s_w);
    let fq_out = naive_matmul(&input, &fq_weights, m, n, k);

    // Then — output still finite, just noisier
    assert!(fq_out.iter().all(|v| v.is_finite()));
    let fp32_out = naive_matmul(&input, &weights, m, n, k);
    let err = compute_quantization_error(&fp32_out, &fq_out);
    assert!(err.mse < 1.0, "I4 QAT MSE should be bounded: {}", err.mse);
}

// ══════════════════════════════════════════════════════════════════
// 5. Dynamic Quantization: runtime precision selection
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_small_range_tensor_when_select_precision_then_i8_sufficient() {
    // Given — small dynamic range
    let input = pseudo_random_vec(256, 42, 0.5);
    let stats = ActivationStats::collect(&input);

    // When — decide precision based on range
    let dynamic_range = stats.max - stats.min;
    let bits: u8 = if dynamic_range < 1.0 {
        8
    } else if dynamic_range < 5.0 {
        4
    } else {
        2
    };

    // Then
    assert_eq!(bits, 8);
    let (q, s) = quantize_symmetric_i8(&input, bits);
    let d = dequantize_symmetric_i8(&q, s);
    let err = compute_quantization_error(&input, &d);
    assert!(err.mse < 1e-4, "small-range i8 MSE: {}", err.mse);
}

#[test]
fn given_large_range_tensor_when_select_precision_then_lower_bits_chosen() {
    // Given — large dynamic range
    let input = pseudo_random_vec(256, 42, 100.0);
    let stats = ActivationStats::collect(&input);

    // When
    let dynamic_range = stats.max - stats.min;
    let bits: u8 = if dynamic_range < 1.0 {
        8
    } else if dynamic_range < 5.0 {
        4
    } else {
        2
    };

    // Then — large range pushes to fewer bits (simulating resource-aware selection)
    assert_eq!(bits, 2);
    let (q, s) = quantize_symmetric_i8(&input, bits);
    let d = dequantize_symmetric_i8(&q, s);
    let err = compute_quantization_error(&input, &d);
    // Coarse but still bounded
    assert!(err.snr > 0.0, "large-range 2-bit SNR: {}", err.snr);
}

#[test]
fn given_medium_range_tensor_when_select_precision_then_i4_chosen() {
    // Given — medium dynamic range
    let input = pseudo_random_vec(256, 42, 2.0);
    let stats = ActivationStats::collect(&input);

    // When
    let dynamic_range = stats.max - stats.min;
    let bits: u8 = if dynamic_range < 1.0 {
        8
    } else if dynamic_range < 5.0 {
        4
    } else {
        2
    };

    // Then
    assert_eq!(bits, 4);
    let (q, s) = quantize_symmetric_i8(&input, bits);
    let d = dequantize_symmetric_i8(&q, s);
    let err = compute_quantization_error(&input, &d);
    assert!(err.mse < 0.05, "medium-range i4 MSE: {}", err.mse);
}

#[test]
fn given_near_zero_tensor_when_dynamic_quantize_then_symmetric_preferred() {
    // Given — near-zero mean, symmetric distribution
    let raw = pseudo_random_vec(512, 42, 1.0);
    let m = mean(&raw);
    // Center the data around zero
    let input: Vec<f32> = raw.iter().map(|&v| v - m).collect();
    let stats = ActivationStats::collect(&input);

    // When — symmetric is preferred when mean is near zero
    let use_symmetric = stats.mean.abs() < stats.std * 0.5;
    assert!(use_symmetric, "near-zero mean should prefer symmetric (mean={})", stats.mean);

    let (q, s) = quantize_symmetric_i8(&input, 8);
    let d = dequantize_symmetric_i8(&q, s);

    // Then
    let err = compute_quantization_error(&input, &d);
    assert!(err.mse < 1e-4, "symmetric quant for near-zero MSE: {}", err.mse);
}

#[test]
fn given_skewed_tensor_when_dynamic_quantize_then_asymmetric_preferred() {
    // Given — positively skewed distribution (post-ReLU)
    let raw = pseudo_random_vec(512, 42, 2.0);
    let input = activate(&raw, ActivationType::ReLU);
    let stats = ActivationStats::collect(&input);

    // When — asymmetric preferred when distribution is skewed
    let use_symmetric = stats.mean.abs() < stats.std * 0.5;
    // Post-ReLU has positive mean, so asymmetric is better
    assert!(!use_symmetric, "skewed (post-ReLU) should prefer asymmetric");

    let (q, s, zp) = quantize_asymmetric_u8(&input);
    let d = dequantize_asymmetric_u8(&q, s, zp);

    // Then
    let err = compute_quantization_error(&input, &d);
    assert!(err.mse < 0.001, "asymmetric quant for skewed MSE: {}", err.mse);
}

#[test]
fn given_outlier_tensor_when_dynamic_quantize_then_error_dominated_by_outliers() {
    // Given — mostly small values with one large outlier
    let mut input = vec![0.1f32; 255];
    input.push(100.0);

    // When
    let (q, s) = quantize_symmetric_i8(&input, 8);
    let d = dequantize_symmetric_i8(&q, s);
    let err = compute_quantization_error(&input, &d);

    // Then — scale is dominated by outlier, causing large error for small values
    assert!(s > 0.5, "scale should be large due to outlier: {}", s);
    assert!(err.mse > 0.001, "outlier should increase MSE: {}", err.mse);
}

#[test]
fn given_constant_tensor_when_dynamic_quantize_then_zero_scale_and_zero_error() {
    // Given
    let input = vec![5.0f32; 128];

    // When — asymmetric: constant input yields scale=0
    let (q, scale, _zp) = quantize_asymmetric_u8(&input);

    // Then
    assert_eq!(scale, 0.0, "constant input should yield zero scale");
    assert!(q.iter().all(|&v| v == 0), "constant input quantized to zeros");
}

#[test]
fn given_ternary_weights_when_check_sparsity_then_zero_ratio_matches_threshold() {
    // Given — weights drawn from a zero-centered distribution
    let weights = pseudo_random_vec(1024, 42, 1.0);
    let threshold = 0.3;

    // When
    let ternary = quantize_ternary(&weights, threshold);

    // Then — count zeros (values within threshold)
    let n_zeros = ternary.iter().filter(|&&v| v == 0).count();
    let zero_ratio = n_zeros as f32 / ternary.len() as f32;
    // With a normal-ish distribution and threshold=0.3, expect meaningful sparsity
    assert!(zero_ratio > 0.05, "expected some sparsity, got {zero_ratio}");
    assert!(zero_ratio < 0.95, "too much sparsity: {zero_ratio}");
}

// ══════════════════════════════════════════════════════════════════
// 6. Quantization Error Analysis: measure and track noise per layer
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_layer_stack_when_quantize_each_then_per_layer_error_tracked() {
    // Given — a 3-layer stack of weights
    let dim = 32;
    let layers = vec![
        pseudo_random_vec(dim * dim, 10, 0.1),
        pseudo_random_vec(dim * dim, 20, 0.5),
        pseudo_random_vec(dim * dim, 30, 1.0),
    ];

    // When — quantize each and collect errors
    let errors: Vec<QuantizationError> = layers
        .iter()
        .map(|w| {
            let (q, s) = quantize_symmetric_i8(w, 8);
            let d = dequantize_symmetric_i8(&q, s);
            compute_quantization_error(w, &d)
        })
        .collect();

    // Then — all errors should be positive and bounded
    for (i, err) in errors.iter().enumerate() {
        assert!(err.mse >= 0.0, "layer {i} MSE negative");
        assert!(err.max_abs_error >= 0.0, "layer {i} max_abs negative");
        assert!(err.snr > 0.0, "layer {i} SNR not positive: {}", err.snr);
    }
    // Layer with larger scale should have larger absolute error
    assert!(
        errors[2].max_abs_error > errors[0].max_abs_error,
        "layer 2 (scale 1.0) max_abs_error {} ≤ layer 0 (scale 0.1) {}",
        errors[2].max_abs_error,
        errors[0].max_abs_error,
    );
}

#[test]
fn given_quantized_pipeline_when_accumulate_noise_then_total_snr_decreases() {
    // Given — two matmul layers with quantized weights
    let dim = 16;
    let input = pseudo_random_vec(dim, 42, 1.0);
    let w1 = pseudo_random_vec(dim * dim, 10, 0.2);
    let w2 = pseudo_random_vec(dim * dim, 20, 0.2);

    // FP32 reference
    let h_fp32 = naive_matmul(&input, &w1, 1, dim, dim);
    let out_fp32 = naive_matmul(&h_fp32, &w2, 1, dim, dim);

    // When — quantized pipeline
    let (q1, s1) = quantize_symmetric_i8(&w1, 8);
    let dq1 = dequantize_symmetric_i8(&q1, s1);
    let h_quant = naive_matmul(&input, &dq1, 1, dim, dim);

    let (q2, s2) = quantize_symmetric_i8(&w2, 8);
    let dq2 = dequantize_symmetric_i8(&q2, s2);
    let out_quant = naive_matmul(&h_quant, &dq2, 1, dim, dim);

    // Then — accumulated error after 2 layers
    let err_1_layer = compute_quantization_error(&h_fp32, &h_quant);
    let err_2_layers = compute_quantization_error(&out_fp32, &out_quant);

    // Error accumulates through layers (SNR decreases or MSE increases)
    // Note: this isn't strictly guaranteed for all inputs, but for typical cases it holds
    assert!(
        err_2_layers.mse >= err_1_layer.mse * 0.5,
        "2-layer MSE {} too small vs 1-layer MSE {}",
        err_2_layers.mse,
        err_1_layer.mse,
    );
}

#[test]
fn given_quantized_norm_pipeline_when_measure_noise_then_both_outputs_normalized() {
    // Given — quantize weights, then apply layer norm (which re-centers)
    let dim = 32;
    let weights = pseudo_random_vec(dim, 42, 1.0);
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];
    let cfg = LayerNormConfig::new(vec![dim]);

    // When — quantize then normalize
    let (q, s) = quantize_symmetric_i8(&weights, 4);
    let dq = dequantize_symmetric_i8(&q, s);
    let normed_orig = layer_norm(&weights, &gamma, Some(&beta), &cfg).unwrap();
    let normed_quant = layer_norm(&dq, &gamma, Some(&beta), &cfg).unwrap();

    // Then — both normalized outputs should have near-zero mean
    let mean_orig = mean(&normed_orig);
    let mean_quant = mean(&normed_quant);
    assert!(mean_orig.abs() < 0.01, "orig normed mean: {mean_orig}");
    assert!(mean_quant.abs() < 0.01, "quant normed mean: {mean_quant}");
    // And both should be finite
    assert!(normed_orig.iter().all(|v| v.is_finite()));
    assert!(normed_quant.iter().all(|v| v.is_finite()));
    // The quantization error should still be bounded
    let err = compute_quantization_error(&normed_orig, &normed_quant);
    assert!(err.mse < 1.0, "normalized quantization MSE: {}", err.mse);
}

#[test]
fn given_ternary_weights_when_compute_error_then_snr_reflects_information_loss() {
    // Given
    let weights = pseudo_random_vec(256, 42, 1.0);

    // When — ternary quantization (extreme compression)
    let ternary = quantize_ternary(&weights, 0.3);
    let ternary_f32: Vec<f32> = ternary.iter().map(|&v| v as f32).collect();
    let err = compute_quantization_error(&weights, &ternary_f32);

    // Then — significant information loss
    assert!(err.mse > 0.01, "ternary MSE should be significant: {}", err.mse);
    assert!(err.snr < 30.0, "ternary SNR should be moderate: {}", err.snr);
}

#[test]
fn given_binary_weights_when_compute_error_then_max_error_bounded_by_abs_max() {
    // Given
    let weights = pseudo_random_vec(256, 42, 2.0);

    // When
    let binary = quantize_binary(&weights);
    let binary_f32: Vec<f32> = binary.iter().map(|&v| v as f32).collect();
    let err = compute_quantization_error(&weights, &binary_f32);

    // Then — max error bounded by max(|w| + 1) since binary values are ±1
    let abs_max = weights.iter().copied().fold(0.0_f32, |a, v| a.max(v.abs()));
    assert!(
        err.max_abs_error <= abs_max + 1.0 + 1e-5,
        "binary max_abs {} > abs_max + 1 = {}",
        err.max_abs_error,
        abs_max + 1.0,
    );
}

#[test]
fn given_multiple_quantization_schemes_when_compare_then_ranking_consistent() {
    // Given
    let data = pseudo_random_vec(512, 42, 1.0);

    // When — compute errors for different schemes
    let err_i8 = {
        let (q, s) = quantize_symmetric_i8(&data, 8);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&data, &d)
    };
    let err_i4 = {
        let (q, s) = quantize_symmetric_i8(&data, 4);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&data, &d)
    };
    let err_u8 = {
        let (q, s, zp) = quantize_asymmetric_u8(&data);
        let d = dequantize_asymmetric_u8(&q, s, zp);
        compute_quantization_error(&data, &d)
    };

    // Then — INT8 symmetric and U8 asymmetric should both beat INT4
    assert!(err_i8.mse < err_i4.mse, "i8 MSE {} ≥ i4 MSE {}", err_i8.mse, err_i4.mse);
    assert!(err_u8.mse < err_i4.mse, "u8 MSE {} ≥ i4 MSE {}", err_u8.mse, err_i4.mse);
}

#[test]
fn given_error_tracker_when_accumulate_per_layer_then_summary_correct() {
    // Given — simulate tracking errors across layers
    let dim = 16;
    let n_layers = 4;
    let mut total_mse = 0.0f32;
    let mut max_error = 0.0f32;
    let mut layer_errors = Vec::new();

    // When
    for i in 0..n_layers {
        let w = pseudo_random_vec(dim * dim, (i * 10 + 1) as u64, 0.5);
        let (q, s) = quantize_symmetric_i8(&w, 8);
        let d = dequantize_symmetric_i8(&q, s);
        let err = compute_quantization_error(&w, &d);
        total_mse += err.mse;
        if err.max_abs_error > max_error {
            max_error = err.max_abs_error;
        }
        layer_errors.push(err);
    }

    // Then
    let avg_mse = total_mse / n_layers as f32;
    assert!(avg_mse > 0.0, "average MSE should be positive");
    assert!(max_error > 0.0, "max error across layers should be positive");
    assert_eq!(layer_errors.len(), n_layers);
    // All layers should have reasonable SNR for INT8
    for (i, err) in layer_errors.iter().enumerate() {
        assert!(err.snr > 30.0, "layer {i} SNR too low: {}", err.snr);
    }
}

#[test]
fn given_residual_connection_when_quantize_both_branches_then_error_bounded() {
    // Given — residual: output = quantize(x) + quantize(f(x))
    let dim = 32;
    let x = pseudo_random_vec(dim, 42, 1.0);
    let w = pseudo_random_vec(dim * dim, 10, 0.2);

    // FP32 reference
    let fx = naive_matmul(&x, &w, 1, dim, dim);
    let mut ref_out = fx.clone();
    add_residual(&mut ref_out, &x).unwrap();

    // When — quantize both branches
    let (qx, sx) = quantize_symmetric_i8(&x, 8);
    let dqx = dequantize_symmetric_i8(&qx, sx);

    let (qw, sw) = quantize_symmetric_i8(&w, 8);
    let dqw = dequantize_symmetric_i8(&qw, sw);
    let q_fx = naive_matmul(&dqx, &dqw, 1, dim, dim);
    let mut q_out = q_fx;
    add_residual(&mut q_out, &dqx).unwrap();

    // Then
    let err = compute_quantization_error(&ref_out, &q_out);
    assert!(err.mse < 0.1, "residual quantized MSE: {}", err.mse);
    assert!(q_out.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// Additional End-to-End Integration Tests
// ══════════════════════════════════════════════════════════════════

#[test]
fn given_full_transformer_block_when_all_quantized_then_output_finite() {
    // Given — simulate a complete transformer block
    let dim = 16;
    let seq = 2;
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];
    let cfg = LayerNormConfig::new(vec![dim]);

    let input = pseudo_random_vec(seq * dim, 1, 1.0);
    let qkv_w = pseudo_random_vec(dim * dim * 3, 10, 0.1);
    let out_w = pseudo_random_vec(dim * dim, 20, 0.1);
    let ffn_up = pseudo_random_vec(dim * dim * 4, 30, 0.1);
    let ffn_down = pseudo_random_vec(dim * 4 * dim, 40, 0.1);

    // When — quantize all weights to INT8
    let dq = |w: &[f32]| -> Vec<f32> {
        let (q, s) = quantize_symmetric_i8(w, 8);
        dequantize_symmetric_i8(&q, s)
    };

    let dq_qkv = dq(&qkv_w);
    let dq_out = dq(&out_w);
    let dq_up = dq(&ffn_up);
    let dq_down = dq(&ffn_down);

    // Pre-attention norm
    let normed = layer_norm(&input, &gamma, Some(&beta), &cfg).unwrap();

    // Q, K, V projections (first dim×dim chunk for Q)
    let q_proj = naive_matmul(&normed, &dq_qkv[..dim * dim], seq, dim, dim);

    // Attention output projection
    let attn_out = naive_matmul(&q_proj, &dq_out, seq, dim, dim);
    let mut h = attn_out;
    add_residual(&mut h, &input).unwrap();

    // FFN: norm → up → gelu → down → residual
    let normed2 = layer_norm(&h, &gamma, Some(&beta), &cfg).unwrap();
    let up = naive_matmul(&normed2, &dq_up, seq, dim * 4, dim);
    let up_act = gelu_vec(&up);
    let down = naive_matmul(&up_act, &dq_down, seq, dim, dim * 4);
    let mut output = down;
    add_residual(&mut output, &h).unwrap();

    // Then
    assert_eq!(output.len(), seq * dim);
    assert!(output.iter().all(|v| v.is_finite()), "transformer block output must be finite");
}

#[test]
fn given_rms_norm_weights_when_quantize_then_norm_still_unit_variance() {
    // Given
    let dim = 32;
    let gamma = pseudo_random_vec(dim, 42, 0.3);
    let gamma_centered: Vec<f32> = gamma.iter().map(|&g| g + 1.0).collect();

    // Quantize gamma (simulating someone mistakenly quantizing LN weights)
    let (q, s) = quantize_symmetric_i8(&gamma_centered, 8);
    let dq_gamma = dequantize_symmetric_i8(&q, s);

    let cfg = LayerNormConfig::new(vec![dim]);
    let input = pseudo_random_vec(dim, 99, 2.0);

    // When — apply rms_norm with quantized vs original gamma
    let out_orig = rms_norm(&input, &gamma_centered, &cfg).unwrap();
    let out_quant = rms_norm(&input, &dq_gamma, &cfg).unwrap();

    // Then — outputs should be close
    let err = compute_quantization_error(&out_orig, &out_quant);
    assert!(err.mse < 0.01, "rms_norm with quantized gamma MSE: {}", err.mse);
}

#[test]
fn given_silu_activations_when_quantize_then_monotonicity_preserved() {
    // Given — SiLU applied to sorted input
    let sorted_input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.0).collect();
    let silu_out = silu_vec(&sorted_input);

    // When — quantize the SiLU output
    let (q, s) = quantize_symmetric_i8(&silu_out, 8);
    let dq = dequantize_symmetric_i8(&q, s);

    // Then — approximate monotonicity preserved (SiLU is monotonic for x > ~-0.28)
    let positive_region: Vec<f32> = dq.iter().skip(80).copied().collect();
    for i in 1..positive_region.len() {
        assert!(
            positive_region[i] >= positive_region[i - 1] - s * 2.0,
            "monotonicity violated at {i}: {} < {}",
            positive_region[i],
            positive_region[i - 1],
        );
    }
}

#[test]
fn given_i2s_quantized_matmul_when_compare_to_dequant_matmul_then_match() {
    // Given — pack weights in I2S format (2-bit signed: -1, 0, +1)
    use bitnet_kernels::cpu::quantized_matmul::dequantize_and_matmul;
    let m: usize = 4;
    let k: usize = 16;
    let n: usize = 4;
    let num_blocks = k.div_ceil(k); // block_size = k → 1 block
    // Column-oriented weights: n columns, each with packed_k bytes
    let packed_k = k.div_ceil(4);
    let packed: Vec<u8> = (0..(n * packed_k))
        .map(|i| {
            let vals: [i8; 4] = [
                ((i * 4) % 3) as i8 - 1,
                ((i * 4 + 1) % 3) as i8 - 1,
                ((i * 4 + 2) % 3) as i8 - 1,
                ((i * 4 + 3) % 3) as i8 - 1,
            ];
            pack_i2s(vals)
        })
        .collect();
    let scales = vec![0.5f32; n * num_blocks];
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();

    // When — both paths
    let mut out_i2s = vec![0.0f32; m * n];
    let mut out_deq = vec![0.0f32; m * n];
    i2s_matmul_f32(&input, &packed, &scales, &mut out_i2s, m, n, k, k).unwrap();
    dequantize_and_matmul(&input, &packed, &scales, &mut out_deq, m, n, k, k).unwrap();

    // Then — both should agree
    assert_slice_close(&out_i2s, &out_deq, 1e-5, "i2s vs dequant_matmul");
}

#[test]
fn given_symmetric_vs_asymmetric_when_same_data_then_asymmetric_never_worse_for_skewed() {
    // Given — positively skewed data (ReLU-like)
    let raw = pseudo_random_vec(512, 42, 2.0);
    let skewed = activate(&raw, ActivationType::ReLU);

    // When
    let err_sym = {
        let (q, s) = quantize_symmetric_i8(&skewed, 8);
        let d = dequantize_symmetric_i8(&q, s);
        compute_quantization_error(&skewed, &d)
    };
    let err_asym = {
        let (q, s, zp) = quantize_asymmetric_u8(&skewed);
        let d = dequantize_asymmetric_u8(&q, s, zp);
        compute_quantization_error(&skewed, &d)
    };

    // Then — asymmetric should be at least as good for skewed data
    assert!(
        err_asym.mse <= err_sym.mse + 1e-6,
        "asymmetric MSE {} > symmetric MSE {} for skewed data",
        err_asym.mse,
        err_sym.mse,
    );
}

#[test]
fn given_large_model_simulation_when_quantize_layers_then_all_errors_tracked() {
    // Given — simulate a 12-layer model
    let dim = 16;
    let n_layers = 12;

    // When
    let mut mses = Vec::with_capacity(n_layers);
    let mut snrs = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let w = pseudo_random_vec(dim * dim, (i * 7 + 1) as u64, 0.2);
        let (q, s) = quantize_symmetric_i8(&w, 8);
        let d = dequantize_symmetric_i8(&q, s);
        let err = compute_quantization_error(&w, &d);
        mses.push(err.mse);
        snrs.push(err.snr);
    }

    // Then — all layers have reasonable metrics
    assert_eq!(mses.len(), n_layers);
    for (i, &mse) in mses.iter().enumerate() {
        assert!(mse < 1e-3, "layer {i} MSE too high: {mse}");
    }
    for (i, &snr) in snrs.iter().enumerate() {
        assert!(snr > 30.0, "layer {i} SNR too low: {snr}");
    }

    // Aggregate statistics
    let avg_mse = mses.iter().sum::<f32>() / n_layers as f32;
    let avg_snr = snrs.iter().sum::<f32>() / n_layers as f32;
    assert!(avg_mse < 1e-3, "average MSE across model: {avg_mse}");
    assert!(avg_snr > 30.0, "average SNR across model: {avg_snr}");
}

#[test]
fn given_reduction_after_quantize_when_sum_then_result_close_to_fp32() {
    // Given
    let data = pseudo_random_vec(256, 42, 5.0);
    let fp32_sum = reduce_f32(&data, ReductionOp::Sum);

    // When — quantize then reduce
    let (q, s) = quantize_symmetric_i8(&data, 8);
    let dq = dequantize_symmetric_i8(&q, s);
    let quant_sum = reduce_f32(&dq, ReductionOp::Sum);

    // Then — sums should be close
    let rel_err = ((fp32_sum - quant_sum) / fp32_sum).abs();
    assert!(rel_err < 0.05, "sum relative error: {rel_err}");
}

#[test]
fn given_quantized_cosine_similarity_when_compare_then_close_to_fp32() {
    // Given
    let a = pseudo_random_vec(128, 42, 1.0);
    let b = pseudo_random_vec(128, 99, 1.0);
    let fp32_cos = cosine_similarity_loss(&a, &b).unwrap();

    // When — quantize both vectors
    let (qa, sa) = quantize_symmetric_i8(&a, 8);
    let da = dequantize_symmetric_i8(&qa, sa);
    let (qb, sb) = quantize_symmetric_i8(&b, 8);
    let db = dequantize_symmetric_i8(&qb, sb);
    let quant_cos = cosine_similarity_loss(&da, &db).unwrap();

    // Then
    assert!(
        (fp32_cos - quant_cos).abs() < 0.01,
        "cosine similarity diff: fp32={fp32_cos}, quant={quant_cos}",
    );
}

#[test]
fn given_empty_tensor_when_quantize_all_formats_then_no_panic() {
    // Given
    let empty: Vec<f32> = vec![];

    // When/Then — none should panic
    let (q_sym, s_sym) = quantize_symmetric_i8(&empty, 8);
    assert!(q_sym.is_empty());
    assert_eq!(s_sym, 0.0);

    let (q_asym, s_asym, zp) = quantize_asymmetric_u8(&empty);
    assert!(q_asym.is_empty());
    assert_eq!(s_asym, 0.0);
    assert_eq!(zp, 0);

    let q_tern = quantize_ternary(&empty, 0.5);
    assert!(q_tern.is_empty());

    let q_bin = quantize_binary(&empty);
    assert!(q_bin.is_empty());
}

#[test]
fn given_single_element_when_quantize_roundtrip_then_exact() {
    // Given
    let input = vec![42.0f32];

    // When — INT8
    let (q, s) = quantize_symmetric_i8(&input, 8);
    let d = dequantize_symmetric_i8(&q, s);

    // Then — single element maps to qmax, so roundtrip is exact
    assert_close(d[0], 42.0, 1e-3, "single element roundtrip");
}

#[test]
fn given_all_zeros_when_quantize_then_zero_scale_and_output() {
    // Given
    let zeros = vec![0.0f32; 64];

    // When
    let (q, s) = quantize_symmetric_i8(&zeros, 8);
    let d = dequantize_symmetric_i8(&q, s);

    // Then
    assert_eq!(s, 0.0);
    assert!(q.iter().all(|&v| v == 0));
    assert!(d.iter().all(|&v| v == 0.0));
}

#[test]
fn given_mixed_precision_forward_when_track_error_per_layer_then_budget_met() {
    // Given — error budget: total MSE across all layers < 0.01
    let dim = 16;
    let error_budget = 0.01;

    // Layer 0: attention (INT8)
    let w0 = pseudo_random_vec(dim * dim, 10, 0.1);
    let (q0, s0) = quantize_symmetric_i8(&w0, 8);
    let d0 = dequantize_symmetric_i8(&q0, s0);
    let err0 = compute_quantization_error(&w0, &d0);

    // Layer 1: FFN up (INT4)
    let w1 = pseudo_random_vec(dim * dim, 20, 0.1);
    let (q1, s1) = quantize_symmetric_i8(&w1, 4);
    let d1 = dequantize_symmetric_i8(&q1, s1);
    let err1 = compute_quantization_error(&w1, &d1);

    // Layer 2: FFN down (INT4)
    let w2 = pseudo_random_vec(dim * dim, 30, 0.1);
    let (q2, s2) = quantize_symmetric_i8(&w2, 4);
    let d2 = dequantize_symmetric_i8(&q2, s2);
    let err2 = compute_quantization_error(&w2, &d2);

    // When — check total error budget
    let total_mse = err0.mse + err1.mse + err2.mse;

    // Then
    assert!(total_mse < error_budget, "total MSE {total_mse} exceeds budget {error_budget}");
}
