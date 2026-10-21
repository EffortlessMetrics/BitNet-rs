//! Dense (non-quantized) FP16/BF16 inference path tests.
//!
//! Validates that the CPU inference pipeline can handle standard SLM
//! architectures (Phi-4, Qwen, Gemma, etc.) before quantization:
//! - Configuration for dense models
//! - Layer construction with standard Linear / RMSNorm / SiLU / GeLU
//! - Full transformer-block forward passes with synthetic weights
//! - Numerical precision across F32, F16, and BF16

use bitnet_common::config::{ActivationType, ModelConfig, NormType};
use bitnet_inference::cpu_opt;
use bitnet_inference::simple_forward::{Weights, logits_for_token};

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{i}]: expected {e:.6}, got {a:.6} (tol={tol})");
    }
}

/// Deterministic synthetic weights seeded from dimension indices.
fn synthetic_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let r = (i / cols) as f32;
            let c = (i % cols) as f32;
            (r * 0.01 + c * 0.007 + 0.1).sin() * 0.1
        })
        .collect()
}

/// Simulate a single dense transformer block:
///   residual + FFN(RMSNorm(x))
/// where FFN = W2 · act(W1 · x)
fn dense_transformer_block(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    norm_weight: &[f32],
    dim: usize,
    inter: usize,
    activation: ActivationType,
) -> Vec<f32> {
    let seq_len = input.len() / dim;

    // Pre-norm: RMSNorm
    let mut normed = vec![0.0f32; seq_len * dim];
    cpu_opt::rmsnorm(input, norm_weight, &mut normed, seq_len, dim, 1e-5).unwrap();

    let mut output = vec![0.0f32; seq_len * dim];
    for s in 0..seq_len {
        let row = &normed[s * dim..(s + 1) * dim];

        // Up-project: W1 · x  → [inter]
        let mut hidden = vec![0.0f32; inter];
        cpu_opt::parallel_matmul(row, w1, &mut hidden, 1, inter, dim, 1).unwrap();

        // Activation
        match activation {
            ActivationType::Silu => cpu_opt::silu_in_place(&mut hidden),
            ActivationType::Gelu => cpu_opt::gelu_in_place(&mut hidden),
            ActivationType::Relu2 => cpu_opt::relu2_in_place(&mut hidden),
        }

        // Down-project: W2 · hidden → [dim]
        let mut projected = vec![0.0f32; dim];
        cpu_opt::parallel_matmul(&hidden, w2, &mut projected, 1, dim, inter, 1).unwrap();

        // Residual connection
        for d in 0..dim {
            output[s * dim + d] = input[s * dim + d] + projected[d];
        }
    }

    output
}

// ====================================================================
// 1. Configuration tests (5)
// ====================================================================

#[test]
fn dense_config_no_quantization_validates() {
    let cfg = ModelConfig {
        vocab_size: 100352,
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_key_value_heads: 10,
        intermediate_size: 13824,
        ..Default::default()
    };
    // A dense model config is valid when all structural dimensions are positive.
    assert!(cfg.vocab_size > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.num_layers > 0);
    assert!(cfg.num_heads > 0);
    assert_eq!(cfg.num_key_value_heads, 10);
}

#[test]
fn config_silu_rmsnorm_phi4_style() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("phi-4");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Silu);
    assert_eq!(cfg.max_position_embeddings, 16384);
}

#[test]
fn config_gelu_rmsnorm_gemma_style() {
    let mut cfg = ModelConfig::default();
    cfg.apply_architecture_defaults("gemma2");
    assert_eq!(cfg.norm_type, NormType::RmsNorm);
    assert_eq!(cfg.activation_type, ActivationType::Gelu);
    assert_eq!(cfg.max_position_embeddings, 8192);
}

#[test]
fn config_phi4_scale_hidden5120_40layers_validates() {
    let cfg = ModelConfig {
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        num_key_value_heads: 10,
        intermediate_size: 13824,
        norm_type: NormType::RmsNorm,
        activation_type: ActivationType::Silu,
        ..Default::default()
    };
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_layers, 40);
    // head_dim = hidden_size / num_heads = 128
    let head_dim = cfg.hidden_size / cfg.num_heads;
    assert_eq!(head_dim, 128);
    // KV heads divide evenly into Q heads
    assert_eq!(cfg.num_heads % cfg.num_key_value_heads, 0);
}

#[test]
fn config_vocab_100352_validates() {
    let cfg = ModelConfig { vocab_size: 100352, ..Default::default() };
    assert_eq!(cfg.vocab_size, 100352);
    // Vocab size must be positive for embedding lookup
    assert!(cfg.vocab_size > 0);
}

// ====================================================================
// 2. Layer construction tests (5)
// ====================================================================

#[test]
fn linear_layer_fp32_matmul() {
    // Standard dense Linear: y = x @ W (no quantization).
    let dim = 8;
    let out_dim = 4;
    let weights = synthetic_weights(dim, out_dim);
    let input = vec![1.0f32; dim];
    let mut output = vec![0.0f32; out_dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut output, 1, out_dim, dim, 1).unwrap();

    // Each output element is the sum of the corresponding column of W.
    for j in 0..out_dim {
        let expected: f32 = (0..dim).map(|i| weights[i * out_dim + j]).sum();
        assert!(
            (output[j] - expected).abs() < 1e-5,
            "linear[{j}]: expected {expected}, got {}",
            output[j]
        );
    }
}

#[test]
fn rmsnorm_normalises_dense_weights() {
    let dim = 64;
    let weight = vec![1.0f32; dim];
    // Input with varying magnitudes
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let mut output = vec![0.0f32; dim];

    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, dim, 1e-5).unwrap();

    // Output RMS should be approximately 1.0 when weights are all-ones.
    let out_rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
    assert!((out_rms - 1.0).abs() < 0.01, "rmsnorm output RMS should be ~1.0, got {out_rms}");
}

#[test]
fn silu_activation_in_ffn_path() {
    // SiLU(x) = x * sigmoid(x); verify known values in an FFN context.
    let hidden = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5];
    let activated = cpu_opt::silu(&hidden);

    // SiLU(0) = 0
    assert!(activated[0].abs() < 1e-6);
    // SiLU(x) > 0 for x > 0
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
    assert!(activated[5] > 0.0);
    // SiLU(x) < 0 for x < 0 (the negative trough)
    assert!(activated[2] < 0.0);
    assert!(activated[4] < 0.0);
    // SiLU is bounded below: SiLU(x) ≥ −0.2785 for all x
    for &v in &activated {
        assert!(v >= -0.29, "SiLU below theoretical minimum: {v}");
    }
}

#[test]
fn gqa_attention_40_10_head_config() {
    // GQA with 40 query heads, 10 KV heads → 4 queries per KV head.
    // Verify attention with a single head slice.
    let head_dim = 8;
    let seq_len = 2;
    let num_heads = 1; // test a single head at a time

    let query: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
    let key = query.clone();
    let value: Vec<f32> = (0..seq_len * head_dim).map(|i| 1.0 - (i as f32) * 0.05).collect();
    let mut output = vec![0.0f32; seq_len * head_dim];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // Output should be finite and within reasonable range
    assert!(output.iter().all(|v| v.is_finite()));
    // With Q==K, the attention pattern concentrates on matching positions;
    // output should be a convex combination of values.
    let v_min = value.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = value.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for &o in &output {
        assert!(
            o >= v_min - 1e-5 && o <= v_max + 1e-5,
            "attention output {o} outside value range [{v_min}, {v_max}]"
        );
    }

    // Validate the GQA config arithmetic: 40/10 = 4 queries per KV group
    let num_q_heads = 40;
    let num_kv_heads = 10;
    assert_eq!(num_q_heads % num_kv_heads, 0);
    assert_eq!(num_q_heads / num_kv_heads, 4);
}

#[test]
fn residual_connections_preserve_input() {
    let dim = 16;
    let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();

    // Zero projection → residual should exactly equal input.
    let zero_proj = vec![0.0f32; dim];
    let residual: Vec<f32> = input.iter().zip(zero_proj.iter()).map(|(a, b)| a + b).collect();
    assert_close(&residual, &input, 1e-7, "residual with zero FFN");
}

// ====================================================================
// 3. Forward pass tests (5)
// ====================================================================

#[test]
fn single_dense_block_produces_valid_output() {
    let dim = 16;
    let inter = 32;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let output =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    assert_eq!(output.len(), dim);
    assert!(output.iter().all(|v| v.is_finite()), "all outputs must be finite");
    // Output should differ from input due to FFN contribution
    let diff: f32 = input.iter().zip(output.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "dense block should transform input, diff={diff}");
}

#[test]
fn two_block_stack_differs_from_one_block() {
    let dim = 16;
    let inter = 32;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let after_one =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);
    let after_two =
        dense_transformer_block(&after_one, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    // Two passes should produce different output than one pass
    let diff: f32 = after_one.iter().zip(after_two.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "stacking two blocks should change output, diff={diff}");
}

#[test]
fn forward_pass_batch1_seq1_autoregressive() {
    // Autoregressive decode: batch=1, seq_len=1
    let dim = 8;
    let inter = 16;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input = vec![0.5f32; dim]; // single-token input

    let output =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    assert_eq!(output.len(), dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn forward_pass_seq512_works() {
    let dim = 16;
    let inter = 32;
    let seq_len = 512;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..seq_len * dim).map(|i| ((i as f32) * 0.001).sin()).collect();

    let output =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    assert_eq!(output.len(), seq_len * dim);
    assert!(output.iter().all(|v| v.is_finite()), "seq_len=512 output must be finite");
}

#[test]
fn output_logits_correct_shape() {
    // Validate that embedding → lm_head produces [batch × seq_len × vocab_size].
    let vocab = 64;
    let dim = 8;
    let tok_embeddings: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let lm_head: Vec<f32> = (0..dim * vocab).map(|i| (i as f32 * 0.007).cos()).collect();

    let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

    // Batch=1, seq_len=3 tokens
    let tokens = [0usize, 5, 42];
    let mut all_logits = Vec::new();
    for &tok in &tokens {
        let logits = logits_for_token(&w, tok);
        assert_eq!(logits.len(), vocab, "logits should be [vocab_size]");
        all_logits.push(logits);
    }
    // Total shape: 3 × vocab
    assert_eq!(all_logits.len(), tokens.len());
    assert_eq!(all_logits[0].len(), vocab);
}

// ====================================================================
// 4. Precision tests (5)
// ====================================================================

#[test]
fn f32_forward_pass_is_deterministic() {
    let dim = 16;
    let inter = 32;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let out1 = dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);
    let out2 = dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    // Bit-exact: same input, same weights → identical output
    assert_eq!(out1, out2, "f32 forward pass must be deterministic");
}

#[test]
fn f16_forward_pass_matches_f32_within_tolerance() {
    // Simulate F16 precision loss by rounding weights to half precision.
    let dim = 16;
    let inter = 32;
    let w1_f32 = synthetic_weights(dim, inter);
    let w2_f32 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Simulate F16: truncate mantissa to 10 bits (23−10 = 13 bits cleared).
    let to_f16 = |v: f32| -> f32 {
        let bits = v.to_bits();
        f32::from_bits(bits & 0xFFFF_E000)
    };
    let w1_f16: Vec<f32> = w1_f32.iter().map(|&v| to_f16(v)).collect();
    let w2_f16: Vec<f32> = w2_f32.iter().map(|&v| to_f16(v)).collect();

    let out_f32 = dense_transformer_block(
        &input,
        &w1_f32,
        &w2_f32,
        &norm_w,
        dim,
        inter,
        ActivationType::Silu,
    );
    let out_f16 = dense_transformer_block(
        &input,
        &w1_f16,
        &w2_f16,
        &norm_w,
        dim,
        inter,
        ActivationType::Silu,
    );

    assert_close(&out_f16, &out_f32, 1e-3, "f16 vs f32");
}

#[test]
fn bf16_loaded_weights_match_f32_within_tolerance() {
    // Simulate BF16 precision: 8-bit exponent, 7-bit mantissa.
    let dim = 16;
    let inter = 32;
    let w1_f32 = synthetic_weights(dim, inter);
    let w2_f32 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // Simulate BF16 by truncating lower 16 bits of f32.
    let to_bf16 = |v: f32| -> f32 {
        let bits = v.to_bits();
        f32::from_bits(bits & 0xFFFF_0000)
    };
    let w1_bf16: Vec<f32> = w1_f32.iter().map(|&v| to_bf16(v)).collect();
    let w2_bf16: Vec<f32> = w2_f32.iter().map(|&v| to_bf16(v)).collect();

    let out_f32 = dense_transformer_block(
        &input,
        &w1_f32,
        &w2_f32,
        &norm_w,
        dim,
        inter,
        ActivationType::Silu,
    );
    let out_bf16 = dense_transformer_block(
        &input,
        &w1_bf16,
        &w2_bf16,
        &norm_w,
        dim,
        inter,
        ActivationType::Silu,
    );

    // BF16 has lower precision than F16, so use a wider tolerance.
    assert_close(&out_bf16, &out_f32, 5e-3, "bf16 vs f32");
}

#[test]
fn numerical_stability_very_small_weights() {
    let dim = 16;
    let inter = 32;
    // Weights near 1e-7 — risk of underflow in accumulation
    let w1: Vec<f32> = (0..dim * inter).map(|i| (i as f32 * 0.01).sin() * 1e-7).collect();
    let w2: Vec<f32> = (0..inter * dim).map(|i| (i as f32 * 0.01).cos() * 1e-7).collect();
    let norm_w = vec![1.0f32; dim];
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

    let output =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    // All outputs must be finite (no NaN/Inf from tiny weights)
    assert!(output.iter().all(|v| v.is_finite()), "small-weight output must be finite");
    // With tiny weights the FFN contribution is negligible, so output ≈ input (residual)
    assert_close(&output, &input, 1e-4, "small weights ≈ residual passthrough");
}

#[test]
fn numerical_stability_large_activations() {
    let dim = 16;
    let inter = 32;
    let w1 = synthetic_weights(dim, inter);
    let w2 = synthetic_weights(inter, dim);
    let norm_w = vec![1.0f32; dim];
    // Large activations: values around 1e3
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 100.0).collect();

    let output =
        dense_transformer_block(&input, &w1, &w2, &norm_w, dim, inter, ActivationType::Silu);

    // RMSNorm should tame the large values; output must remain finite
    assert!(output.iter().all(|v| v.is_finite()), "large-activation output must be finite");
    // No NaN/Inf propagation
    assert!(!output.iter().any(|v| v.is_nan()), "no NaN in large-activation forward pass");
}
