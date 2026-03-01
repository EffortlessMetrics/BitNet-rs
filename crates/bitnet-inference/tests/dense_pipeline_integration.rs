//! Dense model pipeline integration tests.
//!
//! Validates end-to-end data flow through the CPU inference ops:
//! matmul → activation → normalization → attention, with known-value
//! inputs for deterministic golden-output verification.

use bitnet_inference::cpu_opt;

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < tol, "{label}[{i}]: expected {e:.6}, got {a:.6} (tol={tol})",);
    }
}

/// Build an identity-like weight matrix (I_{dim}) stored row-major.
fn identity_weights(dim: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w[i * dim + i] = 1.0;
    }
    w
}

// ────────────────────────────────────────────────────────────────
// 1. Matmul → SiLU pipeline
// ────────────────────────────────────────────────────────────────

#[test]
fn matmul_then_silu_identity_weight() {
    // Identity matmul should preserve input, then SiLU transforms it.
    let dim = 4;
    let input = vec![1.0f32, -1.0, 0.5, 2.0];
    let weights = identity_weights(dim);
    let mut output = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut output, 1, dim, dim, 1).unwrap();

    // After identity matmul, output == input
    assert_close(&output, &input, 1e-6, "identity matmul");

    // Now apply SiLU
    let activated = cpu_opt::silu(&output);

    // SiLU(x) = x * sigmoid(x)
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    assert_close(&activated, &expected, 1e-5, "silu after matmul");
}

#[test]
fn matmul_then_gelu_known_values() {
    let dim = 4;
    let input = vec![0.0f32, 1.0, -1.0, 0.5];
    let weights = identity_weights(dim);
    let mut output = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut output, 1, dim, dim, 1).unwrap();

    let activated = cpu_opt::gelu(&output);

    // GELU(0) ≈ 0
    assert!(activated[0].abs() < 1e-5, "gelu(0) should be ~0");
    // GELU(1) ≈ 0.841
    assert!((activated[1] - 0.841).abs() < 0.01, "gelu(1) ≈ 0.841, got {}", activated[1]);
    // GELU(-1) ≈ -0.159
    assert!((activated[2] - (-0.159)).abs() < 0.01, "gelu(-1) ≈ -0.159, got {}", activated[2]);
    // GELU(0.5) ≈ 0.346
    assert!((activated[3] - 0.346).abs() < 0.01, "gelu(0.5) ≈ 0.346, got {}", activated[3]);
}

// ────────────────────────────────────────────────────────────────
// 2. Matmul → Normalization pipeline
// ────────────────────────────────────────────────────────────────

#[test]
fn matmul_then_rmsnorm_pipeline() {
    let dim = 4;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights_mat = identity_weights(dim);
    let mut matmul_out = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights_mat, &mut matmul_out, 1, dim, dim, 1).unwrap();

    // RMSNorm with unit weights
    let norm_weight = vec![1.0f32; dim];
    let mut norm_out = vec![0.0f32; dim];
    let eps = 1e-5;

    cpu_opt::rmsnorm(&matmul_out, &norm_weight, &mut norm_out, 1, dim, eps).unwrap();

    // Verify: rms = sqrt(mean(x^2) + eps)
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / dim as f32;
    let rms = (mean_sq + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();

    assert_close(&norm_out, &expected, 1e-5, "matmul→rmsnorm");

    // RMSNorm output should have roughly unit RMS
    let out_rms: f32 = (norm_out.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
    assert!((out_rms - 1.0).abs() < 0.01, "rmsnorm output should have ~unit rms, got {out_rms}");
}

#[test]
fn matmul_then_layernorm_pipeline() {
    let dim = 4;
    let input = vec![2.0f32, 4.0, 6.0, 8.0];
    let weights_mat = identity_weights(dim);
    let mut matmul_out = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights_mat, &mut matmul_out, 1, dim, dim, 1).unwrap();

    let norm_weight = vec![1.0f32; dim];
    let norm_bias = vec![0.0f32; dim];
    let mut norm_out = vec![0.0f32; dim];
    let eps = 1e-5;

    cpu_opt::layernorm(&matmul_out, &norm_weight, &norm_bias, &mut norm_out, 1, dim, eps).unwrap();

    // LayerNorm should center and scale: mean ≈ 0, std ≈ 1
    let mean: f32 = norm_out.iter().sum::<f32>() / dim as f32;
    assert!(mean.abs() < 1e-5, "layernorm output should have ~zero mean, got {mean}");

    let var: f32 = norm_out.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
    assert!((var - 1.0).abs() < 0.01, "layernorm output should have ~unit variance, got {var}");
}

// ────────────────────────────────────────────────────────────────
// 3. Full FFN block: matmul → activation → matmul → norm
// ────────────────────────────────────────────────────────────────

#[test]
fn full_ffn_block_silu_rmsnorm() {
    // Simulates a single FFN block: input → W1 → SiLU → W2 → RMSNorm
    let dim = 4;
    let inter = 8; // intermediate size

    // W1: dim→inter (identity-padded: first `dim` cols are identity)
    let mut w1 = vec![0.0f32; dim * inter];
    for i in 0..dim {
        w1[i * inter + i] = 1.0; // first `dim` columns are identity
    }

    // W2: inter→dim (identity-padded: first `dim` rows are identity)
    let mut w2 = vec![0.0f32; inter * dim];
    for i in 0..dim {
        w2[i * dim + i] = 1.0;
    }

    let input = vec![1.0f32, -0.5, 2.0, 0.0];

    // Step 1: matmul W1
    let mut hidden = vec![0.0f32; inter];
    cpu_opt::parallel_matmul(&input, &w1, &mut hidden, 1, inter, dim, 1).unwrap();

    // Step 2: SiLU activation
    cpu_opt::silu_in_place(&mut hidden);

    // Step 3: matmul W2 (project back to dim)
    let mut projected = vec![0.0f32; dim];
    cpu_opt::parallel_matmul(&hidden, &w2, &mut projected, 1, dim, inter, 1).unwrap();

    // Step 4: RMSNorm
    let norm_weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];
    cpu_opt::rmsnorm(&projected, &norm_weight, &mut output, 1, dim, 1e-5).unwrap();

    // Output should be finite and non-zero for non-zero inputs
    assert!(output.iter().all(|v| v.is_finite()), "all outputs should be finite");
    // SiLU(0) = 0, so last element should be 0 after norm (but norm divides by rms so it stays 0)
    // Actually, rms is computed over all elements, so 0/rms = 0
    assert!(output[3].abs() < 1e-6, "SiLU(0) should propagate to ~0");
}

#[test]
fn full_ffn_block_gelu_layernorm() {
    // Simulates a GPT-style FFN: input → W1 → GELU → W2 → LayerNorm
    let dim = 4;
    let inter = 8;

    let mut w1 = vec![0.0f32; dim * inter];
    for i in 0..dim {
        w1[i * inter + i] = 1.0;
    }

    let mut w2 = vec![0.0f32; inter * dim];
    for i in 0..dim {
        w2[i * dim + i] = 1.0;
    }

    let input = vec![0.5f32, 1.0, -1.0, 0.0];

    let mut hidden = vec![0.0f32; inter];
    cpu_opt::parallel_matmul(&input, &w1, &mut hidden, 1, inter, dim, 1).unwrap();
    cpu_opt::gelu_in_place(&mut hidden);

    let mut projected = vec![0.0f32; dim];
    cpu_opt::parallel_matmul(&hidden, &w2, &mut projected, 1, dim, inter, 1).unwrap();

    let norm_weight = vec![1.0f32; dim];
    let norm_bias = vec![0.0f32; dim];
    let mut output = vec![0.0f32; dim];
    cpu_opt::layernorm(&projected, &norm_weight, &norm_bias, &mut output, 1, dim, 1e-5).unwrap();

    assert!(output.iter().all(|v| v.is_finite()), "all outputs should be finite");

    // LayerNorm should center output
    let mean: f32 = output.iter().sum::<f32>() / dim as f32;
    assert!(mean.abs() < 1e-4, "layernorm should center output, got mean={mean}");
}

// ────────────────────────────────────────────────────────────────
// 4. Multi-row batched pipeline
// ────────────────────────────────────────────────────────────────

#[test]
fn batched_matmul_silu_rmsnorm() {
    let batch = 3;
    let dim = 4;

    // Batch of 3 rows
    let input = vec![
        1.0, 0.0, 0.0, 0.0, // row 0: one-hot
        0.0, 1.0, 0.0, 0.0, // row 1: one-hot
        0.5, 0.5, 0.5, 0.5, // row 2: uniform
    ];
    let weights = identity_weights(dim);
    let mut matmul_out = vec![0.0f32; batch * dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut matmul_out, batch, dim, dim, 2).unwrap();

    // Apply SiLU in-place on each row
    cpu_opt::silu_in_place(&mut matmul_out);

    // RMSNorm on batched output
    let norm_weight = vec![1.0f32; dim];
    let mut norm_out = vec![0.0f32; batch * dim];
    cpu_opt::rmsnorm(&matmul_out, &norm_weight, &mut norm_out, batch, dim, 1e-5).unwrap();

    // All outputs should be finite
    assert!(norm_out.iter().all(|v| v.is_finite()), "all batch outputs should be finite");

    // Row 2 (uniform input) should have equal elements after identity matmul + SiLU + RMSNorm
    let row2 = &norm_out[2 * dim..3 * dim];
    let first = row2[0];
    for (i, &v) in row2.iter().enumerate() {
        assert!(
            (v - first).abs() < 1e-5,
            "uniform row should have equal norm outputs, row2[{i}] = {v}, expected {first}"
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 5. Attention → Norm pipeline
// ────────────────────────────────────────────────────────────────

#[test]
fn attention_then_rmsnorm() {
    let seq_len = 2;
    let head_dim = 4;
    let num_heads = 1;
    let size = num_heads * seq_len * head_dim;

    // Orthogonal Q and K → minimal cross-attention
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let key = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let value = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut attn_out = vec![0.0f32; size];

    cpu_opt::parallel_attention(&query, &key, &value, &mut attn_out, seq_len, head_dim, num_heads)
        .unwrap();

    // Attention output should be finite and non-zero
    assert!(attn_out.iter().all(|v| v.is_finite()), "attention output should be finite");
    assert!(attn_out.iter().any(|v| v.abs() > 1e-6), "attention output should be non-zero");

    // Then apply RMSNorm
    let norm_weight = vec![1.0f32; head_dim];
    let mut norm_out = vec![0.0f32; size];
    cpu_opt::rmsnorm(&attn_out, &norm_weight, &mut norm_out, seq_len, head_dim, 1e-5).unwrap();

    assert!(norm_out.iter().all(|v| v.is_finite()), "norm output should be finite");
}

// ────────────────────────────────────────────────────────────────
// 6. Self-attention identity check
// ────────────────────────────────────────────────────────────────

#[test]
fn self_attention_single_token() {
    // Single token: attention over 1 position should return the value itself
    let seq_len = 1;
    let head_dim = 4;
    let num_heads = 2;
    let size = num_heads * seq_len * head_dim;

    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let key = vec![0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
    let value = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let mut output = vec![0.0f32; size];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // With seq_len=1, softmax([score]) = [1.0], so output = value
    assert_close(&output, &value, 1e-5, "single-token attention should return value");
}

// ────────────────────────────────────────────────────────────────
// 7. Numerical stability under extreme values
// ────────────────────────────────────────────────────────────────

#[test]
fn pipeline_with_large_values() {
    let dim = 4;
    let input = vec![100.0f32, -100.0, 50.0, -50.0];
    let weights = identity_weights(dim);
    let mut matmul_out = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut matmul_out, 1, dim, dim, 1).unwrap();

    // SiLU should not overflow/NaN
    let activated = cpu_opt::silu(&matmul_out);
    assert!(activated.iter().all(|v| v.is_finite()), "silu should be finite for large inputs");

    // SiLU(100) ≈ 100 (sigmoid(100) ≈ 1)
    assert!((activated[0] - 100.0).abs() < 0.01, "silu(100) ≈ 100");
    // SiLU(-100) ≈ 0 (sigmoid(-100) ≈ 0)
    assert!(activated[1].abs() < 0.01, "silu(-100) ≈ 0");

    // RMSNorm should still be finite
    let norm_weight = vec![1.0f32; dim];
    let mut norm_out = vec![0.0f32; dim];
    cpu_opt::rmsnorm(&activated, &norm_weight, &mut norm_out, 1, dim, 1e-5).unwrap();
    assert!(
        norm_out.iter().all(|v| v.is_finite()),
        "rmsnorm should be finite after large-value silu"
    );
}

#[test]
fn pipeline_with_near_zero_values() {
    let dim = 4;
    let input = vec![1e-7f32, -1e-7, 1e-8, 0.0];
    let weights = identity_weights(dim);
    let mut matmul_out = vec![0.0f32; dim];

    cpu_opt::parallel_matmul(&input, &weights, &mut matmul_out, 1, dim, dim, 1).unwrap();

    // SiLU near zero: SiLU(x) ≈ x/2 for small x
    let activated = cpu_opt::silu(&matmul_out);
    assert!(activated.iter().all(|v| v.is_finite()), "silu should be finite for near-zero inputs");

    // RMSNorm should handle near-zero (eps prevents division by zero)
    let norm_weight = vec![1.0f32; dim];
    let mut norm_out = vec![0.0f32; dim];
    cpu_opt::rmsnorm(&activated, &norm_weight, &mut norm_out, 1, dim, 1e-5).unwrap();
    assert!(norm_out.iter().all(|v| v.is_finite()), "rmsnorm should handle near-zero input");
}

// ────────────────────────────────────────────────────────────────
// 8. Activation function consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn silu_inplace_matches_functional() {
    let input = vec![-2.0f32, -1.0, 0.0, 0.5, 1.0, 2.0];
    let functional = cpu_opt::silu(&input);

    let mut inplace = input.clone();
    cpu_opt::silu_in_place(&mut inplace);

    assert_close(&inplace, &functional, 1e-6, "silu in-place vs functional");
}

#[test]
fn gelu_inplace_matches_functional() {
    let input = vec![-2.0f32, -1.0, 0.0, 0.5, 1.0, 2.0];
    let functional = cpu_opt::gelu(&input);

    let mut inplace = input.clone();
    cpu_opt::gelu_in_place(&mut inplace);

    assert_close(&inplace, &functional, 1e-6, "gelu in-place vs functional");
}

// ────────────────────────────────────────────────────────────────
// 9. Norm idempotence (applying twice with unit weights)
// ────────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_is_not_idempotent() {
    // RMSNorm applied twice changes the output (unlike a projection).
    // This verifies the transform is non-trivial.
    let dim = 4;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; dim];
    let mut out1 = vec![0.0f32; dim];
    let mut out2 = vec![0.0f32; dim];

    cpu_opt::rmsnorm(&input, &weight, &mut out1, 1, dim, 1e-5).unwrap();
    cpu_opt::rmsnorm(&out1, &weight, &mut out2, 1, dim, 1e-5).unwrap();

    // After first norm, RMS should be ~1 (with unit weights).
    // After second norm, output should change because norm(norm(x)) ≠ norm(x)
    // unless x is already unit-rms. For non-uniform x, it won't be.
    let diff: f32 = out1.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();
    // For [1,2,3,4], first norm is non-uniform, second norm changes ratios → not idempotent
    assert!(diff > 1e-5, "rmsnorm should not be idempotent on non-uniform input");
}

// ────────────────────────────────────────────────────────────────
// 10. Layernorm no-bias matches layernorm with zero bias
// ────────────────────────────────────────────────────────────────

#[test]
fn layernorm_no_bias_matches_zero_bias() {
    let dim = 4;
    let rows = 2;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0f32; dim];
    let zero_bias = vec![0.0f32; dim];
    let eps = 1e-5;

    let mut out_bias = vec![0.0f32; rows * dim];
    let mut out_nobias = vec![0.0f32; rows * dim];

    cpu_opt::layernorm(&input, &weight, &zero_bias, &mut out_bias, rows, dim, eps).unwrap();
    cpu_opt::layernorm_no_bias(&input, &weight, &mut out_nobias, rows, dim, eps).unwrap();

    assert_close(&out_bias, &out_nobias, 1e-6, "layernorm(bias=0) == layernorm_no_bias");
}

// ────────────────────────────────────────────────────────────────
// 11. Error handling in pipeline
// ────────────────────────────────────────────────────────────────

#[test]
fn matmul_dimension_mismatch_returns_error() {
    let mut output = vec![0.0f32; 4];
    // A is 1×4, B is 3×4 → k mismatch (4 ≠ 3)
    let result = cpu_opt::parallel_matmul(&[1.0; 4], &[1.0; 12], &mut output, 1, 4, 3, 1);
    assert!(result.is_err(), "should error on dimension mismatch");
}

#[test]
fn rmsnorm_weight_size_mismatch_returns_error() {
    let input = vec![1.0f32; 8];
    let weight = vec![1.0f32; 3]; // wrong size (should be 4)
    let mut output = vec![0.0f32; 8];
    let result = cpu_opt::rmsnorm(&input, &weight, &mut output, 2, 4, 1e-5);
    assert!(result.is_err(), "should error on weight size mismatch");
}

// ────────────────────────────────────────────────────────────────
// 12. Residual connection pattern
// ────────────────────────────────────────────────────────────────

#[test]
fn residual_add_after_ffn_block() {
    // Standard transformer pattern: output = norm(input + FFN(input))
    let dim = 4;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights = identity_weights(dim);
    let mut ffn_out = vec![0.0f32; dim];

    // FFN with identity weights → output = SiLU(input)
    cpu_opt::parallel_matmul(&input, &weights, &mut ffn_out, 1, dim, dim, 1).unwrap();
    cpu_opt::silu_in_place(&mut ffn_out);

    // Residual add: input + FFN(input)
    let residual: Vec<f32> = input.iter().zip(&ffn_out).map(|(a, b)| a + b).collect();

    // Normalize
    let norm_weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];
    cpu_opt::rmsnorm(&residual, &norm_weight, &mut output, 1, dim, 1e-5).unwrap();

    assert!(output.iter().all(|v| v.is_finite()), "residual + norm should be finite");
    // Residual should be larger than FFN alone (input + SiLU(input) > SiLU(input))
    assert!(
        residual.iter().zip(&ffn_out).all(|(r, f)| r.abs() >= f.abs()),
        "residual should increase magnitude"
    );
}
