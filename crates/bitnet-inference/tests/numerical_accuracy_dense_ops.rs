//! Numerical accuracy tests for dense inference operations.
//!
//! Validates correctness of CPU ops against known mathematical
//! reference values computed by hand or by PyTorch-equivalent formulas.

use bitnet_inference::cpu_opt;

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: expected {e:.8}, got {a:.8} (diff={:.2e}, tol={tol})",
            (a - e).abs(),
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 1. SiLU known reference values
// ────────────────────────────────────────────────────────────────

#[test]
fn silu_known_values() {
    // Reference: torch.nn.functional.silu(torch.tensor([...]))
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let input = vec![-3.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let result = cpu_opt::silu(&input);

    // Hand-computed references:
    // SiLU(-3) = -3 * sigmoid(-3) = -3 * 0.04743 = -0.14228
    // SiLU(-2) = -2 * sigmoid(-2) = -2 * 0.11920 = -0.23840
    // SiLU(-1) = -1 * sigmoid(-1) = -1 * 0.26894 = -0.26894
    // SiLU(-0.5) = -0.5 * sigmoid(-0.5) = -0.5 * 0.37754 = -0.18877
    // SiLU(0) = 0
    // SiLU(0.5) = 0.5 * sigmoid(0.5) = 0.5 * 0.62246 = 0.31123
    // SiLU(1) = 1 * sigmoid(1) = 0.73106
    // SiLU(2) = 2 * sigmoid(2) = 2 * 0.88080 = 1.76160
    // SiLU(3) = 3 * sigmoid(3) = 3 * 0.95257 = 2.85772
    let expected =
        vec![-0.14228, -0.23840, -0.26894, -0.18877, 0.0, 0.31123, 0.73106, 1.76160, 2.85772];

    assert_close(&result, &expected, 1e-3, "silu reference values");
}

#[test]
fn silu_monotonicity_for_positive() {
    // SiLU is monotonically increasing for x > 0
    let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let result = cpu_opt::silu(&input);
    for i in 1..result.len() {
        assert!(
            result[i] >= result[i - 1],
            "silu should be monotonic for positive x: silu({:.1}) = {} < silu({:.1}) = {}",
            input[i],
            result[i],
            input[i - 1],
            result[i - 1],
        );
    }
}

#[test]
fn silu_approaches_identity_for_large_x() {
    // For large positive x, SiLU(x) ≈ x (sigmoid → 1)
    for x in [10.0f32, 20.0, 50.0, 100.0] {
        let result = cpu_opt::silu(&[x]);
        let ratio = result[0] / x;
        assert!((ratio - 1.0).abs() < 0.001, "silu({x}) / {x} = {ratio}, should be ~1.0",);
    }
}

#[test]
fn silu_approaches_zero_for_large_negative_x() {
    // For large negative x, SiLU(x) ≈ 0 (sigmoid → 0)
    for x in [-10.0f32, -20.0, -50.0] {
        let result = cpu_opt::silu(&[x]);
        assert!(result[0].abs() < 0.01, "silu({x}) = {}, should be ~0", result[0],);
    }
}

// ────────────────────────────────────────────────────────────────
// 2. GELU known reference values
// ────────────────────────────────────────────────────────────────

#[test]
fn gelu_known_values() {
    // Reference: torch.nn.functional.gelu(torch.tensor([...]))
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let input = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let result = cpu_opt::gelu(&input);

    // PyTorch reference values (approximate GELU):
    // GELU(-2.0) ≈ -0.0454
    // GELU(-1.0) ≈ -0.1588
    // GELU(-0.5) ≈ -0.1543
    // GELU(0.0)  = 0.0
    // GELU(0.5)  ≈ 0.3457
    // GELU(1.0)  ≈ 0.8412
    // GELU(2.0)  ≈ 1.9546
    let expected = vec![-0.0454, -0.1588, -0.1543, 0.0, 0.3457, 0.8412, 1.9546];

    assert_close(&result, &expected, 0.01, "gelu reference values");
}

#[test]
fn gelu_symmetry_property() {
    // GELU is NOT symmetric: GELU(-x) ≠ -GELU(x) in general
    // But GELU(0) = 0
    let result = cpu_opt::gelu(&[0.0]);
    assert!(result[0].abs() < 1e-8, "gelu(0) should be exactly 0");

    // GELU(-x) + x ≈ GELU(x) for x → ∞? No, but GELU(x) ≈ x for large x
    let pos = cpu_opt::gelu(&[5.0])[0];
    assert!((pos - 5.0).abs() < 0.01, "gelu(5.0) should be ~5.0, got {pos}",);
}

// ────────────────────────────────────────────────────────────────
// 3. RMSNorm numerical accuracy
// ────────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_known_values() {
    // RMSNorm([1, 2, 3, 4], weight=[1,1,1,1], eps=1e-5)
    // rms = sqrt(mean([1,4,9,16]) + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.73861
    // output[i] = x[i] / rms * weight[i]
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];
    let eps = 1e-5;

    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, eps).unwrap();

    let rms = (7.5f32 + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();

    assert_close(&output, &expected, 1e-6, "rmsnorm known values");
}

#[test]
fn rmsnorm_preserves_direction() {
    // RMSNorm should preserve relative ratios between elements
    let input = vec![2.0f32, 4.0, 6.0, 8.0];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];

    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5).unwrap();

    // Ratios should be preserved: output[1]/output[0] = input[1]/input[0] = 2
    let ratio_in = input[1] / input[0];
    let ratio_out = output[1] / output[0];
    assert!(
        (ratio_in - ratio_out).abs() < 1e-6,
        "rmsnorm should preserve ratios: in={ratio_in}, out={ratio_out}",
    );
}

#[test]
fn rmsnorm_with_custom_weights() {
    // Custom weights scale individual dimensions
    let input = vec![1.0f32, 1.0, 1.0, 1.0];
    let weight = vec![2.0f32, 0.5, 1.0, 3.0];
    let mut output = vec![0.0f32; 4];

    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5).unwrap();

    // For uniform input, rms = sqrt(1 + eps) ≈ 1.0
    // output[i] = (1/rms) * weight[i] ≈ weight[i]
    for (i, (&o, &w)) in output.iter().zip(weight.iter()).enumerate() {
        assert!(
            (o - w).abs() < 0.01,
            "rmsnorm with uniform input: output[{i}]={o} should be ~weight[{i}]={w}",
        );
    }
}

#[test]
fn rmsnorm_eps_prevents_div_by_zero() {
    // All-zero input: rms = sqrt(0 + eps) = sqrt(eps)
    let input = vec![0.0f32; 4];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];

    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5).unwrap();

    // Output should be 0/sqrt(eps) * 1 = 0
    assert!(output.iter().all(|v| v.abs() < 1e-6), "rmsnorm of zero input should be ~zero",);
}

// ────────────────────────────────────────────────────────────────
// 4. LayerNorm numerical accuracy
// ────────────────────────────────────────────────────────────────

#[test]
fn layernorm_known_values() {
    // LayerNorm([1, 2, 3, 4], weight=[1,1,1,1], bias=[0,0,0,0], eps=1e-5)
    // mean = 2.5, var = 1.25, std = sqrt(1.25 + 1e-5) ≈ 1.11803
    // output[i] = (x[i] - mean) / std * weight[i] + bias[i]
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let bias = vec![0.0f32; 4];
    let mut output = vec![0.0f32; 4];
    let eps = 1e-5;

    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 4, eps).unwrap();

    let mean = 2.5;
    let var = 1.25;
    let inv_std = 1.0 / (var + eps as f64).sqrt();
    let expected: Vec<f32> = input.iter().map(|&x| ((x as f64 - mean) * inv_std) as f32).collect();

    assert_close(&output, &expected, 1e-5, "layernorm known values");
}

#[test]
fn layernorm_output_has_zero_mean_unit_var() {
    // For weight=1, bias=0: output should have mean≈0, var≈1
    let input = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    let weight = vec![1.0f32; 5];
    let bias = vec![0.0f32; 5];
    let mut output = vec![0.0f32; 5];

    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 5, 1e-5).unwrap();

    let mean: f32 = output.iter().sum::<f32>() / 5.0;
    let var: f32 = output.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 5.0;

    assert!(mean.abs() < 1e-5, "layernorm output mean should be ~0, got {mean}");
    assert!((var - 1.0).abs() < 0.01, "layernorm output variance should be ~1, got {var}");
}

#[test]
fn layernorm_bias_shifts_output() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let bias = vec![10.0f32; 4]; // shift by 10
    let mut output = vec![0.0f32; 4];

    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 4, 1e-5).unwrap();

    // Mean of output should be ~10 (bias shifts the centered distribution)
    let mean: f32 = output.iter().sum::<f32>() / 4.0;
    assert!((mean - 10.0).abs() < 0.01, "layernorm with bias=10 should have mean ~10, got {mean}");
}

// ────────────────────────────────────────────────────────────────
// 5. Matrix multiplication accuracy
// ────────────────────────────────────────────────────────────────

#[test]
fn matmul_2x2_known_values() {
    // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let mut c = vec![0.0f32; 4];

    cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 2, 2, 1).unwrap();

    assert_close(&c, &[19.0, 22.0, 43.0, 50.0], 1e-5, "2x2 matmul");
}

#[test]
fn matmul_3x2_times_2x3() {
    // [1 2]   [7  8  9]   [1*7+2*10  1*8+2*11  1*9+2*12]   [27  30  33]
    // [3 4] × [10 11 12] = [3*7+4*10  3*8+4*11  3*9+4*12] = [61  68  75]
    // [5 6]                [5*7+6*10  5*8+6*11  5*9+6*12]   [95 106 117]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
    let mut c = vec![0.0f32; 9];

    cpu_opt::parallel_matmul(&a, &b, &mut c, 3, 3, 2, 1).unwrap();

    let expected = vec![27.0, 30.0, 33.0, 61.0, 68.0, 75.0, 95.0, 106.0, 117.0f32];
    assert_close(&c, &expected, 1e-4, "3x2 × 2x3 matmul");
}

#[test]
fn matmul_with_multiple_threads() {
    // Same computation with 1 and 4 threads should give identical results
    let dim = 16;
    let a: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..dim * dim).map(|i| ((dim * dim - i) as f32) * 0.01).collect();

    let mut c1 = vec![0.0f32; dim * dim];
    let mut c4 = vec![0.0f32; dim * dim];

    cpu_opt::parallel_matmul(&a, &b, &mut c1, dim, dim, dim, 1).unwrap();
    cpu_opt::parallel_matmul(&a, &b, &mut c4, dim, dim, dim, 4).unwrap();

    assert_close(&c1, &c4, 1e-4, "matmul 1-thread vs 4-thread");
}

// ────────────────────────────────────────────────────────────────
// 6. Attention numerical accuracy
// ────────────────────────────────────────────────────────────────

#[test]
fn attention_uniform_scores() {
    // When Q and K are identical, all positions get equal attention
    let seq_len = 3;
    let head_dim = 2;
    let num_heads = 1;
    let size = num_heads * seq_len * head_dim;

    // All queries and keys are the same → uniform attention weights
    let qk = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
    let value = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; size];

    cpu_opt::parallel_attention(&qk, &qk, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // With uniform attention, output should be mean of values
    let mean_v: Vec<f32> = (0..head_dim)
        .map(|d| (0..seq_len).map(|s| value[s * head_dim + d]).sum::<f32>() / seq_len as f32)
        .collect();

    // Each position should output the mean value
    for s in 0..seq_len {
        let pos_output = &output[s * head_dim..(s + 1) * head_dim];
        assert_close(pos_output, &mean_v, 0.01, &format!("uniform attn pos {s}"));
    }
}

#[test]
fn attention_scaling_by_sqrt_head_dim() {
    // Verify that attention uses 1/√d scaling
    let seq_len = 2;
    let head_dim = 4;
    let num_heads = 1;
    let size = num_heads * seq_len * head_dim;

    // Q = [1,0,0,0], K = [1,0,0,0] → dot=1, scaled = 1/√4 = 0.5
    // Q = [1,0,0,0], K = [0,0,0,0] → dot=0, scaled = 0
    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let key = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let value = vec![10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut output = vec![0.0f32; size];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // Position 0 (Q=[1,0,0,0]) attends to:
    //   K[0]=[1,0,0,0] with score 1/√4=0.5 → softmax weight will be larger
    //   K[1]=[0,0,0,0] with score 0/√4=0   → softmax weight will be smaller
    // So output[0] should be weighted average of V[0] and V[1], biased toward V[0]
    assert!(output[0] > 5.0, "position 0 should attend more to value 0 (10), got {}", output[0],);
}

// ────────────────────────────────────────────────────────────────
// 7. Floating point edge cases
// ────────────────────────────────────────────────────────────────

#[test]
fn operations_handle_subnormal_numbers() {
    let tiny = f32::MIN_POSITIVE / 2.0; // subnormal
    let input = vec![tiny, -tiny, tiny * 2.0, 0.0];

    let silu_result = cpu_opt::silu(&input);
    assert!(silu_result.iter().all(|v| v.is_finite()), "silu handles subnormals");

    let gelu_result = cpu_opt::gelu(&input);
    assert!(gelu_result.iter().all(|v| v.is_finite()), "gelu handles subnormals");
}

#[test]
fn matmul_accumulation_precision() {
    // Test that large matrix accumulation doesn't lose too much precision
    let dim = 64;
    // All 1.0s: each output element should be exactly `dim`
    let a = vec![1.0f32; dim * dim];
    let b = vec![1.0f32; dim * dim];
    let mut c = vec![0.0f32; dim * dim];

    cpu_opt::parallel_matmul(&a, &b, &mut c, dim, dim, dim, 1).unwrap();

    let expected_val = dim as f32;
    for (i, &v) in c.iter().enumerate() {
        assert!(
            (v - expected_val).abs() < 0.01,
            "matmul accumulation: c[{i}] = {v}, expected {expected_val}",
        );
    }
}

// ────────────────────────────────────────────────────────────────
// 8. Multi-row norm consistency
// ────────────────────────────────────────────────────────────────

#[test]
fn rmsnorm_multi_row_independent() {
    // Each row should be normalized independently
    let dim = 3;
    let rows = 2;
    let input = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 3.0]; // row0=[1,0,0], row1=[0,0,3]
    let weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; rows * dim];

    cpu_opt::rmsnorm(&input, &weight, &mut output, rows, dim, 1e-5).unwrap();

    // Row 0: rms = sqrt((1+0+0)/3 + eps) ≈ sqrt(0.333)
    // Row 1: rms = sqrt((0+0+9)/3 + eps) ≈ sqrt(3.0)
    // These are different, so the rows should normalize differently
    let row0 = &output[0..3];
    let row1 = &output[3..6];

    // Row 0 should have a non-zero first element
    assert!(row0[0].abs() > 0.1, "row0[0] should be non-zero after norm");
    // Row 1 should have a non-zero last element
    assert!(row1[2].abs() > 0.1, "row1[2] should be non-zero after norm");

    // The non-zero elements have the same rms-normalized magnitude since each has
    // only one non-zero element (|1|/rms_row0 and |3|/rms_row1 can be close).
    // Instead verify that the zero elements stay zero.
    assert!(row0[1].abs() < 1e-6, "row0[1] (zero input) should stay ~0 after norm");
    assert!(row1[0].abs() < 1e-6, "row1[0] (zero input) should stay ~0 after norm");
}

#[test]
fn layernorm_multi_row_independent() {
    let dim = 4;
    let rows = 2;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let weight = vec![1.0f32; dim];
    let bias = vec![0.0f32; dim];
    let mut output = vec![0.0f32; rows * dim];

    cpu_opt::layernorm(&input, &weight, &bias, &mut output, rows, dim, 1e-5).unwrap();

    // Both rows should be independently normalized to mean≈0, var≈1
    for row in 0..rows {
        let row_data = &output[row * dim..(row + 1) * dim];
        let mean: f32 = row_data.iter().sum::<f32>() / dim as f32;
        let var: f32 = row_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;

        assert!(mean.abs() < 1e-5, "row {row} mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.01, "row {row} variance should be ~1, got {var}");
    }

    // Despite different magnitudes (1-4 vs 10-40), normalized patterns should be identical
    let row0 = &output[0..4];
    let row1 = &output[4..8];
    assert_close(row0, row1, 1e-5, "proportionally scaled rows should normalize identically");
}
