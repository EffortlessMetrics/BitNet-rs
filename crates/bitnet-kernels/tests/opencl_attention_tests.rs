//! Tests for the OpenCL scaled dot-product attention kernel.
//!
//! These tests verify correctness via a CPU reference implementation that
//! mirrors the kernel logic. Hardware-dependent tests are `#[ignore]`.

// ---- CPU reference implementation ----

/// Compute attention scores: QK^T / sqrt(d_k) with optional causal masking.
fn ref_attention_scores(
    q: &[f32],
    k: &[f32],
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    causal: bool,
) -> Vec<f32> {
    let inv_sqrt_dk = 1.0 / (d_k as f32).sqrt();
    let mut scores = vec![0.0f32; num_heads * seq_q * seq_k];

    for h in 0..num_heads {
        for qi in 0..seq_q {
            for ki in 0..seq_k {
                let out_idx = h * seq_q * seq_k + qi * seq_k + ki;
                if causal && ki > qi {
                    scores[out_idx] = -1e9;
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..d_k {
                    dot += q[h * seq_q * d_k + qi * d_k + d] * k[h * seq_k * d_k + ki * d_k + d];
                }
                scores[out_idx] = dot * inv_sqrt_dk;
            }
        }
    }
    scores
}

/// Row-wise softmax: numerically stable exp(x - max) / sum.
fn ref_softmax_rows(data: &mut [f32], num_rows: usize, row_len: usize) {
    for r in 0..num_rows {
        let start = r * row_len;
        let end = start + row_len;
        let row = &mut data[start..end];

        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / (sum + 1e-8);
        for v in row.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Weighted sum: output[q][d] = sum_k weights[q][k] * V[k][d].
fn ref_weighted_sum(
    weights: &[f32],
    v: &[f32],
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * seq_q * d_v];
    for h in 0..num_heads {
        for qi in 0..seq_q {
            for di in 0..d_v {
                let mut sum = 0.0f32;
                for ki in 0..seq_k {
                    sum += weights[h * seq_q * seq_k + qi * seq_k + ki]
                        * v[h * seq_k * d_v + ki * d_v + di];
                }
                output[h * seq_q * d_v + qi * d_v + di] = sum;
            }
        }
    }
    output
}

/// Full reference attention pipeline: scores → softmax → weighted_sum.
fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    causal: bool,
) -> Vec<f32> {
    let mut scores = ref_attention_scores(q, k, num_heads, seq_q, seq_k, d_k, causal);
    ref_softmax_rows(&mut scores, num_heads * seq_q, seq_k);
    ref_weighted_sum(&scores, v, num_heads, seq_q, seq_k, d_v)
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_approx_eq_slice(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(approx_eq(x, y, eps), "mismatch at index {i}: {x} vs {y} (eps={eps})");
    }
}

// ---- Kernel source validation ----

#[test]
fn attention_kernel_source_is_not_empty() {
    let src = bitnet_kernels::kernels::ATTENTION_SRC;
    assert!(!src.is_empty());
}

#[test]
fn attention_kernel_has_three_kernels() {
    let src = bitnet_kernels::kernels::ATTENTION_SRC;
    assert!(src.contains("__kernel void attention_scores"));
    assert!(src.contains("__kernel void attention_softmax"));
    assert!(src.contains("__kernel void attention_weighted_sum"));
}

#[test]
fn attention_scores_kernel_has_causal_param() {
    let src = bitnet_kernels::kernels::ATTENTION_SRC;
    assert!(
        src.contains("const int causal"),
        "attention_scores should accept a causal masking flag"
    );
}

#[test]
fn attention_scores_kernel_uses_float4_vectorization() {
    let src = bitnet_kernels::kernels::ATTENTION_SRC;
    assert!(src.contains("vload4"), "should use float4 vectorized loads");
}

#[test]
fn attention_softmax_kernel_uses_local_memory() {
    let src = bitnet_kernels::kernels::ATTENTION_SRC;
    assert!(src.contains("__local float*"), "softmax should use local memory for reduction");
    assert!(
        src.contains("CLK_LOCAL_MEM_FENCE"),
        "softmax should synchronize with local memory barrier"
    );
}

// ---- Attention scores tests ----

#[test]
fn test_attention_scores_identity_keys() {
    // Q = K = identity-like: score[i][i] should be highest
    let d_k = 4;
    let seq = 3;
    let num_heads = 1;

    // Each row is a one-hot-ish vector
    #[rustfmt::skip]
    let q = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ];
    let k = q.clone();

    let scores = ref_attention_scores(&q, &k, num_heads, seq, seq, d_k, false);

    // Diagonal should be inv_sqrt(4) = 0.5, off-diagonal 0.0
    let inv_sqrt = 1.0 / (d_k as f32).sqrt();
    for qi in 0..seq {
        for ki in 0..seq {
            let s = scores[qi * seq + ki];
            if qi == ki {
                assert!(approx_eq(s, inv_sqrt, 1e-6), "diagonal should be {inv_sqrt}, got {s}");
            } else {
                assert!(approx_eq(s, 0.0, 1e-6), "off-diagonal should be 0, got {s}");
            }
        }
    }
}

#[test]
fn test_attention_scores_scaling() {
    let d_k = 16;
    let num_heads = 1;
    let seq = 2;

    // All ones: dot product = d_k, scaled = d_k / sqrt(d_k) = sqrt(d_k)
    let q = vec![1.0f32; num_heads * seq * d_k];
    let k = vec![1.0f32; num_heads * seq * d_k];

    let scores = ref_attention_scores(&q, &k, num_heads, seq, seq, d_k, false);
    let expected = (d_k as f32).sqrt();

    for &s in &scores {
        assert!(approx_eq(s, expected, 1e-5), "expected {expected}, got {s}");
    }
}

#[test]
fn test_causal_masking() {
    let d_k = 2;
    let seq = 4;
    let num_heads = 1;

    let q = vec![1.0f32; num_heads * seq * d_k];
    let k = vec![1.0f32; num_heads * seq * d_k];

    let scores = ref_attention_scores(&q, &k, num_heads, seq, seq, d_k, true);

    for qi in 0..seq {
        for ki in 0..seq {
            let s = scores[qi * seq + ki];
            if ki > qi {
                assert!(s <= -1e8, "future position [{qi}][{ki}] should be masked, got {s}");
            } else {
                assert!(
                    s > -1e8,
                    "past/present position [{qi}][{ki}] should NOT be masked, got {s}"
                );
            }
        }
    }
}

#[test]
fn test_causal_mask_first_row_only_first_key() {
    let d_k = 2;
    let seq = 3;
    let q = vec![1.0f32; seq * d_k];
    let k = vec![1.0f32; seq * d_k];

    let scores = ref_attention_scores(&q, &k, 1, seq, seq, d_k, true);

    // Row 0: only key 0 is valid
    assert!(scores[0 * seq + 0] > -1e8);
    assert!(scores[0 * seq + 1] <= -1e8);
    assert!(scores[0 * seq + 2] <= -1e8);
}

// ---- Softmax tests ----

#[test]
fn test_softmax_sums_to_one() {
    let row_len = 5;
    let num_rows = 3;
    let mut data: Vec<f32> = (0..num_rows * row_len).map(|i| i as f32 * 0.3 - 1.0).collect();

    ref_softmax_rows(&mut data, num_rows, row_len);

    for r in 0..num_rows {
        let start = r * row_len;
        let sum: f32 = data[start..start + row_len].iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "row {r} softmax sum = {sum}, expected ~1.0");
    }
}

#[test]
fn test_softmax_non_negative() {
    let row_len = 8;
    let mut data: Vec<f32> = (0..row_len).map(|i| (i as f32) - 4.0).collect();
    ref_softmax_rows(&mut data, 1, row_len);

    for (i, &v) in data.iter().enumerate() {
        assert!(v >= 0.0, "softmax[{i}] = {v} should be non-negative");
    }
}

#[test]
fn test_softmax_numerical_stability_large_values() {
    // If not numerically stable, exp(1000) would overflow
    let row_len = 4;
    let mut data = vec![1000.0f32, 1001.0, 999.0, 1000.5];

    ref_softmax_rows(&mut data, 1, row_len);

    let sum: f32 = data.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5), "softmax sum = {sum}");

    // Largest input (1001.0) should get largest probability
    assert!(
        data[1] > data[0] && data[1] > data[2] && data[1] > data[3],
        "index 1 (input=1001) should have highest prob: {data:?}"
    );
}

#[test]
fn test_softmax_uniform_input() {
    let n = 6;
    let mut data = vec![3.0f32; n];
    ref_softmax_rows(&mut data, 1, n);

    let expected = 1.0 / n as f32;
    for (i, &v) in data.iter().enumerate() {
        assert!(approx_eq(v, expected, 1e-5), "uniform softmax[{i}] = {v}, expected {expected}");
    }
}

#[test]
fn test_softmax_single_element() {
    let mut data = vec![42.0f32];
    ref_softmax_rows(&mut data, 1, 1);
    assert!(approx_eq(data[0], 1.0, 1e-5), "single-element softmax should be 1.0");
}

// ---- Weighted sum tests ----

#[test]
fn test_weighted_sum_identity_weights() {
    // Identity-like weights: weight[q][k] = 1 iff k==q
    let seq = 3;
    let d_v = 2;
    let num_heads = 1;

    #[rustfmt::skip]
    let weights = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let v = vec![
        10.0, 20.0,
        30.0, 40.0,
        50.0, 60.0,
    ];

    let output = ref_weighted_sum(&weights, &v, num_heads, seq, seq, d_v);

    // Output should equal V since weights are identity
    assert_approx_eq_slice(&output, &v, 1e-6);
}

#[test]
fn test_weighted_sum_uniform_weights() {
    let seq = 3;
    let d_v = 2;

    // Uniform weights: each row is [1/3, 1/3, 1/3]
    let w = 1.0 / 3.0;
    let weights = vec![w; seq * seq];
    #[rustfmt::skip]
    let v = vec![
        3.0, 6.0,
        6.0, 12.0,
        9.0, 18.0,
    ];

    let output = ref_weighted_sum(&weights, &v, 1, seq, seq, d_v);

    // Each output row should be average of V rows: (3+6+9)/3=6, (6+12+18)/3=12
    for qi in 0..seq {
        assert!(approx_eq(output[qi * d_v], 6.0, 1e-4));
        assert!(approx_eq(output[qi * d_v + 1], 12.0, 1e-4));
    }
}

// ---- Full pipeline tests ----

#[test]
fn test_full_attention_pipeline_non_causal() {
    let num_heads = 1;
    let seq = 2;
    let d_k = 4;
    let d_v = 4;

    let q = vec![1.0f32; num_heads * seq * d_k];
    let k = vec![1.0f32; num_heads * seq * d_k];
    let v = vec![1.0f32; num_heads * seq * d_v];

    let output = ref_attention(&q, &k, &v, num_heads, seq, seq, d_k, d_v, false);

    // With all-ones Q, K, V and non-causal: all outputs should be 1.0
    for (i, &val) in output.iter().enumerate() {
        assert!(approx_eq(val, 1.0, 1e-4), "output[{i}] = {val}, expected 1.0");
    }
}

#[test]
fn test_full_attention_pipeline_causal() {
    let num_heads = 1;
    let seq = 3;
    let d_k = 2;
    let d_v = 2;

    #[rustfmt::skip]
    let q = vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    ];
    #[rustfmt::skip]
    let k = vec![
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    ];
    #[rustfmt::skip]
    let v = vec![
        1.0, 0.0,
        0.0, 1.0,
        0.5, 0.5,
    ];

    let output = ref_attention(&q, &k, &v, num_heads, seq, seq, d_k, d_v, true);

    // Row 0: only sees V[0] → output = [1.0, 0.0]
    assert!(approx_eq(output[0], 1.0, 1e-4));
    assert!(approx_eq(output[1], 0.0, 1e-4));

    // Row 1: sees V[0] and V[1] equally → output = [0.5, 0.5]
    assert!(approx_eq(output[2], 0.5, 1e-4));
    assert!(approx_eq(output[3], 0.5, 1e-4));
}

// ---- Multi-head tests ----

#[test]
fn test_multi_head_dimensions() {
    let num_heads = 4;
    let seq = 3;
    let d_k = 8;
    let d_v = 8;

    let q = vec![0.1f32; num_heads * seq * d_k];
    let k = vec![0.1f32; num_heads * seq * d_k];
    let v = vec![0.5f32; num_heads * seq * d_v];

    let output = ref_attention(&q, &k, &v, num_heads, seq, seq, d_k, d_v, false);

    assert_eq!(output.len(), num_heads * seq * d_v);

    // All heads see same data so outputs should be identical across heads
    let head_size = seq * d_v;
    let head0 = &output[..head_size];
    for h in 1..num_heads {
        let head_h = &output[h * head_size..(h + 1) * head_size];
        assert_approx_eq_slice(head0, head_h, 1e-5);
    }
}

#[test]
fn test_multi_head_independent() {
    // Different Q per head should produce different outputs
    let num_heads = 2;
    let seq = 2;
    let d_k = 2;
    let d_v = 2;

    #[rustfmt::skip]
    let q = vec![
        // Head 0
        1.0, 0.0,
        1.0, 0.0,
        // Head 1
        0.0, 1.0,
        0.0, 1.0,
    ];
    #[rustfmt::skip]
    let k = vec![
        // Head 0
        1.0, 0.0,
        0.0, 1.0,
        // Head 1
        1.0, 0.0,
        0.0, 1.0,
    ];
    #[rustfmt::skip]
    let v = vec![
        // Head 0
        10.0, 0.0,
         0.0, 10.0,
        // Head 1
        10.0, 0.0,
         0.0, 10.0,
    ];

    let output = ref_attention(&q, &k, &v, num_heads, seq, seq, d_k, d_v, false);

    // Head 0: Q aligns with K[0] → more weight on V[0]
    // Head 1: Q aligns with K[1] → more weight on V[1]
    let head0_out = &output[..seq * d_v];
    let head1_out = &output[seq * d_v..];

    // Head 0 row 0: should lean toward [10, 0]
    assert!(head0_out[0] > head0_out[1], "head0 should favor V[0] dim 0");

    // Head 1 row 0: should lean toward [0, 10]
    assert!(head1_out[1] > head1_out[0], "head1 should favor V[1] dim 1");
}

// ---- Edge cases ----

#[test]
fn test_seq_len_one() {
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![5.0, 6.0];

    let output = ref_attention(&q, &k, &v, 1, 1, 1, 4, 2, false);

    // Single element: softmax of single score = 1.0, output = V
    assert_approx_eq_slice(&output, &v, 1e-5);
}

#[test]
fn test_d_k_one() {
    let seq = 3;
    let q = vec![2.0, 3.0, 4.0];
    let k = vec![1.0, 1.0, 1.0];
    let v = vec![10.0, 20.0, 30.0];

    let scores = ref_attention_scores(&q, &k, 1, seq, seq, 1, false);

    // d_k=1, inv_sqrt=1.0, so scores = q[i] * k[j] = q[i]
    for qi in 0..seq {
        for ki in 0..seq {
            assert!(
                approx_eq(scores[qi * seq + ki], q[qi], 1e-6),
                "score[{qi}][{ki}] should be q[{qi}]={}",
                q[qi]
            );
        }
    }
}

#[test]
fn test_large_seq_len() {
    let seq = 256;
    let d_k = 4;
    let d_v = 4;

    let q = vec![0.01f32; seq * d_k];
    let k = vec![0.01f32; seq * d_k];
    let v = vec![1.0f32; seq * d_v];

    let output = ref_attention(&q, &k, &v, 1, seq, seq, d_k, d_v, false);

    // All values equal → uniform softmax → output = V average = [1.0, ...]
    for (i, &val) in output.iter().enumerate() {
        assert!(approx_eq(val, 1.0, 1e-3), "output[{i}] = {val}, expected ~1.0");
    }
}

#[test]
fn test_asymmetric_seq_lengths() {
    // seq_q != seq_k (cross-attention scenario)
    let seq_q = 2;
    let seq_k = 4;
    let d_k = 3;
    let d_v = 3;

    let q = vec![1.0f32; seq_q * d_k];
    let k = vec![1.0f32; seq_k * d_k];
    let v = vec![2.0f32; seq_k * d_v];

    let output = ref_attention(&q, &k, &v, 1, seq_q, seq_k, d_k, d_v, false);

    assert_eq!(output.len(), seq_q * d_v);
    // All-ones Q/K, all-twos V → output = 2.0
    for (i, &val) in output.iter().enumerate() {
        assert!(approx_eq(val, 2.0, 1e-4), "output[{i}] = {val}");
    }
}

// ---- Property-like tests ----

#[test]
fn test_softmax_preserves_ordering() {
    let mut data = vec![1.0f32, 3.0, 2.0, 5.0, 4.0];
    let original = data.clone();
    ref_softmax_rows(&mut data, 1, 5);

    // Softmax should preserve relative ordering
    for i in 0..data.len() {
        for j in (i + 1)..data.len() {
            if original[i] > original[j] {
                assert!(
                    data[i] > data[j],
                    "ordering violated: softmax[{i}]={} <= softmax[{j}]={}",
                    data[i],
                    data[j]
                );
            }
        }
    }
}

#[test]
fn test_softmax_many_rows_all_sum_to_one() {
    let row_len = 10;
    let num_rows = 20;
    let mut data: Vec<f32> =
        (0..num_rows * row_len).map(|i| ((i * 7 + 3) % 100) as f32 / 10.0 - 5.0).collect();

    ref_softmax_rows(&mut data, num_rows, row_len);

    for r in 0..num_rows {
        let sum: f32 = data[r * row_len..(r + 1) * row_len].iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "row {r}: softmax sum = {sum}");
    }
}

#[test]
fn test_attention_output_bounded() {
    // Output should be bounded by min/max of V values
    let seq = 4;
    let d_k = 3;
    let d_v = 3;

    let q: Vec<f32> = (0..seq * d_k).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq * d_k).map(|i| (i as f32) * 0.2).collect();
    #[rustfmt::skip]
    let v: Vec<f32> = (0..seq * d_v).map(|i| (i as f32) * 0.5 + 1.0).collect();

    let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let output = ref_attention(&q, &k, &v, 1, seq, seq, d_k, d_v, false);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            val >= v_min - 1e-4 && val <= v_max + 1e-4,
            "output[{i}] = {val} outside V range [{v_min}, {v_max}]"
        );
    }
}

// ---- Hardware-dependent tests (require OpenCL runtime) ----

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_attention_scores_on_device() {
    // Placeholder: compile attention.cl, run attention_scores kernel, compare to ref
    todo!("Implement OpenCL device test for attention_scores");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_attention_softmax_on_device() {
    todo!("Implement OpenCL device test for attention_softmax");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_attention_weighted_sum_on_device() {
    todo!("Implement OpenCL device test for attention_weighted_sum");
}

#[test]
#[ignore = "requires OpenCL device - run with --ignored on Intel Arc hardware"]
fn test_opencl_full_attention_pipeline_on_device() {
    todo!("Implement OpenCL device test for full attention pipeline");
}
