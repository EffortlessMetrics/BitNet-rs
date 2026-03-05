#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    unused_imports,
    dead_code
)]

const TOLERANCE: f32 = 1e-4;

/// Basic scaled dot-product attention implementation
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0; seq_len * seq_len];

    // Compute Q @ K^T
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Apply softmax per row
    for i in 0..seq_len {
        softmax(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }

    // Compute attention @ V
    let mut output = vec![0.0; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut sum = 0.0;
            for j in 0..seq_len {
                sum += scores[i * seq_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }

    output
}

/// In-place softmax computation
fn softmax(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let mut sum = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

/// Apply causal mask (upper triangle becomes -inf)
fn apply_causal_mask(scores: &mut [f32], seq_len: usize) {
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Basic dot product
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

mod basic_attention {
    use super::*;

    #[test]
    fn test_attention_single_head() {
        let seq_len = 4;
        let head_dim = 8;

        let q = vec![1.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v = vec![2.0; seq_len * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        // With uniform Q, K, V, output should be uniform (all 2.0)
        assert_eq!(output.len(), seq_len * head_dim);
        for &val in &output {
            assert!((val - 2.0).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_attention_scaling() {
        let seq_len = 2;
        let head_dim = 4;

        let q = vec![1.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v = vec![1.0; seq_len * head_dim];

        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale_factor);

        // Check output shape and basic properties
        assert_eq!(output.len(), seq_len * head_dim);
        for &val in &output {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_attention_output_shape() {
        let seq_len = 8;
        let head_dim = 16;

        let q = vec![0.5; seq_len * head_dim];
        let k = vec![0.5; seq_len * head_dim];
        let v = vec![0.5; seq_len * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        assert_eq!(output.len(), seq_len * head_dim);
    }

    #[test]
    fn test_attention_uniform_scores() {
        let seq_len = 4;
        let head_dim = 8;

        let q = vec![1.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) + 1.0).collect();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        // With uniform attention weights (1/seq_len each),
        // output should be average of V across sequence
        let mut expected = vec![0.0; head_dim];
        for d in 0..head_dim {
            for i in 0..seq_len {
                expected[d] += v[i * head_dim + d];
            }
            expected[d] /= seq_len as f32;
        }

        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!(
                    (output[i * head_dim + d] - expected[d]).abs() < TOLERANCE,
                    "Mismatch at position ({}, {}): {} vs {}",
                    i,
                    d,
                    output[i * head_dim + d],
                    expected[d]
                );
            }
        }
    }

    #[test]
    fn test_attention_identity_key_value() {
        let seq_len = 3;
        let head_dim = 4;

        // Identity K and V means output should be close to V
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        // Output should approximate input
        assert_eq!(output.len(), seq_len * head_dim);
        for &val in &output {
            assert!(val.is_finite());
        }
    }
}

mod causal_masking {
    use super::*;

    #[test]
    fn test_causal_mask_triangular() {
        let seq_len = 4;
        let mut scores = vec![1.0; seq_len * seq_len];

        apply_causal_mask(&mut scores, seq_len);

        // Upper triangle should be -inf
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    assert!(scores[i * seq_len + j].is_infinite());
                    assert!(scores[i * seq_len + j] < 0.0);
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_first_token() {
        let seq_len = 5;
        let mut scores = vec![1.0; seq_len * seq_len];

        apply_causal_mask(&mut scores, seq_len);

        // First token (i=0) should only attend to itself
        assert_eq!(scores[(0 * seq_len)], 1.0);
        for j in 1..seq_len {
            assert!(scores[0 * seq_len + j].is_infinite());
        }
    }

    #[test]
    fn test_causal_mask_last_token() {
        let seq_len = 5;
        let mut scores = vec![1.0; seq_len * seq_len];

        apply_causal_mask(&mut scores, seq_len);

        // Last token (i=seq_len-1) should attend to all tokens
        for j in 0..seq_len {
            assert_eq!(scores[(seq_len - 1) * seq_len + j], 1.0);
        }
    }

    #[test]
    fn test_causal_mask_preserves_lower() {
        let seq_len = 4;
        let mut scores = (0..seq_len * seq_len).map(|i| (i as f32) + 1.0).collect::<Vec<_>>();
        let original = scores.clone();

        apply_causal_mask(&mut scores, seq_len);

        // Lower triangle and diagonal should be unchanged
        for i in 0..seq_len {
            for j in 0..=i {
                assert_eq!(scores[i * seq_len + j], original[i * seq_len + j]);
            }
        }
    }
}

mod numerical_stability {
    use super::*;

    #[test]
    fn test_attention_softmax_stability() {
        let seq_len = 8;
        let head_dim = 16;

        // Large Q and K values should not overflow
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| 100.0 + (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| 100.0 - (i as f32) * 0.1).collect();
        let v = vec![1.0; seq_len * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        // All outputs should be finite
        for &val in &output {
            assert!(val.is_finite(), "Got NaN or inf: {}", val);
            assert!(val >= 0.0, "Got negative value: {}", val);
        }
    }

    #[test]
    fn test_attention_zero_queries() {
        let seq_len = 4;
        let head_dim = 8;

        let q = vec![0.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) + 1.0).collect();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        // Zero Q should produce uniform attention
        let mut expected = vec![0.0; head_dim];
        for d in 0..head_dim {
            for i in 0..seq_len {
                expected[d] += v[i * head_dim + d];
            }
            expected[d] /= seq_len as f32;
        }

        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!(
                    (output[i * head_dim + d] - expected[d]).abs() < TOLERANCE,
                    "Zero Q test failed at ({}, {}): {} vs {}",
                    i,
                    d,
                    output[i * head_dim + d],
                    expected[d]
                );
            }
        }
    }

    #[test]
    fn test_attention_small_head_dim() {
        let seq_len = 4;
        let head_dim = 2;

        let q = vec![1.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v = vec![2.0; seq_len * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        assert_eq!(output.len(), seq_len * head_dim);
        for &val in &output {
            assert!(val.is_finite());
            assert!((val - 2.0).abs() < TOLERANCE);
        }
    }
}

mod bitnet_config {
    use super::*;

    #[test]
    fn test_attention_head_dim_64() {
        let seq_len = 4;
        let head_dim = 64; // Common for 2B models

        let q = vec![0.1; seq_len * head_dim];
        let k = vec![0.1; seq_len * head_dim];
        let v = vec![0.5; seq_len * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = reference_attention(&q, &k, &v, seq_len, head_dim, scale);

        assert_eq!(output.len(), seq_len * head_dim);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_multi_head_independence() {
        let seq_len = 4;
        let head_dim = 8;
        let num_heads = 2;

        let q = vec![1.0; seq_len * num_heads * head_dim];
        let k = vec![1.0; seq_len * num_heads * head_dim];
        let v = vec![2.0; seq_len * num_heads * head_dim];

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute each head separately
        let mut outputs = Vec::new();
        for head in 0..num_heads {
            let q_head = (0..seq_len * head_dim)
                .map(|i| q[head * seq_len * head_dim + i])
                .collect::<Vec<_>>();
            let k_head = (0..seq_len * head_dim)
                .map(|i| k[head * seq_len * head_dim + i])
                .collect::<Vec<_>>();
            let v_head = (0..seq_len * head_dim)
                .map(|i| v[head * seq_len * head_dim + i])
                .collect::<Vec<_>>();

            outputs.push(reference_attention(&q_head, &k_head, &v_head, seq_len, head_dim, scale));
        }

        // All heads should produce identical output for identical inputs
        for i in 1..num_heads {
            for j in 0..seq_len * head_dim {
                assert!(
                    (outputs[i][j] - outputs[0][j]).abs() < TOLERANCE,
                    "Head {} differs from head 0 at position {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_attention_kv_cache_append() {
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // First, compute attention with seq_len=2
        let q1 = vec![1.0; 2 * head_dim];
        let k1 = vec![1.0; 2 * head_dim];
        let v1 = vec![2.0; 2 * head_dim];

        let output1 = reference_attention(&q1, &k1, &v1, 2, head_dim, scale);

        // Then append new KV pair (simulating cache append)
        let q2 = vec![1.0; 3 * head_dim];
        let k2 = vec![1.0; 3 * head_dim];
        let v2 = vec![2.0; 3 * head_dim];

        let output2 = reference_attention(&q2, &k2, &v2, 3, head_dim, scale);

        // Both outputs should be finite and well-formed
        assert_eq!(output1.len(), 2 * head_dim);
        assert_eq!(output2.len(), 3 * head_dim);

        for &val in &output1 {
            assert!(val.is_finite());
        }
        for &val in &output2 {
            assert!(val.is_finite());
        }
    }
}
