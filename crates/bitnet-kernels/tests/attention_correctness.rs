#![allow(clippy::manual_range_contains, clippy::approx_constant)]
//! Attention kernel correctness regression tests (CPU path).
//!
//! Verifies numerical correctness of scaled dot-product attention,
//! softmax normalization, causal masking, multi-head attention, GQA,
//! KV cache operations, and edge cases using hand-computed expected
//! values.  Deterministic, no model files required.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, CpuAttention, CpuAttentionConfig, GqaConfig,
    attention_with_kv_cache, causal_attention, causal_mask, multi_head_attention_cpu,
    scaled_dot_product_attention,
};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_slice,
};

const EPS: f32 = 1e-4;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
}

/// Reference softmax for verification.
fn ref_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 { vec![0.0; row.len()] } else { exps.iter().map(|&e| e / sum).collect() }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Scaled dot-product attention: Q·K^T / √d_k
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sdpa_identity_query_key_known_output() {
    // Q = K = identity-like vectors; head_dim=2, seq=2.
    // Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[10,20],[30,40]]
    // scores = Q·K^T = [[1,0],[0,1]], scaled by 1/√2 ≈ 0.7071
    // softmax([[0.7071, 0], [0, 0.7071]]) per row:
    //   row0: [exp(0.7071), exp(0)] / Z ≈ [0.6681, 0.3319]
    //   row1: [exp(0), exp(0.7071)] / Z ≈ [0.3319, 0.6681]
    // output row0 = 0.6681*[10,20] + 0.3319*[30,40] = [16.64, 19.98] ... wait
    // Let me compute carefully.
    let head_dim = 2;
    let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 queries
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
    let v = vec![10.0, 20.0, 30.0, 40.0]; // 2 values

    let result = scaled_dot_product_attention(&q, &k, &v, 2, 2, head_dim, false).unwrap();

    // scale = 1/√2 ≈ 0.7071
    let scale = 1.0 / (2.0_f32).sqrt();
    // scores = [[1*scale, 0], [0, 1*scale]]
    let row0 = ref_softmax(&[1.0 * scale, 0.0 * scale]);
    let row1 = ref_softmax(&[0.0 * scale, 1.0 * scale]);

    // output[0] = row0[0]*v[0] + row0[1]*v[1]
    let expected_00 = row0[0] * 10.0 + row0[1] * 30.0;
    let expected_01 = row0[0] * 20.0 + row0[1] * 40.0;
    let expected_10 = row1[0] * 10.0 + row1[1] * 30.0;
    let expected_11 = row1[0] * 20.0 + row1[1] * 40.0;

    assert!(approx_eq(result[0], expected_00), "got {} want {}", result[0], expected_00);
    assert!(approx_eq(result[1], expected_01), "got {} want {}", result[1], expected_01);
    assert!(approx_eq(result[2], expected_10), "got {} want {}", result[2], expected_10);
    assert!(approx_eq(result[3], expected_11), "got {} want {}", result[3], expected_11);
}

#[test]
fn sdpa_explicit_scale_overrides_default() {
    let head_dim = 4;
    let q = vec![1.0; head_dim];
    let k = vec![1.0; head_dim];
    let v = vec![5.0; head_dim];
    let custom_scale = 0.25;

    let result =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, custom_scale, 1, 1, head_dim)
            .unwrap();

    // Single KV pair → softmax of single element = 1.0 → output = v
    for (i, &val) in result.iter().enumerate() {
        assert!(approx_eq(val, 5.0), "index {i}: got {val} want 5.0");
    }
}

#[test]
fn sdpa_cross_attention_different_seq_lengths() {
    // seq_q=1 query attending to seq_k=3 keys
    let head_dim = 2;
    let q = vec![1.0, 0.0]; // 1 query
    let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 keys
    let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3 values

    let result = scaled_dot_product_attention(&q, &k, &v, 1, 3, head_dim, false).unwrap();
    assert_eq!(result.len(), head_dim);

    // Compute reference: scores = [q·k0, q·k1, q·k2] = [1, 0, 1], scale=1/√2
    let scale = 1.0 / (2.0_f32).sqrt();
    let weights = ref_softmax(&[1.0 * scale, 0.0 * scale, 1.0 * scale]);
    let expected_0 = weights[0] * 1.0 + weights[1] * 0.0 + weights[2] * 0.5;
    let expected_1 = weights[0] * 0.0 + weights[1] * 1.0 + weights[2] * 0.5;

    assert!(approx_eq(result[0], expected_0), "got {} want {}", result[0], expected_0);
    assert!(approx_eq(result[1], expected_1), "got {} want {}", result[1], expected_1);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Softmax normalization: attention weights sum to 1.0
// ═══════════════════════════════════════════════════════════════════

#[test]
fn attention_weights_sum_to_one_uniform_scores() {
    // With uniform Q and K, each softmax row should sum to 1.0.
    let head_dim = 4;
    let seq_len = 8;
    let q = vec![1.0; seq_len * head_dim];
    let k = vec![1.0; seq_len * head_dim];
    let v = vec![1.0; seq_len * head_dim];

    let result =
        scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();

    // Output should be V (all 1.0) since uniform attention * uniform V = V
    for (i, &val) in result.iter().enumerate() {
        assert!(approx_eq(val, 1.0), "index {i}: got {val} want 1.0");
    }
}

#[test]
fn attention_weights_sum_to_one_varying_scores() {
    // Use Q/K that produce non-uniform scores; output is a convex combination
    // so each output element must be within [min(v), max(v)].
    let head_dim = 2;
    let q = vec![2.0, 0.0]; // 1 query
    let k = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]; // 3 keys
    let v = vec![0.0, 0.0, 10.0, 10.0, 5.0, 5.0]; // 3 values

    let result = scaled_dot_product_attention(&q, &k, &v, 1, 3, head_dim, false).unwrap();

    // Output must be in convex hull: each dim in [0, 10]
    for &val in &result {
        assert!(val >= -EPS && val <= 10.0 + EPS, "out of convex hull: {val}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Causal masking: future positions are masked
// ═══════════════════════════════════════════════════════════════════

#[test]
fn causal_mask_blocks_future_positions() {
    let seq = 4;
    let mask = causal_mask(seq);

    // Lower triangle + diagonal = 0.0 (attend)
    for i in 0..seq {
        for j in 0..=i {
            assert_eq!(mask[i * seq + j], 0.0, "({i},{j}) should be 0.0");
        }
    }
    // Upper triangle = -inf (block)
    for i in 0..seq {
        for j in (i + 1)..seq {
            assert!(
                mask[i * seq + j] == f32::NEG_INFINITY,
                "({i},{j}) should be -inf, got {}",
                mask[i * seq + j]
            );
        }
    }
}

#[test]
fn causal_sdpa_first_token_attends_only_to_self() {
    let head_dim = 4;
    let seq = 3;
    let q = vec![1.0; seq * head_dim];
    let k = vec![1.0; seq * head_dim];
    // V: each token has distinct values
    let mut v = vec![0.0; seq * head_dim];
    for t in 0..seq {
        for d in 0..head_dim {
            v[t * head_dim + d] = (t * 10 + d) as f32;
        }
    }

    let result = scaled_dot_product_attention(&q, &k, &v, seq, seq, head_dim, true).unwrap();

    // First row (token 0): with causal mask, can only attend to position 0
    // → output should exactly equal V[0]
    for d in 0..head_dim {
        assert!(approx_eq(result[d], v[d]), "token 0, dim {d}: got {} want {}", result[d], v[d]);
    }
}

#[test]
fn causal_sdpa_last_token_attends_to_all_preceding() {
    let head_dim = 2;
    let seq = 3;
    // Uniform Q and K → uniform attention over visible positions
    let q = vec![1.0; seq * head_dim];
    let k = vec![1.0; seq * head_dim];
    let v = vec![
        10.0, 0.0, // token 0
        0.0, 10.0, // token 1
        5.0, 5.0, // token 2
    ];

    let result = scaled_dot_product_attention(&q, &k, &v, seq, seq, head_dim, true).unwrap();

    // Last token (index 2) attends uniformly to all 3 → average of V rows
    let expected_0 = (10.0 + 0.0 + 5.0) / 3.0;
    let expected_1 = (0.0 + 10.0 + 5.0) / 3.0;
    assert!(
        approx_eq(result[2 * head_dim], expected_0),
        "got {} want {}",
        result[2 * head_dim],
        expected_0
    );
    assert!(
        approx_eq(result[2 * head_dim + 1], expected_1),
        "got {} want {}",
        result[2 * head_dim + 1],
        expected_1
    );
}

// ═══════════════════════════════════════════════════════════════════
// 4. Multi-head attention: different heads attend independently
// ═══════════════════════════════════════════════════════════════════

#[test]
fn multi_head_different_heads_produce_different_outputs() {
    let num_heads = 2;
    let head_dim = 2;
    let seq_len = 2;
    let model_dim = num_heads * head_dim; // 4

    // Head 0: Q=[1,0], K=[1,0;0,1] → attends more to token 0
    // Head 1: Q=[0,1], K=[1,0;0,1] → attends more to token 1
    #[rustfmt::skip]
    let q = vec![
        // token 0: [head0: 1,0 | head1: 0,1]
        1.0, 0.0, 0.0, 1.0,
        // token 1: [head0: 1,0 | head1: 0,1]
        1.0, 0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let k = vec![
        // token 0: [head0: 1,0 | head1: 1,0]
        1.0, 0.0, 1.0, 0.0,
        // token 1: [head0: 0,1 | head1: 0,1]
        0.0, 1.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let v = vec![
        // token 0: [head0: 10,20 | head1: 10,20]
        10.0, 20.0, 10.0, 20.0,
        // token 1: [head0: 30,40 | head1: 30,40]
        30.0, 40.0, 30.0, 40.0,
    ];

    let result = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();
    assert_eq!(result.len(), seq_len * model_dim);

    // Head 0 output for token 0 (indices 0..2) should differ from
    // head 1 output for token 0 (indices 2..4) because they have
    // different Q vectors attending to different K patterns.
    let head0_t0 = &result[0..2];
    let head1_t0 = &result[2..4];
    let differs = head0_t0.iter().zip(head1_t0).any(|(a, b)| (a - b).abs() > EPS);
    assert!(differs, "heads should produce different outputs");
}

#[test]
fn multi_head_single_head_equals_sdpa() {
    let num_heads = 1;
    let head_dim = 4;
    let seq_len = 3;
    let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let k = vec![1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let mha = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();

    let sdpa = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();

    for (i, (&a, &b)) in mha.iter().zip(sdpa.iter()).enumerate() {
        assert!(approx_eq(a, b), "index {i}: mha={a} sdpa={b}");
    }
}

#[test]
fn multi_head_causal_preserves_causality() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 4;
    let model_dim = num_heads * head_dim;

    // Use distinct V per token so we can detect if future leaks in
    let mut v = vec![0.0; seq_len * model_dim];
    for t in 0..seq_len {
        for d in 0..model_dim {
            v[t * model_dim + d] = ((t + 1) * 100 + d) as f32;
        }
    }
    let q = vec![1.0; seq_len * model_dim];
    let k = vec![1.0; seq_len * model_dim];

    let causal_out =
        multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, true).unwrap();
    let non_causal_out =
        multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();

    // Token 0 with causal: attends only to self → equals V[0]
    // Token 0 without causal: attends to all → average of all V rows
    // These should differ (unless all V rows are identical, which they're not)
    let t0_causal = &causal_out[0..model_dim];
    let t0_non_causal = &non_causal_out[0..model_dim];
    let differs = t0_causal.iter().zip(t0_non_causal).any(|(a, b)| (a - b).abs() > EPS);
    assert!(differs, "causal token 0 should differ from non-causal (different visible set)");
}

// ═══════════════════════════════════════════════════════════════════
// 5. KV cache operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn kv_cache_append_and_readback() {
    let num_heads = 2;
    let head_dim = 4;
    let cfg =
        KvCacheConfig { num_layers: 1, num_heads, head_dim, max_seq_len: 16, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();
    let te = num_heads * head_dim; // 8

    // Append token 0
    let k0: Vec<f32> = (0..te).map(|i| i as f32).collect();
    let v0: Vec<f32> = (0..te).map(|i| (i as f32) + 100.0).collect();
    kv_cache_append(&mut cache, 0, &k0, &v0).unwrap();

    // Append token 1
    let k1: Vec<f32> = (0..te).map(|i| (i as f32) + 50.0).collect();
    let v1: Vec<f32> = (0..te).map(|i| (i as f32) + 200.0).collect();
    kv_cache_append(&mut cache, 0, &k1, &v1).unwrap();

    // Read back all
    let (keys, vals) = kv_cache_slice(&cache, 0, 0, 2).unwrap();
    assert_eq!(keys.len(), 2 * te);
    assert_eq!(vals.len(), 2 * te);

    // Verify token 0
    for i in 0..te {
        assert!(approx_eq(keys[i], k0[i]), "key[{i}]");
        assert!(approx_eq(vals[i], v0[i]), "val[{i}]");
    }
    // Verify token 1
    for i in 0..te {
        assert!(approx_eq(keys[te + i], k1[i]), "key[{i}+te]");
        assert!(approx_eq(vals[te + i], v1[i]), "val[{i}+te]");
    }
}

#[test]
fn kv_cache_clear_and_reuse() {
    let cfg = KvCacheConfig {
        num_layers: 2,
        num_heads: 1,
        head_dim: 2,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    kv_cache_append(&mut cache, 0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    kv_cache_append(&mut cache, 1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();

    kv_cache_clear(&mut cache);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);

    // Re-append after clear should work
    kv_cache_append(&mut cache, 0, &[9.0, 10.0], &[11.0, 12.0]).unwrap();
    let (k, v) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert!(approx_eq(k[0], 9.0));
    assert!(approx_eq(v[1], 12.0));
}

#[test]
fn attention_with_kv_cache_incremental_decoding() {
    let head_dim = 4;

    let mut k_cache = Vec::new();
    let mut v_cache = Vec::new();

    // Step 1: first token
    let q1 = vec![1.0; head_dim];
    let k1 = vec![1.0; head_dim];
    let v1 = vec![10.0; head_dim];

    let out1 =
        attention_with_kv_cache(&q1, &mut k_cache, &mut v_cache, &k1, &v1, head_dim).unwrap();
    assert_eq!(out1.len(), head_dim);
    // Only one KV → output = V
    for &val in &out1 {
        assert!(approx_eq(val, 10.0), "step1: got {val}");
    }
    assert_eq!(k_cache.len(), head_dim);

    // Step 2: second token with different V
    let q2 = vec![1.0; head_dim];
    let k2 = vec![1.0; head_dim];
    let v2 = vec![20.0; head_dim];

    let out2 =
        attention_with_kv_cache(&q2, &mut k_cache, &mut v_cache, &k2, &v2, head_dim).unwrap();
    assert_eq!(out2.len(), head_dim);
    assert_eq!(k_cache.len(), 2 * head_dim);
    // Uniform Q·K → equal attention → average of V: (10+20)/2 = 15
    for &val in &out2 {
        assert!(approx_eq(val, 15.0), "step2: got {val} want 15.0");
    }

    // Step 3: third token
    let q3 = vec![1.0; head_dim];
    let k3 = vec![1.0; head_dim];
    let v3 = vec![30.0; head_dim];

    let out3 =
        attention_with_kv_cache(&q3, &mut k_cache, &mut v_cache, &k3, &v3, head_dim).unwrap();
    // 3 values → average = (10+20+30)/3 = 20
    for &val in &out3 {
        assert!(approx_eq(val, 20.0), "step3: got {val} want 20.0");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Edge cases: seq_len=1, single head
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sdpa_seq_len_1_returns_value_unchanged() {
    let head_dim = 8;
    let v: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 3.14).collect();
    let q = vec![0.5; head_dim];
    let k = vec![0.5; head_dim];

    let result = scaled_dot_product_attention(&q, &k, &v, 1, 1, head_dim, false).unwrap();

    // Single KV → softmax([score]) = [1.0] → output = V exactly
    for (i, (&got, &want)) in result.iter().zip(v.iter()).enumerate() {
        assert!(approx_eq(got, want), "dim {i}: got {got} want {want}");
    }
}

#[test]
fn sdpa_seq_len_1_causal_returns_value_unchanged() {
    let head_dim = 4;
    let v = vec![42.0, 43.0, 44.0, 45.0];
    let q = vec![1.0; head_dim];
    let k = vec![1.0; head_dim];

    let result = scaled_dot_product_attention(&q, &k, &v, 1, 1, head_dim, true).unwrap();

    for (i, (&got, &want)) in result.iter().zip(v.iter()).enumerate() {
        assert!(approx_eq(got, want), "dim {i}: got {got} want {want}");
    }
}

#[test]
fn multi_head_single_head_single_token() {
    let head_dim = 4;
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let q = vec![0.1; head_dim];
    let k = vec![0.1; head_dim];

    let result = multi_head_attention_cpu(&q, &k, &v, 1, head_dim, 1, false).unwrap();

    for (i, (&got, &want)) in result.iter().zip(v.iter()).enumerate() {
        assert!(approx_eq(got, want), "dim {i}: got {got} want {want}");
    }
}

#[test]
fn cpu_attention_batched_single_batch() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 2;
    let total = seq_len * num_heads * head_dim;

    let q = vec![0.1; total];
    let k = vec![0.1; total];
    let v: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let attn = CpuAttention::new(CpuAttentionConfig {
        batch_size: 1,
        num_heads,
        seq_len,
        head_dim,
        scale: None,
        causal_mask: false,
    })
    .unwrap();

    let result = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(result.len(), total);

    // All finite
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "index {i} is not finite: {val}");
    }
}

#[test]
fn causal_attention_wrapper_matches_mha_causal() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let total = seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k: Vec<f32> = (0..total).map(|i| ((total - i) as f32) * 0.01).collect();
    let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

    let cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false, // causal_attention forces true
        use_alibi: false,
        scale: None,
    };
    let causal_result = causal_attention(&q, &k, &v, &cfg).unwrap();
    let mha_causal =
        multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, true).unwrap();

    for (i, (&a, &b)) in causal_result.iter().zip(mha_causal.iter()).enumerate() {
        assert!(approx_eq(a, b), "index {i}: causal_attn={a} mha={b}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. GQA: grouped query attention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gqa_1to1_equals_standard_mha() {
    // When num_kv_heads == num_q_heads, GQA should match standard MHA
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let total = seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
    let k: Vec<f32> = (0..total).map(|i| ((total - i) as f32) * 0.05).collect();
    let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

    let gqa_cfg = GqaConfig {
        num_q_heads: num_heads,
        num_kv_heads: num_heads,
        head_dim,
        seq_len,
        causal: false,
        scale: None,
    };
    let gqa_result = AttentionKernel::grouped_query_attention(&q, &k, &v, &gqa_cfg).unwrap();
    let mha_result =
        multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();

    for (i, (&a, &b)) in gqa_result.iter().zip(mha_result.iter()).enumerate() {
        assert!(approx_eq(a, b), "index {i}: gqa={a} mha={b}");
    }
}

#[test]
fn gqa_multiple_queries_per_kv_head() {
    // 4 query heads, 2 KV heads → 2:1 grouping
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 2;
    let seq_len = 2;

    let q_dim = num_q_heads * head_dim; // 8
    let kv_dim = num_kv_heads * head_dim; // 4

    let q = vec![1.0; seq_len * q_dim];
    let k = vec![1.0; seq_len * kv_dim];
    let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32) * 10.0).collect();

    let cfg =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: false, scale: None };
    let result = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).unwrap();
    assert_eq!(result.len(), seq_len * q_dim);

    // Query heads 0,1 share KV head 0; query heads 2,3 share KV head 1.
    // With uniform Q and K, all attention weights are equal → output = mean(V)
    // per KV head group.
    //
    // Both Q-heads in same group should produce identical output.
    // Extract head 0 and head 1 output for token 0:
    let h0_t0 = &result[0..head_dim];
    let h1_t0 = &result[head_dim..2 * head_dim];
    for (i, (&a, &b)) in h0_t0.iter().zip(h1_t0.iter()).enumerate() {
        assert!(approx_eq(a, b), "grouped heads 0,1 should match at dim {i}: {a} vs {b}");
    }

    let h2_t0 = &result[2 * head_dim..3 * head_dim];
    let h3_t0 = &result[3 * head_dim..4 * head_dim];
    for (i, (&a, &b)) in h2_t0.iter().zip(h3_t0.iter()).enumerate() {
        assert!(approx_eq(a, b), "grouped heads 2,3 should match at dim {i}: {a} vs {b}");
    }

    // The two groups should differ (different KV heads → different V)
    let differs = h0_t0.iter().zip(h2_t0).any(|(a, b)| (a - b).abs() > EPS);
    assert!(differs, "different KV head groups should produce different outputs");
}

#[test]
fn gqa_causal_masks_future() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q = vec![1.0; seq_len * q_dim];
    let k = vec![1.0; seq_len * kv_dim];
    let mut v = vec![0.0; seq_len * kv_dim];
    for t in 0..seq_len {
        for d in 0..kv_dim {
            v[t * kv_dim + d] = ((t + 1) * 100 + d) as f32;
        }
    }

    let cfg_causal =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: true, scale: None };
    let cfg_non_causal =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: false, scale: None };

    let causal_out = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg_causal).unwrap();
    let non_causal_out =
        AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg_non_causal).unwrap();

    // Token 0 should differ: causal sees only self, non-causal sees all
    let t0_causal = &causal_out[0..q_dim];
    let t0_non_causal = &non_causal_out[0..q_dim];
    let differs = t0_causal.iter().zip(t0_non_causal).any(|(a, b)| (a - b).abs() > EPS);
    assert!(differs, "GQA causal should differ from non-causal at token 0");
}

#[test]
fn gqa_rejects_non_divisible_heads() {
    let cfg = GqaConfig {
        num_q_heads: 5,
        num_kv_heads: 3,
        head_dim: 4,
        seq_len: 2,
        causal: false,
        scale: None,
    };
    let q = vec![1.0; 2 * 5 * 4];
    let k = vec![1.0; 2 * 3 * 4];
    let v = vec![1.0; 2 * 3 * 4];
    assert!(AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// 8. Determinism / reproducibility
// ═══════════════════════════════════════════════════════════════════

#[test]
fn sdpa_deterministic_across_calls() {
    let head_dim = 8;
    let seq = 4;
    let q: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32) * 0.03).collect();
    let k: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32) * 0.02).collect();
    let v: Vec<f32> = (0..seq * head_dim).map(|i| (i as f32) * 0.07).collect();

    let r1 = scaled_dot_product_attention(&q, &k, &v, seq, seq, head_dim, true).unwrap();
    let r2 = scaled_dot_product_attention(&q, &k, &v, seq, seq, head_dim, true).unwrap();

    assert_eq!(r1.len(), r2.len());
    for (i, (&a, &b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(a, b, "non-deterministic at index {i}: {a} vs {b}");
    }
}

#[test]
fn mha_deterministic_across_calls() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let total = seq_len * num_heads * head_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k: Vec<f32> = (0..total).map(|i| (i as f32) * 0.02).collect();
    let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.03).collect();

    let r1 = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, true).unwrap();
    let r2 = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, true).unwrap();

    for (i, (&a, &b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(a, b, "non-deterministic at index {i}");
    }
}
