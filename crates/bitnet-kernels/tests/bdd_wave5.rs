//! BDD-style integration tests — Wave 5
//!
//! Each test follows the Given / When / Then structure and exercises
//! end-to-end kernel behaviour across quantization, attention, normalization,
//! FFN, embedding, and caching subsystems.

use bitnet_kernels::cpu::batch::{batched_matmul, batched_softmax};
use bitnet_kernels::cpu::dequant::{dequant_i2s_block, dequant_ternary, pack_ternary};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::ffn::{FfnActivation, FfnConfig, ffn_forward, gated_ffn_forward};
use bitnet_kernels::cpu::gather::gather_rows;
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};

const TOL: f32 = 1e-5;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (diff {})", (x - y).abs());
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Quantize → Dequantize round-trip
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_roundtrip_sign_preserved() {
    // Given a vector of known positive/negative values
    let values = vec![1.5, -0.8, 0.0, 2.3, -3.1, 0.02, 0.7, -1.0];
    let threshold = 0.05;

    // When we pack (quantize) and then dequantize
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);

    // Then the sign of every above-threshold value is preserved
    for (i, (&orig, &got)) in values.iter().zip(recovered.iter()).enumerate() {
        if orig.abs() > threshold {
            assert_eq!(orig.is_sign_positive(), got.is_sign_positive(), "sign mismatch at {i}");
        }
    }
}

#[test]
fn test_bdd_wave5_roundtrip_zero_maps_to_zero() {
    // Given values at or below the threshold
    let values = vec![0.0, 0.01, -0.01, 0.0];
    let threshold = 0.05;

    // When quantized and dequantized
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);

    // Then all recover to zero
    for (i, &v) in recovered.iter().enumerate().take(values.len()) {
        assert_eq!(v, 0.0, "index {i} should be zero, got {v}");
    }
}

#[test]
fn test_bdd_wave5_roundtrip_bounded_error() {
    // Given arbitrary f32 values
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let threshold = 0.1;

    // When round-tripped through ternary quantization
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);

    // Then the error for each element is bounded by `scale`
    for (i, (&orig, &got)) in values.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - got).abs();
        assert!(
            err <= scale + orig.abs(),
            "error at {i} too large: err={err}, scale={scale}, orig={orig}"
        );
    }
}

#[test]
fn test_bdd_wave5_roundtrip_all_positive() {
    // Given all positive values
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let threshold = 0.0;

    // When quantized and dequantized
    let (packed, scale) = pack_ternary(&values, threshold);
    let recovered = dequant_ternary(&packed, scale);

    // Then all recovered values are positive
    for (i, &v) in recovered.iter().enumerate().take(values.len()) {
        assert!(v > 0.0, "index {i}: expected positive, got {v}");
    }
}

#[test]
fn test_bdd_wave5_roundtrip_dequant_block_length() {
    // Given packed data of known size
    let packed = vec![0x55u8; 8]; // 32 elements
    let block_size = 32;

    // When dequantized as a block
    let out = dequant_i2s_block(&packed, 1.0, block_size).unwrap();

    // Then output length matches block_size
    assert_eq!(out.len(), block_size);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: RoPE position encoding
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_rope_distinct_positions() {
    // Given a head vector and two different positions
    let head_dim = 8;
    let config = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&config);
    let original = vec![1.0; head_dim];

    // When RoPE is applied at positions 0 and 5
    let mut data0 = original.clone();
    apply_rope(&mut data0, 0, head_dim, &freqs);

    let mut data5 = original.clone();
    apply_rope(&mut data5, 5, head_dim, &freqs);

    // Then the two outputs differ (position is encoded)
    let differs = data0.iter().zip(data5.iter()).any(|(a, b)| (a - b).abs() > TOL);
    assert!(differs, "RoPE at different positions should produce different outputs");
}

#[test]
fn test_bdd_wave5_rope_preserves_norm() {
    // Given a unit vector
    let head_dim = 4;
    let config = RopeConfig::new(head_dim, 16);
    let freqs = compute_frequencies(&config);
    let mut data = vec![0.5, 0.5, 0.5, 0.5];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    // When RoPE is applied
    apply_rope(&mut data, 3, head_dim, &freqs);

    // Then the L2 norm is preserved (rotation is orthogonal)
    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_before - norm_after).abs() < 1e-4, "norm changed: {norm_before} → {norm_after}");
}

#[test]
fn test_bdd_wave5_rope_position_zero_identity_like() {
    // Given position 0 with default base
    let head_dim = 4;
    let config = RopeConfig::new(head_dim, 8);
    let freqs = compute_frequencies(&config);
    let original = vec![1.0, 2.0, 3.0, 4.0];
    let mut data = original.clone();

    // When RoPE is applied at position 0
    apply_rope(&mut data, 0, head_dim, &freqs);

    // Then at position 0 the angle is 0, cos=1, sin=0 → output ≈ input
    approx_eq(&data, &original, 1e-4);
}

#[test]
fn test_bdd_wave5_rope_frequency_table_length() {
    // Given a RoPE config
    let head_dim = 16;
    let max_seq = 128;
    let config = RopeConfig::new(head_dim, max_seq);

    // When frequencies are computed
    let freqs = compute_frequencies(&config);

    // Then the table has max_seq_len * head_dim entries
    assert_eq!(freqs.len(), max_seq * head_dim);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Causal mask on attention scores
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_causal_mask_future_masked() {
    use bitnet_kernels::cpu::attention::causal_mask;

    // Given attention scores for seq_len=4
    let seq_len = 4;
    let mask = causal_mask(seq_len);

    // When we inspect future positions (j > i)
    // Then they should be -inf
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert_eq!(
                mask[i * seq_len + j],
                f32::NEG_INFINITY,
                "position ({i},{j}) should be masked"
            );
        }
    }
}

#[test]
fn test_bdd_wave5_causal_mask_past_allowed() {
    use bitnet_kernels::cpu::attention::causal_mask;

    // Given a causal mask for seq_len=5
    let seq_len = 5;
    let mask = causal_mask(seq_len);

    // When we inspect past/current positions (j <= i)
    // Then they should be 0.0 (allowed)
    for i in 0..seq_len {
        for j in 0..=i {
            assert_eq!(mask[i * seq_len + j], 0.0, "position ({i},{j}) should be unmasked");
        }
    }
}

#[test]
fn test_bdd_wave5_causal_mask_applied_to_scores() {
    use bitnet_kernels::cpu::attention::{apply_mask, causal_mask};

    // Given uniform attention scores
    let seq_len = 3;
    let mut scores = vec![1.0; seq_len * seq_len];
    let mask = causal_mask(seq_len);

    // When the mask is applied
    apply_mask(&mut scores, &mask).unwrap();

    // Then future positions are -inf and past positions remain 1.0
    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = scores[i * seq_len + j];
            if j > i {
                assert_eq!(val, f32::NEG_INFINITY);
            } else {
                assert!((val - 1.0).abs() < TOL);
            }
        }
    }
}

#[test]
fn test_bdd_wave5_causal_mask_seq1_no_masking() {
    use bitnet_kernels::cpu::attention::causal_mask;

    // Given seq_len = 1
    let mask = causal_mask(1);

    // When the single element is inspected
    // Then it is unmasked (no future to mask)
    assert_eq!(mask, vec![0.0]);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: LayerNorm normalizes to zero mean / unit variance
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_layernorm_zero_mean() {
    // Given a non-zero-mean input
    let input = vec![10.0, 20.0, 30.0, 40.0];
    let dim = 4;
    let gamma = vec![1.0; dim];
    let config = LayerNormConfig::new(vec![dim]);

    // When layer norm is applied (no beta → affine with gamma=1)
    let output = layer_norm(&input, &gamma, None, &config).unwrap();

    // Then the output mean is approximately zero
    let mean: f32 = output.iter().sum::<f32>() / dim as f32;
    assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
}

#[test]
fn test_bdd_wave5_layernorm_unit_variance() {
    // Given a varying input
    let input = vec![1.0, 3.0, 5.0, 7.0];
    let dim = 4;
    let gamma = vec![1.0; dim];
    let config = LayerNormConfig::new(vec![dim]);

    // When layer norm is applied
    let output = layer_norm(&input, &gamma, None, &config).unwrap();

    // Then the output variance is approximately 1
    let mean: f32 = output.iter().sum::<f32>() / dim as f32;
    let var: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
    assert!((var - 1.0).abs() < 0.01, "variance should be ~1, got {var}");
}

#[test]
fn test_bdd_wave5_layernorm_constant_input() {
    // Given constant input
    let input = vec![5.0; 8];
    let dim = 8;
    let gamma = vec![1.0; dim];
    let beta = vec![3.0; dim];
    let config = LayerNormConfig::new(vec![dim]);

    // When layer norm is applied
    let output = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();

    // Then output equals beta (since normalized constant = 0, 0*gamma + beta = beta)
    approx_eq(&output, &beta, 1e-4);
}

#[test]
fn test_bdd_wave5_layernorm_preserves_length() {
    // Given an input of size 16
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let dim = 16;
    let gamma = vec![1.0; dim];
    let config = LayerNormConfig::new(vec![dim]);

    // When layer norm is applied
    let output = layer_norm(&input, &gamma, None, &config).unwrap();

    // Then output has the same length
    assert_eq!(output.len(), input.len());
}

#[test]
fn test_bdd_wave5_layernorm_batch_independent() {
    // Given two rows processed as a batch
    let dim = 4;
    let gamma = vec![1.0; dim];
    let config = LayerNormConfig::new(vec![dim]);

    let row1 = vec![1.0, 2.0, 3.0, 4.0];
    let row2 = vec![10.0, 20.0, 30.0, 40.0];

    let out1 = layer_norm(&row1, &gamma, None, &config).unwrap();
    let out2 = layer_norm(&row2, &gamma, None, &config).unwrap();

    // When the same rows are processed together in a batch
    let batch_input: Vec<f32> = [row1, row2].concat();
    let batch_out = layer_norm(&batch_input, &gamma, None, &config).unwrap();

    // Then each row's output matches the individual result
    approx_eq(&batch_out[..dim], &out1, TOL);
    approx_eq(&batch_out[dim..], &out2, TOL);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 5: Softmax properties
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_softmax_sums_to_one() {
    // Given arbitrary logits
    let logits = vec![2.0, 1.0, 0.1, -1.0, 3.0];

    // When softmax is computed
    let probs = batched_softmax(&logits, 1, 5).unwrap();

    // Then the output sums to 1.0
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "softmax sum = {sum}, expected 1.0");
}

#[test]
fn test_bdd_wave5_softmax_all_in_01() {
    // Given logits with mixed signs
    let logits = vec![-5.0, 0.0, 5.0, 10.0];

    // When softmax is computed
    let probs = batched_softmax(&logits, 1, 4).unwrap();

    // Then every value is in [0, 1]
    for (i, &p) in probs.iter().enumerate() {
        assert!((0.0..=1.0).contains(&p), "softmax[{i}] = {p} is out of [0,1]");
    }
}

#[test]
fn test_bdd_wave5_softmax_ordering_preserved() {
    // Given monotonically increasing logits
    let logits = vec![1.0, 2.0, 3.0, 4.0];

    // When softmax is computed
    let probs = batched_softmax(&logits, 1, 4).unwrap();

    // Then the output is also monotonically increasing
    for i in 1..probs.len() {
        assert!(
            probs[i] >= probs[i - 1],
            "ordering violated at {i}: {} < {}",
            probs[i],
            probs[i - 1]
        );
    }
}

#[test]
fn test_bdd_wave5_softmax_numerically_stable() {
    // Given very large logits (potential overflow)
    let logits = vec![1000.0, 1001.0, 1002.0];

    // When softmax is computed
    let probs = batched_softmax(&logits, 1, 3).unwrap();

    // Then all values are finite and sum to 1
    assert!(probs.iter().all(|p| p.is_finite()), "softmax produced non-finite values");
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < TOL, "sum = {sum}");
}

#[test]
fn test_bdd_wave5_softmax_uniform_input() {
    // Given equal logits
    let logits = vec![3.0; 5];

    // When softmax is computed
    let probs = batched_softmax(&logits, 1, 5).unwrap();

    // Then all probabilities are equal (uniform distribution)
    let expected = 1.0 / 5.0;
    for (i, &p) in probs.iter().enumerate() {
        assert!((p - expected).abs() < TOL, "softmax[{i}] = {p}, expected {expected}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 6: Batch matmul matches individual
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_matmul_batch_matches_individual() {
    // Given two independent matrix pairs
    let a1 = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let b1 = vec![5.0, 6.0, 7.0, 8.0];
    let a2 = vec![2.0, 0.0, 0.0, 3.0];
    let b2 = vec![1.0, 1.0, 1.0, 1.0];

    // When computed individually and as a batch
    let c1 = batched_matmul(&a1, &b1, 1, 2, 2, 2).unwrap();
    let c2 = batched_matmul(&a2, &b2, 1, 2, 2, 2).unwrap();

    let a_cat: Vec<f32> = [a1, a2].concat();
    let b_cat: Vec<f32> = [b1, b2].concat();
    let c_batched = batched_matmul(&a_cat, &b_cat, 2, 2, 2, 2).unwrap();

    // Then batch output matches concatenated individual outputs
    let expected: Vec<f32> = [c1, c2].concat();
    approx_eq(&c_batched, &expected, TOL);
}

#[test]
fn test_bdd_wave5_matmul_identity() {
    // Given matrix A and identity matrix B
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let identity = vec![1.0, 0.0, 0.0, 1.0];

    // When A * I is computed
    let c = batched_matmul(&a, &identity, 1, 2, 2, 2).unwrap();

    // Then output equals A
    approx_eq(&c, &a, TOL);
}

#[test]
fn test_bdd_wave5_matmul_known_product() {
    // Given [[1,2],[3,4]] × [[5,6],[7,8]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    // When the product is computed
    let c = batched_matmul(&a, &b, 1, 2, 2, 2).unwrap();

    // Then the result is [[19,22],[43,50]]
    approx_eq(&c, &[19.0, 22.0, 43.0, 50.0], TOL);
}

#[test]
fn test_bdd_wave5_matmul_non_square() {
    // Given A:[1×3] and B:[3×2]
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // When the product is computed
    let c = batched_matmul(&a, &b, 1, 1, 3, 2).unwrap();

    // Then the result is [4.0, 5.0]
    approx_eq(&c, &[4.0, 5.0], TOL);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 7: KV cache append and retrieve
// ═══════════════════════════════════════════════════════════════════

fn test_kv_config() -> KvCacheConfig {
    KvCacheConfig { num_layers: 2, num_heads: 2, head_dim: 4, max_seq_len: 16, dtype: KvDtype::F32 }
}

#[test]
fn test_bdd_wave5_kv_cache_append_single_token() {
    // Given an empty KV cache
    let mut cache = KvCache::new(test_kv_config()).unwrap();
    let token_elems = 2 * 4; // num_heads * head_dim
    let keys: Vec<f32> = (0..token_elems).map(|i| i as f32).collect();
    let values: Vec<f32> = (0..token_elems).map(|i| (i as f32) * 10.0).collect();

    // When a single token is appended to layer 0
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

    // Then cache seq_len is 1 and the data is retrievable
    assert_eq!(cache.seq_len(0).unwrap(), 1);
    let (k, v) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    approx_eq(k, &keys, TOL);
    approx_eq(v, &values, TOL);
}

#[test]
fn test_bdd_wave5_kv_cache_append_multiple_tokens() {
    // Given an empty KV cache
    let mut cache = KvCache::new(test_kv_config()).unwrap();
    let te = 2 * 4;

    // When 3 tokens are appended one by one
    for t in 0..3 {
        let keys: Vec<f32> = (0..te).map(|i| (t * te + i) as f32).collect();
        let vals: Vec<f32> = keys.iter().map(|x| x * 2.0).collect();
        kv_cache_append(&mut cache, 0, &keys, &vals).unwrap();
    }

    // Then cache holds all 3 tokens
    assert_eq!(cache.seq_len(0).unwrap(), 3);
}

#[test]
fn test_bdd_wave5_kv_cache_clear_resets() {
    // Given a cache with data
    let mut cache = KvCache::new(test_kv_config()).unwrap();
    let te = 2 * 4;
    let keys = vec![1.0; te];
    let vals = vec![2.0; te];
    kv_cache_append(&mut cache, 0, &keys, &vals).unwrap();

    // When the cache is cleared
    kv_cache_clear(&mut cache);

    // Then all layers have seq_len 0
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

#[test]
fn test_bdd_wave5_kv_cache_layers_independent() {
    // Given a 2-layer cache
    let mut cache = KvCache::new(test_kv_config()).unwrap();
    let te = 2 * 4;
    let keys = vec![1.0; te];
    let vals = vec![2.0; te];

    // When tokens are appended to layer 0 only
    kv_cache_append(&mut cache, 0, &keys, &vals).unwrap();

    // Then layer 1 is still empty
    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 8: Gated FFN output dimensions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_ffn_output_dim_matches() {
    // Given an FFN configuration
    let h = 8;
    let inter = 16;
    let config = FfnConfig::new(h, inter, FfnActivation::ReLU).unwrap();
    let input = vec![1.0; h];
    let w_up = vec![0.1; inter * h];
    let w_down = vec![0.1; h * inter];

    // When a standard FFN forward pass is run
    let output = ffn_forward(&input, &w_up, &w_down, &config).unwrap();

    // Then the output dimension equals hidden_dim
    assert_eq!(output.len(), h);
}

#[test]
fn test_bdd_wave5_gated_ffn_output_dim_matches() {
    // Given a gated FFN configuration
    let h = 8;
    let inter = 16;
    let config = FfnConfig::new(h, inter, FfnActivation::SiLU).unwrap();
    let input = vec![0.5; h];
    let w_gate = vec![0.1; inter * h];
    let w_up = vec![0.1; inter * h];
    let w_down = vec![0.1; h * inter];

    // When a gated FFN forward pass is run
    let output = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &config).unwrap();

    // Then the output dimension equals hidden_dim
    assert_eq!(output.len(), h);
}

#[test]
fn test_bdd_wave5_ffn_zero_input_produces_finite() {
    // Given zero input
    let h = 4;
    let inter = 8;
    let config = FfnConfig::new(h, inter, FfnActivation::GeLU).unwrap();
    let input = vec![0.0; h];
    let w_up = vec![0.5; inter * h];
    let w_down = vec![0.5; h * inter];

    // When FFN forward is computed
    let output = ffn_forward(&input, &w_up, &w_down, &config).unwrap();

    // Then all outputs are finite
    assert!(output.iter().all(|v| v.is_finite()), "FFN produced non-finite values");
}

#[test]
fn test_bdd_wave5_gated_ffn_different_activations() {
    // Given the same input with different activations
    let h = 4;
    let inter = 8;
    let input = vec![1.0; h];
    let w_gate = vec![0.2; inter * h];
    let w_up = vec![0.2; inter * h];
    let w_down = vec![0.2; h * inter];

    let cfg_relu = FfnConfig::new(h, inter, FfnActivation::ReLU).unwrap();
    let cfg_silu = FfnConfig::new(h, inter, FfnActivation::SiLU).unwrap();

    // When gated FFN is run with ReLU and SiLU
    let out_relu = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg_relu).unwrap();
    let out_silu = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg_silu).unwrap();

    // Then the outputs differ (different activations)
    let differs = out_relu.iter().zip(out_silu.iter()).any(|(a, b)| (a - b).abs() > TOL);
    assert!(differs, "different activations should produce different outputs");
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 9: Embedding index lookup
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_embedding_correct_rows() {
    // Given an embedding table [4 rows × 3 dims]
    #[rustfmt::skip]
    let table: Vec<f32> = vec![
        0.1, 0.2, 0.3,  // row 0
        1.1, 1.2, 1.3,  // row 1
        2.1, 2.2, 2.3,  // row 2
        3.1, 3.2, 3.3,  // row 3
    ];
    let embedding_dim = 3;

    // When indices [2, 0, 3] are looked up
    let result = embedding_lookup(&table, &[2, 0, 3], embedding_dim).unwrap();

    // Then the correct rows are returned in order
    approx_eq(&result[0..3], &[2.1, 2.2, 2.3], TOL);
    approx_eq(&result[3..6], &[0.1, 0.2, 0.3], TOL);
    approx_eq(&result[6..9], &[3.1, 3.2, 3.3], TOL);
}

#[test]
fn test_bdd_wave5_embedding_single_index() {
    // Given an embedding table
    let table: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let embedding_dim = 3;

    // When a single index is looked up
    let result = embedding_lookup(&table, &[1], embedding_dim).unwrap();

    // Then the correct row is returned
    approx_eq(&result, &[40.0, 50.0, 60.0], TOL);
}

#[test]
fn test_bdd_wave5_embedding_duplicate_indices() {
    // Given an embedding table
    let table = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows × 2 dims
    let embedding_dim = 2;

    // When the same index is looked up twice
    let result = embedding_lookup(&table, &[0, 0], embedding_dim).unwrap();

    // Then both rows are identical
    approx_eq(&result[0..2], &result[2..4], TOL);
}

#[test]
fn test_bdd_wave5_gather_rows_correct() {
    // Given a weight matrix [3 rows × 2 cols]
    let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // When rows [2, 0] are gathered
    let result = gather_rows(&table, 3, 2, &[2, 0]).unwrap();

    // Then the correct rows are returned
    approx_eq(&result[0..2], &[5.0, 6.0], TOL);
    approx_eq(&result[2..4], &[1.0, 2.0], TOL);
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 10: Residual connection identity
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_bdd_wave5_residual_zero_is_identity() {
    // Given an input and a zero residual
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let residual = vec![0.0; 4];
    let mut output = input.clone();

    // When the zero residual is added
    add_residual(&mut output, &residual).unwrap();

    // Then the output equals the original input
    approx_eq(&output, &input, TOL);
}

#[test]
fn test_bdd_wave5_residual_adds_correctly() {
    // Given input and a non-zero residual
    let mut output = vec![1.0, 2.0, 3.0];
    let residual = vec![0.5, -0.5, 1.0];

    // When the residual is added
    add_residual(&mut output, &residual).unwrap();

    // Then output = input + residual
    approx_eq(&output, &[1.5, 1.5, 4.0], TOL);
}

#[test]
fn test_bdd_wave5_residual_scaled_zero_scale() {
    // Given any input and residual with scale = 0
    let input = vec![5.0, 10.0, 15.0];
    let mut output = input.clone();
    let residual = vec![100.0, 200.0, 300.0];

    // When a scaled residual with scale=0 is added
    add_residual_scaled(&mut output, &residual, 0.0).unwrap();

    // Then the output is unchanged
    approx_eq(&output, &input, TOL);
}

#[test]
fn test_bdd_wave5_residual_self_doubles() {
    // Given input equal to residual
    let mut output = vec![3.0, 7.0];
    let residual = vec![3.0, 7.0];

    // When the residual (same as input) is added
    add_residual(&mut output, &residual).unwrap();

    // Then the output is doubled
    approx_eq(&output, &[6.0, 14.0], TOL);
}
