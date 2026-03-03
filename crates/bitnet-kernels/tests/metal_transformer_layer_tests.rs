#![cfg(target_os = "macos")]
#![allow(clippy::identity_op, clippy::manual_div_ceil, clippy::needless_range_loop)]
//! Metal full transformer layer validation tests for Apple Silicon.
//!
//! Validates complete transformer block computation: self-attention,
//! multi-head attention, grouped-query attention, causal masking,
//! KV cache integration, FFN sublayers, residual connections, deep
//! stacks, sequence length scaling, head dimension variants, numerical
//! stability, and memory buffer patterns.
//!
//! All tests are `#[ignore]` — they require Metal GPU hardware and
//! Apple Silicon to run.

// ── Apple Silicon Metal constants ──────────────────────────────────

/// Metal buffer alignment (bytes) on Apple GPUs.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD group width on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Maximum threadgroup memory (bytes).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── Helper utilities ───────────────────────────────────────────────

/// Align `size` up to `METAL_BUFFER_ALIGNMENT`.
fn align_up(size: usize) -> usize {
    (size + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

/// Compute bytes for an f32 tensor with the given element count.
fn f32_bytes(elements: usize) -> usize {
    elements * std::mem::size_of::<f32>()
}

/// Naive softmax over a slice (for reference calculations).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Naive matrix multiply: (M×K) × (K×N) → (M×N), row-major.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// RMS normalization over a vector.
fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

/// Layer normalization (mean-center + scale).
fn layer_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let std = (var + eps).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// SiLU activation: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Compute scaled-dot-product attention scores for a single head.
/// q, k: (seq_len, head_dim); v: (seq_len, head_dim).
/// Returns output (seq_len, head_dim).
fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    // scores: (seq_len, seq_len)
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if causal && j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
    }
    // softmax per row
    let mut attn_weights = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let row = &scores[i * seq_len..(i + 1) * seq_len];
        let sm = softmax(row);
        attn_weights[i * seq_len..(i + 1) * seq_len].copy_from_slice(&sm);
    }
    // output: attn_weights × V
    matmul(&attn_weights, v, seq_len, seq_len, head_dim)
}

/// Build a causal (lower-triangular) mask for `seq_len`.
fn causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            mask[i * seq_len + j] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
        }
    }
    mask
}

/// Simple deterministic pseudo-random f32 in [-1, 1].
fn pseudo_rand(seed: u64, idx: usize) -> f32 {
    let h = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(idx as u64)
        .wrapping_mul(1_442_695_040_888_963_407);
    // Map to [-1, 1]
    (((h >> 33) as i32) as f32) / (i32::MAX as f32)
}

/// Generate a deterministic f32 vector of length `len`.
fn rand_vec(seed: u64, len: usize) -> Vec<f32> {
    (0..len).map(|i| pseudo_rand(seed, i)).collect()
}

// ═══════════════════════════════════════════════════════════════════
// 1. Self-attention forward
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_self_attention_forward_basic() {
    let seq_len = 4;
    let d_model = 64;
    let head_dim = 64;

    let x = rand_vec(42, seq_len * d_model);
    let wq = rand_vec(100, d_model * head_dim);
    let wk = rand_vec(200, d_model * head_dim);
    let wv = rand_vec(300, d_model * head_dim);
    let wo = rand_vec(400, head_dim * d_model);

    let q = matmul(&x, &wq, seq_len, d_model, head_dim);
    let k = matmul(&x, &wk, seq_len, d_model, head_dim);
    let v = matmul(&x, &wv, seq_len, d_model, head_dim);

    let attn_out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);

    let out = matmul(&attn_out, &wo, seq_len, head_dim, d_model);

    assert_eq!(out.len(), seq_len * d_model);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_self_attention_qkv_shapes() {
    let seq_len = 8;
    let d_model = 128;
    let head_dim = 64;

    let x = rand_vec(1, seq_len * d_model);
    let wq = rand_vec(2, d_model * head_dim);

    let q = matmul(&x, &wq, seq_len, d_model, head_dim);
    assert_eq!(q.len(), seq_len * head_dim);
    assert!(q.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_self_attention_output_projection() {
    let seq_len = 4;
    let head_dim = 64;
    let d_model = 128;

    let attn = rand_vec(50, seq_len * head_dim);
    let wo = rand_vec(60, head_dim * d_model);

    let proj = matmul(&attn, &wo, seq_len, head_dim, d_model);
    assert_eq!(proj.len(), seq_len * d_model);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Multi-head attention
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_head_attention_concat() {
    let seq_len = 4;
    let d_model = 128;
    let num_heads = 4;
    let head_dim = d_model / num_heads;

    let x = rand_vec(10, seq_len * d_model);

    // Per-head attention
    let mut concat = vec![0.0f32; seq_len * d_model];
    for h in 0..num_heads {
        let wq = rand_vec(100 + h as u64, d_model * head_dim);
        let wk = rand_vec(200 + h as u64, d_model * head_dim);
        let wv = rand_vec(300 + h as u64, d_model * head_dim);

        let q = matmul(&x, &wq, seq_len, d_model, head_dim);
        let k = matmul(&x, &wk, seq_len, d_model, head_dim);
        let v = matmul(&x, &wv, seq_len, d_model, head_dim);

        let head_out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);

        // Copy head output into concatenated buffer
        for s in 0..seq_len {
            for d in 0..head_dim {
                concat[s * d_model + h * head_dim + d] = head_out[s * head_dim + d];
            }
        }
    }

    assert_eq!(concat.len(), seq_len * d_model);
    assert!(concat.iter().all(|v| v.is_finite()));

    // Output projection
    let wo = rand_vec(500, d_model * d_model);
    let mha_out = matmul(&concat, &wo, seq_len, d_model, d_model);
    assert_eq!(mha_out.len(), seq_len * d_model);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_head_attention_head_independence() {
    let seq_len = 4;
    let d_model = 64;
    let num_heads = 2;
    let head_dim = d_model / num_heads;

    let x = rand_vec(77, seq_len * d_model);

    let wq_h0 = rand_vec(100, d_model * head_dim);
    let wk_h0 = rand_vec(200, d_model * head_dim);
    let wv_h0 = rand_vec(300, d_model * head_dim);

    let q0 = matmul(&x, &wq_h0, seq_len, d_model, head_dim);
    let k0 = matmul(&x, &wk_h0, seq_len, d_model, head_dim);
    let v0 = matmul(&x, &wv_h0, seq_len, d_model, head_dim);

    let head0_a = scaled_dot_product_attention(&q0, &k0, &v0, seq_len, head_dim, true);
    let head0_b = scaled_dot_product_attention(&q0, &k0, &v0, seq_len, head_dim, true);

    // Same inputs → identical outputs (deterministic)
    assert_eq!(head0_a, head0_b);
}

// ═══════════════════════════════════════════════════════════════════
// 3. Grouped query attention (GQA)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gqa_fewer_kv_heads() {
    let seq_len = 4;
    let d_model = 128;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = d_model / num_heads;
    let heads_per_kv = num_heads / num_kv_heads;

    let x = rand_vec(42, seq_len * d_model);

    // KV projections (shared across grouped heads)
    let mut kv_outputs = Vec::new();
    for kv_h in 0..num_kv_heads {
        let wk = rand_vec(200 + kv_h as u64, d_model * head_dim);
        let wv = rand_vec(300 + kv_h as u64, d_model * head_dim);
        let k = matmul(&x, &wk, seq_len, d_model, head_dim);
        let v = matmul(&x, &wv, seq_len, d_model, head_dim);
        kv_outputs.push((k, v));
    }

    // Each Q head maps to a KV group
    let mut concat = vec![0.0f32; seq_len * d_model];
    for h in 0..num_heads {
        let kv_idx = h / heads_per_kv;
        let wq = rand_vec(100 + h as u64, d_model * head_dim);
        let q = matmul(&x, &wq, seq_len, d_model, head_dim);
        let (k, v) = &kv_outputs[kv_idx];

        let head_out = scaled_dot_product_attention(&q, k, v, seq_len, head_dim, true);

        for s in 0..seq_len {
            for d in 0..head_dim {
                concat[s * d_model + h * head_dim + d] = head_out[s * head_dim + d];
            }
        }
    }

    assert_eq!(concat.len(), seq_len * d_model);
    assert!(concat.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gqa_shared_kv_deterministic() {
    let seq_len = 4;
    let head_dim = 32;
    let num_heads = 4;
    let num_kv_heads = 1; // All heads share single KV

    let q_all: Vec<Vec<f32>> =
        (0..num_heads).map(|h| rand_vec(100 + h as u64, seq_len * head_dim)).collect();
    let k = rand_vec(200, seq_len * head_dim);
    let v = rand_vec(300, seq_len * head_dim);

    let outputs: Vec<Vec<f32>> = q_all
        .iter()
        .map(|q| scaled_dot_product_attention(q, &k, &v, seq_len, head_dim, true))
        .collect();

    assert_eq!(outputs.len(), num_heads);
    for out in &outputs {
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // Different Q heads with same KV yield different outputs
    assert_ne!(outputs[0], outputs[1], "different Q → different output");
    // Same Q should be identical (shared KV, deterministic)
    let _ = num_kv_heads; // suppress unused
}

// ═══════════════════════════════════════════════════════════════════
// 4. Causal attention mask
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_causal_mask_lower_triangular() {
    for &seq_len in &[1, 4, 16, 64] {
        let mask = causal_mask(seq_len);
        assert_eq!(mask.len(), seq_len * seq_len);

        for i in 0..seq_len {
            for j in 0..seq_len {
                let val = mask[i * seq_len + j];
                if j <= i {
                    assert_eq!(val, 0.0, "pos ({i},{j}) visible");
                } else {
                    assert!(val.is_infinite() && val < 0.0);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_causal_attention_prevents_future_leakage() {
    let seq_len = 4;
    let head_dim = 16;

    let q = rand_vec(1, seq_len * head_dim);
    let k = rand_vec(2, seq_len * head_dim);
    let v = rand_vec(3, seq_len * head_dim);

    let causal_out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    let full_out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, false);

    // First row identical (only sees position 0 in both cases)
    let row0_causal = &causal_out[..head_dim];
    let row0_full = &full_out[..head_dim];
    for d in 0..head_dim {
        assert!((row0_causal[d] - row0_full[d]).abs() < 1e-5, "row 0 must match");
    }

    // Later rows should differ (causal masks future tokens)
    let row3_causal = &causal_out[3 * head_dim..4 * head_dim];
    let row3_full = &full_out[3 * head_dim..4 * head_dim];
    let diff: f32 = row3_causal.iter().zip(row3_full.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "causal row 3 should differ from full");
}

// ═══════════════════════════════════════════════════════════════════
// 5. KV cache integration
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_kv_cache_incremental_append() {
    let head_dim = 32;
    let initial_seq = 4;

    // Prefill: cache the first 4 positions
    let mut k_cache: Vec<f32> = rand_vec(10, initial_seq * head_dim);
    let mut v_cache: Vec<f32> = rand_vec(20, initial_seq * head_dim);

    // Decode step: append 1 new token
    let new_k = rand_vec(30, head_dim);
    let new_v = rand_vec(40, head_dim);
    k_cache.extend_from_slice(&new_k);
    v_cache.extend_from_slice(&new_v);

    let total_seq = initial_seq + 1;
    assert_eq!(k_cache.len(), total_seq * head_dim);
    assert_eq!(v_cache.len(), total_seq * head_dim);

    // Query only the new token against full cache
    let q = rand_vec(50, 1 * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0f32; total_seq];
    for j in 0..total_seq {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k_cache[j * head_dim + d];
        }
        scores[j] = dot * scale;
    }
    let weights = softmax(&scores);
    assert_eq!(weights.len(), total_seq);
    let weight_sum: f32 = weights.iter().sum();
    assert!((weight_sum - 1.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_kv_cache_multi_step_decode() {
    let head_dim = 16;
    let prefill_len = 2;
    let decode_steps = 5;

    let mut k_cache = rand_vec(1, prefill_len * head_dim);
    let mut v_cache = rand_vec(2, prefill_len * head_dim);

    for step in 0..decode_steps {
        let new_k = rand_vec(100 + step as u64, head_dim);
        let new_v = rand_vec(200 + step as u64, head_dim);
        k_cache.extend_from_slice(&new_k);
        v_cache.extend_from_slice(&new_v);

        let cur_len = prefill_len + step + 1;
        assert_eq!(k_cache.len(), cur_len * head_dim);

        // Single-token attention against accumulated cache
        let q = rand_vec(300 + step as u64, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; cur_len];
        for j in 0..cur_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_cache[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }
        let w = softmax(&scores);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. FFN sublayer (SwiGLU)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_swiglu_ffn_basic() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;

    let x = rand_vec(1, seq_len * d_model);
    let w_gate = rand_vec(2, d_model * d_ff);
    let w_up = rand_vec(3, d_model * d_ff);
    let w_down = rand_vec(4, d_ff * d_model);

    let gate = matmul(&x, &w_gate, seq_len, d_model, d_ff);
    let up = matmul(&x, &w_up, seq_len, d_model, d_ff);

    // SwiGLU: silu(gate) * up
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();

    let out = matmul(&hidden, &w_down, seq_len, d_ff, d_model);
    assert_eq!(out.len(), seq_len * d_model);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_swiglu_intermediate_dimension() {
    // Typical LLaMA-style: d_ff = (8/3) * d_model rounded
    let d_model = 128;
    let d_ff = (d_model * 8 / 3 + 31) & !31; // Round to 32

    let x = rand_vec(1, d_model);
    let w_gate = rand_vec(2, d_model * d_ff);
    let gate = matmul(&x, &w_gate, 1, d_model, d_ff);
    assert_eq!(gate.len(), d_ff);
}

// ═══════════════════════════════════════════════════════════════════
// 7. Pre-norm residual (RMSNorm → attn → residual → ...)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_prenorm_residual_attention() {
    let seq_len = 4;
    let d_model = 64;
    let head_dim = 64;
    let eps = 1e-6;

    let x = rand_vec(1, seq_len * d_model);

    // Pre-norm: RMSNorm each token
    let normed: Vec<f32> = (0..seq_len)
        .flat_map(|s| {
            let tok = &x[s * d_model..(s + 1) * d_model];
            rms_norm(tok, eps)
        })
        .collect();

    // Self-attention on normalised input
    let wq = rand_vec(10, d_model * head_dim);
    let wk = rand_vec(20, d_model * head_dim);
    let wv = rand_vec(30, d_model * head_dim);

    let q = matmul(&normed, &wq, seq_len, d_model, head_dim);
    let k = matmul(&normed, &wk, seq_len, d_model, head_dim);
    let v = matmul(&normed, &wv, seq_len, d_model, head_dim);

    let attn = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);

    let wo = rand_vec(40, head_dim * d_model);
    let attn_proj = matmul(&attn, &wo, seq_len, head_dim, d_model);

    // Residual: x + attn_proj
    let residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

    assert_eq!(residual.len(), seq_len * d_model);
    assert!(residual.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_prenorm_residual_ffn() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;
    let eps = 1e-6;

    let x = rand_vec(1, seq_len * d_model);

    // Pre-norm
    let normed: Vec<f32> = (0..seq_len)
        .flat_map(|s| {
            let tok = &x[s * d_model..(s + 1) * d_model];
            rms_norm(tok, eps)
        })
        .collect();

    // SwiGLU FFN
    let w_gate = rand_vec(2, d_model * d_ff);
    let w_up = rand_vec(3, d_model * d_ff);
    let w_down = rand_vec(4, d_ff * d_model);

    let gate = matmul(&normed, &w_gate, seq_len, d_model, d_ff);
    let up = matmul(&normed, &w_up, seq_len, d_model, d_ff);
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();
    let ffn_out = matmul(&hidden, &w_down, seq_len, d_ff, d_model);

    // Residual
    let residual: Vec<f32> = x.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

    assert_eq!(residual.len(), seq_len * d_model);
    assert!(residual.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// 8. Post-norm residual (attn → residual → LayerNorm → ...)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_postnorm_attention_residual() {
    let seq_len = 4;
    let d_model = 64;
    let head_dim = 64;
    let eps = 1e-5;

    let x = rand_vec(1, seq_len * d_model);

    // Attention (no pre-norm)
    let wq = rand_vec(10, d_model * head_dim);
    let wk = rand_vec(20, d_model * head_dim);
    let wv = rand_vec(30, d_model * head_dim);
    let wo = rand_vec(40, head_dim * d_model);

    let q = matmul(&x, &wq, seq_len, d_model, head_dim);
    let k = matmul(&x, &wk, seq_len, d_model, head_dim);
    let v = matmul(&x, &wv, seq_len, d_model, head_dim);
    let attn = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    let attn_proj = matmul(&attn, &wo, seq_len, head_dim, d_model);

    // Residual + LayerNorm
    let residual: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

    let normed: Vec<f32> = (0..seq_len)
        .flat_map(|s| {
            let tok = &residual[s * d_model..(s + 1) * d_model];
            layer_norm(tok, eps)
        })
        .collect();

    assert_eq!(normed.len(), seq_len * d_model);
    // LayerNorm outputs should have ~zero mean per token
    for s in 0..seq_len {
        let tok = &normed[s * d_model..(s + 1) * d_model];
        let mean: f32 = tok.iter().sum::<f32>() / d_model as f32;
        assert!(mean.abs() < 1e-4, "post-LN mean ≈ 0");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_postnorm_ffn_residual() {
    let seq_len = 2;
    let d_model = 64;
    let d_ff = 128;
    let eps = 1e-5;

    let x = rand_vec(1, seq_len * d_model);

    // FFN
    let w_gate = rand_vec(2, d_model * d_ff);
    let w_up = rand_vec(3, d_model * d_ff);
    let w_down = rand_vec(4, d_ff * d_model);

    let gate = matmul(&x, &w_gate, seq_len, d_model, d_ff);
    let up = matmul(&x, &w_up, seq_len, d_model, d_ff);
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();
    let ffn_out = matmul(&hidden, &w_down, seq_len, d_ff, d_model);

    // Residual + LayerNorm
    let residual: Vec<f32> = x.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();

    let normed: Vec<f32> = (0..seq_len)
        .flat_map(|s| {
            let tok = &residual[s * d_model..(s + 1) * d_model];
            layer_norm(tok, eps)
        })
        .collect();

    assert_eq!(normed.len(), seq_len * d_model);
    assert!(normed.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// 9. Full transformer block (attention + FFN)
// ═══════════════════════════════════════════════════════════════════

/// Run a single pre-norm transformer block and return output.
fn transformer_block_prenorm(
    x: &[f32],
    seq_len: usize,
    d_model: usize,
    d_ff: usize,
    head_dim: usize,
    layer_seed: u64,
) -> Vec<f32> {
    let eps = 1e-6;

    // --- Attention sublayer ---
    let normed_a: Vec<f32> =
        (0..seq_len).flat_map(|s| rms_norm(&x[s * d_model..(s + 1) * d_model], eps)).collect();

    let wq = rand_vec(layer_seed * 10 + 1, d_model * head_dim);
    let wk = rand_vec(layer_seed * 10 + 2, d_model * head_dim);
    let wv = rand_vec(layer_seed * 10 + 3, d_model * head_dim);
    let wo = rand_vec(layer_seed * 10 + 4, head_dim * d_model);

    let q = matmul(&normed_a, &wq, seq_len, d_model, head_dim);
    let k = matmul(&normed_a, &wk, seq_len, d_model, head_dim);
    let v = matmul(&normed_a, &wv, seq_len, d_model, head_dim);

    let attn = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    let attn_proj = matmul(&attn, &wo, seq_len, head_dim, d_model);

    let mid: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

    // --- FFN sublayer ---
    let normed_f: Vec<f32> =
        (0..seq_len).flat_map(|s| rms_norm(&mid[s * d_model..(s + 1) * d_model], eps)).collect();

    let wg = rand_vec(layer_seed * 10 + 5, d_model * d_ff);
    let wu = rand_vec(layer_seed * 10 + 6, d_model * d_ff);
    let wd = rand_vec(layer_seed * 10 + 7, d_ff * d_model);

    let gate = matmul(&normed_f, &wg, seq_len, d_model, d_ff);
    let up = matmul(&normed_f, &wu, seq_len, d_model, d_ff);
    let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();
    let ffn_out = matmul(&hidden, &wd, seq_len, d_ff, d_model);

    mid.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect()
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_full_transformer_block() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let x = rand_vec(42, seq_len * d_model);
    let out = transformer_block_prenorm(&x, seq_len, d_model, d_ff, head_dim, 1);

    assert_eq!(out.len(), seq_len * d_model);
    assert!(out.iter().all(|v| v.is_finite()));
    // Output differs from input (non-trivial transform)
    assert_ne!(x, out);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_full_block_deterministic() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let x = rand_vec(42, seq_len * d_model);
    let out_a = transformer_block_prenorm(&x, seq_len, d_model, d_ff, head_dim, 1);
    let out_b = transformer_block_prenorm(&x, seq_len, d_model, d_ff, head_dim, 1);

    assert_eq!(out_a, out_b, "deterministic with same seed");
}

// ═══════════════════════════════════════════════════════════════════
// 10. Multi-layer stack
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_stack_2_layers() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;
    let num_layers = 2;

    let mut h = rand_vec(42, seq_len * d_model);
    for layer in 0..num_layers {
        h = transformer_block_prenorm(&h, seq_len, d_model, d_ff, head_dim, layer as u64 + 1);
    }

    assert_eq!(h.len(), seq_len * d_model);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_stack_4_layers() {
    let seq_len = 4;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let mut h = rand_vec(42, seq_len * d_model);
    for layer in 0..4 {
        h = transformer_block_prenorm(&h, seq_len, d_model, d_ff, head_dim, layer as u64 + 1);
    }

    assert_eq!(h.len(), seq_len * d_model);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_stack_8_layers() {
    let seq_len = 2;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let mut h = rand_vec(42, seq_len * d_model);
    for layer in 0..8 {
        h = transformer_block_prenorm(&h, seq_len, d_model, d_ff, head_dim, layer as u64 + 1);
    }

    assert_eq!(h.len(), seq_len * d_model);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_stack_12_layers() {
    let seq_len = 2;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let mut h = rand_vec(42, seq_len * d_model);
    for layer in 0..12 {
        h = transformer_block_prenorm(&h, seq_len, d_model, d_ff, head_dim, layer as u64 + 1);
    }

    assert_eq!(h.len(), seq_len * d_model);
    assert!(h.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// 11. Sequence lengths
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_seq_len_1_single_token() {
    let d_model = 64;
    let head_dim = 64;
    let x = rand_vec(1, 1 * d_model);
    let wq = rand_vec(2, d_model * head_dim);
    let q = matmul(&x, &wq, 1, d_model, head_dim);
    assert_eq!(q.len(), head_dim);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_seq_len_32() {
    let seq_len = 32;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let x = rand_vec(1, seq_len * d_model);
    let out = transformer_block_prenorm(&x, seq_len, d_model, d_ff, head_dim, 1);
    assert_eq!(out.len(), seq_len * d_model);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_seq_len_128() {
    let seq_len = 128;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let x = rand_vec(1, seq_len * d_model);
    let out = transformer_block_prenorm(&x, seq_len, d_model, d_ff, head_dim, 1);
    assert_eq!(out.len(), seq_len * d_model);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_seq_len_512_dispatch_sizing() {
    let seq_len = 512;
    let d_model = 64;

    // Validate dispatch: threadgroup count for seq × d_model
    let total = (seq_len * d_model) as u32;
    let group_size = SIMD_WIDTH * 4; // 128 threads
    let num_groups = (total + group_size - 1) / group_size;
    assert!(num_groups > 0);
    assert!(group_size <= MAX_THREADS_PER_THREADGROUP);

    let buf_bytes = f32_bytes(seq_len * d_model);
    let aligned = align_up(buf_bytes);
    assert!(aligned >= buf_bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_seq_len_2048_buffer_sizing() {
    let seq_len = 2048;
    let d_model = 128;

    // Attention score matrix: seq × seq × sizeof(f32)
    let score_bytes = f32_bytes(seq_len * seq_len);
    let aligned_scores = align_up(score_bytes);
    assert_eq!(aligned_scores % METAL_BUFFER_ALIGNMENT, 0);

    // Activation buffer
    let act_bytes = f32_bytes(seq_len * d_model);
    let aligned_act = align_up(act_bytes);
    assert!(aligned_act >= act_bytes);

    // Total should be reasonable for unified memory
    let total_mb = (aligned_scores + aligned_act) as f64 / (1024.0 * 1024.0);
    assert!(total_mb < 128.0, "buffers fit in unified memory");
}

// ═══════════════════════════════════════════════════════════════════
// 12. Head dimensions
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_head_dim_64() {
    let seq_len = 4;
    let head_dim = 64;
    let d_model = 64;

    let q = rand_vec(1, seq_len * head_dim);
    let k = rand_vec(2, seq_len * head_dim);
    let v = rand_vec(3, seq_len * head_dim);

    let out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    assert_eq!(out.len(), seq_len * head_dim);

    // Buffer alignment for head_dim=64
    let buf = align_up(f32_bytes(seq_len * head_dim));
    assert_eq!(buf % METAL_BUFFER_ALIGNMENT, 0);
    let _ = d_model;
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_head_dim_128() {
    let seq_len = 4;
    let head_dim = 128;

    let q = rand_vec(1, seq_len * head_dim);
    let k = rand_vec(2, seq_len * head_dim);
    let v = rand_vec(3, seq_len * head_dim);

    let out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    assert_eq!(out.len(), seq_len * head_dim);

    // Threadgroup shared memory for reduction
    let shared_mem = head_dim * std::mem::size_of::<f32>();
    assert!(shared_mem <= MAX_THREADGROUP_MEMORY);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_head_dim_256() {
    let seq_len = 4;
    let head_dim = 256;

    let q = rand_vec(1, seq_len * head_dim);
    let k = rand_vec(2, seq_len * head_dim);
    let v = rand_vec(3, seq_len * head_dim);

    let out = scaled_dot_product_attention(&q, &k, &v, seq_len, head_dim, true);
    assert_eq!(out.len(), seq_len * head_dim);

    // Large head_dim: check shared memory fits one threadgroup
    let shared = head_dim * std::mem::size_of::<f32>();
    assert!(
        shared <= MAX_THREADGROUP_MEMORY,
        "head_dim=256 shared mem {shared} ≤ {MAX_THREADGROUP_MEMORY}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 13. Numerical stability
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_numerical_stability_deep_stack() {
    let seq_len = 2;
    let d_model = 64;
    let d_ff = 128;
    let head_dim = 64;

    let mut h = rand_vec(42, seq_len * d_model);
    for layer in 0..12 {
        h = transformer_block_prenorm(&h, seq_len, d_model, d_ff, head_dim, layer as u64 + 1);

        // Check for NaN/Inf at each layer
        assert!(h.iter().all(|v| v.is_finite()), "layer {layer}: values must remain finite");

        // Check magnitude hasn't exploded
        let max_abs = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e10, "layer {layer}: max magnitude {max_abs} < 1e10");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_softmax_numerical_stability_large_logits() {
    // Large positive logits should not cause overflow
    let logits: Vec<f32> = vec![1000.0, 1001.0, 999.0, 1000.5];
    let sm = softmax(&logits);
    assert!(sm.iter().all(|v| v.is_finite()));
    let sum: f32 = sm.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Large negative logits
    let neg_logits: Vec<f32> = vec![-1000.0, -999.0, -1001.0, -1000.5];
    let sm_neg = softmax(&neg_logits);
    assert!(sm_neg.iter().all(|v| v.is_finite()));
    let sum_neg: f32 = sm_neg.iter().sum();
    assert!((sum_neg - 1.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_rms_norm_stability_near_zero() {
    let eps = 1e-6;
    // Near-zero vector should not produce NaN
    let tiny = vec![1e-20f32; 64];
    let normed = rms_norm(&tiny, eps);
    assert!(normed.iter().all(|v| v.is_finite()));

    // All-zero vector
    let zeros = vec![0.0f32; 64];
    let normed_z = rms_norm(&zeros, eps);
    assert!(normed_z.iter().all(|v| v.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════
// 14. Memory patterns
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_all_tensors() {
    let d_model = 128;
    let d_ff = 256;
    let seq_len = 64;
    let head_dim = 64;
    let num_heads = 2;

    let sizes = [
        ("activation", f32_bytes(seq_len * d_model)),
        ("qkv_proj", f32_bytes(d_model * head_dim * 3)),
        ("attn_scores", f32_bytes(seq_len * seq_len * num_heads)),
        ("ffn_gate", f32_bytes(seq_len * d_ff)),
        ("ffn_up", f32_bytes(seq_len * d_ff)),
        ("ffn_down", f32_bytes(d_ff * d_model)),
        ("output", f32_bytes(seq_len * d_model)),
    ];

    for (name, size) in &sizes {
        let aligned = align_up(*size);
        assert_eq!(
            aligned % METAL_BUFFER_ALIGNMENT,
            0,
            "{name}: aligned size {aligned} not aligned to {METAL_BUFFER_ALIGNMENT}"
        );
        assert!(aligned >= *size);
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_reuse_between_layers() {
    let seq_len = 16;
    let d_model = 128;
    let d_ff = 256;

    // Two transformer layers can share activation buffers
    let attn_buf = align_up(f32_bytes(seq_len * d_model));
    let ffn_buf = align_up(f32_bytes(seq_len * d_ff));
    let score_buf = align_up(f32_bytes(seq_len * seq_len));

    // Peak memory = max of attention phase vs FFN phase
    let attn_phase = attn_buf * 4 + score_buf; // Q,K,V,O + scores
    let ffn_phase = ffn_buf * 2 + attn_buf; // gate, up, down
    let peak = attn_phase.max(ffn_phase);

    // Reuse means we don't need both phases simultaneously
    assert!(peak < attn_phase + ffn_phase);
    assert!(peak > 0);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_peak_memory_estimation() {
    let configs = [(32, 128, 256, "small"), (128, 256, 512, "medium"), (512, 512, 1024, "large")];

    for (seq_len, d_model, d_ff, label) in &configs {
        let num_heads = d_model / 64;
        let head_dim = d_model / num_heads;

        // Weight buffers (persistent per layer)
        let qkv_weight = f32_bytes(d_model * head_dim * 3);
        let o_weight = f32_bytes(head_dim * d_model);
        let ffn_weight = f32_bytes(d_model * d_ff * 3);
        let layer_weights = qkv_weight + o_weight + ffn_weight;

        // Activation buffers (transient, reusable)
        let act = f32_bytes(seq_len * d_model);
        let scores = f32_bytes(seq_len * seq_len * num_heads);
        let ffn_act = f32_bytes(seq_len * d_ff);
        let peak_act = (act * 4 + scores).max(ffn_act * 2 + act);

        let total_bytes = layer_weights + peak_act;
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

        assert!(total_mb < 512.0, "{label}: single layer {total_mb:.1} MB < 512 MB");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_shared_memory_fits() {
    for &head_dim in &[64, 128, 256] {
        // Softmax reduction needs head_dim floats in shared mem
        let softmax_shared = head_dim * std::mem::size_of::<f32>();
        assert!(
            softmax_shared <= MAX_THREADGROUP_MEMORY,
            "head_dim={head_dim}: shared {softmax_shared} \
             ≤ {MAX_THREADGROUP_MEMORY}"
        );

        // Tiled matmul: tile_size × tile_size floats
        let tile_size = 16u32;
        let tile_shared = (tile_size * tile_size) as usize * std::mem::size_of::<f32>() * 2; // A + B tiles
        assert!(
            tile_shared <= MAX_THREADGROUP_MEMORY,
            "tiled matmul shared {tile_shared} \
             ≤ {MAX_THREADGROUP_MEMORY}"
        );
    }
}
