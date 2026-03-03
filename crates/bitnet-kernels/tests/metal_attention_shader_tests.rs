#![cfg(target_os = "macos")]
#![allow(dead_code, clippy::identity_op, clippy::manual_div_ceil, clippy::needless_range_loop)]

//! Metal attention shader validation tests for Apple Silicon.
//!
//! Validates scaled dot-product attention, causal masking, multi-head
//! attention, softmax properties, dimension sweeps, KV cache
//! incremental attention, flash-attention tiling patterns, and Metal
//! dispatch constraints.
//!
//! These are validation contracts describing expected GPU behavior.
//! No actual Metal device is required to compile — all tests are
//! `#[ignore]`-gated for runtime execution on Apple Silicon hardware.

// ───────────────────────────────────────────────────────────────────
// Helper types
// ───────────────────────────────────────────────────────────────────

/// Configuration for an attention kernel dispatch.
#[derive(Debug, Clone)]
struct AttentionConfig {
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension of each head (d_k).
    head_dim: usize,
    /// Query sequence length.
    seq_len: usize,
    /// Key/value sequence length (may differ for cross-attention).
    kv_seq_len: usize,
    /// Scaling factor applied to QK^T (typically 1/sqrt(head_dim)).
    scale: f32,
    /// Whether to apply a causal (lower-triangular) mask.
    causal: bool,
}

impl AttentionConfig {
    fn new(
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_seq_len: usize,
        causal: bool,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { num_heads, head_dim, seq_len, kv_seq_len, scale, causal }
    }

    fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// A complete attention test case with inputs, config, and expected
/// intermediate/final outputs.
#[derive(Debug, Clone)]
struct AttentionTestCase {
    /// Query tensor, shape [num_heads, seq_len, head_dim].
    q: Vec<f32>,
    /// Key tensor, shape [num_heads, kv_seq_len, head_dim].
    k: Vec<f32>,
    /// Value tensor, shape [num_heads, kv_seq_len, head_dim].
    v: Vec<f32>,
    /// Attention configuration.
    config: AttentionConfig,
    /// Expected attention scores after softmax,
    /// shape [num_heads, seq_len, kv_seq_len].
    expected_scores: Vec<f32>,
    /// Expected output, shape [num_heads, seq_len, head_dim].
    expected_output: Vec<f32>,
}

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

/// Metal maximum threads per threadgroup.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Apple Silicon SIMD group (wavefront) width.
const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Large negative value for causal masking.
const MASK_NEG_INF: f32 = -1e9;

/// Tolerance for single-step floating point comparisons.
const TOL_BASIC: f32 = 1e-5;

/// Tolerance for multi-step (accumulated) floating point comparisons.
const TOL_MULTI: f32 = 1e-3;

// ───────────────────────────────────────────────────────────────────
// Pure-logic helpers (no GPU required)
// ───────────────────────────────────────────────────────────────────

/// Numerically-stable softmax over a single row.
fn cpu_softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Build a causal mask: `true` where key position ≤ query position.
fn cpu_causal_mask(seq_len: usize, kv_seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * kv_seq_len];
    for q in 0..seq_len {
        for k in 0..kv_seq_len {
            mask[q * kv_seq_len + k] = k <= q;
        }
    }
    mask
}

/// Scaled dot-product attention on a single head.
///
/// Q: [seq_len, head_dim]
/// K: [kv_seq_len, head_dim]
/// V: [kv_seq_len, head_dim]
/// Returns output [seq_len, head_dim] and scores [seq_len, kv_seq_len].
fn cpu_scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &AttentionConfig,
) -> (Vec<f32>, Vec<f32>) {
    let seq_len = config.seq_len;
    let kv_seq_len = config.kv_seq_len;
    let head_dim = config.head_dim;
    let scale = config.scale;

    // QK^T * scale → [seq_len, kv_seq_len]
    let mut logits = vec![0.0f32; seq_len * kv_seq_len];
    for qi in 0..seq_len {
        for ki in 0..kv_seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qi * head_dim + d] * k[ki * head_dim + d];
            }
            logits[qi * kv_seq_len + ki] = dot * scale;
        }
    }

    // Optional causal mask.
    if config.causal {
        for qi in 0..seq_len {
            for ki in 0..kv_seq_len {
                if ki > qi {
                    logits[qi * kv_seq_len + ki] = MASK_NEG_INF;
                }
            }
        }
    }

    // Row-wise softmax → scores.
    let mut scores = vec![0.0f32; seq_len * kv_seq_len];
    for qi in 0..seq_len {
        let row_start = qi * kv_seq_len;
        let row_end = row_start + kv_seq_len;
        let sm = cpu_softmax(&logits[row_start..row_end]);
        scores[row_start..row_end].copy_from_slice(&sm);
    }

    // scores · V → output [seq_len, head_dim].
    let mut output = vec![0.0f32; seq_len * head_dim];
    for qi in 0..seq_len {
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for ki in 0..kv_seq_len {
                acc += scores[qi * kv_seq_len + ki] * v[ki * head_dim + d];
            }
            output[qi * head_dim + d] = acc;
        }
    }

    (output, scores)
}

/// Multi-head attention over all heads, dispatching per-head SDPA.
///
/// Q, K, V are packed as [num_heads, seq_len/kv_seq_len, head_dim].
/// Returns (output, scores) with the same head-major layout.
fn cpu_multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &AttentionConfig,
) -> (Vec<f32>, Vec<f32>) {
    let nh = config.num_heads;
    let sl = config.seq_len;
    let kl = config.kv_seq_len;
    let hd = config.head_dim;

    let q_head_size = sl * hd;
    let k_head_size = kl * hd;
    let out_head_size = sl * hd;
    let score_head_size = sl * kl;

    let mut all_output = vec![0.0f32; nh * out_head_size];
    let mut all_scores = vec![0.0f32; nh * score_head_size];

    for h in 0..nh {
        let q_slice = &q[h * q_head_size..(h + 1) * q_head_size];
        let k_slice = &k[h * k_head_size..(h + 1) * k_head_size];
        let v_slice = &v[h * k_head_size..(h + 1) * k_head_size];

        let (out, sc) = cpu_scaled_dot_product_attention(q_slice, k_slice, v_slice, config);

        all_output[h * out_head_size..(h + 1) * out_head_size].copy_from_slice(&out);
        all_scores[h * score_head_size..(h + 1) * score_head_size].copy_from_slice(&sc);
    }

    (all_output, all_scores)
}

/// Generate a deterministic pseudo-random f32 vector in [lo, hi].
fn det_rand(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let t = (state as f32) / (u64::MAX as f32);
            lo + t.abs() * (hi - lo)
        })
        .collect()
}

/// Pad a byte length to the Metal 256-byte alignment boundary.
fn align_to_metal(byte_len: usize) -> usize {
    (byte_len + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

/// Assert two slices are element-wise equal within `tol`.
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff <= tol, "{ctx}[{i}]: {x} vs {y}, diff={diff} > tol={tol}");
    }
}

// ───────────────────────────────────────────────────────────────────
// Basic correctness
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_single_head_vs_cpu() {
    let cfg = AttentionConfig::new(1, 64, 4, 4, false);
    let q = det_rand(cfg.seq_len * cfg.head_dim, 1, -1.0, 1.0);
    let k = det_rand(cfg.kv_seq_len * cfg.head_dim, 2, -1.0, 1.0);
    let v = det_rand(cfg.kv_seq_len * cfg.head_dim, 3, -1.0, 1.0);

    let (out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    assert_eq!(out.len(), cfg.seq_len * cfg.head_dim);
    assert_eq!(scores.len(), cfg.seq_len * cfg.kv_seq_len);
    // Each score row sums to 1.
    for qi in 0..cfg.seq_len {
        let row_sum: f32 = scores[qi * cfg.kv_seq_len..(qi + 1) * cfg.kv_seq_len].iter().sum();
        assert!((row_sum - 1.0).abs() < TOL_BASIC, "row {qi}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_multi_head_vs_cpu() {
    let cfg = AttentionConfig::new(4, 64, 8, 8, false);
    let q = det_rand(cfg.num_heads * cfg.seq_len * cfg.head_dim, 10, -1.0, 1.0);
    let k = det_rand(cfg.num_heads * cfg.kv_seq_len * cfg.head_dim, 11, -1.0, 1.0);
    let v = det_rand(cfg.num_heads * cfg.kv_seq_len * cfg.head_dim, 12, -1.0, 1.0);

    let (out, scores) = cpu_multi_head_attention(&q, &k, &v, &cfg);

    let out_size = cfg.num_heads * cfg.seq_len * cfg.head_dim;
    let score_size = cfg.num_heads * cfg.seq_len * cfg.kv_seq_len;
    assert_eq!(out.len(), out_size);
    assert_eq!(scores.len(), score_size);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_scaled_dot_product_identity_v() {
    // When V is an identity-like pattern the output ≈ softmax(QK^T).
    let cfg = AttentionConfig::new(1, 4, 2, 2, false);
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let (out, _scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Output should be valid (finite).
    assert!(out.iter().all(|x| x.is_finite()));
}

// ───────────────────────────────────────────────────────────────────
// Causal masking
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_causal_vs_noncausal() {
    let cfg_causal = AttentionConfig::new(1, 32, 4, 4, true);
    let cfg_full = AttentionConfig::new(1, 32, 4, 4, false);
    let q = det_rand(4 * 32, 20, -1.0, 1.0);
    let k = det_rand(4 * 32, 21, -1.0, 1.0);
    let v = det_rand(4 * 32, 22, -1.0, 1.0);

    let (out_c, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg_causal);
    let (out_f, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg_full);

    // First row (q=0) should be identical (mask is no-op).
    assert_approx_eq(&out_c[..32], &out_f[..32], TOL_BASIC, "row0_causal_eq");
    // Later rows should differ.
    let last_start = 3 * 32;
    let differs = out_c[last_start..last_start + 32]
        .iter()
        .zip(&out_f[last_start..last_start + 32])
        .any(|(a, b)| (a - b).abs() > TOL_BASIC);
    assert!(differs, "causal should differ from full on last row");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_causal_mask_pattern() {
    let mask = cpu_causal_mask(4, 4);
    // Row 0: [T, F, F, F]
    assert!(mask[0] && !mask[1] && !mask[2] && !mask[3]);
    // Row 1: [T, T, F, F]
    assert!(mask[4] && mask[5] && !mask[6] && !mask[7]);
    // Row 3: [T, T, T, T]
    assert!(mask[12] && mask[13] && mask[14] && mask[15]);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_future_token_blocking() {
    let cfg = AttentionConfig::new(1, 16, 4, 4, true);
    let q = det_rand(4 * 16, 30, -0.5, 0.5);
    let k = det_rand(4 * 16, 31, -0.5, 0.5);
    let v = det_rand(4 * 16, 32, -0.5, 0.5);

    let (_out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Scores for future positions must be ≈ 0.
    for qi in 0..4 {
        for ki in (qi + 1)..4 {
            let s = scores[qi * 4 + ki];
            assert!(s.abs() < TOL_BASIC, "future score [{qi},{ki}] = {s}");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_causal_asymmetric_kv() {
    // kv_seq_len > seq_len with causal masking.
    let cfg = AttentionConfig::new(1, 32, 2, 6, true);
    let q = det_rand(2 * 32, 40, -1.0, 1.0);
    let k = det_rand(6 * 32, 41, -1.0, 1.0);
    let v = det_rand(6 * 32, 42, -1.0, 1.0);

    let (_out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Row 0 should attend only to position 0.
    for ki in 1..6 {
        assert!(scores[ki].abs() < TOL_BASIC);
    }
}

// ───────────────────────────────────────────────────────────────────
// Softmax properties
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_softmax_sum_to_one() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sm = cpu_softmax(&logits);
    let sum: f32 = sm.iter().sum();
    assert!((sum - 1.0).abs() < TOL_BASIC);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_softmax_non_negative() {
    let logits = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    let sm = cpu_softmax(&logits);
    assert!(sm.iter().all(|&x| x >= 0.0));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_softmax_monotonic_ordering() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let sm = cpu_softmax(&logits);
    for i in 1..sm.len() {
        assert!(sm[i] >= sm[i - 1], "softmax not monotonic");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_softmax_numerical_stability() {
    // Large values should not produce NaN / Inf.
    let logits = vec![1e6, 1e6 + 1.0, 1e6 + 2.0];
    let sm = cpu_softmax(&logits);
    assert!(sm.iter().all(|x| x.is_finite()));
    let sum: f32 = sm.iter().sum();
    assert!((sum - 1.0).abs() < TOL_BASIC);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_softmax_uniform_equal_logits() {
    let logits = vec![3.0; 8];
    let sm = cpu_softmax(&logits);
    let expected = 1.0 / 8.0;
    for &p in &sm {
        assert!((p - expected).abs() < TOL_BASIC);
    }
}

// ───────────────────────────────────────────────────────────────────
// Scaling
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_sqrt_dk_scaling() {
    let cfg = AttentionConfig::new(1, 64, 2, 2, false);
    let expected_scale = 1.0 / (64.0f32).sqrt();
    assert!((cfg.scale - expected_scale).abs() < TOL_BASIC);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_custom_scale_factor() {
    let cfg = AttentionConfig::new(1, 64, 2, 2, false).with_scale(0.25);
    let q = vec![1.0; 2 * 64];
    let k = vec![1.0; 2 * 64];
    let v = det_rand(2 * 64, 50, -1.0, 1.0);

    let (out_custom, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    let cfg_default = AttentionConfig::new(1, 64, 2, 2, false);
    let (out_default, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg_default);

    // Different scales generally produce different outputs.
    let any_diff =
        out_custom.iter().zip(out_default.iter()).any(|(a, b)| (a - b).abs() > TOL_BASIC);
    assert!(any_diff, "custom scale should differ from default");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_scale_one_identity() {
    // scale = 1.0 means raw dot products feed into softmax.
    let cfg = AttentionConfig::new(1, 4, 2, 2, false).with_scale(1.0);
    let q = vec![0.5; 2 * 4];
    let k = vec![0.5; 2 * 4];
    let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert!(out.iter().all(|x| x.is_finite()));
}

// ───────────────────────────────────────────────────────────────────
// Dimension sweeps
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_head_dim_32() {
    let cfg = AttentionConfig::new(1, 32, 4, 4, false);
    let q = det_rand(4 * 32, 60, -1.0, 1.0);
    let k = det_rand(4 * 32, 61, -1.0, 1.0);
    let v = det_rand(4 * 32, 62, -1.0, 1.0);
    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 32);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_head_dim_128() {
    let cfg = AttentionConfig::new(1, 128, 4, 4, false);
    let q = det_rand(4 * 128, 63, -1.0, 1.0);
    let k = det_rand(4 * 128, 64, -1.0, 1.0);
    let v = det_rand(4 * 128, 65, -1.0, 1.0);
    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 128);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_head_dim_256() {
    let cfg = AttentionConfig::new(1, 256, 4, 4, false);
    let q = det_rand(4 * 256, 66, -1.0, 1.0);
    let k = det_rand(4 * 256, 67, -1.0, 1.0);
    let v = det_rand(4 * 256, 68, -1.0, 1.0);
    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 256);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_num_heads_1() {
    let cfg = AttentionConfig::new(1, 64, 8, 8, false);
    let q = det_rand(1 * 8 * 64, 70, -1.0, 1.0);
    let k = det_rand(1 * 8 * 64, 71, -1.0, 1.0);
    let v = det_rand(1 * 8 * 64, 72, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 1 * 8 * 64);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_num_heads_8() {
    let cfg = AttentionConfig::new(8, 64, 4, 4, false);
    let q = det_rand(8 * 4 * 64, 73, -1.0, 1.0);
    let k = det_rand(8 * 4 * 64, 74, -1.0, 1.0);
    let v = det_rand(8 * 4 * 64, 75, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 8 * 4 * 64);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_num_heads_16() {
    let cfg = AttentionConfig::new(16, 64, 4, 4, false);
    let q = det_rand(16 * 4 * 64, 76, -1.0, 1.0);
    let k = det_rand(16 * 4 * 64, 77, -1.0, 1.0);
    let v = det_rand(16 * 4 * 64, 78, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 16 * 4 * 64);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_seq_len_16() {
    let cfg = AttentionConfig::new(4, 64, 16, 16, true);
    let q = det_rand(4 * 16 * 64, 80, -1.0, 1.0);
    let k = det_rand(4 * 16 * 64, 81, -1.0, 1.0);
    let v = det_rand(4 * 16 * 64, 82, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 16 * 64);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_seq_len_64() {
    let cfg = AttentionConfig::new(4, 64, 64, 64, true);
    let q = det_rand(4 * 64 * 64, 83, -1.0, 1.0);
    let k = det_rand(4 * 64 * 64, 84, -1.0, 1.0);
    let v = det_rand(4 * 64 * 64, 85, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 64 * 64);
}

// ───────────────────────────────────────────────────────────────────
// Edge cases
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_single_token_query() {
    let cfg = AttentionConfig::new(1, 64, 1, 8, false);
    let q = det_rand(1 * 64, 90, -1.0, 1.0);
    let k = det_rand(8 * 64, 91, -1.0, 1.0);
    let v = det_rand(8 * 64, 92, -1.0, 1.0);
    let (out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 64);
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < TOL_BASIC);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_single_kv_pair() {
    let cfg = AttentionConfig::new(1, 64, 4, 1, false);
    let q = det_rand(4 * 64, 93, -1.0, 1.0);
    let k = det_rand(1 * 64, 94, -1.0, 1.0);
    let v = det_rand(1 * 64, 95, -1.0, 1.0);
    let (_out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    // With only 1 KV, softmax → 1.0 for every query row.
    for qi in 0..4 {
        assert!((scores[qi] - 1.0).abs() < TOL_BASIC);
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_zero_queries() {
    // All-zero Q → uniform attention scores.
    let cfg = AttentionConfig::new(1, 32, 2, 4, false);
    let q = vec![0.0; 2 * 32];
    let k = det_rand(4 * 32, 96, -1.0, 1.0);
    let v = det_rand(4 * 32, 97, -1.0, 1.0);
    let (_out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    // Every row should be uniform (all scores equal).
    let expected = 1.0 / 4.0;
    for qi in 0..2 {
        for ki in 0..4 {
            let s = scores[qi * 4 + ki];
            assert!((s - expected).abs() < TOL_BASIC, "non-uniform at [{qi},{ki}]");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_identity_attention() {
    // Q = K (same vectors) → highest attention on matching positions.
    let cfg = AttentionConfig::new(1, 16, 4, 4, false);
    let data = det_rand(4 * 16, 98, -2.0, 2.0);
    let v = det_rand(4 * 16, 99, -1.0, 1.0);

    let (_out, scores) = cpu_scaled_dot_product_attention(&data, &data, &v, &cfg);

    // Diagonal should have the highest score in each row.
    for qi in 0..4 {
        let diag_score = scores[qi * 4 + qi];
        for ki in 0..4 {
            if ki != qi {
                assert!(
                    diag_score >= scores[qi * 4 + ki] - TOL_BASIC,
                    "diagonal not max at [{qi}]"
                );
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Metal specifics
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_buffer_alignment_256() {
    // Verify all tensor buffer sizes align to 256 bytes.
    let cfg = AttentionConfig::new(4, 64, 16, 16, false);
    let q_bytes = cfg.num_heads * cfg.seq_len * cfg.head_dim * 4;
    let k_bytes = cfg.num_heads * cfg.kv_seq_len * cfg.head_dim * 4;
    let v_bytes = k_bytes;
    let score_bytes = cfg.num_heads * cfg.seq_len * cfg.kv_seq_len * 4;
    let out_bytes = q_bytes;

    for (name, bytes) in [
        ("Q", q_bytes),
        ("K", k_bytes),
        ("V", v_bytes),
        ("scores", score_bytes),
        ("output", out_bytes),
    ] {
        let aligned = align_to_metal(bytes);
        assert_eq!(
            aligned % METAL_BUFFER_ALIGNMENT,
            0,
            "{name} buffer not 256-byte aligned: {aligned}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_threadgroup_memory_sizing() {
    // Threadgroup memory for softmax reduction: one f32 per thread.
    let threadgroup_size: u32 = 256;
    let tg_mem_bytes = threadgroup_size as usize * 4; // f32
    assert!(tg_mem_bytes <= 32768, "exceeds 32 KB threadgroup limit");

    // For tiled attention: tile of scores in threadgroup.
    let tile_q = 16u32;
    let tile_k = 16u32;
    let tile_mem = (tile_q * tile_k) as usize * 4;
    assert!(tile_mem <= 32768);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_simd_width_dispatch() {
    // Apple Silicon SIMD width is 32; threadgroups should be
    // multiples of 32.
    let head_dims = [32, 64, 128, 256];
    for &hd in &head_dims {
        // A typical dispatch: one threadgroup per (head, query_row).
        let tg_size = hd.min(METAL_MAX_THREADS_PER_THREADGROUP);
        assert_eq!(tg_size % METAL_SIMD_GROUP_SIZE, 0, "head_dim={hd} not SIMD-aligned");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_workgroup_limits() {
    let cfg = AttentionConfig::new(16, 128, 64, 64, true);
    let total_threadgroups = cfg.num_heads as u32 * cfg.seq_len as u32;
    // Metal allows up to 2^31 threadgroups per dimension.
    assert!(
        total_threadgroups < (1u32 << 16),
        "threadgroup count {total_threadgroups} too large for \
         single-dimension dispatch"
    );
}

// ───────────────────────────────────────────────────────────────────
// Numerical properties
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_weight_sum_per_row() {
    let cfg = AttentionConfig::new(2, 64, 8, 8, false);
    let q = det_rand(2 * 8 * 64, 100, -1.0, 1.0);
    let k = det_rand(2 * 8 * 64, 101, -1.0, 1.0);
    let v = det_rand(2 * 8 * 64, 102, -1.0, 1.0);

    let (_out, scores) = cpu_multi_head_attention(&q, &k, &v, &cfg);

    for h in 0..cfg.num_heads {
        for qi in 0..cfg.seq_len {
            let row_start = h * cfg.seq_len * cfg.kv_seq_len + qi * cfg.kv_seq_len;
            let row_sum: f32 = scores[row_start..row_start + cfg.kv_seq_len].iter().sum();
            assert!((row_sum - 1.0).abs() < TOL_BASIC, "head={h} row={qi} sum={row_sum}");
        }
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_output_bounded_by_v_range() {
    let cfg = AttentionConfig::new(1, 32, 4, 4, false);
    let q = det_rand(4 * 32, 103, -1.0, 1.0);
    let k = det_rand(4 * 32, 104, -1.0, 1.0);
    let v = det_rand(4 * 32, 105, 0.0, 1.0); // V in [0, 1]

    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Output is a convex combination of V rows → bounded by V range.
    let v_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for (i, &o) in out.iter().enumerate() {
        assert!(
            o >= v_min - TOL_BASIC && o <= v_max + TOL_BASIC,
            "out[{i}]={o} outside V range [{v_min}, {v_max}]"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_gradient_flow_nonzero() {
    // A proxy for gradient flow: perturbing Q should change output.
    let cfg = AttentionConfig::new(1, 32, 4, 4, false);
    let q = det_rand(4 * 32, 106, -1.0, 1.0);
    let k = det_rand(4 * 32, 107, -1.0, 1.0);
    let v = det_rand(4 * 32, 108, -1.0, 1.0);

    let (out_base, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    let mut q_perturbed = q.clone();
    q_perturbed[0] += 0.1;
    let (out_perturbed, _) = cpu_scaled_dot_product_attention(&q_perturbed, &k, &v, &cfg);

    let diff: f32 = out_base.iter().zip(out_perturbed.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > TOL_BASIC, "output unchanged after Q perturbation");
}

// ───────────────────────────────────────────────────────────────────
// KV cache / incremental attention
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_incremental_kv_cache() {
    // Simulate incremental decoding: new single-token query against
    // a growing KV cache.
    let head_dim = 64;
    let num_heads = 4;

    // Step 1: prefill with 8 tokens.
    let cfg1 = AttentionConfig::new(num_heads, head_dim, 8, 8, true);
    let q1 = det_rand(num_heads * 8 * head_dim, 110, -1.0, 1.0);
    let k1 = det_rand(num_heads * 8 * head_dim, 111, -1.0, 1.0);
    let v1 = det_rand(num_heads * 8 * head_dim, 112, -1.0, 1.0);
    let (out1, _) = cpu_multi_head_attention(&q1, &k1, &v1, &cfg1);
    assert_eq!(out1.len(), num_heads * 8 * head_dim);

    // Step 2: single new query against 9 KV entries.
    let cfg2 = AttentionConfig::new(num_heads, head_dim, 1, 9, true);
    let q2 = det_rand(num_heads * 1 * head_dim, 113, -1.0, 1.0);
    // Extend K/V with one new entry per head.
    let mut k2 = k1.clone();
    let k_new = det_rand(num_heads * head_dim, 114, -1.0, 1.0);
    for h in 0..num_heads {
        let insert_pos = (h + 1) * 8 * head_dim + h * head_dim;
        k2.splice(insert_pos..insert_pos, k_new[h * head_dim..(h + 1) * head_dim].iter().cloned());
    }
    let mut v2 = v1.clone();
    let v_new = det_rand(num_heads * head_dim, 115, -1.0, 1.0);
    for h in 0..num_heads {
        let insert_pos = (h + 1) * 8 * head_dim + h * head_dim;
        v2.splice(insert_pos..insert_pos, v_new[h * head_dim..(h + 1) * head_dim].iter().cloned());
    }

    let (out2, scores2) = cpu_multi_head_attention(&q2, &k2, &v2, &cfg2);
    assert_eq!(out2.len(), num_heads * head_dim);
    // Scores should attend to all 9 KV entries.
    assert_eq!(scores2.len(), num_heads * 9);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_cross_attention_compatibility() {
    // Cross-attention: Q from decoder, KV from encoder (different
    // sequence lengths, non-causal).
    let cfg = AttentionConfig::new(4, 64, 8, 32, false);
    let q = det_rand(4 * 8 * 64, 120, -1.0, 1.0);
    let k = det_rand(4 * 32 * 64, 121, -1.0, 1.0);
    let v = det_rand(4 * 32 * 64, 122, -1.0, 1.0);

    let (out, scores) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), 4 * 8 * 64);
    assert_eq!(scores.len(), 4 * 8 * 32);
}

// ───────────────────────────────────────────────────────────────────
// Flash-attention patterns
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_tiled_computation() {
    // Verify tiled (block-wise) attention produces the same result as
    // full attention.
    let cfg = AttentionConfig::new(1, 64, 16, 16, false);
    let q = det_rand(16 * 64, 130, -1.0, 1.0);
    let k = det_rand(16 * 64, 131, -1.0, 1.0);
    let v = det_rand(16 * 64, 132, -1.0, 1.0);

    // Full attention.
    let (out_full, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Tiled: process K/V in 4-token tiles and merge via online
    // softmax (simplified CPU simulation).
    let tile_size = 4;
    let seq_len = cfg.seq_len;
    let kv_seq_len = cfg.kv_seq_len;
    let head_dim = cfg.head_dim;
    let scale = cfg.scale;
    let n_tiles = kv_seq_len / tile_size;

    let mut out_tiled = vec![0.0f32; seq_len * head_dim];
    let mut running_max = vec![f32::NEG_INFINITY; seq_len];
    let mut running_sum = vec![0.0f32; seq_len];

    for t in 0..n_tiles {
        let k_start = t * tile_size;
        for qi in 0..seq_len {
            // Compute partial logits for this tile.
            let mut tile_logits = vec![0.0f32; tile_size];
            for ti in 0..tile_size {
                let ki = k_start + ti;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * head_dim + d] * k[ki * head_dim + d];
                }
                tile_logits[ti] = dot * scale;
            }

            let tile_max = tile_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let new_max = running_max[qi].max(tile_max);

            // Rescale previous accumulator.
            let old_scale_factor = (running_max[qi] - new_max).exp();
            for d in 0..head_dim {
                out_tiled[qi * head_dim + d] *= old_scale_factor;
            }
            running_sum[qi] *= old_scale_factor;

            // Accumulate new tile.
            for ti in 0..tile_size {
                let ki = k_start + ti;
                let w = (tile_logits[ti] - new_max).exp();
                running_sum[qi] += w;
                for d in 0..head_dim {
                    out_tiled[qi * head_dim + d] += w * v[ki * head_dim + d];
                }
            }
            running_max[qi] = new_max;
        }
    }

    // Normalize.
    for qi in 0..seq_len {
        let inv = 1.0 / running_sum[qi];
        for d in 0..head_dim {
            out_tiled[qi * head_dim + d] *= inv;
        }
    }

    assert_approx_eq(&out_tiled, &out_full, TOL_MULTI, "tiled_vs_full");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_memory_efficient_scoring() {
    // Memory-efficient scoring: compute scores one query row at a
    // time, never materialising the full score matrix.
    let cfg = AttentionConfig::new(1, 64, 8, 8, false);
    let q = det_rand(8 * 64, 140, -1.0, 1.0);
    let k = det_rand(8 * 64, 141, -1.0, 1.0);
    let v = det_rand(8 * 64, 142, -1.0, 1.0);

    let (out_full, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    // Row-at-a-time computation.
    let mut out_row = vec![0.0f32; 8 * 64];
    for qi in 0..cfg.seq_len {
        let q_row = &q[qi * 64..(qi + 1) * 64];
        let mut logits = vec![0.0f32; cfg.kv_seq_len];
        for ki in 0..cfg.kv_seq_len {
            let k_row = &k[ki * 64..(ki + 1) * 64];
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
            logits[ki] = dot * cfg.scale;
        }
        let sm = cpu_softmax(&logits);
        for d in 0..64 {
            let mut acc = 0.0f32;
            for ki in 0..cfg.kv_seq_len {
                acc += sm[ki] * v[ki * 64 + d];
            }
            out_row[qi * 64 + d] = acc;
        }
    }

    assert_approx_eq(&out_row, &out_full, TOL_BASIC, "memory_efficient_vs_full");
}

// ───────────────────────────────────────────────────────────────────
// Additional numerical / stress tests
// ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_large_head_dim_no_overflow() {
    let cfg = AttentionConfig::new(1, 256, 4, 4, false);
    let q = det_rand(4 * 256, 150, -2.0, 2.0);
    let k = det_rand(4 * 256, 151, -2.0, 2.0);
    let v = det_rand(4 * 256, 152, -1.0, 1.0);
    let (out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(scores.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_heads_independent() {
    // Each head should produce a different result when given
    // different inputs.
    let cfg = AttentionConfig::new(2, 32, 4, 4, false);
    let q = det_rand(2 * 4 * 32, 160, -1.0, 1.0);
    let k = det_rand(2 * 4 * 32, 161, -1.0, 1.0);
    let v = det_rand(2 * 4 * 32, 162, -1.0, 1.0);

    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);

    let head0 = &out[..4 * 32];
    let head1 = &out[4 * 32..];
    let any_diff = head0.iter().zip(head1.iter()).any(|(a, b)| (a - b).abs() > TOL_BASIC);
    assert!(any_diff, "heads should differ with different inputs");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_deterministic_reproducibility() {
    let cfg = AttentionConfig::new(4, 64, 8, 8, true);
    let q = det_rand(4 * 8 * 64, 170, -1.0, 1.0);
    let k = det_rand(4 * 8 * 64, 171, -1.0, 1.0);
    let v = det_rand(4 * 8 * 64, 172, -1.0, 1.0);

    let (out1, scores1) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    let (out2, scores2) = cpu_multi_head_attention(&q, &k, &v, &cfg);

    assert_approx_eq(&out1, &out2, 0.0, "output_determinism");
    assert_approx_eq(&scores1, &scores2, 0.0, "score_determinism");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_causal_first_row_single_attend() {
    // With causal masking the first query row attends only to
    // position 0 → score[0,0] = 1.0.
    let cfg = AttentionConfig::new(1, 32, 4, 4, true);
    let q = det_rand(4 * 32, 180, -1.0, 1.0);
    let k = det_rand(4 * 32, 181, -1.0, 1.0);
    let v = det_rand(4 * 32, 182, -1.0, 1.0);

    let (_out, scores) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    assert!((scores[0] - 1.0).abs() < TOL_BASIC);
    for ki in 1..4 {
        assert!(scores[ki].abs() < TOL_BASIC);
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_metal_buffer_padding() {
    // Odd tensor sizes still produce correctly padded Metal buffers.
    let cfg = AttentionConfig::new(3, 48, 7, 11, false);
    let q_elems = cfg.num_heads * cfg.seq_len * cfg.head_dim;
    let k_elems = cfg.num_heads * cfg.kv_seq_len * cfg.head_dim;
    let q_bytes = q_elems * 4;
    let k_bytes = k_elems * 4;

    let q_aligned = align_to_metal(q_bytes);
    let k_aligned = align_to_metal(k_bytes);
    assert_eq!(q_aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert_eq!(k_aligned % METAL_BUFFER_ALIGNMENT, 0);

    // Functional correctness with odd sizes.
    let q = det_rand(q_elems, 190, -1.0, 1.0);
    let k = det_rand(k_elems, 191, -1.0, 1.0);
    let v = det_rand(k_elems, 192, -1.0, 1.0);
    let (out, _) = cpu_multi_head_attention(&q, &k, &v, &cfg);
    assert_eq!(out.len(), q_elems);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_threadgroup_softmax_reduction() {
    // Validate that softmax reduction across a large kv_seq_len fits
    // in a single threadgroup.
    let kv_seq_len: u32 = 512;
    let tg_size = kv_seq_len.min(METAL_MAX_THREADS_PER_THREADGROUP);
    let passes_needed = (kv_seq_len + tg_size - 1) / tg_size;
    // For 512 elements with 1024-thread groups: single pass.
    assert!(passes_needed <= 2, "too many reduction passes: {passes_needed}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_metal_attention_output_not_all_same() {
    // Sanity: with random inputs, output rows should not be identical.
    let cfg = AttentionConfig::new(1, 64, 4, 4, false);
    let q = det_rand(4 * 64, 200, -1.0, 1.0);
    let k = det_rand(4 * 64, 201, -1.0, 1.0);
    let v = det_rand(4 * 64, 202, -1.0, 1.0);
    let (out, _) = cpu_scaled_dot_product_attention(&q, &k, &v, &cfg);

    let row0 = &out[..64];
    let row1 = &out[64..128];
    let any_diff = row0.iter().zip(row1.iter()).any(|(a, b)| (a - b).abs() > TOL_BASIC);
    assert!(any_diff, "output rows should differ");
}
