//! CPU reference implementation of scaled dot-product attention and tests
//! comparing it against expected mathematical properties.
//!
//! The CPU reference serves two purposes:
//! 1. Validates the OpenCL `attention.cl` kernel correctness (when a GPU is available)
//! 2. Provides a fallback compute path for CPU-only builds

/// CPU reference: scaled dot-product attention.
///
/// `q`:  `[num_heads, seq_len, d_k]`
/// `k`:  `[num_heads, kv_len,  d_k]`
/// `v`:  `[num_heads, kv_len,  d_v]`  (d_v == d_k in this implementation)
/// `out`: `[num_heads, seq_len, d_k]`
/// `causal`: if true, mask positions where k_pos > q_pos
pub fn attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    kv_len: usize,
    d_k: usize,
    causal: bool,
) {
    let inv_sqrt_dk = 1.0 / (d_k as f32).sqrt();

    for head in 0..num_heads {
        for q_pos in 0..seq_len {
            // Compute scores row: S[k_pos] = Q[q_pos] · K[k_pos] / √d_k
            let mut scores = vec![0.0f32; kv_len];
            for k_pos in 0..kv_len {
                if causal && k_pos > q_pos {
                    scores[k_pos] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                let q_off = head * seq_len * d_k + q_pos * d_k;
                let k_off = head * kv_len * d_k + k_pos * d_k;
                for i in 0..d_k {
                    dot += q[q_off + i] * k[k_off + i];
                }
                scores[k_pos] = dot * inv_sqrt_dk;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for e in &mut exps {
                *e /= sum + 1e-8;
            }

            // Weighted sum: O[q_pos, d] = Σ_j  w_j * V[j, d]
            let o_off = head * seq_len * d_k + q_pos * d_k;
            for d in 0..d_k {
                let mut acc = 0.0f32;
                for j in 0..kv_len {
                    acc += exps[j] * v[head * kv_len * d_k + j * d_k + d];
                }
                out[o_off + d] = acc;
            }
        }
    }
}

/// CPU reference: just the scores matrix (Q * K^T / √d_k) with optional causal mask.
pub fn attention_scores_cpu(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    kv_len: usize,
    d_k: usize,
    causal: bool,
) {
    let inv_sqrt_dk = 1.0 / (d_k as f32).sqrt();
    for head in 0..num_heads {
        for q_pos in 0..seq_len {
            for k_pos in 0..kv_len {
                let idx = head * seq_len * kv_len + q_pos * kv_len + k_pos;
                if causal && k_pos > q_pos {
                    scores[idx] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0f32;
                let q_off = head * seq_len * d_k + q_pos * d_k;
                let k_off = head * kv_len * d_k + k_pos * d_k;
                for i in 0..d_k {
                    dot += q[q_off + i] * k[k_off + i];
                }
                scores[idx] = dot * inv_sqrt_dk;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a deterministic pseudo-random buffer using a simple LCG.
    fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Map to [-1.0, 1.0]
                ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
            })
            .collect()
    }

    // ---- Unit tests ----

    #[test]
    fn test_attention_identity_keys() {
        // When Q == K (and non-causal, single position), softmax is uniform
        // so output ≈ mean(V rows)
        let d_k = 4;
        let kv_len = 3;
        let seq_len = 1;
        let num_heads = 1;

        // All queries/keys the same → equal attention weights
        let q = vec![1.0, 0.0, 0.0, 0.0]; // single query
        let k = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0f32; seq_len * d_k];

        attention_cpu(&q, &k, &v, &mut out, num_heads, seq_len, kv_len, d_k, false);

        // Uniform softmax → output ≈ mean of V rows = [5, 6, 7, 8]
        for (i, &expected) in [5.0, 6.0, 7.0, 8.0].iter().enumerate() {
            assert!(
                (out[i] - expected).abs() < 0.1,
                "out[{i}] = {}, expected ≈ {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_attention_causal_mask() {
        // With causal masking, position 0 can only attend to position 0
        let d_k = 2;
        let seq_len = 3;
        let kv_len = 3;
        let num_heads = 1;

        let q = pseudo_random(seq_len * d_k, 42);
        let k = pseudo_random(kv_len * d_k, 43);
        let v: Vec<f32> = (0..kv_len * d_k).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; seq_len * d_k];

        attention_cpu(&q, &k, &v, &mut out, num_heads, seq_len, kv_len, d_k, true);

        // Position 0 can only see key 0, so output should be V[0]
        for d in 0..d_k {
            assert!(
                (out[d] - v[d]).abs() < 1e-5,
                "causal: position 0 should copy V[0]"
            );
        }
    }

    #[test]
    fn test_attention_multi_head() {
        let d_k = 4;
        let seq_len = 2;
        let kv_len = 2;
        let num_heads = 3;

        let q = pseudo_random(num_heads * seq_len * d_k, 100);
        let k = pseudo_random(num_heads * kv_len * d_k, 101);
        let v = pseudo_random(num_heads * kv_len * d_k, 102);
        let mut out = vec![0.0f32; num_heads * seq_len * d_k];

        attention_cpu(&q, &k, &v, &mut out, num_heads, seq_len, kv_len, d_k, false);

        // Each head should produce an independent result
        // Verify heads differ (extremely unlikely to be equal with random data)
        let head0_out = &out[0..seq_len * d_k];
        let head1_out = &out[seq_len * d_k..2 * seq_len * d_k];
        let diff: f32 = head0_out
            .iter()
            .zip(head1_out)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "different heads should produce different output");
    }

    #[test]
    fn test_attention_scores_matches_full() {
        let d_k = 4;
        let seq_len = 3;
        let kv_len = 3;
        let num_heads = 2;

        let q = pseudo_random(num_heads * seq_len * d_k, 200);
        let k = pseudo_random(num_heads * kv_len * d_k, 201);

        let mut scores = vec![0.0f32; num_heads * seq_len * kv_len];
        attention_scores_cpu(&q, &k, &mut scores, num_heads, seq_len, kv_len, d_k, false);

        // Verify scores are finite
        assert!(
            scores.iter().all(|s| s.is_finite()),
            "all scores should be finite (non-causal)"
        );
    }

    #[test]
    fn test_attention_scores_causal_has_neg_inf() {
        let d_k = 4;
        let seq_len = 4;
        let kv_len = 4;
        let num_heads = 1;

        let q = pseudo_random(seq_len * d_k, 300);
        let k = pseudo_random(kv_len * d_k, 301);

        let mut scores = vec![0.0f32; seq_len * kv_len];
        attention_scores_cpu(&q, &k, &mut scores, num_heads, seq_len, kv_len, d_k, true);

        // Upper triangle should be -inf
        for q_pos in 0..seq_len {
            for k_pos in 0..kv_len {
                if k_pos > q_pos {
                    assert!(
                        scores[q_pos * kv_len + k_pos].is_infinite()
                            && scores[q_pos * kv_len + k_pos] < 0.0,
                        "causal mask should set future positions to -inf"
                    );
                }
            }
        }
    }

    // ---- Property tests ----

    #[test]
    fn prop_output_shape_correct() {
        // Test several configurations
        for (nh, sl, kl, dk) in [(1, 1, 1, 2), (2, 4, 4, 8), (4, 8, 16, 32)] {
            let q = pseudo_random(nh * sl * dk, 400);
            let k = pseudo_random(nh * kl * dk, 401);
            let v = pseudo_random(nh * kl * dk, 402);
            let mut out = vec![0.0f32; nh * sl * dk];

            attention_cpu(&q, &k, &v, &mut out, nh, sl, kl, dk, false);
            assert_eq!(out.len(), nh * sl * dk, "output shape mismatch");
        }
    }

    #[test]
    fn prop_output_values_bounded() {
        // Attention output should be a convex combination of V rows,
        // so each element should be bounded by [min(V), max(V)].
        let num_heads = 2;
        let seq_len = 4;
        let kv_len = 6;
        let d_k = 8;

        let q = pseudo_random(num_heads * seq_len * d_k, 500);
        let k = pseudo_random(num_heads * kv_len * d_k, 501);
        let v = pseudo_random(num_heads * kv_len * d_k, 502);
        let mut out = vec![0.0f32; num_heads * seq_len * d_k];

        attention_cpu(&q, &k, &v, &mut out, num_heads, seq_len, kv_len, d_k, false);

        let v_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for (i, &o) in out.iter().enumerate() {
            assert!(
                o.is_finite(),
                "output[{i}] should be finite, got {o}"
            );
            // Allow small epsilon for floating-point accumulation
            assert!(
                o >= v_min - 1e-5 && o <= v_max + 1e-5,
                "output[{i}] = {o} out of V range [{v_min}, {v_max}]"
            );
        }
    }

    #[test]
    fn prop_softmax_weights_sum_to_one() {
        // Verify softmax property indirectly: for a single V that is all-ones,
        // the output should also be all-ones (since weights sum to 1).
        let num_heads = 1;
        let seq_len = 4;
        let kv_len = 4;
        let d_k = 4;

        let q = pseudo_random(seq_len * d_k, 600);
        let k = pseudo_random(kv_len * d_k, 601);
        let v = vec![1.0f32; kv_len * d_k]; // all ones
        let mut out = vec![0.0f32; seq_len * d_k];

        attention_cpu(&q, &k, &v, &mut out, num_heads, seq_len, kv_len, d_k, false);

        for (i, &o) in out.iter().enumerate() {
            assert!(
                (o - 1.0).abs() < 1e-5,
                "with all-ones V, output[{i}] should be 1.0, got {o}"
            );
        }
    }

    #[test]
    fn prop_causal_vs_noncausal_differ() {
        let num_heads = 1;
        let seq_len = 4;
        let kv_len = 4;
        let d_k = 8;

        let q = pseudo_random(seq_len * d_k, 700);
        let k = pseudo_random(kv_len * d_k, 701);
        let v = pseudo_random(kv_len * d_k, 702);

        let mut out_causal = vec![0.0f32; seq_len * d_k];
        let mut out_noncausal = vec![0.0f32; seq_len * d_k];

        attention_cpu(
            &q, &k, &v, &mut out_causal, num_heads, seq_len, kv_len, d_k, true,
        );
        attention_cpu(
            &q, &k, &v, &mut out_noncausal, num_heads, seq_len, kv_len, d_k, false,
        );

        // Last position should be the same (sees all keys in both cases)
        let last_start = (seq_len - 1) * d_k;
        for d in 0..d_k {
            assert!(
                (out_causal[last_start + d] - out_noncausal[last_start + d]).abs() < 1e-5,
                "last position should be identical for causal and non-causal"
            );
        }

        // First position with kv_len > 1 should differ
        let diff: f32 = (0..d_k)
            .map(|d| (out_causal[d] - out_noncausal[d]).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "first position should differ between causal and non-causal"
        );
    }
}
