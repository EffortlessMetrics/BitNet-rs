//! Property-based tests for GPU HAL components.
//!
//! Tests mathematical invariants, monotonicity, bounds,
//! and round-trip properties of GPU HAL operations.

#![allow(clippy::cast_precision_loss)]

use bitnet_gpu_hal::{
    EmbeddingTable, GenerationConfig, HalError, MemoryPool, StepOutcome, apply_repetition_penalty,
    apply_rope, apply_rope_inverse, apply_temperature, argmax, attention_forward,
    attention_output_shape, build_causal_mask, build_rope_tables, check_stop, compression_ratio,
    ffn_forward, rms_norm, softmax, ternary_dequantize, ternary_quantize, top_k,
};
use proptest::prelude::*;

// ── Helpers ───────────────────────────────────────────────────────────────

fn finite_logits(
    min: f32,
    max: f32,
    len_range: std::ops::Range<usize>,
) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..max, len_range)
        .prop_filter("need at least one finite value", |v| v.iter().any(|x| x.is_finite()))
}

fn even_dim() -> impl Strategy<Value = usize> {
    (1usize..=32).prop_map(|n| n * 2)
}

// ── Softmax properties ────────────────────────────────────────────────────

proptest! {
    /// ∀ logits, softmax(logits).sum() ≈ 1.0.
    #[test]
    fn softmax_sums_to_one(logits in finite_logits(-20.0, 20.0, 1..200)) {
        let mut probs = logits;
        softmax(&mut probs);
        let total: f32 = probs.iter().sum();
        prop_assert!(
            (total - 1.0).abs() < 1e-4,
            "softmax sum = {}, expected ≈1.0", total
        );
    }

    /// All softmax outputs are non-negative.
    #[test]
    fn softmax_all_non_negative(
        logits in finite_logits(-20.0, 20.0, 1..200)
    ) {
        let mut probs = logits;
        softmax(&mut probs);
        for &p in &probs {
            prop_assert!(p >= 0.0, "softmax produced negative value {}", p);
        }
    }

    /// ∀ logits, if a > b then softmax(a) > softmax(b).
    #[test]
    fn softmax_monotonic(
        base in finite_logits(-10.0, 10.0, 2..50)
    ) {
        let mut probs = base.clone();
        softmax(&mut probs);
        for i in 0..base.len() {
            for j in (i + 1)..base.len() {
                if (base[i] - base[j]).abs() > f32::EPSILON {
                    if base[i] > base[j] {
                        prop_assert!(
                            probs[i] >= probs[j],
                            "monotonicity: logit[{}]={} > logit[{}]={} \
                             but prob[{}]={} < prob[{}]={}",
                            i, base[i], j, base[j], i, probs[i], j, probs[j]
                        );
                    } else {
                        prop_assert!(
                            probs[j] >= probs[i],
                            "monotonicity: logit[{}]={} > logit[{}]={} \
                             but prob[{}]={} < prob[{}]={}",
                            j, base[j], i, base[i], j, probs[j], i, probs[i]
                        );
                    }
                }
            }
        }
    }
}

// ── Temperature properties ────────────────────────────────────────────────

proptest! {
    /// Temperature=0 always picks the argmax (greedy).
    #[test]
    fn temperature_zero_is_argmax(
        logits in finite_logits(-10.0, 10.0, 2..50)
    ) {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits
            .iter()
            .filter(|&&x| (x - max_val).abs() < f32::EPSILON)
            .count();
        prop_assume!(max_count == 1);

        let expected = argmax(&logits);
        let mut l = logits;
        apply_temperature(&mut l, 0.0);
        // With temp=0, logits are unmodified (greedy path).
        let actual = argmax(&l);
        prop_assert_eq!(expected, actual, "temp=0 changed argmax");
    }

    /// Temperature=1.0 is a strict no-op.
    #[test]
    fn temperature_one_is_noop(
        logits in finite_logits(-10.0, 10.0, 1..50)
    ) {
        let original = logits.clone();
        let mut l = logits;
        apply_temperature(&mut l, 1.0);
        prop_assert_eq!(l, original, "temp=1.0 mutated logits");
    }

    /// Temperature preserves argmax order.
    #[test]
    fn temperature_preserves_argmax(
        logits in finite_logits(-5.0, 5.0, 2..50),
        temperature in 0.01f32..5.0f32,
    ) {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits
            .iter()
            .filter(|&&x| (x - max_val).abs() < f32::EPSILON)
            .count();
        prop_assume!(max_count == 1);

        let expected = argmax(&logits);
        let mut l = logits;
        apply_temperature(&mut l, temperature);
        let actual = argmax(&l);
        prop_assert_eq!(expected, actual, "temperature changed argmax");
    }
}

// ── Top-k properties ──────────────────────────────────────────────────────

proptest! {
    /// After top-k, at most k values are finite (> -inf).
    #[test]
    fn top_k_reduces_candidates(
        logits in finite_logits(-5.0, 5.0, 2..100),
        k in 1usize..50,
    ) {
        let k_capped = k.min(logits.len());
        let mut l = logits;
        top_k(&mut l, k_capped);
        let kept = l.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        prop_assert!(kept <= k_capped, "kept={} > k={}", kept, k_capped);
    }

    /// Top-k with k=0 is a no-op.
    #[test]
    fn top_k_zero_is_noop(
        logits in finite_logits(-5.0, 5.0, 2..50)
    ) {
        let original = logits.clone();
        let mut l = logits;
        top_k(&mut l, 0);
        prop_assert_eq!(l, original, "top_k(0) mutated logits");
    }

    /// Top-k with k >= len is a no-op.
    #[test]
    fn top_k_ge_len_is_noop(
        logits in finite_logits(-5.0, 5.0, 2..50)
    ) {
        let original = logits.clone();
        let mut l = logits;
        let len = l.len();
        top_k(&mut l, len + 10);
        prop_assert_eq!(l, original, "top_k(len+10) mutated logits");
    }
}

// ── Repetition penalty properties ─────────────────────────────────────────

proptest! {
    /// Penalised positive logits have lower (or equal) score.
    #[test]
    fn repetition_penalty_reduces_score(
        base_logit in 0.1f32..10.0f32,
        penalty in 1.01f32..3.0f32,
        vocab_size in 2usize..50,
    ) {
        let mut logits = vec![0.0f32; vocab_size];
        logits[0] = base_logit;
        let original = logits[0];
        apply_repetition_penalty(&mut logits, &[0], penalty);
        prop_assert!(
            logits[0] <= original,
            "penalty={} increased logit: {} → {}",
            penalty, original, logits[0]
        );
    }

    /// Penalty=1.0 is a strict no-op.
    #[test]
    fn repetition_penalty_one_is_noop(
        logits in finite_logits(-5.0, 5.0, 2..50),
        token_ids in prop::collection::vec(0u32..50, 1..5),
    ) {
        let original = logits.clone();
        let mut l = logits;
        apply_repetition_penalty(&mut l, &token_ids, 1.0);
        prop_assert_eq!(l, original, "penalty=1.0 mutated logits");
    }
}

// ── Quantization properties ───────────────────────────────────────────────

proptest! {
    /// All ternary quantized values are in {-1, 0, +1}.
    #[test]
    fn ternary_values_in_range(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..200)
    ) {
        let q = ternary_quantize(&values);
        for &v in &q {
            prop_assert!(
                v == -1 || v == 0 || v == 1,
                "ternary value {} not in {{-1, 0, +1}}", v
            );
        }
    }

    /// Quantize → dequantize round-trip error is bounded.
    #[test]
    fn quantize_dequantize_bounded_error(
        values in prop::collection::vec(-5.0f32..5.0f32, 1..100)
    ) {
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        prop_assume!(max_abs > 0.001);

        let q = ternary_quantize(&values);
        let scale = max_abs;
        let reconstructed = ternary_dequantize(&q, scale);

        prop_assert_eq!(values.len(), reconstructed.len());
        for (i, (&orig, &recon)) in
            values.iter().zip(reconstructed.iter()).enumerate()
        {
            let err = (orig - recon).abs();
            prop_assert!(
                err <= max_abs + f32::EPSILON,
                "element {}: error {} > max_abs {}", i, err, max_abs
            );
        }
    }

    /// Compression ratio is always positive for non-empty inputs.
    #[test]
    fn compression_ratio_positive(n in 1usize..10_000) {
        let ratio = compression_ratio(n);
        prop_assert!(ratio > 0.0, "compression ratio {} <= 0", ratio);
    }

    /// Compression ratio is zero for empty input.
    #[test]
    fn compression_ratio_zero_for_empty(_seed in 0u32..10) {
        let ratio = compression_ratio(0);
        prop_assert!(ratio.abs() < f32::EPSILON);
    }
}

// ── Attention properties ──────────────────────────────────────────────────

proptest! {
    /// Attention output shape matches (seq_len, head_dim).
    #[test]
    fn attention_output_shape_matches(
        seq_len in 1usize..16,
        head_dim in even_dim(),
    ) {
        let (out_s, out_d) = attention_output_shape(seq_len, head_dim);
        prop_assert_eq!(out_s, seq_len);
        prop_assert_eq!(out_d, head_dim);
    }

    /// Causal mask upper-triangle is all NEG_INFINITY.
    #[test]
    fn causal_mask_zeros_future(seq_len in 1usize..32) {
        let mask = build_causal_mask(seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let val = mask[i * seq_len + j];
                if j > i {
                    prop_assert!(
                        val == f32::NEG_INFINITY,
                        "mask[{}][{}] should be -inf", i, j
                    );
                } else {
                    prop_assert!(
                        (val - 0.0).abs() < f32::EPSILON,
                        "mask[{}][{}] should be 0.0", i, j
                    );
                }
            }
        }
    }

    /// Attention forward output has the correct length.
    #[test]
    fn attention_forward_output_len(
        seq_len in 1usize..8,
        head_dim in even_dim(),
    ) {
        let n = seq_len * head_dim;
        let q = vec![1.0f32; n];
        let k = vec![1.0f32; n];
        let v = vec![1.0f32; n];
        let out = attention_forward(&q, &k, &v, seq_len, head_dim).unwrap();
        prop_assert_eq!(out.len(), n);
    }
}

// ── Embedding properties ──────────────────────────────────────────────────

proptest! {
    /// Valid IDs always succeed.
    #[test]
    fn lookup_within_bounds(
        vocab_size in 1u32..100,
        dim in 1usize..64,
        token_id in 0u32..100,
    ) {
        let table = EmbeddingTable::new(vocab_size, dim, 0.5);
        let id = token_id % vocab_size;
        let result = table.lookup(id);
        prop_assert!(result.is_ok(), "lookup({}) failed unexpectedly", id);
        prop_assert_eq!(result.unwrap().len(), dim);
    }

    /// Invalid IDs always error.
    #[test]
    fn lookup_out_of_bounds_fails(
        vocab_size in 1u32..100,
        dim in 1usize..32,
        offset in 0u32..50,
    ) {
        let table = EmbeddingTable::new(vocab_size, dim, 0.0);
        let bad_id = vocab_size + offset;
        let result = table.lookup(bad_id);
        let is_oob = matches!(result, Err(HalError::OutOfBounds { .. }));
        prop_assert!(
            is_oob,
            "expected OutOfBounds for id={}, vocab={}", bad_id, vocab_size
        );
    }

    /// Batch lookup output length = input length × dim.
    #[test]
    fn batch_lookup_length_matches(
        vocab_size in 1u32..50,
        dim in 1usize..32,
        ids in prop::collection::vec(0u32..50, 1..20),
    ) {
        let table = EmbeddingTable::new(vocab_size, dim, 1.0);
        let valid_ids: Vec<u32> = ids.iter().map(|&id| id % vocab_size).collect();
        let result = table.batch_lookup(&valid_ids).unwrap();
        prop_assert_eq!(result.len(), valid_ids.len() * dim);
    }
}

// ── RoPE properties ───────────────────────────────────────────────────────

proptest! {
    /// cos and sin values are bounded in [-1, 1].
    #[test]
    fn cos_sin_bounded(
        dim in even_dim(),
        seq_len in 1usize..32,
    ) {
        let (cos_t, sin_t) =
            build_rope_tables(dim, seq_len, 10_000.0).unwrap();
        for (i, &c) in cos_t.iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&c),
                "cos[{}]={} out of [-1,1]", i, c
            );
        }
        for (i, &s) in sin_t.iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&s),
                "sin[{}]={} out of [-1,1]", i, s
            );
        }
    }

    /// RoPE rotation preserves the L2 norm: ||RoPE(x)|| ≈ ||x||.
    #[test]
    fn rotation_preserves_norm(
        dim in even_dim(),
    ) {
        let half = dim / 2;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let (cos_t, sin_t) =
            build_rope_tables(dim, 1, 10_000.0).unwrap();
        let cos_row = &cos_t[..half];
        let sin_row = &sin_t[..half];

        let rotated = apply_rope(&x, cos_row, sin_row);

        let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_rot: f32 =
            rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
        prop_assert!(
            (norm_orig - norm_rot).abs() < 1e-4,
            "norm changed: {} → {}", norm_orig, norm_rot
        );
    }

    /// Inverse rotation recovers original: RoPE⁻¹(RoPE(x)) ≈ x.
    #[test]
    fn inverse_rotation_recovers(dim in even_dim()) {
        let half = dim / 2;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let (cos_t, sin_t) =
            build_rope_tables(dim, 1, 10_000.0).unwrap();
        let cos_row = &cos_t[..half];
        let sin_row = &sin_t[..half];

        let rotated = apply_rope(&x, cos_row, sin_row);
        let recovered = apply_rope_inverse(&rotated, cos_row, sin_row);

        for (i, (&orig, &rec)) in x.iter().zip(recovered.iter()).enumerate() {
            prop_assert!(
                (orig - rec).abs() < 1e-4,
                "element {}: {} ≠ {}", i, orig, rec
            );
        }
    }

    /// RoPE tables have correct length = seq_len × (dim/2).
    #[test]
    fn rope_table_dimensions(
        dim in even_dim(),
        seq_len in 1usize..32,
    ) {
        let (cos_t, sin_t) =
            build_rope_tables(dim, seq_len, 10_000.0).unwrap();
        let expected = seq_len * (dim / 2);
        prop_assert_eq!(cos_t.len(), expected);
        prop_assert_eq!(sin_t.len(), expected);
    }
}

// ── Transformer properties ────────────────────────────────────────────────

proptest! {
    /// RMS norm output shape equals input shape.
    #[test]
    fn rms_norm_output_shape(
        n in 1usize..100,
    ) {
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let weight = vec![1.0f32; n];
        let original_len = x.len();
        rms_norm(&mut x, &weight, 1e-5);
        prop_assert_eq!(x.len(), original_len);
    }

    /// RMS norm with unit weights produces unit RMS (approximately).
    #[test]
    fn rms_norm_unit_rms(
        values in prop::collection::vec(0.1f32..10.0f32, 2..50)
    ) {
        let n = values.len();
        let mut x = values;
        let weight = vec![1.0f32; n];
        rms_norm(&mut x, &weight, 1e-6);
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        prop_assert!(
            (rms - 1.0).abs() < 0.1,
            "RMS after norm = {}, expected ≈1.0", rms
        );
    }

    /// FFN output shape equals input shape.
    #[test]
    fn ffn_output_shape(
        n in 1usize..200,
    ) {
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let out = ffn_forward(&x);
        prop_assert_eq!(out.len(), n);
    }
}

// ── Generation properties ─────────────────────────────────────────────────

proptest! {
    /// Generation always stops at or before max_tokens.
    #[test]
    fn stop_at_max_tokens(
        max_tokens in 1usize..100,
        tokens in prop::collection::vec(3u32..1000, 1..200),
    ) {
        let config = GenerationConfig {
            max_tokens,
            eos_token_id: bitnet_gpu_hal::DEFAULT_EOS_TOKEN,
        };
        let mut stopped = false;
        for (i, &tok) in tokens.iter().enumerate() {
            let outcome = check_stop(tok, i + 1, &config);
            if outcome != StepOutcome::Continue {
                stopped = true;
                prop_assert!(
                    i < max_tokens || outcome == StepOutcome::Eos,
                    "stopped at step {} beyond max_tokens={}",
                    i + 1, max_tokens
                );
                break;
            }
        }
        if !stopped && tokens.len() >= max_tokens {
            prop_assert!(false, "did not stop after {} tokens", max_tokens);
        }
    }

    /// EOS token always triggers early stop.
    #[test]
    fn eos_stops_early(
        max_tokens in 10usize..100,
        prefix_len in 0usize..9,
    ) {
        let eos = bitnet_gpu_hal::DEFAULT_EOS_TOKEN;
        let config = GenerationConfig {
            max_tokens,
            eos_token_id: eos,
        };
        // Generate prefix_len non-EOS tokens, then EOS.
        let outcome = check_stop(eos, prefix_len + 1, &config);
        prop_assert_eq!(
            outcome,
            StepOutcome::Eos,
            "EOS at step {} did not stop",
            prefix_len + 1
        );
    }

    /// Non-EOS, non-max step returns Continue.
    #[test]
    fn non_eos_non_max_continues(
        token in 3u32..1000,
        step in 1usize..50,
    ) {
        let config = GenerationConfig {
            max_tokens: 100,
            eos_token_id: bitnet_gpu_hal::DEFAULT_EOS_TOKEN,
        };
        let outcome = check_stop(token, step, &config);
        prop_assert_eq!(outcome, StepOutcome::Continue);
    }
}

// ── Memory properties ─────────────────────────────────────────────────────

proptest! {
    /// Cannot allocate more than total budget.
    #[test]
    fn allocate_within_budget(
        total in 64usize..10_000,
        request in 1usize..20_000,
    ) {
        let mut pool = MemoryPool::new(total);
        let result = pool.allocate(request);
        if request <= total {
            prop_assert!(result.is_ok(), "allocation of {} <= {} failed", request, total);
        } else {
            let is_oom = matches!(result, Err(HalError::OutOfMemory { .. }));
            prop_assert!(
                is_oom,
                "allocation of {} > {} should fail", request, total
            );
        }
    }

    /// Deallocate increases available space.
    #[test]
    fn deallocate_frees_space(
        total in 128usize..10_000,
        alloc_pct in 10usize..90,
    ) {
        let mut pool = MemoryPool::new(total);
        let alloc_size = total * alloc_pct / 100;
        prop_assume!(alloc_size > 0 && alloc_size <= total);

        pool.allocate(alloc_size).unwrap();
        let before = pool.available();
        pool.deallocate(alloc_size);
        let after = pool.available();
        prop_assert!(
            after >= before,
            "available did not increase: {} → {}", before, after
        );
        prop_assert_eq!(after, before + alloc_size);
    }

    /// Pool used + available always equals total.
    #[test]
    fn pool_invariant_used_plus_available(
        total in 1usize..10_000,
        alloc_size in 0usize..5_000,
    ) {
        let mut pool = MemoryPool::new(total);
        let size = alloc_size.min(total);
        let _ = pool.allocate(size);
        prop_assert_eq!(
            pool.used() + pool.available(),
            pool.total(),
            "invariant violated: {} + {} ≠ {}",
            pool.used(),
            pool.available(),
            pool.total()
        );
    }

    /// Sequential allocations respect budget.
    #[test]
    fn sequential_allocations_respect_budget(
        total in 256usize..4096,
        sizes in prop::collection::vec(1usize..128, 1..10),
    ) {
        let mut pool = MemoryPool::new(total);
        let mut allocated = 0usize;
        for &size in &sizes {
            let result = pool.allocate(size);
            if allocated + size <= total {
                prop_assert!(result.is_ok());
                allocated += size;
            } else {
                let is_oom = matches!(result, Err(HalError::OutOfMemory { .. }));
                prop_assert!(is_oom, "sequential alloc should fail on budget overflow");
                break;
            }
        }
    }
}
