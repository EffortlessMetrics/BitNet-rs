//! Wave 34 property tests for `bitnet-inference`.
//!
//! 20 properties covering:
//! - Temperature scaling preserves ordering (monotonicity)
//! - Top-k always returns k or fewer tokens
//! - Sampling with seed 42 is deterministic
//! - Repetition penalty reduces repeated-token probabilities
//! - Greedy decoding picks argmax
//! - Context window bounds (seq_len ≤ max_seq_len)
//! - Logits processing: finite inputs → finite outputs
//! - Stop sequence detection correctness

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use bitnet_inference::SamplingConfig;
use bitnet_inference::context_window::ContextWindow;
use bitnet_logits::{apply_repetition_penalty, apply_temperature, apply_top_k, softmax_in_place};
use bitnet_sampling::{SamplingStrategy, greedy_sample};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn finite_logits(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-10.0f32..10.0, min_len..=max_len)
}

// ── 1. Temperature scaling preserves ordering ───────────────────────────────

proptest! {
    /// Multiplying logits by 1/temperature preserves the relative order of
    /// elements (monotone transform for temperature > 0).
    #[test]
    fn prop_temperature_preserves_ordering(
        logits in finite_logits(2, 64),
        temp in 0.01f32..5.0,
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        // For every pair (i, j), if logits[i] > logits[j] then scaled[i] > scaled[j].
        for i in 0..logits.len() {
            for j in (i + 1)..logits.len() {
                if logits[i] > logits[j] {
                    prop_assert!(scaled[i] > scaled[j],
                        "ordering violated at ({i},{j}): orig {} > {} but scaled {} <= {}",
                        logits[i], logits[j], scaled[i], scaled[j]);
                } else if logits[i] < logits[j] {
                    prop_assert!(scaled[i] < scaled[j]);
                }
            }
        }
    }

    /// Temperature = 1.0 is an identity operation.
    #[test]
    fn prop_temperature_one_is_identity(logits in finite_logits(1, 64)) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 1.0);
        prop_assert_eq!(scaled, logits);
    }

    /// Higher temperature flattens the distribution (max - min shrinks).
    #[test]
    fn prop_higher_temperature_flattens(
        logits in finite_logits(2, 32),
    ) {
        // Compare temp=0.5 (sharpen) vs temp=2.0 (flatten).
        let mut sharp = logits.clone();
        let mut flat = logits.clone();
        apply_temperature(&mut sharp, 0.5);
        apply_temperature(&mut flat, 2.0);
        let range_sharp = sharp.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - sharp.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_flat = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - flat.iter().cloned().fold(f32::INFINITY, f32::min);
        prop_assert!(range_flat <= range_sharp + 1e-5,
            "flat range {range_flat} should be <= sharp range {range_sharp}");
    }
}

// ── 2. Top-k returns k or fewer tokens ──────────────────────────────────────

proptest! {
    /// After top-k filtering, at most k entries remain finite.
    #[test]
    fn prop_top_k_returns_at_most_k(
        logits in finite_logits(4, 64),
        k in 1usize..=16,
    ) {
        let mut buf = logits.clone();
        apply_top_k(&mut buf, k);
        let finite_count = buf.iter().filter(|v| v.is_finite()).count();
        prop_assert!(finite_count <= k,
            "top-k({k}) left {finite_count} finite entries (expected ≤ {k})");
    }

    /// Top-k with k ≥ len is a no-op.
    #[test]
    fn prop_top_k_noop_when_k_ge_len(logits in finite_logits(1, 32)) {
        let mut buf = logits.clone();
        let n = buf.len();
        apply_top_k(&mut buf, n + 10);
        prop_assert_eq!(buf, logits);
    }
}

// ── 3. Deterministic sampling with seed ─────────────────────────────────────

proptest! {
    /// Two strategies seeded identically produce the same token.
    #[test]
    fn prop_seed_determinism(logits in finite_logits(4, 64)) {
        let cfg = SamplingConfig {
            temperature: 0.8,
            seed: Some(42),
            ..SamplingConfig::default()
        };
        let mut s1 = SamplingStrategy::new(cfg.clone());
        let mut s2 = SamplingStrategy::new(cfg);
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        prop_assert_eq!(t1, t2, "same seed must produce same token");
    }

    /// Different seeds (generally) produce different sequences over many draws.
    #[test]
    fn prop_different_seeds_diverge(logits in finite_logits(32, 64)) {
        let cfg1 = SamplingConfig { temperature: 1.0, seed: Some(1), ..SamplingConfig::default() };
        let cfg2 = SamplingConfig { temperature: 1.0, seed: Some(999), ..SamplingConfig::default() };
        let mut s1 = SamplingStrategy::new(cfg1);
        let mut s2 = SamplingStrategy::new(cfg2);
        let mut same = 0usize;
        for _ in 0..10 {
            let t1 = s1.sample(&logits, &[]).unwrap();
            let t2 = s2.sample(&logits, &[]).unwrap();
            if t1 == t2 { same += 1; }
        }
        // With 64+ tokens and different seeds, it's astronomically unlikely
        // that ALL 10 draws match. Allow up to 9 matches (conservative).
        prop_assert!(same < 10, "all 10 draws matched despite different seeds");
    }
}

// ── 4. Repetition penalty ───────────────────────────────────────────────────

proptest! {
    /// Repetition penalty reduces the probability of repeated tokens.
    #[test]
    fn prop_repetition_penalty_reduces_repeated(
        logits in proptest::collection::vec(0.1f32..5.0, 8..=32),
        penalty in 1.1f32..3.0,
        token_idx in 0usize..8,
    ) {
        let mut penalised = logits.clone();
        let token_id = token_idx as u32;
        apply_repetition_penalty(&mut penalised, &[token_id], penalty);
        // Positive logits are divided by penalty → smaller.
        if (token_idx) < logits.len() && logits[token_idx] > 0.0 {
            prop_assert!(penalised[token_idx] < logits[token_idx],
                "positive logit at {token_idx} should decrease");
        }
    }

    /// Repetition penalty=1.0 is a no-op.
    #[test]
    fn prop_repetition_penalty_one_noop(logits in finite_logits(4, 32)) {
        let mut penalised = logits.clone();
        let ids: Vec<u32> = (0..logits.len() as u32).collect();
        apply_repetition_penalty(&mut penalised, &ids, 1.0);
        for (i, (&orig, &pen)) in logits.iter().zip(penalised.iter()).enumerate() {
            prop_assert!((orig - pen).abs() < 1e-6,
                "penalty=1.0 changed logit[{i}]: {orig} → {pen}");
        }
    }
}

// ── 5. Greedy decoding picks argmax ─────────────────────────────────────────

proptest! {
    /// Greedy sample returns the index of the maximum logit.
    #[test]
    fn prop_greedy_picks_argmax(logits in finite_logits(2, 64)) {
        let token = greedy_sample(&logits).unwrap();
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(logits[token as usize], max_val,
            "greedy token {} has logit {} but max is {}",
            token, logits[token as usize], max_val);
    }

    /// Greedy with temperature=0.0 via SamplingStrategy also picks argmax.
    #[test]
    fn prop_greedy_via_strategy(logits in finite_logits(2, 64)) {
        let cfg = SamplingConfig {
            temperature: 0.0,
            seed: Some(0),
            ..SamplingConfig::default()
        };
        let mut strategy = SamplingStrategy::new(cfg);
        let token = strategy.sample(&logits, &[]).unwrap();
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(logits[token as usize], max_val);
    }
}

// ── 6. Context window bounds ────────────────────────────────────────────────

proptest! {
    /// Position never exceeds max_length after advance + truncate.
    #[test]
    fn prop_context_window_position_bound(
        max_len in 1usize..4096,
        advances in proptest::collection::vec(1usize..512, 1..10),
    ) {
        let mut cw = ContextWindow::new(max_len);
        for n in &advances {
            cw.advance(*n);
            cw.truncate_to_fit();
        }
        prop_assert!(cw.position <= cw.max_length,
            "position {} > max_length {}", cw.position, cw.max_length);
    }

    /// remaining() + position == max_length (when position ≤ max_length).
    #[test]
    fn prop_context_remaining_plus_position(
        max_len in 1usize..8192,
        advance in 0usize..8192,
    ) {
        let mut cw = ContextWindow::new(max_len);
        cw.advance(advance);
        cw.truncate_to_fit();
        prop_assert_eq!(
            cw.remaining() + cw.position, cw.max_length,
            "remaining({}) + position({}) != max_length({})",
            cw.remaining(), cw.position, cw.max_length
        );
    }

    /// After reset, position and prompt_end are both 0.
    #[test]
    fn prop_context_reset_zeroes(
        max_len in 1usize..4096,
        advance in 1usize..2048,
    ) {
        let mut cw = ContextWindow::new(max_len);
        cw.advance(advance);
        cw.mark_prompt_end();
        cw.reset();
        prop_assert_eq!(cw.position, 0);
        prop_assert_eq!(cw.prompt_end, 0);
    }
}

// ── 7. Logits processing: finite in → finite out ────────────────────────────

proptest! {
    /// Softmax of finite inputs produces finite, non-negative outputs.
    #[test]
    fn prop_softmax_finite_in_finite_out(logits in finite_logits(1, 64)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(p.is_finite() && p >= 0.0,
                "softmax output[{i}] = {p} is not finite/non-negative");
        }
    }

    /// Softmax outputs sum to ≈ 1.0.
    #[test]
    fn prop_softmax_sum_to_one(logits in finite_logits(1, 64)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5,
            "softmax sum = {sum}, expected ≈1.0");
    }

    /// Temperature then softmax still produces valid probabilities.
    #[test]
    fn prop_temperature_softmax_valid(
        logits in finite_logits(2, 64),
        temp in 0.1f32..5.0,
    ) {
        let mut buf = logits;
        apply_temperature(&mut buf, temp);
        softmax_in_place(&mut buf);
        let sum: f32 = buf.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4,
            "temp({temp}) + softmax sum = {sum}");
        for &p in &buf {
            prop_assert!(p >= 0.0 && p.is_finite());
        }
    }
}

// ── 8. Stop sequence detection ──────────────────────────────────────────────

proptest! {
    /// A stop-token ID is always detected.
    #[test]
    fn prop_stop_token_id_detected(stop_id in 0u32..1000) {
        let criteria = StopCriteria {
            stop_token_ids: vec![stop_id],
            ..StopCriteria::default()
        };
        let result = check_stop(&criteria, stop_id, &[], "");
        prop_assert_eq!(result, Some(StopReason::StopTokenId(stop_id)));
    }

    /// EOS token is detected when no explicit stop-token matches.
    #[test]
    fn prop_eos_detected(eos_id in 0u32..1000) {
        let criteria = StopCriteria {
            eos_token_id: Some(eos_id),
            ..StopCriteria::default()
        };
        let result = check_stop(&criteria, eos_id, &[], "");
        prop_assert_eq!(result, Some(StopReason::EosToken));
    }

    /// No trigger → None returned.
    #[test]
    fn prop_no_stop_returns_none(token in 500u32..600) {
        let criteria = StopCriteria {
            stop_token_ids: vec![9999],
            stop_strings: vec!["</s>".to_string()],
            max_tokens: 1000,
            eos_token_id: Some(9998),
        };
        let generated: Vec<u32> = (0..10).collect();
        let result = check_stop(&criteria, token, &generated, "hello world");
        prop_assert!(result.is_none(), "unexpected stop: {result:?}");
    }
}
