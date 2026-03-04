//! Property-based tests — wave 30 (inference).
//!
//! Covers: GenerationConfig builder invariants, InferenceConfig validation,
//! GenerationBudget tracking, BudgetTracker monotonicity, SamplingConfig
//! construction, SamplingStrategy determinism, temperature scaling, top-k/top-p
//! filtering, greedy sampling, and softmax normalization.
//!
//! 40+ property tests validating: config builder round-trips, budget tracking
//! invariants, sampling correctness, and generation config validation.

#![cfg(feature = "cpu")]

use std::time::Duration;

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::generation_budget::{BudgetTracker, GenerationBudget, StopReason};
use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_temperature, argmax, greedy_sample, softmax_in_place,
};
use proptest::prelude::*;

// ── Strategy helpers ────────────────────────────────────────────────────────

fn finite_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0, min_len..=max_len)
}

fn arb_sampling_config() -> impl Strategy<Value = SamplingConfig> {
    (
        0.0f32..2.0,          // temperature
        0u32..100,            // top_k
        0.01f32..1.0,         // top_p
        0.5f32..2.0,          // repetition_penalty
        any::<Option<u64>>(), // seed
    )
        .prop_map(|(temp, top_k, top_p, rep_pen, seed)| SamplingConfig {
            temperature: temp,
            top_k,
            top_p,
            repetition_penalty: rep_pen,
            seed,
        })
}

// ── GenerationConfig builder properties ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// with_max_tokens sets the value correctly.
    #[test]
    fn gen_config_max_tokens_builder(tokens in 1u32..10000) {
        let cfg = GenerationConfig::default().with_max_tokens(tokens);
        prop_assert_eq!(cfg.max_new_tokens, tokens);
    }

    /// with_temperature sets the value correctly.
    #[test]
    fn gen_config_temperature_builder(temp in 0.0f32..5.0) {
        let cfg = GenerationConfig::default().with_temperature(temp);
        prop_assert!((cfg.temperature - temp).abs() < f32::EPSILON);
    }

    /// with_top_k sets the value correctly.
    #[test]
    fn gen_config_top_k_builder(k in 0u32..200) {
        let cfg = GenerationConfig::default().with_top_k(k);
        prop_assert_eq!(cfg.top_k, k);
    }

    /// with_top_p sets the value correctly.
    #[test]
    fn gen_config_top_p_builder(p in 0.01f32..1.0) {
        let cfg = GenerationConfig::default().with_top_p(p);
        prop_assert!((cfg.top_p - p).abs() < f32::EPSILON);
    }

    /// with_repetition_penalty sets the value correctly.
    #[test]
    fn gen_config_rep_penalty_builder(pen in 0.1f32..5.0) {
        let cfg = GenerationConfig::default().with_repetition_penalty(pen);
        prop_assert!((cfg.repetition_penalty - pen).abs() < f32::EPSILON);
    }

    /// greedy() preset has temperature 0 and top_k 1.
    #[test]
    fn gen_config_greedy_invariants(_seed in 0u32..10) {
        let cfg = GenerationConfig::greedy();
        prop_assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
        prop_assert_eq!(cfg.top_k, 1);
    }

    /// creative() preset has temperature > 0.
    #[test]
    fn gen_config_creative_has_positive_temp(_seed in 0u32..10) {
        let cfg = GenerationConfig::creative();
        prop_assert!(cfg.temperature > 0.0);
    }

    /// balanced() preset has moderate temperature.
    #[test]
    fn gen_config_balanced_moderate_temp(_seed in 0u32..10) {
        let cfg = GenerationConfig::balanced();
        prop_assert!(cfg.temperature > 0.0);
        prop_assert!(cfg.temperature <= 1.0);
    }

    /// validate() accepts valid configs.
    #[test]
    fn gen_config_validate_valid(
        tokens in 1u32..1000,
        temp in 0.0f32..3.0,
        top_p in 0.01f32..1.0,
        rep_pen in 0.1f32..5.0,
    ) {
        let cfg = GenerationConfig::default()
            .with_max_tokens(tokens)
            .with_temperature(temp)
            .with_top_p(top_p)
            .with_repetition_penalty(rep_pen);
        prop_assert!(cfg.validate().is_ok());
    }

    /// validate() rejects zero max_tokens.
    #[test]
    fn gen_config_validate_zero_tokens(_seed in 0u32..10) {
        let cfg = GenerationConfig::default().with_max_tokens(0);
        prop_assert!(cfg.validate().is_err());
    }

    /// validate() rejects negative temperature.
    #[test]
    fn gen_config_validate_negative_temp(_seed in 0u32..10) {
        let cfg = GenerationConfig::default().with_temperature(-1.0);
        prop_assert!(cfg.validate().is_err());
    }

    /// validate() rejects zero repetition penalty.
    #[test]
    fn gen_config_validate_zero_rep_penalty(_seed in 0u32..10) {
        let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
        prop_assert!(cfg.validate().is_err());
    }

    /// Stop token IDs round-trip through builder.
    #[test]
    fn gen_config_stop_token_roundtrip(ids in prop::collection::vec(0u32..200000, 0..10)) {
        let cfg = GenerationConfig::default().with_stop_token_ids(ids.clone());
        prop_assert_eq!(&cfg.stop_token_ids, &ids);
    }

    /// is_stop_token returns true for configured stop tokens.
    #[test]
    fn gen_config_is_stop_token(ids in prop::collection::vec(1u32..200000, 1..5)) {
        let mut cfg = GenerationConfig::default().with_stop_token_ids(ids.clone());
        cfg.rebuild_stop_token_set();
        for &id in &ids {
            prop_assert!(cfg.is_stop_token(id));
        }
    }

    /// stop_sequences round-trip through builder.
    #[test]
    fn gen_config_stop_sequences_roundtrip(seqs in prop::collection::vec("[a-z]{1,10}", 0..5)) {
        let cfg = GenerationConfig::default().with_stop_sequences(seqs.clone());
        prop_assert_eq!(&cfg.stop_sequences, &seqs);
    }

    /// with_eos_token_id sets eos.
    #[test]
    fn gen_config_eos_builder(id in 0u32..200000) {
        let cfg = GenerationConfig::default().with_eos_token_id(Some(id));
        prop_assert_eq!(cfg.eos_token_id, Some(id));
    }
}

// ── InferenceConfig properties ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// cpu_optimized() has valid defaults.
    #[test]
    fn inference_config_cpu_optimized_valid(_seed in 0u32..10) {
        let cfg = InferenceConfig::cpu_optimized();
        prop_assert!(cfg.validate().is_ok());
    }

    /// gpu_optimized() has valid defaults.
    #[test]
    fn inference_config_gpu_optimized_valid(_seed in 0u32..10) {
        let cfg = InferenceConfig::gpu_optimized();
        prop_assert!(cfg.validate().is_ok());
    }

    /// memory_efficient() has valid defaults.
    #[test]
    fn inference_config_memory_efficient_valid(_seed in 0u32..10) {
        let cfg = InferenceConfig::memory_efficient();
        prop_assert!(cfg.validate().is_ok());
    }

    /// Builder chain produces valid config.
    #[test]
    fn inference_config_builder_chain(
        threads in 1usize..32,
        batch in 1usize..64,
        pool in 1usize..2_000_000_000,
    ) {
        let cfg = InferenceConfig::cpu_optimized()
            .with_threads(threads)
            .with_batch_size(batch)
            .with_memory_pool_size(pool);
        prop_assert!(cfg.validate().is_ok());
        prop_assert_eq!(cfg.num_threads, threads);
        prop_assert_eq!(cfg.batch_size, batch);
    }

    /// Default InferenceConfig has positive num_threads.
    #[test]
    fn inference_config_default_threads_positive(_seed in 0u32..10) {
        let cfg = InferenceConfig::cpu_optimized();
        prop_assert!(cfg.num_threads > 0);
    }
}

// ── GenerationBudget properties ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// New budget tracks zero tokens.
    #[test]
    fn budget_tracker_starts_at_zero(max_tokens in 1usize..1000) {
        let budget = GenerationBudget::new(max_tokens);
        let tracker = BudgetTracker::new(budget);
        prop_assert_eq!(tracker.tokens_generated(), 0);
        prop_assert_eq!(tracker.tokens_remaining(), max_tokens);
        prop_assert!(tracker.can_continue());
    }

    /// Recording tokens increments count.
    #[test]
    fn budget_tracker_increments(max_tokens in 2usize..100, n in 1usize..50) {
        let n = n.min(max_tokens - 1);
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        prop_assert_eq!(tracker.tokens_generated(), n);
        prop_assert_eq!(tracker.tokens_remaining(), max_tokens - n);
    }

    /// Tracker stops at max_tokens.
    #[test]
    fn budget_tracker_stops_at_max(max_tokens in 1usize..50) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..max_tokens {
            tracker.record_token();
        }
        prop_assert!(!tracker.can_continue());
        prop_assert_eq!(tracker.stop_reason(), Some(StopReason::MaxTokens));
    }

    /// EOS recording sets stop reason.
    #[test]
    fn budget_tracker_eos(max_tokens in 1usize..100) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_eos();
        prop_assert!(!tracker.can_continue());
        prop_assert_eq!(tracker.stop_reason(), Some(StopReason::EndOfSequence));
    }

    /// User stop recording sets stop reason.
    #[test]
    fn budget_tracker_user_stop(max_tokens in 1usize..100) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_user_stop();
        prop_assert!(!tracker.can_continue());
        prop_assert_eq!(tracker.stop_reason(), Some(StopReason::UserStop));
    }

    /// tokens_remaining + tokens_generated == max_tokens.
    #[test]
    fn budget_tracker_remaining_plus_generated(
        max_tokens in 2usize..100,
        n in 0usize..50,
    ) {
        let n = n.min(max_tokens - 1);
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        prop_assert_eq!(tracker.tokens_remaining() + tracker.tokens_generated(), max_tokens);
    }

    /// token_utilization is in [0.0, 1.0].
    #[test]
    fn budget_tracker_utilization_bounded(
        max_tokens in 1usize..100,
        n in 0usize..50,
    ) {
        let n = n.min(max_tokens);
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        let util = tracker.token_utilization();
        prop_assert!(util >= 0.0);
        prop_assert!(util <= 1.0);
    }

    /// Summary preserves tokens_generated.
    #[test]
    fn budget_summary_preserves_count(max_tokens in 1usize..50, n in 0usize..20) {
        let n = n.min(max_tokens);
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        let summary = tracker.summary();
        prop_assert_eq!(summary.tokens_generated, n);
        prop_assert_eq!(summary.max_tokens, max_tokens);
    }

    /// Unlimited budget allows many tokens.
    #[test]
    fn budget_unlimited_allows_many(n in 1usize..1000) {
        let budget = GenerationBudget::unlimited();
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        prop_assert!(tracker.can_continue());
    }

    /// elapsed is non-negative.
    #[test]
    fn budget_elapsed_non_negative(max_tokens in 1usize..100) {
        let budget = GenerationBudget::new(max_tokens);
        let tracker = BudgetTracker::new(budget);
        prop_assert!(tracker.elapsed() >= Duration::ZERO);
    }

    /// tokens_per_second is non-negative.
    #[test]
    fn budget_tps_non_negative(max_tokens in 1usize..100) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_token();
        prop_assert!(tracker.tokens_per_second() >= 0.0);
    }
}

// ── Sampling correctness properties ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// greedy_sample returns the argmax index.
    #[test]
    fn greedy_sample_is_argmax(logits in finite_f32_vec(2, 200)) {
        let expected = argmax(&logits) as u32;
        let result = greedy_sample(&logits);
        prop_assert!(result.is_ok());
        prop_assert_eq!(result.unwrap(), expected);
    }

    /// greedy_sample output is within vocab range.
    #[test]
    fn greedy_sample_within_range(logits in finite_f32_vec(2, 200)) {
        let result = greedy_sample(&logits).unwrap();
        prop_assert!((result as usize) < logits.len());
    }

    /// Temperature=0 sampling equals greedy.
    #[test]
    fn temp_zero_is_greedy(logits in finite_f32_vec(2, 200), seed in any::<u64>()) {
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let result = strategy.sample(&logits, &[]).unwrap();
        let expected = greedy_sample(&logits).unwrap();
        prop_assert_eq!(result, expected);
    }

    /// Sampling output is always within vocab range.
    #[test]
    fn sampling_within_vocab_range(
        logits in finite_f32_vec(2, 200),
        config in arb_sampling_config(),
    ) {
        let mut strategy = SamplingStrategy::new(config);
        let result = strategy.sample(&logits, &[]).unwrap();
        prop_assert!((result as usize) < logits.len());
    }

    /// Same seed produces deterministic output.
    #[test]
    fn sampling_seed_determinism(
        logits in finite_f32_vec(2, 100),
        seed in any::<u64>(),
        temp in 0.1f32..2.0,
    ) {
        let config1 = SamplingConfig {
            temperature: temp,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let config2 = config1.clone();
        let mut s1 = SamplingStrategy::new(config1);
        let mut s2 = SamplingStrategy::new(config2);
        let r1 = s1.sample(&logits, &[]).unwrap();
        let r2 = s2.sample(&logits, &[]).unwrap();
        prop_assert_eq!(r1, r2);
    }

    /// softmax output sums to ~1.0.
    #[test]
    fn softmax_sums_to_one(logits in finite_f32_vec(2, 200)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "softmax sum was {}", sum);
    }

    /// softmax output is non-negative.
    #[test]
    fn softmax_non_negative(logits in finite_f32_vec(2, 200)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(p >= 0.0);
        }
    }

    /// apply_temperature preserves argmax for positive temperature.
    #[test]
    fn temperature_preserves_argmax(logits in finite_f32_vec(2, 200), temp in 0.01f32..10.0) {
        let original_max = argmax(&logits);
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        let scaled_max = argmax(&scaled);
        prop_assert_eq!(original_max, scaled_max);
    }

    /// apply_temperature with temp=1.0 preserves values (identity).
    #[test]
    fn temperature_one_is_identity(logits in finite_f32_vec(2, 100)) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 1.0);
        for (a, b) in logits.iter().zip(scaled.iter()) {
            prop_assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    /// argmax output is within bounds.
    #[test]
    fn argmax_within_bounds(logits in finite_f32_vec(1, 200)) {
        let idx = argmax(&logits);
        prop_assert!((idx as usize) < logits.len());
    }

    /// argmax picks the actual maximum value.
    #[test]
    fn argmax_picks_max(logits in finite_f32_vec(1, 200)) {
        let idx = argmax(&logits) as usize;
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!((logits[idx] - max_val).abs() < f32::EPSILON);
    }
}

// ── StopReason properties ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// StopReason variants are distinct.
    #[test]
    fn stop_reason_variants_distinct(_seed in 0u32..10) {
        prop_assert_ne!(StopReason::MaxTokens, StopReason::EndOfSequence);
        prop_assert_ne!(StopReason::MaxTokens, StopReason::UserStop);
        prop_assert_ne!(StopReason::MaxTokens, StopReason::TimeLimit);
        prop_assert_ne!(StopReason::MaxTokens, StopReason::MemoryLimit);
        prop_assert_ne!(StopReason::EndOfSequence, StopReason::UserStop);
    }

    /// StopReason Debug formatting is non-empty.
    #[test]
    fn stop_reason_debug_non_empty(
        idx in 0usize..5,
    ) {
        let reasons = [
            StopReason::MaxTokens,
            StopReason::TimeLimit,
            StopReason::MemoryLimit,
            StopReason::EndOfSequence,
            StopReason::UserStop,
        ];
        let dbg = format!("{:?}", reasons[idx]);
        prop_assert!(!dbg.is_empty());
    }
}
