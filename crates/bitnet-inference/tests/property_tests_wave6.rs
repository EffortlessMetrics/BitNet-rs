//! Wave 6 property tests: inference pipeline invariants.
//!
//! Key invariants:
//! - Temperature > 0 preserves relative ordering of logits
//! - Top-k filtering keeps at most k non-NEG_INFINITY elements
//! - Config builder presets all pass validation
//! - Stop sequence detection is suffix-complete (appending the stop string triggers)
//! - Softmax output sums to ≈ 1.0 for finite inputs

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use bitnet_inference::config::GenerationConfig;
use bitnet_logits::{apply_temperature, apply_top_k, argmax, softmax_in_place};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategy helpers
// ---------------------------------------------------------------------------

/// Non-empty vec of finite logits in a reasonable range.
fn logit_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0f32, 2..=max_len)
}

// ---------------------------------------------------------------------------
// Properties: Temperature preserves relative ordering
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Applying temperature > 0 preserves which logit is largest (argmax is stable).
    #[test]
    fn prop_temperature_preserves_argmax(
        logits in logit_vec(32),
        temp in 0.01f32..5.0,
    ) {
        // Record argmax before temperature
        let argmax_before = argmax(&logits);

        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);

        let argmax_after = argmax(&scaled);

        // If there's a unique maximum, argmax must be preserved.
        // With ties, argmax may shift, so only check when max is unique.
        let max_val = logits[argmax_before];
        let is_unique_max = logits.iter().filter(|&&v| (v - max_val).abs() < 1e-7).count() == 1;
        if is_unique_max {
            prop_assert_eq!(argmax_after, argmax_before);
        }
    }

    /// Temperature scaling is reversible: applying T then 1/T ≈ identity.
    #[test]
    fn prop_temperature_roundtrip(
        logits in logit_vec(16),
        temp in 0.1f32..4.0,
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        apply_temperature(&mut scaled, 1.0 / temp);

        for (i, (&original, &restored)) in logits.iter().zip(scaled.iter()).enumerate() {
            prop_assert!(
                (original - restored).abs() < 1e-3,
                "roundtrip mismatch at [{i}]: original={original}, restored={restored}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: Top-k limits cardinality
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// apply_top_k returns at most k finite entries.
    #[test]
    fn prop_top_k_limits_cardinality(
        logits in logit_vec(32),
        k in 1usize..16,
    ) {
        let mut filtered = logits.clone();
        let kept = apply_top_k(&mut filtered, k);

        prop_assert!(
            kept <= k,
            "top_k({k}) kept {kept} entries, expected ≤ {k}"
        );

        // Count finite entries independently
        let finite_count = filtered.iter().filter(|v| v.is_finite()).count();
        prop_assert!(
            finite_count <= k,
            "finite entries {finite_count} > k={k}"
        );
    }

    /// apply_top_k with k ≥ len is a no-op (all entries remain).
    #[test]
    fn prop_top_k_noop_when_large(logits in logit_vec(16)) {
        let k = logits.len() + 1;
        let mut filtered = logits.clone();
        let kept = apply_top_k(&mut filtered, k);

        prop_assert_eq!(kept, logits.len(), "k > len should keep all entries");
        for (i, (&before, &after)) in logits.iter().zip(filtered.iter()).enumerate() {
            prop_assert!(
                (before - after).abs() < 1e-7,
                "entry [{i}] changed: {before} → {after}"
            );
        }
    }

    /// After top-k + softmax, probabilities sum to ≈ 1.0.
    #[test]
    fn prop_top_k_then_softmax_sums_to_one(
        logits in logit_vec(16),
        k in 1usize..8,
    ) {
        let mut filtered = logits;
        apply_top_k(&mut filtered, k);
        softmax_in_place(&mut filtered);

        let sum: f32 = filtered.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum {sum} ≠ 1.0 after top-k({k})"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: Config presets all pass validation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// All named presets produce configs that pass validate().
    #[test]
    fn prop_preset_configs_valid(_dummy in 0u8..3) {
        let presets = [
            GenerationConfig::greedy(),
            GenerationConfig::creative(),
            GenerationConfig::balanced(),
        ];
        for (i, cfg) in presets.iter().enumerate() {
            prop_assert!(
                cfg.validate().is_ok(),
                "preset {i} failed validation: {:?}",
                cfg.validate().err()
            );
        }
    }

    /// Builder methods preserve values and resulting config still validates.
    #[test]
    fn prop_builder_chain_validates(
        max_tokens in 1u32..4096,
        temp in 0.0f32..2.0,
        top_k in 1u32..200,
        seed in any::<u64>(),
    ) {
        let cfg = GenerationConfig::default()
            .with_max_tokens(max_tokens)
            .with_temperature(temp)
            .with_top_k(top_k)
            .with_seed(seed);

        prop_assert_eq!(cfg.max_new_tokens, max_tokens);
        prop_assert!((cfg.temperature - temp).abs() < 1e-7);
        prop_assert_eq!(cfg.top_k, top_k);
        prop_assert_eq!(cfg.seed, Some(seed));
        prop_assert!(cfg.validate().is_ok(), "config failed validation: {:?}", cfg.validate().err());
    }

    /// with_stop_sequences preserves all provided sequences.
    #[test]
    fn prop_stop_sequences_preserved(n in 0usize..5) {
        let seqs: Vec<String> = (0..n).map(|i| format!("stop_{i}")).collect();
        let cfg = GenerationConfig::default().with_stop_sequences(seqs.clone());
        prop_assert_eq!(cfg.stop_sequences.len(), n);
        for (i, s) in cfg.stop_sequences.iter().enumerate() {
            prop_assert_eq!(s, &seqs[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: Stop sequence detection is suffix-complete
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// A stop token ID is detected immediately.
    #[test]
    fn prop_stop_token_id_detected(token_id in 0u32..1000) {
        let criteria = StopCriteria {
            stop_token_ids: vec![token_id],
            ..Default::default()
        };
        let result = check_stop(&criteria, token_id, &[], "");
        prop_assert_eq!(result, Some(StopReason::StopTokenId(token_id)));
    }

    /// EOS token is detected when emitted.
    #[test]
    fn prop_eos_token_detected(eos_id in 1u32..1000) {
        let criteria = StopCriteria {
            eos_token_id: Some(eos_id),
            ..Default::default()
        };
        let result = check_stop(&criteria, eos_id, &[], "");
        prop_assert_eq!(result, Some(StopReason::EosToken));
    }

    /// Stop string detection: any decoded tail containing the stop string triggers.
    #[test]
    fn prop_stop_string_suffix_complete(
        prefix in "[a-z]{0,10}",
        stop in "[a-z]{1,5}",
        suffix in "[a-z]{0,5}",
    ) {
        let criteria = StopCriteria {
            stop_strings: vec![stop.clone()],
            ..Default::default()
        };
        let tail = format!("{prefix}{stop}{suffix}");
        let result = check_stop(&criteria, 999, &[], &tail);
        prop_assert_eq!(
            result,
            Some(StopReason::StopString(stop.clone())),
        );
    }

    /// max_tokens budget: triggers when generated length reaches the limit.
    #[test]
    fn prop_max_tokens_triggers(budget in 1usize..32) {
        let criteria = StopCriteria {
            max_tokens: budget,
            ..Default::default()
        };
        let generated: Vec<u32> = (0..budget as u32).collect();
        let result = check_stop(&criteria, 999, &generated, "");
        prop_assert_eq!(result, Some(StopReason::MaxTokens));
    }

    /// No stop condition met → returns None.
    #[test]
    fn prop_no_stop_returns_none(token_id in 500u32..1000) {
        let criteria = StopCriteria {
            stop_token_ids: vec![0, 1, 2], // low IDs only
            max_tokens: 100,
            eos_token_id: Some(3),
            stop_strings: vec!["XYZZY".to_string()],
        };
        let result = check_stop(&criteria, token_id, &[1], "hello world");
        prop_assert!(result.is_none(), "unexpected stop: {result:?}");
    }

    /// Stop token ID takes priority over EOS when both match.
    #[test]
    fn prop_stop_token_priority_over_eos(id in 1u32..100) {
        let criteria = StopCriteria {
            stop_token_ids: vec![id],
            eos_token_id: Some(id),
            ..Default::default()
        };
        let result = check_stop(&criteria, id, &[], "");
        // Stop token ID is checked first
        prop_assert_eq!(result, Some(StopReason::StopTokenId(id)));
    }
}

// ---------------------------------------------------------------------------
// Properties: Softmax output invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Softmax of finite logits sums to ≈ 1.0 and all entries are non-negative.
    #[test]
    fn prop_softmax_sums_to_one(logits in logit_vec(32)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);

        let sum: f32 = probs.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {sum}, expected ≈ 1.0"
        );
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(p >= 0.0, "probs[{i}] = {p} is negative");
        }
    }
}
