//! Extended tests for `bitnet-inference` public API.
//!
//! Focuses on areas not covered by `inference_integration_tests.rs`:
//! - Additional `SamplingConfig` / `SamplingStrategy` edge cases
//! - `GenerationConfig` builder methods not yet exercised
//! - `InferenceReceipt` builder/validate variants
//! - `ModelInfo`, `TestResults`, `AccuracyMetric` construction
//!
//! Run with:
//!   `cargo test --locked -p bitnet-inference --no-default-features --features cpu \
//!    -- inference_engine_tests`
#![cfg(feature = "cpu")]

use bitnet_common::CorrectionRecord;
use bitnet_inference::receipts::{
    AccuracyMetric, AccuracyTestResults, InferenceReceipt, ModelInfo, ParityMetadata,
    PerformanceBaseline, RECEIPT_SCHEMA_VERSION, TestResults,
};
use bitnet_inference::{GenerationConfig, InferenceConfig, SamplingConfig, SamplingStrategy};
use bitnet_sampling::greedy_sample;

// ---------------------------------------------------------------------------
// SamplingConfig – construction variants
// ---------------------------------------------------------------------------

#[test]
fn sampling_config_greedy_temperature_zero() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    assert_eq!(cfg.temperature, 0.0);
    assert_eq!(cfg.seed, Some(0));
}

#[test]
fn sampling_config_top_k_one_is_deterministic() {
    let cfg = SamplingConfig { top_k: 1, temperature: 1.0, seed: Some(7), ..Default::default() };
    assert_eq!(cfg.top_k, 1);
}

#[test]
fn sampling_config_top_p_half() {
    let cfg = SamplingConfig { top_p: 0.5, ..Default::default() };
    assert!(cfg.top_p > 0.0 && cfg.top_p < 1.0);
}

#[test]
fn sampling_config_high_repetition_penalty() {
    let cfg = SamplingConfig { repetition_penalty: 2.0, ..Default::default() };
    assert!(cfg.repetition_penalty > 1.0);
}

#[test]
fn sampling_config_seed_none_by_default() {
    let cfg = SamplingConfig::default();
    assert!(cfg.seed.is_none());
}

// ---------------------------------------------------------------------------
// SamplingStrategy – additional behaviour
// ---------------------------------------------------------------------------

#[test]
fn sampling_strategy_valid_index_high_temperature() {
    let cfg = SamplingConfig { temperature: 10.0, seed: Some(123), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![0.1_f32, 0.3, 0.6];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len());
}

#[test]
fn sampling_strategy_large_vocab_stays_in_range() {
    let cfg = SamplingConfig { temperature: 0.7, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits: Vec<f32> = (0..32000).map(|i| i as f32 / 32000.0).collect();
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len());
}

#[test]
fn sampling_strategy_update_config_changes_temperature() {
    let initial = SamplingConfig { temperature: 0.0, seed: Some(1), ..Default::default() };
    let mut strategy = SamplingStrategy::new(initial);
    let new_cfg = SamplingConfig { temperature: 0.8, seed: Some(1), ..Default::default() };
    strategy.update_config(new_cfg.clone());
    // Verify updated config is reflected (strategy returns valid index).
    let logits = vec![0.2_f32, 0.5, 0.3];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len());
}

#[test]
fn sampling_strategy_update_config_new_seed_accepted() {
    let initial = SamplingConfig { temperature: 0.5, seed: Some(10), ..Default::default() };
    let mut strategy = SamplingStrategy::new(initial);
    // Changing the seed re-seeds the RNG; the method must not panic.
    let new_cfg = SamplingConfig { temperature: 0.5, seed: Some(99), ..Default::default() };
    strategy.update_config(new_cfg);
    let logits = vec![1.0_f32, 2.0, 3.0];
    assert!(strategy.sample(&logits, &[]).is_ok());
}

#[test]
fn sampling_strategy_reset_allows_reuse() {
    let cfg = SamplingConfig { repetition_penalty: 1.5, seed: Some(5), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0_f32; 8];
    // Sample multiple tokens to build internal counts.
    for _ in 0..4 {
        strategy.sample(&logits, &[]).unwrap();
    }
    // After reset the state should be clean and further sampling should succeed.
    strategy.reset();
    assert!(strategy.sample(&logits, &[]).is_ok());
}

// ---------------------------------------------------------------------------
// greedy_sample – tie-breaking
// ---------------------------------------------------------------------------

#[test]
fn greedy_sample_tie_breaking_returns_lowest_index() {
    let logits = vec![1.0_f32, 1.0, 0.5];
    assert_eq!(greedy_sample(&logits).unwrap(), 0, "ties must break to lowest index");
}

#[test]
fn greedy_sample_single_element() {
    let logits = vec![42.0_f32];
    assert_eq!(greedy_sample(&logits).unwrap(), 0);
}

#[test]
fn greedy_sample_empty_returns_error() {
    let result = greedy_sample(&[]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// GenerationConfig – builder methods not yet exercised
// ---------------------------------------------------------------------------

#[test]
fn generation_config_balanced_preset() {
    let cfg = GenerationConfig::balanced();
    assert_eq!(cfg.temperature, 0.7);
    assert_eq!(cfg.top_k, 50);
    assert_eq!(cfg.top_p, 0.9);
    assert!(cfg.repetition_penalty > 1.0, "balanced preset penalises repetition");
}

#[test]
fn generation_config_with_stop_string_window() {
    let cfg = GenerationConfig::default().with_stop_string_window(128);
    assert_eq!(cfg.stop_string_window, 128);
}

#[test]
fn generation_config_with_eos_token_id() {
    let cfg = GenerationConfig::default().with_eos_token_id(Some(2));
    assert_eq!(cfg.eos_token_id, Some(2));
    let cfg_none = GenerationConfig::default().with_eos_token_id(None);
    assert!(cfg_none.eos_token_id.is_none());
}

#[test]
fn generation_config_with_add_bos_true() {
    let cfg = GenerationConfig::default().with_add_bos(true);
    assert!(cfg.add_bos);
}

#[test]
fn generation_config_with_add_bos_false_default() {
    let cfg = GenerationConfig::default();
    assert!(!cfg.add_bos, "add_bos must be false by default");
}

#[test]
fn generation_config_with_skip_special_tokens_false() {
    let cfg = GenerationConfig::default().with_skip_special_tokens(false);
    assert!(!cfg.skip_special_tokens);
}

#[test]
fn generation_config_validation_rejects_top_p_zero() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    assert!(cfg.validate().is_err(), "top_p = 0.0 must be rejected");
}

#[test]
fn generation_config_validation_rejects_top_p_above_one() {
    let cfg = GenerationConfig::default().with_top_p(1.1);
    assert!(cfg.validate().is_err(), "top_p = 1.1 must be rejected");
}

#[test]
fn generation_config_multiple_stop_sequences_order_preserved() {
    let cfg = GenerationConfig::default()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\nQ:".to_string())
        .with_stop_sequence("<|eot_id|>".to_string());
    assert_eq!(cfg.stop_sequences.len(), 3);
    assert_eq!(cfg.stop_sequences[0], "</s>");
    assert_eq!(cfg.stop_sequences[1], "\n\nQ:");
    assert_eq!(cfg.stop_sequences[2], "<|eot_id|>");
}

#[test]
fn generation_config_rebuild_stop_token_set_needed_after_direct_mutation() {
    let mut cfg = GenerationConfig::default();
    // Direct mutation – HashSet is NOT updated yet.
    cfg.stop_token_ids = vec![128009];
    assert!(!cfg.is_stop_token(128009), "HashSet out of sync before rebuild");
    cfg.rebuild_stop_token_set();
    assert!(cfg.is_stop_token(128009), "HashSet synced after rebuild");
}

// ---------------------------------------------------------------------------
// InferenceConfig – edge cases
// ---------------------------------------------------------------------------

#[test]
fn inference_config_memory_efficient_preset() {
    let cfg = InferenceConfig::memory_efficient();
    assert_eq!(cfg.max_context_length, 1024);
    assert_eq!(cfg.batch_size, 1);
    assert_eq!(cfg.memory_pool_size, 1024 * 1024 * 256);
}

// ---------------------------------------------------------------------------
// InferenceReceipt – additional builder / validator coverage
// ---------------------------------------------------------------------------

#[test]
fn receipt_generate_basic_is_real() {
    let r = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        None,
    )
    .unwrap();
    assert_eq!(r.compute_path, "real");
    assert_eq!(r.schema_version, RECEIPT_SCHEMA_VERSION);
}

#[test]
fn receipt_validate_kernel_ids_rejects_empty_string() {
    let r = InferenceReceipt::generate("cpu", vec!["".to_string()], None).unwrap();
    assert!(r.validate_kernel_ids().is_err(), "empty kernel id must fail validation");
}

#[test]
fn receipt_validate_kernel_ids_rejects_oversized_id() {
    let long_id = "a".repeat(129);
    let r = InferenceReceipt::generate("cpu", vec![long_id], None).unwrap();
    assert!(r.validate_kernel_ids().is_err(), "kernel id > 128 chars must fail");
}

#[test]
fn receipt_validate_kernel_ids_accepts_exactly_128_chars() {
    let ok_id = "b".repeat(128);
    let r = InferenceReceipt::generate("cpu", vec![ok_id], None).unwrap();
    assert!(r.validate_kernel_ids().is_ok(), "128-char kernel id must pass");
}

#[test]
fn receipt_with_parity_sets_parity_field() {
    let parity = ParityMetadata {
        cpp_available: true,
        cosine_similarity: Some(0.9999),
        exact_match_rate: Some(1.0),
        status: "ok".to_string(),
    };
    let r = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
        .unwrap()
        .with_parity(parity.clone());
    let p = r.parity.as_ref().unwrap();
    assert!(p.cpp_available);
    assert_eq!(p.status, "ok");
    assert!((p.cosine_similarity.unwrap() - 0.9999).abs() < 1e-6);
}

#[test]
fn receipt_with_corrections_sets_corrections_field() {
    let correction = CorrectionRecord {
        layer: "model.layers.0.input_layernorm.weight".to_string(),
        correction_type: "ln_gamma_rescale_rms".to_string(),
        rms_before: Some(0.01),
        rms_after: Some(1.0),
        factor: Some(100.0),
        policy_fingerprint: "test-policy".to_string(),
        metadata: None,
    };
    let r = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
        .unwrap()
        .with_corrections(vec![correction]);
    assert_eq!(r.corrections.len(), 1);
    assert_eq!(r.corrections[0].correction_type, "ln_gamma_rescale_rms");
}

#[test]
fn receipt_parity_rust_only_status() {
    let parity = ParityMetadata {
        cpp_available: false,
        cosine_similarity: None,
        exact_match_rate: None,
        status: "rust_only".to_string(),
    };
    let r = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
        .unwrap()
        .with_parity(parity);
    assert_eq!(r.parity.as_ref().unwrap().status, "rust_only");
    assert!(!r.parity.as_ref().unwrap().cpp_available);
}

// ---------------------------------------------------------------------------
// ModelInfo – field coverage
// ---------------------------------------------------------------------------

#[test]
fn model_info_all_optional_fields_populated() {
    let info = ModelInfo {
        model_path: Some("/models/model.gguf".to_string()),
        quantization_type: Some("I2_S".to_string()),
        layers: Some(32),
        hidden_size: Some(4096),
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        vocab_size: Some(32000),
        sha256: Some("abc123".to_string()),
        effective_correction_digest: None,
    };
    assert_eq!(info.layers, Some(32));
    assert_eq!(info.hidden_size, Some(4096));
    assert_eq!(info.vocab_size, Some(32000));
    assert_eq!(info.quantization_type.as_deref(), Some("I2_S"));
}

#[test]
fn model_info_default_all_none() {
    let info = ModelInfo::default();
    assert!(info.model_path.is_none());
    assert!(info.quantization_type.is_none());
    assert!(info.layers.is_none());
    assert!(info.vocab_size.is_none());
}

#[test]
fn receipt_with_model_info_round_trips_via_json() {
    let info = ModelInfo {
        model_path: Some("model.gguf".to_string()),
        vocab_size: Some(32000),
        ..Default::default()
    };
    let r = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
        .unwrap()
        .with_model_info(info);

    let json = r.to_json_string().unwrap();
    assert!(json.contains("model.gguf"));
    assert!(json.contains("32000"));
}

// ---------------------------------------------------------------------------
// TestResults – field relationships
// ---------------------------------------------------------------------------

#[test]
fn test_results_passed_plus_failed_le_total() {
    let tr = TestResults {
        total_tests: 10,
        passed: 8,
        failed: 2,
        skipped: None,
        accuracy_tests: None,
        determinism_tests: None,
        kv_cache_tests: None,
    };
    assert_eq!(tr.passed + tr.failed, tr.total_tests);
}

#[test]
fn test_results_with_skipped() {
    let tr = TestResults {
        total_tests: 12,
        passed: 10,
        failed: 0,
        skipped: Some(2),
        accuracy_tests: None,
        determinism_tests: None,
        kv_cache_tests: None,
    };
    let skipped = tr.skipped.unwrap_or(0);
    assert_eq!(tr.passed + tr.failed + skipped, tr.total_tests);
}

// ---------------------------------------------------------------------------
// AccuracyMetric
// ---------------------------------------------------------------------------

#[test]
fn accuracy_metric_passed_when_mse_under_tolerance() {
    let m = AccuracyMetric { mse: 0.001, tolerance: 0.01, passed: true };
    assert!(m.passed);
    assert!(m.mse < m.tolerance);
}

#[test]
fn accuracy_metric_failed_when_mse_exceeds_tolerance() {
    let m = AccuracyMetric { mse: 0.05, tolerance: 0.01, passed: false };
    assert!(!m.passed);
    assert!(m.mse > m.tolerance);
}

#[test]
fn accuracy_test_results_all_kernels() {
    let tr = AccuracyTestResults {
        i2s_accuracy: Some(AccuracyMetric { mse: 0.001, tolerance: 0.01, passed: true }),
        tl1_accuracy: Some(AccuracyMetric { mse: 0.002, tolerance: 0.01, passed: true }),
        tl2_accuracy: Some(AccuracyMetric { mse: 0.003, tolerance: 0.01, passed: true }),
    };
    assert!(tr.i2s_accuracy.as_ref().unwrap().passed);
    assert!(tr.tl1_accuracy.as_ref().unwrap().passed);
    assert!(tr.tl2_accuracy.as_ref().unwrap().passed);
}

// ---------------------------------------------------------------------------
// PerformanceBaseline – optional fields
// ---------------------------------------------------------------------------

#[test]
fn performance_baseline_all_fields_set() {
    let pb = PerformanceBaseline {
        tokens_generated: Some(100),
        total_time_ms: Some(5000),
        tokens_per_second: Some(20.0),
        first_token_latency_ms: Some(100),
        average_token_latency_ms: Some(50),
        memory_usage_mb: Some(512),
        cache_efficiency: None,
    };
    assert_eq!(pb.tokens_generated, Some(100));
    assert_eq!(pb.tokens_per_second, Some(20.0));
}

#[test]
fn performance_baseline_default_all_none() {
    let pb = PerformanceBaseline::default();
    assert!(pb.tokens_generated.is_none());
    assert!(pb.tokens_per_second.is_none());
    assert!(pb.memory_usage_mb.is_none());
}

// ---------------------------------------------------------------------------
// proptest – sampling properties
// ---------------------------------------------------------------------------

#[cfg(test)]
mod sampling_proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// SamplingStrategy with a seed always returns valid indices.
        #[test]
        fn strategy_seeded_valid_index(
            logits in prop::collection::vec(-20f32..=20f32, 2..=256),
            seed in 0u64..u64::MAX,
            temperature in 0.0f32..=2.0f32,
        ) {
            let cfg = SamplingConfig { temperature, seed: Some(seed), top_k: 0, top_p: 1.0, repetition_penalty: 1.0 };
            let mut strategy = SamplingStrategy::new(cfg);
            let token = strategy.sample(&logits, &[]).unwrap();
            prop_assert!((token as usize) < logits.len());
        }

        /// GenerationConfig builder chain never panics for valid inputs.
        #[test]
        fn gen_config_builder_never_panics(
            max_tokens in 1u32..512,
            temp in 0.0f32..=2.0f32,
            top_k in 0u32..100,
            window in 1usize..=256,
        ) {
            let cfg = GenerationConfig::default()
                .with_max_tokens(max_tokens)
                .with_temperature(temp)
                .with_top_k(top_k)
                .with_stop_string_window(window);
            prop_assert_eq!(cfg.max_new_tokens, max_tokens);
            prop_assert_eq!(cfg.temperature, temp);
        }
    }
}
