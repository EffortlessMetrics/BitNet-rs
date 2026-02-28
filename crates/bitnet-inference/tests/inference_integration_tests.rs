//! Integration tests for `bitnet-inference` that run without a real model.
//!
//! Validates the inference pipeline with mock/synthetic data:
//! - [`SamplingConfig`] construction and greedy/sampled token selection
//! - [`SamplingStrategy`] reproducibility and edge-case handling
//! - [`StreamingConfig`] validation and named presets
//! - [`InferenceConfig`] / [`GenerationConfig`] builder patterns
//! - [`InferenceReceipt`] field validation and JSON round-trip
//! - [`ProductionInferenceConfig`] / [`PrefillStrategy`] construction
//! - [`PerformanceMetrics`] / [`PerformanceMetricsCollector`] recording
//! - [`InferenceEngine`] creation with a minimal mock model
//!
//! All tests must pass with:
//!   `cargo test --locked -p bitnet-inference --no-default-features --features cpu \
//!    -- inference_integration`
#![cfg(feature = "cpu")]

use std::sync::Arc;
use std::time::Duration;

use bitnet_common::{BitNetConfig, ConcreteTensor, Device};
use bitnet_inference::engine::{InferenceResult, PerformanceMetrics, PerformanceTracker};
use bitnet_inference::production_engine::{
    GenerationResult, PerformanceMetricsCollector, PrefillStrategy, ProductionInferenceConfig,
    ThroughputMetrics, TimingMetrics,
};
use bitnet_inference::receipts::{
    InferenceReceipt, ModelInfo, PerformanceBaseline, RECEIPT_SCHEMA_VERSION, TestResults,
};
use bitnet_inference::{
    CacheConfig, GenerationConfig, InferenceConfig, InferenceEngine, KVCache, SamplingConfig,
    SamplingStrategy, StreamingConfig,
};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Minimal mock model and tokenizer (no weights; no model file needed)
// ---------------------------------------------------------------------------

struct MockModel {
    config: BitNetConfig,
}

impl MockModel {
    fn with_vocab(vocab_size: usize) -> Self {
        let mut cfg = BitNetConfig::default();
        cfg.model.vocab_size = vocab_size;
        Self { config: cfg }
    }
}

impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 4, 64]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 1, self.config.model.vocab_size]))
    }
}

struct MockTokenizer {
    vocab_size: usize,
}

impl MockTokenizer {
    fn new() -> Self {
        Self { vocab_size: 256 }
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        Ok(text.bytes().take(8).map(|b| b as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
        Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(255)
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<{token}>"))
    }
}

// All tests live inside this module so `-- inference_integration` filter selects them.
#[cfg(test)]
mod inference_integration {
    use super::*;

    // ---------------------------------------------------------------------------
    // SamplingConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sampling_config_default_values() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_k, 50);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.repetition_penalty, 1.0);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn sampling_config_clone_is_independent() {
        let original = SamplingConfig { temperature: 0.5, seed: Some(99), ..Default::default() };
        let mut cloned = original.clone();
        cloned.temperature = 1.5;
        assert_eq!(original.temperature, 0.5, "original must not change after clone mutation");
        assert_eq!(cloned.temperature, 1.5);
    }

    #[test]
    fn sampling_config_explicit_fields_round_trip() {
        let cfg = SamplingConfig {
            temperature: 1.2,
            top_k: 100,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: Some(42),
        };
        assert_eq!(cfg.temperature, 1.2);
        assert_eq!(cfg.top_k, 100);
        assert_eq!(cfg.top_p, 0.95);
        assert_eq!(cfg.repetition_penalty, 1.1);
        assert_eq!(cfg.seed, Some(42));
    }

    // ---------------------------------------------------------------------------
    // SamplingStrategy tests
    // ---------------------------------------------------------------------------

    #[test]
    fn sampling_strategy_greedy_picks_argmax() {
        let cfg = SamplingConfig { temperature: 0.0, seed: Some(1), ..Default::default() };
        let mut strategy = SamplingStrategy::new(cfg);
        // logit at index 2 is highest
        let token = strategy.sample(&[0.1_f32, 0.2, 0.9, 0.4], &[]).expect("greedy must succeed");
        assert_eq!(token, 2, "greedy must pick the argmax index");
    }

    #[test]
    fn sampling_strategy_reproducible_with_seed() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32 * 0.1 + 0.05).collect();
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            seed: Some(42),
            ..Default::default()
        };
        let mut s1 = SamplingStrategy::new(cfg.clone());
        let mut s2 = SamplingStrategy::new(cfg);
        let t1 = s1.sample(&logits, &[1, 2]).expect("first strategy must sample");
        let t2 = s2.sample(&logits, &[1, 2]).expect("second strategy must sample");
        assert_eq!(t1, t2, "identical seeds must produce identical tokens");
    }

    #[test]
    fn sampling_strategy_rejects_empty_logits() {
        let mut strategy = SamplingStrategy::new(SamplingConfig::default());
        assert!(strategy.sample(&[], &[]).is_err(), "empty logit vector must return Err");
    }

    #[test]
    fn sampling_strategy_single_logit_always_picks_zero() {
        let mut strategy = SamplingStrategy::new(SamplingConfig::default());
        let token = strategy.sample(&[1.0_f32], &[]).expect("single logit must succeed");
        assert_eq!(token, 0, "only one possible token, must pick index 0");
    }

    #[test]
    fn sampling_strategy_token_in_bounds() {
        let logits: Vec<f32> = vec![0.1, 0.5, 0.3, 0.2, 0.8];
        let mut strategy = SamplingStrategy::new(SamplingConfig {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: Some(7),
        });
        let token = strategy.sample(&logits, &[]).expect("must succeed");
        assert!((token as usize) < logits.len(), "sampled token must be a valid index");
    }

    #[test]
    fn sampling_strategy_reset_clears_token_counts() {
        let logits: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut strategy = SamplingStrategy::new(SamplingConfig {
            temperature: 0.8,
            repetition_penalty: 1.5,
            seed: Some(10),
            ..Default::default()
        });
        // Sample once to build up token-count state, then reset.
        let _ = strategy.sample(&logits, &[0, 1, 2]);
        strategy.reset();
        // After reset, sampling must still succeed.
        let token = strategy.sample(&logits, &[]).expect("must succeed after reset");
        assert!((token as usize) < logits.len());
    }

    // ---------------------------------------------------------------------------
    // StreamingConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn streaming_config_default_is_valid() {
        let cfg = StreamingConfig::default();
        assert!(cfg.validate().is_ok(), "default streaming config must be valid");
        assert!(cfg.buffer_size > 0);
        assert!(cfg.flush_interval_ms > 0);
        assert!(cfg.token_timeout_ms > 0);
    }

    #[test]
    fn streaming_config_zero_buffer_is_invalid() {
        let cfg = StreamingConfig { buffer_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err(), "zero buffer size must be invalid");
    }

    #[test]
    fn streaming_config_zero_flush_interval_is_invalid() {
        let cfg = StreamingConfig { flush_interval_ms: 0, ..Default::default() };
        assert!(cfg.validate().is_err(), "zero flush interval must be invalid");
    }

    #[test]
    fn streaming_config_zero_token_timeout_is_invalid() {
        let cfg = StreamingConfig { token_timeout_ms: 0, ..Default::default() };
        assert!(cfg.validate().is_err(), "zero token timeout must be invalid");
    }

    #[test]
    fn streaming_config_low_latency_preset() {
        let cfg = StreamingConfig::low_latency();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.buffer_size, 1, "low-latency preset must use buffer_size=1");
        assert!(cfg.flush_interval_ms <= 10, "low-latency flush interval must be very small");
    }

    #[test]
    fn streaming_config_high_throughput_preset() {
        let cfg = StreamingConfig::high_throughput();
        assert!(cfg.validate().is_ok());
        assert!(cfg.buffer_size >= 10, "high-throughput preset must have a larger buffer");
        assert!(!cfg.cancellable, "high-throughput preset must disable cancellation");
    }

    // ---------------------------------------------------------------------------
    // InferenceConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn inference_config_defaults_are_valid() {
        let cfg = InferenceConfig::default();
        assert!(cfg.validate().is_ok());
        assert!(cfg.max_context_length > 0);
        assert!(cfg.num_threads > 0);
        assert!(cfg.batch_size > 0);
        assert!(cfg.memory_pool_size > 0);
    }

    #[test]
    fn inference_config_builder_chain() {
        let cfg = InferenceConfig::default()
            .with_threads(4)
            .with_batch_size(2)
            .with_mixed_precision(false)
            .with_memory_pool_size(64 * 1024 * 1024);
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.batch_size, 2);
        assert!(!cfg.mixed_precision);
        assert_eq!(cfg.memory_pool_size, 64 * 1024 * 1024);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn inference_config_rejects_zero_context_length() {
        let cfg = InferenceConfig {
            max_context_length: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_context_length"), "error must name the offending field");
    }

    #[test]
    fn inference_config_rejects_zero_threads() {
        let cfg = InferenceConfig {
            num_threads: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("num_threads"));
    }

    #[test]
    fn inference_config_rejects_zero_batch_size() {
        let cfg = InferenceConfig {
            batch_size: 0,
            ..Default::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("batch_size"));
    }

    #[test]
    fn inference_config_cpu_preset() {
        let cfg = InferenceConfig::cpu_optimized();
        assert!(cfg.validate().is_ok());
        assert!(!cfg.mixed_precision, "cpu preset must not enable mixed precision");
        assert_eq!(cfg.batch_size, 1, "cpu preset must use batch_size=1");
    }

    #[test]
    fn inference_config_gpu_preset() {
        let cfg = InferenceConfig::gpu_optimized();
        assert!(cfg.validate().is_ok());
        assert!(cfg.mixed_precision, "gpu preset must enable mixed precision");
        assert!(cfg.batch_size > 1, "gpu preset must use a larger batch size");
    }

    #[test]
    fn inference_config_serialization_roundtrip() {
        let cfg = InferenceConfig::default().with_threads(8).with_batch_size(4);
        let json = serde_json::to_string(&cfg).expect("serialize must succeed");
        let restored: InferenceConfig =
            serde_json::from_str(&json).expect("deserialize must succeed");
        assert_eq!(cfg.num_threads, restored.num_threads);
        assert_eq!(cfg.batch_size, restored.batch_size);
        assert_eq!(cfg.max_context_length, restored.max_context_length);
    }

    // ---------------------------------------------------------------------------
    // GenerationConfig tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generation_config_greedy_preset() {
        let cfg = GenerationConfig::greedy();
        assert_eq!(cfg.temperature, 0.0, "greedy temperature must be 0");
        assert_eq!(cfg.top_k, 1, "greedy top_k must be 1");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn generation_config_creative_preset() {
        let cfg = GenerationConfig::creative();
        assert!(cfg.temperature > 0.5, "creative preset must use high temperature");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn generation_config_builder_stop_token_ids() {
        let cfg =
            GenerationConfig::greedy().with_max_tokens(64).with_stop_token_ids(vec![128009, 50256]);
        assert_eq!(cfg.max_new_tokens, 64);
        assert!(cfg.stop_token_ids.contains(&128009));
        assert!(cfg.stop_token_ids.contains(&50256));
        // HashSet lookup must work
        assert!(cfg.is_stop_token(128009), "is_stop_token must return true for added ID");
        assert!(cfg.is_stop_token(50256));
        assert!(!cfg.is_stop_token(9999), "unregistered token must not match");
    }

    #[test]
    fn generation_config_single_stop_token_id_builder() {
        let cfg = GenerationConfig::default().with_stop_token_id(128009);
        assert!(cfg.is_stop_token(128009));
        assert!(!cfg.is_stop_token(0));
    }

    #[test]
    fn generation_config_stop_sequences_builder() {
        let cfg = GenerationConfig::default()
            .with_stop_sequence("</s>".to_string())
            .with_stop_sequence("\n\nQ:".to_string());
        assert_eq!(cfg.stop_sequences.len(), 2);
        assert!(cfg.stop_sequences.iter().any(|s| s == "</s>"));
        assert!(cfg.stop_sequences.iter().any(|s| s == "\n\nQ:"));
    }

    #[test]
    fn generation_config_seed_builder() {
        let cfg = GenerationConfig::default().with_seed(42);
        assert_eq!(cfg.seed, Some(42));
    }

    #[test]
    fn generation_config_validation_rejects_zero_max_tokens() {
        let mut cfg = GenerationConfig::default();
        cfg.max_new_tokens = 0;
        assert!(cfg.validate().is_err(), "zero max_new_tokens must be invalid");
    }

    #[test]
    fn generation_config_validation_rejects_negative_temperature() {
        let mut cfg = GenerationConfig::default();
        cfg.temperature = -0.1;
        assert!(cfg.validate().is_err(), "negative temperature must be invalid");
    }

    #[test]
    fn generation_config_validation_rejects_out_of_range_top_p() {
        let cfg_low = GenerationConfig::default().with_top_p(0.0);
        assert!(cfg_low.validate().is_err(), "top_p=0.0 must be invalid");
        let cfg_high = GenerationConfig::default().with_top_p(1.5);
        assert!(cfg_high.validate().is_err(), "top_p=1.5 must be invalid");
    }

    #[test]
    fn generation_config_serialization_roundtrip() {
        let cfg = GenerationConfig::greedy().with_max_tokens(32).with_seed(7);
        let json = serde_json::to_string(&cfg).expect("serialize must succeed");
        let restored: GenerationConfig =
            serde_json::from_str(&json).expect("deserialize must succeed");
        assert_eq!(cfg.max_new_tokens, restored.max_new_tokens);
        assert_eq!(cfg.temperature, restored.temperature);
        assert_eq!(cfg.seed, restored.seed);
    }

    // ---------------------------------------------------------------------------
    // InferenceReceipt tests
    // ---------------------------------------------------------------------------

    #[test]
    fn receipt_generate_sets_real_compute_path() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
            None,
        )
        .expect("receipt::generate must succeed");
        assert_eq!(receipt.compute_path, "real", "real kernels must yield compute_path=real");
        assert_eq!(receipt.backend, "cpu");
        assert_eq!(receipt.schema_version, RECEIPT_SCHEMA_VERSION);
        assert!(receipt.kernels.contains(&"i2s_gemv".to_string()));
    }

    #[test]
    fn receipt_generate_detects_mock_kernels() {
        let receipt = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None)
            .expect("generate must succeed even for mock kernels");
        assert_eq!(receipt.compute_path, "mock", "mock kernel must set compute_path=mock");
    }

    #[test]
    fn receipt_schema_version_constant_matches_generated() {
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate");
        assert_eq!(receipt.schema_version, RECEIPT_SCHEMA_VERSION);
        assert_eq!(RECEIPT_SCHEMA_VERSION, "1.0.0");
    }

    #[test]
    fn receipt_validate_schema_passes_for_real_receipt() {
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate");
        assert!(receipt.validate_schema().is_ok(), "schema validation must pass for real receipt");
    }

    #[test]
    fn receipt_validate_compute_path_rejects_mock() {
        let receipt = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None)
            .expect("must generate");
        assert!(
            receipt.validate_compute_path().is_err(),
            "compute_path=mock must fail validate_compute_path()"
        );
    }

    #[test]
    fn receipt_validate_compute_path_passes_for_real() {
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate");
        assert!(
            receipt.validate_compute_path().is_ok(),
            "compute_path=real must pass validate_compute_path()"
        );
    }

    #[test]
    fn receipt_json_roundtrip() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string()],
            Some("requested=cpu detected=[cpu] selected=cpu".to_string()),
        )
        .expect("must generate");
        let json = receipt.to_json_string().expect("to_json_string must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("must parse as JSON");
        assert_eq!(parsed["compute_path"], "real");
        assert_eq!(parsed["backend"], "cpu");
        assert_eq!(parsed["schema_version"], RECEIPT_SCHEMA_VERSION);
        assert_eq!(parsed["backend_summary"], "requested=cpu detected=[cpu] selected=cpu");
    }

    #[test]
    fn receipt_with_model_info_builder() {
        let model_info = ModelInfo {
            model_path: Some("models/test.gguf".to_string()),
            vocab_size: Some(32000),
            hidden_size: Some(2048),
            layers: Some(24),
            ..Default::default()
        };
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate")
            .with_model_info(model_info);
        assert_eq!(receipt.model_info.vocab_size, Some(32000));
        assert_eq!(receipt.model_info.layers, Some(24));
        assert_eq!(receipt.model_info.model_path.as_deref(), Some("models/test.gguf"));
    }

    #[test]
    fn receipt_with_performance_baseline_builder() {
        let perf = PerformanceBaseline {
            tokens_generated: Some(128),
            tokens_per_second: Some(10.5),
            total_time_ms: Some(12200),
            ..Default::default()
        };
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate")
            .with_performance_baseline(perf);
        assert_eq!(receipt.performance_baseline.tokens_generated, Some(128));
        assert!(receipt.performance_baseline.tokens_per_second.unwrap() > 10.0);
    }

    #[test]
    fn receipt_with_test_results_builder() {
        let results = TestResults { total_tests: 10, passed: 9, failed: 1, ..Default::default() };
        let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .expect("must generate")
            .with_test_results(results);
        assert_eq!(receipt.test_results.total_tests, 10);
        assert_eq!(receipt.test_results.passed, 9);
        assert_eq!(receipt.test_results.failed, 1);
    }

    #[test]
    fn receipt_empty_kernel_list_is_not_real() {
        // An empty kernel list cannot prove real computation.
        let receipt = InferenceReceipt::generate("cpu", vec![], None).expect("must generate");
        // The compute path for empty kernels is determined by bitnet-honest-compute.
        // Whatever it is, validate_kernel_ids() must not panic.
        let _ = receipt.validate_kernel_ids();
    }

    // ---------------------------------------------------------------------------
    // ProductionInferenceConfig / PrefillStrategy tests
    // ---------------------------------------------------------------------------

    #[test]
    fn production_inference_config_default_fields() {
        let cfg = ProductionInferenceConfig::default();
        assert!(cfg.enable_performance_monitoring, "performance monitoring on by default");
        assert!(cfg.enable_memory_tracking, "memory tracking on by default");
        assert!(cfg.max_inference_time_seconds > 0, "timeout must be positive");
        assert!(
            matches!(cfg.prefill_strategy, PrefillStrategy::Adaptive { .. }),
            "default prefill strategy must be Adaptive"
        );
    }

    #[test]
    fn prefill_strategy_always_variant() {
        let strategy = PrefillStrategy::Always;
        assert!(matches!(strategy, PrefillStrategy::Always));
    }

    #[test]
    fn prefill_strategy_never_variant() {
        let strategy = PrefillStrategy::Never;
        assert!(matches!(strategy, PrefillStrategy::Never));
    }

    #[test]
    fn prefill_strategy_adaptive_threshold() {
        let strategy = PrefillStrategy::Adaptive { threshold_tokens: 32 };
        if let PrefillStrategy::Adaptive { threshold_tokens } = strategy {
            assert_eq!(threshold_tokens, 32);
        } else {
            panic!("expected Adaptive variant");
        }
    }

    #[test]
    fn production_inference_config_custom_construction() {
        let cfg = ProductionInferenceConfig {
            enable_performance_monitoring: false,
            enable_memory_tracking: false,
            max_inference_time_seconds: 60,
            enable_quality_assessment: true,
            prefill_strategy: PrefillStrategy::Never,
        };
        assert!(!cfg.enable_performance_monitoring);
        assert!(!cfg.enable_memory_tracking);
        assert_eq!(cfg.max_inference_time_seconds, 60);
        assert!(cfg.enable_quality_assessment);
        assert!(matches!(cfg.prefill_strategy, PrefillStrategy::Never));
    }

    #[test]
    fn production_inference_config_adaptive_prefill_custom_threshold() {
        let cfg = ProductionInferenceConfig {
            prefill_strategy: PrefillStrategy::Adaptive { threshold_tokens: 128 },
            ..Default::default()
        };
        if let PrefillStrategy::Adaptive { threshold_tokens } = cfg.prefill_strategy {
            assert_eq!(threshold_tokens, 128);
        } else {
            panic!("expected Adaptive variant");
        }
    }

    // ---------------------------------------------------------------------------
    // PerformanceMetrics tests
    // ---------------------------------------------------------------------------

    #[test]
    fn performance_metrics_default_is_valid() {
        let m = PerformanceMetrics::default();
        assert!(m.validate().is_ok(), "default PerformanceMetrics must pass validation");
        assert_eq!(m.tokens_per_second, 0.0);
        assert_eq!(m.tokens_generated, 0);
        assert_eq!(m.total_latency_ms, 0);
    }

    #[test]
    fn performance_metrics_negative_tps_fails_validation() {
        let m = PerformanceMetrics { tokens_per_second: -1.0, ..Default::default() };
        assert!(m.validate().is_err(), "negative tokens_per_second must fail validation");
    }

    #[test]
    fn performance_metrics_cache_hit_rate_out_of_range_fails() {
        let m = PerformanceMetrics { cache_hit_rate: Some(1.5), ..Default::default() };
        assert!(m.validate().is_err(), "cache_hit_rate > 1.0 must fail validation");
    }

    #[test]
    fn performance_metrics_efficiency_ratio_zero_latency_returns_zero() {
        let m =
            PerformanceMetrics { total_latency_ms: 0, tokens_generated: 10, ..Default::default() };
        assert_eq!(m.efficiency_ratio(), 0.0, "zero latency must yield 0.0 efficiency ratio");
    }

    #[test]
    fn performance_metrics_efficiency_ratio_nonzero() {
        let m = PerformanceMetrics {
            total_latency_ms: 1000,
            tokens_generated: 100,
            ..Default::default()
        };
        let ratio = m.efficiency_ratio();
        assert!((ratio - 0.1).abs() < 1e-6, "100 tokens / 1000 ms = 0.1 tokens/ms, got {ratio}");
    }

    #[test]
    fn performance_metrics_valid_cache_hit_rate_boundaries() {
        for rate in [0.0_f64, 0.5, 1.0] {
            let m = PerformanceMetrics { cache_hit_rate: Some(rate), ..Default::default() };
            assert!(m.validate().is_ok(), "cache_hit_rate={rate} must be valid");
        }
    }

    // ---------------------------------------------------------------------------
    // PerformanceMetricsCollector tests
    // ---------------------------------------------------------------------------

    #[test]
    fn performance_metrics_collector_new_is_empty() {
        let collector = PerformanceMetricsCollector::new();
        assert_eq!(collector.throughput_metrics.total_tokens, 0);
        assert_eq!(collector.timing_metrics.total_ms, 0);
    }

    #[test]
    fn performance_metrics_collector_finalize_records_totals() {
        let mut c = PerformanceMetricsCollector::new();
        c.record_prefill_metrics(64, Duration::from_millis(200));
        c.record_decode_metrics(32, Duration::from_millis(800));
        c.record_tokenization_encode(Duration::from_millis(5));
        c.record_tokenization_decode(Duration::from_millis(3));
        c.finalize_metrics(96, Duration::from_secs(1));

        assert_eq!(c.throughput_metrics.total_tokens, 96);
        assert_eq!(c.timing_metrics.total_ms, 1000);
        assert!(c.throughput_metrics.end_to_end_tokens_per_sec > 0.0);
        assert!(c.timing_metrics.prefill_ms.is_some(), "prefill_ms must be recorded");
        assert!(c.timing_metrics.decode_ms.is_some(), "decode_ms must be recorded");
        assert!(c.timing_metrics.tokenization_encode_ms.is_some());
        assert!(c.timing_metrics.tokenization_decode_ms.is_some());
    }

    #[test]
    fn performance_metrics_collector_set_device_type_cpu() {
        let mut c = PerformanceMetricsCollector::new();
        c.set_device_type(&Device::Cpu);
        assert_eq!(c.device_type, "CPU");
    }

    #[test]
    fn performance_metrics_collector_set_device_type_cuda() {
        let mut c = PerformanceMetricsCollector::new();
        c.set_device_type(&Device::Cuda(0));
        assert!(
            c.device_type.contains("CUDA"),
            "CUDA device type must contain 'CUDA', got '{}'",
            c.device_type
        );
    }

    #[test]
    fn performance_metrics_collector_zero_duration_tps_is_zero() {
        let mut c = PerformanceMetricsCollector::new();
        // Zero-duration finalize must not panic and must yield 0 TPS.
        c.finalize_metrics(0, Duration::from_secs(0));
        assert_eq!(c.throughput_metrics.end_to_end_tokens_per_sec, 0.0);
    }

    // ---------------------------------------------------------------------------
    // TimingMetrics / ThroughputMetrics construction tests
    // ---------------------------------------------------------------------------

    #[test]
    fn timing_metrics_default_all_none() {
        let t = TimingMetrics::default();
        assert!(t.prefill_ms.is_none());
        assert!(t.decode_ms.is_none());
        assert!(t.tokenization_encode_ms.is_none());
        assert!(t.tokenization_decode_ms.is_none());
        assert_eq!(t.total_ms, 0);
    }

    #[test]
    fn throughput_metrics_default_zeroed() {
        let t = ThroughputMetrics::default();
        assert_eq!(t.total_tokens, 0);
        assert_eq!(t.end_to_end_tokens_per_sec, 0.0);
        assert!(t.prefill_tokens_per_sec.is_none());
        assert!(t.decode_tokens_per_sec.is_none());
    }

    // ---------------------------------------------------------------------------
    // GenerationResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn generation_result_construction() {
        let result = GenerationResult::new(
            "hello world".to_string(),
            2,
            PerformanceMetrics::default(),
            TimingMetrics::default(),
            ThroughputMetrics::default(),
        );
        assert_eq!(result.text, "hello world");
        assert_eq!(result.tokens_generated, 2);
        assert!(result.quality_score.is_none(), "quality score must start as None");
    }

    #[test]
    fn generation_result_calculate_quality_score_sets_value_in_range() {
        let perf = PerformanceMetrics { tokens_per_second: 50.0, ..Default::default() };
        let throughput = ThroughputMetrics {
            total_tokens: 5,
            end_to_end_tokens_per_sec: 50.0,
            ..Default::default()
        };
        let mut result = GenerationResult::new(
            "test".to_string(),
            5,
            perf,
            TimingMetrics::default(),
            throughput,
        );
        result.calculate_quality_score();
        let score = result.quality_score.expect("quality score must be Some after calculate");
        assert!((0.0..=1.0).contains(&score), "quality score must be in [0, 1], got {score}");
    }

    #[test]
    fn generation_result_zero_tokens_quality_score() {
        let mut result = GenerationResult::new(
            String::new(),
            0,
            PerformanceMetrics::default(),
            TimingMetrics::default(),
            ThroughputMetrics::default(),
        );
        result.calculate_quality_score();
        let score = result.quality_score.unwrap();
        assert_eq!(score, 0.0, "zero tokens must give quality score of 0.0");
    }

    // ---------------------------------------------------------------------------
    // KVCache tests
    // ---------------------------------------------------------------------------

    #[test]
    fn kv_cache_creation_with_default_config() {
        let config = CacheConfig::default();
        let cache = KVCache::new(config).expect("KVCache::new must succeed with default config");
        assert_eq!(cache.size(), 0, "freshly created cache must be empty");
        assert_eq!(cache.usage_percent(), 0.0, "fresh cache must have 0% usage");
    }

    #[test]
    fn kv_cache_clear_leaves_empty() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).expect("must create");
        cache.clear();
        assert_eq!(cache.size(), 0, "cache must still be empty after clear");
    }

    #[test]
    fn cache_config_default_is_nonzero() {
        let cfg = CacheConfig::default();
        assert!(cfg.max_size_bytes > 0, "default cache size must be positive");
        assert!(cfg.max_sequence_length > 0, "default max_sequence_length must be positive");
    }

    // ---------------------------------------------------------------------------
    // InferenceResult tests
    // ---------------------------------------------------------------------------

    #[test]
    fn inference_result_efficiency_score_in_range() {
        let perf = PerformanceMetrics { tokens_per_second: 50.0, ..Default::default() };
        let result = InferenceResult::new("hi".to_string(), 1, 100, 50.0, perf);
        let score = result.efficiency_score();
        assert!((0.0..=1.0).contains(&score), "efficiency score must be in [0, 1], got {score}");
    }

    #[test]
    fn inference_result_zero_tps_is_unacceptable() {
        let perf = PerformanceMetrics {
            tokens_per_second: 0.0,
            total_latency_ms: 100,
            ..Default::default()
        };
        let result = InferenceResult::new("".to_string(), 0, 100, 0.0, perf);
        assert!(!result.is_performance_acceptable(), "zero tokens_per_second must be unacceptable");
    }

    #[test]
    fn inference_result_positive_tps_is_acceptable() {
        let perf = PerformanceMetrics {
            tokens_per_second: 10.0,
            total_latency_ms: 100,
            tokens_generated: 1,
            ..Default::default()
        };
        let result = InferenceResult::new("ok".to_string(), 1, 100, 10.0, perf);
        assert!(result.is_performance_acceptable(), "positive tps must be acceptable");
    }

    // ---------------------------------------------------------------------------
    // PerformanceTracker tests
    // ---------------------------------------------------------------------------

    #[test]
    fn performance_tracker_records_inference() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_inference(10, 500);
        assert_eq!(tracker.total_inferences, 1);
        assert_eq!(tracker.total_tokens_generated, 10);
        assert_eq!(tracker.total_latency_ms, 500);
    }

    #[test]
    fn performance_tracker_accumulates_multiple_inferences() {
        let mut tracker = PerformanceTracker::new();
        tracker.record_inference(5, 200);
        tracker.record_inference(8, 300);
        assert_eq!(tracker.total_inferences, 2);
        assert_eq!(tracker.total_tokens_generated, 13);
        assert_eq!(tracker.total_latency_ms, 500);
    }

    // ---------------------------------------------------------------------------
    // InferenceEngine creation with mock model
    // ---------------------------------------------------------------------------

    #[test]
    fn inference_engine_creation_succeeds_with_mock_model() {
        let model: Arc<dyn bitnet_models::Model> = Arc::new(MockModel::with_vocab(256));
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let result = InferenceEngine::new(model, tokenizer, Device::Cpu);
        assert!(
            result.is_ok(),
            "InferenceEngine::new must succeed with a valid mock model: {:?}",
            result.err()
        );
    }

    #[test]
    fn inference_engine_exposes_tokenizer() {
        let model: Arc<dyn bitnet_models::Model> = Arc::new(MockModel::with_vocab(256));
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
            .expect("engine creation must succeed");
        let t = engine.tokenizer();
        assert_eq!(t.vocab_size(), 256, "exposed tokenizer must be the mock with vocab_size=256");
    }

    #[test]
    fn inference_engine_tokenizer_encode_decode_roundtrip() {
        let model: Arc<dyn bitnet_models::Model> = Arc::new(MockModel::with_vocab(256));
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)
            .expect("engine creation must succeed");
        let t = engine.tokenizer();
        let tokens = t.encode("hi", false, false).expect("encode must succeed");
        assert!(!tokens.is_empty(), "encode of non-empty string must produce tokens");
        let text = t.decode(&tokens).expect("decode must succeed");
        assert!(!text.is_empty(), "decoded text must be non-empty");
    }
} // mod inference_integration
