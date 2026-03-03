//! Wave 34 snapshot tests for inference configuration, sampling, generation,
//! performance metrics, error formatting, and request/response types.
//!
//! Covers: InferenceConfig (config + config_builder), SamplingConfig/Strategy,
//! GenerationConfig, PerformanceMetrics, InferenceResult, GenerationBudget,
//! BudgetTracker, StopReason, RopeConfig, ComputeCost, InferenceRequest,
//! InferenceResponse, FinishReason, TimingInfo, UsageInfo, GenerationOutput,
//! TokenOutput, OutputSummary, and error/validation messages.

use std::time::Duration;

use bitnet_inference::compute_cost::{ModelDims, estimate_bandwidth, estimate_flops_per_token};
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};
use bitnet_inference::engine::{InferenceResult, PerformanceMetrics, PerformanceTracker};
use bitnet_inference::generation::{SampleConfig, SamplingStrategy as GenSamplingStrategy};
use bitnet_inference::generation_budget::{GenerationBudget, StopReason as BudgetStopReason};
use bitnet_inference::generation_output::{StopReason as OutputStopReason, TokenOutput};
use bitnet_inference::request_types::{
    FinishReason, InferenceRequest, InferenceResponse, TimingInfo, UsageInfo,
};
use bitnet_inference::rope_config::{RopeConfig, RopeScaling};
use bitnet_inference::run_metrics::{
    self, AggregateMetrics, InferenceMetrics as RunInferenceMetrics,
};
use bitnet_inference::sampling::SamplingConfig;

// ============================================================================
// InferenceConfig (config module) defaults
// ============================================================================

#[test]
fn w34_inference_config_default_debug() {
    let mut cfg = InferenceConfig::default();
    // Pin num_threads so snapshot is deterministic across machines
    cfg.num_threads = 8;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_inference_config_default_json() {
    let mut cfg = InferenceConfig::default();
    cfg.num_threads = 8;
    insta::assert_snapshot!(serde_json::to_string_pretty(&cfg).unwrap());
}

// ============================================================================
// InferenceConfig builder presets
// ============================================================================

#[test]
fn w34_builder_preset_fast_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_preset_balanced_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_preset_quality_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Quality).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_preset_deterministic_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_preset_debug_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_custom_overrides() {
    let cfg = InferenceConfigBuilder::new()
        .preset(InferencePreset::Balanced)
        .temperature(0.42)
        .max_tokens(16)
        .top_k(25)
        .seed(1234)
        .build()
        .unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_builder_validation_error_bad_top_p() {
    let err = InferenceConfigBuilder::new().top_p(0.0).build().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn w34_builder_validation_error_bad_temperature() {
    let err = InferenceConfigBuilder::new().temperature(-1.0).build().unwrap_err();
    insta::assert_snapshot!(err);
}

// ============================================================================
// GenerationConfig presets and serialization
// ============================================================================

#[test]
fn w34_gen_config_default_json() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(serde_json::to_string_pretty(&cfg).unwrap());
}

#[test]
fn w34_gen_config_greedy_json() {
    let cfg = GenerationConfig::greedy();
    insta::assert_snapshot!(serde_json::to_string_pretty(&cfg).unwrap());
}

#[test]
fn w34_gen_config_creative_json() {
    let cfg = GenerationConfig::creative();
    insta::assert_snapshot!(serde_json::to_string_pretty(&cfg).unwrap());
}

#[test]
fn w34_gen_config_balanced_json() {
    let cfg = GenerationConfig::balanced();
    insta::assert_snapshot!(serde_json::to_string_pretty(&cfg).unwrap());
}

#[test]
fn w34_gen_config_with_stops_debug() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(8)
        .with_stop_sequences(vec!["</s>".into(), "\n\nQ:".into()])
        .with_stop_token_ids(vec![128009])
        .with_eos_token_id(Some(2))
        .with_seed(42);
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// SamplingConfig (sampling module, re-exported from bitnet-sampling)
// ============================================================================

#[test]
fn w34_sampling_config_default_debug() {
    let cfg = SamplingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_sampling_config_greedy() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_sampling_config_creative() {
    let cfg = SamplingConfig {
        temperature: 0.9,
        top_k: 100,
        top_p: 0.95,
        repetition_penalty: 1.15,
        seed: None,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// Generation module SampleConfig / GenSamplingStrategy
// ============================================================================

#[test]
fn w34_gen_sample_config_default() {
    let cfg = SampleConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_gen_sampling_strategy_default() {
    let cfg = SampleConfig::default();
    let strat = GenSamplingStrategy::new(cfg);
    insta::assert_debug_snapshot!(strat);
}

// ============================================================================
// PerformanceMetrics
// ============================================================================

#[test]
fn w34_perf_metrics_default_debug() {
    let m = PerformanceMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_perf_metrics_default_json() {
    let m = PerformanceMetrics::default();
    insta::assert_snapshot!(serde_json::to_string_pretty(&m).unwrap());
}

#[test]
fn w34_perf_metrics_populated() {
    let m = PerformanceMetrics {
        total_latency_ms: 2500,
        tokens_generated: 64,
        tokens_per_second: 25.6,
        first_token_latency_ms: Some(150),
        average_token_latency_ms: Some(36.72),
        memory_usage_bytes: Some(536_870_912),
        cache_hit_rate: Some(0.87),
        backend_type: "cpu-avx2".into(),
        model_load_time_ms: Some(800),
        tokenizer_encode_time_ms: Some(5),
        tokenizer_decode_time_ms: Some(3),
        forward_pass_time_ms: Some(2200),
        sampling_time_ms: Some(45),
    };
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_perf_metrics_populated_json() {
    let m = PerformanceMetrics {
        total_latency_ms: 2500,
        tokens_generated: 64,
        tokens_per_second: 25.6,
        first_token_latency_ms: Some(150),
        average_token_latency_ms: Some(36.72),
        memory_usage_bytes: Some(536_870_912),
        cache_hit_rate: Some(0.87),
        backend_type: "cpu-avx2".into(),
        model_load_time_ms: Some(800),
        tokenizer_encode_time_ms: Some(5),
        tokenizer_decode_time_ms: Some(3),
        forward_pass_time_ms: Some(2200),
        sampling_time_ms: Some(45),
    };
    insta::assert_snapshot!(serde_json::to_string_pretty(&m).unwrap());
}

#[test]
fn w34_perf_metrics_validate_ok() {
    let m = PerformanceMetrics { cache_hit_rate: Some(0.5), ..Default::default() };
    insta::assert_snapshot!(format!("{:?}", m.validate()));
}

#[test]
fn w34_perf_metrics_validate_bad_cache_rate() {
    let m = PerformanceMetrics { cache_hit_rate: Some(1.5), ..Default::default() };
    insta::assert_snapshot!(format!("{:?}", m.validate()));
}

#[test]
fn w34_perf_metrics_validate_bad_tps() {
    let m = PerformanceMetrics { tokens_per_second: -1.0, ..Default::default() };
    insta::assert_snapshot!(format!("{:?}", m.validate()));
}

#[test]
fn w34_perf_metrics_efficiency_ratio() {
    let m =
        PerformanceMetrics { total_latency_ms: 1000, tokens_generated: 50, ..Default::default() };
    insta::assert_snapshot!(format!("efficiency_ratio={:.4}", m.efficiency_ratio()));
}

// ============================================================================
// InferenceResult
// ============================================================================

#[test]
fn w34_inference_result_debug() {
    let result =
        InferenceResult::new("Hello world".into(), 4, 200, 20.0, PerformanceMetrics::default());
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w34_inference_result_efficiency_score() {
    let result = InferenceResult::new(
        "test".into(),
        10,
        500,
        50.0,
        PerformanceMetrics { tokens_per_second: 50.0, ..Default::default() },
    );
    insta::assert_snapshot!(format!(
        "efficiency_score={:.2} acceptable={}",
        result.efficiency_score(),
        result.is_performance_acceptable(),
    ));
}

// ============================================================================
// PerformanceTracker
// ============================================================================

#[test]
fn w34_performance_tracker_default_debug() {
    let t = PerformanceTracker::default();
    insta::assert_snapshot!(format!(
        "total_inferences={} total_tokens={} cache_hits={} cache_misses={}",
        t.total_inferences, t.total_tokens_generated, t.cache_hits, t.cache_misses,
    ));
}

// ============================================================================
// GenerationBudget / BudgetTracker
// ============================================================================

#[test]
fn w34_gen_budget_default_debug() {
    let b = GenerationBudget::default();
    insta::assert_debug_snapshot!(b);
}

#[test]
fn w34_gen_budget_with_limits() {
    let b = GenerationBudget::new(128)
        .with_time_limit(Duration::from_secs(30))
        .with_memory_limit(1_073_741_824);
    insta::assert_debug_snapshot!(b);
}

#[test]
fn w34_gen_budget_unlimited() {
    let b = GenerationBudget::unlimited();
    insta::assert_debug_snapshot!(b);
}

#[test]
fn w34_budget_stop_reason_display() {
    let reasons = [
        BudgetStopReason::MaxTokens,
        BudgetStopReason::TimeLimit,
        BudgetStopReason::MemoryLimit,
        BudgetStopReason::EndOfSequence,
        BudgetStopReason::UserStop,
    ];
    let displays: Vec<String> = reasons.iter().map(|r| format!("{r}")).collect();
    insta::assert_debug_snapshot!(displays);
}

// ============================================================================
// GenerationOutput / TokenOutput / StopReason
// ============================================================================

#[test]
fn w34_token_output_debug() {
    let tok = TokenOutput {
        token_id: 42,
        text: "hello".into(),
        logit: 2.5,
        probability: 0.85,
        latency_us: 1200,
        position: 3,
    };
    insta::assert_debug_snapshot!(tok);
}

#[test]
fn w34_output_stop_reasons_debug() {
    let reasons = [
        OutputStopReason::MaxTokens,
        OutputStopReason::EosToken,
        OutputStopReason::StopSequence("</s>".into()),
        OutputStopReason::UserAbort,
        OutputStopReason::Error("out of memory".into()),
    ];
    let debug: Vec<String> = reasons.iter().map(|r| format!("{r:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

// ============================================================================
// RopeConfig
// ============================================================================

#[test]
fn w34_rope_config_default_debug() {
    let cfg = RopeConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_rope_scaling_variants_debug() {
    let variants = [
        RopeScaling::None,
        RopeScaling::Linear(2.0),
        RopeScaling::Dynamic(1.5),
        RopeScaling::Yarn { factor: 4.0, original_max_pos: 8192 },
        RopeScaling::NTKAware(8.0),
    ];
    let debug: Vec<String> = variants.iter().map(|v| format!("{v:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

#[test]
fn w34_rope_config_custom() {
    let cfg = RopeConfig {
        head_dim: 64,
        max_position: 8192,
        base: 500000.0,
        scaling: RopeScaling::NTKAware(4.0),
        interleaved: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// ComputeCost: ModelDims / FlopsEstimate / BandwidthEstimate
// ============================================================================

#[test]
fn w34_model_dims_llama7b_debug() {
    let dims = ModelDims::new(4096, 32, 32, 32, 11008, 32000);
    insta::assert_debug_snapshot!(dims);
}

#[test]
fn w34_flops_estimate_llama7b() {
    let dims = ModelDims::new(4096, 32, 32, 32, 11008, 32000);
    let flops = estimate_flops_per_token(&dims);
    insta::assert_debug_snapshot!(flops);
}

#[test]
fn w34_bandwidth_estimate_llama7b() {
    let dims = ModelDims::new(4096, 32, 32, 32, 11008, 32000);
    let bw = estimate_bandwidth(&dims, 512, 2);
    insta::assert_debug_snapshot!(bw);
}

#[test]
fn w34_model_dims_bitnet2b() {
    let dims = ModelDims::new(2048, 24, 16, 4, 5504, 100000);
    insta::assert_debug_snapshot!(dims);
}

#[test]
fn w34_flops_estimate_bitnet2b() {
    let dims = ModelDims::new(2048, 24, 16, 4, 5504, 100000);
    let flops = estimate_flops_per_token(&dims);
    insta::assert_snapshot!(format!(
        "attn={} ffn={} lm_head={} total={}",
        flops.attention_flops, flops.ffn_flops, flops.lm_head_flops, flops.total_flops,
    ));
}

// ============================================================================
// InferenceRequest / InferenceResponse / FinishReason / TimingInfo / UsageInfo
// ============================================================================

#[test]
fn w34_inference_request_default_debug() {
    let req = InferenceRequest::default();
    insta::assert_debug_snapshot!(req);
}

#[test]
fn w34_inference_request_builder() {
    let req = InferenceRequest::new("What is 2+2?")
        .with_id("req-001")
        .with_max_tokens(32)
        .with_temperature(0.0)
        .with_top_p(1.0)
        .with_top_k(1)
        .with_seed(42)
        .with_stream(true);
    insta::assert_debug_snapshot!(req);
}

#[test]
fn w34_inference_request_greedy_check() {
    let req = InferenceRequest::new("test").with_temperature(0.0).with_seed(42);
    insta::assert_snapshot!(format!(
        "is_greedy={} is_deterministic={}",
        req.is_greedy(),
        req.is_deterministic(),
    ));
}

#[test]
fn w34_finish_reason_variants_debug() {
    let reasons = [
        FinishReason::MaxTokens,
        FinishReason::StopSequence,
        FinishReason::EosToken,
        FinishReason::Error("timeout".into()),
    ];
    let debug: Vec<String> = reasons.iter().map(|r| format!("{r:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

#[test]
fn w34_timing_info_debug() {
    let t = TimingInfo {
        prompt_eval_ms: 50,
        generation_ms: 2000,
        total_ms: 2050,
        tokens_per_sec: 32.0,
    };
    insta::assert_debug_snapshot!(t);
}

#[test]
fn w34_usage_info_debug() {
    let u = UsageInfo::new(15, 32);
    insta::assert_debug_snapshot!(u);
}

#[test]
fn w34_inference_response_debug() {
    let resp = InferenceResponse {
        id: "resp-001".into(),
        text: "The answer is 4.".into(),
        token_ids: vec![1, 450, 1234, 338, 29871, 29946, 29889],
        token_count: 7,
        prompt_tokens: 5,
        finish_reason: FinishReason::EosToken,
        timing: TimingInfo {
            prompt_eval_ms: 30,
            generation_ms: 350,
            total_ms: 380,
            tokens_per_sec: 18.4,
        },
        usage: UsageInfo::new(5, 7),
    };
    insta::assert_debug_snapshot!(resp);
}

// ============================================================================
// RunMetrics: InferenceMetrics / AggregateMetrics / format_metrics
// ============================================================================

#[test]
fn w34_run_metrics_default_debug() {
    let m = RunInferenceMetrics::new();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_run_metrics_format_empty() {
    let m = RunInferenceMetrics::new();
    insta::assert_snapshot!(run_metrics::format_metrics(&m));
}

#[test]
fn w34_run_metrics_with_durations() {
    let m = RunInferenceMetrics {
        prompt_tokens: 10,
        generated_tokens: 5,
        total_duration: Duration::from_millis(500),
        prefill_duration: Duration::from_millis(100),
        decode_durations: vec![
            Duration::from_millis(80),
            Duration::from_millis(75),
            Duration::from_millis(82),
            Duration::from_millis(78),
            Duration::from_millis(85),
        ],
        peak_memory_bytes: Some(268_435_456),
    };
    insta::assert_snapshot!(run_metrics::format_metrics(&m));
}

#[test]
fn w34_aggregate_metrics_single_run() {
    let runs = vec![RunInferenceMetrics {
        prompt_tokens: 10,
        generated_tokens: 20,
        total_duration: Duration::from_millis(1000),
        prefill_duration: Duration::from_millis(100),
        decode_durations: vec![Duration::from_millis(45); 20],
        peak_memory_bytes: None,
    }];
    let agg = AggregateMetrics::from_runs(&runs);
    insta::assert_debug_snapshot!(agg);
}

#[test]
fn w34_aggregate_metrics_multiple_runs() {
    let runs: Vec<RunInferenceMetrics> = (1..=3)
        .map(|i| RunInferenceMetrics {
            prompt_tokens: 10,
            generated_tokens: 10 * i,
            total_duration: Duration::from_millis(500 * i as u64),
            prefill_duration: Duration::from_millis(50),
            decode_durations: vec![Duration::from_millis(40); 10 * i],
            peak_memory_bytes: None,
        })
        .collect();
    let agg = AggregateMetrics::from_runs(&runs);
    insta::assert_debug_snapshot!(agg);
}
