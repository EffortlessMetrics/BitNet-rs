//! Wave 18 snapshot tests for inference engine and model structures.
//!
//! Covers: GenerationConfig presets, SamplingConfig/SamplingStrategy,
//! InferenceConfig variants, CacheConfig, KVCache stats, InferenceMetrics,
//! ThroughputMetrics, TimingMetrics, InferenceResult, PerformanceMetrics,
//! ModelConfig defaults, BatchConfig, StreamingConfig, MetricsReport,
//! LatencyHistogram, and MemoryProfiler.

use std::time::Duration;

use bitnet_common::ModelConfig;
use bitnet_inference::cache::{CacheConfig, KVCache};
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::engine::{InferenceResult, PerformanceMetrics};
use bitnet_inference::generation::{
    GenConfig, SampleConfig, SamplingStrategy as GenSamplingStrategy,
};
use bitnet_inference::metrics::{
    InferenceMetrics, LatencyHistogram, MemoryProfiler, MetricsCollector, MetricsReport,
    ThroughputTracker,
};
use bitnet_inference::streaming::StreamingConfig;
use bitnet_inference::{BatchConfig, ThroughputMetrics, TimingMetrics};

// ============================================================================
// GenerationConfig presets
// ============================================================================

#[test]
fn w18_generation_config_default_debug() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_generation_config_greedy_debug() {
    let cfg = GenerationConfig::greedy();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_generation_config_creative_debug() {
    let cfg = GenerationConfig::creative();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_generation_config_balanced_debug() {
    let cfg = GenerationConfig::balanced();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_generation_config_with_stops() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_stop_sequences(vec!["</s>".into(), "\n\n".into()])
        .with_stop_token_ids(vec![128009, 128001])
        .with_eos_token_id(Some(2))
        .with_seed(42);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_generation_config_logits_tap() {
    let cfg = GenerationConfig::default().with_logits_tap_steps(5).with_logits_topk(20);
    insta::assert_snapshot!(format!(
        "logits_tap_steps={} logits_topk={}",
        cfg.logits_tap_steps, cfg.logits_topk
    ));
}

// ============================================================================
// Generation-module SamplingConfig / SamplingStrategy
// ============================================================================

#[test]
fn w18_gen_sample_config_default_debug() {
    let cfg = SampleConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_gen_sampling_strategy_default_debug() {
    let cfg = SampleConfig::default();
    let strategy = GenSamplingStrategy::new(cfg);
    insta::assert_debug_snapshot!(strategy);
}

#[test]
fn w18_gen_sampling_strategy_greedy_debug() {
    let cfg = SampleConfig {
        temperature: 0.0,
        top_k: None,
        top_p: None,
        repetition_penalty: 1.0,
        do_sample: false,
    };
    let strategy = GenSamplingStrategy::new(cfg);
    insta::assert_debug_snapshot!(strategy);
}

// ============================================================================
// Generation-module GenConfig (autoregressive)
// ============================================================================

#[test]
fn w18_gen_config_default_debug() {
    let cfg = GenConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// InferenceConfig
// ============================================================================

#[test]
fn w18_inference_config_cpu_optimized() {
    // Pin thread count for reproducible snapshots
    let mut cfg = InferenceConfig::cpu_optimized();
    cfg.num_threads = 4;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_inference_config_gpu_optimized() {
    let mut cfg = InferenceConfig::gpu_optimized();
    cfg.num_threads = 4;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_inference_config_memory_efficient() {
    let mut cfg = InferenceConfig::memory_efficient();
    cfg.num_threads = 4;
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// CacheConfig
// ============================================================================

#[test]
fn w18_cache_config_default_debug() {
    let cfg = CacheConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// KVCache stats after operations
// ============================================================================

#[test]
fn w18_kv_cache_empty_stats() {
    let cfg = CacheConfig::default();
    let cache = KVCache::new(cfg).unwrap();
    insta::assert_debug_snapshot!(cache.stats());
}

#[test]
fn w18_kv_cache_after_store() {
    let cfg = CacheConfig::default();
    let mut cache = KVCache::new(cfg).unwrap();
    cache.store(0, 0, vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    cache.store(0, 1, vec![9.0, 10.0, 11.0, 12.0], vec![13.0, 14.0, 15.0, 16.0]).unwrap();
    cache.store(1, 0, vec![0.5; 4], vec![0.25; 4]).unwrap();
    insta::assert_debug_snapshot!(cache.stats());
}

#[test]
fn w18_kv_cache_after_hits_and_misses() {
    let cfg = CacheConfig::default();
    let mut cache = KVCache::new(cfg).unwrap();
    cache.store(0, 0, vec![1.0; 4], vec![2.0; 4]).unwrap();
    // 2 hits, 1 miss
    let _ = cache.get(0, 0);
    let _ = cache.get(0, 0);
    let _ = cache.get(0, 99);
    insta::assert_debug_snapshot!(cache.stats());
}

#[test]
fn w18_kv_cache_prefill_tracking() {
    let cfg = CacheConfig::default();
    let mut cache = KVCache::new(cfg).unwrap();
    cache.record_prefill(64);
    cache.record_incremental(1);
    cache.record_incremental(1);
    insta::assert_snapshot!(format!(
        "prefilled={} total={}",
        cache.num_tokens_prefilled(),
        cache.num_tokens_total()
    ));
}

// ============================================================================
// InferenceMetrics
// ============================================================================

#[test]
fn w18_inference_metrics_sample() {
    let m = InferenceMetrics::new(128, 64, 25.0, 3200.0, 1_048_576, 0.85);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_inference_metrics_zero_time() {
    let m = InferenceMetrics::new(0, 0, 0.0, 0.0, 0, 0.0);
    insta::assert_debug_snapshot!(m);
}

// ============================================================================
// MetricsCollector → snapshot
// ============================================================================

#[test]
fn w18_metrics_collector_after_requests() {
    let c = MetricsCollector::new();
    c.record_request(32, 16, 2_000_000_000, 50_000_000);
    c.record_request(64, 32, 4_000_000_000, 80_000_000);
    c.record_cache_hit();
    c.record_cache_hit();
    c.record_cache_miss();
    c.update_peak_memory(2_097_152);
    insta::assert_debug_snapshot!(c.snapshot());
}

// ============================================================================
// ThroughputMetrics & TimingMetrics
// ============================================================================

#[test]
fn w18_throughput_metrics_default_debug() {
    let m = ThroughputMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_throughput_metrics_sample() {
    let m = ThroughputMetrics {
        prefill_tokens_per_sec: Some(1200.0),
        decode_tokens_per_sec: Some(45.5),
        end_to_end_tokens_per_sec: 42.0,
        total_tokens: 128,
    };
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_timing_metrics_default_debug() {
    let m = TimingMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_timing_metrics_sample() {
    let m = TimingMetrics {
        prefill_ms: Some(120),
        decode_ms: Some(3000),
        tokenization_encode_ms: Some(5),
        tokenization_decode_ms: Some(3),
        total_ms: 3128,
    };
    insta::assert_debug_snapshot!(m);
}

// ============================================================================
// PerformanceMetrics & InferenceResult
// ============================================================================

#[test]
fn w18_performance_metrics_default_debug() {
    let m = PerformanceMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_performance_metrics_sample() {
    let m = PerformanceMetrics {
        total_latency_ms: 5000,
        tokens_generated: 128,
        tokens_per_second: 25.6,
        first_token_latency_ms: Some(45),
        average_token_latency_ms: Some(39.0),
        memory_usage_bytes: Some(536_870_912),
        cache_hit_rate: Some(0.92),
        backend_type: "cpu-avx2".to_string(),
        model_load_time_ms: Some(1200),
        tokenizer_encode_time_ms: Some(5),
        tokenizer_decode_time_ms: Some(3),
        forward_pass_time_ms: Some(4500),
        sampling_time_ms: Some(200),
    };
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w18_inference_result_sample() {
    let perf = PerformanceMetrics {
        total_latency_ms: 2000,
        tokens_generated: 32,
        tokens_per_second: 16.0,
        first_token_latency_ms: Some(50),
        average_token_latency_ms: Some(62.5),
        memory_usage_bytes: None,
        cache_hit_rate: None,
        backend_type: "cpu".to_string(),
        model_load_time_ms: None,
        tokenizer_encode_time_ms: None,
        tokenizer_decode_time_ms: None,
        forward_pass_time_ms: None,
        sampling_time_ms: None,
    };
    let result =
        InferenceResult::new("The capital of France is Paris.".to_string(), 32, 2000, 16.0, perf);
    insta::assert_debug_snapshot!(result);
}

// ============================================================================
// ModelConfig (bitnet-common)
// ============================================================================

#[test]
fn w18_model_config_default_debug() {
    let cfg = ModelConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// BatchConfig
// ============================================================================

#[test]
fn w18_batch_config_default_debug() {
    let cfg = BatchConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w18_batch_config_custom() {
    let cfg = BatchConfig::new(16, Duration::from_millis(500)).with_max_total_tokens(16384);
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// StreamingConfig
// ============================================================================

#[test]
fn w18_streaming_config_default_debug() {
    let cfg = StreamingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// LatencyHistogram
// ============================================================================

#[test]
fn w18_latency_histogram_with_samples() {
    let mut h = LatencyHistogram::new();
    for i in 1..=20 {
        h.record(i as f64 * 5.0);
    }
    insta::assert_snapshot!(format!(
        "count={} mean={:.1} min={:.1} max={:.1} p50={:.1} p95={:.1} p99={:.1}",
        h.count(),
        h.mean().unwrap(),
        h.min().unwrap(),
        h.max().unwrap(),
        h.p50().unwrap(),
        h.p95().unwrap(),
        h.p99().unwrap(),
    ));
}

// ============================================================================
// MemoryProfiler
// ============================================================================

#[test]
fn w18_memory_profiler_after_ops() {
    let mp = MemoryProfiler::new();
    mp.record_allocation(4096);
    mp.record_allocation(8192);
    mp.record_deallocation(2048);
    insta::assert_snapshot!(format!(
        "current={} peak={} allocs={} deallocs={}",
        mp.current_bytes(),
        mp.peak_bytes(),
        mp.allocation_count(),
        mp.deallocation_count(),
    ));
}

// ============================================================================
// MetricsReport
// ============================================================================

#[test]
fn w18_metrics_report_build() {
    let c = MetricsCollector::new();
    c.record_request(32, 48, 3_000_000_000, 60_000_000);
    c.record_cache_hit();
    c.record_cache_hit();
    c.record_cache_miss();
    c.update_peak_memory(4_194_304);

    let mut h = LatencyHistogram::new();
    h.record(30.0);
    h.record(45.0);
    h.record(60.0);

    let t = ThroughputTracker::new(Duration::from_mins(1));
    let mp = MemoryProfiler::new();
    mp.record_allocation(4_194_304);

    let report = MetricsReport::build(&c, &mut h, &t, &mp);
    insta::assert_debug_snapshot!(report);
}
