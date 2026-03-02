//! Wave 24 snapshot tests — inference pipeline configs, batch engine,
//! token stream, prefix cache, kv-cache-optimized, thread pool, and
//! production engine structs.
//!
//! Covers: ProductionInferenceConfig, PrefillStrategy, DeviceCapabilities,
//! GenerationResult, BatchEngine types, StreamConfig, StreamEvent, StreamStats,
//! PrefixCache types, PagedKvCache types, ThreadPool types.

use bitnet_inference::engine::PerformanceMetrics;
use bitnet_inference::production_engine::{
    DeviceCapabilities, GenerationResult, PrefillStrategy, ProductionInferenceConfig,
};
use bitnet_inference::{ThroughputMetrics, TimingMetrics};

// ============================================================================
// Section 1 — ProductionInferenceConfig / PrefillStrategy
// ============================================================================

#[test]
fn w24_production_config_default() {
    let cfg = ProductionInferenceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_prefill_strategy_always() {
    let s = PrefillStrategy::Always;
    insta::assert_debug_snapshot!(s);
}

#[test]
fn w24_prefill_strategy_adaptive() {
    let s = PrefillStrategy::Adaptive { threshold_tokens: 64 };
    insta::assert_debug_snapshot!(s);
}

#[test]
fn w24_prefill_strategy_never() {
    let s = PrefillStrategy::Never;
    insta::assert_debug_snapshot!(s);
}

#[test]
fn w24_production_config_custom() {
    let cfg = ProductionInferenceConfig {
        enable_performance_monitoring: false,
        enable_memory_tracking: false,
        max_inference_time_seconds: 60,
        enable_quality_assessment: true,
        prefill_strategy: PrefillStrategy::Always,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ============================================================================
// Section 2 — DeviceCapabilities
// ============================================================================

#[test]
fn w24_device_capabilities_cpu_only() {
    let caps = DeviceCapabilities {
        memory_bytes: None,
        compute_capability: None,
        supports_mixed_precision: false,
        optimal_batch_size: 1,
    };
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w24_device_capabilities_gpu() {
    let caps = DeviceCapabilities {
        memory_bytes: Some(8_589_934_592), // 8 GiB
        compute_capability: Some((8, 6)),
        supports_mixed_precision: true,
        optimal_batch_size: 32,
    };
    insta::assert_debug_snapshot!(caps);
}

// ============================================================================
// Section 3 — GenerationResult
// ============================================================================

#[test]
fn w24_generation_result_basic() {
    let perf = PerformanceMetrics {
        total_latency_ms: 1000,
        tokens_generated: 16,
        tokens_per_second: 16.0,
        first_token_latency_ms: Some(80),
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
    let timing = TimingMetrics {
        prefill_ms: Some(80),
        decode_ms: Some(920),
        tokenization_encode_ms: Some(2),
        tokenization_decode_ms: Some(1),
        total_ms: 1003,
    };
    let throughput = ThroughputMetrics {
        prefill_tokens_per_sec: Some(500.0),
        decode_tokens_per_sec: Some(17.4),
        end_to_end_tokens_per_sec: 16.0,
        total_tokens: 16,
    };
    let result =
        GenerationResult::new("The answer is 42.".to_string(), 16, perf, timing, throughput);
    insta::assert_debug_snapshot!(result);
}

#[test]
fn w24_generation_result_with_quality() {
    let perf = PerformanceMetrics::default();
    let timing = TimingMetrics::default();
    let throughput = ThroughputMetrics::default();
    let mut result = GenerationResult::new("Hello".to_string(), 1, perf, timing, throughput);
    result.quality_score = Some(0.75);
    insta::assert_debug_snapshot!(result);
}

// ============================================================================
// Section 4 — Batch engine types
// ============================================================================

use bitnet_inference::batch_engine::{
    BatchConfig as BEBatchConfig, BatchRequest, BatchResponse, BatchStats, Priority,
    SchedulingPolicy,
};

#[test]
fn w24_priority_all_variants() {
    let variants: Vec<Priority> =
        vec![Priority::Low, Priority::Normal, Priority::High, Priority::Critical];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w24_scheduling_policy_all_variants() {
    let variants: Vec<SchedulingPolicy> = vec![
        SchedulingPolicy::FIFO,
        SchedulingPolicy::PriorityBased,
        SchedulingPolicy::ShortestJobFirst,
        SchedulingPolicy::RoundRobin,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w24_batch_request_sample() {
    let req = BatchRequest {
        id: "req-001".to_string(),
        prompt: "What is 2+2?".to_string(),
        max_tokens: 32,
        temperature: 0.7,
        priority: Priority::Normal,
    };
    insta::assert_debug_snapshot!(req);
}

#[test]
fn w24_batch_response_sample() {
    let resp = BatchResponse {
        request_id: "req-001".to_string(),
        text: "The answer is 4.".to_string(),
        tokens_generated: 5,
        finish_reason: "stop".to_string(),
        time_ms: 250,
    };
    insta::assert_debug_snapshot!(resp);
}

#[test]
fn w24_batch_config_default() {
    let cfg = BEBatchConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_batch_config_custom() {
    let cfg = BEBatchConfig {
        max_batch_size: 32,
        max_queue_depth: 5000,
        scheduling_policy: SchedulingPolicy::PriorityBased,
        timeout_ms: 60_000,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_batch_stats_default() {
    let stats = BatchStats::default();
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w24_batch_stats_populated() {
    let stats = BatchStats {
        total_requests: 1000,
        completed_requests: 980,
        failed_requests: 20,
        avg_latency_ms: 125.5,
        avg_tokens_per_second: 42.3,
        queue_depth: 5,
        active_batch_size: 8,
        uptime_ms: 3_600_000,
    };
    insta::assert_debug_snapshot!(stats);
}

// ============================================================================
// Section 5 — Token stream types
// ============================================================================

use bitnet_inference::token_stream::{StreamConfig, StreamEvent, StreamStats, TokenBuffer};

#[test]
fn w24_stream_config_default() {
    let cfg = StreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_stream_config_low_latency() {
    let cfg = StreamConfig {
        buffer_size: 1,
        flush_on_whitespace: true,
        flush_on_newline: true,
        max_pending_tokens: 8,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_stream_event_all_variants() {
    let events: Vec<StreamEvent> = vec![
        StreamEvent::Token(42),
        StreamEvent::Text("hello".to_string()),
        StreamEvent::EndOfStream,
        StreamEvent::Error("invalid utf8".to_string()),
    ];
    insta::assert_debug_snapshot!(events);
}

#[test]
fn w24_stream_stats_default() {
    let stats = StreamStats::default();
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w24_stream_stats_populated() {
    let stats = StreamStats {
        tokens_generated: 128,
        text_chunks_emitted: 32,
        avg_tokens_per_chunk: 4.0,
        total_bytes: 512,
    };
    insta::assert_debug_snapshot!(stats);
}

#[test]
fn w24_token_buffer_empty() {
    let buf = TokenBuffer::new();
    insta::assert_debug_snapshot!(buf);
}

// ============================================================================
// Section 6 — Prefix cache types
// ============================================================================

use bitnet_inference::prefix_cache::{EvictionPolicy, PrefixCacheConfig, PrefixCacheStats};

#[test]
fn w24_prefix_eviction_policy_all_variants() {
    let policies: Vec<EvictionPolicy> =
        vec![EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO, EvictionPolicy::TTL];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn w24_prefix_cache_config_default() {
    let cfg = PrefixCacheConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_prefix_cache_config_small() {
    let cfg = PrefixCacheConfig {
        max_entries: 64,
        max_memory_bytes: 32 * 1024 * 1024,
        eviction_policy: EvictionPolicy::LFU,
        min_prefix_length: 8,
        ttl_seconds: 1800,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_prefix_cache_stats_sample() {
    let stats = PrefixCacheStats {
        hit_rate: 0.85,
        miss_rate: 0.15,
        eviction_count: 42,
        memory_usage: 128 * 1024 * 1024,
        avg_prefix_match_length: 12.5,
    };
    insta::assert_debug_snapshot!(stats);
}

// ============================================================================
// Section 7 — KV cache optimized types
// ============================================================================

use bitnet_inference::kv_cache_optimized::{CacheEvictionPolicy, CacheMetrics, EvictionConfig};

#[test]
fn w24_cache_eviction_policy_all_variants() {
    let policies: Vec<CacheEvictionPolicy> = vec![
        CacheEvictionPolicy::LRU,
        CacheEvictionPolicy::SlidingWindow,
        CacheEvictionPolicy::AttentionBased,
    ];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn w24_eviction_config_default() {
    let cfg = EvictionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_eviction_config_sliding_window() {
    let cfg = EvictionConfig {
        policy: CacheEvictionPolicy::SlidingWindow,
        max_pages: 512,
        window_size: 32,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_cache_metrics_default() {
    let m = CacheMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w24_cache_metrics_populated() {
    let m = CacheMetrics {
        hits: 9500,
        misses: 500,
        memory_bytes: 67_108_864, // 64 MiB
        evictions: 120,
    };
    insta::assert_snapshot!(format!(
        "hits={} misses={} memory_bytes={} evictions={} hit_rate={:.4}",
        m.hits,
        m.misses,
        m.memory_bytes,
        m.evictions,
        m.hit_rate()
    ));
}

// ============================================================================
// Section 8 — Thread pool types
// ============================================================================

use bitnet_inference::thread_pool::{ThreadPoolConfig, ThreadPoolMetrics};

#[test]
fn w24_thread_pool_config_custom() {
    let cfg = ThreadPoolConfig {
        num_threads: 8,
        affinity: true,
        priority: 2,
        name_prefix: "bitnet-w24".to_string(),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_thread_pool_metrics_idle() {
    let m = ThreadPoolMetrics {
        active_threads: 0,
        queue_depth: 0,
        tasks_completed: 0,
        utilization: 0.0,
    };
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w24_thread_pool_metrics_busy() {
    let m = ThreadPoolMetrics {
        active_threads: 6,
        queue_depth: 12,
        tasks_completed: 50_000,
        utilization: 0.78,
    };
    insta::assert_debug_snapshot!(m);
}
