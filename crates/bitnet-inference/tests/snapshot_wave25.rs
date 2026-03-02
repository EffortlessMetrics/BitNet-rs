//! Wave 25 snapshot tests for bitnet-inference.
//!
//! Covers: Engine configuration snapshots, sampling strategy parameter snapshots,
//! generation output format snapshots, and token stream state snapshots.

use std::time::Duration;

// =========================================================================
// Section 1 — Token stream configuration and events
// =========================================================================

use bitnet_inference::token_stream::{StreamConfig, StreamEvent, StreamStats};

#[test]
fn w25_stream_config_default_debug() {
    let cfg = StreamConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_stream_event_token() {
    let evt = StreamEvent::Token(42);
    insta::assert_debug_snapshot!(evt);
}

#[test]
fn w25_stream_event_text() {
    let evt = StreamEvent::Text("Hello, world!".into());
    insta::assert_debug_snapshot!(evt);
}

#[test]
fn w25_stream_event_all_variants() {
    let events: Vec<StreamEvent> = vec![
        StreamEvent::Token(100),
        StreamEvent::Text("hello".into()),
        StreamEvent::EndOfStream,
        StreamEvent::Error("invalid UTF-8 sequence".into()),
    ];
    insta::assert_debug_snapshot!(events);
}

#[test]
fn w25_stream_stats_default() {
    let stats = StreamStats::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 2 — Warmup configuration and results
// =========================================================================

use bitnet_inference::warmup::{WarmupConfig, WarmupResult, WarmupStatus};

#[test]
fn w25_warmup_config_default_debug() {
    let cfg = WarmupConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_warmup_config_fast_debug() {
    let cfg = WarmupConfig::fast();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_warmup_config_thorough_debug() {
    let cfg = WarmupConfig::thorough();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_warmup_config_custom() {
    let cfg = WarmupConfig::default()
        .with_iterations(10)
        .with_seq_len(256)
        .with_timeout(Duration::from_secs(120));
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_warmup_status_all_variants() {
    let variants = [WarmupStatus::Complete, WarmupStatus::TimedOut, WarmupStatus::Skipped];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w25_warmup_result_success() {
    let result = WarmupResult {
        iterations_completed: 3,
        total_time: Duration::from_millis(450),
        iteration_times: vec![
            Duration::from_millis(200),
            Duration::from_millis(150),
            Duration::from_millis(100),
        ],
        timed_out: false,
        status: WarmupStatus::Complete,
    };
    insta::assert_snapshot!(format!(
        "completed={} avg={:?} success={}",
        result.iterations_completed,
        result.avg_iteration_time(),
        result.is_success()
    ));
}

#[test]
fn w25_warmup_result_timed_out() {
    let result = WarmupResult {
        iterations_completed: 1,
        total_time: Duration::from_secs(30),
        iteration_times: vec![Duration::from_secs(30)],
        timed_out: true,
        status: WarmupStatus::TimedOut,
    };
    insta::assert_debug_snapshot!(result);
}

// =========================================================================
// Section 3 — Memory pool configuration
// =========================================================================

use bitnet_inference::memory_pool::{GrowthStrategy, PoolConfig, PoolStatistics};

#[test]
fn w25_pool_config_default_debug() {
    let cfg = PoolConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_pool_config_builder_custom() {
    let cfg = PoolConfig::builder()
        .initial_size(4 << 20)
        .growth_strategy(GrowthStrategy::Fixed)
        .max_pool_size(128 << 20)
        .max_allocation_size(16 << 20)
        .build();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_growth_strategy_variants() {
    let strategies = [GrowthStrategy::Fixed, GrowthStrategy::Double];
    insta::assert_debug_snapshot!(strategies);
}

#[test]
fn w25_pool_statistics_default() {
    let stats = PoolStatistics::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 4 — KV cache eviction policies
// =========================================================================

use bitnet_inference::kv_cache_optimized::{CacheEvictionPolicy, CacheMetrics, EvictionConfig};

#[test]
fn w25_cache_eviction_policy_all_variants() {
    let policies = [
        CacheEvictionPolicy::LRU,
        CacheEvictionPolicy::SlidingWindow,
        CacheEvictionPolicy::AttentionBased,
    ];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn w25_eviction_config_default_debug() {
    let cfg = EvictionConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_eviction_config_sliding_window() {
    let cfg = EvictionConfig {
        policy: CacheEvictionPolicy::SlidingWindow,
        max_pages: 512,
        window_size: 32,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_cache_metrics_empty() {
    let m = CacheMetrics::default();
    insta::assert_snapshot!(format!("hit_rate={:.4}", m.hit_rate()));
}

#[test]
fn w25_cache_metrics_populated() {
    let m = CacheMetrics { hits: 950, misses: 50, memory_bytes: 33_554_432, evictions: 12 };
    insta::assert_snapshot!(format!(
        "hits={} misses={} hit_rate={:.4} evictions={} memory_bytes={}",
        m.hits,
        m.misses,
        m.hit_rate(),
        m.evictions,
        m.memory_bytes
    ));
}

// =========================================================================
// Section 5 — Profiler configuration
// =========================================================================

use bitnet_inference::profiler::{LayerProfile, MemorySnapshot, ProfilerConfig};

#[test]
fn w25_profiler_config_default_debug() {
    let cfg = ProfilerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_profiler_config_disabled_debug() {
    let cfg = ProfilerConfig::disabled();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_profiler_config_custom() {
    let cfg = ProfilerConfig::default().with_warmup(5).with_sample_size(100).with_memory(true);
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_layer_profile_attention() {
    let profile = LayerProfile {
        layer_name: "layer_0.attention".into(),
        layer_type: "attention".into(),
        forward_time_us: 1250.5,
        backward_time_us: 0.0,
        memory_bytes: 4_194_304,
        flops_estimate: 805_306_368,
    };
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn w25_layer_profile_ffn() {
    let profile = LayerProfile {
        layer_name: "layer_0.ffn".into(),
        layer_type: "ffn".into(),
        forward_time_us: 850.3,
        backward_time_us: 0.0,
        memory_bytes: 8_388_608,
        flops_estimate: 1_610_612_736,
    };
    insta::assert_debug_snapshot!(profile);
}

#[test]
fn w25_memory_snapshot_debug() {
    let snap = MemorySnapshot {
        label: "post_attention_layer_5".into(),
        timestamp_us: 15432.7,
        memory_bytes: 67_108_864,
    };
    insta::assert_debug_snapshot!(snap);
}

// =========================================================================
// Section 6 — Streaming configuration
// =========================================================================

use bitnet_inference::streaming::{GenerationStats, StreamingConfig};

#[test]
fn w25_streaming_config_default_debug() {
    let cfg = StreamingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_generation_stats_default_debug() {
    let stats = GenerationStats::default();
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 7 — Prefix cache configuration
// =========================================================================

use bitnet_inference::prefix_cache::{EvictionPolicy, PrefixCacheConfig, PrefixCacheStats};

#[test]
fn w25_eviction_policy_all_variants() {
    let policies =
        [EvictionPolicy::LRU, EvictionPolicy::LFU, EvictionPolicy::FIFO, EvictionPolicy::TTL];
    insta::assert_debug_snapshot!(policies);
}

#[test]
fn w25_prefix_cache_config_default_debug() {
    let cfg = PrefixCacheConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_prefix_cache_config_custom() {
    let cfg = PrefixCacheConfig {
        max_entries: 256,
        max_memory_bytes: 128 * 1024 * 1024,
        eviction_policy: EvictionPolicy::LFU,
        min_prefix_length: 8,
        ttl_seconds: 7200,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w25_prefix_cache_stats_snapshot() {
    let stats = PrefixCacheStats {
        hit_rate: 0.85,
        miss_rate: 0.15,
        eviction_count: 42,
        memory_usage: 67_108_864,
        avg_prefix_match_length: 12.5,
    };
    insta::assert_debug_snapshot!(stats);
}

// =========================================================================
// Section 8 — Batch configuration
// =========================================================================

use bitnet_inference::batch::BatchConfig;

#[test]
fn w25_batch_config_default_debug() {
    let cfg = BatchConfig::default();
    insta::assert_debug_snapshot!(cfg);
}
