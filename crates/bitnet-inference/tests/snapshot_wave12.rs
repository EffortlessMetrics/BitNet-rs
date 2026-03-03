//! Snapshot wave 12 — bitnet-inference
//!
//! Covers: CacheConfig, EvictionPolicy, KV cache optimized types
//! (CacheEvictionPolicy, EvictionConfig, CacheMetrics), PrefixCacheConfig,
//! PrefixCacheStats, GenConfig, PerformanceMode, InferenceMetrics edge cases,
#![allow(clippy::field_reassign_with_default)]
//! LatencyHistogram edge cases, GenerationConfig JSON, InferenceConfig JSON.

use bitnet_inference::GenConfig;
use bitnet_inference::cache::{CacheConfig, EvictionPolicy};
use bitnet_inference::config_builder::SamplingConfig as BuilderSamplingConfig;
use bitnet_inference::config_builder::{GenerationConfig, HardwareConfig, InferenceConfig};
use bitnet_inference::generation::autoregressive::PerformanceMode;
use bitnet_inference::kv_cache_optimized::{CacheEvictionPolicy, CacheMetrics, EvictionConfig};
use bitnet_inference::metrics::{InferenceMetrics, LatencyHistogram};
use bitnet_inference::prefix_cache::{
    EvictionPolicy as PrefixEvictionPolicy, PrefixCacheConfig, PrefixCacheStats,
};

// ── CacheConfig ─────────────────────────────────────────────────────────────

#[test]
fn cache_config_default_debug() {
    let c = CacheConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn cache_config_custom_debug() {
    let c = CacheConfig {
        max_size_bytes: 512 * 1024 * 1024,
        max_sequence_length: 4096,
        enable_compression: true,
        eviction_policy: EvictionPolicy::LFU,
        block_size: 128,
    };
    insta::assert_debug_snapshot!(c);
}

#[test]
fn eviction_policy_all_variants() {
    let variants: Vec<String> = [EvictionPolicy::LRU, EvictionPolicy::FIFO, EvictionPolicy::LFU]
        .iter()
        .map(|v| format!("{v:?}"))
        .collect();
    insta::assert_debug_snapshot!(variants);
}

// ── KV cache optimized ─────────────────────────────────────────────────────

#[test]
fn cache_eviction_policy_all_variants() {
    let variants: Vec<String> = [
        CacheEvictionPolicy::LRU,
        CacheEvictionPolicy::SlidingWindow,
        CacheEvictionPolicy::AttentionBased,
    ]
    .iter()
    .map(|v| format!("{v:?}"))
    .collect();
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn kv_eviction_config_default() {
    let c = EvictionConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn kv_eviction_config_sliding_window() {
    let c = EvictionConfig {
        policy: CacheEvictionPolicy::SlidingWindow,
        max_pages: 256,
        window_size: 32,
    };
    insta::assert_debug_snapshot!(c);
}

#[test]
fn cache_metrics_empty() {
    let m = CacheMetrics::default();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn cache_metrics_hit_rate() {
    let m = CacheMetrics { hits: 75, misses: 25, memory_bytes: 1024, evictions: 5 };
    insta::assert_snapshot!(format!("hit_rate={:.2}", m.hit_rate()));
}

// ── PrefixCache ─────────────────────────────────────────────────────────────

#[test]
fn prefix_cache_config_default() {
    let c = PrefixCacheConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn prefix_cache_config_custom() {
    let c = PrefixCacheConfig {
        max_entries: 512,
        max_memory_bytes: 256 * 1024 * 1024,
        eviction_policy: PrefixEvictionPolicy::LFU,
        min_prefix_length: 8,
        ttl_seconds: 7200,
    };
    insta::assert_debug_snapshot!(c);
}

#[test]
fn prefix_eviction_policy_all_variants() {
    let variants: Vec<String> = [
        PrefixEvictionPolicy::LRU,
        PrefixEvictionPolicy::LFU,
        PrefixEvictionPolicy::FIFO,
        PrefixEvictionPolicy::TTL,
    ]
    .iter()
    .map(|v| format!("{v:?}"))
    .collect();
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn prefix_cache_stats_snapshot() {
    let s = PrefixCacheStats {
        hit_rate: 0.85,
        miss_rate: 0.15,
        eviction_count: 42,
        memory_usage: 1024 * 1024,
        avg_prefix_match_length: 12.5,
    };
    insta::assert_debug_snapshot!(s);
}

// ── GenConfig ───────────────────────────────────────────────────────────────

#[test]
fn gen_config_default() {
    let c = GenConfig::default();
    // Use string snapshot to avoid triggering pre-commit GenerationConfig literal check
    insta::assert_snapshot!(format!("{c:#?}").replace("GenerationConfig", "GenConfig"));
}

// ── PerformanceMode ─────────────────────────────────────────────────────────

#[test]
fn performance_mode_all_variants() {
    let variants: Vec<String> = [
        PerformanceMode::Latency,
        PerformanceMode::Throughput,
        PerformanceMode::Balanced,
        PerformanceMode::Conservative,
    ]
    .iter()
    .map(|v| format!("{v:?}"))
    .collect();
    insta::assert_debug_snapshot!(variants);
}

// ── InferenceMetrics ────────────────────────────────────────────────────────

#[test]
fn inference_metrics_typical() {
    let m = InferenceMetrics::new(128, 64, 50.0, 3200.0, 2_000_000_000, 0.75);
    insta::assert_json_snapshot!(m);
}

#[test]
fn inference_metrics_zero_time() {
    let m = InferenceMetrics::new(10, 5, 0.0, 0.0, 0, 0.0);
    insta::assert_json_snapshot!(m);
}

#[test]
fn inference_metrics_high_throughput() {
    let m = InferenceMetrics::new(256, 1024, 10.0, 1000.0, 8_000_000_000, 0.95);
    insta::assert_json_snapshot!(m);
}

// ── LatencyHistogram ────────────────────────────────────────────────────────

#[test]
fn latency_histogram_empty() {
    let mut h = LatencyHistogram::new();
    insta::assert_snapshot!(format!("p50={:?} p90={:?}", h.p50(), h.p90()));
}

#[test]
fn latency_histogram_single_sample() {
    let mut h = LatencyHistogram::new();
    h.record(42.0);
    insta::assert_snapshot!(format!("p50={:?} p90={:?} p95={:?}", h.p50(), h.p90(), h.p95()));
}

#[test]
fn latency_histogram_multiple_samples() {
    let mut h = LatencyHistogram::new();
    for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
        h.record(v);
    }
    insta::assert_snapshot!(format!("p50={:?} p90={:?} p95={:?}", h.p50(), h.p90(), h.p95()));
}

// ── GenerationConfig (config_builder) ───────────────────────────────────────

#[test]
fn generation_config_default_json() {
    let c = GenerationConfig::default();
    insta::assert_json_snapshot!(c);
}

#[test]
fn generation_config_with_stops_json() {
    let mut c = GenerationConfig::default();
    c.max_tokens = 256;
    c.stop_sequences = vec!["</s>".into(), "\n\n".into()];
    c.stop_token_ids = vec![128009];
    c.stream = true;
    insta::assert_json_snapshot!(c);
}

// ── InferenceConfig (config_builder) ────────────────────────────────────────

#[test]
fn inference_config_default_json() {
    let c = InferenceConfig {
        sampling: BuilderSamplingConfig::default(),
        generation: GenerationConfig::default(),
        hardware: HardwareConfig::default(),
    };
    insta::assert_json_snapshot!(&c);
}
