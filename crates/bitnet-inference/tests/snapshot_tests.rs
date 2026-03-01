//! Snapshot tests for stable bitnet-inference configuration defaults and error formats.
//! These test key structural constants that should never silently change.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};

#[test]
fn generation_config_default_max_new_tokens() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(format!("max_new_tokens={}", cfg.max_new_tokens));
}

#[test]
fn generation_config_default_temperature() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(format!("temperature={}", cfg.temperature));
}

#[test]
fn generation_config_greedy_temperature_is_zero() {
    let cfg = GenerationConfig::greedy();
    insta::assert_snapshot!(format!("temperature={} top_k={}", cfg.temperature, cfg.top_k));
}

#[test]
fn generation_config_creative_temperature_and_top_k() {
    let cfg = GenerationConfig::creative();
    insta::assert_snapshot!(format!(
        "temperature={} top_k={} top_p={}",
        cfg.temperature, cfg.top_k, cfg.top_p
    ));
}

#[test]
fn inference_config_default_context_length() {
    let cfg = InferenceConfig::default();
    insta::assert_snapshot!(format!("max_context_length={}", cfg.max_context_length));
}

#[test]
fn inference_config_default_batch_size() {
    let cfg = InferenceConfig::default();
    insta::assert_snapshot!(format!("batch_size={}", cfg.batch_size));
}

#[test]
fn generation_config_validate_zero_tokens_error() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

// == Wave 4: thread_pool =====================================================

use bitnet_inference::thread_pool::{ThreadPoolConfig, ThreadPoolMetrics};

#[test]
fn thread_pool_config_default_snapshot() {
    let cfg = ThreadPoolConfig { num_threads: 4, ..ThreadPoolConfig::default() };
    insta::assert_snapshot!(
        "thread_pool_config_default",
        format!(
            "num_threads={} affinity={} priority={} name_prefix={}",
            cfg.num_threads, cfg.affinity, cfg.priority, cfg.name_prefix,
        )
    );
}

#[test]
fn thread_pool_config_custom_snapshot() {
    let cfg = ThreadPoolConfig {
        num_threads: 16,
        affinity: true,
        priority: 2,
        name_prefix: "custom-inf".to_string(),
    };
    insta::assert_debug_snapshot!("thread_pool_config_custom", cfg);
}

#[test]
fn thread_pool_metrics_idle_snapshot() {
    let metrics = ThreadPoolMetrics {
        active_threads: 0,
        queue_depth: 0,
        tasks_completed: 0,
        utilization: 0.0,
    };
    insta::assert_snapshot!(
        "thread_pool_metrics_idle",
        format!(
            "active={} queue={} completed={} utilization={:.4}",
            metrics.active_threads,
            metrics.queue_depth,
            metrics.tasks_completed,
            metrics.utilization,
        )
    );
}

#[test]
fn thread_pool_metrics_active_snapshot() {
    let metrics = ThreadPoolMetrics {
        active_threads: 4,
        queue_depth: 12,
        tasks_completed: 1000,
        utilization: 0.75,
    };
    insta::assert_snapshot!(
        "thread_pool_metrics_active",
        format!(
            "active={} queue={} completed={} utilization={:.4}",
            metrics.active_threads,
            metrics.queue_depth,
            metrics.tasks_completed,
            metrics.utilization,
        )
    );
}

// == Wave 4: metrics =========================================================

use bitnet_inference::metrics::{
    InferenceMetrics, LatencyHistogram, MemoryProfiler, MetricsCollector, MetricsReport,
};

#[test]
fn inference_metrics_snapshot() {
    let metrics = InferenceMetrics::new(128, 64, 15.5, 3200.0, 1_073_741_824, 0.85);
    insta::assert_debug_snapshot!("inference_metrics_full", metrics);
}

#[test]
fn inference_metrics_zero_time() {
    let metrics = InferenceMetrics::new(10, 0, 0.0, 0.0, 0, 0.0);
    insta::assert_snapshot!(
        "inference_metrics_zero_time",
        format!(
            "tokens_per_second={} generated_tokens={}",
            metrics.tokens_per_second, metrics.generated_tokens,
        )
    );
}

#[test]
fn metrics_collector_snapshot_after_requests() {
    let collector = MetricsCollector::new();
    collector.record_request(128, 64, 3_200_000_000, 15_500_000);
    collector.record_cache_hit();
    collector.record_cache_hit();
    collector.record_cache_miss();
    collector.update_peak_memory(1_073_741_824);
    let snapshot = collector.snapshot();
    insta::assert_snapshot!(
        "metrics_collector_snapshot",
        format!(
            "prompt_tokens={} generated_tokens={} peak_memory_bytes={}",
            snapshot.prompt_tokens, snapshot.generated_tokens, snapshot.peak_memory_bytes,
        )
    );
}

#[test]
fn latency_histogram_percentiles_snapshot() {
    let mut hist = LatencyHistogram::new();
    for &v in &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
        hist.record(v);
    }
    insta::assert_snapshot!(
        "latency_histogram_percentiles",
        format!(
            "count={} mean={:.1} min={:.1} max={:.1} p50={:.1} p90={:.1} p99={:.1}",
            hist.count(),
            hist.mean().unwrap(),
            hist.min().unwrap(),
            hist.max().unwrap(),
            hist.p50().unwrap(),
            hist.p90().unwrap(),
            hist.p99().unwrap(),
        )
    );
}

#[test]
fn memory_profiler_snapshot() {
    let profiler = MemoryProfiler::new();
    profiler.record_allocation(1024);
    profiler.record_allocation(2048);
    profiler.record_deallocation(1024);
    insta::assert_snapshot!(
        "memory_profiler_state",
        format!(
            "current_bytes={} peak_bytes={} alloc_count={} dealloc_count={}",
            profiler.current_bytes(),
            profiler.peak_bytes(),
            profiler.allocation_count(),
            profiler.deallocation_count(),
        )
    );
}

#[test]
fn metrics_report_json_snapshot() {
    let collector = MetricsCollector::new();
    collector.record_request(64, 32, 2_000_000_000, 10_000_000);
    collector.record_cache_hit();
    collector.update_peak_memory(536_870_912);

    let mut histogram = LatencyHistogram::new();
    for &v in &[10.0, 20.0, 30.0, 40.0, 50.0] {
        histogram.record(v);
    }

    let memory = MemoryProfiler::new();
    memory.record_allocation(536_870_912);

    let report = MetricsReport {
        inference: collector.snapshot(),
        latency_p50_ms: histogram.p50(),
        latency_p90_ms: histogram.p90(),
        latency_p95_ms: histogram.p95(),
        latency_p99_ms: histogram.p99(),
        latency_mean_ms: histogram.mean(),
        latency_min_ms: histogram.min(),
        latency_max_ms: histogram.max(),
        latency_samples: histogram.count(),
        throughput_tps: 16.0,
        memory_current_bytes: memory.current_bytes(),
        memory_peak_bytes: memory.peak_bytes(),
        memory_allocation_count: memory.allocation_count(),
        memory_deallocation_count: memory.deallocation_count(),
    };

    insta::assert_json_snapshot!("metrics_report_json", report);
}

// == Wave 4: batch ===========================================================

use bitnet_inference::batch::{BatchConfig, BatchRequest, BatchScheduler};
use std::time::Duration;

#[test]
fn batch_config_default_snapshot() {
    let cfg = BatchConfig::default();
    insta::assert_snapshot!(
        "batch_config_default",
        format!(
            "max_batch_size={} timeout_ms={} max_total_tokens={}",
            cfg.max_batch_size,
            cfg.timeout.as_millis(),
            cfg.max_total_tokens,
        )
    );
}

#[test]
fn batch_config_custom_snapshot() {
    let cfg = BatchConfig::new(16, Duration::from_millis(500)).with_max_total_tokens(16384);
    insta::assert_debug_snapshot!("batch_config_custom", cfg);
}

#[test]
fn batch_config_json_roundtrip() {
    let cfg = BatchConfig::new(4, Duration::from_secs(10)).with_max_total_tokens(4096);
    insta::assert_json_snapshot!("batch_config_json", cfg);
}

#[test]
fn batch_scheduler_empty_request() {
    let scheduler = BatchScheduler::new(BatchConfig::default());
    let batch = BatchRequest::new();
    let order = scheduler.schedule(&batch);
    insta::assert_debug_snapshot!("batch_scheduler_empty", order);
}

#[test]
fn batch_scheduler_ordering_snapshot() {
    let scheduler = BatchScheduler::new(
        BatchConfig::new(4, Duration::from_secs(30)).with_max_total_tokens(8192),
    );
    let mut batch = BatchRequest::new();
    batch.add(
        "This is a long prompt with many words to test ordering".into(),
        GenerationConfig::greedy(),
    );
    batch.add("Short".into(), GenerationConfig::greedy());
    batch.add("Medium length prompt".into(), GenerationConfig::greedy());
    let order = scheduler.schedule(&batch);
    insta::assert_debug_snapshot!("batch_scheduler_ordering", order);
}
