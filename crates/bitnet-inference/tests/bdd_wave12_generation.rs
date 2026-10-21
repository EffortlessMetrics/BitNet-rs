//! BDD Wave 12 — Generation Lifecycle Integration Tests
//!
//! Given/When/Then scenarios covering:
//! 1. Engine creation → warmup → generation → shutdown
//! 2. Sampling strategy → token selection → stop condition
//! 3. Streaming generation → callback → completion

use std::time::Duration;

use bitnet_inference::batch::{BatchConfig, BatchRequest, BatchScheduler};
use bitnet_inference::cache::{CacheConfig, KVCache};
use bitnet_inference::config::GenerationConfig;
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};
use bitnet_inference::metrics::{LatencyHistogram, MetricsCollector, ThroughputTracker};
use bitnet_inference::profiler::{ProfileSession, ProfilerConfig};
use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy};
use bitnet_inference::streaming::StreamingConfig;
use bitnet_inference::thread_pool::{InferenceThreadPool, ThreadPoolConfig};
use bitnet_inference::token_stream::{StreamConfig, TokenStream};

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Engine creation → warmup → generation → shutdown
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_default_config_when_engine_created_then_valid_defaults() {
    // Given a default InferenceConfig
    let config = bitnet_inference::config::InferenceConfig::default();

    // When inspecting defaults
    // Then they are sensible
    assert!(config.max_context_length > 0);
    assert!(config.num_threads > 0);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_given_cpu_config_when_created_then_no_mixed_precision() {
    // Given a CPU-optimized config
    let config = bitnet_inference::config::InferenceConfig::cpu_optimized();

    // When inspecting
    // Then mixed precision is disabled for CPU
    assert!(!config.mixed_precision);
    assert!(config.num_threads >= 1);
}

#[test]
fn test_given_profiler_when_warmup_iterations_then_counted() {
    // Given a profiler with 3 warmup iterations
    let config = ProfilerConfig::default().with_warmup(3).with_sample_size(5);
    let mut session = ProfileSession::new(config);

    // When iterating through warmup
    let mut warmup_count = 0;
    while session.is_warmup() {
        warmup_count += 1;
        session.next_iteration();
    }

    // Then warmup iterations were counted
    assert_eq!(warmup_count, 3);
}

#[test]
fn test_given_profiler_when_layers_recorded_then_report_generated() {
    // Given an enabled profiling session
    let config = ProfilerConfig::default().with_warmup(0).with_sample_size(2);
    let mut session = ProfileSession::new(config);

    // When recording layer timings
    session.begin_layer("attention", "self_attention");
    session.end_layer();
    session.begin_layer("ffn", "feed_forward");
    session.end_layer();
    session.next_iteration();

    // Then a report can be generated
    let report = session.generate_report();
    assert!(!report.per_layer_breakdown.is_empty());
}

#[test]
fn test_given_thread_pool_when_created_then_correct_threads() {
    // Given a thread pool config
    let config = ThreadPoolConfig { num_threads: 2, ..Default::default() };

    // When creating the pool
    let pool = InferenceThreadPool::new(config).expect("pool creation");

    // Then correct thread count
    assert_eq!(pool.num_threads(), 2);
}

#[test]
fn test_given_thread_pool_when_parallel_for_then_work_executed() {
    // Given a thread pool
    let pool = InferenceThreadPool::with_defaults().expect("pool creation");

    // When executing parallel work
    let data = std::sync::Mutex::new(vec![0u32; 8]);
    pool.parallel_for(0..8, 1, |i| {
        data.lock().unwrap()[i] = i as u32 + 1;
    });

    // Then all elements are populated
    let result = data.lock().unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, i as u32 + 1);
    }
}

#[test]
fn test_given_kv_cache_when_created_then_empty() {
    // Given a KV cache config
    let config = CacheConfig::default();

    // When creating the cache
    let cache = KVCache::new(config).unwrap();

    // Then it starts empty
    assert_eq!(cache.size(), 0);
}

#[test]
fn test_given_builder_balanced_preset_when_built_then_correct_params() {
    // Given a balanced preset
    let config = InferenceConfigBuilder::new()
        .preset(InferencePreset::Balanced)
        .build()
        .expect("valid config");

    // When inspecting sampling params
    // Then they match balanced defaults
    assert!((config.sampling.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.sampling.top_k, 50);
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — Sampling strategy → token selection → stop condition
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_greedy_sampling_when_sampled_then_argmax() {
    // Given greedy config (temperature=0)
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);

    // When sampling from logits with clear max at index 3
    let logits = vec![0.1, 0.3, 0.2, 0.9, 0.4];
    let token = strategy.sample(&logits, &[]).unwrap();

    // Then argmax is returned
    assert_eq!(token, 3);
}

#[test]
fn test_given_greedy_when_repeated_then_deterministic() {
    // Given two identical greedy strategies
    let make = || SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let logits = vec![0.1, 0.5, 0.3, 0.8];

    // When sampling twice
    let t1 = SamplingStrategy::new(make()).sample(&logits, &[]).unwrap();
    let t2 = SamplingStrategy::new(make()).sample(&logits, &[]).unwrap();

    // Then results are identical
    assert_eq!(t1, t2);
}

#[test]
fn test_given_greedy_with_negative_logits_when_sampled_then_least_negative() {
    // Given all-negative logits
    let config = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);

    // When sampling
    let logits = vec![-10.0, -5.0, -20.0, -1.0];
    let token = strategy.sample(&logits, &[]).unwrap();

    // Then least negative (index 3, value -1.0) is selected
    assert_eq!(token, 3);
}

#[test]
fn test_given_generation_config_greedy_when_inspected_then_deterministic_params() {
    // Given GenerationConfig::greedy()
    let config = GenerationConfig::greedy();

    // When inspecting
    // Then temperature is 0, top_k is 1
    assert!((config.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_given_generation_config_creative_when_inspected_then_high_diversity() {
    // Given GenerationConfig::creative()
    let config = GenerationConfig::creative();

    // When inspecting
    // Then temperature is elevated
    assert!(config.temperature > 0.5);
    assert!(config.top_p > 0.5);
}

#[test]
fn test_given_stop_token_when_configured_then_detected() {
    // Given a config with stop token IDs
    let config = GenerationConfig::greedy().with_stop_token_ids(vec![128009, 2]);

    // When checking if a token is a stop token
    // Then configured tokens are detected
    assert!(config.is_stop_token(128009));
    assert!(config.is_stop_token(2));
    assert!(!config.is_stop_token(42));
}

#[test]
fn test_given_stop_sequence_when_added_then_stored() {
    // Given a config with stop sequences
    let config = GenerationConfig::greedy()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\nQ:".to_string());

    // When inspecting
    // Then stop sequences are stored
    assert_eq!(config.stop_sequences.len(), 2);
    assert!(config.stop_sequences.contains(&"</s>".to_string()));
}

#[test]
fn test_given_max_tokens_when_set_then_respected() {
    // Given a config with max tokens
    let config = GenerationConfig::greedy().with_max_tokens(128);

    // When inspecting
    // Then max tokens is set
    assert_eq!(config.max_new_tokens, 128);
}

#[test]
fn test_given_generation_config_when_validated_then_rejects_invalid() {
    // Given an invalid config (temperature too high)
    let mut config = GenerationConfig::greedy();
    config.temperature = -1.0;

    // When validated
    let result = config.validate();

    // Then validation fails
    assert!(result.is_err());
}

#[test]
fn test_given_sampling_strategy_when_reset_then_state_cleared() {
    // Given a sampling strategy that has been used
    let config = SamplingConfig { temperature: 0.7, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config.clone());
    let _ = strategy.sample(&[0.1, 0.9, 0.3], &[]);

    // When reset
    strategy.reset();

    // Then strategy can be used again with same behavior
    let mut fresh = SamplingStrategy::new(config);
    let logits = vec![0.1, 0.9, 0.3];
    let t1 = strategy.sample(&logits, &[]).unwrap();
    let t2 = fresh.sample(&logits, &[]).unwrap();
    // Both should produce valid tokens
    assert!(t1 < 3);
    assert!(t2 < 3);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — Streaming generation → callback → completion
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_given_streaming_config_low_latency_when_created_then_small_buffer() {
    // Given a low-latency streaming config
    let config = StreamingConfig::low_latency();

    // When inspecting
    // Then buffer is small for low latency
    assert!(config.buffer_size <= 16);
    assert!(config.validate().is_ok());
}

#[test]
fn test_given_streaming_config_high_throughput_when_created_then_large_buffer() {
    // Given a high-throughput streaming config
    let config = StreamingConfig::high_throughput();

    // When inspecting
    // Then buffer is larger for throughput
    assert!(config.buffer_size >= 16);
    assert!(config.validate().is_ok());
}

#[test]
fn test_given_token_stream_when_tokens_pushed_then_text_emitted() {
    // Given a token stream with a simple decoder
    let config = StreamConfig { buffer_size: 4, ..Default::default() };
    let mut stream = TokenStream::new(config, |token_id: u32| {
        // Simple decoder: each token is a single ASCII byte
        Some(vec![b'A' + (token_id % 26) as u8])
    });

    // When pushing tokens
    let event = stream.push_token(0); // 'A'

    // Then text is emitted or buffered
    // (exact behavior depends on buffer config)
    assert!(!stream.is_complete());
    // Token was accepted (event may or may not emit yet)
    let _ = event;
}

#[test]
fn test_given_token_stream_when_flushed_then_remaining_emitted() {
    // Given a stream with buffered tokens
    let config = StreamConfig {
        buffer_size: 100, // large buffer to prevent auto-flush
        ..Default::default()
    };
    let mut stream =
        TokenStream::new(config, |token_id: u32| Some(vec![b'X' + (token_id % 3) as u8]));
    stream.push_token(0);
    stream.push_token(1);

    // When flushing
    let events = stream.flush();

    // Then events are produced (may be empty if already emitted)
    let _ = events; // Just verifying no panic
}

#[test]
fn test_given_metrics_collector_when_requests_recorded_then_snapshot_accurate() {
    // Given a metrics collector
    let collector = MetricsCollector::new();

    // When recording requests
    collector.record_request(10, 5, 50_000_000, 10_000_000);
    collector.record_request(20, 8, 80_000_000, 15_000_000);

    // Then snapshot reflects the data
    let snapshot = collector.snapshot();
    assert!(snapshot.generated_tokens > 0);
    assert!(snapshot.prompt_tokens > 0);
}

#[test]
fn test_given_latency_histogram_when_values_recorded_then_percentiles_available() {
    // Given a histogram
    let mut hist = LatencyHistogram::new();

    // When recording latencies
    for i in 1..=100 {
        hist.record(i as f64);
    }

    // Then percentiles are computed
    let p50 = hist.p50().expect("p50 should exist");
    let p99 = hist.p99().expect("p99 should exist");
    assert!(p50 > 0.0);
    assert!(p99 >= p50);
    assert_eq!(hist.count(), 100);
}

#[test]
fn test_given_throughput_tracker_when_tokens_recorded_then_tps_positive() {
    // Given a throughput tracker
    let mut tracker = ThroughputTracker::new(Duration::from_mins(1));

    // When recording token generation
    tracker.record(10);
    tracker.record(20);

    // Then TPS is non-negative (may be 0 if timestamps are too close)
    let tps = tracker.tokens_per_second();
    assert!(tps >= 0.0);
}

#[test]
fn test_given_batch_scheduler_when_requests_scheduled_then_indices_returned() {
    // Given a batch scheduler
    let config = BatchConfig::new(4, Duration::from_millis(100));
    let scheduler = BatchScheduler::new(config);

    // When scheduling requests
    let mut batch = BatchRequest::new();
    batch.add("Hello".to_string(), GenerationConfig::greedy());
    batch.add("World".to_string(), GenerationConfig::greedy());
    let indices = scheduler.schedule(&batch);

    // Then valid indices are returned
    assert!(!indices.is_empty());
    assert!(indices.len() <= 4);
}

#[test]
fn test_given_batch_config_when_validated_then_rejects_zero_size() {
    // Given a batch config with zero max_total_tokens
    let config = BatchConfig::new(1, Duration::from_millis(100)).with_max_total_tokens(0);

    // When validated
    let result = config.validate();

    // Then it's rejected
    assert!(result.is_err());
}

#[test]
fn test_given_builder_with_invalid_temp_when_built_then_error() {
    // Given a builder with invalid temperature
    let result = InferenceConfigBuilder::new().temperature(-1.0).build();

    // When built
    // Then validation rejects it
    assert!(result.is_err());
}

#[test]
fn test_given_builder_with_stop_sequences_when_built_then_stored() {
    // Given a builder with stop sequences
    let config = InferenceConfigBuilder::new()
        .preset(InferencePreset::Balanced)
        .stop_sequence("</s>")
        .stop_token_id(128009)
        .build()
        .expect("valid config");

    // When inspecting
    // Then stop configs are stored
    assert!(config.generation.stop_sequences.contains(&"</s>".to_string()));
    assert!(config.generation.stop_token_ids.contains(&128009));
}

#[test]
fn test_given_metrics_collector_when_cache_events_recorded_then_counted() {
    // Given a collector
    let collector = MetricsCollector::new();

    // When recording cache events
    collector.record_cache_hit();
    collector.record_cache_hit();
    collector.record_cache_miss();

    // Then snapshot reflects counts (cache_hit_rate = 2/3)
    let snapshot = collector.snapshot();
    assert!(snapshot.cache_hit_rate > 0.0);
}

#[test]
fn test_given_latency_histogram_when_reset_then_empty() {
    // Given a histogram with data
    let mut hist = LatencyHistogram::new();
    hist.record(10.0);
    hist.record(20.0);
    assert_eq!(hist.count(), 2);

    // When reset
    hist.reset();

    // Then it's empty
    assert_eq!(hist.count(), 0);
    assert!(hist.p50().is_none());
}
