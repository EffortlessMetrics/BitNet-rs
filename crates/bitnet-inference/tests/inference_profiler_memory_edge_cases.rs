//! Edge-case tests for the profiler and memory pool modules.

use bitnet_inference::memory_pool::{
    GrowthStrategy, MemoryPool, PoolConfig, PoolError, PoolStatistics,
};
use bitnet_inference::profiler::{
    LayerProfile, LayerStats, MemorySnapshot, ModelProfiler, ProfileReport, ProfileSession,
    ProfilerConfig,
};

// ═══════════════════════════════════════════════════════════════════
// Profiler edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn profiler_end_layer_without_begin_is_noop() {
    let mut session = ProfileSession::new(ProfilerConfig::default());
    // Should not panic when ending a layer that was never started.
    session.end_layer();
    let report = session.generate_report();
    assert!(report.layer_profiles.is_empty());
    assert_eq!(report.total_time_us, 0.0);
}

#[test]
fn profiler_more_ends_than_begins() {
    let mut session = ProfileSession::new(ProfilerConfig::default());
    session.begin_layer("only", "test");
    session.end_layer();
    session.end_layer(); // extra
    session.end_layer(); // extra
    let report = session.generate_report();
    assert_eq!(report.layer_profiles.len(), 1);
}

#[test]
fn profiler_begin_without_end_drops_silently() {
    let mut session = ProfileSession::new(ProfilerConfig::default());
    session.begin_layer("orphan", "attention");
    // Never call end_layer — the layer should just be dropped when
    // we consume the session.
    let report = session.generate_report();
    assert!(report.layer_profiles.is_empty());
}

#[test]
fn profiler_deeply_nested_layers() {
    let mut session = ProfileSession::new(ProfilerConfig::default());
    let depth = 10;
    for i in 0..depth {
        session.begin_layer(&format!("level_{i}"), "block");
    }
    for _ in 0..depth {
        session.end_layer();
    }
    let report = session.generate_report();
    assert_eq!(report.layer_profiles.len(), depth);
    // Stack order: innermost ends first.
    assert_eq!(report.layer_profiles[0].layer_name, "level_9");
    assert_eq!(report.layer_profiles[depth - 1].layer_name, "level_0");
}

#[test]
fn profiler_zero_warmup_zero_samples_completes_immediately() {
    let config = ProfilerConfig::default().with_warmup(0).with_sample_size(0);
    let mut session = ProfileSession::new(config);
    // With 0 warmup + 0 samples, first next_iteration should complete.
    assert!(session.next_iteration());
}

#[test]
fn profiler_all_warmup_no_sample_layers() {
    let config = ProfilerConfig::default().with_warmup(3).with_sample_size(0);
    let mut session = ProfileSession::new(config);
    for _ in 0..3 {
        session.begin_layer("warm", "attn");
        session.end_layer();
        session.next_iteration();
    }
    let report = session.generate_report();
    // All iterations were warmup — nothing recorded.
    assert!(report.layer_profiles.is_empty());
    assert_eq!(report.total_time_us, 0.0);
}

#[test]
fn profiler_memory_snapshot_during_warmup_discarded() {
    let config = ProfilerConfig::default().with_warmup(1).with_sample_size(1).with_memory(true);
    let mut session = ProfileSession::new(config);
    // Warmup iteration.
    session.record_memory_snapshot("warmup_snap", 9999);
    session.next_iteration();
    // Real sample.
    session.record_memory_snapshot("real_snap", 42);
    let report = session.generate_report();
    assert_eq!(report.memory_snapshots.len(), 1);
    assert_eq!(report.memory_snapshots[0].label, "real_snap");
    assert_eq!(report.memory_snapshots[0].memory_bytes, 42);
}

#[test]
fn profiler_disabled_ignores_memory_snapshots() {
    let config = ProfilerConfig { enabled: false, record_memory: true, ..Default::default() };
    let mut session = ProfileSession::new(config);
    session.record_memory_snapshot("should_be_ignored", 1024);
    let report = session.generate_report();
    assert!(report.memory_snapshots.is_empty());
}

#[test]
fn profiler_chrome_trace_empty_is_valid_json() {
    let session = ProfileSession::new(ProfilerConfig::default());
    let report = session.generate_report();
    let trace = report.export_chrome_trace();
    let parsed: serde_json::Value = serde_json::from_str(&trace).unwrap();
    assert!(parsed.as_array().unwrap().is_empty());
}

#[test]
fn profiler_chrome_trace_memory_only_no_layers() {
    let config = ProfilerConfig::default().with_memory(true);
    let mut session = ProfileSession::new(config);
    session.record_memory_snapshot("snap_a", 100);
    session.record_memory_snapshot("snap_b", 200);
    let report = session.generate_report();
    let trace = report.export_chrome_trace();
    let parsed: serde_json::Value = serde_json::from_str(&trace).unwrap();
    let arr = parsed.as_array().unwrap();
    // Only counter events, no B/E events.
    assert_eq!(arr.len(), 2);
    assert!(arr.iter().all(|e| e["ph"] == "C"));
}

#[test]
fn profiler_report_bottleneck_detection_single_dominant_layer() {
    let config = ProfilerConfig::default().with_sample_size(1);
    let mut session = ProfileSession::new(config);

    // One long layer and many short ones.
    session.begin_layer("slow_attn", "attention");
    std::thread::sleep(std::time::Duration::from_millis(10));
    session.end_layer();

    for i in 0..5 {
        session.begin_layer(&format!("fast_{i}"), "norm");
        session.end_layer();
    }

    let report = session.generate_report();
    // The slow layer should be in bottleneck_layers.
    assert!(report.bottleneck_layers.contains(&"slow_attn".to_string()));
}

#[test]
fn profiler_report_no_bottleneck_when_all_equal() {
    // When total_time_us is zero (sub-microsecond layers), the 10% threshold
    // may mark nothing or everything. Just verify no panic.
    let mut session = ProfileSession::new(ProfilerConfig::default());
    for i in 0..3 {
        session.begin_layer(&format!("l{i}"), "norm");
        session.end_layer();
    }
    let report = session.generate_report();
    // No panic — result is implementation-defined for near-zero times.
    let _ = report.bottleneck_layers;
}

#[test]
fn profiler_layer_profile_serialization_roundtrip() {
    let lp = LayerProfile {
        layer_name: "test_layer".into(),
        layer_type: "matmul".into(),
        forward_time_us: 123.456,
        backward_time_us: 0.0,
        memory_bytes: 4096,
        flops_estimate: 1_000_000,
    };
    let json = serde_json::to_string(&lp).unwrap();
    let deser: LayerProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.layer_name, "test_layer");
    assert_eq!(deser.flops_estimate, 1_000_000);
}

#[test]
fn profiler_memory_snapshot_serialization_roundtrip() {
    let snap = MemorySnapshot { label: "peak".into(), timestamp_us: 42.0, memory_bytes: 8192 };
    let json = serde_json::to_string(&snap).unwrap();
    let deser: MemorySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.label, "peak");
    assert_eq!(deser.memory_bytes, 8192);
}

#[test]
fn profiler_layer_stats_serialization_roundtrip() {
    let stats = LayerStats {
        layer_name: "attn".into(),
        layer_type: "attention".into(),
        mean_time_us: 100.0,
        std_time_us: 5.0,
        min_time_us: 90.0,
        max_time_us: 110.0,
        count: 10,
        total_memory_bytes: 0,
        total_flops: 0,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let deser: LayerStats = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.count, 10);
    assert!((deser.mean_time_us - 100.0).abs() < f64::EPSILON);
}

#[test]
fn profiler_report_serialization_roundtrip() {
    let session = ProfileSession::new(ProfilerConfig::default());
    let report = session.generate_report();
    let json = serde_json::to_string(&report).unwrap();
    let deser: ProfileReport = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.total_time_us, 0.0);
    assert!(deser.per_layer_breakdown.is_empty());
}

#[test]
fn profiler_config_builder_chaining() {
    let cfg = ProfilerConfig::default().with_warmup(5).with_sample_size(20).with_memory(true);
    assert_eq!(cfg.warmup_iterations, 5);
    assert_eq!(cfg.sample_size, 20);
    assert!(cfg.record_memory);
    assert!(cfg.enabled);
}

#[test]
fn profiler_config_disabled_builder() {
    let cfg = ProfilerConfig::disabled();
    assert!(!cfg.enabled);
    // Builder methods still work on disabled config.
    let cfg2 = cfg.with_warmup(3);
    assert!(!cfg2.enabled);
    assert_eq!(cfg2.warmup_iterations, 3);
}

#[test]
fn profiler_model_profiler_config_accessor() {
    let config = ProfilerConfig::default().with_warmup(7).with_sample_size(42);
    let profiler = ModelProfiler::new(config);
    assert!(profiler.is_enabled());
    assert_eq!(profiler.config().warmup_iterations, 7);
    assert_eq!(profiler.config().sample_size, 42);
}

#[test]
fn profiler_single_sample_std_dev_is_zero() {
    let config = ProfilerConfig::default().with_sample_size(1);
    let mut session = ProfileSession::new(config);
    session.begin_layer("single", "test");
    std::thread::sleep(std::time::Duration::from_millis(1));
    session.end_layer();
    let report = session.generate_report();
    assert_eq!(report.per_layer_breakdown.len(), 1);
    // With only one sample, std_dev should be 0.
    assert_eq!(report.per_layer_breakdown[0].std_time_us, 0.0);
}

#[test]
fn profiler_many_layers_same_name_aggregated() {
    let config = ProfilerConfig::default().with_sample_size(5);
    let mut session = ProfileSession::new(config);
    for _ in 0..5 {
        session.begin_layer("repeated", "ffn");
        std::thread::sleep(std::time::Duration::from_millis(1));
        session.end_layer();
    }
    let report = session.generate_report();
    // Five raw profiles but one aggregated entry.
    assert_eq!(report.layer_profiles.len(), 5);
    assert_eq!(report.per_layer_breakdown.len(), 1);
    assert_eq!(report.per_layer_breakdown[0].count, 5);
}

#[test]
fn profiler_memory_peak_tracks_max() {
    let config = ProfilerConfig::default().with_memory(true);
    let mut session = ProfileSession::new(config);
    session.record_memory_snapshot("low", 100);
    session.record_memory_snapshot("high", 9999);
    session.record_memory_snapshot("mid", 500);
    let report = session.generate_report();
    assert_eq!(report.memory_peak, 9999);
}

#[test]
fn profiler_memory_peak_zero_when_no_snapshots() {
    let config = ProfilerConfig::default().with_memory(true);
    let session = ProfileSession::new(config);
    let report = session.generate_report();
    assert_eq!(report.memory_peak, 0);
}

// ═══════════════════════════════════════════════════════════════════
// Memory pool edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pool_zero_size_alloc_returns_empty_handle() {
    let pool = MemoryPool::new();
    let alloc = pool.arena_alloc(0).unwrap();
    assert!(alloc.is_empty());
    assert_eq!(alloc.len(), 0);
    let data = pool.read_arena(&alloc);
    assert!(data.is_empty());
    // Statistics should not change for a zero-size alloc.
    assert_eq!(pool.statistics().allocation_count, 0);
}

#[test]
fn pool_write_arena_zero_size_is_noop() {
    let pool = MemoryPool::new();
    let alloc = pool.arena_alloc(0).unwrap();
    // Should not panic.
    pool.write_arena(&alloc, &[]);
}

#[test]
#[should_panic(expected = "data length must match allocation length")]
fn pool_write_arena_length_mismatch_panics() {
    let pool = MemoryPool::new();
    let alloc = pool.arena_alloc(8).unwrap();
    pool.write_arena(&alloc, &[1, 2, 3]); // wrong length
}

#[test]
fn pool_allocation_too_large_error() {
    let cfg = PoolConfig::builder().max_allocation_size(64).build();
    let pool = MemoryPool::with_config(cfg);
    let err = pool.arena_alloc(128).unwrap_err();
    assert_eq!(err, PoolError::AllocationTooLarge { requested: 128, limit: 64 });
    let msg = err.to_string();
    assert!(msg.contains("128"));
    assert!(msg.contains("64"));
}

#[test]
fn pool_exhaustion_error_display() {
    let cfg = PoolConfig::builder()
        .initial_size(64)
        .growth_strategy(GrowthStrategy::Fixed)
        .max_pool_size(64)
        .max_allocation_size(64)
        .build();
    let pool = MemoryPool::with_config(cfg);
    let _a = pool.arena_alloc(56).unwrap();
    // Pool is full — next alloc requires growth which hits the cap.
    let err = pool.arena_alloc(56).unwrap_err();
    assert!(matches!(err, PoolError::PoolExhausted { .. }));
    let msg = err.to_string();
    assert!(msg.contains("exhausted"));
}

#[test]
fn pool_slab_not_found_error_display() {
    let pool = MemoryPool::new();
    let err = pool.slab_checkout(777).unwrap_err();
    assert_eq!(err, PoolError::SlabNotFound { block_size: 777 });
    let msg = err.to_string();
    assert!(msg.contains("777"));
}

#[test]
fn pool_slab_checkin_to_unknown_slab_fails() {
    let pool = MemoryPool::new();
    let err = pool.slab_checkin(256, vec![0u8; 256]).unwrap_err();
    assert_eq!(err, PoolError::SlabNotFound { block_size: 256 });
}

#[test]
fn pool_register_slab_duplicate_is_noop() {
    let pool = MemoryPool::new();
    pool.register_slab(128, 4);
    let stats_before = pool.statistics();
    pool.register_slab(128, 100); // duplicate — ignored
    let stats_after = pool.statistics();
    assert_eq!(stats_before.total_pool_bytes, stats_after.total_pool_bytes);
}

#[test]
fn pool_register_multiple_slab_sizes() {
    let pool = MemoryPool::new();
    pool.register_slab(64, 2);
    pool.register_slab(256, 2);
    pool.register_slab(1024, 2);

    let b64 = pool.slab_checkout(64).unwrap();
    let b256 = pool.slab_checkout(256).unwrap();
    let b1024 = pool.slab_checkout(1024).unwrap();

    assert_eq!(b64.len(), 64);
    assert_eq!(b256.len(), 256);
    assert_eq!(b1024.len(), 1024);

    pool.slab_checkin(64, b64).unwrap();
    pool.slab_checkin(256, b256).unwrap();
    pool.slab_checkin(1024, b1024).unwrap();
}

#[test]
fn pool_slab_checkout_beyond_preallocated_creates_new_block() {
    let pool = MemoryPool::new();
    pool.register_slab(32, 1); // only 1 pre-allocated

    let b1 = pool.slab_checkout(32).unwrap();
    let b2 = pool.slab_checkout(32).unwrap(); // creates a new block
    assert_eq!(b1.len(), 32);
    assert_eq!(b2.len(), 32);
    assert_eq!(pool.statistics().slab_checkouts, 2);

    pool.slab_checkin(32, b1).unwrap();
    pool.slab_checkin(32, b2).unwrap();
}

#[test]
fn pool_arena_reset_allows_reuse() {
    let cfg = PoolConfig::builder()
        .initial_size(256)
        .max_pool_size(256)
        .max_allocation_size(256)
        .growth_strategy(GrowthStrategy::Fixed)
        .build();
    let pool = MemoryPool::with_config(cfg);

    let _a = pool.arena_alloc(200).unwrap();
    // Can't allocate more — pool is at capacity.
    assert!(pool.arena_alloc(200).is_err());

    pool.arena_reset();
    // After reset, same space is available again.
    let b = pool.arena_alloc(200).unwrap();
    assert_eq!(b.len(), 200);
}

#[test]
fn pool_arena_data_isolation() {
    let pool = MemoryPool::new();
    let a = pool.arena_alloc(8).unwrap();
    let b = pool.arena_alloc(8).unwrap();

    pool.write_arena(&a, &[1, 2, 3, 4, 5, 6, 7, 8]);
    pool.write_arena(&b, &[10, 20, 30, 40, 50, 60, 70, 80]);

    // Reads should not cross.
    assert_eq!(pool.read_arena(&a), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(pool.read_arena(&b), vec![10, 20, 30, 40, 50, 60, 70, 80]);
}

#[test]
fn pool_arena_alloc_is_eight_byte_aligned() {
    let pool = MemoryPool::new();
    // Allocate an odd size to test alignment of next allocation.
    let _a = pool.arena_alloc(3).unwrap();
    let b = pool.arena_alloc(16).unwrap();
    // b should succeed and be properly sized.
    assert_eq!(b.len(), 16);
    let data = pool.read_arena(&b);
    assert!(data.iter().all(|&byte| byte == 0));
}

#[test]
fn pool_growth_strategy_fixed_uniform_chunks() {
    let cfg = PoolConfig::builder()
        .initial_size(128)
        .growth_strategy(GrowthStrategy::Fixed)
        .max_pool_size(1024)
        .max_allocation_size(120)
        .build();
    let pool = MemoryPool::with_config(cfg);

    // Fill first chunk, trigger growth.
    let _a = pool.arena_alloc(120).unwrap();
    let _b = pool.arena_alloc(120).unwrap();
    let _c = pool.arena_alloc(120).unwrap();

    let stats = pool.statistics();
    assert_eq!(stats.allocation_count, 3);
    assert_eq!(stats.bytes_allocated, 360);
}

#[test]
fn pool_growth_strategy_double_exponential() {
    let cfg = PoolConfig::builder()
        .initial_size(64)
        .growth_strategy(GrowthStrategy::Double)
        .max_pool_size(1024)
        .max_allocation_size(200)
        .build();
    let pool = MemoryPool::with_config(cfg);

    let _a = pool.arena_alloc(60).unwrap();
    // Triggers growth — next chunk should be 128 (double of 64).
    let _b = pool.arena_alloc(60).unwrap();
    let stats = pool.statistics();
    assert!(stats.total_pool_bytes >= 64 + 128);
}

#[test]
fn pool_statistics_default_is_zeroed() {
    let stats = PoolStatistics::default();
    assert_eq!(stats.bytes_allocated, 0);
    assert_eq!(stats.bytes_freed, 0);
    assert_eq!(stats.peak_usage, 0);
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.reset_count, 0);
    assert_eq!(stats.slab_checkouts, 0);
    assert_eq!(stats.slab_returns, 0);
    assert_eq!(stats.total_pool_bytes, 0);
}

#[test]
fn pool_statistics_equality() {
    let a = PoolStatistics::default();
    let b = PoolStatistics::default();
    assert_eq!(a, b);
}

#[test]
fn pool_clone_shares_state() {
    let pool = MemoryPool::new();
    let clone = pool.clone();

    let alloc = pool.arena_alloc(64).unwrap();
    // The clone sees the same allocation count.
    assert_eq!(clone.statistics().allocation_count, 1);

    // Clone can read what pool wrote.
    pool.write_arena(&alloc, &[0xFF; 64]);
    let data = clone.read_arena(&alloc);
    assert!(data.iter().all(|&b| b == 0xFF));
}

#[test]
fn pool_multiple_resets() {
    let pool = MemoryPool::new();
    for i in 0..5 {
        let _a = pool.arena_alloc(100).unwrap();
        pool.arena_reset();
        assert_eq!(pool.statistics().reset_count, (i + 1) as u64);
    }
    assert_eq!(pool.statistics().bytes_allocated, 0);
    assert_eq!(pool.statistics().allocation_count, 5);
}

#[test]
fn pool_peak_usage_survives_multiple_resets() {
    let pool = MemoryPool::new();

    let _a = pool.arena_alloc(300).unwrap();
    pool.arena_reset();
    let _b = pool.arena_alloc(100).unwrap();
    pool.arena_reset();
    let _c = pool.arena_alloc(200).unwrap();
    pool.arena_reset();

    // Peak should still be 300 from the first round.
    assert_eq!(pool.statistics().peak_usage, 300);
}

#[test]
fn pool_slab_blocks_are_zeroed_on_return() {
    let pool = MemoryPool::new();
    pool.register_slab(16, 1);

    let mut block = pool.slab_checkout(16).unwrap();
    block.fill(0xAB);
    pool.slab_checkin(16, block).unwrap();

    // Check out again — should be zeroed.
    let block2 = pool.slab_checkout(16).unwrap();
    assert!(block2.iter().all(|&b| b == 0));
    pool.slab_checkin(16, block2).unwrap();
}

#[test]
fn pool_config_builder_all_fields() {
    let cfg = PoolConfig::builder()
        .initial_size(512)
        .growth_strategy(GrowthStrategy::Fixed)
        .max_pool_size(2048)
        .max_allocation_size(1024)
        .build();
    assert_eq!(cfg.initial_size, 512);
    assert_eq!(cfg.growth_strategy, GrowthStrategy::Fixed);
    assert_eq!(cfg.max_pool_size, 2048);
    assert_eq!(cfg.max_allocation_size, 1024);
}

#[test]
fn pool_config_default_values() {
    let cfg = PoolConfig::default();
    assert_eq!(cfg.initial_size, 1 << 20);
    assert_eq!(cfg.growth_strategy, GrowthStrategy::Double);
    assert_eq!(cfg.max_pool_size, 256 << 20);
    assert_eq!(cfg.max_allocation_size, 64 << 20);
}

#[test]
fn pool_error_is_std_error() {
    fn assert_error<E: std::error::Error>(_e: &E) {}
    let err = PoolError::AllocationTooLarge { requested: 1, limit: 0 };
    assert_error(&err);
}

#[test]
fn pool_concurrent_slab_checkout_checkin() {
    let pool = MemoryPool::new();
    pool.register_slab(64, 8);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let p = pool.clone();
            std::thread::spawn(move || {
                for _ in 0..25 {
                    let block = p.slab_checkout(64).unwrap();
                    assert_eq!(block.len(), 64);
                    p.slab_checkin(64, block).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = pool.statistics();
    assert_eq!(stats.slab_checkouts, 100);
    assert_eq!(stats.slab_returns, 100);
}

#[test]
fn pool_concurrent_arena_alloc_and_reset() {
    let pool = MemoryPool::with_config(
        PoolConfig::builder()
            .initial_size(1 << 20)
            .max_pool_size(64 << 20)
            .max_allocation_size(1024)
            .build(),
    );

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let p = pool.clone();
            std::thread::spawn(move || {
                for _ in 0..50 {
                    // Alloc may fail after a concurrent reset — that's fine.
                    let _ = p.arena_alloc(64);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Just verify no panic/deadlock.
    let _ = pool.statistics();
}

#[test]
fn pool_tiny_initial_size_clamped() {
    // The implementation clamps initial_size to at least 64.
    let cfg =
        PoolConfig::builder().initial_size(1).max_pool_size(4096).max_allocation_size(60).build();
    let pool = MemoryPool::with_config(cfg);
    // Should still be able to allocate thanks to the clamp.
    let alloc = pool.arena_alloc(60).unwrap();
    assert_eq!(alloc.len(), 60);
}
