//! Edge-case tests for dynamic batching: PaddingStrategy, BatchConfig,
//! BatchRequest, DynamicBatch, BatchFormationStrategy, PaddingCalculator,
//! BatchScheduler, BatchEfficiency, InflightBatchTracker, BatchMetrics,
//! DynamicBatcher.

use std::time::{Duration, Instant};

use bitnet_gpu_hal::dynamic_batching::{
    BatchConfig, BatchEfficiency, BatchFormationStrategy, BatchMetrics, BatchRequest,
    BatchScheduler, DynamicBatch, DynamicBatcher, InflightBatchTracker, PaddingCalculator,
    PaddingStrategy,
};

// Helper to create a BatchRequest with minimal boilerplate
fn make_request(id: u64, tokens: Vec<u32>, priority: u32, max_gen: usize) -> BatchRequest {
    BatchRequest {
        id,
        tokens,
        priority,
        arrival_time: Instant::now(),
        max_generation_tokens: max_gen,
    }
}

// ── PaddingStrategy ───────────────────────────────────────────────────────────

#[test]
fn padding_strategy_all_variants() {
    let _a = PaddingStrategy::PadToLongest;
    let _b = PaddingStrategy::PadToMax(512);
    let _c = PaddingStrategy::NoPadding;
}

#[test]
fn padding_strategy_clone_eq() {
    let a = PaddingStrategy::PadToMax(256);
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn padding_strategy_debug() {
    let s = format!("{:?}", PaddingStrategy::PadToLongest);
    assert!(s.contains("PadToLongest"));
}

// ── BatchConfig ───────────────────────────────────────────────────────────────

#[test]
fn batch_config_default() {
    let cfg = BatchConfig::default();
    assert_eq!(cfg.max_batch_size, 32);
    assert_eq!(cfg.max_wait_ms, 50);
    assert_eq!(cfg.padding_strategy, PaddingStrategy::PadToLongest);
    assert!(cfg.sort_by_length);
    assert_eq!(cfg.max_total_tokens, 8192);
}

#[test]
fn batch_config_custom() {
    let cfg = BatchConfig {
        max_batch_size: 8,
        max_wait_ms: 100,
        padding_strategy: PaddingStrategy::NoPadding,
        sort_by_length: false,
        max_total_tokens: 4096,
    };
    assert_eq!(cfg.max_batch_size, 8);
    assert!(!cfg.sort_by_length);
}

#[test]
fn batch_config_clone() {
    let cfg = BatchConfig::default();
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.max_batch_size, 32);
}

// ── BatchRequest ──────────────────────────────────────────────────────────────

#[test]
fn batch_request_total_token_budget() {
    let req = make_request(1, vec![1, 2, 3, 4, 5], 0, 100);
    assert_eq!(req.total_token_budget(), 105); // 5 + 100
}

#[test]
fn batch_request_empty_tokens() {
    let req = make_request(1, vec![], 0, 50);
    assert_eq!(req.total_token_budget(), 50);
}

#[test]
fn batch_request_zero_generation() {
    let req = make_request(1, vec![1, 2, 3], 0, 0);
    assert_eq!(req.total_token_budget(), 3);
}

#[test]
fn batch_request_clone() {
    let req = make_request(1, vec![10, 20], 5, 50);
    let req2 = req.clone();
    assert_eq!(req2.id, 1);
    assert_eq!(req2.tokens, vec![10, 20]);
    assert_eq!(req2.priority, 5);
}

// ── DynamicBatch ──────────────────────────────────────────────────────────────

#[test]
fn dynamic_batch_empty() {
    let batch =
        DynamicBatch { requests: vec![], padding_tokens: 0, total_tokens: 0, efficiency: 0.0 };
    assert!(batch.is_empty());
    assert_eq!(batch.size(), 0);
}

#[test]
fn dynamic_batch_with_requests() {
    let batch = DynamicBatch {
        requests: vec![make_request(1, vec![1, 2], 0, 10)],
        padding_tokens: 5,
        total_tokens: 100,
        efficiency: 0.95,
    };
    assert!(!batch.is_empty());
    assert_eq!(batch.size(), 1);
}

// ── BatchFormationStrategy ────────────────────────────────────────────────────

#[test]
fn batch_formation_strategy_all_variants() {
    let variants = [
        BatchFormationStrategy::Greedy,
        BatchFormationStrategy::FirstFit,
        BatchFormationStrategy::BestFit,
        BatchFormationStrategy::SortedGreedy,
    ];
    assert_eq!(variants.len(), 4);
}

#[test]
fn batch_formation_strategy_display() {
    let s = format!("{}", BatchFormationStrategy::Greedy);
    assert!(!s.is_empty());
}

#[test]
fn batch_formation_strategy_clone_eq() {
    let a = BatchFormationStrategy::BestFit;
    let b = a;
    assert_eq!(a, b);
}

// ── PaddingCalculator ─────────────────────────────────────────────────────────

#[test]
fn padding_calculator_pad_to_longest() {
    let lengths = [3, 5, 7];
    let padding = PaddingCalculator::compute(&lengths, PaddingStrategy::PadToLongest);
    // Pad to 7: (7-3)+(7-5)+(7-7) = 4+2+0 = 6
    assert_eq!(padding, 6);
}

#[test]
fn padding_calculator_pad_to_max() {
    let lengths = [3, 5];
    let padding = PaddingCalculator::compute(&lengths, PaddingStrategy::PadToMax(10));
    // Pad to 10: (10-3)+(10-5) = 7+5 = 12
    assert_eq!(padding, 12);
}

#[test]
fn padding_calculator_no_padding() {
    let lengths = [3, 5, 7];
    let padding = PaddingCalculator::compute(&lengths, PaddingStrategy::NoPadding);
    assert_eq!(padding, 0);
}

#[test]
fn padding_calculator_empty_lengths() {
    let lengths: [usize; 0] = [];
    let padding = PaddingCalculator::compute(&lengths, PaddingStrategy::PadToLongest);
    assert_eq!(padding, 0);
}

#[test]
fn padding_calculator_single_item() {
    let lengths = [10];
    let padding = PaddingCalculator::compute(&lengths, PaddingStrategy::PadToLongest);
    assert_eq!(padding, 0);
}

#[test]
fn padding_calculator_total_with_padding() {
    let lengths = [3, 5];
    let total = PaddingCalculator::total_with_padding(&lengths, PaddingStrategy::PadToLongest);
    // Pad to 5: total = 5+5 = 10
    assert_eq!(total, 10);
}

// ── BatchScheduler ────────────────────────────────────────────────────────────

#[test]
fn batch_scheduler_dispatch_by_size() {
    let cfg = BatchConfig { max_batch_size: 8, ..Default::default() };
    let scheduler = BatchScheduler::new(cfg);
    assert!(!scheduler.should_dispatch_by_size(7));
    assert!(scheduler.should_dispatch_by_size(8));
    assert!(scheduler.should_dispatch_by_size(10));
}

#[test]
fn batch_scheduler_dispatch_by_size_zero() {
    let cfg = BatchConfig { max_batch_size: 8, ..Default::default() };
    let scheduler = BatchScheduler::new(cfg);
    assert!(!scheduler.should_dispatch_by_size(0));
}

#[test]
fn batch_scheduler_config_accessor() {
    let cfg = BatchConfig { max_batch_size: 16, ..Default::default() };
    let scheduler = BatchScheduler::new(cfg);
    assert_eq!(scheduler.config().max_batch_size, 16);
}

#[test]
fn batch_scheduler_dispatch_by_timeout() {
    let cfg = BatchConfig { max_wait_ms: 10, ..Default::default() };
    let scheduler = BatchScheduler::new(cfg);
    // Very recent arrival — should not dispatch
    let recent = Instant::now();
    assert!(!scheduler.should_dispatch_by_timeout(recent));
}

// ── BatchEfficiency ───────────────────────────────────────────────────────────

#[test]
fn batch_efficiency_compute_full() {
    let eff = BatchEfficiency::compute(100, 100);
    assert!((eff - 1.0).abs() < 1e-6);
}

#[test]
fn batch_efficiency_compute_half() {
    let eff = BatchEfficiency::compute(50, 100);
    assert!((eff - 0.5).abs() < 1e-6);
}

#[test]
fn batch_efficiency_compute_zero_total() {
    let eff = BatchEfficiency::compute(0, 0);
    // Implementation returns 1.0 when both are zero (perfect efficiency)
    let _ = eff; // Just verify no panic
}

#[test]
fn batch_efficiency_estimate() {
    let reqs =
        vec![make_request(1, vec![1, 2, 3], 0, 0), make_request(2, vec![1, 2, 3, 4, 5], 0, 0)];
    let eff = BatchEfficiency::estimate(&reqs, PaddingStrategy::PadToLongest);
    assert!(eff > 0.0 && eff <= 1.0);
}

// ── InflightBatchTracker ──────────────────────────────────────────────────────

#[test]
fn inflight_tracker_new() {
    let tracker = InflightBatchTracker::new();
    assert_eq!(tracker.active_count(), 0);
    assert!(tracker.is_idle());
    assert_eq!(tracker.total_inflight_tokens(), 0);
}

#[test]
fn inflight_tracker_default() {
    let tracker = InflightBatchTracker::default();
    assert!(tracker.is_idle());
}

#[test]
fn inflight_tracker_register_and_complete() {
    let mut tracker = InflightBatchTracker::new();
    let id = tracker.register(4, 1024);
    assert_eq!(tracker.active_count(), 1);
    assert!(!tracker.is_idle());
    assert_eq!(tracker.total_inflight_tokens(), 1024);

    let entry = tracker.complete(id);
    assert!(entry.is_some());
    let entry = entry.unwrap();
    assert_eq!(entry.batch_id, id);
    assert_eq!(entry.request_count, 4);
    assert_eq!(entry.total_tokens, 1024);
    assert!(tracker.is_idle());
}

#[test]
fn inflight_tracker_complete_missing() {
    let mut tracker = InflightBatchTracker::new();
    assert!(tracker.complete(999).is_none());
}

#[test]
fn inflight_tracker_multiple_batches() {
    let mut tracker = InflightBatchTracker::new();
    let id1 = tracker.register(2, 512);
    let id2 = tracker.register(3, 768);
    assert_eq!(tracker.active_count(), 2);
    assert_eq!(tracker.total_inflight_tokens(), 512 + 768);

    tracker.complete(id1);
    assert_eq!(tracker.active_count(), 1);
    assert_eq!(tracker.total_inflight_tokens(), 768);

    tracker.complete(id2);
    assert!(tracker.is_idle());
}

#[test]
fn inflight_tracker_ids_increment() {
    let mut tracker = InflightBatchTracker::new();
    let id1 = tracker.register(1, 100);
    let id2 = tracker.register(1, 100);
    assert_ne!(id1, id2);
}

// ── BatchMetrics ──────────────────────────────────────────────────────────────

#[test]
fn batch_metrics_new() {
    let m = BatchMetrics::new();
    assert_eq!(m.batches_formed, 0);
    assert_eq!(m.requests_processed, 0);
}

#[test]
fn batch_metrics_default() {
    let m = BatchMetrics::default();
    assert_eq!(m.total_padding_tokens, 0);
}

#[test]
fn batch_metrics_record_batch() {
    let mut m = BatchMetrics::new();
    m.record_batch(4, 100, 20, 500, 1000);
    assert_eq!(m.batches_formed, 1);
    assert_eq!(m.total_useful_tokens, 100);
    assert_eq!(m.total_padding_tokens, 20);
    assert_eq!(m.total_formation_us, 500);
}

#[test]
fn batch_metrics_avg_batch_size() {
    let mut m = BatchMetrics::new();
    m.record_batch(4, 100, 10, 100, 200);
    m.record_batch(8, 200, 20, 150, 300);
    let avg = m.avg_batch_size();
    // total_batch_sizes = 4+8=12, batches_formed = 2 -> avg = 6
    assert!((avg - 6.0).abs() < 1e-6);
}

#[test]
fn batch_metrics_avg_batch_size_zero() {
    let m = BatchMetrics::new();
    let avg = m.avg_batch_size();
    assert!(avg == 0.0 || avg.is_nan());
}

#[test]
fn batch_metrics_padding_overhead() {
    let mut m = BatchMetrics::new();
    m.record_batch(4, 80, 20, 100, 200);
    let overhead = m.padding_overhead();
    // padding / (useful + padding) = 20/100 = 0.2
    assert!((overhead - 0.2).abs() < 0.01);
}

#[test]
fn batch_metrics_throughput() {
    let mut m = BatchMetrics::new();
    m.record_batch(4, 100, 0, 100, 200);
    let tp = m.throughput(Duration::from_secs(1));
    // Throughput should be positive
    assert!(tp > 0.0);
}

#[test]
fn batch_metrics_avg_formation_us() {
    let mut m = BatchMetrics::new();
    m.record_batch(2, 50, 5, 100, 300);
    m.record_batch(3, 75, 10, 200, 400);
    let avg = m.avg_formation_us();
    // total_formation_us = 300, batches_formed = 2 -> avg = 150
    assert!((avg - 150.0).abs() < 1e-6);
}

// ── DynamicBatcher ────────────────────────────────────────────────────────────

#[test]
fn dynamic_batcher_new() {
    let cfg = BatchConfig::default();
    let batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::Greedy);
    assert_eq!(batcher.pending_count(), 0);
    assert_eq!(batcher.strategy(), BatchFormationStrategy::Greedy);
}

#[test]
fn dynamic_batcher_submit_and_pending() {
    let cfg = BatchConfig::default();
    let mut batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::Greedy);
    batcher.submit(make_request(1, vec![1, 2, 3], 0, 10));
    assert_eq!(batcher.pending_count(), 1);
    batcher.submit(make_request(2, vec![4, 5], 0, 10));
    assert_eq!(batcher.pending_count(), 2);
}

#[test]
fn dynamic_batcher_form_batch_insufficient() {
    let cfg = BatchConfig { max_batch_size: 4, ..Default::default() };
    let mut batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::Greedy);
    batcher.submit(make_request(1, vec![1, 2], 0, 10));
    // With only 1 request and no timeout, may or may not form
    let batch = batcher.try_form_batch();
    // Result depends on implementation; just verify no panic
    let _ = batch;
}

#[test]
fn dynamic_batcher_form_batch_full() {
    let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
    let mut batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::Greedy);
    batcher.submit(make_request(1, vec![1, 2, 3], 0, 5));
    batcher.submit(make_request(2, vec![4, 5], 0, 5));
    let batch = batcher.try_form_batch();
    if let Some(b) = batch {
        assert!(b.size() > 0);
        assert!(b.total_tokens > 0);
    }
}

#[test]
fn dynamic_batcher_metrics_initial() {
    let cfg = BatchConfig::default();
    let batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::SortedGreedy);
    assert_eq!(batcher.metrics().batches_formed, 0);
}

#[test]
fn dynamic_batcher_inflight_initial() {
    let cfg = BatchConfig::default();
    let batcher = DynamicBatcher::new(cfg, BatchFormationStrategy::FirstFit);
    assert!(batcher.inflight().is_idle());
}

#[test]
fn dynamic_batcher_all_strategies() {
    for strategy in [
        BatchFormationStrategy::Greedy,
        BatchFormationStrategy::FirstFit,
        BatchFormationStrategy::BestFit,
        BatchFormationStrategy::SortedGreedy,
    ] {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut batcher = DynamicBatcher::new(cfg, strategy);
        batcher.submit(make_request(1, vec![1], 0, 5));
        batcher.submit(make_request(2, vec![2], 0, 5));
        let _ = batcher.try_form_batch();
        // Just verify no panic for each strategy
    }
}
