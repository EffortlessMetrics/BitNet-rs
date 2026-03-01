//! Integration tests for the GPU memory profiler and allocation optimizer.

use bitnet_opencl::allocation_optimizer::{AllocationOptimizer, AllocationPattern};
use bitnet_opencl::memory_profiler::{AllocationTag, MemoryBudget, MemoryProfiler};

// ── Basic tracking ───────────────────────────────────────────────────────────

#[test]
fn empty_profiler_zero_stats() {
    let p = MemoryProfiler::new(false);
    let snap = p.snapshot();
    assert_eq!(snap.total_allocated, 0);
    assert_eq!(snap.peak, 0);
    assert_eq!(snap.live_count, 0);
    assert!(snap.fragmentation_ratio.abs() < f64::EPSILON);
    assert!(snap.by_category.is_empty());
}

#[test]
fn track_single_alloc() {
    let p = MemoryProfiler::new(false);
    let id = p.track_alloc(4096, AllocationTag::Weights).unwrap();
    assert!(id > 0);
    assert_eq!(p.total_allocated(), 4096);
    assert_eq!(p.peak_allocated(), 4096);
    let snap = p.snapshot();
    assert_eq!(snap.live_count, 1);
    assert_eq!(snap.by_category.get(&AllocationTag::Weights).copied(), Some(4096));
}

#[test]
fn track_alloc_free_sequence() {
    let p = MemoryProfiler::new(false);
    let a = p.track_alloc(100, AllocationTag::Activations).unwrap();
    let b = p.track_alloc(200, AllocationTag::KvCache).unwrap();
    let c = p.track_alloc(300, AllocationTag::Scratch).unwrap();
    assert_eq!(p.total_allocated(), 600);

    assert!(p.track_free(b));
    assert_eq!(p.total_allocated(), 400);

    assert!(p.track_free(a));
    assert_eq!(p.total_allocated(), 300);

    assert!(p.track_free(c));
    assert_eq!(p.total_allocated(), 0);
}

#[test]
fn peak_memory_survives_frees() {
    let p = MemoryProfiler::new(false);
    let a = p.track_alloc(1000, AllocationTag::Weights).unwrap();
    let b = p.track_alloc(2000, AllocationTag::Weights).unwrap();
    p.track_free(a);
    p.track_free(b);
    // Peak should still reflect the high-water mark.
    assert_eq!(p.peak_allocated(), 3000);
    assert_eq!(p.total_allocated(), 0);
}

#[test]
fn free_unknown_id_is_noop() {
    let p = MemoryProfiler::new(false);
    assert!(!p.track_free(42));
}

#[test]
fn double_free_returns_false() {
    let p = MemoryProfiler::new(false);
    let id = p.track_alloc(64, AllocationTag::Scratch).unwrap();
    assert!(p.track_free(id));
    assert!(!p.track_free(id));
}

// ── Fragmentation ────────────────────────────────────────────────────────────

#[test]
fn fragmentation_ratio_zero_when_nothing_freed() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(512, AllocationTag::Weights).unwrap();
    let snap = p.snapshot();
    assert!(snap.fragmentation_ratio.abs() < f64::EPSILON);
}

#[test]
fn fragmentation_ratio_increases_after_free() {
    let p = MemoryProfiler::new(false);
    let a = p.track_alloc(500, AllocationTag::Activations).unwrap();
    p.track_alloc(500, AllocationTag::Activations).unwrap();
    p.track_free(a);
    let snap = p.snapshot();
    // One of two allocations freed → ~0.5.
    assert!(snap.fragmentation_ratio > 0.0);
    assert!(snap.fragmentation_ratio <= 1.0);
}

// ── Budget enforcement ───────────────────────────────────────────────────────

#[test]
fn budget_allows_within_limit() {
    let p = MemoryProfiler::new(false);
    let mut budget = MemoryBudget::new();
    budget.set_limit(AllocationTag::Weights, 2048);
    p.set_budget(budget);

    assert!(p.track_alloc(1024, AllocationTag::Weights).is_ok());
    assert!(p.track_alloc(1024, AllocationTag::Weights).is_ok());
}

#[test]
fn budget_rejects_over_limit() {
    let p = MemoryProfiler::new(false);
    let mut budget = MemoryBudget::new();
    budget.set_limit(AllocationTag::Weights, 1000);
    p.set_budget(budget);

    p.track_alloc(800, AllocationTag::Weights).unwrap();
    let res = p.track_alloc(300, AllocationTag::Weights);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("budget exceeded"));
}

#[test]
fn budget_only_applies_to_tagged_category() {
    let p = MemoryProfiler::new(false);
    let mut budget = MemoryBudget::new();
    budget.set_limit(AllocationTag::Weights, 100);
    p.set_budget(budget);

    // Scratch has no limit.
    assert!(p.track_alloc(99999, AllocationTag::Scratch).is_ok());
    // Weights over limit.
    assert!(p.track_alloc(200, AllocationTag::Weights).is_err());
}

#[test]
fn check_budget_without_allocating() {
    let p = MemoryProfiler::new(false);
    let mut budget = MemoryBudget::new();
    budget.set_limit(AllocationTag::KvCache, 500);
    p.set_budget(budget);

    assert!(p.check_budget(AllocationTag::KvCache, 500).is_ok());
    assert!(p.check_budget(AllocationTag::KvCache, 501).is_err());
}

// ── Timeline ─────────────────────────────────────────────────────────────────

#[test]
fn timeline_records_events() {
    let p = MemoryProfiler::new(false);
    let id = p.track_alloc(128, AllocationTag::Scratch).unwrap();
    p.track_free(id);
    let tl = p.timeline();
    assert_eq!(tl.len(), 2);
    assert!(tl.events[0].is_alloc);
    assert!(!tl.events[1].is_alloc);
}

#[test]
fn timeline_running_total_is_correct() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(100, AllocationTag::Weights).unwrap();
    p.track_alloc(200, AllocationTag::Weights).unwrap();
    let tl = p.timeline();
    assert_eq!(tl.events[0].running_total, 100);
    assert_eq!(tl.events[1].running_total, 300);
}

#[test]
fn empty_timeline() {
    let p = MemoryProfiler::new(false);
    let tl = p.timeline();
    assert!(tl.is_empty());
    assert_eq!(tl.len(), 0);
}

// ── Concurrency ──────────────────────────────────────────────────────────────

#[test]
fn concurrent_allocations() {
    use std::sync::Arc;
    use std::thread;

    let p = Arc::new(MemoryProfiler::new(false));
    let mut handles = Vec::new();

    for _ in 0..10 {
        let profiler = Arc::clone(&p);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let _ = profiler.track_alloc(64, AllocationTag::Scratch);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // 10 threads × 100 allocs × 64 bytes = 64 000.
    assert_eq!(p.total_allocated(), 64_000);
    assert_eq!(p.peak_allocated(), 64_000);
    let snap = p.snapshot();
    assert_eq!(snap.live_count, 1000);
}

#[test]
fn concurrent_alloc_and_free() {
    use std::sync::Arc;
    use std::thread;

    let p = Arc::new(MemoryProfiler::new(false));

    // Allocate 100 entries.
    let mut ids = Vec::new();
    for _ in 0..100 {
        ids.push(p.track_alloc(32, AllocationTag::Activations).unwrap());
    }

    // Free them from many threads.
    let ids = Arc::new(ids);
    let mut handles = Vec::new();
    for i in 0..10 {
        let profiler = Arc::clone(&p);
        let ids = Arc::clone(&ids);
        handles.push(thread::spawn(move || {
            for j in 0..10 {
                profiler.track_free(ids[i * 10 + j]);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(p.total_allocated(), 0);
}

// ── Report formatting ────────────────────────────────────────────────────────

#[test]
fn report_contains_key_sections() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(256, AllocationTag::Weights).unwrap();
    p.track_alloc(128, AllocationTag::Activations).unwrap();
    let report = p.report();
    assert!(report.contains("GPU Memory Profile"));
    assert!(report.contains("Total allocated:"));
    assert!(report.contains("Peak:"));
    assert!(report.contains("Fragmentation:"));
    assert!(report.contains("Live allocations:"));
    assert!(report.contains("Timeline events:"));
}

#[test]
fn report_on_empty_profiler() {
    let p = MemoryProfiler::new(false);
    let report = p.report();
    assert!(report.contains("Total allocated: 0 bytes"));
    assert!(report.contains("Peak: 0 bytes"));
}

// ── Category-based tracking ──────────────────────────────────────────────────

#[test]
fn category_breakdown_in_snapshot() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(100, AllocationTag::Weights).unwrap();
    p.track_alloc(200, AllocationTag::Activations).unwrap();
    p.track_alloc(300, AllocationTag::KvCache).unwrap();
    p.track_alloc(400, AllocationTag::Scratch).unwrap();

    let snap = p.snapshot();
    assert_eq!(snap.by_category.get(&AllocationTag::Weights).copied(), Some(100));
    assert_eq!(snap.by_category.get(&AllocationTag::Activations).copied(), Some(200));
    assert_eq!(snap.by_category.get(&AllocationTag::KvCache).copied(), Some(300));
    assert_eq!(snap.by_category.get(&AllocationTag::Scratch).copied(), Some(400));
}

#[test]
fn category_decreases_on_free() {
    let p = MemoryProfiler::new(false);
    let a = p.track_alloc(500, AllocationTag::KvCache).unwrap();
    p.track_alloc(500, AllocationTag::KvCache).unwrap();
    p.track_free(a);
    let snap = p.snapshot();
    assert_eq!(snap.by_category.get(&AllocationTag::KvCache).copied(), Some(500));
}

// ── Env-var gating ───────────────────────────────────────────────────────────

#[test]
#[serial_test::serial(bitnet_env)]
fn profiler_disabled_without_env() {
    temp_env::with_var("BITNET_GPU_MEMORY_PROFILE", None::<&str>, || {
        let p = MemoryProfiler::new(true);
        assert!(!p.is_enabled());
        // Allocations are silently ignored.
        let id = p.track_alloc(1024, AllocationTag::Weights).unwrap();
        assert_eq!(id, 0);
        assert_eq!(p.total_allocated(), 0);
    });
}

#[test]
#[serial_test::serial(bitnet_env)]
fn profiler_enabled_with_env() {
    temp_env::with_var("BITNET_GPU_MEMORY_PROFILE", Some("1"), || {
        let p = MemoryProfiler::new(true);
        assert!(p.is_enabled());
        let id = p.track_alloc(1024, AllocationTag::Weights).unwrap();
        assert!(id > 0);
    });
}

// ── Optimizer ────────────────────────────────────────────────────────────────

#[test]
fn optimizer_detects_steady_pattern() {
    let p = MemoryProfiler::new(false);
    for _ in 0..6 {
        p.track_alloc(100, AllocationTag::Weights).unwrap();
    }
    let opt = AllocationOptimizer::from_profiler(&p);
    assert_eq!(opt.pattern_for(&AllocationTag::Weights), AllocationPattern::Steady);
}

#[test]
fn optimizer_detects_growing_pattern() {
    let p = MemoryProfiler::new(false);
    for size in &[100u64, 100, 100, 200, 300, 400] {
        p.track_alloc(*size, AllocationTag::Activations).unwrap();
    }
    let opt = AllocationOptimizer::from_profiler(&p);
    assert_eq!(opt.pattern_for(&AllocationTag::Activations), AllocationPattern::Growing);
}

#[test]
fn optimizer_suggested_prealloc() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(100, AllocationTag::KvCache).unwrap();
    p.track_alloc(400, AllocationTag::KvCache).unwrap();
    p.track_alloc(200, AllocationTag::KvCache).unwrap();
    let opt = AllocationOptimizer::from_profiler(&p);
    assert_eq!(opt.suggested_prealloc(&AllocationTag::KvCache), Some(400));
}

#[test]
fn optimizer_suggested_pool_size() {
    let p = MemoryProfiler::new(false);
    p.track_alloc(100, AllocationTag::Scratch).unwrap();
    p.track_alloc(200, AllocationTag::Scratch).unwrap();
    let opt = AllocationOptimizer::from_profiler(&p);
    assert_eq!(opt.suggested_pool_size(&AllocationTag::Scratch), Some(300));
}

#[test]
fn optimizer_fragmentation_gaps() {
    let p = MemoryProfiler::new(false);
    let a = p.track_alloc(256, AllocationTag::Activations).unwrap();
    p.track_alloc(512, AllocationTag::Activations).unwrap();
    p.track_free(a);
    let opt = AllocationOptimizer::from_profiler(&p);
    let gaps = opt.fragmentation_gaps();
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].size, 256);
}

#[test]
fn optimizer_recommendations_for_growing() {
    let p = MemoryProfiler::new(false);
    for size in &[10u64, 10, 10, 50, 100, 200] {
        p.track_alloc(*size, AllocationTag::KvCache).unwrap();
    }
    let opt = AllocationOptimizer::from_profiler(&p);
    let recs = opt.recommendations();
    assert!(
        recs.iter().any(|r| r.message.contains("growing")),
        "expected a 'growing' recommendation"
    );
}

#[test]
fn optimizer_recommendations_for_steady_with_pool() {
    let p = MemoryProfiler::new(false);
    for _ in 0..6 {
        p.track_alloc(100, AllocationTag::Scratch).unwrap();
    }
    let opt = AllocationOptimizer::from_profiler(&p);
    let recs = opt.recommendations();
    assert!(
        recs.iter().any(|r| r.message.contains("pre-allocate")),
        "expected a pre-allocate recommendation"
    );
}

#[test]
fn optimizer_no_data_returns_unknown_pattern() {
    let p = MemoryProfiler::new(false);
    let opt = AllocationOptimizer::from_profiler(&p);
    assert_eq!(opt.pattern_for(&AllocationTag::Weights), AllocationPattern::Unknown);
    assert!(opt.suggested_prealloc(&AllocationTag::Weights).is_none());
}
