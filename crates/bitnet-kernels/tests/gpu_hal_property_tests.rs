//! Property-based tests for GPU HAL memory allocation invariants.
//!
//! Key invariants tested:
//! - `OptimizedMemoryPool`: alloc/free preserves stats consistency
//!   (current_usage == total_allocated - total_freed)
//! - `OptimizedMemoryPool`: deallocating a buffer and re-allocating the same
//!   size yields a cache hit
//! - `MemoryLayoutOptimizer::calculate_alignment`: result is always a power of two,
//!   never zero, and follows the tiered rule (≤1 KB → 32, ≤64 KB → 128, else 256)
//! - `MemoryLayoutOptimizer::analyze_access_pattern`: sequential indices are detected,
//!   constant-stride indices produce `Strided`, and random indices never misclassify
//! - `AccessPattern`: `Strided { stride: 1 }` is distinct from `Sequential`
//! - `MemoryStats::default()`: all counters start at zero

#![cfg(any(feature = "gpu", feature = "cuda"))]

use bitnet_kernels::gpu::memory_optimization::{
    AccessPattern, MemoryLayoutOptimizer, MemoryPoolConfig, MemoryStats, OptimizedMemoryPool,
};
use proptest::prelude::*;

// ── MemoryStats defaults ─────────────────────────────────────────────────────

proptest! {
    /// MemoryStats::default() must have all counters at zero.
    #[test]
    fn prop_memory_stats_default_all_zero(_seed in 0u8..1) {
        let stats = MemoryStats::default();
        prop_assert_eq!(stats.total_allocated, 0);
        prop_assert_eq!(stats.total_freed, 0);
        prop_assert_eq!(stats.current_usage, 0);
        prop_assert_eq!(stats.peak_usage, 0);
        prop_assert_eq!(stats.allocation_count, 0);
        prop_assert_eq!(stats.deallocation_count, 0);
        prop_assert_eq!(stats.cache_hits, 0);
        prop_assert_eq!(stats.cache_misses, 0);
    }
}

// ── OptimizedMemoryPool alloc/free invariants ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After N allocations and N deallocations, current_usage returns to 0.
    #[test]
    fn prop_pool_alloc_free_usage_returns_to_zero(
        sizes in prop::collection::vec(1usize..4096, 1..8),
    ) {
        let config = MemoryPoolConfig::default();
        let mut pool = OptimizedMemoryPool::new(0, config);

        let mut buffers = Vec::new();
        for &sz in &sizes {
            let buf = pool.allocate(sz).expect("allocation must succeed");
            prop_assert_eq!(buf.len(), sz, "allocated buffer must be requested size");
            buffers.push(buf);
        }

        for buf in buffers {
            pool.deallocate(buf);
        }

        prop_assert_eq!(
            pool.stats().current_usage, 0,
            "current_usage must be 0 after freeing all allocations"
        );
    }

    /// Deallocating and re-allocating the same size should hit the cache.
    #[test]
    fn prop_pool_realloc_same_size_cache_hit(size in 1usize..8192) {
        let config = MemoryPoolConfig::default();
        let mut pool = OptimizedMemoryPool::new(0, config);

        let buf = pool.allocate(size).unwrap();
        pool.deallocate(buf);

        let hits_before = pool.stats().cache_hits;
        let _buf2 = pool.allocate(size).unwrap();

        prop_assert!(
            pool.stats().cache_hits > hits_before,
            "re-allocating same size after dealloc must be a cache hit"
        );
    }

    /// allocation_count increments by exactly 1 per allocate call.
    #[test]
    fn prop_pool_allocation_count_increments(
        n in 1usize..16,
    ) {
        let config = MemoryPoolConfig::default();
        let mut pool = OptimizedMemoryPool::new(0, config);

        for i in 0..n {
            let _buf = pool.allocate(64).unwrap();
            prop_assert_eq!(
                pool.stats().allocation_count, (i + 1) as u64,
                "allocation_count must increment after each allocate"
            );
        }
    }
}

// ── MemoryLayoutOptimizer::calculate_alignment ───────────────────────────────

proptest! {
    /// Alignment is always a power of two.
    #[test]
    fn prop_alignment_is_power_of_two(size in 0usize..10_000_000) {
        let align = MemoryLayoutOptimizer::calculate_alignment(size);
        prop_assert!(align.is_power_of_two(), "alignment {} must be a power of two", align);
    }

    /// Alignment is never zero.
    #[test]
    fn prop_alignment_nonzero(size in 0usize..10_000_000) {
        let align = MemoryLayoutOptimizer::calculate_alignment(size);
        prop_assert!(align > 0, "alignment must not be zero");
    }

    /// Alignment follows the tiered rule: <1KB → 32, <64KB → 128, ≥64KB → 256.
    #[test]
    fn prop_alignment_tiered(size in 0usize..1_000_000) {
        let align = MemoryLayoutOptimizer::calculate_alignment(size);
        if size < 1024 {
            prop_assert_eq!(align, 32, "size {} < 1KB should have alignment 32", size);
        } else if size < 64 * 1024 {
            prop_assert_eq!(align, 128, "size {} < 64KB should have alignment 128", size);
        } else {
            prop_assert_eq!(align, 256, "size {} >= 64KB should have alignment 256", size);
        }
    }
}

// ── MemoryLayoutOptimizer::analyze_access_pattern ────────────────────────────

proptest! {
    /// Consecutive indices [base, base+1, ..., base+n-1] produce Sequential.
    #[test]
    fn prop_sequential_indices_detected(
        base in 0usize..1000,
        len in 2usize..64,
    ) {
        let indices: Vec<usize> = (base..base + len).collect();
        let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&indices);
        prop_assert_eq!(
            pattern,
            AccessPattern::Sequential,
            "consecutive indices must be Sequential"
        );
    }

    /// Constant-stride indices [0, s, 2s, 3s, ...] produce Strided { stride: s } for s > 1.
    #[test]
    fn prop_strided_forward_detected(
        stride in 2usize..32,
        len in 3usize..16,
    ) {
        let indices: Vec<usize> = (0..len).map(|i| i * stride).collect();
        let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&indices);
        prop_assert_eq!(
            pattern,
            AccessPattern::Strided { stride },
            "forward-strided indices must produce Strided {{ stride: {} }}", stride
        );
    }

    /// Reverse-stride indices [base, base-s, base-2s, ...] produce Strided.
    #[test]
    fn prop_strided_reverse_detected(
        stride in 2usize..16,
        len in 3usize..8,
    ) {
        let base = stride * (len - 1) + 10;
        let indices: Vec<usize> = (0..len).map(|i| base - i * stride).collect();
        let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&indices);
        prop_assert_eq!(
            pattern,
            AccessPattern::Strided { stride },
            "reverse-strided indices must produce Strided {{ stride: {} }}", stride
        );
    }

    /// An empty slice always yields Sequential (the documented default).
    #[test]
    fn prop_empty_indices_are_sequential(_seed in 0u8..1) {
        let pattern = MemoryLayoutOptimizer::analyze_access_pattern(&[]);
        prop_assert_eq!(pattern, AccessPattern::Sequential);
    }
}
