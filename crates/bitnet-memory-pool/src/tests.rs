//! Comprehensive tests for the memory pool allocator.

use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;

use crate::align::{AlignedAlloc, Alignment};
use crate::arena::ArenaAllocator;
use crate::pool::MemoryPool;
use crate::slab::SlabAllocator;
use crate::stats::PoolStats;

// ===== Alignment tests =====

#[test]
fn alignment_as_bytes() {
    assert_eq!(Alignment::Align16.as_bytes(), 16);
    assert_eq!(Alignment::Align32.as_bytes(), 32);
    assert_eq!(Alignment::Align64.as_bytes(), 64);
}

#[test]
fn alignment_is_aligned_true() {
    assert!(Alignment::Align16.is_aligned(0));
    assert!(Alignment::Align16.is_aligned(16));
    assert!(Alignment::Align16.is_aligned(32));
    assert!(Alignment::Align32.is_aligned(64));
    assert!(Alignment::Align64.is_aligned(128));
}

#[test]
fn alignment_is_aligned_false() {
    assert!(!Alignment::Align16.is_aligned(1));
    assert!(!Alignment::Align32.is_aligned(16));
    assert!(!Alignment::Align64.is_aligned(32));
}

#[test]
fn aligned_alloc_16() {
    let a = AlignedAlloc::new(128, Alignment::Align16);
    assert_eq!(a.size(), 128);
    assert_eq!(a.alignment(), 16);
    assert!(Alignment::Align16.is_aligned(a.as_ptr() as usize));
}

#[test]
fn aligned_alloc_32() {
    let a = AlignedAlloc::new(256, Alignment::Align32);
    assert!(Alignment::Align32.is_aligned(a.as_ptr() as usize));
}

#[test]
fn aligned_alloc_64() {
    let a = AlignedAlloc::new(512, Alignment::Align64);
    assert!(Alignment::Align64.is_aligned(a.as_ptr() as usize));
}

#[test]
fn aligned_alloc_zeroed() {
    let a = AlignedAlloc::new(1024, Alignment::Align64);
    assert!(a.as_slice().iter().all(|&b| b == 0));
}

#[test]
fn aligned_alloc_write_read() {
    let mut a = AlignedAlloc::new(64, Alignment::Align16);
    a.as_slice_mut()[0] = 0xAB;
    a.as_slice_mut()[63] = 0xCD;
    assert_eq!(a.as_slice()[0], 0xAB);
    assert_eq!(a.as_slice()[63], 0xCD);
}

#[test]
fn aligned_alloc_mut_ptr() {
    let mut a = AlignedAlloc::new(32, Alignment::Align32);
    let p = a.as_mut_ptr();
    assert!(!p.is_null());
}

#[test]
#[should_panic(expected = "allocation size must be non-zero")]
fn aligned_alloc_zero_size_panics() {
    let _a = AlignedAlloc::new(0, Alignment::Align16);
}

#[test]
fn aligned_alloc_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<AlignedAlloc>();
    assert_sync::<AlignedAlloc>();
}

// ===== Arena tests =====

#[test]
fn arena_create() {
    let arena = ArenaAllocator::new(4096);
    assert_eq!(arena.capacity(), 4096);
    assert_eq!(arena.used(), 0);
    assert_eq!(arena.remaining(), 4096);
}

#[test]
fn arena_simple_alloc() {
    let arena = ArenaAllocator::new(4096);
    let ptr = arena.alloc(128, 1);
    assert!(ptr.is_some());
    assert!(arena.used() >= 128);
}

#[test]
fn arena_aligned_alloc() {
    let arena = ArenaAllocator::new(4096);
    let ptr = arena.alloc(128, 64).unwrap();
    assert_eq!(ptr.as_ptr() as usize % 64, 0);
}

#[test]
fn arena_multiple_allocs() {
    let arena = ArenaAllocator::new(4096);
    let p1 = arena.alloc(64, 1).unwrap();
    let p2 = arena.alloc(64, 1).unwrap();
    assert_ne!(p1.as_ptr(), p2.as_ptr());
}

#[test]
fn arena_exhaustion() {
    let arena = ArenaAllocator::new(128);
    let _p1 = arena.alloc(64, 1).unwrap();
    let _p2 = arena.alloc(64, 1).unwrap();
    assert!(arena.alloc(64, 1).is_none());
}

#[test]
fn arena_reset() {
    let arena = ArenaAllocator::new(256);
    let _ = arena.alloc(128, 1);
    arena.reset();
    assert_eq!(arena.used(), 0);
    assert_eq!(arena.remaining(), 256);
}

#[test]
fn arena_reset_allows_reuse() {
    let arena = ArenaAllocator::new(128);
    let _ = arena.alloc(128, 1).unwrap();
    assert!(arena.alloc(1, 1).is_none());
    arena.reset();
    assert!(arena.alloc(128, 1).is_some());
}

#[test]
fn arena_zero_size_returns_none() {
    let arena = ArenaAllocator::new(256);
    assert!(arena.alloc(0, 1).is_none());
}

#[test]
fn arena_stats() {
    let arena = ArenaAllocator::new(1024);
    let _ = arena.alloc(100, 1).unwrap();
    let _ = arena.alloc(200, 1).unwrap();
    let stats = arena.stats();
    assert!(stats.allocated_bytes >= 300);
    assert_eq!(stats.allocation_count, 2);
    assert!(stats.peak_bytes >= 300);
}

#[test]
fn arena_peak_after_reset() {
    let arena = ArenaAllocator::new(1024);
    let _ = arena.alloc(512, 1).unwrap();
    let peak_before = arena.stats().peak_bytes;
    arena.reset();
    let peak_after = arena.stats().peak_bytes;
    assert_eq!(peak_before, peak_after);
}

#[test]
#[should_panic(expected = "arena capacity must be non-zero")]
fn arena_zero_capacity_panics() {
    let _a = ArenaAllocator::new(0);
}

#[test]
fn arena_thread_safe() {
    let arena = Arc::new(ArenaAllocator::new(65536));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let a = Arc::clone(&arena);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = a.alloc(64, 1);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    // No data race or panic.
}

// ===== Slab tests =====

#[test]
fn slab_create() {
    let slab = SlabAllocator::new(128, 16);
    assert_eq!(slab.slot_count(), 16);
    assert!(slab.slot_size() >= 128);
    assert_eq!(slab.in_use(), 0);
    assert_eq!(slab.available(), 16);
}

#[test]
fn slab_alloc_dealloc() {
    let slab = SlabAllocator::new(128, 8);
    let ptr = slab.alloc().unwrap();
    assert_eq!(slab.in_use(), 1);
    unsafe { slab.dealloc(ptr) };
    assert_eq!(slab.in_use(), 0);
}

#[test]
fn slab_exhaustion() {
    let slab = SlabAllocator::new(64, 2);
    let _p1 = slab.alloc().unwrap();
    let _p2 = slab.alloc().unwrap();
    assert!(slab.alloc().is_none());
}

#[test]
fn slab_reuse_after_dealloc() {
    let slab = SlabAllocator::new(64, 1);
    let p = slab.alloc().unwrap();
    unsafe { slab.dealloc(p) };
    assert!(slab.alloc().is_some());
}

#[test]
fn slab_alignment() {
    let slab = SlabAllocator::new(100, 4);
    for _ in 0..4 {
        let p = slab.alloc().unwrap();
        assert_eq!(p.as_ptr() as usize % 64, 0);
    }
}

#[test]
fn slab_unique_pointers() {
    let slab = SlabAllocator::new(64, 4);
    let ptrs: Vec<NonNull<u8>> = (0..4).map(|_| slab.alloc().unwrap()).collect();
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            assert_ne!(ptrs[i].as_ptr(), ptrs[j].as_ptr());
        }
    }
}

#[test]
fn slab_stats() {
    let slab = SlabAllocator::new(64, 4);
    let p1 = slab.alloc().unwrap();
    let _p2 = slab.alloc().unwrap();
    unsafe { slab.dealloc(p1) };
    let stats = slab.stats();
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.deallocation_count, 1);
    assert!(stats.allocated_bytes >= 128);
    assert!(stats.freed_bytes >= 64);
}

#[test]
fn slab_peak_usage() {
    let slab = SlabAllocator::new(64, 4);
    let p1 = slab.alloc().unwrap();
    let p2 = slab.alloc().unwrap();
    let p3 = slab.alloc().unwrap();
    unsafe { slab.dealloc(p1) };
    unsafe { slab.dealloc(p2) };
    unsafe { slab.dealloc(p3) };
    let stats = slab.stats();
    assert_eq!(stats.peak_bytes, 3 * slab.slot_size() as u64);
}

#[test]
#[should_panic(expected = "slot size must be non-zero")]
fn slab_zero_size_panics() {
    let _s = SlabAllocator::new(0, 4);
}

#[test]
#[should_panic(expected = "slot count must be non-zero")]
fn slab_zero_count_panics() {
    let _s = SlabAllocator::new(64, 0);
}

#[test]
fn slab_slot_size_rounded_up() {
    let slab = SlabAllocator::new(100, 4);
    assert_eq!(slab.slot_size() % 64, 0);
    assert!(slab.slot_size() >= 100);
}

#[test]
fn slab_thread_safe() {
    let slab = Arc::new(SlabAllocator::new(64, 256));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let s = Arc::clone(&slab);
            thread::spawn(move || {
                for _ in 0..50 {
                    if let Some(p) = s.alloc() {
                        unsafe { s.dealloc(p) };
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

// ===== Pool tests =====

#[test]
fn pool_default() {
    let pool = MemoryPool::new();
    assert_eq!(pool.class_count(), 7);
}

#[test]
fn pool_custom_config() {
    let pool = MemoryPool::with_config(&[128, 512], 16);
    assert_eq!(pool.class_count(), 2);
}

#[test]
fn pool_alloc_small() {
    let pool = MemoryPool::new();
    let guard = pool.alloc(32);
    assert_eq!(guard.size(), 32);
    assert!(guard.as_slice().iter().all(|&b| b == 0));
}

#[test]
fn pool_alloc_exact_class() {
    let pool = MemoryPool::new();
    let guard = pool.alloc(64);
    assert_eq!(guard.size(), 64);
}

#[test]
fn pool_alloc_large_fallback() {
    let pool = MemoryPool::with_config(&[64], 2);
    // Exceeds all size classes → fallback.
    let guard = pool.alloc(1024);
    assert_eq!(guard.size(), 1024);
}

#[test]
fn pool_guard_write_read() {
    let pool = MemoryPool::new();
    let mut guard = pool.alloc(128);
    guard.as_slice_mut()[0] = 42;
    assert_eq!(guard.as_slice()[0], 42);
}

#[test]
fn pool_guard_mut_ptr() {
    let pool = MemoryPool::new();
    let mut guard = pool.alloc(64);
    let p = guard.as_mut_ptr();
    assert!(!p.is_null());
}

#[test]
fn pool_guard_raii_return() {
    let pool = MemoryPool::with_config(&[64], 1);
    {
        let _g = pool.alloc(32);
        // Slab slot is occupied.
        // The second alloc would need to fallback since only 1 slot.
    }
    // Guard dropped — slot returned. Should be able to alloc from slab again.
    let _g2 = pool.alloc(32);
    let stats = pool.stats();
    // We should see 2 allocations and 1 deallocation from the slab.
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.deallocation_count, 1);
}

#[test]
fn pool_multiple_allocs() {
    let pool = MemoryPool::new();
    let guards: Vec<_> = (0..10).map(|_| pool.alloc(64)).collect();
    for (i, g) in guards.iter().enumerate() {
        for (j, h) in guards.iter().enumerate() {
            if i != j {
                assert_ne!(g.as_ptr(), h.as_ptr());
            }
        }
    }
}

#[test]
fn pool_stats_initial() {
    let pool = MemoryPool::new();
    let stats = pool.stats();
    assert_eq!(stats.allocated_bytes, 0);
    assert_eq!(stats.freed_bytes, 0);
    assert_eq!(stats.allocation_count, 0);
}

#[test]
fn pool_stats_after_alloc() {
    let pool = MemoryPool::new();
    let _g = pool.alloc(64);
    let stats = pool.stats();
    assert!(stats.allocated_bytes > 0);
    assert_eq!(stats.allocation_count, 1);
}

#[test]
fn pool_stats_after_free() {
    let pool = MemoryPool::new();
    {
        let _g = pool.alloc(64);
    }
    let stats = pool.stats();
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.deallocation_count, 1);
}

#[test]
fn pool_stats_fallback_tracking() {
    let pool = MemoryPool::with_config(&[64], 2);
    {
        let _g = pool.alloc(1024);
    }
    let stats = pool.stats();
    assert!(stats.freed_bytes >= 1024);
}

#[test]
fn pool_clone_shares_state() {
    let pool = MemoryPool::new();
    let pool2 = pool.clone();
    let _g = pool.alloc(64);
    let stats = pool2.stats();
    assert_eq!(stats.allocation_count, 1);
}

#[test]
fn pool_thread_safe_alloc() {
    let pool = MemoryPool::with_config(&[256], 256);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let p = pool.clone();
            thread::spawn(move || {
                let mut guards = Vec::new();
                for _ in 0..20 {
                    guards.push(p.alloc(64));
                }
                drop(guards);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let stats = pool.stats();
    assert_eq!(stats.allocation_count, 160);
    assert_eq!(stats.deallocation_count, 160);
}

#[test]
fn pool_thread_safe_mixed_sizes() {
    let pool = MemoryPool::new();
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let p = pool.clone();
            thread::spawn(move || {
                let size = 64 * (i + 1);
                let mut guards = Vec::new();
                for _ in 0..10 {
                    guards.push(p.alloc(size));
                }
                drop(guards);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn pool_guard_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<crate::pool::PoolGuard>();
    assert_sync::<crate::pool::PoolGuard>();
}

#[test]
fn pool_guard_send_across_threads() {
    let pool = MemoryPool::new();
    let mut guard = pool.alloc(128);
    guard.as_slice_mut()[0] = 99;
    let handle = thread::spawn(move || {
        assert_eq!(guard.as_slice()[0], 99);
    });
    handle.join().unwrap();
}

#[test]
#[should_panic(expected = "allocation size must be non-zero")]
fn pool_alloc_zero_panics() {
    let pool = MemoryPool::new();
    let _g = pool.alloc(0);
}

#[test]
#[should_panic(expected = "need at least one size class")]
fn pool_empty_classes_panics() {
    let _p = MemoryPool::with_config(&[], 4);
}

#[test]
fn pool_dedup_classes() {
    let pool = MemoryPool::with_config(&[64, 64, 128, 128], 4);
    assert_eq!(pool.class_count(), 2);
}

// ===== PoolStats tests =====

#[test]
fn stats_in_use_bytes() {
    let s = PoolStats {
        allocated_bytes: 1000,
        freed_bytes: 300,
        peak_bytes: 1000,
        allocation_count: 10,
        deallocation_count: 3,
        capacity_bytes: 4096,
    };
    assert_eq!(s.in_use_bytes(), 700);
}

#[test]
fn stats_in_use_saturating() {
    let s = PoolStats {
        allocated_bytes: 0,
        freed_bytes: 100,
        peak_bytes: 0,
        allocation_count: 0,
        deallocation_count: 0,
        capacity_bytes: 0,
    };
    assert_eq!(s.in_use_bytes(), 0);
}

#[test]
fn stats_fragmentation_zero_capacity() {
    let s = PoolStats {
        allocated_bytes: 0,
        freed_bytes: 0,
        peak_bytes: 0,
        allocation_count: 0,
        deallocation_count: 0,
        capacity_bytes: 0,
    };
    assert!((s.fragmentation() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn stats_fragmentation_nothing_allocated() {
    let s = PoolStats {
        allocated_bytes: 0,
        freed_bytes: 0,
        peak_bytes: 0,
        allocation_count: 0,
        deallocation_count: 0,
        capacity_bytes: 4096,
    };
    assert!((s.fragmentation() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn stats_fragmentation_half_used() {
    let s = PoolStats {
        allocated_bytes: 2048,
        freed_bytes: 0,
        peak_bytes: 2048,
        allocation_count: 1,
        deallocation_count: 0,
        capacity_bytes: 4096,
    };
    assert!((s.fragmentation() - 0.5).abs() < 1e-10);
}

#[test]
fn stats_utilisation_full() {
    let s = PoolStats {
        allocated_bytes: 4096,
        freed_bytes: 0,
        peak_bytes: 4096,
        allocation_count: 1,
        deallocation_count: 0,
        capacity_bytes: 4096,
    };
    assert!((s.utilisation() - 1.0).abs() < 1e-10);
}

#[test]
fn stats_utilisation_empty() {
    let s = PoolStats {
        allocated_bytes: 0,
        freed_bytes: 0,
        peak_bytes: 0,
        allocation_count: 0,
        deallocation_count: 0,
        capacity_bytes: 4096,
    };
    assert!((s.utilisation() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn stats_utilisation_zero_capacity() {
    let s = PoolStats {
        allocated_bytes: 0,
        freed_bytes: 0,
        peak_bytes: 0,
        allocation_count: 0,
        deallocation_count: 0,
        capacity_bytes: 0,
    };
    assert!((s.utilisation() - 0.0).abs() < f64::EPSILON);
}

// ===== Additional edge-case tests =====

#[test]
fn arena_large_alignment_padding() {
    let arena = ArenaAllocator::new(4096);
    // First alloc of 1 byte misaligns the cursor.
    let _ = arena.alloc(1, 1).unwrap();
    // Second alloc with 64-byte alignment should still succeed.
    let p = arena.alloc(64, 64).unwrap();
    assert_eq!(p.as_ptr() as usize % 64, 0);
}

#[test]
fn pool_slab_exhaustion_fallback() {
    let pool = MemoryPool::with_config(&[64], 1);
    let _g1 = pool.alloc(32); // takes the single slab slot
    let _g2 = pool.alloc(32); // must fallback to system allocator
    let stats = pool.stats();
    // Both allocations should be tracked.
    assert_eq!(stats.allocation_count, 2);
}

#[test]
fn pool_allocation_alignment_check() {
    let pool = MemoryPool::new();
    for _ in 0..20 {
        let g = pool.alloc(64);
        // All slab slots are 64-byte aligned.
        assert_eq!(g.as_ptr() as usize % 64, 0);
    }
}

#[test]
fn pool_large_allocation() {
    let pool = MemoryPool::new();
    let g = pool.alloc(1_000_000);
    assert_eq!(g.size(), 1_000_000);
    // Fallback allocs use 64-byte alignment.
    assert_eq!(g.as_ptr() as usize % 64, 0);
}

#[test]
fn aligned_alloc_various_sizes() {
    for &size in &[1, 7, 16, 33, 64, 127, 256, 1023, 4096] {
        let a = AlignedAlloc::new(size, Alignment::Align64);
        assert_eq!(a.size(), size);
        assert!(Alignment::Align64.is_aligned(a.as_ptr() as usize));
    }
}

#[test]
fn slab_alloc_all_then_free_all() {
    let slab = SlabAllocator::new(64, 8);
    let ptrs: Vec<_> = (0..8).map(|_| slab.alloc().unwrap()).collect();
    assert!(slab.alloc().is_none());
    for p in ptrs {
        unsafe { slab.dealloc(p) };
    }
    assert_eq!(slab.available(), 8);
}

#[test]
fn slab_interleaved_alloc_dealloc() {
    let slab = SlabAllocator::new(64, 4);
    let p1 = slab.alloc().unwrap();
    let p2 = slab.alloc().unwrap();
    unsafe { slab.dealloc(p1) };
    let p3 = slab.alloc().unwrap();
    assert_eq!(slab.in_use(), 2);
    unsafe { slab.dealloc(p2) };
    unsafe { slab.dealloc(p3) };
    assert_eq!(slab.in_use(), 0);
}

#[test]
fn pool_default_is_same_as_new() {
    let p1 = MemoryPool::new();
    let p2 = MemoryPool::default();
    assert_eq!(p1.class_count(), p2.class_count());
}

// ===== proptest property tests =====

mod proptests {
    use proptest::prelude::*;

    use crate::align::{AlignedAlloc, Alignment};
    use crate::arena::ArenaAllocator;
    use crate::pool::MemoryPool;
    use crate::slab::SlabAllocator;

    proptest! {
        #[test]
        fn aligned_alloc_always_aligned(size in 1_usize..8192) {
            let a = AlignedAlloc::new(size, Alignment::Align64);
            prop_assert_eq!(a.as_ptr() as usize % 64, 0);
            prop_assert_eq!(a.size(), size);
        }

        #[test]
        fn aligned_alloc_zeroed(size in 1_usize..4096) {
            let a = AlignedAlloc::new(size, Alignment::Align16);
            prop_assert!(a.as_slice().iter().all(|&b| b == 0));
        }

        #[test]
        fn arena_alloc_within_capacity(
            cap in 256_usize..16384,
            alloc_size in 1_usize..256
        ) {
            let arena = ArenaAllocator::new(cap);
            if alloc_size <= cap {
                prop_assert!(arena.alloc(alloc_size, 1).is_some());
            }
        }

        #[test]
        fn arena_used_monotonic(allocs in proptest::collection::vec(1_usize..64, 1..20)) {
            let total: usize = allocs.iter().sum();
            let arena = ArenaAllocator::new(total + 4096);
            let mut prev_used = 0;
            for size in &allocs {
                let _ = arena.alloc(*size, 1);
                let used = arena.used();
                prop_assert!(used >= prev_used);
                prev_used = used;
            }
        }

        #[test]
        fn slab_available_plus_in_use_equals_count(
            slot_size in 64_usize..512,
            slot_count in 1_usize..32,
            allocs in 0_usize..32
        ) {
            let slab = SlabAllocator::new(slot_size, slot_count);
            let n = allocs.min(slot_count);
            let mut ptrs = Vec::new();
            for _ in 0..n {
                if let Some(p) = slab.alloc() {
                    ptrs.push(p);
                }
            }
            prop_assert_eq!(slab.in_use() + slab.available(), slot_count);
            for p in ptrs {
                unsafe { slab.dealloc(p) };
            }
        }

        #[test]
        fn pool_alloc_returns_correct_size(size in 1_usize..65536) {
            let pool = MemoryPool::new();
            let guard = pool.alloc(size);
            prop_assert_eq!(guard.size(), size);
        }

        #[test]
        fn pool_alloc_pointer_aligned_64(size in 1_usize..65536) {
            let pool = MemoryPool::new();
            let guard = pool.alloc(size);
            prop_assert_eq!(guard.as_ptr() as usize % 64, 0);
        }

        #[test]
        fn pool_stats_alloc_count_accurate(n in 1_usize..50) {
            let pool = MemoryPool::new();
            let guards: Vec<_> = (0..n).map(|_| pool.alloc(64)).collect();
            let stats = pool.stats();
            prop_assert_eq!(stats.allocation_count, n as u64);
            drop(guards);
            let stats2 = pool.stats();
            prop_assert_eq!(stats2.deallocation_count, n as u64);
        }

        #[test]
        fn pool_stats_in_use_non_negative(
            allocs in 1_usize..20,
            frees in 0_usize..20
        ) {
            let pool = MemoryPool::new();
            let mut guards: Vec<_> = (0..allocs).map(|_| pool.alloc(64)).collect();
            let to_free = frees.min(guards.len());
            guards.truncate(guards.len() - to_free);
            let stats = pool.stats();
            prop_assert!(stats.in_use_bytes() <= stats.allocated_bytes);
        }

        #[test]
        fn alignment_is_aligned_consistent(addr in 0_usize..1_000_000) {
            let a16 = Alignment::Align16;
            let a32 = Alignment::Align32;
            let a64 = Alignment::Align64;
            // 64-aligned ⇒ 32-aligned ⇒ 16-aligned
            if a64.is_aligned(addr) {
                prop_assert!(a32.is_aligned(addr));
                prop_assert!(a16.is_aligned(addr));
            }
            if a32.is_aligned(addr) {
                prop_assert!(a16.is_aligned(addr));
            }
        }

        #[test]
        fn arena_reset_restores_capacity(cap in 256_usize..8192) {
            let arena = ArenaAllocator::new(cap);
            let _ = arena.alloc(cap / 2, 1);
            arena.reset();
            prop_assert_eq!(arena.remaining(), cap);
        }

        #[test]
        fn slab_stats_freed_le_allocated(
            slot_count in 2_usize..16,
            alloc_n in 1_usize..16,
            free_n in 0_usize..16
        ) {
            let slab = SlabAllocator::new(64, slot_count);
            let n = alloc_n.min(slot_count);
            let mut ptrs = Vec::new();
            for _ in 0..n {
                if let Some(p) = slab.alloc() {
                    ptrs.push(p);
                }
            }
            let f = free_n.min(ptrs.len());
            for p in ptrs.drain(..f) {
                unsafe { slab.dealloc(p) };
            }
            let stats = slab.stats();
            prop_assert!(stats.freed_bytes <= stats.allocated_bytes);
            // clean up remaining
            for p in ptrs {
                unsafe { slab.dealloc(p) };
            }
        }
    }
}
