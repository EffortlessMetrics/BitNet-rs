//! Allocation audit primitives for CLI receipt instrumentation.
//!
//! This module owns the process-global allocator counters and scoped guard used
//! by the CUDA/CPU receipt paths. Keeping it separate from `main.rs` prevents
//! allocator state management from being mixed into command dispatch logic.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ALLOCATION_AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_AUDIT_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_AUDIT_DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub(crate) struct AllocationAuditAllocator;

unsafe impl GlobalAlloc for AllocationAuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_dealloc(layout.size());
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && ALLOCATION_AUDIT_ENABLED.load(Ordering::Relaxed) {
            record_dealloc(layout.size());
            record_alloc(new_size);
        }
        new_ptr
    }
}

fn record_alloc(size: usize) {
    ALLOCATION_AUDIT_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_ALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

fn record_dealloc(size: usize) {
    ALLOCATION_AUDIT_DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_AUDIT_DEALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AllocationAuditSnapshot {
    pub(crate) alloc_count: u64,
    pub(crate) alloc_bytes: u64,
    pub(crate) dealloc_count: u64,
    pub(crate) dealloc_bytes: u64,
}

impl AllocationAuditSnapshot {
    pub(crate) fn current() -> Self {
        Self {
            alloc_count: ALLOCATION_AUDIT_ALLOC_COUNT.load(Ordering::Relaxed),
            alloc_bytes: ALLOCATION_AUDIT_ALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_count: ALLOCATION_AUDIT_DEALLOC_COUNT.load(Ordering::Relaxed),
            dealloc_bytes: ALLOCATION_AUDIT_DEALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn delta_since(start: Self) -> Self {
        let current = Self::current();
        Self {
            alloc_count: current.alloc_count.saturating_sub(start.alloc_count),
            alloc_bytes: current.alloc_bytes.saturating_sub(start.alloc_bytes),
            dealloc_count: current.dealloc_count.saturating_sub(start.dealloc_count),
            dealloc_bytes: current.dealloc_bytes.saturating_sub(start.dealloc_bytes),
        }
    }
}

pub(crate) struct AllocationAuditGuard {
    previous: bool,
}

impl AllocationAuditGuard {
    pub(crate) fn enable(enabled: bool) -> Self {
        let previous = ALLOCATION_AUDIT_ENABLED.swap(enabled, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for AllocationAuditGuard {
    fn drop(&mut self) {
        ALLOCATION_AUDIT_ENABLED.store(self.previous, Ordering::Relaxed);
    }
}
