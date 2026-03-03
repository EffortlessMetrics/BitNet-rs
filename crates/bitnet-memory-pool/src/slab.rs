//! Fixed-size slab allocator.
//!
//! Pre-allocates a contiguous region divided into equal-sized slots and hands
//! them out one at a time. Freed slots are returned to a free-list for reuse.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::stats::PoolStats;

/// A thread-safe slab allocator with fixed-size slots.
pub struct SlabAllocator {
    inner: Mutex<SlabInner>,
}

struct SlabInner {
    buf: NonNull<u8>,
    layout: Layout,
    slot_size: usize,
    slot_count: usize,
    free_list: Vec<usize>,
    allocated: u64,
    freed: u64,
    peak_slots: usize,
    current_slots: usize,
}

// SAFETY: the mutex serialises access; the buffer is exclusively owned.
unsafe impl Send for SlabInner {}

impl SlabAllocator {
    /// Creates a new slab allocator with `slot_count` slots of `slot_size` bytes each.
    ///
    /// `slot_size` is rounded up to the next multiple of 64 for cache-line alignment.
    ///
    /// # Panics
    ///
    /// Panics if `slot_size` or `slot_count` is zero, or allocation fails.
    #[must_use]
    pub fn new(slot_size: usize, slot_count: usize) -> Self {
        assert!(slot_size > 0, "slot size must be non-zero");
        assert!(slot_count > 0, "slot count must be non-zero");
        // Round up to 64-byte alignment for cache-line friendliness.
        let aligned_slot = (slot_size + 63) & !63;
        let total = aligned_slot.checked_mul(slot_count).expect("slab total size overflow");
        let layout = Layout::from_size_align(total, 64).expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let buf = unsafe { alloc::alloc_zeroed(layout) };
        let buf = NonNull::new(buf).expect("slab allocation failed");

        let free_list: Vec<usize> = (0..slot_count).rev().collect();

        Self {
            inner: Mutex::new(SlabInner {
                buf,
                layout,
                slot_size: aligned_slot,
                slot_count,
                free_list,
                allocated: 0,
                freed: 0,
                peak_slots: 0,
                current_slots: 0,
            }),
        }
    }

    /// Allocates one slot, returning a pointer to it.
    ///
    /// Returns `None` if all slots are in use.
    pub fn alloc(&self) -> Option<NonNull<u8>> {
        let mut inner = self.inner.lock().expect("slab lock poisoned");
        let idx = inner.free_list.pop()?;
        let offset = idx * inner.slot_size;
        inner.allocated += inner.slot_size as u64;
        inner.current_slots += 1;
        if inner.current_slots > inner.peak_slots {
            inner.peak_slots = inner.current_slots;
        }
        NonNull::new(unsafe { inner.buf.as_ptr().add(offset) })
    }

    /// Returns a previously allocated slot to the free-list.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to [`SlabAllocator::alloc`]
    /// on this same allocator and must not have been freed already.
    pub unsafe fn dealloc(&self, ptr: NonNull<u8>) {
        let mut inner = self.inner.lock().expect("slab lock poisoned");
        let base = inner.buf.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;
        debug_assert!(addr >= base, "pointer below slab base");
        let offset = addr - base;
        debug_assert!(
            offset.is_multiple_of(inner.slot_size),
            "pointer not aligned to slot boundary"
        );
        let idx = offset / inner.slot_size;
        debug_assert!(idx < inner.slot_count, "slot index out of range");
        inner.free_list.push(idx);
        inner.freed += inner.slot_size as u64;
        inner.current_slots -= 1;
    }

    /// Returns the (aligned) slot size in bytes.
    #[must_use]
    pub fn slot_size(&self) -> usize {
        self.inner.lock().expect("slab lock poisoned").slot_size
    }

    /// Returns the total number of slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.inner.lock().expect("slab lock poisoned").slot_count
    }

    /// Returns the number of slots currently in use.
    #[must_use]
    pub fn in_use(&self) -> usize {
        self.inner.lock().expect("slab lock poisoned").current_slots
    }

    /// Returns the number of free slots.
    #[must_use]
    pub fn available(&self) -> usize {
        let inner = self.inner.lock().expect("slab lock poisoned");
        inner.slot_count - inner.current_slots
    }

    /// Returns a snapshot of pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().expect("slab lock poisoned");
        PoolStats {
            allocated_bytes: inner.allocated,
            freed_bytes: inner.freed,
            peak_bytes: (inner.peak_slots as u64) * (inner.slot_size as u64),
            allocation_count: inner.allocated / inner.slot_size as u64,
            deallocation_count: inner.freed / inner.slot_size as u64,
            capacity_bytes: inner.layout.size() as u64,
        }
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().expect("slab lock poisoned");
        // SAFETY: buf was allocated with this layout.
        unsafe { alloc::dealloc(inner.buf.as_ptr(), inner.layout) }
    }
}
