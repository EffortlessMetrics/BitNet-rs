//! Bump / arena allocator.
//!
//! Allocates linearly from a contiguous buffer and frees everything at once via
//! [`ArenaAllocator::reset`].

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::stats::PoolStats;

/// A thread-safe bump allocator backed by a single contiguous buffer.
///
/// Individual allocations cannot be freed; call [`ArenaAllocator::reset`] to
/// reclaim all memory at once.
pub struct ArenaAllocator {
    inner: Mutex<ArenaInner>,
}

struct ArenaInner {
    buf: NonNull<u8>,
    layout: Layout,
    offset: usize,
    peak: usize,
    alloc_count: u64,
}

// SAFETY: the mutex serialises access; the buffer is exclusively owned.
unsafe impl Send for ArenaInner {}

impl ArenaAllocator {
    /// Creates a new arena with `capacity` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero or allocation fails.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "arena capacity must be non-zero");
        let layout = Layout::from_size_align(capacity, 64).expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let buf = unsafe { alloc::alloc_zeroed(layout) };
        let buf = NonNull::new(buf).expect("arena allocation failed");
        Self { inner: Mutex::new(ArenaInner { buf, layout, offset: 0, peak: 0, alloc_count: 0 }) }
    }

    /// Allocates `size` bytes with `align`-byte alignment from the arena.
    ///
    /// Returns `None` if the arena does not have enough remaining capacity.
    pub fn alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }
        let mut inner = self.inner.lock().expect("arena lock poisoned");
        let base = inner.buf.as_ptr() as usize;
        let current = base + inner.offset;
        let aligned = (current + align - 1) & !(align - 1);
        let padding = aligned - current;
        let needed = padding + size;
        if inner.offset + needed > inner.layout.size() {
            return None;
        }
        inner.offset += needed;
        if inner.offset > inner.peak {
            inner.peak = inner.offset;
        }
        inner.alloc_count += 1;
        NonNull::new((base + inner.offset - size) as *mut u8)
    }

    /// Resets the arena, logically freeing all allocations.
    ///
    /// Previously returned pointers become invalid after this call.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().expect("arena lock poisoned");
        inner.offset = 0;
        inner.alloc_count = 0;
    }

    /// Returns the total capacity of the arena in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.lock().expect("arena lock poisoned").layout.size()
    }

    /// Returns the number of bytes currently in use.
    #[must_use]
    pub fn used(&self) -> usize {
        self.inner.lock().expect("arena lock poisoned").offset
    }

    /// Returns the number of bytes still available.
    #[must_use]
    pub fn remaining(&self) -> usize {
        let inner = self.inner.lock().expect("arena lock poisoned");
        inner.layout.size() - inner.offset
    }

    /// Returns a snapshot of pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().expect("arena lock poisoned");
        PoolStats {
            allocated_bytes: inner.offset as u64,
            freed_bytes: 0,
            peak_bytes: inner.peak as u64,
            allocation_count: inner.alloc_count,
            deallocation_count: 0,
            capacity_bytes: inner.layout.size() as u64,
        }
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().expect("arena lock poisoned");
        // SAFETY: buf was allocated with this layout.
        unsafe { alloc::dealloc(inner.buf.as_ptr(), inner.layout) }
    }
}
