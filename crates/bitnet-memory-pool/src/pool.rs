//! Size-class memory pool with RAII guards.
//!
//! [`MemoryPool`] maintains separate [`SlabAllocator`]s for common allocation
//! sizes (size classes). Allocations that do not fit any size class fall back
//! to the system allocator. All returned [`PoolGuard`]s automatically return
//! memory to the pool on drop.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use crate::slab::SlabAllocator;
use crate::stats::PoolStats;

/// Default size classes (bytes) covering typical inference tensor scratch sizes.
const DEFAULT_SIZE_CLASSES: &[usize] = &[64, 256, 1024, 4096, 16384, 65536, 262_144];
/// Default number of slots per size class.
const DEFAULT_SLOTS_PER_CLASS: usize = 32;

/// A thread-safe size-class memory pool.
///
/// Each size class is backed by a [`SlabAllocator`]. Allocations that exceed
/// the largest class use the system allocator with 64-byte alignment.
pub struct MemoryPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    classes: Vec<SizeClass>,
    /// Fallback allocations (system allocator).
    fallback: Mutex<FallbackState>,
}

struct SizeClass {
    max_size: usize,
    slab: SlabAllocator,
}

struct FallbackState {
    allocated: u64,
    freed: u64,
    peak: u64,
    current: u64,
    alloc_count: u64,
    dealloc_count: u64,
}

impl MemoryPool {
    /// Creates a pool with the default size classes and 32 slots each.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_SIZE_CLASSES, DEFAULT_SLOTS_PER_CLASS)
    }

    /// Creates a pool with custom size classes and a uniform slot count.
    ///
    /// # Panics
    ///
    /// Panics if `size_classes` is empty.
    #[must_use]
    pub fn with_config(size_classes: &[usize], slots_per_class: usize) -> Self {
        assert!(!size_classes.is_empty(), "need at least one size class");
        let mut sorted: Vec<usize> = size_classes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let classes = sorted
            .into_iter()
            .map(|max_size| SizeClass {
                max_size,
                slab: SlabAllocator::new(max_size, slots_per_class),
            })
            .collect();

        Self {
            inner: Arc::new(PoolInner {
                classes,
                fallback: Mutex::new(FallbackState {
                    allocated: 0,
                    freed: 0,
                    peak: 0,
                    current: 0,
                    alloc_count: 0,
                    dealloc_count: 0,
                }),
            }),
        }
    }

    /// Allocates `size` bytes from the pool, returning an RAII guard.
    ///
    /// The guard automatically returns the memory to the pool on drop.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero or allocation fails.
    #[must_use]
    pub fn alloc(&self, size: usize) -> PoolGuard {
        assert!(size > 0, "allocation size must be non-zero");

        // Try to find a suitable size class.
        for (idx, class) in self.inner.classes.iter().enumerate() {
            if size <= class.max_size
                && let Some(ptr) = class.slab.alloc()
            {
                return PoolGuard {
                    ptr,
                    size,
                    source: GuardSource::Slab(idx),
                    pool: Arc::clone(&self.inner),
                };
            }
        }

        // Fallback: system allocator with 64-byte alignment.
        let layout = Layout::from_size_align(size, 64).expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        {
            let mut fb = self.inner.fallback.lock().expect("fallback lock poisoned");
            fb.allocated += size as u64;
            fb.current += size as u64;
            fb.alloc_count += 1;
            if fb.current > fb.peak {
                fb.peak = fb.current;
            }
        }
        PoolGuard {
            ptr,
            size,
            source: GuardSource::Fallback(layout),
            pool: Arc::clone(&self.inner),
        }
    }

    /// Returns aggregated statistics across all size classes and fallback.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let mut total = PoolStats {
            allocated_bytes: 0,
            freed_bytes: 0,
            peak_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            capacity_bytes: 0,
        };
        for class in &self.inner.classes {
            let s = class.slab.stats();
            total.allocated_bytes += s.allocated_bytes;
            total.freed_bytes += s.freed_bytes;
            total.peak_bytes += s.peak_bytes;
            total.allocation_count += s.allocation_count;
            total.deallocation_count += s.deallocation_count;
            total.capacity_bytes += s.capacity_bytes;
        }
        let fb = self.inner.fallback.lock().expect("fallback lock poisoned");
        total.allocated_bytes += fb.allocated;
        total.freed_bytes += fb.freed;
        total.peak_bytes += fb.peak;
        total.allocation_count += fb.alloc_count;
        total.deallocation_count += fb.dealloc_count;
        total
    }

    /// Returns the number of size classes.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.inner.classes.len()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MemoryPool {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// ---------- RAII guard ----------

enum GuardSource {
    Slab(usize),
    Fallback(Layout),
}

/// RAII guard for a pool allocation.
///
/// Automatically returns the memory to the originating pool on drop.
pub struct PoolGuard {
    ptr: NonNull<u8>,
    size: usize,
    source: GuardSource,
    pool: Arc<PoolInner>,
}

impl PoolGuard {
    /// Returns a pointer to the allocation.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a mutable pointer to the allocation.
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the requested allocation size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the allocation as a byte slice.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for at least `self.size` bytes and is zeroed on alloc.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Returns the allocation as a mutable byte slice.
    #[must_use]
    pub const fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid, unique, and initialised.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        match self.source {
            GuardSource::Slab(idx) => {
                // SAFETY: ptr was obtained from this slab and has not been freed.
                unsafe { self.pool.classes[idx].slab.dealloc(self.ptr) };
            }
            GuardSource::Fallback(layout) => {
                // SAFETY: ptr was allocated with this layout.
                unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
                let mut fb = self.pool.fallback.lock().expect("fallback lock poisoned");
                fb.freed += self.size as u64;
                fb.current -= self.size as u64;
                fb.dealloc_count += 1;
            }
        }
    }
}

// SAFETY: PoolGuard owns its allocation exclusively.
unsafe impl Send for PoolGuard {}
unsafe impl Sync for PoolGuard {}
