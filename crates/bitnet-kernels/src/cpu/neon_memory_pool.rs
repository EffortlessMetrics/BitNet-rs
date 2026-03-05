//! NEON-aligned arena-style memory pool for Apple Silicon tensor operations.
//!
//! Provides efficient buffer reuse with 16-byte alignment (NEON requirement)
//! and optional 64-byte cache-line alignment. Buffers are grouped into
//! size-class buckets (small ≤4 KB, medium ≤256 KB, large >256 KB) so that
//! similarly-sized allocations can be satisfied from a recycled buffer without
//! hitting the global allocator.

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ── Size-class thresholds ───────────────────────────────────────────────────

const SMALL_THRESHOLD: usize = 4 * 1024; // ≤ 4 KB
const MEDIUM_THRESHOLD: usize = 256 * 1024; // ≤ 256 KB

/// Default alignment satisfying NEON 128-bit register loads.
const NEON_ALIGN: usize = 16;

/// Cache-line alignment for avoiding false sharing.
const CACHE_LINE_ALIGN: usize = 64;

// ── Statistics ──────────────────────────────────────────────────────────────

/// Allocation statistics tracked by a [`TensorPool`].
#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub reuse_hits: AtomicU64,
    pub total_bytes_allocated: AtomicU64,
}

impl PoolStats {
    fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            reuse_hits: self.reuse_hits.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes_allocated.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time copy of [`PoolStats`] for inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStatsSnapshot {
    pub allocations: u64,
    pub deallocations: u64,
    pub reuse_hits: u64,
    pub total_bytes_allocated: u64,
}

// ── Size class ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SizeClass {
    Small,
    Medium,
    Large,
}

fn classify(size: usize) -> SizeClass {
    if size <= SMALL_THRESHOLD {
        SizeClass::Small
    } else if size <= MEDIUM_THRESHOLD {
        SizeClass::Medium
    } else {
        SizeClass::Large
    }
}

// ── Raw aligned buffer ─────────────────────────────────────────────────────

/// A raw, aligned heap allocation.
struct RawBuf {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Usable capacity in bytes (≥ requested size).
    capacity: usize,
}

// SAFETY: The buffer owns its allocation exclusively.
unsafe impl Send for RawBuf {}

impl RawBuf {
    fn new(size: usize, align: usize) -> Self {
        assert!(size > 0, "allocation size must be > 0");
        let layout = Layout::from_size_align(size, align).expect("invalid layout");
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        Self { ptr, layout, capacity: size }
    }
}

impl Drop for RawBuf {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with the same layout.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

// ── Bucket ──────────────────────────────────────────────────────────────────

/// Per-size-class free list.
#[derive(Default)]
struct Bucket {
    free: Vec<RawBuf>,
}

// ── TensorPool ──────────────────────────────────────────────────────────────

/// Arena-style memory pool that recycles `f32` tensor buffers.
///
/// Allocations are aligned to at least `NEON_ALIGN` bytes; pass
/// `cache_line_aligned = true` at construction to use `CACHE_LINE_ALIGN`
/// instead.
pub struct TensorPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    small: Mutex<Bucket>,
    medium: Mutex<Bucket>,
    large: Mutex<Bucket>,
    align: usize,
    stats: PoolStats,
}

impl TensorPool {
    /// Create a new pool with the default 16-byte NEON alignment.
    pub fn new() -> Self {
        Self::with_alignment(false)
    }

    /// Create a pool that aligns every allocation to a 64-byte cache line.
    pub fn cache_line_aligned() -> Self {
        Self::with_alignment(true)
    }

    fn with_alignment(cache_line: bool) -> Self {
        let align = if cache_line { CACHE_LINE_ALIGN } else { NEON_ALIGN };
        Self {
            inner: Arc::new(PoolInner {
                small: Mutex::new(Bucket::default()),
                medium: Mutex::new(Bucket::default()),
                large: Mutex::new(Bucket::default()),
                align,
                stats: PoolStats::default(),
            }),
        }
    }

    /// Allocate a buffer of at least `size` bytes.
    ///
    /// Returns a previously-pooled buffer when one of sufficient capacity is
    /// available; otherwise allocates fresh memory from the system.
    pub fn alloc(&self, size: usize) -> PoolBuffer {
        assert!(size > 0, "cannot allocate zero bytes");

        let class = classify(size);
        let bucket = self.bucket(class);
        let mut guard = bucket.lock().expect("poisoned");

        // Try to find a recycled buffer that is large enough.
        if let Some(idx) = guard.free.iter().position(|b| b.capacity >= size) {
            let buf = guard.free.swap_remove(idx);
            drop(guard);
            self.inner.stats.allocations.fetch_add(1, Ordering::Relaxed);
            self.inner.stats.reuse_hits.fetch_add(1, Ordering::Relaxed);
            return PoolBuffer { buf: Some(buf), len: size, pool: Arc::clone(&self.inner) };
        }
        drop(guard);

        // No suitable buffer — allocate a new one.
        let buf = RawBuf::new(size, self.inner.align);
        self.inner.stats.allocations.fetch_add(1, Ordering::Relaxed);
        self.inner.stats.total_bytes_allocated.fetch_add(size as u64, Ordering::Relaxed);

        PoolBuffer { buf: Some(buf), len: size, pool: Arc::clone(&self.inner) }
    }

    /// Soft reset — move all outstanding pooled memory back to the free lists.
    ///
    /// This does **not** release memory to the system; it merely marks all
    /// currently-free buffers as available for reuse. Outstanding
    /// [`PoolBuffer`]s are unaffected (they return to the pool on drop).
    pub fn clear(&self) {
        // The pool already returns buffers on drop; `clear` is a no-op on the
        // internal free lists because we never hand out references to the lists
        // themselves. We simply drain and re-add (effectively a no-op for the
        // current lock-based design), but it resets insertion order.
        for class in [SizeClass::Small, SizeClass::Medium, SizeClass::Large] {
            let bucket = self.bucket(class);
            let mut guard = bucket.lock().expect("poisoned");
            // Rotate to keep the free list compact.
            guard.free.shrink_to_fit();
        }
    }

    /// Release unused pooled memory back to the system allocator.
    pub fn shrink(&self) {
        for class in [SizeClass::Small, SizeClass::Medium, SizeClass::Large] {
            let bucket = self.bucket(class);
            let mut guard = bucket.lock().expect("poisoned");
            guard.free.clear();
            guard.free.shrink_to_fit();
        }
    }

    /// Point-in-time statistics snapshot.
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.inner.stats.snapshot()
    }

    /// Number of buffers sitting idle in all buckets.
    pub fn free_count(&self) -> usize {
        [SizeClass::Small, SizeClass::Medium, SizeClass::Large]
            .iter()
            .map(|c| self.bucket(*c).lock().expect("poisoned").free.len())
            .sum()
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    fn bucket(&self, class: SizeClass) -> &Mutex<Bucket> {
        match class {
            SizeClass::Small => &self.inner.small,
            SizeClass::Medium => &self.inner.medium,
            SizeClass::Large => &self.inner.large,
        }
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

// ── PoolBuffer ──────────────────────────────────────────────────────────────

/// RAII wrapper around a pooled allocation.
///
/// On [`Drop`], the buffer is returned to the originating [`TensorPool`] for
/// later reuse rather than being freed.
pub struct PoolBuffer {
    buf: Option<RawBuf>,
    /// Logical length (≤ capacity).
    len: usize,
    pool: Arc<PoolInner>,
}

impl PoolBuffer {
    /// View the allocation as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let ptr = self.buf.as_ref().expect("use after return").ptr.as_ptr();
        // SAFETY: ptr is valid for `self.len` bytes and exclusively owned.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// View the allocation as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        let ptr = self.buf.as_ref().expect("use after return").ptr.as_ptr();
        // SAFETY: ptr is valid for `self.len` bytes.
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Pointer to the start of the allocation.
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ref().expect("use after return").ptr.as_ptr()
    }

    /// Length in bytes of the logical allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` — zero-size allocations are rejected at `alloc` time.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Underlying capacity (may be larger than [`len`](Self::len)).
    pub fn capacity(&self) -> usize {
        self.buf.as_ref().expect("use after return").capacity
    }

    /// Interpret the buffer as a mutable `f32` slice.
    ///
    /// # Panics
    ///
    /// Panics if `len` is not a multiple of 4.
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert!(self.len % 4 == 0, "len must be a multiple of 4 for f32 cast");
        let ptr = self.buf.as_ref().expect("use after return").ptr.as_ptr();
        // SAFETY: alignment ≥ 16 satisfies f32 alignment (4); length is a multiple of 4.
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, self.len / 4) }
    }

    /// Interpret the buffer as a `f32` slice.
    ///
    /// # Panics
    ///
    /// Panics if `len` is not a multiple of 4.
    pub fn as_f32(&self) -> &[f32] {
        assert!(self.len % 4 == 0, "len must be a multiple of 4 for f32 cast");
        let ptr = self.buf.as_ref().expect("use after return").ptr.as_ptr();
        // SAFETY: alignment ≥ 16 satisfies f32 alignment (4); length is a multiple of 4.
        unsafe { std::slice::from_raw_parts(ptr as *const f32, self.len / 4) }
    }
}

impl Deref for PoolBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for PoolBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for PoolBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            let class = classify(buf.capacity);
            let bucket = match class {
                SizeClass::Small => &self.pool.small,
                SizeClass::Medium => &self.pool.medium,
                SizeClass::Large => &self.pool.large,
            };
            if let Ok(mut guard) = bucket.lock() {
                guard.free.push(buf);
            }
            // If the mutex is poisoned we simply drop the buffer (deallocate).
            self.pool.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ── Thread-local pool ───────────────────────────────────────────────────────

thread_local! {
    static THREAD_POOL: RefCell<Option<TensorPool>> = const { RefCell::new(None) };
}

/// Initialise (or replace) the calling thread's local [`TensorPool`].
pub fn init_thread_pool(cache_line_aligned: bool) {
    THREAD_POOL.with(|cell| {
        *cell.borrow_mut() = Some(if cache_line_aligned {
            TensorPool::cache_line_aligned()
        } else {
            TensorPool::new()
        });
    });
}

/// Allocate from the calling thread's pool (initialised on first use).
pub fn thread_pool_alloc(size: usize) -> PoolBuffer {
    THREAD_POOL.with(|cell| {
        let mut opt = cell.borrow_mut();
        let pool = opt.get_or_insert_with(TensorPool::new);
        pool.alloc(size)
    })
}

/// Snapshot of the thread-local pool stats, or `None` if uninitialised.
pub fn thread_pool_stats() -> Option<PoolStatsSnapshot> {
    THREAD_POOL.with(|cell| cell.borrow().as_ref().map(|p| p.stats()))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    // -- basic allocation / deallocation --

    #[test]
    fn test_basic_alloc_dealloc() {
        let pool = TensorPool::new();
        let buf = pool.alloc(64);
        assert_eq!(buf.len(), 64);
        drop(buf);
        let s = pool.stats();
        assert_eq!(s.allocations, 1);
        assert_eq!(s.deallocations, 1);
    }

    #[test]
    fn test_alloc_returns_non_null() {
        let pool = TensorPool::new();
        let buf = pool.alloc(128);
        assert!(!buf.as_ptr().is_null());
    }

    #[test]
    fn test_alloc_small_bucket() {
        let pool = TensorPool::new();
        let _buf = pool.alloc(1024); // 1 KB → small
        assert_eq!(pool.stats().allocations, 1);
    }

    #[test]
    fn test_alloc_medium_bucket() {
        let pool = TensorPool::new();
        let _buf = pool.alloc(64 * 1024); // 64 KB → medium
        assert_eq!(pool.stats().allocations, 1);
    }

    #[test]
    fn test_alloc_large_bucket() {
        let pool = TensorPool::new();
        let _buf = pool.alloc(512 * 1024); // 512 KB → large
        assert_eq!(pool.stats().allocations, 1);
    }

    // -- reuse --

    #[test]
    fn test_reuse_after_drop() {
        let pool = TensorPool::new();
        let ptr1 = {
            let buf = pool.alloc(256);
            buf.as_ptr() as usize
        };
        let buf2 = pool.alloc(256);
        let ptr2 = buf2.as_ptr() as usize;
        assert_eq!(ptr1, ptr2, "should reuse the same allocation");
        assert_eq!(pool.stats().reuse_hits, 1);
    }

    #[test]
    fn test_no_reuse_when_size_exceeds_capacity() {
        let pool = TensorPool::new();
        {
            let _buf = pool.alloc(64);
        }
        let buf = pool.alloc(128);
        assert_eq!(buf.len(), 128);
        // The first 64-byte buffer could not satisfy a 128-byte request in the
        // same bucket, so a fresh allocation is made.
        assert_eq!(pool.stats().reuse_hits, 0);
    }

    #[test]
    fn test_reuse_larger_capacity_for_smaller_request() {
        let pool = TensorPool::new();
        let ptr = {
            let buf = pool.alloc(256);
            buf.as_ptr() as usize
        };
        // A smaller request should be served by the recycled 256-byte buffer.
        let buf2 = pool.alloc(128);
        assert_eq!(buf2.as_ptr() as usize, ptr);
        assert_eq!(pool.stats().reuse_hits, 1);
    }

    #[test]
    fn test_reuse_does_not_cross_size_class() {
        let pool = TensorPool::new();
        {
            // Small bucket (4 KB).
            let _buf = pool.alloc(4096);
        }
        // Medium bucket (8 KB) — should NOT find the small buffer.
        let _buf = pool.alloc(8192);
        assert_eq!(pool.stats().reuse_hits, 0);
    }

    // -- alignment --

    #[test]
    fn test_neon_alignment_16() {
        let pool = TensorPool::new();
        for size in [16, 64, 1024, 4096, 65536] {
            let buf = pool.alloc(size);
            assert_eq!(
                buf.as_ptr() as usize % NEON_ALIGN,
                0,
                "buffer of size {size} not 16-byte aligned"
            );
        }
    }

    #[test]
    fn test_cache_line_alignment_64() {
        let pool = TensorPool::cache_line_aligned();
        for size in [64, 256, 4096, 65536] {
            let buf = pool.alloc(size);
            assert_eq!(
                buf.as_ptr() as usize % CACHE_LINE_ALIGN,
                0,
                "buffer of size {size} not 64-byte aligned"
            );
        }
    }

    #[test]
    fn test_alignment_after_reuse() {
        let pool = TensorPool::cache_line_aligned();
        {
            let _buf = pool.alloc(256);
        }
        let buf = pool.alloc(128);
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE_ALIGN, 0);
    }

    // -- statistics --

    #[test]
    fn test_stats_initial() {
        let pool = TensorPool::new();
        let s = pool.stats();
        assert_eq!(s.allocations, 0);
        assert_eq!(s.deallocations, 0);
        assert_eq!(s.reuse_hits, 0);
        assert_eq!(s.total_bytes_allocated, 0);
    }

    #[test]
    fn test_stats_after_multiple_allocs() {
        let pool = TensorPool::new();
        let _a = pool.alloc(100);
        let _b = pool.alloc(200);
        let s = pool.stats();
        assert_eq!(s.allocations, 2);
        assert_eq!(s.total_bytes_allocated, 300);
    }

    #[test]
    fn test_stats_reuse_counted() {
        let pool = TensorPool::new();
        {
            let _buf = pool.alloc(64);
        }
        let _buf = pool.alloc(64);
        let s = pool.stats();
        assert_eq!(s.allocations, 2);
        assert_eq!(s.reuse_hits, 1);
        assert_eq!(s.total_bytes_allocated, 64); // Only the first alloc added bytes.
    }

    #[test]
    fn test_stats_dealloc_count() {
        let pool = TensorPool::new();
        {
            let _a = pool.alloc(32);
            let _b = pool.alloc(64);
        }
        assert_eq!(pool.stats().deallocations, 2);
    }

    // -- f32 view --

    #[test]
    fn test_as_f32_mut_write_read() {
        let pool = TensorPool::new();
        let mut buf = pool.alloc(16); // 4 × f32
        let slice = buf.as_f32_mut();
        slice[0] = 1.0;
        slice[1] = 2.0;
        slice[2] = 3.0;
        slice[3] = 4.0;
        assert_eq!(buf.as_f32(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn test_as_f32_panics_on_misaligned_len() {
        let pool = TensorPool::new();
        let buf = pool.alloc(17);
        let _ = buf.as_f32();
    }

    // -- clear / shrink --

    #[test]
    fn test_clear_keeps_free_buffers() {
        let pool = TensorPool::new();
        {
            let _buf = pool.alloc(128);
        }
        assert_eq!(pool.free_count(), 1);
        pool.clear();
        // clear does not release memory; buffers stay in the pool.
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_shrink_releases_free_buffers() {
        let pool = TensorPool::new();
        {
            let _buf = pool.alloc(128);
        }
        assert_eq!(pool.free_count(), 1);
        pool.shrink();
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_shrink_does_not_affect_outstanding() {
        let pool = TensorPool::new();
        let buf = pool.alloc(64);
        pool.shrink();
        // The outstanding buffer is still valid.
        assert_eq!(buf.len(), 64);
    }

    // -- drop behaviour --

    #[test]
    fn test_buffer_returns_to_pool_on_drop() {
        let pool = TensorPool::new();
        {
            let _buf = pool.alloc(256);
        }
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_multiple_drops_return_all() {
        let pool = TensorPool::new();
        {
            let _a = pool.alloc(64);
            let _b = pool.alloc(128);
            let _c = pool.alloc(256);
        }
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn test_pool_drop_frees_all_memory() {
        // Primarily ensures no leaks (Miri / valgrind would catch this).
        let pool = TensorPool::new();
        {
            let _a = pool.alloc(1024);
            let _b = pool.alloc(65536);
            let _c = pool.alloc(512 * 1024);
        }
        drop(pool);
    }

    // -- PoolBuffer Deref --

    #[test]
    fn test_deref_write_read() {
        let pool = TensorPool::new();
        let mut buf = pool.alloc(8);
        buf[0] = 0xAA;
        buf[7] = 0xBB;
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[7], 0xBB);
    }

    #[test]
    fn test_is_empty_always_false() {
        let pool = TensorPool::new();
        let buf = pool.alloc(1);
        assert!(!buf.is_empty());
    }

    // -- zero-size rejection --

    #[test]
    #[should_panic(expected = "cannot allocate zero bytes")]
    fn test_alloc_zero_panics() {
        let pool = TensorPool::new();
        let _buf = pool.alloc(0);
    }

    // -- concurrent access --

    #[test]
    fn test_concurrent_alloc_dealloc() {
        let pool = Arc::new(TensorPool::new());
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&pool);
                let b = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    b.wait();
                    for _ in 0..50 {
                        let mut buf = p.alloc(256);
                        buf[0] = 42;
                        drop(buf);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(pool.stats().allocations, 200);
        assert_eq!(pool.stats().deallocations, 200);
    }

    #[test]
    fn test_concurrent_different_size_classes() {
        let pool = Arc::new(TensorPool::new());
        let sizes = [512, 32 * 1024, 512 * 1024]; // small, medium, large
        let handles: Vec<_> = sizes
            .iter()
            .map(|&sz| {
                let p = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        let _buf = p.alloc(sz);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(pool.stats().allocations, 60);
    }

    // -- thread-local pool --

    #[test]
    fn test_thread_pool_alloc() {
        init_thread_pool(false);
        let buf = thread_pool_alloc(128);
        assert_eq!(buf.len(), 128);
        assert_eq!(buf.as_ptr() as usize % NEON_ALIGN, 0);
    }

    #[test]
    fn test_thread_pool_stats() {
        init_thread_pool(false);
        let _ = thread_pool_alloc(64);
        let s = thread_pool_stats().expect("pool should be initialised");
        assert_eq!(s.allocations, 1);
    }

    #[test]
    fn test_thread_pool_auto_init() {
        // Do NOT call init_thread_pool — it should auto-initialise.
        let buf = thread_pool_alloc(32);
        assert_eq!(buf.len(), 32);
    }

    // -- size class boundaries --

    #[test]
    fn test_size_class_boundary_small() {
        assert_eq!(classify(SMALL_THRESHOLD), SizeClass::Small);
        assert_eq!(classify(SMALL_THRESHOLD + 1), SizeClass::Medium);
    }

    #[test]
    fn test_size_class_boundary_medium() {
        assert_eq!(classify(MEDIUM_THRESHOLD), SizeClass::Medium);
        assert_eq!(classify(MEDIUM_THRESHOLD + 1), SizeClass::Large);
    }

    #[test]
    fn test_size_class_one_byte() {
        assert_eq!(classify(1), SizeClass::Small);
    }
}
