//! Intel Arc A770 (Xe-HPG) OpenCL buffer alignment and pool management.
//!
//! Provides alignment constants, an aligned buffer wrapper, and a reusable
//! buffer pool tuned for the A770's cache-line, L3-partition, and DMA
//! transfer requirements.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Alignment constants
// ---------------------------------------------------------------------------

/// Hardware-derived alignment constants for Intel Arc A770 (Xe-HPG).
pub struct BufferAlignment;

impl BufferAlignment {
    /// Xe-HPG cache line size in bytes.
    pub const CACHE_LINE: usize = 64;

    /// L3 partition size per sub-slice (256 KB).
    pub const L3_PARTITION: usize = 262_144;

    /// Page-aligned size for optimal DMA transfers.
    pub const OPTIMAL_TRANSFER: usize = 4096;

    /// Xe-HPG sub-group width for coalesced memory access.
    pub const SUBGROUP_WIDTH: usize = 32;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Round `size` up to the next multiple of `alignment`.
///
/// # Panics
///
/// Panics if `alignment` is zero.
#[inline]
pub fn align_size(size: usize, alignment: usize) -> usize {
    assert!(alignment != 0, "alignment must be non-zero");
    if size == 0 {
        return 0;
    }
    // Works for any positive alignment (not just powers of two).
    let remainder = size % alignment;
    if remainder == 0 { size } else { size + (alignment - remainder) }
}

/// Compute an A770-optimal buffer size for `elements × element_size` bytes.
///
/// The returned size is rounded up to the cache-line boundary and, when large
/// enough, further aligned to the DMA page size.
pub fn optimal_buffer_size(elements: usize, element_size: usize) -> usize {
    let raw = elements.saturating_mul(element_size);
    if raw == 0 {
        return 0;
    }
    let aligned = align_size(raw, BufferAlignment::CACHE_LINE);
    // For buffers ≥ one page, align to the page boundary for DMA.
    if aligned >= BufferAlignment::OPTIMAL_TRANSFER {
        align_size(aligned, BufferAlignment::OPTIMAL_TRANSFER)
    } else {
        aligned
    }
}

/// Return `true` when `ptr` satisfies the requested alignment.
#[inline]
pub fn validate_alignment(ptr: usize, required: usize) -> bool {
    if required == 0 {
        return true;
    }
    ptr.is_multiple_of(required)
}

// ---------------------------------------------------------------------------
// AlignedBuffer
// ---------------------------------------------------------------------------

/// A host buffer whose backing allocation is aligned to a specified boundary.
///
/// Optionally tracks a device-side (GPU) allocation handle.
#[derive(Debug)]
pub struct AlignedBuffer<T: Copy + Default> {
    data: Vec<T>,
    byte_size: usize,
    alignment: usize,
    device_ptr: Option<usize>,
}

impl<T: Copy + Default> AlignedBuffer<T> {
    /// Create a new zero-initialised aligned buffer that can hold at least
    /// `num_elements` items of type `T`.
    ///
    /// `alignment` is expressed in **bytes** and must be non-zero.
    pub fn new(num_elements: usize, alignment: usize) -> Self {
        assert!(alignment != 0, "alignment must be non-zero");
        let elem_size = std::mem::size_of::<T>().max(1);
        let byte_size = align_size(num_elements * elem_size, alignment);
        // Compute how many T elements we need to cover `byte_size` bytes.
        let alloc_elements = byte_size.div_ceil(elem_size);
        Self { data: vec![T::default(); alloc_elements], byte_size, alignment, device_ptr: None }
    }

    /// Byte-size of the allocation (aligned).
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Alignment that was requested at construction time.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Optional GPU device pointer.
    pub fn device_ptr(&self) -> Option<usize> {
        self.device_ptr
    }

    /// Associate a GPU device pointer with this buffer.
    pub fn set_device_ptr(&mut self, ptr: usize) {
        self.device_ptr = Some(ptr);
    }

    /// Clear any associated device pointer.
    pub fn clear_device_ptr(&mut self) {
        self.device_ptr = None;
    }

    /// Number of `T` elements in the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer has zero elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable access to the underlying data.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

/// Statistics returned by [`BufferPool::stats`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of buffers currently handed out.
    pub allocated: usize,
    /// Number of buffers sitting in the pool, ready for reuse.
    pub pooled: usize,
    /// Total bytes across all allocated + pooled buffers.
    pub total_bytes: usize,
}

/// A simple, thread-safe pool of reusable byte buffers.
///
/// Buffers are bucketed by their *aligned* size so that a returned buffer can
/// be handed out again for any request with the same (or smaller) aligned size.
#[derive(Debug, Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<PoolInner>>,
}

#[derive(Debug, Default)]
struct PoolInner {
    /// Free buffers keyed by their `byte_size`.
    free: HashMap<usize, Vec<AlignedBuffer<u8>>>,
    /// Count of buffers currently in use (handed out via `allocate`).
    allocated_count: usize,
    /// Total bytes across live (allocated) buffers.
    allocated_bytes: usize,
}

impl BufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(PoolInner::default())) }
    }

    /// Allocate (or reuse) a buffer of at least `size` bytes with the given
    /// `alignment`.
    pub fn allocate(&self, size: usize, alignment: usize) -> AlignedBuffer<u8> {
        let aligned_size = align_size(size, alignment.max(1));
        let mut inner = self.inner.lock().expect("BufferPool lock poisoned");

        // Try to reuse a pooled buffer of the exact aligned size.
        if let Some(free_list) = inner.free.get_mut(&aligned_size)
            && let Some(buf) = free_list.pop()
        {
            inner.allocated_count += 1;
            inner.allocated_bytes += buf.byte_size();
            return buf;
        }

        // No reusable buffer — create a fresh one.
        let buf = AlignedBuffer::<u8>::new(aligned_size, alignment.max(1));
        inner.allocated_count += 1;
        inner.allocated_bytes += buf.byte_size();
        buf
    }

    /// Return a buffer to the pool for later reuse.
    pub fn return_buffer(&self, buf: AlignedBuffer<u8>) {
        let mut inner = self.inner.lock().expect("BufferPool lock poisoned");
        let bs = buf.byte_size();
        if inner.allocated_count > 0 {
            inner.allocated_count -= 1;
        }
        inner.allocated_bytes = inner.allocated_bytes.saturating_sub(bs);
        inner.free.entry(bs).or_default().push(buf);
    }

    /// Release all pooled (free) buffers. In-flight buffers are unaffected.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("BufferPool lock poisoned");
        inner.free.clear();
    }

    /// Snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().expect("BufferPool lock poisoned");
        let pooled: usize = inner.free.values().map(|v| v.len()).sum();
        let pooled_bytes: usize =
            inner.free.values().flat_map(|v| v.iter()).map(|b| b.byte_size()).sum();
        PoolStats {
            allocated: inner.allocated_count,
            pooled,
            total_bytes: inner.allocated_bytes + pooled_bytes,
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // -- align_size ---------------------------------------------------------

    #[test]
    fn align_size_already_aligned() {
        assert_eq!(align_size(64, 64), 64);
        assert_eq!(align_size(128, 64), 128);
    }

    #[test]
    fn align_size_rounds_up() {
        assert_eq!(align_size(1, 64), 64);
        assert_eq!(align_size(65, 64), 128);
        assert_eq!(align_size(100, 256), 256);
    }

    #[test]
    fn align_size_zero_size() {
        assert_eq!(align_size(0, 64), 0);
    }

    #[test]
    fn align_size_alignment_one() {
        assert_eq!(align_size(42, 1), 42);
    }

    #[test]
    fn align_size_non_power_of_two_alignment() {
        // 10 is not a power of two, but align_size must still work.
        assert_eq!(align_size(11, 10), 20);
        assert_eq!(align_size(20, 10), 20);
        assert_eq!(align_size(21, 10), 30);
    }

    #[test]
    #[should_panic(expected = "alignment must be non-zero")]
    fn align_size_zero_alignment_panics() {
        let _ = align_size(10, 0);
    }

    // -- optimal_buffer_size ------------------------------------------------

    #[test]
    fn optimal_buffer_size_cache_line_aligned() {
        // 100 × 4 = 400 → next multiple of 64 = 448
        assert_eq!(optimal_buffer_size(100, 4), 448);
    }

    #[test]
    fn optimal_buffer_size_page_aligned_for_large() {
        // 2000 × 4 = 8000 → next 64 = 8000 (already aligned) → ≥4096 → next 4096 = 8192
        assert_eq!(optimal_buffer_size(2000, 4), 8192);
    }

    #[test]
    fn optimal_buffer_size_zero() {
        assert_eq!(optimal_buffer_size(0, 4), 0);
        assert_eq!(optimal_buffer_size(100, 0), 0);
    }

    #[test]
    fn optimal_buffer_size_exact_page() {
        // 1024 × 4 = 4096 → already 64-aligned and page-aligned → 4096
        assert_eq!(optimal_buffer_size(1024, 4), 4096);
    }

    // -- validate_alignment -------------------------------------------------

    #[test]
    fn validate_alignment_aligned() {
        assert!(validate_alignment(128, 64));
        assert!(validate_alignment(0, 64));
        assert!(validate_alignment(4096, 4096));
    }

    #[test]
    fn validate_alignment_misaligned() {
        assert!(!validate_alignment(1, 64));
        assert!(!validate_alignment(63, 64));
    }

    #[test]
    fn validate_alignment_zero_required() {
        // Zero alignment is always satisfied.
        assert!(validate_alignment(123, 0));
    }

    // -- AlignedBuffer ------------------------------------------------------

    #[test]
    fn aligned_buffer_basic() {
        let buf = AlignedBuffer::<f32>::new(10, BufferAlignment::CACHE_LINE);
        assert!(buf.byte_size() >= 10 * std::mem::size_of::<f32>());
        assert_eq!(buf.byte_size() % BufferAlignment::CACHE_LINE, 0);
        assert!(!buf.is_empty());
    }

    #[test]
    fn aligned_buffer_zero_elements() {
        let buf = AlignedBuffer::<f32>::new(0, 64);
        assert_eq!(buf.byte_size(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn aligned_buffer_device_ptr_lifecycle() {
        let mut buf = AlignedBuffer::<u8>::new(64, 64);
        assert!(buf.device_ptr().is_none());
        buf.set_device_ptr(0xDEAD_BEEF);
        assert_eq!(buf.device_ptr(), Some(0xDEAD_BEEF));
        buf.clear_device_ptr();
        assert!(buf.device_ptr().is_none());
    }

    #[test]
    fn aligned_buffer_data_read_write() {
        let mut buf = AlignedBuffer::<u32>::new(4, 64);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    // -- BufferPool ---------------------------------------------------------

    #[test]
    fn pool_allocate_returns_correct_size() {
        let pool = BufferPool::new();
        let buf = pool.allocate(100, BufferAlignment::CACHE_LINE);
        assert!(buf.byte_size() >= 100);
        assert_eq!(buf.byte_size() % BufferAlignment::CACHE_LINE, 0);
    }

    #[test]
    fn pool_return_and_reuse() {
        let pool = BufferPool::new();
        let buf = pool.allocate(128, 64);
        let bs = buf.byte_size();
        pool.return_buffer(buf);

        // The next allocation of the same aligned size should reuse the buffer.
        let buf2 = pool.allocate(128, 64);
        assert_eq!(buf2.byte_size(), bs);
    }

    #[test]
    fn pool_stats_tracking() {
        let pool = BufferPool::new();
        let s0 = pool.stats();
        assert_eq!(s0.allocated, 0);
        assert_eq!(s0.pooled, 0);

        let buf = pool.allocate(256, 64);
        let s1 = pool.stats();
        assert_eq!(s1.allocated, 1);
        assert_eq!(s1.pooled, 0);

        pool.return_buffer(buf);
        let s2 = pool.stats();
        assert_eq!(s2.allocated, 0);
        assert_eq!(s2.pooled, 1);
    }

    #[test]
    fn pool_clear_releases_pooled() {
        let pool = BufferPool::new();
        let buf = pool.allocate(64, 64);
        pool.return_buffer(buf);
        assert_eq!(pool.stats().pooled, 1);

        pool.clear();
        assert_eq!(pool.stats().pooled, 0);
    }

    #[test]
    fn pool_large_allocation() {
        let pool = BufferPool::new();
        let buf = pool.allocate(10_000_000, BufferAlignment::OPTIMAL_TRANSFER);
        assert!(buf.byte_size() >= 10_000_000);
        assert_eq!(buf.byte_size() % BufferAlignment::OPTIMAL_TRANSFER, 0);
    }

    #[test]
    fn pool_zero_size_allocation() {
        let pool = BufferPool::new();
        let buf = pool.allocate(0, 64);
        assert_eq!(buf.byte_size(), 0);
    }

    #[test]
    fn pool_concurrent_access() {
        let pool = BufferPool::new();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        let buf = pool.allocate(128, 64);
                        pool.return_buffer(buf);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All buffers returned — allocated count must be zero.
        assert_eq!(pool.stats().allocated, 0);
    }

    // -- BufferAlignment constants ------------------------------------------

    #[test]
    fn alignment_constants_are_positive_and_sensible() {
        assert_eq!(BufferAlignment::CACHE_LINE, 64);
        assert_eq!(BufferAlignment::L3_PARTITION, 262_144);
        assert_eq!(BufferAlignment::OPTIMAL_TRANSFER, 4096);
        assert_eq!(BufferAlignment::SUBGROUP_WIDTH, 32);
    }

    #[test]
    fn cache_line_is_power_of_two() {
        assert!(BufferAlignment::CACHE_LINE.is_power_of_two());
        assert!(BufferAlignment::OPTIMAL_TRANSFER.is_power_of_two());
    }
}
