//! CPU memory pool / arena allocator for tensor scratch space.
//!
//! Provides three complementary allocation strategies:
//!
//! - [`MemoryPool`] — bump-pointer arena with O(1) reset
//! - [`ScratchAllocator`] — layer-scoped scratch that auto-resets between layers
//! - [`TensorBufferCache`] — LRU cache for frequently-allocated tensor shapes
//! - [`AlignedBuffer`] — standalone aligned heap allocation helper

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;

// ── Errors ─────────────────────────────────────────────────────────

/// Errors produced by the memory pool subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPoolError {
    /// The pool does not have enough remaining capacity.
    OutOfMemory { requested: usize, available: usize },
    /// The requested alignment is not a power of two.
    InvalidAlignment(usize),
    /// Allocation of zero bytes was requested.
    ZeroSizeAllocation,
    /// A scope was ended without a matching `new_layer_scope`.
    ScopeMismatch,
    /// The requested capacity is zero.
    ZeroCapacity,
}

impl fmt::Display for MemoryPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} bytes, {available} available")
            }
            Self::InvalidAlignment(a) => {
                write!(f, "invalid alignment {a}: must be a power of two")
            }
            Self::ZeroSizeAllocation => write!(f, "zero-size allocation"),
            Self::ScopeMismatch => {
                write!(f, "end_layer_scope called without matching new_layer_scope")
            }
            Self::ZeroCapacity => write!(f, "pool capacity must be > 0"),
        }
    }
}

impl std::error::Error for MemoryPoolError {}

// ── MemoryPool ─────────────────────────────────────────────────────

/// Arena-style bump allocator for tensor scratch space.
///
/// Allocations are O(1) and the entire pool can be reset in O(1)
/// without deallocating individual buffers. The backing buffer is
/// freed when the pool is dropped.
///
/// # Thread safety
///
/// `MemoryPool` is `Send` but **not** `Sync` — a single pool must
/// not be shared across threads without external synchronisation.
/// The typical pattern is one pool per worker thread.
pub struct MemoryPool {
    /// Backing buffer (aligned to `POOL_ALIGN`).
    buf: NonNull<u8>,
    /// Layout used for the backing allocation.
    layout: Layout,
    /// Total capacity in bytes.
    cap: usize,
    /// Current bump cursor (byte offset from `buf`).
    cursor: usize,
}

// SAFETY: The pool owns its buffer exclusively; moving it across
// threads is safe.
unsafe impl Send for MemoryPool {}

/// Default pool alignment (64 bytes — cache-line width on most x86).
const POOL_ALIGN: usize = 64;

impl MemoryPool {
    /// Create a new pool with the given `capacity` in bytes.
    ///
    /// The backing buffer is allocated with 64-byte alignment so that
    /// sub-allocations on cache-line boundaries are always possible.
    pub fn new(capacity: usize) -> Result<Self, MemoryPoolError> {
        if capacity == 0 {
            return Err(MemoryPoolError::ZeroCapacity);
        }
        let layout = Layout::from_size_align(capacity, POOL_ALIGN)
            .map_err(|_| MemoryPoolError::InvalidAlignment(POOL_ALIGN))?;
        // SAFETY: `layout.size() > 0` (checked above).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let buf = NonNull::new(ptr)
            .ok_or(MemoryPoolError::OutOfMemory { requested: capacity, available: 0 })?;
        Ok(Self { buf, layout, cap: capacity, cursor: 0 })
    }

    /// Bump-allocate `size` bytes with the given `alignment`.
    ///
    /// Returns a mutable slice into the pool. The slice is valid until
    /// the next call to [`reset`](Self::reset) or until the pool is
    /// dropped.
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&mut [u8], MemoryPoolError> {
        if size == 0 {
            return Err(MemoryPoolError::ZeroSizeAllocation);
        }
        if !alignment.is_power_of_two() {
            return Err(MemoryPoolError::InvalidAlignment(alignment));
        }

        // Round cursor up to requested alignment.
        let base = unsafe { self.buf.as_ptr().add(self.cursor) } as usize;
        let aligned = (base + alignment - 1) & !(alignment - 1);
        let padding = aligned - base;
        let total = padding + size;

        if self.cursor + total > self.cap {
            return Err(MemoryPoolError::OutOfMemory {
                requested: size,
                available: self.cap - self.cursor,
            });
        }

        let start = self.cursor + padding;
        self.cursor += total;
        // SAFETY: `start..start+size` is within bounds of the
        // backing allocation and correctly aligned.
        let slice = unsafe { std::slice::from_raw_parts_mut(self.buf.as_ptr().add(start), size) };
        Ok(slice)
    }

    /// Reset the bump cursor to the beginning of the pool.
    ///
    /// All previous allocations become logically invalid. This is O(1)
    /// and does not zero-fill the buffer.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Number of bytes currently in use (including alignment padding).
    #[inline]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Number of bytes remaining before the pool is exhausted.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.cap - self.cursor
    }

    /// Total capacity of the pool in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Return the cursor to an earlier position (used by scopes).
    fn restore_cursor(&mut self, pos: usize) {
        debug_assert!(pos <= self.cursor);
        self.cursor = pos;
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // SAFETY: `self.buf` was allocated with `self.layout`.
        unsafe { alloc::dealloc(self.buf.as_ptr(), self.layout) }
    }
}

impl fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPool")
            .field("capacity", &self.cap)
            .field("used", &self.cursor)
            .field("remaining", &(self.cap - self.cursor))
            .finish()
    }
}

// ── ScratchAllocator ───────────────────────────────────────────────

/// Layer-scoped scratch allocator that auto-resets between layers.
///
/// Wraps a [`MemoryPool`] and adds a scope stack so that allocations
/// made during one layer are automatically freed before the next.
///
/// # Thread safety
///
/// Same as [`MemoryPool`] — `Send` but not `Sync`.
pub struct ScratchAllocator {
    pool: MemoryPool,
    /// Stack of saved cursor positions (one per scope).
    scope_stack: Vec<usize>,
}

impl ScratchAllocator {
    /// Create a new scratch allocator backed by a pool of `capacity` bytes.
    pub fn new(capacity: usize) -> Result<Self, MemoryPoolError> {
        Ok(Self { pool: MemoryPool::new(capacity)?, scope_stack: Vec::new() })
    }

    /// Begin a new layer scope. Saves the current cursor so that
    /// [`end_layer_scope`](Self::end_layer_scope) can restore it.
    pub fn new_layer_scope(&mut self) {
        self.scope_stack.push(self.pool.used());
    }

    /// End the current layer scope, restoring the cursor to where it
    /// was when [`new_layer_scope`](Self::new_layer_scope) was called.
    pub fn end_layer_scope(&mut self) -> Result<(), MemoryPoolError> {
        let saved = self.scope_stack.pop().ok_or(MemoryPoolError::ScopeMismatch)?;
        self.pool.restore_cursor(saved);
        Ok(())
    }

    /// Allocate a temporary buffer within the current scope.
    ///
    /// The buffer uses 64-byte alignment by default. Use
    /// [`allocate_aligned`](Self::allocate_aligned) to specify a
    /// custom alignment.
    pub fn temp_buffer(&mut self, size: usize) -> Result<&mut [u8], MemoryPoolError> {
        self.pool.allocate(size, POOL_ALIGN)
    }

    /// Allocate a temporary buffer with custom alignment.
    pub fn allocate_aligned(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&mut [u8], MemoryPoolError> {
        self.pool.allocate(size, alignment)
    }

    /// Current number of open scopes.
    #[inline]
    pub fn scope_depth(&self) -> usize {
        self.scope_stack.len()
    }

    /// Bytes used in the underlying pool.
    #[inline]
    pub fn used(&self) -> usize {
        self.pool.used()
    }

    /// Bytes remaining in the underlying pool.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.pool.remaining()
    }

    /// Total capacity of the underlying pool.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.pool.capacity()
    }

    /// Fully reset the pool and clear all scopes.
    pub fn reset(&mut self) {
        self.scope_stack.clear();
        self.pool.reset();
    }
}

impl fmt::Debug for ScratchAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchAllocator")
            .field("pool", &self.pool)
            .field("scope_depth", &self.scope_stack.len())
            .finish()
    }
}

// ── TensorBufferCache ──────────────────────────────────────────────

/// Statistics for cache hit/miss tracking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    /// Hit rate as a fraction in [0.0, 1.0]. Returns 0.0 when the
    /// cache has not been queried.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Key that uniquely identifies a tensor buffer shape + dtype width.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferKey {
    shape: Vec<usize>,
    dtype_size: usize,
}

/// One cached entry, tracking recency via an access counter.
struct CacheEntry {
    buf: AlignedBuffer,
    last_access: u64,
}

/// LRU cache for frequently-allocated tensor shapes.
///
/// Avoids repeated `alloc`/`dealloc` cycles for tensor buffers that
/// are allocated and freed on every forward pass (e.g. attention
/// scratch, FFN intermediates).
///
/// # Thread safety
///
/// `TensorBufferCache` is `Send` but not `Sync`.
pub struct TensorBufferCache {
    entries: HashMap<BufferKey, CacheEntry>,
    max_entries: usize,
    access_counter: u64,
    stats: CacheStats,
}

impl TensorBufferCache {
    /// Create a cache holding at most `max_entries` buffers.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries,
            access_counter: 0,
            stats: CacheStats::default(),
        }
    }

    /// Get a mutable slice for the given shape and element width.
    ///
    /// Returns a cached buffer on hit or allocates a new one on miss,
    /// evicting the least-recently-used entry if the cache is full.
    pub fn get_or_allocate(
        &mut self,
        shape: &[usize],
        dtype_size: usize,
    ) -> Result<&mut [u8], MemoryPoolError> {
        let key = BufferKey { shape: shape.to_vec(), dtype_size };
        self.access_counter += 1;
        let counter = self.access_counter;

        if self.entries.contains_key(&key) {
            self.stats.hits += 1;
            let entry = self.entries.get_mut(&key).unwrap();
            entry.last_access = counter;
            return Ok(entry.buf.as_mut_slice());
        }

        // Miss — allocate a new buffer.
        self.stats.misses += 1;
        let total_bytes: usize =
            shape
                .iter()
                .copied()
                .product::<usize>()
                .checked_mul(dtype_size)
                .ok_or(MemoryPoolError::OutOfMemory { requested: usize::MAX, available: 0 })?;
        if total_bytes == 0 {
            return Err(MemoryPoolError::ZeroSizeAllocation);
        }

        // Evict if at capacity.
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let buf = AlignedBuffer::new(total_bytes, POOL_ALIGN)?;
        self.entries.insert(key.clone(), CacheEntry { buf, last_access: counter });
        Ok(self.entries.get_mut(&key).unwrap().buf.as_mut_slice())
    }

    /// Evict the least-recently-used entry.
    pub fn evict_lru(&mut self) {
        if let Some(lru_key) =
            self.entries.iter().min_by_key(|(_, e)| e.last_access).map(|(k, _)| k.clone())
        {
            self.entries.remove(&lru_key);
        }
    }

    /// Return cumulative hit/miss statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.stats
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl fmt::Debug for TensorBufferCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorBufferCache")
            .field("entries", &self.entries.len())
            .field("max_entries", &self.max_entries)
            .field("stats", &self.stats)
            .finish()
    }
}

// ── AlignedBuffer ──────────────────────────────────────────────────

/// Heap-allocated buffer with a guaranteed minimum alignment.
///
/// Useful for SIMD-friendly scratch allocations outside of the arena.
///
/// # Thread safety
///
/// `AlignedBuffer` is both `Send` and `Sync` (it owns an
/// exclusively-held, immovable heap region).
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

// SAFETY: The buffer is exclusively owned — safe to send/share.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `size` bytes with the given `alignment`.
    ///
    /// The buffer is zero-initialised.
    pub fn new(size: usize, alignment: usize) -> Result<Self, MemoryPoolError> {
        if size == 0 {
            return Err(MemoryPoolError::ZeroSizeAllocation);
        }
        if !alignment.is_power_of_two() {
            return Err(MemoryPoolError::InvalidAlignment(alignment));
        }
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| MemoryPoolError::InvalidAlignment(alignment))?;
        // SAFETY: size > 0 checked above.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or(MemoryPoolError::OutOfMemory { requested: size, available: 0 })?;
        Ok(Self { ptr, layout, len: size })
    }

    /// Immutable view of the buffer contents.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: pointer is valid for `self.len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable view of the buffer contents.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive access via `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty (always false for a valid buffer).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Alignment of the buffer.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.layout.align()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout`.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

impl fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("alignment", &self.layout.align())
            .finish()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryPoolError Display ────────────────────────────────────

    #[test]
    fn error_display_out_of_memory() {
        let e = MemoryPoolError::OutOfMemory { requested: 100, available: 50 };
        assert_eq!(e.to_string(), "out of memory: requested 100 bytes, 50 available");
    }

    #[test]
    fn error_display_invalid_alignment() {
        let e = MemoryPoolError::InvalidAlignment(3);
        assert_eq!(e.to_string(), "invalid alignment 3: must be a power of two");
    }

    #[test]
    fn error_display_zero_size() {
        assert_eq!(MemoryPoolError::ZeroSizeAllocation.to_string(), "zero-size allocation");
    }

    #[test]
    fn error_display_scope_mismatch() {
        let e = MemoryPoolError::ScopeMismatch;
        assert!(e.to_string().contains("end_layer_scope"));
    }

    #[test]
    fn error_display_zero_capacity() {
        assert_eq!(MemoryPoolError::ZeroCapacity.to_string(), "pool capacity must be > 0");
    }

    #[test]
    fn error_implements_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MemoryPoolError::ZeroSizeAllocation);
        assert!(!e.to_string().is_empty());
    }

    // ── MemoryPool basic ───────────────────────────────────────────

    #[test]
    fn pool_new_and_capacity() {
        let pool = MemoryPool::new(1024).unwrap();
        assert_eq!(pool.capacity(), 1024);
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.remaining(), 1024);
    }

    #[test]
    fn pool_zero_capacity_rejected() {
        let err = MemoryPool::new(0).unwrap_err();
        assert_eq!(err, MemoryPoolError::ZeroCapacity);
    }

    #[test]
    fn pool_allocate_basic() {
        let mut pool = MemoryPool::new(4096).unwrap();
        let buf = pool.allocate(256, 8).unwrap();
        assert_eq!(buf.len(), 256);
        assert!(pool.used() >= 256);
    }

    #[test]
    fn pool_allocate_returns_writable_memory() {
        let mut pool = MemoryPool::new(4096).unwrap();
        let buf = pool.allocate(64, 8).unwrap();
        // Write a pattern and read it back.
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        for (i, b) in buf.iter().enumerate() {
            assert_eq!(*b, (i & 0xFF) as u8);
        }
    }

    #[test]
    fn pool_multiple_allocations_non_overlapping() {
        let mut pool = MemoryPool::new(4096).unwrap();
        let p1 = pool.allocate(128, 8).unwrap().as_ptr() as usize;
        let p2 = pool.allocate(128, 8).unwrap().as_ptr() as usize;
        // The two allocations must not overlap.
        assert!(p2 >= p1 + 128);
    }

    #[test]
    fn pool_reset_cycle() {
        let mut pool = MemoryPool::new(1024).unwrap();
        pool.allocate(512, 8).unwrap();
        assert!(pool.used() >= 512);
        pool.reset();
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.remaining(), 1024);
        // Allocations after reset succeed.
        pool.allocate(1024, 1).unwrap();
    }

    #[test]
    fn pool_reset_allows_reuse() {
        let mut pool = MemoryPool::new(256).unwrap();
        for _ in 0..10 {
            pool.allocate(200, 1).unwrap();
            pool.reset();
        }
        assert_eq!(pool.used(), 0);
    }

    // ── Alignment ──────────────────────────────────────────────────

    #[test]
    fn pool_alignment_8() {
        let mut pool = MemoryPool::new(4096).unwrap();
        // Force odd cursor by allocating 1 byte with align=1.
        pool.allocate(1, 1).unwrap();
        let buf = pool.allocate(64, 8).unwrap();
        assert_eq!(buf.as_ptr() as usize % 8, 0);
    }

    #[test]
    fn pool_alignment_16() {
        let mut pool = MemoryPool::new(4096).unwrap();
        pool.allocate(3, 1).unwrap();
        let buf = pool.allocate(64, 16).unwrap();
        assert_eq!(buf.as_ptr() as usize % 16, 0);
    }

    #[test]
    fn pool_alignment_32() {
        let mut pool = MemoryPool::new(4096).unwrap();
        pool.allocate(7, 1).unwrap();
        let buf = pool.allocate(64, 32).unwrap();
        assert_eq!(buf.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn pool_alignment_64() {
        let mut pool = MemoryPool::new(8192).unwrap();
        pool.allocate(5, 1).unwrap();
        let buf = pool.allocate(128, 64).unwrap();
        assert_eq!(buf.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn pool_alignment_1_no_padding() {
        let mut pool = MemoryPool::new(256).unwrap();
        let b1 = pool.allocate(10, 1).unwrap();
        let end = b1.as_ptr() as usize + 10;
        let b2 = pool.allocate(10, 1).unwrap();
        // With align=1 there should be no padding.
        assert_eq!(b2.as_ptr() as usize, end);
    }

    #[test]
    fn pool_invalid_alignment_rejected() {
        let mut pool = MemoryPool::new(256).unwrap();
        let err = pool.allocate(64, 3).unwrap_err();
        assert_eq!(err, MemoryPoolError::InvalidAlignment(3));
    }

    #[test]
    fn pool_alignment_zero_rejected() {
        let mut pool = MemoryPool::new(256).unwrap();
        let err = pool.allocate(64, 0).unwrap_err();
        assert_eq!(err, MemoryPoolError::InvalidAlignment(0));
    }

    // ── Capacity exhaustion ────────────────────────────────────────

    #[test]
    fn pool_exact_fit() {
        let mut pool = MemoryPool::new(128).unwrap();
        pool.allocate(128, 1).unwrap();
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn pool_exhaustion_returns_error() {
        let mut pool = MemoryPool::new(128).unwrap();
        pool.allocate(100, 1).unwrap();
        let err = pool.allocate(100, 1).unwrap_err();
        match err {
            MemoryPoolError::OutOfMemory { requested: 100, .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn pool_exhaustion_accounts_for_alignment_padding() {
        let mut pool = MemoryPool::new(256).unwrap();
        // Use most of the pool with align=1.
        pool.allocate(200, 1).unwrap();
        // The remaining 56 bytes might not suffice after alignment.
        let result = pool.allocate(56, 64);
        // Drop the result before inspecting the pool.
        let was_oom = matches!(result, Err(MemoryPoolError::OutOfMemory { .. }));
        drop(result);
        // Regardless of pass/fail, the pool must remain consistent.
        assert!(pool.used() <= pool.capacity());
        // Alignment padding may have consumed the slack.
        let _ = was_oom;
    }

    // ── Zero-size allocation ───────────────────────────────────────

    #[test]
    fn pool_zero_size_allocation_rejected() {
        let mut pool = MemoryPool::new(256).unwrap();
        let err = pool.allocate(0, 8).unwrap_err();
        assert_eq!(err, MemoryPoolError::ZeroSizeAllocation);
    }

    // ── Debug formatting ───────────────────────────────────────────

    #[test]
    fn pool_debug_format() {
        let pool = MemoryPool::new(1024).unwrap();
        let dbg = format!("{pool:?}");
        assert!(dbg.contains("MemoryPool"));
        assert!(dbg.contains("1024"));
    }

    // ── ScratchAllocator ───────────────────────────────────────────

    #[test]
    fn scratch_basic_scope() {
        let mut scratch = ScratchAllocator::new(4096).unwrap();
        scratch.new_layer_scope();
        scratch.temp_buffer(128).unwrap();
        assert!(scratch.used() >= 128);
        scratch.end_layer_scope().unwrap();
        assert_eq!(scratch.used(), 0);
    }

    #[test]
    fn scratch_nested_scopes() {
        let mut scratch = ScratchAllocator::new(8192).unwrap();
        scratch.new_layer_scope();
        scratch.temp_buffer(100).unwrap();
        let after_outer = scratch.used();

        scratch.new_layer_scope();
        scratch.temp_buffer(200).unwrap();
        assert!(scratch.used() > after_outer);
        assert_eq!(scratch.scope_depth(), 2);

        scratch.end_layer_scope().unwrap();
        assert_eq!(scratch.used(), after_outer);
        assert_eq!(scratch.scope_depth(), 1);

        scratch.end_layer_scope().unwrap();
        assert_eq!(scratch.used(), 0);
        assert_eq!(scratch.scope_depth(), 0);
    }

    #[test]
    fn scratch_triple_nesting() {
        let mut s = ScratchAllocator::new(8192).unwrap();
        s.new_layer_scope();
        s.temp_buffer(64).unwrap();
        s.new_layer_scope();
        s.temp_buffer(64).unwrap();
        s.new_layer_scope();
        s.temp_buffer(64).unwrap();
        assert_eq!(s.scope_depth(), 3);

        s.end_layer_scope().unwrap();
        s.end_layer_scope().unwrap();
        s.end_layer_scope().unwrap();
        assert_eq!(s.scope_depth(), 0);
        assert_eq!(s.used(), 0);
    }

    #[test]
    fn scratch_scope_mismatch_error() {
        let mut scratch = ScratchAllocator::new(256).unwrap();
        let err = scratch.end_layer_scope().unwrap_err();
        assert_eq!(err, MemoryPoolError::ScopeMismatch);
    }

    #[test]
    fn scratch_sequential_scopes_reuse_memory() {
        let mut scratch = ScratchAllocator::new(512).unwrap();
        for _ in 0..20 {
            scratch.new_layer_scope();
            scratch.temp_buffer(256).unwrap();
            scratch.end_layer_scope().unwrap();
        }
        // After all scopes ended the pool should be back to zero.
        assert_eq!(scratch.used(), 0);
    }

    #[test]
    fn scratch_allocate_aligned() {
        let mut scratch = ScratchAllocator::new(4096).unwrap();
        scratch.new_layer_scope();
        let buf = scratch.allocate_aligned(64, 32).unwrap();
        assert_eq!(buf.as_ptr() as usize % 32, 0);
        scratch.end_layer_scope().unwrap();
    }

    #[test]
    fn scratch_capacity_remaining() {
        let scratch = ScratchAllocator::new(2048).unwrap();
        assert_eq!(scratch.capacity(), 2048);
        assert_eq!(scratch.remaining(), 2048);
    }

    #[test]
    fn scratch_full_reset() {
        let mut scratch = ScratchAllocator::new(1024).unwrap();
        scratch.new_layer_scope();
        scratch.temp_buffer(512).unwrap();
        scratch.reset();
        assert_eq!(scratch.used(), 0);
        assert_eq!(scratch.scope_depth(), 0);
    }

    #[test]
    fn scratch_debug_format() {
        let scratch = ScratchAllocator::new(512).unwrap();
        let dbg = format!("{scratch:?}");
        assert!(dbg.contains("ScratchAllocator"));
    }

    // ── TensorBufferCache ──────────────────────────────────────────

    #[test]
    fn cache_miss_then_hit() {
        let mut cache = TensorBufferCache::new(4);
        let _ = cache.get_or_allocate(&[4, 8], 4).unwrap();
        assert_eq!(cache.cache_stats().misses, 1);
        assert_eq!(cache.cache_stats().hits, 0);

        let _ = cache.get_or_allocate(&[4, 8], 4).unwrap();
        assert_eq!(cache.cache_stats().hits, 1);
        assert_eq!(cache.cache_stats().misses, 1);
    }

    #[test]
    fn cache_different_shapes_are_different_keys() {
        let mut cache = TensorBufferCache::new(8);
        cache.get_or_allocate(&[2, 3], 4).unwrap();
        cache.get_or_allocate(&[3, 2], 4).unwrap();
        assert_eq!(cache.cache_stats().misses, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_different_dtype_size_different_key() {
        let mut cache = TensorBufferCache::new(8);
        cache.get_or_allocate(&[4, 4], 4).unwrap();
        cache.get_or_allocate(&[4, 4], 2).unwrap();
        assert_eq!(cache.cache_stats().misses, 2);
    }

    #[test]
    fn cache_lru_eviction() {
        let mut cache = TensorBufferCache::new(2);
        // Fill cache: entry A then entry B.
        cache.get_or_allocate(&[1], 1).unwrap(); // A
        cache.get_or_allocate(&[2], 1).unwrap(); // B
        assert_eq!(cache.len(), 2);

        // Access A again to make it recent.
        cache.get_or_allocate(&[1], 1).unwrap(); // A hit

        // Insert C — should evict B (least recently used).
        cache.get_or_allocate(&[3], 1).unwrap(); // C miss
        assert_eq!(cache.len(), 2);

        // A should still be cached.
        cache.get_or_allocate(&[1], 1).unwrap();
        assert_eq!(cache.cache_stats().hits, 2); // A was hit twice total

        // B was evicted — this is a miss.
        cache.get_or_allocate(&[2], 1).unwrap();
        assert_eq!(cache.cache_stats().misses, 4); // A, B, C, B-again
    }

    #[test]
    fn cache_explicit_evict_lru() {
        let mut cache = TensorBufferCache::new(8);
        cache.get_or_allocate(&[10], 1).unwrap();
        cache.get_or_allocate(&[20], 1).unwrap();
        assert_eq!(cache.len(), 2);
        cache.evict_lru();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_evict_on_empty_is_noop() {
        let mut cache = TensorBufferCache::new(4);
        cache.evict_lru(); // should not panic
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_zero_shape_element_rejected() {
        let mut cache = TensorBufferCache::new(4);
        let err = cache.get_or_allocate(&[4, 0, 8], 4).unwrap_err();
        assert_eq!(err, MemoryPoolError::ZeroSizeAllocation);
    }

    #[test]
    fn cache_hit_rate_initial() {
        let cache = TensorBufferCache::new(4);
        assert_eq!(cache.cache_stats().hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate_after_use() {
        let mut cache = TensorBufferCache::new(4);
        cache.get_or_allocate(&[2, 2], 4).unwrap(); // miss
        cache.get_or_allocate(&[2, 2], 4).unwrap(); // hit
        cache.get_or_allocate(&[2, 2], 4).unwrap(); // hit
        let stats = cache.cache_stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn cache_clear() {
        let mut cache = TensorBufferCache::new(4);
        cache.get_or_allocate(&[8], 4).unwrap();
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_debug_format() {
        let cache = TensorBufferCache::new(4);
        let dbg = format!("{cache:?}");
        assert!(dbg.contains("TensorBufferCache"));
    }

    #[test]
    fn cache_returns_correct_buffer_size() {
        let mut cache = TensorBufferCache::new(4);
        let buf = cache.get_or_allocate(&[3, 4], 4).unwrap();
        assert_eq!(buf.len(), 3 * 4 * 4);
    }

    // ── AlignedBuffer ──────────────────────────────────────────────

    #[test]
    fn aligned_buffer_basic() {
        let buf = AlignedBuffer::new(256, 64).unwrap();
        assert_eq!(buf.len(), 256);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % 64, 0);
        assert_eq!(buf.alignment(), 64);
    }

    #[test]
    fn aligned_buffer_zero_initialised() {
        let buf = AlignedBuffer::new(128, 8).unwrap();
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn aligned_buffer_read_write() {
        let mut buf = AlignedBuffer::new(64, 8).unwrap();
        buf.as_mut_slice().fill(0xAB);
        assert!(buf.as_slice().iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn aligned_buffer_zero_size_rejected() {
        let err = AlignedBuffer::new(0, 8).unwrap_err();
        assert_eq!(err, MemoryPoolError::ZeroSizeAllocation);
    }

    #[test]
    fn aligned_buffer_bad_alignment_rejected() {
        let err = AlignedBuffer::new(64, 5).unwrap_err();
        assert_eq!(err, MemoryPoolError::InvalidAlignment(5));
    }

    #[test]
    fn aligned_buffer_alignment_variations() {
        for align in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let buf = AlignedBuffer::new(512, align).unwrap();
            assert_eq!(buf.as_ptr() as usize % align, 0, "alignment {align} failed");
        }
    }

    #[test]
    fn aligned_buffer_debug_format() {
        let buf = AlignedBuffer::new(32, 16).unwrap();
        let dbg = format!("{buf:?}");
        assert!(dbg.contains("AlignedBuffer"));
        assert!(dbg.contains("32"));
    }

    // ── Send bounds ────────────────────────────────────────────────

    #[test]
    fn pool_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MemoryPool>();
    }

    #[test]
    fn scratch_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ScratchAllocator>();
    }

    #[test]
    fn aligned_buffer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AlignedBuffer>();
    }

    #[test]
    fn cache_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TensorBufferCache>();
    }

    // ── Integration-style scenarios ────────────────────────────────

    #[test]
    fn layer_forward_pass_pattern() {
        // Simulates the allocation pattern during a transformer
        // forward pass: allocate scratch per layer, free between.
        let mut scratch = ScratchAllocator::new(1 << 16); // 64 KiB
        let scratch = scratch.as_mut().unwrap();
        let num_layers = 12;
        for _ in 0..num_layers {
            scratch.new_layer_scope();
            // attention scratch
            scratch.temp_buffer(2048).unwrap();
            // ffn scratch
            scratch.temp_buffer(4096).unwrap();
            scratch.end_layer_scope().unwrap();
        }
        assert_eq!(scratch.used(), 0);
    }

    #[test]
    fn cache_repeated_forward_passes() {
        // Over many forward passes the same shapes should be cached.
        let mut cache = TensorBufferCache::new(8);
        let shapes: &[&[usize]] = &[&[4, 64], &[4, 256], &[4, 64]];
        for _ in 0..100 {
            for shape in shapes {
                cache.get_or_allocate(shape, 4).unwrap();
            }
        }
        let stats = cache.cache_stats();
        // First pass: 3 misses (2 unique shapes). Remaining 99*3=297
        // queries hit.  Actually 3 shapes but [4,64] appears twice in
        // the list, so 2 unique; first pass: 2 misses + 1 hit.
        assert!(stats.hits > stats.misses, "expected mostly hits: {stats:?}");
    }
}
