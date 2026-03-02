//! Optimized memory allocation for CPU inference.
//!
//! Provides SIMD-aligned allocators, arena and pool allocators for
//! efficient tensor memory management during forward passes, and a
//! memory planner that computes reuse opportunities across tensors.

use bitnet_common::{BitNetError, KernelError, Result};
use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

fn alloc_failed(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::ExecutionFailed { reason: reason.to_string() })
}

// ── Constants ──────────────────────────────────────────────────────

/// Default alignment for AVX2 (256-bit = 32 bytes).
pub const ALIGN_AVX2: usize = 32;

/// Default alignment for AVX-512 (512-bit = 64 bytes).
pub const ALIGN_AVX512: usize = 64;

/// Default SIMD alignment used throughout the crate.
pub const DEFAULT_ALIGNMENT: usize = ALIGN_AVX512;

/// Maximum supported alignment (4 KiB page boundary).
pub const MAX_ALIGNMENT: usize = 4096;

// ── AlignedAllocator ───────────────────────────────────────────────

/// Allocator that ensures SIMD-friendly alignment (32 or 64 byte).
///
/// Wraps the global allocator and adjusts every layout so the
/// returned pointer satisfies the requested alignment.
#[derive(Debug, Clone, Copy)]
pub struct AlignedAllocator {
    alignment: usize,
}

impl AlignedAllocator {
    /// Create an aligned allocator with the given power-of-two alignment.
    pub fn new(alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(invalid_arg("alignment must be a power of two"));
        }
        if alignment > MAX_ALIGNMENT {
            return Err(invalid_arg("alignment exceeds maximum"));
        }
        Ok(Self { alignment })
    }

    /// Create an allocator with 32-byte (AVX2) alignment.
    #[inline]
    pub fn avx2() -> Self {
        Self { alignment: ALIGN_AVX2 }
    }

    /// Create an allocator with 64-byte (AVX-512) alignment.
    #[inline]
    pub fn avx512() -> Self {
        Self { alignment: ALIGN_AVX512 }
    }

    /// The alignment this allocator guarantees.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

impl Default for AlignedAllocator {
    fn default() -> Self {
        Self { alignment: DEFAULT_ALIGNMENT }
    }
}

// ── alloc_aligned / alloc_zeroed ───────────────────────────────────

/// Allocate `size` bytes with the given power-of-two alignment.
///
/// Returns a non-null pointer to the allocated block. The caller must
/// free the memory via [`dealloc_aligned`].
///
/// # Safety
/// The returned pointer **must** be deallocated with [`dealloc_aligned`]
/// using the same `size` and `alignment`.
pub fn alloc_aligned(size: usize, alignment: usize) -> Result<NonNull<u8>> {
    if size == 0 {
        return Err(invalid_arg("allocation size must be > 0"));
    }
    if !alignment.is_power_of_two() {
        return Err(invalid_arg("alignment must be a power of two"));
    }
    let layout = Layout::from_size_align(size, alignment)
        .map_err(|e| alloc_failed(&format!("invalid layout: {e}")))?;
    // SAFETY: layout has non-zero size.
    let ptr = unsafe { alloc::alloc(layout) };
    NonNull::new(ptr).ok_or_else(|| alloc_failed("global allocator returned null"))
}

/// Allocate `size` bytes of **zeroed** memory with the given alignment.
pub fn alloc_zeroed(size: usize, alignment: usize) -> Result<NonNull<u8>> {
    if size == 0 {
        return Err(invalid_arg("allocation size must be > 0"));
    }
    if !alignment.is_power_of_two() {
        return Err(invalid_arg("alignment must be a power of two"));
    }
    let layout = Layout::from_size_align(size, alignment)
        .map_err(|e| alloc_failed(&format!("invalid layout: {e}")))?;
    // SAFETY: layout has non-zero size.
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    NonNull::new(ptr).ok_or_else(|| alloc_failed("global allocator returned null"))
}

/// Free memory previously allocated by [`alloc_aligned`] or
/// [`alloc_zeroed`].
///
/// # Safety
/// `ptr`, `size`, and `alignment` must exactly match the original
/// allocation.
pub unsafe fn dealloc_aligned(ptr: NonNull<u8>, size: usize, alignment: usize) {
    let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };
    unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
}

// ── TensorBuffer ───────────────────────────────────────────────────

/// Pre-allocated buffer for tensor data with alignment guarantees.
pub struct TensorBuffer {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
    /// Number of f32 elements that fit in this buffer.
    num_elements: usize,
}

impl TensorBuffer {
    /// Allocate a new tensor buffer that can hold `num_elements` f32 values.
    pub fn new(num_elements: usize, alignment: usize) -> Result<Self> {
        if num_elements == 0 {
            return Err(invalid_arg("num_elements must be > 0"));
        }
        let size = num_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_arg("allocation size overflow"))?;
        let ptr = alloc_zeroed(size, alignment)?;
        Ok(Self { ptr, size, alignment, num_elements })
    }

    /// Allocate with the default SIMD alignment.
    pub fn with_default_alignment(num_elements: usize) -> Result<Self> {
        Self::new(num_elements, DEFAULT_ALIGNMENT)
    }

    /// Pointer to the underlying buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr().cast()
    }

    /// Mutable pointer to the underlying buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr().cast()
    }

    /// View the buffer as an f32 slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: buffer was allocated for `num_elements` f32 values and zeroed.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.num_elements) }
    }

    /// View the buffer as a mutable f32 slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: same as `as_slice`, we have &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.num_elements) }
    }

    /// Number of f32 elements.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Total byte size of the buffer.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.size
    }

    /// The alignment of this buffer.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Zero-fill the buffer contents.
    pub fn zero(&mut self) {
        // SAFETY: ptr is valid for `size` bytes.
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size);
        }
    }
}

impl Drop for TensorBuffer {
    fn drop(&mut self) {
        // SAFETY: matches the allocation layout.
        unsafe {
            dealloc_aligned(self.ptr, self.size, self.alignment);
        }
    }
}

// SAFETY: The buffer owns its allocation and is not aliased.
unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

// ── ArenaAllocator ─────────────────────────────────────────────────

/// Arena-based allocator for inference: allocate forward, reset
/// between batches.
///
/// All memory is freed at once when the arena is dropped or
/// explicitly [`reset`](ArenaAllocator::reset).
pub struct ArenaAllocator {
    /// Backing storage.
    storage: NonNull<u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current bump offset.
    offset: AtomicUsize,
    /// Alignment for the arena itself.
    alignment: usize,
    /// High-water mark (peak usage).
    peak: AtomicUsize,
}

impl ArenaAllocator {
    /// Create an arena with `capacity` bytes and the given alignment.
    pub fn new(capacity: usize, alignment: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(invalid_arg("arena capacity must be > 0"));
        }
        let storage = alloc_aligned(capacity, alignment)?;
        Ok(Self {
            storage,
            capacity,
            offset: AtomicUsize::new(0),
            alignment,
            peak: AtomicUsize::new(0),
        })
    }

    /// Create an arena with 64-byte alignment.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::new(capacity, DEFAULT_ALIGNMENT)
    }

    /// Allocate `size` bytes from the arena, respecting `align`.
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(invalid_arg("arena alloc size must be > 0"));
        }
        if !align.is_power_of_two() {
            return Err(invalid_arg("alignment must be a power of two"));
        }
        loop {
            let current = self.offset.load(Ordering::Relaxed);
            let base = self.storage.as_ptr() as usize + current;
            let aligned = (base + align - 1) & !(align - 1);
            let padding = aligned - base;
            let total = padding + size;
            let new_offset = current + total;
            if new_offset > self.capacity {
                return Err(alloc_failed("arena out of memory"));
            }
            if self
                .offset
                .compare_exchange_weak(current, new_offset, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Update peak.
                let _ = self.peak.fetch_max(new_offset, Ordering::Relaxed);
                // SAFETY: pointer is within the arena allocation.
                return Ok(unsafe { NonNull::new_unchecked(aligned as *mut u8) });
            }
        }
    }

    /// Allocate space for `count` values of type `T`.
    pub fn alloc_slice<T>(&self, count: usize) -> Result<NonNull<T>> {
        let size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| invalid_arg("alloc_slice size overflow"))?;
        let align = std::mem::align_of::<T>().max(self.alignment);
        let ptr = self.alloc(size, align)?;
        Ok(ptr.cast())
    }

    /// Reset the arena (free all allocations at once).
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }

    /// Bytes currently in use.
    #[inline]
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Total arena capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remaining free bytes.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }

    /// High-water mark (peak usage since creation or last reset).
    #[inline]
    pub fn peak_usage(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        // SAFETY: matches the original allocation.
        unsafe {
            dealloc_aligned(self.storage, self.capacity, self.alignment);
        }
    }
}

// SAFETY: Atomic offset makes concurrent alloc safe.
unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

// ── PoolAllocator ──────────────────────────────────────────────────

/// Pool allocator for fixed-size tensor allocations.
///
/// Maintains a free-list of identically sized blocks so allocations
/// and deallocations are O(1).
pub struct PoolAllocator {
    block_size: usize,
    alignment: usize,
    /// Free blocks ready for reuse.
    free_list: Vec<NonNull<u8>>,
    /// All blocks ever allocated (for cleanup).
    all_blocks: Vec<NonNull<u8>>,
    /// High-water mark: max live blocks.
    peak_live: usize,
    /// Currently live (allocated, not returned) blocks.
    live_count: usize,
}

impl PoolAllocator {
    /// Create a pool for blocks of `block_size` bytes.
    pub fn new(block_size: usize, alignment: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(invalid_arg("block_size must be > 0"));
        }
        if !alignment.is_power_of_two() {
            return Err(invalid_arg("alignment must be a power of two"));
        }
        Ok(Self {
            block_size,
            alignment,
            free_list: Vec::new(),
            all_blocks: Vec::new(),
            peak_live: 0,
            live_count: 0,
        })
    }

    /// Create a pool with default alignment for `num_f32` floats.
    pub fn for_f32(num_f32: usize) -> Result<Self> {
        Self::new(num_f32 * std::mem::size_of::<f32>(), DEFAULT_ALIGNMENT)
    }

    /// Pre-allocate `n` blocks.
    pub fn preallocate(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            let ptr = alloc_aligned(self.block_size, self.alignment)?;
            self.all_blocks.push(ptr);
            self.free_list.push(ptr);
        }
        Ok(())
    }

    /// Acquire a block from the pool, allocating a new one if the
    /// free list is empty.
    pub fn acquire(&mut self) -> Result<NonNull<u8>> {
        let ptr = if let Some(p) = self.free_list.pop() {
            p
        } else {
            let p = alloc_aligned(self.block_size, self.alignment)?;
            self.all_blocks.push(p);
            p
        };
        self.live_count += 1;
        if self.live_count > self.peak_live {
            self.peak_live = self.live_count;
        }
        Ok(ptr)
    }

    /// Return a block to the pool for reuse.
    ///
    /// # Safety
    /// `ptr` must have been obtained from this pool's [`acquire`](PoolAllocator::acquire).
    pub unsafe fn release(&mut self, ptr: NonNull<u8>) {
        self.free_list.push(ptr);
        self.live_count = self.live_count.saturating_sub(1);
    }

    /// Number of free blocks ready for reuse.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Number of currently live blocks.
    #[inline]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Peak number of simultaneously live blocks.
    #[inline]
    pub fn peak_live(&self) -> usize {
        self.peak_live
    }

    /// Total blocks ever allocated by this pool.
    #[inline]
    pub fn total_allocated(&self) -> usize {
        self.all_blocks.len()
    }

    /// Block size in bytes.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl Drop for PoolAllocator {
    fn drop(&mut self) {
        for ptr in &self.all_blocks {
            // SAFETY: each pointer was allocated with the pool's layout.
            unsafe {
                dealloc_aligned(*ptr, self.block_size, self.alignment);
            }
        }
    }
}

// ── BufferPool ─────────────────────────────────────────────────────

/// Pool of reusable [`TensorBuffer`]s.
///
/// Buffers are bucketed by element count so a returned buffer is
/// always at least as large as requested.
pub struct BufferPool {
    alignment: usize,
    /// Map from element count → free buffers.
    buckets: HashMap<usize, Vec<TensorBuffer>>,
    /// Total buffers held in the pool.
    total_pooled: usize,
    /// Total buffers ever created.
    total_created: usize,
}

impl BufferPool {
    /// Create a new empty buffer pool.
    pub fn new(alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(invalid_arg("alignment must be a power of two"));
        }
        Ok(Self { alignment, buckets: HashMap::new(), total_pooled: 0, total_created: 0 })
    }

    /// Create with default alignment.
    pub fn with_default_alignment() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            buckets: HashMap::new(),
            total_pooled: 0,
            total_created: 0,
        }
    }

    /// Round `n` up to the nearest bucket boundary.
    fn bucket_size(n: usize) -> usize {
        // Round up to nearest multiple of 256 for cache-friendly sizes.
        (n + 255) & !255
    }

    /// Acquire a buffer of at least `num_elements` f32 values.
    pub fn acquire(&mut self, num_elements: usize) -> Result<TensorBuffer> {
        let key = Self::bucket_size(num_elements);
        if let Some(bucket) = self.buckets.get_mut(&key)
            && let Some(mut buf) = bucket.pop()
        {
            self.total_pooled -= 1;
            buf.zero();
            return Ok(buf);
        }
        self.total_created += 1;
        TensorBuffer::new(key, self.alignment)
    }

    /// Return a buffer to the pool.
    pub fn release(&mut self, buf: TensorBuffer) {
        let key = Self::bucket_size(buf.num_elements());
        self.buckets.entry(key).or_default().push(buf);
        self.total_pooled += 1;
    }

    /// Number of buffers currently in the pool.
    #[inline]
    pub fn pooled_count(&self) -> usize {
        self.total_pooled
    }

    /// Total buffers ever created.
    #[inline]
    pub fn total_created(&self) -> usize {
        self.total_created
    }

    /// Drop all pooled buffers, freeing their memory.
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.total_pooled = 0;
    }
}

// ── MemoryPlanner / memory_reuse_plan ──────────────────────────────

/// Describes a tensor's lifetime during a forward pass.
#[derive(Debug, Clone)]
pub struct TensorLifetime {
    /// Unique identifier for this tensor.
    pub id: usize,
    /// Size in bytes.
    pub size: usize,
    /// Required alignment.
    pub alignment: usize,
    /// The first operation index where this tensor is produced.
    pub first_use: usize,
    /// The last operation index where this tensor is consumed.
    pub last_use: usize,
}

/// Mapping from a tensor id to the offset in a shared memory block.
#[derive(Debug, Clone)]
pub struct MemoryAssignment {
    /// Tensor id.
    pub tensor_id: usize,
    /// Byte offset into the shared allocation.
    pub offset: usize,
    /// Allocated size (may be >= requested to satisfy alignment).
    pub size: usize,
}

/// A complete memory reuse plan.
#[derive(Debug, Clone)]
pub struct MemoryReusePlan {
    /// Total bytes needed for the shared allocation.
    pub total_bytes: usize,
    /// Per-tensor offset assignments.
    pub assignments: Vec<MemoryAssignment>,
    /// Number of tensors that share memory with another tensor.
    pub reuses: usize,
}

/// Plan memory usage for a model's forward pass.
///
/// Performs greedy lifetime analysis: tensors whose lifetimes do not
/// overlap may share the same region of a single allocation.
pub struct MemoryPlanner {
    tensors: Vec<TensorLifetime>,
}

impl MemoryPlanner {
    /// Create a new planner.
    pub fn new() -> Self {
        Self { tensors: Vec::new() }
    }

    /// Register a tensor with its lifetime.
    pub fn add_tensor(&mut self, lifetime: TensorLifetime) {
        self.tensors.push(lifetime);
    }

    /// Compute the reuse plan.
    pub fn plan(&self) -> MemoryReusePlan {
        memory_reuse_plan(&self.tensors)
    }
}

impl Default for MemoryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute which tensors can share memory.
///
/// Uses a greedy first-fit-decreasing algorithm: sort by descending
/// size, then place each tensor into the first gap whose lifetime
/// does not overlap.
pub fn memory_reuse_plan(tensors: &[TensorLifetime]) -> MemoryReusePlan {
    if tensors.is_empty() {
        return MemoryReusePlan { total_bytes: 0, assignments: Vec::new(), reuses: 0 };
    }

    // Sort indices by descending size for better packing.
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| tensors[b].size.cmp(&tensors[a].size));

    struct Slot {
        offset: usize,
        size: usize,
        last_use: usize,
    }

    let mut slots: Vec<Slot> = Vec::new();
    let mut assignments: Vec<MemoryAssignment> =
        vec![MemoryAssignment { tensor_id: 0, offset: 0, size: 0 }; tensors.len()];
    let mut reuses: usize = 0;
    let mut total_bytes: usize = 0;

    for &idx in &order {
        let t = &tensors[idx];
        let aligned_size = (t.size + t.alignment - 1) & !(t.alignment - 1);

        // Try to find a slot whose lifetime ended before this tensor starts.
        let mut placed = false;
        for slot in &mut slots {
            if slot.last_use < t.first_use && slot.size >= aligned_size {
                assignments[idx] =
                    MemoryAssignment { tensor_id: t.id, offset: slot.offset, size: aligned_size };
                slot.last_use = t.last_use;
                reuses += 1;
                placed = true;
                break;
            }
        }

        if !placed {
            let offset = total_bytes;
            total_bytes += aligned_size;
            assignments[idx] = MemoryAssignment { tensor_id: t.id, offset, size: aligned_size };
            slots.push(Slot { offset, size: aligned_size, last_use: t.last_use });
        }
    }

    MemoryReusePlan { total_bytes, assignments, reuses }
}

// ── InferenceArena ─────────────────────────────────────────────────

/// Complete arena for one inference pass.
///
/// Wraps an [`ArenaAllocator`] with convenience methods for
/// allocating tensors and resetting between inference steps.
pub struct InferenceArena {
    arena: ArenaAllocator,
    generation: usize,
}

impl InferenceArena {
    /// Create an inference arena with the given byte capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self { arena: ArenaAllocator::with_capacity(capacity)?, generation: 0 })
    }

    /// Allocate a buffer of `n` f32 values.
    pub fn alloc_f32(&self, n: usize) -> Result<NonNull<f32>> {
        self.arena.alloc_slice::<f32>(n)
    }

    /// Allocate a buffer of `n` u8 values.
    pub fn alloc_u8(&self, n: usize) -> Result<NonNull<u8>> {
        self.arena.alloc(n, self.arena.alignment)
    }

    /// Reset the arena for the next inference pass.
    pub fn next_pass(&mut self) {
        self.arena.reset();
        self.generation += 1;
    }

    /// Current generation (number of resets).
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Bytes used in the current pass.
    #[inline]
    pub fn used(&self) -> usize {
        self.arena.used()
    }

    /// Total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Peak usage across all passes.
    #[inline]
    pub fn peak_usage(&self) -> usize {
        self.arena.peak_usage()
    }
}

// ── ScratchSpace ───────────────────────────────────────────────────

/// Thread-local scratch space for temporary computations.
///
/// Each thread gets its own [`TensorBuffer`] that is reused across
/// calls, avoiding repeated allocation.
pub struct ScratchSpace {
    capacity_elements: usize,
    alignment: usize,
}

impl ScratchSpace {
    /// Create a scratch-space descriptor.
    pub fn new(capacity_elements: usize, alignment: usize) -> Result<Self> {
        if capacity_elements == 0 {
            return Err(invalid_arg("scratch capacity must be > 0"));
        }
        if !alignment.is_power_of_two() {
            return Err(invalid_arg("alignment must be a power of two"));
        }
        Ok(Self { capacity_elements, alignment })
    }

    /// Create with default SIMD alignment.
    pub fn with_capacity(capacity_elements: usize) -> Result<Self> {
        Self::new(capacity_elements, DEFAULT_ALIGNMENT)
    }

    /// Number of f32 elements.
    #[inline]
    pub fn capacity_elements(&self) -> usize {
        self.capacity_elements
    }

    /// Byte capacity.
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.capacity_elements * std::mem::size_of::<f32>()
    }

    /// Borrow the thread-local buffer, initialising it on first use.
    ///
    /// The closure receives a mutable slice of `f32` values. The
    /// contents are **not** zeroed between calls for performance;
    /// the caller must initialise the region it uses.
    pub fn with<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        thread_local! {
            static TLS_BUF: RefCell<Option<TensorBuffer>> = const { RefCell::new(None) };
        }
        let cap = self.capacity_elements;
        let align = self.alignment;
        TLS_BUF.with(|cell| {
            let mut borrow = cell.borrow_mut();
            if borrow.as_ref().is_none_or(|b| b.num_elements() < cap) {
                *borrow = Some(TensorBuffer::new(cap, align)?);
            }
            let buf = borrow.as_mut().unwrap();
            Ok(f(buf.as_mut_slice()))
        })
    }
}

// ── numa_aware_alloc ───────────────────────────────────────────────

/// NUMA node preference for allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// Use the OS default policy (no explicit NUMA binding).
    Default,
    /// Prefer the specified NUMA node (0-indexed).
    PreferNode(usize),
    /// Interleave pages across all NUMA nodes.
    Interleave,
}

/// Allocate `size` bytes with NUMA-awareness.
///
/// On platforms without NUMA support (or when the `libnuma` library
/// is unavailable) this falls back to [`alloc_aligned`].
pub fn numa_aware_alloc(size: usize, alignment: usize, _policy: NumaPolicy) -> Result<NonNull<u8>> {
    // Placeholder: fall back to the standard aligned allocator.
    // A future implementation may use libnuma / mbind(2) on Linux.
    alloc_aligned(size, alignment)
}

/// Free memory allocated by [`numa_aware_alloc`].
///
/// # Safety
/// Same invariants as [`dealloc_aligned`].
pub unsafe fn numa_aware_dealloc(ptr: NonNull<u8>, size: usize, alignment: usize) {
    unsafe { dealloc_aligned(ptr, size, alignment) };
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── alloc_aligned ──────────────────────────────────────────────

    #[test]
    fn test_alloc_aligned_basic() {
        let ptr = alloc_aligned(128, 64).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        unsafe { dealloc_aligned(ptr, 128, 64) };
    }

    #[test]
    fn test_alloc_aligned_avx2() {
        let ptr = alloc_aligned(256, ALIGN_AVX2).unwrap();
        assert_eq!(ptr.as_ptr() as usize % ALIGN_AVX2, 0);
        unsafe { dealloc_aligned(ptr, 256, ALIGN_AVX2) };
    }

    #[test]
    fn test_alloc_aligned_avx512() {
        let ptr = alloc_aligned(512, ALIGN_AVX512).unwrap();
        assert_eq!(ptr.as_ptr() as usize % ALIGN_AVX512, 0);
        unsafe { dealloc_aligned(ptr, 512, ALIGN_AVX512) };
    }

    #[test]
    fn test_alloc_aligned_zero_size_error() {
        assert!(alloc_aligned(0, 64).is_err());
    }

    #[test]
    fn test_alloc_aligned_bad_alignment() {
        assert!(alloc_aligned(128, 3).is_err());
    }

    #[test]
    fn test_alloc_aligned_page_alignment() {
        let ptr = alloc_aligned(4096, 4096).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 4096, 0);
        unsafe { dealloc_aligned(ptr, 4096, 4096) };
    }

    // ── alloc_zeroed ───────────────────────────────────────────────

    #[test]
    fn test_alloc_zeroed_is_zeroed() {
        let ptr = alloc_zeroed(256, 64).unwrap();
        let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), 256) };
        assert!(slice.iter().all(|&b| b == 0));
        unsafe { dealloc_aligned(ptr, 256, 64) };
    }

    #[test]
    fn test_alloc_zeroed_alignment() {
        let ptr = alloc_zeroed(128, ALIGN_AVX512).unwrap();
        assert_eq!(ptr.as_ptr() as usize % ALIGN_AVX512, 0);
        unsafe { dealloc_aligned(ptr, 128, ALIGN_AVX512) };
    }

    #[test]
    fn test_alloc_zeroed_zero_size_error() {
        assert!(alloc_zeroed(0, 64).is_err());
    }

    #[test]
    fn test_alloc_zeroed_bad_alignment() {
        assert!(alloc_zeroed(64, 7).is_err());
    }

    // ── AlignedAllocator ───────────────────────────────────────────

    #[test]
    fn test_aligned_allocator_new() {
        let a = AlignedAllocator::new(32).unwrap();
        assert_eq!(a.alignment(), 32);
    }

    #[test]
    fn test_aligned_allocator_bad_alignment() {
        assert!(AlignedAllocator::new(13).is_err());
    }

    #[test]
    fn test_aligned_allocator_exceeds_max() {
        assert!(AlignedAllocator::new(8192).is_err());
    }

    #[test]
    fn test_aligned_allocator_avx2() {
        assert_eq!(AlignedAllocator::avx2().alignment(), 32);
    }

    #[test]
    fn test_aligned_allocator_avx512() {
        assert_eq!(AlignedAllocator::avx512().alignment(), 64);
    }

    #[test]
    fn test_aligned_allocator_default() {
        assert_eq!(AlignedAllocator::default().alignment(), DEFAULT_ALIGNMENT);
    }

    // ── TensorBuffer ───────────────────────────────────────────────

    #[test]
    fn test_tensor_buffer_basic() {
        let buf = TensorBuffer::with_default_alignment(1024).unwrap();
        assert_eq!(buf.num_elements(), 1024);
        assert_eq!(buf.byte_size(), 1024 * 4);
        assert_eq!(buf.alignment(), DEFAULT_ALIGNMENT);
    }

    #[test]
    fn test_tensor_buffer_zeroed_on_create() {
        let buf = TensorBuffer::with_default_alignment(512).unwrap();
        assert!(buf.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_buffer_write_read() {
        let mut buf = TensorBuffer::with_default_alignment(4).unwrap();
        let s = buf.as_mut_slice();
        s[0] = 1.0;
        s[1] = 2.0;
        s[2] = 3.0;
        s[3] = 4.0;
        assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_buffer_zero() {
        let mut buf = TensorBuffer::with_default_alignment(8).unwrap();
        buf.as_mut_slice()[0] = 42.0;
        buf.zero();
        assert!(buf.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_buffer_alignment() {
        let buf = TensorBuffer::new(256, ALIGN_AVX512).unwrap();
        assert_eq!(buf.as_ptr() as usize % ALIGN_AVX512, 0);
    }

    #[test]
    fn test_tensor_buffer_zero_elements_error() {
        assert!(TensorBuffer::with_default_alignment(0).is_err());
    }

    #[test]
    fn test_tensor_buffer_custom_alignment() {
        let buf = TensorBuffer::new(64, ALIGN_AVX2).unwrap();
        assert_eq!(buf.alignment(), ALIGN_AVX2);
        assert_eq!(buf.as_ptr() as usize % ALIGN_AVX2, 0);
    }

    // ── ArenaAllocator ─────────────────────────────────────────────

    #[test]
    fn test_arena_basic_alloc() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        let ptr = arena.alloc(64, 64).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        let _a = arena.alloc(128, 32).unwrap();
        let _b = arena.alloc(256, 64).unwrap();
        let _c = arena.alloc(64, 16).unwrap();
        assert!(arena.used() > 0);
    }

    #[test]
    fn test_arena_reset() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        let _p = arena.alloc(1024, 64).unwrap();
        assert!(arena.used() > 0);
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_remaining() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        assert_eq!(arena.remaining(), 4096);
        let _p = arena.alloc(1024, 64).unwrap();
        assert!(arena.remaining() < 4096);
    }

    #[test]
    fn test_arena_oom() {
        let arena = ArenaAllocator::with_capacity(128).unwrap();
        assert!(arena.alloc(256, 64).is_err());
    }

    #[test]
    fn test_arena_zero_capacity_error() {
        assert!(ArenaAllocator::with_capacity(0).is_err());
    }

    #[test]
    fn test_arena_alloc_slice() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        let ptr = arena.alloc_slice::<f32>(128).unwrap();
        assert_eq!(ptr.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
    }

    #[test]
    fn test_arena_peak_usage() {
        let arena = ArenaAllocator::with_capacity(8192).unwrap();
        let _a = arena.alloc(1024, 64).unwrap();
        let _b = arena.alloc(2048, 64).unwrap();
        let peak = arena.peak_usage();
        assert!(peak >= 3072);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert!(arena.peak_usage() >= 3072);
    }

    #[test]
    fn test_arena_alloc_zero_size_error() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        assert!(arena.alloc(0, 64).is_err());
    }

    #[test]
    fn test_arena_alloc_bad_alignment() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        assert!(arena.alloc(64, 5).is_err());
    }

    #[test]
    fn test_arena_successive_resets() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        for _ in 0..10 {
            let _p = arena.alloc(256, 64).unwrap();
            arena.reset();
            assert_eq!(arena.used(), 0);
        }
    }

    #[test]
    fn test_arena_capacity_exact() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        assert_eq!(arena.capacity(), 4096);
    }

    // ── PoolAllocator ──────────────────────────────────────────────

    #[test]
    fn test_pool_acquire_release() {
        let mut pool = PoolAllocator::new(256, 64).unwrap();
        let ptr = pool.acquire().unwrap();
        assert_eq!(pool.live_count(), 1);
        unsafe { pool.release(ptr) };
        assert_eq!(pool.live_count(), 0);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = PoolAllocator::new(256, 64).unwrap();
        let p1 = pool.acquire().unwrap();
        unsafe { pool.release(p1) };
        let p2 = pool.acquire().unwrap();
        assert_eq!(p1.as_ptr(), p2.as_ptr());
        unsafe { pool.release(p2) };
    }

    #[test]
    fn test_pool_preallocate() {
        let mut pool = PoolAllocator::new(128, 64).unwrap();
        pool.preallocate(5).unwrap();
        assert_eq!(pool.free_count(), 5);
        assert_eq!(pool.total_allocated(), 5);
    }

    #[test]
    fn test_pool_peak_live() {
        let mut pool = PoolAllocator::new(64, 32).unwrap();
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        assert_eq!(pool.peak_live(), 2);
        unsafe {
            pool.release(a);
            pool.release(b);
        }
        assert_eq!(pool.peak_live(), 2);
    }

    #[test]
    fn test_pool_zero_block_size_error() {
        assert!(PoolAllocator::new(0, 64).is_err());
    }

    #[test]
    fn test_pool_bad_alignment() {
        assert!(PoolAllocator::new(64, 3).is_err());
    }

    #[test]
    fn test_pool_for_f32() {
        let pool = PoolAllocator::for_f32(256).unwrap();
        assert_eq!(pool.block_size(), 256 * 4);
    }

    #[test]
    fn test_pool_multiple_acquire() {
        let mut pool = PoolAllocator::new(128, 64).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            ptrs.push(pool.acquire().unwrap());
        }
        assert_eq!(pool.live_count(), 10);
        assert_eq!(pool.total_allocated(), 10);
        for p in ptrs {
            unsafe { pool.release(p) };
        }
        assert_eq!(pool.live_count(), 0);
        assert_eq!(pool.free_count(), 10);
    }

    // ── BufferPool ─────────────────────────────────────────────────

    #[test]
    fn test_buffer_pool_acquire() {
        let mut pool = BufferPool::with_default_alignment();
        let buf = pool.acquire(128).unwrap();
        assert!(buf.num_elements() >= 128);
        assert_eq!(pool.total_created(), 1);
    }

    #[test]
    fn test_buffer_pool_release_and_reuse() {
        let mut pool = BufferPool::with_default_alignment();
        let buf = pool.acquire(128).unwrap();
        let n = buf.num_elements();
        pool.release(buf);
        assert_eq!(pool.pooled_count(), 1);
        let buf2 = pool.acquire(128).unwrap();
        assert_eq!(buf2.num_elements(), n);
        assert_eq!(pool.pooled_count(), 0);
        assert_eq!(pool.total_created(), 1);
    }

    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = BufferPool::with_default_alignment();
        let b1 = pool.acquire(64).unwrap();
        let b2 = pool.acquire(64).unwrap();
        pool.release(b1);
        pool.release(b2);
        assert_eq!(pool.pooled_count(), 2);
        pool.clear();
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_buffer_pool_bad_alignment() {
        assert!(BufferPool::new(3).is_err());
    }

    #[test]
    fn test_buffer_pool_bucketing() {
        // Requesting 100 and 200 elements should bucket to 256 each.
        let mut pool = BufferPool::with_default_alignment();
        let b1 = pool.acquire(100).unwrap();
        assert_eq!(b1.num_elements(), 256);
        let b2 = pool.acquire(200).unwrap();
        assert_eq!(b2.num_elements(), 256);
        pool.release(b1);
        pool.release(b2);
    }

    #[test]
    fn test_buffer_pool_different_buckets() {
        let mut pool = BufferPool::with_default_alignment();
        let small = pool.acquire(100).unwrap();
        let large = pool.acquire(1000).unwrap();
        assert_ne!(small.num_elements(), large.num_elements());
        pool.release(small);
        pool.release(large);
    }

    // ── MemoryPlanner / memory_reuse_plan ──────────────────────────

    #[test]
    fn test_memory_reuse_plan_empty() {
        let plan = memory_reuse_plan(&[]);
        assert_eq!(plan.total_bytes, 0);
        assert!(plan.assignments.is_empty());
        assert_eq!(plan.reuses, 0);
    }

    #[test]
    fn test_memory_reuse_plan_single_tensor() {
        let tensors =
            vec![TensorLifetime { id: 0, size: 1024, alignment: 64, first_use: 0, last_use: 5 }];
        let plan = memory_reuse_plan(&tensors);
        assert_eq!(plan.total_bytes, 1024);
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.reuses, 0);
    }

    #[test]
    fn test_memory_reuse_plan_no_overlap() {
        let tensors = vec![
            TensorLifetime { id: 0, size: 512, alignment: 64, first_use: 0, last_use: 2 },
            TensorLifetime { id: 1, size: 512, alignment: 64, first_use: 3, last_use: 5 },
        ];
        let plan = memory_reuse_plan(&tensors);
        // Tensor 1 can reuse tensor 0's slot.
        assert_eq!(plan.reuses, 1);
        assert!(plan.total_bytes <= 512);
    }

    #[test]
    fn test_memory_reuse_plan_full_overlap() {
        let tensors = vec![
            TensorLifetime { id: 0, size: 512, alignment: 64, first_use: 0, last_use: 5 },
            TensorLifetime { id: 1, size: 256, alignment: 64, first_use: 1, last_use: 4 },
        ];
        let plan = memory_reuse_plan(&tensors);
        // Both are alive at the same time — no reuse.
        assert_eq!(plan.reuses, 0);
        assert!(plan.total_bytes >= 768);
    }

    #[test]
    fn test_memory_reuse_plan_chain() {
        // Three tensors used sequentially.
        let tensors = vec![
            TensorLifetime { id: 0, size: 256, alignment: 64, first_use: 0, last_use: 1 },
            TensorLifetime { id: 1, size: 256, alignment: 64, first_use: 2, last_use: 3 },
            TensorLifetime { id: 2, size: 256, alignment: 64, first_use: 4, last_use: 5 },
        ];
        let plan = memory_reuse_plan(&tensors);
        assert_eq!(plan.reuses, 2);
        assert_eq!(plan.total_bytes, 256);
    }

    #[test]
    fn test_memory_planner_api() {
        let mut planner = MemoryPlanner::new();
        planner.add_tensor(TensorLifetime {
            id: 0,
            size: 128,
            alignment: 64,
            first_use: 0,
            last_use: 1,
        });
        planner.add_tensor(TensorLifetime {
            id: 1,
            size: 128,
            alignment: 64,
            first_use: 2,
            last_use: 3,
        });
        let plan = planner.plan();
        assert_eq!(plan.reuses, 1);
    }

    #[test]
    fn test_memory_planner_default() {
        let planner = MemoryPlanner::default();
        let plan = planner.plan();
        assert_eq!(plan.total_bytes, 0);
    }

    #[test]
    fn test_memory_reuse_plan_alignment_respected() {
        let tensors =
            vec![TensorLifetime { id: 0, size: 100, alignment: 64, first_use: 0, last_use: 1 }];
        let plan = memory_reuse_plan(&tensors);
        // 100 rounded to 64-byte alignment = 128.
        assert_eq!(plan.assignments[0].size, 128);
    }

    // ── InferenceArena ─────────────────────────────────────────────

    #[test]
    fn test_inference_arena_basic() {
        let mut arena = InferenceArena::new(8192).unwrap();
        let _p = arena.alloc_f32(64).unwrap();
        assert!(arena.used() > 0);
        arena.next_pass();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.generation(), 1);
    }

    #[test]
    fn test_inference_arena_alloc_u8() {
        let arena = InferenceArena::new(4096).unwrap();
        let p = arena.alloc_u8(128).unwrap();
        assert!(!p.as_ptr().is_null());
    }

    #[test]
    fn test_inference_arena_generations() {
        let mut arena = InferenceArena::new(4096).unwrap();
        for i in 0..5 {
            assert_eq!(arena.generation(), i);
            let _p = arena.alloc_f32(32).unwrap();
            arena.next_pass();
        }
        assert_eq!(arena.generation(), 5);
    }

    #[test]
    fn test_inference_arena_peak() {
        let mut arena = InferenceArena::new(16384).unwrap();
        let _a = arena.alloc_f32(512).unwrap();
        let _b = arena.alloc_f32(512).unwrap();
        let peak = arena.peak_usage();
        arena.next_pass();
        assert_eq!(arena.used(), 0);
        assert!(arena.peak_usage() >= peak);
    }

    #[test]
    fn test_inference_arena_capacity() {
        let arena = InferenceArena::new(2048).unwrap();
        assert_eq!(arena.capacity(), 2048);
    }

    // ── ScratchSpace ───────────────────────────────────────────────

    #[test]
    fn test_scratch_space_basic() {
        let scratch = ScratchSpace::with_capacity(256).unwrap();
        let result = scratch
            .with(|buf| {
                buf[0] = 42.0;
                buf[0]
            })
            .unwrap();
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_scratch_space_capacity() {
        let scratch = ScratchSpace::with_capacity(1024).unwrap();
        assert_eq!(scratch.capacity_elements(), 1024);
        assert_eq!(scratch.byte_capacity(), 1024 * 4);
    }

    #[test]
    fn test_scratch_space_zero_error() {
        assert!(ScratchSpace::with_capacity(0).is_err());
    }

    #[test]
    fn test_scratch_space_bad_alignment() {
        assert!(ScratchSpace::new(64, 5).is_err());
    }

    #[test]
    fn test_scratch_space_reuse() {
        let scratch = ScratchSpace::with_capacity(128).unwrap();
        scratch.with(|buf| buf[0] = 1.0).unwrap();
        // Second call reuses the same TLS buffer.
        scratch.with(|buf| assert_eq!(buf.len(), 128)).unwrap();
    }

    // ── NUMA ───────────────────────────────────────────────────────

    #[test]
    fn test_numa_aware_alloc_default() {
        let ptr = numa_aware_alloc(256, 64, NumaPolicy::Default).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        unsafe { numa_aware_dealloc(ptr, 256, 64) };
    }

    #[test]
    fn test_numa_aware_alloc_prefer_node() {
        let ptr = numa_aware_alloc(512, 64, NumaPolicy::PreferNode(0)).unwrap();
        assert!(!ptr.as_ptr().is_null());
        unsafe { numa_aware_dealloc(ptr, 512, 64) };
    }

    #[test]
    fn test_numa_aware_alloc_interleave() {
        let ptr = numa_aware_alloc(1024, 64, NumaPolicy::Interleave).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        unsafe { numa_aware_dealloc(ptr, 1024, 64) };
    }

    #[test]
    fn test_numa_policy_eq() {
        assert_eq!(NumaPolicy::Default, NumaPolicy::Default);
        assert_eq!(NumaPolicy::PreferNode(1), NumaPolicy::PreferNode(1));
        assert_ne!(NumaPolicy::Default, NumaPolicy::Interleave);
    }

    // ── Edge cases / integration ───────────────────────────────────

    #[test]
    fn test_alloc_aligned_large() {
        let ptr = alloc_aligned(1 << 20, 64).unwrap(); // 1 MiB
        assert_eq!(ptr.as_ptr() as usize % 64, 0);
        unsafe { dealloc_aligned(ptr, 1 << 20, 64) };
    }

    #[test]
    fn test_tensor_buffer_large() {
        let buf = TensorBuffer::with_default_alignment(1 << 16).unwrap();
        assert_eq!(buf.num_elements(), 1 << 16);
    }

    #[test]
    fn test_arena_fill_exactly() {
        // Create arena of exactly 64 bytes, allocate all of it.
        let arena = ArenaAllocator::new(64, 64).unwrap();
        let _p = arena.alloc(64, 1).unwrap();
        assert!(arena.remaining() == 0 || arena.alloc(1, 1).is_err());
    }

    #[test]
    fn test_pool_stress() {
        let mut pool = PoolAllocator::new(64, 32).unwrap();
        let mut ptrs = Vec::new();
        for _ in 0..100 {
            ptrs.push(pool.acquire().unwrap());
        }
        for p in ptrs {
            unsafe { pool.release(p) };
        }
        assert_eq!(pool.total_allocated(), 100);
        assert_eq!(pool.live_count(), 0);
        assert_eq!(pool.free_count(), 100);
    }

    #[test]
    fn test_buffer_pool_stress() {
        let mut pool = BufferPool::with_default_alignment();
        let mut bufs = Vec::new();
        for i in 1..=20 {
            bufs.push(pool.acquire(i * 100).unwrap());
        }
        for b in bufs {
            pool.release(b);
        }
        assert_eq!(pool.total_created(), 20);
    }

    #[test]
    fn test_memory_reuse_plan_many_tensors() {
        let tensors: Vec<TensorLifetime> = (0..16)
            .map(|i| TensorLifetime {
                id: i,
                size: 256,
                alignment: 64,
                first_use: i * 2,
                last_use: i * 2 + 1,
            })
            .collect();
        let plan = memory_reuse_plan(&tensors);
        // All sequential, so 15 reuses.
        assert_eq!(plan.reuses, 15);
        assert_eq!(plan.total_bytes, 256);
    }

    #[test]
    fn test_memory_reuse_plan_mixed_sizes() {
        let tensors = vec![
            TensorLifetime { id: 0, size: 1024, alignment: 64, first_use: 0, last_use: 1 },
            TensorLifetime { id: 1, size: 512, alignment: 64, first_use: 2, last_use: 3 },
            TensorLifetime { id: 2, size: 2048, alignment: 64, first_use: 0, last_use: 3 },
        ];
        let plan = memory_reuse_plan(&tensors);
        // Tensor 0 and 1 overlap with 2. Tensor 1 can reuse tensor 0's slot.
        assert_eq!(plan.assignments.len(), 3);
        assert!(plan.total_bytes >= 2048);
    }

    #[test]
    fn test_inference_arena_multiple_alloc_types() {
        let arena = InferenceArena::new(16384).unwrap();
        let _f = arena.alloc_f32(128).unwrap();
        let _u = arena.alloc_u8(512).unwrap();
        let _f2 = arena.alloc_f32(64).unwrap();
        assert!(arena.used() > 0);
    }

    #[test]
    fn test_alloc_aligned_min_alignment() {
        let ptr = alloc_aligned(16, 1).unwrap();
        unsafe { dealloc_aligned(ptr, 16, 1) };
    }

    #[test]
    fn test_pool_acquire_after_preallocate() {
        let mut pool = PoolAllocator::new(128, 64).unwrap();
        pool.preallocate(3).unwrap();
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        let c = pool.acquire().unwrap();
        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.total_allocated(), 3);
        // Fourth acquire should grow the pool.
        let d = pool.acquire().unwrap();
        assert_eq!(pool.total_allocated(), 4);
        unsafe {
            pool.release(a);
            pool.release(b);
            pool.release(c);
            pool.release(d);
        }
    }

    #[test]
    fn test_tensor_buffer_ptr_alignment_avx2() {
        let buf = TensorBuffer::new(128, ALIGN_AVX2).unwrap();
        assert_eq!(buf.as_ptr() as usize % ALIGN_AVX2, 0);
    }

    #[test]
    fn test_arena_alloc_slice_u8() {
        let arena = ArenaAllocator::with_capacity(4096).unwrap();
        let ptr = arena.alloc_slice::<u8>(512).unwrap();
        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_constants() {
        assert_eq!(ALIGN_AVX2, 32);
        assert_eq!(ALIGN_AVX512, 64);
        assert_eq!(DEFAULT_ALIGNMENT, 64);
        assert_eq!(MAX_ALIGNMENT, 4096);
    }
}
