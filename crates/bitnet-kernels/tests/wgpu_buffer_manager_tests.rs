//! Comprehensive tests for wgpu-style GPU buffer pool management and Metal alignment.
//!
//! These tests verify the **logic** of buffer management utilities needed for
//! Metal/Apple Silicon (and any wgpu backend). No actual GPU device is required;
//! all structures use plain Rust types so the tests run on any platform with
//! `--features cpu`.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Tiny bitflags helper — no external crate needed.
macro_rules! bitflags_manual {
    ($repr:ty => $name:ident { $($variant:ident = $val:expr),* $(,)? }) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        struct $name($repr);

        #[allow(dead_code)]
        impl $name {
            $(const $variant: Self = Self($val);)*

            fn empty() -> Self { Self(0) }
            fn contains(self, other: Self) -> bool { (self.0 & other.0) == other.0 }
            fn union(self, other: Self) -> Self { Self(self.0 | other.0) }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut first = true;
                $(
                    if self.contains(Self::$variant) {
                        if !first { write!(f, " | ")?; }
                        write!(f, stringify!($variant))?;
                        first = false;
                    }
                )*
                if first { write!(f, "(empty)")?; }
                Ok(())
            }
        }
    };
}

/// Metal requires 256-byte alignment for buffer offsets.
const METAL_ALIGNMENT: u64 = 256;

/// Default memory budget for tests (64 MiB).
const DEFAULT_BUDGET: u64 = 64 * 1024 * 1024;

// Mirrors `wgpu::BufferUsages` without pulling in the real crate.
bitflags_manual! {
    u32 => BufferUsages {
        MAP_READ  = 0x01,
        MAP_WRITE = 0x02,
        COPY_SRC  = 0x04,
        COPY_DST  = 0x08,
        STORAGE   = 0x20,
        UNIFORM   = 0x40,
    }
}

/// Unique ID for every logical buffer.
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
struct GpuBuffer {
    id: u64,
    /// Requested size in bytes.
    size: u64,
    /// Actual backing size (rounded up to alignment).
    backing_size: u64,
    usage: BufferUsages,
    label: Option<String>,
    mapped: bool,
}

impl GpuBuffer {
    fn new(size: u64, usage: BufferUsages, label: Option<&str>) -> Self {
        let backing_size = align_up(size, METAL_ALIGNMENT);
        Self {
            id: next_id(),
            size,
            backing_size,
            usage,
            label: label.map(String::from),
            mapped: false,
        }
    }
}

/// Round `value` up to the next multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> u64 {
    assert!(alignment.is_power_of_two(), "alignment must be power-of-two");
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Size-class bucket helpers
// ---------------------------------------------------------------------------

/// Returns the size-class bucket for a given size.
/// Buckets: 256, 512, 1 KiB, 2 KiB, 4 KiB, …, 256 MiB, etc.
fn size_class(size: u64) -> u64 {
    let min = METAL_ALIGNMENT;
    if size <= min {
        return min;
    }
    min.max(size.next_power_of_two())
}

// ---------------------------------------------------------------------------
// Buffer pool
// ---------------------------------------------------------------------------

struct BufferPool {
    /// Free-lists keyed by (size_class, usage).
    free: HashMap<(u64, u32), Vec<GpuBuffer>>,
    /// Currently live (handed-out) buffers by id.
    live: HashMap<u64, GpuBuffer>,
    /// Total bytes currently allocated (backing size).
    total_allocated: u64,
    /// Memory budget in bytes; 0 = unlimited.
    budget: u64,
}

impl BufferPool {
    fn new(budget: u64) -> Self {
        Self { free: HashMap::new(), live: HashMap::new(), total_allocated: 0, budget }
    }

    /// Allocate (or reuse) a buffer of at least `size` bytes.
    fn alloc(&mut self, size: u64, usage: BufferUsages, label: Option<&str>) -> Option<GpuBuffer> {
        let sc = size_class(size);
        let key = (sc, usage.0);

        // Try reuse from free-list.
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(mut buf) = list.pop() {
                buf.size = size;
                buf.label = label.map(String::from);
                buf.mapped = false;
                self.live.insert(buf.id, buf.clone());
                return Some(buf);
            }
        }

        // Budget check.
        if self.budget > 0 && self.total_allocated + sc > self.budget {
            return None;
        }

        let buf = GpuBuffer {
            id: next_id(),
            size,
            backing_size: sc,
            usage,
            label: label.map(String::from),
            mapped: false,
        };
        self.total_allocated += buf.backing_size;
        self.live.insert(buf.id, buf.clone());
        Some(buf)
    }

    /// Return a buffer to the pool for later reuse.
    fn dealloc(&mut self, id: u64) -> bool {
        if let Some(mut buf) = self.live.remove(&id) {
            buf.mapped = false;
            let key = (buf.backing_size, buf.usage.0);
            self.free.entry(key).or_default().push(buf);
            true
        } else {
            false
        }
    }

    /// Number of buffers on the free-list.
    fn free_count(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Number of live (in-use) buffers.
    fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Purge all free-list entries, reclaiming their memory.
    fn trim(&mut self) {
        for (_, list) in self.free.drain() {
            for buf in &list {
                self.total_allocated -= buf.backing_size;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ring buffer (circular staging)
// ---------------------------------------------------------------------------

struct RingBuffer {
    capacity: u64,
    head: u64,
    tail: u64,
    /// Number of bytes currently occupied.
    used: u64,
}

impl RingBuffer {
    fn new(capacity: u64) -> Self {
        let capacity = align_up(capacity, METAL_ALIGNMENT);
        Self { capacity, head: 0, tail: 0, used: 0 }
    }

    /// Try to allocate `size` bytes, returning the offset if successful.
    fn push(&mut self, size: u64) -> Option<u64> {
        let aligned = align_up(size, METAL_ALIGNMENT);
        if self.used + aligned > self.capacity {
            return None;
        }
        let offset = self.head;
        self.head = (self.head + aligned) % self.capacity;
        self.used += aligned;
        Some(offset)
    }

    /// Free `size` bytes from the tail (FIFO order).
    fn pop(&mut self, size: u64) -> bool {
        let aligned = align_up(size, METAL_ALIGNMENT);
        if aligned > self.used {
            return false;
        }
        self.tail = (self.tail + aligned) % self.capacity;
        self.used -= aligned;
        true
    }

    fn available(&self) -> u64 {
        self.capacity - self.used
    }

    fn is_empty(&self) -> bool {
        self.used == 0
    }
}

// ---------------------------------------------------------------------------
// Double (ping-pong) buffer
// ---------------------------------------------------------------------------

struct DoubleBuffer {
    buffers: [GpuBuffer; 2],
    /// Index of the buffer the CPU writes to.
    write_idx: usize,
}

impl DoubleBuffer {
    fn new(size: u64, usage: BufferUsages) -> Self {
        Self {
            buffers: [
                GpuBuffer::new(size, usage, Some("ping")),
                GpuBuffer::new(size, usage, Some("pong")),
            ],
            write_idx: 0,
        }
    }

    fn write_buffer(&self) -> &GpuBuffer {
        &self.buffers[self.write_idx]
    }

    fn read_buffer(&self) -> &GpuBuffer {
        &self.buffers[1 - self.write_idx]
    }

    fn swap(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

// ===================================================================
// Tests
// ===================================================================

// ---- align_up -----------------------------------------------------------

#[test]
fn test_align_up_zero() {
    assert_eq!(align_up(0, METAL_ALIGNMENT), 0);
}

#[test]
fn test_align_up_exact() {
    assert_eq!(align_up(256, METAL_ALIGNMENT), 256);
    assert_eq!(align_up(512, METAL_ALIGNMENT), 512);
}

#[test]
fn test_align_up_rounds() {
    assert_eq!(align_up(1, METAL_ALIGNMENT), 256);
    assert_eq!(align_up(255, METAL_ALIGNMENT), 256);
    assert_eq!(align_up(257, METAL_ALIGNMENT), 512);
}

#[test]
fn test_align_up_large() {
    let val = 1_000_000u64;
    let aligned = align_up(val, METAL_ALIGNMENT);
    assert!(aligned >= val);
    assert_eq!(aligned % METAL_ALIGNMENT, 0);
}

// ---- size_class ---------------------------------------------------------

#[test]
fn test_size_class_minimum() {
    assert_eq!(size_class(0), METAL_ALIGNMENT);
    assert_eq!(size_class(1), METAL_ALIGNMENT);
    assert_eq!(size_class(256), METAL_ALIGNMENT);
}

#[test]
fn test_size_class_power_of_two() {
    assert_eq!(size_class(512), 512);
    assert_eq!(size_class(1024), 1024);
    assert_eq!(size_class(4096), 4096);
}

#[test]
fn test_size_class_rounds_up() {
    assert_eq!(size_class(257), 512);
    assert_eq!(size_class(513), 1024);
    assert_eq!(size_class(3000), 4096);
}

// ---- GpuBuffer ----------------------------------------------------------

#[test]
fn test_buffer_creation_basic() {
    let buf = GpuBuffer::new(100, BufferUsages::STORAGE, Some("weights"));
    assert_eq!(buf.size, 100);
    assert_eq!(buf.backing_size, 256); // aligned up
    assert_eq!(buf.label.as_deref(), Some("weights"));
    assert!(!buf.mapped);
}

#[test]
fn test_buffer_backing_alignment() {
    for size in [1, 128, 255, 256, 257, 511, 512, 1000, 4096] {
        let buf = GpuBuffer::new(size, BufferUsages::STORAGE, None);
        assert!(buf.backing_size >= size);
        assert_eq!(buf.backing_size % METAL_ALIGNMENT, 0, "size {size} not aligned");
    }
}

#[test]
fn test_buffer_unique_ids() {
    let a = GpuBuffer::new(256, BufferUsages::STORAGE, None);
    let b = GpuBuffer::new(256, BufferUsages::STORAGE, None);
    assert_ne!(a.id, b.id);
}

// ---- BufferUsages -------------------------------------------------------

#[test]
fn test_usage_contains() {
    let u = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    assert!(u.contains(BufferUsages::STORAGE));
    assert!(u.contains(BufferUsages::COPY_DST));
    assert!(!u.contains(BufferUsages::MAP_READ));
}

#[test]
fn test_usage_union() {
    let a = BufferUsages::MAP_READ;
    let b = BufferUsages::COPY_SRC;
    let c = a.union(b);
    assert!(c.contains(a));
    assert!(c.contains(b));
}

#[test]
fn test_usage_empty() {
    let e = BufferUsages::empty();
    assert!(!e.contains(BufferUsages::STORAGE));
}

#[test]
fn test_usage_all_flags() {
    let all = BufferUsages::MAP_READ
        | BufferUsages::MAP_WRITE
        | BufferUsages::COPY_SRC
        | BufferUsages::COPY_DST
        | BufferUsages::STORAGE
        | BufferUsages::UNIFORM;
    assert!(all.contains(BufferUsages::MAP_READ));
    assert!(all.contains(BufferUsages::UNIFORM));
}

#[test]
fn test_storage_copy_dst_combo() {
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    assert!(usage.contains(BufferUsages::STORAGE));
    assert!(usage.contains(BufferUsages::COPY_DST));
    assert!(!usage.contains(BufferUsages::MAP_READ));
}

#[test]
fn test_map_read_copy_dst_staging() {
    let staging = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
    assert!(staging.contains(BufferUsages::MAP_READ));
    assert!(staging.contains(BufferUsages::COPY_DST));
    assert!(!staging.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_map_write_copy_src_upload() {
    let upload = BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC;
    assert!(upload.contains(BufferUsages::MAP_WRITE));
    assert!(upload.contains(BufferUsages::COPY_SRC));
}

#[test]
fn test_uniform_copy_dst_combo() {
    let ubo = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
    assert!(ubo.contains(BufferUsages::UNIFORM));
    assert!(ubo.contains(BufferUsages::COPY_DST));
    assert!(!ubo.contains(BufferUsages::STORAGE));
}

// ---- BufferPool allocation / deallocation -------------------------------

#[test]
fn test_pool_alloc_basic() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf = pool.alloc(1024, BufferUsages::STORAGE, Some("test")).unwrap();
    assert!(buf.backing_size >= 1024);
    assert_eq!(pool.live_count(), 1);
}

#[test]
fn test_pool_dealloc_moves_to_free() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf = pool.alloc(1024, BufferUsages::STORAGE, None).unwrap();
    let id = buf.id;
    assert!(pool.dealloc(id));
    assert_eq!(pool.live_count(), 0);
    assert_eq!(pool.free_count(), 1);
}

#[test]
fn test_pool_dealloc_nonexistent() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    assert!(!pool.dealloc(999_999));
}

#[test]
fn test_pool_reuse_same_size_class() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf1 = pool.alloc(500, BufferUsages::STORAGE, None).unwrap();
    let id1 = buf1.id;
    pool.dealloc(id1);

    // Second alloc in same size class should recycle.
    let buf2 = pool.alloc(400, BufferUsages::STORAGE, None).unwrap();
    assert_eq!(buf2.id, id1, "buffer should be reused");
    assert_eq!(buf2.size, 400);
}

#[test]
fn test_pool_no_reuse_different_usage() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf1 = pool.alloc(256, BufferUsages::STORAGE, None).unwrap();
    let id1 = buf1.id;
    pool.dealloc(id1);

    let buf2 = pool.alloc(256, BufferUsages::UNIFORM, None).unwrap();
    assert_ne!(buf2.id, id1, "different usage should not reuse");
}

#[test]
fn test_pool_no_reuse_different_size_class() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf1 = pool.alloc(256, BufferUsages::STORAGE, None).unwrap();
    pool.dealloc(buf1.id);

    let buf2 = pool.alloc(1024, BufferUsages::STORAGE, None).unwrap();
    assert_ne!(buf2.id, buf1.id, "different size class should not reuse");
}

#[test]
fn test_pool_multiple_allocs() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let mut ids = Vec::new();
    for i in 0..10 {
        let buf = pool.alloc(256 * (i + 1), BufferUsages::STORAGE, None).unwrap();
        ids.push(buf.id);
    }
    assert_eq!(pool.live_count(), 10);
    for id in &ids {
        assert!(pool.dealloc(*id));
    }
    assert_eq!(pool.live_count(), 0);
    assert!(pool.free_count() > 0);
}

#[test]
fn test_pool_trim_reclaims_memory() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf = pool.alloc(4096, BufferUsages::STORAGE, None).unwrap();
    let alloc_before = pool.total_allocated;
    pool.dealloc(buf.id);
    assert_eq!(pool.total_allocated, alloc_before); // still counted

    pool.trim();
    assert_eq!(pool.free_count(), 0);
    assert!(pool.total_allocated < alloc_before);
}

// ---- Metal 256-byte alignment -------------------------------------------

#[test]
fn test_metal_alignment_constant() {
    assert_eq!(METAL_ALIGNMENT, 256);
    assert!(METAL_ALIGNMENT.is_power_of_two());
}

#[test]
fn test_pool_backing_always_aligned() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    for size in [1, 7, 33, 100, 255, 256, 257, 1000, 65537] {
        let buf = pool.alloc(size, BufferUsages::STORAGE, None).unwrap();
        assert_eq!(
            buf.backing_size % METAL_ALIGNMENT,
            0,
            "backing_size {} not 256-aligned for request {}",
            buf.backing_size,
            size
        );
    }
}

// ---- Buffer mapping lifecycle -------------------------------------------

#[test]
fn test_buffer_map_write() {
    let mut buf = GpuBuffer::new(512, BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC, None);
    assert!(!buf.mapped);
    buf.mapped = true;
    assert!(buf.mapped);
    buf.mapped = false;
    assert!(!buf.mapped);
}

#[test]
fn test_buffer_map_read() {
    let mut buf = GpuBuffer::new(512, BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
    buf.mapped = true;
    assert!(buf.mapped);
}

#[test]
fn test_mapped_buffer_returned_unmapped() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let mut buf = pool.alloc(256, BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC, None).unwrap();
    buf.mapped = true;
    // Simulate returning to pool — pool.dealloc resets mapped.
    pool.live.get_mut(&buf.id).unwrap().mapped = true;
    pool.dealloc(buf.id);

    // Re-acquire
    let reused = pool.alloc(256, BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC, None).unwrap();
    assert!(!reused.mapped, "recycled buffer must be unmapped");
}

// ---- Ring buffer --------------------------------------------------------

#[test]
fn test_ring_buffer_capacity_aligned() {
    let ring = RingBuffer::new(1000);
    assert_eq!(ring.capacity % METAL_ALIGNMENT, 0);
    assert!(ring.capacity >= 1000);
}

#[test]
fn test_ring_buffer_push_pop() {
    let mut ring = RingBuffer::new(1024);
    let off = ring.push(256).unwrap();
    assert_eq!(off, 0);
    assert_eq!(ring.available(), ring.capacity - 256);

    assert!(ring.pop(256));
    assert!(ring.is_empty());
}

#[test]
fn test_ring_buffer_wrap_around() {
    let mut ring = RingBuffer::new(1024);
    // Fill most of the ring.
    ring.push(512).unwrap();
    ring.push(256).unwrap();
    ring.pop(512);
    // Next push should wrap.
    let off = ring.push(512).unwrap();
    assert!(off < ring.capacity);
}

#[test]
fn test_ring_buffer_full() {
    let mut ring = RingBuffer::new(512);
    ring.push(256).unwrap();
    ring.push(256).unwrap();
    assert!(ring.push(256).is_none(), "ring should be full");
}

#[test]
fn test_ring_buffer_empty_after_drain() {
    let mut ring = RingBuffer::new(1024);
    ring.push(256).unwrap();
    ring.push(256).unwrap();
    ring.pop(256);
    ring.pop(256);
    assert!(ring.is_empty());
    assert_eq!(ring.available(), ring.capacity);
}

#[test]
fn test_ring_buffer_alignment_of_offsets() {
    let mut ring = RingBuffer::new(4096);
    for _ in 0..8 {
        let off = ring.push(100).unwrap(); // 100 → 256 aligned
        assert_eq!(off % METAL_ALIGNMENT, 0, "offset {off} not aligned");
    }
}

// ---- Double buffer (ping-pong) ------------------------------------------

#[test]
fn test_double_buffer_distinct_ids() {
    let db = DoubleBuffer::new(1024, BufferUsages::STORAGE);
    assert_ne!(db.buffers[0].id, db.buffers[1].id);
}

#[test]
fn test_double_buffer_swap() {
    let mut db = DoubleBuffer::new(1024, BufferUsages::STORAGE);
    let w0 = db.write_buffer().id;
    let r0 = db.read_buffer().id;
    assert_ne!(w0, r0);

    db.swap();
    assert_eq!(db.write_buffer().id, r0);
    assert_eq!(db.read_buffer().id, w0);
}

#[test]
fn test_double_buffer_labels() {
    let db = DoubleBuffer::new(512, BufferUsages::STORAGE);
    assert_eq!(db.buffers[0].label.as_deref(), Some("ping"));
    assert_eq!(db.buffers[1].label.as_deref(), Some("pong"));
}

#[test]
fn test_double_buffer_backing_aligned() {
    let db = DoubleBuffer::new(300, BufferUsages::STORAGE);
    for buf in &db.buffers {
        assert_eq!(buf.backing_size % METAL_ALIGNMENT, 0);
    }
}

// ---- Memory pressure / budget -------------------------------------------

#[test]
fn test_pool_budget_exceeded() {
    let mut pool = BufferPool::new(1024);
    let _b1 = pool.alloc(512, BufferUsages::STORAGE, None).unwrap();
    let _b2 = pool.alloc(512, BufferUsages::STORAGE, None).unwrap();
    let result = pool.alloc(512, BufferUsages::STORAGE, None);
    assert!(result.is_none(), "should fail when over budget");
}

#[test]
fn test_pool_budget_allows_after_free() {
    let mut pool = BufferPool::new(1024);
    let b1 = pool.alloc(512, BufferUsages::STORAGE, None).unwrap();
    let _b2 = pool.alloc(512, BufferUsages::STORAGE, None).unwrap();

    pool.dealloc(b1.id);
    // Free-list buffer reuse does not add to total_allocated.
    let result = pool.alloc(512, BufferUsages::STORAGE, None);
    assert!(result.is_some(), "should succeed by reusing freed buffer");
}

#[test]
fn test_pool_budget_trim_frees_memory() {
    let mut pool = BufferPool::new(2048);
    let b1 = pool.alloc(1024, BufferUsages::STORAGE, None).unwrap();
    let _b2 = pool.alloc(1024, BufferUsages::STORAGE, None).unwrap();
    pool.dealloc(b1.id);

    // Trim reclaims free-list memory from the budget.
    pool.trim();

    let result = pool.alloc(1024, BufferUsages::STORAGE, None);
    assert!(result.is_some(), "should succeed after trim");
}

#[test]
fn test_pool_unlimited_budget() {
    let mut pool = BufferPool::new(0); // 0 = unlimited
    for _ in 0..100 {
        assert!(pool.alloc(1 << 20, BufferUsages::STORAGE, None).is_some());
    }
}

// ---- Buffer labels (Metal GPU Capture) ----------------------------------

#[test]
fn test_buffer_label_set() {
    let buf = GpuBuffer::new(256, BufferUsages::STORAGE, Some("layernorm_weights"));
    assert_eq!(buf.label.as_deref(), Some("layernorm_weights"));
}

#[test]
fn test_buffer_label_none() {
    let buf = GpuBuffer::new(256, BufferUsages::STORAGE, None);
    assert!(buf.label.is_none());
}

#[test]
fn test_pool_alloc_preserves_label() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf = pool.alloc(256, BufferUsages::STORAGE, Some("attention_qkv")).unwrap();
    assert_eq!(buf.label.as_deref(), Some("attention_qkv"));
}

#[test]
fn test_pool_reuse_updates_label() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let buf1 = pool.alloc(256, BufferUsages::STORAGE, Some("old_label")).unwrap();
    pool.dealloc(buf1.id);

    let buf2 = pool.alloc(256, BufferUsages::STORAGE, Some("new_label")).unwrap();
    assert_eq!(buf2.label.as_deref(), Some("new_label"));
}

// ---- Size alignment edge cases ------------------------------------------

#[test]
fn test_align_up_one_byte() {
    assert_eq!(align_up(1, METAL_ALIGNMENT), 256);
}

#[test]
fn test_align_up_power_of_two_alignments() {
    for &align in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let result = align_up(100, align);
        assert_eq!(result % align, 0);
        assert!(result >= 100);
    }
}

#[test]
fn test_size_class_large_value() {
    let sc = size_class(1_000_000);
    assert!(sc >= 1_000_000);
    assert!(sc.is_power_of_two());
}

// ---- Mixed workload integration -----------------------------------------

#[test]
fn test_mixed_alloc_dealloc_cycle() {
    let mut pool = BufferPool::new(DEFAULT_BUDGET);
    let mut ids = VecDeque::new();

    // Alloc 20 buffers with varying sizes and usages.
    for i in 0..20u64 {
        let usage = if i % 2 == 0 {
            BufferUsages::STORAGE
        } else {
            BufferUsages::UNIFORM | BufferUsages::COPY_DST
        };
        let buf = pool.alloc(256 * (i + 1), usage, None).unwrap();
        ids.push_back(buf.id);
    }
    assert_eq!(pool.live_count(), 20);

    // Free the first 10.
    for _ in 0..10 {
        let id = ids.pop_front().unwrap();
        pool.dealloc(id);
    }
    assert_eq!(pool.live_count(), 10);
    assert!(pool.free_count() >= 10);

    // Allocate 5 more — some should be recycled.
    for i in 0..5u64 {
        pool.alloc(256 * (i + 1), BufferUsages::STORAGE, None).unwrap();
    }
    assert_eq!(pool.live_count(), 15);
}

#[test]
fn test_debug_format_usage_flags() {
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let dbg = format!("{usage:?}");
    assert!(dbg.contains("COPY_DST"));
    assert!(dbg.contains("STORAGE"));
}
